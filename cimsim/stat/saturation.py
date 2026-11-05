"""Exact sparse ADC saturation correction for mode 'statistical' (A3, findings F8).

With a calibrated ADC range (full scale FS below the nominal nrow * L_c), MAC values m > FS
saturate at the top ADC code. This error is deterministic and sparse, and the conversions that
saturate are strongly correlated across input planes (two's-complement sign extension: planes
3-7 of a small negative input are all set together, and their shift-add weights largely cancel).
The analytic model M3 treats conversions as independent and overestimates it 2-26x (F8).

Statistical mode therefore splits every conversion's readout error into

    readout(m) - m = [readout(m) - Q_u(min(m, FS_nom))]  +  [Q_u(min(m, FS_nom)) - m]
                       saturation term S(m), here           rounding term, M3 (cimsim/stat/analytic.py)

with readout() the noise-free ADC of the executors and Q_u the same ADC with no upper code limit
(rounding only). M3 keeps the clamped noise moments and subtracts S(m) from its deterministic part. S(m) is non-zero only for m above a threshold m*_cp. A conversion's MAC is bounded
by the input digit sum of the chunk (sum_r d_c[r]) and by L_c times the weight-plane popcount of the
column, so the MAC is computed exactly only for (token, chunk, cycle) rows and (plane, column) pairs
whose bounds reach m*_cp. With CIMConfig.stat_sat_margin set (default), columns are further limited to
those whose largest calibration MAC came within the margin of m*_cp: the popcount bound alone flags
about half of all conversions when plane densities are near 0.5, although almost none saturate. The result is exact for noise-free conversions whenever the rounding term is
exact too (ADC step 1/n of a MAC unit); with analog noise the mean shift of noisy values near FS is
the M3 residual.
"""

from __future__ import annotations

import torch

from cimsim.config import CIMConfig
from cimsim.core.bitplanes import input_cycles
from cimsim.core.executor import uniform_chunks


_BLOCK_ELEMS = 1 << 22          # elements per [rows, columns] block in saturation_correction (~32 MB int64)


def _codes(q: torch.Tensor, bits: int, signed: bool) -> torch.Tensor:
    qi = q.round().to(torch.int64)
    if signed:
        qi = torch.where(qi < 0, qi + (1 << bits), qi)
    return qi


def saturation_tables(cfg: CIMConfig, full_scale: torch.Tensor | None, max_mac: int) -> torch.Tensor:
    """S(m) for m = 0..max_mac per (cycle, plane): float64 [C, P, max_mac + 1]."""
    cycles = input_cycles(None, cfg.x_bits, cfg.signed_act, cfg.act_enc)
    C, P = len(cycles), cfg.w_bits
    m = torch.arange(max_mac + 1, dtype=torch.float64)
    out = torch.zeros(C, P, max_mac + 1, dtype=torch.float64)
    L = None if cfg.adc_bits is None else float(2 ** cfg.adc_bits - 1)
    for ci, c in enumerate(cycles):
        fsn = float(cfg.nrow * c.levels)
        for pi in range(P):
            fs = fsn if full_scale is None else float(full_scale.reshape(C, P)[ci, pi])
            # executor readout (noise-free): array clamp at FS_nom, ADC range clamp at FS, rounding
            v = torch.clamp(torch.clamp(m, 0.0, fsn) / fs, 0.0, 1.0)
            if L is not None:
                v = torch.round(v * L) / L
            readout = v * fs
            mu = torch.clamp(m, 0.0, fsn)
            q_u = mu if L is None else torch.round(mu / fs * L) / L * fs
            out[ci, pi] = readout - q_u
    return out


def deterministic_tables(cfg: CIMConfig, full_scale: torch.Tensor | None, max_mac: int,
                         hist: torch.Tensor | None = None, mass_max: float = 0.05):
    """Exact sparse deterministic readout term T(m) per (cycle, plane) (F8, F9): float64 [C, P, max_mac + 1],
    plus the linear gain g0 [C, P] it is taken relative to and a mode flag [C, P] (1 = linear, 0 = saturation only).

    The noise-free readout is exactly (1 + g0)·m on a region m < m*: g0 = 0 for a calibrated ADC with a
    step of 1/n MAC units, g0 = round(1/Δ)·Δ − 1 for a full-range ADC (exact for m ≤ ~1/(2|g0|)). Above
    m* the readout is nonlinear (rounding drift, saturation). Linear mode: T(m) = readout(m) − (1 + g0)·m,
    zero below m*. If the readout is not linear on the bulk of the calibration MACs (calibration mass at
    m ≥ m* above mass_max, e.g. a full-range ADC whose step is not 1/n of a MAC unit), only saturation is
    taken exactly: T(m) = saturation_tables(), and M3 keeps the rounding error (independent conversions)."""
    sat = saturation_tables(cfg, full_scale, max_mac)
    cycles = input_cycles(None, cfg.x_bits, cfg.signed_act, cfg.act_enc)
    C, P = len(cycles), cfg.w_bits
    m = torch.arange(max_mac + 1, dtype=torch.float64)
    L = None if cfg.adc_bits is None else float(2 ** cfg.adc_bits - 1)
    out = sat.clone()
    g0 = torch.zeros(C, P, dtype=torch.float64)
    mode = torch.zeros(C, P, dtype=torch.int64)
    for ci, c in enumerate(cycles):
        fsn = float(cfg.nrow * c.levels)
        for pi in range(P):
            fs = fsn if full_scale is None else float(full_scale.reshape(C, P)[ci, pi])
            v = torch.clamp(torch.clamp(m, 0.0, fsn) / fs, 0.0, 1.0)
            if L is not None:
                v = torch.round(v * L) / L
            readout = v * fs
            g = float(readout[1] - 1.0) if max_mac >= 1 else 0.0
            if abs(g) >= 0.25:                                      # ADC step > ~1.5 MAC units: not near-linear
                continue
            T = readout - (1.0 + g) * m
            nz = torch.nonzero(T.abs() > 1e-9).flatten()
            if nz.numel() and hist is not None:
                h = hist[ci, pi].double()
                tot = float(h.sum())
                n = min(h.numel(), max_mac + 1)
                above = float(h[int(nz[0]):n].sum()) if int(nz[0]) < n else 0.0
                if tot > 0 and above / tot > mass_max:
                    continue                                        # keep saturation-only table
            out[ci, pi] = T
            g0[ci, pi] = g
            mode[ci, pi] = 1
    return out, g0, mode


def thresholds(tables: torch.Tensor) -> torch.Tensor:
    """Smallest m with S(m) != 0 per (cycle, plane); max_mac + 1 if none. int64 [C, P]."""
    nz = tables.abs() > 1e-9
    M1 = tables.shape[-1]
    idx = torch.arange(M1).view(1, 1, -1).expand_as(nz)
    return torch.where(nz, idx, torch.full_like(idx, M1)).min(dim=-1).values


def saturation_plan(q_w: torch.Tensor, cfg: CIMConfig, tables: torch.Tensor,
                    chunks: list[tuple[int, int]] | None, K: int,
                    col_mask: torch.Tensor | None = None) -> dict | None:
    """Weight-dependent part of saturation_correction: per (chunk, cycle), the (plane, column) pairs whose
    bound L_c x weight-plane popcount (and col_mask) reaches the threshold, and the smallest threshold among
    them. Depends only on the weights, tables and col_mask, so CIMLinear computes it once and reuses it
    for every call (for short sequences this part dominated the cost). None if nothing can saturate."""
    cycles = input_cycles(None, cfg.x_bits, cfg.signed_act, cfg.act_enc)
    C, P = len(cycles), cfg.w_bits
    M1 = tables.shape[-1]
    thr = thresholds(tables)                                        # [C, P]
    if bool((thr >= M1).all()) or (col_mask is not None and not bool(col_mask.any())):
        return None
    coef = torch.empty(C, P, dtype=torch.float64)
    for ci, c in enumerate(cycles):
        for pi in range(P):
            wsign = -1 if (cfg.w_signed and pi == P - 1) else 1
            coef[ci, pi] = c.sign * wsign * 2.0 ** (c.shift + pi)
    flat_tab = (tables * coef.view(C, P, 1)).reshape(C, P * M1)     # coef folded in
    chunk_list = uniform_chunks(K, cfg.nrow) if chunks is None else chunks
    uw = _codes(q_w.cpu(), P, cfg.w_signed)                          # [N, K]
    cm = None if col_mask is None else col_mask.cpu()
    pshift = torch.arange(P).view(P, 1, 1)
    entries = []
    for j, (k0, k1) in enumerate(chunk_list):
        wbits = ((uw[:, k0:k1].unsqueeze(0) >> pshift) & 1)          # [P, N, r] int64
        wpop = wbits.sum(dim=-1)                                     # [P, N]
        for ci, c in enumerate(cycles):
            t_cp = thr[ci]                                           # [P]
            col_ok = (wpop * c.levels) >= t_cp.view(P, 1)            # [P, N]
            if cm is not None:
                col_ok = col_ok & cm[ci]
            if not bool(col_ok.any()):
                continue
            t_min = int(t_cp[col_ok.any(dim=1)].min())
            p_idx, n_idx = torch.nonzero(col_ok, as_tuple=True)
            entries.append((j, ci, t_min, p_idx, n_idx))
    if not entries:
        return None
    return {"entries": entries, "flat_tab": flat_tab, "chunks": chunk_list, "M1": M1}


def saturation_correction(q_x: torch.Tensor, q_w: torch.Tensor, cfg: CIMConfig, tables: torch.Tensor,
                          chunks: list[tuple[int, int]] | None = None,
                          stats: dict | None = None, col_mask: torch.Tensor | None = None,
                          plan: dict | None | bool = False) -> torch.Tensor | None:
    """sum over chunks, cycles, planes of coef_cp * S(m) for every output: float64 [T, N],
    or None if no conversion can saturate. stats (optional) accumulates 'macs_exact' and
    'macs_total' (MACs of the exact sub-matmuls vs a full bit-accurate pass).
    col_mask: optional bool [C, P, N]; only these (cycle, plane, column) conversions are checked
    (set from calibration column maxima, see CIMConfig.stat_sat_margin). None = popcount bound only.
    plan: result of saturation_plan() for these weights (reused across calls); False = build it here."""
    T, K = q_x.shape
    N = q_w.shape[0]
    cycles = input_cycles(None, cfg.x_bits, cfg.signed_act, cfg.act_enc)
    C, P = len(cycles), cfg.w_bits
    if stats is not None:
        stats["macs_total"] = stats.get("macs_total", 0) + T * K * N * C * P
        stats.setdefault("macs_exact", 0)
    if plan is False:
        plan = saturation_plan(q_w, cfg, tables, chunks, K, col_mask)
    if plan is None:
        return None
    M1, flat_tab, chunk_list = plan["M1"], plan["flat_tab"], plan["chunks"]
    ux = _codes(q_x.cpu(), cfg.x_bits, cfg.signed_act)               # [T, K]
    qw_cpu = q_w.cpu()
    out = torch.zeros(T * N, dtype=torch.float64)
    macs = 0
    uwc = {}
    for j, ci, t_min, p_idx, n_idx in plan["entries"]:
        c = cycles[ci]
        k0, k1 = chunk_list[j]
        r = k1 - k0
        dig = (ux[:, k0:k1] >> c.bit_lo) & ((1 << c.n_bits) - 1)       # [T, r]
        rows = torch.nonzero(dig.sum(dim=1) >= t_min).flatten()
        if rows.numel() == 0:
            continue
        if j not in uwc:                                             # weight codes of this chunk, once per call
            uwc[j] = _codes(qw_cpu[:, k0:k1], P, cfg.w_signed)       # [N, r]
        wsel = ((uwc[j][n_idx] >> p_idx.view(-1, 1)) & 1).to(torch.float32).t()  # [r, S]
        macs += rows.numel() * r * p_idx.numel()
        base = (p_idx * M1).view(1, -1)
        rb = max(1, _BLOCK_ELEMS // max(1, p_idx.numel()))          # bound the [rows, S] working set
        for r0 in range(0, rows.numel(), rb):
            rws = rows[r0:r0 + rb]
            m = (dig[rws].to(torch.float32) @ wsel).round().to(torch.int64)   # exact: values <= r * L_c
            corr = flat_tab[ci][base + m.clamp_max(M1 - 1)]                    # [R, S]
            nz = corr != 0
            if not bool(nz.any()):
                continue
            ri, si = torch.nonzero(nz, as_tuple=True)
            out.index_add_(0, rws[ri] * N + n_idx[si], corr[ri, si])
    if stats is not None:
        stats["macs_exact"] = stats.get("macs_exact", 0) + macs
    return out.view(T, N).to(q_x.device)
