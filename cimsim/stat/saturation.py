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

import os

import torch

from cimsim.config import CIMConfig
from cimsim.core.bitplanes import input_cycles
from cimsim.core.executor import uniform_chunks


_BLOCK_ELEMS = 1 << 22          # elements per [rows, columns] block in saturation_correction (~32 MB int64)
BATCH_T_MAX = 256               # calls with at most this many tokens use the batched path (see saturation_correction)
_BATCH_ELEMS = 1 << 22          # elements per [entries, tokens, columns] block in the batched path
_BATCH_ENTRIES = 256            # entries per block in the batched path
_BUCKET_ENTRIES = 1024          # entries per padded bucket in the plan
PLAN_CACHE_MB = float(os.environ.get("CIMSIM_SAT_PLAN_MB", "32"))   # per-layer limit for cached weight bits


def _codes(q: torch.Tensor, bits: int, signed: bool) -> torch.Tensor:
    qi = q.round().to(torch.int64)
    if signed:
        qi = torch.where(qi < 0, qi + (1 << bits), qi)
    return qi


def rounding_exact(cfg: CIMConfig, full_scale: torch.Tensor | None) -> bool:
    """True if statistical mode reproduces the noise-free rounding of a calibrated ADC exactly: every
    (cycle, plane) has an ADC step of 1/n MAC unit (n = 1, 2, ...), so integer MAC values below the full
    scale are read without error and only saturation remains (computed exactly). With any other step the
    rounding error goes to the analytic model, which treats conversions as independent and overestimates
    it (findings F13). Full-range ADCs (full_scale None) use the linear-gain tables checked in GATE A3
    and are not judged here."""
    if cfg.adc_bits is None or full_scale is None:
        return True
    r = float(2 ** cfg.adc_bits - 1) / full_scale.double().flatten()
    return bool(((r - r.round()).abs() < 1e-9).all() and (r.round() >= 1).all())


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
                    col_mask: torch.Tensor | None = None, cache: bool = True) -> dict | None:
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
    wbs = []                                                         # per entry: weight bits [r, S] uint8 (cache only)
    cache_bytes = 0
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
            if cache:
                cache_bytes += int(p_idx.numel()) * (k1 - k0)
                if cache_bytes > PLAN_CACHE_MB * 1e6:
                    cache = False
                    wbs = []
                else:
                    wbs.append(wbits[p_idx, n_idx].t().contiguous().to(torch.uint8))     # [r, S]
    if not entries:
        return None
    plan = {"entries": entries, "flat_tab": flat_tab, "chunks": chunk_list, "M1": M1, "cached": bool(cache),
            "cache_bytes": cache_bytes if cache else 0}
    if cache:
        _bucket_entries(plan, wbs)
    return plan


def _bucket_entries(plan: dict, wbs: list[torch.Tensor]) -> None:
    """Pad the cached weight bits of all entries into buckets of entries with similar column counts
    (padded size at most 2x the real one). plan['entries'][e] gets its location (bucket, row) in plan['loc'][e]."""
    entries, chunk_list = plan["entries"], plan["chunks"]
    r_max = max(k1 - k0 for k0, k1 in chunk_list)
    order = sorted(range(len(entries)), key=lambda e: entries[e][3].numel())
    loc = [None] * len(entries)
    buckets = []
    pos = 0
    while pos < len(order):
        s0 = max(entries[order[pos]][3].numel(), 16)
        members = []
        while pos < len(order) and len(members) < _BUCKET_ENTRIES and entries[order[pos]][3].numel() <= 2 * s0:
            members.append(order[pos])
            pos += 1
        smax = max(entries[e][3].numel() for e in members)
        E = len(members)
        wb = torch.zeros(E, r_max, smax, dtype=torch.uint8)
        pp = torch.zeros(E, smax, dtype=torch.int64)
        nn = torch.zeros(E, smax, dtype=torch.int64)
        for i, e in enumerate(members):
            j, ci, t_min, p_idx, n_idx = entries[e]
            S = p_idx.numel()
            r = chunk_list[j][1] - chunk_list[j][0]
            wb[i, :r, :S] = wbs[e]
            pp[i, :S] = p_idx
            nn[i, :S] = n_idx
            loc[e] = (len(buckets), i, S)
        buckets.append({
            "wb": wb, "p": pp, "n": nn, "smax": smax,
            "ej": torch.tensor([entries[e][0] for e in members], dtype=torch.int64),
            "ec": torch.tensor([entries[e][1] for e in members], dtype=torch.int64),
            "tmin": torch.tensor([entries[e][2] for e in members], dtype=torch.int64),
            "S": torch.tensor([entries[e][3].numel() for e in members], dtype=torch.int64),
            "r": torch.tensor([chunk_list[entries[e][0]][1] - chunk_list[entries[e][0]][0] for e in members],
                              dtype=torch.int64),
        })
    plan["buckets"], plan["loc"], plan["r_max"] = buckets, loc, r_max


def saturation_correction(q_x: torch.Tensor, q_w: torch.Tensor, cfg: CIMConfig, tables: torch.Tensor,
                          chunks: list[tuple[int, int]] | None = None,
                          stats: dict | None = None, col_mask: torch.Tensor | None = None,
                          plan: dict | None | bool = False, batch: bool | None = None) -> torch.Tensor | None:
    """sum over chunks, cycles, planes of coef_cp * S(m) for every output: float64 [T, N],
    or None if no conversion can saturate. stats (optional) accumulates 'macs_exact' and
    'macs_total' (MACs of the exact sub-matmuls vs a full bit-accurate pass).
    col_mask: optional bool [C, P, N]; only these (cycle, plane, column) conversions are checked
    (set from calibration column maxima, see CIMConfig.stat_sat_margin). None = popcount bound only.
    plan: result of saturation_plan() for these weights (reused across calls); False = build it here
    (without the weight-bit cache, for dynamic operands).
    batch: True = compute all entries as a few padded batched matmuls (no row selection; fast for short
    calls, where the per-entry Python loop dominates), False = loop over entries with row selection
    (less work for long calls), None = batched if the plan has cached bits and T <= BATCH_T_MAX.
    Both give the same result up to float64 summation order."""
    T, K = q_x.shape
    N = q_w.shape[0]
    cycles = input_cycles(None, cfg.x_bits, cfg.signed_act, cfg.act_enc)
    C, P = len(cycles), cfg.w_bits
    if stats is not None:
        stats["macs_total"] = stats.get("macs_total", 0) + T * K * N * C * P
        stats.setdefault("macs_exact", 0)
    if plan is False:
        plan = saturation_plan(q_w, cfg, tables, chunks, K, col_mask, cache=False)
    if plan is None:
        return None
    M1, flat_tab, chunk_list = plan["M1"], plan["flat_tab"], plan["chunks"]
    ux = _codes(q_x.cpu(), cfg.x_bits, cfg.signed_act)               # [T, K]
    out = torch.zeros(T * N, dtype=torch.float64)
    if batch is None:
        batch = plan["cached"] and T <= BATCH_T_MAX
    if batch:
        if not plan["cached"]:
            raise ValueError("batched saturation path needs a plan built with cache=True within the size limit")
        macs = _batched(plan, ux, cfg, cycles, T, N, K, out, stats is not None)
        if stats is not None:
            stats["macs_exact"] = stats.get("macs_exact", 0) + macs
        return out.view(T, N).to(q_x.device)
    qw_cpu = q_w.cpu()
    macs = 0
    uwc = {}
    mmin = _mmin(plan)
    digc = {}                     # digits and row sums of the current (chunk, cycle); entries are ordered by them
    for e, (j, ci, t_min, p_idx, n_idx) in enumerate(plan["entries"]):
        c = cycles[ci]
        k0, k1 = chunk_list[j]
        r = k1 - k0
        if (j, ci) not in digc:
            d = ((ux[:, k0:k1] >> c.bit_lo) & ((1 << c.n_bits) - 1)).to(torch.float32)   # [T, r]
            digc = {(j, ci): (d, d.sum(dim=1))}
        dig, dsum = digc[(j, ci)]
        rows = torch.nonzero(dsum >= t_min).flatten()
        if rows.numel() == 0:
            continue
        if plan["cached"]:
            b, i, S = plan["loc"][e]
            wsel = plan["buckets"][b]["wb"][i, :r, :S].to(torch.float32)             # [r, S] from the cache
        else:
            if j not in uwc:                                         # weight codes of this chunk, once per call
                uwc[j] = _codes(qw_cpu[:, k0:k1], P, cfg.w_signed)   # [N, r]
            wsel = ((uwc[j][n_idx] >> p_idx.view(-1, 1)) & 1).to(torch.float32).t()  # [r, S]
        macs += rows.numel() * r * p_idx.numel()
        thr = mmin[ci][p_idx]                                        # [S]
        rb = max(1, _BLOCK_ELEMS // max(1, p_idx.numel()))          # bound the [rows, S] working set
        for r0 in range(0, rows.numel(), rb):
            rws = rows[r0:r0 + rb]
            m = dig[rws] @ wsel                                      # [R, S], exact: values <= r * L_c
            ri, si = torch.nonzero(m >= thr, as_tuple=True)          # only these can have S(m) != 0
            if ri.numel() == 0:
                continue
            mv = m[ri, si].round().to(torch.int64).clamp_max(M1 - 1)
            corr = flat_tab[ci][p_idx[si] * M1 + mv]
            keep = corr != 0
            ri, si, corr = ri[keep], si[keep], corr[keep]
            out.index_add_(0, rws[ri] * N + n_idx[si], corr)
    if stats is not None:
        stats["macs_exact"] = stats.get("macs_exact", 0) + macs
    return out.view(T, N).to(q_x.device)



def _mmin(plan: dict) -> torch.Tensor:
    """Smallest MAC value m with a non-zero table entry, per (cycle, plane): float32 [C, P] (M1 if none).
    Conversions below it contribute nothing and are skipped by both paths."""
    mmin = plan.get("mmin")
    if mmin is None:
        M1, flat_tab = plan["M1"], plan["flat_tab"]
        nzt = flat_tab.view(flat_tab.shape[0], -1, M1) != 0
        first = torch.where(nzt.any(dim=2), nzt.to(torch.int8).argmax(dim=2), torch.full(nzt.shape[:2], M1))
        mmin = plan["mmin"] = first.to(torch.float32)
    return mmin


def _batched(plan: dict, ux: torch.Tensor, cfg: CIMConfig, cycles, T: int, N: int, K: int,
             out: torch.Tensor, want_macs: bool) -> int:
    """All entries as padded batched matmuls: for a block of entries (chunk j, cycle c, columns S),
    m = digits[j, c] [T, r] @ weight bits [r, S] (exact in float32), then the table lookup and a sparse
    accumulation. Every token row is computed (no row selection); rows below an entry's bound give S(m) = 0.
    Adds into out [T * N]; returns the MACs the per-entry path would have computed (rows reaching the
    bound x r x S) when want_macs, else 0."""
    chunk_list, M1, flat_tab, r_max = plan["chunks"], plan["M1"], plan["flat_tab"], plan["r_max"]
    C, J = len(cycles), len(chunk_list)
    digs = torch.stack([((ux >> c.bit_lo) & ((1 << c.n_bits) - 1)) for c in cycles]).to(torch.float32)  # [C, T, K]
    widths = [k1 - k0 for k0, k1 in chunk_list]
    uniform = all(chunk_list[j][0] == j * r_max for j in range(J)) and all(w == r_max for w in widths[:-1])
    if uniform:
        pad = J * r_max - K
        d = torch.nn.functional.pad(digs, (0, pad)) if pad else digs
        D = d.view(C, T, J, r_max).permute(2, 0, 1, 3).contiguous()            # [J, C, T, r_max]
    else:
        D = torch.zeros(J, C, T, r_max)
        for j, (k0, k1) in enumerate(chunk_list):
            D[j, :, :, :k1 - k0] = digs[:, :, k0:k1]
    Dsum = D.sum(dim=3) if want_macs else None                                 # [J, C, T], exact integers
    mmin = _mmin(plan)
    macs = 0
    for bk in plan["buckets"]:
        E, smax = bk["wb"].shape[0], bk["smax"]
        step = max(1, min(_BATCH_ENTRIES, _BATCH_ELEMS // max(1, T * smax)))
        for e0 in range(0, E, step):
            sl = slice(e0, e0 + step)
            ej, ec = bk["ej"][sl], bk["ec"][sl]
            Dsel = D[ej, ec]                                                    # [e, T, r_max]
            wsel = bk["wb"][sl].to(torch.float32)                               # [e, r_max, smax]
            m = torch.bmm(Dsel, wsel)                                           # [e, T, smax], exact integers
            if want_macs:
                rows = (Dsum[ej, ec] >= bk["tmin"][sl].view(-1, 1)).sum(dim=1)            # [e]
                macs += int((rows * bk["r"][sl] * bk["S"][sl]).sum())
            # only conversions with m >= the smallest m that has a non-zero table entry can contribute;
            # the others give corr = 0 and are skipped (same set of non-zero additions, same order)
            pe = bk["p"][sl]                                                    # [e, smax]
            hit = m >= mmin[ec.view(-1, 1), pe].unsqueeze(1)                    # [e, T, smax]
            ei, ti, si = torch.nonzero(hit, as_tuple=True)
            if ei.numel() == 0:
                continue
            mv = m[ei, ti, si].round().to(torch.int64).clamp_max(M1 - 1)
            corr = flat_tab[ec[ei], pe[ei, si] * M1 + mv]
            keep = corr != 0
            ei, ti, si, corr = ei[keep], ti[keep], si[keep], corr[keep]
            out.index_add_(0, ti * N + bk["n"][sl][ei, si], corr)
    return macs
