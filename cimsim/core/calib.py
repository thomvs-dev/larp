"""ADC range calibration.

During calibration a layer records, for every (input cycle, weight plane),
the histogram of ideal analog MAC values m (integers in [0, nrow * L_c]).
The calibrated ADC full scale is chosen per group of (cycle, plane) pairs,
by one of two methods:

  mse         minimise the squared readout error sum_m h(m) (m - Q_FS(m))^2
              on the calibration data, weighted by the squared shift-add
              coefficient when the full scale is shared across planes.
              Ties go to the largest full scale (most headroom).
              Q_FS is the reference ADC: round(min(m / FS, 1) * L) / L * FS.
  percentile  the adc_range_pct percentile of m / (nominal full scale).

Granularity:

  layer        one fraction of the nominal full scale for the whole layer
  cycle        one fraction per input cycle
  cycle_plane  one fraction per (input cycle, weight plane)

Full scale values are integers in [1, nrow * L_c] (MAC units). Values of m
above the full scale saturate at the top ADC code.
"""

from __future__ import annotations

import math

import torch

from cimsim.config import CIMConfig
from cimsim.core.bitplanes import input_cycles


def _codes(q: torch.Tensor, bits: int, signed: bool) -> torch.Tensor:
    qi = q.to(torch.int32)
    if signed:
        qi = torch.where(qi < 0, qi + (1 << bits), qi)
    return qi


def mac_histogram(
    q_x: torch.Tensor,
    q_w: torch.Tensor,
    cfg: CIMConfig,
    chunks: list[tuple[int, int]] | None = None,
    token_block: int = 128,
    col_max: bool = False,
):
    """Histogram of ideal MAC values per (cycle, plane): int64 [C, P, M + 1],
    M = nrow * max(L_c). Uses the same bit decomposition as the executors.
    col_max: also return the largest MAC per (cycle, plane, output column), int64 [C, P, N]
    (used by stat_saturation 'exact' to skip columns that never come near saturation).
    MAC values are integers < 2^24, so they are exact in float32; the histogram is taken with one
    torch.histc call per block over (cycle, plane)-offset values (no int64 copies)."""
    T, K = q_x.shape
    N = q_w.shape[0]
    nrow, P = cfg.nrow, cfg.w_bits
    cycles = input_cycles(None, cfg.x_bits, cfg.signed_act, cfg.act_enc)
    C = len(cycles)
    max_len = max(k1 - k0 for k0, k1 in chunks) if chunks else min(K, nrow)
    M = max(nrow, max_len) * max(c.levels for c in cycles)
    nb = C * P * (M + 1)
    tb = max(1, min(token_block, (1 << 23) // max(1, N)))     # per-bin count per call < 2^23: exact in float32
    ux = _codes(q_x, cfg.x_bits, cfg.signed_act)
    digits = torch.stack([((ux >> c.bit_lo) & ((1 << c.n_bits) - 1)).to(torch.float32) for c in cycles], dim=1)
    uw = _codes(q_w, P, cfg.w_signed).to(torch.int16)
    shifts = torch.arange(P, dtype=torch.int16).view(P, 1, 1)
    base = (torch.arange(C * P, dtype=torch.float32) * (M + 1)).view(1, C, P, 1)
    hist = torch.zeros(nb, dtype=torch.float64)
    cmax = torch.zeros(C, P, N, dtype=torch.float32) if col_max else None
    chunk_list = [(k0, min(K, k0 + nrow)) for k0 in range(0, K, nrow)] if chunks is None else chunks
    for k0, k1 in chunk_list:
        r = k1 - k0
        wk = ((uw[:, k0:k1].unsqueeze(0) >> shifts) & 1).to(torch.float32).reshape(P * N, r).t().contiguous()
        for t0 in range(0, T, tb):
            t1 = min(T, t0 + tb)
            xk = digits[t0:t1, :, k0:k1].reshape((t1 - t0) * C, r)
            m = (xk @ wk).view(t1 - t0, C, P, N)
            if col_max:
                cmax = torch.maximum(cmax, m.amax(dim=0))
            hist += torch.histc(m.add_(base), bins=nb, min=-0.5, max=nb - 0.5).double()
    hist = hist.round().to(torch.int64).view(C, P, M + 1)
    if col_max:
        return hist, cmax.to(torch.int64)
    return hist


def _quantile_fraction(hist_rows: list[torch.Tensor], fs_noms: list[float], pct: float) -> float:
    """Weighted quantile of u = m / FS_nom over several histograms."""
    vals, wts = [], []
    for h, fsn in zip(hist_rows, fs_noms):
        nz = torch.nonzero(h).flatten()
        vals.append(nz.double() / fsn)
        wts.append(h[nz].double())
    v = torch.cat(vals)
    w = torch.cat(wts)
    if w.sum() == 0:
        return 0.0
    order = torch.argsort(v)
    v, w = v[order], w[order]
    cum = torch.cumsum(w, 0) / w.sum()
    idx = int(torch.searchsorted(cum, torch.tensor(pct / 100.0, dtype=torch.float64)).clamp(max=len(v) - 1))
    return float(v[idx])


def _readout(m: torch.Tensor, F: torch.Tensor, adc_bits: int | None) -> torch.Tensor:
    """Reference readout for integer MAC m (row) and full scales F (column)."""
    v = torch.clamp(m / F, 0.0, 1.0)
    if adc_bits is not None:
        L = float(2 ** adc_bits - 1)
        v = torch.round(v * L) / L
    return v * F


def _error_table(hist: torch.Tensor, fsn: int, adc_bits: int | None) -> torch.Tensor:
    """Squared readout error for every candidate FS = 1..fsn: [fsn] (index FS-1)."""
    M1 = hist.numel()
    m = torch.arange(M1, dtype=torch.float64).view(1, -1)
    F = torch.arange(1, fsn + 1, dtype=torch.float64).view(-1, 1)
    err = (m - _readout(m, F, adc_bits)) ** 2                  # [fsn, M+1]
    return err @ hist.double()


def full_scale_from_histogram(hist: torch.Tensor, cfg: CIMConfig) -> torch.Tensor:
    """Calibrated full scale [C, P] (float64, integer values) from mac_histogram output."""
    cycles = input_cycles(None, cfg.x_bits, cfg.signed_act, cfg.act_enc)
    C, P = hist.shape[0], hist.shape[1]
    fs_nom = [cfg.nrow * c.levels for c in cycles]
    if cfg.adc_range_gran == "layer":
        groups = [[(c, p) for c in range(C) for p in range(P)]]
    elif cfg.adc_range_gran == "cycle":
        groups = [[(c, p) for p in range(P)] for c in range(C)]
    else:
        groups = [[(c, p)] for c in range(C) for p in range(P)]
    fs = torch.empty(C, P, dtype=torch.float64)

    if cfg.adc_range_method == "percentile":
        for g in groups:
            frac = _quantile_fraction([hist[c, p] for c, p in g], [float(fs_nom[c]) for c, _ in g],
                                      cfg.adc_range_pct)
            for c, p in g:
                fs[c, p] = min(fs_nom[c], max(1.0, math.ceil(frac * fs_nom[c] - 1e-9)))
        return fs

    # mse: per (c, p) error tables, combined per group with significance weights
    tables = {}
    for c in range(C):
        for p in range(P):
            tables[(c, p)] = _error_table(hist[c, p], fs_nom[c], cfg.adc_bits)
    for g in groups:
        if len(g) == 1:
            c, p = g[0]
            e = tables[(c, p)]
            best = e.min()
            ok = torch.nonzero(e <= best * (1 + 1e-12) + 1e-12).flatten()
            fs[c, p] = float(ok.max() + 1)                      # largest FS among the minima
            continue
        fmax = max(fs_nom[c] for c, _ in g)
        fracs = torch.arange(1, fmax + 1, dtype=torch.float64) / fmax
        total = torch.zeros(fmax, dtype=torch.float64)
        for c, p in g:
            cyc = cycles[c]
            w = float(2.0 ** (cyc.shift + p)) ** 2              # squared shift-add weight
            idx = torch.clamp(torch.ceil(fracs * fs_nom[c] - 1e-9), 1, fs_nom[c]).long() - 1
            total += w * tables[(c, p)][idx]
        best = total.min()
        k = int(torch.nonzero(total <= best * (1 + 1e-12) + 1e-12).flatten().max())
        for c, p in g:
            fs[c, p] = float(min(fs_nom[c], max(1, math.ceil(fracs[k].item() * fs_nom[c] - 1e-9))))
    return fs
