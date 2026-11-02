"""A3 analytic error model (M3): layer error statistics from the MAC histogram.

One conversion in the bit-accurate executors (cimsim/core/executor.py), in MAC units:

    x   = m + n                      n ~ N(0, s(m)^2) on analog conversions, 0 otherwise
    x_c = clamp(x, 0, U)             U = min(FS_nominal, FS)   (array range, then ADC range)
    out = D * round(x_c / D)         D = FS / L   (no rounding if adc_bits is None)

    s(m)^2 = (sigma_rand * FS_nom)^2 + (sigma_nl * FS_nom)^2 / (m / L_c + 1)

For every MAC value m seen during calibration this module computes E[out | m] and Var[out | m]
exactly (Gaussian CDF over the ADC bins). Per (cycle, plane) the deterministic part
d(m) = E[out | m] - m is fitted as g_cp * m + b_cp over the histogram. The layer output error

    e = sum over chunks, cycles c, planes p of  coef_cp * (out - m),   coef_cp = sign * 2^(shift_c + shift_p)

is then modelled as

    e ~ g * y_ideal + N(mu, sigma^2)

with g the significance-weighted mean of g_cp, and mu, sigma^2 collecting the intercepts, noise,
fit residuals and the spread of g_cp around g. Conversions are treated as independent, which is
exact for the analog noise and an approximation for the deterministic rounding / clipping residual.
Findings behind this model: docs/findings.md F7.
"""

from __future__ import annotations

import math

import torch

from cimsim.config import CIMConfig
from cimsim.core.bitplanes import input_cycles
from cimsim.core.executor import analog_mask
from cimsim.stat.saturation import saturation_tables

_SQRT2 = math.sqrt(2.0)


def _Phi(z: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(z / _SQRT2))


def _phi(z: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def conversion_moments(m: torch.Tensor, s: torch.Tensor, upper: float, fs: float,
                       adc_levels: int | None) -> tuple[torch.Tensor, torch.Tensor]:
    """E[out | m] and Var[out | m] for one (cycle, plane). m, s: float64 [n]. s may be 0."""
    m = m.double()
    s = s.double().expand_as(m)
    if adc_levels is None:
        # clamp(N(m, s^2), 0, U): closed form for a clipped Gaussian
        det = s <= 0
        ss = torch.where(det, torch.ones_like(s), s)
        a, b = (0.0 - m) / ss, (upper - m) / ss
        Pa, Pb, pa, pb = _Phi(a), _Phi(b), _phi(a), _phi(b)
        mid = Pb - Pa
        e1 = upper * (1 - Pb) + m * mid + ss * (pa - pb)
        e2 = upper ** 2 * (1 - Pb) + (m * m + ss * ss) * mid + 2 * m * ss * (pa - pb) + ss * ss * (a * pa - b * pb)
        xc = m.clamp(0.0, upper)
        mean = torch.where(det, xc, e1)
        var = torch.where(det, torch.zeros_like(m), (e2 - e1 * e1).clamp_min(0.0))
        return mean, var
    L = int(adc_levels)
    D = fs / L
    k = torch.arange(L + 1, dtype=torch.float64)
    thr = (k[:-1] + 0.5) * D                                   # upper edge of codes 0 .. L-1
    above = thr >= upper                                        # clamp(x) < thr always
    det = s <= 0
    ss = torch.where(det, torch.ones_like(s), s)
    cdf = _Phi((thr.view(1, -1) - m.view(-1, 1)) / ss.view(-1, 1))             # [n, L]
    cdf_det = (m.clamp(0.0, upper).view(-1, 1) < thr.view(1, -1)).double()
    cdf = torch.where(det.view(-1, 1), cdf_det, cdf)
    cdf = torch.where(above.view(1, -1), torch.ones_like(cdf), cdf)
    cdf = torch.cat([torch.zeros(len(m), 1, dtype=torch.float64), cdf,
                     torch.ones(len(m), 1, dtype=torch.float64)], dim=1)
    p = (cdf[:, 1:] - cdf[:, :-1]).clamp_min(0.0)                               # [n, L+1]
    vals = k * D
    mean = p @ vals
    var = (p @ (vals * vals) - mean * mean).clamp_min(0.0)
    return mean, var


def layer_stat_params(hist: torch.Tensor, cfg: CIMConfig, n_chunks: int,
                      full_scale: torch.Tensor | None = None, exclude_saturation: bool = False) -> dict:
    """hist: [C, P, M+1] counts of ideal MAC values (cimsim.core.calib.mac_histogram).
    Returns {'gain', 'mean', 'std'} in integer output units plus per-(cycle, plane) diagnostics.
    exclude_saturation: remove the noise-free saturation term S(m) (cimsim.stat.saturation) from the
    deterministic part d(m); it is then computed exactly per conversion (F8). The noise variance keeps
    the clamp, so noise on saturated conversions is absorbed as in the executors."""
    cycles = input_cycles(None, cfg.x_bits, cfg.signed_act, cfg.act_enc)
    C, P = len(cycles), cfg.w_bits
    amask = analog_mask(P, C, cfg.hybrid_levels)
    s_rand, s_nl = cfg.rand_noise_sigma / 100.0, cfg.non_linear_sigma / 100.0
    L = None if cfg.adc_bits is None else 2 ** cfg.adc_bits - 1
    sat = saturation_tables(cfg, full_scale, hist.shape[-1] - 1) if exclude_saturation else None
    rows = []
    for ci, c in enumerate(cycles):
        fsn = float(cfg.nrow * c.levels)
        for pi in range(P):
            wsign = -1 if (cfg.w_signed and pi == P - 1) else 1
            coef = c.sign * wsign * 2.0 ** (c.shift + pi)
            h = hist[ci, pi].double()
            tot = float(h.sum())
            if tot == 0:
                continue
            idx = torch.nonzero(h).flatten()
            w = h[idx] / tot
            m = idx.double()
            fs = fsn if full_scale is None else float(full_scale.reshape(C, P)[ci, pi])
            if amask[ci, pi]:
                s = torch.sqrt((s_rand * fsn) ** 2 + (s_nl * fsn) ** 2 / (m / c.levels + 1.0))
            else:
                s = torch.zeros_like(m)
            mean_out, var_out = conversion_moments(m, s, min(fsn, fs), fs, L)
            d = mean_out - m
            if exclude_saturation:
                d = d - sat[ci, pi, idx]
            Em, Em2 = float((w * m).sum()), float((w * m * m).sum())
            Vm = max(Em2 - Em * Em, 0.0)
            Ed = float((w * d).sum())
            if Vm > 0:
                g = float((w * (m - Em) * (d - Ed)).sum()) / Vm
            else:
                g = 0.0
            b = Ed - g * Em
            resid = float((w * (d - g * m - b) ** 2).sum())
            rows.append(dict(c=ci, p=pi, coef=coef, g=g, b=b, Em=Em, Em2=Em2, Vm=Vm,
                             noise_var=float((w * var_out).sum()), resid_var=resid))
    wts = [r["coef"] ** 2 * r["Em2"] for r in rows]
    gbar = sum(wt * r["g"] for wt, r in zip(wts, rows)) / sum(wts) if rows and sum(wts) > 0 else 0.0
    mu = n_chunks * sum(r["coef"] * (r["b"] + (r["g"] - gbar) * r["Em"]) for r in rows)
    var = n_chunks * sum(r["coef"] ** 2 * (r["noise_var"] + r["resid_var"] + (r["g"] - gbar) ** 2 * r["Vm"])
                         for r in rows)
    var_noise = n_chunks * sum(r["coef"] ** 2 * r["noise_var"] for r in rows)
    return {"gain": gbar, "mean": mu, "std": math.sqrt(var), "std_noise_only": math.sqrt(var_noise),
            "n_chunks": n_chunks, "per_cp": rows}
