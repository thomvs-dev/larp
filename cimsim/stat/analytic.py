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
from cimsim.stat.saturation import deterministic_tables

_SQRT2 = math.sqrt(2.0)
_FLIP_Q_MAX = 0.05          # largest per-conversion flip probability sampled as discrete flips


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
    p = code_probs(m, s, upper, fs, L)
    vals = k * D
    mean = p @ vals
    var = (p @ (vals * vals) - mean * mean).clamp_min(0.0)
    return mean, var


def code_probs(m: torch.Tensor, s: torch.Tensor, upper: float, fs: float, L: int) -> torch.Tensor:
    """P(ADC code = k | m) for k = 0..L: float64 [n, L + 1] (Gaussian noise s, clamp to [0, upper])."""
    m = m.double()
    s = s.double().expand_as(m)
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
    return (cdf[:, 1:] - cdf[:, :-1]).clamp_min(0.0)                              # [n, L+1]


def layer_stat_params(hist: torch.Tensor, cfg: CIMConfig, n_chunks: int,
                      full_scale: torch.Tensor | None = None, exclude_saturation: bool = False,
                      flip_lambda_max: float | None = None,
                      flip_kurtosis_min: float | None = None) -> dict:
    """hist: [C, P, M+1] counts of ideal MAC values (cimsim.core.calib.mac_histogram).
    Returns {'gain', 'mean', 'std'} in integer output units plus per-(cycle, plane) diagnostics.
    exclude_saturation: remove the exact sparse deterministic term T(m) (cimsim.stat.saturation.
    deterministic_tables: saturation, and nonlinear rounding above the exactly linear region) from d(m);
    it is then computed exactly per conversion (F8, F9). The noise variance keeps the clamp, so noise on
    saturated conversions is absorbed as in the executors.
    flip_lambda_max: analog (cycle, plane) pairs whose per-conversion probability of a noise-induced ADC
    code change q is at most 0.05 and whose expected number per output (n_chunks · q) is below this value
    are returned in 'flips' for discrete sampling (F9) and left out of the Gaussian part; None = Gaussian
    for every pair.
    flip_kurtosis_min: if the discrete flips would give the layer's output error an excess kurtosis below
    this value (many flips per output, close to Gaussian), every pair is modelled as Gaussian instead:
    the discrete path costs an exact MAC per candidate and gives the same result there."""
    cycles = input_cycles(None, cfg.x_bits, cfg.signed_act, cfg.act_enc)
    C, P = len(cycles), cfg.w_bits
    amask = analog_mask(P, C, cfg.hybrid_levels)
    s_rand, s_nl = cfg.rand_noise_sigma / 100.0, cfg.non_linear_sigma / 100.0
    L = None if cfg.adc_bits is None else 2 ** cfg.adc_bits - 1
    sat = deterministic_tables(cfg, full_scale, hist.shape[-1] - 1, hist)[0] if exclude_saturation else None
    rows, flips = [], []
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
            if flip_lambda_max is not None and L is not None and bool((s > 0).any()):
                f = _flip_distribution(m, s, w, min(fsn, fs), fs, L)
                # discrete only for rare flips: Poisson placement is exact for q << 1 (Binomial otherwise)
                if f is not None and f["q"] <= _FLIP_Q_MAX and n_chunks * f["q"] < flip_lambda_max:
                    f.update(c=ci, p=pi, coef=coef, levels=c.levels, bit_lo=c.bit_lo, n_bits=c.n_bits)
                    m_all = torch.arange(hist.shape[-1], dtype=torch.float64)
                    s_all = torch.sqrt((s_rand * fsn) ** 2 + (s_nl * fsn) ** 2 / (m_all / c.levels + 1.0))
                    f.update(_flip_tables(m_all, s_all, min(fsn, fs), fs, L))
                    flips.append(f)
                    mean_out = f["k0"] * (fs / L)                   # noise-free readout; flips sampled separately
                    var_out = torch.zeros_like(m)
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
    for f in flips:
        f.pop("k0")
    flip_var = n_chunks * sum(f["coef"] ** 2 * f["step"] ** 2 * f["q"] * float((f["probs"] * f["values"] ** 2).sum())
                              for f in flips)
    # excess kurtosis of the output error: 4th cumulant of the compound-Poisson flip sum over total variance^2
    k4 = n_chunks * sum(f["coef"] ** 4 * f["step"] ** 4 * f["q"] * float((f["probs"] * f["values"] ** 4).sum())
                        for f in flips)
    v_tot = var + flip_var
    kurt = k4 / v_tot ** 2 if v_tot > 0 else 0.0
    if flips and flip_kurtosis_min is not None and kurt < flip_kurtosis_min:
        out = layer_stat_params(hist, cfg, n_chunks, full_scale, exclude_saturation, flip_lambda_max=None)
        out["flip_kurtosis"] = kurt
        out["flip_fallback"] = "gaussian"
        return out
    return {"gain": gbar, "mean": mu, "std": math.sqrt(var), "std_noise_only": math.sqrt(var_noise),
            "n_chunks": n_chunks, "per_cp": rows, "flips": flips,
            "flip_pairs": len(flips), "flip_events_per_output": n_chunks * sum(f["q"] for f in flips),
            "std_flips": math.sqrt(flip_var), "flip_kurtosis": kurt,
            "mean_flips": n_chunks * sum(f["coef"] * f["step"] * f["q"] * float((f["probs"] * f["values"]).sum())
                                         for f in flips)}


def _flip_tables(m, s, upper, fs, L):
    """Per-MAC-value flip tables for exact thinning (F9): pany[m] = P(code != noise-free code | m),
    pmax = max pany, jvals = offsets with non-negligible probability, cond_cdf[m, :] = cumulative
    P(j | m, j != 0) over jvals."""
    p = code_probs(m, s, upper, fs, L)                              # [M+1, L+1]
    D = fs / L
    k0 = torch.round(m.clamp(0.0, upper) / D)
    M1 = m.numel()
    pj = torch.zeros(M1, 2 * L + 1, dtype=torch.float64)
    j = (torch.arange(L + 1, dtype=torch.float64).view(1, -1) - k0.view(-1, 1)).long() + L
    pj.scatter_add_(1, j, p)
    p0 = pj[:, L].clone()
    pj[:, L] = 0.0
    support = torch.nonzero(pj.max(dim=0).values > 1e-15).flatten()
    pj = pj[:, support]
    pany = (1.0 - p0).clamp(0.0, 1.0)
    tot = pj.sum(1, keepdim=True)
    cond = torch.where(tot > 0, pj / tot.clamp_min(1e-300), torch.full_like(pj, 1.0 / max(1, pj.shape[1])))
    cdf = torch.cumsum(cond, 1)
    cdf[:, -1] = 1.0
    return {"pany": pany, "pmax": float(pany.max()), "jvals": (support - L).double(), "cond_cdf": cdf}


def _flip_distribution(m, s, w, upper, fs, L):
    """Distribution of the noise-induced code offset j = code - noise-free code for one (cycle, plane),
    averaged over the calibration histogram (weights w). Returns q = P(j != 0), the non-zero offsets
    and their conditional probabilities, step = ADC step, k0 = noise-free code per m; None if q = 0."""
    p = code_probs(m, s, upper, fs, L)                              # [n, L+1]
    D = fs / L
    k0 = torch.round(m.clamp(0.0, upper) / D)                       # executor rounding (round half to even)
    j = torch.arange(L + 1, dtype=torch.float64).view(1, -1) - k0.view(-1, 1)   # [n, L+1]
    pj = torch.zeros(2 * L + 1, dtype=torch.float64)
    pj.index_add_(0, (j + L).long().flatten(), (w.view(-1, 1) * p).flatten())
    q = float(1.0 - pj[L])
    if q <= 1e-15:
        return None
    pj[L] = 0.0
    nz = torch.nonzero(pj > 1e-15).flatten()
    probs = pj[nz] / pj[nz].sum()
    return {"q": q, "values": (nz - L).double(), "probs": probs, "step": D, "k0": k0}
