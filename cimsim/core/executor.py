"""Bit-accurate SRAM ACiM matrix multiply on integer operands.

Semantics follow ASiM (module/asim_linear.py, Apache 2.0); see
docs/part_a_plan.md Section 2. All cycles and weight planes of one row
chunk are computed with a single matmul, looping over row chunks and
token blocks to bound memory.
"""

from __future__ import annotations

import math

import torch

from cimsim.config import CIMConfig
from cimsim.core.bitplanes import input_cycles, input_digits, weight_planes


def analog_mask(n_wplanes: int, n_cycles: int, hybrid_levels: int) -> torch.Tensor:
    """[n_cycles, n_wplanes] bool. True where the conversion is analog
    (gets noise). ASiM rule: i + j <= B_w + n_cycles - hybrid_levels - 2."""
    j = torch.arange(n_cycles).view(-1, 1)
    i = torch.arange(n_wplanes).view(1, -1)
    return (i + j) <= (n_wplanes + n_cycles - hybrid_levels - 2)


def analytic_counts(T: int, K: int, N: int, cfg: CIMConfig) -> dict:
    """Event counts for one forward call of a [T,K] x [K,N] layer.
    Same formulas in every mode (docs/part_a_plan.md 2.5)."""
    n_chunks = math.ceil(K / cfg.nrow)
    cycles = input_cycles(None, cfg.x_bits, cfg.signed_act, cfg.act_enc)
    C, P = len(cycles), cfg.w_bits
    n_analog = int(analog_mask(P, C, cfg.hybrid_levels).sum())
    return {
        "macs": T * K * N,
        "adc_conversions": T * n_chunks * C * P * N,
        "adc_conversions_analog": T * n_chunks * n_analog * N,
        "array_cycles": T * n_chunks * C,
    }


def cim_matmul_int(
    q_x: torch.Tensor,
    q_w: torch.Tensor,
    cfg: CIMConfig,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """q_x: [T, K] integer-valued, q_w: [N, K] integer-valued.
    Returns y_int [T, N] float64 (integer units, before dequantization)."""
    T, K = q_x.shape
    N = q_w.shape[0]
    dev = q_x.device
    nrow = cfg.nrow

    planes, wmeta = weight_planes(q_w, cfg.w_bits)                  # [P, N, K]
    cycles = input_cycles(q_x, cfg.x_bits, cfg.signed_act, cfg.act_enc)
    C, P = len(cycles), cfg.w_bits

    # coefficient per (cycle, weight plane): sign * 2^(shift_c + shift_i)
    coef = torch.empty(C, P, dtype=torch.float64, device=dev)
    for ci, c in enumerate(cycles):
        for pi, (ws, wsg) in enumerate(wmeta):
            coef[ci, pi] = c.sign * wsg * (2.0 ** (c.shift + ws))
    full_scale = torch.tensor([nrow * c.levels for c in cycles], dtype=torch.float64, device=dev)
    fs_view = full_scale.view(C, 1, 1, 1)                                 # broadcast over [C,Tb,P,N]
    amask = analog_mask(P, C, cfg.hybrid_levels).to(dev).view(C, 1, P, 1)

    noisy = cfg.rand_noise_sigma > 0 or cfg.non_linear_sigma > 0
    exact_readout = cfg.adc_bits is None and not noisy
    adc_levels = None if cfg.adc_bits is None else float(2 ** cfg.adc_bits - 1)
    s_rand = cfg.rand_noise_sigma / 100.0
    s_nl = cfg.non_linear_sigma / 100.0

    y = torch.zeros(T, N, dtype=torch.float64, device=dev)
    n_chunks = math.ceil(K / nrow)
    tb = max(1, cfg.token_block)

    for t0 in range(0, T, tb):
        t1 = min(T, t0 + tb)
        Tb = t1 - t0
        digits = input_digits(q_x[t0:t1], cfg.x_bits, cfg.signed_act, cycles)   # [C, Tb, K]
        y_blk = torch.zeros(Tb, N, dtype=torch.float64, device=dev)
        for k in range(n_chunks):
            k0, k1 = k * nrow, min(K, (k + 1) * nrow)
            r = k1 - k0
            xk = digits[:, :, k0:k1].reshape(C * Tb, r)
            wk = planes[:, :, k0:k1].reshape(P * N, r).t()
            m = (xk @ wk).view(C, Tb, P, N)                            # exact small integers

            if exact_readout:
                m_hat = m.double()
            else:
                v = torch.clamp(m.double() / fs_view, 0.0, 1.0)
                if noisy:
                    noise = torch.zeros_like(v)
                    if s_rand > 0:
                        noise += torch.randn(v.shape, generator=generator, device=dev, dtype=v.dtype) * s_rand
                    if s_nl > 0:
                        noise += (torch.randn(v.shape, generator=generator, device=dev, dtype=v.dtype)
                                  / torch.sqrt(v * nrow + 1.0) * s_nl)
                    v = torch.where(amask, torch.clamp(v + noise, 0.0, 1.0), v)
                if adc_levels is not None:
                    v = torch.round(v * adc_levels) / adc_levels
                m_hat = v * fs_view
            y_blk += torch.einsum("ctpn,cp->tn", m_hat, coef)
        y[t0:t1] = y_blk
    return y
