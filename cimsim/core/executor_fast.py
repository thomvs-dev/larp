"""CPU-optimised bit-accurate executor.

Same semantics as cimsim/core/executor.py (the reference, which stays the
ground truth for ASiM equivalence). Differences are implementation only:

1. Bit decomposition uses small integer types; each cycle's digit is one
   shift+mask. Weight planes are built per row chunk.
2. Layout [T, C, K] for input digits, so the MAC result comes out as
   [T, C*P, N] and the shift-add over (cycle, weight plane) is one batched
   matmul with the coefficient vector (no permute copy).
3. Noise-free ADC = round(m * a_c) with the scale b_c folded into the
   shift-add coefficients (2 elementwise passes instead of 5). For each
   cycle, a_c is chosen at setup so that round(m * a_c) reproduces the
   reference readout for every integer m in [0, full scale]; if no float32
   a_c does, the cycle falls back to a lookup table (exact, slower).
4. Noisy readout runs in float32 with in-place ops.
5. Per-chunk shift-add in float32, accumulated in float64 across chunks.
   Matches the reference to ~1e-6 relative (not bit-for-bit).
6. Shortcut: if the readout is lossless for every cycle (no noise, ADC
   reproduces every integer in [0, full scale]), returns the integer matmul.

tests/test_fast_executor.py compares this against the reference.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch

from cimsim.config import CIMConfig
from cimsim.core.bitplanes import Cycle, input_cycles
from cimsim.core.executor import analog_mask


def _codes(q: torch.Tensor, bits: int, signed: bool) -> torch.Tensor:
    qi = q.to(torch.int32)
    if signed:
        qi = torch.where(qi < 0, qi + (1 << bits), qi)
    return qi


@lru_cache(maxsize=256)
def _ref_codes(full_scale: int, adc_bits: int) -> torch.Tensor:
    """Reference ADC code k(m) for m = 0..FS (float64 formula from executor.py)."""
    m = torch.arange(full_scale + 1, dtype=torch.float64)
    L = float(2 ** adc_bits - 1)
    return torch.round(torch.clamp(m / full_scale, 0.0, 1.0) * L)


@lru_cache(maxsize=256)
def _adc_scale(full_scale: int, adc_bits: int) -> float | None:
    """float32 a such that round(float32(m) * a) == reference code for all m,
    or None if no candidate near L/FS works."""
    L = 2 ** adc_bits - 1
    ref = _ref_codes(full_scale, adc_bits).float()
    m = torch.arange(full_scale + 1, dtype=torch.float32)
    a0 = torch.tensor(L / full_scale, dtype=torch.float32)
    cands = [a0]
    up, dn = a0.clone(), a0.clone()
    for _ in range(8):
        up = torch.nextafter(up, torch.tensor(float("inf")))
        dn = torch.nextafter(dn, torch.tensor(float("-inf")))
        cands += [up.clone(), dn.clone()]
    for a in cands:
        if torch.equal(torch.round(m * a), ref):
            return float(a)
    return None


def readout_is_lossless(cfg: CIMConfig, cycles: list[Cycle]) -> bool:
    if cfg.rand_noise_sigma > 0 or cfg.non_linear_sigma > 0:
        return False
    if cfg.adc_bits is None:
        return True
    for c in cycles:
        fs = cfg.nrow * c.levels
        if not torch.equal(_ref_codes(fs, cfg.adc_bits) * (fs / (2 ** cfg.adc_bits - 1)),
                           torch.arange(fs + 1, dtype=torch.float64)):
            return False
    return True


def cim_matmul_int_fast(
    q_x: torch.Tensor,
    q_w: torch.Tensor,
    cfg: CIMConfig,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Same contract as executor.cim_matmul_int. Returns float64 [T, N]."""
    T, K = q_x.shape
    N = q_w.shape[0]
    dev = q_x.device
    nrow = cfg.nrow
    P = cfg.w_bits
    cycles = input_cycles(None, cfg.x_bits, cfg.signed_act, cfg.act_enc)
    C = len(cycles)

    if readout_is_lossless(cfg, cycles):
        return q_x.double() @ q_w.double().t()

    noisy = cfg.rand_noise_sigma > 0 or cfg.non_linear_sigma > 0
    s_rand = cfg.rand_noise_sigma / 100.0
    s_nl = cfg.non_linear_sigma / 100.0
    adc = cfg.adc_bits
    adc_L = None if adc is None else float(2 ** adc - 1)
    fs_list = [nrow * c.levels for c in cycles]

    # shift-add coefficients [C, P] (powers of two)
    coef = torch.empty(C, P, dtype=torch.float64)
    for ci, c in enumerate(cycles):
        for p in range(P):
            coef[ci, p] = c.sign * (-1 if p == P - 1 else 1) * 2.0 ** (c.shift + p)

    # noise-free readout plan per cycle: scale a_c (or LUT) and output factor b_c
    pre_scale = torch.ones(C, dtype=torch.float32)
    post = torch.ones(C, dtype=torch.float64)
    lut_cycles: dict[int, torch.Tensor] = {}
    if not noisy:
        for ci, fs in enumerate(fs_list):
            if adc is None:
                continue
            a = _adc_scale(fs, adc)
            if a is None:
                lut_cycles[ci] = _ref_codes(fs, adc).float().to(dev)
            else:
                pre_scale[ci] = a
            post[ci] = fs / adc_L
    coef_eff = (coef * post.view(C, 1)).to(torch.float32).reshape(1, 1, C * P).to(dev)
    pre_v = pre_scale.to(dev).view(1, C, 1, 1)
    any_scale = adc is not None and not noisy

    if noisy:
        fs_v = torch.tensor(fs_list, dtype=torch.float32, device=dev).view(1, C, 1, 1)
        amask = analog_mask(P, C, cfg.hybrid_levels).to(dev).view(1, C, P, 1)
        all_analog = bool(amask.all())
        coef_noisy = coef.to(torch.float32).reshape(1, 1, C * P).to(dev)

    # input digits [T, C, K] uint8
    ux = _codes(q_x, cfg.x_bits, cfg.signed_act)
    digits = torch.stack([((ux >> c.bit_lo) & ((1 << c.n_bits) - 1)).to(torch.uint8) for c in cycles], dim=1)
    uw = _codes(q_w, P, True).to(torch.int16)
    shifts = torch.arange(P, dtype=torch.int16, device=dev).view(P, 1, 1)

    y = torch.zeros(T, N, dtype=torch.float64, device=dev)
    tb = max(1, cfg.token_block)

    for k0 in range(0, K, nrow):
        k1 = min(K, k0 + nrow)
        r = k1 - k0
        wk = ((uw[:, k0:k1].unsqueeze(0) >> shifts) & 1).to(torch.float32)       # [P, N, r]
        wk = wk.reshape(P * N, r).t().contiguous()                                # [r, P*N]
        for t0 in range(0, T, tb):
            t1 = min(T, t0 + tb)
            Tb = t1 - t0
            xk = digits[t0:t1, :, k0:k1].reshape(Tb * C, r).to(torch.float32)
            m = (xk @ wk).view(Tb, C, P, N)                                      # exact integers

            if not noisy:
                if any_scale:
                    m.mul_(pre_v).round_()
                    for ci, lut in lut_cycles.items():
                        m[:, ci] = lut[m[:, ci].to(torch.int64)]
                part = torch.matmul(coef_eff, m.view(Tb, C * P, N))              # [Tb, 1, N]
            else:
                v = m.div_(fs_v).clamp_(0.0, 1.0)
                noise = torch.empty_like(v)
                if s_rand > 0:
                    noise.normal_(0.0, s_rand, generator=generator)
                else:
                    noise.zero_()
                if s_nl > 0:
                    nl = torch.empty_like(v).normal_(0.0, 1.0, generator=generator)
                    nl.div_(torch.sqrt(v * nrow + 1.0)).mul_(s_nl)
                    noise.add_(nl)
                if all_analog:
                    v.add_(noise).clamp_(0.0, 1.0)
                else:
                    v = torch.where(amask, (v + noise).clamp_(0.0, 1.0), v)
                if adc_L is not None:
                    v.mul_(adc_L).round_().div_(adc_L)
                v.mul_(fs_v)
                part = torch.matmul(coef_noisy, v.reshape(Tb, C * P, N))
            y[t0:t1] += part.view(Tb, N).double()
    return y
