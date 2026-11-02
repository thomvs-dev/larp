"""Symmetric integer quantization (ASiM-compatible).

Weights:     q = round(clamp(|w| / s, 0, 1) * (2^(B-1) - 1)) * sign(w)
Signed act:  same form with x_bits.
Unsigned act: q = round(clamp(x / s, 0, 1) * (2^B - 1))

s is the absolute max over the chosen granularity. Returned scales have
shapes that broadcast against the integer tensors.
"""

from __future__ import annotations

import torch

EPS = 1e-7


def qmax_signed(bits: int) -> int:
    return 2 ** (bits - 1) - 1


def qmax_unsigned(bits: int) -> int:
    return 2 ** bits - 1


def quantize_weight(w: torch.Tensor, bits: int, gran: str) -> tuple[torch.Tensor, torch.Tensor]:
    """w: [N, K]. Returns (q_int as float tensor, scale [N,1] or [1,1])."""
    if gran == "per_tensor":
        s = w.abs().max().reshape(1, 1) + EPS
    elif gran == "per_channel":
        s = w.abs().amax(dim=1, keepdim=True) + EPS
    else:
        raise ValueError(gran)
    qm = qmax_signed(bits)
    q = torch.round(torch.clamp(w.abs() / s, 0.0, 1.0) * qm) * torch.sign(w)
    return q, s


def quantize_weight_asym(w: torch.Tensor, bits: int, gran: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Asymmetric min-max weight quantization with zero point.
    w: [N, K]. Returns (q_u in [0, 2^bits - 1], step [N,1] or [1,1], zero [N,1] or [1,1]).
    Real weight = (q_u - zero) * step."""
    if gran == "per_tensor":
        lo, hi = w.min().reshape(1, 1), w.max().reshape(1, 1)
    elif gran == "per_channel":
        lo, hi = w.amin(dim=1, keepdim=True), w.amax(dim=1, keepdim=True)
    else:
        raise ValueError(gran)
    lo = torch.minimum(lo, torch.zeros_like(lo))   # range always contains 0
    hi = torch.maximum(hi, torch.zeros_like(hi))
    qm = 2 ** bits - 1
    step = (hi - lo).clamp_min(EPS) / qm
    zero = torch.clamp(torch.round(-lo / step), 0, qm)
    q = torch.clamp(torch.round(w / step) + zero, 0, qm)
    return q, step, zero


def quantize_act(x: torch.Tensor, bits: int, gran: str, signed: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """x: [T, K]. Returns (q_int as float tensor, scale [T,1] or [1,1])."""
    if gran == "per_tensor":
        s = (x.abs().max() if signed else x.max()).reshape(1, 1) + EPS
    elif gran == "per_token":
        s = (x.abs().amax(dim=1, keepdim=True) if signed else x.amax(dim=1, keepdim=True)) + EPS
    else:
        raise ValueError(gran)
    if signed:
        qm = qmax_signed(bits)
        q = torch.round(torch.clamp(x.abs() / s, 0.0, 1.0) * qm) * torch.sign(x)
    else:
        qm = qmax_unsigned(bits)
        q = torch.round(torch.clamp(x / s, 0.0, 1.0) * qm)
    return q, s


def dequant_factor(w_bits: int, x_bits: int, signed_act: bool) -> float:
    qx = qmax_signed(x_bits) if signed_act else qmax_unsigned(x_bits)
    return 1.0 / (qmax_signed(w_bits) * qx)
