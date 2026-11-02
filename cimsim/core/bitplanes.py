"""Bit decomposition of quantized operands.

Both operands use two's complement (equivalent to ASiM's sign-bit
decomposition). Weight planes are binary and stored in separate columns.
Inputs are applied in "cycles": one bit per cycle (bit-serial) or groups
of `act_enc` bits per cycle (bit-parallel, multi-level DAC input).

Cycle ordering and indices follow ASiM so the hybrid rule
(i + j <= B_w + n_cycles - hybrid_levels - 2) means the same thing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Cycle:
    index: int      # ASiM cycle index j (0 = least significant)
    shift: int      # power-of-two weight of this cycle's digit
    sign: int       # +1, or -1 for the activation sign cycle
    levels: int     # max digit value L_c = 2^(bits in group) - 1
    bit_lo: int     # first bit of the group (LSB = 0)
    n_bits: int     # bits in the group


def to_twos_complement_bits(q: torch.Tensor, bits: int) -> torch.Tensor:
    """q: integer-valued float tensor in [-(2^(bits-1)-1), 2^(bits-1)-1].
    Returns bit tensor [bits, *q.shape] (index 0 = LSB) with values 0/1."""
    u = torch.where(q < 0, q + 2.0 ** bits, q).to(torch.int64)
    shifts = torch.arange(bits, device=q.device).view(-1, *([1] * q.dim()))
    return ((u.unsqueeze(0) >> shifts) & 1).to(torch.float32)


def to_unsigned_bits(q: torch.Tensor, bits: int) -> torch.Tensor:
    u = q.to(torch.int64)
    shifts = torch.arange(bits, device=q.device).view(-1, *([1] * q.dim()))
    return ((u.unsqueeze(0) >> shifts) & 1).to(torch.float32)


def weight_planes(q_w: torch.Tensor, w_bits: int, signed: bool = True) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """q_w: [N, K]. Returns planes [w_bits, N, K] and per-plane (shift, sign).
    Plane index i is the bit index (0 = LSB). Signed (two's complement): the
    MSB plane has sign -1. Unsigned (asymmetric weights with zero point): all +1."""
    if signed:
        planes = to_twos_complement_bits(q_w, w_bits)
        meta = [(i, -1 if i == w_bits - 1 else 1) for i in range(w_bits)]
    else:
        planes = to_unsigned_bits(q_w, w_bits)
        meta = [(i, 1) for i in range(w_bits)]
    return planes, meta


def input_cycles(q_x: torch.Tensor, x_bits: int, signed: bool, act_enc: int | None) -> list[Cycle]:
    """Cycle schedule for the input (does not depend on data)."""
    cycles: list[Cycle] = []
    if act_enc is None:  # bit-serial
        for j in range(x_bits):
            neg = signed and j == x_bits - 1
            cycles.append(Cycle(j, j, -1 if neg else 1, 1, j, 1))
        return cycles

    mag_bits = x_bits - 1 if signed else x_bits
    n_groups = math.ceil(mag_bits / act_enc)
    for j in range(n_groups):
        lo = j * act_enc
        nb = min(act_enc, mag_bits - lo)
        cycles.append(Cycle(j, lo, 1, 2 ** nb - 1, lo, nb))
    if signed:
        j = n_groups
        cycles.append(Cycle(j, x_bits - 1, -1, 1, x_bits - 1, 1))
    return cycles


def input_digits(q_x: torch.Tensor, x_bits: int, signed: bool, cycles: list[Cycle]) -> torch.Tensor:
    """q_x: [T, K]. Returns digits [n_cycles, T, K] (float32, small integers)."""
    bits = to_twos_complement_bits(q_x, x_bits) if signed else to_unsigned_bits(q_x, x_bits)
    out = []
    for c in cycles:
        d = torch.zeros_like(bits[0])
        for k in range(c.n_bits):
            d = d + bits[c.bit_lo + k] * (2.0 ** k)
        out.append(d)
    return torch.stack(out, dim=0)


def recompose_input(digits: torch.Tensor, cycles: list[Cycle]) -> torch.Tensor:
    """Inverse of input_digits, for tests."""
    y = torch.zeros_like(digits[0], dtype=torch.float64)
    for d, c in zip(digits, cycles):
        y = y + c.sign * (2.0 ** c.shift) * d.double()
    return y


def recompose_weight(planes: torch.Tensor, meta: list[tuple[int, int]]) -> torch.Tensor:
    y = torch.zeros_like(planes[0], dtype=torch.float64)
    for p, (shift, sign) in zip(planes, meta):
        y = y + sign * (2.0 ** shift) * p.double()
    return y
