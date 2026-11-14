"""Hadamard rotation of CIM-layer inputs (A6, outlier mitigation).

For a layer y = x W^T with input width K, an orthogonal matrix Q gives y = (x Q)(W Q)^T. The weight
W Q is computed once, offline, before the CIM layers are built, so it is quantized and bit-sliced
like any other weight. The input x Q is computed online in digital logic in front of the array.

Q is block-diagonal with normalised Sylvester Hadamard blocks of size B = the largest power of two
that divides K (capped by `block`): OPT-125M 768 -> 256, 3072 -> 1024, Qwen2.5-1.5B 1536 -> 512,
8960 -> 256. Q is symmetric and Q Q = I. A fast Walsh-Hadamard transform needs K * log2(B)
additions per token; these are counted as `hadamard_adds` (once per distinct input: k/v share the
q input and gate shares the up input, so only q_proj / up_proj / c_attn / fc1 count it).

Config (top-level YAML key, read by eval.runner.swap_from_config):
    hadamard:
      roles: [down]          # any of qkv, o, up_gate, down
      block: null            # optional cap on the block size
"""

from __future__ import annotations

import fnmatch
import math

import torch
from torch import nn

from cimsim.config import ROLE_PATTERNS

SHARED_INPUT = ("*.k_proj", "*.v_proj", "*.gate_proj")   # same input as q_proj / up_proj


def hadamard(n: int, dtype=torch.float64) -> torch.Tensor:
    """Normalised Sylvester Hadamard matrix, n a power of two."""
    if n < 1 or n & (n - 1):
        raise ValueError(f"Hadamard size must be a power of two, got {n}")
    h = torch.ones(1, 1, dtype=dtype)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(n)


def block_size(k: int, cap: int | None = None) -> int:
    b = k & -k                      # largest power of two dividing k
    if cap is not None:
        b = min(b, 1 << int(math.floor(math.log2(cap))))
    return b


def rotate_last(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """x[..., K] -> x Q with Q = blockdiag(h, ..., h)."""
    b = h.shape[0]
    shp = x.shape
    return (x.reshape(*shp[:-1], shp[-1] // b, b) @ h.to(x.dtype)).reshape(shp)


def _role(name: str, roles) -> bool:
    return any(fnmatch.fnmatch(name, p) for r in roles for p in ROLE_PATTERNS[r])


def rotate_weights(model: nn.Module, roles, cap: int | None = None) -> dict[str, int]:
    """Replace W by W Q in every Linear / GPT-2 Conv1D of the given roles (before swapping).
    Returns {layer name: block size}."""
    for r in roles:
        if r not in ("qkv", "o", "up_gate", "down"):
            raise ValueError(f"hadamard role {r!r} not supported")
    done = {}
    for name, m in model.named_modules():
        if not _role(name, roles):
            continue
        conv1d = type(m).__name__ == "Conv1D"
        if not (isinstance(m, nn.Linear) or conv1d):
            continue
        w = m.weight.data
        k = w.shape[0] if conv1d else w.shape[1]
        b = block_size(k, cap)
        h = hadamard(b)
        with torch.no_grad():
            if conv1d:              # weight [K, N]: Q W
                w64 = rotate_last(w.double().t(), h).t()
            else:                   # weight [N, K]: W Q
                w64 = rotate_last(w.double(), h)
            m.weight.data.copy_(w64.to(w.dtype))
        done[name] = b
    return done


def add_input_rotation(model: nn.Module, blocks: dict[str, int], counter=None) -> list:
    """Forward pre-hooks that rotate the input of each named (swapped) layer. Returns the handles."""
    mods = dict(model.named_modules())
    handles = []
    for name, b in blocks.items():
        h = hadamard(b, torch.float32)
        count = counter is not None and not any(fnmatch.fnmatch(name, p) for p in SHARED_INPUT)
        log2b = int(math.log2(b))

        def pre(mod, args, h=h, name=name, count=count, log2b=log2b):
            x = args[0]
            if count:
                counter.add(name, hadamard_adds=x.numel() * log2b)
            return (rotate_last(x, h.to(x.device)),) + tuple(args[1:])
        handles.append(mods[name].register_forward_pre_hook(pre))
    return handles
