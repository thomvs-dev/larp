"""Model shapes for the analytic system model (Part B, B2 / B3).

Only decoder blocks of the Llama / Qwen2 kind (gated MLP) and the OPT / GPT-2 kind (plain MLP) are
described: per block q, k, v, o projections and the MLP matrices. The LM head is kept digital, as in
Part A (docs/assumptions.md #9).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class LinearLayer:
    name: str          # e.g. "layers.3.mlp.down_proj"
    role: str          # qkv / o / up_gate / down (cimsim.config.ROLE_PATTERNS)
    k: int             # inputs (rows of the array)
    n: int             # outputs

    @property
    def params(self) -> int:
        return self.k * self.n


@dataclass(frozen=True)
class ModelShape:
    name: str
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    d_ff: int
    vocab: int
    mlp: str = "swiglu"        # "swiglu": gate, up, down; "mlp": up, down
    norm: str = "rmsnorm"
    act: str = "silu"
    rope: bool = True

    def __post_init__(self):
        if self.mlp not in ("swiglu", "mlp"):
            raise ValueError("mlp must be 'swiglu' or 'mlp'")
        if min(self.n_layers, self.d_model, self.n_heads, self.n_kv_heads, self.head_dim, self.d_ff, self.vocab) < 1:
            raise ValueError("shape fields must be >= 1")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelShape":
        d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise KeyError(f"unknown model shape fields: {sorted(unknown)}")
        return cls(**d)

    def block_layers(self) -> list[tuple[str, str, int, int]]:
        d, f = self.d_model, self.d_ff
        out = [("self_attn.q_proj", "qkv", d, self.n_heads * self.head_dim),
               ("self_attn.k_proj", "qkv", d, self.n_kv_heads * self.head_dim),
               ("self_attn.v_proj", "qkv", d, self.n_kv_heads * self.head_dim),
               ("self_attn.o_proj", "o", self.n_heads * self.head_dim, d)]
        if self.mlp == "swiglu":
            out += [("mlp.gate_proj", "up_gate", d, f), ("mlp.up_proj", "up_gate", d, f)]
        else:
            out += [("mlp.up_proj", "up_gate", d, f)]
        out += [("mlp.down_proj", "down", f, d)]
        return out

    def linear_layers(self) -> list[LinearLayer]:
        return [LinearLayer(f"layers.{i}.{n}", r, k, o) for i in range(self.n_layers) for n, r, k, o in self.block_layers()]

    def linear_params(self) -> int:
        return sum(l.params for l in self.linear_layers())

    def head_params(self) -> int:
        return self.d_model * self.vocab
