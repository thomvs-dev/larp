"""Event counts per token, derived from model shapes (Part B, B3).

The formulas are the ones the executors count with (cimsim/core/layers.py, cimsim/core/attention.py);
tests/test_system_counts.py checks them against the totals in committed Part A events.json files.
A linear layer costs the same for every token, in prefill and in decode. Attention depends on the
number of keys a query attends to (its position).
"""

from __future__ import annotations

import math

from .shapes import LinearLayer, ModelShape


def n_cycles(x_bits: int, act_enc: int | None = None) -> int:
    """Input cycles of signed x_bits activations: bit-serial = x_bits; bit-parallel groups of act_enc
    magnitude bits plus the sign cycle."""
    if act_enc is None:
        return x_bits
    return 1 + math.ceil((x_bits - 1) / act_enc)


def linear_events(layer: LinearLayer, nrow: int, w_bits: int = 8, x_bits: int = 8, act_enc: int | None = None) -> dict:
    """Events of one token through one linear layer on an analog array."""
    chunks = math.ceil(layer.k / nrow)
    c = n_cycles(x_bits, act_enc)
    return {"macs": layer.k * layer.n,
            "array_cycles": chunks * c,
            "adc_conversions": chunks * c * w_bits * layer.n,
            "weight_bits": layer.k * layer.n * w_bits}


def attention_events(shape: ModelShape, keys: int, nrow: int, w_bits: int = 8, x_bits: int = 8) -> dict:
    """Events of one query token attending to `keys` positions, for one block with both attention
    matmuls on analog arrays (Q.K^T: keys in columns, contraction head_dim; P.V: values in rows,
    contraction over the keys). KV bits written are per new token."""
    h, d, c = shape.n_heads, shape.head_dim, n_cycles(x_bits)
    qk_chunks = math.ceil(d / nrow)
    sv_chunks = math.ceil(keys / nrow)
    return {"macs": h * (keys * d + keys * d),
            "array_cycles": h * (qk_chunks * c + sv_chunks * c),
            "adc_conversions": h * (qk_chunks * keys * c * w_bits + sv_chunks * c * w_bits * d),
            "kv_bits_written": 2 * shape.n_kv_heads * d * w_bits}


def unit_elements(shape: ModelShape, keys: int) -> dict:
    """Elements each non-linear digital unit processes for one token in one block (keys = positions
    attended to), as {unit: (elements, vectors)} for DigitalSpec.unit_energy."""
    d, f, h, hd = shape.d_model, shape.d_ff, shape.n_heads, shape.head_dim
    out = {shape.norm: (2 * d, 2), shape.act: (f, 0), "residual": (2 * d, 0), "softmax": (h * keys, h)}
    if shape.mlp == "swiglu":
        out["gate_mul"] = (f, 0)
    if shape.rope:
        out["rope"] = ((h + shape.n_kv_heads) * hd, 0)
    return out
