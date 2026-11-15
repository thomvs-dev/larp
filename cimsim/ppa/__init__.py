"""Event counts x per-event costs -> energy and latency per token (Part B, B3).

`token_report` gives one token's energy by component for a phase:
- decode: the token at position `seq_len` (attends to seq_len keys); every streamed layer is loaded once;
  the LM head runs.
- prefill: the average token of a prompt of `seq_len` tokens processed in one pass; streamed layers are
  loaded once per pass (shared by seq_len tokens); the LM head runs once per prompt.

KV cache (`kv`): "buffer" = keys / values in an on-chip SRAM buffer, attention matmuls on the digital
MAC unit, the buffer is read once per attended key; "array" = keys / values written into CIM arrays,
attention matmuls analog (events of cimsim.core.attention).
"""

from __future__ import annotations

from cimsim.mapper import Mapping
from cimsim.system.costs import SystemCosts
from cimsim.system.counts import attention_events, linear_events, unit_elements
from cimsim.system.shapes import ModelShape

PHASES = ("prefill", "decode")
KV = ("buffer", "array")


def _mean_keys(seq_len: int) -> float:
    return (seq_len + 1) / 2.0


def token_report(shape: ModelShape, costs: SystemCosts, mapping: Mapping, phase: str, seq_len: int,
                 kv: str = "buffer", w_bits: int = 8, x_bits: int = 8) -> dict:
    if phase not in PHASES or kv not in KV:
        raise ValueError(f"phase in {PHASES}, kv in {KV}")
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1")
    a, dg, nrow = costs.acim, costs.digital, costs.acim.nrow
    keys = seq_len if phase == "decode" else _mean_keys(seq_len)
    share = 1.0 if phase == "decode" else 1.0 / seq_len          # per-pass costs carried by one token
    e: dict = {}

    # linear layers on the analog arrays (the same for every token)
    conv = cyc = macs = 0
    for m in mapping.layers:
        ev = linear_events(m.layer, nrow, w_bits, x_bits)
        conv += ev["adc_conversions"]
        cyc += ev["array_cycles"]
        macs += ev["macs"]
    lin = a.energy(conv, cyc, wl_pulses=0.0, bl_activity=costs.input_bit_density)
    e["array"] = lin["bitline"] + lin["cell_fixed"] + lin["wordline"] + lin["column_fixed"]
    e["adc"] = lin["adc"]
    e["shift_add_registers"] = lin["shift_add"] + lin["out_register"] + lin["in_register"]

    # weight streaming: DRAM read + array write of every non-resident layer
    stream_bits = mapping.streamed_bits * share
    e["dram_weights"] = costs.dram.energy(stream_bits)
    e["array_write_weights"] = stream_bits * costs.write_e_bit

    # attention and KV cache
    kvb = 2 * shape.n_kv_heads * shape.head_dim * costs.kv_bits * shape.n_layers       # bits per new token
    att_macs = shape.n_layers * shape.n_heads * 2 * keys * shape.head_dim
    if kv == "buffer":
        xb, wb, acc = costs.head_macs_bits
        e["attention"] = att_macs * dg.mac(xb, wb, acc)
        e["kv_cache"] = dg.sram_read(kvb * keys) + dg.sram_read(kvb)                   # read all attended, write the new
    else:
        # attention_events needs an integer key count; prefill uses the mean position
        k_int = max(1, round(keys))
        ev = attention_events(shape, k_int, nrow, w_bits, x_bits)
        att = a.energy(ev["adc_conversions"] * shape.n_layers, ev["array_cycles"] * shape.n_layers,
                       wl_pulses=0.0, bl_activity=costs.input_bit_density)
        e["attention"] = att["total"]
        e["kv_cache"] = kvb * costs.write_e_bit                                         # write the new token's K, V
        conv += ev["adc_conversions"] * shape.n_layers
    # non-linear units
    units = 0.0
    k_units = max(1, round(keys))
    for unit, (elements, vectors) in unit_elements(shape, k_units).items():
        units += dg.unit_energy(unit, elements, vectors, bits=costs.digital_bits)
    e["digital_units"] = units * shape.n_layers + dg.unit_energy(shape.norm, shape.d_model, 1, costs.digital_bits)

    # LM head: digital MACs + its weights from DRAM; once per token in decode, once per prompt in prefill
    xb, wb, acc = costs.head_macs_bits
    e["lm_head_macs"] = shape.head_params() * dg.mac(xb, wb, acc) * share
    e["lm_head_dram"] = costs.dram.energy(shape.head_params() * wb) * share

    total = sum(e.values())
    # latency: weight / head stream at DRAM bandwidth, then one array evaluation per input cycle and layer
    t_stream = costs.dram.time(stream_bits + shape.head_params() * wb * share)
    t_compute = len(mapping.layers) * x_bits * costs.t_eval
    return {"phase": phase, "seq_len": seq_len, "kv": kv, "energy_J": e, "total_J": total,
            "share": {k: v / total for k, v in e.items()},
            "events": {"macs": macs, "adc_conversions": conv, "array_cycles": cyc, "attention_macs": att_macs,
                       "streamed_weight_bits": stream_bits},
            "latency_s": {"dram_stream": t_stream, "array_compute": t_compute, "total": t_stream + t_compute},
            "tokens_per_s": 1.0 / (t_stream + t_compute)}


def area_report(shape: ModelShape, costs: SystemCosts, mapping: Mapping) -> dict:
    """mm^2 of the CIM arrays on chip (cells + one ADC per column are not separated in the M1 anchor:
    the paper gives the cell area only, so this is the cell area)."""
    return {"array_cells": mapping.cells_used, "arrays": mapping.arrays,
            "cell_area_mm2": mapping.cells_used * costs.acim.area_cell * 1e6,
            "note": "bit-cell area only; ADC and periphery area is not available from the M1 anchor"}
