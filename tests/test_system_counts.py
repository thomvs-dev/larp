"""Analytic per-token event counts (cimsim.system.counts) against the totals that Part A runs wrote to
events.json, and the mapper / system model built on them."""

import json
from pathlib import Path

import pytest

from cimsim.mapper import map_weights, tile
from cimsim.ppa import area_report, token_report
from cimsim.system import ModelShape, attention_events, linear_events, load_costs
from cimsim.system.shapes import LinearLayer

ROOT = Path(__file__).parents[1]
Q15 = ModelShape.from_yaml(ROOT / "config/models/qwen2.5-1.5b.yaml")
Q05 = ModelShape.from_yaml(ROOT / "config/models/qwen2.5-0.5b.yaml")


def run(name):
    d = ROOT / "results" / name
    return json.loads((d / "events.json").read_text()), json.loads((d / "result.json").read_text())


def linear_totals(shape, tokens, nrow):
    tot = {"macs": 0, "adc_conversions": 0, "array_cycles": 0, "weight_bits": 0}
    for l in shape.linear_layers():
        for k, v in linear_events(l, nrow).items():
            tot[k] += v
    return {k: (v if k == "weight_bits" else v * tokens) for k, v in tot.items()}


@pytest.mark.parametrize("shape,name,nrow", [(Q15, "a4_qwen15b_stat_adc6cal", 128), (Q05, "a4_qwen05b_stat_adc6cal", 128),
                                             (Q15, "a5r_qwen15b_stat/all_n64_adc5", 64)])
def test_linear_counts_equal_part_a_events(shape, name, nrow):
    ev, res = run(name)
    t = linear_totals(shape, res["windows"] * res["seq_len"], nrow)
    assert t["macs"] == ev["macs"]
    assert t["adc_conversions"] == ev["adc_conversions"]
    assert t["array_cycles"] == ev["array_cycles"]
    assert t["weight_bits"] == ev["weight_bits_written"]


def test_attention_counts_equal_part_a_events():
    ev, res = run("a4_qwen15b_stat_adc6cal_attn_sub8k")
    lin = linear_totals(Q15, res["windows"] * res["seq_len"], 128)
    att = {"macs": 0, "adc_conversions": 0, "array_cycles": 0, "kv_bits_written": 0}
    for keys in range(1, res["seq_len"] + 1):
        for k, v in attention_events(Q15, keys, 128).items():
            att[k] += v * Q15.n_layers * res["windows"]
    assert lin["macs"] + att["macs"] == ev["macs"]
    assert lin["adc_conversions"] + att["adc_conversions"] == ev["adc_conversions"]
    assert lin["array_cycles"] + att["array_cycles"] == ev["array_cycles"]
    assert att["kv_bits_written"] == ev["kv_bits_written"]


def test_tiling_and_residency():
    l = LinearLayer("x", "down", 300, 100)
    assert tile(l, 128, 256, 8) == (3 * 4, 12 * 128 * 256)
    layers = Q15.linear_layers()
    full = map_weights(layers, 2e10, 128, 256)
    assert full.streamed_bits == 0 and full.working_bits == 0 and full.total_bits == Q15.linear_params() * 8
    small = map_weights(layers, 1.2e8, 128, 256)
    assert small.working_bits == max(m.cells for m in small.layers)
    assert small.cells_used <= 1.2e8 and 0 < small.streamed_bits <= small.total_bits
    mid = map_weights(layers, 2e9, 128, 256)
    assert small.streamed_bits > mid.streamed_bits > 0 and mid.cells_used <= 2e9
    with pytest.raises(ValueError):
        map_weights(layers, 1e6, 128, 256)


def test_token_report_structure():
    c = load_costs(ROOT, 128, 6)
    layers = Q15.linear_layers()
    small, full = map_weights(layers, 1.2e8, 128, c.array_cols), map_weights(layers, 2e10, 128, c.array_cols)
    dec = token_report(Q15, c, small, "decode", 512)
    pre = token_report(Q15, c, small, "prefill", 512)
    assert dec["total_J"] == pytest.approx(sum(dec["energy_J"].values()))
    assert sum(dec["share"].values()) == pytest.approx(1.0)
    # compute per token is the same in both phases; streaming is shared by the prompt in prefill
    for k in ("array", "adc", "shift_add_registers"):
        assert dec["energy_J"][k] == pytest.approx(pre["energy_J"][k])
    assert pre["energy_J"]["dram_weights"] == pytest.approx(dec["energy_J"]["dram_weights"] / 512)
    assert dec["energy_J"]["dram_weights"] == pytest.approx(small.streamed_bits * c.dram.e_bit)
    assert token_report(Q15, c, full, "decode", 512)["energy_J"]["dram_weights"] == 0.0
    # event counts per token equal the Part A totals per processed token
    ev, res = run("a4_qwen15b_stat_adc6cal")
    assert dec["events"]["adc_conversions"] * res["windows"] * res["seq_len"] == ev["adc_conversions"]
    # attention and KV cost grow with position; the linear part does not
    far = token_report(Q15, c, small, "decode", 2048)
    assert far["energy_J"]["attention"] > dec["energy_J"]["attention"]
    assert far["energy_J"]["kv_cache"] > dec["energy_J"]["kv_cache"]
    arr = token_report(Q15, c, small, "decode", 2048, kv="array")
    assert arr["events"]["adc_conversions"] > far["events"]["adc_conversions"]
    assert area_report(Q15, c, full)["array_cells"] == full.total_bits
    with pytest.raises(ValueError):
        token_report(Q15, c, small, "train", 8)
