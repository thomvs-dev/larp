"""Part B: energy / latency / area per token for one Part A run, from its events.json and config.

Reads results/<run>/{events.json, result.json, config.yaml}. Checks that the counts derived from the
model shape reproduce the run's event totals, then costs one token in prefill and in decode for each
on-chip array budget, sequence length and KV-cache placement. Writes results/<name>/{ppa_report.json,
summary.md}. No model is loaded.

    python scripts/run_ppa.py --run a4_qwen15b_stat_adc6cal --shape config/models/qwen2.5-1.5b.yaml --name b3_ppa_qwen15b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cimsim.mapper import map_weights  # noqa: E402
from cimsim.ppa import area_report, token_report  # noqa: E402
from cimsim.system import ModelShape, attention_events, linear_events, load_costs  # noqa: E402
from eval.runner import git_hash  # noqa: E402


def check_counts(shape: ModelShape, cfg: dict, events: dict, result: dict) -> dict:
    """Derived totals for the run's processed tokens against its events.json."""
    d = cfg["default"]
    tokens = result["windows"] * result["seq_len"]
    tot = {"macs": 0, "adc_conversions": 0, "array_cycles": 0, "weight_bits_written": 0, "kv_bits_written": 0}
    for layer in shape.linear_layers():
        ev = linear_events(layer, d["nrow"], d["w_bits"], d["x_bits"])
        for k in ("macs", "adc_conversions", "array_cycles"):
            tot[k] += ev[k] * tokens
        tot["weight_bits_written"] += ev["weight_bits"]
    if cfg.get("attention"):
        for keys in range(1, result["seq_len"] + 1):
            ev = attention_events(shape, keys, d["nrow"], d["w_bits"], d["x_bits"])
            for k in ("macs", "adc_conversions", "array_cycles", "kv_bits_written"):
                tot[k] += ev[k] * shape.n_layers * result["windows"]
    rows = {k: {"derived": v, "events_json": events.get(k, 0), "equal": v == events.get(k, 0)} for k, v in tot.items()}
    return {"tokens_processed": tokens, "all_equal": all(r["equal"] for r in rows.values()), "fields": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="folder under results/ of a Part A run")
    ap.add_argument("--shape", required=True, help="config/models/*.yaml")
    ap.add_argument("--name", required=True)
    ap.add_argument("--budgets-mbit", default="min,512,2048,all",
                    help="on-chip CIM array budgets in Mbit; 'min' = the largest layer only, 'all' = every weight resident")
    ap.add_argument("--seq-lens", default="128,512,2048")
    ap.add_argument("--kv", default="buffer,array")
    ap.add_argument("--dram", default="default", help="entry of config/system/dram.yaml")
    a = ap.parse_args()

    run = ROOT / "results" / a.run
    events = json.loads((run / "events.json").read_text(encoding="utf-8"))
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    d = cfg["default"]
    if d.get("macro", "acim") != "acim":
        raise SystemExit("run_ppa.py costs all-analog runs only (hybrid policies: B4)")
    shape = ModelShape.from_yaml(ROOT / a.shape)
    check = check_counts(shape, cfg, events, result)
    costs = load_costs(ROOT, d["nrow"], d["adc_bits"], dram=a.dram)
    layers = shape.linear_layers()
    full = map_weights(layers, float("inf"), d["nrow"], costs.array_cols, d["w_bits"])
    largest = max(m.cells for m in full.layers)
    budgets = []
    for b in a.budgets_mbit.split(","):
        bits = largest if b == "min" else full.cells_used if b == "all" else float(b) * 1e6
        if bits >= largest:
            budgets.append((b, bits))
    rows = []
    for label, bits in budgets:
        mp = map_weights(layers, bits, d["nrow"], costs.array_cols, d["w_bits"])
        ar = area_report(shape, costs, mp)
        for kv in a.kv.split(","):
            for L in [int(x) for x in a.seq_lens.split(",")]:
                for phase in ("prefill", "decode"):
                    r = token_report(shape, costs, mp, phase, L, kv, d["w_bits"], d["x_bits"])
                    r.update({"budget": label, "budget_bits": bits, "resident_fraction": 1 - mp.streamed_bits / mp.total_bits,
                              "area": ar})
                    rows.append(r)
    conv = costs.acim.conversion_breakdown(0.0, costs.input_bit_density)
    out = {"git": git_hash()[:7], "run": a.run, "run_git": result.get("git"), "model": result.get("model"), "ppl": result.get("ppl"),
           "config": cfg, "shape": a.shape, "dram": a.dram, "node_nm": costs.node_nm, "vdd": costs.vdd,
           "count_check": check, "energy_per_conversion_J": conv, "write_J_per_bit": costs.write_e_bit,
           "dram_J_per_bit": costs.dram.e_bit, "reports": rows}
    od = ROOT / "results" / a.name
    od.mkdir(parents=True, exist_ok=True)
    (od / "ppa_report.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    mj = 1e3
    lines = [f"# Part B cost report, {result.get('model')} (git {out['git']})", "",
             f"Run `{a.run}` (git {str(result.get('git'))[:7]}, PPL {result.get('ppl'):.3f}), nrow {d['nrow']}, ADC {d['adc_bits']} b, "
             f"{costs.node_nm:g} nm / {costs.vdd:g} V, DRAM `{a.dram}` {costs.dram.e_bit * 1e12:.2f} pJ/bit.",
             f"Count check: derived totals {'equal' if check['all_equal'] else 'DIFFER FROM'} `events.json` "
             f"({check['tokens_processed']} tokens processed).",
             f"One conversion: array {(conv['bitline'] + conv['cell_fixed']) * 1e15:.0f} fJ, ADC {conv['adc'] * 1e15:.0f} fJ, "
             f"shift-add + register {(conv['shift_add'] + conv['out_register']) * 1e15:.0f} fJ. "
             f"Array write {costs.write_e_bit * 1e15:.0f} fJ/bit.", "",
             "Energy per token, mJ. Columns: array + ADC + shift-add (compute), DRAM read of streamed weights, array write of "
             "streamed weights, attention + KV cache, non-linear units, LM head (digital MACs + its weights from DRAM).", "",
             "| Budget (Mbit) | Resident | Cell area (mm²) | KV | Phase | Seq | Total | Compute | DRAM weights | Array write | Attn + KV | Units | LM head | Tokens/s bound |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        e = r["energy_J"]
        lines.append(
            f"| {r['budget_bits'] / 1e6:.0f} | {100 * r['resident_fraction']:.0f} % | {r['area']['cell_area_mm2']:.0f} | {r['kv']} | "
            f"{r['phase']} | {r['seq_len']} | {r['total_J'] * mj:.3f} | {(e['array'] + e['adc'] + e['shift_add_registers']) * mj:.3f} | "
            f"{e['dram_weights'] * mj:.3f} | {e['array_write_weights'] * mj:.3f} | {(e['attention'] + e['kv_cache']) * mj:.4f} | "
            f"{e['digital_units'] * mj:.4f} | {(e['lm_head_macs'] + e['lm_head_dram']) * mj:.3f} | {r['tokens_per_s']:.1f} |")
    lines += ["", "Limits: cell area only (no ADC / periphery area in the M1 anchor); latency is a bound from DRAM bandwidth "
              "plus one array evaluation per input cycle and layer (digital units and scheduling not timed); analog costs are "
              "the M1 macro moved from 65 nm to the result node by the scaling rule (docs/assumptions.md B-11, B-25)."]
    (od / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
