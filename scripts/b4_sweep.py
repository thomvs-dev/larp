"""B4: perplexity (Part A) against energy per token (Part B) for the operating points in
config/system/b4_sweep.yaml. No model is loaded. Writes results/<name>/{summary.json, summary.md} and
figures/b4_energy.png.

    python scripts/b4_sweep.py --name b4_sweep
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cimsim.circuits import ADCSpec  # noqa: E402
from cimsim.mapper import map_weights  # noqa: E402
from cimsim.ppa import token_report  # noqa: E402
from cimsim.system import ModelShape, load_costs  # noqa: E402
from eval.runner import git_hash  # noqa: E402


def hadamard_adds_per_token(shape: ModelShape, block: int = 256) -> int:
    """Fast Walsh-Hadamard adds per token with every CIM-layer input rotated: K log2(block) per distinct
    input (attention input, o_proj input, MLP input, down_proj input), cimsim/models/rotate.py."""
    inputs = [shape.d_model, shape.n_heads * shape.head_dim, shape.d_model, shape.d_ff]
    return shape.n_layers * sum(k * int(math.log2(min(block, k & -k))) for k in inputs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config/system/b4_sweep.yaml"))
    ap.add_argument("--name", default="b4_sweep")
    ap.add_argument("--no-figure", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    L, kv = cfg["seq_len"], cfg["kv"]
    m4 = json.loads((ROOT / cfg["dcim"]["results"]).read_text(encoding="utf-8"))
    adc_lines = yaml.safe_load((ROOT / "config/circuits/adc_survey_lines.yaml").read_text(encoding="utf-8"))
    rows = []
    for model, mc in cfg["models"].items():
        shape = ModelShape.from_yaml(ROOT / mc["shape"])
        layers = shape.linear_layers()
        for pt in mc["points"]:
            res = json.loads((ROOT / "results" / pt["run"] / "result.json").read_text(encoding="utf-8"))
            nrow, bits = pt.get("nrow", 128), pt.get("adc_bits", 6)
            adc_variants = cfg["adc_variants"] if pt["compute"] == "acim" else {"-": None}
            for dram in cfg["drams"]:
                base = load_costs(ROOT, nrow, bits, dram=dram)
                dcim_mac = None
                if pt["compute"] == "dcim":       # M4 at the result VDD (same node), 8 b x 8 b
                    f = (base.vdd / m4["config"]["vdd"]) ** 2
                    e1 = (m4["fitted"]["e_bitop_base_fJ"] + m4["fitted"]["e_bitop_toggle_fJ"] * cfg["dcim"]["toggle_rate"]) * 1e-15
                    dcim_mac = 64 * e1 * f
                for adc_name, line in adc_variants.items():
                    costs = base
                    if line:
                        costs = replace(base, acim=replace(base.acim, adc=ADCSpec.from_dict(adc_lines[line], bits=bits)))
                    full = map_weights(layers, float("inf"), nrow, costs.array_cols)
                    largest = max(m.cells for m in full.layers)
                    for b in cfg["budgets_mbit"]:
                        budget = largest if b == "min" else float(b) * 1e6
                        if budget < largest:
                            continue
                        mp = map_weights(layers, budget, nrow, costs.array_cols)
                        for phase in ("prefill", "decode"):
                            r = token_report(shape, costs, mp, phase, L, kv, compute=pt["compute"], dcim_mac_J=dcim_mac,
                                             hadamard_adds=hadamard_adds_per_token(shape) if pt.get("hadamard") else 0)
                            e = r["energy_J"]
                            rows.append({
                                "model": model, "point": pt["name"], "run": pt["run"], "run_git": str(res.get("git"))[:7],
                                "ppl": res["ppl"], "compute": pt["compute"], "nrow": nrow if pt["compute"] == "acim" else None,
                                "adc_bits": bits if pt["compute"] == "acim" else None, "adc": adc_name, "dram": dram,
                                "budget": b, "budget_bits": budget, "phase": phase, "seq_len": L,
                                "total_J": r["total_J"], "compute_J": e["array"] + e["adc"] + e["shift_add_registers"],
                                "adc_J": e["adc"], "weights_J": e["dram_weights"] + e["array_write_weights"],
                                "lm_head_J": e["lm_head_macs"] + e["lm_head_dram"],
                                "other_J": e["attention"] + e["kv_cache"] + e["digital_units"],
                                "tokens_per_s": r["tokens_per_s"]})
    out = {"git": git_hash()[:7], "config": cfg, "rows": rows}
    od = ROOT / "results" / a.name
    od.mkdir(parents=True, exist_ok=True)
    (od / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    def sel(**kw):
        return [r for r in rows if all(r[k] == v for k, v in kw.items())]

    mj = 1e3
    lines = [f"# B4 sweep (git {out['git']})", "",
             f"Energy per token at sequence length {L}, KV cache in an SRAM buffer, 22 nm / 0.8 V. PPL: first 8,192 WikiText-2 test "
             "tokens (Part A runs). Compute = array + ADC + shift-add of the linear layers (or the digital / DCIM equivalent). "
             "Decode and prefill totals at the minimum on-chip budget (the largest layer only) with the default DRAM.", "",
             "| Model | Point | ADC | PPL | Compute (mJ) | of which ADC | Prefill total (mJ) | Decode total (mJ) | Decode, GDDR5 (mJ) | Decode, 2 Gbit on chip (mJ) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for model in cfg["models"]:
        seen = []
        for r in sel(model=model, dram="default", budget="min", phase="decode"):
            key = (r["point"], r["adc"])
            if key in seen:
                continue
            seen.append(key)
            q = dict(model=model, point=r["point"], adc=r["adc"])
            pre = sel(**q, dram="default", budget="min", phase="prefill")[0]
            gd = sel(**q, dram="gddr5", budget="min", phase="decode")[0]
            big = sel(**q, dram="default", budget=2048, phase="decode")
            lines.append(f"| {model} | {r['point']} | {r['adc']} | {r['ppl']:.3f} | {r['compute_J'] * mj:.3f} | "
                         f"{100 * r['adc_J'] / r['compute_J']:.0f} % | {pre['total_J'] * mj:.3f} | {r['total_J'] * mj:.2f} | "
                         f"{gd['total_J'] * mj:.1f} | {big[0]['total_J'] * mj:.2f} |" if big else "")
    lines += ["", "Notes: all points are within 0.2 % of W8A8 perplexity (the Hadamard point is 1 % better), so the comparison "
              "is on energy. ADC `m1` = the SAR ADC of the calibrated M1 macro moved to 22 nm; `best_sar` = best reported SAR "
              "Walden figure of merit (survey line, not a calibrated macro). DCIM uses the M4 calibration at an input toggle "
              "rate measured on OPT-125M, not on Qwen. Digital INT8 reads each weight from an on-chip SRAM buffer."]
    (od / "summary.md").write_text("\n".join(l for l in lines if l != "") + "\n", encoding="utf-8")
    print("\n".join(lines))

    if not a.no_figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        model = "qwen2.5-1.5b"
        for ax, phase in zip(axes, ("prefill", "decode")):
            pts = sel(model=model, dram="default", budget="min", phase=phase)
            labels = [f"{r['point']}" + (f"\nADC {r['adc']}" if r["adc"] != "-" else "") for r in pts]
            bottom = [0.0] * len(pts)
            for key, lab in (("compute_J", "compute (array, ADC, shift-add)"), ("weights_J", "weights: DRAM read + array write"),
                             ("lm_head_J", "LM head"), ("other_J", "attention, KV, units")):
                vals = [r[key] * mj for r in pts]
                ax.barh(labels, vals, left=bottom, label=lab)
                bottom = [b + v for b, v in zip(bottom, vals)]
            ax.set_xlabel("energy per token (mJ)")
            ax.set_title(f"Qwen2.5-1.5B, {phase}, sequence length {L}")
            ax.invert_yaxis()
            ax.tick_params(axis="y", labelsize=7)
        axes[0].legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        (ROOT / "figures").mkdir(exist_ok=True)
        fig.savefig(ROOT / "figures" / "b4_energy.png", dpi=160)
        print("figure: figures/b4_energy.png")


if __name__ == "__main__":
    main()
