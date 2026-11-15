"""GATE B1 summary: worst held-out error per macro from results/b1_m*/results.json, and the three macros'
energy per 4b x 4b MAC moved to the Part B result node with the scaling rule (docs/assumptions.md B-11).

python validation/b1_gate.py [--out results/b1_gate]
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml

from cimsim.circuits import ScalingRule

ROOT = Path(__file__).resolve().parents[1]
MACROS = {"M1": "b1_m1_jia", "M2": "b1_m2_sinangil", "M4": "b1_m4_chih"}


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/b1_gate"))
    ap.add_argument("--limit", type=float, default=30.0, help="gate limit, % (CLAUDE.md: 20-30 %)")
    a = ap.parse_args()
    res = {k: json.loads((ROOT / "results" / v / "results.json").read_text(encoding="utf-8")) for k, v in MACROS.items()}
    sc = yaml.safe_load((ROOT / "config/circuits/scaling.yaml").read_text())
    rule = ScalingRule.from_dict(sc["rule"])
    node, vdd = sc["target"]["node_nm"], sc["target"]["vdd"]

    gate = {}
    for k, r in res.items():
        held = [c for c in r["checks"] if c["kind"].startswith("held-out")]
        worst = max(held, key=lambda c: abs(c["err_pct"]))
        over = [c for c in held if abs(c["err_pct"]) > a.limit]
        gate[k] = {"macro": r["macro"], "git": r["git"], "held_out_points": len(held),
                   "worst": {"check": worst["check"], "err_pct": worst["err_pct"]},
                   "over_limit": [{"check": c["check"], "err_pct": c["err_pct"]} for c in over]}

    # energy per 4b x 4b MAC at each macro's own node / VDD, then at the result node
    m1, m2, m4 = res["M1"], res["M2"], res["M4"]
    native = {
        "M1": {"e": 16 * m1["fitted"]["energy_per_1b_mac_fJ"], "node": m1["config"]["node_nm"], "vdd": m1["config"]["vdd"],
               "note": "16 1-b MACs, array + 8-b SAR ADC, activity 0.5"},
        "M2": {"e": m2["fitted"]["energy_per_4b4b_mac_avg_fJ"], "node": m2["config"]["node_nm"], "vdd": m2["config"]["vdd"],
               "note": "average point of Fig. 13(a), 4-b output"},
        "M4": {"e": m4["fitted"]["mac_energy_4b4b_fJ_toggle018"], "node": m4["config"]["node_nm"], "vdd": m4["config"]["vdd"],
               "note": "18 % input toggle, exact digital"},
    }
    for v in native.values():
        v["factor"] = rule.energy_factor(v["node"], v["vdd"], node, vdd)
        v["e_target"] = v["e"] * v["factor"]
    vals = [v["e_target"] for v in native.values()]
    out = {"git": git_hash(), "limit_pct": a.limit, "gate": gate, "target": {"node_nm": node, "vdd": vdd},
           "mac_4b4b_fJ": native, "spread_max_over_min": max(vals) / min(vals)}
    d = Path(a.out)
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    lines = [f"# GATE B1 summary (git {out['git']}, limit {a.limit:g} %)", "",
             "| Macro | Held-out points | Worst held-out error | Points over the limit |", "|---|---|---|---|"]
    for k, g in gate.items():
        over = "; ".join(f"{c['check']} {c['err_pct']:+.1f} %" for c in g["over_limit"]) or "none"
        lines.append(f"| {k} {g['macro']} | {g['held_out_points']} | {g['worst']['err_pct']:+.1f} % ({g['worst']['check']}) | {over} |")
    lines += ["", f"Energy per 4b x 4b MAC, each macro's calibrated value moved to {node:g} nm / {vdd:g} V with "
              f"E ~ node^{rule.energy_node_exp:g} x VDD^2:", "",
              "| Macro | Own node / VDD | fJ per MAC there | Factor | fJ per MAC at the result node | Condition |",
              "|---|---|---|---|---|---|"]
    for k, v in native.items():
        lines.append(f"| {k} | {v['node']:g} nm / {v['vdd']:g} V | {v['e']:.1f} | {v['factor']:.3f} | {v['e_target']:.1f} | {v['note']} |")
    lines += ["", f"Spread (largest / smallest): {out['spread_max_over_min']:.2f}x. The three macros differ in architecture, "
              "output precision and data pattern, so this is a plausibility check of the scaling rule, not a test of it."]
    (d / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
