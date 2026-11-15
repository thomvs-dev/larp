"""B0 component table: cited per-operation costs at their source node and at the Part B result node.

Reads config/circuits/*.yaml only. Writes results/<name>/summary.{json,md}. No model run, no fitted value:
the analog array parameters (e_wl_cell, e_bl_cell) are fitted per macro in B1 and are not in this table.

    python scripts/b0_components.py --name b0_components
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml

from cimsim.circuits import UNIT_OPS, ADCSpec, DigitalSpec, ScalingRule

ROOT = Path(__file__).resolve().parents[1]


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="b0_components")
    ap.add_argument("--unit-bits", type=int, default=16, help="word width of the non-linear units")
    args = ap.parse_args()

    cfg_dir = ROOT / "config" / "circuits"
    scaling = yaml.safe_load((cfg_dir / "scaling.yaml").read_text())
    rule = ScalingRule.from_dict(scaling["rule"])
    node, vdd = scaling["target"]["node_nm"], scaling["target"]["vdd"]
    src = DigitalSpec.from_dict(yaml.safe_load((cfg_dir / "digital_45nm.yaml").read_text()))
    dst = src.scaled(node, vdd, rule)
    adc_cfg = yaml.safe_load((cfg_dir / "adc_survey_lines.yaml").read_text())

    def digital_rows(s: DigitalSpec) -> dict:
        b = args.unit_bits
        rows = {
            "add_8b": s.add(8), "add_16b": s.add(16), "add_24b": s.add(24),
            "mul_8x8b": s.mul(8, 8), "mul_16x16b": s.mul(16),
            "int8_mac_acc24": s.mac(8, 8, 24),
            "register_bit": s.reg(1), "sram_buffer_bit": s.sram_read(1),
        }
        for unit in sorted(UNIT_OPS):
            rows[f"{unit}_per_element_{b}b"] = s.unit_energy(unit, 1, vectors=0, bits=b)
        return rows

    out = {
        "git": git_hash(),
        "config": {"scaling": scaling, "digital_source": {"node_nm": src.node_nm, "vdd": src.vdd},
                   "unit_bits": args.unit_bits},
        "energy_factor_source_to_target": rule.energy_factor(src.node_nm, src.vdd, node, vdd),
        "digital_J": {"source": digital_rows(src), "target": digital_rows(dst)},
        # ADC reference lines are node-independent survey envelopes (assumptions B-2): not scaled.
        "adc_J_per_conversion": {
            name: {str(b): ADCSpec.from_dict(d, bits=b).energy_per_conversion() for b in range(4, 9)}
            for name, d in adc_cfg.items()
        },
    }
    res = ROOT / "results" / args.name
    res.mkdir(parents=True, exist_ok=True)
    (res / "summary.json").write_text(json.dumps(out, indent=2))

    fj = 1e15
    lines = [f"# B0 component costs (git {out['git']})", "",
             f"Digital operations: published at {src.node_nm:g} nm / {src.vdd:g} V, scaled to {node:g} nm / {vdd:g} V "
             f"with E ~ node^{rule.energy_node_exp:g} · VDD² (factor {out['energy_factor_source_to_target']:.3f}). "
             "Sources: `docs/assumptions.md` B-11 … B-15.", "",
             f"| Operation | {src.node_nm:g} nm (fJ) | {node:g} nm (fJ) |", "|---|---|---|"]
    for k in out["digital_J"]["source"]:
        lines.append(f"| {k} | {out['digital_J']['source'][k] * fj:.2f} | {out['digital_J']['target'][k] * fj:.2f} |")
    lines += ["", "ADC energy per conversion on the survey reference lines (not scaled, not calibrated; B-1, B-2), fJ:", "",
              "| Line | 4 b | 5 b | 6 b | 7 b | 8 b |", "|---|---|---|---|---|---|"]
    for name, row in out["adc_J_per_conversion"].items():
        lines.append(f"| {name} | " + " | ".join(f"{row[str(b)] * fj:.1f}" for b in range(4, 9)) + " |")
    lines += ["", "Not in this table: analog array energies per cell (wordline, bit line), fitted per macro in B1.",
              "These are component inputs, not results: no energy per token is reported before GATE B1."]
    (res / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
