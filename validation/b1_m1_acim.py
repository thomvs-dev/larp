"""GATE B1, macro M1 (Jia JSSC 2020, 65 nm, charge-domain, 8-b SAR): fit the analog array model to the
1.2 V column energies of Table I, then compare with the 0.85 V numbers and the abstract's 1b-TOPS/W.

Mapping onto cimsim.circuits.AnalogArraySpec (docs/assumptions.md B-21):
- one conversion = one column of N = 2304 cells evaluated for one input bit = 2304 1-b MACs;
- array energy split 66 % data-dependent / 34 % data-independent (Sec. III-B); the data-dependent part is
  charged at `calib_activity` (assumed input density of the Table I measurement);
- ADC: SAR with the paper's 1.2 fF unit capacitor and conventional switching; the comparator + logic energy
  per bit cycle is the fitted remainder of the measured ADC energy;
- 1b-TOPS/W = 2 ops per 1-b MAC over array + ADC energy (this reproduces the abstract's 192 from Table I);
- other supplies by VDD^2 (ScalingRule).

python validation/b1_m1_acim.py [--out results/b1_m1_jia]
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml

from cimsim.circuits import AnalogArraySpec
from cimsim.circuits.macros import m1_spec

ROOT = Path(__file__).resolve().parents[1]


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def build(cfg: dict) -> AnalogArraySpec:
    k = yaml.safe_load((ROOT / "config/circuits/sar_switching.yaml").read_text())["conventional"]
    return m1_spec(cfg, k)


def err(model: float, ref: float) -> float:
    return 100.0 * (model - ref) / ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config/macros/m1_jia_jssc20.yaml"))
    ap.add_argument("--out", default=str(ROOT / "results/b1_m1_jia"))
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    s = build(cfg)
    v0, n, act = cfg["vdd"], cfg["nrow"], cfg["calib_activity"]
    ck = cfg["check_points"]
    rows = []

    def add(name, model, ref, kind):
        rows.append({"check": name, "kind": kind, "model": model, "paper": ref, "err_pct": err(model, ref)})

    b = s.conversion_breakdown(wl_pulses=0.0, bl_activity=act)
    array = b["bitline"] + b["cell_fixed"]
    f12 = ck["fig12_fj_per_1b_mac"]
    add("array energy per column, 1.2 V (pJ)", array * 1e12, cfg["energy_pj"][v0]["cima"], "fit")
    add("ADC energy per column, 1.2 V (pJ)", b["adc"] * 1e12, cfg["energy_pj"][v0]["adc"], "fit")
    add("array energy per 1-b MAC, 1.2 V (fJ, Fig. 12)", array / n * 1e15, f12["cima"], "consistency")
    add("ADC energy per 1-b MAC, 1.2 V (fJ, Fig. 12)", b["adc"] / n * 1e15, f12["adc"], "consistency")
    add("1b-TOPS/W, 1.2 V (abstract)", s.tops_per_w(n, 0.0, act), ck["tops_w_1b"][1.2], "consistency (confirms 2 ops per 1-b MAC, array + ADC)")
    add("ADC / array energy (Sec. II-B: 18 %)", 100 * b["adc"] / array, 100 * ck["adc_energy_overhead"], "consistency")
    f = (0.85 / v0) ** 2
    lo = cfg["energy_pj"][0.85]
    add("array energy per column, 0.85 V (pJ)", array * f * 1e12, lo["cima"], "held-out (VDD^2 rule)")
    add("ADC energy per column, 0.85 V (pJ)", b["adc"] * f * 1e12, lo["adc"], "held-out (VDD^2 rule)")
    add("1b-TOPS/W, 0.85 V (abstract)", s.tops_per_w(n, 0.0, act) / f, ck["tops_w_1b"][0.85], "held-out (VDD^2 rule)")

    e_dac = s.adc.dac_switch_factor * 2 ** s.adc.bits * s.adc.c_unit * v0 ** 2
    fitted = {
        "e_bl_cell_fJ_per_active_cell": s.e_bl_cell * 1e15, "e_cell_fixed_fJ": s.e_cell_fixed * 1e15,
        "adc_dac_fJ": e_dac * 1e15, "adc_comparator_logic_fJ_per_bit_cycle": s.adc.e_comparator * 1e15,
        "energy_per_1b_mac_fJ": (array + b["adc"]) / n * 1e15,
        "weight_load_fJ_per_bit": {str(v): cfg["energy_pj"][v]["load_32b"] / 32 * 1e3 for v in cfg["energy_pj"]},
        "weight_load_over_1b_mac": f12["load_per_bit"] / f12["total"],
    }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"macro": cfg["macro"], "git": git_hash(), "config": cfg, "fitted": fitted, "checks": rows}
    (out / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    lines = [f"# B1 calibration, {cfg['macro']} (git {res['git']})", "",
             f"Fitted at 1.2 V, 65 nm, per cell and evaluation: data-dependent {fitted['e_bl_cell_fJ_per_active_cell']:.2f} fJ "
             f"per active cell (assumed activity {act}), data-independent {fitted['e_cell_fixed_fJ']:.2f} fJ. ADC (8-b SAR): "
             f"DAC {fitted['adc_dac_fJ']:.0f} fJ (1.2 fF unit, conventional switching) + "
             f"{fitted['adc_comparator_logic_fJ_per_bit_cycle']:.0f} fJ per bit cycle (comparator + logic, fitted). "
             f"Array + ADC: {fitted['energy_per_1b_mac_fJ']:.2f} fJ per 1-b MAC.", "",
             "| Check | Kind | Model | Paper | Error |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['check']} | {r['kind']} | {r['model']:.4g} | {r['paper']:.4g} | {r['err_pct']:+.1f} % |")
    lines += ["", "Notes:",
              "- 192 / 400 1b-TOPS/W equal 2 x 2304 / (array + ADC energy per column) from Table I: the abstract counts "
              "2 ops per 1-b MAC and 'MAC hardware' is the array plus the ADC (no reshaping buffer, no ABN, no NMC).",
              f"- Weight loading (Table I, Fig. 12): {fitted['weight_load_fJ_per_bit']['1.2']:.0f} fJ per bit at 1.2 V, "
              f"{fitted['weight_load_fJ_per_bit']['0.85']:.0f} fJ at 0.85 V, {fitted['weight_load_over_1b_mac']:.1f}x one 1-b MAC. "
              "Input for the array-write cost in B3.",
              "- Not checked: area and throughput (the paper gives no macro-only area or evaluation time that the "
              "0.60 / 0.24 1b-TOPS/mm2 can be rebuilt from). The data pattern of Table I is not stated; activity 0.5 is an "
              "assumption and only moves energy between the two array terms at the calibration point."]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
