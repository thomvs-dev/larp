"""GATE B1, macro M2 (Sinangil JSSC 2021, 7 nm, 4-b flash): fit the analog array model to the two
labelled points of Fig. 13(a), then compare with the rest of the paper's numbers.

Mapping onto cimsim.circuits.AnalogArraySpec (docs/assumptions.md B-20):
- one conversion = one 4-b output = 4 bit columns x 64 rows = 256 cells, standing for 64 4b x 4b MACs;
- macro energy per cycle = 16 conversions; E(mac) = intercept + slope * mac (mac = output code 0..15);
- intercept -> flash ADC and every other data-independent column energy (15 x e_comparator);
- slope     -> wordline + bit-line energy, split by the paper's 5 % RWL share at the average point;
- supply scaling by VDD^2 (ScalingRule) for the points at other voltages.

python validation/b1_m2_acim.py [--out results/b1_m2_sinangil]
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml

from cimsim.circuits import ADCSpec, AnalogArraySpec

ROOT = Path(__file__).resolve().parents[1]
FULL = 15  # largest 4-b code


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def build(cfg: dict) -> AnalogArraySpec:
    (m_a, e_a), (m_b, e_b) = [(p["mac"], p["e_pj"] * 1e-12) for p in cfg["calib_points"]]
    n_out = cfg["cols"] // cfg["w_bits"]
    cells = cfg["rows"] * cfg["w_bits"]                       # cells per conversion
    slope = (e_b - e_a) / (m_b - m_a) / n_out                 # J per conversion and code
    fixed = e_a / n_out - slope * m_a                         # J per conversion
    e_wl_code = cfg["rwl_power_share"] * (e_a / n_out) / m_a  # wordline J per conversion and code (5 % at the average point)
    e_bl_code = slope - e_wl_code
    area = cfg["macro_area_mm2"] * 1e-6
    sp = cfg["area_split"]
    return AnalogArraySpec(
        nrow=cells,
        adc=ADCSpec(arch="flash", bits=cfg["adc_bits"], e_comparator=fixed / (2 ** cfg["adc_bits"] - 1)),
        e_wl_cell=e_wl_code / cells,                          # per cell and RWL pulse (code = pulses per row at full weights)
        e_bl_cell=e_bl_code * FULL / cells,                   # per cell at full bit-line activity
        t_array=cfg["t_cycle"],                               # whole CIM cycle (pulses, charge sharing, ADC); no split published
        adc_share=cfg["w_bits"],                             # one flash ADC per weight = per 4 bit columns (t_adc = 0: no serialisation)
        area_cell=area * sp["sram_macro"] / (cfg["rows"] * cfg["cols"]),
        area_adc=area * sp["cim_circuits"] / n_out,
        area_col_periph=area * sp["test"] / cfg["cols"],
        vdd=cfg["vdd"], node_nm=cfg["node_nm"],
    )


def macro_energy(s: AnalogArraySpec, cfg: dict, mac: float, vdd: float | None = None) -> float:
    """J per macro cycle at output code `mac`; other supplies by VDD^2."""
    n_out = cfg["cols"] // cfg["w_bits"]
    e = n_out * s.energy_per_conversion(wl_pulses=mac, bl_activity=mac / FULL)
    return e if vdd is None else e * (vdd / cfg["vdd"]) ** 2


def tops_w(s, cfg, mac, vdd=None) -> float:
    macs = cfg["rows"] * (cfg["cols"] // cfg["w_bits"])
    return 2.0 * macs / macro_energy(s, cfg, mac, vdd) / 1e12


def err(model: float, ref: float) -> float:
    return 100.0 * (model - ref) / ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config/macros/m2_sinangil_jssc21.yaml"))
    ap.add_argument("--out", default=str(ROOT / "results/b1_m2_sinangil"))
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    s = build(cfg)
    ck = cfg["check_points"]
    avg_mac = cfg["calib_points"][0]["mac"]
    rows = []

    def add(name, model, ref, kind):
        rows.append({"check": name, "kind": kind, "model": model, "paper": ref, "err_pct": err(model, ref)})

    fit_macs = {p["mac"] for p in cfg["calib_points"]}
    for p in cfg["energy_vs_mac_pj"]:
        add(f"energy at MAC {p['mac']}, 0.8 V (pJ, Fig. 13a)", macro_energy(s, cfg, p["mac"]) * 1e12, p["e"],
            "fit" if p["mac"] in fit_macs else "held-out (digitised curve)")
    t = ck["tops_w_0p8v"]
    add("TOPS/W average, 0.8 V", tops_w(s, cfg, avg_mac), t["avg"], "held-out")
    add("TOPS/W lowest (MAC 15), 0.8 V", tops_w(s, cfg, FULL), t["min"], "held-out")
    add("TOPS/W highest (MAC 0), 0.8 V", tops_w(s, cfg, 0), t["max"], "held-out")
    for p in ck["max_energy_vs_vdd_pj"]:
        add(f"energy at MAC 15, {p['vdd']:.2f} V (pJ, Fig. 13b)", macro_energy(s, cfg, FULL, p["vdd"]) * 1e12, p["e"],
            "held-out (VDD^2 rule)")
    t = ck["tops_w_1p0v"]
    add("TOPS/W average, 1.0 V (Table I)", tops_w(s, cfg, avg_mac, 1.0), t["avg"], "held-out (VDD^2 rule)")
    add("TOPS/W lowest, 1.0 V (Table I)", tops_w(s, cfg, FULL, 1.0), t["min"], "held-out (VDD^2 rule)")
    add("TOPS/W highest, 1.0 V (Table I)", tops_w(s, cfg, 0, 1.0), t["max"], "held-out (VDD^2 rule)")
    macs = cfg["rows"] * (cfg["cols"] // cfg["w_bits"])
    add("throughput 0.8 V (GOPS)", 2 * macs / s.cycle_time() / 1e9, ck["throughput_gops"][0.8], "consistency (cycle time is an input)")
    ar = s.area(cfg["rows"], cfg["cols"])
    add("macro area (mm2)", ar["total"] * 1e6, cfg["macro_area_mm2"], "consistency (area is an input)")

    fitted = {
        "e_wl_cell_aJ_per_pulse": s.e_wl_cell * 1e18, "e_bl_cell_fJ": s.e_bl_cell * 1e15,
        "c_wl_cell_fF": s.e_wl_cell / cfg["vdd"] ** 2 * 1e15, "c_bl_cell_fF": s.e_bl_cell / cfg["vdd"] ** 2 * 1e15,
        "e_comparator_fJ": s.adc.e_comparator * 1e15, "adc_and_fixed_fJ_per_conversion": s.adc.energy_per_conversion() * 1e15,
        "energy_per_4b4b_mac_avg_fJ": macro_energy(s, cfg, avg_mac) / macs * 1e15,
        "area_cell_um2": s.area_cell * 1e12, "area_adc_um2": s.area_adc * 1e12,
    }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"macro": cfg["macro"], "git": git_hash(), "config": cfg, "fitted": fitted, "checks": rows}
    (out / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    lines = [f"# B1 calibration, {cfg['macro']} (git {res['git']})", "",
             f"Fitted at 0.8 V, 7 nm, per conversion (one 4-b output, 256 cells, 64 MACs): data-independent "
             f"{fitted['adc_and_fixed_fJ_per_conversion']:.1f} fJ (charged to the 15 flash comparators, "
             f"{fitted['e_comparator_fJ']:.2f} fJ each); wordline {fitted['e_wl_cell_aJ_per_pulse']:.2f} aJ per cell and pulse "
             f"({fitted['c_wl_cell_fF']:.4f} fF); bit line {fitted['e_bl_cell_fJ']:.3f} fJ per cell at full activity "
             f"({fitted['c_bl_cell_fF']:.3f} fF effective, compute and compensation capacitors included). "
             f"Average 4b x 4b MAC: {fitted['energy_per_4b4b_mac_avg_fJ']:.2f} fJ.", "",
             "| Check | Kind | Model | Paper | Error |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['check']} | {r['kind']} | {r['model']:.4g} | {r['paper']:.4g} | {r['err_pct']:+.1f} % |")
    lines += ["", "Notes:",
              "- The measured curve has a step between MAC 0 and MAC 1 that a line cannot follow; the fit uses the "
              "operating range (codes 6 and 15), so MAC 0 is the worst point.",
              "- Table I at 1.0 V and Fig. 13(b) disagree in the paper itself: Fig. 13(b) gives 13.1 pJ at MAC 15 and "
              "1.0 V (= 156 TOPS/W), Table I gives 189.3 TOPS/W as the lowest value at 1.0 V. The Table I values at "
              "1.0 V also scale more slowly than VDD^2 from the 0.8 V values (average: x1.09 in energy)."]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
