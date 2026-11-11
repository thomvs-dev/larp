"""GATE B1, macro M4 (Chih ISSCC 2021, digital CIM): fit the DCIM cost model to the
two measured TOPS/W points, then compare with held-out numbers from the paper.

python validation/b1_m4_dcim.py [--out results/b1_m4_chih]
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml

from cimsim.circuits.dcim import DCIMSpec, fit_toggle_line

ROOT = Path(__file__).resolve().parents[1]


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def build(cfg: dict) -> DCIMSpec:
    pts = [(p["toggle"], p["tops_w"]) for p in cfg["calib_points"]]
    e0, e1 = fit_toggle_line(pts, 4, 4)
    n_bits = cfg["rows"] * cfg["cols"] * cfg["col_wbits"]
    return DCIMSpec(rows=cfg["rows"], cols=cfg["cols"], col_wbits=cfg["col_wbits"], e_bitop_base=e0,
                    e_bitop_toggle=e1, t_cycle=cfg["t_cycle"], pipeline_cycles=cfg["pipeline_cycles"],
                    area_per_weight_bit=cfg["macro_area_mm2"] * 1e-6 / n_bits, vdd=cfg["vdd"],
                    node_nm=cfg["node_nm"])


def err(model: float, ref: float) -> float:
    return 100.0 * (model - ref) / ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config/macros/m4_chih_isscc21.yaml"))
    ap.add_argument("--out", default=str(ROOT / "results/b1_m4_chih"))
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    s = build(cfg)
    ck = cfg["check_points"]
    rows = []

    def add(name, model, ref, kind):
        rows.append({"check": name, "kind": kind, "model": model, "paper": ref, "err_pct": err(model, ref)})

    for p in cfg["calib_points"]:
        add(f"TOPS/W 4b/4b toggle {p['toggle']:.2f}", s.tops_per_w(4, 4, p["toggle"]), p["tops_w"], "fit")
    add("throughput 4b/4b (GOPS)", s.peak_ops_per_s(4, 4) / 1e9, ck["throughput_4b4b_gops"], "held-out")
    add("area efficiency 4b/4b (TOPS/mm2)", s.tops_per_mm2(4, 4), ck["area_eff_tops_mm2"], "held-out")
    add("TOPS/W 8b/8b toggle 0.18", s.tops_per_w(8, 8, 0.18), ck["tops_w_8b8b_est"], "held-out (paper estimate)")
    add("throughput 8b/8b (GOPS)", s.peak_ops_per_s(8, 8) / 1e9, ck["throughput_8b8b_gops_est"], "held-out (paper estimate)")
    add("cycle time 8b/8b (ns)", s.pass_time(8) * 1e9, ck["cycle_time_8b8b_ns_est"], "held-out (paper estimate)")
    for p in ck["toggle_curve_0p72v"]:
        add(f"TOPS/W 4b/4b toggle {p['toggle']:.4f} (Fig. 16.4.5)", s.tops_per_w(4, 4, p["toggle"]), p["tops_w"],
            "held-out (digitised curve)")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {
        "macro": cfg["macro"], "git": git_hash(), "config": cfg,
        "fitted": {"e_bitop_base_fJ": s.e_bitop_base * 1e15, "e_bitop_toggle_fJ": s.e_bitop_toggle * 1e15,
                   "area_per_weight_bit_um2": s.area_per_weight_bit * 1e12,
                   "mac_energy_4b4b_fJ_toggle018": s.mac_energy(4, 4, 0.18) * 1e15},
        "checks": rows,
    }
    (out / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    lines = [f"# B1 calibration, {cfg['macro']} (git {res['git']})", "",
             f"Fitted: e_bitop = {s.e_bitop_base*1e15:.3f} fJ + {s.e_bitop_toggle*1e15:.3f} fJ x toggle "
             f"(per 1b x 1b product, 0.72 V, 50 % weight ones); area {s.area_per_weight_bit*1e12:.3f} um^2 per weight bit.", "",
             "| Check | Kind | Model | Paper | Error |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['check']} | {r['kind']} | {r['model']:.1f} | {r['paper']:.1f} | {r['err_pct']:+.1f} % |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
