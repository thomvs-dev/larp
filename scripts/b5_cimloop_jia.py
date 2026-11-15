"""B5: one CiMLoop comparison case. The Jia JSSC 2020 macro (M1) in AccelForge (the maintained
successor of CiMLoop, same authors; its `jia_jssc_2020` model is the CiMLoop macro model) against
this simulator's M1 calibration (`results/b1_m1_jia/`) and the paper.

Two steps, so that neither environment needs the other's packages:

  run      needs accelforge + a clone of github.com/Accelergy-Project/accelforge (for the example
           architecture files); does not import cimsim. Writes results/<name>/accelforge.json.
  compare  needs cimsim; does not import accelforge. Reads accelforge.json, writes
           results/<name>/summary.{json,md}.

Cases (1 = 1-b inputs and weights, the conditions of the paper's Table I; 8 = 8-b inputs and weights,
the operating point Part B uses), each at 0.85 V and 1.2 V. Workload: AccelForge's own peak-throughput
matmul for this macro (M = 1, N = 256 outputs, K = 2,304 inputs), so one 1-b run is one evaluation of
the full array; Table I values are per column, i.e. per 1 / 256 of that.

    python scripts/b5_cimloop_jia.py run --af-repo third_party/accelforge --name b5_cimloop_jia
    python scripts/b5_cimloop_jia.py compare --name b5_cimloop_jia
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = [(1, 0.85), (1, 1.2), (8, 0.85), (8, 1.2)]
N_OUT, K_IN = 256, 2304                 # AccelForge's workload for this macro: N = 4 * 64, K = 3 * 4 * 192
GROUPS = {                              # as in AccelForge notebooks/compute_in_memory/jia_jssc_2020.ipynb
    "adc": ["ADC"],
    "array": ["RowDrivers", "WeightDrivers", "CimUnit", "BitcellCapacitor", "ColumnDrivers"],
    "nmc_datapath": ["OutDatapath", "ShiftAdd"],
    "zero_gating": ["InputZeroGating"],
}
TOPS_COMPONENTS = ["RowDrivers", "CimUnit", "ADC", "ColumnDrivers", "BitcellCapacitor", "WeightDrivers"]


def _git(cwd: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=cwd, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


# ------------------------------------------------------------------------------------------------
# run (AccelForge side)
# ------------------------------------------------------------------------------------------------
def run(af_repo: Path, name: str) -> None:
    from importlib.metadata import version

    arch_dir = (af_repo / "examples" / "arches" / "compute_in_memory").resolve()
    if not (arch_dir / "jia_jssc_2020.yaml").is_file():
        raise SystemExit(f"jia_jssc_2020.yaml not found under {arch_dir}; pass --af-repo <accelforge clone>")
    sys.path.insert(0, str(arch_dir))
    from _load_spec import get_spec   # AccelForge's own loader for its CiM examples

    out = {"our_git": _git(ROOT), "accelforge_version": version("accelforge"),
           "accelforge_repo_git": _git(af_repo), "workload": {"M": 1, "N": N_OUT, "K": K_IN}, "cases": {}}
    print(f"git {out['our_git']}  accelforge {out['accelforge_version']}  repo {out['accelforge_repo_git']}", flush=True)
    for bits, v in CASES:
        t0 = time.time()
        spec = get_spec("jia_jssc_2020", add_dummy_main_memory=True)
        spec.variables.voltage = v
        spec.variables.input_bits = spec.variables.weight_bits = spec.variables.output_bits = bits
        for e in spec.workload.einsums:
            for ta in e.tensor_accesses:
                ta.bits_per_value = bits
        r = spec.map_workload_to_arch(print_progress=False)
        pc = r.per_compute()
        case = {
            "bits": bits, "vdd": v,
            "energy_total_J": {k: float(x) for k, x in r.energy(per_component=True).items()},
            "energy_per_mac_J": {k: float(x) for k, x in pc.energy(per_component=True).items()},
            "per_mac_total_J": float(pc.energy()),
            "latency_per_mac_s": float(pc.latency()),
            "seconds": round(time.time() - t0, 1),
        }
        out["cases"][f"{bits}b_{v}V"] = case
        print(f"{bits} b, {v} V: {case['per_mac_total_J'] * 1e15:.3f} fJ per MAC ({case['seconds']} s)", flush=True)
    d = ROOT / "results" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "accelforge.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"written: results/{name}/accelforge.json")


# ------------------------------------------------------------------------------------------------
# compare (this simulator's side)
# ------------------------------------------------------------------------------------------------
def ours(bits: int, v: float) -> dict:
    """M1 calibration (validation/b1_m1_acim.py) at another supply by VDD^2 (docs/assumptions.md B-21)."""
    import yaml

    from cimsim.circuits.macros import m1_spec

    cfg = yaml.safe_load((ROOT / "config/macros/m1_jia_jssc20.yaml").read_text(encoding="utf-8"))
    k = yaml.safe_load((ROOT / "config/circuits/sar_switching.yaml").read_text(encoding="utf-8"))["conventional"]
    s = m1_spec(cfg, k)
    f = (v / cfg["vdd"]) ** 2
    b = s.conversion_breakdown(wl_pulses=0.0, bl_activity=cfg["calib_activity"])
    array, adc = (b["bitline"] + b["cell_fixed"]) * f, b["adc"] * f
    n = cfg["nrow"]
    # one conversion = one column, one input bit, n rows; a bits x bits MAC uses bits^2 / n of a conversion
    per_mac = (array + adc) * bits * bits / n
    return {"array_per_column_J": array, "adc_per_column_J": adc, "per_mac_J": per_mac,
            "tops_w": 2 / per_mac / 1e12}


def af_case(c: dict, bits: int) -> dict:
    tot = c["energy_total_J"]
    g = {k: sum(tot.get(x, 0.0) for x in cs) for k, cs in GROUPS.items()}
    macs = N_OUT * K_IN
    e_tops = sum(c["energy_per_mac_J"].get(x, 0.0) for x in TOPS_COMPONENTS)
    return {"array_per_column_J": g["array"] / N_OUT, "adc_per_column_J": g["adc"] / N_OUT,
            "nmc_per_column_J": g["nmc_datapath"] / N_OUT, "zero_gating_per_column_J": g["zero_gating"] / N_OUT,
            "per_mac_J": (g["array"] + g["adc"]) / macs, "per_mac_all_J": c["per_mac_total_J"],
            "tops_w": 2 / e_tops / 1e12 if e_tops > 0 else None, "groups_total_J": g}


def compare(name: str) -> dict:
    import yaml

    d = ROOT / "results" / name
    af = json.loads((d / "accelforge.json").read_text(encoding="utf-8"))
    cfg = yaml.safe_load((ROOT / "config/macros/m1_jia_jssc20.yaml").read_text(encoding="utf-8"))
    paper = cfg["energy_pj"]
    rows = []
    for bits, v in CASES:
        c = af["cases"].get(f"{bits}b_{v}V")
        if c is None:
            continue
        a, o = af_case(c, bits), ours(bits, v)
        p = paper.get(v) if bits == 1 else None
        rows.append({"bits": bits, "vdd": v, "accelforge": a, "ours": o,
                     "paper": None if p is None else {"array_per_column_J": p["cima"] * 1e-12,
                                                      "adc_per_column_J": p["adc"] * 1e-12,
                                                      "nmc_per_column_J": p["nmc_output"] * 1e-12,
                                                      "tops_w": cfg["check_points"]["tops_w_1b"][v]}})

    def pct(x, ref):
        return "—" if x is None or not ref else f"{100 * (x - ref) / ref:+.1f} %"

    out = {"our_git": _git(ROOT)[:7], "accelforge": {k: af[k] for k in ("accelforge_version", "accelforge_repo_git", "our_git")},
           "rows": rows}
    (d / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    pj, fj = 1e12, 1e15
    lines = [f"# B5: Jia JSSC 2020 macro, AccelForge (CiMLoop model) vs this simulator (git {out['our_git']})", "",
             f"AccelForge {af['accelforge_version']} (repo {af['accelforge_repo_git']}), run at our git {af['our_git']}. "
             "Paper = Table I of Jia et al. (per column and evaluation, 1-b). This simulator = M1 calibration "
             "(`results/b1_m1_jia/`), fitted to Table I at 1.2 V, other supplies by VDD².", "",
             "## Table I conditions (1-b inputs and weights), per column and evaluation", "",
             "| VDD | Quantity | Paper | AccelForge | This simulator | AccelForge vs paper | Ours vs paper |",
             "|---|---|---|---|---|---|---|"]
    for r in [r for r in rows if r["bits"] == 1]:
        a, o, p = r["accelforge"], r["ours"], r["paper"]
        for key, lab, sc, unit in (("array_per_column_J", "array (CIMA)", pj, "pJ"), ("adc_per_column_J", "ADC", pj, "pJ"),
                                   ("nmc_per_column_J", "NMC data path", pj, "pJ"), ("tops_w", "1b-TOPS/W", 1, "")):
            ov = o.get(key)
            lines.append(f"| {r['vdd']} V | {lab} {unit} | {p[key] * sc:.3g} | "
                         f"{'—' if a.get(key) is None else f'{a[key] * sc:.3g}'} | {'not modelled' if ov is None else f'{ov * sc:.3g}'} | "
                         f"{pct(a.get(key), p[key])} | {pct(ov, p[key])} |")
    lines += ["", "## Part B operating point (8-b inputs and weights), energy per 8 b × 8 b MAC, array + ADC", "",
              "| VDD | AccelForge (fJ) | This simulator (fJ) | Ours / AccelForge | AccelForge, all components (fJ) |",
              "|---|---|---|---|---|"]
    for r in [r for r in rows if r["bits"] == 8]:
        a, o = r["accelforge"], r["ours"]
        lines.append(f"| {r['vdd']} V | {a['per_mac_J'] * fj:.1f} | {o['per_mac_J'] * fj:.1f} | "
                     f"{o['per_mac_J'] / a['per_mac_J']:.2f}× | {a['per_mac_all_J'] * fj:.1f} |")
    lines += ["", "Differences are explained in `validation/tool_comparison.md`."]
    (d / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--af-repo", required=True, type=Path)
    r.add_argument("--name", default="b5_cimloop_jia")
    c = sub.add_parser("compare")
    c.add_argument("--name", default="b5_cimloop_jia")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.af_repo, a.name)
    else:
        sys.path.insert(0, str(ROOT))
        compare(a.name)


if __name__ == "__main__":
    main()
