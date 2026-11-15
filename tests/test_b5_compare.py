"""Compare step of scripts/b5_cimloop_jia.py on a stand-in AccelForge result (no accelforge import)."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def load():
    spec = importlib.util.spec_from_file_location("b5", ROOT / "scripts/b5_cimloop_jia.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def fake_case(m, array_pj, adc_pj, nmc_pj, per_mac_fj):
    n = m.N_OUT
    tot = {"RowDrivers": 0.0, "WeightDrivers": 0.0, "CimUnit": array_pj * n * 1e-12, "BitcellCapacitor": 0.0,
           "ColumnDrivers": 0.0, "ADC": adc_pj * n * 1e-12, "OutDatapath": nmc_pj * n * 1e-12, "ShiftAdd": 0.0,
           "InputZeroGating": 0.0}
    macs = m.N_OUT * m.K_IN
    per = {k: v / macs for k, v in tot.items()}
    return {"energy_total_J": tot, "energy_per_mac_J": per, "per_mac_total_J": per_mac_fj * 1e-15,
            "latency_per_mac_s": 1e-12}


def test_compare_against_paper_values(tmp_path, monkeypatch):
    m = load()
    monkeypatch.setattr(m, "ROOT", tmp_path)
    for p in ("config/macros/m1_jia_jssc20.yaml", "config/circuits/sar_switching.yaml"):
        (tmp_path / p).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / p).write_text((ROOT / p).read_text())
    d = tmp_path / "results/x"
    d.mkdir(parents=True)
    af = {"our_git": "abc", "accelforge_version": "0", "accelforge_repo_git": "def",
          "cases": {"1b_1.2V": fake_case(m, 20.4, 3.56, 14.7, 11.0), "1b_0.85V": fake_case(m, 9.7, 1.79, 8.3, 5.0),
                    "8b_1.2V": fake_case(m, 20.4, 3.56, 14.7, 700.0)}}
    (d / "accelforge.json").write_text(json.dumps(af))
    out = m.compare("x")
    r12 = next(r for r in out["rows"] if r["bits"] == 1 and r["vdd"] == 1.2)
    # stand-in AccelForge equals the paper; ours is the M1 fit (exact at 1.2 V)
    assert r12["accelforge"]["array_per_column_J"] == pytest.approx(20.4e-12)
    assert r12["ours"]["array_per_column_J"] == pytest.approx(20.4e-12)
    assert r12["ours"]["adc_per_column_J"] == pytest.approx(3.56e-12)
    assert r12["ours"]["tops_w"] == pytest.approx(192.3, rel=1e-3)
    assert r12["accelforge"]["tops_w"] == pytest.approx(2 * 2304 / (23.96e-12) / 1e12, rel=1e-6)
    r8 = next(r for r in out["rows"] if r["bits"] == 8)
    assert r8["ours"]["per_mac_J"] == pytest.approx(64 * r12["ours"]["per_mac_J"])
    assert (d / "summary.md").read_text().count("| 1.2 V |") == 5
