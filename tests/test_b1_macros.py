"""Calibration scripts for the analog macros M1 / M2 (GATE B1): fit points are reproduced and the
published cross-checks hold. Paper numbers live in config/macros/*.yaml."""

import importlib.util
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "validation" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_m2_fit_points_and_headline():
    m = load("b1_m2_acim")
    cfg = yaml.safe_load((ROOT / "config/macros/m2_sinangil_jssc21.yaml").read_text())
    s = m.build(cfg)
    for p in cfg["calib_points"]:
        assert m.macro_energy(s, cfg, p["mac"]) == pytest.approx(p["e_pj"] * 1e-12, rel=1e-9)
    assert m.tops_w(s, cfg, 6) == pytest.approx(351.0, rel=0.02)           # paper: 351 "in average"
    assert m.tops_w(s, cfg, 15) == pytest.approx(262.3, rel=0.01)          # paper: lowest
    # wordline share at the average point is the paper's 5 %
    b = s.conversion_breakdown(wl_pulses=6, bl_activity=6 / 15)
    assert b["wordline"] / sum(b.values()) == pytest.approx(0.05, rel=1e-6)
    assert s.area(64, 64)["total"] == pytest.approx(0.0032e-6, rel=1e-9)
    assert 2 * 1024 / s.cycle_time() == pytest.approx(372.4e9, rel=1e-3)


def test_m2_operating_range_within_gate():
    m = load("b1_m2_acim")
    cfg = yaml.safe_load((ROOT / "config/macros/m2_sinangil_jssc21.yaml").read_text())
    s = m.build(cfg)
    for p in cfg["energy_vs_mac_pj"]:
        if p["mac"] >= 1:
            assert m.macro_energy(s, cfg, p["mac"]) == pytest.approx(p["e"] * 1e-12, rel=0.07)
    for p in cfg["check_points"]["max_energy_vs_vdd_pj"]:
        assert m.macro_energy(s, cfg, 15, p["vdd"]) == pytest.approx(p["e"] * 1e-12, rel=0.08)


def test_m1_fit_and_voltage_points():
    m = load("b1_m1_acim")
    cfg = yaml.safe_load((ROOT / "config/macros/m1_jia_jssc20.yaml").read_text())
    s = m.build(cfg)
    b = s.conversion_breakdown(0.0, cfg["calib_activity"])
    assert b["bitline"] + b["cell_fixed"] == pytest.approx(20.4e-12, rel=1e-9)
    assert b["adc"] == pytest.approx(3.56e-12, rel=1e-9)
    assert b["bitline"] / (b["bitline"] + b["cell_fixed"]) == pytest.approx(0.66)
    assert s.tops_per_w(2304, 0.0, 0.5) == pytest.approx(192.0, rel=0.01)   # abstract, from Table I energies
    f = (0.85 / 1.2) ** 2
    assert s.tops_per_w(2304, 0.0, 0.5) / f == pytest.approx(400.0, rel=0.05)
    # sparsity-proportional: no active input leaves the data-independent third
    assert s.conversion_breakdown(0.0, 0.0)["bitline"] == 0.0
