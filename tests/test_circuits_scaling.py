from pathlib import Path

import pytest
import yaml

from cimsim.circuits import ScalingRule

RULE = ScalingRule(energy_node_exp=1.0, area_node_exp=2.0, delay_node_exp=1.0)

ROOT = Path(__file__).parents[1]


def test_identity_and_round_trip():
    assert RULE.energy_factor(45, 0.9, 45, 0.9) == 1.0
    f = RULE.energy_factor(45, 0.9, 22, 0.8) * RULE.energy_factor(22, 0.8, 45, 0.9)
    assert f == pytest.approx(1.0)


def test_energy_is_node_times_vdd_squared():
    assert RULE.energy_factor(45, 0.9, 22, 0.8) == pytest.approx((22 / 45) * (0.8 / 0.9) ** 2)
    assert RULE.area_factor(65, 22) == pytest.approx((22 / 65) ** 2)
    assert RULE.delay_factor(65, 22) == pytest.approx(22 / 65)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        RULE.energy_factor(0, 0.9, 22, 0.8)
    with pytest.raises(KeyError):
        ScalingRule.from_dict({"energy_node_exp": 1, "area_node_exp": 2, "delay_node_exp": 1, "x": 1})


def test_config_loads():
    cfg = yaml.safe_load((ROOT / "config/circuits/scaling.yaml").read_text())
    rule = ScalingRule.from_dict(cfg["rule"])
    assert rule.area_node_exp == 2.0
    assert cfg["target"]["node_nm"] == 22
