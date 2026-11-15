from pathlib import Path

import pytest
import yaml

from cimsim.circuits import UNIT_OPS, DigitalSpec, ScalingRule

ROOT = Path(__file__).parents[1]


def spec():
    return DigitalSpec.from_dict(yaml.safe_load((ROOT / "config/circuits/digital_45nm.yaml").read_text()))


def test_published_points_are_reproduced():
    """Per-bit values against the published 45 nm points (Horowitz ISSCC 2014): 16 / 32-b exact,
    8-b add 17 % low, 8-b multiply 3 % low."""
    cfg = yaml.safe_load((ROOT / "config/circuits/digital_45nm.yaml").read_text())
    s = spec()
    for bits, e in cfg["points"]["add"].items():
        assert s.add(bits) == pytest.approx(e, rel=0.18)
    assert s.add(32) == pytest.approx(0.1e-12, rel=1e-6)
    assert s.add(16) == pytest.approx(0.05e-12, rel=1e-6)
    for bits, e in cfg["points"]["mul"].items():
        assert s.mul(bits) == pytest.approx(e, rel=0.04)


def test_mac_and_buffers():
    s = spec()
    assert s.mac(8, 8, 24) == pytest.approx(s.mul(8, 8) + s.add(24))
    assert s.sram_read(32) == pytest.approx(5e-12)
    assert s.reg(16) == pytest.approx(0.12e-12)
    assert s.lut(16) == s.sram_read(16)


def test_scaling_applies_to_every_energy():
    rule = ScalingRule(1.0, 2.0, 1.0)
    a, b = spec(), spec().scaled(22, 0.8, rule)
    f = (22 / 45) * (0.8 / 0.9) ** 2
    assert (b.node_nm, b.vdd) == (22, 0.8)
    for name in ("e_add_bit", "e_mul_bit2", "e_reg_bit", "e_sram_bit"):
        assert getattr(b, name) == pytest.approx(getattr(a, name) * f)
    assert b.unit_energy("softmax", 100) == pytest.approx(a.unit_energy("softmax", 100) * f)


def test_unit_energy_counts():
    s = spec()
    n_mul, n_add, n_lut = UNIT_OPS["rmsnorm"]["element"]
    v_mul, v_add, v_lut = UNIT_OPS["rmsnorm"]["vector"]
    want = 10 * (n_mul * s.mul(16) + n_add * s.add(16) + n_lut * s.lut(16)) \
        + 2 * (v_mul * s.mul(16) + v_add * s.add(16) + v_lut * s.lut(16))
    assert s.unit_energy("rmsnorm", 10, vectors=2) == pytest.approx(want)
    assert s.unit_energy("relu", 1000, vectors=0) == 0.0
    assert s.unit_energy("layernorm", 1) > s.unit_energy("rmsnorm", 1)
    assert s.hadamard(100, bits=16) == pytest.approx(100 * s.add(16))
    with pytest.raises(KeyError):
        s.unit_energy("tanh", 1)
