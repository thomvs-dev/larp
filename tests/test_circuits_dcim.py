import pytest
import torch

from cimsim.circuits.dcim import DCIMSpec, fit_toggle_line, input_toggle_rate


def chih_like(**kw):
    d = dict(rows=256, cols=64, col_wbits=4, e_bitop_base=1e-15, e_bitop_toggle=3e-15, t_cycle=2e-9,
             pipeline_cycles=1, area_per_weight_bit=3e-12)
    d.update(kw)
    return DCIMSpec(**d)


def test_fit_reproduces_two_points_exactly():
    e0, e1 = fit_toggle_line([(0.18, 89.0), (0.50, 52.0)], 4, 4)
    s = chih_like(e_bitop_base=e0, e_bitop_toggle=e1)
    assert s.tops_per_w(4, 4, 0.18) == pytest.approx(89.0)
    assert s.tops_per_w(4, 4, 0.50) == pytest.approx(52.0)
    assert e0 > 0 and e1 > 0


def test_energy_scales_with_bit_products():
    s = chih_like()
    assert s.mac_energy(8, 8, 0.3) == pytest.approx(4 * s.mac_energy(4, 4, 0.3))
    assert s.energy(1000, 4, 4, 0.0) == pytest.approx(1000 * 16 * 1e-15)
    with pytest.raises(ValueError):
        s.e_bitop(1.5)


def test_throughput_and_columns_per_weight():
    s = chih_like()
    assert s.outputs_per_pass(4) == 64 and s.outputs_per_pass(8) == 32 and s.outputs_per_pass(16) == 16
    assert s.peak_ops_per_s(4, 4) == pytest.approx(2 * 256 * 64 / 10e-9)
    assert s.pass_time(8) == pytest.approx(18e-9)


def test_toggle_rate_known_patterns():
    # all zeros -> no toggles; alternating 0b0101 / 0b1010 MSB-first streams toggle every step
    assert input_toggle_rate(torch.zeros(5, 3), 8) == 0.0
    x = torch.tensor([[5.0], [5.0]])  # 0101 0101 (unsigned, 4 b): every transition toggles, incl. token boundary
    assert input_toggle_rate(x, 4, signed=False) == pytest.approx(1.0)
    x = torch.tensor([[15.0], [0.0]])  # 1111 0000: one toggle in 7 transitions
    assert input_toggle_rate(x, 4, signed=False) == pytest.approx(1 / 7)
    # signed -1 in two's complement = all ones
    assert input_toggle_rate(torch.full((3, 2), -1.0), 8) == 0.0


def test_toggle_rate_random_is_half():
    torch.manual_seed(0)
    x = torch.randint(-127, 128, (512, 64)).float()
    assert input_toggle_rate(x, 8) == pytest.approx(0.5, abs=0.01)


def test_m4_config_and_script(tmp_path):
    import subprocess, sys, json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(root / "validation/b1_m4_dcim.py"), "--out", str(tmp_path)], check=True,
                   capture_output=True, cwd=root)
    res = json.loads((tmp_path / "results.json").read_text())
    for r in res["checks"]:
        assert abs(r["err_pct"]) < 30, r
