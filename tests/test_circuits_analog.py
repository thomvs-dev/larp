from pathlib import Path

import pytest
import yaml

from cimsim.circuits import ADCSpec, AnalogArraySpec

# Test values only; not hardware data.
ADC = ADCSpec(arch="flash", bits=4, e_comparator=2e-15)

ROOT = Path(__file__).parents[1]


def spec(**kw):
    d = dict(nrow=64, adc=ADC, e_wl_cell=0.1e-15, e_bl_cell=0.5e-15, e_col_fixed=3e-15, e_add_bit=1e-15,
             e_reg_bit=2e-15, acc_bits=12, in_bits=4, adc_share=4, t_array=2e-9, t_adc=0.5e-9,
             area_cell=1e-13, area_adc=2e-11, area_col_periph=1e-12, area_row_periph=3e-12)
    d.update(kw)
    return AnalogArraySpec(**d)


def test_conversion_energy_is_the_sum_of_its_parts():
    s = spec()
    b = s.conversion_breakdown(wl_pulses=7.5, bl_activity=0.25)
    assert b["wordline"] == pytest.approx(64 * 0.1e-15 * 7.5)
    assert b["cell_fixed"] == 0.0 and spec(e_cell_fixed=1e-16).conversion_breakdown(0, 0)["cell_fixed"] == pytest.approx(64e-16)
    assert b["bitline"] == pytest.approx(64 * 0.5e-15 * 0.25)
    assert b["adc"] == pytest.approx(15 * 2e-15)
    assert b["shift_add"] == pytest.approx(12e-15)
    assert b["out_register"] == pytest.approx(4 * 2e-15)     # default: ADC bits
    assert s.energy_per_conversion(7.5, 0.25) == pytest.approx(sum(b.values()))


def test_energy_grows_with_data_and_nrow():
    s = spec()
    assert s.energy_per_conversion(1, 0.5) > s.energy_per_conversion(1, 0.1) > s.energy_per_conversion(0, 0.0)
    a = spec(nrow=128).conversion_breakdown(1, 0.5)
    b = s.conversion_breakdown(1, 0.5)
    assert a["bitline"] == pytest.approx(2 * b["bitline"]) and a["adc"] == b["adc"]


def test_adc_bits_override():
    s = spec()
    assert s.conversion_breakdown(0, 0, adc_bits=6)["adc"] == pytest.approx(63 * 2e-15)
    assert s.conversion_breakdown(0, 0, adc_bits=6)["out_register"] == pytest.approx(6 * 2e-15)


def test_event_counts_to_energy():
    s = spec()
    e = s.energy(adc_conversions=1000, array_cycles=10, wl_pulses=1.0, bl_activity=0.3)
    assert e["in_register"] == pytest.approx(10 * 64 * 4 * 2e-15)
    assert e["total"] == pytest.approx(1000 * s.energy_per_conversion(1.0, 0.3) + e["in_register"])
    assert s.energy(0, 0, 0, 0)["total"] == 0.0


def test_adc_sharing_changes_time_and_area_only():
    a, b = spec(adc_share=1), spec(adc_share=8)
    assert a.energy_per_conversion(1, 0.3) == b.energy_per_conversion(1, 0.3)
    assert a.cycle_time() == pytest.approx(2.5e-9) and b.cycle_time() == pytest.approx(6e-9)
    assert a.area(64, 64)["adc"] == pytest.approx(64 * 2e-11)
    assert b.area(64, 64)["adc"] == pytest.approx(8 * 2e-11)
    assert b.n_adcs(65) == 9
    ar = b.area(64, 64)
    assert ar["total"] == pytest.approx(ar["cells"] + ar["adc"] + ar["col_periph"] + ar["row_periph"])


def test_tops_per_w():
    s = spec(e_reg_bit=0.0)
    e = s.energy_per_conversion(1, 0.3)
    assert s.tops_per_w(16, 1, 0.3) == pytest.approx(2 * 16 / e / 1e12)


def test_input_checks_and_from_dict():
    s = spec()
    with pytest.raises(ValueError):
        s.energy_per_conversion(1, 1.5)
    with pytest.raises(ValueError):
        spec(nrow=0)
    d = dict(nrow=8, e_wl_cell=1e-16, e_bl_cell=1e-16, adc=dict(arch="flash", bits=4, e_comparator=1e-15))
    assert AnalogArraySpec.from_dict(d).adc.bits == 4
    with pytest.raises(KeyError):
        AnalogArraySpec.from_dict({**d, "bogus": 1})


def test_sar_adc_energy():
    f = yaml.safe_load((ROOT / "config/circuits/sar_switching.yaml").read_text())
    kw = dict(arch="sar", bits=10, c_unit=1e-15, vref=1.0, e_comparator=0.0)
    conv = ADCSpec(dac_switch_factor=f["conventional"], **kw).energy_per_conversion()
    mono = ADCSpec(dac_switch_factor=f["monotonic"], **kw).energy_per_conversion()
    assert conv == pytest.approx(1363.3e-15, rel=1e-3)      # 10-bit average, C = 1 fF, Vref = 1 V
    assert mono == pytest.approx(255.5e-15, rel=1e-3)
    full = ADCSpec(dac_switch_factor=0.25, e_logic_bit=1e-15, **{**kw, "bits": 8, "e_comparator": 2e-15})
    assert full.energy_per_conversion() == pytest.approx(0.25 * 256 * 1e-15 + 8 * 3e-15)
    with pytest.raises(ValueError):
        ADCSpec(arch="sar", bits=8, e_comparator=1e-15)
