import math

import pytest
import yaml

from cimsim.circuits import ADCSpec, adc_energy, crossover_enob


def bound(bits=6, **kw):
    d = dict(arch="bound", bits=bits, fom_walden=10e-15, fom_schreier_db=170.0)
    d.update(kw)
    return ADCSpec(**d)


def test_walden_term_doubles_per_bit():
    e = [bound(b).energy_per_conversion() for b in range(4, 9)]
    for lo, hi in zip(e, e[1:]):
        assert hi / lo == pytest.approx(2.0, rel=1e-12)
    assert bound(6).energy_per_conversion() == pytest.approx(10e-15 * 64)


def test_schreier_term_quadruples_per_bit_at_high_resolution():
    a, b = bound(16), bound(17)
    assert a.schreier_energy() > a.walden_energy()
    assert b.energy_per_conversion() / a.energy_per_conversion() == pytest.approx(10 ** 0.602, rel=1e-9)


def test_crossover_matches_terms():
    n = crossover_enob(10e-15, 170.0)
    s = bound(16, enob=n)
    assert s.walden_energy() == pytest.approx(s.schreier_energy(), rel=1e-9)
    assert 4 < 8 < n  # Walden term governs every ADC width the simulator uses


def test_enob_lowers_energy_and_is_capped():
    assert bound(6, enob=5.0).energy_per_conversion() == pytest.approx(bound(5).energy_per_conversion())
    with pytest.raises(ValueError):
        bound(6, enob=7.0)
    s = ADCSpec.from_dict(dict(arch="bound", bits=8, enob=7.5, fom_walden=1e-15, fom_schreier_db=170.0), bits=6)
    assert s.bits == 6 and s.enob == 6.0


def test_flash_scales_with_comparator_count():
    f = ADCSpec(arch="flash", bits=4, e_comparator=2e-15, e_fixed=5e-15)
    assert f.energy_per_conversion() == pytest.approx(15 * 2e-15 + 5e-15)
    assert f.with_bits(5).energy_per_conversion() == pytest.approx(31 * 2e-15 + 5e-15)


def test_missing_parameters_rejected():
    with pytest.raises(ValueError):
        ADCSpec(arch="bound", bits=6, fom_walden=1e-15)
    with pytest.raises(ValueError):
        ADCSpec(arch="flash", bits=4)
    with pytest.raises(ValueError):
        ADCSpec(arch="pipelined", bits=6)
    with pytest.raises(KeyError):
        ADCSpec.from_dict(dict(arch="bound", bits=6, fom_walden=1e-15, fom_schreier_db=170.0, fom=1))


def test_adc_energy_total():
    s = bound(6)
    assert adc_energy(1000, s) == pytest.approx(1000 * 640e-15)
    with pytest.raises(ValueError):
        adc_energy(-1, s)


def test_example_config_loads():
    from pathlib import Path
    cfg = yaml.safe_load((Path(__file__).parents[1] / "config/circuits/adc_survey_lines.yaml").read_text())
    for d in cfg.values():
        s = ADCSpec.from_dict(d)
        assert math.isfinite(s.energy_per_conversion()) and s.energy_per_conversion() > 0
