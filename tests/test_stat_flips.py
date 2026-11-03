"""Discrete ADC code-flip sampling in mode 'statistical' (stat_noise: auto, findings F9)."""

import pytest
import torch

from cimsim.config import CIMConfig
from cimsim.core.layers import CIMLinear

BASE = dict(w_bits=8, x_bits=8, nrow=128, signed_act=True, w_gran="per_channel", x_gran="per_token",
            adc_bits=6, adc_range="calibrated")


def _x(T, K, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(T, K, generator=g)
    x[:, :4] *= 25
    return x


def _errors(sigma, noise_mode, backend="fast", T=1024, K=512, N=64, seed=0):
    torch.manual_seed(0)
    lin = torch.nn.Linear(K, N)
    cfg = {**BASE, "rand_noise_sigma": sigma, "stat_noise": noise_mode, "seed": seed}
    st = CIMLinear.from_linear(lin, CIMConfig(mode="statistical", **cfg), name="L")
    st.start_calibration(); st(_x(256, K, 1)); st.finish_calibration()
    ba = CIMLinear.from_linear(lin, CIMConfig(mode="bit_accurate", backend=backend, **cfg), name="L")
    ba.adc_fs = st.adc_fs
    x = _x(T, K, 2)
    y0 = CIMLinear(lin.weight, lin.bias, ba.cfg.replace(mode="fake_quant"), name="F")(x).double()
    return st, st(x).double() - y0, ba(x).double() - y0


@pytest.mark.parametrize("backend", ["fast", "reference"])
def test_rare_flips_match_bit_accurate_shape(backend):
    """Noise 0.10 % of full scale, ADC step 1 MAC unit: a few % of outputs carry a large error. The discrete
    model reproduces the fraction of affected outputs and the std; the Gaussian puts error on every output."""
    st, e_st, e_ba = _errors(0.10, "auto", backend)
    assert st.stat_info["flip_pairs"] > 0 and st.stat_info["std"] < 1e-6       # all noise is discrete here
    f_st, f_ba = float((e_st.abs() > 1e-9).double().mean()), float((e_ba.abs() > 1e-9).double().mean())
    assert 0.005 < f_ba < 0.1
    assert f_st == pytest.approx(f_ba, rel=0.2)
    assert float(e_st.std() / e_ba.std()) == pytest.approx(1.0, rel=0.15)
    _, e_g, _ = _errors(0.10, "gaussian", backend)
    assert float((e_g.abs() > 1e-9).double().mean()) > 0.99


def test_flips_seeded():
    _, a, _ = _errors(0.10, "auto", seed=3)
    _, b, _ = _errors(0.10, "auto", seed=3)
    _, c, _ = _errors(0.10, "auto", seed=4)
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_frequent_flips_stay_gaussian():
    """Above the rare-flip regime (q > 0.05 per conversion) the Gaussian is kept."""
    st, _, _ = _errors(0.5, "auto", T=64)
    assert st.stat_info["flip_pairs"] == 0 and st.stat_info["std"] > 0
