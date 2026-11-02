"""A3 analytic error model (cimsim/stat/analytic.py) and mode 'statistical'."""

import math

import pytest
import torch

from cimsim.config import CIMConfig
from cimsim.core.layers import CIMLinear
from cimsim.events import EventCounter
from cimsim.stat.analytic import conversion_moments
from cimsim.stat.measure import ErrorProbe

BASE = dict(w_bits=8, x_bits=8, nrow=128, signed_act=True, w_gran="per_channel", x_gran="per_token")


def _x(T, K, relu=False, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(T, K, generator=g)
    x[:, :4] *= 25                                   # outlier channels, as in LLMs
    return x.clamp_min(0) if relu else x


def _pair(kw, K=512, N=64, relu=False):
    """Statistical layer calibrated on one input set; bit-accurate twin with the same weights and ADC range."""
    torch.manual_seed(0)
    lin = torch.nn.Linear(K, N)
    cfg = {**BASE, **kw}
    st = CIMLinear.from_linear(lin, CIMConfig(mode="statistical", **cfg), name="L")
    st.start_calibration()
    st(_x(192, K, relu, 1))
    st.finish_calibration()
    ba = CIMLinear.from_linear(lin, CIMConfig(mode="bit_accurate", **cfg), name="L")
    ba.adc_fs = st.adc_fs
    return st, ba


def _measured(ba, K, relu=False):
    pr = ErrorProbe(max_samples_per_layer=10 ** 7)
    ba._probe = pr
    ba(_x(192, K, relu, 2))
    s = pr.samples()["L"]
    i, e = s["ideal"].double(), s["e"].double()
    return i, e


def test_conversion_moments_match_monte_carlo():
    torch.manual_seed(0)
    m = torch.tensor([0.0, 3.0, 40.0, 62.0, 63.0])
    for s, fs, upper, L in [(0.7, 63.0, 63.0, 63), (2.0, 85.0, 85.0, 255), (1.5, 200.0, 128.0, 63), (3.0, 128.0, 128.0, None)]:
        mean, var = conversion_moments(m, torch.full_like(m, s), upper, fs, L)
        x = (m.view(-1, 1) + s * torch.randn(len(m), 400_000, dtype=torch.float64)).clamp(0, upper)
        out = x if L is None else torch.round(x / (fs / L)) * (fs / L)
        assert torch.allclose(mean, out.mean(1), atol=0.02 * max(s, 1))
        assert torch.allclose(var, out.var(1), rtol=0.02, atol=1e-3)


def test_conversion_moments_noise_free_is_deterministic_adc():
    m = torch.arange(0, 129, dtype=torch.float64)
    mean, var = conversion_moments(m, torch.zeros_like(m), 128.0, 128.0, 255)
    D = 128 / 255
    assert torch.allclose(mean, torch.round(m / D) * D)
    assert float(var.abs().max()) == 0.0


@pytest.mark.parametrize("bits,gain", [(8, 2 * 128 / 255 - 1), (7, 128 / 127 - 1)])
def test_full_range_adc_is_a_gain_error(bits, gain):
    st, ba = _pair(dict(adc_bits=bits))
    assert st.stat_info["gain"] == pytest.approx(gain, rel=1e-6)
    assert st.stat_info["std"] < 1e-6
    i, e = _measured(ba, 512)
    assert float((i * e).sum() / (i * i).sum()) == pytest.approx(gain, rel=1e-3)


@pytest.mark.parametrize("kw,relu", [
    (dict(adc_bits=6, adc_range="calibrated", adc_range_gran="layer", rand_noise_sigma=0.25), False),
    (dict(adc_bits=6, adc_range="calibrated", adc_range_gran="layer", rand_noise_sigma=0.5, hybrid_levels=4), False),
    (dict(adc_bits=8, non_linear_sigma=2.0), False),
    (dict(adc_bits=8, rand_noise_sigma=0.2, act_enc=2), False),
    (dict(adc_bits=None, rand_noise_sigma=0.5), False),
    (dict(adc_bits=8, adc_range="calibrated", adc_range_gran="layer", rand_noise_sigma=0.5), True),  # post-ReLU: clipping at 0
])
def test_noise_std_matches_bit_accurate(kw, relu):
    st, ba = _pair(kw, relu=relu)
    i, e = _measured(ba, 512, relu)
    r = e - st.stat_info["gain"] * i
    assert st.stat_info["std"] == pytest.approx(float(r.std()), rel=0.05)
    assert abs(float(r.mean()) - st.stat_info["mean"]) < 4 * float(r.std()) / math.sqrt(r.numel())


def test_calibrated_lossless_has_no_error_model():
    st, _ = _pair(dict(adc_bits=6, adc_range="calibrated", adc_range_gran="layer"))
    assert st.stat_info["gain"] == 0.0 and st.stat_info["mean"] == 0.0 and st.stat_info["std"] == 0.0


def test_statistical_forward_seeded_counted_and_close_to_bit_accurate():
    # 0.25 %: many small flips, error close to Gaussian (at ~0.1 % it is rare large jumps and a
    # 4k-sample std is too noisy for this check; the 24k-sample validation matched to 3 %)
    kw = dict(adc_bits=6, adc_range="calibrated", adc_range_gran="layer", rand_noise_sigma=0.25)
    torch.manual_seed(0)
    lin = torch.nn.Linear(512, 64)
    outs, counters = [], []
    for mode in ("statistical", "statistical", "bit_accurate"):
        ctr = EventCounter()
        layer = CIMLinear.from_linear(lin, CIMConfig(mode=mode, **{**BASE, **kw}), name="L", counter=ctr)
        layer.start_calibration()
        layer(_x(64, 512))
        layer.finish_calibration()
        outs.append(layer(_x(64, 512, seed=3)))
        counters.append(ctr.total())
    assert torch.equal(outs[0], outs[1])                      # same seed -> same sample
    assert counters[0] == counters[2]                         # analytic event counts in both modes
    ideal = CIMLinear.from_linear(lin, CIMConfig(mode="fake_quant", **BASE), name="L")(_x(64, 512, seed=3))
    e_st, e_ba = (outs[0] - ideal).std(), (outs[2] - ideal).std()
    assert float(e_st / e_ba) == pytest.approx(1.0, rel=0.15)


def test_statistical_requires_calibration():
    layer = CIMLinear.from_linear(torch.nn.Linear(64, 8), CIMConfig(mode="statistical", **BASE), name="L")
    with pytest.raises(RuntimeError, match="statistical"):
        layer(torch.randn(4, 64))
