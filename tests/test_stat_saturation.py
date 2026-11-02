"""Exact sparse saturation in mode 'statistical' (cimsim/stat/saturation.py, findings F8)."""

import pytest
import torch

import cimsim.core.layers as layers_mod
from cimsim.config import CIMConfig
from cimsim.core.layers import CIMLinear, CIMConv2d
from cimsim.stat.measure import ErrorProbe

BASE = dict(w_bits=8, x_bits=8, nrow=128, signed_act=True, w_gran="per_channel", x_gran="per_token")


def _x(T, K, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(T, K, generator=g)
    x[:, :4] *= 25                       # outlier channels: small codes, dense sign-extension planes (F3)
    return x


def _pair(monkeypatch, kw, fs_value=None, K=512, N=64, backend="fast"):
    """Statistical layer + bit-accurate twin with the same weights and ADC range. fs_value forces
    the calibrated full scale (to create saturation on synthetic data)."""
    if fs_value is not None:
        orig = layers_mod.full_scale_from_histogram
        monkeypatch.setattr(layers_mod, "full_scale_from_histogram",
                            lambda h, cfg: torch.full_like(orig(h, cfg), float(fs_value)))
    torch.manual_seed(0)
    lin = torch.nn.Linear(K, N)
    cfg = {**BASE, **kw}
    st = CIMLinear.from_linear(lin, CIMConfig(mode="statistical", **cfg), name="L")
    st.start_calibration()
    st(_x(192, K, 1))
    st.finish_calibration()
    ba = CIMLinear.from_linear(lin, CIMConfig(mode="bit_accurate", backend=backend, **cfg), name="L")
    ba.adc_fs = st.adc_fs
    return st, ba


def _errors(st, ba, K, seed=2):
    x = _x(192, K, seed)
    out = {}
    for name, layer in (("st", st), ("ba", ba)):
        pr = ErrorProbe(max_samples_per_layer=10 ** 7)
        # the probe only hooks bit-accurate layers; for the statistical layer compare outputs directly
        layer._probe = pr if name == "ba" else None
        out[name] = layer(x).double()
    s = pr.samples()["L"]
    return out, s["ideal"].double(), s["e"].double()


@pytest.mark.parametrize("kw,fs,K", [
    (dict(adc_bits=5, adc_range="calibrated"), 31, 512),              # step 1 MAC, frequent saturation
    (dict(adc_bits=5, adc_range="calibrated"), 31, 600),              # K not a multiple of nrow
    (dict(adc_bits=6, adc_range="calibrated", act_enc=2), 63, 512),   # bit-parallel, L_c = 3
    (dict(adc_bits=5, adc_range="calibrated", w_quant="asymmetric"), 31, 512),
    (dict(adc_bits=6, adc_range="calibrated", nrow=64), 21, 512),     # step 1/3 MAC unit, nrow 64
])
@pytest.mark.parametrize("backend", ["fast", "reference"])
def test_noise_free_saturation_matches_bit_accurate(monkeypatch, kw, fs, K, backend):
    st, ba = _pair(monkeypatch, kw, fs, K=K, backend=backend)
    out, ideal, e = _errors(st, ba, K)
    assert float((e != 0).double().mean()) > 0.01                     # saturation is present
    diff = (out["st"] - out["ba"]).norm() / out["ba"].norm()
    if (2 ** kw["adc_bits"] - 1) % fs == 0:                           # step = 1/n MAC unit: rounding-free, exact
        assert st.stat_info["std"] < 1e-6 and abs(st.stat_info["gain"]) < 1e-12
        assert diff < 1e-5, diff                                      # fast backend: float32 ADC scale (<= 1e-5 vs reference)
    else:                                                             # rounding part is the Gaussian M3 term
        x = _x(192, K, 2)
        y_id = CIMLinear(ba.weight, ba.bias, ba.cfg.replace(mode="fake_quant"), name="F")(x).double()
        r = float((out["st"] - y_id).std() / (out["ba"] - y_id).std())
        assert 0.9 < r < 1.1, r


def test_exact_saturation_beats_independent_model(monkeypatch):
    """The v1 model (stat_saturation 'analytic') overestimates correlated saturation error (F8)."""
    st, ba = _pair(monkeypatch, dict(adc_bits=5, adc_range="calibrated", stat_saturation="analytic"), 31)
    _, ideal, e = _errors(st, ba, 512)
    assert st.stat_info["std"] > 1.5 * float(e.std())


def test_full_range_adc_has_no_saturation_work(monkeypatch):
    st, ba = _pair(monkeypatch, dict(adc_bits=8, adc_range="full"))
    st(_x(64, 512, 3))
    assert st.sat_stats["macs_exact"] == 0 and st.sat_stats["macs_total"] > 0


def test_conv_custom_chunks_saturation_exact(monkeypatch):
    orig = layers_mod.full_scale_from_histogram
    monkeypatch.setattr(layers_mod, "full_scale_from_histogram",
                        lambda h, cfg: torch.full_like(orig(h, cfg), 15.0))
    torch.manual_seed(0)
    conv = torch.nn.Conv2d(16, 8, 3, padding=1)
    cfg = dict(BASE, nrow=64, adc_bits=4, adc_range="calibrated", x_gran="per_token")
    st = CIMConv2d.from_conv2d(conv, CIMConfig(mode="statistical", **cfg), name="C")
    x = torch.randn(4, 16, 8, 8)
    st.start_calibration(); st(x); st.finish_calibration()
    ba = CIMConv2d.from_conv2d(conv, CIMConfig(mode="bit_accurate", **cfg), name="C")
    ba.adc_fs = st.adc_fs
    x2 = torch.randn(4, 16, 8, 8)
    y_st, y_ba = st(x2).double(), ba(x2).double()
    assert st.stat_info["std"] < 1e-6
    assert (y_st - y_ba).norm() / y_ba.norm() < 1e-6


@pytest.mark.parametrize("sigma", [0.5, 2.0])
def test_noise_with_saturation_error_std(monkeypatch, sigma):
    """Noise + saturation: statistical error std within 10 % of bit-accurate."""
    kw = dict(adc_bits=5, adc_range="calibrated", rand_noise_sigma=sigma)
    st, ba = _pair(monkeypatch, kw, 31)
    x = _x(192, 512, 2)
    errs = {}
    for name, layer in (("st", st), ("ba", ba)):
        fq = CIMLinear(layer.weight, layer.bias, layer.cfg.replace(mode="fake_quant"), name="F")
        y_ideal = fq(x).double()
        errs[name] = layer(x).double() - y_ideal
    r = float(errs["st"].std() / errs["ba"].std())
    assert 0.9 < r < 1.1, r


def test_margin_filter_matches_strict_bound(monkeypatch):
    """Calibration column filter (stat_sat_margin) gives the same output as the strict popcount bound
    on held-out data, with less exact work."""
    kw = dict(adc_bits=5, adc_range="calibrated")
    st, _ = _pair(monkeypatch, kw, 31)
    strict, _ = _pair(monkeypatch, {**kw, "stat_sat_margin": None}, 31)
    x = _x(192, 512, 2)
    y1, y2 = st(x).double(), strict(x).double()
    assert torch.equal(y1, y2)
    assert st.sat_stats["macs_exact"] <= strict.sat_stats["macs_exact"]
    assert 0.0 < st.stat_info["sat_cols_frac"] <= 1.0
