"""Calibration histogram cache (cimsim/calib_cache.py): cached calibration must give the same ADC full
scales, statistical parameters, saturation columns and model outputs as a fresh calibration, and the
cache must miss whenever the histograms could differ."""

import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cimsim import calibrate_adc_range  # noqa: E402
from cimsim.config import LayerConfigMap  # noqa: E402
from cimsim.core.attention import swap_attention  # noqa: E402
from cimsim.core.layers import _CIMBase  # noqa: E402
from cimsim.models.swap import swap_linears  # noqa: E402
from test_swap import tiny  # noqa: E402

BASE = {"mode": "statistical", "nrow": 32, "adc_bits": 6, "adc_range": "calibrated", "adc_range_gran": "layer"}
IDS = torch.randint(2, 90, (1, 48), generator=torch.Generator().manual_seed(1))
TEST_IDS = torch.randint(2, 90, (2, 24), generator=torch.Generator().manual_seed(2))


def build(cfg: dict, attention: bool = True, arch: str = "llama", perturb: float = 0.0):
    torch.manual_seed(0)
    model = tiny(arch).eval()
    if perturb:
        with torch.no_grad():
            next(model.parameters()).add_(perturb)
    m = LayerConfigMap.from_dict(cfg)
    swap_linears(model, m)
    if attention:
        swap_attention(model, m)
    return model


def calibrate(model, cache_dir, data_key="d0"):
    calls = []

    def run():
        calls.append(1)
        model(IDS)

    report = {}
    calibrate_adc_range(model, run, cache_dir=cache_dir, data_key=data_key, report=report)
    report["ran"] = len(calls)
    return report


def state(model):
    out = {}
    for m in model.modules():
        if isinstance(m, _CIMBase):
            out[m.name] = [m.adc_fs, m.stat_params, m.sat_cols, getattr(m, "_cal_K", None), m._flips, m.stat_info]
    with torch.no_grad():
        out["logits"] = model(TEST_IDS).logits
    return out


def same(x, y) -> bool:
    if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
        return isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor) and torch.equal(x, y)
    if isinstance(x, dict):
        return isinstance(y, dict) and x.keys() == y.keys() and all(same(x[k], y[k]) for k in x)
    if isinstance(x, (list, tuple)):
        return isinstance(y, (list, tuple)) and len(x) == len(y) and all(same(u, v) for u, v in zip(x, y))
    return x == y


def assert_same(a, b):
    assert a.keys() == b.keys()
    for k in a:
        assert same(a[k], b[k]), k


def fresh(cfg, **kw):
    model = build(cfg, **kw)
    rep = calibrate(model, None)
    assert rep["cache"] == "off" and rep["ran"] == 1
    return state(model)


def test_hit_equals_fresh_and_skips_forward(tmp_path):
    cfg = {"default": {**BASE, "rand_noise_sigma": 0.05}}
    m1 = build(cfg)
    r1 = calibrate(m1, tmp_path)
    assert r1["cache"] == "miss" and r1["ran"] == 1 and r1["misses"] > 0
    m2 = build(cfg)
    r2 = calibrate(m2, tmp_path)
    assert r2["cache"] == "hit" and r2["ran"] == 0 and r2["hits"] == r1["misses"]
    ref = fresh(cfg)
    assert_same(state(m1), ref)
    assert_same(state(m2), ref)


@pytest.mark.parametrize("change", [{"adc_bits": 5}, {"rand_noise_sigma": 0.1}, {"adc_range_method": "percentile"},
                                    {"hybrid_levels": 2}, {"seed": 3}, {"stat_sat_margin": None}])
def test_histogram_independent_settings_hit(tmp_path, change):
    calibrate(build({"default": BASE}), tmp_path)
    cfg = {"default": {**BASE, **change}}
    m = build(cfg)
    r = calibrate(m, tmp_path)
    assert r["cache"] == "hit" and r["ran"] == 0
    assert_same(state(m), fresh(cfg))


def test_bit_accurate_reuses_statistical_histograms(tmp_path):
    calibrate(build({"default": BASE}, attention=False), tmp_path)
    cfg = {"default": {**BASE, "mode": "bit_accurate"}}
    m = build(cfg, attention=False)
    r = calibrate(m, tmp_path)
    assert r["cache"] == "hit"
    assert_same(state(m), fresh(cfg, attention=False))


def test_nrow_change_on_one_role_is_partial(tmp_path):
    calibrate(build({"default": BASE}), tmp_path)
    cfg = {"default": BASE, "layers": {"*.mlp.down_proj": {"nrow": 16}}}
    m = build(cfg)
    r = calibrate(m, tmp_path)
    assert r["cache"] == "partial" and r["misses"] == 2 and r["ran"] == 1      # 2 decoder layers
    assert_same(state(m), fresh(cfg))
    r2 = calibrate(build(cfg), tmp_path)
    assert r2["cache"] == "hit"


@pytest.mark.parametrize("what", ["w_bits", "x_bits_upstream", "data", "weights", "fp_layer"])
def test_changes_that_alter_histograms_miss(tmp_path, what):
    calibrate(build({"default": BASE}), tmp_path)
    cfg, kw, data = {"default": BASE}, {}, "d0"
    if what == "w_bits":
        cfg = {"default": {**BASE, "w_bits": 6}}
    elif what == "x_bits_upstream":                     # only layer 0 changes; layer 1 inputs change too
        cfg = {"default": BASE, "layers": {"model.layers.0.*": {"x_bits": 6}}}
    elif what == "data":
        data = "d1"
    elif what == "weights":
        kw = {"perturb": 1e-3}
    elif what == "fp_layer":                            # an fp layer upstream changes later inputs
        cfg = {"default": BASE, "layers": {"model.layers.0.mlp.down_proj": {"mode": "fp"}}}
    m = build(cfg, **kw)
    r = calibrate(m, tmp_path, data_key=data)
    assert r["cache"] == "miss" and r["hits"] == 0


def test_cache_off_without_data_key(tmp_path):
    m = build({"default": BASE})
    rep = {}
    calibrate_adc_range(m, lambda: m(IDS), cache_dir=tmp_path, report=rep)
    assert rep["cache"] == "off" and not any(tmp_path.iterdir())


def test_source_change_misses(tmp_path, monkeypatch):
    import cimsim.calib_cache as cc
    calibrate(build({"default": BASE}), tmp_path)
    monkeypatch.setattr(cc, "source_fingerprint", lambda: "changed-code")
    r = calibrate(build({"default": BASE}), tmp_path)
    assert r["cache"] == "miss"
