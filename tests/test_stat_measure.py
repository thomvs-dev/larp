"""A3 error probe (cimsim/stat/measure.py)."""

import json

import pytest
import torch

from cimsim.config import CIMConfig
from cimsim.core.layers import CIMLinear
from cimsim.quant import quantize_act
from cimsim.stat.measure import ErrorProbe


def _layer(**kw):
    torch.manual_seed(0)
    lin = torch.nn.Linear(256, 48)
    cfg = CIMConfig(**{"mode": "bit_accurate", "w_bits": 8, "x_bits": 8, "nrow": 128, "signed_act": True,
                       "w_gran": "per_channel", "x_gran": "per_token", **kw})
    return CIMLinear.from_linear(lin, cfg, name="blk.0.fc")


def _x():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(40, 256, generator=g)
    x[:, 3] *= 20                                          # one outlier channel, as in LLMs
    return x


@pytest.mark.parametrize("backend", ["fast", "reference"])
def test_probe_does_not_change_output(backend):
    x = _x()
    a = _layer(adc_bits=6, backend=backend, rand_noise_sigma=1.0)
    b = _layer(adc_bits=6, backend=backend, rand_noise_sigma=1.0)
    probe = ErrorProbe()
    b._probe = probe
    assert torch.equal(a(x), b(x))
    assert "blk.0.fc" in probe.layers


def test_lossless_config_has_zero_error():
    # nrow 255, ADC 8 b, bit-serial: one ADC step per MAC unit -> exact readout
    layer = _layer(nrow=255, adc_bits=8)
    probe = ErrorProbe()
    layer._probe = probe
    layer(_x())
    s = probe.summary()["blk.0.fc"]
    assert s["frac_exact"] == 1.0 and s["std"] == 0.0 and s["rel_err"] == 0.0


def test_error_matches_direct_computation():
    layer = _layer(adc_bits=5)
    x = _x()
    probe = ErrorProbe(max_samples_per_layer=10**6)
    layer._probe = probe
    layer(x)
    qw, _, _ = layer._weights()
    qx, _ = quantize_act(x, 8, "per_token", True)
    from cimsim.core.executor_fast import cim_matmul_int_fast
    y = cim_matmul_int_fast(qx, qw, layer.cfg).double()
    e = y - qx.double() @ qw.double().t()
    s = probe.summary()["blk.0.fc"]
    assert s["elements"] == e.numel()
    assert s["mean"] == pytest.approx(float(e.mean()), rel=1e-6, abs=1e-9)
    assert s["std"] == pytest.approx(float(e.std(unbiased=False)), rel=1e-6)
    assert s["rel_err"] == pytest.approx(float(e.norm() / (qx.double() @ qw.double().t()).norm()), rel=1e-6)
    assert s["std"] > 0 and s["frac_exact"] < 1
    json.dumps(s)                                          # summary is JSON-serialisable


def test_noise_error_is_unbiased_and_grows_with_sigma():
    x = _x()
    stds = []
    for sig in (0.5, 2.0):                                    # percent of full scale
        layer = _layer(nrow=255, adc_bits=8, rand_noise_sigma=sig)    # lossless ADC: error is noise only
        probe = ErrorProbe()
        layer._probe = probe
        layer(x)
        s = probe.summary()["blk.0.fc"]
        assert abs(s["bias_over_std"]) < 0.1
        stds.append(s["std"])
    assert stds[1] > 2 * stds[0]


def test_attach_tiny_hf_model():
    transformers = pytest.importorskip("transformers")
    from cimsim.config import LayerConfigMap
    from cimsim.models.swap import swap_linears
    torch.manual_seed(0)
    cfg = transformers.OPTConfig(hidden_size=64, num_hidden_layers=2, ffn_dim=128, num_attention_heads=4,
                                 vocab_size=100, max_position_embeddings=64, word_embed_proj_dim=64)
    model = transformers.OPTForCausalLM(cfg).eval()
    swap_linears(model, LayerConfigMap(default=CIMConfig(mode="bit_accurate", nrow=64, adc_bits=6)))
    probe = ErrorProbe()
    names = probe.attach(model)
    assert len(names) == 12
    with torch.no_grad():
        model(torch.randint(0, 100, (1, 16)))
    assert set(probe.summary()) == set(names)
    smp = probe.samples()[names[0]]
    assert smp["tok_e_ms"].numel() == 16
    ErrorProbe.detach(model)
    assert all(getattr(m, "_probe", None) is None for m in model.modules())
