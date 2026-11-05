"""CIM attention matmuls (QKᵀ, P·V) on tiny HF models (A4)."""

import math

import pytest
import torch

transformers = pytest.importorskip("transformers")

from cimsim.config import CIMConfig, LayerConfigMap  # noqa: E402
from cimsim.core.attention import CIMMatMul, swap_attention  # noqa: E402
from cimsim.events import EventCounter  # noqa: E402
from cimsim.models.swap import set_mode, swap_linears  # noqa: E402
from test_swap import tiny  # noqa: E402  (pytest puts tests/ on sys.path)

ARCHS = ["opt", "gpt2", "llama", "qwen2"]


def _logits(model, ids):
    with torch.no_grad():
        return model(ids).logits


@pytest.mark.parametrize("arch", ARCHS)
def test_fp_attention_matches_eager(arch):
    model = tiny(arch).eval()
    model.config._attn_implementation = "eager"
    ids = torch.randint(0, 100, (2, 16))
    ref = _logits(model, ids)
    names = swap_attention(model, LayerConfigMap(default=CIMConfig(mode="fp")))
    assert len(names) == 4 and all(n.endswith((".qk", ".sv")) for n in names)
    assert torch.allclose(_logits(model, ids), ref, atol=1e-5)


@pytest.mark.parametrize("arch", ARCHS)
def test_ideal_bit_accurate_equals_fake_quant(arch):
    """Ideal readout (no ADC, no noise): bit-accurate attention matmuls equal the integer matmul."""
    model = tiny(arch).eval()
    ids = torch.randint(0, 100, (1, 16))
    swap_attention(model, LayerConfigMap(default=CIMConfig(mode="fake_quant", nrow=8)))
    fq = _logits(model, ids)
    set_mode(model, mode="bit_accurate", adc_bits=None)
    assert torch.allclose(_logits(model, ids), fq, atol=1e-4)
    ref = tiny(arch).eval()
    ref.config._attn_implementation = "eager"
    r = _logits(ref, ids)
    assert ((fq - r).norm() / r.norm()) < 0.05                 # 8-bit Q, K, P, V close to FP


def test_sv_value_scale_folded_into_probabilities():
    """P·V with per-position V scales folded into P equals the float product up to 8-bit rounding."""
    torch.manual_seed(0)
    P = torch.softmax(torch.randn(32, 40), -1)
    V = torch.randn(40, 16) * torch.linspace(0.1, 5, 40).unsqueeze(1)   # very different scales per position
    m = CIMMatMul(CIMConfig(mode="fake_quant", nrow=16), "sv")
    out = m.matmul(P, V.t())                                      # kv_gran auto: values per position
    ref = P @ V
    assert float((out - ref).norm() / ref.norm()) < 0.02


def test_events_count_only_causal_pairs_and_kv_writes():
    model = tiny("llama").eval()                                # 2 layers, 4 heads, 2 KV heads, D 16
    T = 16
    counter = EventCounter()
    swap_attention(model, LayerConfigMap(default=CIMConfig(mode="fake_quant", nrow=8, adc_bits=6)), counter)
    _logits(model, torch.randint(0, 100, (1, T)))
    rec = counter.per_layer["model.layers.0.self_attn.qk"]
    pairs = T * (T + 1) // 2                                    # causal (query, key) pairs per head
    H, Hkv, D, P, C = 4, 2, 16, 8, 8
    assert rec["macs"] == H * pairs * D
    assert rec["adc_conversions"] == H * pairs * math.ceil(D / 8) * C * P
    assert rec["kv_bits_written"] == Hkv * T * D * 8
    sv = counter.per_layer["model.layers.0.self_attn.sv"]
    assert sv["macs"] == H * pairs * D
    chunks = sum(math.ceil((t + 1) / 8) for t in range(T))
    assert sv["adc_conversions"] == H * chunks * C * P * D
    assert sv["kv_bits_written"] == Hkv * T * D * 8


def test_statistical_attention_with_calibration_runs_and_matches_bit_accurate_noise_free():
    from cimsim import calibrate_adc_range
    model = tiny("qwen2").eval()
    ids = torch.randint(0, 100, (1, 32))
    cfg = {"mode": "statistical", "nrow": 16, "adc_bits": 4, "adc_range": "calibrated", "adc_range_gran": "layer"}
    m = LayerConfigMap.from_dict({"default": cfg})
    swap_linears(model, m)
    swap_attention(model, m)
    names = calibrate_adc_range(model, lambda: model(ids))
    assert len(names) == 14 + 4
    st = _logits(model, ids)
    set_mode(model, mode="bit_accurate")
    ba = _logits(model, ids)
    assert torch.isfinite(st).all()
    assert float((st - ba).norm() / ba.norm()) < 1e-3          # noise-free: exact paths agree


def test_policy_roles_for_attention():
    model = tiny("qwen2")
    m = LayerConfigMap.from_dict({"default": {"mode": "fake_quant"}, "policy": {"qk": "dcim", "sv": "digital"}})
    swap_attention(model, m)
    att = model.model.layers[0].self_attn
    assert att.cim_qk.cfg.macro == "dcim" and att.cim_sv.cfg.macro == "digital"
    assert att.cim_sv.cfg.signed_act is False and att.cim_qk.cfg.signed_act is True


def test_key_scale_per_channel_folded_into_query():
    """Keys with a few large channels: per-channel key scales (folded into Q) reduce the error a little."""
    torch.manual_seed(0)
    Q = torch.randn(32, 64)
    K = torch.randn(48, 64)
    K[:, :2] *= 200                                               # outlier channels (like Qwen keys)
    ref = Q @ K.t()
    err = {}
    for g in ("token", "channel"):
        m = CIMMatMul(CIMConfig(mode="fake_quant", nrow=64, kv_gran=g), "qk")
        out = m.matmul(Q, K)
        small = Q[:, 2:] @ K[:, 2:].t()                           # contribution of the normal channels
        err[g] = float((out - ref).norm() / small.norm())
    # folding moves the outlier channels into the (also quantized) query, so the gain is limited
    assert err["channel"] < err["token"]
