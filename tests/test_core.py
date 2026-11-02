"""Core correctness tests (GATE A1)."""

import math

import pytest
import torch

from cimsim.config import CIMConfig, LayerConfigMap
from cimsim.core.bitplanes import (input_cycles, input_digits, recompose_input,
                                   recompose_weight, weight_planes)
from cimsim.core.executor import analytic_counts, cim_matmul_int
from cimsim.core.layers import CIMLinear
from cimsim.events import EventCounter


def rand_int(shape, bits, signed=True, seed=0):
    g = torch.Generator().manual_seed(seed)
    if signed:
        m = 2 ** (bits - 1) - 1
        return torch.randint(-m, m + 1, shape, generator=g).float()
    return torch.randint(0, 2 ** bits, shape, generator=g).float()


# ---------- bit decomposition ----------

@pytest.mark.parametrize("bits", [2, 4, 8])
def test_weight_planes_roundtrip(bits):
    q = rand_int((7, 33), bits)
    planes, meta = weight_planes(q, bits)
    assert set(planes.unique().tolist()) <= {0.0, 1.0}
    assert torch.equal(recompose_weight(planes, meta), q.double())


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("signed", [True, False])
@pytest.mark.parametrize("enc", [None, 1, 2, 3])
def test_input_digits_roundtrip(bits, signed, enc):
    q = rand_int((5, 40), bits, signed)
    cyc = input_cycles(q, bits, signed, enc)
    d = input_digits(q, bits, signed, cyc)
    for di, c in zip(d, cyc):
        assert di.min() >= 0 and di.max() <= c.levels
    assert torch.equal(recompose_input(d, cyc), q.double())


def test_cycle_counts_match_asim():
    # signed bit-parallel: ceil((x-1)/enc) + 1 cycles; unsigned: ceil(x/enc)
    assert len(input_cycles(None, 8, True, 3)) == math.ceil(7 / 3) + 1
    assert len(input_cycles(None, 8, False, 3)) == math.ceil(8 / 3)
    assert len(input_cycles(None, 8, True, None)) == 8


# ---------- GATE A1: ideal config is exact ----------

@pytest.mark.parametrize("T,K,N", [(3, 64, 5), (4, 300, 17), (2, 1000, 9)])
@pytest.mark.parametrize("nrow", [16, 64, 256])
@pytest.mark.parametrize("enc", [None, 2, 3])
@pytest.mark.parametrize("signed", [True, False])
def test_ideal_exact(T, K, N, nrow, enc, signed):
    wb, xb = 8, 8
    if enc is not None and enc > (xb - 1 if signed else xb):
        pytest.skip()
    cfg = CIMConfig(mode="bit_accurate", w_bits=wb, x_bits=xb, nrow=nrow, adc_bits=None,
                    act_enc=enc, signed_act=signed, token_block=2)
    qx = rand_int((T, K), xb, signed, seed=1)
    qw = rand_int((N, K), wb, True, seed=2)
    y = cim_matmul_int(qx, qw, cfg)
    ref = qx.double() @ qw.double().t()
    assert torch.equal(y, ref)


def test_adc_error_decreases_with_bits():
    qx = rand_int((8, 512), 8, seed=3)
    qw = rand_int((32, 512), 8, seed=4)
    ref = qx.double() @ qw.double().t()
    errs = []
    for b in [3, 5, 7, 9, 12]:
        cfg = CIMConfig(mode="bit_accurate", nrow=128, adc_bits=b)
        errs.append((cim_matmul_int(qx, qw, cfg) - ref).abs().mean().item())
    assert all(a >= b for a, b in zip(errs, errs[1:])), errs
    assert errs[-1] < errs[0]


def test_noise_seeded():
    qx = rand_int((4, 128), 8, seed=5)
    qw = rand_int((8, 128), 8, seed=6)
    cfg = CIMConfig(mode="bit_accurate", nrow=64, adc_bits=8, rand_noise_sigma=1.0, non_linear_sigma=1.0)
    a = cim_matmul_int(qx, qw, cfg, torch.Generator().manual_seed(1))
    b = cim_matmul_int(qx, qw, cfg, torch.Generator().manual_seed(1))
    c = cim_matmul_int(qx, qw, cfg, torch.Generator().manual_seed(2))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_hybrid_reduces_noise_error():
    qx = rand_int((16, 256), 8, seed=7)
    qw = rand_int((16, 256), 8, seed=8)
    ref = qx.double() @ qw.double().t()
    base = dict(mode="bit_accurate", nrow=64, adc_bits=None, rand_noise_sigma=2.0)
    e0 = (cim_matmul_int(qx, qw, CIMConfig(**base), torch.Generator().manual_seed(0)) - ref).pow(2).mean()
    e1 = (cim_matmul_int(qx, qw, CIMConfig(**base, hybrid_levels=4), torch.Generator().manual_seed(0)) - ref).pow(2).mean()
    assert e1 < e0


# ---------- layer + events ----------

def test_layer_modes_close_to_fp():
    torch.manual_seed(0)
    lin = torch.nn.Linear(256, 64)
    x = torch.randn(2, 10, 256)
    ref = lin(x)
    for mode in ["fp", "fake_quant", "bit_accurate"]:
        cfg = CIMConfig(mode=mode, adc_bits=None, nrow=64)
        y = CIMLinear.from_linear(lin, cfg, "l")(x)
        rel = ((y - ref).norm() / ref.norm()).item()
        assert rel < 0.02, (mode, rel)


def test_fake_quant_equals_bit_accurate_ideal():
    torch.manual_seed(0)
    lin = torch.nn.Linear(300, 40)
    x = torch.randn(6, 300)
    y_fq = CIMLinear.from_linear(lin, CIMConfig(mode="fake_quant"), "l")(x)
    y_ba = CIMLinear.from_linear(lin, CIMConfig(mode="bit_accurate", adc_bits=None, nrow=64), "l")(x)
    assert torch.allclose(y_fq, y_ba, rtol=1e-4, atol=1e-5)


def test_event_counts():
    cfg = CIMConfig(mode="fake_quant", w_bits=8, x_bits=8, nrow=128)
    c = analytic_counts(T=10, K=300, N=7, cfg=cfg)
    n_chunks = 3
    assert c["macs"] == 10 * 300 * 7
    assert c["adc_conversions"] == 10 * n_chunks * 8 * 8 * 7
    assert c["array_cycles"] == 10 * n_chunks * 8
    assert c["adc_conversions_analog"] == c["adc_conversions"]
    c_h = analytic_counts(10, 300, 7, cfg.replace(hybrid_levels=2))
    assert c_h["adc_conversions_analog"] < c["adc_conversions_analog"]

    counter = EventCounter()
    lin = torch.nn.Linear(300, 7)
    layer = CIMLinear.from_linear(lin, cfg, "l0", counter)
    layer(torch.randn(10, 300))
    layer(torch.randn(10, 300))
    tot = counter.total()
    assert tot["macs"] == 2 * c["macs"]
    assert tot["weight_bits_written"] == 300 * 7 * 8   # counted once


def test_config_overrides():
    m = LayerConfigMap.from_dict({
        "default": {"mode": "bit_accurate", "adc_bits": 6},
        "layers": {"*.fc2": {"adc_bits": 8}, "model.layers.0.*": {"mode": "fp"}},
    })
    assert m.for_layer("model.layers.3.fc1").adc_bits == 6
    assert m.for_layer("model.layers.3.fc2").adc_bits == 8
    assert m.for_layer("model.layers.0.fc2").mode == "fp"
    with pytest.raises(ValueError):
        CIMConfig(mode="nope")
    with pytest.raises(ValueError):
        CIMConfig(x_bits=4, act_enc=4, signed_act=True)
