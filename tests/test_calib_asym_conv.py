"""Calibrated ADC range, asymmetric weights, CIMConv2d."""

import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from cimsim import CIMConfig, CIMConv2d, CIMLinear, EventCounter, calibrate_adc_range
from cimsim.core.bitplanes import recompose_weight, weight_planes
from cimsim.core.calib import full_scale_from_histogram, mac_histogram
from cimsim.core.executor import cim_matmul_int, nominal_full_scale
from cimsim.core.executor_fast import cim_matmul_int_fast
from cimsim.quant import quantize_weight, quantize_weight_asym, quantize_act


def rand_int(shape, bits, signed=True, seed=0):
    g = torch.Generator().manual_seed(seed)
    if signed:
        m = 2 ** (bits - 1) - 1
        return torch.randint(-m, m + 1, shape, generator=g).float()
    return torch.randint(0, 2 ** bits, shape, generator=g).float()


def rel(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


# ---------------- calibrated ADC range: executors ----------------

@pytest.mark.parametrize("enc", [None, 3])
def test_explicit_nominal_full_scale_is_unchanged(enc):
    qx, qw = rand_int((9, 300), 8, seed=1), rand_int((7, 300), 8, seed=2)
    cfg = CIMConfig(mode="bit_accurate", nrow=64, adc_bits=6, act_enc=enc)
    P = cfg.w_bits
    fs = nominal_full_scale(cfg).view(-1, 1).expand(-1, P)
    for fn in (cim_matmul_int, cim_matmul_int_fast):
        assert torch.equal(fn(qx, qw, cfg), fn(qx, qw, cfg, full_scale=fs))


@pytest.mark.parametrize("adc", [None, 4, 6, 8])
@pytest.mark.parametrize("enc,signed_w", [(None, True), (2, True), (None, False), (3, False)])
def test_fast_matches_reference_calibrated(adc, enc, signed_w):
    cfg = CIMConfig(mode="bit_accurate", nrow=64, adc_bits=adc, act_enc=enc,
                    w_quant="symmetric" if signed_w else "asymmetric", token_block=8)
    qx = rand_int((21, 200), 8, seed=3)
    qw = rand_int((13, 200), 8, signed=signed_w, seed=4)
    C = nominal_full_scale(cfg).numel()
    g = torch.Generator().manual_seed(5)
    nom = nominal_full_scale(cfg).view(C, 1).expand(C, cfg.w_bits)
    fs = torch.clamp(torch.round(nom * torch.rand(C, cfg.w_bits, generator=g, dtype=torch.float64) * 0.8 + 1), 1, None)
    ref = cim_matmul_int(qx, qw, cfg, full_scale=fs)
    fast = cim_matmul_int_fast(qx, qw, cfg, full_scale=fs)
    assert rel(fast, ref) < 1e-5


def test_calibrated_noise_statistics_match():
    qx, qw = rand_int((48, 256), 8, seed=6), rand_int((32, 256), 8, seed=7)
    cfg = CIMConfig(mode="bit_accurate", nrow=64, adc_bits=8, rand_noise_sigma=1.0, non_linear_sigma=1.0)
    fs = torch.full((8, 8), 40.0, dtype=torch.float64)
    ideal = qx.double() @ qw.double().t()
    e_ref = torch.cat([cim_matmul_int(qx, qw, cfg, torch.Generator().manual_seed(s), full_scale=fs) - ideal for s in range(4)])
    e_fast = torch.cat([cim_matmul_int_fast(qx, qw, cfg, torch.Generator().manual_seed(9 + s), full_scale=fs) - ideal
                        for s in range(4)])
    assert abs(e_fast.std() / e_ref.std() - 1) < 0.05


def test_mac_histogram_matches_brute_force():
    cfg = CIMConfig(mode="bit_accurate", nrow=16, act_enc=2, w_quant="asymmetric")
    qx, qw = rand_int((5, 40), 8, seed=8), rand_int((3, 40), 8, signed=False, seed=9)
    from cimsim.core.bitplanes import input_cycles, input_digits
    cycles = input_cycles(None, 8, True, 2)
    d = input_digits(qx, 8, True, cycles)                    # [C, T, K]
    planes, _ = weight_planes(qw, 8, signed=False)           # [P, N, K]
    h = mac_histogram(qx, qw, cfg)
    M = h.shape[-1] - 1
    brute = torch.zeros_like(h)
    for k0 in range(0, 40, 16):
        k1 = min(40, k0 + 16)
        m = torch.einsum("ctk,pnk->ctpn", d[:, :, k0:k1].double(), planes[:, :, k0:k1].double()).long()
        for c in range(m.shape[0]):
            for p in range(m.shape[2]):
                brute[c, p] += torch.bincount(m[c, :, p].reshape(-1), minlength=M + 1)
    assert torch.equal(h, brute)


def test_full_scale_from_histogram_granularity():
    cfg = CIMConfig(mode="bit_accurate", nrow=64, adc_range="calibrated", adc_range_method="percentile",
                    adc_range_pct=100.0, adc_range_gran="cycle_plane")
    qx, qw = rand_int((30, 128), 8, seed=10), rand_int((20, 128), 8, seed=11)
    h = mac_histogram(qx, qw, cfg)
    fs = full_scale_from_histogram(h, cfg)
    # pct 100 = max observed MAC per (cycle, plane)
    for c in range(8):
        for p in range(8):
            assert fs[c, p] == max(1, int(torch.nonzero(h[c, p]).max()))
    fs_layer = full_scale_from_histogram(h, cfg.replace(adc_range_gran="layer"))
    assert torch.all(fs_layer == fs_layer[0, 0])
    assert torch.all(fs_layer <= 64) and torch.all(fs >= 1)


def test_calibration_reduces_error_with_outlier_activations():
    """F3 scenario: signed per-token activations with outlier channels, 6-bit ADC."""
    torch.manual_seed(0)
    x = torch.randn(64, 512)
    x[:, [5, 77, 300]] *= torch.tensor([20.0, 40.0, 60.0])
    lin = nn.Linear(512, 64, bias=False)
    ref = CIMLinear.from_linear(lin, CIMConfig(mode="fake_quant"), "l")(x)
    full = CIMLinear.from_linear(lin, CIMConfig(mode="bit_accurate", nrow=128, adc_bits=6), "l")(x)
    cal = CIMLinear.from_linear(lin, CIMConfig(mode="bit_accurate", nrow=128, adc_bits=6, adc_range="calibrated",
                                               adc_range_pct=99.9), "l")
    calibrate_adc_range(cal, lambda: cal(x))
    out = cal(x)
    err_full = ((full - ref).norm() / ref.norm()).item()
    err_cal = ((out - ref).norm() / ref.norm()).item()
    assert err_cal < err_full


@pytest.mark.parametrize("gran", ["cycle_plane", "cycle", "layer"])
def test_mse_calibration_never_worse_than_full_on_calibration_data(gran):
    torch.manual_seed(1)
    x = torch.randn(32, 384)
    x[:, [3, 100]] *= 30.0
    lin = nn.Linear(384, 48, bias=False)
    ref = CIMLinear.from_linear(lin, CIMConfig(mode="fake_quant"), "l")(x)
    for adc in (5, 6, 8):
        full = CIMLinear.from_linear(lin, CIMConfig(mode="bit_accurate", nrow=128, adc_bits=adc), "l")(x)
        cal = CIMLinear.from_linear(lin, CIMConfig(mode="bit_accurate", nrow=128, adc_bits=adc,
                                                   adc_range="calibrated", adc_range_gran=gran), "l")
        calibrate_adc_range(cal, lambda: cal(x))
        e_full = ((full - ref).norm() / ref.norm()).item()
        e_cal = ((cal(x) - ref).norm() / ref.norm()).item()
        assert e_cal <= e_full + 1e-6, (adc, e_cal, e_full)


def test_mse_calibration_finds_integer_step():
    """When observed MAC values stay below 2^b - 1, MSE calibration picks FS = 2^b - 1 (ADC step = 1 MAC
    unit), which reads every integer MAC exactly."""
    from cimsim.core.calib import full_scale_from_histogram
    cfg = CIMConfig(mode="bit_accurate", nrow=128, adc_bits=6, adc_range="calibrated")
    hist = torch.zeros(8, 8, 129, dtype=torch.int64)
    hist[:, :, :50] = 1                                   # MAC values 0..49 observed
    fs = full_scale_from_histogram(hist, cfg)
    assert torch.all(fs == 63)


def test_calibration_api_and_state():
    lin = nn.Linear(100, 10)
    layer = CIMLinear.from_linear(lin, CIMConfig(mode="bit_accurate", nrow=32, adc_bits=6, adc_range="calibrated"), "l")
    x = torch.randn(4, 100)
    with pytest.raises(RuntimeError):
        layer(x)
    names = calibrate_adc_range(layer, lambda: layer(x))
    assert names == ["l"]
    assert layer.adc_fs.shape == (8, 8)
    assert "adc_fs" in layer.state_dict()
    y = layer(x)
    assert y.shape == (4, 10) and torch.isfinite(y).all()


# ---------------- asymmetric weights ----------------

def test_unsigned_weight_planes_roundtrip():
    q = rand_int((5, 17), 4, signed=False, seed=12)
    planes, meta = weight_planes(q, 4, signed=False)
    assert all(s == 1 for _, s in meta)
    assert torch.equal(recompose_weight(planes, meta), q.double())


def test_asym_fake_quant_matches_dequantized_weights():
    torch.manual_seed(1)
    lin = nn.Linear(64, 16)
    x = torch.randn(5, 64)
    cfg = CIMConfig(mode="fake_quant", w_bits=4, w_quant="asymmetric")
    y = CIMLinear.from_linear(lin, cfg, "l")(x)
    q, step, zero = quantize_weight_asym(lin.weight.detach(), 4, "per_channel")
    w_hat = (q - zero) * step
    qx, sx = quantize_act(x, 8, "per_token", True)
    x_hat = qx * sx / 127
    ref = x_hat @ w_hat.t() + lin.bias.detach()
    assert torch.allclose(y, ref, atol=1e-5)


def test_asym_bit_accurate_ideal_equals_fake_quant():
    torch.manual_seed(2)
    lin = nn.Linear(300, 20)
    x = torch.randn(7, 300)
    a = CIMLinear.from_linear(lin, CIMConfig(mode="fake_quant", w_bits=4, w_quant="asymmetric"), "l")(x)
    for backend in ("reference", "fast"):
        b = CIMLinear.from_linear(lin, CIMConfig(mode="bit_accurate", w_bits=4, w_quant="asymmetric", adc_bits=None,
                                                 nrow=64, backend=backend), "l")(x)
        assert torch.allclose(a, b, rtol=1e-4, atol=1e-5)


def test_asym_4bit_beats_symmetric_on_skewed_weights():
    torch.manual_seed(3)
    w = torch.rand(32, 256) * 0.1 - 0.02         # mostly positive weights
    lin = nn.Linear(256, 32, bias=False)
    lin.weight.data.copy_(w)
    x = torch.randn(16, 256)
    ref = x @ w.t()
    e = {}
    for wq in ("symmetric", "asymmetric"):
        y = CIMLinear.from_linear(lin, CIMConfig(mode="fake_quant", w_bits=4, x_bits=8, w_quant=wq), "l")(x)
        e[wq] = ((y - ref).norm() / ref.norm()).item()
    assert e["asymmetric"] < e["symmetric"]


# ---------------- CIMConv2d ----------------

def test_conv_fp_and_fake_quant():
    torch.manual_seed(4)
    conv = nn.Conv2d(16, 8, 3, stride=2, padding=1)
    x = torch.relu(torch.randn(2, 16, 9, 9))
    ref = conv(x)
    y_fp = CIMConv2d.from_conv2d(conv, CIMConfig(mode="fp"), "c")(x)
    assert torch.allclose(y_fp, ref, atol=1e-6)
    y_fq = CIMConv2d.from_conv2d(conv, CIMConfig(mode="fake_quant", signed_act=False, w_gran="per_tensor",
                                                 x_gran="per_tensor"), "c")(x)
    assert ((y_fq - ref).norm() / ref.norm()) < 0.02


@pytest.mark.parametrize("backend", ["reference", "fast"])
def test_conv_bit_accurate_ideal_equals_fake_quant(backend):
    torch.manual_seed(5)
    conv = nn.Conv2d(20, 6, 3, padding=1, bias=False)
    x = torch.relu(torch.randn(2, 20, 6, 6))
    base = dict(signed_act=False, w_gran="per_tensor", x_gran="per_tensor", nrow=64)
    a = CIMConv2d.from_conv2d(conv, CIMConfig(mode="fake_quant", **base), "c")(x)
    b = CIMConv2d.from_conv2d(conv, CIMConfig(mode="bit_accurate", adc_bits=None, backend=backend, **base), "c")(x)
    assert torch.allclose(a, b, rtol=1e-4, atol=1e-5)


def test_conv_chunks_and_events():
    conv = nn.Conv2d(64, 4, 3, padding=1)
    layer = CIMConv2d.from_conv2d(conv, CIMConfig(mode="fake_quant", nrow=128), "c", EventCounter())
    ch = layer._chunks()
    assert ch[0] == (0, 14 * 9) and ch[-1] == (56 * 9, 64 * 9) and len(ch) == 5
    layer(torch.randn(1, 64, 5, 5))
    tot = layer.counter.total()
    assert tot["array_cycles"] == 25 * 5 * 8          # T=25 pixels, 5 chunks, 8 cycles
    assert tot["macs"] == 25 * 576 * 4


# ---------------- CIMConv2d vs ASiMConv2d ----------------

_ROOT = Path(__file__).resolve().parents[1]
_ASIM = next((p for p in (_ROOT / "third_party" / "ASiM", _ROOT / "ASiM")
              if (p / "module" / "asim_conv.py").exists()), None)
asim = pytest.mark.skipif(_ASIM is None, reason="ASiM not cloned")


def _asim_conv_cls():
    if str(_ASIM) not in sys.path:
        sys.path.insert(0, str(_ASIM))
    if "tools.common_tools" not in sys.modules:
        stub = types.ModuleType("tools.common_tools")
        stub.setup_seed = lambda seed=0: torch.manual_seed(seed)
        pkg = types.ModuleType("tools")
        pkg.common_tools = stub
        sys.modules.setdefault("tools", pkg)
        sys.modules["tools.common_tools"] = stub
    from module.asim_conv import ASiMConv2d
    return ASiMConv2d


def _pair(cin, cout, k, stride, pad, nrow, adc, enc, signed, backend="fast"):
    torch.manual_seed(6)
    ASiMConv2d = _asim_conv_cls()
    conv = nn.Conv2d(cin, cout, k, stride=stride, padding=pad, bias=True)
    a = ASiMConv2d(cin, cout, k, stride=stride, padding=pad, bias=True, wbit=8, xbit=8, adc_prec=adc, nrow=nrow,
                   act_enc=enc, signed_act=signed, mode="Simulation", device="cpu")
    with torch.no_grad():
        a.weight.copy_(conv.weight)
        a.bias.copy_(conv.bias)
    cfg = CIMConfig(mode="bit_accurate", w_bits=8, x_bits=8, adc_bits=adc, nrow=nrow, act_enc=enc,
                    signed_act=signed, w_gran="per_tensor", x_gran="per_tensor", backend=backend)
    ours = CIMConv2d.from_conv2d(conv, cfg, "c")
    x = torch.randn(2, cin, 8, 8)
    if not signed:
        x = torch.relu(x)
    with torch.no_grad():
        return a(x), ours(x), conv, x


@asim
@pytest.mark.parametrize("backend", ["reference", "fast"])
@pytest.mark.parametrize("cin,k,stride,pad,nrow,adc,enc,signed", [
    (16, 3, 1, 1, 128, 8, None, False),
    (32, 3, 2, 1, 128, 6, None, False),
    (64, 3, 1, 1, 255, 7, None, False),
    (16, 3, 1, 1, 128, 7, 2, False),
    (32, 3, 1, 1, 128, 6, 3, False),
    (16, 1, 2, 0, 128, 6, None, False),
    (16, 3, 1, 1, 128, 6, None, True),
])
def test_conv_matches_asim(cin, k, stride, pad, nrow, adc, enc, signed, backend):
    ref, ours, _, _ = _pair(cin, 8, k, stride, pad, nrow, adc, enc, signed, backend)
    assert rel(ours, ref) < 1e-4


@asim
def test_asim_conv_drops_chunks_when_channel_groups_exceed_w_update():
    """ASiM computes w_update = ceil(C*k*k / nrow) but splits into ceil(C / floor(nrow/(k*k))) channel
    groups; when the second is larger, the last groups are skipped. C=64, k=3, nrow=64: 10 groups, 9 used.
    Our conv uses all groups, so it matches the exact convolution while ASiM does not."""
    ref, ours, conv, x = _pair(64, 8, 3, 1, 1, 64, 30, None, False)      # 30-bit ADC = effectively ideal
    exact = CIMConv2d.from_conv2d(conv, CIMConfig(mode="fake_quant", signed_act=False, w_gran="per_tensor",
                                                  x_gran="per_tensor"), "c")(x)
    assert rel(ours, exact) < 1e-4
    assert rel(ref, exact) > 1e-2
