"""Fractional ADC full scales (2^b - 1) / n (A5, findings F12): calibration candidates, fast vs reference
executor, statistical vs bit-accurate."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cimsim.config import CIMConfig  # noqa: E402
from cimsim.core.calib import candidate_full_scales, full_scale_from_histogram, mac_histogram  # noqa: E402
from cimsim.core.executor import cim_matmul_int  # noqa: E402
from cimsim.core.executor_fast import cim_matmul_int_fast  # noqa: E402
from test_stat_saturation import _errors, _pair  # noqa: E402


def test_candidates_include_exact_fractional_steps():
    c = candidate_full_scales(64, 7).tolist()
    assert 63.5 in c and abs(127 / 3 - min(c, key=lambda v: abs(v - 127 / 3))) < 1e-12
    assert all(float(v).is_integer() for v in candidate_full_scales(64, None).tolist())
    assert max(candidate_full_scales(128, 8).tolist()) == 128 and 127.5 in candidate_full_scales(128, 8).tolist()
    assert set(candidate_full_scales(128, 6).tolist()) >= {63.0, 31.5, 21.0, 128.0}


def _ints(T, K, N, seed, xmax=127):
    g = torch.Generator().manual_seed(seed)
    qx = torch.randint(-xmax, xmax + 1, (T, K), generator=g).double()
    qw = torch.randint(-127, 128, (N, K), generator=g).double()
    return qx, qw


@pytest.mark.parametrize("nrow,bits,expect", [(64, 7, 63.5), (128, 8, 127.5)])
def test_calibration_picks_exact_fractional_scale(nrow, bits, expect):
    """All MACs below (2^b - 1) / 2: the calibrated full scale is the half step (2^b - 1) / 2 and the
    readout equals the integer matmul (no integer full scale <= nrow reads these MACs exactly)."""
    cfg = CIMConfig(mode="bit_accurate", nrow=nrow, adc_bits=bits, adc_range="calibrated", adc_range_gran="layer")
    qx, qw = _ints(64, 2 * nrow, 16, 0, xmax=7)        # small codes: MACs per plane stay below nrow / 2 + few
    h = mac_histogram(qx, qw, cfg)
    hmax = int(torch.nonzero(h.sum(dim=(0, 1))).max())
    assert hmax < expect
    fs = full_scale_from_histogram(h, cfg)
    assert torch.all(fs == expect)
    for ex in (cim_matmul_int, cim_matmul_int_fast):
        y = ex(qx, qw, cfg, full_scale=fs)
        assert torch.equal(y, qx @ qw.t()), ex.__name__


@pytest.mark.parametrize("fs", [63.5, 127 / 3, 25.4, 12.7])
@pytest.mark.parametrize("noise", [0.0, 0.3])
def test_fast_matches_reference_with_fractional_scale(fs, noise):
    cfg = CIMConfig(mode="bit_accurate", nrow=64, adc_bits=7, adc_range="calibrated", rand_noise_sigma=noise)
    qx, qw = _ints(48, 200, 24, 1)
    full = torch.full((8, 8), float(fs), dtype=torch.float64)
    ref = cim_matmul_int(qx, qw, cfg, generator=torch.Generator().manual_seed(0), full_scale=full)
    fast = cim_matmul_int_fast(qx, qw, cfg, generator=torch.Generator().manual_seed(0), full_scale=full)
    ideal = qx @ qw.t()
    if noise == 0:
        assert float((fast - ref).abs().max() / ideal.abs().max()) < 1e-6
    else:                                              # different noise streams: compare error size
        r = float((fast - ideal).std() / (ref - ideal).std())
        assert 0.9 < r < 1.1, r


@pytest.mark.parametrize("fs,nrow", [(127 / 4, 128), (127 / 6, 64)])     # step 1/4, 1/6 MAC unit, saturating
@pytest.mark.parametrize("backend", ["fast", "reference"])
def test_statistical_exact_with_fractional_scale(monkeypatch, fs, nrow, backend):
    """Noise-free, ADC step 1/n MAC unit: statistical (exact saturation) = bit-accurate."""
    st, ba = _pair(monkeypatch, dict(adc_bits=7, adc_range="calibrated", nrow=nrow), fs, K=512, backend=backend)
    out, ideal, e = _errors(st, ba, 512)
    assert float((e.abs() > 0.5).double().mean()) > 0.01   # saturation present (fast backend: float32 noise ~1e-6 elsewhere)
    assert st.stat_info["std"] < 1e-6 and abs(st.stat_info["gain"]) < 1e-12
    diff = (out["st"] - out["ba"]).norm() / out["ba"].norm()
    assert diff < 1e-5, diff
