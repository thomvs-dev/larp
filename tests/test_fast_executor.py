"""Fast (CPU-optimised) executor vs reference executor."""

import itertools

import pytest
import torch

from cimsim.config import CIMConfig
from cimsim.core.executor import cim_matmul_int
from cimsim.core.executor_fast import _adc_scale, _ref_codes, cim_matmul_int_fast


def rand_int(shape, bits, signed=True, seed=0):
    g = torch.Generator().manual_seed(seed)
    if signed:
        m = 2 ** (bits - 1) - 1
        return torch.randint(-m, m + 1, shape, generator=g).float()
    return torch.randint(0, 2 ** bits, shape, generator=g).float()


CASES = [
    # (w_bits, x_bits, signed, act_enc, nrow, adc_bits)
    (8, 8, True, None, 128, 6),
    (8, 8, True, None, 64, 4),
    (8, 8, True, 2, 64, 6),
    (8, 8, True, 3, 128, 7),
    (8, 8, False, None, 128, 5),
    (8, 8, False, 3, 64, 6),
    (4, 4, True, None, 256, 8),
    (4, 4, True, 2, 32, 3),
    (6, 6, True, 4, 100, 7),     # nrow not a power of two
    (8, 8, True, None, 128, None),
    (8, 8, True, None, 128, 8),  # lossless at 8 bits? (FS=128 -> 255 levels: not identity)
    (8, 8, True, None, 255, 8),  # FS = 255 = 2^8-1 -> lossless shortcut
]


@pytest.mark.parametrize("wb,xb,signed,enc,nrow,adc", CASES)
def test_fast_matches_reference_noise_free(wb, xb, signed, enc, nrow, adc):
    qx = rand_int((37, 300), xb, signed, seed=1)
    qw = rand_int((23, 300), wb, True, seed=2)
    cfg = CIMConfig(mode="bit_accurate", w_bits=wb, x_bits=xb, signed_act=signed, act_enc=enc,
                    nrow=nrow, adc_bits=adc, token_block=16)
    ref = cim_matmul_int(qx, qw, cfg)
    fast = cim_matmul_int_fast(qx, qw, cfg)
    scale = ref.abs().max()
    assert ((fast - ref).abs().max() / scale).item() < 1e-5


def test_adc_scale_reproduces_reference_codes():
    # every (FS, adc) pair used by common configs either has an exact float32 scale or falls back to LUT
    for nrow, L, b in itertools.product([16, 32, 64, 100, 128, 256, 512], [1, 3, 7, 15], range(2, 11)):
        fs = nrow * L
        a = _adc_scale(fs, b)
        if a is not None:
            m = torch.arange(fs + 1, dtype=torch.float32)
            assert torch.equal(torch.round(m * torch.tensor(a, dtype=torch.float32)), _ref_codes(fs, b).float())


def test_lossless_shortcut_exact():
    qx = rand_int((9, 500), 8, seed=3)
    qw = rand_int((11, 500), 8, seed=4)
    cfg = CIMConfig(mode="bit_accurate", nrow=255, adc_bits=8)
    assert torch.equal(cim_matmul_int_fast(qx, qw, cfg), qx.double() @ qw.double().t())


@pytest.mark.parametrize("kw", [
    dict(rand_noise_sigma=1.0),
    dict(non_linear_sigma=2.0),
    dict(rand_noise_sigma=0.5, non_linear_sigma=1.0, hybrid_levels=3),
    dict(rand_noise_sigma=1.0, act_enc=3),
])
def test_fast_noise_statistics_match_reference(kw):
    qx = rand_int((64, 256), 8, seed=5)
    qw = rand_int((48, 256), 8, seed=6)
    cfg = CIMConfig(mode="bit_accurate", nrow=64, adc_bits=8, **kw)
    ideal = qx.double() @ qw.double().t()
    e_ref = torch.cat([cim_matmul_int(qx, qw, cfg, torch.Generator().manual_seed(s)) - ideal for s in range(4)])
    e_fast = torch.cat([cim_matmul_int_fast(qx, qw, cfg, torch.Generator().manual_seed(100 + s)) - ideal for s in range(4)])
    assert abs(e_fast.std() / e_ref.std() - 1) < 0.05
    assert abs(e_fast.mean() - e_ref.mean()) < 0.05 * e_ref.std()


def test_fast_noise_seeded():
    qx = rand_int((8, 128), 8, seed=7)
    qw = rand_int((8, 128), 8, seed=8)
    cfg = CIMConfig(mode="bit_accurate", nrow=64, adc_bits=8, rand_noise_sigma=1.0)
    a = cim_matmul_int_fast(qx, qw, cfg, torch.Generator().manual_seed(1))
    b = cim_matmul_int_fast(qx, qw, cfg, torch.Generator().manual_seed(1))
    assert torch.equal(a, b)
