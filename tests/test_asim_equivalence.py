"""Compare against ASiM's ASiMLinear (Simulation mode) with zero noise.

Runs only when ASiM is cloned at third_party/ASiM or ASiM/ (repo root):
    git clone https://github.com/Keio-CSG/ASiM third_party/ASiM
"""

import sys
from pathlib import Path

import pytest
import torch

from cimsim.config import CIMConfig
from cimsim.core.layers import CIMLinear

_ROOT = Path(__file__).resolve().parents[1]
ASIM = next((p for p in (_ROOT / "third_party" / "ASiM", _ROOT / "ASiM")
             if (p / "module" / "asim_linear.py").exists()), _ROOT / "third_party" / "ASiM")
pytestmark = pytest.mark.skipif(not (ASIM / "module" / "asim_linear.py").exists(),
                                reason="ASiM not cloned at third_party/ASiM or ASiM/")


def _asim_linear_cls():
    if str(ASIM) not in sys.path:
        sys.path.insert(0, str(ASIM))
    # ASiM's tools.common_tools imports torchvision/matplotlib; asim_linear only
    # needs setup_seed from it, so provide a light stub.
    if "tools.common_tools" not in sys.modules:
        import types
        stub = types.ModuleType("tools.common_tools")
        stub.setup_seed = lambda seed=0: torch.manual_seed(seed)
        pkg = types.ModuleType("tools")
        pkg.common_tools = stub
        sys.modules.setdefault("tools", pkg)
        sys.modules["tools.common_tools"] = stub
    from module.asim_linear import ASiMLinear
    return ASiMLinear


CASES = [
    # (wbit, xbit, adc, nrow, act_enc, signed)
    (4, 4, 6, 64, None, True),
    (8, 8, 8, 128, None, True),
    (8, 8, 5, 64, None, True),
    (8, 8, 7, 64, 2, True),
    (8, 8, 7, 64, 3, True),
    (8, 8, 6, 64, None, False),
    (8, 8, 7, 64, 3, False),
]


@pytest.mark.parametrize("backend", ["reference", "fast"])
@pytest.mark.parametrize("wbit,xbit,adc,nrow,enc,signed", CASES)
def test_matches_asim_zero_noise(wbit, xbit, adc, nrow, enc, signed, backend):
    ASiMLinear = _asim_linear_cls()
    torch.manual_seed(0)
    K, N, B = 300, 24, 6
    w = torch.randn(N, K)
    b = torch.randn(N)
    x = torch.randn(B, K)
    if not signed:
        x = x.abs()

    ref_mod = ASiMLinear(K, N, bias=True, wbit=wbit, xbit=xbit, adc_prec=adc, nrow=nrow,
                         act_enc=enc, signed_act=signed, layer="fc", mode="Simulation", device="cpu")
    with torch.no_grad():
        ref_mod.weight.copy_(w)
        ref_mod.bias.copy_(b)
        ref = ref_mod(x)

    lin = torch.nn.Linear(K, N)
    with torch.no_grad():
        lin.weight.copy_(w)
        lin.bias.copy_(b)
    cfg = CIMConfig(mode="bit_accurate", w_bits=wbit, x_bits=xbit, adc_bits=adc, nrow=nrow,
                    act_enc=enc, signed_act=signed, w_gran="per_tensor", x_gran="per_tensor",
                    backend=backend)
    ours = CIMLinear.from_linear(lin, cfg, "l")(x)

    scale = ref.abs().max()
    assert ((ours - ref).abs().max() / scale).item() < 1e-4
