"""Build matched ASiM and cimsim versions of a CNN for side-by-side runs (A2).

Only nn.Conv2d layers (groups=1) are replaced; layers named in `keep_fp`
(stem conv, classifier) stay in floating point in both versions.
"""

from __future__ import annotations

import copy
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

from cimsim import CIMConfig, CIMConv2d

ROOT = Path(__file__).resolve().parents[1]


def asim_root() -> Path:
    for p in (ROOT / "third_party" / "ASiM", ROOT / "ASiM"):
        if (p / "module" / "asim_conv.py").exists():
            return p
    raise FileNotFoundError("ASiM not found at third_party/ASiM or ASiM/ "
                            "(git clone https://github.com/Keio-CSG/ASiM third_party/ASiM)")


def asim_conv_class():
    root = asim_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # ASiM's tools.common_tools imports torchvision/matplotlib; the conv module only needs setup_seed
    if "tools.common_tools" not in sys.modules:
        stub = types.ModuleType("tools.common_tools")
        stub.setup_seed = lambda seed=0: torch.manual_seed(seed)
        pkg = types.ModuleType("tools")
        pkg.common_tools = stub
        sys.modules.setdefault("tools", pkg)
        sys.modules["tools.common_tools"] = stub
    from module.asim_conv import ASiMConv2d
    return ASiMConv2d


def _replace(model: nn.Module, make, keep_fp: tuple[str, ...]) -> list[str]:
    names = [n for n, m in model.named_modules()
             if isinstance(m, nn.Conv2d) and m.groups == 1 and n not in keep_fp]
    for n in names:
        parent_name, _, child = n.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, make(n, getattr(parent, child)))
    return names


def build_cimsim(model: nn.Module, cfg: CIMConfig, keep_fp=("conv1",)) -> nn.Module:
    m = copy.deepcopy(model)
    _replace(m, lambda n, c: CIMConv2d.from_conv2d(c, cfg, n), keep_fp)
    return m.eval()


def build_asim(model: nn.Module, cfg: CIMConfig, keep_fp=("conv1",)) -> nn.Module:
    """cfg fields used: mode (fake_quant -> ASiM 'Inference', bit_accurate -> 'Simulation'), w_bits,
    x_bits, adc_bits, nrow, act_enc, signed_act, rand_noise_sigma, non_linear_sigma, hybrid_levels."""
    ASiMConv2d = asim_conv_class()
    mode = {"fake_quant": "Inference", "bit_accurate": "Simulation"}[cfg.mode]

    def make(name, c: nn.Conv2d):
        a = ASiMConv2d(c.in_channels, c.out_channels, c.kernel_size[0], stride=c.stride[0],
                       padding=c.padding[0], bias=c.bias is not None, wbit=cfg.w_bits, xbit=cfg.x_bits,
                       adc_prec=cfg.adc_bits if cfg.adc_bits is not None else 30, nrow=cfg.nrow,
                       rand_noise_sigma=cfg.rand_noise_sigma, non_linear_sigma=cfg.non_linear_sigma,
                       act_enc=cfg.act_enc, signed_act=cfg.signed_act,
                       hybrid_levels=cfg.hybrid_levels or None, mode=mode, device="cpu")
        with torch.no_grad():
            a.weight.copy_(c.weight)
            if c.bias is not None:
                a.bias.copy_(c.bias)
        return a

    m = copy.deepcopy(model)
    _replace(m, make, keep_fp)
    return m.eval()
