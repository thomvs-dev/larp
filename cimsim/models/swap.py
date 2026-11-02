"""Replace Linear layers in a model with CIMLinear according to a LayerConfigMap.

Handles torch nn.Linear and HuggingFace GPT-2 style Conv1D (weight [in, out]).
"""

from __future__ import annotations

import fnmatch

import torch.nn as nn

from cimsim.config import LayerConfigMap
from cimsim.core.layers import CIMLinear, CIMConv2d, _CIMBase
from cimsim.events import EventCounter

DEFAULT_EXCLUDE = ("lm_head", "*.lm_head", "score")


def _is_hf_conv1d(m: nn.Module) -> bool:
    return type(m).__name__ == "Conv1D" and hasattr(m, "nf") and m.weight.dim() == 2


def _set_submodule(root: nn.Module, name: str, new: nn.Module) -> None:
    parent_name, _, child = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child, new)


def swap_linears(
    model: nn.Module,
    cfg_map: LayerConfigMap,
    counter: EventCounter | None = None,
    exclude: tuple[str, ...] = DEFAULT_EXCLUDE,
    conv2d: bool = True,
) -> list[str]:
    """Replace layers in place. Returns the names of replaced layers.
    Layers whose name matches an `exclude` pattern are left untouched
    (the LM head is excluded by default; it is usually kept digital)."""
    targets = []
    for name, m in model.named_modules():
        if not name or any(fnmatch.fnmatch(name, p) for p in exclude):
            continue
        if isinstance(m, nn.Linear) or _is_hf_conv1d(m):
            targets.append((name, m))
        elif conv2d and isinstance(m, nn.Conv2d) and m.groups == 1:
            targets.append((name, m))

    replaced = []
    for name, m in targets:
        cfg = cfg_map.for_layer(name)
        if isinstance(m, nn.Linear):
            new = CIMLinear.from_linear(m, cfg, name, counter)
        elif isinstance(m, nn.Conv2d):
            new = CIMConv2d.from_conv2d(m, cfg, name, counter)
        else:
            new = CIMLinear.from_conv1d(m, cfg, name, counter)
        new.to(m.weight.device)
        _set_submodule(model, name, new)
        replaced.append(name)
    return replaced


def set_mode(model: nn.Module, **changes) -> None:
    """Change config fields on every CIM layer (e.g. set_mode(model, mode='fp'))."""
    for m in model.modules():
        if isinstance(m, _CIMBase):
            m.cfg = m.cfg.replace(**changes)
            m.requantize()
