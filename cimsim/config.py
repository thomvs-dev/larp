"""Configuration for CIM layers.

A CIMConfig describes how one layer is executed. A LayerConfigMap holds a
default config plus per-layer overrides matched by glob pattern on the
module name (e.g. "*.mlp.down_proj").
"""

from __future__ import annotations

import dataclasses
import fnmatch
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import yaml

MODES = ("fp", "fake_quant", "bit_accurate", "statistical")
GRANULARITIES_W = ("per_tensor", "per_channel")
GRANULARITIES_X = ("per_tensor", "per_token")


@dataclass
class CIMConfig:
    mode: str = "fake_quant"
    w_bits: int = 8
    x_bits: int = 8
    w_gran: str = "per_channel"
    x_gran: str = "per_token"
    signed_act: bool = True
    act_enc: Optional[int] = None        # bit-parallel encoding bits; None = bit-serial
    nrow: int = 256                      # rows summed per ADC conversion
    adc_bits: Optional[int] = 8          # None = no ADC (ideal readout)
    rand_noise_sigma: float = 0.0        # % of full scale (ASiM convention)
    non_linear_sigma: float = 0.0        # % (ASiM convention)
    hybrid_levels: int = 0               # MSB cycle levels computed without analog noise
    token_block: int = 128               # tokens per inner block (memory control)
    seed: int = 0
    backend: str = "fast"                # "fast" (CPU-optimised) or "reference" (ASiM-exact loop structure)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.w_gran not in GRANULARITIES_W:
            raise ValueError(f"w_gran must be one of {GRANULARITIES_W}")
        if self.x_gran not in GRANULARITIES_X:
            raise ValueError(f"x_gran must be one of {GRANULARITIES_X}")
        if self.w_bits < 2 or self.x_bits < 2:
            raise ValueError("w_bits and x_bits must be >= 2")
        if self.nrow < 1:
            raise ValueError("nrow must be >= 1")
        if self.adc_bits is not None and self.adc_bits < 1:
            raise ValueError("adc_bits must be >= 1 or None")
        if self.act_enc is not None:
            limit = self.x_bits - 1 if self.signed_act else self.x_bits
            if not 1 <= self.act_enc <= limit:
                raise ValueError(
                    f"act_enc must be in [1, {limit}] for "
                    f"{'signed' if self.signed_act else 'unsigned'} activations"
                )
        if self.hybrid_levels < 0:
            raise ValueError("hybrid_levels must be >= 0")
        if self.backend not in ("fast", "reference"):
            raise ValueError("backend must be 'fast' or 'reference'")

    def replace(self, **changes: Any) -> "CIMConfig":
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LayerConfigMap:
    """Default config plus ordered glob overrides. Later patterns win."""

    default: CIMConfig = field(default_factory=CIMConfig)
    overrides: list[tuple[str, dict]] = field(default_factory=list)

    def for_layer(self, name: str) -> CIMConfig:
        changes: dict = {}
        for pattern, delta in self.overrides:
            if fnmatch.fnmatch(name, pattern):
                changes.update(delta)
        return self.default.replace(**changes) if changes else self.default

    @classmethod
    def from_dict(cls, d: dict) -> "LayerConfigMap":
        default = CIMConfig(**d.get("default", {}))
        layers = d.get("layers", {}) or {}
        overrides = [(pat, dict(delta)) for pat, delta in layers.items()]
        # validate overrides early
        for _, delta in overrides:
            default.replace(**delta)
        return cls(default=default, overrides=overrides)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LayerConfigMap":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f) or {})
