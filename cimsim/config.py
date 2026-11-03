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
W_QUANT = ("symmetric", "asymmetric")
ADC_RANGES = ("full", "calibrated")
ADC_RANGE_GRANS = ("layer", "cycle", "cycle_plane")


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
    w_quant: str = "symmetric"           # "symmetric" (two's complement planes, ASiM) or
                                         # "asymmetric" (unsigned planes + zero point, corrected digitally)
    adc_range: str = "full"              # "full": ADC full scale = nrow * L_c (ASiM);
                                         # "calibrated": full scale set from calibration data
    adc_range_gran: str = "cycle_plane"  # calibrated range shared per "layer", per input "cycle",
                                         # or per (cycle, weight plane) = "cycle_plane"
    adc_range_method: str = "mse"        # "mse": full scale minimising squared readout error on the
                                         #   calibration data (significance-weighted when shared);
                                         # "percentile": adc_range_pct of calibration MAC values
    adc_range_pct: float = 99.99         # used by method "percentile"
    stat_saturation: str = "exact"       # mode 'statistical': "exact" = ADC saturation computed exactly
                                         # on the sparse conversions that can saturate (F8);
                                         # "analytic" = folded into the Gaussian model M3 (A3 v1)
    stat_noise: str = "auto"             # mode 'statistical': "auto" = analog noise sampled as discrete ADC code
                                         # flips where flips are rare per output (F9), Gaussian elsewhere;
                                         # "gaussian" = Gaussian for every (cycle, plane) (v2)
    stat_flip_lambda_max: float = 4.0    # "auto": discrete when expected code flips per output < this
    stat_sat_margin: Optional[int] = 8   # 'exact': only columns whose calibration MAC maximum came within
                                         # this many MAC units of the saturation threshold are computed;
                                         # None = every column allowed by the popcount bound (strict, slow)

    @property
    def w_signed(self) -> bool:
        return self.w_quant == "symmetric"

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
        if self.w_quant not in W_QUANT:
            raise ValueError(f"w_quant must be one of {W_QUANT}")
        if self.adc_range not in ADC_RANGES:
            raise ValueError(f"adc_range must be one of {ADC_RANGES}")
        if self.adc_range_gran not in ADC_RANGE_GRANS:
            raise ValueError(f"adc_range_gran must be one of {ADC_RANGE_GRANS}")
        if self.adc_range_method not in ("mse", "percentile"):
            raise ValueError("adc_range_method must be 'mse' or 'percentile'")
        if self.stat_noise not in ("auto", "gaussian"):
            raise ValueError("stat_noise must be 'auto' or 'gaussian'")
        if self.stat_flip_lambda_max <= 0:
            raise ValueError("stat_flip_lambda_max must be > 0")
        if self.stat_sat_margin is not None and self.stat_sat_margin < 0:
            raise ValueError("stat_sat_margin must be >= 0 or None")
        if self.stat_saturation not in ("exact", "analytic"):
            raise ValueError("stat_saturation must be 'exact' or 'analytic'")
        if not 0.0 < self.adc_range_pct <= 100.0:
            raise ValueError("adc_range_pct must be in (0, 100]")

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
