"""Part B per-action cost models (energy, latency, area). Every parameter comes
from config and is cited in docs/assumptions.md; no hardware numbers in code."""

from .adc import ADCSpec, adc_energy, crossover_enob
from .analog import AnalogArraySpec
from .digital import UNIT_OPS, DigitalSpec
from .scaling import ScalingRule

__all__ = ["ADCSpec", "adc_energy", "crossover_enob", "AnalogArraySpec", "DigitalSpec", "UNIT_OPS", "ScalingRule"]
