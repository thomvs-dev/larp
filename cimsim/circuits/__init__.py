"""Part B per-action cost models (energy, latency, area). Every parameter comes
from config and is cited in docs/assumptions.md; no hardware numbers in code."""

from .adc import ADCSpec, adc_energy, crossover_enob

__all__ = ["ADCSpec", "adc_energy", "crossover_enob"]
