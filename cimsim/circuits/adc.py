"""ADC energy per conversion (Part B, B0).

Two architectures:

- ``bound``: Murmann's two empirical limits on conversion energy, taken as a
  lower envelope of published ADCs (SAR-like converters at low to moderate
  speed):
    * Walden:   E_W = FOM_W * 2**ENOB                       (2x per bit)
    * Schreier: E_S = (1/2) * 10**((SNDR - FOM_S) / 10)    (4x per bit, thermal)
  with SNDR = 6.02 * ENOB + 1.76 dB, from FOM_S = SNDR + 10 log10(fs / (2 P)).
  E = max(E_W, E_S). At 4-8 b the Walden term dominates for any realistic pair
  (FOM_W, FOM_S).
- ``flash``: (2**b - 1) comparators fire per conversion, plus a fixed encoder /
  reference cost: E = (2**b - 1) * e_comparator + e_fixed.

No number is built in: every parameter comes from config, and each config value
must cite its source in docs/assumptions.md (Part B section). Speed (throughput
per ADC) is not modelled yet; the bound model is only valid below the speed
"corner" of the survey data (Murmann, "Energy limits in A/D converters", 2012:
corner around 100-300 MHz).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

ARCHS = ("bound", "flash")


@dataclass(frozen=True)
class ADCSpec:
    arch: str
    bits: int
    enob: float | None = None          # default: bits (ideal converter)
    fom_walden: float | None = None    # J / conversion-step   (arch 'bound')
    fom_schreier_db: float | None = None  # dB                 (arch 'bound')
    e_comparator: float | None = None  # J per comparator decision (arch 'flash')
    e_fixed: float = 0.0               # J per conversion, any arch (reference, encoder, sampling switch)

    def __post_init__(self):
        if self.arch not in ARCHS:
            raise ValueError(f"arch must be one of {ARCHS}, got {self.arch!r}")
        if self.bits < 1:
            raise ValueError("bits must be >= 1")
        if self.enob is not None and not (0 < self.enob <= self.bits):
            raise ValueError("enob must be in (0, bits]")
        if self.arch == "bound" and (self.fom_walden is None or self.fom_schreier_db is None):
            raise ValueError("arch 'bound' needs fom_walden and fom_schreier_db")
        if self.arch == "flash" and self.e_comparator is None:
            raise ValueError("arch 'flash' needs e_comparator")
        if self.e_fixed < 0:
            raise ValueError("e_fixed must be >= 0")

    @classmethod
    def from_dict(cls, d: dict, bits: int | None = None) -> "ADCSpec":
        """Build from a config dict. ``bits`` overrides d['bits'] (per-layer ADC bits)."""
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known - {"source"}
        if unknown:
            raise KeyError(f"unknown ADC spec fields: {sorted(unknown)}")
        kw = {k: v for k, v in d.items() if k in known}
        if bits is not None:
            kw["bits"] = bits
            if kw.get("enob") is not None and kw["enob"] > bits:
                kw["enob"] = float(bits)
        return cls(**kw)

    def with_bits(self, bits: int) -> "ADCSpec":
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d["bits"] = bits
        if d["enob"] is not None and d["enob"] > bits:
            d["enob"] = float(bits)
        return ADCSpec(**d)

    @property
    def effective_bits(self) -> float:
        return float(self.bits if self.enob is None else self.enob)

    def walden_energy(self) -> float:
        return self.fom_walden * 2.0 ** self.effective_bits

    def schreier_energy(self) -> float:
        sndr = 6.02 * self.effective_bits + 1.76
        return 0.5 * 10.0 ** ((sndr - self.fom_schreier_db) / 10.0)

    def energy_per_conversion(self) -> float:
        """Joules per conversion."""
        if self.arch == "bound":
            core = max(self.walden_energy(), self.schreier_energy())
        else:
            core = (2 ** self.bits - 1) * self.e_comparator
        return core + self.e_fixed


def adc_energy(conversions: int, spec: ADCSpec) -> float:
    """Total ADC energy (J) for a number of conversions at this spec.

    Use the event counter's ``adc_conversions`` (all cycles), not
    ``adc_conversions_analog``: in the hybrid scheme the digital cycles are
    still ADC-quantized (docs/assumptions.md #6)."""
    if conversions < 0:
        raise ValueError("conversions must be >= 0")
    return conversions * spec.energy_per_conversion()


def crossover_enob(fom_walden: float, fom_schreier_db: float) -> float:
    """ENOB above which the Schreier (thermal) term exceeds the Walden term."""
    # FOM_W 2^n = 0.5 * 10^((6.02 n + 1.76 - FOM_S)/10)
    # log10: log10(FOM_W) + n log10 2 = log10 0.5 + 0.602 n + 0.176 - FOM_S/10
    a = math.log10(fom_walden) - math.log10(0.5) - 0.176 + fom_schreier_db / 10.0
    return a / (0.602 - math.log10(2.0))
