"""Analog SRAM CIM array cost model (Part B, B0).

One *conversion* is one column read out for one input cycle and one chunk of
`nrow` rows (event counter `adc_conversions`). Its energy is

    nrow * (e_wl_cell * wl_pulses + e_bl_cell * bl_activity    array, data-dependent
            + e_cell_fixed)                                     array, data-independent
    + e_col_fixed                                              column reset / precharge / sampling
    + E_ADC(bits)                                              cimsim.circuits.adc
    + acc_bits * e_add_bit                                     digital shift-add
    + out_bits * e_reg_bit                                     output register

and one *array cycle* (one chunk of rows driven with one input digit, event
counter `array_cycles`) adds `nrow * in_bits * e_reg_bit` for the input
register.

- `e_wl_cell`: C V^2 of one cell's share of its input line (read wordline, or
  the x / xb lines of a charge-domain cell), per pulse. `wl_pulses` is the mean
  number of pulses per row and cycle: the input bit density for bit-serial
  inputs (0..1), the mean input digit for pulse-count inputs (Sinangil JSSC
  2021: up to 15 pulses for a 4-b input).
- `e_bl_cell`: C V^2 of one cell's share of the bit line / its compute
  capacitor. `bl_activity` (0..1) is the fraction of that charge moved per
  conversion: for bit-line discharge it is MAC / nrow (energy grows with the
  MAC result, Sinangil Fig. 13a); for a charge-domain cell it is the fraction of
  cells whose capacitor is switched.

`e_wl_cell`, `e_bl_cell` (and `e_col_fixed` where a macro reports a breakdown)
are the parameters fitted per macro in B1. Everything else is cited. No number
is built in. ADC sharing (`adc_share` columns per ADC) serialises conversions:
it changes latency and area, not the number of conversions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

from .adc import ADCSpec


@dataclass(frozen=True)
class AnalogArraySpec:
    nrow: int                     # rows summed per conversion
    adc: ADCSpec
    e_wl_cell: float              # J per (cell, input pulse)
    e_bl_cell: float              # J per cell at bl_activity = 1
    e_cell_fixed: float = 0.0     # J per (cell, conversion), independent of data (clocking, reset of the cell)
    e_col_fixed: float = 0.0      # J per conversion, independent of data and nrow
    e_add_bit: float = 0.0        # J per adder bit (shift-add)
    e_reg_bit: float = 0.0        # J per register bit (input / output buffers)
    acc_bits: int = 0             # shift-add accumulator width
    out_bits: int = 0             # bits latched per conversion (default: ADC bits)
    in_bits: int = 1              # bits latched per row and array cycle
    adc_share: int = 1            # columns per ADC
    t_array: float = 0.0          # s: input drive + bit-line settle of one array cycle
    t_adc: float = 0.0            # s per conversion
    area_cell: float = 0.0        # m^2 per bit cell, compute capacitor included
    area_adc: float = 0.0         # m^2 per ADC
    area_col_periph: float = 0.0  # m^2 per column (shift-add, registers, column switches)
    area_row_periph: float = 0.0  # m^2 per row (wordline driver, input register)
    vdd: float | None = None      # calibration supply (documentation; scaling is explicit, circuits.scaling)
    node_nm: float | None = None

    def __post_init__(self):
        if self.nrow < 1 or self.adc_share < 1 or self.in_bits < 0 or self.acc_bits < 0 or self.out_bits < 0:
            raise ValueError("nrow, adc_share must be >= 1; bit widths >= 0")
        for name in ("e_wl_cell", "e_bl_cell", "e_cell_fixed", "e_col_fixed", "e_add_bit", "e_reg_bit", "t_array", "t_adc",
                     "area_cell", "area_adc", "area_col_periph", "area_row_periph"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")

    @classmethod
    def from_dict(cls, d: dict) -> "AnalogArraySpec":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known - {"source", "fit"}
        if unknown:
            raise KeyError(f"unknown analog array spec fields: {sorted(unknown)}")
        kw = {k: v for k, v in d.items() if k in known}
        if isinstance(kw.get("adc"), dict):
            kw["adc"] = ADCSpec.from_dict(kw["adc"])
        return cls(**kw)

    # --- energy --------------------------------------------------------------
    @staticmethod
    def _check_data(wl_pulses: float, bl_activity: float) -> None:
        if wl_pulses < 0:
            raise ValueError("wl_pulses must be >= 0")
        if not 0.0 <= bl_activity <= 1.0:
            raise ValueError("bl_activity must be in [0, 1]")

    def conversion_breakdown(self, wl_pulses: float, bl_activity: float, adc_bits: int | None = None) -> dict:
        """J per conversion, by component."""
        self._check_data(wl_pulses, bl_activity)
        adc = self.adc if adc_bits is None else self.adc.with_bits(adc_bits)
        return {
            "wordline": self.nrow * self.e_wl_cell * wl_pulses,
            "bitline": self.nrow * self.e_bl_cell * bl_activity,
            "cell_fixed": self.nrow * self.e_cell_fixed,
            "column_fixed": self.e_col_fixed,
            "adc": adc.energy_per_conversion(),
            "shift_add": self.acc_bits * self.e_add_bit,
            "out_register": (self.out_bits or adc.bits) * self.e_reg_bit,
        }

    def energy_per_conversion(self, wl_pulses: float, bl_activity: float, adc_bits: int | None = None) -> float:
        return sum(self.conversion_breakdown(wl_pulses, bl_activity, adc_bits).values())

    def energy_per_array_cycle(self) -> float:
        """J per array cycle that is not counted per conversion (input register of nrow rows)."""
        return self.nrow * self.in_bits * self.e_reg_bit

    def energy(self, adc_conversions: int, array_cycles: int, wl_pulses: float, bl_activity: float,
               adc_bits: int | None = None) -> dict:
        """J by component for a layer's event counts (`adc_conversions`, `array_cycles`)."""
        if adc_conversions < 0 or array_cycles < 0:
            raise ValueError("event counts must be >= 0")
        out = {k: v * adc_conversions for k, v in self.conversion_breakdown(wl_pulses, bl_activity, adc_bits).items()}
        out["in_register"] = array_cycles * self.energy_per_array_cycle()
        out["total"] = sum(out.values())
        return out

    # --- latency -------------------------------------------------------------
    def cycle_time(self) -> float:
        """s per array cycle: one array activation, then `adc_share` conversions in series on each ADC."""
        return self.t_array + self.adc_share * self.t_adc

    # --- area ----------------------------------------------------------------
    def n_adcs(self, cols: int) -> int:
        return math.ceil(cols / self.adc_share)

    def area(self, rows: int, cols: int) -> dict:
        """m^2 by component for an array of rows x cols bit cells."""
        if rows < 1 or cols < 1:
            raise ValueError("rows and cols must be >= 1")
        out = {
            "cells": rows * cols * self.area_cell,
            "adc": self.n_adcs(cols) * self.area_adc,
            "col_periph": cols * self.area_col_periph,
            "row_periph": rows * self.area_row_periph,
        }
        out["total"] = sum(out.values())
        return out

    # --- macro-level figure (for calibration against papers) ------------------
    def tops_per_w(self, macs_per_conversion: float, wl_pulses: float, bl_activity: float,
                   cols: int | None = None) -> float:
        """1 MAC = 2 ops. `macs_per_conversion`: multi-bit MACs one conversion stands for
        (e.g. nrow / (x_cycles * w_bits) for bit-serial inputs and bit-sliced weights).
        `cols`: array width, to share the per-cycle input-register energy; None ignores it."""
        e = self.energy_per_conversion(wl_pulses, bl_activity)
        if cols:
            e += self.energy_per_array_cycle() / cols
        return 2.0 * macs_per_conversion / e / 1e12
