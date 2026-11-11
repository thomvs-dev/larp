"""Digital SRAM CIM (DCIM) cost model (Part B, B0).

Structure follows a bit-serial DCIM macro with one adder tree per column
(Chih et al., ISSCC 2021, paper 16.4): every input-bit cycle, each column
multiplies `rows` 1-b inputs with its stored `col_wbits`-bit weights (NOR
gates) and reduces them in an adder tree; a partial-sum accumulator
shift-adds the cycles; one extra pipeline cycle finishes the accumulation.

Energy is charged per 1b x 1b product (`bitop`), with a term linear in the
input toggle rate:   e_bitop = e_bitop_base + e_bitop_toggle * toggle_rate.
Chih et al. report that energy depends on input toggle rate and weight
sparsity and mostly comes from the first adder-tree stages; the weight-ones
fraction is fixed at the calibration value (50 % in Chih) and not modelled.

All parameters come from config (config/macros/*.yaml) and are cited in
docs/assumptions.md. Supply-voltage and node scaling are not modelled: a spec
is valid at its calibration VDD and node only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import torch

from cimsim.core.bitplanes import to_twos_complement_bits, to_unsigned_bits


@dataclass(frozen=True)
class DCIMSpec:
    rows: int                    # inputs per column (adder-tree width)
    cols: int                    # columns per macro
    col_wbits: int               # weight bits stored per column
    e_bitop_base: float          # J per 1b x 1b product, toggle-independent part
    e_bitop_toggle: float        # J per 1b x 1b product per unit input toggle rate
    t_cycle: float               # s per input-bit cycle
    pipeline_cycles: int = 1     # extra cycles per pass (accumulator)
    area_per_weight_bit: float = 0.0  # m^2 per stored weight bit, whole macro included
    vdd: float | None = None     # calibration supply (documentation; no scaling)
    node_nm: float | None = None

    def __post_init__(self):
        for name in ("rows", "cols", "col_wbits"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.e_bitop_base < 0 or self.e_bitop_toggle < 0 or self.t_cycle <= 0:
            raise ValueError("energies must be >= 0 and t_cycle > 0")

    @classmethod
    def from_dict(cls, d: dict) -> "DCIMSpec":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known - {"source", "fit"}
        if unknown:
            raise KeyError(f"unknown DCIM spec fields: {sorted(unknown)}")
        return cls(**{k: v for k, v in d.items() if k in known})

    # --- per-action ---------------------------------------------------------
    def e_bitop(self, toggle_rate: float) -> float:
        if not 0.0 <= toggle_rate <= 1.0:
            raise ValueError("toggle_rate must be in [0, 1]")
        return self.e_bitop_base + self.e_bitop_toggle * toggle_rate

    def mac_energy(self, x_bits: int, w_bits: int, toggle_rate: float) -> float:
        """J per x_bits x w_bits MAC."""
        return x_bits * w_bits * self.e_bitop(toggle_rate)

    def energy(self, macs: int, x_bits: int, w_bits: int, toggle_rate: float) -> float:
        """J for `macs` MACs (event counter `macs` of a layer mapped to dcim)."""
        if macs < 0:
            raise ValueError("macs must be >= 0")
        return macs * self.mac_energy(x_bits, w_bits, toggle_rate)

    # --- macro-level figures (for calibration against papers) ----------------
    def outputs_per_pass(self, w_bits: int) -> int:
        """Output channels computed in parallel: a w_bits weight uses ceil(w_bits / col_wbits) columns."""
        return self.cols // math.ceil(w_bits / self.col_wbits)

    def pass_time(self, x_bits: int) -> float:
        return (x_bits + self.pipeline_cycles) * self.t_cycle

    def peak_ops_per_s(self, x_bits: int, w_bits: int) -> float:
        """1 MAC = 2 ops; all rows and columns busy."""
        return 2 * self.rows * self.outputs_per_pass(w_bits) / self.pass_time(x_bits)

    def tops_per_w(self, x_bits: int, w_bits: int, toggle_rate: float) -> float:
        return 2.0 / self.mac_energy(x_bits, w_bits, toggle_rate) / 1e12

    def area(self) -> float:
        """m^2 of one macro."""
        return self.area_per_weight_bit * self.rows * self.cols * self.col_wbits

    def tops_per_mm2(self, x_bits: int, w_bits: int) -> float:
        return self.peak_ops_per_s(x_bits, w_bits) / 1e12 / (self.area() * 1e6)


def fit_toggle_line(points: list[tuple[float, float]], x_bits: int, w_bits: int) -> tuple[float, float]:
    """Fit (e_bitop_base, e_bitop_toggle) to measured (toggle_rate, TOPS/W) points at one precision.

    Least squares on energy per bitop (exact for two points)."""
    if len(points) < 2:
        raise ValueError("need >= 2 points")
    t = torch.tensor([p[0] for p in points], dtype=torch.float64)
    e = torch.tensor([2.0 / (p[1] * 1e12) / (x_bits * w_bits) for p in points], dtype=torch.float64)
    A = torch.stack([torch.ones_like(t), t], dim=1)
    sol = torch.linalg.lstsq(A, e.unsqueeze(1)).solution.squeeze(1)
    return float(sol[0]), float(sol[1])


def input_toggle_rate(q_x: torch.Tensor, x_bits: int, signed: bool = True) -> float:
    """Fraction of bit transitions on each input line when q_x [T, K] is streamed
    bit-serially, MSB first, token after token (as in Chih et al. ISSCC 2021)."""
    if q_x.dim() != 2:
        raise ValueError("q_x must be [T, K]")
    bits = to_twos_complement_bits(q_x, x_bits) if signed else to_unsigned_bits(q_x, x_bits)  # [B, T, K], LSB first
    stream = bits.flip(0).permute(1, 0, 2).reshape(-1, q_x.shape[1])  # [T*B, K], MSB first per token
    if stream.shape[0] < 2:
        return 0.0
    return float((stream[1:] != stream[:-1]).double().mean())
