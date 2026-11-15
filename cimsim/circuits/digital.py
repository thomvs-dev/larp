"""Digital unit costs (Part B, B0): integer add / multiply, register and SRAM
buffer access, and the non-linear units of a transformer block.

Per-operation energies are config values measured or cited at one node / VDD
(config/circuits/digital_45nm.yaml, docs/assumptions.md B-12 ... B-15);
`DigitalSpec.scaled` moves them to the result node with a ScalingRule.

Non-linear units (softmax, LayerNorm / RMSNorm, SiLU / GELU, RoPE) are costed
as fixed-point operation counts per element (`UNIT_OPS`). The counts are a
modelling assumption, not silicon data (B-15); Part B reports the share of
these units in every total so the assumption can be judged.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

from .scaling import ScalingRule

# Per element: (multiplies, adds, table reads) at the unit's word width.
# Per vector (once per normalised vector / softmax row): same triple.
UNIT_OPS = {
    # exp by table + linear interpolation (1 read, 1 mul, 1 add); running sum (1 add); multiply by 1 / sum (1 mul)
    "softmax":   {"element": (2, 2, 1), "vector": (2, 1, 1)},      # vector: reciprocal of the sum (table + 2 mul + 1 add)
    # square (1 mul), sum (1 add), scale by 1 / rms (1 mul), gain (1 mul)
    "rmsnorm":   {"element": (3, 1, 0), "vector": (2, 1, 1)},      # vector: inverse square root (table + 2 mul + 1 add)
    # mean (1 add), subtract (1 add), square (1 mul), sum (1 add), scale (1 mul), gain (1 mul), bias (1 add)
    "layernorm": {"element": (3, 4, 0), "vector": (2, 1, 1)},
    # sigmoid / Phi by table + interpolation (1 read, 1 mul, 1 add), product with x (1 mul)
    "silu":      {"element": (2, 1, 1), "vector": (0, 0, 0)},
    "gelu":      {"element": (2, 1, 1), "vector": (0, 0, 0)},
    "relu":      {"element": (0, 0, 0), "vector": (0, 0, 0)},
    # gated MLP (SwiGLU): gate * up, one product per element
    "gate_mul":  {"element": (1, 0, 0), "vector": (0, 0, 0)},
    # rotation of a pair of dimensions: 4 mul + 2 add per pair = 2 mul + 1 add per element (sin / cos precomputed)
    "rope":      {"element": (2, 1, 0), "vector": (0, 0, 0)},
    # residual connection
    "residual":  {"element": (0, 1, 0), "vector": (0, 0, 0)},
}


@dataclass(frozen=True)
class DigitalSpec:
    node_nm: float
    vdd: float
    e_add_bit: float        # J per bit of an integer adder
    e_mul_bit2: float       # J per bit^2 of an integer multiplier (a_bits x b_bits)
    e_reg_bit: float        # J per bit read from / written to a small register file
    e_sram_bit: float       # J per bit read from an on-chip SRAM buffer
    e_lut_bit: float | None = None   # J per bit of a table read; default: e_sram_bit

    def __post_init__(self):
        if self.node_nm <= 0 or self.vdd <= 0:
            raise ValueError("node_nm and vdd must be > 0")
        for name in ("e_add_bit", "e_mul_bit2", "e_reg_bit", "e_sram_bit"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")

    @classmethod
    def from_dict(cls, d: dict) -> "DigitalSpec":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known - {"source", "points"}
        if unknown:
            raise KeyError(f"unknown digital spec fields: {sorted(unknown)}")
        return cls(**{k: v for k, v in d.items() if k in known})

    def scaled(self, node_nm: float, vdd: float, rule: ScalingRule) -> "DigitalSpec":
        f = rule.energy_factor(self.node_nm, self.vdd, node_nm, vdd)
        return replace(self, node_nm=node_nm, vdd=vdd,
                       e_add_bit=self.e_add_bit * f, e_mul_bit2=self.e_mul_bit2 * f,
                       e_reg_bit=self.e_reg_bit * f, e_sram_bit=self.e_sram_bit * f,
                       e_lut_bit=None if self.e_lut_bit is None else self.e_lut_bit * f)

    # --- per-operation energies (J) ------------------------------------------
    def add(self, bits: int) -> float:
        return bits * self.e_add_bit

    def mul(self, a_bits: int, b_bits: int | None = None) -> float:
        return a_bits * (a_bits if b_bits is None else b_bits) * self.e_mul_bit2

    def mac(self, x_bits: int, w_bits: int, acc_bits: int) -> float:
        """One integer multiply-accumulate (event counter `digital_macs`)."""
        return self.mul(x_bits, w_bits) + self.add(acc_bits)

    def reg(self, bits: int) -> float:
        return bits * self.e_reg_bit

    def sram_read(self, bits: int) -> float:
        return bits * self.e_sram_bit

    def lut(self, bits: int) -> float:
        return bits * (self.e_sram_bit if self.e_lut_bit is None else self.e_lut_bit)

    def _ops(self, ops: tuple[int, int, int], bits: int) -> float:
        n_mul, n_add, n_lut = ops
        return n_mul * self.mul(bits) + n_add * self.add(bits) + n_lut * self.lut(bits)

    def unit_energy(self, unit: str, elements: int, vectors: int = 1, bits: int = 16) -> float:
        """J for a non-linear unit over `elements` values in `vectors` rows, `bits`-wide fixed point."""
        if unit not in UNIT_OPS:
            raise KeyError(f"unknown unit {unit!r}; known: {sorted(UNIT_OPS)}")
        if elements < 0 or vectors < 0:
            raise ValueError("elements and vectors must be >= 0")
        u = UNIT_OPS[unit]
        return elements * self._ops(u["element"], bits) + vectors * self._ops(u["vector"], bits)

    def hadamard(self, adds: int, bits: int = 16) -> float:
        """J for the online input rotation (event counter `hadamard_adds`)."""
        return adds * self.add(bits)
