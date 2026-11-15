"""Technology-node and supply-voltage scaling (Part B, B0).

Used only to move a calibrated or cited per-action cost to another node / VDD,
never to invent a macro (docs/part_b_plan.md §3).

Rule (docs/assumptions.md B-11):
    energy  ~ node**energy_node_exp * vdd**2      (switched capacitance C ~ node, E = C V^2)
    area    ~ node**area_node_exp
    delay   ~ node**delay_node_exp                (no VDD term)
The exponents come from config (config/circuits/scaling.yaml). Classical
constant-field scaling gives 1 / 2 / 1. Node names below ~22 nm are not
physical dimensions, so the rule is checked against the calibrated macros at
65, 22 and 7 nm in B1 and the spread is reported with every scaled number.
"""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class ScalingRule:
    energy_node_exp: float
    area_node_exp: float
    delay_node_exp: float

    @classmethod
    def from_dict(cls, d: dict) -> "ScalingRule":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known - {"source"}
        if unknown:
            raise KeyError(f"unknown scaling fields: {sorted(unknown)}")
        return cls(**{k: float(v) for k, v in d.items() if k in known})

    @staticmethod
    def _check(*values: float) -> None:
        if any(v <= 0 for v in values):
            raise ValueError("node and vdd must be > 0")

    def energy_factor(self, node_from: float, vdd_from: float, node_to: float, vdd_to: float) -> float:
        self._check(node_from, vdd_from, node_to, vdd_to)
        return (node_to / node_from) ** self.energy_node_exp * (vdd_to / vdd_from) ** 2

    def area_factor(self, node_from: float, node_to: float) -> float:
        self._check(node_from, node_to)
        return (node_to / node_from) ** self.area_node_exp

    def delay_factor(self, node_from: float, node_to: float) -> float:
        self._check(node_from, node_to)
        return (node_to / node_from) ** self.delay_node_exp
