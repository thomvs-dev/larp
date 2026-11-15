"""Costs of the Part B result system, built from config files only (docs/assumptions.md B-23 ... B-29)."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml

from cimsim.circuits import AnalogArraySpec, DigitalSpec, ScalingRule
from cimsim.circuits.macros import m1_spec, retarget


@dataclass(frozen=True)
class DRAMSpec:
    e_bit: float               # J per bit read from DRAM, interface included
    bandwidth_bits: float      # bits / s per channel
    channels: int = 1

    @classmethod
    def from_dict(cls, d: dict) -> "DRAMSpec":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known - {"source"}
        if unknown:
            raise KeyError(f"unknown DRAM fields: {sorted(unknown)}")
        return cls(**{k: v for k, v in d.items() if k in known})

    def energy(self, bits: float) -> float:
        return bits * self.e_bit

    def time(self, bits: float) -> float:
        return bits / (self.bandwidth_bits * self.channels)


@dataclass(frozen=True)
class SystemCosts:
    acim: AnalogArraySpec      # at the result node, with the run's nrow and ADC bits
    digital: DigitalSpec       # at the result node
    dram: DRAMSpec
    write_e_bit: float         # J per bit written into a CIM array (weights, keys, values)
    input_bit_density: float
    t_eval: float              # s per array evaluation (one input cycle of one layer)
    array_cols: int
    kv_bits: int
    digital_bits: int
    head_macs_bits: tuple
    node_nm: float
    vdd: float


def load_costs(root: str | Path, nrow: int, adc_bits: int, dram: str = "default",
               system: str = "config/system/acim_22nm.yaml") -> SystemCosts:
    root = Path(root)
    y = lambda p: yaml.safe_load((root / p).read_text(encoding="utf-8"))  # noqa: E731
    sysc, sc = y(system), y("config/circuits/scaling.yaml")
    rule = ScalingRule.from_dict(sc["rule"])
    node, vdd = sc["target"]["node_nm"], sc["target"]["vdd"]
    macro = y(sysc["macro"])
    src = m1_spec(macro, y("config/circuits/sar_switching.yaml")["conventional"])
    dig = DigitalSpec.from_dict(y("config/circuits/digital_45nm.yaml")).scaled(node, vdd, rule)
    acim = retarget(src, rule, node, vdd, nrow=nrow, adc_bits=adc_bits, acc_bits=sysc["acc_bits"])
    # shift-add and registers of the result system use the digital library, not the macro
    from dataclasses import replace
    acim = replace(acim, e_add_bit=dig.e_add_bit, e_reg_bit=dig.e_reg_bit)
    f = rule.energy_factor(src.node_nm, src.vdd, node, vdd)
    return SystemCosts(acim=acim, digital=dig, dram=DRAMSpec.from_dict(y("config/system/dram.yaml")[dram]),
                       write_e_bit=sysc["write_e_bit"] * f, input_bit_density=sysc["input_bit_density"],
                       t_eval=sysc["t_eval"] * rule.delay_factor(src.node_nm, node), array_cols=sysc["array_cols"],
                       kv_bits=sysc["kv_bits"], digital_bits=sysc["digital_bits"],
                       head_macs_bits=tuple(sysc["head_macs_bits"]), node_nm=node, vdd=vdd)
