"""Calibrated macros as cost-model specs (Part B, B1), and their transfer to another node / operating point."""

from __future__ import annotations

from dataclasses import replace

from .adc import ADCSpec
from .analog import AnalogArraySpec
from .scaling import ScalingRule


def m1_spec(cfg: dict, dac_switch_factor: float) -> AnalogArraySpec:
    """Jia JSSC 2020 at its own node / VDD from config/macros/m1_jia_jssc20.yaml (docs/assumptions.md B-21)."""
    v = cfg["vdd"]
    e = cfg["energy_pj"][v]
    n, share, act = cfg["nrow"], cfg["data_dependent_share"], cfg["calib_activity"]
    bits = cfg["adc_bits"]
    e_dac = dac_switch_factor * 2 ** bits * cfg["adc_c_unit"] * v ** 2
    adc = ADCSpec(arch="sar", bits=bits, c_unit=cfg["adc_c_unit"], vref=v, dac_switch_factor=dac_switch_factor,
                  e_comparator=(e["adc"] * 1e-12 - e_dac) / bits)     # comparator + SAR logic per bit cycle
    return AnalogArraySpec(nrow=n, adc=adc, e_wl_cell=0.0,
                           e_bl_cell=share * e["cima"] * 1e-12 / n / act,   # input broadcast + cell capacitor, per active cell
                           e_cell_fixed=(1 - share) * e["cima"] * 1e-12 / n,
                           area_cell=cfg["bitcell_um2"] * 1e-12, vdd=v, node_nm=cfg["node_nm"])


def retarget(spec: AnalogArraySpec, rule: ScalingRule, node_nm: float, vdd: float, nrow: int | None = None,
             adc_bits: int | None = None, **extra) -> AnalogArraySpec:
    """Move a calibrated analog spec to another node / VDD and operating point (rows per conversion,
    ADC bits). Per-cell and per-comparator energies scale with the rule; the SAR DAC scales through
    its unit capacitor (~ node) and reference (= VDD); areas and times scale with the rule."""
    f = rule.energy_factor(spec.node_nm, spec.vdd, node_nm, vdd)
    fa, ft = rule.area_factor(spec.node_nm, node_nm), rule.delay_factor(spec.node_nm, node_nm)
    a = spec.adc
    bits = a.bits if adc_bits is None else adc_bits
    if a.arch == "sar":
        adc = replace(a, bits=bits, c_unit=a.c_unit * (node_nm / spec.node_nm) ** rule.energy_node_exp, vref=vdd,
                      e_comparator=a.e_comparator * f, e_logic_bit=a.e_logic_bit * f, e_fixed=a.e_fixed * f)
    elif a.arch == "flash":
        adc = replace(a, bits=bits, e_comparator=a.e_comparator * f, e_fixed=a.e_fixed * f)
    else:
        adc = a.with_bits(bits)                      # survey bounds are node-independent
    return replace(spec, nrow=spec.nrow if nrow is None else nrow, adc=adc,
                   e_wl_cell=spec.e_wl_cell * f, e_bl_cell=spec.e_bl_cell * f, e_cell_fixed=spec.e_cell_fixed * f,
                   e_col_fixed=spec.e_col_fixed * f, e_add_bit=spec.e_add_bit * f, e_reg_bit=spec.e_reg_bit * f,
                   t_array=spec.t_array * ft, t_adc=spec.t_adc * ft, area_cell=spec.area_cell * fa,
                   area_adc=spec.area_adc * fa, area_col_periph=spec.area_col_periph * fa,
                   area_row_periph=spec.area_row_periph * fa, vdd=vdd, node_nm=node_nm, **extra)
