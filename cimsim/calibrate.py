"""Model-level ADC range calibration.

    from cimsim import calibrate_adc_range
    calibrate_adc_range(model, lambda: model(calib_ids))

Every CIM layer whose config has mode "bit_accurate" and adc_range
"calibrated" records the histogram of ideal analog MAC values during run_fn
and then sets its ADC full scale (see cimsim/core/calib.py). While run_fn
executes, all CIM layers produce ideal (fake-quant) outputs, so calibration
statistics do not depend on noise or ADC error in earlier layers.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from cimsim.core.layers import _CIMBase


def calibrate_adc_range(model: nn.Module, run_fn: Callable[[], object]) -> list[str]:
    cim = [m for m in model.modules() if isinstance(m, _CIMBase)]
    targets = [m for m in cim if m.cfg.mode == "bit_accurate" and m.cfg.adc_range == "calibrated"]
    if not targets:
        return []
    for m in cim:
        m._ideal_override = True
    for m in targets:
        m.start_calibration()
    try:
        with torch.no_grad():
            run_fn()
        for m in targets:
            m.finish_calibration()
    finally:
        for m in cim:
            m._ideal_override = False
            m._calibrating = False
    return [m.name for m in targets]
