"""Model-level ADC range calibration.

    from cimsim import calibrate_adc_range
    calibrate_adc_range(model, lambda: model(calib_ids))

Every CIM layer whose config has mode "bit_accurate" and adc_range
"calibrated", and every layer with mode "statistical", records the histogram
of ideal analog MAC values during run_fn. It then sets its ADC full scale
(cimsim/core/calib.py) and, for statistical layers, the analytic error model
(cimsim/stat/analytic.py). While run_fn
executes, all CIM layers produce ideal (fake-quant) outputs, so calibration
statistics do not depend on noise or ADC error in earlier layers.

With cache_dir and data_key, the recorded histograms are stored per layer and reused by later runs
with the same model, calibration data and histogram-relevant settings (cimsim/calib_cache.py). Layers
found in the cache are not recorded again; if all are found, run_fn is not called.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from cimsim.core.layers import _CIMBase


def needs_calibration(cfg) -> bool:
    """bit_accurate layers with a calibrated ADC range, and every statistical layer
    (its error model is computed from the same MAC histogram)."""
    if cfg.macro != "acim":
        return False                     # dcim / digital layers are exact; nothing to calibrate
    return (cfg.mode == "bit_accurate" and cfg.adc_range == "calibrated") or cfg.mode == "statistical"


def calibrate_adc_range(model: nn.Module, run_fn: Callable[[], object], cache_dir=None,
                        data_key: str | None = None, report: dict | None = None) -> list[str]:
    """Returns the names of the calibrated layers. cache_dir + data_key: reuse / store MAC histograms
    (data_key must identify the calibration data that run_fn feeds). report, if given, receives
    {"cache": "off" | "hit" | "partial" | "miss", "hits": n, "misses": n, "cache_dir": str}."""
    cim = [m for m in model.modules() if isinstance(m, _CIMBase)]
    targets = [m for m in cim if needs_calibration(m.cfg)]
    if not targets:
        return []
    cache = None
    if cache_dir is not None and data_key is not None:
        from cimsim.calib_cache import CalibCache, layer_key, model_key
        cache = CalibCache(cache_dir, model_key(model, cim, data_key, needs_calibration))
    hits, misses, lkeys = [], [], {}
    for m in targets:
        if cache is not None:
            lkeys[m.name] = layer_key(m, needs_calibration)
            m._hist, m._cal_colmax = None, None
            if cache.load(m, lkeys[m.name]):
                hits.append(m)
                continue
        misses.append(m)
    for m in cim:
        m._ideal_override = True
    for m in misses:
        m.start_calibration()
        m._want_colmax = cache is not None
    try:
        if misses:
            with torch.no_grad():
                run_fn()
            if cache is not None:
                for m in misses:
                    if m._hist is not None:
                        cache.save(m, lkeys[m.name])
        for m in targets:
            m.finish_calibration()
    finally:
        for m in cim:
            m._ideal_override = False
            m._calibrating = False
            m._want_colmax = False
    if report is not None:
        state = "off" if cache is None else ("hit" if not misses else ("miss" if not hits else "partial"))
        report.update({"cache": state, "hits": len(hits), "misses": len(misses),
                       "cache_dir": str(cache.dir) if cache is not None else None})
    return [m.name for m in targets]
