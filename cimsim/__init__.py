"""cimsim: SRAM compute-in-memory simulator for LLM inference."""

from cimsim.config import CIMConfig, LayerConfigMap
from cimsim.events import EventCounter
from cimsim.core.layers import CIMLinear, CIMConv2d
from cimsim.calibrate import calibrate_adc_range

__all__ = ["CIMConfig", "LayerConfigMap", "EventCounter", "CIMLinear", "CIMConv2d",
           "calibrate_adc_range"]
__version__ = "0.1.0"
