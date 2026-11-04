"""Event counting for CIM layers.

Counts are integers per layer. Part B multiplies them by per-action energy
and latency, so every mode must produce them.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

FIELDS = (
    "macs",
    "adc_conversions",
    "adc_conversions_analog",
    "array_cycles",
    "weight_bits_written",
    "dcim_column_ops",      # macro 'dcim': one adder-tree reduction per (row chunk, input cycle, weight bit, column)
    "digital_macs",         # macro 'digital': MACs on a digital integer unit
    "kv_bits_written",      # attention (A4): key / value bits written into CIM arrays (new tokens only)
)


@dataclass
class EventCounter:
    per_layer: dict = field(default_factory=lambda: defaultdict(lambda: dict.fromkeys(FIELDS, 0)))

    def add(self, layer: str, **counts: int) -> None:
        rec = self.per_layer[layer]
        for k, v in counts.items():
            if k not in rec:
                raise KeyError(f"unknown event {k!r}")
            rec[k] += int(v)

    def total(self) -> dict:
        out = dict.fromkeys(FIELDS, 0)
        for rec in self.per_layer.values():
            for k, v in rec.items():
                out[k] += v
        return out

    def reset(self) -> None:
        self.per_layer.clear()

    def reset_compute(self) -> None:
        """Zero every count except weight_bits_written (weights are written into the arrays once).
        Used after calibration so that events.json covers only the evaluated tokens."""
        for rec in self.per_layer.values():
            for k in rec:
                if k != "weight_bits_written":
                    rec[k] = 0

    def to_dict(self) -> dict:
        return {"total": self.total(), "per_layer": {k: dict(v) for k, v in self.per_layer.items()}}
