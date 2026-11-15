"""Analytic mapper (Part B, B2): array tiling and the weight residency / streaming policy. No search.

Tiling: a K x N layer with w_bits-bit weights needs ceil(K / nrow) x ceil(N * w_bits / cols) arrays of
nrow x cols bit cells; unused cells of the edge arrays are counted as overhead.

Residency policy (docs/assumptions.md B-27): the chip has `budget_bits` of CIM array cells. If every
layer fits, all weights are written once and stay. Otherwise the largest layer's arrays are reserved as
working space, the remaining budget is filled with resident layers in model order, and every other
layer is streamed from DRAM and written into the working arrays each time it is used: once per token
in decode, once per pass over the whole prompt in prefill.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from cimsim.system.shapes import LinearLayer


@dataclass(frozen=True)
class LayerMap:
    layer: LinearLayer
    arrays: int
    cells: int            # bit cells occupied, edge overhead included
    weight_bits: int
    resident: bool


@dataclass(frozen=True)
class Mapping:
    layers: tuple
    budget_bits: float
    working_bits: int      # cells reserved for streamed layers (0 if everything is resident)

    @property
    def resident_cells(self) -> int:
        return sum(m.cells for m in self.layers if m.resident)

    @property
    def cells_used(self) -> int:
        return self.resident_cells + self.working_bits

    @property
    def streamed_bits(self) -> int:
        """Weight bits loaded per pass over the model."""
        return sum(m.weight_bits for m in self.layers if not m.resident)

    @property
    def total_bits(self) -> int:
        return sum(m.weight_bits for m in self.layers)

    @property
    def arrays(self) -> int:
        return sum(m.arrays for m in self.layers if m.resident) + max(
            [m.arrays for m in self.layers if not m.resident], default=0)


def tile(layer: LinearLayer, nrow: int, cols: int, w_bits: int) -> tuple[int, int]:
    """(arrays, bit cells) for one layer."""
    arrays = math.ceil(layer.k / nrow) * math.ceil(layer.n * w_bits / cols)
    return arrays, arrays * nrow * cols


def map_weights(layers: list[LinearLayer], budget_bits: float, nrow: int, cols: int, w_bits: int = 8) -> Mapping:
    tiles = [tile(l, nrow, cols, w_bits) for l in layers]
    total = sum(c for _, c in tiles)
    if total <= budget_bits:
        return Mapping(tuple(LayerMap(l, a, c, l.params * w_bits, True) for l, (a, c) in zip(layers, tiles)),
                       budget_bits, 0)
    working = max(c for _, c in tiles)
    if working > budget_bits:
        raise ValueError(f"budget {budget_bits:.3g} bits is below the largest layer ({working} cells)")
    left = budget_bits - working
    out = []
    for l, (a, c) in zip(layers, tiles):
        res = c <= left
        if res:
            left -= c
        out.append(LayerMap(l, a, c, l.params * w_bits, res))
    return Mapping(tuple(out), budget_bits, working)
