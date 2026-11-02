"""CIM layer modules."""

from __future__ import annotations

import zlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from cimsim.config import CIMConfig
from cimsim.events import EventCounter
from cimsim.quant import quantize_weight, quantize_act, dequant_factor
from cimsim.core.executor import cim_matmul_int, analytic_counts
from cimsim.core.executor_fast import cim_matmul_int_fast


class CIMLinear(nn.Module):
    """Drop-in replacement for nn.Linear executed on a simulated SRAM CIM macro.

    weight: [out_features, in_features] (nn.Linear layout).
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        cfg: CIMConfig,
        name: str = "",
        counter: EventCounter | None = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight.detach().clone(), requires_grad=False)
        self.bias = None if bias is None else nn.Parameter(bias.detach().clone(), requires_grad=False)
        self.cfg = cfg
        self.name = name
        self.counter = counter
        self.out_features, self.in_features = self.weight.shape
        self._qw: torch.Tensor | None = None
        self._sw: torch.Tensor | None = None
        self._gen: torch.Generator | None = None
        self._weights_counted = False

    @classmethod
    def from_linear(cls, lin: nn.Linear, cfg: CIMConfig, name: str = "", counter: EventCounter | None = None) -> "CIMLinear":
        return cls(lin.weight, lin.bias, cfg, name, counter)

    @classmethod
    def from_conv1d(cls, conv, cfg: CIMConfig, name: str = "", counter: EventCounter | None = None) -> "CIMLinear":
        """HuggingFace GPT-2 Conv1D stores weight as [in, out]."""
        return cls(conv.weight.t(), conv.bias, cfg, name, counter)

    def extra_repr(self) -> str:
        c = self.cfg
        return (f"in={self.in_features}, out={self.out_features}, mode={c.mode}, "
                f"W{c.w_bits}A{c.x_bits}, nrow={c.nrow}, adc={c.adc_bits}, enc={c.act_enc}")

    def requantize(self) -> None:
        self._qw = None

    def _weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._qw is None or self._qw.device != self.weight.device:
            w = self.weight.float()
            self._qw, self._sw = quantize_weight(w, self.cfg.w_bits, self.cfg.w_gran)
        return self._qw, self._sw

    def _generator(self, device: torch.device) -> torch.Generator:
        if self._gen is None or self._gen.device != device:
            self._gen = torch.Generator(device=device)
            # distinct but reproducible stream per layer
            self._gen.manual_seed(self.cfg.seed * 1_000_003 + zlib.crc32(self.name.encode()) % 1_000_003)
        return self._gen

    def _count(self, T: int) -> None:
        if self.counter is None:
            return
        self.counter.add(self.name, **analytic_counts(T, self.in_features, self.out_features, self.cfg))
        if not self._weights_counted:
            self.counter.add(self.name, weight_bits_written=self.in_features * self.out_features * self.cfg.w_bits)
            self._weights_counted = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        if cfg.mode == "fp":
            return F.linear(x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))

        in_dtype = x.dtype
        lead = x.shape[:-1]
        x2 = x.reshape(-1, self.in_features).float()
        T = x2.shape[0]

        qw, sw = self._weights()
        qx, sx = quantize_act(x2, cfg.x_bits, cfg.x_gran, cfg.signed_act)

        if cfg.mode == "fake_quant":
            y_int = qx @ qw.t()
        elif cfg.mode == "bit_accurate":
            fn = cim_matmul_int_fast if cfg.backend == "fast" else cim_matmul_int
            y_int = fn(qx, qw, cfg, self._generator(x2.device)).float()
        elif cfg.mode == "statistical":
            raise NotImplementedError("statistical mode is milestone A3")
        else:
            raise ValueError(cfg.mode)

        y = y_int * sx * sw.t() * dequant_factor(cfg.w_bits, cfg.x_bits, cfg.signed_act)
        if self.bias is not None:
            y = y + self.bias.float()
        self._count(T)
        return y.reshape(*lead, self.out_features).to(in_dtype)
