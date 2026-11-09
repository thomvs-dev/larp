"""CIM layer modules: CIMLinear and CIMConv2d.

Both share _CIMBase: quantize -> integer CIM matmul (fake_quant or
bit_accurate) -> dequantize, plus ADC range calibration and event counting.
"""

from __future__ import annotations

import math

import zlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from cimsim.config import CIMConfig
from cimsim.events import EventCounter
from cimsim.quant import (quantize_weight, quantize_weight_asym, quantize_act, dequant_factor,
                          qmax_signed, qmax_unsigned)
from cimsim.core.executor import cim_matmul_int, analytic_counts
from cimsim.core.executor_fast import cim_matmul_int_fast
from cimsim.core.bitplanes import input_cycles
from cimsim.core.calib import mac_histogram, full_scale_from_histogram
from cimsim.stat.analytic import layer_stat_params
from cimsim.stat.saturation import deterministic_tables, saturation_correction, saturation_plan, thresholds


def _codes_int(q: torch.Tensor, bits: int, signed: bool) -> torch.Tensor:
    qi = q.round().to(torch.int64)
    return torch.where(qi < 0, qi + (1 << bits), qi) if signed else qi


class _CIMBase(nn.Module):
    """Shared integer-matmul path. Subclasses provide the [T, K] input rows
    and the [N, K] weight matrix, and optionally row chunks."""

    def __init__(self, cfg: CIMConfig, name: str, counter: EventCounter | None) -> None:
        super().__init__()
        self.cfg = cfg
        self.name = name
        self.counter = counter
        self._qw: torch.Tensor | None = None
        self._wscale: torch.Tensor | None = None
        self._wzero: torch.Tensor | None = None
        self._gen: torch.Generator | None = None
        self._weights_counted = False
        self._calibrating = False
        self._ideal_override = False                 # set by calibrate_adc_range
        self._probe = None                           # optional callable(layer, qx, qw, y_int); A3 error measurement
        self._hist: torch.Tensor | None = None
        self.register_buffer("adc_fs", None)        # [C, P] calibrated ADC full scale, or None
        self.register_buffer("stat_params", None)   # [gain, mean, std] for mode 'statistical' (A3)
        self.stat_info: dict | None = None          # diagnostics from layer_stat_params
        self._sat_tables: torch.Tensor | None = None  # S(m) tables for stat_saturation 'exact'
        self._sat_key = None
        self._sat_plan = None                        # (key, saturation_plan) for static weights
        self._qw_norm = None                         # (id of weight cache, max row L2 norm of codes)
        self._det_hist: torch.Tensor | None = None   # calibration histogram kept for the T(m) table choice
        self._flips: list = []                       # discrete code-flip distributions (stat_noise 'auto', F9)
        self._cal_colmax: torch.Tensor | None = None  # [C, P, N] largest calibration MAC per column
        self._want_colmax = False                    # record column maxima in any mode (calibration cache)
        self.register_buffer("sat_cols", None)       # [C, P, N] bool: columns checked for saturation
        self.sat_stats: dict = {}                    # exact-saturation work: macs_exact / macs_total
        self._dynamic_operand = False                # True for CIMMatMul: the "weight" changes every call

    # --- to be provided by subclasses ---
    def _weight_matrix(self) -> torch.Tensor:  # [N, K]
        raise NotImplementedError

    def _chunks(self) -> list[tuple[int, int]] | None:
        return None

    # --- shared ---
    def requantize(self) -> None:
        self._qw = None
        self._sat_plan = None
        self._qw_norm = None

    def _weights(self):
        """Integer weight codes (float32 [N, K]), scale and zero point. The codes are cached as int8 /
        uint8 when they fit (w_bits <= 8) and converted per call: for 1.5B-parameter models a float32
        cache would double the weight memory."""
        if self._qw is None or self._qw.device != self._weight_matrix().device:
            w = self._weight_matrix().float()
            if self.cfg.w_signed:
                qw, self._wscale = quantize_weight(w, self.cfg.w_bits, self.cfg.w_gran)
                self._wzero = None
                store = torch.int8 if self.cfg.w_bits <= 8 else None
            else:
                qw, self._wscale, self._wzero = quantize_weight_asym(w, self.cfg.w_bits, self.cfg.w_gran)
                store = torch.uint8 if self.cfg.w_bits <= 8 else None
            self._qw = qw.to(store) if store is not None else qw
            self._sat_plan = None
            self._qw_norm = None
        qw = self._qw if self._qw.dtype == torch.float32 else self._qw.float()
        return qw, self._wscale, self._wzero

    def _generator(self, device: torch.device) -> torch.Generator:
        if self._gen is None or self._gen.device != device:
            self._gen = torch.Generator(device=device)
            # distinct but reproducible stream per layer
            self._gen.manual_seed(self.cfg.seed * 1_000_003 + zlib.crc32(self.name.encode()) % 1_000_003)
        return self._gen

    def _count(self, T: int, K: int, N: int) -> None:
        if self.counter is None:
            return
        chunks = self._chunks()
        self.counter.add(self.name, **analytic_counts(T, K, N, self.cfg, None if chunks is None else len(chunks)))
        if not self._weights_counted and self.cfg.macro != "digital":
            self.counter.add(self.name, weight_bits_written=K * N * self.cfg.w_bits)
            self._weights_counted = True

    # calibration
    def start_calibration(self) -> None:
        self._calibrating = True
        self._hist = None
        self._cal_colmax = None

    def _n_chunks(self) -> int:
        chunks = self._chunks()
        return len(chunks) if chunks is not None else math.ceil(self._weight_matrix().shape[1] / self.cfg.nrow)

    def finish_calibration(self) -> None:
        """Sets the ADC full scale (adc_range: calibrated) and, for mode 'statistical',
        the analytic error model, both from the MAC histogram recorded since start_calibration."""
        self._calibrating = False
        self._sat_plan = None
        self._qw_norm = None
        if self._hist is None:
            raise RuntimeError(f"{self.name}: no calibration data seen")
        if self.cfg.adc_range == "calibrated":
            self.adc_fs = full_scale_from_histogram(self._hist, self.cfg)
        if self.cfg.mode == "statistical":
            exact_sat = self.cfg.stat_saturation == "exact"
            info = layer_stat_params(self._hist, self.cfg, self._n_chunks(), self.adc_fs,
                                     exclude_saturation=exact_sat,
                                     flip_lambda_max=self.cfg.stat_flip_lambda_max if self.cfg.stat_noise == "auto" else None,
                                     flip_kurtosis_min=self.cfg.stat_flip_kurtosis_min)
            self.stat_params = torch.tensor([info["gain"], info["mean"], info["std"]], dtype=torch.float64)
            self._flips = info["flips"]
            self.stat_info = {k: v for k, v in info.items() if k not in ("per_cp", "flips")}
            self.stat_info["stat_noise"] = self.cfg.stat_noise
            self.stat_info["stat_saturation"] = self.cfg.stat_saturation
            self._sat_tables = None
            self._det_hist = self._hist.clone() if exact_sat else None
            if exact_sat and self.cfg.stat_sat_margin is not None and self._cal_colmax is not None:
                thr = thresholds(self._saturation_tables(self._hist))               # [C, P]
                self.sat_cols = self._cal_colmax >= (thr - self.cfg.stat_sat_margin).unsqueeze(-1)
                self.stat_info["sat_cols_frac"] = float(self.sat_cols.double().mean())
            else:
                self.sat_cols = None
        self._cal_colmax = None
        self._hist = None

    def _int_matmul(self, qx: torch.Tensor, qw: torch.Tensor) -> torch.Tensor:
        """qx [T, K], qw [N, K] -> y_int [T, N] (float32), before zero-point correction."""
        cfg = self.cfg
        if self._calibrating:
            want_max = ((self._want_colmax or (cfg.mode == "statistical" and cfg.stat_saturation == "exact"))
                        and not self._dynamic_operand)
            h = mac_histogram(qx.cpu(), qw.cpu(), cfg, self._chunks(), col_max=want_max)
            if want_max:
                h, cm = h
                self._cal_colmax = cm if self._cal_colmax is None else torch.maximum(self._cal_colmax, cm)
            self._hist = h if self._hist is None else self._pad_add(self._hist, h)
            return qx @ qw.t()                                   # ideal output while calibrating
        if cfg.mode == "fake_quant" or cfg.macro != "acim" or self._ideal_override:
            return qx @ qw.t()
        if cfg.mode == "bit_accurate":
            fs = None
            if cfg.adc_range == "calibrated":
                if self.adc_fs is None:
                    raise RuntimeError(f"{self.name}: adc_range is 'calibrated' but the layer has not "
                                       "been calibrated (use cimsim.calibrate_adc_range)")
                fs = self.adc_fs
            fn = cim_matmul_int_fast if cfg.backend == "fast" else cim_matmul_int
            y = fn(qx, qw, cfg, self._generator(qx.device), full_scale=fs, chunks=self._chunks()).float()
            if self._probe is not None:
                self._probe(self, qx, qw, y)
            return y
        if cfg.mode == "statistical":
            if self.stat_params is None:
                raise RuntimeError(f"{self.name}: mode 'statistical' needs the error model; run "
                                   "cimsim.calibrate_adc_range (it also prepares statistical layers)")
            g, mu, sd = (float(v) for v in self.stat_params)
            y = self._exact_matmul(qx, qw) * (1.0 + g) + mu
            if sd > 0:
                y = y + sd * torch.randn(y.shape, generator=self._generator(qx.device), dtype=torch.float64,
                                         device=qx.device)
            if self._flips:
                self._add_flips(y, qx.cpu(), qw.cpu(), self._generator(qx.device))
            if cfg.stat_saturation == "exact":
                tables = self._saturation_tables()
                col_mask = self.sat_cols if cfg.stat_sat_margin is not None else None
                plan = False                                     # dynamic operand: build per call
                if not self._dynamic_operand:
                    key = (id(self._qw), self._sat_key, id(col_mask))
                    if self._sat_plan is None or self._sat_plan[0] != key:
                        self._sat_plan = (key, saturation_plan(qw, cfg, tables, self._chunks(), qw.shape[1],
                                                               col_mask))
                    plan = self._sat_plan[1]
                s = saturation_correction(qx, qw, cfg, tables, self._chunks(), self.sat_stats,
                                          col_mask=col_mask, plan=plan)
                if s is not None:
                    y = y + s
            return y.float()
        raise ValueError(cfg.mode)

    def _exact_matmul(self, qx: torch.Tensor, qw: torch.Tensor) -> torch.Tensor:
        """Exact integer product qx @ qw.T as float64. Uses a float32 matmul when every partial sum is an
        integer below 2^24 (then float32 is exact in any summation order): by Cauchy-Schwarz each partial
        sum is at most max_t ||qx_t||_2 * max_n ||qw_n||_2; the check uses 2^23 to leave room for rounding
        in the norms. Otherwise float64."""
        key = id(self._qw) if not self._dynamic_operand else None
        cached = getattr(self, "_qw_norm", None)
        if key is not None and cached is not None and cached[0] == key:
            wn = cached[1]
        else:
            wn = float(qw.float().norm(dim=1).max()) if qw.numel() else 0.0
            if key is not None:
                self._qw_norm = (key, wn)
        xn = float(qx.float().norm(dim=1).max()) if qx.numel() else 0.0
        if xn * wn < 2.0 ** 23:
            return (qx.float() @ qw.float().t()).double()
        return qx.double() @ qw.double().t()

    def _add_flips(self, y: torch.Tensor, qx: torch.Tensor, qw: torch.Tensor, gen: torch.Generator) -> None:
        """Discrete ADC code flips from analog noise, sampled by exact thinning (F9). For each (cycle, plane)
        in self._flips: candidate conversions (token, chunk, column) are drawn at the largest per-conversion
        flip probability pmax (Poisson count, uniform positions); the ideal MAC m of each candidate is
        computed exactly; it is kept with probability pany(m) / pmax and changes its output by
        coef · step · j, j drawn from P(j | m). The result has the per-conversion distribution of the
        bit-accurate executor (independent noise per conversion), including absorption at m = 0 and above
        the ADC full scale."""
        T, N = y.shape
        chunks = self._chunks()
        K = qx.shape[1]
        chunk_list = [(k0, min(K, k0 + self.cfg.nrow)) for k0 in range(0, K, self.cfg.nrow)] if chunks is None else chunks
        n_chunks = len(chunk_list)
        ux = _codes_int(qx, self.cfg.x_bits, self.cfg.signed_act)
        uw = _codes_int(qw, self.cfg.w_bits, self.cfg.w_signed)
        flat = y.view(-1)
        for f in self._flips:
            lam = torch.tensor([T * N * n_chunks * f["pmax"]], dtype=torch.float64)
            k = int(torch.poisson(lam, generator=gen).item())
            if k == 0:
                continue
            ti = torch.randint(0, T, (k,), generator=gen)
            ni = torch.randint(0, N, (k,), generator=gen)
            ci = torch.randint(0, n_chunks, (k,), generator=gen)
            m = torch.empty(k, dtype=torch.int64)
            mask_x = (1 << f["n_bits"]) - 1
            for c_idx, (k0, k1) in enumerate(chunk_list):
                sel = torch.nonzero(ci == c_idx).flatten()
                if sel.numel() == 0:
                    continue
                d = (ux[ti[sel], k0:k1] >> f["bit_lo"]) & mask_x
                b = (uw[ni[sel], k0:k1] >> f["p"]) & 1
                m[sel] = (d * b).sum(1)
            m = m.clamp_max(f["pany"].numel() - 1)
            keep = torch.rand(k, generator=gen, dtype=torch.float64) * f["pmax"] < f["pany"][m]
            if not bool(keep.any()):
                continue
            mk = m[keep]
            u = torch.rand(mk.numel(), 1, generator=gen, dtype=torch.float64)
            jj = torch.searchsorted(f["cond_cdf"][mk], u).flatten().clamp_max(f["jvals"].numel() - 1)
            idx = ti[keep] * N + ni[keep]
            flat.index_add_(0, idx.to(y.device), ((f["coef"] * f["step"]) * f["jvals"][jj]).to(y.dtype).to(y.device))

    def _saturation_tables(self, hist: torch.Tensor | None = None) -> torch.Tensor:
        """S(m) tables, rebuilt when the ADC range or the relevant config fields change."""
        c = self.cfg
        fs = self.adc_fs if c.adc_range == "calibrated" else None
        key = (None if fs is None else tuple(fs.flatten().tolist()), c.adc_bits, c.nrow, c.x_bits, c.w_bits,
               c.act_enc, c.signed_act, c.w_quant, self._weight_matrix().shape[1])
        if self._sat_tables is None or self._sat_key != key:
            chunks = self._chunks()
            K = self._weight_matrix().shape[1]
            max_len = max(k1 - k0 for k0, k1 in chunks) if chunks else min(K, c.nrow)
            levels = max(cy.levels for cy in input_cycles(None, c.x_bits, c.signed_act, c.act_enc))
            M = max_len * levels
            hist = hist if hist is not None else self._det_hist
            if hist is not None and hist.shape[-1] < M + 1:
                hist = F.pad(hist, (0, M + 1 - hist.shape[-1]))
            self._sat_tables = deterministic_tables(c, None if fs is None else fs.cpu(), M,
                                                    None if hist is None else hist[..., :M + 1])[0]
            self._sat_key = key
        return self._sat_tables

    @staticmethod
    def _pad_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.shape[-1] < b.shape[-1]:
            a = F.pad(a, (0, b.shape[-1] - a.shape[-1]))
        elif b.shape[-1] < a.shape[-1]:
            b = F.pad(b, (0, a.shape[-1] - b.shape[-1]))
        return a + b

    def _compute(self, qx: torch.Tensor, sx: torch.Tensor) -> torch.Tensor:
        """Integer rows qx [T, K] with per-row (or per-tensor) scale sx -> dequantized [T, N]."""
        cfg = self.cfg
        qw, wscale, wzero = self._weights()
        y_int = self._int_matmul(qx, qw)
        if cfg.w_signed:
            return y_int * sx * wscale.t() * dequant_factor(cfg.w_bits, cfg.x_bits, cfg.signed_act)
        # asymmetric weights: subtract zero-point term digitally (exact), then scale
        y_int = y_int - qx.sum(dim=1, keepdim=True) * wzero.t()
        qmx = qmax_signed(cfg.x_bits) if cfg.signed_act else qmax_unsigned(cfg.x_bits)
        return y_int * (sx / qmx) * wscale.t()


class CIMLinear(_CIMBase):
    """Drop-in replacement for nn.Linear executed on a simulated SRAM CIM macro.

    weight: [out_features, in_features] (nn.Linear layout).
    """

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None, cfg: CIMConfig,
                 name: str = "", counter: EventCounter | None = None) -> None:
        super().__init__(cfg, name, counter)
        self.weight = nn.Parameter(weight.detach().clone(), requires_grad=False)
        self.bias = None if bias is None else nn.Parameter(bias.detach().clone(), requires_grad=False)
        self.out_features, self.in_features = self.weight.shape

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
                f"W{c.w_bits}A{c.x_bits}, nrow={c.nrow}, adc={c.adc_bits}, enc={c.act_enc}, "
                f"adc_range={c.adc_range}, w_quant={c.w_quant}")

    def _weight_matrix(self) -> torch.Tensor:
        return self.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        if cfg.mode == "fp" and not self._calibrating:
            return F.linear(x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))
        in_dtype = x.dtype
        lead = x.shape[:-1]
        x2 = x.reshape(-1, self.in_features).float()
        qx, sx = quantize_act(x2, cfg.x_bits, cfg.x_gran, cfg.signed_act)
        y = self._compute(qx, sx)
        if self.bias is not None:
            y = y + self.bias.float()
        self._count(x2.shape[0], self.in_features, self.out_features)
        return y.reshape(*lead, self.out_features).to(in_dtype)


class CIMConv2d(_CIMBase):
    """Drop-in replacement for nn.Conv2d (groups = 1) on a simulated SRAM CIM macro.

    The convolution is computed as an unfolded matmul: each output pixel is a
    row of K = C_in * kh * kw inputs. Rows are grouped per array activation by
    whole input channels, as in ASiM: floor(nrow / (kh * kw)) channels per group.
    (If kh * kw > nrow, one channel per group; the group is then longer than
    nrow and MAC values above the nominal full scale saturate.)

    Activation quantization granularity: "per_tensor" = whole input batch
    (ASiM); "per_token" = per image.
    """

    def __init__(self, conv: nn.Conv2d, cfg: CIMConfig, name: str = "", counter: EventCounter | None = None) -> None:
        super().__init__(cfg, name, counter)
        if conv.groups != 1:
            raise ValueError("CIMConv2d supports groups=1 only")
        if conv.padding_mode != "zeros":
            raise ValueError("CIMConv2d supports padding_mode='zeros' only")
        if isinstance(conv.padding, str):
            raise ValueError("CIMConv2d needs numeric padding")
        self.weight = nn.Parameter(conv.weight.detach().clone(), requires_grad=False)
        self.bias = None if conv.bias is None else nn.Parameter(conv.bias.detach().clone(), requires_grad=False)
        self.out_channels, self.in_channels, self.kh, self.kw = self.weight.shape
        self.stride, self.padding, self.dilation = conv.stride, conv.padding, conv.dilation

    @classmethod
    def from_conv2d(cls, conv: nn.Conv2d, cfg: CIMConfig, name: str = "", counter: EventCounter | None = None) -> "CIMConv2d":
        return cls(conv, cfg, name, counter)

    def extra_repr(self) -> str:
        c = self.cfg
        return (f"{self.in_channels}->{self.out_channels}, k={self.kh}x{self.kw}, stride={self.stride}, "
                f"pad={self.padding}, mode={c.mode}, W{c.w_bits}A{c.x_bits}, nrow={c.nrow}, adc={c.adc_bits}")

    def _weight_matrix(self) -> torch.Tensor:
        return self.weight.reshape(self.out_channels, -1)

    def _chunks(self) -> list[tuple[int, int]]:
        kk = self.kh * self.kw
        cpm = max(1, self.cfg.nrow // kk)
        return [(c0 * kk, min(self.in_channels, c0 + cpm) * kk) for c0 in range(0, self.in_channels, cpm)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        if cfg.mode == "fp" and not self._calibrating:
            return F.conv2d(x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype),
                            self.stride, self.padding, self.dilation)
        in_dtype = x.dtype
        B = x.shape[0]
        xf = x.float()
        # quantize on the image tensor, then unfold (zero padding stays 0)
        if cfg.x_gran == "per_tensor":
            qx4, sx = quantize_act(xf.reshape(1, -1), cfg.x_bits, "per_tensor", cfg.signed_act)
            qx4 = qx4.reshape_as(xf)
            sx_img = sx.reshape(1, 1).expand(B, 1)
        else:
            qx4, sx_img = quantize_act(xf.reshape(B, -1), cfg.x_bits, "per_token", cfg.signed_act)
            qx4 = qx4.reshape_as(xf)
        cols = F.unfold(qx4, (self.kh, self.kw), dilation=self.dilation, padding=self.padding, stride=self.stride)
        L = cols.shape[-1]
        qx = cols.transpose(1, 2).reshape(B * L, -1)                       # [B*L, K]
        sx = sx_img.repeat_interleave(L, dim=0)                              # [B*L, 1]
        y = self._compute(qx, sx)                                            # [B*L, O]
        Ho = (x.shape[2] + 2 * self.padding[0] - self.dilation[0] * (self.kh - 1) - 1) // self.stride[0] + 1
        Wo = (x.shape[3] + 2 * self.padding[1] - self.dilation[1] * (self.kw - 1) - 1) // self.stride[1] + 1
        y = y.reshape(B, Ho, Wo, self.out_channels).permute(0, 3, 1, 2)
        if self.bias is not None:
            y = y + self.bias.float().view(1, -1, 1, 1)
        self._count(B * L, self.in_channels * self.kh * self.kw, self.out_channels)
        return y.contiguous().to(in_dtype)
