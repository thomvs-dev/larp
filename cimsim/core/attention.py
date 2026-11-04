"""Attention matmuls with dynamic operands on CIM arrays (A4): S = Q·Kᵀ and O = P·V.

Mapping (one array set per attention layer and matmul):
- qk: keys are written into the array, one key vector per column (rows = head dim D). A query row is the
  bit-serial input. Contraction over D. Keys are quantized per channel by default (kv_gran 'auto' /
  'channel': one scale per head-dim row, folded into Q before Q is quantized) or per key token
  ('token': one scale per column, factors out of the MAC); queries per query token.
- sv: values are written into the array, one position per row (rows = key positions S, columns = D).
  The softmax probabilities P are the (unsigned) bit-serial input. Values are quantized per position by
  default (kv_gran 'auto' / 'token'; 'channel' = one scale per head-dim column); a per-position scale
  sits on the contraction axis, so it is folded into P digitally before P is quantized:
  O = sum_s (P[t, s] * sV[s]) * qV[s, :]. Contraction over S, split into chunks of nrow positions.

Both use the same integer path as CIMLinear (_CIMBase._int_matmul): fp / fake_quant / bit_accurate /
statistical, ADC calibration, noise, exact saturation. The dynamic operand plays the role of the weight
matrix for one call.

Events (per call): MACs and ADC conversions only for unmasked (query, key) pairs; for sv the number of
row chunks of query t is ceil(valid keys of t / nrow). kv_bits_written counts the keys and values of the
new tokens of this call (prefill: all tokens; decode: one), once per KV head (GQA heads share them).

Limits: in mode 'statistical' the Gaussian part of sv is scaled with the chunk count of the call
(calibration and evaluation use the same window length); chunks that are fully masked for a query are
still simulated (with zero inputs) in bit-accurate mode, so noise on them is included there.

Use: swap_attention(model, cfg_map, counter) attaches the modules and selects the 'cim' attention
implementation; layer names are '<attention module>.qk' and '<attention module>.sv'.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from cimsim.config import CIMConfig, LayerConfigMap
from cimsim.core.bitplanes import input_cycles
from cimsim.core.layers import _CIMBase
from cimsim.events import EventCounter
from cimsim.quant import quantize_act, quantize_weight, dequant_factor


class CIMMatMul(_CIMBase):
    """a [T, K] @ b[N, K]ᵀ with both operands produced at run time. kind 'qk' or 'sv' (see module doc)."""

    def __init__(self, cfg: CIMConfig, kind: str, name: str = "", counter: EventCounter | None = None) -> None:
        if kind not in ("qk", "sv"):
            raise ValueError(kind)
        if cfg.w_quant != "symmetric":
            raise ValueError("CIMMatMul supports symmetric (two's complement) operands only")
        # sv inputs are softmax probabilities (>= 0): unsigned bit-serial input
        cfg = cfg.replace(signed_act=(kind == "qk"), x_gran="per_token", w_gran="per_channel")
        super().__init__(cfg, name, counter)
        self.kind = kind
        self._cur_w: torch.Tensor | None = None      # [N, K] integer codes of the current call
        self._cal_K: int | None = None               # contraction length seen during calibration
        self._dynamic_operand = True

    def extra_repr(self) -> str:
        c = self.cfg
        return (f"kind={self.kind}, mode={c.mode}, macro={c.macro}, W{c.w_bits}A{c.x_bits}, nrow={c.nrow}, "
                f"adc={c.adc_bits}, adc_range={c.adc_range}")

    # _CIMBase hooks: the dynamic operand is the weight matrix of the call
    def _weight_matrix(self) -> torch.Tensor:
        if self._cur_w is None:
            raise RuntimeError(f"{self.name}: no operand set")
        return self._cur_w

    def _weights(self):
        return self._cur_w, None, None

    def _n_chunks(self) -> int:
        K = self._cal_K if self._cal_K is not None else self._cur_w.shape[1]
        return math.ceil(K / self.cfg.nrow)

    def requantize(self) -> None:
        pass

    def finish_calibration(self) -> None:
        super().finish_calibration()
        self.sat_cols = None                         # columns change every call: strict saturation bound

    def _int_matmul(self, qx: torch.Tensor, qw: torch.Tensor) -> torch.Tensor:
        if self._calibrating:
            self._cal_K = qx.shape[1]
        cfg = self.cfg
        if (cfg.mode == "statistical" and not self._calibrating and not self._ideal_override
                and cfg.macro == "acim" and self.stat_params is not None and self._cal_K):
            # Gaussian part and mean scale with the number of row chunks of this call
            r = math.ceil(qx.shape[1] / cfg.nrow) / math.ceil(self._cal_K / cfg.nrow)
            if r != 1.0:
                saved = self.stat_params
                g, mu, sd = (float(v) for v in saved)
                self.stat_params = torch.tensor([g, mu * r, sd * math.sqrt(r)], dtype=torch.float64)
                try:
                    return super()._int_matmul(qx, qw)
                finally:
                    self.stat_params = saved
        return super()._int_matmul(qx, qw)

    def _count(self, *a, **k) -> None:              # counted per head in matmul()
        pass

    def fold_contraction_scale(self) -> bool:
        """True if the stored operand is quantized with one scale per contraction index (folded into the
        input): qk with keys per channel, sv with values per token."""
        g = self.cfg.kv_gran
        if g == "auto":
            g = "channel" if self.kind == "qk" else "token"
        return (self.kind == "qk") == (g == "channel")

    @torch.no_grad()
    def matmul(self, a: torch.Tensor, b: torch.Tensor, valid: torch.Tensor | None = None,
               b_rowscale_into_a: bool | None = None) -> torch.Tensor:
        """a [T, K] float, b [N, K] float -> a @ bᵀ [T, N] through the CIM path.
        b_rowscale_into_a: quantize b with one scale per contraction index k (column of b) and multiply that
        scale into a before a is quantized (default from cfg.kv_gran, see fold_contraction_scale).
        valid: bool [T, N] (qk) or [T, K] (sv) of unmasked pairs, used for event counts only."""
        cfg = self.cfg
        if cfg.mode == "fp" and not self._calibrating:
            return a @ b.t()
        if b_rowscale_into_a is None:
            b_rowscale_into_a = self.fold_contraction_scale()
        if b_rowscale_into_a:
            qv, sv = quantize_weight(b.t().float(), cfg.w_bits, "per_channel")     # [S, D] codes, [S, 1]
            a = a.float() * sv.t()
            qb = qv.t().contiguous()
            bscale = None
        else:
            qb, bscale = quantize_weight(b.float(), cfg.w_bits, "per_channel")      # [N, K], [N, 1]
        qa, sa = quantize_act(a.float(), cfg.x_bits, "per_token", cfg.signed_act)
        self._cur_w = qb
        try:
            y = self._int_matmul(qa, qb)
        finally:
            self._cur_w = None
        f = dequant_factor(cfg.w_bits, cfg.x_bits, cfg.signed_act)
        y = y * sa * f
        if bscale is not None:
            y = y * bscale.t()
        self._count_call(a.shape[0], a.shape[1], b.shape[0], valid)
        return y

    def _count_call(self, T: int, K: int, N: int, valid: torch.Tensor | None) -> None:
        if self.counter is None or self._calibrating:
            return
        cfg = self.cfg
        C = len(input_cycles(None, cfg.x_bits, cfg.signed_act, cfg.act_enc))
        P = cfg.w_bits
        if valid is None:
            v = torch.full((T,), K if self.kind == "sv" else N, dtype=torch.int64)
        else:
            v = valid.sum(1).to(torch.int64)                                      # valid keys per query
        if self.kind == "qk":
            macs = int(v.sum()) * K
            chunks_t = torch.full_like(v, math.ceil(K / cfg.nrow))
            conv = int((chunks_t * v).sum()) * C * P                              # one per (key column, chunk)
            cycles = int(chunks_t.sum()) * C
        else:
            macs = int(v.sum()) * N
            chunks_t = (v + cfg.nrow - 1) // cfg.nrow
            conv = int(chunks_t.sum()) * C * P * N
            cycles = int(chunks_t.sum()) * C
        if cfg.macro == "digital":
            self.counter.add(self.name, macs=macs, digital_macs=macs)
        elif cfg.macro == "dcim":
            self.counter.add(self.name, macs=macs, array_cycles=cycles, dcim_column_ops=conv)
        else:
            from cimsim.core.executor import analog_mask
            n_an = int(analog_mask(P, C, cfg.hybrid_levels).sum())
            self.counter.add(self.name, macs=macs, adc_conversions=conv,
                             adc_conversions_analog=conv // (C * P) * n_an, array_cycles=cycles)

    def count_kv_write(self, n_tokens: int, n_kv_heads: int, head_dim: int, batch: int) -> None:
        if self.counter is None or self._calibrating or self.cfg.mode == "fp" or self.cfg.macro == "digital":
            return
        self.counter.add(self.name, kv_bits_written=batch * n_kv_heads * n_tokens * head_dim * self.cfg.w_bits)


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


def cim_attention_forward(module: nn.Module, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                          attention_mask: torch.Tensor | None, scaling: float | None = None,
                          dropout: float = 0.0, **kwargs):
    """Eager attention with QKᵀ and P·V on module.cim_qk / module.cim_sv. Same signature and outputs as the
    HF eager attention functions (query [B, H, T, D], key / value [B, Hkv, S, D], additive float mask)."""
    qk: CIMMatMul = module.cim_qk
    sv: CIMMatMul = module.cim_sv
    if scaling is None:
        scaling = query.size(-1) ** -0.5
    B, H, T, D = query.shape
    Hkv, S = key.shape[1], key.shape[2]
    n_rep = H // Hkv
    k_all, v_all = _repeat_kv(key, n_rep), _repeat_kv(value, n_rep)
    qk.count_kv_write(T, Hkv, D, B)                   # new tokens of this call: keys ...
    sv.count_kv_write(T, Hkv, value.shape[-1], B)     # ... and values
    out = torch.empty(B, H, T, value.shape[-1], dtype=torch.float32, device=query.device)
    for bi in range(B):
        m = None if attention_mask is None else attention_mask[bi if attention_mask.shape[0] > 1 else 0]
        for hi in range(H):
            mh = None if m is None else m[hi if m.shape[0] > 1 else 0][:, :S].float()
            valid = None if mh is None else (mh > torch.finfo(torch.float32).min / 2)
            s = qk.matmul(query[bi, hi].float(), k_all[bi, hi].float(), valid) * scaling
            if mh is not None:
                s = s + mh
            p = torch.softmax(s, dim=-1)
            o = sv.matmul(p, v_all[bi, hi].float().t(), valid)
            out[bi, hi] = o
    return out.to(query.dtype).transpose(1, 2).contiguous(), None


_REGISTERED = False


def register_cim_attention() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    from transformers import AttentionInterface
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, AttentionMaskInterface
    AttentionInterface.register("cim", cim_attention_forward)
    AttentionMaskInterface.register("cim", ALL_MASK_ATTENTION_FUNCTIONS["eager"])
    _REGISTERED = True


def _is_attention(m: nn.Module) -> bool:
    return type(m).__name__.endswith("Attention") and (hasattr(m, "q_proj") or hasattr(m, "c_attn"))


def swap_attention(model: nn.Module, cfg_map: LayerConfigMap, counter: EventCounter | None = None) -> list[str]:
    """Attach CIMMatMul modules (cim_qk, cim_sv) to every attention module and select the 'cim' attention
    implementation. Config for each comes from cfg_map.for_layer('<module>.qk' / '<module>.sv').
    Returns the new layer names."""
    register_cim_attention()
    names = []
    for name, m in model.named_modules():
        if not _is_attention(m):
            continue
        m.cim_qk = CIMMatMul(cfg_map.for_layer(f"{name}.qk"), "qk", f"{name}.qk", counter)
        m.cim_sv = CIMMatMul(cfg_map.for_layer(f"{name}.sv"), "sv", f"{name}.sv", counter)
        names += [f"{name}.qk", f"{name}.sv"]
    if not names:
        return names
    for mod in model.modules():
        cfg = getattr(mod, "config", None)
        if cfg is not None and hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = "cim"
    return names
