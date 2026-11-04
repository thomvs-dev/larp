# A4 — attention matmuls (Q·Kᵀ, P·V) on CIM arrays — Qwen2.5-0.5B, sandbox

Cloud sandbox, 2 threads, Qwen2.5-0.5B (bf16 copy loaded as float32), first 2,048 WikiText-2 test tokens in
4 windows of **512** tokens (2,048-token windows do not fit the sandbox's 7 GB: logits + log-softmax of a
152k vocabulary ≈ 2.5 GB). Calibration on 512 train tokens. Code: `cimsim/core/attention.py` (runs used the
working tree on top of git 2d1d11f; committed with this file).

| Run | config | PPL | results |
|---|---|---|---|
| FP32 | none | 17.411 | `results/a4s_qwen05b_fp32_s512_sandbox/` |
| W8A8 linear (ideal) | `e0_w8a8.yaml` | 17.895 | `results/a4s_qwen05b_w8a8_s512_sandbox/` |
| + 8-bit Q·Kᵀ only, keys per **token** | `a4_attn_qk_only_ktoken.yaml` | 18.642 | `results/a4s_qwen05b_attn_qk_only_ktoken_s512_sandbox/` |
| + 8-bit Q·Kᵀ only, keys per **channel** | `a4_attn_qk_only_kchannel.yaml` | 17.636 | `results/a4s_qwen05b_attn_qk_only_kchannel_s512_sandbox/` |
| + 8-bit Q, K, P, V (keys per channel, values per token) | `a4_attn_w8a8.yaml` | 17.737 | `results/a4s_qwen05b_w8a8_attn_s512_sandbox/` |
| + attention on analog CIM, ADC 6 b calibrated: statistical | `a4_attn_only_adc6cal.yaml` | 17.596 | `results/a3g_qwen05b_attn_only_adc6cal_sandbox/` |
| same, bit-accurate | (same run) | 17.596 | (same) |
| all linear layers on analog CIM (ADC 6 b cal.), attention float | `a4_stat_adc6cal.yaml` | 17.838 | `results/a4s_qwen05b_stat_adc6cal_s512_sandbox/` |
| all linear layers **and** attention on analog CIM | `a4_stat_adc6cal_attn.yaml` | 17.620 | `results/a4s_qwen05b_stat_adc6cal_attn_s512_sandbox/` |

Findings (see `docs/findings.md` F10):
- 8-bit Q·Kᵀ with one scale per **key token** costs +4.2 % over W8A8 linear (18.642 vs 17.895). Qwen's keys
  have a few channels much larger than the rest; a per-token scale leaves the other channels few levels.
  One scale per **key channel** (folded into Q before Q is quantized) removes the loss (17.636). Default
  `kv_gran: auto` = keys per channel, values per token.
- With that, 8-bit attention operands and analog attention with a calibrated 6-bit ADC show no measurable
  loss: all runs with quantized attention lie within ±1.5 % of W8A8 linear, which is the run-to-run scale
  of differences at 2,048 tokens (lower values than W8A8 linear are not an improvement).
- Statistical = bit-accurate exactly on the attention-only analog run (17.596 both, 2.3× faster; exact
  saturation work 0.09 % of MACs).
- Cost of attention at 512-token windows: +3.1 % MACs (7.55e11 vs 7.33e11), +5.0 % ADC conversions
  (3.85e11 vs 3.66e11), 1.0e8 KV bits written for 2,048 tokens (24 layers × 2 KV heads × 64 × 8 b × K and V).
  Attention's share grows linearly with context length.
- Not yet run on Thomas's machine at 2,048-token windows or on Qwen2.5-1.5B.
