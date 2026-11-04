# A4 smoke runs — Qwen2.5-0.5B (Thomas's machine, git 6bf9d61, 14 threads)

First 8,192 WikiText-2 test tokens (4 windows of 2,048). Statistical config `config/experiments/a4_stat_adc6cal.yaml`:
bit-serial W8A8, nrow 128, ADC 6 b on all 168 linear layers, MSE-calibrated full scale per layer
(2,048 train tokens), no analog noise. LM head digital (tied embeddings).

| Run | PPL | vs FP32 | vs W8A8 | s (eval) | ms/token | results |
|---|---|---|---|---|---|---|
| FP32 | 14.569 | — | — | 24.4 | 3.0 | `results/e0_qwen05b_fp32_sub8k/` |
| W8A8 fake-quant (per-channel / per-token) | 14.821 | +1.73 % | — | 34.2 | 4.2 | `results/e0_qwen05b_w8a8_sub8k/` |
| statistical, ADC 6 b calibrated | 14.809 | +1.64 % | −0.08 % | 164.9 | 20.1 | `results/a4_qwen05b_stat_adc6cal_sub8k/` |

- Calibration chose full scale 63 (ADC step = 1 MAC unit) on all 168 layers, as on OPT-125M (F5): a
  calibrated 6-bit ADC adds no measurable loss on Qwen2.5-0.5B without noise.
- W8A8 alone costs +1.7 % on Qwen2.5-0.5B (OPT-125M full set: +1.0 %, F1); the ADC adds nothing on top.
- Calibration took 1,461 s (0.71 s per calibration token, more than a bit-accurate pass). The MAC
  histogram has since been rewritten (float32 + one `torch.histc` per block, identical output, 2.1× faster
  on a 896→4864 layer in the sandbox).
- `events.json` of the statistical run includes the 2,048 calibration tokens (counts for 10,240 tokens;
  multiply every count except `weight_bits_written` by 8,192 / 10,240). `run_ppl.py` now resets the
  counter after calibration.
- Not yet checked: statistical vs bit-accurate on Qwen (GATE A3 used OPT-125M only). Planned:
  `scripts/a3_gate.py --model Qwen/Qwen2.5-0.5B` on 4,096 tokens.

# Qwen2.5-1.5B (Thomas's machine, git 0e2ca2c, 14 threads, same protocol)

| Run | PPL | vs FP32 | vs W8A8 | s (eval) | ms/token | results |
|---|---|---|---|---|---|---|
| FP32 | 10.113 | — | — | 95.6 | 11.7 | `results/e0_qwen15b_fp32_sub8k/` |
| W8A8 fake-quant | 10.338 | +2.23 % | — | 129.4 | 15.8 | `results/e0_qwen15b_w8a8_sub8k/` |
| statistical, ADC 6 b calibrated | 10.342 | +2.27 % | +0.03 % | 539.5 | 65.9 | `results/a4_qwen15b_stat_adc6cal_sub8k/` |

- Full scale 63 (1 MAC unit per ADC step) on all 196 layers; the analog macro adds +0.03 % over W8A8.
- W8A8 costs more on the larger Qwen (+2.2 % vs +1.7 % for 0.5B); the loss is in the 8-bit activation /
  weight quantization, not in the CIM array.
- Calibration 3,193 s (53 min) with the faster histogram — the main cost of a run; evaluation 66 ms/token
  (full WikiText-2 test set, 299k tokens: ~5.5 h).
- events.json now covers only the evaluated tokens (MACs equal to the W8A8 run: 1.07e13). ADC conversions
  per token: 6.55e8 (nrow 128) vs 3.28e8 for the e0 W8A8 config (default nrow 256).
- Ran within the laptop's 16 GB (peak not reported).
- Statistical vs bit-accurate on Qwen not yet checked: the first gate attempt stopped because
  `scripts/a3_gate.py` only accepted configs with bit_accurate layers (fixed: it now takes statistical configs too).
