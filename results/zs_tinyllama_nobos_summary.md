# TinyLlama-1.1B v1.1 zero-shot without BOS (A4)

Thomas's laptop, git 6169ebe, lm-eval 0.4.13, `add_bos_token=False` (findings F11), batch 8, first 1,000 documents per task, 0-shot, 6 threads. Replaces `results/zs_tinyllama_summary.md` (those runs had `<s>` at the start of every request).
Analog config: `config/experiments/a4_stat_adc6cal.yaml` (W8A8, bit-serial, nrow 128, ADC 6 b MSE-calibrated, statistical mode); calibration chose FS 63 on all 154 layers.

Reference: `results/zs_tinyllama_w8a8_nobos`. Δ = run − reference on the same documents (pp); 95 % interval from the paired standard error; p = exact McNemar.

| Task | Metric | Run | n | Acc | Ref acc | Δ (pp) | 95 % interval (pp) | p |
|---|---|---|---|---|---|---|---|---|
| arc_easy | acc | `zs_tinyllama_fp32_nobos` | 1000 | 62.20 | 62.10 | +0.10 | -1.09 … +1.29 | 1 |
| arc_easy | acc | `zs_tinyllama_stat_adc6cal_nobos` | 1000 | 62.90 | 62.10 | +0.80 | -0.20 … +1.80 | 0.169 |
| arc_easy | acc_norm | `zs_tinyllama_fp32_nobos` | 1000 | 57.20 | 57.20 | +0.00 | -1.27 … +1.27 | 1 |
| arc_easy | acc_norm | `zs_tinyllama_stat_adc6cal_nobos` | 1000 | 56.70 | 57.20 | -0.50 | -1.44 … +0.44 | 0.405 |
| hellaswag | acc | `zs_tinyllama_fp32_nobos` | 1000 | 44.60 | 44.50 | +0.10 | -0.61 … +0.81 | 1 |
| hellaswag | acc | `zs_tinyllama_stat_adc6cal_nobos` | 1000 | 44.20 | 44.50 | -0.30 | -0.82 … +0.22 | 0.453 |
| hellaswag | acc_norm | `zs_tinyllama_fp32_nobos` | 1000 | 54.70 | 54.80 | -0.10 | -1.00 … +0.80 | 1 |
| hellaswag | acc_norm | `zs_tinyllama_stat_adc6cal_nobos` | 1000 | 54.70 | 54.80 | -0.10 | -0.62 … +0.42 | 1 |
| piqa | acc | `zs_tinyllama_fp32_nobos` | 1000 | 72.20 | 72.60 | -0.40 | -1.23 … +0.43 | 0.481 |
| piqa | acc | `zs_tinyllama_stat_adc6cal_nobos` | 1000 | 72.60 | 72.60 | +0.00 | -0.55 … +0.55 | 1 |
| piqa | acc_norm | `zs_tinyllama_fp32_nobos` | 1000 | 74.00 | 74.20 | -0.20 | -1.12 … +0.72 | 0.832 |
| piqa | acc_norm | `zs_tinyllama_stat_adc6cal_nobos` | 1000 | 74.10 | 74.20 | -0.10 | -0.69 … +0.49 | 1 |

Result: FP32, W8A8 and analog agree on every task and metric (|Δ| ≤ 0.8 pp, all 95 % intervals contain 0, McNemar p ≥ 0.17). The FP32-below-W8A8 gap of the earlier runs is gone.

Runtime (ms/token ARC-e / PIQA / HellaSwag): FP32 9.0 / 15.4 / 8.1 (67 min), W8A8 10.7 / 10.6 / 10.1 (71 min), analog statistical 30.0 / 27.8 / 26.8 (3.2 h) + calibration 36.8 min.
