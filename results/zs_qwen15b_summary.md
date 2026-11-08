Reference: `results/zs_qwen15b_w8a8`. Δ = run − reference on the same documents (pp); 95 % interval from the paired standard error; p = exact McNemar.

| Task | Metric | Run | n | Acc | Ref acc | Δ (pp) | 95 % interval (pp) | p |
|---|---|---|---|---|---|---|---|---|
| arc_easy | acc | `zs_qwen15b_fp32` | 1000 | 75.00 | 74.20 | +0.80 | -0.47 … +2.07 | 0.28 |
| arc_easy | acc | `zs_qwen15b_stat_adc6cal` | 1000 | 73.90 | 74.20 | -0.30 | -0.82 … +0.22 | 0.453 |
| arc_easy | acc_norm | `zs_qwen15b_fp32` | 1000 | 70.80 | 71.30 | -0.50 | -2.15 … +1.15 | 0.635 |
| arc_easy | acc_norm | `zs_qwen15b_stat_adc6cal` | 1000 | 71.00 | 71.30 | -0.30 | -0.89 … +0.29 | 0.508 |
| hellaswag | acc | `zs_qwen15b_fp32` | 1000 | 45.90 | 45.60 | +0.30 | -0.72 … +1.32 | 0.701 |
| hellaswag | acc | `zs_qwen15b_stat_adc6cal` | 1000 | 45.50 | 45.60 | -0.10 | -0.54 … +0.34 | 1 |
| hellaswag | acc_norm | `zs_qwen15b_fp32` | 1000 | 58.30 | 59.00 | -0.70 | -1.79 … +0.39 | 0.281 |
| hellaswag | acc_norm | `zs_qwen15b_stat_adc6cal` | 1000 | 59.00 | 59.00 | +0.00 | -0.62 … +0.62 | 1 |
| piqa | acc | `zs_qwen15b_fp32` | 1000 | 75.60 | 75.60 | +0.00 | -1.04 … +1.04 | 1 |
| piqa | acc | `zs_qwen15b_stat_adc6cal` | 1000 | 75.70 | 75.60 | +0.10 | -0.34 … +0.54 | 1 |
| piqa | acc_norm | `zs_qwen15b_fp32` | 1000 | 76.00 | 76.00 | +0.00 | -1.14 … +1.14 | 1 |
| piqa | acc_norm | `zs_qwen15b_stat_adc6cal` | 1000 | 76.00 | 76.00 | +0.00 | -0.68 … +0.68 | 1 |

Analog statistical vs W8A8: |Δ| ≤ 0.3 pp on every task and metric, all intervals contain 0, McNemar p ≥ 0.45. FP32 vs W8A8 ≤ 0.8 pp, p ≥ 0.28.

Runtime (Thomas's laptop, git f5560a1, 6 threads, batch 8): FP32 12.6–12.9 ms/token (74 min), W8A8 14.4–15.1 (86 min),
analog statistical 62.0–71.3 (6.4 h) + calibration 84.7 min.
