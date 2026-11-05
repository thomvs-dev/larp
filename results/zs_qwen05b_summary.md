Reference: `results/zs_qwen05b_w8a8`. Δ = run − reference on the same documents (pp); 95 % interval from the paired standard error; p = exact McNemar.

| Task | Metric | Run | n | Acc | Ref acc | Δ (pp) | 95 % interval (pp) | p |
|---|---|---|---|---|---|---|---|---|
| arc_easy | acc | `zs_qwen05b_fp32` | 1000 | 63.50 | 62.40 | +1.10 | -0.33 … +2.53 | 0.169 |
| arc_easy | acc | `zs_qwen05b_stat_adc6cal` | 1000 | 62.10 | 62.40 | -0.30 | -0.89 … +0.29 | 0.508 |
| arc_easy | acc_norm | `zs_qwen05b_fp32` | 1000 | 59.40 | 58.50 | +0.90 | -0.44 … +2.24 | 0.243 |
| arc_easy | acc_norm | `zs_qwen05b_stat_adc6cal` | 1000 | 58.70 | 58.50 | +0.20 | -0.19 … +0.59 | 0.625 |
| hellaswag | acc | `zs_qwen05b_fp32` | 1000 | 40.10 | 40.00 | +0.10 | -0.71 … +0.91 | 1 |
| hellaswag | acc | `zs_qwen05b_stat_adc6cal` | 1000 | 40.00 | 40.00 | +0.00 | -0.62 … +0.62 | 1 |
| hellaswag | acc_norm | `zs_qwen05b_fp32` | 1000 | 49.40 | 49.70 | -0.30 | -1.49 … +0.89 | 0.743 |
| hellaswag | acc_norm | `zs_qwen05b_stat_adc6cal` | 1000 | 50.10 | 49.70 | +0.40 | -0.15 … +0.95 | 0.289 |
| piqa | acc | `zs_qwen05b_fp32` | 1000 | 70.80 | 71.30 | -0.50 | -1.63 … +0.63 | 0.487 |
| piqa | acc | `zs_qwen05b_stat_adc6cal` | 1000 | 71.70 | 71.30 | +0.40 | -0.28 … +1.08 | 0.388 |
| piqa | acc_norm | `zs_qwen05b_fp32` | 1000 | 70.60 | 70.20 | +0.40 | -0.74 … +1.54 | 0.608 |
| piqa | acc_norm | `zs_qwen05b_stat_adc6cal` | 1000 | 70.00 | 70.20 | -0.20 | -0.75 … +0.35 | 0.727 |
