Reference: `results/zs_tinyllama_w8a8`. Δ = run − reference on the same documents (pp); 95 % interval from the paired standard error; p = exact McNemar.

| Task | Metric | Run | n | Acc | Ref acc | Δ (pp) | 95 % interval (pp) | p |
|---|---|---|---|---|---|---|---|---|
| arc_easy | acc | `zs_tinyllama_fp32` | 1000 | 47.50 | 55.90 | -8.40 | -10.93 … -5.87 | 1.37e-10 |
| arc_easy | acc | `zs_tinyllama_stat_adc6cal` | 1000 | 56.30 | 55.90 | +0.40 | -0.56 … +1.36 | 0.541 |
| arc_easy | acc_norm | `zs_tinyllama_fp32` | 1000 | 45.00 | 50.20 | -5.20 | -7.36 … -3.04 | 3.39e-06 |
| arc_easy | acc_norm | `zs_tinyllama_stat_adc6cal` | 1000 | 50.30 | 50.20 | +0.10 | -0.71 … +0.91 | 1 |
| hellaswag | acc | `zs_tinyllama_fp32` | 1000 | 37.20 | 38.90 | -1.70 | -3.46 … +0.06 | 0.0748 |
| hellaswag | acc | `zs_tinyllama_stat_adc6cal` | 1000 | 38.90 | 38.90 | +0.00 | -0.39 … +0.39 | 1 |
| hellaswag | acc_norm | `zs_tinyllama_fp32` | 1000 | 45.80 | 49.50 | -3.70 | -6.02 … -1.38 | 0.00231 |
| hellaswag | acc_norm | `zs_tinyllama_stat_adc6cal` | 1000 | 49.20 | 49.50 | -0.30 | -1.06 … +0.46 | 0.607 |
| piqa | acc | `zs_tinyllama_fp32` | 1000 | 69.00 | 71.20 | -2.20 | -3.76 … -0.64 | 0.00815 |
| piqa | acc | `zs_tinyllama_stat_adc6cal` | 1000 | 71.00 | 71.20 | -0.20 | -0.93 … +0.53 | 0.791 |
| piqa | acc_norm | `zs_tinyllama_fp32` | 1000 | 70.40 | 71.60 | -1.20 | -2.72 … +0.32 | 0.155 |
| piqa | acc_norm | `zs_tinyllama_stat_adc6cal` | 1000 | 70.60 | 71.60 | -1.00 | -1.96 … -0.04 | 0.0639 |

**FP32 rows are not trusted.** FP32 is 8.4 pp below W8A8 on ARC-e (p = 1e-10) and below it on every task, spread evenly over the
1,000 documents, while TinyLlama FP32 perplexity on the same laptop is lower than W8A8 (7.707 vs 7.793,
`results/a4_ppl_full_summary.md`). Quantization does not add 8 pp of accuracy, so the FP32 zero-shot run is suspected faulty. Cause not found:
a right-padded batch of 8 gives the same logits as single requests on a small random Llama (GQA, SDPA) in the sandbox. Batch 1 ruled out (2026-11-08, git 1f4ff48, `results/zs_tinyllama_fp32_b1_diag/`): FP32 at batch 1 on the first 200 ARC-e documents gives acc 45.0 / acc_norm 44.0, identical to the batch-8 FP32 run on the same documents (W8A8 52.5 / 51.0, analog 52.0 / 50.5). Swap path ruled out (2026-11-08, git 0a37d80, `results/zs_tinyllama_fpswap_diag/`): all 154 linear layers swapped in `mode: fp` give 45.0 / 44.0 on the same 200 documents, identical to the unswapped model. So the 7 pp gap is caused by 8-bit quantization itself, not by batching or the swap code. Both FP32 (47.5) and the analog / W8A8 runs are below published TinyLlama ARC-e figures (about 55–62), so the lm-eval setup for this model needs checking (BOS / prompt handling is the first suspect; Qwen, which has no BOS token, is not affected). Investigation continues on a local copy of the model. The analog − W8A8 rows compare two runs through the same swapped-model path and are not affected.

Runtime (Thomas's laptop, git f5560a1, 6 threads, batch 8): FP32 11.5–12.7 ms/token (84 min), W8A8 9.8–10.3 (70 min),
analog statistical 33.9–39.6 (4.2 h) + calibration 38.5 min.
