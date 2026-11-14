# Noise per layer type: statistical vs bit-accurate check (sandbox, 2026-11-14)

OPT-125M, base `a4_stat_adc6cal.yaml` (all layers nrow 128, calibrated ADC 6 b, statistical, noise-free), noise 0.10 % on the listed fc1 layers only, first 4,096 WikiText-2 test tokens (2 windows), 2 sandbox threads, git 78c9d39 + working tree (no code change). "BA" = those fc1 layers bit-accurate, all other layers statistical (noise-free); "stat" = everything statistical. Noise-free: 25.594 (stat), 25.591 (fc1 layer 1 bit-accurate).

| Noisy fc1 layers | Stat (seed 0 / 1 / 2 / 3) | BA, layer generator (seed 0 / 1 / 2 / 3) | BA, independent generator 500+seed (0 / 1 / 2 / 3) |
|---|---|---|---|
| all 12 | 217.1 / 265.9 | 121.9 | — |
| 0–5 | 215.1 / 257.4 | 121.2 / 131.1 | — |
| 0 | 25.63 | 25.65 | — |
| 1 | 371.3 / 79.7 / 277.5 / 116.5 (harness seed 0: 261.6) | 88.09 / 87.96 / 89.35 / 125.19 | 244.8 / 52.0 / 145.3 / 76.8 |
| 2 | 32.50 | 50.07 | — |
| 3 | 25.75 | 26.04 | — |
| 4 | 25.64 | 25.65 | — |
| 5 | 25.62 / 25.60 / 25.68 | 25.65 / 25.62 / 25.55 | — |

Layer 1 fc1 at sigma 0.001 % and 0.05 % bit-accurate: 25.591 (= noise-free).

Layer-level comparison on the same inputs (fc1, all 12 layers, 512 tokens; layer 1 alone, 2,048 tokens; layer 1, 96 tokens × 60 repeats): error std ratio stat / BA 1.00–1.05, same fraction of non-zero outputs, same kurtosis, same per-column mean error (corr 0.89 on layer 1, the large deterministic columns 2019 / 1540 / 837 equal), same per-token error energy, same split of errors between positive and negative ideal outputs, same mean variance over repeats (264,496 vs 264,570). Flip tables checked: m = 0 flips up only, saturated m (> full scale) no flips, signs of the input-sign cycle correct.

Reading:
- Damage comes almost entirely from layers 1 and 2; layers 0, 3, 4, 5 alone change PPL by < 2 %.
- Bit-accurate PPL at this point is itself chaotic: layer 1 alone gives 52–245 over independent generators (sd of log PPL ≈ 0.5). The tight bit-accurate triple on the laptop (122 / 135 / 123, all fc1) and 88.1 / 88.0 / 89.4 here come from the per-layer generator streams of seeds 0–2; why these streams give nearly equal PPL is not explained (the streams themselves are independent: correlation of draws < 3e-4, no shared flip positions).
- Pooling the 8 bit-accurate and 5 statistical layer-1 values: geometric means 102 vs 190, Mann-Whitney one-sided p ≈ 0.06. Statistical may still be ~2× pessimistic here; not proven either way with these sample sizes.
- No layer-level difference was found between the two modes.

Files: `scripts/` (harnesses as run), `configs/`, one folder per run with `result.json`.
