# A4: full WikiText-2 test-set perplexity, result models

Thomas's laptop (i7-12700H, 6 threads, CPU float32), git 811eead, 2,048-token non-overlapping windows,
statistical runs calibrated on 2,048 WikiText-2 train tokens. Config `a4_stat_adc6cal.yaml`: W8A8
(per-channel weights, per-token activations), bit-serial inputs, nrow 128, 6-bit ADC with MSE-calibrated
range, noise-free, all linear layers on analog CIM, LM head and attention digital.

| Model | Tokens | FP32 | W8A8 fake-quant | Analog ADC 6 b cal (statistical) | analog − W8A8 per-window NLL (nats) |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | 298,862 | 13.069 | 13.332 (+2.01 %) | 13.329 (−0.03 % vs W8A8) | −0.00026 ± 0.00035 (not significant) |
| TinyLlama-1.1B (v1.1) | 339,802 | 7.707 | 7.793 (+1.12 %) | 7.793 (−0.01 % vs W8A8) | −0.00012 ± 0.00019 (not significant) |
| Qwen2.5-1.5B (2026-11-05, git 2d1d11f) | 298,862 | 9.264 | 9.487 (+2.41 %) | 9.492 (+0.05 %) | +0.00047 ± 0.00042 (not significant) |

The ± is one standard error of the mean per-window difference (146 / 166 windows). W8A8 − FP32 is
significant on both new models (+0.0199 ± 0.0005 nats Qwen2.5-0.5B, +0.0111 ± 0.0005 TinyLlama).
With a calibrated 6-bit ADC and no analog noise, the CIM error adds nothing measurable on top of
8-bit quantization on three models (F5).

Runtime (evaluation only, calibration separate):

| Model | FP32 | W8A8 | Statistical | Calibration |
|---|---|---|---|---|
| Qwen2.5-0.5B | 22 min (4.5 ms/token) | 37 min (7.4) | 83 min (16.6) | 22 min |
| TinyLlama-1.1B | 74 min (13.0) | 61 min (10.8) | 214 min (37.9) | 32 min |

TinyLlama FP32 was slower than W8A8; not investigated (other load on the laptop is the likely cause).
Statistical mode costs 2.2–3.5× W8A8 fake-quant on 2,048-token windows with the code at 811eead
(Qwen2.5-1.5B before the speed-up: 69 ms/token).

Result folders: `results/e0_qwen05b_fp32/`, `results/e0_qwen05b_w8a8/`, `results/a4_qwen05b_stat_adc6cal/`,
`results/e0_tinyllama_fp32/`, `results/e0_tinyllama_w8a8/`, `results/a4_tinyllama_stat_adc6cal/`.
