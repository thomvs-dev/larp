# GATE A3 — evaluation of the v1 chain (Thomas's machine, git 33e49ce) with statistical v2 numbers

Bit-accurate and v1 statistical PPLs: `results/a3g_opt125m_*/gate.json` (14 threads, i7-12700H).
Statistical v2 PPLs: `results/a3s_opt125m_*_v2_sandbox/` (cloud sandbox, 2 threads, same calibration
tokens; fake-quant differs between the machines by +0.11 % at 8,192 tokens and +0.29 % at 4,096 tokens,
float summation order). Official v2 numbers need the `--reuse-ba` reruns on Thomas's machine.
Pass: |median statistical − bit-accurate (fast, seed 0)| / bit-accurate ≤ max(2 %, bit-accurate spread).

| Config | bit-accurate | spread | statistical v1 (median) | v1 diff | statistical v2 (median) | v2 diff | verdict |
|---|---|---|---|---|---|---|---|
| f5 (cal. ADC 6 b, no noise) | 30.808 / 30.806 (ref) | 0.01 % | 54.585 | +77.2 % | 30.799 | −0.03 % | pass (v2) |
| adc8 (full-range 8 b) | 30.483 / 30.423 (ref) | 0.20 % | 30.636 | +0.50 % | = v1 | +0.50 % | pass |
| adc7 (full-range 7 b) | 30.607 / 30.530 (ref) | 0.25 % | 31.315 | +2.31 % | = v1 | +2.31 % | **fail** |
| rn008 (noise 0.08 %) | 25.716 / 25.731 (seed 1) | 0.06 % | 45.710 | +77.7 % | 25.725 | +0.04 % | pass (v2) |
| rn010 (noise 0.10 %) | 137.197 / 143.281 (seed 1) | 4.43 % | 57.876 | −57.8 % | 30.812 | −77.5 % | **fail** |
| rn02_hyb6 (noise 0.2 %, hybrid 6) | 29.232 / 28.954 (seed 1) | 0.96 % | 52.874 | +80.9 % | 29.444 | +0.73 % | pass (v2) |

Speed-up of statistical v1 over bit-accurate (Thomas's machine): 10–15× noise-free, 94–102× with noise
(noisy bit-accurate costs 464–505 ms/token). v2 adds the saturation path (sandbox: ~1 % of bit-accurate MACs).

Causes of the two failures (findings F9):
- adc7: full-range ADC error is a pure gain only for MAC ≤ 64; above that the rounding is nonlinear. Those
  conversions are the same correlated sign-extension rows as in F8, and M3's independent-conversion sum
  overestimates the residual 4–5× on the layers that have it (L2 q_proj σ 5.8 % vs measured 1.1 % of the
  ideal RMS), adding spurious noise → PPL too high (+2.3 % at 7 b, +0.5 % at 8 b).
- rn010: the Gaussian has the right variance per layer (ratio 1.00–1.03, `results/a3s_opt125m_rn010_layer_shape_sandbox/`)
  but bit-accurate error at this noise level is rare large ADC code flips (median 3.5 % of outputs non-zero,
  up to ~2·10^4 MAC units). Whole-model PPL 137–143 vs 30.8 for the Gaussian: error shape, not size, decides PPL here.
