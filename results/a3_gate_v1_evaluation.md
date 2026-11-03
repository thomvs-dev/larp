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

## Statistical v3 (sandbox, 2 threads; bit-accurate from Thomas's runs above)

v3 = exact sparse deterministic readout term (saturation and nonlinear rounding above the exactly linear
region) + discrete ADC code flips for rare noise flips (`stat_noise: auto`). rn010 uses the final flip
sampler (exact thinning, git 0e6512f); rn008 and rn02_hyb6 were run with the earlier averaged-placement
sampler (git 1916262); f5 / adc8 / adc7 are noise-free and identical in both.

| Config | bit-accurate | statistical v3 | median | diff | tolerance | verdict |
|---|---|---|---|---|---|---|
| f5 | 30.808 | 30.799 (all seeds) | 30.799 | −0.03 % | 2 % | pass |
| adc8 | 30.483 | 30.443 (all seeds) | 30.443 | −0.13 % | 2 % | pass |
| adc7 | 30.607 | 30.566 (all seeds) | 30.566 | −0.13 % | 2 % | pass (v2: +2.31 %) |
| rn008 | 25.716 / 25.731 | 25.966 / 27.082 / 25.747 | 25.966 | +0.97 % | 2 % | pass |
| rn02_hyb6 | 29.232 / 28.954 | 29.123 / 29.678 / 29.108 | 29.123 | −0.37 % | 2 % | pass |
| rn010 | 137.197 / 143.281 | 12 seeds: 69, 86, 113, 133, 134, 171, 178, 213, 238, 301, 385, 440 | 174.2 | +27 % | 4.4 % (2 seeds) | open: needs more bit-accurate seeds |

rn010: the statistical PPL varies by a factor ~6 across seeds; both bit-accurate seeds lie inside its
interquartile range (133–301). Layer level on real activations the v3 error matches bit-accurate (std ratio
0.99–1.06, fraction of non-zero outputs within 2 %, kurtosis ~620 vs ~680, per-position flip rates equal:
`results/a3s_opt125m_rn010_layer_shape_sandbox/`, `results/a3s_opt125m_rn010_token_rates_sandbox/`).
Whether bit-accurate PPL is equally variable is not known from two seeds; Thomas's run with seeds 2–4 decides.
Cost: noise-free v3 ~11–12 ms/token on 2 threads (fake-quant ~1.9); rn010 ~17 ms/token (thinning);
rn02_hyb6 24 ms/token (many low-significance flip pairs, tunable with `stat_flip_lambda_max`).

## rn010 with 5 bit-accurate seeds (Thomas's machine, git db1c52f, `--ref median`)

Run: `results/a3g_opt125m_rn010/gate.json` (4,096 test tokens, 14 threads; bit-accurate seeds 0–1 reused
from git 33e49ce, seeds 2–4 new; the bit-accurate executor is unchanged between the two commits).

| | n | sorted PPL | median | interquartile range | geometric mean | sd of log PPL |
|---|---|---|---|---|---|---|
| bit-accurate (fast) | 5 | 76.0, 137.2, 143.3, 150.5, 175.5 | 143.3 | 137.2–150.5 | 131.6 | 0.32 |
| statistical v3 | 8 | 94.8, 113.1, 144.3, 144.8, 156.0, 183.3, 265.0, 308.7 | 150.4 | 136.5–203.7 | 164.0 | 0.40 |

- Bit-accurate PPL at this noise level varies by a factor 2.3 between seeds (76–176). The two-seed
  reference used before (137 / 143, spread 4.4 %) was an undersample; the earlier "+27 %" was not a model error.
- Median vs median: +5.0 %. Bootstrap 95 % interval of the median ratio: 0.79–2.05.
- Geometric-mean ratio: 1.25, approximate 95 % interval 0.80–1.94 (Welch t on log PPL, p = 0.30).
  Mann-Whitney p = 0.35. No detectable difference between the two distributions.
- Limit: 2 of 8 statistical seeds (265, 309) lie above the largest bit-accurate seed (176). With 5
  bit-accurate seeds a heavier statistical upper tail (up to ~2× in geometric mean) cannot be excluded.
- The gate script's verdict (pass, tolerance = spread 131 %) is correct by the rule as written but the rule is
  weak here: any statistical median between ~0 and ~330 would pass. See the criterion proposal below.
- Speed: statistical 26 ms/token vs bit-accurate 443–515 ms/token → 17.7×.

### Proposed criterion for configs at the noise cliff (awaiting Thomas)

A config is "chaotic" when the sd of log PPL over ≥ 5 bit-accurate seeds exceeds 0.10 (≈ 10 %). For those:
pass if (1) Mann-Whitney between ≥ 5 bit-accurate and ≥ 8 statistical seeds gives p > 0.05, (2) the
statistical median lies inside the bit-accurate interquartile range widened by 2 % (here 134.5–153.5: 150.4
inside), and (3) layer-level error agrees (std ratio within 10 %, non-zero fraction within 5 %; done for rn010
in `results/a3s_opt125m_rn010_layer_shape_sandbox/`). All other configs keep the 2 % rule. rn010 passes all
three. The report states that whole-model PPL at the noise cliff is a distribution, not a number, and gives
both distributions.
