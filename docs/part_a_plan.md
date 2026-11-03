# Part A — Detailed Plan: LLM Accuracy under SRAM ACiM Non-idealities

Companion to `CLAUDE.md` (Section 3). `CLAUDE.md` holds the checklist and status; this file holds the design, the maths, the interfaces and the experiment matrix. If they disagree, fix one of them in the same session.

---

## 1. What Part A delivers

1. A Python package `cimsim` that replaces `nn.Linear` (and GPT-2 `Conv1D`) layers and attention matmuls in HuggingFace decoder LLMs with SRAM CIM equivalents.
2. Three execution modes, all config-driven:
   - `fake_quant` — integer quantization only, no array effects (the "ideal digital" reference).
   - `bit_accurate` — full bit-plane / ADC / noise simulation, ASiM-compatible semantics.
   - `statistical` — calibrated error injection on a normal matmul (fast, for full perplexity runs).
3. An event counter in every mode (ADC conversions, array cycles, array writes, MACs).
4. A results set: sensitivity of perplexity to ADC bits, rows per ADC conversion (`nrow`), noise, encoding and outliers, and a comparison of outlier mitigations at equal ADC cost.

---

## 2. Computation model (bit_accurate mode)

Matches ASiM `module/asim_linear.py` (Apache 2.0) so GATE A2 is a like-for-like comparison. Differences are listed in 2.6.

### 2.1 Quantization

- Weights: symmetric signed, `q_w = round(clamp(w / s_w, -1, 1) * (2^(B_w-1) - 1))`.
  Granularity: `per_tensor` (ASiM) or `per_channel` (needed for LLMs).
- Activations: signed symmetric (default for LLMs) or unsigned.
  Granularity: `per_tensor` (ASiM) or `per_token`.
- Scale: absolute max (ASiM). Percentile / calibrated clipping added in A5–A6.
- Dequantization: `y = y_int * s_w * s_x / ((2^(B_w-1)-1) * (2^(B_x-1)-1))` (signed case).

### 2.2 Bit decomposition

Two's complement for both operands (ASiM's sign-bit decomposition is equivalent to two's complement):

- Weight planes `q_w = -2^(B_w-1) b_{B_w-1} + Σ_{i<B_w-1} 2^i b_i`, each `b_i ∈ {0,1}`, stored in separate columns.
- Input cycles:
  - *bit-serial*: one bit per cycle, `B_x` cycles, MSB cycle weighted `-2^(B_x-1)` when signed.
  - *bit-parallel* with encoding `e` bits: signed → sign bit alone in its own cycle, remaining `B_x-1` bits grouped `e` at a time from the LSB (the top group may be smaller); unsigned → all `B_x` bits grouped from the LSB.
  - Each cycle `c` has a digit `d_c ∈ [0, L_c]` with `L_c = 2^(group size) - 1`, a shift `2^(offset_c)` and a sign.

### 2.3 One array conversion

Rows are split into chunks of `nrow` (one chunk = one array activation). For each chunk `k`, input cycle `c`, weight plane `i`, output column `n`:

```
m      = Σ_{r ∈ chunk k} d_c[r] · b_i[r, n]          # ideal analog MAC, integer in [0, nrow·L_c]
v      = clamp(m / (nrow·L_c), 0, 1)                  # normalised to full scale
if analog cycle:
    v += N(0, σ_rand)                                 # σ_rand = rand_noise_sigma / 100
    v += N(0, 1) / sqrt(v·nrow + 1) · σ_nl            # σ_nl = non_linear_sigma / 100 (ASiM's form; = m/L_c + 1)
    v  = clamp(v, 0, 1)
if adc_bits is not None:
    v  = round(v · (2^adc_bits - 1)) / (2^adc_bits - 1)
m_hat  = v · nrow · L_c
y_int += sign_c · sign_i · 2^(offset_c + i) · m_hat
```

Notes:
- Full scale uses `nrow` even for a shorter last chunk (as ASiM).
- "Analog cycle" follows ASiM's hybrid rule: cycle (i, c) gets noise only if `i + c ≤ B_w + n_cycles - hybrid_levels - 2`. MSB cycles above that are digital (no noise, still ADC-quantized, as in ASiM).
- `adc_bits = None` disables the ADC. With zero noise this must reproduce the integer matmul exactly (GATE A1).

### 2.4 Vectorisation

For each chunk, all input cycles and weight planes are computed with one matmul:
`X_k [C·T, r] @ W_k [r, P·N] → [C, T, P, N]`. The loop runs over chunks and over token blocks (`token_block` in config) to bound memory. Partial sums of 0/1 × small digits are exact in float32; accumulation into `y_int` is float64 so large K stays exact.

### 2.5 Event counts per forward call

| Counter | Formula |
|---|---|
| `macs` | T · K · N |
| `adc_conversions` | T · n_chunks · n_cycles · B_w · N |
| `adc_conversions_analog` | same, analog cycles only |
| `array_cycles` | T · n_chunks · n_cycles (one per array activation, all columns in parallel) |
| `weight_bits_written` | K · N · B_w, counted once when weights are mapped (static) or per call (dynamic operands, A4) |

Counters are also produced in `fake_quant` and `statistical` modes (computed analytically, same formulas), so later energy numbers do not depend on mode.

### 2.6 Known differences from ASiM (deliberate)

- Per-channel weights and per-token activations available (ASiM: per-tensor only). ASiM defaults are used for GATE A2.
- Dedicated RNG (`torch.Generator`, seeded from config) for reproducible noise.
- Batched computation instead of Python loops over every bit pair.
- Event counters.

---

## 3. Statistical mode (A3)

Goal: full-test-set and result-model (1.1B) perplexities without bit-accurate cost. Bit-accurate OPT-125M runs at ~74 ms/token on the i7-12700H; fake-quant at ~1.3 ms/token. A statistical layer costs one exact integer matmul plus sampling, so a ~30–50× speed-up is the target.

### 3.1 What is modelled

The macro error of one layer in integer MAC units, before any scaling:

    e[t, n] = y_bit_accurate[t, n] − (q_x · q_wᵀ)[t, n]

Statistical mode computes `y = q_x · q_wᵀ + ε`, with ε drawn from a fitted model of e, then applies the usual scales and zero-point correction. Working in integer units keeps the model a property of the macro and the activation statistics, not of the float scales.

Why e is not simply Gaussian (from F3/F5):
- With a calibrated ADC (step = 1 MAC unit) most conversions are exact; the remaining error is saturation of large MACs, which is one-sided (negative bias) and concentrated in tokens with dense high bit planes.
- With ASiM's full-range ADC, e is the sum of many rounding errors on planes weighted up to 2^14 — close to Gaussian, but its size depends on the token (sign-extension planes, F3).
- Analog noise is added before the ADC, so at small σ it is partly removed by rounding and at large σ it adds to the rounding error.

### 3.2 Step 1 — measurement (`cimsim/stat/measure.py`, `scripts/measure_error.py`)

An `ErrorProbe` attached to every bit-accurate layer computes e on real activations during a normal bit-accurate forward pass (errors propagate as in a real run). Tokens: WikiText-2 *train*, after the ADC calibration tokens (disjoint from calibration and from the test set). Per layer it records: mean, std, skew, excess kurtosis, fraction of exact outputs, max |e|, relative error, per-output-channel mean and std, per-token RMS error and its correlation with token signal size and input statistics, and std of e vs |ideal output|. Samples are saved for fitting (`samples.pt`, not in git).

Measurement configs (OPT-125M, 2,048 tokens each): `f5_adc6_calibrated`, `a3_adc6cal_rn025`, `a3_adc6cal_rn05`, `a3_adc6cal_rn1` (calibrated ADC 6 b, noise 0 / 0.25 / 0.5 / 1 % of full scale), `example_bit_accurate` (full-range ADC 8 b), `a3_adc7` (full-range ADC 7 b). F3's collapsed config is excluded: a statistical model of a model that outputs noise is not useful.

### 3.3 Step 2 — error model (decided from the measurements, findings F7)

Candidates considered: M0 (one Gaussian per layer), M1 (Gaussian per output channel, fitted), M2 (M1 × per-token factor), M3 (analytic from the MAC histogram). The measurements (`results/a3m_summary.md`) show the error is a deterministic gain (full-range ADC) plus Gaussian noise that is the same for every token and channel, with a variance that follows from the ADC and noise parameters. M3 is therefore used directly; M1/M2 are not needed.

**M3 as implemented** (`cimsim/stat/analytic.py`): for each (cycle c, weight plane p), with the calibration histogram h_cp(m) of ideal MAC values,
- E[out | m] and Var[out | m] of one conversion are computed exactly: Gaussian noise s(m) (random + nonlinear term, analog conversions only), clamp to [0, min(FS_nom, FS)], rounding to the ADC grid (Gaussian CDF over the bins);
- d(m) = E[out | m] − m is fitted as g_cp·m + b_cp (weighted least squares over h_cp);
- layer model: e = g·y_ideal + N(μ, σ²) with g = Σ w_cp g_cp / Σ w_cp (w_cp = coef_cp²·E[m²]), μ = n_chunks Σ coef_cp (b_cp + (g_cp − g) E[m]), σ² = n_chunks Σ coef_cp² (E_h[Var(out|m)] + fit residual + (g_cp − g)² Var_h(m)).

`mode: statistical` computes y_int = (1 + g)·(q_x·q_wᵀ) + μ + σ·ξ (ξ ~ N(0,1), per-layer seeded generator). The histogram pass is the existing ADC calibration pass (`calibrate_adc_range` now also prepares statistical layers), so the ADC range and the error model come from the same data. Parameters are saved per run (`stat_params.json`).

Validated at layer level against bit-accurate (`tests/test_stat_analytic.py`). Known limitation: large nonlinear rounding error (full-range ADC 6 b, F3) is overestimated (2.3×) because conversions are treated as independent.

**v2: exact sparse saturation (F8).** GATE A3 v1 failed on `f5_adc6_calibrated` (PPL ~55 vs 30.81): ADC saturation is correlated across input planes and M3 overestimated it 2–26×. With `stat_saturation: exact` (default) the saturation term S(m) = readout(m) − Q_u(min(m, FS_nom)) is computed exactly per conversion (`cimsim/stat/saturation.py`), and M3 models only the rest (clamped noise moments minus S(m)):

    y_int = (1 + g)·(q_x·q_wᵀ) + μ + σ·ξ + Σ_{chunks, c, p} coef_cp · S_cp(m)

The MAC m is computed only for (token, chunk, cycle) rows and (plane, column) pairs whose bounds (input digit sum; L_c × weight-plane popcount) reach the saturation threshold and whose calibration column maximum came within `stat_sat_margin` (default 8) MAC units of it. `stat_saturation: analytic` keeps v1 for comparison.

**v3 (F9).** (1) The exact term is generalised from saturation to all nonlinear readout error: T(m) = readout(m) − (1 + g0)·m above the exactly linear region (full-range ADC: m ≥ 65), with a fallback to saturation only where the readout is not near-linear. (2) `stat_noise: auto` (default): analog noise on (cycle, plane) pairs with rare flips (per-conversion probability ≤ 0.05, expected flips per output < `stat_flip_lambda_max` = 4) is sampled as discrete ADC code changes by exact thinning (candidates at the largest rate, exact MAC per candidate, accept with P(flip | m)/P_max, offset from P(j | m)); those pairs are removed from the Gaussian. `stat_noise: gaussian` keeps v2. Exact for noise-free conversions when the ADC step is 1/n of a MAC unit (all calibrated gate configs). Work per run is reported as `sat_work_fraction` (exact MACs / bit-accurate MACs) in `gate.json`.

### 3.4 Event counts

`_CIMBase._count` uses `analytic_counts` in every mode, so statistical mode produces the same counts as bit-accurate (asserted in `tests/test_stat_analytic.py`).

### 3.5 GATE A3 protocol (CPU budget) — `scripts/a3_gate.py`

One command per config. One calibration pass (2,048 train tokens) sets ADC ranges and the statistical model for both modes; then on the first test tokens: fake-quant, statistical (seeds 0, 1, 2) and bit-accurate runs.

| Config | What it tests | Tokens | Bit-accurate runs (spread) |
|---|---|---|---|
| `f5_adc6_calibrated` | saturation only (sparse, channel-concentrated) | 8,192 | fast + reference |
| `example_bit_accurate` | gain error 0.39 % | 8,192 | fast + reference |
| `a3_adc7` | gain error 0.79 % | 8,192 | fast + reference |
| `a3g_adc6cal_all_rn008` | noise, predicted layer error ~0.3 % | 4,096 | fast seeds 0, 1 |
| `a3g_adc6cal_all_rn010` | noise, predicted layer error ~3 % | 4,096 | fast seeds 0, 1 |
| `a3g_adc6cal_all_rn02_hyb6` | noise with hybrid levels (analog mask) | 4,096 | fast seeds 0, 1 |

The original noise levels (0.25 / 0.5 / 1 %) were dropped: they make every layer's error larger than its signal (F7), so perplexity is meaningless in both modes. Noise configs use 4,096 tokens because noisy bit-accurate runs cost ~0.42 s/token (noise sampling). Pass: |median statistical PPL − bit-accurate PPL| / bit-accurate PPL ≤ max(2 %, spread of the bit-accurate runs). Speed-up recorded per config. Summary: `python scripts/a3_gate.py --table` → `results/a3_gate_summary.md`. After a change to statistical mode only, `--reuse-ba` reruns fake-quant and statistical and keeps the bit-accurate PPLs (checks model, config, tokens, calibration tokens and the ADC full-scale hash); earlier statistical results stay in `statistical_previous`.

## 4. Package layout and interfaces

```
cimsim/
  config.py          CIMConfig (dataclass), load/merge YAML, per-layer override by glob
  quant.py           quantize_weight, quantize_act (per-tensor / per-channel / per-token)
  core/
    bitplanes.py     weight planes, input cycle digits (bit-serial / bit-parallel)
    executor.py      cim_matmul_int(x_int, w_int, cfg, gen, counter) -> y_int  (bit_accurate)
    layers.py        CIMLinear (wraps a trained Linear), CIMMatMul (dynamic operands, A4)
  events.py          EventCounter (per-layer and total)
  noise/             statistical error models (A3), circuit-derived (A7)
  models/
    swap.py          replace Linear / Conv1D in HF models according to config
eval/
  perplexity.py      WikiText-2 sliding-window perplexity
  zeroshot.py        lm-eval-harness wrapper (A4)
validation/
  asim_repro/        A2 scripts and comparison
config/
  macro/default.yaml
  experiments/*.yaml
tests/
```

### `CIMConfig` fields

| Field | Default | Meaning |
|---|---|---|
| `mode` | `fake_quant` | `fp`, `fake_quant`, `bit_accurate`, `statistical` |
| `w_bits` / `x_bits` | 8 / 8 | weight / activation precision |
| `w_gran` / `x_gran` | `per_channel` / `per_token` | quantization granularity |
| `signed_act` | true | signed activations |
| `act_enc` | null | bit-parallel encoding bits; null = bit-serial |
| `nrow` | 256 | rows per ADC conversion |
| `adc_bits` | 8 | ADC precision; null = no ADC |
| `rand_noise_sigma` | 0.0 | random noise, % of full scale |
| `non_linear_sigma` | 0.0 | ASiM-style nonlinearity, % |
| `hybrid_levels` | 0 | MSB cycle levels kept digital |
| `token_block` | 128 | tokens per inner block (memory control) |
| `seed` | 0 | noise RNG seed |
| `backend` | `fast` | `fast` (CPU-optimised) or `reference` (ASiM-mirroring loop) |
| `w_quant` | `symmetric` | `symmetric` (two's-complement planes, ASiM) or `asymmetric` (unsigned planes + digital zero-point correction) |
| `adc_range` | `full` | `full` (FS = nrow·L_c, ASiM) or `calibrated` (set by `cimsim.calibrate_adc_range`) |
| `adc_range_gran` | `cycle_plane` | calibrated FS shared per `layer`, per `cycle`, or per (cycle, plane) |
| `adc_range_method` | `mse` | `mse` (min squared readout error, ties → largest FS) or `percentile` |
| `adc_range_pct` | 99.99 | percentile for method `percentile` |
| `stat_saturation` | `exact` | mode `statistical`: `exact` = ADC saturation computed exactly on sparse candidate conversions (F8); `analytic` = folded into M3 (v1) |
| `stat_noise` | `auto` | mode `statistical`: `auto` = rare noise flips sampled as discrete ADC code changes (exact thinning, F9), Gaussian elsewhere; `gaussian` = v2 |
| `stat_flip_lambda_max` | 4.0 | `auto`: discrete only where expected flips per output are below this (and per-conversion flip probability ≤ 0.05) |
| `stat_sat_margin` | 8 | `exact`: check only columns whose calibration MAC maximum came within this many MAC units of saturation; `null` = popcount bound only (strict, slow) |

Per-layer overrides in YAML:

```yaml
default: {mode: bit_accurate, w_bits: 8, x_bits: 8, adc_bits: 6, nrow: 128}
layers:
  "*.mlp.down_proj": {adc_bits: 8}
  "*.self_attn.*":   {mode: fake_quant}
  "lm_head":         {mode: fp}
```

---

## 5. Tests (pytest, `tests/`)

| Test | Checks |
|---|---|
| `test_bitplanes` | Decompose → recompose returns the original integers (signed/unsigned, serial/parallel, all encodings) |
| `test_ideal_exact` | `adc_bits=None`, zero noise → equals `x_int @ w_int.T` exactly, random shapes incl. K not a multiple of nrow |
| `test_asim_equivalence` | Same inputs/config as ASiM's `ASiMLinear(mode='Simulation')` with zero noise → outputs match to float tolerance (runs only if `third_party/ASiM` exists) |
| `test_adc_monotonic` | Error vs ideal decreases as `adc_bits` increases |
| `test_noise_seeded` | Same seed → identical output; different seed → different |
| `test_events` | Counter values match formulas in 2.5 |
| `test_swap` | Swapping a tiny GPT-2 / OPT config leaves FP-mode outputs unchanged |

---

## 6. Experiment matrix

Development models: OPT-125M, GPT-2 (124M). Result models: Qwen2.5-0.5B, TinyLlama-1.1B (Llama-3.2-1B if access allows).
Metric: WikiText-2 test perplexity (seq len = model context, capped at 2048; stride = seq len; pipeline check: OPT-125M FP32 = 27.66 vs published 27.65); zero-shot on ARC-e, PIQA, HellaSwag, Winogrande from A4.

| ID | Milestone | Sweep | Fixed | Output |
|---|---|---|---|---|
| E0 | A0 | FP32, W8A8 (per-channel and per-tensor), W6A6, W4A8 (fake_quant) | per-channel W, per-token A | baseline table |
| E1 | A2 | ADC 3–8 b, nrow 16–256, σ_rand 0–4%, bit-serial vs act_enc 2/4 | ResNet-20 CIFAR-10, W4A4 / W8A8; CPU: fixed 1,000-image test subset, full 10k for a few anchor points | ASiM vs ours table |
| E2 | A3 | 6+ configs | OPT-125M, first 8,192 tokens | statistical vs bit-accurate perplexity, speed-up |
| E3 | A5 | ADC 4–10 b × nrow 32–512 | W8A8, all layers analog | perplexity heat map per model |
| E4 | A5 | σ_rand on one layer at a time | ADC 8 b | per-layer sensitivity bar chart |
| E5 | A5 | outlier stats per layer | — | outlier magnitude vs sensitivity |
| E6 | A5 | bit-serial vs act_enc 2/3/4 | ADC 6–8 b | perplexity vs ADC conversions |
| E7 | A6 | hybrid outlier channels (0.5–5% digital), SmoothQuant α, Hadamard rotation, noise-aware LoRA | equal ADC conversions | Pareto: perplexity vs conversions |

Every run: config + git hash + seed + raw numbers under `results/<ID>_<short_name>/`.

---

## 7. Timeline (estimate, ~10–12 weeks part-time)

| Week | Milestone | Done when |
|---|---|---|
| 1 | A0 | Repo runs, E0 baselines recorded |
| 2–3 | A1 | GATE A1 passes, one OPT-125M layer timed |
| 4 | A2 | GATE A2 table written |
| 5 | A3 | GATE A3 passes |
| 6–7 | A4 | Result models run in statistical mode, zero-shot wired |
| 8–9 | A5 | E3–E6 done |
| 10–11 | A6 | E7 Pareto plot |
| 12 | Write-up | `docs/part_a_results.md` |
| (parallel) | A7 | optional |

## 8. Compute notes (CPU-only project)

No GPU is available; everything runs on CPU. Measured speeds and the protocol are in `docs/cpu_guide.md`. Summary:

- bit_accurate cost per layer ≈ `n_cycles · B_w` matmuls of size T×K×N plus the ADC readout over the same number of elements. On 2 cores, OPT-125M runs at ~94 ms/token (bit-serial, noise-free) and ~150 ms/token with noise, vs 1.6 ms/token for fake_quant.
- Whole-model bit_accurate runs: OPT-125M and GPT-2 only, on the first 8,192 tokens (`--max-tokens 8192`), compared with fake_quant on the same subset.
- Full-test-set numbers and all sweeps on result models (E3–E7) come from `statistical` mode, so A3 is on the critical path.
- Two executors: `backend: fast` (default, CPU-optimised) and `backend: reference` (mirrors ASiM). Agreement tested in `tests/test_fast_executor.py`.
