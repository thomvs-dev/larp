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

1. For each configuration (layer type, B_w, B_x, nrow, ADC bits, encoding, noise), run `bit_accurate` on real activations captured from the model (calibration set: 32 sequences × 512 tokens from WikiText-2 train).
2. Record `e = y_bit_accurate - y_fake_quant` in integer units. Fit: mean, std, and dependence on `|y_ideal|` (binned). Store as a small table in `results/calib/<config_hash>.pt`.
3. In `statistical` mode: `y = y_fake_quant + ε`, with ε sampled from the fitted model (Gaussian with output-dependent std as the first model; add a heavier-tailed option if residuals are not Gaussian).
4. GATE A3: perplexity on a WikiText-2 subset within ±2% relative of `bit_accurate`, across at least 6 configurations spanning low and high noise.

---

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
