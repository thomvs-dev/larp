# Findings log

Observations from runs that affect the design or the paper. Each entry: what was seen, evidence (results path), cause, consequence.

## F1 — Per-tensor scaling breaks LLMs (outliers)

- Seen: W8A8 fake-quant, per-tensor weights + activations (ASiM's default): GPT-2 +27.5 PPL, OPT-125M +2.8. With per-channel weights / per-token activations: +0.20 / +0.29.
- Evidence: `results/E0_summary.md`, `results/e0_*_w8a8*`.
- Cause: a few large-magnitude activation channels set the scale; most values get very few levels.
- Consequence: ASiM's per-tensor setup cannot be used unchanged for LLMs. Outlier handling is central to A5/A6.

## F2 — W4A8 gap to published 4-bit RTN is the symmetric weight format (resolved)

- Seen: symmetric W4A8 (levels −7…+7): OPT-125M 51.09, GPT-2 47.46.
- Asymmetric per-channel weights (16 levels + zero point, `w_quant: asymmetric`): **OPT-125M 37.39, GPT-2 38.90** (full test set). Published 4-bit RTN for OPT-125M (weights only, GPTQ paper) is 37.28; the remaining +0.1 is the 8-bit activations.
- Evidence: `results/e0_opt125m_w4a8_asym/`, `results/e0_gpt2_w4a8_asym/`, `results/e0_*_w4a8/`.
- Consequence: at 4 bits, the symmetric two's-complement weight format (ASiM's) costs ~14 PPL on OPT-125M and ~8.6 on GPT-2. Any 4-bit CIM result must state the weight format; asymmetric weights need unsigned planes plus a digital zero-point term, which is cheap (one Σx per row). At 8 bits the difference is negligible.

## F3 — 6-bit ADC with signed two's-complement inputs destroys the model

- Seen: OPT-125M, bit-accurate, W8A8, bit-serial, nrow 128, ADC 6 b (fc2 ADC 8 b), no noise, first 8,192 tokens: **PPL 12,962** vs 30.45 for fake-quant W8A8 on the same tokens.
- Evidence: `results/sub8k_opt125m_ba_example/`, `results/sub8k_opt125m_w8a8/`.
- Not a bug: the layer matches ASiM to 1e-6 at ADC 5–8 b (`tests/test_asim_equivalence.py`). This is ASiM's computation model behaving as defined.
- Mechanism (layer test on LLM-like activations with outlier channels, per-token scaling, 768×256, nrow 128):
  - With outlier channels setting the per-token scale, 94% of activation codes have |q| ≤ 7 and 41% are negative.
  - In two's complement, a small negative value has all upper bits set (−3 = 11111101). Activation bit planes 3–7 are therefore almost pure sign extension: bit density ≈ 0.41 on every one of them.
  - Each such plane produces a large column sum (≈ 0.41 × nrow) that the digital shift-add is supposed to cancel against the other high planes. Each sum is ADC-quantized separately before cancelling, with weights up to 2^14.
  - With a 6-bit ADC over a 0–128 range (step ≈ 2), the rounding errors on these planes do not cancel and swamp the real result.
  - Relative output error: ADC 6 b → 2.08 (worse than no signal); ADC 8 b → 0.0039; ADC 10 b → 0.0010.
- Secondary effect: with ASiM's convention (full scale = nrow, 2^b − 1 uniform steps), integer MAC values only map exactly onto ADC codes when 2^b − 1 is a multiple of nrow. Error is not monotonic in ADC bits for some nrow (nrow 255: 8 b exact, 10 b 7.7% error, 12 b 3.0%).
- Consequence:
  1. For LLMs with signed activations, the useful ADC operating range starts around 8 b at nrow 128 under two's-complement bit-serial input. The 6 b example config was a bad default; it is now 8 b.
  2. **Research angle:** input encoding matters much more for LLMs than for post-ReLU CNNs (unsigned inputs, no sign extension). Candidates to evaluate in A5/A6 at equal ADC cost: sign-magnitude inputs (sign applied digitally, no sign-extension planes), offset-binary inputs with digital correction, skipping pure-sign-extension planes, hybrid digital MSB cycles.
  3. A5 sweeps should include nrow values where 2^b − 1 aligns with nrow, and report the alignment effect separately from pure ADC precision.

## F4 — ASiM's conv layer skips channel groups for some nrow values

- Seen: `ASiMConv2d` computes the number of array activations as `w_update = ceil(C_in * k * k / nrow)` but splits the input into `ceil(C_in / floor(nrow / (k * k)))` channel groups. When the second number is larger, the last groups are never computed. Example: C_in = 64, 3×3, nrow = 64 → 10 groups, 9 computed; one channel's contribution is missing from every output.
- Evidence: `tests/test_calib_asym_conv.py::test_asim_conv_drops_chunks_when_channel_groups_exceed_w_update` (ASiM output differs from the exact convolution by > 1% with an effectively ideal ADC; ours matches).
- Affected ResNet-20 settings: nrow = 64 (64-channel layers) and nrow = 32 (16-channel layers). Unaffected: nrow = 128, 255, 256.
- Consequence: `CIMConv2d` uses all groups (no bug replication). A2 includes one nrow = 64 config to show the effect; other A2 configs use unaffected nrow values.

## F5 — Calibrating the ADC range to a 1-MAC-unit step removes ADC error for LLM layers

- ASiM maps the ADC's 2^b − 1 levels onto the full MAC range [0, nrow]. With nrow = 128 and a 6-bit ADC, one ADC step is 128/63 ≈ 2.03 MAC units, so most integer MAC values are rounded.
- In practice the MAC value of one (input bit, weight bit) column stays far below nrow: a binary input plane times a binary weight plane has density ≈ 0.2, so the sum over 128 rows is ≈ 26 ± 5 on the synthetic LLM-like layer.
- Calibrating the full scale by minimising squared readout error on calibration data (`adc_range: calibrated`, `adc_range_method: mse`) picks FS = 2^b − 1 (ADC step exactly 1 MAC unit) whenever the observed MACs fit: every integer MAC is then read exactly.
- Synthetic layer (768→256, per-token signed activations with outlier channels, nrow 128, bit-serial W8A8), held-out data, relative output error vs fake-quant:

  | ADC bits | full range (ASiM) | MSE-calibrated, per layer | MSE-calibrated, per (cycle, plane) |
  |---|---|---|---|
  | 4 | 6.95 | 3.29 | 3.29 |
  | 5 | 4.69 | 1.51 | 1.51 |
  | 6 | 2.65 | 0.0000 | 0.0000 |
  | 7 | 0.0079 | 0.0000 | 0.0000 |
  | 8 | 0.0039 | 0.0000 | 0.0000 |

- Percentile calibration (99.9 / 99.99 %) is worse than full range at 7–8 b: the few clipped conversions land on high-significance planes (shift-add weights up to 2^14). MSE is the default for that reason; percentile is kept as an option.
- A single full scale per layer works as well as per (cycle, plane) here, which is the simplest hardware (one ADC reference per macro).
- **Real model (OPT-125M, first 8,192 test tokens, bit-serial W8A8, nrow 128, ADC 6 b, fc2 ADC 8 b, no noise):**

  | ADC range | PPL |
  |---|---|
  | full (ASiM), F3 run | 12,962 |
  | MSE-calibrated, one full scale per layer | **30.81** |
  | MSE-calibrated, per (cycle, plane) | 30.81 (bit-identical) |
  | fake-quant W8A8 reference (same tokens) | 30.45 |

  Calibration chose FS = 63 (ADC step = 1 MAC unit) for every 6-bit layer, and FS = 85 (step 1/3) for fc2 at 8 b; the per-(cycle, plane) run differs only in planes whose MACs were always 0. Remaining +0.36 PPL: MAC values above 63 on test data saturate. Calibration (2,048 train tokens, 72 layers) took 327 s on the i7-12700H.
- Evidence: `results/f5_opt125m_adc6_cal_layer/`, `results/f5_opt125m_adc6_cal_cp/` (incl. `adc_full_scale.json`), `results/sub8k_opt125m_ba_example/`.
- Consequence: F3's collapse was ASiM's full-range ADC mapping, not a 6-bit limit. "Set the ADC LSB to one MAC unit (full scale 2^b − 1)" turns a 6-bit ADC from unusable to within ~1% of ideal on OPT-125M. This is a headline design rule for SRAM ACiM on LLMs. Open points for A5: ADC 4–5 b (where clipping becomes unavoidable), larger nrow, GPT-2, noise on top, and a headroom margin against saturation.

## F6 — At low ADC precision, whole-network results are chaotic

- Seen: ResNet-20, nrow 128, ADC 6 b, no noise: cimsim and ASiM disagree on 30% of top-1 predictions for the same 10 images, although every one of the 20 conv layers matches ASiM to 1.3e-6 when given identical inputs (checked with hooks inside the network). At fake-quant and ADC 10 b the two agree on all predictions.
- Cause: float rounding differences of ~1e-7 flip a few activation codes at the next layer's quantizer; with a coarse ADC each flip changes many column MACs by 1, which moves ADC codes and is multiplied by large shift-add weights. The error compounds over 18 layers.
- Measured (A2 spread run, 1,000 images, `results/a2_spread_*`): cimsim `fast` vs `reference` backends, which agree to ≤ 1e-5 per layer, differ by up to 1.4% accuracy and on 29% of predictions at nrow 128 / ADC 6 b (61.3 vs 60.2%); cimsim vs ASiM differ on 33%. Below ADC 6 b, where accuracy is near chance, 57–77% of predictions differ between any two runs. Each implementation on its own is deterministic (reruns reproduce exactly).
- Consequence: A2 compares accuracy over many images, not logits; at low ADC precision a per-config spread between simulators of a few % is expected and is not a semantic difference. Report run-to-run spread (e.g. different batch compositions) alongside such numbers. The same sensitivity applies to LLM bit-accurate runs.

## F7 — Layer error = deterministic gain + input-independent Gaussian noise; analog noise has a cliff at ~0.1 % of full scale

- Measured (A3 step 1, OPT-125M, 2,048 train tokens per config, bit-accurate, error in integer MAC units; `results/a3m_summary.md`, `results/a3m_opt125m_*/`):
  - **Full-range ADC (ASiM), 7–8 b:** the error is a pure gain, e = g · y_ideal, residual 0.0–0.1 % of the error. g = round(1/Δ)·Δ − 1 with Δ = nrow/(2^b − 1): 1/255 = 0.392 % at 8 b, 1/127 = 0.787 % at 7 b, 0.098 % for fc2 at 10 b — exactly as measured. Cause: for MAC values below ~64 every value maps to the same multiple of the ADC step, so each conversion is scaled by the same factor. It is not random error, and it partly cancels in LayerNorm.
  - **Calibrated ADC 6 b (step = 1 MAC unit), no noise:** most outputs exact; the error is saturation (MAC > 63) concentrated in a few output channels of 19 of 72 layers (q_proj, v_proj), heavy-tailed (kurtosis ~10^4).
  - **Analog noise:** Gaussian (kurtosis ≈ 0), unbiased, the same for every token (q90/q50 1.03) and channel (q10–q90 0.98–1.02), independent of the signal. Std per output = sqrt(n_chunks · Σ_{c,p} 4^{shift} · v), v = per-conversion error variance. Analytic prediction matches the measured std to ≤ 0.2 % on every 6-bit layer (180 layer-configs: 60 layers × 3 noise levels). The 8-bit fc2 layers need the MAC histogram (clipping at 0 of post-ReLU inputs, below).
- **Noise cliff:** with a step of 1 MAC unit, rounding absorbs noise below ~0.3 LSB, and the 2^14 shift-add weight of the MSB × MSB conversion amplifies every flip that gets through (that one conversion carries 56 % of the noise variance). Predicted median relative error vs σ (% of nominal full scale): 0.05 % → 2e-7, 0.08 % → 3e-3, 0.09 % → 1.2e-2, 0.10 % → 3e-2, 0.15 % → 0.3, 0.25 % → 1.1 (measured 1.4). The 0.25–1 % levels used as defaults in ASiM-style studies destroy every layer of OPT-125M (error 1.4–8× the signal).
- **A finer ADC makes noise worse:** fc2 with an 8-bit calibrated ADC (step 1/3 MAC unit) passes the noise unrounded, so at noise levels where the 6-bit layers are fine fc2 is not. Under noise, a 1-MAC-unit ADC step is a noise filter, not only a resolution choice.
- **hybrid_levels** (top significance levels computed digitally) reduce the noise variance share: 1 level → 44 %, 2 → 16 %, 4 → 1.6 %, 6 → 0.13 %.
- Model built from this (A3 step 2, `cimsim/stat/analytic.py`): e ≈ g · y_ideal + N(μ, σ²), with g, μ, σ computed per layer from the MAC histogram of the calibration pass — no fitting to bit-accurate runs. Layer-level check vs bit-accurate (`tests/test_stat_analytic.py`): noise std within 5 % (measured 0.2–2.5 %) for calibrated / full-range ADC, nonlinearity noise, hybrid levels, 2-bit input cycles, no ADC, post-ReLU inputs (clipping at 0 handled); gain exact.
- Limitation: where the deterministic rounding error is large and nonlinear (full-range ADC 6 b, the F3 collapse regime) the model treats conversions as independent and overestimates the error std by 2.3×. Those configs destroy the model either way.
- Consequences: (1) statistical mode uses the analytic model (M3) directly; it extrapolates to configs that were never run bit-accurately (needed for A5). (2) Noise studies on LLMs must use σ ≲ 0.1 % of full scale, or hybrid levels, or input encodings that avoid noisy MSB-plane conversions (F3 research angle). (3) The ADC step should be one MAC unit on every layer, fc2 included, when analog noise is present.

## F8 — Saturation errors are correlated across input planes; the independent-conversion model overestimates them 2–26× (GATE A3 v1 failure)

- Seen: GATE A3, `f5_adc6_calibrated` (OPT-125M, calibrated ADC 6 b, no noise, first 8,192 test tokens, git 33e49ce): statistical mode v1 PPL 53.97 / 58.84 / 54.58 (seeds 0–2) vs bit-accurate 30.81 and fake-quant 30.45. Fails the ±2 % gate by ~+79 %.
- Evidence: `results/a3g_opt125m_f5/gate.json`, `results/a3g_opt125m_f5/stat_params.json` vs measured layer errors in `results/a3m_opt125m_f5_adc6_calibrated/error_stats.json`. Predicted vs measured error std (MAC units): L2 q_proj 2911 vs 263 (11×), L3 q_proj 2163 vs 233, L9 fc1 6922 vs 263 (26×), L9 v_proj 1908 vs 287 (6.6×), L0 k_proj 86 vs 45.
- Cause: M3 treats every conversion as independent. With a calibrated ADC (step = 1 MAC unit) the only noise-free error is saturation (m > FS). It hits the two's-complement sign-extension input planes (3–7, F3) together, on the same rows and by nearly the same amount, and their shift-add weights (+8, +16, +32, +64, −128) mostly cancel. The independent model adds the squares instead. Direct check on a synthetic layer (per-conversion saturation errors computed explicitly, 512→64, nrow 128): true error RMS is 2.2× / 5.5× / 6.3× smaller than the independent sum at FS 48 / 40 / 36, matching M3's overestimate on the same layer (2.9–6.7×). Second problem: the real error is sparse (99.4–100 % of outputs exact, a few channels, kurtosis 10^4–10^5) while M3 spreads Gaussian noise over every output.
- Fix (statistical mode v2, `stat_saturation: exact`, default; `cimsim/stat/saturation.py`): each conversion's readout error is split into a saturation term S(m) = readout(m) − Q_u(min(m, FS_nom)) (Q_u = same ADC without the upper code limit) and a rounding / noise term. S(m) is computed exactly, only for conversions that can saturate; M3 keeps the clamped noise moments and drops S(m) from its deterministic part. Candidates: input digit sum of the chunk and L_c × weight-plane popcount of the column must both reach the saturation threshold, and (`stat_sat_margin`, default 8 MAC units) the column's largest MAC during calibration must have come within the margin of it. The popcount bound alone flags about half of all conversions when plane densities are ≈ 0.5 (synthetic 768→3072 layer: 15.7 % of bit-accurate MACs, 4.2 s vs 0.06 s fake-quant); with the calibration filter 0.08 % (0.23 s).
- Validation (`tests/test_stat_saturation.py`, both backends): noise-free, ADC step 1/n MAC unit: statistical = bit-accurate to ≤ 1e-5 (fast backend; < 1e-6 reference) for bit-serial, bit-parallel (enc 2), asymmetric weights, K not a multiple of nrow, conv channel groups; margin filter output identical to the strict bound; noise + saturation: error std within 10 %; v1 model overestimates by > 1.5× on the same layer.
- Real model (cloud sandbox, 2 threads, OPT-125M, f5 config, same calibration as the gate; `results/a3s_opt125m_f5_v2_sandbox/check.json`): statistical v2 PPL **30.799** on the first 8,192 test tokens vs bit-accurate 30.808 (Thomas's run, `results/a3g_opt125m_f5/gate.json`) → −0.03 % (v1: +79 %). Per-layer check on test tokens 0–2,047, same layer inputs: with the strict bound statistical = bit-accurate exactly on every 6-bit layer (fc2: 2e-7, fast-backend float rounding); the margin filter missed 7 of 1.7e8 outputs (bit-accurate has 1.0e7 outputs with saturation error), the largest miss being most of L8 q_proj's error (rel. 2.6e-4). Exact work 0.97 % of bit-accurate MACs; 2.3 % of (cycle, plane, column) triples checked (max 7.2 %). Speed on 2 threads: 8.35 ms/token statistical vs 1.55 fake-quant (~11× faster than bit-accurate at ~94 ms/token); the saturation loop (Python, per chunk and cycle) dominates and is the place to optimise if the 30–50× target matters.
- Noise 0.10 % (`a3g_adc6cal_all_rn010`), layer level on real activations (test tokens 0–511, sandbox; `results/a3s_opt125m_rn010_layer_shape_sandbox/layers.json`): the Gaussian has the right variance (statistical / bit-accurate std 1.004–1.030 over 72 layers) but the wrong shape. In bit-accurate mode only a median 3.5 % of outputs carry any error, each a large jump (max ~2·10^4 MAC units, MSB × MSB conversion, F7), excess kurtosis ~680; fc2 layers are hit hardest (13–16 % relative error, 8–11 % of outputs). Whether the shape matters for PPL is decided by the GATE A3 noise runs; if it does, the fix is sparse sampling of ADC code flips per conversion (Poisson count, then location and size), which also covers 0.08 %.
- Limits: (1) the margin filter is an assumption — a column that never came within `stat_sat_margin` of saturating on calibration data is not checked at test time; `stat_sat_margin: null` gives the strict (slow) bound. (2) Deterministic rounding error at ADC steps that are not 1/n of a MAC unit (full-range ADC with 2^b − 1 not a multiple of nrow) is still modelled with independent conversions (the F7 limitation, 3× on the synthetic FS 40 / 6 b case). Gate configs: full-range ADC 7–8 b is a pure gain there (F7) and is modelled exactly; calibrated layers use FS = 63 / 85 (step 1, 1/3).
- Consequence: GATE A3 configs with a calibrated ADC (f5 and the three noise configs) are rerun in statistical mode v2 with the bit-accurate PPLs reused (`scripts/a3_gate.py --reuse-ba`). General lesson for A5–A7: per-conversion error models must not assume independence across input planes for deterministic error sources (saturation, rounding); analog noise is independent and fine.

## Machine / speed notes

- Thomas's machine: Intel Core i7-12700H (6 P-cores + 8 E-cores, 20 threads), 16 GB DDR4, no GPU. torch uses 14 threads.
- Bit-accurate OPT-125M, 8,192 tokens, bit-serial: 608 s (≈ 74 ms/token, ~2.5 min per 2048-token window). fake_quant on the same tokens: 10.4 s.
- ResNet-20, 50 CIFAR-sized images, nrow 128, ADC 6 b, 2 sandbox cores: cimsim 4.1 s, ASiM 33.2 s.
