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

## Machine / speed notes

- Thomas's machine: Intel Core i7-12700H (6 P-cores + 8 E-cores, 20 threads), 16 GB DDR4, no GPU. torch uses 14 threads.
- Bit-accurate OPT-125M, 8,192 tokens, bit-serial: 608 s (≈ 74 ms/token, ~2.5 min per 2048-token window). fake_quant on the same tokens: 10.4 s.
- ResNet-20, 50 CIFAR-sized images, nrow 128, ADC 6 b, 2 sandbox cores: cimsim 4.1 s, ASiM 33.2 s.
