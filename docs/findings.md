# Findings log

Observations from runs that affect the design or the paper. Each entry: what was seen, evidence (results path), cause, consequence.

## F1 — Per-tensor scaling breaks LLMs (outliers)

- Seen: W8A8 fake-quant, per-tensor weights + activations (ASiM's default): GPT-2 +27.5 PPL, OPT-125M +2.8. With per-channel weights / per-token activations: +0.20 / +0.29.
- Evidence: `results/E0_summary.md`, `results/e0_*_w8a8*`.
- Cause: a few large-magnitude activation channels set the scale; most values get very few levels.
- Consequence: ASiM's per-tensor setup cannot be used unchanged for LLMs. Outlier handling is central to A5/A6.

## F2 — W4A8 worse than published 4-bit RTN (to confirm)

- Seen: OPT-125M W4A8 51.09 (+23.4) vs roughly +10 for published 4-bit RTN.
- Likely cause: symmetric weights with levels −7…+7 (15 levels, matches two's-complement planes) vs asymmetric 16-level min-max in published RTN.
- Status: open — needs an asymmetric weight option (A1 sub-task, low priority).

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
- Consequence: F3's collapse at 6 b is mostly an artefact of ASiM's full-range ADC mapping. Whether 6 b holds for real OPT-125M layers is the next experiment (`config/experiments/f5_adc6_calibrated*.yaml`). If it does, "ADC range = integer MAC step" is a headline design rule for SRAM ACiM on LLMs.

## F6 — At low ADC precision, whole-network results are chaotic

- Seen: ResNet-20, nrow 128, ADC 6 b, no noise: cimsim and ASiM disagree on 30% of top-1 predictions for the same 10 images, although every one of the 20 conv layers matches ASiM to 1.3e-6 when given identical inputs (checked with hooks inside the network). At fake-quant and ADC 10 b the two agree on all predictions.
- Cause: float rounding differences of ~1e-7 flip a few activation codes at the next layer's quantizer; with a coarse ADC each flip changes many column MACs by 1, which moves ADC codes and is multiplied by large shift-add weights. The error compounds over 18 layers.
- Consequence: A2 compares accuracy over many images, not logits; at low ADC precision a per-config spread between simulators of a few % is expected and is not a semantic difference. Report run-to-run spread (e.g. different batch compositions) alongside such numbers. The same sensitivity applies to LLM bit-accurate runs.

## Machine / speed notes

- Thomas's machine: Intel Core i7-12700H (6 P-cores + 8 E-cores, 20 threads), 16 GB DDR4, no GPU. torch uses 14 threads.
- Bit-accurate OPT-125M, 8,192 tokens, bit-serial: 608 s (≈ 74 ms/token, ~2.5 min per 2048-token window). fake_quant on the same tokens: 10.4 s.
- ResNet-20, 50 CIFAR-sized images, nrow 128, ADC 6 b, 2 sandbox cores: cimsim 4.1 s, ASiM 33.2 s.
