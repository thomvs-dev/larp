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

## Machine / speed notes

- Thomas's machine: Intel Core i7-12700H (6 P-cores + 8 E-cores, 20 threads), 16 GB DDR4, no GPU. torch uses 14 threads.
- Bit-accurate OPT-125M, 8,192 tokens, bit-serial: 608 s (≈ 74 ms/token, ~2.5 min per 2048-token window). fake_quant on the same tokens: 10.4 s.
