# Modelling assumptions

Every assumption about circuits, published macros or simulator semantics goes here, with a source.

## Part A — bit_accurate mode

| # | Assumption | Source / reason |
|---|---|---|
| 1 | Two's complement for weights and activations; weight MSB plane and activation sign cycle carry negative weight | ASiM `module/asim_linear.py` (`_decompose_weight`, `_decompose_feature`) |
| 2 | Each weight bit is a separate physical column; one ADC conversion per (column, input cycle, row chunk) | ASiM loop structure |
| 3 | ADC full scale = `nrow × L_c` even for a shorter last chunk; uniform `2^b - 1` steps over full scale | ASiM (`mac / nrow`, `round(mac * (2^adc - 1))`) |
| 4 | Random noise: Gaussian, σ in % of full scale, per conversion | ASiM `rand_noise_sigma` |
| 5 | Nonlinearity: Gaussian, σ% / sqrt(v·nrow + 1), per conversion (v = normalised MAC) | ASiM `non_linear_sigma` (form copied as-is; physical meaning to be checked in A7) |
| 6 | Hybrid: cycles with i + j > B_w + n_cycles − hybrid_levels − 2 get no noise but are still ADC-quantized | ASiM |
| 7 | Bit-parallel signed inputs: sign bit in its own cycle; magnitude bits grouped from LSB, top group may be smaller | ASiM |
| 8 | Quantization scale = absolute max (per tensor in ASiM; per channel / per token also offered) | ASiM + standard LLM practice |
| 9 | LM head kept digital (not swapped) by default | Common practice; vocab projection is large and accuracy-critical |
| 10 | Calibrated ADC range (`adc_range: calibrated`): ADC full scale set per layer / cycle / (cycle, plane), integer MAC units, values above it saturate at the top code | Option beyond ASiM; models a programmable ADC reference. Calibrated on ideal (fake-quant) activations from separate calibration data (LLM: WikiText-2 train split) |
| 11 | Analog noise stays relative to the nominal full scale nrow·L_c when the ADC range is calibrated | Noise comes from the array (bit-line charge), not from the ADC reference |
| 12 | Asymmetric weights (`w_quant: asymmetric`): unsigned weight planes, zero-point term z·Σx subtracted digitally and exactly | Option beyond ASiM; Σx per row assumed available digitally (input buffer) |
| 13 | Conv layers: unfolded rows grouped by whole input channels, floor(nrow / (kh·kw)) channels per array activation; all groups computed | ASiM `asim_conv.py` grouping, without its group-skipping issue (findings F4) |
| 14 | Conv activation scale: `per_tensor` = whole input batch (ASiM), `per_token` = per image | ASiM uses the batch max |

## Part B — cost model (B0)

Every per-action cost parameter used by `cimsim/circuits/` is listed here with its source. Values marked
"reference line" are survey envelopes, used until GATE B1 calibrates each macro.

| # | Assumption | Source / reason |
|---|---|---|
| B-1 | ADC energy per conversion (`arch: bound`) = max(Walden, Schreier) with FOM_W · 2^ENOB and ½ · 10^((SNDR − FOM_S)/10), SNDR = 6.02 ENOB + 1.76 | Walden, IEEE JSAC 1999 (FOM_W); Schreier & Temes 2005 (FOM_S); use of the two as energy bounds: Murmann, "Energy Limits in A/D Converters" (talk, 2012), also the basis of the CiMLoop ADC plug-in (Andrulis et al., arXiv 2404.06553) |
| B-2 | Reference values FOM_W = 10 fJ/conv-step, FOM_S = 170 dB (`config/circuits/adc_survey_lines.yaml`) | Reference lines drawn on the ISSCC/VLSI 1997–2012 survey plot in Murmann 2012 (above). Not best-in-class: the best SAR FOM_W reported by 2022 is ~0.4 fJ/conv-step (Tang et al., IEEE TCAS-I 69(6), 2022) |
| B-3 | Throughput per ADC not modelled; bound model valid below the survey's speed corner (~100–300 MHz) | Murmann 2012. CiMLoop's plug-in adds throughput-dependent bounds; to add when the mapper (B2) fixes ADC rates |
| B-4 | Flash ADC energy = (2^b − 1) comparator decisions + fixed term; `e_comparator` has no default and must be cited per macro | Comparator count of a flash ADC; M2 (Sinangil JSSC 2021) uses a 4-b flash |
| B-5 | ADC energy is charged on `adc_conversions` (all input cycles), not `adc_conversions_analog` | Hybrid (digital) cycles are still ADC-quantized (#6) |
| B-6 | 1 MAC = 2 ops in all TOPS/W and TOPS figures | Common convention; confirmed for M2 by 1024 MAC × 2 / 5.5 ns = 372.4 GOPS (`validation/macro_calibration.md`) |
| B-7 | DCIM energy per 1b × 1b product = e0 + e1 · input toggle rate; weight-ones fraction fixed at the calibration value (50 %) | Structure: Chih ISSCC 2021 (energy depends on toggle rate and weight sparsity, mostly first adder-tree stages). e0, e1 fitted to its 89 / 52 TOPS/W at 18 / 50 % toggle (`results/b1_m4_chih/`) |
| B-8 | DCIM pass time = (x_bits + 1) · t_cycle; a w_bits weight occupies ⌈w_bits / 4⌉ 4-b columns | Chih ISSCC 2021 text (extra accumulator cycle) and Fig. 16.4.6 (10 ns at 4b, 18 ns at 8b; 3300 / 917 GOPS) |
| B-9 | Input toggle rate = fraction of bit transitions per input line, inputs streamed bit-serially MSB first, token after token | Chih ISSCC 2021 (MSB-first bit-serial input); measured on per-token 8-b activations by `scripts/b1_toggle_rates.py` |
| B-10 | No supply-voltage or node scaling: each fitted spec is valid only at its calibration VDD and node | Not yet modelled (B0 task "technology node scaling") |
