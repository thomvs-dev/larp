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
