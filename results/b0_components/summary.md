# B0 component costs (git 75af28a)

Digital operations: published at 45 nm / 0.9 V, scaled to 22 nm / 0.8 V with E ~ node^1 · VDD² (factor 0.386). Sources: `docs/assumptions.md` B-11 … B-15.

| Operation | 45 nm (fJ) | 22 nm (fJ) |
|---|---|---|
| add_8b | 25.00 | 9.66 |
| add_16b | 50.00 | 19.31 |
| add_24b | 75.00 | 28.97 |
| mul_8x8b | 193.73 | 74.83 |
| mul_16x16b | 774.91 | 299.34 |
| int8_mac_acc24 | 268.73 | 103.80 |
| register_bit | 7.50 | 2.90 |
| sram_buffer_bit | 156.25 | 60.36 |
| gate_mul_per_element_16b | 774.91 | 299.34 |
| gelu_per_element_16b | 4099.82 | 1583.69 |
| layernorm_per_element_16b | 2524.74 | 975.26 |
| relu_per_element_16b | 0.00 | 0.00 |
| residual_per_element_16b | 50.00 | 19.31 |
| rmsnorm_per_element_16b | 2374.74 | 917.32 |
| rope_per_element_16b | 1599.82 | 617.98 |
| silu_per_element_16b | 4099.82 | 1583.69 |
| softmax_per_element_16b | 4149.82 | 1603.00 |

ADC energy per conversion on the survey reference lines (not scaled, not calibrated; B-1, B-2), fJ:

| Line | 4 b | 5 b | 6 b | 7 b | 8 b |
|---|---|---|---|---|---|
| bound_survey_lines | 160.0 | 320.0 | 640.0 | 1280.0 | 2560.0 |
| bound_best_sar | 6.4 | 12.8 | 25.6 | 51.2 | 102.4 |

Not in this table: analog array energies per cell (wordline, bit line), fitted per macro in B1.
These are component inputs, not results: no energy per token is reported before GATE B1.
