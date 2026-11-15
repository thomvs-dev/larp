# B4 sweep (git 152fda8)
Energy per token at sequence length 512, KV cache in an SRAM buffer, 22 nm / 0.8 V. PPL: first 8,192 WikiText-2 test tokens (Part A runs). Compute = array + ADC + shift-add of the linear layers (or the digital / DCIM equivalent). Decode and prefill totals at the minimum on-chip budget (the largest layer only) with the default DRAM.
| Model | Point | ADC | PPL | Compute (mJ) | of which ADC | Prefill total (mJ) | Decode total (mJ) | Decode, GDDR5 (mJ) | Decode, 2 Gbit on chip (mJ) |
|---|---|---|---|---|---|---|---|---|---|
| qwen2.5-1.5b | W8A8 digital INT8 | - | 10.338 | 0.769 | 0 % | 0.870 | 50.46 | 174.3 | 42.65 |
| qwen2.5-1.5b | W8A8 DCIM (M4) | - | 10.338 | 0.187 | 0 % | 0.290 | 50.19 | 174.0 | 42.32 |
| qwen2.5-1.5b | analog nrow 128 / 6 b | m1 | 10.342 | 0.372 | 63 % | 0.474 | 50.37 | 174.2 | 42.51 |
| qwen2.5-1.5b | analog nrow 128 / 6 b | best_sar | 10.342 | 0.154 | 11 % | 0.257 | 50.16 | 174.0 | 42.29 |
| qwen2.5-1.5b | analog nrow 64 / 5 b | m1 | 10.356 | 0.545 | 70 % | 0.647 | 50.55 | 174.4 | 42.68 |
| qwen2.5-1.5b | analog nrow 64 / 5 b | best_sar | 10.356 | 0.181 | 9 % | 0.283 | 50.18 | 174.0 | 42.32 |
| qwen2.5-1.5b | analog 128 / 6 b + Hadamard | m1 | 10.229 | 0.372 | 63 % | 0.474 | 50.37 | 174.2 | 42.51 |
| qwen2.5-1.5b | analog 128 / 6 b + Hadamard | best_sar | 10.229 | 0.154 | 11 % | 0.257 | 50.16 | 174.0 | 42.29 |
| qwen2.5-0.5b | W8A8 digital INT8 | - | 14.821 | 0.210 | 0 % | 0.243 | 16.09 | 55.7 | 7.98 |
| qwen2.5-0.5b | W8A8 DCIM (M4) | - | 14.821 | 0.051 | 0 % | 0.085 | 16.02 | 55.7 | 7.84 |
| qwen2.5-0.5b | analog nrow 128 / 6 b | m1 | 14.809 | 0.102 | 63 % | 0.135 | 16.07 | 55.7 | 7.89 |
| qwen2.5-0.5b | analog nrow 128 / 6 b | best_sar | 14.809 | 0.042 | 11 % | 0.076 | 16.01 | 55.6 | 7.83 |
Notes: all points are within 0.2 % of W8A8 perplexity (the Hadamard point is 1 % better), so the comparison is on energy. ADC `m1` = the SAR ADC of the calibrated M1 macro moved to 22 nm; `best_sar` = best reported SAR Walden figure of merit (survey line, not a calibrated macro). DCIM uses the M4 calibration at an input toggle rate measured on OPT-125M, not on Qwen. Digital INT8 reads each weight from an on-chip SRAM buffer.
