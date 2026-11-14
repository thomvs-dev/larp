# A6 mitigations `a6m_gpt2_ba`

gpt2, bit_accurate, first 8192 WikiText-2 test tokens. All layers analog at the point unless the variant says otherwise. `*` = statistical result not valid (F13). Fake-quant rows are digital references (no conversions).

| Run | PPL | / fake-quant | analog ADC conversions | DCIM column ops | Hadamard adds |
|---|---|---|---|---|---|
| fake_quant | 33.776 | 1.000 | - | 0.000e+00 | 0.000e+00 |
| fake_quant_had | 33.728 | 0.999 | - | 0.000e+00 | 4.832e+09 |
| n128_adc6_base | 198.377 | 5.873 | 3.479e+11 | 0.000e+00 | 0.000e+00 |
| n128_adc6_had_down | 33.970 | 1.006 | 3.479e+11 | 0.000e+00 | 3.020e+09 |
| n128_adc6_had_all | 33.583 | 0.994 | 3.479e+11 | 0.000e+00 | 4.832e+09 |
| n128_adc6_dcim_down | 34.013 | 1.007 | 2.319e+11 | 1.160e+11 | 0.000e+00 |
| n128_adc6_down_bp1 | 34.013 | 1.007 | 3.479e+11 | 0.000e+00 | 0.000e+00 |
| n256_adc7_base | 68.797 | 2.037 | 1.739e+11 | 0.000e+00 | 0.000e+00 |
| n256_adc7_had_down | 33.741 | 0.999 | 1.739e+11 | 0.000e+00 | 3.020e+09 |
| n256_adc7_had_all | 33.722 | 0.998 | 1.739e+11 | 0.000e+00 | 4.832e+09 |
| n256_adc7_dcim_down | 33.877 | 1.003 | 1.160e+11 | 5.798e+10 | 0.000e+00 |
| n256_adc7_down_bp1 | 33.877 | 1.003 | 1.739e+11 | 0.000e+00 | 0.000e+00 |
