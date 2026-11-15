# A6 mitigations `a6m_qwen15b_stat`

Qwen/Qwen2.5-1.5B, statistical, first 8192 WikiText-2 test tokens. All layers analog at the point unless the variant says otherwise. `*` = statistical result not valid (F13). Fake-quant rows are digital references (no conversions).

| Run | PPL | / fake-quant | analog ADC conversions | DCIM column ops | Hadamard adds |
|---|---|---|---|---|---|
| fake_quant | 10.338 | 1.000 | - | 0.000e+00 | 0.000e+00 |
| fake_quant_had | 10.133 | 0.980 | - | 0.000e+00 | 2.595e+10 |
| n128_adc6_base | 10.342 | 1.000 | 5.367e+12 | 0.000e+00 | 0.000e+00 |
| n128_adc6_had_all | 10.229 | 0.989 | 5.367e+12 | 0.000e+00 | 2.595e+10 |
