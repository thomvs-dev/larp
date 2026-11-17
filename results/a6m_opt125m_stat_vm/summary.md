# A6 mitigations `a6m_opt125m_stat_vm`

/home/claude/hf_local/opt-125m, statistical, first 4096 WikiText-2 test tokens. All layers analog at the point unless the variant says otherwise. `*` = statistical result not valid (F13). Fake-quant rows are digital references (no conversions).

| Run | PPL | / fake-quant | analog ADC conversions | DCIM column ops | Hadamard adds |
|---|---|---|---|---|---|
| fake_quant | 25.381 | 1.000 | - | 0.000e+00 | 0.000e+00 |
| fake_quant_had | 25.026 | 0.986 | - | 0.000e+00 | 2.416e+09 |
| n128_adc6_base | 25.591 | 1.008 | 1.739e+11 | 0.000e+00 | 0.000e+00 |
| n128_adc6_had_down | 25.592 | 1.008 | 1.739e+11 | 0.000e+00 | 1.510e+09 |
| n128_adc6_had_all | 25.024 | 0.986 | 1.739e+11 | 0.000e+00 | 2.416e+09 |
| n128_adc6_dcim_down | 25.591 | 1.008 | 1.160e+11 | 5.798e+10 | 0.000e+00 |
| n128_adc6_down_bp1 | 25.591 | 1.008 | 1.739e+11 | 0.000e+00 | 0.000e+00 |
