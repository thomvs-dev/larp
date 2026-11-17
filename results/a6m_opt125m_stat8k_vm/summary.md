# A6 mitigations `a6m_opt125m_stat8k_vm`

/home/claude/hf_local/opt-125m, statistical, first 8192 WikiText-2 test tokens. All layers analog at the point unless the variant says otherwise. `*` = statistical result not valid (F13). Fake-quant rows are digital references (no conversions).

| Run | PPL | / fake-quant | analog ADC conversions | DCIM column ops | Hadamard adds |
|---|---|---|---|---|---|
| fake_quant | 30.484 | 1.000 | - | 0.000e+00 | 0.000e+00 |
| fake_quant_had | 30.116 | 0.988 | - | 0.000e+00 | 4.832e+09 |
| n128_adc6_base | 30.799 | 1.010 | 3.479e+11 | 0.000e+00 | 0.000e+00 |
| n128_adc6_had_all | 30.113 | 0.988 | 3.479e+11 | 0.000e+00 | 4.832e+09 |
