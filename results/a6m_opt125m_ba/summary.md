# A6 mitigations `a6m_opt125m_ba`

facebook/opt-125m, bit_accurate, first 8192 WikiText-2 test tokens. All layers analog at the point unless the variant says otherwise. `*` = statistical result not valid (F13). Fake-quant rows are digital references (no conversions).

| Run | PPL | / fake-quant | analog ADC conversions | DCIM column ops | Hadamard adds |
|---|---|---|---|---|---|
| fake_quant | 30.460 | 1.000 | - | 0.000e+00 | 0.000e+00 |
