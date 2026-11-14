# A6 mitigations `a6m_qwen15b_stat`

Qwen/Qwen2.5-1.5B, statistical, first 8192 WikiText-2 test tokens. All layers analog at the point unless the variant says otherwise. `*` = statistical result not valid (F13). Fake-quant rows are digital references (no conversions).

| Run | PPL | / fake-quant | analog ADC conversions | DCIM column ops | Hadamard adds |
|---|---|---|---|---|---|
