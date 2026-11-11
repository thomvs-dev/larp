# A5 sweep `a5fix_opt125m_ba_sandbox`

/mnt/user-data/uploads/japan/hf_local/opt-125m, bit_accurate, first 4096 WikiText-2 test tokens. Base: other layers as `a5_base_adc8cal.yaml`. Fake-quant PPL on the same tokens: 25.381.

Cells: PPL (ratio to fake-quant). `—` = not run.

## all

| nrow | ADC 8 b | ADC 7 b |
|---|---|---|
| 64 | 25.38 (1.000) | 25.38 (1.000) |
| 128 | 25.38 (1.000) | — |
