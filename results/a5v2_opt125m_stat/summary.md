# A5 sweep `a5v2_opt125m_stat`

facebook/opt-125m, statistical, first 8192 WikiText-2 test tokens. Base: other layers as `a5_base_adc8cal.yaml`. Fake-quant PPL on the same tokens: 30.460.

Cells: PPL (ratio to fake-quant). `—` = not run.

## qkv

| nrow | ADC 6 b | ADC 5 b |
|---|---|---|
| 64 | — | 31.72 (1.041) |
| 128 | 30.97 (1.017) | — |

## o

| nrow | ADC 6 b | ADC 5 b |
|---|---|---|
| 256 | 324.64 (10.658) | — |

## all

| nrow | ADC 6 b | ADC 5 b |
|---|---|---|
| 64 | — | 31.94 (1.048) |
