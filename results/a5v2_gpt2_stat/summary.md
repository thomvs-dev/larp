# A5 sweep `a5v2_gpt2_stat`

gpt2, statistical, first 8192 WikiText-2 test tokens. Base: other layers as `a5_base_adc8cal.yaml`. Fake-quant PPL on the same tokens: 33.776.

Cells: PPL (ratio to fake-quant). `—` = not run.

## o

| nrow | ADC 7 b | ADC 5 b |
|---|---|---|
| 64 | — | 33.82 (1.001) |

## up_gate

| nrow | ADC 7 b | ADC 5 b |
|---|---|---|
| 64 | — | 34.23 (1.013) |

## down

| nrow | ADC 7 b | ADC 5 b |
|---|---|---|
| 256 | 214821.37 (6360.257) | — |

## all

| nrow | ADC 7 b | ADC 5 b |
|---|---|---|
| 256 | 212640.92 (6295.700) | — |
