# A5 sweep `a5v3_smoke_opt125m`

facebook/opt-125m, statistical, first 8192 WikiText-2 test tokens. Base: other layers as `a5_base_adc8cal.yaml`. Fake-quant PPL on the same tokens: 30.460.

Cells: PPL (ratio to fake-quant). `—` = not run. `*` = statistical mode not valid at this point (ADC step not 1/n MAC unit on some layers, findings F13).

## qkv

| nrow | ADC 6 b |
|---|---|
| 128 | 30.97 (1.017) |

## o

| nrow | ADC 6 b |
|---|---|
| 256 | 324.64 (10.658)* |
