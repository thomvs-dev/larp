# A5 sweep `a5r_qwen15b_stat`

Qwen/Qwen2.5-1.5B, statistical, first 8192 WikiText-2 test tokens. Base: other layers as `a5_base_adc8cal.yaml`. Fake-quant PPL on the same tokens: 10.338.

Cells: PPL (ratio to fake-quant). `—` = not run. `*` = statistical mode not valid at this point (ADC step not 1/n MAC unit on some layers, findings F13).

## all

| nrow | ADC 6 b | ADC 5 b |
|---|---|---|
| 64 | — | 10.36 (1.002) |
| 128 | 10.34 (1.000) | — |
