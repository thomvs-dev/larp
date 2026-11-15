# A5 sweep `a5f_gpt2_ba`

gpt2, bit_accurate, first 8192 WikiText-2 test tokens. Base: other layers as `a5_base_adc8full.yaml`. Fake-quant PPL on the same tokens: 33.776.

Cells: PPL (ratio to fake-quant). `—` = not run. `*` = statistical mode not valid at this point (ADC step not 1/n MAC unit on some layers, findings F13).

## all

| nrow | ADC 8 b | ADC 7 b | ADC 6 b |
|---|---|---|---|
| 128 | 33.96 (1.006) | 34.99 (1.036) | 70519.35 (2087.880) |
