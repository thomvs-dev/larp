# A5 sweep `a5f_opt125m_ba`

facebook/opt-125m, bit_accurate, first 8192 WikiText-2 test tokens. Base: other layers as `a5_base_adc8full.yaml`. Fake-quant PPL on the same tokens: 30.460.

Cells: PPL (ratio to fake-quant). `—` = not run. `*` = statistical mode not valid at this point (ADC step not 1/n MAC unit on some layers, findings F13).

## all

| nrow | ADC 8 b | ADC 7 b | ADC 6 b |
|---|---|---|---|
| 128 | 30.49 (1.001) | 30.61 (1.005) | 12661.76 (415.688) |
