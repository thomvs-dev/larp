# A5 noise per layer type `a5n_opt125m_ba`

facebook/opt-125m, bit_accurate, first 4096 WikiText-2 test tokens, base `a4_stat_adc6cal.yaml`. Fake-quant 25.344, noise-free analog 25.596.

Noise on one layer type, the others noise-free. Cells: median PPL over seeds (ratio to noise-free analog) [min – max]. `*` = statistical mode not valid (findings F13).

| Layer type | sigma 0.1 % |
|---|---|
| up_gate | 122.64 (4.791) [122.07 – 134.60] |
