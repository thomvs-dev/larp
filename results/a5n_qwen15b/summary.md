# A5 noise per layer type `a5n_qwen15b`

Qwen/Qwen2.5-1.5B, statistical, first 4096 WikiText-2 test tokens, base `a4_stat_adc6cal.yaml`. Fake-quant 8.717, noise-free analog 8.741.

Noise on one layer type, the others noise-free. Cells: median PPL over seeds (ratio to noise-free analog) [min – max]. `*` = statistical mode not valid (findings F13).

| Layer type | sigma 0.05 % | sigma 0.1 % |
|---|---|---|
| qkv | — | 8.92 (1.020) |
| o | — | 8.74 (0.999) |
| up_gate | — | 9.64 (1.103) |
| down | — | 10.59 (1.211) |
| all | — | 13.46 (1.540) |
