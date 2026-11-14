# A5 noise per layer type `a5n_opt125m`

facebook/opt-125m, statistical, first 4096 WikiText-2 test tokens, base `a4_stat_adc6cal.yaml`. Fake-quant 25.344, noise-free analog 25.594.

Noise on one layer type, the others noise-free. Cells: median PPL over seeds (ratio to noise-free analog) [min – max]. `*` = statistical mode not valid (findings F13).

| Layer type | sigma 0.1 % | sigma 0.2 % |
|---|---|---|
| qkv | 27.80 (1.086) [27.59 – 28.86] | 10752.60 (420.116) [10113.96 – 10897.43] |
| o | 25.67 (1.003) [25.66 – 25.71] | 150.57 (5.883) [141.22 – 159.55] |
| up_gate | 326.30 (12.749) [320.53 – 461.71] | 12761.38 (498.601) [12711.00 – 12798.02] |
| down | 26.49 (1.035) [26.27 – 26.71] | 4925.38 (192.440) [4845.42 – 5045.85] |
| all | 230.05 (8.988) [60.21 – 235.54] | 13202.77 (515.847) [12845.98 – 13266.35] |
