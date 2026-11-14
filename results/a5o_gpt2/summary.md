# Activation statistics `a5o_gpt2`

gpt2, FP32, first 2048 WikiText-2 test tokens (1024-token windows), inputs of the CIM layers (8-bit per-token codes as in the CIM layers). Per role: median over layers [max].

| Role | layers | peak/rms (median tok) | peak/rms (p99 tok) | outlier ch > 6x | top-1 % ch energy | code = 0 | code < 0 | abs(code) <= 1 | sign-plane density |
|---|---|---|---|---|---|---|---|---|---|
| qkv | 12 | 14.5 [17.4] | 18.8 [20.6] | 4 [16] | 0.54 [0.80] | 0.07 [0.14] | 0.45 [0.51] | 0.21 [0.39] | 0.45 [0.51] |
| o | 12 | 6.9 [10.0] | 10.8 [16.1] | 4 [17] | 0.18 [0.33] | 0.05 [0.08] | 0.47 [0.49] | 0.14 [0.23] | 0.47 [0.49] |
| up_gate | 12 | 8.0 [10.9] | 14.2 [18.4] | 3 [9] | 0.22 [0.37] | 0.03 [0.05] | 0.47 [0.49] | 0.10 [0.16] | 0.47 [0.49] |
| down | 12 | 13.4 [18.6] | 23.8 [36.6] | 2 [16] | 0.13 [0.32] | 0.05 [0.07] | 0.82 [0.84] | 0.15 [0.19] | 0.82 [0.84] |
