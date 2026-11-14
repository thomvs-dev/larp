# Activation statistics `a5o_qwen15b`

Qwen/Qwen2.5-1.5B, FP32, first 2048 WikiText-2 test tokens (2048-token windows), inputs of the CIM layers (8-bit per-token codes as in the CIM layers). Per role: median over layers [max].

| Role | layers | peak/rms (median tok) | peak/rms (p99 tok) | outlier ch > 6x | top-1 % ch energy | code = 0 | code < 0 | abs(code) <= 1 | sign-plane density |
|---|---|---|---|---|---|---|---|---|---|
| qkv | 84 | 15.4 [18.0] | 28.7 [36.5] | 16 [121] | 0.46 [0.56] | 0.09 [0.12] | 0.45 [0.46] | 0.27 [0.35] | 0.45 [0.46] |
| o | 28 | 8.4 [12.0] | 13.3 [18.0] | 6 [13] | 0.16 [0.57] | 0.07 [0.17] | 0.47 [0.51] | 0.18 [0.36] | 0.47 [0.51] |
| up_gate | 56 | 14.2 [20.8] | 23.7 [31.9] | 7 [12] | 0.34 [0.77] | 0.06 [0.15] | 0.46 [0.47] | 0.19 [0.42] | 0.46 [0.47] |
| down | 28 | 26.2 [47.4] | 59.0 [87.5] | 22 [195] | 0.18 [0.99] | 0.22 [0.62] | 0.39 [0.42] | 0.47 [0.89] | 0.39 [0.42] |
