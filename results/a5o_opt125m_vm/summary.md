# Activation statistics `a5o_opt125m_vm`

/home/claude/hf_local/opt-125m, FP32, first 2048 WikiText-2 test tokens (2048-token windows), inputs of the CIM layers (8-bit per-token codes as in the CIM layers). Per role: median over layers [max].

| Role | layers | peak/rms (median tok) | peak/rms (p99 tok) | outlier ch > 6x | top-1 % ch energy | code = 0 | code < 0 | abs(code) <= 1 | sign-plane density |
|---|---|---|---|---|---|---|---|---|---|
| qkv | 36 | 9.9 [13.7] | 13.9 [18.8] | 6 [7] | 0.39 [0.72] | 0.04 [0.09] | 0.49 [0.52] | 0.13 [0.26] | 0.49 [0.52] |
| o | 12 | 7.3 [9.8] | 12.0 [15.3] | 2 [5] | 0.16 [0.31] | 0.07 [0.10] | 0.47 [0.49] | 0.18 [0.25] | 0.47 [0.49] |
| up_gate | 12 | 12.3 [17.9] | 16.2 [22.1] | 1 [3] | 0.26 [0.68] | 0.05 [0.09] | 0.40 [0.58] | 0.14 [0.26] | 0.40 [0.58] |
| down | 12 | 15.0 [24.5] | 27.2 [40.2] | 40 [351] | 0.22 [0.40] | 0.93 [0.99] | 0.00 [0.00] | 0.93 [0.99] | 0.00 [0.00] |
