| Layer type | n | ADC b | rel. error (median / max) | exact (%) | bias/std | excess kurtosis | token RMS q90/median | corr(log token err, log token signal) |
|---|---|---|---|---|---|---|---|---|
| model.decoder.layers.*.self_attn.q_proj | 12 | 6 | 3.71e+00 / 4.48e+00 | 0.0 | +0.001 | -0.00 | 1.03 | -0.01 |
| model.decoder.layers.*.self_attn.k_proj | 12 | 6 | 3.62e+00 / 4.52e+00 | 0.0 | +0.000 | 0.00 | 1.03 | +0.01 |
| model.decoder.layers.*.self_attn.v_proj | 12 | 6 | 4.04e+00 / 4.59e+00 | 0.0 | -0.000 | -0.00 | 1.03 | -0.00 |
| model.decoder.layers.*.self_attn.out_proj | 12 | 6 | 1.32e+00 / 2.06e+00 | 0.0 | -0.000 | 0.00 | 1.03 | +0.00 |
| model.decoder.layers.*.fc1 | 12 | 6 | 1.91e+00 / 4.70e+00 | 0.0 | -0.000 | 0.00 | 1.02 | +0.00 |
| model.decoder.layers.*.fc2 | 12 | 8 | 2.18e+00 / 5.27e+00 | 0.0 | +0.005 | 0.04 | 1.06 | +0.68 |
