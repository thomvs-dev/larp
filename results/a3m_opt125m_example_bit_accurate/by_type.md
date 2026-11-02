| Layer type | n | ADC b | rel. error (median / max) | exact (%) | bias/std | excess kurtosis | token RMS q90/median | corr(log token err, log token signal) |
|---|---|---|---|---|---|---|---|---|
| model.decoder.layers.*.self_attn.q_proj | 12 | 8 | 3.94e-03 / 6.76e-03 | 0.7 | +0.009 | 10.12 | 1.23 | +0.99 |
| model.decoder.layers.*.self_attn.k_proj | 12 | 8 | 3.92e-03 / 4.05e-03 | 0.6 | -0.006 | 0.71 | 1.22 | +1.00 |
| model.decoder.layers.*.self_attn.v_proj | 12 | 8 | 4.09e-03 / 6.07e-03 | 1.2 | +0.003 | 833.30 | 1.29 | +0.99 |
| model.decoder.layers.*.self_attn.out_proj | 12 | 8 | 3.92e-03 / 3.92e-03 | 0.7 | -0.014 | 0.94 | 1.32 | +1.00 |
| model.decoder.layers.*.fc1 | 12 | 8 | 3.92e-03 / 4.00e-03 | 0.2 | -1.408 | 1.47 | 1.24 | +1.00 |
| model.decoder.layers.*.fc2 | 12 | 10 | 9.78e-04 / 9.78e-04 | 8.2 | -0.000 | 1.06 | 1.30 | +1.00 |
