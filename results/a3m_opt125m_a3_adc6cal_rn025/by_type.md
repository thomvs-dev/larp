| Layer type | n | ADC b | rel. error (median / max) | exact (%) | bias/std | excess kurtosis | token RMS q90/median | corr(log token err, log token signal) |
|---|---|---|---|---|---|---|---|---|
| model.decoder.layers.*.self_attn.q_proj | 12 | 6 | 2.84e+00 / 3.68e+00 | 0.0 | +0.000 | 0.33 | 1.04 | +0.01 |
| model.decoder.layers.*.self_attn.k_proj | 12 | 6 | 3.05e+00 / 3.87e+00 | 0.0 | +0.000 | 0.33 | 1.04 | -0.00 |
| model.decoder.layers.*.self_attn.v_proj | 12 | 6 | 3.50e+00 / 3.99e+00 | 0.0 | -0.000 | 0.33 | 1.04 | +0.00 |
| model.decoder.layers.*.self_attn.out_proj | 12 | 6 | 6.63e-01 / 1.09e+00 | 0.0 | +0.000 | 0.33 | 1.04 | +0.00 |
| model.decoder.layers.*.fc1 | 12 | 6 | 1.45e+00 / 3.98e+00 | 0.0 | -0.000 | 0.33 | 1.02 | -0.01 |
| model.decoder.layers.*.fc2 | 12 | 8 | 1.47e+00 / 3.36e+00 | 0.0 | +0.005 | 0.03 | 1.05 | +0.68 |
