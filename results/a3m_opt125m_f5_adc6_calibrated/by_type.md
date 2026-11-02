| Layer type | n | ADC b | rel. error (median / max) | exact (%) | bias/std | excess kurtosis | token RMS q90/median | corr(log token err, log token signal) |
|---|---|---|---|---|---|---|---|---|
| model.decoder.layers.*.self_attn.q_proj | 12 | 6 | 2.08e-03 / 2.16e-02 | 100.0 | -0.001 | 437093.37 | 0.00 | +0.04 |
| model.decoder.layers.*.self_attn.k_proj | 12 | 6 | 9.11e-07 / 3.61e-03 | 100.0 | +0.000 | 154275.63 | 0.00 | +0.05 |
| model.decoder.layers.*.self_attn.v_proj | 12 | 6 | 4.27e-03 / 2.93e-02 | 100.0 | +0.000 | 198230.53 | 0.00 | +0.03 |
| model.decoder.layers.*.self_attn.out_proj | 12 | 6 | 0.00e+00 / 3.56e-06 | 100.0 | +0.000 | 0.00 | 0.00 | +nan |
| model.decoder.layers.*.fc1 | 12 | 6 | 2.42e-04 / 5.89e-03 | 100.0 | +0.002 | 459446.90 | 0.00 | +0.02 |
| model.decoder.layers.*.fc2 | 12 | 8 | 6.34e-07 / 8.74e-07 | 100.0 | -3.161 | 0.15 | 1.37 | +0.77 |
