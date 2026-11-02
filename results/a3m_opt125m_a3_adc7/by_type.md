| Layer type | n | ADC b | rel. error (median / max) | exact (%) | bias/std | excess kurtosis | token RMS q90/median | corr(log token err, log token signal) |
|---|---|---|---|---|---|---|---|---|
| model.decoder.layers.*.self_attn.q_proj | 12 | 7 | 7.95e-03 / 1.35e-02 | 0.4 | +0.009 | 37.56 | 1.24 | +0.99 |
| model.decoder.layers.*.self_attn.k_proj | 12 | 7 | 7.87e-03 / 8.24e-03 | 0.3 | -0.006 | 0.83 | 1.22 | +1.00 |
| model.decoder.layers.*.self_attn.v_proj | 12 | 7 | 8.25e-03 / 1.21e-02 | 0.6 | +0.003 | 723.97 | 1.29 | +0.97 |
| model.decoder.layers.*.self_attn.out_proj | 12 | 7 | 7.87e-03 / 7.87e-03 | 0.4 | -0.013 | 0.94 | 1.32 | +1.00 |
| model.decoder.layers.*.fc1 | 12 | 7 | 7.87e-03 / 8.04e-03 | 0.1 | -1.409 | 1.65 | 1.23 | +1.00 |
| model.decoder.layers.*.fc2 | 12 | 9 | 1.96e-03 / 1.96e-03 | 4.1 | -0.000 | 1.05 | 1.30 | +1.00 |
