| Layer type | n | ADC b | rel. error (median / max) | exact (%) | bias/std | excess kurtosis | token RMS q90/median | corr(log token err, log token signal) |
|---|---|---|---|---|---|---|---|---|
| model.decoder.layers.*.self_attn.q_proj | 12 | 6 | 5.27e+00 / 7.75e+00 | 0.0 | +0.001 | 0.00 | 1.03 | -0.01 |
| model.decoder.layers.*.self_attn.k_proj | 12 | 6 | 5.23e+00 / 8.14e+00 | 0.0 | -0.000 | 0.00 | 1.03 | +0.01 |
| model.decoder.layers.*.self_attn.v_proj | 12 | 6 | 5.89e+00 / 8.48e+00 | 0.0 | -0.000 | -0.00 | 1.03 | +0.01 |
| model.decoder.layers.*.self_attn.out_proj | 12 | 6 | 2.42e+00 / 3.45e+00 | 0.0 | -0.001 | -0.00 | 1.03 | +0.00 |
| model.decoder.layers.*.fc1 | 12 | 6 | 2.84e+00 / 4.68e+00 | 0.0 | -0.000 | -0.00 | 1.02 | +0.00 |
| model.decoder.layers.*.fc2 | 12 | 8 | 3.82e+00 / 8.20e+00 | 0.0 | +0.005 | 0.04 | 1.05 | +0.65 |
