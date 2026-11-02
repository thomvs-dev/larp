# A3 step 1 — error measurement on OPT-125M (2,048 WikiText-2 train tokens per config)

Raw data: `results/a3m_opt125m_*/error_stats.json` (`samples.pt` not in git). Error e in integer MAC units, relative error = ‖e‖/‖ideal‖ per layer.

| Config | rel. error, median over layers excl. fc2 (max) | fc2 rel. error (median) | error = gain × ideal? | excess kurtosis (median) | token RMS spread q90/q50 |
|---|---|---|---|---|---|
| cal. ADC 6 b (fc2 8 b), no noise | 1.09e-04 (2.93e-02) | 6.28e-07 | no (residual 100%) | 10976.99 | 1.46 |
| cal. ADC 6 b, noise 0.25 % | 1.43e+00 (3.99e+00) | 1.40e+00 | no (residual 100%) | 0.33 | 1.04 |
| cal. ADC 6 b, noise 0.5 % | 2.35e+00 (4.70e+00) | 2.18e+00 | no (residual 100%) | 0.00 | 1.03 |
| cal. ADC 6 b, noise 1 % | 4.43e+00 (8.48e+00) | 3.75e+00 | no (residual 100%) | 0.00 | 1.03 |
| full-range ADC 8 b (fc2 10 b) | 3.92e-03 (6.76e-03) | 9.78e-04 | yes (residual 0.1% of error, median) | 1.19 | 1.27 |
| full-range ADC 7 b (fc2 9 b) | 7.87e-03 (1.35e-02) | 1.96e-03 | yes (residual 0.0% of error, median) | 1.26 | 1.27 |
