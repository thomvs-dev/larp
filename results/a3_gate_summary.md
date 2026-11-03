| Config | tokens | fake-quant | bit-accurate (runs) | spread | statistical (seeds) | diff | tolerance | pass | speed-up |
|---|---|---|---|---|---|---|---|---|---|
| a3g_opt125m_adc7 | 8192 | 30.449 | fast_seed0 30.607, reference_seed0 30.530 | 0.25% | 31.315, 31.428, 31.210 | +2.31% | 2.00% | NO | 10x |
| a3g_opt125m_adc8 | 8192 | 30.449 | fast_seed0 30.483, reference_seed0 30.423 | 0.20% | 30.636, 30.716, 30.615 | +0.50% | 2.00% | yes | 11x |
| a3g_opt125m_f5 | 8192 | 30.449 | fast_seed0 30.808, reference_seed0 30.806 | 0.01% | 53.971, 58.839, 54.585 | +77.18% | 2.00% | NO | 15x |
| a3g_opt125m_rn008 | 4096 | 25.308 | fast_seed0 25.716, fast_seed1 25.731 | 0.06% | 43.561, 47.835, 45.710 | +77.75% | 2.00% | NO | 94x |
| a3g_opt125m_rn010 | 4096 | 25.308 | fast_seed0 137.197, fast_seed1 143.281, fast_seed2 150.460, fast_seed3 175.524, fast_seed4 76.018 | 130.90% | 94.836, 144.252, 308.707, 144.808, 155.987, 264.976, 113.140, 183.325 | +4.97% | 130.90% | yes | 18x |
| a3g_opt125m_rn02_hyb6 | 4096 | 25.308 | fast_seed0 29.232, fast_seed1 28.954 | 0.96% | 50.902, 57.195, 52.874 | +80.88% | 2.00% | NO | 102x |
