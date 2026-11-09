# A5: calibration histogram cache, real-model check

OPT-125M, cloud sandbox (2 threads), git 35f7427, first 4,094 WikiText-2 test tokens (2 windows of 2,048),
calibration on 2,048 WikiText-2 train tokens, statistical mode, W8A8 bit-serial, nrow 128, MSE-calibrated
ADC range per layer. Cache directory empty before the first run.

| Run | Config | Cache | Calibration (s) | PPL | `adc_full_scale.json` md5 |
|---|---|---|---|---|---|
| `a5cc_opt125m_adc6_miss` | `a4_stat_adc6cal.yaml` | miss (recorded) | 273.0 | 25.5911 | 6112af06… |
| `a5cc_opt125m_adc6_hit` | `a4_stat_adc6cal.yaml` | hit (72 / 72 layers) | 10.2 | 25.5911 | 6112af06… |
| `a5cc_opt125m_adc5_hit` | `a5_stat_adc5cal.yaml` | hit (72 / 72 layers) | 10.2 | 12194.75 | 3918e6c0… |
| `a5cc_opt125m_adc5_nocache` | `a5_stat_adc5cal.yaml` | off (`--no-calib-cache`) | 284.4 | 12194.75 | 3918e6c0… |

Cached and fresh calibration give bit-identical perplexity and ADC full scales; a config that differs
only in ADC bits reuses the histograms. The remaining 10 s on a hit are model fingerprint, loading the
cache files (15 MB for OPT-125M) and computing full scales / statistical tables.

Side observation (not part of the cache check): with a 5-bit calibrated ADC OPT-125M collapses
(PPL 12,195 vs 25.59 at 6 b on the same tokens). F5 predicted large clipping error at 5 b on the
synthetic layer (relative error 1.51); this is the first whole-model number. To be studied in the A5
ADC-bits sweep.
