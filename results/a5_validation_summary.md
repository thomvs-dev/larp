# A5: statistical vs bit-accurate at sweep points outside GATE A3

OPT-125M, cloud sandbox (2 threads), git f59892f, first 4,096 WikiText-2 test tokens (2 windows), W8A8
bit-serial, MSE-calibrated ADC range per layer, no noise, calibration on 2,048 train tokens.
Script `scripts/a3_gate.py` (one calibration shared by both modes; statistical is deterministic without noise).

| Point | Fake-quant | Bit-accurate | Statistical | Stat − BA | ms/token BA / stat | Exact saturation work |
|---|---|---|---|---|---|---|
| ADC 6 b, nrow 64 (`a5v_opt125m_adc6cal_n64`) | 25.381 | 25.381 | 25.381 | 0.00 % | 103.3 / 1.6 (66×) | 0 % |
| ADC 5 b, nrow 128 (`a5v_opt125m_adc5cal_n128`) | 25.381 | 10,265.9 | 12,194.8 | +18.8 % | 68.1 / 52.5 (1.3×) | 43 % |
| ADC 4 b, nrow 128 (`a5v_opt125m_adc4cal_n128`) | 25.381 | 12,108.7 | 11,422.3 | −5.7 % | 71.8 / 115.5 (0.6×) | 66 % |
| ADC 6 b, nrow 256 (`a5v_opt125m_adc6cal_n256`) | 25.381 | 11,442.5 | 12,899.1 | +12.7 % | 54.4 / 30.1 (1.8×) | 32 % |

Reference inside GATE A3: ADC 6 b, nrow 128 on 8,192 tokens: bit-accurate 30.808, statistical 30.80
(`results/a3g_opt125m_f5/`).

Reading:
- nrow 64 with a 6-bit ADC is lossless (every column MAC fits in 63 codes): both modes equal fake-quant.
- ADC 5 b and 4 b at nrow 128, and ADC 6 b at nrow 256, destroy the model in both modes (PPL 10k–13k).
  Both modes agree that the model is destroyed; the exact numbers differ by 6–19 %, which at this
  level means nothing (F6: chaotic regime). Formally these points fail the 2 % rule.
- In that regime statistical mode has no speed advantage (0.6–1.8×): 32–66 % of all conversions
  need the exact saturation path.
- The working region on OPT-125M is narrow: a calibrated ADC needs 2^b − 1 ≳ nrow / 2. A likely
  cause, not yet tested: with two's-complement inputs the sign-extension planes have bit density
  ≈ 0.41 (F3), so their column MACs sit near 0.41 · nrow (≈ 52 at nrow 128, ≈ 105 at nrow 256);
  above 2^b − 1 they saturate on many columns at once.
