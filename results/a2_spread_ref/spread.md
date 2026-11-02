# Spread: a2_spread_fast (backend fast) vs a2_spread_ref (backend reference), 1000 images

| Config | sim | acc A (%) | acc B (%) | diff B - A | images with different class (%) |
|---|---|---|---|---|---|
| bs_n128_adc6 | cimsim | 60.20 | 61.30 | +1.10 | 29.2 |
| bs_n128_adc5 | cimsim | 12.80 | 13.30 | +0.50 | 69.6 |
| bs_n128_adc4 | cimsim | 9.90 | 9.50 | -0.40 | 75.0 |
| bs_n255_adc6 | cimsim | 13.00 | 11.60 | -1.40 | 68.7 |
| bp2_n128_adc6 | cimsim | 12.90 | 11.60 | -1.30 | 56.8 |
| bs_n128_adc6 | cimsim vs ASiM (run A) | 59.00 | 60.20 | +1.20 | 33.0 |
| bs_n128_adc5 | cimsim vs ASiM (run A) | 13.10 | 12.80 | -0.30 | 77.2 |
| bs_n128_adc4 | cimsim vs ASiM (run A) | 10.30 | 9.90 | -0.40 | 75.2 |
| bs_n255_adc6 | cimsim vs ASiM (run A) | 13.00 | 13.00 | +0.00 | 69.1 |
| bp2_n128_adc6 | cimsim vs ASiM (run A) | 13.10 | 12.90 | -0.20 | 70.6 |
