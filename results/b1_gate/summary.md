# GATE B1 summary (git baf03ef, limit 30 %)

| Macro | Held-out points | Worst held-out error | Points over the limit |
|---|---|---|---|
| M1 m1_jia_jssc20 | 3 | +5.5 % (array energy per column, 0.85 V (pJ)) | none |
| M2 m2_sinangil_jssc21 | 20 | +40.4 % (energy at MAC 0, 0.8 V (pJ, Fig. 13a)) | energy at MAC 0, 0.8 V (pJ, Fig. 13a) +40.4 %; TOPS/W average, 1.0 V (Table I) -30.8 %; TOPS/W highest, 1.0 V (Table I) -35.0 % |
| M4 m4_chih_isscc21 | 9 | -12.7 % (TOPS/W 4b/4b toggle 0.0625 (Fig. 16.4.5)) | none |

Energy per 4b x 4b MAC, each macro's calibrated value moved to 22 nm / 0.8 V with E ~ node^1 x VDD^2:

| Macro | Own node / VDD | fJ per MAC there | Factor | fJ per MAC at the result node | Condition |
|---|---|---|---|---|---|
| M1 | 65 nm / 1.2 V | 166.4 | 0.150 | 25.0 | 16 1-b MACs, array + 8-b SAR ADC, activity 0.5 |
| M2 | 7 nm / 0.8 V | 5.8 | 3.143 | 18.1 | average point of Fig. 13(a), 4-b output |
| M4 | 22 nm / 0.72 V | 22.5 | 1.235 | 27.7 | 18 % input toggle, exact digital |

Spread (largest / smallest): 1.53x. The three macros differ in architecture, output precision and data pattern, so this is a plausibility check of the scaling rule, not a test of it.
