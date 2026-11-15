# B1 calibration, m1_jia_jssc20 (git baf03ef)

Fitted at 1.2 V, 65 nm, per cell and evaluation: data-dependent 11.69 fJ per active cell (assumed activity 0.5), data-independent 3.01 fJ. ADC (8-b SAR): DAC 589 fJ (1.2 fF unit, conventional switching) + 371 fJ per bit cycle (comparator + logic, fitted). Array + ADC: 10.40 fJ per 1-b MAC.

| Check | Kind | Model | Paper | Error |
|---|---|---|---|---|
| array energy per column, 1.2 V (pJ) | fit | 20.4 | 20.4 | +0.0 % |
| ADC energy per column, 1.2 V (pJ) | fit | 3.56 | 3.56 | +0.0 % |
| array energy per 1-b MAC, 1.2 V (fJ, Fig. 12) | consistency | 8.854 | 8.85 | +0.0 % |
| ADC energy per 1-b MAC, 1.2 V (fJ, Fig. 12) | consistency | 1.545 | 1.55 | -0.3 % |
| 1b-TOPS/W, 1.2 V (abstract) | consistency (confirms 2 ops per 1-b MAC, array + ADC) | 192.3 | 192 | +0.2 % |
| ADC / array energy (Sec. II-B: 18 %) | consistency | 17.45 | 18 | -3.1 % |
| array energy per column, 0.85 V (pJ) | held-out (VDD^2 rule) | 10.24 | 9.7 | +5.5 % |
| ADC energy per column, 0.85 V (pJ) | held-out (VDD^2 rule) | 1.786 | 1.79 | -0.2 % |
| 1b-TOPS/W, 0.85 V (abstract) | held-out (VDD^2 rule) | 383.3 | 400 | -4.2 % |

Notes:
- 192 / 400 1b-TOPS/W equal 2 x 2304 / (array + ADC energy per column) from Table I: the abstract counts 2 ops per 1-b MAC and 'MAC hardware' is the array plus the ADC (no reshaping buffer, no ABN, no NMC).
- Weight loading (Table I, Fig. 12): 600 fJ per bit at 1.2 V, 300 fJ at 0.85 V, 55.2x one 1-b MAC. Input for the array-write cost in B3.
- Not checked: area and throughput (the paper gives no macro-only area or evaluation time that the 0.60 / 0.24 1b-TOPS/mm2 can be rebuilt from). The data pattern of Table I is not stated; activity 0.5 is an assumption and only moves energy between the two array terms at the calibration point.
