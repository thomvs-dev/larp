# B1 calibration, m2_sinangil_jssc21 (git baf03ef)

Fitted at 0.8 V, 7 nm, per conversion (one 4-b output, 256 cells, 64 MACs): data-independent 289.6 fJ (charged to the 15 flash comparators, 19.31 fJ each); wordline 12.00 aJ per cell and pulse (0.0188 fF); bit line 0.593 fJ per cell at full activity (0.927 fF effective, compute and compensation capacitors included). Average 4b x 4b MAC: 5.76 fJ.

| Check | Kind | Model | Paper | Error |
|---|---|---|---|---|
| energy at MAC 0, 0.8 V (pJ, Fig. 13a) | held-out (digitised curve) | 4.633 | 3.3 | +40.4 % |
| energy at MAC 1, 0.8 V (pJ, Fig. 13a) | held-out (digitised curve) | 4.844 | 4.55 | +6.5 % |
| energy at MAC 2, 0.8 V (pJ, Fig. 13a) | held-out (digitised curve) | 5.056 | 5.05 | +0.1 % |
| energy at MAC 4, 0.8 V (pJ, Fig. 13a) | held-out (digitised curve) | 5.478 | 5.45 | +0.5 % |
| energy at MAC 6, 0.8 V (pJ, Fig. 13a) | fit | 5.9 | 5.9 | +0.0 % |
| energy at MAC 8, 0.8 V (pJ, Fig. 13a) | held-out (digitised curve) | 6.322 | 6.3 | +0.4 % |
| energy at MAC 10, 0.8 V (pJ, Fig. 13a) | held-out (digitised curve) | 6.744 | 6.65 | +1.4 % |
| energy at MAC 12, 0.8 V (pJ, Fig. 13a) | held-out (digitised curve) | 7.167 | 7.05 | +1.7 % |
| energy at MAC 15, 0.8 V (pJ, Fig. 13a) | fit | 7.8 | 7.8 | +0.0 % |
| TOPS/W average, 0.8 V | held-out | 347.1 | 351 | -1.1 % |
| TOPS/W lowest (MAC 15), 0.8 V | held-out | 262.6 | 262.3 | +0.1 % |
| TOPS/W highest (MAC 0), 0.8 V | held-out | 442 | 610.5 | -27.6 % |
| energy at MAC 15, 0.65 V (pJ, Fig. 13b) | held-out (VDD^2 rule) | 5.149 | 5.1 | +1.0 % |
| energy at MAC 15, 0.70 V (pJ, Fig. 13b) | held-out (VDD^2 rule) | 5.972 | 5.9 | +1.2 % |
| energy at MAC 15, 0.75 V (pJ, Fig. 13b) | held-out (VDD^2 rule) | 6.855 | 6.8 | +0.8 % |
| energy at MAC 15, 0.85 V (pJ, Fig. 13b) | held-out (VDD^2 rule) | 8.805 | 9 | -2.2 % |
| energy at MAC 15, 0.90 V (pJ, Fig. 13b) | held-out (VDD^2 rule) | 9.872 | 10.3 | -4.2 % |
| energy at MAC 15, 0.95 V (pJ, Fig. 13b) | held-out (VDD^2 rule) | 11 | 11.7 | -6.0 % |
| energy at MAC 15, 1.00 V (pJ, Fig. 13b) | held-out (VDD^2 rule) | 12.19 | 13.1 | -7.0 % |
| TOPS/W average, 1.0 V (Table I) | held-out (VDD^2 rule) | 222.2 | 321 | -30.8 % |
| TOPS/W lowest, 1.0 V (Table I) | held-out (VDD^2 rule) | 168 | 189.3 | -11.2 % |
| TOPS/W highest, 1.0 V (Table I) | held-out (VDD^2 rule) | 282.9 | 435.5 | -35.0 % |
| throughput 0.8 V (GOPS) | consistency (cycle time is an input) | 372.4 | 372.4 | -0.0 % |
| macro area (mm2) | consistency (area is an input) | 0.0032 | 0.0032 | +0.0 % |

Notes:
- The measured curve has a step between MAC 0 and MAC 1 that a line cannot follow; the fit uses the operating range (codes 6 and 15), so MAC 0 is the worst point.
- Table I at 1.0 V and Fig. 13(b) disagree in the paper itself: Fig. 13(b) gives 13.1 pJ at MAC 15 and 1.0 V (= 156 TOPS/W), Table I gives 189.3 TOPS/W as the lowest value at 1.0 V. The Table I values at 1.0 V also scale more slowly than VDD^2 from the 0.8 V values (average: x1.09 in energy).
