# B5: Jia JSSC 2020 macro, AccelForge (CiMLoop model) vs this simulator (git 2931d87)

AccelForge 1.0.551 (repo df6a79f4), run at our git 2931d87. Paper = Table I of Jia et al. (per column and evaluation, 1-b). This simulator = M1 calibration (`results/b1_m1_jia/`), fitted to Table I at 1.2 V, other supplies by VDD².

## Table I conditions (1-b inputs and weights), per column and evaluation

| VDD | Quantity | Paper | AccelForge | This simulator | AccelForge vs paper | Ours vs paper |
|---|---|---|---|---|---|---|
| 0.85 V | array (CIMA) pJ | 9.7 | 10 | 10.2 | +3.2 % | +5.5 % |
| 0.85 V | ADC pJ | 1.79 | 1.79 | 1.79 | -0.2 % | -0.2 % |
| 0.85 V | NMC data path pJ | 8.3 | 7.71 | not modelled | -7.1 % | — |
| 0.85 V | 1b-TOPS/W  | 400 | 391 | 383 | -2.4 % | -4.2 % |
| 1.2 V | array (CIMA) pJ | 20.4 | 19.7 | 20.4 | -3.6 % | +0.0 % |
| 1.2 V | ADC pJ | 3.56 | 3.56 | 3.56 | -0.0 % | +0.0 % |
| 1.2 V | NMC data path pJ | 14.7 | 14.7 | not modelled | -0.0 % | — |
| 1.2 V | 1b-TOPS/W  | 192 | 198 | 192 | +3.3 % | +0.2 % |

## Part B operating point (8-b inputs and weights), energy per 8 b × 8 b MAC, array + ADC

| VDD | AccelForge (fJ) | This simulator (fJ) | Ours / AccelForge | AccelForge, all components (fJ) |
|---|---|---|---|---|
| 0.85 V | 347.3 | 333.9 | 0.96× | 406.2 |
| 1.2 V | 684.3 | 665.6 | 0.97× | 782.8 |

Differences are explained in `validation/tool_comparison.md`.
