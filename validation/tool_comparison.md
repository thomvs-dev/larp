# B5: comparison with an existing tool (CiMLoop / AccelForge)

GATE B5: differences between this simulator and an existing CIM tool on an overlapping case, explained.

## Setup

- Tool: AccelForge 1.0.551 (repository git df6a79f4), the maintained successor of CiMLoop from the same
  group. Its `jia_jssc_2020` model is the CiMLoop model of the Jia JSSC 2020 macro (`docs/decisions.md`, 2026-11-16).
- Case: the Jia JSSC 2020 macro (65 nm, charge-domain, 8-b SAR ADC). It is our calibration macro M1 and the
  anchor of the Part B analog arrays. Workload is AccelForge's own peak-throughput matmul for this macro
  (M = 1, N = 256 outputs, K = 2,304 inputs). Four runs: 1-b and 8-b inputs and weights, at 0.85 V and 1.2 V.
- This simulator: the M1 calibration (`results/b1_m1_jia/`), fitted to Table I at 1.2 V and moved to
  0.85 V by VDD².
- Run: the i7-12700H laptop, WSL Ubuntu, Python 3.14, our git 2931d87 (AccelForge does not install on native
  Windows: `islpy` has no Windows wheel, and `hwcomponents-neurosim` exceeds the path length limit). The
  compare step ran in the project environment at the same commit.
- Script: `scripts/b5_cimloop_jia.py` (`run`, then `compare`). Results: `results/b5_cimloop_jia/`
  (`accelforge.json` holds per-component energies, `summary.md` holds the tables).

Grouping of AccelForge components is the one used in AccelForge's Jia notebook:

| Group | AccelForge components |
|---|---|
| array | RowDrivers, WeightDrivers, CimUnit, BitcellCapacitor, ColumnDrivers |
| ADC | ADC |
| NMC data path | OutDatapath, ShiftAdd |
| zero gating | InputZeroGating |

## Results

Table I conditions (1-b), per column and evaluation:

| VDD | Quantity | Paper | AccelForge | This simulator | AccelForge vs paper | Ours vs paper |
|---|---|---|---|---|---|---|
| 0.85 V | array pJ | 9.7 | 10.0 | 10.2 | +3.2 % | +5.5 % |
| 0.85 V | ADC pJ | 1.79 | 1.79 | 1.79 | −0.2 % | −0.2 % |
| 0.85 V | NMC data path pJ | 8.3 | 7.71 | not modelled | −7.1 % | — |
| 0.85 V | 1b-TOPS/W | 400 | 391 | 383 | −2.4 % | −4.2 % |
| 1.2 V | array pJ | 20.4 | 19.7 | 20.4 | −3.6 % | 0.0 % |
| 1.2 V | ADC pJ | 3.56 | 3.56 | 3.56 | 0.0 % | 0.0 % |
| 1.2 V | NMC data path pJ | 14.7 | 14.7 | not modelled | 0.0 % | — |
| 1.2 V | 1b-TOPS/W | 192 | 198 | 192 | +3.3 % | +0.2 % |

Part B operating point (8-b inputs and weights), energy per 8 b × 8 b MAC, array + ADC:

| VDD | AccelForge (fJ) | This simulator (fJ) | Ours / AccelForge | AccelForge, all components (fJ) |
|---|---|---|---|---|
| 0.85 V | 347.3 | 333.9 | 0.96× | 406.2 |
| 1.2 V | 684.3 | 665.6 | 0.97× | 782.8 |

The two tools agree within 4.5 % on every compared quantity. Each is within 5.5 % of the paper at the
Table I conditions.

## Explanation of the differences

1. **1-b array energy (AccelForge −3.6 % at 1.2 V, ours 0.0 %).** Our M1 model was fitted to Table I at
   1.2 V, so it matches there by construction. AccelForge's array energy comes from its component models and
   the input and weight value distributions of its workload, not from a fit. Its 3–4 % offset is the size of
   an independent model's error, not a disagreement in what is counted.
2. **1-b array energy at 0.85 V (ours +5.5 %, AccelForge +3.2 %).** Both tools scale the 1.2 V numbers with
   VDD², and the paper's array energy falls a little faster than VDD² (20.4 → 9.7 pJ is ×0.475; VDD² is
   ×0.502). Our error is the held-out 0.85 V error already reported for M1 (`results/b1_m1_jia/`).
3. **ADC: equal within 0.2 %.** Both tools take the ADC energy per conversion from the paper. At 8 b both
   count 64 conversions per column per 8 b × 8 b evaluation (8 input bits × 8 weight-bit columns): the ADC
   energy in AccelForge is exactly 64.0× the 1-b value, the same factor our event counts give.
4. **1b-TOPS/W (AccelForge +3.3 % at 1.2 V).** This follows from item 1: lower array energy gives higher
   TOPS/W. The TOPS/W convention is the same in both tools (2 operations per 1-b MAC, array + ADC only, NMC
   data path excluded; `docs/decisions.md`, 2026-11-15 and AccelForge's notebook).
5. **8-b operating point (ours 3–4 % below AccelForge).** Going from 1-b to 8-b, AccelForge scales every
   array component by exactly 64× except the row drivers (83×) and the zero-gating logic (86×).
   AccelForge's row-driver energy depends on how many input bits are 1 in its 8-b input distribution, which
   differs from the input pattern behind Table I. Our model keeps the Table I input activity at 8 b, so its
   array energy is exactly 64× the 1-b value. At 1.2 V the extra row-driver energy is 6.7 % of AccelForge's
   8-b array energy, which accounts for the gap. Which input activity is right depends on the activations:
   for an LLM layer the density of the high input bit planes sets it (F15), and neither tool's default
   uses LLM activations. This simulator's system model counts conversions and array cycles; the cost per
   cycle uses the calibrated activity (`docs/assumptions.md` B-17, B-18).
6. **NMC data path (not modelled here).** The near-memory data path of the Jia macro (shift-add,
   accumulation, batch-norm and activation logic per column) is 8.3 / 14.7 pJ per column in the paper, about
   as much as the array. AccelForge models it (−7.1 % / 0.0 %). Our M1 model covers array + ADC, which is the
   part Table I's TOPS/W covers. Part B's system model adds its own shift-add and register energy per
   conversion from cited 45 nm digital costs scaled to 22 nm (`cimsim/circuits/analog.py`,
   `docs/assumptions.md` B-12 … B-14), not Jia's NMC block. That 22 nm term is 46 fJ per conversion
   (`results/b3_ppa_qwen15b/`) against 567 fJ total. Taking Jia's NMC energy instead would raise the
   compute energy per token. It would not change the Part B result that decode energy is set by
   DRAM weight reads (0.37 mJ compute against 50 mJ per decoded token for Qwen2.5-1.5B).
7. **Latency was not compared.** The M1 calibration fits energy only. AccelForge gives 0.54 µs per 1-b
   evaluation of the full array at 1.2 V and 34.5 µs per 8-b evaluation (`accelforge.json`,
   `latency_per_mac_s` × 256 × 2,304). Part B reports latency as a DRAM-bandwidth bound, where the macro
   timing does not set the result.

Note on `summary.json`: in the 8-b rows, the `ours` fields `array_per_column_J` / `adc_per_column_J` are
the 1-b per-column values that the 8-b per-MAC energy is built from. `tops_w` there counts one
8 b × 8 b MAC as 2 operations. The tables above use the per-MAC values.

## What the comparison does and does not show

- It shows that the per-action costs behind the Part B analog arrays, at the M1 anchor (65 nm, before
  scaling to 22 nm), match an independent published CIM tool within 4.5 %. It also shows that the two
  tools count ADC conversions the same way at the 8-b operating point.
- It does not check node scaling to 22 nm, the system model (DRAM, mapping, LM head), or accuracy:
  AccelForge has no accuracy simulation. The accuracy side was checked against ASiM in GATE A2.
- NeuroSim V1.5 was not run (the 2026-11-15 plan names one CiMLoop case only).

**GATE B5: passed.** Differences are at most 4.5 % between the tools, and each is explained above.
