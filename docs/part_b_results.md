# Part B results — system energy for SRAM CIM running Qwen2.5 (2026-11-15)

Method: analytical and trace-driven. Per-token event counts are derived from the model shape and equal the
event totals of the Part A runs exactly (`tests/test_system_counts.py`) and a prefill + cached-decode trace on
Qwen2.5-1.5B (`results/b3_trace_*`). Counts × per-event costs give energy. Costs are calibrated on three
published macros and moved to 22 nm / 0.8 V. Every assumption: `docs/assumptions.md` B-1 … B-32.

## 1. Macro calibration (GATE B1)

| Macro | Fit | Held-out |
|---|---|---|
| M4 Chih ISSCC 2021, 22 nm digital CIM | 89 / 52 TOPS/W at 18 / 50 % toggle | −12.7 … +10.0 % (toggle curve, 8 b × 8 b) |
| M1 Jia JSSC 2020, 65 nm charge-domain, 8-b SAR | Table I energies at 1.2 V | 0.85 V: array +5.5 %, ADC −0.2 %, 383 vs 400 1b-TOPS/W (−4.2 %) |
| M2 Sinangil JSSC 2021, 7 nm, 4-b flash | 5.9 / 7.8 pJ at codes 6 / 15, 0.8 V | codes 1–12 ≤ +6.5 %, VDD 0.65–1.0 V ≤ −7.0 % |

Exceptions (stated, not fitted): M2 code 0 (+40 % energy, −27.6 % TOPS/W) and M2 Table I at 1.0 V
(−31 / −35 %), which disagrees with Fig. 13(b) of the same paper. Area is an input for M2 and M4 and not
published for M1, so area is not independently validated. Moved to 22 nm / 0.8 V, the three macros give
25.0 / 18.1 / 27.7 fJ per 4 b × 4 b MAC (`results/b1_gate/summary.md`).

## 2. Result system

Analog arrays: the M1 macro at 22 nm / 0.8 V with the simulator's operating point (bit-serial W8A8, nrow 128,
6-b ADC): one conversion 567 fJ = array 164 + ADC 357 + shift-add and register 46. Array write 90 fJ/bit. DRAM
3.97 pJ/bit (HBM2, the lowest cited value), 12.8 GB/s. Weights that do not fit on chip are read from DRAM and
written into working arrays once per decoded token, or once per prompt in prefill. LM head digital, read from
DRAM per decoded token.

## 3. Energy per token (sequence length 512, KV in an SRAM buffer, minimum on-chip arrays)

| Model | Compute option | PPL (8,192 tokens) | Compute (mJ) | Prefill total (mJ) | Decode total (mJ) |
|---|---|---|---|---|---|
| Qwen2.5-1.5B | digital INT8 | 10.338 | 0.769 | 0.870 | 50.46 |
| | DCIM (M4) | 10.338 | 0.187 | 0.290 | 50.19 |
| | analog 128 / 6 b, M1 ADC | 10.342 | 0.372 | 0.474 | 50.37 |
| | analog 128 / 6 b, best-SAR ADC | 10.342 | 0.154 | 0.257 | 50.16 |
| | analog 64 / 5 b, M1 ADC | 10.356 | 0.545 | 0.647 | 50.55 |
| | analog 128 / 6 b + Hadamard, M1 ADC | 10.229 | 0.372 | 0.474 | 50.37 |
| Qwen2.5-0.5B | digital INT8 | 14.821 | 0.210 | 0.243 | 16.09 |
| | DCIM (M4) | 14.821 | 0.051 | 0.085 | 16.02 |
| | analog 128 / 6 b, M1 ADC | 14.809 | 0.102 | 0.135 | 16.07 |
| | analog 128 / 6 b, best-SAR ADC | 14.809 | 0.042 | 0.076 | 16.01 |

Source: `results/b4_sweep/summary.md`, figure `figures/b4_energy.png`; per-component tables for sequence
lengths 128 / 512 / 2048, both KV placements and four on-chip budgets in `results/b3_ppa_qwen15b/summary.md`
and `results/b3_ppa_qwen05b/summary.md`.

## 4. Findings

1. **Decode energy is weight movement.** On Qwen2.5-1.5B a decoded token costs 50.4 mJ: 41.6 mJ DRAM read of
   the linear weights, 7.4 mJ LM-head weights from DRAM, 0.95 mJ array writes, 0.37 mJ compute. ADC conversions
   are 0.23 mJ (0.5 %). The compute macro (digital, DCIM, analog, any ADC) changes the decode total by less
   than 1 %. With GDDR5-class DRAM (14 pJ/bit) the total is 174 mJ.
2. **Prefill energy is compute.** The same weights are shared by all prompt tokens, so at 512 tokens a prefill
   token costs 0.26–0.87 mJ and the compute macro decides it: analog with the M1 ADC is 2.1× below digital INT8
   and 2.0× above DCIM; with a best-reported SAR ADC analog is 0.82× DCIM.
3. **At nrow 128 the ADC is 63 % of analog compute energy** with the M1-class ADC. Halving nrow to use a 5-b
   ADC (64 / 5 b) raises compute energy by 47 % with that ADC: the number of conversions doubles and the SAR
   energy falls by less than half. Part A showed both points are lossless.
4. **Keeping all weights on chip is not an edge option at 22 nm:** 10.5 Gbit of cells is 2,161 mm² (cell area
   only). 2 Gbit on chip (412 mm²) lowers decode by 16 %. The minimum (one layer of working arrays) is
   110 Mbit, 23 mm².
5. **Decode rate is bounded by DRAM bandwidth:** 8.3 tokens/s on one 12.8 GB/s channel for Qwen2.5-1.5B,
   26 tokens/s for Qwen2.5-0.5B.
6. **Attention, the KV cache and the non-linear units are below 0.1 % of a decoded token** up to 2,048 tokens;
   KV in CIM arrays against an SRAM buffer changes the total by less than 0.05 mJ.
7. **Hadamard rotation costs nothing measurable** (3 M adds per token) and improves perplexity by 1 %.

## 5. Limits

- The 22 nm analog numbers rest on a 65 nm macro and the classical scaling rule (energy ∝ node · VDD²).
- DRAM energy is the HBM2 value; no LPDDR5 energy per bit was found. A higher value strengthens finding 1.
- The LM head is assumed off chip and digital; an on-chip copy would remove 7.4 mJ per decoded token.
- Area is bit-cell area; ADC and periphery area of the M1 anchor is not published.
- Latency is a bound (DRAM stream time plus array evaluations); digital units and scheduling are not timed.
- DCIM uses an input toggle rate measured on OPT-125M and an 8 b × 8 b point outside the fitted data.
- The digital baseline and the SRAM buffer use an 8 KB SRAM read energy for much larger memories.
- Tool comparison (B5): AccelForge's CiMLoop model of the Jia macro agrees with our M1 model within 4.5 % (array, ADC, TOPS/W, 8 b × 8 b MAC energy); our M1 model leaves out Jia's near-memory data path (`validation/tool_comparison.md`).
- Not done: M3 calibration, NeuroSim comparison, hybrid policy per layer role, array-size sweeps,
  pipelined / single-slope ADCs.
