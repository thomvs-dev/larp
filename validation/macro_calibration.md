# Macro calibration (GATE B1)

Calibration macros for the Part B cost model, their reported numbers, and the conditions under which each
number was measured. GATE B1: the model reproduces each macro's TOPS/W and area within the stated error
(target ≤ 20–30 %). The error table (section 3) stays empty until `cimsim/ppa/` exists.

Status: 2026-11-15. **M1, M2 and M4 are calibrated** (section 3; `results/b1_m1_jia/`, `results/b1_m2_sinangil/`,
`results/b1_m4_chih/`, summary `results/b1_gate/`). M3 still needs the right PDF. History: headline numbers first checked against abstracts and publisher/author pages, then against
the full texts of M1, M2 and M4. The PDF supplied for M3 is a
different paper (see M3).

Convention used in this file: **1 MAC = 2 ops**, unless a row says otherwise. TOPS/W and TOPS/mm² are quoted
exactly as the paper states them; the "8b-equivalent" values are derived, not reported.

## 1. Reported numbers

### M1 — Jia et al., IEEE JSSC 55(9):2609–2621, 2020 (doi 10.1109/JSSC.2020.2987714)

"A Programmable Heterogeneous Microprocessor Based on Bit-Scalable In-Memory Computing" (Princeton).

| Item | Value | Source |
|---|---|---|
| Node | 65 nm CMOS | JSSC abstract |
| IMC accelerator | 590 kb (= 768 × 768 bit cells) | JSSC abstract (590 kb); 768 × 768 from the plan table, consistent (589,824 b) |
| Precision | 1–8 b inputs and weights, bit-parallel weights / bit-serial inputs (BP/BS) | JSSC abstract, arXiv 1811.04047 |
| Compute | charge-domain (capacitor-based) | JSSC abstract |
| Energy efficiency | 192 / 400 **1b**-TOPS/W, "for MAC hardware" | JSSC abstract |
| Area efficiency | 0.60 / 0.24 1b-TOPS/mm², MAC hardware | JSSC abstract |
| Supply | 1.2 / 0.85 V (the two numbers above, in that order) | JSSC abstract |
| Scaling | both "scale directly with the bit precision of the input vector and matrix elements" | JSSC abstract |
| CIMU area | 8.5 mm² | Hot Chips 2019 slides (HC31, Jia) |
| Earlier preprint | 152 / 297 1b-TOPS/W, 4.7 / 1.9 1b-TOPS at 1.2 / 0.85 V | arXiv 1811.04047 (VLSI 2019 version) |

Derived (not reported): 8 b × 8 b equivalent = 1b figure / 64 → 3.0 / 6.25 TOPS/W, 0.0094 / 0.0038 TOPS/mm².

Full text (JSSC 2020): column ADC is an 8-b SAR; ADC precision chosen for N = 2304 rows; the 8-b ADC costs
15 % area and 18 % energy overhead; M-BC bit cell 1.8 µm² (80 % over a logic-rule 6T).

Table I (1.2 V | 0.85 V, 100 | 40 MHz), energy per CIMA column and evaluation: CIMA 20.4 | 9.7 pJ, ADC 3.56 | 1.79 pJ,
ABN 9.78 | 4.92 pJ; NMC data path 14.7 | 8.3 pJ per output; reshaping buffer 35 | 12 pJ per 32-b input; CIMA load
19.2 | 9.6 pJ per 32-b word. Fig. 12 (1.2 V, per 1-b MAC): reshaping 0.47 fJ, ADC 1.55 fJ, CIMA 8.85 fJ, total 10.87 fJ;
CIMA loading 600 fJ per bit = 55.2× one 1-b MAC. Sec. III-B: input broadcast and charging of the cell capacitors are
about 66 % of the CIMA energy and scale with input sparsity. SAR unit capacitor 1.2 fF.

Resolved 2026-11-15:
- "1b-TOPS" counts 2 ops per 1-b MAC, and "MAC hardware" is the array plus the ADC: 2 · 2304 / (20.4 + 3.56 pJ) =
  192.3 and 2 · 2304 / (9.7 + 1.79 pJ) = 401, the abstract's 192 | 400.

Open:
- Why the preprint (152 / 297) is lower than the JSSC (192 / 400). Possibly a
  different scope (the whole accelerator vs the MAC hardware only).
- Hot Chips slide gives 0.26 / 0.10 (1b-)throughput per mm² for the 8.5 mm² CIMU. The JSSC abstract gives
  0.60 / 0.24 for MAC hardware. The ratio of 0.43 would mean MAC hardware ≈ 43 % of CIMU area. Not checked.
- Input / weight bit density of the measurement (CiMLoop reports 2.3× density dependence on one macro).

### M2 — Sinangil et al., IEEE JSSC 56(1):188–198, 2021 (doi 10.1109/JSSC.2020.3031290)

"A 7-nm Compute-in-Memory SRAM Macro Supporting Multi-Bit Input, Weight and Output and Achieving 351 TOPS/W and
372.4 GOPS" (TSMC). ISSCC 2020 companion: Dong et al., paper 15.3.

| Item | Value | Source |
|---|---|---|
| Node | 7 nm FinFET | JSSC abstract |
| Bit cell | foundry 8T, built around a standard two-port compiler macro | JSSC abstract |
| Parallelism | 1024 4 b × 4 b MACs simultaneously | JSSC abstract |
| Input encoding | 4-b input = number of read-wordline pulses | JSSC abstract |
| Weight encoding | 4-b weight by charge sharing among binary-weighted computation caps | JSSC abstract |
| ADC | 4-bit flash; unit computation cap = the sense-amplifier input cap inside the flash ADC | JSSC abstract |
| Access time | 5.5 ns at 0.8 V, room temperature | JSSC abstract |
| Energy efficiency | 351 TOPS/W (4 b × 4 b) | JSSC title / abstract |
| Throughput | 372.4 GOPS | JSSC title / abstract |

Full text (JSSC 2021, Sec. IV, Fig. 11–13, Table I):
- 4-kb unit macro, 4-b inputs / weights / outputs, 0.65–1 V. Macro area 0.0032 mm²: ~25 % original 8T macro,
  ~53 % CIM circuits (ADC, capacitors, row circuitry, reference decap), ~22 % test / reconfiguration.
- **351 TOPS/W at 0.8 V** (5.5 ns, 372.4 GOPS) and **321 TOPS/W at 1.0 V** (4.5 ns, 455.1 GOPS). The paper
  states the ops convention: 1024 MACs/cycle × 2 ops/MAC.
- Energy grows with the MAC result (more RWL pulses, deeper RBL discharge; Fig. 13a).
- Derived: 372.4 GOPS / 0.0032 mm² = 116 TOPS/mm² at 0.8 V (whole macro incl. test circuits).

Fig. 11 / Fig. 13 / Table I, read 2026-11-15:
- Fig. 13(a), 0.8 V, 5.5 ns: macro energy per cycle vs MAC result (4-b code) 3.3 pJ at 0 … 7.8 pJ at 15, "5.9 pJ in
  average" at code 6 (digitised points in `config/macros/m2_sinangil_jssc21.yaml`). 2048 ops / 5.9 pJ = 347 TOPS/W: the
  351 is the average point; the range 262.3–610.5 TOPS/W is codes 15 and 0. Max power 1.42 mW at 0.8 V.
- Fig. 13(b): energy at MAC 15 vs VDD, 5.1 pJ at 0.65 V to 13.1 pJ at 1.0 V.
- Table I at 1.0 V: 189.3–435.5 TOPS/W, 321 average. This does not match Fig. 13(b): 13.1 pJ at 1.0 V is
  156 TOPS/W, below the table's lowest value. Bit cell 0.053 µm².

Open:
- Which input / weight values produce each MAC result in Fig. 13(a) (the model assumes pulses per row = code, weights
  at maximum).
- The Table I vs Fig. 13(b) disagreement at 1.0 V.

### M3 — Wang et al., IEEE JSSC 2023 (Intel Labs; VLSI 2022 companion pp. 36–37, doi 10.1109/VLSITechnologyandCir46769.2022.9830322)

"A Charge Domain SRAM Compute-in-Memory Macro With C-2C Ladder-Based 8-Bit MAC Unit in 22-nm FinFET Process for
Edge Inference".

| Item | Value | Source |
|---|---|---|
| Node | Intel 22FFL (22 nm FinFET) | JSSC abstract |
| Bit cell | 9T SRAM with local weight multiplexing | JSSC abstract |
| Compute | C-2C (1:2 ratioed) capacitor ladder, charge domain, passive | JSSC abstract |
| Precision | 8 b input activation, 8 b weight | JSSC abstract |
| MVM per cycle | 8-b 64 × 16 MVM in one clock cycle; 64 kb SRAM over eight sub-banks | JSSC text (pp. 1039–1049, via document preview) |
| Peak parallelism | "a maximum of 2k MACs in one clock cycle" | JSSC abstract |
| Clock / latency | one cycle, about 5 ns at 200 MHz | JSSC text |
| Energy efficiency | 32.2 TOPS/W **peak**, at 0.7 V, no sparsity | JSSC abstract (peak) and text (0.7 V, "no sparsity") |
| Area efficiency | 4.0 TOPS/mm² peak, at 1.1 V | JSSC abstract and text |
| Chip area | 1.5 mm² (whole test chip) | JSSC text |
| Accuracy | MVM error < 0.5 %; < 1 % deviation over PVT | JSSC abstract |

Derived check: 64 × 16 = 1,024 MACs per cycle, which is half of the abstract's 2k. Either two cores run in
parallel or the 2k counts something else. Open.

**PDF problem (2026-11-11):** the file supplied as M3 is a different paper, "A Charge Domain SRAM
Computing-in-Memory Macro With Quantized Interval-Optimized ADC and Input Bit-Level Sparsity-Optimized P2O-DAC for
8-b MAC Operation" (TSMC 40 nm, 16 kb, 0.22 mm², 16.8 TOPS/W from **post-layout simulation**). It has no silicon
measurement, so it is not suitable as a calibration macro. The Wang et al. C-2C PDF is still needed.

Open:
- Analog core / macro area. The 1.5 mm² is the whole chip; the 4.0 TOPS/mm² figure implies a smaller area.
- Clock frequency at 1.1 V.
- ADC type and bits.
- What "no sparsity" means for the input data (random? all ones?).

### M4 — Chih et al., ISSCC 2021, paper 16.4, pp. 252–254 (doi 10.1109/ISSCC42613.2021.9365766)

"An 89TOPS/W and 16.3TOPS/mm² All-Digital SRAM-Based Full-Precision Compute-In Memory Macro in 22nm for
Machine-Learning Edge Applications" (TSMC).

| Item | Value | Source |
|---|---|---|
| Node | 22 nm logic | title; TSMC research page (ISSCC 2021 list) |
| Type | all-digital SRAM CIM, full precision (no accuracy loss) | title; TSMC page |
| Precision | inputs 1–8 b; weights 4 / 8 / 12 / 16 b; unsigned or two's complement | TSMC page |
| Organisation | 256 inputs, 16 slices of 4-bit weight nibbles | SemiWiki summary of the paper (Dillinger, 2021-03-15) |
| Energy efficiency | 89 TOPS/W | title |
| Area efficiency | 16.3 TOPS/mm² | title |
| FOM | TOPS/W × TOPS/mm² = 1450 | TSMC page |
| Data dependence | TOPS/W depends on input toggle rate; the paper's plot assumes 50 % ones in the weights | SemiWiki summary |

Full text (ISSCC 2021, text and Fig. 16.4.5–16.4.7):
- 6T bit cell 0.379 µm² + 4T NOR per weight bit; 256 inputs × 64 columns × 4-b weights = 64 kb; macro size
  **0.202 mm²**. Per column: 256 4-b weights, 256 × 4 bit-wise multipliers, adder tree, partial-sum accumulator.
- Bit-serial inputs, MSB first; 4-b input needs 4 + 1 cycles (one extra cycle for the accumulator).
  Cycle time 10 ns (4b/4b) → 2 ns per input-bit cycle; 18 ns (8b/8b, estimated).
- **89 TOPS/W at 0.72 V, 4b/4b, "sparse pattern": 18 % input toggle rate, 50 % ones in weights; 52 TOPS/W
  for the "dense pattern": 50 % toggle rate**, 25 °C, measured. 3.3 TOPS at 0.72 V → 16.3 TOPS/mm².
- 8b/8b (authors' estimate): 917 GOPS, 24.7 TOPS/W. Simulated 97 TOPS/W at 0.68 V.
- Energy depends on weight sparsity and input toggle rate and comes mainly from the first adder-tree stages.
- Fig. 16.4.5: TOPS/W vs VDD (0.52–0.92 V) for 6.25 / 12.5 / 25 / 50 % toggle rate.

## 2. Normalisation rules (applied when the model is compared with a macro)

1. Compare at the paper's own precision, voltage and data pattern. Never compare against a derived 8b-equivalent.
2. Ops: 1 MAC = 2 ops. M2's throughput confirms this convention. Check each macro.
3. Peak vs average: M3 is peak; M2 may be average; M1 and M4 are unclear. The model reports both. The
   comparison uses the matching one.
4. Bit density: the simulator's MAC histograms give input/weight bit densities. For the calibration runs, use
   the density of the paper's test pattern if stated, otherwise random 50 %.
5. Scope: macro only (array + ADC + shift-add + drivers). No IO, no off-macro buffers, unless the paper's
   number includes them.
6. Area: compare against the area the paper uses for its TOPS/mm² (core/macro), not the die area.

## 3. Error table (filled at GATE B1)

| Macro | Point (precision, VDD, data) | Reported TOPS/W | Model TOPS/W | Error | Reported area | Model area | Error |
|---|---|---|---|---|---|---|---|
| M1 | 1b/1b, 1.2 V, array + ADC | 192 (1b) | 192.3 (fit: Table I energies) | +0.2 % | not given for the MAC hardware | — | not checked |
| M1 | 1b/1b, 0.85 V | 400 (1b) | 383.3 | −4.2 % | | | |
| M1 | array / ADC energy per column, 0.85 V | 9.7 / 1.79 pJ | 10.24 / 1.786 pJ | +5.5 / −0.2 % | | | |
| M2 | 4b/4b, 0.8 V, average (code 6) | 351 | 347.1 (fit: 5.9 pJ) | −1.1 % | 0.0032 mm² | 0.0032 (input) | 0 % |
| M2 | 4b/4b, 0.8 V, lowest / highest (codes 15 / 0) | 262.3 / 610.5 | 262.6 (fit) / 442.0 | +0.1 / −27.6 % | | | |
| M2 | energy vs MAC code 1–12, 0.8 V (Fig. 13a, digitised ±0.1 pJ) | 4.55 … 7.05 pJ | 4.84 … 7.17 pJ | +0.1 … +6.5 % | | | |
| M2 | energy at code 15, 0.65–1.0 V (Fig. 13b) | 5.1 … 13.1 pJ | 5.15 … 12.19 pJ | +1.2 … −7.0 % | | | |
| M2 | 4b/4b, 1.0 V, lowest / average / highest (Table I) | 189.3 / 321 / 435.5 | 168.0 / 222.2 / 282.9 | −11.2 / **−30.8 / −35.0 %** | | | |
| M3 | | | | | | | |
| M4 | 4b/4b, 0.72 V, 18 % toggle | 89 | 89.0 (fit) | 0 % | 0.202 mm² | 0.202 (input) | 0 % |
| M4 | 4b/4b, 0.72 V, 50 % toggle | 52 | 52.0 (fit) | 0 % | | | |
| M4 | 8b/8b, 0.72 V, 18 % toggle (paper estimate) | 24.7 | 22.3 | −9.9 % | | | |
| M4 | 4b/4b, 0.72 V, 6.25 / 12.5 / 25 % toggle (Fig. 16.4.5, digitised ±5) | 138 / 95 / 70 | 120.5 / 101.4 / 77.0 | −12.7 / +6.7 / +10.0 % | | | |

M4 model (`cimsim/circuits/dcim.py`, `config/macros/m4_chih_isscc21.yaml`, `validation/b1_m4_dcim.py`, results in
`results/b1_m4_chih/`): energy per 1b × 1b product e = e0 + e1 · toggle, fitted to the two measured points:
e0 = 0.842 fJ, e1 = 3.123 fJ (0.72 V, 22 nm, 50 % weight ones). Area per weight bit 3.08 µm² (macro area / 64 kb).
Throughput 2 · 256 · (64 / ⌈w_bits / 4⌉) / ((x_bits + 1) · 2 ns): 3277 GOPS at 4b/4b (paper 3300), 910 at 8b/8b
(paper estimate 917). Area and throughput only check internal consistency (macro area is an input); the
independent checks are the 8b/8b estimate and the toggle-rate curve, all within ±13 %. The fitted line is too
flat at very low toggle rates (−12.7 % at 6.25 %), which matters for post-ReLU layers (OPT fc2 toggles 2.5 %,
`results/b1_toggle_opt125m_sandbox/`).

M2 model (`config/macros/m2_sinangil_jssc21.yaml`, `validation/b1_m2_acim.py`, `results/b1_m2_sinangil/`): one
conversion = one 4-b output = 4 bit columns × 64 rows = 256 cells = 64 MACs; macro energy = 16 conversions,
E = a + b · code, fitted to the two labelled points of Fig. 13(a) (5.9 pJ at code 6, 7.8 pJ at code 15). Per
conversion at 0.8 V: data-independent 289.6 fJ (charged to the 15 flash comparators, 19.3 fJ each), wordline 12.0 aJ
per cell and pulse (the paper's 5 % RWL share at the average point), bit line 0.593 fJ per cell at full activity.
Held-out: codes 1–12 within +6.5 %, the VDD sweep of Fig. 13(b) within −7.0 % with the VDD² rule. Outside or at the
limit: code 0 (energy +40 %, TOPS/W −27.6 %; the measured curve steps up between codes 0 and 1) and the Table I
values at 1.0 V (−31 / −35 % for average / highest), which disagree with the paper's own Fig. 13(b).

M1 model (`config/macros/m1_jia_jssc20.yaml`, `validation/b1_m1_acim.py`, `results/b1_m1_jia/`): one conversion = one
column of 2304 cells for one input bit. Array energy 20.4 pJ at 1.2 V split 66 % data-dependent (11.69 fJ per active
cell at an assumed input activity of 0.5) and 34 % data-independent (3.01 fJ per cell). ADC: 8-b SAR, DAC 589 fJ
(1.2 fF unit, conventional switching) + 371 fJ per bit cycle (fitted remainder of 3.56 pJ). Held-out: the 0.85 V
energies and 400 1b-TOPS/W with the VDD² rule, within +5.5 %. Area and throughput not checked (no macro-only area or
evaluation time in the paper).

Cross-node plausibility (`validation/b1_gate.py`, `results/b1_gate/summary.md`): energy per 4b × 4b MAC moved to
22 nm / 0.8 V with E ∝ node · VDD²: M1 25.0 fJ, M2 18.1 fJ, M4 27.7 fJ (spread 1.53×). Different architectures,
output precisions and data patterns, so this bounds the scaling rule's error only loosely.

GATE B1 status (2026-11-15): M4 and M1 within 13 %; M2 within 7 % in its operating range at 0.8 V and over VDD in
Fig. 13(b), with three points over 30 % (above). Area: M2 and M4 areas are inputs, M1 not available — the gate's
area half is not independently tested on any macro. Passed with these exceptions documented (2026-11-15, `docs/decisions.md`).

## 4. What is needed from the full texts

The open items above need the full paper texts (IEEE Xplore). For each macro the comparison table and the
efficiency-vs-voltage figure are the parts used.
