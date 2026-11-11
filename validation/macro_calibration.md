# Macro calibration (GATE B1)

Calibration macros for the Part B cost model, their reported numbers, and the conditions under which each
number was measured. GATE B1: the model reproduces each macro's TOPS/W and area within the stated error
(target ≤ 20–30 %). The error table (section 3) stays empty until `cimsim/ppa/` exists.

Status: 2026-11-11. Headline numbers checked against the papers' abstracts and publisher/author pages
(sources per row). Full-text conditions (figure numbers, data patterns, the voltage of the headline number
where the abstract does not give it) are still open; listed per macro under "Open".

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

Open:
- ADC type and bits (plan table says 8 b; abstract only says "area- and energy-efficient analog-to-digital
  conversion").
- Whether "1b-TOPS" counts 2 ops per 1-b MAC.
- What "MAC hardware" includes, and why the preprint (152 / 297) is lower than the JSSC (192 / 400). Possibly a
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

Derived check: 1024 MAC × 2 ops / 5.5 ns = 372.4 GOPS. The throughput is the 0.8 V number and uses 1 MAC =
2 ops.

Open:
- Supply voltage of the 351 TOPS/W point. It may not be 0.8 V; efficiency is usually quoted at a lower
  voltage than throughput.
- Data pattern used for the 351 TOPS/W measurement. One citing paper calls it an "average" efficiency at INT4.
- Macro area. Not in the abstract, so TOPS/mm² cannot be computed yet.
- Array organisation (the plan table's 64 × 64 is unverified).

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

Open:
- Precision and supply voltage of the 89 TOPS/W and 16.3 TOPS/mm² points (likely 4 b × 4 b at different
  voltages; unverified).
- Input toggle rate used for the headline number.
- Macro capacity (kb).

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
| M1 | | | | | | | |
| M2 | | | | | | | |
| M3 | | | | | | | |
| M4 | | | | | | | |

## 4. What is needed from the full texts

The open items above need the paper PDFs, which the sandbox cannot fetch: IEEE Xplore is paywalled, and page
fetches currently need approval. With Thomas's university access, the cheapest route is one PDF per macro
saved to `D:\japan\hf_local\papers\`; the agent then reads the comparison table and the efficiency-vs-voltage
figure of each.
