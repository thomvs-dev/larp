# Part B — Detailed Plan: Energy, Latency, Area (TOPS/W) for SRAM CIM running LLMs

Companion to `CLAUDE.md` Section 4. Started 2026-11-03 in parallel with A3/A4 (decision in `CLAUDE.md` Section 8).
Target result: Qwen2.5-1.5B (statistical mode) with energy/token, latency/token (prefill and decode
separately), area, and macro-level TOPS/W, from one run. OPT-125M for development and cross-checks.

Rule for this part: every per-action cost has a source in `docs/assumptions.md` (paper, table/figure,
node, voltage). No energy number is reported unless the model reproduces the calibration macros (GATE B1).

---

## 1. Cost model structure

Energy of one run = Σ over components and actions: (event count) × (energy per action). The event counts
already come from the simulator (`cimsim/events.py`, all modes):

| Event (counter) | Per-action cost needed | Notes |
|---|---|---|
| `array_cycles` (array activations) | wordline/DAC drive for nrow rows, bitline (or compute-line) charge/share per column | scales with nrow, columns, input-digit levels |
| `adc_conversions` | ADC energy per conversion at b bits; latency per conversion | architecture-dependent (SAR / flash / pipelined / single-slope, see A7/B0) |
| shift-add (one per conversion) | digital accumulate energy | scales with output bit width |
| `macs` (digital layers: fp / fake_quant / DCIM) | digital MAC energy (INT8) or DCIM adder-tree energy | for layers mapped off the analog array |
| `weight_bits_written` | SRAM write energy per bit | static weights once; KV cache per token (A4 `CIMMatMul`) |
| non-linear ops (softmax, LayerNorm/RMSNorm, SiLU/GELU, RoPE) | per-element energy of digital units | counted per token from model shapes |
| DRAM traffic (weights that do not fit, B2/B3) | pJ/bit DRAM + interface | only if weights are reloaded |

Latency: array cycles and conversions per macro, ADC sharing ratio (columns per ADC → serialised
conversions), macro count and mapping (B2), prefill = GEMM (many tokens per weight load), decode = GEMV
(one token; weight-stationary arrays mostly idle → utilisation matters).

Data-value dependence: CiMLoop shows energy of analog CIM depends on input/weight bit densities (2.3× on
one macro). The simulator already has the MAC histograms from calibration; B1 decides whether bitline
energy is modelled per toggle (density-dependent) or per activation (constant).

## 2. Calibration macros (GATE B1)

Candidates chosen because CiMLoop modelled and validated them with open models (lets B5 compare directly)
and because they span the architecture choices this simulator models. **All numbers must be checked
against the original papers before use** (status column; details and open items in
`validation/macro_calibration.md`).

| ID | Paper | Node | Type | Precision | Array | ADC | Status |
|---|---|---|---|---|---|---|---|
| M1 | Jia et al., "A programmable heterogeneous microprocessor based on bit-scalable in-memory computing", IEEE JSSC 2020 | 65 nm | analog SRAM, charge-domain, bit-serial input | 1–8 b in / 1–8 b w | 768×768 (590 kb), N = 2304 rows | 8 b SAR | full text checked: 192/400 1b-TOPS/W, 0.60/0.24 1b-TOPS/mm² at 1.2/0.85 V; ADC 15 % area, 18 % energy |
| M2 | Sinangil et al., "A 7-nm compute-in-memory SRAM macro supporting multi-bit input, weight and output and achieving 351 TOPS/W and 372.4 GOPS", IEEE JSSC 2021 | 7 nm | analog SRAM | 4 b / 4 b | 4 kb, 1024 4b×4b MACs/cycle | 4 b flash | full text checked: 351 TOPS/W at 0.8 V, 321 at 1.0 V; 0.0032 mm²; data pattern open |
| M3 | Wang et al., "A charge domain SRAM compute-in-memory macro with C-2C ladder-based 8-bit MAC unit …", IEEE JSSC 2023 (VLSI 2022 companion) | 22 nm | analog SRAM, charge-domain | 8 b / 8 b | 64×16 8-b MVM per cycle, 64 kb, 8 sub-banks | open | abstract checked: 32.2 TOPS/W peak at 0.7 V no sparsity, 4.0 TOPS/mm² at 1.1 V; supplied PDF was a different paper, full text still needed |
| M4 | Chih et al., "An 89TOPS/W and 16.3TOPS/mm² all-digital SRAM-based full-precision compute-in-memory macro in 22nm for machine-learning edge applications", ISSCC 2021 | 22 nm | digital SRAM (DCIM, adder tree) | 4–8 b | 256 × 64 × 4-b (64 kb) | none | **calibrated 2026-11-11** (`results/b1_m4_chih/`): 89 / 52 TOPS/W at 18 / 50 % toggle, 0.72 V, 0.202 mm² |

Source for M1–M3 being CiMLoop's validation set: CiMLoop (arXiv 2405.07259) Section V, Table III; CiMLoop
reports 7 % energy-efficiency error on supply-voltage sweeps and 8 % average area-breakdown error. A
ReRAM macro in that set is out of scope here.

TOPS/W pitfalls to normalise in `validation/macro_calibration.md`: 1 MAC = 2 ops or 1; peak vs average;
input/weight sparsity or bit density used for the measurement; supply voltage; bit precision ("TOPS/W/b"
figures); macro-only vs including periphery/IO. Each calibration point records these.

GATE B1 (from `CLAUDE.md`): model reproduces each macro's reported TOPS/W and area within ≤ 20–30 %,
error table in `validation/macro_calibration.md`.

## 3. Tasks in order

B0 — component library (sandbox, no heavy compute)
1. `cimsim/circuits/`: per-action energy/latency/area models with config fields (node, VDD, nrow,
   columns, ADC type/bits, ADC sharing ratio, DAC levels). Start from published component numbers
   (ADC surveys, macro breakdowns), each cited in `docs/assumptions.md`.
2. Technology scaling rule (node, VDD) with its source; used only to move a calibrated macro to another
   node, never to invent a macro.
3. Digital units: INT8 MAC, adder tree, softmax/norm/activation per-element costs (cited).

B1 — macro model and calibration
4. `cimsim/ppa/`: event counts × costs → energy, latency, area per layer and per run; `ppa_report.json`.
5. Encode M1–M4 as configs (`config/macros/*.yaml`), run their reported workload conditions, error table.
6. GATE B1.

Then B2 (mapper: Qwen2.5-1.5B does not fit in one macro; tiling, macro count, weight reload), B3
(prefill/decode, DRAM, KV cache), B4 (Qwen2.5-1.5B results: PPL vs energy/token Pareto), B5 (CiMLoop /
NeuroSim comparison on M1–M3).

## 4. Decisions (2026-11-11, Thomas: "go with your decision"; deadline tight)

- Process node: **22 nm** (M4 and M3 are 22 nm; M1 65 nm and M2 7 nm bracket it).
- Result system: **all-analog baseline**, hybrid (per-role DCIM) as a sweep axis in B4.
- Headline metric: **system-level energy / token** (macro + digital units + DRAM), prefill and decode
  separately. Macro TOPS/W is reported only for GATE B1.

## 5. Reduced scope (tight deadline)

Order of work, minimum needed for a paper section:

1. Analog array / driver / shift-add cost models; calibrate **M2** (4-b flash, all conditions known).
   With M4 this gives one analog and one digital macro.
2. `cimsim/ppa/`: event counts × costs → `ppa_report.json` per run (energy per layer / phase, latency, area).
3. B2 simplified: analytic tiling (arrays per layer from shapes, nrow, columns), fixed on-chip SRAM budget;
   whatever does not fit streams from DRAM. No mapping search.
4. B3 simplified: one cited DRAM pJ/bit and bandwidth; prefill vs decode from token counts; KV cache in a
   separate SRAM buffer (KV-in-array as an option only if time allows); sequence lengths 128 / 512 / 2048.
5. B4: Qwen2.5-1.5B and one other result model, sweep ADC bits × nrow × policy (all-analog / hybrid),
   PPL vs energy/token. Event counts come from statistical runs that exist or are cheap to rerun.
6. Optional if time: M3 (needs Wang JSSC 2023 PDF), M1, CiMLoop comparison (B5), ADC sharing / pipelining.

GATE B1 under this scope: M4 + M2 within 30 %, plus M3 if the PDF arrives. Fewer than three macros
does not meet the Part B exit criterion (≥ 3); that criterion stays and is reported as not met if M3 / M1
are not done.
