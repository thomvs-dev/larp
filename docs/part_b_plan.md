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
against the original papers before use** (status column).

| ID | Paper | Node | Type | Precision | Array | ADC | Status |
|---|---|---|---|---|---|---|---|
| M1 | Jia et al., "A programmable heterogeneous microprocessor based on bit-scalable in-memory computing", IEEE JSSC 2020 | 65 nm | analog SRAM, charge-domain, bit-serial input | 1–8 b in / 1–8 b w | 768×768 | 8 b | to verify from paper |
| M2 | Sinangil et al., "A 7-nm compute-in-memory SRAM macro supporting multi-bit input, weight and output and achieving 351 TOPS/W and 372.4 GOPS", IEEE JSSC 2021 | 7 nm | analog SRAM | 4 b / 4 b | 64×64 | 4 b | 351 TOPS/W from title; conditions to verify |
| M3 | Wang et al., "A charge domain SRAM compute-in-memory macro with C-2C ladder-based 8-bit MAC unit …", IEEE JSSC 2023 (VLSI 2022 companion) | 22 nm | analog SRAM, charge-domain | 8 b / 8 b | 512×128 (64×128 active) | 8 b | to verify |
| M4 | Chih et al., "An 89TOPS/W and 16.3TOPS/mm² all-digital SRAM-based full-precision compute-in-memory macro in 22nm for machine-learning edge applications", ISSCC 2021 | 22 nm | digital SRAM (DCIM, adder tree) | 4–8 b | 256 inputs, 16 slices | none | 89 TOPS/W, 16.3 TOPS/mm² from title; conditions to verify |

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

## 4. Open questions for Thomas

- Process node for the Qwen2.5-1.5B results (22 nm and 28 nm have the most published SRAM CIM data).
- Whether the result system is "all analog CIM", "hybrid (sensitive layers digital)", or both.
- Target: macro-level TOPS/W only, or system-level (incl. DRAM and digital units) — system-level is the
  honest LLM number and is usually several times lower than macro TOPS/W.
