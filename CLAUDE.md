# SRAM CIM Simulator for LLMs — Project Plan

Owner: Thomas
Folder: `D:\japan`
Status: **Part A — A1 core done (GATE A1 passed); A0 baselines waiting on runs on Thomas's machine**
Last updated: 2026-11-02

---

## 0. Instructions for any agent working in this folder

Read this before doing anything else in a new session.

1. Read this whole file. Section 6 (Status) and Section 7 (Progress log) say where work stopped. For Part A design details (maths, config fields, tests, experiment IDs E0–E7) read `docs/part_a_plan.md`. Run `pytest -q` before and after changing core code.
2. Work on the **next unchecked task in the current milestone**. Do not skip ahead, and do not start Part B before Part A's exit criteria are met, unless Thomas says otherwise.
3. Validation gates (marked **GATE**) must pass before the next milestone starts. If one fails, fix it or report why; do not work around it.
4. When you finish a task:
   - tick its checkbox here,
   - add a dated line to Section 7 (what was done, where the code/results are),
   - update the Status line at the top and Section 6.
5. Record any design decision that changes the plan in Section 8 (Decisions log), with the reason.
6. Do not reorder milestones, drop tasks, or change end goals without asking Thomas. Adding sub-tasks is fine.
7. Every experiment saves its config, git commit hash, seed and raw outputs under `results/<experiment_name>/`. No numbers in the log without a results path.
8. Use plain, functional language in code comments, docs and reports.

---

## 1. Goal

Build an open-source simulator for **SRAM-based compute-in-memory (CIM) running LLM / transformer inference** that reports, from one run:

- model quality under analog non-idealities (perplexity, zero-shot accuracy), and
- hardware cost (energy, latency, area), split by prefill and decode.

### Why this is needed (gap in existing tools)

| Tool | Covers | Missing for LLMs |
|---|---|---|
| ASiM (Keio) | Bit-accurate SRAM charge-domain ACiM accuracy, PyTorch | No PPA; noise sigma is user input, not circuit-derived; no decoder-LLM features |
| DNN+NeuroSim V1.5 (Georgia Tech) | Accuracy + PPA, many devices, transformers via TensorRT PTQ | Generic macro model, slow, no KV cache / prefill-decode modelling |
| CiMLoop (MIT) | Fast data-value-dependent energy, mapping search | No accuracy simulation |
| MICSim, EdgeCIM, accelerator papers | Transformer/SLM PPA estimates | No SRAM analog accuracy; one-off evaluation setups |

Nothing open couples SRAM ACiM accuracy on decoder LLMs with KV-cache-aware system cost. That is this project.

### Scope

- Target: **edge small language models**, ≤ ~1.5B parameters for results (OPT-125M / GPT-2 for development; TinyLlama-1.1B, Qwen-0.5B/1.5B for results).
- Macro types: analog charge-domain SRAM CIM first; digital SRAM CIM (DCIM) and hybrid as configurable alternatives.
- Out of scope for now: training on CIM, RRAM/PCM/FeFET, models ≥ 7B in bit-accurate mode.

---

## 2. Core design rules (apply to all milestones)

1. **Event counting is built into the executor from day one.** Every forward pass records ADC conversions, array reads, array writes (including KV writes), and digital ops. Part B multiplies these counts by per-action costs; it must not need a second simulation.
2. **Two simulation modes:**
   - *Bit-accurate*: bit-sliced weights, bit-serial / bit-parallel inputs, per-`nrow` partial sums, ADC quantization, noise injection. Used on a few layers and short sequences.
   - *Statistical (fast)*: error distribution measured in bit-accurate mode, injected into ordinary matmuls. Used for full perplexity runs. It must be calibrated against bit-accurate mode (see A3).
3. **Ideal config must be exact.** With infinite ADC bits and zero noise, a CIM layer must match the integer matmul bit-for-bit. Keep this as a permanent unit test.
4. **Everything is config-driven** (YAML): macro spec, mapping, noise, workload. No hard-coded hardware parameters in model code.
5. **Every assumption about a published macro is written down** in `docs/assumptions.md` with its source.

### Planned repository layout

```
config/        YAML macro, noise, mapping and experiment specs
cimsim/
  core/        bit-plane MAC executor, CIMLinear, CIMMatMul (dynamic operands), modes
  noise/       noise models (Gaussian, nonlinear, later circuit-derived)
  events/      event counter collected during forward pass
  models/      layer swapping for HuggingFace models, per-layer mapping policy
  circuits/    (Part B) per-action energy/latency/area component models
  ppa/         (Part B) event counts -> energy/latency/area
  mapper/      (Part B) layer -> array tiling, KV placement, weight reload
  system/      (Part B) prefill/decode, DRAM, peripheral nonlinear units
eval/          perplexity, lm-eval-harness wrappers
validation/    ASiM reproduction, macro calibration
tests/         pytest
results/       one folder per experiment
docs/          assumptions.md, design notes
```

Stack: Python 3.11, PyTorch, HuggingFace Transformers, lm-eval-harness, pytest, YAML configs.

---

## 3. Part A — LLM accuracy under SRAM ACiM non-idealities

### End goal of Part A

A working, tested tool that swaps Linear (and optionally attention matmul) layers in HuggingFace decoder LLMs for SRAM CIM layers, and a results set showing how ADC precision, row parallelism, analog noise, encoding and activation outliers affect perplexity — including which mitigation works best at equal ADC-conversion cost. Results should be strong enough for a paper / workshop submission.

### Part A exit criteria

- [ ] All A milestones ticked, all GATEs passed.
- [ ] Results for at least 2 development models and 2 result models (Section 1 scope).
- [ ] Reproducible: one command per figure, from config to plot.
- [ ] Short write-up in `docs/part_a_results.md`.

### A0 — Setup and baselines

- [x] Create repo skeleton (layout above), `pyproject.toml`, pytest. (formatter/pre-commit: optional, not set up)
- [ ] Record available hardware (GPU model, VRAM) in Section 9.  ← needs Thomas
- [ ] Perplexity eval on WikiText-2 for OPT-125M and GPT-2 in FP16 — baseline numbers. Script ready: `scripts/run_ppl.py` (commands in README). Must run on Thomas's machine: HuggingFace is blocked from the cloud sandbox.
- [ ] Same for W8A8 integer fake-quant (per-tensor and per-channel), plus W6A6 and W4A8 — configs in `config/experiments/e0_*.yaml`.
- [x] Clone ASiM into `third_party/` (read-only reference), check its licence and note it in Section 8. (Apache 2.0; `third_party/` is git-ignored, clone command in README)

### A1 — Core bit-accurate CIM executor

- [x] `CIMLinear`: weight bit-slicing, input bit-serial and bit-parallel encoding, row grouping by `nrow`.
- [x] ADC model: configurable bits, fixed full-scale range, clipping (ASiM semantics).
- [ ] ADC range option: per-layer calibrated full scale (needed for A5/A6; not in ASiM).
- [x] Noise injection: Gaussian random noise and nonlinearity (ASiM-style `rand_noise_sigma`, `non_linear_sigma`), seeded per layer.
- [x] Signed weight handling documented: two's complement planes (`docs/assumptions.md` #1). Differential / offset schemes: optional, later.
- [x] Event counter: ADC conversions (total / analog), array cycles, MACs, weight bits written.
- [x] Vectorized across bit-planes (one matmul per row chunk); CPU benchmark done. GPU benchmark pending (`scripts/bench_layer.py`).
- [x] **GATE A1:** ideal-config test matches integer matmul exactly; tests pass for random shapes. (95 tests pass)

### A2 — Reproduce ASiM (correctness check)

- [x] Layer-level equivalence with ASiM `ASiMLinear` (zero noise, 7 configs incl. bit-parallel, unsigned): max rel. error ≤ 1e-6 (`tests/test_asim_equivalence.py`). Noise std matches within ~1% (manual check).
- [ ] ResNet-20 / CIFAR-10 with your executor; sweep ADC bits, `nrow`, noise sigma, bit-serial vs bit-parallel. (needs a CIMConv2d — add before this task)
- [ ] Run the same sweeps in ASiM.
- [ ] Comparison table in `validation/asim_repro.md`.
- [ ] **GATE A2:** accuracy trends match ASiM; per-point difference within ~1% absolute (or explained).

### A3 — Statistical fast mode

- [ ] Measure output-error distribution per (layer type, ADC bits, nrow, noise) in bit-accurate mode.
- [ ] Implement fast mode that injects that error into a standard matmul.
- [ ] Event counts still produced in fast mode (computed analytically).
- [ ] **GATE A3:** on OPT-125M, fast-mode perplexity within agreed tolerance (start with ±2% relative) of bit-accurate on a subset of WikiText-2; speed-up recorded.

### A4 — LLM integration

- [ ] Layer swapping for HuggingFace models (OPT, GPT-2, Llama-family, Qwen).
- [ ] Per-layer mapping policy in config: which of Q/K/V/O, FFN up/gate/down run on analog CIM, DCIM, or digital.
- [ ] `CIMMatMul` for dynamic operands (QKᵀ, S·V) with write events counted (KV written into arrays).
- [ ] Eval: WikiText-2 perplexity + small zero-shot set (e.g. ARC-e, PIQA, HellaSwag via lm-eval-harness).
- [ ] Run on TinyLlama-1.1B and Qwen-0.5B in fast mode; record runtime.

### A5 — Sensitivity study

- [ ] Perplexity vs ADC bits and `nrow`, per layer type.
- [ ] Per-layer noise sensitivity (inject noise one layer at a time).
- [ ] Activation outlier statistics per layer (max/std, channel outliers) and correlation with sensitivity.
- [ ] Bit-serial vs bit-parallel on LLMs, compared with ASiM's CNN/ViT findings.

### A6 — Outlier mitigation at equal cost

- [ ] Hybrid: outlier channels in digital, rest analog.
- [ ] SmoothQuant-style scaling and rotation-based (Hadamard) preprocessing.
- [ ] Noise-aware fine-tuning (short, LoRA-based if needed).
- [ ] Compare all at equal ADC-conversion count (from event counter) — Pareto plot: perplexity vs conversions.

### A7 — Circuit-derived noise (stretch, can run in parallel with B0)

- [ ] Model capacitor mismatch (Pelgrom), kT/C, charge injection, comparator offset, ADC INL/DNL.
- [ ] Map circuit parameters (cap size, node, ADC type) → error sigma / distribution.
- [ ] Replace user-supplied sigma with derived values; compare against reported silicon numbers where available.

---

## 4. Part B — System-level energy, latency and area for LLMs

Start only after Part A exit criteria are met (or Thomas says so).

### End goal of Part B

A single run of the simulator outputs **perplexity + energy/token + latency/token (prefill and decode separately) + area** for a configured SRAM CIM system running an edge LLM, with energy/area calibrated against published macros, and design-space sweeps that produce Pareto fronts.

### Part B exit criteria

- [ ] All B milestones ticked, all GATEs passed.
- [ ] Calibration error table against ≥3 published SRAM CIM macros.
- [ ] Coupled DSE results for at least 2 result models.
- [ ] Write-up in `docs/part_b_results.md`; combined A+B tool release.

### B0 — Component library

- [ ] Per-action energy / latency and area for: bitcell read/write, compute bit-line charge/share, SAR and flash ADC (scaling with bits), DAC / wordline driver, shift-add, DCIM adder tree, buffers.
- [ ] Digital peripheral units: softmax, LayerNorm/RMSNorm, SiLU/GELU, RoPE.
- [ ] Technology node scaling; every number cited in `docs/assumptions.md`.

### B1 — Macro model and calibration

- [ ] Event counts × per-action cost → macro energy, latency, area.
- [ ] Select 3–4 published SRAM CIM macros (analog and digital), record their reported specs.
- [ ] **GATE B1:** model reproduces their TOPS/W and area within a stated error (target ≤ 20–30%); error table in `validation/macro_calibration.md`.

### B2 — Mapper

- [ ] Layer → array tiling, weight duplication, array allocation.
- [ ] Policy for weights that do not fit on chip (reload schedule).

### B3 — LLM system model

- [ ] Prefill (GEMM) vs decode (GEMV) phases, reported separately.
- [ ] DRAM bandwidth/energy model for weight reload.
- [ ] KV cache placement options: in CIM arrays (count writes) vs separate buffer; attention dataflow options.
- [ ] Sequence length sweep (128 → 2K+) to show where attention / KV traffic dominates.

### B4 — Coupled runs and DSE

- [ ] One command → perplexity + energy/latency/area per phase.
- [ ] Sweeps: ADC bits, nrow, array size, number of macros, mapping policy (analog / DCIM / hybrid per layer).
- [ ] Pareto fronts: perplexity vs energy/token.

### B5 — Comparison with existing tools

- [ ] Run overlapping cases in NeuroSim V1.5 and/or CiMLoop; compare energy and accuracy.
- [ ] **GATE B5:** differences explained in `validation/tool_comparison.md`.

---

## 5. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Bit-accurate simulation too slow for 1B models | Fast mode (A3), bit-accurate only on subsets; vectorize from A1 |
| Published macros don't report all parameters | Document assumptions; report sensitivity to them |
| Scope creep (DCIM, attention, circuit noise all at once) | Follow milestone order; A7 and DCIM are optional until A exit |
| Results don't match ASiM | GATE A2 blocks progress until explained |
| GPU limits | Development on OPT-125M/GPT-2; larger models only in fast mode |

---

## 6. Status

- Current part: **A**
- Current milestone: **A0 (baselines) → then A1 leftovers → A2**
- Next tasks, in order:
  1. Thomas: set up env (README "Setup"), run `pytest -q`, run E0 baseline commands (README "A0 baselines") for OPT-125M and GPT-2, record GPU in Section 9.
  2. Agent: put E0 numbers in a table in `results/E0_summary.md`; sanity-check FP16 PPL (OPT-125M ≈ 27–28, GPT-2 ≈ 29–30 at seq 1024; investigate if far off).
  3. Agent: GPU benchmark with `scripts/bench_layer.py`; if bit-accurate is too slow, optimise the readout path (float32 where exact).
  4. Agent: per-layer calibrated ADC range option (A1 leftover).
  5. Agent: `CIMConv2d` (needed for A2 ResNet-20), then A2.
- Blockers: HuggingFace downloads blocked in the cloud sandbox — model runs happen on Thomas's machine.

---

## 7. Progress log

Format: `YYYY-MM-DD — what was done — where (code / results path)`

- 2026-11-02 — Plan created (this file). Folder was empty. — `CLAUDE.md`
- 2026-11-02 — Detailed Part A plan written. — `docs/part_a_plan.md`
- 2026-11-02 — A0 skeleton: package `cimsim`, configs, perplexity eval, run script, README, assumptions doc. — `cimsim/`, `eval/`, `scripts/run_ppl.py`, `config/experiments/`
- 2026-11-02 — A1 core: bit-planes, bit-accurate executor (ASiM semantics), CIMLinear (fp / fake_quant / bit_accurate), event counter, HF layer swap (OPT, GPT-2, Llama). 95 tests pass incl. exact ideal match and ASiM layer equivalence (≤1e-6). — `cimsim/core/`, `cimsim/models/swap.py`, `tests/`
- 2026-11-02 — CPU benchmark (2 cores), one 768×3072 layer, 128 tokens, nrow 128: fp 0.004 s, fake_quant 0.004 s, bit_accurate 3.6 s (bit-serial), 1.6 s (enc=3). — `scripts/bench_layer.py`

---

## 8. Decisions log

Format: `YYYY-MM-DD — decision — reason`

- 2026-11-02 — Focus on edge SLMs (≤ ~1.5B) — bit-accurate sim of 7B+ not feasible on student hardware; on-chip SRAM story fits edge.
- 2026-11-02 — Build own core (not a fork), use ASiM as reference — need clean support for event counting, DCIM/hybrid, dynamic matmuls; ASiM reproduction is the correctness check.
- 2026-11-02 — Part A before Part B — SRAM analog accuracy on LLMs is the clearer gap; event counters in A make B mostly bookkeeping.
- 2026-11-02 — ASiM is Apache 2.0: reuse of ideas and code allowed with attribution; kept out of repo (git-ignored `third_party/`).
- 2026-11-02 — bit_accurate mode copies ASiM semantics exactly (incl. full scale = nrow on short chunks, nonlinearity form) — so GATE A2 is like-for-like. Deviations become separate config options, never silent changes.
- 2026-11-02 — Default quantization for LLMs: per-channel weights, per-token activations; ASiM's per-tensor kept as an option — per-tensor is known to hurt LLMs badly (E0 will show by how much).
- 2026-11-02 — LM head excluded from swapping by default (kept digital).

---

## 9. Open questions / environment

- GPU available (model, VRAM): _TBD_
- Process node to assume for PPA: _TBD_
- Target venue / deadline: _TBD_
- ASiM licence terms: Apache 2.0 (checked 2026-11-02)
- Cloud sandbox: CPU only, 2 cores, no HuggingFace access. Model downloads and GPU runs happen on Thomas's machine.

## 10. References

- ASiM — https://arxiv.org/abs/2411.11022 — code: https://github.com/Keio-CSG/ASiM
- NeuroSim V1.5 — https://arxiv.org/abs/2505.02314 — code: https://github.com/neurosim/NeuroSim
- CiMLoop — https://arxiv.org/abs/2405.07259 — code: https://github.com/mit-emze/cimloop
- MICSim — https://arxiv.org/abs/2409.14838
- EdgeCIM — https://arxiv.org/abs/2604.11512
- FusionCIM — https://arxiv.org/abs/2604.25317
- HASTILY — https://arxiv.org/abs/2502.12344
- Ouroboros — https://arxiv.org/abs/2603.02737
