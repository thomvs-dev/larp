# SRAM CIM Simulator for LLMs — Project Plan

Owner: Thomas
Folder: `D:\japan`
Status: **Part A — A0, A1, A2 done (GATE A2 passed). A3: statistical v2 passes 4 of 6 GATE A3 configs; adc7 (+2.3 %) and rn010 (−77.5 %) fail (F9); fix pending Thomas's choice**
Last updated: 2026-11-03

---

## 0. Instructions for any agent working in this folder

Read this before doing anything else in a new session.

1. Read this whole file. Section 6 (Status) and Section 7 (Progress log) say where work stopped. For Part A design details (maths, config fields, tests, experiment IDs E0–E7) read `docs/part_a_plan.md`. Findings that affect design are in `docs/findings.md`. Run `pytest -q` before and after changing core code.
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
9. **There is no GPU.** All runs are on CPU. Follow `docs/cpu_guide.md` (speeds, 8k-token subset protocol for bit-accurate runs). Keep one codebase: CPU optimisations go behind `backend: fast` and must be tested against `backend: reference`; never change `executor.py` semantics for speed.

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
- [x] Record hardware in Section 9: i7-12700H, 16 GB, no GPU.
- [x] Perplexity eval on WikiText-2 for OPT-125M and GPT-2 in FP32 (CPU): OPT-125M 27.655 (published 27.65), GPT-2 29.951. Pipeline validated. (`results/E0_summary.md`)
- [x] Same for W8A8 integer fake-quant (per-tensor and per-channel), plus W6A6 and W4A8. W8A8 per-channel/per-token: +0.29 / +0.20. Per-tensor: +2.8 / +27.5 (outliers). (`results/E0_summary.md`)
- [x] Code on GitHub: https://github.com/thomvs-dev/larp (branch `main`). `D:\japan` must be linked to it so runs record the commit hash ← Thomas (commands in README "Git").
- [x] Clone ASiM into `third_party/` (read-only reference), check its licence and note it in Section 8. (Apache 2.0; `third_party/` is git-ignored, clone command in README)

### A1 — Core bit-accurate CIM executor

- [x] `CIMLinear`: weight bit-slicing, input bit-serial and bit-parallel encoding, row grouping by `nrow`.
- [x] ADC model: configurable bits, fixed full-scale range, clipping (ASiM semantics).
- [x] ADC range option: calibrated full scale per layer / cycle / (cycle, plane), MSE (default) or percentile; `cimsim.calibrate_adc_range`; `run_ppl.py` calibrates on WikiText-2 train tokens. (`cimsim/core/calib.py`, findings F5)
- [x] Asymmetric (zero-point) weight quantization option (`w_quant: asymmetric`): unsigned planes + exact digital zero-point correction, both executors. F2 confirmed: OPT-125M W4A8 asym 37.39 vs published 4-bit RTN 37.28 (symmetric: 51.09). (`results/e0_*_w4a8_asym/`)
- [x] Noise injection: Gaussian random noise and nonlinearity (ASiM-style `rand_noise_sigma`, `non_linear_sigma`), seeded per layer.
- [x] Signed weight handling documented: two's complement planes (`docs/assumptions.md` #1). Differential / offset schemes: optional, later.
- [x] Event counter: ADC conversions (total / analog), array cycles, MACs, weight bits written.
- [x] Vectorized across bit-planes (one matmul per row chunk); CPU benchmark done (`scripts/bench_layer.py`).
- [x] CPU-optimised executor `backend: fast` (`cimsim/core/executor_fast.py`), tested against reference and ASiM. 2.4–3.4× faster noise-free, ~1.5× with noise.
- [x] **GATE A1:** ideal-config test matches integer matmul exactly; tests pass for random shapes. (122 tests pass)

### A2 — Reproduce ASiM (correctness check)

- [x] Layer-level equivalence with ASiM `ASiMLinear` (zero noise, 7 configs incl. bit-parallel, unsigned): max rel. error ≤ 1e-6 (`tests/test_asim_equivalence.py`). Noise std matches within ~1% (manual check).
- [x] `CIMConv2d` (ASiM channel-group row mapping), matches `ASiMConv2d` ≤1e-6 (14 cases, both backends); ASiM conv group-skipping issue found (F4).
- [x] Pretrained CIFAR-10 ResNet-20 (`cimsim/models/cifar_resnet.py`, hash-checked weights) + side-by-side sweep script (`validation/a2_asim_compare.py`, 13 configs, both simulators).
- [x] Smoke run (200 images, 3 configs): fake-quant 92.0/92.0, ADC 8 b 92.0/92.0, ADC 6 b 59.0/60.0 (cimsim/ASiM); cimsim ~9× faster. (`results/a2_smoke/`)
- [x] Full sweep on Thomas's machine (1,000 test images, 13 configs). (`results/a2_resnet20/`)
- [x] Comparison table + discussion in `validation/asim_repro.md`.
- [x] Spread run for gate condition 4: low-ADC configs with `--backend reference` vs `fast`, compared image by image (`validation/a2_spread.py`). (`results/a2_spread_*`)
- [x] **GATE A2:** accuracy trends match ASiM (vs ADC bits, nrow, noise, encoding); per-config difference ≤ 1% absolute at ADC ≥ 8 b and for fake-quant; at ADC ≤ 6 b within the run-to-run spread of either simulator (findings F6); `bs_n64_adc8` difference explained by F4. (Criterion approved by Thomas 2026-11-02.) Passed 2026-11-02: max diff 0.8% at ADC ≥ 8 b; n64 +2.1% = F4; ADC ≤ 6 b diffs ≤ 1.2% vs measured spread ≤ 1.4% (`validation/asim_repro.md`).

### A3 — Statistical fast mode

- [x] A3 design: error model candidates M0–M3, measurement configs, GATE A3 protocol (`docs/part_a_plan.md` §3).
- [x] Error measurement tool: `ErrorProbe` + `scripts/measure_error.py`, runs on OPT-125M for 6 configs. (`results/a3m_opt125m_*/`)
- [x] Measure output-error distribution per (layer type, ADC bits, nrow, noise) in bit-accurate mode; analyse which error model is needed. Result: deterministic gain + input-independent Gaussian noise, both analytic (findings F7, `results/a3m_summary.md`).
- [x] Implement fast mode that injects that error into a standard matmul: `mode: statistical`, analytic model M3 from the calibration MAC histogram (`cimsim/stat/analytic.py`); layer-level agreement with bit-accurate within 5 % in std, gain exact (`tests/test_stat_analytic.py`).
- [x] Event counts still produced in fast mode (computed analytically) — same counts as bit-accurate, tested.
- [x] GATE A3 runs v1: 6 configs with `scripts/a3_gate.py` (Thomas, git 33e49ce). f5: statistical 54–59 vs bit-accurate 30.81 — **fail** (F8). Other configs: see `results/a3_gate_summary.md` once complete.
- [x] Fix for F8: exact sparse saturation term in statistical mode (`stat_saturation: exact`, `cimsim/stat/saturation.py`), calibration column filter (`stat_sat_margin`), `--reuse-ba` in the gate script. Tested against both backends (`tests/test_stat_saturation.py`).
- [x] GATE A3 v1 chain complete (all 6 configs, `results/a3_gate_summary.md`); evaluated with sandbox v2 statistical PPLs (`results/a3_gate_v1_evaluation.md`): f5, adc8, rn008, rn02_hyb6 pass; adc7 (+2.31 %) and rn010 (−77.5 %) fail (F9).
- [ ] Fix F9 (option chosen by Thomas, see Section 6), then GATE A3 reruns with `--reuse-ba` ← Thomas.
- [ ] **GATE A3:** on OPT-125M, fast-mode perplexity within agreed tolerance (start with ±2% relative) of bit-accurate on a subset of WikiText-2; speed-up recorded.

### A4 — LLM integration

- [ ] Layer swapping for HuggingFace models (OPT, GPT-2, Llama-family, Qwen).
- [ ] Per-layer mapping policy in config: which of Q/K/V/O, FFN up/gate/down run on analog CIM, DCIM, or digital.
- [ ] `CIMMatMul` for dynamic operands (QKᵀ, S·V) with write events counted (KV written into arrays).
- [ ] Eval: WikiText-2 perplexity + small zero-shot set (e.g. ARC-e, PIQA, HellaSwag via lm-eval-harness).
- [ ] Run on TinyLlama-1.1B and Qwen-0.5B in statistical mode (CPU); record runtime.

### A5 — Sensitivity study

- [ ] Perplexity vs ADC bits and `nrow`, per layer type.
- [ ] Per-layer noise sensitivity (inject noise one layer at a time).
- [ ] Activation outlier statistics per layer (max/std, channel outliers) and correlation with sensitivity.
- [ ] Bit-serial vs bit-parallel on LLMs, compared with ASiM's CNN/ViT findings.
- [ ] Input encoding study (F3): two's complement vs sign-magnitude vs offset-binary inputs, and skipping sign-extension planes — perplexity vs ADC bits at equal ADC conversions. Needs new encoding options in both executors (reference first, then fast, tested against each other).
- [ ] ADC level alignment (F3): report error vs ADC bits for nrow aligned (2^b − 1 multiple of nrow) and not aligned, separately from ADC precision.
- [ ] ADC range calibration study (F5): full vs MSE-calibrated (layer / cycle_plane) at ADC 4–8 b, nrow 64–256, on OPT-125M and GPT-2 (bit-accurate, 8k tokens), then result models via A3.

### A6 — Outlier mitigation at equal cost

- [ ] Hybrid: outlier channels in digital, rest analog.
- [ ] SmoothQuant-style scaling and rotation-based (Hadamard) preprocessing.
- [ ] Noise-aware fine-tuning (short, LoRA-based if needed).
- [ ] Compare all at equal ADC-conversion count (from event counter) — Pareto plot: perplexity vs conversions.

### A7 — Circuit-derived noise (stretch, can run in parallel with B0)

- [ ] Model capacitor mismatch (Pelgrom), kT/C, charge injection, comparator offset, ADC INL/DNL.
- [ ] Map circuit parameters (cap size, node, ADC type) → error sigma / distribution.
- [ ] Replace user-supplied sigma with derived values; compare against reported silicon numbers where available.
- [ ] ADC architecture error models: SAR (comparator offset, capacitor-DAC mismatch → DNL), flash (per-comparator offsets), pipelined (inter-stage gain / offset errors → large errors at stage-transition codes), single-slope (ramp nonlinearity). Applied after the MAC, configurable per layer; compare perplexity sensitivity to error *type* at equal error magnitude (errors on high-significance planes are amplified up to 2^14, F3/F5).

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
- [ ] Per-architecture ADC cost: energy / area / latency per conversion for SAR, flash, pipelined, single-slope at 4–8 b, plus ADC sharing ratio (columns per ADC) as a config field.

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
- [ ] Conversion scheduling: ADC sharing (serialised conversions), pipelined ADC latency, and overlap of array compute with conversion (macro-level pipelining) in the latency model.

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
| No GPU (CPU only) | `backend: fast`; bit-accurate only on OPT-125M/GPT-2, first 8,192 tokens; result models only in statistical mode; see `docs/cpu_guide.md` |

---

## 6. Status

- Current part: **A**
- Current milestone: **A3** (statistical fast mode)
- Next tasks, in order:
  1. Thomas: choose the F9 fix (options in the 2026-11-03 report: exact sparse deterministic term for full-range rounding above MAC 64; discrete code-flip sampling for rare noise flips).
  2. Agent: implement, test against bit-accurate at layer level and in the sandbox (OPT-125M available), push.
  3. Thomas: pull, `pytest -q`, rerun all six configs with `--reuse-ba` (no bit-accurate reruns; ~10–15 min each, mostly calibration), `--table`.
  4. Agent: GATE A3 verdict, then A4.
  5. Note: bit-accurate numbers at coarse ADC carry float-rounding spread (F6); compare statistical against bit-accurate on the same machine (fake-quant differs between Thomas's machine and the sandbox by 0.1–0.3 %).
- Blockers: HuggingFace downloads blocked in the cloud sandbox — model runs happen on Thomas's machine. No GPU anywhere: bit-accurate runs are limited to the first 8,192 tokens; A3 statistical mode is on the critical path for everything at full scale.

---

## 7. Progress log

Format: `YYYY-MM-DD — what was done — where (code / results path)`

- 2026-11-02 — Plan created (this file). Folder was empty. — `CLAUDE.md`
- 2026-11-02 — Detailed Part A plan written. — `docs/part_a_plan.md`
- 2026-11-02 — A0 skeleton: package `cimsim`, configs, perplexity eval, run script, README, assumptions doc. — `cimsim/`, `eval/`, `scripts/run_ppl.py`, `config/experiments/`
- 2026-11-02 — A1 core: bit-planes, bit-accurate executor (ASiM semantics), CIMLinear (fp / fake_quant / bit_accurate), event counter, HF layer swap (OPT, GPT-2, Llama). 95 tests pass incl. exact ideal match and ASiM layer equivalence (≤1e-6). — `cimsim/core/`, `cimsim/models/swap.py`, `tests/`
- 2026-11-02 — CPU benchmark (2 cores), one 768×3072 layer, 128 tokens, nrow 128: fp 0.004 s, fake_quant 0.004 s, bit_accurate 3.6 s (bit-serial), 1.6 s (enc=3). — `scripts/bench_layer.py`
- 2026-11-02 — CPU-optimised executor (`backend: fast`, default): LUT-free ADC via verified float32 scale, [T,C,K] layout so shift-add is one batched matmul, lossless-readout shortcut. Full OPT-125M-sized model on 2 cores: fake_quant 1.6 ms/token; bit-accurate serial ADC6 94 ms/token (reference 186); enc3 43 ms/token; 1% noise 148 ms/token. 122 tests pass (fast vs reference ≤1e-5 noise-free, noise std within 5%, ASiM equivalence on both backends). — `cimsim/core/executor_fast.py`, `tests/test_fast_executor.py`, `docs/cpu_guide.md`
- 2026-11-02 — Fix for Thomas's env (huggingface_hub 2.4, transformers 5.22, Python 3.13, Windows): WikiText loaded as `Salesforce/wikitext` (bare `wikitext` rejected by hub 2.x), `dtype=` instead of deprecated `torch_dtype=`. pytest passes on Thomas's machine. — `eval/perplexity.py`, `scripts/run_ppl.py`
- 2026-11-02 — Run script: progress/ETA per window, per-window NLL saved, `--threads`, CPU float32 default. Tests find ASiM at `third_party/ASiM` or `ASiM/` (Thomas cloned it at repo root). — `scripts/run_ppl.py`, `eval/perplexity.py`
- 2026-11-02 — E0 first baseline on Thomas's machine (CPU, 14 threads): OPT-125M FP32, WikiText-2 test, seq 1024 non-overlapping, 280 windows: PPL 31.76, 270 s. Higher than the commonly cited 27.65 because that number uses seq 2048; check run at 2048 pending. — `results/e0_opt125m_fp32/`
- 2026-11-02 — OPT-125M FP32 at seq 2048: PPL 27.655 (published 27.65) — evaluation pipeline validated. 319 s on 14 threads. — `results/e0_opt125m_fp32_s2048/`
- 2026-11-02 — E0 complete (11 runs, full test set, fake_quant): OPT-125M FP32 27.655 / W8A8 27.942 / W8A8 per-tensor 30.451 / W6A6 31.952 / W4A8 51.094; GPT-2 FP32 29.951 / W8A8 30.151 / per-tensor 57.452 / W6A6 36.391 / W4A8 47.457. Per-tensor scaling breaks GPT-2 (outliers). W4A8 worse than published RTN — likely symmetric ±7 vs asymmetric 16-level, to confirm. — `results/E0_summary.md`, `results/e0_*/`
- 2026-11-02 — First whole-model bit-accurate run (Thomas's i7-12700H): OPT-125M W8A8 bit-serial nrow 128 ADC 6 b, 8,192 tokens: PPL 12,962 vs 30.45 fake-quant on the same tokens; 608 s (74 ms/token). Diagnosed as genuine (finding F3): two's-complement sign-extension planes + coarse ADC; ADC 8 b brings layer error to 0.4%. Example config moved to ADC 8 b; F3 config kept for reproduction. — `results/sub8k_*`, `docs/findings.md`, `config/experiments/f3_adc6_collapse.yaml`
- 2026-11-02 — Repo pushed to GitHub (thomvs-dev/larp), results now tracked in git. Reference `executor.py` in the agent's working copy had an undocumented float32 change; reverted to the validated float64 version (identical to the one on Thomas's machine) before pushing. — `cimsim/core/executor.py`
- 2026-11-02 — Calibrated ADC range: executors take `full_scale [C,P]` and custom row `chunks`; `mac_histogram` + MSE / percentile calibration; `calibrate_adc_range`; `run_ppl.py --calib-tokens` (train split). On a synthetic LLM-like layer, MSE calibration makes a 6-bit ADC lossless (rel. error 2.65 → 0.0000, held-out): it picks an ADC step of exactly 1 MAC unit (F5). Percentile calibration hurts at ≥7 b (clipping on high-significance planes). — `cimsim/core/calib.py`, `cimsim/calibrate.py`, `docs/findings.md`
- 2026-11-02 — Asymmetric weights (`w_quant: asymmetric`), unsigned planes in both executors, exact zero-point correction. — `cimsim/quant.py`, `cimsim/core/*`
- 2026-11-02 — Layers refactored onto `_CIMBase`; `CIMConv2d` added (ASiM channel-group mapping); `swap_linears` also replaces Conv2d. Matches `ASiMConv2d` ≤1e-6 in 14 layer cases; found ASiM conv skips channel groups when ceil(C/floor(nrow/k²)) > ceil(C·k²/nrow) (F4). 173 tests pass. — `cimsim/core/layers.py`, `tests/test_calib_asym_conv.py`
- 2026-11-02 — A2 prep: pretrained ResNet-20 (chenyaofo, hash 4118986f, strict load OK), ASiM bridge, sweep script (smoke-tested on synthetic data). All 20 conv layers match ASiM to 1.3e-6 inside the network, but at ADC 6 b whole-network predictions differ on 30% of images (float-noise amplification, F6). cimsim ~8× faster than ASiM on ResNet-20. — `cimsim/models/cifar_resnet.py`, `validation/`
- 2026-11-02 — F2 resolved: W4A8 with asymmetric weights OPT-125M 37.39 / GPT-2 38.90 (symmetric 51.09 / 47.46; published 4-bit RTN OPT-125M 37.28). — `results/e0_*_w4a8_asym/`
- 2026-11-02 — F5 confirmed on OPT-125M (8,192 tokens, bit-serial W8A8, nrow 128, ADC 6 b): MSE-calibrated ADC range PPL 30.81 (per layer and per (cycle, plane) identical) vs 12,962 with ASiM's full-range ADC and 30.45 fake-quant. Calibration picks FS = 63 (1 MAC unit per ADC step) for every 6-bit layer. Calibration 327 s. — `results/f5_opt125m_adc6_cal_*/`
- 2026-11-02 — A2 smoke (200 images): cimsim vs ASiM within 1% on 3 configs; ~9× faster. Fixed a Windows crash in the A2 script (table written in cp1252, now UTF-8 + ASCII; `--table-only` rebuilds a table from results.json); all result files now written as UTF-8. — `results/a2_smoke/`, `validation/a2_asim_compare.py`, `scripts/run_ppl.py`
- 2026-11-02 — GATE A2 criterion approved by Thomas; ADC-architecture sub-tasks added to A7 (error models), B0 (cost, sharing ratio), B3 (conversion scheduling). — `CLAUDE.md`
- 2026-11-02 — `run_ppl.py`: window length defaults to model context (cap 2048); `--max-tokens` for the CPU subset; git stderr silenced. — `scripts/run_ppl.py`
- 2026-11-02 — A2 full sweep (1,000 images, 13 configs, git bcde8c5b): fake-quant and all ADC ≥ 8 b configs within 0.8% of ASiM; bs_n64_adc8 cimsim 92.4 / ASiM 90.3, explained by F4; ADC ≤ 6 b diffs −0.4 … +1.2%. Trends match for ADC bits, nrow, noise, encoding. cimsim 1.6–66× faster on bit-accurate configs. Gate conditions 1–3 pass; condition 4 needs measured spread. Added `--backend` and per-image predictions to the A2 script, and `validation/a2_spread.py`. — `results/a2_resnet20/`, `validation/asim_repro.md`
- 2026-11-02 — A2 spread run (5 low-ADC configs, 1,000 images): cimsim fast vs reference differ by up to 1.4% accuracy, 29% of predictions at n128/ADC 6 b; cimsim vs ASiM ≤ 1.2%, 33%. Reruns reproduce the sweep exactly. **GATE A2 passed.** F6 updated with measured spread. — `results/a2_spread_fast/`, `results/a2_spread_ref/`, `validation/asim_repro.md`, `docs/findings.md`
- 2026-11-02 — A3 design (M0–M3 error models, GATE A3 protocol with F6 spread) and error-measurement tool: `_probe` hook in `_CIMBase` (no effect on outputs, tested), `ErrorProbe`, `scripts/measure_error.py`; shared run setup moved to `eval/runner.py`; 4 measurement configs. 179 tests pass. — `docs/part_a_plan.md` §3, `cimsim/stat/`, `scripts/measure_error.py`, `config/experiments/a3_*.yaml`
- 2026-11-03 — A3 measurements (OPT-125M, 2,048 train tokens, 6 configs): full-range ADC 7–8 b error is a pure gain (g = round(1/Δ)·Δ − 1 = 0.39 % / 0.79 %, residual ≤ 0.1 %); calibrated ADC 6 b noise-free error is sparse channel-concentrated saturation; analog noise error is Gaussian, token- and channel-independent, and matches an analytic formula to ≤ 0.2 %. Noise 0.25–1 % destroys every layer (error 1.4–8× signal); cliff at ~0.1 % (F7). — `results/a3m_summary.md`, `results/a3m_opt125m_*/`, `docs/findings.md` F7
- 2026-11-03 — `mode: statistical` with analytic error model M3 (exact per-conversion moments from the calibration MAC histogram → layer gain + Gaussian). Calibration pass also prepares statistical layers. Layer-level check vs bit-accurate: noise std within 5 % (0.2–2.5 %) over 7 configs incl. nonlinearity, hybrid, bit-parallel, no ADC, post-ReLU; gain exact; event counts identical. GATE script (one calibration for both modes, PPL fast/reference/seeds, verdict, speed-up) and 3 revised noise configs. 192 tests pass. — `cimsim/stat/analytic.py`, `cimsim/core/layers.py`, `scripts/a3_gate.py`, `tests/test_stat_analytic.py`, `config/experiments/a3g_*.yaml`

- 2026-11-03 — GATE A3 v1, first config (Thomas's machine, git 33e49ce): f5 calibrated ADC 6 b, 8,192 tokens: fake-quant 30.449, bit-accurate fast 30.808, statistical 53.97 / 58.84 / 54.58 (seeds 0–2) → fail (+79 %). Statistical 6.0 ms/token vs bit-accurate 93 ms/token. Cause found (F8): saturation errors are correlated across input planes and cancel in the shift-add; M3's independence assumption overestimates them 2–26× per layer (L9 fc1 predicted std 6922 vs measured 263) and spreads them over every output. — `results/a3g_opt125m_f5/`, `docs/findings.md` F8
- 2026-11-03 — Statistical mode v2: exact sparse saturation term (`stat_saturation: exact`, default) with popcount bounds and a calibration column filter (`stat_sat_margin: 8`); M3 keeps the clamped noise moments minus S(m). Synthetic 768→3072 layer, 2,048 tokens, 2 cores: exact work 0.08 % of bit-accurate MACs, 0.23 s vs 0.06 s fake-quant (bound only: 15.7 %, 4.2 s). Gate script: `--reuse-ba`, ADC full-scale hash, `sat_work_fraction`. 208 tests pass. — `cimsim/stat/saturation.py`, `cimsim/stat/analytic.py`, `cimsim/core/layers.py`, `cimsim/core/calib.py`, `scripts/a3_gate.py`, `tests/test_stat_saturation.py`
- 2026-11-03 — OPT-125M and WikiText-2 copied into the cloud sandbox from `D:\japan\hf_local` (git-ignored; `CIMSIM_WIKITEXT_DIR` selects local parquet files in `load_wikitext2_ids`). Real-model check of statistical v2 on f5 (2 threads): PPL 30.799 on the first 8,192 test tokens vs bit-accurate 30.808 from the gate run (−0.03 %); strict saturation term exact on every 6-bit layer; margin filter missed 7 of 1.7e8 outputs; exact work 0.97 % of bit-accurate MACs; 8.35 ms/token vs 1.55 fake-quant. — `results/a3s_opt125m_f5_v2_sandbox/check.json`, `scripts/a3_real_check.py`, `eval/perplexity.py`
- 2026-11-03 — Statistical v2 PPLs for the noise configs in the sandbox (4,096 test tokens, seeds 0/1/2; bit-accurate PPLs come from Thomas's gate runs): rn008 25.713 / 25.725 / 25.788, rn010 30.812 / 31.497 / 30.567, rn02_hyb6 29.444 / 29.777 / 29.063; fake-quant 25.381. Layer-level shape at rn010: statistical std matches bit-accurate (ratio 1.00–1.03) but bit-accurate error is sparse (median 3.5 % of outputs non-zero, kurtosis ~680). From Thomas's v1 chain so far: f5 reference backend 30.806 (spread vs fast 0.006 %); adc8 fast 30.483 vs statistical 30.636 / 30.716 / 30.615 (+0.50 %). — `results/a3s_opt125m_*_v2_sandbox/stat_only.json`, `results/a3s_opt125m_rn010_layer_shape_sandbox/`, `scripts/a3_stat_only.py`, `scripts/a3_layer_shape.py`
- 2026-11-03 — GATE A3 v1 chain finished on Thomas's machine (git 33e49ce): bit-accurate f5 30.808 / 30.806, adc8 30.483 / 30.423, adc7 30.607 / 30.530, rn008 25.716 / 25.731, rn010 137.197 / 143.281, rn02_hyb6 29.232 / 28.954; v1 statistical passes adc8 only. With v2 statistical (sandbox): f5 −0.03 %, adc8 +0.50 %, rn008 +0.04 %, rn02_hyb6 +0.73 % pass; adc7 +2.31 % and rn010 −77.5 % fail. Causes: nonlinear rounding above MAC 64 overestimated by the independent model (adc7); rare large code flips not represented by a Gaussian (rn010) (F9). — `results/a3g_opt125m_*/`, `results/a3_gate_summary.md`, `results/a3_gate_v1_evaluation.md`, `docs/findings.md` F9

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
- 2026-11-02 — ADC range calibration defaults to MSE (squared readout error on calibration data, ties → largest full scale) — percentile clipping hurts at ≥7 b because clipped conversions land on planes with shift-add weight up to 2^14 (F5). Calibration statistics are taken with all CIM layers in ideal mode, on data separate from evaluation (LLM: WikiText-2 train).
- 2026-11-02 — A2 uses a pretrained CIFAR-10 ResNet-20 (chenyaofo) instead of ASiM's ResNet-18 example — ASiM ships no trained weights and QAT-training ResNet-18 on a CPU is impractical; the comparison only needs both simulators on the same weights. Stem conv and classifier stay in float in both.
- 2026-11-02 — GATE A2 criterion refined (approved by Thomas): ≤1% at ADC ≥ 8 b and fake-quant; at ADC ≤ 6 b within run-to-run spread — whole-network results are chaotic there even with identical per-layer semantics (F6).
- 2026-11-02 — Pipelined ADC considered (Thomas's question): not the default for the macro — at the 6 b that F5 shows is sufficient, SAR/flash are cheaper; pipelined needs residue amplifiers (area, static power), heavy column sharing (serialised conversions), adds latency, and its error pattern (large errors at stage-transition codes) is the worst case under 2^14 shift-add amplification. Kept as a simulated option: error model in A7, cost and sharing in B0/B3, so the comparison is made with numbers.
- 2026-11-02 — Results (`results/*/`) are tracked in git (small JSON/YAML) so every number in the log is reproducible from the repo. Large artefacts, if any later, go elsewhere.
- 2026-11-02 — Evaluation window = model context length, capped at 2048 (OPT/Qwen/TinyLlama 2048, GPT-2 1024) — matches quantization literature so baselines compare directly with published tables (OPT-125M reproduces 27.65). CPU bit-accurate subset defined by tokens (first 8,192) rather than windows.
- 2026-11-02 — No GPU: project is CPU-only. One codebase, two executors (`fast` default, `reference` kept for ASiM-exact checks) rather than a separate CPU codebase — a fork would drift from the validated semantics. Whole-model bit-accurate runs limited to OPT-125M/GPT-2 on the first 8,192 tokens; full-set and result-model numbers come from statistical mode (A3).
- 2026-11-03 — Statistical mode uses the analytic model M3 (no fitting to bit-accurate runs) instead of the planned fitted M1 — measurements show the error is a deterministic gain plus input-independent Gaussian noise, both computable from the MAC histogram (≤ 0.2 % on 180 layer-configs); it also extrapolates to unmeasured configs, which A5 needs. Known limit: nonlinear rounding error at full-range ADC 6 b (F3 regime) overestimated 2.3×.
- 2026-11-03 — GATE A3 configs revised: noise 0.25 / 0.5 / 1 % replaced by 0.08 %, 0.10 % and 0.2 % with hybrid_levels 6, with calibrated ADC 6 b on every layer incl. fc2 — the old levels make every layer's error larger than its signal (F7), and an 8-bit fc2 passes noise that 6-bit layers round away.
- 2026-11-03 — GATE A3 noise configs evaluated on 4,096 tokens with bit-accurate seed spread (seeds 0, 1); noise-free configs on 8,192 tokens with fast-vs-reference spread — noisy bit-accurate runs cost ~0.42 s/token on the i7-12700H (noise sampling dominates).

- 2026-11-03 — Statistical mode computes ADC saturation exactly (sparse) instead of modelling it — saturation is deterministic, sparse and correlated across planes (F8); no Gaussian approximation matched it. Option chosen by Thomas over a covariance-corrected Gaussian (wrong error shape) and per-layer bit-accurate fallback (≤ 4× speed-up).
- 2026-11-03 — Saturation candidates limited by calibration column maxima (`stat_sat_margin`, default 8 MAC units) — the strict popcount bound flags ~half of all conversions at plane density ≈ 0.5 and costs ~15 % of a bit-accurate pass. Assumption: a column that never came within 8 MAC units of saturating on 2,048 calibration tokens does not saturate at test time. `stat_sat_margin: null` restores the strict bound for checks.

---

## 9. Open questions / environment

- Machine: Intel Core i7-12700H (6P + 8E cores, 20 threads), 16 GB DDR4, no GPU. torch uses 14 threads by default. Bit-accurate OPT-125M ≈ 74 ms/token.
- Process node to assume for PPA: _TBD_
- Target venue / deadline: _TBD_
- ASiM licence terms: Apache 2.0 (checked 2026-11-02)
- Cloud sandbox: CPU only, 2 cores, no HuggingFace access. Model downloads and long runs happen on Thomas's machine (CPU). Local copies for sandbox checks: Thomas saves models / datasets into `D:\japan\hf_local\` (git-ignored), the agent stages them and sets `CIMSIM_WIKITEXT_DIR` to the parquet folder. Available so far: OPT-125M, WikiText-2 raw (train/test parquet).
- Git: https://github.com/thomvs-dev/larp, branch `main`. Agents push from the cloud sandbox; Thomas pulls into `D:\japan`. Results produced on Thomas's machine are committed by him (or staged into the sandbox and committed by the agent).
- Agent file sync (cloud sandbox → `D:\japan`): `device_commit_files` silently keeps old content when a staging path is reused. Stage every file under a fresh folder (e.g. `/mnt/user-data/outputs/sync_<n>/`), then verify with `md5sum` on the device. Or patch in place with `device_bash`.

## 10. References

- ASiM — https://arxiv.org/abs/2411.11022 — code: https://github.com/Keio-CSG/ASiM
- NeuroSim V1.5 — https://arxiv.org/abs/2505.02314 — code: https://github.com/neurosim/NeuroSim
- CiMLoop — https://arxiv.org/abs/2405.07259 — code: https://github.com/mit-emze/cimloop
- MICSim — https://arxiv.org/abs/2409.14838
- EdgeCIM — https://arxiv.org/abs/2604.11512
- FusionCIM — https://arxiv.org/abs/2604.25317
- HASTILY — https://arxiv.org/abs/2502.12344
- Ouroboros — https://arxiv.org/abs/2603.02737
