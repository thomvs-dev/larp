# SRAM CIM Simulator for LLMs — Project Plan

Owner: Thomas
Folder: `D:\japan`
Status: **Part A — A0 baselines done (E0), A1 core done (GATE A1 passed). Next: A1 leftovers, then A2**
Last updated: 2026-11-02

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
- [ ] ADC range option: per-layer calibrated full scale (needed for A5/A6; not in ASiM).
- [ ] Asymmetric (zero-point) weight quantization option, to confirm the W4A8 gap vs published 4-bit RTN (E0 finding 2). Low priority.
- [x] Noise injection: Gaussian random noise and nonlinearity (ASiM-style `rand_noise_sigma`, `non_linear_sigma`), seeded per layer.
- [x] Signed weight handling documented: two's complement planes (`docs/assumptions.md` #1). Differential / offset schemes: optional, later.
- [x] Event counter: ADC conversions (total / analog), array cycles, MACs, weight bits written.
- [x] Vectorized across bit-planes (one matmul per row chunk); CPU benchmark done (`scripts/bench_layer.py`).
- [x] CPU-optimised executor `backend: fast` (`cimsim/core/executor_fast.py`), tested against reference and ASiM. 2.4–3.4× faster noise-free, ~1.5× with noise.
- [x] **GATE A1:** ideal-config test matches integer matmul exactly; tests pass for random shapes. (122 tests pass)

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
- [ ] Run on TinyLlama-1.1B and Qwen-0.5B in statistical mode (CPU); record runtime.

### A5 — Sensitivity study

- [ ] Perplexity vs ADC bits and `nrow`, per layer type.
- [ ] Per-layer noise sensitivity (inject noise one layer at a time).
- [ ] Activation outlier statistics per layer (max/std, channel outliers) and correlation with sensitivity.
- [ ] Bit-serial vs bit-parallel on LLMs, compared with ASiM's CNN/ViT findings.
- [ ] Input encoding study (F3): two's complement vs sign-magnitude vs offset-binary inputs, and skipping sign-extension planes — perplexity vs ADC bits at equal ADC conversions. Needs new encoding options in both executors (reference first, then fast, tested against each other).
- [ ] ADC level alignment (F3): report error vs ADC bits for nrow aligned (2^b − 1 multiple of nrow) and not aligned, separately from ADC precision.

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
| No GPU (CPU only) | `backend: fast`; bit-accurate only on OPT-125M/GPT-2, first 8,192 tokens; result models only in statistical mode; see `docs/cpu_guide.md` |

---

## 6. Status

- Current part: **A**
- Current milestone: **A1 leftovers → A2**
- Next tasks, in order:
  1. Thomas: link `D:\japan` to the GitHub repo (README "Git").
  2. Agent: per-layer calibrated ADC range option (A1 leftover).
  3. Agent: `CIMConv2d` (needed for A2 ResNet-20), then A2 on a fixed 1,000-image CIFAR-10 subset.
  4. Agent (low priority): asymmetric weight option to check E0 finding 2.
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
- 2026-11-02 — `run_ppl.py`: window length defaults to model context (cap 2048); `--max-tokens` for the CPU subset; git stderr silenced. — `scripts/run_ppl.py`

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
- 2026-11-02 — Results (`results/*/`) are tracked in git (small JSON/YAML) so every number in the log is reproducible from the repo. Large artefacts, if any later, go elsewhere.
- 2026-11-02 — Evaluation window = model context length, capped at 2048 (OPT/Qwen/TinyLlama 2048, GPT-2 1024) — matches quantization literature so baselines compare directly with published tables (OPT-125M reproduces 27.65). CPU bit-accurate subset defined by tokens (first 8,192) rather than windows.
- 2026-11-02 — No GPU: project is CPU-only. One codebase, two executors (`fast` default, `reference` kept for ASiM-exact checks) rather than a separate CPU codebase — a fork would drift from the validated semantics. Whole-model bit-accurate runs limited to OPT-125M/GPT-2 on the first 8,192 tokens; full-set and result-model numbers come from statistical mode (A3).

---

## 9. Open questions / environment

- Machine: Intel Core i7-12700H (6P + 8E cores, 20 threads), 16 GB DDR4, no GPU. torch uses 14 threads by default. Bit-accurate OPT-125M ≈ 74 ms/token.
- Process node to assume for PPA: _TBD_
- Target venue / deadline: _TBD_
- ASiM licence terms: Apache 2.0 (checked 2026-11-02)
- Cloud sandbox: CPU only, 2 cores, no HuggingFace access. Model downloads and runs happen on Thomas's machine (CPU).
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
