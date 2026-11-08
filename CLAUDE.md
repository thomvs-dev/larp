# SRAM CIM Simulator for LLMs — Project Plan

Owner: Thomas
Folder: `D:\japan`
Status: **Part A — A0–A3 done (GATE A2, GATE A3 passed). A4 (LLM integration) started. Part B started in parallel (B0/B1 groundwork, `docs/part_b_plan.md`); result model: Qwen2.5-1.5B**
Last updated: 2026-11-08

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
- [x] Fix F9 (Thomas chose A1 + B1): exact sparse deterministic term incl. nonlinear rounding, discrete code-flip sampling by exact thinning (`stat_noise: auto`); 216 tests pass. Sandbox v3: f5, adc8, adc7, rn008, rn02_hyb6 pass; rn010 open (`results/a3_gate_v1_evaluation.md` §v3).
- [x] rn010: bit-accurate seeds 2–4 + statistical v3 seeds 0–7 with `--ref median` (Thomas, git db1c52f): bit-accurate 76–176 (median 143.3), statistical 95–309 (median 150.4), +5.0 %, Mann-Whitney p = 0.35 (`results/a3g_opt125m_rn010/`, `results/a3_gate_v1_evaluation.md` §rn010).
- [x] Criterion for noise-cliff configs approved by Thomas 2026-11-04 (`docs/part_a_plan.md` §3.5).
- [x] Official GATE A3 reruns of all six configs with v3 and `--reuse-ba` (Thomas, git 79b2b4a): f5 −0.01 %, adc8 −0.20 %, adc7 −0.25 %, rn008 −0.09 %, rn010 +4.97 % (medians), rn02_hyb6 +0.25 %; speed-up 6–29× except rn02_hyb6 1.1× (`results/a3_gate_summary.md`, `results/a3_gate_v1_evaluation.md` §Official).
- [x] Speed fix v3.1: Gaussian instead of discrete flips where flips would leave the output error near-Gaussian (`stat_flip_kurtosis_min`, default 1.0); rn02_hyb6 sandbox 29.444 median at 12.5 ms/token; 217 tests pass (`results/a3s_opt125m_rn02_hyb6_v31_sandbox/`).
- [x] Official rerun of rn02_hyb6 with v3.1 (Thomas, git 0517165): 29.544 / 29.891 / 29.012 vs 29.232, +1.07 %, 55× (`results/a3g_opt125m_rn02_hyb6/`).
- [x] **GATE A3:** on OPT-125M, fast-mode perplexity within agreed tolerance (start with ±2% relative) of bit-accurate on a subset of WikiText-2; speed-up recorded. Passed 2026-11-04: all six configs (f5 −0.01 %, adc8 −0.20 %, adc7 −0.25 %, rn008 −0.09 %, rn010 +4.97 % on medians under the noise-cliff rule, rn02_hyb6 +1.07 %); speed-up 6–55× (`results/a3_gate_summary.md`, `results/a3_gate_v1_evaluation.md` §verdict).

### A4 — LLM integration

- [x] Layer swapping for HuggingFace models (OPT, GPT-2, Llama-family, Qwen): Qwen2 added to the tiny-model tests (q/k/v biases, tied LM head kept digital), statistical mode with calibration runs on tiny Llama / Qwen2; quantized weight codes cached as int8 (memory for 1.5B models). Real-model check: Qwen2.5-0.5B statistical vs bit-accurate 11.994 vs 11.993 (+0.009 %, 4,096 tokens, `results/a3g_qwen05b_adc6cal/`). (`tests/test_swap.py`, `cimsim/core/layers.py`)
- [x] Per-layer mapping policy in config: which of Q/K/V/O, FFN up/gate/down run on analog CIM, DCIM, or digital. `macro: acim | dcim | digital` per layer; YAML `policy:` maps roles qkv / o / up_gate / down (same across OPT, GPT-2, Llama, Qwen2) to a macro; `layers:` globs override. dcim / digital compute exactly and count `dcim_column_ops` / `digital_macs` instead of ADC conversions. (`cimsim/config.py` ROLE_PATTERNS, `cimsim/events.py`, `config/experiments/a4_policy_example.yaml`, `tests/test_swap.py`)
- [x] `CIMMatMul` for dynamic operands (QKᵀ, S·V) with write events counted (KV written into arrays). HF attention interface `cim` (OPT, GPT-2, Llama, Qwen2), YAML `attention: true`, roles `qk` / `sv`; keys per channel / values per token by default (`kv_gran`, F10); events only for causal pairs, `kv_bits_written` per new token. Qwen2.5-0.5B (sandbox, 512-token windows): attention on analog CIM statistical = bit-accurate (17.596), full model on CIM incl. attention 17.620 vs W8A8 linear 17.895. Thomas's machine, 2,048-token windows: 0.5B statistical = bit-accurate 12.043; 1.5B 8k tokens analog full model 10.378 vs ideal 8-bit attention 10.413 vs W8A8 linear 10.338. 242 tests. (`cimsim/core/attention.py`, `tests/test_attention.py`, `results/a4_attention_summary.md`)
- [ ] Eval: WikiText-2 perplexity + small zero-shot set (e.g. ARC-e, PIQA, HellaSwag via lm-eval-harness).
  - [x] Zero-shot wrapper: `eval/zeroshot.py` (lm-eval HFLM around the swapped, calibrated model; one calibration per run, all tasks in one process; per-document scores, events and ms/token per task; resume), `scripts/run_zeroshot.py`, paired comparison `scripts/zeroshot_compare.py` (Δ on the same documents, paired 95 % interval, exact McNemar). Batch 1 with CIM attention, 8 otherwise (§8). Tested on a tiny model with a local lm-eval task (`tests/test_zeroshot.py`).
  - [x] Qwen2.5-0.5B: FP32, W8A8, analog ADC 6 b cal (statistical), first 1,000 documents per task (Thomas, git 2ce11a2): analog vs W8A8 differs by at most 0.4 pp on every task and metric (paired 95 % intervals contain 0, McNemar p ≥ 0.29); FP32 vs W8A8 at most 1.1 pp (p ≥ 0.17). Statistical run 6.0 h (51–73 ms/token) + 14 min calibration; FP32 27 min; W8A8 46 min. (`results/zs_qwen05b_*/`, `results/zs_qwen05b_summary.md`)
  - [x] WikiText-2 perplexity on the full test set for all three result models (Qwen2.5-0.5B, TinyLlama-1.1B, Qwen2.5-1.5B): FP32, W8A8, analog statistical (`results/a4_ppl_full_summary.md`).
  - [ ] Zero-shot (1,000 docs per task, FP32 / W8A8 / analog statistical) for TinyLlama-1.1B and Qwen2.5-1.5B. Runs done (Thomas, git f5560a1). Analog − W8A8: Qwen2.5-1.5B |Δ| ≤ 0.3 pp, McNemar p ≥ 0.45; TinyLlama |Δ| ≤ 0.4 pp except PIQA acc_norm −1.0 pp (interval −1.96 … −0.04, p = 0.064; one borderline result among 12 comparisons) (`results/zs_qwen15b_summary.md`, `results/zs_tinyllama_summary.md`). Open: TinyLlama FP32 zero-shot is 1.2–8.4 pp *below* W8A8 on every task while its FP32 perplexity is better; suspected faulty run, cause not found (padding test on a random Llama passes). Diagnostic: FP32 ARC-e, batch 1, 200 docs.
  - [x] Statistical-mode speed on short calls (sub-task moved up from A5 because it blocks the TinyLlama / Qwen-1.5B zero-shot runs): vectorise the (chunk, cycle) loop in the exact saturation term; target ≤ 25 ms/token on 0.5B zero-shot calls, outputs identical to the current implementation. Sandbox (2 threads, real Qwen2.5-0.5B), logits bit-identical to the previous code: zero-shot batch shape 8 × 110 22.7 → 15.6 ms/token, 1 × 64 85.8 → 28.5, perplexity window 1 × 2048 22.2 → 15.5; fake-quant floor 5.4 / 11.1 / 5.8 (`results/bench_stat_speed_qwen05b_sandbox/`). Target met. Thomas's laptop (git 3fa5aa9, 14 / 8 / 6 / 4 threads): 8 × 110 12.5 / 12.1 / 11.7 / 10.5 ms/token, 1 × 64 16.1 / 18.6 / 17.7 / 16.4, 1 × 2048 10.2 / 10.7 / 10.2 / 20.7 (the 4-thread rows 8 × 110 and 1 × 2048 look disturbed: fake-quant floor 2× the other runs); fake-quant floor 3.7–4.6 at 8–14 threads; batched = loop output (hash); thread count between 6 and 14 changes little, so the 51–73 ms/token of the old zero-shot run was not the thread count (`results/bench_stat_speed_qwen05b_laptop/`). Real zero-shot ms/token to be re-measured in the A4 runs (the earlier W8A8 zero-shot ran at about 2× the fake-quant floor of this benchmark).
- [x] Run on TinyLlama-1.1B and Qwen-0.5B in statistical mode (CPU); record runtime. Full WikiText-2 test set (Thomas, git 811eead, 6 threads): Qwen2.5-0.5B FP32 13.069 / W8A8 13.332 / analog ADC 6 b cal 13.329; TinyLlama-1.1B (v1.1) 7.707 / 7.793 / 7.793; analog − W8A8 not significant on either. Statistical 16.6 / 37.9 ms/token (83 / 214 min) + calibration 22 / 32 min (`results/a4_ppl_full_summary.md`).

### A5 — Sensitivity study

- [ ] Perplexity vs ADC bits and `nrow`, per layer type.
- [ ] Per-layer noise sensitivity (inject noise one layer at a time).
- [ ] Activation outlier statistics per layer (max/std, channel outliers) and correlation with sensitivity.
- [ ] Bit-serial vs bit-parallel on LLMs, compared with ASiM's CNN/ViT findings.
- [ ] Input encoding study (F3): two's complement vs sign-magnitude vs offset-binary inputs, and skipping sign-extension planes — perplexity vs ADC bits at equal ADC conversions. Needs new encoding options in both executors (reference first, then fast, tested against each other).
- [ ] ADC level alignment (F3): report error vs ADC bits for nrow aligned (2^b − 1 multiple of nrow) and not aligned, separately from ADC precision.
- [ ] Statistical-mode speed before large sweeps (started under A4, see sub-task there): vectorise the exact saturation / rounding loop (noise-free speed-up now 6–15×; aim ≥ 30×), tested against the current implementation.
- [ ] Cache calibration histograms per (model, calibration tokens, nrow, bit widths, encoding, quantization): the MAC histogram is taken with all layers ideal, so it does not depend on ADC bits, ADC range method or noise; sweeps over those reuse one calibration (Qwen2.5-1.5B: 53 min each otherwise).
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

Start only after Part A exit criteria are met (or Thomas says so). **Thomas, 2026-11-03: start B0/B1 now in parallel with A3/A4.** Detailed plan: `docs/part_b_plan.md`.

### End goal of Part B

A single run of the simulator outputs **perplexity + energy/token + latency/token (prefill and decode separately) + area** for a configured SRAM CIM system running an edge LLM, with energy/area calibrated against published macros, and design-space sweeps that produce Pareto fronts.

### Part B exit criteria

- [ ] All B milestones ticked, all GATEs passed.
- [ ] Calibration error table against ≥3 published SRAM CIM macros.
- [ ] Coupled DSE results for at least 2 result models.
- [ ] Write-up in `docs/part_b_results.md`; combined A+B tool release.

### B0 — Component library

- [x] Part B plan: cost-model structure, calibration macro candidates (Jia JSSC'20 65 nm, Sinangil JSSC'21 7 nm, Wang JSSC'23 22 nm C-2C, Chih ISSCC'21 22 nm DCIM — CiMLoop's validated set + one DCIM), TOPS/W normalisation rules (`docs/part_b_plan.md`). Macro numbers still to be verified from the original papers.
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
- Current milestone: **A4** (LLM integration)
- Next tasks, in order:
  1. Done 2026-11-05: attention-on-CIM runs at 2,048-token windows (0.5B statistical = bit-accurate 12.043; 1.5B 8k analog full model 10.378 vs ideal 8-bit attention 10.413).
  2. Done 2026-11-06: zero-shot wrapper and Qwen2.5-0.5B runs (FP32 / W8A8 / statistical, 1,000 docs per task); analog = W8A8 within 0.4 pp, no significant difference.
  3. Done in the sandbox 2026-11-06: statistical mode 1.4–1.7× faster on real calls, bit-identical (`results/bench_stat_speed_qwen05b_sandbox/`). Laptop benchmark done (thread count 6–14 equivalent; `--threads 6`). Done 2026-11-07: full test-set perplexity for Qwen2.5-0.5B and TinyLlama-1.1B (`results/a4_ppl_full_summary.md`). Done 2026-11-08: zero-shot TinyLlama-1.1B and Qwen2.5-1.5B (analog = W8A8). Next, Thomas: commit the six `results/zs_tinyllama_*`, `results/zs_qwen15b_*` folders; diagnostic rerun of TinyLlama FP32 zero-shot at batch 1 (the last open A4 point). Then A5, starting with the calibration histogram cache.
  4. Agent (parallel, sandbox): Part B B0 — verify M1–M4 specs from the papers, start `cimsim/circuits/` with cited per-action costs (`docs/part_b_plan.md` §3). Open questions for Thomas in §4 of that file (process node, all-analog vs hybrid, macro vs system TOPS/W).
  5. Note: compare statistical against bit-accurate on the same machine (fake-quant differs between Thomas's machine and the sandbox by 0.1–0.3 %).
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
- 2026-11-03 — Statistical v3 (F9): exact sparse deterministic readout term (saturation + nonlinear rounding above the linear region; full-range ADC exact above m = 64) and discrete ADC code-flip sampling for rare noise flips, by exact thinning (candidate MAC computed, accept with P(flip|m)/P_max). Exact saturation path processed in row blocks (memory). Gate script `--ref median`. 216 tests pass. Sandbox (2 threads): f5 30.799 (−0.03 %), adc8 30.443 (−0.13 %), adc7 30.566 (−0.13 %), rn008 median 25.966 (+0.97 %), rn02_hyb6 29.123 (−0.37 %); rn010 12 seeds 69–440, median 174 vs bit-accurate 137 / 143 (open). rn010 layer level matches bit-accurate (std ratio 0.99–1.06, non-zero fraction within 2 %, equal per-position flip rates). — `cimsim/stat/`, `cimsim/core/layers.py`, `tests/test_stat_flips.py`, `results/a3s_opt125m_*_v3*_sandbox/`, `results/a3s_opt125m_rn010_token_rates_sandbox/`, `results/a3_gate_v1_evaluation.md`
- 2026-11-04 — rn010 official gate run (Thomas, git db1c52f, 5 bit-accurate + 8 statistical seeds, `--ref median`): bit-accurate 137.2 / 143.3 / 150.5 / 175.5 / 76.0 — bit-accurate PPL at the noise cliff varies by a factor 2.3, so the two-seed reference was undersampled. Statistical 94.8–308.7. Medians 143.3 vs 150.4 (+5.0 %), geometric-mean ratio 1.25 (approx. 95 % 0.80–1.94), Mann-Whitney p = 0.35; speed-up 17.7×. Criterion proposal for noise-cliff configs written. — `results/a3g_opt125m_rn010/`, `results/a3_gate_summary.md`, `results/a3_gate_v1_evaluation.md` §rn010, `docs/findings.md` F9
- 2026-11-04 — Official GATE A3 v3 reruns (Thomas, git 79b2b4a): all six pass on accuracy (f5 −0.01 %, adc8 −0.20 %, adc7 −0.25 %, rn008 −0.09 %, rn010 +4.97 % medians, rn02_hyb6 +0.25 %). Statistical 6.3–9.2 ms/token noise-free (6–15× faster than bit-accurate), rn008 29×, rn010 18×, rn02_hyb6 450 ms/token (1.1×). Fix v3.1: kurtosis test for discrete flips (`stat_flip_kurtosis_min`); rn02_hyb6 sandbox 29.444 / 29.777 / 29.063 at 12.5 ms/token. 217 tests pass. — `results/a3g_opt125m_*/`, `results/a3_gate_summary.md`, `results/a3_gate_v1_evaluation.md` §Official, `results/a3s_opt125m_rn02_hyb6_v31_sandbox/`, `cimsim/stat/analytic.py`, `tests/test_stat_flips.py`
- 2026-11-04 — rn02_hyb6 v3.1 official rerun (Thomas, git 0517165): 29.544 / 29.891 / 29.012 vs bit-accurate 29.232 (+1.07 %), 9.2 ms/token, 55×. Noise-cliff criterion approved by Thomas. **GATE A3 passed** (all six configs). — `results/a3g_opt125m_rn02_hyb6/`, `results/a3_gate_summary.md`, `results/a3_gate_v1_evaluation.md` §verdict, `docs/part_a_plan.md` §3.5
- 2026-11-04 — A4 start: Qwen2 swapping tested on tiny models; per-layer mapping policy (`macro` field, role `policy:` in YAML) with separate dcim / digital event counts; int8 cache for quantized weights (Qwen2.5-1.5B linear weights: 5.2 GB float32 + 1.3 GB codes instead of 10.5 GB). 228 tests pass. — `cimsim/config.py`, `cimsim/events.py`, `cimsim/core/executor.py`, `cimsim/core/layers.py`, `cimsim/calibrate.py`, `tests/test_swap.py`, `config/experiments/a4_*.yaml`
- 2026-11-04 — Qwen2.5-0.5B smoke runs (Thomas, git 6bf9d61, 8,192 tokens): FP32 14.569, W8A8 14.821 (+1.73 %), statistical ADC 6 b calibrated 14.809 (−0.08 % vs W8A8); calibration picks FS 63 on all 168 layers (F5 holds on Qwen); calibration 1,461 s, statistical 20 ms/token. MAC histogram rewritten (float32 + histc, identical output, 2.1× faster); events.json now excludes the calibration pass (`EventCounter.reset_compute`). 229 tests pass. — `results/a4_qwen05b_summary.md`, `results/e0_qwen05b_*_sub8k/`, `results/a4_qwen05b_stat_adc6cal_sub8k/`, `cimsim/core/calib.py`, `cimsim/events.py`, `scripts/run_ppl.py`
- 2026-11-04 — Qwen2.5-1.5B on 8,192 tokens (Thomas, git 0e2ca2c): FP32 10.113, W8A8 10.338 (+2.23 %), statistical ADC 6 b calibrated 10.342 (+0.03 % vs W8A8); FS 63 on all 196 layers; calibration 3,193 s, evaluation 66 ms/token; events exclude calibration (MACs equal W8A8). Gate script now accepts statistical configs (first Qwen gate attempt stopped on that check); smoke-tested on OPT-125M in the sandbox. — `results/a4_qwen05b_summary.md`, `results/e0_qwen15b_*_sub8k/`, `results/a4_qwen15b_stat_adc6cal_sub8k/`, `scripts/a3_gate.py`
- 2026-11-04 — Qwen2.5-0.5B gate check (Thomas, git 724b0a5, 4,096 tokens): statistical 11.994 vs bit-accurate 11.993 (+0.009 %), 18× faster; calibration 695 s (2.1× faster than before). Statistical mode validated on Qwen2. — `results/a3g_qwen05b_adc6cal/`, `results/a4_qwen05b_summary.md`
- 2026-11-04 — Attention on CIM (A4): `CIMMatMul` (qk: keys in array columns, contraction D; sv: values in array rows, contraction S, per-position V scale folded into P), registered as HF attention implementation `cim`; causal-pair event counts, `kv_bits_written`; gate script `--seq-len`. Qwen2.5-0.5B sandbox (2,048 tokens, 512-token windows): 8-bit Q·Kᵀ with per-token keys +4.2 %, per-channel keys 0 % (F10); analog attention statistical = bit-accurate (17.596); all linear + attention on CIM 17.620 (W8A8 linear 17.895, FP32 17.411). 242 tests pass. — `cimsim/core/attention.py`, `tests/test_attention.py`, `results/a4_attention_summary.md`, `results/a4s_qwen05b_*_s512_sandbox/`, `results/a3g_qwen05b_attn_only_adc6cal_sandbox/`, `docs/findings.md` F10
- 2026-11-05 — Qwen2.5-1.5B full WikiText-2 test set (Thomas, git 2d1d11f, 298,862 tokens): FP32 9.264, W8A8 9.487 (+2.41 %), statistical ADC 6 b calibrated 9.492 (+0.05 % vs W8A8; per-window NLL difference +0.00047 ± 0.00042 nats, not significant). Evaluation 345 min (69 ms/token), calibration 44 min. — `results/e0_qwen15b_fp32/`, `results/e0_qwen15b_w8a8/`, `results/a4_qwen15b_stat_adc6cal/`, `results/a4_qwen05b_summary.md`
- 2026-11-05 — Attention on CIM validated on Thomas's machine (git 67b4623, 2,048-token windows): Qwen2.5-0.5B 4,096 tokens statistical 12.043 = bit-accurate 12.043 (fake-quant 12.053), 5.3x faster; Qwen2.5-1.5B 8,192 tokens: W8A8 linear + 8-bit attention 10.413 (+0.73 % vs W8A8 linear), analog linear + attention 10.378 (-0.34 % vs ideal, not significant); attention adds 6.7 % MACs / 6.9 % ADC conversions. — `results/a3g_qwen05b_attn_only_adc6cal/`, `results/e0_qwen15b_w8a8_attn_sub8k/`, `results/a4_qwen15b_stat_adc6cal_attn_sub8k/`, `results/a4_attention_summary.md`
- 2026-11-05 — Zero-shot wrapper (A4): lm-eval 0.4.13 HFLM around the swapped and calibrated model; per-task `zs_<task>.json` (lm-eval acc / acc_norm, per-document scores, seconds, model tokens, ms/token), `events_<task>.json`, resume; paired comparison script. Statistical mode was 5.6× slower per token on 64-token calls than on 2,048-token calls (random-weight Qwen2-0.5B-shaped model, 4 layers, 2 sandbox threads: 31.5 vs 5.6 ms/token, git 9fcc9fb): the weight-only part of the exact saturation term (bit planes, popcounts, candidate columns) was recomputed on every call. It is now computed once per layer (`saturation_plan`, cached in `_CIMBase`, cleared on requantize / calibration): 64-token calls 16.9 ms/token (1.9× faster), 2,048-token calls 5.3 (unchanged); output identical to the previous implementation on every layer of that model and to a per-call plan (`test_cached_plan_matches_fresh_plan`). Remaining short-call overhead: the Python loop over (chunk, cycle) entries (A5 vectorisation task). 249 tests pass. — `eval/zeroshot.py`, `scripts/run_zeroshot.py`, `scripts/zeroshot_compare.py`, `cimsim/stat/saturation.py`, `cimsim/core/layers.py`, `tests/test_zeroshot.py`, `tests/test_stat_saturation.py`, `scripts/bench_short_calls.py`, `results/bench_short_calls_sandbox_before/`, `results/bench_short_calls_sandbox/`
- 2026-11-06 — Qwen2.5-0.5B zero-shot (Thomas, git 2ce11a2, lm-eval 0.4.13, Python 3.13, batch 8, first 1,000 documents per task, 0-shot): FP32 ARC-e 63.5 / PIQA 70.8 / HellaSwag 40.1 (acc), W8A8 62.4 / 71.3 / 40.0, analog ADC 6 b calibrated statistical 62.1 / 71.7 / 40.0. Paired differences analog − W8A8: ARC-e −0.3 pp, PIQA +0.4, HellaSwag 0.0 (acc_norm −0.2…+0.4), all intervals contain 0, McNemar p ≥ 0.29. Runtime: FP32 3.6–5.9 ms/token (27 min total), W8A8 7.6–8.4 (46 min), statistical 51–73 (6.0 h, +14 min calibration, FS 63 on all 168 layers). Smoke runs (20 docs) predicted 32–42 ms/token for statistical; the full runs were 1.7× slower, cause not identified. Tokens per document 109 / 70 / 170. — `results/zs_qwen05b_fp32/`, `results/zs_qwen05b_w8a8/`, `results/zs_qwen05b_stat_adc6cal/`, `results/zs_qwen05b_summary.md`, `results/zs_smoke_qwen05b_*/`
- 2026-11-06 — Statistical-mode short-call speed (A4 sub-task), real Qwen2.5-0.5B in the sandbox (2 threads): batched exact-saturation path for calls ≤ 256 tokens (padded buckets, uint8 weight-bit cache, 106 MB for 0.5B; c5b4383), then (fa06edf) conversions below the smallest MAC with a non-zero table entry skipped before the int64 conversion and table gather, digits and row sums once per (chunk, cycle) in the loop path, and the statistical matmul in float32 when the Cauchy-Schwarz bound is below 2^23 (exact). Zero-shot batch shape 8 × 110: 22.7 → 15.6 ms/token; 1 × 64: 85.8 → 49.1 → 28.5; 1 × 2048 window: 22.2 → 15.5; fake-quant floor 5.4 / 11.1 / 5.8. Logits bit-identical to the previous code on five shapes; 254 tests pass. Benchmark script measures several thread counts after one calibration, `--save` / `--load` for the calibrated model. — `cimsim/stat/saturation.py`, `cimsim/core/layers.py`, `tests/test_stat_saturation.py`, `scripts/bench_real_short_calls.py`, `results/bench_stat_speed_qwen05b_sandbox/`, `results/bench_real_short_calls_qwen05b_sandbox/`, `results/bench_short_calls_sandbox_v2/`
- 2026-11-06 — Laptop check of the statistical-mode speed (Thomas, git 3fa5aa9, Qwen2.5-0.5B, calibration 376 s, 1,024 tokens): 8 × 110 12.5 ms/token at 14 threads (6 threads 11.7), 1 × 2048 10.2, fake-quant floor 3.7–4.6; old code had been 51–73 ms/token on zero-shot. Batched and loop paths give identical logits. Speed sub-task closed. — `results/bench_stat_speed_qwen05b_laptop/`
- 2026-11-07 — Full WikiText-2 test set on the laptop (Thomas, git 811eead, 6 threads, 2,048-token windows): Qwen2.5-0.5B FP32 13.069, W8A8 13.332 (+2.01 %), analog ADC 6 b calibrated statistical 13.329 (per-window NLL vs W8A8 −0.00026 ± 0.00035, n.s.); TinyLlama-1.1B v1.1 FP32 7.707, W8A8 7.793 (+1.12 %), statistical 7.793 (−0.00012 ± 0.00019, n.s.). Evaluation time FP32 / W8A8 / statistical: Qwen 4.5 / 7.4 / 16.6 ms/token, TinyLlama 13.0 / 10.8 / 37.9; calibration 22 / 32 min. A4 runtime task closed. — `results/e0_qwen05b_fp32/`, `results/e0_qwen05b_w8a8/`, `results/a4_qwen05b_stat_adc6cal/`, `results/e0_tinyllama_*/`, `results/a4_tinyllama_stat_adc6cal/`, `results/a4_ppl_full_summary.md`
- 2026-11-08 — Zero-shot TinyLlama-1.1B v1.1 and Qwen2.5-1.5B (Thomas, git f5560a1, lm-eval 0.4.13, batch 8, first 1,000 docs, 0-shot, 6 threads). acc ARC-e / PIQA / HellaSwag: TinyLlama FP32 47.5 / 69.0 / 37.2 (suspect), W8A8 55.9 / 71.2 / 38.9, analog 56.3 / 71.0 / 38.9; Qwen2.5-1.5B FP32 75.0 / 75.6 / 45.9, W8A8 74.2 / 75.6 / 45.6, analog 73.9 / 75.7 / 45.5. Analog − W8A8 not significant on any task (paired, McNemar). Statistical 34–40 ms/token TinyLlama (4.2 h), 62–71 Qwen2.5-1.5B (6.4 h); calibration 39 / 85 min. TinyLlama FP32 zero-shot below W8A8 on every task, cause open. — `results/zs_tinyllama_*/`, `results/zs_qwen15b_*/`, `results/zs_tinyllama_summary.md`, `results/zs_qwen15b_summary.md`
- 2026-11-03 — Part B plan written: cost model (event counts × cited per-action costs), calibration macro candidates M1–M4, GATE B1 normalisation rules. — `docs/part_b_plan.md`

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
- 2026-11-03 — F9 fix: Thomas chose A1 (exact sparse deterministic term) + B1 (discrete code-flip sampling) over accepting the limits. Flip sampling uses exact thinning instead of a flip rate averaged over MAC values, so the per-conversion distribution equals the bit-accurate executor's. Discrete sampling only for per-conversion flip probability ≤ 0.05 (Poisson placement is wrong when flips are frequent) and < 4 expected flips per output.
- 2026-11-03 — Approved by Thomas 2026-11-04 (with the distribution rule below): for noisy GATE A3 configs, compare the median statistical PPL with the median of ≥ 5 bit-accurate seeds (`--ref median`); at the noise cliff a single bit-accurate seed is not a stable reference.
- 2026-11-04 — Approved by Thomas (2026-11-04), replaces the tolerance = spread rule for noise-cliff configs: a config is chaotic if sd of log PPL over ≥ 5 bit-accurate seeds > 0.10; then pass needs Mann-Whitney p > 0.05 (≥ 5 vs ≥ 8 seeds), statistical median inside the bit-accurate interquartile range ± 2 %, and layer-level agreement (std ratio within 10 %, non-zero fraction within 5 %). Other configs keep 2 %. Reason: rn010 tolerance = spread = 131 % accepts almost any result.
- 2026-11-04 — Discrete flip sampling only where it changes the error shape: a layer keeps discrete flips if they give its output error an excess kurtosis ≥ 1, otherwise Gaussian — with many flips per output the two give the same result (rn02_hyb6: v2 Gaussian 29.444 = v3.1) and the discrete path costs an exact MAC per flip (450 ms/token vs 12.5).
- 2026-11-04 — Mapping policy as a per-layer `macro` field (acim / dcim / digital) rather than separate layer classes — one layer type keeps quantization identical across macros, so a perplexity difference between policies comes only from analog error; dcim and digital are exact integer compute and differ only in the events Part B costs.
- 2026-11-04 — Quantized weight codes cached as int8 / uint8 (w_bits ≤ 8) and converted to float per call — the float32 cache doubled linear-weight memory, which does not fit Qwen2.5-1.5B on the 16 GB laptop. Outputs unchanged (all tests pass).
- 2026-11-04 — Attention matmuls via the HF attention-function registry (custom implementation `cim`) instead of patching model classes — one function covers OPT, GPT-2, Llama and Qwen2 and leaves model code untouched; fp mode reproduces eager attention to 1e-5.
- 2026-11-04 — Keys quantized per channel (scale folded into Q), values per position (scale folded into P) by default — per-token keys cost +4.2 % PPL on Qwen2.5-0.5B, per-channel 0 % (F10); both folds keep both array operands integer.
- 2026-11-03 — Result model for the A4/B4 headline results: Qwen2.5-1.5B (Thomas). Fits the 16 GB laptop in statistical mode; 3B would need int8 weight storage and ~day-long runs. (There is no Qwen2.5-2.5B.)
- 2026-11-03 — Part B started in parallel with A3/A4 (Thomas) — B0/B1 are literature + modelling work in the sandbox and do not compete with the bit-accurate runs on the laptop. Energy numbers for Qwen come from statistical-mode event counts, so they still depend on GATE A3 for accuracy; no TOPS/W figure is reported before GATE B1.
- 2026-11-05 — Zero-shot runs use lm-eval-harness with the model object passed in (not a model name), 0-shot, the first N documents of each task (`--limit`, lm-eval does not shuffle) so that every config sees the same questions, and paired statistics (McNemar, paired interval) instead of comparing two independent accuracies — differences between configs are ~1 pp, smaller than the unpaired standard error at N = 1,000 (~1.5 pp). N = 1,000 per task as a start: full HellaSwag (10,042 docs, ~3.8M tokens) would take days in statistical mode on the laptop.
- 2026-11-05 — Zero-shot batch size: 1 when attention runs on CIM (right padding would enter the per-channel key scales, F10), 8 otherwise — with linear layers only the padded positions cannot affect the real tokens, and batching removes most of the per-call overhead of statistical mode on short requests (noise-free scores identical to batch 1, `tests/test_zeroshot.py`).
- 2026-11-06 — Statistical-mode short-call speed moved from A5 into A4 as a sub-task — measured 51–73 ms/token on 0.5B zero-shot makes the TinyLlama and 1.5B zero-shot runs 12–24 h each; vectorising the (chunk, cycle) loop is needed before they run. Milestone order unchanged; the A5 item stays with a note. (Thomas to confirm.)
- 2026-11-03 — Calibration macros: CiMLoop's validated SRAM set (Jia JSSC 2020, Sinangil JSSC 2021, Wang JSSC 2023) plus one digital CIM (Chih ISSCC 2021) — open CiMLoop models allow a direct B5 comparison, and the set covers bit-serial charge-domain, multi-bit analog and DCIM.

---

## 9. Open questions / environment

- Machine: Intel Core i7-12700H (6P + 8E cores, 20 threads), 16 GB DDR4, no GPU. torch uses 14 threads by default. Bit-accurate OPT-125M ≈ 74 ms/token.
- Process node to assume for PPA: _TBD_ (asked 2026-11-03; 22 / 28 nm have most published SRAM CIM data)
- Target venue / deadline: _TBD_
- ASiM licence terms: Apache 2.0 (checked 2026-11-02)
- Cloud sandbox: CPU only, 2 cores, no HuggingFace access. Model downloads and long runs happen on Thomas's machine (CPU). Local copies for sandbox checks: Thomas saves models / datasets into `D:\japan\hf_local\` (git-ignored), the agent stages them and sets `CIMSIM_WIKITEXT_DIR` to the parquet folder. Available so far: OPT-125M, Qwen2.5-0.5B (bf16, `/home/claude/hf_local/qwen2.5-0.5b`), WikiText-2 raw (train/test parquet).
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
