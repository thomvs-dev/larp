# E0 — Baselines (A0)

WikiText-2 test set, full (≈286.5k tokens), non-overlapping windows of the model's context length (OPT-125M: 2048, GPT-2: 1024). CPU, float32, 14 threads. `fake_quant` = integer quantization with no array effects (ideal digital reference). Run date 2026-11-02. Raw outputs: `results/e0_*/`.

## Perplexity

| Config | Weights | Activations | OPT-125M | Δ vs FP32 | GPT-2 | Δ vs FP32 |
|---|---|---|---|---|---|---|
| FP32 | — | — | **27.655** | — | **29.951** | — |
| W8A8 | 8 b per-channel | 8 b per-token | 27.942 | +0.29 | 30.151 | +0.20 |
| W8A8 per-tensor (ASiM style) | 8 b per-tensor | 8 b per-tensor | 30.451 | +2.80 | 57.452 | +27.50 |
| W6A6 | 6 b per-channel | 6 b per-token | 31.952 | +4.30 | 36.391 | +6.44 |
| W4A8 | 4 b per-channel | 8 b per-token | 51.094 | +23.44 | 47.457 | +17.51 |

Extra reference: OPT-125M FP32 at seq 1024 = 31.761 (`results/e0_opt125m_fp32/`).

## Checks

- **FP32 matches published values.** OPT-125M 27.655 vs 27.65 (GPTQ/SmoothQuant tables, seq 2048). GPT-2 29.95 matches the commonly reported non-overlapping 1024-token value (~29.9). The evaluation pipeline is correct.
- **MAC counter consistent.** Both models: 2.435e13 MACs over the test set = 7.08M linear-layer weights/layer × 12 layers × ~286.7k tokens. (OPT-125M and GPT-2 have identical linear-layer sizes, LM head excluded.)
- **W8A8 per-channel / per-token is close to lossless** (+0.2 to +0.3), as expected for these models. This is the reference that analog CIM configs in A5–A6 are compared against.

## Findings

1. **Per-tensor scaling (ASiM's default) breaks GPT-2:** +27.5 perplexity at W8A8, vs +0.2 with per-token scaling. OPT-125M is hurt less (+2.8). This is the activation-outlier problem: a few channels with large magnitude set the per-tensor scale, and every other value gets only a few quantization levels. It directly motivates A5 (outlier statistics) and A6 (mitigation), and it means ASiM's per-tensor setup cannot be used unchanged for LLMs.
2. **W4A8 is much worse than published 4-bit round-to-nearest numbers** (OPT-125M +23 here vs roughly +10 for 4-bit RTN in the GPTQ paper). Most likely cause: our weights use **symmetric** quantization with levels −7…+7 (15 levels, ASiM convention, matches two's-complement bit planes in the array), while the published RTN uses **asymmetric** min-max with 16 levels and a zero point. Not a bug, but it should be confirmed with an asymmetric-weight option before any W4 result is reported. W8 is unaffected in practice.
3. **W6A6 costs +4.3 / +6.4** — usable as a lower-precision operating point for sweeps.

## Open items

- `git` hash is `unknown` for all runs: `D:\japan` is not a git repository yet.
- Confirm finding 2 (asymmetric 4-bit weight option) — low priority, W8 is the main operating point.
