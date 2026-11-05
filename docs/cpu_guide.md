# Running on CPU (no GPU)

The project runs CPU-only. There is one codebase; the device is a runtime setting. `backend: fast` (default) is the CPU-optimised bit-accurate executor; `backend: reference` is the slower loop that mirrors ASiM line by line. Both are tested against each other and against ASiM.

## Measured speed

OPT-125M-sized model (12 layers, d=768, FFN 3072), random weights, 128 tokens, **2 CPU cores** (cloud sandbox). Scale roughly by 1 / cores on a machine with more physical cores.

| Mode | ms / token | 1 window (1024 tok) | Full WikiText-2 test (~287k tok) |
|---|---|---|---|
| fp32, no swap | 1.2 | 1 s | ~6 min |
| fake_quant W8A8 | 1.6 | 2 s | ~8 min |
| bit_accurate, bit-serial, ADC 6, nrow 128 | 94 | 1.6 min | ~7.5 h |
| bit_accurate, enc 3, ADC 6, nrow 128 | 43 | 45 s | ~3.5 h |
| bit_accurate, bit-serial, ADC 8, 1% noise | 148 | 2.5 min | ~12 h |
| bit_accurate, `backend: reference` | 186 | 3.2 min | ~15 h |

**Thomas's machine (i7-12700H, 14 threads):** OPT-125M, first 8,192 tokens: fake_quant 10 s total; bit-accurate bit-serial 608 s (≈ 74 ms/token, ~2.5 min per 2048-token window). FP32 full test set ≈ 5 min. ADC range calibration (2,048 train tokens, OPT-125M, 72 layers): ≈ 5.5 min. ResNet-20, 200 CIFAR-10 images, bit-accurate: cimsim ≈ 12 s, ASiM ≈ 103 s.

Fast vs reference executor on one 768×3072 layer: 2.4–3.4× faster noise-free, ~1.5× with noise (random number generation dominates). Configs where the ADC is lossless (e.g. nrow 255 with an 8-bit ADC, or no ADC) skip the bit-level loop and cost the same as fake_quant.

## Evaluation protocol on CPU

| Run type | Models | Tokens |
|---|---|---|
| fp32, fake_quant, statistical | all | full WikiText-2 test set |
| bit_accurate (whole model) | OPT-125M, GPT-2 only | **first 8,192 tokens** (`--max-tokens 8192`: 4 × 2048 for OPT, 8 × 1024 for GPT-2) |
| bit_accurate (calibration for A3) | all | selected layers, a few sequences |

- Window length = model context length, capped at 2048 (OPT/Qwen/TinyLlama 2048, GPT-2 1024), matching published quantization papers.
- The 8k subset is always the first 8,192 tokens of the test set, so every config is evaluated on the same tokens. Compare a bit_accurate run against fake_quant **on the same subset**, not against the full-set number.
- `result.json` stores per-window NLL so the spread across windows can be reported.
- Result models (Qwen2.5-0.5B, TinyLlama-1.1B) are never run bit_accurate end to end on CPU. They use statistical mode (A3).

## Zero-shot runs (A4)

Zero-shot requests are short (about 30–150 tokens per forward call). Statistical mode has a per-call cost (Python loop over row chunks and input cycles in the exact saturation term), so per token it is slower on short calls than on 2,048-token windows: random-weight Qwen2-0.5B-shaped model, 2 threads, 16.9 ms/token at 64 tokens vs 5.3 at 2,048 (`results/bench_short_calls_sandbox/`; 31.5 before the per-layer saturation plan cache, `results/bench_short_calls_sandbox_before/`). Batching requests (default 8 when attention is not on CIM) brings calls back to a few hundred tokens.

Approximate model tokens per task (0-shot, lm-eval prompts): ARC-e ~140 per document (4 choices), PIQA ~100 (2 choices), HellaSwag ~380 (4 choices). With `--limit 1000` a run is ~0.6M model tokens; full HellaSwag alone is ~3.8M. Use `--limit` on the laptop.

## Practical settings

- `token_block: 128` (default) keeps the per-layer working set to a few hundred MB. Lower it to 64 if memory is tight.
- `--threads N` in `scripts/run_ppl.py`: set to the number of **physical** cores.
- Model RAM in float32: OPT-125M / GPT-2 ≈ 0.5 GB, Qwen2.5-0.5B ≈ 2 GB, TinyLlama-1.1B ≈ 4.5 GB, plus activations.
- Long runs print progress and ETA per window. Run them in a separate terminal and let them finish; results are written at the end.

## Install (Windows, CPU)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[llm,dev]"
git clone https://github.com/Keio-CSG/ASiM third_party/ASiM
pytest -q
```
