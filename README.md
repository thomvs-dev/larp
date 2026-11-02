# cimsim — SRAM CIM simulator for LLMs

Simulates SRAM compute-in-memory execution of transformer / LLM layers: quantization, bit-plane decomposition, row-parallel analog MAC, ADC quantization, analog noise, and event counts for later energy modelling.

Plan and status: `CLAUDE.md`. Design and maths: `docs/part_a_plan.md`.

## Git

Code lives at https://github.com/thomvs-dev/larp (branch `main`). To link an existing `D:\japan` folder to it without touching your files:

```bash
git init -b main
git remote add origin https://github.com/thomvs-dev/larp.git
git fetch origin
git reset --mixed origin/main     # adopt the remote history; working files are kept
git branch -u origin/main
git status                       # should show no differences
```

After that: `git pull` to get agent changes; `git add results; git commit -m "..."; git push` to share new results.

If the agent has already committed result files that also exist (untracked) in your folder, `git pull` stops with "untracked working tree files would be overwritten". The committed copies are byte-identical to yours, so take them with:

```bash
git fetch origin
git reset --hard origin/main      # only safe when you have no uncommitted code edits (check `git status` first)
```

## Setup (CPU)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[llm,dev]"
git clone https://github.com/Keio-CSG/ASiM third_party/ASiM            # reference for equivalence tests (Apache 2.0); ASiM/ at repo root also works
pytest -q
```

The project runs CPU-only. Speeds, run budgets and the evaluation protocol are in `docs/cpu_guide.md`.

## Modes

| mode | what runs |
|---|---|
| `fp` | original layer |
| `fake_quant` | integer quantization, normal matmul (ideal digital reference) |
| `bit_accurate` | full bit-plane / ADC / noise simulation (ASiM-compatible) |
| `statistical` | calibrated error injection (milestone A3, not implemented yet) |

## A0 baselines (E0)

```bash
python scripts/run_ppl.py --model facebook/opt-125m --config none --name e0_opt125m_fp32_s2048
python scripts/run_ppl.py --model facebook/opt-125m --config config/experiments/e0_w8a8.yaml --name e0_opt125m_w8a8
python scripts/run_ppl.py --model facebook/opt-125m --config config/experiments/e0_w8a8_per_tensor.yaml --name e0_opt125m_w8a8_pt
python scripts/run_ppl.py --model facebook/opt-125m --config config/experiments/e0_w6a6.yaml --name e0_opt125m_w6a6
python scripts/run_ppl.py --model facebook/opt-125m --config config/experiments/e0_w4a8.yaml --name e0_opt125m_w4a8
# repeat with --model gpt2 (names e0_gpt2_*)
```

Window length defaults to the model's context length, capped at 2048 (OPT: 2048, GPT-2: 1024).
Reference check: OPT-125M FP32 gives 27.66 (published 27.65).

Results go to `results/<name>/` (result.json, config.yaml, events.json).

Add `--threads N` (N = physical cores) to any command. Each E0 run takes a few minutes on CPU.

## Bit-accurate run (slow on CPU — use the 8k-token subset)

```bash
# reference on the same subset
python scripts/run_ppl.py --model facebook/opt-125m --config config/experiments/e0_w8a8.yaml --name sub8k_opt125m_w8a8 --max-tokens 8192
# bit-accurate (first 8,192 tokens)
python scripts/run_ppl.py --model facebook/opt-125m --config config/experiments/example_bit_accurate.yaml \
    --name sub8k_opt125m_ba_example --max-tokens 8192
python scripts/bench_layer.py --K 768 --N 3072 --T 128      # time one layer
```

Set `backend: reference` in a config to use the slower executor that mirrors ASiM's loop structure (results agree with `fast` to ~1e-6).

## ADC range calibration (finding F5)

```bash
# same 8,192 tokens as the F3 collapse run, ADC full scale calibrated on 2,048 WikiText-2 *train* tokens
python scripts/run_ppl.py --model facebook/opt-125m --config config/experiments/f5_adc6_calibrated.yaml --name f5_opt125m_adc6_cal_layer --max-tokens 8192
python scripts/run_ppl.py --model facebook/opt-125m --config config/experiments/f5_adc6_calibrated_cycle_plane.yaml --name f5_opt125m_adc6_cal_cp --max-tokens 8192
```

## A2: ResNet-20 vs ASiM (CIFAR-10)

```bash
pip install pillow                     # needed by `datasets` to decode CIFAR-10 images
python validation/a2_asim_compare.py --name a2_smoke --configs quick --n-images 200      # ~5 min
python validation/a2_asim_compare.py --name a2_resnet20 --n-images 1000                 # 13 configs, 1-2 h (ASiM is ~10x slower)
```

Results: `results/<name>/results.json` and `results/<name>/table.md`.

## Use in code

```python
from cimsim import CIMConfig, LayerConfigMap, EventCounter
from cimsim.models.swap import swap_linears

counter = EventCounter()
cfg = LayerConfigMap.from_yaml("config/experiments/example_bit_accurate.yaml")
swap_linears(model, cfg, counter)        # Linear, GPT-2 Conv1D and Conv2d; LM head stays digital
# if any layer has adc_range: calibrated
from cimsim import calibrate_adc_range
calibrate_adc_range(model, lambda: model(calib_ids))
out = model(input_ids)
print(counter.total())
```
