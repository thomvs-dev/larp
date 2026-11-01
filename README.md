# cimsim — SRAM CIM simulator for LLMs

Simulates SRAM compute-in-memory execution of transformer / LLM layers: quantization, bit-plane decomposition, row-parallel analog MAC, ADC quantization, analog noise, and event counts for later energy modelling.

Plan and status: `CLAUDE.md`. Design and maths: `docs/part_a_plan.md`.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # pick the CUDA build for your GPU
pip install -e ".[llm,dev]"
git clone https://github.com/Keio-CSG/ASiM third_party/ASiM            # reference for equivalence tests (Apache 2.0)
pytest -q
```

## Modes

| mode | what runs |
|---|---|
| `fp` | original layer |
| `fake_quant` | integer quantization, normal matmul (ideal digital reference) |
| `bit_accurate` | full bit-plane / ADC / noise simulation (ASiM-compatible) |
| `statistical` | calibrated error injection (milestone A3, not implemented yet) |

## A0 baselines (E0)

```bash
python scripts/run_ppl.py --model facebook/opt-125m --config none --name e0_opt125m_fp16
python scripts/run_ppl.py --model facebook/opt-125m --config config/experiments/e0_w8a8.yaml --name e0_opt125m_w8a8
python scripts/run_ppl.py --model facebook/opt-125m --config config/experiments/e0_w8a8_per_tensor.yaml --name e0_opt125m_w8a8_pt
python scripts/run_ppl.py --model facebook/opt-125m --config config/experiments/e0_w6a6.yaml --name e0_opt125m_w6a6
python scripts/run_ppl.py --model facebook/opt-125m --config config/experiments/e0_w4a8.yaml --name e0_opt125m_w4a8
# repeat with --model gpt2 (names e0_gpt2_*)
```

Results go to `results/<name>/` (result.json, config.yaml, events.json).

## Bit-accurate run (slow — use few windows)

```bash
python scripts/run_ppl.py --model facebook/opt-125m --config config/experiments/example_bit_accurate.yaml \
    --name try_bitacc --max-windows 2
python scripts/bench_layer.py --K 768 --N 3072 --T 512      # time one layer
```

## Use in code

```python
from cimsim import CIMConfig, LayerConfigMap, EventCounter
from cimsim.models.swap import swap_linears

counter = EventCounter()
cfg = LayerConfigMap.from_yaml("config/experiments/example_bit_accurate.yaml")
swap_linears(model, cfg, counter)        # LM head stays digital by default
out = model(input_ids)
print(counter.total())
```
