"""The decode-trace check (scripts/b3_decode_trace.py) on a tiny random Qwen2: counts derived from the
model shape equal the counted events of a prefill and of cached decode steps."""

import importlib.util
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")

from cimsim.events import EventCounter  # noqa: E402
from cimsim.system import ModelShape  # noqa: E402
from eval.runner import swap_from_config  # noqa: E402

ROOT = Path(__file__).parents[1]
SHAPE = ModelShape(name="tiny", n_layers=2, d_model=64, n_heads=4, n_kv_heads=2, head_dim=16, d_ff=160, vocab=97)


def tiny():
    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(hidden_size=64, intermediate_size=160, num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2, vocab_size=97, max_position_embeddings=128)
    return transformers.Qwen2ForCausalLM(cfg).eval()


def load_script():
    spec = importlib.util.spec_from_file_location("b3_decode_trace", ROOT / "scripts/b3_decode_trace.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("config,attention", [("b3_trace_fq_n128.yaml", False), ("b3_trace_fq_n128_attn.yaml", True)])
def test_trace_equals_derived_counts(config, attention, tmp_path):
    import yaml
    m = load_script()
    cfg = yaml.safe_load((ROOT / "config/experiments" / config).read_text())
    cfg["default"]["nrow"] = 24                      # several row chunks on a 64-wide model
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    model, counter = tiny(), EventCounter()
    swap_from_config(model, str(p), counter)
    ids = torch.randint(0, 97, (1, 40))
    pre, rows = m.trace(model, ids, 30, 10, counter)
    res = m.compare(SHAPE, 24, attention, 30, 10, pre, rows)
    assert res["all_equal"], [c for c in res["checks"] if not c["equal"]][:2]
