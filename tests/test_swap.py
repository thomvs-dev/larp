"""Layer swapping on tiny randomly initialised HF models (no downloads)."""

import pytest
import torch

transformers = pytest.importorskip("transformers")

from cimsim.config import CIMConfig, LayerConfigMap  # noqa: E402
from cimsim.core.layers import CIMLinear  # noqa: E402
from cimsim.events import EventCounter  # noqa: E402
from cimsim.models.swap import set_mode, swap_linears  # noqa: E402


def tiny(arch: str):
    torch.manual_seed(0)
    if arch == "opt":
        cfg = transformers.OPTConfig(hidden_size=64, num_hidden_layers=2, ffn_dim=128,
                                     num_attention_heads=4, vocab_size=100, max_position_embeddings=64,
                                     word_embed_proj_dim=64)
        return transformers.OPTForCausalLM(cfg)
    if arch == "gpt2":
        cfg = transformers.GPT2Config(n_embd=64, n_layer=2, n_head=4, vocab_size=100, n_positions=64)
        return transformers.GPT2LMHeadModel(cfg)
    if arch == "llama":
        cfg = transformers.LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                       num_attention_heads=4, num_key_value_heads=2, vocab_size=100,
                                       max_position_embeddings=64)
        return transformers.LlamaForCausalLM(cfg)
    raise ValueError(arch)


@pytest.mark.parametrize("arch,expected", [("opt", 12), ("gpt2", 8), ("llama", 14)])
def test_swap_fp_unchanged(arch, expected):
    model = tiny(arch).eval()
    ids = torch.randint(0, 100, (1, 16))
    with torch.no_grad():
        ref = model(ids).logits
    replaced = swap_linears(model, LayerConfigMap(default=CIMConfig(mode="fp")))
    assert len(replaced) == expected
    assert all("lm_head" not in n for n in replaced)
    with torch.no_grad():
        out = model(ids).logits
    assert torch.allclose(out, ref, atol=1e-5)


@pytest.mark.parametrize("arch", ["opt", "gpt2", "llama"])
def test_swap_quant_modes_run_and_count(arch):
    model = tiny(arch).eval()
    ids = torch.randint(0, 100, (1, 16))
    with torch.no_grad():
        ref = model(ids).logits
    counter = EventCounter()
    swap_linears(model, LayerConfigMap(default=CIMConfig(mode="fake_quant")), counter)
    with torch.no_grad():
        fq = model(ids).logits
    set_mode(model, mode="bit_accurate", adc_bits=None, nrow=32)
    with torch.no_grad():
        ba = model(ids).logits
    assert torch.allclose(fq, ba, atol=1e-3)            # ideal bit-accurate == fake-quant
    assert ((fq - ref).norm() / ref.norm()) < 0.05       # W8A8 close to FP
    assert counter.total()["macs"] > 0


def test_swap_per_layer_override():
    model = tiny("llama")
    m = LayerConfigMap.from_dict({"default": {"mode": "fake_quant"},
                                  "layers": {"*.mlp.down_proj": {"mode": "fp"}}})
    swap_linears(model, m)
    mods = dict(model.named_modules())
    assert mods["model.layers.0.mlp.down_proj"].cfg.mode == "fp"
    assert mods["model.layers.0.mlp.up_proj"].cfg.mode == "fake_quant"
    assert isinstance(mods["model.layers.1.self_attn.q_proj"], CIMLinear)


def test_perplexity_runs_on_tiny_model():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from eval.perplexity import perplexity
    model = tiny("opt")
    ids = torch.randint(0, 100, (200,))
    res = perplexity(model, ids, seq_len=32, max_windows=3, progress=False)
    assert res["windows"] == 3 and res["tokens"] == 3 * 31
    assert res["ppl"] > 1 and len(res["nll_per_window"]) == 3
