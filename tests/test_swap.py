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
    if arch == "qwen2":
        cfg = transformers.Qwen2Config(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                       num_attention_heads=4, num_key_value_heads=2, vocab_size=100,
                                       max_position_embeddings=64, tie_word_embeddings=True)
        return transformers.Qwen2ForCausalLM(cfg)
    raise ValueError(arch)


@pytest.mark.parametrize("arch,expected", [("opt", 12), ("gpt2", 8), ("llama", 14), ("qwen2", 14)])
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


@pytest.mark.parametrize("arch", ["opt", "gpt2", "llama", "qwen2"])
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


def test_calibrated_bit_accurate_on_tiny_model():
    from cimsim import calibrate_adc_range
    model = tiny("llama").eval()
    ids = torch.randint(0, 100, (1, 16))
    m = LayerConfigMap.from_dict({"default": {"mode": "bit_accurate", "nrow": 32, "adc_bits": 6,
                                              "adc_range": "calibrated", "adc_range_gran": "layer"}})
    swap_linears(model, m)
    names = calibrate_adc_range(model, lambda: model(ids))
    assert len(names) == 14
    with torch.no_grad():
        out = model(ids).logits
    assert torch.isfinite(out).all()


def test_qwen2_qkv_bias_and_tied_head_kept():
    """Qwen2 has biases on q/k/v (not o) and a tied LM head; the head stays digital and tied."""
    model = tiny("qwen2")
    swap_linears(model, LayerConfigMap(default=CIMConfig(mode="fake_quant")))
    mods = dict(model.named_modules())
    assert mods["model.layers.0.self_attn.q_proj"].bias is not None
    assert mods["model.layers.0.self_attn.o_proj"].bias is None
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()


@pytest.mark.parametrize("arch", ["llama", "qwen2"])
def test_statistical_mode_on_tiny_model(arch):
    from cimsim import calibrate_adc_range
    model = tiny(arch).eval()
    ids = torch.randint(0, 100, (1, 32))
    m = LayerConfigMap.from_dict({"default": {"mode": "statistical", "nrow": 32, "adc_bits": 6,
                                              "adc_range": "calibrated", "rand_noise_sigma": 0.05}})
    swap_linears(model, m)
    assert len(calibrate_adc_range(model, lambda: model(ids))) == 14
    with torch.no_grad():
        out = model(ids).logits
    assert torch.isfinite(out).all()


ROLE_NAMES = {
    "opt": {"qkv": "model.decoder.layers.0.self_attn.q_proj", "o": "model.decoder.layers.0.self_attn.out_proj",
            "up_gate": "model.decoder.layers.0.fc1", "down": "model.decoder.layers.0.fc2"},
    "gpt2": {"qkv": "transformer.h.0.attn.c_attn", "o": "transformer.h.0.attn.c_proj",
             "up_gate": "transformer.h.0.mlp.c_fc", "down": "transformer.h.0.mlp.c_proj"},
    "llama": {"qkv": "model.layers.0.self_attn.k_proj", "o": "model.layers.0.self_attn.o_proj",
              "up_gate": "model.layers.0.mlp.gate_proj", "down": "model.layers.0.mlp.down_proj"},
    "qwen2": {"qkv": "model.layers.1.self_attn.v_proj", "o": "model.layers.1.self_attn.o_proj",
              "up_gate": "model.layers.1.mlp.up_proj", "down": "model.layers.1.mlp.down_proj"},
}


@pytest.mark.parametrize("arch", list(ROLE_NAMES))
def test_policy_roles_resolve_per_architecture(arch):
    model = tiny(arch)
    m = LayerConfigMap.from_dict({"default": {"mode": "fake_quant"},
                                  "policy": {"qkv": "acim", "o": "dcim", "up_gate": "digital",
                                             "down": {"macro": "dcim", "w_bits": 4}},
                                  "layers": {"*.0.*fc2": {"w_bits": 6}}})
    swap_linears(model, m)
    mods = dict(model.named_modules())
    want = {"qkv": "acim", "o": "dcim", "up_gate": "digital", "down": "dcim"}
    for role, name in ROLE_NAMES[arch].items():
        assert mods[name].cfg.macro == want[role], (arch, role, name)
    down = mods[ROLE_NAMES[arch]["down"]].cfg
    assert down.w_bits == (6 if arch == "opt" else 4)          # explicit layers override wins over policy


def test_policy_rejects_unknown_role():
    with pytest.raises(ValueError):
        LayerConfigMap.from_dict({"policy": {"ffn": "dcim"}})


def test_dcim_and_digital_are_exact_and_counted_separately():
    """dcim / digital layers give the fake-quant result even in a noisy bit-accurate config and record
    no ADC conversions; acim layers in the same model still do."""
    model = tiny("llama").eval()
    ids = torch.randint(0, 100, (1, 16))
    swap_linears(model, LayerConfigMap(default=CIMConfig(mode="fake_quant")))
    with torch.no_grad():
        fq = model(ids).logits
    model = tiny("llama").eval()
    counter = EventCounter()
    m = LayerConfigMap.from_dict({"default": {"mode": "bit_accurate", "nrow": 32, "adc_bits": 4,
                                              "rand_noise_sigma": 1.0, "macro": "dcim"},
                                  "policy": {"up_gate": "digital", "down": "digital"}})
    swap_linears(model, m, counter)
    with torch.no_grad():
        out = model(ids).logits
    assert torch.allclose(out, fq, atol=1e-4)
    t = counter.total()
    assert t["adc_conversions"] == 0 and t["dcim_column_ops"] > 0 and t["digital_macs"] > 0
    per = counter.per_layer
    assert per["model.layers.0.mlp.up_proj"]["weight_bits_written"] == 0
    assert per["model.layers.0.self_attn.q_proj"]["weight_bits_written"] > 0
    set_mode(model, macro="acim")
    counter.reset()
    with torch.no_grad():
        model(ids)
    assert counter.total()["adc_conversions"] > 0


def test_events_exclude_calibration_after_reset_compute():
    from cimsim import calibrate_adc_range
    model = tiny("qwen2").eval()
    ids = torch.randint(0, 100, (1, 16))
    counter = EventCounter()
    swap_linears(model, LayerConfigMap.from_dict({"default": {"mode": "statistical", "nrow": 32, "adc_bits": 6,
                                                              "adc_range": "calibrated"}}), counter)
    calibrate_adc_range(model, lambda: model(ids))
    wbits = counter.total()["weight_bits_written"]
    counter.reset_compute()
    assert counter.total()["macs"] == 0 and counter.total()["weight_bits_written"] == wbits > 0
    with torch.no_grad():
        model(ids)
    ref = EventCounter()
    m2 = tiny("qwen2").eval()
    swap_linears(m2, LayerConfigMap(default=CIMConfig(mode="fake_quant", nrow=32, adc_bits=6)), ref)
    with torch.no_grad():
        m2(ids)
    assert counter.total() == ref.total()
