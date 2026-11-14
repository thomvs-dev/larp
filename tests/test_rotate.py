"""Hadamard rotation (A6): exact in fp, changes only quantized results, events counted."""

import pytest
import torch
import yaml

transformers = pytest.importorskip("transformers")

from cimsim.events import EventCounter  # noqa: E402
from cimsim.models.rotate import block_size, hadamard, rotate_last  # noqa: E402
from eval.runner import swap_from_config  # noqa: E402
from tests.test_swap import tiny  # noqa: E402


def test_hadamard_orthogonal_and_blocks():
    h = hadamard(8)
    assert torch.allclose(h @ h, torch.eye(8, dtype=h.dtype))
    assert block_size(768) == 256 and block_size(8960) == 256 and block_size(1536) == 512
    assert block_size(3072, cap=512) == 512
    x = torch.randn(3, 24, dtype=torch.float64)
    assert torch.allclose(rotate_last(rotate_last(x, hadamard(8)), hadamard(8)), x)


def _cfg(tmp_path, mode, roles):
    p = tmp_path / "c.yaml"
    d = {"default": {"mode": mode}}
    if roles:
        d["hadamard"] = {"roles": roles}
    p.write_text(yaml.safe_dump(d))
    return str(p)


@pytest.mark.parametrize("arch", ["opt", "gpt2", "qwen2"])
def test_rotation_exact_in_fp(arch, tmp_path):
    ids = torch.randint(0, 100, (1, 16))
    ref = tiny(arch).eval()
    with torch.no_grad():
        want = ref(ids).logits
    m = tiny(arch).eval()
    c = EventCounter()
    swap_from_config(m, _cfg(tmp_path, "fp", ["qkv", "o", "up_gate", "down"]), c)
    with torch.no_grad():
        got = m(ids).logits
    assert torch.allclose(got, want, atol=1e-4)
    assert c.total()["hadamard_adds"] > 0


def test_rotation_changes_fake_quant_and_counts_shared_input_once(tmp_path):
    ids = torch.randint(0, 100, (1, 16))
    a, b = tiny("qwen2").eval(), tiny("qwen2").eval()
    swap_from_config(a, _cfg(tmp_path, "fake_quant", None), EventCounter())
    c = EventCounter()
    swap_from_config(b, _cfg(tmp_path, "fake_quant", ["qkv"]), c)
    with torch.no_grad():
        assert not torch.allclose(a(ids).logits, b(ids).logits)
    per = c.to_dict()["per_layer"]
    assert per["model.layers.0.self_attn.q_proj"]["hadamard_adds"] == 16 * 64 * 6      # 64 -> block 64
    assert per["model.layers.0.self_attn.k_proj"]["hadamard_adds"] == 0


def test_a6_point_configs():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from a6_mitigation import point_config, run_name
    from cimsim.config import LayerConfigMap
    base = {"default": {"mode": "statistical", "nrow": 128, "adc_bits": 6, "adc_range": "calibrated"}}
    c = point_config(base, "bit_accurate", 256, 7, "down_b+1")
    m = LayerConfigMap.from_dict(c)
    assert m.for_layer("transformer.h.0.mlp.c_proj").adc_bits == 8
    assert m.for_layer("transformer.h.0.mlp.c_fc").adc_bits == 7
    assert m.for_layer("transformer.h.0.mlp.c_fc").mode == "bit_accurate"
    assert LayerConfigMap.from_dict(point_config(base, "statistical", 128, 6, "dcim_down")).for_layer(
        "model.layers.0.mlp.down_proj").macro == "dcim"
    assert point_config(base, "statistical", 128, 6, "had_down")["hadamard"] == {"roles": ["down"]}
    assert base["default"]["nrow"] == 128 and "hadamard" not in base
    assert run_name(128, 6, "down_b+1") == "n128_adc6_down_bp1"
