import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from a5_outliers import layer_stats, role_of  # noqa: E402


def test_layer_stats_outlier_channel_and_planes():
    torch.manual_seed(0)
    x = torch.randn(256, 64)
    x[:, 3] *= 20.0                                    # one outlier channel
    s = layer_stats(x)
    assert s["outlier_channels_6x"] == 1
    assert s["top1pct_channel_energy_share"] > 0.8
    assert abs(s["code_negative_frac"] - s["plane_density"][-1]) < 1e-12   # sign plane set iff code < 0
    r = layer_stats(torch.relu(torch.randn(128, 32)))
    assert r["code_negative_frac"] == 0.0 and r["plane_density"][-1] == 0.0


def test_role_of_names():
    assert role_of("model.decoder.layers.0.self_attn.q_proj") == "qkv"
    assert role_of("model.decoder.layers.0.fc2") == "down"
    assert role_of("lm_head") is None
