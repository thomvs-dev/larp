"""A5 / F13: statistical vs bit-accurate layer error as a function of the calibrated ADC step.

One synthetic 768 -> 256 layer (outlier input channels as in tests/test_stat_saturation.py), full scale
forced to a given value. Step = FS / (2^b - 1) MAC units. Writes results/<name>/result.json.

    python scripts/a5_step_check.py --name a5_step_check_sandbox
"""
import argparse, json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cimsim.core.layers as layers_mod  # noqa: E402
from cimsim.config import CIMConfig  # noqa: E402
from cimsim.core.layers import CIMLinear  # noqa: E402
from cimsim.stat.saturation import rounding_exact  # noqa: E402
from eval.runner import git_hash  # noqa: E402

BASE = dict(w_bits=8, x_bits=8, signed_act=True, w_gran="per_channel", x_gran="per_token")
POINTS = [(256, 6, 63), (256, 6, 81), (256, 6, 83), (256, 6, 126), (256, 7, 127), (256, 7, 132), (256, 7, 152),
          (128, 6, 63), (128, 6, 70), (64, 6, 21)]


def _x(T, K, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(T, K, generator=g)
    x[:, :4] *= 25
    return x


def run(orig, nrow, bits, fs, K=768, N=256):
    layers_mod.full_scale_from_histogram = lambda h, cfg: torch.full_like(orig(h, cfg), float(fs))
    torch.manual_seed(0)
    lin = torch.nn.Linear(K, N)
    cfg = dict(BASE, nrow=nrow, adc_bits=bits, adc_range="calibrated")
    st = CIMLinear.from_linear(lin, CIMConfig(mode="statistical", **cfg), name="L")
    st.start_calibration()
    st(_x(256, K, 1))
    st.finish_calibration()
    ba = CIMLinear.from_linear(lin, CIMConfig(mode="bit_accurate", **cfg), name="L")
    ba.adc_fs = st.adc_fs
    fq = CIMLinear(ba.weight, ba.bias, ba.cfg.replace(mode="fake_quant"), name="F")
    x = _x(256, K, 2)
    yi = fq(x).double()
    es, eb = st(x).double() - yi, ba(x).double() - yi
    return {"nrow": nrow, "adc_bits": bits, "full_scale": fs, "step_mac": fs / (2 ** bits - 1),
            "rounding_exact": rounding_exact(st.cfg, st.adc_fs),
            "err_over_signal_ba": float(eb.norm() / yi.norm()), "err_over_signal_stat": float(es.norm() / yi.norm()),
            "stat_minus_ba_over_signal": float((es - eb).norm() / yi.norm())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="a5_step_check")
    args = ap.parse_args()
    orig = layers_mod.full_scale_from_histogram
    rows = [run(orig, *p) for p in POINTS]
    out = ROOT / "results" / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps({"git": git_hash(), "seed": 0, "layer": "768x256 synthetic",
                                                 "tokens": 256, "points": rows}, indent=2), encoding="utf-8")
    for r in rows:
        print(f"nrow {r['nrow']:3d} adc {r['adc_bits']} FS {r['full_scale']:4} step {r['step_mac']:.3f} "
              f"exact {r['rounding_exact']!s:5}  err/signal BA {r['err_over_signal_ba']:.4f} "
              f"stat {r['err_over_signal_stat']:.4f}  |stat-BA|/signal {r['stat_minus_ba_over_signal']:.4f}")


if __name__ == "__main__":
    main()
