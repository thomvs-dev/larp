"""A2 / E1: run the same pretrained CIFAR-10 ResNet-20 through ASiM and cimsim
with matched configurations and compare top-1 accuracy.

Example (Thomas's machine; downloads ResNet-20 weights from GitHub and CIFAR-10 from HuggingFace):
    python validation/a2_asim_compare.py --name a2_resnet20 --n-images 1000
    python validation/a2_asim_compare.py --name a2_smoke --configs quick --n-images 200
Smoke test without downloads:
    python validation/a2_asim_compare.py --name a2_synth --synthetic --configs quick --n-images 20

Conventions (both simulators): all conv layers except the stem conv run on the
simulated macro; stem conv, BatchNorm, ReLU, pooling and the classifier stay in
float. Weights and activations are quantized per tensor (ASiM), activations
unsigned (post-ReLU). Batches are identical for both simulators because ASiM's
activation scale is the max over the batch.

Writes results/<name>/results.json (updated after every run) and
results/<name>/table.md.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cimsim import CIMConfig  # noqa: E402
from cimsim.models.cifar_resnet import CIFAR10_MEAN, CIFAR10_STD, load_pretrained_resnet20, resnet20  # noqa: E402
from validation.asim_bridge import build_asim, build_cimsim  # noqa: E402

BASE = dict(w_bits=8, x_bits=8, signed_act=False, w_gran="per_tensor", x_gran="per_tensor")

CONFIGS = {
    # name: CIMConfig fields (on top of BASE)
    "ideal_w8a8":        dict(mode="fake_quant"),
    "bs_n128_adc8":      dict(mode="bit_accurate", nrow=128, adc_bits=8),
    "bs_n128_adc6":      dict(mode="bit_accurate", nrow=128, adc_bits=6),
    "bs_n128_adc5":      dict(mode="bit_accurate", nrow=128, adc_bits=5),
    "bs_n128_adc4":      dict(mode="bit_accurate", nrow=128, adc_bits=4),
    "bs_n255_adc6":      dict(mode="bit_accurate", nrow=255, adc_bits=6),
    "bs_n255_adc8":      dict(mode="bit_accurate", nrow=255, adc_bits=8),
    "bs_n128_adc8_rn1":  dict(mode="bit_accurate", nrow=128, adc_bits=8, rand_noise_sigma=1.0),
    "bs_n128_adc8_rn2":  dict(mode="bit_accurate", nrow=128, adc_bits=8, rand_noise_sigma=2.0),
    "bs_n128_adc8_nl2":  dict(mode="bit_accurate", nrow=128, adc_bits=8, non_linear_sigma=2.0),
    "bp2_n128_adc6":     dict(mode="bit_accurate", nrow=128, adc_bits=6, act_enc=2),
    "bp2_n128_adc8":     dict(mode="bit_accurate", nrow=128, adc_bits=8, act_enc=2),
    # ASiM skips the last channel group of 64-channel 3x3 layers at nrow 64 (docs/findings.md F4):
    "bs_n64_adc8":       dict(mode="bit_accurate", nrow=64, adc_bits=8),
}
QUICK = ["ideal_w8a8", "bs_n128_adc8", "bs_n128_adc6"]


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def load_cifar10_test(n: int, synthetic: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """First n images of the CIFAR-10 test split, normalised, [n,3,32,32] float32, labels [n]."""
    if synthetic:
        g = torch.Generator().manual_seed(0)
        return torch.randn(n, 3, 32, 32, generator=g), torch.randint(0, 10, (n,), generator=g)
    import numpy as np
    from datasets import load_dataset
    ds = load_dataset("uoft-cs/cifar10", split="test")
    ds = ds.select(range(min(n, len(ds))))
    imgs = np.stack([np.asarray(im.convert("RGB"), dtype=np.uint8) for im in ds["img"]])
    x = torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)
    return (x - mean) / std, torch.tensor(ds["label"])


@torch.no_grad()
def accuracy(model, x, y, batch) -> float:
    correct = 0
    for i in range(0, len(x), batch):
        correct += (model(x[i:i + batch]).argmax(1) == y[i:i + batch]).sum().item()
    return 100.0 * correct / len(x)


def write_table(res: dict, out: Path) -> None:
    """results/<name>/table.md (UTF-8, ASCII content so it also prints on a Windows console)."""
    lines = [f"# A2 / E1 - {res['model']}, {res['n_images']} CIFAR-10 test images", "",
             f"FP32: {res['fp32_acc']:.2f}%. Git {res['git'][:8]}. Stem conv and classifier in float.", "",
             "| Config | cimsim acc (%) | ASiM acc (%) | diff (cimsim - ASiM) | cimsim s | ASiM s |",
             "|---|---|---|---|---|---|"]
    for cname, r in res["runs"].items():
        a = r.get("cimsim", {}).get("acc")
        b = r.get("asim", {}).get("acc")
        d = f"{a - b:+.2f}" if a is not None and b is not None else ""
        fa = f"{a:.2f}" if a is not None else ""
        fb = f"{b:.2f}" if b is not None else ""
        lines.append(f"| {cname} | {fa} | {fb} | {d} | {r.get('cimsim', {}).get('seconds', '')} | "
                     f"{r.get('asim', {}).get('seconds', '')} |")
    (out / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--n-images", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--sim", choices=["both", "cimsim", "asim"], default="both")
    ap.add_argument("--configs", default="default", help="'default', 'quick', or comma-separated names")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--synthetic", action="store_true", help="random weights and images (no downloads)")
    ap.add_argument("--table-only", action="store_true",
                    help="rebuild results/<name>/table.md from an existing results.json and exit")
    args = ap.parse_args()
    out = ROOT / "results" / args.name
    if args.table_only:
        write_table(json.loads((out / "results.json").read_text(encoding="utf-8")), out)
        print((out / "table.md").read_text(encoding="utf-8"))
        return
    if args.threads:
        torch.set_num_threads(args.threads)

    names = list(CONFIGS) if args.configs == "default" else QUICK if args.configs == "quick" \
        else [c.strip() for c in args.configs.split(",")]
    model = resnet20().eval() if args.synthetic else load_pretrained_resnet20()
    x, y = load_cifar10_test(args.n_images, args.synthetic)

    out.mkdir(parents=True, exist_ok=True)
    res = {"model": "resnet20 (chenyaofo, CIFAR-10)", "n_images": len(x), "batch": args.batch,
           "git": git_hash(), "threads": torch.get_num_threads(), "synthetic": args.synthetic,
           "base": BASE, "fp32_acc": accuracy(model, x, y, args.batch), "runs": {}}
    print(f"FP32 accuracy on {len(x)} images: {res['fp32_acc']:.2f}%", flush=True)

    sims = ["cimsim", "asim"] if args.sim == "both" else [args.sim]
    for cname in names:
        cfg = CIMConfig(**{**BASE, **CONFIGS[cname], "seed": args.seed})
        res["runs"][cname] = {"config": CONFIGS[cname]}
        for sim in sims:
            torch.manual_seed(args.seed)                       # ASiM draws noise from the global RNG
            m = build_cimsim(model, cfg) if sim == "cimsim" else build_asim(model, cfg)
            t0 = time.time()
            acc = accuracy(m, x, y, args.batch)
            dt = time.time() - t0
            res["runs"][cname][sim] = {"acc": acc, "seconds": round(dt, 1)}
            print(f"{cname:20s} {sim:7s} acc {acc:6.2f}%  ({dt:.0f} s)", flush=True)
            (out / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    write_table(res, out)
    print((out / "table.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
