"""Time one CIMLinear layer in each mode.

Example: python scripts/bench_layer.py --K 768 --N 3072 --T 512 --device cuda
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cimsim import CIMConfig, CIMLinear  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=768)
    ap.add_argument("--N", type=int, default=3072)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--nrow", type=int, default=128)
    ap.add_argument("--act-enc", type=int, default=None)
    ap.add_argument("--token-block", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    lin = torch.nn.Linear(a.K, a.N).to(a.device)
    x = torch.randn(a.T, a.K, device=a.device)
    for mode in ["fp", "fake_quant", "bit_accurate"]:
        cfg = CIMConfig(mode=mode, nrow=a.nrow, adc_bits=6, act_enc=a.act_enc, token_block=a.token_block)
        layer = CIMLinear.from_linear(lin, cfg, "bench").to(a.device)
        layer(x)  # warm-up
        if a.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        layer(x)
        if a.device == "cuda":
            torch.cuda.synchronize()
        print(f"{mode:13s} T={a.T} K={a.K} N={a.N} nrow={a.nrow} enc={a.act_enc}: {time.time() - t0:.3f} s")


if __name__ == "__main__":
    main()
