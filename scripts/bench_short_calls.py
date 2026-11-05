"""ms / token of fake_quant and statistical mode vs call length (tokens per forward call), on a randomly
initialised Qwen2-0.5B-shaped model with 4 layers (no download). Zero-shot requests are short (30-150
tokens), perplexity windows are 2,048 tokens; this shows the per-call overhead.

    python scripts/bench_short_calls.py --threads 2 --name bench_short_calls_sandbox

Writes results/<name>/result.json. Random weights: the exact-saturation work differs from a trained
model, so use the numbers as ratios between call lengths, not as absolute model speeds.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cimsim import calibrate_adc_range  # noqa: E402
from cimsim.config import LayerConfigMap  # noqa: E402
from cimsim.models.swap import swap_linears  # noqa: E402
from eval.runner import git_hash  # noqa: E402


def main() -> None:
    import transformers
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--lengths", default="64,512,2048")
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(hidden_size=896, intermediate_size=4864, num_hidden_layers=args.layers,
                                   num_attention_heads=14, num_key_value_heads=2, vocab_size=151936,
                                   max_position_embeddings=4096, tie_word_embeddings=True)
    lengths = [int(v) for v in args.lengths.split(",")]
    res = {"git": git_hash(), "threads": args.threads, "layers": args.layers, "ms_per_token": {}}
    ids = torch.randint(0, 151936, (1, 4096))
    for mode in ("fake_quant", "statistical"):
        model = transformers.Qwen2ForCausalLM(cfg).eval()
        swap_linears(model, LayerConfigMap.from_dict({"default": {
            "mode": mode, "nrow": 128, "adc_bits": 6,
            "adc_range": "calibrated" if mode == "statistical" else "full"}}))
        if mode == "statistical":
            calibrate_adc_range(model, lambda: model(ids[:, :1024]))
        out = {}
        with torch.no_grad():
            model(ids[:, :64])                                         # warm-up
            for L in lengths:
                n = max(1, 4096 // L)
                t = time.time()
                for i in range(n):
                    model(ids[:, i * L % 2048:i * L % 2048 + L] if L < 2048 else ids[:, :L])
                out[str(L)] = round(1000 * (time.time() - t) / (n * L), 2)
        res["ms_per_token"][mode] = out
        print(mode, out, flush=True)
    d = ROOT / "results" / args.name
    d.mkdir(parents=True, exist_ok=True)
    (d / "result.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
