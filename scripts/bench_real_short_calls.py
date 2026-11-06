"""Real-model timing of statistical mode for zero-shot and perplexity call shapes (A4 speed sub-task).

Loads a local model (e.g. Qwen2.5-0.5B), swaps it with a statistical config and calibrates once (or loads a
model saved with --save). Then, for each thread count and each call shape BxL (batch x tokens), times the
forward pass with the default path choice and with the per-entry loop path forced; for shapes up to
BATCH_T_MAX tokens also with the batched path forced, and checks that loop and batched give identical logits
(SHA-256 of the logit bytes). Finally times the same calls with every CIM layer computing ideally
(fake-quant floor).

    CIMSIM_WIKITEXT_DIR=<dir with train/test.parquet> python scripts/bench_real_short_calls.py \\
        --model <model dir> --name bench_real_short_calls_qwen05b_sandbox --threads 2

Writes results/<name>/result.json."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cimsim.stat.saturation as sat  # noqa: E402
from cimsim.core.layers import _CIMBase  # noqa: E402
from cimsim.events import EventCounter  # noqa: E402
from eval.perplexity import load_wikitext2_ids  # noqa: E402
from eval.runner import calibrate_if_needed, git_hash, load_model, swap_from_config  # noqa: E402


def _time(model, x, repeats, keep=False):
    """ms per token over `repeats` calls; with keep, also the SHA-256 of the last call's logits (exact
    identity check without holding the logits: 8 x 110 logits of a 152k-word vocabulary are 535 MB)."""
    model(x)                                                           # warm-up
    t = time.time()
    digest = None
    for i in range(repeats):
        out = model(x).logits
        if keep and i == repeats - 1:
            ms = 1000 * (time.time() - t) / (repeats * x.numel())
            digest = hashlib.sha256(out.contiguous().numpy().tobytes()).hexdigest()
            return ms, digest
        del out
    return 1000 * (time.time() - t) / (repeats * x.numel()), digest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", default="config/experiments/a4_stat_adc6cal.yaml")
    ap.add_argument("--name", required=True)
    ap.add_argument("--threads", default="2", help="comma-separated thread counts, e.g. 14,8,6,4")
    ap.add_argument("--calib-tokens", type=int, default=1024)
    ap.add_argument("--shapes", default="1x64,8x110,1x2048", help="comma-separated BxL call shapes")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--save", default=None, help="save the calibrated model here (torch.save) for --load")
    ap.add_argument("--load", default=None, help="load a model saved with --save instead of calibrating")
    args = ap.parse_args()
    threads = [int(v) for v in args.threads.split(",")]
    torch.set_num_threads(threads[0])
    counter = EventCounter()
    if args.load:
        model, tok = torch.load(args.load, weights_only=False)
        calib_s = None
    else:
        model, tok, _ = load_model(args.model, "cpu")
        model.eval()
        swap_from_config(model, str(ROOT / args.config), counter)
        _, calib_s = calibrate_if_needed(model, tok, args.calib_tokens, args.calib_tokens, "cpu")
        print("calibrated", round(calib_s, 1), "s", flush=True)
        if args.save:
            for m in model.modules():                                  # the counter holds lambdas (not picklable)
                if hasattr(m, "counter"):
                    m.counter = None
            torch.save((model, tok), args.save)
    for m in model.modules():
        if hasattr(m, "counter"):
            m.counter = counter
    gc.collect()
    ids = load_wikitext2_ids(tok, split="test")
    cim = [m for m in model.modules() if isinstance(m, _CIMBase)]
    res = {"git": git_hash(), "model": args.model, "config": args.config, "threads": threads,
           "calib_tokens": args.calib_tokens, "calib_seconds": None if calib_s is None else round(calib_s, 1), "loaded": args.load,
           "batch_t_max": sat.BATCH_T_MAX, "ms_per_token": {}, "paths_identical": {}}
    default_tmax = sat.BATCH_T_MAX
    with torch.no_grad():
        model(ids[:64].unsqueeze(0))                                   # builds the per-layer plans
        plans = [m._sat_plan[1] for m in cim if getattr(m, "_sat_plan", None) and m._sat_plan[1]]
        res["plans"] = {"layers_with_plan": len(plans), "layers": len(cim),
                        "cached": sum(1 for p in plans if p["cached"]),
                        "entries": sum(len(p["entries"]) for p in plans),
                        "cache_MB": round(sum(p["cache_bytes"] for p in plans) / 1e6, 1)}
        print(res["plans"], flush=True)
        for n in threads:
            torch.set_num_threads(n)
            for shape in args.shapes.split(","):
                B, L = (int(v) for v in shape.split("x"))
                x = ids[:B * L].view(B, L)
                row, ref, same = {}, None, None
                do_batched = B * L <= default_tmax                     # above it the default is the loop path
                for label, tmax in (("default", default_tmax), ("loop", 0), ("batched", 10 ** 9)):
                    if label == "batched" and not do_batched:
                        continue
                    sat.BATCH_T_MAX = tmax
                    row[label], out = _time(model, x, args.repeats, keep=do_batched and label != "default")
                    if label == "loop":
                        ref = out
                    elif label == "batched":
                        same = ref == out
                ref = None
                sat.BATCH_T_MAX = default_tmax
                for m in cim:
                    m._ideal_override = True
                row["fake_quant"], _ = _time(model, x, args.repeats)
                for m in cim:
                    m._ideal_override = False
                row = {k: round(v, 2) for k, v in row.items()}
                res["ms_per_token"][f"{n}t_{shape}"] = row
                if same is not None:
                    res["paths_identical"][f"{n}t_{shape}"] = same
                print(n, "threads", shape, row, res["paths_identical"].get(f"{n}t_{shape}"), flush=True)
    d = ROOT / "results" / args.name
    d.mkdir(parents=True, exist_ok=True)
    (d / "result.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
