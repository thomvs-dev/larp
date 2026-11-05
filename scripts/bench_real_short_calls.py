"""Real-model check of the batched exact-saturation path (A4 speed sub-task).

Loads a local model (e.g. Qwen2.5-0.5B), swaps it with a statistical config, calibrates once, then for
several call lengths times the per-entry loop path against the batched path (cimsim.stat.saturation.BATCH_T_MAX
0 vs large), compares the logits of both, and reports the size of the cached weight-bit plans.

    CIMSIM_WIKITEXT_DIR=<dir with train/test.parquet> python scripts/bench_real_short_calls.py \
        --model <model dir> --name bench_real_short_calls_qwen05b_sandbox --threads 2

Writes results/<name>/result.json."""
from __future__ import annotations

import argparse
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", default="config/experiments/a4_stat_adc6cal.yaml")
    ap.add_argument("--name", required=True)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--calib-tokens", type=int, default=1024)
    ap.add_argument("--lengths", default="64,128,256,512")
    ap.add_argument("--repeats", type=int, default=6)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    model, tok, _ = load_model(args.model, "cpu")
    model.eval()
    swap_from_config(model, str(ROOT / args.config), EventCounter())
    _, calib_s = calibrate_if_needed(model, tok, args.calib_tokens, args.calib_tokens, "cpu")
    print("calibrated", round(calib_s, 1), "s", flush=True)
    ids = load_wikitext2_ids(tok, split="test")
    cim = [m for m in model.modules() if isinstance(m, _CIMBase)]
    res = {"git": git_hash(), "model": args.model, "config": args.config, "threads": args.threads,
           "calib_tokens": args.calib_tokens, "calib_seconds": round(calib_s, 1), "ms_per_token": {},
           "max_abs_logit_diff": {}}
    with torch.no_grad():
        model(ids[:64].unsqueeze(0))                                   # builds the per-layer plans
        plans = [m._sat_plan[1] for m in cim if getattr(m, "_sat_plan", None) and m._sat_plan[1]]
        res["plans"] = {"layers_with_plan": len(plans), "layers": len(cim),
                        "cached": sum(1 for p in plans if p["cached"]),
                        "entries": sum(len(p["entries"]) for p in plans),
                        "cache_MB": round(sum(p["cache_bytes"] for p in plans) / 1e6, 1)}
        print(res["plans"], flush=True)
        for L in (int(v) for v in args.lengths.split(",")):
            row = {}
            logits = {}
            for label, tmax in (("loop", 0), ("batched", 10 ** 6)):
                sat.BATCH_T_MAX = tmax
                model(ids[:L].unsqueeze(0))                            # warm-up
                t = time.time()
                for i in range(args.repeats):
                    out = model(ids[i * L:(i + 1) * L].unsqueeze(0)).logits
                    if i == 0:
                        logits[label] = out.clone()
                row[label] = round(1000 * (time.time() - t) / (args.repeats * L), 2)
            res["ms_per_token"][str(L)] = row
            res["max_abs_logit_diff"][str(L)] = float((logits["loop"] - logits["batched"]).abs().max())
            print(L, row, res["max_abs_logit_diff"][str(L)], flush=True)
    d = ROOT / "results" / args.name
    d.mkdir(parents=True, exist_ok=True)
    (d / "result.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
