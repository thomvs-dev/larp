"""Zero-shot accuracy (lm-eval-harness) for one model under one CIM config (A4).

Example:
    python scripts/run_zeroshot.py --model Qwen/Qwen2.5-0.5B --config config/experiments/a4_stat_adc6cal.yaml \
        --name zs_qwen05b_stat_adc6cal --limit 500 --threads 14

The model is swapped and (if a layer has adc_range: calibrated) calibrated on WikiText-2 train tokens
exactly as in scripts/run_ppl.py, then every task runs in the same process (one calibration per run).
Writes results/<name>/{zeroshot.json, zs_<task>.json, events_<task>.json, config.yaml,
adc_full_scale.json}. An interrupted run restarts where it stopped (tasks already written are skipped).
Compare runs with scripts/zeroshot_compare.py (paired, same documents).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cimsim.events import EventCounter  # noqa: E402
from eval.runner import calibrate_if_needed, context_len, git_hash, load_model, swap_from_config  # noqa: E402
from eval.zeroshot import DEFAULT_TASKS, evaluate_tasks, has_cim_attention  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", required=True, help="YAML path, or 'none' for the unmodified model")
    ap.add_argument("--name", required=True)
    ap.add_argument("--tasks", default=",".join(DEFAULT_TASKS), help="comma-separated lm-eval task names")
    ap.add_argument("--limit", type=int, default=None,
                    help="first N documents per task (same N -> same documents in every run)")
    ap.add_argument("--num-fewshot", type=int, default=0)
    ap.add_argument("--include-path", default=None, help="extra lm-eval task YAML folder")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--calib-tokens", type=int, default=2048,
                    help="WikiText-2 train tokens for ADC range calibration (only if needed)")
    ap.add_argument("--calib-seq-len", type=int, default=None,
                    help="calibration window; default = model context, capped at 2048 (as run_ppl.py)")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="default: 1 if attention runs on CIM, else 8 (see eval/zeroshot.py)")
    ap.add_argument("--no-resume", action="store_true", help="rerun tasks that already have results")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    out = ROOT / "results" / args.name

    t_load = time.time()
    model, tok, dtype = load_model(args.model, args.device, args.dtype)
    counter = EventCounter()
    cfg_dict, replaced = swap_from_config(model, args.config, counter)
    ctx = context_len(model)
    cal_len = min(ctx, 2048) if args.calib_seq_len is None else min(args.calib_seq_len, ctx)
    calibrated, calib_s = calibrate_if_needed(model, tok, cal_len, args.calib_tokens, args.device)
    load_s = round(time.time() - t_load - calib_s, 1)

    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg_dict or {"fp": True}), encoding="utf-8")
    if calibrated:
        fs = {m.name: m.adc_fs.tolist() for m in model.modules() if getattr(m, "adc_fs", None) is not None}
        (out / "adc_full_scale.json").write_text(json.dumps(fs), encoding="utf-8")

    bs = args.batch_size or (1 if has_cim_attention(model) else 8)
    info = {"model": args.model, "config": args.config, "git": git_hash(), "threads": torch.get_num_threads(),
            "dtype": str(dtype), "replaced_layers": len(replaced), "calibrated_layers": len(calibrated),
            "calib_tokens": args.calib_tokens if calibrated else 0, "calib_seq_len": cal_len if calibrated else 0,
            "calib_seconds": calib_s, "load_seconds": load_s, "batch_size": bs}
    summaries = evaluate_tasks(model, tok, tasks, out, counter=counter, limit=args.limit,
                               num_fewshot=args.num_fewshot, include_path=args.include_path,
                               run_info=info, resume=not args.no_resume, batch_size=bs)
    res = {**info, "limit": args.limit, "num_fewshot": args.num_fewshot, "tasks": summaries}
    (out / "zeroshot.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
