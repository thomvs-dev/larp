"""Run WikiText-2 perplexity for one model under one CIM config.

Example:
    python scripts/run_ppl.py --model facebook/opt-125m \
        --config config/experiments/e0_w8a8.yaml --name e0_opt125m_w8a8

Writes results/<name>/{result.json, config.yaml, events.json}.
The unmodified model (no CIM swap) is selected with --config none.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cimsim.events import EventCounter  # noqa: E402
from cimsim.stat.saturation import rounding_exact  # noqa: E402
from eval.perplexity import load_wikitext2_ids, perplexity  # noqa: E402
from eval.runner import calibrate_if_needed, context_len, git_hash, load_model, swap_from_config  # noqa: E402


def free_saturation_plans(model) -> int:
    """Statistical mode keeps one saturation plan per layer (candidate (plane, column) index lists per chunk
    and cycle, plus a weight-bit cache). When most columns can saturate these reach ~0.1 GB per layer; on
    Qwen2.5-1.5B (196 layers) the process ran out of memory. This drops the plan after every layer call and
    switches the weight-bit cache off. Outputs are unchanged (the plan is rebuilt from the weights)."""
    import cimsim.stat.saturation as sat
    sat.PLAN_CACHE_MB = 0.0

    def _drop(mod, _inp, _out):
        mod._sat_plan = None

    n = 0
    for m in model.modules():
        if hasattr(m, "_sat_plan"):
            m.register_forward_hook(_drop)
            n += 1
    return n


def main() -> None:
    faulthandler.enable()                # Python stack of every thread on a native crash (access violation)
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", required=True, help="YAML path, or 'none' for the unmodified model")
    ap.add_argument("--name", required=True)
    ap.add_argument("--seq-len", type=int, default=None,
                    help="window length; default = model context length, capped at 2048")
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="evaluate only the first N tokens (CPU subset: 8192); sets max-windows = N // seq-len")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32", choices=["float16", "bfloat16", "float32"],
                    help="model dtype; float16 is switched to float32 on CPU")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads (default: torch default)")
    ap.add_argument("--no-calib-cache", action="store_true",
                    help="do not read or write the calibration histogram cache (cimsim/calib_cache.py)")
    ap.add_argument("--free-sat-plans", action="store_true",
                    help="statistical mode: drop each layer's saturation plan after its forward call and do not "
                         "cache weight bits. Same outputs, slower per window; needed when most columns saturate "
                         "on a large model (plans of all layers together do not fit in memory)")
    ap.add_argument("--window-checkpoint", action="store_true",
                    help="save per-window NLLs in results/<name>/windows.json and continue from them after a "
                         "crash. Noise-free configs only. events.json then covers only the windows of the last "
                         "process (result.json: resumed_windows)")
    ap.add_argument("--calib-tokens", type=int, default=2048,
                    help="tokens from the WikiText-2 *train* split used for ADC range calibration "
                         "(only if a layer has adc_range: calibrated)")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    model, tok, dtype = load_model(args.model, args.device, args.dtype)
    counter = EventCounter()
    cfg_dict, replaced = swap_from_config(model, args.config, counter)

    ids = load_wikitext2_ids(tok)
    ctx = context_len(model)
    seq_len = min(ctx, 2048) if args.seq_len is None else min(args.seq_len, ctx)
    max_windows = args.max_windows
    if args.max_tokens is not None:
        max_windows = max(1, args.max_tokens // seq_len)

    cal_cache = {}
    calibrated, calib_s = calibrate_if_needed(model, tok, seq_len, args.calib_tokens, args.device,
                                              **({"cache_dir": None} if args.no_calib_cache else {}),
                                              info=cal_cache)
    if args.free_sat_plans:
        free_saturation_plans(model)
    counter.reset_compute()              # events.json: evaluated tokens only (calibration pass excluded)
    t0 = time.time()
    ck = None
    if args.window_checkpoint:
        (ROOT / "results" / args.name).mkdir(parents=True, exist_ok=True)
        ck = str(ROOT / "results" / args.name / "windows.json")
    res = perplexity(model, ids, seq_len, max_windows, args.device, checkpoint=ck)
    res.update({
        "model": args.model, "config": args.config, "seq_len": seq_len, "dtype": str(dtype),
        "device": args.device, "threads": torch.get_num_threads(), "max_windows": max_windows, "max_tokens": args.max_tokens,
        "replaced_layers": len(replaced), "seconds": round(time.time() - t0, 1),
        "calibrated_layers": len(calibrated), "calib_tokens": args.calib_tokens if calibrated else 0,
        "calib_seconds": calib_s, "calib_cache": cal_cache.get("cache", "off"),
        "git": git_hash(),
    })

    # statistical mode is validated only where every layer's calibrated ADC step is 1/n MAC unit (F13).
    # Checked here, not in layers.py: an edit there would invalidate the calibration cache (HIST_SOURCES).
    stat = [m for m in model.modules() if getattr(getattr(m, "cfg", None), "mode", None) == "statistical"
            and getattr(m.cfg, "macro", "acim") == "acim" and getattr(m.cfg, "adc_range", None) == "calibrated"
            and getattr(m, "adc_fs", None) is not None]
    if stat:
        inexact = [m.name for m in stat if not rounding_exact(m.cfg, m.adc_fs)]
        res["stat_inexact_layers"] = len(inexact)
        res["stat_inexact_names"] = inexact[:20]
        if inexact:
            print(f"WARNING: {len(inexact)} statistical layers have an ADC step that is not 1/n MAC unit; "
                  f"statistical mode overestimates their rounding error (findings F13): {inexact[:5]}", flush=True)

    out = ROOT / "results" / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (out / "config.yaml").write_text(yaml.safe_dump(cfg_dict or {"fp": True}), encoding="utf-8")
    (out / "events.json").write_text(json.dumps(counter.to_dict()["total"], indent=2), encoding="utf-8")
    if calibrated:
        fs = {m.name: m.adc_fs.tolist() for m in model.modules()
              if getattr(m, "adc_fs", None) is not None}
        (out / "adc_full_scale.json").write_text(json.dumps(fs), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
