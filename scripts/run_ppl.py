"""Run WikiText-2 perplexity for one model under one CIM config.

Example:
    python scripts/run_ppl.py --model facebook/opt-125m \
        --config config/experiments/e0_w8a8.yaml --name e0_opt125m_w8a8

Writes results/<name>/{result.json, config.yaml, events.json}.
The unmodified model (no CIM swap) is selected with --config none.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cimsim.calibrate import calibrate_adc_range  # noqa: E402
from cimsim.config import LayerConfigMap  # noqa: E402
from cimsim.events import EventCounter  # noqa: E402
from cimsim.models.swap import swap_linears  # noqa: E402
from eval.perplexity import load_wikitext2_ids, perplexity  # noqa: E402


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def main() -> None:
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
    ap.add_argument("--calib-tokens", type=int, default=2048,
                    help="tokens from the WikiText-2 *train* split used for ADC range calibration "
                         "(only if a layer has adc_range: calibrated)")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, args.dtype)
    if args.device == "cpu" and dtype == torch.float16:
        dtype = torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)
    try:  # transformers >= 4.56 uses `dtype`; older versions use `torch_dtype`
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model = model.to(args.device)

    counter = EventCounter()
    cfg_dict = None
    replaced: list[str] = []
    if args.config.lower() != "none":
        with open(args.config, "r", encoding="utf-8") as f:
            cfg_dict = yaml.safe_load(f)
        replaced = swap_linears(model, LayerConfigMap.from_dict(cfg_dict), counter)

    ids = load_wikitext2_ids(tok)
    ctx = getattr(model.config, "max_position_embeddings", None) or getattr(model.config, "n_positions", 2048)
    seq_len = min(ctx, 2048) if args.seq_len is None else min(args.seq_len, ctx)
    max_windows = args.max_windows
    if args.max_tokens is not None:
        max_windows = max(1, args.max_tokens // seq_len)

    calibrated, calib_s = [], 0.0
    if any(getattr(m, "cfg", None) is not None and m.cfg.mode == "bit_accurate"
           and m.cfg.adc_range == "calibrated" for m in model.modules()):
        tc = time.time()
        train_ids = load_wikitext2_ids(tok, split="train")[: args.calib_tokens]
        cal_len = min(seq_len, args.calib_tokens)
        def run_calib():
            for i in range(0, train_ids.numel() - cal_len + 1, cal_len):
                model(train_ids[i:i + cal_len].unsqueeze(0).to(args.device))
        print(f"calibrating ADC range on {args.calib_tokens} train tokens ...", flush=True)
        calibrated = calibrate_adc_range(model, run_calib)
        calib_s = round(time.time() - tc, 1)
        print(f"calibrated {len(calibrated)} layers in {calib_s} s", flush=True)
    t0 = time.time()
    res = perplexity(model, ids, seq_len, max_windows, args.device)
    res.update({
        "model": args.model, "config": args.config, "seq_len": seq_len, "dtype": str(dtype),
        "device": args.device, "threads": torch.get_num_threads(), "max_windows": max_windows, "max_tokens": args.max_tokens,
        "replaced_layers": len(replaced), "seconds": round(time.time() - t0, 1),
        "calibrated_layers": len(calibrated), "calib_tokens": args.calib_tokens if calibrated else 0,
        "calib_seconds": calib_s,
        "git": git_hash(),
    })

    out = ROOT / "results" / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(res, indent=2))
    (out / "config.yaml").write_text(yaml.safe_dump(cfg_dict or {"fp": True}))
    (out / "events.json").write_text(json.dumps(counter.to_dict()["total"], indent=2))
    if calibrated:
        fs = {m.name: m.adc_fs.tolist() for m in model.modules()
              if getattr(m, "adc_fs", None) is not None}
        (out / "adc_full_scale.json").write_text(json.dumps(fs))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
