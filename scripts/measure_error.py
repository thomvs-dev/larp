"""A3 step 1: measure per-layer output error of a bit-accurate CIM config on real LLM activations.

Example (OPT-125M, ~3 min per 2,048 tokens on the i7-12700H):
    python scripts/measure_error.py --model facebook/opt-125m \
        --config config/experiments/f5_adc6_calibrated.yaml --name a3m_opt125m_f5

Tokens come from the WikiText-2 *train* split, after the tokens used for ADC range calibration,
so they are disjoint from both calibration and the test-set evaluation.

Writes results/<name>/:
    error_stats.json   per-layer statistics (cimsim/stat/measure.py)
    by_type.md         table aggregated by layer type (layer index removed from the name)
    samples.pt         per-layer samples for fitting the statistical model (not tracked in git)
    config.yaml, meta.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cimsim.events import EventCounter  # noqa: E402
from cimsim.stat.measure import ErrorProbe  # noqa: E402
from eval.perplexity import load_wikitext2_ids  # noqa: E402
from eval.runner import calibrate_if_needed, context_len, git_hash, load_model, swap_from_config  # noqa: E402


def layer_type(name: str) -> str:
    return re.sub(r"\.\d+\.", ".*.", name)


def by_type_table(stats: dict) -> str:
    groups = defaultdict(list)
    for name, s in stats.items():
        groups[layer_type(name)].append(s)
    lines = ["| Layer type | n | ADC b | rel. error (median / max) | exact (%) | bias/std | excess kurtosis | "
             "token RMS q90/median | corr(log token err, log token signal) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for t, ss in groups.items():
        med = lambda k: sorted(x[k] for x in ss)[len(ss) // 2]  # noqa: E731
        lines.append(
            f"| {t} | {len(ss)} | {ss[0]['adc_bits']} | {med('rel_err'):.2e} / {max(x['rel_err'] for x in ss):.2e} | "
            f"{100 * med('frac_exact'):.1f} | {med('bias_over_std'):+.3f} | {med('excess_kurtosis'):.2f} | "
            f"{sorted(x['token_rms_q10_q50_q90_max'][2] / max(x['token_rms_q10_q50_q90_max'][1], 1e-12) for x in ss)[len(ss) // 2]:.2f} | "
            f"{med('corr_log_token_ms__log_ideal_ms'):+.2f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--tokens", type=int, default=2048, help="train tokens to measure on")
    ap.add_argument("--calib-tokens", type=int, default=2048,
                    help="train tokens for ADC range calibration (measurement starts after them)")
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--samples", type=int, default=20000, help="sampled error elements kept per layer")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    model, tok, dtype = load_model(args.model, args.device)
    model.eval()
    cfg_dict, replaced = swap_from_config(model, args.config, EventCounter())
    seq_len = min(context_len(model), 2048) if args.seq_len is None else args.seq_len
    calibrated, calib_s = calibrate_if_needed(model, tok, seq_len, args.calib_tokens, args.device)

    probe = ErrorProbe(args.samples)
    names = probe.attach(model)
    if not names:
        raise SystemExit("no bit_accurate CIM layers in this config")
    ids = load_wikitext2_ids(tok, split="train")[args.calib_tokens: args.calib_tokens + args.tokens]
    win = min(seq_len, ids.numel())
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, ids.numel() - win + 1, win):
            model(ids[i:i + win].unsqueeze(0).to(args.device))
            print(f"  {i + win}/{ids.numel()} tokens, {time.time() - t0:.0f} s", flush=True)
    secs = round(time.time() - t0, 1)
    probe.detach(model)

    stats = probe.summary()
    out = ROOT / "results" / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "error_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    (out / "by_type.md").write_text(by_type_table(stats), encoding="utf-8")
    torch.save(probe.samples(), out / "samples.pt")
    (out / "config.yaml").write_text(yaml.safe_dump(cfg_dict), encoding="utf-8")
    meta = {"model": args.model, "config": args.config, "tokens": int(ids.numel()), "seq_len": win,
            "train_token_offset": args.calib_tokens, "calibrated_layers": len(calibrated),
            "calib_seconds": calib_s, "seconds": secs, "threads": torch.get_num_threads(),
            "layers": len(names), "git": git_hash()}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print((out / "by_type.md").read_text(encoding="utf-8"))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
