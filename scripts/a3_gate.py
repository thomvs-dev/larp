"""GATE A3: statistical vs bit-accurate perplexity for one config (docs/part_a_plan.md §3.5).

    python scripts/a3_gate.py --model facebook/opt-125m --config config/experiments/f5_adc6_calibrated.yaml \
        --name a3g_opt125m_f5 --tokens 8192 --ba-backends fast,reference

One calibration pass (ideal outputs, WikiText-2 train tokens) records the MAC histograms. They set
the ADC full scale (adc_range: calibrated) and the statistical error model, so both modes use the
same ADC ranges. Then, on the first --tokens WikiText-2 test tokens:
  - bit-accurate PPL for every backend in --ba-backends (seed --ba-seeds[0]) and every extra seed in
    --ba-seeds (backend fast): the run-to-run spread of the bit-accurate simulation (findings F6),
  - statistical PPL for every seed in --stat-seeds,
  - fake-quant PPL on the same tokens, for reference.
Pass: |median stat PPL - bit-accurate PPL (fast, first seed)| / bit-accurate PPL <= max(--tol, spread),
spread = (max - min) / min over the bit-accurate runs.

Writes results/<name>/{gate.json, config.yaml, stat_params.json}. --table prints a summary of all
results/a3g_*/gate.json files and writes results/a3_gate_summary.md.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cimsim.core.layers import _CIMBase  # noqa: E402
from cimsim.events import EventCounter  # noqa: E402
from eval.perplexity import load_wikitext2_ids, perplexity  # noqa: E402
from eval.runner import calibrate_if_needed, context_len, git_hash, load_model, swap_from_config  # noqa: E402


def set_mode(model, **changes) -> None:
    """Change config fields on every CIM layer that is not fp / fake_quant in the YAML."""
    for m in model.modules():
        if isinstance(m, _CIMBase) and m._yaml_mode in ("bit_accurate", "statistical"):
            m.cfg = dataclasses.replace(m.cfg, **changes)
            m._gen = None                                   # re-seed the layer's noise stream


def run(model, ids, seq_len, n_windows, device) -> dict:
    t0 = time.time()
    r = perplexity(model, ids, seq_len, n_windows, device, progress=False)
    dt = time.time() - t0
    return {"ppl": r["ppl"], "seconds": round(dt, 1), "ms_per_token": round(1000 * dt / (n_windows * seq_len), 2)}


def table() -> None:
    rows = ["| Config | tokens | fake-quant | bit-accurate (runs) | spread | statistical (seeds) | diff | tolerance | pass | speed-up |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for f in sorted((ROOT / "results").glob("a3g_*/gate.json")):
        g = json.loads(f.read_text(encoding="utf-8"))
        ba = ", ".join(f"{k} {v['ppl']:.3f}" for k, v in g["bit_accurate"].items())
        st = ", ".join(f"{v['ppl']:.3f}" for v in g["statistical"].values())
        rows.append(f"| {f.parent.name} | {g['tokens']} | {g['fake_quant']['ppl']:.3f} | {ba} | {100 * g['spread']:.2f}% | "
                    f"{st} | {100 * g['rel_diff']:+.2f}% | {100 * g['tolerance']:.2f}% | {'yes' if g['pass'] else 'NO'} | "
                    f"{g['speedup']:.0f}x |")
    text = "\n".join(rows) + "\n"
    (ROOT / "results" / "a3_gate_summary.md").write_text(text, encoding="utf-8")
    print(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/opt-125m")
    ap.add_argument("--config")
    ap.add_argument("--name")
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--ba-backends", default="fast", help="comma list: fast,reference")
    ap.add_argument("--ba-seeds", default="0", help="comma list; seeds after the first run with backend fast")
    ap.add_argument("--stat-seeds", default="0,1,2")
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--calib-tokens", type=int, default=2048)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--table", action="store_true", help="summarise results/a3g_*/gate.json and exit")
    args = ap.parse_args()
    if args.table:
        table()
        return
    if not (args.config and args.name):
        ap.error("--config and --name are required")
    if args.threads:
        torch.set_num_threads(args.threads)

    model, tok, _ = load_model(args.model, args.device)
    model.eval()
    cfg_dict, _ = swap_from_config(model, args.config, EventCounter())
    cim = [m for m in model.modules() if isinstance(m, _CIMBase)]
    for m in cim:
        m._yaml_mode = m.cfg.mode
    if not any(m._yaml_mode == "bit_accurate" for m in cim):
        raise SystemExit("config has no bit_accurate layers")
    seq_len = min(context_len(model), 2048)
    n_windows = max(1, args.tokens // seq_len)
    ids = load_wikitext2_ids(tok)

    set_mode(model, mode="statistical")                     # histogram pass prepares both modes
    _, calib_s = calibrate_if_needed(model, tok, seq_len, args.calib_tokens, args.device)
    out = ROOT / "results" / args.name
    out.mkdir(parents=True, exist_ok=True)
    stat_params = {m.name: m.stat_info for m in cim if m.stat_info is not None}
    (out / "stat_params.json").write_text(json.dumps(stat_params, indent=1), encoding="utf-8")
    res = {"model": args.model, "config": args.config, "tokens": n_windows * seq_len, "seq_len": seq_len,
           "calib_tokens": args.calib_tokens, "calib_seconds": calib_s, "threads": torch.get_num_threads(),
           "git": git_hash(), "bit_accurate": {}, "statistical": {}}

    def save():
        (out / "gate.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    set_mode(model, mode="fake_quant")
    res["fake_quant"] = run(model, ids, seq_len, n_windows, args.device)
    print(f"fake_quant            PPL {res['fake_quant']['ppl']:.3f}", flush=True)
    for s in [int(v) for v in args.stat_seeds.split(",")]:
        set_mode(model, mode="statistical", seed=s)
        res["statistical"][f"seed{s}"] = r = run(model, ids, seq_len, n_windows, args.device)
        print(f"statistical seed {s}    PPL {r['ppl']:.3f}  ({r['ms_per_token']} ms/token)", flush=True)
        save()
    seeds = [int(v) for v in args.ba_seeds.split(",")]
    plan = [(b, seeds[0]) for b in args.ba_backends.split(",")] + [("fast", s) for s in seeds[1:]]
    for backend, s in plan:
        set_mode(model, mode="bit_accurate", backend=backend, seed=s)
        key = f"{backend}_seed{s}"
        print(f"bit_accurate {key} ...", flush=True)
        res["bit_accurate"][key] = r = run(model, ids, seq_len, n_windows, args.device)
        print(f"bit_accurate {key:12s} PPL {r['ppl']:.3f}  ({r['ms_per_token']} ms/token)", flush=True)
        save()

    ba = [v["ppl"] for v in res["bit_accurate"].values()]
    ref = res["bit_accurate"][f"{plan[0][0]}_seed{plan[0][1]}"]
    st_med = statistics.median(v["ppl"] for v in res["statistical"].values())
    res["spread"] = (max(ba) - min(ba)) / min(ba)
    res["rel_diff"] = (st_med - ref["ppl"]) / ref["ppl"]
    res["tolerance"] = max(args.tol, res["spread"])
    res["pass"] = abs(res["rel_diff"]) <= res["tolerance"]
    st_ms = statistics.median(v["ms_per_token"] for v in res["statistical"].values())
    res["speedup"] = ref["ms_per_token"] / st_ms
    save()
    (out / "config.yaml").write_text(yaml.safe_dump(cfg_dict), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("spread", "rel_diff", "tolerance", "pass", "speedup")}, indent=2))


if __name__ == "__main__":
    main()
