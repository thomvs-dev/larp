"""Statistical-mode PPL only (no bit-accurate runs), for comparison with bit-accurate PPLs from a
GATE A3 run on another machine. Same calibration as scripts/a3_gate.py (2,048 train tokens,
statistical mode, one pass for ADC range and error model).

    python scripts/a3_stat_only.py --model <name or dir> --config <yaml> --tokens 4096 --out results/<name>
Writes <out>/stat_only.json (PPL per seed, fake-quant, ms/token, sat_work_fraction, ADC full-scale hash).
"""
from __future__ import annotations

import argparse, dataclasses, hashlib, json, sys, time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cimsim.core.layers import _CIMBase  # noqa: E402
from cimsim.events import EventCounter  # noqa: E402
from eval.perplexity import load_wikitext2_ids, perplexity  # noqa: E402
from eval.runner import calibrate_if_needed, git_hash, load_model, swap_from_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/opt-125m")
    ap.add_argument("--config", required=True)
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--calib-tokens", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model, tok, _ = load_model(a.model, "cpu"); model.eval()
    swap_from_config(model, a.config, EventCounter())
    cim = [m for m in model.modules() if isinstance(m, _CIMBase) and m.cfg.mode in ("bit_accurate", "statistical")]
    for m in cim:
        m.cfg = dataclasses.replace(m.cfg, mode="statistical"); m._gen = None
    seq = 2048
    _, cs = calibrate_if_needed(model, tok, seq, a.calib_tokens, "cpu")
    h = hashlib.md5()
    for m in cim:
        if m.adc_fs is not None:
            h.update(m.name.encode()); h.update(m.adc_fs.double().cpu().numpy().tobytes())
    ids = load_wikitext2_ids(tok)
    nw = max(1, a.tokens // seq)
    res = {"model": a.model, "config": a.config, "tokens": nw * seq, "calib_tokens": a.calib_tokens,
           "calib_seconds": cs, "threads": torch.get_num_threads(), "git": git_hash(), "adc_fs_hash": h.hexdigest(),
           "statistical": {}, "stat_params": {m.name: m.stat_info for m in cim}}
    for s in [int(v) for v in a.seeds.split(",")]:
        for m in cim:
            m.cfg = dataclasses.replace(m.cfg, seed=s); m._gen = None; m.sat_stats = {}
        t = time.time()
        r = perplexity(model, ids, seq, nw, "cpu", progress=False)
        dt = time.time() - t
        ex = sum(m.sat_stats.get("macs_exact", 0) for m in cim); tot = sum(m.sat_stats.get("macs_total", 0) for m in cim)
        res["statistical"][f"seed{s}"] = {"ppl": r["ppl"], "seconds": round(dt, 1),
                                          "ms_per_token": round(1000 * dt / (nw * seq), 2),
                                          "sat_work_fraction": ex / tot if tot else 0.0}
        print(f"statistical seed {s}: {res['statistical'][f'seed{s}']}", flush=True)
        (out / "stat_only.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for m in cim:
        m.cfg = dataclasses.replace(m.cfg, mode="fake_quant")
    t = time.time()
    r = perplexity(model, ids, seq, nw, "cpu", progress=False)
    res["fake_quant"] = {"ppl": r["ppl"], "seconds": round(time.time() - t, 1)}
    (out / "stat_only.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("fake_quant", res["fake_quant"], flush=True)


if __name__ == "__main__":
    main()
