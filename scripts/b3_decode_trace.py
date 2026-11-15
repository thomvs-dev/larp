"""B3: check the per-token event counts the system model derives from model shapes against a real
prefill + cached decode trace.

Runs one prompt of --prompt tokens through the swapped model (prefill, use_cache), then feeds --steps
tokens one at a time with the KV cache (decode; the fed tokens are the next WikiText-2 tokens, so no
sampling is involved). The event counter is read after the prefill and after every decode step and
compared with cimsim.system.counts. Writes results/<name>/{trace.json, summary.md}.

    python scripts/b3_decode_trace.py --model Qwen/Qwen2.5-1.5B --shape config/models/qwen2.5-1.5b.yaml `
        --config config/experiments/b3_trace_fq_n128.yaml --name b3_trace_qwen15b --threads 14
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

from cimsim.events import EventCounter  # noqa: E402
from cimsim.system import ModelShape, attention_events, linear_events  # noqa: E402
from eval.runner import git_hash, swap_from_config  # noqa: E402

KEYS = ("macs", "adc_conversions", "array_cycles", "kv_bits_written")


def expected(shape: ModelShape, nrow: int, attention: bool, positions: list[int]) -> dict:
    """Events of the tokens at the given 1-based positions (keys attended = position)."""
    out = dict.fromkeys(KEYS, 0)
    for layer in shape.linear_layers():
        ev = linear_events(layer, nrow)
        for k in ("macs", "adc_conversions", "array_cycles"):
            out[k] += ev[k] * len(positions)
    if attention:
        for p in positions:
            ev = attention_events(shape, p, nrow)
            for k in KEYS:
                out[k] += ev[k] * shape.n_layers
    return out


@torch.no_grad()
def trace(model, ids: torch.Tensor, prompt: int, steps: int, counter: EventCounter) -> tuple[dict, list[dict]]:
    counter.reset_compute()
    out = model(ids[:, :prompt], use_cache=True)
    pre = {k: counter.total().get(k, 0) for k in KEYS}
    past, rows, last = out.past_key_values, [], dict(pre)
    for s in range(steps):
        out = model(ids[:, prompt + s: prompt + s + 1], past_key_values=past, use_cache=True)
        past = out.past_key_values
        now = {k: counter.total().get(k, 0) for k in KEYS}
        rows.append({k: now[k] - last[k] for k in KEYS})
        last = now
    return pre, rows


def compare(shape, nrow, attention, prompt, steps, pre, rows) -> dict:
    exp_pre = expected(shape, nrow, attention, list(range(1, prompt + 1)))
    checks = [{"what": f"prefill, {prompt} tokens", "measured": pre, "expected": exp_pre,
               "equal": all(pre[k] == exp_pre[k] for k in KEYS)}]
    for s, r in enumerate(rows):
        e = expected(shape, nrow, attention, [prompt + s + 1])
        checks.append({"what": f"decode step {s + 1} (position {prompt + s + 1})", "measured": r, "expected": e,
                       "equal": all(r[k] == e[k] for k in KEYS)})
    return {"all_equal": all(c["equal"] for c in checks), "checks": checks}


def main() -> None:
    import yaml
    from eval.perplexity import load_wikitext2_ids
    from eval.runner import load_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--shape", required=True, help="config/models/*.yaml")
    ap.add_argument("--config", default=str(ROOT / "config/experiments/b3_trace_fq_n128.yaml"))
    ap.add_argument("--name", required=True)
    ap.add_argument("--prompt", type=int, default=64)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--threads", type=int, default=None)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    print(f"git {git_hash()}", flush=True)
    shape = ModelShape.from_yaml(a.shape)
    model, tok, _ = load_model(a.model, "cpu", "float32")
    counter = EventCounter()
    cfg, replaced = swap_from_config(model, a.config, counter)
    nrow, attention = cfg["default"]["nrow"], bool(cfg.get("attention", False))
    ids = load_wikitext2_ids(tok)[: a.prompt + a.steps].reshape(1, -1)
    t0 = time.time()
    pre, rows = trace(model, ids, a.prompt, a.steps, counter)
    res = compare(shape, nrow, attention, a.prompt, a.steps, pre, rows)
    res.update({"git": git_hash(), "model": a.model, "shape": a.shape, "config": cfg, "prompt": a.prompt,
                "steps": a.steps, "replaced_layers": len(replaced), "seconds": round(time.time() - t0, 1)})
    out = ROOT / "results" / a.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "trace.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    bad = [c for c in res["checks"] if not c["equal"]]
    lines = [f"# B3 decode trace, {a.model} (git {res['git']})", "",
             f"Prompt {a.prompt} tokens, {a.steps} cached decode steps, config `{Path(a.config).name}`, "
             f"attention on CIM: {attention}. {len(res['checks']) - len(bad)} of {len(res['checks'])} checks equal "
             "(measured event counts = counts derived from the model shape).", "",
             "| Check | MACs measured | MACs derived | ADC conversions measured | derived | Equal |", "|---|---|---|---|---|---|"]
    show = [res["checks"][0], res["checks"][1], res["checks"][-1]] + bad[:5]
    for c in show:
        lines.append(f"| {c['what']} | {c['measured']['macs']} | {c['expected']['macs']} | "
                     f"{c['measured']['adc_conversions']} | {c['expected']['adc_conversions']} | {c['equal']} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("RESULT:", "ALL EQUAL" if res["all_equal"] else f"{len(bad)} MISMATCHES", flush=True)


if __name__ == "__main__":
    main()
