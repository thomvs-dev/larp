"""Diagnostic for the TinyLlama FP32 zero-shot gap (A4): tokenization and BOS handling under lm-eval.

Part 1 (--tokens-only, seconds): tokenizer settings, and the exact token ids lm-eval scores for the first
ARC-e requests (context / continuation split from HFLM._encode_pair), decoded token by token.
Part 2: accuracy (default task ARC-e) on the first N documents for each (config, add_bos_token) pair, with per-choice
log-likelihoods saved so FP32 and W8A8 can be compared per request.

Example (laptop):
    python scripts/diag_zs_tokens.py --model TinyLlama/TinyLlama_v1.1 --name diag_zs_tokens_tinyllama --tokens-only
    python scripts/diag_zs_tokens.py --model TinyLlama/TinyLlama_v1.1 --name diag_zs_tokens_tinyllama --limit 200 --threads 6

Writes results/<name>/{tokens.json, grid.json, ll_<config>_bos<setting>.json}.
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
from eval.runner import git_hash, load_model, swap_from_config  # noqa: E402

BOS_SETTINGS = {"default": None, "on": True, "off": False}


def tokenizer_info(tok) -> dict:
    keys = ("add_bos_token", "add_eos_token", "legacy", "add_prefix_space", "padding_side")
    info = {"class": type(tok).__name__, "bos_token_id": tok.bos_token_id, "eos_token_id": tok.eos_token_id,
            "pad_token_id": tok.pad_token_id}
    for k in keys:
        info[k] = getattr(tok, k, "n/a")
    for s in ("Hello world", " Paris", "Answer: Paris"):
        ids = tok.encode(s)
        info[f"encode({s!r})"] = [ids, tok.convert_ids_to_tokens(ids)]
    return info


def task_manager(include_path):
    from lm_eval.tasks import TaskManager
    return TaskManager(include_path=include_path) if include_path else TaskManager()


def task_requests(task_name: str, limit: int, include_path=None):
    from lm_eval.tasks import get_task_dict
    task = get_task_dict([task_name], task_manager(include_path))[task_name]
    task.build_all_requests(limit=limit, rank=0, world_size=1)
    return [inst.args for inst in task.instances]


def dump_tokens(model, tok, n_req: int, task_name: str, include_path=None) -> list[dict]:
    from lm_eval.models.huggingface import HFLM
    out = []
    reqs = task_requests(task_name, 2, include_path)[:n_req]
    for name, setting in BOS_SETTINGS.items():
        lm = HFLM(pretrained=model, tokenizer=tok, batch_size=1, add_bos_token=setting)
        for ctx, cont in reqs:
            c_enc, k_enc = lm._encode_pair(ctx, cont)
            out.append({"bos": name, "context": ctx, "continuation": cont,
                        "context_ids_head": c_enc[:6], "context_tokens_head": tok.convert_ids_to_tokens(c_enc[:6]),
                        "context_tokens_tail": tok.convert_ids_to_tokens(c_enc[-4:]),
                        "continuation_ids": k_enc, "continuation_tokens": tok.convert_ids_to_tokens(k_enc),
                        "n_context": len(c_enc)})
    return out


def run_grid(args, out: Path) -> dict:
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    grid = {}
    for cfg_name, cfg_path in (("fp32", "none"), ("w8a8", args.w8a8_config)):
        model, tok, _ = load_model(args.model, "cpu", "float32")
        swap_from_config(model, cfg_path, EventCounter())
        model.eval()
        for bos_name, setting in BOS_SETTINGS.items():
            key = f"{cfg_name}_bos{bos_name}"
            t0 = time.time()
            lm = HFLM(pretrained=model, tokenizer=tok, batch_size=args.batch_size, add_bos_token=setting)
            res = lm_eval.simple_evaluate(model=lm, tasks=[args.task], limit=args.limit, batch_size=args.batch_size,
                                          log_samples=True, bootstrap_iters=0,
                                          task_manager=task_manager(args.include_path))
            r = res["results"][args.task]
            docs = []
            for s in res["samples"][args.task]:
                docs.append({"doc_id": int(s["doc_id"]), "acc": int(round(float(s["acc"]))),
                             "acc_norm": int(round(float(s["acc_norm"]))),
                             "ll": [float(x[0]) for x in s["filtered_resps"]],
                             "target": s.get("target")})
            docs.sort(key=lambda d: d["doc_id"])
            (out / f"ll_{key}.json").write_text(json.dumps(docs), encoding="utf-8")
            grid[key] = {"acc": float(r["acc,none"]), "acc_norm": float(r["acc_norm,none"]),
                         "n": len(docs), "seconds": round(time.time() - t0, 1)}
            print(f"{key}: acc {grid[key]['acc']:.4f}  acc_norm {grid[key]['acc_norm']:.4f}  "
                  f"({grid[key]['seconds'] / 60:.1f} min)", flush=True)
        del model
    return grid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--w8a8-config", default=str(ROOT / "config/experiments/e0_w8a8.yaml"))
    ap.add_argument("--tokens-only", action="store_true")
    ap.add_argument("--task", default="arc_easy")
    ap.add_argument("--include-path", default=None, help="extra lm-eval task YAML folder (tests)")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    out = ROOT / "results" / args.name
    out.mkdir(parents=True, exist_ok=True)

    import lm_eval
    import transformers
    model, tok, _ = load_model(args.model, "cpu", "float32")
    meta = {"model": args.model, "git": git_hash(), "lm_eval": lm_eval.__version__,
            "transformers": transformers.__version__, "torch": torch.__version__,
            "tokenizer": tokenizer_info(tok), "requests": dump_tokens(model, tok, 4, args.task, args.include_path)}
    (out / "tokens.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta["tokenizer"], indent=1, ensure_ascii=False))
    for r in meta["requests"]:
        print(f"[bos {r['bos']}] n_ctx {r['n_context']} head {r['context_tokens_head']} "
              f"tail {r['context_tokens_tail']} | cont {r['continuation_tokens']}")
    if args.tokens_only:
        return
    del model
    grid = run_grid(args, out)
    (out / "grid.json").write_text(json.dumps({"limit": args.limit, "batch_size": args.batch_size,
                                               "git": meta["git"], "grid": grid}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
