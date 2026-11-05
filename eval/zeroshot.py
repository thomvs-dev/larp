"""Zero-shot multiple-choice evaluation (ARC-e, PIQA, HellaSwag, ...) through lm-eval-harness (A4).

The model is built, swapped and calibrated by the caller (eval/runner.py), then wrapped in lm-eval's HFLM
as a pre-initialised model, so every cimsim mode (fake_quant, bit_accurate, statistical) is evaluated
the same way as in the perplexity runs.

Settings:
- Batch size: 1 whenever attention runs on CIM (CIMMatMul present): lm-eval right-pads the shorter
  requests of a batch, and per-channel key scales (F10) would then include padding positions. With
  linear layers only, padding cannot change the real tokens (causal mask, right padding, per-token
  activation scales, per-row error terms), so larger batches are allowed; they cut the per-call cost of
  statistical mode on short requests. Noise-free configs give the same scores at any batch size (up to
  float summation order); with analog noise the random draws differ between batch sizes.
- `limit` takes the first N documents of the task split (lm-eval does not shuffle), so runs with the
  same limit are paired: every config sees the same questions. paired_compare() uses that.
- One task per lm-eval call, so events and runtime are recorded per task.

Per task the caller gets: lm-eval metrics (acc, acc_norm, stderr), per-document correctness
(for paired tests), events, seconds, and the number of tokens run through the model.

Requires: lm-eval >= 0.4 (pip install -e ".[eval]").
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch

DEFAULT_TASKS = ("arc_easy", "piqa", "hellaswag")
METRICS = ("acc", "acc_norm")


class TokenCounter:
    """Counts input tokens of every model forward call (for ms / token)."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.tokens = 0
        self.calls = 0
        self._h = model.register_forward_pre_hook(self._hook, with_kwargs=True)

    def _hook(self, module, args, kwargs):
        ids = args[0] if args else kwargs.get("input_ids")
        if ids is not None:
            self.tokens += int(ids.numel())
            self.calls += 1

    def remove(self) -> None:
        self._h.remove()


def has_cim_attention(model: torch.nn.Module) -> bool:
    from cimsim.core.attention import CIMMatMul
    return any(isinstance(m, CIMMatMul) for m in model.modules())


def _task_file(out_dir: Path, task: str) -> Path:
    return out_dir / f"zs_{task}.json"


def evaluate_tasks(model, tok, tasks, out_dir: Path, counter=None, limit: int | None = None,
                   num_fewshot: int = 0, include_path: str | None = None, run_info: dict | None = None,
                   resume: bool = True, batch_size: int = 1) -> dict:
    """Runs each task and writes out_dir/zs_<task>.json (summary + per-document scores) and
    out_dir/events_<task>.json. A task whose file exists with the same limit / num_fewshot is skipped
    (resume after an interrupted run). Returns {task: summary}."""
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager

    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    if batch_size > 1 and has_cim_attention(model):
        raise ValueError("batch_size > 1 is not allowed with attention on CIM (padding would enter the "
                         "per-channel key scales); use batch_size 1")
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=batch_size)
    tm = TaskManager(include_path=include_path) if include_path else TaskManager()
    summaries = {}
    for task in tasks:
        f = _task_file(out_dir, task)
        if resume and f.exists():
            old = json.loads(f.read_text(encoding="utf-8"))
            if old.get("limit") == limit and old.get("num_fewshot") == num_fewshot:
                print(f"[{task}] already done ({f.name}), skipped", flush=True)
                summaries[task] = old["summary"]
                continue
        if counter is not None:
            counter.reset_compute()
        tc = TokenCounter(model)
        t0 = time.time()
        print(f"[{task}] running (limit {limit}, {num_fewshot}-shot) ...", flush=True)
        try:
            res = lm_eval.simple_evaluate(model=lm, tasks=[task], num_fewshot=num_fewshot, limit=limit,
                                          batch_size=batch_size, log_samples=True, task_manager=tm,
                                          bootstrap_iters=0)
        finally:
            tc.remove()
        secs = time.time() - t0
        summary, docs = _extract(res, task)
        summary.update({"seconds": round(secs, 1), "model_tokens": tc.tokens, "forward_calls": tc.calls,
                        "ms_per_token": round(1000 * secs / max(tc.tokens, 1), 2)})
        rec = {"task": task, "limit": limit, "num_fewshot": num_fewshot, "batch_size": batch_size, "summary": summary,
               "lm_eval_version": getattr(lm_eval, "__version__", "unknown"),
               "task_version": res.get("versions", {}).get(task), "docs": docs, **(run_info or {})}
        f.write_text(json.dumps(rec, indent=1), encoding="utf-8")
        if counter is not None:
            (out_dir / f"events_{task}.json").write_text(json.dumps(counter.to_dict()["total"], indent=2),
                                                         encoding="utf-8")
        print(f"[{task}] " + "  ".join(f"{m} {summary[m]:.4f}" for m in METRICS if m in summary)
              + f"  n {summary['n']}  {secs / 60:.1f} min  {summary['ms_per_token']} ms/token", flush=True)
        summaries[task] = summary
    return summaries


def _extract(res: dict, task: str) -> tuple[dict, list[dict]]:
    r = res["results"][task]
    summary = {}
    for m in METRICS:
        if f"{m},none" in r:
            summary[m] = float(r[f"{m},none"])
    docs = []
    for s in res["samples"][task]:
        d = {"doc_id": int(s["doc_id"])}
        for m in METRICS:
            if m in s:
                d[m] = int(round(float(s[m])))
        docs.append(d)
    docs.sort(key=lambda d: d["doc_id"])
    summary["n"] = len(docs)
    for m in METRICS:                    # binomial standard error of the accuracy
        if m in summary and docs:
            p = summary[m]
            summary[f"{m}_stderr"] = math.sqrt(p * (1 - p) / max(len(docs) - 1, 1))
    return summary, docs


def load_task_result(result_dir: Path, task: str) -> dict:
    return json.loads(_task_file(Path(result_dir), task).read_text(encoding="utf-8"))


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar test (binomial on the discordant pairs b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def paired_compare(docs_a: list[dict], docs_b: list[dict], metric: str = "acc") -> dict:
    """b − a on the documents both runs scored. Returns accuracies, difference, its standard error
    (paired), 95 % interval (normal approximation), discordant counts and exact McNemar p."""
    a = {d["doc_id"]: d[metric] for d in docs_a if metric in d}
    b = {d["doc_id"]: d[metric] for d in docs_b if metric in d}
    ids = sorted(set(a) & set(b))
    n = len(ids)
    if n == 0:
        raise ValueError("no common documents")
    only_b = sum(1 for i in ids if b[i] and not a[i])        # b right, a wrong
    only_a = sum(1 for i in ids if a[i] and not b[i])
    acc_a = sum(a[i] for i in ids) / n
    acc_b = sum(b[i] for i in ids) / n
    diff = acc_b - acc_a
    var = ((only_b + only_a) / n - diff ** 2) / max(n - 1, 1)
    se = math.sqrt(max(var, 0.0))
    return {"n": n, "acc_a": acc_a, "acc_b": acc_b, "diff": diff, "se": se,
            "ci95": (diff - 1.96 * se, diff + 1.96 * se), "b_only": only_b, "a_only": only_a,
            "mcnemar_p": mcnemar_p(only_a, only_b)}
