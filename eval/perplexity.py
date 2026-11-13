"""WikiText-2 perplexity with non-overlapping windows (stride = seq_len).

Requires: transformers, datasets.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import torch


def load_wikitext2_ids(tokenizer, split: str = "test") -> torch.Tensor:
    from datasets import load_dataset

    # Offline copy: CIMSIM_WIKITEXT_DIR holds {train,test}.parquet saved from Salesforce/wikitext
    local = os.environ.get("CIMSIM_WIKITEXT_DIR")
    if local:
        ds = load_dataset("parquet", data_files=str(Path(local) / f"{split}.parquet"), split="train")
        text = "\n\n".join(ds["text"])
        return tokenizer(text, return_tensors="pt").input_ids[0]
    # huggingface_hub >= 2 requires "owner/name"; older versions also accept the bare name
    try:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    except Exception:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(ds["text"])
    return tokenizer(text, return_tensors="pt").input_ids[0]


def _rss_gb() -> float | None:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2 ** 30
    except Exception:
        return None


@torch.no_grad()
def perplexity(model, ids: torch.Tensor, seq_len: int, max_windows: int | None = None,
               device: str = "cpu", progress: bool = True, checkpoint=None) -> dict:
    """Windows are taken from the start of the test set, so a run with
    max_windows=n always uses the same tokens (paired comparison across configs)."""
    model.eval()
    n_windows = ids.numel() // seq_len
    if max_windows is not None:
        n_windows = min(n_windows, max_windows)
    nll_sum, n_tok = 0.0, 0
    per_window = []
    # checkpoint (path): per-window NLLs are saved after every window and reused by a later call with the
    # same tokens, so a crashed run continues at the first missing window. Only for deterministic configs
    # (no analog noise): with noise the random stream of a resumed run differs from an uninterrupted one.
    ck_key = {"seq_len": seq_len, "n_windows": n_windows,
              "ids_sum": int(ids[:n_windows * seq_len].sum()), "nll_per_window": []}
    resumed = 0
    if checkpoint is not None and os.path.exists(checkpoint):
        with open(checkpoint, encoding="utf-8") as f:
            old = json.load(f)
        if all(old.get(k) == ck_key[k] for k in ("seq_len", "n_windows", "ids_sum")):
            per_window = list(old["nll_per_window"])[:n_windows]
            resumed = len(per_window)
            nll_sum = sum(per_window) * (seq_len - 1)
            n_tok = resumed * (seq_len - 1)
            if progress and resumed:
                print(f"  resumed {resumed} windows from {checkpoint}", flush=True)
    t_start = time.time()
    for w in range(resumed, n_windows):
        chunk = ids[w * seq_len:(w + 1) * seq_len].unsqueeze(0).to(device)
        out = model(chunk, labels=chunk)
        # HF loss is the mean over seq_len - 1 predicted tokens
        nll = out.loss.float().item()
        del out                              # logits of a 2,048-token window are > 1 GB on large vocabularies
        per_window.append(nll)
        if checkpoint is not None:
            with open(checkpoint, "w", encoding="utf-8") as f:
                json.dump({**ck_key, "nll_per_window": per_window}, f)
        nll_sum += nll * (seq_len - 1)
        n_tok += seq_len - 1
        if progress:
            el = time.time() - t_start
            eta = el / (w + 1 - resumed) * (n_windows - w - 1)
            rss = _rss_gb()
            print(f"  window {w + 1}/{n_windows}  running ppl {math.exp(nll_sum / n_tok):.3f}  "
                  f"elapsed {el / 60:.1f} min  eta {eta / 60:.1f} min"
                  + (f"  rss {rss:.1f} GB" if rss is not None else ""), flush=True)
    return {"ppl": math.exp(nll_sum / n_tok), "nll": nll_sum / n_tok, "tokens": n_tok,
            "windows": n_windows, "nll_per_window": per_window, "resumed_windows": resumed}
