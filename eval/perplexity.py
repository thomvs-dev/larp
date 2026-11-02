"""WikiText-2 perplexity with non-overlapping windows (stride = seq_len).

Requires: transformers, datasets.
"""

from __future__ import annotations

import math
import time

import torch


def load_wikitext2_ids(tokenizer, split: str = "test") -> torch.Tensor:
    from datasets import load_dataset

    # huggingface_hub >= 2 requires "owner/name"; older versions also accept the bare name
    try:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    except Exception:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(ds["text"])
    return tokenizer(text, return_tensors="pt").input_ids[0]


@torch.no_grad()
def perplexity(model, ids: torch.Tensor, seq_len: int, max_windows: int | None = None,
               device: str = "cpu", progress: bool = True) -> dict:
    """Windows are taken from the start of the test set, so a run with
    max_windows=n always uses the same tokens (paired comparison across configs)."""
    model.eval()
    n_windows = ids.numel() // seq_len
    if max_windows is not None:
        n_windows = min(n_windows, max_windows)
    nll_sum, n_tok = 0.0, 0
    per_window = []
    t_start = time.time()
    for w in range(n_windows):
        chunk = ids[w * seq_len:(w + 1) * seq_len].unsqueeze(0).to(device)
        out = model(chunk, labels=chunk)
        # HF loss is the mean over seq_len - 1 predicted tokens
        nll = out.loss.float().item()
        per_window.append(nll)
        nll_sum += nll * (seq_len - 1)
        n_tok += seq_len - 1
        if progress:
            el = time.time() - t_start
            eta = el / (w + 1) * (n_windows - w - 1)
            print(f"  window {w + 1}/{n_windows}  running ppl {math.exp(nll_sum / n_tok):.3f}  "
                  f"elapsed {el / 60:.1f} min  eta {eta / 60:.1f} min", flush=True)
    return {"ppl": math.exp(nll_sum / n_tok), "nll": nll_sum / n_tok, "tokens": n_tok,
            "windows": n_windows, "nll_per_window": per_window}
