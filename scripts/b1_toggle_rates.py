"""Input toggle rate per linear layer of a HuggingFace model (bit-serial, MSB first,
per-token symmetric quantization). Input for the DCIM energy model (cimsim.circuits.dcim).

python scripts/b1_toggle_rates.py --model facebook/opt-125m --tokens 512 --bits 8 --name b1_toggle_opt125m
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cimsim.circuits.dcim import input_toggle_rate
from eval.perplexity import load_wikitext2_ids

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--name", required=True)
    ap.add_argument("--threads", type=int, default=None)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    tok = AutoTokenizer.from_pretrained(a.model)
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    ids = load_wikitext2_ids(tok, split="test")[: a.tokens].unsqueeze(0)
    qmax = 2 ** (a.bits - 1) - 1
    rates = {}

    def hook(name):
        def f(mod, inp, out):
            x = inp[0].reshape(-1, inp[0].shape[-1]).float()
            s = x.abs().amax(1, keepdim=True).clamp_min(1e-8) / qmax
            q = torch.round(x / s).clamp(-qmax, qmax)
            rates[name] = input_toggle_rate(q, a.bits, signed=True)
        return f

    for n, mod in m.named_modules():
        if isinstance(mod, (torch.nn.Linear,)) or type(mod).__name__ == "Conv1D":
            if "lm_head" not in n:
                mod.register_forward_hook(hook(n))
    with torch.no_grad():
        m(ids)
    by = {}
    for n, r in rates.items():
        by.setdefault(n.split(".")[-1], []).append(r)
    summ = {k: {"mean": sum(v) / len(v), "min": min(v), "max": max(v), "n": len(v)} for k, v in by.items()}
    try:
        git = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        git = "unknown"
    out = ROOT / "results" / a.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "toggle.json").write_text(json.dumps({"args": vars(a), "git": git, "quant": "per-token symmetric, signed",
                                                 "by_role": summ, "per_layer": rates}, indent=1), encoding="utf-8")
    for k, v in summ.items():
        print(f"{k:12s} mean {v['mean']:.3f}  min {v['min']:.3f}  max {v['max']:.3f}  (n={v['n']})")


if __name__ == "__main__":
    main()
