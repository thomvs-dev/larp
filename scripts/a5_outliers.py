"""A5: activation outlier statistics per linear layer (one FP32 forward pass, no CIM).

For every nn.Linear input (the activation that drives the CIM array rows), on the first --tokens
WikiText-2 test tokens:
- per-token peak ratio max|x| / rms(x) (median and p99 over tokens),
- channel outliers: per-channel rms over tokens; number of channels above 6x the median channel rms,
  and the share of total energy in the top 1 % of channels,
- the 8-bit per-token codes the CIM layer sees (same quantizer as `cimsim.quant.quantize_act`,
  signed, per token): fraction of exact zeros, fraction negative, fraction with |code| <= 1, and the
  density of each two's-complement input bit plane (bit 7 = sign bit; for a negative code the high
  planes are set, F3).
Per-layer rows go to results/<name>/layers.json, a per-role summary to results/<name>/summary.md.
Roles follow `cimsim.config.ROLE_PATTERNS` (qkv / o / up_gate / down); the LM head is skipped.

    python scripts/a5_outliers.py --model facebook/opt-125m --name a5o_opt125m --threads 14
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import statistics
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cimsim.config import ROLE_PATTERNS  # noqa: E402
from cimsim.quant import quantize_act  # noqa: E402
from eval.perplexity import load_wikitext2_ids  # noqa: E402
from eval.runner import load_model  # noqa: E402

ROLES = ("qkv", "o", "up_gate", "down")


def role_of(name: str) -> str | None:
    for role in ROLES:
        if any(fnmatch.fnmatch(name, p) for p in ROLE_PATTERNS[role]):
            return role
    return None


def layer_stats(x: torch.Tensor, x_bits: int = 8) -> dict:
    """x: [T, K] float activations of one layer input."""
    x = x.float()
    rms_tok = x.pow(2).mean(1).sqrt().clamp_min(1e-12)
    peak = x.abs().amax(1) / rms_tok                                 # per token
    ch_rms = x.pow(2).mean(0).sqrt()                                 # per channel
    med = float(ch_rms.median())
    e = ch_rms.pow(2)
    k1 = max(1, ch_rms.numel() // 100)
    q, _ = quantize_act(x, x_bits, "per_token", signed=True)         # same quantizer as the CIM layers
    qi = q.round().to(torch.int64)
    u = torch.where(qi < 0, qi + (1 << x_bits), qi)                  # two's-complement codes
    planes = [float(((u >> b) & 1).double().mean()) for b in range(x_bits)]
    return {
        "K": int(x.shape[1]),
        "peak_ratio_median": float(peak.median()), "peak_ratio_p99": float(torch.quantile(peak, 0.99)),
        "outlier_channels_6x": int((ch_rms > 6 * med).sum()) if med > 0 else 0,
        "top1pct_channel_energy_share": float(e.topk(k1).values.sum() / e.sum()) if float(e.sum()) > 0 else 0.0,
        "code_zero_frac": float((qi == 0).double().mean()),
        "code_negative_frac": float((qi < 0).double().mean()),
        "code_abs_le1_frac": float((qi.abs() <= 1).double().mean()),
        "plane_density": planes,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    model, tok, _ = load_model(args.model, "cpu")
    model.eval()
    ids = load_wikitext2_ids(tok)[:args.tokens].unsqueeze(0)
    rows, hooks = {}, []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and role_of(name):
            def hook(m, inp, out, name=name):
                rows[name] = {"role": role_of(name), **layer_stats(inp[0].reshape(-1, inp[0].shape[-1]))}
            hooks.append(mod.register_forward_hook(hook))
        elif type(mod).__name__ == "Conv1D" and role_of(name):     # GPT-2 uses Conv1D for its linears
            def hook(m, inp, out, name=name):
                rows[name] = {"role": role_of(name), **layer_stats(inp[0].reshape(-1, inp[0].shape[-1]))}
            hooks.append(mod.register_forward_hook(hook))
    with torch.no_grad():
        model(ids)
    for h in hooks:
        h.remove()

    out = ROOT / "results" / args.name
    out.mkdir(parents=True, exist_ok=True)
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    (out / "layers.json").write_text(json.dumps({"model": args.model, "tokens": int(ids.shape[1]), "git": git,
                                                 "layers": rows}, indent=1), encoding="utf-8")
    lines = [f"# Activation statistics `{args.name}`", "",
             f"{args.model}, FP32, first {ids.shape[1]} WikiText-2 test tokens, inputs of the CIM layers "
             f"(8-bit per-token codes as in the CIM layers). Per role: median over layers [max].", "",
             "| Role | layers | peak/rms (median tok) | peak/rms (p99 tok) | outlier ch > 6x | top-1 % ch energy "
             "| code = 0 | code < 0 | abs(code) <= 1 | sign-plane density |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for role in ROLES:
        rr = [r for r in rows.values() if r["role"] == role]
        if not rr:
            continue

        def cell(key, fmt="{:.2f}", f=lambda r, k: r[k]):
            v = [f(r, key) for r in rr]
            return f"{fmt.format(statistics.median(v))} [{fmt.format(max(v))}]"
        lines.append(f"| {role} | {len(rr)} | {cell('peak_ratio_median', '{:.1f}')} | {cell('peak_ratio_p99', '{:.1f}')} "
                     f"| {cell('outlier_channels_6x', '{:.0f}')} | {cell('top1pct_channel_energy_share')} "
                     f"| {cell('code_zero_frac')} | {cell('code_negative_frac')} | {cell('code_abs_le1_frac')} "
                     f"| {cell('plane_density', '{:.2f}', lambda r, k: r[k][-1])} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"summary: {out / 'summary.md'}")


if __name__ == "__main__":
    main()
