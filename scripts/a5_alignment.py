"""A5: ADC step alignment, from the step-1 bit-accurate sweeps (no new runs).

For every sweep point, read the calibrated full scales of the changed layers (adc_full_scale.json) and
compute the ADC step in MAC units, step = full scale / (2^b − 1). A layer is aligned when its step is
1/n MAC unit (n ≥ 1 integer: every integer MAC value has its own code). The script relates the largest
step of a point to its perplexity / W8A8.

    python scripts/a5_alignment.py      # -> results/a5_alignment/summary.{json,md}
"""

from __future__ import annotations

import fnmatch
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cimsim.config import ROLE_PATTERNS  # noqa: E402

SWEEPS = {"OPT-125M": "a5s2_opt125m_ba", "GPT-2": "a5s2_gpt2_ba"}
ROLES = ("qkv", "o", "up_gate", "down")


def aligned(step: float) -> bool:
    inv = 1.0 / step
    return abs(inv - round(inv)) < 1e-6 and round(inv) >= 1


def collect() -> list[dict]:
    rows = []
    for model, sweep in SWEEPS.items():
        s = json.loads((ROOT / "results" / sweep / "summary.json").read_text(encoding="utf-8"))
        for p in s["points"]:
            f = ROOT / "results" / sweep / p["point"] / "adc_full_scale.json"
            if not f.exists():
                continue
            fs = json.loads(f.read_text(encoding="utf-8"))
            levels = 2 ** p["adc_bits"] - 1
            roles = ROLES if p["role"] == "all" else (p["role"],)
            steps = []
            for name, v in fs.items():
                if not any(fnmatch.fnmatch(name, pat) for r in roles for pat in ROLE_PATTERNS[r]):
                    continue
                flat = [x for row in v for x in row] if isinstance(v[0], list) else list(v)
                steps.append(max(flat) / levels)
            if not steps:
                continue
            rows.append({"model": model, "point": p["point"], "role": p["role"], "nrow": p["nrow"],
                         "adc_bits": p["adc_bits"], "ratio_to_fake_quant": p["ratio_to_fake_quant"],
                         "max_step": max(steps), "layers": len(steps),
                         "aligned_layers": sum(aligned(x) for x in steps)})
    return rows


def main() -> None:
    rows = collect()
    out = ROOT / "results" / "a5_alignment"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    al = [r for r in rows if r["aligned_layers"] == r["layers"]]
    na = [r for r in rows if r["aligned_layers"] < r["layers"]]
    worst_al = max(r["ratio_to_fake_quant"] for r in al)
    best_na = min(r["ratio_to_fake_quant"] for r in na)
    lines = ["# A5: ADC step alignment (step-1 bit-accurate sweeps, 8,192 tokens)", "",
             "Step = calibrated full scale / (2^b − 1), in MAC units, per changed layer. Aligned = step is "
             "1/n MAC unit for every changed layer.", "",
             f"- Points with every layer aligned: {len(al)}; PPL / W8A8 at most {worst_al:.3f}.",
             f"- Points with at least one layer at a step > 1 MAC unit: {len(na)}; PPL / W8A8 at least "
             f"{best_na:.3f}.", "",
             "| Model | Point | PPL / W8A8 | largest step (MAC) | aligned layers |", "|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["model"], r["role"], r["nrow"], r["adc_bits"])):
        lines.append(f"| {r['model']} | {r['point']} | {r['ratio_to_fake_quant']:.3f} | {r['max_step']:.3f} "
                     f"| {r['aligned_layers']} / {r['layers']} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"aligned {len(al)} (max ratio {worst_al:.3f}), not aligned {len(na)} (min ratio {best_na:.3f})")


if __name__ == "__main__":
    main()
