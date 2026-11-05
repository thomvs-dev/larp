"""Paired comparison of zero-shot runs (same documents) against a reference run.

Example:
    python scripts/zeroshot_compare.py --ref results/zs_qwen15b_w8a8 \
        results/zs_qwen15b_fp32 results/zs_qwen15b_stat_adc6cal --out results/zs_qwen15b_summary.md

For each task and metric: accuracy of every run, difference to the reference on the common documents,
paired standard error / 95 % interval and exact McNemar p.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.zeroshot import METRICS, load_task_result, paired_compare  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="reference result folder")
    ap.add_argument("runs", nargs="+", help="result folders to compare with the reference")
    ap.add_argument("--tasks", default=None, help="comma-separated; default = tasks present in the reference")
    ap.add_argument("--out", default=None, help="write the table as markdown here")
    args = ap.parse_args()

    ref = Path(args.ref)
    tasks = args.tasks.split(",") if args.tasks else sorted(p.stem[3:] for p in ref.glob("zs_*.json"))
    lines = [f"Reference: `{ref.as_posix()}`. Δ = run − reference on the same documents (pp); "
             "95 % interval from the paired standard error; p = exact McNemar.", "",
             "| Task | Metric | Run | n | Acc | Ref acc | Δ (pp) | 95 % interval (pp) | p |",
             "|---|---|---|---|---|---|---|---|---|"]
    for task in tasks:
        r = load_task_result(ref, task)
        for m in METRICS:
            for run in args.runs:
                try:
                    x = load_task_result(Path(run), task)
                except FileNotFoundError:
                    continue
                c = paired_compare(r["docs"], x["docs"], m)
                lo, hi = c["ci95"]
                lines.append(f"| {task} | {m} | `{Path(run).name}` | {c['n']} | {100 * c['acc_b']:.2f} | "
                             f"{100 * c['acc_a']:.2f} | {100 * c['diff']:+.2f} | {100 * lo:+.2f} … {100 * hi:+.2f} | "
                             f"{c['mcnemar_p']:.3g} |")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
