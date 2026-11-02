"""Compare two A2 runs image by image (findings F6).

    python validation/a2_spread.py results/a2_spread_fast results/a2_spread_ref

For every config present in both results.json files and every simulator present in both,
prints accuracy in each run, the accuracy difference, and the fraction of images whose
predicted class differs. Writes <run_b>/spread.md.

Used for GATE A2: cimsim with backend 'fast' and 'reference' has identical semantics and differs
only in float rounding, so the difference between the two is the run-to-run spread of one simulator.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def disagreement(a: str, b: str) -> float:
    return 100.0 * sum(p != q for p, q in zip(a, b)) / len(a)


def main() -> None:
    da, db = Path(sys.argv[1]), Path(sys.argv[2])
    ra = json.loads((da / "results.json").read_text(encoding="utf-8"))
    rb = json.loads((db / "results.json").read_text(encoding="utf-8"))
    lines = [f"# Spread: {da.name} (backend {ra.get('backend', 'fast')}) vs {db.name} "
             f"(backend {rb.get('backend', 'fast')}), {ra['n_images']} images", "",
             "| Config | sim | acc A (%) | acc B (%) | diff B - A | images with different class (%) |",
             "|---|---|---|---|---|---|"]
    rows = []
    for cname in ra["runs"]:
        if cname not in rb["runs"]:
            continue
        for sim in ("cimsim", "asim"):
            a, b = ra["runs"][cname].get(sim), rb["runs"][cname].get(sim)
            if not a or not b:
                continue
            dis = f"{disagreement(a['pred'], b['pred']):.1f}" if "pred" in a and "pred" in b else ""
            rows.append(f"| {cname} | {sim} | {a['acc']:.2f} | {b['acc']:.2f} | {b['acc'] - a['acc']:+.2f} | {dis} |")
    # cimsim vs ASiM inside each run, where both are present
    for tag, r, d in (("A", ra, da), ("B", rb, db)):
        for cname, run in r["runs"].items():
            if "cimsim" in run and "asim" in run and "pred" in run["cimsim"] and "pred" in run["asim"]:
                rows.append(f"| {cname} | cimsim vs ASiM (run {tag}) | {run['asim']['acc']:.2f} | "
                            f"{run['cimsim']['acc']:.2f} | {run['cimsim']['acc'] - run['asim']['acc']:+.2f} | "
                            f"{disagreement(run['cimsim']['pred'], run['asim']['pred']):.1f} |")
    text = "\n".join(lines + rows) + "\n"
    (db / "spread.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
