"""A6: mitigations at an operating point, with the event counts for the equal-cost comparison.

At each point (all layers at nrow N, ADC b bits, calibrated full scale) the script runs:
    base        all layers analog at (N, b)
    had_down    + Hadamard rotation of the down-projection input (cimsim.models.rotate)
    had_all     + Hadamard rotation of every CIM-layer input
    dcim_down   down projections on digital CIM (exact, no ADC)
    down_b+1    down projections with one more ADC bit
and two references on the same tokens: fake_quant (W8A8) and fake_quant_had (W8A8 with every
input rotated). Results go to results/<name>/<run>/; summary.{json,md} lists PPL, ratio to
fake-quant and the event counts (analog ADC conversions, DCIM column ops, Hadamard additions);
pareto.png plots PPL against analog ADC conversions.

    python scripts/a6_mitigation.py --model gpt2 --name a6m_gpt2_ba --mode bit_accurate --points 128:6,256:7 --tokens 8192 --threads 14
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from a5_sweep import run_point  # noqa: E402

ALL = ["qkv", "o", "up_gate", "down"]
VARIANTS = ("base", "had_down", "had_all", "dcim_down", "down_b+1")


def point_config(base: dict, mode: str, nrow: int, bits: int, variant: str) -> dict:
    c = copy.deepcopy(base)
    c["default"].update(mode=mode, nrow=nrow, adc_bits=bits)
    if variant == "had_down":
        c["hadamard"] = {"roles": ["down"]}
    elif variant == "had_all":
        c["hadamard"] = {"roles": list(ALL)}
    elif variant == "dcim_down":
        c["policy"] = {"down": "dcim"}
    elif variant == "down_b+1":
        c["policy"] = {"down": {"adc_bits": bits + 1}}
    elif variant != "base":
        raise ValueError(variant)
    return c


def run_name(nrow: int, bits: int, variant: str) -> str:
    return f"n{nrow}_adc{bits}_{variant.replace('+', 'p')}"


def load(out: Path, name: str):
    r, e = out / name / "result.json", out / name / "events.json"
    if not r.exists():
        return None
    res = json.loads(r.read_text(encoding="utf-8"))
    ev = json.loads(e.read_text(encoding="utf-8")) if e.exists() else {}
    ev = ev.get("total", ev)
    return res, ev


def summary(args, out: Path, points) -> None:
    fq = load(out, "fake_quant")
    fq_ppl = fq[0]["ppl"] if fq else None
    rows = []
    for name in ["fake_quant", "fake_quant_had"] + [run_name(n, b, v) for n, b in points for v in args.variants]:
        got = load(out, name)
        if got is None:
            continue
        res, ev = got
        rows.append({"run": name, "ppl": res["ppl"], "ratio_to_fake_quant": res["ppl"] / fq_ppl if fq_ppl else None,
                     "adc_conversions_analog": ev.get("adc_conversions_analog", 0),
                     "dcim_column_ops": ev.get("dcim_column_ops", 0), "hadamard_adds": ev.get("hadamard_adds", 0),
                     "stat_inexact_layers": res.get("stat_inexact_layers", 0), "git": res.get("git")})
    (out / "summary.json").write_text(json.dumps({"model": args.model, "mode": args.mode, "tokens": args.tokens,
                                                  "rows": rows}, indent=1), encoding="utf-8")
    lines = [f"# A6 mitigations `{args.name}`", "",
             f"{args.model}, {args.mode}, first {args.tokens} WikiText-2 test tokens. All layers analog at the "
             f"point unless the variant says otherwise. `*` = statistical result not valid (F13). "
             f"Fake-quant rows are digital references (no conversions).", "",
             "| Run | PPL | / fake-quant | analog ADC conversions | DCIM column ops | Hadamard adds |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        star = "*" if r["stat_inexact_layers"] else ""
        ratio = f"{r['ratio_to_fake_quant']:.3f}" if r["ratio_to_fake_quant"] else "-"
        conv = "-" if r["run"].startswith("fake_quant") else f"{r['adc_conversions_analog']:.3e}"
        lines.append(f"| {r['run']} | {r['ppl']:.3f}{star} | {ratio} | {conv} "
                     f"| {r['dcim_column_ops']:.3e} | {r['hadamard_adds']:.3e} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        for r in rows:
            if r["run"].startswith("fake_quant"):
                ax.axhline(r["ppl"], ls="--" if r["run"] == "fake_quant" else ":", lw=1, color="gray")
                continue
            ax.scatter(r["adc_conversions_analog"], r["ppl"], s=20)
            ax.annotate(r["run"], (r["adc_conversions_analog"], r["ppl"]), fontsize=6)
        ax.set_xlabel("analog ADC conversions")
        ax.set_ylabel("perplexity")
        ax.set_yscale("log")
        ax.set_title(f"{args.model}, {args.mode} (dashed: W8A8 fake-quant, dotted: rotated)")
        fig.tight_layout()
        fig.savefig(out / "pareto.png", dpi=150)
    except Exception as e:  # plot is optional
        print(f"plot skipped: {e}")
    print(f"summary: {out / 'summary.md'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", default="bit_accurate", choices=["statistical", "bit_accurate"])
    ap.add_argument("--base", default=str(ROOT / "config/experiments/a4_stat_adc6cal.yaml"))
    ap.add_argument("--points", default="128:6,256:7", help="nrow:adc_bits, all layers")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("--free-sat-plans", action="store_true")
    ap.add_argument("--window-checkpoint", action="store_true")
    ap.add_argument("--plan", action="store_true", help="list the runs and stop")
    ap.add_argument("--table", action="store_true", help="only rebuild summary / plot")
    args = ap.parse_args()
    args.variants = [v for v in args.variants.split(",") if v]
    for v in args.variants:
        if v not in VARIANTS:
            raise SystemExit(f"unknown variant {v}; one of {VARIANTS}")
    points = [tuple(int(x) for x in p.split(":")) for p in args.points.split(",") if p]
    out = ROOT / "results" / args.name
    base = yaml.safe_load(open(args.base, encoding="utf-8"))
    runs = [("fake_quant", {"default": {"mode": "fake_quant", "w_bits": 8, "x_bits": 8}}),
            ("fake_quant_had", {"default": {"mode": "fake_quant", "w_bits": 8, "x_bits": 8},
                                "hadamard": {"roles": list(ALL)}})]
    runs += [(run_name(n, b, v), point_config(base, args.mode, n, b, v)) for n, b in points for v in args.variants]
    if args.plan:
        for name, _ in runs:
            print(name)
        return
    if not args.table:
        out.mkdir(parents=True, exist_ok=True)
        for name, cfg in runs:
            run_point(args, out, name, cfg)
            summary(args, out, points)
    summary(args, out, points)


if __name__ == "__main__":
    main()
