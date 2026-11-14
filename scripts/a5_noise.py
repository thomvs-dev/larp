"""A5 noise sensitivity per layer type (reduced plan, CLAUDE.md A5).

Operating point: every layer on analog CIM, W8A8 bit-serial, nrow 128, calibrated ADC 6 b, statistical mode.
Gaussian analog noise (rand_noise_sigma, % of nominal full scale) is switched on for one layer type at a
time (qkv / o / up_gate / down) or for all layers; the other layers stay noise-free. Each (role, sigma) is
run with several seeds (noise near the cliff is chaotic, findings F9). References on the same tokens:
fake-quant and the noise-free operating point.

    python scripts/a5_noise.py --model facebook/opt-125m --name a5n_opt125m --threads 14
    python scripts/a5_noise.py --model Qwen/Qwen2.5-1.5B --name a5n_qwen15b --threads 14 --seeds 0
    python scripts/a5_noise.py --model facebook/opt-125m --name a5n_opt125m_ba --mode bit_accurate \
        --roles up_gate --sigmas 0.10                             # bit-accurate check of one cell
    python scripts/a5_noise.py --model ... --name ... --plan      # list the runs
    python scripts/a5_noise.py --model ... --name ... --table     # rebuild the summary

Resumable: finished runs are skipped. Output: results/<name>/<run>/, results/<name>/summary.{json,md}.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a5_sweep import ROOT, run_point  # noqa: E402

ROLES = ("qkv", "o", "up_gate", "down", "all")


def run_name(role: str, sigma: float, seed: int) -> str:
    return f"{role}_rn{round(sigma * 1000):04d}_s{seed}"          # sigma 0.10 % -> rn0100


def noise_config(base: dict, role: str, sigma: float, seed: int) -> dict:
    cfg = json.loads(json.dumps(base))
    cfg.setdefault("default", {})["seed"] = seed
    if role == "all":
        cfg["default"]["rand_noise_sigma"] = sigma
    else:
        cfg.setdefault("policy", {})[role] = {"rand_noise_sigma": sigma}
    return cfg


def summary(args, out: Path) -> None:
    def ppl(name):
        f = out / name / "result.json"
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
    fq, clean = ppl("fake_quant"), ppl("noise_free")
    rows = []
    for role in ROLES:
        for sigma in args.sigma_list:
            runs = [(s, ppl(run_name(role, sigma, s))) for s in args.seed_list]
            vals = [r["ppl"] for _, r in runs if r]
            if not vals:
                continue
            rows.append({"role": role, "sigma_pct": sigma, "seeds": [s for s, r in runs if r], "ppl": vals,
                         "median": statistics.median(vals), "min": min(vals), "max": max(vals),
                         "stat_inexact_layers": max((r.get("stat_inexact_layers") or 0) for _, r in runs if r)})
    s = {"model": args.model, "tokens": args.tokens, "base": Path(args.base).name,
         "fake_quant_ppl": fq["ppl"] if fq else None, "noise_free_ppl": clean["ppl"] if clean else None, "rows": rows}
    (out / "summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    ref = clean["ppl"] if clean else None
    lines = [f"# A5 noise per layer type `{args.name}`", "",
             f"{args.model}, {args.mode or 'statistical'}, first {args.tokens} WikiText-2 test tokens, base `{Path(args.base).name}`. "
             f"Fake-quant {fq['ppl']:.3f}, noise-free analog {clean['ppl']:.3f}." if fq and clean else "", "",
             "Noise on one layer type, the others noise-free. Cells: median PPL over seeds (ratio to noise-free "
             "analog) [min – max]. `*` = statistical mode not valid (findings F13).", "",
             "| Layer type | " + " | ".join(f"sigma {x:g} %" for x in args.sigma_list) + " |",
             "|---|" + "---|" * len(args.sigma_list)]
    for role in ROLES:
        cells = []
        for sigma in args.sigma_list:
            m = [r for r in rows if r["role"] == role and r["sigma_pct"] == sigma]
            if not m:
                cells.append("—")
                continue
            r = m[0]
            c = f"{r['median']:.2f}" + (f" ({r['median'] / ref:.3f})" if ref else "")
            if len(r["ppl"]) > 1:
                c += f" [{r['min']:.2f} – {r['max']:.2f}]"
            cells.append(c + ("*" if r["stat_inexact_layers"] else ""))
        if any(c != "—" for c in cells):
            lines.append(f"| {role} | " + " | ".join(cells) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"summary: {out / 'summary.md'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--base", default=str(ROOT / "config/experiments/a4_stat_adc6cal.yaml"))
    ap.add_argument("--roles", default="qkv,o,up_gate,down,all")
    ap.add_argument("--sigmas", default="0.10,0.20", help="rand_noise_sigma values, %% of nominal full scale")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--mode", default=None, choices=["statistical", "bit_accurate"],
                    help="override the base mode (bit_accurate: check of statistical results on small models)")
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--table", action="store_true")
    args = ap.parse_args()
    args.free_sat_plans = False
    args.window_checkpoint = False          # noisy runs: a resumed window would use a different random stream
    args.sigma_list = [float(x) for x in args.sigmas.split(",")]
    args.seed_list = [int(x) for x in args.seeds.split(",")]
    roles = [r for r in args.roles.split(",") if r]
    for r in roles:
        assert r in ROLES, f"unknown role {r}; one of {ROLES}"

    out = ROOT / "results" / args.name
    out.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(Path(args.base).read_text(encoding="utf-8")) or {}
    if args.mode:
        base.setdefault("default", {})["mode"] = args.mode
    runs = [(sigma, role, seed) for sigma in args.sigma_list for role in roles for seed in args.seed_list]
    if args.table:
        summary(args, out)
        return
    if args.plan:
        print(f"{len(runs)} noisy runs + fake_quant + noise_free")
        for sigma, role, seed in runs:
            n = run_name(role, sigma, seed)
            print(("  done " if (out / n / "result.json").exists() else "       ") + n)
        return

    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    print(f"git {head}, base {Path(args.base).name}", flush=True)
    fq = json.loads(json.dumps(base))
    fq.setdefault("default", {})["mode"] = "fake_quant"
    if run_point(args, out, "fake_quant", fq) is None or run_point(args, out, "noise_free", base) is None:
        raise SystemExit("reference run failed; see results/<name>/fake_quant or noise_free")
    for sigma, role, seed in runs:                      # low sigma first, all roles, then the next sigma
        n = run_name(role, sigma, seed)
        if run_point(args, out, n, noise_config(base, role, sigma, seed)) is None:
            print(f"    {n} skipped after {args.attempts} failed attempts")
        summary(args, out)


if __name__ == "__main__":
    main()
