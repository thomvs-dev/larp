"""A5 sweep: perplexity vs ADC bits and nrow, per layer type (one role changed at a time, or all layers).

Each point is one run of scripts/run_ppl.py with a generated config: the base config (default
config/experiments/a5_base_adc8cal.yaml: W8A8 bit-serial, nrow 128, MSE-calibrated ADC 8 b on every
layer, which reads every MAC up to 127 exactly) with `nrow` and `adc_bits` changed for one role
(qkv, o, up_gate, down) or for all layers. (The first pass used a4_stat_adc6cal.yaml as base; on
GPT-2 that base alone collapses, so per-role points were not informative.)
A fake-quant run on the same tokens is the reference.

Descent per (role, nrow): ADC bits from high to low; after the first point whose PPL is above
--collapse × the fake-quant PPL, lower bits are skipped and reported as "not run (below a collapsed
point)". Assumption: with an MSE-calibrated ADC, fewer bits never give a lower error.

Redundant points are skipped: ADC bits b with 2^b - 1 > 2 * nrow (bit-serial) read every MAC exactly
already at b - 1 bits (full scale (2^(b-1) - 1) / 1 or / 2 >= nrow - 1), e.g. 8 b at nrow 64.
--all-bits runs them anyway.

Resumable: a point whose result.json exists is not rerun (delete the folder to rerun it).
Calibration histograms come from the cache (cimsim/calib_cache.py); a new nrow for one role
recalibrates only that role's layers.

Examples:
    python scripts/a5_sweep.py --model facebook/opt-125m --name a5s_opt125m_ba --plan
    python scripts/a5_sweep.py --model facebook/opt-125m --name a5s_opt125m_ba --threads 6
    python scripts/a5_sweep.py --model facebook/opt-125m --name a5s_opt125m_ba --table
    # statistical mode at chosen points (validation against the bit-accurate sweep):
    python scripts/a5_sweep.py --model facebook/opt-125m --name a5s_opt125m_stat --mode statistical \\
        --points down:128:5,qkv:256:6
Results: results/<name>/<point>/ (run_ppl output), results/<name>/summary.{json,md}.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ROLES = ("qkv", "o", "up_gate", "down", "all")


def point_name(role: str, nrow: int, bits: int) -> str:
    return f"{role}_n{nrow}_adc{bits}"


def point_config(base: dict, mode: str, role: str, nrow: int, bits: int) -> dict:
    cfg = json.loads(json.dumps(base))
    cfg.setdefault("default", {})["mode"] = mode
    if role == "all":
        cfg["default"].update({"nrow": nrow, "adc_bits": bits})
    else:
        cfg.setdefault("policy", {})[role] = {"nrow": nrow, "adc_bits": bits}
    return cfg


def run_point(args, out: Path, name: str, cfg: dict | None) -> dict:
    d = out / name
    res_file = ROOT / "results" / args.name / name / "result.json"
    if res_file.exists():
        return json.loads(res_file.read_text(encoding="utf-8"))
    d.mkdir(parents=True, exist_ok=True)
    if cfg is None:
        cfg_arg = None
    else:
        cfg_arg = out / "configs" / f"{name}.yaml"
        cfg_arg.parent.mkdir(parents=True, exist_ok=True)
        cfg_arg.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    cmd = [sys.executable, str(ROOT / "scripts" / "run_ppl.py"), "--model", args.model,
           "--config", str(cfg_arg) if cfg_arg else "none", "--name", f"{args.name}/{name}",
           "--max-tokens", str(args.tokens)]
    if args.threads:
        cmd += ["--threads", str(args.threads)]
    if args.free_sat_plans and cfg is not None:
        cmd += ["--free-sat-plans"]
    if args.window_checkpoint and cfg is not None:
        cmd += ["--window-checkpoint"]
    t0 = time.time()
    print(f"--- {name} ...", flush=True)
    for attempt in range(1, args.attempts + 1):
        log_path = d / ("log.txt" if attempt == 1 else f"log_attempt{attempt}.txt")
        with open(log_path, "w", encoding="utf-8") as log:
            r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        if r.returncode == 0 and res_file.exists():
            break
        print(f"    attempt {attempt} failed (exit {r.returncode}); see {log_path}", flush=True)
    else:
        (d / "failed.txt").write_text(f"exit {r.returncode} after {args.attempts} attempts\n", encoding="utf-8")
        return None
    res = json.loads(res_file.read_text(encoding="utf-8"))
    print(f"    PPL {res['ppl']:.3f}  ({(time.time() - t0) / 60:.1f} min, calibration cache "
          f"{res.get('calib_cache')})", flush=True)
    if res.get("stat_inexact_layers"):
        print(f"    NOT VALID in statistical mode: {res['stat_inexact_layers']} layers with an ADC step "
              f"that is not 1/n MAC unit (findings F13)", flush=True)
    return res


def redundant(nrow: int, bits: int) -> bool:
    return 2 ** bits - 1 > 2 * nrow


def parse_points(s: str) -> list[tuple[str, int, int]]:
    pts = []
    for p in s.split(","):
        role, n, b = p.split(":")
        assert role in ROLES, role
        pts.append((role, int(n), int(b)))
    return pts


def summary(args, out: Path, ref: dict | None) -> None:
    rows = []
    for d in sorted(out.iterdir()):
        f = d / "result.json"
        if not f.exists() or d.name == "fake_quant":
            continue
        parts = d.name.rsplit("_", 2)
        role, n, b = parts[0], int(parts[1][1:]), int(parts[2][3:])
        res = json.loads(f.read_text(encoding="utf-8"))
        ev = json.loads((d / "events.json").read_text(encoding="utf-8")) if (d / "events.json").exists() else {}
        rows.append({"point": d.name, "role": role, "nrow": n, "adc_bits": b, "ppl": res["ppl"],
                     "ratio_to_fake_quant": res["ppl"] / ref["ppl"] if ref else None,
                     "adc_conversions": ev.get("adc_conversions"), "seconds": res.get("seconds"),
                     "calib_cache": res.get("calib_cache"), "git": res.get("git"),
                     "stat_inexact_layers": res.get("stat_inexact_layers")})
    s = {"model": args.model, "mode": args.mode, "tokens": args.tokens, "collapse_factor": args.collapse,
         "fake_quant_ppl": ref["ppl"] if ref else None, "points": rows}
    (out / "summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    lines = [f"# A5 sweep `{args.name}`", "",
             f"{args.model}, {args.mode}, first {args.tokens} WikiText-2 test tokens. Base: other layers "
             f"as `{Path(args.base).name}`. Fake-quant PPL on the same tokens: "
             f"{ref['ppl']:.3f}." if ref else "", "",
             "Cells: PPL (ratio to fake-quant). `—` = not run. `*` = statistical mode not valid at this "
             "point (ADC step not 1/n MAC unit on some layers, findings F13).", ""]
    bits_all = sorted({r["adc_bits"] for r in rows}, reverse=True)
    for role in ROLES:
        rr = [r for r in rows if r["role"] == role]
        if not rr:
            continue
        lines += [f"## {role}", "", "| nrow | " + " | ".join(f"ADC {b} b" for b in bits_all) + " |",
                  "|---|" + "---|" * len(bits_all)]
        for n in sorted({r["nrow"] for r in rr}):
            cells = []
            for b in bits_all:
                m = [r for r in rr if r["nrow"] == n and r["adc_bits"] == b]
                star = "*" if m and m[0].get("stat_inexact_layers") else ""
                cells.append(f"{m[0]['ppl']:.2f} ({m[0]['ratio_to_fake_quant']:.3f}){star}" if m and ref else
                             (f"{m[0]['ppl']:.2f}{star}" if m else "—"))
            lines.append(f"| {n} | " + " | ".join(cells) + " |")
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"summary: {out / 'summary.md'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True, help="results/<name>/ holds every point")
    ap.add_argument("--base", default=str(ROOT / "config/experiments/a5_base_adc8cal.yaml"))
    ap.add_argument("--mode", default="bit_accurate", choices=["bit_accurate", "statistical"])
    ap.add_argument("--roles", default="all,qkv,o,up_gate,down")
    ap.add_argument("--nrows", default="128,64,256")
    ap.add_argument("--bits", default="8,7,6,5,4")
    ap.add_argument("--points", default=None, help="explicit role:nrow:bits list instead of the descent")
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--collapse", type=float, default=2.0, help="PPL / fake-quant PPL above which lower bits are skipped")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--free-sat-plans", action="store_true", help="passed to run_ppl.py (memory, see there)")
    ap.add_argument("--window-checkpoint", action="store_true",
                    help="passed to run_ppl.py: a retried point continues at the first unfinished window")
    ap.add_argument("--attempts", type=int, default=2, help="runs per point before it is marked failed and skipped")
    ap.add_argument("--all-bits", action="store_true", help="do not skip redundant high-bit points")
    ap.add_argument("--plan", action="store_true", help="print the points (descent order) and exit")
    ap.add_argument("--table", action="store_true", help="rebuild summary.{json,md} from finished points")
    args = ap.parse_args()

    out = ROOT / "results" / args.name
    out.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(Path(args.base).read_text(encoding="utf-8")) or {}
    roles = [r for r in args.roles.split(",") if r]
    for r in roles:
        assert r in ROLES, f"unknown role {r}; one of {ROLES}"
    nrows = [int(x) for x in args.nrows.split(",")]
    bits = sorted((int(x) for x in args.bits.split(",")), reverse=True)

    ref_file = out / "fake_quant" / "result.json"
    if args.table:
        summary(args, out, json.loads(ref_file.read_text()) if ref_file.exists() else None)
        return
    if args.plan:
        pts = parse_points(args.points) if args.points else [(r, n, b) for n in nrows for r in roles for b in bits
                                                             if args.all_bits or not redundant(n, b)]
        print(f"{len(pts)} points at most (descent stops per (role, nrow) after a collapse), + fake-quant reference")
        for p in pts:
            done = (out / point_name(*p) / "result.json").exists()
            print(("  done " if done else "       ") + point_name(*p))
        return

    fq = point_config(base, "fake_quant", "all", base.get("default", {}).get("nrow", 128),
                      base.get("default", {}).get("adc_bits", 6))
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    print(f"git {head}, base {Path(args.base).name}", flush=True)
    ref = run_point(args, out, "fake_quant", fq)
    if ref is None:
        raise SystemExit("fake-quant reference failed; see results/<name>/fake_quant/")
    limit = args.collapse * ref["ppl"]
    if args.points:
        for role, n, b in parse_points(args.points):
            if run_point(args, out, point_name(role, n, b), point_config(base, args.mode, role, n, b)) is None:
                print(f"    {point_name(role, n, b)} skipped after {args.attempts} failed attempts")
    else:
        for n in nrows:
            for role in roles:
                for b in bits:
                    if not args.all_bits and redundant(n, b):
                        continue
                    res = run_point(args, out, point_name(role, n, b), point_config(base, args.mode, role, n, b))
                    if res is None:
                        print(f"    {point_name(role, n, b)} skipped after {args.attempts} failed attempts; descent continues")
                        continue
                    if res["ppl"] > limit:
                        print(f"    collapsed (> {args.collapse} x fake-quant); lower bits skipped for {role}, nrow {n}")
                        break
    summary(args, out, ref)


if __name__ == "__main__":
    main()
