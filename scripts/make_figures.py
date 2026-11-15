"""Part A figures, one command, from the committed results (no model runs).

    python scripts/make_figures.py          # all figures -> figures/*.png
    python scripts/make_figures.py a5_alignment a6_pareto    # selected figures

Figures:
    a3_validation   statistical vs bit-accurate perplexity (GATE A3, A5 step 2, A6 check)
    a4_zeroshot     zero-shot accuracy, analog CIM − W8A8, paired 95 % intervals, result models
    a5_boundary     PPL / W8A8 vs ADC bits per layer type and nrow (bit-accurate, OPT-125M, GPT-2)
    a5_alignment    PPL / W8A8 vs largest calibrated ADC step (same points)
    a5_planes       density of each input bit plane per layer type (OPT-125M, GPT-2, Qwen2.5-1.5B)
    a5_noise        PPL vs analog noise, all layers noisy (OPT-125M, statistical)
    a6_pareto       mitigations: PPL / W8A8 vs analog ADC conversions per token
    headline        energy per token, digital INT8 vs SRAM CIM (Part B, results/b4_sweep/)
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "figures"
R = ROOT / "results"

# validated categorical palette (dataviz reference instance, light mode); fixed order, never cycled
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MK = ["o", "s", "D", "^", "v", "P", "X", "*"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ROLES = ["qkv", "o", "up_gate", "down", "all"]
ROLE_NAME = {"qkv": "q / k / v", "o": "attention out", "up_gate": "MLP up / gate", "down": "MLP down", "all": "all layers"}


def j(path):
    return json.loads((R / path).read_text(encoding="utf-8"))


def style(ax, title=None):
    if title:
        ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.tick_params(labelsize=7, colors=INK2)
    for s in ax.spines.values():
        s.set_color(GRID)


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"figures/{name}.png")


def logfmt(ax):
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


# --------------------------------------------------------------------------------------------------
def a5_boundary():
    models = [("OPT-125M", "a5s2_opt125m_ba"), ("GPT-2", "a5s2_gpt2_ba")]
    nrows = [64, 128, 256]
    fig, axes = plt.subplots(2, 5, figsize=(12, 5), sharex=True, sharey=True)
    for i, (mname, sweep) in enumerate(models):
        pts = j(f"{sweep}/summary.json")["points"]
        for k, role in enumerate(ROLES):
            ax = axes[i, k]
            for c, n in enumerate(nrows):
                p = sorted([q for q in pts if q["role"] == role and q["nrow"] == n], key=lambda q: q["adc_bits"])
                if not p:
                    continue
                ax.plot([q["adc_bits"] for q in p], [q["ratio_to_fake_quant"] for q in p], color=C[c], lw=2,
                        marker=MK[c], ms=6, mec="#fcfcfb", mew=1, label=f"nrow {n}", zorder=3)
                # boundary predicted by 2^b - 1 >= nrow / 2 (smallest such b)
            ax.axhline(2.0, color=INK2, lw=1, ls=":", zorder=1)
            logfmt(ax)
            style(ax, f"{mname}, {ROLE_NAME[role]}")
            if k == 0:
                ax.set_ylabel("perplexity / W8A8", fontsize=8, color=INK2)
            if i == 1:
                ax.set_xlabel("ADC bits", fontsize=8, color=INK2)
    axes[0, 0].set_yticks([1, 2, 10, 100, 1000])
    h, lab = axes[0, 0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color=INK2, ls=":", lw=1))
    lab.append("collapse threshold (2×)")
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("Bit-accurate, 8,192 WikiText-2 tokens, one layer type changed (others lossless at 8 b), "
                 "calibrated ADC", fontsize=9, color=INK2, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    save(fig, "a5_boundary")


def a5_alignment():
    subprocess.run([sys.executable, str(ROOT / "scripts" / "a5_alignment.py")], check=True, cwd=ROOT)
    rows = j("a5_alignment/summary.json")
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for c, model in enumerate(["OPT-125M", "GPT-2"]):
        rr = [r for r in rows if r["model"] == model]
        ax.scatter([r["max_step"] for r in rr], [r["ratio_to_fake_quant"] for r in rr], s=36, color=C[c],
                   marker=MK[c], edgecolors="#fcfcfb", linewidths=1, zorder=3, label=model)
    ax.axvline(1.0, color=INK2, lw=1, ls="--", zorder=1)
    ax.axhline(2.0, color=INK2, lw=1, ls=":", zorder=1)
    na = [r for r in rows if r["aligned_layers"] < r["layers"]]
    al = [r for r in rows if r["aligned_layers"] == r["layers"]]
    ax.text(0.42, 40, f"step 1 or 1/2 MAC unit:\n{len(al)} points, all ≤ "
            f"{max(r['ratio_to_fake_quant'] for r in al):.2f}× W8A8", fontsize=7, color=INK2)
    ax.text(2.05, 2.6, f"step > 1 MAC unit:\n{len(na)} points, all ≥ "
            f"{min(r['ratio_to_fake_quant'] for r in na):.2f}× W8A8", fontsize=7, color=INK2)
    logfmt(ax)
    ax.set_yticks([1, 2, 10, 100, 1000])
    ax.set_xlabel("largest calibrated ADC step of the changed layers (MAC units)", fontsize=8, color=INK2)
    ax.set_ylabel("perplexity / W8A8", fontsize=8, color=INK2)
    style(ax, "ADC step alignment, A5 step-1 points (bit-accurate)")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "a5_alignment")


def a5_planes():
    models = [("OPT-125M", "a5o_opt125m_sandbox"), ("GPT-2", "a5o_gpt2"), ("Qwen2.5-1.5B", "a5o_qwen15b")]
    act = {"OPT-125M": "ReLU", "GPT-2": "GELU", "Qwen2.5-1.5B": "SwiGLU"}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, (mname, run) in zip(axes, models):
        layers = j(f"{run}/layers.json")["layers"]
        for c, role in enumerate(ROLES[:4]):
            rr = [r["plane_density"] for r in layers.values() if r["role"] == role]
            med = [statistics.median(col) for col in zip(*rr)]
            lab = ROLE_NAME[role] + (f" (after {act[mname]})" if role == "down" else "")
            ax.plot(range(8), med, color=C[c], lw=2, marker=MK[c], ms=5, mec="#fcfcfb", mew=1, label=lab, zorder=3)
        ax.set_xticks(range(8))
        ax.set_xticklabels([str(b) if b < 7 else "7\nsign" for b in range(8)])
        ax.set_xlabel("input bit plane (8-bit two's complement)", fontsize=8, color=INK2)
        style(ax, mname)
        ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    axes[0].set_ylabel("fraction of rows with the bit set\n(median over layers)", fontsize=8, color=INK2)
    axes[0].set_ylim(-0.02, 1.0)
    fig.tight_layout()
    save(fig, "a5_planes")


def a5_noise():
    s = j("a5n_opt125m/summary.json")
    rows = [r for r in s["rows"] if r["role"] == "all"]
    rows.sort(key=lambda r: r["sigma_pct"])
    x = [0.0] + [r["sigma_pct"] for r in rows]
    med = [s["noise_free_ppl"]] + [r["median"] for r in rows]
    lo = [s["noise_free_ppl"]] + [r["min"] for r in rows]
    hi = [s["noise_free_ppl"]] + [r["max"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.fill_between(x, lo, hi, color=C[0], alpha=0.15, lw=0, zorder=2)
    ax.plot(x, med, color=C[0], lw=2, marker="o", ms=6, mec="#fcfcfb", mew=1, zorder=3,
            label="all layers noisy, median of 3 seeds (band: min-max)")
    ax.axhline(s["fake_quant_ppl"], color=INK2, lw=1, ls="--", label="W8A8 fake-quant")
    ax.axvline(100 * 0.1 / 128, color=INK2, lw=1, ls=":", label="σ = 0.078 % (half an ADC step = 5 σ)")
    logfmt(ax)
    ax.set_xlabel("analog noise σ (% of nominal full scale, 128 MAC units)", fontsize=8, color=INK2)
    ax.set_ylabel("perplexity", fontsize=8, color=INK2)
    style(ax, "OPT-125M, all layers nrow 128 / 6 b, statistical, 4,096 tokens")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "a5_noise")


def a3_validation():
    pts = []   # (bit-accurate, statistical, label group)
    for c in ["f5", "adc8", "adc7", "rn008", "rn010", "rn02_hyb6"]:
        g = j(f"a3g_opt125m_{c}/gate.json")
        st = statistics.median(v["ppl"] for v in g["statistical"].values())
        pts.append((g["ref_ppl"], st, "GATE A3, OPT-125M"))
    for m, mn in [("opt125m", "OPT-125M"), ("gpt2", "GPT-2")]:
        ba = {p["point"]: p["ppl"] for p in j(f"a5s2_{m}_ba/summary.json")["points"]}
        al = {r["point"]: r for r in j("a5_alignment/summary.json") if r["model"] == mn}
        for p in j(f"a5v2_{m}_stat/summary.json")["points"]:
            r = al.get(p["point"])
            exact = r is not None and r["aligned_layers"] == r["layers"]
            pts.append((ba[p["point"]], p["ppl"], "A5 step 2, step 1/n MAC" if exact else "A5 step 2, step > 1 MAC (F13)"))
    a6s = {r["run"]: r["ppl"] for r in j("a6m_opt125m_stat8k_sandbox/summary.json")["rows"]}
    for r in j("a6m_opt125m_ba/summary.json")["rows"]:
        if r["run"].startswith("n128"):
            pts.append((r["ppl"], a6s[r["run"]], "A6, OPT-125M (rotation)"))
    groups = ["GATE A3, OPT-125M", "A5 step 2, step 1/n MAC", "A6, OPT-125M (rotation)", "A5 step 2, step > 1 MAC (F13)"]
    fig, ax = plt.subplots(figsize=(5, 4.2))
    lim = (20, 600)
    ax.plot(lim, lim, color=INK2, lw=1, ls="--", zorder=1, label="statistical = bit-accurate")
    off = []
    for c, gname in enumerate(groups):
        pp = [p for p in pts if p[2] == gname]
        off += [p for p in pp if p[1] > lim[1]]
        ax.scatter([p[0] for p in pp], [min(p[1], lim[1] * 0.93) for p in pp], s=40, color=C[[0, 2, 3, 7][c]],
                   marker=MK[c], edgecolors="#fcfcfb", linewidths=1, zorder=3, label=gname)
    if off:
        ax.annotate(f"{len(off)} GPT-2 points, statistical ≈ {statistics.median(p[1] for p in off):.1e} (off scale)",
                    (off[0][0], lim[1] * 0.93), xytext=(-150, -14), textcoords="offset points", fontsize=6,
                    color=INK2)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks([20, 50, 100, 200, 500])
    logfmt(ax)
    ax.set_yticks([20, 50, 100, 200, 500])
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("bit-accurate perplexity", fontsize=8, color=INK2)
    ax.set_ylabel("statistical perplexity (median of seeds)", fontsize=8, color=INK2)
    style(ax, "Statistical mode vs bit-accurate")
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.tight_layout()
    save(fig, "a3_validation")


def a4_zeroshot():
    from eval.zeroshot import load_task_result, paired_compare
    models = [("Qwen2.5-0.5B", "zs_qwen05b_w8a8", "zs_qwen05b_stat_adc6cal"),
              ("TinyLlama-1.1B", "zs_tinyllama_w8a8_nobos", "zs_tinyllama_stat_adc6cal_nobos"),
              ("Qwen2.5-1.5B", "zs_qwen15b_w8a8", "zs_qwen15b_stat_adc6cal")]
    tasks = [("arc_easy", "ARC-e"), ("piqa", "PIQA"), ("hellaswag", "HellaSwag")]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for c, (mname, ref, run) in enumerate(models):
        for t, (task, tname) in enumerate(tasks):
            a, b = load_task_result(R / ref, task), load_task_result(R / run, task)
            pc = paired_compare(a["docs"], b["docs"], "acc")
            x = t + (c - 1) * 0.22
            ax.errorbar(x, 100 * pc["diff"], yerr=[[100 * (pc["diff"] - pc["ci95"][0])], [100 * (pc["ci95"][1] - pc["diff"])]],
                        fmt=MK[c], color=C[c], ms=6, mec="#fcfcfb", mew=1, capsize=3, lw=1.5,
                        label=mname if t == 0 else None, zorder=3)
    ax.axhline(0, color=INK2, lw=1, ls="--")
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([t[1] for t in tasks])
    ax.set_ylabel("accuracy, analog CIM − W8A8 (pp)", fontsize=8, color=INK2)
    style(ax, "Zero-shot, first 1,000 documents per task, paired 95 % interval")
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper left")
    ax.set_ylim(-4, 4)
    fig.tight_layout()
    save(fig, "a4_zeroshot")


def headline():
    rows = j("b4_sweep/summary.json")["rows"]

    def get(model, point, adc, phase):
        for r in rows:
            if (r["model"], r["point"], r["adc"], r["phase"], r["dram"], r["budget"]) == (model, point, adc, phase, "default", "min"):
                return r
        raise KeyError((model, point, adc, phase))

    opts = [("W8A8 digital INT8", "-", "Digital INT8\n(conventional)"),
            ("W8A8 DCIM (M4)", "-", "Digital SRAM CIM"),
            ("analog nrow 128 / 6 b", "m1", "Analog SRAM CIM\n(calibrated ADC)"),
            ("analog nrow 128 / 6 b", "best_sar", "Analog SRAM CIM\n(best SAR ADC)")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.15, 1]})
    models = [("qwen2.5-0.5b", "Qwen2.5-0.5B"), ("qwen2.5-1.5b", "Qwen2.5-1.5B")]
    w = 0.38
    for i, (m, mname) in enumerate(models):
        base = get(m, opts[0][0], opts[0][1], "prefill")["compute_J"]
        vals = [get(m, p, adc, "prefill")["compute_J"] * 1e3 for p, adc, _ in opts]
        xs = [k + (i - 0.5) * w for k in range(len(opts))]
        bars = a.bar(xs, vals, w, color=C[i], label=mname, zorder=3)
        for x, v in zip(xs, vals):
            a.text(x, v, f"{base * 1e3 / v:.1f}×" if v < base * 1e3 * 0.999 else "1×", ha="center", va="bottom",
                   fontsize=6.5, color=INK2)
    a.set_xticks(range(len(opts)), [o[2] for o in opts], fontsize=7)
    a.set_ylabel("mJ per token", fontsize=8, color=INK2)
    a.legend(fontsize=7, frameon=False)
    style(a, "Compute energy per token (linear layers), same perplexity as W8A8")

    m = "qwen2.5-1.5b"
    parts = [("compute_J", "compute", C[6]), ("weights_J", "weights from DRAM + array write", C[3]),
             ("lm_head_J", "LM head", C[2]), ("other_J", "attention, KV, other", C[4])]
    labels = []
    for k, (p, adc, name) in enumerate(opts[::-1]):
        r = get(m, p, adc, "decode")
        left = 0.0
        for key, lab, col in parts:
            v = r[key] * 1e3
            b.barh(k, v, left=left, color=col, label=lab if k == 0 else None, zorder=3, height=0.6)
            left += v
        b.text(left + 0.5, k, f"{left:.1f} mJ", va="center", fontsize=7, color=INK2)
        labels.append(name.replace("\n", " "))
    b.set_yticks(range(len(opts)), labels, fontsize=7)
    b.set_xlabel("mJ per generated token", fontsize=8, color=INK2)
    b.set_xlim(0, 60)
    b.legend(fontsize=6.5, frameon=False, loc="lower center", bbox_to_anchor=(0.45, -0.42), ncol=2)
    style(b, "Qwen2.5-1.5B decode: where the energy goes")
    fig.tight_layout()
    save(fig, "headline")


def a6_pareto():
    import fig_a6_pareto
    fig_a6_pareto.main()


FIGS = {"a3_validation": a3_validation, "a4_zeroshot": a4_zeroshot, "a5_boundary": a5_boundary,
        "a5_alignment": a5_alignment, "a5_planes": a5_planes, "a5_noise": a5_noise, "a6_pareto": a6_pareto,
        "headline": headline}

if __name__ == "__main__":
    names = sys.argv[1:] or list(FIGS)
    if "a3_validation" in names and "a5_alignment" not in names:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "a5_alignment.py")], check=True, cwd=ROOT)
    for n in names:
        FIGS[n]()
