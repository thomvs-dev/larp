"""Figure A6: perplexity (relative to W8A8 fake-quant) against analog ADC conversions per token, per
mitigation, one panel per model. Reads results/<run>/summary.json written by scripts/a6_mitigation.py.

    python scripts/fig_a6_pareto.py            # -> figures/a6_pareto.png, figures/a6_pareto.md (table)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PANELS = [("a6m_gpt2_ba", "GPT-2, bit-accurate"), ("a6m_opt125m_ba", "OPT-125M, bit-accurate"),
          ("a6m_qwen15b_stat", "Qwen2.5-1.5B, statistical")]
# fixed variant order -> categorical slots 1-5 (validated palette, dataviz reference instance) + marker shape
VARIANTS = [("base", "no mitigation", "#2a78d6", "o"), ("had_down", "Hadamard, down input", "#eb6834", "s"),
            ("had_all", "Hadamard, all inputs", "#1baf7a", "D"), ("dcim_down", "down on DCIM", "#eda100", "^"),
            ("down_bp1", "down +1 ADC bit", "#e87ba4", "v")]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def rows(run: str):
    d = json.loads((ROOT / "results" / run / "summary.json").read_text(encoding="utf-8"))
    tok = d["tokens"]
    by = {r["run"]: r for r in d["rows"]}
    return by, tok


def main() -> None:
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, len(PANELS), figsize=(11, 3.6))
    table = ["| Model | Point | Variant | PPL | / W8A8 | analog conversions / token | DCIM ops / token | Hadamard adds / token |",
             "|---|---|---|---|---|---|---|---|"]
    used = set()
    for ax, (run, title) in zip(axes, PANELS):
        by, tok = rows(run)
        fq = by["fake_quant"]["ppl"]
        fqh = by.get("fake_quant_had")
        points = sorted({k.split("_")[0] + "_" + k.split("_")[1] for k in by if k.startswith("n")})
        for p in points:
            for key, label, col, mk in VARIANTS:
                r = by.get(f"{p}_{key}")
                if r is None:
                    continue
                x, y = r["adc_conversions_analog"] / tok, r["ppl"] / fq
                ax.scatter(x, y, s=64, color=col, marker=mk, edgecolors="#fcfcfb", linewidths=1.5, zorder=3,
                           label=label if key not in used else None)
                used.add(key)
                table.append(f"| {title.split(',')[0]} | {p.replace('_', ' / ')} | {label} | {r['ppl']:.3f} | {y:.3f} "
                             f"| {x:.3g} | {r['dcim_column_ops'] / tok:.3g} | {r['hadamard_adds'] / tok:.3g} |")
            base = by.get(f"{p}_base")
            if base is not None:
                ax.annotate(p.replace("n", "nrow ").replace("_adc", " / ") + " b",
                            (base["adc_conversions_analog"] / tok, base["ppl"] / fq), xytext=(-7, 3),
                            textcoords="offset points", fontsize=7, color=INK2, ha="right")
        ax.axhline(1.0, color=INK2, lw=1, ls="--", zorder=1)
        if fqh is not None:
            ax.axhline(fqh["ppl"] / fq, color=INK2, lw=1, ls=":", zorder=1)
        if max(r["ppl"] / fq for k, r in by.items()) > 2:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_yticks([1, 2, 4, 6])
            # zoom on the mitigated points (all within 1 % of W8A8); markers offset in x within the inset
            ins = ax.inset_axes([0.42, 0.22, 0.52, 0.46])
            xr = [r["adc_conversions_analog"] / tok for k, r in by.items() if k.startswith("n")]
            dx = (max(xr) - min(xr)) * 0.035
            for p in points:
                for i, (key, label, col, mk) in enumerate(VARIANTS):
                    r = by.get(f"{p}_{key}")
                    if r is None or r["ppl"] / fq > 1.05:
                        continue
                    ins.scatter(r["adc_conversions_analog"] / tok + (i - 2) * dx, r["ppl"] / fq, s=36, color=col,
                                marker=mk, edgecolors="#fcfcfb", linewidths=1.2, zorder=3)
            ins.axhline(1.0, color=INK2, lw=1, ls="--")
            if fqh is not None:
                ins.axhline(fqh["ppl"] / fq, color=INK2, lw=1, ls=":")
            ins.set_ylim(0.99, 1.01)
            ins.tick_params(labelsize=6, colors=INK2)
            ins.xaxis.get_offset_text().set_fontsize(6)
            ins.grid(True, color=GRID, lw=0.5)
            ins.set_title("zoom 0.99-1.01 (x offsets per variant)", fontsize=6, color=INK2)
            for sp in ins.spines.values():
                sp.set_color(INK2)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("analog ADC conversions per token", fontsize=8, color=INK2)
        ax.grid(True, color=GRID, lw=0.6, zorder=0)
        ax.tick_params(labelsize=7, colors=INK2)
        for s in ax.spines.values():
            s.set_color(GRID)
    axes[0].set_ylabel("perplexity / W8A8 fake-quant", fontsize=8, color=INK2)
    h, lab = [], []
    for a in axes:
        hh, ll = a.get_legend_handles_labels()
        for x, y in zip(hh, ll):
            if y not in lab:
                h.append(x)
                lab.append(y)
    h += [plt.Line2D([], [], color=INK2, ls="--", lw=1), plt.Line2D([], [], color=INK2, ls=":", lw=1)]
    lab += ["W8A8 fake-quant", "W8A8 fake-quant, rotated"]
    fig.legend(h, lab, loc="lower center", ncol=7, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "a6_pareto.png", dpi=200)
    (out / "a6_pareto.md").write_text("# Figure A6 data\n\n" + "\n".join(table) + "\n", encoding="utf-8")
    print(f"figure: {out / 'a6_pareto.png'}")


if __name__ == "__main__":
    main()
