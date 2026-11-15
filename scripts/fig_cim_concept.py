"""Concept figure for the README: how one analog SRAM CIM column computes a dot product.

    python scripts/fig_cim_concept.py figures/cim_concept.png
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREEN, PURPLE, AMBER = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#eda100"

fig, ax = plt.subplots(figsize=(11, 4.6))
ax.set_xlim(0, 22); ax.set_ylim(0, 9.2); ax.axis("off")

def box(x, y, w, h, fc, ec, text=None, fs=8, color=INK, r=0.15, lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color)

def arrow(x0, y0, x1, y1, color=INK2, lw=1.2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10, color=color, lw=lw))

# inputs (bit-serial on wordlines)
ax.text(1.4, 8.6, "inputs x (8-bit)\none bit per cycle", ha="center", va="center", fontsize=8, color=INK)
rows = 6
y0, dy = 1.6, 1.0
for i in range(rows):
    y = y0 + i * dy
    bit = [1, 0, 1, 1, 0, 1][i]
    box(0.7, y + 0.15, 1.4, 0.6, BLUE if bit else "white", BLUE, str(bit), fs=8, color="white" if bit else BLUE)
    arrow(2.2, y + 0.45, 3.1, y + 0.45, BLUE)
ax.text(1.4, 1.05, "⋮  nrow rows", ha="center", va="center", fontsize=8, color=INK2)

# array
box(3.1, 1.45, 4.6, 6.3, "#f6f5f2", GRID, r=0.2)
ax.text(5.4, 8.2, "SRAM array: weights stored\nas bits in the cells", ha="center", va="center", fontsize=8, color=INK)
for i in range(rows):
    for j in range(3):
        y = y0 + i * dy
        w = [[1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1], [1, 0, 0], [1, 1, 1]][i][j]
        box(3.4 + j * 1.45, y + 0.15, 1.05, 0.6, ORANGE if w else "white", ORANGE, str(w), fs=8,
            color="white" if w else ORANGE)
# bit lines
for j in range(3):
    xb = 3.92 + j * 1.45
    ax.plot([xb, xb], [1.45, 1.1], color=PURPLE, lw=1.5)
ax.text(5.4, 0.55, "each bit line adds up its cells: an analog sum\nof up to nrow products (one 'MAC value')",
        ha="center", va="center", fontsize=7.5, color=PURPLE)

# ADC
arrow(7.8, 4.5, 9.2, 4.5, PURPLE, lw=1.6)
ax.text(8.5, 4.85, "sums", ha="center", va="bottom", fontsize=7.5, color=PURPLE)
box(9.2, 3.7, 2.6, 1.6, "white", GREEN, "ADC\nb bits", fs=9, color=GREEN, lw=1.6)
ax.text(10.5, 6.2, "turns the analog sum into\na number with 2^b levels.\nToo few levels → rounding\nand clipping errors",
        ha="center", va="center", fontsize=7.5, color=INK2)
arrow(10.5, 2.7, 10.5, 3.65, AMBER)
ax.text(10.5, 2.35, "analog noise", ha="center", va="center", fontsize=7.5, color=AMBER)
ax.text(10.5, 1.5, "one conversion per bit line\nper input bit (shown for one)", ha="center", va="center", fontsize=7, color=INK2)

# shift-add
arrow(11.9, 4.5, 13.3, 4.5)
box(13.3, 3.7, 2.8, 1.6, "white", INK2, "digital\nshift-add", fs=9, color=INK)
ax.text(14.7, 6.2, "combines all input bits,\nweight bits and row groups\ninto the 8-bit × 8-bit result",
        ha="center", va="center", fontsize=7.5, color=INK2)
arrow(16.2, 4.5, 17.6, 4.5)
box(17.6, 3.7, 3.9, 1.6, "white", INK, "layer output\n(one matrix–vector product)", fs=8.5, color=INK)

ax.text(19.55, 7.6, "why it saves energy", ha="center", va="center", fontsize=8.5, color=INK, weight="bold")
ax.text(19.55, 6.6, "the multiply-adds happen where\nthe weights are stored, so\nweights are not moved to a\nseparate arithmetic unit",
        ha="center", va="center", fontsize=7.5, color=INK2)
ax.text(19.55, 2.4, "what can go wrong", ha="center", va="center", fontsize=8.5, color=INK, weight="bold")
ax.text(19.55, 1.4, "ADC too coarse for nrow,\nanalog noise, inputs that fill\nthe high bit planes",
        ha="center", va="center", fontsize=7.5, color=INK2)

fig.tight_layout()
fig.savefig(sys.argv[1], dpi=200)
print(sys.argv[1])
