"""fig_learners for paper_short.tex: six learners under three label treatments, one row of three panels.

Point estimates only (intervals are in Supplementary Material S5), so the panels stay readable.
    python3 make_fig_learners.py <fonts dir with Charis SIL .ttf> <output dir>
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D

FONTS, OUT = map(Path, sys.argv[1:3])
for f in FONTS.glob("*.ttf"):
    font_manager.fontManager.addfont(str(f))

P9 = Path(__file__).resolve().parents[3] / "analysis" / "results" / "v2" / "phase9"
MM = 1 / 25.4
SURFACE, INK, SECONDARY, MUTED, GRID, AXIS = "#ffffff", "#0b0b0b", "#52514e", "#6f6e69", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Charis SIL"], "mathtext.fontset": "stix", "font.size": 8,
    "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "pdf.fonttype": 42, "axes.unicode_minus": False,
})

seq = json.load(open(P9 / "sequence_models" / "metrics.json"))
graph = json.load(open(P9 / "graph_models" / "metrics.json"))
metrics = {**seq["metrics"], **graph["metrics"]}
fits = {**seq["fits"], **graph["fits"]}

# Names as in Table 2 and Table 3 of the paper.
ARMS = [("L1_pn_reliable", "PN\nreliable"),
        ("L4_nnpu_reliable", "nnPU\nreliable"),
        ("L2_pn_union", "PN\nunion"),
        ("L5_mswspu", "MSWS-\nPU"),
        ("L6_mswspu_propensity", "MSWS-PU\n+ prop.")]
LEARNERS = [("mlp", "MLP", "#2a78d6", "o"), ("gru", "GRU", "#eb6834", "s"), ("lstm", "LSTM", "#1baf7a", "^"),
            ("tcn", "TCN", "#eda100", "D"), ("transformer", "Transformer", "#e87ba4", "v"),
            ("gat_k2", "ego-GAT", "#4a3aa7", "P")]
PANELS = [("pop", "(a) ROC-AUC, recorded crashes 2025"),
          ("within", "(b) Within-set concordance"),
          ("level", "(c) Predicted / observed crash hours")]


def value(learner, arm, measure):
    m = metrics[f"{learner}__{arm}__b30"]
    if measure == "pop":
        return m["population"]["official_all"]["roc_auc"]
    if measure == "within":
        return m["matched"]["official_all"]["within_set_concordance"]
    return m["predicted_over_observed_hourly"]


pi_rec = fits["mlp__L4_nnpu_reliable__b30"]["prior"]
implied = [fits[f"mlp__{arm}__b30"]["prior"] / pi_rec for arm, _ in ARMS]

x = np.arange(len(ARMS))
offsets = np.linspace(-0.24, 0.24, len(LEARNERS))
fig, axes = plt.subplots(1, 3, figsize=(190 * MM, 68 * MM), layout="constrained", facecolor=SURFACE)
for ax, (measure, title) in zip(axes, PANELS):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, length=0)
    for offset, (learner, name, colour, marker) in zip(offsets, LEARNERS):
        y = [value(learner, arm, measure) for arm, _ in ARMS]
        lw, ms, z = (1.6, 6, 4) if learner == "mlp" else (1.0, 5, 3)
        ax.plot(x + offset, y, color=colour, linewidth=lw, marker=marker, markersize=ms,
                markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=z, label=name)
    if measure == "level":
        ax.set_yscale("log")
        for xi, level in zip(x, implied):
            ax.plot([xi - 0.32, xi + 0.32], [level, level], color=MUTED, linewidth=1.1, linestyle=(0, (2.5, 1.5)), zorder=2)
        ticks = [0.25, 0.5, 1, 2, 4, 8]
        ax.set_yticks(ticks, [str(t) for t in ticks])
        ax.minorticks_off()
        ax.set_ylim(0.25, 12)
    ax.set_title(title, loc="left", color=INK)
    ax.set_xticks(x, [label for _, label in ARMS], color=SECONDARY, fontsize=7)
    ax.set_xlim(-0.5, len(ARMS) - 0.5)

handles, labels = axes[0].get_legend_handles_labels()
handles.append(Line2D([0], [0], color=MUTED, linewidth=1.1, linestyle=(0, (2.5, 1.5))))
labels.append("level implied by the class prior")
fig.legend(handles, labels, loc="outside upper center", ncol=7, frameon=False, labelcolor=SECONDARY,
           handlelength=2.0, columnspacing=1.4)
fig.savefig(OUT / "fig_learners.pdf", facecolor=SURFACE)
print("wrote", OUT / "fig_learners.pdf")
