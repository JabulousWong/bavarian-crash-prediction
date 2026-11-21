"""fig_calibration for paper_short.tex, with the MSWS-PU model added.

Deciles are recomputed from the stored 2025 predictions exactly as in analysis/phase6_minute_pu.py
(hourly risk 1-(1-p)^12; decile edges from the uniform sample; observed rate from the recorded crash hours).
MSWS-PU targets crashes of any severity; its scores are put on the recorded-crash scale by pi_rec / pi_all
(as PN scores are by pi / pi_L), which changes the level and not the ranking.
    python3 make_fig_calibration.py <fonts dir with Charis SIL .ttf> <output dir>
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

FONTS, OUT = map(Path, sys.argv[1:3])
for f in FONTS.glob("*.ttf"):
    font_manager.fontManager.addfont(str(f))

V2 = Path(__file__).resolve().parents[3] / "analysis" / "results" / "v2"
MM = 1 / 25.4
SURFACE, INK, SECONDARY, MUTED, GRID, AXIS = "#ffffff", "#0b0b0b", "#52514e", "#6f6e69", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Charis SIL"], "mathtext.fontset": "stix", "font.size": 8,
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "pdf.fonttype": 42, "axes.unicode_minus": False,
})

test = pd.read_parquet(V2 / "phase6" / "minute_pu" / "test_predictions_2025.parquet")
units = json.load(open(V2 / "phase6" / "minute_dataset_summary.json"))["years"]["2025"]["population_edge_5min_units"]
kind = test["kind"].to_numpy()
marginal, official = kind == "marginal", kind == "official_case"
population_hours = units / 12
priors = json.load(open(V2 / "phase6" / "minute_pu" / "metrics.json"))["priors_per_unit"]
RATIO = priors["all_crashes_assumed"] / priors["injury_from_police_register"]  # pi_all / pi_rec = 8.654


def deciles(column, scale=1.0):
    hourly = 1 - (1 - test[column].to_numpy() * scale) ** 12
    edges = np.quantile(hourly[marginal], np.linspace(0, 1, 11))
    edges[0], edges[-1] = -np.inf, np.inf
    du = np.digitize(hourly[marginal], edges[1:-1])
    dp = np.digitize(hourly[official], edges[1:-1])
    predicted = np.array([hourly[marginal][du == d].mean() for d in range(10)])
    observed = np.array([(dp == d).sum() / (population_hours * (du == d).mean()) for d in range(10)])
    return predicted * 1e4, observed * 1e4


SERIES = [("PN", "pn_reliable__b30", BLUE, "o", 1.0),
          (r"nnPU, $\pi_{\mathrm{rec}}$", "nnpu_reliable__b30", ORANGE, "s", 1.0),
          (r"MSWS-PU + propensity, $\times\,\pi_{\mathrm{rec}}/\pi_{\mathrm{all}}$",
           "mswspu_credibility_propensity__b30", VIOLET, "D", 1 / RATIO)]

fig, ax = plt.subplots(figsize=(90 * MM, 100 * MM), layout="constrained", facecolor=SURFACE)
ax.set_facecolor(SURFACE)
ax.grid(color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(AXIS)
ax.tick_params(colors=MUTED, length=0)

xlim = ylim = (0.06, 12)
ax.plot(xlim, xlim, color=AXIS, linewidth=1.0, zorder=1)
for label, column, colour, marker, scale in SERIES:
    predicted, observed = deciles(column, scale)
    ax.plot(predicted, observed, color=colour, linewidth=1.5, marker=marker, markersize=5,
            markeredgecolor=SURFACE, markeredgewidth=0.9, label=label, zorder=3)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.set_xticks([0.1, 0.3, 1, 3, 10], ["0.1", "0.3", "1", "3", "10"])
ax.set_yticks([0.1, 0.3, 1, 3, 10], ["0.1", "0.3", "1", "3", "10"])
ax.minorticks_off()
ax.set_aspect("equal")
ax.set_xlabel("Predicted injury crashes per 10,000 edge-hours", color=SECONDARY)
ax.set_ylabel("Observed injury crashes per 10,000 edge-hours", color=SECONDARY)
ax.text(0.075, 0.068, "predicted = observed", color=MUTED, fontsize=6.8, rotation=45, rotation_mode="anchor",
        transform_rotates_text=True, va="bottom")
fig.legend(loc="outside upper left", ncol=1, frameon=False, labelcolor=SECONDARY, handlelength=2.2)
fig.savefig(OUT / "fig_calibration.pdf", facecolor=SURFACE)
print("wrote", OUT / "fig_calibration.pdf")
