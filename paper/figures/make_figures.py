"""Figures for the manuscript and the thesis, drawn from the saved analysis results.

Reads analysis/results/v2 only (nothing is refitted) and writes each figure as a
vector PDF and a 300-dpi PNG into paper/figures and thesis/image.

    python paper/figures/make_figures.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "analysis" / "results" / "v2"
OUT = Path(__file__).resolve().parent
THESIS = ROOT / "thesis" / "image"
MM = 1 / 25.4

# Print surface and inks; four categorical series, validated for colour-vision deficiency
# (adjacent and all-pairs) on a white surface. Markers repeat the identity for greyscale print.
SURFACE, INK, SECONDARY, MUTED, GRID, AXIS = "#ffffff", "#0b0b0b", "#52514e", "#6f6e69", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8,
    "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": False,
})


def style(ax, grid_axis: str = "y") -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    if grid_axis == "y":
        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, length=0)
    ax.xaxis.label.set_color(SECONDARY)
    ax.yaxis.label.set_color(SECONDARY)


def save(fig, name: str) -> None:
    for folder in (OUT, THESIS):
        folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", facecolor=SURFACE)
    fig.savefig(OUT / f"{name}.png", dpi=300, facecolor=SURFACE)
    plt.close(fig)
    for suffix in (".pdf", ".png"):
        shutil.copyfile(OUT / f"{name}{suffix}", THESIS / f"{name}{suffix}")
    print(f"wrote {name}")


def decomposition() -> None:
    """Ranking across the network against ranking within matched risk sets, by information source."""
    metrics = json.loads((V2 / "phase7" / "two_stage_metrics.json").read_text(encoding="utf-8"))["metrics"]
    rows = [
        ("edge_length_alone", "Edge length alone"),
        ("frequency_prior_alone", "Frequency prior alone (2016\u20132023 crashes)"),
        ("pn_union__context", "Context (time, length, mean flow)"),
        ("pn_union__context+frequency", "Context + frequency"),
        ("pn_union__context+traffic", "Context + minute traffic"),
        ("pn_union__context+frequency+traffic", "Context + frequency + minute traffic"),
        ("pn_official_hour__context+frequency+traffic", "Same, police hour-timed labels"),
    ]
    panels = [
        ("Across the network", "ROC-AUC, police crashes against marginal units",
         lambda m: (m["population"]["official_all"]["roc_auc"], m["population"]["official_all"]["roc_auc_ci95"])),
        ("Within matched risk sets", "Concordance, crash against its four controls",
         lambda m: (m["matched"]["official"]["within_set_concordance"], m["matched"]["official"]["ci95"])),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(190 * MM, 70 * MM), sharey=True, layout="constrained", facecolor=SURFACE)
    y = np.arange(len(rows))[::-1]
    for ax, (title, label, pick) in zip(axes, panels):
        style(ax, grid_axis="x")
        ax.axvline(0.5, color=AXIS, linewidth=1.0, zorder=1)
        values = [pick(metrics[key]) for key, _ in rows]
        centre = np.array([value for value, _ in values])
        low = centre - np.array([ci[0] for _, ci in values])
        high = np.array([ci[1] for _, ci in values]) - centre
        ax.errorbar(centre, y, xerr=[low, high], fmt="o", color=BLUE, ecolor=BLUE, elinewidth=1.2, capsize=0,
                    markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlabel(label)
        ax.set_xlim(0.47, 0.81)
        ax.set_xticks([0.5, 0.6, 0.7, 0.8])
    axes[0].set_yticks(y, [label for _, label in rows])
    axes[0].tick_params(axis="y", colors=SECONDARY)
    axes[0].set_ylim(-0.6, len(rows) - 0.1)
    axes[0].text(0.505, len(rows) - 0.45, "no information", color=MUTED, fontsize=7, ha="left", va="center")
    save(fig, "fig_decomposition")


def calibration() -> None:
    """Predicted against observed hourly injury-crash rate by decile of predicted risk."""
    phase6 = json.loads((V2 / "phase6" / "minute_pu" / "metrics.json").read_text(encoding="utf-8"))["hourly_calibration"]
    phase7 = json.loads((V2 / "phase7" / "two_stage_metrics.json").read_text(encoding="utf-8"))["hourly_calibration"]
    series = [
        ("PN, Waze-timed police-confirmed labels", phase6["arms"]["pn_reliable"]["deciles"], BLUE, "o"),
        ("nnPU, register prior", phase6["arms"]["nnpu_reliable"]["deciles"], ORANGE, "s"),
        ("nnPU + reporting propensity", phase6["arms"]["nnpu_reliable_propensity"]["deciles"], AQUA, "^"),
        ("Frequency prior alone", phase7["scores"]["frequency_prior_alone"]["deciles"], VIOLET, "D"),
    ]
    fig, ax = plt.subplots(figsize=(90 * MM, 96 * MM), layout="constrained", facecolor=SURFACE)
    style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color(AXIS)
    limits = (0.06, 12)
    ax.plot(limits, limits, color=AXIS, linewidth=1.0, zorder=1)
    for label, deciles, colour, marker in series:
        predicted = np.array([d["predicted"] for d in deciles]) * 1e4
        observed = np.array([d["observed"] for d in deciles]) * 1e4
        ax.plot(predicted, observed, color=colour, linewidth=1.5, marker=marker, markersize=5.5,
                markeredgecolor=SURFACE, markeredgewidth=1.0, label=label, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    ticks = [0.1, 0.3, 1, 3, 10]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.set_yticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_aspect("equal")
    ax.set_xlabel("Predicted injury crashes per 10,000 edge-hours")
    ax.set_ylabel("Observed injury crashes per 10,000 edge-hours")
    ax.text(0.14, 0.105, "predicted = observed", color=MUTED, fontsize=7, rotation=45, rotation_mode="anchor", va="top")
    fig.legend(loc="outside upper left", ncol=1, frameon=False, labelcolor=SECONDARY, handlelength=2.2)
    save(fig, "fig_calibration")


def label_hiding() -> None:
    """Label-hiding experiment: what each estimator recovers as police labels are removed."""
    runs = pd.read_csv(V2 / "phase8" / "label_hiding_runs.csv")
    measures = ["predicted_over_observed", "predicted_over_observed_low_third", "auc_hidden_type"]
    summary = runs.groupby(["mechanism", "labelled_share", "method"], sort=False)[measures].agg(["mean", "std"]).reset_index()
    styles = {"pn": (BLUE, "o", "PN (U as negative)"), "pn_rescaled": (ORANGE, "s", "PN rescaled by the known prior"),
              "nnpu": (AQUA, "^", "nnPU, register prior"), "nnpu_propensity": (VIOLET, "D", "nnPU + estimated propensity")}
    panels = [("predicted_over_observed", "Predicted / observed crashes,\nwhole network"),
              ("predicted_over_observed_low_third", "Predicted / observed crashes,\nthird of units with lightest traffic"),
              ("auc_hidden_type", "ROC-AUC on crashes\nwhose label would be hidden")]
    columns = [("scar", "Labels removed at random"), ("sar_flow", "Labels removed more often in light traffic")]
    fig, axes = plt.subplots(3, 2, figsize=(190 * MM, 175 * MM), sharex=True, sharey="row", layout="constrained", facecolor=SURFACE)
    for column, (mechanism, title) in enumerate(columns):
        for row, (metric, label) in enumerate(panels):
            ax = axes[row, column]
            style(ax)
            if metric != "auc_hidden_type":
                ax.axhline(1.0, color=AXIS, linewidth=1.0, zorder=1)
            for offset, (method, (colour, marker, name)) in zip((-1.2, -0.4, 0.4, 1.2), styles.items()):
                # With every label kept the mechanisms coincide, and propensity weights are all one.
                start = summary.loc[(summary["mechanism"] == "scar") & (summary["labelled_share"] == 1.0)
                                    & (summary["method"] == ("nnpu" if method == "nnpu_propensity" else method))]
                part = summary.loc[(summary["mechanism"] == mechanism) & (summary["method"] == method) & (summary["labelled_share"] < 1.0)]
                part = pd.concat([start, part]).sort_values("labelled_share", ascending=False).dropna(subset=[(metric, "mean")])
                if part.empty:
                    continue
                x = 100 * (1 - part["labelled_share"]) + offset  # small dodge: PN rescaled and nnPU nearly coincide
                ax.errorbar(x, part[(metric, "mean")], yerr=part[(metric, "std")], color=colour, linewidth=1.5, elinewidth=0.8,
                            marker=marker, markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=3,
                            label=name if (row, column) == (0, 1) else None)
            if column == 0:
                ax.set_ylabel(label)
            if row == 0:
                ax.set_title(title, color=INK, loc="left")
            if row == 2:
                ax.set_xlabel("Police-recorded crashes without a label in training (%)")
    fig.legend(loc="outside upper left", ncol=4, frameon=False, labelcolor=SECONDARY, handlelength=2.4)
    save(fig, "fig_label_hiding")


def temporal_profile() -> None:
    """Network-wide time profile of police injury crashes 2016-2023 relative to a uniform spread."""
    hour = pd.read_csv(V2 / "phase7" / "temporal_profile_hour.csv")
    month = pd.read_csv(V2 / "phase7" / "temporal_profile_month.csv")
    fig, axes = plt.subplots(1, 2, figsize=(190 * MM, 66 * MM), layout="constrained", facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    style(ax)
    ax.axhline(1.0, color=AXIS, linewidth=1.0, zorder=1)
    for weekend, colour, marker, name in ((False, BLUE, "o", "Monday to Friday"), (True, ORANGE, "s", "Saturday and Sunday")):
        part = hour.loc[hour["weekend"] == weekend].sort_values("hour")
        ax.plot(part["hour"], part["multiplier"], color=colour, linewidth=1.5, marker=marker, markersize=4.5,
                markeredgecolor=SURFACE, markeredgewidth=0.8, label=name, zorder=3)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel("Hour of day (local time)")
    ax.set_ylabel("Crash rate relative to uniform")
    ax.set_ylim(0, 2.6)
    ax.legend(loc="upper left", frameon=False, labelcolor=SECONDARY, handlelength=2.2)
    ax = axes[1]
    style(ax)
    ax.axhline(1.0, color=AXIS, linewidth=1.0, zorder=1)
    ax.plot(month["month"], month["multiplier"], color=BLUE, linewidth=1.5, marker="o", markersize=4.5,
            markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=3)
    ax.set_xticks(range(1, 13), list("JFMAMJJASOND"))
    ax.set_xlabel("Month")
    ax.set_ylim(0, 2.6)
    ax.set_yticklabels([])
    save(fig, "fig_temporal_profile")


if __name__ == "__main__":
    decomposition()
    calibration()
    label_hiding()
    temporal_profile()
