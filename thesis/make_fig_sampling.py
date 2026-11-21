"""Figures of the conventional-resampling comparison (analysis/phase9_sampling.py).

fig_sampling_effects   what resampling changes and what it does not: ranking, recall and
                       alerts at the default 0.5 cut-off, probability level before/after correction
fig_sampling_reported  precision and F1 on a 1:1 balanced test set (as often reported) against
                       the same models at the natural prevalence of 2025
fig_sampling_calibration  decile reliability of the gradient-boosting model, raw and corrected

Run with the analysis environment: python make_fig_sampling.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "analysis" / "results" / "v2" / "phase9" / "sampling" / "metrics.json"
OUT = HERE / "image"

BLUE, ORANGE, SLATE = "#2a78d6", "#eb6834", "#3d3c38"   # colour-blind-safe slots 1-3
GREY, INK, MUTED = "#898781", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#444",
    "xtick.color": "#666", "ytick.color": "#666", "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42, "savefig.facecolor": "white",
})

METHODS = [("none", "None\n(1:260)"), ("rus_1to10", "RUS\n1:10"), ("rus_1to4", "RUS\n1:4"), ("rus_1to1", "RUS\n1:1"),
           ("ros_1to1", "ROS\n1:1"), ("smote_1to1", "SMOTE\n1:1"), ("adasyn_1to1", "ADASYN\n1:1"),
           ("rus_smote", "RUS+\nSMOTE")]
LEARNERS = [("lr", "Logistic regression", BLUE, "o"), ("hgb", "Gradient boosting", ORANGE, "s"), ("mlp", "MLP", SLATE, "D")]
OFFSET = {"lr": -0.18, "hgb": 0.0, "mlp": 0.18}

results = json.load(open(RESULTS))
M = results["metrics"]
get = lambda learner, method: M[f"{learner}__{method}__b30"]
x = np.arange(len(METHODS))


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def method_axis(ax, labels=True):
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in METHODS] if labels else [], fontsize=8.5)
    ax.set_xlim(-0.6, len(METHODS) - 0.4)
    ax.grid(axis="x", visible=False)
    for edge in (0.5, 3.5, 6.5):  # groups: none | undersampling | oversampling | combined
        ax.axvline(edge, color="#e1e0d9", lw=0.8, zorder=0)


# ------------------------------------------------------------------ effects (2 x 2)
fig, axes = plt.subplots(2, 2, figsize=(10, 6.6), sharex=True)
(a, b), (c, d) = axes
for learner, name, colour, marker in LEARNERS:
    xs = x + OFFSET[learner]
    auc = np.array([get(learner, m)["population"]["roc_auc"] for m, _ in METHODS])
    lo, hi = np.array([get(learner, m)["population"]["roc_auc_ci95"] for m, _ in METHODS]).T
    a.errorbar(xs, auc, yerr=[auc - lo, hi - auc], fmt=marker, color=colour, ms=6, mec="white", mew=0.8,
               elinewidth=1.6, capsize=0, label=name)
    recall = [get(learner, m)["threshold_0.5"]["recall"] for m, _ in METHODS]
    b.plot(xs, recall, marker, color=colour, ms=6, mec="white", mew=0.8, label=name)
    alerts = [max(get(learner, m)["threshold_0.5"]["alerts_per_day"], 0.5) for m, _ in METHODS]
    c.plot(xs, alerts, marker, color=colour, ms=6, mec="white", mew=0.8, label=name)
    raw = [get(learner, m)["level"]["predicted_over_observed_raw"] for m, _ in METHODS]
    corrected = [get(learner, m)["level"]["predicted_over_observed_corrected"] for m, _ in METHODS]
    d.plot(xs, raw, marker, color=colour, ms=6, mec="white", mew=0.8, alpha=0.35)
    d.plot(xs, corrected, marker, color=colour, ms=6, mec="white", mew=0.8)

a.set_ylim(0.70, 0.84); a.set_ylabel("ROC-AUC, 2025 police crashes\nagainst all edge-5-min units")
a.set_title("(a) Ranking across the network", loc="left", fontsize=10.5, color="#222")
b.set_ylim(-0.03, 1.0); b.set_ylabel("Share of 2025 crashes alerted")
b.set_title("(b) Recall at the default cut-off 0.5", loc="left", fontsize=10.5, color="#222")
c.set_yscale("log"); c.set_ylabel("Alerts per day, whole network\n(edge-5-min units, log scale)")
c.set_title("(c) Alarms at the default cut-off 0.5", loc="left", fontsize=10.5, color="#222")
d.set_yscale("log"); d.axhline(1, color="#b9b8b0", lw=1.2, zorder=0)
d.set_ylabel("Mean predicted risk / observed rate\n(log scale)")
d.set_title("(d) Probability level: raw (faint) and corrected", loc="left", fontsize=10.5, color="#222")
for ax in axes.flat:
    method_axis(ax)
a.legend(loc="lower left", ncol=3, fontsize=8.5, handletextpad=0.3, columnspacing=1.0)
fig.tight_layout(h_pad=1.6, w_pad=2.2)
save(fig, "fig_sampling_effects")

# ------------------------------------------------------------------ reported vs natural (dumbbell)
resampled = [(m, label.replace("\n", " ")) for m, label in METHODS if m != "none"]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
y = np.arange(len(resampled))[::-1]
for ax, title, balanced_key, natural_key, xlabel in (
        (axes[0], "Precision at cut-off 0.5", "precision", "precision_natural", "Precision (log scale)"),
        (axes[1], "F1 at cut-off 0.5", "f1", "f1_natural", "F1 (log scale)")):
    for learner, name, colour, marker in LEARNERS:
        ys = y + {"lr": 0.22, "hgb": 0.0, "mlp": -0.22}[learner]
        bal = np.array([get(learner, m)["balanced_test_0.5"][balanced_key] for m, _ in resampled])
        nat = np.array([get(learner, m)["threshold_0.5"][natural_key] for m, _ in resampled])
        for yi, lo, hi in zip(ys, nat, bal):
            ax.plot([lo, hi], [yi, yi], color=colour, lw=1.4, alpha=0.45, solid_capstyle="round")
        ax.plot(bal, ys, marker, color=colour, ms=6, mec="white", mew=0.8, label=name)
        ax.plot(nat, ys, marker, color="white", ms=6, mec=colour, mew=1.4)
    ax.set_xscale("log"); ax.set_xlim(1e-5, 1.5); ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontsize=10.5, color="#222"); ax.grid(axis="y", visible=False)
axes[0].set_yticks(y); axes[0].set_yticklabels([label for _, label in resampled])
axes[0].text(0.62, y[0] + 0.55, "1:1 balanced\ntest set", ha="center", va="bottom", fontsize=8.5, color=MUTED)
axes[0].text(4e-5, y[0] + 0.55, "natural prevalence\n2025", ha="center", va="bottom", fontsize=8.5, color=MUTED)
axes[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8.5)
axes[0].set_ylim(-0.6, len(resampled) + 0.4)
fig.tight_layout(w_pad=1.5)
save(fig, "fig_sampling_reported")

# ------------------------------------------------------------------ calibration of one learner
show = [("none", "None"), ("rus_1to4", "RUS 1:4"), ("ros_1to1", "ROS 1:1"), ("smote_1to1", "SMOTE 1:1")]
fig, axes = plt.subplots(1, len(show), figsize=(10, 2.9), sharex=True, sharey=True)
for ax, (method, label) in zip(axes, show):
    level = get("hgb", method)["level"]
    for kind, colour, filled in (("deciles_raw", GREY, False), ("deciles_corrected", ORANGE, True)):
        rows = level[kind]
        pred = [r["predicted"] for r in rows]
        obs = [max(r["observed"], 1e-7) for r in rows]
        ax.plot(pred, obs, "-", color=colour, lw=1.6, alpha=0.8)
        ax.plot(pred, obs, "o", ms=5, color=colour if filled else "white", mec=colour, mew=1.2,
                label="corrected" if filled else "raw")
    ax.plot([1e-7, 1], [1e-7, 1], color="#b9b8b0", lw=1.0, zorder=0)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1e-7, 1); ax.set_ylim(1e-7, 1e-3)
    ax.set_title(label, loc="left", fontsize=10.5, color="#222"); ax.set_xlabel("Predicted risk per unit")
axes[0].set_ylabel("Observed rate per unit")
axes[0].legend(loc="upper left", fontsize=8.5)
fig.tight_layout(w_pad=1.0)
save(fig, "fig_sampling_calibration")
