"""Figures of the results chapter, drawn from the stored result files of the main analysis (results/v2).

fig_temporal_profile  phase7/temporal_profile_{hour,month}.csv
fig_decomposition     phase7/two_stage_metrics.json
fig_calibration       phase6/minute_pu/metrics.json (hourly_calibration)
fig_label_hiding      phase8/metrics.json (hiding_summary)
fig_learners, fig_learner_ablation, fig_learner_label_effects, fig_learner_grid
                      phase9/{sequence_models,sequence_models_long,graph_models}/metrics.json (E8)
The script also prints the rows of the E8 tables (learner results, variance decomposition, training schedule).

Run with the analysis environment: python make_figures.py [--results <results/v2 directory>]
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--results", default=str(HERE.parent / "analysis" / "results" / "v2"))
parser.add_argument("--out", default=str(HERE / "image"))
args = parser.parse_args()
V2, OUT = Path(args.results), Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

# Categorical slots in fixed order (blue, orange, aqua, yellow, magenta, green, violet, red); ink and chrome.
# Colour-blind-safe categorical palette: minimum pairwise CIEDE2000 >= 17 under normal vision and simulated
# deuteranopia, protanopia and tritanopia (Machado et al. 2009). Series are also told apart by marker shape.
BLUE, ORANGE, SLATE, SKY, PLUM, MAUVE = "#2a78d6", "#eb6834", "#3d3c38", "#7cc6f0", "#621172", "#995476"
RED = "#e34948"   # only as the warm end of the blue-red diverging heatmap
GREY, INK, MUTED, RULE = "#898781", "#0b0b0b", "#52514e", "#c3c2b7"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#444",
    "xtick.color": "#666", "ytick.color": "#666", "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42, "savefig.facecolor": "white",
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ temporal profile
hour = pd.read_csv(V2 / "phase7" / "temporal_profile_hour.csv")
month = pd.read_csv(V2 / "phase7" / "temporal_profile_month.csv")
fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw={"width_ratios": [2.6, 1.0]})
for weekend, colour, marker, label in ((False, BLUE, "o", "Monday to Friday"), (True, ORANGE, "s", "Saturday and Sunday")):
    part = hour[hour.weekend == weekend].sort_values("hour")
    a.plot(part.hour, part.multiplier, marker=marker, ms=4.5, lw=2, color=colour, label=label)
a.axhline(1, color="#b9b8b0", lw=1.2, zorder=0)
a.set_xticks(range(0, 24, 3)); a.set_xlabel("Hour of day (local time)"); a.set_ylabel("Crash rate relative to uniform")
a.set_ylim(0, max(hour.multiplier.max(), 2.5) * 1.05); a.legend(loc="upper left")
b.plot(month.month, month.multiplier, marker="o", ms=4.5, lw=2, color=BLUE)
b.axhline(1, color="#b9b8b0", lw=1.2, zorder=0)
b.set_xticks(range(1, 13)); b.set_xticklabels(list("JFMAMJJASOND")); b.set_xlabel("Month"); b.set_ylim(a.get_ylim())
b.tick_params(labelleft=False)
fig.tight_layout(w_pad=2)
save(fig, "fig_temporal_profile")

# ------------------------------------------------------------------ decomposition
two = json.load(open(V2 / "phase7" / "two_stage_metrics.json"))["metrics"]
rows = [("Edge length alone", "edge_length_alone"),
        ("Frequency prior alone (2016–2023 crashes)", "frequency_prior_alone"),
        ("Context (time, length, mean flow)", "pn_union__context"),
        ("Context + frequency", "pn_union__context+frequency"),
        ("Context + minute traffic", "pn_union__context+traffic"),
        ("Context + frequency + minute traffic", "pn_union__context+frequency+traffic"),
        ("Same, police hour-timed labels", "pn_official_hour__context+frequency+traffic")]
fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.7), sharey=True)
y = np.arange(len(rows))[::-1]
for ax, title, getter, xlabel, xlim in (
        (a, "Across the network", lambda m: (m["population"]["official_all"]["roc_auc"], m["population"]["official_all"]["roc_auc_ci95"]),
         "ROC-AUC, police crashes against marginal units", (0.5, 0.85)),
        (b, "Within matched risk sets", lambda m: (m["matched"]["official"]["within_set_concordance"], m["matched"]["official"]["ci95"]),
         "Concordance, crash against its four controls", (0.45, 0.70))):
    for yi, (label, key) in zip(y, rows):
        value, (lo, hi) = getter(two[key])
        ax.plot([lo, hi], [yi, yi], color=BLUE, lw=2.2, solid_capstyle="round")
        ax.plot(value, yi, "o", color=BLUE, ms=7, mec="white", mew=0.8, zorder=3)
    ax.axvline(0.5, color="#b9b8b0", lw=1.4, zorder=0)
    ax.set_xlim(*xlim); ax.set_title(title, loc="center", fontsize=11, color="#222"); ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
a.text(0.505, y[0] + 0.55, "no information", color=GREY, fontsize=9, va="bottom")
a.set_yticks(y); a.set_yticklabels([r[0] for r in rows]); a.set_ylim(-0.7, len(rows) - 0.3)
fig.tight_layout(w_pad=2)
save(fig, "fig_decomposition")

# ------------------------------------------------------------------ calibration
minute = json.load(open(V2 / "phase6" / "minute_pu" / "metrics.json"))["hourly_calibration"]["arms"]
series = [("PN, Waze-timed police-confirmed labels", minute["pn_reliable"]["deciles"], BLUE, "o"),
          ("nnPU, register prior", minute["nnpu_reliable"]["deciles"], ORANGE, "s"),
          ("nnPU + reporting propensity", minute["nnpu_reliable_propensity"]["deciles"], SLATE, "^")]
fig, ax = plt.subplots(figsize=(5.6, 5.6))
ax.plot([0.05, 20], [0.05, 20], color="#c9c8c0", lw=1.2, zorder=0)
ax.text(0.085, 0.062, "predicted = observed", rotation=45, color=GREY, fontsize=9, rotation_mode="anchor", transform_rotates_text=True)
for label, dec, colour, marker in series:
    ax.plot([d["predicted"] * 1e4 for d in dec], [d["observed"] * 1e4 for d in dec], marker=marker, ms=6, lw=2, color=colour, label=label)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.05, 15); ax.set_ylim(0.05, 15)
ticks = [0.1, 0.3, 1, 3, 10]
ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks]); ax.set_yticks(ticks); ax.set_yticklabels([str(t) for t in ticks])
ax.minorticks_off()
ax.set_xlabel("Predicted injury crashes per 10,000 edge-hours"); ax.set_ylabel("Observed injury crashes per 10,000 edge-hours")
ax.legend(loc="upper left", bbox_to_anchor=(0, 1.02))
ax.set_aspect("equal")
save(fig, "fig_calibration")

# ------------------------------------------------------------------ label hiding
hiding = pd.DataFrame(json.load(open(V2 / "phase8" / "metrics.json"))["hiding_summary"])
for column in ("predicted_over_observed", "predicted_over_observed_low_third", "auc_hidden_type"):
    hiding[column + "_mean"] = hiding[column].map(lambda d: d["mean"])
    hiding[column + "_std"] = hiding[column].map(lambda d: d["std"])
hiding["hidden_pct"] = 100 * (1 - hiding.labelled_share)
methods = [("pn", "PN (U as negative)", BLUE, "o"), ("pn_rescaled", "PN rescaled by the known prior", ORANGE, "s"),
           ("nnpu", "nnPU, register prior", SKY, "^"), ("nnpu_propensity", "nnPU + estimated propensity", SLATE, "D")]
panels = [("predicted_over_observed", "Predicted / observed crashes,\nwhole network"),
          ("predicted_over_observed_low_third", "Predicted / observed crashes,\nthird of units with lightest traffic"),
          ("auc_hidden_type", "ROC-AUC on crashes\nwhose label would be hidden")]
fig, axes = plt.subplots(3, 2, figsize=(9.5, 9.5), sharex=True)
for col, (mechanism, title) in enumerate((("scar", "Labels removed at random"), ("sar_flow", "Labels removed more often in light traffic"))):
    for row, (column, ylabel) in enumerate(panels):
        ax = axes[row, col]
        for offset, (method, label, colour, marker) in zip((-0.9, -0.3, 0.3, 0.9), methods):
            part = hiding[(hiding.mechanism == mechanism) & (hiding.method == method)].sort_values("hidden_pct")
            part = part[np.isfinite(part[column + "_mean"])]
            if part.empty:
                continue
            ax.errorbar(part.hidden_pct + offset, part[column + "_mean"], yerr=part[column + "_std"], marker=marker, ms=6, lw=2,
                        color=colour, ecolor=colour, elinewidth=1, capsize=0, label=label if (row == 0 and col == 1) else None)
        if row < 2:
            ax.axhline(1, color="#b9b8b0", lw=1.2, zorder=0)
        if col == 0:
            ax.set_ylabel(ylabel)
        if row == 0:
            ax.set_title(title, loc="left", fontsize=11, color="#222")
        if row == 2:
            ax.set_xlabel("Police-recorded crashes without a label in training (%)")
for row in range(3):
    lo = min(ax.get_ylim()[0] for ax in axes[row]); hi = max(ax.get_ylim()[1] for ax in axes[row])
    for ax in axes[row]:
        ax.set_ylim(lo, hi)
handles, labels = axes[0, 1].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.01))
fig.tight_layout(rect=(0, 0, 1, 0.97))
save(fig, "fig_label_hiding")

# ------------------------------------------------------------------ learners (E8)
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D

P9 = V2 / "phase9"
seq = json.load(open(P9 / "sequence_models" / "metrics.json"))
seq_long = json.load(open(P9 / "sequence_models_long" / "metrics.json"))
graph = json.load(open(P9 / "graph_models" / "metrics.json"))
metrics = {**seq["metrics"], **graph["metrics"]}
fits = {**seq["fits"], **graph["fits"]}
paired = {**seq["paired_comparisons"], **graph["paired_comparisons"]}

ARMS = [("L1_pn_reliable", "PN\nreliable"), ("L4_nnpu_reliable", "nnPU\nreliable"), ("L2_pn_union", "PN\nunion"),
        ("L3_pn_union_credibility", "PN union\ncredibility"), ("L5_mswspu", "MSWS-PU"), ("L6_mswspu_propensity", "MSWS-PU\npropensity")]
LEARNERS = [("mlp", "MLP, aggregated features"), ("gru", "GRU"), ("lstm", "LSTM"), ("tcn", "TCN"), ("transformer", "Transformer"),
            ("gat_k0", "ego-GAT, k = 0"), ("gat_k1", "ego-GAT, k = 1"), ("gat_k2", "ego-GAT, k = 2"), ("gat_k2_opp", "ego-GAT, k = 2 + opposite")]
NAME = dict(LEARNERS)
STYLE = {"mlp": (BLUE, "o"), "gru": (ORANGE, "s"), "lstm": (SLATE, "^"), "tcn": (SKY, "D"), "transformer": (PLUM, "v"),
         "gat_k2": (MAUVE, "P")}
SHOWN = ["mlp", "gru", "lstm", "tcn", "transformer", "gat_k2"]


def cell(learner, arm, source=metrics):
    return source[f"{learner}__{arm}__b30"]


def value(m, measure):
    if measure == "pop":
        e = m["population"]["official_all"]; return e["roc_auc"], e["roc_auc_ci95"]
    if measure == "pop_unreported":
        e = m["population"]["official_not_waze_reported"]; return e["roc_auc"], e["roc_auc_ci95"]
    if measure == "police":
        e = m["matched"]["official_all"]; return e["within_set_concordance"], e["ci95"]
    if measure == "waze":
        e = m["matched"]["waze_all"]; return e["within_set_concordance"], e["ci95"]
    if measure == "level":
        r = m["predicted_over_observed_hourly"]; return r, (r, r)
    raise KeyError(measure)


def delta(key, endpoint):
    d = paired[key][endpoint]
    return d["delta"], d["ci95"]


ENDPOINTS = {"pop": "population_official_all", "waze": "matched_waze_all", "police": "matched_official_all"}
pi_reg = fits["mlp__L4_nnpu_reliable__b30"]["prior"]
expected = {arm: fits[f"mlp__{arm}__b30"]["prior"] / pi_reg for arm, _ in ARMS}

# ---- fig_learners: every learner under every label treatment (four measures)
panels = [("pop", "(a) ROC-AUC across the network, all police crashes"),
          ("police", "(b) Within-set concordance, police risk sets"),
          ("waze", "(c) Within-set concordance, Waze-timed risk sets"),
          ("level", "(d) Predicted / observed recorded crash hours")]
offsets = np.linspace(-0.3, 0.3, len(SHOWN))
x = np.arange(len(ARMS))
fig, axes = plt.subplots(2, 2, figsize=(8.6, 7.4), sharex=True)
for ax, (measure, title) in zip(axes.flat, panels):
    widths = []
    for offset, learner in zip(offsets, SHOWN):
        colour, marker = STYLE[learner]
        vals = [value(cell(learner, arm), measure) for arm, _ in ARMS]
        y = np.array([v[0] for v in vals]); lo = np.array([v[1][0] for v in vals]); hi = np.array([v[1][1] for v in vals])
        widths += list(hi - lo)
        ax.plot(x + offset, y, color=colour, lw=1, alpha=0.4, zorder=2)
        ax.plot(x + offset, y, marker, color=colour, ms=6, mec="white", mew=0.7, zorder=4, label=NAME[learner])
    if measure != "level":
        # One reference bar instead of a bar per point: the median width of the 95 % bootstrap intervals.
        w = float(np.median(widths)); y0, y1 = ax.get_ylim()
        base = y0 + 0.06 * (y1 - y0)
        ax.vlines(len(ARMS) - 0.75, base, base + w, color=MUTED, lw=2, zorder=3)
        ax.annotate(f"median 95 % interval\n(width {w:.3f})", (len(ARMS) - 0.85, base + w / 2), ha="right", va="center",
                    color=MUTED, fontsize=8, linespacing=1.15)
    ax.axvline(1.5, color=RULE, lw=1, zorder=0)
    ax.set_title(title, loc="left", fontsize=10.5, color="#222")
    ax.grid(axis="x", visible=False)
    ax.set_xlim(-0.6, len(ARMS) - 0.4)
    if measure == "level":
        ax.set_yscale("log")
        ax.axhline(1, color="#b9b8b0", lw=1.2, zorder=0)
        for xi, (arm, _) in enumerate(ARMS):
            ax.plot([xi - 0.42, xi + 0.42], [expected[arm]] * 2, color=MUTED, lw=1.3, ls=(0, (2.5, 1.5)), zorder=1)
        ticks = [0.3, 0.5, 1, 2, 4, 8]
        ax.set_yticks(ticks); ax.set_yticklabels([str(t) for t in ticks]); ax.minorticks_off()
        ax.set_ylim(0.3, 12)
for ax in axes[1]:
    ax.set_xticks(x); ax.set_xticklabels([label for _, label in ARMS], fontsize=8)
    ax.annotate("484 reliable\npositives", (0.5, -0.25), xycoords=("data", "axes fraction"), ha="center", va="top",
                color=MUTED, fontsize=8.5, linespacing=1.15)
    ax.annotate("4,591 Waze-timed positives\n(reliable and weak)", (3.5, -0.25), xycoords=("data", "axes fraction"), ha="center",
                va="top", color=MUTED, fontsize=8.5, linespacing=1.15)
handles, labels = axes[0, 0].get_legend_handles_labels()
handles.append(Line2D([0], [0], color=MUTED, lw=1.3, ls=(0, (2.5, 1.5)))); labels.append("level implied by the class prior")
fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0), fontsize=8.5, handlelength=2.2, columnspacing=1.2)
fig.tight_layout(rect=(0, 0.05, 1, 0.93), h_pad=2.0, w_pad=1.6)
save(fig, "fig_learners")

# ---- fig_learner_ablation: paired differences along the information ladder (union positives, L2)
rows = [("Learned sequence encoder in place of aggregated features", None),
        ("GRU − MLP", "gru_minus_mlp__L2_pn_union"), ("LSTM − MLP", "lstm_minus_mlp__L2_pn_union"),
        ("TCN − MLP", "tcn_minus_mlp__L2_pn_union"), ("Transformer − MLP", "transformer_minus_mlp__L2_pn_union"),
        ("Neighbouring edges (ego-GAT ladder)", None),
        ("Attention block alone (k = 0) − GRU", "gat_k0_minus_gru__L2_pn_union"),
        ("One edge up- and downstream (k = 1) − k = 0", "gat_k1_minus_gat_k0__L2_pn_union"),
        ("Second hop (k = 2) − k = 1", "gat_k2_minus_gat_k1__L2_pn_union"),
        ("Opposite carriageway − k = 2", "gat_k2_opp_minus_gat_k2__L2_pn_union"),
        ("Full ego graph − MLP", "gat_k2_opp_minus_mlp__L2_pn_union")]
titles = [("pop", "Network-wide ROC-AUC,\nall police crashes"), ("waze", "Within-set concordance,\nWaze-timed risk sets"),
          ("police", "Within-set concordance,\npolice risk sets")]
fig, axes = plt.subplots(1, 3, figsize=(9.2, 4.4), sharey=True)
y = np.arange(len(rows))[::-1]
for ax, (measure, title) in zip(axes, titles):
    for yi, (label, key) in zip(y, rows):
        if key is None:
            continue
        d, (lo, hi) = delta(key, ENDPOINTS[measure])
        ax.plot([lo, hi], [yi, yi], color=BLUE, lw=2.2, solid_capstyle="round")
        ax.plot(d, yi, "o", color=BLUE, ms=7, mec="white", mew=0.8, zorder=3)
    ax.axvline(0, color=RULE, lw=1.4, zorder=0)
    ax.set_title(title, loc="left", fontsize=10.5, color="#222")
    ax.set_xlabel("Paired difference, 2025")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-0.04, 0.03)
    ax.set_xticks([-0.02, 0, 0.02]); ax.set_xticklabels(["−0.02", "0", "0.02"])
axes[0].set_yticks(y); axes[0].set_yticklabels([r[0] for r in rows]); axes[0].set_ylim(-0.7, len(rows) - 0.3)
for tick, (label, key) in zip(axes[0].get_yticklabels(), rows):
    if key is None:
        tick.set_fontweight("bold"); tick.set_color("#222")
fig.tight_layout(w_pad=3.2)
save(fig, "fig_learner_ablation")

# ---- fig_learner_label_effects: the three label-treatment differences of E3, in every learner
comparisons = [("Weak positives added\n(PN union − PN reliable)", "{l}__L2_pn_union_minus_L1_pn_reliable"),
               ("nnPU risk in place of PN\n(MSWS-PU − PN union, credibility)", "{l}__L5_mswspu_minus_L3_pn_union_credibility"),
               ("Propensity weights added\n(MSWS-PU, propensity − MSWS-PU)", "{l}__L6_mswspu_propensity_minus_L5_mswspu")]
measures = [("pop", "ROC-AUC across the network, all police crashes"), ("waze", "Within-set concordance, Waze-timed risk sets")]
fig, axes = plt.subplots(2, 3, figsize=(9.2, 6.4), sharey=True)
y = np.arange(len(LEARNERS))[::-1]
for row, (measure, ylabel) in enumerate(measures):
    for col, (title, pattern) in enumerate(comparisons):
        ax = axes[row, col]
        for yi, (learner, _) in zip(y, LEARNERS):
            d, (lo, hi) = delta(pattern.format(l=learner), ENDPOINTS[measure])
            ax.plot([lo, hi], [yi, yi], color=BLUE, lw=2.2, solid_capstyle="round")
            ax.plot(d, yi, "o", color=BLUE, ms=6.5, mec="white", mew=0.8, zorder=3)
        ax.axvline(0, color=RULE, lw=1.4, zorder=0)
        ax.set_xlim(-0.03, 0.09); ax.set_xticks([0, 0.03, 0.06, 0.09])
        ax.grid(axis="y", visible=False)
        for yy in (4.5, 0.5):  # separators: MLP | sequence | graph (rows are reversed)
            pass
        ax.axhline(y[0] - 0.5, color="#e1e0d9", lw=1, zorder=0)   # below MLP
        ax.axhline(y[4] - 0.5, color="#e1e0d9", lw=1, zorder=0)   # below Transformer
        if row == 0:
            ax.set_title(title, loc="left", fontsize=10, color="#222")
        if row == 1:
            ax.set_xlabel("Paired difference, 2025")
        if col == 0:
            ax.set_ylabel(ylabel, fontsize=9.5)
axes[0, 0].set_yticks(y); axes[0, 0].set_yticklabels([n for _, n in LEARNERS]); axes[0, 0].set_ylim(-0.7, len(LEARNERS) - 0.3)
fig.tight_layout(w_pad=1.2, h_pad=1.5)
save(fig, "fig_learner_label_effects")

# ---- fig_learner_grid (appendix): the complete learner x treatment grid, four measures
SEQ_STEPS = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
             "#184f95", "#104281", "#0d366b"]
seq_cmap = LinearSegmentedColormap.from_list("blue_seq", SEQ_STEPS)
div_cmap = LinearSegmentedColormap.from_list("blue_red", [BLUE, "#f0efec", RED])
grid_panels = [("pop", "ROC-AUC across the network,\nall police crashes", "{:.3f}"),
               ("police", "Within-set concordance,\npolice risk sets", "{:.3f}"),
               ("waze", "Within-set concordance,\nWaze-timed risk sets", "{:.3f}"),
               ("level", "Predicted / observed\nrecorded crash hours", "{:.2f}")]
fig, axes = plt.subplots(2, 2, figsize=(8.8, 8.4))
for ax, (measure, title, fmt) in zip(axes.flat, grid_panels):
    Z = np.array([[value(cell(l, a), measure)[0] for a, _ in ARMS] for l, _ in LEARNERS])
    if measure == "level":
        shown = np.log10(Z); norm = TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1); cmap = div_cmap
    else:
        shown = Z; norm = Normalize(vmin=Z.min(), vmax=Z.max()); cmap = seq_cmap
    ax.imshow(shown, cmap=cmap, norm=norm, aspect="auto")
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            rgba = cmap(norm(shown[i, j])); lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            ax.text(j, i, fmt.format(Z[i, j]), ha="center", va="center", fontsize=8.5, color="white" if lum < 0.5 else INK)
    ax.set_xticks(range(len(ARMS))); ax.set_xticklabels([label for _, label in ARMS], fontsize=8)
    ax.set_yticks(range(len(LEARNERS))); ax.set_yticklabels([n for _, n in LEARNERS] if ax in axes[:, 0] else [])
    ax.set_title(title, loc="left", fontsize=10.5, color="#222")
    ax.grid(False); ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for yy in (0.5, 4.5):
        ax.axhline(yy, color="white", lw=2.5)
    ax.axvline(1.5, color="white", lw=2.5)
fig.tight_layout(w_pad=1.0, h_pad=2.0)
save(fig, "fig_learner_grid")

# ---- numbers for the E8 tables (printed; pasted into thesis.tex)
def fmt_ci(v, ci):
    return f"{v:.3f} $[{ci[0]:.3f}, {ci[1]:.3f}]$"

print("\n%% tab:learner-results rows (L2 and L5 blocks)")
for arm, block in (("L2_pn_union", "All Waze-timed positives, PN risk (L2)"), ("L5_mswspu", "MSWS-PU: nnPU risk, any-crash prior, credibility weights (L5)")):
    print(f"\\multicolumn{{8}}{{l}}{{\\emph{{{block}}}}} \\\\")
    for learner, name in LEARNERS:
        m = cell(learner, arm); f = fits[f"{learner}__{arm}__b30"]
        print(f"{name} & {fmt_ci(*value(m, 'pop'))} & {fmt_ci(*value(m, 'pop_unreported'))} & {fmt_ci(*value(m, 'police'))} & "
              f"{fmt_ci(*value(m, 'waze'))} & {value(m, 'level')[0]:.2f} & {f['parameters']:,} & {f['seconds']:.0f} \\\\")

print("\n%% variance decomposition over the 9 x 6 grid")
def decompose(Z):
    grand = Z.mean(); row = Z.mean(1, keepdims=True); col = Z.mean(0, keepdims=True)
    ss_total = ((Z - grand) ** 2).sum(); ss_row = (Z.shape[1] * (row - grand) ** 2).sum(); ss_col = (Z.shape[0] * (col - grand) ** 2).sum()
    return ss_row / ss_total, ss_col / ss_total, 1 - (ss_row + ss_col) / ss_total, row.max() - row.min(), col.max() - col.min()
for measure, label in (("pop", "ROC-AUC, all police crashes"), ("pop_unreported", "ROC-AUC, not reported to Waze"),
                       ("police", "Concordance, police risk sets"), ("waze", "Concordance, Waze-timed risk sets"), ("level", "log10 predicted / observed")):
    Z = np.array([[value(cell(l, a), measure)[0] for a, _ in ARMS] for l, _ in LEARNERS])
    if measure == "level":
        Z = np.log10(Z)
    r, c, i, rr, cr = decompose(Z)
    print(f"{label} & {100*r:.0f} & {100*c:.0f} & {100*i:.0f} & {rr:.3f} & {cr:.3f} \\\\")

print("\n%% tab:learner-robustness rows: default schedule vs longer schedule (pop AUC police | Waze concordance), L1 / L2 / L5")
for learner in ("mlp", "gru", "lstm", "tcn", "transformer"):
    parts = []
    for arm in ("L1_pn_reliable", "L2_pn_union", "L5_mswspu"):
        a = cell(learner, arm, seq["metrics"]); b = cell(learner, arm, seq_long["metrics"])
        parts.append(f"{value(a,'pop')[0]:.3f} & {value(b,'pop')[0]:.3f} & {value(a,'waze')[0]:.3f} & {value(b,'waze')[0]:.3f}")
    print(f"{NAME[learner]} & " + " & ".join(parts) + " \\\\")
diffs = []
for learner in ("mlp", "gru", "lstm", "tcn", "transformer"):
    for arm, _ in ARMS:
        for meas in ("pop", "police", "waze"):
            diffs.append(value(cell(learner, arm, seq_long["metrics"]), meas)[0] - value(cell(learner, arm, seq["metrics"]), meas)[0])
print(f"max |long - default| over all cells and measures: {max(abs(d) for d in diffs):.4f}")

print("\n%% early-stopping steps with reliable positives (L1) and union (L2)")
for learner, _ in LEARNERS:
    print(learner, "L1", fits[f"{learner}__L1_pn_reliable__b30"]["best_steps"], "L2", fits[f"{learner}__L2_pn_union__b30"]["best_steps"],
          "params", fits[f"{learner}__L2_pn_union__b30"]["parameters"], "sec", round(fits[f"{learner}__L2_pn_union__b30"]["seconds"], 1))
print("expected level by arm:", {a: round(v, 2) for a, v in expected.items()})
