"""Conceptual figure of positive-unlabelled learning on synthetic two-dimensional data.

fig_pu_concept   (a) the latent classes, which are never observed;
                 (b) what a PU learner sees under SCAR: a random share c of the crashes is
                     labelled, everything else is unlabelled;
                 (c) the decision boundary of a classifier that treats U as negative, against
                     the boundary after the Elkan-Noto correction p(y|x) = p(s|x)/c;
                 (d) the same data under SAR: the labelling probability rises with traffic
                     volume, so the labelled crashes are a biased sample of all crashes.

Synthetic illustration only; no thesis data enter it.
Run with the analysis environment: python make_fig_pu_concept.py
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.linear_model import LogisticRegression

OUT = Path(__file__).resolve().parent / "image"

BLUE, ORANGE, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#a9a8a1", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#444",
    "legend.frameon": False, "pdf.fonttype": 42, "savefig.facecolor": "white",
})

rng = np.random.default_rng(7)
N_POS, N_NEG, C = 120, 360, 0.4          # crashes, non-crashes, label frequency under SCAR

pos = rng.multivariate_normal([1.1, 1.0], [[0.55, 0.15], [0.15, 0.55]], N_POS)
neg = rng.multivariate_normal([-0.6, -0.5], [[0.9, 0.1], [0.1, 0.9]], N_NEG)
X = np.vstack([pos, neg])
y = np.r_[np.ones(N_POS), np.zeros(N_NEG)].astype(int)

# SCAR: every crash labelled with the same probability c
s_scar = (y == 1) & (rng.random(len(y)) < C)
# SAR: labelling probability depends on x1 (traffic volume), average close to c
e_x = 1 / (1 + np.exp(-2.6 * (X[:, 0] - 1.3)))
s_sar = (y == 1) & (rng.random(len(y)) < e_x)

# classifiers for panel (c)
grid_x, grid_y = np.meshgrid(np.linspace(-3.2, 3.6, 300), np.linspace(-3.2, 3.6, 300))
G = np.c_[grid_x.ravel(), grid_y.ravel()]
true_clf = LogisticRegression().fit(X, y)                    # oracle, knows y
naive_clf = LogisticRegression().fit(X, s_scar.astype(int))  # treats U as negative
c_hat = naive_clf.predict_proba(X[s_scar])[:, 1].mean()      # Elkan-Noto estimator e1
p_true = true_clf.predict_proba(G)[:, 1].reshape(grid_x.shape)
p_naive = naive_clf.predict_proba(G)[:, 1].reshape(grid_x.shape)
p_pu = np.clip(p_naive / c_hat, 0, 1)

MS = 22


def frame(ax, title):
    ax.set_xlim(-3.2, 3.6)
    ax.set_ylim(-3.2, 3.6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Traffic volume $x_1$")
    ax.set_ylabel("Speed variance $x_2$")
    ax.set_title(title, loc="left", fontsize=10, color=INK)


def unlabelled(ax, mask):
    ax.scatter(*X[mask].T, s=MS, facecolor="white", edgecolor=GREY, lw=0.9, zorder=2)


def labelled(ax, mask):
    ax.scatter(*X[mask].T, s=MS + 6, color=ORANGE, edgecolor="white", lw=0.5, marker="o", zorder=3)


fig, axes = plt.subplots(2, 2, figsize=(9.0, 8.2))
(a, b), (c, d) = axes

frame(a, "(a) Latent classes (not observed)")
a.scatter(*neg.T, s=MS, color=BLUE, alpha=0.75, edgecolor="white", lw=0.4, label="Non-crash, $Y=0$")
a.scatter(*pos.T, s=MS, color=ORANGE, alpha=0.9, edgecolor="white", lw=0.4, label="Crash, $Y=1$")
a.legend(loc="upper left", fontsize=8.5, handletextpad=0.2)

frame(b, f"(b) PU observation under SCAR, $c={C}$")
unlabelled(b, ~s_scar)
labelled(b, s_scar)
b.legend(handles=[
    Line2D([], [], ls="", marker="o", color=ORANGE, markeredgecolor="white", label="Labelled positive, $S=1$"),
    Line2D([], [], ls="", marker="o", markerfacecolor="white", markeredgecolor=GREY, label="Unlabelled, $S=0$"),
], loc="upper left", fontsize=8.5, handletextpad=0.2)

frame(c, "(c) Decision boundaries at 0.5")
unlabelled(c, ~s_scar)
labelled(c, s_scar)
for p, col, ls in ((p_true, INK, ":"), (p_naive, MUTED, "--"), (p_pu, ORANGE, "-")):
    c.contour(grid_x, grid_y, p, levels=[0.5], colors=col, linestyles=ls, linewidths=1.8)
c.legend(handles=[
    Line2D([], [], color=INK, ls=":", lw=1.8, label="Oracle, trained on $Y$"),
    Line2D([], [], color=MUTED, ls="--", lw=1.8, label="Naive, $U$ treated as negative"),
    Line2D([], [], color=ORANGE, ls="-", lw=1.8, label=r"PU, $\hat p(s{=}1\mid x)/\hat c$"),
], loc="upper left", fontsize=8.5, frameon=True, facecolor="white", edgecolor="none", framealpha=1)

frame(d, "(d) PU observation under SAR, $e(x)$ rises with $x_1$")
unlabelled(d, ~s_sar)
labelled(d, s_sar)
d.annotate("", xy=(3.3, -2.9), xytext=(0.2, -2.9),
           arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))
d.text(1.75, -2.75, "labelling probability $e(x)$", ha="center", va="bottom", fontsize=8.5, color=MUTED)

fig.tight_layout()
for ext, kw in (("pdf", {}), ("png", {"dpi": 220})):
    fig.savefig(OUT / f"fig_pu_concept.{ext}", bbox_inches="tight", **kw)
print(f"wrote fig_pu_concept  (labelled SCAR={s_scar.sum()}, SAR={s_sar.sum()} of {N_POS}; c_hat={c_hat:.2f})")
