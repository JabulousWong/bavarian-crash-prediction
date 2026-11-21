"""Figure of the pre-crash feature windows used by real-time crash prediction studies, 2017-2026.

Each row is one study whose window definition was verified in the full text (or, for Yang et al. 2026
and Azati et al. 2025, in the abstract). Bars show the minutes before the recorded crash time from which
the positive-class traffic features are taken; gaps mark the aggregation slices the study stacks.
Blue: the window starts at least five minutes before the recorded time (a buffer is kept).
Orange: the window reaches into the last five minutes before (or the interval containing) the recorded time.

Run: python make_fig_windows.py   (writes image/fig_precrash_windows.{pdf,png})
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

HERE = Path(__file__).resolve().parent
OUT = HERE / "image"
OUT.mkdir(exist_ok=True)

BLUE, ORANGE = "#2a78d6", "#eb6834"          # the two categorical slots used throughout the thesis figures
GREY, INK, MUTED, RULE = "#898781", "#0b0b0b", "#52514e", "#c3c2b7"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False, "axes.edgecolor": RULE,
    "axes.labelcolor": "#444", "xtick.color": "#666", "ytick.color": "#444", "axes.grid": True,
    "grid.color": "#e1e0d9", "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.facecolor": "white",
})

# (label, slices in minutes before the recorded crash time, buffer kept?, data source)
STUDIES = [
    ("You et al. (2017)",          [(5, 10)],                                   True,  "loop detectors, 5-min"),
    ("Ke et al. (2018)",           [(5, 10), (10, 15)],                         True,  "microwave radar, 5-min"),
    ("Chen and Qin (2019)",        [(0, 5)],                                    False, "loop detectors + CTM, 1-min"),
    ("Katrakazas et al. (2019)",   [(0, 4)],                                    False, "raw speed series, 1-min (best of 1-5)"),
    ("Peng et al. (2020)",         [(0, 5), (5, 10), (10, 15), (15, 20)],       False, "loop detectors, 5-min"),
    ("Cai et al. (2020)",          [(5, 10)],                                   True,  "MVDS radar, 5-min"),
    ("Li et al. (2020)",           [(5, 10), (10, 15), (15, 20)],               True,  "arterial detectors + signals, 5-min"),
    ("Man et al. (2022)",          [(5, 10), (10, 15), (15, 20), (20, 25), (25, 30)], True, "MIDAS loops, 5-min"),
    ("Ma et al. (2023)",           [(0, 10)],                                   False, "navigation app, 10-min step"),
    ("Mo et al. (2024)",           [(5, 10), (10, 15)],                         True,  "toll-plaza detectors, 5-min"),
    ("Zhang et al. (2024)",        [(0, 20)],                                   False, "HERE probe speeds, 1-min"),
    ("Alafi and Moradi (2024)",    [(0, 4), (4, 8), (8, 12)],                   False, "urban sensors, 4-min"),
    ("Wang et al. (2025)",         [(5, 10), (10, 15), (15, 20)],               True,  "floating car data, 5-min"),
    ("Hossain et al. (2025)",      [(1, 9)],                                    False, "loop detectors, 1-min"),
    ("Azati et al. (2025)",        [(0, 60)],                                   False, "hourly segment data"),
    ("Wang et al. (2026)",         [(5, 10), (10, 15), (15, 20)],               True,  "connected vehicles, 5-min"),
    ("Guo et al. (2026)",          [(0, 10), (10, 20), (20, 30)],               False, "navigation app, 10-min step (+ crash slice)"),
    ("Han et al. (2026)",          [(5, 10), (10, 15), (15, 20), (20, 25)],     True,  "detectors, secondary crashes, 5-min"),
    ("Yang et al. (2026)",         [(5, 10)],                                   True,  "network-wide detectors, 5-min"),
]

XMAX = 32.0
fig, ax = plt.subplots(figsize=(8.8, 7.4))
fig.subplots_adjust(right=0.70)   # room for the data-source notes in the right-hand margin
n = len(STUDIES)
ys = list(range(n))[::-1]   # first study at the top

# convention band and the two reference rules
ax.axvspan(5, 10, color="#efeee8", zorder=0)
ax.axvline(5, color=RULE, lw=1.0, zorder=1)
ax.axvline(0, color=INK, lw=1.2, zorder=1)

H, GAP = 0.56, 0.22
for y, (label, slices, buffer, source) in zip(ys, STUDIES):
    colour = BLUE if buffer else ORANGE
    for (a, b) in slices:
        if b > XMAX:   # the hourly design: the bar runs off the left edge
            ax.add_patch(Rectangle((a + GAP / 2, y - H / 2), XMAX - a - GAP / 2, H, color=colour, lw=0, zorder=3))
            ax.text(XMAX - 0.4, y, "0 to 60 min", color="white", fontsize=8, va="center", ha="left", zorder=4)
        else:
            ax.add_patch(Rectangle((a + GAP / 2, y - H / 2), (b - a) - GAP, H, color=colour, lw=0, zorder=3))
    # data source in the right-hand margin, beyond the crash-time rule
    ax.text(-0.7, y, source, color=MUTED, fontsize=8.5, va="center", ha="left", clip_on=False)

ax.set_ylim(-0.7, n - 0.3)
ax.set_yticks(ys)
ax.set_yticklabels([s[0] for s in STUDIES], fontsize=9.5, color="#222")
ax.tick_params(axis="y", length=0)
ax.set_xlim(XMAX, -0.3)       # the crash time sits at the right-hand edge
ax.set_xticks(range(0, 31, 5))
ax.set_xlabel("Minutes before the recorded crash time")
ax.grid(axis="y", visible=False)
ax.text(-0.7, n - 0.35, "recorded crash time", color=INK, fontsize=8.5, ha="left", va="bottom", clip_on=False)
ax.text(7.5, n - 0.35, "5-10 min convention\n(Abdel-Aty et al., 2004)", color=MUTED, fontsize=8, ha="center", va="bottom")

handles = [Patch(color=BLUE, label="window starts 5 min or more before the recorded time"),
           Patch(color=ORANGE, label="window reaches into the last 5 min before the recorded time")]
fig.legend(handles=handles, loc="upper center", ncol=1, bbox_to_anchor=(0.42, 1.0), fontsize=9.2, handlelength=1.6)
fig.savefig(OUT / "fig_precrash_windows.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_precrash_windows.png", dpi=220, bbox_inches="tight")
print("wrote", OUT / "fig_precrash_windows.{pdf,png}")
