"""Range chart of real-time crash prediction studies by publication year and pre-crash window design, 2016-2026,
x = year, each study drawn as a bar from the start to the end of its pre-crash window,
hollow circles = no explicit minute range. Studies are those whose window definition was verified in the full text or abstract.

Run: python make_fig_window_trend.py   (writes image/fig_precrash_window_trend.{pdf,png} and prints the counts)
"""
from collections import Counter
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
OUT = HERE / "image"
OUT.mkdir(exist_ok=True)

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"     # categorical slots 1-3 of the thesis figures
INK, MUTED, RULE = "#0b0b0b", "#52514e", "#c3c2b7"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": RULE, "axes.labelcolor": "#444",
    "xtick.color": "#666", "ytick.color": "#666", "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42, "savefig.facecolor": "white",
})

# window-design categories
A = "5 min or more before the crash"
B = "less than 5 min before the crash"
C = "other time unit (10-min step, hourly,\nsignal cycle, seconds)"
COLOUR = {A: BLUE, B: ORANGE, C: GREEN}

# (year, category, study) - one row per study
STUDIES = [
    (2016, A, "Sun and Sun 2016, 5-10 min"),
    (2017, A, "You et al. 2017, 5-10 min"),
    (2018, A, "Ke et al. 2018, 5-15 min"),
    (2019, B, "Chen and Qin 2019, 0-5 min"),
    (2019, B, "Katrakazas et al. 2019, 0-4 min"),
    (2020, A, "Cai et al. 2020, 5-10 min"),
    (2020, A, "Li et al. 2020, 5-20 min"),
    (2020, B, "Peng et al. 2020, 0-20 min"),
    (2021, A, "Yang et al. 2021, ending 5 min before"),
    (2022, A, "Man et al. 2022, 5-30 min"),
    (2022, A, "Li and Abdel-Aty 2022, next 5+ min"),
    (2023, A, "Syed and Hasan 2023, previous 5/10/15 min"),
    (2023, C, "Ma et al. 2023, 10-min step"),
    (2024, A, "Mo et al. 2024, 5-15 min"),
    (2024, B, "Zhang et al. 2024, 0-20 min"),
    (2024, B, "Alafi and Moradi 2024, 0-12 min"),
    (2024, C, "Hussain et al. 2024, signal cycles"),
    (2024, C, "Zhu et al. 2024, seconds"),
    (2025, A, "Wang et al. 2025, 5-20 min"),
    (2025, B, "Hossain et al. 2025, 1-9 min"),
    (2025, C, "Azati et al. 2025, hourly"),
    (2026, A, "Wang et al. 2026, 5-20 min"),
    (2026, A, "Han et al. 2026, 5-25 min"),
    (2026, A, "Yang et al. 2026, 5-10 min"),
    (2026, C, "Guo et al. 2026, 10-min step"),
]

counts = Counter((y, c) for y, c, _ in STUDIES)
years = list(range(2016, 2027))
print("year  A  B  C")
for y in years:
    print(y, *(counts.get((y, cat), 0) for cat in (A, B, C)))
print("total", {k: sum(v for (y, c), v in counts.items() if c == k) for k in (A, B, C)}, "n =", len(STUDIES))

# window as (start, end) in minutes before the recorded crash time; studies missing here state no explicit minute
# range (Yang 2021: ends 5 min before, start not stated; Li and Abdel-Aty 2022: "next 5+ min"; Syed and Hasan 2023:
# 5, 10 and 15 min compared; Ma 2023 and Guo 2026: 10-min steps; Hussain 2024: signal cycles; Zhu 2024: seconds;
# Azati 2025: hourly units)
RANGE = {
    "Sun and Sun 2016": (10, 5), "You et al. 2017": (10, 5), "Ke et al. 2018": (15, 5), "Chen and Qin 2019": (5, 0),
    "Katrakazas et al. 2019": (4, 0), "Cai et al. 2020": (10, 5), "Li et al. 2020": (20, 5), "Peng et al. 2020": (20, 0),
    "Man et al. 2022": (30, 5), "Mo et al. 2024": (15, 5), "Zhang et al. 2024": (20, 0), "Alafi and Moradi 2024": (12, 0),
    "Wang et al. 2025": (20, 5), "Hossain et al. 2025": (9, 1), "Wang et al. 2026": (20, 5), "Han et al. 2026": (25, 5),
    "Yang et al. 2026": (10, 5),
}

def window(study):
    return RANGE.get(study.rsplit(",", 1)[0])

Y_NS = -5   # row for studies without an explicit minute range
fig, ax = plt.subplots(figsize=(8.0, 4.8))
ax.axhspan(0, 5, color="#f3e3da", zorder=0, lw=0)
ax.text(2015.5, 2.5, "last 5 min before crash", ha="left", va="center", fontsize=8.5, color="#a0532f")
for y in years:
    drawn = [(st, window(st)) for yr, _, st in STUDIES if yr == y and window(st)]
    drawn.sort(key=lambda t: (-t[1][0], -t[1][1]))
    offsets = {1: [0], 2: [-0.16, 0.16], 3: [-0.26, 0, 0.26]}[len(drawn)] if drawn else []
    for dx, (st, (a, b)) in zip(offsets, drawn):
        ax.plot([y + dx, y + dx], [b, a], color=BLUE, lw=6, solid_capstyle="butt", zorder=3)
    ns = [st for yr, _, st in STUDIES if yr == y and not window(st)]
    for k, st in enumerate(ns):
        dx = (k - (len(ns) - 1) / 2) * 0.24
        ax.scatter(y + dx, Y_NS, s=60, facecolor="white", edgecolor=MUTED, linewidth=1.3, zorder=3)
ax.axhline(-2.5, color=RULE, lw=0.8, ls=(0, (3, 3)), zorder=1)
ax.set_xlim(2015.4, 2026.6)
ax.set_xticks(years)
ax.set_ylim(-7.5, 32)
ax.set_yticks([Y_NS, 0, 5, 10, 15, 20, 25, 30], ["no explicit\nrange", "0 (crash)", "5", "10", "15", "20", "25", "30"])
ax.set_xlabel("Year of publication")
ax.set_ylabel("Minutes before the recorded crash time")
ax.grid(axis="x", visible=False)
fig.savefig(OUT / "fig_precrash_window_trend.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_precrash_window_trend.png", dpi=220, bbox_inches="tight")
print("wrote", OUT / "fig_precrash_window_trend.{pdf,png}")
