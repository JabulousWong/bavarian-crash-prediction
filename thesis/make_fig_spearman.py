"""Scatter plots of Unfallatlas crashes against each loop-detector variable (Data chapter), drawn from
analysis/results/v2/data_spearman (written by analysis/data_spearman.py). One 3 x 3 figure per level, the eighth
cell holds the legend and the ninth is empty; each panel title carries the Spearman rho of 2024 and 2025, and the grey line is a LOWESS
smoother fitted to both years pooled.

fig_crash_detector_scatter_spatial   one point per detector edge, crashes per km against the annual edge summary
fig_crash_detector_scatter_temporal  one point per hour of the week, crash count against the network-mean profile
fig_crash_detector_scatter_{spatial,temporal}_{rear_end,run_off_road,goods_vehicle}  the same for crash subgroups

Run: python make_fig_spearman.py   (writes image/fig_crash_detector_scatter_*.{pdf,png})
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter
import matplotlib.patheffects as pe
from statsmodels.nonparametric.smoothers_lowess import lowess

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "analysis" / "results" / "v2" / "data_spearman"
OUT = HERE / "image"
OUT.mkdir(exist_ok=True)

BLUE, ORANGE = "#2a78d6", "#eb6834"     # categorical slots 1-2 of the thesis figures
INK, MUTED, RULE = "#0b0b0b", "#52514e", "#c3c2b7"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": RULE, "axes.labelcolor": "#444",
    "xtick.color": "#666", "ytick.color": "#666", "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42, "savefig.facecolor": "white",
})
YEARS = {2024: (BLUE, "o"), 2025: (ORANGE, "s")}
# variable, panel title, x-axis label, scale factor for the axis
VARIABLES = [
    ("mean_flow", "Mean hourly volume", "veh/h", 1),
    ("mean_occupancy", "Mean occupancy", "%", 1),
    ("congested_share", "Share of congested hours", "% of hours with v < 60 km/h", 100),
    ("mean_net_gap", "Mean net time gap", "s", 1),
    ("mean_speed", "Mean speed", "km/h", 1),
    ("mean_speed_sd", "Within-minute speed SD", "km/h", 1),
    ("hgv_share", "HGV share", "%", 100),
]
LOWESS_FRAC = 0.5
HALO = [pe.Stroke(linewidth=4, foreground="white"), pe.Normal()]
PLAIN = FuncFormatter(lambda v, _: f"{v:g}")

rho = pd.read_csv(DATA / "spearman.csv")
LEVELS = {
    "spatial": dict(table=pd.read_csv(DATA / "edge_year_table.csv"), level="edge", per_km=True, size=12, alpha=0.55, symlog={"congested_share": 1, "y": 1},
                    trim=0.01,  # LOWESS drawn only between the 1st and 99th percentile of x; the tails are a few edges
                    unit="one point per detector edge"),
    "temporal": dict(table=pd.read_csv(DATA / "hour_of_week_table.csv"), level="hour_of_week", per_km=False, size=18, alpha=0.8, symlog={}, trim=0,
                     unit="one point per hour of the week"),
}
# crash group: column in the data tables (spatial: <column>_per_km), axis label, file-name suffix
GROUPS = [("crashes", "Crashes", ""), ("rear_end", "Rear-end crashes", "_rear_end"),
          ("run_off_road", "Run-off-road crashes", "_run_off_road"),
          ("goods_vehicle", "Goods-vehicle crashes", "_goods_vehicle")]

for (name, spec), (group, group_label, file_suffix) in [(level, group) for level in LEVELS.items() for group in GROUPS]:
    table = spec["table"]
    target = f"{group}_per_km" if spec["per_km"] else group
    ylabel = f"{group_label} per km" if spec["per_km"] else group_label
    fig, axes = plt.subplots(3, 3, figsize=(7.6, 7.0))
    for ax, (column, title, unit, factor) in zip(axes.flat, VARIABLES):
        coefficients = []
        for year, (colour, marker) in YEARS.items():
            part = table[table.year == year]
            ax.scatter(part[column] * factor, part[target], s=spec["size"], marker=marker, color=colour,
                       edgecolor="white", linewidth=0.4, alpha=spec["alpha"])
            row = rho[(rho.level == spec["level"]) & (rho.target == target)
                      & (rho.variable == column) & (rho.year == year)].iloc[0]
            coefficients.append(f"{row.rho + 0:.2f}".replace("-0.00", "0.00") + ("" if row.p_value < 0.05 else r"$^{\mathrm{ns}}$"))
        x = table[column] * factor
        inner = x.between(x.quantile(spec["trim"]), x.quantile(1 - spec["trim"]))
        curve = lowess(table.loc[inner, target], x[inner], frac=LOWESS_FRAC, it=0)  # no robustness steps: they discard the few non-zero edges
        ax.plot(curve[:, 0], curve[:, 1], color=MUTED, lw=2, path_effects=HALO, zorder=4)
        ax.set_title(f"{title}\nρ = {coefficients[0]} (2024), {coefficients[1]} (2025)",
                     loc="left", fontsize=8.5, color=INK)
        ax.set_xlabel(unit, fontsize=8.5)
        if column in spec["symlog"]:
            ax.set_xscale("symlog", linthresh=spec["symlog"][column]); ax.xaxis.set_major_formatter(PLAIN)
        if "y" in spec["symlog"]:
            ax.set_yscale("symlog", linthresh=spec["symlog"]["y"]); ax.yaxis.set_major_formatter(PLAIN)
            ax.yaxis.set_major_locator(FixedLocator([0, 0.5, 1, 2, 5, 10, 20]))
        ax.set_ylim(bottom=0)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    axes.flat[8].axis("off")
    legend = axes.flat[7]
    legend.axis("off")
    handles = [Line2D([], [], ls="", marker=marker, ms=7, color=colour, markeredgecolor="white", label=str(year))
               for year, (colour, marker) in YEARS.items()]
    handles.append(Line2D([], [], color=MUTED, lw=2, label="LOWESS, both years"))
    legend.legend(handles=handles, loc="center left", fontsize=10, handletextpad=0.3,
                  title=f"{spec['unit'].capitalize()}\n" + r"$^{\mathrm{ns}}$: p ≥ 0.05", title_fontsize=9, alignment="left")
    fig.tight_layout(h_pad=1.0, w_pad=1.0)
    for suffix, extra in (("pdf", {}), ("png", {"dpi": 220})):
        fig.savefig(OUT / f"fig_crash_detector_scatter_{name}{file_suffix}.{suffix}", bbox_inches="tight", **extra)
    plt.close(fig)
    print("wrote", f"fig_crash_detector_scatter_{name}{file_suffix}")
