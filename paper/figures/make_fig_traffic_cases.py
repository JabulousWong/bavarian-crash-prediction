"""Detector traffic around three recorded crash hours (illustration of the hour-only time reference).

Reads the per-case detector extracts in <repo>/traffic_case_output and writes
fig_traffic_cases.pdf / .png next to this script, in the style of make_figures.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as md
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "traffic_case_output"
MM = 1 / 25.4

SURFACE, INK, SECONDARY, MUTED, GRID, AXIS = "#ffffff", "#0b0b0b", "#52514e", "#6f6e69", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
BAND = "#ecebe6"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 8,
    "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": False,
})

TOWARDS = {"A99_Sbg": "A99 to Salzburg", "A9_Nbg": "A9 to Nuremberg", "A99_Stg_Lin": "A99 to Stuttgart"}


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, length=0)
    ax.yaxis.label.set_color(SECONDARY)


def main() -> None:
    cases = json.loads((SRC / "selected_cases_direction.json").read_text(encoding="utf-8"))
    ref = pd.read_csv(SRC / "upstream_downstream_reference.csv", dtype={"uid": str}).set_index("uid")

    fig, axes = plt.subplots(2, 3, figsize=(174 * MM, 92 * MM), sharex="col")
    for i, c in enumerate(cases):
        H = pd.Timestamp(c["date"]) + pd.Timedelta(hours=c["hour"])
        r = ref.loc[c["uid"]]
        series = {}
        for letter, unit in zip("AB", c["units"]):
            d = pd.read_csv(SRC / f"{c['uid']}_{letter}.csv", parse_dates=["t_start"])
            series[unit] = d
        lines = [(r.upstream_unit, BLUE, "Upstream detector"),
                 (r.downstream_unit, ORANGE, "Downstream detector")]
        for j, (col, ylab) in enumerate([("v_kfz", "Speed (km/h)"), ("q_kfz_h", "Flow (veh/h)")]):
            ax = axes[j, i]
            style(ax)
            ax.axvspan(H, H + pd.Timedelta(hours=1), color=BAND, lw=0, zorder=0)
            for unit, colour, label in lines:
                d = series[unit]
                ax.plot(d.t_start, d[col], color=colour, lw=0.8, label=label)
            ax.set_xlim(H - pd.Timedelta(minutes=30), H + pd.Timedelta(minutes=90))
            ax.xaxis.set_major_locator(md.MinuteLocator(byminute=[0, 30]))
            ax.xaxis.set_major_formatter(md.DateFormatter("%H:%M"))
            if i == 0:
                ax.set_ylabel(ylab)
            if j == 0:
                d = pd.Timestamp(c["date"])
                ax.set_title(f"({'abc'[i]}) {TOWARDS[c['route']]}, {d.day} {d:%b %Y}",
                             loc="left", color=INK)
            else:
                ax.set_xlabel("Time of day")
                ax.xaxis.label.set_color(SECONDARY)
            ax.set_ylim(bottom=0)
    handles = [plt.Line2D([], [], color=BLUE, lw=1.2), plt.Line2D([], [], color=ORANGE, lw=1.2),
               plt.Rectangle((0, 0), 1, 1, color=BAND)]
    fig.legend(handles, ["Upstream detector", "Downstream detector", "Recorded crash hour [H, H+1)"],
               loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=0.8, w_pad=1.2)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(HERE / f"fig_traffic_cases.{ext}", facecolor=SURFACE, **kw)
    plt.close(fig)


if __name__ == "__main__":
    main()
