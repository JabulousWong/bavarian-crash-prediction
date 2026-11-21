"""Shared pieces for the viz maps: palette, projection, Basis-DLM context layer, cities, road labels, figure layout.

Context layer (drawn underneath every map, not used by any model):
- viz/basemap_bavaria.json: every Bavarian Autobahn mainline axis and the state outline from Basis-DLM (LDBV Bayern), WGS84.
  Regenerate with the system python3 (needs geopandas): python3 viz/export_basemap.py
"""
import json, math, re, textwrap
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patheffects as pe

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "analysis/results/v2"
OUT = ROOT / "thesis" / "image"
LAT0, EARTH = math.radians(48.9), 6_371_008.8   # equirectangular projection used by phase1 for the corridor edges

# palette (reference instance, light mode)
SURFACE, INK, SECONDARY, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # blue steps 100..700
MISS = ["#eda084", "#d95b28", "#9e3400", "#5e1a00"]   # orange ordinal ramp (second sequential context), light -> dark = more crashes missed
DEEMPH, CORRIDOR_B, STATE_FILL, STATE_EDGE = "#c3c2b7", "#a5a399", "#f5f4ef", "#cfcec5"

# map windows: "south" = bbox of Oberbayern, Niederbayern and Schwaben plus margin; "bavaria" = the whole state
EXTENTS = {"south": ((9.5, 13.95), (47.2, 49.3)), "bavaria": ((8.85, 13.95), (47.2, 50.62))}
CITIES = {"München": (11.576, 48.137), "Nürnberg": (11.077, 49.454), "Augsburg": (10.898, 48.371), "Regensburg": (12.102, 49.013),
          "Ingolstadt": (11.425, 48.766), "Passau": (13.46, 48.574), "Rosenheim": (12.129, 47.856), "Landshut": (12.15, 48.537),
          "Deggendorf": (12.96, 48.84), "Kempten": (10.317, 47.726), "Lindau": (9.69, 47.55), "Würzburg": (9.93, 49.79), "Hof": (11.92, 50.32),
          "Garmisch": (11.095, 47.492), "Bayreuth": (11.578, 49.948), "Bamberg": (10.89, 49.892), "Schweinfurt": (10.23, 50.05),
          "Aschaffenburg": (9.15, 49.977), "Neu-Ulm": (10.01, 48.39)}
# road-number labels: (lon, lat) anchors, each snapped to the nearest vertex of that road
ROAD_LABELS = {"A3": [(9.4, 49.95), (10.5, 49.6), (12.55, 48.95), (13.1, 48.68)], "A6": [(10.4, 49.25), (12.1, 49.5)], "A7": [(10.0, 50.4), (10.1, 49.3), (10.2, 48.1)],
               "A8": [(10.5, 48.4), (12.5, 47.85)], "A9": [(11.85, 50.2), (11.75, 49.75), (11.5, 49.1)], "A70": [(10.6, 49.95)], "A71": [(10.25, 50.3)],
               "A73": [(11.0, 49.75), (10.95, 50.2)], "A92": [(12.55, 48.68)], "A93": [(12.1, 50.1), (12.05, 49.3), (12.19, 47.68)],
               "A94": [(12.55, 48.25)], "A95": [(11.3, 47.85)], "A96": [(10.6, 48.07)], "A99": [(11.42, 48.19)]}


def xy_m(lon, lat):
    """lon/lat -> km in the phase1 equirectangular frame."""
    return EARTH * np.radians(lon) * math.cos(LAT0) / 1000, EARTH * np.radians(lat) / 1000


def wkt_segments(wkts):
    """LINESTRING WKT in local metres -> list of (N x 2) km arrays."""
    return [np.array([[float(a) for a in pt.split()] for pt in re.findall(r"[-\d.]+ [-\d.]+", w)]) / 1000 for w in wkts]


BASE = json.load(open(Path(__file__).parent / "basemap_bavaria.json"))
NET = {r["road"]: [np.column_stack(xy_m(np.array(ln)[:, 0], np.array(ln)[:, 1])) for ln in r["lines"]] for r in BASE["roads"]}
NET_SEGS = [ln for lines in NET.values() for ln in lines]
STATE_RINGS = [np.column_stack(xy_m(np.array(rg)[:, 0], np.array(rg)[:, 1])) for rg in BASE["bavaria"]]
NET_KM = sum(r["km"] for r in BASE["roads"])


def make_figure(n_panels, extent="south", W=14.0, sub_lines=4, head=1.0, leg=0.62, foot=0.42):
    """Figure with n map panels side by side; each panel gets the state outline, the grey Autobahn network, labels, cities and a scale bar.
    Vertical layout in inches: title + subtitle block (head), the maps, a legend strip under each map (leg), footnote (foot)."""
    LON, LAT = EXTENTS[extent]
    xlim = xy_m(np.array(LON), np.array([0, 0]))[0]; ylim = xy_m(np.array([0, 0]), np.array(LAT))[1]
    dx, dy = xlim[1] - xlim[0], ylim[1] - ylim[0]
    slot_w = (0.485 if n_panels == 2 else 0.98) * W; slot_h = slot_w * dy / dx; H = slot_h + head + leg + foot
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 9, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, n_panels, figsize=(W, H), facecolor=SURFACE, squeeze=False); axes = axes[0]
    fig.subplots_adjust(left=0.01, right=0.99, top=1 - head / H, bottom=(leg + foot) / H, wspace=0.03)
    for ax in axes:
        draw_context(ax, xlim, ylim)
    return fig, axes, {"xlim": xlim, "ylim": ylim, "sub_lines": sub_lines}


def draw_context(ax, xlim, ylim):
    ax.set_facecolor(SURFACE); ax.set_aspect("equal"); ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    for s_ in ax.spines.values(): s_.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    for ring in STATE_RINGS:
        ax.fill(ring[:, 0], ring[:, 1], fc=STATE_FILL, ec=STATE_EDGE, lw=0.7, zorder=0)
    ax.add_collection(LineCollection(NET_SEGS, colors=DEEMPH, linewidths=1.2, capstyle="round", zorder=1))
    for road, anchors in ROAD_LABELS.items():
        pts = np.vstack(NET[road])
        for lo, la in anchors:
            ax_, ay_ = xy_m(np.array([lo]), np.array([la]))
            x, y = pts[np.argmin(np.hypot(pts[:, 0] - ax_[0], pts[:, 1] - ay_[0]))]
            if xlim[0] < x < xlim[1] and ylim[0] < y < ylim[1]:
                ax.text(x, y, road, fontsize=5.8, color=SECONDARY, ha="center", va="center", zorder=5,
                        bbox=dict(boxstyle="round,pad=0.22", fc=SURFACE, ec="#b5b3aa", lw=0.5))
    for name, (lo, la) in CITIES.items():
        x, y = xy_m(np.array([lo]), np.array([la]))
        if xlim[0] < x[0] < xlim[1] and ylim[0] < y[0] < ylim[1]:
            ax.plot(x, y, "o", ms=2.6, color=MUTED, mec=SURFACE, mew=0.8, zorder=1.5)
            ax.annotate(name, (x[0], y[0]), xytext=(3, 3), textcoords="offset points", color=MUTED, fontsize=7, zorder=6,
                        path_effects=[pe.withStroke(linewidth=2.2, foreground=SURFACE)])
    x0, y0 = xlim[0] + 12, ylim[0] + 7
    ax.plot([x0, x0 + 50], [y0, y0], color=MUTED, lw=1.2); ax.text(x0 + 25, y0 + 3, "50 km", ha="center", color=MUTED, fontsize=7.5)


def header(ax, title, subtitle, sub_lines=4):
    """Bold title plus a wrapped subtitle; every panel reserves sub_lines subtitle lines so titles sit on one line."""
    wrapped = textwrap.fill(subtitle, 92)
    assert wrapped.count("\n") + 1 <= sub_lines, wrapped
    ax.set_title(title, loc="left", color=INK, fontsize=11.5, fontweight="bold", pad=10 + 11.5 * (sub_lines - 1) + 11.5)
    ax.annotate(wrapped, xy=(0, 1), xycoords="axes fraction", xytext=(0, 5 + 10.8 * sub_lines), textcoords="offset points",
                color=SECONDARY, fontsize=8, va="top", linespacing=1.35)


def legend_below(ax, handles, labels, title, ncol):
    leg = ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(0, -0.012), ncol=ncol, frameon=False, fontsize=8, title=title,
                    title_fontsize=8, labelcolor=SECONDARY, columnspacing=1.6, handlelength=2.2, borderaxespad=0)
    leg._legend_box.align = "left"


def footnote(fig, text):
    fig.text(0.01, 0.012, text, color=MUTED, fontsize=7.5, linespacing=1.4)
