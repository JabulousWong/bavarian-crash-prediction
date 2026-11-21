"""Thesis versions of the three maps (legend only; captions live in the thesis), drawn from the Basis-DLM rerun.

map_detector_coverage  traffic detector cross-sections (FG1) and roadside weather stations (FG3) on the Autobahn network
map_network_coverage   every Autobahn in Suedbayern vs. the detector corridors (preliminary v1 and final v2)
map_risk_2025          A: mean predicted hourly risk per edge of the best v2 model, 2025; B: police crashes a top-10 % alert would miss

Context layer (grey Autobahn network, state outline, cities, road labels): viz/mapbase.py. Run with the analysis venv.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from mapbase import (ROOT, V2, OUT, SURFACE, INK, SECONDARY, MUTED, GRID, SEQ, MISS, DEEMPH, CORRIDOR_B, NET_KM, xy_m, wkt_segments,
                     make_figure, legend_below)

EXTENT = "south"
DPI = 300


def read_sites(path):
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines()[1:]:
        f = line.split(";")
        if len(f) < 7:
            continue
        rows.append({"site": f[0], "lon": float(f[-2].replace(",", ".")), "lat": float(f[-1].replace(",", "."))})
    return pd.DataFrame(rows).drop_duplicates("site")


# ------------------------------------------------------------------ detector coverage
traffic = read_sites(ROOT / "autobahn/Traffic/selection_FG1.csv")
weather = read_sites(ROOT / "autobahn/Weather/selection_FG3.csv")
weather_loc = weather.round({"lon": 5, "lat": 5}).drop_duplicates(["lon", "lat"])
fig, axes, L = make_figure(1, EXTENT, W=8.0, head=0.08, leg=0.4, foot=0.0)
ax = axes[0]
ax.plot(*xy_m(traffic.lon.to_numpy(), traffic.lat.to_numpy()), "o", ms=2.0, color=SEQ[4], mec="none", lw=0, zorder=4)
ax.plot(*xy_m(weather_loc.lon.to_numpy(), weather_loc.lat.to_numpy()), "^", ms=3.8, mfc="none", mec=MISS[1], mew=0.45, lw=0, zorder=3)
legend_below(ax, [Line2D([], [], color=DEEMPH, lw=1.6), Line2D([], [], marker="o", color=SEQ[4], mec="none", lw=0, ms=5),
                  Line2D([], [], marker="^", mfc="none", mec=MISS[1], mew=1.0, lw=0, ms=7)],
             [f"Autobahn network ({NET_KM:,.0f} km in Bavaria)", f"traffic detector cross-sections ({len(traffic):,})",
              f"weather station locations ({len(weather_loc):,})"], "", ncol=3)
fig.savefig(OUT / "map_detector_coverage.png", dpi=DPI, facecolor=SURFACE)
print(f"detector coverage: traffic sites {len(traffic)} | weather sites {len(weather)} at {len(weather_loc)} locations")

# ------------------------------------------------------------------ network coverage
v1 = pd.read_csv(ROOT / "analysis/results/phase1/detector_corridor_edges.csv")
v2 = pd.read_csv(V2 / "phase1/detector_corridor_edges.csv")
km1 = v1.length_straight_m.sum() / 2000
km2 = v2.length_km_post_m.sum() / 2000
sites = read_sites(ROOT / "autobahn/Traffic/selection_FG1.csv")
fig, axes, L = make_figure(1, EXTENT, W=8.0, head=0.08, leg=0.55, foot=0.0)
ax = axes[0]
ax.add_collection(LineCollection(wkt_segments(v1.geometry_wkt), colors=SEQ[1], linewidths=5.0, capstyle="round", zorder=2))
ax.add_collection(LineCollection(wkt_segments(v2.geometry_wkt), colors=SEQ[4], linewidths=2.0, capstyle="round", zorder=3))
ax.plot(*xy_m(sites.lon.to_numpy(), sites.lat.to_numpy()), "o", ms=1.8, color=INK, mec=SURFACE, mew=0.4, lw=0, zorder=4)
handles = [Line2D([], [], color=DEEMPH, lw=1.6), Line2D([], [], color=SEQ[1], lw=6), Line2D([], [], color=SEQ[4], lw=2.5),
           Line2D([], [], marker="o", color=INK, mec=SURFACE, lw=0, ms=5)]
legend_below(ax, handles, [f"all Autobahns (Basis-DLM mainline axes, {NET_KM:,.0f} km in Bavaria)",
                           f"preliminary network: {len(v1):,} edges, {km1:,.0f} km per direction",
                           f"study network: {len(v2):,} edges, {km2:,.0f} km per direction",
                           f"detector cross-sections ({len(sites):,})"], "", ncol=2)
fig.savefig(OUT / "map_network_coverage.png", dpi=DPI, facecolor=SURFACE)
print(f"network coverage: v1 {len(v1)} edges, {km1:.0f} km/direction | v2 {len(v2)} edges, {km2:.0f} km/direction | sites {len(sites)}")

# ------------------------------------------------------------------ risk map 2025
MODEL = "mswspu__context+frequency+traffic"
edges = pd.read_csv(V2 / "phase1/detector_corridor_edges.csv")
segs = wkt_segments(edges.geometry_wkt)
pred = pd.read_parquet(V2 / "phase7/two_stage_predictions_2025.parquet")
marg = pred[pred.kind == "marginal"]
cases = pred[pred.kind == "official_case"].copy()
crashes = pd.read_csv(V2 / "phase1/crash_network_assignment_2025.csv", dtype={"uid": str})
crashes = crashes[crashes.nearest_corridor_m <= 100]

hourly = lambda p: 1 - (1 - p) ** 12
edge_risk = marg.groupby("edge_id")[MODEL].mean().pipe(hourly)
network_mean = hourly(marg[MODEL]).mean()
ratio = (edge_risk / network_mean).reindex(edges.edge_id).to_numpy()
bins = [0, 0.5, 1, 2, 4, np.inf]
labels = ["< 0.5×", "0.5–1×", "1–2×", "2–4×", "> 4×"]
klass = np.digitize(ratio, bins[1:-1])
colors_a = np.array([SEQ[1], SEQ[2], SEQ[3], SEQ[5], SEQ[6]])[klass]

score_sorted = np.sort(marg[MODEL].to_numpy())
cases["pct"] = np.searchsorted(score_sorted, cases[MODEL].to_numpy()) / len(score_sorted)
cases["missed"] = cases.pct < 0.9
cases["clear_miss"] = cases.pct < 0.5
SECTION_KM, MIN_N = 12, 10
e = edges.assign(road=edges.route.str.split("_").str[0], km_mid=(edges.km_from + edges.km_to) / 2)
e["section"] = ((e.km_mid - e.groupby("road").km_mid.transform("min")) // SECTION_KM).astype(int)
ext = e.groupby(["road", "section"]).agg(lo=("km_from", "min"), hi=("km_to", "max"), last=("section", "max"))
ext["last"] = ext.index.get_level_values("section") == ext.groupby(level="road")["last"].transform("max")
short_tail = ext[(ext.hi - ext.lo < SECTION_KM / 2) & ext["last"] & (ext.index.get_level_values("section") > 0)].index
e.loc[pd.MultiIndex.from_frame(e[["road", "section"]]).isin(short_tail), "section"] -= 1
sec = cases.join(e.set_index("edge_id")[["road", "section"]], on="edge_id").groupby(["road", "section"]).missed.agg(n="size", missed="sum")
sec["share"] = sec.missed / sec.n
e = e.join(sec, on=["road", "section"])
n_sec = e.n.fillna(0).to_numpy(); share_sec = e.share.to_numpy()
bins_b = [0, 0.4, 0.6, 0.8, 1.01]; labels_b = ["< 40%", "40–60%", "60–80%", "≥ 80%"]
kb = np.digitize(np.nan_to_num(share_sec, nan=0), bins_b[1:-1])
colors_b = np.where(n_sec >= MIN_N, np.array(MISS)[kb], CORRIDOR_B)
widths_b = np.where(n_sec >= MIN_N, 2.6, 1.5)
alert = np.linspace(0, 1, 201); missed_curve = np.array([(cases.pct < 1 - a).mean() for a in alert])
reg = crashes.assign(hour_start=pd.to_datetime(crashes.date) + pd.to_timedelta(crashes.USTUNDE, unit="h"))
cases["t_end"] = pd.to_datetime(cases.t_end)
cases = cases.merge(reg[["nearest_edge", "hour_start", "lon", "lat"]].rename(columns={"nearest_edge": "edge_id", "hour_start": "t_end"}),
                    on=["edge_id", "t_end"], how="left").drop_duplicates("row_id")
clear = cases[cases.clear_miss & cases.lon.notna()]

# panel A: one full-width figure (captions live in the thesis)
fig, axes, L = make_figure(1, EXTENT, W=8.0, head=0.08, leg=0.62, foot=0.0)
ax = axes[0]
ax.add_collection(LineCollection(segs, colors=colors_a, linewidths=2.6, capstyle="round", zorder=2))
ax.plot(*xy_m(crashes.lon.to_numpy(), crashes.lat.to_numpy()), "o", ms=1.9, color=INK, mec=SURFACE, mew=0.4, lw=0, zorder=3)
handles = [Line2D([], [], color=c, lw=4) for c in [SEQ[1], SEQ[2], SEQ[3], SEQ[5], SEQ[6]]] + \
          [Line2D([], [], marker="o", color=INK, mec=SURFACE, lw=0, ms=5), Line2D([], [], color=DEEMPH, lw=1.6)]
legend_below(ax, handles, labels + [f"police injury crash 2025 ({len(crashes):,})", "other Autobahn, not modelled"],
             "mean predicted hourly risk of the edge / network mean", ncol=4)
fig.savefig(OUT / "map_risk_2025_a.png", dpi=DPI, facecolor=SURFACE)

# panel B: second full-width figure with the alert-coverage inset (three legend rows)
fig, axes, L = make_figure(1, EXTENT, W=8.0, head=0.08, leg=0.95, foot=0.0)
ax = axes[0]
ax.add_collection(LineCollection(segs, colors=colors_b, linewidths=widths_b, capstyle="round", zorder=2))
ax.plot(*xy_m(clear.lon.to_numpy(), clear.lat.to_numpy()), "o", ms=3.2, color=INK, mec=SURFACE, mew=0.6, lw=0, zorder=4)
handles = [Line2D([], [], color=c, lw=4) for c in MISS] + \
          [Line2D([], [], marker="o", color=INK, mec=SURFACE, lw=0, ms=5.5), Line2D([], [], color=CORRIDOR_B, lw=2.2), Line2D([], [], color=DEEMPH, lw=1.6)]
legend_below(ax, handles, labels_b + [f"crash hour scored in the bottom half of all units ({len(clear)})", f"fewer than {MIN_N} crashes in the section",
                                      "other Autobahn, not modelled"],
             f"share of the section's 2025 crashes missed by a top-10 % alert (all sections: {cases.missed.mean():.0%})", ncol=3)
ins = ax.inset_axes([0.75, 0.04, 0.24, 0.20]); ins.set_facecolor(SURFACE)
ins.plot([0, 1], [1, 0], color=GRID, lw=0.8, zorder=1); ins.plot(alert, missed_curve, color=MISS[2], lw=1.6, zorder=2)
for a in (0.1, 0.3):
    m_ = float((cases.pct < 1 - a).mean()); ins.plot(a, m_, "o", ms=3, color=MISS[3], zorder=3)
    ins.annotate(f"{a:.0%} alerted: {m_:.0%} missed", (a, m_), xytext=(4, 3), textcoords="offset points", fontsize=6, color=SECONDARY, va="bottom")
ins.set_xlim(0, 1); ins.set_ylim(0, 1); ins.set_xticks([0, 0.5, 1]); ins.set_yticks([0, 0.5, 1])
ins.set_xticklabels(["0", "50%", "100%"]); ins.set_yticklabels(["0", "50%", "100%"])
ins.tick_params(labelsize=6, colors=MUTED, length=2, pad=1.5)
for side, sp in ins.spines.items():
    sp.set_visible(side in ("left", "bottom")); sp.set_color(GRID)
ins.set_xlabel("share of all units alerted", fontsize=6.2, color=SECONDARY, labelpad=1.5)
ins.set_ylabel("share of crashes missed", fontsize=6.2, color=SECONDARY, labelpad=1.5)
ins.text(0.97, 0.97, "random ranking = diagonal", transform=ins.transAxes, fontsize=5.8, color=MUTED, va="top", ha="right")
fig.savefig(OUT / "map_risk_2025_b.png", dpi=DPI, facecolor=SURFACE)
print("risk map: edges", len(edges), "| class counts A:", np.bincount(klass, minlength=5).tolist())
sections = sec[sec.n >= MIN_N]
print("sections:", len(sec), "| with >=", MIN_N, "crashes:", len(sections), "| missed-share classes:", np.bincount(np.digitize(sections.share, bins_b[1:-1]), minlength=4).tolist(), labels_b)
print("crashes missed when alerting on top 10/20/30/50% of units:", [f"{(cases.pct < 1 - a).mean():.3f}" for a in (0.1, 0.2, 0.3, 0.5)])
print("clear misses:", len(clear), "of", len(cases), "| crashes on map:", len(crashes), "| network mean hourly risk:", f"{network_mean:.3e}")
print("edge risk ratio quantiles:", np.nanquantile(ratio, [0.1, 0.5, 0.9]).round(2).tolist(), "| max", np.nanmax(ratio).round(1))
