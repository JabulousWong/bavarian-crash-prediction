"""Thesis map: BASt permanent counting stations vs the Autobahn GmbH detector cross-sections (legend only; caption lives in the thesis).

BASt station list: Jawe2024 (WGS84 coordinates); hourly completeness from the 2024 Autobahn hourly file (2024_A_S.zip),
an hour counts as valid when both directions carry the flag '-' or 'u'. Colour-blind-safe pair: blue #2a78d6 / orange #eb6834,
with different marker shapes. Run with the system python3.
"""
import sys, io, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from matplotlib.lines import Line2D
from mapbase import ROOT, V2, OUT, SURFACE, DEEMPH, EXTENTS, NET_KM, xy_m, wkt_segments, make_figure, legend_below

BLUE, ORANGE = "#2a78d6", "#eb6834"
BAST = ROOT / "bast"  # BASt hourly counts: Jawe2024_stations.csv, 2024_A_S.zip
DPI = 300


def read_sites(path):
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines()[1:]:
        f = line.split(";")
        if len(f) < 7:
            continue
        rows.append({"site": f[0], "lon": float(f[-2].replace(",", ".")), "lat": float(f[-1].replace(",", "."))})
    return pd.DataFrame(rows).drop_duplicates("site")


traffic = read_sites(ROOT / "autobahn/Traffic/selection_FG1.csv")
st = pd.read_csv(BAST / "Jawe2024_stations.csv", sep=";", encoding="latin1", decimal=",")
st = st[(st.Land_Code == "BY") & (st.Str_Kl == "A")]
(x0, x1), (y0, y1) = EXTENTS["south"]
st = st[st.Koor_WGS84_E.between(x0, x1) & st.Koor_WGS84_N.between(y0, y1)].copy()

# 2024 hourly completeness per Bavarian station
p = subprocess.Popen(["unzip", "-p", str(BAST / "2024_A_S.zip")], stdout=subprocess.PIPE)
rows = []
for i, line in enumerate(io.TextIOWrapper(p.stdout, encoding="latin1")):
    if i == 0:
        continue
    f = line.split(";")
    if f[2].strip() == "09":
        rows.append((int(f[1]), f[10].strip() in ("-", "u") and f[12].strip() in ("-", "u")))
valid = pd.DataFrame(rows, columns=["zst", "ok"]).groupby("zst").ok.mean()
st["valid"] = st.DZ_Nr.map(valid).fillna(0.0)
good = st.valid >= 0.9

# statistics quoted in the thesis text
tx = 1000 * np.column_stack(xy_m(traffic.lon.to_numpy(), traffic.lat.to_numpy()))
bx = 1000 * np.column_stack(xy_m(st.Koor_WGS84_E.to_numpy(), st.Koor_WGS84_N.to_numpy()))
d_ag, _ = cKDTree(tx).query(bx)
e = pd.read_csv(V2 / "phase1/detector_corridor_edges.csv")
pts = np.vstack([np.linspace(s[i], s[i + 1], max(2, int(np.linalg.norm(s[i + 1] - s[i]) / 20)))
                 for s in (w * 1000 for w in wkt_segments(e.geometry_wkt)) for i in range(len(s) - 1)])
on_net = cKDTree(pts).query(bx)[0] <= 300
km = e.length_km_post_m.sum() / 2000
print(f"BASt stations in window {len(st)} | <=250 m from an Autobahn GmbH cross-section {(d_ag <= 250).sum()} (median {np.median(d_ag):.0f} m)")
print(f"on study network {on_net.sum()} ({km:.0f} km/direction) | of these >=90 % valid hours 2024: {(good & on_net).sum()}")
print(f"Autobahn GmbH cross-sections {len(traffic)} | window stations with >=90 % valid hours {good.sum()}")

fig, axes, L = make_figure(1, "south", W=8.0, head=0.08, leg=0.55, foot=0.0)
ax = axes[0]
ax.plot(*xy_m(traffic.lon.to_numpy(), traffic.lat.to_numpy()), "o", ms=1.8, color=BLUE, mec="none", lw=0, zorder=6)
g, b = st[good], st[~good]
ax.plot(*xy_m(g.Koor_WGS84_E.to_numpy(), g.Koor_WGS84_N.to_numpy()), "s", ms=4.6, color=ORANGE, mec="#3d3c38", mew=0.45, lw=0, zorder=5)
ax.plot(*xy_m(b.Koor_WGS84_E.to_numpy(), b.Koor_WGS84_N.to_numpy()), "s", ms=4.6, mfc=SURFACE, mec=ORANGE, mew=1.0, lw=0, zorder=4)
handles = [Line2D([], [], color=DEEMPH, lw=1.6),
           Line2D([], [], marker="o", color=BLUE, mec="none", lw=0, ms=5),
           Line2D([], [], marker="s", color=ORANGE, mec="#3d3c38", mew=0.6, lw=0, ms=7),
           Line2D([], [], marker="s", mfc=SURFACE, mec=ORANGE, mew=1.2, lw=0, ms=7)]
legend_below(ax, handles, [f"Autobahn network ({NET_KM:,.0f} km in Bavaria)",
                           f"Autobahn GmbH cross-sections, 1 min, lane-resolved ({len(traffic):,})",
                           f"BASt counting stations, hourly counts, ≥ 90 % valid hours in 2024 ({good.sum()})",
                           f"BASt counting stations, < 90 % valid or no data in 2024 ({(~good).sum()})"], "", ncol=2)
fig.savefig(OUT / "map_bast_vs_detectors.png", dpi=DPI, facecolor=SURFACE)
