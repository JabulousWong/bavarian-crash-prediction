"""Figure 1 of paper_short.tex: study network only. Run with the analysis venv."""
import sys
from pathlib import Path
ROOT_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_REPO / "viz"))
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from mapbase import ROOT, V2, SURFACE, INK, SEQ, DEEMPH, NET_KM, xy_m, wkt_segments, make_figure, legend_below


def read_sites(path):
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines()[1:]:
        f = line.split(";")
        if len(f) >= 7:
            rows.append({"site": f[0], "lon": float(f[-2].replace(",", ".")), "lat": float(f[-1].replace(",", "."))})
    return pd.DataFrame(rows).drop_duplicates("site")


v2 = pd.read_csv(V2 / "phase1/detector_corridor_edges.csv")
km2 = v2.length_km_post_m.sum() / 2000
sites = read_sites(ROOT / "autobahn/Traffic/selection_FG1.csv")
fig, axes, L = make_figure(1, "south", W=8.0, head=0.08, leg=0.4, foot=0.0)
ax = axes[0]
ax.add_collection(LineCollection(wkt_segments(v2.geometry_wkt), colors=SEQ[4], linewidths=2.2, capstyle="round", zorder=3))
ax.plot(*xy_m(sites.lon.to_numpy(), sites.lat.to_numpy()), "o", ms=1.8, color=INK, mec=SURFACE, mew=0.4, lw=0, zorder=4)
legend_below(ax, [Line2D([], [], color=DEEMPH, lw=1.6), Line2D([], [], color=SEQ[4], lw=2.5),
                  Line2D([], [], marker="o", color=INK, mec=SURFACE, lw=0, ms=5)],
             [f"all Autobahns ({NET_KM:,.0f} km in Bavaria)", f"study network: {len(v2):,} edges, {km2:,.0f} km per direction",
              f"detector cross-sections ({len(sites):,})"], "", ncol=3)
fig.savefig(Path(__file__).parent / "map_network_coverage.png", dpi=300, facecolor=SURFACE)
