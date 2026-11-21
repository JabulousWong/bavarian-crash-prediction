"""System python3 (geopandas/pyproj): export Basis-DLM road axes for the analysis venv.

Writes WGS84 WKB parquet files so the venv (shapely only) can classify points:
  road_axes.parquet          L_42003_Strassenachse, all road classes: class, function, road number
  bab_carriageways.parquet   L_42005_Fahrbahnachse, Widmung = Bundesautobahn: one-way carriageway axes,
                             digitised in the direction of travel
"""
import sys, time, warnings
from pathlib import Path
import pyogrio, shapely
import pyarrow as pa, pyarrow.parquet as pq
warnings.filterwarnings("ignore")
# usage: python3 dlm/export_dlm.py <path/to/by_basisdlm_plus.gpkg>
SRC = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[1] / "data" / "by_basisdlm_plus.gpkg")
OUT = str(Path(__file__).resolve().parent)
t0 = time.time()

def road(series):
    return series.fillna("").str.split(",").str[0].str.strip()

axes = pyogrio.read_dataframe(SRC, layer="L_42003_Strassenachse", columns=["WDM_widmung", "FKT_funktion", "BEZ_bezeichnung"]).to_crs(4326)
pq.write_table(pa.table({"widmung": axes.WDM_widmung.fillna("?"), "funktion": axes.FKT_funktion.fillna("?"),
                         "road": road(axes.BEZ_bezeichnung), "wkb": shapely.to_wkb(axes.geometry.values)}),
               f"{OUT}/road_axes.parquet", compression="zstd")
cw = pyogrio.read_dataframe(SRC, layer="L_42005_Fahrbahnachse", where="WDM_widmung = 'Bundesautobahn'",
                            columns=["BEZ_bezeichnung", "FAR_fahrtrichtung", "ZUS_zustand"]).to_crs(4326)
assert (cw.FAR_fahrtrichtung.str.startswith("Einbahn")).all()
pq.write_table(pa.table({"road": road(cw.BEZ_bezeichnung), "zustand": cw.ZUS_zustand.fillna("?"),
                         "wkb": shapely.to_wkb(cw.geometry.values)}), f"{OUT}/bab_carriageways.parquet", compression="zstd")
print(f"road axes {len(axes)}, BAB carriageways {len(cw)}, {time.time()-t0:.0f}s")
