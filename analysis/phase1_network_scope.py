"""Reconstruct an auditable Suedbayern motorway corridor from detector metadata."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from shapely import LineString, points
from shapely.strtree import STRtree


ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "autobahn" / "Traffic" / "selection_FG1.csv"
OUT = ROOT / "analysis" / "results" / "phase1"
LAT0 = math.radians(48.9)
EARTH_M = 6_371_008.8
ROUTE_PATTERN = re.compile(r"A\d+(?:-[A-Za-z]+)?_[A-Za-z0-9_]+")


def xy_m(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return EARTH_M * np.radians(lon) * math.cos(LAT0), EARTH_M * np.radians(lat)


def read_detector_metadata() -> pd.DataFrame:
    records = []
    for line_no, line in enumerate(META.read_text(encoding="utf-8-sig").splitlines()[1:], 2):
        fields = line.split(";")
        if len(fields) < 7:
            continue
        site, unit, group = fields[:3]
        route_text = ";".join(fields[3:-3])
        km, lon, lat = (float(value.replace(",", ".")) for value in fields[-3:])
        routes = sorted(set(ROUTE_PATTERN.findall(route_text)))
        for route in routes:
            records.append(
                {
                    "line": line_no,
                    "site": site,
                    "unit": unit,
                    "group": group,
                    "route": route,
                    "km": km,
                    "lon": lon,
                    "lat": lat,
                }
            )
    return pd.DataFrame(records)


def build_edges(detectors: pd.DataFrame, max_km_gap: float = 15.0) -> pd.DataFrame:
    # Lane units at a cross-section share a point. One point per route/site is enough.
    sites = detectors.drop_duplicates(["route", "site"]).copy()
    rows = []
    for route, group in sites.groupby("route"):
        group = group.sort_values("km").reset_index(drop=True)
        x, y = xy_m(group["lon"].to_numpy(), group["lat"].to_numpy())
        for idx in range(len(group) - 1):
            km_gap = group.loc[idx + 1, "km"] - group.loc[idx, "km"]
            straight_m = math.hypot(x[idx + 1] - x[idx], y[idx + 1] - y[idx])
            # Negative kilometre jumps indicate a discontinuity or restart. Long
            # gaps are not bridged by a fictitious straight motorway segment.
            if km_gap < 0 or km_gap > max_km_gap or straight_m > 25_000:
                continue
            rows.append(
                {
                    "route": route,
                    "site_from": group.loc[idx, "site"],
                    "site_to": group.loc[idx + 1, "site"],
                    "km_from": group.loc[idx, "km"],
                    "km_to": group.loc[idx + 1, "km"],
                    "length_straight_m": straight_m,
                    "geometry": LineString([(x[idx], y[idx]), (x[idx + 1], y[idx + 1])]),
                }
            )
    return pd.DataFrame(rows)


def load_bavaria_crashes(year: int) -> pd.DataFrame:
    path = ROOT / "accidents" / f"Unfallorte{year}_standardized.csv"
    data = pd.read_csv(path, sep=";", dtype={"uid": str})
    return data.loc[data["ULAND"] == 9].copy()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    detectors = read_detector_metadata()
    edges = build_edges(detectors)
    if edges.empty:
        raise RuntimeError("No network edges could be constructed")
    tree = STRtree(edges["geometry"].tolist())
    edge_export = edges.drop(columns="geometry").copy()
    edge_export["geometry_wkt"] = edges["geometry"].astype(str)
    edge_export.index.name = "edge_id"
    edge_export.to_csv(OUT / "detector_corridor_edges.csv")
    summary = {
        "metadata_rows": 2379,
        "route_site_memberships": int(len(detectors.drop_duplicates(["route", "site"]))),
        "routes": int(detectors["route"].nunique()),
        "constructed_edges": int(len(edges)),
        "max_km_gap": 15.0,
        "note": (
            "Detector-derived corridor proxy. Final Suedbayern membership must be "
            "checked against authoritative operational road geometry."
        ),
        "years": {},
    }
    edge_routes = edges["route"].to_numpy()
    for year in (2024, 2025):
        crashes = load_bavaria_crashes(year)
        x, y = xy_m(crashes["lon"].to_numpy(), crashes["lat"].to_numpy())
        crash_points = points(x, y)
        edge_idx, distances = tree.query_nearest(
            crash_points, return_distance=True, all_matches=False
        )
        # Vector queries return input indices in row 0 and tree indices in row 1.
        input_idx, tree_idx = edge_idx
        nearest_distance = np.full(len(crashes), np.nan)
        nearest_route = np.full(len(crashes), "", dtype=object)
        nearest_edge = np.full(len(crashes), -1, dtype=int)
        nearest_distance[input_idx] = distances
        nearest_route[input_idx] = edge_routes[tree_idx]
        nearest_edge[input_idx] = tree_idx
        crashes["nearest_corridor_m"] = nearest_distance
        crashes["nearest_route"] = nearest_route
        crashes["nearest_edge"] = nearest_edge
        crashes["site_from"] = [edges.iloc[i]["site_from"] if i >= 0 else "" for i in nearest_edge]
        crashes["site_to"] = [edges.iloc[i]["site_to"] if i >= 0 else "" for i in nearest_edge]
        keep_columns = [
            "year", "uid", "date", "USTUNDE", "UKATEGORIE", "lon", "lat",
            "nearest_route", "nearest_edge", "site_from", "site_to",
            "nearest_corridor_m",
        ]
        crashes[keep_columns].to_csv(
            OUT / f"crash_network_assignment_{year}.csv", index=False
        )
        summary["years"][str(year)] = {
            "bavaria_crashes": int(len(crashes)),
            "within_50m": int((nearest_distance <= 50).sum()),
            "within_100m": int((nearest_distance <= 100).sum()),
            "within_150m": int((nearest_distance <= 150).sum()),
            "within_250m": int((nearest_distance <= 250).sum()),
            "within_500m": int((nearest_distance <= 500).sum()),
        }
    (OUT / "network_scope_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
