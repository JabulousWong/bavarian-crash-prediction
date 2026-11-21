"""Create source-separated edge-hour label tables for downstream modelling."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from shapely import from_wkt, points
from shapely.strtree import STRtree

import dlm_assign


ROOT = Path(__file__).resolve().parents[1]
PHASE1 = ROOT / "analysis" / "results" / "phase1"
OUT = ROOT / "analysis" / "results" / "phase2"
LAT0 = math.radians(48.9)
EARTH_M = 6_371_008.8


def xy_m(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return EARTH_M * np.radians(lon) * math.cos(LAT0), EARTH_M * np.radians(lat)


def assign_nearest(frame: pd.DataFrame, tree: STRtree) -> pd.DataFrame:
    x, y = xy_m(frame["lon"].to_numpy(), frame["lat"].to_numpy())
    pair, distance = tree.query_nearest(points(x, y), return_distance=True, all_matches=False)
    input_idx, edge_idx = pair
    result = frame.copy().reset_index(drop=True)
    result["edge_id"] = -1
    result["corridor_distance_m"] = np.nan
    result.loc[input_idx, "edge_id"] = edge_idx
    result.loc[input_idx, "corridor_distance_m"] = distance
    result["edge_id"] = result["edge_id"].astype(int)
    return result


def official_events(year: int) -> pd.DataFrame:
    assigned = pd.read_csv(PHASE1 / f"crash_network_assignment_{year}.csv", dtype={"uid": str})
    assigned["event_hour"] = pd.to_datetime(assigned["date"]) + pd.to_timedelta(
        assigned["USTUNDE"], unit="h"
    )
    return assigned.rename(
        columns={"nearest_corridor_m": "corridor_distance_m", "nearest_edge": "edge_id"}
    )


def waze_events(year: int, tree: STRtree) -> pd.DataFrame:
    clusters = pd.read_csv(PHASE1 / f"waze_accident_clusters_{year}.csv")
    clusters["first_ts_local"] = pd.to_datetime(clusters["first_ts_local"], utc=True).dt.tz_convert(
        "Europe/Berlin"
    )
    clusters["event_hour"] = clusters["first_ts_local"].dt.tz_localize(None).dt.floor("h")
    # Basis-DLM membership and carriageway (dlm_assign) replace the nearest undirected chord.
    edges = pd.read_csv(PHASE1 / "detector_corridor_edges.csv")
    assigned = dlm_assign.assign(clusters["lon"].to_numpy(), clusters["lat"].to_numpy(), edges)
    result = clusters.reset_index(drop=True).copy()
    result["edge_id"] = assigned["edge_id"].to_numpy()
    result["corridor_distance_m"] = assigned["corridor_distance_m"].to_numpy()
    result["dlm_category"] = assigned["dlm_category"].to_numpy()
    return result


def aggregate_labels(official: pd.DataFrame, waze: pd.DataFrame, threshold: int) -> pd.DataFrame:
    o = official.loc[official["corridor_distance_m"] <= threshold].copy()
    w = waze.loc[waze["corridor_distance_m"] <= threshold].copy()
    official_hour = o.groupby(["edge_id", "event_hour"]).agg(
        official_crashes=("uid", "nunique"),
        worst_injury_category=("UKATEGORIE", "min"),
        official_min_distance_m=("corridor_distance_m", "min"),
    )
    waze_hour = w.groupby(["edge_id", "event_hour"]).agg(
        waze_clusters=("reports", "size"),
        waze_reports=("reports", "sum"),
        waze_max_confidence=("max_confidence", "max"),
        waze_max_reliability=("max_reliability", "max"),
        waze_min_distance_m=("corridor_distance_m", "min"),
    )
    joined = official_hour.join(waze_hour, how="outer").reset_index()
    joined["official_observed"] = joined["official_crashes"].fillna(0).gt(0)
    joined["waze_observed"] = joined["waze_clusters"].fillna(0).gt(0)
    joined["source_pattern"] = np.select(
        [
            joined["official_observed"] & joined["waze_observed"],
            joined["official_observed"],
            joined["waze_observed"],
        ],
        ["official_and_waze_same_edge_hour", "official_only", "waze_only"],
        default="unlabeled",
    )
    joined["network_threshold_m"] = threshold
    return joined


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    edges = pd.read_csv(PHASE1 / "detector_corridor_edges.csv")
    tree = STRtree(from_wkt(edges["geometry_wkt"].to_numpy()).tolist())
    summary: dict[str, dict] = {}
    for year in (2024, 2025):
        official = official_events(year)
        waze = waze_events(year, tree)
        summary[str(year)] = {}
        for threshold in (50, 100, 150):
            labels = aggregate_labels(official, waze, threshold)
            labels.to_parquet(OUT / f"label_events_{year}_{threshold}m.parquet", index=False)
            counts = labels["source_pattern"].value_counts()
            summary[str(year)][f"{threshold}m"] = {
                "event_edge_hours": int(len(labels)),
                "official_crashes": int(
                    official.loc[official["corridor_distance_m"] <= threshold, "uid"].nunique()
                ),
                "waze_clusters": int(
                    (waze["corridor_distance_m"] <= threshold).sum()
                ),
                "source_patterns": {str(k): int(v) for k, v in counts.items()},
            }
    (OUT / "label_cohort_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
