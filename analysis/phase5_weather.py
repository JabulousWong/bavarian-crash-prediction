"""Attach roadside weather (TLS FG3) to the matched edge-hour datasets.

Six quantities follow the thesis selection: NI, WFD, FBT, TT1, SW (hourly mean)
and FBZ (hourly maximum; code 255 = "cannot determine" is treated as missing).
The as-of rule is the traffic rule: outcome in [h,h+1), weather from [h-1,h).

Spatial assignment, per edge and quantity: the edge midpoint borrows the nearest
sensor on the same motorway within 10 km, otherwise the nearest sensor overall,
with a 30 km hard cap. Distances are great-circle, not kilometre-post
differences. The second-ranked sensor fills hours the first one did not report.
Phase 2 artifacts are read, never overwritten.
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl


ROOT = Path(__file__).resolve().parents[1]
PHASE1 = ROOT / "analysis" / "results" / "phase1"
PHASE2_DATA = ROOT / "analysis" / "results" / "phase2" / "datasets"
CACHE = ROOT / "analysis" / "cache" / "weather_hourly"
OUT = ROOT / "analysis" / "results" / "phase5"
VARIABLES = {"ni": "mean", "wfd": "mean", "fbz": "max", "fbt": "mean", "tt1": "mean", "sw": "mean"}
SAME_ROAD_CAP_M = 10_000
HARD_CAP_M = 30_000
EARTH_M = 6_371_008.8


def read_selection(path: Path) -> pd.DataFrame:
    """Parse a selection_FG*.csv; junction rows carry extra route fields."""
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines()[1:]:
        fields = line.split(";")
        if len(fields) < 7:
            continue
        rows.append(
            {
                "site": fields[0],
                "unit": fields[1],
                "motorway": fields[3].split("_")[0],
                "lon": float(fields[-2].replace(",", ".")),
                "lat": float(fields[-1].replace(",", ".")),
            }
        )
    return pd.DataFrame(rows).drop_duplicates("unit")


def source_files(year: int) -> dict[str, Path]:
    folder = ROOT / "autobahn" / ("Weather_2024" if year == 2024 else "Weather")
    pattern = re.compile(rf"^FG3_(.+?)(_agg1min)?_{year}-01-01_bis_.+\.csv$")
    result = {}
    for path in folder.glob("FG3_*.csv"):
        match = pattern.match(path.name)
        if match:
            result[match.group(1)] = path
    return result


def file_variable(path: Path) -> str:
    with path.open(encoding="utf-8-sig") as handle:
        return handle.readline().strip().split(";")[-1].lower()


def aggregate_one(task: tuple[str, str, str, int]) -> pl.DataFrame | None:
    unit, variable, source_text, year = task
    data = pl.read_csv(
        source_text,
        separator=";",
        null_values=["null", ""],
        try_parse_dates=True,
        schema_overrides={"t_start": pl.Datetime},
    )
    if data.height == 0:
        return None
    data = data.rename({data.columns[1]: "value"}).with_columns(
        pl.col("t_start").cast(pl.Datetime("us")),
        pl.col("value").cast(pl.Float64, strict=False),
    )
    data = data.filter(pl.col("t_start").dt.year() == year)
    if variable == "fbz":
        data = data.with_columns(
            pl.when(pl.col("value") == 255).then(None).otherwise(pl.col("value")).alias("value")
        )
    data = data.drop_nulls("value")
    if data.height == 0:
        return None
    reducer = pl.col("value").max() if VARIABLES[variable] == "max" else pl.col("value").mean()
    return (
        data.group_by(pl.col("t_start").dt.truncate("1h").alias("feature_hour"))
        .agg(reducer.alias("value"), pl.len().alias("n_records"))
        .with_columns(pl.lit(unit).alias("unit"), pl.lit(variable).alias("variable"))
    )


def sensor_hours(year: int, workers: int) -> pd.DataFrame:
    output = CACHE / f"{year}.parquet"
    if output.exists():
        return pd.read_parquet(output)
    tasks = []
    for unit, path in sorted(source_files(year).items()):
        variable = file_variable(path)
        if variable in VARIABLES:
            tasks.append((unit, variable, str(path), year))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = [part for part in pool.map(aggregate_one, tasks, chunksize=8) if part is not None]
    hourly = pl.concat(parts).sort(["variable", "unit", "feature_hour"])
    output.parent.mkdir(parents=True, exist_ok=True)
    hourly.write_parquet(output, compression="zstd")
    return hourly.to_pandas()


def haversine_m(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_M * np.arcsin(np.sqrt(a))


def edge_midpoints() -> pd.DataFrame:
    edges = pd.read_csv(PHASE1 / "detector_corridor_edges.csv")[["edge_id", "route", "site_from", "site_to"]]
    sites = read_selection(ROOT / "autobahn" / "Traffic" / "selection_FG1.csv").drop_duplicates("site")
    sites = sites.set_index("site")[["lon", "lat"]]
    edges = edges.join(sites.add_prefix("from_"), on="site_from").join(sites.add_prefix("to_"), on="site_to")
    edges["lon"] = edges[["from_lon", "to_lon"]].mean(axis=1)
    edges["lat"] = edges[["from_lat", "to_lat"]].mean(axis=1)
    edges["motorway"] = edges["route"].str.split("_").str[0]
    return edges[["edge_id", "motorway", "lon", "lat"]]


def assign_sensors(hourly: pd.DataFrame) -> pd.DataFrame:
    """Rank up to two sensors per edge and quantity among sensors with data."""
    edges = edge_midpoints()
    sensors = read_selection(ROOT / "autobahn" / "Weather" / "selection_FG3.csv")
    available = hourly[["unit", "variable"]].drop_duplicates().merge(sensors, on="unit")
    rows = []
    for variable, group in available.groupby("variable"):
        group = group.reset_index(drop=True)
        distance = haversine_m(
            edges["lon"].to_numpy()[:, None], edges["lat"].to_numpy()[:, None],
            group["lon"].to_numpy()[None, :], group["lat"].to_numpy()[None, :],
        )
        same_road = edges["motorway"].to_numpy()[:, None] == group["motorway"].to_numpy()[None, :]
        preferred = same_road & (distance <= SAME_ROAD_CAP_M)
        # Preferred sensors sort first; everything beyond the hard cap is dropped.
        order_key = np.where(preferred, distance, distance + HARD_CAP_M)
        for position, edge_id in enumerate(edges["edge_id"]):
            if np.isnan(distance[position]).all():
                continue
            ranked = np.argsort(order_key[position])[:2]
            for rank, index in enumerate(ranked, 1):
                if not distance[position, index] <= HARD_CAP_M:
                    continue
                rows.append(
                    {
                        "edge_id": int(edge_id),
                        "variable": variable,
                        "rank": rank,
                        "unit": group.loc[index, "unit"],
                        "distance_m": float(distance[position, index]),
                        "same_motorway_within_cap": bool(preferred[position, index]),
                    }
                )
    return pd.DataFrame(rows)


def attach_weather(rows: pd.DataFrame, hourly: pd.DataFrame, assignment: pd.DataFrame) -> pd.DataFrame:
    """Add wx_<variable> columns for each row's (edge_id, feature_hour)."""
    result = rows.copy()
    key_hour = pd.to_datetime(result["feature_hour"]).astype("datetime64[us]")
    for variable in VARIABLES:
        series = hourly.loc[hourly["variable"] == variable, ["unit", "feature_hour", "value"]].copy()
        series["feature_hour"] = series["feature_hour"].astype("datetime64[us]")
        value = pd.Series(np.nan, index=result.index)
        for rank in (1, 2):
            chosen = assignment.loc[
                (assignment["variable"] == variable) & (assignment["rank"] == rank), ["edge_id", "unit"]
            ]
            keys = pd.DataFrame({"edge_id": result["edge_id"].to_numpy(), "feature_hour": key_hour.to_numpy()})
            merged = keys.merge(chosen, on="edge_id", how="left").merge(
                series, on=["unit", "feature_hour"], how="left"
            )
            value = value.fillna(pd.Series(merged["value"].to_numpy(), index=result.index))
        result[f"wx_{variable}"] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    (OUT / "weather").mkdir(parents=True, exist_ok=True)
    (OUT / "datasets").mkdir(parents=True, exist_ok=True)
    summary = {}
    for year in (2024, 2025):
        hourly = sensor_hours(year, args.workers)
        assignment = assign_sensors(hourly)
        assignment.to_csv(OUT / "weather" / f"sensor_assignment_{year}.csv", index=False)
        first = assignment.loc[assignment["rank"] == 1]
        summary[str(year)] = {
            "sensors_with_data": hourly.groupby("variable")["unit"].nunique().to_dict(),
            "edges_assigned": first.groupby("variable")["edge_id"].nunique().to_dict(),
            "assignment_distance_m_quantiles_50_90_max": {
                variable: [float(group["distance_m"].quantile(q)) for q in (0.5, 0.9, 1.0)]
                for variable, group in first.groupby("variable")
            },
            "share_same_motorway_within_10km": float(first["same_motorway_within_cap"].mean()),
            "datasets": {},
        }
        for suffix in ("", "_union"):
            name = f"matched_edge_hours_{year}_100m{suffix}"
            dataset = attach_weather(pd.read_parquet(PHASE2_DATA / f"{name}.parquet"), hourly, assignment)
            dataset.to_parquet(OUT / "datasets" / f"{name}_wx.parquet", index=False)
            columns = [f"wx_{variable}" for variable in VARIABLES]
            summary[str(year)]["datasets"][name] = {
                "rows": int(len(dataset)),
                "non_missing": {column: float(dataset[column].notna().mean()) for column in columns},
                "all_six_observed": float(dataset[columns].notna().all(axis=1).mean()),
            }
    (OUT / "weather" / "weather_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
