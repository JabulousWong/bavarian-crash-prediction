"""Aggregate required one-minute detector channels to leakage-safe hourly features."""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import polars as pl


ROOT = Path(__file__).resolve().parents[1]
PHASE1 = ROOT / "analysis" / "results" / "phase1"
PHASE2 = ROOT / "analysis" / "results" / "phase2"
CACHE = ROOT / "analysis" / "cache" / "traffic_hourly"


def read_metadata() -> pd.DataFrame:
    path = ROOT / "autobahn" / "Traffic" / "selection_FG1.csv"
    records = []
    for line in path.read_text(encoding="utf-8-sig").splitlines()[1:]:
        fields = line.split(";")
        if len(fields) < 7:
            continue
        records.append({"site": fields[0], "unit": fields[1]})
    return pd.DataFrame(records).drop_duplicates()


def required_units(threshold: int) -> pd.DataFrame:
    edges = pd.read_csv(PHASE1 / "detector_corridor_edges.csv")
    edge_ids: set[int] = set()
    for year in (2024, 2025):
        labels = pd.read_parquet(PHASE2 / f"label_events_{year}_{threshold}m.parquet")
        edge_ids.update(labels["edge_id"].astype(int).unique())
    selected = edges.loc[edges["edge_id"].isin(edge_ids)]
    sites = set(selected["site_from"]) | set(selected["site_to"])
    metadata = read_metadata()
    return metadata.loc[metadata["site"].isin(sites)].copy()


def source_files(year: int) -> dict[str, Path]:
    folder = ROOT / "autobahn" / ("Traffic_2024" if year == 2024 else "Traffic")
    result = {}
    suffix = re.compile(rf"^FG1_Kurz_(.+)_agg1min_{year}-01-01_bis_{year + 1}-01-01\.csv$")
    for path in folder.glob("*.csv"):
        match = suffix.match(path.name)
        if match:
            result[match.group(1)] = path
    return result


def aggregate_one(task: tuple[str, str, str]) -> dict:
    unit, source_text, output_text = task
    source, output = Path(source_text), Path(output_text)
    if output.exists():
        return {"unit": unit, "status": "cached", "rows": None}
    data = pl.read_csv(
        source,
        separator=";",
        null_values=["null", ""],
        try_parse_dates=True,
        schema_overrides={"t_start": pl.Datetime},
    )
    numeric = [
        "q_kfz_h", "q_lkw_h", "q_pkw_h", "v_kfz", "v_lkw", "v_pkw",
        "b", "t_netto", "s_kfz",
    ]
    for column in numeric:
        if column not in data.columns:
            data = data.with_columns(pl.lit(None, dtype=pl.Float64).alias(column))
    data = data.with_columns(
        [pl.col(column).cast(pl.Float64, strict=False).alias(column) for column in numeric]
    )
    data = data.with_columns(
        pl.col("t_start").dt.truncate("1h").alias("feature_hour"),
        (pl.col("q_kfz_h") / 60).alias("n_kfz"),
        (pl.col("q_lkw_h") / 60).alias("n_lkw"),
        (pl.col("q_pkw_h") / 60).alias("n_pkw"),
        (pl.col("v_kfz") * pl.col("q_kfz_h")).alias("v_kfz_num"),
        (pl.col("v_lkw") * pl.col("q_lkw_h")).alias("v_lkw_num"),
        (pl.col("v_pkw") * pl.col("q_pkw_h")).alias("v_pkw_num"),
    )
    hourly = (
        data.group_by("feature_hour")
        .agg(
            pl.col("n_kfz").sum().alias("n_kfz"),
            pl.col("n_lkw").sum().alias("n_lkw"),
            pl.col("n_pkw").sum().alias("n_pkw"),
            (pl.col("v_kfz_num").sum() / pl.col("q_kfz_h").sum()).alias("v_kfz"),
            (pl.col("v_lkw_num").sum() / pl.col("q_lkw_h").sum()).alias("v_lkw"),
            (pl.col("v_pkw_num").sum() / pl.col("q_pkw_h").sum()).alias("v_pkw"),
            pl.col("b").mean().alias("occupancy_mean"),
            pl.col("t_netto").mean().alias("net_gap_mean"),
            pl.col("s_kfz").mean().alias("speed_sd_mean"),
            pl.col("q_kfz_h").is_not_null().sum().alias("valid_minutes_flow"),
            pl.col("v_kfz").is_not_null().sum().alias("valid_minutes_speed"),
        )
        .sort("feature_hour")
        .with_columns(pl.lit(unit).alias("unit"))
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    hourly.write_parquet(output, compression="zstd")
    return {"unit": unit, "status": "written", "rows": hourly.height}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, choices=[2024, 2025], required=True)
    parser.add_argument("--threshold", type=int, default=100)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    required = required_units(args.threshold)
    files = source_files(args.year)
    output_dir = CACHE / str(args.year)
    units = sorted(set(required["unit"]))
    if args.limit:
        units = units[: args.limit]
    tasks, missing = [], []
    for unit in units:
        source = files.get(unit)
        if source is None:
            missing.append(unit)
            continue
        tasks.append((unit, str(source), str(output_dir / f"{unit}.parquet")))
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(aggregate_one, task) for task in tasks]
        for count, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if count % 25 == 0 or count == len(futures):
                print(f"{args.year}: completed {count}/{len(futures)}", flush=True)
    manifest = {
        "year": args.year,
        "threshold_m": args.threshold,
        "required_units": len(units),
        "available_units": len(tasks),
        "missing_units": missing,
        "written": sum(item["status"] == "written" for item in results),
        "cached": sum(item["status"] == "cached" for item in results),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
