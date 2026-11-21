"""Combine lane-channel hourly caches into cross-section hourly features."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import polars as pl


ROOT = Path(__file__).resolve().parents[1]
CHANNEL_CACHE = ROOT / "analysis" / "cache" / "traffic_hourly"
SITE_CACHE = ROOT / "analysis" / "cache" / "traffic_site_hourly"


def read_metadata() -> pd.DataFrame:
    path = ROOT / "autobahn" / "Traffic" / "selection_FG1.csv"
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines()[1:]:
        fields = line.split(";")
        if len(fields) >= 7:
            rows.append({"site": fields[0], "unit": fields[1]})
    return pd.DataFrame(rows).drop_duplicates()


def aggregate_site(task: tuple[int, str, list[str], str]) -> dict:
    year, site, units, output_text = task
    output = Path(output_text)
    if output.exists():
        return {"site": site, "status": "cached"}
    paths = [CHANNEL_CACHE / str(year) / f"{unit}.parquet" for unit in units]
    paths = [path for path in paths if path.exists()]
    if not paths:
        return {"site": site, "status": "missing"}
    data = pl.concat([pl.read_parquet(path) for path in paths], how="vertical")
    site_hour = (
        data.group_by("feature_hour")
        .agg(
            pl.col("n_kfz").sum().alias("n_kfz"),
            pl.col("n_lkw").sum().alias("n_lkw"),
            pl.col("n_pkw").sum().alias("n_pkw"),
            ((pl.col("v_kfz") * pl.col("n_kfz")).sum() / pl.col("n_kfz").sum()).alias("v_kfz"),
            ((pl.col("v_lkw") * pl.col("n_lkw")).sum() / pl.col("n_lkw").sum()).alias("v_lkw"),
            ((pl.col("v_pkw") * pl.col("n_pkw")).sum() / pl.col("n_pkw").sum()).alias("v_pkw"),
            pl.col("occupancy_mean").mean().alias("occupancy_mean"),
            pl.col("net_gap_mean").mean().alias("net_gap_mean"),
            pl.col("speed_sd_mean").mean().alias("speed_sd_mean"),
            pl.col("valid_minutes_flow").sum().alias("valid_lane_minutes_flow"),
            pl.col("valid_minutes_speed").sum().alias("valid_lane_minutes_speed"),
            pl.col("unit").n_unique().alias("lane_channels"),
        )
        .sort("feature_hour")
        .with_columns(pl.lit(site).alias("site"))
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    site_hour.write_parquet(output, compression="zstd")
    return {"site": site, "status": "written"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, choices=[2024, 2025], required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    metadata = read_metadata()
    available = {p.stem for p in (CHANNEL_CACHE / str(args.year)).glob("*.parquet")}
    metadata = metadata.loc[metadata["unit"].isin(available)]
    output_dir = SITE_CACHE / str(args.year)
    tasks = [
        (args.year, site, sorted(set(group["unit"])), str(output_dir / f"{site}.parquet"))
        for site, group in metadata.groupby("site")
    ]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(aggregate_site, task) for task in tasks]
        for count, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if count % 50 == 0 or count == len(futures):
                print(f"{args.year}: completed {count}/{len(futures)} sites", flush=True)
    manifest = {
        "year": args.year,
        "sites": len(tasks),
        "written": sum(r["status"] == "written" for r in results),
        "cached": sum(r["status"] == "cached" for r in results),
        "missing": [r["site"] for r in results if r["status"] == "missing"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
