"""Raw 5-min sequences for the rows of the minute-level dataset (v2).

For every row of `results/v2/phase6/minute_dataset_{year}.parquet` the 24 bins of
[t_end-120, t_end) at the upstream and downstream cross-section are written in
row order, i.e. exactly the span that the aggregated W and C features summarise.
Channels per bin (raw, NaN where the bin has no data):
  flow       vehicles in the bin
  heavy      heavy-vehicle share
  speed      flow-weighted mean speed
  occ        mean lane occupancy
  sd         mean lane speed standard deviation
  vminlane   lowest lane speed
  valid      lane-minutes with a flow value
The series come from the same lane files and aggregation as Phase 6
(`phase6_minute_dataset.section_series`); the dataset itself is read only.
Output: results/v2/phase9/sequences_{year}.npy, float32, shape (rows, 2, 24, 7).
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from phase2_aggregate_sites import read_metadata
from phase2_aggregate_traffic import source_files
from phase6_minute_dataset import section_series


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "analysis" / "results" / "v2" / "phase6"
OUT = ROOT / "analysis" / "results" / "v2" / "phase9"
BINS = 24
CHANNELS = ["flow", "heavy", "speed", "occ", "sd", "vminlane", "valid"]


def site_windows(task: tuple[int, str, list[str], np.ndarray]) -> tuple[str, np.ndarray | None]:
    """Windows of BINS bins ending before each requested bin index."""
    year, site, unit_paths, end_index = task
    series = section_series(year, unit_paths)
    if series is None:
        return site, None
    series = series.with_columns(
        pl.when(pl.col("valid") > 0).then(pl.col("n")).otherwise(None).alias("flow"),
        pl.when(pl.col("n") > 0).then(pl.col("n_lkw") / pl.col("n")).otherwise(None).alias("heavy"),
        pl.when(pl.col("nv") > 0).then(pl.col("vq") / pl.col("nv")).otherwise(None).alias("speed"),
        pl.col("vmin").alias("vminlane"),
    )
    values = series.select(CHANNELS).to_numpy().astype(np.float32)
    padded = np.vstack([np.full((BINS, len(CHANNELS)), np.nan, dtype=np.float32), values])
    # Row i of `padded` is bin i - BINS; the window for end index e is bins e-BINS .. e-1.
    take = end_index[:, None] + np.arange(BINS)[None, :]
    return site, padded[take]


def build(year: int, workers: int) -> dict:
    rows = pd.read_parquet(DATA / f"minute_dataset_{year}.parquet", columns=["row_id", "site_up", "site_dn", "t_end"])
    end_index = ((rows["t_end"] - pd.Timestamp(datetime(year, 1, 1))) // pd.Timedelta(minutes=5)).to_numpy()
    if (end_index < BINS).any() or (rows["t_end"].dt.minute % 5 != 0).any():
        raise ValueError("t_end must lie on the 5-min grid and at least two hours into the year")
    metadata, files = read_metadata(), source_files(year)
    sites = pd.unique(pd.concat([rows["site_up"], rows["site_dn"]]).dropna())
    needed = {}
    for site in sites:
        index = np.unique(np.concatenate([end_index[rows["site_up"].to_numpy() == site],
                                          end_index[rows["site_dn"].to_numpy() == site]]))
        needed[site] = index
    tasks = []
    for site in sites:
        paths = [str(files[unit]) for unit in metadata.loc[metadata["site"] == site, "unit"] if unit in files]
        if paths:
            tasks.append((year, site, paths, needed[site]))
    print(f"{year}: {len(tasks)} of {len(sites)} cross-sections have lane files", flush=True)
    sequences = np.full((len(rows), 2, BINS, len(CHANNELS)), np.nan, dtype=np.float32)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for count, (site, windows) in enumerate(pool.map(site_windows, tasks, chunksize=4), 1):
            if windows is not None:
                for side, column in enumerate(("site_up", "site_dn")):
                    members = np.flatnonzero(rows[column].to_numpy() == site)
                    sequences[members, side] = windows[np.searchsorted(needed[site], end_index[members])]
            if count % 100 == 0 or count == len(tasks):
                print(f"{year}: cross-sections {count}/{len(tasks)}", flush=True)
    np.save(OUT / f"sequences_{year}.npy", sequences)
    observed = ~np.isnan(sequences[..., CHANNELS.index("flow")])
    return {"rows": int(len(rows)), "shape": list(sequences.shape),
            "share_bins_with_flow": {"up": float(observed[:, 0].mean()), "dn": float(observed[:, 1].mean())},
            "rows_with_no_flow_in_any_bin": float((~observed.any(axis=(1, 2))).mean())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"bins": BINS, "bin_minutes": 5, "channels": CHANNELS, "sides": ["up", "dn"],
               "span": "[t_end-120 min, t_end)", "years": {}}
    for year in (2024, 2025):
        summary["years"][str(year)] = build(year, args.workers)
    (OUT / "sequences_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
