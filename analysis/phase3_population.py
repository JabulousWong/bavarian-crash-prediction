"""Label-independent finite-population edge-hour sampling for PU risk estimation.

U estimates p(X) on the declared provisional edge-ID/hour frame. It is NOT the
phase2 event-buffered control set or a sample conditional on S=0. Source labels
are attached only after sampling. Existing phase1/2 artifacts are read-only.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import pandas as pd
import polars as pl

from phase2_aggregate_traffic import aggregate_one, source_files
from phase2_aggregate_sites import read_metadata
from phase2_baseline import BASE_FEATURES

ROOT = Path(__file__).resolve().parents[1]
P1 = ROOT / "analysis/results/phase1"
P2 = ROOT / "analysis/results/phase2"
OUT = ROOT / "analysis/results/phase3/population"
CACHE = ROOT / "analysis/cache/phase3_population"
OLD_CHANNELS = ROOT / "analysis/cache/traffic_hourly"
OLD_SITES = ROOT / "analysis/cache/traffic_site_hourly"
SEED = 20260929
THRESHOLDS = (50, 100, 150)


def fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest()}


def prepare_caches(year: int, sites: set[str], workers: int) -> dict:
    """Complete traffic coverage using metadata alone, independent of crashes."""
    metadata = read_metadata()
    metadata = metadata.loc[metadata.site.isin(sites)]
    sources = source_files(year)
    tasks = []
    missing = []
    units = sorted(set(metadata.unit))
    for unit in units:
        old = OLD_CHANNELS / str(year) / f"{unit}.parquet"
        if old.exists():
            continue
        if unit not in sources:
            missing.append(unit)
            continue
        new = CACHE / "channels" / str(year) / f"{unit}.parquet"
        tasks.append((unit, str(sources[unit]), str(new)))
    statuses = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(aggregate_one, task) for task in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            statuses.append(future.result())
            if i % 10 == 0 or i == len(futures):
                print(f"{year}: additional population channels {i}/{len(futures)}", flush=True)
    missing_sites = []
    additional_sites = []
    for site, group in metadata.groupby("site"):
        if (OLD_SITES / str(year) / f"{site}.parquet").exists():
            continue
        output = CACHE / "sites" / str(year) / f"{site}.parquet"
        additional_sites.append(site)
        if output.exists():
            continue
        paths = []
        for unit in sorted(set(group.unit)):
            old = OLD_CHANNELS / str(year) / f"{unit}.parquet"
            new = CACHE / "channels" / str(year) / f"{unit}.parquet"
            if old.exists() or new.exists():
                paths.append(old if old.exists() else new)
        if not paths:
            missing_sites.append(site)
            continue
        data = pl.concat([pl.read_parquet(p) for p in paths], how="vertical")
        # Same lane aggregation as phase2, to keep the feature definition fixed.
        result = data.group_by("feature_hour").agg(
            *[pl.col(c).sum().alias(c) for c in ("n_kfz", "n_lkw", "n_pkw")],
            *[(pl.col(f"v_{c}") * pl.col(f"n_{c}")).sum().truediv(
                pl.col(f"n_{c}").sum()).alias(f"v_{c}") for c in ("kfz", "lkw", "pkw")],
            *[pl.col(c).mean().alias(c) for c in
              ("occupancy_mean", "net_gap_mean", "speed_sd_mean")],
            pl.col("valid_minutes_flow").sum().alias("valid_lane_minutes_flow"),
            pl.col("valid_minutes_speed").sum().alias("valid_lane_minutes_speed"),
            pl.col("unit").n_unique().alias("lane_channels"),
        ).sort("feature_hour").with_columns(pl.lit(site).alias("site"))
        output.parent.mkdir(parents=True, exist_ok=True)
        result.write_parquet(output, compression="zstd")
    return {
        "required_sites": len(sites), "required_channels": len(units),
        "additional_channels": len(tasks), "additional_sites": len(additional_sites),
        "missing_raw_channels": missing, "missing_sites": missing_sites,
        "additional_source_files": [str(sources[t[0]]) for t in tasks],
    }


def eligible_hours(year: int, coverage: dict) -> tuple[pd.DatetimeIndex, dict]:
    """Use complete source-overlap outcome hours with unambiguous prior hour."""
    start = pd.Timestamp(coverage["coverage_start_local"]).tz_convert("Europe/Berlin")
    end = pd.Timestamp(coverage["coverage_end_local"]).tz_convert("Europe/Berlin")
    start = max(start.tz_localize(None).ceil("h"), pd.Timestamp(year, 1, 1, 1))
    end = min(end.tz_localize(None).floor("h"), pd.Timestamp(year + 1, 1, 1))
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    previous = (hours - pd.Timedelta(hours=1)).tz_localize(
        "Europe/Berlin", ambiguous="NaT", nonexistent="NaT")
    current = hours.tz_localize("Europe/Berlin", ambiguous="NaT", nonexistent="NaT")
    following = (hours + pd.Timedelta(hours=1)).tz_localize(
        "Europe/Berlin", ambiguous="NaT", nonexistent="NaT")
    valid = (previous.notna() & current.notna() & following.notna()
             & ((current - previous) == pd.Timedelta(hours=1))
             & ((following - current) == pd.Timedelta(hours=1)))
    return hours[valid], {
        "first_outcome_hour": str(hours[valid][0]),
        "last_outcome_hour": str(hours[valid][-1]),
        "eligible_outcome_hours": int(valid.sum()),
        "dst_boundary_outcome_hours_excluded": [str(t) for t in hours[~valid]],
        "coverage_rule": "complete outcome hours inside observed Waze min/max envelope",
        "coverage_caveat": "min/max report timestamps do not prove uninterrupted feed availability",
        "traffic_timezone_assumption": "raw naive t_start assumed Europe/Berlin, as in phase2; provider confirmation still required",
    }


def draw_marginal(edge_ids: np.ndarray, hours: pd.DatetimeIndex,
                  sample_size: int, year: int) -> pd.DataFrame:
    n = len(edge_ids) * len(hours)
    count = min(sample_size, n)
    positions = np.random.default_rng(SEED + year).choice(n, count, replace=False)
    return pd.DataFrame({
        "year": year, "population_position": positions,
        "edge_id": edge_ids[positions // len(hours)],
        "outcome_hour": hours[positions % len(hours)],
        "inclusion_probability": count / n,
        "inverse_inclusion_weight": n / count,
    }).sort_values(["edge_id", "outcome_hour"]).reset_index(drop=True)


def attach_labels(frame: pd.DataFrame, events: dict[int, pd.DataFrame]) -> pd.DataFrame:
    for threshold, event in events.items():
        columns = ["edge_id", "event_hour", "official_observed", "waze_observed"]
        selected = event[columns].rename(columns={
            "event_hour": "outcome_hour",
            "official_observed": f"official_observed_{threshold}m",
            "waze_observed": f"waze_observed_{threshold}m",
        })
        frame = frame.merge(selected, on=["edge_id", "outcome_hour"], how="left",
                            validate="one_to_one")
        for prefix in ("official", "waze"):
            name = f"{prefix}_observed_{threshold}m"
            frame[name] = frame[name].eq(True)
    return frame


def attach_features(frame: pd.DataFrame, year: int, edges: pd.DataFrame) -> pd.DataFrame:
    frame = frame.merge(edges[["edge_id", "route", "site_from", "site_to"]],
                        on="edge_id", validate="many_to_one")
    frame["feature_hour"] = frame.outcome_hour - pd.Timedelta(hours=1)
    names = [c.removeprefix("from_") for c in BASE_FEATURES if c.startswith("from_")]
    for side in ("from", "to"):
        site_column = f"site_{side}"
        parts = []
        for site, group in frame.groupby(site_column):
            path = OLD_SITES / str(year) / f"{site}.parquet"
            if not path.exists():
                path = CACHE / "sites" / str(year) / f"{site}.parquet"
            if not path.exists():
                continue
            data = pd.read_parquet(path, columns=["feature_hour", *names])
            if data.feature_hour.duplicated().any():
                raise ValueError(f"Duplicate feature hour in {path}")
            data = data.merge(group[["feature_hour"]].drop_duplicates(), on="feature_hour")
            data[site_column] = site
            parts.append(data)
        data = pd.concat(parts, ignore_index=True).rename(columns={n: f"{side}_{n}" for n in names})
        frame = frame.merge(data, on=[site_column, "feature_hour"], how="left",
                            validate="many_to_one")
    frame["hour"] = frame.outcome_hour.dt.hour
    frame["weekday"] = frame.outcome_hour.dt.weekday
    frame["month"] = frame.outcome_hour.dt.month
    frame["is_weekend"] = frame.weekday >= 5
    return frame


def validate_sampling() -> dict:
    hours = pd.date_range("2024-01-02", periods=7, freq="h")
    edge_ids = np.array([2, 19, 84])
    census = draw_marginal(edge_ids, hours, 100, 2024)
    assert len(census) == 21
    assert not census.duplicated(["edge_id", "outcome_hour"]).any()
    assert census.groupby("edge_id").size().eq(7).all()
    a, b = (draw_marginal(edge_ids, hours, 10, 2024) for _ in range(2))
    pd.testing.assert_frame_equal(a, b)
    assert np.allclose(a.inclusion_probability, 10 / 21)
    labels = {50: pd.DataFrame({"edge_id": [2], "event_hour": [hours[0]],
                                "official_observed": [True], "waze_observed": [True]})}
    joined = attach_labels(census, labels)
    assert joined.official_observed_50m.sum() == 1  # P remains in marginal census.
    return {"toy_census_complete_unique": True, "deterministic_sample": True,
            "equal_inclusion_probabilities": True, "positives_not_removed": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=200_000)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.sample_size <= 0 or args.workers <= 0:
        parser.error("sample-size and workers must be positive")
    OUT.mkdir(parents=True, exist_ok=True)
    edge_path = P1 / "detector_corridor_edges.csv"
    edges = pd.read_csv(edge_path).sort_values("edge_id")
    if edges.edge_id.duplicated().any():
        raise ValueError("edge_id must be unique")
    sites = set(edges.site_from) | set(edges.site_to)
    source_path = P1 / "source_summary.json"
    coverage = {r["year"]: r for r in json.loads(source_path.read_text())}
    summary = {
        "status": "correct marginal sampling on an explicitly provisional edge-ID/hour frame",
        "seed": SEED, "sampling": "simple random sample without replacement, before label lookup",
        "frame_unit": "inherited detector-corridor edge ID x eligible outcome hour",
        "U_role": "marginal p(X); never condition on source label, proximity to events or case strata",
        "P_role": "census of source-official injury edge-hours within the same frame; not proven SCAR for all crashes",
        "year_isolation": "2024 for training/prior scenarios; 2025 marginal sample for locked evaluation/audit only",
        "geometry_audit": {
            "inherited_edge_ids": len(edges), "sites": len(sites),
            "exact_geometry_duplicate_rows": int(edges.geometry_wkt.duplicated().sum()),
            "repeated_unordered_site_pairs": int(edges.apply(
                lambda r: tuple(sorted((r.site_from, r.site_to))), axis=1).duplicated().sum()),
            "zero_length_edge_ids": edges.loc[edges.length_straight_m.eq(0), "edge_id"].tolist(),
            "limitation": "IDs retained for compatibility; not a verified non-overlapping physical motorway network. Counts and rates cannot be presented as actual Suedbayern network prevalence.",
        },
        "sources": [fingerprint(edge_path), fingerprint(source_path)],
        "checks": validate_sampling(), "years": {},
    }
    for year in (2024, 2025):
        print(f"{year}: preparing label-independent population", flush=True)
        cache = prepare_caches(year, sites, args.workers)
        hours, time_info = eligible_hours(year, coverage[year])
        population_n = len(edges) * len(hours)
        rows = draw_marginal(edges.edge_id.to_numpy(), hours, args.sample_size, year)
        events = {}
        threshold_summary = {}
        for threshold in THRESHOLDS:
            path = P2 / f"label_events_{year}_{threshold}m.parquet"
            summary["sources"].append(fingerprint(path))
            event = pd.read_parquet(path)
            event["event_hour"] = pd.to_datetime(event.event_hour)
            keep = event.edge_id.isin(edges.edge_id) & event.event_hour.isin(hours)
            retained = event.loc[keep].copy()
            events[threshold] = retained
            official = retained.loc[retained.official_observed]
            threshold_summary[str(threshold)] = {
                "official_positive_edge_hours": len(official),
                "official_crash_records": int(official.official_crashes.sum()),
                "waze_positive_edge_hours": int(retained.waze_observed.sum()),
                "union_positive_edge_hours": len(retained),
                "events_excluded_outside_frame": int((~keep).sum()),
                "observed_official_edge_hour_fraction": len(official) / population_n,
                "interpretation": "observed source-official fraction, not identified all-crash class prior",
            }
        rows = attach_labels(rows, events)
        # Attach features once to the union of sampled rows and all positive rows.
        pkeys = events[150].loc[events[150].official_observed, ["edge_id", "event_hour"]]
        keys = pd.concat([rows[["edge_id", "outcome_hour"]],
                          pkeys.rename(columns={"event_hour": "outcome_hour"})]).drop_duplicates()
        feature_data = attach_features(keys, year, edges)
        rows = rows.merge(feature_data, on=["edge_id", "outcome_hour"], validate="one_to_one")
        assert not rows.duplicated(["edge_id", "outcome_hour"]).any()
        assert rows.feature_hour.eq(rows.outcome_hour - pd.Timedelta(hours=1)).all()
        for threshold in THRESHOLDS:
            official = events[threshold].loc[events[threshold].official_observed].rename(
                columns={"event_hour": "outcome_hour"})
            positive = official.merge(feature_data, on=["edge_id", "outcome_hour"],
                                      validate="one_to_one")
            assert len(positive) == threshold_summary[str(threshold)]["official_positive_edge_hours"]
            positive["year"] = year
            positive.to_parquet(OUT / f"official_positive_{year}_{threshold}m.parquet", index=False)
            threshold_summary[str(threshold)]["sample_official_edge_hours"] = int(
                rows[f"official_observed_{threshold}m"].sum())
            threshold_summary[str(threshold)]["sample_waze_edge_hours"] = int(
                rows[f"waze_observed_{threshold}m"].sum())
        rows.to_parquet(OUT / f"marginal_edge_hours_{year}.parquet", index=False)
        edges.assign(eligible_hours=len(hours), population_edge_hours=len(hours)).to_csv(
            OUT / f"population_edges_{year}.csv", index=False)
        pd.DataFrame({"outcome_hour": hours, "feature_hour": hours - pd.Timedelta(hours=1)}).to_csv(
            OUT / f"population_hours_{year}.csv", index=False)
        per_year = {
            **time_info, "cache_completion": cache, "population_edge_hours": population_n,
            "sample_edge_hours": len(rows), "inclusion_probability": len(rows) / population_n,
            "sample_distinct_edges": int(rows.edge_id.nunique()),
            "rows_with_all_traffic_features_finite": int(np.isfinite(rows[BASE_FEATURES]).all(axis=1).sum()),
            "feature_missingness_does_not_filter_population": True,
            "thresholds": threshold_summary,
        }
        summary["years"][str(year)] = per_year
        (OUT / "population_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({"year": year, "population_edge_hours": population_n,
                          "sample_rows": len(rows), "thresholds": threshold_summary}), flush=True)
    summary["output_hashes"] = [fingerprint(p) for p in sorted(OUT.glob("*.parquet"))]
    (OUT / "population_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
