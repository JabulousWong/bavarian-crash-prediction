"""Build a fixed matched edge-hour dataset with strictly pre-outcome traffic features."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PHASE1 = ROOT / "analysis" / "results" / "phase1"
PHASE2 = ROOT / "analysis" / "results" / "phase2"
SITE_CACHE = ROOT / "analysis" / "cache" / "traffic_site_hourly"
OUT = PHASE2 / "datasets"
THRESHOLD = 100
SEED = 20260929


def source_events(year: int) -> pd.DataFrame:
    data = pd.read_parquet(PHASE2 / f"label_events_{year}_{THRESHOLD}m.parquet")
    data["event_hour"] = pd.to_datetime(data["event_hour"])
    return data


def event_exclusion(events: pd.DataFrame, buffer_h: int = 2) -> set[tuple[int, pd.Timestamp]]:
    excluded: set[tuple[int, pd.Timestamp]] = set()
    for row in events[["edge_id", "event_hour"]].itertuples(index=False):
        for offset in range(-buffer_h, buffer_h + 1):
            excluded.add((int(row.edge_id), row.event_hour + pd.Timedelta(hours=offset)))
    return excluded


def sample_matched(year: int, events: pd.DataFrame, mode: str = "official") -> pd.DataFrame:
    if mode == "official":
        cases = events.loc[events["official_observed"]].copy()
    elif mode == "union":
        cases = events.loc[events["official_observed"] | events["waze_observed"]].copy()
    else:
        raise ValueError(f"Unknown sampling mode: {mode}")
    cases = cases.sort_values(["event_hour", "edge_id"]).reset_index(drop=True)
    excluded = event_exclusion(events)
    used_controls: set[tuple[int, pd.Timestamp]] = set()
    rng = np.random.default_rng(SEED + year)
    rows = []
    for stratum_id, case in cases.iterrows():
        case_hour = pd.Timestamp(case["event_hour"])
        edge_id = int(case["edge_id"])
        rows.append(
            {
                "year": year,
                "stratum_id": f"{mode}-{year}-{stratum_id}",
                "edge_id": edge_id,
                "outcome_hour": case_hour,
                "target_official": int(bool(case["official_observed"])),
                "target_union": 1,
                "official_observed": bool(case["official_observed"]),
                "waze_observed": bool(case["waze_observed"]),
                "source_pattern": case["source_pattern"],
            }
        )
        candidates = []
        for day_delta in range(-28, 29):
            if day_delta == 0:
                continue
            candidate = case_hour + pd.Timedelta(days=day_delta)
            if candidate.year != year:
                continue
            if (candidate.weekday() >= 5) != (case_hour.weekday() >= 5):
                continue
            key = (edge_id, candidate)
            if key in excluded or key in used_controls:
                continue
            candidates.append(candidate)
        if len(candidates) < 4:
            # Keep the stratum auditable but do not create an incomplete risk set.
            rows = [row for row in rows if row["stratum_id"] != f"{mode}-{year}-{stratum_id}"]
            continue
        selected = rng.choice(np.array(candidates, dtype="datetime64[ns]"), size=4, replace=False)
        for value in selected:
            control_hour = pd.Timestamp(value)
            used_controls.add((edge_id, control_hour))
            rows.append(
                {
                    "year": year,
                    "stratum_id": f"{mode}-{year}-{stratum_id}",
                    "edge_id": edge_id,
                    "outcome_hour": control_hour,
                    "target_official": 0,
                    "target_union": 0,
                    "official_observed": False,
                    "waze_observed": False,
                    "source_pattern": "event-buffered_control",
                }
            )
    frame = pd.DataFrame(rows)
    complete = frame.groupby("stratum_id")["target_union"].agg(["sum", "count"])
    valid_ids = complete.index[(complete["sum"] == 1) & (complete["count"] == 5)]
    return frame.loc[frame["stratum_id"].isin(valid_ids)].reset_index(drop=True)


def add_site_features(rows: pd.DataFrame, year: int) -> pd.DataFrame:
    edges = pd.read_csv(PHASE1 / "detector_corridor_edges.csv").set_index("edge_id")
    rows = rows.join(edges[["route", "site_from", "site_to"]], on="edge_id")
    rows["feature_hour"] = rows["outcome_hour"] - pd.Timedelta(hours=1)
    feature_names = [
        "n_kfz", "n_lkw", "n_pkw", "v_kfz", "v_lkw", "v_pkw",
        "occupancy_mean", "net_gap_mean", "speed_sd_mean",
        "valid_lane_minutes_flow", "valid_lane_minutes_speed", "lane_channels",
    ]
    for side in ("from", "to"):
        site_column = f"site_{side}"
        parts = []
        for site, group in rows.groupby(site_column):
            path = SITE_CACHE / str(year) / f"{site}.parquet"
            if not path.exists():
                continue
            site_data = pd.read_parquet(path)
            site_data["feature_hour"] = pd.to_datetime(site_data["feature_hour"])
            needed = group[["feature_hour"]].drop_duplicates()
            selected = site_data.merge(needed, on="feature_hour", how="inner")
            selected[site_column] = site
            parts.append(selected[[site_column, "feature_hour", *feature_names]])
        feature_table = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        rename = {name: f"{side}_{name}" for name in feature_names}
        feature_table = feature_table.rename(columns=rename)
        rows = rows.merge(feature_table, on=[site_column, "feature_hour"], how="left")
    rows["hour"] = rows["outcome_hour"].dt.hour
    rows["weekday"] = rows["outcome_hour"].dt.weekday
    rows["month"] = rows["outcome_hour"].dt.month
    rows["is_weekend"] = rows["weekday"] >= 5
    rows["asof_rule"] = "features from [h-1,h); outcome in [h,h+1)"
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for year in (2024, 2025):
        events = source_events(year)
        summary[str(year)] = {}
        for mode in ("official", "union"):
            matched = sample_matched(year, events, mode=mode)
            dataset = add_site_features(matched, year)
            suffix = "" if mode == "official" else "_union"
            output = OUT / f"matched_edge_hours_{year}_{THRESHOLD}m{suffix}.parquet"
            dataset.to_parquet(output, index=False)
            traffic_columns = [column for column in dataset if column.startswith(("from_", "to_"))]
            summary[str(year)][mode] = {
                "rows": int(len(dataset)),
                "case_edge_hours": int(dataset["target_union"].sum()),
                "official_case_edge_hours": int(dataset["target_official"].sum()),
                "control_edge_hours": int((dataset["target_union"] == 0).sum()),
                "strata": int(dataset["stratum_id"].nunique()),
                "cases_with_waze_same_edge_hour_label": int(
                    dataset.loc[dataset["target_union"] == 1, "waze_observed"].sum()
                ),
                "rows_with_complete_traffic": int(dataset[traffic_columns].notna().all(axis=1).sum()),
                "network_threshold_m": THRESHOLD,
            }
    (OUT / "matched_dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
