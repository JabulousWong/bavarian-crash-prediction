"""Minute-level sub-study dataset: Waze-timed accident events on edge x 5-min units.

Official records give only the accident hour. The first Waze report (t0) gives a
minute-level reference, available for Waze-reported events only. Each row is an
(edge, t_end) unit whose traffic features use 5-min bins strictly before t_end:
window W = [t_end-30, t_end) and context C = [t_end-120, t_end-30) minutes.

Row kinds
- waze_case / waze_control: t_end = floor5(t0) - buffer, buffer in BUFFERS minutes
  (the report lags the crash, so the buffer keeps W ahead of the report); four
  same-edge controls at +-7k days, outside other accident reports.
- marginal: label-independent uniform sample of edge x 5-min units (U for PU
  risk; known events are not removed).
- official_case / official_control: the Phase 2 official matched cohort with
  t_end = start of the published accident hour, so W = [h-30, h) is pre-outcome.

Waze events take the carriageway from the report heading (magvar). Police
records have no direction field, but their coordinates lie on one carriageway:
the heading of Waze reports made at other times within 15-25 m of the crash point
assigns it, with the crash's own accident reports and then the Phase 1 nearest
edge as fallbacks. All existing artifacts and raw sources are read only.
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
import shapely

import dlm_assign
from scipy.spatial import cKDTree
from shapely.strtree import STRtree

import phase1_match_audit as audit
from phase1_network_scope import xy_m
from phase2_aggregate_traffic import source_files
from phase2_aggregate_sites import read_metadata


ROOT = Path(__file__).resolve().parents[1]
PHASE1 = ROOT / "analysis" / "results" / "phase1"
PHASE2 = ROOT / "analysis" / "results" / "phase2"
OUT = ROOT / "analysis" / "results" / "phase6"
SEED = 20260930
BUFFERS = (15, 30, 45)
NETWORK_THRESHOLD_M = 100
MATCH_RADIUS_M = 500
MATCH_WINDOW_H = (-0.25, 1.75)
REPEAT_RADIUS_M = 5000
REPEAT_HOURS = 3
DESTINATIONS = {
    "Nbg": (11.08, 49.45), "Pas": (13.43, 48.57), "Fue": (10.70, 47.57), "Wue": (9.93, 49.79),
    "Mch": (11.58, 48.14), "Sbg": (13.04, 47.80), "Stg": (9.18, 48.78), "Deg": (12.96, 48.84),
    "Hof": (11.91, 50.31), "Kff": (12.19, 47.61), "Ros": (12.12, 47.86), "Sta": (11.34, 48.00),
    "Gap": (11.10, 47.49), "Lin": (9.68, 47.55),
}
SITE_FEATURES = [
    "flow_w", "heavy_w", "v_w", "vmin5_w", "vstd_w", "vminlane_w", "occ_w", "occmax_w", "sd_w",
    "vtrend_w", "v_c", "flow_c", "occ_c", "dv_wc", "flow_ratio", "docc_wc", "valid_w",
]


def load_edges() -> pd.DataFrame:
    edges = pd.read_csv(PHASE1 / "detector_corridor_edges.csv")
    coords = shapely.get_coordinates(shapely.from_wkt(edges["geometry_wkt"].to_numpy())).reshape(-1, 2, 2)
    edges["dx"], edges["dy"] = coords[:, 1, 0] - coords[:, 0, 0], coords[:, 1, 1] - coords[:, 0, 1]
    edges["mx"], edges["my"] = coords[:, :, 0].mean(axis=1), coords[:, :, 1].mean(axis=1)
    return edges


def route_travel_signs(edges: pd.DataFrame) -> pd.DataFrame:
    """+1 if traffic on a route runs with increasing kilometre (site_from -> site_to).

    Primary evidence: length-weighted agreement between the increasing-km bearing
    and the bearing to the destination named in the route. Where that vote is weak
    (ring road), right-hand traffic decides: a carriageway lies to the right of
    the opposite carriageway's chord when seen along its direction of travel.
    """
    usable = edges.loc[edges["length_straight_m"] > 300].copy()
    usable["road"] = usable["route"].str.split("_").str[0]
    rows = []
    for route, group in usable.groupby("route"):
        lon, lat = DESTINATIONS[route.split("_")[-1]]
        cx, cy = xy_m(np.array([lon]), np.array([lat]))
        to_destination = np.arctan2(cx[0] - group["mx"], cy[0] - group["my"])
        along_km = np.arctan2(group["dx"], group["dy"])
        weight = group["length_straight_m"]
        vote = float((np.sign(np.cos(to_destination - along_km)) * weight).sum() / weight.sum())
        opposite = usable.loc[(usable["road"] == group["road"].iloc[0]) & (usable["route"] != route)]
        right_share = np.nan
        if len(opposite):
            distance, index = cKDTree(opposite[["mx", "my"]].to_numpy()).query(group[["mx", "my"]].to_numpy())
            other = opposite.iloc[index]
            length = np.hypot(group["dx"], group["dy"]).to_numpy()
            cross = (group["dx"].to_numpy() * (group["my"].to_numpy() - other["my"].to_numpy())
                     - group["dy"].to_numpy() * (group["mx"].to_numpy() - other["mx"].to_numpy())) / length
            keep = (distance < 400) & (np.abs(cross) > 3)
            if keep.any():
                right_share = float((cross[keep] < 0).mean())
        if abs(vote) >= 0.5 or np.isnan(right_share):
            sign, basis = (1 if vote > 0 else -1), "destination bearing"
        else:
            sign, basis = (1 if right_share > 0.5 else -1), "right-hand carriageway offset"
        rows.append({"route": route, "destination_vote": vote, "right_of_increasing_km_share": right_share,
                     "travel_sign": sign, "basis": basis})
    return pd.DataFrame(rows)


def eligible_window(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    coverage = {item["year"]: item for item in json.loads((PHASE1 / "source_summary.json").read_text(encoding="utf-8"))}
    end = pd.Timestamp(coverage[year]["coverage_end_local"]).tz_localize(None).floor("D")
    return pd.Timestamp(year, 1, 1, 3), min(end, pd.Timestamp(year + 1, 1, 1))


def waze_events(year: int, edges: pd.DataFrame, signs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """On-network accident clusters with heading, credibility signals, police match and edge.

    Clusters that repeat an earlier report are flagged, not removed.
    """
    clusters = pd.read_csv(PHASE1 / f"waze_accident_clusters_{year}.csv")
    clusters["first_ts_utc"] = pd.to_datetime(clusters["first_ts_utc"], utc=True)
    clusters["t0"] = clusters["first_ts_utc"].dt.tz_convert("Europe/Berlin").dt.tz_localize(None).astype("datetime64[ns]")
    clusters["cluster_id"] = np.arange(len(clusters))
    alerts, _ = audit.load_waze(year)
    alerts = alerts.loc[(alerts["type"] == "ACCIDENT") & alerts["lon"].notna(), ["ts_utc", "lon", "lat", "magvar", "subtype"]]
    alerts["magvar"] = pd.to_numeric(alerts["magvar"], errors="coerce")
    first = clusters[["cluster_id", "first_ts_utc", "lon", "lat"]].merge(
        alerts, left_on="first_ts_utc", right_on="ts_utc", suffixes=("", "_alert"))
    first["gap"] = np.hypot(first["lon"] - first["lon_alert"], first["lat"] - first["lat_alert"])
    first = first.sort_values("gap").drop_duplicates("cluster_id")
    clusters = clusters.merge(first[["cluster_id", "magvar", "subtype"]], on="cluster_id", how="left")

    x, y = xy_m(clusters["lon"].to_numpy(), clusters["lat"].to_numpy())
    clusters["x"], clusters["y"] = x, y
    seconds = clusters["t0"].to_numpy().astype("datetime64[s]").astype(np.int64)
    tree = cKDTree(np.column_stack([x, y]))
    repeated = np.zeros(len(clusters), dtype=bool)
    for index, neighbours in enumerate(tree.query_ball_point(np.column_stack([x, y]), REPEAT_RADIUS_M)):
        delta = seconds[index] - seconds[np.asarray(neighbours)]
        repeated[index] = ((delta > 0) & (delta <= REPEAT_HOURS * 3600)).any()
    clusters["repeat_of_earlier_report"] = repeated

    crashes = pd.read_csv(PHASE1 / f"crash_network_assignment_{year}.csv", dtype={"uid": str})
    crashes["hour_start"] = (pd.to_datetime(crashes["date"]) + pd.to_timedelta(crashes["USTUNDE"], unit="h")).astype("datetime64[ns]")
    cx, cy = xy_m(crashes["lon"].to_numpy(), crashes["lat"].to_numpy())
    crash_seconds = crashes["hour_start"].to_numpy().astype("datetime64[s]").astype(np.int64)
    confirmed = np.zeros(len(clusters), dtype=bool)
    worst = np.full(len(clusters), np.nan)
    crash_matched = np.zeros(len(crashes), dtype=bool)
    lower, upper = (hours * 3600 for hours in MATCH_WINDOW_H)
    for index, neighbours in enumerate(cKDTree(np.column_stack([cx, cy])).query_ball_point(np.column_stack([x, y]), MATCH_RADIUS_M)):
        if not neighbours:
            continue
        neighbours = np.asarray(neighbours)
        delta = seconds[index] - crash_seconds[neighbours]
        accepted = neighbours[(delta >= lower) & (delta <= upper)]
        if len(accepted):
            confirmed[index] = True
            worst[index] = crashes["UKATEGORIE"].to_numpy()[accepted].min()
            crash_matched[accepted] = True
    clusters["official_confirmed"], clusters["worst_injury_category"] = confirmed, worst
    crashes["waze_matched"] = crash_matched

    sign = edges["route"].map(signs.set_index("route")["travel_sign"]).fillna(1).to_numpy()
    travel_bearing = np.arctan2(edges["dx"] * sign, edges["dy"] * sign).to_numpy()
    # Basis-DLM membership and carriageway (dlm_assign) replace heading + 100 m chord distance.
    assigned = dlm_assign.assign(clusters["lon"].to_numpy(), clusters["lat"].to_numpy(), edges, signs)
    members = np.flatnonzero(assigned["member"].to_numpy())
    edge_position = pd.Index(edges["edge_id"]).get_indexer(assigned["edge_id"].to_numpy()[members])
    heading = np.radians(clusters["magvar"].to_numpy()[members])
    chosen = pd.DataFrame({
        "cluster_id": clusters["cluster_id"].to_numpy()[members],
        "edge_id": assigned["edge_id"].to_numpy()[members],
        "distance_m": assigned["corridor_distance_m"].to_numpy()[members],
        "heading_cos": np.nan_to_num(np.cos(heading - travel_bearing[edge_position])),
        "dlm_category": assigned["dlm_category"].to_numpy()[members],
    })
    events = clusters.merge(chosen, on="cluster_id")
    audit_counts = {
        "accident_clusters_bavaria": int(len(clusters)),
        "clusters_within_100m_of_network": int(len(events)),
        "dropped_as_repeat_within_5km_3h": int(events["repeat_of_earlier_report"].sum()),
        "heading_aligned_cos_gt_0.5": float((events["heading_cos"] > 0.5).mean()),
        "police_confirmed_among_network_clusters": int(events["official_confirmed"].sum()),
    }
    return events.reset_index(drop=True), clusters, crashes, audit_counts


def heading_field(years: tuple[int, ...] = (2024, 2025)) -> dict:
    """Every Waze report with a heading: a crowd-sourced map of carriageway direction."""
    frames = []
    for year in years:
        alerts, _ = audit.load_waze(year)
        alerts = alerts.loc[alerts["lon"].notna(), ["type", "ts_wall", "lon", "lat", "magvar"]].copy()
        alerts["magvar"] = pd.to_numeric(alerts["magvar"], errors="coerce")
        frames.append(alerts.dropna(subset=["magvar"]))
    alerts = pd.concat(frames, ignore_index=True)
    x, y = xy_m(alerts["lon"].to_numpy(), alerts["lat"].to_numpy())
    heading = np.radians(alerts["magvar"].to_numpy())
    return {
        "tree": cKDTree(np.column_stack([x, y])),
        "seconds": alerts["ts_wall"].to_numpy().astype("datetime64[s]").astype(np.int64),
        "sin": np.sin(heading), "cos": np.cos(heading),
        "accident": (alerts["type"] == "ACCIDENT").to_numpy(),
    }


def official_directions(crashes: pd.DataFrame, edges: pd.DataFrame, signs: pd.DataFrame,
                        field: dict) -> tuple[pd.DataFrame, dict]:
    """Carriageway of police crashes from the Basis-DLM carriageway axes (dlm_assign).

    Network membership and the directed edge come from the official road model. The
    heading of Waze accident reports made around the crash itself is kept only as an
    independent check of the DLM direction.
    """
    assigned = dlm_assign.assign(crashes["lon"].to_numpy(), crashes["lat"].to_numpy(), edges, signs)
    keep = assigned["member"].to_numpy()
    network = crashes.loc[keep].copy().reset_index(drop=True)
    chosen = assigned.loc[keep].reset_index(drop=True)
    network["nearest_edge"] = chosen["edge_id"].to_numpy()
    network["nearest_corridor_m"] = chosen["corridor_distance_m"].to_numpy()
    network["edge_directed"] = chosen["edge_id"].to_numpy()
    network["direction_basis"] = chosen["direction_basis"].to_numpy()
    network["dlm_category"] = chosen["dlm_category"].to_numpy()

    sign = edges["route"].map(signs.set_index("route")["travel_sign"]).fillna(1).to_numpy()
    travel_bearing = np.arctan2(edges["dx"] * sign, edges["dy"] * sign).to_numpy()
    position = pd.Index(edges["edge_id"]).get_indexer(network["edge_directed"])
    x, y = xy_m(network["lon"].to_numpy(), network["lat"].to_numpy())
    seconds = network["hour_start"].to_numpy().astype("datetime64[s]").astype(np.int64)
    lower, upper = (hours * 3600 for hours in MATCH_WINDOW_H)
    around = field["tree"].query_ball_point(np.column_stack([x, y]), 300)
    agree = []
    for index in range(len(network)):
        nearby = np.asarray(around[index], dtype=int)
        delta = field["seconds"][nearby] - seconds[index]
        own = nearby[(delta >= lower) & (delta <= upper)]
        own = own[field["accident"][own]]
        if len(own) < 1:
            continue
        sx, cx = field["sin"][own].mean(), field["cos"][own].mean()
        if np.hypot(sx, cx) < 0.8:
            continue
        agree.append(np.cos(np.arctan2(sx, cx) - travel_bearing[position[index]]) > 0)
    summary = {
        "network_crashes": int(len(network)),
        "basis": network["direction_basis"].value_counts().to_dict(),
        "dlm_category": network["dlm_category"].value_counts().to_dict(),
        "check_against_own_accident_reports": {
            "crashes_with_own_reports": int(len(agree)),
            "own_report_heading_agrees_with_dlm_edge": float(np.mean(agree)) if agree else None,
        },
    }
    return network, summary


def build_rows(year: int, events: pd.DataFrame, clusters: pd.DataFrame, crashes: pd.DataFrame,
               edges: pd.DataFrame, marginal_size: int) -> pd.DataFrame:
    start, end = eligible_window(year)
    rng = np.random.default_rng(SEED + year)
    seconds = clusters["t0"].to_numpy().astype("datetime64[s]").astype(np.int64)
    tree = cKDTree(clusters[["x", "y"]].to_numpy())
    labels = pd.read_parquet(PHASE2 / f"label_events_{year}_{NETWORK_THRESHOLD_M}m.parquet")
    official_hours: dict[int, np.ndarray] = {
        int(edge): group["event_hour"].to_numpy().astype("datetime64[s]").astype(np.int64)
        for edge, group in labels.loc[labels["official_observed"]].groupby("edge_id")
    }
    meta = ["cluster_id", "official_confirmed", "worst_injury_category", "reports", "max_confidence",
            "max_reliability", "subtype", "heading_cos"]
    rows = []
    for event in events.itertuples(index=False):
        t0 = pd.Timestamp(event.t0).floor("5min")
        if not (start + pd.Timedelta(hours=1) <= t0 < end):
            continue
        nearby = seconds[np.asarray(tree.query_ball_point([event.x, event.y], REPEAT_RADIUS_M))]
        official = official_hours.get(int(event.edge_id), np.empty(0, dtype=np.int64))
        candidates = []
        for weeks in (-4, -3, -2, -1, 1, 2, 3, 4):
            candidate = t0 + pd.Timedelta(days=7 * weeks)
            if not (start + pd.Timedelta(hours=1) <= candidate < end):
                continue
            stamp = candidate.value // 1_000_000_000
            if (np.abs(nearby - stamp) <= REPEAT_HOURS * 3600).any():
                continue
            if len(official) and ((official - stamp <= 2 * 3600) & (stamp - official <= 3 * 3600)).any():
                continue
            candidates.append(candidate)
        complete = len(candidates) >= 4
        controls = rng.choice(np.array(candidates, dtype="datetime64[ns]"), size=4, replace=False) if complete else []
        base = {name: getattr(event, name) for name in meta}
        stratum = f"waze-{year}-{event.cluster_id}"
        for buffer in BUFFERS:
            shift = pd.Timedelta(minutes=buffer)
            rows.append({**base, "kind": "waze_case", "stratum_id": stratum, "edge_id": int(event.edge_id),
                         "buffer_min": buffer, "reference_time": t0, "t_end": t0 - shift, "complete_risk_set": complete})
            for control in controls:
                control = pd.Timestamp(control)
                rows.append({"kind": "waze_control", "stratum_id": stratum, "edge_id": int(event.edge_id),
                             "buffer_min": buffer, "reference_time": control, "t_end": control - shift,
                             "complete_risk_set": True})
    frame = pd.DataFrame(rows)

    bins = int((end - start) / pd.Timedelta(minutes=5))
    marginal = pd.DataFrame({
        "kind": "marginal",
        "edge_id": rng.choice(edges["edge_id"].to_numpy(), size=marginal_size, replace=True),
        "t_end": start + pd.to_timedelta(rng.integers(0, bins, size=marginal_size) * 5, unit="min"),
    })
    marginal["reference_time"] = marginal["t_end"]

    matched = pd.read_parquet(PHASE2 / "datasets" / f"matched_edge_hours_{year}_{NETWORK_THRESHOLD_M}m.parquet",
                              columns=["stratum_id", "edge_id", "outcome_hour", "target_official"])
    crash_hours = crashes.loc[crashes["nearest_corridor_m"] <= NETWORK_THRESHOLD_M].groupby(
        ["nearest_edge", "hour_start"]).agg(waze_matched=("waze_matched", "max"),
                                            worst_injury_category=("UKATEGORIE", "min"),
                                            edge_directed=("edge_directed", "first"),
                                            direction_basis=("direction_basis", "first"))
    official = matched.rename(columns={"outcome_hour": "t_end"})
    official["t_end"] = official["t_end"].astype("datetime64[ns]")
    official["kind"] = np.where(official["target_official"] == 1, "official_case", "official_control")
    official = official.join(crash_hours, on=["edge_id", "t_end"]).drop(columns="target_official")
    official.loc[official["kind"] == "official_control", ["waze_matched", "worst_injury_category"]] = np.nan
    # The whole risk set follows its case onto the carriageway inferred for the crash.
    # Control hours were drawn for the Phase 1 edge and are not re-screened on the new one.
    official.loc[official["kind"] == "official_control", ["edge_directed", "direction_basis"]] = np.nan
    by_stratum = official.groupby("stratum_id")
    official["direction_basis"] = by_stratum["direction_basis"].transform("first")
    official["edge_id_phase1"] = official["edge_id"]
    official["edge_id"] = by_stratum["edge_directed"].transform("first").fillna(official["edge_id"]).astype(int)
    official = official.drop(columns="edge_directed")
    official["reference_time"] = official["t_end"]
    official["complete_risk_set"] = True

    result = pd.concat([frame, marginal, official], ignore_index=True)
    result["year"] = year
    result["row_id"] = [f"{year}-{index}" for index in range(len(result))]
    for column in ("t_end", "reference_time"):
        result[column] = result[column].astype("datetime64[us]")
    return result


def section_series(year: int, unit_paths: list[str]) -> pl.DataFrame | None:
    """All-year 5-min cross-section series on a complete grid (missing bins are null)."""
    numeric = ["q_kfz_h", "q_lkw_h", "v_kfz", "b", "s_kfz"]
    lanes = []
    for path in unit_paths:
        data = pl.read_csv(path, separator=";", null_values=["null", ""], try_parse_dates=True,
                           schema_overrides={"t_start": pl.Datetime})
        for column in numeric:
            if column not in data.columns:
                data = data.with_columns(pl.lit(None, dtype=pl.Float64).alias(column))
        data = data.with_columns([pl.col(column).cast(pl.Float64, strict=False) for column in numeric])
        has_speed = pl.col("v_kfz").is_not_null() & (pl.col("q_kfz_h") > 0)
        lanes.append(
            data.group_by(pl.col("t_start").cast(pl.Datetime("us")).dt.truncate("5m").alias("bin")).agg(
                (pl.col("q_kfz_h") / 60).sum().alias("n"),
                (pl.col("q_lkw_h") / 60).sum().alias("n_lkw"),
                pl.when(has_speed).then(pl.col("v_kfz") * pl.col("q_kfz_h") / 60).otherwise(0.0).sum().alias("vq"),
                pl.when(has_speed).then(pl.col("q_kfz_h") / 60).otherwise(0.0).sum().alias("nv"),
                pl.col("b").mean().alias("occ"),
                pl.col("s_kfz").mean().alias("sd"),
                pl.col("v_kfz").min().alias("vmin"),
                pl.col("q_kfz_h").is_not_null().sum().cast(pl.Float64).alias("valid"),
            )
        )
    if not lanes:
        return None
    section = pl.concat(lanes).group_by("bin").agg(
        pl.col("n").sum(), pl.col("n_lkw").sum(), pl.col("vq").sum(), pl.col("nv").sum(),
        pl.col("occ").mean(), pl.col("sd").mean(), pl.col("vmin").min(), pl.col("valid").sum(),
    )
    grid = pl.DataFrame({"bin": pl.datetime_range(datetime(year, 1, 1), datetime(year + 1, 1, 1), "5m",
                                                  closed="left", time_unit="us", eager=True)})
    return grid.join(section, on="bin", how="left").sort("bin")


def site_features(task: tuple[int, str, list[str], str]) -> tuple[pl.DataFrame | None, str, float]:
    """All-year 5-min cross-section series -> trailing window features at needed t_end."""
    year, site, unit_paths, needed_path = task
    series = section_series(year, unit_paths)
    if series is None:
        return None, site, float("nan")
    mean_flow = series["n"].mean()
    mean_flow = float("nan") if mean_flow is None else float(mean_flow)

    def trailing_sum(column: str, size: int, shift: int) -> pl.Expr:
        return pl.col(column).fill_null(0.0).rolling_sum(size).shift(shift)

    def ratio(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
        return pl.when(denominator > 0).then(numerator / denominator).otherwise(None)

    speed = ratio(pl.col("vq"), pl.col("nv"))
    series = series.with_columns(
        trailing_sum("n", 6, 1).alias("n_w"), trailing_sum("n_lkw", 6, 1).alias("nlkw_w"),
        trailing_sum("valid", 6, 1).alias("valid_w"), trailing_sum("valid", 18, 7).alias("valid_c"),
        trailing_sum("n", 18, 7).alias("n_c"),
        ratio(trailing_sum("vq", 6, 1), trailing_sum("nv", 6, 1)).alias("v_w"),
        ratio(trailing_sum("vq", 18, 7), trailing_sum("nv", 18, 7)).alias("v_c"),
        ratio(trailing_sum("vq", 3, 1), trailing_sum("nv", 3, 1)).alias("v_last"),
        ratio(trailing_sum("vq", 3, 4), trailing_sum("nv", 3, 4)).alias("v_first"),
        speed.rolling_min(6, min_samples=1).shift(1).alias("vmin5_w"),
        speed.rolling_std(6, min_samples=3).shift(1).alias("vstd_w"),
        pl.col("vmin").rolling_min(6, min_samples=1).shift(1).alias("vminlane_w"),
        pl.col("occ").rolling_mean(6, min_samples=1).shift(1).alias("occ_w"),
        pl.col("occ").rolling_max(6, min_samples=1).shift(1).alias("occmax_w"),
        pl.col("sd").rolling_mean(6, min_samples=1).shift(1).alias("sd_w"),
        pl.col("occ").rolling_mean(18, min_samples=1).shift(7).alias("occ_c"),
    ).with_columns(
        pl.when(pl.col("valid_w") > 0).then(pl.col("n_w")).otherwise(None).alias("flow_w"),
        pl.when(pl.col("valid_c") > 0).then(pl.col("n_c") / 3).otherwise(None).alias("flow_c"),
        ratio(pl.col("nlkw_w"), pl.col("n_w")).alias("heavy_w"),
        (pl.col("v_last") - pl.col("v_first")).alias("vtrend_w"),
        (pl.col("v_w") - pl.col("v_c")).alias("dv_wc"),
        (pl.col("occ_w") - pl.col("occ_c")).alias("docc_wc"),
    ).with_columns(
        pl.when(pl.col("valid_c") > 0).then(ratio(pl.col("flow_w"), pl.col("flow_c"))).otherwise(None).alias("flow_ratio"),
    )
    needed = pl.read_parquet(needed_path)
    selected = series.join(needed, left_on="bin", right_on="t_end", how="inner").select(
        pl.col("bin").alias("t_end"), *SITE_FEATURES).with_columns(pl.lit(site).alias("site"))
    return selected, site, mean_flow


def attach_features(rows: pd.DataFrame, year: int, edges: pd.DataFrame, signs: pd.DataFrame,
                    workers: int, scratch: Path) -> tuple[pd.DataFrame, dict[str, float]]:
    sign = edges["route"].map(signs.set_index("route")["travel_sign"]).fillna(1)
    lookup = edges.assign(
        site_up=np.where(sign > 0, edges["site_from"], edges["site_to"]),
        site_dn=np.where(sign > 0, edges["site_to"], edges["site_from"]),
    ).set_index("edge_id")[["route", "site_up", "site_dn", "length_straight_m"]]
    rows = rows.join(lookup, on="edge_id")
    metadata, files = read_metadata(), source_files(year)
    scratch.mkdir(parents=True, exist_ok=True)
    needed = pd.concat([rows[["site_up", "t_end"]].rename(columns={"site_up": "site"}),
                        rows[["site_dn", "t_end"]].rename(columns={"site_dn": "site"})]).drop_duplicates()
    tasks = []
    for site, group in needed.groupby("site"):
        paths = [str(files[unit]) for unit in metadata.loc[metadata["site"] == site, "unit"] if unit in files]
        if not paths:
            continue
        path = scratch / f"{year}_{site}.parquet"
        pl.from_pandas(group[["t_end"]]).with_columns(pl.col("t_end").cast(pl.Datetime("us"))).write_parquet(path)
        tasks.append((year, site, paths, str(path)))
    print(f"{year}: {len(tasks)} cross-sections, {sum(len(t[2]) for t in tasks)} lane files", flush=True)
    parts, mean_flow = [], {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for count, (part, site, flow) in enumerate(pool.map(site_features, tasks, chunksize=4), 1):
            mean_flow[site] = flow
            if part is not None:
                parts.append(part)
            if count % 200 == 0 or count == len(tasks):
                print(f"{year}: cross-sections {count}/{len(tasks)}", flush=True)
    table = pl.concat(parts).to_pandas()
    table["t_end"] = table["t_end"].astype("datetime64[us]")
    for side in ("up", "dn"):
        renamed = table.rename(columns={name: f"{side}_{name}" for name in SITE_FEATURES} | {"site": f"site_{side}"})
        rows = rows.merge(renamed, on=[f"site_{side}", "t_end"], how="left")
    for path in scratch.glob(f"{year}_*.parquet"):
        path.unlink()
    return rows, mean_flow


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--marginal", type=int, default=300_000)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    edges = load_edges()
    signs = route_travel_signs(edges)
    signs.to_csv(OUT / "route_travel_signs.csv", index=False)
    summary = {"buffers_min": BUFFERS, "route_sign_basis": signs["basis"].value_counts().to_dict(), "years": {}}
    frames, flows = {}, {}
    field = heading_field()
    for year in (2024, 2025):
        events, clusters, crashes, counts = waze_events(year, edges, signs)
        events = events.loc[~events["repeat_of_earlier_report"]].reset_index(drop=True)
        crashes, direction_summary = official_directions(crashes, edges, signs, field)
        rows = build_rows(year, events, clusters, crashes, edges, args.marginal)
        rows, flows[year] = attach_features(rows, year, edges, signs, args.workers, OUT / "_needed")
        frames[year] = rows
        start, end = eligible_window(year)
        cases = rows.loc[(rows["kind"] == "waze_case") & (rows["buffer_min"] == BUFFERS[0])]
        official_cases = rows.loc[rows["kind"] == "official_case"]
        summary["years"][str(year)] = {
            **counts,
            "official_direction": direction_summary,
            "eligible_window": [start.isoformat(), end.isoformat()],
            "population_edge_5min_units": int(len(edges) * ((end - start) / pd.Timedelta(minutes=5))),
            "waze_timed_events": int(len(cases)),
            "waze_timed_police_confirmed": int(cases["official_confirmed"].sum()),
            "waze_timed_events_with_complete_risk_set": int(cases["complete_risk_set"].sum()),
            "official_case_edge_hours": int(len(official_cases)),
            "official_case_edge_hours_waze_matched": int(official_cases["waze_matched"].fillna(False).astype(bool).sum()),
            "rows_by_kind": rows["kind"].value_counts().to_dict(),
            "rows_with_any_upstream_traffic": float(rows["up_flow_w"].notna().mean()),
            "rows_with_any_downstream_traffic": float(rows["dn_flow_w"].notna().mean()),
        }
    # Static exposure proxy from the training year only.
    static = pd.Series(flows[2024], name="mean_flow_2024")
    for year, rows in frames.items():
        rows["up_mean_flow_2024"] = rows["site_up"].map(static)
        rows["dn_mean_flow_2024"] = rows["site_dn"].map(static)
        rows.to_parquet(OUT / f"minute_dataset_{year}.parquet", index=False)
    scratch = OUT / "_needed"
    if scratch.exists() and not any(scratch.iterdir()):
        scratch.rmdir()
    (OUT / "minute_dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
