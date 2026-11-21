"""Version 2 of the analysis chain: cleaned network, carriageway-resolved labels.

Everything is written under results/v2 and cache/v2; version 1 artifacts stay
untouched. Three things change relative to version 1:

1. Network. Only main-carriageway cross-sections become nodes (ramp, parking
   and auxiliary detectors are excluded), each site belongs to one route, sites
   at the same kilometre are merged, and consecutive nodes are at least 200 m
   apart. Edges whose chord length contradicts the kilometre gap are dropped.
2. Carriageway. Police crashes take the edge on the carriageway inferred from
   their coordinates (heading of nearby Waze reports, Phase 6); Waze clusters
   take it from their own heading. Matched controls are then drawn on those
   edges, so exclusion rules apply to the carriageway actually used.
3. Hourly cross-section speeds. A lane without vehicles in an hour no longer
   turns the cross-section speed into a missing value.

Steps `network`, `cohort` and `traffic` build the new inputs. The remaining
steps run the existing phase scripts unchanged, with their result directories
redirected to v2.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import shapely
from shapely.strtree import STRtree

import phase1_network_scope as scope
import phase2_aggregate_traffic as channel_cache
import phase2_baseline
import phase2_bootstrap
import phase2_build_dataset
import phase2_label_cohort
import phase2_label_comparison
import phase5_weather
import phase5_weather_eval
import phase6_minute_dataset
import phase7_frequency_prior


ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "analysis" / "results"
V2 = V1 / "v2"
CACHE_V1 = ROOT / "analysis" / "cache"
CACHE_V2 = CACHE_V1 / "v2"
YEARS = (2024, 2025)
THRESHOLD_M = 100
MAIN_CODES = {"H", "DE", "UFS", "HFS", "DET", "D"}  # main-carriageway lane detectors
MIN_SPACING_KM = 0.2
MAX_KM_GAP = 15.0
STEPS = ("network", "cohort", "traffic", "phase2", "phase5", "phase6", "phase7", "phase8")


def modelling_modules():
    # Imported on demand: these load torch, which worker processes must not pay for.
    import phase6_minute_pu
    import phase7_two_stage
    import phase8_pu_value
    return phase6_minute_pu, phase7_two_stage, phase8_pu_value


def redirect() -> None:
    """Point the phase scripts at the v2 directories."""
    phase6_minute_dataset.PHASE1 = V2 / "phase1"
    phase6_minute_dataset.PHASE2 = V2 / "phase2"
    phase6_minute_dataset.OUT = V2 / "phase6"
    phase2_build_dataset.PHASE1 = V2 / "phase1"
    phase2_build_dataset.PHASE2 = V2 / "phase2"
    phase2_build_dataset.SITE_CACHE = CACHE_V2 / "traffic_site_hourly"
    phase2_build_dataset.OUT = V2 / "phase2" / "datasets"
    phase2_baseline.DATA = V2 / "phase2" / "datasets"
    phase2_baseline.OUT = V2 / "phase2" / "baseline"
    phase2_label_comparison.DATA = V2 / "phase2" / "datasets"
    phase2_label_comparison.OUT = V2 / "phase2" / "label_comparison"
    phase2_bootstrap.BASE = V2 / "phase2" / "label_comparison"
    phase5_weather.PHASE1 = V2 / "phase1"
    phase5_weather.PHASE2_DATA = V2 / "phase2" / "datasets"
    phase5_weather.OUT = V2 / "phase5"
    phase5_weather_eval.DATA = V2 / "phase5" / "datasets"
    phase5_weather_eval.PHASE1 = V2 / "phase1"
    phase5_weather_eval.OUT = V2 / "phase5" / "weather_eval"
    phase7_frequency_prior.PHASE6 = V2 / "phase6"
    phase7_frequency_prior.OUT = V2 / "phase7"


def redirect_modelling():
    phase6_minute_pu, phase7_two_stage, phase8_pu_value = modelling_modules()
    phase6_minute_pu.DATA = V2 / "phase6"
    phase6_minute_pu.OUT = V2 / "phase6" / "minute_pu"
    phase6_minute_pu.HOURLY_PREDICTIONS = V2 / "phase2" / "label_comparison" / "test_predictions_2025.parquet"
    phase7_two_stage.PHASE6 = V2 / "phase6"
    phase7_two_stage.OUT = V2 / "phase7"
    phase8_pu_value.DATA = V2 / "phase6"
    phase8_pu_value.OUT = V2 / "phase8"
    return phase6_minute_pu, phase7_two_stage, phase8_pu_value


def call(module, *arguments: str) -> None:
    sys.argv = [module.__name__, *arguments]
    print(f"--- {module.__name__} {' '.join(arguments)}", flush=True)
    module.main()


# ---------------------------------------------------------------- network

def detector_units() -> pd.DataFrame:
    """One row per lane unit with its listed routes and a main-carriageway flag."""
    rows = []
    for line in scope.META.read_text(encoding="utf-8-sig").splitlines()[1:]:
        fields = line.split(";")
        if len(fields) < 7:
            continue
        routes = list(dict.fromkeys(scope.ROUTE_PATTERN.findall(";".join(fields[3:-3]))))
        if not routes:
            continue
        km, lon, lat = (float(value.replace(",", ".")) for value in fields[-3:])
        code = re.search(r"([A-Za-z]+)(\d*)$", fields[1])
        name, number = (code.group(1), code.group(2)) if code else ("", "")
        # TLS channels 1-64 are the two main directions; higher channels are ramps or auxiliaries.
        main = name in MAIN_CODES and not (name == "DE" and number and int(number) >= 65)
        rows.append({"site": fields[0], "unit": fields[1], "routes": routes, "km": km, "lon": lon, "lat": lat, "main": main})
    return pd.DataFrame(rows)


def site_route(site: str, routes: list[str]) -> tuple[str, str]:
    """A junction site lists several routes; its own name says which one it measures."""
    if len(routes) == 1:
        return routes[0], "single listed route"
    road = re.match(r"^[A-Z]+_?(\d+)[-_]", site)
    for route in routes:
        head, *destinations = route.split("_")
        if road and re.fullmatch(rf"A{road.group(1)}(-[A-Za-z]+)?", head) and any(
                re.search(rf"[-_]{destination}(_|$)", site) for destination in destinations):
            return route, "site name"
    return routes[0], "first listed route"


def consistent_chain(nodes: pd.DataFrame) -> list[int]:
    """Longest kilometre-ordered sequence of sites whose spacing agrees with their coordinates.

    Some sites carry a kilometre value from another road or a junction-local
    count. A chord cannot be longer than the road between two sites, and a road
    far longer than its chord contradicts the kilometre sequence; sites that do
    not fit the longest consistent sequence are left out.
    """
    x, y = scope.xy_m(nodes["lon"].to_numpy(), nodes["lat"].to_numpy())
    km = nodes["km"].to_numpy()
    best, previous = np.ones(len(nodes), dtype=int), np.full(len(nodes), -1)
    for j in range(len(nodes)):
        for i in range(j):
            gap_m, straight = (km[j] - km[i]) * 1000, float(np.hypot(x[j] - x[i], y[j] - y[i]))
            if straight <= 1.2 * gap_m + 200 and gap_m <= 3 * straight + 500 and best[i] + 1 > best[j]:
                best[j], previous[j] = best[i] + 1, i
    chain, index = [], int(np.argmax(best)) if len(nodes) else -1
    while index >= 0:
        chain.append(index)
        index = int(previous[index])
    return chain[::-1]


def build_network() -> dict:
    units = detector_units()
    main_units = units.loc[units["main"]]
    sites = main_units.groupby("site").agg(routes=("routes", "first"), km=("km", "first"), lon=("lon", "first"),
                                           lat=("lat", "first"), lanes=("unit", "nunique")).reset_index()
    chosen = [site_route(site, routes) for site, routes in zip(sites["site"], sites["routes"])]
    sites["route"], sites["route_basis"] = [c[0] for c in chosen], [c[1] for c in chosen]
    counts = {"metadata_sites": int(units["site"].nunique()), "main_carriageway_sites": int(len(sites)),
              "sites_listing_several_routes": int((sites["routes"].map(len) > 1).sum()),
              "route_basis": sites["route_basis"].value_counts().to_dict(),
              "merged_same_km": 0, "sites_off_the_kilometre_sequence": [], "skipped_closer_than_200m": 0,
              "edges_dropped_km_gap": 0}
    rows, kept_sites = [], []
    for route, group in sites.groupby("route"):
        # Same kilometre: keep the cross-section with most lanes.
        group = group.sort_values(["km", "lanes", "site"], ascending=[True, False, True])
        merged = group.drop_duplicates("km").reset_index(drop=True)
        counts["merged_same_km"] += len(group) - len(merged)
        chain = consistent_chain(merged)
        counts["sites_off_the_kilometre_sequence"] += merged.loc[~merged.index.isin(chain), "site"].tolist()
        nodes, last_km = [], -np.inf
        for node in merged.loc[chain].itertuples(index=False):
            if node.km - last_km < MIN_SPACING_KM:
                counts["skipped_closer_than_200m"] += 1
                continue
            nodes.append(node)
            last_km = node.km
        kept_sites += [node.site for node in nodes]
        x, y = scope.xy_m(np.array([node.lon for node in nodes]), np.array([node.lat for node in nodes]))
        for index in range(len(nodes) - 1):
            km_gap = nodes[index + 1].km - nodes[index].km
            straight = float(np.hypot(x[index + 1] - x[index], y[index + 1] - y[index]))
            if km_gap > MAX_KM_GAP or straight > 25_000:
                counts["edges_dropped_km_gap"] += 1
                continue
            rows.append({"route": route, "site_from": nodes[index].site, "site_to": nodes[index + 1].site,
                         "km_from": nodes[index].km, "km_to": nodes[index + 1].km, "length_straight_m": straight,
                         "length_km_post_m": km_gap * 1000,
                         "geometry_wkt": shapely.LineString([(x[index], y[index]), (x[index + 1], y[index + 1])]).wkt})
    edges = pd.DataFrame(rows)
    edges.index.name = "edge_id"
    out = V2 / "phase1"
    out.mkdir(parents=True, exist_ok=True)
    edges.to_csv(out / "detector_corridor_edges.csv")
    main_units.loc[main_units["site"].isin(kept_sites), ["site", "unit"]].to_csv(out / "main_carriageway_units.csv", index=False)
    for name in [f"waze_accident_clusters_{year}.csv" for year in YEARS] + ["source_summary.json"]:
        shutil.copyfile(V1 / "phase1" / name, out / name)
    pair = [tuple(sorted(item)) for item in zip(edges["site_from"], edges["site_to"])]
    counts.update({
        "edges": int(len(edges)), "routes": int(edges["route"].nunique()),
        "duplicate_geometries": int(edges["geometry_wkt"].duplicated().sum()),
        "duplicate_site_pairs": int(pd.Series(pair).duplicated().sum()),
        "edge_length_m_quantiles_0_10_50_90_100": [float(edges["length_straight_m"].quantile(q)) for q in (0, 0.1, 0.5, 0.9, 1)],
        "network_length_km_both_directions": float(edges["length_km_post_m"].sum() / 1000),
    })
    return counts


# ----------------------------------------------------------------- cohort

def build_cohort() -> dict:
    """Carriageway-resolved crash assignment and hourly label events on the v2 network."""
    edges = phase6_minute_dataset.load_edges()
    signs = phase6_minute_dataset.route_travel_signs(edges)
    field = phase6_minute_dataset.heading_field()
    geometry = shapely.from_wkt(edges["geometry_wkt"].to_numpy())
    tree = STRtree(geometry)
    lookup = edges.set_index("edge_id")
    summary = {}
    (V2 / "phase2").mkdir(parents=True, exist_ok=True)
    for year in YEARS:
        crashes = scope.load_bavaria_crashes(year).reset_index(drop=True)
        x, y = scope.xy_m(crashes["lon"].to_numpy(), crashes["lat"].to_numpy())
        points = shapely.points(x, y)
        (row, nearest), distance = tree.query_nearest(points, return_distance=True, all_matches=False)
        crashes["nearest_edge_undirected"], crashes["nearest_corridor_m"] = -1, np.nan
        crashes.loc[row, "nearest_edge_undirected"] = edges["edge_id"].to_numpy()[nearest]
        crashes.loc[row, "nearest_corridor_m"] = distance
        crashes["nearest_edge"] = crashes["nearest_edge_undirected"]
        crashes["hour_start"] = (pd.to_datetime(crashes["date"]) + pd.to_timedelta(crashes["USTUNDE"], unit="h")).astype("datetime64[ns]")
        crashes["crash_row"] = np.arange(len(crashes))
        directed, direction = phase6_minute_dataset.official_directions(crashes, edges, signs, field)
        crashes["direction_basis"] = ""
        crashes.loc[directed["crash_row"], "nearest_edge"] = directed["edge_directed"].to_numpy()
        crashes.loc[directed["crash_row"], "direction_basis"] = directed["direction_basis"].to_numpy()
        # Basis-DLM membership (official_directions -> dlm_assign): scaled corridor distance, NaN off-network.
        on_network = directed["crash_row"].to_numpy()
        crashes["nearest_corridor_m"] = np.nan
        crashes.loc[on_network, "nearest_corridor_m"] = directed["nearest_corridor_m"].to_numpy()
        for column, source in (("nearest_route", "route"), ("site_from", "site_from"), ("site_to", "site_to")):
            crashes[column] = crashes["nearest_edge"].map(lookup[source])
        crashes[["year", "uid", "date", "USTUNDE", "UKATEGORIE", "lon", "lat", "nearest_route", "nearest_edge",
                 "site_from", "site_to", "nearest_corridor_m", "nearest_edge_undirected", "direction_basis"]].to_csv(
            V2 / "phase1" / f"crash_network_assignment_{year}.csv", index=False)

        events, _, _, waze_counts = phase6_minute_dataset.waze_events(year, edges, signs)
        waze = events.rename(columns={"distance_m": "corridor_distance_m"})
        waze["event_hour"] = waze["t0"].dt.floor("h")
        official = crashes.loc[crashes["nearest_corridor_m"] <= THRESHOLD_M].rename(
            columns={"nearest_corridor_m": "corridor_distance_m", "nearest_edge": "edge_id", "hour_start": "event_hour"})
        labels = phase2_label_cohort.aggregate_labels(official, waze, THRESHOLD_M)
        labels.to_parquet(V2 / "phase2" / f"label_events_{year}_{THRESHOLD_M}m.parquet", index=False)
        summary[str(year)] = {
            "police_crashes_within_100m": int(len(official)), "direction": direction, "waze": waze_counts,
            "event_edge_hours": int(len(labels)),
            "source_patterns": {str(key): int(value) for key, value in labels["source_pattern"].value_counts().items()},
        }
    (V2 / "phase2" / "label_cohort_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


# ---------------------------------------------------------------- traffic

def site_hourly(task: tuple[str, list[str], str]) -> str:
    site, paths, output = task
    data = pl.concat([pl.read_parquet(path) for path in paths], how="vertical_relaxed")
    flows = {"v_kfz": "n_kfz", "v_lkw": "n_lkw", "v_pkw": "n_pkw"}
    data = data.with_columns([pl.col(column).fill_nan(None) for column in
                              (*flows, "occupancy_mean", "net_gap_mean", "speed_sd_mean")])

    def weighted(speed: str, flow: str) -> pl.Expr:
        usable = pl.col(speed).is_not_null() & (pl.col(flow) > 0)
        total = pl.when(usable).then(pl.col(flow)).otherwise(0.0).sum()
        return pl.when(total > 0).then(pl.when(usable).then(pl.col(speed) * pl.col(flow)).otherwise(0.0).sum() / total).otherwise(None).alias(speed)

    hourly = data.group_by("feature_hour").agg(
        pl.col("n_kfz").sum(), pl.col("n_lkw").sum(), pl.col("n_pkw").sum(),
        *[weighted(speed, flow) for speed, flow in flows.items()],
        pl.col("occupancy_mean").mean(), pl.col("net_gap_mean").mean(), pl.col("speed_sd_mean").mean(),
        pl.col("valid_minutes_flow").sum().alias("valid_lane_minutes_flow"),
        pl.col("valid_minutes_speed").sum().alias("valid_lane_minutes_speed"),
        pl.col("unit").n_unique().alias("lane_channels"),
    ).sort("feature_hour").with_columns(pl.lit(site).alias("site"))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    hourly.write_parquet(output, compression="zstd")
    return site


def build_traffic(workers: int) -> dict:
    """Hourly cross-section features from main-carriageway lanes, speed aggregation corrected."""
    edges = pd.read_csv(V2 / "phase1" / "detector_corridor_edges.csv")
    needed = set(edges["site_from"]) | set(edges["site_to"])
    units = pd.read_csv(V2 / "phase1" / "main_carriageway_units.csv")
    units = units.loc[units["site"].isin(needed)]
    summary = {}
    for year in YEARS:
        sources = channel_cache.source_files(year)
        located, missing, to_build = {}, [], []
        for unit in units["unit"]:
            for folder in (CACHE_V1 / "traffic_hourly" / str(year), CACHE_V1 / "phase3_population" / "channels" / str(year),
                           CACHE_V2 / "channels" / str(year)):
                if (folder / f"{unit}.parquet").exists():
                    located[unit] = folder / f"{unit}.parquet"
                    break
            else:
                if unit in sources:
                    located[unit] = CACHE_V2 / "channels" / str(year) / f"{unit}.parquet"
                    to_build.append((unit, str(sources[unit]), str(located[unit])))
                else:
                    missing.append(unit)
        tasks = []
        for site, group in units.groupby("site"):
            paths = [str(located[unit]) for unit in group["unit"] if unit in located]
            if paths:
                tasks.append((site, paths, str(CACHE_V2 / "traffic_site_hourly" / str(year) / f"{site}.parquet")))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            list(pool.map(channel_cache.aggregate_one, to_build, chunksize=4))
            done = list(pool.map(site_hourly, tasks, chunksize=8))
        summary[str(year)] = {"sites_needed": len(needed), "sites_written": len(done), "lane_channels_built": len(to_build),
                              "lane_channels_without_raw_file": missing}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", nargs="+", choices=STEPS, default=list(STEPS))
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    redirect()
    log_path = V2 / "v2_build_summary.json"
    V2.mkdir(parents=True, exist_ok=True)
    log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {}
    if "network" in args.steps:
        log["network"] = build_network()
        print(json.dumps(log["network"], indent=2), flush=True)
    if "cohort" in args.steps:
        log["cohort"] = build_cohort()
        print(json.dumps(log["cohort"], indent=2), flush=True)
    if "traffic" in args.steps:
        log["traffic"] = build_traffic(args.workers)
        print(json.dumps(log["traffic"], indent=2), flush=True)
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    if "phase2" in args.steps:
        for module in (phase2_build_dataset, phase2_baseline, phase2_label_comparison, phase2_bootstrap):
            call(module)
    if "phase5" in args.steps:
        call(phase5_weather, "--workers", str(args.workers))
        call(phase5_weather_eval, "--replicates", "1000")
    if "phase6" in args.steps:
        call(phase6_minute_dataset, "--workers", str(args.workers))
        call(redirect_modelling()[0])
    if "phase7" in args.steps:
        call(phase7_frequency_prior)
        call(redirect_modelling()[1])
    if "phase8" in args.steps:
        call(redirect_modelling()[2])


if __name__ == "__main__":
    main()
