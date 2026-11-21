"""Spearman rank correlation between police-recorded crashes and loop-detector measurements.

Two descriptive views for the Data chapter, each computed separately for 2024 and 2025:
  * spatial: one row per detector edge, crash count / crash density against the annual
    detector summary of the edge (mean of its two bounding cross-sections);
  * temporal: one row per hour of the week (168 cells), network crash count against the
    network-mean detector profile of that hour.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
SITE_CACHE = ROOT / "analysis" / "cache" / "v2" / "traffic_site_hourly"
EDGES = ROOT / "analysis" / "results" / "v2" / "phase1" / "detector_corridor_edges.csv"
LABELS = ROOT / "analysis" / "results" / "v2" / "phase2" / "label_events_{year}_100m.parquet"
ASSIGNMENT = ROOT / "analysis" / "results" / "v2" / "phase1" / "crash_network_assignment_{year}.csv"
ATTRIBUTES = ROOT / "accidents" / "Unfallorte{year}_standardized.csv"
THRESHOLD_M = 100
OUT = ROOT / "analysis" / "results" / "v2" / "data_spearman"

MIN_VALID_SHARE = 0.5  # a site-year enters only with valid flow in at least half of its hours
CONGESTED_KMH = 60.0
VARIABLES = {
    "mean_flow": "Mean hourly volume (veh/h)",
    "hgv_share": "HGV share",
    "mean_speed": "Mean speed (km/h)",
    "mean_occupancy": "Mean occupancy (%)",
    "mean_speed_sd": "Mean within-minute speed SD (km/h)",
    "mean_net_gap": "Mean net time gap (s)",
    "congested_share": f"Share of hours with v < {CONGESTED_KMH:.0f} km/h",
}
# crash subgroup -> rule on the Unfallatlas record (codes in Appendix tab:unfallatlas_variables)
SUBGROUPS = {
    "rear_end": lambda r: r["UART"] == 2,             # collision with vehicle moving ahead or waiting
    "sideswipe": lambda r: r["UART"] == 3,            # collision with vehicle moving laterally, same direction
    "run_off_road": lambda r: r["UART"].isin([8, 9]),  # leaving the carriageway to the right or left
    "goods_vehicle": lambda r: r["IstGkfz"] == 1,
    "darkness": lambda r: r["ULICHTVERH"] == 2,
    "wet_or_winter_road": lambda r: r["USTRZUSTAND"] >= 1,
}


def crash_records(year: int) -> pd.DataFrame:
    """Network crashes of the label cohort (within THRESHOLD_M of an edge) with their subgroup flags."""
    assigned = pd.read_csv(Path(str(ASSIGNMENT).format(year=year)), dtype={"uid": str})
    assigned = assigned[assigned["nearest_corridor_m"] <= THRESHOLD_M]
    attributes = pd.read_csv(Path(str(ATTRIBUTES).format(year=year)), sep=";", dtype={"uid": str},
                             usecols=["uid", "UART", "IstGkfz", "ULICHTVERH", "USTRZUSTAND"])
    records = assigned[["uid", "date", "USTUNDE", "nearest_edge"]].merge(attributes, on="uid", how="left")
    hour = pd.to_datetime(records["date"]) + pd.to_timedelta(records["USTUNDE"], unit="h")
    flags = {name: rule(records).astype(int) for name, rule in SUBGROUPS.items()}
    return pd.DataFrame({"edge_id": records["nearest_edge"].astype(int),
                         "how": hour.dt.dayofweek * 24 + hour.dt.hour, **flags})


def site_hours(year: int, sites: set[str]) -> pd.DataFrame:
    frames = []
    for site in sorted(sites):
        path = SITE_CACHE / str(year) / f"{site}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    data = pd.concat(frames, ignore_index=True)
    return data[data["valid_lane_minutes_flow"] > 0]


def site_summary(hours: pd.DataFrame) -> pd.DataFrame:
    grouped = hours.groupby("site")
    weighted_speed = (hours["v_kfz"] * hours["n_kfz"]).groupby(hours["site"]).sum()
    speed_volume = hours["n_kfz"].where(hours["v_kfz"].notna(), 0).groupby(hours["site"]).sum()
    summary = pd.DataFrame({
        "valid_hours": grouped.size(),
        "mean_flow": grouped["n_kfz"].mean(),
        "hgv_share": grouped["n_lkw"].sum() / grouped["n_kfz"].sum(),
        "mean_speed": weighted_speed / speed_volume,
        "mean_occupancy": grouped["occupancy_mean"].mean(),
        "mean_speed_sd": grouped["speed_sd_mean"].mean(),
        "mean_net_gap": grouped["net_gap_mean"].mean(),
        "congested_share": (hours["v_kfz"] < CONGESTED_KMH).groupby(hours["site"]).mean(),
    })
    hours_in_year = hours["feature_hour"].dt.year.map(lambda y: 8784 if y % 4 == 0 else 8760).iloc[0]
    return summary[summary["valid_hours"] >= MIN_VALID_SHARE * hours_in_year]


def edge_table(year: int, edges: pd.DataFrame, sites: pd.DataFrame, labels: pd.DataFrame,
               records: pd.DataFrame) -> pd.DataFrame:
    values = []
    for column in VARIABLES:
        pair = pd.concat([edges["site_from"].map(sites[column]), edges["site_to"].map(sites[column])], axis=1)
        values.append(pair.mean(axis=1).rename(column))
    table = pd.concat([edges[["edge_id", "route", "length_km_post_m"]], *values], axis=1)
    official = labels[labels["official_observed"]]
    crashes = official.groupby("edge_id")["official_crashes"].sum()
    table["crashes"] = table["edge_id"].map(crashes).fillna(0).astype(int)
    table["crashes_per_km"] = table["crashes"] / (table["length_km_post_m"] / 1000)
    subgroups = records.groupby("edge_id")[list(SUBGROUPS)].sum()
    for name in SUBGROUPS:
        table[name] = table["edge_id"].map(subgroups[name]).fillna(0).astype(int)
        table[f"{name}_per_km"] = table[name] / (table["length_km_post_m"] / 1000)
    table["year"] = year
    return table.dropna(subset=list(VARIABLES))


def hour_of_week_table(year: int, hours: pd.DataFrame, labels: pd.DataFrame, records: pd.DataFrame) -> pd.DataFrame:
    hours = hours.assign(how=hours["feature_hour"].dt.dayofweek * 24 + hours["feature_hour"].dt.hour)
    profile = pd.DataFrame({
        "mean_flow": hours.groupby("how")["n_kfz"].mean(),
        "hgv_share": hours.groupby("how")["n_lkw"].sum() / hours.groupby("how")["n_kfz"].sum(),
        "mean_speed": (hours["v_kfz"] * hours["n_kfz"]).groupby(hours["how"]).sum()
        / hours["n_kfz"].where(hours["v_kfz"].notna(), 0).groupby(hours["how"]).sum(),
        "mean_occupancy": hours.groupby("how")["occupancy_mean"].mean(),
        "mean_speed_sd": hours.groupby("how")["speed_sd_mean"].mean(),
        "mean_net_gap": hours.groupby("how")["net_gap_mean"].mean(),
        "congested_share": (hours["v_kfz"] < CONGESTED_KMH).groupby(hours["how"]).mean(),
    })
    official = labels[labels["official_observed"]]
    event = pd.to_datetime(official["event_hour"])
    crashes = official["official_crashes"].groupby(event.dt.dayofweek * 24 + event.dt.hour).sum()
    profile["crashes"] = profile.index.map(crashes).fillna(0).astype(int)
    subgroups = records.groupby("how")[list(SUBGROUPS)].sum()
    for name in SUBGROUPS:
        profile[name] = profile.index.map(subgroups[name]).fillna(0).astype(int)
    profile["year"] = year
    return profile.reset_index()


def correlate(table: pd.DataFrame, targets: list[str], level: str) -> list[dict]:
    rows = []
    for year, group in table.groupby("year"):
        for target in targets:
            for column, label in VARIABLES.items():
                rho, p = spearmanr(group[column], group[target])
                rows.append({"level": level, "year": int(year), "target": target, "variable": column,
                             "label": label, "n": len(group), "rho": float(rho), "p_value": float(p)})
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    edges = pd.read_csv(EDGES)
    needed = set(edges["site_from"]) | set(edges["site_to"])
    spatial, temporal, coverage = [], [], {}
    for year in (2024, 2025):
        labels = pd.read_parquet(Path(str(LABELS).format(year=year)))
        records = crash_records(year)
        hours = site_hours(year, needed)
        sites = site_summary(hours)
        hours = hours[hours["site"].isin(sites.index)]
        spatial.append(edge_table(year, edges, sites, labels, records))
        temporal.append(hour_of_week_table(year, hours, labels, records))
        coverage[year] = {"sites_used": int(len(sites)), "edges_used": int(len(spatial[-1])),
                          "edges_total": int(len(edges)),
                          "crashes_on_used_edges": int(spatial[-1]["crashes"].sum()),
                          "crashes_on_network": int(labels.loc[labels["official_observed"], "official_crashes"].sum()),
                          "edges_with_crash": int((spatial[-1]["crashes"] > 0).sum()),
                          "crash_records": int(len(records)),
                          "subgroups_on_network": {name: int(records[name].sum()) for name in SUBGROUPS},
                          "subgroups_on_used_edges": {name: int(spatial[-1][name].sum()) for name in SUBGROUPS}}
    spatial = pd.concat(spatial, ignore_index=True)
    temporal = pd.concat(temporal, ignore_index=True)
    spatial.to_csv(OUT / "edge_year_table.csv", index=False)
    temporal.to_csv(OUT / "hour_of_week_table.csv", index=False)
    results = pd.DataFrame(correlate(spatial, ["crashes", "crashes_per_km"] + [f"{n}_per_km" for n in SUBGROUPS], "edge")
                           + correlate(temporal, ["crashes"] + list(SUBGROUPS), "hour_of_week"))
    results.to_csv(OUT / "spearman.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(coverage, indent=2))
    with pd.option_context("display.width", 160, "display.max_rows", 100):
        print(json.dumps(coverage, indent=2))
        print(results.pivot_table(index=["level", "target", "variable"], columns="year",
                                  values=["rho", "p_value"]).round(4))


if __name__ == "__main__":
    main()
