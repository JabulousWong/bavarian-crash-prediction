"""Stratified Waze--Unfallatlas matching descriptives. Read-only inputs.

Uses the loaders, clustering and matching rule of analysis/phase1_match_audit.py
(500 m, report time within [-0.25, +1.75] h of the start of the published crash
hour) and only adds the breakdowns quoted in the thesis: detection of police
injury crashes by severity, road-user involvement, hour of day and membership
of the version 2 study network; the delay of the first matching report; and the
share of Waze accident reports and clusters with a police injury record by
subtype, confidence and cluster size.
"""
import argparse, json, pathlib, sys
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import phase1_match_audit as audit  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--output", default=str(pathlib.Path(__file__).resolve().parent / "audit_results"))
parser.add_argument("--radius", type=float, default=500)
args = parser.parse_args()
out = pathlib.Path(args.output)
out.mkdir(parents=True, exist_ok=True)
LOWER_H, UPPER_H = -0.25, 1.75
NETWORK = ROOT / "analysis" / "results" / "v2" / "phase1"


def match(crashes, reports, radius):
    """Per-crash and per-report match flags plus the delay of the first matching report."""
    cx, cy = audit.xy_m(crashes["lon"].to_numpy(), crashes["lat"].to_numpy())
    wx, wy = audit.xy_m(reports["lon"].to_numpy(), reports["lat"].to_numpy())
    report_s = audit.wall_datetime_ns(reports["ts_local"]) / 1e9
    crash_s = audit.wall_datetime_ns(crashes["hour_start_wall"]) / 1e9
    crash_hit, report_hit = np.zeros(len(crashes), bool), np.zeros(len(reports), bool)
    first_delay = np.full(len(crashes), np.nan)
    for i, candidates in enumerate(cKDTree(np.column_stack([wx, wy])).query_ball_point(np.column_stack([cx, cy]), radius)):
        if not candidates:
            continue
        candidates = np.asarray(candidates, dtype=int)
        delta = report_s[candidates] - crash_s[i]
        keep = (delta >= LOWER_H * 3600) & (delta <= UPPER_H * 3600)
        if keep.any():
            crash_hit[i], report_hit[candidates[keep]] = True, True
            first_delay[i] = delta[keep].min() / 60
    return crash_hit, report_hit, first_delay


def rate(mask, hit):
    n = int(mask.sum())
    return {"n": n, "matched": int(hit[mask].sum()), "rate": float(hit[mask].mean()) if n else None}


results = {"rule": {"radius_m": args.radius, "window_h": [LOWER_H, UPPER_H]}, "years": {}}
for year in audit.YEARS:
    waze, _ = audit.load_waze(year)
    crashes, _, _ = audit.load_crashes(year)
    accidents = waze.loc[(waze["type"] == "ACCIDENT") & waze["ts_local"].notna() & waze["lon"].notna()].reset_index(drop=True)
    crash_hit, report_hit, delay = match(crashes, accidents, args.radius)
    for column in ("UKATEGORIE", "IstRad", "IstPKW", "IstFuss", "IstKrad", "IstGkfz", "IstSonstige"):
        crashes[column] = pd.to_numeric(crashes[column], errors="coerce")
    network = pd.read_csv(NETWORK / f"crash_network_assignment_{year}.csv", dtype={"uid": str})
    on_network = crashes["uid"].isin(network.loc[network["nearest_corridor_m"] <= 100, "uid"]).to_numpy()
    everything = np.ones(len(crashes), bool)
    forward = {"all_bavaria": rate(everything, crash_hit)}
    for code, name in ((1, "fatal"), (2, "serious"), (3, "slight")):
        forward[f"bavaria_{name}"] = rate((crashes["UKATEGORIE"] == code).to_numpy(), crash_hit)
        forward[f"network_{name}"] = rate(on_network & (crashes["UKATEGORIE"] == code).to_numpy(), crash_hit)
    forward["study_network_100m"] = rate(on_network, crash_hit)
    forward["off_network"] = rate(~on_network, crash_hit)
    for column, name in (("IstGkfz", "goods_vehicle"), ("IstPKW", "passenger_car"), ("IstKrad", "motorcycle"),
                         ("IstFuss", "pedestrian"), ("IstRad", "bicycle")):
        forward[f"involved_{name}"] = rate((crashes[column] == 1).to_numpy(), crash_hit)
    by_hour = {str(hour): rate((crashes["hour"] == hour).to_numpy(), crash_hit) for hour in range(24)}
    by_hour_network = {str(hour): rate(on_network & (crashes["hour"] == hour).to_numpy(), crash_hit) for hour in range(24)}

    def quantiles(values):
        values = values[~np.isnan(values)]
        return {"n": int(len(values)), **{f"q{int(q * 100)}": float(np.quantile(values, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}}

    confidence = pd.to_numeric(accidents["confidence"], errors="coerce").fillna(0).to_numpy()
    subtype = accidents["subtype"].fillna("").str.strip().to_numpy()
    reverse = {"all_reports": rate(np.ones(len(accidents), bool), report_hit),
               "subtype_major": rate(subtype == "ACCIDENT_MAJOR", report_hit),
               "subtype_minor": rate(subtype == "ACCIDENT_MINOR", report_hit),
               "subtype_unspecified": rate(subtype == "", report_hit),
               "confidence_0": rate(confidence == 0, report_hit),
               "confidence_ge_1": rate(confidence >= 1, report_hit),
               "confidence_ge_2": rate(confidence >= 2, report_hit)}
    clusters = audit.cluster_reports(accidents)
    _, cluster_hit, _ = match(crashes, clusters.rename(columns={"first_ts_local": "ts_local"}), args.radius)
    size = clusters["reports"].to_numpy()
    reverse_clusters = {"all_clusters": rate(np.ones(len(clusters), bool), cluster_hit),
                        "reports_1": rate(size == 1, cluster_hit), "reports_2": rate(size == 2, cluster_hit),
                        "reports_3_4": rate((size >= 3) & (size <= 4), cluster_hit),
                        "reports_ge_5": rate(size >= 5, cluster_hit)}
    results["years"][str(year)] = {
        "crashes": int(len(crashes)), "accident_reports": int(len(accidents)), "clusters": int(len(clusters)),
        "forward": forward, "forward_by_hour": by_hour, "forward_by_hour_network": by_hour_network,
        "first_report_minutes_after_crash_hour_start": {"bavaria": quantiles(delay), "study_network": quantiles(delay[on_network])},
        "reverse_reports": reverse, "reverse_clusters": reverse_clusters,
    }
(out / "waze_matching_strata.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print(json.dumps({year: {"forward": item["forward"], "delay": item["first_report_minutes_after_crash_hour_start"],
                         "reverse_reports": item["reverse_reports"], "reverse_clusters": item["reverse_clusters"]}
                  for year, item in results["years"].items()}, indent=1))
