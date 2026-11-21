"""Extra training rows for the second internal review (2026-10-28), v2 network, 2024 only.

  waze_hour    every non-repeat Waze-timed event with t_end = start of the clock
               hour of its first report (the anchor of the official crash hours)
  waze_repeat  events flagged as repeats (5 km / 3 h), t_end = floor5(t0) - 30 min,
               so that alternative de-duplication rules can be trained
Flags for alternative repeat rules (2 km / 1 h, 1 km / 30 min, none) are stored for
every on-network event, and the official crash hour on the same edge that each
confirmed event points to (for one-to-one matching).
Also: chance level of the Bavaria-wide matching under several time shifts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import v2_pipeline
v2_pipeline.redirect()
import phase6_minute_dataset as p6d
import phase1_match_audit as audit

V2 = Path(__file__).resolve().parents[1] / "analysis" / "results" / "v2"
OUT = V2 / "phase11"
RULES = {"5km_3h": (5000, 3), "2km_1h": (2000, 1), "1km_0.5h": (1000, 0.5)}


def repeat_flags(events: pd.DataFrame, clusters: pd.DataFrame) -> pd.DataFrame:
    """Repeat of an earlier Bavarian cluster under each rule (same definition as phase 6)."""
    seconds = clusters["t0"].to_numpy().astype("datetime64[s]").astype(np.int64)
    tree = cKDTree(clusters[["x", "y"]].to_numpy())
    ev_sec = events["t0"].to_numpy().astype("datetime64[s]").astype(np.int64)
    out = {}
    for name, (radius, hours) in RULES.items():
        flags = np.zeros(len(events), bool)
        for i, nb in enumerate(tree.query_ball_point(events[["x", "y"]].to_numpy(), radius)):
            delta = ev_sec[i] - seconds[np.asarray(nb)]
            flags[i] = ((delta > 0) & (delta <= hours * 3600)).any()
        out[f"repeat_{name}"] = flags
    return pd.DataFrame(out, index=events.index)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    edges = p6d.load_edges()
    signs = p6d.route_travel_signs(edges)
    year = 2024
    events, clusters, crashes, counts = p6d.waze_events(year, edges, signs)
    start, end = p6d.eligible_window(year)
    events = events.join(repeat_flags(events, clusters))
    assert (events["repeat_5km_3h"] == events["repeat_of_earlier_report"]).all()
    t5 = events["t0"].dt.floor("5min")
    events = events.loc[(t5 >= start + pd.Timedelta(hours=1)) & (t5 < end)].copy()
    events["t0_floor5"] = events["t0"].dt.floor("5min")

    meta = ["cluster_id", "official_confirmed", "worst_injury_category", "reports", "max_confidence",
            "max_reliability", "subtype", "heading_cos", "edge_id"] + [f"repeat_{n}" for n in RULES]
    hour = events.loc[~events["repeat_5km_3h"], meta + ["t0_floor5"]].copy()
    hour["kind"], hour["t_end"] = "waze_hour", hour["t0_floor5"].dt.floor("h")
    rep = events.loc[events["repeat_5km_3h"], meta + ["t0_floor5"]].copy()
    rep["kind"], rep["t_end"] = "waze_repeat", rep["t0_floor5"] - pd.Timedelta(minutes=30)
    rows = pd.concat([hour, rep], ignore_index=True)
    rows["reference_time"] = rows["t0_floor5"]
    rows["buffer_min"] = np.where(rows["kind"] == "waze_repeat", 30, 0)
    rows["edge_id"] = rows["edge_id"].astype(int)
    rows["t_end"] = rows["t_end"].astype("datetime64[us]")
    rows["reference_time"] = rows["reference_time"].astype("datetime64[us]")
    rows, flows = p6d.attach_features(rows, year, edges, signs, 16, OUT / "_needed")
    static = pd.Series(flows, name="mean_flow_2024")
    rows["up_mean_flow_2024"] = rows["site_up"].map(static)
    rows["dn_mean_flow_2024"] = rows["site_dn"].map(static)
    rows["official_confirmed"] = rows["official_confirmed"].fillna(False).astype(bool)
    rows.to_parquet(OUT / "extra_rows_2024.parquet", index=False)

    # Repeat flags for the existing (non-repeat) events, keyed by cluster_id.
    events[["cluster_id", "edge_id", "t0_floor5", "official_confirmed"] + [f"repeat_{n}" for n in RULES]].to_parquet(
        OUT / "event_flags_2024.parquet", index=False)

    # Chance level under several shifts (Bavaria, radius 500 m, report level), both years.
    shifts = {}
    for y in (2024, 2025):
        waze, _ = audit.load_waze(y)
        crashes_all, _, _ = audit.load_crashes(y)
        acc = waze.loc[(waze["type"] == "ACCIDENT") & waze["ts_local"].notna() & waze["lon"].notna() & waze["lat"].notna()]
        shifts[y] = {d: audit.match_counts(crashes_all, acc, 500, shift_days=d)["matched_crash_rate"]
                     for d in (0, -28, -21, -14, -7, 7, 14, 21, 28, 35, 42)}
    summary = {"rows_by_kind": rows["kind"].value_counts().to_dict(),
               "events_in_window": int(len(events)),
               "repeat_counts": {n: int(events[f"repeat_{n}"].sum()) for n in RULES},
               "chance_level_by_shift_days": shifts}
    (OUT / "extra_rows_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
