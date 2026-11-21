"""Neighbour-edge 5-min sequences for the rows of the minute-level dataset (v2).

The v2 network is a set of directed chains: every edge has exactly one upstream and
one downstream edge on its route, except at chain ends. For a row on edge e with
cross-sections (s0, s1) the ego graph of radius k = 2 is

  up2 = (s-2, s-1)   up1 = (s-1, s0)   ego = (s0, s1)   dn1 = (s1, s2)   dn2 = (s2, s3)

plus the opposite-carriageway edge `opp` (the edge of the paired route of the same
motorway whose midpoint is closest, travelling the other way, within OPPOSITE_MAX_M).
The ego sections are already in `sequences_{year}.npy`; this script writes the six
other sections in order SECTIONS, same bins and channels as `phase9_sequence_dataset`
(NaN where the section does not exist or has no data), and the edge table that says
which neighbour edges exist.
Output: results/v2/phase9/graph_sequences_{year}.npy, float32, (rows, 6, 24, 7);
        results/v2/phase9/graph_edges.csv; graph_summary.json.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import phase6_minute_dataset
from phase2_aggregate_sites import read_metadata
from phase2_aggregate_traffic import source_files
from phase9_sequence_dataset import BINS, CHANNELS, site_windows


ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "analysis" / "results" / "v2"
DATA = V2 / "phase6"
OUT = V2 / "phase9"
SECTIONS = ["s_m2", "s_m1", "s_p2", "s_p3", "opp_up", "opp_dn"]
OPPOSITE_MAX_M = 1000.0


def graph_edges() -> pd.DataFrame:
    """Directed edges with their up/downstream cross-sections, chain neighbours and opposite edge."""
    phase6_minute_dataset.PHASE1 = V2 / "phase1"
    edges = phase6_minute_dataset.load_edges()
    signs = phase6_minute_dataset.route_travel_signs(edges)
    sign = edges["route"].map(signs.set_index("route")["travel_sign"]).fillna(1).to_numpy()
    edges["site_up"] = np.where(sign > 0, edges["site_from"], edges["site_to"])
    edges["site_dn"] = np.where(sign > 0, edges["site_to"], edges["site_from"])
    edges["tx"], edges["ty"] = edges["dx"] * sign, edges["dy"] * sign
    if edges["site_up"].duplicated().any() or edges["site_dn"].duplicated().any():
        raise ValueError("expected a set of chains: one edge leaving and one entering each cross-section")
    by_up = edges.set_index("site_up")["edge_id"]
    by_dn = edges.set_index("site_dn")["edge_id"]
    edges["up1"] = edges["site_up"].map(by_dn).astype("Int64")
    edges["dn1"] = edges["site_dn"].map(by_up).astype("Int64")
    lookup = edges.set_index("edge_id")
    edges["up2"] = edges["up1"].map(lookup["up1"]).astype("Int64")
    edges["dn2"] = edges["dn1"].map(lookup["dn1"]).astype("Int64")

    motorway = edges["route"].str.split("_").str[0]
    opposite, distance = [], []
    for row in edges.itertuples():
        other = edges.loc[(motorway == motorway[row.Index]) & (edges["route"] != row.route)
                          & (edges["tx"] * row.tx + edges["ty"] * row.ty < 0)]
        if other.empty:
            opposite.append(pd.NA), distance.append(np.nan)
            continue
        gap = np.hypot(other["mx"] - row.mx, other["my"] - row.my)
        best = int(np.argmin(gap.to_numpy()))
        within = gap.iloc[best] <= OPPOSITE_MAX_M
        opposite.append(int(other["edge_id"].iloc[best]) if within else pd.NA)
        distance.append(float(gap.iloc[best]))
    edges["opp"] = pd.array(opposite, dtype="Int64")
    edges["opp_distance_m"] = distance
    edges["s_m1"] = edges["up1"].map(lookup["site_up"])
    edges["s_m2"] = edges["up2"].map(lookup["site_up"])
    edges["s_p2"] = edges["dn1"].map(lookup["site_dn"])
    edges["s_p3"] = edges["dn2"].map(lookup["site_dn"])
    edges["opp_up"] = edges["opp"].map(lookup["site_up"])
    edges["opp_dn"] = edges["opp"].map(lookup["site_dn"])
    return edges[["edge_id", "route", "site_up", "site_dn", "up1", "up2", "dn1", "dn2", "opp", "opp_distance_m"] + SECTIONS]


def build(year: int, edges: pd.DataFrame, workers: int) -> dict:
    rows = pd.read_parquet(DATA / f"minute_dataset_{year}.parquet", columns=["row_id", "edge_id", "t_end"])
    rows = rows.join(edges.set_index("edge_id")[SECTIONS], on="edge_id")
    end_index = ((rows["t_end"] - pd.Timestamp(datetime(year, 1, 1))) // pd.Timedelta(minutes=5)).to_numpy()
    if (end_index < BINS).any() or (rows["t_end"].dt.minute % 5 != 0).any():
        raise ValueError("t_end must lie on the 5-min grid and at least two hours into the year")
    metadata, files = read_metadata(), source_files(year)
    columns = {name: rows[name].to_numpy() for name in SECTIONS}
    sites = pd.unique(pd.concat([rows[name] for name in SECTIONS]).dropna())
    needed = {site: np.unique(np.concatenate([end_index[columns[name] == site] for name in SECTIONS])) for site in sites}
    tasks = []
    for site in sites:
        paths = [str(files[unit]) for unit in metadata.loc[metadata["site"] == site, "unit"] if unit in files]
        if paths:
            tasks.append((year, site, paths, needed[site]))
    print(f"{year}: {len(tasks)} of {len(sites)} neighbour cross-sections have lane files", flush=True)
    sequences = np.full((len(rows), len(SECTIONS), BINS, len(CHANNELS)), np.nan, dtype=np.float32)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for count, (site, windows) in enumerate(pool.map(site_windows, tasks, chunksize=4), 1):
            if windows is not None:
                for slot, name in enumerate(SECTIONS):
                    members = np.flatnonzero(columns[name] == site)
                    sequences[members, slot] = windows[np.searchsorted(needed[site], end_index[members])]
            if count % 100 == 0 or count == len(tasks):
                print(f"{year}: cross-sections {count}/{len(tasks)}", flush=True)
    np.save(OUT / f"graph_sequences_{year}.npy", sequences)
    observed = ~np.isnan(sequences[..., CHANNELS.index("flow")]).all(axis=2)
    return {"rows": int(len(rows)), "shape": list(sequences.shape),
            "share_rows_with_section": {name: float(rows[name].notna().mean()) for name in SECTIONS},
            "share_rows_with_any_flow": {name: float(observed[:, slot].mean()) for slot, name in enumerate(SECTIONS)}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    edges = graph_edges()
    edges.to_csv(OUT / "graph_edges.csv", index=False)
    summary = {"sections": SECTIONS, "bins": BINS, "channels": CHANNELS, "opposite_max_m": OPPOSITE_MAX_M,
               "edges": int(len(edges)),
               "edges_with": {name: int(edges[name].notna().sum()) for name in ("up1", "up2", "dn1", "dn2", "opp")},
               "opposite_distance_m_quantiles_0_50_90": [float(edges["opp_distance_m"].quantile(q)) for q in (0, 0.5, 0.9)],
               "years": {}}
    print(json.dumps({k: v for k, v in summary.items() if k != "years"}, indent=2), flush=True)
    for year in (2024, 2025):
        summary["years"][str(year)] = build(year, edges, args.workers)
    (OUT / "graph_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
