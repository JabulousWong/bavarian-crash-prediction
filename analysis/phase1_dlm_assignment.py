"""Phase 1 crash assignment on the motorway network including ramps (Basis-DLM).

Rewrites results/phase1/crash_network_assignment_{year}.csv. Columns keep their
meaning for downstream scripts: `nearest_edge` is the directed corridor edge of the
crash's carriageway, `nearest_corridor_m` the tolerance-scaled corridor distance
(<= 100 = member, NaN = not a motorway crash or no matching chord). The chord-based
originals are kept as crash_network_assignment_{year}_chord.csv.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import dlm_assign


ROOT = Path(__file__).resolve().parents[1]
PHASE1 = ROOT / "analysis" / "results" / "phase1"
OFFICIAL_BAB_INJURY = {2024: 3236, 2025: 3199}


def main() -> None:
    edges = pd.read_csv(PHASE1 / "detector_corridor_edges.csv")
    signs = dlm_assign.travel_signs(edges)
    summary = {"rule": dlm_assign.__doc__.split("\n\n")[1].replace("\n", " "), "years": {}}

    for year in (2024, 2025):
        original = PHASE1 / f"crash_network_assignment_{year}_chord.csv"
        if not original.exists():
            (PHASE1 / f"crash_network_assignment_{year}.csv").rename(original)
        crashes = pd.read_csv(original, dtype={"uid": str})
        a = dlm_assign.assign(crashes["lon"].to_numpy(), crashes["lat"].to_numpy(), edges, signs)
        out = crashes.copy()
        lookup = edges.set_index("edge_id")
        matched = a["edge_id"].to_numpy() >= 0
        out["nearest_edge"] = np.where(matched, a["edge_id"], crashes["nearest_edge"])
        out["nearest_corridor_m"] = a["corridor_distance_m"].to_numpy()
        for column, source in (("nearest_route", "route"), ("site_from", "site_from"), ("site_to", "site_to")):
            out[column] = out["nearest_edge"].map(lookup[source])
        for column in ("dlm_category", "dlm_road", "direction_basis", "chord_distance_m", "envelope_m", "member"):
            out[column] = a[column].to_numpy()
        out["chord_nearest_edge"] = crashes["nearest_edge"]
        out["chord_nearest_m"] = crashes["nearest_corridor_m"]
        out.to_csv(PHASE1 / f"crash_network_assignment_{year}.csv", index=False)
        member = out["member"]
        old = crashes["nearest_corridor_m"] <= 100
        both = member & old
        summary["years"][str(year)] = {
            "bavaria_crashes": int(len(out)),
            "motorway_by_dlm": out["dlm_category"].isin(dlm_assign.MOTORWAY).sum().item(),
            "motorway_by_dlm_categories": out.loc[out["dlm_category"].isin(dlm_assign.MOTORWAY), "dlm_category"].value_counts().to_dict(),
            "official_motorway_injury_crashes": OFFICIAL_BAB_INJURY[year],
            "members": int(member.sum()),
            "members_by_category": out.loc[member, "dlm_category"].value_counts().to_dict(),
            "members_direction_known": float(out.loc[member, "direction_basis"].str.contains("carriageway").mean()),
            "chord_rule_within_100m": int(old.sum()),
            "in_both": int(both.sum()),
            "chord_rule_only": int((old & ~member).sum()),
            "chord_rule_only_by_category": out.loc[old & ~member, "dlm_category"].value_counts().to_dict(),
            "dlm_only": int((member & ~old).sum()),
            "same_directed_edge_in_both": float((out.loc[both, "nearest_edge"] == crashes.loc[both, "nearest_edge"]).mean()),
            "members_at_tolerance_x0.5_x1.5": [int((out["nearest_corridor_m"] <= 50).sum()), int((out["nearest_corridor_m"] <= 150).sum())],
        }
    (PHASE1 / "dlm_assignment_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=int), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=int))


if __name__ == "__main__":
    main()
