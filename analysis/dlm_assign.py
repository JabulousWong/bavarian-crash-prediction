"""Network membership and carriageway direction from the official Basis-DLM road model.

Replaces the "within 100 m of a straight detector-to-detector chord" rule used by
Phase 1/2/6/7 and the v2 cohort. A point (crash or Waze cluster) is

  main        on a Bundesautobahn carriageway axis (L_42005_Fahrbahnachse, <= 10 m);
  ramp        nearest road axis (L_42003_Strassenachse) is a Bundesautobahn
              Auf-/Abfahrt;
  bab_other   nearest road axis is another Bundesautobahn axis (rest area, ...);
  other_road  nearest road axis belongs to any other road class.

Carriageway axes are one-way and digitised in the direction of travel (ATKIS),
so main-carriageway points take the axis direction at their foot point. Ramp and
facility points take the direction of the carriageway touching the ramp's end
(<= 30 m); otherwise the direction stays unknown.

Motorway points (main, ramp, bab_other) are matched to the nearest corridor chord
of the same road number whose travel direction agrees with the carriageway
(cosine > 0.3; any direction if unknown), searched within 2.5 km. They belong to
the corridor when the distance to that chord is at most max(100 m, envelope + 50 m),
the envelope being the chord's largest deviation from the real carriageway of its
road. `corridor_distance_m` is that distance scaled so that 100 marks the rule's
tolerance: existing thresholds (<= 100) therefore select members, and 50 / 150
halve or extend the tolerance. Non-members get NaN.

Inputs are exported once by dlm/export_dlm.py (system python with geopandas).
"""

from __future__ import annotations

import hashlib
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely
from shapely.strtree import STRtree


ROOT = Path(__file__).resolve().parents[1]
DLM = ROOT / "dlm"
LAT0 = math.radians(48.9)
EARTH_M = 6_371_008.8
MAIN_M = 10.0
RAMP_TOUCH_M = 30.0
SEARCH_M = 2_500.0
MIN_TOLERANCE_M = 100.0
ENVELOPE_MARGIN_M = 50.0
DIRECTION_COS = 0.3
MOTORWAY = ("main", "ramp", "bab_other")


def _xy(coords: np.ndarray) -> np.ndarray:
    return np.column_stack([EARTH_M * np.radians(coords[:, 0]) * math.cos(LAT0), EARTH_M * np.radians(coords[:, 1])])


def road_of_route(route: str) -> str:
    """'A8-West_Mch' -> 'A8', 'A99_Stg_Lin' -> 'A99'."""
    return route.split("_")[0].split("-")[0]


@lru_cache(maxsize=1)
def _layers() -> dict:
    axes = pq.read_table(DLM / "road_axes.parquet").to_pandas()
    carriage = pq.read_table(DLM / "bab_carriageways.parquet").to_pandas()
    carriage = carriage.loc[~carriage["zustand"].str.startswith("Außer")].reset_index(drop=True)
    axis_geom = shapely.transform(shapely.from_wkb(axes["wkb"].to_numpy()), _xy)
    carriage_geom = shapely.transform(shapely.from_wkb(carriage["wkb"].to_numpy()), _xy)
    return {
        "axis_geom": axis_geom, "axis_tree": STRtree(axis_geom),
        "axis_widmung": axes["widmung"].to_numpy(), "axis_funktion": axes["funktion"].to_numpy(),
        "axis_road": axes["road"].to_numpy(),
        "cw_geom": carriage_geom, "cw_tree": STRtree(carriage_geom), "cw_road": carriage["road"].to_numpy(),
    }


def _direction(lines: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Unit direction of each line (as digitised) at the foot point of each point."""
    s = shapely.line_locate_point(lines, points)
    length = shapely.length(lines)
    a = shapely.get_coordinates(shapely.line_interpolate_point(lines, np.clip(s - 2, 0, length)))
    b = shapely.get_coordinates(shapely.line_interpolate_point(lines, np.clip(s + 2, 0, length)))
    v = b - a
    n = np.hypot(v[:, 0], v[:, 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        return v / n[:, None]


def classify(lon: np.ndarray, lat: np.ndarray) -> pd.DataFrame:
    """Road category, road number and travel direction of each point."""
    L = _layers()
    xy = _xy(np.column_stack([lon, lat]).astype(float))
    pts = shapely.points(xy)
    n = len(pts)
    (pi, ai), _ = L["axis_tree"].query_nearest(pts, return_distance=True, all_matches=False)
    axis = np.full(n, -1)
    axis[pi] = ai
    (pc, ci), dist = L["cw_tree"].query_nearest(pts, return_distance=True, all_matches=False)
    cw = np.full(n, -1)
    d_main = np.full(n, np.inf)
    cw[pc], d_main[pc] = ci, dist
    widmung = np.where(axis >= 0, L["axis_widmung"][np.maximum(axis, 0)], "?")
    funktion = np.where(axis >= 0, L["axis_funktion"][np.maximum(axis, 0)], "?")
    category = np.select(
        [d_main <= MAIN_M, widmung == "Bundesautobahn"],
        ["main", np.where(funktion == "Auf- oder Abfahrt", "ramp", "bab_other")], default="other_road")
    road = np.where(axis >= 0, L["axis_road"][np.maximum(axis, 0)], "")
    direction = np.full((n, 2), np.nan)
    basis = np.full(n, "", dtype=object)

    main = np.flatnonzero(category == "main")
    road[main] = L["cw_road"][cw[main]]
    direction[main] = _direction(L["cw_geom"][cw[main]], pts[main])
    basis[main] = "DLM carriageway axis"

    for i in np.flatnonzero((category == "ramp") | (category == "bab_other")):
        line = L["axis_geom"][axis[i]]
        ends = shapely.points(shapely.get_coordinates(line)[[0, -1]])
        touching = np.unique(L["cw_tree"].query(ends, predicate="dwithin", distance=RAMP_TOUCH_M)[1])
        if len(touching):
            best = touching[np.argmin(shapely.distance(pts[i], L["cw_geom"][touching]))]
            direction[i] = _direction(L["cw_geom"][[best]], pts[[i]])[0]
            road[i] = L["cw_road"][best]
            basis[i] = "DLM ramp -> touching carriageway"
        else:
            basis[i] = "DLM ramp, direction unknown"
    basis[category == "other_road"] = "not a motorway (DLM)"
    return pd.DataFrame({"dlm_category": category, "dlm_road": road, "dir_x": direction[:, 0], "dir_y": direction[:, 1],
                         "direction_basis": basis, "dlm_main_axis_m": d_main})


def _chords(edges: pd.DataFrame, signs: pd.DataFrame | None) -> tuple[np.ndarray, np.ndarray]:
    geometry = shapely.from_wkt(edges["geometry_wkt"].to_numpy())
    coords = np.stack([shapely.get_coordinates(g)[[0, -1]] for g in geometry])
    vec = coords[:, 1] - coords[:, 0]
    if signs is None:
        signs = travel_signs(edges)
    sign = edges["route"].map(signs.set_index("route")["travel_sign"]).fillna(1).to_numpy()
    travel = vec * sign[:, None]
    travel = travel / np.maximum(np.hypot(travel[:, 0], travel[:, 1]), 1e-9)[:, None]
    return geometry, travel


def travel_signs(edges: pd.DataFrame) -> pd.DataFrame:
    from phase6_minute_dataset import route_travel_signs
    frame = edges.copy()
    coords = np.stack([shapely.get_coordinates(g)[[0, -1]] for g in shapely.from_wkt(frame["geometry_wkt"].to_numpy())])
    frame["dx"], frame["dy"] = coords[:, 1, 0] - coords[:, 0, 0], coords[:, 1, 1] - coords[:, 0, 1]
    frame["mx"], frame["my"] = coords[:, :, 0].mean(axis=1), coords[:, :, 1].mean(axis=1)
    return route_travel_signs(frame)


def envelopes(edges: pd.DataFrame) -> np.ndarray:
    """Largest distance of each chord from the real carriageway axes of its road (cached on disk)."""
    key = hashlib.sha256("|".join(edges["route"] + edges["geometry_wkt"]).encode()).hexdigest()[:16]
    cache = DLM / f"envelope_{key}.npy"
    if cache.exists():
        return np.load(cache)
    L = _layers()
    geometry = shapely.from_wkt(edges["geometry_wkt"].to_numpy())
    roads = edges["route"].map(road_of_route).to_numpy()
    result = np.full(len(edges), np.inf)
    for road in np.unique(roads):
        own = L["cw_geom"][L["cw_road"] == road]
        if not len(own):
            continue
        tree = STRtree(own)
        for i in np.flatnonzero(roads == road):
            g = geometry[i]
            samples = shapely.line_interpolate_point(g, np.linspace(0, 1, max(3, int(g.length // 50) + 1)), normalized=True)
            (_, nearest), distance = tree.query_nearest(samples, return_distance=True, all_matches=False)
            result[i] = float(distance.max())
    np.save(cache, result)
    return result


def assign(lon: np.ndarray, lat: np.ndarray, edges: pd.DataFrame, signs: pd.DataFrame | None = None) -> pd.DataFrame:
    """Classification plus directed corridor edge and membership for each point (row order kept)."""
    lon, lat = np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
    result = classify(lon, lat)
    n = len(result)
    geometry, travel = _chords(edges, signs)
    envelope = envelopes(edges)
    edge_ids = edges["edge_id"].to_numpy()
    edge_road = edges["route"].map(road_of_route).to_numpy()
    pts = shapely.points(_xy(np.column_stack([lon, lat])))
    motorway = np.flatnonzero(result["dlm_category"].isin(MOTORWAY).to_numpy())
    p_idx, e_idx = STRtree(geometry).query(pts[motorway], predicate="dwithin", distance=SEARCH_M)
    p = motorway[p_idx]
    pairs = pd.DataFrame({"p": p, "e": e_idx, "d": shapely.distance(pts[p], geometry[e_idx])})
    direction = result[["dir_x", "dir_y"]].to_numpy()[p]
    cos = (direction * travel[e_idx]).sum(axis=1)
    keep = (edge_road[e_idx] == result["dlm_road"].to_numpy()[p]) & (np.isnan(cos) | (cos > DIRECTION_COS))
    best = pairs.loc[keep].sort_values(["p", "d"]).drop_duplicates("p")
    edge = np.full(n, -1)
    chord = np.full(n, np.nan)
    env = np.full(n, np.nan)
    edge[best["p"]] = edge_ids[best["e"]]
    chord[best["p"]] = best["d"].to_numpy()
    env[best["p"]] = envelope[best["e"]]
    tolerance = np.maximum(MIN_TOLERANCE_M, env + ENVELOPE_MARGIN_M)
    with np.errstate(invalid="ignore"):
        scaled = chord * 100.0 / tolerance
        member = scaled <= 100.0
    result["edge_id"] = edge
    result["chord_distance_m"] = chord
    result["envelope_m"] = env
    result["corridor_distance_m"] = np.where(edge >= 0, scaled, np.nan)
    result["member"] = member
    return result
