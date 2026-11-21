"""Audit and reproduce Waze--Unfallatlas matching without modifying raw data."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict, deque
from pathlib import Path
from zoneinfo import ZoneInfo
import zipfile

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
WAZE_ROOT = ROOT / "waze_alerts"
CRASH_ROOT = ROOT / "accidents"  # folder with the Unfallorte<year>_EPSG25832_CSV.zip downloads
OUT = ROOT / "analysis" / "results" / "phase1"
YEARS = (2024, 2025)
LAT0 = math.radians(48.9)
EARTH_M = 6_371_008.8
BERLIN = ZoneInfo("Europe/Berlin")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def xy_m(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Local equirectangular coordinates, adequate for sub-km matching."""
    return (
        EARTH_M * np.radians(lon) * math.cos(LAT0),
        EARTH_M * np.radians(lat),
    )


def datetime_ns(series: pd.Series) -> np.ndarray:
    """Return UTC epoch nanoseconds independent of pandas' storage resolution."""
    utc = series.dt.tz_convert("UTC")
    return utc.astype("datetime64[ns, UTC]").astype("int64").to_numpy()


def wall_datetime_ns(series: pd.Series) -> np.ndarray:
    """Local civil-time nanoseconds; used because Unfallatlas gives no UTC fold."""
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        series = series.dt.tz_localize(None)
    return series.astype("datetime64[ns]").astype("int64").to_numpy()


def load_waze(year: int) -> tuple[pd.DataFrame, list[dict]]:
    files = sorted((WAZE_ROOT / str(year)).glob("*.csv"))
    if len(files) != 12:
        raise RuntimeError(f"Expected 12 Waze files for {year}, found {len(files)}")
    frames, provenance = [], []
    for path in files:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        frame["source_file"] = path.name
        frames.append(frame)
        provenance.append({"path": str(path), "sha256": sha256(path)})
    data = pd.concat(frames, ignore_index=True)
    data["ts_utc"] = pd.to_datetime(data["ts"], utc=True, errors="coerce")
    data["ts_local"] = data["ts_utc"].dt.tz_convert(BERLIN)
    data["ts_wall"] = data["ts_local"].dt.tz_localize(None)
    coords = data["geoWKT"].str.extract(
        r"Point\(\s*(?P<lon>-?\d+(?:\.\d+)?)\s+(?P<lat>-?\d+(?:\.\d+)?)\s*\)"
    )
    data["lon"] = pd.to_numeric(coords["lon"], errors="coerce")
    data["lat"] = pd.to_numeric(coords["lat"], errors="coerce")
    for column in ("reportRating", "Reliability", "confidence"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["type"] = data["type"].str.upper().str.strip()
    return data, provenance


def load_crashes(year: int) -> tuple[pd.DataFrame, dict, dict]:
    standardized = ROOT / "accidents" / f"Unfallorte{year}_standardized.csv"
    if standardized.exists():
        data = pd.read_csv(standardized, sep=";", dtype=str)
        provenance = {
            "path": str(standardized),
            "sha256": sha256(standardized),
            "date_source": "user-standardized date field",
        }
        lon_col, lat_col = "lon", "lat"
    else:
        path = CRASH_ROOT / f"Unfallorte{year}_EPSG25832_CSV.zip"
        with zipfile.ZipFile(path) as archive:
            member = next(
                name for name in archive.namelist() if name.lower().endswith((".csv", ".txt"))
            )
            with archive.open(member) as stream:
                data = pd.read_csv(stream, sep=";", dtype=str, decimal=",")
        provenance = {
            "path": str(path),
            "member": member,
            "sha256": sha256(path),
            "date_source": "audited identifier fallback",
        }
        lon_col, lat_col = "XGCSWGS84", "YGCSWGS84"
    data = data.loc[pd.to_numeric(data["ULAND"], errors="coerce") == 9].copy()
    raw_bavaria = len(data)
    if provenance["date_source"] == "user-standardized date field":
        data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d", errors="coerce")
    else:
        ident_col = "UIDENTSTLAE" if "UIDENTSTLAE" in data else "UIDENTSTLA"
        ident = data[ident_col].fillna("").astype(str).str.strip()
        data["date"] = pd.to_datetime(
            "20" + ident.str.slice(2, 8), format="%Y%m%d", errors="coerce"
        )
    data["hour"] = pd.to_numeric(data["USTUNDE"], errors="coerce")
    naive_start = data["date"] + pd.to_timedelta(data["hour"], unit="h")
    data["hour_start_wall"] = naive_start
    # Published accident times are local civil time. Ambiguous/nonexistent DST
    # hours are flagged instead of silently guessed.
    data["hour_start_local"] = naive_start.dt.tz_localize(
        BERLIN, ambiguous="NaT", nonexistent="NaT"
    )
    data["lon"] = pd.to_numeric(data[lon_col].str.replace(",", "."), errors="coerce")
    data["lat"] = pd.to_numeric(data[lat_col].str.replace(",", "."), errors="coerce")
    invalid_date_or_hour = int(data["hour_start_wall"].isna().sum())
    invalid_coordinate = int(
        (~data["lon"].between(8.8, 13.9) | ~data["lat"].between(47.2, 50.7)).sum()
    )
    valid = (
        data["hour_start_wall"].notna()
        & data["lon"].between(8.8, 13.9)
        & data["lat"].between(47.2, 50.7)
    )
    data = data.loc[valid].reset_index(drop=True)
    audit = {
        "bavaria_records_raw": int(raw_bavaria),
        "excluded_invalid_or_dst_ambiguous_date_hour": invalid_date_or_hour,
        "excluded_invalid_coordinate": invalid_coordinate,
        "usable_records": int(len(data)),
    }
    return data, provenance, audit


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def cluster_reports(reports: pd.DataFrame, radius_m: float = 250, minutes: int = 30) -> pd.DataFrame:
    """Connected-component clustering with explicit spatial and temporal epsilons."""
    reports = reports.sort_values("ts_utc").reset_index(drop=True).copy()
    x, y = xy_m(reports["lon"].to_numpy(), reports["lat"].to_numpy())
    seconds = datetime_ns(reports["ts_utc"]) / 1e9
    cell = radius_m
    grid: dict[tuple[int, int], deque[int]] = defaultdict(deque)
    active: deque[int] = deque()
    uf = UnionFind(len(reports))
    max_dt = minutes * 60
    for idx in range(len(reports)):
        while active and seconds[idx] - seconds[active[0]] > max_dt:
            old = active.popleft()
            key = (math.floor(x[old] / cell), math.floor(y[old] / cell))
            bucket = grid[key]
            if bucket and bucket[0] == old:
                bucket.popleft()
        key = (math.floor(x[idx] / cell), math.floor(y[idx] / cell))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for other in grid[(key[0] + dx, key[1] + dy)]:
                    if (x[idx] - x[other]) ** 2 + (y[idx] - y[other]) ** 2 <= radius_m**2:
                        uf.union(idx, other)
        grid[key].append(idx)
        active.append(idx)
    reports["cluster_root"] = [uf.find(i) for i in range(len(reports))]
    grouped = reports.groupby("cluster_root", sort=False)
    clusters = grouped.agg(
        first_ts_utc=("ts_utc", "min"),
        last_ts_utc=("ts_utc", "max"),
        lon=("lon", "mean"),
        lat=("lat", "mean"),
        reports=("type", "size"),
        max_confidence=("confidence", "max"),
        max_reliability=("Reliability", "max"),
    ).reset_index(drop=True)
    clusters["first_ts_local"] = clusters["first_ts_utc"].dt.tz_convert(BERLIN)
    return clusters


def match_counts(
    crashes: pd.DataFrame,
    reports: pd.DataFrame,
    radius_m: float,
    shift_days: int = 0,
    lower_h: float = -0.25,
    upper_h: float = 1.75,
) -> dict:
    cx, cy = xy_m(crashes["lon"].to_numpy(), crashes["lat"].to_numpy())
    wx, wy = xy_m(reports["lon"].to_numpy(), reports["lat"].to_numpy())
    tree = cKDTree(np.column_stack([wx, wy]))
    report_ns = wall_datetime_ns(reports["ts_local"])
    if shift_days:
        report_ns = report_ns + shift_days * 24 * 3600 * 1_000_000_000
    crash_ns = wall_datetime_ns(crashes["hour_start_wall"])
    lower_ns = lower_h * 3600 * 1_000_000_000
    upper_ns = upper_h * 3600 * 1_000_000_000
    matched_crashes = np.zeros(len(crashes), dtype=bool)
    matched_reports = np.zeros(len(reports), dtype=bool)
    pairs = 0
    candidate_lists = tree.query_ball_point(np.column_stack([cx, cy]), radius_m)
    for i, candidates in enumerate(candidate_lists):
        if not candidates:
            continue
        candidates = np.asarray(candidates, dtype=int)
        delta = report_ns[candidates] - crash_ns[i]
        accepted = candidates[(delta >= lower_ns) & (delta <= upper_ns)]
        if len(accepted):
            matched_crashes[i] = True
            matched_reports[accepted] = True
            pairs += len(accepted)
    return {
        "radius_m": radius_m,
        "shift_days": shift_days,
        "time_window_h": [lower_h, upper_h],
        "crashes": int(len(crashes)),
        "reports": int(len(reports)),
        "matched_crashes": int(matched_crashes.sum()),
        "matched_crash_rate": float(matched_crashes.mean()) if len(crashes) else None,
        "matched_reports": int(matched_reports.sum()),
        "matched_report_rate": float(matched_reports.mean()) if len(reports) else None,
        "accepted_pairs": int(pairs),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    provenance = {"waze": [], "crashes": []}
    for year in YEARS:
        waze, waze_sources = load_waze(year)
        crashes, crash_source, crash_audit = load_crashes(year)
        provenance["waze"].extend(waze_sources)
        provenance["crashes"].append(crash_source)
        accidents = waze.loc[
            (waze["type"] == "ACCIDENT")
            & waze["ts_local"].notna()
            & waze["lon"].notna()
            & waze["lat"].notna()
        ].copy()
        coverage_start = accidents["ts_local"].min()
        coverage_end = accidents["ts_local"].max()
        coverage_start_wall = coverage_start.tz_localize(None)
        coverage_end_wall = coverage_end.tz_localize(None)
        # A crash belongs to the complete common period only when the entire
        # primary matching window is covered by the Waze archive.
        common = crashes.loc[
            (crashes["hour_start_wall"] - pd.Timedelta(minutes=15) >= coverage_start_wall)
            & (crashes["hour_start_wall"] + pd.Timedelta(minutes=105) <= coverage_end_wall)
        ].copy()
        clusters = cluster_reports(accidents)
        clusters.to_csv(OUT / f"waze_accident_clusters_{year}.csv", index=False)
        # Cluster-first timestamps are retained for later event-level analyses.
        cluster_reports_for_match = clusters.rename(columns={"first_ts_local": "ts_local"})
        metrics = []
        for population_name, population in (("full_year", crashes), ("common_period", common)):
            for radius in (250, 500, 1000):
                for shift in (0, 7):
                    result = match_counts(population, accidents, radius, shift_days=shift)
                    result.update({"year": year, "population": population_name, "unit": "report"})
                    metrics.append(result)
                cluster_result = match_counts(
                    population,
                    cluster_reports_for_match,
                    radius,
                    shift_days=0,
                )
                cluster_result.update(
                    {"year": year, "population": population_name, "unit": "cluster"}
                )
                metrics.append(cluster_result)
        pd.DataFrame(metrics).to_csv(OUT / f"match_metrics_{year}.csv", index=False)
        cluster_size = clusters["reports"].value_counts().sort_index()
        summary = {
            "year": year,
            "waze_all_rows": int(len(waze)),
            "waze_accident_reports": int(len(accidents)),
            "waze_invalid_timestamps": int(waze["ts_local"].isna().sum()),
            "waze_invalid_coordinates": int((waze["lon"].isna() | waze["lat"].isna()).sum()),
            "coverage_start_local": coverage_start.isoformat(),
            "coverage_end_local": coverage_end.isoformat(),
            "official_crashes_full_year": int(len(crashes)),
            "official_crashes_common_period": int(len(common)),
            "official_crash_audit": crash_audit,
            "waze_accident_clusters_250m_30min": int(len(clusters)),
            "cluster_size_distribution": {str(k): int(v) for k, v in cluster_size.items()},
        }
        all_summaries.append(summary)
    (OUT / "source_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    (OUT / "source_summary.json").write_text(
        json.dumps(all_summaries, indent=2), encoding="utf-8"
    )
    print(json.dumps(all_summaries, indent=2))


if __name__ == "__main__":
    main()
