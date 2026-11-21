"""Stage 1 of a two-stage design: long-term crash frequency per edge from 2016-2023.

Police injury crashes of 2016-2023 are assigned to the detector-corridor edges
(100 m, carriageway from the Waze heading map where evidence exists, as in
Phase 6). A safety performance function (SPF; Poisson mean, NB2 dispersion by moments)
relates the eight-year count to length, a traffic-volume proxy and lane count;
the empirical Bayes (EB) estimate shrinks each edge's count towards its SPF value.
Network-wide hour-by-daytype and month profiles turn the annual edge rate into a
prior per edge x 5-min unit. Nothing from 2024 or 2025 crashes enters the prior;
those years only validate it. The volume proxy is the 2024 detector mean flow,
because no earlier traffic data are available.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
import statsmodels.api as sm
from scipy.stats import spearmanr
from shapely.strtree import STRtree

import dlm_assign

from phase1_network_scope import xy_m
from phase2_aggregate_sites import read_metadata
from phase6_minute_dataset import NETWORK_THRESHOLD_M, heading_field, load_edges, route_travel_signs


ROOT = Path(__file__).resolve().parents[1]
PHASE6 = ROOT / "analysis" / "results" / "phase6"
OUT = ROOT / "analysis" / "results" / "phase7"
HISTORY_YEARS = tuple(range(2016, 2024))
VALIDATION_YEARS = (2024, 2025)


def load_crashes(year: int) -> pd.DataFrame:
    data = pd.read_csv(ROOT / "accidents" / f"Unfallorte{year}_standardized.csv", sep=";", dtype=str)
    data = data.loc[pd.to_numeric(data["ULAND"], errors="coerce") == 9].copy()
    for column in ("lon", "lat"):
        data[column] = pd.to_numeric(data[column].str.replace(",", "."), errors="coerce")
    for column in ("UMONAT", "USTUNDE", "UWOCHENTAG", "UKATEGORIE"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["year"] = year
    return data.loc[data["lon"].between(8.8, 13.9) & data["lat"].between(47.2, 50.7),
                    ["year", "UMONAT", "USTUNDE", "UWOCHENTAG", "UKATEGORIE", "lon", "lat"]]


def assign_edges(crashes: pd.DataFrame, edges: pd.DataFrame, signs: pd.DataFrame, field: dict) -> pd.DataFrame:
    """Network crashes with a directed edge from the Basis-DLM carriageway axes (dlm_assign)."""
    assigned = dlm_assign.assign(crashes["lon"].to_numpy(), crashes["lat"].to_numpy(), edges, signs)
    keep = np.flatnonzero(assigned["member"].to_numpy())
    return crashes.iloc[keep].assign(edge_id=assigned["edge_id"].to_numpy()[keep],
                                     direction_basis=assigned["direction_basis"].to_numpy()[keep],
                                     dlm_category=assigned["dlm_category"].to_numpy()[keep])


def edge_covariates(edges: pd.DataFrame, signs: pd.DataFrame) -> pd.DataFrame:
    flows = pd.read_parquet(PHASE6 / "minute_dataset_2024.parquet",
                            columns=["edge_id", "up_mean_flow_2024", "dn_mean_flow_2024"]).groupby("edge_id").first()
    lanes = read_metadata().groupby("site")["unit"].nunique()
    frame = edges[["edge_id", "route", "site_from", "site_to", "length_straight_m"]].join(flows, on="edge_id")
    frame["road"] = frame["route"].str.split("_").str[0]
    frame["length_km"] = frame["length_straight_m"].clip(lower=50) / 1000
    flow = frame[["up_mean_flow_2024", "dn_mean_flow_2024"]].mean(axis=1)
    frame["flow_missing"] = flow.isna().astype(float)
    frame["log_flow"] = np.log(flow.fillna(flow.median()).clip(lower=0.1))
    frame["log_length"] = np.log(frame["length_km"])
    frame["lanes"] = frame[["site_from", "site_to"]].apply(lambda column: column.map(lanes)).mean(axis=1).fillna(2)
    return frame


def poisson_deviance(observed: np.ndarray, predicted: np.ndarray) -> float:
    predicted = predicted * observed.sum() / predicted.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(observed > 0, observed * np.log(observed / predicted), 0.0)
    return float(2 * (term - (observed - predicted)).mean())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    edges = load_edges()
    signs = route_travel_signs(edges)
    field = heading_field()
    assigned = {year: assign_edges(load_crashes(year), edges, signs, field)
                for year in HISTORY_YEARS + VALIDATION_YEARS}
    history = pd.concat([assigned[year] for year in HISTORY_YEARS], ignore_index=True)

    frame = edge_covariates(edges, signs)
    frame["hist_count"] = frame["edge_id"].map(history.groupby("edge_id").size()).fillna(0)
    for year in VALIDATION_YEARS:
        frame[f"count_{year}"] = frame["edge_id"].map(assigned[year].groupby("edge_id").size()).fillna(0)

    design = pd.concat([frame[["log_length", "log_flow", "flow_missing", "lanes"]],
                        pd.get_dummies(frame["road"], prefix="road", drop_first=True, dtype=float)], axis=1)
    design = sm.add_constant(design)
    # Poisson mean model (IRLS tolerates sparse road dummies); NB2 dispersion by moments.
    spf = sm.GLM(frame["hist_count"], design, family=sm.families.Poisson()).fit()
    frame["spf_count"] = spf.predict(design)
    residual = (frame["hist_count"] - frame["spf_count"]) ** 2 - frame["spf_count"]
    alpha = float(max(residual.sum() / (frame["spf_count"] ** 2).sum(), 0.0))
    weight = 1 / (1 + alpha * frame["spf_count"])
    frame["eb_count"] = weight * frame["spf_count"] + (1 - weight) * frame["hist_count"]
    years = len(HISTORY_YEARS)
    for name in ("hist", "spf", "eb"):
        frame[f"{name}_per_year"] = frame[f"{name}_count"] / years

    # Network-wide temporal profile relative to a uniform spread over time.
    weekend = history["UWOCHENTAG"].isin([1, 7])  # 1 = Sunday, 7 = Saturday
    profile = history.groupby([weekend.rename("weekend"), history["USTUNDE"].rename("hour")]).size().rename("crashes").reset_index()
    day_share = np.where(profile["weekend"], 2 / 7, 5 / 7)
    profile["multiplier"] = (profile["crashes"] / len(history)) / (day_share / 24)
    month = history.groupby("UMONAT").size().rename("crashes").reset_index().rename(columns={"UMONAT": "month"})
    days = pd.Series([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], index=range(1, 13))
    month["multiplier"] = (month["crashes"] / len(history)) / (month["month"].map(days) / 365.25)

    validation = {}
    predictors = {"length only": frame["length_km"], "SPF (length, volume proxy, lanes, road)": frame["spf_count"],
                  "observed 2016-2023 count": frame["hist_count"], "empirical Bayes": frame["eb_count"]}
    top = max(1, int(round(0.1 * len(frame))))
    for year in VALIDATION_YEARS:
        observed = frame[f"count_{year}"].to_numpy()
        validation[str(year)] = {"crashes_on_network": int(observed.sum()),
                                 "share_on_top10pct_edges_oracle": float(np.sort(observed)[-top:].sum() / observed.sum())}
        for name, values in predictors.items():
            values = values.to_numpy(dtype=float)
            validation[str(year)][name] = {
                "spearman": float(spearmanr(values, observed).statistic),
                "share_of_crashes_on_top10pct_edges": float(observed[np.argsort(values)[-top:]].sum() / observed.sum()),
                "poisson_deviance": poisson_deviance(observed, np.clip(values, 1e-6, None)),
            }

    frame.drop(columns=["site_from", "site_to"]).to_csv(OUT / "edge_frequency_prior.csv", index=False)
    profile.to_csv(OUT / "temporal_profile_hour.csv", index=False)
    month.to_csv(OUT / "temporal_profile_month.csv", index=False)
    summary = {
        "history_years": list(HISTORY_YEARS),
        "network_crashes_by_year": {str(year): int(len(data)) for year, data in assigned.items()},
        "history_direction_basis": history["direction_basis"].value_counts(normalize=True).to_dict(),
        "edges": int(len(frame)), "edges_without_history": int((frame["hist_count"] == 0).sum()),
        "spf": {"dispersion_alpha": alpha,
                "coefficients": {name: float(value) for name, value in spf.params.items() if not name.startswith("road_")},
                "deviance_explained": float(1 - spf.deviance / spf.null_deviance)},
        "validation": validation,
    }
    (OUT / "frequency_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
