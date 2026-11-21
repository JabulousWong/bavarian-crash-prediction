"""Sensitivity of the label-set comparisons to the Waze clustering, matching and repeat rules.

One-at-a-time departures from the primary construction (250 m / 30 min clustering, 500 m and
-15..+105 min matching, 5 km / 3 h repeat rule), fixed before running. Only the 2024 Waze-timed
training rows change; the 2025 evaluation on the recorded crashes does not. Protocol of phase 11
part m1m7: MLP, PN risk, prior = labelled rate, 5 seeds, 30-min buffer. Exploratory.

  --stage rows    rebuild events and attach traffic features (no torch)
  --stage fit     train and evaluate
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

V2 = Path(__file__).resolve().parents[1] / "analysis" / "results" / "v2"
OUT = V2 / "phase13"
TMP = OUT / "_phase1"
BUDGET = 0.063
BASE = dict(cluster=(250, 30), radius=500, window=(-0.25, 1.75), repeat=(5000, 3))
VARIANTS = {
    "base": {},
    "cluster_125m_30min": {"cluster": (125, 30)}, "cluster_500m_30min": {"cluster": (500, 30)},
    "cluster_250m_15min": {"cluster": (250, 15)}, "cluster_250m_60min": {"cluster": (250, 60)},
    "window_0_105": {"window": (0.0, 1.75)}, "window_m15_75": {"window": (-0.25, 1.25)}, "window_m15_135": {"window": (-0.25, 2.25)},
    "radius_250m": {"radius": 250}, "radius_1000m": {"radius": 1000},
    "repeat_5km_2h": {"repeat": (5000, 2)},
}


def stage_rows(workers):
    import shutil
    import v2_pipeline
    v2_pipeline.redirect()
    import phase6_minute_dataset as p6d
    import phase1_match_audit as audit
    TMP.mkdir(parents=True, exist_ok=True)
    (TMP / "crash_network_assignment_2024.csv").unlink(missing_ok=True)
    (TMP / "crash_network_assignment_2024.csv").symlink_to(V2 / "phase1" / "crash_network_assignment_2024.csv")
    year = 2024
    waze, _ = audit.load_waze(year)
    acc = waze.loc[(waze["type"] == "ACCIDENT") & waze["ts_utc"].notna() & waze["lon"].notna() & waze["lat"].notna()].copy()
    stored = pd.read_csv(V2 / "phase1" / "waze_accident_clusters_2024.csv")
    check = audit.cluster_reports(acc)
    print("clusters stored", len(stored), "reclustered", len(check), flush=True)
    assert abs(len(check) - len(stored)) <= 5, "reclustering does not reproduce the stored clusters"
    edges = p6d.load_edges()
    signs = p6d.route_travel_signs(edges)
    start, end = p6d.eligible_window(year)
    frames, info = [], {}
    for name, change in VARIANTS.items():
        c = {**BASE, **change}
        clusters = audit.cluster_reports(acc, radius_m=c["cluster"][0], minutes=c["cluster"][1]) if c["cluster"] != BASE["cluster"] else stored
        clusters.to_csv(TMP / "waze_accident_clusters_2024.csv", index=False)
        p6d.PHASE1, p6d.MATCH_RADIUS_M, p6d.MATCH_WINDOW_H = TMP, c["radius"], c["window"]
        p6d.REPEAT_RADIUS_M, p6d.REPEAT_HOURS = c["repeat"]
        ev, _, _, _ = p6d.waze_events(year, edges, signs)
        ev = ev.loc[~ev["repeat_of_earlier_report"]].copy()
        t5 = ev["t0"].dt.floor("5min")
        ev = ev.loc[(t5 >= start + pd.Timedelta(hours=1)) & (t5 < end)].copy()
        rows = pd.DataFrame({"variant": name, "edge_id": ev["edge_id"].astype(int).to_numpy(),
                             "t_end": (ev["t0"].dt.floor("5min") - pd.Timedelta(minutes=30)).astype("datetime64[us]").to_numpy(),
                             "official_confirmed": ev["official_confirmed"].astype(bool).to_numpy(),
                             "reports": ev["reports"].to_numpy()})
        frames.append(rows)
        info[name] = {"clusters_bavaria": int(len(clusters)), "events": int(len(rows)), "confirmed": int(rows["official_confirmed"].sum()),
                      "single_report_share": float((clusters["reports"] == 1).mean())}
        print(name, info[name], f"{time.strftime('%X')}", flush=True)
    allrows = pd.concat(frames, ignore_index=True)
    uniq = allrows[["edge_id", "t_end"]].drop_duplicates().reset_index(drop=True)
    uniq["reference_time"] = uniq["t_end"]
    feat, flows = p6d.attach_features(uniq, year, edges, signs, workers, OUT / "_needed")
    static = pd.Series(flows, name="mean_flow_2024")
    feat["up_mean_flow_2024"], feat["dn_mean_flow_2024"] = feat["site_up"].map(static), feat["site_dn"].map(static)
    out = allrows.merge(feat, on=["edge_id", "t_end"], how="left")
    assert len(out) == len(allrows)
    out["kind"], out["buffer_min"] = "waze_case", 30
    out.to_parquet(OUT / "variant_rows_2024.parquet", index=False)
    (OUT / "rows_summary.json").write_text(json.dumps(info, indent=2))
    shutil.rmtree(TMP, ignore_errors=True)
    print("rows written", len(out), flush=True)


def stage_fit():
    import phase11_review_analyses as p11
    from phase6_minute_pu import CONTEXT, TRAFFIC, Preprocessor, raw_features, predict
    D = p11.Data()
    rows = pd.read_parquet(OUT / "variant_rows_2024.parquet")
    prep = Preprocessor(TRAFFIC + CONTEXT).fit(D.xtr_raw.loc[D.is_u])
    xu = D.xtr[D.is_u]
    def fit(xp):
        prior = len(xp) / D.units[2024]
        models = [p11.fit_mlp(xp, np.ones(len(xp)), xu, prior, "pn", p11.SEED + s) for s in range(5)]
        s = predict(models, D.xte, p11.DEVICE)
        e = D.evaluate(s)
        e["caught_at_budget"] = float((s[D.t_c] >= np.quantile(s[D.t_m], 1 - BUDGET)).mean())
        return e
    official = fit(D.xtr[D.pos["official"]])
    print("official", round(official["auc"], 4), flush=True)
    res, ev, rng = {"official": p11.strip(official), "variants": {}}, {}, np.random.default_rng(p11.SEED + 131)
    info = json.loads((OUT / "rows_summary.json").read_text())
    for name in VARIANTS:
        f = rows.loc[rows["variant"] == name].reset_index(drop=True)
        x = prep.transform(raw_features(f))
        conf = f["official_confirmed"].to_numpy()
        ev[name] = {"union": fit(x), "reliable": fit(x[conf]), "wazeonly": fit(x[~conf])}
        r = res["variants"][name] = {**info[name], "n_reliable": int(conf.sum()), "n_wazeonly": int((~conf).sum()),
                                     **{a: p11.strip(ev[name][a]) for a in ev[name]}}
        e = ev[name]
        r["gain_union_minus_reliable"] = p11.paired(e["union"], e["reliable"], D, rng)
        r["union_minus_official"] = p11.paired(e["union"], official, D, rng)
        r["wazeonly_minus_official"] = p11.paired(e["wazeonly"], official, D, rng)
        if name != "base":
            r["union_minus_base_union"] = p11.paired(e["union"], ev["base"]["union"], D, rng)
        print(name, r["n_reliable"], r["n_wazeonly"], *(round(e[a]["auc"], 4) for a in e),
              "gain", round(r["gain_union_minus_reliable"]["auc_delta"], 4), flush=True)
    (OUT / "label_sensitivity.json").write_text(json.dumps(res, indent=2, default=float))
    print("done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("rows", "fit"), required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    stage_rows(a.workers) if a.stage == "rows" else stage_fit()
