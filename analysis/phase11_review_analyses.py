"""Re-analyses for the second internal review (2026-10-28), v2 network, protocol of phase 6.

  m1m7   label sets and time anchors under one training protocol (official, reliable,
         union, Waze-only; 30-min buffer anchor vs clock-hour anchor), learning curves
  m2     class prior misspecified (x0.25..x4), stationarity check of the mean on U,
         level by quarter of 2025 (base-rate drift), decile reliability with intervals
  m5     early stopping on week blocks with a 2-h gap; development split inside 2024
  m6     block bootstraps (edge, route, week, route x week), resampled U, refit
         variability, equivalence (TOST, margin 0.01)
  m8     label accounting, weight distributions, truncation, cross-fitted propensity
  m9     alternative repeat rules, one-to-one matching of reliable positives
  m11    recall at fixed alert budgets, average precision, net benefit
All results are exploratory: 2025 was examined during development.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

import phase6_minute_pu as p6
from phase6_minute_pu import CONTEXT, TRAFFIC, Preprocessor, placements, predict, raw_features, risk_parts, within_stratum

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "analysis" / "results" / "v2"
OUT = V2 / "phase11"
SEED = 20261029
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FOLD_ROADS = {"A8": 0, "A99": 1, "A3": 1, "A9": 2, "A7": 2, "A93": 2, "A96": 3, "A94": 3, "A92": 3, "A95": 3}


# ------------------------------------------------------------------ helpers
def ci(v, rng, reps=1000, q=(0.025, 0.975)):
    d = v[rng.integers(0, len(v), size=(reps, len(v)))].mean(axis=1)
    return [float(np.quantile(d, x)) for x in q]


def cluster_ci(v, g, rng, reps=1000, q=(0.025, 0.975)):
    codes, inv = np.unique(g, return_inverse=True)
    s, c = np.bincount(inv, weights=v), np.bincount(inv)
    d = []
    for _ in range(reps):
        k = rng.integers(0, len(codes), len(codes))
        d.append(s[k].sum() / c[k].sum())
    return [float(np.quantile(d, x)) for x in q]


def hourly(p):
    return 1 - (1 - np.clip(p, 0, 1)) ** 12


def road(route):
    return route.str.split("_").str[0].str.split("-").str[0]


def fit_mlp(xp, wp, xu, prior, mode, seed, hold_p=None, hold_u=None, max_steps=800, check_every=10, patience=12):
    """phase6_minute_pu.fit_mlp with explicit hold-out masks (None: random 20%, as in phase 6)."""
    rng = np.random.default_rng(seed)
    if hold_p is None:
        hold_p, hold_u = rng.random(len(xp)) < 0.2, rng.random(len(xu)) < 0.2
    train_p, train_u = hold_p == 0, hold_u == 0  # values: 0 train, 1 hold, 2 gap (dropped)
    t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=DEVICE)
    tp, tw, tu = t(xp[train_p]), t(wp[train_p]), t(xu[train_u])
    vp, vw, vu = t(xp[hold_p == 1]), t(wp[hold_p == 1]), t(xu[hold_u == 1])
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(xp.shape[1], 64), torch.nn.ReLU(), torch.nn.Linear(64, 32),
                                torch.nn.ReLU(), torch.nn.Linear(32, 1)).to(DEVICE)
    with torch.no_grad():
        model[-1].bias.fill_(float(np.log(prior / (1 - prior))))
        model[-1].weight.mul_(0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    def objective(pos, neg, uneg):
        return pos + uneg if mode == "pn" else pos + torch.clamp(neg, min=0)

    best, state, stale = float("inf"), None, 0
    for step in range(1, max_steps + 1):
        opt.zero_grad()
        pos, neg, uneg = risk_parts(model, tp, tw, tu, prior)
        loss = -neg if (mode == "nnpu" and neg.item() < 0) else pos + (uneg if mode == "pn" else neg)
        (loss / prior).backward()
        opt.step()
        if step % check_every == 0:
            with torch.no_grad():
                val = objective(*risk_parts(model, vp, vw, vu, prior)).item()
            if val < best - 1e-12:
                best, stale = val, 0
                state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    break
    model.load_state_dict(state)
    model.eval()
    return model


class Data:
    def __init__(self):
        self.train = pd.read_parquet(V2 / "phase6" / "minute_dataset_2024.parquet")
        self.test = pd.read_parquet(V2 / "phase6" / "minute_dataset_2025.parquet")
        extra = pd.read_parquet(OUT / "extra_rows_2024.parquet")
        flags = pd.read_parquet(OUT / "event_flags_2024.parquet")
        for f in (self.train, self.test):
            f["official_confirmed"] = f["official_confirmed"].fillna(False).astype(bool)
        # attach repeat flags of alternative rules to the existing Waze rows
        self.train = self.train.merge(flags[["cluster_id"] + [c for c in flags if c.startswith("repeat_")]],
                                      on="cluster_id", how="left")
        extra = extra[[c for c in extra.columns if c in self.train.columns or c.startswith("repeat_")]]
        self.train = pd.concat([self.train, extra], ignore_index=True)
        summary = json.loads((V2 / "phase6" / "minute_dataset_summary.json").read_text())
        self.units = {y: summary["years"][str(y)]["population_edge_5min_units"] for y in (2024, 2025)}
        self.xtr_raw, self.xte_raw = raw_features(self.train), raw_features(self.test)
        self.is_u = (self.train["kind"] == "marginal").to_numpy()
        prep = Preprocessor(TRAFFIC + CONTEXT).fit(self.xtr_raw.loc[self.is_u])
        self.xtr, self.xte = prep.transform(self.xtr_raw), prep.transform(self.xte_raw)
        k = self.train["kind"].to_numpy()
        b = self.train["buffer_min"].to_numpy()
        conf = self.train["official_confirmed"].to_numpy()
        w30 = (k == "waze_case") & (b == 30)
        wh = k == "waze_hour"
        self.pos = {
            "official": k == "official_case",
            "reliable_b30": w30 & conf, "union_b30": w30, "wazeonly_b30": w30 & ~conf,
            "reliable_hour": wh & conf, "union_hour": wh, "wazeonly_hour": wh & ~conf,
        }
        self.pos["official+wazeonly_b30"] = self.pos["official"] | self.pos["wazeonly_b30"]
        self.pos["official+wazeonly_hour"] = self.pos["official"] | self.pos["wazeonly_hour"]
        self.prior_rec = self.pos["official"].sum() / self.units[2024]
        self.prior_all = self.prior_rec * p6.ALL_TO_INJURY_RATIO
        # credibility weights from the phase 6 model (fitted on 2024 Waze-timed events)
        base = self.train.loc[k != "waze_hour"].loc[lambda f: f["kind"] != "waze_repeat"]
        idx = base.index.to_numpy()
        lw = p6.label_weights(base.reset_index(drop=True), self.test, self.xtr_raw.loc[idx].reset_index(drop=True),
                              self.xte_raw, self.is_u[idx])
        self.cred = np.ones(len(self.train))
        self.cred[idx] = lw["credibility_weight"]
        self.inv_prop = np.ones(len(self.train))
        self.inv_prop[idx] = lw["inverse_propensity"]
        self.labels_info = {"credibility_model": lw["credibility_model"], "propensity_model": lw["propensity_model"]}
        tk = self.test["kind"].to_numpy()
        self.t_m, self.t_c = tk == "marginal", tk == "official_case"
        self.t_sets = np.isin(tk, ["official_case", "official_control"])
        self.case_edges = self.test.loc[self.t_c, "edge_id"].to_numpy()
        self.n_hours = self.units[2025] / 12

    def fit(self, posmask, prior, mode="pn", weight=None, seeds=5, seed0=SEED, hold=None, rows=None, x_eval=None):
        rows = np.ones(len(self.train), bool) if rows is None else rows
        pm, um = posmask & rows, self.is_u & rows
        w = np.ones(pm.sum()) if weight is None else weight[pm]
        hp, hu = (None, None) if hold is None else (hold[pm], hold[um])
        models = [fit_mlp(self.xtr[pm], w, self.xtr[um], prior, mode, seed0 + s, hp, hu) for s in range(seeds)]
        x_eval = self.xte if x_eval is None else x_eval
        return predict(models, x_eval, DEVICE), predict(models, self.xtr[um], DEVICE)

    def evaluate(self, s, rng=None):
        pl = placements(s[self.t_c], np.sort(s[self.t_m]))
        sub = self.test.loc[self.t_sets]
        w = within_stratum(sub, s[self.t_sets], (sub["kind"] == "official_case").to_numpy())
        hu, hc = hourly(s[self.t_m]), hourly(s[self.t_c])
        return {"auc": float(pl.mean()), "within": float(w.mean()),
                "level": float(hu.mean() * self.n_hours / self.t_c.sum()),
                "slope": calibration_slope(hu, hc, self.n_hours), "log_skill": log_skill(hu, hc, self.n_hours, self.prior_rec),
                "_pl": pl, "_w": w}


def calibration_slope(hu, hc, n_hours):
    hu, hc = np.clip(hu, 1e-12, 1 - 1e-12), np.clip(hc, 1e-12, 1 - 1e-12)
    z = np.log(np.r_[hu, hc] / (1 - np.r_[hu, hc]))
    y = np.r_[np.zeros(len(hu)), np.ones(len(hc))]
    w = np.r_[np.full(len(hu), (n_hours - len(hc)) / len(hu)), np.ones(len(hc))]
    X = np.c_[np.ones_like(z), z]
    b = np.array([np.log(len(hc) / (n_hours - len(hc))) - np.average(z, weights=w), 1.0])
    for _ in range(60):
        p = 1 / (1 + np.exp(-X @ b))
        H = (X * (w * p * (1 - p))[:, None]).T @ X
        step = np.linalg.solve(H, X.T @ (w * (y - p)))
        b = b + step
        if np.abs(step).max() < 1e-10:
            break
    return float(b[1])


def log_skill(hu, hc, n_hours, prior):
    def L(u, c):
        u, c = np.clip(u, 1e-12, 1 - 1e-12), np.clip(c, 1e-12, 1 - 1e-12)
        return -(np.log1p(-u).mean() + (np.log(c) - np.log1p(-c)).sum() / n_hours)
    h0 = hourly(np.array([prior]))[0]
    return float(1 - L(hu, hc) / L(np.full(len(hu), h0), np.full(len(hc), h0)))


def strip(d):
    return {k: v for k, v in d.items() if not k.startswith("_")}


def paired(a, b, D, rng):
    dp, dw = a["_pl"] - b["_pl"], a["_w"] - b["_w"]
    return {"auc_delta": float(dp.mean()), "ci_crashes": ci(dp, rng), "ci_edges": cluster_ci(dp, D.case_edges, rng),
            "within_delta": float(dw.mean()), "within_ci": ci(dw, rng)}


# ------------------------------------------------------------------ parts
def part_m1m7(D, rng):
    res = {"arms": {}, "paired": {}, "learning_curve": []}
    ev = {}
    for name, mask in D.pos.items():
        s, _ = D.fit(mask, mask.sum() / D.units[2024])
        ev[name] = D.evaluate(s)
        res["arms"][name] = {"positives": int(mask.sum()), **strip(ev[name])}
        print("m1m7", name, round(ev[name]["auc"], 4), flush=True)
    for a, b in [("union_b30", "official"), ("union_hour", "official"), ("official+wazeonly_b30", "official"),
                 ("official+wazeonly_hour", "official"), ("wazeonly_b30", "official"), ("wazeonly_hour", "official"),
                 ("union_b30", "reliable_b30"), ("union_hour", "reliable_hour"), ("reliable_hour", "reliable_b30"),
                 ("union_hour", "union_b30"), ("official", "reliable_hour")]:
        res["paired"][f"{a}_minus_{b}"] = paired(ev[a], ev[b], D, rng)
    # learning curves: PN, positives subsampled without replacement
    for source, sizes in (("official", (250, 484, 750, 1150)),
                          ("union_b30", (250, 484, 750, 1150, 2000, 3000, 4591)),
                          ("union_hour", (250, 484, 750, 1150, 2000, 3000, 4591)),
                          ("wazeonly_hour", (250, 484, 750, 1150, 2000, 4107))):
        idx = np.flatnonzero(D.pos[source])
        for n in sizes:
            for r in range(5 if n < len(idx) else 1):
                pick = np.zeros(len(D.train), bool)
                pick[rng.choice(idx, size=min(n, len(idx)), replace=False)] = True
                s, _ = D.fit(pick, pick.sum() / D.units[2024], seeds=2, seed0=SEED + 100 * r)
                e = D.evaluate(s)
                res["learning_curve"].append({"source": source, "n": int(pick.sum()), "rep": r, "auc": e["auc"], "within": e["within"]})
        print("curve", source, flush=True)
    # anchor agreement for confirmed events
    return res


def part_m2(D, rng):
    res = {"prior_factor": {}, "stationarity": {}, "quarters": {}, "deciles": {}}
    m = D.pos["reliable_b30"]
    keep = {}
    for f in (0.25, 0.5, 1.0, 2.0, 4.0):
        s, su = D.fit(m, D.prior_rec * f, mode="nnpu")
        e = D.evaluate(s)
        res["prior_factor"][str(f)] = {**strip(e), "mean_score_on_training_U_over_prior": float(su.mean() / (D.prior_rec * f))}
        if f == 1.0:
            keep["nnpu"] = s
    s, su = D.fit(m, m.sum() / D.units[2024], mode="pn")
    res["stationarity"]["pn_reliable"] = {"mean_on_training_U_over_labeled_rate": float(su.mean() / (m.sum() / D.units[2024]))}
    res["stationarity"]["nnpu_reliable"] = res["prior_factor"]["1.0"]["mean_score_on_training_U_over_prior"]
    keep["pn_reliable"] = s
    keep["official"], _ = D.fit(D.pos["official"], D.prior_rec)
    # level by quarter of 2025 (base-rate drift within the evaluation year)
    q_u = D.test.loc[D.t_m, "t_end"].dt.quarter.to_numpy()
    q_c = D.test.loc[D.t_c, "t_end"].dt.quarter.to_numpy()
    for name, s in keep.items():
        hu, hc = hourly(s[D.t_m]), hourly(s[D.t_c])
        res["quarters"][name] = {}
        for q in (1, 2, 3, 4):
            share = (q_u == q).mean()
            obs = int((q_c == q).sum())
            res["quarters"][name][str(q)] = {"observed": obs, "predicted": float(hu[q_u == q].mean() * D.n_hours * share),
                                            "ratio": float(hu[q_u == q].mean() * D.n_hours * share / obs) if obs else None}
        # deciles with bootstrap intervals (units and crashes resampled)
        edges = np.quantile(hu, np.linspace(0, 1, 11))
        du, dc = np.digitize(hu, edges[1:-1]), np.digitize(hc, edges[1:-1])
        rows = []
        for d in range(10):
            pred = float(hu[du == d].mean())
            obs = float((dc == d).sum() / (D.n_hours * (du == d).mean()))
            draws = []
            for _ in range(500):
                bu = rng.integers(0, len(hu), len(hu))
                bc = rng.integers(0, len(hc), len(hc))
                draws.append((dc[bc] == d).sum() / (D.n_hours * (du[bu] == d).mean()))
            rows.append({"decile": d + 1, "crashes": int((dc == d).sum()), "predicted": pred, "observed": obs,
                         "observed_ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]})
        res["deciles"][name] = rows
    return res


def week_blocks(D, rng, share=0.2, gap_h=2):
    t = D.train["t_end"]
    week = t.dt.isocalendar().week.to_numpy().astype(int)
    weeks = np.unique(week)
    held = rng.choice(weeks, size=int(round(share * len(weeks))), replace=False)
    hold = np.isin(week, held).astype(int)
    # gap: training rows within gap_h hours of a held-out row's time on either side are dropped
    ns = t.to_numpy().astype("datetime64[ns]").astype(np.int64)
    held_times = np.sort(ns[hold == 1])
    tt = ns
    pos = np.searchsorted(held_times, tt)
    near = np.full(len(tt), np.inf)
    for off in (-1, 0):
        j = np.clip(pos + off, 0, len(held_times) - 1)
        near = np.minimum(near, np.abs(tt - held_times[j]))
    gap = (hold == 0) & (near <= gap_h * 3600e9)
    hold[gap] = 2
    return hold


def part_m5(D, rng):
    res = {"week_blocks": {}, "dev_split_2024": {}}
    arms = {"official": ("official", "pn", None), "reliable_b30": ("reliable_b30", "pn", None),
            "nnpu_reliable": ("reliable_b30", "nnpu", None), "union_b30": ("union_b30", "pn", None),
            "mswspu": ("union_b30", "nnpu", "cred")}
    hold = week_blocks(D, rng)
    ev_r, ev_b = {}, {}
    for name, (pos, mode, w) in arms.items():
        m = D.pos[pos]
        prior = (m.sum() / D.units[2024]) if mode == "pn" else (D.prior_rec if pos == "reliable_b30" else D.prior_all)
        weight = D.cred if w == "cred" else None
        s_r, _ = D.fit(m, prior, mode, weight)
        s_b, _ = D.fit(m, prior, mode, weight, hold=hold)
        ev_r[name], ev_b[name] = D.evaluate(s_r), D.evaluate(s_b)
        res["week_blocks"][name] = {"random_holdout": strip(ev_r[name]), "week_block_holdout": strip(ev_b[name])}
    res["week_blocks"]["paired_week_blocks"] = {f"{a}_minus_{b}": paired(ev_b[a], ev_b[b], D, rng)
                                                for a, b in [("union_b30", "official"), ("mswspu", "official"),
                                                             ("union_b30", "reliable_b30"), ("nnpu_reliable", "reliable_b30")]}
    res["week_blocks"]["gap_rows_dropped"] = int((hold == 2).sum())
    # development split inside 2024: train Jan-Aug, evaluate Sep-Dec (1-day gap); never-evaluated labels
    t = D.train["t_end"]
    cut = pd.Timestamp(2024, 9, 1)
    tr_rows = (t < cut - pd.Timedelta(days=1)).to_numpy()
    ev_rows = (t >= cut).to_numpy()
    k = D.train["kind"].to_numpy()
    m_ev, c_ev = (k == "marginal") & ev_rows, (k == "official_case") & ev_rows
    sets_ev = np.isin(k, ["official_case", "official_control"]) & ev_rows
    share_tr = (D.is_u & tr_rows).sum() / D.is_u.sum()
    out = {}
    for name, (pos, mode, w) in arms.items():
        m = D.pos[pos]
        units_tr = D.units[2024] * share_tr
        prior = ((m & tr_rows).sum() / units_tr) if mode == "pn" else (
            (D.pos["official"] & tr_rows).sum() / units_tr * (1 if pos == "reliable_b30" else p6.ALL_TO_INJURY_RATIO))
        weight = D.cred if w == "cred" else None
        s, _ = D.fit(m, prior, mode, weight, rows=tr_rows, x_eval=D.xtr)
        pl = placements(s[c_ev], np.sort(s[m_ev]))
        sub = D.train.loc[sets_ev]
        wv = within_stratum(sub, s[sets_ev], (sub["kind"] == "official_case").to_numpy())
        out[name] = {"_pl": pl, "_w": wv, "auc": float(pl.mean()), "within": float(wv.mean())}
    edges_ev = D.train.loc[c_ev, "edge_id"].to_numpy()
    res["dev_split_2024"]["arms"] = {n: strip(v) for n, v in out.items()}
    res["dev_split_2024"]["crash_hours_evaluated"] = int(c_ev.sum())
    res["dev_split_2024"]["paired"] = {}
    for a, b in [("union_b30", "official"), ("mswspu", "official"), ("union_b30", "reliable_b30"), ("nnpu_reliable", "reliable_b30")]:
        dp = out[a]["_pl"] - out[b]["_pl"]
        res["dev_split_2024"]["paired"][f"{a}_minus_{b}"] = {"auc_delta": float(dp.mean()), "ci_crashes": ci(dp, rng),
                                                            "ci_edges": cluster_ci(dp, edges_ev, rng)}
    return res


def part_m6(D, rng):
    pred = pd.read_parquet(V2 / "phase6" / "minute_pu" / "test_predictions_2025.parquet")
    kind = pred["kind"].to_numpy()
    m, c = kind == "marginal", kind == "official_case"
    case = pred.loc[c].merge(D.test.loc[D.t_c, ["row_id", "route"]], on="row_id", how="left")
    groups = {"edge": case["edge_id"].to_numpy(), "route": road(case["route"]).to_numpy(),
              "week": case["t_end"].dt.isocalendar().week.to_numpy().astype(int),
              "day": case["t_end"].dt.dayofyear.to_numpy(),
              "route_x_week": (road(case["route"]) + "_" + case["t_end"].dt.isocalendar().week.astype(str)).to_numpy()}
    comps = [("pn_union__b30", "pn_official_hour__b30"), ("mswspu_credibility__b30", "pn_official_hour__b30"),
             ("pn_union__b30", "pn_reliable__b30"), ("nnpu_reliable__b30", "pn_reliable__b30"),
             ("mswspu_credibility__b30", "pn_union__b30"), ("nnpu_reliable_propensity__b30", "nnpu_reliable__b30")]
    res = {"block_bootstrap": {}, "resampled_U": {}, "equivalence_margin": 0.01, "tost": {}}
    for a, b in comps:
        pa = placements(pred.loc[c, a].to_numpy(), np.sort(pred.loc[m, a].to_numpy()))
        pb = placements(pred.loc[c, b].to_numpy(), np.sort(pred.loc[m, b].to_numpy()))
        d = pa - pb
        entry = {"delta": float(d.mean()), "crashes": ci(d, rng)}
        for g, v in groups.items():
            entry[g] = cluster_ci(d, v, rng)
        res["block_bootstrap"][f"{a}_minus_{b}"] = entry
        # two-sample bootstrap: crashes and the sampled units both resampled
        ua, ub = pred.loc[m, a].to_numpy(), pred.loc[m, b].to_numpy()
        ca, cb = pred.loc[c, a].to_numpy(), pred.loc[c, b].to_numpy()
        draws = []
        for _ in range(300):
            iu = rng.integers(0, len(ua), len(ua))
            ic = rng.integers(0, len(ca), len(ca))
            draws.append(placements(ca[ic], np.sort(ua[iu])).mean() - placements(cb[ic], np.sort(ub[iu])).mean())
        res["resampled_U"][f"{a}_minus_{b}"] = [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]
        lo, hi = cluster_ci(d, groups["edge"], rng, q=(0.05, 0.95))
        res["tost"][f"{a}_minus_{b}"] = {"ci90_edges": [lo, hi], "equivalent_within_0.01": bool(lo > -0.01 and hi < 0.01),
                                         "superior_by_more_than_0.01": bool(lo > 0.01)}
    # refit variability: three independent refits with disjoint seed sets
    refits = {"official": [], "union_b30": [], "reliable_b30": [], "nnpu_reliable": []}
    for r in range(3):
        s0 = SEED + 1000 * (r + 1)
        for name, (pos, mode) in {"official": ("official", "pn"), "union_b30": ("union_b30", "pn"),
                                  "reliable_b30": ("reliable_b30", "pn"), "nnpu_reliable": ("reliable_b30", "nnpu")}.items():
            mk = D.pos[pos]
            prior = mk.sum() / D.units[2024] if mode == "pn" else D.prior_rec
            s, _ = D.fit(mk, prior, mode, seed0=s0)
            refits[name].append(D.evaluate(s)["auc"])
    res["refit_variability"] = {n: {"aucs": v, "sd": float(np.std(v, ddof=1))} for n, v in refits.items()}
    res["refit_variability"]["union_minus_official"] = [a - b for a, b in zip(refits["union_b30"], refits["official"])]
    res["refit_variability"]["nnpu_minus_pn_reliable"] = [a - b for a, b in zip(refits["nnpu_reliable"], refits["reliable_b30"])]
    return res


def part_m8(D, rng):
    res = {}
    tr = D.train
    k = tr["kind"].to_numpy()
    flags = pd.read_parquet(OUT / "event_flags_2024.parquet")
    summary = json.loads((V2 / "phase6" / "minute_dataset_summary.json").read_text())["years"]["2024"]
    # accounting
    conf = tr.loc[(k == "waze_case") & (tr["buffer_min"] == 30) & tr["official_confirmed"]]
    off = tr.loc[k == "official_case", ["edge_id", "t_end", "waze_matched"]].rename(columns={"t_end": "H"})
    mm = conf.merge(off, on="edge_id")
    mm["lag"] = (mm["reference_time"] - mm["H"]).dt.total_seconds() / 60
    mm = mm.loc[mm["lag"].between(-15, 105)]
    res["accounting"] = {
        "network_clusters_with_official_record_incl_repeats_full_year": summary["police_confirmed_among_network_clusters"],
        "of_these_repeats_5km_3h": None,
        "waze_timed_events_with_official_record": int(len(conf)),
        "official_crash_hours": int((k == "official_case").sum()),
        "official_crash_hours_with_waze_cluster": int(off["waze_matched"].fillna(False).astype(bool).sum()),
        "confirmed_events_pointing_to_a_crash_hour_on_same_edge": int(mm["cluster_id"].nunique()),
        "distinct_crash_hours_hit_by_confirmed_events": int(mm[["edge_id", "H"]].drop_duplicates().shape[0]),
        "confirmed_events_whose_crash_is_on_another_edge_or_off_network": int(len(conf) - mm["cluster_id"].nunique()),
        "crash_hours_hit_by_more_than_one_event": int((mm.groupby(["edge_id", "H"]).size() > 1).sum()),
        "events_hitting_more_than_one_crash_hour": int((mm.groupby("cluster_id").size() > 1).sum()),
    }
    allconf = flags.loc[flags["official_confirmed"].fillna(False).astype(bool)]
    res["accounting"]["confirmed_events_in_eligible_window_incl_repeats"] = int(len(allconf))
    res["accounting"]["of_these_repeats_5km_3h"] = int(allconf["repeat_5km_3h"].sum())
    # weight distributions
    pos_cred = D.cred[D.pos["union_b30"]]
    weak = D.cred[D.pos["wazeonly_b30"]]
    inv = D.inv_prop[D.pos["reliable_b30"]]
    kish = lambda w: float(w.sum() ** 2 / (w ** 2).sum())
    res["weights"] = {
        "credibility_weak": {"quantiles": [float(np.quantile(weak, q)) for q in (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1)],
                             "share_at_floor_0.05": float((weak <= 0.0500001).mean()), "kish_ess_union": kish(pos_cred),
                             "n_union": int(len(pos_cred))},
        "inverse_propensity_reliable": {"quantiles": [float(np.quantile(inv, q)) for q in (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1)],
                                        "share_at_cap_10": float((inv >= 9.9999).mean()), "kish_ess": kish(inv), "n": int(len(inv))},
    }
    # truncation and cross-fitting of the propensity
    cols = ["hour_sin", "hour_cos", "weekend", "up_flow_w", "dn_flow_w", "up_mean_flow_2024", "length_km"]
    prep = Preprocessor(cols).fit(D.xtr_raw.loc[D.is_u])
    oc = k == "official_case"
    y = tr.loc[oc, "waze_matched"].fillna(False).astype(bool).to_numpy()
    X_oc = prep.transform(D.xtr_raw.loc[oc])
    X_all = prep.transform(D.xtr_raw)
    lr = LogisticRegression(max_iter=1000).fit(X_oc, y)
    e_in = lr.predict_proba(X_all)[:, 1]
    # cross-fitting: 5 folds over the official crash hours, grouped by edge; Waze rows take the
    # model that did not see crash hours on their edge
    edges_oc = tr.loc[oc, "edge_id"].to_numpy()
    uniq = np.unique(edges_oc)
    fold_of_edge = dict(zip(uniq, rng.integers(0, 5, len(uniq))))
    e_cf = e_in.copy()
    fold_rows = np.array([fold_of_edge.get(e, -1) for e in tr["edge_id"].to_numpy()])
    for f in range(5):
        sel = np.array([fold_of_edge[e] != f for e in edges_oc])
        mdl = LogisticRegression(max_iter=1000).fit(X_oc[sel], y[sel])
        rows = fold_rows == f
        e_cf[rows] = mdl.predict_proba(X_all[rows])[:, 1]
    test = D.test
    not_rep = D.t_c & ~test["waze_matched"].fillna(False).astype(bool).to_numpy()
    e_test = lr.predict_proba(prep.transform(raw_features(test)))[:, 1]
    thirds = np.quantile(e_test[D.t_m], [1 / 3, 2 / 3])
    third_u = np.digitize(e_test[D.t_m], thirds)
    third_c = np.digitize(e_test[D.t_c], thirds)
    m = D.pos["reliable_b30"]
    out = {}
    base_s, _ = D.fit(m, D.prior_rec, "nnpu")
    variants = {"none": None}
    for clip in (0.05, 0.1, 0.2, 0.3):
        variants[f"in_sample_clip_{clip}"] = 1 / np.clip(e_in, clip, 1)
    variants["cross_fitted_clip_0.1"] = 1 / np.clip(e_cf, 0.1, 1)
    for name, w in variants.items():
        s = base_s if w is None else D.fit(m, D.prior_rec, "nnpu", weight=w)[0]
        pl_all = placements(s[D.t_c], np.sort(s[D.t_m]))
        pl_nr = placements(s[not_rep], np.sort(s[D.t_m]))
        hu, hc = hourly(s[D.t_m]), hourly(s[D.t_c])
        lev = {}
        for t3 in range(3):
            share = (third_u == t3).mean()
            lev[str(t3)] = float(hu[third_u == t3].mean() * D.n_hours * share / (third_c == t3).sum())
        wpos = np.ones(m.sum()) if w is None else w[m]
        out[name] = {"auc_all": float(pl_all.mean()), "auc_not_reported": float(pl_nr.mean()),
                     "level_by_propensity_third": lev, "kish_ess": kish(wpos)}
    res["propensity_variants"] = out
    return res


def part_m9(D, rng):
    res = {}
    k = D.train["kind"].to_numpy()
    b = D.train["buffer_min"].to_numpy()
    conf = D.train["official_confirmed"].to_numpy()
    base = (k == "waze_case") & (b == 30)
    rep = k == "waze_repeat"
    sets = {
        "5km_3h (primary)": base,
        "none": base | rep,
        "2km_1h": base | (rep & ~D.train["repeat_2km_1h"].fillna(False).astype(bool).to_numpy()),
        "1km_0.5h": base | (rep & ~D.train["repeat_1km_0.5h"].fillna(False).astype(bool).to_numpy()),
    }
    ev = {}
    for name, mask in sets.items():
        s, _ = D.fit(mask, mask.sum() / D.units[2024])
        ev[name] = D.evaluate(s)
        r, _ = D.fit(mask & conf, (mask & conf).sum() / D.units[2024])
        er = D.evaluate(r)
        res[name] = {"union_positives": int(mask.sum()), "reliable_positives": int((mask & conf).sum()),
                     "union": strip(ev[name]), "reliable": {"auc": er["auc"], "within": er["within"]}}
    res["paired_vs_primary"] = {n: paired(ev[n], ev["5km_3h (primary)"], D, rng) for n in sets if n != "5km_3h (primary)"}
    # one-to-one matching: each official crash hour on the same edge keeps only its first Waze event
    tr = D.train
    rel = tr.loc[base & conf]
    off = tr.loc[k == "official_case", ["edge_id", "t_end"]].rename(columns={"t_end": "H"})
    mm = rel.reset_index().merge(off, on="edge_id")
    mm["lag"] = (mm["reference_time"] - mm["H"]).dt.total_seconds() / 60
    mm = mm.loc[mm["lag"].between(-15, 105)].sort_values("lag")
    first = mm.drop_duplicates(["edge_id", "H"]).drop_duplicates("index")["index"].to_numpy()
    one = np.zeros(len(tr), bool)
    one[first] = True
    e1, _ = D.fit(one, one.sum() / D.units[2024])
    n1, _ = D.fit(one, D.prior_rec, "nnpu")
    res["one_to_one"] = {"reliable_positives": int(one.sum()), "pn": strip(D.evaluate(e1)), "nnpu": strip(D.evaluate(n1))}
    # union with one-to-one reliable (other confirmed events become weak positives: union unchanged)
    return res


def part_m11(D, rng):
    pred = pd.read_parquet(V2 / "phase6" / "minute_pu" / "test_predictions_2025.parquet")
    two = pd.read_parquet(V2 / "phase7" / "two_stage_predictions_2025.parquet")
    kind, tkind = pred["kind"].to_numpy(), two["kind"].to_numpy()
    sources = {"pn_reliable": (pred, kind, "pn_reliable__b30"), "nnpu_reliable": (pred, kind, "nnpu_reliable__b30"),
               "nnpu_reliable_propensity": (pred, kind, "nnpu_reliable_propensity__b30"),
               "pn_official": (pred, kind, "pn_official_hour__b30"), "pn_union": (pred, kind, "pn_union__b30"),
               "mswspu": (pred, kind, "mswspu_credibility__b30"),
               "two_stage_official": (two, tkind, "pn_official_hour__context+frequency+traffic"),
               "frequency_prior_alone": (two, tkind, "frequency_prior_alone")}
    n_hours = D.n_hours
    base = 1150 / (D.units[2024] / 12)
    res = {"budget": {}, "average_precision": {}, "net_benefit": {}}
    for name, (frame, kd, col) in sources.items():
        s = frame[col].to_numpy()
        su, sc = np.sort(s[kd == "marginal"]), s[kd == "official_case"]
        res["budget"][name] = {}
        for share in (0.01, 0.02, 0.03, 0.063, 0.10):
            thr = np.quantile(su, 1 - share)
            res["budget"][name][str(share)] = float((sc > thr).mean())
        # AP at natural prevalence: precision at each crash's score, negatives from the unit sample
        order = np.sort(sc)[::-1]
        tp = np.arange(1, len(order) + 1)
        flagged = (len(su) - np.searchsorted(su, order, side="left")) / len(su) * n_hours
        prec = tp / np.maximum(flagged, tp)
        res["average_precision"][name] = float(prec.mean())
        if name in ("pn_reliable", "nnpu_reliable", "nnpu_reliable_propensity", "pn_official"):
            hu, hc = hourly(s[kd == "marginal"]), hourly(sc)
            nb = {}
            for kx in (1, 2, 3, 5, 10):
                pt = kx * base
                tpr = (hc >= pt).sum() / n_hours
                fpr = (hu >= pt).mean() - tpr
                nb[str(kx)] = float(tpr - fpr * pt / (1 - pt))
            res["net_benefit"][name] = nb
    prev = len(pred.loc[kind == "official_case"]) / n_hours
    res["net_benefit"]["treat_all"] = {str(kx): float(prev - (1 - prev) * kx * base / (1 - kx * base)) for kx in (1, 2, 3, 5, 10)}
    res["prevalence_per_edge_hour"] = prev
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="m11,m6,m8,m2,m9,m1m7,m5")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    D = Data()
    print("data ready", {n: int(v.sum()) for n, v in D.pos.items()}, flush=True)
    for part in args.parts.split(","):
        t0 = time.perf_counter()
        rng = np.random.default_rng(SEED + sum(map(ord, part)))
        res = globals()[f"part_{part}"](D, rng)
        (OUT / f"{part}.json").write_text(json.dumps(res, indent=2, default=float))
        print(f"{part} done in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
