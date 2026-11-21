"""Control for the restricted-label comparison (colleague's point 4).

The 484 reliable positives of Table 3 are official crashes that Waze also reported, so the
gain from adding the Waze-only events is mixed with the Waze reporting mechanism. This control
keeps a RANDOM 484 of the 1,150 official crash hours of 2024 and adds the same 4,107 Waze-only
events, under the protocol of phase11 part m1m7 (MLP, PN risk, prior = labelled rate, 5 seeds).

Fixed before running:
  primary    AUC(random 484 + Waze-only) - AUC(random 484), population ROC-AUC, 2025 crashes
  also       within-set concordance, level, share of crashes caught at a 6.3% alert budget,
             the gain on 2025 crashes Waze did and did not report
  repeats    10 random draws; both arms of a repeat share the same 484
  references reliable_b30 and union_b30 refitted with the seeds of m1m7 (sanity check on Table 3)
Exploratory: 2025 was examined during development.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from phase11_review_analyses import OUT as P11, SEED, V2, Data, cluster_ci, ci, paired, strip

OUT = V2 / "phase12"
BUDGET = 0.063


def measures(D, s):
    e = D.evaluate(s)
    cut = np.quantile(s[D.t_m], 1 - BUDGET)
    e["caught_at_budget"] = float((s[D.t_c] >= cut).mean())
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    D = Data()
    official = np.flatnonzero(D.pos["official"])
    n_keep = int(D.pos["reliable_b30"].sum())
    weak = D.pos["wazeonly_b30"]
    print("positives", len(official), n_keep, int(weak.sum()), flush=True)
    waze_seen = D.test.loc[D.t_c, "waze_matched"].fillna(False).astype(bool).to_numpy()
    rate = lambda m: m.sum() / D.units[2024]

    refs = {}
    for name in ("reliable_b30", "union_b30", "official", "wazeonly_b30", "official+wazeonly_b30"):
        s, _ = D.fit(D.pos[name], rate(D.pos[name]))
        refs[name] = measures(D, s)
        print("ref", name, round(refs[name]["auc"], 4), flush=True)

    rng_draw = np.random.default_rng(SEED + 12)
    rows, store = [], {"random": [], "random+waze": []}
    for r in range(args.reps):
        keep = np.zeros(len(D.train), bool)
        keep[rng_draw.choice(official, size=n_keep, replace=False)] = True
        for arm, mask in (("random", keep), ("random+waze", keep | weak)):
            s, _ = D.fit(mask, rate(mask), seed0=SEED + 1000 * (r + 1))
            e = measures(D, s)
            store[arm].append(e)
            rows.append({"rep": r, "arm": arm, "positives": int(mask.sum()), **strip(e)})
        a, b = store["random+waze"][-1], store["random"][-1]
        print(f"rep {r}: random {b['auc']:.4f} +waze {a['auc']:.4f} gain {a['auc'] - b['auc']:+.4f} "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)

    avg = {k: {"_pl": np.mean([e["_pl"] for e in v], axis=0), "_w": np.mean([e["_w"] for e in v], axis=0)} for k, v in store.items()}
    rng = np.random.default_rng(SEED + 13)
    res = {"note": "exploratory; see module docstring", "reps": args.reps, "n_keep": n_keep,
           "references": {k: strip(v) for k, v in refs.items()}, "runs": rows,
           "summary": {arm: {m: {"mean": float(np.mean([e[m] for e in v])), "sd": float(np.std([e[m] for e in v], ddof=1))}
                             for m in ("auc", "within", "level", "log_skill", "slope", "caught_at_budget")}
                       for arm, v in store.items()},
           "gain_random": paired(avg["random+waze"], avg["random"], D, rng),
           "gain_reliable": paired(refs["union_b30"], refs["reliable_b30"], D, rng),
           "gain_by_rep": [float(store["random+waze"][i]["auc"] - store["random"][i]["auc"]) for i in range(args.reps)]}
    d_gap = (avg["random+waze"]["_pl"] - avg["random"]["_pl"]) - (refs["union_b30"]["_pl"] - refs["reliable_b30"]["_pl"])
    res["gain_random_minus_gain_reliable"] = {"delta": float(d_gap.mean()), "ci_crashes": ci(d_gap, rng), "ci_edges": cluster_ci(d_gap, D.case_edges, rng)}
    dp = avg["random+waze"]["_pl"] - avg["random"]["_pl"]
    res["gain_random_by_waze_report"] = {
        "waze_reported": {"delta": float(dp[waze_seen].mean()), "ci_crashes": ci(dp[waze_seen], rng), "n": int(waze_seen.sum())},
        "not_waze_reported": {"delta": float(dp[~waze_seen].mean()), "ci_crashes": ci(dp[~waze_seen], rng), "n": int((~waze_seen).sum())}}
    res["random_vs_official"] = {arm: paired(avg[arm], refs["official"], D, rng) for arm in avg}
    drel = refs["union_b30"]["_pl"] - refs["reliable_b30"]["_pl"]
    res["gain_reliable_by_waze_report"] = {
        "waze_reported": {"delta": float(drel[waze_seen].mean()), "ci_crashes": ci(drel[waze_seen], rng), "n": int(waze_seen.sum())},
        "not_waze_reported": {"delta": float(drel[~waze_seen].mean()), "ci_crashes": ci(drel[~waze_seen], rng), "n": int((~waze_seen).sum())}}
    (OUT / "random42.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: res[k] for k in ("summary", "gain_random", "gain_reliable", "gain_random_minus_gain_reliable", "gain_random_by_waze_report", "gain_reliable_by_waze_report", "references")}, indent=1, default=float), flush=True)
    print(f"done in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
