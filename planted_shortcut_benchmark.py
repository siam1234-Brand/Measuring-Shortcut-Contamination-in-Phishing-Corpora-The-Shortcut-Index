"""Controlled validation of the Shortcut Index (SI) with planted ground truth.

Regenerates Sec. V-E and Table IV of "The Shortcut Index: Screening Phishing
Corpora for Shortcut Contamination" (ICCIT 2026).  Every corpus here is synthetic, so the truth
(is a shortcut present, and which attributes carry it) is known.  The results
therefore test what SI responds to; they say nothing about real corpora.

Usage:  python -W ignore planted_shortcut_benchmark.py [--seeds 20] [--out results] [--procs 2]
Needs:  numpy, pandas, scikit-learn, scipy, mlxtend, problexity,
        and shortcut_index.py (the reference implementation) on the path.

Scenarios (n = 2000 records, 15 background attributes, seed-controlled)
  clean:         N  no label signal
                 W  weak signal spread over all attributes (shift 0.35 sd)
                 W2 stronger signal spread over all attributes (0.60 sd)
  genuine:       G  one genuinely informative attribute (shift 3 sd); not a
                    construction artefact, reported separately, because no
                    single-corpus screen can tell it from a shortcut
  contaminated:  S1 one label proxy, agrees with the label for 90-100% of rows
                 S2 five redundant proxies, 85-95% agreement each
                 S3 conjunction: a two-attribute region (about 18% of rows)
                    that is pure, outside which the label is a coin flip
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import problexity as px
from shortcut_index import shortcut_index  # noqa: E402

warnings.filterwarnings("ignore")
N, D = 2000, 15


# ---------------------------------------------------------------- generators
def _background(rng, y, shift=0.0, n=N, d=D):
    """15 attributes: 5 Gaussian, 5 log-normal, 5 Poisson counts."""
    z = rng.normal(size=(n, d)) + shift * y[:, None]
    X = np.empty_like(z)
    X[:, :5] = z[:, :5]
    X[:, 5:10] = np.exp(z[:, 5:10])
    X[:, 10:] = rng.poisson(3.0 * np.exp(0.5 * z[:, 10:]))
    return X


def _proxy(rng, y, q):
    """Label proxy: values 1-5 for class 0, 7-10 for class 1, flipped w.p. 1-q.

    The same shape as the urgency_score attribute of the synthetic corpus.
    """
    flip = rng.random(len(y)) > q
    cls = np.where(flip, 1 - y, y)
    return np.where(cls == 1, rng.integers(7, 11, len(y)),
                    rng.integers(1, 6, len(y))).astype(float)


def make(scenario, seed, q=None, n_proxies=5, removed=0, d=D):
    rng = np.random.default_rng(10_000 + seed)
    y = rng.integers(0, 2, N)
    cols = {}
    if scenario == "N":
        X = _background(rng, y, d=d)
    elif scenario == "W":
        X = _background(rng, y, 0.35, d=d)
    elif scenario == "W2":
        X = _background(rng, y, 0.60, d=d)
    elif scenario == "G":
        X = _background(rng, y)
        X[:, 0] = rng.normal(size=N) + 3.0 * y
    elif scenario == "S1":
        X = _background(rng, y, d=d)
        cols["proxy"] = _proxy(rng, y, rng.uniform(0.90, 1.0))
    elif scenario == "S2":
        X = _background(rng, y)
        for i in range(n_proxies):
            cols[f"proxy{i}"] = _proxy(rng, y, rng.uniform(0.85, 0.95))
        for i in range(removed):               # progressive ablation
            cols.pop(f"proxy{i}")
    elif scenario == "S3":
        X = _background(rng, y)                # features first, label after
        ra = rng.normal(size=N)
        rb = rng.normal(size=N)
        region = (ra > np.quantile(ra, 0.55)) & (rb > np.quantile(rb, 0.60))
        y = np.where(region, 1, rng.integers(0, 2, N))
        # outside the region the label stays a coin flip
        X[:, 0], X[:, 1] = ra, rb
    elif scenario == "DOSE":
        X = _background(rng, y)
        cols["proxy"] = _proxy(rng, y, q)
    else:
        raise ValueError(scenario)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    for k, v in cols.items():
        df[k] = v
    return df, y


# ---------------------------------------------------------------- detectors
def det_single_auc(X, y):
    return max(abs(2 * roc_auc_score(y, X[c]) - 1) for c in X)


def det_mi(X, y):
    p = y.mean()
    h = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    return float(mutual_info_classif(X, y, random_state=0).max() / h)


def det_tree_imp(X, y):
    t = DecisionTreeClassifier(criterion="entropy", max_depth=15,
                               random_state=0).fit(X, y)
    return float(t.feature_importances_.max())


def det_drop_one(X, y):
    """Largest fall in 5-fold balanced accuracy when one attribute is removed."""
    cv = StratifiedKFold(5, shuffle=True, random_state=0)

    def acc(cols):
        return cross_val_score(
            DecisionTreeClassifier(max_depth=8, random_state=0),
            X[cols], y, cv=cv, scoring="balanced_accuracy").mean()

    base = acc(list(X.columns))
    return max(base - acc([c for c in X.columns if c != d]) for d in X.columns)


def score_corpus(args):
    scenario, seed, kw = args
    kw = dict(kw)
    null = kw.pop("null", False)
    X, y = make(scenario, seed, **kw)
    r = shortcut_index(X, y)
    Xa = X.to_numpy()
    out = dict(scenario=scenario, seed=seed, **kw,
               SI=r.si, c1=r.c1, c2=r.c2, c3=r.c3,
               best_feature=r.best_feature,
               max_single_auc=det_single_auc(X, y),
               max_mi=det_mi(X, y),
               tree_importance=det_tree_imp(X, y),
               drop_one=det_drop_one(X, y),
               F3=1 - px.classification.f3(Xa, y),
               L1=1 - px.classification.l1(Xa, y),
               in_domain_tree=cross_val_score(
                   DecisionTreeClassifier(criterion="entropy", max_depth=15,
                                          random_state=0),
                   X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                   scoring="balanced_accuracy").mean())
    if null:
        out["SI_null"] = shortcut_index(X, y, shuffle_labels=True).si
    return out


def dose_job(args):
    q, seed = args
    X, y = make("DOSE", seed, q=q)
    r = shortcut_index(X, y)
    rn = shortcut_index(X, y, shuffle_labels=True)
    return dict(q=q, seed=seed, SI=r.si, c1=r.c1, c2=r.c2, c3=r.c3, SI_null=rn.si)


def ablate_job(args):
    removed, seed = args
    X, y = make("S2", seed, removed=removed)
    r = shortcut_index(X, y)
    return dict(removed=removed, seed=seed, SI=r.si)


def _auc(p, n):
    r = rankdata(np.r_[p, n])
    return (r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n))


def auc_ci(pos, neg, B=2000, seed=0):
    pos, neg = np.asarray(pos), np.asarray(neg)
    f = _auc
    rng = np.random.default_rng(seed)
    bs = [f(rng.choice(pos, len(pos)), rng.choice(neg, len(neg)))
          for _ in range(B)]
    return f(pos, neg), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default="results")
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    scen = ["N", "W", "W2", "G", "S1", "S2", "S3"]
    jobs = [(s, k, {"null": s in ("N", "S1")}) for s in scen
            for k in range(a.seeds)]
    with Pool(a.procs) as p:
        rows = p.map(score_corpus, jobs, chunksize=2)
        df = pd.DataFrame(rows)
        df.to_csv(f"{a.out}/benchmark.csv", index=False)

        dose = pd.DataFrame(p.map(
            dose_job, [(q, k) for q in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)
                       for k in range(a.seeds)], chunksize=2))
        dose.to_csv(f"{a.out}/dose_response.csv", index=False)

        abl = pd.DataFrame(p.map(
            ablate_job, [(j, k) for j in range(6) for k in range(a.seeds)],
            chunksize=2))
        abl.to_csv(f"{a.out}/progressive_ablation.csv", index=False)

    det = ["SI", "c1", "c2", "c3", "max_single_auc", "max_mi",
           "tree_importance", "drop_one", "F3", "L1"]
    clean = df[df.scenario.isin(["N", "W", "W2"])]
    summary = {"AUROC": {}, "scenario_means": {}}
    for d in det:
        summary["AUROC"][d] = {}
        for grp, names in (("S1", ["S1"]), ("S2", ["S2"]), ("S3", ["S3"]),
                           ("all", ["S1", "S2", "S3"]), ("G", ["G"])):
            pos = df[df.scenario.isin(names)][d]
            summary["AUROC"][d][grp] = [round(float(v), 3)
                                        for v in auc_ci(pos, clean[d])]
    def flat(frame):
        frame = frame.round(3)
        frame.columns = ["_".join(map(str, c)) for c in frame.columns]
        return json.loads(frame.to_json(orient="index"))

    summary["scenario_means"] = flat(df.groupby("scenario")[
        det + ["SI_null", "in_domain_tree"]].agg(["mean", "std"]))
    summary["dose"] = flat(dose.groupby("q")[["SI", "SI_null"]]
                           .agg(["mean", "std"]))
    summary["ablation"] = flat(abl.groupby("removed")[["SI"]]
                               .agg(["mean", "std"]))
    summary["flag_rate_SI_gt_0.8"] = {
        k: float(v) for k, v in df.assign(f=df.SI > 0.8)
        .groupby("scenario")["f"].mean().items()}
    json.dump(summary, open(f"{a.out}/summary.json", "w"), indent=1, default=str)
    json.dump(summary["AUROC"], open(f"{a.out}/auroc.json", "w"), indent=1, default=str)
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    sys.exit(main())
