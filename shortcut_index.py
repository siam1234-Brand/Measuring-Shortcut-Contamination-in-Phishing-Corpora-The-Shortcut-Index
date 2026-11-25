"""The Shortcut Index (SI): how much of a corpus's labels simple rules recover.

Reference implementation of the metric defined in Section III of
"The Shortcut Index: Screening Phishing Corpora for Shortcut Contamination"
(M. M. R. Siam, S. D. Dipon, S. R. Shanta; ICCIT 2026).

    SI(D) = (c1 + c2 + c3) / 3  in [0, 1]

c1 and c3 are estimated out of sample and rescaled against the majority-class
rate b = max(pi, 1 - pi), so a rule no better than always guessing the majority
class scores 0; c2 is the adjusted Rand index, which is already chance-corrected:

    c1  single-feature separability   best one-attribute threshold rule,
                                      chosen on the training folds and scored
                                      on the held-out fold
    c2  unsupervised recoverability   |ARI| of k-means (k=2) against the label;
                                      labels score, never fit
    c3  rule determinism              highest-confidence Apriori class rule,
                                      mined on the training folds and
                                      re-scored on the held-out fold

Nothing here reads a second corpus. A high SI (the paper uses SI > 0.8 as a
flag) means simple rules recover the labels; inspect the rule SI names, remove
artefact attributes and recompute. SI is a screen, not a forecast of
cross-corpus accuracy loss, and a corpus with genuinely easy signal also scores
high.

Implementation notes matching the paper: c3 tertile edges are fitted on the
training fold only; c2 uses a stratified subsample of 100,000 records for
larger corpora (seed 42).

Command line
------------
    python shortcut_index.py data.csv --label is_phishing
    python shortcut_index.py data.csv --label y --null       # shuffled-label test
    python shortcut_index.py data.csv --label y --json out.json

Library
-------
    from shortcut_index import shortcut_index
    result = shortcut_index(X, y)      # DataFrame / array, binary labels
    print(result.si, result.c1, result.c2, result.c3)

Requires numpy, pandas and scikit-learn. mlxtend is used for c3 when it is
installed; without it c3 falls back to an equivalent exhaustive search over
itemsets up to `max_len`, which is what the paper's settings bound anyway.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Paper defaults (Section III).
N_SPLITS = 5
SEED = 42
PERCENTILES = np.arange(1, 100)      # 1st to 99th percentile thresholds
KMEANS_RESTARTS = 10
MIN_SUPPORT = 0.05
MAX_LEN = 3
MIN_CONFIDENCE = 0.70
N_BINS = 3                           # equal-frequency tertiles


@dataclass
class SIResult:
    """One corpus scored. `raw_*` are before majority-baseline rescaling."""
    si: float
    c1: float
    c2: float
    c3: float
    raw_c1: float
    raw_c2: float
    raw_c3: float
    baseline: float
    n_rows: int
    n_features: int
    best_feature: str | None
    best_rule: str | None

    def __str__(self) -> str:
        return (
            f"SI = {self.si:.3f}\n"
            f"  c1 single-feature separability  {self.c1:.3f}"
            f"   (raw {self.raw_c1:.4f}, best: {self.best_feature})\n"
            f"  c2 unsupervised recoverability  {self.c2:.3f}"
            f"   (raw |ARI| {self.raw_c2:.4f})\n"
            f"  c3 rule determinism             {self.c3:.3f}"
            f"   (raw {self.raw_c3:.4f}, best: {self.best_rule})\n"
            f"  majority-class baseline         {self.baseline:.4f}"
            f"   over {self.n_rows:,} rows x {self.n_features} features"
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _rescale(score: float, baseline: float) -> float:
    """Map the majority-class rate to 0 and a perfect score to 1, clipped."""
    if baseline >= 1.0:
        return 0.0
    return float(np.clip((score - baseline) / (1.0 - baseline), 0.0, 1.0))


def _numeric_frame(X) -> pd.DataFrame:
    """Keep the numeric columns and fill gaps with the column median."""
    df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
    df.columns = [str(c) for c in df.columns]
    num = df.select_dtypes(include=[np.number]).copy()
    if num.empty:
        raise ValueError("no numeric columns to score")
    return num.fillna(num.median(numeric_only=True)).replace(
        [np.inf, -np.inf], 0.0)


def _check_binary(y) -> np.ndarray:
    y = np.asarray(y).ravel()
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError(f"expected two classes, found {len(classes)}")
    return (y == classes[1]).astype(int)


# ---------------------------------------------------------------------------
# c1: best single-feature threshold rule, cross-validated
# ---------------------------------------------------------------------------
def component_c1(X: pd.DataFrame, y: np.ndarray, n_splits: int = N_SPLITS,
                 seed: int = SEED) -> tuple[float, str | None]:
    """Mean held-out accuracy of the rule chosen on the training folds.

    The search is what makes this honest: the attribute, the threshold and the
    direction are all fixed on the training folds, then applied unchanged to
    the held-out fold. Selecting and scoring on the same rows would be
    optimistically biased, badly so on a small corpus.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    values = X.to_numpy(dtype=float)
    fold_scores, picks = [], []

    for train_idx, test_idx in cv.split(values, y):
        Xtr, ytr = values[train_idx], y[train_idx]
        Xte, yte = values[test_idx], y[test_idx]
        best = (-1.0, 0, 0.0, True)           # acc, column, threshold, above

        for j in range(Xtr.shape[1]):
            col = Xtr[:, j]
            thresholds = np.unique(np.percentile(col, PERCENTILES))
            if thresholds.size == 0:
                continue
            # vectorised over thresholds: (n_train, n_thresholds)
            above = col[:, None] > thresholds[None, :]
            acc_above = (above == ytr[:, None]).mean(axis=0)
            acc_below = 1.0 - acc_above
            k_above, k_below = int(acc_above.argmax()), int(acc_below.argmax())
            if acc_above[k_above] >= acc_below[k_below]:
                cand = (float(acc_above[k_above]), j,
                        float(thresholds[k_above]), True)
            else:
                cand = (float(acc_below[k_below]), j,
                        float(thresholds[k_below]), False)
            if cand[0] > best[0]:
                best = cand

        _, j, thr, above = best
        pred = (Xte[:, j] > thr) if above else (Xte[:, j] <= thr)
        fold_scores.append(float((pred == yte).mean()))
        picks.append((X.columns[j], thr, above))

    # report the attribute chosen most often across folds
    names = [p[0] for p in picks]
    best_name = max(set(names), key=names.count) if names else None
    return float(np.mean(fold_scores)), best_name


# ---------------------------------------------------------------------------
# c2: unsupervised recoverability
# ---------------------------------------------------------------------------
def component_c2(X: pd.DataFrame, y: np.ndarray, seed: int = SEED) -> float:
    """|ARI| between a 2-means partition and the label.

    The labels are used only to score the partition, never to fit it, so a high
    value means the two classes are already separated in the feature geometry
    before any supervision.
    """
    if len(X) > 100_000:
        from sklearn.model_selection import train_test_split
        _, idx = train_test_split(np.arange(len(X)), test_size=100_000,
                                  stratify=y, random_state=seed)
        X, y = X.iloc[idx], y[idx]
    Z = StandardScaler().fit_transform(X.to_numpy(dtype=float))
    km = KMeans(n_clusters=2, n_init=KMEANS_RESTARTS, random_state=seed)
    return float(abs(adjusted_rand_score(y, km.fit_predict(Z))))


# ---------------------------------------------------------------------------
# c3: rule determinism
# ---------------------------------------------------------------------------
def _fit_edges(X: pd.DataFrame, n_bins: int = N_BINS) -> dict:
    """Equal-frequency bin edges fitted on the rows given (the training fold)."""
    edges = {}
    for col in X.columns:
        s = X[col]
        if s.nunique() <= 2:
            edges[col] = None
            continue
        e = np.unique(np.quantile(s, np.linspace(0, 1, n_bins + 1)))
        edges[col] = e if len(e) > 2 else None
    return edges


def _apply_edges(X: pd.DataFrame, edges: dict) -> pd.DataFrame:
    out = {}
    for col in X.columns:
        e = edges[col]
        if e is None:
            out[col] = X[col].astype(str)
        else:
            out[col] = pd.cut(X[col], np.r_[-np.inf, e[1:-1], np.inf],
                              labels=False).astype(str)
    return pd.DataFrame(out, index=X.index)


def _onehot(bins: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    items = pd.get_dummies(bins, prefix_sep="=").astype(bool)
    items["__class__=1"] = y.astype(bool)
    items["__class__=0"] = ~y.astype(bool)
    return items


def _best_class_rule(items: pd.DataFrame, min_support: float, max_len: int,
                     min_confidence: float):
    """Highest-confidence rule whose consequent is a class, mined on `items`.

    Returns (antecedent_columns, class_column, confidence) or None. Uses
    mlxtend's apriori when available and an equivalent bounded search when not.
    """
    class_cols = ["__class__=1", "__class__=0"]
    feature_cols = [c for c in items.columns if c not in class_cols]
    n = len(items)
    if n == 0:
        return None

    try:
        from mlxtend.frequent_patterns import apriori, association_rules
        freq = apriori(items, min_support=min_support, use_colnames=True,
                       max_len=max_len)
        if freq.empty:
            return None
        rules = association_rules(freq, metric="confidence",
                                  min_threshold=min_confidence)
        rules = rules[rules["consequents"].apply(
            lambda c: len(c) == 1 and next(iter(c)) in class_cols)]
        if rules.empty:
            return None
        top = rules.loc[rules["confidence"].idxmax()]
        return (list(top["antecedents"]), next(iter(top["consequents"])),
                float(top["confidence"]))
    except ImportError:
        pass

    # Fallback: the paper's support threshold and length cap make an explicit
    # search over antecedents of size 1..max_len-1 tractable.
    from itertools import combinations
    arrays = {c: items[c].to_numpy() for c in feature_cols}
    cls = {c: items[c].to_numpy() for c in class_cols}
    best = None
    for size in range(1, max_len):
        for combo in combinations(feature_cols, size):
            mask = arrays[combo[0]].copy()
            for c in combo[1:]:
                mask &= arrays[c]
            support = mask.mean()
            if support < min_support:
                continue
            covered = mask.sum()
            for c in class_cols:
                conf = float((mask & cls[c]).sum() / covered)
                if conf >= min_confidence and (best is None or conf > best[2]):
                    best = (list(combo), c, conf)
    return best


def component_c3(X: pd.DataFrame, y: np.ndarray, n_splits: int = N_SPLITS,
                 seed: int = SEED) -> tuple[float, str | None]:
    """Mean held-out confidence of the strongest class rule.

    Without the support threshold a rule covering a single record could reach
    confidence 1.000, so the threshold is doing real work here.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    baseline = max(float(np.mean(y)), 1.0 - float(np.mean(y)))
    scores, descriptions = [], []

    for train_idx, test_idx in cv.split(X, y):
        edges = _fit_edges(X.iloc[train_idx])
        bins_tr = _apply_edges(X.iloc[train_idx], edges)
        bins_te = _apply_edges(X.iloc[test_idx], edges)
        tr = _onehot(bins_tr, y[train_idx])
        rule = _best_class_rule(tr, MIN_SUPPORT, MAX_LEN, MIN_CONFIDENCE)
        if rule is None:
            scores.append(baseline)       # no rule beats always-guess-majority
            continue
        antecedents, consequent, _ = rule

        te = _onehot(bins_te, y[test_idx])
        missing = [c for c in antecedents + [consequent] if c not in te.columns]
        if missing:
            scores.append(baseline)
            continue
        mask = te[antecedents[0]].to_numpy().copy()
        for c in antecedents[1:]:
            mask &= te[c].to_numpy()
        covered = int(mask.sum())
        if covered == 0:
            scores.append(baseline)
            continue
        held_out = float((mask & te[consequent].to_numpy()).sum() / covered)
        scores.append(held_out)
        descriptions.append(
            f"{' AND '.join(antecedents)} -> {consequent}")

    return float(np.mean(scores)), (descriptions[0] if descriptions else None)


# ---------------------------------------------------------------------------
# the index
# ---------------------------------------------------------------------------
def shortcut_index(X, y, n_splits: int = N_SPLITS, seed: int = SEED,
                   shuffle_labels: bool = False) -> SIResult:
    """Score one corpus. Set `shuffle_labels` to run the null test.

    Under permutation the feature distributions are untouched and only the
    label assignment changes, so a well-behaved index collapses towards zero.
    """
    Xn = _numeric_frame(X)
    yb = _check_binary(y)
    if len(Xn) != len(yb):
        raise ValueError("X and y have different lengths")
    if shuffle_labels:
        yb = np.random.default_rng(seed).permutation(yb)

    baseline = max(float(np.mean(yb)), 1.0 - float(np.mean(yb)))
    raw_c1, best_feature = component_c1(Xn, yb, n_splits, seed)
    raw_c2 = component_c2(Xn, yb, seed)
    raw_c3, best_rule = component_c3(Xn, yb, n_splits, seed)

    c1 = _rescale(raw_c1, baseline)
    c2 = float(np.clip(raw_c2, 0.0, 1.0))     # ARI is already baseline-free
    c3 = _rescale(raw_c3, baseline)

    return SIResult(
        si=(c1 + c2 + c3) / 3.0, c1=c1, c2=c2, c3=c3,
        raw_c1=raw_c1, raw_c2=raw_c2, raw_c3=raw_c3, baseline=baseline,
        n_rows=len(Xn), n_features=Xn.shape[1],
        best_feature=best_feature, best_rule=best_rule,
    )


def generalisation_gap(in_domain: float, external: Iterable[float]) -> float:
    """In-domain balanced accuracy minus mean external balanced accuracy (Sec. IV)."""
    ext = list(external)
    if not ext:
        raise ValueError("need at least one external accuracy")
    return float(in_domain) - float(np.mean(ext))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv", help="labelled corpus; one row per record")
    ap.add_argument("--label", required=True, help="name of the label column")
    ap.add_argument("--drop", nargs="*", default=[],
                    help="columns to exclude (ids, free text, raw URLs)")
    ap.add_argument("--folds", type=int, default=N_SPLITS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--null", action="store_true",
                    help="also run the shuffled-label null test")
    ap.add_argument("--json", metavar="PATH", help="write the result as JSON")
    args = ap.parse_args(argv)

    df = pd.read_csv(args.csv)
    if args.label not in df.columns:
        print(f"label column {args.label!r} not in {args.csv}", file=sys.stderr)
        return 2
    y = df[args.label]
    X = df.drop(columns=[args.label] + [c for c in args.drop if c in df.columns])

    result = shortcut_index(X, y, args.folds, args.seed)
    print(f"{args.csv}\n{result}")

    payload = {"corpus": args.csv, **asdict(result)}
    if args.null:
        null = shortcut_index(X, y, args.folds, args.seed, shuffle_labels=True)
        print(f"\nshuffled labels: SI = {null.si:.4f} "
              f"(c1 {null.c1:.3f}, c2 {null.c2:.3f}, c3 {null.c3:.3f})")
        drop = result.si - null.si
        print(f"the index drops by {drop:.4f}; a value that does not drop is "
              f"responding to the feature distributions, not the labels")
        payload["null"] = asdict(null)

    if result.si > 0.8:
        print("\nSI above 0.8 (flag): trace the named attribute to its source, "
              "remove artefact attributes and recompute SI, and report the "
              "named rule's accuracy and an external evaluation.")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())