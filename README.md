# The Shortcut Index: Screening Phishing Corpora for Shortcut Contamination

Code and results for the paper

> Md. Mahabubur Rahaman Siam, Soumik Das Dipon, and Sayma Rahman Shanta.
> **The Shortcut Index: Screening Phishing Corpora for Shortcut Contamination.**
> 2026 29th International Conference on Computer and Information Technology (ICCIT),
> Cox's Bazar, Bangladesh, 18–20 December 2026. IEEE.
>
> American International University-Bangladesh (AIUB), Dhaka, Bangladesh.

The Shortcut Index (SI) is a cheap screening score computed from one labelled corpus. It measures how much of the corpus's labels simple rules recover out of sample, and it names the rule responsible:

```
SI(D) = (c1 + c2 + c3) / 3   in [0, 1]

c1  best single-attribute threshold rule, chosen on training folds, scored held out,
    rescaled by the majority-class rate b = max(pi, 1 - pi)
c2  |ARI| of 2-means clusters (10 restarts, standardised attributes) against the labels
c3  strongest Apriori class-association rule on equal-frequency tertiles
    (edges fitted on the training fold), held-out confidence rescaled by b
```

Settings: 5-fold stratified CV, seed 42, thresholds at the 1st–99th percentiles, Apriori support >= 0.05, confidence >= 0.70, itemsets of length <= 3, and a stratified 100,000-record subsample for c2 on larger corpora.

**How to read SI.** SI > 0.8 is a flag: inspect the rule SI names, trace the attribute to its source, remove artefact attributes and recompute, and report an external evaluation (Sec. VI of the paper). SI is a screen, not a forecast of cross-corpus accuracy loss: the pre-specified predictive test in the paper was null (Spearman rho = 0.14, p = 0.67, twelve corpora).

## Contents

| File | What it is |
|---|---|
| `shortcut_index.py` | Reference implementation of SI (Sec. III). Library and command line. |
| `planted_shortcut_benchmark.py` | Seeded planted-shortcut benchmark and baseline comparison (Sec. V-E, Table IV). |
| `results/benchmark.csv` | Per-corpus scores for the seven planted scenarios (20 seeds each). |
| `results/dose_response.csv` | SI as proxy agreement rises from 0.5 to 1.0. |
| `results/progressive_ablation.csv` | SI as the five redundant proxies are removed. |
| `results/auroc.json` | AUROC with 95% bootstrap intervals for every screen (Table IV). |
| `results/summary.json` | Scenario means, dose response, ablation and flag rates (Sec. V-E). |
| `results/table1_reported.csv` | Table I of the paper (SI, components and transfer gap per corpus), as reported. |
| `data/README.md` | Where to download every public corpus, with sizes and licences. |
| `tests/test_quick.py` | Behaviour tests for `shortcut_index.py`, including a check that c3 equals the value obtained with mlxtend. |
| `ShortcutIndex_Colab.ipynb` | Colab demo: SI on corpora with known structure, and a scorer for your own CSV. |

## Install

```bash
python -m pip install -r requirements.txt
```

Tested with Python 3.13 (versions pinned in `requirements.txt`).

## Usage

Score a corpus (CSV with a binary label column; numeric columns are scored):

```bash
python shortcut_index.py data.csv --label label
python shortcut_index.py data.csv --label label --null        # shuffled-label test
python shortcut_index.py data.csv --label label --json out.json
```

From Python:

```python
from shortcut_index import shortcut_index
r = shortcut_index(X, y)      # pandas DataFrame or array, binary labels
print(r.si, r.c1, r.c2, r.c3)
```

Regenerate Sec. V-E and Table IV (about 11 minutes on two cores):

```bash
python -W ignore planted_shortcut_benchmark.py --seeds 20 --out results --procs 2
```

Run the behaviour tests:

```bash
python tests/test_quick.py
```

## Data

All corpora are public. They are not redistributed here; see `data/README.md` for sources, sizes and licences.

## Citation

See `CITATION.cff`. A DOI will be added after publication in IEEE Xplore.

## Licence

Code: MIT (see `LICENSE`). Each dataset keeps its own licence.
