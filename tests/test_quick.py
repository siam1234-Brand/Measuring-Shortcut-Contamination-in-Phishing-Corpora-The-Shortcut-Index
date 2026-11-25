"""Behaviour tests for shortcut_index.py.  Run: python tests/test_quick.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from shortcut_index import shortcut_index

rng = np.random.default_rng(0); n = 1500
y = rng.integers(0, 2, n)
planted = pd.DataFrame({
    'urgency_score': np.where(y == 1, rng.integers(7, 11, n),
                              rng.integers(1, 6, n)),
    'a': rng.normal(size=n), 'b': rng.normal(size=n),
    'c': rng.normal(50, 10, n)})
noise = pd.DataFrame({f'f{i}': rng.normal(size=n) for i in range(4)})
weak = pd.DataFrame({'f0': rng.normal(size=n) + 0.45 * y,
                     'f1': rng.normal(size=n), 'f2': rng.normal(size=n)})

checks = [
    ('planted shortcut scores high', shortcut_index(planted, y).si > 0.9),
    ('pure noise scores zero',      shortcut_index(noise, y).si < 0.05),
    ('weak signal scores low',      shortcut_index(weak, y).si < 0.25),
]
for X, label in ((planted, 'planted'), (noise, 'noise'), (weak, 'weak')):
    checks.append((f'null test collapses ({label})',
                   shortcut_index(X, y, shuffle_labels=True).si < 0.06))
r = shortcut_index(planted, y)
checks.append(('components bounded',
               all(0 <= v <= 1 for v in (r.si, r.c1, r.c2, r.c3))))
checks.append(('SI is the mean of its components',
               abs(r.si - (r.c1 + r.c2 + r.c3) / 3) < 1e-9))

# c3 must not depend on the rule miner: mlxtend's Apriori and the built-in
# bounded search give the same c3 value (tied rules may be named differently).
from shortcut_index import component_c3
region = pd.DataFrame({'g0': rng.normal(size=n), 'g1': rng.normal(size=n),
                       'g2': rng.normal(size=n)})
datasets = (('planted', planted), ('weak', weak), ('region', region))
try:
    import mlxtend.frequent_patterns  # noqa: F401
    with_mlx = {k: component_c3(X, y)[0] for k, X in datasets}
    saved = sys.modules.get('mlxtend.frequent_patterns')
    sys.modules['mlxtend.frequent_patterns'] = None   # forces the fallback
    try:
        without = {k: component_c3(X, y)[0] for k, X in datasets}
    finally:
        sys.modules['mlxtend.frequent_patterns'] = saved
    checks.append(('c3 equals the mlxtend value',
                   all(abs(with_mlx[k] - without[k]) < 1e-12 for k, _ in datasets)))
except ImportError:
    print('  skip  c3 equals the mlxtend value (mlxtend not installed)')

for name, ok in checks:
    print(('  ok   ' if ok else '  FAIL '), name)
passed = sum(ok for _, ok in checks)
print(f"\n{passed}/{len(checks)} passed")
sys.exit(0 if passed == len(checks) else 1)