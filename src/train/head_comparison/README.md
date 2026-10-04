# Head comparison: softmax vs harmonic vs hierarchical (CIFAR-100, ViT)

Compares five output heads on a fixed ViT trunk: `softmax` (baseline),
`harmonic_logits`, `harmonic_dist` (+ HarMax), `hier_fixed`
(HierarchicalHarmonicHead, identity assignment), `hier_full` (+ periodic
reassignment + exponent annealing).

## Run one arm

```bash
MPLBACKEND=Agg .venv/bin/python -m train.head_comparison.train_heads \
    --head hier_full --variant vit_pico --epochs 50 --seed 0 \
    --output-dir results --experiment-name hier_full_seed0 --gpu 0
```

One run per (head, seed); three seeds minimum for the verdict table.
`--no-analyzer` skips the post-hoc `ModelAnalyzer` pass (spectra,
calibration, weight health); representation structure (silhouette,
explained variance, grokking dynamics) is computed by `analyze_heads.py`
from the saved `test_features.npz` either way.

## Aggregate

```bash
MPLBACKEND=Agg .venv/bin/python -m train.head_comparison.analyze_heads \
    --runs-dir results --out comparison
```

Writes `comparison.json` + `comparison.md`: per-arm means with bootstrap
CIs, paired permutation contrasts vs `softmax`, and representation
structure. Pre-registered hypotheses: H1 fine-accuracy parity; H2 hier
wins coarse accuracy + purity; H3 harmonic wins tail accuracy / ECE; H5
harmonic shows more structured representations at parity.

## Notes

- Coarse (superclass) labels load alongside the shared fine loader via
  `keras.datasets.cifar100(label_mode="coarse")`; the fine→coarse table
  is derived by majority vote with a unanimity guard, not hardcoded.
- `HierarchicalHarmonicHead` trains with `SparseCategoricalCrossentropy
  (from_logits=False)`; its `logits` mode is already-normalized
  log-probabilities and must NOT go through `from_logits=True`.
- No augmentation: trunk, data order (per seed) and budget are fixed so
  head differences are attributable. Absolute numbers trail tuned recipes.

## Measured verdicts (5 arms x 3 seeds x 50 epochs, vit_pico, GPU1-uniform)

`comparison.json` / `comparison.md` (via `analyze_heads.py`) hold the
full tables with bootstrap CIs and paired permutation contrasts. Headline
means (fine / coarse / tail-decile / ECE):

| arm | fine | coarse | tail | ECE |
|---|---|---|---|---|
| softmax | 0.455 | 0.587 | 0.218 | 0.285 |
| harmonic_logits | 0.384 | 0.540 | 0.152 | 0.082 |
| harmonic_dist | 0.384 | 0.540 | 0.157 | 0.083 |
| hier_fixed | 0.274 | 0.458 | 0.004 | 0.116 |
| hier_full | 0.233 | 0.414 | 0.001 | 0.030 |

- H1 (accuracy parity): rejected -- softmax leads by ~7pts, CIs disjoint.
  The two flat harmonic usages tie (0.3838 vs 0.3843).
- H2 (hier wins coarse): rejected -- hierarchy costs coarse accuracy here.
- H3 (tails/calibration): split -- tails go to softmax, but calibration
  goes to harmonic by 3-9x (analyzer ECE agrees directionally; Brier
  nuances it: harmonic is honest-but-diffuse, softmax sharp-but-
  overconfident).
- H5 (structure): top-2 PCA variance 0.08 (softmax) vs 0.48 (harmonic)
  vs 0.73 (hier_full); silhouettes ~0 everywhere at this accuracy;
  spectral power-law alphas sit in different regimes (~1.8 vs ~2.6).
- Permutation p-values floor at 0.25 with n=3 pairs (all contrasts
  same-signed = maximally significant); the CIs carry the inference.

## Configuration caveat (measured on GPU0, n=3 seeds each)

Three single-config probe families, each at s0/s1/s2 (s1/s2 on GPU0,
`s*_gpu0` dirs; `cfg_` prefix keeps them out of the `cmp_` verdict
table), resolve the prime suspects:

| probe | fine | coarse | tail | ECE |
|---|---|---|---|---|
| `harmonic_logits` n=13 (`cfg_n13`) | 0.4492 ± 0.0033 | 0.5864 | 0.2263 | 0.1876 |
| `hier_full` `(20,5)` n=13 (`cfg_branch20_n13`) | 0.1997 ± 0.0123 | 0.3947 | 0.0000 | 0.1953 |
| `hier_full` warm-start (`cfg_warmstart`, `--reassign-start 20`) | 0.2828 ± 0.0030 | 0.4538 | 0.0020 | 0.1327 |

- Exponent was the flat-harmonic story: n=13 closes the whole gap to
  softmax (0.4492 vs 0.4548 fine, 0.5864 vs 0.5871 coarse, 0.2263 vs
  0.2183 tail) while keeping the calibration win (ECE 0.19 vs 0.29).
  The paper's sqrt(D)~14 heuristic holds; n=4 was just wrong.
- Taxonomy-aligned branching does NOT save the hierarchy: `(20,5)` +
  n=13 scores 0.1997, worse than the `(10,10)` n=4 `hier_full`
  baseline (0.2329). Config is not the hierarchy's problem.
- LR/schedule remain shared-with-softmax (untested); reassignment
  timing is below.

## Implementation suspect (measured): reassignment churn confirmed, hierarchy still loses

Warm-starting (`--reassign-start 20`, same n, same tree) beats both
`hier_full` (0.2828 vs 0.2329, +5.0pts, all 3 seeds) and `hier_fixed`
(0.2828 vs 0.2736, +0.9pts, all 3 seeds): learning the assignment
late beats both early-churn and never-reassign. The mechanism is
churn, not concept — `reassign` from epoch 1 permutes member rows
against random-trunk class means (smoke: 93/100 classes moved at epoch
2), so prototypes chase a moving target while the trunk is still noise.
But the hierarchy still trails flat by 17pts, so warm-start is a
palliative, not a fix.

## Verdict

- Ship flat harmonic with n ~= sqrt(D) as a calibrated softmax
  drop-in: parity on fine/coarse/tail with a large ECE win
  (`cfg_n13` 0.4492/0.5864/0.2263/0.19 vs softmax
  0.4548/0.5871/0.2183/0.29).
- Do not ship the tree head on this evidence: assignment learning
  works (warm-start beats fixed and early-churn on all 3 seeds) but
  the hierarchy itself costs ~17pts of fine accuracy on CIFAR-100,
  and neither the paper's exponent nor taxonomy-aligned branching
  recovers it. Next step is a new idea (loss, trunk co-design), not
  another config.
