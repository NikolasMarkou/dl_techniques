# train/embeddings_experimental

Two-stage trainer and sweep harness for the `embeddings_experimental` model
family.

**Where things live.** Three documents, split by what question they answer:

| document | answers |
|---|---|
| this file | how the study is run, and **what every metric means** |
| [`RESULTS.md`](RESULTS.md) | **what a run actually measured** |
| [`models/embeddings_experimental/README.md`](../../dl_techniques/models/embeddings_experimental/README.md) | what the **architectures** are |

```
config.py                MODEL_REGISTRY + ExperimentConfig (the study axes)
paths.py                 single-producer run-directory paths
data.py                  packed, padding-free ASCII datasets
train_embeddings.py      stage 1 (MLM) + stage 2 (SimCSE); one cell per invocation
evaluate_embeddings.py   SQuAD retrieval + SST-2 probe -> eval.json
metric_directions.py     the ONE producer of "is higher better?"
sweep.py                 the grid driver, one subprocess per cell
report.py                aggregation -> summary.md + CSVs
```

## One setting you should not change casually

`ExperimentConfig.position_embedding_type` defaults to **`'sinusoidal'`**, not
to the encoder's own `'learned'`. It is a config field rather than an inherited
default precisely so that it lands in every run's `config.json`.

A learned table is initialized at `initializer_range=0.02`, which puts it at
essentially the word table's norm (0.1985 against 0.1987) — and training then
*abandons* it, shrinking it to 0.1612 over 3000 steps while the word table grows
to 0.3283. The transformer arm could not bootstrap position-dependent attention
from that and converged to a **bag of characters**: reordering its entire
context moved the output by 0.83% of activation scale while replacing the
context moved 52.58%. Switching this one field bought **0.85 nats** at 256
context.

**It is not a free win for every arm.** Measured at 256 context, 3000 steps:
`ascii_bert` −0.85 nats, `ascii_clifford_bert` **+0.14** (worse). The
convolutional blocks already have positional structure, and a sinusoidal signal
large enough to dominate LayerNorm compresses their token signal. The study
holds one setting across all arms deliberately — the design compares *blocks*
under one encoder — so this is a cost the comparison pays knowingly, and it is
reported rather than tuned away.

`tests/test_train/test_embeddings_experimental/test_the_positional_signal_survives_the_embedding_sum.py`
pins both directions. Read its docstring before moving the threshold: the
obvious oracle (reorder-vs-replace) is decisive on a trained model and **blind
at initialization**, which is why the guard measures signal magnitude instead.

## What is measured

MLM loss and contrastive loss are *optimisation* diagnostics: contrastive loss
falls when a batch gets easier to discriminate, which a model can achieve while
producing a degenerate space. So the study also evaluates the embeddings
directly, on the only labelled text available offline:

| task | metric | chance |
|---|---|---|
| SQuAD v1.1 retrieval (**primary**) | MRR@10, recall@1/@10 over 2,067 unique contexts | recall@1 = 0.048% |
| SST-2 linear probe (secondary) | accuracy, frozen encoder + logistic probe | 50.92% majority |
| geometry (diagnostic) | anisotropy, effective rank, alignment, uniformity | — |

**STS is not evaluated and that is a data fact.** `glue` on this machine is
`glue/sst2` and nothing else; there is no STS-B, MRPC or QQP in TFDS or in the
raw download cache, and no sentence-pair-plus-float pipeline in the repo.

Diagnostics carry **no verdict and no p-value**: a random projection maximizes
effective rank, minimizes anisotropy and minimizes uniformity while retrieving
nothing. They explain *why* a primary number moved.

## Metric reference

Every metric below, what it means, and how to read it. The anchor numbers are
from a real 4-arm smoke run (tiny/mean, 3000 MLM + 1000 contrastive steps, one
seed) so each figure has a scale attached rather than being abstract.

### Retrieval metrics — SQuAD v1.1

The task: each of 10,570 questions is a query, and its gold paragraph is the one
correct answer among **2,067 unique paragraphs**. We embed every paragraph and
every question, rank the paragraphs by cosine similarity to each question, and
ask where the correct one landed. All of these read from that single rank.

```python
import numpy as np
# 2 questions, 3 candidate paragraphs. Higher score = more similar.
sim  = np.array([[0.9, 0.1, 0.5],      # question 0
                 [0.2, 0.8, 0.3]])     # question 1
gold = np.array([2, 1])                # the correct paragraph for each

g      = sim[np.arange(len(gold)), gold][:, None]   # the gold score
better = np.sum(sim >  g, axis=1)                   # how many beat it
tied   = np.sum(sim == g, axis=1) - 1               # ties count AGAINST it
ranks  = 1 + better + tied
print(ranks)   # [2 1]  -> q0's answer came 2nd, q1's came 1st
```

Ties count against the gold answer on purpose. A collapsed model gives every
pair the same score, and under a generous tie rule that would look perfect:

```python
sim, gold = np.ones((3, 3)), np.arange(3)          # every score identical
g = sim[np.arange(3), gold][:, None]
ranks = 1 + np.sum(sim > g, 1) + (np.sum(sim == g, 1) - 1)
print(ranks, np.mean(ranks <= 1))   # [3 3 3] 0.0  -> scored last, not first
```

**Recall@k** — the fraction of questions whose correct paragraph appeared in the
top `k`. Recall@1 is "how often was the right answer ranked first"; recall@10 is
"how often was it in the top ten". Higher is better, range 0-1. It is the most
directly interpretable metric here, and it is blind to *how far* down a miss
fell: rank 11 and rank 2000 are both misses at k=10. **Anchor:** chance is
1/2067 = **0.048%**; the smoke run gave 1.15% (baseline) to 4.25% (best), so
24-88x chance.

```python
ranks = np.array([1, 3, 12, 40])       # where the answer landed, 4 questions
for k in (1, 5, 10):
    print(k, np.mean(ranks <= k))
# 1  0.25   one of four was ranked first
# 5  0.50   two were in the top five
# 10 0.50   the 12th and 40th are still misses
```

**MRR@10** (mean reciprocal rank) — average of `1/rank`, counting 0 for anything
below rank 10. A first place contributes 1.0, second 0.5, tenth 0.1. Higher is
better, range 0-1. Unlike recall it is sensitive to *where* in the top ten the
answer sat, so it distinguishes two systems that both "found it in the top ten".
It is the study's **primary** endpoint because it is the standard single-number
retrieval score and it uses more of the ranking than recall@1 alone.
**Anchor:** 0.0246 (baseline) to 0.0647 (best).

```python
ranks, k = np.array([1, 3, 12, 40]), 10
rr = np.where(ranks <= k, 1.0 / ranks, 0.0)
print(rr, rr.mean())
# [1.  0.333 0.  0.]  0.3333
# rank 1 is worth 1.0, rank 3 only 0.333 -- unlike recall@10, which
# scores rank 1 and rank 3 identically.
```

**nDCG@10** — implemented but deliberately **not** reported. With exactly one
correct document it reduces to the mean of `1/log2(rank+1)`, which is another
weighting of the same information. All three of recall@k, MRR@k and nDCG@k are
positive-weighted sums of one underlying "recall curve", so reporting them all
would present one measurement as three. It exists in the metrics module for
future graded-relevance work.

```python
gain = np.where(ranks <= k, 1 / np.log2(ranks + 1), 0.0)
print(gain, gain.mean())   # [1.  0.5  0.  0.] 0.375

# Per query nDCG and MRR always agree on which rank is better, but their
# MEANS can disagree, because they discount deep ranks differently:
for name, r in (("A", np.array([1, 100])), ("B", np.array([2, 2]))):
    mrr  = np.mean(np.where(r <= 100, 1 / r, 0))
    ndcg = np.mean(np.where(r <= 100, 1 / np.log2(r + 1), 0))
    print(name, round(mrr, 4), round(ndcg, 4))
# A 0.505  0.5751     <- MRR prefers A
# B 0.5    0.6309     <- nDCG prefers B
```

**median_rank / mean_rank** — the middle and average position of the correct
paragraph, out of 2,067. Lower is better. These are the only ranking numbers
that are *not* a function of the truncated top-k curve, so they see the long
tail that the others discard, and median is robust to a handful of catastrophic
misses that would drag the mean. Reported as a diagnostic. **Anchor:** 457
(baseline) to 274 (best) — i.e. even the best arm typically ranks the right
paragraph around 274th, which is exactly why the absolute retrieval numbers are
still near the floor at this budget.

```python
ranks = np.array([1, 3, 12, 40])
print(np.median(ranks), np.mean(ranks))   # 7.5  14.0
# The single rank-40 miss drags the mean to 14 but moves the median only to
# 7.5 -- which is why the median is the more stable summary.
```

**chance_recall_at_1** — `1 / pool_size`, printed beside the result so a low
number can be told apart from a *chance-level* number. This distinction is the
single most likely misreading of a short run.

```python
pool = 2067
print(1 / pool, 10 / pool)   # 0.000484  0.004838
# So recall@1 of 1.15% is ~24x chance, not "almost zero".
```

### Classification metric — SST-2

**sst2_probe_accuracy** — freeze the encoder, embed 8,000 training sentences and
all 872 validation sentences, fit a logistic-regression classifier on those
fixed vectors, and report validation accuracy. Higher is better. Because the
encoder never receives a gradient, this measures **how linearly separable the
embedding space already is** for sentiment, not how well the model can be
fine-tuned. **Anchor:** the majority class is **50.92%**, so that is chance; the
smoke run gave 58.9% to 64.0%.

```python
from sklearn.linear_model import LogisticRegression
rng = np.random.default_rng(0)
# Pretend embeddings where the two classes are slightly offset.
ytr = rng.integers(0, 2, 400); Xtr = rng.standard_normal((400, 16)) + ytr[:, None] * 0.8
yte = rng.integers(0, 2, 200); Xte = rng.standard_normal((200, 16)) + yte[:, None] * 0.8

clf = LogisticRegression(max_iter=2000).fit(Xtr, ytr)   # encoder is NOT trained
print(clf.score(Xte, yte), max(np.bincount(yte)) / len(yte))
# 0.88  0.505   -> 88% against a 50.5% majority: the space is separable
```

**sst2_probe_cv_accuracy / sst2_probe_best_c** — the classifier's regularization
strength `C` is chosen by 5-fold cross-validation on the *training* subsample
only, and both the chosen value and its CV score are recorded. This matters
because the arms are depth- and width-matched but **not** parameter-matched, so
a fixed `C` would quietly favour whichever hidden width happened to suit it. A
large gap between CV accuracy and validation accuracy means the probe overfit,
not that the encoder failed.

### Geometry diagnostics

These need no labels. They describe the *shape* of the embedding cloud and
answer "why did retrieval move", never "which arm is better" — a random
projection scores well on all of them while retrieving nothing. That is why they
carry **no verdict and no p-value**.

**anisotropy** — the average cosine similarity between two randomly chosen
embeddings. If the space is healthy this is near 0: unrelated texts point in
unrelated directions. As it approaches 1 every embedding points the same way,
the classic "cone collapse" of an undertrained encoder, and cosine similarity
stops discriminating because everything is similar to everything. Range -1 to 1,
lower generally healthier. **Anchor:** 0.132 (convnext_v2, well spread) to 0.616
(clifford, quite concentrated); the same baseline model measured 0.856 after
only 5 steps and 0.429 after 3000, so it falls as training proceeds.

```python
def anisotropy(x):
    u = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, total = len(u), u.sum(0)
    return float((total @ total - n) / (n * (n - 1)))   # mean pairwise cosine

rng = np.random.default_rng(0)
print(anisotropy(rng.standard_normal((200, 64))))              # -0.0005  healthy
print(anisotropy(np.tile(rng.standard_normal(64), (200, 1))))  #  1.0     collapsed
```

**effective_rank** — how many dimensions the embedding space *actually* uses,
computed as the exponential of the entropy of the singular-value spectrum. A
model with 128 output dimensions that only ever varies along 5 of them has an
effective rank near 5. This catches **dimensional collapse**, which anisotropy
can miss: a space can have low average pairwise cosine while still living in a
thin subspace. Range 1 to the embedding width, higher generally healthier.
**Anchor:** 35.5 (baseline) to 78.6 (clifford), out of a 128-wide embedding.

```python
def effective_rank(x):
    s = np.linalg.svd(x - x.mean(0), compute_uv=False)
    p = s / s.sum(); p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))            # exp(entropy)

rng = np.random.default_rng(0)
print(effective_rank(rng.standard_normal((200, 8))))        # 7.96  uses all 8 dims
low = rng.standard_normal((200, 2)) @ rng.standard_normal((2, 8))
print(effective_rank(low))                                  # 1.99  8 dims, 2 used
```

**alignment** — the average squared distance between a question and its own gold
paragraph, after normalizing both to unit length. **Lower is better**: it asks
"does the model put a query near its answer". Range 0 to 4. Read it only
together with uniformity, because a model that maps *everything* to one point
achieves perfect alignment of 0.0 and is useless. **Anchor:** 0.872 (clifford)
to 1.446 (convnext).

```python
def unit(x): return x / np.linalg.norm(x, axis=1, keepdims=True)

rng = np.random.default_rng(0)
q    = rng.standard_normal((100, 16))
near = q + rng.standard_normal((100, 16)) * 0.1    # answers close to questions
far  = rng.standard_normal((100, 16))              # unrelated answers
for name, pos in (("near", near), ("far", far)):
    print(name, round(float(np.mean(np.sum((unit(q) - unit(pos)) ** 2, 1))), 4))
# near 0.0114     far 2.1144      lower = query sits nearer its answer
```

**uniformity** — how evenly the embeddings spread over the unit sphere,
`log(mean(exp(-2 * squared distance)))` over all pairs. **Lower (more negative)
is better**: it rewards using the whole space. A fully collapsed model scores
0.0, the worst possible value. Together with alignment it forms the tension that
contrastive learning trades off — pull matching pairs together (alignment) while
pushing everything else apart (uniformity). **Anchor, and it is a good
illustration:** clifford has the *best* alignment (0.872) and the *worst*
uniformity (-1.420), while convnext_v2 has the worst alignment (1.445) and the
best uniformity (-3.161). Neither is simply "better"; they sit at different
points on that trade-off, which is also why clifford's anisotropy is highest.

```python
def uniformity(x, t=2.0):
    u = x / np.linalg.norm(x, axis=1, keepdims=True)
    d = ((u[:, None] - u[None, :]) ** 2).sum(-1)
    iu = np.triu_indices(len(u), 1)
    return float(np.log(np.exp(-t * d[iu]).mean()))

rng = np.random.default_rng(0)
print(uniformity(rng.standard_normal((200, 16))))                        # -3.51 spread
blob = np.tile(rng.standard_normal(16), (200, 1)) + rng.standard_normal((200, 16)) * 1e-3
print(uniformity(blob))                                                  # -0.0  collapsed
```

**embedding_norm_stats** — mean, standard deviation, min and max of the
embedding vector lengths, plus `cos_to_centroid_mean` (how aligned the average
embedding is with the overall centre). Purely descriptive. A large `norm_std`
tells you cosine similarity and dot-product would rank differently, and a
`cos_to_centroid_mean` near 1 is the "one dominant direction" degeneracy in its
sharpest form.

```python
rng = np.random.default_rng(0)
x = rng.standard_normal((100, 16)) * rng.uniform(0.5, 3, (100, 1))   # varied lengths
n = np.linalg.norm(x, axis=1)
print(round(float(n.mean()), 3), round(float(n.std()), 3))   # 6.77  3.217
# A large spread means cosine and dot-product would rank differently,
# because dot-product rewards simply being a longer vector.
```

**\*_pad_fraction** — the proportion of each evaluated batch that was padding
rather than real characters. Not a quality metric: it is the *honesty* check on
the comparison. One arm's block cannot honour a padding mask, so if a corpus
were mostly padding the evaluation would partly be measuring the padding policy.
Batches are sorted by length and padded to the batch maximum to keep this small.
**Anchor:** 0.005 (contexts), 0.027 (questions), 0.068 (SST-2) — low enough that
the confound is not driving the results.

```python
lengths = np.array([5, 7, 200])            # three texts in one batch
width   = lengths.max()                    # everything padded to the longest
print(1 - lengths.sum() / (len(lengths) * width))   # 0.6467 -> 65% padding

# Sorting by length first keeps similar texts together:
short = np.array([5, 7])
print(1 - short.sum() / (len(short) * short.max()))  # 0.1429 -> 14% padding
```

### Training metrics

**mlm_val_loss / mlm_val_accuracy** — masked-language-modelling loss and
accuracy: hide some characters, predict them, measure cross-entropy in nats and
top-1 accuracy. Lower loss and higher accuracy are better. These measure how
well the model predicts text, which is related to but *not the same as* how good
its embeddings are. **The anchor is what makes the loss readable:** a uniform
guess over the 101-id vocabulary costs `ln(101)` = **4.615 nats**, and simply
knowing the character frequencies of this corpus — with no context at all —
costs **3.130 nats**. So a model at 2.84 has learned little beyond letter
frequencies, while one at 1.15 has learned a great deal of context. In the smoke
run the transformer baseline sat at 2.838, only 0.29 nats below the
context-free baseline, and the convolutional arms at ~1.15-1.32.

```python
V = 101
print(np.log(V))          # 4.615 nats -- guessing uniformly at random

# Knowing only how often each character occurs, with no context at all:
p = np.array([0.18, 0.10, 0.08, 0.07, 0.07] + [0.5 / 96] * 96); p /= p.sum()
print(-(p * np.log(p)).sum())   # 3.742 nats for these toy frequencies
# Measured on the real corpus (1.03M characters): 3.130 nats.
# So a model at 2.838 has learned little more than letter frequencies;
# one at 1.153 has learned a great deal of context.
```

**contrastive_val_loss** — the SimCSE objective on held-out text: encode the
same sentence twice with different dropout, and check the model can match the
two copies to each other among the other sentences in the batch. Lower is
better. Treat it with suspicion as a quality measure: it falls whenever a batch
becomes easier to discriminate, which a model can achieve while producing a
degenerate space, and it is measured in the *projection* space that is discarded
before evaluation. It is here as an optimisation diagnostic, which is precisely
why the SQuAD and SST-2 metrics exist.

```python
def infonce(noise, n=8, temp=0.05, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, 8)); a /= np.linalg.norm(a, axis=1, keepdims=True)
    b = a + rng.standard_normal((n, 8)) * noise
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    logits = (a @ b.T) / temp                       # every view against every other
    correct = logits[np.arange(n), np.arange(n)]    # the matching pair
    return float(np.mean(-correct + np.log(np.exp(logits).sum(1))))

print(infonce(0.1))     # 0.0431  the two views stay close -> easy
print(infonce(3.0))     # 12.86   the views drift apart   -> hard
print(np.log(8))        # 2.079   chance for a batch of 8
# Note it can exceed chance: at a sharp temperature a confidently WRONG
# match is punished far more than a random guess.
```

### Statistical fields in the report

**p_value / p_adjusted** — the raw and multiple-comparison-corrected probability
of seeing a difference this large if the two arms were really equivalent.
Smaller means stronger evidence; below 0.05 is the conventional threshold.
`p_adjusted` is always ≥ `p_value` because testing many things raises the chance
that one looks impressive by luck.

```python
p = np.array([0.001, 0.02, 0.03, 0.5]); m = len(p)
order = np.argsort(p)
adj = np.minimum.accumulate((p[order] * m / np.arange(1, m + 1))[::-1])[::-1]
out = np.empty(m); out[order] = np.minimum(adj, 1)
print(out)   # [0.004 0.04  0.04  0.5]
# Each p is scaled by (number of tests / its rank), so testing four things
# makes a raw 0.02 into an adjusted 0.04 -- still significant here, but
# a raw 0.03 among 18 tests would not be.
```

**verdict** — `BETTER` / `WORSE` only for the primary endpoint and only when the
corrected p-value clears the bar; `SECONDARY` for supporting metrics, which get
a corrected p but never a verdict; `INDISTINGUISHABLE` when the test ran and
found nothing; and `UNDERPOWERED` when there were too few seeds for the test to
have rejected *whatever* the data showed. The last one is not a weaker version
of "no difference" — it means the question was not askable with that many seeds.

```python
import math
for m in (1, 3, 18):
    print(m, math.ceil(1 + math.log2(m / 0.05)))
# 1  6      uncorrected, 6 seeds is the minimum
# 3  7      the primary family (3 arms) needs 7
# 18 10     the secondary family needs 10
# Below these, the test cannot return a significant result for ANY effect
# size -- which is what UNDERPOWERED reports.
```

## Running

```bash
# one cell
CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg .venv/bin/python -m train.embeddings_experimental.train_embeddings \
    --model ascii_bert --variant small --pooling-strategy mean --gpu 0

# the study (defaults: both arms x 3 pooling strategies x 6 seeds)
CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg .venv/bin/python -m train.embeddings_experimental.sweep \
    --variants tiny --gpu 0

# the report
.venv/bin/python -m train.embeddings_experimental.report --in-dir results/embeddings_study
```

`--dry-run` prints the grid and exits, which is the cheap way to check a cell
budget before spending GPU hours.

## Three things worth knowing before changing anything here

**Stage 1 is packed, not padded.** Every row is exactly `max_seq_length` real
characters and the attention mask is all ones. This is not a throughput
optimization: the Clifford arm's block has `supports_masking = False`, so a
padded batch would make a stage-1 comparison measure the padding policy as much
as the block. `data.py` carries the measurement.

**Seven seeds is a floor, not a taste.** The report's paired test is a two-sided
sign-flip permutation test, so with `n` pairs the smallest reachable p-value is
about `2/2**n`: n=3 gives 0.248, n=5 gives 0.063, n=6 gives 0.031. The primary
endpoint is then Holm-corrected across the non-baseline arms, which tightens the
bar to `alpha/m`, so the requirement is `n >= 1 + log2(m/alpha)`. Verified
against the real test: m=1 needs **6** seeds, m=3 needs **7**, m=18 needs
**10**, m=63 needs **12**.

**At five seeds or fewer — six, once corrected — no effect size however large
can be reported significant.** Such comparisons are labelled `UNDERPOWERED`
rather than `INDISTINGUISHABLE`, because "no significant difference" from an
underpowered test reads like a finding and is not one. Choosing the correction
family is therefore choosing the study's GPU budget, which is why the families
are stated in `summary.md` rather than buried.

**Evaluation caveats**, all carried into `summary.md`:

- SQuAD contexts *are* Wikipedia paragraphs and the MLM corpus is Wikipedia.
  Every arm shares the leak so a relative comparison survives it; no absolute
  claim does.
- Contexts are mean 780 / median 705 / p90 1166 characters against a window of a
  few hundred, so retrieval matches a context **prefix**, not a passage.
- Evaluation cannot pack, so padding returns — on the arm whose block cannot
  mask it. `embed_texts` sorts by length and pads to the batch maximum, and each
  corpus reports its pad fraction. Measured on a real encoder at `max_length=64`:
  contexts 0.000, questions 0.061, SST-2 0.026.
- Stage 2 saves the encoder, not the SimCSE wrapper, so the metrics use
  `pooled_output`; the contrastive loss lives in projection space and the two can
  move in opposite directions.

**Neither stage uses a custom `train_step`.** Stage 1 delegates to the existing
`MaskedLanguageModel` wrapper; stage 2's forward pass returns both dropout views
stacked so the contrastive loss is an ordinary `compile(loss=...)` function.
`SimCSEModel.call` keeps dropout ON regardless of the training flag, because in
SimCSE the positive pair *is* the dropout noise — with it off the two views are
identical and the validation number would look excellent and mean nothing. Use
`SimCSEModel.embed` for a deterministic embedding.

## Adding an arm

Add one entry to `MODEL_REGISTRY` in `config.py`. The trainer, the sweep and the
report pick it up with no further edits; `--model` is generated from the
registry, so the CLI and the study axes cannot drift from it.

## Known deviations

**A benign TF message on the ConvNeXt arm.** Grappler's layout optimizer logs
`layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of
permutation 4 ... TransposeNHWCToNCHW-LayoutOptimizer` from the dropout inside
`convnext_block`. It is an optimization pass declining to rewrite a tensor whose
height axis is the singleton introduced by the sequence lift; the pass is
skipped, training proceeds, and losses are finite. Not an error.

**Stage 2 compiles with `jit_compile=False`.** The SimCSE step fails to compile
under XLA on this TF 2.18 build (`FAILED_PRECONDITION: Can not combine dim
orders and requirements`). Scoped to that stage — stage 1 compiles fine.
