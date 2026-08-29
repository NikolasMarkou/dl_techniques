# train/embeddings_experimental

Two-stage trainer and sweep harness for the `embeddings_experimental` model
family.

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

**Recall@k** — the fraction of questions whose correct paragraph appeared in the
top `k`. Recall@1 is "how often was the right answer ranked first"; recall@10 is
"how often was it in the top ten". Higher is better, range 0-1. It is the most
directly interpretable metric here, and it is blind to *how far* down a miss
fell: rank 11 and rank 2000 are both misses at k=10. **Anchor:** chance is
1/2067 = **0.048%**; the smoke run gave 1.15% (baseline) to 4.25% (best), so
24-88x chance.

**MRR@10** (mean reciprocal rank) — average of `1/rank`, counting 0 for anything
below rank 10. A first place contributes 1.0, second 0.5, tenth 0.1. Higher is
better, range 0-1. Unlike recall it is sensitive to *where* in the top ten the
answer sat, so it distinguishes two systems that both "found it in the top ten".
It is the study's **primary** endpoint because it is the standard single-number
retrieval score and it uses more of the ranking than recall@1 alone.
**Anchor:** 0.0246 (baseline) to 0.0647 (best).

**nDCG@10** — implemented but deliberately **not** reported. With exactly one
correct document it reduces to the mean of `1/log2(rank+1)`, which is another
weighting of the same information. All three of recall@k, MRR@k and nDCG@k are
positive-weighted sums of one underlying "recall curve", so reporting them all
would present one measurement as three. It exists in the metrics module for
future graded-relevance work.

**median_rank / mean_rank** — the middle and average position of the correct
paragraph, out of 2,067. Lower is better. These are the only ranking numbers
that are *not* a function of the truncated top-k curve, so they see the long
tail that the others discard, and median is robust to a handful of catastrophic
misses that would drag the mean. Reported as a diagnostic. **Anchor:** 457
(baseline) to 274 (best) — i.e. even the best arm typically ranks the right
paragraph around 274th, which is exactly why the absolute retrieval numbers are
still near the floor at this budget.

**chance_recall_at_1** — `1 / pool_size`, printed beside the result so a low
number can be told apart from a *chance-level* number. This distinction is the
single most likely misreading of a short run.

### Classification metric — SST-2

**sst2_probe_accuracy** — freeze the encoder, embed 8,000 training sentences and
all 872 validation sentences, fit a logistic-regression classifier on those
fixed vectors, and report validation accuracy. Higher is better. Because the
encoder never receives a gradient, this measures **how linearly separable the
embedding space already is** for sentiment, not how well the model can be
fine-tuned. **Anchor:** the majority class is **50.92%**, so that is chance; the
smoke run gave 58.9% to 64.0%.

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

**effective_rank** — how many dimensions the embedding space *actually* uses,
computed as the exponential of the entropy of the singular-value spectrum. A
model with 128 output dimensions that only ever varies along 5 of them has an
effective rank near 5. This catches **dimensional collapse**, which anisotropy
can miss: a space can have low average pairwise cosine while still living in a
thin subspace. Range 1 to the embedding width, higher generally healthier.
**Anchor:** 35.5 (baseline) to 78.6 (clifford), out of a 128-wide embedding.

**alignment** — the average squared distance between a question and its own gold
paragraph, after normalizing both to unit length. **Lower is better**: it asks
"does the model put a query near its answer". Range 0 to 4. Read it only
together with uniformity, because a model that maps *everything* to one point
achieves perfect alignment of 0.0 and is useless. **Anchor:** 0.872 (clifford)
to 1.446 (convnext).

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

**embedding_norm_stats** — mean, standard deviation, min and max of the
embedding vector lengths, plus `cos_to_centroid_mean` (how aligned the average
embedding is with the overall centre). Purely descriptive. A large `norm_std`
tells you cosine similarity and dot-product would rank differently, and a
`cos_to_centroid_mean` near 1 is the "one dominant direction" degeneracy in its
sharpest form.

**\*_pad_fraction** — the proportion of each evaluated batch that was padding
rather than real characters. Not a quality metric: it is the *honesty* check on
the comparison. One arm's block cannot honour a padding mask, so if a corpus
were mostly padding the evaluation would partly be measuring the padding policy.
Batches are sorted by length and padded to the batch maximum to keep this small.
**Anchor:** 0.005 (contexts), 0.027 (questions), 0.068 (SST-2) — low enough that
the confound is not driving the results.

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

**contrastive_val_loss** — the SimCSE objective on held-out text: encode the
same sentence twice with different dropout, and check the model can match the
two copies to each other among the other sentences in the batch. Lower is
better. Treat it with suspicion as a quality measure: it falls whenever a batch
becomes easier to discriminate, which a model can achieve while producing a
degenerate space, and it is measured in the *projection* space that is discarded
before evaluation. It is here as an optimisation diagnostic, which is precisely
why the SQuAD and SST-2 metrics exist.

### Statistical fields in the report

**p_value / p_adjusted** — the raw and multiple-comparison-corrected probability
of seeing a difference this large if the two arms were really equivalent.
Smaller means stronger evidence; below 0.05 is the conventional threshold.
`p_adjusted` is always ≥ `p_value` because testing many things raises the chance
that one looks impressive by luck.

**verdict** — `BETTER` / `WORSE` only for the primary endpoint and only when the
corrected p-value clears the bar; `SECONDARY` for supporting metrics, which get
a corrected p but never a verdict; `INDISTINGUISHABLE` when the test ran and
found nothing; and `UNDERPOWERED` when there were too few seeds for the test to
have rejected *whatever* the data showed. The last one is not a weaker version
of "no difference" — it means the question was not askable with that many seeds.

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
