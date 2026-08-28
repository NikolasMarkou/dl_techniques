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
