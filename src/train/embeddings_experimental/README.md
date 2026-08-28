# train/embeddings_experimental

Two-stage trainer and sweep harness for the `embeddings_experimental` model
family.

```
config.py             MODEL_REGISTRY + ExperimentConfig (the study axes)
data.py               packed, padding-free ASCII datasets
train_embeddings.py   stage 1 (MLM) + stage 2 (SimCSE); one cell per invocation
sweep.py              the grid driver, one subprocess per cell
report.py             aggregation -> summary.md + CSVs
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

**Six seeds is a floor, not a taste.** The report's paired test is a two-sided
sign-flip permutation test, so with `n` pairs the smallest reachable p-value is
about `2/2**n`. Measured against maximally separated arms: n=3 gives p=0.248,
n=5 gives p=0.063, n=6 gives p=0.031. **At five seeds or fewer no effect size
however large can be reported significant** — such comparisons are labelled
`UNDERPOWERED` rather than `INDISTINGUISHABLE`, because "no significant
difference" from an underpowered test reads like a finding and is not one.

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
