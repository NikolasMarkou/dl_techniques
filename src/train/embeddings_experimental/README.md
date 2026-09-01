# train/embeddings_experimental

Two-stage trainer and sweep harness for the `embeddings_experimental` model
family (`src/dl_techniques/models/embeddings_experimental/`: `ascii_bert`,
`ascii_clifford_bert`, `ascii_convnext_bert`). One cell = one
`(model, variant, pooling, seed)` combination.

Stage 1 is character-level MLM pretraining on packed Wikipedia. Stage 2 is
SimCSE contrastive fine-tuning. Then the encoder is evaluated on SQuAD
retrieval and an SST-2 linear probe.

| Document | Answers |
|---|---|
| this file | how to run it |
| [`RESULTS.md`](RESULTS.md) | what a run measured, and what every metric means |
| [`../../dl_techniques/models/embeddings_experimental/README.md`](../../dl_techniques/models/embeddings_experimental/README.md) | what the architectures are |

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

## Run

```bash
# one cell
MPLBACKEND=Agg .venv/bin/python -m train.embeddings_experimental.train_embeddings \
    --model ascii_bert --variant small --pooling-strategy mean --gpu 0

# the study as actually run: 3 arms x 7 seeds = 21 cells, ~3.7 h on one 4090
MPLBACKEND=Agg .venv/bin/python -m train.embeddings_experimental.sweep \
    --variants tiny --pooling mean --seeds 0 1 2 3 4 5 6 --gpu 0 \
    --sweep-root results/embeddings_study_512_sinusoidal \
    --trainer-arg=--max-seq-length=512 \
    --trainer-arg=--steps-per-epoch=6000 \
    --trainer-arg=--contrastive-steps-per-epoch=2000 \
    --trainer-arg=--max-train-samples=60000 \
    --trainer-arg=--position-embedding-type=sinusoidal

# the report — the sweep does NOT run it for you
.venv/bin/python -m train.embeddings_experimental.report \
    --in-dir results/embeddings_study_512_sinusoidal

# re-evaluate an existing run directory
MPLBACKEND=Agg .venv/bin/python -m train.embeddings_experimental.evaluate_embeddings \
    --run-dir results/<cell>/ --gpu 0
```

**Check the grid size before you launch.** Bare `sweep.py` defaults to every
registered arm x 4 pooling strategies x 7 seeds = 112 cells, about 20 hours.
`--dry-run` prints the grid and exits.

**A long sweep must survive its launching shell.** Under an agent harness that
reaps the process group, use `setsid nohup ... &` — a plain background job was
killed after one cell and reported exit code 0 having done nothing.

## Settings that decide the answer

Three axes move the arm comparison by more than the block differences the study
exists to measure, and they move arms in opposite directions. There is no
setting fair to every arm, so the choice is part of the result — pick
deliberately and report it.

- **Embeddings are the deliverable**: `--position-embedding-type learned`,
  `--pooling-strategy mean`.
- **The language model is the deliverable**: `--position-embedding-type
  sinusoidal`, short `--max-seq-length`.

A learned position table initializes ~40x smaller than sinusoidal and does not
grow, which left the transformer arm unable to use position at all. Details and
numbers are in `RESULTS.md`.

## CLI — `train_embeddings.py`

| Flag | Default | Notes |
|---|---|---|
| `--model` | `ascii_bert` | Generated from `MODEL_REGISTRY`; baseline is `ascii_bert`. |
| `--variant` | `tiny` | `tiny`, `small`, `base`. |
| `--pooling-strategy` | `mean` | `cls`, `mean`, `attention`, `max`. |
| `--seed` | `0` | |
| `--max-seq-length` | `256` | Stage-1 context. |
| `--max-train-samples` | `20000` | |
| `--max-val-samples` | `1000` | |
| `--min-article-length` | `0` | 0 = no filter, correct for a packed stream. |
| `--shuffle-shards` | `4` | |
| `--wikipedia-cache-dir` | none | |
| `--mlm-epochs` | `1` | |
| `--mlm-batch-size` | `32` | |
| `--mlm-learning-rate` | `5e-4` | |
| `--mlm-warmup-ratio` | `0.06` | |
| `--mlm-weight-decay` | `0.01` | |
| `--mlm-gradient-clip-norm` | `1.0` | |
| `--mask-ratio` | `0.15` | |
| `--random-token-ratio` | `0.1` | |
| `--unchanged-ratio` | `0.1` | |
| `--steps-per-epoch` | none | Stage-1 step budget. |
| `--no-contrastive` | off | Stop after MLM. |
| `--contrastive-epochs` | `1` | |
| `--contrastive-batch-size` | `64` | Also the number of in-batch InfoNCE negatives. |
| `--contrastive-learning-rate` | `1e-4` | |
| `--contrastive-temperature` | `0.05` | |
| `--contrastive-steps-per-epoch` | none | |
| `--projection-dim` | `256` | |
| `--vocab-size` | ASCII vocab | |
| `--hidden-dropout-rate` | `0.1` | |
| `--stochastic-depth-rate` | `0.0` | |
| `--contrastive-seq-length` | follows `--max-seq-length` | Set it shorter to hold the negative count fixed while the pretraining context varies. Batch 64 at 1024 does not fit on a 24 GB card, and halving the batch makes the contrastive task easier in the same direction as any improvement a longer context should show. |
| `--position-embedding-type` | `sinusoidal` | `learned` or `sinusoidal`. |
| `--no-embedding-eval` | off | Skips SQuAD/SST-2. That evaluation is what makes this a study of embedding quality rather than of optimisation. |
| `--tfds-data-dir` | `/media/arxwn/data0_4tb/datasets/tensorflow_datasets` | |
| `--eval-max-queries` | `2000` | |
| `--eval-probe-train-n` | `8000` | |
| `--eval-batch-size` | `64` | |
| `--output-dir` | `results` | A relative path resolves against the **repo root**, not the cwd. |
| `--experiment-name` | none | |
| `--gpu` | none | |
| `--mixed-bfloat16` | off | |

## CLI — `sweep.py`

| Flag | Default | Notes |
|---|---|---|
| `--models` | every registered arm | |
| `--variants` | `tiny` | |
| `--pooling` | all four strategies | |
| `--seeds` | `0 1 2 3 4 5 6` | Seven is a floor, not a taste — see below. |
| `--sweep-root` | `results/embeddings_study` | Relative paths resolve against the repo root. |
| `--gpu` | `0` | |
| `--max-cells` | `200` | Safety cap on grid size. |
| `--cell-timeout-s` | `7200.0` | Per-cell wall clock. |
| `--global-timeout-s` | none | |
| `--dry-run` | off | Print the grid and exit. |
| `--python-exe` | `sys.executable` | |
| `--trainer-arg` | none | Extra trainer flag applied to every cell; repeatable. Use `--trainer-arg=--flag=value`. |

## CLI — `report.py` and `evaluate_embeddings.py`

`report.py`: `--in-dir` (default `results/embeddings_study`), `--out-dir`
(default: `--in-dir`), `--models` (default: every arm with cells under
the root — for an old sweep root that can include a withdrawn arm, which changes
the Holm family size and therefore every adjusted p-value; a warning names any
such arm).

`evaluate_embeddings.py`: `--run-dir` (required), `--tfds-data-dir`,
`--max-length`, `--max-queries`, `--probe-train-n`, `--batch-size`, `--seed`,
`--no-squad`, `--no-sst2`, `--gpu`.

## What lands on disk

Per cell, under the run directory:

```
encoder.keras     the encoder handed from stage 1 to stage 2 and to evaluation
eval.json         SQuAD retrieval + SST-2 probe + geometry diagnostics
results.json      what the sweep collects
cell.log          only written when the cell subprocess EXITS
```

`report.py` writes `summary.md` plus `headline_summary.csv` and
`paired_summary.csv` into `--out-dir`.

A running cell is invisible because `cell.log` is flushed only at exit. Watch
the sweep log, or count `results.json` files.

## Gotchas

- **Seven seeds is a floor.** The report's paired test is a two-sided sign-flip
  permutation test, so with `n` pairs the smallest reachable p is about
  `2/2**n`; Holm correction across `m` non-baseline arms tightens it to
  `alpha/m`, giving `n >= 1 + log2(m/alpha)`. Verified: m=1 needs 6, m=2 needs
  7, m=12 needs 9, m=18 needs 10. At 5 seeds or fewer no effect size however
  large can be reported significant; those comparisons are labelled
  `UNDERPOWERED`, not `INDISTINGUISHABLE`. As run, the study meets the primary
  floor (m=2) and misses the secondary one (m=12), so every secondary
  comparison — `mlm_val_loss` included — is underpowered.
- **Stage 1 is packed, not padded.** Every row is exactly `max_seq_length` real
  characters and the attention mask is all ones. This is not a throughput
  choice: the Clifford arm's block has `supports_masking = False`, so a padded
  batch would make stage 1 measure the padding policy as much as the block.
- **Evaluation cannot pack, so padding returns.** `embed_texts` sorts by length
  and pads to the batch maximum, and each corpus reports its pad fraction.
  Pad *width* is inert for all three registered arms.
- **SQuAD contexts are Wikipedia paragraphs and the MLM corpus is Wikipedia.**
  Every arm shares the leak, so relative comparisons survive it; no absolute
  claim does.
- **Query and document lengths differ by an order of magnitude** (questions ~59
  characters, contexts ~774) and a sinusoidal encoder's pooled vector is
  length-dependent, so any retrieval number is measuring length mismatch as well
  as content.
- **Stage 2 saves the encoder, not the SimCSE wrapper**, so metrics use
  `pooled_output` while the contrastive loss lives in projection space; the two
  can move in opposite directions.
- **`SimCSEModel.call` keeps dropout ON regardless of the training flag** — in
  SimCSE the positive pair *is* the dropout noise. With it off the two views are
  identical and validation looks excellent and means nothing. Use
  `SimCSEModel.embed` for a deterministic embedding.
- **Stage 2 compiles with `jit_compile=False`.** The SimCSE step fails under XLA
  on this TF 2.18 build (`FAILED_PRECONDITION: Can not combine dim orders and
  requirements`). Stage 1 compiles fine.
- **A benign TF message on the ConvNeXt arm.** Grappler logs `layout failed:
  INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4` from
  the dropout inside `convnext_block`. The pass is skipped, training proceeds,
  losses are finite. Not an error.

## Adding an arm

Add one entry to `MODEL_REGISTRY` in `config.py`. The trainer, sweep and report
pick it up with no further edits — `--model` is generated from the registry, so
the CLI and the study axes cannot drift from it.
