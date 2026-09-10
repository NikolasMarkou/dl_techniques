# `train/doc_scanner` — DocScanner training (two independent stages)

DocScanner unwarps a photographed document in two stages: a U2NET-P **localization** module that
predicts the page mask, and a RAFT-lineage **rectification** module that emits a full-resolution
backward warping flow over 12 refinement iterations. The paper trains the two
**independently** (arXiv:2110.14968v2 §4.3), so this directory has **two entry points**, not one.

| Module | What it does |
|---|---|
| `common.py` | The config, the CLI flags, the data gate, the `tf.data` pipeline over both corpora, the losses, the optimizers, `build_model` and `train()`. |
| `train_doc_scanner_segmenter.py` | Stage 1 entry point: BCE on seven deep-supervision heads, Adam, batch 32. |
| `train_doc_scanner_rectifier.py` | Stage 2 entry point: the Eq. 9-14 sequence loss, AdamW, batch 12. |

```bash
# what each CLI offers -- allocates nothing, touches no GPU
MPLBACKEND=Agg .venv/bin/python -m train.doc_scanner.train_doc_scanner_segmenter --help
MPLBACKEND=Agg .venv/bin/python -m train.doc_scanner.train_doc_scanner_rectifier --help

# the synthetic corpus needs no download
MPLBACKEND=Agg .venv/bin/python -m train.doc_scanner.train_doc_scanner_segmenter \
    --epochs 2 --steps-per-epoch 20 --gpu 1
MPLBACKEND=Agg .venv/bin/python -m train.doc_scanner.train_doc_scanner_rectifier \
    --epochs 2 --steps-per-epoch 20 --gpu 1

# the staged UVDoc corpus
MPLBACKEND=Agg .venv/bin/python -m train.doc_scanner.train_doc_scanner_rectifier \
    --data-source uvdoc --epochs 2 --steps-per-epoch 20 --gpu 1
```

Each run writes `results/doc_scanner_<stage>_<variant>_<stamp>/` containing `best_model.keras`,
`config.json`, `training_log.csv` and `training_history.json`. **Nothing under `results/` is ever
deleted by anything in this directory.**

## The two recipes, and what the paper actually says

Verbatim from §4.3:

> *Localization module.* "We use Adam optimizer with a batch size of 32. The initial learning rate
> is set as 1×10⁻⁴, and reduced by a factor of 0.1 after 30 epochs. After 45 epochs, the training
> loss converges. … we randomly replace the background of the distorted image with the texture
> images from Describable Texture Dataset (DTD)."

> *Rectification module.* "We use AdamW optimizer with a batch size of 12. The total training
> iteration is set as 560k, and the learning rate reaches the maximum 1×10⁻⁴ after 27k iterations
> for learning rate warm-up."

| Knob | Segmenter | Rectifier | Source |
|---|---|---|---|
| Optimizer | Adam | AdamW | paper |
| Batch size | 32 | 12 | paper |
| Peak learning rate | 1e-4 | 1e-4 | paper |
| Schedule | ×0.1 at epoch 30 | cosine after warmup | paper (segmenter); cosine is this port's |
| Warmup length | none | 4.8% of the run (2 of 40 epochs, matching 27k of 560k iterations) | paper |
| **Warmup curve** | — | **linear** | **THIS REPO'S CHOICE** |
| **Weight decay** | — (Adam) | **1e-4** | **THIS REPO'S CHOICE** |
| **Gradient clip** | **global norm 1.0** | **global norm 1.0** | **THIS REPO'S CHOICE** |

### The three unspecified hyperparameters

The paper is **silent** on the warmup curve, the weight decay and the gradient clip — everywhere.
That was established by direct raw-text extraction of the ar5iv HTML, not from a summary. The
values this port uses are **chosen, not recovered**, and must not be quoted as a reproduction of
anything:

- **Warmup curve — linear.** The paper gives a warmup *length* and no curve. `WarmupSchedule`
  (`dl_techniques.optimization`) is this repo's one warmup implementation and its ramp is linear;
  a second curve would be a new abstraction with one call site. Pinned by
  `TestTheOptimizerRecipes::test_the_rectifier_warms_up_linearly_from_near_zero`, so the claim
  above cannot quietly go stale.
- **Weight decay — 1e-4.** `AdamW` with no stated decay is ambiguous: Keras defaults to `0.004`,
  PyTorch to `0.01`, and RAFT — the architecture this update block and Eq. 9's sequence loss are
  inherited from — trains with `1e-5`. 1e-4 sits between the ancestor and the framework defaults.
  It is applied by the optimizer and by **nothing else**: no layer in the `doc_scanner` package
  carries a `kernel_regularizer`, because AdamW's decoupled decay plus an L2 penalty decays the
  same parameter twice.
- **Gradient clip — global norm 1.0.** RAFT's value. A 12-step recurrent unroll whose loss is an
  L1 in *absolute pixels* is exactly the setting an unclipped step diverges in.

The learning-rate *floor* (`--final-lr-fraction`, default 0.01 of the peak) is likewise this
port's; the paper states no floor.

## Data

Two corpora, `--data-source synthetic` (default) and `--data-source uvdoc`. Both emit the same
contract, so nothing downstream branches on the source.

### `synthetic`

`dl_techniques.datasets.document_rectification.synthetic_warp` composes closed-form-invertible
warps, so the backward map `f_gt` **and** the forward map `g` are both exact — measured round trip
7.8e-16 in float64. Each sample index *is* its seed, so a sample is reproducible and the
train/validation split is a genuinely held-out set of seeds rather than a re-draw.

Flat page content comes from `--pages-root` (default: the staged DIBCO / hDIBCO / NoisyOffice
scans, 809 images). Backgrounds come from `--backgrounds-root`; DTD is the paper's own named
source and ships on this machine as an **unextracted tarball**, so an empty root falls back to a
procedural low-frequency texture and logs a warning. The background is composited *after* the
geometry and cannot perturb `f_gt`, `g` or `mask` — a run without DTD is less realistic, not
wrong.

### `uvdoc`

`dl_techniques.datasets.document_rectification.uvdoc` densifies UVDoc's coarse 89×61
correspondence lattice, which is already in the backward direction. `load_geometry` costs about
1.06 s and there are only 4,032 distinct geometries behind the 20,000 renders, so geometry is
memoized by `geom_name`: a full pass pays 4,032 densifications (~1.2 h) instead of 20,000 (~6 h).

### Staging the corpus: `prepare_doc_scanner_data.py`

```bash
python -m train.doc_scanner.prepare_doc_scanner_data --dry-run   # say what it would do
python -m train.doc_scanner.prepare_doc_scanner_data             # 27.5 GB, resumable
```

It writes exactly one directory, `<root>/doc_scanner/uvdoc/`, holding `UVDoc_final.zip` and its
`UVDoc_final.zip.ok` marker. **Nothing is extracted** — `UVDocSource` reads members straight out
of the zip, and half the archive's 182,492 members are `__MACOSX/` resource forks.

The path it writes is *derived* from `common.DEFAULT_UVDOC_ROOT` rather than typed again, so the
script cannot fill a directory the trainer does not read (D-047).

Four properties, each with a guard in `tests/test_train/test_doc_scanner/test_prepare_data.py`:

* **Idempotent** — a second run makes zero HTTP requests and writes nothing. Measured against the
  real staged archive: `already-staged`, 182,492 members, size and mtime unchanged.
* **Resumable** — an interruption leaves `UVDoc_final.zip.part`, which the next run continues with
  `Range: bytes=<n>-`. A host that answers `200` to a ranged GET restarts rather than concatenates.
* **Complete or absent, never in between** — completeness is proven by decoding and CRC-checking
  every member, then renaming `.part` and writing `.ok` **last**. A full-length body with one
  flipped byte is refused and leaves no `.zip`: its central directory still lists every member, so
  a size or `infolist()` check would call it staged (D-048).
* **Refuses rather than fills the volume** — `--min-free-gb` (default 100) is checked before the
  URL is even verified.

If it is not staged, `require_staged_uvdoc` raises the trainers' own `MissingTrainingDataError`
naming what is missing (`.part` present means *interrupted*, which re-running resumes) and points
at `--data-source synthetic` as the alternative that needs no download.

### Supervision

| Stage | `y` | Loss |
|---|---|---|
| segmenter | the `(H, W, 1)` page mask, repeated 7× | `BinaryCrossentropy` on each of `d0..d6`; the reported `loss` is their sum, which is U2-Net's own deep-supervision objective |
| rectifier | the `(H, W, 4)` stack `[f_gt(2), g(2)]` | `DocScannerFlowSequenceLoss` (Eq. 9-14), wrapped by `DocScannerRectifierObjective` |

`g` rides in the target tensor because `L_line` needs it and **there is no custom `train_step`
anywhere in this port**: the 12-iteration sequence is a model *output* under `training=True`, so
the whole objective is reachable through stock `compile(loss=...)` / `fit()`.

### Why `val_loss` is not on the same scale as `loss` (rectifier only)

`DocScannerRectifier` uses Keras' own `training` flag to select its output **rank**: `training=True`
returns the whole `(B, 12, H, W, 2)` sequence, anything else returns the last iteration alone. Keras'
`test_step` calls the model with `training=False`, so the validation pass of a plain
`fit(validation_data=...)` hands the compiled loss a rank-4 tensor — and `DocScannerFlowSequenceLoss`
rightly refuses it, naming that exact situation. Measured: without a fix, a run trains for a full
epoch and then dies at the first validation batch.

`DocScannerRectifierObjective` resolves that by scoring the validation pass on what the model
actually emits at inference — the final iteration — whose Eq. 9 weight is exactly `γ⁰ = 1.0`. So
`val_loss` is the `k = K` term of the training objective, unchanged and unscaled, while `loss` is
the weighted sum over all 12. `loss` is therefore roughly `Σγⁱ ≈ 6.9×` the size at equal quality.
Both are minimized and both are monotone in the same thing; checkpoint selection compares
`val_loss` only against itself. The alternative — relaxing the sequence loss to accept rank 4 —
was rejected because it would delete the guard that catches a whole run done at `training=False`,
i.e. one refinement iteration instead of twelve.

## The CLI contract

`main()`'s **first statement parses argv**, so `--help` prints a `usage:` line and exits without
claiming a GPU, building a model or walking a corpus. Exit 0 is not evidence of that — a script
with no parser at all ignores `--help`, runs its whole job and exits 0 anyway — so
`tests/test_train/test_doc_scanner/test_cli_contract.py` asserts the `usage:` line itself and
asserts, through sentinels over `setup_gpu` / `require_training_data` / `train` / `set_seeds` /
`build_model` / `create_dataset`, that none of them was reached.

`require_training_data` runs **before** `setup_gpu` and raises `MissingTrainingDataError` naming
the real reason — which root was looked at, what was found there, and the alternative — rather
than producing an empty-glob mystery after a GPU has been claimed.

**`--stage` is deliberately not a flag.** The stage is fixed by which script you ran; a flag would
let `train_doc_scanner_segmenter.py --stage rectifier` train the other module under this script's
name, recipe and run-directory prefix. It is the one `DocScannerTrainingConfig` field with no CLI
flag, and the exemption is checked from both ends. `--gpu` is the one parser dest that is not a
config field: it acts on the process.

Every other flag reaches the config field it names, generated row by row from the real parser and
the real stage defaults — a hand-typed table is one chance per row to pick a probe value that is
already the default. `DocScannerTrainingConfig` is REGISTERED in
`tests/test_train/test_config_fields_are_live.py`, which is the other end of the same contract:
a field nothing reads fails there.

## Tests

```bash
MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest tests/test_train/test_doc_scanner/ -q
```

`test_cli_contract.py` covers both entry points; `test_pipeline.py` covers the sample producers,
the `tf.data` shapes, the objective's rank dispatch, both optimizer recipes and a short real
`fit()` for each stage. The `y_true` halves are asserted against
`tests/test_datasets/test_document_rectification/backward_map_convention.py` — the shared
instrument that *is* the `f_gt` / `g` contract — rather than against a comment restating it, and
the swap is proven to reden it. The UVDoc arm skips itself when the archive is not staged.

No test writes into repo-root `results/`; every run directory goes to `tmp_path`.
