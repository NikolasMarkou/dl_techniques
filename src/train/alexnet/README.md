# AlexNet training

Pattern-1 vision-classification pipeline for
[`dl_techniques.models.vision.alexnet`](../../../src/dl_techniques/models/vision/alexnet/).

```bash
MPLBACKEND=Agg .venv/bin/python -m train.alexnet.train_alexnet --help
```

Invoke from the **repo root** so runs land in the repo-root `results/` directory
(`--output-dir` is resolved against the current working directory, never the repo
root implicitly).

## Flags

| Flag | Default | Notes |
|---|---|---|
| `--dataset` | `imagenette` | `imagenette`, `cifar10`, `cifar100`, `imagenet` |
| `--image-size` | `227` | AlexNet's own extent; must be >= 55 |
| `--num-classes` | auto | 10 / 10 / 100 / 1000 |
| `--dropout` | `0.5` | after fc6 and fc7 |
| `--lrn-depth-radius` | `2` | n = 2·radius+1, so the paper's n=5 |
| `--lrn-alpha` / `--lrn-beta` / `--lrn-k` | `1e-4` / `0.75` / `1.0` | the paper's LRN constants |
| `--epochs` | `90` | the paper's count |
| `--batch-size` | `256` | the paper's batch |
| `--optimizer` | `sgd` | `sgd`, `adam`, `adamw` |
| `--learning-rate` | `0.1` | the paper's initial LR |
| `--lr-schedule` | `exponential_decay` | **there is no `step_decay` key in this library** |
| `--lr-decay-steps` / `--lr-decay-factor` | `30` / `0.1` | LR × factor every N epochs |
| `--weight-decay` | `5e-4` | the paper's value |
| `--momentum` | `0.9` | the paper's value |
| `--gpu` | `None` | device index |

## Imagenette (the default dataset)

AlexNet is an ImageNet architecture, and Imagenette (`imagenette/320px-v2`, 9,469
train / 3,925 validation images, 10 ImageNet classes) is the ImageNet subset on disk.
It is read from TFDS under `$TFDS_DATA_DIR` and never downloaded; an absent record set
fails loudly. The pipeline is the shared `train.common.build_raw_image_dataset` with
`crop_policy="random_resized_crop"`, at the model's own 227 input:

- training: a random-resized crop (area 0.08 to 1, aspect 3/4 to 4/3) of the
  full-resolution image, resized to 227, plus a horizontal flip;
- validation: the shorter side resized to 227 / 0.875 = 259, then the centre 227 crop;
- both normalised with the ImageNet mean/std, since Imagenette is ImageNet data.

```bash
MPLBACKEND=Agg .venv/bin/python -m train.alexnet.train_alexnet \
    --epochs 30 --batch-size 128 --learning-rate 0.01 --gpu 0
```

CIFAR-10 and CIFAR-100 are each normalised with their own training-set mean/std
(`CIFAR10_MEAN`/`STD`, `CIFAR100_MEAN`/`STD` in `dl_techniques/datasets/vision/classification.py`),
loaded by `train.common.classification_data` (the pixels are upsampled to `--image-size` before
standardizing). Selection is a held-out slice of train (`--validation-split`, default 0.1; ImageNet:
`--holdout-per-class`), never the test set; Imagenette keeps its TFDS train/validation pair.

## Model analysis

After training, one weights + spectral analysis of the trained model (the best
epoch's weights) is written to `model_analysis/` by the shared
`train.common.run_summary.run_data_free_analysis`; if it does not land, the run
fails. `--epoch-analysis-every N` also runs it every N epochs into
`epoch_analysis/`; the default 0 never does. Measured on the RTX 4070 at 227 px
with batch 128: an Imagenette epoch takes 9 to 10 s and the per-epoch analysis 296
to 305 s (plan-2026-10-14T062616-81938e81 D-114). The end-of-run analysis runs on
four BLAS threads: 177.8 s, against 381.7 s on one (default recipe, batch 256,
3 repeats each; D-243).

## The default recipe

90 epochs, SGD momentum 0.9, LR 0.1 divided by 10 every 30 epochs, weight decay
5e-4, batch 256, with one naming caveat:

> **The step decay ("divide by 10 every 30 epochs") is `--lr-schedule exponential_decay` here.**
> `dl_techniques.optimization` has three schedule types — `cosine_decay`,
> `exponential_decay`, `cosine_decay_restarts` — and **no** `step_decay`. Passing one
> raises `ValueError` naming the three that exist. The default divides by 10
> every 30 epochs; some reproductions halve instead, which is
> `--lr-decay-factor 0.5`.

> **Two defaults are not the paper's.** The paper (section 5) used a batch of 128 and
> a learning rate of 0.01, divided by 10 whenever the validation error stopped
> improving (three times in about 90 epochs). The batch 256 and LR 0.1 defaults here
> are not those numbers.

## Input size, and why CIFAR is awkward here

AlexNet was defined at 227×227. The model's minimum spatial extent is **55** — the
final pool is `padding='valid'` and collapses below that, which the model refuses up
front rather than returning an all-NaN map.

CIFAR is 32×32 natively, so a CIFAR run must upsample. The trainer enforces the
minimum and errors by name if `--image-size` is too small. Cost grows with the
square of the input, so a CIFAR smoke run at `--image-size 64` is far cheaper than
the faithful 227 — at the price of no longer being the paper's configuration.

## No run was executed against this pipeline

The trainer was verified by a **synthetic** 2-epoch run (batch 4, image size 64)
that completed the full pipeline — checkpointing, `val_accuracy` monitoring, epoch
analyzer, metrics figures — writing only into a temporary directory. **No accuracy
figure is claimed for this script**; the model package's parameter counts and shapes
are all measured, but its trained behaviour is not.

## Weight decay

Follows the repo rule: `AdamW` applies decoupled decay internally, so the script
passes `weight_decay` to `optimizer_builder` only. With `sgd` or `adam` the L2
penalty is reported in the log. It is never applied twice.