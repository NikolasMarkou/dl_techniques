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
| `--dataset` | `imagenet` | `cifar10`, `cifar100`, `imagenet` |
| `--image-size` | `227` | AlexNet's own extent; must be >= 55 |
| `--num-classes` | auto | 10 / 100 / 1000 |
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

## The paper's recipe

90 epochs, SGD momentum 0.9, LR 0.1 divided by 10 every 30 epochs, weight decay
5e-4, batch 256. That is the default here, with one naming caveat:

> **The paper's "halve every 30 epochs" is `--lr-schedule exponential_decay` here.**
> `dl_techniques.optimization` has three schedule types — `cosine_decay`,
> `exponential_decay`, `cosine_decay_restarts` — and **no** `step_decay`. Passing one
> raises `ValueError` naming the three that exist. The paper actually divided by 10
> every 30 epochs; some reproductions halve instead, which is
> `--lr-decay-factor 0.5`.

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