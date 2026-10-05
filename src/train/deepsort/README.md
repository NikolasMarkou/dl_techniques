# DeepSORT Appearance Trainer

Trains `DeepSortAppearanceNet` (`dl_techniques.models.vision.deepsort`) as a
person re-identification embedding. The multi-target tracker itself
(`DeepSortTracker`) trains nothing — it is classical cascade/IoU association
over the learned gallery.

## Objectives

`--loss-mode cosine-softmax` (default, the reference mode): identity
classification over cosine-softmax logits (stock sparse CE). `--loss-mode
triplet`: batch-hard softmargin triplet loss (`SoftmarginTripletLoss`) over
the L2-normalized features (needs `--k-shots >= 2`). The magnet loss is
transcribed as `magnet_loss_fn` in `losses/` for custom loops; it is
batch-level and cannot compile under stock `fit()`, so it is not a trainer
mode.

## Data

PK batches (`--p-ids` × `--k-shots`) from `--data-source synthetic`
(default: seeded color/stripe identities) or `--data-source market1501`
(standard on-disk Market1501 layout under `--market1501-root`; TFDS ships no
Market1501 builder in the pinned version). Validation holds out disjoint
identities (10%) and reports CMC rank-1 + mAP over cosine ranking.

## Usage

```bash
MPLBACKEND=Agg .venv/bin/python -m train.deepsort.train_deepsort \
    --loss-mode cosine-softmax --epochs 5 --p-ids 8 --k-shots 4 --gpu 1

MPLBACKEND=Agg .venv/bin/python -m train.deepsort.train_deepsort \
    --data-source market1501 --market1501-root /data/Market-1501-v15.09.15 \
    --loss-mode triplet --epochs 50 --gpu 1
```

## Flags

| Flag | Default | Notes |
| :--- | :--- | :--- |
| `--loss-mode` | `cosine-softmax` | `cosine-softmax` or `triplet` |
| `--data-source` | `synthetic` | `synthetic` or `market1501` |
| `--market1501-root` | None | Required for `market1501` |
| `--num-synthetic-ids` / `--shots-per-id` | 24 / 8 | Synthetic identity pool |
| `--p-ids` / `--k-shots` | 8 / 4 | PK batch geometry (batch is always P×K; the reference trains at batch 128 — raise P/K for real runs) |
| `--validation-id-fraction` | 0.1 | Disjoint-ID holdout (transcribed) |
| `--no-augment` | off | Disables training left-right flips (the reference's only aug) |
| `--seed` | 0 | Data, init and shuffle seeding |
| `--dropout-rate` | 0.4 | Transcribed keep_prob 0.6 |
| `--epochs` | 5 | |
| `--steps-per-epoch` / `--val-steps` | 20 / 5 | |
| `--learning-rate` | 1e-3 | Reference Adam default |
| `--optimizer` | `adam` | `adam`, `adamw` or `sgd` (momentum honored for `sgd`) |
| `--lr-schedule` | `constant` | Reference trains at fixed LR; cosine/exponential available |
| `--warmup-epochs` / `--weight-decay` | 0 / 0.0 | Weight decay is AdamW-only (never doubled) |
| `--momentum` | 0.9 | Honored for `--optimizer sgd` only |
| `--gradient-clipping` | 1.0 | Global norm via `optimizer_builder` |
| `--early-stopping-patience` | 10 | Monitor is `val_loss` |
| `--output-dir` / `--experiment-name` | `results` / auto | Standard run-directory contract |
| `--gpu` | None | Passed to `setup_gpu`, not a config field |

## Outputs

`config.json`, `training_history.json`, `training_log.csv`,
`best_model.keras` (`val_loss`), `final_model.keras`, plus logged
rank-1/mAP. Non-finite `val_loss` → exit code 1.
