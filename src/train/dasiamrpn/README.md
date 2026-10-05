# DaSiamRPN Trainer

Trains `DaSiamRPN` (`dl_techniques.models.vision.dasiamrpn`) on centered
exemplar/search pairs with per-anchor classification + box-delta losses
(`DaSiamRPNClsLoss` / `DaSiamRPNRegLoss`, combined via `loss_weights` at
compile time). Anchors come from the model package's reference-transcribed
factory, matched in `datasets/vision/tracking.py`.

## Data

Same pair protocol as the SiamFC trainer: `--data-source synthetic`
(default, seeded, used by smoke tests) or `--data-source coco` (real COCO
2017 pairs via TFDS). Pairs stay centered; the ground-truth box sits at the
anchor-grid center so centered anchors match positive.

## Usage

```bash
MPLBACKEND=Agg .venv/bin/python -m train.dasiamrpn.train_dasiamrpn \
    --variant otb --epochs 5 --batch-size 2 --data-source synthetic --gpu 1

MPLBACKEND=Agg .venv/bin/python -m train.dasiamrpn.train_dasiamrpn \
    --variant big --data-source coco --steps-per-epoch 500 --val-steps 50 \
    --epochs 50 --gpu 1
```

## Flags

| Flag | Default | Notes |
| :--- | :--- | :--- |
| `--variant` | `otb` | `big` (wide), `vot`, or `otb` — the released configs |
| `--data-source` | `synthetic` | `synthetic` or `coco` |
| `--coco-data-dir` / `--coco-split` | None / `train` | TFDS source; validation uses `validation` |
| `--num-synthetic-train` / `--num-synthetic-val` | 2048 / 256 | |
| `--steps-per-epoch` / `--val-steps` | None | Derived for synthetic; required for COCO |
| `--no-augment` / `--brightness-delta` | off / 0.125 | Photometric jitter only |
| `--seed` | 0 | Data, init and shuffle seeding |
| `--exemplar-size` / `--search-size` | 127 / 271 | Score grid is 19 |
| `--no-batch-norm` | off | Removes backbone BatchNorm |
| `--pos-iou` / `--neg-iou` | 0.6 / 0.3 | Anchor matching thresholds |
| `--cls-weight` / `--reg-weight` | 1.0 / 1.0 | `loss_weights` at compile time, not baked into the losses |
| `--huber-delta` | 1.0 | Smooth-L1 knee |
| `--batch-size` / `--epochs` | 2 / 5 | RPN forwards are heavy; raise deliberately |
| `--learning-rate` | 1e-3 | AdamW default |
| `--optimizer` | `adamw` | `adamw` or `sgd` |
| `--lr-schedule` | `cosine_decay` | Plus `--warmup-epochs` / `--weight-decay` (AdamW only) / `--gradient-clipping` / `--momentum` (`sgd` only) |
| `--early-stopping-patience` | 10 | Monitor is `val_loss` |
| `--output-dir` / `--experiment-name` | `results` / auto | Standard run-directory contract |
| `--gpu` | None | Passed to `setup_gpu`, not a config field |

## Outputs

`config.json`, `training_history.json`, `training_log.csv`,
`best_model.keras` (`val_loss`), `final_model.keras`. Non-finite
`val_loss` → exit code 1.
