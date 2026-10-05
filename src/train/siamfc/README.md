# SiamFC Trainer

Trains `SiamFC` (`dl_techniques.models.vision.siamfc`) on centered
exemplar/search pairs with the radius-based logistic loss
(`SiamFCLogisticLoss`).

## Data

`--data-source synthetic` (default): seeded noise backgrounds with a
rectangle target — offline, deterministic, used by the smoke tests.
`--data-source coco`: real pairs from COCO 2017 via TFDS (one uniformly
sampled box per image; requires the TFDS download or `--coco-data-dir`).
Pairs stay centered (no translation jitter) so the centered label stays
valid; only photometric jitter applies (`--no-augment` disables it).

## Usage

```bash
MPLBACKEND=Agg .venv/bin/python -m train.siamfc.train_siamfc \
    --epochs 10 --batch-size 8 --data-source synthetic --gpu 1

MPLBACKEND=Agg .venv/bin/python -m train.siamfc.train_siamfc \
    --data-source coco --steps-per-epoch 500 --val-steps 50 \
    --epochs 50 --gpu 1
```

## Flags

| Flag | Default | Notes |
| :--- | :--- | :--- |
| `--data-source` | `synthetic` | `synthetic` or `coco` |
| `--coco-data-dir` | None | TFDS data dir (default cache when None) |
| `--coco-split` | `train` | Train split; validation always uses `validation` |
| `--num-synthetic-train` / `--num-synthetic-val` | 2048 / 256 | Synthetic pair counts |
| `--steps-per-epoch` / `--val-steps` | None | Derived from synthetic counts; required for COCO |
| `--no-augment` | off | Disables photometric jitter |
| `--brightness-delta` | 0.125 | Brightness jitter bound |
| `--seed` | 0 | Seeds data, init (via `set_seeds`) and shuffle |
| `--exemplar-size` / `--search-size` | 127 / 255 | Paper sizes; score map is 17 |
| `--no-batch-norm` | off | Removes backbone BatchNorm |
| `--pos-radius-px` / `--neg-radius-px` | 25.0 / 50.0 | Reference label radii (search pixels) |
| `--batch-size` / `--epochs` | 8 / 10 | |
| `--learning-rate` | 1e-3 | AdamW default; the paper used SGD 1e-2→1e-5 |
| `--optimizer` | `adamw` | `adamw` or `sgd` (momentum honored for `sgd`) |
| `--lr-schedule` | `cosine_decay` | Plus `--warmup-epochs`, `--weight-decay` (AdamW only, never doubled with a kernel regularizer), `--gradient-clipping` |
| `--early-stopping-patience` | 10 | Monitor is `val_loss` |
| `--output-dir` / `--experiment-name` | `results` / auto | Run dir holds `config.json`, `training_log.csv`, `best_model.keras`, `final_model.keras` |
| `--gpu` | None | Passed to `setup_gpu`, not a config field |

## Outputs

Standard run-directory contract under `--output-dir`: `config.json`,
`training_history.json`, `training_log.csv`, `best_model.keras`
(`val_loss`), `final_model.keras`. `main()` returns a process exit code
(non-finite `val_loss` → 1).
