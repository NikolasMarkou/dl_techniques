# ACC-UNet Training

Production training pipeline for `dl_techniques.models.accunet.AccUNet`
(Pattern-4 segmentation trainer). Single-stage supervised training with
two built-in data modes, AdamW + cosine LR schedule, and the standard
`train.common.create_callbacks` callback stack.

## Quick recipes

**Smoke test (CPU, ~1 minute):**

```bash
MPLBACKEND=Agg CUDA_VISIBLE_DEVICES="" .venv/bin/python -m train.accunet.train_accunet \
    --data-mode synthetic --epochs 1 --batch-size 4 \
    --image-size 64 --num-samples 32 --num-classes 1 \
    --output-dir /tmp/accunet_smoke
```

**Oxford-IIIT-Pet binary segmentation (single GPU):**

```bash
MPLBACKEND=Agg .venv/bin/python -m train.accunet.train_accunet \
    --data-mode oxford_pets --image-size 128 --batch-size 16 \
    --epochs 30 --gpu 0
```

`--data-mode oxford_pets` requires `tensorflow-datasets`
(`pip install tensorflow-datasets`). Synthetic mode has no extra
dependency and is the recommended smoke path.

## Input-size contract

AccUNet requires `H, W` divisible by 16 — see
`src/dl_techniques/models/accunet/README.md` §7. The trainer enforces
this for both data modes and raises a clear `ValueError` if you pass a
non-conforming `--image-size`.

## Loss / metric defaults

| Mode | Default loss | Default metrics | Default `--monitor` |
|------|--------------|-----------------|----------------------|
| Binary (`--num-classes 1`) | `combo` (BCE + Dice) | `binary_accuracy`, `dice` | `val_loss` |
| Multi-class (`--num-classes N>=2`) | `combo` | `sparse_categorical_accuracy` | `val_loss` |

Alternative losses are exposed via `--loss`
(`dice`, `focal`, `cross_entropy`, `focal_tversky`, `tversky`,
`lovasz`, `boundary`, `hausdorff`). For binary segmentation, `val_dice`
is a useful early-stopping monitor — pass `--monitor val_dice` to opt in.
Be aware that `create_callbacks` resolves mode by name and may need
ModelCheckpoint mode hints if you change this; `val_loss` is the safe
default.

## Pretrained initialization

```bash
MPLBACKEND=Agg .venv/bin/python -m train.accunet.train_accunet \
    --data-mode oxford_pets --image-size 128 --epochs 30 --gpu 0 \
    --init-from results/some_previous_accunet_run/final_model.keras
```

Loads the backbone (and head, if names match) via
`dl_techniques.utils.weight_transfer.load_weights_from_checkpoint`. The
target model must be built first; the trainer does this via a dummy
forward pass before applying the transfer.

## Artifacts

Each run writes to `results/accunet_<model_name>_<timestamp>/`:

- `final_model.keras` — saved model (round-trip checked at end of training)
- `model_best.keras` — `ModelCheckpoint` best-by-monitor
- `training.csv` — `CSVLogger` per-epoch metrics
- `tensorboard/` — TensorBoard summaries
- `viz/epoch_NNN.png` — `[image | gt | pred]` grids saved every
  `--grid-every-n-epochs` epochs (default 5)

## Reproducibility

`--seed <int>` (default 42) seeds Python, NumPy, TF, and Keras at
startup. The `oxford_pets` data pipeline accepts the same seed for the
shuffle buffer, but TF `Dataset.shuffle` is not fully deterministic
across re-runs at high `num_parallel_calls` — expect tiny ordering
deltas unless you also pin `tf.config.experimental.enable_op_determinism`
(which slows training and is intentionally not turned on by default).
