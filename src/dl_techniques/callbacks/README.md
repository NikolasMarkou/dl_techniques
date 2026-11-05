# Callbacks Module

`dl_techniques.callbacks` contains Keras 3 callbacks for training control, schedules, diagnostics, visualization and evaluation. `__init__.py` exports nothing, so import each class from its module:

```python
from dl_techniques.callbacks.consecutive_increase_early_stopping import ConsecutiveIncreaseEarlyStopping
```

## Contracts shared by every callback (and where they differ)

**Cadence.** Most callbacks fire when `(epoch + 1) % frequency == 0`, so `frequency=5` fires after epochs 5, 10, and so on (1-based). Exceptions:

- `jepa_visualization` uses `epoch % frequency`, so it also fires after the first epoch.
- `EpochAnalyzerCallback` counts from `start_epoch`.
- `PeriodicReassignCallback` uses multiples of `every_n_epochs` that are at or after `start_epoch`.

A cadence of 0 is not validated everywhere; see Known issues.

**Failure policy.** Three kinds coexist:

| Policy | Callbacks |
|---|---|
| Fail loud: an error ends `fit` | `EpochAnalyzerCallback` (on purpose, D-117), `PeriodicReassignCallback`, the schedule callbacks, `COCOMultiTaskPredictionGridCallback` (by accident). |
| Fail soft: log a warning and continue | `ConvUnextBottleneckMonitorCallback`, the JEPA, depth and training-curve callbacks, `SelfIteratePoolCallback`. |
| Fail soft with a placeholder | `COCOMAPCallback` writes `0.0` for both mAP keys when evaluation fails. |

**Writing into `logs`.** `COCOMAPCallback` adds keys to the epoch logs and `SpatialLossLogger` adds keys to the batch logs. Only callbacks that run AFTER the writer see those keys, so list writers before `ModelCheckpoint`, `EarlyStopping` and `CSVLogger`. `History` keeps epoch logs only, so batch-level keys never reach it.

**State.** No callback resets its state in `on_train_begin`. Build a fresh instance for each `fit` call. Step counters are per process, so a resumed run starts them at 0 again.

**Live objects are not serialized.** `NoiseSigmaCurriculumCallback`, `SelfIteratePoolCallback` and `PeriodicReassignCallback` hold variables, arrays or layers by reference and omit them from `get_config()`. A reconstructed instance does nothing until those objects are supplied again.

**Plotting.** Every plotting callback acquires pyplot through `dl_techniques.utils.matplotlib_backend.import_pyplot`. That function defaults to the Agg backend when `MPLBACKEND` is unset and respects it when set.

## Available callbacks

| Domain | Class (module) | Hook | Notes |
|---|---|---|---|
| Early stopping | `ConsecutiveIncreaseEarlyStopping` (`consecutive_increase_early_stopping`) | `on_epoch_end`, `on_train_end` | Stops after `patience` consecutive strict rises (TopoLM rule). Restores the snapshot taken just before the final streak, not the best value seen. Set `checkpoint_path` to keep snapshots on disk. |
| Schedule | `TemperatureAnnealingCallback` (`temperature_annealing`) | `on_epoch_begin` | Linear, cosine or exp schedule from `t_init` to `t_final` on layers exposing `temperature` and `softplus_temperature`. |
| Schedule | `HarmonicExponentAnnealingCallback` (`anneal_harmonic_exponent`) | `on_epoch_begin` | Per-level exponents from `n_init` to `n_final` on layers exposing `num_levels` and `set_n_per_level` (`HierarchicalHarmonicHead`). |
| Schedule | `NoiseSigmaCurriculumCallback` (`noise_sigma_curriculum`) | `on_epoch_begin` | Assigns a scheduled upper (and optional lower) noise sigma to a `keras.Variable` that the `tf.data` noise function reads. |
| Schedule | `EMAShadowCallback` (`ema_shadow_callback`) | `on_train_batch_end` | Calls `model.update_ema_shadow(decay=...)` every `update_every` steps from `start_step` on. `decay` is a float or a schedule (`cosine_ema_schedule`, `linear_ema_schedule`, re-exported here). |
| Data mutation | `SelfIteratePoolCallback` (`self_iterate_pool`) | `on_epoch_end` | Runs `model.predict` over a RAM pool and rewrites the live input buffer in place: a `mix_ratio` share of slots gets the model output fed back, the rest gets fresh additive noise. |
| Data mutation | `PeriodicReassignCallback` (`periodic_reassign`) | `on_epoch_end` | Calls `head.reassign(class_means)` from trunk features over the training arrays. |
| Diagnostics | `SpatialLossLogger` (`spatial_loss_logger`) | `on_train_batch_end` | Splits the reported loss into task and `SpatialSmoothness` components and writes `spatial/<tap>`, `spatial/total`, `spatial/unweighted`, `spatial/share`, `task_loss` and, when `compile_loss` is logged, `spatial/accounted` and `spatial/unaccounted`. |
| Diagnostics | `EpochAnalyzerCallback` (`analyzer_callback`) | `on_train_begin`, `on_epoch_end` | Runs `ModelAnalyzer` (weights and spectral analysis by default) into `epoch_NNN/`. Failures raise. |
| Diagnostics | `ConvUnextBottleneckMonitorCallback` (`convunext_bottleneck_monitor`) | `on_epoch_end`, `on_train_end` | Bottleneck health statistics and fixed-scale feature-map grids. Pass the FULL `expose_bottleneck=True` model, not the training view. |
| Diagnostics | `TrainingCurvesCallback` (`training_curves`) | `on_epoch_end` | One PNG per metric group (loss, classification, segmentation, detection, depth, other), overwritten in place. |
| Visualization | `LatentMaskOverlayCallback`, `PatchPredictionErrorCallback` (`jepa_visualization`) | `on_epoch_end` | Mask overlays and per-patch latent prediction error for `VideoJEPA`-style models. |
| Visualization | `DepthPredictionGridCallback`, `DepthMetricsCurveCallback` (`depth_visualization`) | `on_epoch_end` | RGB / GT / scale-and-shift-aligned prediction grids, and metric curves. |
| Visualization | `COCOMultiTaskPredictionGridCallback` (`coco_multitask_visualization`) | `on_epoch_end` | Segmentation grids, classification top-K text and optional detection boxes for a dict-output COCO model. |
| Evaluation | `COCOMAPCallback` (`coco_map_callback`) | `on_epoch_end` | Decode, NMS and `COCOeval`. Writes `val_map50` and `val_map5095`. Requires `pycocotools` at import time. |

## Usage

### Early stopping on consecutive increases

```python
from dl_techniques.callbacks.consecutive_increase_early_stopping import (
    ConsecutiveIncreaseEarlyStopping,
)

stop = ConsecutiveIncreaseEarlyStopping(monitor="val_loss", patience=3)
model.fit(train_ds, validation_data=val_ds, epochs=100, callbacks=[stop])
print(stop.stopped_at_evaluation)
```

On the in-memory default, every non-increasing evaluation calls `model.get_weights()`, which costs one full weight copy per evaluation.

### Noise curriculum

```python
import keras
import tensorflow as tf
from dl_techniques.callbacks.noise_sigma_curriculum import NoiseSigmaCurriculumCallback

sigma_max = keras.Variable(0.05, trainable=False, dtype="float32")

def add_noise(clean):
    sigma = tf.random.uniform([], 0.0, tf.convert_to_tensor(sigma_max))
    return clean + sigma * tf.random.normal(tf.shape(clean)), clean

train_ds = clean_ds.map(add_noise).batch(32)
curriculum = NoiseSigmaCurriculumCallback(
    sigma_max_var=sigma_max, sigma_max_start=0.05, sigma_max_end=0.5, total_epochs=50
)
```

The variable is read inside `map`, so a prefetch buffer that spans an epoch boundary serves a few batches at the previous sigma.

### Self-iterating denoiser pool

```python
import numpy as np
from dl_techniques.callbacks.self_iterate_pool import SelfIteratePoolCallback

clean = patches.astype(np.float32)  # [N, H, W, C] in [-1, 1]
current = np.clip(clean + 0.1 * np.random.randn(*clean.shape), -1, 1).astype(np.float32)

def gen():
    for i in range(len(clean)):
        yield current[i], clean[i]  # reads the live buffer every epoch

pool_cb = SelfIteratePoolCallback(
    clean_pool=clean, current_input=current,
    get_sigma=lambda: float(sigma_max), mix_ratio=0.5,
)
```

Do not `.repeat()` the pool dataset: the callback writes `current` in place.

### EMA shadow weights

```python
from dl_techniques.callbacks.ema_shadow_callback import EMAShadowCallback, cosine_ema_schedule

ema = EMAShadowCallback(
    decay=cosine_ema_schedule(0.5, 0.9999, total_steps=10_000), start_step=100
)
```

The model must implement `update_ema_shadow(decay=...)`. If it does not, the callback logs one warning and disables itself.

### COCO mAP with checkpointing

```python
import keras
from dl_techniques.callbacks.coco_map_callback import COCOMAPCallback

callbacks = [
    COCOMAPCallback(val_loader, image_size=640, num_classes=80, reg_max=16, strides=(8, 16, 32)),
    keras.callbacks.ModelCheckpoint("best.keras", monitor="val_map5095", mode="max", save_best_only=True),
    keras.callbacks.CSVLogger("log.csv"),
]
```

The loader must not shuffle. `COCOMAPCallback` must come before any callback that reads its keys.

### Spatial loss split

```python
from dl_techniques.callbacks.spatial_loss_logger import SpatialLossLogger

spatial = SpatialLossLogger(log_every=100)
model.fit(train_ds, callbacks=[spatial])
print(spatial.last_recorded)
```