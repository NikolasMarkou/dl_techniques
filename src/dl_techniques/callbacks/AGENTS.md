# Callbacks Package

Keras 3 callbacks for training control, schedules, diagnostics, visualization and evaluation.

## Public API

`__init__.py` exports every public name lazily (PEP 562 `__getattr__`, table `_LAZY_ATTRIBUTES`): the three schedule helpers and `COCO_CLASS_NAMES` as well as the classes, so `from dl_techniques.callbacks import EMAShadowCallback` works and imports only that class's module. `COCOMAPCallback` is importable by name but left out of `__all__`, so a star import does not fail where `pycocotools` is missing. Importing from the defining module still works:

```python
from dl_techniques.callbacks.analyzer_callback import EpochAnalyzerCallback
from dl_techniques.callbacks.anneal_harmonic_exponent import HarmonicExponentAnnealingCallback
from dl_techniques.callbacks.coco_map_callback import COCOMAPCallback
from dl_techniques.callbacks.coco_multitask_visualization import COCOMultiTaskPredictionGridCallback
from dl_techniques.callbacks.consecutive_increase_early_stopping import ConsecutiveIncreaseEarlyStopping
from dl_techniques.callbacks.convunext_bottleneck_monitor import ConvUnextBottleneckMonitorCallback
from dl_techniques.callbacks.depth_visualization import DepthPredictionGridCallback, DepthMetricsCurveCallback
from dl_techniques.callbacks.ema_shadow_callback import EMAShadowCallback, cosine_ema_schedule, linear_ema_schedule
from dl_techniques.callbacks.jepa_visualization import LatentMaskOverlayCallback, PatchPredictionErrorCallback
from dl_techniques.callbacks.noise_sigma_curriculum import NoiseSigmaCurriculumCallback
from dl_techniques.callbacks.periodic_reassign import PeriodicReassignCallback
from dl_techniques.callbacks.self_iterate_pool import SelfIteratePoolCallback
from dl_techniques.callbacks.spatial_loss_logger import SpatialLossLogger
from dl_techniques.callbacks.temperature_annealing import TemperatureAnnealingCallback
from dl_techniques.callbacks.training_curves import TrainingCurvesCallback
```

Importing `coco_map_callback` (or `COCOMAPCallback` from the package root) raises `ImportError` when `pycocotools` is missing; it is declared in the `data` extra of `pyproject.toml`. `_shared.py` is a private helper module with no public callback: `interpolate_schedule` (linear / cosine / exp) and `walk_layers`.

## Modules

- `analyzer_callback.py`: `EpochAnalyzerCallback`. Runs `ModelAnalyzer` (data-free weight and spectral analysis by default) into `epoch_000` (before training), `epoch_001`, then `start_epoch + k * epoch_frequency`. A failure raises out of `fit` on purpose (log-and-continue hid NaN weights behind exit code 0). An unbuilt model skips `epoch_000` with a warning.
- `anneal_harmonic_exponent.py`: `HarmonicExponentAnnealingCallback`. Linear, cosine or exp schedule of per-level exponents on layers exposing `num_levels` and `set_n_per_level` (`HierarchicalHarmonicHead`). `n_init` / `n_final` are a scalar or one value per level.
- `coco_map_callback.py`: `COCOMAPCallback`. Runs the model over a positional COCO val loader, then `yolo_decode`, per-class NMS, category-id remap and `COCOeval`. Writes `val_map50` (stats[1]) and `val_map5095` (stats[0]) into the epoch logs. A model that emits no detections on images with ground truth scores a true 0.0; every other non-measurement (the model raising, a missing detection key, `loadRes` rejecting the results, no ground truth among the evaluated images) logs `NaN`, never a placeholder 0.0. `max_eval_batches` restricts `imgIds` to the evaluated images.
- `coco_multitask_visualization.py`: `COCOMultiTaskPredictionGridCallback`. RGB / GT segmentation / predicted segmentation grid, a classification top-K text file, and optional detection boxes, for a dict-output model with `segmentation` and `classification` keys. Assumes 80 COCO classes, drawn in 80 distinct colors (`_class_palette`: `tab20`, `tab20b`, `tab20c`, `Set3`, `Paired`, duplicates skipped). A failure is logged with its traceback and the run continues.
- `consecutive_increase_early_stopping.py`: `ConsecutiveIncreaseEarlyStopping`. Stops after `patience` consecutive strict rises (TopoLM, ICLR 2025). Deliberately not `keras.callbacks.EarlyStopping`, which counts from the best value. A flat evaluation resets the streak. A `NaN` or infinite monitored value stops training at once, takes no snapshot and does not move the reference value, so the restore brings back the last finite snapshot. The snapshot is taken only on non-increasing evaluations, BEFORE the decision is recorded (load-bearing order), and restored only when the rule stopped training. Snapshots live in memory, or on disk when `checkpoint_path` is set.
- `convunext_bottleneck_monitor.py`: `ConvUnextBottleneckMonitorCallback`. Reads the trailing bottleneck output of an explicitly passed FULL `expose_bottleneck=True` model (not `self.model`, which is the single-output training view). Logs latent health statistics and writes first-k and top-energy channel grids on a FIXED color scale (the energy set is recomputed per epoch) plus cumulative health curves, restarted in `on_train_begin`. The only bottleneck monitor in the repo.
- `depth_visualization.py`: `DepthPredictionGridCallback` (RGB / GT / prediction aligned with `align_scale_and_shift`, sharing the GT's valid range) and `DepthMetricsCurveCallback` (train/val metric panels).
- `ema_shadow_callback.py`: `EMAShadowCallback`. Calls `model.update_ema_shadow(decay=...)` on a step schedule. A float `decay` is normalized once into a constant callable. Deliberately does not compound `decay ** elapsed_steps`. The step is read from `model.optimizer.iterations` (global, 0-based, no counter on the instance), so a resumed run continues the schedule and `start_step` / `update_every` are global-step quantities. A model compiled with `steps_per_execution != 1`, or an optimizer with `gradient_accumulation_steps` set, is refused in `on_train_begin` (with accumulation `iterations` advances once per k batches, not per batch). Under `mixed_float16` the counter and the accumulation setting are read from the `LossScaleOptimizer`'s inner optimizer, whose wrapper reports neither; a batch the loss scaler skips leaves `iterations` unchanged, so that step's update fires again. Reuses the schedules from `models.vision.depth_anything.teacher_ema` and re-exports them.
- `jepa_visualization.py`: `LatentMaskOverlayCallback` and `PatchPredictionErrorCallback`, for models exposing `encode_frames`, `predictor`, `mask_gen`, `mask_token` and `config.patches_per_side` / `embed_dim` / `mask_prediction_enabled` (`VideoJEPA`). Square patch grids only. Both draw the tube mask with `mask_gen(B, seed=mask_seed)` (same default seed `DEFAULT_MASK_SEED`, one fixed mask, the global Keras seed generator untouched) and fire on `(epoch + 1) % frequency == 0`, so after epochs `frequency`, `2 * frequency`, ..., into `epoch_{epoch+1:03d}_*.png`.
- `noise_sigma_curriculum.py`: `NoiseSigmaCurriculumCallback`. Assigns a scheduled `sigma_max` (and optionally `sigma_min`) to live `keras.Variable`s read inside `tf.data.map`. The variables are not serialized; a `from_config` instance is a logged no-op.
- `periodic_reassign.py`: `PeriodicReassignCallback`. Calls `head.reassign(class_means)` from trunk-feature means over the training arrays. The feature extractor is passed directly (required for a nested trunk) or built from a top-level layer name. The first reassignment is at `start_epoch` (1-based), then every `every_n_epochs` from there (`start_epoch + k * every_n_epochs`, the `EpochAnalyzerCallback` rule); the cached extractor is dropped in `on_train_begin`. Fails loud on a missing head, a missing method or an absent class. `get_config()` cannot rebuild it: the arrays are left out, and `head` is in the config only when it was given by name (a live head layer is not).
- `self_iterate_pool.py`: `SelfIteratePoolCallback`. Epoch-boundary regeneration over a RAM pool whose `current_input` buffer the `from_generator` source reads. A `mix_ratio` share of slots gets the model output fed back as-is; the rest gets fresh ADDITIVE Gaussian noise (mix, never replace; never re-noise the fed-back slots; additive only). History, epoch count and slot RNG are reset in `on_train_begin`. Fails soft.
- `spatial_loss_logger.py`: `SpatialLossLogger`. Writes per-tap `SpatialSmoothness` penalties, `spatial/total`, `spatial/unweighted`, `spatial/share` and `task_loss` into the batch logs. When a `compile_loss` key is present it also writes `spatial/accounted` and `spatial/unaccounted`, the field that reveals a discarded `add_loss`. Taps are found with `_shared.walk_layers` (`_flatten_layers` with a fallback; `Layer.submodules` does not exist on Keras 3.8). The tap scalars are read to the host on every batch (the running mean needs every batch, so `log_every` does not cut it), as one stacked copy rather than one per tap: about 0.94 ms for 48 taps on CPU against 7.9 ms per-tap; the GPU cost was not measured. Use it with `steps_per_execution=1`, which it needs anyway.
- `temperature_annealing.py`: `TemperatureAnnealingCallback`. Linear, cosine or exp schedule of `temperature` on soft-selection layers. Honors `softplus_temperature` by assigning the softplus inverse.
- `training_curves.py`: `TrainingCurvesCallback`. Groups epoch-log keys into loss / classification / segmentation / detection / depth / other and overwrites one PNG per group in `output_dir`.

## Conventions

### Base class and hooks

Every class subclasses `keras.callbacks.Callback`. The epoch hooks receive a 0-based `epoch`. Batch-end `logs` values in Keras 3 are running means over the epoch so far, not the last batch's value; never combine them with a last-batch quantity without saying so.

### Cadence

New callbacks fire on `(epoch + 1) % frequency == 0`. Validate `frequency >= 1` in `__init__` and raise `ValueError` naming the value; do not clamp silently, and do not leave the modulo to divide by zero inside `fit`. Every cadence argument in the package follows this: `frequency`, `monitor_freq`, `epoch_frequency`, `regen_freq`, `every_n_epochs`, `log_every`, `patience` and `update_every` all raise below 1 (`start_step` below 0). When a `start_epoch` exists, count the cadence from it (as `EpochAnalyzerCallback` does) and document the first firing epoch.

### Failure policy

Choose by what the callback does:

- **It changes training** (schedules, data mutation, weight restore): fail loud, unless a stale state is explicitly acceptable and the warning says so (`SelfIteratePoolCallback`).
- **It produces a metric other callbacks act on**: on failure, write nothing or `NaN`, never a placeholder number. A fabricated `0.0` drives `ModelCheckpoint` and `EarlyStopping` silently.
- **It only plots or logs**: catch everything, including the plotting path, warn with `exc_info=True`, and continue.

### Writing into `logs`

A callback that adds keys must document its list-order requirement: only callbacks after it see the keys. Batch-level keys never reach `History`. When a caller needs the values back, keep a copy (`SpatialLossLogger.last_recorded`).

### Monitored values

Treat `NaN` and `inf` explicitly. Every comparison with `NaN` is `False`, so a rule written as `value > previous` reads a diverged run as an improvement.

### State lifecycle

- Reset per-run state in `on_train_begin`, or document that an instance is single-use. The early-stopping streak and snapshot, the curve and history lists, the self-iterate pool RNG and the `PeriodicReassignCallback` extractor are reset there, so one instance may be reused across `fit` calls.
- A step or update counter that drives a schedule must come from `model.optimizer.iterations`, not a Python counter, or a resumed run restarts the schedule (`EMAShadowCallback` does this, and refuses `steps_per_execution > 1` and `gradient_accumulation_steps > 1`, the two settings that make `iterations` stop advancing once per batch).

### Live objects

Hold variables, arrays and layers by reference, exclude them from `get_config()`, and say in the docstring that a reconstructed instance is inert until they are supplied again. Every `get_config()` key must match a constructor argument name.

### Registration

Keras never deserializes callbacks, so `register_dl_technique` is not needed. Six classes carry it for historical reasons. Do not add it to new callbacks. Do not remove an existing key without searching for string lookups of it.

### Layer discovery

Walk with `_shared.walk_layers` (`_flatten_layers` with a fallback; the root model is excluded). Never use `Layer.submodules`. When a name filter matches nothing, warn at WARNING level: a schedule that touches no layer is a silent dead knob.

### Shared math

Schedule interpolation (linear / cosine / exp) lives in `_shared.interpolate_schedule`, used by the temperature, harmonic-exponent and noise-sigma callbacks. A new schedule callback imports it rather than adding a copy. `models/vision/depth_anything/teacher_ema.py` still carries its own cosine/linear algebra (deliberately not moved).

### Plotting

- Acquire pyplot only through `dl_techniques.utils.matplotlib_backend.import_pyplot`, never at module scope.
- Close every figure in a `finally`.
- Keep color scales fixed across epochs whenever epochs are compared.
- Do not call APIs removed in Matplotlib 3.9 (`cm.get_cmap`); use `matplotlib.colormaps[name]`.

### Fixed evaluation batches

Cache them at construction and cap the FORWARD pass to the samples you plot, not just the plot. Random masks or augmentations drawn per epoch defeat cross-epoch comparison; seed them, or draw once and cache.

### Imports and layering

- Callbacks must not import from `src/train/`.
- Current cross-package imports: `analyzer`, `metrics.depth_metrics`, `layers.regularization.spatial_smoothness`, `utils.yolo_decode`, and `models.vision.depth_anything.teacher_ema` (the one `callbacks` to `models` edge).
- New files use `import keras` and qualify at the call site.

### Docstrings

Both Sphinx `:param` and Google `Args:` styles are live. Match the file you edit. Keep plan ids and decision numbers in `DECISION` anchors, not in prose.

### Documentation

Every class must have a row in the README table. Check that a symbol is importable before documenting it.

## Re-deriving counts

Run these instead of trusting any number in a doc:

```bash
ls src/dl_techniques/callbacks/*.py | grep -v __init__ | wc -l
grep -c "^class .*(keras.callbacks.Callback)" src/dl_techniques/callbacks/*.py | grep -v ":0"
grep -c "^@register_dl_technique(" src/dl_techniques/callbacks/*.py | grep -v ":0"
grep -rn "% self\.\(frequency\|monitor_freq\|epoch_frequency\|regen_freq\|every_n_epochs\|log_every\)" src/dl_techniques/callbacks
```
