# Callbacks Package

Keras 3 callbacks for training control, schedules, diagnostics, visualization and evaluation.

## Public API

`__init__.py` exports nothing. Import every class from its own module:

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

Importing `coco_map_callback` raises `ImportError` when `pycocotools` is missing.

## Modules

- `analyzer_callback.py`: `EpochAnalyzerCallback`. Runs `ModelAnalyzer` (data-free weight and spectral analysis by default) into `epoch_000` (before training), `epoch_001`, then `start_epoch + k * epoch_frequency`. A failure raises out of `fit` on purpose (D-117: log-and-continue hid NaN weights behind exit code 0). An unbuilt model skips `epoch_000` with a warning (D-003).
- `anneal_harmonic_exponent.py`: `HarmonicExponentAnnealingCallback`. Linear, cosine or exp schedule of per-level exponents on layers exposing `num_levels` and `set_n_per_level` (`HierarchicalHarmonicHead`). `n_init` / `n_final` are a scalar or one value per level.
- `coco_map_callback.py`: `COCOMAPCallback`. Runs the model over a positional COCO val loader, then `yolo_decode`, per-class NMS, category-id remap and `COCOeval`. Writes `val_map50` (stats[1]) and `val_map5095` (stats[0]) into the epoch logs. `max_eval_batches` restricts `imgIds` to the evaluated images.
- `coco_multitask_visualization.py`: `COCOMultiTaskPredictionGridCallback`. RGB / GT segmentation / predicted segmentation grid, a classification top-K text file, and optional detection boxes, for a dict-output model with `segmentation` and `classification` keys. Assumes 80 COCO classes.
- `consecutive_increase_early_stopping.py`: `ConsecutiveIncreaseEarlyStopping`. Stops after `patience` consecutive strict rises (TopoLM, ICLR 2025). Deliberately not `keras.callbacks.EarlyStopping`, which counts from the best value. A flat evaluation resets the streak. The snapshot is taken only on non-increasing evaluations, BEFORE the decision is recorded (load-bearing order), and restored only when the rule stopped training. Snapshots live in memory, or on disk when `checkpoint_path` is set.
- `convunext_bottleneck_monitor.py`: `ConvUnextBottleneckMonitorCallback`. Reads the trailing bottleneck output of an explicitly passed FULL `expose_bottleneck=True` model (not `self.model`, which is the single-output training view). Logs latent health statistics and writes first-k and top-energy channel grids on a FIXED color scale (D-001; the energy set is recomputed per epoch, D-002) plus cumulative health curves. The only bottleneck monitor in the repo (D-011).
- `depth_visualization.py`: `DepthPredictionGridCallback` (RGB / GT / prediction aligned with `align_scale_and_shift`, sharing the GT's valid range) and `DepthMetricsCurveCallback` (train/val metric panels).
- `ema_shadow_callback.py`: `EMAShadowCallback`. Calls `model.update_ema_shadow(decay=...)` on a step schedule. A float `decay` is normalized once into a constant callable (D-015). Deliberately does not compound `decay ** elapsed_steps` (D-016). Reuses the schedules from `models.vision.depth_anything.teacher_ema` and re-exports them.
- `jepa_visualization.py`: `LatentMaskOverlayCallback` and `PatchPredictionErrorCallback`, for models exposing `encode_frames`, `predictor`, `mask_gen`, `mask_token` and `config.patches_per_side` / `embed_dim` / `mask_prediction_enabled` (`VideoJEPA`). Square patch grids only. Both draw the tube mask with `mask_gen(B, seed=mask_seed)` (same default seed, one fixed mask, the global Keras seed generator untouched) and fire after epochs `frequency`, `2 * frequency`, ... into `epoch_{epoch+1:03d}_*.png`.
- `noise_sigma_curriculum.py`: `NoiseSigmaCurriculumCallback`. Assigns a scheduled `sigma_max` (and optionally `sigma_min`) to live `keras.Variable`s read inside `tf.data.map`. The variables are not serialized; a `from_config` instance is a logged no-op.
- `periodic_reassign.py`: `PeriodicReassignCallback`. Calls `head.reassign(class_means)` from trunk-feature means over the training arrays. The feature extractor is passed directly (required for a nested trunk) or built from a top-level layer name. Fails loud on a missing head, a missing method or an absent class.
- `self_iterate_pool.py`: `SelfIteratePoolCallback`. Epoch-boundary regeneration over a RAM pool whose `current_input` buffer the `from_generator` source reads. A `mix_ratio` share of slots gets the model output fed back as-is; the rest gets fresh ADDITIVE Gaussian noise (D-003: mix, never replace; never re-noise the fed-back slots; additive only). Fails soft.
- `spatial_loss_logger.py`: `SpatialLossLogger`. Writes per-tap `SpatialSmoothness` penalties, `spatial/total`, `spatial/unweighted`, `spatial/share` and `task_loss` into the batch logs. When a `compile_loss` key is present it also writes `spatial/accounted` and `spatial/unaccounted`, the field that reveals a discarded `add_loss`. Taps are found by walking `_flatten_layers`, with a fallback (`Layer.submodules` does not exist on Keras 3.8).
- `temperature_annealing.py`: `TemperatureAnnealingCallback`. Linear, cosine or exp schedule of `temperature` on soft-selection layers. Honors `softplus_temperature` by assigning the softplus inverse.
- `training_curves.py`: `TrainingCurvesCallback`. Groups epoch-log keys into loss / classification / segmentation / detection / depth / other and overwrites one PNG per group in `output_dir`.

## Conventions

### Base class and hooks

Every class subclasses `keras.callbacks.Callback`. The epoch hooks receive a 0-based `epoch`. Batch-end `logs` values in Keras 3 are running means over the epoch so far, not the last batch's value; never combine them with a last-batch quantity without saying so.

### Cadence

New callbacks fire on `(epoch + 1) % frequency == 0`. Validate `frequency >= 1` in `__init__` and raise; do not clamp silently, and do not leave the modulo to divide by zero inside `fit`. When a `start_epoch` exists, count the cadence from it (as `EpochAnalyzerCallback` does) and document the first firing epoch.

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

- Reset per-run state in `on_train_begin`, or document that an instance is single-use.
- A step or update counter that drives a schedule must come from `model.optimizer.iterations`, not a Python counter, or a resumed run restarts the schedule.

### Live objects

Hold variables, arrays and layers by reference, exclude them from `get_config()`, and say in the docstring that a reconstructed instance is inert until they are supplied again. Every `get_config()` key must match a constructor argument name.

### Registration

Keras never deserializes callbacks, so `register_dl_technique` is not needed. Six classes carry it for historical reasons. Do not add it to new callbacks. Do not remove an existing key without searching for string lookups of it.

### Layer discovery

Walk with `_flatten_layers` and a fallback (see `spatial_loss_logger._walk_layers`). Never use `Layer.submodules`. When a name filter matches nothing, warn at WARNING level: a schedule that touches no layer is a silent dead knob.

### Shared math

Schedule interpolation (linear / cosine / exp) is currently copied in three modules. A new schedule callback should import a shared helper rather than add a fourth copy.

### Plotting

- Acquire pyplot only through `dl_techniques.utils.matplotlib_backend.import_pyplot`, never at module scope.
- Close every figure in a `finally`.
- Keep color scales fixed across epochs whenever epochs are compared (D-001).
- Do not call APIs removed in Matplotlib 3.9 (`cm.get_cmap`); use `matplotlib.colormaps[name]`.

### Fixed evaluation batches

Cache them at construction and cap the FORWARD pass to the samples you plot, not just the plot. Random masks or augmentations drawn per epoch defeat cross-epoch comparison; seed them, or draw once and cache.

### Imports and layering

- Callbacks must not import from `src/train/`.
- Current cross-package imports: `analyzer`, `metrics.depth_metrics`, `layers.regularization.spatial_smoothness`, `utils.yolo_decode`, and `models.vision.depth_anything.teacher_ema` (the one `callbacks` to `models` edge).
- New files use `import keras` and qualify at the call site.

### Docstrings

Both Sphinx `:param` and Google `Args:` styles are live. Match the file you edit. Keep plan ids in `DECISION` anchors, not in prose.

### Documentation

Every class must have a row in the README table. Check that a symbol is importable before documenting it.

## Re-deriving counts

Run these instead of trusting any number in a doc:

```bash
ls src/dl_techniques/callbacks/*.py | grep -v __init__ | wc -l
grep -c "^class .*(keras.callbacks.Callback)" src/dl_techniques/callbacks/*.py | grep -v ":0"
grep -rl register_dl_technique src/dl_techniques/callbacks --include=*.py
grep -rn "% self\.\(frequency\|monitor_freq\|epoch_frequency\|regen_freq\|every_n_epochs\)" src/dl_techniques/callbacks
```

## Conventions

- Callbacks follow the standard Keras `keras.callbacks.Callback` interface
- Integrates with the `analyzer` package for model diagnostics during training
- Visualization callbacks lazy-import matplotlib — callers should set `MPLBACKEND=Agg` for headless environments
