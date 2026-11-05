## Audit

The digest lists `README.md` and `__init__.py` but includes neither file's content, so both docs below are rewritten from the code. The package has 15 modules (excluding `__init__.py`) and 17 callback classes.

### High severity

| Location | Finding |
|---|---|
| `coco_map_callback.COCOMAPCallback` | When evaluation fails, the callback writes `val_map50 = val_map5095 = 0.0` into the logs. That is indistinguishable from a real zero, so `ModelCheckpoint` / `EarlyStopping` act on a fabricated value. Image ids are also assigned by batch position, and the docstring asks for `config.shuffle = False` but nothing checks it. A shuffled loader silently scores predictions against the wrong images. |
| `consecutive_increase_early_stopping` | A `NaN` monitor value fails every comparison, so it counts as "not increased". It resets the streak and snapshots the NaN weights, so a diverged run never stops and the restore target becomes the broken model. With `min_delta > 0`, each sub-threshold rise also re-snapshots, so the restore target ratchets toward progressively worse weights. |
| `ema_shadow_callback.EMAShadowCallback` | `_step` and `_update_count` live in Python and restart at 0 on every `fit` and every process. On resume, `start_step` is applied again and a decay schedule restarts at its low end, pulling the shadow hard toward the live weights. Use `model.optimizer.iterations` instead. Separately, the `get_config` docstring claims the sibling callbacks implement no `get_config`/`from_config`; three of them do. |
| `spatial_loss_logger.SpatialLossLogger` | In Keras 3, batch-end `logs["loss"]` is the running epoch mean, while `last_spatial_loss` is the last batch's value, so `task_loss` and `share` mix the two. The `compile_loss` log key may not exist on Keras 3.8 (verify). If it is absent, `spatial/unaccounted`, the only advertised detector for a discarded `add_loss`, is never emitted and nothing warns. The docstrings name `spatial_share` while the code writes `spatial/share`, and the module docstring's "identically 0.0" claim contradicts the in-code comment. |
| `noise_sigma_curriculum.NoiseSigmaCurriculumCallback` | `schedule="exp"` validates only the `sigma_max` endpoints. With `sigma_min_var` set and the default `sigma_min_start=0.0`, `math.log(0)` raises in `on_epoch_begin` and kills `fit`. `sigma_min > sigma_max` is not rejected. |
| `coco_multitask_visualization` | `on_epoch_end` uses `try/finally` with no `except`, so any segmentation or classification plotting error propagates and kills training; its siblings swallow such errors. `cm.get_cmap` was removed in Matplotlib 3.9 (verify what `import_pyplot(with_cm=True)` returns). `tab20` resampled to 80 classes repeats each color four times. |

### Medium severity

| Location | Finding |
|---|---|
| `self_iterate_pool` | `mean_residual` is computed after `current_input` has been overwritten. Regenerated slots compare `clip(denoised)` to `denoised` (about 0); fresh slots compare new noise against a prediction made on the old input. The logged number measures neither. Nothing caps how many times a slot is fed back. With a `.repeat()` dataset, prefetch can read the buffer across the in-place write. |
| Cadence arguments | `frequency` / `monitor_freq` / `epoch_frequency` of 0 raises `ZeroDivisionError` outside any `try` in `convunext_bottleneck_monitor`, `analyzer_callback`, both depth callbacks, `training_curves` and `coco_multitask_visualization`. `jepa_visualization` and `coco_map_callback` clamp silently instead; the rest raise in `__init__`. |
| `periodic_reassign` | The first reassignment happens at the first multiple of `every_n_epochs` at or after `start_epoch`, not at `start_epoch` as documented (`EpochAnalyzerCallback` counts from `start_epoch`). `get_config` emits `head_layer_name`, but the constructor argument is `head`, and `x`, `y` are required. The class is registered, yet cannot round-trip. |
| `depth_visualization` | `DepthPredictionGridCallback` forwards the whole validation set; `max_samples` limits plotting only. `DepthMetricsCurveCallback` appends values only when a key is present, so under `validation_freq > 1` the validation curve is plotted against the wrong epochs. |
| `jepa_visualization` | `mask_gen` draws a fresh random mask every epoch, and the two callbacks draw different masks from each other; only the pixels are fixed across epochs. Both fire on `epoch % frequency`, which includes epoch 0, unlike every sibling. `_to_display_image` divides standardized inputs whose max exceeds 1.5 by 255. |
| `temperature_annealing`, `anneal_harmonic_exponent` | When `layer_names` matches no layer, the callback is a silent no-op (logged at debug only). `TemperatureAnnealingCallback` overwrites a trainable variable every epoch, discarding the gradient updates. A NumPy array passed as `n_init` is treated as a scalar and fails `float()`. |
| State lifecycle | Nothing resets state in `on_train_begin`, so reusing an instance across `fit` calls carries over the streak (early stopping), `_batch` (spatial logger), the step counters (EMA), and the curve histories (training curves, depth metrics). |

### Low severity / documentation

| Location | Finding |
|---|---|
| `training_curves` | The docstring says output goes to `{output_dir}/training_curves/`; the code writes to `output_dir`. `detection.png` is missing from the list. `float(v)` over the logs runs outside the `try`. |
| `spatial_loss_logger` | The verbose path divides by `reported` with no zero guard. The `logs is None` fallback reads `self.logs`, which does not exist. |
| Duplication | The schedule interpolation is copied in three modules and the layer-tree walk in three. |
| Registration | 6 of 17 classes carry `register_dl_technique`. Keras never deserializes callbacks, so the decorator only adds registry surface. |
| Docstrings | Process noise: "M2 (plan_...)", "installed during an earlier plan", "risk spike (plan STEP 1)". `COCOMAPCallback` is described as "CliffordNet" but is generic YOLOv12. |
| Layering | `ema_shadow_callback` imports from `models.vision.depth_anything`. `coco_map_callback` raises `ImportError` at import time without `pycocotools`. |
| `AGENTS.md` | Lists 5 of 15 modules. Tells callers to set `MPLBACKEND`, although `import_pyplot` already defaults to Agg. |
