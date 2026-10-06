# Callbacks Package

Custom Keras callbacks for training instrumentation.

## Modules

- `analyzer_callback.py` — Keras callback that runs `ModelAnalyzer` at specified training intervals to generate analysis reports during training
- `anneal_harmonic_exponent.py` — `HarmonicExponentAnnealingCallback`: anneals per-level harmonic exponents (`linear` / `cosine` / `exp` schedules) on layers exposing `num_levels` + `set_n_per_level` (i.e. `HierarchicalHarmonicHead`)
- `consecutive_increase_early_stopping.py` — `ConsecutiveIncreaseEarlyStopping`: stop after N evaluations whose monitored value strictly RISES. Deliberately NOT `keras.callbacks.EarlyStopping`, which counts from the best value seen — the two disagree exactly where it matters, on a fine validation cadence. A single flat evaluation resets the streak. Restores the snapshot taken immediately before the final streak (the last *good* evaluation, not the best one seen). Snapshots are in memory by default and on disk when `checkpoint_path=` is set
- `spatial_loss_logger.py` — `SpatialLossLogger`: splits each batch's reported loss into task and spatial components, per tap and in total. `task_loss = loss - spatial/total` holds by construction whatever the training loop did with the term, so it cannot reveal a backbone whose `add_loss` is being discarded; the field that does is **`spatial/unaccounted`**, derived from `loss - compile_loss`, which is non-zero exactly when the head is not aggregating. Reads the taps by walking the model's layer tree (`_flatten_layers` with a public fallback — `Layer.submodules` does not exist on Keras 3.8), so taps nested inside blocks are found
- `depth_visualization.py` — Depth estimation monitoring callbacks:
  - `DepthPredictionGridCallback` — RGB | GT depth | predicted depth comparison grids
  - `DepthMetricsCurveCallback` — Training/validation metric curve plots (loss, AbsRel, delta)

## Conventions

- `__init__.py` is empty — import directly from modules
- Callbacks follow the standard Keras `keras.callbacks.Callback` interface
- Integrates with the `analyzer` package for model diagnostics during training
- Visualization callbacks lazy-import matplotlib — callers should set `MPLBACKEND=Agg` for headless environments
