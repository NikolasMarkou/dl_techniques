# Callbacks Package

Custom Keras callbacks for training instrumentation.

## Modules

- `analyzer_callback.py` — Keras callback that runs `ModelAnalyzer` at specified training intervals to generate analysis reports during training
- `depth_visualization.py` — Depth estimation monitoring callbacks:
  - `DepthPredictionGridCallback` — RGB | GT depth | predicted depth comparison grids
  - `DepthMetricsCurveCallback` — Training/validation metric curve plots (loss, AbsRel, delta)

## Conventions

- `__init__.py` is empty — import directly from modules
- Callbacks follow the standard Keras `keras.callbacks.Callback` interface
- Integrates with the `analyzer` package for model diagnostics during training
- Visualization callbacks lazy-import matplotlib — callers should set `MPLBACKEND=Agg` for headless environments
