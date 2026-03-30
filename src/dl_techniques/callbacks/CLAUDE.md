# Callbacks Package

Custom Keras callbacks for training instrumentation.

## Modules

- `analyzer_callback.py` — Keras callback that runs `ModelAnalyzer` at specified training intervals to generate analysis reports during training

## Conventions

- `__init__.py` is empty — import directly from modules
- Callbacks follow the standard Keras `keras.callbacks.Callback` interface
- Integrates with the `analyzer` package for model diagnostics during training
