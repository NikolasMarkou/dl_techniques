# Analyzer Package

Comprehensive model analysis framework for evaluating trained Keras models. Provides weight health diagnostics, calibration metrics, information flow analysis, spectral analysis, and training dynamics insights.

## Architecture

- `model_analyzer.py` — Main `ModelAnalyzer` orchestrator that delegates to specialized analyzers
- `config.py` — `AnalysisConfig` dataclass controlling which analyses to run
- `data_types.py` — `DataInput`, `AnalysisResults`, `TrainingMetrics` data containers
- `constants.py` — Enums: `LayerType`, `SmoothingMethod`, `StatusCode`, `MetricNames`
- `calibration_metrics.py` — Brier score, ECE, reliability diagrams
- `spectral_metrics.py` / `spectral_utils.py` — Spectral weight analysis (WeightWatcher-style)
- `utils.py` — Shared helper functions

### Subpackages

- `analyzers/` — Specialized analyzer classes (all inherit from `base.py`):
  - `weight_analyzer.py`, `calibration_analyzer.py`, `information_flow_analyzer.py`, `spectral_analyzer.py`, `training_dynamics_analyzer.py`
- `visualizers/` — Corresponding visualization classes (all inherit from `base.py`):
  - `weight_visualizer.py`, `calibration_visualizer.py`, `information_flow_visualizer.py`, `spectral_visualizer.py`, `training_dynamics_visualizer.py`, `summary_visualizer.py`

## Public API

```python
from dl_techniques.analyzer import ModelAnalyzer, AnalysisConfig, DataInput
```

## Conventions

- Analyzers and visualizers follow a base class pattern in their respective subpackages
- All analysis is config-driven via `AnalysisConfig` flags
- Version tracked in `__init__.py` (`__version__ = '1.1.0'`)

## Testing

Tests in `tests/test_analyzer/`. Uses pytest fixtures with config dicts.
