# `dl_techniques.analyzer`

Post-training diagnostics for trained Keras models. Point it at one or more models (plus optional
test data and training histories) and it produces weight-health, calibration, information-flow,
training-dynamics and spectral (WeightWatcher/SETOL) metrics, writes a set of figures, and dumps
everything to `analysis_results.json`.

Nothing here changes a model — except `create_smoothed_model`, which returns a new one.

## Quick start

```python
from dl_techniques.analyzer import ModelAnalyzer, AnalysisConfig, DataInput

analyzer = ModelAnalyzer(
    models={"ResNet_v1": model_a, "ConvNext_v2": model_b},   # keys become plot labels
    config=AnalysisConfig(analyze_spectral=True, plot_style="publication"),
    output_dir="results/my_analysis",
    training_history={                                        # optional
        "ResNet_v1": history_a.history,                       # the .history DICT, not the object
        "ConvNext_v2": history_b.history,
    },
)

results = analyzer.analyze(DataInput(x_data=x_test, y_data=y_test))

print(results.spectral_analysis[["name", "alpha", "learning_phase", "stable_rank"]])
print(analyzer.get_summary_statistics())
```

`analyze(data=None)` is legal — it just skips calibration and information flow.

## API

| Symbol | Notes |
|---|---|
| `ModelAnalyzer(models, config=None, output_dir=None, training_history=None)` | `models` must be non-empty (`ValueError` otherwise). `output_dir=None` creates `analysis_<YYYYmmdd_HHMMSS>/` in the cwd. |
| `.analyze(data=None, analysis_types=None)` | returns `AnalysisResults`; also writes figures and `analysis_results.json` |
| `.get_summary_statistics()` | dict of headline numbers per model |
| `.create_pareto_analysis(save_plot=True)` | needs ≥ `config.pareto_analysis_threshold` (2) models |
| `.create_smoothed_model(model_name, method='detX', percent=0.8, save_path=None)` | SVD truncation; requires a prior spectral run |
| `.save_results(filename='analysis_results.json')` | called automatically by `analyze()` |
| `AnalysisConfig` | dataclass of toggles, see below |
| `DataInput(x_data, y_data)` | `NamedTuple`; `.from_tuple((x, y))` and `.from_object(obj)` helpers |
| `AnalysisResults`, `TrainingMetrics` | result containers |
| `LayerType`, `SmoothingMethod`, `StatusCode`, `MetricNames` | enums / name constants |

`analysis_types` is a set drawn from `{'weights', 'calibration', 'information_flow',
'training_dynamics', 'spectral'}`. `'calibration'` and `'information_flow'` require `data` —
asking for them without it raises `ValueError`. When `analysis_types` is omitted the set comes
from the `analyze_*` flags on the config.

`analyze()` accepts a `DataInput`, a plain `(x, y)` tuple, or any object with `x_data` / `y_data`
attributes.

## `AnalysisConfig`

| Field | Default | Notes |
|---|---|---|
| `analyze_weights` / `analyze_calibration` / `analyze_information_flow` / `analyze_training_dynamics` / `analyze_spectral` | all `True` | master toggles |
| `n_samples` | `1000` | cap on samples for data-dependent analyses |
| `activation_layer_name` / `activation_layer_index` | `None` | pin the layer used for activation stats |
| `weight_layer_types` | `None` (all) | e.g. `['Dense', 'Conv2D']` |
| `analyze_biases` | `False` | |
| `compute_weight_pca` | `True` | |
| `calibration_bins` | `10` | ECE bin count |
| `smooth_training_curves` / `smoothing_window` | `True` / `5` | |
| `spectral_min_evals` / `spectral_max_evals` | `10` / `15000` | layers outside this eigenvalue range are skipped; above the cap the analyzer switches to truncated SVD |
| `spectral_bootstraps` | `50` | `pl_pvalue` resolution; `0` skips the test (it is ~100× the cost of the alpha fit) |
| `spectral_concentration_analysis` | `True` | Gini / dominance / participation ratio |
| `spectral_randomize` | `False` | randomized-weight comparison (slow) |
| `spectral_glorot_fix` | `False` | |
| `spectral_per_layer_diagnostics` | `False` | one power-law fit plot per layer into `spectral_plots/` |
| `plot_style` | `'publication'` | also `'presentation'`, `'draft'` |
| `color_palette` / `fig_width` / `fig_height` / `dpi` | `'deep'` / 12 / 8 / 300 | |
| `save_plots` / `save_format` | `True` / `'png'` | |
| `json_include_per_sample_data` | `False` | per-sample confidence/entropy arrays — bulky |
| `json_include_raw_esds` | `False` | raw eigenvalue arrays — very bulky |
| `max_layers_heatmap` / `max_layers_info_flow` | `12` / `8` | plot truncation |
| `pareto_analysis_threshold` | `2` | minimum models for Pareto plots |
| `verbose` | `True` | |

`AnalysisConfig.setup_plotting_style()` is called in `ModelAnalyzer.__init__` and forces
`matplotlib.use('Agg')` plus global `rcParams`. It mutates process-global matplotlib state.

## Output

Written into `output_dir`:

| File | Produced by |
|---|---|
| `analysis_results.json` | always |
| `summary_dashboard.png` | always |
| `weight_learning_journey.png` | `analyze_weights` |
| `confidence_calibration_analysis.png` | `analyze_calibration` |
| `information_flow_analysis.png` | `analyze_information_flow` — **currently never produced, see below** |
| `training_dynamics.png` | `analyze_training_dynamics` (needs `training_history`) |
| `spectral_summary.png`, `spectral_funnel_diagram.png` | `analyze_spectral` |
| `spectral_plots/*.png` | `spectral_per_layer_diagnostics=True` |
| `trap_plots/*.png` | correlation-trap detection (needs `spectral_randomize=True`) |
| `pareto_analysis.png` | `create_pareto_analysis()` |

## Metrics

### Calibration (`results.calibration_metrics`, `results.confidence_metrics`)

| Key | Meaning | Ideal |
|---|---|---|
| `ece` | expected calibration error over `calibration_bins` bins | 0 |
| `brier_score` | MSE between predicted probabilities and one-hot labels | 0 |
| `per_class_ece` | ECE per class | 0 |
| `mean_entropy`, `std_entropy`, `median_entropy`, `min_entropy`, `max_entropy` | Shannon entropy of the predictive distribution (in `results.confidence_metrics`) | context-dependent |
| `max_probability`, `margin`, `gini_coefficient` | per-sample confidence arrays (in `results.confidence_metrics`) | — |

`mean_confidence` is not stored on the results object; it is computed in
`get_summary_statistics()`.

### Information flow (`results.information_flow`, per layer)

> **Broken.** `information_flow_analyzer.py` captures activations with
> `layer.register_forward_hook(...)`, which is a PyTorch API — `keras.layers.Layer` has no such
> method. Every model raises `AttributeError`, the analyzer catches and logs it, and
> `results.information_flow` comes back empty with no
> `information_flow_analysis.png`. Set `analyze_information_flow=False` until it is fixed.

The metrics it is meant to produce, per layer: `mean_activation`, `std_activation`, `sparsity`,
`positive_ratio`, `effective_rank`.

### Training dynamics (`results.training_metrics`)

| Field | Definition |
|---|---|
| `epochs_to_convergence` | first epoch reaching 95% of peak validation performance (`CONVERGENCE_THRESHOLD = 0.95`) |
| `training_stability_score` | std of validation loss over the last 10 epochs (`TRAINING_STABILITY_WINDOW`) |
| `overfitting_index` | mean (val loss − train loss) over the final 33% of training (`OVERFITTING_ANALYSIS_FRACTION`) |
| `peak_performance` | best validation metric and the epoch it happened |
| `final_gap` | val loss − train loss at the last epoch |

### Spectral (`results.spectral_analysis`, a `pandas.DataFrame`, one row per layer)

Always-reliable columns (no fitting, no distributional assumption — read these first):

| Column | Meaning |
|---|---|
| `stable_rank` | `sum(λ) / max(λ)`; divide by `min(M, N)` for capacity utilisation |
| `entropy` | normalised Shannon entropy of the eigenvalue distribution, `[0, 1]`; near 0 = rank collapse |
| `gini_coefficient` | eigenvalue inequality; > 0.8 = extreme concentration |

Fit-dependent columns (verify before trusting — see below):

| Column | Meaning |
|---|---|
| `alpha` | power-law exponent of the ESD tail; the primary training-quality indicator |
| `alpha_hat` | `alpha * log10(λ_max)`; lower is better, for **within-model** layer ranking only |
| `alpha_hat_normalized` | `alpha * log10(λ_max / N)`; comparable across differing layer widths |
| `alpha_weighted` | deprecated alias of `alpha_hat` |
| `learning_phase` | categorical from alpha (see the phase table) |
| `dominance_ratio` | `λ_max / sum(rest)`; > 1.0 = rank-1 spike |
| `xmin`, `D`, `sigma`, `num_pl_spikes` | fit threshold, KS distance, alpha standard error, tail size |
| `pl_pvalue` | bootstrap goodness-of-fit; `-1.0` means the test did not run |

Contextual / expensive columns: `participation_ratio`, `min_participation_ratio`,
`concentration_score`, `erg_log_det`, `erg_delta_lambda_min`, `erg_satisfied`, `mp_softrank`,
`rank_loss`, `weak_rank_loss`, `lambda_max`, `sv_max`, `sv_min`, `norm`, `spectral_norm`,
`log_norm`, `log_spectral_norm`, `log_alpha_norm`, `matrix_rank`, `num_evals`, `status`,
`warning`, `has_esd`. Identity columns: `model_name`, `layer_id`, `name`, `layer_type`,
`num_params`, `N`, `M`, `Q`, `rf`. With `spectral_randomize=True` you also get `has_trap`,
`num_rand_spikes`, `trap_severity`; with `spectral_concentration_analysis=True`,
`critical_weight_count`.

Alpha phases:

| `alpha` | Phase | Reading |
|---|---|---|
| `< 0` | failed | fit did not converge — use `stable_rank` / `entropy` instead |
| `[1.0, 2.0)` | over-regularized | correlation traps / rank-1 spikes; lower the LR, raise the batch size |
| `[2.0, 2.5)` | ideal | SETOL critical point |
| `[2.5, 4.0)` | good | normal SOTA working range |
| `(4.0, 6.0]` | fair | train longer, reduce regularization |
| `> 6.0` | under-trained | nearly random; check the layer is receiving gradients |

The full theory, including the ERG condition and the funnel diagnostic, is in `SETOL.md`;
correlation traps are in `CORRELATION_TRAPS.md`.

## Gotchas

- **`training_history` takes the `.history` dict, not the Keras `History` object.** Its keys must
  match the `models` keys exactly, or training dynamics silently produces nothing.
- **Do not trust `alpha` when** `pl_pvalue < 0.1` (the ESD probably is not a power law),
  `sigma > alpha / 3` (the CI spans several phases), `num_pl_spikes < 50` (MLE variance too
  high), or the layer exceeded `spectral_max_evals` (truncated SVD biases alpha upward).
- `alpha_hat`, `lambda_max` and every norm-based column are **not comparable across
  architectures**. Only `alpha` is roughly scale-invariant.
- Conv2D kernels are matricized `(kh, kw, in_c, out_c) -> (kh*kw*in_c, out_c)`, which destroys
  spatial structure. Spectral metrics describe the linear map, not the convolution.
- `concentration_score` has no absolute scale — use it only to rank layers within one model.
- Layers with fewer than `spectral_min_evals` eigenvalues never appear in the DataFrame at all.
- Multi-input models are detected and warned about; calibration and information flow are
  limited for them.
- Spectral analysis is the expensive part: an SVD per layer, ×`spectral_bootstraps` if
  `pl_pvalue` is on, ×2 if `spectral_randomize` is on. Turn `analyze_spectral` off for quick runs.
- `AnalysisResults` holds numpy arrays and a DataFrame; JSON serialization drops them unless the
  `json_include_*` flags are set.

## See also

- `CLAUDE.md` — module map and authoring rules.
- `SETOL.md`, `CORRELATION_TRAPS.md` — the theory behind the spectral metrics.
- Tests: `tests/test_analyzer/`.
