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
| `.get_summary_statistics()` | dict keyed by analysis **category** — `n_models`, `n_multi_input_models`, `multi_input_models`, `analyses_performed`, `model_performance`, `calibration_summary`, `confidence_summary`, `weight_summary`, `training_summary`, `spectral_summary`, `spectral_summary_per_model` — with per-model sub-dicts one level down. `summary[model_name]` does **not** exist; `summary['calibration_summary'][model_name]` does |
| `with ModelAnalyzer(...) as analyzer:` | on exit, calls `config.restore_plotting_style()`; the only shipped caller of it. Without the `with`, the global matplotlib style stays applied |
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
| `weight_layer_types` | `None` (all) | e.g. `['Dense', 'Conv2D']` |
| `analyze_biases` | `False` | |
| `compute_weight_pca` | `True` | |
| `calibration_bins` | `10` | ECE bin count |
| `output_activation` | `None` (infer) | `'softmax'` / `'sigmoid'` / `'logits'`; pins what the head emits instead of inferring it |
| `smooth_training_curves` / `smoothing_window` | `True` / `5` | |
| `spectral_min_evals` / `spectral_max_evals` | `10` / `15000` | layers outside this eigenvalue range are skipped; above the cap the analyzer switches to truncated SVD |
| `spectral_bootstraps` | `50` | `pl_pvalue` resolution; `0` skips the test (it is ~100× the cost of the alpha fit) |
| `spectral_concentration_analysis` | `True` | Gini / dominance / participation ratio |
| `spectral_randomize` | `False` | randomized-weight comparison (slow) |
| `spectral_n_randomizations` | `5` | independent permutations averaged per layer when `spectral_randomize` is on |
| `spectral_glorot_fix` | `False` | |
| `spectral_per_layer_diagnostics` | `False` | one power-law fit plot per layer into `spectral_plots/` |
| `plot_style` | `'publication'` | also `'presentation'`, `'draft'` |
| `color_palette` / `fig_width` / `fig_height` / `dpi` | `'deep'` / 12 / 8 / 300 | |
| `save_plots` / `save_format` | `True` / `'png'` | |
| `json_include_per_sample_data` | `False` | per-sample confidence/entropy arrays — bulky |
| `json_include_raw_esds` | `False` | emit `spectral_esds` / `spectral_rand_esds` (the full per-layer eigenvalue spectra) into the artifact — very bulky |
| `max_layers_heatmap` / `max_layers_info_flow` | `12` / `8` | plot truncation |
| `pareto_analysis_threshold` | `2` | minimum models for Pareto plots |
| `memory_limit_mb` | `2048` | budget for the activations `InformationFlowAnalyzer` holds at once; `None` is explicitly unbounded. The resulting batch is shared by every model in the run (the minimum over them) — see the information-flow section for why |
| `random_state` | `None` | seeds every stochastic site (data subsampling, spectral randomization, the goodness-of-fit bootstrap, power iteration); `None` is unseeded and NOT reproducible |
| `verbose` | `True` | |

`AnalysisConfig.setup_plotting_style()` is called in `ModelAnalyzer.__init__` and forces
`matplotlib.use('Agg')` plus global `rcParams`. It mutates process-global matplotlib state, and
that mutation **outlives the analyzer**.

`AnalysisConfig.restore_plotting_style()` puts the captured `rcParams` back (the backend is
deliberately not restored — `Agg` is a repo-wide headless requirement). It runs on exactly one
shipped path: `ModelAnalyzer.__exit__`. Use the analyzer as a context manager if the surrounding
process plots afterwards —

```python
with ModelAnalyzer(models={"m": model}, config=AnalysisConfig()) as analyzer:
    results = analyzer.analyze(data)
```

— or call `config.restore_plotting_style()` yourself. A plain `ModelAnalyzer(...)` still leaves the
style applied for the rest of the process; that is deliberate, because every visualizer reads the
global state and `create_pareto_analysis` returns a `Figure` the caller may save after the call.

## Output

Written into `output_dir`:

| File | Produced by |
|---|---|
| `analysis_results.json` | always |
| `summary_dashboard.png` | always |
| `weight_learning_journey.png` | `analyze_weights` |
| `confidence_calibration_analysis.png` | `analyze_calibration` |
| `information_flow_analysis.png` | `analyze_information_flow` |
| `training_dynamics.png` | `analyze_training_dynamics` (needs `training_history`) |
| `spectral_summary.png`, `spectral_funnel_diagram.png` | `analyze_spectral` |
| `spectral_plots/*.png` | `spectral_per_layer_diagnostics=True` |
| `trap_plots/*.png` | correlation-trap detection (needs `spectral_randomize=True`) |
| `pareto_analysis.png` | `create_pareto_analysis()` |

Everything above is written only when `config.save_plots` is `True`. A figure that fails to render
is **not** an exception: `_save_figure` catches it and logs
`ERROR ... Could not save figure <name>: <reason>`, so the run completes with that one file
missing. Check the log, not the exit status, before concluding a figure is unsupported. One such
failure was real and is fixed: with exactly two models the weight-PCA similarity panel had a zero
second-component spread, `set_aspect('equal', adjustable='box')` collapsed the axes box to zero
height, and `savefig` raised `LinAlgError: Singular matrix` — losing the whole dashboard, not just
that panel (see `D-033`).

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

Activations are captured by temporarily assigning a recording wrapper to `layer.call` on each
selected layer *instance*, running one eager `model(x_sample, training=False)` pass, and restoring
every wrapper in a `finally` block, so the model is left exactly as it was handed in
(`analyzers/information_flow_analyzer.py`). Two properties of that mechanism are load-bearing: the
pass must be **eager** (under `model.predict(...)` Keras traces the forward function and the wrapper
is handed a `SymbolicTensor`, so nothing concrete is captured), and no functional feature-extractor
sub-model is sliced (that needs `model.input` / `layer.output`, which a subclassed model does not
have).

Per layer: `layer_type`, `output_shape`, `mean_activation`, `std_activation`, `sparsity`,
`positive_ratio`, `effective_rank`, `capture_index`.

- `results.information_flow[model][layer]` is keyed in **forward-pass invocation order**, recorded
  at capture time; `capture_index` carries that position so the order survives a dict rebuild. The
  static layer walk is *not* depth order — sort on `capture_index`, never on iteration order.
- The capture batch is `min(config.n_samples, 200)` samples, further capped so the retained
  activations fit `config.memory_limit_mb` (default 2048; `None` is unbounded). A capped run logs a
  warning naming the old and new batch size.
- **All models in one run share ONE capture batch size — the minimum over them** — and it is
  reported as `results.information_flow_batch_size` (also written into the saved JSON).
  `effective_rank` is the singular-value entropy of a `(batch, features)` matrix and is therefore
  bounded above by `batch`, so a per-model batch would make the bigger model measure less
  information for a purely mechanical reason. MEASURED on a 64-filter and a 256-filter conv model
  at `memory_limit_mb=20`: sized per model they read `effective_rank` **195.05** (batch 200) and
  **77.38** (batch 79); at the shared batch of 79 they read **77.35** and **77.38**. Two runs are
  only comparable when this number matches.
- A weight-shared layer keeps the tensor from its **last** invocation, and its `capture_index` is
  that last position.

Reference command: `pytest tests/test_analyzer/test_analyzer_docs.py -k information_flow` runs the
analysis end to end and asserts both the populated dict and the written PNG, so this section cannot
go stale silently.

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
| `alpha_weighted` | `alpha * log10(λ_max)` on un-normalized eigenvalues — the **canonical** WeightWatcher name for this quantity (`METRICS.ALPHA_WEIGHTED`, "also called AlphaHat"); lower is better |
| `alpha_hat` | the SETOL papers' notation `α̂` for the same quantity. It is an **alias of `alpha_weighted`, not the other way round**, and it is not a WeightWatcher column name; the two columns are bit-identical (`(df['alpha_weighted'] == df['alpha_hat']).all()` is `True`, guarded by `test_analyzer_docs.py -k AlphaWeighted`) |
| `alpha_hat_normalized` | `alpha * log10(λ_max / N)`; a SETOL extra, not part of the WW metric set; comparable across differing layer widths |
| `learning_phase` | categorical from alpha (see the phase table) |
| `dominance_ratio` | `λ_max / sum(rest)`; > 1.0 = rank-1 spike |
| `xmin`, `D`, `sigma`, `num_pl_spikes` | fit threshold, KS distance, alpha standard error, tail size |
| `pl_pvalue` | bootstrap goodness-of-fit (Clauset et al. 2009); `-1.0` means the test did not run — read the caveats below before treating a low value as a rejection |
| `mp_softrank` | `λ_plus / λ_max`, where `λ_plus = σ²(1 + 1/√Q)²` is the **theoretical** Marchenko-Pastur edge computed by `calc_mp_edges` from the layer's own bulk variance. Values near 1 mean the whole spectrum sits inside its MP bulk (i.e. it looks random); small values mean the spectrum has grown well past the bulk. It is **not clamped**: a value above 1.0 means the entire spectrum lies inside its own theoretical edge. Measured on a two-model probe: `1.141264` and `0.708324` |

Contextual / expensive columns: `participation_ratio`, `min_participation_ratio`,
`concentration_score`, `erg_log_det`, `erg_delta_lambda_min`, `erg_satisfied`,
`rank_loss`, `weak_rank_loss`, `lambda_max`, `sv_max`, `sv_min`, `norm`, `spectral_norm`,
`log_norm`, `log_spectral_norm`, `log_alpha_norm`, `matrix_rank`, `num_evals`, `status`,
`warning`, `has_esd`, `alpha_unreliable`, `spectrum_truncated`. Identity columns: `model_name`,
`layer_id`, `name`, `layer_type`, `num_params`, `N`, `M`, `Q`, `rf`.

- `alpha_unreliable` is `True` when the fitted `alpha` exceeds `SPECTRAL_ALPHA_SANITY_MAX = 8.0`
  (WeightWatcher's unreliable-fit bound). The value is **flagged, never clamped** — a runaway alpha
  stays visible instead of being rewritten into a plausible "under-trained" label.
- `spectrum_truncated` is `True` when the SVD returned fewer singular values than `min(N, M)`. On
  that path `sv_min`, `rank_loss`, `weak_rank_loss`, `entropy` and `matrix_rank` are `NaN`, not `0` —
  a complete healthy spectrum also produces zeros, so zeros could not be told apart from truncation.
- `gini_coefficient`, `dominance_ratio`, `participation_ratio`, `min_participation_ratio`,
  `concentration_score` and `critical_weight_count` require `spectral_concentration_analysis=True`
  (the default), and are absent from the frame when it is off.
- `spectral_randomize=True` adds `has_trap`, `num_rand_spikes`, `trap_severity`,
  `trap_severity_label`, `trap_threshold`, `mp_lambda_minus`, `mp_lambda_plus`, `rand_sv_max`,
  `rand_sv_ratio` and `rand_distance`. Each is the mean over `spectral_n_randomizations`
  independent permutations, so `num_rand_spikes` can be fractional; `has_trap` is a majority vote
  over those draws, not an `any()`.

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

- **`model_performance[m]['accuracy']` is `None` when the model has no accuracy metric, never
  `0.0`.** Keras 3 reports the aggregated compiled metrics under the single name
  `compile_metrics`, so the accuracy is resolved from `get_metrics_result()` and by metric CLASS
  (`SparseCategoricalAccuracy` and friends), which also finds a metric you renamed. Before this,
  every normally-compiled model reported `accuracy: 0.0` at `status: 'success'` — a value a real
  classifier can produce, so it was indistinguishable from a real score. A failed evaluation
  reports `None` too; read `status` before reading the number. `results.model_metrics[m]` keeps
  the raw `compile_metrics` key as well.
- **`training_history` takes the `.history` dict, not the Keras `History` object.** Its keys must
  match the `models` keys exactly, or training dynamics silently produces nothing.
- **`pl_pvalue` semantics, and two known divergences from Clauset et al. 2009.**
  `-1.0` (`SPECTRAL_PVALUE_NOT_COMPUTED`) means the test **did not run** and is not a rejection:
  it is returned when the tail above `xmin` has fewer than 5 points, when the observed KS distance
  is unavailable, and when no bootstrap draw produced a successful fit. Before the sentinel was
  wired through, a tail shorter than the fitter's 10-point floor was reported as exactly `0.0` —
  indistinguishable from "certainly not a power law" — on a measured 30% of layers. Two
  asymmetries remain, both of which push the p-value **downward**, i.e. towards falsely rejecting
  the power law:
  1. the observed KS distance is the one from the reported fit (fixed `xmin`), while each synthetic
     bootstrap sample is refitted with a **free** `xmin` search, which minimises `D_syn`;
  2. bootstrap draws whose own fit fails are dropped from the numerator while the denominator stays
     `n_bootstraps`.
  One exit still returns a decisive `0.0` rather than the sentinel: `alpha <= 1.0` or `xmin <= 0`,
  which is a genuinely unusable fit rather than an uncomputable test.
- **Two deliberate divergences from WeightWatcher, in metrics this README frames as
  "WeightWatcher/SETOL".** Neither is a bug and neither will be "restored to parity".
  1. **`rank_loss` / `matrix_rank` take their tolerance from the weights' SOURCE dtype**, not from
     the float64 the SVD runs in. WeightWatcher casts `W.astype(float)` before the SVD
     (`weightwatcher.py:2822`, `:2861`) and so reads `np.finfo(float64).eps = 2.22e-16`; that cast
     is an artifact of scipy compatibility, not a decision about precision. MEASURED on an 80x60
     **float32** matrix of exact rank 40: WW's tolerance reports `rank_loss = 0` while the 20
     surplus singular values sit at 1.3e-6..6.9e-6, i.e. float32 round-off of an exact zero, three
     orders of magnitude above the float64 tolerance. For a float64 source the two agree
     bit-for-bit.
  2. **The Glorot normalization (`spectral_glorot_fix`) is restated in this package's flattened
     conv dimensions**, because a conv kernel is matricized to ONE `(kh*kw*in_c, out_c)` matrix
     here while WeightWatcher decomposes it per receptive-field position. `kappa` is
     `sqrt(2/(fan_in+fan_out))` with Keras' own fans (`fan_in = kh*kw*in_c`,
     `fan_out = kh*kw*out_c`); WW's `(N+M)*rf` is correct only for ITS per-slice `N`/`M`. Copying
     WW's spelling here double-counts `kh*kw` — measured 0.0177667 against the true 0.0340207 on a
     `(3,3,64,128)` kernel.
- **Do not trust `alpha` when** `pl_pvalue < 0.1` (the ESD probably is not a power law),
  `sigma > alpha / 3` (the CI spans several phases), `num_pl_spikes < 50` (MLE variance too
  high), or the layer exceeded `spectral_max_evals` (truncated SVD biases alpha upward).
- `lambda_max` and every norm-based column (`norm`, `spectral_norm`, `log_norm`,
  `log_spectral_norm`, `log_alpha_norm`) are **not comparable across architectures**; `alpha` is
  roughly scale-invariant. `alpha_weighted` / `alpha_hat` are the exception WeightWatcher makes:
  it reports α̂ as "suitable for DNNs of differing hyperparameters and depths simultaneously" —
  but that claim is for the **layer-averaged** α̂, `(1/L) Σ_l α_l log10 λ_l^max`, while this
  DataFrame carries the **per-layer** term. Average the column over a model's layers before
  comparing two architectures; a single row is a within-model ranking quantity.
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
