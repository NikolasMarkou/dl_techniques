"""
Configuration for Model Analyzer

Configuration classes and plotting setup utilities.
"""

import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, List, Sequence, Tuple

# ---------------------------------------------------------------------

@dataclass
class AnalysisConfig:
    """Configuration for all analysis types."""

    # Analysis toggles
    analyze_weights: bool = True
    analyze_calibration: bool = True
    analyze_information_flow: bool = True
    analyze_training_dynamics: bool = True
    analyze_spectral: bool = True
    # OFF by default: the Fisher pass runs one eager forward and backward per sample.
    analyze_fisher: bool = False

    # Sampling parameters
    n_samples: int = 1000

    # Weight analysis options
    weight_layer_types: Optional[List[str]] = None
    analyze_biases: bool = False
    compute_weight_pca: bool = True

    # Calibration options
    calibration_bins: int = 10
    # DECISION plan-2026-10-05-analyzer-audit/F-044
    # Bin count for the PUBLISHED `per_class_ece` / `per_class_conditional_top1_ece`.
    # It used to be computed inline as `max(2, calibration_bins // 2)`, so at the
    # defaults `ece` was a 10-bin ECE and `per_class_ece` a 5-bin ECE with nothing at
    # the public surface saying so — a reader comparing the two was comparing different
    # estimators. `None` keeps the halving; an int overrides it outright.
    # Do NOT set this equal to `calibration_bins` by default: a per-class column holds
    # far less mass than the pooled top-1 score, so it needs coarser bins to be stable.
    per_class_calibration_bins: Optional[int] = None
    # DECISION plan-2026-09-01T225724-e79ad4bd/D-032
    # What the models' final layer emits. `None` infers it per model. Set it when
    # the inference is wrong or when you want the artifact to record the choice.
    # 'logits' is softmaxed before calibration; 'softmax' and 'sigmoid' are used
    # as-is. See decisions.md D-032.
    output_activation: Optional[Literal['softmax', 'sigmoid', 'logits']] = None

    # Training analysis options
    smooth_training_curves: bool = True
    smoothing_window: int = 5

    # Spectral Analysis (WeightWatcher) options
    spectral_min_evals: int = 10
    spectral_max_evals: int = 15000
    spectral_glorot_fix: bool = False
    spectral_randomize: bool = False
    # Number of independent element-wise permutations averaged per layer when
    # `spectral_randomize` is on. A single draw made every correlation-trap verdict
    # a coin flip; >= 5 is required (plan-2026-09-01T225724-e79ad4bd D-017).
    spectral_n_randomizations: int = 5
    spectral_concentration_analysis: bool = True

    # NEW FLAG: Controls generation of individual layer plot files
    spectral_per_layer_diagnostics: bool = False
    spectral_bootstraps: int = 50

    # Fisher analysis options (arXiv:2609.36262). Every field below is read by
    # `FisherConfig.from_analysis_config`; a meta-test pins that none goes unread, because
    # an unread knob advertises behaviour the code does not have (F-028 / D-029).
    # `None` infers: sequence model iff integer (B, L) input and (B, L, V) output.
    fisher_force_sequence: Optional[bool] = None
    # N_F: inputs the Fisher estimate averages over. 100 is the diagnostic default; the
    # paper's publication runs use 800.
    fisher_n_samples: int = 100
    # G: responses per prompt in the gradient arm (>= 2 for the group-centered signals).
    fisher_n_responses: int = 1
    # N_g: prompts of the gradient arm, disjoint from the Fisher prompts.
    fisher_n_gradient_prompts: int = 100
    fisher_compute_gradients: bool = False
    # "rl" | "sft" | "opd_pg" | "opd_full" | "none". The callables each needs travel in a
    # `FisherSignal` passed to `ModelAnalyzer`, never here: `save_results` serializes every
    # public field of this class, and a callable would lose the whole artifact.
    fisher_signal_type: str = "rl"
    fisher_token_level: bool = True
    fisher_max_tokens: int = 512
    fisher_n_positions_per_response: int = 8
    fisher_n_token_pairs_per_position: int = 8
    # Sampling microbatch only; scoring is always one sample per backward pass.
    fisher_batch_size: int = 8
    fisher_accumulate_dtype: str = "float32"
    fisher_sampling_temperature: float = 1.0
    # sigma^2 of a Gaussian likelihood. Setting it selects the regression output kind;
    # the analyzer refuses to guess a variance.
    fisher_regression_variance: Optional[float] = None
    # "response" | "response_plus_prompt" (sequence models).
    fisher_score_scope: str = "response"
    # The q grid (percent) for recall and retention. None keeps (0.05, 0.1, 0.5, 1, 5).
    fisher_selection_percentages: Optional[Tuple[float, ...]] = None
    # Coordinates with Fisher at or below this are "Fisher-dead": counted, and excluded
    # from recall and retention.
    fisher_zero_threshold: float = 1e-12
    fisher_sft_chi2_cap: float = 1e4
    fisher_assert_no_weight_drift: bool = True
    # Ordered model names: the first is the initial policy, the rest are later checkpoints
    # of the SAME architecture. None skips the persistence arm.
    fisher_persistence_indices: Optional[Sequence[str]] = None
    # When True, `save_results` also emits the per-parameter Fisher table and the
    # parameter-sized token-level arrays. Kept False: one row per parameter.
    json_include_fisher_raw: bool = False

    # Visualization settings
    plot_style: str = 'publication'
    color_palette: str = 'deep'
    fig_width: int = 12
    fig_height: int = 8
    dpi: int = 300
    save_plots: bool = True
    save_format: str = 'png'

    # Advanced options
    verbose: bool = True
    # DECISION plan-2026-10-05-analyzer-audit/F-086
    # Seed for every stochastic site the ANALYZER actually reaches: the `DataSampler`
    # draw, the spectral randomization permutations, and the goodness-of-fit bootstrap.
    # `None` keeps the historical unseeded behaviour. See decisions.md D-031.
    #
    # This list used to end with `_power_iteration`, which was FALSE. `_power_iteration`
    # is reachable only via `get_top_eigenvectors(method='power_iteration', rng=...)`,
    # and the analyzer's single call site (`summarize_critical_weights`, reached from
    # `_describe_model`) uses the default `method='direct'` and passes no `rng`. Nothing
    # in `src/` threads this field into the power-iteration path, so naming it here told
    # a reader that a `random_state` made a power-iteration result reproducible when no
    # configuration could reach one. Do NOT re-add it unless a call site is wired up.
    random_state: Optional[int] = None

    # JSON serialization options
    json_include_per_sample_data: bool = False  # Set to False to exclude bulky per-sample arrays (e.g., confidence, entropy)
    # When True, `save_results` emits `spectral_esds` (and `spectral_rand_esds`
    # when randomization ran) into the artifact. Kept False by default: one entry
    # per analyzed layer, each the full eigenvalue spectrum.
    json_include_raw_esds: bool = False

    # Configurable visualization parameters (addressing hardcoded values)
    max_layers_heatmap: int = 12  # Maximum layers to show in weight health heatmap
    max_layers_info_flow: int = 8  # Maximum layers to show in information flow analysis
    pareto_analysis_threshold: int = 2  # Minimum models needed for Pareto analysis

    # Performance settings
    # DECISION plan-2026-09-01T225724-e79ad4bd/D-022
    # Budget for the activations `InformationFlowAnalyzer` holds in memory at once.
    # This defaults to a real number, NOT None: the analyzer retains every wrapped
    # layer's output simultaneously, which is an analytic 16.94 GB for ResNet50 at
    # B=200. `None` means explicitly unbounded and is honoured as such.
    # See decisions.md D-022.
    memory_limit_mb: Optional[int] = 2048

    # DECISION plan-2026-09-01T225724-e79ad4bd/D-029
    # `setup_plotting_style` mutates process-global matplotlib state and stashes the
    # pre-existing rcParams here. It is a DECLARED field, not an attribute conjured
    # by that method: an undeclared attribute is absent from `fields()`/`asdict()`
    # and forced `save_results` to filter it out BY STRING NAME. Do NOT un-declare
    # it, and do NOT serialize it - it is private and holds an unJSONable RcParams.
    # See decisions.md D-029.
    _original_rcParams: Optional[Dict[str, Any]] = field(default=None, repr=False)

    def get_figure_size(self, scale: float = 1.0) -> Tuple[float, float]:
        """Get figure size with optional scaling."""
        return (self.fig_width * scale, self.fig_height * scale)

    def setup_plotting_style(self) -> None:
        """Set up matplotlib style based on configuration."""
        # Use non-interactive backend for file-based rendering.
        # This MUST stay above the snapshot below - see
        # `restore_plotting_style` for why the ordering is load-bearing.
        matplotlib.use('Agg')

        # Save current rcParams to restore later if needed
        self._original_rcParams = plt.rcParams.copy()

        plt.style.use('default')

        # Apply style presets
        style_settings = {
            'publication': {
                'font.size': 10,
                'axes.titlesize': 12,
                'axes.labelsize': 11,
                'xtick.labelsize': 9,
                'ytick.labelsize': 9,
                'legend.fontsize': 9,
                'figure.titlesize': 14,
                'lines.linewidth': 2,
                'lines.markersize': 6,
                'axes.linewidth': 1,
                'grid.alpha': 0.3,
            },
            'presentation': {
                'font.size': 14,
                'axes.titlesize': 18,
                'axes.labelsize': 16,
                'xtick.labelsize': 14,
                'ytick.labelsize': 14,
                'legend.fontsize': 14,
                'figure.titlesize': 20,
                'lines.linewidth': 3,
                'lines.markersize': 10,
                'axes.linewidth': 2,
                'grid.alpha': 0.4,
            },
            'draft': {
                'font.size': 12,
                'axes.titlesize': 14,
                'axes.labelsize': 12,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'legend.fontsize': 11,
                'figure.titlesize': 16,
                'lines.linewidth': 2.5,
                'lines.markersize': 8,
                'axes.linewidth': 1.5,
                'grid.alpha': 0.3,
            }
        }

        if self.plot_style in style_settings:
            plt.rcParams.update(style_settings[self.plot_style])

        plt.rcParams.update({
            'figure.figsize': (self.fig_width, self.fig_height),
            'figure.dpi': 100,
            'savefig.dpi': self.dpi,
            'axes.grid': True,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.axisbelow': True,
            'figure.autolayout': False,
        })

        sns.set_theme(style='whitegrid', palette=self.color_palette)

    def restore_plotting_style(self) -> None:
        """Restore the rcParams captured by the last ``setup_plotting_style`` call.

        ``setup_plotting_style`` mutates process-global matplotlib state, which
        outlives the ``ModelAnalyzer`` that triggered it. This puts it back.

        The BACKEND is deliberately not restored: ``Agg`` is a repo-wide headless
        requirement (pinned by
        ``tests/test_callbacks/test_the_matplotlib_backend_is_headless.py``), not
        part of this configuration's styling. The mechanism is the statement ORDER
        in :meth:`setup_plotting_style`, which calls ``matplotlib.use('Agg')``
        BEFORE snapshotting ``plt.rcParams`` - so the snapshot already carries
        ``Agg`` and this blanket ``update`` cannot put an interactive backend back.
        Do NOT "fix" that order to capture the true pre-state: it would silently
        un-headless the session. Pinned by
        ``TestTheConfigsPrivateStateIsDeclared::test_the_backend_is_deliberately_not_restored``.

        Does nothing if no style has been applied yet.
        """
        if self._original_rcParams is None:
            return
        plt.rcParams.update(self._original_rcParams)

# ---------------------------------------------------------------------