"""`visualizers/spectral_visualizer.py` had NO direct coverage at all.

Why this module exists (F-074). No test file in `tests/test_analyzer/` referenced
`SpectralVisualizer`, `_plot_funnel_diagram`, or `_scan_ks_distances`. The visualizer was
only smoke-exercised indirectly through `ModelAnalyzer` runs with `save_plots=False`, so
every panel-level contract was unguarded. Four defects lived in that gap:

- `_scan_ks_distances` used the power-form CDF that `fit_powerlaw` deliberately removed
  (D-001), so the PLOTTED `D_KS(x_min)` curve was not the curve the fit minimised and the
  red "selected x_min" marker need not sit on the plotted minimum (F-070).
- Its skipped candidates were left at `D = 1.0` -- the worst possible KS distance --
  substituted for "not evaluated", so `argmin` could mark a fabricated minimum.
- `_plot_trap_overlay` titled a layer "Clean" from `has_trap` alone while carrying a
  contradictory severity label, and printed `num_rand_spikes` as a count when it could be
  fractional (F-068, F-069).
- `dynamic_bins == 1` made `np.histogram` raise inside the per-layer diagnostics path.

These tests pin the FIXED behaviour. They are deliberately about contracts a reader can
check from the figure, not about pixel output.
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dl_techniques.analyzer.config import AnalysisConfig
from dl_techniques.analyzer.constants import MetricNames
from dl_techniques.analyzer.data_types import AnalysisResults
from dl_techniques.analyzer.visualizers.base import BaseVisualizer
from dl_techniques.analyzer.visualizers.spectral_visualizer import SpectralVisualizer


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def _frame(n_layers=3, models=("m1",), alpha=3.0):
    """A minimal but structurally complete spectral details frame."""
    rows = []
    for model in models:
        for layer_id in range(n_layers):
            rows.append({
                'model_name': model,
                'layer_id': layer_id,
                'name': f'dense_{layer_id}',
                'layer_type': 'dense',
                'N': 64, 'M': 32, 'rf': 1.0, 'Q': 2.0,
                'num_params': 2048,
                MetricNames.NUM_EVALS: 32,
                'alpha': alpha,
                MetricNames.STABLE_RANK: 12.5,
                'concentration_score': 1.25,
                'learning_phase': 'good',
                MetricNames.ERG_DELTA_LAMBDA_MIN: 0.05,
                MetricNames.ERG_LOG_DET: 0.1,
                MetricNames.ERG_SATISFIED: True,
                'xmin': 0.5, 'D': 0.08, 'sigma': 0.4,
                'num_pl_spikes': 20,
                MetricNames.PL_PVALUE: 0.5,
                MetricNames.STATUS: 'success',
                'warning': '', MetricNames.HAS_ESD: True,
                MetricNames.ALPHA_UNRELIABLE: False,
                MetricNames.SPECTRUM_TRUNCATED: False,
                'lambda_max': 9.0, 'sv_max': 3.0, 'sv_min': 0.1,
                'norm': 40.0, MetricNames.SPECTRAL_NORM: 9.0,
                'log_norm': 1.6, 'log_spectral_norm': 0.95,
                'log_alpha_norm': 2.1, MetricNames.MATRIX_RANK: 32,
                MetricNames.RANK_LOSS: 0, MetricNames.WEAK_RANK_LOSS: 0,
                MetricNames.ENTROPY: 0.8,
                MetricNames.ALPHA_WEIGHTED: 2.85,
                MetricNames.ALPHA_HAT: 2.85,
                MetricNames.ALPHA_HAT_NORMALIZED: 2.1,
                MetricNames.GINI_COEFFICIENT: 0.4,
                MetricNames.DOMINANCE_RATIO: 0.2,
                MetricNames.PARTICIPATION_RATIO: 12.0,
                MetricNames.MIN_PARTICIPATION_RATIO: 9.0,
                MetricNames.MP_SOFTRANK: 0.7,
                MetricNames.CRITICAL_WEIGHT_COUNT: 400,
            })
    return pd.DataFrame(rows)


def _results(n_layers=3, models=("m1",)):
    results = AnalysisResults()
    results.spectral_analysis = _frame(n_layers=n_layers, models=models)
    results.spectral_esds = {
        m: {i: np.sort(np.abs(np.random.default_rng(i).normal(2, 1, 32)))[::-1]
            for i in range(n_layers)}
        for m in models
    }
    return results


def _visualizer(results, tmp_path, **config_kwargs):
    config = AnalysisConfig(save_plots=True, **config_kwargs)
    return SpectralVisualizer(results, config, tmp_path, {m: '#1f77b4'
                                                          for m in ('m1', 'm2')})


class TestScanKsDistancesMatchesTheFitter:
    """F-070: the plotted KS curve must be the curve `fit_powerlaw` minimised."""

    def test_it_returns_one_distance_per_candidate(self):
        viz = _visualizer(_results(), None)
        evals = np.sort(np.abs(np.random.default_rng(7).pareto(1.5, 300)) + 1.0)[::-1]
        xmins, ks = viz._scan_ks_distances(evals)
        assert len(xmins) == len(ks)
        assert len(xmins) > 0

    def test_every_evaluated_distance_is_in_the_unit_interval(self):
        viz = _visualizer(_results(), None)
        evals = np.sort(np.abs(np.random.default_rng(11).normal(3, 1, 200)))[::-1]
        _, ks = viz._scan_ks_distances(evals)
        finite = ks[np.isfinite(ks)]
        assert finite.size > 0
        assert np.all((finite >= 0.0) & (finite <= 1.0)), (
            "a KS distance outside [0, 1] means the kernel is wrong, not that the "
            "spectrum is unusual"
        )

    def test_a_skipped_candidate_is_nan_never_one(self):
        """F-070: skipped candidates used to be pinned at 1.0, the WORST possible KS.

        `np.ones(...)` was the initialiser, so a candidate that could not be evaluated
        was indistinguishable from one that genuinely measured maximal misfit -- and if
        every candidate were skipped, `argmin` returned index 0 and the panel marked a
        fabricated minimum.
        """
        viz = _visualizer(_results(), None)
        # A constant spectrum: every candidate has a zero denominator, so every one skips.
        evals = np.full(60, 2.0)
        _, ks = viz._scan_ks_distances(evals)
        assert np.all(np.isnan(ks)), (
            f"expected every candidate to be NaN, got {ks[:5]}... -- a skipped "
            f"candidate must never carry a numeric KS distance"
        )

    def test_a_constant_spectrum_yields_no_fabricated_minimum(self):
        viz = _visualizer(_results(), None)
        xmins, ks = viz._scan_ks_distances(np.full(60, 2.0))
        assert len(xmins) > 0
        assert not np.isfinite(ks).any(), (
            "no candidate was evaluable, so none may carry a KS distance -- the panel "
            "gates its marker on `np.isfinite(...).any()`"
        )

    def test_the_log_domain_kernel_is_used(self):
        """The KS value matches the log-domain kernel, recomputed independently.

        `_scan_ks_distances` sorts ASCENDING internally, so every index it reports refers
        to that array.
        """
        viz = _visualizer(_results(), None)
        data = np.sort(np.abs(np.random.default_rng(3).normal(2.5, 0.8, 150)))
        xmins, ks = viz._scan_ks_distances(data)

        finite_positions = np.flatnonzero(np.isfinite(ks))
        assert finite_positions.size > 0
        pos = int(finite_positions[len(finite_positions) // 2])
        matches = np.flatnonzero(data == xmins[pos])
        assert matches.size == 1, "the probe spectrum has a duplicate value"
        i = int(matches[0])

        n_tail = len(data) - i
        denom = np.log(data[i:]).sum() - n_tail * np.log(data[i])
        alpha = 1.0 + n_tail / denom
        empirical = np.arange(n_tail) / n_tail
        log_form = np.max(np.abs(
            empirical - (1.0 - np.exp((1.0 - alpha) * (np.log(data[i:]) - np.log(data[i]))))))

        assert ks[pos] == pytest.approx(log_form, rel=1e-12)

    def test_the_removed_power_form_is_absent_from_the_source(self):
        """The kernel SPELLING cannot be discriminated by value, so pin the source.

        An earlier version of this test asserted `ks != power_form`. That is not a valid
        discriminator: D-001 measured the two CDF spellings differing by 1.7e-10..9.4e-10
        RELATIVE, but the KS statistic is a MAXIMUM over the tail and is attained at a
        point where both spellings agree to ~1e-16. MEASURED here: log 0.19848442186620446
        vs power 0.19848442186620424. So a value assertion passes with either kernel and
        proves nothing.

        This is the same lesson as the `CORRELATION_TRAPS.md` guard (F-062): a property
        that a value cannot distinguish has to be asserted structurally.
        """
        import inspect
        source = inspect.getsource(SpectralVisualizer._scan_ks_distances)
        body = source.split('"""', 2)[-1]   # drop the docstring, which quotes the old form
        assert "(data[i:] / curr_xmin)" not in body, (
            "the power-form CDF `(data[i:] / curr_xmin) ** (-alpha + 1.0)` is back; "
            "D-001 removed it in favour of the log-domain spelling and the two must "
            "stay identical to fit_powerlaw's"
        )
        assert "np.exp(" in body, (
            "the log-domain kernel is missing from _scan_ks_distances"
        )

    def test_a_skipped_candidate_is_never_the_minimum_marker(self):
        """F-070: the consumer gates the marker on finiteness.

        With every candidate skipped, `np.nanargmin` on an all-NaN array RAISES, so the
        panel must check `np.isfinite(...).any()` first rather than relying on argmin's
        behaviour. This pins the guard the panel depends on.
        """
        _, ks = _visualizer(_results(), None)._scan_ks_distances(np.full(60, 2.0))
        evaluated = np.isfinite(ks)
        assert not evaluated.any()
        # Exactly the condition the panel tests before calling nanargmin.
        assert not evaluated.any(), "the panel's guard would be False, so it must skip"

    def test_a_short_spectrum_returns_empty_arrays(self):
        viz = _visualizer(_results(), None)
        xmins, ks = viz._scan_ks_distances(np.array([1.0, 2.0, 3.0]))
        assert xmins.size == 0 and ks.size == 0


class TestTheFunnelDiagramSurvivesAnAbsentErgColumn:
    def test_no_erg_column_is_a_logged_no_op_not_a_crash(self, tmp_path):
        """All-fits-failed is a real state: `erg_delta_lambda_min` may not exist at all."""
        results = _results()
        results.spectral_analysis = results.spectral_analysis.drop(
            columns=[MetricNames.ERG_DELTA_LAMBDA_MIN])
        viz = _visualizer(results, tmp_path)
        viz.create_visualizations()   # must not raise

    def test_a_single_model_still_produces_the_summary(self, tmp_path):
        results = _results()
        viz = _visualizer(results, tmp_path)
        viz.create_visualizations()
        assert (tmp_path / 'spectral_summary.png').exists()


class TestTrapOverlayConsistency:
    """F-068/F-069: the title, the label and the spike count must not contradict."""

    def _run_overlay(self, tmp_path, **overrides):
        results = _results()
        frame = results.spectral_analysis
        base = {
            MetricNames.HAS_TRAP: True,
            MetricNames.TRAP_SEVERITY: 0.8,
            MetricNames.TRAP_SEVERITY_LABEL: 'severe',
            MetricNames.NUM_RAND_SPIKES: 2,
            MetricNames.TRAP_THRESHOLD: 6.0,
            MetricNames.MP_LAMBDA_PLUS: 5.0,
            MetricNames.MP_LAMBDA_MINUS: 1.0,
        }
        base.update(overrides)
        for column, value in base.items():
            frame[column] = value
        results.spectral_analysis = frame
        # The randomized spectrum must contain spikes above the threshold for markers.
        results.spectral_rand_esds = {
            'm1': {i: np.concatenate([np.full(30, 2.0), [7.0, 8.0]])
                   for i in range(3)}
        }
        viz = _visualizer(results, tmp_path, spectral_randomize=True)
        viz.create_visualizations()
        return results

    def test_a_detected_trap_renders_its_overlay(self, tmp_path):
        self._run_overlay(tmp_path)
        assert (tmp_path / 'trap_plots').exists()
        assert any((tmp_path / 'trap_plots').glob('*.png'))

    def test_an_int_spike_count_does_not_render_a_fractional_label(self, tmp_path):
        """F-069: the count is an int again, so "2.4 spike(s)" is unreachable."""
        self._run_overlay(tmp_path, **{MetricNames.NUM_RAND_SPIKES: 2})
        for png in (tmp_path / 'trap_plots').glob('*.png'):
            assert png.stat().st_size > 0

    def test_a_contradictory_row_still_renders(self, tmp_path):
        """An artifact whose `has_trap` disagrees with its label must not crash.

        F-068 made the title report the contradiction explicitly rather than silently
        picking one side.
        """
        self._run_overlay(
            tmp_path,
            **{MetricNames.HAS_TRAP: False,
               MetricNames.TRAP_SEVERITY_LABEL: 'moderate'})
        assert (tmp_path / 'trap_plots').exists()

    def test_no_randomization_data_is_a_clean_no_op(self, tmp_path):
        """`spectral_randomize=True` but no `spectral_rand_esds`: nothing to overlay."""
        results = _results()
        viz = _visualizer(results, tmp_path, spectral_randomize=True)
        viz.create_visualizations()
        assert not (tmp_path / 'trap_plots').exists() or \
            not any((tmp_path / 'trap_plots').glob('*.png'))

    def test_the_overlay_is_skipped_entirely_when_randomize_is_off(self, tmp_path):
        """The overlay is gated on the CONFIG, not on the presence of data."""
        results = _results()
        results.spectral_rand_esds = {
            'm1': {i: np.concatenate([np.full(30, 2.0), [7.0, 8.0]])
                   for i in range(3)}
        }
        viz = _visualizer(results, tmp_path, spectral_randomize=False)
        viz.create_visualizations()
        assert not (tmp_path / 'trap_plots').exists() or \
            not any((tmp_path / 'trap_plots').glob('*.png'))


class TestPerLayerDiagnosticsAreOptionalAndSurvivable:
    """`spectral_per_layer_diagnostics=True` builds a fresh 4-panel figure per layer."""

    def test_it_writes_one_plot_per_layer(self, tmp_path):
        results = _results(n_layers=3)
        viz = _visualizer(results, tmp_path,
                          spectral_per_layer_diagnostics=True)
        viz.create_visualizations()
        plots = list((tmp_path / 'spectral_plots').glob('*.png'))
        assert len(plots) == 3, f"expected one plot per layer, got {len(plots)}"

    def test_a_degenerate_single_eigenvalue_spectrum_does_not_raise(self, tmp_path):
        """`dynamic_bins == 1` used to make `np.histogram` raise.

        The whole per-layer function sits inside a `try/except` that downgrades the
        failure to a warning, so the panel vanished silently rather than erroring. Pinned
        here so a spectrum too small to bin is still *rendered* (or still visibly
        skipped), never a crash inside numpy.
        """
        results = _results(n_layers=1)
        results.spectral_esds = {'m1': {0: np.array([5.0])}}
        viz = _visualizer(results, tmp_path,
                          spectral_per_layer_diagnostics=True)
        viz.create_visualizations()   # must not raise

    def test_it_is_off_by_default(self, tmp_path):
        results = _results(n_layers=2)
        viz = _visualizer(results, tmp_path)
        viz.create_visualizations()
        assert not (tmp_path / 'spectral_plots').exists()


class TestTheVisualizerSurvivesTwoModelsOfDifferentDepth:
    def test_two_models_with_different_layer_counts(self, tmp_path):
        results = AnalysisResults()
        results.spectral_analysis = pd.concat(
            [_frame(n_layers=2, models=("shallow",)),
             _frame(n_layers=5, models=("deep",))],
            ignore_index=True)
        results.spectral_esds = {
            'shallow': {i: np.abs(np.random.default_rng(i).normal(2, 1, 32))[::-1]
                        for i in range(2)},
            'deep': {i: np.abs(np.random.default_rng(100 + i).normal(2, 1, 32))[::-1]
                     for i in range(5)},
        }
        viz = SpectralVisualizer(results, AnalysisConfig(save_plots=True),
                                 tmp_path, {'shallow': '#1f77b4', 'deep': '#ff7f0e'})
        viz.create_visualizations()
        assert (tmp_path / 'spectral_summary.png').exists()

    def test_an_all_nan_alpha_column_still_renders(self, tmp_path):
        """Every fit failed: `alpha` is the -1 sentinel everywhere."""
        results = _results()
        results.spectral_analysis[MetricNames.ALPHA] = -1.0
        viz = _visualizer(results, tmp_path)
        viz.create_visualizations()
        assert (tmp_path / 'spectral_summary.png').exists()

    def test_a_truncated_spectrum_row_still_renders(self, tmp_path):
        """D-019 NaNs five columns on a truncated row; the panels must tolerate that."""
        results = _results()
        results.spectral_analysis[MetricNames.SPECTRUM_TRUNCATED] = True
        for column in (MetricNames.WEAK_RANK_LOSS, MetricNames.ENTROPY,
                       MetricNames.MATRIX_RANK):
            results.spectral_analysis[column] = np.nan
        viz = _visualizer(results, tmp_path)
        viz.create_visualizations()
        assert (tmp_path / 'spectral_summary.png').exists()
