"""Guards for `dl_techniques.analyzer.analyzers.spectral_analyzer`.

Each test in this module exists to pin a behaviour that was measured WRONG against the
unfixed code; every guard was demonstrated RED before its fix landed.
"""

import numpy as np
import pandas as pd
import pytest

from dl_techniques.analyzer.config import AnalysisConfig
from dl_techniques.analyzer.constants import MetricNames, StatusCode
from dl_techniques.analyzer.analyzers.spectral_analyzer import SpectralAnalyzer


@pytest.fixture()
def analyzer() -> SpectralAnalyzer:
    """A SpectralAnalyzer with no models — `_get_summary` is a pure frame reduction."""
    return SpectralAnalyzer(models={}, config=AnalysisConfig())


def _frame(rows) -> pd.DataFrame:
    """Build a details-shaped frame from `(alpha, alpha_weighted, status)` triples."""
    return pd.DataFrame(
        [
            {
                MetricNames.ALPHA: alpha,
                MetricNames.ALPHA_WEIGHTED: alpha_weighted,
                MetricNames.STATUS: status,
            }
            for alpha, alpha_weighted, status in rows
        ]
    )


# Measured on the unfixed code (findings/correctness-bugs.md C3): a real fit of
# alpha=1.800772 beside a `failed` sentinel row of alpha=-1.0 summarised to 0.40038621,
# and the derived alpha_weighted summarised to 5.4645 against the real layer's 0.9290.
_REAL_ALPHA = 1.800772
_REAL_ALPHA_WEIGHTED = 0.9290
_FAILED_ALPHA = -1.0
_FAILED_ALPHA_WEIGHTED = 10.0


class TestSummaryExcludesFailedFits:
    """C3 — a `status='failed'` row must not be averaged into the spectral summary."""

    def test_a_failed_fit_is_not_averaged_into_the_summary(self, analyzer):
        details = _frame(
            [
                (_REAL_ALPHA, _REAL_ALPHA_WEIGHTED, StatusCode.SUCCESS.value),
                (_FAILED_ALPHA, _FAILED_ALPHA_WEIGHTED, StatusCode.FAILED.value),
            ]
        )

        # Anti-vacuity: the two rows must actually disagree, so a summary equal to the
        # success-only value cannot be right by coincidence.
        cross_row_mean = float(np.mean([_REAL_ALPHA, _FAILED_ALPHA]))
        assert abs(cross_row_mean - _REAL_ALPHA) > 1.0

        summary = analyzer._get_summary(details)

        assert summary[MetricNames.ALPHA] == pytest.approx(_REAL_ALPHA), (
            "the failed fit's -1.0 sentinel was averaged into the alpha summary: "
            f"reported={summary[MetricNames.ALPHA]!r} "
            f"success-only={_REAL_ALPHA!r} cross-row mean={cross_row_mean!r}"
        )
        assert summary[MetricNames.ALPHA_WEIGHTED] == pytest.approx(
            _REAL_ALPHA_WEIGHTED
        ), (
            "the failed fit's +10.0 alpha_weighted sentinel was averaged in: "
            f"reported={summary[MetricNames.ALPHA_WEIGHTED]!r}"
        )

    def test_the_summary_reports_how_many_fits_failed(self, analyzer):
        details = _frame(
            [
                (_REAL_ALPHA, _REAL_ALPHA_WEIGHTED, StatusCode.SUCCESS.value),
                (_FAILED_ALPHA, _FAILED_ALPHA_WEIGHTED, StatusCode.FAILED.value),
            ]
        )

        summary = analyzer._get_summary(details)

        assert summary['failed_layers'] == 1
        assert summary['successful_layers_analyzed'] == 1
        # `total_layers_analyzed` keeps its published meaning: every row described.
        assert summary['total_layers_analyzed'] == 2

    def test_an_all_success_frame_is_unchanged(self, analyzer):
        """Anti-vacuity arm: the filter must be inert when nothing failed."""
        details = _frame(
            [
                (2.0, 1.0, StatusCode.SUCCESS.value),
                (3.0, 2.0, StatusCode.SUCCESS.value),
            ]
        )

        summary = analyzer._get_summary(details)

        assert summary[MetricNames.ALPHA] == pytest.approx(2.5)
        assert summary[MetricNames.ALPHA_WEIGHTED] == pytest.approx(1.5)
        assert summary['total_layers_analyzed'] == 2
        assert summary['failed_layers'] == 0

    def test_a_frame_without_a_status_column_still_summarizes(self, analyzer):
        """A caller-supplied frame predating the status column must not crash."""
        details = pd.DataFrame([{MetricNames.ALPHA: 2.0}, {MetricNames.ALPHA: 4.0}])

        summary = analyzer._get_summary(details)

        assert summary[MetricNames.ALPHA] == pytest.approx(3.0)
        assert summary['total_layers_analyzed'] == 2


def _two_models_with_different_spectra():
    """Two single-Dense models whose weight spectra differ by construction.

    Returns:
        Dict mapping model name to model. Model ``m_a`` carries an isotropic Gaussian
        kernel; ``m_b`` carries a strongly anisotropic one, so their fitted alphas differ.
    """
    import keras

    rng = np.random.default_rng(20260902)

    def _build(name: str, kernel: np.ndarray) -> keras.Model:
        inputs = keras.Input(shape=(kernel.shape[0],), name=f"{name}_in")
        dense = keras.layers.Dense(kernel.shape[1], use_bias=False, name=f"{name}_dense")
        model = keras.Model(inputs=inputs, outputs=dense(inputs), name=name)
        dense.set_weights([kernel.astype("float32")])
        return model

    n, m = 64, 32
    isotropic = rng.normal(size=(n, m))
    anisotropic = isotropic * np.linspace(1.0, 40.0, m)[None, :]
    return {"m_a": _build("m_a", isotropic), "m_b": _build("m_b", anisotropic)}


class TestPerModelSpectralSummary:
    """C4 — the spectral summary must not collapse every model into one mean."""

    def test_the_summary_is_reported_per_model(self):
        from dl_techniques.analyzer.data_types import AnalysisResults

        models = _two_models_with_different_spectra()
        config = AnalysisConfig(analyze_spectral=True)
        results = AnalysisResults()
        SpectralAnalyzer(models=models, config=config).analyze(results)

        frame = results.spectral_analysis
        assert frame is not None and not frame.empty

        # Anti-vacuity: the two models must genuinely disagree on alpha, otherwise a
        # cross-model mean would be indistinguishable from a per-model one.
        per_model_alpha = {
            name: float(group[MetricNames.ALPHA].mean())
            for name, group in frame.groupby('model_name')
        }
        assert set(per_model_alpha) == {"m_a", "m_b"}
        assert abs(per_model_alpha["m_a"] - per_model_alpha["m_b"]) > 1e-3, (
            f"degenerate probe: both models fitted the same alpha {per_model_alpha}"
        )

        per_model = results.spectral_summary_per_model
        assert set(per_model) == {"m_a", "m_b"}, (
            "spectral summary is not reported per model; the only summary available is the "
            f"cross-model aggregate {results.spectral_summary.get(MetricNames.ALPHA)!r}"
        )
        for name, expected in per_model_alpha.items():
            assert per_model[name][MetricNames.ALPHA] == pytest.approx(expected), (
                f"model '{name}' summary alpha is not its own layers' mean: "
                f"reported={per_model[name][MetricNames.ALPHA]!r} expected={expected!r}"
            )

    def test_the_flat_aggregate_is_retained_for_backwards_compatibility(self):
        """The published flat `spectral_summary` keeps its cross-model meaning."""
        from dl_techniques.analyzer.data_types import AnalysisResults

        models = _two_models_with_different_spectra()
        results = AnalysisResults()
        SpectralAnalyzer(models=models, config=AnalysisConfig(analyze_spectral=True)).analyze(results)

        frame = results.spectral_analysis
        expected = float(frame[MetricNames.ALPHA].mean())
        assert results.spectral_summary[MetricNames.ALPHA] == pytest.approx(expected)

    def test_get_summary_statistics_surfaces_the_per_model_summary(self, tmp_path):
        """The public entry point must expose both shapes, not only the aggregate."""
        from dl_techniques.analyzer.model_analyzer import ModelAnalyzer

        models = _two_models_with_different_spectra()
        analyzer = ModelAnalyzer(
            models=models,
            config=AnalysisConfig(
                analyze_spectral=True,
                save_plots=False,
                save_format='json',
                verbose=False,
            ),
            output_dir=str(tmp_path / "c4"),
        )
        analyzer.analyze(analysis_types={"spectral"})
        summary = analyzer.get_summary_statistics()

        assert 'spectral_summary' in summary
        assert set(summary['spectral_summary_per_model']) == {"m_a", "m_b"}
        assert summary['spectral_summary_per_model']["m_a"][MetricNames.ALPHA] != pytest.approx(
            summary['spectral_summary_per_model']["m_b"][MetricNames.ALPHA]
        )


class TestRecommendationsIgnoreUncomputedPvalues:
    """C5-real — a "not computed" p-value must not be reported as a poor fit."""

    def test_a_not_computed_pvalue_is_not_counted_as_a_poor_fit(self, analyzer):
        from dl_techniques.analyzer.constants import SPECTRAL_PVALUE_NOT_COMPUTED

        details = pd.DataFrame(
            [
                {MetricNames.ALPHA: 2.0, 'pl_pvalue': SPECTRAL_PVALUE_NOT_COMPUTED,
                 MetricNames.STATUS: StatusCode.SUCCESS.value},
                {MetricNames.ALPHA: 2.5, 'pl_pvalue': SPECTRAL_PVALUE_NOT_COMPUTED,
                 MetricNames.STATUS: StatusCode.SUCCESS.value},
            ]
        )

        recommendations = analyzer._generate_recommendations(
            details, analyzer._get_summary(details))

        assert not any('poor power-law fit' in r for r in recommendations), (
            f"uncomputed p-values were reported as poor fits: {recommendations}"
        )

    def test_a_genuinely_poor_fit_is_still_reported(self, analyzer):
        """Anti-vacuity arm: the recommendation must still fire on a real p < 0.1."""
        details = pd.DataFrame(
            [{MetricNames.ALPHA: 2.0, 'pl_pvalue': 0.0,
              MetricNames.STATUS: StatusCode.SUCCESS.value}]
        )

        recommendations = analyzer._generate_recommendations(
            details, analyzer._get_summary(details))

        assert any('poor power-law fit' in r for r in recommendations)
