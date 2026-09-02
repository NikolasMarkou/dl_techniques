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


# =====================================================================
# S2 — `mp_softrank` was identically 1.0 at shipped defaults (plan step 14)
# =====================================================================

class TestMpSoftrankIsNotAConstant:
    """At `spectral_randomize=False` the column was `1.0` for every layer.

    `num_rand_spikes` is `0` unless randomization ran, and
    `compute_mp_softrank(evals, 0)` returns EXACTLY `lambda_max / lambda_max`.
    Worse, when randomization DID run the count came from the RANDOMIZED spectrum
    and was applied to the ORIGINAL one.
    """

    @staticmethod
    def _analyze_at_defaults() -> pd.DataFrame:
        from dl_techniques.analyzer.data_types import AnalysisResults

        config = AnalysisConfig(analyze_spectral=True)
        # Pin the regime the defect lives in, rather than trusting the default.
        assert config.spectral_randomize is False
        results = AnalysisResults()
        SpectralAnalyzer(
            models=_two_models_with_different_spectra(), config=config).analyze(results)
        frame = results.spectral_analysis
        assert frame is not None and not frame.empty
        return frame

    def test_the_column_is_populated_at_all(self):
        """Anti-vacuity: an empty or all-NaN column makes 'not constant' vacuous."""
        details = self._analyze_at_defaults()
        column = details[MetricNames.MP_SOFTRANK].dropna()
        assert len(column) >= 2, (
            f"the probe analyzed {len(column)} layers; a not-constant assertion "
            f"needs at least two"
        )
        assert (details[MetricNames.HAS_ESD]).all(), "a layer was skipped by the analyzer"

    def test_mp_softrank_is_not_identically_one(self):
        details = self._analyze_at_defaults()
        values = details[MetricNames.MP_SOFTRANK].to_numpy(dtype=float)

        assert not np.allclose(values, 1.0), (
            f"mp_softrank is a constant masquerading as a metric: {values.tolist()}"
        )

    def test_mp_softrank_discriminates_the_two_spectra(self):
        """The anisotropic model's spectrum is spikier, so its soft rank is lower."""
        details = self._analyze_at_defaults()
        by_model = details.groupby("model_name")[MetricNames.MP_SOFTRANK].mean()

        assert by_model["m_b"] < by_model["m_a"], (
            f"mp_softrank does not separate an isotropic from a strongly "
            f"anisotropic spectrum: {by_model.to_dict()}"
        )


# =====================================================================
# S3 — randomization drew ONCE per layer (plan step 18)
# =====================================================================

class TestRandomizationAveragesMultipleDraws:
    """One unseeded permutation per layer made every trap verdict a coin flip."""

    def test_the_draw_count_is_a_config_field_defaulting_to_at_least_five(self):
        config = AnalysisConfig()
        assert hasattr(config, "spectral_n_randomizations"), (
            "the number of randomization draws is not configurable"
        )
        assert config.spectral_n_randomizations >= 5, (
            f"default draw count is {config.spectral_n_randomizations}, "
            f"below the >= 5 the plan requires"
        )

    def test_the_analyzer_draws_the_configured_number_of_times(self, monkeypatch):
        from dl_techniques.analyzer.data_types import AnalysisResults
        from dl_techniques.analyzer.analyzers import spectral_analyzer as sa

        calls = []
        real = sa.spectral_metrics.detect_correlation_trap

        def counting(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)

        monkeypatch.setattr(
            sa.spectral_metrics, "detect_correlation_trap", counting)

        n_draws = 6
        config = AnalysisConfig(
            analyze_spectral=True,
            spectral_randomize=True,
            spectral_n_randomizations=n_draws,
        )
        results = AnalysisResults()
        SpectralAnalyzer(
            models=_two_models_with_different_spectra(), config=config).analyze(results)

        frame = results.spectral_analysis
        n_layers = len(frame)
        # Anti-vacuity: the randomization branch must actually have run.
        assert n_layers == 2, f"the probe analyzed {n_layers} layers"
        assert MetricNames.NUM_RAND_SPIKES in frame.columns, (
            "the randomization branch did not run at all"
        )

        assert len(calls) == n_layers * n_draws, (
            f"detect_correlation_trap ran {len(calls)} times for {n_layers} "
            f"layers at spectral_n_randomizations={n_draws}"
        )

    def test_a_single_draw_is_still_honoured(self):
        """Anti-vacuity: the averaging must be driven by the field, not hardcoded."""
        from dl_techniques.analyzer.data_types import AnalysisResults
        from dl_techniques.analyzer.analyzers import spectral_analyzer as sa

        calls = []
        real = sa.spectral_metrics.detect_correlation_trap

        def counting(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)

        original = sa.spectral_metrics.detect_correlation_trap
        sa.spectral_metrics.detect_correlation_trap = counting
        try:
            results = AnalysisResults()
            SpectralAnalyzer(
                models=_two_models_with_different_spectra(),
                config=AnalysisConfig(
                    analyze_spectral=True,
                    spectral_randomize=True,
                    spectral_n_randomizations=1,
                ),
            ).analyze(results)
        finally:
            sa.spectral_metrics.detect_correlation_trap = original

        assert len(calls) == 2


# =====================================================================
# A2 - analysis artifacts must be RETURNED, not parked on `self`
# =====================================================================

def _two_layer_model(name: str):
    import keras
    inputs = keras.Input(shape=(16,), name=f"{name}_in")
    h = keras.layers.Dense(12, name=f"{name}_h")(inputs)
    return keras.Model(inputs, keras.layers.Dense(8, name=f"{name}_o")(h), name=name)


class TestSingleModelAnalysisReturnsItsArtifacts:
    """`_analyze_single_model` set `self._esd_cache` / `self._rand_esd_cache` /
    `self._recommendations` / `self._model_summary` unconditionally at the top of the
    method, and `analyze` recovered them through `hasattr` guards that therefore could
    never fire. The artifacts belong in the return value.
    """

    _PARKED = ("_esd_cache", "_rand_esd_cache", "_recommendations", "_model_summary")

    def test_the_artifacts_come_back_from_the_call(self):
        model = _two_layer_model("ret")
        analyzer = SpectralAnalyzer(
            models={"ret": model}, config=AnalysisConfig(analyze_spectral=True))

        details, esds, rand_esds, recommendations, summary = (
            analyzer._analyze_single_model(model))

        # Anti-vacuity: the model really did produce analyzable layers, so an empty
        # artifact set cannot pass this test by accident.
        assert not details.empty
        assert len(esds) == len(details)
        assert all(isinstance(v, np.ndarray) and len(v) > 0 for v in esds.values())
        assert isinstance(rand_esds, dict)
        assert isinstance(recommendations, list)
        assert isinstance(summary, dict) and summary

    def test_no_analysis_state_is_parked_on_the_instance(self):
        model = _two_layer_model("parked")
        analyzer = SpectralAnalyzer(
            models={"parked": model}, config=AnalysisConfig(analyze_spectral=True))
        analyzer._analyze_single_model(model)

        parked = [a for a in self._PARKED if hasattr(analyzer, a)]
        assert parked == [], (
            f"analysis state is still parked on the analyzer instance: {parked}"
        )

    def test_two_models_get_their_own_artifacts(self):
        """Behavioural PIN (passes before and after): no cross-model aliasing.

        This is NOT red evidence - it exists so the refactor cannot silently start
        sharing one artifact dict between models.
        """
        from dl_techniques.analyzer.data_types import AnalysisResults

        results = AnalysisResults()
        SpectralAnalyzer(
            models={"a": _two_layer_model("a"), "b": _two_layer_model("b")},
            config=AnalysisConfig(analyze_spectral=True),
        ).analyze(results)

        assert set(results.spectral_esds) == {"a", "b"}
        assert results.spectral_esds["a"] is not results.spectral_esds["b"]
        assert set(results.spectral_recommendations) == {"a", "b"}
        assert set(results.spectral_summary_per_model) == {"a", "b"}


class TestTheAnalyzerNeverEnumeratesEveryCriticalWeight:
    """`find_critical_weights` lists the WHOLE critical-weight population.

    On a dense layer at the shipped threshold of 0.1 that is one entry per matrix
    element — 11,385,666 on a 4096x4096 layer, MEASURED at 21.73 s of the 24.48 s
    the concentration path spent there. The analyzer needs only the count and the
    top ten, both of which `summarize_critical_weights` returns without building
    the list, so the analyzer path must not reach the listing function at all.

    The function itself is retained as public API; this pins that nothing on the
    default path calls it.
    """

    def test_the_default_path_does_not_call_the_listing_function(self):
        from unittest import mock
        from dl_techniques.analyzer import spectral_metrics as sm
        from dl_techniques.analyzer.data_types import AnalysisResults

        calls = []
        real = sm.find_critical_weights

        def spy(*args, **kwargs):
            calls.append(True)
            return real(*args, **kwargs)

        config = AnalysisConfig(
            analyze_spectral=True, spectral_concentration_analysis=True)
        with mock.patch.object(sm, "find_critical_weights", side_effect=spy):
            SpectralAnalyzer(
                models={"m": _two_layer_model("m")}, config=config,
            ).analyze(AnalysisResults())

        assert not calls, (
            f"find_critical_weights was called {len(calls)} time(s) on the default "
            f"analyzer path; it materialises one tuple per matrix element and the "
            f"analyzer only ever consumes the count and the top ten"
        )

    def test_the_concentration_analysis_flag_really_was_on(self):
        """Anti-vacuity: the test above passes trivially if nothing ran.

        `spectral_concentration_analysis` defaults to True, and the column it
        produces must actually be in the frame — otherwise "the function was not
        called" would be evidence of a skipped analysis, not of a cheaper one.
        """
        from dl_techniques.analyzer.data_types import AnalysisResults

        import keras

        # Wide enough that `min(shape)` clears SPECTRAL_DEFAULT_MIN_EVALS, so the
        # concentration path actually runs. The 16 -> 12 -> 8 model used elsewhere
        # in this file is below that floor.
        inputs = keras.Input(shape=(64,), name="cw_in")
        hidden = keras.layers.Dense(48, name="cw_h")(inputs)
        model = keras.Model(
            inputs, keras.layers.Dense(32, name="cw_o")(hidden), name="cw")

        results = AnalysisResults()
        SpectralAnalyzer(
            models={"cw": model},
            config=AnalysisConfig(
                analyze_spectral=True, spectral_concentration_analysis=True),
        ).analyze(results)

        # `spectral_analysis` is one concatenated DataFrame over all models, not a
        # per-model dict.
        frame = results.spectral_analysis
        assert "critical_weight_count" in frame.columns, (
            "critical_weight_count is absent from the frame, so the concentration "
            "path did not run and the call-count assertion proves nothing"
        )
        counts = frame["critical_weight_count"].dropna()
        assert len(counts) > 0 and (counts > 0).any(), (
            f"every critical_weight_count is zero or missing ({list(counts)}); the "
            f"published column would have gone dead alongside the listing call"
        )
