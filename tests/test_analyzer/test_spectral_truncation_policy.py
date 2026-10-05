"""A truncated spectrum must not publish whole-spectrum quantities (F-078, F-079).

Two decisions, tested together because F-079 is what gives F-078 a referent.

F-078 — the POLICY. When a layer exceeds the cap, `compute_eigenvalues` returns only the
LARGEST singular values; the small ones were never computed. D-019 correctly NaN-ed
`weak_rank_loss`, `entropy` and `matrix_rank`, but three more quantities that INTEGRATE over
the whole spectrum were still published as ordinary numbers:

| column | why a truncated spectrum cannot answer it |
|---|---|
| `norm` | Σ λ — the missing tail is mass the sum never saw |
| `log_norm` | log10(Σ λ) — ditto |
| `stable_rank` | (Σ λ)/λ_max — a strict UNDER-estimate: the mass it integrates over is exactly what is absent |
| `log_alpha_norm` | log10 Σ λ^α — integrates the entire spectrum |

Plus two outside that function: `mp_softrank` (its `λ_plus` is built from the whole-
spectrum mean, so the "theoretical MP edge" was not this layer's theoretical edge), and the
distribution-derived concentration ratios `gini_coefficient` (a Lorenz-curve quantity),
`dominance_ratio` (divides by the sum of the REST) and `participation_ratio` /
`min_participation_ratio` (divide by the total).

`stable_rank` is the damaging one: it is in `SPECTRAL_DEFAULT_SUMMARY_METRICS`, so it was
averaged into `spectral_summary` and drawn on the dashboard as a real capacity-utilisation
number.

F-079 — the REACHABILITY. That policy was inert, because the two conditions could never
both hold:

- `_describe_model` rejected any layer with `M > spectral_max_evals`
- `compute_eigenvalues` truncated only when `n_comp < M or M > max_evals`, and D-002 pins
  `n_comp = M` always

So `M > spectral_max_evals` was required to truncate and simultaneously forbidden by the
gate: `spectrum_truncated` was False on every row the analyzer had ever produced. Both
halves are now fixed — the gate admits the layer, and `max_evals` actually bounds `k` in the
SVD (it previously only selected the branch, so a 20000-wide layer with a 15000 cap ran
`svds(k=19999)`, a near-full decomposition MORE expensive than the dense SVD it replaced).

What must SURVIVE truncation. These come from quantities a truncated SVD still returns, and
NaN-ing them would throw away real information:

- `spectral_norm`, `log_spectral_norm`, `lambda_max`, `sv_max` — the largest singular value
  is exactly what the truncated branch returns
- `alpha_weighted`, `alpha_hat`, `alpha_hat_normalized` — all derived from that λ_max
- `concentration_score`, `critical_weight_count` — derived from the weight matrix itself,
  which was read in full

This module pins both halves, because a fix that NaN-ed everything would pass the first half
and quietly destroy the second.
"""

import keras
import numpy as np
import pandas as pd
import pytest

from dl_techniques.analyzer.analyzers.spectral_analyzer import SpectralAnalyzer
from dl_techniques.analyzer.config import AnalysisConfig
from dl_techniques.analyzer.constants import MetricNames
from dl_techniques.analyzer.data_types import AnalysisResults

#: Whole-spectrum columns that MUST be NaN on a truncated row.
MUST_BE_NAN = [
    MetricNames.WEAK_RANK_LOSS,
    MetricNames.ENTROPY,
    MetricNames.MATRIX_RANK,
    MetricNames.RANK_LOSS,
    MetricNames.NORM,
    MetricNames.LOG_NORM,
    MetricNames.STABLE_RANK,
    MetricNames.LOG_ALPHA_NORM,
    MetricNames.MP_SOFTRANK,
    MetricNames.GINI_COEFFICIENT,
    MetricNames.DOMINANCE_RATIO,
    MetricNames.PARTICIPATION_RATIO,
    MetricNames.MIN_PARTICIPATION_RATIO,
]

#: Columns computed from λ_max or from the weight matrix, which stay VALID.
MUST_REMAIN_VALID = [
    MetricNames.SPECTRAL_NORM,
    MetricNames.LOG_SPECTRAL_NORM,
    MetricNames.LAMBDA_MAX,
    MetricNames.SV_MAX,
    MetricNames.ALPHA_WEIGHTED,
    MetricNames.ALPHA_HAT,
    MetricNames.CONCENTRATION_SCORE,
    MetricNames.CRITICAL_WEIGHT_COUNT,
]


def _model(width):
    keras.utils.set_random_seed(0)
    return keras.Sequential([
        keras.layers.Input(shape=(width,)),
        keras.layers.Dense(width, activation='relu'),
        keras.layers.Dense(width, activation='relu'),
    ])


def _analyze(width=64, **config_kwargs):
    """Run the analyzer END TO END. F-079 is what makes this exercise the policy."""
    kwargs = dict(analyze_spectral=True, spectral_randomize=False,
                  spectral_bootstraps=0)
    kwargs.update(config_kwargs)
    analyzer = SpectralAnalyzer({'m': _model(width)}, AnalysisConfig(**kwargs))
    results = AnalysisResults()
    analyzer.analyze(results)
    return results.spectral_analysis


class TestTheTruncatedPathIsNowReachable:
    """F-079. The predecessor of this class asserted the OPPOSITE."""

    def test_a_cap_below_the_layer_width_now_yields_truncated_rows(self):
        frame = _analyze(width=64, spectral_min_evals=4, spectral_max_evals=8)
        assert frame is not None and not frame.empty, (
            "a layer of M=64 with spectral_max_evals=8 produced NO rows. F-079 removed "
            "the `M > spectral_max_evals` rejection from the admission gate; if this is "
            "empty again the gate has come back and the F-078 policy is inert."
        )
        assert bool(frame[MetricNames.SPECTRUM_TRUNCATED].any()), (
            "rows were produced but none is flagged truncated, so the capped-SVD path "
            "was not taken"
        )

    def test_the_cap_actually_bounds_the_decomposition(self):
        """F-079's second half: `max_evals` must bound `k`, not just pick the branch.

        Without this, a layer above the cap ran `svds(k = M - 1)` — a near-full ARPACK
        decomposition, more expensive than the dense SVD it replaces — while still
        reporting `truncated=True`.
        """
        frame = _analyze(width=64, spectral_min_evals=4, spectral_max_evals=8)
        truncated = frame[frame[MetricNames.SPECTRUM_TRUNCATED]]
        assert not truncated.empty
        assert (truncated[MetricNames.NUM_EVALS] <= 8).all(), (
            f"num_evals should be bounded by the cap of 8, got "
            f"{sorted(truncated[MetricNames.NUM_EVALS].tolist())} — `max_evals` is only "
            f"selecting the branch, not bounding k (F-079)"
        )
        # And genuinely fewer than the full spectrum, so `truncated` is honest.
        assert (truncated[MetricNames.NUM_EVALS] < truncated['M']).all()


class TestACompleteSpectrumLayersAreUnaffected:
    def test_nothing_is_nan_on_a_complete_spectrum(self):
        frame = _analyze(width=64)
        assert frame is not None and not frame.empty
        for column in MUST_BE_NAN:
            if column in frame.columns:
                values = pd.to_numeric(frame[column], errors='coerce')
                assert values.notna().all(), (
                    f"{column} is NaN on a COMPLETE spectrum: {frame[column].tolist()}"
                )

    def test_spectrum_truncated_is_false_at_the_default_cap(self):
        frame = _analyze(width=64)
        assert not bool(frame[MetricNames.SPECTRUM_TRUNCATED].any()), (
            "a 64-wide layer is far below the 15000 default cap and must not truncate"
        )

    def test_num_evals_equals_min_dim_on_a_complete_spectrum(self):
        frame = _analyze(width=64)
        assert (frame[MetricNames.NUM_EVALS] == frame['M']).all()


class TestATruncatedLayerNaNsOnlyWhatItCannotKnow:
    @pytest.fixture(scope="class")
    def truncated(self):
        frame = _analyze(width=64, spectral_min_evals=4, spectral_max_evals=8)
        assert frame is not None and not frame.empty
        return frame

    @pytest.mark.parametrize("column", MUST_BE_NAN)
    def test_a_whole_spectrum_column_is_nan(self, truncated, column):
        if column not in truncated.columns:
            pytest.skip(f"{column} absent from the frame on this configuration")
        row = truncated[truncated[MetricNames.SPECTRUM_TRUNCATED]][column].iloc[0]
        assert np.isnan(float(row)), (
            f"{column} = {row!r} on a TRUNCATED spectrum. F-078 requires NaN here: the "
            f"quantity integrates over eigenvalues that were never computed."
        )

    @pytest.mark.parametrize("column", MUST_REMAIN_VALID)
    def test_a_lambda_max_or_matrix_column_survives(self, truncated, column):
        """The other half of F-078: do not NaN what is genuinely known.

        A fix that NaN-ed every spectrum-derived column would pass the test above while
        destroying real information — λ_max is exactly what a truncated SVD returns, and
        the weight matrix was read in full.
        """
        if column not in truncated.columns:
            pytest.skip(f"{column} absent from the frame on this configuration")
        rows = truncated[truncated[MetricNames.SPECTRUM_TRUNCATED]]
        row = rows[column].iloc[0]
        assert not np.isnan(float(row)), (
            f"{column} = {row!r} on a truncated spectrum, but it is computed from λ_max "
            f"or from the weight matrix and IS knowable. F-078 NaN-ed whole-spectrum "
            f"INTEGRALS only; this over-NaN-ed a valid column."
        )


class TestSummaryMeansSkipTheNaNs:
    """A NaN in a layer row must not poison the model-level mean.

    `_get_summary` filters with `pd.to_numeric(...).notna()`, so the means are taken over
    the surviving layers. This is pinned because the alternative — a single NaN turning a
    whole model's `stable_rank` into NaN — is exactly the failure the NaN was meant to
    avoid.
    """

    def test_a_summary_over_a_mixed_frame_stays_finite(self):
        frame = pd.DataFrame([
            {MetricNames.STATUS: 'success', MetricNames.STABLE_RANK: 10.0},
            {MetricNames.STATUS: 'success', MetricNames.STABLE_RANK: float('nan')},
            {MetricNames.STATUS: 'success', MetricNames.STABLE_RANK: 20.0},
        ])
        analyzer = SpectralAnalyzer({}, AnalysisConfig())
        summary = analyzer._get_summary(frame)
        assert summary[MetricNames.STABLE_RANK] == pytest.approx(15.0), (
            f"the mean over the non-NaN rows is 15.0, got "
            f"{summary.get(MetricNames.STABLE_RANK)!r} — a NaN leaked into the summary"
        )

    def test_an_all_nan_column_is_simply_absent_from_the_summary(self):
        frame = pd.DataFrame([
            {MetricNames.STATUS: 'success', MetricNames.STABLE_RANK: float('nan')},
            {MetricNames.STATUS: 'success', MetricNames.STABLE_RANK: float('nan')},
        ])
        analyzer = SpectralAnalyzer({}, AnalysisConfig())
        summary = analyzer._get_summary(frame)
        assert MetricNames.STABLE_RANK not in summary

    def test_an_all_truncated_model_reports_no_stable_rank_at_all(self):
        """The realistic worst case: every layer truncated. The mean must be absent, not NaN."""
        frame = _analyze(width=64, spectral_min_evals=4, spectral_max_evals=8)
        analyzer = SpectralAnalyzer({}, AnalysisConfig())
        summary = analyzer._get_summary(frame)
        assert MetricNames.STABLE_RANK not in summary, (
            f"stable_rank is {summary.get(MetricNames.STABLE_RANK)!r} for a model whose "
            f"every layer truncated; it must be absent, not a NaN that poisons a "
            f"caller's own average"
        )