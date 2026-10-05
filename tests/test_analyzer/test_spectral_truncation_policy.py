"""A truncated spectrum must not publish whole-spectrum quantities (F-078).

The defect. When a layer exceeds `config.spectral_max_evals`, `compute_eigenvalues` takes
the truncated-`svds` branch and returns only the LARGEST singular values; the small ones
were never computed. D-019 correctly NaN-ed `weak_rank_loss`, `entropy` and `matrix_rank`
for that case. But three more quantities that INTEGRATE over the whole spectrum were still
published as ordinary numbers:

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

What must SURVIVE. These are computed from quantities a truncated SVD still returns, and
NaN-ing them would throw away real information:

- `spectral_norm`, `log_spectral_norm`, `lambda_max`, `sv_max` — the largest singular value
  is exactly what the truncated branch returns
- `alpha_weighted`, `alpha_hat`, `alpha_hat_normalized` — all derived from that λ_max
- `concentration_score`, `critical_weight_count` — derived from the weight matrix itself,
  which was read in full

This module pins both halves, because a fix that NaN-ed everything would pass the first
half and quietly destroy the second.
"""

import matplotlib
matplotlib.use('Agg')

import keras
import numpy as np
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
    kwargs = dict(analyze_spectral=True, spectral_randomize=False,
                  spectral_bootstraps=0)
    kwargs.update(config_kwargs)
    analyzer = SpectralAnalyzer({'m': _model(width)}, AnalysisConfig(**kwargs))
    results = AnalysisResults()
    analyzer.analyze(results)
    return results.spectral_analysis


def _analyze_layers_directly(width=64, max_evals=8):
    """Drive `_analyze_layers` past `_describe_model`'s admission gate.

    THE TRUNCATED PATH IS UNREACHABLE THROUGH `analyze()` (F-079). `_describe_model`
    admits a layer only when `spectral_min_evals <= M <= spectral_max_evals`, while
    `compute_eigenvalues` truncates only when `n_comp < M or M > max_evals` -- and
    D-002 pins `n_comp = M` always. So `M > spectral_max_evals` is required to truncate and
    simultaneously forbidden by the admission gate. The two conditions can never both
    hold, which means `spectrum_truncated` is False on every row the analyzer has ever
    produced.

    That is worth recording on its own, but it also means this policy cannot be exercised
    end-to-end. This helper therefore calls `_analyze_layers` directly on a hand-built
    frame with a small `max_evals`, which is exactly the code path F-078 changed.
    """
    analyzer = SpectralAnalyzer({'m': _model(width)},
                                AnalysisConfig(spectral_bootstraps=0,
                                               spectral_min_evals=4,
                                               spectral_max_evals=max_evals))
    details, all_layers = analyzer._describe_model(
        _model(width))
    assert details.empty, (
        "the probe model produced no describable layers, so there is nothing to analyze"
    )
    # Rebuild a frame WITHOUT the M <= spectral_max_evals gate so the layer reaches
    # `_analyze_layers`, where the truncation actually happens.
    import pandas as pd
    rows = []
    for layer_id, layer in enumerate(all_layers):
        from dl_techniques.analyzer import spectral_utils
        from dl_techniques.analyzer.constants import LayerType
        layer_type = spectral_utils.infer_layer_type(layer)
        has_weights, weights, _, _ = spectral_utils.get_layer_weights_and_bias(layer)
        if not has_weights or layer_type == LayerType.UNKNOWN:
            continue
        Wmats, N, M, rf = spectral_utils.get_weight_matrices(weights, layer_type)
        rows.append({'layer_id': layer_id, 'name': layer.name,
                     'layer_type': layer_type.value, 'N': N, 'M': M, 'rf': rf,
                     'Q': N / M if M else -1,
                     'num_params': int(np.prod(weights.shape)),
                     MetricNames.NUM_EVALS: M})
    details = pd.DataFrame(rows)
    assert not details.empty
    details.set_index('layer_id', inplace=True)

    esds, rand_esds = {}, {}
    analyzer._analyze_layers(details, all_layers, esds, rand_esds,
                             rng=np.random.default_rng(0))
    return details


class TestTheTruncatedPathIsUnreachableThroughAnalyze:
    """F-079: the precondition for everything below. Asserted, not assumed."""

    def test_a_small_max_evals_rejects_the_layer_entirely(self):
        """M > spectral_max_evals means no ROW AT ALL, not a truncated row.

        `spectral_analysis` stays None (rather than an empty frame) when nothing is
        admitted, so the assertion is on None-or-empty rather than `.empty`.
        """
        frame = _analyze(width=64, spectral_min_evals=4, spectral_max_evals=8)
        admitted = 0 if frame is None else len(frame)
        assert admitted == 0, (
            f"expected NO rows at spectral_max_evals=8, got {admitted}. If this now "
            f"produces truncated rows, the admission gate has changed and F-079's "
            f"unreachability claim -- and this module's need to bypass the gate -- is "
            f"stale."
        )

    def test_a_normal_run_reports_no_truncation(self):
        frame = _analyze(width=64)
        assert not frame.empty
        assert not bool(frame[MetricNames.SPECTRUM_TRUNCATED].any())


class TestAWholeSpectrumLayersAreUnaffected:
    def test_nothing_is_nan_on_a_complete_spectrum(self):
        frame = _analyze(width=64)
        assert not frame.empty, "the analyzer produced no rows at all"
        for column in MUST_BE_NAN:
            if column in frame.columns:
                values = pd_numeric(frame[column])
                assert values.notna().all(), (
                    f"{column} is NaN on a COMPLETE spectrum: {frame[column].tolist()}"
                )

    def test_spectrum_truncated_is_false(self):
        frame = _analyze(width=64)
        assert not bool(frame[MetricNames.SPECTRUM_TRUNCATED].any())


class TestATruncatedLayerNaNsOnlyWhatItCannotKnow:
    @pytest.fixture(scope="class")
    def truncated(self):
        frame = _analyze_layers_directly(width=64, max_evals=8)
        assert not frame.empty, "the direct probe produced no rows"
        return frame

    def test_the_truncation_flag_is_set(self, truncated):
        assert bool(truncated[MetricNames.SPECTRUM_TRUNCATED].any()), (
            f"the direct probe did not produce a truncated spectrum, so the F-078 "
            f"branch was never taken and this class is vacuous: "
            f"{truncated[MetricNames.SPECTRUM_TRUNCATED].tolist()}"
        )

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
        destroying real information -- λ_max is exactly what a truncated SVD returns, and
        the weight matrix was read in full.
        """
        if column not in truncated.columns:
            pytest.skip(f"{column} absent from the frame on this configuration")
        rows = truncated[truncated[MetricNames.SPECTRUM_TRUNCATED]]
        row = rows[column].iloc[0]
        assert not np.isnan(float(row)), (
            f"{column} = {row!r} on a truncated spectrum, but it is computed from "
            f"λ_max or from the weight matrix and IS knowable. F-078 NaN-ed "
            f"whole-spectrum INTEGRALS only; this over-NaN-ed a valid column."
        )

    def test_num_evals_is_the_truncated_count_not_min_dim(self, truncated):
        """The truncated row genuinely carries fewer eigenvalues than min(N, M)."""
        rows = truncated[truncated[MetricNames.SPECTRUM_TRUNCATED]]
        assert (rows[MetricNames.NUM_EVALS] < rows['M']).all(), (
            "num_evals should be the number of singular values actually returned"
        )


class TestSummaryMeansSkipTheNaNs:
    """A NaN in a layer row must not poison the model-level mean.

    `_get_summary` filters with `pd.to_numeric(...).notna()`, so the means are taken over
    the surviving layers. This pins that, because the alternative -- a single NaN turning a
    whole model's `stable_rank` into NaN -- is exactly the failure the NaN was meant to
    avoid.
    """

    def test_a_summary_over_a_mixed_frame_stays_finite(self):
        import pandas as pd
        frame = pd.DataFrame([
            {MetricNames.STATUS: 'success', MetricNames.STABLE_RANK: 10.0},
            {MetricNames.STATUS: 'success', MetricNames.STABLE_RANK: float('nan')},
            {MetricNames.STATUS: 'success', MetricNames.STABLE_RANK: 20.0},
        ])
        analyzer = SpectralAnalyzer({}, AnalysisConfig())
        summary = analyzer._get_summary(frame)
        assert summary[MetricNames.STABLE_RANK] == pytest.approx(15.0), (
            f"the mean over the non-NaN rows is 15.0, got "
            f"{summary.get(MetricNames.STABLE_RANK)!r} -- a NaN leaked into the summary"
        )

    def test_an_all_nan_column_is_simply_absent_from_the_summary(self):
        import pandas as pd
        frame = pd.DataFrame([
            {MetricNames.STATUS: 'success', MetricNames.STABLE_RANK: float('nan')},
            {MetricNames.STATUS: 'success', MetricNames.STABLE_RANK: float('nan')},
        ])
        analyzer = SpectralAnalyzer({}, AnalysisConfig())
        summary = analyzer._get_summary(frame)
        assert MetricNames.STABLE_RANK not in summary


def pd_numeric(series):
    import pandas as pd
    return pd.to_numeric(series, errors='coerce')