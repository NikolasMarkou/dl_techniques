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
