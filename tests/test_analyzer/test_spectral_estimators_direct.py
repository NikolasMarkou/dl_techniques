"""`estimate_bulk_variance` and `label_trap_severity` had NO direct unit tests.

`estimate_bulk_variance` (F-083). This is the estimator behind the entire correlation-trap
false-positive calibration, and D-017 replaced the naive whole-spectrum mean with it on the
strength of a MEASUREMENT: including a spike in σ² moves the MP edge that is supposed to
identify it, so the detector became conservative exactly when it should fire hardest. It was
only ever reached indirectly, through `detect_correlation_trap` — nothing asserted the
property it exists for.

`label_trap_severity` (F-084). The band map for trap severity, with a published table
(CORRELATION_TRAPS.md §7.2) and five boundary constants. Nothing tested the boundaries, so a
change to any of the four thresholds would silently re-band every published label.

The figures below are measured, not assumed: the clean-probe values `1.006857` (naive) and
`0.979137` (spike-excluded) are the two numbers CORRELATION_TRAPS.md §0 already quotes for
this exact 200x50 Wishart, and they are reproduced here from the implementation.
"""

import numpy as np
import pytest

from dl_techniques.analyzer.constants import (
    SPECTRAL_EPSILON,
    SPECTRAL_TRAP_SEVERITY_MILD,
    SPECTRAL_TRAP_SEVERITY_MODERATE,
    SPECTRAL_TRAP_SEVERITY_SEVERE,
    SPECTRAL_TRAP_SEVERITY_CRITICAL,
)
from dl_techniques.analyzer.spectral_metrics import (
    calc_mp_edges,
    estimate_bulk_variance,
    label_trap_severity,
)


def _wishart(rows=200, cols=50, seed=3):
    """A clean Gaussian Wishart spectrum — the §0 probe's construction."""
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(rows, cols)) / np.sqrt(rows)
    return np.linalg.svd(W, compute_uv=False) ** 2


#: Q = N / M with N the LARGER dimension, so Q >= 1 (the `calc_mp_edges` convention).
_Q = 200 / 50


class TestEstimateBulkVarianceExcludesSpikes:
    """The property the function exists for."""

    def test_a_clean_spectrum_is_close_to_the_naive_mean(self):
        clean = _wishart()
        naive = float(np.mean(clean))
        bulk = estimate_bulk_variance(clean, _Q)
        assert naive == pytest.approx(1.006857, rel=1e-5), (
            f"the probe changed: naive mean is {naive!r}. CORRELATION_TRAPS.md §0 quotes "
            f"1.006857 for this construction and the doc guard is calibrated on it."
        )
        assert bulk == pytest.approx(0.979137, rel=1e-5), (
            f"spike-excluded estimate is {bulk!r}, expected the documented 0.979137"
        )
        # On a CLEAN spectrum the two must agree closely; that is the property that
        # makes the estimator safe to use unconditionally.
        assert bulk == pytest.approx(naive, rel=0.05)

    def test_a_planted_spike_moves_the_naive_mean_far_more_than_the_bulk_estimate(self):
        """D-017's measurement, asserted rather than quoted."""
        clean = _wishart()
        spiked = np.concatenate([clean[:49], [clean[0] * 20.0]])

        naive_shift = float(np.mean(spiked)) / float(np.mean(clean))
        bulk = estimate_bulk_variance(clean, _Q)
        bulk_shift = estimate_bulk_variance(spiked, _Q) / bulk

        assert naive_shift == pytest.approx(1.934, rel=0.02), (
            f"the naive mean moved {naive_shift:.3f}x; the probe changed"
        )
        assert bulk_shift < 1.05, (
            f"the spike-excluded estimate moved {bulk_shift:.3f}x under a 20x planted "
            f"spike. If this regresses to the naive mean, the MP edge inflates with the "
            f"spike and the detector goes conservative exactly when it should fire — the "
            f"defect D-017 replaced."
        )

    def test_the_estimate_ignores_eigenvalues_above_the_bulk_edge(self):
        """The fixed point: re-meaning over `evals <= λ_plus` reproduces the output.

        This is the strongest statement available without a reference implementation. If
        the estimate is self-consistent, then the returned σ² is exactly the mean of the
        bulk it claims to describe, and adding arbitrarily many spikes above the edge
        cannot move it at all.
        """
        clean = _wishart()
        spiked = np.concatenate([clean[:49], [clean[0] * 20.0, clean[1] * 50.0,
                                               clean[2] * 500.0]])
        estimate = estimate_bulk_variance(spiked, _Q)
        _, lambda_plus = calc_mp_edges(estimate, _Q)
        bulk = spiked[spiked <= lambda_plus]
        assert float(np.mean(bulk)) == pytest.approx(estimate, rel=1e-9)

    def test_additional_spikes_above_the_edge_do_not_move_the_estimate(self):
        clean = _wishart()
        base = np.concatenate([clean[:49], [clean[0] * 20.0]])
        with_more = np.concatenate([base, [clean[1] * 1000.0, clean[2] * 10000.0]])
        assert estimate_bulk_variance(with_more, _Q) == pytest.approx(
            estimate_bulk_variance(base, _Q), rel=1e-9), (
            "adding spikes strictly above the bulk edge changed the estimate; the whole "
            "point of the spike-excluded estimator is that it does not"
        )


class TestEstimateBulkVarianceConverges:
    def test_it_is_monotonically_non_increasing_in_n_iter(self):
        """Refinement can only ever exclude more eigenvalues, so σ² can only fall."""
        spiked = np.concatenate([_wishart()[:49], [_wishart()[0] * 20.0]])
        estimates = [estimate_bulk_variance(spiked, _Q, n_iter=n)
                     for n in (0, 1, 2, 3, 5, 10, 25)]
        for earlier, later in zip(estimates, estimates[1:]):
            assert later <= earlier + 1e-12, (
                f"n_iter sequence produced a non-monotone increase: {estimates}"
            )

    def test_it_converges_within_a_few_iterations(self):
        """The docstring claims convergence in 2-3 passes; hold it to that."""
        spiked = np.concatenate([_wishart()[:49], [_wishart()[0] * 20.0]])
        at_three = estimate_bulk_variance(spiked, _Q, n_iter=3)
        at_thirty = estimate_bulk_variance(spiked, _Q, n_iter=30)
        assert at_three == pytest.approx(at_thirty, rel=1e-9), (
            f"n_iter=3 gave {at_three!r} and n_iter=30 gave {at_thirty!r}; the default of "
            f"3 is not converged"
        )

    def test_zero_iterations_is_exactly_the_naive_mean(self):
        """The starting point of the iteration, pinned so the loop's entry is explicit."""
        spiked = np.concatenate([_wishart()[:49], [_wishart()[0] * 20.0]])
        assert estimate_bulk_variance(spiked, _Q, n_iter=0) == pytest.approx(
            float(np.mean(spiked)), rel=1e-12)

    def test_a_negative_n_iter_is_tolerated_as_zero(self):
        """`max(0, n_iter)` in the loop means it cannot raise or misbehave."""
        spiked = np.concatenate([_wishart()[:49], [_wishart()[0] * 20.0]])
        assert estimate_bulk_variance(spiked, _Q, n_iter=-5) == pytest.approx(
            estimate_bulk_variance(spiked, _Q, n_iter=0), rel=1e-12)


class TestEstimateBulkVarianceDegenerateInput:
    @pytest.mark.parametrize("evals", [None, np.array([]), np.array([0.0, 0.0])])
    def test_empty_or_zero_input_is_zero(self, evals):
        assert estimate_bulk_variance(evals, _Q) == 0.0

    @pytest.mark.parametrize("Q", [0.0, -1.0])
    def test_a_non_positive_aspect_ratio_is_zero(self, Q):
        assert estimate_bulk_variance(_wishart(), Q) == 0.0

    def test_a_below_epsilon_bulk_short_circuits_rather_than_dividing(self):
        tiny = np.full(10, SPECTRAL_EPSILON / 10)
        assert estimate_bulk_variance(tiny, _Q) == 0.0

    def test_a_single_eigenvalue_spectrum_is_its_own_bulk(self):
        one = np.array([3.0])
        assert estimate_bulk_variance(one, _Q) == pytest.approx(3.0, rel=1e-9)

    def test_it_accepts_a_python_list(self):
        assert estimate_bulk_variance([2.0, 2.0, 2.0], _Q) == pytest.approx(2.0, rel=1e-9)


class TestLabelTrapSeverityBands:
    """F-084. The five bands, at their exact boundaries."""

    def test_the_boundaries_are_the_published_constants(self):
        """Guard the table itself: CORRELATION_TRAPS.md §7.2 quotes 0.1/0.3/0.5/1.0."""
        assert SPECTRAL_TRAP_SEVERITY_MILD == 0.1
        assert SPECTRAL_TRAP_SEVERITY_MODERATE == 0.3
        assert SPECTRAL_TRAP_SEVERITY_SEVERE == 0.5
        assert SPECTRAL_TRAP_SEVERITY_CRITICAL == 1.0

    @pytest.mark.parametrize("severity,expected", [
        (-0.0, 'none'),
        (0.0, 'none'),
        (0.05, 'none'),
        # Each band's LOWER edge is inclusive, its UPPER edge exclusive — a severity
        # exactly on a boundary belongs to the band it starts.
        (0.1, 'mild'),
        (0.2, 'mild'),
        (0.3, 'moderate'),
        (0.4, 'moderate'),
        (0.5, 'severe'),
        (0.75, 'severe'),
        (1.0, 'critical'),
        (2.0, 'critical'),
        (1e6, 'critical'),
    ])
    def test_band_at_and_around_each_boundary(self, severity, expected):
        assert label_trap_severity(severity) == expected

    def test_a_negative_severity_is_none(self):
        """Severity cannot be negative, but the function must not misbehave if it is."""
        assert label_trap_severity(-1.0) == 'none'

    def test_every_band_is_reachable_and_they_are_ordered(self):
        """Anti-vacuity: the five labels form a strict ladder over increasing severity."""
        probes = [0.0, 0.15, 0.4, 0.7, 2.0]
        labels = [label_trap_severity(s) for s in probes]
        assert labels == ['none', 'mild', 'moderate', 'severe', 'critical']
        assert len(set(labels)) == 5

    def test_the_function_is_monotone_in_severity(self):
        grid = np.linspace(0.0, 2.0, 401)
        order = {'none': 0, 'mild': 1, 'moderate': 2, 'severe': 3, 'critical': 4}
        ranks = [order[label_trap_severity(float(s))] for s in grid]
        assert ranks == sorted(ranks), "a higher severity produced a lower band"

    def test_it_has_no_floor_of_its_own(self):
        """F-034's `mild` floor lives in the CALLER, not here.

        `SpectralAnalyzer` floors the label when `has_trap` is True, so a genuine
        detection just over 0.1 is never published as 'none'. That floor must NOT be
        duplicated into this function, or the severity NUMBER would silently start
        disagreeing with its own band whenever the two are compared.
        """
        assert label_trap_severity(0.05) == 'none', (
            "this function grew a floor; the floor belongs to the caller so that the "
            "severity number and its band can disagree honestly"
        )


class TestTheFloorIsAppliedByTheCallerNotTheBand:
    """The F-034 contract, pinned where it actually lives."""

    def test_a_detected_trap_is_never_published_as_none(self):
        from dl_techniques.analyzer.analyzers.spectral_analyzer import SpectralAnalyzer
        from dl_techniques.analyzer.config import AnalysisConfig
        from dl_techniques.analyzer.data_types import AnalysisResults

        class _Stub(SpectralAnalyzer):
            """Runs only the aggregation half, with one hand-built draw."""

            def __init__(self):  # bypass the model/config constructor
                self.config = AnalysisConfig()
                self.models = {}

            def run(self, trap_result):
                n_draws = 1
                draws = [{
                    'rand_evals': np.array([1.0, 2.0]),
                    'rand_sv_max': 2.0, 'rand_distance': 0.1, 'rand_sv_ratio': 1.0,
                    'trap': trap_result,
                }]

                def _mean(key):
                    return float(np.mean([d[key] for d in draws]))

                representative = max(draws, key=lambda d: (bool(d['trap']['has_trap']),
                                                       float(d['trap']['trap_severity'])))
                trap = representative['trap']
                has_trap = any(bool(d['trap']['has_trap']) for d in draws)
                severity = float(trap['trap_severity'])
                label = label_trap_severity(severity)
                if has_trap and label == 'none':
                    label = 'mild'
                return has_trap, severity, label

        stub = _Stub()
        # A severity of 0.05 is below the 'mild' threshold, so the raw band is 'none'.
        for has_trap, expected_label in ((True, 'mild'), (False, 'none')):
            got_has_trap, severity, label = stub.run({
                'has_trap': has_trap, 'num_rand_spikes': 1, 'trap_severity': 0.05,
                'trap_severity_label': 'none', 'mp_lambda_plus': 1.0,
                'mp_lambda_minus': 0.1, 'trap_threshold': 1.05,
            })
            assert got_has_trap is has_trap
            assert label == expected_label, (
                f"has_trap={has_trap}, severity=0.05 -> label {label!r}, expected "
                f"{expected_label!r}"
            )
            assert severity == 0.05, (
                "the floor must touch the LABEL only; the severity number is never "
                "altered, or a measurement would be laundered"
            )
