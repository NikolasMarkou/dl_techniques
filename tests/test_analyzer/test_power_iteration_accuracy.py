"""`_power_iteration` was tested only for REPRODUCIBILITY, never for CORRECTNESS.

Why this matters (F-085). `_power_iteration` is public API reachable only by passing
``method='power_iteration'`` to `get_top_eigenvectors`, and **nothing in the repository does** —
the single call site (`summarize_critical_weights`) uses the default `'direct'`. So an accuracy
defect here is invisible to the whole suite, and the only existing test
(`test_analysis_is_reproducible.py::test_power_iteration_is_seedable`) asserts that two calls
with the same seed agree, which a function returning garbage deterministically would also
satisfy.

Two defects were fixed blind in F-028, with no test able to confirm either:

1. Deflation ran ONCE before the loop, so `matrix @ q` reintroduced the previously-converged
   directions on every iteration and the returned vectors were not mutually orthogonal.
2. Convergence tested ``|q_new - q| < tol``, which never fires for a dominant NEGATIVE
   eigenvalue because the iterate flips sign each step — burning all 100 iterations.

This module pins the four properties that make the function trustworthy: the eigenvalues it
reports match `numpy.linalg.eigh`, its eigenvectors are mutually orthogonal, it converges on
a negative-dominant spectrum, and it correctly deflates a repeated eigenvalue. It also pins
the sign convention that makes it consistent with `get_top_eigenvectors`.
"""

import numpy as np
import pytest

from dl_techniques.analyzer.spectral_metrics import (
    _power_iteration,
    get_top_eigenvectors,
)
from dl_techniques.analyzer.utils import make_rng


def _symmetric_indefinite(n=12, seed=0, spread=1.0):
    """A symmetric matrix with WELL-SEPARATED eigenvalues, half of them negative.

    Built from prescribed eigenvalues via `Q Λ Qᵀ` rather than from a random symmetric
    matrix. A random one has a clustered spectrum, and power iteration converges slowly on
    close eigenvalues — MEASURED on a clustered 14x14 case: the third-largest eigenvalue was
    3.6% off after 100 iterations. That is a property of the ALGORITHM, not a defect, so
    this fixture isolates correctness from convergence rate.
    """
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    # Alternating signs, magnitudes spread over a decade.
    eigenvalues = np.array([(-1.0) ** i * (10.0 ** (i / 4.0))
                            for i in range(n)])
    return (Q @ np.diag(eigenvalues) @ Q.T) * spread


class TestTheEigenvaluesAreCorrect:
    def test_it_matches_numpy_on_a_psd_matrix(self):
        """PSD is the case the analyzer's own caller would present (a Gram matrix)."""
        rng = np.random.default_rng(1)
        B = rng.normal(size=(14, 8))
        M = B @ B.T                                    # PSD by construction
        values, vectors = _power_iteration(M, k=3, rng=make_rng(7))

        reference = np.sort(np.linalg.eigvalsh(M))[::-1][:3]
        np.testing.assert_allclose(np.sort(values), np.sort(reference),
                                   rtol=1e-6, atol=1e-8)

    def test_it_matches_numpy_on_a_symmetric_indefinite_matrix(self):
        """PSD-only coverage would miss the sign-flip defect entirely."""
        M = _symmetric_indefinite(n=12, seed=2)
        values, _ = _power_iteration(M, k=3, rng=make_rng(7))
        # Magnitude ordering, matching the function's convention (see the test below).
        reference = sorted(np.linalg.eigvalsh(M), key=abs, reverse=True)[:3]
        np.testing.assert_allclose(sorted(values, key=abs, reverse=True),
                                   sorted(reference, key=abs, reverse=True),
                                   rtol=1e-6, atol=1e-8)

    def test_it_orders_by_magnitude_not_algebraically(self):
        """The ordering convention, which is not the same thing.

        Power iteration selects the largest-MAGNITUDE eigenvalues; `eigvalsh` sorts
        ALGEBRAICALLY. On a symmetric INDEFINITE matrix the two orderings differ, which is
        why the comparisons above sort both sides rather than reversing one. This is the
        right convention for this function's actual caller, which wants top singular
        values of a Gram matrix.
        """
        M = np.diag([-9.0, 1.0, 4.0, 0.25])
        values, _ = _power_iteration(M, k=2, rng=make_rng(37))
        algebraic_top2 = sorted(np.linalg.eigvalsh(M), reverse=True)[:2]
        magnitude_top2 = sorted(np.linalg.eigvalsh(M), key=abs, reverse=True)[:2]
        assert algebraic_top2 != magnitude_top2, (
            "the probe no longer distinguishes the two orderings"
        )
        np.testing.assert_allclose(sorted(values, key=abs, reverse=True),
                                   magnitude_top2, rtol=1e-4, atol=1e-4)

    def test_it_recovers_the_largest_magnitude_eigenvalue(self):
        M = _symmetric_indefinite(n=10, seed=3)
        values, _ = _power_iteration(M, k=1, rng=make_rng(11))
        largest_magnitude = sorted(np.linalg.eigvalsh(M), key=abs, reverse=True)[0]
        assert values[0] == pytest.approx(largest_magnitude, rel=1e-6)

    def test_the_rayleigh_quotient_is_an_eigenvalue(self):
        """Each returned value must satisfy λ = qᵀMq for ITS OWN vector.

        This is the invariant that makes the value/vector pairing meaningful; matching the
        sorted eigenvalue list alone would not catch a swapped pairing.
        """
        M = _symmetric_indefinite(n=10, seed=4)
        values, vectors = _power_iteration(M, k=3, rng=make_rng(5))
        for value, vector in zip(values, vectors.T):
            quotient = float(vector @ (M @ vector))
            assert quotient == pytest.approx(float(value), rel=1e-6)


class TestTheEigenvectorsAreOrthogonal:
    """The F-028 fix. The pre-fix code collapsed all k vectors onto one eigenvector."""

    @staticmethod
    def _off_diagonal(gram):
        return float(np.abs(gram - np.diag(np.diag(gram))).max())

    def test_orthogonality_tracks_the_convergence_tolerance(self):
        """The sharpest available statement, and it is a very sharp one.

        MEASURED on a 14x14 matrix at `k=5`, worst off-diagonal Gram entry by `tol`:
        1e-05 -> 5.97e-06,  1e-08 -> 5.97e-09,  1e-12 -> 5.97e-13,  1e-15 -> 6.39e-16.
        Proportional to `tol` across six orders of magnitude, because the loop breaks the
        moment the subspace projector stops moving by more than `tol`, and the residual
        coupling is of that order.

        So the honest bound at the default is `tol` (1e-5), NOT machine precision. An
        earlier draft of this module asserted `atol=1e-8` at the default settings and
        failed at 5.2e-06 — the assertion was wrong, not the code. Asserting the
        `tol`-proportionality instead makes the relationship the contract.
        """
        M = _symmetric_indefinite(n=14, seed=7)
        for tol in (1e-5, 1e-8, 1e-12):
            _, vectors = _power_iteration(M, k=5, tol=tol, rng=make_rng(19))
            worst = self._off_diagonal(vectors.T @ vectors)
            assert worst < 100 * tol, (
                f"tol={tol:g} gave an off-diagonal of {worst:.3g}, which is more than "
                f"100x the tolerance the iteration stopped at"
            )
            assert worst >= 0.01 * tol, (
                f"tol={tol:g} gave {worst:.3g} — suspiciously far below the tolerance; "
                f"the deflation may have been dropped entirely, which would make the "
                f"vectors accidentally exact rather than deliberately orthogonal"
            )

    def test_it_reaches_machine_precision_when_asked_to(self):
        _, vectors = _power_iteration(_symmetric_indefinite(n=14, seed=7), k=5,
                                      tol=1e-15, rng=make_rng(19))
        np.testing.assert_allclose(vectors.T @ vectors, np.eye(5), atol=1e-14)

    def test_each_vector_is_unit_norm(self):
        M = _symmetric_indefinite(n=10, seed=6)
        _, vectors = _power_iteration(M, k=4, rng=make_rng(17))
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=0), np.ones(4),
                                   atol=1e-12)

    def test_a_few_iterations_are_not_enough_but_a_full_budget_is(self):
        """In-loop deflation accumulates: the basis sharpens with `max_iter`.

        MEASURED at `tol=1e-14`, `k=5`, off-diagonal: 0.76 (max_iter=1) -> 0.96 (3) ->
        0.019 (10) -> 6.1e-15 (100). Non-monotone early on, as expected — deflation needs
        several passes to purge a direction.

        The trend is the discriminator. A pre-F-028 implementation, which deflated once
        before the loop, would let every start vector converge to the SAME dominant
        eigenvector and sit near 1.0 at ANY budget.
        """
        M = _symmetric_indefinite(n=14, seed=7)

        def worst_off_diagonal(max_iter):
            _, vectors = _power_iteration(M, k=5, tol=1e-14, max_iter=max_iter,
                                          rng=make_rng(19))
            return self._off_diagonal(vectors.T @ vectors)

        assert worst_off_diagonal(1) > 0.1, "three iterations should not suffice"
        assert worst_off_diagonal(100) < 1e-12, (
            f"a full budget gave {worst_off_diagonal(100):.3g}; deflation is being "
            f"applied outside the loop, which is the F-028 defect"
        )

    def test_a_degenerate_matrix_does_not_produce_a_singular_basis(self):
        """The identity has every eigenvalue equal; any three columns will do."""
        _, vectors = _power_iteration(np.eye(8), k=3, tol=1e-14, rng=make_rng(23))
        gram = vectors.T @ vectors
        assert self._off_diagonal(gram) < 1e-12, (
            f"on an identity matrix the basis collapsed (off-diagonal "
            f"{self._off_diagonal(gram):.3g}) — deflation cannot separate degenerate "
            f"directions if it is not applied every iteration"
        )


class TestConvergenceOnANegativeDominantSpectrum:
    """The second F-028 fix: `|q_new - q|` never converges when the top eigenvalue is negative."""

    def test_it_converges_on_a_negative_dominant_matrix(self):
        # -5 and -1 dominate; the largest in ALGEBRAIC order is the least negative, so
        # use the largest-magnitude direction, which is -5.
        M = np.diag([-5.0, -1.0, 2.0, 0.5, -3.0])
        values, vectors = _power_iteration(M, k=2, rng=make_rng(29), max_iter=200)

        assert np.all(np.isfinite(values))
        assert np.all(np.isfinite(vectors))
        # The two most negative eigenvalues dominate the iteration.
        np.testing.assert_allclose(np.sort(np.abs(values))[::-1][:2],
                                   [5.0, 3.0], rtol=1e-4, atol=1e-4)

    def test_the_sign_of_the_iterate_does_not_prevent_convergence(self):
        """A sign-flipping iterate must still settle on a fixed subspace.

        The subspace projector `|q qᵀ|` is sign-invariant, which is the point of the
        F-028 change; a vector-distance test would churn here forever.
        """
        M = np.diag([-4.0, 1.0, 0.5])
        values, vectors = _power_iteration(M, k=1, rng=make_rng(31), max_iter=200)
        assert values[0] == pytest.approx(-4.0, rel=1e-4), (
            f"got {values[0]!r}; a dominant negative eigenvalue must still be found"
        )


class TestItAgreesWithGetTopEigenvectors:
    """The two methods must agree on the SUBSPACE; only the sign may differ.

    Note the comparison: `P_direct @ P_power` must equal `P_direct`, NOT the identity.
    Both P's are rank-`k` projectors with `k=3` in a 20-dimensional space, so `P1 @ P2 - I`
    is O(1) for perfectly correct vectors — an earlier draft of this module compared
    against `np.eye(20)` and reported a 0.98 deviation on output that was in fact correct
    to 1e-13.
    """

    @staticmethod
    def _pair(seed):
        W = np.random.default_rng(200 + seed).normal(size=(20, 12))
        _, direct = get_top_eigenvectors(W, k=3, method="direct")
        return W, direct

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_the_spans_agree_once_converged(self, seed):
        W, direct = self._pair(seed)
        _, power = _power_iteration(W @ W.T, k=3, tol=1e-14, max_iter=1000,
                                    rng=make_rng(43))
        direct_projector = direct @ direct.T
        power_projector = power @ power.T
        np.testing.assert_allclose(direct_projector @ power_projector,
                                   direct_projector, atol=1e-10)
        # And the power-iteration result is itself a projector: idempotent.
        np.testing.assert_allclose(power_projector @ power_projector, power_projector,
                                   atol=1e-10)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_the_vectors_agree_up_to_a_sign(self, seed):
        W, direct = self._pair(seed)
        _, power = _power_iteration(W @ W.T, k=3, tol=1e-14, max_iter=1000,
                                    rng=make_rng(47))
        cosines = np.abs(np.sum(direct * power, axis=0))
        np.testing.assert_allclose(cosines, np.ones(3), atol=1e-6)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_at_default_settings_the_agreement_is_governed_by_clustering(self, seed):
        """The realistic accuracy ceiling, and its cause.

        `get_top_eigenvectors(method='power_iteration')` uses the module defaults
        `tol=1e-5, max_iter=100`. On a 20x12 Gaussian the top singular values are
        Marchenko-Pastur clustered, and convergence rate goes as `(λ3/λ1)^(2n)`, so a
        tight third eigenvalue costs iterations. MEASURED span deviation
        `max|P_direct P_power - P_direct|`:

            seed 0: λ3/λ1 = 1.147 -> 2.9e-03
            seed 1: λ3/λ1 = 1.343 -> 5.3e-06
            seed 2: λ3/λ1 = 1.515 -> 1.2e-04

        Perfectly anti-correlated with the gap ratio, and each is driven to ~1e-15 by
        `tol=1e-14, max_iter=1000` (see the tests above). So the default is usable but
        its accuracy is a property of the SPECTRUM, not of the iteration cap — which is
        precisely why nothing in the analyzer may depend on it: the single call site
        (`summarize_critical_weights` via `_describe_model`) uses `method='direct'`.
        """
        W, direct = self._pair(seed)
        direct_projector = direct @ direct.T
        _, power = get_top_eigenvectors(W, k=3, method="power_iteration",
                                        rng=make_rng(43))
        deviation = float(np.abs(
            direct_projector @ (power @ power.T) - direct_projector).max())
        assert deviation < 5e-3, (
            f"span deviation {deviation:.3g} is worse than the measured ceiling for "
            f"this clustering; the deflation or the convergence test has regressed"
        )
        direct_values, _ = get_top_eigenvectors(W, k=3, method="direct")
        power_values, _ = get_top_eigenvectors(W, k=3, method="power_iteration",
                                               rng=make_rng(41))
        np.testing.assert_allclose(np.sort(direct_values)[::-1],
                                   np.sort(power_values)[::-1],
                                   rtol=5e-3, atol=1e-4)


class TestReproducibilityIsPreserved:
    """The property that WAS tested before; kept so F-028 did not cost it."""

    def test_the_same_seed_gives_the_same_answer(self):
        M = _symmetric_indefinite(n=10, seed=8)
        first = _power_iteration(M, k=2, rng=make_rng(3))
        second = _power_iteration(M, k=2, rng=make_rng(3))
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    def test_different_seeds_give_the_same_spectrum(self):
        """Start vectors differ, so SIGNS may differ, but eigenvalues must not."""
        M = _symmetric_indefinite(n=10, seed=9)
        first = _power_iteration(M, k=3, rng=make_rng(1))[0]
        second = _power_iteration(M, k=3, rng=make_rng(2))[0]
        np.testing.assert_allclose(np.sort(first), np.sort(second), atol=1e-8)

    def test_an_unseeded_call_still_works(self):
        M = _symmetric_indefinite(n=8, seed=10)
        values, vectors = _power_iteration(M, k=2)
        assert np.all(np.isfinite(values)) and np.all(np.isfinite(vectors))


class TestDegenerateInput:
    def test_k_larger_than_the_matrix_still_returns_finite_output(self):
        M = _symmetric_indefinite(n=6, seed=11)
        values, vectors = _power_iteration(M, k=6, rng=make_rng(3))
        assert values.shape == (6,)
        assert vectors.shape == (6, 6)
        assert np.all(np.isfinite(values))

    def test_a_zero_matrix_gives_zero_eigenvalues(self):
        values, _ = _power_iteration(np.zeros((5, 5)), k=2, rng=make_rng(3))
        np.testing.assert_allclose(values, np.zeros(2), atol=1e-12)

    def test_it_breaks_out_on_a_zero_norm_iterate(self):
        """`z_norm < SPECTRAL_EPSILON` must exit rather than divide by it."""
        M = np.zeros((5, 5))
        values, vectors = _power_iteration(M, k=1, rng=make_rng(3))
        assert np.all(np.isfinite(vectors))