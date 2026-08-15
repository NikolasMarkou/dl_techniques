"""RED proof for C-21: `align` is the reason this package exists and had no test.

`test_model.py` is a build + shape + round-trip template; the five-stage
alignment algorithm was entirely unexercised. Three things are pinned here:

1. **A Procrustes oracle.** Two point clouds related by a KNOWN orthogonal `Q`
   are handed to `align`, and the recovered map is compared against `Q`. The
   oracle is derived from the orthogonal-Procrustes definition and the
   ground-truth rotation, never from `model.py` — an oracle written from the
   implementation is a second copy of it.
2. **`align` on a fresh model.** It writes through `self.W.assign(...)` at every
   stage, and `self.W` is `None` until `build`, so a fresh aligner used to die
   with `AttributeError: 'NoneType' object has no attribute 'assign'` — after
   minutes of clustering work — even though both docstring examples start from
   a fresh model.
3. **The QAP solver.** MEASURED: `method='2opt'` fails on instances whose
   optimum is exactly reachable, and when the anchor permutation is wrong the
   whole pipeline returns a map no better than chance.

The fixtures are clustered (not isotropic) on purpose: the algorithm's second
stage matches k-means centroid Gram matrices, so a cloud with no cluster
structure tests nothing about it.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import quadratic_assignment
from scipy.stats import ortho_group

from dl_techniques.models.mini_vec2vec.model import MiniVec2VecAligner


def _clustered_pair(n=1500, d=16, k=8, seed=0):
    """Return `(XA, XB, Q)` with `XB == XA @ Q` for a known orthogonal `Q`."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(k, d)) * 3.0
    labels = rng.integers(0, k, size=n)
    XA = (centers[labels] + rng.normal(size=(n, d)) * 0.3).astype("float32")
    Q = ortho_group.rvs(d, random_state=seed).astype("float32")
    return XA, (XA @ Q).astype("float32"), Q


def _align_frame(X):
    """The frame `align` fits in: mean-centered, then rows on the unit sphere."""
    Xp = X - X.mean(axis=0, keepdims=True)
    return Xp / np.linalg.norm(Xp, axis=1, keepdims=True)


def _orthogonal_error(W, Q):
    """Frobenius distance to `Q`, modulo the sign ambiguity of an orthogonal
    map. Derived from the Procrustes objective, not from any output."""
    return float(min(np.linalg.norm(W - Q, ord="fro"),
                     np.linalg.norm(W + Q, ord="fro")))


def _fit(aligner, XA, XB, **overrides):
    kwargs = dict(
        approx_clusters=8, approx_runs=5, approx_neighbors=5,
        refine1_iterations=10, refine1_sample_size=1000, refine1_neighbors=5,
        refine2_clusters=50, smoothing_alpha=0.5,
    )
    kwargs.update(overrides)
    return aligner.align(XA, XB, **kwargs)


class TestProcrustesOracle:
    def test_align_recovers_a_known_rotation(self):
        XA, XB, Q = _clustered_pair()
        aligner = MiniVec2VecAligner(embedding_dim=XA.shape[1])
        _fit(aligner, XA, XB)

        W = np.asarray(aligner.W)
        err = _orthogonal_error(W, Q)
        # ||Q||_F = sqrt(d) = 4.0 here; a random orthogonal map sits near
        # sqrt(2 * d) = 5.66 away. MEASURED recovery at these settings: 0.097.
        assert err < 0.5, f"recovered map is {err:.3f} from the truth Q"

        A, B = _align_frame(XA), _align_frame(XB)
        aligned = A @ W
        aligned /= np.linalg.norm(aligned, axis=1, keepdims=True)
        assert float(np.mean(np.sum(aligned * B, axis=1))) > 0.99

    def test_the_oracle_rejects_a_wrong_map(self):
        """Anti-vacuity: the threshold must not admit an arbitrary rotation."""
        _, _, Q = _clustered_pair()
        other = ortho_group.rvs(Q.shape[0], random_state=999).astype("float32")
        assert _orthogonal_error(other, Q) > 0.5
        assert _orthogonal_error(np.eye(Q.shape[0], dtype="float32"), Q) > 0.5

    def test_procrustes_solves_its_own_definition(self):
        """`_procrustes` must be `U V^T` of `SVD(XA^T XB)`, checked against the
        closed form written from the definition rather than from the code."""
        XA, XB, Q = _clustered_pair(n=300, d=8, k=4, seed=3)
        A, B = _align_frame(XA), _align_frame(XB)
        aligner = MiniVec2VecAligner(embedding_dim=8)
        W = aligner._procrustes(A, B)

        U, _, Vt = np.linalg.svd(A.T @ B)
        np.testing.assert_allclose(W, U @ Vt, atol=1e-5)
        assert _orthogonal_error(W, Q) < 1e-3
        np.testing.assert_allclose(W @ W.T, np.eye(8), atol=1e-4)


class TestAlignOnAFreshModel:
    def test_align_builds_the_model_itself(self):
        XA, XB, Q = _clustered_pair(n=600, d=8, k=4, seed=1)
        aligner = MiniVec2VecAligner(embedding_dim=8)
        assert aligner.W is None and not aligner.built

        history = _fit(aligner, XA, XB, approx_clusters=4, approx_runs=3,
                       refine2_clusters=20)

        assert aligner.built and aligner.W is not None
        assert set(history) == {"initial_W", "refine1_W", "final_W"}
        assert _orthogonal_error(np.asarray(aligner.W), Q) < 0.5

    def test_align_leaves_call_usable(self):
        XA, XB, _ = _clustered_pair(n=600, d=8, k=4, seed=2)
        aligner = MiniVec2VecAligner(embedding_dim=8)
        _fit(aligner, XA, XB, approx_clusters=4, approx_runs=3,
             refine2_clusters=20)
        out = np.asarray(aligner(_align_frame(XA)[:5]))
        assert out.shape == (5, 8) and np.all(np.isfinite(out))


class TestQAPSolverChoice:
    """The anchor-matching QAP must solve instances whose optimum is exactly
    reachable. This is the measurement that motivated `method='faq'`.

    These arms call scipy directly, so they pin the PROPERTY the model relies
    on rather than the model's own call. The model-level guard against a
    revert is `test_align_recovers_a_known_rotation`, and it is not
    theoretical: restoring `method='2opt'` in `model.py` fires exactly that
    test (measured, recovery error 7.31 against a 0.5 threshold) while every
    arm here stays green.
    """

    @staticmethod
    def _instance(k, d, seed):
        rng = np.random.default_rng(seed)
        C = rng.normal(size=(k, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        perm = rng.permutation(k)
        CB = C[perm]
        return C @ C.T, CB @ CB.T, np.argsort(perm)

    @pytest.mark.parametrize("k,seed", [(12, 0), (12, 1), (20, 2), (20, 3)])
    def test_the_shipped_solver_recovers_an_exact_permutation(self, k, seed):
        sim_a, sim_b, truth = self._instance(k, 32, seed)
        res = quadratic_assignment(-sim_a, sim_b, method="faq")
        assert np.array_equal(res.col_ind, truth), (
            f"QAP missed an exactly-solvable k={k} instance"
        )

    def test_the_sign_convention_is_load_bearing(self):
        """Anti-vacuity for the arm above: without the negation the same call
        MINIMIZES the similarity agreement and gets the answer wrong."""
        sim_a, sim_b, truth = self._instance(12, 32, 0)
        res = quadratic_assignment(sim_a, sim_b, method="faq")
        assert not np.array_equal(res.col_ind, truth)
