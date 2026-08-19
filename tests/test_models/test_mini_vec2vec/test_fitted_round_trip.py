"""F-69: round-trip a FITTED ``W``, not the ``Identity()``-initialized one.

``test_model.py::test_keras_round_trip`` saves a model straight out of
``create_mini_vec2vec_aligner``. ``W`` is identity-initialized in ``build``, so
that comparison is satisfied **identically** by a model that dropped ``W`` and
re-ran its initializer: identity in, identity out. ``test_smoke.py``'s
``test_the_untrained_aligner_is_the_identity`` asserts identity on purpose, and
``test_align.py`` never touches serialization. ``W`` is this package's single
load-bearing weight, and a user who ran a 30-round QAP plus a 75-iteration
refinement would lose the entire fit with no error and no red test.

MEASURED (dead-component injection, ``load_own_variables`` stubbed to a no-op on
``MiniVec2VecAligner`` so a loaded model keeps its fresh identity ``W`` — the
literal F-69 failure mode): ``test_model.py::test_keras_round_trip`` stays
GREEN, ``test_smoke.py::test_the_untrained_aligner_is_the_identity`` stays
GREEN — 16 pre-existing tests, all green — while the 3 round-trip assertions in
this file go RED (the 2 that stay green are this file's own controls: the
fixture-quality check, and the pre-existing untrained round trip kept here to
record why F-69 was invisible).

The fit here is a small clustered pair with a KNOWN rotation, reusing
``test_align.py``'s settings so the whole file stays inside a unit-test budget.
"""

import os

import keras
import numpy as np
import pytest
from scipy.stats import ortho_group

from dl_techniques.models.mini_vec2vec.model import (
    MiniVec2VecAligner,
    create_mini_vec2vec_aligner,
)

DIM = 8


def _clustered_pair(n=600, d=DIM, k=4, seed=1):
    """``(XA, XB, Q)`` with ``XB == XA @ Q`` for a known orthogonal ``Q``."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(k, d)) * 3.0
    labels = rng.integers(0, k, size=n)
    XA = (centers[labels] + rng.normal(size=(n, d)) * 0.3).astype("float32")
    Q = ortho_group.rvs(d, random_state=seed).astype("float32")
    return XA, (XA @ Q).astype("float32"), Q


@pytest.fixture(scope="module")
def fitted():
    """An aligner whose ``W`` has actually been fitted, plus its inputs."""
    XA, XB, Q = _clustered_pair()
    aligner = MiniVec2VecAligner(embedding_dim=DIM)
    aligner.align(
        XA, XB,
        approx_clusters=4, approx_runs=5, approx_neighbors=5,
        refine1_iterations=10, refine1_sample_size=300, refine1_neighbors=5,
        refine2_clusters=20, smoothing_alpha=0.5,
    )
    return aligner, XA, XB, Q


def _np(t) -> np.ndarray:
    return keras.ops.convert_to_numpy(t)


class TestTheFitIsWorthSaving:
    """Non-vacuity control, run BEFORE the round trip is asserted.

    If the fitted ``W`` were still (near) the identity, every assertion in this
    file would be satisfied by the very failure it exists to detect.
    """

    def test_the_fitted_matrix_is_far_from_the_initializer(self, fitted):
        aligner, _, _, _ = fitted
        W = _np(aligner.W)
        eye = np.eye(DIM, dtype="float32")

        distance = float(np.linalg.norm(W - eye, ord="fro"))
        assert distance > 1.0, (
            f"the fitted W is only {distance:.4f} from the identity "
            "initializer; this fixture cannot discriminate a lost fit"
        )


class TestFittedRoundTrip:

    def test_the_fitted_W_survives_a_keras_round_trip(self, fitted, tmp_path):
        aligner, XA, _, _ = fitted
        before_W = _np(aligner.W).copy()

        path = os.path.join(str(tmp_path), "fitted_aligner.keras")
        aligner.save(path)
        loaded = keras.models.load_model(path)

        after_W = _np(loaded.W)
        assert not np.allclose(after_W, np.eye(DIM), atol=1e-3), (
            "the reloaded W is the identity — the fit was silently replaced by "
            "a fresh run of the initializer (F-69's failure mode)"
        )
        np.testing.assert_allclose(
            before_W, after_W, atol=0, rtol=0,
            err_msg="the fitted transformation matrix W was not restored",
        )

    def test_the_reloaded_model_transforms_identically(self, fitted, tmp_path):
        aligner, XA, _, _ = fitted
        x = XA[:16]
        before = _np(aligner(x, training=False))

        path = os.path.join(str(tmp_path), "fitted_aligner_fwd.keras")
        aligner.save(path)
        after = _np(keras.models.load_model(path)(x, training=False))

        # Non-vacuity: the transform must not be the identity map, or this
        # would pass on a model that lost W entirely.
        assert not np.allclose(before, x, atol=1e-3), (
            "the fitted aligner is the identity map on this input"
        )
        np.testing.assert_allclose(
            before, after, atol=1e-6,
            err_msg="the reloaded aligner does not reproduce the fitted map",
        )

    def test_the_reloaded_model_still_recovers_the_known_rotation(
        self, fitted, tmp_path
    ):
        """The end-to-end claim: the fit, not just the bits, comes back."""
        aligner, XA, XB, Q = fitted

        path = os.path.join(str(tmp_path), "fitted_aligner_q.keras")
        aligner.save(path)
        loaded = keras.models.load_model(path)

        W = _np(loaded.W)
        err = float(min(
            np.linalg.norm(W - Q, ord="fro"),
            np.linalg.norm(W + Q, ord="fro"),
        ))
        # ||Q||_F = sqrt(d); a random orthogonal map sits near sqrt(2d).
        assert err < 0.5, (
            f"the reloaded W is {err:.3f} from the ground-truth rotation "
            f"(a fresh identity would be {float(min(np.linalg.norm(np.eye(DIM) - Q, ord='fro'), np.linalg.norm(np.eye(DIM) + Q, ord='fro'))):.3f})"
        )

    def test_an_untrained_round_trip_cannot_discriminate(self, tmp_path):
        """Why the fixture above is fitted: this is the pre-existing test.

        It passes here, and it also passes when ``W`` is not restored at all.
        Kept so the reason F-69 was invisible is recorded next to its fix.
        """
        model = create_mini_vec2vec_aligner(embedding_dim=DIM)
        x = np.random.default_rng(0).random((2, DIM)).astype("float32")
        before = _np(model(x, training=False))

        path = os.path.join(str(tmp_path), "unfitted.keras")
        model.save(path)
        after = _np(keras.models.load_model(path)(x, training=False))

        np.testing.assert_allclose(before, after, atol=1e-4)
        # ...and here is the reason that proves nothing:
        np.testing.assert_allclose(_np(model.W), np.eye(DIM), atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
