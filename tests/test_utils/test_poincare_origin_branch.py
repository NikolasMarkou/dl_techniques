"""C-45(a): the `tanh(x)/x -> 1` limiting branch must be REACHABLE.

``safe_norm`` already clamps its result to ``eps``, so the subsequent
``where(norm_v < self.eps, ...)`` in ``exp_map_0`` / ``log_map_0`` tested
``eps < eps`` -- **never true**. The limiting branches documented at the origin
could never be selected, so the stated protection was not the protection in
force.

This is LATENT, not a wrong answer: ``tanh(sqrt(c) * 1e-5) / (sqrt(c) * 1e-5)``
is ~1, so both paths agree numerically. A value comparison therefore proves
nothing, and the guard has to observe the SELECTOR. It does that by poisoning
the branch that must NOT be taken: with ``ops.tanh`` (resp. ``ops.arctanh``)
returning ``nan``, the scaled path is ``0 * nan = nan``, so a finite result at
the origin is possible only if the limiting branch fired.

An earlier review claimed NaN GRADIENTS from this function at the origin. That
claim was WITHDRAWN after measurement (TF 2.18's ``tf.norm`` backward returns 0,
not NaN), and it holds off XLA only: under ``jit_compile=True`` sqrt's backward
is ``0.5 * dy / y``, so ``dy = 0`` at ``y = 0`` is NaN, and every SHGCN model
trained under XLA had a NaN loss from its second step
(plan-2026-10-14T062616-81938e81 D-155). ``TestTheOriginGradientIsFiniteUnderXla``
pins that, with a liveness arm showing the bare form really is NaN there.
"""

import numpy as np
import keras
import tensorflow as tf
import pytest

from dl_techniques.utils.geometry import poincare_math
from dl_techniques.utils.geometry.poincare_math import PoincareMath


@pytest.fixture
def math_util() -> PoincareMath:
    return PoincareMath(eps=1e-5)


class TestTheRawNormIsWhatTheBranchTests:

    def test_the_helper_returns_both_norms(self, math_util):
        zeros = np.zeros((3, 4), dtype="float32")
        raw, floored = math_util.norm_and_floored_norm(
            zeros, axis=-1, keepdims=True)

        assert float(np.max(np.asarray(raw))) == 0.0, (
            "ASSERT-RAW-NORM-IS-RAW: the first element must NOT be floored, or "
            "the branch test is `eps < eps` again."
        )
        assert float(np.min(np.asarray(floored))) == pytest.approx(1e-5)

    def test_safe_norm_still_floors(self, math_util):
        """Anti-vacuity: the existing contract is unchanged."""
        zeros = np.zeros((3, 4), dtype="float32")
        assert float(np.min(np.asarray(math_util.safe_norm(zeros, keepdims=True)))) \
            == pytest.approx(1e-5)

    def test_the_condition_is_true_at_the_origin(self, math_util):
        zeros = np.zeros((3, 4), dtype="float32")
        raw, _ = math_util.norm_and_floored_norm(zeros, axis=-1, keepdims=True)
        assert bool(np.all(np.asarray(raw) < math_util.eps))


class TestTheLimitingBranchActuallyFires:
    """Poison the scaled path; a finite origin result proves the branch ran."""

    def test_exp_map_0_takes_the_limit_at_the_origin(
            self, math_util, monkeypatch):
        monkeypatch.setattr(
            keras.ops, "tanh",
            lambda x: keras.ops.multiply(keras.ops.zeros_like(x), float("nan")))

        result = np.asarray(math_util.exp_map_0(
            np.zeros((3, 4), dtype="float32"), 1.0))

        assert np.all(np.isfinite(result)), (
            "ASSERT-EXP-LIMIT-BRANCH-FIRES: with tanh poisoned, a finite result "
            "at the origin is only reachable through the `v` branch. NaN here "
            "means the branch is still dead."
        )
        np.testing.assert_array_equal(result, np.zeros((3, 4), dtype="float32"))

    def test_log_map_0_takes_the_limit_at_the_origin(
            self, math_util, monkeypatch):
        monkeypatch.setattr(
            keras.ops, "arctanh",
            lambda x: keras.ops.multiply(keras.ops.zeros_like(x), float("nan")))

        result = np.asarray(math_util.log_map_0(
            np.zeros((3, 4), dtype="float32"), 1.0))

        assert np.all(np.isfinite(result)), (
            "ASSERT-LOG-LIMIT-BRANCH-FIRES: with arctanh poisoned, a finite "
            "result at the origin is only reachable through the `y` branch."
        )

    def test_the_poison_really_is_poison(self, math_util, monkeypatch):
        """Liveness arm: away from the origin the poisoned path must show."""
        monkeypatch.setattr(
            keras.ops, "tanh",
            lambda x: keras.ops.multiply(keras.ops.zeros_like(x), float("nan")))

        away = np.full((3, 4), 0.5, dtype="float32")
        result = np.asarray(math_util.exp_map_0(away, 1.0))

        assert np.all(np.isnan(result)), (
            "the monkeypatch is not reaching the code under test, so the two "
            "assertions above would pass vacuously"
        )


class TestForwardValuesAreUnchanged:
    """The fix is latent by construction -- prove it changed no answer."""

    def test_exp_log_round_trip(self, math_util):
        rng = np.random.default_rng(0)
        x = (rng.normal(size=(16, 8)) * 0.1).astype("float32")
        recovered = np.asarray(
            math_util.log_map_0(math_util.exp_map_0(x, 1.0), 1.0))
        np.testing.assert_allclose(recovered, x, atol=1e-6)

    def test_the_origin_still_maps_to_the_origin(self, math_util):
        zeros = np.zeros((3, 4), dtype="float32")
        np.testing.assert_array_equal(
            np.asarray(math_util.exp_map_0(zeros, 1.0)), zeros)


class TestTheOriginGradientIsFiniteUnderXla:
    """D-155: sqrt's backward at a zero vector is NaN under XLA, 0 eager.

    A zero-initialised SHGCN bias sits at the origin on step 1, and the floor
    and the ``where`` in front of the norm send ``dy = 0`` into sqrt's
    backward. TF's SqrtGrad kernel returns 0 for ``dy == 0``; XLA computes
    ``0.5 * 0 / 0 = nan``. The gradient must be finite, and exactly 0, in both
    modes.
    """

    @staticmethod
    def _gradient(fn, jit_compile):
        def grad(x):
            with tf.GradientTape() as tape:
                tape.watch(x)
                loss = tf.reduce_sum(fn(x))
            return tape.gradient(loss, x)
        return np.asarray(tf.function(grad, jit_compile=jit_compile)(
            tf.zeros((3, 4), dtype="float32")))

    @pytest.mark.parametrize("jit_compile", [False, True])
    @pytest.mark.parametrize("which", [
        "raw_norm", "floored_norm", "exp_map_0", "log_map_0", "project"])
    def test_the_gradient_at_the_origin_is_zero(
            self, math_util, which, jit_compile):
        fns = {
            "raw_norm": lambda x: math_util.norm_and_floored_norm(x)[0],
            "floored_norm": lambda x: math_util.norm_and_floored_norm(x)[1],
            "exp_map_0": lambda x: math_util.exp_map_0(x, 1.0),
            "log_map_0": lambda x: math_util.log_map_0(x, 1.0),
            "project": lambda x: math_util.project(x, 1.0),
        }
        grad = self._gradient(fns[which], jit_compile)
        assert np.all(np.isfinite(grad)), (which, jit_compile, grad)
        if which in ("raw_norm", "floored_norm"):
            np.testing.assert_array_equal(grad, np.zeros((3, 4), "float32"))

    def test_the_bare_norm_really_is_nan_under_xla(self):
        """Liveness: the instrument sees the defect the fix removes."""
        bare = lambda x: keras.ops.maximum(keras.ops.norm(x, axis=-1), 1e-5)  # noqa: E731
        assert np.all(self._gradient(bare, False) == 0.0)
        assert np.all(np.isnan(self._gradient(bare, True)))
