"""`DenoisingScoreMatchingLoss.call` must return a PER-SAMPLE vector (D-063).

The class shipped with `return ops.mean(loss)` -- a scalar. `Loss.__call__`
then multiplies that scalar by `sample_weight` and averages, so every per-sample
weight collapses into `mean(sample_weight)` and a weighting the caller believes
is per-sample is silently a global rescale. MEASURED at 9d71a8c4d on the fixture
below: base 1.5745, `sample_weight=[1, 1, 1, 100]` 40.5424, ratio 25.7500 which
is `mean(w)` to four decimals, where the per-sample answer is 59.5290.

This matters because the class docstring itself points a caller who wants
min-SNR timestep weighting at `sample_weight` (D-034 removed the dead
`loss_weight_type` knob on exactly that recommendation), so `sample_weight` is
the one knob on this loss that has to work.

The gradient probe uses **SGD, not Adam**, for the reason recorded in
`tests/test_losses/test_tabm_loss.py`: an Adam-based movement probe normalizes
by gradient magnitude and reports ~1.000x whether or not the weight reached the
objective. Measured pre-fix row3/row0 SGD movement ratio: 0.6745 (the
unweighted ratio -- row 3 carried a 100x weight and moved as if it had 1x).
"""

import numpy as np
import pytest
import keras
import tensorflow as tf
from keras import ops

from dl_techniques.models.nano_vlm_world_model.train import (
    DenoisingScoreMatchingLoss,
    VLMDenoisingLoss,
)


BATCH = 4


def _pair(seed: int = 0, shape=(BATCH, 3)):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=shape).astype("float32"),
            rng.normal(size=shape).astype("float32"))


class TestDsmLossReturnsOneValuePerSample:

    @pytest.mark.parametrize("shape", [(BATCH, 3), (BATCH, 2, 3), (BATCH, 2, 3, 5)])
    def test_call_shape_is_the_batch_axis(self, shape):
        y_true, y_pred = _pair(shape=shape)
        out = DenoisingScoreMatchingLoss().call(
            ops.convert_to_tensor(y_true), ops.convert_to_tensor(y_pred))
        assert tuple(ops.shape(out)) == (BATCH,)

    def test_call_equals_the_per_sample_mse(self):
        y_true, y_pred = _pair()
        got = ops.convert_to_numpy(DenoisingScoreMatchingLoss().call(
            ops.convert_to_tensor(y_true), ops.convert_to_tensor(y_pred)))
        np.testing.assert_allclose(
            got, np.mean((y_pred - y_true) ** 2, axis=1), rtol=1e-6)

    def test_rank_one_inputs_do_not_reduce_to_a_scalar(self):
        """`axis=list(range(1, 1))` is `axis=[]`, which is not a portable
        "reduce nothing"; the rank guard exists for this case."""
        y_true, y_pred = _pair(shape=(BATCH,))
        out = DenoisingScoreMatchingLoss().call(
            ops.convert_to_tensor(y_true), ops.convert_to_tensor(y_pred))
        assert tuple(ops.shape(out)) == (BATCH,)

    def test_unweighted_value_is_unchanged_by_the_repair(self):
        """The fix must not move the loss anyone is already training against.
        Measured 1.5745 at 9d71a8c4d on this exact fixture."""
        y_true, y_pred = _pair()
        value = float(ops.convert_to_numpy(
            DenoisingScoreMatchingLoss()(y_true, y_pred)))
        np.testing.assert_allclose(value, 1.5745, atol=1e-4)


class TestSampleWeightIsNotAGlobalRescale:

    def test_weighted_value_matches_the_per_sample_semantics(self):
        y_true, y_pred = _pair()
        w = np.array([1.0, 1.0, 1.0, 100.0], dtype="float32")
        loss = DenoisingScoreMatchingLoss()
        base = float(ops.convert_to_numpy(loss(y_true, y_pred)))
        weighted = float(ops.convert_to_numpy(
            loss(y_true, y_pred, sample_weight=w)))
        per = np.mean((y_pred - y_true) ** 2, axis=1)
        np.testing.assert_allclose(weighted, np.sum(per * w) / BATCH, rtol=1e-4)
        # RED at 9d71a8c4d: this ratio was exactly mean(w) == 25.75.
        assert abs(weighted / base - float(w.mean())) > 1.0

    def test_zeroing_every_row_but_one_leaves_that_row(self):
        y_true, y_pred = _pair()
        loss = DenoisingScoreMatchingLoss()
        per = np.mean((y_pred - y_true) ** 2, axis=1)
        for row in range(BATCH):
            w = np.zeros(BATCH, dtype="float32")
            w[row] = 1.0
            got = float(ops.convert_to_numpy(
                loss(y_true, y_pred, sample_weight=w)))
            np.testing.assert_allclose(got, per[row] / BATCH, rtol=1e-5)

    def test_sample_weight_reaches_the_backward_pass_PER_ROW(self):
        """SGD, deliberately -- see the module docstring."""
        y_true, y_pred = _pair()
        w = np.array([1.0, 1.0, 1.0, 100.0], dtype="float32")
        loss = DenoisingScoreMatchingLoss()

        def _movement(sample_weight):
            v = tf.Variable(y_pred)
            with tf.GradientTape() as tape:
                value = loss(ops.convert_to_tensor(y_true), v,
                             sample_weight=sample_weight)
            grad = tape.gradient(value, v)
            before = v.numpy().copy()
            keras.optimizers.SGD(learning_rate=1.0).apply_gradients(
                [(tf.convert_to_tensor(grad), v)])
            return np.abs(v.numpy() - before).sum(axis=1)

        unweighted = _movement(None)
        weighted = _movement(ops.convert_to_tensor(w))
        ratio_u = unweighted[3] / unweighted[0]
        ratio_w = weighted[3] / weighted[0]
        # A global rescale leaves the RATIO between rows untouched; that is
        # what shipped (measured 0.6745 both ways).
        np.testing.assert_allclose(ratio_w / ratio_u, 100.0, rtol=1e-3)


class TestTheCombinedLossIsUnaffected:
    """`VLMDenoisingLoss` calls `self.dsm_loss(...)` (the reducing `__call__`),
    so it must still add scalars, not vectors."""

    def test_combined_loss_is_still_a_scalar(self):
        y_true, y_pred = _pair()
        combined = VLMDenoisingLoss()
        out = combined.call(
            {"target_vision": ops.convert_to_tensor(y_true)},
            {"denoised_vision": ops.convert_to_tensor(y_pred)},
        )
        assert tuple(ops.shape(out)) == ()
        assert np.isfinite(float(ops.convert_to_numpy(out)))
