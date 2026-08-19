"""Tests for `dl_techniques.losses.tabm_loss.TabMLoss`.

This loss had no test file at all before 2026-08-18 (`grep -rl TabMLoss tests/`
returned nothing), which is how F-68 survived: `call` returned a tensor whose
leading axis was `batch * k`, not `batch`, so `sample_weight` / `class_weight`
could not broadcast against it.

The `class_weight` probe deliberately uses **SGD, not Adam**. This repo has
MEASURED that a total-|dW| probe is blind to class weighting under Adam, because
Adam normalizes by the gradient magnitude. Re-measured on THIS probe, one epoch,
1 positive in 32 rows, `class_weight={0: 1.0, 1: 100.0}` versus none:

    SGD(0.5):   plain 0.5578  weighted 7.7787  ratio 13.945
    Adam(0.5):  plain 7.4980  weighted 7.4999  ratio  1.000

So an Adam-based probe reports 1.000 whether or not the weight reaches the
objective at all. Do not switch this test to Adam.
"""

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.losses.tabm_loss import TabMLoss


def _predictions(batch: int = 6, k: int = 3, outputs: int = 1, seed: int = 0):
    rng = np.random.default_rng(seed)
    y_pred = rng.uniform(0.05, 0.95, size=(batch, k, outputs)).astype("float32")
    y_true = rng.integers(0, 2, size=(batch, 1)).astype("float32")
    return y_true, y_pred


class TestTabMLossShape:
    """`call` must reduce the ensemble axis, leaving one value per input row."""

    @pytest.mark.parametrize("base", ["mse", "binary_crossentropy"])
    @pytest.mark.parametrize("k", [1, 3, 5])
    def test_call_returns_one_value_per_row(self, base, k):
        y_true, y_pred = _predictions(batch=7, k=k)
        loss = TabMLoss(base)
        out = loss.call(ops.convert_to_tensor(y_true), ops.convert_to_tensor(y_pred))
        assert tuple(ops.shape(out)) == (7,), (
            f"call() returned shape {tuple(ops.shape(out))} for batch=7, k={k}; "
            f"anything but (7,) is not the axis Keras weights against"
        )

    def test_a_loss_instance_base_loss_is_also_per_row(self):
        """The scalar shape F-68 described occurs only for a `Loss` INSTANCE."""
        y_true, y_pred = _predictions(batch=7, k=3)
        loss = TabMLoss(keras.losses.BinaryCrossentropy())
        out = loss.call(
            ops.convert_to_tensor(y_true), ops.convert_to_tensor(y_pred)
        )
        assert tuple(ops.shape(out)) == (7,)

    def test_value_is_the_mean_over_ensemble_members(self):
        y_true, y_pred = _predictions(batch=4, k=3)
        loss = TabMLoss("mse")
        out = ops.convert_to_numpy(
            loss.call(ops.convert_to_tensor(y_true), ops.convert_to_tensor(y_pred))
        )
        expected = np.mean((y_pred - y_true[:, None, :]) ** 2, axis=(1, 2))
        np.testing.assert_allclose(out, expected, atol=1e-6)


class TestTabMLossSampleWeighting:
    """The defect F-68 names, measured directly on `Loss.__call__`."""

    def test_sample_weight_does_not_raise(self):
        """RED at HEAD: `InvalidArgumentError: required broadcastable shapes`."""
        y_true, y_pred = _predictions(batch=6, k=3)
        loss = TabMLoss("binary_crossentropy")
        w = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 100.0], dtype="float32")
        value = float(ops.convert_to_numpy(loss(
            ops.convert_to_tensor(y_true),
            ops.convert_to_tensor(y_pred),
            sample_weight=ops.convert_to_tensor(w),
        )))
        assert np.isfinite(value)

    def test_sample_weight_reaches_the_named_row(self):
        """Up-weighting one row must move the loss by that row's own amount.

        This is the assertion a scalar `call()` cannot satisfy: with a scalar,
        `reduce_weighted_values` gives `scalar * mean(weight)` for every row
        alike, so the answer would be independent of WHICH row is weighted.
        """
        y_true, y_pred = _predictions(batch=6, k=3)
        loss = TabMLoss("binary_crossentropy")
        per_row = ops.convert_to_numpy(
            loss.call(ops.convert_to_tensor(y_true), ops.convert_to_tensor(y_pred))
        )
        heaviest = int(np.argmax(per_row))
        lightest = int(np.argmin(per_row))
        assert heaviest != lightest

        def weighted(idx):
            w = np.ones((6,), dtype="float32")
            w[idx] = 100.0
            return float(ops.convert_to_numpy(loss(
                ops.convert_to_tensor(y_true),
                ops.convert_to_tensor(y_pred),
                sample_weight=ops.convert_to_tensor(w),
            )))

        assert weighted(heaviest) > weighted(lightest) * 1.5, (
            "weighting the highest-loss row and the lowest-loss row gave "
            "comparable totals -- the weight is not reaching individual rows"
        )


class TestTabMLossClassWeightUnderSGD:
    """`class_weight` in `fit()` must change the weights it is supposed to.

    SGD is mandatory here (see the module docstring): Adam's per-parameter
    normalization hides the weighting from a total-|dW| probe.
    """

    @staticmethod
    def _build(k: int = 3, seed: int = 7):
        keras.utils.set_random_seed(seed)
        inp = keras.Input(shape=(4,))
        h = keras.layers.Dense(k, activation="sigmoid",
                               kernel_initializer="he_normal")(inp)
        out = keras.layers.Reshape((k, 1))(h)
        return keras.Model(inp, out)

    def _train_delta(self, class_weight):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(32, 4)).astype("float32")
        y = np.zeros((32, 1), dtype="float32")
        y[0] = 1.0  # one positive in 32 rows
        model = self._build()
        before = [np.array(w) for w in model.get_weights()]
        model.compile(
            optimizer=keras.optimizers.SGD(learning_rate=0.5),
            loss=TabMLoss("binary_crossentropy"),
        )
        model.fit(x, y, epochs=1, batch_size=32, verbose=0,
                  class_weight=class_weight)
        after = model.get_weights()
        return sum(float(np.abs(a - b).sum()) for a, b in zip(after, before))

    def test_class_weight_changes_the_update_magnitude(self):
        plain = self._train_delta(None)
        weighted = self._train_delta({0: 1.0, 1: 100.0})
        assert plain > 0.0, "control failed: SGD did not move the weights at all"
        ratio = weighted / plain
        assert ratio > 2.0, (
            f"class_weight={{0: 1.0, 1: 100.0}} on a 1-positive-in-32 batch moved "
            f"the weights only {ratio:.3f}x as far as no weighting at all -- the "
            f"per-class weight is not reaching the objective"
        )


class TestTabMLossSerialization:
    """`get_config` round trip, for both base-loss spellings."""

    @pytest.mark.parametrize("base", ["mse", "binary_crossentropy"])
    def test_round_trip_preserves_value(self, base):
        y_true, y_pred = _predictions()
        loss = TabMLoss(base, share_training_batches=True)
        rebuilt = TabMLoss.from_config(loss.get_config())
        a = ops.convert_to_numpy(loss.call(
            ops.convert_to_tensor(y_true), ops.convert_to_tensor(y_pred)))
        b = ops.convert_to_numpy(rebuilt.call(
            ops.convert_to_tensor(y_true), ops.convert_to_tensor(y_pred)))
        np.testing.assert_allclose(a, b, atol=1e-7)
        assert rebuilt.share_training_batches is True

    def test_share_training_batches_false_keeps_the_row_axis(self):
        """With unshared batches `y_true` already carries `B * k` rows, so the
        returned axis is `B * k` too -- one value per SUPPLIED row, which is
        the axis `sample_weight` is sized against (decisions.md D-064).

        This assertion used to read `(4,)`, the averaged-over-k shape, which is
        the shared mode's answer applied to a batch that has no ensemble axis
        left to average. It made `sample_weight` an unconditional crash here.
        """
        rng = np.random.default_rng(1)
        y_pred = rng.uniform(0.05, 0.95, size=(4, 3, 1)).astype("float32")
        y_true = rng.integers(0, 2, size=(12, 1)).astype("float32")
        loss = TabMLoss("binary_crossentropy", share_training_batches=False)
        out = loss.call(ops.convert_to_tensor(y_true),
                        ops.convert_to_tensor(y_pred))
        assert tuple(ops.shape(out)) == (12,)


class TestTabMLossUnsharedWeighting:
    """The `share_training_batches=False` half of the F-68 family (D-064).

    RED at 9d71a8c4d: every test here raised
    `InvalidArgumentError: Incompatible shapes: [4] vs. [12] [Op:Mul]`.
    """

    @staticmethod
    def _unshared(seed: int = 1):
        rng = np.random.default_rng(seed)
        y_pred = rng.uniform(0.05, 0.95, size=(4, 3, 1)).astype("float32")
        y_true = rng.integers(0, 2, size=(12, 1)).astype("float32")
        return y_true, y_pred

    def test_sample_weight_of_the_label_length_is_accepted(self):
        y_true, y_pred = self._unshared()
        loss = TabMLoss("binary_crossentropy", share_training_batches=False)
        value = float(ops.convert_to_numpy(
            loss(y_true, y_pred, sample_weight=np.ones(12, "float32"))))
        assert np.isfinite(value)

    def test_each_row_is_weighted_INDIVIDUALLY(self):
        """Anti-vacuity: a per-row weight must not degenerate to a global
        rescale. Zeroing all rows but one must leave exactly that row's loss."""
        y_true, y_pred = self._unshared()
        loss = TabMLoss("binary_crossentropy", share_training_batches=False)
        per_row = ops.convert_to_numpy(
            loss.call(ops.convert_to_tensor(y_true),
                      ops.convert_to_tensor(y_pred)))
        for row in (0, 5, 11):
            w = np.zeros(12, dtype="float32")
            w[row] = 1.0
            got = float(ops.convert_to_numpy(
                loss(y_true, y_pred, sample_weight=w)))
            # `sum_over_batch_size` divides by the full 12 rows.
            np.testing.assert_allclose(got, per_row[row] / 12.0, rtol=1e-5)

    def test_row_i_is_paired_with_prediction_row_i(self):
        """Guards the ordering claim in the D-064 anchor: `reshape` is the
        inverse of the model's own `(B * k, D) -> (B, k, D)`. A transposed
        pairing would still have shape `(12,)` and still weight per row."""
        y_true, y_pred = self._unshared()
        loss = TabMLoss("binary_crossentropy", share_training_batches=False)
        got = ops.convert_to_numpy(
            loss.call(ops.convert_to_tensor(y_true),
                      ops.convert_to_tensor(y_pred)))
        flat = y_pred.reshape(12, 1)
        expected = -(
            y_true * np.log(flat) + (1.0 - y_true) * np.log(1.0 - flat)
        ).mean(axis=-1)
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)
        # Anti-vacuity: the transposed pairing really is a different answer.
        transposed = y_pred.transpose(1, 0, 2).reshape(12, 1)
        assert not np.allclose(flat, transposed)
