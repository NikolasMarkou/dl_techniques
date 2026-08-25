"""The windowed attention layer must work in GRAPH mode, not only eagerly.

Why this module exists, and why 1944 green attention tests did not already cover
it. `WindowAttention` short-circuits the degenerate single-window regime
`1 < N < window_size ** 2` by handing `SingleWindowAttention` a static slot map.
Until 2026-08-25 that map travelled as a `call()` KEYWORD ARGUMENT holding a
numpy array. Keras 3 maps `dtype_policy.convert_input` over call arguments, so
inside a `tf.function` trace the array became a SYMBOLIC tensor and the callee's
`np.asarray(...)` raised::

    NotImplementedError: Cannot convert a symbolic tf.Tensor
    (functional_1/window_attention_1/Const:0) to a numpy array.

The trigger is the graph path AND that `N` regime together, which is exactly
what every pre-existing instrument missed:

* the 42-cell external-golden harness
  (`test_window_attention_restructure_is_inert.py`), the 28-cell all-ones-mask
  no-op class and the 15-cell slot-count class all call the layer EAGERLY;
* `model(x)` eager, `layer(x)` direct, functional BUILD and `model.save()` all
  pass even with the defect present -- only `predict()` / `fit()` / an explicit
  `@tf.function` fail;
* every functional-model test in the suite misses the regime by luck:
  `test_model_save_load` uses `N=50` with `window_size` in {7, 5, 4} and
  `TestAllThreePartitionModes` uses `N=16` with `window_size=2` -- all
  `N > window_size ** 2`, i.e. on the padding path, which never supplies a slot
  map at all;
* no test anywhere compiled a model and ran a training step on this layer.

So: every cell below runs on a FUNCTIONAL Keras model, in all three partition
modes, at an `N` INSIDE the short-circuit regime and one OUTSIDE it, through
`predict()`, through an explicit `@tf.function`, and through one `fit()` step.
Do not weaken any of those four axes -- dropping the graph axis, the functional
axis, or the inside-the-regime `N` each independently makes this module green
against the very defect it exists to catch.

See decisions.md D-015 (plan-2026-08-25T053412-0f1fa04f).
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.attention.window_attention import WindowAttention

# ---------------------------------------------------------------------

DIM = 32
NUM_HEADS = 4
WINDOW_SIZE = 4

# `window_size ** 2 == 16`. The first entry of each pair is INSIDE the
# degenerate short-circuit regime `1 < N < window_size ** 2` (the one that
# raised), the second is OUTSIDE it (the padding path, as a control).
SEQ_LENS = [
    pytest.param(9, id="N9-inside-short-circuit"),
    pytest.param(25, id="N25-outside-short-circuit"),
]
PARTITION_MODES = ["grid", "zigzag", "band"]


def _model(partition_mode: str, seq_len: int) -> keras.Model:
    """A functional Keras model wrapping one `WindowAttention`.

    Functional, not a bare layer call: the defect this module guards is
    invisible unless the layer is traced as part of a model.
    """
    keras.utils.set_random_seed(17)
    inputs = keras.Input(shape=(seq_len, DIM))
    outputs = WindowAttention(
        dim=DIM,
        window_size=WINDOW_SIZE,
        num_heads=NUM_HEADS,
        partition_mode=partition_mode,
        # The band refuses the relative-position bias (D-010); the tile modes
        # keep it, because the slot map's only job is to gather that bias.
        use_relative_position_bias=(partition_mode != "band"),
        dropout_rate=0.0,
    )(inputs)
    return keras.Model(inputs, outputs)


# ---------------------------------------------------------------------


class TestTheGraphPathWorksInEveryPartitionMode:
    """`predict`, `@tf.function` and `fit` on a functional model, all modes."""

    @pytest.mark.parametrize("partition_mode", PARTITION_MODES)
    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_predict_runs(self, partition_mode, seq_len):
        """`model.predict()` traces the layer; it must not raise.

        Why this can fail if the implementation is wrong: `predict` runs the
        layer inside a `tf.function`, where any numpy-valued `call()` argument
        has been converted to a symbolic tensor.
        """
        model = _model(partition_mode, seq_len)
        x = np.random.default_rng(0).normal(
            size=(2, seq_len, DIM)
        ).astype("float32")

        y = model.predict(x, verbose=0)

        assert y.shape == (2, seq_len, DIM)
        assert np.all(np.isfinite(y))

    @pytest.mark.parametrize("partition_mode", PARTITION_MODES)
    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_predict_agrees_with_the_eager_call(self, partition_mode, seq_len):
        """Running is not enough: the graph must compute the SAME thing.

        Why this can fail if the implementation is wrong: a fix that made the
        trace succeed by taking a DIFFERENT branch -- e.g. falling back to the
        padding path whenever the slot map is unavailable -- would pass the
        test above while silently changing the answer and reinstating the
        O(window_size ** 4) cost.
        """
        model = _model(partition_mode, seq_len)
        x = np.random.default_rng(1).normal(
            size=(2, seq_len, DIM)
        ).astype("float32")

        eager = keras.ops.convert_to_numpy(model(x, training=False))
        graph = model.predict(x, verbose=0)

        np.testing.assert_allclose(graph, eager, atol=1e-6, rtol=0)

    @pytest.mark.parametrize("partition_mode", PARTITION_MODES)
    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_an_explicit_tf_function_runs(self, partition_mode, seq_len):
        """The same check without Keras' own training machinery in the way."""
        model = _model(partition_mode, seq_len)
        x = np.random.default_rng(2).normal(
            size=(2, seq_len, DIM)
        ).astype("float32")

        @tf.function
        def traced(z):
            return model(z, training=False)

        y = keras.ops.convert_to_numpy(traced(tf.constant(x)))

        assert y.shape == (2, seq_len, DIM)
        assert np.all(np.isfinite(y))

    @pytest.mark.parametrize("partition_mode", PARTITION_MODES)
    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_one_fit_step_runs_and_moves_the_weights(
        self, partition_mode, seq_len
    ):
        """A layer whose whole purpose is to be trained must survive `fit`.

        Why this can fail if the implementation is wrong: `fit` traces both the
        forward and the backward pass. The weight-movement assertion is what
        stops a vacuous pass -- a model that trained on a detached graph would
        `fit` happily and move nothing.
        """
        model = _model(partition_mode, seq_len)
        rng = np.random.default_rng(3)
        x = rng.normal(size=(4, seq_len, DIM)).astype("float32")
        y = rng.normal(size=(4, seq_len, DIM)).astype("float32")

        before = [
            keras.ops.convert_to_numpy(w) for w in model.trainable_weights
        ]
        model.compile(optimizer=keras.optimizers.SGD(0.1), loss="mse")
        history = model.fit(x, y, epochs=1, batch_size=4, verbose=0)
        after = [
            keras.ops.convert_to_numpy(w) for w in model.trainable_weights
        ]

        assert np.isfinite(history.history["loss"][0])
        moved = [
            w.name
            for w, b, a in zip(model.trainable_weights, before, after)
            if not np.array_equal(b, a)
        ]
        assert moved, (
            f"one fit() step on partition_mode={partition_mode!r}, N={seq_len} "
            f"moved NONE of the {len(before)} trainable weights"
        )


class TestTheSlotMapDoesNotSurviveIntoTheNextCall:
    """The slot map is instance state; a stale one is otherwise invisible."""

    @pytest.mark.parametrize("partition_mode", ["grid", "zigzag"])
    def test_the_map_is_cleared_after_every_call(self, partition_mode):
        """Why this can fail if the implementation is wrong: the slot map moved
        off the call-argument channel onto the layer instance, so a caller that
        forgot to clear it would leak the previous call's LAYOUT into the next
        one -- right dtype, plausible length, and it moves only the
        relative-position bias. `WindowAttention._attend` clears it in a
        `finally`; this pins that.
        """
        layer = WindowAttention(
            dim=DIM,
            window_size=WINDOW_SIZE,
            num_heads=NUM_HEADS,
            partition_mode=partition_mode,
            use_relative_position_bias=True,
            dropout_rate=0.0,
        )
        x = np.zeros((2, 9, DIM), dtype="float32")

        layer(x, training=False)

        assert layer.attention._window_slots is None
