"""N-05: ``CliffordNetBlock`` under ``@tf.function`` with a SYMBOLIC ``training``.

The defect (decisions.md D-056)
-------------------------------
``CliffordNetBlock.call`` forwards ``training`` to ``input_norm`` and
``ctx_norm``, and ``normalization_type="batch_norm"`` is the default. Keras
3.8's ``BatchNormalization.call`` branches on ``training`` with a Python ``if``,
so a caller that traces the block inside a ``@tf.function`` taking ``training``
as an argument got ``OperatorNotAllowedInGraphError`` — the flag arrives as a
placeholder, not a constant. This is what kept a ``strict`` xfail alive at
``tests/test_models/test_video_jepa/test_video_jepa.py`` since 2026-08-10.

The framework half of it is measured here too (``test_the_framework_constraint_
is_real``), because the fix is only justified if Keras genuinely cannot do this:
if a later Keras accepts a symbolic flag natively, that test goes red and the
``keras.ops.cond`` gate can be deleted rather than carried forever.

What is deliberately NOT claimed
--------------------------------
This does not make the block "graph-safe" in general — it makes the two
``training``-gated norm calls graph-safe. ``GGR`` accepts ``training`` and
ignores it (no norm, no dropout), which is asserted below rather than assumed.
"""

import numpy as np
import pytest
import tensorflow as tf
import keras

from dl_techniques.layers.geometric.clifford_block import (
    CliffordNetBlock,
    _call_with_training_gate,
)

CHANNELS = 8
SHAPE = (2, 6, 6, CHANNELS)


def _inputs(seed: int = 0):
    return np.random.RandomState(seed).randn(*SHAPE).astype("float32")


@pytest.fixture
def built_block():
    block = CliffordNetBlock(
        channels=CHANNELS, shifts=[1, 2], normalization_type="batch_norm"
    )
    block(_inputs())  # eager warmup / build
    return block


class TestTheFrameworkConstraintIsReal:
    """Control: the gate exists because Keras itself cannot take the flag.

    If this ever goes GREEN for BatchNormalization, ``_call_with_training_gate``
    is obsolete — delete it rather than keeping a workaround for a fixed bug.
    """

    @pytest.mark.parametrize("flag", [False, True])
    def test_bare_batch_norm_rejects_a_symbolic_flag(self, flag):
        norm = keras.layers.BatchNormalization()
        norm.build((None, CHANNELS))

        @tf.function
        def fn(v, training):
            return norm(v, training=training)

        with pytest.raises(Exception) as excinfo:
            fn(tf.constant(np.zeros((2, CHANNELS), "float32")),
               tf.constant(flag))
        assert "OperatorNotAllowedInGraph" in type(excinfo.value).__name__ or (
            "not allowed" in str(excinfo.value).lower()
        ), f"unexpected error: {type(excinfo.value).__name__}: {excinfo.value}"

    def test_layer_norm_does_not_need_the_gate(self):
        """Asymmetric control: not every norm has the constraint."""
        norm = keras.layers.LayerNormalization()
        norm.build((None, CHANNELS))

        @tf.function
        def fn(v, training):
            return norm(v, training=training)

        out = fn(tf.constant(np.zeros((2, CHANNELS), "float32")),
                 tf.constant(False))
        assert tuple(out.shape) == (2, CHANNELS)


class TestTheBlockTracesWithASymbolicFlag:
    """The N-05 regression proper."""

    @pytest.mark.parametrize("flag", [False, True])
    def test_batch_norm_block_traces(self, built_block, flag):
        @tf.function
        def fn(v, training):
            return built_block(v, training=training)

        out = fn(tf.constant(_inputs(1)), tf.constant(flag))
        assert tuple(out.shape) == SHAPE
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))

    def test_python_bool_and_none_still_trace(self, built_block):
        @tf.function
        def fn(v, training):
            return built_block(v, training=training)

        for flag in (None, False, True):
            out = fn(tf.constant(_inputs(2)), flag)
            assert tuple(out.shape) == SHAPE


class TestTheGateChangesNoNumbers:
    """The Python-``bool`` path must be untouched, bit for bit.

    ``keras.ops.cond`` is reached only when the flag is a tensor. Every trainer
    and every eager call in this repo passes a Python bool or ``None``, so the
    fix has to be a no-op for them — asserted at ZERO tolerance, not ``atol``.
    """

    def test_inference_is_bit_identical_across_the_two_paths(self, built_block):
        direct = keras.ops.convert_to_numpy(
            built_block(_inputs(3), training=False)
        )

        @tf.function
        def fn(v, training):
            return built_block(v, training=training)

        gated = keras.ops.convert_to_numpy(
            fn(tf.constant(_inputs(3)), tf.constant(False))
        )
        assert np.array_equal(direct, gated), (
            f"max|delta| = {np.abs(direct - gated).max()}; the gate must not "
            "move inference numerics"
        )

    def test_the_true_branch_still_updates_the_moving_statistics(self):
        """Anti-vacuity: a gate that always took the False branch would pass
        every test above while silently disabling batch-norm training."""
        block = CliffordNetBlock(
            channels=CHANNELS, shifts=[1, 2], normalization_type="batch_norm"
        )
        block(_inputs(), training=False)
        before = [np.array(v) for v in block.ctx_norm.weights]

        @tf.function
        def fn(v, training):
            return block(v, training=training)

        fn(tf.constant(_inputs(4) * 5.0 + 3.0), tf.constant(True))
        after = [np.array(v) for v in block.ctx_norm.weights]

        moved = max(
            float(np.abs(a - b).max()) for a, b in zip(after, before)
        )
        assert moved > 0.0, (
            "no ctx_norm statistic moved under a symbolic training=True; the "
            "gate is stuck on the inference branch"
        )


class TestTheHelperItself:
    """Unit-level contract of ``_call_with_training_gate``."""

    def test_none_and_bool_bypass_the_conditional(self):
        seen = []

        class Spy(keras.layers.Layer):
            def call(self, x, training=None):
                seen.append(training)
                return x

        spy = Spy()
        x = keras.ops.convert_to_tensor(np.zeros((2, 3), "float32"))
        _call_with_training_gate(spy, x, training=None)
        _call_with_training_gate(spy, x, training=True)
        _call_with_training_gate(spy, x, training=False)
        assert seen == [None, True, False], seen

    def test_ggr_ignores_training(self, built_block):
        """The third forwarding site (``self.ggr``) needs no gate — proven,
        not assumed, so a future GGR that grows a dropout is caught here."""
        import inspect

        source = inspect.getsource(type(built_block.ggr).call)
        body = source.split('"""')[-1]
        assert "training" not in body, (
            "GGR.call now uses `training`; if it gained a norm or a dropout it "
            "must be routed through _call_with_training_gate as well"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
