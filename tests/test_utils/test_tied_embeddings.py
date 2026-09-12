"""Contract for ``dl_techniques.utils.tied_embeddings.tied_embedding_logits``.

The shared helper backing all 5 weight-tied LM-head sites in
``models/language/`` (``gpt2``, ``hnet``, ``wave_field``, ``zamba2``,
``masked_language_model``). See ``plans/plan-2026-09-12T123331-28fd855f``'s
``decisions.md`` D-002 (why this helper exists), D-003 (the CRITICAL
mixed-precision regression an adversarial review caught) and D-004 (the
real-measurement correction of D-003's own diagnosis: an operand cast is a
no-op under ``AutocastScope``, and the RESULT must be cast instead).

The ``mixed_float16`` assertions below run a REAL ``model(x)`` forward pass
through an actual Keras ``Model`` -- never a bare function call on
hand-built tensors -- because D-004 measured that a bare-tensor probe cannot
see the ``AutocastScope`` effect that makes an operand-level cast a no-op in
practice. A bare-tensor probe is exactly the instrument that led the earlier,
now-superseded revision of this helper to the wrong fix.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import keras

from dl_techniques.utils.tied_embeddings import tied_embedding_logits


# ---------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------


class _TiedLogitsModel(keras.Model):
    """The smallest real model that exercises `tied_embedding_logits` under
    an actual `AutocastScope` -- i.e. through `Embedding.__call__`, not a
    hand-built tensor with no policy attached.
    """

    def __init__(self, vocab_size: int = 32, hidden_size: int = 8,
                 use_bias: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = keras.layers.Embedding(vocab_size, hidden_size)
        self.use_bias = use_bias
        if use_bias:
            self.output_bias = self.add_weight(
                name="output_bias", shape=(vocab_size,), initializer="zeros",
                trainable=True,
            )
        else:
            self.output_bias = None

    def call(self, input_ids, training=None):
        hidden = self.token_embeddings(input_ids)
        return tied_embedding_logits(
            hidden, self.token_embeddings.embeddings, bias=self.output_bias,
        )


def _old_operand_cast_logits(hidden_states, embedding_weights, *, bias=None):
    """The pre-D-004-fix expression, kept ONLY as a RED-proof reference.

    This is what `tied_embedding_logits` used to do: cast the embedding
    OPERAND to `hidden_states.dtype` before the matmul, with no cast on the
    result. D-004 measured that under a real `AutocastScope` forward pass
    this is indistinguishable from casting nothing at all -- the embedding
    variable is already autocast to the compute dtype by the time it is
    read -- so it produces `float16` output under `mixed_float16`, not the
    `float32` this module now guarantees. See decisions.md D-003/D-004.
    """
    embedding_weights = keras.ops.cast(embedding_weights, hidden_states.dtype)
    logits = keras.ops.matmul(hidden_states, keras.ops.transpose(embedding_weights))
    if bias is not None:
        logits = logits + bias
    return logits


class _OldTiedLogitsModel(keras.Model):
    """Same shape as `_TiedLogitsModel`, wired to the OLD expression."""

    def __init__(self, vocab_size: int = 32, hidden_size: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = keras.layers.Embedding(vocab_size, hidden_size)

    def call(self, input_ids, training=None):
        hidden = self.token_embeddings(input_ids)
        return _old_operand_cast_logits(hidden, self.token_embeddings.embeddings)


@pytest.fixture
def restore_global_policy():
    """Always restore the global dtype policy, even if the test body raises."""
    previous = keras.mixed_precision.global_policy().name
    try:
        yield
    finally:
        keras.mixed_precision.set_global_policy(previous)


# ---------------------------------------------------------------------
# float32 (default policy): a plain arithmetic contract, atol=0
# ---------------------------------------------------------------------


class TestFloat32Default:
    """At the default `float32` policy, the helper must be bit-identical to
    a manual matmul -- no cast, no precision loss, so `atol=0` is the
    correct bound (not a tightened-for-show one)."""

    def test_matches_manual_matmul_exactly(self):
        rng = np.random.default_rng(0)
        hidden = keras.ops.convert_to_tensor(
            rng.normal(size=(2, 5, 8)).astype(np.float32)
        )
        table = keras.ops.convert_to_tensor(
            rng.normal(size=(32, 8)).astype(np.float32)
        )

        got = tied_embedding_logits(hidden, table)
        expected = keras.ops.matmul(hidden, keras.ops.transpose(table))

        assert str(got.dtype) in ("float32", "<dtype: 'float32'>")
        assert_allclose(
            keras.ops.convert_to_numpy(got),
            keras.ops.convert_to_numpy(expected),
            rtol=0, atol=0,
        )

    def test_bias_is_added_after_the_cast_and_matches_manually(self):
        rng = np.random.default_rng(1)
        hidden = keras.ops.convert_to_tensor(
            rng.normal(size=(2, 5, 8)).astype(np.float32)
        )
        table = keras.ops.convert_to_tensor(
            rng.normal(size=(32, 8)).astype(np.float32)
        )
        bias = keras.ops.convert_to_tensor(rng.normal(size=(32,)).astype(np.float32))

        got = tied_embedding_logits(hidden, table, bias=bias)
        expected = keras.ops.matmul(hidden, keras.ops.transpose(table)) + bias

        assert_allclose(
            keras.ops.convert_to_numpy(got),
            keras.ops.convert_to_numpy(expected),
            rtol=0, atol=0,
        )


# ---------------------------------------------------------------------
# mixed_float16: the D-003/D-004 regression this module exists to guard
# ---------------------------------------------------------------------


class TestMixedFloat16RealForwardPass:
    """Every assertion here runs a REAL `model(x)` forward pass. A bare
    `tied_embedding_logits(tensor, tensor)` call with hand-built tensors
    never enters an `AutocastScope` and cannot see the effect D-004
    measured -- that is the exact blind spot that produced the wrong
    diagnosis in D-003."""

    def test_output_is_float32_under_mixed_float16(self, restore_global_policy):
        keras.mixed_precision.set_global_policy("mixed_float16")
        model = _TiedLogitsModel(vocab_size=32, hidden_size=8)
        input_ids = keras.random.randint((2, 5), minval=0, maxval=32, dtype="int32")

        logits = model(input_ids)

        assert str(logits.dtype) == "<dtype: 'float32'>", (
            f"expected float32 loss-facing logits under mixed_float16, got {logits.dtype}"
        )

    def test_bias_output_is_float32_under_mixed_float16(self, restore_global_policy):
        keras.mixed_precision.set_global_policy("mixed_float16")
        model = _TiedLogitsModel(vocab_size=32, hidden_size=8, use_bias=True)
        input_ids = keras.random.randint((2, 5), minval=0, maxval=32, dtype="int32")

        logits = model(input_ids)

        assert str(logits.dtype) == "<dtype: 'float32'>", (
            f"expected float32 logits with a bias under mixed_float16, got {logits.dtype}"
        )

    def test_output_is_finite_under_mixed_float16(self, restore_global_policy):
        """Not a dtype claim -- a sanity floor that the float32 cast did not
        silently introduce NaN/Inf (the repo-wide fp16-mask trap class)."""
        keras.mixed_precision.set_global_policy("mixed_float16")
        model = _TiedLogitsModel(vocab_size=32, hidden_size=8, use_bias=True)
        input_ids = keras.random.randint((2, 5), minval=0, maxval=32, dtype="int32")

        logits = model(input_ids)

        assert np.isfinite(keras.ops.convert_to_numpy(logits)).all()


class TestTheGuardIsProvenRed:
    """Meta-test: proves `TestMixedFloat16RealForwardPass` actually
    discriminates the fix from the defect it targets, using the SAME
    real-forward-pass instrument.

    `_old_operand_cast_logits` is byte-for-byte the expression
    `tied_embedding_logits` carried before D-004 (operand cast, no result
    cast). Run through the identical real-model harness, it must NOT
    produce float32 output under `mixed_float16` -- otherwise the assertions
    above would have passed against the old, defective code too, which
    means they were never a real guard.
    """

    def test_the_old_operand_cast_expression_fails_this_guard(
        self, restore_global_policy
    ):
        keras.mixed_precision.set_global_policy("mixed_float16")
        old_model = _OldTiedLogitsModel(vocab_size=32, hidden_size=8)
        input_ids = keras.random.randint((2, 5), minval=0, maxval=32, dtype="int32")

        old_logits = old_model(input_ids)

        assert str(old_logits.dtype) != "<dtype: 'float32'>", (
            "the OLD operand-cast expression unexpectedly produced float32 "
            "output -- if this fires, the new fix's float32 guarantee is "
            "not actually distinguishing behaviour and needs re-deriving"
        )

    def test_the_new_expression_differs_from_the_old_one_in_dtype(
        self, restore_global_policy
    ):
        """Direct side-by-side: same weights, same input, only the helper
        differs -- isolates the dtype delta to the fix itself."""
        keras.mixed_precision.set_global_policy("mixed_float16")
        keras.utils.set_random_seed(0)
        new_model = _TiedLogitsModel(vocab_size=32, hidden_size=8)
        keras.utils.set_random_seed(0)
        old_model = _OldTiedLogitsModel(vocab_size=32, hidden_size=8)
        input_ids = keras.random.randint((2, 5), minval=0, maxval=32, dtype="int32")

        new_logits = new_model(input_ids)
        old_logits = old_model(input_ids)

        assert str(new_logits.dtype) == "<dtype: 'float32'>"
        assert str(old_logits.dtype) == "<dtype: 'float16'>"
