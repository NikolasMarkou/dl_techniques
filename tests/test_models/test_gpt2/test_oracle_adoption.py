"""
Oracle adoption for ``models/gpt2`` -- Phase 5 batch A.

Zero adoption of the three shared instruments before this file. All three are
adopted; no new oracle is authored.

Measured 2026-08-21 (CPU) on ``GPT2(vocab_size=256, embed_dim=64, depth=2,
num_heads=4, max_seq_len=32, dropout_rate=0.0, attention_dropout_rate=0.0)``
after one real optimizer step: **30** trainable weights, **0** dead, **0**
non-finite.

The positional embedding is the weight this buys most for. A decoder that builds
``wpe`` and then never adds it -- or adds a sliced constant instead -- produces
logits of exactly the right shape and a perfectly finite loss, and only a
per-weight gradient reading says so. ``test_the_positional_embedding_is_live``
pins it by name rather than trusting the count.

The dropout rates are pinned to 0.0 for the gradient reading: at a non-zero rate
the single draw the tape sees can mask a path and report its weights dead as a
property of the DRAW, not of the model.
"""

import numpy as np
import keras
import pytest
import tensorflow as tf

from dl_techniques.models.gpt2.gpt2 import GPT2

from ..gradient_flow_oracle import (
    assert_gradients_reach_every_trainable_weight,
    default_loss,
    stop_all_gradients,
)
from ..knob_sensitivity_oracle import assert_structural_knob_changes_weights
from ..smoke_contract_oracle import (
    assert_contract_rejects_a_broken_forward,
    assert_finite,
    broken_forward,
)

VOCAB_SIZE = 256
EMBED_DIM = 64
MAX_SEQ_LEN = 32

#: Measured 2026-08-21 at depth=2, num_heads=4, embed_dim=64.
GF_N_WEIGHTS = 30


def _ids(batch: int = 2, length: int = MAX_SEQ_LEN) -> np.ndarray:
    return np.random.default_rng(0).integers(
        0, VOCAB_SIZE, (batch, length)
    ).astype("int32")


def _model(**overrides):
    kwargs = dict(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, depth=2,
                  num_heads=4, max_seq_len=MAX_SEQ_LEN, dropout_rate=0.0,
                  attention_dropout_rate=0.0)
    kwargs.update(overrides)
    model = GPT2(**kwargs)
    # Build at the model's OWN max_seq_len: the decoder refuses a sequence
    # longer than its positional table, so a fixed 32-token warm-up would make
    # every `max_seq_len < 32` arm of the knob sweep raise instead of measure.
    model(_ids(1, kwargs["max_seq_len"]), training=False)
    return model


def _one_adam_step(model, inputs) -> None:
    optimizer = keras.optimizers.Adam(1e-3)
    variables = list(model.trainable_variables)
    optimizer.build(variables)
    with tf.GradientTape() as tape:
        loss = default_loss(model(inputs, training=True))
    optimizer.apply_gradients(zip(tape.gradient(loss, variables), variables))


class TestGPT2GradientFlow:

    def test_gradients_reach_every_trainable_weight_after_one_step(self):
        model = _model()
        x = _ids()
        _one_adam_step(model, x)

        report = assert_gradients_reach_every_trainable_weight(model, x)

        assert len(report) == GF_N_WEIGHTS == len(model.trainable_weights)

    def test_the_positional_embedding_is_live(self):
        """Named, not counted -- see the module docstring."""
        model = _model()
        x = _ids()
        _one_adam_step(model, x)
        report = assert_gradients_reach_every_trainable_weight(model, x)

        positional = [
            p for p in report
            if "position" in p.lower() or p.rsplit("/", 2)[-2:][0] == "wpe"
        ]
        assert positional, (
            f"no weight path names a positional embedding; paths: {sorted(report)}"
        )
        for path in positional:
            assert report[path] > 0.0, f"{path} receives no gradient"

    def test_the_gradient_assertion_can_fail(self):
        model = _model()
        with broken_forward(model, stop_all_gradients):
            with pytest.raises(AssertionError, match="received NO gradient"):
                assert_gradients_reach_every_trainable_weight(model, _ids())


class TestGPT2KnobSensitivity:

    def test_depth_changes_the_parameterisation(self):
        builders = {d: (lambda d=d: _model(depth=d)) for d in (1, 2, 3)}
        signatures = assert_structural_knob_changes_weights(builders, knob="depth")
        counts = [len(signatures[d]) for d in (1, 2, 3)]
        assert counts == sorted(counts) and counts[0] < counts[-1], counts

    def test_max_seq_len_changes_the_parameterisation(self):
        """A knob that reaches the POSITIONAL table and nothing else.

        `depth` above would pass on a model with no positional embedding at all.
        """
        builders = {n: (lambda n=n: _model(max_seq_len=n)) for n in (16, 32, 64)}
        assert_structural_knob_changes_weights(builders, knob="max_seq_len")

    def test_the_knob_assertion_can_fail(self):
        builders = {"a": (lambda: _model(depth=2)), "b": (lambda: _model(depth=2))}
        with pytest.raises(AssertionError, match="is a no-op"):
            assert_structural_knob_changes_weights(builders, knob="depth")


class TestGPT2SmokeContract:

    def test_the_forward_contract_rejects_a_broken_forward(self):
        model = _model()
        x = _ids()
        batch, length = x.shape

        def contract(out):
            assert isinstance(out, dict), f"expected a dict, got {type(out)}"
            assert set(out) == {"logits", "last_hidden_state"}, (
                f"key set drifted: {sorted(out)}"
            )
            assert tuple(out["logits"].shape) == (batch, length, VOCAB_SIZE)
            assert tuple(out["last_hidden_state"].shape) == (batch, length, EMBED_DIM)
            assert_finite(out)

        rejections = assert_contract_rejects_a_broken_forward(model, x, contract)
        assert set(rejections) == {
            "collapse_to_scalar", "slice_leading_axis", "append_trailing_axis",
        }

    def test_a_nonpositive_depth_is_rejected_at_construction(self):
        """The argument-validation half of the contract."""
        with pytest.raises(ValueError, match="depth must be positive"):
            GPT2(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, depth=0,
                 num_heads=4, max_seq_len=MAX_SEQ_LEN)

    def test_a_sequence_longer_than_the_positional_table_is_refused_at_build(self):
        """Positions beyond `max_seq_len` have no defined encoding.

        SCOPE, and why this asserts on an UNBUILT model only: the guard lives in
        `TextDecoder.build` (`layers/transformers/text_decoder.py`), so it fires
        on the FIRST call and not on later ones. That limitation is deliberate
        and already documented at the guard site, together with the measurement
        that the unguarded path is DEVICE-DEPENDENT -- on CPU the embedding
        gather raises `InvalidArgumentError`, on GPU `GatherV2` clips the
        out-of-range indices and returns finite, plausible-looking garbage with
        no exception at all.

        So this test pins the half that is a contract (the clean, actionable
        refusal on an unbuilt model) and deliberately does NOT assert anything
        about an already-built model, because there is no device-independent
        behaviour there to assert. Writing that second assertion is how a suite
        acquires a test that passes on one machine and fails on another.
        """
        # DECISION plan-2026-08-19T163559-499b6f0e/D-094
        # Construct FRESH here; do NOT reuse `_model()`, which warms the model
        # up. The guard is in `TextDecoder.build` and fires once. On an
        # already-built model the over-length sequence reaches the positional
        # embedding unguarded, and the outcome is DEVICE-DEPENDENT (CPU:
        # InvalidArgumentError from GatherV2; GPU: clipped indices, finite
        # garbage, no exception). Asserting on that path at all -- with
        # `pytest.raises` of any type -- produces a test that passes on one
        # machine and fails on another. See D-094 in plans/plan-2026-08-19T163559-499b6f0e/decisions.md.
        model = GPT2(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, depth=2,
                     num_heads=4, max_seq_len=16, dropout_rate=0.0,
                     attention_dropout_rate=0.0)
        with pytest.raises(ValueError, match="exceeds max_seq_len"):
            model(_ids(1, 32), training=False)
