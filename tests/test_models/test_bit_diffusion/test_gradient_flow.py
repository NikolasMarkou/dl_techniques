"""Every trainable weight is on the backward graph -- measured AFTER training starts.

Adopts ``tests/test_models/gradient_flow_oracle.py``. Two things about this
model make the naive adoption report a false result, and both were measured
rather than assumed.

**1. Never at initialisation.** ``DiTXA`` is adaLN-Zero: every block's
modulation ``Dense`` is zero in kernel and bias and the final projection is zero
too, so a fresh model emits the EXACT zero tensor and **49 of its 51** trainable
weights carry an identically-zero gradient. That is correct behaviour -- the
conditioning path is meant to start switched off and grow in -- but it is a
false positive large enough to hide a real dead weight among it.

**2. The oracle's default loss is a FIXED POINT here, so it can never train out
of that state.** ``default_loss`` is the mean of squares of the output. At a
zero output its own derivative is zero, so *every* gradient is zero, the
optimizer step is a no-op, and the next report is identical. Measured: SGD at
``lr = 1.0`` for three steps left **51 of 51** weights dead, forever. The loss
below is therefore a mean-squared error against a fixed random target, which has
a non-zero derivative at a zero prediction -- the same shape the real trainer
uses.

**3. ONE optimizer step is not enough for this architecture; two are.** The
zero-initialised layers sit in SERIES on the conditioning path: a block's
``adaln_modulation`` feeds gates that are zero, and the final layer's projection
is zero as well. Step 1 can only wake the layer nearest the loss. Measured, at
``lr = 0.5``:

===========================  ==================
state                        dead weights (/51)
===========================  ==================
at initialisation            49
after 1 optimizer step       41
after 2 optimizer steps      **0**
===========================  ==================

So the claim this file makes is "after training has actually started, no weight
is stranded", and the 41-after-one-step reading is pinned too, so the shape of
the ramp is a recorded fact rather than a number someone tunes until green.

**Waivers: none.** ``expect_zero`` is empty on purpose. Every one of the 51
trainable weights receives a finite, not-identically-zero gradient after two
steps.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.vision_language.bit_diffusion.model import DiTXA
from dl_techniques.models.vision_language.bit_diffusion.token_decoder import (
    SharedTokenDecoder,
)

from ..gradient_flow_oracle import (
    assert_gradients_reach_every_trainable_weight,
    default_loss,
    gradient_report,
    stop_all_gradients,
)
from ..smoke_contract_oracle import broken_forward
from ._ditxa_helpers import batch

BATCH = 4
#: Measured 2026-09-02 at lr=0.5 on the `tiny` variant. See the module docstring.
DEAD_AT_INIT = 49
DEAD_AFTER_ONE_STEP = 41
TRAINABLE_TENSORS = 51


def fresh_model():
    """A built ``tiny`` DiTXA plus a mixed-direction, partly-masked batch.

    The batch mixes both directions and masks one sample's conditioning
    stream: with an all-forward batch the reverse conditioning embedder never
    sees a live path, and the oracle would convict a perfectly healthy model.
    """
    keras.utils.set_random_seed(0)
    model = DiTXA.from_variant("tiny", class_dropout_rate=0.1, label_seed=3)
    inputs = batch(
        model,
        batch_size=BATCH,
        direction=[0.0, 1.0, 0.0, 1.0],
        cond_mask=[1.0, 1.0, 1.0, 0.0],
    )
    model(inputs)
    return model, inputs


def fixed_target(model):
    """A constant regression target, so the loss has a non-zero derivative."""
    shape = (BATCH, model.input_size, model.input_size, model.out_channels)
    return np.random.default_rng(9).normal(size=shape).astype("float32")


def mse_against(target):
    """``outputs -> mean((outputs - target)^2)``."""
    tensor = keras.ops.convert_to_tensor(target)
    return lambda outputs: keras.ops.mean(
        keras.ops.square(outputs - keras.ops.cast(tensor, outputs.dtype))
    )


def train_steps(model, inputs, target, steps):
    """Run ``steps`` stock optimizer steps. No custom ``train_step`` anywhere."""
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.5), loss="mse")
    model.fit(inputs, target, batch_size=BATCH, epochs=steps, verbose=0)
    return model


def dead_count(report):
    return sum(1 for value in report.values() if value is None or value == 0.0)


class TestTheRampIsWhatWeThinkItIs:
    """The three readings the adoption decision rests on, pinned."""

    def test_the_oracles_default_loss_cannot_move_this_model_at_all(self):
        """A zero output is a stationary point of a mean-of-squares loss.

        Recorded so nobody "simplifies" the loss below back to the default and
        then widens the waiver list to make the result fit.
        """
        model, inputs = fresh_model()
        report = gradient_report(model, inputs, loss_fn=default_loss, training=True)
        assert dead_count(report) == len(report) == TRAINABLE_TENSORS, (
            f"{dead_count(report)} of {len(report)} dead under default_loss"
        )

    def test_at_initialisation_almost_everything_reads_dead(self):
        model, inputs = fresh_model()
        report = gradient_report(
            model, inputs, loss_fn=mse_against(fixed_target(model)), training=True
        )
        assert len(report) == TRAINABLE_TENSORS
        assert dead_count(report) == DEAD_AT_INIT, dead_count(report)

    def test_one_optimizer_step_is_not_enough(self):
        """adaLN-Zero puts two zero-init layers in SERIES on the same path."""
        model, inputs = fresh_model()
        target = fixed_target(model)
        train_steps(model, inputs, target, steps=1)
        report = gradient_report(
            model, inputs, loss_fn=mse_against(target), training=True
        )
        assert dead_count(report) == DEAD_AFTER_ONE_STEP, dead_count(report)


class TestAfterTrainingStartsNothingIsStranded:
    """The claim. Zero dead weights, zero waivers."""

    def test_every_trainable_weight_receives_a_gradient(self):
        model, inputs = fresh_model()
        target = fixed_target(model)
        train_steps(model, inputs, target, steps=2)
        report = assert_gradients_reach_every_trainable_weight(
            model,
            inputs,
            loss_fn=mse_against(target),
            expect_zero=(),  # no waivers; see the module docstring
            training=True,
        )
        assert len(report) == TRAINABLE_TENSORS, len(report)
        assert all(np.isfinite(v) for v in report.values())

    def test_the_assertion_is_capable_of_failing(self):
        """The shared dead-component injection. Every weight leaves the graph."""
        model, inputs = fresh_model()
        target = fixed_target(model)
        train_steps(model, inputs, target, steps=2)
        with broken_forward(model, stop_all_gradients):
            with pytest.raises(AssertionError, match="NO gradient|zero"):
                assert_gradients_reach_every_trainable_weight(
                    model, inputs, loss_fn=mse_against(target), training=True
                )

    def test_the_conditioning_stream_is_reached_from_both_directions(self):
        """A stronger, model-specific claim on top of the oracle's report.

        Both conditioning embedders run on every sample (D-005), so BOTH must
        be on the backward graph. An all-forward batch would leave the reverse
        one reading dead and would have to be waived -- the waiver would be
        hiding the test's own batch design, not a property of the model.
        """
        model, inputs = fresh_model()
        target = fixed_target(model)
        train_steps(model, inputs, target, steps=2)
        report = gradient_report(
            model, inputs, loss_fn=mse_against(target), training=True
        )
        for name in ("cond_embedder_forward", "cond_embedder_reverse"):
            reached = {
                path: value for path, value in report.items() if name in path
            }
            assert reached, f"no weight path contains {name!r}"
            assert all(
                value is not None and value > 0.0 for value in reached.values()
            ), (name, reached)


class TestTheDecoderTrainsFromInitialisation:
    """``SharedTokenDecoder`` has no zero-init kernel, so no ramp applies."""

    def test_every_weight_receives_a_gradient_with_no_training_at_all(self):
        keras.utils.set_random_seed(0)
        decoder = SharedTokenDecoder(
            vocab_size=13, hidden_dim=16, token_seq_len=4, token_emb_dim=8
        )
        x = np.random.default_rng(3).normal(size=(BATCH, 32)).astype("float32")
        decoder(x)
        report = assert_gradients_reach_every_trainable_weight(
            decoder, x, expect_zero=(), training=True
        )
        assert len(report) == 6, len(report)

    def test_the_decoder_assertion_is_capable_of_failing(self):
        keras.utils.set_random_seed(0)
        decoder = SharedTokenDecoder(
            vocab_size=13, hidden_dim=16, token_seq_len=4, token_emb_dim=8
        )
        x = np.random.default_rng(3).normal(size=(BATCH, 32)).astype("float32")
        decoder(x)
        with broken_forward(decoder, stop_all_gradients):
            with pytest.raises(AssertionError, match="NO gradient|zero"):
                assert_gradients_reach_every_trainable_weight(decoder, x)
