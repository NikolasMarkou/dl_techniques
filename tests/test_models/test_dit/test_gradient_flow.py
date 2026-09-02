"""Every trainable weight is on the backward graph -- measured AFTER training starts.

Adopts ``tests/test_models/gradient_flow_oracle.py``. Three things about this
model make the naive adoption report a false result, and all three were measured
rather than assumed.

**1. Never at initialisation.** ``DiT`` is adaLN-Zero: every block's
``adaln/linear`` is zero in kernel and bias and the final layer's read-out
projection is zero too, so a fresh model emits the EXACT zero tensor. Under a
real loss **37 of its 39** trainable tensors carry an identically-zero gradient.
That is correct behaviour -- the conditioning path starts switched off and grows
in -- but it is a false positive large enough to hide a real dead weight inside.

**2. The oracle's default loss is a FIXED POINT here, so it can never train out
of that state.** ``default_loss`` is the mean of squares of the output. At a zero
output its own derivative is zero, so *every* gradient is zero, an optimizer step
is a no-op, and the next report is identical. Measured: **39 of 39** dead, and no
number of steps changes it. The loss used below is therefore the package's real
objective, :class:`~dl_techniques.losses.ddpm_hybrid_loss.DDPMHybridLoss`, whose
epsilon-MSE term has a non-zero derivative at a zero prediction -- the same
objective ``src/train/dit/`` will compile.

**3. ONE optimizer step is not enough for this architecture; two are.** The
zero-initialised layers sit in SERIES: the final layer's read-out kernel is zero,
so ``dL/d(tokens)`` is zero everywhere upstream of it, and each block's
``adaln/linear`` is zero, so its gates annihilate the attention and MLP branches.
Step 1 can only wake the layer nearest the loss. Measured on :data:`TINY` at
``SGD(lr=0.5)`` against the real objective:

===============================  ===================
state                            dead tensors (/39)
===============================  ===================
under ``default_loss``, any step  39
at initialisation                 37
after 1 optimizer step            29
after 2 optimizer steps           **0**
===============================  ===================

The claim this file makes is therefore "after training has actually started, no
weight is stranded", and the 37/29 readings are pinned too, so the SHAPE of the
ramp is a recorded fact rather than a number someone tunes until green.

**Waivers: none.** ``expect_zero`` is empty on purpose. All 39 trainable tensors
receive a finite, not-identically-zero gradient after two steps.

**The two non-trainable tensors never appear here** and that is the point of
them: ``pos_embed`` (the frozen sin-cos table) and ``t_embedder/freqs`` (the
frequency ladder) are not in ``trainable_weights``, so the oracle cannot see
them. A separate arm below asserts they do not move across the optimizer steps,
which is the claim their absence from this report would otherwise leave unmade.
"""

from typing import Any, List, Optional, Tuple, Union

import keras
import numpy as np
import pytest

from dl_techniques.losses.ddpm_hybrid_loss import DDPMHybridLoss
from dl_techniques.models.vision_language.dit.model import DiT

from ..gradient_flow_oracle import (
    assert_gradients_reach_every_trainable_weight,
    default_loss,
    gradient_report,
    stop_all_gradients,
)
from ..smoke_contract_oracle import broken_forward
from ._dit_helpers import BATCH, TINY, ddpm_training_batch, np_

#: Measured 2026-09-02 on :data:`TINY` at ``SGD(lr=0.5)``. See the docstring.
TRAINABLE_TENSORS = 39
DEAD_UNDER_DEFAULT_LOSS = 39
DEAD_AT_INIT = 37
DEAD_AFTER_ONE_STEP = 29

#: A short chain keeps ``DDPMSchedule`` cheap; ``'linear'`` is only defined for
#: ``num_timesteps >= 50``, so 50 is the smallest legal value.
TIMESTEPS = 50


def make_loss() -> DDPMHybridLoss:
    """The package's real training objective, at :data:`TINY`'s channel count."""
    return DDPMHybridLoss(
        schedule_name="linear",
        num_timesteps=TIMESTEPS,
        in_channels=TINY["in_channels"],
    )


def fresh(model_cls: type = DiT) -> Tuple[DiT, list, np.ndarray, DDPMHybridLoss]:
    """A seeded, BUILT model plus the real ``(inputs, y_true)`` pair and loss.

    The explicit ``name=`` is load-bearing for the seeded-defect subclass: Keras
    derives a root scope name from the class name, and a leading underscore is
    rejected as a root scope. Fixing the name here rather than renaming the
    subclass keeps the subclass module-private, and keeps every weight path in
    the reports below independent of which class produced them.
    """
    keras.utils.set_random_seed(0)
    model = model_cls(**dict(TINY, label_seed=3), name="dit")
    loss = make_loss()
    inputs, y_true = ddpm_training_batch(model, loss, batch=BATCH, seed=7)
    model(inputs, training=False)
    return model, inputs, y_true, loss


def objective(loss: DDPMHybridLoss, y_true: np.ndarray):
    """``outputs -> scalar`` adapter so the oracle can drive a ``keras.Loss``.

    Interface contract: closes over a CONSTANT ``y_true``; the returned callable
    is pure in ``outputs``. The reduction is an explicit ``mean`` because the
    oracle needs a scalar and a ``Loss``'s own reduction is a configurable.
    """
    target = keras.ops.convert_to_tensor(y_true)
    return lambda outputs: keras.ops.mean(
        loss(keras.ops.cast(target, outputs.dtype), outputs)
    )


def train_steps(model: DiT, inputs, y_true, loss, steps: int, lr: float = 0.5):
    """Run ``steps`` stock optimizer steps. No custom ``train_step`` anywhere."""
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr), loss=loss)
    model.fit(inputs, y_true, batch_size=BATCH, epochs=steps, verbose=0)
    return model


def dead_count(report: dict) -> int:
    return sum(1 for value in report.values() if value is None or value == 0.0)


# ---------------------------------------------------------------------
# The readings the adoption decision rests on
# ---------------------------------------------------------------------


class TestTheRampIsWhatWeThinkItIs:
    """Four measurements, pinned, so nobody re-tunes them into agreement."""

    def test_the_oracles_default_loss_cannot_move_this_model_at_all(self) -> None:
        """A zero output is a stationary point of a mean-of-squares loss.

        Recorded so nobody "simplifies" the objective back to the default and
        then widens ``expect_zero`` until the result fits.
        """
        model, inputs, _, _ = fresh()
        report = gradient_report(model, inputs, loss_fn=default_loss, training=True)
        assert len(report) == TRAINABLE_TENSORS, len(report)
        assert dead_count(report) == DEAD_UNDER_DEFAULT_LOSS, dead_count(report)

    def test_the_default_loss_stays_a_fixed_point_across_optimizer_steps(self) -> None:
        """Not merely dead at init -- UNREACHABLE. The step is a no-op."""
        model, inputs, y_true, loss = fresh()
        before = [np_(w).copy() for w in model.trainable_weights]
        model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.5), loss="mse")
        model.fit(
            inputs,
            np.zeros_like(np_(model(inputs, training=False))),
            batch_size=BATCH,
            epochs=3,
            verbose=0,
        )
        after = [np_(w) for w in model.trainable_weights]
        moved = sum(1 for a, b in zip(before, after) if np.any(a != b))
        assert moved == 0, (
            f"{moved} tensors moved under a mean-of-squares loss on a zero "
            "output; the fixed-point claim above is no longer true"
        )

    def test_at_initialisation_almost_everything_reads_dead(self) -> None:
        model, inputs, y_true, loss = fresh()
        report = gradient_report(
            model, inputs, loss_fn=objective(loss, y_true), training=True
        )
        assert len(report) == TRAINABLE_TENSORS, len(report)
        assert dead_count(report) == DEAD_AT_INIT, dead_count(report)

    def test_the_two_live_tensors_at_init_are_the_final_read_out(self) -> None:
        """WHICH two, not just how many -- the number alone is a coincidence.

        Only the final layer's read-out projection sees a gradient before any
        training, because it is the one zero-init layer whose INPUT is non-zero.
        """
        model, inputs, y_true, loss = fresh()
        report = gradient_report(
            model, inputs, loss_fn=objective(loss, y_true), training=True
        )
        live = sorted(
            path.split("/", 1)[-1]
            for path, value in report.items()
            if value is not None and value != 0.0
        )
        assert live == ["final_layer/linear/bias", "final_layer/linear/kernel"], live

    def test_one_optimizer_step_is_not_enough(self) -> None:
        """adaLN-Zero puts two zero-init layers in SERIES on the same path."""
        model, inputs, y_true, loss = fresh()
        train_steps(model, inputs, y_true, loss, steps=1)
        report = gradient_report(
            model, inputs, loss_fn=objective(loss, y_true), training=True
        )
        assert dead_count(report) == DEAD_AFTER_ONE_STEP, dead_count(report)


# ---------------------------------------------------------------------
# The claim
# ---------------------------------------------------------------------


class TestAfterTrainingStartsNothingIsStranded:
    """Zero dead weights, zero waivers, after two real optimizer steps."""

    def test_every_trainable_weight_receives_a_gradient(self) -> None:
        model, inputs, y_true, loss = fresh()
        train_steps(model, inputs, y_true, loss, steps=2)
        report = assert_gradients_reach_every_trainable_weight(
            model,
            inputs,
            loss_fn=objective(loss, y_true),
            expect_zero=(),  # no waivers; see the module docstring
            training=True,
        )
        assert len(report) == TRAINABLE_TENSORS, len(report)
        assert all(np.isfinite(v) for v in report.values()), report

    def test_the_frozen_tables_did_not_move_across_those_steps(self) -> None:
        """The claim the trainable-only report cannot make.

        ``pos_embed`` and the timestep frequency ladder are non-trainable, so
        the oracle never sees them. If an optimizer ever did reach them, this
        file would stay green and the model would silently drift off the
        published sin-cos table.
        """
        model, inputs, y_true, loss = fresh()
        frozen = [w for w in model.weights if not w.trainable]
        assert sorted(w.path.split("/", 1)[-1] for w in frozen) == [
            "pos_embed",
            "t_embedder/freqs",
        ]
        before = [np_(w).copy() for w in frozen]
        train_steps(model, inputs, y_true, loss, steps=2)
        for weight, original in zip(frozen, before):
            np.testing.assert_allclose(np_(weight), original, rtol=0, atol=0.0)

    def test_the_assertion_is_capable_of_failing(self) -> None:
        """The shared dead-component injection. Every weight leaves the graph."""
        model, inputs, y_true, loss = fresh()
        train_steps(model, inputs, y_true, loss, steps=2)
        with broken_forward(model, stop_all_gradients):
            with pytest.raises(AssertionError, match="NO gradient|zero"):
                assert_gradients_reach_every_trainable_weight(
                    model, inputs, loss_fn=objective(loss, y_true), training=True
                )

    def test_one_disconnected_variable_is_convicted_on_its_own(self) -> None:
        """The sharper anti-vacuity arm: ONE variable leaves the graph.

        ``stop_all_gradients`` proves the oracle can fail when EVERYTHING is
        dead, which a report that simply crashed would also satisfy. This arm
        disconnects exactly the label table -- by dropping ``y_embedder`` from
        ``c = t_emb + y_emb`` -- and requires the oracle to name that one path
        while the other 38 stay live. A dead-weight detector that cannot
        localize is not a detector.
        """
        model, inputs, y_true, loss = fresh(_DiTWithTheLabelPathCut)
        train_steps(model, inputs, y_true, loss, steps=2)
        report = gradient_report(
            model, inputs, loss_fn=objective(loss, y_true), training=True
        )
        dead = sorted(
            path.split("/", 1)[-1]
            for path, value in report.items()
            if value is None or value == 0.0
        )
        assert dead == ["y_embedder/embedding_table/embeddings"], dead

        with pytest.raises(AssertionError, match="y_embedder"):
            assert_gradients_reach_every_trainable_weight(
                model, inputs, loss_fn=objective(loss, y_true), training=True
            )

    def test_the_conditioning_path_is_reached_from_both_embedders(self) -> None:
        """A stronger, model-specific claim on top of the oracle's report.

        ``c = t_emb + y_emb`` is the only route by which the timestep and the
        class label reach anything. A port that dropped either addend would
        still emit the right shape, still round-trip, and still train -- into a
        model that ignores its conditioning.
        """
        model, inputs, y_true, loss = fresh()
        train_steps(model, inputs, y_true, loss, steps=2)
        report = gradient_report(
            model, inputs, loss_fn=objective(loss, y_true), training=True
        )
        for name in ("t_embedder", "y_embedder"):
            reached = {p: v for p, v in report.items() if f"/{name}/" in p}
            assert reached, f"no weight path contains {name!r}"
            assert all(v is not None and v > 0.0 for v in reached.values()), (
                name,
                reached,
            )


class _DiTWithTheLabelPathCut(DiT):
    """Seeded defect: the conditioning vector drops its label term.

    ``c = t_emb`` instead of ``c = t_emb + y_emb``. The label table is still
    CONSTRUCTED, still built, still serialized and still counted -- it is simply
    not on the graph. That is the failure a gradient-flow oracle exists for, and
    it has no shape, count, config or round-trip symptom.
    """

    def call(
        self,
        inputs: Union[List[Any], Tuple[Any, ...], dict],
        training: Optional[bool] = None,
    ) -> Any:
        x, t, y = inputs[0], inputs[1], inputs[2]
        tokens = self.x_embedder(x, training=training)
        tokens = tokens + keras.ops.cast(self.pos_embed, tokens.dtype)
        self.y_embedder(y, training=training)  # built and run, but discarded
        c = self.t_embedder(t, training=training)
        for block in self.blocks:
            tokens = block([tokens, c], training=training)
        return self.unpatchify(self.final_layer([tokens, c], training=training))
