"""RED proofs for C-42 (plan-2026-08-14T233721-d4f9beb2, D-030/D-031/D-032).

Three independent sub-defects, one assertion group each:

(a) `HierarchicalReasoningCore.h_init` / `l_init` were declared `trainable=True`
    and reported in `count_params()`, but every path from them to the loss
    crosses the truncated-BPTT `stop_gradient` in `call`, so the optimizer
    never touched them. (D-030)
(b) `_forward_complete` froze carry and outputs on the BATCH-GLOBAL
    `ops.all(halted)` scalar. A sequence that halted early was reset and re-run
    until the batch's slowest member finished, and the caller got its restarted
    partial run. (D-031)
(c) `create_hierarchical_reasoning_model` documented an "optionally compiled"
    model and built an optimizer it discarded, so `.fit()` raised. (D-032)
"""

import numpy as np
import keras
import pytest
import tensorflow as tf

from dl_techniques.models.hierarchical_reasoning_model.model import (
    HierarchicalReasoningModel,
    create_hierarchical_reasoning_model,
)

CFG = dict(
    vocab_size=16,
    seq_len=8,
    embed_dim=32,
    num_puzzle_identifiers=4,
    h_layers=1,
    l_layers=1,
    num_heads=2,
    halt_max_steps=4,
    halt_exploration_prob=0.0,
    batch_size=2,
)


@pytest.fixture
def batch():
    return {
        "token_ids": np.array(
            [[1, 2, 3, 4, 5, 6, 7, 8], [2, 3, 4, 5, 6, 7, 8, 1]], dtype="int32"
        ),
        "puzzle_ids": np.array([0, 1], dtype="int32"),
    }


@pytest.fixture
def model():
    keras.utils.set_random_seed(0)
    return HierarchicalReasoningModel(**CFG)


class TestInitialStatesAreHonestlyNonTrainable:
    """(a) D-030 — the reported trainability must match the measured one."""

    def test_h_init_and_l_init_are_not_reported_trainable(self, model, batch):
        model(batch, training=False)
        core = model.core
        assert not core.h_init.trainable, (
            "h_init is reported trainable, but `call` stop_gradients the "
            "incoming carry that is the only path from it to the loss — see "
            "the companion gradient test and decisions.md D-030"
        )
        assert not core.l_init.trainable, "l_init: same argument as h_init"

    def test_the_gradient_really_is_absent_and_the_tape_really_is_live(
        self, model, batch
    ):
        """The measurement behind the flag, with its own LIVENESS ARM.

        A bare "gradient is None" assertion is satisfied by any broken tape.
        `lm_head.kernel` is differentiated on the SAME tape, in the same call,
        and must come back non-zero — so a `None` for `h_init` is a real
        gradient barrier and not a dead probe.
        """
        with tf.GradientTape() as tape:
            outputs = model(batch, training=True)
            loss = keras.ops.mean(outputs["logits"] ** 2)

        core = model.core
        g_h, g_l, g_live = tape.gradient(
            loss, [core.h_init, core.l_init, core.lm_head.kernel]
        )

        assert g_live is not None, "liveness arm: the tape returned nothing"
        live_mag = float(np.max(np.abs(np.asarray(g_live))))
        assert live_mag > 1e-6, (
            f"liveness arm: lm_head.kernel gradient is {live_mag:.3e}; the tape "
            "is dead, so the h_init result below proves nothing"
        )

        for name, grad in (("h_init", g_h), ("l_init", g_l)):
            if grad is not None:
                mag = float(np.max(np.abs(np.asarray(grad))))
                assert mag == 0.0, (
                    f"{name} received a NON-ZERO gradient ({mag:.3e}) — the "
                    "truncated-BPTT barrier changed, so D-030's premise that "
                    "these are effectively buffers no longer holds and their "
                    "`trainable` flag must be revisited"
                )

    def test_flipping_trainable_left_the_saved_weight_layout_alone(
        self, model, batch, tmp_path
    ):
        """The cost claim in D-030, pinned rather than asserted in prose."""
        model(batch, training=False)
        paths = [w.path for w in model.weights]
        assert paths[0].endswith("h_init") and paths[1].endswith("l_init"), (
            "h_init/l_init are no longer the first two weights; the "
            "checkpoint-compatibility claim in D-030 rested on that ordering"
        )

        path = tmp_path / "hrm.keras"
        model.save(path)
        restored = keras.models.load_model(path)
        before = np.asarray(model(batch, training=False)["logits"])
        after = np.asarray(restored(batch, training=False)["logits"])
        assert np.max(np.abs(before - after)) == 0.0, (
            "a .keras round-trip no longer reproduces the logits bit-exactly"
        )


class _ScriptedHaltModel(HierarchicalReasoningModel):
    """A model whose halting schedule and per-step output are DICTATED.

    `_forward_step` is replaced wholesale so the (b) assertion depends on the
    halting DEFINITION — "a halted sequence's returned output is the one it
    produced at the step it halted on" — and not on anything the reasoning core
    computes. `_forward_complete` is untouched, and it is the only thing under
    test.

    The schedule is `halt_at`: sequence i halts at 1-based step `halt_at[i]`.
    The step's output for every sequence is the 1-based step index, so the
    expected final output is just `halt_at` itself.
    """

    def __init__(self, halt_at, **kwargs):
        super().__init__(**kwargs)
        self._halt_at = np.asarray(halt_at, dtype="int32")
        self._script_step = 0

    def _forward_step(self, carry, batch, training=None):
        self._script_step += 1
        step = self._script_step
        n = len(self._halt_at)

        halted = keras.ops.convert_to_tensor(self._halt_at <= step)
        outputs = {
            "logits": keras.ops.full((n, 1), float(step)),
            "q_halt_logits": keras.ops.full((n,), float(step)),
        }
        new_carry = {
            "steps": keras.ops.full((n,), step, dtype="int32"),
            "halted": halted,
            "marker": keras.ops.full((n,), float(step)),
        }
        return new_carry, outputs, keras.ops.all(halted)


class TestEarlyHaltingSequenceKeepsItsOwnAnswer:
    """(b) D-031 — the freeze is per sequence, not batch-global."""

    def test_each_sequence_returns_what_it_produced_at_its_halting_step(
        self, batch
    ):
        halt_at = [1, 4]  # sequence 0 halts immediately; sequence 1 runs on
        keras.utils.set_random_seed(0)
        model = _ScriptedHaltModel(halt_at, **CFG)

        outputs = model._forward_complete(batch, training=False)
        got = np.asarray(outputs["logits"]).reshape(-1)

        # Built from the halting definition alone: sequence i's answer is the
        # one it emitted at step halt_at[i], and the script makes that value
        # equal to halt_at[i].
        expected = np.asarray(halt_at, dtype="float32")
        assert np.array_equal(got, expected), (
            f"got {got.tolist()}, expected {expected.tolist()}: an early-halting "
            "sequence's output was overwritten by a later, RESTARTED step — the "
            "freeze is gating on a batch-global scalar (decisions.md D-031)"
        )

    def test_a_lockstep_batch_is_unaffected(self, batch):
        """ANTI-VACUITY control: the batch-global gate was RIGHT here.

        With every sequence halting on the same step the old scalar and the new
        per-sequence mask agree exactly, so this must pass under both. If it
        ever fails, the per-sequence rewrite broke the common case and the test
        above is measuring the wrong thing.
        """
        keras.utils.set_random_seed(0)
        model = _ScriptedHaltModel([2, 2], **CFG)
        outputs = model._forward_complete(batch, training=False)
        got = np.asarray(outputs["logits"]).reshape(-1)
        assert np.array_equal(got, np.asarray([2.0, 2.0])), (
            f"a lockstep batch returned {got.tolist()}, not [2.0, 2.0]"
        )

    def test_the_script_itself_asserts_a_split_batch(self, batch):
        """Third control: the schedule really does split at step 1.

        If both sequences halted together the (b) assertion would be satisfied
        by the batch-global gate too, and would prove nothing.
        """
        keras.utils.set_random_seed(0)
        stepper = _ScriptedHaltModel([1, 4], **CFG)
        carry, _, all_finished = stepper._forward_step(None, batch, training=False)
        assert np.array_equal(
            np.asarray(carry["halted"]), np.asarray([True, False])
        ), "the scripted schedule itself is wrong; the (b) proof is invalid"
        assert not bool(np.asarray(all_finished)), (
            "ops.all(halted) is already True at step 1, so the batch-global gate "
            "and the per-sequence mask would agree and (b) would be vacuous"
        )


class TestFactoryReturnsACompiledModel:
    """(c) D-032 — 'optionally compiled' must mean compiled."""

    def _factory(self, **kwargs):
        keras.utils.set_random_seed(0)
        return create_hierarchical_reasoning_model(
            vocab_size=16,
            seq_len=8,
            embed_dim=32,
            num_puzzle_identifiers=4,
            h_layers=1,
            l_layers=1,
            num_heads=2,
            halt_max_steps=2,
            **kwargs,
        )

    def test_fit_succeeds_without_a_manual_compile(self, batch):
        model = self._factory()
        assert model.compiled, (
            "the factory returned an uncompiled model while its docstring "
            "promises an optionally compiled one (decisions.md D-032)"
        )
        # R-038 / D-056: the factory's default compile supervises `logits`
        # ONLY -- the Q/ACT term couples two model OUTPUTS and cannot be a
        # per-output Keras loss (D-032). Keras therefore reports the q_head as
        # gradient-free, DELIBERATELY. Asserted rather than suppressed; the
        # paired liveness control is
        # test_the_q_head_trains_where_it_is_supposed_to.py.
        with pytest.warns(UserWarning, match="Gradients do not exist"):
            history = model.fit(
                batch,
                {"logits": batch["token_ids"]},
                epochs=1,
                verbose=0,
                batch_size=2,
            )
        loss = history.history["loss"][0]
        assert np.isfinite(loss), f"fit ran but produced a non-finite loss {loss}"

    def test_the_learning_rate_argument_reaches_the_optimizer(self):
        model = self._factory(learning_rate=3e-3)
        lr = float(keras.ops.convert_to_numpy(model.optimizer.learning_rate))
        assert lr == pytest.approx(3e-3), (
            f"learning_rate=3e-3 produced an optimizer at {lr}; the argument is "
            "inert again"
        )

    def test_optimizer_none_still_returns_an_uncompiled_model(self):
        """ANTI-VACUITY control: the opt-out route stays open.

        `src/train/hrm/train_hrm.py` drives this model from its own
        `GradientTape` with `create_hrm_loss`, so compiling unconditionally
        would be a different defect.
        """
        model = self._factory(optimizer=None)
        assert not model.compiled, (
            "optimizer=None compiled the model anyway; the custom-loop route "
            "documented in the README no longer has an opt-out"
        )
