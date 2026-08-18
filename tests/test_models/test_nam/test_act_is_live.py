"""RED proofs for C-35 (plan-2026-08-14T233721-d4f9beb2, D-033/D-034).

(a) `NAM.call` used to set `new_halted = is_last_step` whenever `training` was
    not exactly `True`, so at inference every sequence ran the full
    `halt_max_steps` and the learned `q_halt` head was never consulted — flatly
    contradicting the class docstring's "simple expressions like ``1 + 2`` take
    1 step". (D-033)
(b) `src/train/nam/train_nam.py` collected `q_halt_logits` into `all_q_halt` and
    consumed it in NO loss; there was no `L_halt` in the file. The head was
    neither trained nor read. (D-034)
(d) `src/dl_techniques/models/CLAUDE.md` filed the package as "Neural additive
    model"; it is a Neural Arithmetic MODULE.

The halting probes drive `q_halt` by writing the cell's `halt_head` weights
directly, so each arm's halting decision is DICTATED rather than inferred from
whatever a randomly initialised head happens to emit.
"""

import pathlib

import numpy as np
import keras
import pytest
import tensorflow as tf

from dl_techniques.models.nam import NAM, NAMConfig

HALT_MAX_STEPS = 4


@pytest.fixture
def config():
    return NAMConfig(
        hidden_size=32,
        num_heads=4,
        num_tree_layers=1,
        intermediate_size=64,
        memory_size=8,
        num_read_heads=2,
        max_expression_len=16,
        halt_max_steps=HALT_MAX_STEPS,
        hidden_dropout_rate=0.0,
        attention_dropout_rate=0.0,
    )


@pytest.fixture
def batch():
    ids = np.zeros((2, 16), dtype="int32")
    ids[:, :5] = np.array([[1, 5, 12, 7, 2], [1, 9, 13, 3, 2]], dtype="int32")
    return {"input_ids": ids}


def _pin_q_halt(model, q_halt_value):
    """Force `q_halt` to a constant, whatever the hidden state is.

    Zeroing the kernel makes the head input-independent, so the arm's halting
    decision comes from `q_halt_value` alone and nothing else.
    """
    head = model.cell.halt_head
    kernel, bias = head.kernel, head.bias
    kernel.assign(keras.ops.zeros_like(kernel))
    bias.assign(keras.ops.convert_to_tensor([q_halt_value, 0.0], dtype=bias.dtype))


def _run_to_halt(model, batch, training=False):
    """Drive the ACT loop and return the 1-based step each sequence halted on."""
    carry = model.initial_carry(batch)
    halted_at = np.zeros(batch["input_ids"].shape[0], dtype="int32")
    for step in range(1, HALT_MAX_STEPS + 1):
        carry, _ = model(carry, batch, training=training)
        halted = np.asarray(keras.ops.convert_to_numpy(carry["halted"])).reshape(-1)
        halted_at = np.where((halted_at == 0) & halted, step, halted_at)
        if halted_at.all():
            break
    return halted_at


class TestInferenceConsultsTheLearnedHaltSignal:
    """(a) D-033."""

    def test_a_positive_q_halt_halts_on_the_first_step(self, config, batch):
        keras.utils.set_random_seed(0)
        model = NAM(config=config)
        model(model.initial_carry(batch), batch, training=False)
        _pin_q_halt(model, +5.0)

        halted_at = _run_to_halt(model, batch, training=False)
        assert np.array_equal(halted_at, np.array([1, 1])), (
            f"with q_halt pinned to +5 at inference the sequences halted at "
            f"{halted_at.tolist()}, not [1, 1] — the learned halt signal is not "
            "consulted outside training (decisions.md D-033)"
        )

    def test_a_negative_q_halt_still_runs_the_full_budget(self, config, batch):
        """ANTI-VACUITY control: halting is not simply 'always step 1'.

        Without this arm, the assertion above is satisfied by any change that
        makes the model halt immediately regardless of `q_halt`.
        """
        keras.utils.set_random_seed(0)
        model = NAM(config=config)
        model(model.initial_carry(batch), batch, training=False)
        _pin_q_halt(model, -5.0)

        halted_at = _run_to_halt(model, batch, training=False)
        assert np.array_equal(
            halted_at, np.full(2, HALT_MAX_STEPS, dtype="int32")
        ), (
            f"with q_halt pinned to -5 the sequences halted at "
            f"{halted_at.tolist()}, not at the {HALT_MAX_STEPS}-step ceiling — "
            "the halt signal's SIGN is being ignored"
        )

    def test_training_and_inference_agree_on_the_predicate(self, config, batch):
        """The two branches must halt on the same rule, exploration aside."""
        keras.utils.set_random_seed(0)
        model = NAM(config=config)
        model(model.initial_carry(batch), batch, training=False)
        _pin_q_halt(model, +5.0)
        object.__setattr__(model.config, "halt_exploration_prob", 0.0)

        train_halt = _run_to_halt(model, batch, training=True)
        eval_halt = _run_to_halt(model, batch, training=False)
        assert np.array_equal(train_halt, eval_halt), (
            f"training halted at {train_halt.tolist()} but inference at "
            f"{eval_halt.tolist()} for the same pinned q_halt; the head would be "
            "trained under one rule and read under another"
        )


class TestHaltHeadReceivesGradient:
    """(b) D-034 — the halting loss must actually reach `halt_head`."""

    def _halt_loss(self, model, batch, targets):
        """The `L_halt` term of `train_nam.py`, in the same shape."""
        carry = model.initial_carry(batch)
        carry, outputs = model(carry, batch, training=True)
        rel_error = tf.abs(outputs["result"] - targets) / (tf.abs(targets) + 1e-8)
        step_correct = tf.cast(tf.less(rel_error, 0.01), tf.float32)
        return tf.reduce_mean(
            keras.losses.binary_crossentropy(
                tf.stop_gradient(step_correct),
                tf.expand_dims(outputs["q_halt_logits"], axis=-1),
                from_logits=True,
            )
        )

    def test_l_halt_moves_halt_head_and_the_probe_is_live(self, config, batch):
        keras.utils.set_random_seed(0)
        model = NAM(config=config)
        model(model.initial_carry(batch), batch, training=True)
        targets = tf.constant([[7.0], [12.0]])

        with tf.GradientTape() as tape:
            loss = self._halt_loss(model, batch, targets)
        kernel = model.cell.halt_head.kernel
        grad = tape.gradient(loss, kernel)

        assert grad is not None, (
            "L_halt produced NO gradient path to halt_head.kernel — the "
            "halting head is untrainable (decisions.md D-034)"
        )
        magnitude = float(np.max(np.abs(np.asarray(grad))))
        assert magnitude > 1e-8, (
            f"L_halt's gradient w.r.t. halt_head.kernel is {magnitude:.3e}; the "
            "term is present but inert"
        )

    def test_the_result_head_is_shielded_from_the_halting_target(
        self, config, batch
    ):
        """`L_halt` must not become a second objective on `result`.

        HONEST SCOPE: this pins the PROPERTY, not the `stop_gradient` that
        states it. Measured on this stack, `cast(less(...))` is already
        non-differentiable — `tape.gradient` through it returns `None` — so
        removing the explicit `stop_gradient` would not make this assertion
        fire. The barrier is documentation of intent plus insurance against a
        future soft/relaxed correctness target; the assertion guards the
        invariant either way, and would fire the moment someone swaps the hard
        threshold for a differentiable surrogate without re-adding a barrier.
        """
        keras.utils.set_random_seed(0)
        model = NAM(config=config)
        model(model.initial_carry(batch), batch, training=True)
        targets = tf.constant([[7.0], [12.0]])

        with tf.GradientTape() as tape:
            loss = self._halt_loss(model, batch, targets)
        result_kernel = model.result_head.kernel
        grad = tape.gradient(loss, result_kernel)

        magnitude = 0.0 if grad is None else float(
            np.max(np.abs(np.asarray(grad)))
        )
        assert magnitude == 0.0, (
            f"L_halt reached result_head.kernel with magnitude {magnitude:.3e}; "
            "the correctness target is not stop_gradient'ed and the loss can be "
            "minimised by moving the prediction instead of the halt signal"
        )


class TestPackageIsFiledUnderItsRealArchitecture:
    """(d) — the index entry must name what the code is."""

    def test_models_claude_md_does_not_call_nam_a_neural_additive_model(self):
        root = pathlib.Path(__file__).resolve().parents[3]
        text = (root / "src" / "dl_techniques" / "models" / "CLAUDE.md").read_text()
        entry = next(
            line for line in text.splitlines() if line.startswith("- `nam/`")
        )
        assert "additive" not in entry.lower() or "Not" in entry, (
            f"models/CLAUDE.md files nam/ as {entry!r}; the package is a Neural "
            "Arithmetic MODULE (tree parse + NTM memory + TRM halting) and "
            "contains no per-feature additive model"
        )
        assert "Arithmetic" in entry, (
            f"models/CLAUDE.md's nam/ entry {entry!r} does not name the actual "
            "architecture"
        )
