"""
Construction + M2 .keras round-trip test for the Hierarchical Reasoning Model.

HRM has a build() override (SAM D-008 pattern). call() dispatches a dict batch
({token_ids (B,T), puzzle_ids (B,)}) and returns a dict with logits +
q_halt_logits + q_continue_logits. Pins identical outputs across save -> load.
"""

import os
import keras
import pytest
import numpy as np

from dl_techniques.models.hierarchical_reasoning_model.model import (
    create_hierarchical_reasoning_model,
)

SEQ_LEN = 16


def _model():
    return create_hierarchical_reasoning_model(
        vocab_size=256, seq_len=SEQ_LEN, variant="micro")


def _batch(batch=2):
    rng = np.random.default_rng(0)
    return {
        "token_ids": rng.integers(0, 256, (batch, SEQ_LEN)).astype("int32"),
        "puzzle_ids": rng.integers(0, 1000, (batch,)).astype("int32"),
    }


class TestHRM:

    def test_forward_dict(self):
        out = _model()(_batch(), training=False)
        assert "logits" in out
        for v in out.values():
            arr = keras.ops.convert_to_numpy(v)
            assert not np.any(np.isnan(arr))

    def test_training_mode_forward_runs(self):
        """The training path must execute at all.

        RED-proof: with ``keras.random.uniform(..., dtype="int32")`` at
        model.py:697 this raises ``ValueError: keras.random.uniform requires a
        floating point dtype`` before producing anything, and with the
        ``if not is_last_step:`` guard it raises on the ambiguous truth value of
        a ``(batch,)`` bool tensor. Both fire under the DEFAULT config
        (halt_max_steps=16, halt_exploration_prob=0.1), so every existing test
        avoided them only by passing training=False.

        Asserting merely that the model is constructible would be VACUOUS —
        construction was always fine; the forward pass was not.
        """
        out = _model()(_batch(batch=4), training=True)
        assert "logits" in out
        assert "target_q_continue" in out, (
            "training mode must emit the Bellman bootstrap target")
        for name, v in out.items():
            arr = keras.ops.convert_to_numpy(v)
            assert np.all(np.isfinite(arr)), f"{name} is not finite"

    def test_bellman_target_is_detached(self):
        """The bootstrap target must carry no gradient (B-3).

        A differentiable target lets the TD loss be minimised by dragging the
        target toward the prediction instead of fitting it — the standard
        target-network collapse. tiny_recursive_model already stop_gradients it.
        """
        import tensorflow as tf

        model = _model()
        x = _batch(batch=4)
        model(x, training=True)  # materialise weights

        with tf.GradientTape() as tape:
            out = model(x, training=True)
            target = keras.ops.sum(out["target_q_continue"])
        grads = tape.gradient(target, model.trainable_variables)

        moved = [
            v.name for g, v in zip(grads, model.trainable_variables)
            if g is not None and float(keras.ops.max(keras.ops.abs(g))) > 0.0
        ]
        assert not moved, (
            f"target_q_continue is differentiable w.r.t. {len(moved)} variables "
            f"(e.g. {moved[:3]}); it must be stop_gradient-ed")

    def test_fit_runs_under_graph_mode(self):
        """``fit()`` must work, which means the whole call path must be graph-safe.

        RED-proof: with the old ``for _ in range(...): if all_finished: break``
        in ``_forward_complete``, this raises OperatorNotAllowedInGraphError —
        ``all_finished`` is a symbolic scalar and ``fit()`` traces. The eager
        tests above all pass in that state, which is exactly why it survived.

        Asserting only that the loss is finite would be weak, so this also pins
        that weights actually move.
        """
        model = _model()
        x = _batch(batch=8)
        rng = np.random.default_rng(1)
        y = {"logits": rng.integers(0, 256, (8, SEQ_LEN)).astype("int32")}

        model.compile(
            optimizer="adam",
            loss={"logits": keras.losses.SparseCategoricalCrossentropy(
                from_logits=True)},
        )
        # Materialise the weights BEFORE snapshotting: an unbuilt subclassed
        # model has an EMPTY trainable_variables, which would make the
        # "weights moved" check below vacuously zero-length rather than false.
        model(x, training=False)
        before = [keras.ops.convert_to_numpy(v).copy()
                  for v in model.trainable_variables]
        assert before, "model must be built before snapshotting weights"
        history = model.fit(x, y, epochs=1, batch_size=4, verbose=0)

        assert np.isfinite(history.history["loss"][0])
        moved = sum(
            1 for a, v in zip(before, model.trainable_variables)
            if np.max(np.abs(a - keras.ops.convert_to_numpy(v))) > 0.0
        )
        assert moved > 0, "fit() ran but no trainable weight moved"

    def test_keras_round_trip(self, tmp_path):
        model = _model()
        x = _batch()
        before = keras.ops.convert_to_numpy(model(x, training=False)["logits"])

        path = os.path.join(str(tmp_path), "hrm.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        after = keras.ops.convert_to_numpy(loaded(x, training=False)["logits"])

        # GPU fp32 reduction noise -> atol 1e-4 (SYSTEM invariant)
        np.testing.assert_allclose(before, after, atol=1e-4,
                                   err_msg="HRM differs after .keras round-trip")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
