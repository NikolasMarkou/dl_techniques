"""
Construction + M2 round-trip test for RelGT (relational graph transformer).

create_relgt_model(output_dim, model_size) takes a dict of graph tensors and
returns (B, output_dim). NOTE: RelGT performs stochastic local-neighborhood
sampling, so its forward is NOT deterministic even at inference — output-identity
across a save/load cannot be asserted. M2 is therefore verified by confirming
ALL weights are preserved (by order) across the .keras round-trip.
"""

import os
import keras
import pytest
import numpy as np

from dl_techniques.models.relgt.model import RELGT, create_relgt_model

B, N, F = 2, 8, 16


def _inputs():
    rng = np.random.default_rng(0)
    return {
        "node_features": rng.random((B, N, F)).astype("float32"),
        "node_types": rng.integers(0, 10, (B, N)).astype("int32"),
        "hop_distances": rng.integers(0, 3, (B, N)).astype("int32"),
        "relative_times": rng.random((B, N, 1)).astype("float32"),
        "subgraph_adjacency": rng.random((B, N, N)).astype("float32"),
    }


def _model():
    return create_relgt_model(output_dim=2, model_size="small")


class TestRelGT:

    def test_forward_shape(self):
        out = _model()(_inputs(), training=False)
        assert tuple(out.shape) == (B, 2)
        assert not np.any(np.isnan(keras.ops.convert_to_numpy(out)))

    def test_keras_round_trip_weights_preserved(self, tmp_path):
        model = _model()
        x = _inputs()
        _ = model(x, training=False)  # build

        path = os.path.join(str(tmp_path), "relgt.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        _ = loaded(x, training=False)

        assert len(model.weights) == len(loaded.weights)
        # Forward is stochastic -> compare weights (by order) instead of outputs.
        for w_orig, w_load in zip(model.weights, loaded.weights):
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(w_orig),
                keras.ops.convert_to_numpy(w_load),
                atol=1e-6,
                err_msg=f"weight {w_orig.path} not preserved after round-trip",
            )


class _FixedTokenEncoder:
    """Stands in for ``RELGTTokenEncoder``, which is NOT deterministic.

    The real encoder draws fresh ``keras.random.normal`` features for its GNN
    positional encoding on every call, so two forwards of the same model on the
    same inputs differ — which would swamp any delta this probe measures.
    Replacing it with a constant makes ``RELGT.call`` deterministic while leaving
    the transformer-block loop, the part under test, entirely untouched.
    """

    def __init__(self, tokens):
        self.tokens = tokens

    def __call__(self, inputs, training=None):
        return self.tokens


class TestBlockChaining:
    """Blocks must COMPOSE: block ``i+1`` consumes block ``i``'s token output.

    A shape assertion cannot see this — the prediction is ``(B, output_dim)``
    whether the blocks are chained or all fed the same tokens. The distinguishing
    behaviour is whether an EARLIER block's contribution can reach the output at
    all. Unchained, only the last block's output is read, so block 0 is dead
    weight and neutralizing it provably changes nothing.
    """

    K, E = 8, 32

    def _probe_model(self, num_blocks=3):
        model = RELGT(
            output_dim=2,
            embedding_dim=self.E,
            num_heads=2,
            num_global_centroids=4,
            ffn_dim=64,
            num_transformer_blocks=num_blocks,
            dropout_rate=0.0,
        )
        rng = np.random.default_rng(7)
        tokens = keras.ops.convert_to_tensor(
            rng.standard_normal((B, self.K, self.E)).astype("float32")
        )
        model.token_encoder = _FixedTokenEncoder(tokens)
        return model, tokens

    def test_neutralizing_block_zero_changes_the_prediction(self):
        model, tokens = self._probe_model()
        x = _inputs()

        baseline = keras.ops.convert_to_numpy(model(x, training=False))

        # Control: the probe is only meaningful if the forward is deterministic,
        # otherwise any delta below could be resampling noise.
        repeat = keras.ops.convert_to_numpy(model(x, training=False))
        determinism_delta = float(np.max(np.abs(repeat - baseline)))
        assert determinism_delta == 0.0, (
            f"probe forward is not deterministic (delta {determinism_delta}); "
            "every other assertion in this test would be measuring noise"
        )

        # Neutralize block 0 IN THE SAME MODEL INSTANCE — comparing against a
        # second, freshly built model would move both sides and prove nothing.
        block_zero = model.transformer_blocks[0]
        saved = [np.array(keras.ops.convert_to_numpy(w)) for w in block_zero.weights]
        block_zero.set_weights([np.zeros_like(w) for w in saved])
        try:
            neutralized = keras.ops.convert_to_numpy(model(x, training=False))
        finally:
            block_zero.set_weights(saved)

        assert np.all(np.isfinite(neutralized)), "neutralized forward produced non-finite values"

        block_zero_delta = float(np.max(np.abs(neutralized - baseline)))
        assert block_zero_delta > 1e-4, (
            "zeroing TransformerBlock_0 left the prediction unchanged "
            f"(delta {block_zero_delta}): the blocks are NOT chained, only the "
            "last block's output reaches the prediction head"
        )

        # Liveness: a model that returned a constant would fail the assertion
        # above for the wrong reason and could never be caught by it. Perturbing
        # the tokens must move the prediction too.
        restored = keras.ops.convert_to_numpy(model(x, training=False))
        assert float(np.max(np.abs(restored - baseline))) == 0.0, "weight restore failed"

        model.token_encoder = _FixedTokenEncoder(tokens + 1.0)
        perturbed = keras.ops.convert_to_numpy(model(x, training=False))
        liveness_delta = float(np.max(np.abs(perturbed - baseline)))
        assert liveness_delta > 1e-4, (
            f"perturbing the input tokens did not move the prediction "
            f"(delta {liveness_delta}): the probe is degenerate"
        )

    def test_no_trainable_variable_is_gradient_less(self):
        """Every parameter of every block must reach the loss.

        Chaining only the LOCAL token pathway is not enough: a block's global
        module (``global_centroids``, ``GlobalAttention``, ``ResidualProjection``,
        ``CombinationNorm``, ``CombinationFFN``) contributes only to its ``(B, E)``
        summary, and every summary but the LAST one is overwritten by the loop.
        Those parameters are then computed, discarded, and silently skipped by
        ``fit()`` ("Gradients do not exist for variables ..."). This asserts the
        docstring's claim that ``num_transformer_blocks`` buys real depth, which
        the block-zeroing probe above cannot see (the live local path alone
        satisfies it).
        """
        import tensorflow as tf

        model = RELGT(
            output_dim=2,
            embedding_dim=self.E,
            num_heads=2,
            num_global_centroids=4,
            ffn_dim=64,
            num_transformer_blocks=3,
            dropout_rate=0.0,
        )
        x = _inputs()
        _ = model(x, training=False)  # build

        with tf.GradientTape() as tape:
            preds = model(x, training=True)
            loss = tf.reduce_mean(tf.square(preds - 1.0))
        grads = tape.gradient(loss, model.trainable_variables)

        no_grad = [
            v.path for v, g in zip(model.trainable_variables, grads) if g is None
        ]
        assert not no_grad, (
            f"{len(no_grad)} of {len(model.trainable_variables)} trainable variables "
            f"receive NO gradient with num_transformer_blocks=3, so their blocks' "
            f"computation never reaches the output: {no_grad}"
        )

    def test_model_builds_blocks_that_return_tokens(self):
        model, _ = self._probe_model(num_blocks=2)
        for block in model.transformer_blocks:
            assert block.return_tokens is True, (
                f"{block.name} does not return its token sequence, so it cannot be chained"
            )
            assert block.get_config()["return_tokens"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
