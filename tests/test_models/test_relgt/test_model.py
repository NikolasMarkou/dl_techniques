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

    def test_the_broadcast_summary_does_not_swamp_the_token_signal(self):
        """The chained sequence must stay TOKEN-specific, not become a constant.

        The block adds its ``(B, E)`` summary onto every token. Both that summary
        and the local tokens leave a LayerNorm, so both have per-feature RMS
        exactly 1.0 — while the token-VARYING part of the sequence is much
        smaller. Added at unit weight the summary is therefore a token-INVARIANT
        component LARGER than the signal it is added to, and the next block's
        self-attention sees a query-independent per-key bias that pushes every
        query toward the same distribution (attention degenerating toward
        pooling). Gradient reachability — the only property the test above pins —
        cannot see this at all.

        MEASURED on THIS fixture (``embedding_dim=32``, 3 blocks, untrained,
        constant token stub), at three values of ``SUMMARY_BROADCAST_SCALE``:

        =========================  =======  ==========  =======
        quantity                       1.0  0.1 (ship)     10.0
        =========================  =======  ==========  =======
        injected RMS                1.0000      0.1000  10.0000
        injected RMS / block-0 std  1.1402      0.1178  11.7894
        block-1 across-token std    0.5441      0.7678   0.0594
        =========================  =======  ==========  =======

        The fixture is not seeded, so these move by a few percent per process
        (block-0 std was 0.8770 / 0.8488 / 0.8482 in the three runs above); the
        injected RMS does not move at all, because a LayerNorm output has
        per-feature RMS exactly 1.0. The floors are set from those numbers with
        an order of magnitude of margin on each side: ratio < 1.0 fires at 1.0
        (the pre-fix, unscaled add), and block-1 std > 0.20 fires at 10.0. On the
        SHIPPED model path (real ``RELGTTokenEncoder``, where the across-token
        std is smaller) the unscaled ratio is worse still: 2.60 at block 0 and
        3.94 at block 1.
        """
        model, tokens = self._probe_model(num_blocks=3)
        x = _inputs()
        _ = model(x, training=False)  # build seed_encoder and every block

        seed = model.seed_encoder(x["node_features"][:, 0:1, :])
        block0, block1 = model.transformer_blocks[0], model.transformer_blocks[1]

        _, out0 = block0([tokens, seed], training=False)
        local0 = block0.local_transformer(tokens, training=False)

        # What the block ACTUALLY injects, derived by difference — deliberately
        # not `SUMMARY_BROADCAST_SCALE * summary`, which would make the test
        # self-referential (raising the constant would raise the numerator AND
        # the reference, and the assertion could never fire).
        injected = keras.ops.convert_to_numpy(out0 - local0)
        out0_np = keras.ops.convert_to_numpy(out0)

        # It really is token-invariant, so a single scalar describes its scale.
        assert float(np.max(np.std(injected, axis=1))) < 1e-5, (
            "the injected component varies across tokens; the ratio below no "
            "longer measures a shared-vs-distinct decomposition"
        )

        injected_rms = float(np.sqrt(np.mean(injected ** 2)))
        block0_std = float(np.mean(np.std(out0_np, axis=1)))

        # Liveness: a block that stopped adding the summary at all would make
        # the ratio 0 and pass vacuously (and would re-strand 32% of the
        # parameters — see the gradient test above).
        assert injected_rms > 1e-3, (
            f"the block injects nothing (RMS {injected_rms}); the summary is not "
            "reaching the token sequence"
        )

        ratio = injected_rms / block0_std
        assert ratio < 1.0, (
            f"the broadcast summary ({injected_rms:.4f} per-feature RMS) is "
            f"larger than the token-varying signal it is added to "
            f"(across-token per-feature std {block0_std:.4f}, ratio {ratio:.4f}): "
            "block 1's self-attention receives a query-independent bias larger "
            "than its content term"
        )

        # Block 1 — the first block that CONSUMES a chained sequence — must still
        # produce tokens that differ from each other.
        _, out1 = block1([out0, seed], training=False)
        block1_std = float(np.mean(np.std(keras.ops.convert_to_numpy(out1), axis=1)))
        assert block1_std > 0.20, (
            f"block 1's output tokens are nearly identical to each other "
            f"(across-token per-feature std {block1_std:.4f}): the chained "
            "sequence has collapsed toward a constant"
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
