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
        """Every parameter of every block must reach the loss with NON-ZERO gradient.

        Two assertions, because ``g is None`` alone is too weak: a change that
        makes a block's contribution vanish numerically (a zero-initialized
        scale, a stop-gradient, a multiply by 0) leaves the graph path intact
        and every gradient non-``None`` while the parameters are just as dead.
        The second assertion therefore requires a gradient norm above 1e-9.

        MEASURED (``num_transformer_blocks=3``, 107 trainable variables): the
        smallest LIVE gradient norm across three unseeded runs was 2.1e-06 /
        9.3e-06 / 1.9e-05 (always a ``GlobalAttention/query/bias``), so the
        floor sits about three orders of magnitude below the real signal.

        Three variables are ALLOW-LISTED at 2e-12..3e-11, i.e. zero plus float
        noise: ``TransformerBlock_{0,1,2}/GlobalAttention/key/bias``. That is
        structural, not a defect — in scaled dot-product attention a key bias
        shifts every logit for a given query by the same amount, and softmax is
        shift-invariant, so it cancels exactly. Keras' ``MultiHeadAttention``
        creates it regardless. Widening this allow-list is almost certainly the
        wrong response to a failure here.

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

        # A key bias cancels in the softmax — see the docstring. Matched by
        # suffix so it covers every block without naming them.
        structurally_zero = "/GlobalAttention/key/bias"
        dead = [
            (v.path, float(tf.norm(g)))
            for v, g in zip(model.trainable_variables, grads)
            if not v.path.endswith(structurally_zero) and float(tf.norm(g)) <= 1e-9
        ]
        assert not dead, (
            f"{len(dead)} of {len(model.trainable_variables)} trainable variables "
            f"have a gradient that is present but numerically ZERO (<= 1e-9), so "
            f"they are dead despite reaching the loss on paper: {dead}"
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
        (the pre-fix, unscaled add), and block-1 std > 0.20 fires at 10.0.

        **The ratio GROWS with depth**, so checking block 0 is not enough: the
        across-token std decays block over block while the injected RMS is
        pinned at the scale by the LayerNorm. This runs at
        ``num_transformer_blocks=4`` — the largest shipped preset
        (``create_relgt_model("large")``) — and asserts on EVERY block. Re-derived
        here at scale 0.1, 20 unseeded draws, per-block ratio (min / max):

        ====== =============== ================================
        block  this fixture    SHIPPED path (real token encoder)
        ====== =============== ================================
        0      0.114 / 0.124   0.230 / 0.493
        1      0.117 / 0.141   0.278 / 0.569
        2      0.124 / 0.156   0.297 / 0.712
        3      0.133 / 0.192   0.317 / **1.023**
        ====== =============== ================================

        Two consequences worth stating plainly. (a) The floors below bound THIS
        FIXTURE, whose constant unit-normal tokens keep the across-token std near
        0.75; on the shipped path the same statistic is ~3x smaller and the ratio
        ~3x larger. (b) On the shipped path the margin at depth 4 is already thin
        — 1 of 20 draws touched 1.023 at block 3 — and past depth 4 the ratio
        exceeds 1.0 routinely (at depth 8, blocks 4-7 crossed on 2-3 of 5 draws).
        ``SUMMARY_BROADCAST_SCALE`` is therefore a DEPTH-BOUNDED fix, documented
        as such at the constant; do not raise ``num_transformer_blocks`` past 4
        without re-measuring.
        """
        num_blocks = 4  # the largest shipped preset: create_relgt_model("large")
        model, tokens = self._probe_model(num_blocks=num_blocks)
        x = _inputs()
        _ = model(x, training=False)  # build seed_encoder and every block

        seed = model.seed_encoder(x["node_features"][:, 0:1, :])

        cur = tokens
        ratios, stds = [], []
        for k in range(num_blocks):
            block = model.transformer_blocks[k]
            _, out = block([cur, seed], training=False)
            local = block.local_transformer(cur, training=False)

            # What the block ACTUALLY injects, derived by difference —
            # deliberately not `SUMMARY_BROADCAST_SCALE * summary`, which would
            # make the test self-referential (raising the constant would raise
            # the numerator AND the reference, and the assertion could never
            # fire).
            injected = keras.ops.convert_to_numpy(out - local)
            out_np = keras.ops.convert_to_numpy(out)

            # It really is token-invariant, so a single scalar describes its
            # scale.
            assert float(np.max(np.std(injected, axis=1))) < 1e-5, (
                f"block {k}'s injected component varies across tokens; the ratio "
                "below no longer measures a shared-vs-distinct decomposition"
            )

            injected_rms = float(np.sqrt(np.mean(injected ** 2)))
            block_std = float(np.mean(np.std(out_np, axis=1)))

            # Liveness: a block that stopped adding the summary at all would make
            # the ratio 0 and pass vacuously (and would re-strand 32% of the
            # parameters — see the gradient test above).
            assert injected_rms > 1e-3, (
                f"block {k} injects nothing (RMS {injected_rms}); the summary is "
                "not reaching the token sequence"
            )

            ratios.append(injected_rms / block_std)
            stds.append(block_std)
            cur = out

        worst = int(np.argmax(ratios))
        assert ratios[worst] < 1.0, (
            f"at block {worst} of {num_blocks} the broadcast summary is larger "
            f"than the token-varying signal it is added to (across-token "
            f"per-feature std {stds[worst]:.4f}, ratio {ratios[worst]:.4f}; all "
            f"blocks: {[round(r, 4) for r in ratios]}): the next block's "
            "self-attention receives a query-independent bias larger than its "
            "content term. The ratio grows with depth — if this fires only at "
            "the deepest block, the model is too DEEP for the current "
            "SUMMARY_BROADCAST_SCALE, not mis-scaled at block 0."
        )

        # The chained blocks must still produce tokens that differ from each
        # other. Checked from block 1 on — block 0 consumes the fixture directly.
        for k in range(1, num_blocks):
            assert stds[k] > 0.20, (
                f"block {k}'s output tokens are nearly identical to each other "
                f"(across-token per-feature std {stds[k]:.4f}; all blocks: "
                f"{[round(s, 4) for s in stds]}): the chained sequence has "
                "collapsed toward a constant"
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
