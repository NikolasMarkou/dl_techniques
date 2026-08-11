"""Test suite for the ``BeitAttention`` layer.

``BeitAttention`` is BEiT's self-attention block: separate q/k/v projections in which
**k carries no bias at all**, plus a learnable relative-position bias table of shape
``((2*Wh - 1) * (2*Ww - 1) + 3, num_heads)`` added to the attention logits before the
softmax. Input and output are both ``(B, Wh*Ww + 1, dim)`` — one cls token followed by
the patch tokens.

Coverage:
1. Initialization & validation raises
2. Relative-position INDEX vs an oracle transcribed from the research finding
3. Forward pass (shapes, head counts, non-square grids, training flag)
4. Bias liveness (the table actually reaches the logits) + a dead-component mutation
5. Structural no-k-bias assertions (exact bias-parameter counts)
6. Gradient flow to every trainable weight including the bias table
7. Serialization: ``get_config`` round-trip and ``.keras`` VALUE equality
8. Edge cases (bias disabled, ``qv_bias=False``, masks)

ORACLE PROVENANCE (read before editing ``TestBeitAttentionRelativePositionIndex``):
``_oracle_relative_position_index`` below is transcribed from the pseudocode in
``plans/plan-2026-08-11T012340-f63796dc/findings/beit-architecture-web.md`` §2, which
was itself fetched from ``microsoft/unilm/beit/modeling_finetune.py``. It is written
with plain Python loops rather than the finding's vectorized numpy so that it is
structurally UNLIKE the implementation: an oracle that re-uses the implementation's own
expression agrees by construction and proves nothing. Do NOT "simplify" it by importing
or copying ``BeitAttention._build_relative_position_index``.
"""

import os
import tempfile

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.attention.beit_attention import BeitAttention


# ==============================================================================
# Oracle — transcribed from findings/beit-architecture-web.md §2, NOT from the port
# ==============================================================================

def _oracle_relative_position_index(wh: int, ww: int) -> np.ndarray:
    """Expected ``(Wh*Ww + 1, Wh*Ww + 1)`` index matrix, built from the finding.

    The finding's construction, restated per-element:

    * patch ``i`` occupies grid cell ``(y_i, x_i)`` with ``i = y_i * Ww + x_i``
      (row-major flattening of a ``(Wh, Ww)`` meshgrid with ``indexing='ij'``);
    * the raw displacement ``(y_i - y_j, x_i - x_j)`` is shifted by ``(Wh-1, Ww-1)``
      so it starts at zero, then the first component is multiplied by ``2*Ww - 1``
      and the two are summed;
    * the cls row, the cls column and the cls/cls cell take ``M-3``, ``M-2`` and
      ``M-1`` respectively, assigned in that order.
    """
    m = (2 * wh - 1) * (2 * ww - 1) + 3
    n = wh * ww
    index = np.zeros((n + 1, n + 1), dtype=np.int64)

    # Row-major flattening: patch p <-> (p // ww, p % ww).
    for pi in range(n):
        yi, xi = divmod(pi, ww)
        for pj in range(n):
            yj, xj = divmod(pj, ww)
            dy = (yi - yj) + (wh - 1)
            dx = (xi - xj) + (ww - 1)
            index[pi + 1, pj + 1] = dy * (2 * ww - 1) + dx

    # Order is load-bearing: row 0, then column 0, then the [0, 0] cell.
    index[0, 0:] = m - 3
    index[0:, 0] = m - 2
    index[0, 0] = m - 1
    return index


def _layer_index_matrix(layer: BeitAttention) -> np.ndarray:
    """Read the built layer's flattened index buffer back as a square matrix."""
    n_tokens = layer.num_tokens
    flat = keras.ops.convert_to_numpy(layer._rel_pos_index)
    return np.asarray(flat).reshape(n_tokens, n_tokens)


def _build(layer: BeitAttention, batch: int = 2) -> np.ndarray:
    """Build ``layer`` against its expected input shape and return a sample input."""
    layer.build((batch, layer.num_tokens, layer.dim))
    rng = np.random.default_rng(0)
    return rng.normal(size=(batch, layer.num_tokens, layer.dim)).astype("float32")


# ==============================================================================
# 1. Initialization & validation
# ==============================================================================

class TestBeitAttentionInitialization:
    """Construction, attribute storage, and every documented ValueError."""

    def test_valid_construction_defaults(self):
        layer = BeitAttention(dim=48, num_heads=4, window_size=(4, 4))
        assert layer.dim == 48
        assert layer.num_heads == 4
        assert layer.head_dim == 12
        assert layer.window_size == (4, 4)
        assert layer.num_patches == 16
        assert layer.num_tokens == 17
        assert layer.num_relative_distance == (2 * 4 - 1) * (2 * 4 - 1) + 3
        assert layer.use_relative_position_bias is True
        assert layer.qv_bias is True
        assert layer.use_proj_bias is True
        np.testing.assert_allclose(layer._scale_value, 12 ** -0.5, rtol=1e-12)

    def test_int_window_size_means_square_grid(self):
        layer = BeitAttention(dim=32, num_heads=4, window_size=5)
        assert layer.window_size == (5, 5)
        assert layer.num_tokens == 26

    def test_non_square_window_size(self):
        layer = BeitAttention(dim=32, num_heads=4, window_size=(3, 5))
        assert layer.window_size == (3, 5)
        assert layer.num_patches == 15
        assert layer.num_relative_distance == (2 * 3 - 1) * (2 * 5 - 1) + 3

    def test_explicit_scale_override(self):
        layer = BeitAttention(dim=32, num_heads=4, window_size=4, scale=0.25)
        assert layer.scale == 0.25
        assert layer._scale_value == 0.25

    @pytest.mark.parametrize("bad_dim", [0, -8])
    def test_non_positive_dim_raises(self, bad_dim):
        with pytest.raises(ValueError, match="dim"):
            BeitAttention(dim=bad_dim, num_heads=4, window_size=4)

    @pytest.mark.parametrize("bad_heads", [0, -3])
    def test_non_positive_num_heads_raises(self, bad_heads):
        with pytest.raises(ValueError, match="num_heads"):
            BeitAttention(dim=32, num_heads=bad_heads, window_size=4)

    def test_indivisible_dim_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            BeitAttention(dim=30, num_heads=4, window_size=4)

    @pytest.mark.parametrize("bad_ws", [0, -4, (0, 4), (4, -1), (3, 0)])
    def test_non_positive_window_size_raises(self, bad_ws):
        with pytest.raises(ValueError, match="window_size"):
            BeitAttention(dim=32, num_heads=4, window_size=bad_ws)

    def test_missing_window_size_raises(self):
        with pytest.raises(ValueError, match="window_size"):
            BeitAttention(dim=32, num_heads=4)

    @pytest.mark.parametrize("bad_ws", [(1, 2, 3), "4", 4.5])
    def test_malformed_window_size_raises(self, bad_ws):
        with pytest.raises(ValueError, match="window_size"):
            BeitAttention(dim=32, num_heads=4, window_size=bad_ws)

    @pytest.mark.parametrize("rate", [-0.1, 1.5])
    def test_attn_dropout_out_of_range_raises(self, rate):
        with pytest.raises(ValueError, match="attn_dropout_rate"):
            BeitAttention(
                dim=32, num_heads=4, window_size=4, attn_dropout_rate=rate
            )

    @pytest.mark.parametrize("rate", [-0.1, 1.5])
    def test_proj_dropout_out_of_range_raises(self, rate):
        with pytest.raises(ValueError, match="proj_dropout_rate"):
            BeitAttention(
                dim=32, num_heads=4, window_size=4, proj_dropout_rate=rate
            )

    def test_wrong_sequence_length_raises_at_build(self):
        layer = BeitAttention(dim=32, num_heads=4, window_size=(4, 4))
        # 16 patches + cls = 17; 16 is the classic off-by-one (forgot the cls token).
        with pytest.raises(ValueError, match="sequence length"):
            layer.build((2, 16, 32))


# ==============================================================================
# 2. Relative position index — SC-2, the critical oracle comparison
# ==============================================================================

class TestBeitAttentionRelativePositionIndex:
    """The index matrix must match the finding-derived oracle EXACTLY."""

    @pytest.mark.parametrize("wh,ww", [(4, 4), (3, 5), (5, 3), (2, 2), (1, 4)])
    def test_index_matches_reference_oracle(self, wh, ww):
        layer = BeitAttention(dim=16, num_heads=2, window_size=(wh, ww))
        _build(layer)
        actual = _layer_index_matrix(layer)
        expected = _oracle_relative_position_index(wh, ww)
        assert actual.shape == expected.shape
        np.testing.assert_array_equal(actual, expected)

    def test_index_matches_oracle_square_window(self):
        """Explicit square case named by SC-2."""
        layer = BeitAttention(dim=16, num_heads=2, window_size=(4, 4))
        _build(layer)
        np.testing.assert_array_equal(
            _layer_index_matrix(layer), _oracle_relative_position_index(4, 4)
        )

    def test_index_matches_oracle_non_square_window(self):
        """Non-square case: catches a transposed ``2*Ww-1`` vs ``2*Wh-1`` stride."""
        layer = BeitAttention(dim=16, num_heads=2, window_size=(3, 5))
        _build(layer)
        np.testing.assert_array_equal(
            _layer_index_matrix(layer), _oracle_relative_position_index(3, 5)
        )

    def test_transposed_grid_gives_a_different_index(self):
        """(3,5) and (5,3) must not produce the same matrix (guards the stride)."""
        a = BeitAttention(dim=16, num_heads=2, window_size=(3, 5))
        b = BeitAttention(dim=16, num_heads=2, window_size=(5, 3))
        _build(a)
        _build(b)
        assert not np.array_equal(_layer_index_matrix(a), _layer_index_matrix(b))

    @pytest.mark.parametrize("wh,ww", [(4, 4), (3, 5)])
    def test_cls_slots_are_the_last_three_rows(self, wh, ww):
        layer = BeitAttention(dim=16, num_heads=2, window_size=(wh, ww))
        _build(layer)
        idx = _layer_index_matrix(layer)
        m = layer.num_relative_distance
        n = wh * ww
        for j in range(1, n + 1):
            assert idx[0, j] == m - 3, f"cls->token slot wrong at column {j}"
        for i in range(1, n + 1):
            assert idx[i, 0] == m - 2, f"token->cls slot wrong at row {i}"
        assert idx[0, 0] == m - 1

    @pytest.mark.parametrize("wh,ww", [(4, 4), (3, 5), (2, 7)])
    def test_every_index_is_inside_the_table(self, wh, ww):
        layer = BeitAttention(dim=16, num_heads=2, window_size=(wh, ww))
        _build(layer)
        idx = _layer_index_matrix(layer)
        assert idx.min() >= 0
        assert idx.max() == layer.num_relative_distance - 1

    def test_patch_block_uses_only_coordinate_rows(self):
        """Patch-to-patch entries must never land in the 3 cls-reserved rows."""
        layer = BeitAttention(dim=16, num_heads=2, window_size=(3, 5))
        _build(layer)
        idx = _layer_index_matrix(layer)
        patch_block = idx[1:, 1:]
        assert patch_block.max() < layer.num_relative_distance - 3

    def test_table_shape_matches_the_formula(self):
        layer = BeitAttention(dim=18, num_heads=3, window_size=(3, 5))
        _build(layer)
        expected_rows = (2 * 3 - 1) * (2 * 5 - 1) + 3
        assert layer.relative_position_bias_table.shape == (expected_rows, 3)


# ==============================================================================
# 3. Forward pass
# ==============================================================================

class TestBeitAttentionForwardPass:
    """Output shapes and mode flags."""

    @pytest.mark.parametrize("num_heads", [1, 2, 4, 8])
    def test_output_shape_at_several_head_counts(self, num_heads):
        layer = BeitAttention(dim=32, num_heads=num_heads, window_size=(4, 4))
        x = _build(layer, batch=3)
        out = layer(x, training=False)
        assert out.shape == (3, 17, 32)

    def test_output_shape_non_square_window(self):
        layer = BeitAttention(dim=24, num_heads=4, window_size=(3, 5))
        x = _build(layer, batch=2)
        out = layer(x, training=False)
        assert out.shape == (2, 16, 24)

    def test_output_is_finite(self):
        layer = BeitAttention(dim=32, num_heads=4, window_size=(4, 4))
        x = _build(layer)
        out = keras.ops.convert_to_numpy(layer(x, training=False))
        assert np.all(np.isfinite(out))

    def test_training_true_and_false_both_run(self):
        layer = BeitAttention(
            dim=32,
            num_heads=4,
            window_size=(4, 4),
            attn_dropout_rate=0.1,
            proj_dropout_rate=0.1,
        )
        x = _build(layer)
        out_train = layer(x, training=True)
        out_eval = layer(x, training=False)
        assert out_train.shape == out_eval.shape == (2, 17, 32)

    def test_compute_output_shape(self):
        layer = BeitAttention(dim=32, num_heads=4, window_size=(4, 4))
        assert layer.compute_output_shape((None, 17, 32)) == (None, 17, 32)

    def test_builds_lazily_through_call(self):
        layer = BeitAttention(dim=32, num_heads=4, window_size=(4, 4))
        rng = np.random.default_rng(1)
        x = rng.normal(size=(2, 17, 32)).astype("float32")
        out = layer(x, training=False)
        assert layer.built
        assert out.shape == (2, 17, 32)


# ==============================================================================
# 4. Bias liveness — SC-3
# ==============================================================================

def _neutralize_qk(layer: BeitAttention) -> None:
    """Zero the q and k projection kernels so every pre-bias logit is identical.

    With ``q @ k^T == 0`` for every pair, the attention distribution is EXACTLY uniform
    unless something else reaches the logits. The relative-position bias is the only
    other term, so any position dependence in the output is attributable to it alone.

    Note this is why the test does NOT simply feed a constant input: a constant input
    makes every VALUE vector identical too, and a convex combination of identical
    vectors is that vector regardless of the weights — the bias would then be invisible
    in the output no matter how live it is. The values must vary; the logits must not.
    """
    layer.q_dense.kernel.assign(keras.ops.zeros_like(layer.q_dense.kernel))
    layer.k_dense.kernel.assign(keras.ops.zeros_like(layer.k_dense.kernel))
    if layer.q_dense.use_bias:
        layer.q_dense.bias.assign(keras.ops.zeros_like(layer.q_dense.bias))


class TestBeitAttentionBiasIsLive:
    """The relative-position bias table must actually reach the attention logits.

    DEAD-COMPONENT MUTATION (documented, verified during development, NOT left in the
    code): re-assigning ``relative_position_bias_table`` to all zeros makes the bias a
    no-op. With the q/k kernels neutralized every logit is then identical, attention is
    exactly uniform, every output row collapses to the same mean-pooled vector, and

        ``test_bias_makes_attention_output_position_dependent``

    goes RED at its final assertion, the one reading::

        assert spread > 1e-4, (
            "Attention output is uniform across positions ... "
        )

    That is the liveness assertion by name; the ``baseline_spread`` and finiteness
    setup assertions above it stay GREEN under the mutation, which is what makes the
    guard specific rather than merely noisy.
    """

    @staticmethod
    def _seeded_table(layer: BeitAttention, seed: int = 3) -> np.ndarray:
        rng = np.random.default_rng(seed)
        values = rng.normal(
            scale=2.0, size=tuple(layer.relative_position_bias_table.shape)
        ).astype("float32")
        layer.relative_position_bias_table.assign(values)
        return values

    @staticmethod
    def _varied_input(layer: BeitAttention, seed: int = 21) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.normal(
            size=(1, layer.num_tokens, layer.dim)
        ).astype("float32")

    def test_bias_makes_attention_output_position_dependent(self):
        layer = BeitAttention(
            dim=32, num_heads=2, window_size=(3, 5), use_proj_bias=False
        )
        n_tokens = layer.num_tokens
        layer.build((1, n_tokens, layer.dim))
        _neutralize_qk(layer)

        x = self._varied_input(layer)

        zero_out = keras.ops.convert_to_numpy(layer(x, training=False))[0]
        # Setup assertion (stays GREEN under the mutation): with the zero-initialized
        # table the attention really is uniform, so every output row is the same.
        baseline_spread = float(np.abs(zero_out - zero_out[0:1]).max())
        assert baseline_spread == pytest.approx(0.0, abs=1e-5)

        self._seeded_table(layer)
        biased_out = keras.ops.convert_to_numpy(layer(x, training=False))[0]

        # Setup assertion (also GREEN under the mutation): outputs stay finite.
        assert np.all(np.isfinite(biased_out))

        spread = float(np.abs(biased_out - biased_out[0:1]).max())
        # LIVENESS ASSERTION — this is the one that goes RED when the table is zeroed.
        assert spread > 1e-4, (
            "Attention output is uniform across positions under a non-uniform "
            "relative-position bias table: the table is allocated and serialized but "
            "never reaches the attention logits."
        )

    def test_two_token_pairs_with_different_displacement_get_different_weights(self):
        """SC-3's other half, read off the output rather than the hidden logits."""
        layer = BeitAttention(
            dim=32, num_heads=1, window_size=(3, 5), use_proj_bias=False
        )
        layer.build((1, layer.num_tokens, layer.dim))
        _neutralize_qk(layer)
        self._seeded_table(layer)
        x = self._varied_input(layer, seed=22)
        out = keras.ops.convert_to_numpy(layer(x, training=False))[0]
        # Patch rows 1 and 2 are horizontal neighbours; row 1 and the cls row 0 have a
        # different relation entirely. All three must differ once the bias is live.
        assert not np.allclose(out[0], out[1], atol=1e-5)
        assert not np.allclose(out[1], out[2], atol=1e-5)

    def test_disabling_the_bias_changes_the_output(self):
        """An otherwise-identical layer with the bias OFF must differ."""
        kwargs = dict(
            dim=32,
            num_heads=2,
            window_size=(3, 5),
            use_proj_bias=False,
        )
        with_bias = BeitAttention(use_relative_position_bias=True, **kwargs)
        without_bias = BeitAttention(use_relative_position_bias=False, **kwargs)

        n_tokens = with_bias.num_tokens
        with_bias.build((1, n_tokens, 32))
        without_bias.build((1, n_tokens, 32))

        # Give both layers identical projection weights so the ONLY difference is the
        # bias table, then neutralize q/k in both so uniform attention is the shared
        # baseline.
        for src, dst in (
                (with_bias.q_dense, without_bias.q_dense),
                (with_bias.k_dense, without_bias.k_dense),
                (with_bias.v_dense, without_bias.v_dense),
                (with_bias.proj, without_bias.proj),
        ):
            dst.set_weights(src.get_weights())
        _neutralize_qk(with_bias)
        _neutralize_qk(without_bias)

        self._seeded_table(with_bias)
        x = self._varied_input(with_bias, seed=23)

        a = keras.ops.convert_to_numpy(with_bias(x, training=False))
        b = keras.ops.convert_to_numpy(without_bias(x, training=False))
        assert not np.allclose(a, b, atol=1e-5), (
            "use_relative_position_bias=True and False produced the same output "
            "under a non-uniform bias table"
        )

    def test_no_bias_table_weight_when_disabled(self):
        layer = BeitAttention(
            dim=32, num_heads=2, window_size=(4, 4),
            use_relative_position_bias=False,
        )
        _build(layer)
        assert layer.relative_position_bias_table is None
        assert not any(
            "relative_position_bias_table" in w.name for w in layer.weights
        )

    def test_bias_is_shared_across_the_batch(self):
        """Same tokens in two batch rows -> same output rows (bias has no batch axis)."""
        layer = BeitAttention(dim=32, num_heads=2, window_size=(3, 5))
        n_tokens = layer.num_tokens
        layer.build((2, n_tokens, 32))
        self._seeded_table(layer)
        rng = np.random.default_rng(5)
        row = rng.normal(size=(1, n_tokens, 32)).astype("float32")
        x = np.concatenate([row, row], axis=0)
        out = keras.ops.convert_to_numpy(layer(x, training=False))
        np.testing.assert_allclose(out[0], out[1], atol=1e-6, rtol=0)


# ==============================================================================
# 5. No-k-bias structural assertions — SC-4
# ==============================================================================

def _bias_param_count(layer: BeitAttention) -> int:
    """Total number of scalar BIAS parameters across the layer's Dense sub-layers."""
    total = 0
    for sub in (layer.q_dense, layer.k_dense, layer.v_dense, layer.proj):
        if sub.use_bias:
            total += int(np.prod(sub.bias.shape))
    return total


class TestBeitAttentionNoKBias:
    """K must have NO bias weight; the exact bias-parameter count is derived."""

    def test_k_dense_use_bias_is_false(self):
        layer = BeitAttention(dim=32, num_heads=4, window_size=(4, 4))
        assert layer.k_dense.use_bias is False

    def test_k_dense_use_bias_is_false_even_when_qv_bias_is_true(self):
        layer = BeitAttention(
            dim=32, num_heads=4, window_size=(4, 4), qv_bias=True
        )
        _build(layer)
        assert layer.k_dense.use_bias is False
        assert layer.q_dense.use_bias is True
        assert layer.v_dense.use_bias is True

    def test_k_dense_has_exactly_one_weight(self):
        layer = BeitAttention(dim=32, num_heads=4, window_size=(4, 4))
        _build(layer)
        assert len([w for w in layer.k_dense.weights]) == 1
        assert layer.k_dense.weights[0].shape == (32, 32)
        assert layer.k_dense.bias is None

    def test_exact_bias_parameter_count_qv_and_proj(self):
        dim = 32
        layer = BeitAttention(
            dim=dim, num_heads=4, window_size=(4, 4),
            qv_bias=True, use_proj_bias=True,
        )
        _build(layer)
        # q + v + proj, and NOTHING for k.
        expected = 3 * dim
        assert _bias_param_count(layer) == expected

    def test_exact_bias_parameter_count_without_qv_bias(self):
        dim = 32
        layer = BeitAttention(
            dim=dim, num_heads=4, window_size=(4, 4),
            qv_bias=False, use_proj_bias=True,
        )
        _build(layer)
        # proj only.
        assert _bias_param_count(layer) == dim

    def test_exact_bias_parameter_count_without_any_bias(self):
        layer = BeitAttention(
            dim=32, num_heads=4, window_size=(4, 4),
            qv_bias=False, use_proj_bias=False,
        )
        _build(layer)
        assert _bias_param_count(layer) == 0

    def test_no_weight_is_named_k_bias(self):
        layer = BeitAttention(dim=32, num_heads=4, window_size=(4, 4))
        _build(layer)
        k_weight_paths = [w.path for w in layer.k_dense.weights]
        assert not any("bias" in p for p in k_weight_paths), k_weight_paths


# ==============================================================================
# 6. Gradient flow
# ==============================================================================

class TestBeitAttentionGradientFlow:
    """Every trainable variable, including the bias table, must receive gradient."""

    def test_all_trainable_variables_get_nonzero_gradients(self):
        layer = BeitAttention(dim=32, num_heads=4, window_size=(3, 5))
        x = _build(layer, batch=4)
        x_tf = tf.convert_to_tensor(x)

        with tf.GradientTape() as tape:
            out = layer(x_tf, training=True)
            loss = tf.reduce_mean(tf.square(out))
        grads = tape.gradient(loss, layer.trainable_variables)

        assert len(layer.trainable_variables) > 0
        for var, grad in zip(layer.trainable_variables, grads):
            assert grad is not None, f"no gradient for {var.path}"
            assert float(tf.reduce_max(tf.abs(grad))) > 0.0, (
                f"all-zero gradient for {var.path}"
            )

    def test_bias_table_is_trainable_and_receives_gradient(self):
        layer = BeitAttention(dim=32, num_heads=2, window_size=(4, 4))
        x = _build(layer, batch=2)
        table = layer.relative_position_bias_table
        assert table.trainable is True

        with tf.GradientTape() as tape:
            out = layer(tf.convert_to_tensor(x), training=True)
            loss = tf.reduce_mean(tf.square(out))
        grad = tape.gradient(loss, table)
        assert grad is not None
        assert float(tf.reduce_max(tf.abs(grad))) > 0.0


# ==============================================================================
# 7. Serialization — SC-5
# ==============================================================================

class TestBeitAttentionSerialization:
    """``get_config`` completeness and ``.keras`` VALUE-preserving round-trip."""

    def test_get_config_round_trip_reproduces_every_param(self):
        layer = BeitAttention(
            dim=48,
            num_heads=3,
            window_size=(3, 5),
            use_relative_position_bias=False,
            qv_bias=False,
            use_proj_bias=False,
            attn_dropout_rate=0.15,
            proj_dropout_rate=0.25,
            scale=0.3,
            kernel_initializer="he_normal",
            bias_initializer="ones",
            kernel_regularizer=keras.regularizers.L2(1e-4),
            bias_regularizer=keras.regularizers.L1(1e-5),
            name="beit_attn_cfg",
        )
        config = layer.get_config()
        restored = BeitAttention.from_config(config)

        assert restored.dim == 48
        assert restored.num_heads == 3
        assert tuple(restored.window_size) == (3, 5)
        assert restored.use_relative_position_bias is False
        assert restored.qv_bias is False
        assert restored.use_proj_bias is False
        assert restored.attn_dropout_rate == pytest.approx(0.15)
        assert restored.proj_dropout_rate == pytest.approx(0.25)
        assert restored.scale == pytest.approx(0.3)
        assert restored.name == "beit_attn_cfg"
        assert isinstance(
            restored.kernel_initializer, keras.initializers.HeNormal
        )
        assert isinstance(restored.bias_initializer, keras.initializers.Ones)
        assert isinstance(restored.kernel_regularizer, keras.regularizers.L2)
        assert isinstance(restored.bias_regularizer, keras.regularizers.L1)

    def test_get_config_declares_every_constructor_parameter(self):
        import inspect

        layer = BeitAttention(dim=32, num_heads=4, window_size=(4, 4))
        config = layer.get_config()
        params = set(inspect.signature(BeitAttention.__init__).parameters)
        params -= {"self", "kwargs"}
        missing = params - set(config)
        assert not missing, f"get_config() omits {sorted(missing)}"

    def test_keras_roundtrip_preserves_values_and_bias_table(self):
        layer = BeitAttention(dim=32, num_heads=2, window_size=(3, 5))
        n_tokens = layer.num_tokens

        inputs = keras.Input(shape=(n_tokens, 32))
        outputs = layer(inputs)
        model = keras.Model(inputs, outputs)

        rng = np.random.default_rng(7)
        table_values = rng.normal(
            size=tuple(layer.relative_position_bias_table.shape)
        ).astype("float32")
        layer.relative_position_bias_table.assign(table_values)

        x = rng.normal(size=(2, n_tokens, 32)).astype("float32")
        out_before = keras.ops.convert_to_numpy(model(x, training=False))

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.keras")
            model.save(path)
            restored = keras.models.load_model(path)
            out_after = keras.ops.convert_to_numpy(restored(x, training=False))

            restored_layer = next(
                sub for sub in restored.layers
                if isinstance(sub, BeitAttention)
            )
            restored_table = keras.ops.convert_to_numpy(
                restored_layer.relative_position_bias_table
            )

        np.testing.assert_allclose(out_before, out_after, atol=1e-6, rtol=0)
        # VALUES, not shapes/counts: a fresh-weight restore matches every count.
        np.testing.assert_array_equal(restored_table, table_values)

    def test_keras_roundtrip_preserves_the_index_buffer(self):
        layer = BeitAttention(dim=32, num_heads=2, window_size=(3, 5))
        n_tokens = layer.num_tokens
        inputs = keras.Input(shape=(n_tokens, 32))
        model = keras.Model(inputs, layer(inputs))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.keras")
            model.save(path)
            restored = keras.models.load_model(path)
        restored_layer = next(
            sub for sub in restored.layers if isinstance(sub, BeitAttention)
        )
        np.testing.assert_array_equal(
            _layer_index_matrix(restored_layer),
            _oracle_relative_position_index(3, 5),
        )

    def test_roundtrip_keeps_k_bias_absent(self):
        layer = BeitAttention(dim=32, num_heads=2, window_size=(4, 4))
        n_tokens = layer.num_tokens
        inputs = keras.Input(shape=(n_tokens, 32))
        model = keras.Model(inputs, layer(inputs))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.keras")
            model.save(path)
            restored = keras.models.load_model(path)
        restored_layer = next(
            sub for sub in restored.layers if isinstance(sub, BeitAttention)
        )
        assert restored_layer.k_dense.use_bias is False
        assert _bias_param_count(restored_layer) == 3 * 32


# ==============================================================================
# 8. Edge cases
# ==============================================================================

class TestBeitAttentionEdgeCases:
    """Optional paths: bias disabled, no qv bias, masks, dtypes."""

    def test_forward_without_relative_position_bias(self):
        layer = BeitAttention(
            dim=32, num_heads=4, window_size=(4, 4),
            use_relative_position_bias=False,
        )
        x = _build(layer)
        out = keras.ops.convert_to_numpy(layer(x, training=False))
        assert out.shape == (2, 17, 32)
        assert np.all(np.isfinite(out))
        assert layer._rel_pos_index is None

    def test_forward_without_qv_bias(self):
        layer = BeitAttention(
            dim=32, num_heads=4, window_size=(4, 4), qv_bias=False
        )
        x = _build(layer)
        out = layer(x, training=False)
        assert out.shape == (2, 17, 32)
        assert _bias_param_count(layer) == 32

    def test_rank2_key_mask_is_applied(self):
        layer = BeitAttention(dim=32, num_heads=2, window_size=(4, 4))
        x = _build(layer, batch=1)
        n_tokens = layer.num_tokens

        keep_all = np.ones((1, n_tokens), dtype="float32")
        keep_some = keep_all.copy()
        keep_some[0, -3:] = 0.0

        out_all = keras.ops.convert_to_numpy(
            layer(x, attention_mask=keep_all, training=False)
        )
        out_some = keras.ops.convert_to_numpy(
            layer(x, attention_mask=keep_some, training=False)
        )
        out_none = keras.ops.convert_to_numpy(layer(x, training=False))

        np.testing.assert_allclose(out_all, out_none, atol=1e-5, rtol=0)
        assert not np.allclose(out_all, out_some, atol=1e-5)
        assert np.all(np.isfinite(out_some))

    def test_rank3_pairwise_mask_is_accepted(self):
        layer = BeitAttention(dim=32, num_heads=2, window_size=(4, 4))
        x = _build(layer, batch=1)
        n_tokens = layer.num_tokens
        keep = np.tril(np.ones((n_tokens, n_tokens), dtype="float32"))[None]
        out = keras.ops.convert_to_numpy(
            layer(x, attention_mask=keep, training=False)
        )
        assert out.shape == (1, n_tokens, 32)
        assert np.all(np.isfinite(out))

    def test_bad_rank_mask_raises(self):
        layer = BeitAttention(dim=32, num_heads=2, window_size=(4, 4))
        x = _build(layer, batch=1)
        bad = np.ones((1,), dtype="float32")
        with pytest.raises(ValueError, match="attention_mask"):
            layer(x, attention_mask=bad, training=False)

    def test_degenerate_single_patch_grid(self):
        layer = BeitAttention(dim=16, num_heads=2, window_size=(1, 1))
        x = _build(layer, batch=2)
        assert layer.num_tokens == 2
        assert layer.num_relative_distance == 4
        out = layer(x, training=False)
        assert out.shape == (2, 2, 16)

    def test_float64_policy_forward_is_finite(self):
        previous = keras.mixed_precision.global_policy()
        try:
            keras.mixed_precision.set_global_policy("float64")
            layer = BeitAttention(dim=16, num_heads=2, window_size=(3, 3))
            layer.build((2, layer.num_tokens, 16))
            rng = np.random.default_rng(9)
            x = rng.normal(size=(2, layer.num_tokens, 16)).astype("float64")
            out = keras.ops.convert_to_numpy(layer(x, training=False))
            assert np.all(np.isfinite(out))
            assert out.dtype == np.float64
        finally:
            keras.mixed_precision.set_global_policy(previous)

    def test_uniform_attention_reduces_to_a_mean_pool_of_the_values(self):
        """Numeric check of the attention math itself, independent of the bias.

        With zeroed q/k kernels and a zero bias table every logit is 0, so the softmax
        is exactly uniform and ``attn @ v`` is the mean of the value vectors. The
        output must therefore equal ``proj(mean_t v_t)`` at every position.
        """
        layer = BeitAttention(
            dim=16, num_heads=1, window_size=(2, 2),
            qv_bias=False, use_proj_bias=False,
        )
        x = _build(layer, batch=1)
        _neutralize_qk(layer)

        out = keras.ops.convert_to_numpy(layer(x, training=False))
        v = keras.ops.convert_to_numpy(layer.v_dense(x))
        pooled = v.mean(axis=1, keepdims=True)
        expected = keras.ops.convert_to_numpy(layer.proj(pooled))

        assert out.shape == (1, 5, 16)
        # Tolerance note: this compares two DIFFERENT matmul orderings of the same
        # quantity (mean-then-project vs. attend-then-project), so on a TF32-enabled
        # GPU the two paths disagree at ~1e-3 relative even though the mathematical
        # identity is exact. The failure this guards against (attention that is not
        # uniform) is O(1), not O(1e-3).
        np.testing.assert_allclose(
            out, np.repeat(expected, 5, axis=1), atol=1e-3, rtol=1e-2
        )
