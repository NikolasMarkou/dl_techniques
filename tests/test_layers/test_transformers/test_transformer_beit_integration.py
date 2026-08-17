"""``TransformerLayer(attention_type='beit')`` -- the BEiT block, wired end to end.

Registering the ``'beit'`` attention type is only half of the integration. This module
covers the other half, in ``layers/transformers/transformer.py``:

* ``build_transformer_attention_required_params()`` -- the SINGLE per-type table
  (D-015) that must now answer for ``'beit'``, and must still answer identically for
  every type it answered for before;
* ``_get_attention_params()`` -- the branch that assembles the factory call. Its most
  dangerous property is that ``create_attention_layer`` FILTERS unknown kwargs away
  SILENTLY, so a branch that copies ``'window'``'s ``dropout_rate`` verbatim builds a
  perfectly healthy layer whose attention dropout is permanently 0.0;
* the composition itself -- that a BEiT block's normalization epsilons, layer-scale
  gamma (SIGNED, initialized to 0.1), stochastic depth and residual wiring are all
  reachable through the EXISTING ``TransformerLayer`` surface with no signature change.

Assertions here read the ACTUAL sub-layer attributes and weights, never the kwargs
dict that was passed in -- a kwarg that never arrived and a kwarg that arrived and was
honoured are indistinguishable from the caller's side.
"""

import os
import tempfile

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from dl_techniques.layers.attention.beit_attention import BeitAttention
from dl_techniques.layers.transformers.transformer import (
    TransformerLayer,
    build_transformer_attention_required_params,
)

# The block geometry used throughout: a 4x4 patch grid plus one cls token.
WH, WW = 4, 4
NUM_TOKENS = WH * WW + 1
HIDDEN = 32
HEADS = 4
EPS = 1e-12


def _beit_block(**overrides) -> TransformerLayer:
    """A BEiT-shaped block: pre-norm, 1e-12 epsilons, signed layer scale @ 0.1."""
    config = dict(
        hidden_size=HIDDEN,
        num_heads=HEADS,
        intermediate_size=4 * HIDDEN,
        attention_type='beit',
        window_size=(WH, WW),
        normalization_position='pre',
        use_layer_scale=True,
        layer_scale_init_value=0.1,
        attention_norm_args={'epsilon': EPS},
        ffn_norm_args={'epsilon': EPS},
        use_stochastic_depth=True,
        stochastic_depth_rate=0.05,
    )
    config.update(overrides)
    return TransformerLayer(**config)


def _sample(batch: int = 2, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(batch, NUM_TOKENS, HIDDEN)).astype('float32')


# ==============================================================================
# 1. The single-source-of-truth required-params table
# ==============================================================================

class TestBeitRequiredParamsTable:
    """``build_transformer_attention_required_params`` is the ONLY per-type table."""

    def test_beit_requires_window_size(self):
        params = build_transformer_attention_required_params(
            attention_type='beit', hidden_size=64, num_heads=4, window_size=(4, 4)
        )
        assert params == {'window_size': (4, 4)}

    def test_beit_accepts_a_scalar_grid(self):
        params = build_transformer_attention_required_params(
            attention_type='beit', hidden_size=64, num_heads=4, window_size=7
        )
        assert params == {'window_size': 7}

    def test_beit_falls_back_to_the_shared_default(self):
        """No dedicated param on the caller -> the encoder's documented default."""
        params = build_transformer_attention_required_params(
            attention_type='beit', hidden_size=64, num_heads=4
        )
        assert params == {'window_size': 8}

    @pytest.mark.parametrize(
        "attention_type,expected",
        [
            ('window', {'window_size': (4, 4)}),
            ('group_query', {'num_kv_heads': 4}),
            ('differential', {'head_dim': 16, 'lambda_init': 0.8}),
            ('multi_head_latent', {'kv_latent_dim': 16}),
            ('multi_head', {}),
            ('fnet', {}),
            ('anchor', {}),
            ('lighthouse', {}),
            ('multi_head_cross', {}),
            ('nonexistent_type', {}),
        ],
    )
    def test_the_other_types_are_unchanged(self, attention_type, expected):
        """Regression control: adding 'beit' must not perturb any other answer.

        Why this can fail if the implementation is wrong: the 'beit' case shares a
        branch with 'window' (``attention_type in ('window', 'beit')``). A mistyped
        membership test -- or a branch inserted above an earlier one -- would change
        what an unrelated type receives, and every affected model would keep building
        while silently constructing a differently-parameterized attention layer.
        """
        params = build_transformer_attention_required_params(
            attention_type=attention_type,
            hidden_size=64,
            num_heads=4,
            window_size=(4, 4),
        )
        assert params == expected

    def test_the_table_is_not_duplicated_into_the_decoder(self):
        """D-015: the decoder must READ this table, never carry a second copy.

        Why this can fail if the implementation is wrong: two hand-maintained copies
        of this table is precisely what F-07/D-018 were, and the observable symptom
        was a type being unconstructable on one block while working on the other.
        Adding 'beit' to only one copy would reproduce it, so this asserts the
        decoder's self-attention params carry the beit key too.
        """
        from dl_techniques.layers.transformers.transformer_decoder import (
            TransformerDecoderLayer,
        )

        decoder = TransformerDecoderLayer(
            hidden_size=HIDDEN,
            num_heads=HEADS,
            intermediate_size=4 * HIDDEN,
            self_attention_type='beit',
            attention_args={'window_size': (WH, WW)},
        )
        assert isinstance(decoder.self_attention, BeitAttention)
        assert decoder.self_attention.window_size == (WH, WW)


# ==============================================================================
# 2. Attention-parameter assembly (the silent-drop surface)
# ==============================================================================

class TestBeitAttentionParams:
    """What ``_get_attention_params('attn')`` hands to the attention factory."""

    def test_exact_param_dict(self):
        block = _beit_block(attention_dropout_rate=0.3)
        assert block._get_attention_params('attn') == {
            'dim': HIDDEN,
            'num_heads': HEADS,
            'window_size': (WH, WW),
            'attn_dropout_rate': 0.3,
            'name': 'attn',
        }

    def test_attention_dropout_rate_actually_reaches_the_layer(self):
        """The kwarg NAME matters: 'dropout_rate' is not a ``BeitAttention`` parameter.

        Why this can fail if the implementation is wrong: ``BeitAttention`` declares
        ``attn_dropout_rate``/``proj_dropout_rate``, not ``dropout_rate``. A branch
        copied from ``'window'`` would pass ``dropout_rate=0.3``; before 2026-08-17
        ``create_attention_layer`` filtered undeclared kwargs out SILENTLY, so the
        value vanished and the block's attention dropout sat at 0.0 forever with no
        error anywhere. That factory now RAISES on the undeclared name
        (plan-2026-08-17T183311-79c63e38/D-011), so the copied branch would fail
        loudly instead. This test still reads the rate off the BUILT sub-layer,
        because the raise cannot tell you the right name arrived and was WIRED.
        """
        block = _beit_block(attention_dropout_rate=0.3)
        assert block.attention.attn_dropout_rate == 0.3
        assert block.attention.attn_dropout.rate == 0.3
        # proj dropout is deliberately NOT doubled with the block's own dropout.
        assert block.attention.proj_dropout_rate == 0.0

    def test_attention_args_override_the_defaults(self):
        """A caller's ``attention_args`` still wins, as it does for every type."""
        block = TransformerLayer(
            hidden_size=HIDDEN,
            num_heads=HEADS,
            intermediate_size=4 * HIDDEN,
            attention_type='beit',
            attention_args={'window_size': (2, 3), 'qv_bias': False},
        )
        assert block.attention.window_size == (2, 3)
        assert block.attention.qv_bias is False
        assert block.attention.q_dense.use_bias is False
        assert block.attention.k_dense.use_bias is False

    def test_beit_is_not_maskless(self):
        """H-6 / D-016: 'beit' must stay OUT of the maskless set.

        Why this can fail if the implementation is wrong: adding it there would stop
        ``TransformerLayer.call`` from forwarding ``attention_mask`` at all -- a
        silently non-masking block, which is the exact regression D-016 records for
        ``'window'``. ``BeitAttention`` accepts and honours the mask, so the default
        masked branch is the correct one.
        """
        assert 'beit' not in TransformerLayer._MASKLESS_ATTENTION_TYPES

    def test_the_mask_is_forwarded_and_changes_the_output(self):
        """Not just "accepts the kwarg": the mask must MOVE the block's output."""
        block = _beit_block(use_stochastic_depth=False)
        x = _sample(batch=1)
        keep_all = np.ones((1, NUM_TOKENS), dtype='float32')
        keep_some = keep_all.copy()
        keep_some[0, -4:] = 0.0

        out_all = ops.convert_to_numpy(
            block(x, attention_mask=keep_all, training=False)
        )
        out_some = ops.convert_to_numpy(
            block(x, attention_mask=keep_some, training=False)
        )
        assert np.all(np.isfinite(out_some))
        assert not np.allclose(out_all, out_some, atol=1e-6)


# ==============================================================================
# 3. Composition: the sub-layers the block actually wires
# ==============================================================================

class TestBeitBlockComposition:
    """Everything asserted here is read off the BUILT layer, not off the config."""

    def test_attention_sublayer_is_a_beit_attention(self):
        block = _beit_block()
        assert isinstance(block.attention, BeitAttention)
        assert block.attention.dim == HIDDEN
        assert block.attention.num_heads == HEADS
        assert block.attention.window_size == (WH, WW)
        assert block.attention.num_tokens == NUM_TOKENS

    def test_both_normalization_epsilons_are_1e_12(self):
        """Read ``epsilon`` off the built sub-layers, not the kwargs dict."""
        block = _beit_block()
        block.build((None, NUM_TOKENS, HIDDEN))
        assert block.attention_norm.epsilon == EPS
        assert block.output_norm.epsilon == EPS

    def test_the_epsilon_assertion_is_falsifiable(self):
        """Control: a block that does NOT ask for 1e-12 must report something else.

        Why this can fail if the implementation is wrong: if ``attention_norm_args``
        were being ignored, the assertion above would still need a value to compare
        against -- and Keras' own default (1e-3 for this factory's LayerNorm) is not
        1e-12, so the two blocks must disagree. If they agree, the epsilon above was
        never applied by the caller and the test proves nothing.
        """
        default_block = TransformerLayer(
            hidden_size=HIDDEN,
            num_heads=HEADS,
            intermediate_size=4 * HIDDEN,
            attention_type='beit',
            window_size=(WH, WW),
        )
        default_block.build((None, NUM_TOKENS, HIDDEN))
        assert default_block.attention_norm.epsilon != EPS

    def test_layer_scale_gamma_is_signed_and_initialized_to_0_1(self):
        """The gamma weight must carry NO non-negativity constraint.

        Why this can fail if the implementation is wrong: ``LearnableMultiplier``
        DEFAULTS to a ``non_neg`` constraint; BEiT's layer scale is signed, and a
        clamped gamma cannot represent a negative residual scaling. ``TransformerLayer``
        passes ``constraint=None`` explicitly, and this reads the constraint off the
        real ``keras.Variable`` rather than trusting that.
        """
        block = _beit_block()
        block.build((None, NUM_TOKENS, HIDDEN))

        for scale in (block.attention_layer_scale, block.ffn_layer_scale):
            assert scale.constraint is None
            assert scale.gamma.constraint is None
            gamma = ops.convert_to_numpy(scale.gamma)
            assert gamma.shape == (HIDDEN,)
            np.testing.assert_allclose(gamma, 0.1, atol=1e-7, rtol=0)
            # Signed in fact, not merely unconstrained in config: a negative
            # assignment must survive the constraint machinery.
            scale.gamma.assign(ops.full(gamma.shape, -0.25))
            assert float(np.min(ops.convert_to_numpy(scale.gamma))) < 0.0

    def test_stochastic_depth_is_wired_at_the_requested_rate(self):
        block = _beit_block()
        assert block.attention_stochastic_depth is not None
        assert block.ffn_stochastic_depth is not None
        assert block.attention_stochastic_depth.drop_path_rate == 0.05

    def test_k_projection_has_no_bias_through_the_whole_stack(self):
        """The BEiT invariant survives factory -> TransformerLayer composition."""
        block = _beit_block()
        block.build((None, NUM_TOKENS, HIDDEN))
        assert block.attention.k_dense.use_bias is False
        assert block.attention.k_dense.bias is None
        assert block.attention.q_dense.bias is not None
        assert block.attention.v_dense.bias is not None


# ==============================================================================
# 4. Forward pass, gradients, serialization
# ==============================================================================

class TestBeitBlockForwardAndSerialization:

    def test_forward_shape(self):
        block = _beit_block()
        out = block(_sample(), training=False)
        assert out.shape == (2, NUM_TOKENS, HIDDEN)
        assert np.all(np.isfinite(ops.convert_to_numpy(out)))

    def test_wrong_sequence_length_raises_loudly(self):
        """Wh*Ww + 1 is a hard contract; a mismatch must not gather a wrong bias."""
        block = _beit_block()
        with pytest.raises(ValueError, match="sequence length"):
            block.build((None, NUM_TOKENS + 3, HIDDEN))

    def test_gradients_reach_the_relative_position_bias_table(self):
        block = _beit_block(use_stochastic_depth=False)
        x = tf.constant(_sample(batch=2, seed=3))
        with tf.GradientTape() as tape:
            loss = ops.mean(ops.square(block(x, training=True)))
        table = block.attention.relative_position_bias_table
        grads = tape.gradient(loss, block.trainable_variables)
        by_id = {id(v): g for v, g in zip(block.trainable_variables, grads)}

        assert all(g is not None for g in grads), "a trainable variable got no gradient"
        table_grad = by_id[id(table)]
        assert table_grad is not None
        assert float(np.max(np.abs(ops.convert_to_numpy(table_grad)))) > 0.0

    def test_keras_round_trip_preserves_values(self):
        """A full model containing the block must reload to the SAME numbers.

        ``training=False`` is passed explicitly: ``training=None`` is not inference
        for ``StochasticDepth``, so an implicit call would compare two stochastic
        draws and could pass or fail for reasons unrelated to serialization.
        """
        inputs = keras.Input(shape=(NUM_TOKENS, HIDDEN))
        block = _beit_block()
        model = keras.Model(inputs, block(inputs))

        x = _sample(batch=3, seed=11)
        before = ops.convert_to_numpy(model(x, training=False))

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'beit_block.keras')
            model.save(path)
            restored = keras.models.load_model(path)
            after = ops.convert_to_numpy(restored(x, training=False))

        np.testing.assert_allclose(before, after, atol=1e-6, rtol=0)

        restored_block = next(
            layer for layer in restored.layers if isinstance(layer, TransformerLayer)
        )
        assert isinstance(restored_block.attention, BeitAttention)
        assert restored_block.attention.window_size == (WH, WW)
        assert restored_block.attention_norm.epsilon == EPS
        np.testing.assert_array_equal(
            ops.convert_to_numpy(block.attention.relative_position_bias_table),
            ops.convert_to_numpy(
                restored_block.attention.relative_position_bias_table
            ),
        )

    def test_get_config_round_trips_the_block(self):
        block = _beit_block()
        rebuilt = TransformerLayer.from_config(block.get_config())
        assert rebuilt.attention_type == 'beit'
        assert tuple(rebuilt.window_size) == (WH, WW)
        assert isinstance(rebuilt.attention, BeitAttention)
        assert rebuilt.layer_scale_init_value == 0.1

    def test_non_square_patch_grid(self):
        """Wh != Ww must reach the layer intact through the factory door."""
        block = TransformerLayer(
            hidden_size=HIDDEN,
            num_heads=HEADS,
            intermediate_size=4 * HIDDEN,
            attention_type='beit',
            window_size=(3, 5),
        )
        assert block.attention.window_size == (3, 5)
        assert block.attention.num_relative_distance == (2 * 3 - 1) * (2 * 5 - 1) + 3
        x = np.random.default_rng(5).normal(size=(2, 16, HIDDEN)).astype('float32')
        out = block(x, training=False)
        assert out.shape == (2, 16, HIDDEN)
        assert np.all(np.isfinite(ops.convert_to_numpy(out)))
