"""Tests for TransformerDecoderLayer (plan_2026-06-12_0bb1729b, F5)."""

import os
import tempfile
from typing import Any, Dict

import pytest
import numpy as np
import tensorflow as tf
import keras
from keras import ops, layers, models

from dl_techniques.layers.transformers.transformer_decoder import TransformerDecoderLayer


class TestTransformerDecoderLayer:
    """Init / forward / gradient / serialization / training / norm / causal."""

    HIDDEN = 64
    HEADS = 4
    INTER = 128
    DEC_SEQ = 8
    ENC_SEQ = 20
    BATCH = 2

    # --- Fixtures ---
    @pytest.fixture
    def config(self) -> Dict[str, Any]:
        return {'hidden_size': self.HIDDEN, 'num_heads': self.HEADS, 'intermediate_size': self.INTER}

    @pytest.fixture
    def decoder_input(self) -> tf.Tensor:
        return tf.random.normal((self.BATCH, self.DEC_SEQ, self.HIDDEN))

    @pytest.fixture
    def encoder_output(self) -> tf.Tensor:
        return tf.random.normal((self.BATCH, self.ENC_SEQ, self.HIDDEN))

    # --- Initialization ---
    def test_initialization_defaults(self, config):
        layer = TransformerDecoderLayer(**config)
        assert layer.hidden_size == self.HIDDEN
        assert layer.self_attention_type == 'multi_head'
        assert layer.cross_attention_type == 'multi_head_cross'
        assert layer.use_causal_mask is True
        assert not layer.built

    @pytest.mark.parametrize("bad_kwargs, match", [
        ({'hidden_size': 0}, "hidden_size must be positive"),
        ({'num_heads': 0}, "num_heads must be positive"),
        ({'hidden_size': 65}, "must be divisible"),         # 65 % 4 != 0
        ({'intermediate_size': 0}, "intermediate_size must be positive"),
        ({'normalization_position': 'middle'}, "must be 'pre' or 'post'"),
    ])
    def test_invalid_args_raise(self, bad_kwargs, match):
        base = {'hidden_size': self.HIDDEN, 'num_heads': self.HEADS, 'intermediate_size': self.INTER}
        base.update(bad_kwargs)
        with pytest.raises(ValueError, match=match):
            TransformerDecoderLayer(**base)

    # --- Forward pass ---
    @pytest.mark.parametrize("normalization_position", ['pre', 'post'])
    def test_forward_with_encoder_memory(self, config, decoder_input, encoder_output, normalization_position):
        """enc_seq != dec_seq -> output keeps decoder shape, no NaN."""
        layer = TransformerDecoderLayer(
            **config, normalization_position=normalization_position,
            dropout_rate=0.0, attention_dropout_rate=0.0,
        )
        out = layer(decoder_input, encoder_output, training=False)
        assert out.shape == (self.BATCH, self.DEC_SEQ, self.HIDDEN)
        assert not np.any(np.isnan(ops.convert_to_numpy(out)))
        assert layer.built

    def test_build_populates_sublayers(self, config, decoder_input, encoder_output):
        layer = TransformerDecoderLayer(**config)
        _ = layer(decoder_input, encoder_output, training=False)
        assert layer.self_attention.built
        assert layer.cross_attention.built
        assert layer.ffn_layer.built
        assert layer.self_attention_norm.built
        assert layer.cross_attention_norm.built
        assert layer.ffn_norm.built

    # --- Gradient flow ---
    def test_gradient_flow(self, config, decoder_input, encoder_output):
        layer = TransformerDecoderLayer(**config)
        d = tf.Variable(decoder_input)
        e = tf.Variable(encoder_output)
        with tf.GradientTape() as tape:
            out = layer(d, e, training=True)
            loss = tf.reduce_mean(tf.square(out))
        grads = tape.gradient(loss, layer.trainable_variables)
        assert len(layer.trainable_variables) > 0
        assert all(g is not None for g in grads), \
            [v.path for v, g in zip(layer.trainable_variables, grads) if g is None]

    # --- Serialization ---
    @pytest.mark.parametrize("normalization_position", ['pre', 'post'])
    def test_serialization_round_trip(self, config, decoder_input, encoder_output, normalization_position):
        dec_in = layers.Input(shape=(self.DEC_SEQ, self.HIDDEN))
        enc_in = layers.Input(shape=(self.ENC_SEQ, self.HIDDEN))
        out = TransformerDecoderLayer(
            **config, normalization_position=normalization_position,
            dropout_rate=0.0, attention_dropout_rate=0.0,
        )(dec_in, enc_in)
        model = models.Model([dec_in, enc_in], out)
        original = model([decoder_input, encoder_output], training=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "decoder.keras")
            model.save(filepath)
            loaded = models.load_model(filepath)
            reloaded = loaded([decoder_input, encoder_output], training=False)
        np.testing.assert_allclose(
            ops.convert_to_numpy(original),
            ops.convert_to_numpy(reloaded),
            rtol=1e-6, atol=1e-6,
        )

    def test_get_config_round_trip(self, config):
        layer = TransformerDecoderLayer(
            **config, normalization_position='pre', ffn_type='swiglu',
            use_causal_mask=False, cross_attention_args={'qk_norm_type': None},
        )
        cfg = layer.get_config()
        rebuilt = TransformerDecoderLayer.from_config(cfg)
        assert rebuilt.hidden_size == layer.hidden_size
        assert rebuilt.normalization_position == 'pre'
        assert rebuilt.ffn_type == 'swiglu'
        assert rebuilt.use_causal_mask is False

    # --- Training mode (dropout active vs inactive) ---
    def test_training_mode_dropout(self, config, decoder_input, encoder_output):
        layer = TransformerDecoderLayer(**config, dropout_rate=0.5, attention_dropout_rate=0.5)
        out_infer_a = layer(decoder_input, encoder_output, training=False)
        out_infer_b = layer(decoder_input, encoder_output, training=False)
        # Inference deterministic.
        np.testing.assert_allclose(
            ops.convert_to_numpy(out_infer_a), ops.convert_to_numpy(out_infer_b),
            rtol=1e-6, atol=1e-6,
        )
        # Training stochastic (dropout) -> generally differs from inference.
        out_train = layer(decoder_input, encoder_output, training=True)
        assert out_train.shape == out_infer_a.shape

    # --- Causal masking correctness ---
    def test_causal_self_attention(self, config, encoder_output):
        """A change to a future decoder token must not affect an earlier token's
        output (causal self-attention). Cross-attention to encoder memory is
        unmasked, so we hold encoder_output fixed and perturb only the future
        decoder position."""
        layer = TransformerDecoderLayer(
            **config, use_causal_mask=True, dropout_rate=0.0, attention_dropout_rate=0.0,
        )
        base = tf.random.normal((1, self.DEC_SEQ, self.HIDDEN))
        enc = encoder_output[:1]
        out_base = ops.convert_to_numpy(layer(base, enc, training=False))

        # Perturb the LAST decoder position only.
        perturbed = ops.convert_to_numpy(base).copy()
        perturbed[:, -1, :] += 10.0
        out_pert = ops.convert_to_numpy(layer(tf.constant(perturbed), enc, training=False))

        # Position 0 output must be unchanged (cannot attend to the future).
        np.testing.assert_allclose(out_base[:, 0, :], out_pert[:, 0, :], rtol=1e-5, atol=1e-5)
        # The perturbed (last) position output MUST change (sanity: mask isn't trivial).
        assert not np.allclose(out_base[:, -1, :], out_pert[:, -1, :], rtol=1e-3, atol=1e-3)

    def test_non_causal_self_attention(self, config, encoder_output):
        """With use_causal_mask=False, an early position CAN see a later one."""
        layer = TransformerDecoderLayer(
            **config, use_causal_mask=False, dropout_rate=0.0, attention_dropout_rate=0.0,
        )
        base = tf.random.normal((1, self.DEC_SEQ, self.HIDDEN))
        enc = encoder_output[:1]
        out_base = ops.convert_to_numpy(layer(base, enc, training=False))
        perturbed = ops.convert_to_numpy(base).copy()
        perturbed[:, -1, :] += 10.0
        out_pert = ops.convert_to_numpy(layer(tf.constant(perturbed), enc, training=False))
        # Position 0 SHOULD change now (bidirectional self-attention).
        assert not np.allclose(out_base[:, 0, :], out_pert[:, 0, :], rtol=1e-4, atol=1e-4)

    # --- Stacking ---
    def test_stacked_decoder_layers(self, config, decoder_input, encoder_output):
        l1 = TransformerDecoderLayer(**config, dropout_rate=0.0, attention_dropout_rate=0.0)
        l2 = TransformerDecoderLayer(**config, dropout_rate=0.0, attention_dropout_rate=0.0)
        x = l1(decoder_input, encoder_output, training=False)
        x = l2(x, encoder_output, training=False)
        assert x.shape == (self.BATCH, self.DEC_SEQ, self.HIDDEN)
        assert not np.any(np.isnan(ops.convert_to_numpy(x)))
