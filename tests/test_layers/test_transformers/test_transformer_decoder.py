"""Tests for TransformerDecoderLayer (plan_2026-06-12_0bb1729b, F5)."""

import os
import logging
import tempfile
from typing import Any, Dict

import pytest
import numpy as np
import tensorflow as tf
import keras
from keras import ops, layers, models

from dl_techniques.layers.transformers.transformer_decoder import TransformerDecoderLayer
from dl_techniques.layers.ffn.diff_ffn import DifferentialFFN


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


class _DroppedKeyRecorder(logging.Handler):
    """Captures ``create_ffn_layer(...): dropping`` WARNINGs off the ``dl`` logger.

    The repo's centralized logger is named ``"dl"`` (``dl_techniques/utils/logger.py``),
    NOT ``"dl_techniques"`` -- attaching to the latter records nothing. Reading stderr
    by eye is not an instrument; this is.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.dropped: list = []

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if 'dropping' in msg and 'unsupported parameter' in msg:
            self.dropped.append(msg)


class TestDecoderDifferentialFFNActivation:
    """`ffn_type='differential'` must forward the site's activation as `branch_activation`.

    Regression guard for a live silent drop: ``_get_ffn_config`` used to place
    ``'differential'`` in the same generic branch as ``mlp/glu/geglu/residual/swin_mlp``
    and inject ``'activation'``, which ``DifferentialFFN`` does not accept -- so
    ``create_ffn_layer``'s parameter filter dropped it on EVERY construction and the FFN
    was built at DifferentialFFN's own default. ``TransformerLayer._get_ffn_config``
    already had a dedicated ``differential`` branch; this pins the decoder to it.
    """

    HIDDEN = 32
    HEADS = 4
    INTER = 64

    # DifferentialFFN's own default is 'gelu' (diff_ffn.py __init__), and so is
    # TransformerDecoderLayer's default `activation`. The probe value must differ from
    # BOTH, or the assertion would pass against the pre-fix code by coincidence.
    NON_DEFAULT_ACTIVATION = 'relu'

    def _build(self):
        return TransformerDecoderLayer(
            hidden_size=self.HIDDEN, num_heads=self.HEADS, intermediate_size=self.INTER,
            ffn_type='differential', activation=self.NON_DEFAULT_ACTIVATION,
        )

    def test_differential_default_is_not_the_probe_value(self):
        """Anti-vacuity: the probe activation must not be DifferentialFFN's default."""
        default_ffn = DifferentialFFN(hidden_dim=self.INTER, output_dim=self.HIDDEN)
        assert keras.activations.serialize(default_ffn.branch_activation) == 'gelu'
        assert self.NON_DEFAULT_ACTIVATION != 'gelu'

    def test_site_activation_reaches_branch_activation(self):
        layer = self._build()
        ffn = layer.ffn_layer
        assert isinstance(ffn, DifferentialFFN)
        got = keras.activations.serialize(ffn.branch_activation)
        assert got == self.NON_DEFAULT_ACTIVATION, (
            f"TransformerDecoderLayer(ffn_type='differential', "
            f"activation='{self.NON_DEFAULT_ACTIVATION}') built a DifferentialFFN whose "
            f"branch_activation is '{got}'. The site's activation was dropped; "
            f"DifferentialFFN takes `branch_activation`, not `activation`."
        )

    def test_gate_activation_is_left_at_its_default(self):
        """The site's generic `activation` must NOT be forwarded to the gate.

        Mirrors ``TransformerLayer._get_ffn_config``, which forwards only
        ``branch_activation``: the sigmoid gate is DifferentialFFN's defining feature.
        """
        ffn = self._build().ffn_layer
        assert keras.activations.serialize(ffn.gate_activation) == 'sigmoid'

    def test_no_dropped_key_warnings_for_this_construction(self):
        """Generalizes past this one parameter name: ZERO keys may be silently dropped."""
        handler = _DroppedKeyRecorder()
        dl_logger = logging.getLogger('dl')
        dl_logger.addHandler(handler)
        try:
            self._build()
        finally:
            dl_logger.removeHandler(handler)
        assert handler.dropped == [], (
            "TransformerDecoderLayer(ffn_type='differential') silently dropped "
            f"caller/site key(s): {handler.dropped}"
        )

    def test_dropped_key_harness_bites(self):
        """The capture harness must be proven to see a real drop, not merely be silent."""
        handler = _DroppedKeyRecorder()
        dl_logger = logging.getLogger('dl')
        dl_logger.addHandler(handler)
        try:
            TransformerDecoderLayer(
                hidden_size=self.HIDDEN, num_heads=self.HEADS, intermediate_size=self.INTER,
                ffn_type='differential', ffn_args={'nosuchparam': 1},
            )
        finally:
            dl_logger.removeHandler(handler)
        assert any('nosuchparam' in m for m in handler.dropped), (
            f"harness saw {handler.dropped}; it cannot detect a dropped key and so its "
            f"silence in the sibling test proves nothing"
        )

    def test_differential_forward_and_round_trip(self):
        layer = self._build()
        dec = tf.random.normal((2, 6, self.HIDDEN))
        enc = tf.random.normal((2, 9, self.HIDDEN))
        out = layer(dec, enc, training=False)
        assert out.shape == (2, 6, self.HIDDEN)
        assert not np.any(np.isnan(ops.convert_to_numpy(out)))

        inp_d = layers.Input(shape=(6, self.HIDDEN))
        inp_e = layers.Input(shape=(9, self.HIDDEN))
        model = models.Model([inp_d, inp_e], layer(inp_d, inp_e))
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'm.keras')
            model.save(path)
            restored = models.load_model(path)
        rebuilt_ffn = restored.layers[-1].ffn_layer
        assert keras.activations.serialize(
            rebuilt_ffn.branch_activation) == self.NON_DEFAULT_ACTIVATION
