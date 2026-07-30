"""Tests for TransformerDecoderLayer (plan_2026-06-12_0bb1729b, F5)."""

import os
import logging
import tempfile
from typing import Any, Dict, Optional

import pytest
import numpy as np
import tensorflow as tf
import keras
from keras import ops, layers, models

from dl_techniques.layers.transformers.transformer_decoder import TransformerDecoderLayer
from dl_techniques.layers.ffn.diff_ffn import DifferentialFFN
from dl_techniques.layers.ffn.factory import STRICT_DROPPED_KEY_MARKER


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


def _strictness_break(fn) -> Optional[str]:
    """Run ``fn``; return the strict-factory dropped-key message, or ``None``.

    This replaced a ``logging.Handler`` on the ``dl`` logger. ``create_ffn_layer``
    used to WARN about a key it had to drop and now RAISES (D-023), so a warning
    recorder would capture nothing forever and every zero it reported would be
    vacuous -- the guard would still be green while measuring nothing at all.
    This classifier isolates the ONE failure mode strictness introduces from
    every other raise (missing required param, rank mismatch), which were
    already loud before the flip.

    :param fn: A zero-argument callable that constructs an FFN-owning layer.
    :return: The ``ValueError`` message if ``fn`` failed on a dropped key, else
        ``None`` -- both for success and for any other exception.
    :rtype: Optional[str]
    """
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - classification is the point
        msg = str(exc)
        return msg if STRICT_DROPPED_KEY_MARKER in msg else None
    return None


class TestDecoderDifferentialFFNActivation:
    """`ffn_type='differential'` must forward the site's activation as `branch_activation`.

    Regression guard for a live silent drop: ``_get_ffn_config`` used to place
    ``'differential'`` in the same generic branch as ``mlp/glu/geglu/residual/swin_mlp``
    and inject ``'activation'``, which ``DifferentialFFN`` does not accept -- so
    ``create_ffn_layer``'s parameter filter dropped it on EVERY construction and the FFN
    was built at DifferentialFFN's own default. ``TransformerLayer._get_ffn_config``
    already had a dedicated ``differential`` branch; this pins the decoder to it.

    STILL LOAD-BEARING after step 4 folded both dispatchers into the single
    ``build_transformer_ffn_config``, and the pre-filter does NOT subsume it:
    with the ``activation`` -> ``branch_activation`` rename removed, the
    pre-filter simply DISCARDS ``activation`` (``DifferentialFFN`` does not
    accept it), so zero dropped-key warnings are emitted, the whole 21-type grid
    stays green, and the FFN silently reverts to its own 'gelu'. Verified by
    removing the rename in place: the grid was green and these were the only
    assertions that fired.
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

    def test_no_dropped_key_for_this_construction(self):
        """Generalizes past this one parameter name: ZERO keys may be dropped.

        Was an assertion on the factory's dropped-key WARNING; the factory
        raises now (D-023), so the assertion is that construction SUCCEEDS --
        a dropped key can no longer be silent, it is a hard failure.
        """
        broke = _strictness_break(self._build)
        assert broke is None, (
            "TransformerDecoderLayer(ffn_type='differential') hands "
            f"create_ffn_layer a key DifferentialFFN does not accept: {broke}"
        )

    def test_dropped_key_harness_bites(self):
        """The classifier must be proven to SEE a real drop, not merely be quiet."""
        broke = _strictness_break(lambda: TransformerDecoderLayer(
            hidden_size=self.HIDDEN, num_heads=self.HEADS,
            intermediate_size=self.INTER,
            ffn_type='differential', ffn_args={'nosuchparam': 1},
        ))
        assert broke is not None and 'nosuchparam' in broke, (
            f"the classifier returned {broke!r}; it cannot detect a dropped key "
            f"and so its None in the sibling test proves nothing"
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


# ---------------------------------------------------------------------
# Step-4 guards: the shared FFN_REGISTRY pre-filter, decoder side.
#
# The decoder's own hand-maintained copy of the per-type injection table is
# gone -- `_get_ffn_config` now delegates to `build_transformer_ffn_config`,
# the same function `TransformerLayer` uses. These are the decoder halves of
# the encoder guards in `test_transformer.py`; the cross-dispatcher parity test
# lives THERE (one home) and is not restated here.
#
# The recorder attaches to the logger named 'dl' (`utils/logger.py` uses
# `logging.getLogger("dl")`, NOT "dl_techniques"); `_DroppedKeyRecorder` above
# is reused rather than redefined.
# ---------------------------------------------------------------------

from dl_techniques.layers.ffn.factory import FFN_REGISTRY

_GRID_HIDDEN = 32
_GRID_HEADS = 4
_GRID_INTER = 64
_ALL_FFN_TYPES = sorted(FFN_REGISTRY)

#: Types that RAISED on this dispatcher while `TransformerLayer` handled them,
#: because the decoder's table simply had no branch for them. Closing these was
#: the point of deleting that table (state.md step-3 divergence inventory).
_FORMER_DECODER_COVERAGE_GAPS = (
    'lowrank', 'monarch', 'squared_relu', 'reglu', 'bilinear',
)


def _required_ffn_args(ffn_type: str) -> Dict[str, Any]:
    sized = {
        'hidden_dim': _GRID_INTER,
        'output_dim': _GRID_HIDDEN, 'units': _GRID_HIDDEN,
        'features': _GRID_HIDDEN, 'filters': _GRID_HIDDEN,
    }
    return {
        p: sized.get(p, 8)
        for p in FFN_REGISTRY[ffn_type]['required_params']
    }


def _build_decoder(ffn_type: str, ffn_args: Dict[str, Any]):
    layer = TransformerDecoderLayer(
        hidden_size=_GRID_HIDDEN, num_heads=_GRID_HEADS,
        intermediate_size=_GRID_INTER, ffn_type=ffn_type,
        activation='relu', ffn_args=ffn_args,
    )
    layer(
        np.zeros((2, 5, _GRID_HIDDEN), dtype='float32'),
        encoder_output=np.zeros((2, 6, _GRID_HIDDEN), dtype='float32'),
    )
    return layer


class TestFFNTypeGridDecoder:
    """0 of 21 types may lose a caller key, at either grid condition."""

    def test_ffn_type_grid_harness_bites(self) -> None:
        """RED-proof the instrument BEFORE trusting any zero it reports."""
        broke = _strictness_break(
            lambda: _build_decoder('mlp', {'hiden_dim_typo': 3})
        )
        assert broke is not None and 'hiden_dim_typo' in broke, (
            f"_strictness_break returned {broke!r} for a deliberately "
            f"misspelled ffn_args key; it is blind."
        )

    def test_ffn_type_grid_harness_does_not_always_fire(self) -> None:
        """CONTROL: it must report None for a clean build AND for a raise of a
        different kind, or the grid below is 42 guaranteed failures / 42
        meaningless passes."""
        assert _strictness_break(lambda: _build_decoder('mlp', {})) is None
        assert _strictness_break(lambda: _build_decoder('counting', {})) is None

    def test_ffn_type_grid_covers_every_registry_type(self) -> None:
        assert len(_ALL_FFN_TYPES) == 21

    @pytest.mark.parametrize('ffn_type', _ALL_FFN_TYPES)
    @pytest.mark.parametrize(
        'condition', ['site-default-only', 'caller-supplies-required'])
    def test_ffn_type_grid_is_not_broken_by_strictness(
            self, ffn_type, condition) -> None:
        args = (
            {} if condition == 'site-default-only'
            else _required_ffn_args(ffn_type)
        )
        broke = _strictness_break(lambda: _build_decoder(ffn_type, args))
        assert broke is None, (
            f"TransformerDecoderLayer(ffn_type={ffn_type!r}) [{condition}] "
            f"fails construction on a key that type does not accept: {broke}"
        )

    @pytest.mark.parametrize('ffn_type', _FORMER_DECODER_COVERAGE_GAPS)
    def test_former_decoder_coverage_gap_now_builds_at_site_defaults(
        self, ffn_type
    ) -> None:
        """These five raised "Required parameters missing ['hidden_dim', ...]".

        The decoder's deleted table injected the dims for only 6 types; the
        encoder's injected them for 11. Sharing one policy function closed the
        difference by construction rather than by hand-adding five branches.
        """
        layer = _build_decoder(ffn_type, {})
        assert layer.ffn_layer is not None


class TestFFNArgsSurviveThePreFilter:
    """PRE-MORTEM #3, decoder side: `ffn_args` must never be pre-filtered."""

    def test_valid_caller_key_reaches_the_constructed_ffn(self) -> None:
        layer = TransformerDecoderLayer(
            hidden_size=_GRID_HIDDEN, num_heads=_GRID_HEADS,
            intermediate_size=_GRID_INTER, ffn_type='mlp',
            ffn_args={'use_bias': False},
        )
        assert layer.ffn_layer.use_bias is False, (
            "ffn_args={'use_bias': False} did not reach MLPBlock -- the "
            "pre-filter ate a CALLER key (assemble_ffn_config, D-017)."
        )

    def test_caller_key_overrides_the_wrapper_default(self) -> None:
        layer = TransformerDecoderLayer(
            hidden_size=_GRID_HIDDEN, num_heads=_GRID_HEADS,
            intermediate_size=_GRID_INTER, ffn_type='mlp',
            activation='relu', ffn_args={'activation': 'tanh'},
        )
        assert layer.ffn_layer.activation_name == 'tanh'

    def test_invalid_caller_key_stays_visible_to_the_factory(self) -> None:
        """The strict raise (D-023) needs this key intact to fire at all.

        Was an assertion on the dropped-key warning; now asserts the raise, and
        that the raise NAMES the key. If the pre-filter ever eats `ffn_args`,
        construction succeeds silently and this goes red.
        """
        with pytest.raises(ValueError, match='branch_activation'):
            TransformerDecoderLayer(
                hidden_size=_GRID_HIDDEN, num_heads=_GRID_HEADS,
                intermediate_size=_GRID_INTER, ffn_type='mlp',
                ffn_args={'branch_activation': 'relu'},
            )
