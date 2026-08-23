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


# ---------------------------------------------------------------------
# F-03: maskless self-attention types
# ---------------------------------------------------------------------

from dl_techniques.layers.transformers.transformer import TransformerLayer


# R-038 closure -- plan-2026-08-22T035419-a11304c8 / D-251.
# This module drives `OrthogonalHypersphereInitializer` (the DEFAULT
# `kernel_initializer` of `OrthoBlock` / `OrthoGLUFFN` / `NeuroGrid`) at shapes
# where more orthogonal vectors are requested than the space has dimensions --
# which is the ORDINARY case for a dimension-reducing projection. The advisory
# at `initializers/hypersphere_orthogonal_initializer.py:190` is ours, it is
# correct, and it reports a fallback that is the designed behaviour. Suppressed
# HERE rather than in `pyproject.toml` so that a module which starts tripping
# the fallback UNEXPECTEDLY still fails under `error::UserWarning`. The advisory
# is pinned live (message text included) by
# `tests/test_the_deliberate_advisories_still_fire.py`.
pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:Orthogonality constraint violation:UserWarning"),
]

# Per-type minimum decoder sequence length. `lighthouse` defaults to
# num_levels=3 / pooling_factor=4, so it requires seq_len % 16 == 0 and
# raises a ValueError about divisibility long before any mask question is
# reached -- at T=8 this test class would have measured that constraint
# instead of the defect.
_MASKLESS_SEQ_LEN = {'anchor': 8, 'lighthouse': 32, 'fnet': 8}


class TestMasklessSelfAttentionTypes:
    """F-03: `{'anchor', 'lighthouse', 'fnet'}` must construct AND run.

    `TransformerLayer` already owns the dispatch (`_MASKLESS_ATTENTION_TYPES`)
    and skips `attention_mask=` for those types. The decoder passed it
    unconditionally at both of its self-attention call sites.

    RED captures at HEAD (`03980608`), which differ from the plan's prediction
    for two of the three types:

    * `anchor`  -- `TypeError: got an unexpected keyword argument
      'attention_mask'`, as predicted. Raised by Keras' `inspect`-based
      signature check, not by `call` itself.
    * `lighthouse` -- the SAME `TypeError`, but only at a legal sequence length.
      At the module's usual `DEC_SEQ = 8` it dies earlier and unrelatedly on
      `seq_len N=8 must be divisible by pooling_factor ** (num_levels - 1) = 16`.
    * `fnet` -- NOT a construction-time error, contrary to the plan. It
      constructed fine at the time of measurement: the attention factory then
      silently dropped the `dim` / `num_heads` the decoder injects rather than
      rejecting them. (That factory is strict as of 2026-08-17,
      plan-2026-08-17T183311-79c63e38/D-011, so the same injection would now
      raise -- but the decoder stopped injecting them for `fnet` before that,
      see `test_fnet_self_attention_params_match_TransformerLayer`.) Its
      `call()` DOES accept `attention_mask`. It dies at run time inside
      `FNetFourierTransform.call` with
      `InvalidArgumentError: Incompatible shapes: [2,8,64] vs. [1,8,8,1]` --
      it interprets the mask as a rank-2 padding mask and the decoder hands it
      the rank-3 causal mask. The prescribed fix (do not pass a mask at all)
      still fixes it, for a different reason than the plan gave.
    """

    HIDDEN, HEADS, INTER, ENC_SEQ, BATCH = 64, 4, 128, 10, 2

    def _run(self, attention_type: str, normalization_position: str = 'post'):
        layer = TransformerDecoderLayer(
            hidden_size=self.HIDDEN, num_heads=self.HEADS,
            intermediate_size=self.INTER, self_attention_type=attention_type,
            normalization_position=normalization_position,
        )
        seq = _MASKLESS_SEQ_LEN[attention_type]
        x = keras.random.normal([self.BATCH, seq, self.HIDDEN])
        enc = keras.random.normal([self.BATCH, self.ENC_SEQ, self.HIDDEN])
        return layer, np.array(layer(x, enc, training=False)), seq

    # `normalization_position` is parametrized because `call()` has TWO
    # self-attention call sites, one per branch, and 'post' is the default:
    # a suite that only used the default would leave the 'pre' site unfixed
    # and unmeasured.
    @pytest.mark.parametrize("normalization_position", ['post', 'pre'])
    @pytest.mark.parametrize("attention_type", ['anchor', 'lighthouse', 'fnet'])
    def test_constructs_and_runs(self, attention_type, normalization_position):
        layer, out, seq = self._run(attention_type, normalization_position)
        assert out.shape == (self.BATCH, seq, self.HIDDEN)
        assert np.all(np.isfinite(out))

    @pytest.mark.parametrize("normalization_position", ['post', 'pre'])
    @pytest.mark.parametrize("attention_type", ['anchor', 'lighthouse', 'fnet'])
    def test_an_explicit_self_attention_mask_is_also_tolerated(
            self, attention_type, normalization_position):
        """A caller may pass a mask; a maskless type must ignore it, not crash."""
        layer = TransformerDecoderLayer(
            hidden_size=self.HIDDEN, num_heads=self.HEADS,
            intermediate_size=self.INTER, self_attention_type=attention_type,
            normalization_position=normalization_position,
        )
        seq = _MASKLESS_SEQ_LEN[attention_type]
        x = keras.random.normal([self.BATCH, seq, self.HIDDEN])
        enc = keras.random.normal([self.BATCH, self.ENC_SEQ, self.HIDDEN])
        mask = ops.ones((self.BATCH, seq, seq))
        out = np.array(layer(x, enc, self_attention_mask=mask, training=False))
        assert out.shape == (self.BATCH, seq, self.HIDDEN)

    @pytest.mark.parametrize("attention_type", ['anchor', 'lighthouse', 'fnet'])
    def test_serialization_round_trip_by_value(self, attention_type):
        """I3: a `.keras` round-trip must restore VALUES on the new path."""
        seq = _MASKLESS_SEQ_LEN[attention_type]
        dec_in = keras.Input(shape=(seq, self.HIDDEN))
        enc_in = keras.Input(shape=(self.ENC_SEQ, self.HIDDEN))
        out = TransformerDecoderLayer(
            hidden_size=self.HIDDEN, num_heads=self.HEADS,
            intermediate_size=self.INTER, self_attention_type=attention_type,
            dropout_rate=0.0, attention_dropout_rate=0.0,
        )(dec_in, enc_in)
        model = models.Model([dec_in, enc_in], out)
        x = keras.random.normal([self.BATCH, seq, self.HIDDEN])
        enc = keras.random.normal([self.BATCH, self.ENC_SEQ, self.HIDDEN])
        ref = np.array(model([x, enc], training=False))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, f"dec_{attention_type}.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
        got = np.array(loaded([x, enc], training=False))
        assert float(np.max(np.abs(ref))) > 1e-3, "round-trip compared all-zero values"
        np.testing.assert_allclose(ref, got, rtol=0, atol=0)

    # --- the DRY guard ---

    def test_maskless_set_is_the_SAME_frozenset_object_as_TransformerLayer(self):
        """The decoder must READ `TransformerLayer`'s set, never redeclare it.

        Object identity, not equality: a locally re-declared
        `frozenset({'fnet', 'anchor', 'lighthouse'})` would compare EQUAL and
        then silently drift the day a fourth maskless type is added to one
        side only. That drift is exactly what D-018 already had to undo for
        the FFN parameter table in this same file.
        """
        assert (
            TransformerDecoderLayer._MASKLESS_ATTENTION_TYPES
            is TransformerLayer._MASKLESS_ATTENTION_TYPES
        ), "the decoder re-declared the maskless set instead of reading it"

    def test_every_maskless_type_is_covered_by_this_suite(self):
        """If a fourth maskless type is added, this suite must be extended."""
        assert set(TransformerLayer._MASKLESS_ATTENTION_TYPES) == set(_MASKLESS_SEQ_LEN)

    def test_fnet_self_attention_params_match_TransformerLayer(self):
        """`fnet` is parameter-free; neither dispatcher may inject dim/num_heads.

        Measured when this test was written: the attention factory did NOT reject
        the injected `dim`/`num_heads` -- it dropped them silently, so this is not
        what made `fnet` fail (the plan predicted it was). It was fixed anyway
        because the FFN factory had already turned exactly this silent drop into a
        hard raise (D-023), and the decoder should not be the one site left holding
        a latent break.

        RE-MEASURED 2026-08-17: that prediction came true. `create_attention_layer`
        is now strict (plan-2026-08-17T183311-79c63e38/D-011) and RAISES on
        `dim`/`num_heads` for `fnet`, so this equality is no longer a tidiness
        guard -- it is what keeps both dispatchers constructible at all.
        """
        dec = TransformerDecoderLayer(
            hidden_size=self.HIDDEN, num_heads=self.HEADS,
            intermediate_size=self.INTER, self_attention_type='fnet',
        )
        enc = TransformerLayer(
            hidden_size=self.HIDDEN, num_heads=self.HEADS,
            intermediate_size=self.INTER, attention_type='fnet',
        )
        assert dec._self_attention_params('a') == enc._get_attention_params('a')

    # --- the flipped scope pin: F-07 is now CLOSED for all four types ---

    @pytest.mark.parametrize("attention_type, param, expected", [
        # `window_size` is `TransformerLayer`'s own default (8), read from the
        # shared `_DEFAULT_ATTENTION_WINDOW_SIZE`, NOT re-typed here.
        ('window', 'window_size', 8),
        # `num_kv_heads` degrades to `num_heads` (= plain MHA), matching
        # `TransformerLayer.n_kv_head`'s `None -> num_heads` resolution.
        ('group_query', 'num_kv_heads', HEADS),
        ('differential', 'head_dim', HIDDEN // HEADS),
        ('multi_head_latent', 'kv_latent_dim', max(1, HIDDEN // 4)),
    ])
    def test_f07_parameter_gaps_now_carry_a_default(
            self, attention_type, param, expected):
        """The flipped F-07 SCOPE PIN — same site, same four pairs, inverted.

        Until step 6 of `plan-2026-07-31T132403-b3f540cb` this was
        `test_f07_parameter_gaps_still_raise`, a `pytest.raises(ValueError,
        match=<param>)` asserting that the decoder supplied NO type-specific
        attention params and that all four types were therefore unconstructable.
        Its docstring said closing F-07 must be "a deliberate act with a test to
        flip, not a side effect that nobody notices". This is that flip, at the
        same site: construction must now SUCCEED and the defaulted value must be
        the one `TransformerLayer` would have used.

        RED captures at `5727c907`, before the fix — all four
        `ValueError`, all four raised by the attention factory's own strict
        required-params check (there was no decoder-side wrapper to reword
        them):

        * `window` — "Failed to create 'window' attention layer
          (create_grid_window_attention). Required parameters: ['dim',
          'window_size', 'num_heads']. Provided parameters: ['dim', 'num_heads']."
        * `group_query` — same shape, "Required parameters: ['dim', 'num_heads',
          'num_kv_heads']".
        * `differential` — "... ['dim', 'num_heads', 'head_dim']".
        * `multi_head_latent` — "... ['dim', 'num_heads', 'kv_latent_dim']".

        `window` is included: G-07 was resolved BY EXECUTION rather than by
        argument (see
        :meth:`TestWindowSelfAttentionIsMaskedNotMaskless.test_window_honours_the_causal_mask_at_window_size_squared`)
        and the verdict was that `window` genuinely HONOURS a causal mask, so it
        earns a default rather than joining `_MASKLESS_ATTENTION_TYPES`.
        """
        layer = TransformerDecoderLayer(
            hidden_size=self.HIDDEN, num_heads=self.HEADS,
            intermediate_size=self.INTER, self_attention_type=attention_type,
        )
        params = layer._self_attention_params('self_attention')
        assert param in params, (
            f"'{attention_type}' still supplies no '{param}' — F-07 is not closed"
        )
        assert params[param] == expected

    # `head_dim` is the one override that is not free: the factory enforces
    # `dim == num_heads * head_dim`, so overriding it requires moving
    # `num_heads` with it (MEASURED: `head_dim=8` at `num_heads=4` raises
    # "dim (64) must equal num_heads * head_dim (4 * 8 = 32)").
    @pytest.mark.parametrize("attention_type, param, override, num_heads", [
        ('window', 'window_size', 4, HEADS),
        ('group_query', 'num_kv_heads', 2, HEADS),
        ('differential', 'head_dim', 32, 2),
        ('multi_head_latent', 'kv_latent_dim', 5, HEADS),
    ])
    def test_f07_defaults_are_overridable_by_attention_args(
            self, attention_type, param, override, num_heads):
        """A default must not become a ceiling: `attention_args` merges LAST."""
        layer = TransformerDecoderLayer(
            hidden_size=self.HIDDEN, num_heads=num_heads,
            intermediate_size=self.INTER, self_attention_type=attention_type,
            attention_args={param: override},
        )
        assert layer._self_attention_params('self_attention')[param] == override

    @pytest.mark.parametrize("attention_type", [
        'window', 'group_query', 'differential', 'multi_head_latent'])
    def test_f07_types_agree_with_TransformerLayer_on_the_required_params(
            self, attention_type):
        """D-015: both dispatchers must read ONE table, not two copies.

        Compares only the type-specific REQUIRED keys, because the two blocks
        legitimately differ on the generic conveniences (the encoder forwards
        `dropout_rate`/`use_bias`/`kernel_initializer` to more types than the
        decoder does). It is the required keys whose divergence WAS F-07.
        """
        dec = TransformerDecoderLayer(
            hidden_size=self.HIDDEN, num_heads=self.HEADS,
            intermediate_size=self.INTER, self_attention_type=attention_type,
        )._self_attention_params('a')
        enc = TransformerLayer(
            hidden_size=self.HIDDEN, num_heads=self.HEADS,
            intermediate_size=self.INTER, attention_type=attention_type,
        )._get_attention_params('a')
        required = {'window_size', 'num_kv_heads', 'head_dim',
                    'lambda_init', 'kv_latent_dim'}
        assert {k: v for k, v in dec.items() if k in required} == \
               {k: v for k, v in enc.items() if k in required}

    def test_maskless_type_warns_that_causality_is_not_enforced(self, caplog):
        """A maskless self-attention makes `use_causal_mask=True` a no-op.

        The decoder cannot hand these types a mask at all, so a caller who
        asked for causality does not get it. That must be said out loud at
        construction rather than discovered as a leak during training. It is a
        warning and not a raise because the combination is legitimate for a
        non-autoregressive decoder -- it is the SILENCE that is the defect.
        """
        with caplog.at_level(logging.WARNING):
            TransformerDecoderLayer(
                hidden_size=self.HIDDEN, num_heads=self.HEADS,
                intermediate_size=self.INTER, self_attention_type='fnet',
                use_causal_mask=True,
            )
        assert any("causal" in r.message.lower() for r in caplog.records), (
            "constructing a maskless-attention decoder with use_causal_mask=True "
            "produced no warning; the causality request is silently dropped"
        )

    def test_no_causality_warning_on_the_ordinary_path(self, caplog):
        """CONTROL: a warning that always fires would carry no information."""
        with caplog.at_level(logging.WARNING):
            layer = TransformerDecoderLayer(
                hidden_size=self.HIDDEN, num_heads=self.HEADS,
                intermediate_size=self.INTER, self_attention_type='multi_head',
                use_causal_mask=True,
            )
        assert layer.use_causal_mask is True
        assert not any("causal" in r.message.lower() for r in caplog.records)

    def test_no_causality_warning_when_causality_was_not_requested(self, caplog):
        """CONTROL: maskless + `use_causal_mask=False` is a coherent request."""
        with caplog.at_level(logging.WARNING):
            TransformerDecoderLayer(
                hidden_size=self.HIDDEN, num_heads=self.HEADS,
                intermediate_size=self.INTER, self_attention_type='fnet',
                use_causal_mask=False,
            )
        assert not any("causal" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# F-09 -- `layer_idx` forwarding through `TransformerDecoderLayer.call`
# ---------------------------------------------------------------------------

_F09_HIDDEN, _F09_HEADS, _F09_INTER = 32, 4, 64
_F09_DEC_SEQ, _F09_ENC_SEQ, _F09_BATCH = 6, 5, 2
_F09_HEAD_DIM = 8

# 0 is `DifferentialMultiHeadAttention.call`'s OWN default -- the value that
# silently ran at every stack depth before this fix -- so it is the shallow
# end of the probe. 5 is deep enough that the schedule has visibly moved:
# MEASURED `get_lambda(0) = 0.1600` vs `get_lambda(5) = 0.4954` (>3x).
_F09_SHALLOW_IDX, _F09_DEEP_IDX = 0, 5


def _f09_seed_non_zero_weights(layer: TransformerDecoderLayer, seed: int) -> None:
    """Overwrite every built variable with seeded NON-ZERO values.

    Default initializers put ZEROS in every bias and beta, which makes several
    sub-paths structurally unobservable (prior-plan D-008). This assigns from a
    seeded numpy RNG instead and asserts in-fixture that at least one bias is
    genuinely non-zero.

    `lambda_param` is deliberately EXCLUDED from the randomization and pinned to
    the layer's own default `0.8`. `get_lambda` computes
    ``clip(lambda_param * schedule(layer_idx), 0.1, 0.9)``, so a randomly drawn
    negative or near-zero `lambda_param` would saturate the clip at `0.1` for
    BOTH indices and make the depth-variance probe vacuous through no fault of
    the code under test.
    """
    rng = np.random.default_rng(seed)
    for w in layer.weights:
        dtype = keras.backend.standardize_dtype(w.dtype)
        if w.path.endswith('lambda_param'):
            w.assign(np.full(w.shape, 0.8, dtype=dtype))
        else:
            w.assign(rng.normal(0.0, 0.35, size=w.shape).astype(dtype))

    biases = [w for w in layer.weights if w.path.endswith(('bias', 'beta'))]
    assert biases, "fixture found no bias/beta variables to seed"
    assert any(np.any(np.asarray(b) != 0.0) for b in biases), (
        "fixture failed: every bias is still zero, so bias-dependent paths "
        "would be unobservable"
    )


def _f09_build_decoder(
        normalization_position: str = 'post',
        self_attention_type: str = 'differential',
        seed: int = 1234,
) -> TransformerDecoderLayer:
    """Construct, BUILD (by one call) and seed a decoder layer."""
    attention_args = (
        {'head_dim': _F09_HEAD_DIM} if self_attention_type == 'differential' else {}
    )
    layer = TransformerDecoderLayer(
        hidden_size=_F09_HIDDEN, num_heads=_F09_HEADS,
        intermediate_size=_F09_INTER,
        self_attention_type=self_attention_type,
        attention_args=attention_args,
        normalization_position=normalization_position,
        dropout_rate=0.0, attention_dropout_rate=0.0,
    )
    x = np.zeros((_F09_BATCH, _F09_DEC_SEQ, _F09_HIDDEN), dtype='float32')
    e = np.zeros((_F09_BATCH, _F09_ENC_SEQ, _F09_HIDDEN), dtype='float32')
    layer(x, e, training=False)
    _f09_seed_non_zero_weights(layer, seed)
    return layer


def _f09_inputs(seed: int = 7):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(_F09_BATCH, _F09_DEC_SEQ, _F09_HIDDEN)).astype('float32')
    e = rng.normal(size=(_F09_BATCH, _F09_ENC_SEQ, _F09_HIDDEN)).astype('float32')
    return x, e


class TestDecoderLayerIdxForwarding:
    """F-09: the decoder must forward `layer_idx` to `differential` self-attention.

    `TransformerDecoderLayer.call()` had NO `layer_idx` parameter at all, so
    `DifferentialMultiHeadAttention.call()`'s own default `layer_idx=0` ran at
    every stack depth and the per-layer lambda schedule was inert. The sibling
    `TransformerLayer.call()` has accepted and forwarded `layer_idx` at both its
    pre-norm and post-norm `differential` branches all along; this suite pins the
    decoder to the same contract.

    RED captures at `5500c6bc` (before the fix):

    * mechanical -- `layer(x, e, layer_idx=5)` raised
      ``ValueError: Arguments `target` and `attr` must have the same structure``
      from Keras' own kwarg handling, NOT the predicted
      ``TypeError: unexpected keyword argument``. The CLASS of failure held
      (the parameter does not exist); the predicted type did not.
    * behavioural -- two identically-weighted decoders at `layer_idx` 0 vs 5
      produced `np.allclose(out0, out5) == True`, max |diff| exactly `0.0`:
      the decoder was provably depth-invariant.

    NOTE for future readers: `TransformerDecoderLayer` has ZERO production
    consumers, so nothing in `src/` currently builds a stack that would supply a
    real per-layer index. The forwarding is correct but unexercised outside these
    tests; a future stack builder must pass its own loop index.
    """

    @pytest.mark.parametrize("normalization_position", ['post', 'pre'])
    def test_call_accepts_layer_idx(self, normalization_position):
        """MECHANICAL: the parameter must exist at all (RED-today by itself)."""
        layer = _f09_build_decoder(normalization_position)
        x, e = _f09_inputs()
        out = np.asarray(layer(x, e, layer_idx=_F09_DEEP_IDX, training=False))
        assert out.shape == (_F09_BATCH, _F09_DEC_SEQ, _F09_HIDDEN)
        assert np.all(np.isfinite(out))

    @pytest.mark.parametrize("normalization_position", ['post', 'pre'])
    def test_depth_changes_the_output(self, normalization_position):
        """BEHAVIOURAL (SC-8): two IDENTICALLY-weighted decoders at different
        depths must produce DIFFERENT outputs.

        Weights are copied with `set_weights` rather than re-seeded, because
        `keras.utils.set_random_seed` does NOT reproduce a subsequent
        `keras.random.*` draw on this backend (measured at step 4) -- two layers
        built under the same seed are not guaranteed identical.
        """
        shallow = _f09_build_decoder(normalization_position, seed=1234)
        deep = _f09_build_decoder(normalization_position, seed=4321)
        deep.set_weights(shallow.get_weights())

        x, e = _f09_inputs()
        out_shallow = np.asarray(
            shallow(x, e, layer_idx=_F09_SHALLOW_IDX, training=False))
        out_deep = np.asarray(
            deep(x, e, layer_idx=_F09_DEEP_IDX, training=False))

        max_diff = float(np.max(np.abs(out_shallow - out_deep)))
        assert not np.allclose(out_shallow, out_deep), (
            f"decoder output is depth-INVARIANT (max |diff| = {max_diff:.3e}): "
            f"layer_idx={_F09_DEEP_IDX} produced the same result as "
            f"layer_idx={_F09_SHALLOW_IDX}, so the per-layer lambda schedule is "
            f"not reaching the differential attention"
        )

    @pytest.mark.parametrize("normalization_position", ['post', 'pre'])
    def test_same_depth_is_identical(self, normalization_position):
        """CONTROL: the difference above must come from `layer_idx`, not from
        the two layers holding different weights."""
        a = _f09_build_decoder(normalization_position, seed=1234)
        b = _f09_build_decoder(normalization_position, seed=4321)
        b.set_weights(a.get_weights())

        x, e = _f09_inputs()
        out_a = np.asarray(a(x, e, layer_idx=_F09_DEEP_IDX, training=False))
        out_b = np.asarray(b(x, e, layer_idx=_F09_DEEP_IDX, training=False))
        np.testing.assert_allclose(out_a, out_b, rtol=1e-6, atol=1e-6)

    def test_the_lambda_schedule_itself_moves(self):
        """ORACLE: pins the magnitude the decoder is supposed to be exposing.

        If this ever goes flat, `test_depth_changes_the_output` would be
        measuring nothing even with the forwarding correct.
        """
        layer = _f09_build_decoder('post')
        lam0 = float(np.asarray(layer.self_attention.get_lambda(_F09_SHALLOW_IDX)))
        lam5 = float(np.asarray(layer.self_attention.get_lambda(_F09_DEEP_IDX)))
        assert abs(lam5 - lam0) > 0.1, (
            f"lambda schedule is flat: get_lambda({_F09_SHALLOW_IDX})={lam0:.4f} "
            f"vs get_lambda({_F09_DEEP_IDX})={lam5:.4f}"
        )

    def test_layer_idx_is_accepted_by_a_non_differential_type(self):
        """`layer_idx` is a uniform `call()` parameter, as on `TransformerLayer`:
        a type that has no use for it must ignore it, not crash."""
        layer = _f09_build_decoder('post', self_attention_type='multi_head')
        x, e = _f09_inputs()
        out = np.asarray(layer(x, e, layer_idx=_F09_DEEP_IDX, training=False))
        assert out.shape == (_F09_BATCH, _F09_DEC_SEQ, _F09_HIDDEN)
        assert np.all(np.isfinite(out))

    def test_symbolic_trace_honours_layer_idx(self):
        """The new parameter must work under a real `tf.function` trace with a
        dynamic batch/sequence, not only eagerly.

        `layer_idx` is a PYTHON int baked into the trace (exactly as on
        `TransformerLayer`), so the two traces below are two different concrete
        functions -- which is precisely what must be exercised.
        """
        shallow = _f09_build_decoder('post', seed=1234)
        deep = _f09_build_decoder('post', seed=4321)
        deep.set_weights(shallow.get_weights())

        spec = [
            tf.TensorSpec(shape=[None, None, _F09_HIDDEN], dtype=tf.float32),
            tf.TensorSpec(shape=[None, None, _F09_HIDDEN], dtype=tf.float32),
        ]
        fn_shallow = tf.function(
            lambda x, e: shallow(x, e, layer_idx=_F09_SHALLOW_IDX, training=False),
            input_signature=spec)
        fn_deep = tf.function(
            lambda x, e: deep(x, e, layer_idx=_F09_DEEP_IDX, training=False),
            input_signature=spec)

        x, e = _f09_inputs()
        out_shallow = np.asarray(fn_shallow(x, e))
        out_deep = np.asarray(fn_deep(x, e))

        assert out_shallow.shape == (_F09_BATCH, _F09_DEC_SEQ, _F09_HIDDEN)
        assert np.all(np.isfinite(out_deep))
        max_diff = float(np.max(np.abs(out_shallow - out_deep)))
        assert not np.allclose(out_shallow, out_deep), (
            f"symbolic path is depth-INVARIANT (max |diff| = {max_diff:.3e})"
        )

    def test_layer_idx_is_not_a_constructor_parameter(self):
        """I7: `layer_idx` is a `call()` argument. It must NOT leak into
        `get_config()`, which would change the serialized surface."""
        layer = _f09_build_decoder('post')
        assert 'layer_idx' not in layer.get_config()


# ---------------------------------------------------------------------
# F-07 / G-06 / G-07: decoder self-attention defaults, and the measured
# verdict on `window`.
# ---------------------------------------------------------------------

_F07_HIDDEN, _F07_HEADS, _F07_INTER = 64, 4, 128
_F07_ENC_SEQ, _F07_BATCH = 10, 2
#: `window` self-attention is causal ONLY at `seq_len == window_size ** 2`
#: (measured; see `TestWindowSelfAttentionIsMaskedNotMaskless`). 4**2 = 16.
_F07_WINDOW_SIZE = 4
_F07_WINDOW_SEQ = _F07_WINDOW_SIZE ** 2


def _f07_seeded_inputs(seq_len: int, seed: int = 20260731):
    """I8/(c): seeded NON-ZERO decoder + encoder activations."""
    rng = np.random.default_rng(seed)
    x = ops.convert_to_tensor(
        rng.normal(size=(_F07_BATCH, seq_len, _F07_HIDDEN)).astype("float32"))
    enc = ops.convert_to_tensor(
        rng.normal(size=(_F07_BATCH, _F07_ENC_SEQ, _F07_HIDDEN)).astype("float32"))
    assert float(np.max(np.abs(np.asarray(x)))) > 1e-2
    return x, enc


def _f07_decoder(attention_type: str, **kwargs):
    return TransformerDecoderLayer(
        hidden_size=_F07_HIDDEN, num_heads=_F07_HEADS,
        intermediate_size=_F07_INTER, self_attention_type=attention_type,
        dropout_rate=0.0, attention_dropout_rate=0.0, **kwargs)


class TestF07DecoderAttentionDefaults:
    """The four formerly-unconstructable types must also RUN and round-trip."""

    @pytest.mark.parametrize("normalization_position", ['post', 'pre'])
    @pytest.mark.parametrize("attention_type, extra_args, seq_len", [
        ('window', {'window_size': _F07_WINDOW_SIZE}, _F07_WINDOW_SEQ),
        ('group_query', {}, 8),
        ('differential', {}, 8),
        ('multi_head_latent', {}, 8),
    ])
    def test_constructs_and_runs(self, attention_type, extra_args, seq_len,
                                 normalization_position):
        """Construction alone proves nothing: the defaulted VALUE has to be a
        value the layer can actually build and run with. `call()` has two
        self-attention sites, so both `normalization_position` branches run."""
        layer = _f07_decoder(attention_type, attention_args=extra_args,
                             normalization_position=normalization_position)
        x, enc = _f07_seeded_inputs(seq_len)
        out = np.asarray(layer(x, enc, training=False))
        assert out.shape == (_F07_BATCH, seq_len, _F07_HIDDEN)
        assert np.all(np.isfinite(out))
        assert float(np.max(np.abs(out))) > 1e-6, "output is all-zero"

    @pytest.mark.parametrize("attention_type, extra_args, seq_len", [
        ('window', {'window_size': _F07_WINDOW_SIZE}, _F07_WINDOW_SEQ),
        ('group_query', {}, 8),
        ('differential', {}, 8),
        ('multi_head_latent', {}, 8),
    ])
    def test_serialization_round_trip_by_value(self, attention_type,
                                               extra_args, seq_len):
        """I7: a `.keras` round-trip must restore VALUES on the new path."""
        dec_in = keras.Input(shape=(seq_len, _F07_HIDDEN))
        enc_in = keras.Input(shape=(_F07_ENC_SEQ, _F07_HIDDEN))
        out = _f07_decoder(attention_type, attention_args=extra_args)(dec_in, enc_in)
        model = models.Model([dec_in, enc_in], out)
        x, enc = _f07_seeded_inputs(seq_len)
        ref = np.asarray(model([x, enc], training=False))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, f"dec_f07_{attention_type}.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
        got = np.asarray(loaded([x, enc], training=False))
        assert float(np.max(np.abs(ref))) > 1e-3, "round-trip compared all-zero values"
        np.testing.assert_allclose(ref, got, rtol=0, atol=0)


class TestWindowSelfAttentionIsMaskedNotMaskless:
    """G-07, RESOLVED BY EXECUTION: `window` HONOURS the causal mask.

    The finding left open whether `create_grid_window_attention`'s layer merely
    ACCEPTS `attention_mask` or actually applies it — "accepts" proves nothing
    here, since `fnet` also accepts the kwarg and then dies on a shape mismatch.
    Measured at plan step 6: it applies it exactly, at `seq_len ==
    window_size ** 2`, and raises loudly at every other length. So `window` must
    NOT join `_MASKLESS_ATTENTION_TYPES` — that would silently drop a mask the
    layer would have honoured — and instead earns a `window_size` default.
    """

    def _mask(self, T):
        r = ops.arange(T)[:, None]
        c = ops.arange(T)[None, :]
        return ops.cast(r >= c, "float32")[None, :, :]

    def test_window_is_not_in_the_maskless_set(self):
        assert 'window' not in TransformerDecoderLayer._MASKLESS_ATTENTION_TYPES
        assert 'window' not in TransformerLayer._MASKLESS_ATTENTION_TYPES

    def test_window_honours_the_causal_mask_at_window_size_squared(self):
        """The verdict, with its own live control.

        Perturb the LAST token and require every earlier position to move by
        exactly `0.0` under the causal mask, while the SAME perturbation with
        no mask moves them a lot. Without the control this test would pass for
        a layer that returned a constant.
        """
        from dl_techniques.layers.attention import create_attention_layer
        keras.utils.set_random_seed(7)
        attn = create_attention_layer(
            'window', dim=_F07_HIDDEN, window_size=_F07_WINDOW_SIZE,
            num_heads=_F07_HEADS, dropout_rate=0.0)
        rng = np.random.default_rng(99)
        x = rng.normal(size=(_F07_BATCH, _F07_WINDOW_SEQ, _F07_HIDDEN)).astype("float32")
        x2 = x.copy()
        x2[:, -1, :] += 100.0
        m = self._mask(_F07_WINDOW_SEQ)

        masked = np.asarray(attn(ops.convert_to_tensor(x), attention_mask=m, training=False))
        masked_p = np.asarray(attn(ops.convert_to_tensor(x2), attention_mask=m, training=False))
        leak = float(np.max(np.abs(masked_p - masked)[:, :-1, :]))
        assert leak == 0.0, f"causal mask is NOT honoured: leak {leak:.6e}"

        plain = np.asarray(attn(ops.convert_to_tensor(x), training=False))
        plain_p = np.asarray(attn(ops.convert_to_tensor(x2), training=False))
        control = float(np.max(np.abs(plain_p - plain)[:, :-1, :]))
        assert control > 1e-2, (
            f"UNMASKED control did not move ({control:.6e}) — this probe cannot "
            f"distinguish a honoured mask from a dead layer")
        assert float(np.max(np.abs(masked - plain))) > 1e-2, (
            "masked and unmasked outputs are identical — the mask is ignored")

    def test_window_decoder_is_causal_end_to_end(self):
        """The same verdict through the block the fix actually changed."""
        layer = _f07_decoder('window', attention_args={'window_size': _F07_WINDOW_SIZE})
        x, enc = _f07_seeded_inputs(_F07_WINDOW_SEQ)
        base = np.asarray(layer(x, enc, training=False))
        x2 = np.asarray(x).copy()
        x2[:, -1, :] += 50.0
        pert = np.asarray(layer(ops.convert_to_tensor(x2), enc, training=False))
        # Cross-attention and the FFN are position-wise, so a self-attention
        # leak is the only way an earlier position can move.
        leak = float(np.max(np.abs(pert - base)[:, :-1, :]))
        assert leak == 0.0, f"decoder `window` self-attention leaks the future: {leak:.6e}"
        assert float(np.max(np.abs(pert - base)[:, -1, :])) > 1e-2, \
            "the perturbed position itself did not move — probe is vacuous"

    @pytest.mark.parametrize("seq_len", [8, 32])
    def test_window_raises_loudly_at_any_other_sequence_length(self, seq_len):
        """The documented cost of the verdict: it is a CALL-time raise, not a
        silent non-causal block. Pinned so the docstring cannot go stale."""
        layer = _f07_decoder('window', attention_args={'window_size': _F07_WINDOW_SIZE})
        x, enc = _f07_seeded_inputs(seq_len)
        with pytest.raises(ValueError, match=r"rank-3 pairwise attention_mask"):
            layer(x, enc, training=False)

    def test_window_without_a_causal_mask_runs_at_any_length(self):
        """The documented escape hatch: `use_causal_mask=False` works anywhere."""
        layer = _f07_decoder('window', attention_args={'window_size': _F07_WINDOW_SIZE},
                             use_causal_mask=False)
        x, enc = _f07_seeded_inputs(12)
        out = np.asarray(layer(x, enc, training=False))
        assert out.shape == (_F07_BATCH, 12, _F07_HIDDEN)
        assert np.all(np.isfinite(out))


class TestDecoderAttentionConstructionErrorIsFriendly:
    """F-07's second half: the decoder built both attention layers bare.

    RED capture at `5727c907`: an invalid override produced the attention
    factory's own message only — "Failed to create 'multi_head' attention layer
    (MultiHeadAttention). Required parameters: ['dim']. Provided parameters:
    ['dim', 'num_heads', 'dropout_rate', 'use_bias']." — which names neither
    which of the block's TWO attention layers failed nor which of the caller's
    keys were overrides. `TransformerLayer` has carried such a wrapper all along.
    """

    @pytest.mark.parametrize("kwargs, role, args_name, key", [
        ({'attention_args': {'num_heads': 0}}, 'self-attention', 'attention_args', 'num_heads'),
        ({'cross_attention_args': {'num_heads': 0}}, 'cross-attention', 'cross_attention_args', 'num_heads'),
    ])
    def test_message_names_the_role_and_the_caller_args(
            self, kwargs, role, args_name, key):
        with pytest.raises(ValueError) as exc:
            TransformerDecoderLayer(
                hidden_size=_F07_HIDDEN, num_heads=_F07_HEADS,
                intermediate_size=_F07_INTER, **kwargs)
        msg = str(exc.value)
        assert role in msg, f"message does not say which attention failed: {msg}"
        assert args_name in msg, f"message does not name the arg dict: {msg}"
        assert key in msg, f"message does not list the caller's key: {msg}"
        assert "Original error" in msg, "the underlying factory error was swallowed"
