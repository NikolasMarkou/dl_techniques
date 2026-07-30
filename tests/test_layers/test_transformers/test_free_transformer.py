"""Test suite for FreeTransformerLayer (encoder cross-attention) + BinaryMapper."""
import pytest
import numpy as np
import tensorflow as tf
import keras
import os
import tempfile

from dl_techniques.layers.transformers.free_transformer import (
    FreeTransformerLayer, BinaryMapper,
)


class TestBinaryMapper:
    def test_forward_shape(self):
        mapper = BinaryMapper(num_bits=4)
        logits = keras.random.normal([2, 8, 4])
        out = mapper(logits, training=False)
        # one-hot over 2^num_bits categories
        assert tuple(out.shape) == (2, 8, 16)

    def test_invalid_num_bits(self):
        with pytest.raises(ValueError):
            BinaryMapper(num_bits=0)

    def test_get_config_round_trip(self):
        mapper = BinaryMapper(num_bits=6)
        rebuilt = BinaryMapper.from_config(mapper.get_config())
        assert rebuilt.num_bits == 6


class TestFreeTransformerLayer:
    HID = 32
    HEADS = 4
    INTER = 64
    T = 12

    def _layer(self, **kw):
        params = dict(hidden_size=self.HID, num_heads=self.HEADS,
                      intermediate_size=self.INTER)
        params.update(kw)
        return FreeTransformerLayer(**params)

    @pytest.fixture
    def x(self):
        return keras.random.normal([2, self.T, self.HID])

    def test_standard_path_forward(self, x):
        """use_free_transformer=False behaves like a standard transformer block."""
        out = self._layer(use_free_transformer=False)(x, training=False)
        assert tuple(out.shape) == (2, self.T, self.HID)

    def test_free_path_training_returns_tuple(self, x):
        """Training path returns (output, bit_logits)."""
        layer = self._layer(use_free_transformer=True, num_latent_bits=4)
        result = layer(x, training=True)
        assert isinstance(result, (tuple, list)) and len(result) == 2
        output, bit_logits = result
        assert tuple(output.shape) == (2, self.T, self.HID)
        assert tuple(bit_logits.shape) == (2, self.T, 4)
        assert np.all(np.isfinite(np.array(output)))

    def test_free_path_inference(self, x):
        """Inference returns the SAME structure as training: (output, bit_logits).
        bit_logits is the uniform-prior zeros tensor."""
        layer = self._layer(use_free_transformer=True, num_latent_bits=4)
        result = layer(x, training=False)
        assert isinstance(result, (tuple, list)) and len(result) == 2
        output, bit_logits = result
        assert tuple(output.shape) == (2, self.T, self.HID)
        assert tuple(bit_logits.shape) == (2, self.T, 4)
        assert np.allclose(np.array(bit_logits), 0.0)  # uniform prior

    def test_output_structure_matches_compute_output_shape(self, x):
        """call() output structure must match compute_output_shape in BOTH modes
        (the structure depends only on use_free_transformer, not on training)."""
        # use_free_transformer=True -> 2-tuple in both modes
        layer = self._layer(use_free_transformer=True, num_latent_bits=4)
        cos = layer.compute_output_shape((2, self.T, self.HID))
        assert isinstance(cos, tuple) and len(cos) == 2 and isinstance(cos[0], tuple)
        for mode in (True, False):
            result = layer(x, training=mode)
            assert isinstance(result, (tuple, list)) and len(result) == 2
            assert tuple(result[0].shape) == tuple(cos[0])
            assert tuple(result[1].shape) == tuple(cos[1])
        # use_free_transformer=False -> single tensor matching single shape
        plain = self._layer(use_free_transformer=False)
        cos_plain = plain.compute_output_shape((2, self.T, self.HID))
        out = plain(x, training=False)
        assert not isinstance(out, (tuple, list))
        assert tuple(out.shape) == tuple(cos_plain)

    def test_encoder_is_cross_attention(self, x):
        """The redesign: the encoder must use cross-attention, so its output
        depends on the K/V (the sequence S), not only the query (zeta)."""
        layer = self._layer(use_free_transformer=True, num_latent_bits=4)
        layer(x, training=True)  # build
        assert layer.encoder_attention_type == 'multi_head_cross'
        q = keras.random.normal([2, self.T, self.HID])
        kv1 = keras.random.normal([2, self.T, self.HID])
        kv2 = keras.random.normal([2, self.T, self.HID])
        o1 = np.array(layer.encoder_attention(q, kv_input=kv1, training=False))
        o2 = np.array(layer.encoder_attention(q, kv_input=kv2, training=False))
        assert not np.allclose(o1, o2, atol=1e-5)

    def test_graph_trace_training(self, x):
        layer = self._layer(use_free_transformer=True, num_latent_bits=4)

        @tf.function
        def traced(inp):
            output, _ = layer(inp, training=True)
            return output

        out = traced(tf.constant(np.array(x)))
        assert tuple(out.shape) == (2, self.T, self.HID)

    def test_get_config_round_trip(self):
        layer = self._layer(use_free_transformer=True, num_latent_bits=8)
        rebuilt = FreeTransformerLayer.from_config(layer.get_config())
        assert rebuilt.use_free_transformer is True
        assert rebuilt.num_latent_bits == 8
        assert rebuilt.encoder_attention_type == 'multi_head_cross'

    def test_model_save_load_round_trip(self, x):
        inp = keras.Input(shape=(self.T, self.HID))
        out = FreeTransformerLayer(hidden_size=self.HID, num_heads=self.HEADS,
                                   intermediate_size=self.INTER,
                                   use_free_transformer=False)(inp)
        model = keras.Model(inp, out)
        ref = model(x, training=False)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "free.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
        out2 = loaded(x, training=False)
        np.testing.assert_allclose(np.array(ref), np.array(out2), atol=1e-5)


class TestEncoderFFNKwargFiltering:
    """The encoder FFN bundle must not hand `create_ffn_layer` keys it rejects.

    `encoder_ffn_type` defaults to 'swiglu', which does not accept `activation`,
    yet the bundle passed `activation=self.activation` unconditionally. The
    factory filtered it out silently -- no warning at any log level -- so the
    layer's own default config was quietly discarding a parameter. This was one
    of only two (call site, ffn_type) pairs in the repo that armed that drop,
    measured over an instrumented 1634-test run.

    The layer now pre-filters its OWN generic defaults to what the chosen
    `encoder_ffn_type` accepts, as `gated_linear_attention_block.py` does.
    """

    @staticmethod
    def _build(caplog_level="WARNING", **kw):
        layer = FreeTransformerLayer(
            hidden_size=32, num_heads=4, intermediate_size=128,
            use_free_transformer=True, dropout_rate=0.0,
            attention_dropout_rate=0.0, **kw
        )
        layer(keras.random.normal([2, 8, 32]), training=False)
        return layer

    def test_default_config_drops_no_key_silently(self, caplog):
        """swiglu default: zero dropped-key warnings.

        This is the defect. Reverting the pre-filter makes the factory warn about
        `activation`, and this assertion fails on the captured record.
        """
        import logging
        with caplog.at_level(logging.WARNING):
            self._build()
        dropped = [
            r.getMessage() for r in caplog.records
            if "unsupported parameter" in r.getMessage()
        ]
        assert not dropped, (
            f"the DEFAULT encoder FFN config still hands swiglu keys it "
            f"rejects: {dropped}"
        )

    def test_misspelled_encoder_ffn_args_key_warns_exactly_once(self, caplog):
        """The caller's own `encoder_ffn_args` is still NOT protected -- but is now loud.

        The pre-filter deliberately covers only this layer's generic defaults;
        `create_ffn_layer` re-applies the same signature intersection and cannot
        distinguish an explicit caller key from a convenience default. That
        residual gap is unchanged by design, so the factory's `logger.warning`
        is what makes it findable. Removing that warning fails here.
        """
        import logging
        with caplog.at_level(logging.WARNING):
            self._build(encoder_ffn_args={"activatoin": "gelu"})
        naming_the_typo = [
            r.getMessage() for r in caplog.records
            if "unsupported parameter" in r.getMessage()
            and "activatoin" in r.getMessage()
        ]
        assert len(naming_the_typo) == 1, (
            f"expected exactly one warning naming the misspelled key, got "
            f"{len(naming_the_typo)}: {naming_the_typo}"
        )

    def test_an_ffn_type_that_accepts_activation_still_receives_it(self):
        """CONTROL: the pre-filter must not strip a key the type DOES accept.

        Without this, a filter that dropped `activation` unconditionally would
        satisfy both tests above while silently changing every mlp-style encoder
        FFN. `mlp` lists `activation` as a real parameter, so it must arrive.
        """
        layer = self._build(encoder_ffn_type="mlp",
                            encoder_ffn_args={"activation": "relu"})
        # MLPBlock stores the raw value as `activation_name` and the resolved
        # callable as `activation_fn`; assert on the resolved one so a silently
        # ignored kwarg cannot pass by leaving the name set.
        resolved = layer.encoder_ffn.activation_fn
        assert resolved is not None
        activation_repr = getattr(resolved, "__name__", str(resolved))
        assert "relu" in activation_repr.lower(), (
            f"the pre-filter stripped an `activation` that mlp accepts: "
            f"{activation_repr}"
        )

    def test_unknown_encoder_ffn_type_raises_naming_alternatives(self):
        """An unknown type must fail loudly, matching the GLA site's behaviour."""
        with pytest.raises(ValueError, match="Unknown encoder_ffn_type"):
            FreeTransformerLayer(
                hidden_size=32, num_heads=4, intermediate_size=128,
                use_free_transformer=True, encoder_ffn_type="not_a_real_ffn",
            )
