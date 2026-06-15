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
        layer = self._layer(use_free_transformer=True, num_latent_bits=4)
        out = layer(x, training=False)
        assert tuple(out.shape) == (2, self.T, self.HID)

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
