"""Tests for ``LocalEncoder`` (``dl_techniques.layers.blt.local_encoder``)."""

import os

import keras
import numpy as np

from dl_techniques.layers.blt.local_encoder import LocalEncoder

VOCAB, HID, SEQ, NP_, GDIM = 32, 16, 10, 8, 16
B = 2


def _tokens():
    return keras.ops.convert_to_tensor(
        np.random.default_rng(0).integers(0, VOCAB, size=(B, SEQ)).astype("int32")
    )


def _patch_ids():
    return keras.ops.convert_to_tensor(
        np.random.default_rng(1).integers(0, NP_, size=(B, SEQ)).astype("int32")
    )


# ---------------------------------------------------------------------
# LocalEncoder (multi-arg call -> Model-wrapper round-trip)
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class _EncWrapper(keras.Model):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.enc = LocalEncoder(
            vocab_size=VOCAB, local_dim=HID, num_local_layers=1, num_heads_local=2,
            max_sequence_length=64, max_patches=NP_, global_dim=GDIM,
            cross_attention_queries=2,
        )

    def call(self, inputs, training=None):
        return self.enc(inputs[0], inputs[1], training=training)


class TestLocalEncoder:

    def test_forward_pass(self):
        enc = LocalEncoder(
            vocab_size=VOCAB, local_dim=HID, num_local_layers=1, num_heads_local=2,
            max_sequence_length=64, max_patches=NP_, global_dim=GDIM,
            cross_attention_queries=2,
        )
        out = enc(_tokens(), _patch_ids())
        assert tuple(out.shape) == (B, NP_, GDIM)

    def test_compute_output_shape(self):
        enc = LocalEncoder(
            vocab_size=VOCAB, local_dim=HID, max_patches=NP_, global_dim=GDIM,
        )
        assert enc.compute_output_shape((B, SEQ)) == (B, NP_, GDIM)

    def test_serialization_round_trip(self, tmp_path):
        model = _EncWrapper()
        inputs = [_tokens(), _patch_ids()]
        y0 = model(inputs)
        path = os.path.join(tmp_path, "enc.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        y1 = loaded(inputs)
        assert tuple(y0.shape) == tuple(y1.shape) == (B, NP_, GDIM)
