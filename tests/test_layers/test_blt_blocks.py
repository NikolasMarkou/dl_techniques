"""Tests for the BLT (Byte Latent Transformer) building blocks.

Covers all seven layers in ``blt_blocks.py``. Single-tensor-input layers
(``EntropyModel``, ``GlobalTransformer``) use a functional ``.keras`` round-trip;
the multi-argument-``call`` layers (``LocalEncoder``, ``LocalDecoder``) use a
``keras.Model`` wrapper round-trip (recreates the layer + restores weights),
which exercises the ``build()`` weight-structure consistency the H6 fix touches.
"""

import os
import keras
import numpy as np
import pytest

from dl_techniques.layers.blt_blocks import (
    ByteTokenizer,
    EntropyModel,
    DynamicPatcher,
    PatchPooling,
    LocalEncoder,
    GlobalTransformer,
    LocalDecoder,
)

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
# ByteTokenizer (utility layer, no call)
# ---------------------------------------------------------------------

class TestByteTokenizer:

    def test_text_round_trip(self):
        tok = ByteTokenizer()
        ids = tok.text_to_bytes("Hello", add_bos=True, add_eos=True)
        assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id
        assert tok.tokens_to_text(ids) == "Hello"

    def test_compute_output_shape(self):
        assert ByteTokenizer().compute_output_shape((B, SEQ)) == (B, None)

    def test_get_config_round_trip(self):
        tok = ByteTokenizer(vocab_size=300, byte_offset=5)
        rebuilt = ByteTokenizer.from_config(tok.get_config())
        assert rebuilt.vocab_size == 300 and rebuilt.byte_offset == 5


# ---------------------------------------------------------------------
# DynamicPatcher (stateless tensor op)
# ---------------------------------------------------------------------

class TestDynamicPatcher:

    def test_forward_pass(self):
        patcher = DynamicPatcher(max_patches=NP_)
        entropy = keras.ops.convert_to_tensor(
            np.random.default_rng(0).standard_normal((B, SEQ)).astype("float32")
        )
        out = patcher(entropy)
        assert tuple(out.shape) == (B, NP_)

    def test_get_config_round_trip(self):
        patcher = DynamicPatcher(entropy_threshold=2.0, max_patches=16)
        rebuilt = DynamicPatcher.from_config(patcher.get_config())
        assert rebuilt.max_patches == 16


# ---------------------------------------------------------------------
# EntropyModel (single-tensor call -> functional .keras round-trip)
# ---------------------------------------------------------------------

class TestEntropyModel:

    def _make(self):
        return EntropyModel(vocab_size=VOCAB, hidden_dim=HID, num_layers=1,
                            num_heads=2, max_seq_len=64)

    def test_forward_pass(self):
        out = self._make()(_tokens())
        assert tuple(out.shape) == (B, SEQ, VOCAB)

    def test_compute_output_shape(self):
        assert self._make().compute_output_shape((B, SEQ)) == (B, SEQ, VOCAB)

    def test_serialization_round_trip(self, tmp_path):
        inp = keras.Input(shape=(SEQ,), dtype="int32")
        out = self._make()(inp)
        model = keras.Model(inp, out)
        toks = _tokens()
        y0 = model(toks)
        path = os.path.join(tmp_path, "entropy.keras")
        model.save(path)
        loaded = keras.models.load_model(path, custom_objects={"EntropyModel": EntropyModel})
        y1 = loaded(toks)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(y1),
            rtol=1e-5, atol=1e-5,
        )


# ---------------------------------------------------------------------
# GlobalTransformer (single-tensor call -> functional .keras round-trip)
# ---------------------------------------------------------------------

class TestGlobalTransformer:

    def _make(self):
        return GlobalTransformer(global_dim=GDIM, num_global_layers=1,
                                num_heads_global=2, max_patches=NP_)

    def test_forward_pass(self):
        x = keras.ops.convert_to_tensor(
            np.random.default_rng(0).standard_normal((B, NP_, GDIM)).astype("float32")
        )
        assert tuple(self._make()(x).shape) == (B, NP_, GDIM)

    def test_compute_output_shape(self):
        assert self._make().compute_output_shape((B, NP_, GDIM)) == (B, NP_, GDIM)

    def test_serialization_round_trip(self, tmp_path):
        inp = keras.Input(shape=(NP_, GDIM))
        out = self._make()(inp)
        model = keras.Model(inp, out)
        x = np.random.default_rng(0).standard_normal((B, NP_, GDIM)).astype("float32")
        y0 = model(x)
        path = os.path.join(tmp_path, "global.keras")
        model.save(path)
        loaded = keras.models.load_model(
            path, custom_objects={"GlobalTransformer": GlobalTransformer}
        )
        y1 = loaded(x)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(y1),
            rtol=1e-5, atol=1e-5,
        )


# ---------------------------------------------------------------------
# PatchPooling (multi-arg call)
# ---------------------------------------------------------------------

class TestPatchPooling:

    def test_forward_pass(self):
        pool = PatchPooling(output_dim=GDIM, num_queries=2, max_patches=NP_)
        byte_hiddens = keras.ops.convert_to_tensor(
            np.random.default_rng(0).standard_normal((B, SEQ, HID)).astype("float32")
        )
        out = pool(byte_hiddens, _patch_ids())
        assert out.shape[0] == B and out.shape[-1] == GDIM

    def test_get_config_round_trip(self):
        pool = PatchPooling(output_dim=GDIM, num_queries=2, max_patches=NP_)
        rebuilt = PatchPooling.from_config(pool.get_config())
        assert rebuilt.output_dim == GDIM


# ---------------------------------------------------------------------
# LocalEncoder / LocalDecoder (multi-arg call -> Model-wrapper round-trip)
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


@keras.saving.register_keras_serializable()
class _DecWrapper(keras.Model):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.dec = LocalDecoder(
            vocab_size=VOCAB, local_dim=HID, global_dim=GDIM, num_local_layers=1,
            num_heads_local=2,
        )

    def call(self, inputs, training=None):
        return self.dec(inputs[0], inputs[1], inputs[2], training=training)


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


class TestLocalDecoder:

    def _gctx(self):
        return keras.ops.convert_to_tensor(
            np.random.default_rng(2).standard_normal((B, NP_, GDIM)).astype("float32")
        )

    def test_forward_pass(self):
        dec = LocalDecoder(vocab_size=VOCAB, local_dim=HID, global_dim=GDIM,
                          num_local_layers=1, num_heads_local=2)
        out = dec(_tokens(), self._gctx(), _patch_ids())
        assert tuple(out.shape) == (B, SEQ, VOCAB)

    def test_serialization_round_trip(self, tmp_path):
        model = _DecWrapper()
        inputs = [_tokens(), self._gctx(), _patch_ids()]
        y0 = model(inputs)
        path = os.path.join(tmp_path, "dec.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        y1 = loaded(inputs)
        assert tuple(y0.shape) == tuple(y1.shape) == (B, SEQ, VOCAB)
