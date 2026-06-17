"""Mirrored tests for the SD3 from-scratch text encoders.

Covers CLIPTextEncoder, OpenCLIPTextEncoder, T5Encoder:
instantiation, forward shapes, compute_output_shape (pre/post build),
get_config/from_config round-trip, full .keras save/load round-trip, variable
batch + variable L, and a CLIP causal-mask faithfulness check.
"""

import os
import tempfile

import keras
import numpy as np
import pytest

from dl_techniques.models.sd3_mmdit.text_encoders import (
    CLIPTextEncoder,
    OpenCLIPTextEncoder,
    T5Encoder,
)

# ---------------------------------------------------------------------
# tiny configs (full dims are huge; tests use these)
# ---------------------------------------------------------------------

CLIP_TINY = dict(
    vocab_size=256, embed_dim=32, num_layers=2, num_heads=4, max_seq_len=16
)
OPENCLIP_TINY = dict(
    vocab_size=256, embed_dim=48, num_layers=2, num_heads=6, max_seq_len=16
)
T5_TINY = dict(
    vocab_size=256, embed_dim=32, num_layers=2, num_heads=4, ff_dim=64,
    rel_attention_num_buckets=8, rel_attention_max_distance=16,
)


def _ids(B, L, vocab):
    return np.random.randint(0, vocab, size=(B, L)).astype("int32")


def _mask(B, L):
    m = np.ones((B, L), dtype="int32")
    if L > 2:
        m[:, -1] = 0  # mark last position as padding
    return m


# =====================================================================
# CLIP / OpenCLIP shared behavior
# =====================================================================


class TestCLIPLikeEncoders:

    @pytest.fixture(params=["clip", "openclip"])
    def kind(self, request):
        return request.param

    def _make(self, kind):
        if kind == "clip":
            return CLIPTextEncoder(**CLIP_TINY), CLIP_TINY["embed_dim"]
        return OpenCLIPTextEncoder(**OPENCLIP_TINY), OPENCLIP_TINY["embed_dim"]

    def test_instantiation(self, kind):
        enc, _ = self._make(kind)
        assert isinstance(enc, CLIPTextEncoder)

    def test_forward_shapes(self, kind):
        enc, D = self._make(kind)
        B, L = 2, 12
        vocab = CLIP_TINY["vocab_size"]
        out = enc(_ids(B, L, vocab), attention_mask=_mask(B, L))
        assert out["pooled"].shape == (B, D)
        assert out["last_hidden"].shape == (B, L, D)
        assert out["penultimate"].shape == (B, L, D)
        assert np.all(np.isfinite(np.asarray(out["pooled"])))
        assert np.all(np.isfinite(np.asarray(out["last_hidden"])))

    def test_forward_no_mask(self, kind):
        enc, D = self._make(kind)
        out = enc(_ids(2, 10, CLIP_TINY["vocab_size"]))
        assert out["pooled"].shape == (2, D)

    def test_compute_output_shape_pre_build(self, kind):
        enc, D = self._make(kind)
        shp = enc.compute_output_shape((None, None))
        assert shp["pooled"] == (None, D)
        assert shp["last_hidden"] == (None, None, D)
        assert shp["penultimate"] == (None, None, D)

    def test_compute_output_shape_matches(self, kind):
        enc, D = self._make(kind)
        B, L = 3, 9
        _ = enc(_ids(B, L, CLIP_TINY["vocab_size"]))
        shp = enc.compute_output_shape((B, L))
        assert shp["pooled"] == (B, D)
        assert shp["last_hidden"] == (B, L, D)

    def test_get_config_has_args(self, kind):
        enc, _ = self._make(kind)
        cfg = enc.get_config()
        for key in ("vocab_size", "embed_dim", "num_layers", "num_heads",
                    "max_seq_len", "eps"):
            assert key in cfg
        if kind == "clip":
            assert cfg["act_fn"] == "quick_gelu"
        else:
            # OpenCLIP drops act_fn (fixed to gelu).
            assert "act_fn" not in cfg

    def test_from_config_round_trip(self, kind):
        enc, _ = self._make(kind)
        cfg = enc.get_config()
        clone = type(enc).from_config(cfg)
        assert isinstance(clone, type(enc))
        assert clone.embed_dim == enc.embed_dim
        assert clone.act_fn == enc.act_fn

    def test_keras_save_load(self, kind):
        enc, D = self._make(kind)
        B, L = 2, 12
        vocab = CLIP_TINY["vocab_size"]
        ids = keras.Input(shape=(L,), dtype="int32", name="ids")
        out = enc(ids)
        model = keras.Model(ids, out)
        x = _ids(B, L, vocab)
        y0 = model.predict(x, verbose=0)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.keras")
            model.save(path)
            reloaded = keras.models.load_model(path)
            y1 = reloaded.predict(x, verbose=0)

        for key in ("pooled", "last_hidden", "penultimate"):
            np.testing.assert_allclose(y0[key], y1[key], atol=1e-5)

    def test_variable_batch_and_length(self, kind):
        enc, D = self._make(kind)
        for B, L in [(1, 4), (3, 8), (2, 15)]:
            out = enc(_ids(B, L, CLIP_TINY["vocab_size"]))
            assert out["pooled"].shape == (B, D)
            assert out["last_hidden"].shape == (B, L, D)

    def test_invalid_embed_dim_raises(self):
        """embed_dim not divisible by num_heads must raise (H4)."""
        bad = dict(CLIP_TINY)
        bad["embed_dim"] = 30  # 30 % 4 != 0
        with pytest.raises(ValueError, match="must be divisible by num_heads"):
            CLIPTextEncoder(**bad)

    def test_causal_mask_faithfulness(self, kind):
        """Changing a LATER token must NOT change an EARLIER token's hidden state.

        Strong faithfulness check: CLIP text is causally masked, so the hidden
        state at position p depends only on positions <= p (we compare
        last_hidden BEFORE the final LN's global dependence -- actually
        last_hidden's LayerNorm is per-token, so causality still holds
        position-wise).
        """
        enc, _ = self._make(kind)
        L = 10
        ids = _ids(1, L, CLIP_TINY["vocab_size"])
        out0 = np.asarray(enc(ids)["last_hidden"])
        ids2 = ids.copy()
        # change the LAST token only
        ids2[0, -1] = (ids2[0, -1] + 7) % CLIP_TINY["vocab_size"]
        out1 = np.asarray(enc(ids2)["last_hidden"])
        # earlier positions (0 .. L-2) must be unchanged
        np.testing.assert_allclose(out0[:, : L - 1, :], out1[:, : L - 1, :],
                                   atol=1e-5)


# =====================================================================
# T5
# =====================================================================


class TestT5Encoder:

    def _make(self):
        return T5Encoder(**T5_TINY), T5_TINY["embed_dim"]

    def test_instantiation(self):
        enc, _ = self._make()
        assert isinstance(enc, T5Encoder)

    def test_forward_shape(self):
        enc, D = self._make()
        B, L = 2, 12
        out = enc(_ids(B, L, T5_TINY["vocab_size"]), attention_mask=_mask(B, L))
        assert out.shape == (B, L, D)
        assert np.all(np.isfinite(np.asarray(out)))

    def test_forward_no_mask(self):
        enc, D = self._make()
        out = enc(_ids(2, 10, T5_TINY["vocab_size"]))
        assert out.shape == (2, 10, D)

    def test_compute_output_shape_pre_build(self):
        enc, D = self._make()
        assert enc.compute_output_shape((None, None)) == (None, None, D)

    def test_compute_output_shape_matches(self):
        enc, D = self._make()
        B, L = 3, 9
        _ = enc(_ids(B, L, T5_TINY["vocab_size"]))
        assert enc.compute_output_shape((B, L)) == (B, L, D)

    def test_get_config_has_args(self):
        enc, _ = self._make()
        cfg = enc.get_config()
        for key in ("vocab_size", "embed_dim", "num_layers", "num_heads",
                    "ff_dim", "rel_attention_num_buckets",
                    "rel_attention_max_distance", "eps"):
            assert key in cfg

    def test_from_config_round_trip(self):
        enc, _ = self._make()
        clone = T5Encoder.from_config(enc.get_config())
        assert clone.embed_dim == enc.embed_dim
        assert clone.num_layers == enc.num_layers

    def test_keras_save_load(self):
        enc, D = self._make()
        B, L = 2, 12
        vocab = T5_TINY["vocab_size"]
        ids = keras.Input(shape=(L,), dtype="int32", name="ids")
        out = enc(ids)
        model = keras.Model(ids, out)
        x = _ids(B, L, vocab)
        y0 = model.predict(x, verbose=0)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t5.keras")
            model.save(path)
            reloaded = keras.models.load_model(path)
            y1 = reloaded.predict(x, verbose=0)
        np.testing.assert_allclose(y0, y1, atol=1e-5)

    def test_variable_batch_and_length(self):
        """Relative-position bias must handle different L (dynamic-L path)."""
        enc, D = self._make()
        for B, L in [(1, 4), (3, 8), (2, 15)]:
            out = enc(_ids(B, L, T5_TINY["vocab_size"]))
            assert out.shape == (B, L, D)

    def test_invalid_embed_dim_raises(self):
        """embed_dim not divisible by num_heads must raise (H4)."""
        bad = dict(T5_TINY)
        bad["embed_dim"] = 30  # 30 % 4 != 0
        with pytest.raises(ValueError, match="must be divisible by num_heads"):
            T5Encoder(**bad)

    def test_relative_position_bucket_range(self):
        """Buckets must stay within [0, num_buckets)."""
        import keras.ops as ops
        L = 16
        ctx = ops.arange(0, L, dtype="int32")[:, None]
        mem = ops.arange(0, L, dtype="int32")[None, :]
        rel = mem - ctx
        buckets = T5Encoder._relative_position_bucket(
            rel, num_buckets=T5_TINY["rel_attention_num_buckets"],
            max_distance=T5_TINY["rel_attention_max_distance"],
        )
        b = np.asarray(buckets)
        assert b.min() >= 0
        assert b.max() < T5_TINY["rel_attention_num_buckets"]
