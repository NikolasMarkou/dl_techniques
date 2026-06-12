"""Tests for the 3D multi-axis mRoPE layer (`Ideogram4MRoPE`).

The load-bearing gate is `test_matches_numpy_reference_*`: the Keras layer's
cos/sin tables must match a numpy reproduction of the PyTorch forward
(band interleave included) element-wise at atol=1e-6.
"""

import os
import tempfile

import keras
import numpy as np
import pytest

from dl_techniques.layers.embedding.multi_axis_rope import (
    Ideogram4MRoPE,
    apply_rotary_pos_emb,
    _rotate_half,
)


# ---------------------------------------------------------------------
# Numpy reference reproductions of the PyTorch math.
# ---------------------------------------------------------------------


def numpy_mrope_reference(position_ids, head_dim, base, mrope_section):
    """Reproduce PyTorch `Ideogram4MRoPE.forward` exactly, in numpy.

    position_ids: (B, L, 3) int array of (t, h, w).
    Returns (cos, sin), each (B, L, head_dim).
    """
    position_ids = np.asarray(position_ids)
    B, L, _ = position_ids.shape

    inv_freq = 1.0 / (
        base ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim)
    )  # (head_dim/2,)

    # (3, B, L) permute of (B, L, 3).
    pos = np.transpose(position_ids.astype(np.float32), (2, 0, 1))  # (3, B, L)
    # freqs[axis, b, l, f] = inv_freq[f] * pos[axis, b, l]
    # (3, B, head_dim/2, 1) @ (3, B, 1, L) -> (3, B, head_dim/2, L) -> transpose
    freqs = np.einsum("f,abl->ablf", inv_freq, pos)  # (3, B, L, head_dim/2)

    freqs_t = freqs[0].copy()  # (B, L, head_dim/2)  -- t axis base
    for axis, offset in ((1, 1), (2, 2)):
        length = mrope_section[axis] * 3
        idx = np.arange(offset, length, 3)
        freqs_t[..., idx] = freqs[axis][..., idx]

    emb = np.concatenate([freqs_t, freqs_t], axis=-1)  # (B, L, head_dim)
    return np.cos(emb), np.sin(emb)


def numpy_rotate_half(x):
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return np.concatenate([-x2, x1], axis=-1)


def numpy_apply_rotary(q, k, cos, sin):
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    q_embed = q * cos + numpy_rotate_half(q) * sin
    k_embed = k * cos + numpy_rotate_half(k) * sin
    return q_embed, k_embed


# ---------------------------------------------------------------------


class TestIdeogram4MRoPE:
    """Comprehensive test suite for the Ideogram4MRoPE layer."""

    # --- construction / validation ---------------------------------

    def test_init_stores_config(self):
        layer = Ideogram4MRoPE(head_dim=32, rope_theta=10000.0, mrope_section=(3, 2, 2))
        assert layer.head_dim == 32
        assert layer.rope_theta == 10000.0
        assert layer.mrope_section == (3, 2, 2)

    def test_invalid_head_dim_nonpositive(self):
        with pytest.raises(ValueError, match="head_dim must be a positive integer"):
            Ideogram4MRoPE(head_dim=0, rope_theta=1e4, mrope_section=(2, 1, 1))

    def test_invalid_head_dim_odd(self):
        with pytest.raises(ValueError, match="head_dim must be even"):
            Ideogram4MRoPE(head_dim=31, rope_theta=1e4, mrope_section=(2, 1, 1))

    def test_invalid_rope_theta(self):
        with pytest.raises(ValueError, match="rope_theta must be positive"):
            Ideogram4MRoPE(head_dim=32, rope_theta=0.0, mrope_section=(2, 1, 1))

    def test_invalid_mrope_section_length(self):
        with pytest.raises(ValueError, match="mrope_section must have length 3"):
            Ideogram4MRoPE(head_dim=32, rope_theta=1e4, mrope_section=(2, 1))

    def test_invalid_mrope_section_nonpositive(self):
        with pytest.raises(ValueError, match="entries must be positive"):
            Ideogram4MRoPE(head_dim=32, rope_theta=1e4, mrope_section=(2, 0, 1))

    def test_invalid_mrope_section_out_of_bounds(self):
        # head_dim=16 -> half=8. h band of 4 reaches slot arange(1,12,3)=[1,4,7,10],
        # max 10 >= 8 -> error.
        with pytest.raises(ValueError, match="exceeds"):
            Ideogram4MRoPE(head_dim=16, rope_theta=1e4, mrope_section=(2, 4, 1))

    # --- the load-bearing numpy-reference gate ---------------------

    @pytest.mark.parametrize(
        "head_dim,mrope_section,base",
        [
            (16, (2, 1, 1), 10000.0),
            (32, (3, 2, 2), 10000.0),
            (32, (3, 2, 2), 5_000_000.0),
        ],
    )
    def test_matches_numpy_reference_small(self, head_dim, mrope_section, base):
        rng = np.random.default_rng(0)
        B, L = 4, 11
        position_ids = rng.integers(0, 50, size=(B, L, 3)).astype(np.int32)

        layer = Ideogram4MRoPE(
            head_dim=head_dim, rope_theta=base, mrope_section=mrope_section
        )
        cos_k, sin_k = layer(keras.ops.convert_to_tensor(position_ids))
        cos_k = keras.ops.convert_to_numpy(cos_k)
        sin_k = keras.ops.convert_to_numpy(sin_k)

        cos_ref, sin_ref = numpy_mrope_reference(
            position_ids, head_dim, base, mrope_section
        )

        assert cos_k.shape == cos_ref.shape == (B, L, head_dim)
        np.testing.assert_allclose(cos_k, cos_ref, atol=1e-6)
        np.testing.assert_allclose(sin_k, sin_ref, atol=1e-6)

    def test_matches_numpy_reference_real_config(self):
        """Full Ideogram4 config: head_dim=256, mrope_section=(24,20,20)."""
        head_dim = 256
        base = 5_000_000.0
        mrope_section = (24, 20, 20)
        rng = np.random.default_rng(7)
        B, L = 2, 9
        position_ids = rng.integers(0, 128, size=(B, L, 3)).astype(np.int32)

        layer = Ideogram4MRoPE(
            head_dim=head_dim, rope_theta=base, mrope_section=mrope_section
        )
        cos_k, sin_k = layer(keras.ops.convert_to_tensor(position_ids))
        cos_k = keras.ops.convert_to_numpy(cos_k)
        sin_k = keras.ops.convert_to_numpy(sin_k)

        cos_ref, sin_ref = numpy_mrope_reference(
            position_ids, head_dim, base, mrope_section
        )

        assert cos_k.shape == (B, L, head_dim)
        np.testing.assert_allclose(cos_k, cos_ref, atol=1e-6)
        np.testing.assert_allclose(sin_k, sin_ref, atol=1e-6)

    def test_interleave_actually_uses_distinct_axes(self):
        """Sanity: distinct t/h/w positions must produce distinct slot values,
        proving the band interleave is wired (not just collapsing to t)."""
        head_dim = 32
        mrope_section = (3, 2, 2)
        # Make each axis a different constant so axis-sourced slots differ.
        position_ids = np.zeros((1, 1, 3), dtype=np.int32)
        position_ids[..., 0] = 1   # t
        position_ids[..., 1] = 5   # h
        position_ids[..., 2] = 9   # w
        layer = Ideogram4MRoPE(
            head_dim=head_dim, rope_theta=10000.0, mrope_section=mrope_section
        )
        cos_k, _ = layer(keras.ops.convert_to_tensor(position_ids))
        cos_ref, _ = numpy_mrope_reference(
            position_ids, head_dim, 10000.0, mrope_section
        )
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(cos_k), cos_ref, atol=1e-6
        )
        # h slots (arange(1,6,3)=[1,4]) and w slots (arange(2,6,3)=[2,5]) must
        # differ from a pure-t reference at those slots.
        cos_t_only, _ = numpy_mrope_reference(
            np.broadcast_to(position_ids[..., :1], (1, 1, 3)).copy(),
            head_dim, 10000.0, mrope_section,
        )
        assert not np.allclose(cos_ref[..., 1], cos_t_only[..., 1])
        assert not np.allclose(cos_ref[..., 2], cos_t_only[..., 2])

    # --- apply_rotary_pos_emb helper -------------------------------

    def test_rotate_half_matches_numpy(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal((2, 3, 4, 8)).astype(np.float32)
        out = keras.ops.convert_to_numpy(_rotate_half(keras.ops.convert_to_tensor(x)))
        np.testing.assert_allclose(out, numpy_rotate_half(x), atol=1e-6)

    def test_apply_rotary_pos_emb_matches_numpy(self):
        rng = np.random.default_rng(2)
        B, H, L, D = 2, 4, 7, 16
        q = rng.standard_normal((B, H, L, D)).astype(np.float32)
        k = rng.standard_normal((B, H, L, D)).astype(np.float32)
        cos = rng.standard_normal((B, L, D)).astype(np.float32)
        sin = rng.standard_normal((B, L, D)).astype(np.float32)

        q_e, k_e = apply_rotary_pos_emb(
            keras.ops.convert_to_tensor(q),
            keras.ops.convert_to_tensor(k),
            keras.ops.convert_to_tensor(cos),
            keras.ops.convert_to_tensor(sin),
        )
        q_ref, k_ref = numpy_apply_rotary(q, k, cos, sin)
        np.testing.assert_allclose(keras.ops.convert_to_numpy(q_e), q_ref, atol=1e-6)
        np.testing.assert_allclose(keras.ops.convert_to_numpy(k_e), k_ref, atol=1e-6)

    # --- shapes / config -------------------------------------------

    def test_compute_output_shape(self):
        layer = Ideogram4MRoPE(head_dim=64, rope_theta=1e4, mrope_section=(3, 2, 2))
        cos_shape, sin_shape = layer.compute_output_shape((None, 10, 3))
        assert cos_shape == (None, 10, 64)
        assert sin_shape == (None, 10, 64)

    def test_get_config_round_trip_in_memory(self):
        layer = Ideogram4MRoPE(head_dim=32, rope_theta=5e6, mrope_section=(3, 2, 2))
        cfg = layer.get_config()
        rebuilt = Ideogram4MRoPE.from_config(cfg)
        assert rebuilt.head_dim == 32
        assert rebuilt.rope_theta == 5e6
        assert rebuilt.mrope_section == (3, 2, 2)

    # --- .keras serialization round-trip (inv_freq must restore) ---

    def test_keras_serialization_round_trip(self):
        head_dim = 32
        mrope_section = (3, 2, 2)
        base = 5_000_000.0
        rng = np.random.default_rng(3)
        B, L = 2, 6
        position_ids = rng.integers(0, 40, size=(B, L, 3)).astype(np.int32)

        inputs = keras.Input(shape=(L, 3), dtype="int32")
        cos, sin = Ideogram4MRoPE(
            head_dim=head_dim, rope_theta=base, mrope_section=mrope_section
        )(inputs)
        model = keras.Model(inputs, [cos, sin])

        cos_before, sin_before = model(keras.ops.convert_to_tensor(position_ids))
        cos_before = keras.ops.convert_to_numpy(cos_before)
        sin_before = keras.ops.convert_to_numpy(sin_before)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mrope.keras")
            model.save(path)
            reloaded = keras.models.load_model(path)

        cos_after, sin_after = reloaded(keras.ops.convert_to_tensor(position_ids))
        cos_after = keras.ops.convert_to_numpy(cos_after)
        sin_after = keras.ops.convert_to_numpy(sin_after)

        np.testing.assert_allclose(cos_before, cos_after, atol=1e-6)
        np.testing.assert_allclose(sin_before, sin_after, atol=1e-6)

        # And the reloaded output still matches the numpy reference.
        cos_ref, sin_ref = numpy_mrope_reference(
            position_ids, head_dim, base, mrope_section
        )
        np.testing.assert_allclose(cos_after, cos_ref, atol=1e-6)
        np.testing.assert_allclose(sin_after, sin_ref, atol=1e-6)
