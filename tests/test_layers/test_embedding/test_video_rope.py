import pytest
import numpy as np

import keras
from keras import ops

from dl_techniques.layers.embedding.video_rope import VideoRoPE3D


class TestVideoRoPE3DInit:
    """Constructor validation and band-split bookkeeping."""

    def test_initialization_defaults(self):
        layer = VideoRoPE3D(head_dim=24)
        assert layer.head_dim == 24
        assert layer.rope_theta == 10000.0
        # band = 2 * ((24 // 3) // 2) = 2 * 4 = 8; 3 bands cover the head fully.
        assert layer._band == 8
        assert layer._rotated_dim == 24
        assert layer._pass_dim == 0

    def test_band_split_with_remainder(self):
        # head_dim=25: band = 2 * ((25 // 3) // 2) = 2 * (8 // 2) = 8;
        # rotated_dim = 24, pass_dim = 1.
        layer = VideoRoPE3D(head_dim=25)
        assert layer._band == 8
        assert layer._rotated_dim == 24
        assert layer._pass_dim == 1

    def test_validation_errors(self):
        with pytest.raises(ValueError):
            VideoRoPE3D(head_dim=0)
        with pytest.raises(ValueError):
            VideoRoPE3D(head_dim=-4)
        with pytest.raises(ValueError):
            VideoRoPE3D(head_dim=24, rope_theta=0.0)
        with pytest.raises(ValueError):
            VideoRoPE3D(head_dim=24, rope_theta=-1.0)
        # head_dim too small for any band (band would be 0).
        with pytest.raises(ValueError):
            VideoRoPE3D(head_dim=2)

    def test_get_config_round_trip(self):
        layer = VideoRoPE3D(head_dim=24, rope_theta=5000.0)
        config = layer.get_config()
        assert config["head_dim"] == 24
        assert config["rope_theta"] == 5000.0

        rebuilt = VideoRoPE3D.from_config(config)
        assert rebuilt.head_dim == layer.head_dim
        assert rebuilt.rope_theta == layer.rope_theta
        assert rebuilt._band == layer._band


class TestVideoRoPE3DForward:
    """Shape and finiteness of the rotated q/k."""

    def test_output_shapes(self):
        layer = VideoRoPE3D(head_dim=24)
        num_frames, h_patches, w_patches = 2, 3, 3
        n = num_frames * h_patches * w_patches

        q = keras.random.normal((2, 4, n, 24))
        k = keras.random.normal((2, 4, n, 24))

        q_rot, k_rot = layer(
            q, k, num_frames=num_frames, height_patches=h_patches,
            width_patches=w_patches,
        )

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

        q_np = ops.convert_to_numpy(q_rot)
        k_np = ops.convert_to_numpy(k_rot)
        assert not np.any(np.isnan(q_np))
        assert not np.any(np.isnan(k_np))
        assert not np.any(np.isinf(q_np))
        assert not np.any(np.isinf(k_np))

    def test_pass_through_channels_are_unrotated(self):
        """With a remainder band, the trailing channels must equal the input exactly."""
        head_dim = 25  # band=8, rotated_dim=24, pass_dim=1
        layer = VideoRoPE3D(head_dim=head_dim)
        num_frames, h_patches, w_patches = 2, 2, 2
        n = num_frames * h_patches * w_patches

        q = keras.random.normal((1, 1, n, head_dim))
        k = keras.random.normal((1, 1, n, head_dim))

        q_rot, _ = layer(
            q, k, num_frames=num_frames, height_patches=h_patches,
            width_patches=w_patches,
        )

        q_np = ops.convert_to_numpy(q)
        q_rot_np = ops.convert_to_numpy(q_rot)

        np.testing.assert_allclose(
            q_rot_np[..., 24:], q_np[..., 24:], atol=1e-6, rtol=0.0
        )

    def test_rank_and_head_dim_validation(self):
        layer = VideoRoPE3D(head_dim=24)
        bad_rank = keras.random.normal((2, 12, 24))  # rank 3, not 4
        good = keras.random.normal((2, 1, 12, 24))

        with pytest.raises(ValueError):
            layer(bad_rank, good, num_frames=1, height_patches=3, width_patches=4)

        bad_head_dim = keras.random.normal((2, 1, 12, 16))
        with pytest.raises(ValueError):
            layer(good, bad_head_dim, num_frames=1, height_patches=3, width_patches=4)


class TestVideoRoPE3DAxisIndependence:
    """Delta-impulse-style probe: each axis must rotate ONLY its own band.

    This is the guard `plan.md` explicitly requires: varying only ONE grid
    coordinate (frame, height, or width) must change ONLY that axis' band of
    the rotated output, leaving the other two bands' values identical --
    proving the band-split and the position decomposition are wired to the
    correct channels, not merely producing the right SHAPE.
    """

    HEAD_DIM = 24  # band = 8 for each of the 3 axes, no remainder.
    BAND = 8

    def _rotate_fixed_vector(self, num_frames, h_patches, w_patches):
        """Rotate the SAME input vector at every grid position.

        Every token carries an identical, non-trivial input vector. Any
        difference between two output rows is therefore due ENTIRELY to the
        rotation (i.e. to the two rows' grid positions), not to different
        input content -- the delta-impulse-style control this probe needs.
        """
        layer = VideoRoPE3D(head_dim=self.HEAD_DIM)
        n = num_frames * h_patches * w_patches

        # A fixed, non-uniform vector broadcast to every one of the N tokens
        # (heads=1, batch=1) -- not one-hot, but every band carries the SAME
        # nonzero content at every grid position, so any output difference is
        # attributable purely to the rotation angle.
        rng = np.random.RandomState(0)
        base_vector = rng.normal(size=(self.HEAD_DIM,)).astype("float32")
        q_np = np.tile(base_vector, (1, 1, n, 1))
        q = keras.ops.convert_to_tensor(q_np)
        k = q  # identical; only q's rotation is inspected below.

        q_rot, _ = layer(
            q, k, num_frames=num_frames, height_patches=h_patches,
            width_patches=w_patches,
        )
        return ops.convert_to_numpy(q_rot)[0, 0]  # (N, head_dim)

    def test_varying_frame_only_moves_the_frame_band(self):
        num_frames, h_patches, w_patches = 3, 2, 2
        rotated = self._rotate_fixed_vector(num_frames, h_patches, w_patches)

        # frame=0,h=0,w=0 -> flat index 0; frame=1,h=0,w=0 -> flat index 4;
        # frame=2,h=0,w=0 -> flat index 8 (tokens_per_frame = h*w = 4).
        idx_f0 = 0
        idx_f1 = 1 * (h_patches * w_patches) + 0
        idx_f2 = 2 * (h_patches * w_patches) + 0

        band = self.BAND
        frame_band = slice(0, band)
        height_band = slice(band, 2 * band)
        width_band = slice(2 * band, 3 * band)

        # The frame band MUST differ across all three frame positions (the
        # test would be vacuous if the rotation angle were a no-op).
        assert not np.allclose(rotated[idx_f0, frame_band], rotated[idx_f1, frame_band])
        assert not np.allclose(rotated[idx_f0, frame_band], rotated[idx_f2, frame_band])

        # The height and width bands MUST be identical: same h, same w,
        # same input content -- only the frame coordinate changed.
        np.testing.assert_allclose(
            rotated[idx_f0, height_band], rotated[idx_f1, height_band],
            atol=1e-5, rtol=0.0,
        )
        np.testing.assert_allclose(
            rotated[idx_f0, width_band], rotated[idx_f1, width_band],
            atol=1e-5, rtol=0.0,
        )

    def test_varying_height_only_moves_the_height_band(self):
        num_frames, h_patches, w_patches = 2, 3, 2
        rotated = self._rotate_fixed_vector(num_frames, h_patches, w_patches)

        tokens_per_frame = h_patches * w_patches
        # frame=0 fixed; h=0,w=0 -> idx 0; h=1,w=0 -> idx w_patches; h=2,w=0 -> idx 2*w_patches
        idx_h0 = 0
        idx_h1 = 1 * w_patches
        idx_h2 = 2 * w_patches
        assert idx_h2 < tokens_per_frame  # sanity: still frame 0

        band = self.BAND
        frame_band = slice(0, band)
        height_band = slice(band, 2 * band)
        width_band = slice(2 * band, 3 * band)

        assert not np.allclose(rotated[idx_h0, height_band], rotated[idx_h1, height_band])
        assert not np.allclose(rotated[idx_h0, height_band], rotated[idx_h2, height_band])

        np.testing.assert_allclose(
            rotated[idx_h0, frame_band], rotated[idx_h1, frame_band],
            atol=1e-5, rtol=0.0,
        )
        np.testing.assert_allclose(
            rotated[idx_h0, width_band], rotated[idx_h1, width_band],
            atol=1e-5, rtol=0.0,
        )

    def test_varying_width_only_moves_the_width_band(self):
        num_frames, h_patches, w_patches = 2, 2, 3
        rotated = self._rotate_fixed_vector(num_frames, h_patches, w_patches)

        # frame=0, h=0 fixed; w=0 -> idx 0; w=1 -> idx 1; w=2 -> idx 2.
        idx_w0 = 0
        idx_w1 = 1
        idx_w2 = 2

        band = self.BAND
        frame_band = slice(0, band)
        height_band = slice(band, 2 * band)
        width_band = slice(2 * band, 3 * band)

        assert not np.allclose(rotated[idx_w0, width_band], rotated[idx_w1, width_band])
        assert not np.allclose(rotated[idx_w0, width_band], rotated[idx_w2, width_band])

        np.testing.assert_allclose(
            rotated[idx_w0, frame_band], rotated[idx_w1, frame_band],
            atol=1e-5, rtol=0.0,
        )
        np.testing.assert_allclose(
            rotated[idx_w0, height_band], rotated[idx_w1, height_band],
            atol=1e-5, rtol=0.0,
        )


class TestVideoRoPE3DRelativePositionInvariance:
    """RoPE's defining property: rotated q.k depends on relative, not absolute, position."""

    def test_dot_product_depends_only_on_relative_frame_offset(self):
        # height_patches=width_patches=1 collapses the height/width bands to
        # a single (angle-zero) grid position, isolating the frame axis.
        head_dim = 24
        layer = VideoRoPE3D(head_dim=head_dim)
        num_frames = 6

        rng = np.random.RandomState(1)
        q_vec = rng.normal(size=(head_dim,)).astype("float32")
        k_vec = rng.normal(size=(head_dim,)).astype("float32")

        def rotated_dot(frame_q: int, frame_k: int) -> float:
            n = num_frames  # height_patches=width_patches=1
            q_np = np.tile(q_vec, (1, 1, n, 1))
            k_np = np.tile(k_vec, (1, 1, n, 1))
            q = keras.ops.convert_to_tensor(q_np)
            k = keras.ops.convert_to_tensor(k_np)

            q_rot, k_rot = layer(
                q, k, num_frames=num_frames, height_patches=1, width_patches=1,
            )
            q_rot_np = ops.convert_to_numpy(q_rot)[0, 0]
            k_rot_np = ops.convert_to_numpy(k_rot)[0, 0]
            return float(np.dot(q_rot_np[frame_q], k_rot_np[frame_k]))

        # Same relative offset (+2), two different absolute position pairs.
        dot_a = rotated_dot(frame_q=0, frame_k=2)
        dot_b = rotated_dot(frame_q=3, frame_k=5)

        np.testing.assert_allclose(dot_a, dot_b, atol=1e-4, rtol=0.0)

        # Control: a DIFFERENT relative offset must generally give a
        # different dot product (proves the probe is not vacuously satisfied
        # by e.g. an identity rotation).
        dot_c = rotated_dot(frame_q=0, frame_k=1)
        assert not np.isclose(dot_a, dot_c, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
