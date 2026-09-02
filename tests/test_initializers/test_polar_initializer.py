"""Tests for PolarInitializer (exact per-vector norm, uniform direction).

Several tests here are single-claim guards for defects found by review and
verified by measurement against the previous implementation:

* a scalar ``axis`` could not express He fan-in for a conv kernel, and the
  ``axis=0`` default was catastrophically wrong there. Measured on
  ``(3, 3, 64, 128)`` (fan_in 576): per-element std 0.8165 against He's 0.0589
  (13.9x) and a per-output-unit fan-in energy of 384.0 against a target of 2.0
  (192x), which compounds to ~2.6e11 over ten layers.
* the draw went through ``np.random.default_rng(None)``, which bypasses
  ``keras.utils.set_random_seed`` entirely: two builds under the same global
  seed measured as different tensors.
* ``gain`` was unvalidated -- ``norm=1.0, gain=-2.0`` realized a norm of 2.0
  (i.e. ``|gain| * norm``) -- and ``norm=0`` was accepted, giving an all-zero
  layer with no symmetry breaking.
* both internal buffers were hard-coded float32, so a ``dtype='float64'``
  request came back float32-quantized: exactness 1.1e-07 instead of 4.4e-16.
"""

import os
import tempfile

import keras
import numpy as np
import pytest

from dl_techniques.initializers import PolarInitializer
from dl_techniques.initializers.polar_initializer import HE_EQUIVALENT_NORM


def _np(tensor) -> np.ndarray:
    """Backend tensor to numpy."""
    return keras.ops.convert_to_numpy(tensor)


class TestPolarInitializer:
    """Exact-norm, direction and validation behaviour of PolarInitializer."""

    # ------------------------------------------------------------------
    # The core guarantee
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "shape,axis",
        [((64, 32), 0), ((48, 16), 0), ((10, 20), 1), ((7, 5, 8), 2),
         ((3, 3, 8, 16), (0, 1, 2)), ((4, 6, 5, 3), (0, 2))],
    )
    def test_exact_norm(self, shape, axis):
        """Every vector over `axis` has L2 norm exactly `norm`."""
        init = PolarInitializer(norm=2.5, axis=axis, seed=1)
        w = _np(init(shape))
        norms = np.sqrt(np.sum(np.square(w), axis=axis))
        np.testing.assert_allclose(norms, 2.5, rtol=1e-5, atol=1e-5)

    def test_auto_norm_matches_he_energy(self):
        """norm=None targets sqrt(2), the He-normal weight-vector energy."""
        w = _np(PolarInitializer(seed=0)((200, 8)))
        np.testing.assert_allclose(
            np.linalg.norm(w, axis=0), np.sqrt(2.0), rtol=1e-5
        )
        assert HE_EQUIVALENT_NORM == pytest.approx(np.sqrt(2.0))

    def test_gain(self):
        """gain multiplies the target norm."""
        w = _np(PolarInitializer(norm=1.0, gain=3.0, seed=0)((32, 8)))
        np.testing.assert_allclose(np.linalg.norm(w, axis=0), 3.0, rtol=1e-5)

    def test_gain_multiplies_the_he_default_too(self):
        """gain scales sqrt(2), so gain=2 with norm=None targets 2*sqrt(2)."""
        w = _np(PolarInitializer(gain=2.0, seed=0)((64, 8)))
        np.testing.assert_allclose(
            np.linalg.norm(w, axis=0), 2.0 * np.sqrt(2.0), rtol=1e-5
        )

    def test_the_direction_is_uniform_on_the_sphere(self):
        """The claim the PolarQuant framing rests on, measured rather than asserted.

        For a uniform direction on S^(n-1) every coordinate has mean 0 and
        E[coord^2] = 1/n. Nothing previously tested the direction at all.
        """
        n = 3
        w = _np(PolarInitializer(norm=1.0, seed=5)((n, 20000)))

        np.testing.assert_allclose(w.mean(axis=1), 0.0, atol=0.02)
        np.testing.assert_allclose((w ** 2).mean(axis=1), 1.0 / n, atol=0.01)

    # ------------------------------------------------------------------
    # He equivalence across ranks -- the C-1 guard
    # ------------------------------------------------------------------

    def test_the_default_axis_is_the_fan_in_block(self):
        """axis=None reduces over every axis except the last, at any rank."""
        init = PolarInitializer(seed=0)

        assert init._resolve_axes((16, 8)) == (0,)
        assert init._resolve_axes((3, 3, 64, 128)) == (0, 1, 2)
        assert init._resolve_axes((5, 4, 3, 2, 1)) == (0, 1, 2, 3)

    @pytest.mark.parametrize("shape", [(576, 128), (3, 3, 64, 128), (5, 7, 16, 32)])
    def test_the_default_gives_every_output_unit_he_energy(self, shape):
        """Each output unit's fan-in energy is exactly 2, at every rank.

        Guard for the measured defect: with the old scalar axis=0 default a
        (3, 3, 64, 128) kernel gave each output unit an energy of 384.0.
        """
        w = _np(PolarInitializer(seed=0)(shape))
        fan_in_axes = tuple(range(len(shape) - 1))

        energy = np.sum(np.square(w), axis=fan_in_axes)
        np.testing.assert_allclose(energy, 2.0, rtol=1e-4)

    @pytest.mark.parametrize("shape", [(576, 128), (3, 3, 64, 128)])
    def test_the_per_element_std_matches_he_normal(self, shape):
        """The bank sits at He's sqrt(2/fan_in), not 13.9x above it."""
        w = _np(PolarInitializer(seed=0)(shape))
        fan_in = int(np.prod(shape[:-1]))

        assert w.std() == pytest.approx(np.sqrt(2.0 / fan_in), rel=0.02)

    def test_a_conv_stack_does_not_blow_up(self):
        """Ten conv layers keep the activation scale bounded.

        With the old default the per-layer std gain measured 13.9x, i.e. ~2.6e11
        over ten layers. He's own gain is sqrt(2) per linear pass (it is sized
        for a following ReLU), so the bound below is loose on purpose while
        still being ~7 orders of magnitude tighter than the defect.
        """
        rng = np.random.default_rng(0)
        x = keras.ops.convert_to_tensor(
            rng.normal(size=(8, 32, 32, 16)).astype("float32")
        )
        scale = 1.0
        for _ in range(10):
            kernel = PolarInitializer(seed=0)((3, 3, 16, 16))
            y = keras.ops.conv(x, kernel, strides=1, padding="same")
            scale *= float(_np(keras.ops.std(y)) / _np(keras.ops.std(x)))

        assert scale < 100.0, f"activation scale exploded by {scale:.3g}"

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------

    def test_reproducible_with_seed(self):
        """Two instances built with the same seed agree bit for bit."""
        a = _np(PolarInitializer(norm=1.0, seed=7)((32, 16)))
        b = _np(PolarInitializer(norm=1.0, seed=7)((32, 16)))
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        a = _np(PolarInitializer(seed=1)((32, 16)))
        b = _np(PolarInitializer(seed=2)((32, 16)))
        assert not np.allclose(a, b)

    def test_a_seedless_instance_honours_the_global_seed(self):
        """keras.utils.set_random_seed controls a seedless instance.

        The previous np.random.default_rng(None) drew from OS entropy and
        ignored the global seed, so a model built entirely from this initializer
        was irreproducible even after set_random_seed.
        """
        keras.utils.set_random_seed(1234)
        a = _np(PolarInitializer(norm=1.0)((16, 8)))
        keras.utils.set_random_seed(1234)
        b = _np(PolarInitializer(norm=1.0)((16, 8)))

        np.testing.assert_array_equal(a, b)

        # Anti-vacuity: a different global seed must give a different draw,
        # otherwise the assertion above would pass on a constant tensor.
        keras.utils.set_random_seed(4321)
        c = _np(PolarInitializer(norm=1.0)((16, 8)))
        assert not np.allclose(a, c)

    def test_an_instance_replays_like_every_keras_initializer(self):
        """One INSTANCE emits the same tensor at a matching shape, seeded or not.

        This is the Keras contract, not a defect: keras.initializers.RandomNormal
        behaves identically with and without a seed. The remedy when two weights
        must differ is dl_techniques.initializers.clone_initializer.
        """
        seeded = PolarInitializer(norm=1.0, seed=3)
        np.testing.assert_array_equal(_np(seeded((8, 4))), _np(seeded((8, 4))))

        reference = keras.initializers.RandomNormal(seed=3)
        np.testing.assert_array_equal(
            _np(reference((8, 4))), _np(reference((8, 4)))
        )

        seedless = PolarInitializer(norm=1.0)
        np.testing.assert_array_equal(_np(seedless((8, 4))), _np(seedless((8, 4))))

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def test_invalid_norm(self):
        with pytest.raises(ValueError, match="norm must be positive"):
            PolarInitializer(norm=-1.0)

    def test_zero_norm_is_rejected(self):
        """norm=0 was accepted and produced an all-zero, gradient-dead layer."""
        with pytest.raises(ValueError, match="dead layer"):
            PolarInitializer(norm=0.0)

    @pytest.mark.parametrize("gain", [0.0, -1.0, -2.0])
    def test_non_positive_gain_is_rejected(self, gain):
        """A negative gain realized |gain| * norm, breaking the stated contract."""
        with pytest.raises(ValueError, match="gain must be positive"):
            PolarInitializer(gain=gain)

    def test_invalid_axis(self):
        with pytest.raises(ValueError, match="out of range"):
            PolarInitializer(axis=5)((4, 4))

    def test_negative_axis_is_wrapped(self):
        """axis=-1 addresses the last axis, and is not the same as axis=0."""
        w = _np(PolarInitializer(norm=1.0, axis=-1, seed=0)((16, 8)))
        np.testing.assert_allclose(np.linalg.norm(w, axis=1), 1.0, rtol=1e-5)

    @pytest.mark.parametrize("axis", [(0, 0), (0, -2)])
    def test_duplicate_axes_are_rejected(self, axis):
        """A repeated axis (before or after wrapping) is an error, not a silent no-op."""
        with pytest.raises(ValueError, match="repeat|duplicate"):
            PolarInitializer(axis=axis)((4, 4))

    def test_empty_axis_sequence_is_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            PolarInitializer(axis=())

    def test_rank_one_needs_an_explicit_axis(self):
        """A rank-1 tensor has no fan-out axis, so the default is undefined there."""
        with pytest.raises(ValueError, match="rank-1"):
            PolarInitializer()((32,))

        # ... but an explicit axis gives the whole tensor the target norm.
        w = _np(PolarInitializer(norm=1.0, axis=0, seed=0)((32,)))
        assert np.linalg.norm(w) == pytest.approx(1.0, rel=1e-5)

    def test_a_length_one_vector_collapses_to_a_sign(self):
        """S^0 is two points, so every entry is +/- target. Documented, not a bug."""
        w = _np(PolarInitializer(norm=1.0, seed=0)((1, 8)))
        np.testing.assert_allclose(np.abs(w), 1.0, rtol=1e-5)

    # ------------------------------------------------------------------
    # dtype
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_dtype_is_honored(self, dtype):
        w = PolarInitializer(norm=1.0, seed=0)((16, 8), dtype=dtype)
        assert keras.backend.standardize_dtype(w.dtype) == dtype

    def test_float64_is_exact_to_float64(self):
        """The norm is set in the requested dtype, not float32 then widened.

        Measured on the previous implementation, which hard-coded a float32
        buffer: a float64 request was exact only to 1.1e-07.
        """
        w = _np(PolarInitializer(norm=1.0, seed=0)((512, 8), dtype="float64"))
        assert w.dtype == np.float64
        assert np.abs(np.linalg.norm(w, axis=0) - 1.0).max() < 1e-12

    def test_dtype_none_follows_floatx(self):
        original = keras.config.floatx()
        try:
            for floatx in ("float32", "float64"):
                keras.config.set_floatx(floatx)
                w = PolarInitializer(norm=1.0, seed=0)((16, 8), dtype=None)
                assert keras.backend.standardize_dtype(w.dtype) == floatx
        finally:
            keras.config.set_floatx(original)

    def test_a_half_precision_request_is_normalized_in_float32(self):
        """float16 output, but the norm is set before the cast.

        Normalizing in float16 would leave the guarantee at ~4.5e-05; setting it
        in float32 and casting keeps the error at the float16 representation
        limit rather than compounding it.
        """
        w = PolarInitializer(norm=1.0, seed=0)((512, 8), dtype="float16")
        assert keras.backend.standardize_dtype(w.dtype) == "float16"

        norms = np.linalg.norm(_np(w).astype(np.float64), axis=0)
        assert np.abs(norms - 1.0).max() < 1e-3

    def test_call_accepts_extra_kwargs(self):
        """The Keras call path may pass arguments this initializer ignores."""
        w = PolarInitializer(norm=1.0, seed=0)((4, 4), None, partition_shape=None)
        assert tuple(w.shape) == (4, 4)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def test_serialization_roundtrip(self):
        init = PolarInitializer(norm=1.5, axis=(0, 1), gain=2.0, seed=9)
        cfg = init.get_config()
        restored = PolarInitializer.from_config(cfg)

        assert restored.norm == 1.5
        assert restored.axis == (0, 1)
        assert restored.gain == 2.0
        assert restored.seed == 9
        np.testing.assert_allclose(
            _np(init((16, 8, 4))), _np(restored((16, 8, 4))), atol=1e-7, rtol=0
        )

    def test_a_none_norm_roundtrips(self):
        """The He-equivalent default survives a config round trip."""
        init = PolarInitializer(seed=4)
        restored = PolarInitializer.from_config(init.get_config())

        assert restored.norm is None
        assert restored.axis is None
        np.testing.assert_array_equal(_np(init((32, 8))), _np(restored((32, 8))))

    def test_a_legacy_scalar_axis_config_still_loads(self):
        """Configs written when `axis` was a bare int must still deserialize."""
        restored = PolarInitializer.from_config(
            {"norm": 1.0, "axis": 0, "gain": 1.0, "seed": 9}
        )
        assert restored.axis == (0,)
        np.testing.assert_allclose(
            np.linalg.norm(_np(restored((16, 8))), axis=0), 1.0, rtol=1e-5
        )

    def test_the_config_keeps_the_seed_the_caller_passed(self):
        """A seedless initializer stays seedless across a round trip (Keras convention)."""
        assert PolarInitializer().get_config()["seed"] is None
        assert PolarInitializer(seed=11).get_config()["seed"] == 11

    def test_keras_serialize_deserialize(self):
        init = PolarInitializer(norm=1.0, seed=3)
        restored = keras.initializers.deserialize(keras.initializers.serialize(init))

        assert isinstance(restored, PolarInitializer)
        np.testing.assert_allclose(
            _np(init((8, 8))), _np(restored((8, 8))), atol=1e-7, rtol=0
        )

    def test_repr_names_the_caller_seed(self):
        assert "seed=None" in repr(PolarInitializer())
        assert "seed=5" in repr(PolarInitializer(seed=5))

    # ------------------------------------------------------------------
    # Integration
    # ------------------------------------------------------------------

    def test_use_in_dense_save_load(self):
        inputs = keras.Input(shape=(16,))
        out = keras.layers.Dense(
            8, kernel_initializer=PolarInitializer(norm=1.0, seed=2)
        )(inputs)
        model = keras.Model(inputs, out)
        x = np.random.randn(3, 16).astype("float32")
        before = _np(model(x))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
            after = _np(loaded(x))
        np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-7)

    def test_use_in_conv2d_save_load(self):
        """The rank-4 path the previous suite never exercised."""
        inputs = keras.Input(shape=(8, 8, 3))
        out = keras.layers.Conv2D(
            4, 3, padding="same",
            kernel_initializer=PolarInitializer(seed=2),
        )(inputs)
        model = keras.Model(inputs, out)

        kernel = _np(model.layers[1].kernel)
        np.testing.assert_allclose(
            np.sum(np.square(kernel), axis=(0, 1, 2)), 2.0, rtol=1e-4
        )

        x = np.random.randn(2, 8, 8, 3).astype("float32")
        before = _np(model(x))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.keras")
            model.save(path)
            after = _np(keras.models.load_model(path)(x))
        np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
