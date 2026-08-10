"""Tests for IdentityPlusNoise (near-identity square coupling-matrix init).

Contract under test (read off `initializers/identity_plus_noise.py`, not
assumed): the initializer accepts ONLY a square rank-2 shape and returns
``eye(H) + normal(0, stddev, seed)``; ``stddev == 0`` short-circuits to the
exact identity; ``seed=None`` draws from the global Keras RNG (so successive
calls differ) while an explicit int makes the draw stateless and repeatable.
"""

import os
import tempfile

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.initializers import IdentityPlusNoise


def _np(x):
    return ops.convert_to_numpy(x)


class TestConstruction:
    def test_defaults(self):
        init = IdentityPlusNoise()
        assert init.stddev == 0.01
        assert init.seed is None

    def test_stddev_is_coerced_to_float(self):
        """An int stddev must not survive as an int -- `stddev == 0.0` is the
        exact-identity short circuit and is compared against a float."""
        init = IdentityPlusNoise(stddev=1, seed=0)
        assert isinstance(init.stddev, float)
        assert init.stddev == 1.0

    def test_seed_is_retained_verbatim(self):
        assert IdentityPlusNoise(stddev=0.02, seed=1234).seed == 1234


class TestShapeContract:
    def test_square_shape_is_preserved(self):
        w = _np(IdentityPlusNoise(stddev=0.01, seed=0)((16, 16)))
        assert w.shape == (16, 16)

    def test_non_square_2d_rejected(self):
        with pytest.raises(ValueError, match="square"):
            IdentityPlusNoise(stddev=0.01, seed=0)((3, 4))

    def test_rank_1_rejected(self):
        with pytest.raises(ValueError, match="square"):
            IdentityPlusNoise(stddev=0.01, seed=0)((8,))

    def test_rank_3_rejected_even_when_trailing_dims_are_square(self):
        """A batched (2, 3, 3) request is NOT accepted -- the guard is on
        `len(shape) != 2`, so this initializer cannot be used for a conv
        kernel or any leading-dim weight."""
        with pytest.raises(ValueError, match="square"):
            IdentityPlusNoise(stddev=0.01, seed=0)((2, 3, 3))

    def test_error_message_names_the_public_class(self):
        """Regression: the message used to say `_IdentityPlusNoise`, the old
        private name, which no longer exists anywhere in the repo."""
        with pytest.raises(ValueError) as exc:
            IdentityPlusNoise()((3, 4))
        assert "_IdentityPlusNoise" not in str(exc.value)
        assert "IdentityPlusNoise" in str(exc.value)


class TestStructure:
    def test_zero_stddev_is_the_exact_identity(self):
        w = _np(IdentityPlusNoise(stddev=0.0, seed=7)((5, 5)))
        np.testing.assert_array_equal(w, np.eye(5, dtype="float32"))

    def test_diagonal_is_near_one_and_offdiagonal_near_zero(self):
        w = _np(IdentityPlusNoise(stddev=0.01, seed=7)((32, 32)))
        diag = np.diag(w)
        off = w[~np.eye(32, dtype=bool)]
        assert abs(diag.mean() - 1.0) < 0.01
        assert abs(off.mean()) < 0.01
        # It is genuinely eye + noise, not exactly eye.
        assert not np.array_equal(w, np.eye(32, dtype="float32"))

    def test_residual_matches_the_requested_stddev(self):
        stddev = 0.05
        w = _np(IdentityPlusNoise(stddev=stddev, seed=3)((256, 256)))
        residual = w - np.eye(256, dtype="float32")
        assert abs(residual.std() - stddev) < 0.1 * stddev
        assert abs(residual.mean()) < 0.01 * stddev


class TestDeterminism:
    def test_same_seed_gives_identical_draws(self):
        a = _np(IdentityPlusNoise(stddev=0.01, seed=5)((8, 8)))
        b = _np(IdentityPlusNoise(stddev=0.01, seed=5)((8, 8)))
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_give_different_draws(self):
        a = _np(IdentityPlusNoise(stddev=0.01, seed=5)((8, 8)))
        b = _np(IdentityPlusNoise(stddev=0.01, seed=6)((8, 8)))
        assert not np.allclose(a, b)

    def test_seed_none_is_stateful_and_does_not_repeat(self):
        """Documented behaviour, asserted so a silent switch to a fixed
        default seed would be caught: `seed=None` draws from the global RNG,
        so two successive calls differ."""
        init = IdentityPlusNoise(stddev=0.01, seed=None)
        assert not np.allclose(_np(init((8, 8))), _np(init((8, 8))))


class TestDtype:
    def test_default_dtype_is_float32(self):
        assert _np(IdentityPlusNoise(stddev=0.01, seed=0)((4, 4))).dtype == np.float32

    def test_explicit_dtype_is_honoured(self):
        w = _np(IdentityPlusNoise(stddev=0.01, seed=0)((4, 4), dtype="float64"))
        assert w.dtype == np.float64

    def test_explicit_dtype_is_honoured_on_the_zero_stddev_path(self):
        """The `stddev == 0` short circuit returns `eye` directly and must
        not silently drop back to float32."""
        w = _np(IdentityPlusNoise(stddev=0.0, seed=0)((4, 4), dtype="float64"))
        assert w.dtype == np.float64


class TestSerialization:
    def test_get_config_keys(self):
        cfg = IdentityPlusNoise(stddev=0.03, seed=11).get_config()
        assert cfg == {"stddev": 0.03, "seed": 11}

    def test_get_config_roundtrip_reproduces_the_draw(self):
        init = IdentityPlusNoise(stddev=0.03, seed=11)
        restored = IdentityPlusNoise.from_config(init.get_config())
        assert restored.stddev == 0.03
        assert restored.seed == 11
        np.testing.assert_array_equal(_np(init((8, 8))), _np(restored((8, 8))))

    def test_registered_name_resolves(self):
        """The `custom_objects` key Keras actually looks a CLASS up by is its
        registered name, not its bare class name (see D-014)."""
        name = keras.saving.get_registered_name(IdentityPlusNoise)
        assert name == "Custom>IdentityPlusNoise"
        assert keras.saving.get_registered_object(name) is IdentityPlusNoise

    def test_serialized_object_carries_the_registered_name(self):
        blob = keras.saving.serialize_keras_object(
            IdentityPlusNoise(stddev=0.02, seed=3)
        )
        assert blob["registered_name"] == keras.saving.get_registered_name(
            IdentityPlusNoise
        )
        assert blob["config"] == {"stddev": 0.02, "seed": 3}

    def test_keras_serialize_deserialize(self):
        init = IdentityPlusNoise(stddev=0.02, seed=3)
        restored = keras.saving.deserialize_keras_object(
            keras.saving.serialize_keras_object(init)
        )
        assert isinstance(restored, IdentityPlusNoise)
        assert restored.stddev == 0.02
        assert restored.seed == 3
        np.testing.assert_array_equal(_np(init((8, 8))), _np(restored((8, 8))))


class TestUseInLayer:
    def test_dense_kernel_save_load_preserves_values(self):
        """A square Dense kernel initialized by IdentityPlusNoise survives a
        `.keras` round trip with its VALUES intact (not merely its shape).

        `training=False` is passed explicitly on both calls -- `training=None`
        is not inference in this repo.
        """
        inputs = keras.Input(shape=(8,))
        out = keras.layers.Dense(
            8,
            use_bias=False,
            kernel_initializer=IdentityPlusNoise(stddev=0.05, seed=17),
        )(inputs)
        model = keras.Model(inputs, out)

        x = np.random.RandomState(0).randn(4, 8).astype("float32")
        before = _np(model(x, training=False))
        kernel_before = model.layers[1].get_weights()[0]

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.keras")
            model.save(path)
            loaded = keras.models.load_model(path)

        after = _np(loaded(x, training=False))
        kernel_after = loaded.layers[1].get_weights()[0]

        np.testing.assert_allclose(kernel_before, kernel_after, rtol=0, atol=0)
        np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-7)
        # And the restored kernel is the trained/initialized one, not a fresh
        # draw that happens to have the right shape.
        assert not np.array_equal(kernel_after, np.eye(8, dtype="float32"))

    def test_loaded_initializer_config_survives(self):
        inputs = keras.Input(shape=(8,))
        out = keras.layers.Dense(
            8, kernel_initializer=IdentityPlusNoise(stddev=0.05, seed=17)
        )(inputs)
        model = keras.Model(inputs, out)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
        init = loaded.layers[1].kernel_initializer
        assert isinstance(init, IdentityPlusNoise)
        assert init.stddev == 0.05
        assert init.seed == 17
