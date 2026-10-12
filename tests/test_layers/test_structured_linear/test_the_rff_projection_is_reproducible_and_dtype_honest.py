"""RFFKernelLayer: seeded reproducibility and dtype-honest initializers.

Two defects, both fixed together: `build()` closed over eager float32 draws
from `keras.random.*(seed=None)` -- which resolve to the stateful global
SeedGenerator that `keras.utils.set_random_seed` does not pin, and for which
the layer offered no `seed` knob -- and the wrapping lambdas ignored the
requested `dtype`, crashing under mixed_float16.
"""

import keras
import numpy as np

from dl_techniques.layers.structured_linear.random_fourier_features import (
    RFFKernelLayer,
)


def test_the_same_seed_draws_the_same_projection():
    first = RFFKernelLayer(input_dim=8, n_features=32, seed=0)
    first.build((None, 8))
    second = RFFKernelLayer(input_dim=8, n_features=32, seed=0)
    second.build((None, 8))
    np.testing.assert_array_equal(np.asarray(first.omega), np.asarray(second.omega))
    np.testing.assert_array_equal(np.asarray(first.b), np.asarray(second.b))


def test_a_different_seed_draws_a_different_projection():
    first = RFFKernelLayer(input_dim=8, n_features=32, seed=0)
    first.build((None, 8))
    other = RFFKernelLayer(input_dim=8, n_features=32, seed=1)
    other.build((None, 8))
    assert float(np.abs(np.asarray(first.omega) - np.asarray(other.omega)).max()) > 0


def test_an_unseeded_layer_still_builds():
    layer = RFFKernelLayer(input_dim=8, n_features=16)
    layer.build((None, 8))
    assert layer.omega.shape == (8, 16)
    assert layer.b.shape == (16,)


def test_the_seed_round_trips_through_get_config():
    assert RFFKernelLayer(input_dim=8, seed=7).get_config()["seed"] == 7
    assert RFFKernelLayer(input_dim=8).get_config()["seed"] is None


def test_the_seed_does_not_cover_the_output_projection():
    # `seed` fixes the frozen omega/b only; the Dense kernel still draws from
    # the global RNG. Two same-seed layers therefore share projections and
    # differ downstream -- the documented contract, not a defect.
    first = RFFKernelLayer(input_dim=8, n_features=32, seed=0)
    first.build((None, 8))
    second = RFFKernelLayer(input_dim=8, n_features=32, seed=0)
    second.build((None, 8))
    assert float(np.abs(np.asarray(first.linear.kernel) - np.asarray(second.linear.kernel)).max()) > 0


def test_the_layer_builds_under_mixed_float16(mixed_float16_policy):
    layer = RFFKernelLayer(input_dim=8, n_features=16)
    layer.build((None, 8))
    out = np.asarray(layer(np.ones((2, 8), dtype=np.float32)))
    assert np.all(np.isfinite(out))


def test_a_keras_archive_restores_the_projection_bit_exactly(tmp_path):
    layer = RFFKernelLayer(input_dim=8, n_features=16, output_dim=5, seed=0)
    inputs = keras.Input(shape=(8,))
    model = keras.Model(inputs, layer(inputs))
    path = str(tmp_path / "rff.keras")
    model.save(path)
    restored = keras.models.load_model(path)
    np.testing.assert_array_equal(
        np.asarray(model.layers[1].omega), np.asarray(restored.layers[1].omega)
    )
    np.testing.assert_array_equal(
        np.asarray(model.layers[1].b), np.asarray(restored.layers[1].b)
    )
