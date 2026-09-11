import numpy as np
import keras
import pytest

from dl_techniques.layers.geometric.ray_point_combinator import (
    RayPointCombinator,
)


class TestRayPointCombinator:
    """Test suite for RayPointCombinator."""

    @pytest.fixture
    def raw_ray(self) -> np.ndarray:
        rng = np.random.default_rng(0)
        return rng.normal(size=(2, 8, 8, 3)).astype("float32")

    @pytest.fixture
    def raw_distance(self) -> np.ndarray:
        rng = np.random.default_rng(1)
        return rng.normal(size=(2, 8, 8, 1)).astype("float32") * 10.0

    @pytest.fixture
    def layer_instance(self) -> RayPointCombinator:
        return RayPointCombinator()

    # ------------------------------------------------------------------
    # Invariant 1: unit-norm ray
    # ------------------------------------------------------------------

    def test_ray_is_unit_norm(self, layer_instance, raw_ray, raw_distance):
        ray, _, _ = layer_instance((raw_ray, raw_distance))
        norms = keras.ops.convert_to_numpy(
            keras.ops.sqrt(keras.ops.sum(keras.ops.square(ray), axis=-1))
        )
        np.testing.assert_allclose(
            norms, np.ones_like(norms), atol=1e-6, rtol=0
        )

    # ------------------------------------------------------------------
    # Invariant 2: strictly positive distance
    # ------------------------------------------------------------------

    def test_distance_is_strictly_positive(self, layer_instance, raw_ray):
        raw_distance = np.array(
            [-1e6, -100.0, -1.0, 0.0, 1.0, 100.0, 1e6, -1e-3],
            dtype="float32",
        ).reshape((2, 4, 1, 1))
        raw_ray_local = np.broadcast_to(
            raw_ray[:, :4, :1, :], (2, 4, 1, 3)
        ).copy()
        _, distance, _ = layer_instance((raw_ray_local, raw_distance))
        distance_np = keras.ops.convert_to_numpy(distance)
        assert np.all(distance_np > 0.0)

    # ------------------------------------------------------------------
    # Invariant 3: P == d_hat * r_hat exactly
    # ------------------------------------------------------------------

    def test_point_equals_distance_times_ray(
            self, layer_instance, raw_ray, raw_distance
    ):
        ray, distance, point = layer_instance((raw_ray, raw_distance))
        expected = keras.ops.convert_to_numpy(distance) * keras.ops.convert_to_numpy(ray)
        np.testing.assert_array_equal(
            keras.ops.convert_to_numpy(point), expected
        )

    # ------------------------------------------------------------------
    # Edge case: all-zero raw ray does not produce NaN
    # ------------------------------------------------------------------

    def test_all_zero_ray_does_not_produce_nan(self, layer_instance):
        raw_ray = np.zeros((1, 2, 2, 3), dtype="float32")
        raw_distance = np.ones((1, 2, 2, 1), dtype="float32")
        ray, distance, point = layer_instance((raw_ray, raw_distance))
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(ray)))
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(distance)))
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(point)))

    # ------------------------------------------------------------------
    # get_config / from_config round-trip
    # ------------------------------------------------------------------

    def test_config_round_trip(self, layer_instance):
        config = layer_instance.get_config()
        restored = RayPointCombinator.from_config(config)
        assert restored.epsilon == layer_instance.epsilon

    # ------------------------------------------------------------------
    # Functional-model composition
    # ------------------------------------------------------------------

    def test_composes_in_functional_model(self, raw_ray, raw_distance):
        ray_input = keras.Input(shape=(8, 8, 3), name="raw_ray")
        distance_input = keras.Input(shape=(8, 8, 1), name="raw_distance")
        ray, distance, point = RayPointCombinator()((ray_input, distance_input))
        model = keras.Model(
            inputs=(ray_input, distance_input),
            outputs=(ray, distance, point),
        )
        out_ray, out_distance, out_point = model((raw_ray, raw_distance))
        assert tuple(out_ray.shape) == (2, 8, 8, 3)
        assert tuple(out_distance.shape) == (2, 8, 8, 1)
        assert tuple(out_point.shape) == (2, 8, 8, 3)
        norms = keras.ops.convert_to_numpy(
            keras.ops.sqrt(keras.ops.sum(keras.ops.square(out_ray), axis=-1))
        )
        np.testing.assert_allclose(
            norms, np.ones_like(norms), atol=1e-6, rtol=0
        )
