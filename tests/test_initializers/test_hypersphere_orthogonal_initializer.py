"""Tests for OrthogonalHypersphereInitializer.

Several tests here are single-claim guards for defects found by review and
verified by measurement against the previous implementation:

* orthogonality was enforced between ROWS only, so every narrowing Dense and
  effectively every Conv2D took a "mathematically impossible" fallback that was
  neither impossible nor necessary. (128, 64), (512, 128), (3, 3, 64, 128) and
  (30000, 512) all warned and degraded.
* that fallback sampled uniformly on the sphere and claimed to maximize "average
  angular separation", a property no configuration has. Measured at (512, 128):
  cond(W) 2.92, singular values 1.03 to 3.01, and 0.2% of pairs exactly
  orthogonal -- discarding the dynamical isometry the Saxe reference is cited
  for. Stacking independent orthonormal bases gives cond 1.0000 and 25%.
* the QR was not sign corrected, so Q was not Haar distributed: over 2000 seeds
  at d=8, Q[0, 0] was negative in 2000 of 2000 draws with E[Q[0, 0]] = -0.2897
  where Haar gives 0.
* np.random.default_rng(None) bypassed keras.utils.set_random_seed, radius=nan
  passed validation and produced an all-NaN tensor, a rank-1 shape was silently
  accepted, and dtype=None ignored floatx.
"""

import pytest
import numpy as np
import tempfile
import os
import warnings
from typing import Any, Dict, Tuple

import keras
from dl_techniques.initializers.hypersphere_orthogonal_initializer import (
    FALLBACK_MODES,
    OrthogonalHypersphereInitializer,
)


def _rows(weights, latent_dim: int) -> np.ndarray:
    """Flatten a weight tensor to its (num_vectors, latent_dim) matrix."""
    return np.asarray(keras.ops.convert_to_numpy(weights)).reshape(-1, latent_dim)


def _condition_number(matrix: np.ndarray) -> float:
    singular = np.linalg.svd(matrix, compute_uv=False)
    return float(singular.max() / singular.min())


def _exactly_orthogonal_fraction(matrix: np.ndarray) -> float:
    unit = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    gram = np.abs(unit @ unit.T)
    np.fill_diagonal(gram, 0.0)
    return float((gram < 1e-6).mean())


class TestOrthogonalHypersphereInitializer:
    """Comprehensive test suite for OrthogonalHypersphereInitializer."""

    @pytest.fixture
    def basic_config(self) -> Dict[str, Any]:
        """Standard configuration for testing."""
        return {
            'radius': 1.5,
            'seed': 42
        }

    @pytest.fixture
    def feasible_shape(self) -> Tuple[int, ...]:
        """Shape where orthogonality is feasible (num_vectors <= latent_dim)."""
        return (10, 128)  # 10 vectors in 128D space

    @pytest.fixture
    def infeasible_shape(self) -> Tuple[int, ...]:
        """Shape where not all ROWS can be orthogonal (num_vectors > latent_dim)."""
        return (256, 64)  # 256 vectors in 64D space -> 4 stacked bases

    @pytest.fixture
    def multi_dim_shape(self) -> Tuple[int, ...]:
        """Multi-dimensional grid shape."""
        return (4, 8, 128)  # 32 vectors (4*8) in 128D space

    def test_initialization(self, basic_config):
        """Test initializer creation and parameter validation."""
        # Basic initialization
        initializer = OrthogonalHypersphereInitializer(**basic_config)

        assert initializer.radius == basic_config['radius']
        assert initializer.seed == basic_config['seed']

    def test_initialization_validation(self):
        """Test parameter validation during initialization."""
        # Valid parameters
        OrthogonalHypersphereInitializer(radius=1.0, seed=None)
        OrthogonalHypersphereInitializer(radius=2.5, seed=123)

        # Invalid radius
        with pytest.raises(ValueError, match="radius must be positive"):
            OrthogonalHypersphereInitializer(radius=0.0)

        with pytest.raises(ValueError, match="radius must be positive"):
            OrthogonalHypersphereInitializer(radius=-1.0)

    def test_call_validation(self, basic_config):
        """Test input validation in __call__ method."""
        initializer = OrthogonalHypersphereInitializer(**basic_config)

        # Valid shapes
        initializer(shape=(10, 64))
        initializer(shape=(5, 5, 32))

        # Invalid shapes
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            initializer(shape=())

        with pytest.raises(ValueError, match="at least 2 dimensions"):
            initializer(shape=(64,))

        with pytest.raises(ValueError, match="dimensions must be positive"):
            initializer(shape=(10, 0))

        with pytest.raises(ValueError, match="dimensions must be positive"):
            initializer(shape=(10, -5))

        with pytest.raises(ValueError, match="dimensions must be positive"):
            initializer(shape=(0, 10))

    def test_feasible_orthogonal_generation(self, basic_config, feasible_shape):
        """Test orthogonal vector generation when mathematically feasible."""
        initializer = OrthogonalHypersphereInitializer(**basic_config)

        weights = initializer(shape=feasible_shape)
        weights_np = keras.ops.convert_to_numpy(weights)

        # Check shape
        assert weights.shape == feasible_shape

        # Check radius property - all vectors should have specified radius
        vector_norms = np.linalg.norm(weights_np, axis=1)
        expected_radius = basic_config['radius']
        np.testing.assert_allclose(
            vector_norms,
            np.full_like(vector_norms, expected_radius),
            rtol=1e-5, atol=1e-5,
            err_msg="All vectors should have the specified radius"
        )

        # Check orthogonality - dot products should be near zero
        num_vectors = feasible_shape[0]
        dot_products = np.dot(weights_np, weights_np.T)

        # Extract off-diagonal elements (should be near zero for orthogonal vectors)
        off_diagonal_mask = ~np.eye(num_vectors, dtype=bool)
        off_diagonal_dots = dot_products[off_diagonal_mask]

        np.testing.assert_allclose(
            off_diagonal_dots,
            np.zeros_like(off_diagonal_dots),
            rtol=1e-4, atol=1e-4,
            err_msg="Orthogonal vectors should have near-zero dot products"
        )

        # Check diagonal elements (should be radius squared)
        diagonal_dots = np.diag(dot_products)
        expected_diagonal = expected_radius ** 2
        np.testing.assert_allclose(
            diagonal_dots,
            np.full_like(diagonal_dots, expected_diagonal),
            rtol=1e-5, atol=1e-5,
            err_msg="Self dot products should equal radius squared"
        )

    def test_the_over_complete_regime_is_silent_and_well_conditioned(
        self, basic_config, infeasible_shape
    ):
        """num_vectors > latent_dim stacks orthonormal bases, and does not warn.

        It previously emitted a UserWarning calling the request "mathematically
        impossible" and degraded to uniform sampling. Ten test modules had to
        suppress that warning because it fires on the ORDINARY case for a
        dimension-reducing projection.
        """
        initializer = OrthogonalHypersphereInitializer(**basic_config)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            weights = initializer(shape=infeasible_shape)

        assert weights.shape == infeasible_shape

        matrix = _rows(weights, infeasible_shape[-1])
        np.testing.assert_allclose(
            np.linalg.norm(matrix, axis=1), basic_config["radius"],
            rtol=1e-5, atol=1e-5,
            err_msg="every vector must still lie on the hypersphere",
        )
        # 256 = 4 x 64 exactly, so this is an exact tight frame.
        assert _condition_number(matrix) == pytest.approx(1.0, abs=1e-4)
        assert _exactly_orthogonal_fraction(matrix) == pytest.approx(0.25, abs=0.01)

    def test_stacked_bases_beat_the_uniform_fallback(self):
        """The construction that replaced uniform sampling is measurably better.

        Anti-vacuity: the retired construction is still reachable behind
        fallback='uniform', and it must fail the same predicate the default
        passes -- otherwise this test would pass on any two random matrices.
        """
        shape = (512, 128)
        block = _rows(
            OrthogonalHypersphereInitializer(seed=0)(shape), shape[-1]
        )
        uniform = _rows(
            OrthogonalHypersphereInitializer(seed=0, fallback="uniform")(shape),
            shape[-1],
        )

        assert _condition_number(block) == pytest.approx(1.0, abs=1e-4)
        assert _condition_number(uniform) > 2.0

        assert _exactly_orthogonal_fraction(block) > 0.2
        assert _exactly_orthogonal_fraction(uniform) < 0.01

        # Mean coherence improves too, though neither is a good spherical code:
        # the Welch bound here is 0.0766.
        def mean_coherence(m):
            unit = m / np.linalg.norm(m, axis=1, keepdims=True)
            gram = np.abs(unit @ unit.T)
            np.fill_diagonal(gram, 0.0)
            return float(gram.mean())

        assert mean_coherence(block) < mean_coherence(uniform)

    @pytest.mark.parametrize("shape,blocks", [
        ((128, 64), 2), ((512, 128), 4), ((3, 3, 64, 128), 5), ((10, 8, 32), 3),
    ])
    def test_the_shapes_that_used_to_degrade(self, shape, blocks):
        """Every shape that took the old fallback now keeps its geometry.

        These are the real consumer shapes: a narrowing Dense (OrthoBlock), an
        OrthoGLU down-projection, a Conv2D kernel, and a NeuroGrid grid.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            weights = OrthogonalHypersphereInitializer(seed=0)(shape=shape)

        matrix = _rows(weights, shape[-1])
        np.testing.assert_allclose(
            np.linalg.norm(matrix, axis=1), 1.0, rtol=1e-5, atol=1e-5
        )
        # A partial trailing block bounds the condition number by sqrt(2).
        assert _condition_number(matrix) <= np.sqrt(2.0) + 1e-4
        assert _exactly_orthogonal_fraction(matrix) == pytest.approx(
            1.0 / blocks, abs=0.05
        )

    def test_the_uniform_fallback_is_still_reachable(self):
        """The retired behaviour stays available, and says it is worse."""
        assert "uniform" in FALLBACK_MODES
        weights = OrthogonalHypersphereInitializer(
            radius=1.5, seed=0, fallback="uniform"
        )(shape=(256, 64))
        matrix = _rows(weights, 64)
        np.testing.assert_allclose(
            np.linalg.norm(matrix, axis=1), 1.5, rtol=1e-5, atol=1e-5
        )

    def test_an_unknown_fallback_is_rejected(self):
        with pytest.raises(ValueError, match="fallback must be one of"):
            OrthogonalHypersphereInitializer(fallback="random")

    def test_the_qr_is_sign_corrected(self):
        """Q must be Haar distributed, which LAPACK's raw QR is not.

        Householder QR fixes the sign of R's diagonal, which made Q[0, 0]
        negative in 2000 of 2000 draws with E[Q[0, 0]] = -0.2897. Multiplying by
        sign(diag(R)) restores it -- the same convention
        he_orthonormal_initializer.py already used.
        """
        first = np.array([
            np.asarray(keras.ops.convert_to_numpy(
                OrthogonalHypersphereInitializer(seed=s)(shape=(8, 8))
            ))[0, 0]
            for s in range(400)
        ])

        assert abs(float(first.mean())) < 0.08, float(first.mean())
        assert 0.35 < float((first < 0).mean()) < 0.65, float((first < 0).mean())

    def test_a_seedless_instance_honours_the_global_seed(self):
        """np.random.default_rng(None) ignored keras.utils.set_random_seed."""
        keras.utils.set_random_seed(1234)
        a = np.asarray(keras.ops.convert_to_numpy(
            OrthogonalHypersphereInitializer()(shape=(8, 16))))
        keras.utils.set_random_seed(1234)
        b = np.asarray(keras.ops.convert_to_numpy(
            OrthogonalHypersphereInitializer()(shape=(8, 16))))
        np.testing.assert_array_equal(a, b)

        keras.utils.set_random_seed(4321)
        c = np.asarray(keras.ops.convert_to_numpy(
            OrthogonalHypersphereInitializer()(shape=(8, 16))))
        assert not np.allclose(a, c)

    def test_reproducibility_on_the_over_complete_branch(self):
        """Determinism was only ever asserted on the feasible branch."""
        shape = (256, 64)
        a = np.asarray(keras.ops.convert_to_numpy(
            OrthogonalHypersphereInitializer(seed=7)(shape)))
        b = np.asarray(keras.ops.convert_to_numpy(
            OrthogonalHypersphereInitializer(seed=7)(shape)))
        np.testing.assert_array_equal(a, b)

    def test_a_rank_one_shape_is_rejected(self):
        """(32,) silently became one random direction spread over a bias."""
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            OrthogonalHypersphereInitializer()(shape=(32,))

    @pytest.mark.parametrize("radius", [float("nan"), float("inf")])
    def test_a_non_finite_radius_is_rejected(self, radius):
        """radius=nan passed `radius <= 0` and produced an all-NaN tensor."""
        with pytest.raises(ValueError, match="radius must be finite"):
            OrthogonalHypersphereInitializer(radius=radius)

    def test_dtype_none_follows_floatx(self):
        original = keras.config.floatx()
        try:
            for floatx in ("float32", "float64"):
                keras.config.set_floatx(floatx)
                weights = OrthogonalHypersphereInitializer(seed=0)(shape=(4, 8))
                assert keras.backend.standardize_dtype(weights.dtype) == floatx
        finally:
            keras.config.set_floatx(original)

    def test_float64_is_not_a_float32_upcast(self):
        weights = np.asarray(keras.ops.convert_to_numpy(
            OrthogonalHypersphereInitializer(seed=0)(shape=(64, 128), dtype="float64")))
        assert weights.dtype == np.float64
        assert not np.array_equal(
            weights, weights.astype(np.float32).astype(np.float64)
        )

    def test_call_accepts_extra_kwargs(self):
        weights = OrthogonalHypersphereInitializer(seed=0)(
            (4, 8), None, partition_shape=None
        )
        assert tuple(weights.shape) == (4, 8)

    def test_multi_dimensional_grids(self, basic_config, multi_dim_shape):
        """Test handling of multi-dimensional weight grids."""
        initializer = OrthogonalHypersphereInitializer(**basic_config)

        weights = initializer(shape=multi_dim_shape)
        weights_np = keras.ops.convert_to_numpy(weights)

        # Check shape preservation
        assert weights.shape == multi_dim_shape

        # Flatten to 2D for analysis
        grid_size = multi_dim_shape[0] * multi_dim_shape[1]  # 4 * 8 = 32
        latent_dim = multi_dim_shape[2]  # 128
        weights_2d = weights_np.reshape(grid_size, latent_dim)

        # Check radius property
        vector_norms = np.linalg.norm(weights_2d, axis=1)
        expected_radius = basic_config['radius']
        np.testing.assert_allclose(
            vector_norms,
            np.full_like(vector_norms, expected_radius),
            rtol=1e-5, atol=1e-5,
            err_msg="Multi-dimensional grids should preserve radius property"
        )

        # The grid path was never checked for orthogonality, only for norms.
        matrix = _rows(weights, multi_dim_shape[-1])
        gram = matrix @ matrix.T
        np.fill_diagonal(gram, 0.0)
        assert np.abs(gram).max() < 1e-4

    def test_reproducibility_with_seed(self):
        """Test that same seed produces identical results."""
        shape = (8, 64)
        seed = 12345
        radius = 2.0

        # Generate weights twice with same seed
        init1 = OrthogonalHypersphereInitializer(radius=radius, seed=seed)
        init2 = OrthogonalHypersphereInitializer(radius=radius, seed=seed)

        weights1 = init1(shape=shape)
        weights2 = init2(shape=shape)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(weights1),
            keras.ops.convert_to_numpy(weights2),
            rtol=1e-10, atol=1e-10,
            err_msg="Same seed should produce identical results"
        )

        # Different seed should produce different results
        init3 = OrthogonalHypersphereInitializer(radius=radius, seed=seed + 1)
        weights3 = init3(shape=shape)

        with pytest.raises(AssertionError):
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(weights1),
                keras.ops.convert_to_numpy(weights3),
                rtol=1e-5, atol=1e-5
            )

    def test_serialization_cycle(self, basic_config, feasible_shape):
        """CRITICAL TEST: Full serialization cycle with model."""
        # Create a layer using the initializer
        layer = keras.layers.Dense(
            units=feasible_shape[1],
            kernel_initializer=OrthogonalHypersphereInitializer(**basic_config)
        )

        # Create model
        inputs = keras.Input(shape=feasible_shape[1:])
        outputs = layer(inputs)
        model = keras.Model(inputs, outputs)

        # Get original weights
        sample_input = keras.random.normal(shape=(4,) + feasible_shape[1:])
        original_weights = model.get_weights()
        original_prediction = model(sample_input)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_weights = loaded_model.get_weights()
            loaded_prediction = loaded_model(sample_input)

            # Verify weights are identical
            for orig, loaded in zip(original_weights, loaded_weights):
                np.testing.assert_allclose(
                    keras.ops.convert_to_numpy(orig),
                    keras.ops.convert_to_numpy(loaded),
                    rtol=1e-10, atol=1e-10,
                    err_msg="Weights should be identical after serialization"
                )

            # Verify predictions are identical
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_prediction),
                keras.ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions should match after serialization"
            )

    def test_config_completeness(self, basic_config):
        """Test that get_config contains all __init__ parameters."""
        initializer = OrthogonalHypersphereInitializer(**basic_config)
        config = initializer.get_config()

        # Check all config parameters are present
        assert config["fallback"] == "block_orthogonal"
        for key in basic_config:
            assert key in config, f"Missing {key} in get_config()"
            assert config[key] == basic_config[key], f"Config value mismatch for {key}"

    def test_config_reconstruction(self, basic_config, feasible_shape):
        """Test that initializer can be reconstructed from config."""
        # Create original initializer
        original_init = OrthogonalHypersphereInitializer(**basic_config)

        # Get config and reconstruct
        config = original_init.get_config()
        reconstructed_init = OrthogonalHypersphereInitializer(**config)

        # Both should produce identical results
        weights_original = original_init(shape=feasible_shape)
        weights_reconstructed = reconstructed_init(shape=feasible_shape)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(weights_original),
            keras.ops.convert_to_numpy(weights_reconstructed),
            rtol=1e-10, atol=1e-10,
            err_msg="Reconstructed initializer should produce identical results"
        )

    @pytest.mark.parametrize("radius", [0.5, 1.0, 2.0, 10.0])
    def test_different_radii(self, radius):
        """Test initialization with different radius values."""
        shape = (5, 32)
        initializer = OrthogonalHypersphereInitializer(radius=radius, seed=42)

        weights = initializer(shape=shape)
        weights_np = keras.ops.convert_to_numpy(weights)

        # Check that all vectors have the specified radius
        vector_norms = np.linalg.norm(weights_np, axis=1)
        np.testing.assert_allclose(
            vector_norms,
            np.full_like(vector_norms, radius),
            rtol=1e-5, atol=1e-5,
            err_msg=f"All vectors should have radius {radius}"
        )

    def test_edge_case_single_vector(self):
        """Test edge case with single vector."""
        shape = (1, 64)
        initializer = OrthogonalHypersphereInitializer(radius=3.0, seed=42)

        weights = initializer(shape=shape)
        weights_np = keras.ops.convert_to_numpy(weights)

        # Check radius
        vector_norm = np.linalg.norm(weights_np[0])
        np.testing.assert_allclose(
            vector_norm, 3.0, rtol=1e-5, atol=1e-5,
            err_msg="Single vector should have correct radius"
        )

    def test_edge_case_1d_latent(self):
        """3 vectors in a 1-D space: 3 stacked 1-D bases, i.e. +/- radius, silently."""
        initializer = OrthogonalHypersphereInitializer(radius=2.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            weights = initializer(shape=(3, 1))

        assert weights.shape == (3, 1)
        np.testing.assert_allclose(
            np.abs(np.asarray(keras.ops.convert_to_numpy(weights))).ravel(),
            2.0, rtol=1e-5, atol=1e-5,
        )

    def test_dtype_handling(self, basic_config, feasible_shape):
        """Test handling of different data types."""
        initializer = OrthogonalHypersphereInitializer(**basic_config)

        # Test default dtype
        weights_default = initializer(shape=feasible_shape)
        assert weights_default.dtype == keras.backend.floatx()

        # Test explicit dtype
        weights_float32 = initializer(shape=feasible_shape, dtype='float32')
        assert weights_float32.dtype == 'float32'

    def test_repr_method(self, basic_config):
        """Test string representation."""
        initializer = OrthogonalHypersphereInitializer(**basic_config)
        repr_str = repr(initializer)

        assert "OrthogonalHypersphereInitializer" in repr_str
        assert f"radius={basic_config['radius']}" in repr_str
        assert f"seed={basic_config['seed']}" in repr_str


# Additional integration tests
class TestOrthogonalHypersphereIntegration:
    """Integration tests with Keras layers and models."""

    def test_dense_layer_integration(self):
        """Test integration with Dense layer."""
        initializer = OrthogonalHypersphereInitializer(radius=1.5, seed=42)

        layer = keras.layers.Dense(
            units=64,
            kernel_initializer=initializer
        )

        # Build layer
        layer.build(input_shape=(None, 32))

        # Check weights have correct properties
        weights = layer.get_weights()[0]  # kernel weights, shape: (32, 64)
        # The initializer creates row vectors (along input dimension) with specified radius
        vector_norms = np.linalg.norm(weights, axis=1)  # norm of each input vector (row)

        np.testing.assert_allclose(
            vector_norms,
            np.full_like(vector_norms, 1.5),
            rtol=1e-5, atol=1e-5,
            err_msg="Dense layer weights should have correct radius"
        )

    def test_embedding_layer_integration(self):
        """Test integration with Embedding layer."""
        vocab_size = 100
        embed_dim = 128

        initializer = OrthogonalHypersphereInitializer(radius=2.0, seed=123)

        layer = keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            embeddings_initializer=initializer
        )

        # Build layer
        layer.build(input_shape=())

        # Check embedding weights
        embeddings = layer.get_weights()[0]  # shape: (vocab_size, embed_dim)
        # Each row is an embedding vector for one vocabulary item
        vector_norms = np.linalg.norm(embeddings, axis=1)

        np.testing.assert_allclose(
            vector_norms,
            np.full_like(vector_norms, 2.0),
            rtol=1e-5, atol=1e-5,
            err_msg="Embedding weights should have correct radius"
        )

    def test_model_compilation_and_training(self):
        """Test that models with this initializer can be compiled and trained."""
        # Create simple model
        model = keras.Sequential([
            keras.layers.Dense(
                64,
                activation='relu',
                kernel_initializer=OrthogonalHypersphereInitializer(radius=1.0, seed=42)
            ),
            keras.layers.Dense(10, activation='softmax')
        ])

        # Compile
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Create dummy data
        x_train = keras.random.normal(shape=(100, 32))
        y_train = np.random.randint(0, 10, size=(100,))  # Use numpy instead

        # Train for one epoch (just to verify it works)
        history = model.fit(
            x_train, y_train,
            epochs=1,
            batch_size=32,
            verbose=0
        )

        assert len(history.history['loss']) == 1
        assert isinstance(history.history['loss'][0], float)

# Run tests with: pytest test_orthogonal_hypersphere_initializer.py -v