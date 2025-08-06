import pytest
import numpy as np
import tensorflow as tf
import keras
import tempfile
import os

from dl_techniques.layers.experimental.hierarchical_memory_system import HierarchicalMemorySystem


class TestHierarchicalMemorySystem:
    """Test suite for HierarchicalMemorySystem implementation."""

    @pytest.fixture
    def input_data_small(self):
        """Create small test input data."""
        return keras.random.uniform([16, 64], seed=42)

    @pytest.fixture
    def input_data_medium(self):
        """Create medium test input data."""
        return keras.random.uniform([32, 128], seed=42)

    @pytest.fixture
    def input_data_large(self):
        """Create large test input data (MNIST-like)."""
        return keras.random.uniform([64, 784], seed=42)

    @pytest.fixture
    def hierarchical_memory_default(self):
        """Create a default hierarchical memory system."""
        return HierarchicalMemorySystem(
            input_dim=128,
            levels=3,
            grid_dimensions=2,
            base_grid_size=5,
            grid_expansion_factor=2.0
        )

    @pytest.fixture
    def hierarchical_memory_1d(self):
        """Create a 1D hierarchical memory system."""
        return HierarchicalMemorySystem(
            input_dim=64,
            levels=2,
            grid_dimensions=1,
            base_grid_size=10,
            grid_expansion_factor=1.5
        )

    @pytest.fixture
    def hierarchical_memory_3d(self):
        """Create a 3D hierarchical memory system."""
        return HierarchicalMemorySystem(
            input_dim=256,
            levels=2,
            grid_dimensions=3,
            base_grid_size=3,
            grid_expansion_factor=2.0
        )

    @pytest.fixture
    def hierarchical_memory_single_level(self):
        """Create a single-level hierarchical memory system."""
        return HierarchicalMemorySystem(
            input_dim=100,
            levels=1,
            grid_dimensions=2,
            base_grid_size=8
        )

    @pytest.fixture
    def hierarchical_memory_custom(self):
        """Create a hierarchical memory system with custom parameters."""
        return HierarchicalMemorySystem(
            input_dim=200,
            levels=4,
            grid_dimensions=2,
            base_grid_size=4,
            grid_expansion_factor=1.8,
            initial_learning_rate=0.05,
            sigma=2.0,
            neighborhood_function='bubble',
            weights_initializer='glorot_uniform',
            regularizer=keras.regularizers.L2(1e-4)
        )

    def test_initialization_default(self, hierarchical_memory_default):
        """Test initialization with default parameters."""
        hm = hierarchical_memory_default
        assert hm.input_dim == 128
        assert hm.levels == 3
        assert hm.grid_dimensions == 2
        assert hm.base_grid_size == 5
        assert hm.grid_expansion_factor == 2.0
        assert hm.initial_learning_rate == 0.1
        assert hm.sigma == 1.0
        assert hm.neighborhood_function == 'gaussian'
        assert hm.som_layers is None  # Not built yet

    def test_initialization_1d(self, hierarchical_memory_1d):
        """Test initialization of 1D hierarchical memory system."""
        hm = hierarchical_memory_1d
        assert hm.input_dim == 64
        assert hm.levels == 2
        assert hm.grid_dimensions == 1
        assert hm.base_grid_size == 10
        assert hm.grid_expansion_factor == 1.5

    def test_initialization_3d(self, hierarchical_memory_3d):
        """Test initialization of 3D hierarchical memory system."""
        hm = hierarchical_memory_3d
        assert hm.input_dim == 256
        assert hm.levels == 2
        assert hm.grid_dimensions == 3
        assert hm.base_grid_size == 3

    def test_initialization_custom_parameters(self, hierarchical_memory_custom):
        """Test initialization with custom parameters."""
        hm = hierarchical_memory_custom
        assert hm.input_dim == 200
        assert hm.levels == 4
        assert hm.grid_dimensions == 2
        assert hm.base_grid_size == 4
        assert hm.grid_expansion_factor == 1.8
        assert hm.initial_learning_rate == 0.05
        assert hm.sigma == 2.0
        assert hm.neighborhood_function == 'bubble'

    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        with pytest.raises(ValueError, match="input_dim must be positive"):
            HierarchicalMemorySystem(input_dim=-10)

        with pytest.raises(ValueError, match="levels must be at least 1"):
            HierarchicalMemorySystem(input_dim=100, levels=0)

        with pytest.raises(ValueError, match="grid_dimensions must be at least 1"):
            HierarchicalMemorySystem(input_dim=100, grid_dimensions=0)

        with pytest.raises(ValueError, match="base_grid_size must be at least 1"):
            HierarchicalMemorySystem(input_dim=100, base_grid_size=0)

        with pytest.raises(ValueError, match="grid_expansion_factor must be positive"):
            HierarchicalMemorySystem(input_dim=100, grid_expansion_factor=-1.0)

    def test_build_process_default(self, hierarchical_memory_default, input_data_medium):
        """Test that default hierarchical memory system builds properly."""
        hm = hierarchical_memory_default
        hm.build(input_data_medium.shape)

        assert hm.built is True
        assert hm.som_layers is not None
        assert len(hm.som_layers) == 3

        # Check grid shapes progression: 5x5 -> 10x10 -> 20x20
        expected_shapes = [(5, 5), (10, 10), (20, 20)]
        for i, som_layer in enumerate(hm.som_layers):
            assert som_layer.grid_shape == expected_shapes[i]
            assert som_layer.input_dim == 128
            assert som_layer.built is True

    def test_build_process_1d(self, hierarchical_memory_1d, input_data_small):
        """Test that 1D hierarchical memory system builds properly."""
        hm = hierarchical_memory_1d
        hm.build(input_data_small.shape)

        assert hm.built is True
        assert len(hm.som_layers) == 2

        # Check 1D grid shapes progression: (10,) -> (15,)
        expected_shapes = [(10,), (15,)]  # 10 * 1.5 = 15
        for i, som_layer in enumerate(hm.som_layers):
            assert som_layer.grid_shape == expected_shapes[i]
            assert som_layer.input_dim == 64

    def test_build_process_3d(self, hierarchical_memory_3d, input_data_large):
        """Test that 3D hierarchical memory system builds properly."""
        hm = hierarchical_memory_3d
        hm.build((64, 256))  # Use compatible input shape

        assert hm.built is True
        assert len(hm.som_layers) == 2

        # Check 3D grid shapes progression: (3,3,3) -> (6,6,6)
        expected_shapes = [(3, 3, 3), (6, 6, 6)]
        for i, som_layer in enumerate(hm.som_layers):
            assert som_layer.grid_shape == expected_shapes[i]
            assert som_layer.input_dim == 256

    def test_build_process_single_level(self, hierarchical_memory_single_level):
        """Test that single-level hierarchical memory system builds properly."""
        hm = hierarchical_memory_single_level
        hm.build((32, 100))

        assert hm.built is True
        assert len(hm.som_layers) == 1
        assert hm.som_layers[0].grid_shape == (8, 8)
        assert hm.som_layers[0].input_dim == 100

    def test_forward_pass_default(self, hierarchical_memory_default, input_data_medium):
        """Test forward pass with default hierarchical memory system."""
        hm = hierarchical_memory_default
        bmu_indices_list, quantization_errors_list = hm(input_data_medium, training=False)

        # Check that we get results for all levels
        assert len(bmu_indices_list) == 3
        assert len(quantization_errors_list) == 3

        # Check shapes for each level
        for i, (bmu_indices, q_errors) in enumerate(zip(bmu_indices_list, quantization_errors_list)):
            assert bmu_indices.shape == (32, 2)  # batch_size, grid_dimensions
            assert q_errors.shape == (32,)  # batch_size
            assert bmu_indices.dtype == tf.int32
            assert q_errors.dtype == tf.float32

        # Check that BMU indices are within valid ranges for each level
        grid_sizes = [5, 10, 20]
        for i, bmu_indices in enumerate(bmu_indices_list):
            assert tf.reduce_all(bmu_indices >= 0)
            assert tf.reduce_all(bmu_indices < grid_sizes[i])

        # Check that quantization errors are non-negative
        for q_errors in quantization_errors_list:
            assert tf.reduce_all(q_errors >= 0)

    def test_forward_pass_1d(self, hierarchical_memory_1d, input_data_small):
        """Test forward pass with 1D hierarchical memory system."""
        hm = hierarchical_memory_1d
        bmu_indices_list, quantization_errors_list = hm(input_data_small, training=False)

        assert len(bmu_indices_list) == 2
        assert len(quantization_errors_list) == 2

        # Check shapes for 1D grids
        for bmu_indices, q_errors in zip(bmu_indices_list, quantization_errors_list):
            assert bmu_indices.shape == (16, 1)  # batch_size, 1 dimension
            assert q_errors.shape == (16,)

        # Check ranges for 1D grids: (10,) and (15,)
        assert tf.reduce_all(bmu_indices_list[0] < 10)
        assert tf.reduce_all(bmu_indices_list[1] < 15)

    def test_forward_pass_3d(self, hierarchical_memory_3d):
        """Test forward pass with 3D hierarchical memory system."""
        # Create compatible input
        input_data = keras.random.uniform([24, 256], seed=42)
        hm = hierarchical_memory_3d
        bmu_indices_list, quantization_errors_list = hm(input_data, training=False)

        assert len(bmu_indices_list) == 2
        assert len(quantization_errors_list) == 2

        # Check shapes for 3D grids
        for bmu_indices, q_errors in zip(bmu_indices_list, quantization_errors_list):
            assert bmu_indices.shape == (24, 3)  # batch_size, 3 dimensions
            assert q_errors.shape == (24,)

        # Check ranges for 3D grids: (3,3,3) and (6,6,6)
        for dim in range(3):
            assert tf.reduce_all(bmu_indices_list[0][:, dim] < 3)
            assert tf.reduce_all(bmu_indices_list[1][:, dim] < 6)

    def test_training_mode_weight_updates(self, hierarchical_memory_default, input_data_medium):
        """Test that weights are updated during training mode."""
        hm = hierarchical_memory_default
        hm.build(input_data_medium.shape)

        # Get initial weights for all levels
        initial_weights = [som.weights_map.numpy().copy() for som in hm.som_layers]

        # Forward pass in training mode
        bmu_indices_list, quantization_errors_list = hm(input_data_medium, training=True)

        # Check that weights have been updated for all levels
        for i, som in enumerate(hm.som_layers):
            updated_weights = som.weights_map.numpy()
            assert not np.allclose(initial_weights[i], updated_weights)
            assert som.iterations.numpy() > 0

    def test_inference_mode_no_weight_updates(self, hierarchical_memory_default, input_data_medium):
        """Test that weights are not updated during inference mode."""
        hm = hierarchical_memory_default
        hm.build(input_data_medium.shape)

        # Get initial weights for all levels
        initial_weights = [som.weights_map.numpy().copy() for som in hm.som_layers]

        # Forward pass in inference mode
        bmu_indices_list, quantization_errors_list = hm(input_data_medium, training=False)

        # Check that weights have not been updated for any level
        for i, som in enumerate(hm.som_layers):
            updated_weights = som.weights_map.numpy()
            assert np.allclose(initial_weights[i], updated_weights)
            assert som.iterations.numpy() == 0

    def test_get_level_weights(self, hierarchical_memory_default, input_data_medium):
        """Test get_level_weights method."""
        hm = hierarchical_memory_default
        hm(input_data_medium, training=False)  # Build the system

        # Test getting weights for each level
        for level in range(3):
            weights = hm.get_level_weights(level)
            expected_shape = hm.som_layers[level].weights_map.shape
            assert weights.shape == expected_shape
            assert tf.reduce_all(tf.equal(weights, hm.som_layers[level].weights_map))

        # Test invalid level
        with pytest.raises(ValueError, match="Level must be between 0 and 2"):
            hm.get_level_weights(3)

        with pytest.raises(ValueError, match="Level must be between 0 and 2"):
            hm.get_level_weights(-1)

    def test_get_all_weights(self, hierarchical_memory_default, input_data_medium):
        """Test get_all_weights method."""
        hm = hierarchical_memory_default
        hm(input_data_medium, training=False)

        all_weights = hm.get_all_weights()
        assert len(all_weights) == 3

        # Check that each weight tensor matches the corresponding SOM layer
        for i, weights in enumerate(all_weights):
            assert weights.shape == hm.som_layers[i].weights_map.shape
            assert tf.reduce_all(tf.equal(weights, hm.som_layers[i].weights_map))

    def test_get_grid_shapes(self, hierarchical_memory_default, input_data_medium):
        """Test get_grid_shapes method."""
        hm = hierarchical_memory_default
        hm(input_data_medium, training=False)

        grid_shapes = hm.get_grid_shapes()
        expected_shapes = [(5, 5), (10, 10), (20, 20)]

        assert len(grid_shapes) == 3
        assert grid_shapes == expected_shapes

    def test_compute_output_shape_default(self, hierarchical_memory_default):
        """Test compute_output_shape for default configuration."""
        input_shape = (32, 128)
        bmu_shapes, error_shapes = hierarchical_memory_default.compute_output_shape(input_shape)

        assert len(bmu_shapes) == 3
        assert len(error_shapes) == 3

        # All levels should have same BMU and error shapes
        for bmu_shape, error_shape in zip(bmu_shapes, error_shapes):
            assert bmu_shape == (32, 2)  # batch_size, grid_dimensions
            assert error_shape == (32,)  # batch_size

    def test_compute_output_shape_1d(self, hierarchical_memory_1d):
        """Test compute_output_shape for 1D configuration."""
        input_shape = (16, 64)
        bmu_shapes, error_shapes = hierarchical_memory_1d.compute_output_shape(input_shape)

        assert len(bmu_shapes) == 2
        assert len(error_shapes) == 2

        for bmu_shape, error_shape in zip(bmu_shapes, error_shapes):
            assert bmu_shape == (16, 1)  # batch_size, 1 dimension
            assert error_shape == (16,)

    def test_compute_output_shape_3d(self, hierarchical_memory_3d):
        """Test compute_output_shape for 3D configuration."""
        input_shape = (24, 256)
        bmu_shapes, error_shapes = hierarchical_memory_3d.compute_output_shape(input_shape)

        assert len(bmu_shapes) == 2
        assert len(error_shapes) == 2

        for bmu_shape, error_shape in zip(bmu_shapes, error_shapes):
            assert bmu_shape == (24, 3)  # batch_size, 3 dimensions
            assert error_shape == (24,)

    def test_hierarchical_structure_progression(self, hierarchical_memory_custom):
        """Test that hierarchical structure follows expected progression."""
        # Build with some input
        test_input = keras.random.uniform([8, 200])
        hierarchical_memory_custom.build(test_input.shape)

        # Check grid size progression: 4 -> 7.2->7 -> 12.96->12 -> 23.328->23
        expected_sizes = [4, 7, 12, 23]  # int(4 * 1.8^n)
        expected_shapes = [(s, s) for s in expected_sizes]

        grid_shapes = hierarchical_memory_custom.get_grid_shapes()
        assert len(grid_shapes) == 4
        assert grid_shapes == expected_shapes

    def test_different_expansion_factors(self):
        """Test hierarchical memory with different expansion factors."""
        test_cases = [
            (1.5, [(5,), (7,), (11,)]),  # 1D with 1.5x expansion: 5, 7.5->7, 11.25->11
            (3.0, [(3, 3), (9, 9), (27, 27)]),  # 2D with 3.0x expansion: 3, 9, 27
            (1.2, [(10, 10), (12, 12), (14, 14)])  # 2D with 1.2x expansion: 10, 12, 14.4->14
        ]

        for expansion_factor, expected_shapes in test_cases:
            grid_dim = len(expected_shapes[0])
            hm = HierarchicalMemorySystem(
                input_dim=50,
                levels=3,
                grid_dimensions=grid_dim,
                base_grid_size=expected_shapes[0][0],
                grid_expansion_factor=expansion_factor
            )

            test_input = keras.random.uniform([4, 50])
            hm.build(test_input.shape)

            grid_shapes = hm.get_grid_shapes()
            assert grid_shapes == expected_shapes

    def test_regularization(self):
        """Test that regularization is properly applied to all levels."""
        hm = HierarchicalMemorySystem(
            input_dim=100,
            levels=2,
            regularizer=keras.regularizers.L2(0.01)
        )

        test_input = keras.random.uniform([8, 100])

        # No losses before calling the layer
        assert len(hm.losses) == 0

        # Apply the layer
        bmu_indices_list, quantization_errors_list = hm(test_input, training=True)

        # Should have regularization losses now (one per SOM layer)
        assert len(hm.losses) >= 2

    def test_serialization(self, hierarchical_memory_default):
        """Test serialization and deserialization of the hierarchical memory system."""
        hm = hierarchical_memory_default
        hm.build((32, 128))

        # Get configs
        config = hm.get_config()
        build_config = hm.get_build_config()

        # Recreate the system
        recreated_hm = HierarchicalMemorySystem.from_config(config)
        recreated_hm.build_from_config(build_config)

        # Check configuration matches
        assert recreated_hm.input_dim == hm.input_dim
        assert recreated_hm.levels == hm.levels
        assert recreated_hm.grid_dimensions == hm.grid_dimensions
        assert recreated_hm.base_grid_size == hm.base_grid_size
        assert recreated_hm.grid_expansion_factor == hm.grid_expansion_factor
        assert recreated_hm.initial_learning_rate == hm.initial_learning_rate
        assert recreated_hm.sigma == hm.sigma
        assert recreated_hm.neighborhood_function == hm.neighborhood_function

        # Check that SOM layers have matching configurations
        for orig_som, recreated_som in zip(hm.som_layers, recreated_hm.som_layers):
            assert orig_som.grid_shape == recreated_som.grid_shape
            assert orig_som.input_dim == recreated_som.input_dim

    def test_serialization_with_custom_objects(self):
        """Test serialization with custom initializers and regularizers."""
        original_hm = HierarchicalMemorySystem(
            input_dim=150,
            levels=2,
            weights_initializer=keras.initializers.HeNormal(),
            regularizer=keras.regularizers.L1(0.01)
        )

        # Build the system
        original_hm.build((16, 150))

        # Get configs
        config = original_hm.get_config()
        build_config = original_hm.get_build_config()

        # Recreate the system
        recreated_hm = HierarchicalMemorySystem.from_config(config)
        recreated_hm.build_from_config(build_config)

        # Check configuration matches
        assert recreated_hm.input_dim == original_hm.input_dim
        assert recreated_hm.levels == original_hm.levels

    def test_model_integration(self, input_data_medium):
        """Test the hierarchical memory system in a model context."""
        # Create a simple model with the hierarchical memory system
        inputs = keras.Input(shape=(128,))
        bmu_indices_list, quantization_errors_list = HierarchicalMemorySystem(
            input_dim=128,
            levels=2,
            base_grid_size=4
        )(inputs)

        # Use outputs from the first level for a simple downstream task
        x = keras.layers.Flatten()(bmu_indices_list[0])
        x = keras.layers.Dense(10)(x)
        outputs = keras.layers.Dense(1)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Test forward pass
        predictions = model(input_data_medium[:8], training=False)
        assert predictions.shape == (8, 1)

    def test_model_save_load(self, input_data_medium):
        """Test saving and loading a model with the hierarchical memory system."""
        # Create a model with the hierarchical memory system
        inputs = keras.Input(shape=(128,))
        bmu_indices_list, quantization_errors_list = HierarchicalMemorySystem(
            input_dim=128,
            levels=2,
            base_grid_size=6,
            name="hierarchical_memory"
        )(inputs)

        # Add simple downstream processing
        x = keras.layers.Flatten()(bmu_indices_list[0])
        x = keras.layers.Dense(10)(x)
        outputs = keras.layers.Dense(1)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Generate predictions before saving
        test_data = input_data_medium[:8]
        original_prediction = model.predict(test_data)

        # Create temporary directory for model
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "model.keras")

            # Save the model
            model.save(model_path)

            # Load the model
            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={"HierarchicalMemorySystem": HierarchicalMemorySystem}
            )

            # Generate prediction with loaded model
            loaded_prediction = loaded_model.predict(test_data)

            # Check predictions match
            assert np.allclose(original_prediction, loaded_prediction, rtol=1e-5)

            # Check layer type is preserved
            hm_layer = loaded_model.get_layer("hierarchical_memory")
            assert isinstance(hm_layer, HierarchicalMemorySystem)

    def test_numerical_stability(self):
        """Test hierarchical memory system stability with extreme input values."""
        hm = HierarchicalMemorySystem(
            input_dim=20,
            levels=2,
            base_grid_size=3
        )

        # Create inputs with different magnitudes
        test_cases = [
            keras.ops.zeros((4, 20)),  # Zeros
            keras.ops.ones((4, 20)) * 1e-10,  # Very small values
            keras.ops.ones((4, 20)) * 1e10,  # Very large values
            keras.random.normal((4, 20)) * 1e5  # Large random values
        ]

        for test_input in test_cases:
            bmu_indices_list, quantization_errors_list = hm(test_input, training=True)

            # Check for NaN/Inf values in all levels
            for i, q_errors in enumerate(quantization_errors_list):
                assert not tf.reduce_any(tf.math.is_nan(q_errors)).numpy(), f"NaN found in level {i}"
                assert not tf.reduce_any(tf.math.is_inf(q_errors)).numpy(), f"Inf found in level {i}"

            # Check BMU indices are valid for all levels
            grid_sizes = [3, 6]  # 3 and 3*2=6
            for i, bmu_indices in enumerate(bmu_indices_list):
                assert tf.reduce_all(bmu_indices >= 0).numpy(), f"Negative BMU in level {i}"
                assert tf.reduce_all(bmu_indices < grid_sizes[i]).numpy(), f"BMU out of range in level {i}"

    def test_gradient_flow(self, input_data_medium):
        """Test that gradients flow properly through the hierarchical system."""
        hm = HierarchicalMemorySystem(
            input_dim=128,
            levels=2,
            base_grid_size=4
        )

        # Create a simple model for gradient testing
        inputs = keras.Input(shape=(128,))
        _, quantization_errors_list = hm(inputs)

        # Create a loss based on quantization errors from first level
        loss = keras.layers.Lambda(lambda x: keras.ops.mean(x))(quantization_errors_list[0])
        model = keras.Model(inputs=inputs, outputs=loss)

        # Test gradient computation
        with tf.GradientTape() as tape:
            tape.watch(input_data_medium)
            output = model(input_data_medium, training=True)

        # Get gradients with respect to inputs
        gradients = tape.gradient(output, input_data_medium)

        # Gradients should exist
        assert gradients is not None

    def test_edge_case_single_level(self, hierarchical_memory_single_level):
        """Test hierarchical memory system with single level."""
        test_input = keras.random.uniform([8, 100])
        bmu_indices_list, quantization_errors_list = hierarchical_memory_single_level(test_input, training=True)

        # Should have exactly one level
        assert len(bmu_indices_list) == 1
        assert len(quantization_errors_list) == 1

        # Check shapes
        assert bmu_indices_list[0].shape == (8, 2)
        assert quantization_errors_list[0].shape == (8,)

    def test_large_expansion_factor(self):
        """Test with large expansion factor."""
        hm = HierarchicalMemorySystem(
            input_dim=50,
            levels=3,
            base_grid_size=2,
            grid_expansion_factor=5.0  # Large expansion
        )

        test_input = keras.random.uniform([4, 50])
        hm.build(test_input.shape)

        # Check grid progression: 2x2 -> 10x10 -> 50x50
        expected_shapes = [(2, 2), (10, 10), (50, 50)]
        grid_shapes = hm.get_grid_shapes()
        assert grid_shapes == expected_shapes

        # Test forward pass
        bmu_indices_list, quantization_errors_list = hm(test_input, training=True)

        # Should work without errors
        assert len(bmu_indices_list) == 3
        assert len(quantization_errors_list) == 3

    def test_small_expansion_factor(self):
        """Test with small expansion factor."""
        hm = HierarchicalMemorySystem(
            input_dim=30,
            levels=4,
            base_grid_size=10,
            grid_expansion_factor=1.1  # Small expansion
        )

        test_input = keras.random.uniform([6, 30])
        hm.build(test_input.shape)

        # Check grid progression: 10x10 -> 11x11 -> 12x12 -> 13x13
        expected_shapes = [(10, 10), (11, 11), (12, 12), (13, 13)]
        grid_shapes = hm.get_grid_shapes()
        assert grid_shapes == expected_shapes

        # Test forward pass
        bmu_indices_list, quantization_errors_list = hm(test_input, training=True)
        assert len(bmu_indices_list) == 4

    def test_different_neighborhood_functions(self):
        """Test hierarchical memory with different neighborhood functions."""
        for neighborhood_func in ['gaussian', 'bubble']:
            hm = HierarchicalMemorySystem(
                input_dim=40,
                levels=2,
                base_grid_size=5,
                neighborhood_function=neighborhood_func
            )

            test_input = keras.random.uniform([6, 40])
            bmu_indices_list, quantization_errors_list = hm(test_input, training=True)

            # Should work with both neighborhood functions
            assert len(bmu_indices_list) == 2
            assert len(quantization_errors_list) == 2

    def test_learning_progression_all_levels(self, hierarchical_memory_default, input_data_medium):
        """Test that learning progresses at all hierarchy levels."""
        hm = hierarchical_memory_default

        # Get initial quantization errors for all levels
        _, initial_errors_list = hm(input_data_medium, training=True)
        initial_mean_errors = [tf.reduce_mean(errors) for errors in initial_errors_list]

        # Train for multiple iterations
        for _ in range(5):
            hm(input_data_medium, training=True)

        # Get final quantization errors for all levels
        _, final_errors_list = hm(input_data_medium, training=False)
        final_mean_errors = [tf.reduce_mean(errors) for errors in final_errors_list]

        # Check that all levels are functioning (errors are non-negative)
        for i, final_error in enumerate(final_mean_errors):
            assert final_error >= 0, f"Negative error in level {i}"

        # Check that all SOM layers have been updated
        for i, som_layer in enumerate(hm.som_layers):
            assert som_layer.iterations.numpy() > 0, f"Level {i} was not trained"

    def test_consistent_bmu_finding_all_levels(self, hierarchical_memory_default):
        """Test that BMU finding is consistent across all levels for the same input."""
        hm = hierarchical_memory_default

        # Create fixed input
        fixed_input = keras.ops.ones((1, 128))

        # Find BMUs multiple times
        bmu_list1, _ = hm(fixed_input, training=False)
        bmu_list2, _ = hm(fixed_input, training=False)
        bmu_list3, _ = hm(fixed_input, training=False)

        # Should be the same BMUs each time for all levels (no training mode)
        for i in range(len(bmu_list1)):
            assert tf.reduce_all(tf.equal(bmu_list1[i], bmu_list2[i])), f"Inconsistent BMU in level {i}"
            assert tf.reduce_all(tf.equal(bmu_list2[i], bmu_list3[i])), f"Inconsistent BMU in level {i}"

    def test_multiple_training_sessions(self, hierarchical_memory_default, input_data_medium):
        """Test multiple training sessions with the same hierarchical memory system."""
        hm = hierarchical_memory_default

        # Build the system by calling it once
        hm(input_data_medium[:1], training=False)

        # Get initial iterations for all levels
        initial_iterations = [som.iterations.numpy() for som in hm.som_layers]

        # First training session
        hm(input_data_medium, training=True)
        first_iterations = [som.iterations.numpy() for som in hm.som_layers]

        # Second training session
        hm(input_data_medium, training=True)
        second_iterations = [som.iterations.numpy() for som in hm.som_layers]

        # Iterations should increase for all levels
        for i in range(len(hm.som_layers)):
            assert first_iterations[i] > initial_iterations[i], f"Level {i} not trained in first session"
            assert second_iterations[i] > first_iterations[i], f"Level {i} not trained in second session"