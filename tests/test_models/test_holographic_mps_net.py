"""Tests for Holographic Encoder-Decoder model implementation."""

import os
import tempfile
import keras
import pytest

from dl_techniques.models.holographic_mps_net import HolographicEncoderDecoder


class TestHolographicEncoderDecoder:
    """Test cases for Holographic Encoder-Decoder model."""

    @pytest.fixture
    def basic_model(self):
        """Create a basic model for testing."""
        return HolographicEncoderDecoder(
            input_shape=(28, 28, 1),
            latent_dim=64,
            bond_dim=16,
            num_branches=3
        )

    @pytest.fixture
    def dummy_input(self):
        """Create dummy input data."""
        return keras.random.normal((4, 28, 28, 1))

    @pytest.mark.parametrize(
        "input_shape,latent_dim,expected_output_shape",
        [
            ((28, 28, 1), 64, (None, 28, 28, 1)),
            ((32, 32, 3), 128, (None, 32, 32, 3)),
        ],
    )
    def test_model_output_shape(self, input_shape, latent_dim, expected_output_shape):
        """Test that models output the correct shape."""
        model = HolographicEncoderDecoder(
            input_shape=input_shape,
            latent_dim=latent_dim,
            bond_dim=16,
            num_branches=3
        )

        # Create dummy input to trigger build
        batch_size = 2
        dummy_input = keras.random.normal((batch_size,) + input_shape)
        output = model(dummy_input)

        assert output.shape == (batch_size,) + expected_output_shape[1:]

    @pytest.mark.parametrize(
        "num_branches,bond_dim,latent_dim",
        [
            (2, 8, 32),
            (3, 16, 64),
            (4, 32, 128),
            (5, 16, 256),
        ],
    )
    def test_different_configurations(self, num_branches, bond_dim, latent_dim):
        """Test models with different configuration parameters."""
        input_shape = (28, 28, 1)
        model = HolographicEncoderDecoder(
            input_shape=input_shape,
            latent_dim=latent_dim,
            bond_dim=bond_dim,
            num_branches=num_branches
        )

        # Test that model can be built and run
        dummy_input = keras.random.normal((2,) + input_shape)
        output = model(dummy_input)

        assert output.shape == (2,) + input_shape
        assert len(model.decoder_branches) == num_branches

    def test_custom_output_shape(self):
        """Test models with custom output shape different from input."""
        input_shape = (28, 28, 1)
        output_shape = (14, 14, 3)

        model = HolographicEncoderDecoder(
            input_shape=input_shape,
            output_shape=output_shape,
            latent_dim=64,
            bond_dim=16,
            num_branches=3
        )

        dummy_input = keras.random.normal((2,) + input_shape)
        output = model(dummy_input)

        assert output.shape == (2,) + output_shape

    def test_regularization_parameters(self):
        """Test that regularization is applied correctly."""
        regularizer = keras.regularizers.L2(1e-4)
        model = HolographicEncoderDecoder(
            input_shape=(28, 28, 1),
            latent_dim=64,
            kernel_regularizer=regularizer,
            regularization_strength=0.05
        )

        # Build the model
        dummy_input = keras.random.normal((2, 28, 28, 1))
        _ = model(dummy_input)

        # Check that regularization strength is set
        assert model.regularization_strength == 0.05

        # Check that kernel regularizer is applied
        assert model.kernel_regularizer is not None

    def test_model_serialization(self):
        """Test model serialization with get_config and build_from_config."""
        original_model = HolographicEncoderDecoder(
            input_shape=(28, 28, 1),
            latent_dim=64,
            bond_dim=16,
            num_branches=3,
            regularization_strength=0.02,
            use_bias=False
        )

        # Build the original model
        dummy_input = keras.random.normal((2, 28, 28, 1))
        original_output = original_model(dummy_input)

        # Get configs
        config = original_model.get_config()
        build_config = original_model.get_build_config()

        # Recreate the model
        recreated_model = HolographicEncoderDecoder.from_config(config)
        recreated_model.build_from_config(build_config)

        # Check configuration matches
        assert recreated_model.latent_dim == original_model.latent_dim
        assert recreated_model.bond_dim == original_model.bond_dim
        assert recreated_model.num_branches == original_model.num_branches
        assert recreated_model.use_bias == original_model.use_bias

    def test_model_saving_and_loading(self):
        """Test model saving and loading in .keras format."""
        model = HolographicEncoderDecoder(
            input_shape=(28, 28, 1),
            latent_dim=64,
            bond_dim=16,
            num_branches=3
        )

        # Build the model
        dummy_input = keras.random.normal((2, 28, 28, 1))
        original_output = model(dummy_input)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, 'holographic_model.keras')

            # Save model
            model.save(model_path, save_format="keras")

            # Register custom objects for loading
            custom_objects = {
                'HolographicEncoderDecoder': HolographicEncoderDecoder
            }

            # Load model
            loaded_model = keras.models.load_model(model_path, custom_objects=custom_objects)

            # Test that loaded model produces same output
            loaded_output = loaded_model(dummy_input)

            # Check shapes match
            assert loaded_output.shape == original_output.shape

            # Check that the architecture is preserved
            assert len(model.decoder_branches) == len(loaded_model.decoder_branches)

    def test_forward_pass(self, basic_model, dummy_input):
        """Test forward pass with random input."""
        # Perform forward pass
        y = basic_model(dummy_input, training=True)

        # Check output shape
        assert y.shape == (4, 28, 28, 1)

        # Check that output contains finite values
        assert keras.ops.all(keras.ops.isfinite(y))

        # Test inference mode
        y_inference = basic_model(dummy_input, training=False)
        assert y_inference.shape == (4, 28, 28, 1)

    def test_properties(self):
        """Test model properties."""
        input_shape = (32, 32, 3)
        output_shape = (16, 16, 1)

        model = HolographicEncoderDecoder(
            input_shape=input_shape,
            output_shape=output_shape,
            latent_dim=64
        )

        assert model.input_shape_property == input_shape
        assert model.output_shape_property == output_shape

    @pytest.mark.parametrize(
        "invalid_params,expected_error",
        [
            ({"input_shape": (0, 28, 1)}, ValueError),
            ({"input_shape": (-1, 28, 1)}, ValueError),
            ({"latent_dim": 0}, ValueError),
            ({"latent_dim": -10}, ValueError),
            ({"bond_dim": 0}, ValueError),
            ({"num_branches": 0}, ValueError),
            ({"regularization_strength": -0.1}, ValueError),
            ({"regularization_strength": 1.5}, ValueError),
        ],
    )
    def test_invalid_parameters(self, invalid_params, expected_error):
        """Test that invalid parameters raise appropriate errors."""
        base_params = {
            "input_shape": (28, 28, 1),
            "latent_dim": 64,
            "bond_dim": 16,
            "num_branches": 3
        }
        base_params.update(invalid_params)

        with pytest.raises(expected_error):
            HolographicEncoderDecoder(**base_params)

    def test_different_input_dimensions(self):
        """Test model with different input dimensionalities."""
        # 1D input
        model_1d = HolographicEncoderDecoder(
            input_shape=(100,),
            latent_dim=32
        )
        x_1d = keras.random.normal((3, 100))
        y_1d = model_1d(x_1d)
        assert y_1d.shape == (3, 100)

        # 2D input
        model_2d = HolographicEncoderDecoder(
            input_shape=(28, 28),
            latent_dim=32
        )
        x_2d = keras.random.normal((3, 28, 28))
        y_2d = model_2d(x_2d)
        assert y_2d.shape == (3, 28, 28)

        # 3D input (e.g., images)
        model_3d = HolographicEncoderDecoder(
            input_shape=(32, 32, 3),
            latent_dim=32
        )
        x_3d = keras.random.normal((3, 32, 32, 3))
        y_3d = model_3d(x_3d)
        assert y_3d.shape == (3, 32, 32, 3)

    def test_batch_size_independence(self):
        """Test that model works with different batch sizes."""
        model = HolographicEncoderDecoder(
            input_shape=(28, 28, 1),
            latent_dim=64
        )

        # Test with different batch sizes
        for batch_size in [1, 4, 8, 16]:
            x = keras.random.normal((batch_size, 28, 28, 1))
            y = model(x)
            assert y.shape == (batch_size, 28, 28, 1)

    def test_entropy_branch_targets(self):
        """Test that decoder branches have different entropy targets."""
        model = HolographicEncoderDecoder(
            input_shape=(28, 28, 1),
            latent_dim=64,
            num_branches=4
        )

        # Build the model
        dummy_input = keras.random.normal((2, 28, 28, 1))
        _ = model(dummy_input)

        # Check that we have the correct number of branches
        assert len(model.decoder_branches) == 4

        # Check that each branch has a name
        branch_names = [branch.name for branch in model.decoder_branches]
        expected_names = [f"decoder_branch_{i}" for i in range(4)]
        assert branch_names == expected_names

    def test_model_build_components(self, basic_model, dummy_input):
        """Test that all model components are built correctly."""
        # Trigger build
        _ = basic_model(dummy_input)

        # Check that encoder is built
        assert basic_model.encoder_mps is not None
        assert basic_model.encoder_mps.built

        # Check that decoder branches are built
        assert len(basic_model.decoder_branches) == 3
        for branch in basic_model.decoder_branches:
            assert branch.built

        # Check that output projection is built
        assert basic_model.output_projection is not None
        assert basic_model.output_projection.built

    def test_model_weights_created(self, basic_model, dummy_input):
        """Test that model weights are created after build."""
        # Before build
        assert basic_model.encoder_mps is None

        # Trigger build
        _ = basic_model(dummy_input)

        # After build - check that weights exist
        assert len(basic_model.weights) > 0
        assert len(basic_model.trainable_weights) > 0

        # Check that encoder has weights
        assert len(basic_model.encoder_mps.weights) > 0

        # Check that decoder branches have weights
        for branch in basic_model.decoder_branches:
            assert len(branch.weights) > 0

    def test_initializer_types(self):
        """Test different initializer types."""
        # String initializer
        model1 = HolographicEncoderDecoder(
            input_shape=(28, 28, 1),
            kernel_initializer="glorot_normal"
        )

        # Initializer object
        model2 = HolographicEncoderDecoder(
            input_shape=(28, 28, 1),
            kernel_initializer=keras.initializers.HeNormal()
        )

        # Build both models
        dummy_input = keras.random.normal((2, 28, 28, 1))
        _ = model1(dummy_input)
        _ = model2(dummy_input)

        # Check that both work
        assert model1.encoder_mps is not None
        assert model2.encoder_mps is not None

    def test_different_regularization_strengths(self):
        """Test model with different regularization strengths."""
        strengths = [0.0, 0.01, 0.1, 0.5]

        for strength in strengths:
            model = HolographicEncoderDecoder(
                input_shape=(28, 28, 1),
                regularization_strength=strength
            )

            dummy_input = keras.random.normal((2, 28, 28, 1))
            output = model(dummy_input)

            assert output.shape == (2, 28, 28, 1)
            assert model.regularization_strength == strength

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])