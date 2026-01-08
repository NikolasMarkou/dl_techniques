"""
Comprehensive test suite for PRISMModel.

Tests cover instantiation, forward pass, serialization, configuration,
shape inference, and integration with training pipeline.
"""

import os
import keras
import pytest
import tempfile
import numpy as np
from typing import Dict, Any, Tuple

from dl_techniques.models.prism.model import PRISMModel


class TestPRISMModelInstantiation:
    """Test model instantiation and configuration validation."""

    @pytest.fixture
    def model_config(self) -> Dict[str, Any]:
        """Default model configuration."""
        return {
            "context_len": 96,
            "forecast_len": 24,
            "num_features": 7,
            "hidden_dim": 64,
            "num_layers": 2,
            "tree_depth": 2,
            "overlap_ratio": 0.25,
            "num_wavelet_levels": 3,
            "router_hidden_dim": 64,
            "router_temperature": 1.0,
            "dropout_rate": 0.1,
            "ffn_expansion": 4,
            "kernel_initializer": "glorot_uniform",
        }

    def test_valid_instantiation(self, model_config: Dict[str, Any]) -> None:
        """Test model can be instantiated with valid config."""
        model = PRISMModel(**model_config)

        assert model.context_len == model_config["context_len"]
        assert model.forecast_len == model_config["forecast_len"]
        assert model.num_features == model_config["num_features"]
        assert model.hidden_dim == model_config["hidden_dim"]
        assert model.num_layers == model_config["num_layers"]
        assert len(model.prism_layers) == model_config["num_layers"]

    def test_hidden_dim_defaults_to_num_features(self) -> None:
        """Test hidden_dim defaults to num_features when not specified."""
        model = PRISMModel(
            context_len=96,
            forecast_len=24,
            num_features=7,
            hidden_dim=None,
        )

        assert model.hidden_dim == 7

    def test_invalid_context_len(self) -> None:
        """Test model rejects invalid context_len."""
        with pytest.raises(ValueError, match="context_len must be > 0"):
            PRISMModel(
                context_len=0,
                forecast_len=24,
                num_features=7,
            )

        with pytest.raises(ValueError, match="context_len must be > 0"):
            PRISMModel(
                context_len=-10,
                forecast_len=24,
                num_features=7,
            )

    def test_invalid_forecast_len(self) -> None:
        """Test model rejects invalid forecast_len."""
        with pytest.raises(ValueError, match="forecast_len must be > 0"):
            PRISMModel(
                context_len=96,
                forecast_len=0,
                num_features=7,
            )

    def test_invalid_num_features(self) -> None:
        """Test model rejects invalid num_features."""
        with pytest.raises(ValueError, match="num_features must be > 0"):
            PRISMModel(
                context_len=96,
                forecast_len=24,
                num_features=-5,
            )


class TestPRISMModelForwardPass:
    """Test forward pass and output shapes."""

    @pytest.fixture
    def model_config(self) -> Dict[str, Any]:
        """Default model configuration."""
        return {
            "context_len": 96,
            "forecast_len": 24,
            "num_features": 7,
            "hidden_dim": 64,
            "num_layers": 2,
        }

    @pytest.fixture
    def sample_input(self, model_config: Dict[str, Any]) -> np.ndarray:
        """Generate sample input tensor."""
        batch_size = 8
        return np.random.randn(
            batch_size,
            model_config["context_len"],
            model_config["num_features"]
        ).astype(np.float32)

    def test_forward_pass_output_shape(
            self,
            model_config: Dict[str, Any],
            sample_input: np.ndarray
    ) -> None:
        """Test forward pass produces correct output shape."""
        model = PRISMModel(**model_config)
        output = model(sample_input)

        expected_shape = (
            sample_input.shape[0],
            model_config["forecast_len"],
            model_config["num_features"]
        )
        assert output.shape == expected_shape

    def test_forward_pass_dtype(
            self,
            model_config: Dict[str, Any],
            sample_input: np.ndarray
    ) -> None:
        """Test output has correct dtype."""
        model = PRISMModel(**model_config)
        output = model(sample_input)

        assert output.dtype == sample_input.dtype

    @pytest.mark.parametrize("batch_size", [1, 4, 16, 32])
    def test_variable_batch_size(
            self,
            model_config: Dict[str, Any],
            batch_size: int
    ) -> None:
        """Test model handles various batch sizes."""
        model = PRISMModel(**model_config)

        inputs = np.random.randn(
            batch_size,
            model_config["context_len"],
            model_config["num_features"]
        ).astype(np.float32)

        output = model(inputs)

        assert output.shape[0] == batch_size
        assert output.shape[1] == model_config["forecast_len"]
        assert output.shape[2] == model_config["num_features"]

    def test_training_vs_inference_mode(
            self,
            model_config: Dict[str, Any],
            sample_input: np.ndarray
    ) -> None:
        """Test model behaves consistently in training vs inference mode."""
        model = PRISMModel(**model_config)

        # Build model first
        _ = model(sample_input, training=False)

        # Get outputs in both modes
        train_output = model(sample_input, training=True)
        infer_output = model(sample_input, training=False)

        # Shapes should match
        assert train_output.shape == infer_output.shape

        # Note: Outputs may differ due to dropout, so we only check shapes
        # If dropout_rate=0, outputs should be identical

    def test_forward_pass_no_dropout(
            self,
            model_config: Dict[str, Any],
            sample_input: np.ndarray
    ) -> None:
        """Test training vs inference outputs match when dropout=0."""
        config = model_config.copy()
        config["dropout_rate"] = 0.0

        model = PRISMModel(**config)

        train_output = model(sample_input, training=True)
        infer_output = model(sample_input, training=False)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(train_output),
            keras.ops.convert_to_numpy(infer_output),
            rtol=1e-6,
            atol=1e-6,
            err_msg="Outputs should match when dropout=0"
        )


class TestPRISMModelShapeInference:
    """Test compute_output_shape functionality."""

    def test_compute_output_shape(self) -> None:
        """Test compute_output_shape returns correct shape."""
        model = PRISMModel(
            context_len=96,
            forecast_len=24,
            num_features=7,
        )

        input_shape = (None, 96, 7)
        computed_shape = model.compute_output_shape(input_shape)

        assert computed_shape == (None, 24, 7)

    def test_compute_output_shape_before_build(self) -> None:
        """Test compute_output_shape works before model is built."""
        model = PRISMModel(
            context_len=96,
            forecast_len=24,
            num_features=7,
        )

        # Should work even without calling model
        input_shape = (None, 96, 7)
        computed_shape = model.compute_output_shape(input_shape)

        assert computed_shape == (None, 24, 7)

    def test_compute_output_shape_matches_actual(self) -> None:
        """Test compute_output_shape matches actual forward pass output."""
        model = PRISMModel(
            context_len=96,
            forecast_len=24,
            num_features=7,
        )

        inputs = np.random.randn(8, 96, 7).astype(np.float32)

        computed_shape = model.compute_output_shape(inputs.shape)
        actual_output = model(inputs)

        assert computed_shape == actual_output.shape

    @pytest.mark.parametrize(
        "context_len,forecast_len,num_features",
        [(48, 12, 3), (96, 24, 7), (192, 48, 12), (336, 96, 21)]
    )
    def test_compute_output_shape_various_configs(
            self,
            context_len: int,
            forecast_len: int,
            num_features: int
    ) -> None:
        """Test compute_output_shape with various configurations."""
        model = PRISMModel(
            context_len=context_len,
            forecast_len=forecast_len,
            num_features=num_features,
        )

        batch_size = 16
        input_shape = (batch_size, context_len, num_features)
        computed_shape = model.compute_output_shape(input_shape)

        assert computed_shape == (batch_size, forecast_len, num_features)


class TestPRISMModelSerialization:
    """Test model serialization and deserialization."""

    @pytest.fixture
    def model_config(self) -> Dict[str, Any]:
        """Default model configuration."""
        return {
            "context_len": 96,
            "forecast_len": 24,
            "num_features": 7,
            "hidden_dim": 64,
            "num_layers": 2,
            "tree_depth": 2,
            "overlap_ratio": 0.25,
            "num_wavelet_levels": 3,
            "router_hidden_dim": 64,
            "router_temperature": 1.0,
            "dropout_rate": 0.1,
            "ffn_expansion": 4,
        }

    @pytest.fixture
    def sample_input(self, model_config: Dict[str, Any]) -> np.ndarray:
        """Generate sample input tensor."""
        return np.random.randn(
            8,
            model_config["context_len"],
            model_config["num_features"]
        ).astype(np.float32)

    def test_get_config_complete(self, model_config: Dict[str, Any]) -> None:
        """Test get_config returns all constructor arguments."""
        model = PRISMModel(**model_config)
        config = model.get_config()

        # Check all required parameters are present
        required_keys = [
            "context_len",
            "forecast_len",
            "num_features",
            "hidden_dim",
            "num_layers",
            "tree_depth",
            "overlap_ratio",
            "num_wavelet_levels",
            "router_hidden_dim",
            "router_temperature",
            "dropout_rate",
            "ffn_expansion",
            "kernel_initializer",
            "kernel_regularizer",
        ]

        for key in required_keys:
            assert key in config, f"Missing key in config: {key}"

    def test_from_config_reconstruction(
            self,
            model_config: Dict[str, Any],
            sample_input: np.ndarray
    ) -> None:
        """Test model can be reconstructed from config."""
        original = PRISMModel(**model_config)
        _ = original(sample_input)  # Build model

        config = original.get_config()
        reconstructed = PRISMModel.from_config(config)

        assert reconstructed.context_len == original.context_len
        assert reconstructed.forecast_len == original.forecast_len
        assert reconstructed.num_features == original.num_features
        assert reconstructed.hidden_dim == original.hidden_dim
        assert reconstructed.num_layers == original.num_layers
        assert reconstructed.tree_depth == original.tree_depth

    def test_serialization_cycle(
            self,
            model_config: Dict[str, Any],
            sample_input: np.ndarray
    ) -> None:
        """Test full save/load cycle preserves functionality."""
        model = PRISMModel(**model_config)

        # Get original output
        original_output = model(sample_input, training=False)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_prism_model.keras")
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)

        # Get loaded output
        loaded_output = loaded_model(sample_input, training=False)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(original_output),
            keras.ops.convert_to_numpy(loaded_output),
            rtol=1e-6,
            atol=1e-6,
            err_msg="Outputs should match after serialization"
        )

    def test_serialization_with_custom_initializer(
            self,
            sample_input: np.ndarray
    ) -> None:
        """Test serialization with custom kernel initializer."""
        from keras import initializers

        model = PRISMModel(
            context_len=96,
            forecast_len=24,
            num_features=7,
            kernel_initializer=initializers.HeNormal(),
        )

        _ = model(sample_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_model.keras")
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)

        # Verify initializer was preserved
        config = loaded_model.get_config()
        assert "kernel_initializer" in config

    def test_serialization_with_regularizer(
            self,
            sample_input: np.ndarray
    ) -> None:
        """Test serialization with kernel regularizer."""
        from keras import regularizers

        model = PRISMModel(
            context_len=96,
            forecast_len=24,
            num_features=7,
            kernel_regularizer=regularizers.L2(0.01),
        )

        _ = model(sample_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_model.keras")
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)

        # Verify regularizer was preserved
        config = loaded_model.get_config()
        assert "kernel_regularizer" in config


class TestPRISMModelPresets:
    """Test preset configurations."""

    @pytest.mark.parametrize(
        "preset",
        ["tiny", "small", "base", "large"]
    )
    def test_preset_creation(self, preset: str) -> None:
        """Test model can be created from presets."""
        model = PRISMModel.from_preset(
            preset=preset,
            context_len=96,
            forecast_len=24,
            num_features=7,
        )

        assert model.context_len == 96
        assert model.forecast_len == 24
        assert model.num_features == 7

        # Check preset-specific configurations
        preset_config = PRISMModel.PRESETS[preset]
        assert model.hidden_dim == preset_config["hidden_dim"]
        assert model.num_layers == preset_config["num_layers"]
        assert model.tree_depth == preset_config["tree_depth"]

    def test_preset_with_override(self) -> None:
        """Test preset parameters can be overridden."""
        model = PRISMModel.from_preset(
            preset="small",
            context_len=96,
            forecast_len=24,
            num_features=7,
            hidden_dim=128,  # Override preset value
            num_layers=3,  # Override preset value
        )

        assert model.hidden_dim == 128
        assert model.num_layers == 3

    def test_invalid_preset_raises_error(self) -> None:
        """Test invalid preset name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown preset"):
            PRISMModel.from_preset(
                preset="nonexistent",
                context_len=96,
                forecast_len=24,
                num_features=7,
            )

    def test_all_presets_functional(self) -> None:
        """Test all presets create functional models."""
        inputs = np.random.randn(4, 96, 7).astype(np.float32)

        for preset in PRISMModel.PRESETS.keys():
            model = PRISMModel.from_preset(
                preset=preset,
                context_len=96,
                forecast_len=24,
                num_features=7,
            )

            output = model(inputs)
            assert output.shape == (4, 24, 7)


class TestPRISMModelBuild:
    """Test model building and weight creation."""

    def test_model_builds_correctly(self) -> None:
        """Test model.build() creates all expected layers."""
        model = PRISMModel(
            context_len=96,
            forecast_len=24,
            num_features=7,
            hidden_dim=64,
            num_layers=2,
        )

        input_shape = (None, 96, 7)
        model.build(input_shape)

        assert model.built
        assert model.input_projection.built

        for layer in model.prism_layers:
            assert layer.built

        assert model.flatten.built
        assert model.forecast_head.built

    def test_weights_created_after_build(self) -> None:
        """Test weights are created after build."""
        model = PRISMModel(
            context_len=96,
            forecast_len=24,
            num_features=7,
        )

        inputs = np.random.randn(4, 96, 7).astype(np.float32)
        _ = model(inputs)  # Triggers build

        weights = model.get_weights()
        assert len(weights) > 0

    def test_model_summary_works(self) -> None:
        """Test model.summary() works after building."""
        model = PRISMModel(
            context_len=96,
            forecast_len=24,
            num_features=7,
            hidden_dim=32,
            num_layers=1,
        )

        inputs = np.random.randn(4, 96, 7).astype(np.float32)
        _ = model(inputs)

        # Should not raise an error
        model.summary()


class TestPRISMModelIntegration:
    """Integration tests with training pipeline."""

    @pytest.fixture
    def training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic training data."""
        context_len = 96
        forecast_len = 24
        num_features = 7
        num_samples = 100

        x_train = np.random.randn(
            num_samples, context_len, num_features
        ).astype(np.float32)

        y_train = np.random.randn(
            num_samples, forecast_len, num_features
        ).astype(np.float32)

        return x_train, y_train

    def test_compile_and_train(
            self,
            training_data: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test model can be compiled and trained."""
        x_train, y_train = training_data

        model = PRISMModel(
            context_len=96,
            forecast_len=24,
            num_features=7,
            hidden_dim=32,
            num_layers=1,
        )

        model.compile(
            optimizer="adam",
            loss="mse",
            metrics=["mae"]
        )

        history = model.fit(
            x_train,
            y_train,
            epochs=2,
            batch_size=16,
            validation_split=0.2,
            verbose=0
        )

        assert "loss" in history.history
        assert "mae" in history.history
        assert len(history.history["loss"]) == 2

    def test_save_trained_model(
            self,
            training_data: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test trained model can be saved and loaded."""
        x_train, y_train = training_data

        model = PRISMModel(
            context_len=96,
            forecast_len=24,
            num_features=7,
            hidden_dim=32,
            num_layers=1,
        )

        model.compile(optimizer="adam", loss="mse")
        model.fit(x_train, y_train, epochs=1, batch_size=16, verbose=0)

        # Get predictions before saving
        test_input = x_train[:5]
        pred_before = model.predict(test_input, verbose=0)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "trained_model.keras")
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)

        # Get predictions after loading
        pred_after = loaded_model.predict(test_input, verbose=0)

        np.testing.assert_allclose(
            pred_before,
            pred_after,
            rtol=1e-6,
            atol=1e-6,
            err_msg="Predictions should match after save/load"
        )

    def test_evaluate_on_test_data(
            self,
            training_data: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test model.evaluate() works correctly."""
        x_train, y_train = training_data

        model = PRISMModel(
            context_len=96,
            forecast_len=24,
            num_features=7,
            hidden_dim=32,
            num_layers=1,
        )

        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        model.fit(x_train, y_train, epochs=1, batch_size=16, verbose=0)

        # Evaluate
        test_loss, test_mae = model.evaluate(
            x_train[:20],
            y_train[:20],
            verbose=0
        )

        assert isinstance(test_loss, float)
        assert isinstance(test_mae, float)
        assert test_loss >= 0
        assert test_mae >= 0


class TestPRISMModelDifferentConfigurations:
    """Test model with various configurations."""

    @pytest.mark.parametrize(
        "context_len,forecast_len",
        [(48, 12), (96, 24), (192, 48), (336, 96)]
    )
    def test_different_sequence_lengths(
            self,
            context_len: int,
            forecast_len: int
    ) -> None:
        """Test model with different sequence lengths."""
        model = PRISMModel(
            context_len=context_len,
            forecast_len=forecast_len,
            num_features=7,
        )

        inputs = np.random.randn(4, context_len, 7).astype(np.float32)
        output = model(inputs)

        assert output.shape == (4, forecast_len, 7)

    @pytest.mark.parametrize("num_features", [1, 3, 7, 12, 21])
    def test_different_feature_dimensions(self, num_features: int) -> None:
        """Test model with different feature dimensions."""
        model = PRISMModel(
            context_len=96,
            forecast_len=24,
            num_features=num_features,
        )

        inputs = np.random.randn(4, 96, num_features).astype(np.float32)
        output = model(inputs)

        assert output.shape == (4, 24, num_features)

    @pytest.mark.parametrize("num_layers", [1, 2, 3, 4])
    def test_different_num_layers(self, num_layers: int) -> None:
        """Test model with different numbers of PRISM layers."""
        model = PRISMModel(
            context_len=96,
            forecast_len=24,
            num_features=7,
            num_layers=num_layers,
        )

        assert len(model.prism_layers) == num_layers

        inputs = np.random.randn(4, 96, 7).astype(np.float32)
        output = model(inputs)

        assert output.shape == (4, 24, 7)

    @pytest.mark.parametrize("tree_depth", [1, 2, 3])
    def test_different_tree_depths(self, tree_depth: int) -> None:
        """Test model with different tree depths."""
        model = PRISMModel(
            context_len=96,
            forecast_len=24,
            num_features=7,
            tree_depth=tree_depth,
        )

        inputs = np.random.randn(4, 96, 7).astype(np.float32)
        output = model(inputs)

        assert output.shape == (4, 24, 7)

    @pytest.mark.parametrize("dropout_rate", [0.0, 0.1, 0.3, 0.5])
    def test_different_dropout_rates(self, dropout_rate: float) -> None:
        """Test model with different dropout rates."""
        model = PRISMModel(
            context_len=96,
            forecast_len=24,
            num_features=7,
            dropout_rate=dropout_rate,
        )

        inputs = np.random.randn(4, 96, 7).astype(np.float32)
        output = model(inputs)

        assert output.shape == (4, 24, 7)

# ---------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------


if __name__ == "__main__":
    pytest.main([__file__, "-v"])