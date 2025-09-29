import pytest
import keras
from keras import ops
import numpy as np
import tensorflow as tf
import tempfile
import os
from typing import List

from dl_techniques.layers.fusion.multimodal_fusion import MultiModalFusion, FusionStrategy

SINGLE_OUTPUT_STRATEGIES: List[FusionStrategy] = [
    'concatenation',
    'addition',
    'multiplication',
    'gated',
    'attention_pooling',
    'bilinear',
    'tensor_fusion'
]


class TestMultiModalFusion:
    """Comprehensive test suite for the MultiModalFusion layer."""

    @pytest.fixture
    def dim(self) -> int:
        """Provides a standard dimension for testing."""
        return 64

    @pytest.fixture
    def sample_input(self, dim: int) -> List[keras.KerasTensor]:
        """Provides a sample multi-modal input (list of 2 tensors)."""
        batch_size, seq_len = 4, 16
        return [
            keras.random.normal(shape=(batch_size, seq_len, dim)),
            keras.random.normal(shape=(batch_size, seq_len, dim)),
        ]

    def test_initialization(self, dim: int):
        """Test layer initialization with default parameters."""
        layer = MultiModalFusion(dim=dim, fusion_strategy='concatenation')
        assert layer.dim == dim
        assert layer.fusion_strategy == 'concatenation'
        assert not layer.built

    @pytest.mark.parametrize("strategy", SINGLE_OUTPUT_STRATEGIES)
    def test_forward_pass_single_output(self, strategy: FusionStrategy, sample_input: List[keras.KerasTensor],
                                        dim: int):
        """Test forward pass and building for single-output strategies."""
        layer = MultiModalFusion(dim=dim, fusion_strategy=strategy)
        output = layer(sample_input)

        assert layer.built, f"Layer should be built after forward pass for strategy '{strategy}'"

        batch_size, seq_len = ops.shape(sample_input[0])[:2]

        if strategy == 'attention_pooling':
            # Special case: attention pooling reduces the sequence dimension
            expected_shape = (batch_size, dim)
        else:
            expected_shape = (batch_size, seq_len, dim)

        assert output.shape == expected_shape, f"Output shape mismatch for strategy '{strategy}'"

    def test_forward_pass_cross_attention(self, sample_input: List[keras.KerasTensor], dim: int):
        """Test forward pass for the cross-attention strategy which returns multiple outputs."""
        layer = MultiModalFusion(
            dim=dim,
            fusion_strategy='cross_attention',
            num_fusion_layers=2  # Test iterative fusion
        )
        outputs = layer(sample_input)

        assert layer.built
        assert isinstance(outputs, tuple)
        assert len(outputs) == len(sample_input)

        for i, out_tensor in enumerate(outputs):
            assert out_tensor.shape == sample_input[i].shape

    @pytest.mark.parametrize("strategy", SINGLE_OUTPUT_STRATEGIES)
    def test_serialization_cycle_single_output(self, strategy: FusionStrategy, sample_input: List[keras.KerasTensor],
                                               dim: int):
        """CRITICAL TEST: Full serialization cycle for single-output strategies."""
        layer_instance = MultiModalFusion(dim=dim, fusion_strategy=strategy)

        # Create a model with the custom layer
        inputs = [keras.Input(shape=s.shape[1:]) for s in sample_input]
        outputs = layer_instance(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)

        # Get original prediction
        original_pred = model(sample_input)

        # Save and load the model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, f'test_model_{strategy}.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input)

            # Verify identical predictions
            np.testing.assert_allclose(
                ops.convert_to_numpy(original_pred),
                ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg=f"Predictions differ after serialization for strategy '{strategy}'"
            )

    def test_serialization_cycle_cross_attention(self, sample_input: List[keras.KerasTensor], dim: int):
        """CRITICAL TEST: Full serialization cycle for cross-attention strategy."""
        layer_instance = MultiModalFusion(dim=dim, fusion_strategy='cross_attention')

        inputs = [keras.Input(shape=s.shape[1:]) for s in sample_input]
        outputs = layer_instance(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)

        original_preds = model(sample_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model_cross_attention.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_preds = loaded_model(sample_input)

            # Verify identical predictions for each output
            for orig, loaded in zip(original_preds, loaded_preds):
                np.testing.assert_allclose(
                    ops.convert_to_numpy(orig),
                    ops.convert_to_numpy(loaded),
                    rtol=1e-6, atol=1e-6,
                    err_msg="Predictions differ after serialization for cross-attention strategy"
                )

    def test_config_completeness(self, dim: int):
        """Test that get_config contains all __init__ parameters."""
        config_params = {
            'dim': dim,
            'fusion_strategy': 'cross_attention',
            'num_fusion_layers': 2,
            'attention_config': {'num_heads': 4, 'dropout_rate': 0.2},
            'ffn_type': 'mlp',
            'ffn_config': {'hidden_dim': dim * 2},
            'norm_type': 'layer_norm',
            'norm_config': {'epsilon': 1e-6},
            'dropout_rate': 0.15,
            'use_residual': False,
        }
        layer = MultiModalFusion(**config_params)
        config = layer.get_config()

        for key, value in config_params.items():
            assert key in config, f"Missing '{key}' in get_config()"
            if isinstance(value, dict):
                assert config[key] == value, f"Config mismatch for nested dict '{key}'"

        assert config['attention_config']['num_heads'] == 4
        assert config['ffn_config']['hidden_dim'] == dim * 2

    @pytest.mark.parametrize("strategy", SINGLE_OUTPUT_STRATEGIES)
    def test_gradients_flow_single_output(self, strategy: FusionStrategy, sample_input: List[keras.KerasTensor],
                                          dim: int):
        """Test that gradients can be computed for single-output strategies."""
        layer = MultiModalFusion(dim=dim, fusion_strategy=strategy)

        with tf.GradientTape() as tape:
            tape.watch(sample_input)
            output = layer(sample_input)
            loss = ops.mean(ops.square(output))

        gradients = tape.gradient(loss, layer.trainable_variables)

        assert all(g is not None for g in gradients), f"Gradient is None for strategy '{strategy}'"
        assert len(gradients) > 0, f"No trainable variables found for strategy '{strategy}'"

    def test_gradients_flow_cross_attention(self, sample_input: List[keras.KerasTensor], dim: int):
        """Test gradient computation for cross-attention strategy."""
        layer = MultiModalFusion(dim=dim, fusion_strategy='cross_attention')

        with tf.GradientTape() as tape:
            tape.watch(sample_input)
            outputs = layer(sample_input)
            # Sum losses from all outputs
            loss = sum(ops.mean(ops.square(o)) for o in outputs)

        gradients = tape.gradient(loss, layer.trainable_variables)

        assert all(g is not None for g in gradients)
        assert len(gradients) > 0

    @pytest.mark.parametrize("training", [True, False])
    @pytest.mark.parametrize("strategy", ['cross_attention', 'concatenation', 'gated'])
    def test_training_modes(self, strategy: FusionStrategy, sample_input: List[keras.KerasTensor], dim: int,
                            training: bool):
        """Test behavior in different training modes for strategies with dropout."""
        layer = MultiModalFusion(dim=dim, fusion_strategy=strategy, dropout_rate=0.5)

        # This test just ensures the call doesn't crash
        _ = layer(sample_input, training=training)
        assert True  # If we reach here, the call was successful

    def test_edge_cases_and_errors(self, sample_input, dim):
        """Test for expected error conditions."""
        # Invalid parameters
        with pytest.raises(ValueError):
            MultiModalFusion(dim=0)
        with pytest.raises(ValueError):
            MultiModalFusion(dim=dim, num_fusion_layers=0)
        with pytest.raises(ValueError):
            MultiModalFusion(dim=dim, dropout_rate=1.1)

        # num_fusion_layers > 1 for non-iterative strategy
        with pytest.raises(ValueError):
            MultiModalFusion(dim=dim, fusion_strategy='concatenation', num_fusion_layers=2)

        # Bilinear with != 2 inputs
        with pytest.raises(ValueError, match="Bilinear fusion requires exactly 2 modalities, got 3"):
            layer = MultiModalFusion(dim=dim, fusion_strategy='bilinear')
            three_inputs = sample_input + [keras.random.normal(shape=(4, 16, dim))]
            layer(three_inputs)

        # Input with < 2 modalities
        with pytest.raises(ValueError, match="Expected at least 2 modalities, got 1"):
            layer = MultiModalFusion(dim=dim)
            layer([sample_input[0]])

        # Dimension mismatch
        with pytest.raises(ValueError, match=f"Modality 1 dimension {dim * 2} doesn't match expected dim {dim}"):
            layer = MultiModalFusion(dim=dim)
            mismatched_input = [
                sample_input[0],
                keras.random.normal(shape=(4, 16, dim * 2))
            ]
            # Must build the layer to trigger the shape check
            layer.build([s.shape for s in mismatched_input])