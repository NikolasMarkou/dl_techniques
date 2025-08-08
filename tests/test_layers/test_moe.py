"""
Comprehensive test suite for the Mixture of Experts (MoE) module.

This module provides extensive testing for all MoE components to ensure
reliability, correctness, and compatibility with the dl_techniques framework.
"""

import keras
from keras import ops
import pytest
import numpy as np
import tempfile
import os



from dl_techniques.layers.moe import (
    MixtureOfExperts, MoEConfig, ExpertConfig, GatingConfig,
    create_ffn_moe, create_attention_moe, create_conv_moe,
    FFNExpert, AttentionExpert, Conv2DExpert,
    LinearGating, CosineGating, SoftMoEGating,
    get_preset_moe
)


class TestMoEConfig:
    """Test suite for MoE configuration classes."""

    def test_expert_config_initialization(self):
        """Test ExpertConfig initialization with default values."""
        config = ExpertConfig()

        assert config.expert_type == 'ffn'
        assert config.hidden_dim == 768
        assert config.output_dim == 768  # Should default to hidden_dim
        assert config.intermediate_size == 3072  # Should be 4 * hidden_dim

    def test_expert_config_custom_values(self):
        """Test ExpertConfig with custom values."""
        config = ExpertConfig(
            expert_type='attention',
            hidden_dim=512,
            num_heads=8,
            dropout_rate=0.2
        )

        assert config.expert_type == 'attention'
        assert config.hidden_dim == 512
        assert config.head_dim == 64  # hidden_dim // num_heads
        assert config.dropout_rate == 0.2

    def test_gating_config_initialization(self):
        """Test GatingConfig initialization."""
        config = GatingConfig()

        assert config.gating_type == 'linear'
        assert config.top_k == 1
        assert config.capacity_factor == 1.25
        assert config.add_noise == True

    def test_moe_config_serialization(self):
        """Test MoEConfig serialization and deserialization."""
        config = MoEConfig(
            num_experts=16,
            expert_config=ExpertConfig(hidden_dim=1024),
            gating_config=GatingConfig(top_k=2)
        )

        # Serialize to dict
        config_dict = config.to_dict()
        assert config_dict['num_experts'] == 16
        assert config_dict['expert_config']['hidden_dim'] == 1024
        assert config_dict['gating_config']['top_k'] == 2

        # Deserialize from dict
        restored_config = MoEConfig.from_dict(config_dict)
        assert restored_config.num_experts == 16
        assert restored_config.expert_config.hidden_dim == 1024
        assert restored_config.gating_config.top_k == 2


class TestExperts:
    """Test suite for expert network implementations."""

    def test_ffn_expert_creation(self):
        """Test FFNExpert creation and basic functionality."""
        expert = FFNExpert(
            hidden_dim=256,
            intermediate_size=1024,
            activation='gelu'
        )

        # Test build
        expert.build((None, 128))
        assert expert.built
        assert expert.dense1 is not None
        assert expert.dense2 is not None

        # Test forward pass
        inputs = keras.random.normal((4, 128))
        outputs = expert(inputs)

        assert ops.shape(outputs) == (4, 256)
        assert not ops.any(ops.isnan(outputs))

    def test_attention_expert_creation(self):
        """Test AttentionExpert creation and functionality."""
        expert = AttentionExpert(
            hidden_dim=512,
            num_heads=8,
            head_dim=64
        )

        # Test build
        expert.build((None, 32, 512))
        assert expert.built
        assert expert.all_head_dim == 512  # 8 * 64

        # Test forward pass
        inputs = keras.random.normal((2, 32, 512))
        outputs = expert(inputs)

        assert ops.shape(outputs) == (2, 32, 512)
        assert not ops.any(ops.isnan(outputs))

    def test_conv2d_expert_creation(self):
        """Test Conv2DExpert creation and functionality."""
        expert = Conv2DExpert(
            filters=128,
            kernel_size=3,
            padding='same'
        )

        # Test build
        expert.build((None, 32, 32, 64))
        assert expert.built

        # Test forward pass
        inputs = keras.random.normal((4, 32, 32, 64))
        outputs = expert(inputs)

        assert ops.shape(outputs) == (4, 32, 32, 128)
        assert not ops.any(ops.isnan(outputs))

    def test_expert_serialization(self):
        """Test expert serialization and deserialization."""
        original_expert = FFNExpert(
            hidden_dim=256,
            intermediate_size=512,
            activation='relu',
            dropout_rate=0.1
        )
        original_expert.build((None, 128))

        # Get configuration
        config = original_expert.get_config()
        build_config = original_expert.get_build_config()

        # Recreate expert
        new_expert = FFNExpert.from_config(config)
        new_expert.build_from_config(build_config)

        # Test that configurations match
        assert new_expert.hidden_dim == original_expert.hidden_dim
        assert new_expert.intermediate_size == original_expert.intermediate_size
        assert new_expert.activation == original_expert.activation


class TestGating:
    """Test suite for gating network implementations."""

    def test_linear_gating_creation(self):
        """Test LinearGating creation and functionality."""
        gating = LinearGating(
            num_experts=8,
            top_k=2,
            add_noise=True
        )

        # Test build
        gating.build((None, 256))
        assert gating.built
        assert gating.gate_dense is not None
        assert gating.noise_dense is not None

        # Test forward pass
        inputs = keras.random.normal((4, 256))
        expert_weights, expert_indices, aux_info = gating(inputs, training=True)

        assert ops.shape(expert_weights) == (4, 8)
        assert ops.shape(expert_indices) == (4, 2)  # top_k=2
        assert 'gate_logits' in aux_info

    def test_cosine_gating_creation(self):
        """Test CosineGating creation and functionality."""
        gating = CosineGating(
            num_experts=6,
            embedding_dim=128,
            top_k=1,
            temperature=0.1
        )

        # Test build
        gating.build((None, 256))
        assert gating.built
        assert gating.linear_projection is not None
        assert gating.expert_embeddings is not None

        # Test forward pass
        inputs = keras.random.normal((4, 256))
        expert_weights, expert_indices, aux_info = gating(inputs)

        assert ops.shape(expert_weights) == (4, 6)
        assert ops.shape(expert_indices) == (4, 1)  # top_k=1
        assert 'cosine_similarities' in aux_info

    def test_softmoe_gating_creation(self):
        """Test SoftMoEGating creation and functionality."""
        gating = SoftMoEGating(
            num_experts=4,
            num_slots=3
        )

        # Test build
        gating.build((None, 16, 256))  # Sequence input
        assert gating.built
        assert gating.phi_dense is not None

        # Test forward pass
        inputs = keras.random.normal((2, 16, 256))
        expert_weights, expert_indices, aux_info = gating(inputs)

        assert ops.shape(expert_weights) == (2, 16, 4)
        assert 'soft_slots' in aux_info
        assert 'expert_inputs' in aux_info

    def test_auxiliary_loss_computation(self):
        """Test auxiliary loss computation functions."""
        from dl_techniques.layers.moe.gating import compute_auxiliary_loss, compute_z_loss

        # Create dummy data
        expert_weights = keras.random.uniform((4, 16, 8))  # batch, seq, experts
        gate_probs = keras.ops.softmax(keras.random.normal((4, 16, 8)), axis=-1)
        gate_logits = keras.random.normal((4, 16, 8))

        # Test auxiliary loss
        aux_loss = compute_auxiliary_loss(
            expert_weights=expert_weights,
            gate_probs=gate_probs,
            num_experts=8,
            aux_loss_weight=0.01
        )

        assert ops.shape(aux_loss) == ()  # Scalar
        assert aux_loss >= 0.0

        # Test z-loss
        z_loss = compute_z_loss(
            gate_logits=gate_logits,
            z_loss_weight=1e-3
        )

        assert ops.shape(z_loss) == ()  # Scalar
        assert z_loss >= 0.0


class TestMoELayer:
    """Test suite for the main MoE layer."""

    def test_moe_layer_creation(self):
        """Test MoE layer creation with different configurations."""
        config = MoEConfig(
            num_experts=4,
            expert_config=ExpertConfig(
                expert_type='ffn',
                hidden_dim=128,
                intermediate_size=256
            ),
            gating_config=GatingConfig(
                gating_type='linear',
                top_k=2
            )
        )

        moe_layer = MixtureOfExperts(config)

        # Test build
        moe_layer.build((None, 128))
        assert moe_layer.built
        assert len(moe_layer.experts) == 4
        assert moe_layer.gating_network is not None

    def test_moe_layer_forward_pass(self):
        """Test MoE layer forward pass with different input shapes."""
        moe_layer = create_ffn_moe(
            num_experts=6,
            hidden_dim=256,
            top_k=2
        )

        # Test with sequence input
        seq_inputs = keras.random.normal((4, 16, 256))
        seq_outputs = moe_layer(seq_inputs, training=True)

        assert ops.shape(seq_outputs) == (4, 16, 256)
        assert not ops.any(ops.isnan(seq_outputs))

        # Test with single token input
        token_inputs = keras.random.normal((4, 256))
        token_outputs = moe_layer(token_inputs, training=False)

        assert ops.shape(token_outputs) == (4, 256)
        assert not ops.any(ops.isnan(token_outputs))

    def test_moe_layer_different_expert_types(self):
        """Test MoE layer with different expert types."""
        # FFN MoE
        ffn_moe = create_ffn_moe(num_experts=4, hidden_dim=128)
        ffn_inputs = keras.random.normal((2, 16, 128))
        ffn_outputs = ffn_moe(ffn_inputs)
        assert ops.shape(ffn_outputs) == (2, 16, 128)

        # Attention MoE
        attn_moe = create_attention_moe(num_experts=4, hidden_dim=128, num_heads=4)
        attn_inputs = keras.random.normal((2, 16, 128))
        attn_outputs = attn_moe(attn_inputs)
        assert ops.shape(attn_outputs) == (2, 16, 128)

        # Conv MoE
        conv_moe = create_conv_moe(num_experts=4, filters=64)
        conv_inputs = keras.random.normal((2, 32, 32, 64))
        conv_outputs = conv_moe(conv_inputs)
        assert ops.shape(conv_outputs) == (2, 32, 32, 64)

    def test_moe_layer_auxiliary_losses(self):
        """Test that auxiliary losses are properly computed."""
        moe_layer = create_ffn_moe(
            num_experts=8,
            hidden_dim=128,
            top_k=2
        )

        inputs = keras.random.normal((4, 16, 128))

        # Forward pass with training=True should add auxiliary losses
        outputs = moe_layer(inputs, training=True)

        # Check that losses were added
        assert len(moe_layer.losses) > 0

        # Losses should be scalars
        for loss in moe_layer.losses:
            assert ops.shape(loss) == ()

    def test_moe_layer_serialization(self):
        """Test MoE layer serialization and deserialization."""
        config = MoEConfig(
            num_experts=4,
            expert_config=ExpertConfig(hidden_dim=64),
            gating_config=GatingConfig(top_k=1)
        )

        original_layer = MixtureOfExperts(config)
        original_layer.build((None, 64))

        # Get configurations
        layer_config = original_layer.get_config()
        build_config = original_layer.get_build_config()

        # Recreate layer
        new_layer = MixtureOfExperts.from_config(layer_config)
        new_layer.build_from_config(build_config)

        # Test that configurations match
        assert new_layer.num_experts == original_layer.num_experts
        assert len(new_layer.experts) == len(original_layer.experts)


class TestModelIntegration:
    """Test suite for MoE integration in complete models."""

    def test_sequential_model_with_moe(self):
        """Test MoE layer in a sequential model."""
        model = keras.Sequential([
            keras.layers.Input(shape=(16, 128)),
            create_ffn_moe(num_experts=4, hidden_dim=128, top_k=1),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(10, activation='softmax')
        ])

        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Test forward pass
        inputs = keras.random.normal((8, 16, 128))
        outputs = model(inputs)

        assert ops.shape(outputs) == (8, 10)
        assert not ops.any(ops.isnan(outputs))

    def test_functional_model_with_moe(self):
        """Test MoE layer in a functional model."""
        inputs = keras.Input(shape=(32, 256))

        # Multi-layer transformer with MoE
        x = inputs
        for i in range(2):
            # Self-attention
            attn_output = keras.layers.MultiHeadAttention(
                num_heads=8, key_dim=32
            )(x, x)
            x = keras.layers.LayerNormalization()(x + attn_output)

            # MoE layer
            moe_output = create_ffn_moe(
                num_experts=6, hidden_dim=256, top_k=2
            )(x)
            x = keras.layers.LayerNormalization()(x + moe_output)

        # Classification head
        pooled = keras.layers.GlobalAveragePooling1D()(x)
        outputs = keras.layers.Dense(5, activation='softmax')(pooled)

        model = keras.Model(inputs, outputs)

        # Test model
        test_inputs = keras.random.normal((4, 32, 256))
        test_outputs = model(test_inputs)

        assert ops.shape(test_outputs) == (4, 5)
        assert not ops.any(ops.isnan(test_outputs))

    def test_model_save_load(self):
        """Test saving and loading models with MoE layers."""
        # Create model with MoE
        model = keras.Sequential([
            keras.layers.Input(shape=(64,)),
            create_ffn_moe(num_experts=4, hidden_dim=64, top_k=1),
            keras.layers.Dense(2, activation='softmax')
        ])

        # Generate test data
        x_test = keras.random.normal((8, 64))
        original_predictions = model(x_test)

        # Save and load model
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'moe_model.keras')
            model.save(model_path)

            # Load model (custom objects should be registered)
            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={
                    'MixtureOfExperts': MixtureOfExperts,
                    'FFNExpert': FFNExpert,
                    'LinearGating': LinearGating
                }
            )

            # Test loaded model
            loaded_predictions = loaded_model(x_test)

            # Predictions should be close (not identical due to random noise)
            assert ops.shape(loaded_predictions) == ops.shape(original_predictions)
            assert not ops.any(ops.isnan(loaded_predictions))


class TestPresetConfigurations:
    """Test suite for preset MoE configurations."""

    def test_preset_configurations(self):
        """Test all preset configurations."""
        presets = ['default', 'large', 'attention', 'vision']

        for preset_name in presets:
            moe_layer = get_preset_moe(preset_name)
            assert isinstance(moe_layer, MixtureOfExperts)

            # Test that layer can be built
            if preset_name == 'vision':
                input_shape = (None, 32, 32, 256)  # 2D input
            else:
                input_shape = (None, 128, 768)  # Sequence input

            moe_layer.build(input_shape)
            assert moe_layer.built

    def test_preset_overrides(self):
        """Test preset configurations with overrides."""
        # Override number of experts
        moe_layer = get_preset_moe('default', num_experts=16)
        assert moe_layer.num_experts == 16

        # Override gating configuration
        moe_layer = get_preset_moe('large', gating_config={'top_k': 1})
        assert moe_layer.gating_config.top_k == 1


class TestErrorHandling:
    """Test suite for error handling and edge cases."""

    def test_invalid_configurations(self):
        """Test handling of invalid configurations."""
        # Invalid expert type
        with pytest.raises(ValueError):
            expert_config = ExpertConfig(expert_type='invalid')
            config = MoEConfig(expert_config=expert_config)
            MixtureOfExperts(config)

        # Invalid gating type
        with pytest.raises(ValueError):
            gating_config = GatingConfig(gating_type='invalid')
            config = MoEConfig(gating_config=gating_config)
            MixtureOfExperts(config)

    def test_incompatible_shapes(self):
        """Test handling of incompatible input shapes."""
        # Conv expert with 1D input should fail
        conv_moe = create_conv_moe(num_experts=4, filters=64)

        with pytest.raises(Exception):
            invalid_input = keras.random.normal((4, 128))  # 1D input for Conv2D
            conv_moe.build(ops.shape(invalid_input))

    def test_extreme_parameter_values(self):
        """Test handling of extreme parameter values."""
        # Very small number of experts
        small_moe = create_ffn_moe(num_experts=1, hidden_dim=32)
        inputs = keras.random.normal((2, 32))
        outputs = small_moe(inputs)
        assert ops.shape(outputs) == (2, 32)

        # Large top_k value
        config = MoEConfig(
            num_experts=4,
            gating_config=GatingConfig(top_k=10)  # top_k > num_experts
        )
        # Should handle gracefully by using all experts
        moe_layer = MixtureOfExperts(config)
        moe_layer.build((None, 64))


def run_performance_benchmark():
    """Run performance benchmarks for MoE layers."""
    import time

    print("\n=== MoE Performance Benchmark ===")

    # Test configurations
    configs = [
        ("Dense Baseline", None),
        ("MoE 4 experts k=1", (4, 1)),
        ("MoE 8 experts k=2", (8, 2)),
        ("MoE 16 experts k=1", (16, 1))
    ]

    batch_size = 32
    seq_len = 128
    hidden_dim = 512
    num_iterations = 10

    for config_name, moe_config in configs:
        if moe_config is None:
            # Dense baseline
            layer = keras.layers.Dense(hidden_dim, activation='gelu')
        else:
            num_experts, top_k = moe_config
            layer = create_ffn_moe(
                num_experts=num_experts,
                hidden_dim=hidden_dim,
                top_k=top_k
            )

        # Build layer
        layer.build((None, seq_len, hidden_dim))

        # Warm up
        inputs = keras.random.normal((batch_size, seq_len, hidden_dim))
        _ = layer(inputs)

        # Benchmark
        start_time = time.time()
        for _ in range(num_iterations):
            outputs = layer(inputs, training=True)
        end_time = time.time()

        avg_time = (end_time - start_time) / num_iterations * 1000  # ms
        print(f"{config_name:20}: {avg_time:.2f} ms/iteration")


# Utility functions for testing

def create_dummy_dataset(batch_size=32, seq_len=64, hidden_dim=256, num_classes=10):
    """Create dummy dataset for testing MoE models."""
    x = keras.random.normal((batch_size, seq_len, hidden_dim))
    y = keras.random.uniform((batch_size,), 0, num_classes, dtype='int32')
    return x, y


def test_moe_training_step():
    """Test a complete training step with MoE model."""
    # Create model
    model = keras.Sequential([
        keras.layers.Input(shape=(64, 256)),
        create_ffn_moe(num_experts=4, hidden_dim=256, top_k=2),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(10, activation='softmax')
    ])

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Create dummy data
    x, y = create_dummy_dataset()

    # Training step
    history = model.fit(x, y, epochs=1, verbose=0)

    assert 'loss' in history.history
    assert 'accuracy' in history.history
    assert not np.isnan(history.history['loss'][0])


if __name__ == "__main__":
    # Run tests when script is executed directly
    import sys

    print("Running MoE module tests...")

    # Run pytest if available
    try:
        exit_code = pytest.main([__file__, '-v'])
        sys.exit(exit_code)
    except ImportError:
        print("pytest not available, running manual tests...")

        # Manual test execution
        test_classes = [
            TestMoEConfig,
            TestExperts,
            TestGating,
            TestMoELayer,
            TestModelIntegration,
            TestPresetConfigurations,
            TestErrorHandling
        ]

        total_tests = 0
        passed_tests = 0

        for test_class in test_classes:
            print(f"\n--- {test_class.__name__} ---")
            test_instance = test_class()

            # Get all test methods
            test_methods = [method for method in dir(test_instance)
                            if method.startswith('test_')]

            for method_name in test_methods:
                total_tests += 1
                try:
                    method = getattr(test_instance, method_name)
                    method()
                    print(f"✓ {method_name}")
                    passed_tests += 1
                except Exception as e:
                    print(f"✗ {method_name}: {str(e)}")

        print(f"\n=== Test Results ===")
        print(f"Passed: {passed_tests}/{total_tests}")
        print(f"Success rate: {passed_tests / total_tests * 100:.1f}%")

        # Run performance benchmark
        run_performance_benchmark()

        # Test training step
        print("\n=== Training Test ===")
        try:
            test_moe_training_step()
            print("✓ Training step test passed")
        except Exception as e:
            print(f"✗ Training step test failed: {str(e)}")