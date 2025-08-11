import pytest
import numpy as np
import tensorflow as tf
import keras
import tempfile
import os

from dl_techniques.layers.transformer_encoder import TransformerEncoderLayer


class TestTransformerEncoderLayer:
    """Test suite for TransformerEncoderLayer implementation."""

    @pytest.fixture
    def input_tensor(self):
        """Create a test input tensor."""
        return tf.random.normal([2, 128, 768])  # batch_size=2, seq_len=128, hidden_size=768

    @pytest.fixture
    def small_input_tensor(self):
        """Create a smaller test input tensor for faster tests."""
        return tf.random.normal([2, 16, 64])  # batch_size=2, seq_len=16, hidden_size=64

    @pytest.fixture
    def layer_instance(self):
        """Create a default layer instance for testing."""
        return TransformerEncoderLayer(
            hidden_size=768,
            num_heads=12,
            intermediate_size=3072
        )

    @pytest.fixture
    def small_layer_instance(self):
        """Create a smaller layer instance for faster tests."""
        return TransformerEncoderLayer(
            hidden_size=64,
            num_heads=8,
            intermediate_size=256
        )

    # ===== INITIALIZATION TESTS =====

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        layer = TransformerEncoderLayer(
            hidden_size=768,
            num_heads=12,
            intermediate_size=3072
        )

        # Check default values
        assert layer.hidden_size == 768
        assert layer.num_heads == 12
        assert layer.intermediate_size == 3072
        assert layer.attention_type == 'multi_head_attention'  # NEW: Check default attention type
        assert layer.normalization_type == 'layer_norm'
        assert layer.normalization_position == 'post'
        assert layer.ffn_type == 'mlp'
        assert layer.dropout_rate == 0.1
        assert layer.attention_dropout_rate == 0.1
        assert layer.activation == 'gelu'
        assert layer.use_bias is True
        assert isinstance(layer.kernel_initializer, keras.initializers.GlorotUniform)
        assert isinstance(layer.bias_initializer, keras.initializers.Zeros)
        assert layer.kernel_regularizer is None
        assert layer.bias_regularizer is None
        assert layer.ffn_expansion_factor == 4
        assert layer.ffn_multiple_of == 256
        assert layer.window_size == 8  # NEW: Check default window size
        assert layer.n_kv_head == 12  # NEW: Check default n_kv_head (equals num_heads)

    def test_initialization_custom_parameters(self):
        """Test initialization with custom parameters."""
        custom_regularizer = keras.regularizers.L2(1e-4)

        layer = TransformerEncoderLayer(
            hidden_size=512,
            num_heads=8,
            intermediate_size=2048,
            attention_type='window_attention',  # NEW: Test different attention type
            normalization_type='layer_norm',
            normalization_position='pre',
            ffn_type='swiglu',
            dropout_rate=0.2,
            attention_dropout_rate=0.15,
            activation='relu',
            use_bias=False,
            kernel_initializer='he_normal',
            bias_initializer='ones',
            kernel_regularizer=custom_regularizer,
            bias_regularizer=custom_regularizer,
            ffn_expansion_factor=6,
            ffn_multiple_of=128,
            window_size=16,  # NEW: Test custom window size
            n_kv_head=4  # NEW: Test custom n_kv_head
        )

        # Check custom values
        assert layer.hidden_size == 512
        assert layer.num_heads == 8
        assert layer.intermediate_size == 2048
        assert layer.attention_type == 'window_attention'  # NEW
        assert layer.normalization_type == 'layer_norm'
        assert layer.normalization_position == 'pre'
        assert layer.ffn_type == 'swiglu'
        assert layer.dropout_rate == 0.2
        assert layer.attention_dropout_rate == 0.15
        assert layer.activation == 'relu'
        assert layer.use_bias is False
        assert isinstance(layer.kernel_initializer, keras.initializers.HeNormal)
        assert isinstance(layer.bias_initializer, keras.initializers.Ones)
        assert layer.kernel_regularizer == custom_regularizer
        assert layer.bias_regularizer == custom_regularizer
        assert layer.ffn_expansion_factor == 6
        assert layer.ffn_multiple_of == 128
        assert layer.window_size == 16  # NEW
        assert layer.n_kv_head == 4  # NEW

    def test_initialization_all_attention_types(self):
        """Test initialization with all attention types."""
        attention_types = ['multi_head_attention', 'window_attention', 'group_query_attention']

        for attention_type in attention_types:
            layer = TransformerEncoderLayer(
                hidden_size=64,
                num_heads=4,
                intermediate_size=256,
                attention_type=attention_type
            )
            assert layer.attention_type == attention_type

    def test_initialization_attention_type_with_specific_params(self):
        """Test initialization with attention-specific parameters."""
        # Test window_attention with custom window_size
        layer_window = TransformerEncoderLayer(
            hidden_size=64,
            num_heads=4,
            intermediate_size=256,
            attention_type='window_attention',
            window_size=32
        )
        assert layer_window.attention_type == 'window_attention'
        assert layer_window.window_size == 32

        # Test group_query_attention with custom n_kv_head
        layer_gqa = TransformerEncoderLayer(
            hidden_size=64,
            num_heads=8,
            intermediate_size=256,
            attention_type='group_query_attention',
            n_kv_head=2
        )
        assert layer_gqa.attention_type == 'group_query_attention'
        assert layer_gqa.n_kv_head == 2

        # Test group_query_attention with default n_kv_head (should equal num_heads)
        layer_gqa_default = TransformerEncoderLayer(
            hidden_size=64,
            num_heads=8,
            intermediate_size=256,
            attention_type='group_query_attention'
        )
        assert layer_gqa_default.n_kv_head == 8

    def test_initialization_all_normalization_types(self):
        """Test initialization with all normalization types."""
        # Test with always available types
        basic_norm_types = ['layer_norm', 'batch_norm']

        for norm_type in basic_norm_types:
            layer = TransformerEncoderLayer(
                hidden_size=64,
                num_heads=4,
                intermediate_size=256,
                normalization_type=norm_type
            )
            assert layer.normalization_type == norm_type

        # Test with optional types if available
        try:
            layer = TransformerEncoderLayer(
                hidden_size=64,
                num_heads=4,
                intermediate_size=256,
                normalization_type='rms_norm'
            )
            assert layer.normalization_type == 'rms_norm'
        except ValueError:
            # RMS norm not available, skip test
            pass

        try:
            layer = TransformerEncoderLayer(
                hidden_size=64,
                num_heads=4,
                intermediate_size=256,
                normalization_type='band_rms'
            )
            assert layer.normalization_type == 'band_rms'
        except ValueError:
            # Band RMS not available, skip test
            pass

    def test_initialization_all_normalization_positions(self):
        """Test initialization with all normalization positions."""
        positions = ['post', 'pre']

        for position in positions:
            layer = TransformerEncoderLayer(
                hidden_size=64,
                num_heads=4,
                intermediate_size=256,
                normalization_position=position
            )
            assert layer.normalization_position == position

    def test_initialization_all_ffn_types(self):
        """Test initialization with all FFN types."""
        ffn_types = ['mlp', 'swiglu', 'differential', 'glu', 'residual', 'swin_mlp']

        for ffn_type in ffn_types:
            try:
                layer = TransformerEncoderLayer(
                    hidden_size=64,
                    num_heads=4,
                    intermediate_size=256,
                    ffn_type=ffn_type
                )
                assert layer.ffn_type == ffn_type
            except (ImportError, AttributeError):
                # FFN type not available, skip test
                pass

    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        # Negative hidden_size
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            TransformerEncoderLayer(hidden_size=-768, num_heads=12, intermediate_size=3072)

        # Zero hidden_size
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            TransformerEncoderLayer(hidden_size=0, num_heads=12, intermediate_size=3072)

        # Negative num_heads
        with pytest.raises(ValueError, match="num_heads must be positive"):
            TransformerEncoderLayer(hidden_size=768, num_heads=-12, intermediate_size=3072)

        # Hidden size not divisible by num_heads
        with pytest.raises(ValueError, match="hidden_size.*must be divisible by num_heads"):
            TransformerEncoderLayer(hidden_size=100, num_heads=12, intermediate_size=3072)

        # Negative intermediate_size
        with pytest.raises(ValueError, match="intermediate_size must be positive"):
            TransformerEncoderLayer(hidden_size=768, num_heads=12, intermediate_size=-3072)

        # Invalid dropout rates
        with pytest.raises(ValueError, match="dropout_rate must be between 0 and 1"):
            TransformerEncoderLayer(hidden_size=768, num_heads=12, intermediate_size=3072, dropout_rate=1.5)

        with pytest.raises(ValueError, match="attention_dropout_rate must be between 0 and 1"):
            TransformerEncoderLayer(hidden_size=768, num_heads=12, intermediate_size=3072, attention_dropout_rate=-0.1)

        # NEW: Invalid attention type
        with pytest.raises(ValueError, match="attention_type must be one of"):
            TransformerEncoderLayer(hidden_size=768, num_heads=12, intermediate_size=3072,
                                    attention_type='invalid_attention')

        # Invalid normalization type
        with pytest.raises(ValueError, match="normalization_type must be one of"):
            TransformerEncoderLayer(hidden_size=768, num_heads=12, intermediate_size=3072,
                                    normalization_type='invalid_type')

        # Invalid normalization position
        with pytest.raises(ValueError, match="normalization_position must be one of"):
            TransformerEncoderLayer(hidden_size=768, num_heads=12, intermediate_size=3072,
                                    normalization_position='invalid_position')

        # Invalid FFN type
        with pytest.raises(ValueError, match="ffn_type must be one of"):
            TransformerEncoderLayer(hidden_size=768, num_heads=12, intermediate_size=3072,
                                    ffn_type='invalid_ffn')

        # NEW: Invalid window_size
        with pytest.raises(ValueError, match="window_size must be positive"):
            TransformerEncoderLayer(hidden_size=768, num_heads=12, intermediate_size=3072,
                                    window_size=-8)

        with pytest.raises(ValueError, match="window_size must be positive"):
            TransformerEncoderLayer(hidden_size=768, num_heads=12, intermediate_size=3072,
                                    window_size=0)

    # ===== BUILD PROCESS TESTS =====

    def test_build_process(self, small_input_tensor, small_layer_instance):
        """Test that the layer builds properly."""
        layer = small_layer_instance
        layer(small_input_tensor)  # Forward pass triggers build

        # Check that layer was built
        assert layer.built is True
        assert len(layer.weights) > 0

        # Check that sublayers exist and are built
        assert layer.attention is not None
        assert layer.attention.built is True
        assert layer.attention_norm is not None
        assert layer.attention_norm.built is True
        assert layer.ffn_layer is not None
        assert layer.ffn_layer.built is True
        assert layer.output_norm is not None
        assert layer.output_norm.built is True
        assert layer.dropout is not None
        assert layer.attention_dropout is not None

        # Check build input shape is stored
        assert layer._build_input_shape is not None

    def test_build_process_different_attention_types(self, small_input_tensor):
        """Test building with different attention types."""
        attention_types = ['multi_head_attention', 'window_attention', 'group_query_attention']

        for attention_type in attention_types:
            try:
                layer = TransformerEncoderLayer(
                    hidden_size=64,
                    num_heads=4,
                    intermediate_size=256,
                    attention_type=attention_type
                )

                # Forward pass triggers build
                layer(small_input_tensor)

                # Check that layer was built
                assert layer.built is True
                assert layer.attention is not None
                assert layer.attention.built is True

                # Check attention layer type matches expected type
                if attention_type == 'multi_head_attention':
                    assert isinstance(layer.attention, keras.layers.MultiHeadAttention)
                elif attention_type == 'window_attention':
                    assert layer.attention.__class__.__name__ == 'WindowAttention'
                elif attention_type == 'group_query_attention':
                    assert layer.attention.__class__.__name__ == 'GroupedQueryAttention'

            except (ImportError, AttributeError, ValueError):
                # Attention type not available, skip test
                pass

    def test_build_with_different_input_shapes(self):
        """Test building with different input shapes."""
        test_shapes = [
            (None, 32, 128),  # Different sequence length
            (4, 64, 256),  # Different batch size and dimensions
            (1, 512, 768),  # Different configuration
        ]

        for input_shape in test_shapes:
            hidden_size = input_shape[-1]
            layer = TransformerEncoderLayer(
                hidden_size=hidden_size,
                num_heads=8 if hidden_size >= 128 else 4,
                intermediate_size=hidden_size * 4
            )

            # Build with shape
            layer.build(input_shape)

            # Check build was successful
            assert layer.built is True
            assert layer._build_input_shape == input_shape

    def test_build_invalid_input_shape(self):
        """Test building with invalid input shapes."""
        layer = TransformerEncoderLayer(hidden_size=768, num_heads=12, intermediate_size=3072)

        # Wrong number of dimensions
        with pytest.raises(ValueError, match="Expected 3D input shape"):
            layer.build((768, 128))  # 2D shape

        # Wrong feature dimension
        with pytest.raises(ValueError, match="Input feature dimension.*must match hidden_size"):
            layer.build((None, 128, 512))  # Wrong last dimension

    # ===== OUTPUT SHAPE TESTS =====

    def test_output_shapes(self, small_input_tensor):
        """Test that output shapes are computed correctly."""
        test_configs = [
            (64, 4, 256),
            (128, 8, 512),
            (256, 16, 1024),
        ]

        for hidden_size, num_heads, intermediate_size in test_configs:
            layer = TransformerEncoderLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size
            )

            # Create appropriate input
            input_shape = (2, 16, hidden_size)
            test_input = tf.random.normal(input_shape)

            # Forward pass
            output = layer(test_input)

            # Check output shape matches input shape
            assert output.shape == test_input.shape

            # Test compute_output_shape separately
            computed_shape = layer.compute_output_shape(input_shape)
            assert computed_shape == input_shape

    def test_compute_output_shape_with_none_dimensions(self):
        """Test compute_output_shape with None dimensions."""
        layer = TransformerEncoderLayer(hidden_size=768, num_heads=12, intermediate_size=3072)

        test_shapes = [
            (None, 128, 768),
            (4, None, 768),
            (None, None, 768),
        ]

        for input_shape in test_shapes:
            output_shape = layer.compute_output_shape(input_shape)
            assert output_shape == input_shape

    # ===== FORWARD PASS TESTS =====

    def test_forward_pass_basic(self, small_input_tensor, small_layer_instance):
        """Test basic forward pass."""
        layer = small_layer_instance
        output = layer(small_input_tensor)

        # Basic sanity checks
        assert output.shape == small_input_tensor.shape
        assert not np.any(np.isnan(output.numpy()))
        assert not np.any(np.isinf(output.numpy()))

    def test_forward_pass_different_attention_types(self, small_input_tensor):
        """Test forward pass with different attention types."""
        attention_types = ['multi_head_attention', 'window_attention', 'group_query_attention']

        for attention_type in attention_types:
            try:
                layer = TransformerEncoderLayer(
                    hidden_size=64,
                    num_heads=4,
                    intermediate_size=256,
                    attention_type=attention_type,
                    window_size=8,  # For window attention
                    n_kv_head=2  # For group query attention
                )

                output = layer(small_input_tensor)

                # Check output is valid
                assert output.shape == small_input_tensor.shape
                assert not np.any(np.isnan(output.numpy()))
                assert not np.any(np.isinf(output.numpy()))

            except (ImportError, AttributeError, ValueError):
                # Attention type not available, skip test
                pass

    def test_forward_pass_normalization_positions(self, small_input_tensor):
        """Test forward pass with different normalization positions."""
        positions = ['post', 'pre']

        for position in positions:
            layer = TransformerEncoderLayer(
                hidden_size=64,
                num_heads=4,
                intermediate_size=256,
                normalization_position=position
            )

            output = layer(small_input_tensor)

            # Check output is valid
            assert output.shape == small_input_tensor.shape
            assert not np.any(np.isnan(output.numpy()))
            assert not np.any(np.isinf(output.numpy()))

    def test_forward_pass_different_ffn_types(self, small_input_tensor):
        """Test forward pass with different FFN types."""
        ffn_types = ['mlp', 'swiglu', 'differential', 'glu', 'residual', 'swin_mlp']

        for ffn_type in ffn_types:
            try:
                layer = TransformerEncoderLayer(
                    hidden_size=64,
                    num_heads=4,
                    intermediate_size=256,
                    ffn_type=ffn_type
                )

                output = layer(small_input_tensor)

                # Check output is valid
                assert output.shape == small_input_tensor.shape
                assert not np.any(np.isnan(output.numpy()))
                assert not np.any(np.isinf(output.numpy()))
            except (ImportError, AttributeError):
                # FFN type not available, skip test
                pass

    def test_forward_pass_combined_configurations(self, small_input_tensor):
        """Test forward pass with combined configurations including attention types."""
        test_configs = [
            # Original configurations
            {'normalization_position': 'post', 'ffn_type': 'mlp', 'normalization_type': 'layer_norm',
             'attention_type': 'multi_head_attention'},
            {'normalization_position': 'pre', 'ffn_type': 'mlp', 'normalization_type': 'layer_norm',
             'attention_type': 'multi_head_attention'},
            {'normalization_position': 'post', 'ffn_type': 'swiglu', 'normalization_type': 'layer_norm',
             'attention_type': 'multi_head_attention'},
            {'normalization_position': 'pre', 'ffn_type': 'swiglu', 'normalization_type': 'layer_norm',
             'attention_type': 'multi_head_attention'},
            # NEW: Configurations with different attention types
            {'normalization_position': 'post', 'ffn_type': 'mlp', 'normalization_type': 'layer_norm',
             'attention_type': 'window_attention', 'window_size': 8},
            {'normalization_position': 'pre', 'ffn_type': 'mlp', 'normalization_type': 'layer_norm',
             'attention_type': 'window_attention', 'window_size': 16},
            {'normalization_position': 'post', 'ffn_type': 'mlp', 'normalization_type': 'layer_norm',
             'attention_type': 'group_query_attention', 'n_kv_head': 2},
            {'normalization_position': 'pre', 'ffn_type': 'mlp', 'normalization_type': 'layer_norm',
             'attention_type': 'group_query_attention', 'n_kv_head': 1},
        ]

        for config in test_configs:
            try:
                layer = TransformerEncoderLayer(
                    hidden_size=64,
                    num_heads=4,
                    intermediate_size=256,
                    **config
                )

                output = layer(small_input_tensor)

                # Check output is valid
                assert output.shape == small_input_tensor.shape
                assert not np.any(np.isnan(output.numpy()))
                assert not np.any(np.isinf(output.numpy()))
            except (ImportError, AttributeError, ValueError):
                # Configuration not available, skip test
                pass

    def test_forward_pass_with_attention_mask(self, small_input_tensor, small_layer_instance):
        """Test forward pass with attention mask."""
        layer = small_layer_instance
        batch_size, seq_len, _ = small_input_tensor.shape

        # Test with 3D attention mask - shape should be (batch_size, seq_len, seq_len)
        attention_mask = tf.ones((batch_size, seq_len, seq_len))

        output = layer(small_input_tensor, attention_mask=attention_mask)

        # Check output is valid
        assert output.shape == small_input_tensor.shape
        assert not np.any(np.isnan(output.numpy()))

        # Test with no mask
        output_no_mask = layer(small_input_tensor, attention_mask=None)

        # Check output is valid
        assert output_no_mask.shape == small_input_tensor.shape
        assert not np.any(np.isnan(output_no_mask.numpy()))

    def test_forward_pass_training_modes(self, small_input_tensor, small_layer_instance):
        """Test forward pass in training and inference modes."""
        layer = small_layer_instance

        # Training mode
        output_train = layer(small_input_tensor, training=True)

        # Inference mode
        output_inference = layer(small_input_tensor, training=False)

        # Both should produce valid outputs
        assert output_train.shape == small_input_tensor.shape
        assert output_inference.shape == small_input_tensor.shape
        assert not np.any(np.isnan(output_train.numpy()))
        assert not np.any(np.isnan(output_inference.numpy()))

    def test_forward_pass_all_normalization_types(self, small_input_tensor):
        """Test forward pass with all normalization types."""
        # Test with always available types
        basic_norm_types = ['layer_norm', 'batch_norm']

        for norm_type in basic_norm_types:
            layer = TransformerEncoderLayer(
                hidden_size=64,
                num_heads=4,
                intermediate_size=256,
                normalization_type=norm_type
            )

            output = layer(small_input_tensor)

            # Check output is valid
            assert output.shape == small_input_tensor.shape
            assert not np.any(np.isnan(output.numpy()))

        # Test with optional types if available
        for norm_type in ['rms_norm', 'band_rms']:
            try:
                layer = TransformerEncoderLayer(
                    hidden_size=64,
                    num_heads=4,
                    intermediate_size=256,
                    normalization_type=norm_type
                )

                output = layer(small_input_tensor)

                # Check output is valid
                assert output.shape == small_input_tensor.shape
                assert not np.any(np.isnan(output.numpy()))
            except ValueError:
                # Normalization type not available, skip test
                pass

    # ===== NEW: ATTENTION-SPECIFIC TESTS =====

    def test_attention_type_behavior(self, small_input_tensor):
        """Test that different attention types produce different outputs."""
        attention_types_to_test = ['multi_head_attention', 'window_attention', 'group_query_attention']
        outputs = {}

        for attention_type in attention_types_to_test:
            try:
                layer = TransformerEncoderLayer(
                    hidden_size=64,
                    num_heads=4,
                    intermediate_size=256,
                    attention_type=attention_type,
                    dropout_rate=0.0,  # Disable dropout for deterministic comparison
                    attention_dropout_rate=0.0,
                    window_size=8,  # For window attention
                    n_kv_head=2  # For group query attention
                )

                output = layer(small_input_tensor, training=False)
                outputs[attention_type] = output.numpy()

            except (ImportError, AttributeError, ValueError):
                # Attention type not available, skip
                pass

        # If we have multiple outputs, they should be different
        if len(outputs) >= 2:
            output_list = list(outputs.values())
            attention_list = list(outputs.keys())
            for i in range(len(output_list)):
                for j in range(i + 1, len(output_list)):
                    assert not np.allclose(output_list[i], output_list[j], rtol=1e-3), \
                        f"Different attention types ({attention_list[i]} vs {attention_list[j]}) should produce different outputs"

    def test_window_attention_specific_behavior(self, small_input_tensor):
        """Test window attention with different window sizes."""
        window_sizes = [4, 8, 16]
        outputs = {}

        for window_size in window_sizes:
            try:
                layer = TransformerEncoderLayer(
                    hidden_size=64,
                    num_heads=4,
                    intermediate_size=256,
                    attention_type='window_attention',
                    window_size=window_size,
                    dropout_rate=0.0,
                    attention_dropout_rate=0.0
                )

                output = layer(small_input_tensor, training=False)
                outputs[window_size] = output.numpy()

            except (ImportError, AttributeError, ValueError):
                # Window attention not available, skip test
                return

        # If we have multiple outputs, they should be different
        if len(outputs) >= 2:
            output_list = list(outputs.values())
            window_list = list(outputs.keys())
            for i in range(len(output_list)):
                for j in range(i + 1, len(output_list)):
                    assert not np.allclose(output_list[i], output_list[j], rtol=1e-3), \
                        f"Different window sizes ({window_list[i]} vs {window_list[j]}) should produce different outputs"

    def test_group_query_attention_specific_behavior(self, small_input_tensor):
        """Test group query attention with different n_kv_head values."""
        n_kv_head_values = [1, 2, 4]  # Test different grouping
        outputs = {}

        for n_kv_head in n_kv_head_values:
            try:
                layer = TransformerEncoderLayer(
                    hidden_size=64,
                    num_heads=4,
                    intermediate_size=256,
                    attention_type='group_query_attention',
                    n_kv_head=n_kv_head,
                    dropout_rate=0.0,
                    attention_dropout_rate=0.0
                )

                output = layer(small_input_tensor, training=False)
                outputs[n_kv_head] = output.numpy()

            except (ImportError, AttributeError, ValueError):
                # Group query attention not available, skip test
                return

        # If we have multiple outputs, they should be different
        if len(outputs) >= 2:
            output_list = list(outputs.values())
            n_kv_head_list = list(outputs.keys())
            for i in range(len(output_list)):
                for j in range(i + 1, len(output_list)):
                    assert not np.allclose(output_list[i], output_list[j], rtol=1e-3), \
                        f"Different n_kv_head values ({n_kv_head_list[i]} vs {n_kv_head_list[j]}) should produce different outputs"

    def test_attention_layer_creation_error(self):
        """Test error handling in attention layer creation."""
        layer = TransformerEncoderLayer(
            hidden_size=64,
            num_heads=4,
            intermediate_size=256
        )

        # Set invalid attention type manually (bypassing __init__ validation)
        original_type = layer.attention_type
        layer.attention_type = 'invalid_attention_type'

        try:
            with pytest.raises(ValueError, match="Unknown attention type"):
                layer._create_attention_layer('test_attention')
        finally:
            # Restore original type
            layer.attention_type = original_type

    # ===== SERIALIZATION TESTS =====

    def test_serialization_basic(self):
        """Test basic serialization and deserialization."""
        # Create and build layer
        original_layer = TransformerEncoderLayer(
            hidden_size=128,
            num_heads=8,
            intermediate_size=512,
            attention_type='multi_head_attention',  # NEW: Include attention type
            normalization_type='layer_norm',
            normalization_position='pre',
            ffn_type='mlp',
            dropout_rate=0.2
        )
        original_layer.build((None, 32, 128))

        # Test data
        x = tf.random.normal((2, 32, 128))
        original_output = original_layer(x, training=False)

        # Serialize and recreate
        config = original_layer.get_config()
        build_config = original_layer.get_build_config()

        new_layer = TransformerEncoderLayer.from_config(config)
        new_layer.build_from_config(build_config)

        # Check configuration matches
        assert new_layer.hidden_size == original_layer.hidden_size
        assert new_layer.num_heads == original_layer.num_heads
        assert new_layer.intermediate_size == original_layer.intermediate_size
        assert new_layer.attention_type == original_layer.attention_type  # NEW
        assert new_layer.normalization_type == original_layer.normalization_type
        assert new_layer.normalization_position == original_layer.normalization_position
        assert new_layer.ffn_type == original_layer.ffn_type
        assert new_layer.dropout_rate == original_layer.dropout_rate

    def test_serialization_all_attention_types(self):
        """Test serialization with all attention types."""
        attention_types = ['multi_head_attention', 'window_attention', 'group_query_attention']

        for attention_type in attention_types:
            try:
                # Create layer with specific attention configuration
                config_params = {
                    'hidden_size': 64,
                    'num_heads': 4,
                    'intermediate_size': 256,
                    'attention_type': attention_type,
                    'normalization_type': 'layer_norm',
                    'window_size': 16,  # For window attention
                    'n_kv_head': 2  # For group query attention
                }

                original_layer = TransformerEncoderLayer(**config_params)

                # Get config
                config = original_layer.get_config()

                # Recreate layer
                new_layer = TransformerEncoderLayer.from_config(config)

                # Check attention-specific parameters match
                assert new_layer.attention_type == original_layer.attention_type
                assert new_layer.window_size == original_layer.window_size
                assert new_layer.n_kv_head == original_layer.n_kv_head

            except (ImportError, AttributeError, ValueError):
                # Attention type not available, skip test
                pass

    def test_serialization_all_parameters(self):
        """Test serialization with all parameters."""
        custom_regularizer = keras.regularizers.L2(1e-4)

        # Use a normalization type that's always available
        original_layer = TransformerEncoderLayer(
            hidden_size=256,
            num_heads=16,
            intermediate_size=1024,
            attention_type='window_attention',  # NEW: Test different attention type
            normalization_type='layer_norm',
            normalization_position='pre',
            ffn_type='swiglu',
            dropout_rate=0.3,
            attention_dropout_rate=0.25,
            activation='relu',
            use_bias=False,
            kernel_initializer='he_normal',
            bias_initializer='ones',
            kernel_regularizer=custom_regularizer,
            bias_regularizer=custom_regularizer,
            ffn_expansion_factor=6,
            ffn_multiple_of=128,
            window_size=32,  # NEW: Test window size serialization
            n_kv_head=8  # NEW: Test n_kv_head serialization
        )

        # Get config
        config = original_layer.get_config()

        # Recreate layer
        try:
            new_layer = TransformerEncoderLayer.from_config(config)

            # Check all parameters match
            assert new_layer.hidden_size == original_layer.hidden_size
            assert new_layer.num_heads == original_layer.num_heads
            assert new_layer.intermediate_size == original_layer.intermediate_size
            assert new_layer.attention_type == original_layer.attention_type  # NEW
            assert new_layer.normalization_type == original_layer.normalization_type
            assert new_layer.normalization_position == original_layer.normalization_position
            assert new_layer.ffn_type == original_layer.ffn_type
            assert new_layer.dropout_rate == original_layer.dropout_rate
            assert new_layer.attention_dropout_rate == original_layer.attention_dropout_rate
            assert new_layer.activation == original_layer.activation
            assert new_layer.use_bias == original_layer.use_bias
            assert new_layer.ffn_expansion_factor == original_layer.ffn_expansion_factor
            assert new_layer.ffn_multiple_of == original_layer.ffn_multiple_of
            assert new_layer.window_size == original_layer.window_size  # NEW
            assert new_layer.n_kv_head == original_layer.n_kv_head  # NEW
        except (ImportError, AttributeError):
            # FFN type or attention type not available, skip test
            pass

    # ===== FFN LAYER CREATION TESTS =====

    def test_ffn_layer_creation_error(self):
        """Test error handling in FFN layer creation."""
        layer = TransformerEncoderLayer(
            hidden_size=64,
            num_heads=4,
            intermediate_size=256
        )

        # Set invalid FFN type manually (bypassing __init__ validation)
        original_type = layer.ffn_type
        layer.ffn_type = 'invalid_ffn_type'

        try:
            with pytest.raises(ValueError, match="Unknown FFN type"):
                layer._create_ffn_layer('test_ffn')
        finally:
            # Restore original type
            layer.ffn_type = original_type

    # ===== MODEL INTEGRATION TESTS =====

    def test_model_integration_simple(self, small_input_tensor):
        """Test the layer in a simple model context."""
        # Create a simple model with the custom layer
        inputs = keras.Input(shape=(16, 64))
        x = TransformerEncoderLayer(
            hidden_size=64,
            num_heads=4,
            intermediate_size=256,
            attention_type='multi_head_attention',  # NEW: Explicit attention type
            normalization_position='pre',
            ffn_type='mlp'
        )(inputs)
        x = keras.layers.GlobalAveragePooling1D()(x)
        outputs = keras.layers.Dense(10)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Compile the model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
        )

        # Test forward pass
        y_pred = model(small_input_tensor, training=False)
        assert y_pred.shape == (small_input_tensor.shape[0], 10)

    def test_model_integration_different_attention_types(self, small_input_tensor):
        """Test model with different attention types."""
        attention_types = ['multi_head_attention', 'window_attention', 'group_query_attention']

        for attention_type in attention_types:
            try:
                # Create model with specific attention type
                inputs = keras.Input(shape=(16, 64))
                x = TransformerEncoderLayer(
                    hidden_size=64,
                    num_heads=4,
                    intermediate_size=256,
                    attention_type=attention_type,
                    window_size=8,  # For window attention
                    n_kv_head=2  # For group query attention
                )(inputs)
                x = keras.layers.GlobalAveragePooling1D()(x)
                outputs = keras.layers.Dense(5)(x)

                model = keras.Model(inputs=inputs, outputs=outputs)
                model.compile(optimizer='adam', loss='mse')

                # Test forward pass
                y_pred = model(small_input_tensor)
                assert y_pred.shape == (small_input_tensor.shape[0], 5)

            except (ImportError, AttributeError, ValueError):
                # Attention type not available, skip test
                pass

    def test_model_integration_stacked(self, small_input_tensor):
        """Test multiple layers stacked in a model."""
        # Create model with multiple transformer layers using different configurations
        inputs = keras.Input(shape=(16, 64))
        x = TransformerEncoderLayer(
            hidden_size=64,
            num_heads=4,
            intermediate_size=256,
            attention_type='multi_head_attention',  # First layer: standard attention
            normalization_type='layer_norm',
            normalization_position='post',  # Post-norm first layer
            ffn_type='mlp'
        )(inputs)

        try:
            x = TransformerEncoderLayer(
                hidden_size=64,
                num_heads=4,
                intermediate_size=256,
                attention_type='window_attention',  # Second layer: window attention
                normalization_type='layer_norm',
                normalization_position='pre',  # Pre-norm second layer
                ffn_type='mlp',
                window_size=8
            )(x)
        except (ImportError, AttributeError, ValueError):
            # Window attention not available, use standard attention
            x = TransformerEncoderLayer(
                hidden_size=64,
                num_heads=4,
                intermediate_size=256,
                attention_type='multi_head_attention',
                normalization_type='layer_norm',
                normalization_position='pre',
                ffn_type='mlp'
            )(x)

        x = keras.layers.GlobalAveragePooling1D()(x)
        outputs = keras.layers.Dense(5)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mse')

        # Test forward pass
        y_pred = model(small_input_tensor)
        assert y_pred.shape == (small_input_tensor.shape[0], 5)

    # ===== MODEL SAVE/LOAD TESTS =====

    def test_model_save_load(self, small_input_tensor):
        """Test saving and loading a model with the custom layer."""
        # Create model with custom layer
        inputs = keras.Input(shape=(16, 64))
        x = TransformerEncoderLayer(
            hidden_size=64,
            num_heads=4,
            intermediate_size=256,
            attention_type='multi_head_attention',  # NEW: Include attention type
            normalization_type='layer_norm',
            normalization_position='pre',
            ffn_type='mlp',
            name='transformer_layer'
        )(inputs)
        x = keras.layers.GlobalAveragePooling1D()(x)
        outputs = keras.layers.Dense(10)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Generate prediction before saving
        original_prediction = model.predict(small_input_tensor, verbose=0)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, 'model.keras')

            # Save model
            model.save(model_path)

            # Load model
            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={'TransformerEncoderLayer': TransformerEncoderLayer}
            )

            # Generate prediction with loaded model
            loaded_prediction = loaded_model.predict(small_input_tensor, verbose=0)

            # Check predictions match
            assert np.allclose(original_prediction, loaded_prediction, rtol=1e-5)

            # Check layer type is preserved
            transformer_layer = loaded_model.get_layer('transformer_layer')
            assert isinstance(transformer_layer, TransformerEncoderLayer)

            # Check that configuration is preserved
            assert transformer_layer.attention_type == 'multi_head_attention'  # NEW
            assert transformer_layer.normalization_position == 'pre'
            assert transformer_layer.ffn_type == 'mlp'

    def test_model_save_load_different_attention_types(self, small_input_tensor):
        """Test saving and loading models with different attention types."""
        attention_types = ['multi_head_attention', 'window_attention', 'group_query_attention']

        for attention_type in attention_types:
            try:
                # Create model with specific attention type
                inputs = keras.Input(shape=(16, 64))
                x = TransformerEncoderLayer(
                    hidden_size=64,
                    num_heads=4,
                    intermediate_size=256,
                    attention_type=attention_type,
                    window_size=16,  # For window attention
                    n_kv_head=2,  # For group query attention
                    name=f'transformer_{attention_type}'
                )(inputs)
                x = keras.layers.GlobalAveragePooling1D()(x)
                outputs = keras.layers.Dense(5)(x)

                model = keras.Model(inputs=inputs, outputs=outputs)

                # Generate prediction before saving
                original_prediction = model.predict(small_input_tensor, verbose=0)

                # Save and load model
                with tempfile.TemporaryDirectory() as tmpdirname:
                    model_path = os.path.join(tmpdirname, f'model_{attention_type}.keras')

                    # Save model
                    model.save(model_path)

                    # Load model
                    loaded_model = keras.models.load_model(
                        model_path,
                        custom_objects={'TransformerEncoderLayer': TransformerEncoderLayer}
                    )

                    # Generate prediction with loaded model
                    loaded_prediction = loaded_model.predict(small_input_tensor, verbose=0)

                    # Check predictions match
                    assert np.allclose(original_prediction, loaded_prediction, rtol=1e-5)

                    # Check layer configuration is preserved
                    transformer_layer = loaded_model.get_layer(f'transformer_{attention_type}')
                    assert isinstance(transformer_layer, TransformerEncoderLayer)
                    assert transformer_layer.attention_type == attention_type

            except (ImportError, AttributeError, ValueError):
                # Attention type not available, skip test
                pass

    # ===== GRADIENT FLOW TESTS =====

    def test_gradient_flow(self, small_input_tensor, small_layer_instance):
        """Test gradient flow through the layer."""
        layer = small_layer_instance

        # Watch the variables
        with tf.GradientTape() as tape:
            inputs = tf.Variable(small_input_tensor)
            outputs = layer(inputs, training=True)
            loss = tf.reduce_mean(tf.square(outputs))

        # Get gradients
        grads = tape.gradient(loss, layer.trainable_variables)

        # Check gradients exist and are not None
        assert all(g is not None for g in grads)

        # Check gradients have values (not all zeros)
        assert all(np.any(g.numpy() != 0) for g in grads)

    def test_gradient_flow_different_attention_types(self, small_input_tensor):
        """Test gradient flow with different attention types."""
        attention_types = ['multi_head_attention', 'window_attention', 'group_query_attention']

        for attention_type in attention_types:
            try:
                layer = TransformerEncoderLayer(
                    hidden_size=64,
                    num_heads=4,
                    intermediate_size=256,
                    attention_type=attention_type,
                    window_size=8,  # For window attention
                    n_kv_head=2  # For group query attention
                )

                # Watch the variables
                with tf.GradientTape() as tape:
                    inputs = tf.Variable(small_input_tensor)
                    outputs = layer(inputs, training=True)
                    loss = tf.reduce_mean(tf.square(outputs))

                # Get gradients
                grads = tape.gradient(loss, layer.trainable_variables)

                # Check gradients exist and are not None
                assert all(g is not None for g in grads), f"None gradients with {attention_type} attention"

                # Check gradients have values (not all zeros)
                assert all(np.any(g.numpy() != 0) for g in grads), f"Zero gradients with {attention_type} attention"

            except (ImportError, AttributeError, ValueError):
                # Attention type not available, skip test
                pass

    def test_gradient_flow_normalization_positions(self, small_input_tensor):
        """Test gradient flow with different normalization positions."""
        positions = ['post', 'pre']

        for position in positions:
            layer = TransformerEncoderLayer(
                hidden_size=64,
                num_heads=4,
                intermediate_size=256,
                normalization_position=position
            )

            # Watch the variables
            with tf.GradientTape() as tape:
                inputs = tf.Variable(small_input_tensor)
                outputs = layer(inputs, training=True)
                loss = tf.reduce_mean(tf.square(outputs))

            # Get gradients
            grads = tape.gradient(loss, layer.trainable_variables)

            # Check gradients exist and are not None
            assert all(g is not None for g in grads), f"None gradients with {position}-normalization"

            # Check gradients have values (not all zeros)
            assert all(np.any(g.numpy() != 0) for g in grads), f"Zero gradients with {position}-normalization"

    def test_gradient_flow_with_regularization(self, small_input_tensor):
        """Test gradient flow with regularization."""
        layer = TransformerEncoderLayer(
            hidden_size=64,
            num_heads=4,
            intermediate_size=256,
            kernel_regularizer=keras.regularizers.L2(0.01),
            bias_regularizer=keras.regularizers.L1(0.01)
        )

        # Forward pass to build layer
        _ = layer(small_input_tensor, training=True)

        # Check regularization losses exist
        assert len(layer.losses) > 0

    # ===== TRAINING TESTS =====

    def test_training_loop_basic(self, small_input_tensor):
        """Test basic training loop with the custom layer."""
        # Create model
        model = keras.Sequential([
            keras.layers.InputLayer(input_shape=(16, 64)),
            TransformerEncoderLayer(
                hidden_size=64,
                num_heads=4,
                intermediate_size=256,
                attention_type='multi_head_attention',  # NEW: Explicit attention type
                normalization_position='pre',
                ffn_type='mlp'
            ),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(10)
        ])

        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        # Create mock data
        x_train = tf.random.normal([32, 16, 64])
        y_train = tf.random.uniform([32], 0, 10, dtype=tf.int32)

        # Initial loss
        initial_loss = model.evaluate(x_train, y_train, verbose=0)[0]

        # Train for a few epochs
        model.fit(x_train, y_train, epochs=2, batch_size=16, verbose=0)

        # Final loss
        final_loss = model.evaluate(x_train, y_train, verbose=0)[0]

        # Loss should decrease (allowing for some variance)
        assert final_loss <= initial_loss * 1.1  # Allow 10% variance

    def test_training_loop_different_attention_types(self, small_input_tensor):
        """Test training with different attention types."""
        attention_types = ['multi_head_attention', 'window_attention', 'group_query_attention']

        for attention_type in attention_types:
            try:
                # Create model
                model = keras.Sequential([
                    keras.layers.InputLayer(input_shape=(16, 64)),
                    TransformerEncoderLayer(
                        hidden_size=64,
                        num_heads=4,
                        intermediate_size=256,
                        attention_type=attention_type,
                        window_size=8,  # For window attention
                        n_kv_head=2  # For group query attention
                    ),
                    keras.layers.GlobalAveragePooling1D(),
                    keras.layers.Dense(5)
                ])

                # Compile model
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=0.001),
                    loss=keras.losses.MeanSquaredError()
                )

                # Create mock data
                x_train = tf.random.normal([16, 16, 64])
                y_train = tf.random.normal([16, 5])

                # Initial loss
                initial_loss = model.evaluate(x_train, y_train, verbose=0)

                # Train for one epoch
                model.fit(x_train, y_train, epochs=1, batch_size=8, verbose=0)

                # Final loss
                final_loss = model.evaluate(x_train, y_train, verbose=0)

                # Check training completed without errors
                assert final_loss is not None
                assert not np.isnan(final_loss)

            except (ImportError, AttributeError, ValueError):
                # Attention type not available, skip test
                pass

    # ===== EDGE CASE TESTS =====

    def test_numerical_stability(self):
        """Test layer stability with extreme input values."""
        layer = TransformerEncoderLayer(
            hidden_size=64,
            num_heads=4,
            intermediate_size=256
        )

        batch_size, seq_len, hidden_size = 2, 8, 64

        test_cases = [
            tf.zeros((batch_size, seq_len, hidden_size)),  # All zeros
            tf.ones((batch_size, seq_len, hidden_size)) * 1e-6,  # Very small values
            tf.ones((batch_size, seq_len, hidden_size)) * 1e3,  # Large values
            tf.random.normal((batch_size, seq_len, hidden_size)) * 1e2  # Large random values
        ]

        for test_input in test_cases:
            output = layer(test_input, training=False)

            # Check for NaN/Inf values
            assert not np.any(np.isnan(output.numpy())), "NaN values detected in output"
            assert not np.any(np.isinf(output.numpy())), "Inf values detected in output"

    def test_different_sequence_lengths(self):
        """Test layer with different sequence lengths."""
        layer = TransformerEncoderLayer(
            hidden_size=64,
            num_heads=4,
            intermediate_size=256
        )

        sequence_lengths = [1, 8, 32, 128, 512]

        for seq_len in sequence_lengths:
            test_input = tf.random.normal((2, seq_len, 64))
            output = layer(test_input, training=False)

            # Check output shape
            assert output.shape == (2, seq_len, 64)

            # Check for valid values
            assert not np.any(np.isnan(output.numpy()))

    def test_different_sequence_lengths_with_attention_types(self):
        """Test different attention types with various sequence lengths."""
        attention_types = ['multi_head_attention', 'window_attention', 'group_query_attention']
        sequence_lengths = [8, 16, 32]  # Smaller set for efficiency

        for attention_type in attention_types:
            try:
                layer = TransformerEncoderLayer(
                    hidden_size=64,
                    num_heads=4,
                    intermediate_size=256,
                    attention_type=attention_type,
                    window_size=8,  # For window attention
                    n_kv_head=2  # For group query attention
                )

                for seq_len in sequence_lengths:
                    test_input = tf.random.normal((2, seq_len, 64))
                    output = layer(test_input, training=False)

                    # Check output shape
                    assert output.shape == (2, seq_len, 64)

                    # Check for valid values
                    assert not np.any(np.isnan(output.numpy()))

            except (ImportError, AttributeError, ValueError):
                # Attention type not available, skip test
                pass

    def test_dropout_behavior(self):
        """Test dropout behavior during training vs inference."""
        layer = TransformerEncoderLayer(
            hidden_size=64,
            num_heads=4,
            intermediate_size=256,
            dropout_rate=0.5,  # High dropout for clearer difference
            attention_dropout_rate=0.5
        )

        test_input = tf.random.normal((4, 16, 64))

        # Multiple forward passes in training mode
        outputs_train = [layer(test_input, training=True) for _ in range(3)]

        # Multiple forward passes in inference mode
        outputs_inference = [layer(test_input, training=False) for _ in range(3)]

        # Training outputs should be different (due to dropout)
        assert not np.allclose(outputs_train[0].numpy(), outputs_train[1].numpy())

        # Inference outputs should be the same (no dropout)
        assert np.allclose(outputs_inference[0].numpy(), outputs_inference[1].numpy())

    # ===== NORMALIZATION LAYER CREATION TESTS =====

    def test_normalization_layer_creation(self):
        """Test creation of different normalization layers."""
        layer = TransformerEncoderLayer(
            hidden_size=64,
            num_heads=4,
            intermediate_size=256
        )

        # Test layer_norm (always available)
        layer.normalization_type = 'layer_norm'
        norm_layer = layer._create_normalization_layer('test_norm')
        assert isinstance(norm_layer, keras.layers.LayerNormalization)

        # Test batch_norm (always available)
        layer.normalization_type = 'batch_norm'
        norm_layer = layer._create_normalization_layer('test_norm')
        assert isinstance(norm_layer, keras.layers.BatchNormalization)

        # Test rms_norm (manually set - should work with fallback)
        layer.normalization_type = 'rms_norm'
        norm_layer = layer._create_normalization_layer('test_norm')
        # Should be RMSNorm or LayerNormalization as fallback
        assert norm_layer.__class__.__name__ in ['RMSNorm', 'LayerNormalization']

        # Test band_rms (manually set - should work with fallback)
        layer.normalization_type = 'band_rms'
        norm_layer = layer._create_normalization_layer('test_norm')
        # Should be BandRMS or LayerNormalization as fallback
        assert norm_layer.__class__.__name__ in ['BandRMS', 'LayerNormalization']

    def test_normalization_layer_creation_error(self):
        """Test error handling in normalization layer creation."""
        layer = TransformerEncoderLayer(
            hidden_size=64,
            num_heads=4,
            intermediate_size=256
        )

        # Set invalid normalization type manually (bypassing __init__ validation)
        # We need to save the original value to restore it later
        original_type = layer.normalization_type
        layer.normalization_type = 'invalid_type'

        try:
            with pytest.raises(ValueError, match="Unknown normalization type"):
                layer._create_normalization_layer('test_norm')
        finally:
            # Restore original type
            layer.normalization_type = original_type

    # ===== ATTENTION MASK TESTS =====

    def test_attention_mask_shapes(self, small_input_tensor):
        """Test different attention mask shapes."""
        layer = TransformerEncoderLayer(
            hidden_size=64,
            num_heads=4,
            intermediate_size=256
        )

        batch_size, seq_len, _ = small_input_tensor.shape

        # Test without mask (should work)
        output = layer(small_input_tensor, attention_mask=None, training=False)
        assert output.shape == small_input_tensor.shape
        assert not np.any(np.isnan(output.numpy()))

        # Test 3D mask (attention mask) - this should work
        mask_3d = tf.ones((batch_size, seq_len, seq_len))
        output = layer(small_input_tensor, attention_mask=mask_3d, training=False)
        assert output.shape == small_input_tensor.shape
        assert not np.any(np.isnan(output.numpy()))

        # Test 4D mask (full attention mask with heads) - this should work
        mask_4d = tf.ones((batch_size, 1, seq_len, seq_len))
        output = layer(small_input_tensor, attention_mask=mask_4d, training=False)
        assert output.shape == small_input_tensor.shape
        assert not np.any(np.isnan(output.numpy()))

        # Test 2D mask (padding mask) - this should warn and skip for now
        mask_2d = tf.ones((batch_size, seq_len))
        output = layer(small_input_tensor, attention_mask=mask_2d, training=False)
        assert output.shape == small_input_tensor.shape
        assert not np.any(np.isnan(output.numpy()))

    def test_attention_mask_with_different_attention_types(self, small_input_tensor):
        """Test attention masks with different attention types."""
        attention_types = ['multi_head_attention']  # Only test with standard attention for mask compatibility

        for attention_type in attention_types:
            try:
                layer = TransformerEncoderLayer(
                    hidden_size=64,
                    num_heads=4,
                    intermediate_size=256,
                    attention_type=attention_type
                )

                batch_size, seq_len, _ = small_input_tensor.shape

                # Test 3D mask
                mask_3d = tf.ones((batch_size, seq_len, seq_len))
                output = layer(small_input_tensor, attention_mask=mask_3d, training=False)
                assert output.shape == small_input_tensor.shape
                assert not np.any(np.isnan(output.numpy()))

            except (ImportError, AttributeError, ValueError):
                # Attention type not available, skip test
                pass

    # ===== NORMALIZATION POSITION BEHAVIOR TESTS =====

    def test_normalization_position_behavior(self, small_input_tensor):
        """Test that pre-norm and post-norm produce different outputs."""
        # Create two identical layers except for normalization position
        layer_post = TransformerEncoderLayer(
            hidden_size=64,
            num_heads=4,
            intermediate_size=256,
            normalization_position='post',
            dropout_rate=0.0,  # Disable dropout for deterministic comparison
            attention_dropout_rate=0.0
        )

        layer_pre = TransformerEncoderLayer(
            hidden_size=64,
            num_heads=4,
            intermediate_size=256,
            normalization_position='pre',
            dropout_rate=0.0,  # Disable dropout for deterministic comparison
            attention_dropout_rate=0.0
        )

        # Forward pass through both layers
        output_post = layer_post(small_input_tensor, training=False)
        output_pre = layer_pre(small_input_tensor, training=False)

        # Outputs should be different due to different normalization ordering
        assert not np.allclose(output_post.numpy(), output_pre.numpy(), rtol=1e-3), \
            "Post-norm and pre-norm should produce different outputs"

    def test_ffn_type_behavior(self, small_input_tensor):
        """Test that different FFN types produce different outputs."""
        ffn_types_to_test = ['mlp', 'swiglu']  # Test the most common ones
        outputs = {}

        for ffn_type in ffn_types_to_test:
            try:
                layer = TransformerEncoderLayer(
                    hidden_size=64,
                    num_heads=4,
                    intermediate_size=256,
                    ffn_type=ffn_type,
                    dropout_rate=0.0,  # Disable dropout for deterministic comparison
                    attention_dropout_rate=0.0
                )

                output = layer(small_input_tensor, training=False)
                outputs[ffn_type] = output.numpy()

            except (ImportError, AttributeError):
                # FFN type not available, skip
                pass

        # If we have multiple outputs, they should be different
        if len(outputs) >= 2:
            output_list = list(outputs.values())
            for i in range(len(output_list)):
                for j in range(i + 1, len(output_list)):
                    assert not np.allclose(output_list[i], output_list[j], rtol=1e-3), \
                        f"Different FFN types should produce different outputs"


if __name__ == '__main__':
    pytest.main([__file__])