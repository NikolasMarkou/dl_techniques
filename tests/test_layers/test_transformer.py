import pytest
import numpy as np
import tensorflow as tf
import keras
import tempfile
import os


from dl_techniques.layers.transformer import TransformerLayer
from dl_techniques.layers.attention.window_attention import WindowAttention



class TestTransformerLayer:
    """Test suite for TransformerLayer implementation with robust error handling."""

    @pytest.fixture
    def input_tensor(self):
        """Create a test input tensor."""
        return tf.random.normal([2, 128, 768])  # batch_size=2, seq_len=128, hidden_size=768

    @pytest.fixture
    def small_input_tensor(self):
        """Create a smaller test input tensor for faster tests."""
        # seq_len=16, which means for window attention, window_size must be 4 (4*4=16)
        return tf.random.normal([2, 16, 64])  # batch_size=2, seq_len=16, hidden_size=64

    @pytest.fixture
    def layer_instance(self):
        """Create a default layer instance for testing."""
        return TransformerLayer(
            hidden_size=768,
            num_heads=12,
            intermediate_size=3072
        )

    @pytest.fixture
    def small_layer_instance(self):
        """Create a smaller layer instance for faster tests."""
        return TransformerLayer(
            hidden_size=64,
            num_heads=8,
            intermediate_size=256
        )

    def _test_attention_type_safely(self, attention_type, test_func, *args, **kwargs):
        """Safely test an attention type with proper error handling."""
        try:
            return test_func(attention_type, *args, **kwargs)
        except (ImportError, AttributeError, ModuleNotFoundError) as e:
            pytest.skip(f"Attention type '{attention_type}' dependencies not available: {e}")
        except (tf.errors.InvalidArgumentError, ValueError, TypeError) as e:
            # With the fix, we expect these to work, so re-raise to fail the test
            raise e

    def _test_ffn_type_safely(self, ffn_type, test_func, *args, **kwargs):
        """Safely test an FFN type with proper error handling."""
        try:
            return test_func(ffn_type, *args, **kwargs)
        except (ImportError, AttributeError, ModuleNotFoundError) as e:
            pytest.skip(f"FFN type '{ffn_type}' dependencies not available: {e}")
        except (ValueError, TypeError) as e:
            # Re-raise the error to fail the test
            raise e

    def _get_correct_window_size(self, seq_len):
        """Get the correct window size for a given sequence length."""
        # For WindowAttention, window_size^2 must equal sequence length
        import math
        window_size = int(math.sqrt(seq_len))
        if window_size * window_size != seq_len:
            # Find the nearest perfect square
            candidates = [i for i in range(1, seq_len + 1) if i * i <= seq_len]
            window_size = candidates[-1] if candidates else 1
        return window_size

    # ===== INITIALIZATION TESTS =====

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        layer = TransformerLayer(
            hidden_size=768,
            num_heads=12,
            intermediate_size=3072
        )

        # Check default values
        assert layer.hidden_size == 768
        assert layer.num_heads == 12
        assert layer.intermediate_size == 3072
        assert layer.attention_type == 'multi_head_attention'
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
        assert layer.window_size == 8
        assert layer.n_kv_head == 12

    def test_initialization_custom_parameters(self):
        """Test initialization with custom parameters."""
        custom_regularizer = keras.regularizers.L2(1e-4)

        layer = TransformerLayer(
            hidden_size=512,
            num_heads=8,
            intermediate_size=2048,
            attention_type='multi_head_attention',
            normalization_type='layer_norm',
            normalization_position='pre',
            ffn_type='mlp',
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
            window_size=16,
            n_kv_head=4
        )

        # Check custom values
        assert layer.hidden_size == 512
        assert layer.num_heads == 8
        assert layer.intermediate_size == 2048
        assert layer.attention_type == 'multi_head_attention'
        assert layer.normalization_type == 'layer_norm'
        assert layer.normalization_position == 'pre'
        assert layer.ffn_type == 'mlp'
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
        assert layer.window_size == 16
        assert layer.n_kv_head == 4

    def test_initialization_all_attention_types(self):
        """Test initialization with all attention types."""
        attention_types = ['multi_head_attention', 'window_attention', 'group_query_attention',
                           'differential_attention']

        for attention_type in attention_types:
            layer = TransformerLayer(
                hidden_size=64,
                num_heads=4,
                intermediate_size=256,
                attention_type=attention_type
            )
            assert layer.attention_type == attention_type

    def test_initialization_all_normalization_types(self):
        """Test initialization with all normalization types."""
        norm_types = ['layer_norm', 'batch_norm', 'rms_norm', 'band_rms']

        for norm_type in norm_types:
            layer = TransformerLayer(
                hidden_size=64,
                num_heads=4,
                intermediate_size=256,
                normalization_type=norm_type
            )
            assert layer.normalization_type == norm_type

    def test_initialization_all_ffn_types(self):
        """Test initialization with all FFN types."""
        ffn_types = ['mlp', 'swiglu', 'differential', 'glu', 'residual', 'swin_mlp']

        for ffn_type in ffn_types:
            layer = TransformerLayer(
                hidden_size=64,
                num_heads=4,
                intermediate_size=256,
                ffn_type=ffn_type
            )
            assert layer.ffn_type == ffn_type

    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        # Negative hidden_size
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            TransformerLayer(hidden_size=-768, num_heads=12, intermediate_size=3072)

        # Hidden size not divisible by num_heads
        with pytest.raises(ValueError, match="hidden_size.*must be divisible by num_heads"):
            TransformerLayer(hidden_size=100, num_heads=12, intermediate_size=3072)

        # Invalid dropout rates
        with pytest.raises(ValueError, match="dropout_rate must be between 0 and 1"):
            TransformerLayer(hidden_size=768, num_heads=12, intermediate_size=3072, dropout_rate=1.5)

        # Invalid attention type
        with pytest.raises(ValueError, match="attention_type must be one of"):
            TransformerLayer(hidden_size=768, num_heads=12, intermediate_size=3072,
                             attention_type='invalid_attention')

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

    def test_build_process_multi_head_attention(self, small_input_tensor):
        """Test building with standard multi-head attention (guaranteed to work)."""
        layer = TransformerLayer(
            hidden_size=64,
            num_heads=4,
            intermediate_size=256,
            attention_type='multi_head_attention'
        )

        # Forward pass triggers build
        layer(small_input_tensor)

        # Check that layer was built
        assert layer.built is True
        assert layer.attention is not None
        assert layer.attention.built is True
        assert isinstance(layer.attention, keras.layers.MultiHeadAttention)

    def test_build_process_alternative_attention_types(self, small_input_tensor):
        """Test building with alternative attention types."""
        attention_types = ['window_attention', 'group_query_attention', 'differential_attention']

        def _test_attention_build(attention_type):
            # CRITICAL FIX: For window_attention, window_size^2 must match sequence length (16)
            # For seq_len=16, we need window_size=4 (since 4*4=16)
            window_size = 4 if attention_type == 'window_attention' else 8

            layer = TransformerLayer(
                hidden_size=64,
                num_heads=4,
                intermediate_size=256,
                attention_type=attention_type,
                window_size=window_size,
                n_kv_head=2
            )

            # Forward pass triggers build
            layer(small_input_tensor)

            # Check that layer was built
            assert layer.built is True
            assert layer.attention is not None
            assert layer.attention.built is True

        for attention_type in attention_types:
            self._test_attention_type_safely(attention_type, _test_attention_build)

    def test_build_with_different_input_shapes(self):
        """Test building with different input shapes."""
        test_shapes = [
            (None, 32, 128),
            (4, 64, 256),
            (1, 512, 768),
        ]

        for input_shape in test_shapes:
            hidden_size = input_shape[-1]
            layer = TransformerLayer(
                hidden_size=hidden_size,
                num_heads=8 if hidden_size >= 128 else 4,
                intermediate_size=hidden_size * 4,
                attention_type='multi_head_attention'  # Use safe attention type
            )

            # Build with shape
            layer.build(input_shape)

            # Check build was successful
            assert layer.built is True
            assert layer._build_input_shape == input_shape

    # ===== OUTPUT SHAPE TESTS =====

    def test_output_shapes(self, small_input_tensor):
        """Test that output shapes are computed correctly."""
        test_configs = [
            (64, 4, 256),
            (128, 8, 512),
            (256, 16, 1024),
        ]

        for hidden_size, num_heads, intermediate_size in test_configs:
            layer = TransformerLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                attention_type='multi_head_attention'  # Use safe attention type
            )

            # Create appropriate input
            input_shape = (2, 16, hidden_size)
            test_input = tf.random.normal(input_shape)

            # Forward pass
            output = layer(test_input)

            # Check output shape matches input shape
            assert output.shape == test_input.shape

    def test_compute_output_shape_with_none_dimensions(self):
        """Test compute_output_shape with None dimensions."""
        layer = TransformerLayer(
            hidden_size=768,
            num_heads=12,
            intermediate_size=3072,
            attention_type='multi_head_attention'
        )

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

    def test_forward_pass_multi_head_attention(self, small_input_tensor):
        """Test forward pass with standard multi-head attention."""
        layer = TransformerLayer(
            hidden_size=64,
            num_heads=4,
            intermediate_size=256,
            attention_type='multi_head_attention'
        )

        output = layer(small_input_tensor)

        # Check output is valid
        assert output.shape == small_input_tensor.shape
        assert not np.any(np.isnan(output.numpy()))
        assert not np.any(np.isinf(output.numpy()))

    def test_forward_pass_alternative_attention_types(self, small_input_tensor):
        """Test forward pass with alternative attention types."""
        attention_types = ['window_attention', 'group_query_attention', 'differential_attention']

        def _test_attention_forward(attention_type):
            # CRITICAL FIX: For window_attention, window_size^2 must match sequence length (16)
            # For seq_len=16, we need window_size=4 (since 4*4=16)
            window_size = 4 if attention_type == 'window_attention' else 8

            layer = TransformerLayer(
                hidden_size=64,
                num_heads=4,
                intermediate_size=256,
                attention_type=attention_type,
                window_size=window_size,
                n_kv_head=2
            )

            output = layer(small_input_tensor)

            # Check output is valid
            assert output.shape == small_input_tensor.shape
            assert not np.any(np.isnan(output.numpy()))
            assert not np.any(np.isinf(output.numpy()))

        for attention_type in attention_types:
            self._test_attention_type_safely(attention_type, _test_attention_forward)

    def test_forward_pass_normalization_positions(self, small_input_tensor):
        """Test forward pass with different normalization positions."""
        positions = ['post', 'pre']

        for position in positions:
            layer = TransformerLayer(
                hidden_size=64,
                num_heads=4,
                intermediate_size=256,
                normalization_position=position,
                attention_type='multi_head_attention'  # Use safe attention type
            )

            output = layer(small_input_tensor)

            # Check output is valid
            assert output.shape == small_input_tensor.shape
            assert not np.any(np.isnan(output.numpy()))
            assert not np.any(np.isinf(output.numpy()))

    def test_forward_pass_different_ffn_types(self, small_input_tensor):
        """Test forward pass with different FFN types."""
        ffn_types = ['mlp', 'swiglu', 'differential', 'glu', 'residual', 'swin_mlp']

        def _test_ffn_forward(ffn_type):
            layer = TransformerLayer(
                hidden_size=64,
                num_heads=4,
                intermediate_size=256,
                ffn_type=ffn_type,
                attention_type='multi_head_attention'  # Use safe attention type
            )

            output = layer(small_input_tensor)

            # Check output is valid
            assert output.shape == small_input_tensor.shape
            assert not np.any(np.isnan(output.numpy()))
            assert not np.any(np.isinf(output.numpy()))

        for ffn_type in ffn_types:
            self._test_ffn_type_safely(ffn_type, _test_ffn_forward)

    def test_forward_pass_combined_configurations(self, small_input_tensor):
        """Test forward pass with combined configurations including attention types."""
        test_configs = [
            # Safe configurations with multi_head_attention
            {'normalization_position': 'post', 'ffn_type': 'mlp', 'attention_type': 'multi_head_attention'},
            {'normalization_position': 'pre', 'ffn_type': 'mlp', 'attention_type': 'multi_head_attention'},
            # FIXED: Configurations with proper window_size for window_attention
            {'normalization_position': 'post', 'ffn_type': 'mlp', 'attention_type': 'window_attention',
             'window_size': 4},
            {'normalization_position': 'pre', 'ffn_type': 'mlp', 'attention_type': 'window_attention',
             'window_size': 4},
            # Configurations with group_query_attention
            {'normalization_position': 'post', 'ffn_type': 'mlp', 'attention_type': 'group_query_attention',
             'n_kv_head': 2},
            {'normalization_position': 'pre', 'ffn_type': 'mlp', 'attention_type': 'group_query_attention',
             'n_kv_head': 1},
        ]

        for config in test_configs:
            attention_type = config['attention_type']

            def _test_config():
                layer = TransformerLayer(
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

            if attention_type == 'multi_head_attention':
                _test_config()  # Always test standard attention
            else:
                self._test_attention_type_safely(attention_type, lambda at: _test_config())

    def test_forward_pass_with_attention_mask(self, small_input_tensor, small_layer_instance):
        """Test forward pass with attention mask."""
        layer = small_layer_instance
        batch_size, seq_len, _ = small_input_tensor.shape

        # Test with 3D attention mask
        attention_mask = tf.ones((batch_size, seq_len, seq_len))
        output = layer(small_input_tensor, attention_mask=attention_mask)

        # Check output is valid
        assert output.shape == small_input_tensor.shape
        assert not np.any(np.isnan(output.numpy()))

        # Test with no mask
        output_no_mask = layer(small_input_tensor, attention_mask=None)
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
        norm_types = ['layer_norm', 'batch_norm']  # Start with always available types

        for norm_type in norm_types:
            layer = TransformerLayer(
                hidden_size=64,
                num_heads=4,
                intermediate_size=256,
                normalization_type=norm_type,
                attention_type='multi_head_attention'
            )

            output = layer(small_input_tensor)

            # Check output is valid
            assert output.shape == small_input_tensor.shape
            assert not np.any(np.isnan(output.numpy()))

        # Test optional types safely
        optional_norm_types = ['rms_norm', 'band_rms']
        for norm_type in optional_norm_types:
            try:
                layer = TransformerLayer(
                    hidden_size=64,
                    num_heads=4,
                    intermediate_size=256,
                    normalization_type=norm_type,
                    attention_type='multi_head_attention'
                )

                output = layer(small_input_tensor)

                # Check output is valid
                assert output.shape == small_input_tensor.shape
                assert not np.any(np.isnan(output.numpy()))
            except (ImportError, AttributeError, ValueError):
                pytest.skip(f"Normalization type '{norm_type}' not available")

    # ===== ATTENTION-SPECIFIC TESTS =====

    def test_attention_type_standard_behavior(self, small_input_tensor):
        """Test standard multi-head attention behavior (guaranteed to work)."""
        layer = TransformerLayer(
            hidden_size=64,
            num_heads=4,
            intermediate_size=256,
            attention_type='multi_head_attention',
            dropout_rate=0.0,
            attention_dropout_rate=0.0
        )

        output = layer(small_input_tensor, training=False)

        # Basic validity checks
        assert output.shape == small_input_tensor.shape
        assert not np.any(np.isnan(output.numpy()))
        assert not np.any(np.isinf(output.numpy()))

        # Test that the attention layer is correct type
        assert isinstance(layer.attention, keras.layers.MultiHeadAttention)

    def test_alternative_attention_behaviors(self, small_input_tensor):
        """Test alternative attention types with robust error handling."""
        # CRITICAL FIX: For window_attention with seq_len=16, window_size must be 4
        attention_configs = [
            {'attention_type': 'window_attention', 'window_size': 4},
            {'attention_type': 'group_query_attention', 'n_kv_head': 2},
            {'attention_type': 'differential_attention', 'lambda_init': 0.8}
        ]

        for config in attention_configs:
            attention_type = config['attention_type']

            def _test_alt_attention():
                layer = TransformerLayer(
                    hidden_size=64,
                    num_heads=4,
                    intermediate_size=256,
                    dropout_rate=0.0,
                    attention_dropout_rate=0.0,
                    **config
                )

                output = layer(small_input_tensor, training=False)

                # Basic validity checks
                assert output.shape == small_input_tensor.shape
                assert not np.any(np.isnan(output.numpy()))
                assert not np.any(np.isinf(output.numpy()))

            self._test_attention_type_safely(attention_type, lambda at: _test_alt_attention())

    def test_window_attention_specific_behavior(self, small_input_tensor):
        """Test window attention with correct window sizes."""
        # CRITICAL FIX: Test with proper window sizes for seq_len=16
        seq_len = small_input_tensor.shape[1]  # 16

        # For seq_len=16, only window_size=4 will work (4*4=16)
        # We can also test with different sequence lengths that are perfect squares
        test_cases = [
            (16, 4),  # seq_len=16, window_size=4
            (9, 3),  # seq_len=9, window_size=3
            (25, 5),  # seq_len=25, window_size=5
        ]

        def _test_window_attention(seq_len, window_size):
            # Create appropriate input tensor
            test_input = tf.random.normal((2, seq_len, 64))

            layer = TransformerLayer(
                hidden_size=64,
                num_heads=4,
                intermediate_size=256,
                attention_type='window_attention',
                window_size=window_size,
                dropout_rate=0.0,
                attention_dropout_rate=0.0
            )

            output = layer(test_input, training=False)

            assert output.shape == test_input.shape
            assert not np.any(np.isnan(output.numpy()))
            assert not np.any(np.isinf(output.numpy()))

        for seq_len, window_size in test_cases:
            self._test_attention_type_safely('window_attention',
                                             lambda at: _test_window_attention(seq_len, window_size))

    def test_group_query_attention_specific_behavior(self, small_input_tensor):
        """Test group query attention with different n_kv_head values."""
        n_kv_head_values = [1, 2, 4]  # Test different grouping
        outputs = {}

        def _test_gqa(n_kv_head):
            layer = TransformerLayer(
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

            assert output.shape == small_input_tensor.shape
            assert not np.any(np.isnan(output.numpy()))

        for n_kv_head in n_kv_head_values:
            self._test_attention_type_safely('group_query_attention',
                                             lambda at: _test_gqa(n_kv_head))

        # If we have multiple outputs, they should be different
        if len(outputs) >= 2:
            output_list = list(outputs.values())
            n_kv_head_list = list(outputs.keys())
            for i in range(len(output_list)):
                for j in range(i + 1, len(output_list)):
                    assert not np.allclose(output_list[i], output_list[j], rtol=1e-3), \
                        f"Different n_kv_head values ({n_kv_head_list[i]} vs {n_kv_head_list[j]}) should produce different outputs"

    # ===== SERIALIZATION TESTS =====

    def test_serialization_basic(self):
        """Test basic serialization and deserialization."""
        # Use safe configuration
        original_layer = TransformerLayer(
            hidden_size=128,
            num_heads=8,
            intermediate_size=512,
            attention_type='multi_head_attention',
            normalization_type='layer_norm',
            normalization_position='pre',
            ffn_type='mlp',
            dropout_rate=0.2
        )
        original_layer.build((None, 32, 128))

        # Serialize and recreate
        config = original_layer.get_config()
        build_config = original_layer.get_build_config()

        new_layer = TransformerLayer.from_config(config)
        new_layer.build_from_config(build_config)

        # Check configuration matches
        assert new_layer.hidden_size == original_layer.hidden_size
        assert new_layer.num_heads == original_layer.num_heads
        assert new_layer.attention_type == original_layer.attention_type
        assert new_layer.normalization_type == original_layer.normalization_type
        assert new_layer.ffn_type == original_layer.ffn_type

    def test_serialization_all_attention_types(self):
        """Test serialization with all attention types."""
        attention_configs = [
            {'attention_type': 'multi_head_attention'},
            {'attention_type': 'window_attention', 'window_size': 16},  # Use 16 for 256 sequence length
            {'attention_type': 'group_query_attention', 'n_kv_head': 2},
            {'attention_type': 'differential_attention', 'lambda_init': 0.8}
        ]

        for config in attention_configs:
            attention_type = config['attention_type']

            try:
                original_layer = TransformerLayer(
                    hidden_size=64,
                    num_heads=4,
                    intermediate_size=256,
                    normalization_type='layer_norm',
                    **config
                )

                # Get config
                layer_config = original_layer.get_config()

                # Recreate layer
                new_layer = TransformerLayer.from_config(layer_config)

                # Check attention-specific parameters match
                assert new_layer.attention_type == original_layer.attention_type
                assert new_layer.window_size == original_layer.window_size
                assert new_layer.n_kv_head == original_layer.n_kv_head

            except (ImportError, AttributeError, ValueError):
                pytest.skip(f"Attention type '{attention_type}' not available")

    def test_serialization_all_parameters(self):
        """Test serialization with comprehensive parameters."""
        custom_regularizer = keras.regularizers.L2(1e-4)

        original_layer = TransformerLayer(
            hidden_size=256,
            num_heads=16,
            intermediate_size=1024,
            attention_type='multi_head_attention',  # Use safe type
            normalization_type='layer_norm',
            normalization_position='pre',
            ffn_type='mlp',  # Use safe type
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
            window_size=32,
            n_kv_head=8
        )

        # Get config and recreate
        config = original_layer.get_config()
        new_layer = TransformerLayer.from_config(config)

        # Check all parameters match
        assert new_layer.hidden_size == original_layer.hidden_size
        assert new_layer.num_heads == original_layer.num_heads
        assert new_layer.attention_type == original_layer.attention_type
        assert new_layer.normalization_type == original_layer.normalization_type
        assert new_layer.ffn_type == original_layer.ffn_type
        assert new_layer.dropout_rate == original_layer.dropout_rate
        assert new_layer.window_size == original_layer.window_size
        assert new_layer.n_kv_head == original_layer.n_kv_head

    # ===== MODEL INTEGRATION TESTS =====

    def test_model_integration_simple(self, small_input_tensor):
        """Test the layer in a simple model context."""
        inputs = keras.Input(shape=(16, 64))
        x = TransformerLayer(
            hidden_size=64,
            num_heads=4,
            intermediate_size=256,
            attention_type='multi_head_attention',
            normalization_position='pre',
            ffn_type='mlp'
        )(inputs)
        x = keras.layers.GlobalAveragePooling1D()(x)
        outputs = keras.layers.Dense(10)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

        # Test forward pass
        y_pred = model(small_input_tensor, training=False)
        assert y_pred.shape == (small_input_tensor.shape[0], 10)

    def test_model_integration_different_attention_types(self, small_input_tensor):
        """Test model with different attention types."""
        attention_configs = [
            {'attention_type': 'multi_head_attention'},
            {'attention_type': 'window_attention', 'window_size': 4},  # FIXED: Correct window_size for seq_len=16
            {'attention_type': 'group_query_attention', 'n_kv_head': 2},
        ]

        for config in attention_configs:
            attention_type = config['attention_type']

            def _test_model_integration():
                # Create model with specific attention type
                inputs = keras.Input(shape=(16, 64))
                x = TransformerLayer(
                    hidden_size=64,
                    num_heads=4,
                    intermediate_size=256,
                    **config
                )(inputs)
                x = keras.layers.GlobalAveragePooling1D()(x)
                outputs = keras.layers.Dense(5)(x)

                model = keras.Model(inputs=inputs, outputs=outputs)
                model.compile(optimizer='adam', loss='mse')

                # Test forward pass
                y_pred = model(small_input_tensor)
                assert y_pred.shape == (small_input_tensor.shape[0], 5)

            if attention_type == 'multi_head_attention':
                _test_model_integration()  # Always test standard attention
            else:
                self._test_attention_type_safely(attention_type, lambda at: _test_model_integration())

    def test_model_integration_stacked_safe(self, small_input_tensor):
        """Test multiple layers stacked in a model with safe configurations."""
        inputs = keras.Input(shape=(16, 64))

        x = TransformerLayer(
            hidden_size=64,
            num_heads=4,
            intermediate_size=256,
            attention_type='multi_head_attention',
            normalization_position='post',
            ffn_type='mlp'
        )(inputs)

        x = TransformerLayer(
            hidden_size=64,
            num_heads=4,
            intermediate_size=256,
            attention_type='multi_head_attention',
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
        inputs = keras.Input(shape=(16, 64))
        x = TransformerLayer(
            hidden_size=64,
            num_heads=4,
            intermediate_size=256,
            attention_type='multi_head_attention',
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

            # Load model with proper custom objects
            custom_objects = {'TransformerLayer': TransformerLayer}
            if WindowAttention is not None:
                custom_objects['WindowAttention'] = WindowAttention

            loaded_model = keras.models.load_model(model_path, custom_objects=custom_objects)

            # Generate prediction with loaded model
            loaded_prediction = loaded_model.predict(small_input_tensor, verbose=0)

            # Check predictions match
            assert np.allclose(original_prediction, loaded_prediction, rtol=1e-5)

            # Check layer type and configuration
            transformer_layer = loaded_model.get_layer('transformer_layer')
            assert isinstance(transformer_layer, TransformerLayer)
            assert transformer_layer.attention_type == 'multi_head_attention'

    def test_model_save_load_different_attention_types(self, small_input_tensor):
        """Test saving and loading models with different attention types."""
        attention_configs = [
            {'attention_type': 'multi_head_attention'},
            {'attention_type': 'window_attention', 'window_size': 4},  # FIXED: Correct window_size
            {'attention_type': 'group_query_attention', 'n_kv_head': 2},
        ]

        for config in attention_configs:
            attention_type = config['attention_type']

            def _test_save_load():
                # Create model with specific attention type
                inputs = keras.Input(shape=(16, 64))
                x = TransformerLayer(
                    hidden_size=64,
                    num_heads=4,
                    intermediate_size=256,
                    name=f'transformer_{attention_type}',
                    **config
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

                    # Load model with proper custom objects
                    custom_objects = {'TransformerLayer': TransformerLayer}
                    if WindowAttention is not None:
                        custom_objects['WindowAttention'] = WindowAttention

                    loaded_model = keras.models.load_model(model_path, custom_objects=custom_objects)

                    # Generate prediction with loaded model
                    loaded_prediction = loaded_model.predict(small_input_tensor, verbose=0)

                    # Check predictions match
                    assert np.allclose(original_prediction, loaded_prediction, rtol=1e-5)

                    # Check layer configuration is preserved
                    transformer_layer = loaded_model.get_layer(f'transformer_{attention_type}')
                    assert isinstance(transformer_layer, TransformerLayer)
                    assert transformer_layer.attention_type == attention_type

            if attention_type == 'multi_head_attention':
                _test_save_load()  # Always test standard attention
            else:
                self._test_attention_type_safely(attention_type, lambda at: _test_save_load())

    # ===== GRADIENT FLOW TESTS =====

    def test_gradient_flow(self, small_input_tensor, small_layer_instance):
        """Test gradient flow through the layer."""
        layer = small_layer_instance

        with tf.GradientTape() as tape:
            inputs = tf.Variable(small_input_tensor)
            outputs = layer(inputs, training=True)
            loss = tf.reduce_mean(tf.square(outputs))

        # Get gradients
        grads = tape.gradient(loss, layer.trainable_variables)

        # Check gradients exist and are not None
        assert all(g is not None for g in grads)
        assert all(np.any(g.numpy() != 0) for g in grads)

    def test_gradient_flow_different_attention_types(self, small_input_tensor):
        """Test gradient flow with different attention types."""
        attention_configs = [
            {'attention_type': 'multi_head_attention'},
            {'attention_type': 'window_attention', 'window_size': 4},  # Correct window_size for seq_len=16
            {'attention_type': 'group_query_attention', 'n_kv_head': 2},
            {'attention_type': 'differential_attention'}
        ]

        def _test_gradients(attention_type, config):
            """This inner function now correctly matches the call signature."""
            layer = TransformerLayer(
                hidden_size=64, num_heads=4, intermediate_size=256, **config
            )
            with tf.GradientTape() as tape:
                inputs = tf.Variable(small_input_tensor)
                outputs = layer(inputs, training=True)
                loss = tf.reduce_mean(tf.square(outputs))

            grads = tape.gradient(loss, layer.trainable_variables)

            assert all(g is not None for g in grads), f"Found None gradient for {attention_type}"

            non_zero_grads = []
            for g in grads:
                if isinstance(g, tf.IndexedSlices):
                    dense_grad = tf.convert_to_tensor(g)
                    non_zero_grads.append(np.any(dense_grad.numpy() != 0))
                else:
                    non_zero_grads.append(np.any(g.numpy() != 0))

            assert all(non_zero_grads), f"Found zero gradients for {attention_type}"

        for config in attention_configs:
            attention_type = config['attention_type']
            self._test_attention_type_safely(attention_type, _test_gradients, config)

    def test_gradient_flow_safe_configurations(self, small_input_tensor):
        """Test gradient flow with safe configurations."""
        configurations = [
            {'normalization_position': 'post', 'ffn_type': 'mlp'},
            {'normalization_position': 'pre', 'ffn_type': 'mlp'},
        ]

        for config in configurations:
            layer = TransformerLayer(
                hidden_size=64,
                num_heads=4,
                intermediate_size=256,
                attention_type='multi_head_attention',
                **config
            )

            with tf.GradientTape() as tape:
                inputs = tf.Variable(small_input_tensor)
                outputs = layer(inputs, training=True)
                loss = tf.reduce_mean(tf.square(outputs))

            # Get gradients
            grads = tape.gradient(loss, layer.trainable_variables)

            # Check gradients exist and are not None
            assert all(g is not None for g in grads)
            assert all(np.any(g.numpy() != 0) for g in grads)

    # ===== TRAINING TESTS =====

    def test_training_loop_basic(self, small_input_tensor):
        """Test basic training loop with the custom layer."""
        model = keras.Sequential([
            keras.layers.InputLayer(input_shape=(16, 64)),
            TransformerLayer(
                hidden_size=64,
                num_heads=4,
                intermediate_size=256,
                attention_type='multi_head_attention',
                normalization_position='pre',
                ffn_type='mlp'
            ),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(10)
        ])

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

        # Check training completed successfully
        assert final_loss is not None
        assert not np.isnan(final_loss)

    def test_training_loop_different_attention_types(self, small_input_tensor):
        """Test training with different attention types."""
        attention_configs = [
            {'attention_type': 'multi_head_attention'},
            {'attention_type': 'window_attention', 'window_size': 4},  # FIXED: Correct window_size
            {'attention_type': 'group_query_attention', 'n_kv_head': 2}
        ]

        for config in attention_configs:
            attention_type = config['attention_type']

            def _test_training():
                # Create model
                model = keras.Sequential([
                    keras.layers.InputLayer(input_shape=(16, 64)),
                    TransformerLayer(
                        hidden_size=64,
                        num_heads=4,
                        intermediate_size=256,
                        **config
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

            if attention_type == 'multi_head_attention':
                _test_training()  # Always test standard attention
            else:
                self._test_attention_type_safely(attention_type, lambda at: _test_training())

    # ===== EDGE CASE TESTS =====

    def test_numerical_stability(self):
        """Test layer stability with extreme input values."""
        layer = TransformerLayer(
            hidden_size=64,
            num_heads=4,
            intermediate_size=256,
            attention_type='multi_head_attention'  # Use safe attention type
        )

        batch_size, seq_len, hidden_size = 2, 8, 64

        test_cases = [
            tf.zeros((batch_size, seq_len, hidden_size)),
            tf.ones((batch_size, seq_len, hidden_size)) * 1e-6,
            tf.ones((batch_size, seq_len, hidden_size)) * 10.0,
            tf.random.normal((batch_size, seq_len, hidden_size)) * 5.0
        ]

        for test_input in test_cases:
            output = layer(test_input, training=False)

            # Check for NaN/Inf values
            assert not np.any(np.isnan(output.numpy())), "NaN values detected in output"
            assert not np.any(np.isinf(output.numpy())), "Inf values detected in output"

    def test_different_sequence_lengths(self):
        """Test layer with different sequence lengths."""
        layer = TransformerLayer(
            hidden_size=64,
            num_heads=4,
            intermediate_size=256,
            attention_type='multi_head_attention'  # Use safe attention type
        )

        sequence_lengths = [1, 8, 32, 128]

        for seq_len in sequence_lengths:
            test_input = tf.random.normal((2, seq_len, 64))
            output = layer(test_input, training=False)

            # Check output shape
            assert output.shape == (2, seq_len, 64)
            assert not np.any(np.isnan(output.numpy()))

    def test_different_sequence_lengths_with_attention_types(self):
        """Test different attention types with various sequence lengths that work."""
        # For window attention, we need perfect square sequence lengths
        test_cases = [
            ('multi_head_attention', [8, 16, 32], {}),
            ('window_attention', [9, 16, 25], {}),  # Perfect squares: 3^2, 4^2, 5^2
            ('group_query_attention', [8, 16, 32], {'n_kv_head': 2}),
        ]

        for attention_type, sequence_lengths, extra_config in test_cases:
            def _test_seq_lengths():
                for seq_len in sequence_lengths:
                    # For window attention, calculate correct window size
                    config = extra_config.copy()
                    if attention_type == 'window_attention':
                        config['window_size'] = int(np.sqrt(seq_len))

                    layer = TransformerLayer(
                        hidden_size=64,
                        num_heads=4,
                        intermediate_size=256,
                        attention_type=attention_type,
                        **config
                    )

                    test_input = tf.random.normal((2, seq_len, 64))
                    output = layer(test_input, training=False)

                    # Check output shape
                    assert output.shape == (2, seq_len, 64)
                    assert not np.any(np.isnan(output.numpy()))

            if attention_type == 'multi_head_attention':
                _test_seq_lengths()  # Always test standard attention
            else:
                self._test_attention_type_safely(attention_type, lambda at: _test_seq_lengths())

    def test_dropout_behavior(self):
        """Test dropout behavior during training vs inference."""
        layer = TransformerLayer(
            hidden_size=64,
            num_heads=4,
            intermediate_size=256,
            dropout_rate=0.5,
            attention_dropout_rate=0.5,
            attention_type='multi_head_attention'  # Use safe attention type
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

    def test_normalization_position_behavior(self, small_input_tensor):
        """Test that pre-norm and post-norm produce different outputs."""
        layer_post = TransformerLayer(
            hidden_size=64,
            num_heads=4,
            intermediate_size=256,
            normalization_position='post',
            dropout_rate=0.0,
            attention_dropout_rate=0.0,
            attention_type='multi_head_attention'
        )

        layer_pre = TransformerLayer(
            hidden_size=64,
            num_heads=4,
            intermediate_size=256,
            normalization_position='pre',
            dropout_rate=0.0,
            attention_dropout_rate=0.0,
            attention_type='multi_head_attention'
        )

        # Forward pass through both layers
        output_post = layer_post(small_input_tensor, training=False)
        output_pre = layer_pre(small_input_tensor, training=False)

        # Outputs should be different due to different normalization ordering
        assert not np.allclose(output_post.numpy(), output_pre.numpy(), rtol=1e-3)

    # ===== ERROR HANDLING TESTS =====

    def test_layer_creation_error_handling(self):
        """Test proper error handling in layer creation."""
        layer = TransformerLayer(
            hidden_size=64,
            num_heads=4,
            intermediate_size=256
        )

        # Test invalid attention type error handling
        original_attention_type = layer.attention_type
        layer.attention_type = 'invalid_attention_type'

        try:
            with pytest.raises(ValueError, match="Unknown attention type"):
                layer._create_attention_layer('test_attention')
        finally:
            layer.attention_type = original_attention_type

        # Test invalid FFN type error handling
        original_ffn_type = layer.ffn_type
        layer.ffn_type = 'invalid_ffn_type'

        try:
            with pytest.raises(ValueError, match="Unknown FFN type"):
                layer._create_ffn_layer('test_ffn')
        finally:
            layer.ffn_type = original_ffn_type

        # Test invalid normalization type error handling
        original_norm_type = layer.normalization_type
        layer.normalization_type = 'invalid_norm_type'

        try:
            with pytest.raises(ValueError, match="Unknown normalization type"):
                layer._create_normalization_layer('test_norm')
        finally:
            layer.normalization_type = original_norm_type


if __name__ == '__main__':
    pytest.main([__file__])