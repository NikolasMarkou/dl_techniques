"""Test suite for WaveFieldAttention layer.

This module contains comprehensive tests for the WaveFieldAttention layer,
a physics-inspired attention mechanism that uses damped wave field convolution
via FFT instead of standard dot-product attention. Tests cover initialization,
build process, output shapes, forward pass, serialization, gradient flow,
and edge cases.
"""

import pytest
import numpy as np
import tensorflow as tf
import keras
import tempfile
import os

# Import the layer to test
from dl_techniques.layers.attention.wave_field_attention import WaveFieldAttention


class TestWaveFieldAttention:
    """Test suite for WaveFieldAttention layer."""

    @pytest.fixture
    def input_tensor(self):
        """Create a test input tensor."""
        return tf.random.normal([2, 10, 64])  # (batch_size, seq_len, dim)

    @pytest.fixture
    def layer_instance(self):
        """Create a default layer instance for testing."""
        return WaveFieldAttention(dim=64, num_heads=8)

    @pytest.fixture
    def different_configs(self):
        """Provide different layer configurations for testing."""
        return [
            {"dim": 32, "num_heads": 4},
            {"dim": 128, "num_heads": 8, "dropout_rate": 0.1},
            {"dim": 256, "num_heads": 16, "use_bias": True, "field_size": 256},
            {"dim": 512, "num_heads": 8, "dropout_rate": 0.2, "max_seq_len": 256},
        ]

    # ==================== Initialization Tests ====================

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        layer = WaveFieldAttention(dim=64, num_heads=8)

        assert layer.dim == 64
        assert layer.num_heads == 8
        assert layer.head_dim == 8
        assert layer.field_size == 512
        assert layer.max_seq_len == 128
        assert layer.dropout_rate == 0.0
        assert layer.use_bias is True
        assert layer.gate_bias_init == 2.0
        assert layer.coupling_noise_stddev == 0.01
        assert layer.kernel_regularizer is None
        assert layer.bias_regularizer is None

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        custom_regularizer = keras.regularizers.L2(1e-4)

        layer = WaveFieldAttention(
            dim=128,
            num_heads=16,
            field_size=1024,
            max_seq_len=256,
            dropout_rate=0.1,
            use_bias=False,
            gate_bias_init=3.0,
            coupling_noise_stddev=0.05,
            kernel_initializer="he_normal",
            kernel_regularizer=custom_regularizer,
        )

        assert layer.dim == 128
        assert layer.num_heads == 16
        assert layer.head_dim == 8
        assert layer.field_size == 1024
        assert layer.max_seq_len == 256
        assert layer.dropout_rate == 0.1
        assert layer.use_bias is False
        assert layer.gate_bias_init == 3.0
        assert layer.coupling_noise_stddev == 0.05
        assert isinstance(layer.kernel_initializer, keras.initializers.HeNormal)
        assert layer.kernel_regularizer == custom_regularizer

    def test_invalid_dim_not_divisible(self):
        """Test that invalid dim raises ValueError."""
        with pytest.raises(ValueError, match="dim \\(63\\) must be divisible by num_heads \\(8\\)"):
            WaveFieldAttention(dim=63, num_heads=8)

    def test_invalid_dim_negative(self):
        """Test that negative dim raises ValueError."""
        with pytest.raises(ValueError, match="dim must be positive, got -64"):
            WaveFieldAttention(dim=-64, num_heads=8)

    def test_invalid_num_heads_negative(self):
        """Test that negative num_heads raises ValueError."""
        with pytest.raises(ValueError, match="num_heads must be positive, got -8"):
            WaveFieldAttention(dim=64, num_heads=-8)

    def test_invalid_dropout_rate(self):
        """Test that invalid dropout_rate raises ValueError."""
        with pytest.raises(ValueError, match="dropout_rate must be in"):
            WaveFieldAttention(dim=64, num_heads=8, dropout_rate=1.5)

    def test_invalid_field_size(self):
        """Test that field_size <= 1 raises ValueError."""
        with pytest.raises(ValueError, match="field_size must be > 1"):
            WaveFieldAttention(dim=64, num_heads=8, field_size=1)

    def test_invalid_max_seq_len(self):
        """Test that non-positive max_seq_len raises ValueError."""
        with pytest.raises(ValueError, match="max_seq_len must be positive"):
            WaveFieldAttention(dim=64, num_heads=8, max_seq_len=0)

    # ==================== Build Process Tests ====================

    def test_build_process(self, input_tensor):
        """Test that the layer builds properly."""
        layer = WaveFieldAttention(dim=64, num_heads=8)
        layer(input_tensor)

        assert layer.built is True

    def test_build_input_shape_validation_2d(self):
        """Test input shape validation rejects 2D input."""
        layer = WaveFieldAttention(dim=64, num_heads=8)

        with pytest.raises(ValueError, match="Expected 3-D input"):
            layer.build((32, 64))

    def test_build_input_shape_validation_dim_mismatch(self):
        """Test input shape validation rejects dimension mismatch."""
        layer = WaveFieldAttention(dim=64, num_heads=8)

        with pytest.raises(ValueError, match="Last dimension .* must match dim"):
            layer.build((None, 10, 32))

    def test_explicit_build(self, input_tensor):
        """Test that build method works when called explicitly."""
        layer = WaveFieldAttention(dim=64, num_heads=8)
        layer.build(input_tensor.shape)

        assert layer.built is True

    def test_wave_parameters_created(self, input_tensor):
        """Test that wave parameters are created during build."""
        layer = WaveFieldAttention(dim=64, num_heads=8)
        layer.build(input_tensor.shape)

        # Check wave parameters exist and have correct shapes
        assert layer.wave_frequency.shape == (8,)
        assert layer.wave_damping.shape == (8,)
        assert layer.wave_phase.shape == (8,)
        assert layer.field_coupling.shape == (8, 8)

    def test_wave_parameter_initial_values(self):
        """Test wave parameter initialisation matches expected linspace values."""
        num_heads = 4
        layer = WaveFieldAttention(dim=32, num_heads=num_heads)
        layer.build((None, 10, 32))

        freq = keras.ops.convert_to_numpy(layer.wave_frequency)
        damp = keras.ops.convert_to_numpy(layer.wave_damping)
        phase = keras.ops.convert_to_numpy(layer.wave_phase)
        coupling = keras.ops.convert_to_numpy(layer.field_coupling)

        expected_freq = np.linspace(0.3, 4.0, num_heads)
        expected_damp = np.linspace(-3.0, 0.5, num_heads)
        expected_phase = np.linspace(0, np.pi, num_heads)
        expected_coupling = np.eye(num_heads)

        np.testing.assert_allclose(freq, expected_freq, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(damp, expected_damp, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(phase, expected_phase, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(coupling, expected_coupling, rtol=1e-5, atol=1e-5)

    # ==================== Output Shape Tests ====================

    def test_output_shapes(self, different_configs):
        """Test that output shapes are computed correctly."""
        for config in different_configs:
            layer = WaveFieldAttention(**config)

            for seq_len in [5, 20, 50]:
                input_shape = (2, seq_len, config["dim"])
                input_tensor = tf.random.normal(input_shape)

                output = layer(input_tensor)
                assert output.shape == input_shape

                computed_shape = layer.compute_output_shape(input_shape)
                assert computed_shape == input_shape

    def test_batch_size_flexibility(self):
        """Test that the layer works with different batch sizes."""
        layer = WaveFieldAttention(dim=64, num_heads=8)

        for batch_size in [1, 4, 16]:
            input_tensor = tf.random.normal([batch_size, 10, 64])
            output = layer(input_tensor)
            assert output.shape == (batch_size, 10, 64)

    # ==================== Forward Pass Tests ====================

    def test_forward_pass_basic(self, input_tensor):
        """Test basic forward pass functionality."""
        layer = WaveFieldAttention(dim=64, num_heads=8)
        output = layer(input_tensor)

        assert not tf.reduce_any(tf.math.is_nan(output))
        assert not tf.reduce_any(tf.math.is_inf(output))
        assert output.shape == input_tensor.shape

    def test_forward_pass_deterministic(self):
        """Test forward pass with deterministic inputs."""
        layer = WaveFieldAttention(
            dim=64,
            num_heads=8,
            dropout_rate=0.0,
        )

        input_tensor = tf.ones([1, 5, 64])

        output1 = layer(input_tensor, training=False)
        output2 = layer(input_tensor, training=False)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output1),
            keras.ops.convert_to_numpy(output2),
            rtol=1e-6, atol=1e-6,
            err_msg="Deterministic forward passes should match"
        )

    def test_training_mode_differences(self, input_tensor):
        """Test that training mode affects dropout behavior."""
        layer = WaveFieldAttention(dim=64, num_heads=8, dropout_rate=0.5)

        layer(input_tensor, training=False)

        # Collect several training-mode outputs; dropout should make at least one differ
        outputs_train = []
        for _ in range(5):
            outputs_train.append(layer(input_tensor, training=True))

        output_inference = layer(input_tensor, training=False)

        any_different = any(
            not tf.reduce_all(tf.equal(ot, output_inference))
            for ot in outputs_train
        )
        assert any_different, "Training mode with dropout should produce different outputs"

    # ==================== Field Position Tests ====================

    def test_field_positions_absolute(self):
        """Test that field positions are absolute and independent of seq length."""
        layer = WaveFieldAttention(dim=32, num_heads=4, field_size=512, max_seq_len=128)
        layer.build((None, 10, 32))

        # Positions for token 0-4 should be the same regardless of total seq length
        pos_short = layer._compute_field_positions(5)
        pos_long = layer._compute_field_positions(10)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(pos_short),
            keras.ops.convert_to_numpy(pos_long[:5]),
            rtol=1e-6, atol=1e-6,
            err_msg="Absolute positions for first N tokens should be identical"
        )

    def test_field_positions_clamped(self):
        """Test that field positions are clamped within valid range."""
        layer = WaveFieldAttention(dim=32, num_heads=4, field_size=64, max_seq_len=32)
        layer.build((None, 10, 32))

        pos = keras.ops.convert_to_numpy(layer._compute_field_positions(100))

        assert np.all(pos >= 0.0)
        assert np.all(pos <= 62.0)  # field_size - 2

    def test_field_stride_computation(self):
        """Test that field stride is computed correctly."""
        layer = WaveFieldAttention(dim=32, num_heads=4, field_size=512, max_seq_len=128)

        expected_stride = (512 - 1) / (128 - 1)
        np.testing.assert_allclose(layer._field_stride, expected_stride, rtol=1e-6)

    # ==================== Scatter / Gather Tests ====================

    def test_scatter_gather_matrices_shapes(self):
        """Test scatter/gather matrix shapes."""
        layer = WaveFieldAttention(dim=32, num_heads=4, field_size=64, max_seq_len=32)
        layer.build((None, 10, 32))

        field_pos = layer._compute_field_positions(10)
        scatter_mat, gather_mat = layer._build_scatter_gather_matrices(field_pos)

        assert scatter_mat.shape == (64, 10)
        assert gather_mat.shape == (10, 64)

    def test_scatter_gather_transpose_relation(self):
        """Test that gather_mat is the transpose of scatter_mat."""
        layer = WaveFieldAttention(dim=32, num_heads=4, field_size=64, max_seq_len=32)
        layer.build((None, 10, 32))

        field_pos = layer._compute_field_positions(7)
        scatter_mat, gather_mat = layer._build_scatter_gather_matrices(field_pos)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(scatter_mat),
            keras.ops.convert_to_numpy(keras.ops.transpose(gather_mat)),
            rtol=1e-6, atol=1e-6,
            err_msg="gather_mat should be transpose of scatter_mat"
        )

    def test_scatter_matrix_row_sums(self):
        """Test that each column of scatter_mat sums to 1 (bilinear weights)."""
        layer = WaveFieldAttention(dim=32, num_heads=4, field_size=64, max_seq_len=32)
        layer.build((None, 10, 32))

        field_pos = layer._compute_field_positions(10)
        scatter_mat, _ = layer._build_scatter_gather_matrices(field_pos)

        col_sums = keras.ops.convert_to_numpy(
            keras.ops.sum(scatter_mat, axis=0)
        )
        np.testing.assert_allclose(
            col_sums, np.ones(10), rtol=1e-6, atol=1e-6,
            err_msg="Each column of scatter_mat should sum to 1"
        )

    # ==================== Wave Kernel Tests ====================

    def test_wave_kernel_fft_shape(self):
        """Test that wave kernel FFT output has correct shape."""
        field_size = 64
        num_heads = 4
        layer = WaveFieldAttention(dim=32, num_heads=num_heads, field_size=field_size)
        layer.build((None, 10, 32))

        kernel_fft = layer._build_wave_kernels_fft()

        # keras.ops.rfft returns (real, imag) tuple
        assert isinstance(kernel_fft, tuple)
        assert len(kernel_fft) == 2
        # rfft of length 2*G yields G+1 frequency bins per component
        assert kernel_fft[0].shape == (num_heads, field_size + 1)
        assert kernel_fft[1].shape == (num_heads, field_size + 1)

    def test_wave_convolve_output_shape(self):
        """Test that wave convolution preserves field shape."""
        field_size = 64
        num_heads = 4
        head_dim = 8
        layer = WaveFieldAttention(dim=32, num_heads=num_heads, field_size=field_size)
        layer.build((None, 10, 32))

        field = tf.random.normal([2, num_heads, field_size, head_dim])
        kernel_fft = layer._build_wave_kernels_fft()
        convolved = layer._wave_convolve(field, kernel_fft)

        assert convolved.shape == (2, num_heads, field_size, head_dim)

    def test_wave_convolve_no_nan(self):
        """Test that wave convolution does not produce NaN/Inf."""
        layer = WaveFieldAttention(dim=64, num_heads=8, field_size=128)
        layer.build((None, 10, 64))

        field = tf.random.normal([2, 8, 128, 8])
        kernel_fft = layer._build_wave_kernels_fft()
        convolved = layer._wave_convolve(field, kernel_fft)

        assert not tf.reduce_any(tf.math.is_nan(convolved))
        assert not tf.reduce_any(tf.math.is_inf(convolved))

    # ==================== Field Coupling Tests ====================

    def test_field_coupling_output_shape(self):
        """Test that field coupling preserves tensor shape."""
        layer = WaveFieldAttention(dim=64, num_heads=8, field_size=128)
        layer.build((None, 10, 64))

        field = tf.random.normal([2, 8, 128, 8])
        coupled = layer._apply_field_coupling(field)

        assert coupled.shape == field.shape

    def test_field_coupling_identity_init(self):
        """Test that with identity coupling, output approximately equals input."""
        layer = WaveFieldAttention(dim=32, num_heads=4, field_size=64)
        layer.build((None, 10, 32))

        field = tf.random.normal([2, 4, 64, 8])

        # Coupling is initialised to identity; softmax(identity) is not
        # exactly identity but should be close to it (diagonal-dominant).
        coupled = layer._apply_field_coupling(field)

        # Not exactly equal due to softmax, but correlated
        field_np = keras.ops.convert_to_numpy(field)
        coupled_np = keras.ops.convert_to_numpy(coupled)

        # Each head's output should correlate strongly with itself
        for h in range(4):
            corr = np.corrcoef(field_np[0, h].flatten(), coupled_np[0, h].flatten())[0, 1]
            assert corr > 0.5, f"Head {h} coupling correlation too low: {corr}"

    # ==================== Serialization Tests ====================

    def test_serialization_config_completeness(self):
        """Test that get_config captures all necessary parameters."""
        layer = WaveFieldAttention(
            dim=256,
            num_heads=16,
            field_size=1024,
            max_seq_len=256,
            dropout_rate=0.2,
            use_bias=False,
            gate_bias_init=3.0,
            coupling_noise_stddev=0.05,
            kernel_initializer="he_normal",
            kernel_regularizer=keras.regularizers.L2(1e-4),
            bias_regularizer=keras.regularizers.L1(1e-5),
        )

        config = layer.get_config()

        required_keys = [
            "dim", "num_heads", "field_size", "max_seq_len",
            "dropout_rate", "use_bias", "gate_bias_init",
            "coupling_noise_stddev", "kernel_initializer",
            "bias_initializer", "kernel_regularizer", "bias_regularizer",
        ]
        for key in required_keys:
            assert key in config, f"Missing key {key} in config"

        assert config["dim"] == 256
        assert config["num_heads"] == 16
        assert config["field_size"] == 1024
        assert config["max_seq_len"] == 256
        assert config["dropout_rate"] == 0.2
        assert config["use_bias"] is False
        assert config["gate_bias_init"] == 3.0
        assert config["coupling_noise_stddev"] == 0.05

    def test_layer_recreation_from_config(self):
        """Test recreating layer from config."""
        original_layer = WaveFieldAttention(
            dim=128,
            num_heads=8,
            field_size=256,
            max_seq_len=64,
            dropout_rate=0.1,
            use_bias=False,
            gate_bias_init=1.5,
        )

        config = original_layer.get_config()
        recreated_layer = WaveFieldAttention.from_config(config)

        assert recreated_layer.dim == original_layer.dim
        assert recreated_layer.num_heads == original_layer.num_heads
        assert recreated_layer.field_size == original_layer.field_size
        assert recreated_layer.max_seq_len == original_layer.max_seq_len
        assert recreated_layer.dropout_rate == original_layer.dropout_rate
        assert recreated_layer.use_bias == original_layer.use_bias
        assert recreated_layer.gate_bias_init == original_layer.gate_bias_init

    def test_serialization_with_build(self, input_tensor):
        """Test serialization after building the layer."""
        original_layer = WaveFieldAttention(
            dim=64,
            num_heads=8,
            dropout_rate=0.1,
            use_bias=True,
        )
        original_layer(input_tensor)

        config = original_layer.get_config()
        recreated_layer = WaveFieldAttention.from_config(config)
        recreated_layer(input_tensor)

        assert len(recreated_layer.weights) == len(original_layer.weights)

    # ==================== Model Integration Tests ====================

    def test_model_integration(self, input_tensor):
        """Test the layer in a model context."""
        inputs = keras.Input(shape=input_tensor.shape[1:])
        x = WaveFieldAttention(dim=64, num_heads=8)(inputs)
        x = keras.layers.LayerNormalization()(x)
        x = keras.layers.GlobalAveragePooling1D()(x)
        outputs = keras.layers.Dense(10)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

        y_pred = model(input_tensor, training=False)
        assert y_pred.shape == (input_tensor.shape[0], 10)

    # ==================== Model Save/Load Tests ====================

    def test_model_save_load(self, input_tensor):
        """Test saving and loading a model with the custom layer."""
        inputs = keras.Input(shape=input_tensor.shape[1:])
        x = WaveFieldAttention(
            dim=64, num_heads=8, name="wave_field_attn"
        )(inputs)
        x = keras.layers.GlobalAveragePooling1D()(x)
        outputs = keras.layers.Dense(10)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        original_prediction = model.predict(input_tensor, verbose=0)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "model.keras")
            model.save(model_path)

            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={"WaveFieldAttention": WaveFieldAttention},
            )

            loaded_prediction = loaded_model.predict(input_tensor, verbose=0)

            np.testing.assert_allclose(
                original_prediction, loaded_prediction,
                rtol=1e-5, atol=1e-5,
                err_msg="Predictions should match after save/load"
            )

            assert isinstance(
                loaded_model.get_layer("wave_field_attn"),
                WaveFieldAttention,
            )

    # ==================== Gradient Flow Tests ====================

    def test_gradient_flow(self, input_tensor):
        """Test gradient flow through the layer."""
        layer = WaveFieldAttention(dim=64, num_heads=8)

        with tf.GradientTape() as tape:
            inputs = tf.Variable(input_tensor)
            outputs = layer(inputs)
            loss = tf.reduce_mean(tf.square(outputs))

        grads = tape.gradient(loss, layer.trainable_variables)

        assert all(g is not None for g in grads), "All gradients should be non-None"
        assert all(
            tf.reduce_any(g != 0) for g in grads
        ), "Gradients should have non-zero values"

    def test_gradient_flow_wave_params(self, input_tensor):
        """Test that gradients flow to wave kernel parameters."""
        layer = WaveFieldAttention(dim=64, num_heads=8)

        with tf.GradientTape() as tape:
            outputs = layer(input_tensor)
            loss = tf.reduce_mean(tf.square(outputs))

        wave_vars = [
            layer.wave_frequency,
            layer.wave_damping,
            layer.wave_phase,
            layer.field_coupling,
        ]
        grads = tape.gradient(loss, wave_vars)

        for var, grad in zip(wave_vars, grads):
            assert grad is not None, f"Gradient for {var.name} should not be None"
            assert tf.reduce_any(grad != 0), (
                f"Gradient for {var.name} should have non-zero values"
            )

    # ==================== Edge Case Tests ====================

    def test_numerical_stability(self):
        """Test layer stability with extreme input values."""
        layer = WaveFieldAttention(dim=64, num_heads=8)

        test_cases = [
            ("zeros", tf.zeros((2, 10, 64))),
            ("tiny", tf.ones((2, 10, 64)) * 1e-10),
            ("large", tf.ones((2, 10, 64)) * 1e3),
            ("random_large", tf.random.normal((2, 10, 64)) * 10),
        ]

        for name, test_input in test_cases:
            output = layer(test_input)
            assert not tf.reduce_any(tf.math.is_nan(output)), (
                f"NaN detected for case '{name}'"
            )
            assert not tf.reduce_any(tf.math.is_inf(output)), (
                f"Inf detected for case '{name}'"
            )

    def test_single_sequence_length(self):
        """Test with sequence length of 1."""
        layer = WaveFieldAttention(dim=64, num_heads=8)
        input_tensor = tf.random.normal([2, 1, 64])
        output = layer(input_tensor)

        assert output.shape == (2, 1, 64)
        assert not tf.reduce_any(tf.math.is_nan(output))

    def test_large_sequence_length(self):
        """Test with large sequence length."""
        layer = WaveFieldAttention(dim=64, num_heads=8, max_seq_len=512)
        input_tensor = tf.random.normal([1, 500, 64])
        output = layer(input_tensor)

        assert output.shape == (1, 500, 64)
        assert not tf.reduce_any(tf.math.is_nan(output))

    def test_sequence_exceeds_max_seq_len(self):
        """Test that sequences longer than max_seq_len still work (positions clamp)."""
        layer = WaveFieldAttention(dim=32, num_heads=4, field_size=64, max_seq_len=10)
        input_tensor = tf.random.normal([1, 20, 32])
        output = layer(input_tensor)

        assert output.shape == (1, 20, 32)
        assert not tf.reduce_any(tf.math.is_nan(output))

    def test_single_head(self):
        """Test with a single attention head."""
        layer = WaveFieldAttention(dim=64, num_heads=1)
        input_tensor = tf.random.normal([2, 10, 64])
        output = layer(input_tensor)

        assert output.shape == (2, 10, 64)
        assert not tf.reduce_any(tf.math.is_nan(output))

    # ==================== Configuration Variation Tests ====================

    def test_different_head_counts(self):
        """Test layer with different numbers of heads."""
        dim = 64
        input_tensor = tf.random.normal([2, 10, dim])

        for num_heads in [1, 2, 4, 8, 16]:
            if dim % num_heads == 0:
                layer = WaveFieldAttention(dim=dim, num_heads=num_heads)
                output = layer(input_tensor)

                assert output.shape == input_tensor.shape
                assert not tf.reduce_any(tf.math.is_nan(output))

    def test_different_field_sizes(self):
        """Test layer with different field discretisation sizes."""
        input_tensor = tf.random.normal([2, 10, 32])

        for field_size in [32, 64, 128, 512]:
            layer = WaveFieldAttention(dim=32, num_heads=4, field_size=field_size)
            output = layer(input_tensor)

            assert output.shape == input_tensor.shape
            assert not tf.reduce_any(tf.math.is_nan(output))

    def test_gate_bias_effect(self):
        """Test that gate_bias_init controls initial gate openness."""
        input_tensor = tf.ones([1, 5, 32])

        # Large positive bias -> gate near 1 -> output closer to gathered field
        layer_open = WaveFieldAttention(dim=32, num_heads=4, gate_bias_init=10.0)
        out_open = layer_open(input_tensor)

        # Large negative bias -> gate near 0 -> output near zero
        layer_closed = WaveFieldAttention(dim=32, num_heads=4, gate_bias_init=-10.0)
        out_closed = layer_closed(input_tensor)

        open_norm = float(tf.reduce_mean(tf.abs(out_open)))
        closed_norm = float(tf.reduce_mean(tf.abs(out_closed)))

        assert open_norm > closed_norm, (
            f"Open gate output norm ({open_norm}) should exceed "
            f"closed gate norm ({closed_norm})"
        )

    def test_dropout_creates_layer_when_nonzero(self):
        """Test that dropout layer is created only when dropout_rate > 0."""
        layer_no_drop = WaveFieldAttention(dim=32, num_heads=4, dropout_rate=0.0)
        assert layer_no_drop.dropout_layer is None

        layer_with_drop = WaveFieldAttention(dim=32, num_heads=4, dropout_rate=0.1)
        assert layer_with_drop.dropout_layer is not None
        assert isinstance(layer_with_drop.dropout_layer, keras.layers.Dropout)

    # ==================== Sublayer Structure Tests ====================

    def test_sublayer_existence(self, input_tensor):
        """Test that all expected sublayers exist after build."""
        layer = WaveFieldAttention(dim=64, num_heads=8)
        layer(input_tensor)

        assert hasattr(layer, "qkv_proj")
        assert hasattr(layer, "output_proj")
        assert hasattr(layer, "gate_proj")
        assert isinstance(layer.qkv_proj, keras.layers.Dense)
        assert isinstance(layer.output_proj, keras.layers.Dense)
        assert isinstance(layer.gate_proj, keras.layers.Dense)

    def test_qkv_projection_output_units(self, input_tensor):
        """Test that QKV projection has 3x dim units."""
        layer = WaveFieldAttention(dim=64, num_heads=8)
        layer(input_tensor)

        assert layer.qkv_proj.units == 3 * 64

    def test_trainable_variable_count(self, input_tensor):
        """Test that the expected number of trainable variables exist."""
        layer = WaveFieldAttention(dim=64, num_heads=8, dropout_rate=0.0)
        layer(input_tensor)

        # Expected trainable vars:
        # qkv_proj: kernel + bias = 2
        # output_proj: kernel + bias = 2
        # gate_proj: kernel + bias = 2
        # wave_frequency, wave_damping, wave_phase = 3
        # field_coupling = 1
        # Total = 10
        assert len(layer.trainable_variables) == 10

    def test_trainable_variable_count_no_bias(self, input_tensor):
        """Test trainable variable count when use_bias=False for projections."""
        layer = WaveFieldAttention(dim=64, num_heads=8, use_bias=False, dropout_rate=0.0)
        layer(input_tensor)

        # qkv_proj: kernel = 1
        # output_proj: kernel = 1
        # gate_proj: always has bias (hard-coded use_bias=True) = 2
        # wave params = 3
        # coupling = 1
        # Total = 8
        assert len(layer.trainable_variables) == 8

    # ==================== Performance / Sanity Tests ====================

    def test_different_seq_lengths_same_weights(self):
        """Test that same weights produce consistent behavior across seq lengths."""
        layer = WaveFieldAttention(dim=32, num_heads=4, dropout_rate=0.0)

        # Build with one length, then run with another
        layer(tf.random.normal([1, 10, 32]))

        out_short = layer(tf.ones([1, 5, 32]), training=False)
        out_long = layer(tf.ones([1, 20, 32]), training=False)

        # First 5 positions should NOT be identical because gather from different
        # field states, but both should be finite
        assert not tf.reduce_any(tf.math.is_nan(out_short))
        assert not tf.reduce_any(tf.math.is_nan(out_long))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])