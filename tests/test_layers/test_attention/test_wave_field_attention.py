"""Test suite for WaveFieldAttention layer (V3.6).

Covers initialization, build, output shapes, forward pass, attention_mask,
query modulation, serialization, gradient flow, and edge cases.
"""

import pytest
import numpy as np
import tensorflow as tf
import keras
import tempfile
import os

from dl_techniques.layers.attention.wave_field_attention import WaveFieldAttention


class TestWaveFieldAttention:
    """Test suite for WaveFieldAttention layer."""

    @pytest.fixture
    def input_tensor(self):
        return tf.random.normal([2, 10, 64])

    @pytest.fixture
    def layer_instance(self):
        return WaveFieldAttention(dim=64, num_heads=8)

    @pytest.fixture
    def different_configs(self):
        return [
            {"dim": 32, "num_heads": 4},
            {"dim": 128, "num_heads": 8, "dropout_rate": 0.1},
            {"dim": 256, "num_heads": 16, "use_bias": True, "field_size": 256},
            {"dim": 512, "num_heads": 8, "dropout_rate": 0.2, "max_seq_len": 256},
        ]

    # ==================== Initialization Tests ====================

    def test_initialization_defaults(self):
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
        with pytest.raises(ValueError, match="dim \\(63\\) must be divisible by num_heads \\(8\\)"):
            WaveFieldAttention(dim=63, num_heads=8)

    def test_invalid_dim_negative(self):
        with pytest.raises(ValueError, match="dim must be positive, got -64"):
            WaveFieldAttention(dim=-64, num_heads=8)

    def test_invalid_num_heads_negative(self):
        with pytest.raises(ValueError, match="num_heads must be positive, got -8"):
            WaveFieldAttention(dim=64, num_heads=-8)

    def test_invalid_dropout_rate(self):
        with pytest.raises(ValueError, match="dropout_rate must be in"):
            WaveFieldAttention(dim=64, num_heads=8, dropout_rate=1.5)

    def test_invalid_field_size(self):
        with pytest.raises(ValueError, match="field_size must be > 1"):
            WaveFieldAttention(dim=64, num_heads=8, field_size=1)

    def test_invalid_max_seq_len(self):
        with pytest.raises(ValueError, match="max_seq_len must be positive"):
            WaveFieldAttention(dim=64, num_heads=8, max_seq_len=0)

    # ==================== Build Process Tests ====================

    def test_build_process(self, input_tensor):
        layer = WaveFieldAttention(dim=64, num_heads=8)
        layer(input_tensor)
        assert layer.built is True

    def test_build_input_shape_validation_2d(self):
        layer = WaveFieldAttention(dim=64, num_heads=8)
        with pytest.raises(ValueError, match="Expected 3-D input"):
            layer.build((32, 64))

    def test_build_input_shape_validation_dim_mismatch(self):
        layer = WaveFieldAttention(dim=64, num_heads=8)
        with pytest.raises(ValueError, match="Last dim .* must match dim"):
            layer.build((None, 10, 32))

    def test_explicit_build(self, input_tensor):
        layer = WaveFieldAttention(dim=64, num_heads=8)
        layer.build(input_tensor.shape)
        assert layer.built is True

    def test_wave_parameters_created(self, input_tensor):
        layer = WaveFieldAttention(dim=64, num_heads=8)
        layer.build(input_tensor.shape)

        assert layer.wave_frequency.shape == (8,)
        assert layer.wave_damping.shape == (8,)
        assert layer.wave_phase.shape == (8,)
        assert layer.field_coupling.shape == (8, 8)

    def test_wave_parameter_initial_values(self):
        num_heads = 4
        layer = WaveFieldAttention(dim=32, num_heads=num_heads)
        layer.build((None, 10, 32))

        freq = keras.ops.convert_to_numpy(layer.wave_frequency)
        damp = keras.ops.convert_to_numpy(layer.wave_damping)
        phase = keras.ops.convert_to_numpy(layer.wave_phase)

        expected_freq = np.linspace(0.3, 4.0, num_heads)
        expected_damp = np.linspace(-3.0, 0.5, num_heads)
        expected_phase = np.linspace(0, np.pi, num_heads)

        np.testing.assert_allclose(freq, expected_freq, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(damp, expected_damp, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(phase, expected_phase, rtol=1e-5, atol=1e-5)

    def test_coupling_noise_applied(self):
        """Coupling init should be identity + noise, not exact identity."""
        num_heads = 4
        layer = WaveFieldAttention(
            dim=32, num_heads=num_heads, coupling_noise_stddev=0.05,
        )
        layer.build((None, 10, 32))

        coupling = keras.ops.convert_to_numpy(layer.field_coupling)
        eye = np.eye(num_heads)

        # Should be close to identity but not exact
        diff = np.abs(coupling - eye)
        assert np.max(diff) > 1e-6, "Coupling noise should perturb away from exact identity"
        np.testing.assert_allclose(
            coupling, eye, atol=0.3,
            err_msg="Coupling should remain close to identity with small noise",
        )

    def test_coupling_noise_zero_gives_identity(self):
        """With stddev=0, coupling should be exact identity."""
        num_heads = 4
        layer = WaveFieldAttention(
            dim=32, num_heads=num_heads, coupling_noise_stddev=0.0,
        )
        layer.build((None, 10, 32))

        coupling = keras.ops.convert_to_numpy(layer.field_coupling)
        np.testing.assert_allclose(coupling, np.eye(num_heads), rtol=1e-6, atol=1e-6)

    # ==================== Output Shape Tests ====================

    def test_output_shapes(self, different_configs):
        for config in different_configs:
            layer = WaveFieldAttention(**config)
            for seq_len in [5, 20, 50]:
                input_shape = (2, seq_len, config["dim"])
                input_tensor = tf.random.normal(input_shape)
                output = layer(input_tensor)
                assert output.shape == input_shape
                assert layer.compute_output_shape(input_shape) == input_shape

    def test_batch_size_flexibility(self):
        layer = WaveFieldAttention(dim=64, num_heads=8)
        for batch_size in [1, 4, 16]:
            input_tensor = tf.random.normal([batch_size, 10, 64])
            output = layer(input_tensor)
            assert output.shape == (batch_size, 10, 64)

    # ==================== Forward Pass Tests ====================

    def test_forward_pass_basic(self, input_tensor):
        layer = WaveFieldAttention(dim=64, num_heads=8)
        output = layer(input_tensor)

        assert not tf.reduce_any(tf.math.is_nan(output))
        assert not tf.reduce_any(tf.math.is_inf(output))
        assert output.shape == input_tensor.shape

    def test_forward_pass_deterministic(self):
        layer = WaveFieldAttention(dim=64, num_heads=8, dropout_rate=0.0)
        input_tensor = tf.ones([1, 5, 64])

        output1 = layer(input_tensor, training=False)
        output2 = layer(input_tensor, training=False)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output1),
            keras.ops.convert_to_numpy(output2),
            rtol=1e-6, atol=1e-6,
        )

    def test_training_mode_differences(self, input_tensor):
        layer = WaveFieldAttention(dim=64, num_heads=8, dropout_rate=0.5)
        layer(input_tensor, training=False)

        outputs_train = [layer(input_tensor, training=True) for _ in range(5)]
        output_inference = layer(input_tensor, training=False)

        any_different = any(
            not tf.reduce_all(tf.equal(ot, output_inference))
            for ot in outputs_train
        )
        assert any_different, "Training mode with dropout should produce different outputs"

    # ==================== Attention Mask Tests ====================

    def test_attention_mask_shape_accepted(self):
        """Layer accepts (B, N) float attention_mask."""
        layer = WaveFieldAttention(dim=32, num_heads=4)
        x = tf.random.normal([2, 10, 32])
        mask = tf.ones([2, 10])
        output = layer(x, attention_mask=mask)
        assert output.shape == (2, 10, 32)

    def test_attention_mask_zeros_padded_output(self):
        """Padded positions (mask=0) should produce zero output."""
        layer = WaveFieldAttention(dim=32, num_heads=4, dropout_rate=0.0)
        x = tf.random.normal([1, 8, 32])

        mask = tf.constant([[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
        output = layer(x, attention_mask=mask, training=False)
        output_np = keras.ops.convert_to_numpy(output)

        # Padded positions (indices 4-7) must be exactly zero
        np.testing.assert_array_equal(
            output_np[0, 4:, :], 0.0,
            err_msg="Padded positions should be zeroed by attention_mask",
        )

    def test_attention_mask_valid_positions_nonzero(self):
        """Valid positions (mask=1) with nonzero input should produce nonzero output."""
        layer = WaveFieldAttention(dim=32, num_heads=4, dropout_rate=0.0)
        x = tf.ones([1, 8, 32])

        mask = tf.constant([[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
        output = layer(x, attention_mask=mask, training=False)
        output_np = keras.ops.convert_to_numpy(output)

        valid_norm = np.linalg.norm(output_np[0, :4, :])
        assert valid_norm > 1e-6, "Valid positions should have nonzero output"

    def test_attention_mask_none_equivalent_to_all_ones(self):
        """No mask should behave identically to all-ones mask."""
        layer = WaveFieldAttention(dim=32, num_heads=4, dropout_rate=0.0)
        x = tf.random.normal([2, 6, 32])

        out_none = layer(x, attention_mask=None, training=False)
        out_ones = layer(x, attention_mask=tf.ones([2, 6]), training=False)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(out_none),
            keras.ops.convert_to_numpy(out_ones),
            rtol=1e-6, atol=1e-6,
        )

    def test_attention_mask_all_zeros_gives_zero_output(self):
        """All-zero mask should produce zero output."""
        layer = WaveFieldAttention(dim=32, num_heads=4, dropout_rate=0.0)
        x = tf.random.normal([1, 5, 32])
        mask = tf.zeros([1, 5])

        output = layer(x, attention_mask=mask, training=False)
        output_np = keras.ops.convert_to_numpy(output)
        np.testing.assert_array_equal(output_np, 0.0)

    def test_attention_mask_gradient_flow(self):
        """Gradients should flow through masked forward pass."""
        layer = WaveFieldAttention(dim=32, num_heads=4)
        x = tf.Variable(tf.random.normal([2, 8, 32]))
        mask = tf.constant([
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ])

        with tf.GradientTape() as tape:
            output = layer(x, attention_mask=mask)
            loss = tf.reduce_mean(tf.square(output))

        grads = tape.gradient(loss, layer.trainable_variables)
        assert all(g is not None for g in grads)

    # ==================== Query Modulation Tests ====================

    def test_query_modulation_affects_output(self):
        """Output should differ when Q projection weights change.

        We verify Q is wired in by perturbing the Q slice of qkv_proj
        weights and checking that output changes.
        """
        layer = WaveFieldAttention(dim=32, num_heads=4, dropout_rate=0.0)
        x = tf.ones([1, 5, 32])
        out_before = keras.ops.convert_to_numpy(layer(x, training=False))

        # Perturb only the Q slice of qkv_proj kernel (first dim columns)
        kernel = layer.qkv_proj.kernel  # (32, 96) for dim=32
        kernel_np = keras.ops.convert_to_numpy(kernel)
        kernel_np[:, :32] += 1.0  # perturb Q columns only
        layer.qkv_proj.kernel.assign(kernel_np)

        out_after = keras.ops.convert_to_numpy(layer(x, training=False))

        assert not np.allclose(out_before, out_after, atol=1e-6), (
            "Perturbing Q weights should change output (Q is used for gather modulation)"
        )

    def test_query_gradient_nonzero(self):
        """Gradients should flow into the Q slice of qkv_proj weights."""
        layer = WaveFieldAttention(dim=32, num_heads=4)
        x = tf.random.normal([2, 8, 32])

        with tf.GradientTape() as tape:
            output = layer(x)
            loss = tf.reduce_mean(tf.square(output))

        grad = tape.gradient(loss, layer.qkv_proj.kernel)
        grad_np = keras.ops.convert_to_numpy(grad)

        # Q slice is first 32 columns
        q_grad = grad_np[:, :32]
        assert np.any(np.abs(q_grad) > 1e-10), (
            "Gradient through Q slice of qkv_proj should be nonzero"
        )

    # ==================== Field Position Tests ====================

    def test_field_positions_absolute(self):
        layer = WaveFieldAttention(dim=32, num_heads=4, field_size=512, max_seq_len=128)
        layer.build((None, 10, 32))

        pos_short = layer._compute_field_positions(5)
        pos_long = layer._compute_field_positions(10)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(pos_short),
            keras.ops.convert_to_numpy(pos_long[:5]),
            rtol=1e-6, atol=1e-6,
        )

    def test_field_positions_clamped(self):
        layer = WaveFieldAttention(dim=32, num_heads=4, field_size=64, max_seq_len=32)
        layer.build((None, 10, 32))

        pos = keras.ops.convert_to_numpy(layer._compute_field_positions(100))
        assert np.all(pos >= 0.0)
        assert np.all(pos <= 62.0)

    def test_field_stride_computation(self):
        layer = WaveFieldAttention(dim=32, num_heads=4, field_size=512, max_seq_len=128)
        expected_stride = (512 - 1) / (128 - 1)
        np.testing.assert_allclose(layer._field_stride, expected_stride, rtol=1e-6)

    # ==================== Scatter / Gather Tests ====================

    def test_scatter_gather_matrices_shapes(self):
        layer = WaveFieldAttention(dim=32, num_heads=4, field_size=64, max_seq_len=32)
        layer.build((None, 10, 32))

        field_pos = layer._compute_field_positions(10)
        scatter_mat, gather_mat = layer._build_scatter_gather_matrices(field_pos)

        assert scatter_mat.shape == (64, 10)
        assert gather_mat.shape == (10, 64)

    def test_scatter_gather_transpose_relation(self):
        layer = WaveFieldAttention(dim=32, num_heads=4, field_size=64, max_seq_len=32)
        layer.build((None, 10, 32))

        field_pos = layer._compute_field_positions(7)
        scatter_mat, gather_mat = layer._build_scatter_gather_matrices(field_pos)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(scatter_mat),
            keras.ops.convert_to_numpy(keras.ops.transpose(gather_mat)),
            rtol=1e-6, atol=1e-6,
        )

    def test_scatter_matrix_column_sums(self):
        layer = WaveFieldAttention(dim=32, num_heads=4, field_size=64, max_seq_len=32)
        layer.build((None, 10, 32))

        field_pos = layer._compute_field_positions(10)
        scatter_mat, _ = layer._build_scatter_gather_matrices(field_pos)

        col_sums = keras.ops.convert_to_numpy(keras.ops.sum(scatter_mat, axis=0))
        np.testing.assert_allclose(col_sums, np.ones(10), rtol=1e-6, atol=1e-6)

    # ==================== Wave Kernel Tests ====================

    def test_wave_kernel_fft_shape(self):
        field_size = 64
        num_heads = 4
        layer = WaveFieldAttention(dim=32, num_heads=num_heads, field_size=field_size)
        layer.build((None, 10, 32))

        kernel_fft = layer._build_wave_kernels_fft()

        assert isinstance(kernel_fft, tuple)
        assert len(kernel_fft) == 2
        assert kernel_fft[0].shape == (num_heads, field_size + 1)
        assert kernel_fft[1].shape == (num_heads, field_size + 1)

    def test_wave_convolve_output_shape(self):
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
        layer = WaveFieldAttention(dim=64, num_heads=8, field_size=128)
        layer.build((None, 10, 64))

        field = tf.random.normal([2, 8, 128, 8])
        kernel_fft = layer._build_wave_kernels_fft()
        convolved = layer._wave_convolve(field, kernel_fft)

        assert not tf.reduce_any(tf.math.is_nan(convolved))
        assert not tf.reduce_any(tf.math.is_inf(convolved))

    # ==================== Field Coupling Tests ====================

    def test_field_coupling_output_shape(self):
        layer = WaveFieldAttention(dim=64, num_heads=8, field_size=128)
        layer.build((None, 10, 64))

        field = tf.random.normal([2, 8, 128, 8])
        coupled = layer._apply_field_coupling(field)

        assert coupled.shape == field.shape

    def test_field_coupling_near_identity_init(self):
        """With small noise, coupling should approximately preserve input."""
        layer = WaveFieldAttention(
            dim=32, num_heads=4, field_size=64, coupling_noise_stddev=0.001,
        )
        layer.build((None, 10, 32))

        field = tf.random.normal([2, 4, 64, 8])
        coupled = layer._apply_field_coupling(field)

        field_np = keras.ops.convert_to_numpy(field)
        coupled_np = keras.ops.convert_to_numpy(coupled)

        for h in range(4):
            corr = np.corrcoef(field_np[0, h].flatten(), coupled_np[0, h].flatten())[0, 1]
            assert corr > 0.5, f"Head {h} coupling correlation too low: {corr}"

    # ==================== Serialization Tests ====================

    def test_serialization_config_completeness(self):
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
        original = WaveFieldAttention(
            dim=128,
            num_heads=8,
            field_size=256,
            max_seq_len=64,
            dropout_rate=0.1,
            use_bias=False,
            gate_bias_init=1.5,
            coupling_noise_stddev=0.02,
        )

        config = original.get_config()
        recreated = WaveFieldAttention.from_config(config)

        assert recreated.dim == original.dim
        assert recreated.num_heads == original.num_heads
        assert recreated.field_size == original.field_size
        assert recreated.max_seq_len == original.max_seq_len
        assert recreated.dropout_rate == original.dropout_rate
        assert recreated.use_bias == original.use_bias
        assert recreated.gate_bias_init == original.gate_bias_init
        assert recreated.coupling_noise_stddev == original.coupling_noise_stddev

    def test_serialization_with_build(self, input_tensor):
        original = WaveFieldAttention(dim=64, num_heads=8, dropout_rate=0.1, use_bias=True)
        original(input_tensor)

        config = original.get_config()
        recreated = WaveFieldAttention.from_config(config)
        recreated(input_tensor)

        assert len(recreated.weights) == len(original.weights)

    # ==================== Model Integration Tests ====================

    def test_model_integration(self, input_tensor):
        inputs = keras.Input(shape=input_tensor.shape[1:])
        x = WaveFieldAttention(dim=64, num_heads=8)(inputs)
        x = keras.layers.LayerNormalization()(x)
        x = keras.layers.GlobalAveragePooling1D()(x)
        outputs = keras.layers.Dense(10)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

        y_pred = model(input_tensor, training=False)
        assert y_pred.shape == (input_tensor.shape[0], 10)

    def test_model_integration_with_mask(self):
        """Model should accept attention_mask via layer call."""
        layer = WaveFieldAttention(dim=32, num_heads=4)
        x = tf.random.normal([2, 10, 32])
        mask = tf.constant([
            [1.0] * 7 + [0.0] * 3,
            [1.0] * 5 + [0.0] * 5,
        ])

        output = layer(x, attention_mask=mask)
        assert output.shape == (2, 10, 32)

        # Verify masked positions are zero per batch element
        out_np = keras.ops.convert_to_numpy(output)
        np.testing.assert_array_equal(out_np[0, 7:, :], 0.0)
        np.testing.assert_array_equal(out_np[1, 5:, :], 0.0)

    # ==================== Model Save/Load Tests ====================

    def test_model_save_load(self, input_tensor):
        inputs = keras.Input(shape=input_tensor.shape[1:])
        x = WaveFieldAttention(dim=64, num_heads=8, name="wave_field_attn")(inputs)
        x = keras.layers.GlobalAveragePooling1D()(x)
        outputs = keras.layers.Dense(10)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        original_prediction = model.predict(input_tensor, verbose=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.keras")
            model.save(model_path)

            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={"WaveFieldAttention": WaveFieldAttention},
            )

            loaded_prediction = loaded_model.predict(input_tensor, verbose=0)

            np.testing.assert_allclose(
                original_prediction, loaded_prediction,
                rtol=1e-5, atol=1e-5,
            )

            assert isinstance(
                loaded_model.get_layer("wave_field_attn"),
                WaveFieldAttention,
            )

    # ==================== Gradient Flow Tests ====================

    def test_gradient_flow(self, input_tensor):
        layer = WaveFieldAttention(dim=64, num_heads=8)

        with tf.GradientTape() as tape:
            inputs = tf.Variable(input_tensor)
            outputs = layer(inputs)
            loss = tf.reduce_mean(tf.square(outputs))

        grads = tape.gradient(loss, layer.trainable_variables)
        assert all(g is not None for g in grads)
        assert all(tf.reduce_any(g != 0) for g in grads)

    def test_gradient_flow_wave_params(self, input_tensor):
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
            assert tf.reduce_any(grad != 0), f"Gradient for {var.name} should have nonzero values"

    # ==================== Edge Case Tests ====================

    def test_numerical_stability(self):
        layer = WaveFieldAttention(dim=64, num_heads=8)

        test_cases = [
            ("zeros", tf.zeros((2, 10, 64))),
            ("tiny", tf.ones((2, 10, 64)) * 1e-10),
            ("large", tf.ones((2, 10, 64)) * 1e3),
            ("random_large", tf.random.normal((2, 10, 64)) * 10),
        ]

        for name, test_input in test_cases:
            output = layer(test_input)
            assert not tf.reduce_any(tf.math.is_nan(output)), f"NaN for '{name}'"
            assert not tf.reduce_any(tf.math.is_inf(output)), f"Inf for '{name}'"

    def test_single_sequence_length(self):
        layer = WaveFieldAttention(dim=64, num_heads=8)
        output = layer(tf.random.normal([2, 1, 64]))
        assert output.shape == (2, 1, 64)
        assert not tf.reduce_any(tf.math.is_nan(output))

    def test_large_sequence_length(self):
        layer = WaveFieldAttention(dim=64, num_heads=8, max_seq_len=512)
        output = layer(tf.random.normal([1, 500, 64]))
        assert output.shape == (1, 500, 64)
        assert not tf.reduce_any(tf.math.is_nan(output))

    def test_sequence_exceeds_max_seq_len(self):
        layer = WaveFieldAttention(dim=32, num_heads=4, field_size=64, max_seq_len=10)
        output = layer(tf.random.normal([1, 20, 32]))
        assert output.shape == (1, 20, 32)
        assert not tf.reduce_any(tf.math.is_nan(output))

    def test_single_head(self):
        layer = WaveFieldAttention(dim=64, num_heads=1)
        output = layer(tf.random.normal([2, 10, 64]))
        assert output.shape == (2, 10, 64)
        assert not tf.reduce_any(tf.math.is_nan(output))

    # ==================== Configuration Variation Tests ====================

    def test_different_head_counts(self):
        dim = 64
        x = tf.random.normal([2, 10, dim])
        for num_heads in [1, 2, 4, 8, 16]:
            if dim % num_heads == 0:
                output = WaveFieldAttention(dim=dim, num_heads=num_heads)(x)
                assert output.shape == x.shape
                assert not tf.reduce_any(tf.math.is_nan(output))

    def test_different_field_sizes(self):
        x = tf.random.normal([2, 10, 32])
        for field_size in [32, 64, 128, 512]:
            output = WaveFieldAttention(dim=32, num_heads=4, field_size=field_size)(x)
            assert output.shape == x.shape
            assert not tf.reduce_any(tf.math.is_nan(output))

    def test_gate_bias_effect(self):
        x = tf.ones([1, 5, 32])

        layer_open = WaveFieldAttention(dim=32, num_heads=4, gate_bias_init=10.0)
        out_open = layer_open(x)

        layer_closed = WaveFieldAttention(dim=32, num_heads=4, gate_bias_init=-10.0)
        out_closed = layer_closed(x)

        open_norm = float(tf.reduce_mean(tf.abs(out_open)))
        closed_norm = float(tf.reduce_mean(tf.abs(out_closed)))

        assert open_norm > closed_norm, (
            f"Open gate norm ({open_norm}) should exceed closed gate norm ({closed_norm})"
        )

    def test_dropout_creates_layer_when_nonzero(self):
        assert WaveFieldAttention(dim=32, num_heads=4, dropout_rate=0.0).dropout_layer is None

        layer = WaveFieldAttention(dim=32, num_heads=4, dropout_rate=0.1)
        assert isinstance(layer.dropout_layer, keras.layers.Dropout)

    # ==================== Sublayer Structure Tests ====================

    def test_sublayer_existence(self, input_tensor):
        layer = WaveFieldAttention(dim=64, num_heads=8)
        layer(input_tensor)

        assert isinstance(layer.qkv_proj, keras.layers.Dense)
        assert isinstance(layer.output_proj, keras.layers.Dense)
        assert isinstance(layer.gate_proj, keras.layers.Dense)

    def test_qkv_projection_output_units(self, input_tensor):
        layer = WaveFieldAttention(dim=64, num_heads=8)
        layer(input_tensor)
        assert layer.qkv_proj.units == 3 * 64

    def test_trainable_variable_count(self, input_tensor):
        layer = WaveFieldAttention(dim=64, num_heads=8, dropout_rate=0.0)
        layer(input_tensor)

        # qkv_proj: kernel + bias = 2
        # output_proj: kernel + bias = 2
        # gate_proj: kernel + bias = 2
        # wave_frequency, wave_damping, wave_phase = 3
        # field_coupling = 1
        # Total = 10
        assert len(layer.trainable_variables) == 10

    def test_trainable_variable_count_no_bias(self, input_tensor):
        layer = WaveFieldAttention(dim=64, num_heads=8, use_bias=False, dropout_rate=0.0)
        layer(input_tensor)

        # qkv_proj: kernel = 1
        # output_proj: kernel = 1
        # gate_proj: always has bias = 2
        # wave params = 3, coupling = 1
        # Total = 8
        assert len(layer.trainable_variables) == 8

    # ==================== Consistency Tests ====================

    def test_different_seq_lengths_same_weights(self):
        layer = WaveFieldAttention(dim=32, num_heads=4, dropout_rate=0.0)
        layer(tf.random.normal([1, 10, 32]))

        out_short = layer(tf.ones([1, 5, 32]), training=False)
        out_long = layer(tf.ones([1, 20, 32]), training=False)

        assert not tf.reduce_any(tf.math.is_nan(out_short))
        assert not tf.reduce_any(tf.math.is_nan(out_long))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])