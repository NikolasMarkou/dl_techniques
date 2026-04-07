import pytest
import numpy as np
import tensorflow as tf
import keras
import os
import tempfile

from dl_techniques.layers.geometric.clifford_block import (
    SparseRollingGeometricProduct,
    GatedGeometricResidual,
    CliffordNetBlock,
    CausalCliffordNetBlock,
)


# ===========================================================================
# TestSparseRollingGeometricProduct
# ===========================================================================


class TestSparseRollingGeometricProduct:
    """Test suite for SparseRollingGeometricProduct."""

    @pytest.fixture
    def channels(self) -> int:
        return 16

    @pytest.fixture
    def shifts(self) -> list:
        return [1, 2]

    @pytest.fixture
    def input_tensor(self) -> tf.Tensor:
        return tf.random.normal([2, 8, 8, 16])

    @pytest.fixture
    def layer_instance(self, channels, shifts) -> SparseRollingGeometricProduct:
        return SparseRollingGeometricProduct(channels=channels, shifts=shifts)

    # ------------------------------------------------------------------

    def test_initialization_defaults(self, channels, shifts):
        """Test initialization with default parameters."""
        layer = SparseRollingGeometricProduct(channels=channels, shifts=shifts)
        assert layer.channels == channels
        assert layer.shifts == shifts
        assert layer.cli_mode == "full"
        assert layer.use_bias is True

    def test_initialization_custom(self, channels, shifts):
        """Test initialization with custom parameters."""
        layer = SparseRollingGeometricProduct(
            channels=channels,
            shifts=shifts,
            cli_mode="inner",
            use_bias=False,
            name="custom_geo",
        )
        assert layer.cli_mode == "inner"
        assert layer.use_bias is False
        assert layer.name == "custom_geo"

    def test_invalid_channels(self, shifts):
        """Test that non-positive channels raises ValueError."""
        with pytest.raises(ValueError, match="channels"):
            SparseRollingGeometricProduct(channels=0, shifts=shifts)

    def test_invalid_shifts(self, channels):
        """Test that empty shifts raises ValueError."""
        with pytest.raises(ValueError, match="shifts"):
            SparseRollingGeometricProduct(channels=channels, shifts=[])

    def test_invalid_cli_mode(self, channels, shifts):
        """Test that unknown cli_mode raises ValueError."""
        with pytest.raises(ValueError, match="cli_mode"):
            SparseRollingGeometricProduct(channels=channels, shifts=shifts, cli_mode="bad")

    def test_build(self, layer_instance, input_tensor):
        """Test that the layer builds and has the projection weight."""
        layer_instance(input_tensor, input_tensor)
        assert layer_instance.built is True
        assert layer_instance.proj.built is True

    def test_output_shape_full_mode(self, channels, input_tensor):
        """Test output shape in full (default) mode."""
        layer = SparseRollingGeometricProduct(channels=channels, shifts=[1, 2])
        output = layer(input_tensor, input_tensor)
        assert output.shape == input_tensor.shape

    def test_output_shape_inner_mode(self, channels, input_tensor):
        """Test output shape in inner-only mode."""
        layer = SparseRollingGeometricProduct(channels=channels, shifts=[1, 2], cli_mode="inner")
        output = layer(input_tensor, input_tensor)
        assert output.shape == input_tensor.shape

    def test_output_shape_wedge_mode(self, channels, input_tensor):
        """Test output shape in wedge-only mode."""
        layer = SparseRollingGeometricProduct(channels=channels, shifts=[1, 2], cli_mode="wedge")
        output = layer(input_tensor, input_tensor)
        assert output.shape == input_tensor.shape

    def test_compute_output_shape(self, layer_instance, input_tensor):
        """Test compute_output_shape matches actual output."""
        layer_instance(input_tensor, input_tensor)
        computed = layer_instance.compute_output_shape(input_tensor.shape)
        assert computed == input_tensor.shape

    def test_compute_output_shape_before_build(self, channels, shifts):
        """Test compute_output_shape works before layer is built."""
        layer = SparseRollingGeometricProduct(channels=channels, shifts=shifts)
        result = layer.compute_output_shape((None, 8, 8, channels))
        assert result == (None, 8, 8, channels)

    def test_wedge_antisymmetry(self, channels, input_tensor):
        """Wedge component must be zero when Z_det == Z_ctx (self-wedge is 0).

        With identical inputs the raw concatenated wedge tensor is all-zeros.
        The projection of that zero tensor equals the bias vector (or zero when
        use_bias=False).  We construct the zero tensor at the correct proj-input
        width: ``|shifts| * D`` channels.
        """
        shifts = [1, 2, 4]
        layer = SparseRollingGeometricProduct(channels=channels, shifts=shifts, cli_mode="wedge")
        output = layer(input_tensor, input_tensor)

        # proj input width = |shifts| * D  (wedge mode: 1 component per shift)
        proj_input_width = len(shifts) * channels
        batch, h, w, _ = input_tensor.shape
        zeros_proj_input = tf.zeros([batch, h, w, proj_input_width])
        projected_zero = layer.proj(zeros_proj_input)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output),
            keras.ops.convert_to_numpy(projected_zero),
            rtol=1e-5, atol=1e-5,
            err_msg="Wedge of identical inputs should equal projection of zeros",
        )

    def test_full_mode_uses_both_components(self, channels):
        """Full mode output differs from inner-only and wedge-only for distinct inputs."""
        x = tf.random.normal([2, 4, 4, channels], seed=0)
        y = tf.random.normal([2, 4, 4, channels], seed=1)

        full_layer = SparseRollingGeometricProduct(channels=channels, shifts=[1], cli_mode="full")
        inner_layer = SparseRollingGeometricProduct(channels=channels, shifts=[1], cli_mode="inner")
        wedge_layer = SparseRollingGeometricProduct(channels=channels, shifts=[1], cli_mode="wedge")

        out_full = full_layer(x, y).numpy()
        out_inner = inner_layer(x, y).numpy()
        out_wedge = wedge_layer(x, y).numpy()

        # All three have different (randomly initialised) projections, so outputs differ
        assert not np.allclose(out_full, out_inner, atol=1e-3)
        assert not np.allclose(out_full, out_wedge, atol=1e-3)

    def test_numerical_stability(self, channels):
        """No NaN / Inf with extreme input values."""
        layer = SparseRollingGeometricProduct(channels=channels, shifts=[1, 2])
        for scale in [1e-8, 1e8]:
            x = tf.ones([2, 4, 4, channels]) * scale
            out = layer(x, x)
            assert not np.any(np.isnan(out.numpy())), f"NaN at scale {scale}"
            assert not np.any(np.isinf(out.numpy())), f"Inf at scale {scale}"

    def test_different_batch_sizes(self, channels, shifts):
        """Layer handles variable batch sizes."""
        layer = SparseRollingGeometricProduct(channels=channels, shifts=shifts)
        for bs in [1, 4, 16]:
            x = tf.random.normal([bs, 6, 6, channels])
            out = layer(x, x)
            assert out.shape[0] == bs

    def test_serialization(self, channels, shifts):
        """get_config / from_config round-trip preserves attributes."""
        original = SparseRollingGeometricProduct(
            channels=channels, shifts=shifts, cli_mode="inner", name="geo_s"
        )
        config = original.get_config()
        restored = SparseRollingGeometricProduct.from_config(config)

        assert restored.channels == original.channels
        assert restored.shifts == original.shifts
        assert restored.cli_mode == original.cli_mode

    def test_model_save_load(self, channels, shifts):
        """Save / load through Keras .keras format preserves outputs."""
        x = tf.random.normal([2, 8, 8, channels])

        inp_a = keras.Input(shape=(8, 8, channels))
        inp_b = keras.Input(shape=(8, 8, channels))
        out = SparseRollingGeometricProduct(channels=channels, shifts=shifts, name="srgp")(inp_a, inp_b)
        model = keras.Model(inputs=[inp_a, inp_b], outputs=out)

        original_pred = model.predict([x, x], verbose=0)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.keras")
            model.save(path)
            loaded = keras.models.load_model(path)

        loaded_pred = loaded.predict([x, x], verbose=0)
        np.testing.assert_allclose(
            original_pred, loaded_pred, rtol=1e-5, atol=1e-5,
            err_msg="Predictions should match after save/load",
        )

    def test_gradient_flow(self, channels, shifts):
        """Gradients propagate through the layer."""
        layer = SparseRollingGeometricProduct(channels=channels, shifts=shifts)
        x = tf.Variable(tf.random.normal([2, 4, 4, channels]))
        with tf.GradientTape() as tape:
            out = layer(x, x)
            loss = tf.reduce_mean(tf.square(out))
        grads = tape.gradient(loss, x)
        assert grads is not None
        assert np.any(grads.numpy() != 0)


# ===========================================================================
# TestGatedGeometricResidual
# ===========================================================================


class TestGatedGeometricResidual:
    """Test suite for GatedGeometricResidual."""

    @pytest.fixture
    def channels(self) -> int:
        return 16

    @pytest.fixture
    def input_tensor(self) -> tf.Tensor:
        return tf.random.normal([2, 8, 8, 16])

    @pytest.fixture
    def layer_instance(self, channels) -> GatedGeometricResidual:
        return GatedGeometricResidual(channels=channels)

    # ------------------------------------------------------------------

    def test_initialization_defaults(self, channels):
        """Test initialization with default parameters."""
        layer = GatedGeometricResidual(channels=channels)
        assert layer.channels == channels
        assert layer.layer_scale_init == 1e-5
        assert layer.drop_path_rate == 0.0

    def test_initialization_custom(self, channels):
        """Test initialization with custom parameters."""
        layer = GatedGeometricResidual(
            channels=channels,
            layer_scale_init=1e-3,
            drop_path_rate=0.1,
            name="custom_ggr",
        )
        assert layer.layer_scale_init == 1e-3
        assert layer.drop_path_rate == 0.1
        assert layer.name == "custom_ggr"

    def test_invalid_channels(self):
        """Test that non-positive channels raises ValueError."""
        with pytest.raises(ValueError, match="channels"):
            GatedGeometricResidual(channels=-1)

    def test_invalid_drop_path_rate(self, channels):
        """Test that drop_path_rate >= 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="drop_path_rate"):
            GatedGeometricResidual(channels=channels, drop_path_rate=1.0)

    def test_build_creates_gamma(self, layer_instance, input_tensor):
        """Build must create the gamma LayerScale weight."""
        layer_instance(input_tensor, input_tensor)
        assert layer_instance.built is True
        assert hasattr(layer_instance, "gamma")
        assert layer_instance.gamma.shape == (layer_instance.channels,)

    def test_gamma_init_value(self, channels, input_tensor):
        """Gamma is initialised to layer_scale_init."""
        init_val = 1e-3
        layer = GatedGeometricResidual(channels=channels, layer_scale_init=init_val)
        layer(input_tensor, input_tensor)
        np.testing.assert_allclose(
            layer.gamma.numpy(),
            np.full((channels,), init_val),
            rtol=1e-6, atol=1e-6,
            err_msg="gamma should be initialised to layer_scale_init",
        )

    def test_output_shape(self, layer_instance, input_tensor):
        """Output shape matches input shape."""
        out = layer_instance(input_tensor, input_tensor)
        assert out.shape == input_tensor.shape

    def test_compute_output_shape(self, layer_instance, input_tensor):
        """compute_output_shape matches actual output."""
        layer_instance(input_tensor, input_tensor)
        computed = layer_instance.compute_output_shape(input_tensor.shape)
        assert computed == input_tensor.shape

    def test_compute_output_shape_before_build(self, channels):
        """compute_output_shape works before build."""
        layer = GatedGeometricResidual(channels=channels)
        result = layer.compute_output_shape((None, 8, 8, channels))
        assert result == (None, 8, 8, channels)

    def test_drop_path_absent_when_zero(self, channels, input_tensor):
        """No StochasticDepth layer when drop_path_rate is 0."""
        layer = GatedGeometricResidual(channels=channels, drop_path_rate=0.0)
        layer(input_tensor, input_tensor)
        assert layer.drop_path is None

    def test_drop_path_present_when_nonzero(self, channels, input_tensor):
        """StochasticDepth is created when drop_path_rate > 0."""
        layer = GatedGeometricResidual(channels=channels, drop_path_rate=0.2)
        layer(input_tensor, input_tensor)
        assert layer.drop_path is not None

    def test_inference_vs_training_no_droppath(self, channels, input_tensor):
        """Without DropPath, training and inference outputs are identical."""
        layer = GatedGeometricResidual(channels=channels, drop_path_rate=0.0)
        g = tf.random.normal(input_tensor.shape)
        out_train = layer(input_tensor, g, training=True)
        out_infer = layer(input_tensor, g, training=False)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(out_train),
            keras.ops.convert_to_numpy(out_infer),
            rtol=1e-6, atol=1e-6,
            err_msg="Training vs inference should match with no DropPath",
        )

    def test_numerical_stability(self, channels):
        """No NaN / Inf with extreme values."""
        layer = GatedGeometricResidual(channels=channels)
        for scale in [1e-8, 1e8]:
            x = tf.ones([2, 4, 4, channels]) * scale
            out = layer(x, x)
            assert not np.any(np.isnan(out.numpy())), f"NaN at scale {scale}"
            assert not np.any(np.isinf(out.numpy())), f"Inf at scale {scale}"

    def test_serialization(self, channels):
        """get_config / from_config round-trip preserves attributes."""
        original = GatedGeometricResidual(
            channels=channels, layer_scale_init=1e-3, drop_path_rate=0.1, name="ggr_s"
        )
        config = original.get_config()
        restored = GatedGeometricResidual.from_config(config)

        assert restored.channels == original.channels
        assert restored.layer_scale_init == original.layer_scale_init
        assert restored.drop_path_rate == original.drop_path_rate

    def test_model_save_load(self, channels):
        """Save / load through Keras .keras format preserves outputs."""
        x = tf.random.normal([2, 8, 8, channels])
        g = tf.random.normal([2, 8, 8, channels])

        inp_h = keras.Input(shape=(8, 8, channels))
        inp_g = keras.Input(shape=(8, 8, channels))
        out = GatedGeometricResidual(channels=channels, name="ggr")(inp_h, inp_g)
        model = keras.Model(inputs=[inp_h, inp_g], outputs=out)

        original_pred = model.predict([x, g], verbose=0)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.keras")
            model.save(path)
            loaded = keras.models.load_model(path)

        loaded_pred = loaded.predict([x, g], verbose=0)
        np.testing.assert_allclose(
            original_pred, loaded_pred, rtol=1e-5, atol=1e-5,
            err_msg="Predictions should match after save/load",
        )

    def test_gradient_flow(self, channels):
        """Gradients flow through GGR and back to inputs."""
        layer = GatedGeometricResidual(channels=channels)
        x = tf.Variable(tf.random.normal([2, 4, 4, channels]))
        with tf.GradientTape() as tape:
            out = layer(x, x)
            loss = tf.reduce_mean(tf.square(out))
        grads = tape.gradient(loss, x)
        assert grads is not None
        assert np.any(grads.numpy() != 0)


# ===========================================================================
# TestCliffordNetBlock
# ===========================================================================


class TestCliffordNetBlock:
    """Test suite for CliffordNetBlock."""

    @pytest.fixture
    def channels(self) -> int:
        return 16

    @pytest.fixture
    def shifts(self) -> list:
        return [1, 2]

    @pytest.fixture
    def input_tensor(self) -> tf.Tensor:
        return tf.random.normal([2, 8, 8, 16])

    @pytest.fixture
    def layer_instance(self, channels, shifts) -> CliffordNetBlock:
        return CliffordNetBlock(channels=channels, shifts=shifts)

    # ------------------------------------------------------------------

    def test_initialization_defaults(self, channels, shifts):
        """Test initialization with default parameters."""
        layer = CliffordNetBlock(channels=channels, shifts=shifts)
        assert layer.channels == channels
        assert layer.shifts == shifts
        assert layer.cli_mode == "full"
        assert layer.ctx_mode == "diff"
        assert layer.use_global_context is False
        assert layer.layer_scale_init == 1e-5
        assert layer.drop_path_rate == 0.0

    def test_initialization_custom(self, channels, shifts):
        """Test initialization with custom parameters."""
        layer = CliffordNetBlock(
            channels=channels,
            shifts=shifts,
            cli_mode="inner",
            ctx_mode="abs",
            use_global_context=True,
            layer_scale_init=1e-3,
            drop_path_rate=0.1,
            name="custom_cb",
        )
        assert layer.cli_mode == "inner"
        assert layer.ctx_mode == "abs"
        assert layer.use_global_context is True
        assert layer.drop_path_rate == 0.1
        assert layer.name == "custom_cb"

    def test_invalid_channels(self, shifts):
        """Test that non-positive channels raises ValueError."""
        with pytest.raises(ValueError, match="channels"):
            CliffordNetBlock(channels=0, shifts=shifts)

    def test_invalid_ctx_mode(self, channels, shifts):
        """Test that unknown ctx_mode raises ValueError."""
        with pytest.raises(ValueError, match="ctx_mode"):
            CliffordNetBlock(channels=channels, shifts=shifts, ctx_mode="unknown")

    def test_build(self, layer_instance, input_tensor):
        """Layer and all sub-layers build correctly."""
        layer_instance(input_tensor)
        assert layer_instance.built is True
        assert layer_instance.input_norm.built is True
        assert layer_instance.linear_det.built is True
        assert layer_instance.dw_conv.built is True
        assert layer_instance.ctx_bn.built is True
        assert layer_instance.local_geo_prod.built is True
        assert layer_instance.ggr.built is True

    def test_global_branch_absent_by_default(self, layer_instance, input_tensor):
        """global_geo_prod is None when use_global_context=False."""
        layer_instance(input_tensor)
        assert layer_instance.global_geo_prod is None

    def test_global_branch_present_when_enabled(self, channels, shifts, input_tensor):
        """global_geo_prod is created when use_global_context=True."""
        layer = CliffordNetBlock(channels=channels, shifts=shifts, use_global_context=True)
        layer(input_tensor)
        assert layer.global_geo_prod is not None

    def test_output_shape(self, layer_instance, input_tensor):
        """Output shape equals input shape (isotropic)."""
        out = layer_instance(input_tensor)
        assert out.shape == input_tensor.shape

    def test_output_shape_global_context(self, channels, shifts, input_tensor):
        """Output shape is preserved with global context branch."""
        layer = CliffordNetBlock(channels=channels, shifts=shifts, use_global_context=True)
        out = layer(input_tensor)
        assert out.shape == input_tensor.shape

    def test_compute_output_shape(self, layer_instance, input_tensor):
        """compute_output_shape matches actual output."""
        layer_instance(input_tensor)
        computed = layer_instance.compute_output_shape(input_tensor.shape)
        assert computed == input_tensor.shape

    def test_compute_output_shape_before_build(self, channels, shifts):
        """compute_output_shape works before layer is built."""
        layer = CliffordNetBlock(channels=channels, shifts=shifts)
        result = layer.compute_output_shape((None, 8, 8, channels))
        assert result == (None, 8, 8, channels)

    def test_ctx_mode_diff_vs_abs_differ(self, channels, shifts):
        """Differential and absolute context modes produce different outputs.

        layer_scale_init=1.0 ensures the geometric interaction term contributes
        meaningfully; with the default ~0 init both outputs collapse to x_prev.
        """
        x = tf.random.normal([2, 8, 8, channels], seed=42)
        layer_diff = CliffordNetBlock(
            channels=channels, shifts=shifts, ctx_mode="diff", layer_scale_init=1.0
        )
        layer_abs = CliffordNetBlock(
            channels=channels, shifts=shifts, ctx_mode="abs", layer_scale_init=1.0
        )
        out_diff = layer_diff(x, training=False).numpy()
        out_abs = layer_abs(x, training=False).numpy()
        assert not np.allclose(out_diff, out_abs, atol=1e-3)

    def test_residual_connection_identity_at_init(self, channels, shifts):
        """With gamma ~ 0 at init, output should be very close to input."""
        layer = CliffordNetBlock(
            channels=channels, shifts=shifts, layer_scale_init=1e-10
        )
        x = tf.random.normal([2, 4, 4, channels])
        out = layer(x)
        # H_mix is scaled by ~0, so X_out ≈ X_prev
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(out),
            keras.ops.convert_to_numpy(x),
            rtol=1e-4, atol=1e-4,
            err_msg="With gamma ~ 0, output should be close to input (residual identity)",
        )

    def test_different_shift_sets(self, channels):
        """Layer works with various shift configurations."""
        x = tf.random.normal([2, 8, 8, channels])
        for shifts in [[1], [1, 2], [1, 2, 4, 8, 16]]:
            layer = CliffordNetBlock(channels=channels, shifts=shifts)
            out = layer(x)
            assert out.shape == x.shape

    def test_all_cli_modes(self, channels, shifts, input_tensor):
        """Layer produces valid outputs for all cli_mode settings."""
        for mode in ("inner", "wedge", "full"):
            layer = CliffordNetBlock(channels=channels, shifts=shifts, cli_mode=mode)
            out = layer(input_tensor)
            assert out.shape == input_tensor.shape
            assert not np.any(np.isnan(out.numpy())), f"NaN in cli_mode={mode}"

    def test_numerical_stability(self, channels, shifts):
        """No NaN / Inf with extreme input values."""
        layer = CliffordNetBlock(channels=channels, shifts=shifts)
        for scale in [1e-8, 1e8]:
            x = tf.ones([2, 4, 4, channels]) * scale
            out = layer(x)
            assert not np.any(np.isnan(out.numpy())), f"NaN at scale {scale}"
            assert not np.any(np.isinf(out.numpy())), f"Inf at scale {scale}"

    def test_different_spatial_sizes(self, channels, shifts):
        """Layer handles different spatial resolutions."""
        layer = CliffordNetBlock(channels=channels, shifts=shifts)
        for hw in [4, 16, 32]:
            x = tf.random.normal([2, hw, hw, channels])
            out = layer(x)
            assert out.shape == (2, hw, hw, channels)

    def test_different_batch_sizes(self, channels, shifts):
        """Layer handles variable batch sizes."""
        layer = CliffordNetBlock(channels=channels, shifts=shifts)
        for bs in [1, 4, 16]:
            x = tf.random.normal([bs, 8, 8, channels])
            out = layer(x)
            assert out.shape[0] == bs

    def test_training_vs_inference_no_droppath(self, channels, shifts, input_tensor):
        """Two inference-mode calls produce identical outputs (deterministic).

        Note: training=True intentionally differs from training=False because
        BatchNormalization in the context stream uses batch statistics during
        training and moving averages during inference.  The meaningful
        determinism check is that inference is reproducible.
        """
        layer = CliffordNetBlock(channels=channels, shifts=shifts, drop_path_rate=0.0)
        out_infer_1 = layer(input_tensor, training=False)
        out_infer_2 = layer(input_tensor, training=False)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(out_infer_1),
            keras.ops.convert_to_numpy(out_infer_2),
            rtol=1e-6, atol=1e-6,
            err_msg="Two inference calls should be identical",
        )

    def test_serialization(self, channels, shifts):
        """get_config / from_config round-trip preserves all attributes."""
        original = CliffordNetBlock(
            channels=channels,
            shifts=shifts,
            cli_mode="wedge",
            ctx_mode="abs",
            use_global_context=True,
            layer_scale_init=1e-3,
            drop_path_rate=0.1,
            name="cb_s",
        )
        config = original.get_config()
        restored = CliffordNetBlock.from_config(config)

        assert restored.channels == original.channels
        assert restored.shifts == original.shifts
        assert restored.cli_mode == original.cli_mode
        assert restored.ctx_mode == original.ctx_mode
        assert restored.use_global_context == original.use_global_context
        assert restored.layer_scale_init == original.layer_scale_init
        assert restored.drop_path_rate == original.drop_path_rate

    def test_model_save_load(self, channels, shifts):
        """Save / load through Keras .keras format preserves outputs."""
        x = tf.random.normal([2, 8, 8, channels])

        inp = keras.Input(shape=(8, 8, channels))
        out = CliffordNetBlock(channels=channels, shifts=shifts, name="cb")(inp)
        model = keras.Model(inputs=inp, outputs=out)

        original_pred = model.predict(x, verbose=0)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.keras")
            model.save(path)
            loaded = keras.models.load_model(path)

        loaded_pred = loaded.predict(x, verbose=0)
        np.testing.assert_allclose(
            original_pred, loaded_pred, rtol=1e-5, atol=1e-5,
            err_msg="Predictions should match after save/load",
        )

    def test_gradient_flow(self, channels, shifts):
        """Gradients propagate through the entire block."""
        layer = CliffordNetBlock(channels=channels, shifts=shifts)
        x = tf.Variable(tf.random.normal([2, 4, 4, channels]))
        with tf.GradientTape() as tape:
            out = layer(x, training=True)
            loss = tf.reduce_mean(tf.square(out))
        grads = tape.gradient(loss, x)
        assert grads is not None
        assert np.any(grads.numpy() != 0)

    def test_stacking_multiple_blocks(self, channels, shifts):
        """Multiple stacked blocks produce valid outputs without shape mismatch."""
        x = tf.random.normal([2, 8, 8, channels])
        blocks = [CliffordNetBlock(channels=channels, shifts=shifts, name=f"b{i}") for i in range(4)]
        for block in blocks:
            x = block(x)
        assert x.shape == (2, 8, 8, channels)
        assert not np.any(np.isnan(x.numpy()))


# ===========================================================================
# TestCausalCliffordNetBlock
# ===========================================================================


class TestCausalCliffordNetBlock:
    """Test suite for CausalCliffordNetBlock (autoregressive NLP variant)."""

    @pytest.fixture
    def channels(self) -> int:
        return 16

    @pytest.fixture
    def shifts(self) -> list:
        return [1, 2]

    @pytest.fixture
    def seq_tensor(self) -> tf.Tensor:
        """4-D sequence tensor (B, 1, seq_len, D)."""
        return tf.random.normal([2, 1, 16, 16])

    @pytest.fixture
    def layer_instance(self, channels, shifts) -> CausalCliffordNetBlock:
        return CausalCliffordNetBlock(channels=channels, shifts=shifts)

    # ---- Initialization ---------------------------------------------------

    def test_initialization_defaults(self, channels, shifts):
        layer = CausalCliffordNetBlock(channels=channels, shifts=shifts)
        assert layer.channels == channels
        assert layer.shifts == shifts
        assert layer.cli_mode == "full"
        assert layer.ctx_mode == "diff"

    def test_initialization_custom(self, channels, shifts):
        layer = CausalCliffordNetBlock(
            channels=channels, shifts=shifts,
            cli_mode="wedge", ctx_mode="abs",
            use_global_context=True, layer_scale_init=1e-3,
            drop_path_rate=0.1,
        )
        assert layer.cli_mode == "wedge"
        assert layer.ctx_mode == "abs"
        assert layer.use_global_context is True

    def test_invalid_channels(self, shifts):
        with pytest.raises(ValueError, match="channels"):
            CausalCliffordNetBlock(channels=0, shifts=shifts)

    def test_invalid_ctx_mode(self, channels, shifts):
        with pytest.raises(ValueError, match="ctx_mode"):
            CausalCliffordNetBlock(channels=channels, shifts=shifts, ctx_mode="bad")

    # ---- Build & Shape ----------------------------------------------------

    def test_build(self, layer_instance, seq_tensor):
        layer_instance(seq_tensor)
        assert layer_instance.built is True
        assert layer_instance.input_norm.built is True
        assert layer_instance.linear_det.built is True
        assert layer_instance.dw_conv.built is True
        assert layer_instance.dw_conv2.built is True
        assert layer_instance.ctx_bn.built is True
        assert layer_instance.local_geo_prod.built is True
        assert layer_instance.ggr.built is True

    def test_output_shape(self, layer_instance, seq_tensor):
        out = layer_instance(seq_tensor)
        assert out.shape == seq_tensor.shape

    def test_output_shape_global_context(self, channels, shifts, seq_tensor):
        layer = CausalCliffordNetBlock(
            channels=channels, shifts=shifts, use_global_context=True,
        )
        out = layer(seq_tensor)
        assert out.shape == seq_tensor.shape

    def test_compute_output_shape(self, layer_instance, seq_tensor):
        layer_instance(seq_tensor)
        computed = layer_instance.compute_output_shape(seq_tensor.shape)
        assert computed == seq_tensor.shape

    def test_different_sequence_lengths(self, channels, shifts):
        layer = CausalCliffordNetBlock(channels=channels, shifts=shifts)
        for seq_len in [4, 16, 64, 128]:
            x = tf.random.normal([2, 1, seq_len, channels])
            out = layer(x)
            assert out.shape == (2, 1, seq_len, channels)

    def test_different_batch_sizes(self, channels, shifts):
        layer = CausalCliffordNetBlock(channels=channels, shifts=shifts)
        for bs in [1, 4, 16]:
            x = tf.random.normal([bs, 1, 16, channels])
            out = layer(x)
            assert out.shape[0] == bs

    # ---- Causality (CRITICAL) ---------------------------------------------

    def test_causality_future_does_not_affect_past(self, channels, shifts):
        """Changing a future token must not alter any earlier position's output.

        Uses layer_scale_init=1.0 to make block output significant (not
        dominated by the residual skip connection).
        """
        layer = CausalCliffordNetBlock(
            channels=channels, shifts=shifts,
            layer_scale_init=1.0, drop_path_rate=0.0,
        )
        x1 = tf.random.normal([1, 1, 16, channels], seed=0)
        x2 = tf.identity(x1).numpy()
        # Change last position only
        x2[0, 0, -1, :] = tf.random.normal([channels], seed=99).numpy()
        x2 = tf.constant(x2)

        out1 = layer(x1, training=False)
        out2 = layer(x2, training=False)

        # All positions except the last must be identical
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(out1[0, 0, :-1]),
            keras.ops.convert_to_numpy(out2[0, 0, :-1]),
            atol=1e-6,
            err_msg="Future position change affected earlier positions (causality violation)",
        )
        # Last position should differ
        assert not np.allclose(
            keras.ops.convert_to_numpy(out1[0, 0, -1]),
            keras.ops.convert_to_numpy(out2[0, 0, -1]),
            atol=1e-3,
        )

    def test_causality_middle_change_no_backward_leak(self, channels, shifts):
        """Changing a middle position must not affect any earlier position."""
        layer = CausalCliffordNetBlock(
            channels=channels, shifts=shifts,
            layer_scale_init=1.0, drop_path_rate=0.0,
        )
        x1 = tf.random.normal([1, 1, 16, channels], seed=0)
        x2 = tf.identity(x1).numpy()
        change_pos = 8
        x2[0, 0, change_pos, :] = tf.random.normal([channels], seed=99).numpy()
        x2 = tf.constant(x2)

        out1 = layer(x1, training=False)
        out2 = layer(x2, training=False)

        # Positions before change_pos must be identical
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(out1[0, 0, :change_pos]),
            keras.ops.convert_to_numpy(out2[0, 0, :change_pos]),
            atol=1e-6,
            err_msg="Backward leak: earlier positions affected by later change",
        )
        # Position at change_pos should differ
        assert not np.allclose(
            keras.ops.convert_to_numpy(out1[0, 0, change_pos]),
            keras.ops.convert_to_numpy(out2[0, 0, change_pos]),
            atol=1e-3,
        )

    def test_causality_stacked_blocks(self, channels, shifts):
        """Causality holds through multiple stacked blocks."""
        blocks = [
            CausalCliffordNetBlock(
                channels=channels, shifts=shifts,
                layer_scale_init=1.0, name=f"b{i}",
            )
            for i in range(4)
        ]
        x1 = tf.random.normal([1, 1, 16, channels], seed=0)
        x2 = tf.identity(x1).numpy()
        x2[0, 0, -1, :] = tf.random.normal([channels], seed=99).numpy()
        x2 = tf.constant(x2)

        o1, o2 = x1, x2
        for block in blocks:
            o1 = block(o1, training=False)
            o2 = block(o2, training=False)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(o1[0, 0, :-1]),
            keras.ops.convert_to_numpy(o2[0, 0, :-1]),
            atol=1e-5,
            err_msg="Causality violation after stacking 4 blocks",
        )

    def test_causality_with_global_context(self, channels, shifts):
        """Global context branch uses causal cumulative mean, not full mean."""
        layer = CausalCliffordNetBlock(
            channels=channels, shifts=shifts,
            use_global_context=True,
            layer_scale_init=1.0, drop_path_rate=0.0,
        )
        x1 = tf.random.normal([1, 1, 16, channels], seed=0)
        x2 = tf.identity(x1).numpy()
        x2[0, 0, -1, :] = tf.random.normal([channels], seed=99).numpy()
        x2 = tf.constant(x2)

        out1 = layer(x1, training=False)
        out2 = layer(x2, training=False)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(out1[0, 0, :-1]),
            keras.ops.convert_to_numpy(out2[0, 0, :-1]),
            atol=1e-5,
            err_msg="Global context branch leaks future info (causality violation)",
        )
        assert not np.allclose(
            keras.ops.convert_to_numpy(out1[0, 0, -1]),
            keras.ops.convert_to_numpy(out2[0, 0, -1]),
            atol=1e-3,
        )

    # ---- Functional -------------------------------------------------------

    def test_ctx_mode_diff_vs_abs_differ(self, channels, shifts):
        x = tf.random.normal([2, 1, 16, channels], seed=42)
        layer_diff = CausalCliffordNetBlock(
            channels=channels, shifts=shifts, ctx_mode="diff", layer_scale_init=1.0,
        )
        layer_abs = CausalCliffordNetBlock(
            channels=channels, shifts=shifts, ctx_mode="abs", layer_scale_init=1.0,
        )
        out_diff = layer_diff(x, training=False).numpy()
        out_abs = layer_abs(x, training=False).numpy()
        assert not np.allclose(out_diff, out_abs, atol=1e-3)

    def test_residual_connection_identity_at_init(self, channels, shifts):
        layer = CausalCliffordNetBlock(
            channels=channels, shifts=shifts, layer_scale_init=1e-10,
        )
        x = tf.random.normal([2, 1, 8, channels])
        out = layer(x)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(out),
            keras.ops.convert_to_numpy(x),
            rtol=1e-4, atol=1e-4,
            err_msg="With gamma ~ 0, output should be close to input",
        )

    def test_all_cli_modes(self, channels, shifts, seq_tensor):
        for mode in ("inner", "wedge", "full"):
            layer = CausalCliffordNetBlock(
                channels=channels, shifts=shifts, cli_mode=mode,
            )
            out = layer(seq_tensor)
            assert out.shape == seq_tensor.shape
            assert not np.any(np.isnan(out.numpy())), f"NaN in cli_mode={mode}"

    def test_different_shift_sets(self, channels):
        x = tf.random.normal([2, 1, 16, channels])
        for shifts in [[1], [1, 2], [1, 2, 4, 8]]:
            layer = CausalCliffordNetBlock(channels=channels, shifts=shifts)
            out = layer(x)
            assert out.shape == x.shape

    def test_numerical_stability(self, channels, shifts):
        layer = CausalCliffordNetBlock(channels=channels, shifts=shifts)
        for scale in [1e-8, 1e8]:
            x = tf.ones([2, 1, 8, channels]) * scale
            out = layer(x)
            assert not np.any(np.isnan(out.numpy())), f"NaN at scale {scale}"
            assert not np.any(np.isinf(out.numpy())), f"Inf at scale {scale}"

    # ---- Determinism & Training -------------------------------------------

    def test_inference_deterministic(self, channels, shifts, seq_tensor):
        layer = CausalCliffordNetBlock(
            channels=channels, shifts=shifts, drop_path_rate=0.0,
        )
        out1 = layer(seq_tensor, training=False)
        out2 = layer(seq_tensor, training=False)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(out1),
            keras.ops.convert_to_numpy(out2),
            rtol=1e-6, atol=1e-6,
        )

    def test_gradient_flow(self, channels, shifts):
        layer = CausalCliffordNetBlock(channels=channels, shifts=shifts)
        x = tf.Variable(tf.random.normal([2, 1, 8, channels]))
        with tf.GradientTape() as tape:
            out = layer(x, training=True)
            loss = tf.reduce_mean(tf.square(out))
        grads = tape.gradient(loss, x)
        assert grads is not None
        assert np.any(grads.numpy() != 0)

    # ---- Serialization ----------------------------------------------------

    def test_serialization(self, channels, shifts):
        original = CausalCliffordNetBlock(
            channels=channels, shifts=shifts,
            cli_mode="wedge", ctx_mode="abs",
            use_global_context=True, layer_scale_init=1e-3,
            drop_path_rate=0.1, name="causal_cb_s",
        )
        config = original.get_config()
        restored = CausalCliffordNetBlock.from_config(config)

        assert restored.channels == original.channels
        assert restored.shifts == original.shifts
        assert restored.cli_mode == original.cli_mode
        assert restored.ctx_mode == original.ctx_mode
        assert restored.use_global_context == original.use_global_context
        assert restored.layer_scale_init == original.layer_scale_init
        assert restored.drop_path_rate == original.drop_path_rate

    def test_model_save_load(self, channels, shifts):
        x = tf.random.normal([2, 1, 16, channels])

        inp = keras.Input(shape=(1, 16, channels))
        out = CausalCliffordNetBlock(
            channels=channels, shifts=shifts, name="causal_cb",
        )(inp)
        model = keras.Model(inputs=inp, outputs=out)

        original_pred = model.predict(x, verbose=0)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.keras")
            model.save(path)
            loaded = keras.models.load_model(path)

        loaded_pred = loaded.predict(x, verbose=0)
        np.testing.assert_allclose(
            original_pred, loaded_pred, rtol=1e-5, atol=1e-5,
            err_msg="Predictions should match after save/load",
        )

    def test_stacking_multiple_blocks(self, channels, shifts):
        x = tf.random.normal([2, 1, 16, channels])
        blocks = [
            CausalCliffordNetBlock(
                channels=channels, shifts=shifts, name=f"cb{i}",
            )
            for i in range(4)
        ]
        for block in blocks:
            x = block(x)
        assert x.shape == (2, 1, 16, channels)
        assert not np.any(np.isnan(x.numpy()))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])