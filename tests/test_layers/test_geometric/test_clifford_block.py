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

    def test_initialization_custom(self, channels):
        """Test initialization with custom parameters."""
        layer = GatedGeometricResidual(
            channels=channels,
            layer_scale_init=1e-3,
            name="custom_ggr",
        )
        assert layer.layer_scale_init == 1e-3
        assert layer.name == "custom_ggr"

    def test_invalid_channels(self):
        """Test that non-positive channels raises ValueError."""
        with pytest.raises(ValueError, match="channels"):
            GatedGeometricResidual(channels=-1)

    # Stochastic depth moved to model level (plan_2026-07-03_eb53492e): GGR no
    # longer owns StochasticDepth, so its rate-validation test is removed.

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

    # Stochastic depth moved to model level (plan_2026-07-03_eb53492e): GGR no
    # longer constructs StochasticDepth, so the present/absent probes are removed.

    def test_inference_vs_training_no_droppath(self, channels, input_tensor):
        """GGR is deterministic: training and inference outputs are identical."""
        layer = GatedGeometricResidual(channels=channels)
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
            channels=channels, layer_scale_init=1e-3, name="ggr_s"
        )
        config = original.get_config()
        restored = GatedGeometricResidual.from_config(config)

        assert restored.channels == original.channels
        assert restored.layer_scale_init == original.layer_scale_init

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

    def test_initialization_custom(self, channels, shifts):
        """Test initialization with custom parameters."""
        layer = CliffordNetBlock(
            channels=channels,
            shifts=shifts,
            cli_mode="inner",
            ctx_mode="abs",
            use_global_context=True,
            layer_scale_init=1e-3,
            name="custom_cb",
        )
        assert layer.cli_mode == "inner"
        assert layer.ctx_mode == "abs"
        assert layer.use_global_context is True
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
        assert layer_instance.ctx_norm.built is True
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

    def test_transform_only_and_external_residual_at_init(self, channels, shifts):
        """Transform-only contract (plan_2026-07-03_eb53492e): with gamma ~ 0
        the block returns the transform (~0), NOT the input; the residual is now
        external, so ``x + block(x) ≈ x`` while ``block(x)`` is NOT ≈ x."""
        layer = CliffordNetBlock(
            channels=channels, shifts=shifts, layer_scale_init=1e-10
        )
        x = tf.random.normal([2, 4, 4, channels])
        h_mix = layer(x)
        # External residual reconstructs the identity.
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(x + h_mix),
            keras.ops.convert_to_numpy(x),
            rtol=1e-4, atol=1e-4,
            err_msg="x + block(x) should ≈ x (external residual) at gamma ~ 0",
        )
        # The block output alone is the transform (~0), NOT the input.
        assert not np.allclose(
            keras.ops.convert_to_numpy(h_mix),
            keras.ops.convert_to_numpy(x),
            atol=1e-4,
        ), "block(x) must be transform-only (≈0), no internal residual"

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
        layer = CliffordNetBlock(channels=channels, shifts=shifts)
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
        assert layer_instance.ctx_norm.built is True
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
            layer_scale_init=1.0,
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
            layer_scale_init=1.0,
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
            layer_scale_init=1.0,
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

    def test_transform_only_and_external_residual_at_init(self, channels, shifts):
        """Transform-only contract (plan_2026-07-03_eb53492e): with gamma ~ 0 the
        block returns the transform (~0), and the residual is external, so
        ``x + block(x) ≈ x`` while ``block(x)`` is NOT ≈ x."""
        layer = CausalCliffordNetBlock(
            channels=channels, shifts=shifts, layer_scale_init=1e-10,
        )
        x = tf.random.normal([2, 1, 8, channels])
        h_mix = layer(x)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(x + h_mix),
            keras.ops.convert_to_numpy(x),
            rtol=1e-4, atol=1e-4,
            err_msg="x + block(x) should ≈ x (external residual) at gamma ~ 0",
        )
        assert not np.allclose(
            keras.ops.convert_to_numpy(h_mix),
            keras.ops.convert_to_numpy(x),
            atol=1e-4,
        ), "block(x) must be transform-only (≈0), no internal residual"

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
            channels=channels, shifts=shifts,
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
            name="causal_cb_s",
        )
        config = original.get_config()
        restored = CausalCliffordNetBlock.from_config(config)

        assert restored.channels == original.channels
        assert restored.shifts == original.shifts
        assert restored.cli_mode == original.cli_mode
        assert restored.ctx_mode == original.ctx_mode
        assert restored.use_global_context == original.use_global_context
        assert restored.layer_scale_init == original.layer_scale_init

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


# ===========================================================================
# Regression tests for issues identified in the deep review
# (plans/plan_2026-05-05_0eac2c81)
# ===========================================================================


class TestReviewRegressions:
    """Regression coverage for fixes applied in iter-1 of the deep-review plan."""

    # --- B4: SRGP shift validation ---

    def test_srgp_rejects_shift_zero(self):
        with pytest.raises(ValueError, match="shifts"):
            SparseRollingGeometricProduct(channels=8, shifts=[0, 1])

    def test_srgp_rejects_negative_shift(self):
        with pytest.raises(ValueError, match="shifts"):
            SparseRollingGeometricProduct(channels=8, shifts=[-1, 1])

    def test_srgp_rejects_bool_shift(self):
        # bool is a subclass of int — explicit reject so True/False
        # don't silently behave as shifts of 1 / 0.
        with pytest.raises(ValueError, match="shifts"):
            SparseRollingGeometricProduct(channels=8, shifts=[True, 1])

    # --- B7: CliffordNetBlock channel validation ---

    def test_cliffordnet_block_rejects_channel_mismatch(self):
        block = CliffordNetBlock(channels=16, shifts=[1, 2])
        x = tf.random.normal([1, 8, 8, 8])  # D=8 != channels=16
        with pytest.raises(ValueError, match="isotropic"):
            block(x)

    # --- B8: global branch needs channels > max(_GLOBAL_SHIFTS) ---
    #
    # The bound is DERIVED from the hardcoded global shifts ([1, 2]), not a
    # literal: SparseRollingGeometricProduct drops any shift s >= channels, so
    # channels=2 would silently build a global branch with shifts=[1] only.
    # The block rejects that up front, hence >= 3 rather than the historical
    # >= 2 this test used to assert.

    def test_cliffordnet_block_global_context_requires_channels_gt_max_shift(self):
        for bad in (1, 2):
            with pytest.raises(ValueError, match="channels >= 3"):
                CliffordNetBlock(
                    channels=bad, shifts=[1], use_global_context=True
                )
        # channels=3 is the smallest width that keeps both global shifts.
        CliffordNetBlock(channels=3, shifts=[1], use_global_context=True)

    def test_causal_cliffordnet_block_global_context_requires_channels_gt_max_shift(
        self,
    ):
        with pytest.raises(ValueError, match="channels >= 3"):
            CausalCliffordNetBlock(channels=1, shifts=[1], use_global_context=True)

    # --- B11: causality regression test (currently passes, lock it in) ---

    def test_causal_cliffordnet_block_no_future_leakage(self):
        """Modifying a future position must not change earlier outputs.

        layer_scale_init=1.0 makes the transform-only block output significant
        (plan_2026-07-03_eb53492e): the internal residual that used to carry the
        raw perturbation is gone, so the transform itself must respond at the
        perturbed position while earlier positions stay byte-identical.
        """
        keras.utils.set_random_seed(7)
        block = CausalCliffordNetBlock(
            channels=8, shifts=[1, 2], layer_scale_init=1.0,
        )
        seq = 12

        x1 = tf.random.normal([1, 1, seq, 8], seed=0)
        out1 = block(x1, training=False).numpy()

        x2 = x1.numpy().copy()
        x2[0, 0, seq - 1, :] = 9999.0  # perturb only the LAST position
        out2 = block(tf.constant(x2), training=False).numpy()

        diff = np.abs(out1 - out2)
        per_pos_max = diff.max(axis=(0, 1, 3))
        # Earlier positions (0..seq-2) must be byte-identical (no leakage).
        assert (per_pos_max[:-1] < 1e-5).all(), (
            f"Future leak detected — earlier-position deltas: {per_pos_max[:-1]}"
        )
        # The last position SHOULD change (otherwise the block ignores its own input).
        assert per_pos_max[-1] > 1e-3, (
            "Last position unchanged after large perturbation — block dead?"
        )


# ===========================================================================
# TestCliffordNetBlockConfigurableNorm — ctx-norm factory routing
# ===========================================================================


class TestCliffordNetBlockConfigurableNorm:
    """Verify that the context-stream normalization layer of
    :class:`CliffordNetBlock` and :class:`CausalCliffordNetBlock` is built
    through ``create_normalization_layer`` and is configurable via the new
    ``normalization_type`` / ``normalization_kwargs`` parameters.
    """

    @pytest.fixture
    def channels(self) -> int:
        return 16

    @pytest.fixture
    def shifts(self):
        return [1, 2]

    # --- CliffordNetBlock ------------------------------------------------

    def test_clifford_default_norm_is_batch_norm(self, channels, shifts):
        # Default flipped to "batch_norm" to match the original CliffordBlock's
        # BatchNormalization; the bias-free denoiser overrides to a degree-0 norm
        # ("zero_centered_rms_norm"). CausalCliffordNetBlock default is unchanged.
        block = CliffordNetBlock(channels=channels, shifts=shifts)
        block.build((None, 8, 8, channels))
        assert type(block.ctx_norm).__name__ == "BatchNormalization"
        assert block.normalization_type == "batch_norm"

    def test_clifford_alternative_norm_selectable(self, channels, shifts):
        block = CliffordNetBlock(
            channels=channels, shifts=shifts, normalization_type="layer_norm",
        )
        block.build((None, 8, 8, channels))
        assert isinstance(block.ctx_norm, keras.layers.LayerNormalization)
        # smoke forward
        x = tf.random.normal([2, 8, 8, channels])
        y = block(x, training=False)
        assert y.shape == x.shape
        assert not np.any(np.isnan(y.numpy()))

    def test_clifford_serialization_round_trip(self, channels, shifts):
        block = CliffordNetBlock(
            channels=channels, shifts=shifts, normalization_type="rms_norm",
        )
        block.build((None, 8, 8, channels))
        # Drive once so weights exist.
        _ = block(tf.random.normal([1, 8, 8, channels]), training=False)
        inputs = keras.Input(shape=(8, 8, channels))
        outputs = block(inputs)
        model = keras.Model(inputs, outputs)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "block.keras")
            model.save(path)
            restored = keras.models.load_model(path)
        # The restored block is a sub-layer of the model.
        restored_block = None
        for layer in restored.layers:
            if isinstance(layer, CliffordNetBlock):
                restored_block = layer
                break
        assert restored_block is not None
        assert restored_block.normalization_type == "rms_norm"

    # --- CausalCliffordNetBlock -----------------------------------------

    def test_causal_default_norm_is_zero_centered_rms(self, channels, shifts):
        block = CausalCliffordNetBlock(channels=channels, shifts=shifts)
        block.build((None, 1, 16, channels))
        assert type(block.ctx_norm).__name__ == "ZeroCenteredRMSNorm"
        assert block.normalization_type == "zero_centered_rms_norm"

    def test_causal_alternative_norm_selectable(self, channels, shifts):
        block = CausalCliffordNetBlock(
            channels=channels, shifts=shifts, normalization_type="layer_norm",
        )
        block.build((None, 1, 16, channels))
        assert isinstance(block.ctx_norm, keras.layers.LayerNormalization)
        x = tf.random.normal([2, 1, 16, channels])
        y = block(x, training=False)
        assert y.shape == x.shape
        assert not np.any(np.isnan(y.numpy()))

    def test_causal_serialization_round_trip(self, channels, shifts):
        block = CausalCliffordNetBlock(
            channels=channels, shifts=shifts, normalization_type="rms_norm",
        )
        inputs = keras.Input(shape=(1, 16, channels))
        outputs = block(inputs)
        model = keras.Model(inputs, outputs)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "causal_block.keras")
            model.save(path)
            restored = keras.models.load_model(path)
        restored_block = None
        for layer in restored.layers:
            if isinstance(layer, CausalCliffordNetBlock):
                restored_block = layer
                break
        assert restored_block is not None
        assert restored_block.normalization_type == "rms_norm"


# ===========================================================================
# TestInertGateNotTrainable
# ===========================================================================


class TestInertGateNotTrainable:
    """Regression coverage for DECISION plan-2026-07-22T090932-e433f233/D-001.

    When ``use_gate=False`` the GGR gate is inert (never referenced in
    ``call()``), so its kernel/bias get no gradient and Keras emits a
    "Gradients do not exist for variables [...]" UserWarning once per run.
    The fix marks the inert sub-layer non-trainable so those variables leave
    ``trainable_variables`` -- while REMAINING in ``weights`` so the ``.keras``
    layout is unchanged.
    """

    @staticmethod
    def _gate_vars(obj, attr):
        return [v for v in getattr(obj, attr) if "gate_dense" in v.path]

    def test_inert_gate_is_not_trainable(self):
        ggr = GatedGeometricResidual(channels=8, use_gate=False)
        ggr.build((None, 4, 4, 8))
        assert ggr.gate_dense.trainable is False
        assert self._gate_vars(ggr, "trainable_variables") == []

    def test_inert_gate_weights_are_still_saved(self):
        """The whole point of the fix: trainable=False, but still in weights."""
        ggr = GatedGeometricResidual(channels=8, use_gate=False)
        ggr.build((None, 4, 4, 8))
        # kernel + bias must still be present for .keras layout stability.
        assert len(self._gate_vars(ggr, "weights")) == 2

    def test_live_gate_remains_trainable(self):
        """use_gate=True consumers must be completely unaffected."""
        ggr = GatedGeometricResidual(channels=8, use_gate=True)
        ggr.build((None, 4, 4, 8))
        assert ggr.gate_dense.trainable is True
        assert len(self._gate_vars(ggr, "trainable_variables")) == 2

    def test_no_missing_gradient_warning_on_fit(self):
        """End-to-end: a training step must not emit the optimizer warning."""
        import warnings

        inputs = keras.Input(shape=(8, 8, 8))
        block = CliffordNetBlock(
            channels=8, shifts=[1, 2], use_gate=False, use_bias=False,
        )
        model = keras.Model(inputs, block(inputs))
        model.compile(optimizer=keras.optimizers.AdamW(1e-4), loss="mse")

        x = np.random.RandomState(0).randn(2, 8, 8, 8).astype("float32")
        y = np.random.RandomState(1).randn(2, 8, 8, 8).astype("float32")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(x, y, epochs=1, verbose=0)

        missing = [
            str(w.message)
            for w in caught
            if "Gradients do not exist" in str(w.message)
        ]
        assert not missing, f"unexpected missing-gradient warning: {missing}"

    def test_flag_survives_keras_round_trip(self):
        """__init__ re-runs on load, so the flag must be re-applied."""
        inputs = keras.Input(shape=(8, 8, 8))
        block = CliffordNetBlock(
            channels=8, shifts=[1, 2], use_gate=False, use_bias=False,
        )
        model = keras.Model(inputs, block(inputs))
        n_weights = len(model.weights)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "inert_gate.keras")
            model.save(path)
            restored = keras.models.load_model(path)

        # Layout preserved...
        assert len(restored.weights) == n_weights
        assert len(self._gate_vars(restored, "weights")) == 2
        # ...and the gate is still excluded from the optimizer's view.
        assert self._gate_vars(restored, "trainable_variables") == []


# ===========================================================================
# TestUseGateCheckpointLayoutInvariant
# ===========================================================================


class TestUseGateCheckpointLayoutInvariant:
    """SC-3: the `use_gate` flag must be checkpoint-NEUTRAL.

    Landed from a review-transcript probe (plan-2026-08-10-3649c19e,
    findings/review-iter-1.md concern 8, iteration-2 priority 5). The invariant
    was verified once by hand and then existed only in that transcript; this
    class makes it executable.

    The contract, in three parts:

    1. ``use_gate=True`` and ``use_gate=False`` produce IDENTICAL weight paths
       and counts (the inert-sublayer pattern, `plans/SYSTEM.md` § Known
       Patterns). Building ``gate_dense`` conditionally would satisfy every
       shape/forward test in this file while silently changing the ``.keras``
       weight layout across the flag, i.e. breaking every existing checkpoint.
    2. ``trainable_variables`` DOES differ (3 vs 1) — that asymmetry is the
       point: the weights are saved but the optimizer never sees them.
    3. A ``.keras`` round-trip with RANDOMIZED weights preserves both the
       weight VALUES and the output VALUES at both flag settings.

    Randomizing every weight before saving is load-bearing, not decoration: a
    freshly-built layer whose weights are silently re-initialized on load can
    round-trip "correctly" because both draws come from the same seeded RNG.
    This repo has already been bitten by exactly that (nested sub-layer weight
    loss with matching counts AND paths).
    """

    CHANNELS = 8
    EXPECTED_PATHS = [
        "ggr/gamma",
        "ggr/gate_dense/bias",
        "ggr/gate_dense/kernel",
    ]

    @staticmethod
    def _build(use_gate: bool):
        """A functional model wrapping one GGR named ``ggr``."""
        inp_a = keras.Input(shape=(4, 4, 8))
        inp_b = keras.Input(shape=(4, 4, 8))
        ggr = GatedGeometricResidual(
            channels=8, use_gate=use_gate, name="ggr"
        )
        return keras.Model([inp_a, inp_b], ggr(inp_a, inp_b))

    @staticmethod
    def _randomize(model, seed):
        """Assign a non-default value to EVERY weight."""
        rng = np.random.RandomState(seed)
        for w in model.weights:
            w.assign(rng.normal(size=w.shape).astype("float32") * 0.1)

    # ------------------------------------------------------------------

    def test_weight_paths_and_counts_are_identical_across_the_flag(self):
        """The measured invariant: 3 weights, same 3 paths, both settings."""
        paths = {
            ug: sorted(w.path for w in self._build(ug).weights)
            for ug in (True, False)
        }
        assert paths[True] == self.EXPECTED_PATHS, paths[True]
        assert paths[False] == paths[True], (
            "`.keras` weight layout differs across use_gate: "
            f"True={paths[True]} False={paths[False]}. gate_dense must be "
            "built UNCONDITIONALLY (inert-sublayer pattern) or every existing "
            "checkpoint carrying the other flag value breaks."
        )
        assert len(paths[False]) == 3

    def test_trainable_surface_differs_but_weights_do_not(self):
        """3 trainable at True, exactly ``ggr/gamma`` at False."""
        trainable = {
            ug: sorted(w.path for w in self._build(ug).trainable_variables)
            for ug in (True, False)
        }
        assert trainable[True] == self.EXPECTED_PATHS
        assert trainable[False] == ["ggr/gamma"], trainable[False]

    @pytest.mark.parametrize("use_gate", [True, False])
    def test_randomized_round_trip_is_value_exact(self, use_gate):
        """max|dW| == 0 and max|dY| == 0 after a `.keras` round-trip.

        ``training=False`` is passed EXPLICITLY on both forward calls:
        ``training=None`` is not inference in this repo.
        """
        rng = np.random.RandomState(0)
        xa = rng.normal(size=(2, 4, 4, 8)).astype("float32")
        xb = rng.normal(size=(2, 4, 4, 8)).astype("float32")

        model = self._build(use_gate)
        self._randomize(model, seed=1 if use_gate else 2)
        before = keras.ops.convert_to_numpy(model([xa, xb], training=False))
        # Anti-vacuity: a degenerate all-zero output would make any dY == 0.
        assert np.abs(before).max() > 0.0

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ggr.keras")
            model.save(path)
            restored = keras.models.load_model(path)
        after = keras.ops.convert_to_numpy(restored([xa, xb], training=False))

        assert len(restored.weights) == len(model.weights)
        dw = max(
            float(
                np.max(
                    np.abs(
                        keras.ops.convert_to_numpy(a)
                        - keras.ops.convert_to_numpy(b)
                    )
                )
            )
            for a, b in zip(model.weights, restored.weights)
        )
        assert dw == 0.0, (
            f"max|dW| = {dw} after a .keras round-trip at use_gate={use_gate}; "
            "the reloaded layer is not carrying the SAVED weights"
        )
        dy = float(np.max(np.abs(before - after)))
        assert dy == 0.0, f"max|dY| = {dy} at use_gate={use_gate}"

    @pytest.mark.parametrize("use_gate", [True, False])
    def test_use_gate_survives_get_config(self, use_gate):
        """The flag must round-trip, or the restored layer changes semantics."""
        model = self._build(use_gate)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ggr.keras")
            model.save(path)
            restored = keras.models.load_model(path)
        cfg = restored.get_layer("ggr").get_config()
        assert cfg["use_gate"] is use_gate, cfg


# ===========================================================================
# TestSequenceModeAndCausalNormSafety
# ===========================================================================


class TestSequenceModeAndCausalNormSafety:
    """Coverage for plan-2026-08-11-54118fdd steps 1-2 (D-001, D-002).

    Two behaviours land here:

    * ``input_mode="sequence"`` — a native rank-3 ``(B, L, D)`` contract on top
      of an UNCHANGED internal 4-D body, so the ``.keras`` weight layout is
      untouched (D-001).
    * causal context-norm safety — the sequence-mode default norm became
      ``"zero_centered_rms_norm"``, and an EXPLICIT sequence-axis-reducing norm
      under ``causal=True`` now raises (D-002, finding F6).

    .. warning::

       Every causality probe in this class perturbs the future with FRESH
       NON-DC noise, never with a constant offset. A DC perturbation measured
       **1.9e-06** against a real leak of **1.067** on the very same code: the
       input ``LayerNormalization`` removes a per-position DC shift before the
       context stream ever sees it, so a DC probe is VACUOUS and would pass
       against the unfixed layer. Each probe additionally asserts that the
       perturbed region ITSELF moved, so a dead block cannot fake a green.
    """

    CHANNELS = 16
    SHIFTS = [1, 2]
    SEQ = 12

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @classmethod
    def _future_probe(cls, rank: int, seed: int, split: int):
        """A pair of inputs identical up to ``split``, NON-DC noise after it.

        :param rank: 3 for ``(B, L, D)``, 4 for ``(B, 1, L, D)``.
        :param seed: RandomState seed.
        :param split: First perturbed position; ``0..split-1`` are the "past".
        :return: ``(x1, x2)`` float32 arrays of the requested rank.
        """
        rng = np.random.RandomState(seed)
        shape = (
            (4, cls.SEQ, cls.CHANNELS)
            if rank == 3
            else (4, 1, cls.SEQ, cls.CHANNELS)
        )
        x1 = rng.normal(size=shape).astype("float32")
        x2 = x1.copy()
        # Fresh independent draw per (position, channel): a per-position DC
        # offset would be normalized away and prove nothing.
        tail = rng.normal(size=x2[..., split:, :].shape).astype("float32") * 3.0
        x2[..., split:, :] = tail
        return x1, x2

    @staticmethod
    def _per_position_delta(out1, out2):
        """max |delta| per sequence position, reduced over batch and channel."""
        diff = np.abs(
            keras.ops.convert_to_numpy(out1) - keras.ops.convert_to_numpy(out2)
        )
        # Sequence axis is the SECOND-TO-LAST axis at both ranks (D-001 I-3).
        axes = tuple(i for i in range(diff.ndim) if i != diff.ndim - 2)
        return diff.max(axis=axes)

    # ------------------------------------------------------------------
    # T1 — F6 regression: the causal DEFAULT no longer leaks
    # ------------------------------------------------------------------

    def test_causal_default_norm_has_no_future_leak(self):
        """``CliffordNetBlock(causal=True)`` at its DEFAULT norm leaks 0.

        Before D-002 the BASE class kept ``normalization_type="batch_norm"``
        even under ``causal=True``; ``batch_norm`` reduces over ``(B, H, W)``
        and on the causal path W IS the sequence axis, so at ``training=True``
        every position's context normalization saw the whole sequence. Measured
        leak on this exact probe shape: **0.3302** on a 3.393-scale signal
        before the fix, **0.0** after.
        """
        keras.utils.set_random_seed(7)
        block = CliffordNetBlock(
            channels=self.CHANNELS,
            shifts=self.SHIFTS,
            causal=True,
            layer_scale_init=1.0,
        )
        split = 8
        x1, x2 = self._future_probe(rank=4, seed=11, split=split)

        out1 = block(x1, training=True)
        out2 = block(x2, training=True)
        per_pos = self._per_position_delta(out1, out2)

        # Non-degeneracy FIRST: if the perturbation did nothing, the causality
        # assertion below is vacuous.
        assert per_pos[split:].max() > 1e-3, (
            "the perturbed FUTURE region did not move "
            f"(max |delta| {per_pos[split:].max()}) — the probe is degenerate "
            "and the causality assertion below would pass against any layer"
        )
        assert per_pos[:split].max() == 0.0, (
            "future leaked into the past at the causal DEFAULT norm: "
            f"per-position past deltas {per_pos[:split]} "
            f"(normalization_type={block.normalization_type!r})"
        )

    # ------------------------------------------------------------------
    # T2 — the explicit unsafe-norm raise, positives AND negatives
    # ------------------------------------------------------------------

    UNSAFE_NORMS = ["batch_norm", "bias_free_batch_norm", "global_response_norm"]

    @pytest.mark.parametrize("norm_type", UNSAFE_NORMS)
    def test_unsafe_norm_raises_under_causal(self, norm_type):
        """All three measured sequence-axis-reducing types are rejected."""
        with pytest.raises(ValueError, match="sequence axis"):
            CliffordNetBlock(
                channels=self.CHANNELS,
                shifts=self.SHIFTS,
                causal=True,
                normalization_type=norm_type,
            )

    @pytest.mark.parametrize("norm_type", UNSAFE_NORMS)
    def test_unsafe_norm_accepted_in_image_mode(self, norm_type):
        """NEGATIVE case: image mode is allowed to reduce over space.

        ``batch_norm`` is the image-mode DEFAULT and is load-bearing for the
        ``xfail(strict=True)`` at
        ``test_video_jepa.py::test_predictor_graph_mode_dropout_zero``. A raise
        that fired here would break that suite.
        """
        block = CliffordNetBlock(
            channels=self.CHANNELS,
            shifts=self.SHIFTS,
            normalization_type=norm_type,
        )
        assert block.normalization_type == norm_type
        assert block.input_mode == "image"

    @pytest.mark.parametrize("norm_type", UNSAFE_NORMS)
    def test_unsafe_norm_accepted_in_noncausal_sequence(self, norm_type):
        """NEGATIVE case: a BIDIRECTIONAL encoder may mix across positions."""
        block = CliffordNetBlock(
            channels=self.CHANNELS,
            shifts=self.SHIFTS,
            input_mode="sequence",
            normalization_type=norm_type,
        )
        assert block.normalization_type == norm_type
        assert block.causal is False

    def test_resolved_default_is_accepted_under_causal(self):
        """NEGATIVE case: the raise must be reachable only from a caller value.

        The mode-derived default resolves to ``"zero_centered_rms_norm"``, so a
        raise keyed on the RESOLVED type instead of the SUPPLIED one would make
        ``CliffordNetBlock(causal=True)`` itself unconstructible.
        """
        block = CliffordNetBlock(
            channels=self.CHANNELS, shifts=self.SHIFTS, causal=True,
        )
        assert block.normalization_type == "zero_centered_rms_norm"
        assert block.input_mode == "sequence"
        # The subclass must be unaffected too (invariant I-5).
        assert (
            CausalCliffordNetBlock(
                channels=self.CHANNELS, shifts=self.SHIFTS,
            ).normalization_type
            == "zero_centered_rms_norm"
        )

    # ------------------------------------------------------------------
    # T6a — rank-3 in, rank-3 out
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("factory_name", ["explicit_sequence", "causal_subclass"])
    def test_rank3_in_rank3_out(self, factory_name):
        """A rank-3 ``(B, L, D)`` input returns a rank-3 output.

        The internal representation stays 4-D (D-001 / invariant I-2); the H=1
        axis is added and removed inside ``call()``, so the public rank is
        preserved and ``compute_output_shape`` agrees.
        """
        if factory_name == "explicit_sequence":
            block = CliffordNetBlock(
                channels=self.CHANNELS, shifts=self.SHIFTS, input_mode="sequence",
            )
        else:
            block = CausalCliffordNetBlock(
                channels=self.CHANNELS, shifts=self.SHIFTS,
            )
        x = np.random.RandomState(3).normal(
            size=(2, self.SEQ, self.CHANNELS)
        ).astype("float32")

        y = block(x, training=False)
        assert len(y.shape) == 3, f"expected rank-3 output, got {y.shape}"
        assert tuple(y.shape) == x.shape
        assert block.compute_output_shape(x.shape) == x.shape
        assert np.isfinite(keras.ops.convert_to_numpy(y)).all()

    # ------------------------------------------------------------------
    # T6b — causal rank-3 has zero future leakage
    # ------------------------------------------------------------------

    def test_causal_rank3_has_no_future_leakage(self):
        """The rank-3 contract must not lose causality on the way in/out."""
        keras.utils.set_random_seed(13)
        block = CliffordNetBlock(
            channels=self.CHANNELS,
            shifts=self.SHIFTS,
            causal=True,
            layer_scale_init=1.0,
        )
        split = 8
        x1, x2 = self._future_probe(rank=3, seed=21, split=split)

        per_pos = self._per_position_delta(
            block(x1, training=True), block(x2, training=True)
        )
        assert per_pos[split:].max() > 1e-3, (
            f"degenerate probe — future region max |delta| {per_pos[split:].max()}"
        )
        assert per_pos[:split].max() == 0.0, (
            f"future leaked into the past at rank 3: {per_pos[:split]}"
        )

    # ------------------------------------------------------------------
    # T6c — the DISCRIMINATION test
    # ------------------------------------------------------------------

    def test_noncausal_sequence_is_bidirectional_while_causal_is_not(self):
        """Non-causal sequence mode DOES move the past; causal does not.

        This is what proves the zero-leak assertions above are not vacuous: the
        same probe, the same shapes, the same assertion machinery, opposite
        outcomes. Only the DISCRIMINATION is asserted (causal == 0 AND
        non-causal > 0), never an absolute magnitude — at the class default
        ``layer_scale_init=1e-5`` the non-causal delta is ~1e-06 purely because
        the layer scale multiplies the whole output, which says nothing about
        bidirectionality. ``layer_scale_init=1.0`` is therefore used here so the
        measured signal is the mechanism, not the scale.

        The perturbation is at the LAST position and the past is probed at the
        positions immediately before it: the non-causal context is two stacked
        ``(1, 3)`` same-padded depthwise convolutions, i.e. a 5-position
        receptive field, so position 0 is legitimately unreachable from
        position L-1 and asserting on it would be wrong.
        """
        split = self.SEQ - 1
        x1, x2 = self._future_probe(rank=3, seed=31, split=split)

        keras.utils.set_random_seed(5)
        causal = CliffordNetBlock(
            channels=self.CHANNELS,
            shifts=self.SHIFTS,
            causal=True,
            layer_scale_init=1.0,
        )
        keras.utils.set_random_seed(5)
        bidir = CliffordNetBlock(
            channels=self.CHANNELS,
            shifts=self.SHIFTS,
            input_mode="sequence",
            layer_scale_init=1.0,
        )

        causal_pos = self._per_position_delta(
            causal(x1, training=True), causal(x2, training=True)
        )
        bidir_pos = self._per_position_delta(
            bidir(x1, training=True), bidir(x2, training=True)
        )

        # Non-degeneracy: both blocks must react at the perturbed position.
        assert causal_pos[split] > 1e-3 and bidir_pos[split] > 1e-3, (
            f"degenerate probe — causal {causal_pos[split]}, "
            f"non-causal {bidir_pos[split]}"
        )
        assert causal_pos[:split].max() == 0.0, (
            f"causal block leaked backwards: {causal_pos[:split]}"
        )
        assert bidir_pos[:split].max() > 0.0, (
            "non-causal sequence mode did NOT propagate a future change "
            f"backwards ({bidir_pos[:split]}) — it is not bidirectional, which "
            "means the causal assertion above cannot discriminate the two modes"
        )

    # ------------------------------------------------------------------
    # T6d — rank-3 and rank-4 parity at the SAME weights
    # ------------------------------------------------------------------

    def test_rank3_and_rank4_are_identical(self):
        """One layer instance, one weight set, two input ranks, identical out.

        The layer is built on rank 4 and then fed rank 3, which is the ordering
        that catches keying the expand/squeeze off the BUILD rank instead of the
        rank of the CURRENT call (D-001 implementation note b).
        """
        keras.utils.set_random_seed(17)
        block = CliffordNetBlock(
            channels=self.CHANNELS,
            shifts=self.SHIFTS,
            input_mode="sequence",
            layer_scale_init=1.0,
        )
        x3 = np.random.RandomState(5).normal(
            size=(2, self.SEQ, self.CHANNELS)
        ).astype("float32")
        x4 = x3.reshape(2, 1, self.SEQ, self.CHANNELS)

        y4 = keras.ops.convert_to_numpy(block(x4, training=False))
        y3 = keras.ops.convert_to_numpy(block(x3, training=False))

        assert y4.shape == (2, 1, self.SEQ, self.CHANNELS)
        assert y3.shape == (2, self.SEQ, self.CHANNELS)
        # Anti-vacuity: an all-zero output would make any delta 0.
        assert np.abs(y3).max() > 0.0
        delta = float(np.max(np.abs(y3 - y4.reshape(y3.shape))))
        assert delta == 0.0, (
            f"rank-3 and rank-4 outputs differ by {delta} at identical weights"
        )

    # ------------------------------------------------------------------
    # T7 — `.keras` round-trip in sequence mode, value-exact
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("causal", [True, False])
    def test_sequence_round_trip_is_value_exact(self, causal):
        """Save/load a rank-3 sequence model and compare weights AND outputs.

        Follows :class:`TestUseGateCheckpointLayoutInvariant`: EVERY weight is
        randomized before saving, because a layer whose weights are silently
        re-initialized on load round-trips "correctly" when both draws come from
        the same seeded RNG. The resolved norm here
        (``zero_centered_rms_norm``) deliberately owns no variance-like weight,
        so an unsigned-quantity constraint cannot turn the randomization itself
        into a probe artifact.
        """
        kwargs = dict(channels=self.CHANNELS, shifts=self.SHIFTS, name="blk")
        if causal:
            block = CliffordNetBlock(causal=True, **kwargs)
        else:
            block = CliffordNetBlock(input_mode="sequence", **kwargs)
        assert block.normalization_type == "zero_centered_rms_norm"

        inputs = keras.Input(shape=(self.SEQ, self.CHANNELS))
        model = keras.Model(inputs, block(inputs))

        rng = np.random.RandomState(1 if causal else 2)
        for w in model.weights:
            w.assign(rng.normal(size=w.shape).astype("float32") * 0.1)

        x = np.random.RandomState(9).normal(
            size=(2, self.SEQ, self.CHANNELS)
        ).astype("float32")
        before = keras.ops.convert_to_numpy(model(x, training=False))
        assert np.abs(before).max() > 0.0

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "seq_block.keras")
            model.save(path)
            restored = keras.models.load_model(path)
        after = keras.ops.convert_to_numpy(restored(x, training=False))

        assert len(restored.weights) == len(model.weights)
        assert restored.get_layer("blk").input_mode == "sequence"
        dw = max(
            float(
                np.max(
                    np.abs(
                        keras.ops.convert_to_numpy(a)
                        - keras.ops.convert_to_numpy(b)
                    )
                )
            )
            for a, b in zip(model.weights, restored.weights)
        )
        assert dw == 0.0, (
            f"max|dW| = {dw} after a sequence-mode .keras round-trip "
            f"(causal={causal}); the reloaded block is not carrying the SAVED "
            "weights"
        )
        dy = float(np.max(np.abs(before - after)))
        assert dy == 0.0, f"max|dY| = {dy} at causal={causal}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])