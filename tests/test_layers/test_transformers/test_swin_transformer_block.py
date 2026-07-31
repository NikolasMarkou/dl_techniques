import pytest
import numpy as np
import keras
import os
import tempfile

from dl_techniques.layers.ffn.swin_mlp import SwinMLP
from dl_techniques.layers.stochastic_depth import StochasticDepth
from dl_techniques.layers.attention.window_attention import WindowAttention
from dl_techniques.layers.transformers.swin_transformer_block import SwinTransformerBlock


class TestSwinTransformerBlock:
    """Test suite for SwinTransformerBlock implementation."""

    @pytest.fixture
    def input_tensor(self):
        """Create a test input tensor."""
        # Shape: (batch_size, height, width, channels)
        # Using 56x56 which is divisible by common window sizes (7, 8, 14, 28)
        return keras.random.normal([2, 56, 56, 96])

    @pytest.fixture
    def layer_instance(self):
        """Create a default layer instance for testing."""
        return SwinTransformerBlock(dim=96, num_heads=3)

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        layer = SwinTransformerBlock(dim=128, num_heads=4)

        # Check default values
        assert layer.dim == 128
        assert layer.num_heads == 4
        assert layer.window_size == 8
        assert layer.shift_size == 0
        assert layer.mlp_ratio == 4.0
        assert layer.qkv_bias is True
        assert layer.dropout_rate == 0.0
        assert layer.attention_dropout_rate == 0.0
        assert layer.stochastic_depth_rate == 0.0
        # activation stored via keras.activations.get() -> function; assert serialized token
        assert keras.activations.serialize(layer.activation) == "gelu"
        assert layer.use_bias is True
        assert isinstance(layer.kernel_initializer, keras.initializers.GlorotUniform)
        assert isinstance(layer.bias_initializer, keras.initializers.Zeros)
        assert layer.kernel_regularizer is None
        assert layer.bias_regularizer is None

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        custom_regularizer = keras.regularizers.L2(1e-4)

        layer = SwinTransformerBlock(
            dim=64,
            num_heads=8,
            window_size=7,
            shift_size=3,
            mlp_ratio=3.0,
            qkv_bias=False,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            stochastic_depth_rate=0.1,
            activation="relu",
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=custom_regularizer,
            bias_regularizer=custom_regularizer,
        )

        # Check custom values
        assert layer.dim == 64
        assert layer.num_heads == 8
        assert layer.window_size == 7
        assert layer.shift_size == 3
        assert layer.mlp_ratio == 3.0
        assert layer.qkv_bias is False
        assert layer.dropout_rate == 0.1
        assert layer.attention_dropout_rate == 0.1
        assert layer.stochastic_depth_rate == 0.1
        assert keras.activations.serialize(layer.activation) == "relu"
        assert layer.use_bias is False
        assert isinstance(layer.kernel_initializer, keras.initializers.HeNormal)
        assert layer.kernel_regularizer == custom_regularizer
        assert layer.bias_regularizer == custom_regularizer

    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        # Test invalid window_size
        with pytest.raises(ValueError, match="window_size must be positive"):
            SwinTransformerBlock(dim=64, num_heads=8, window_size=0)

        with pytest.raises(ValueError, match="window_size must be positive"):
            SwinTransformerBlock(dim=64, num_heads=8, window_size=-1)

        # Test invalid shift_size
        with pytest.raises(ValueError, match="shift_size must be non-negative"):
            SwinTransformerBlock(dim=64, num_heads=8, shift_size=-1)

        with pytest.raises(ValueError, match="shift_size .* must be less than window_size"):
            SwinTransformerBlock(dim=64, num_heads=8, window_size=8, shift_size=8)

        # Test invalid mlp_ratio
        with pytest.raises(ValueError, match="mlp_ratio must be positive"):
            SwinTransformerBlock(dim=64, num_heads=8, mlp_ratio=0)

        with pytest.raises(ValueError, match="mlp_ratio must be positive"):
            SwinTransformerBlock(dim=64, num_heads=8, mlp_ratio=-1)

    def test_build_process(self, input_tensor, layer_instance):
        """Test that the layer builds properly."""
        # Trigger build through forward pass
        output = layer_instance(input_tensor)

        # Check that the layer was built
        assert layer_instance.built is True
        assert len(layer_instance.weights) > 0
        assert layer_instance.norm1 is not None
        assert layer_instance.norm2 is not None
        assert layer_instance.attn is not None
        assert layer_instance.mlp is not None
        assert isinstance(layer_instance.norm1, keras.layers.LayerNormalization)
        assert isinstance(layer_instance.norm2, keras.layers.LayerNormalization)

        # Check that stochastic depth is created when drop_path > 0
        layer_with_drop_path = SwinTransformerBlock(dim=96, num_heads=3, stochastic_depth_rate=0.1)
        layer_with_drop_path(input_tensor)
        assert layer_with_drop_path.stochastic_depth_rate is not None

    def test_output_shapes(self, input_tensor):
        """Test that output shapes are computed correctly."""
        configs_to_test = [
            {"dim": 32, "num_heads": 4, "window_size": 8},
            {"dim": 64, "num_heads": 8, "window_size": 7},
            {"dim": 128, "num_heads": 8, "window_size": 14},
        ]

        for config in configs_to_test:
            # Create input tensor with matching dimensions and compatible spatial size
            dim = config["dim"]
            window_size = config["window_size"]
            # Use spatial size that's divisible by window_size
            spatial_size = 56  # Divisible by 7, 8, 14, 28
            test_input = keras.random.normal([2, spatial_size, spatial_size, dim])
            layer = SwinTransformerBlock(**config)
            output = layer(test_input)

            # Check output shape matches input shape
            assert output.shape == test_input.shape

            # Test compute_output_shape separately
            computed_shape = layer.compute_output_shape(test_input.shape)
            assert computed_shape == test_input.shape

    def test_forward_pass(self, input_tensor, layer_instance):
        """Test that forward pass produces expected values."""
        output = layer_instance(input_tensor)

        # Basic sanity checks
        assert not np.any(np.isnan(output.numpy()))
        assert not np.any(np.isinf(output.numpy()))
        assert output.shape == input_tensor.shape

        # Test with training=False
        output_inference = layer_instance(input_tensor, training=False)
        assert output_inference.shape == input_tensor.shape

        # Test with training=True
        output_training = layer_instance(input_tensor, training=True)
        assert output_training.shape == input_tensor.shape

    def test_different_configurations(self):
        """Test layer with different configurations."""
        configurations = [
            {"dim": 32, "num_heads": 4, "window_size": 8, "shift_size": 0},
            {"dim": 64, "num_heads": 8, "window_size": 8, "shift_size": 4, "mlp_ratio": 2.0},
            {"dim": 96, "num_heads": 3, "window_size": 7, "shift_size": 3, "dropout_rate": 0.1, "stochastic_depth_rate": 0.1},
            {"dim": 128, "num_heads": 8, "window_size": 14, "shift_size": 0, "qkv_bias": False},
        ]

        for config in configurations:
            layer = SwinTransformerBlock(**config)

            # Create appropriate input with compatible spatial dimensions
            dim = config["dim"]
            window_size = config["window_size"]
            # Use spatial size that's divisible by window_size
            spatial_size = 56  # Divisible by 7, 8, 14, 28
            test_input = keras.random.normal([2, spatial_size, spatial_size, dim])

            # Test forward pass
            output = layer(test_input)

            # Check output is valid
            assert not np.any(np.isnan(output.numpy()))
            assert output.shape == test_input.shape

    def test_shifted_window_attention(self):
        """Test that shifted window attention works correctly."""
        # Test with no shift
        layer_no_shift = SwinTransformerBlock(dim=64, num_heads=8, window_size=8, shift_size=0)
        test_input = keras.random.normal([2, 56, 56, 64])  # 56 is divisible by 8
        output_no_shift = layer_no_shift(test_input)

        # Test with shift
        layer_with_shift = SwinTransformerBlock(dim=64, num_heads=8, window_size=8, shift_size=4)
        output_with_shift = layer_with_shift(test_input)

        # Both should produce valid outputs
        assert not np.any(np.isnan(output_no_shift.numpy()))
        assert not np.any(np.isnan(output_with_shift.numpy()))
        assert output_no_shift.shape == test_input.shape
        assert output_with_shift.shape == test_input.shape

        # Outputs should be different due to different attention patterns
        assert not np.allclose(output_no_shift.numpy(), output_with_shift.numpy())

    def test_serialization(self):
        """Test serialization and deserialization of the layer."""
        original_layer = SwinTransformerBlock(
            dim=128,
            num_heads=8,
            window_size=7,
            shift_size=3,
            mlp_ratio=3.0,
            qkv_bias=False,
            dropout_rate=0.1,
            attention_dropout_rate=0.05,
            stochastic_depth_rate=0.1,
            activation="relu",
            use_bias=False,
            kernel_initializer="he_normal",
        )

        # Build the layer with compatible spatial dimensions
        input_shape = (None, 56, 56, 128)  # 56 is divisible by 7
        original_layer.build(input_shape)

        # Get configs
        config = original_layer.get_config()
        build_config = original_layer.get_build_config()

        # Recreate the layer
        recreated_layer = SwinTransformerBlock.from_config(config)
        recreated_layer.build_from_config(build_config)

        # Check configuration matches
        assert recreated_layer.dim == original_layer.dim
        assert recreated_layer.num_heads == original_layer.num_heads
        assert recreated_layer.window_size == original_layer.window_size
        assert recreated_layer.shift_size == original_layer.shift_size
        assert recreated_layer.mlp_ratio == original_layer.mlp_ratio
        assert recreated_layer.qkv_bias == original_layer.qkv_bias
        assert recreated_layer.dropout_rate == original_layer.dropout_rate
        assert recreated_layer.activation == original_layer.activation
        assert recreated_layer.use_bias == original_layer.use_bias

        # Check weights match (shapes should be the same)
        assert len(recreated_layer.weights) == len(original_layer.weights)
        for w1, w2 in zip(original_layer.weights, recreated_layer.weights):
            assert w1.shape == w2.shape

    def test_model_integration(self, input_tensor):
        """Test the layer in a model context."""
        # Create a simple model with the custom layer
        inputs = keras.Input(shape=input_tensor.shape[1:])
        x = SwinTransformerBlock(dim=96, num_heads=3)(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(10)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Compile the model
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
        )

        # Test forward pass
        y_pred = model(input_tensor, training=False)
        assert y_pred.shape == (input_tensor.shape[0], 10)

    def test_model_save_load(self, input_tensor):
        """Test saving and loading a model with the custom layer."""
        # Create a model with the custom layer
        inputs = keras.Input(shape=input_tensor.shape[1:])
        x = SwinTransformerBlock(dim=96, num_heads=3, name="swin_block")(inputs)
        x = keras.layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(10)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Generate a prediction before saving
        original_prediction = model.predict(input_tensor, verbose=0)

        # Create temporary directory for model
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "model.keras")

            # Save the model
            model.save(model_path)

            # Load the model
            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={
                    "SwinMLP": SwinMLP,
                    "StochasticDepth": StochasticDepth,
                    "WindowAttention": WindowAttention,
                    "SwinTransformerBlock": SwinTransformerBlock,

                }
            )

            # Generate prediction with loaded model
            loaded_prediction = loaded_model.predict(input_tensor, verbose=0)

            # Check predictions match
            assert np.allclose(original_prediction, loaded_prediction, rtol=1e-5)

            # Check layer type is preserved
            assert isinstance(loaded_model.get_layer("swin_block"), SwinTransformerBlock)

    def test_numerical_stability(self):
        """Test layer stability with extreme input values."""
        layer = SwinTransformerBlock(dim=32, num_heads=4, window_size=8)

        # Create inputs with different magnitudes
        batch_size = 2
        height, width = 56, 56  # Divisible by 8
        channels = 32

        test_cases = [
            keras.ops.zeros((batch_size, height, width, channels)),  # Zeros
            keras.ops.ones((batch_size, height, width, channels)) * 1e-10,  # Very small values
            keras.ops.ones((batch_size, height, width, channels)) * 1e5,  # Large values
            keras.random.normal((batch_size, height, width, channels)) * 100  # Large random values
        ]

        for test_input in test_cases:
            output = layer(test_input)

            # Check for NaN/Inf values
            assert not np.any(np.isnan(output.numpy())), "NaN values detected in output"
            assert not np.any(np.isinf(output.numpy())), "Inf values detected in output"

    def test_regularization(self, input_tensor):
        """Test that regularization losses are properly applied."""
        # Create layer with regularization
        layer = SwinTransformerBlock(
            dim=96,
            num_heads=3,
            kernel_regularizer=keras.regularizers.L2(0.1),
            bias_regularizer=keras.regularizers.L1(0.1),
            activity_regularizer=keras.regularizers.L2(0.01)
        )

        # Build layer
        layer.build(input_tensor.shape)

        # No regularization losses before calling the layer
        initial_losses = len(layer.losses)

        # Apply the layer
        _ = layer(input_tensor)

        # Should have regularization losses now
        assert len(layer.losses) >= initial_losses

    def test_training_behavior(self, input_tensor):
        """Test different behavior in training vs inference mode."""
        layer = SwinTransformerBlock(dim=96, num_heads=3, stochastic_depth_rate=0.1, dropout_rate=0.1)

        # Test training mode
        training_output = layer(input_tensor, training=True)

        # Test inference mode
        inference_output = layer(input_tensor, training=False)

        # Both should produce valid outputs
        assert not np.any(np.isnan(training_output.numpy()))
        assert not np.any(np.isnan(inference_output.numpy()))
        assert training_output.shape == input_tensor.shape
        assert inference_output.shape == input_tensor.shape

    def test_stochastic_depth_behavior(self):
        """Test that stochastic depth works correctly."""
        # Layer without stochastic depth
        layer_no_drop = SwinTransformerBlock(dim=64, num_heads=8, window_size=8, stochastic_depth_rate=0.0)
        test_input = keras.random.normal([2, 56, 56, 64])  # 56 is divisible by 8

        # Layer with stochastic depth
        layer_with_drop = SwinTransformerBlock(dim=64, num_heads=8, window_size=8, stochastic_depth_rate=0.5)

        # Test in training mode multiple times to see stochastic behavior
        outputs_with_drop = []
        for _ in range(5):
            output = layer_with_drop(test_input, training=True)
            outputs_with_drop.append(output.numpy())

        # Check that outputs are valid
        for output in outputs_with_drop:
            assert not np.any(np.isnan(output))
            assert not np.any(np.isinf(output))

        # In inference mode, output should be deterministic
        inference_output1 = layer_with_drop(test_input, training=False)
        inference_output2 = layer_with_drop(test_input, training=False)
        assert np.allclose(inference_output1.numpy(), inference_output2.numpy())

    def test_different_input_sizes(self):
        """Test layer with different input sizes."""
        layer = SwinTransformerBlock(dim=64, num_heads=8, window_size=8)

        # Use input sizes that are divisible by window_size (8)
        input_sizes = [
            (2, 16, 16, 64),  # 16 is divisible by 8
            (1, 32, 32, 64),  # 32 is divisible by 8
            (3, 24, 24, 64),  # 24 is divisible by 8
            (2, 40, 40, 64),  # 40 is divisible by 8
        ]

        for input_size in input_sizes:
            test_input = keras.random.normal(input_size)
            output = layer(test_input)

            # Check output shape matches input shape
            assert output.shape == test_input.shape
            assert not np.any(np.isnan(output.numpy()))

    def test_mlp_ratio_effects(self):
        """Test that different MLP ratios work correctly."""
        mlp_ratios = [1.0, 2.0, 4.0, 8.0]
        test_input = keras.random.normal([2, 56, 56, 64])  # 56 is divisible by 8

        for mlp_ratio in mlp_ratios:
            layer = SwinTransformerBlock(dim=64, num_heads=8, window_size=8, mlp_ratio=mlp_ratio)
            output = layer(test_input)

            # Check output is valid
            assert output.shape == test_input.shape
            assert not np.any(np.isnan(output.numpy()))

            # Check that MLP hidden dimension is correctly calculated
            expected_hidden_dim = int(64 * mlp_ratio)
            actual_hidden_dim = layer.mlp.hidden_dim
            assert actual_hidden_dim == expected_hidden_dim

    def test_window_size_divisibility_is_handled_by_internal_padding(self):
        """A spatial extent that is not a multiple of ``window_size`` must WORK.

        SCOPE PIN, updated in place (plan step 3, F-05). Until this change the
        method at this site was ``test_window_size_compatibility`` and asserted
        the OPPOSITE::

            incompatible_input = keras.random.normal([2, 32, 32, 64])
            with pytest.raises(Exception):
                layer(incompatible_input)

        "Compatibility" there meant "the layer raises cleanly", which was
        generous: the raise was an unhandled backend ``InvalidArgumentError``
        from ``window_partition``'s reshape ("Input to reshape is a tensor with
        288 values, but the requested shape has 128" at the probe geometry),
        fired late from inside the forward pass, and for a functional model it
        did not fire at construction at all. ``SwinTransformerBlock.call`` now
        pads bottom/right internally and crops back, so the assertion is
        inverted rather than deleted: the same geometry that used to raise must
        now round-trip at its ORIGINAL shape.
        """
        layer = SwinTransformerBlock(dim=64, num_heads=8, window_size=7)

        # Divisible: 56 == 8 * 7. Unchanged behaviour.
        compatible_input = keras.random.normal([2, 56, 56, 64])
        output = layer(compatible_input)
        assert output.shape == compatible_input.shape
        assert not np.any(np.isnan(output.numpy()))

        # NOT divisible: 32 % 7 == 4, so the block pads to 35 internally.
        incompatible_input = keras.random.normal([2, 32, 32, 64])
        output = layer(incompatible_input)
        assert tuple(output.shape) == (2, 32, 32, 64), (
            "The block must return the CALLER's (H, W), not the padded extent."
        )
        assert np.isfinite(output.numpy()).all()

# ---------------------------------------------------------------------------
# F-05 / SC-5 -- block-internal padding for non-divisible (H, W).
#
# `SwinTransformerBlock.call` used to be a hard crash on any `(H, W)` that was
# not a multiple of `window_size`, in ALL FOUR of {static, symbolic} x
# {shift = 0, shift > 0}. Measured at the pre-fix commit `8f60e2e6`, geometry
# `(1, 6, 6, 8)` with `window_size=4`:
#
#   static   shift=0 -> InvalidArgumentError "Input to reshape is a tensor with
#                       288 values, but the requested shape has 128", raised
#                       from `utils/tensors.py::window_partition` via
#                       `swin_transformer_block.py::call`'s DATA partition.
#   static   shift=2 -> InvalidArgumentError "Input to reshape is a tensor with
#                       36 values, but the requested shape has 16", raised one
#                       call EARLIER, from `_build_swmsa_keep_mask`'s own
#                       `window_partition` on the 1-channel region image.
#   symbolic shift=0 -> InvalidArgumentError, 288 -> 128 (graph execution).
#   symbolic shift=2 -> InvalidArgumentError, 288 -> 128 (graph execution).
#                       NOTE: the plan predicted the 36 -> 16 mask-builder
#                       failure here too. It does NOT reproduce symbolically --
#                       both nodes are in the graph and TF surfaces the DATA
#                       partition's failure. The CLASS of failure held; the
#                       predicted message did not.
#
# The symbolic half needs an explicit `tf.function(input_signature=...)` trace.
# A `keras.Input` functional build with a static non-divisible shape does NOT
# raise at construction (G-03): Keras 3 skips `Reshape` element-count
# validation for `KerasTensor`s, so it "succeeds" with a wrong
# `model.output_shape` and defers the crash to the first real forward call. A
# construction-only test is therefore structurally vacuous.
# ---------------------------------------------------------------------------

import tensorflow as tf  # noqa: E402  (test-local, used only by the block below)

PAD_DIM = 8
PAD_HEADS = 2
PAD_WS = 4
PAD_SEED = 1234

#: Every ``(H, W, window_size)`` a SwinTransformerBlock actually sees inside the
#: two CPU-pinned golden-value probes. Derived by instrumenting
#: ``SwinTransformerBlock.call`` across a full forward pass of
#: ``SCUNet(config=[2]*7, dim=16, head_dim=8, window_size=8,
#: input_resolution=64)`` (28 blocks) and
#: ``SwinTransformer(embed_dim=16, depths=[2,2,2,2], num_heads=[1,2,4,8],
#: window_size=2, patch_size=4, input_shape=(64,64,3))`` (8 blocks) -- i.e. the
#: frozen CONFIGs of ``tests/test_models/test_scunet/test_golden_values.py``
#: and ``tests/test_models/test_swin_transformer/test_golden_values.py``.
#: Re-derive by instrumentation if either CONFIG ever changes; do not hand-edit.
GOLDEN_BLOCK_GEOMETRIES = (
    # SCUNet golden probe, window_size = 8.
    (64, 64, 8), (32, 32, 8), (16, 16, 8), (8, 8, 8),
    # SwinTransformer golden probe, window_size = 2.
    (16, 16, 2), (8, 8, 2), (4, 4, 2), (2, 2, 2),
)


def _seeded_padding_block(shift_size, window_size=PAD_WS, dim=PAD_DIM):
    """A built block with seeded NON-ZERO weights *and* non-zero biases.

    Keras' default ``bias_initializer='zeros'`` (plus LayerNormalization's
    zero-init beta) makes several of this block's sub-paths structurally
    unobservable, so every probe below drives a block whose rank-1 variables
    are provably non-zero.
    """
    keras.utils.set_random_seed(PAD_SEED)
    block = SwinTransformerBlock(
        dim=dim,
        num_heads=PAD_HEADS,
        window_size=window_size,
        shift_size=shift_size,
        use_bias=True,
        dropout_rate=0.0,
        attention_dropout_rate=0.0,
        stochastic_depth_rate=0.0,
    )
    block.build((None, 4 * window_size, 4 * window_size, dim))

    rng = np.random.default_rng(PAD_SEED)
    weights = []
    for variable in block.weights:
        value = rng.standard_normal(variable.shape).astype("float32") * 0.1
        if len(variable.shape) == 1:
            value = value + 0.37
        weights.append(value)
    block.set_weights(weights)

    rank1 = [
        np.asarray(v) for v in block.get_weights()
        if np.asarray(v).ndim == 1
    ]
    assert rank1, "Fixture has no bias-like variable to check."
    assert any(np.abs(v).max() > 0.0 for v in rank1), (
        "Fixture assertion failed: every bias-like variable is zero, so the "
        "probe cannot see a bias-carrying code path."
    )
    return block


class TestSwinBlockInternalPadding:
    """F-05: any ``(H, W)`` runs; already-divisible ``(H, W)`` is untouched."""

    @pytest.mark.parametrize("shift_size", [0, 2])
    def test_static_non_divisible_input_round_trips(self, shift_size):
        """Was ``InvalidArgumentError``; must now return the unpadded shape."""
        block = _seeded_padding_block(shift_size)
        x = keras.ops.convert_to_tensor(
            np.random.default_rng(3).standard_normal(
                (1, 6, 6, PAD_DIM)
            ).astype("float32")
        )
        y = block(x)
        assert tuple(y.shape) == (1, 6, 6, PAD_DIM)
        assert np.isfinite(keras.ops.convert_to_numpy(y)).all()

    @pytest.mark.parametrize("shift_size", [0, 2])
    def test_symbolic_non_divisible_input_round_trips(self, shift_size):
        """The dynamic-shape path (``models/thera``'s path) must run too.

        Traced against an explicit ``TensorSpec([None, None, None, C])`` rather
        than compared eagerly: an eager-vs-eager comparison takes the SAME
        branch on both sides and is structurally blind to this.
        """
        block = _seeded_padding_block(shift_size)

        @tf.function(
            input_signature=[
                tf.TensorSpec([None, None, None, PAD_DIM], tf.float32)
            ]
        )
        def traced(t):
            return block(t)

        x = tf.constant(
            np.random.default_rng(3).standard_normal(
                (1, 6, 6, PAD_DIM)
            ).astype("float32")
        )
        y = traced(x)
        assert tuple(y.shape) == (1, 6, 6, PAD_DIM)
        assert np.isfinite(y.numpy()).all()

    @pytest.mark.parametrize("height,width,window_size", GOLDEN_BLOCK_GEOMETRIES)
    def test_golden_geometries_need_zero_padding(
        self, height, width, window_size
    ):
        """Pad amount must be EXACTLY 0 at every geometry a golden probe reaches.

        This is proof (i) of three that the padding is a no-op for the shipped
        models; proof (ii) is the pre/post bit-identity measurement recorded in
        ``verification.md``, and proof (iii) is running the two golden-value
        modules themselves.
        """
        pad_h = (window_size - height % window_size) % window_size
        pad_w = (window_size - width % window_size) % window_size
        assert (pad_h, pad_w) == (0, 0)

    def test_divisible_unshifted_call_passes_no_attention_mask(self):
        """The no-op fast path must stay structurally no-op, not merely equal.

        On an already-divisible, unshifted call the block must still hand
        ``attention_mask=None`` to its attention layer. That is what keeps the
        golden values bit-identical; an all-ones mask happens to be numerically
        equivalent (measured), so a value-only check would not notice the fast
        path silently disappearing.
        """
        block = _seeded_padding_block(0)
        seen = {}
        original = block.attn.call

        def spy(inputs, attention_mask=None, training=None, **kwargs):
            seen["mask"] = attention_mask
            return original(
                inputs, attention_mask=attention_mask, training=training,
                **kwargs
            )

        block.attn.call = spy
        try:
            block(keras.ops.convert_to_tensor(
                np.zeros((1, 8, 8, PAD_DIM), dtype="float32")
            ))
            assert "mask" in seen, "Spy never fired -- the probe is dead."
            assert seen["mask"] is None

            seen.clear()
            block(keras.ops.convert_to_tensor(
                np.zeros((1, 6, 6, PAD_DIM), dtype="float32")
            ))
            assert seen["mask"] is not None, (
                "A padded call must carry a keep mask, or padded positions "
                "are attended to (G-05)."
            )
        finally:
            block.attn.call = original

    def test_geometry_below_window_size_is_padded_not_rejected(self):
        """SCOPE PIN, updated in place: D-006's ``H < ws`` raise no longer holds.

        Before this change ``_resolve_shift_size`` raised
        ``ValueError("... smaller than window_size ...")`` for a statically
        known ``H < window_size`` whenever ``shift_size > 0`` -- while the very
        same geometry ran fine at ``shift_size == 0``. Padding makes such an
        ``H`` legally paddable up to exactly one window, so the rule is now the
        plain reference-Swin ``min(input_resolution) <= window_size -> shift 0``
        and the raise is gone.
        """
        block = _seeded_padding_block(2)
        x = keras.ops.convert_to_tensor(
            np.random.default_rng(5).standard_normal(
                (1, 3, 3, PAD_DIM)
            ).astype("float32")
        )
        assert block._resolve_shift_size(x) == 0
        y = block(x)
        assert tuple(y.shape) == (1, 3, 3, PAD_DIM)
        assert np.isfinite(keras.ops.convert_to_numpy(y)).all()
