"""
Test suite for ``SwinConvBlock`` -- plus a legacy ``SwinMLP`` suite.

.. note::

   Until the H-15 work below, this module contained **only** the ``SwinMLP``
   suite (a near-copy of ``tests/test_layers/test_ffn/test_swin_mlp.py``), so
   its filename asserted a coverage of ``SwinConvBlock`` that did not exist:
   ``grep -n 'effective_block_type\\|input_resolution'`` over this file returned
   **zero** hits, and ``grep -rl SwinConvBlock tests/`` reached only the two
   SCUNet model modules. The ``TestSwinConvBlockInputResolutionIsAdvisory``
   class at the bottom is the first direct coverage of ``SwinConvBlock`` in
   ``tests/``. The ``SwinMLP`` classes are left where they are; moving them is
   out of scope.
"""
import pytest
import numpy as np
import keras
import tempfile
import os
from keras import ops

from dl_techniques.layers.ffn.swin_mlp import SwinMLP
from dl_techniques.layers.transformers.swin_conv_block import SwinConvBlock


class TestSwinMLP:
    """Comprehensive test suite for SwinMLP layer implementation."""

    @pytest.fixture
    def input_tensor(self):
        """Create a test input tensor."""
        return keras.random.normal([8, 16, 128])  # batch, seq_len, features

    @pytest.fixture
    def layer_instance(self):
        """Create a default layer instance for testing."""
        return SwinMLP(hidden_dim=256)

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        layer = SwinMLP(hidden_dim=128)

        # Check default values
        assert layer.hidden_dim == 128
        assert layer.output_dim is None
        # activation is stored via keras.activations.get() -> a function; assert
        # functional equivalence, not the raw string token.
        assert keras.activations.serialize(layer.activation) == "gelu"
        assert layer.dropout_rate == 0.0
        assert isinstance(layer.kernel_initializer, keras.initializers.GlorotUniform)
        assert isinstance(layer.bias_initializer, keras.initializers.Zeros)
        assert layer.kernel_regularizer is None
        assert layer.bias_regularizer is None
        assert layer.activity_regularizer is None

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        custom_regularizer = keras.regularizers.L2(1e-4)

        layer = SwinMLP(
            hidden_dim=256,
            output_dim=64,
            activation="relu",
            dropout_rate=0.2,
            kernel_initializer="he_normal",
            bias_initializer="ones",
            kernel_regularizer=custom_regularizer,
            bias_regularizer=custom_regularizer,
            activity_regularizer=custom_regularizer,
        )

        # Check custom values
        assert layer.hidden_dim == 256
        assert layer.output_dim == 64
        assert keras.activations.serialize(layer.activation) == "relu"
        assert layer.dropout_rate == 0.2
        assert isinstance(layer.kernel_initializer, keras.initializers.HeNormal)
        assert isinstance(layer.bias_initializer, keras.initializers.Ones)
        assert layer.kernel_regularizer == custom_regularizer
        assert layer.bias_regularizer == custom_regularizer
        assert layer.activity_regularizer == custom_regularizer

    def test_build_process(self, input_tensor):
        """Test that the layer builds properly."""
        layer = SwinMLP(hidden_dim=256, dropout_rate=0.1)
        layer(input_tensor)  # Forward pass triggers build

        # Check that weights were created
        assert layer.built is True
        assert len(layer.weights) > 0
        assert hasattr(layer, "fc1")
        assert hasattr(layer, "fc2")
        assert layer.fc1 is not None
        assert layer.fc2 is not None

        # Check sublayers were built
        assert layer.fc1.built is True
        assert layer.fc2.built is True

        # Check dropout layers were created
        assert layer.drop1 is not None
        assert layer.drop2 is not None

    def test_output_shapes(self, input_tensor):
        """Test that output shapes are computed correctly."""
        test_cases = [
            (128, None),  # No output dim specified
            (256, 64),  # Custom output dim
            (512, 32),  # Another custom output dim
        ]

        for hidden_dim, output_dim in test_cases:
            layer = SwinMLP(hidden_dim=hidden_dim, output_dim=output_dim)
            output = layer(input_tensor)

            # Check output shape
            expected_shape = list(input_tensor.shape)
            if output_dim is not None:
                expected_shape[-1] = output_dim
            expected_shape = tuple(expected_shape)

            assert output.shape == expected_shape

            # Test compute_output_shape separately
            computed_shape = layer.compute_output_shape(input_tensor.shape)
            assert computed_shape == expected_shape

    def test_forward_pass(self, input_tensor):
        """Test that forward pass produces expected values."""
        layer = SwinMLP(hidden_dim=256)
        output = layer(input_tensor)

        # Basic sanity checks
        assert not np.any(np.isnan(output.numpy()))
        assert not np.any(np.isinf(output.numpy()))

        # Test with controlled inputs for deterministic output
        controlled_input = keras.ops.ones([2, 4, 8])
        deterministic_layer = SwinMLP(
            hidden_dim=16,
            output_dim=4,
            kernel_initializer="ones",
            bias_initializer="zeros",
            activation="linear"
        )
        result = deterministic_layer(controlled_input)

        # Check output shape
        assert result.shape == (2, 4, 4)
        assert not np.any(np.isnan(result.numpy()))

    def test_different_activations(self, input_tensor):
        """Test layer with different activation functions."""
        activations = ["relu", "gelu", "swish", "tanh", "linear"]

        for act in activations:
            layer = SwinMLP(hidden_dim=128, activation=act)
            output = layer(input_tensor)

            # Check output is valid
            assert not np.any(np.isnan(output.numpy()))
            assert output.shape == input_tensor.shape  # Same shape since output_dim is None

    def test_dropout_behavior(self, input_tensor):
        """Test dropout behavior during training vs inference."""
        layer = SwinMLP(hidden_dim=128, dropout_rate=0.5)

        # Test inference mode (no dropout)
        output_inference = layer(input_tensor, training=False)

        # Test training mode (with dropout)
        output_training = layer(input_tensor, training=True)

        # Both should have same shape
        assert output_inference.shape == output_training.shape
        assert output_inference.shape == input_tensor.shape

    def test_serialization(self):
        """Test serialization and deserialization of the layer."""
        original_layer = SwinMLP(
            hidden_dim=256,
            output_dim=64,
            activation="relu",
            dropout_rate=0.1,
            kernel_initializer="he_normal",
            kernel_regularizer=keras.regularizers.L2(1e-4)
        )

        # Build the layer
        input_shape = (None, 16, 128)
        original_layer.build(input_shape)

        # Get configs
        config = original_layer.get_config()
        build_config = original_layer.get_build_config()

        # Recreate the layer
        recreated_layer = SwinMLP.from_config(config)
        recreated_layer.build_from_config(build_config)

        # Check configuration matches
        assert recreated_layer.hidden_dim == original_layer.hidden_dim
        assert recreated_layer.output_dim == original_layer.output_dim
        assert recreated_layer.activation == original_layer.activation
        assert recreated_layer.dropout_rate == original_layer.dropout_rate

        # Check weights match (shapes should be the same)
        assert len(recreated_layer.weights) == len(original_layer.weights)
        for w1, w2 in zip(original_layer.weights, recreated_layer.weights):
            assert w1.shape == w2.shape

    def test_model_integration(self, input_tensor):
        """Test the layer in a model context."""
        # Create a simple model with the custom layer
        inputs = keras.Input(shape=input_tensor.shape[1:])
        x = SwinMLP(hidden_dim=256)(inputs)
        x = keras.layers.LayerNormalization()(x)
        x = SwinMLP(hidden_dim=128, output_dim=64)(x)
        x = keras.layers.GlobalAveragePooling1D()(x)
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
        x = SwinMLP(hidden_dim=128, name="swin_mlp1")(inputs)
        x = keras.layers.LayerNormalization()(x)
        x = SwinMLP(hidden_dim=64, name="swin_mlp2")(x)
        x = keras.layers.GlobalAveragePooling1D()(x)
        outputs = keras.layers.Dense(5)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Generate a prediction before saving
        original_prediction = model.predict(input_tensor, verbose=0)

        # Create temporary directory for model
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "model.keras")

            # Save the model
            model.save(model_path)

            # Load the model with custom objects
            loaded_model = keras.models.load_model(model_path)

            # Generate prediction with loaded model
            loaded_prediction = loaded_model.predict(input_tensor, verbose=0)

            # Check predictions match
            assert np.allclose(original_prediction, loaded_prediction, rtol=1e-5)

            # Check layer types are preserved
            assert isinstance(loaded_model.get_layer("swin_mlp1"), SwinMLP)
            assert isinstance(loaded_model.get_layer("swin_mlp2"), SwinMLP)

    def test_regularization(self, input_tensor):
        """Test that regularization losses are properly applied."""
        # Create layer with regularization
        layer = SwinMLP(
            hidden_dim=128,
            kernel_regularizer=keras.regularizers.L2(0.01),
            bias_regularizer=keras.regularizers.L1(0.01)
        )

        # No regularization losses before calling the layer
        assert len(layer.losses) == 0

        # Apply the layer
        _ = layer(input_tensor)

        # Should have regularization losses now
        assert len(layer.losses) > 0

    def test_shape_handling(self):
        """Test shape handling with different input formats."""
        layer = SwinMLP(hidden_dim=64, output_dim=32)

        # Test with tuple shape
        tuple_shape = (None, 16, 128)
        output_shape = layer.compute_output_shape(tuple_shape)
        assert output_shape == (None, 16, 32)

        # Test with list shape
        list_shape = [None, 16, 128]
        output_shape = layer.compute_output_shape(list_shape)
        assert output_shape == (None, 16, 32)

        # Test without output_dim (should preserve input shape)
        layer_no_out = SwinMLP(hidden_dim=64)
        output_shape = layer_no_out.compute_output_shape(tuple_shape)
        assert output_shape == tuple_shape

    def test_numerical_stability(self):
        """Test layer stability with extreme input values."""
        layer = SwinMLP(hidden_dim=32)

        # Create inputs with different magnitudes
        batch_size = 4
        seq_len = 8
        features = 16

        test_cases = [
            keras.ops.zeros((batch_size, seq_len, features)),  # Zeros
            keras.ops.ones((batch_size, seq_len, features)) * 1e-10,  # Very small values
            keras.ops.ones((batch_size, seq_len, features)) * 1e5,  # Large values
            keras.random.normal((batch_size, seq_len, features)) * 100  # Large random values
        ]

        for test_input in test_cases:
            output = layer(test_input)

            # Check for NaN/Inf values
            assert not np.any(np.isnan(output.numpy())), "NaN values detected in output"
            assert not np.any(np.isinf(output.numpy())), "Inf values detected in output"

    def test_different_input_shapes(self):
        """Test layer with different input shapes."""
        layer = SwinMLP(hidden_dim=64)

        # Test different input shapes
        test_shapes = [
            (2, 32),  # 2D input
            (4, 16, 32),  # 3D input
            (2, 8, 16, 32),  # 4D input
        ]

        for shape in test_shapes:
            test_input = keras.random.normal(shape)
            output = layer(test_input)

            # Output should have same shape as input (no output_dim specified)
            assert output.shape == test_input.shape

    def test_training_vs_inference_mode(self, input_tensor):
        """Test behavior differences between training and inference modes."""
        layer = SwinMLP(hidden_dim=128, dropout_rate=0.3)

        # Test multiple calls in inference mode - should be consistent
        output1 = layer(input_tensor, training=False)
        output2 = layer(input_tensor, training=False)

        # In inference mode, outputs should be identical (no dropout)
        assert np.allclose(output1.numpy(), output2.numpy())

        # Test training mode - outputs may differ due to dropout
        output_train1 = layer(input_tensor, training=True)
        output_train2 = layer(input_tensor, training=True)

        # Outputs should have the same shape
        assert output_train1.shape == output_train2.shape == input_tensor.shape

    def test_layer_weights_structure(self, input_tensor):
        """Test that layer weights have expected structure."""
        layer = SwinMLP(hidden_dim=256, output_dim=64)
        layer(input_tensor)  # Build the layer

        # Should have weights from two dense layers
        # Each dense layer has kernel + bias = 4 total weights
        assert len(layer.weights) == 4

        # Check weight shapes
        input_dim = input_tensor.shape[-1]
        fc1_kernel_shape = (input_dim, 256)
        fc1_bias_shape = (256,)
        fc2_kernel_shape = (256, 64)
        fc2_bias_shape = (64,)

        expected_shapes = [fc1_kernel_shape, fc1_bias_shape, fc2_kernel_shape, fc2_bias_shape]
        actual_shapes = [w.shape for w in layer.weights]

        assert actual_shapes == expected_shapes


class TestSwinMLPEdgeCases:
    """Test edge cases and error handling for SwinMLP."""

    def test_very_small_hidden_dim(self):
        """Test with very small hidden dimension."""
        layer = SwinMLP(hidden_dim=1, output_dim=1)
        test_input = keras.random.normal([2, 3, 4])

        output = layer(test_input)
        assert output.shape == (2, 3, 1)
        assert not np.any(np.isnan(output.numpy()))

    def test_large_hidden_dim(self):
        """Test with large hidden dimension."""
        layer = SwinMLP(hidden_dim=2048)
        test_input = keras.random.normal([2, 4, 8])

        output = layer(test_input)
        assert output.shape == test_input.shape
        assert not np.any(np.isnan(output.numpy()))

    def test_custom_activation_callable(self):
        """Test with custom activation function as callable."""

        def custom_activation(x):
            return keras.ops.relu(x) * 0.5

        layer = SwinMLP(hidden_dim=64, activation=custom_activation)
        test_input = keras.random.normal([2, 4, 8])

        output = layer(test_input)
        assert output.shape == test_input.shape
        assert not np.any(np.isnan(output.numpy()))

    def test_bias_free_mode(self):
        """Test layer with bias disabled."""
        layer = SwinMLP(hidden_dim=64, use_bias=False)
        test_input = keras.random.normal([2, 4, 8])

        output = layer(test_input)
        assert output.shape == test_input.shape
        assert not np.any(np.isnan(output.numpy()))

        # Build the layer and check that no bias weights exist
        layer(test_input)

        # With use_bias=False, we should only have kernel weights (no bias)
        # 2 kernel weights total (fc1.kernel, fc2.kernel)
        assert len(layer.weights) == 2

        # Check that all weights are 2D (kernels, not 1D biases)
        for weight in layer.weights:
            assert len(weight.shape) == 2

# ---------------------------------------------------------------------------
# H-15 / G-04(b1): ``SwinConvBlock.input_resolution`` is an ADVISORY hint
# ---------------------------------------------------------------------------
#
# ``SwinConvBlock.__init__`` used to downgrade ``block_type`` ``"SW" -> "W"``
# whenever ``input_resolution <= window_size``. Two measured facts retired it:
#
#   * it is bit-identically REDUNDANT in the honest case (the hint matches the
#     real extent) -- ``SwinTransformerBlock._resolve_shift_size`` already
#     applies the same rule, on the REAL tensor; and
#   * it is ACTIVELY WRONG when the hint lies, because ``input_resolution`` is
#     never cross-checked against ``x.shape``.
#
# The measured figures live beside the assertions that pin them.

_CD, _TD, _HD = 16, 16, 8

#: ``(input_resolution, window_size)`` cells where the hint TELLS THE TRUTH:
#: the block is fed a map of exactly ``input_resolution`` square. This is
#: invariant **I-D** -- removing the downgrade must not move numerics here.
#:
#: PROBE-9 (dead-component: force ``_resolve_shift_size``'s single-window rule
#: never to fire, so a real roll + region mask reaches the honest geometry)
#: takes **3 of these 4** cells RED on the ``maxdiff`` assertion. ``(4, 8)``
#: stays GREEN and is therefore NOT discriminating: ``H=4`` pads up to ``ws=8``,
#: giving ``pad_h == 4 == shift``, the enumerated ``pad == shift`` degenerate
#: family. It is kept because the plan names it as an I-D cell, not because it
#: can see a shift.
_HONEST_GEOMETRIES = ((4, 8), (8, 8), (3, 4), (4, 4))


def _conv_block(ws, block_type, input_resolution):
    return SwinConvBlock(
        conv_dim=_CD, trans_dim=_TD, head_dim=_HD, window_size=ws,
        block_type=block_type, input_resolution=input_resolution,
        drop_path_rate=0.0,
    )


def _weight_matched_maxdiff(block_a, block_b, x):
    """Run two blocks on identical WEIGHTS and identical input.

    The weight copy is not optional: two independently constructed blocks draw
    different initializers, so an un-matched A/B measures the initializer, not
    the geometry. The match is ASSERTED, not assumed.
    """
    tx = ops.convert_to_tensor(x)
    block_a(tx)
    block_b(tx)
    block_b.set_weights(block_a.get_weights())
    wa = [np.asarray(w) for w in block_a.get_weights()]
    wb = [np.asarray(w) for w in block_b.get_weights()]
    assert len(wa) == len(wb) and len(wa) > 0
    for u, v in zip(wa, wb):
        assert np.array_equal(u, v), "weight match failed; the A/B is invalid"
    ya = np.asarray(block_a(tx))
    yb = np.asarray(block_b(tx))
    return float(np.max(np.abs(ya - yb)))


class TestSwinConvBlockInputResolutionIsAdvisory:
    """``input_resolution`` is accepted, stored and serialized -- and inert."""

    @pytest.mark.parametrize("res,ws", _HONEST_GEOMETRIES)
    def test_honest_hint_is_bit_identical_to_a_plain_window_block(self, res, ws):
        """**I-D**: at an HONEST hint the shifted block matches a regular one.

        The comparison is against an explicitly constructed ``block_type="W"``
        block -- i.e. against exactly what the removed downgrade used to
        produce -- so the cell keeps discriminating after the removal. A
        ``SW``-vs-``SW`` comparison would degenerate into an identity check.

        Measured at HEAD ``c5d8ad7e``: ``maxdiff 0.000000e+00`` at 4/4 cells.
        """
        rng = np.random.default_rng(1234)
        x = rng.normal(size=(2, res, res, _CD + _TD)).astype("float32")

        shifted = _conv_block(ws, "SW", res)
        regular = _conv_block(ws, "W", None)

        # Precondition, and the H-15 contract: the hint no longer suppresses
        # the shift at construction. Without this the bit-identity below is
        # trivially true and proves nothing.
        assert shifted.effective_block_type == "SW"
        assert shifted.trans_block.shift_size == ws // 2

        assert _weight_matched_maxdiff(shifted, regular, x) == 0.0

    def test_lying_hint_no_longer_drops_the_shift(self):
        """A hint that DISAGREES with the real map must not suppress SW-MSA.

        ``input_resolution=4`` with ``window_size=8`` on a real ``16x16`` map
        (2 windows wide) is the counterexample: the runtime rule
        ``min(H, W) <= window_size`` is FALSE here, so the shift belongs.

        Measured at HEAD ``c5d8ad7e``, before the removal: the hinted block
        differed from the honest one by ``6.526413e-01`` -- the shift was
        dropped anyway, because ``input_resolution`` is never cross-checked
        against ``x.shape``.
        """
        rng = np.random.default_rng(1234)
        x = rng.normal(size=(2, 16, 16, _CD + _TD)).astype("float32")

        hinted = _conv_block(8, "SW", 4)
        honest = _conv_block(8, "SW", None)

        assert hinted.trans_block.shift_size == honest.trans_block.shift_size == 4
        assert _weight_matched_maxdiff(hinted, honest, x) == 0.0

    def test_undersized_hint_no_longer_loses_sw_msa_inside_scunet(self):
        """The shipped path: ``SCUNet`` forwards the hint blind.

        ``SCUNet._create_stage_blocks`` halves ``input_resolution`` per stage
        with no reference to the tensor, so a model built at one resolution and
        fed a larger image used to lose SW-MSA at every stage whose HINT fell
        to ``<= window_size``.

        Measured at HEAD ``c5d8ad7e``: ``SCUNet(input_resolution=64,
        window_size=8)`` fed ``128x128`` differed from the same weights under
        an honest hint by ``1.562450e+00``, from the BOTTLENECK stage alone
        (hint ``64//8 = 8 <= 8``; real grid ``16``, i.e. 2 windows).

        NOT the briefed geometry: the plan predicted the DEFAULT
        ``input_resolution=256`` loses SW-MSA "at every stage". That is
        **false** -- measured, at ``ws=8`` the default's deepest hint is
        ``256//8 = 32 > 8``, so 0 of 7 declared-``SW`` blocks were downgraded.
        The downgrade needed ``input_resolution <= 64`` at ``ws=8``.
        """
        from dl_techniques.models.scunet.model import SCUNet

        cfg = dict(in_nc=3, config=[2] * 7, dim=16, head_dim=8, window_size=8,
                   stochastic_depth_rate=0.0)
        rng = np.random.default_rng(3)
        x = ops.convert_to_tensor(
            (rng.normal(size=(1, 128, 128, 3)) * 0.5).astype("float32"))

        undersized = SCUNet(input_resolution=64, **cfg)
        honest = SCUNet(input_resolution=512, **cfg)
        undersized(x)
        honest(x)
        honest.set_weights(undersized.get_weights())

        effs = [b.effective_block_type
                for b in undersized._flatten_layers()
                if isinstance(b, SwinConvBlock) and b.block_type == "SW"]
        assert effs == ["SW"] * 7, f"a declared SW stage was downgraded: {effs}"

        ya = np.asarray(undersized(x))
        yb = np.asarray(honest(x))
        assert float(np.max(np.abs(ya - yb))) == 0.0

    def test_input_resolution_is_still_an_accepted_config_key(self):
        """The key stays in the public surface: ``SCUNet`` passes it and
        ``get_config()`` must round-trip it. Only its BEHAVIOUR was removed."""
        block = _conv_block(8, "SW", 4)
        config = block.get_config()
        assert config["input_resolution"] == 4

        restored = SwinConvBlock.from_config(config)
        assert restored.input_resolution == 4
        assert restored.block_type == "SW"
        assert restored.effective_block_type == "SW"
        assert restored.trans_block.shift_size == 4

        # Still validated as a value, even though it is advisory.
        with pytest.raises(ValueError, match="input_resolution must be positive"):
            _conv_block(8, "SW", 0)
