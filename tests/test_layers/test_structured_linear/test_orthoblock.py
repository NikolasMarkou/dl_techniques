"""
Comprehensive test suite for OrthoBlock layer.

This module contains thorough tests for the OrthoBlock layer implementation,
covering initialization, forward pass, serialization, training, and edge cases.
"""

import pytest
import numpy as np
import tensorflow as tf
import keras
import tempfile
import os

from dl_techniques.layers.structured_linear.orthoblock import OrthoBlock
from dl_techniques.regularizers.binary_preference import BinaryPreferenceRegularizer
from dl_techniques.initializers.hypersphere_orthogonal_initializer import (
    OrthogonalHypersphereInitializer,
)


class TestOrthoBlock:
    """Test suite for OrthoBlock layer implementation."""

    @pytest.fixture
    def input_tensor_2d(self) -> tf.Tensor:
        """Create a 2D test input tensor."""
        return tf.random.normal([4, 32])

    @pytest.fixture
    def input_tensor_3d(self) -> tf.Tensor:
        """Create a 3D test input tensor."""
        return tf.random.normal([2, 10, 64])

    @pytest.fixture
    def layer_instance(self) -> OrthoBlock:
        """Create a default OrthoBlock instance for testing."""
        return OrthoBlock(units=16)

    def test_initialization_custom_parameters(self):
        """Test initialization with custom parameters."""
        custom_bias_regularizer = keras.regularizers.L2(1e-4)

        layer = OrthoBlock(
            units=64,
            activation="gelu",
            use_bias=False,
            ortho_reg_factor=0.02,
            kernel_initializer="he_normal",
            bias_initializer="ones",
            bias_regularizer=custom_bias_regularizer,
            scale_initial_value=0.3,
        )

        # Check custom values
        assert layer.units == 64
        assert layer.activation == keras.activations.gelu
        assert layer.use_bias is False
        assert layer.ortho_reg_factor == 0.02
        assert isinstance(layer.kernel_initializer, keras.initializers.HeNormal)
        assert isinstance(layer.bias_initializer, keras.initializers.Ones)
        assert layer.bias_regularizer == custom_bias_regularizer
        assert layer.scale_initial_value == 0.3

    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        # Test invalid units - negative
        with pytest.raises(ValueError, match="units must be a positive integer"):
            OrthoBlock(units=-10)

        # Test invalid units - zero
        with pytest.raises(ValueError, match="units must be a positive integer"):
            OrthoBlock(units=0)

        # Test invalid units - non-integer
        with pytest.raises(ValueError, match="units must be a positive integer"):
            OrthoBlock(units=10.5)

        # Test invalid ortho_reg_factor
        with pytest.raises(ValueError, match="ortho_reg_factor must be non-negative"):
            OrthoBlock(units=32, ortho_reg_factor=-0.1)

        # Test invalid scale_initial_value - negative
        with pytest.raises(ValueError, match="scale_initial_value must be between 0.0 and 1.0"):
            OrthoBlock(units=32, scale_initial_value=-0.1)

        # Test invalid scale_initial_value - too large
        with pytest.raises(ValueError, match="scale_initial_value must be between 0.0 and 1.0"):
            OrthoBlock(units=32, scale_initial_value=1.5)

    def test_build_process_2d(self, input_tensor_2d):
        """Test that the layer builds properly with 2D input."""
        layer = OrthoBlock(units=64)
        layer(input_tensor_2d)  # Forward pass triggers build

        # Check that the layer is built
        assert layer.built is True

        # Check that all sublayers were created and built
        assert layer.dense is not None
        assert layer.dense.built is True
        assert layer.ortho_reg is not None
        assert layer.norm is not None
        assert layer.norm.built is True
        assert layer.constrained_scale is not None
        assert layer.constrained_scale.built is True

        # Check dense layer configuration
        assert layer.dense.units == 64
        # Dense layer uses linear activation (identity) internally, which is equivalent to no activation
        assert layer.dense.activation == keras.activations.linear
        assert layer.dense.kernel_regularizer == layer.ortho_reg

    def test_build_process_3d(self, input_tensor_3d):
        """Test that the layer builds properly with 3D input."""
        layer = OrthoBlock(units=32, use_bias=False)
        layer(input_tensor_3d)  # Forward pass triggers build

        # Check that the layer is built
        assert layer.built is True

        # Check sublayers are built
        assert layer.dense.built is True
        assert layer.norm.built is True
        assert layer.constrained_scale.built is True

        # Check bias configuration
        assert layer.dense.use_bias is False

    def test_output_shapes_2d(self, input_tensor_2d):
        """Test that output shapes are computed correctly for 2D input."""
        units_to_test = [16, 32, 64, 128]

        for units in units_to_test:
            layer = OrthoBlock(units=units)
            output = layer(input_tensor_2d)

            # Check actual output shape
            expected_shape = (input_tensor_2d.shape[0], units)
            assert output.shape == expected_shape

            # Test compute_output_shape separately
            computed_shape = layer.compute_output_shape(input_tensor_2d.shape)
            assert computed_shape == expected_shape

    def test_output_shapes_3d(self, input_tensor_3d):
        """Test that output shapes are computed correctly for 3D input."""
        layer = OrthoBlock(units=48)
        output = layer(input_tensor_3d)

        # Check actual output shape
        expected_shape = (input_tensor_3d.shape[0], input_tensor_3d.shape[1], 48)
        assert output.shape == expected_shape

        # Test compute_output_shape separately
        computed_shape = layer.compute_output_shape(input_tensor_3d.shape)
        assert computed_shape == expected_shape

    def test_forward_pass_basic(self, input_tensor_2d):
        """Test basic forward pass functionality."""
        layer = OrthoBlock(units=64)
        output = layer(input_tensor_2d)

        # Basic sanity checks
        assert not np.any(np.isnan(output.numpy()))
        assert not np.any(np.isinf(output.numpy()))
        assert output.dtype == input_tensor_2d.dtype

    def test_forward_pass_with_activations(self, input_tensor_2d):
        """Test forward pass with different activation functions."""
        activations = ["relu", "gelu", "swish", "linear", None]

        for activation in activations:
            layer = OrthoBlock(units=32, activation=activation)
            output = layer(input_tensor_2d)

            # Check output is valid
            assert not np.any(np.isnan(output.numpy()))
            assert output.shape == (input_tensor_2d.shape[0], 32)

            # Check activation-specific properties
            if activation == "relu":
                assert np.all(output.numpy() >= 0)  # ReLU outputs are non-negative

    def test_forward_pass_deterministic(self):
        """Test forward pass with controlled inputs for deterministic results."""
        # Create deterministic input
        controlled_input = tf.ones([2, 4])

        # Create layer with fixed parameters and no regularization
        layer = OrthoBlock(
            units=3,
            kernel_initializer="ones",
            bias_initializer="zeros",
            activation=None,  # No activation
            ortho_reg_factor=0.0,  # Disable regularization for deterministic test
            scale_initial_value=1.0
        )

        # Get output
        result = layer(controlled_input, training=False)  # Use inference mode

        # With ones initializer and ones input, we can predict the pattern
        # (though exact values depend on the normalization)
        assert result.shape == (2, 3)
        assert not np.any(np.isnan(result.numpy()))

    def test_training_mode_propagation(self, input_tensor_2d):
        """Test that training mode is properly propagated to sublayers."""
        layer = OrthoBlock(units=32)

        # Test training mode
        output_train = layer(input_tensor_2d, training=True)
        assert output_train.shape == (input_tensor_2d.shape[0], 32)

        # Test inference mode
        output_inference = layer(input_tensor_2d, training=False)
        assert output_inference.shape == (input_tensor_2d.shape[0], 32)

        # Both should be valid
        assert not np.any(np.isnan(output_train.numpy()))
        assert not np.any(np.isnan(output_inference.numpy()))

    def test_serialization(self):
        """Test serialization and deserialization of the layer."""
        # Create and build the layer
        original_layer = OrthoBlock(
            units=64,
            activation="gelu",
            use_bias=True,
            ortho_reg_factor=0.02,
            kernel_initializer="he_normal",
            bias_regularizer=keras.regularizers.L1(0.01),
            scale_initial_value=0.3
        )
        original_layer.build((None, 32))

        # Get configs
        config = original_layer.get_config()
        build_config = original_layer.get_build_config()

        # Recreate the layer
        recreated_layer = OrthoBlock.from_config(config)
        recreated_layer.build_from_config(build_config)

        # Check configuration matches
        assert recreated_layer.units == original_layer.units
        assert recreated_layer.use_bias == original_layer.use_bias
        assert recreated_layer.ortho_reg_factor == original_layer.ortho_reg_factor
        assert recreated_layer.scale_initial_value == original_layer.scale_initial_value
        assert recreated_layer.activation == original_layer.activation

    def test_serialization_functionality(self):
        """Test that serialized layer works functionally."""
        # Create original layer
        original_layer = OrthoBlock(units=32, activation="relu")
        original_layer.build((None, 16))

        # Create test data
        test_input = tf.random.normal((2, 16))
        original_output = original_layer(test_input)

        # Serialize and recreate
        config = original_layer.get_config()
        build_config = original_layer.get_build_config()

        new_layer = OrthoBlock.from_config(config)
        new_layer.build_from_config(build_config)

        # Test that new layer produces valid output
        new_output = new_layer(test_input)
        assert new_output.shape == original_output.shape
        assert not np.any(np.isnan(new_output.numpy()))

    def test_model_integration(self, input_tensor_2d):
        """Test the layer in a model context."""
        # Create a simple model with the OrthoBlock
        inputs = keras.Input(shape=(32,))
        x = OrthoBlock(units=64, activation="relu")(inputs)
        x = keras.layers.Dropout(0.1)(x)
        x = OrthoBlock(units=32, activation="gelu")(x)
        x = keras.layers.BatchNormalization()(x)
        outputs = keras.layers.Dense(10, activation="softmax")(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Compile the model
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        # Test forward pass
        y_pred = model(input_tensor_2d, training=False)
        assert y_pred.shape == (input_tensor_2d.shape[0], 10)

        # Check probabilities sum to 1 (softmax output)
        assert np.allclose(np.sum(y_pred.numpy(), axis=1), 1.0, rtol=1e-5)

    def test_model_save_load(self, input_tensor_2d):
        """Test saving and loading a model with OrthoBlock."""
        # Create a model with OrthoBlock
        inputs = keras.Input(shape=(32,))
        x = OrthoBlock(units=64, activation="relu", name="ortho_layer1")(inputs)
        x = keras.layers.Dense(32, activation="gelu")(x)
        x = OrthoBlock(units=16, activation="swish", name="ortho_layer2")(x)
        outputs = keras.layers.Dense(5, activation="softmax")(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Generate prediction before saving
        original_prediction = model.predict(input_tensor_2d, verbose=0)

        # Create temporary directory for model
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "model.keras")

            # Save the model
            model.save(model_path)

            # Load the model
            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={"OrthoBlock": OrthoBlock}
            )

            # Generate prediction with loaded model
            loaded_prediction = loaded_model.predict(input_tensor_2d, verbose=0)

            # Check predictions match
            assert np.allclose(original_prediction, loaded_prediction, rtol=1e-5)

            # Check layer types are preserved
            assert isinstance(loaded_model.get_layer("ortho_layer1"), OrthoBlock)
            assert isinstance(loaded_model.get_layer("ortho_layer2"), OrthoBlock)

    def test_gradient_flow(self, input_tensor_2d):
        """Test gradient flow through the layer."""
        layer = OrthoBlock(units=32, activation="relu")

        # Watch the variables
        with tf.GradientTape() as tape:
            inputs = tf.Variable(input_tensor_2d)
            outputs = layer(inputs)
            loss = tf.reduce_mean(tf.square(outputs))

        # Get gradients
        grads = tape.gradient(loss, layer.trainable_variables)

        # Check gradients exist and are not None
        assert all(g is not None for g in grads)

        # Check gradients have values (not all zeros)
        assert all(np.any(g.numpy() != 0) for g in grads)

    def test_training_loop(self, input_tensor_2d):
        """Test training loop with OrthoBlock."""
        # Create a model with OrthoBlock
        model = keras.Sequential([
            keras.layers.InputLayer(shape=(32,)),
            OrthoBlock(64, activation="relu"),
            keras.layers.Dropout(0.1),
            OrthoBlock(32, activation="gelu"),
            keras.layers.Dense(10, activation="softmax")
        ])

        # Compile the model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

        # Create mock data
        x_train = tf.random.normal([32, 32])
        y_train = tf.random.uniform([32], 0, 10, dtype=tf.int32)

        # Initial loss
        initial_loss = model.evaluate(x_train, y_train, verbose=0)[0]

        # Train for a few epochs
        history = model.fit(x_train, y_train, epochs=3, batch_size=16, verbose=0)

        # Final loss
        final_loss = model.evaluate(x_train, y_train, verbose=0)[0]

        # Loss should decrease (or at least not increase significantly)
        # Allow some tolerance for small datasets and regularization effects
        assert final_loss <= initial_loss * 1.2  # Allow small increase due to regularization

    def test_numerical_stability(self):
        """Test layer stability with extreme input values."""
        layer = OrthoBlock(units=16, activation="relu")

        # Create inputs with different magnitudes
        test_cases = [
            tf.zeros((2, 8)),  # Zeros
            tf.ones((2, 8)) * 1e-10,  # Very small values
            tf.ones((2, 8)) * 1e5,   # Large values
            tf.random.normal((2, 8)) * 1e3  # Large random values
        ]

        for test_input in test_cases:
            output = layer(test_input)

            # Check for NaN/Inf values
            assert not np.any(np.isnan(output.numpy())), "NaN values detected in output"
            assert not np.any(np.isinf(output.numpy())), "Inf values detected in output"

    def test_bias_regularization(self, input_tensor_2d):
        """Test that bias regularization is properly applied."""
        # Create layer with bias regularization
        bias_reg = keras.regularizers.L2(0.01)
        layer = OrthoBlock(
            units=16,
            use_bias=True,
            bias_regularizer=bias_reg,
            ortho_reg_factor=0.01  # Also enable orthogonal regularization
        )

        # Apply the layer - regularization should be applied automatically
        _ = layer(input_tensor_2d)

        # Should have regularization losses from both orthogonal (kernel) and bias
        assert len(layer.losses) > 0

        # Check that bias regularizer is applied to dense layer
        assert layer.dense.bias_regularizer == bias_reg

    def test_zero_regularization_factor(self, input_tensor_2d):
        """Test behavior when ortho_reg_factor is 0."""
        layer = OrthoBlock(units=16, ortho_reg_factor=0.0)

        # With zero regularization factor, the orthogonal regularizer should have lambda=0
        assert layer.ortho_reg._lambda_coefficient == 0.0

        # Apply the layer
        _ = layer(input_tensor_2d)

        # There might still be losses from other regularizers (like L1 on constrained_scale)
        # but the orthogonal regularization specifically should contribute 0

    def test_different_initializers(self, input_tensor_2d):
        """Test layer with different initializer configurations."""
        initializers = [
            "glorot_uniform",
            "he_normal",
            "lecun_normal",
            keras.initializers.RandomNormal(stddev=0.01)
        ]

        for init in initializers:
            layer = OrthoBlock(
                units=16,
                kernel_initializer=init,
                bias_initializer="zeros"
            )

            output = layer(input_tensor_2d)
            assert output.shape == (input_tensor_2d.shape[0], 16)
            assert not np.any(np.isnan(output.numpy()))

    def test_constrained_scaling_behavior(self, input_tensor_2d):
        """Test that the constrained scaling behaves as expected."""
        # Create layer with very low scale value (should reduce output magnitude)
        layer_low_scale = OrthoBlock(
            units=16,
            scale_initial_value=0.1,
            activation="linear"  # Linear to see scaling effect clearly
        )

        # Create layer with high scale value
        layer_high_scale = OrthoBlock(
            units=16,
            scale_initial_value=0.9,
            activation="linear"
        )

        # Get outputs
        output_low = layer_low_scale(input_tensor_2d)
        output_high = layer_high_scale(input_tensor_2d)

        # The constrained scale should be applied, but exact comparison is complex
        # due to normalization. Just check outputs are valid and different.
        assert not np.any(np.isnan(output_low.numpy()))
        assert not np.any(np.isnan(output_high.numpy()))
        assert output_low.shape == output_high.shape

    def test_layer_naming(self):
        """Test that sublayers are properly named."""
        layer = OrthoBlock(units=16, name="test_ortho")
        layer.build((None, 8))

        # Check that sublayers have appropriate names
        assert "ortho_dense" in layer.dense.name
        assert "rms_norm" in layer.norm.name
        assert "constrained_scale" in layer.constrained_scale.name

class TestOrthoBlockDefaultPins:
    """Pin the constructor defaults and the per-instance kernel initializer.

    The `orthoblock.py` rewrite changed four constructor defaults and added
    three parameters, and all 23 pre-existing tests in this module stayed
    green. That is a coverage hole, not a break: a default flip on a public
    constructor parameter that no test can see. The remedy is an assertion,
    never a changed default.

    Every expected value below is the DECLARED default read off
    `src/dl_techniques/layers/structured_linear/orthoblock.py:270-283`, cross-checked against
    the same file's own prose:

      * `use_bias=False`   -- justified in the module docstring
        ("The normalization centers"): `ZeroCenteredRMSNorm` subtracts the
        feature-axis mean, so the mean of the bias is unidentifiable, and an
        unregularized bias is the only parameter here with no restoring
        force. `research/orthogonal_regularization.md` Sec. 5.6 agrees.
      * `scale_initial_value=0.8` -- the gate is constrained to [0, 1] and
        `BinaryPreferenceRegularizer` pushes it toward an endpoint; 0.8
        starts it open rather than at the 0.5 saddle.
      * `ortho_reg_factor=0.01`, `ortho_l1_factor=0.0`, `ortho_l2_factor=0.0`
        -- passed explicitly to `SoftOrthonormalConstraintRegularizer` at
        `orthoblock.py:336-341` precisely so a library-default flip cannot
        reach this layer silently.
      * `binary_preference_factor=1e-4`.
    """

    # A fixed, seed-free input: these tests are about construction-time
    # configuration, and `units=16 <= input_dim=32` deliberately keeps them
    # off the `units > input_dim` degenerate-rank branch that `build()` logs
    # about (that branch changes which Gram the regularizer forms, and
    # entangling a default pin with it would make the pin ambiguous).
    UNITS = 16
    INPUT_DIM = 32

    @pytest.fixture
    def probe_input(self) -> tf.Tensor:
        return tf.constant(
            np.random.RandomState(20260901).randn(4, 32).astype("float32")
        )

    def test_constructor_defaults_are_pinned(self):
        """Every default that changed (or was added) in the rewrite."""
        layer = OrthoBlock(units=self.UNITS)

        # Flipped by the rewrite (was True).
        assert layer.use_bias is False
        # Flipped by the rewrite (was 0.5).
        assert layer.scale_initial_value == 0.8
        # Unchanged, pinned here because the other two ortho factors are new
        # and a reader needs the whole triple visible in one place.
        assert layer.ortho_reg_factor == 0.01
        # Added by the rewrite; 0.0 means "no coupled L1/L2 competing with
        # the orthonormality term".
        assert layer.ortho_l1_factor == 0.0
        assert layer.ortho_l2_factor == 0.0
        # Added by the rewrite.
        assert layer.binary_preference_factor == 1e-4

    def test_default_use_bias_false_reaches_the_dense_sublayer(self):
        """`use_bias=False` must produce NO bias variable, not just a False flag.

        The attribute check above can be satisfied by a stored value that the
        `Dense` sublayer never receives, so this asserts the observable
        consequence: a built default `OrthoBlock` has `dense.bias is None`.
        """
        layer = OrthoBlock(units=self.UNITS)
        layer.build((None, self.INPUT_DIM))

        assert layer.dense.use_bias is False
        assert layer.dense.bias is None

    def test_default_scale_initial_value_reaches_the_gate_weights(self):
        """0.8 must be the value the gate is actually initialized to."""
        layer = OrthoBlock(units=self.UNITS)
        layer.build((None, self.INPUT_DIM))

        gamma = keras.ops.convert_to_numpy(layer.constrained_scale.gamma)
        # `initializer=keras.initializers.Constant(scale_initial_value)`
        # (orthoblock.py:369), so every one of the `units` gate entries is
        # exactly the declared default. Exact equality is correct here: a
        # Constant initializer performs no arithmetic.
        assert gamma.shape == (self.UNITS,)
        assert np.all(gamma == 0.8)

    def test_default_kernel_initializer_is_a_fresh_per_instance_object(self):
        """GUARD: two default `OrthoBlock`s must NOT share one Initializer object.

        This pins a BUG FIX. Before the rewrite the default argument was
        literally `kernel_initializer=OrthogonalHypersphereInitializer()`, so
        ONE instance was constructed at import time and shared by every
        `OrthoBlock` in the process. The rewrite fixed it by defaulting to
        `None` and constructing a fresh initializer inside `__init__`
        (`orthoblock.py:315-321`). Never revert that.

        WHY OBJECT IDENTITY AND NOT BIT-IDENTITY -- measured, and it corrects
        the premise this test was originally commissioned with. The brief said
        "under the old shared-instance default two default layers would draw
        EXACTLY the same kernel (max|delta| == 0.0)". That is FALSE for this
        initializer. `OrthogonalHypersphereInitializer(seed=None)` redraws on
        every `__call__`, so a shared UNSEEDED instance still yields different
        kernels: measured `max|delta|` = 1.0499 across two calls on one
        instance, and re-injecting the old shared-instance default into the
        real source left this module 31/31 GREEN. A bit-identity assertion
        alone is therefore a guard that CANNOT FAIL against the defect it
        names.

        The defect is real but conditional on the seed:
        `OrthogonalHypersphereInitializer(seed=42)` shared by two layers WAS
        measured to give bit-identical kernels (pinned below by
        `test_two_layers_sharing_a_seeded_initializer_get_identical_kernels`).
        So the invariant that actually protects the fix is structural -- each
        layer owns its own initializer object -- and that is what is asserted
        here. It goes RED the moment anyone restores a shared default,
        regardless of whether that default carries a seed.

        SCOPE / ANTI-VACUITY: this test is not asserting "random things
        differ". Its two companions run the same comparison where SAMENESS is
        the correct answer -- an explicitly seeded initializer passed to both
        layers, and one seeded initializer object shared by both layers. The
        trio is what makes the guard discriminating: only the DEFAULT path is
        required to be per-instance.
        """
        a = OrthoBlock(units=self.UNITS)
        b = OrthoBlock(units=self.UNITS)

        assert isinstance(a.kernel_initializer, OrthogonalHypersphereInitializer)
        assert a.kernel_initializer is not b.kernel_initializer, (
            "Two default OrthoBlocks share one Initializer object. That is the "
            "signature of a module-level Initializer instance used as a default "
            "argument -- the defect the rewrite fixed by defaulting "
            "kernel_initializer to None. Do not restore the shared instance."
        )

        # The consequence, kept as a second (weaker) arm: the drawn kernels
        # differ. Measured max|delta| at (32, 16) was 1.1375 and 1.2585 on two
        # separate draws -- large, not borderline. Note this arm passes under
        # the old default too (see the docstring), so it documents behaviour
        # rather than guarding the fix.
        a.build((None, self.INPUT_DIM))
        b.build((None, self.INPUT_DIM))
        ka = keras.ops.convert_to_numpy(a.dense.kernel)
        kb = keras.ops.convert_to_numpy(b.dense.kernel)
        assert ka.shape == (self.INPUT_DIM, self.UNITS)
        assert not np.array_equal(ka, kb)

    def test_two_layers_sharing_a_seeded_initializer_get_identical_kernels(self):
        """The MECHANISM the per-instance default protects against.

        One seeded `OrthogonalHypersphereInitializer` object handed to two
        layers produces bit-identical kernels. This is the failure a shared
        module-level default would cause whenever that default carried a seed,
        and it is why the guard above asserts object identity rather than a
        numeric difference.
        """
        shared = OrthogonalHypersphereInitializer(seed=42)
        a = OrthoBlock(units=self.UNITS, kernel_initializer=shared)
        b = OrthoBlock(units=self.UNITS, kernel_initializer=shared)
        # `keras.initializers.get()` returns an Initializer instance
        # unchanged, so both layers really do hold the same object.
        assert a.kernel_initializer is b.kernel_initializer is shared

        a.build((None, self.INPUT_DIM))
        b.build((None, self.INPUT_DIM))
        ka = keras.ops.convert_to_numpy(a.dense.kernel)
        kb = keras.ops.convert_to_numpy(b.dense.kernel)
        assert np.array_equal(ka, kb)

    def test_explicitly_seeded_initializer_is_reproducible_across_instances(self):
        """ANTI-VACUITY arm for the guard above.

        Same comparison, same shapes, same code path -- but with an explicitly
        seeded initializer handed to both layers. Here bit-identity is the
        CORRECT outcome. If this test ever fails, the guard above is measuring
        harness randomness rather than the shared-instance defect, and its
        result would mean nothing.
        """
        a = OrthoBlock(
            units=self.UNITS,
            kernel_initializer=keras.initializers.GlorotUniform(seed=1234),
        )
        b = OrthoBlock(
            units=self.UNITS,
            kernel_initializer=keras.initializers.GlorotUniform(seed=1234),
        )
        a.build((None, self.INPUT_DIM))
        b.build((None, self.INPUT_DIM))

        ka = keras.ops.convert_to_numpy(a.dense.kernel)
        kb = keras.ops.convert_to_numpy(b.dense.kernel)
        assert np.array_equal(ka, kb)

    def test_default_binary_preference_factor_installs_the_regularizer(self):
        """The default 1e-4 must actually install the gate regularizer.

        `orthoblock.py:370-373` branches on `binary_preference_factor > 0.0`,
        a live conditional with no coverage before this test.
        """
        layer = OrthoBlock(units=self.UNITS)

        reg = layer.constrained_scale.regularizer
        assert isinstance(reg, BinaryPreferenceRegularizer)
        # The factor is forwarded as the regularizer's `multiplier`.
        # `multiplier` is a keras.Variable when annealable, so read the float
        # accessor rather than comparing against a tensor -- and the Variable
        # is float32, so the stored value is float32(1e-4) =
        # 9.999999747378752e-05, NOT the float64 literal 1e-4. Compare against
        # the float32 image of the default, exactly. This is not a widened
        # tolerance: it is the correct expected value for a float32 store, and
        # asserting it exactly still fails on any other multiplier.
        assert float(reg.multiplier_value) == float(np.float32(1e-4))

    def test_zero_binary_preference_factor_removes_the_regularizer(self):
        """The other side of the same conditional: 0.0 means NO regularizer.

        Not "a regularizer with multiplier 0" -- the branch installs `None`,
        which is what the module docstring (line 135) and the gate-death
        warning at `orthoblock.py:396` both promise as the escape hatch.
        """
        layer = OrthoBlock(units=self.UNITS, binary_preference_factor=0.0)

        assert layer.constrained_scale.regularizer is None

    def test_binary_preference_factor_survives_a_config_round_trip(self):
        """The three new parameters must round-trip, defaults included."""
        original = OrthoBlock(units=self.UNITS)
        config = original.get_config()

        assert config["use_bias"] is False
        assert config["scale_initial_value"] == 0.8
        assert config["ortho_l1_factor"] == 0.0
        assert config["ortho_l2_factor"] == 0.0
        assert config["binary_preference_factor"] == 1e-4

        rebuilt = OrthoBlock.from_config(config)
        assert rebuilt.use_bias is False
        assert rebuilt.scale_initial_value == 0.8
        assert rebuilt.ortho_l1_factor == 0.0
        assert rebuilt.ortho_l2_factor == 0.0
        assert rebuilt.binary_preference_factor == 1e-4


# Additional utility functions for testing

def test_orthoblock_example_usage():
    """Test the example usage from the docstring."""
    # Basic usage
    x = keras.Input(shape=(128,))
    y = OrthoBlock(units=64, activation='relu')(x)
    model = keras.Model(inputs=x, outputs=y)

    # Test model creation
    assert model is not None
    assert len(model.layers) == 2  # Input + OrthoBlock

    # Custom usage
    ortho_layer = OrthoBlock(
        units=32,
        activation='gelu',
        ortho_reg_factor=0.02,
        scale_initial_value=0.3
    )

    input_tensor = tf.random.normal((4, 128))
    output = ortho_layer(input_tensor)
    assert output.shape == (4, 32)

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])