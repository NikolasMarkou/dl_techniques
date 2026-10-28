"""
Comprehensive test suite for the SqueezeNet V1 model.

This module contains all tests for the SqueezeNet V1 model implementation,
covering initialization, forward pass, training, serialization,
model variants, and edge cases.
"""

import json
import pytest
import numpy as np
import tensorflow as tf
import keras
import tempfile
import os
from typing import Tuple, Dict, Any

from dl_techniques.models.vision.squeezenet.squeezenet_v1 import SqueezeNetV1, FireModule, create_squeezenet_v1
from tests.optimizer_state import build_optimizer_state
from tests.test_models.initializer_replay_oracle import replayed_pairs


class TestSqueezeNetV1:
    """Test suite for SqueezeNet V1 model implementation."""

    @pytest.fixture
    def input_shape(self) -> Tuple[int, int, int]:
        """Create test input shape for standard images."""
        return (224, 224, 3)

    @pytest.fixture
    def small_input_shape(self) -> Tuple[int, int, int]:
        """Create test input shape for smaller images."""
        return (64, 64, 3)

    @pytest.fixture
    def cifar_input_shape(self) -> Tuple[int, int, int]:
        """Create test input shape for CIFAR-10 like images."""
        return (32, 32, 3)

    @pytest.fixture
    def num_classes(self) -> int:
        """Create test number of classes."""
        return 10

    @pytest.fixture
    def sample_data(self, input_shape):
        """Create sample input data."""
        batch_size = 2
        return tf.random.uniform([batch_size] + list(input_shape), 0, 1)

    @pytest.fixture
    def small_sample_data(self, small_input_shape):
        """Create small sample input data."""
        batch_size = 4
        return tf.random.uniform([batch_size] + list(small_input_shape), 0, 1)

    @pytest.fixture
    def cifar_sample_data(self, cifar_input_shape):
        """Create CIFAR-like sample data."""
        batch_size = 8
        return tf.random.uniform([batch_size] + list(cifar_input_shape), 0, 1)

    def test_initialization_defaults(self, num_classes, input_shape):
        """Test initialization with default parameters."""
        model = SqueezeNetV1(num_classes=num_classes, input_shape=input_shape)

        default_config = SqueezeNetV1.MODEL_VARIANTS["1.0"]
        assert model.num_classes == num_classes
        assert model.include_top is True
        assert model.dropout_rate == 0.5
        assert model.use_bypass is False
        assert model.conv1_filters == default_config["conv1_filters"]
        assert model.conv1_kernel == default_config["conv1_kernel"]
        assert model.pool_indices == default_config["pool_indices"]
        assert len(model.fire_modules) == len(default_config["fire_configs"])

    def test_initialization_custom(self, num_classes, small_input_shape):
        """Test initialization with custom parameters."""
        regularizer = keras.regularizers.L1(0.01)
        initializer = keras.initializers.HeNormal()

        model = SqueezeNetV1(
            num_classes=num_classes,
            use_bypass="simple",
            dropout_rate=0.2,
            kernel_regularizer=regularizer,
            kernel_initializer=initializer,
            include_top=False,
            input_shape=small_input_shape
        )

        assert model.num_classes == num_classes
        assert model.use_bypass == "simple"
        assert model.dropout_rate == 0.2
        assert model.kernel_regularizer == regularizer
        assert model.kernel_initializer == keras.initializers.serialize(initializer)
        assert model.include_top is False

    def test_invalid_initialization(self):
        """Test initialization with invalid parameters raises errors."""
        with pytest.raises(ValueError, match="num_classes must be a positive integer"):
            SqueezeNetV1(num_classes=0)
        with pytest.raises(ValueError, match=r"dropout_rate must be in \[0, 1\)"):
            SqueezeNetV1(dropout_rate=1.1)
        with pytest.raises(ValueError, match=r"dropout_rate must be in \[0, 1\)"):
            SqueezeNetV1(dropout_rate=-0.1)

    @pytest.mark.parametrize("value", [True, 2.5, "10", None, 0, -3])
    def test_a_num_classes_that_is_no_positive_integer_is_refused_naming_it(self, value):
        """The shared rule (``utils.validation.positive_int``): a bool, a float and a string used
        to reach the first convolution and fail there with a message about a shape or a group."""
        with pytest.raises(ValueError, match="^SqueezeNetV1: num_classes must be a positive integer"):
            SqueezeNetV1(num_classes=value, input_shape=(64, 64, 3))

    def test_a_numpy_integer_num_classes_is_stored_as_an_int(self):
        model = SqueezeNetV1(num_classes=np.int64(4), input_shape=(64, 64, 3))
        assert type(model.num_classes) is int and model.num_classes == 4
        json.dumps(model.get_config())

    def test_an_unknown_variant_is_refused_by_the_constructor(self):
        with pytest.raises(ValueError, match=r"Unknown variant 'nope'. Available variants: \['1.0', '1.1', '1.0_bypass'\]"):
            SqueezeNetV1(variant="nope")

    @pytest.mark.parametrize("bad", ["bogus", True, 1, "Simple", ["simple"]])
    def test_a_use_bypass_that_is_not_a_bypass_mode_is_refused_by_value(self, bad):
        """``True`` and a misspelling used to build a plain model that reported the value."""
        with pytest.raises(ValueError, match="use_bypass") as caught:
            SqueezeNetV1(use_bypass=bad, input_shape=(64, 64, 3))
        assert repr(bad) in str(caught.value)

    @pytest.mark.parametrize("mode, adds", [(None, 0), (False, 0), ("simple", 4), ("complex", 8)])
    def test_every_bypass_mode_builds_its_own_wiring(self, mode, adds):
        model = SqueezeNetV1(use_bypass=mode, input_shape=(64, 64, 3))
        assert len([l for l in model.layers if isinstance(l, keras.layers.Add)]) == adds

    @pytest.mark.parametrize(
        "variant, expected_conv1_filters, expected_bypass, expected_pool_indices",
        [
            ("1.0", 96, False, [1, 4, 8]),
            ("1.1", 64, False, [1, 3, 5]),
            ("1.0_bypass", 96, "simple", [1, 4, 8]),
        ],
    )
    def test_model_variants(self, variant, expected_conv1_filters, expected_bypass, expected_pool_indices, num_classes):
        """Test all predefined model variants."""
        model = SqueezeNetV1.from_variant(variant, num_classes=num_classes)

        assert model.num_classes == num_classes
        assert model.variant == variant
        assert model.conv1_filters == expected_conv1_filters
        assert model.use_bypass == expected_bypass
        assert model.pool_indices == expected_pool_indices

    def test_invalid_variant(self, num_classes):
        """Test invalid variant raises error."""
        with pytest.raises(ValueError, match="Unknown variant"):
            SqueezeNetV1.from_variant("invalid_variant", num_classes=num_classes)

    def test_forward_pass_with_top(self, num_classes, sample_data):
        """Test forward pass with classification head."""
        model = SqueezeNetV1.from_variant("1.0", num_classes=num_classes)
        outputs = model(sample_data)

        # Check output shape
        expected_shape = (sample_data.shape[0], num_classes)
        assert outputs.shape == expected_shape

        # Logits: finite, non-negative (conv10 carries a ReLU before the pool),
        # and not a probability vector.
        values = outputs.numpy()
        assert np.all(np.isfinite(values))
        assert np.all(values >= 0.0)
        assert not np.allclose(values.sum(axis=-1), 1.0, atol=1e-3)

    def test_forward_pass_without_top(self, small_input_shape, small_sample_data):
        """Test forward pass without classification head (feature extraction)."""
        model = SqueezeNetV1(
            include_top=False,
            input_shape=small_input_shape
        )

        features = model(small_sample_data)

        # Output should be 4D tensor: (batch, height, width, channels)
        assert len(features.shape) == 4
        # The number of output channels is e1x1 + e3x3 of the last Fire module (fire9)
        last_fire_config = SqueezeNetV1.MODEL_VARIANTS["1.0"]["fire_configs"][-1]
        expected_channels = last_fire_config['e1x1'] + last_fire_config['e3x3']
        assert features.shape[-1] == expected_channels

        # Check for valid values
        assert not np.any(np.isnan(features.numpy()))
        assert not np.any(np.isinf(features.numpy()))

    def test_different_input_shapes(self, num_classes):
        """Test with different input shapes."""
        test_shapes = [
            (32, 32, 3),
            (64, 64, 3),
            (96, 96, 1),  # Grayscale
        ]

        for shape in test_shapes:
            # v1.1 pools later, so it keeps more spatial resolution on small
            # inputs. (The 2026-08-15 note that "'1.0' at (32, 32, 3) does build
            # and forward" was true and MISLEADING: it forwarded an all-NaN
            # tensor of the correct shape. "1.0" walks 32 -> 13 -> 6 -> 2 -> 0
            # and its computed floor is 35; "1.1"'s is 31. Since 2026-08-18 the
            # constructor raises below the floor, so this selection is load-
            # bearing, not a resolution preference.)
            variant = "1.1" if shape[0] < 64 else "1.0"

            model = SqueezeNetV1.from_variant(
                variant,
                num_classes=num_classes,
                input_shape=shape
            )
            batch_data = tf.random.uniform([2] + list(shape), 0, 1)
            outputs = model(batch_data)
            assert outputs.shape == (2, num_classes)

    def test_bypass_configurations(self, num_classes, small_input_shape, small_sample_data):
        """Test different bypass configurations build and run."""
        bypass_options = [False, "simple", "complex"]

        for bypass_type in bypass_options:
            model = SqueezeNetV1(
                num_classes=num_classes,
                use_bypass=bypass_type,
                input_shape=small_input_shape
            )
            outputs = model(small_sample_data, training=True)
            assert outputs.shape == (small_sample_data.shape[0], num_classes)

    def test_model_compilation(self, num_classes, cifar_input_shape):
        """Test model compilation with different optimizers and losses."""
        model = SqueezeNetV1(num_classes=num_classes, input_shape=cifar_input_shape, variant="1.1")

        optimizers = [keras.optimizers.Adam(), keras.optimizers.SGD()]
        for optimizer in optimizers:
            model.compile(
                optimizer=optimizer,
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=["accuracy"]
            )
            assert model.optimizer == optimizer

    def test_serialization(self, num_classes):
        """Test serialization and deserialization of the model."""
        original_model = SqueezeNetV1.from_variant(
            "1.0_bypass",
            num_classes=num_classes,
            dropout_rate=0.3,
            kernel_regularizer=keras.regularizers.L2(0.02)
        )

        config = original_model.get_config()
        recreated_model = SqueezeNetV1.from_config(config)

        # Check configuration matches
        assert recreated_model.num_classes == original_model.num_classes
        assert recreated_model.use_bypass == original_model.use_bypass
        assert recreated_model.dropout_rate == original_model.dropout_rate
        assert recreated_model.get_config()['kernel_regularizer']['config'] == \
               original_model.get_config()['kernel_regularizer']['config']
        assert config['variant'] == '1.0_bypass'
        assert recreated_model.variant == original_model.variant

    def test_get_config_is_json_safe_with_an_initializer_instance_and_from_config_leaves_the_dict_alone(
            self, num_classes):
        import copy
        import json
        model = SqueezeNetV1(num_classes=num_classes, input_shape=(64, 64, 3),
                             kernel_initializer=keras.initializers.HeNormal())
        config = model.get_config()
        json.dumps(config)
        given = copy.deepcopy(config)
        SqueezeNetV1.from_config(given)
        assert given == config

    def test_model_save_load(self, num_classes, cifar_input_shape, cifar_sample_data):
        """Test saving and loading a SqueezeNet model."""
        model = SqueezeNetV1(
            num_classes=num_classes,
            input_shape=cifar_input_shape,
            name="test_squeezenet",
            variant="1.1"
        )
        model.compile(
            optimizer="adam",
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        )

        original_outputs = model(cifar_sample_data, training=False)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "squeezenet_model.keras")
            # The optimizer's slot variables are allocated lazily, so a compiled-but-
            # unfitted model would otherwise save an optimizer the reload cannot match.
            # See tests/optimizer_state.py (D-016).
            build_optimizer_state(model)
            model.save(model_path)

            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={"SqueezeNetV1": SqueezeNetV1, "FireModule": FireModule}
            )

            loaded_outputs = loaded_model(cifar_sample_data, training=False)

            assert original_outputs.shape == loaded_outputs.shape
            np.testing.assert_allclose(original_outputs.numpy(), loaded_outputs.numpy(), rtol=1e-6)
            assert loaded_model.count_params() == model.count_params()

    def test_training_integration(self, num_classes, small_input_shape, small_sample_data):
        """Test training integration with a small dataset."""
        model = SqueezeNetV1(num_classes=num_classes, input_shape=small_input_shape)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        )

        labels = tf.random.uniform([small_sample_data.shape[0]], 0, num_classes, dtype=tf.int32)
        dataset = tf.data.Dataset.from_tensor_slices((small_sample_data, labels)).batch(4)

        history = model.fit(dataset, epochs=2, verbose=0)

        assert "loss" in history.history
        assert "accuracy" in history.history
        assert len(history.history["loss"]) == 2

        initial_loss = history.history["loss"][0]
        final_loss = history.history["loss"][-1]
        assert not np.isnan(final_loss)
        assert final_loss < initial_loss * 2

    def test_gradient_flow(self, num_classes, small_input_shape, small_sample_data):
        """Test gradient flow through the model."""
        model = SqueezeNetV1(num_classes=num_classes, input_shape=small_input_shape)
        labels = tf.random.uniform([small_sample_data.shape[0]], 0, num_classes, dtype=tf.int32)

        with tf.GradientTape() as tape:
            outputs = model(small_sample_data, training=True)
            loss = keras.losses.sparse_categorical_crossentropy(
                labels, outputs, from_logits=True)
            loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss, model.trainable_variables)

        assert all(g is not None for g in grads)
        assert np.any(grads[0].numpy() != 0)
        assert np.any(grads[-2].numpy() != 0)

    def test_create_squeezenet_v1_factory(self, num_classes):
        """Test the create_squeezenet_v1 factory function."""
        model = create_squeezenet_v1(
            variant="1.1",
            num_classes=num_classes,
            dropout_rate=0.4
        )
        assert isinstance(model, SqueezeNetV1)
        assert model.num_classes == num_classes
        assert model.conv1_filters == SqueezeNetV1.MODEL_VARIANTS["1.1"]["conv1_filters"]
        assert model.dropout_rate == 0.4

    def test_a_weights_keyword_is_refused_by_name(self, num_classes):
        """``weights=`` is not an argument: it must not be swallowed into a random model."""
        with pytest.raises(ValueError, match="Unrecognized keyword arguments.*weights"):
            create_squeezenet_v1(
                variant="1.0", num_classes=num_classes, weights="imagenet"
            )

    def test_pretrained_true_raises_naming_the_variant(self, num_classes):
        with pytest.raises(NotImplementedError, match="SqueezeNetV1 variant '1.0'"):
            create_squeezenet_v1(variant="1.0", num_classes=num_classes, pretrained=True)

    def test_numerical_stability(self, num_classes, small_input_shape):
        """Test model stability with extreme input values."""
        # MODIFIED: Use a larger input shape to prevent pooling errors.
        model = SqueezeNetV1(num_classes=num_classes, input_shape=small_input_shape)

        test_cases = [
            tf.zeros((2,) + small_input_shape),
            tf.ones((2,) + small_input_shape),
            tf.random.uniform((2,) + small_input_shape, 0, 100.0), # More reasonable large values
        ]

        for i, test_input in enumerate(test_cases):
            outputs = model(test_input)
            # MODIFIED: Add a check for infinity.
            assert not np.any(np.isnan(outputs.numpy())), f"NaN in outputs for test case {i}"
            assert not np.any(np.isinf(outputs.numpy())), f"Inf in outputs for test case {i}"

    def test_batch_size_independence(self, num_classes, cifar_input_shape):
        """Test that model works with different batch sizes."""
        # Using the less aggressive v1.1 since the input is small
        model = SqueezeNetV1(num_classes=num_classes, input_shape=cifar_input_shape, variant="1.1")

        batch_sizes = [1, 4, 8]
        for batch_size in batch_sizes:
            test_data = tf.random.uniform((batch_size,) + cifar_input_shape, 0, 1)
            outputs = model(test_data)
            assert outputs.shape == (batch_size, num_classes)


# Tests for the FireModule itself
class TestFireModule:
    """Test suite for the FireModule layer."""

    @pytest.fixture
    def fire_module_config(self) -> Dict[str, Any]:
        return {'s1x1': 16, 'e1x1': 64, 'e3x3': 64}

    def test_initialization(self, fire_module_config):
        """Test FireModule initialization."""
        module = FireModule(**fire_module_config)
        assert module.s1x1 == fire_module_config['s1x1']
        assert module.e1x1 == fire_module_config['e1x1']
        assert module.e3x3 == fire_module_config['e3x3']
        assert module.squeeze.filters == fire_module_config['s1x1']
        assert module.expand_1x1.filters == fire_module_config['e1x1']
        assert module.expand_3x3.filters == fire_module_config['e3x3']

    def test_invalid_initialization(self):
        """Test that invalid filter counts raise an error."""
        with pytest.raises(ValueError):
            FireModule(s1x1=0, e1x1=1, e3x3=1)
        with pytest.raises(ValueError):
            FireModule(s1x1=1, e1x1=-1, e3x3=1)

    def test_forward_pass_and_shape(self, fire_module_config):
        """Test the forward pass and output shape of the FireModule."""
        input_shape = (4, 32, 32, 96) # batch, h, w, c
        inputs = tf.random.normal(input_shape)

        module = FireModule(**fire_module_config)
        outputs = module(inputs)

        expected_channels = fire_module_config['e1x1'] + fire_module_config['e3x3']
        expected_shape = (input_shape[0], input_shape[1], input_shape[2], expected_channels)

        assert outputs.shape == expected_shape

    def test_the_concatenation_layer_is_named(self, fire_module_config):
        """An unnamed layer takes the process-global auto-name counter (``concatenate_7``)."""
        module = FireModule(**fire_module_config)
        assert module.concat.name == "concat"
        assert sorted(l.name for l in module._flatten_layers(include_self=False)) == [
            "concat", "expand_1x1", "expand_3x3", "squeeze"]

    def test_an_initializer_instance_is_stored_as_its_spec_and_no_two_convs_replay_a_draw(
            self, fire_module_config):
        initializer = keras.initializers.HeNormal()
        module = FireModule(**fire_module_config, kernel_initializer=initializer)
        assert module.kernel_initializer == keras.initializers.serialize(initializer)
        keras.utils.set_random_seed(0)
        module.build((None, 8, 8, 128))
        pairs, compared = replayed_pairs(module)
        assert compared > 0 and pairs == [], pairs

    def test_serialization(self, fire_module_config):
        """Test serialization and deserialization of the FireModule."""
        module = FireModule(**fire_module_config)
        config = module.get_config()

        recreated_module = FireModule.from_config(config)

        assert recreated_module.s1x1 == module.s1x1
        assert recreated_module.e1x1 == module.e1x1
        assert recreated_module.e3x3 == module.e3x3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])