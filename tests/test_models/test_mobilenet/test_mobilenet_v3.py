"""
Comprehensive test suite for MobileNetV3 model implementation.

Following the Complete Guide to Modern Keras 3 Custom Layers and Models,
this test suite ensures a robust, serializable, and production-ready implementation
with thorough validation of all model functionality.

Test Categories:
- Serialization cycle tests (CRITICAL)
- Model initialization and building
- Forward pass validation
- Configuration completeness
- Gradient flow verification
- Training mode behavior
- Variant creation and validation
- Edge cases and error handling
- Performance benchmarks
"""

import copy
import pytest
import tempfile
import os
import numpy as np
from typing import Dict, Any

import keras
import tensorflow as tf

from tests.numerics import reassociation_atol

from dl_techniques.models.vision.mobilenet.mobilenet_v3 import MobileNetV3, create_mobilenetv3


def make_divisible(value: float, divisor: int = 8) -> int:
    """Helper function to replicate width multiplier logic."""
    new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


class TestMobileNetV3:
    """Comprehensive test suite for MobileNetV3 model."""

    @pytest.fixture
    def large_config(self) -> Dict[str, Any]:
        """Standard configuration for MobileNetV3-Large."""
        return {
            'num_classes': 10,
            'variant': 'large',
            'width_multiplier': 1.0,
            'dropout_rate': 0.1,
            'weight_decay': 1e-5,
            'input_shape': (32, 32, 3),
            'include_top': True
        }

    @pytest.fixture
    def small_config(self) -> Dict[str, Any]:
        """Configuration for MobileNetV3-Small."""
        return {
            'num_classes': 100,
            'variant': 'small',
            'width_multiplier': 0.75,
            'dropout_rate': 0.1,
            'weight_decay': 1e-5,
            'input_shape': (64, 64, 3),
            'include_top': True
        }

    @pytest.fixture
    def sample_inputs(self) -> Dict[str, keras.KerasTensor]:
        """Sample inputs for different test scenarios."""
        return {
            'cifar': keras.random.normal(shape=(4, 32, 32, 3)),
            'imagenet': keras.random.normal(shape=(2, 224, 224, 3)),
            'custom': keras.random.normal(shape=(3, 64, 64, 3)),
            'grayscale': keras.random.normal(shape=(2, 28, 28, 1))
        }

    # ============================================================================
    # Core Tests - Initialization and Building
    # ============================================================================

    def test_width_multiplier_scaling(self):
        """Test width multiplier correctly scales layer dimensions."""
        multiplier = 0.75
        model = MobileNetV3(
            variant='large',
            width_multiplier=multiplier,
            input_shape=(32, 32, 3)
        )

        # Check stem convolution
        expected_stem_filters = make_divisible(16 * multiplier)
        assert model.stem.conv1.filters == expected_stem_filters

        # Check a specific block's output channels (e.g., first block of Large)
        expected_block0_out = make_divisible(16 * multiplier)
        assert model.blocks[0].filters == expected_block0_out

        # Check last convolution block
        expected_last_conv = make_divisible(960 * multiplier)
        assert model.last_conv.filters == expected_last_conv

    def test_model_building_on_call(self, large_config, sample_inputs):
        """Test that model builds correctly on first call."""
        config_no_shape = large_config.copy()
        del config_no_shape['input_shape']
        model = MobileNetV3(**config_no_shape)
        assert not model.built

        # First call should trigger building with default shape
        output = model(sample_inputs['imagenet'])
        assert model.built
        assert not isinstance(output, dict)
        assert output.shape == (2, large_config['num_classes'])

    # ============================================================================
    # Critical Serialization Tests
    # ============================================================================

    def test_serialization_cycle_large(self, large_config, sample_inputs):
        """CRITICAL TEST: Full serialization cycle for MobileNetV3-Large."""
        model = MobileNetV3(**large_config)

        # Get prediction from original model
        original_pred = model(sample_inputs['cifar'])

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'mobilenetv3_large.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_inputs['cifar'])

            # Verify identical predictions (compare logits)
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="MobileNetV3-Large predictions differ after serialization"
            )

    def test_serialization_cycle_small(self, small_config, sample_inputs):
        """CRITICAL TEST: Full serialization cycle for MobileNetV3-Small."""
        model = MobileNetV3(**small_config)

        # Get prediction from original model
        original_pred = model(sample_inputs['custom'])

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'mobilenetv3_small.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_inputs['custom'])

            # Verify identical predictions (compare logits)
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="MobileNetV3-Small predictions differ after serialization"
            )

    def test_serialization_without_top(self, sample_inputs):
        """Test serialization of feature extractor (include_top=False)."""
        model = MobileNetV3(
            variant='small',
            include_top=False,
            input_shape=(32, 32, 3)
        )

        # Get features from original model
        original_features = model(sample_inputs['cifar'])

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'mobilenetv3_features.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_features = loaded_model(sample_inputs['cifar'])

            # Verify identical features
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_features),
                keras.ops.convert_to_numpy(loaded_features),
                rtol=1e-6, atol=1e-6,
                err_msg="Feature extractor outputs differ after serialization"
            )

    # ============================================================================
    # Configuration and Variant Tests
    # ============================================================================

    def test_config_completeness(self, large_config):
        """Test that get_config contains all initialization parameters."""
        model = MobileNetV3(**large_config)
        config = model.get_config()

        required_keys = [
            'num_classes', 'variant', 'width_multiplier', 'dropout_rate',
            'weight_decay', 'kernel_initializer', 'include_top', 'input_shape'
        ]

        for key in required_keys:
            assert key in config, f"Missing {key} in get_config()"

    def test_from_config_reconstruction(self, large_config):
        """Test model reconstruction from configuration."""
        model = MobileNetV3(**large_config)
        config = model.get_config()

        # Reconstruct model from config
        reconstructed_model = MobileNetV3.from_config(config)

        # Verify configurations match
        assert reconstructed_model.num_classes == model.num_classes
        assert reconstructed_model.variant == model.variant
        assert reconstructed_model.width_multiplier == model.width_multiplier
        assert reconstructed_model.dropout_rate == model.dropout_rate

    @pytest.mark.parametrize("variant", ["large", "small"])
    def test_variant_creation(self, variant):
        """Test creation of all predefined variants."""
        model = MobileNetV3.from_variant(variant, num_classes=10, input_shape=(32, 32, 3))

        # Check variant-specific properties
        if variant == "large":
            assert len(model.blocks) == len(MobileNetV3.LARGE_CONFIG)
            assert model.last_conv.filters == 960
        else:
            assert len(model.blocks) == len(MobileNetV3.SMALL_CONFIG)
            assert model.last_conv.filters == 576

    def test_convenience_function(self):
        """Test create_mobilenetv3 convenience function."""
        model = create_mobilenetv3(
            variant="small",
            num_classes=10,
            input_shape=(32, 32, 3),
            width_multiplier=0.75
        )

        assert model.num_classes == 10
        assert model.variant == "small"
        assert model.width_multiplier == 0.75
        assert model.input_shape_config == (32, 32, 3)

    # ============================================================================
    # Forward Pass and Output Shape Tests
    # ============================================================================

    def test_forward_pass_shapes(self, sample_inputs):
        """Test forward pass produces correct output shapes."""
        # Large model
        model = MobileNetV3.from_variant("large", num_classes=10, input_shape=(32, 32, 3))
        output = model(sample_inputs['cifar'])
        assert not isinstance(output, dict)
        assert output.shape == (4, 10)

        # Feature extractor
        model = MobileNetV3.from_variant("small", include_top=False, input_shape=(32, 32, 3))
        features = model(sample_inputs['cifar'])
        assert len(features.shape) == 4  # (batch, height, width, channels)

    def test_different_input_sizes(self):
        """Test model with different input sizes."""
        test_configs = [
            ((28, 28, 1), (2, 28, 28, 1)),  # MNIST-like
            ((64, 64, 3), (3, 64, 64, 3)),  # Medium resolution
            ((224, 224, 3), (1, 224, 224, 3)),  # ImageNet-like
        ]

        for input_shape, sample_shape in test_configs:
            model = MobileNetV3.from_variant("small", num_classes=5, input_shape=input_shape)
            sample_input = keras.random.normal(shape=sample_shape)

            output = model(sample_input)
            assert not isinstance(output, dict)
            assert output.shape == (sample_shape[0], 5)

    def test_batch_size_handling(self):
        """Test model handles different batch sizes correctly."""
        model = MobileNetV3.from_variant("small", num_classes=10, input_shape=(32, 32, 3))

        batch_sizes = [1, 4, 8, 16]
        for batch_size in batch_sizes:
            sample_input = keras.random.normal(shape=(batch_size, 32, 32, 3))
            output = model(sample_input)
            assert not isinstance(output, dict)
            assert output.shape == (batch_size, 10)

    # ============================================================================
    # Training and Gradient Tests
    # ============================================================================

    def test_gradients_flow(self, large_config, sample_inputs):
        """Test gradient computation and backpropagation."""
        model = MobileNetV3(**large_config)

        with tf.GradientTape() as tape:
            output = model(sample_inputs['cifar'], training=True)
            loss = keras.ops.mean(keras.ops.square(output))

        gradients = tape.gradient(loss, model.trainable_variables)

        # The head builds only what its call() applies (plan-2026-10-14
        # step 1), so every trainable weight is on the gradient path.
        missing = [v.path for v, g in zip(model.trainable_variables, gradients)
                   if g is None]
        assert not missing, f"trainable weights with no gradient: {missing}"
        non_zero_grads = [g for g in gradients if keras.ops.max(keras.ops.abs(g)) > 1e-8]
        assert len(non_zero_grads) > 0, "All gradients are zero"

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, large_config, sample_inputs, training):
        """Test behavior in different training modes."""
        model = MobileNetV3(**large_config)
        output = model(sample_inputs['cifar'], training=training)
        assert not isinstance(output, dict)
        assert output.shape == (4, large_config['num_classes'])

    def test_model_compilation_and_fit(self, sample_inputs):
        """Test model compiles and can run a training step."""
        model = MobileNetV3.from_variant("small", num_classes=10, input_shape=(32, 32, 3))

        model.compile(
            optimizer='adam',
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        x_train = sample_inputs['cifar']
        y_train = np.random.randint(0, 10, size=(4,))

        history = model.fit(x_train, y_train, epochs=1, verbose=0)

        assert 'loss' in history.history
        assert len(history.history['loss']) == 1

    # ============================================================================
    # Edge Cases and Error Handling
    # ============================================================================

    def test_invalid_configurations(self):
        """Test that invalid configurations raise appropriate errors."""
        # Invalid variant name
        with pytest.raises(ValueError, match="Unknown variant"):
            MobileNetV3(variant="medium", input_shape=(32, 32, 3))

        # Invalid input shape
        with pytest.raises(ValueError, match="input_shape must be 3D"):
            MobileNetV3(input_shape=(32, 32))

    def test_unknown_variant(self):
        """Test error handling for unknown variants in factory method."""
        with pytest.raises(ValueError, match="Unknown variant"):
            MobileNetV3.from_variant("nonexistent_variant")

    # ============================================================================
    # Performance and Memory Tests
    # ============================================================================

    def test_model_parameter_counts(self):
        """Exact counts at 224 x 224 x 3, 1000 classes, re-derived by command.

        The head is Keras Applications' (GAP, Dense 1280/1024 with a bias and
        no norm, hard-swish, dropout, Dense; D-015). The totals sit 6,584
        (large) and 8,168 (small) below ``keras.applications.MobileNetV3*``
        because of the backbone differences this module documents (the
        universal block's squeeze-excitation layout and the large first row's
        extra projection), not the head. MEASURED 2026-10-14
        (plan-2026-10-14T062616-81938e81 step 3) on CPU; the two-DenseBlock
        head of 2026-10-09..14 had been pinned at ~1.24M / ~3.4M.
        Step 9.3 (D-226): the SE bottleneck rounds as the reference does
        (``make_divisible(expanded / 4, 8)``), +5,600 (small) and +1,824
        (large), the hand sum ``2 * expanded * (new - old)`` over the SE
        blocks in ``test_the_v3_squeeze_excite_widths_are_the_references.py``.
        Step 9.3.1 (D-227): the SE convs carry biases, +2,888 (small) and
        +5,080 (large); the totals were ``keras.applications`` plus the
        first row's 320. Step 10.3 (D-232): the first row has no expansion
        conv, -320; the totals equal ``keras.applications.MobileNetV3*``
        (``test_the_v3_widths_follow_keras_applications_at_every_width.py``).
        """
        expected = {
            'small': 2_554_968,
            'large': 5_507_432,
        }
        for variant, count in expected.items():
            model = MobileNetV3.from_variant(variant, num_classes=1000)
            model(keras.random.normal(shape=(1, 224, 224, 3)))
            assert model.count_params() == count, (
                f"{variant}: {model.count_params():,} != {count:,}")

    def test_model_summary_execution(self):
        """Test that model summary executes without errors."""
        model = MobileNetV3.from_variant("small", num_classes=10, input_shape=(32, 32, 3))
        model(keras.random.normal(shape=(1, 32, 32, 3)))

        try:
            model.summary()
        except Exception as e:
            pytest.fail(f"Model summary failed with error: {e}")

    # ============================================================================
    # Integration Tests
    # ============================================================================

    def test_model_in_training_loop(self):
        """Test model in a realistic training scenario."""
        model = MobileNetV3.from_variant("small", num_classes=2, input_shape=(32, 32, 3))
        model.compile(
            optimizer='adam',
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        x_train = keras.random.normal(shape=(20, 32, 32, 3))
        y_train = np.random.randint(0, 2, size=(20,))
        x_val = keras.random.normal(shape=(10, 32, 32, 3))
        y_val = np.random.randint(0, 2, size=(10,))

        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=2,
            batch_size=4,
            verbose=0
        )

        assert len(history.history['loss']) == 2
        assert 'val_loss' in history.history

    def test_model_evaluation_and_prediction(self):
        """Test model evaluation and prediction methods."""
        model = MobileNetV3.from_variant("small", num_classes=5, input_shape=(32, 32, 3))
        model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

        x_test = keras.random.normal(shape=(10, 32, 32, 3))
        y_test = np.random.randint(0, 5, size=(10,))

        results = model.evaluate(x_test, y_test, verbose=0)
        assert len(results) == 2  # loss and accuracy

        predictions = model.predict(x_test, verbose=0)
        assert predictions.shape == (10, 5)

        # The model returns logits: the rows are not normalized, their softmax
        # is. A softmax row sums to 1 within reassociation_atol([5], 1, 1.0).
        atol = reassociation_atol([5], 1, 1.0)
        rows = np.asarray(predictions, dtype=np.float64).sum(axis=1)
        assert np.any(np.abs(rows - 1.0) > atol), (
            f"every predicted row sums to 1 within {atol:.2e}: {rows}")
        probs = keras.ops.convert_to_numpy(keras.ops.softmax(predictions, axis=-1))
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(10), rtol=0, atol=atol)

    def test_a_mobilenet_reports_the_shape_it_runs(self):
        """``compute_output_shape`` and a symbolic call agree with the eager forward.

        The stem is ``self.stem`` (a ``ConvStem``); ``compute_output_shape``
        must read it and not the sub-layers it used to alias. Keras uses a
        model's ``compute_output_shape`` for a symbolic call, so a stale read
        raises ``AttributeError`` on both paths while every test that only runs
        the eager forward stays green.
        """
        model = MobileNetV3(
            variant="small",
            num_classes=4,
            include_top=False,
            input_shape=(32, 32, 3),
        )
        batch = np.zeros((1, 32, 32, 3), "float32")
        eager = model(batch, training=False)

        reported = model.compute_output_shape((None, 32, 32, 3))
        assert tuple(reported) == (None,) + tuple(eager.shape[1:])

        symbolic = model(keras.Input(shape=(32, 32, 3)))
        assert tuple(symbolic.shape) == (None,) + tuple(eager.shape[1:])


class TestMobileNetV3HouseShape:
    """The constructor, ``from_variant`` and ``build`` contract of the house shape."""

    @pytest.mark.parametrize("variant,count", [
        ("small", 2_554_968),
        ("large", 5_507_432),
    ])
    def test_an_explicit_build_creates_every_weight(self, variant, count):
        """``build`` materializes what ``call`` runs.

        The model used to inherit Keras' ``build``, which marked it built with
        0 weights and a warning.
        """
        model = MobileNetV3.from_variant(variant, num_classes=1000)
        model.build((None, 64, 64, 3))
        assert model.count_params() == count

    def test_an_override_does_not_poison_the_variant_table(self):
        before = copy.deepcopy(MobileNetV3.MODEL_VARIANTS)
        MobileNetV3.from_variant("small", width_multiplier=0.5, dropout_rate=0.3)
        assert MobileNetV3.MODEL_VARIANTS == before

    def test_the_constructor_lists_the_variants_it_knows(self):
        with pytest.raises(ValueError) as caught:
            MobileNetV3(variant="medium")
        assert "Unknown variant 'medium'. Available variants: ['large', 'small']" in str(
            caught.value)

    @pytest.mark.parametrize("kwargs,message", [
        ({"width_multiplier": 0.0}, "width_multiplier must be positive, got 0.0"),
        ({"width_multiplier": -1.0}, "width_multiplier must be positive, got -1.0"),
        ({"dropout_rate": 1.0}, "dropout_rate must be in [0, 1), got 1.0"),
        ({"dropout_rate": -0.1}, "dropout_rate must be in [0, 1), got -0.1"),
        ({"num_classes": 0}, "num_classes must be positive when include_top=True, got 0"),
    ])
    def test_an_invalid_value_is_refused_naming_the_argument_and_the_value(
            self, kwargs, message):
        with pytest.raises(ValueError) as caught:
            MobileNetV3(input_shape=(32, 32, 3), **kwargs)
        assert message in str(caught.value), str(caught.value)

    def test_a_headless_model_needs_no_class_count(self):
        model = MobileNetV3(num_classes=0, include_top=False, input_shape=(32, 32, 3))
        assert model.classification_head is None

    def test_the_last_block_is_three_named_layers(self):
        model = MobileNetV3(variant="small", input_shape=(32, 32, 3))
        assert [model.last_conv.name, model.last_bn.name, model.last_activation.name] == [
            "last_conv", "last_bn", "last_hard_swish"]
        assert model.last_conv.filters == 576


class TestMobileNetV3PretrainedTrunk:
    """D-040: a pretrained load restores every layer outside the head.

    Only ``classification_head`` may differ. A trunk of another variant used to
    load the layers that happened to fit, report the rest and return normally.
    """

    @staticmethod
    def _saved(tmp_path, variant, **kwargs):
        donor = MobileNetV3.from_variant(
            variant, num_classes=10, input_shape=(32, 32, 3), **kwargs
        )
        donor.build((None, 32, 32, 3))
        path = os.path.join(str(tmp_path), "donor.keras")
        donor.save(path)
        return donor, path

    @staticmethod
    def _by_path(model):
        return {w.path.split("/", 1)[1]: keras.ops.convert_to_numpy(w)
                for w in model.weights}

    def test_a_checkpoint_of_another_variant_is_refused(self, tmp_path):
        _, path = self._saved(tmp_path, "large")

        with pytest.raises(ValueError) as caught:
            MobileNetV3.from_variant(
                "small", num_classes=10, input_shape=(32, 32, 3), pretrained=path
            )

        message = str(caught.value)
        assert "shape_mismatch" in message and "missing_in_source" in message, message

    def test_a_refused_load_writes_nothing(self, tmp_path):
        _, path = self._saved(tmp_path, "large")
        receiver = MobileNetV3.from_variant(
            "small", num_classes=10, input_shape=(32, 32, 3)
        )
        before = [keras.ops.convert_to_numpy(w).copy() for w in receiver.weights]

        with pytest.raises(ValueError):
            receiver.load_pretrained_weights(path)

        for a, w in zip(before, receiver.weights):
            np.testing.assert_array_equal(a, keras.ops.convert_to_numpy(w))

    def test_another_class_count_loads_the_trunk_and_skips_the_head(self, tmp_path):
        donor, path = self._saved(tmp_path, "small")

        loaded = MobileNetV3.from_variant(
            "small", num_classes=3, input_shape=(32, 32, 3), pretrained=path
        )

        src, dst = self._by_path(donor), self._by_path(loaded)
        trunk = {p: v for p, v in dst.items() if not p.startswith("classification_head")}
        assert trunk
        for p, value in trunk.items():
            np.testing.assert_array_equal(value, src[p], err_msg=p)
        heads = {p: v for p, v in dst.items() if p.startswith("classification_head")}
        assert heads and any(v.shape != src[p].shape for p, v in heads.items())

    def test_a_headless_model_loads_from_a_headed_checkpoint(self, tmp_path):
        donor, path = self._saved(tmp_path, "small")

        headless = MobileNetV3.from_variant(
            "small", include_top=False, input_shape=(32, 32, 3), pretrained=path
        )

        src, dst = self._by_path(donor), self._by_path(headless)
        assert dst
        for p, value in dst.items():
            np.testing.assert_array_equal(value, src[p], err_msg=p)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
