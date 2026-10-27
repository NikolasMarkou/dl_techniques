"""
Comprehensive test suite for MobileNetV1 model implementation.

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
import inspect
import json
import pytest
import re
import tempfile
import os
import numpy as np
from typing import Dict, Any

import keras
import tensorflow as tf

from tests.numerics import reassociation_atol

from dl_techniques.initializers.clone import clone_initializer
from dl_techniques.models.vision.mobilenet.mobilenet_v1 import MobileNetV1, create_mobilenetv1
from dl_techniques.utils.weight_transfer import load_pretrained_checkpoint


class TestMobileNetV1:
    """Comprehensive test suite for MobileNetV1 model."""

    @pytest.fixture
    def default_config(self) -> Dict[str, Any]:
        """Standard configuration for testing."""
        return {
            'num_classes': 10,
            'width_multiplier': 1.0,
            'dropout_rate': 0.001,
            'weight_decay': 1e-5,
            'input_shape': (32, 32, 3),
            'include_top': True
        }

    @pytest.fixture
    def small_config(self) -> Dict[str, Any]:
        """Configuration for a smaller model."""
        return {
            'num_classes': 100,
            'width_multiplier': 0.25,
            'dropout_rate': 0.0,
            'weight_decay': 0.0,
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

    def test_initialization(self, default_config):
        """Test model initialization with standard parameters."""
        model = MobileNetV1(**default_config)
        assert model.num_classes == default_config['num_classes']
        assert model.width_multiplier == default_config['width_multiplier']
        assert model.include_top == default_config['include_top']
        assert model.input_shape_config == default_config['input_shape']

    def test_width_multiplier_scaling(self):
        """Test width multiplier correctly scales filter dimensions."""
        base_filters = 32  # From the initial conv layer
        multiplier = 0.5

        model = MobileNetV1(width_multiplier=multiplier, input_shape=(32, 32, 3))
        expected_filters = int(base_filters * multiplier)
        assert model.stem.conv1.filters == expected_filters

        # Check a later block
        # Block 1 in ARCHITECTURE has 64 filters
        base_filters_block1 = 64
        expected_filters_block1 = int(base_filters_block1 * multiplier)
        assert model.blocks[0].pointwise_conv.filters == expected_filters_block1

    def test_model_building_on_call(self, default_config, sample_inputs):
        """Test that model builds correctly on first call."""
        config_no_shape = default_config.copy()
        del config_no_shape['input_shape']
        model = MobileNetV1(**config_no_shape)
        assert not model.built

        # First call should trigger building
        output = model(sample_inputs['imagenet'])
        assert model.built
        assert not isinstance(output, dict)
        assert output.shape == (2, default_config['num_classes'])

    # ============================================================================
    # Critical Serialization Tests
    # ============================================================================

    def test_serialization_cycle_standard(self, default_config, sample_inputs):
        """CRITICAL TEST: Full serialization cycle for a standard model."""
        model = MobileNetV1(**default_config)
        original_pred = model(sample_inputs['cifar'])

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'mobilenetv1_standard.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_inputs['cifar'])

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Standard model predictions differ after serialization"
            )

    def test_serialization_cycle_small_variant(self, small_config, sample_inputs):
        """CRITICAL TEST: Full serialization cycle for a small variant model."""
        model = MobileNetV1(**small_config)
        original_pred = model(sample_inputs['custom'])

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'mobilenetv1_small.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_inputs['custom'])

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Small variant model predictions differ after serialization"
            )

    def test_serialization_without_top(self, sample_inputs):
        """Test serialization of feature extractor (include_top=False)."""
        model = MobileNetV1(include_top=False, input_shape=(32, 32, 3))
        original_features = model(sample_inputs['cifar'])

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'mobilenetv1_features.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_features = loaded_model(sample_inputs['cifar'])

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_features),
                keras.ops.convert_to_numpy(loaded_features),
                rtol=1e-6, atol=1e-6,
                err_msg="Feature extractor outputs differ after serialization"
            )

    # ============================================================================
    # Configuration and Variant Tests
    # ============================================================================

    def test_config_completeness(self, default_config):
        """Test that get_config contains all initialization parameters."""
        model = MobileNetV1(**default_config)
        config = model.get_config()

        required_keys = [
            'num_classes', 'width_multiplier', 'dropout_rate',
            'weight_decay', 'kernel_initializer', 'include_top', 'input_shape'
        ]

        for key in required_keys:
            assert key in config, f"Missing '{key}' in get_config()"

    def test_from_config_reconstruction(self, default_config):
        """Test model reconstruction from configuration."""
        model = MobileNetV1(**default_config)
        config = model.get_config()

        reconstructed_model = MobileNetV1.from_config(config)

        assert reconstructed_model.num_classes == model.num_classes
        assert reconstructed_model.width_multiplier == model.width_multiplier
        assert reconstructed_model.dropout_rate == model.dropout_rate
        assert reconstructed_model.include_top == model.include_top

    @pytest.mark.parametrize("variant,expected_multiplier", [
        ("large", 1.0), ("medium", 0.75), ("small", 0.5), ("pico", 0.25)
    ])
    def test_variant_creation(self, variant, expected_multiplier):
        """Test creation of all predefined variants."""
        model = MobileNetV1.from_variant(variant, num_classes=10, input_shape=(32, 32, 3))
        assert model.width_multiplier == expected_multiplier

    def test_convenience_function(self):
        """Test create_mobilenetv1 convenience function."""
        model = create_mobilenetv1(
            variant="medium",
            num_classes=10,
            input_shape=(32, 32, 3)
        )
        assert model.num_classes == 10
        assert model.width_multiplier == 0.75
        assert model.input_shape_config == (32, 32, 3)

    # ============================================================================
    # Forward Pass and Output Shape Tests
    # ============================================================================

    def test_forward_pass_shapes(self, sample_inputs):
        """Test forward pass produces correct output shapes."""
        # With classification head
        model = MobileNetV1.from_variant("large", num_classes=10, input_shape=(32, 32, 3))
        output = model(sample_inputs['cifar'])
        assert not isinstance(output, dict)
        assert output.shape == (4, 10)

        # As a feature extractor. Updated for D-066 (C-11): the global pooling
        # belongs to the head, so include_top=False returns the 4-D feature map,
        # matching V2/V3/V4. This assertion previously pinned the defect.
        model_no_top = MobileNetV1.from_variant("large", include_top=False, input_shape=(32, 32, 3))
        features = model_no_top(sample_inputs['cifar'])
        assert len(features.shape) == 4  # (batch, h, w, channels) feature map

    @pytest.mark.parametrize("input_shape, sample_shape", [
        ((28, 28, 1), (2, 28, 28, 1)),   # Grayscale
        ((64, 64, 3), (3, 64, 64, 3)),   # Custom size
        ((224, 224, 3), (1, 224, 224, 3)) # ImageNet size
    ])
    def test_different_input_sizes(self, input_shape, sample_shape):
        """Test model with different input sizes."""
        model = MobileNetV1.from_variant("small", num_classes=5, input_shape=input_shape)
        sample_input = keras.random.normal(shape=sample_shape)
        output = model(sample_input)
        assert not isinstance(output, dict)
        assert output.shape == (sample_shape[0], 5)

    def test_batch_size_handling(self):
        """Test model handles different batch sizes correctly."""
        model = MobileNetV1.from_variant("pico", num_classes=10, input_shape=(32, 32, 3))
        for batch_size in [1, 4, 16]:
            sample_input = keras.random.normal(shape=(batch_size, 32, 32, 3))
            output = model(sample_input)
            assert not isinstance(output, dict)
            assert output.shape == (batch_size, 10)

    # ============================================================================
    # Training and Gradient Tests
    # ============================================================================

    def test_gradients_flow(self, default_config, sample_inputs):
        """Test gradient computation and backpropagation."""
        model = MobileNetV1(**default_config)
        with tf.GradientTape() as tape:
            output = model(sample_inputs['cifar'], training=True)
            loss = keras.ops.mean(keras.ops.square(output))
        gradients = tape.gradient(loss, model.trainable_variables)
        assert len(gradients) > 0
        # The head builds only what its call() applies (plan-2026-10-14
        # step 1), so every trainable weight is on the gradient path.
        missing = [v.path for v, g in zip(model.trainable_variables, gradients)
                   if g is None]
        assert not missing, f"trainable weights with no gradient: {missing}"
        non_zero_grads = [g for g in gradients if keras.ops.max(keras.ops.abs(g)) > 1e-8]
        assert len(non_zero_grads) > 0, "All gradients are zero"

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, default_config, sample_inputs, training):
        """Test model behavior in different training modes."""
        model = MobileNetV1(**default_config)
        output = model(sample_inputs['cifar'], training=training)
        assert not isinstance(output, dict)
        assert output.shape == (4, default_config['num_classes'])

    def test_model_compilation_and_fit(self, sample_inputs):
        """Test model compiles and can run a training step."""
        model = MobileNetV1.from_variant("pico", num_classes=10, input_shape=(32, 32, 3))
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
        with pytest.raises(ValueError, match="width_multiplier must be positive"):
            MobileNetV1(width_multiplier=0)
        with pytest.raises(ValueError, match="width_multiplier must be positive"):
            MobileNetV1(width_multiplier=-1.0)

    def test_unknown_variant(self):
        """Test error handling for unknown variants."""
        with pytest.raises(ValueError, match="Unknown variant 'invalid_variant'"):
            MobileNetV1.from_variant("invalid_variant")

    def test_invalid_input_shape(self):
        """Test error handling for invalid input shapes."""
        with pytest.raises(ValueError, match="input_shape must be 3D"):
            MobileNetV1(input_shape=(32, 32))

    # ============================================================================
    # Performance and Sanity Tests
    # ============================================================================

    def test_model_parameter_counts(self):
        """Every variant has exactly Keras Applications' MobileNet count.

        The head is the paper's (GAP, dropout, one Dense; D-015) and widths use
        the same ``int(filters * alpha)`` truncation as
        ``keras.applications.MobileNet``, so ``count_params()`` at 224 x 224 x 3
        and 1000 classes equals ``keras.applications.MobileNet(weights=None,
        alpha=a, input_shape=(224, 224, 3)).count_params()`` for every alpha.
        MEASURED 2026-10-14 (plan-2026-10-14T062616-81938e81 step 3), both
        sides on CPU. The two-DenseBlock head of 2026-10-09..14 had been
        pinned with ranges 3.5e6-3.8e6 ... 0.4e6-0.5e6 instead.
        """
        expected = {
            "large": 4_253_864,   # alpha 1.0
            "medium": 2_601_976,  # alpha 0.75
            "small": 1_342_536,   # alpha 0.5
            "pico": 475_544,      # alpha 0.25
        }
        for variant, count in expected.items():
            model = MobileNetV1.from_variant(
                variant,
                num_classes=1000,
                input_shape=(224, 224, 3)
            )
            model(keras.random.normal(shape=(1, 224, 224, 3)))
            assert model.count_params() == count, (
                f"{variant} has {model.count_params():,} parameters, "
                f"Keras Applications MobileNet has {count:,}"
            )

    def test_model_summary_execution(self):
        """Test that model summary executes without errors."""
        model = MobileNetV1.from_variant("pico", num_classes=10, input_shape=(32, 32, 3))
        try:
            model.summary()
        except Exception as e:
            pytest.fail(f"Model summary failed with error: {e}")

    # ============================================================================
    # Integration Tests
    # ============================================================================

    def test_model_evaluation_and_prediction(self):
        """Test model evaluation and prediction methods."""
        model = MobileNetV1.from_variant("pico", num_classes=5, input_shape=(32, 32, 3))
        model.compile(
            optimizer='adam',
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
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
        model = MobileNetV1(num_classes=4, include_top=False, input_shape=(32, 32, 3))
        batch = np.zeros((1, 32, 32, 3), "float32")
        eager = model(batch, training=False)

        reported = model.compute_output_shape((None, 32, 32, 3))
        assert tuple(reported) == (None,) + tuple(eager.shape[1:])

        symbolic = model(keras.Input(shape=(32, 32, 3)))
        assert tuple(symbolic.shape) == (None,) + tuple(eager.shape[1:])


class TestMobileNetV1HouseShape:
    """The constructor, ``from_variant`` and ``build`` contract of the house shape."""

    @pytest.mark.parametrize("variant,count", [
        ("small", 1_342_536),
        ("pico", 475_544),
    ])
    def test_an_explicit_build_creates_every_weight(self, variant, count):
        """``build`` materializes what ``call`` runs.

        The model used to inherit Keras' ``build``, which marked it built with
        0 weights and a warning.
        """
        model = MobileNetV1.from_variant(variant, num_classes=1000)
        model.build((None, 64, 64, 3))
        assert model.count_params() == count

    def test_an_explicit_build_of_a_headless_model_creates_no_head_weight(self):
        model = MobileNetV1.from_variant("pico", include_top=False)
        model.build((None, 64, 64, 3))
        assert model.count_params() > 0
        assert not [w.path for w in model.weights if "classification_head" in w.path]

    def test_an_override_does_not_poison_the_variant_table(self):
        before = copy.deepcopy(MobileNetV1.MODEL_VARIANTS)
        MobileNetV1.from_variant("small", width_multiplier=0.25, dropout_rate=0.3)
        assert MobileNetV1.MODEL_VARIANTS == before

    @pytest.mark.parametrize("variant,given", [("small", 0.25), ("pico", 0.5), ("large", 0.75)])
    def test_a_width_multiplier_replaces_the_variants_own_value(self, variant, given):
        """The README says a passed width overrides the variant's; it used to multiply it."""
        model = MobileNetV1.from_variant(variant, width_multiplier=given, input_shape=(32, 32, 3))
        assert model.width_multiplier == given

    def test_the_variants_own_width_applies_when_none_is_given(self):
        model = MobileNetV1.from_variant("small", input_shape=(32, 32, 3))
        assert model.width_multiplier == 0.5

    def test_an_unknown_variant_lists_the_variants_it_knows(self):
        with pytest.raises(ValueError) as caught:
            MobileNetV1.from_variant("huge")
        assert "Unknown variant 'huge'. Available variants: " in str(caught.value)
        assert "'pico'" in str(caught.value)

    @pytest.mark.parametrize("kwargs,message", [
        ({"width_multiplier": 0.0}, "width_multiplier must be positive, got 0.0"),
        ({"width_multiplier": -1.0}, "width_multiplier must be positive, got -1.0"),
        ({"dropout_rate": 1.0}, "dropout_rate must be in [0, 1), got 1.0"),
        ({"dropout_rate": 2.0}, "dropout_rate must be in [0, 1), got 2.0"),
        ({"dropout_rate": -0.1}, "dropout_rate must be in [0, 1), got -0.1"),
        ({"num_classes": 0}, "num_classes must be positive when include_top=True, got 0"),
    ])
    def test_an_invalid_value_is_refused_naming_the_argument_and_the_value(
            self, kwargs, message):
        with pytest.raises(ValueError) as caught:
            MobileNetV1(input_shape=(32, 32, 3), **kwargs)
        # The argument is a whole word: a sub-layer's `classifier_dropout_rate` is not the model's.
        assert re.search(rf"(?<![A-Za-z0-9_]){re.escape(message)}", str(caught.value)), str(caught.value)

    def test_a_headless_model_needs_no_class_count(self):
        model = MobileNetV1(num_classes=0, include_top=False, input_shape=(32, 32, 3))
        assert model.classification_head is None

    @pytest.mark.parametrize("shape", [(), (32, 32), (32, 32, 3, 1)])
    def test_an_input_shape_that_is_not_3d_is_refused(self, shape):
        with pytest.raises(ValueError) as caught:
            MobileNetV1(input_shape=shape)
        assert f"input_shape must be 3D, got {shape}" in str(caught.value), str(caught.value)

    def test_the_input_shape_defaults_to_the_imagenet_shape(self):
        assert MobileNetV1(input_shape=None).input_shape_config == (224, 224, 3)
        assert MobileNetV1.from_variant("small").input_shape_config == (224, 224, 3)

    def test_forward_takes_its_input_as_inputs(self):
        """The forward method names its input argument ``inputs``, as every classifier does."""
        assert list(inspect.signature(MobileNetV1.call).parameters) == [
            "self", "inputs", "training"]

    def test_the_stem_is_one_named_conv_stem_that_truncates_its_width(self):
        """V1 truncates (``int(32 * 0.6) = 19``); V2 to V4 round to a multiple of 8 (24)."""
        model = MobileNetV1(width_multiplier=0.6, input_shape=(32, 32, 3))
        assert model.stem.name == "stem"
        assert model.stem.conv1.filters == 19
        assert model.stem.act1.max_value is None

    def test_the_blocks_are_named_from_zero_like_every_flat_block_list(self):
        """``block_{i}`` counts from 0, as in MobileNetV3 and CliffordNet; the block's own
        layers take the same index (``conv_dw_{i}``)."""
        model = MobileNetV1(width_multiplier=0.25, input_shape=(32, 32, 3))
        assert [b.name for b in model.blocks] == [f"block_{i}" for i in range(13)]
        assert [b.block_id for b in model.blocks] == list(range(13))
        model.build((None, 32, 32, 3))
        paths = {w.path.split("/", 1)[1] for w in model.weights}
        assert any(p.startswith("block_0/conv_dw_0/") for p in paths), sorted(paths)[:6]


class _NoRoundTrip(keras.initializers.Initializer):
    """A custom initializer that cannot describe itself, so it cannot be saved."""

    def __call__(self, shape, dtype=None, **kwargs):
        return keras.initializers.HeNormal(seed=1)(shape, dtype=dtype)

    def get_config(self):
        raise NotImplementedError("this initializer is not serializable")


class TestMobileNetV1Initializer:
    """The initializer is kept as an instance and cloned per site; its spec is validated once.

    MobileNetV4 stores the spec and hands it to the sites. V1 hands each site a
    clone of the caller's instance instead (decision D-039, anchored in
    ``mobilenet_v1.py``), because a stored spec would move every seeded kernel.
    The spec is still computed in ``__init__``: to refuse a bad initializer at
    construction and to feed ``get_config``.
    """

    def test_an_initializer_that_cannot_serialize_is_refused_at_construction(self):
        with pytest.raises(NotImplementedError, match="not serializable"):
            MobileNetV1(width_multiplier=0.25, input_shape=(32, 32, 3),
                        kernel_initializer=_NoRoundTrip())

    @pytest.mark.parametrize("bad", [3, None, ["he_normal"]])
    def test_a_value_that_is_not_an_initializer_is_refused_naming_its_type(self, bad):
        with pytest.raises(TypeError) as caught:
            MobileNetV1(width_multiplier=0.25, input_shape=(32, 32, 3), kernel_initializer=bad)
        assert type(bad).__name__ in str(caught.value), str(caught.value)

    @pytest.mark.parametrize("given", [
        "he_normal",
        {"class_name": "HeNormal", "config": {"seed": 5}},
        keras.initializers.HeNormal(seed=3),
        keras.initializers.GlorotUniform(),
    ], ids=["name", "dict", "seeded-instance", "seedless-instance"])
    def test_get_config_reports_a_json_safe_spec_that_rebuilds_the_model(self, given):
        model = MobileNetV1(width_multiplier=0.25, input_shape=(32, 32, 3),
                            kernel_initializer=given)
        config = json.loads(json.dumps(model.get_config()))
        spec = config["kernel_initializer"]
        assert isinstance(spec, (str, dict))
        if not isinstance(given, keras.initializers.Initializer):
            assert spec == given
        elif given.seed is not None:
            assert spec == json.loads(json.dumps(keras.initializers.serialize(given)))
        rebuilt = MobileNetV1.from_config(copy.deepcopy(config))
        assert json.loads(json.dumps(rebuilt.get_config()))["kernel_initializer"] == spec

    def test_the_model_keeps_an_instance_and_hands_each_site_a_clone(self):
        """Do not 'align' V1 with V4: a stored spec re-draws every seeded kernel."""
        given = keras.initializers.HeNormal(seed=3)
        model = MobileNetV1(width_multiplier=0.25, input_shape=(32, 32, 3),
                            kernel_initializer=given)
        assert isinstance(model.kernel_initializer, keras.initializers.Initializer)
        assert model.stem.conv1.kernel_initializer is not given
        assert model.blocks[0].depthwise_conv.depthwise_initializer is not given

    def test_a_seeded_instance_draws_what_a_clone_of_it_draws(self):
        """The stem kernel of a seeded model is the draw of a clone of the caller's instance."""
        given = keras.initializers.HeNormal(seed=3)
        model = MobileNetV1(width_multiplier=0.25, input_shape=(32, 32, 3),
                            kernel_initializer=given)
        model.build((None, 32, 32, 3))
        kernel = keras.ops.convert_to_numpy(model.stem.conv1.kernel)
        want = keras.ops.convert_to_numpy(clone_initializer(given)(kernel.shape))
        np.testing.assert_array_equal(kernel, want)


class TestMobileNetV1PretrainedTrunk:
    """A pretrained load restores every layer outside the head.

    Only ``classification_head`` may differ. A trunk of another width used to
    load the layers that happened to fit, report the rest and return normally.
    """

    @staticmethod
    def _saved(tmp_path, variant, **kwargs):
        donor = MobileNetV1.from_variant(
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

    def test_a_checkpoint_of_another_width_is_refused(self, tmp_path):
        _, path = self._saved(tmp_path, "pico")

        with pytest.raises(ValueError) as caught:
            MobileNetV1.from_variant(
                "small", num_classes=10, input_shape=(32, 32, 3), pretrained=path
            )

        assert "width_multiplier: checkpoint 0.25, model 0.5" in str(caught.value), str(caught.value)

        # With the configuration comparison off, the layer rule refuses it by itself.
        receiver = MobileNetV1.from_variant("small", num_classes=10, input_shape=(32, 32, 3))
        with pytest.raises(ValueError) as by_layers:
            load_pretrained_checkpoint(
                receiver, path, (32, 32, 3), ignore_config_keys=tuple(receiver.get_config())
            )
        assert "shape_mismatch" in str(by_layers.value), str(by_layers.value)

    def test_a_refused_load_writes_nothing(self, tmp_path):
        _, path = self._saved(tmp_path, "pico")
        receiver = MobileNetV1.from_variant(
            "small", num_classes=10, input_shape=(32, 32, 3)
        )
        before = [keras.ops.convert_to_numpy(w).copy() for w in receiver.weights]

        with pytest.raises(ValueError):
            receiver.load_pretrained_weights(path)

        for a, w in zip(before, receiver.weights):
            np.testing.assert_array_equal(a, keras.ops.convert_to_numpy(w))

    def test_another_class_count_loads_the_trunk_and_skips_the_head(self, tmp_path):
        donor, path = self._saved(tmp_path, "pico")

        loaded = MobileNetV1.from_variant(
            "pico", num_classes=3, input_shape=(32, 32, 3), pretrained=path
        )

        src, dst = self._by_path(donor), self._by_path(loaded)
        trunk = {p: v for p, v in dst.items() if not p.startswith("classification_head")}
        assert trunk
        for p, value in trunk.items():
            np.testing.assert_array_equal(value, src[p], err_msg=p)
        heads = {p: v for p, v in dst.items() if p.startswith("classification_head")}
        assert heads and any(v.shape != src[p].shape for p, v in heads.items())

    def test_a_headless_model_loads_from_a_headed_checkpoint(self, tmp_path):
        donor, path = self._saved(tmp_path, "pico")

        headless = MobileNetV1.from_variant(
            "pico", include_top=False, input_shape=(32, 32, 3), pretrained=path
        )

        src, dst = self._by_path(donor), self._by_path(headless)
        assert dst
        for p, value in dst.items():
            np.testing.assert_array_equal(value, src[p], err_msg=p)



if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])