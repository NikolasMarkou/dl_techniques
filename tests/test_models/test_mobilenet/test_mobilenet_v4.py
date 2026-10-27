"""
Comprehensive test suite for MobileNetV4 model implementation.

Following the Complete Guide to Modern Keras 3 Custom Layers and Models,
this test suite ensures robust, serializable, and production-ready implementation
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
import json
import pytest
import tempfile
import os
import numpy as np
from typing import Dict, Any

import keras
import tensorflow as tf

from tests.numerics import reassociation_atol

from dl_techniques.models.vision.mobilenet.mobilenet_v4 import MobileNetV4, create_mobilenetv4


class TestMobileNetV4:
    """Comprehensive test suite for MobileNetV4 model."""

    @pytest.fixture
    def default_config(self) -> Dict[str, Any]:
        """Standard configuration for testing."""
        return {
            'num_classes': 10,
            'depths': [1, 1, 2, 2],
            'dims': [16, 24, 32, 64],
            'block_types': ['IB', 'IB', 'ExtraDW', 'IB'],
            'strides': [1, 2, 2, 1],
            'width_multiplier': 1.0,
            'use_attention': False,
            'dropout_rate': 0.1,
            'weight_decay': 1e-5,
            'input_shape': (32, 32, 3),
            'include_top': True
        }

    @pytest.fixture
    def hybrid_config(self) -> Dict[str, Any]:
        """Configuration with attention for hybrid testing."""
        return {
            'num_classes': 100,
            'depths': [1, 1, 2, 2],
            'dims': [16, 24, 32, 64],
            'block_types': ['IB', 'IB', 'ExtraDW', 'IB'],
            'strides': [1, 2, 2, 1],
            'width_multiplier': 1.0,
            'use_attention': True,
            'attention_stages': [2, 3],
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
        """Test width multiplier correctly scales dimensions."""
        base_dims = [16, 24, 32, 64]
        multiplier = 0.75
        num_stages = len(base_dims)

        model = MobileNetV4(
            dims=base_dims,
            depths=[1] * num_stages,
            block_types=['IB'] * num_stages,
            strides=[1, 2, 1, 2],
            width_multiplier=multiplier,
            input_shape=(32, 32, 3)
        )

        expected_dims = [int(dim * multiplier) for dim in base_dims]
        assert model.actual_dims == expected_dims

    def test_model_building_on_call(self, default_config, sample_inputs):
        """Test that model builds correctly on first call."""
        config_no_shape = default_config.copy()
        del config_no_shape['input_shape']
        model = MobileNetV4(**config_no_shape)
        assert not model.built

        # First call should trigger building
        output = model(sample_inputs['cifar'])
        assert model.built
        assert not isinstance(output, dict)
        assert output.shape == (4, default_config['num_classes'])

    # ============================================================================
    # Critical Serialization Tests
    # ============================================================================

    def test_serialization_cycle_conv_only(self, default_config, sample_inputs):
        """CRITICAL TEST: Full serialization cycle for conv-only model."""
        model = MobileNetV4(**default_config)

        # Get prediction from original model
        original_pred = model(sample_inputs['cifar'])

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'mobilenetv4_conv.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_inputs['cifar'])

            # Verify identical predictions (compare logits)
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Conv-only model predictions differ after serialization"
            )

    def test_serialization_cycle_hybrid(self, hybrid_config, sample_inputs):
        """CRITICAL TEST: Full serialization cycle for hybrid model with attention."""
        model = MobileNetV4(**hybrid_config)

        # Get prediction from original model
        original_pred = model(sample_inputs['custom'])

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'mobilenetv4_hybrid.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_inputs['custom'])

            # Verify identical predictions (compare logits)
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Hybrid model predictions differ after serialization"
            )

    def test_serialization_without_top(self, sample_inputs):
        """Test serialization of feature extractor (include_top=False)."""
        model = MobileNetV4(
            depths=[1, 1, 2],
            dims=[16, 24, 32],
            block_types=['IB', 'IB', 'ExtraDW'],
            strides=[1, 2, 2],
            include_top=False,
            input_shape=(32, 32, 3)
        )

        # Get features from original model
        original_features = model(sample_inputs['cifar'])

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'mobilenetv4_features.keras')
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

    def test_config_completeness(self, default_config):
        """Test that get_config contains all initialization parameters."""
        model = MobileNetV4(**default_config)
        config = model.get_config()

        # Check all important config parameters are present
        required_keys = [
            'num_classes', 'depths', 'dims', 'block_types', 'strides',
            'width_multiplier', 'use_attention', 'attention_stages',
            'dropout_rate', 'weight_decay', 'kernel_initializer',
            'include_top', 'input_shape'
        ]

        for key in required_keys:
            assert key in config, f"Missing {key} in get_config()"

    def test_from_config_reconstruction(self, default_config):
        """Test model reconstruction from configuration."""
        model = MobileNetV4(**default_config)
        config = model.get_config()

        # Reconstruct model from config
        reconstructed_model = MobileNetV4.from_config(config)

        # Verify configurations match
        assert reconstructed_model.num_classes == model.num_classes
        assert reconstructed_model.depths == model.depths
        assert reconstructed_model.dims == model.dims
        assert reconstructed_model.block_types == model.block_types
        assert reconstructed_model.width_multiplier == model.width_multiplier

    @pytest.mark.parametrize("variant", [
        "small", "medium", "large", "hybrid_medium", "hybrid_large"
    ])
    def test_variant_creation(self, variant):
        """Test creation of all predefined variants."""
        model = MobileNetV4.from_variant(variant, num_classes=10, input_shape=(32, 32, 3))

        # Check variant-specific properties
        variant_config = MobileNetV4.MODEL_VARIANTS[variant]
        assert model.depths == variant_config['depths']
        assert model.dims == variant_config['dims']
        assert model.block_types == variant_config['block_types']
        assert model.use_attention == variant_config['use_attention']

    def test_convenience_function(self):
        """Test create_mobilenetv4 convenience function."""
        model = create_mobilenetv4(
            variant="small",
            num_classes=10,
            input_shape=(32, 32, 3),
            width_multiplier=0.75
        )

        assert model.num_classes == 10
        assert model.width_multiplier == 0.75
        assert model.input_shape_config == (32, 32, 3)

    # ============================================================================
    # block_type -> structure Tests
    # ============================================================================

    # The stage width is the stem output, so the block's INPUT channel count is
    # `_STRUCT_DIM` and its expanded count is `_STRUCT_DIM * 4` (the layer's
    # default expansion_factor). Both numbers are needed to tell a start DW from
    # a middle one, and they must not coincide -- hence a non-unit expansion.
    _STRUCT_DIM = 16
    _STRUCT_EXPANSION = 4

    def _single_stage_block(self, block_type: str):
        """Build a one-stage, one-block MobileNetV4 and return its UIB block."""
        model = MobileNetV4(
            num_classes=4,
            depths=[1],
            dims=[self._STRUCT_DIM],
            block_types=[block_type],
            strides=[1],
            use_attention=False,
            input_shape=(32, 32, 3),
        )
        model(keras.random.normal(shape=(2, 32, 32, 3)))
        return model, model.stages[0][0]

    @pytest.mark.parametrize("block_type,expected_dw_count", [
        ("FFN", 0),
        ("IB", 1),
        ("ConvNext", 1),
        ("ExtraDW", 2),
    ])
    def test_block_types_build_structurally_different_blocks(
        self, block_type, expected_dw_count
    ):
        """The four block types must not build the same block.

        Before `block_type` was wired into `_build_stage` it was an inert label:
        all four of these built the layer's default IB, so this parametrization
        was 0/1/1/2 against a measured 1/1/1/1.

        A depthwise COUNT alone cannot carry the assertion, because `IB` and
        `ConvNext` both own exactly one -- and a parameter count cannot either,
        since two depthwise convs of the same kernel size differ only by the
        channel count. What separates them is WHERE that one depthwise sits,
        which is visible in the channel dimension of its built kernel: the start
        DW is built on the block input, a middle DW on the expanded tensor.
        """
        expanded = self._STRUCT_DIM * self._STRUCT_EXPANSION
        _, block = self._single_stage_block(block_type)

        dws = [dw for dw in (block.start_dw, block.dw1, block.dw2) if dw is not None]
        assert len(dws) == expected_dw_count, (
            f"block_type={block_type!r} built {len(dws)} depthwise convs, "
            f"expected {expected_dw_count}"
        )

        if block_type == "ConvNext":
            assert block.start_dw is not None and block.dw1 is None
            assert block.start_dw.kernel.shape[2] == self._STRUCT_DIM, (
                "ConvNext's depthwise must be PRE-expansion (it sees the block's "
                "input channels); a post-expansion one would see "
                f"{expanded}"
            )
        elif block_type == "IB":
            assert block.dw1 is not None and block.start_dw is None
            assert block.dw1.kernel.shape[2] == expanded, (
                "IB's depthwise must be POST-expansion (it sees the expanded "
                f"channels); a pre-expansion one would see {self._STRUCT_DIM}"
            )
        elif block_type == "ExtraDW":
            assert block.start_dw.kernel.shape[2] == self._STRUCT_DIM
            assert block.dw1.kernel.shape[2] == expanded

    @pytest.mark.parametrize("block_type", ["IB", "ConvNext", "ExtraDW", "FFN"])
    def test_block_types_forward_pass_and_round_trip(self, block_type):
        """Every block type must forward-pass and survive a config round trip.

        `ConvNext` and `FFN` appear in no shipped `MODEL_VARIANTS` row, so
        nothing else in this suite ever builds them; before this test they were
        reachable only through the validator that accepts their names.
        """
        model, _ = self._single_stage_block(block_type)
        x = keras.random.normal(shape=(2, 32, 32, 3))

        out = model(x)
        assert not isinstance(out, dict)
        assert out.shape == (2, 4)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))

        rebuilt = MobileNetV4.from_config(model.get_config())
        assert rebuilt.block_types == [block_type]
        rebuilt_out = rebuilt(x)
        assert not isinstance(rebuilt_out, dict)
        assert rebuilt_out.shape == (2, 4)

    # ============================================================================
    # Forward Pass and Output Shape Tests
    # ============================================================================

    def test_forward_pass_shapes(self, sample_inputs):
        """Test forward pass produces correct output shapes."""
        # Conv-only model
        model = MobileNetV4.from_variant("small", num_classes=10, input_shape=(32, 32, 3))
        output = model(sample_inputs['cifar'])
        assert not isinstance(output, dict)
        assert output.shape == (4, 10)

        # Feature extractor
        model = MobileNetV4.from_variant("small", include_top=False, input_shape=(32, 32, 3))
        features = model(sample_inputs['cifar'])
        # Output should be spatial features from last stage
        assert len(features.shape) == 4  # (batch, height, width, channels)

    def test_different_input_sizes(self):
        """Test model with different input sizes."""
        test_configs = [
            ((28, 28, 1), (2, 28, 28, 1)),  # MNIST-like
            ((64, 64, 3), (3, 64, 64, 3)),  # Medium resolution
            ((224, 224, 3), (1, 224, 224, 3)),  # ImageNet-like
        ]

        for input_shape, sample_shape in test_configs:
            model = MobileNetV4.from_variant("small", num_classes=5, input_shape=input_shape)
            sample_input = keras.random.normal(shape=sample_shape)

            output = model(sample_input)
            assert not isinstance(output, dict)
            assert output.shape == (sample_shape[0], 5)

    def test_batch_size_handling(self):
        """Test model handles different batch sizes correctly."""
        model = MobileNetV4.from_variant("small", num_classes=10, input_shape=(32, 32, 3))

        batch_sizes = [1, 4, 8, 16]
        for batch_size in batch_sizes:
            sample_input = keras.random.normal(shape=(batch_size, 32, 32, 3))
            output = model(sample_input)
            assert not isinstance(output, dict)
            assert output.shape == (batch_size, 10)

    # ============================================================================
    # Training and Gradient Tests
    # ============================================================================

    def test_gradients_flow(self, default_config, sample_inputs):
        """Test gradient computation and backpropagation."""
        model = MobileNetV4(**default_config)

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
    def test_training_modes(self, default_config, sample_inputs, training):
        """Test behavior in different training modes."""
        model = MobileNetV4(**default_config)

        output = model(sample_inputs['cifar'], training=training)
        assert not isinstance(output, dict)
        assert output.shape == (4, default_config['num_classes'])

        # Test with attention model too
        hybrid_config = default_config.copy()
        hybrid_config.update({
            "use_attention": True,
            "attention_stages": [2, 3]
        })
        hybrid_model = MobileNetV4(**hybrid_config)
        output = hybrid_model(sample_inputs['cifar'], training=training)
        assert not isinstance(output, dict)
        assert output.shape == (4, default_config['num_classes'])

    def test_model_compilation_and_fit(self, sample_inputs):
        """Test model compiles and can run a training step."""
        model = MobileNetV4.from_variant("small", num_classes=10, input_shape=(32, 32, 3))

        # Compile model
        model.compile(
            optimizer='adam',
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        # Create dummy training data
        x_train = sample_inputs['cifar']
        y_train = np.random.randint(0, 10, size=(4,))

        # Test one training step
        history = model.fit(x_train, y_train, epochs=1, verbose=0)

        # Check training executed successfully
        assert 'loss' in history.history
        assert len(history.history['loss']) == 1

    # ============================================================================
    # Edge Cases and Error Handling
    # ============================================================================

    def test_invalid_configurations(self):
        """Test that invalid configurations raise appropriate errors."""

        # Mismatched configuration lengths
        with pytest.raises(ValueError, match="All stage configurations must have same length"):
            MobileNetV4(
                depths=[1, 2, 3],
                dims=[16, 24],  # Different length
                block_types=['IB', 'IB', 'ExtraDW'],
                strides=[1, 2, 2],
                input_shape=(32, 32, 3)
            )

        # Invalid block type
        with pytest.raises(ValueError, match=r"block_types\[1\] must be one of"):
            MobileNetV4(
                depths=[1, 2],
                dims=[16, 24],
                block_types=['IB', 'InvalidBlock'],
                strides=[1, 2],
                input_shape=(32, 32, 3)
            )

        # Invalid attention stage index
        with pytest.raises(ValueError, match="attention_stages entries must be in"):
            MobileNetV4(
                depths=[1, 2],
                dims=[16, 24],
                block_types=['IB', 'IB'],
                strides=[1, 2],
                use_attention=True,
                attention_stages=[5],  # Out of range
                input_shape=(32, 32, 3)
            )

    def test_unknown_variant(self):
        """Test error handling for unknown variants."""
        with pytest.raises(ValueError, match="Unknown variant"):
            MobileNetV4.from_variant("nonexistent_variant")

    def test_invalid_input_shape(self):
        """Test error handling for invalid input shapes."""
        with pytest.raises(ValueError, match="input_shape must be 3D"):
            MobileNetV4(input_shape=(32, 32))  # Missing channel dimension

    def test_edge_case_dimensions(self):
        """Test edge cases with small dimensions and configurations."""
        # Very small model
        model = MobileNetV4(
            depths=[1],
            dims=[8],
            block_types=['IB'],
            strides=[1],
            num_classes=2,
            input_shape=(16, 16, 1)
        )

        sample_input = keras.random.normal(shape=(2, 16, 16, 1))
        output = model(sample_input)
        assert not isinstance(output, dict)
        assert output.shape == (2, 2)

    # ============================================================================
    # Performance and Memory Tests
    # ============================================================================

    def test_model_parameter_counts(self):
        """Exact counts at 224 x 224 x 3, 1000 classes, re-derived by command.

        These ladders are not the paper's tables, so there is no external
        total to match; the head is timm's MobileNetV4 head (GAP, bias-free
        Dense 1280, BatchNorm, ReLU, dropout, Dense; D-015), ``C * 1280 +
        5,120 + 1,281,000`` values at 1000 classes for a last stage of width
        ``C`` (1,490,920 for ``small``, C = 160). MEASURED 2026-10-14
        (plan-2026-10-14T062616-81938e81 step 3) on CPU with
        ``MobileNetV4.from_variant(v, num_classes=1000)`` and one forward pass.
        """
        expected = {
            'small': 2_124_024,
            'medium': 3_407_112,
            'large': 4_849_736,
            'hybrid_medium': 3_722_250,
            'hybrid_large': 5_296_538,
        }
        for variant, count in expected.items():
            model = MobileNetV4.from_variant(variant, num_classes=1000)
            model(keras.random.normal(shape=(1, 224, 224, 3)))
            assert model.count_params() == count, (
                f"{variant}: {model.count_params():,} != {count:,}")

    def test_model_summary_execution(self):
        """Test that model summary executes without errors."""
        model = MobileNetV4.from_variant("small", num_classes=10, input_shape=(32, 32, 3))

        # Build model first
        model(keras.random.normal(shape=(1, 32, 32, 3)))

        # Test summary doesn't crash
        try:
            model.summary()
        except Exception as e:
            pytest.fail(f"Model summary failed with error: {e}")

    # ============================================================================
    # Integration Tests
    # ============================================================================

    def test_model_in_training_loop(self):
        """Test model in a realistic training scenario."""
        model = MobileNetV4.from_variant("small", num_classes=2, input_shape=(32, 32, 3))
        model.compile(
            optimizer='adam',
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        # Generate synthetic data
        x_train = keras.random.normal(shape=(20, 32, 32, 3))
        y_train = np.random.randint(0, 2, size=(20,))
        x_val = keras.random.normal(shape=(10, 32, 32, 3))
        y_val = np.random.randint(0, 2, size=(10,))

        # Train for a few epochs
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=2,
            batch_size=4,
            verbose=0
        )

        # Check training completed successfully
        assert len(history.history['loss']) == 2
        assert 'val_loss' in history.history

    def test_model_evaluation_and_prediction(self):
        """Test model evaluation and prediction methods."""
        model = MobileNetV4.from_variant("small", num_classes=5, input_shape=(32, 32, 3))
        model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

        # Test data
        x_test = keras.random.normal(shape=(10, 32, 32, 3))
        y_test = np.random.randint(0, 5, size=(10,))

        # Test evaluation
        results = model.evaluate(x_test, y_test, verbose=0)
        assert len(results) == 2  # loss and accuracy

        # Test prediction (returns logits tensor)
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

        The stem is ``self.stem`` (a ``ConvStem``); both ``compute_output_shape``
        and ``call`` must read it and not the sub-layers it used to alias.
        Keras uses a model's ``compute_output_shape`` for a symbolic call, so a
        stale read raises ``AttributeError`` on both paths while every test that
        only runs the eager forward stays green.
        """
        model = MobileNetV4(
            num_classes=4,
            depths=[1, 1],
            dims=[24, 32],
            block_types=["IB", "IB"],
            strides=[1, 2],
            include_top=False,
            input_shape=(32, 32, 3),
        )
        batch = np.zeros((1, 32, 32, 3), "float32")
        eager = model(batch, training=False)

        reported = model.compute_output_shape((None, 32, 32, 3))
        assert tuple(reported) == (None,) + tuple(eager.shape[1:])

        symbolic = model(keras.Input(shape=(32, 32, 3)))
        assert tuple(symbolic.shape) == (None,) + tuple(eager.shape[1:])


class TestMobileNetV4HouseShape:
    """The constructor, ``from_variant`` and ``build`` contract of the house shape."""

    @pytest.mark.parametrize("override", [
        {"depths": [1, 1, 1, 1, 1, 1, 1]},
        {"dims": [8, 16, 24, 32, 40, 48, 56]},
        {"block_types": ["IB", "IB", "IB", "IB", "IB", "IB", "FFN"]},
        {"strides": [1, 1, 2, 2, 1, 2, 1]},
        {"width_multiplier": 0.5},
        {"use_attention": True},
        {"use_attention": True, "attention_stages": [4, 5]},
        {"dropout_rate": 0.3, "include_top": False},
    ])
    def test_from_variant_replaces_a_field_of_the_variant(self, override):
        """A keyword given to ``from_variant`` replaces the preset's own value.

        The preset fields used to be splatted beside ``**kwargs``, so
        naming one of them raised ``TypeError: got multiple values``.
        """
        model = MobileNetV4.from_variant(
            "small", num_classes=3, input_shape=(32, 32, 3), **override)
        for name, value in override.items():
            stored = getattr(model, name)
            assert stored == (list(value) if isinstance(value, tuple) else value), name

    def test_an_override_does_not_poison_the_variant_table(self):
        before = copy.deepcopy(MobileNetV4.MODEL_VARIANTS)
        MobileNetV4.from_variant("hybrid_medium", depths=[2] * 7, attention_stages=[1])
        assert MobileNetV4.MODEL_VARIANTS == before

    def test_every_variant_row_names_its_strides(self):
        for name, row in MobileNetV4.MODEL_VARIANTS.items():
            assert len(row["strides"]) == len(row["depths"]), name

    @pytest.mark.parametrize("variant,side,count", [
        ("small", 64, 2_124_024),
        ("hybrid_medium", 128, 3_722_250),
    ])
    def test_an_explicit_build_creates_every_weight(self, variant, side, count):
        """``build`` materializes what ``call`` runs.

        The model used to inherit Keras' ``build``, which marked it built with
        0 weights and a warning.
        """
        model = MobileNetV4.from_variant(variant, num_classes=1000)
        model.build((None, side, side, 3))
        assert model.count_params() == count

    def test_the_input_shape_defaults_to_the_imagenet_shape(self):
        assert MobileNetV4(input_shape=None).input_shape_config == (224, 224, 3)
        assert MobileNetV4.from_variant("small").input_shape_config == (224, 224, 3)

    def test_an_initializer_instance_is_stored_as_its_serialized_spec(self):
        model = MobileNetV4.from_variant(
            "small", input_shape=(32, 32, 3),
            kernel_initializer=keras.initializers.HeNormal())
        assert isinstance(model.kernel_initializer, dict)
        config = json.loads(json.dumps(model.get_config()))
        assert config["kernel_initializer"] == model.kernel_initializer
        rebuilt = MobileNetV4.from_config(config)
        assert rebuilt.kernel_initializer == model.kernel_initializer

    def test_layers_are_named_by_stage_and_block_from_one(self):
        model = MobileNetV4.from_variant("hybrid_medium", input_shape=(128, 128, 3))
        names = [layer.name for stage in model.stages for layer in stage]
        expected = []
        for stage_idx, depth in enumerate(model.depths):
            expected += [f"stage{stage_idx + 1}_block{b + 1}" for b in range(depth)]
            if stage_idx in model.attention_stages:
                expected.append(f"stage{stage_idx + 1}_mqa")
        assert names == expected
        assert names[0] == "stage1_block1"
        assert "stage6_mqa" in names and "stage7_mqa" in names

    @pytest.mark.parametrize("kwargs,message", [
        ({"width_multiplier": 0.0}, "width_multiplier must be positive, got 0.0"),
        ({"width_multiplier": -1.0}, "width_multiplier must be positive, got -1.0"),
        ({"dropout_rate": 1.0}, "dropout_rate must be in [0, 1), got 1.0"),
        ({"dropout_rate": -0.1}, "dropout_rate must be in [0, 1), got -0.1"),
        ({"num_classes": 0}, "num_classes must be positive when include_top=True, got 0"),
        ({"block_types": ["IB", "Bogus"], "depths": [1, 1], "dims": [16, 24],
          "strides": [1, 2]},
         "block_types[1] must be one of ['ConvNext', 'ExtraDW', 'FFN', 'IB'], got 'Bogus'"),
        ({"use_attention": True, "attention_stages": [9]},
         "attention_stages entries must be in [0, 6], got 9"),
        ({"width_multiplier": 0.01},
         "width_multiplier=0.01 truncates a width of dims=[16, 24, 40, 80, 112, 192, 320] to 0"),
    ])
    def test_an_invalid_value_is_refused_naming_the_argument_and_the_value(
            self, kwargs, message):
        with pytest.raises(ValueError) as caught:
            MobileNetV4(input_shape=(32, 32, 3), **kwargs)
        assert message in str(caught.value), str(caught.value)

    def test_a_headless_model_needs_no_class_count(self):
        model = MobileNetV4(num_classes=0, include_top=False, input_shape=(32, 32, 3))
        assert model.classification_head is None


class TestMobileNetV4PretrainedTrunk:
    """D-040: a pretrained load restores every layer outside the head.

    Only ``classification_head`` may differ. A trunk of another variant used to
    load the layers that happened to fit, report the rest and return normally.
    """

    @staticmethod
    def _saved(tmp_path, variant, **kwargs):
        donor = MobileNetV4.from_variant(
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
        _, path = self._saved(tmp_path, "hybrid_medium")

        with pytest.raises(ValueError) as caught:
            MobileNetV4.from_variant(
                "small", num_classes=10, input_shape=(32, 32, 3), pretrained=path
            )

        message = str(caught.value)
        assert "shape_mismatch" in message and "missing_in_source" in message, message

    def test_a_refused_load_writes_nothing(self, tmp_path):
        _, path = self._saved(tmp_path, "hybrid_medium")
        receiver = MobileNetV4.from_variant(
            "small", num_classes=10, input_shape=(32, 32, 3)
        )
        before = [keras.ops.convert_to_numpy(w).copy() for w in receiver.weights]

        with pytest.raises(ValueError):
            receiver.load_pretrained_weights(path)

        for a, w in zip(before, receiver.weights):
            np.testing.assert_array_equal(a, keras.ops.convert_to_numpy(w))

    def test_another_class_count_loads_the_trunk_and_skips_the_head(self, tmp_path):
        donor, path = self._saved(tmp_path, "small")

        loaded = MobileNetV4.from_variant(
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

        headless = MobileNetV4.from_variant(
            "small", include_top=False, input_shape=(32, 32, 3), pretrained=path
        )

        src, dst = self._by_path(donor), self._by_path(headless)
        assert dst
        for p, value in dst.items():
            np.testing.assert_array_equal(value, src[p], err_msg=p)


# ============================================================================
# Additional Test Utilities and Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def test_models():
    """Create test models for reuse across tests."""
    models = {}

    # Small conv model for quick tests
    models['small_conv'] = MobileNetV4.from_variant(
        "small",
        num_classes=10,
        input_shape=(32, 32, 3)
    )

    # Medium hybrid model
    models['medium_hybrid'] = MobileNetV4.from_variant(
        "hybrid_medium",
        num_classes=100,
        input_shape=(64, 64, 3)
    )

    return models


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])