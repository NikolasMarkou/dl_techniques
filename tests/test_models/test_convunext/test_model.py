"""
Tests for ConvUNextModel and ConvUNextStem.

Comprehensive test suite covering:
- Instantiation and configuration
- Forward pass correctness
- Serialization/deserialization
- Deep supervision
- include_top behavior
- Weight compatibility
- Variant factory methods
"""

import os
import tempfile
from typing import Dict, Any

import numpy as np
import pytest
import keras

from dl_techniques.layers.stochastic_depth import StochasticDepth
from dl_techniques.layers.norms.global_response_norm import (
    GlobalResponseNormalization,
)
from dl_techniques.models.convunext.model import (
    ConvUNextStem,
    ConvUNextModel,
    create_convunext_variant,
    create_inference_model_from_training_model,
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def sample_input_small() -> np.ndarray:
    """Small sample input for quick tests."""
    return np.random.randn(2, 64, 64, 3).astype(np.float32)


@pytest.fixture
def sample_input_medium() -> np.ndarray:
    """Medium sample input for standard tests."""
    return np.random.randn(2, 128, 128, 3).astype(np.float32)


@pytest.fixture
def stem_config() -> Dict[str, Any]:
    """Default configuration for ConvUNextStem."""
    return {
        'filters': 32,
        'kernel_size': 7,
        'use_bias': True,
        'kernel_initializer': 'he_normal',
        'kernel_regularizer': None,
    }


@pytest.fixture
def minimal_model_config() -> Dict[str, Any]:
    """Minimal configuration for fast model tests."""
    return {
        'input_shape': (64, 64, 3),
        'depth': 2,
        'initial_filters': 16,
        'filter_multiplier': 2,
        'blocks_per_level': 1,
        'convnext_version': 'v2',
        'drop_path_rate': 0.0,
        'output_channels': 1,
    }


# ---------------------------------------------------------------------
# ConvUNextStem Tests
# ---------------------------------------------------------------------

class TestConvUNextStem:
    """Test suite for ConvUNextStem layer."""

    def test_instantiation(self, stem_config: Dict[str, Any]) -> None:
        """Test stem can be instantiated with valid config."""
        stem = ConvUNextStem(**stem_config)
        assert stem.filters == stem_config['filters']
        assert stem.kernel_size == stem_config['kernel_size']
        assert stem.use_bias == stem_config['use_bias']

    def test_forward_pass(
        self,
        stem_config: Dict[str, Any],
        sample_input_small: np.ndarray
    ) -> None:
        """Test forward pass produces correct output shape."""
        stem = ConvUNextStem(**stem_config)
        output = stem(sample_input_small)

        expected_shape = (
            sample_input_small.shape[0],
            sample_input_small.shape[1],
            sample_input_small.shape[2],
            stem_config['filters'],
        )
        assert output.shape == expected_shape

    def test_build_creates_weights(
        self,
        stem_config: Dict[str, Any],
        sample_input_small: np.ndarray
    ) -> None:
        """Test that build() creates expected weights."""
        stem = ConvUNextStem(**stem_config)
        stem(sample_input_small)

        assert stem.built
        assert len(stem.trainable_weights) > 0

    def test_compute_output_shape(
        self,
        stem_config: Dict[str, Any],
        sample_input_small: np.ndarray
    ) -> None:
        """Test compute_output_shape matches actual output."""
        stem = ConvUNextStem(**stem_config)

        computed_shape = stem.compute_output_shape(sample_input_small.shape)
        actual_output = stem(sample_input_small)

        assert computed_shape == actual_output.shape

    def test_compute_output_shape_before_build(
        self,
        stem_config: Dict[str, Any]
    ) -> None:
        """Test compute_output_shape works before layer is built."""
        stem = ConvUNextStem(**stem_config)

        input_shape = (None, 64, 64, 3)
        computed_shape = stem.compute_output_shape(input_shape)

        assert computed_shape == (None, 64, 64, stem_config['filters'])

    def test_training_vs_inference(
        self,
        stem_config: Dict[str, Any],
        sample_input_small: np.ndarray
    ) -> None:
        """Test layer behaves correctly in training vs inference mode."""
        stem = ConvUNextStem(**stem_config)

        train_output = stem(sample_input_small, training=True)
        infer_output = stem(sample_input_small, training=False)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(train_output),
            keras.ops.convert_to_numpy(infer_output),
            rtol=1e-5, atol=1e-5,
            err_msg="Stem outputs should match in train vs inference"
        )

    def test_get_config_complete(self, stem_config: Dict[str, Any]) -> None:
        """Test get_config returns all constructor arguments."""
        stem = ConvUNextStem(**stem_config)
        config = stem.get_config()

        assert 'filters' in config
        assert 'kernel_size' in config
        assert 'use_bias' in config
        assert 'kernel_initializer' in config
        assert 'kernel_regularizer' in config

    def test_from_config_reconstruction(
        self,
        stem_config: Dict[str, Any],
        sample_input_small: np.ndarray
    ) -> None:
        """Test layer can be reconstructed from config."""
        original = ConvUNextStem(**stem_config)
        original(sample_input_small)

        config = original.get_config()
        reconstructed = ConvUNextStem.from_config(config)

        assert reconstructed.filters == original.filters
        assert reconstructed.use_bias == original.use_bias

    def test_serialization_cycle(
        self,
        stem_config: Dict[str, Any],
        sample_input_small: np.ndarray
    ) -> None:
        """Test full save/load cycle preserves functionality."""
        inputs = keras.Input(shape=sample_input_small.shape[1:])
        outputs = ConvUNextStem(**stem_config)(inputs)
        model = keras.Model(inputs, outputs)

        original_output = model(sample_input_small)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_stem.keras')
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)

        loaded_output = loaded_model(sample_input_small)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(original_output),
            keras.ops.convert_to_numpy(loaded_output),
            rtol=1e-6, atol=1e-6,
            err_msg="Stem outputs should match after serialization"
        )

    @pytest.mark.parametrize("use_bias", [True, False])
    def test_bias_configuration(
        self,
        use_bias: bool,
        sample_input_small: np.ndarray
    ) -> None:
        """Test stem works with and without bias."""
        stem = ConvUNextStem(filters=32, use_bias=use_bias)
        output = stem(sample_input_small)

        assert output.shape[-1] == 32

    @pytest.mark.parametrize("kernel_size", [3, 5, 7])
    def test_different_kernel_sizes(
        self,
        kernel_size: int,
        sample_input_small: np.ndarray
    ) -> None:
        """Test stem works with various kernel sizes."""
        stem = ConvUNextStem(filters=32, kernel_size=kernel_size)
        output = stem(sample_input_small)

        assert output.shape == (2, 64, 64, 32)


class TestMergedConvUNextStemKnobs:
    """The two knobs the merge added: `use_bias` and `stem_normalization`.

    This class collapses what used to be two same-named classes (the bias-free
    GRN stem in `models/bias_free_denoisers/bfconvunext.py` and the LayerNorm
    stem here), so each knob needs a guard that is proven to fail when the knob
    is ignored.
    """

    def test_stem_use_bias_true_creates_a_bias_vector(
        self,
        sample_input_small: np.ndarray
    ) -> None:
        """`use_bias=True` must reach the conv AND allocate a bias WEIGHT.

        Asserting the flag alone is not enough: a layer can report
        `use_bias is True` and still never have allocated the vector.
        """
        stem = ConvUNextStem(filters=8, use_bias=True)
        stem(sample_input_small)

        assert stem.conv.use_bias is True
        assert stem.conv.bias is not None
        n_bias_weights = sum(
            1 for w in stem.conv.weights if w.path.endswith('bias')
        )
        assert n_bias_weights == 1, (
            f"use_bias=True stem allocated {n_bias_weights} bias weights"
        )
        assert int(np.prod(stem.conv.bias.shape)) == 8

        # Control: the bias-free arm allocates none.
        stem_bf = ConvUNextStem(filters=8, use_bias=False)
        stem_bf(sample_input_small)
        assert stem_bf.conv.use_bias is False
        assert stem_bf.conv.bias is None
        assert not any(w.path.endswith('bias') for w in stem_bf.conv.weights)

    def test_stem_normalization_layernorm_selects_layer_norm(
        self,
        sample_input_small: np.ndarray
    ) -> None:
        """`stem_normalization` must select the normalization CLASS actually built."""
        stem_ln = ConvUNextStem(filters=8, stem_normalization='layer_norm')
        stem_ln(sample_input_small)
        assert type(stem_ln.norm) is keras.layers.LayerNormalization

        # The default is the bias-free / ConvNeXt-V2 choice, a different class.
        stem_default = ConvUNextStem(filters=8)
        stem_default(sample_input_small)
        assert type(stem_default.norm) is GlobalResponseNormalization
        assert type(stem_default.norm) is not type(stem_ln.norm)

    def test_stem_config_round_trip_at_non_default_knob_values(
        self,
        sample_input_small: np.ndarray
    ) -> None:
        """Full `.keras` round trip built at NON-DEFAULT values for both knobs.

        Built at the defaults, this test would be vacuous: dropping a key from
        `get_config` is silently repaired by the constructor default and the
        round trip still passes (measured in step 2 of this plan). So the stem
        here is `use_bias=False` (default True) and
        `stem_normalization='layer_norm'` (default 'global_response_norm').
        """
        inputs = keras.Input(shape=sample_input_small.shape[1:])
        outputs = ConvUNextStem(
            filters=8,
            use_bias=False,
            stem_normalization='layer_norm',
            name='stem_under_test',
        )(inputs)
        model = keras.Model(inputs, outputs)

        original_output = model(sample_input_small, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_merged_stem.keras')
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)

        loaded_stem = loaded_model.get_layer('stem_under_test')
        assert loaded_stem.use_bias is False
        assert loaded_stem.conv.use_bias is False
        assert loaded_stem.conv.bias is None
        assert loaded_stem.stem_normalization == 'layer_norm'
        assert type(loaded_stem.norm) is keras.layers.LayerNormalization

        loaded_output = loaded_model(sample_input_small, training=False)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(original_output),
            keras.ops.convert_to_numpy(loaded_output),
            rtol=1e-6, atol=1e-6,
            err_msg="Merged stem outputs differ after .keras round-trip",
        )


# ---------------------------------------------------------------------
# ConvUNextModel Tests
# ---------------------------------------------------------------------

class TestConvUNextModelInstantiation:
    """Test suite for ConvUNextModel instantiation."""

    def test_default_instantiation(self) -> None:
        """Test model can be instantiated with defaults."""
        model = ConvUNextModel()
        assert model.depth == 4
        assert model.initial_filters == 64

    def test_custom_instantiation(
        self,
        minimal_model_config: Dict[str, Any]
    ) -> None:
        """Test model can be instantiated with custom config."""
        model = ConvUNextModel(**minimal_model_config)
        assert model.depth == minimal_model_config['depth']
        assert model.initial_filters == minimal_model_config['initial_filters']

    def test_invalid_depth_raises_error(self) -> None:
        """Test model rejects invalid depth."""
        with pytest.raises(ValueError, match="Depth must be >= 2"):
            ConvUNextModel(depth=1)

    @pytest.mark.parametrize("version", ['v1', 'v2'])
    def test_convnext_versions(
        self,
        version: str,
        minimal_model_config: Dict[str, Any]
    ) -> None:
        """Test model works with both ConvNeXt versions."""
        config = minimal_model_config.copy()
        config['convnext_version'] = version
        model = ConvUNextModel(**config)
        assert model.convnext_version == version


class TestConvUNextModelForwardPass:
    """Test suite for ConvUNextModel forward pass."""

    def test_basic_forward_pass(
        self,
        minimal_model_config: Dict[str, Any],
        sample_input_small: np.ndarray
    ) -> None:
        """Test basic forward pass."""
        model = ConvUNextModel(**minimal_model_config)
        output = model(sample_input_small)

        expected_shape = (
            sample_input_small.shape[0],
            sample_input_small.shape[1],
            sample_input_small.shape[2],
            minimal_model_config['output_channels'],
        )
        assert output.shape == expected_shape

    def test_forward_pass_training_mode(
        self,
        minimal_model_config: Dict[str, Any],
        sample_input_small: np.ndarray
    ) -> None:
        """Test forward pass in training mode."""
        model = ConvUNextModel(**minimal_model_config)
        output = model(sample_input_small, training=True)

        assert output.shape[0] == sample_input_small.shape[0]

    def test_forward_pass_inference_mode(
        self,
        minimal_model_config: Dict[str, Any],
        sample_input_small: np.ndarray
    ) -> None:
        """Test forward pass in inference mode."""
        model = ConvUNextModel(**minimal_model_config)
        output = model(sample_input_small, training=False)

        assert output.shape[0] == sample_input_small.shape[0]

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_variable_batch_sizes(
        self,
        batch_size: int,
        minimal_model_config: Dict[str, Any]
    ) -> None:
        """Test model handles various batch sizes."""
        model = ConvUNextModel(**minimal_model_config)
        inputs = np.random.randn(batch_size, 64, 64, 3).astype(np.float32)
        output = model(inputs)

        assert output.shape[0] == batch_size


class TestConvUNextModelIncludeTop:
    """Test suite for include_top functionality."""

    def test_include_top_true_output_shape(
        self,
        minimal_model_config: Dict[str, Any],
        sample_input_small: np.ndarray
    ) -> None:
        """Test include_top=True produces correct output channels."""
        config = minimal_model_config.copy()
        config['include_top'] = True
        config['output_channels'] = 5

        model = ConvUNextModel(**config)
        output = model(sample_input_small)

        assert output.shape[-1] == 5

    def test_include_top_false_output_shape(
        self,
        minimal_model_config: Dict[str, Any],
        sample_input_small: np.ndarray
    ) -> None:
        """Test include_top=False returns decoder features."""
        config = minimal_model_config.copy()
        config['include_top'] = False
        config['output_channels'] = 5

        model = ConvUNextModel(**config)
        output = model(sample_input_small)

        # Output should be initial_filters (highest resolution decoder features)
        assert output.shape[-1] == config['initial_filters']

    def test_include_top_weight_compatibility(
        self,
        minimal_model_config: Dict[str, Any],
        sample_input_small: np.ndarray
    ) -> None:
        """Test weights are compatible between include_top=True and False."""
        config = minimal_model_config.copy()
        config['output_channels'] = 5

        # Create model with include_top=True
        model_with_top = ConvUNextModel(**{**config, 'include_top': True})
        model_with_top(sample_input_small)

        # Create model with include_top=False
        model_without_top = ConvUNextModel(**{**config, 'include_top': False})
        model_without_top(sample_input_small)

        # Both should have final_output_layer created
        assert hasattr(model_with_top, 'final_output_layer')
        assert hasattr(model_without_top, 'final_output_layer')


class TestConvUNextModelDeepSupervision:
    """Test suite for deep supervision functionality."""

    def test_deep_supervision_output_count(
        self,
        minimal_model_config: Dict[str, Any],
        sample_input_small: np.ndarray
    ) -> None:
        """Test deep supervision returns correct number of outputs."""
        config = minimal_model_config.copy()
        config['enable_deep_supervision'] = True
        config['depth'] = 3

        model = ConvUNextModel(**config)
        outputs = model(sample_input_small)

        # Main output + (depth - 1) supervision outputs
        expected_count = 1 + (config['depth'] - 1)
        assert isinstance(outputs, list)
        assert len(outputs) == expected_count

    def test_deep_supervision_disabled_single_output(
        self,
        minimal_model_config: Dict[str, Any],
        sample_input_small: np.ndarray
    ) -> None:
        """Test disabled deep supervision returns single output."""
        config = minimal_model_config.copy()
        config['enable_deep_supervision'] = False

        model = ConvUNextModel(**config)
        output = model(sample_input_small)

        assert not isinstance(output, list)

    def test_deep_supervision_output_shapes(
        self,
        minimal_model_config: Dict[str, Any],
        sample_input_small: np.ndarray
    ) -> None:
        """Test deep supervision outputs have correct shapes."""
        config = minimal_model_config.copy()
        config['enable_deep_supervision'] = True
        config['include_top'] = True
        config['depth'] = 3
        config['output_channels'] = 2

        model = ConvUNextModel(**config)
        outputs = model(sample_input_small)

        # Main output at full resolution
        assert outputs[0].shape == (2, 64, 64, 2)

        # Auxiliary outputs at reduced resolutions
        for i, out in enumerate(outputs[1:], 1):
            assert out.shape[-1] == 2  # All have output_channels

    def test_deep_supervision_with_include_top_false(
        self,
        minimal_model_config: Dict[str, Any],
        sample_input_small: np.ndarray
    ) -> None:
        """Test deep supervision with include_top=False returns features."""
        config = minimal_model_config.copy()
        config['enable_deep_supervision'] = True
        config['include_top'] = False
        config['depth'] = 3

        model = ConvUNextModel(**config)
        outputs = model(sample_input_small)

        assert isinstance(outputs, list)
        # Main output should have initial_filters channels
        assert outputs[0].shape[-1] == config['initial_filters']


class TestConvUNextModelComputeOutputShape:
    """Test suite for compute_output_shape method."""

    def test_compute_output_shape_basic(
        self,
        minimal_model_config: Dict[str, Any]
    ) -> None:
        """Test compute_output_shape returns correct shape."""
        model = ConvUNextModel(**minimal_model_config)
        input_shape = (None, 64, 64, 3)

        output_shape = model.compute_output_shape(input_shape)

        expected = (None, 64, 64, minimal_model_config['output_channels'])
        assert output_shape == expected

    def test_compute_output_shape_with_deep_supervision(
        self,
        minimal_model_config: Dict[str, Any]
    ) -> None:
        """Test compute_output_shape with deep supervision."""
        config = minimal_model_config.copy()
        config['enable_deep_supervision'] = True
        config['depth'] = 3

        model = ConvUNextModel(**config)
        input_shape = (None, 64, 64, 3)

        output_shapes = model.compute_output_shape(input_shape)

        assert isinstance(output_shapes, list)
        # Main + (depth - 1) auxiliary
        assert len(output_shapes) == config['depth']

    def test_compute_output_shape_include_top_false(
        self,
        minimal_model_config: Dict[str, Any]
    ) -> None:
        """Test compute_output_shape with include_top=False."""
        config = minimal_model_config.copy()
        config['include_top'] = False

        model = ConvUNextModel(**config)
        input_shape = (None, 64, 64, 3)

        output_shape = model.compute_output_shape(input_shape)

        # Should return initial_filters (decoder features)
        expected = (None, 64, 64, config['initial_filters'])
        assert output_shape == expected


class TestConvUNextModelSerialization:
    """Test suite for model serialization."""

    def test_get_config_complete(
        self,
        minimal_model_config: Dict[str, Any]
    ) -> None:
        """Test get_config returns all constructor arguments."""
        model = ConvUNextModel(**minimal_model_config)
        config = model.get_config()

        assert 'depth' in config
        assert 'initial_filters' in config
        assert 'filter_multiplier' in config
        assert 'blocks_per_level' in config
        assert 'convnext_version' in config
        assert 'include_top' in config
        assert 'enable_deep_supervision' in config
        assert 'output_channels' in config

    def test_from_config_reconstruction(
        self,
        minimal_model_config: Dict[str, Any]
    ) -> None:
        """Test model can be reconstructed from config."""
        original = ConvUNextModel(**minimal_model_config)
        config = original.get_config()

        reconstructed = ConvUNextModel.from_config(config)

        assert reconstructed.depth == original.depth
        assert reconstructed.initial_filters == original.initial_filters
        assert reconstructed.include_top == original.include_top

    def test_serialization_cycle(
        self,
        minimal_model_config: Dict[str, Any],
        sample_input_small: np.ndarray
    ) -> None:
        """Test full save/load cycle preserves functionality."""
        model = ConvUNextModel(**minimal_model_config)
        original_output = model(sample_input_small)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_model.keras')
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)

        loaded_output = loaded_model(sample_input_small)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(original_output),
            keras.ops.convert_to_numpy(loaded_output),
            rtol=1e-5, atol=1e-5,
            err_msg="Model outputs should match after serialization"
        )

    def test_serialization_with_deep_supervision(
        self,
        minimal_model_config: Dict[str, Any],
        sample_input_small: np.ndarray
    ) -> None:
        """Test serialization with deep supervision enabled."""
        config = minimal_model_config.copy()
        config['enable_deep_supervision'] = True

        model = ConvUNextModel(**config)
        original_outputs = model(sample_input_small)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_ds_model.keras')
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)

        loaded_outputs = loaded_model(sample_input_small)

        assert len(original_outputs) == len(loaded_outputs)

        for orig, loaded in zip(original_outputs, loaded_outputs):
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(orig),
                keras.ops.convert_to_numpy(loaded),
                rtol=1e-5, atol=1e-5,
                err_msg="Deep supervision outputs should match after serialization"
            )


class TestConvUNextModelVariants:
    """Test suite for model variants."""

    @pytest.mark.parametrize("variant", ['tiny', 'small', 'base', 'large', 'xlarge'])
    def test_variant_instantiation(self, variant: str) -> None:
        """Test all variants can be instantiated."""
        model = ConvUNextModel.from_variant(
            variant,
            input_shape=(64, 64, 3),
            output_channels=1
        )
        assert model is not None

    @pytest.mark.parametrize("variant", ['tiny', 'small', 'base'])
    def test_variant_forward_pass(
        self,
        variant: str,
        sample_input_small: np.ndarray
    ) -> None:
        """Test variant forward pass."""
        model = ConvUNextModel.from_variant(
            variant,
            input_shape=(64, 64, 3),
            output_channels=1
        )
        output = model(sample_input_small)

        assert output.shape == (2, 64, 64, 1)

    def test_invalid_variant_raises_error(self) -> None:
        """Test invalid variant raises error."""
        with pytest.raises(ValueError, match="Unknown variant"):
            ConvUNextModel.from_variant('invalid_variant')

    def test_variant_with_custom_overrides(self) -> None:
        """Test variant with custom parameter overrides."""
        model = ConvUNextModel.from_variant(
            'tiny',
            input_shape=(64, 64, 3),
            output_channels=5,
            use_bias=False,
            drop_path_rate=0.2
        )

        assert model.output_channels == 5
        assert model.use_bias is False
        assert model.drop_path_rate == 0.2


class TestCreateConvUNextVariant:
    """Test suite for factory function."""

    def test_factory_function_basic(
        self,
        sample_input_small: np.ndarray
    ) -> None:
        """Test factory function creates valid model."""
        model = create_convunext_variant(
            'tiny',
            input_shape=(64, 64, 3),
            output_channels=1
        )

        output = model(sample_input_small)
        assert output.shape == (2, 64, 64, 1)

    def test_factory_function_include_top_false(
        self,
        sample_input_small: np.ndarray
    ) -> None:
        """Test factory function with include_top=False."""
        model = create_convunext_variant(
            'tiny',
            input_shape=(64, 64, 3),
            output_channels=1,
            include_top=False
        )

        output = model(sample_input_small)
        # Should return features, not predictions
        assert output.shape[-1] == model.initial_filters


class TestCreateInferenceModel:
    """Test suite for inference model creation."""

    def test_create_inference_from_training(
        self,
        minimal_model_config: Dict[str, Any],
        sample_input_small: np.ndarray
    ) -> None:
        """Test creating inference model from training model."""
        config = minimal_model_config.copy()
        config['enable_deep_supervision'] = True

        training_model = ConvUNextModel(**config)
        training_model(sample_input_small)

        inference_model = create_inference_model_from_training_model(training_model)

        assert inference_model.enable_deep_supervision is False

        output = inference_model(sample_input_small)
        assert not isinstance(output, list)

    def test_inference_model_already_configured(
        self,
        minimal_model_config: Dict[str, Any],
        sample_input_small: np.ndarray
    ) -> None:
        """Test that already-configured model is returned unchanged."""
        config = minimal_model_config.copy()
        config['enable_deep_supervision'] = False

        model = ConvUNextModel(**config)
        model(sample_input_small)

        result = create_inference_model_from_training_model(model)

        # Should return the same model
        assert result is model


class TestConvUNextModelTrainingIntegration:
    """Test suite for training integration."""

    def test_compile_and_fit(
        self,
        minimal_model_config: Dict[str, Any]
    ) -> None:
        """Test model can be compiled and trained."""
        model = ConvUNextModel(**minimal_model_config)

        model.compile(
            optimizer='adam',
            loss='mse',
        )

        x_train = np.random.randn(4, 64, 64, 3).astype(np.float32)
        y_train = np.random.randn(4, 64, 64, 1).astype(np.float32)

        history = model.fit(
            x_train, y_train,
            epochs=1,
            batch_size=2,
            verbose=0
        )

        assert 'loss' in history.history
        assert len(history.history['loss']) == 1

    def test_compile_and_fit_with_deep_supervision(
        self,
        minimal_model_config: Dict[str, Any]
    ) -> None:
        """Test training with deep supervision."""
        config = minimal_model_config.copy()
        config['enable_deep_supervision'] = True
        config['depth'] = 3

        model = ConvUNextModel(**config)

        # Multiple outputs require multiple losses
        num_outputs = config['depth']  # 1 main + (depth-1) aux
        losses = ['mse'] * num_outputs
        loss_weights = [1.0] + [0.3] * (num_outputs - 1)

        model.compile(
            optimizer='adam',
            loss=losses,
            loss_weights=loss_weights,
        )

        x_train = np.random.randn(4, 64, 64, 3).astype(np.float32)

        # Create targets for each output
        y_main = np.random.randn(4, 64, 64, 1).astype(np.float32)
        y_aux = [
            np.random.randn(4, 64 // (2 ** i), 64 // (2 ** i), 1).astype(np.float32)
            for i in range(1, num_outputs)
        ]
        y_train = [y_main] + y_aux

        history = model.fit(
            x_train, y_train,
            epochs=1,
            batch_size=2,
            verbose=0
        )

        assert 'loss' in history.history

    def test_save_and_load_trained_model(
        self,
        minimal_model_config: Dict[str, Any]
    ) -> None:
        """Test saving and loading a trained model."""
        model = ConvUNextModel(**minimal_model_config)
        model.compile(optimizer='adam', loss='mse')

        x_train = np.random.randn(4, 64, 64, 3).astype(np.float32)
        y_train = np.random.randn(4, 64, 64, 1).astype(np.float32)

        model.fit(x_train, y_train, epochs=1, verbose=0)

        x_test = np.random.randn(2, 64, 64, 3).astype(np.float32)
        original_pred = model.predict(x_test, verbose=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'trained_model.keras')
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)

        loaded_pred = loaded_model.predict(x_test, verbose=0)

        np.testing.assert_allclose(
            original_pred,
            loaded_pred,
            rtol=1e-5, atol=1e-5,
            err_msg="Predictions should match after save/load"
        )


class TestConvUNextModelEdgeCases:
    """Test suite for edge cases."""

    def test_single_channel_input(
        self,
        minimal_model_config: Dict[str, Any]
    ) -> None:
        """Test model with single channel input."""
        config = minimal_model_config.copy()
        config['input_shape'] = (64, 64, 1)

        model = ConvUNextModel(**config)
        inputs = np.random.randn(2, 64, 64, 1).astype(np.float32)
        output = model(inputs)

        assert output.shape == (2, 64, 64, 1)

    def test_multi_channel_output(
        self,
        minimal_model_config: Dict[str, Any],
        sample_input_small: np.ndarray
    ) -> None:
        """Test model with multiple output channels."""
        config = minimal_model_config.copy()
        config['output_channels'] = 10

        model = ConvUNextModel(**config)
        output = model(sample_input_small)

        assert output.shape[-1] == 10

    def test_minimum_depth(
        self,
        sample_input_small: np.ndarray
    ) -> None:
        """Test model with minimum depth."""
        model = ConvUNextModel(
            input_shape=(64, 64, 3),
            depth=2,
            initial_filters=16,
            blocks_per_level=1,
            output_channels=1
        )

        output = model(sample_input_small)
        assert output.shape == (2, 64, 64, 1)

    def test_no_regularization(
        self,
        minimal_model_config: Dict[str, Any],
        sample_input_small: np.ndarray
    ) -> None:
        """Test model without regularization."""
        config = minimal_model_config.copy()
        config['kernel_regularizer'] = None
        config['drop_path_rate'] = 0.0

        model = ConvUNextModel(**config)
        output = model(sample_input_small)

        assert output is not None

    def test_with_regularization(
        self,
        minimal_model_config: Dict[str, Any],
        sample_input_small: np.ndarray
    ) -> None:
        """Test model with L2 regularization."""
        config = minimal_model_config.copy()
        config['kernel_regularizer'] = 'l2'

        model = ConvUNextModel(**config)
        output = model(sample_input_small)

        assert output is not None

    @pytest.mark.parametrize("activation", ['linear', 'sigmoid', 'softmax'])
    def test_different_final_activations(
        self,
        activation: str,
        minimal_model_config: Dict[str, Any],
        sample_input_small: np.ndarray
    ) -> None:
        """Test model with different final activations."""
        config = minimal_model_config.copy()
        config['final_activation'] = activation

        model = ConvUNextModel(**config)
        output = model(sample_input_small)

        assert output.shape[-1] == config['output_channels']


class TestConvUNextModelWeightLoading:
    """Test suite for weight loading scenarios."""

    def test_load_weights_by_name(
        self,
        minimal_model_config: Dict[str, Any],
        sample_input_small: np.ndarray
    ) -> None:
        """Test loading weights by name."""
        model1 = ConvUNextModel(**minimal_model_config)
        model1(sample_input_small)

        model2 = ConvUNextModel(**minimal_model_config)
        model2(sample_input_small)

        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = os.path.join(tmpdir, 'weights.weights.h5')
            model1.save_weights(weights_path)
            model2.load_weights(weights_path)

        output1 = model1(sample_input_small)
        output2 = model2(sample_input_small)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output1),
            keras.ops.convert_to_numpy(output2),
            rtol=1e-6, atol=1e-6,
            err_msg="Outputs should match after weight loading"
        )

    def test_weight_compatibility_include_top_true_to_false(
        self,
        minimal_model_config: Dict[str, Any],
        sample_input_small: np.ndarray
    ) -> None:
        """Test weight loading from include_top=True to include_top=False."""
        config = minimal_model_config.copy()

        # Train with include_top=True
        model_with_top = ConvUNextModel(**{**config, 'include_top': True})
        model_with_top(sample_input_small)

        # Create model with include_top=False
        model_without_top = ConvUNextModel(**{**config, 'include_top': False})
        model_without_top(sample_input_small)

        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = os.path.join(tmpdir, 'weights.weights.h5')
            model_with_top.save_weights(weights_path)
            # Load with skip_mismatch for compatibility
            model_without_top.load_weights(
                filepath=weights_path,
                skip_mismatch=True
            )

        # Both should work after weight transfer
        output = model_without_top(sample_input_small)
        assert output is not None

# ---------------------------------------------------------------------
# Residual + stochastic-depth wiring
# ---------------------------------------------------------------------

class _ZeroBranch(keras.layers.Layer):
    """Stand-in for a ConvNeXt block whose branch output is identically zero.

    Substituting this for every ConvNeXt block turns each block loop into an
    exact identity *if and only if* the caller adds a residual. Without a
    residual the signal is annihilated at the first block.
    """

    def call(
            self,
            inputs: keras.KerasTensor,
            training: bool = None
    ) -> keras.KerasTensor:
        return keras.ops.zeros_like(inputs)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape)


def _zero_all_convnext_branches(model: ConvUNextModel) -> int:
    """Replace every ConvNeXt block in ``model`` with a zero-output branch.

    The decoder's ``decoder_blocks[idx][0]`` is the 1x1 channel-adjust Conv2D,
    not a ConvNeXt block, and is deliberately left in place.

    Returns the number of blocks replaced.
    """
    replaced = 0
    for stage in model.encoder_stages:
        for i in range(len(stage)):
            stage[i] = _ZeroBranch()
            replaced += 1
    for i in range(len(model.bottleneck_blocks)):
        model.bottleneck_blocks[i] = _ZeroBranch()
        replaced += 1
    for stage in model.decoder_blocks:
        for i in range(1, len(stage)):
            stage[i] = _ZeroBranch()
            replaced += 1
    return replaced


class TestConvUNextResidualAndDropPathWiring:
    """`ConvNextV1Block`/`ConvNextV2Block` are the residual BRANCH only.

    The caller must supply ``residual + StochasticDepth(rate)(block(x))``
    (``convnext_v1_block.py:103-121``). These tests pin that wiring.
    """

    @staticmethod
    def _model(drop_path_rate: float) -> ConvUNextModel:
        return ConvUNextModel(
            input_shape=(32, 32, 3),
            depth=2,
            initial_filters=8,
            blocks_per_level=2,
            convnext_version='v2',
            drop_path_rate=drop_path_rate,
            output_channels=1,
        )

    def test_residual_skip_is_present(self) -> None:
        """Zeroing every block branch must leave the model input-dependent.

        With all ConvNeXt branches returning exactly zero, the ONLY route from
        the input to the output is the additive skip inside each block loop
        (plus the stem / down / up / adjust convs it feeds). Without the
        residual, ``x`` becomes exactly 0 at the first encoder block, every
        stored skip connection is 0, and the output degenerates to a constant
        that does not depend on the input at all -- so the two probe inputs
        would produce bit-identical outputs.
        """
        model = self._model(drop_path_rate=0.0)
        x1 = np.random.randn(2, 32, 32, 3).astype(np.float32)
        x2 = np.random.randn(2, 32, 32, 3).astype(np.float32)

        model(x1, training=False)  # build

        replaced = _zero_all_convnext_branches(model)
        # depth=2, blocks_per_level=2 -> 4 encoder + 2 bottleneck + 4 decoder.
        assert replaced == 10, f"expected 10 ConvNeXt blocks, replaced {replaced}"

        out1 = np.array(model(x1, training=False))
        out2 = np.array(model(x2, training=False))

        spread = float(np.max(np.abs(out1 - out2)))
        assert spread > 1e-4, (
            "output does not depend on the input once every ConvNeXt branch is "
            f"zeroed (max|out(x1) - out(x2)| = {spread}); the residual skip is "
            "missing from at least one block loop"
        )

    def test_drop_path_layers_carry_the_schedule(self) -> None:
        """The drop-path schedule must reach StochasticDepth, not dropout_rate."""
        rate = 0.4  # NON-ZERO: an all-zeros schedule is invariant under
        # shift and reversal and would make this proof vacuous.
        model = self._model(drop_path_rate=rate)
        depth, bpl = model.depth, model.blocks_per_level

        def scheduled(level: int, block_idx: int) -> float:
            return rate * (level * bpl + block_idx) / (depth * bpl)

        all_drop_paths = []

        # Encoder: linear ramp with depth.
        assert len(model.encoder_drop_paths) == depth
        for level, stage in enumerate(model.encoder_drop_paths):
            assert len(stage) == len(model.encoder_stages[level])
            for block_idx, dp in enumerate(stage):
                assert isinstance(dp, StochasticDepth)
                assert dp.drop_path_rate == pytest.approx(
                    scheduled(level, block_idx)
                ), f"encoder L{level} blk{block_idx}"
                all_drop_paths.append(dp)

        # Bottleneck: deliberately FLAT at the max rate (pre-existing shape).
        assert len(model.bottleneck_drop_paths) == len(model.bottleneck_blocks)
        for dp in model.bottleneck_drop_paths:
            assert isinstance(dp, StochasticDepth)
            assert dp.drop_path_rate == pytest.approx(rate)
            all_drop_paths.append(dp)

        # Decoder: mirrors the encoder ramp; index idx corresponds to
        # level = depth - 1 - idx. decoder_blocks[idx][0] is the channel-adjust
        # Conv2D and has no drop-path partner.
        assert len(model.decoder_drop_paths) == depth
        for idx, stage in enumerate(model.decoder_drop_paths):
            level = depth - 1 - idx
            assert len(stage) == len(model.decoder_blocks[idx]) - 1
            for block_idx, dp in enumerate(stage):
                assert isinstance(dp, StochasticDepth)
                assert dp.drop_path_rate == pytest.approx(
                    scheduled(level, block_idx)
                ), f"decoder L{level} blk{block_idx}"
                all_drop_paths.append(dp)

        # Every drop-path is its own instance (no shared object across blocks).
        assert len({id(dp) for dp in all_drop_paths}) == len(all_drop_paths)
        # And the schedule is genuinely non-zero somewhere.
        assert max(dp.drop_path_rate for dp in all_drop_paths) > 0.0

    def test_blocks_own_dropout_rate_is_zero(self) -> None:
        """The block's own dropout_rate is ordinary MLP dropout, not drop-path."""
        model = self._model(drop_path_rate=0.4)

        for stage in model.encoder_stages:
            for block in stage:
                assert block.dropout_rate == 0.0
        for block in model.bottleneck_blocks:
            assert block.dropout_rate == 0.0
        for stage in model.decoder_blocks:
            for block in stage[1:]:
                assert block.dropout_rate == 0.0

    def test_inference_differs_from_pre_fix_wiring(self) -> None:
        """The residual CHANGES inference. This is deliberate, not a regression.

        This is a CHARACTERIZATION test, not a bit-identity claim. It runs the
        same built layers with the same weights and the same input under both
        the pre-fix wiring (``x = block(x)``, the documented bug) and the
        canonical wiring, at explicit ``training=False``, and pins that they
        differ. drop_path_rate is 0.0 so StochasticDepth is an exact identity
        and the measured delta is attributable to the residual alone.
        """
        model = self._model(drop_path_rate=0.0)
        x = np.random.randn(2, 32, 32, 3).astype(np.float32)
        model(x, training=False)  # build

        h = model.stem(x, training=False)

        # Canonical wiring (current code).
        new = h
        for block, drop_path in zip(
                model.encoder_stages[0], model.encoder_drop_paths[0]
        ):
            residual = new
            y = block(new, training=False)
            y = drop_path(y, training=False)
            new = keras.layers.add([residual, y])

        # Pre-fix wiring, verbatim: `x = layer(x, training=training)`.
        old = h
        for block in model.encoder_stages[0]:
            old = block(old, training=False)

        delta = float(np.max(np.abs(np.array(new) - np.array(old))))
        assert delta > 1e-6, (
            "encoder stage 0 is unchanged by the residual at training=False "
            f"(max|delta| = {delta}); the residual is not being applied"
        )
