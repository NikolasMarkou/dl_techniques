"""
Comprehensive pytest test suite for NanoVLM (Vision-Language Model).

This module provides extensive testing for the NanoVLM implementation including:
- Model configuration validation and serialization
- Model initialization and parameter validation
- Architecture building and component integration (tested via forward pass)
- Forward pass functionality with different input formats (dict vs tuple)
- Text generation capabilities and sampling strategies
- Serialization and deserialization of complete models
- Error handling and edge cases
- Factory function testing for different model variants
- Integration testing with vision encoder, text decoder, and modality projection
- Training vs inference behavior
- Multi-modal input processing and output generation
"""

import pytest
import numpy as np
import keras
from keras import ops
import tempfile
import os

from dl_techniques.models.nano_vlm import (
    NanoVLM,
    create_nanovlm,
    create_nanovlm_mini,
    create_nanovlm_base,
    create_nanovlm_222m,
    get_available_variants,
    get_variant_info,
    NANOVLM_CONFIGS
)
# Component imports for type checking
from dl_techniques.models.vision_encoder import VisionEncoder
from dl_techniques.models.text_decoder import TextDecoder
from dl_techniques.layers.modality_projection import ModalityProjection


def get_expected_combined_seq_len(model, text_input_shape):
    """Helper function to calculate the correct combined sequence length."""
    vision_config = model.vision_config
    projection_config = model.projection_config

    vision_patches = (vision_config['img_size'] // vision_config['patch_size']) ** 2
    scale_factor = projection_config.get('scale_factor', 1)
    shuffled_vision_patches = vision_patches // (scale_factor ** 2)
    # Add 1 for the dummy CLS token that is prepended before projection
    projected_vision_seq_len = shuffled_vision_patches + 1

    text_seq_len = text_input_shape[1]
    return projected_vision_seq_len + text_seq_len


class TestNanoVLMConfigurations:
    """Test NanoVLM configuration validation and setup."""

    def test_available_variants(self):
        """Test that all expected variants are available."""
        variants = get_available_variants()
        expected_variants = ['nanovlm_mini', 'nanovlm_base', 'nanovlm_222m']
        assert isinstance(variants, list)
        assert set(variants) == set(expected_variants)

    def test_variant_info_structure(self):
        """Test that variant info has required structure."""
        for variant_name in get_available_variants():
            info = get_variant_info(variant_name)
            required_keys = ['vision_config', 'language_config', 'projection_config', 'total_params']
            for key in required_keys:
                assert key in info, f"Missing key '{key}' in {variant_name}"

    def test_variant_info_invalid_name(self):
        """Test that invalid variant names raise errors."""
        with pytest.raises(ValueError, match="Unknown variant"):
            get_variant_info("invalid_variant")

    def test_config_dimension_consistency(self):
        """Test that configurations have consistent dimensions."""
        for variant_name in get_available_variants():
            config = NANOVLM_CONFIGS[variant_name]
            vision_config = config['vision_config']
            language_config = config['language_config']
            projection_config = config['projection_config']
            assert vision_config['embed_dim'] == projection_config['input_dim']
            assert language_config['hidden_dim'] == projection_config['output_dim']


class TestNanoVLMInitialization:
    """Test NanoVLM model initialization and parameter validation."""

    @pytest.fixture
    def valid_configs(self):
        """Provide valid configuration dictionaries for the NanoVLM constructor."""
        # Get the full config block for the mini variant
        full_config = NANOVLM_CONFIGS['nanovlm_mini']
        # Return a dictionary containing ONLY the keys expected by NanoVLM.__init__
        return {
            "vision_config": full_config["vision_config"],
            "language_config": full_config["language_config"],
            "projection_config": full_config["projection_config"],
        }

    def test_basic_initialization(self, valid_configs):
        """Test basic NanoVLM initialization and component creation."""
        model = NanoVLM(**valid_configs)

        # Check stored configurations
        assert model.vision_config == valid_configs['vision_config']
        assert model.language_config == valid_configs['language_config']
        assert model.projection_config == valid_configs['projection_config']
        assert not model.built

        # Components should be created and are instances of correct classes in __init__
        assert isinstance(model.vision_encoder, VisionEncoder)
        assert isinstance(model.text_decoder, TextDecoder)
        assert isinstance(model.modality_projection, ModalityProjection)
        assert isinstance(model.output_projection, keras.layers.Dense)

    def test_custom_parameters_initialization(self, valid_configs):
        """Test NanoVLM initialization with custom parameters."""
        model = NanoVLM(
            **valid_configs,
            vocab_size=50000,
            dropout_rate=0.2
        )
        assert model.vocab_size == 50000
        assert model.dropout_rate == 0.2

    def test_config_validation_missing_keys(self, valid_configs):
        """Test that missing required config keys raise errors."""
        # Test missing vision config key
        configs = valid_configs.copy()
        invalid_vision_config = configs['vision_config'].copy()
        del invalid_vision_config['embed_dim']

        with pytest.raises(ValueError, match="Missing required vision config key"):
            NanoVLM(
                vision_config=invalid_vision_config,
                language_config=configs['language_config'],
                projection_config=configs['projection_config']
            )

    def test_dimension_mismatch_validation(self, valid_configs):
        """Test that dimension mismatches raise errors."""
        # Test vision embed_dim vs projection input_dim mismatch
        configs = valid_configs.copy()
        invalid_projection_config = configs['projection_config'].copy()
        invalid_projection_config['input_dim'] = 512  # Different from vision embed_dim=384

        with pytest.raises(ValueError, match="Vision embed_dim.*must match.*projection input_dim"):
            NanoVLM(
                vision_config=configs['vision_config'],
                language_config=configs['language_config'],
                projection_config=invalid_projection_config
            )

    def test_vocab_size_consistency_warning(self, valid_configs):
        """Test vocab_size consistency handling."""
        configs = valid_configs.copy()
        language_config = configs['language_config'].copy()
        language_config['vocab_size'] = 25000  # Different from model vocab_size

        model = NanoVLM(
            vision_config=configs['vision_config'],
            language_config=language_config,
            projection_config=configs['projection_config'],
            vocab_size=32000
        )
        assert model.vocab_size == 32000
        assert model.language_config['vocab_size'] == 32000


class TestNanoVLMForwardPass:
    """Test NanoVLM forward pass, which also implicitly tests automatic building."""

    @pytest.fixture
    def model(self):
        """Create a NanoVLM model for testing. It will be built on first use."""
        return create_nanovlm_mini()

    @pytest.fixture
    def sample_inputs(self):
        """Create sample inputs for testing."""
        batch_size = 2
        images = keras.random.normal((batch_size, 224, 224, 3))
        text_tokens = ops.cast(keras.random.uniform((batch_size, 10), minval=0, maxval=1000, dtype='float32'), "int32")
        return {'images': images, 'text_tokens': text_tokens, 'batch_size': batch_size}

    def test_forward_pass_dict_input(self, model, sample_inputs):
        """Test forward pass with dictionary input, which triggers the build."""
        inputs = {'images': sample_inputs['images'], 'text_tokens': sample_inputs['text_tokens']}

        assert not model.built
        logits = model(inputs, training=False)
        assert model.built

        combined_seq_len = get_expected_combined_seq_len(model, sample_inputs['text_tokens'].shape)
        assert logits.shape == (sample_inputs['batch_size'], combined_seq_len, model.vocab_size)

        # Test that sub-layer configs are correct after build
        assert model.output_projection.units == model.vocab_size

    @pytest.mark.parametrize("input_type", [tuple, list])
    def test_forward_pass_sequence_input(self, model, sample_inputs, input_type):
        """Test forward pass with tuple and list input formats."""
        inputs = input_type([sample_inputs['images'], sample_inputs['text_tokens']])
        logits = model(inputs, training=False)
        combined_seq_len = get_expected_combined_seq_len(model, sample_inputs['text_tokens'].shape)
        assert logits.shape == (sample_inputs['batch_size'], combined_seq_len, model.vocab_size)

    def test_forward_pass_with_optional_inputs(self, model, sample_inputs):
        """Test forward pass with optional input components."""
        text_tokens = sample_inputs['text_tokens']
        inputs = {
            'images': sample_inputs['images'],
            'text_tokens': text_tokens,
            'token_type_ids': ops.zeros_like(text_tokens),
            'position_ids': ops.arange(text_tokens.shape[1])[None, :] * ops.ones((sample_inputs['batch_size'], 1), dtype='int32'),
        }
        logits = model(inputs, training=False)
        combined_seq_len = get_expected_combined_seq_len(model, text_tokens.shape)
        assert logits.shape == (sample_inputs['batch_size'], combined_seq_len, model.vocab_size)

    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("seq_len", [5, 25])
    def test_forward_pass_dynamic_shapes(self, model, batch_size, seq_len):
        """Test forward pass with different batch sizes and sequence lengths."""
        images = keras.random.normal((batch_size, 224, 224, 3))
        text_tokens = ops.zeros((batch_size, seq_len), dtype="int32")
        inputs = {'images': images, 'text_tokens': text_tokens}
        logits = model(inputs, training=False)
        combined_seq_len = get_expected_combined_seq_len(model, text_tokens.shape)
        assert logits.shape == (batch_size, combined_seq_len, model.vocab_size)

    def test_invalid_input_formats(self, model):
        """Test that invalid input formats raise errors."""
        images = keras.random.normal((2, 224, 224, 3))
        text_tokens = ops.zeros((2, 10), dtype="int32")
        with pytest.raises(ValueError, match="Input dict must contain 'images' and 'text_tokens' keys"):
            model({'images': images})
        with pytest.raises(ValueError, match="Inputs must be either a dict.*or a tuple/list"):
            model([images, text_tokens, images])

    def test_compute_output_shape(self, model):
        """Test compute_output_shape method."""
        input_shape_dict = {'images': (4, 224, 224, 3), 'text_tokens': (4, 10)}
        output_shape = model.compute_output_shape(input_shape_dict)
        combined_seq_len = get_expected_combined_seq_len(model, input_shape_dict['text_tokens'])
        assert output_shape == (4, combined_seq_len, model.vocab_size)


class TestNanoVLMGeneration:
    """Test NanoVLM text generation functionality."""

    @pytest.fixture
    def model(self):
        """Create a NanoVLM model for testing."""
        return create_nanovlm_mini()

    def test_generate_basic(self, model):
        """Test basic text generation functionality."""
        image = keras.random.normal((1, 224, 224, 3))
        prompt_tokens = ops.zeros((1, 5), dtype="int32")
        generated_tokens = model.generate(
            image=image,
            prompt_tokens=prompt_tokens,
            max_length=10
        )
        assert generated_tokens.shape[0] == 1
        assert generated_tokens.shape[1] >= prompt_tokens.shape[1]
        assert generated_tokens.shape[1] <= prompt_tokens.shape[1] + 10

    def test_generate_early_stopping(self, model):
        """Test early stopping with EOS token."""
        image = keras.random.normal((1, 224, 224, 3))
        prompt_tokens = ops.zeros((1, 2), dtype="int32")
        eos_token_id = 2

        # Mock the sampling function to force an EOS token on the second step
        original_sample = model._sample_next_token
        call_count = 0
        def mock_sample(logits, temperature, top_k):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                return ops.convert_to_tensor(eos_token_id, dtype='int32')
            return original_sample(logits, temperature, top_k)
        model._sample_next_token = mock_sample

        generated = model.generate(image=image, prompt_tokens=prompt_tokens, max_length=20, eos_token_id=eos_token_id)

        assert generated.shape[1] == 4  # 2 prompt + 1 normal + 1 EOS
        assert ops.equal(generated[0, -1], eos_token_id)

        # Restore original method
        model._sample_next_token = original_sample

    def test_sample_next_token_methods(self, model):
        """Test _sample_next_token with different parameters."""
        logits_1d = keras.random.normal((model.vocab_size,))

        # Test greedy sampling (top_k=0)
        token_greedy = model._sample_next_token(logits_1d, temperature=1.0, top_k=0)
        expected_greedy = ops.argmax(logits_1d, axis=-1)
        assert ops.equal(token_greedy, expected_greedy)
        assert token_greedy.shape == ()

        # Test top-k sampling
        token_topk = model._sample_next_token(logits_1d, temperature=1.0, top_k=10)
        assert token_topk.shape == ()
        assert token_topk.dtype == expected_greedy.dtype


class TestNanoVLMSerialization:
    """Test NanoVLM serialization and deserialization, the most critical test."""

    def test_model_save_load_cycle(self):
        """Test complete save/load cycle for a given model variant."""
        model = create_nanovlm_mini()
        images = keras.random.normal((2, 224, 224, 3))
        text_tokens = ops.zeros((2, 5), dtype="int32")
        inputs = {'images': images, 'text_tokens': text_tokens}

        # The first call builds the model and creates weights
        original_outputs = model(inputs, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_nano_vlm.keras')
            model.save(model_path)

            loaded_model = keras.models.load_model(model_path)
            loaded_outputs = loaded_model(inputs, training=False)

            # Check configs are restored
            assert loaded_model.vocab_size == model.vocab_size
            assert loaded_model.vision_config['embed_dim'] == model.vision_config['embed_dim']

            # Check outputs are numerically close
            np.testing.assert_allclose(
                ops.convert_to_numpy(original_outputs),
                ops.convert_to_numpy(loaded_outputs),
                rtol=1e-5, atol=1e-6
            )


class TestNanoVLMFactoryFunctions:
    """Test factory functions for creating NanoVLM models."""

    def test_parameter_counts_scaling(self):
        """Test that parameter counts scale appropriately across variants."""
        models = {
            'mini': create_nanovlm_mini(),
            'base': create_nanovlm_base(),
            'large': create_nanovlm_222m()
        }
        param_counts = {}

        # Create dummy inputs to build the models
        images = keras.random.normal((1, 224, 224, 3))
        text_tokens = ops.zeros((1, 10), dtype="int32")

        for name, model in models.items():
            # Build the model by calling it, then count params
            _ = model({'images': images, 'text_tokens': text_tokens})
            param_counts[name] = model.count_params()

        # Larger models should have more parameters
        assert param_counts['mini'] > 0
        assert param_counts['base'] > param_counts['mini']
        assert param_counts['large'] > param_counts['base']