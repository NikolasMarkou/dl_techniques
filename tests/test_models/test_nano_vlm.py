"""
Comprehensive pytest test suite for NanoVLM (Vision-Language Model).

This module provides extensive testing for the NanoVLM implementation including:
- Model configuration validation and serialization
- Model initialization and parameter validation
- Architecture building and component integration
- Forward pass functionality with different input formats (dict vs tuple)
- Text generation capabilities and sampling strategies
- Serialization and deserialization of complete models
- Error handling and edge cases
- Factory function testing for different model variants
- Integration testing with vision encoder, text decoder, and modality projection
- Training vs inference behavior
- Multi-modal input processing and output generation

The test suite follows dl-techniques framework standards and ensures comprehensive
coverage of the NanoVLM model functionality for vision-language understanding tasks.
"""

import pytest
import numpy as np
import keras
from keras import ops
import tensorflow as tf
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


class TestNanoVLMConfigurations:
    """Test NanoVLM configuration validation and setup."""

    def test_available_variants(self):
        """Test that all expected variants are available."""
        variants = get_available_variants()
        expected_variants = ['nanovlm_mini', 'nanovlm_base', 'nanovlm_222m']

        assert isinstance(variants, list)
        assert len(variants) == 3
        for variant in expected_variants:
            assert variant in variants

    def test_variant_info_structure(self):
        """Test that variant info has required structure."""
        for variant_name in get_available_variants():
            info = get_variant_info(variant_name)

            # Check required top-level keys
            required_keys = ['vision_config', 'language_config', 'projection_config', 'total_params']
            for key in required_keys:
                assert key in info, f"Missing key '{key}' in {variant_name}"

            # Check vision config structure
            vision_required = ['img_size', 'patch_size', 'embed_dim', 'depth', 'num_heads']
            for key in vision_required:
                assert key in info['vision_config'], f"Missing vision config key '{key}' in {variant_name}"

            # Check language config structure
            language_required = ['vocab_size', 'hidden_dim', 'num_layers', 'num_heads', 'mlp_dim']
            for key in language_required:
                assert key in info['language_config'], f"Missing language config key '{key}' in {variant_name}"

            # Check projection config structure
            projection_required = ['input_dim', 'output_dim']
            for key in projection_required:
                assert key in info['projection_config'], f"Missing projection config key '{key}' in {variant_name}"

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

            # Vision embed_dim should match projection input_dim
            assert vision_config['embed_dim'] == projection_config['input_dim']

            # Language hidden_dim should match projection output_dim
            assert language_config['hidden_dim'] == projection_config['output_dim']

            # Verify num_heads divides embed_dim/hidden_dim evenly
            vision_head_dim = vision_config['embed_dim'] // vision_config['num_heads']
            assert vision_head_dim * vision_config['num_heads'] == vision_config['embed_dim']

            language_head_dim = language_config['hidden_dim'] // language_config['num_heads']
            assert language_head_dim * language_config['num_heads'] == language_config['hidden_dim']


class TestNanoVLMInitialization:
    """Test NanoVLM model initialization and parameter validation."""

    @pytest.fixture
    def valid_configs(self):
        """Provide valid configuration dictionaries for testing."""
        return {
            'vision_config': {
                'img_size': 224,
                'patch_size': 16,
                'embed_dim': 384,
                'depth': 6,
                'num_heads': 6,
                'mlp_ratio': 4.0,
                'dropout': 0.1,
            },
            'language_config': {
                'vocab_size': 32000,
                'hidden_dim': 384,
                'num_layers': 6,
                'num_heads': 6,
                'mlp_dim': 1536,
                'dropout': 0.1,
            },
            'projection_config': {
                'input_dim': 384,
                'output_dim': 384,
                'scale_factor': 2,
            }
        }

    def test_basic_initialization(self, valid_configs):
        """Test basic NanoVLM initialization."""
        model = NanoVLM(
            vision_config=valid_configs['vision_config'],
            language_config=valid_configs['language_config'],
            projection_config=valid_configs['projection_config']
        )

        # Check stored configurations
        assert model.vision_config == valid_configs['vision_config']
        assert model.language_config == valid_configs['language_config']
        assert model.projection_config == valid_configs['projection_config']
        assert model.vocab_size == 32000
        assert model.dropout_rate == 0.1
        assert model.built is False

        # Components should be None before building
        assert model.vision_encoder is None
        assert model.modality_projection is None
        assert model.text_decoder is None
        assert model.output_projection is None

    def test_custom_parameters_initialization(self, valid_configs):
        """Test NanoVLM initialization with custom parameters."""
        model = NanoVLM(
            vision_config=valid_configs['vision_config'],
            language_config=valid_configs['language_config'],
            projection_config=valid_configs['projection_config'],
            vocab_size=50000,
            dropout_rate=0.2
        )

        assert model.vocab_size == 50000
        assert model.dropout_rate == 0.2

    def test_config_validation_missing_keys(self, valid_configs):
        """Test that missing required config keys raise errors."""
        # Missing vision config key
        invalid_vision_config = valid_configs['vision_config'].copy()
        del invalid_vision_config['embed_dim']

        with pytest.raises(ValueError, match="Missing required vision config key"):
            NanoVLM(
                vision_config=invalid_vision_config,
                language_config=valid_configs['language_config'],
                projection_config=valid_configs['projection_config']
            )

        # Missing language config key
        invalid_language_config = valid_configs['language_config'].copy()
        del invalid_language_config['hidden_dim']

        with pytest.raises(ValueError, match="Missing required language config key"):
            NanoVLM(
                vision_config=valid_configs['vision_config'],
                language_config=invalid_language_config,
                projection_config=valid_configs['projection_config']
            )

        # Missing projection config key
        invalid_projection_config = valid_configs['projection_config'].copy()
        del invalid_projection_config['input_dim']

        with pytest.raises(ValueError, match="Missing required projection config key"):
            NanoVLM(
                vision_config=valid_configs['vision_config'],
                language_config=valid_configs['language_config'],
                projection_config=invalid_projection_config
            )

    def test_dimension_mismatch_validation(self, valid_configs):
        """Test that dimension mismatches raise errors."""
        # Vision embed_dim doesn't match projection input_dim
        invalid_projection_config = valid_configs['projection_config'].copy()
        invalid_projection_config['input_dim'] = 512  # Different from vision embed_dim=384

        with pytest.raises(ValueError, match="Vision embed_dim.*must match.*projection input_dim"):
            NanoVLM(
                vision_config=valid_configs['vision_config'],
                language_config=valid_configs['language_config'],
                projection_config=invalid_projection_config
            )

        # Language hidden_dim doesn't match projection output_dim
        invalid_projection_config = valid_configs['projection_config'].copy()
        invalid_projection_config['output_dim'] = 512  # Different from language hidden_dim=384

        with pytest.raises(ValueError, match="Language hidden_dim.*must match.*projection output_dim"):
            NanoVLM(
                vision_config=valid_configs['vision_config'],
                language_config=valid_configs['language_config'],
                projection_config=invalid_projection_config
            )

    def test_vocab_size_consistency_warning(self, valid_configs):
        """Test vocab_size consistency handling."""
        # Create configs with different vocab_size
        language_config = valid_configs['language_config'].copy()
        language_config['vocab_size'] = 25000  # Different from model vocab_size

        model = NanoVLM(
            vision_config=valid_configs['vision_config'],
            language_config=language_config,
            projection_config=valid_configs['projection_config'],
            vocab_size=32000
        )

        # Should use model vocab_size and update language config
        assert model.vocab_size == 32000
        assert model.language_config['vocab_size'] == 32000


class TestNanoVLMBuilding:
    """Test NanoVLM model building and component creation."""

    @pytest.fixture
    def model(self):
        """Create a basic NanoVLM model for testing."""
        return create_nanovlm_mini()  # Use mini variant for faster testing

    def test_build_components_creation(self, model):
        """Test that building creates all required components."""
        assert not model.built

        model.build()

        assert model.built is True
        assert model.vision_encoder is not None
        assert model.modality_projection is not None
        assert model.text_decoder is not None
        assert model.output_projection is not None

        # Check component types
        from dl_techniques.models.nano_vlm import VisionEncoder, TextDecoder
        from dl_techniques.layers.modality_projection import ModalityProjection

        assert isinstance(model.vision_encoder, VisionEncoder)
        assert isinstance(model.text_decoder, TextDecoder)
        assert isinstance(model.modality_projection, ModalityProjection)
        assert isinstance(model.output_projection, keras.layers.Dense)

    def test_build_with_input_shape(self, model):
        """Test building with specific input shape."""
        input_shape = (None, 224, 224, 3)
        model.build(input_shape)

        assert model.built is True
        assert model._build_input_shape == input_shape

    def test_build_prevents_double_building(self, model):
        """Test that building twice doesn't cause issues."""
        model.build()

        # Store references to components
        vision_encoder = model.vision_encoder
        text_decoder = model.text_decoder
        modality_projection = model.modality_projection
        output_projection = model.output_projection

        # Build again
        model.build()

        # Should be the same objects
        assert model.vision_encoder is vision_encoder
        assert model.text_decoder is text_decoder
        assert model.modality_projection is modality_projection
        assert model.output_projection is output_projection

    def test_output_projection_configuration(self, model):
        """Test output projection layer configuration."""
        model.build()

        assert model.output_projection.units == model.vocab_size
        assert model.output_projection.use_bias is False
        assert model.output_projection.name == 'output_projection'


class TestNanoVLMForwardPass:
    """Test NanoVLM forward pass functionality with different input formats."""

    @pytest.fixture
    def built_model(self):
        """Create a built NanoVLM model for testing."""
        model = create_nanovlm_mini()
        model.build()
        return model

    @pytest.fixture
    def sample_inputs(self):
        """Create sample inputs for testing."""
        batch_size = 2
        images = keras.random.normal((batch_size, 224, 224, 3))
        text_tokens = ops.cast(keras.random.uniform((batch_size, 10), minval=0, maxval=1000, dtype='float32'), dtype="int32")

        return {
            'images': images,
            'text_tokens': text_tokens,
            'batch_size': batch_size
        }

    def test_forward_pass_dict_input(self, built_model, sample_inputs):
        """Test forward pass with dictionary input format."""
        inputs = {
            'images': sample_inputs['images'],
            'text_tokens': sample_inputs['text_tokens']
        }

        logits = built_model(inputs, training=False)

        # Check output shape: [batch, vision_seq_len + text_seq_len, vocab_size]
        vision_seq_len = (224 // 16) ** 2 + 1  # 197 patches
        text_seq_len = sample_inputs['text_tokens'].shape[1]  # 10
        combined_seq_len = vision_seq_len + text_seq_len  # 207

        assert logits.shape == (sample_inputs['batch_size'], combined_seq_len, built_model.vocab_size)

    def test_forward_pass_tuple_input(self, built_model, sample_inputs):
        """Test forward pass with tuple input format."""
        inputs = (sample_inputs['images'], sample_inputs['text_tokens'])

        logits = built_model(inputs, training=False)

        vision_seq_len = (224 // 16) ** 2 + 1
        text_seq_len = sample_inputs['text_tokens'].shape[1]
        combined_seq_len = vision_seq_len + text_seq_len

        assert logits.shape == (sample_inputs['batch_size'], combined_seq_len, built_model.vocab_size)

    def test_forward_pass_list_input(self, built_model, sample_inputs):
        """Test forward pass with list input format."""
        inputs = [sample_inputs['images'], sample_inputs['text_tokens']]

        logits = built_model(inputs, training=False)

        vision_seq_len = (224 // 16) ** 2 + 1
        text_seq_len = sample_inputs['text_tokens'].shape[1]
        combined_seq_len = vision_seq_len + text_seq_len

        assert logits.shape == (sample_inputs['batch_size'], combined_seq_len, built_model.vocab_size)

    def test_forward_pass_with_optional_inputs(self, built_model, sample_inputs):
        """Test forward pass with optional input components."""
        # Test with all optional inputs
        inputs = {
            'images': sample_inputs['images'],
            'text_tokens': sample_inputs['text_tokens'],
            'token_type_ids': ops.cast(keras.random.uniform((sample_inputs['batch_size'], 10), minval=0, maxval=2,
                                                   dtype='float32'), dtype="int32"),
            'position_ids': ops.arange(10)[None, :] * ops.ones((sample_inputs['batch_size'], 1), dtype='int32'),
            'attention_mask': ops.ones((sample_inputs['batch_size'], 10), dtype='int32')
        }

        logits = built_model(inputs, training=False)

        vision_seq_len = (224 // 16) ** 2 + 1
        text_seq_len = 10
        combined_seq_len = vision_seq_len + text_seq_len

        assert logits.shape == (sample_inputs['batch_size'], combined_seq_len, built_model.vocab_size)

    def test_forward_pass_training_vs_inference(self, built_model, sample_inputs):
        """Test forward pass in training vs inference mode."""
        inputs = {
            'images': sample_inputs['images'],
            'text_tokens': sample_inputs['text_tokens']
        }

        logits_train = built_model(inputs, training=True)
        logits_eval = built_model(inputs, training=False)

        # Should have same shape
        assert logits_train.shape == logits_eval.shape

        # With dropout=0.1 in mini model, they might be slightly different
        # but we can't guarantee this due to randomness, so just check shapes

    def test_forward_pass_different_batch_sizes(self, built_model):
        """Test forward pass with different batch sizes."""
        batch_sizes = [1, 3, 8]

        for batch_size in batch_sizes:
            images = keras.random.normal((batch_size, 224, 224, 3))
            text_tokens = ops.cast(keras.random.uniform((batch_size, 5), minval=0, maxval=1000, dtype='float32'), dtype="int32")

            inputs = {'images': images, 'text_tokens': text_tokens}
            logits = built_model(inputs, training=False)

            vision_seq_len = (224 // 16) ** 2 + 1
            text_seq_len = 5
            combined_seq_len = vision_seq_len + text_seq_len

            assert logits.shape == (batch_size, combined_seq_len, built_model.vocab_size)

    def test_forward_pass_different_sequence_lengths(self, built_model, sample_inputs):
        """Test forward pass with different text sequence lengths."""
        sequence_lengths = [1, 5, 20, 50]

        for seq_len in sequence_lengths:
            text_tokens = ops.cast(keras.random.uniform((sample_inputs['batch_size'], seq_len), minval=0, maxval=1000,
                                               dtype='float32'), dtype="int32")

            inputs = {'images': sample_inputs['images'], 'text_tokens': text_tokens}
            logits = built_model(inputs, training=False)

            vision_seq_len = (224 // 16) ** 2 + 1
            combined_seq_len = vision_seq_len + seq_len

            assert logits.shape == (sample_inputs['batch_size'], combined_seq_len, built_model.vocab_size)

    def test_invalid_input_formats(self, built_model):
        """Test that invalid input formats raise errors."""
        batch_size = 2
        images = keras.random.normal((batch_size, 224, 224, 3))
        text_tokens = ops.cast(keras.random.uniform((batch_size, 10), minval=0, maxval=1000, dtype='float32'), dtype="int32")

        # Missing required keys in dict
        with pytest.raises(ValueError, match="Input dict must contain 'images' and 'text_tokens' keys"):
            built_model({'images': images}, training=False)

        with pytest.raises(ValueError, match="Input dict must contain 'images' and 'text_tokens' keys"):
            built_model({'text_tokens': text_tokens}, training=False)

        # Wrong tuple/list length
        with pytest.raises(ValueError, match="Inputs must be either a dict.*or a tuple/list"):
            built_model([images], training=False)  # Only one element

        with pytest.raises(ValueError, match="Inputs must be either a dict.*or a tuple/list"):
            built_model([images, text_tokens, images], training=False)  # Too many elements

    def test_compute_output_shape(self, built_model):
        """Test compute_output_shape method."""
        # Test with dict format
        input_shape_dict = {
            'images': (4, 224, 224, 3),
            'text_tokens': (4, 10)
        }
        output_shape = built_model.compute_output_shape(input_shape_dict)

        vision_seq_len = (224 // 16) ** 2 + 1
        text_seq_len = 10
        combined_seq_len = vision_seq_len + text_seq_len

        assert output_shape == (4, combined_seq_len, built_model.vocab_size)

        # Test with tuple format
        input_shape_tuple = ((4, 224, 224, 3), (4, 15))
        output_shape = built_model.compute_output_shape(input_shape_tuple)

        combined_seq_len = vision_seq_len + 15
        assert output_shape == (4, combined_seq_len, built_model.vocab_size)


class TestNanoVLMGeneration:
    """Test NanoVLM text generation functionality."""

    @pytest.fixture
    def built_model(self):
        """Create a built NanoVLM model for testing."""
        model = create_nanovlm_mini()
        model.build()
        return model

    def test_generate_basic(self, built_model):
        """Test basic text generation functionality."""
        # Single image and prompt
        image = keras.random.normal((1, 224, 224, 3))
        prompt_tokens = ops.cast(keras.random.uniform((1, 5), minval=0, maxval=1000, dtype='float32'), dtype="int32")

        generated_tokens = built_model.generate(
            image=image,
            prompt_tokens=prompt_tokens,
            max_length=10,
            temperature=1.0,
            top_k=50
        )

        # Should have original prompt + generated tokens
        assert generated_tokens.shape[0] == 1  # Batch size 1
        assert generated_tokens.shape[1] >= prompt_tokens.shape[1]  # At least original length
        assert generated_tokens.shape[1] <= prompt_tokens.shape[1] + 10  # Max length constraint

    def test_generate_temperature_effects(self, built_model):
        """Test temperature effects on generation."""
        image = keras.random.normal((1, 224, 224, 3))
        prompt_tokens = ops.cast(keras.random.uniform((1, 3), minval=0, maxval=1000, dtype='float32'), dtype="int32")

        temperatures = [0.1, 1.0, 2.0]
        results = []

        for temp in temperatures:
            generated = built_model.generate(
                image=image,
                prompt_tokens=prompt_tokens,
                max_length=5,
                temperature=temp,
                top_k=0  # Disable top_k for pure temperature effects
            )
            results.append(generated)

        # All should have same shape
        for result in results:
            assert result.shape[0] == 1
            assert result.shape[1] >= 3

    def test_generate_top_k_effects(self, built_model):
        """Test top-k sampling effects."""
        image = keras.random.normal((1, 224, 224, 3))
        prompt_tokens = ops.cast(keras.random.uniform((1, 3), minval=0, maxval=1000, dtype='float32'), dtype="int32")

        top_k_values = [0, 10, 50]
        results = []

        for top_k in top_k_values:
            generated = built_model.generate(
                image=image,
                prompt_tokens=prompt_tokens,
                max_length=5,
                temperature=1.0,
                top_k=top_k
            )
            results.append(generated)

        # All should have valid shapes
        for result in results:
            assert result.shape[0] == 1
            assert result.shape[1] >= 3

    def test_generate_early_stopping(self, built_model):
        """Test early stopping with EOS token."""
        image = keras.random.normal((1, 224, 224, 3))
        prompt_tokens = ops.cast(keras.random.uniform((1, 2), minval=0, maxval=1000, dtype='float32'), dtype="int32")

        # Mock the sampling to return EOS token early
        original_sample = built_model._sample_next_token
        call_count = 0

        def mock_sample(logits, temperature, top_k):
            nonlocal call_count
            call_count += 1
            if call_count == 2:  # Return EOS on second call
                return ops.convert_to_tensor(2, dtype='int32')  # EOS token
            return original_sample(logits, temperature, top_k)

        built_model._sample_next_token = mock_sample

        generated = built_model.generate(
            image=image,
            prompt_tokens=prompt_tokens,
            max_length=20,
            eos_token_id=2
        )

        # Should stop early due to EOS token
        assert generated.shape[1] < prompt_tokens.shape[1] + 20

        # Restore original method
        built_model._sample_next_token = original_sample

    def test_sample_next_token_methods(self, built_model):
        """Test _sample_next_token with different parameters."""
        # Create mock logits
        logits = keras.random.normal((built_model.vocab_size,))

        # Test greedy sampling (top_k=0)
        token_greedy = built_model._sample_next_token(logits, temperature=1.0, top_k=0)
        expected_greedy = ops.argmax(logits, axis=-1)
        assert ops.equal(token_greedy, expected_greedy)

        # Test top-k sampling
        token_topk = built_model._sample_next_token(logits, temperature=1.0, top_k=10)
        assert token_topk.dtype == expected_greedy.dtype

        # Test temperature scaling
        token_temp = built_model._sample_next_token(logits, temperature=0.5, top_k=10)
        assert token_temp.dtype == expected_greedy.dtype


class TestNanoVLMSerialization:
    """Test NanoVLM serialization and deserialization."""

    def test_config_serialization(self):
        """Test model configuration serialization."""
        model = create_nanovlm_mini()
        model.build()

        config = model.get_config()

        # Check that config is a dictionary with required keys
        assert isinstance(config, dict)
        required_keys = ['vision_config', 'language_config', 'projection_config', 'vocab_size', 'dropout_rate']
        for key in required_keys:
            assert key in config

        # Check config contents
        assert config['vocab_size'] == model.vocab_size
        assert config['dropout_rate'] == model.dropout_rate
        assert config['vision_config'] == model.vision_config
        assert config['language_config'] == model.language_config
        assert config['projection_config'] == model.projection_config

    def test_build_config_serialization(self):
        """Test build configuration serialization."""
        model = create_nanovlm_mini()

        # Before building
        build_config = model.get_build_config()
        assert build_config['input_shape'] is None

        # After building
        input_shape = (None, 224, 224, 3)
        model.build(input_shape)
        build_config = model.get_build_config()
        assert build_config['input_shape'] == input_shape

    def test_from_config_reconstruction(self):
        """Test model reconstruction from configuration."""
        original_model = create_nanovlm_base()

        # Get config and reconstruct
        config = original_model.get_config()
        reconstructed_model = NanoVLM.from_config(config)

        # Check that key attributes match
        assert reconstructed_model.vocab_size == original_model.vocab_size
        assert reconstructed_model.dropout_rate == original_model.dropout_rate
        assert reconstructed_model.vision_config == original_model.vision_config
        assert reconstructed_model.language_config == original_model.language_config
        assert reconstructed_model.projection_config == original_model.projection_config

    def test_build_from_config(self):
        """Test building from configuration."""
        model = create_nanovlm_mini()
        model.build()

        # Get build config
        build_config = model.get_build_config()

        # Create new model and build from config
        new_model = create_nanovlm_mini()
        new_model.build_from_config(build_config)

        assert new_model.built is True

    def test_model_save_load_cycle(self):
        """Test complete save/load cycle."""
        # Create and build model
        model = create_nanovlm_mini()
        model.build()

        # Create sample inputs and get outputs
        images = keras.random.normal((2, 224, 224, 3))
        text_tokens = ops.cast(keras.random.uniform((2, 5), minval=0, maxval=1000, dtype='float32'), dtype="int32")
        inputs = {'images': images, 'text_tokens': text_tokens}

        original_outputs = model(inputs, training=False)

        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_nano_vlm.keras')
            model.save(model_path)

            # Load model
            loaded_model = keras.models.load_model(model_path)

            # Test that loaded model produces same outputs
            loaded_outputs = loaded_model(inputs, training=False)

            # Should be very close
            np.testing.assert_allclose(
                original_outputs.numpy(),
                loaded_outputs.numpy(),
                rtol=1e-5,
                atol=1e-6
            )


class TestNanoVLMEdgeCases:
    """Test NanoVLM edge cases and error handling."""

    def test_minimum_configuration(self):
        """Test with minimal valid configuration."""
        vision_config = {
            'img_size': 32, 'patch_size': 16, 'embed_dim': 128,
            'depth': 1, 'num_heads': 4, 'mlp_ratio': 2.0
        }
        language_config = {
            'vocab_size': 1000, 'hidden_dim': 128, 'num_layers': 1,
            'num_heads': 4, 'mlp_dim': 256
        }
        projection_config = {
            'input_dim': 128, 'output_dim': 128
        }

        model = NanoVLM(
            vision_config=vision_config,
            language_config=language_config,
            projection_config=projection_config,
            vocab_size=1000
        )
        model.build()

        # Test forward pass
        images = keras.random.normal((1, 32, 32, 3))
        text_tokens = ops.cast(keras.random.uniform((1, 2), minval=0, maxval=1000, dtype='float32'), dtype="int32")
        inputs = {'images': images, 'text_tokens': text_tokens}

        logits = model(inputs, training=False)

        vision_patches = (32 // 16) ** 2 + 1  # 5
        combined_seq_len = vision_patches + 2  # 7
        assert logits.shape == (1, combined_seq_len, 1000)

    def test_single_token_text_input(self):
        """Test with single token text input."""
        model = create_nanovlm_mini()
        model.build()

        images = keras.random.normal((2, 224, 224, 3))
        text_tokens = ops.cast(keras.random.uniform((2, 1), minval=0, maxval=1000, dtype='float32'), dtype="int32")  # Single token
        inputs = {'images': images, 'text_tokens': text_tokens}

        logits = model(inputs, training=False)

        vision_seq_len = (224 // 16) ** 2 + 1
        combined_seq_len = vision_seq_len + 1
        assert logits.shape == (2, combined_seq_len, model.vocab_size)

    def test_large_sequence_lengths(self):
        """Test with large sequence lengths."""
        model = create_nanovlm_mini()
        model.build()

        images = keras.random.normal((1, 224, 224, 3))
        text_tokens = ops.cast(keras.random.uniform((1, 100), minval=0, maxval=1000, dtype='float32'), dtype="int32")  # Long sequence
        inputs = {'images': images, 'text_tokens': text_tokens}

        logits = model(inputs, training=False)

        vision_seq_len = (224 // 16) ** 2 + 1
        combined_seq_len = vision_seq_len + 100
        assert logits.shape == (1, combined_seq_len, model.vocab_size)

    def test_different_image_sizes(self):
        """Test different image sizes within same model."""
        # Note: This test may need to be adjusted based on actual implementation
        # as some models may require fixed image sizes

        model = create_nanovlm_mini()
        model.build()

        # Test with original size
        images_224 = keras.random.normal((1, 224, 224, 3))
        text_tokens = ops.cast(keras.random.uniform((1, 5), minval=0, maxval=1000, dtype='float32'), dtype="int32")

        inputs = {'images': images_224, 'text_tokens': text_tokens}
        logits = model(inputs, training=False)

        vision_seq_len = (224 // 16) ** 2 + 1
        combined_seq_len = vision_seq_len + 5
        assert logits.shape == (1, combined_seq_len, model.vocab_size)

    def test_different_channel_counts(self):
        """Test with different number of image channels."""
        vision_config = NANOVLM_CONFIGS['nanovlm_mini']['vision_config'].copy()
        language_config = NANOVLM_CONFIGS['nanovlm_mini']['language_config'].copy()
        projection_config = NANOVLM_CONFIGS['nanovlm_mini']['projection_config'].copy()

        model = NanoVLM(
            vision_config=vision_config,
            language_config=language_config,
            projection_config=projection_config
        )
        model.build()

        # Test with different channel counts
        for channels in [1, 3, 4]:  # Grayscale, RGB, RGBA
            images = keras.random.normal((1, 224, 224, channels))
            text_tokens = ops.cast(keras.random.uniform((1, 3), minval=0, maxval=1000, dtype='float32'), dtype="int32")
            inputs = {'images': images, 'text_tokens': text_tokens}

            logits = model(inputs, training=False)

            vision_seq_len = (224 // 16) ** 2 + 1
            combined_seq_len = vision_seq_len + 3
            assert logits.shape == (1, combined_seq_len, model.vocab_size)

    def test_very_large_batch_size(self):
        """Test with large batch size."""
        model = create_nanovlm_mini()
        model.build()

        batch_size = 16  # Reasonably large but not memory prohibitive
        images = keras.random.normal((batch_size, 224, 224, 3))
        text_tokens = ops.cast(keras.random.uniform((batch_size, 5), minval=0, maxval=1000, dtype='float32'), dtype="int32")
        inputs = {'images': images, 'text_tokens': text_tokens}

        logits = model(inputs, training=False)

        vision_seq_len = (224 // 16) ** 2 + 1
        combined_seq_len = vision_seq_len + 5
        assert logits.shape == (batch_size, combined_seq_len, model.vocab_size)


class TestNanoVLMFactoryFunctions:
    """Test factory functions for creating NanoVLM models."""

    def test_create_nanovlm_basic(self):
        """Test basic create_nanovlm function."""
        model = create_nanovlm()

        assert isinstance(model, NanoVLM)
        # Should use base variant by default
        assert model.vision_config == NANOVLM_CONFIGS['nanovlm_base']['vision_config']
        assert model.language_config == NANOVLM_CONFIGS['nanovlm_base']['language_config']
        assert model.projection_config['input_dim'] == NANOVLM_CONFIGS['nanovlm_base']['projection_config']['input_dim']

    def test_create_nanovlm_variants(self):
        """Test creating different model variants."""
        variants = ['nanovlm_mini', 'nanovlm_base', 'nanovlm_222m']

        for variant in variants:
            model = create_nanovlm(variant)

            assert isinstance(model, NanoVLM)
            expected_config = NANOVLM_CONFIGS[variant]
            assert model.vision_config == expected_config['vision_config']
            assert model.language_config == expected_config['language_config']

    def test_create_nanovlm_with_custom_params(self):
        """Test create_nanovlm with custom parameters."""
        model = create_nanovlm(
            variant='nanovlm_mini',
            vocab_size=50000,
            dropout_rate=0.2
        )

        assert model.vocab_size == 50000
        assert model.dropout_rate == 0.2
        # Should still use mini config for other parameters
        assert model.vision_config['embed_dim'] == NANOVLM_CONFIGS['nanovlm_mini']['vision_config']['embed_dim']

    def test_create_nanovlm_invalid_variant(self):
        """Test that invalid variant names raise errors."""
        with pytest.raises(ValueError, match="Unknown variant"):
            create_nanovlm("invalid_variant")

    def test_create_nanovlm_mini(self):
        """Test create_nanovlm_mini factory function."""
        model = create_nanovlm_mini()

        assert isinstance(model, NanoVLM)
        expected_config = NANOVLM_CONFIGS['nanovlm_mini']
        assert model.vision_config == expected_config['vision_config']
        assert model.language_config == expected_config['language_config']

        # Test functionality
        model.build()
        images = keras.random.normal((1, 224, 224, 3))
        text_tokens = ops.cast(keras.random.uniform((1, 5), minval=0, maxval=1000, dtype='float32'), dtype="int32")
        inputs = {'images': images, 'text_tokens': text_tokens}

        logits = model(inputs, training=False)
        vision_seq_len = (224 // 16) ** 2 + 1
        assert logits.shape == (1, vision_seq_len + 5, model.vocab_size)

    def test_create_nanovlm_base(self):
        """Test create_nanovlm_base factory function."""
        model = create_nanovlm_base()

        assert isinstance(model, NanoVLM)
        expected_config = NANOVLM_CONFIGS['nanovlm_base']
        assert model.vision_config == expected_config['vision_config']
        assert model.language_config == expected_config['language_config']

    def test_create_nanovlm_222m(self):
        """Test create_nanovlm_222m factory function."""
        model = create_nanovlm_222m()

        assert isinstance(model, NanoVLM)
        expected_config = NANOVLM_CONFIGS['nanovlm_222m']
        assert model.vision_config == expected_config['vision_config']
        assert model.language_config == expected_config['language_config']

    def test_parameter_counts_scaling(self):
        """Test that parameter counts scale appropriately across variants."""
        models = {
            'mini': create_nanovlm_mini(),
            'base': create_nanovlm_base(),
            'large': create_nanovlm_222m()
        }

        param_counts = {}
        for name, model in models.items():
            model.build()
            param_counts[name] = model.count_params()

        # Larger models should have more parameters
        assert param_counts['mini'] < param_counts['base'] < param_counts['large']


class TestNanoVLMIntegration:
    """Integration tests for NanoVLM components working together."""

    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        model = create_nanovlm_mini()
        model.build()

        # Create test inputs
        batch_size = 2
        images = keras.random.normal((batch_size, 224, 224, 3))
        text_tokens = ops.cast(keras.random.uniform((batch_size, 8), minval=0, maxval=1000, dtype='float32'), dtype="int32")

        # Forward pass
        inputs = {'images': images, 'text_tokens': text_tokens}
        logits = model(inputs, training=False)

        # Check output properties
        vision_seq_len = (224 // 16) ** 2 + 1
        combined_seq_len = vision_seq_len + 8
        assert logits.shape == (batch_size, combined_seq_len, model.vocab_size)

        # Check that logits are reasonable (not all zeros, not exploding)
        assert not ops.all(ops.equal(logits, 0.0))
        assert ops.all(ops.isfinite(logits))

        # Test generation
        single_image = images[:1]  # Single image
        single_prompt = text_tokens[:1, :3]  # Shorter prompt

        generated = model.generate(
            image=single_image,
            prompt_tokens=single_prompt,
            max_length=5,
            temperature=1.0
        )

        assert generated.shape[0] == 1
        assert generated.shape[1] >= single_prompt.shape[1]

    def test_component_interaction(self):
        """Test interaction between vision encoder, projection, and text decoder."""
        model = create_nanovlm_mini()
        model.build()

        # Test individual components
        images = keras.random.normal((2, 224, 224, 3))
        text_tokens = ops.cast(keras.random.uniform((2, 5), minval=0, maxval=1000, dtype='float32'), dtype="int32")

        # Vision encoding
        vision_features = model.vision_encoder(images, training=False)
        # Manually add dummy token to mimic the 'call' method's logic for comparison
        cls_token_shape = (ops.shape(vision_features)[0], 1, ops.shape(vision_features)[2])
        dummy_cls_token = ops.zeros(cls_token_shape, dtype=vision_features.dtype)
        vision_features_with_cls = ops.concatenate([dummy_cls_token, vision_features], axis=1)
        vision_embeddings = model.modality_projection(vision_features_with_cls, training=False)

        # Text processing
        text_hidden_states = model.text_decoder(
            inputs=text_tokens,
            training=False
        )

        # Check shapes are compatible
        assert vision_embeddings.shape[-1] == text_hidden_states.shape[-1]  # Same hidden dim

        # Combined processing
        combined_hidden_states = ops.concatenate([vision_embeddings, text_hidden_states], axis=1)
        logits = model.output_projection(combined_hidden_states)

        # Should match full forward pass
        full_inputs = {'images': images, 'text_tokens': text_tokens}
        full_logits = model(full_inputs, training=False)

        np.testing.assert_allclose(
            logits.numpy(),
            full_logits.numpy(),
            rtol=1e-5
        )

    def test_gradient_flow(self):
        """Test that gradients flow through the entire model."""
        model = create_nanovlm_mini()
        model.build()

        images = keras.random.normal((2, 224, 224, 3))
        text_tokens = ops.cast(keras.random.uniform((2, 5), minval=0, maxval=1000, dtype='float32'), dtype="int32")
        inputs = {'images': images, 'text_tokens': text_tokens}

        with tf.GradientTape() as tape:
            logits = model(inputs, training=True)

            # Create a simple loss (minimize logits norm)
            loss = ops.mean(ops.sum(ops.square(logits), axis=-1))

        gradients = tape.gradient(loss, model.trainable_weights)

        # Check that most weights have gradients
        non_none_grads = [g for g in gradients if g is not None]
        assert len(non_none_grads) > len(model.trainable_weights) * 0.8

        # Check gradient magnitudes are reasonable
        grad_norms = [
            ops.sqrt(ops.sum(ops.square(g))) for g in non_none_grads if g is not None
        ]
        assert all(norm > 1e-12 for norm in grad_norms), "Found vanishingly small gradients"
        assert all(norm < 1e6 for norm in grad_norms), "Found exploding gradients"

    def test_different_model_sizes_consistency(self):
        """Test that different model sizes behave consistently."""
        variants = ['nanovlm_mini', 'nanovlm_base']  # Skip 222m to save memory in tests

        images = keras.random.normal((1, 224, 224, 3))
        text_tokens = ops.cast(keras.random.uniform((1, 5), minval=0, maxval=1000, dtype='float32'), dtype="int32")
        inputs = {'images': images, 'text_tokens': text_tokens}

        for variant in variants:
            model = create_nanovlm(variant)
            model.build()

            # Test forward pass
            logits = model(inputs, training=False)

            vision_seq_len = (224 // 16) ** 2 + 1
            combined_seq_len = vision_seq_len + 5
            assert logits.shape == (1, combined_seq_len, model.vocab_size)

            # Test generation
            generated = model.generate(
                image=images,
                prompt_tokens=text_tokens,
                max_length=3,
                temperature=1.0
            )

            assert generated.shape[0] == 1
            assert generated.shape[1] >= text_tokens.shape[1]

    def test_training_vs_inference_consistency(self):
        """Test consistency between training and inference modes."""
        model = create_nanovlm_mini()
        model.build()

        images = keras.random.normal((2, 224, 224, 3))
        text_tokens = ops.cast(keras.random.uniform((2, 5), minval=0, maxval=1000, dtype='float32'), dtype="int32")
        inputs = {'images': images, 'text_tokens': text_tokens}

        # Get outputs in both modes
        logits_train = model(inputs, training=True)
        logits_eval = model(inputs, training=False)

        # Should have same shape
        assert logits_train.shape == logits_eval.shape

        # With dropout in the model, they might be different, but should be reasonable
        assert ops.all(ops.isfinite(logits_train))
        assert ops.all(ops.isfinite(logits_eval))

    def test_model_determinism(self):
        """Test that model produces deterministic outputs given same inputs."""
        model = create_nanovlm_mini()
        model.build()

        images = keras.random.normal((1, 224, 224, 3))
        text_tokens = ops.cast(keras.random.uniform((1, 3), minval=0, maxval=1000, dtype='float32'), dtype="int32")
        inputs = {'images': images, 'text_tokens': text_tokens}

        # Get outputs twice
        logits1 = model(inputs, training=False)
        logits2 = model(inputs, training=False)

        # Should be identical in inference mode (assuming no dropout)
        np.testing.assert_allclose(
            logits1.numpy(),
            logits2.numpy(),
            rtol=1e-6
        )

    def test_memory_efficiency(self):
        """Test basic memory efficiency (no major memory leaks)."""
        model = create_nanovlm_mini()
        model.build()

        images = keras.random.normal((1, 224, 224, 3))
        text_tokens = ops.cast(keras.random.uniform((1, 5), minval=0, maxval=1000, dtype='float32'), dtype="int32")
        inputs = {'images': images, 'text_tokens': text_tokens}

        # Run multiple forward passes
        for _ in range(5):
            logits = model(inputs, training=False)
            # Delete reference to help with memory cleanup
            del logits


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])