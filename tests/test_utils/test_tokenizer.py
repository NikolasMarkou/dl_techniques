"""Comprehensive pytest test suite for TiktokenPreprocessor.

This module provides extensive testing coverage for the TiktokenPreprocessor
class, including initialization, preprocessing logic, edge cases, error
handling, serialization, and integration scenarios.

Test categories:
    - Initialization and configuration validation
    - Single text preprocessing
    - Batch text preprocessing
    - Special token handling
    - Padding and truncation logic
    - Attention mask generation
    - Token type ID generation
    - Tensor format conversions (NumPy, TensorFlow)
    - Decoding functionality
    - Edge cases and error conditions
    - API convenience methods
"""

from typing import List

import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.utils.tokenizer import TiktokenPreprocessor


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def default_preprocessor() -> TiktokenPreprocessor:
    """Create a preprocessor with default BERT-like configuration.

    :return: TiktokenPreprocessor instance with default settings.
    :rtype: TiktokenPreprocessor
    """
    return TiktokenPreprocessor(
        encoding_name="cl100k_base",
        max_length=32,
        cls_token_id=101,
        sep_token_id=102,
        pad_token_id=0,
    )


@pytest.fixture
def custom_preprocessor() -> TiktokenPreprocessor:
    """Create a preprocessor with custom configuration.

    :return: TiktokenPreprocessor instance with custom settings.
    :rtype: TiktokenPreprocessor
    """
    return TiktokenPreprocessor(
        encoding_name="cl100k_base",
        max_length=16,
        cls_token_id=1,
        sep_token_id=2,
        pad_token_id=0,
        truncation=False,
        padding='do_not_pad',
    )


@pytest.fixture
def sample_texts() -> List[str]:
    """Provide sample texts for testing.

    :return: List of sample text strings.
    :rtype: List[str]
    """
    return [
        "Hello world!",
        "This is a longer sentence with more tokens.",
        "Short",
        "",  # Empty string edge case
        "A" * 1000,  # Very long string
    ]


# ---------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------


class TestInitialization:
    """Test TiktokenPreprocessor initialization and validation."""

    def test_default_initialization(self) -> None:
        """Test initialization with default parameters."""
        preprocessor = TiktokenPreprocessor()

        assert preprocessor.max_length == 256
        assert preprocessor.cls_token_id == 101
        assert preprocessor.sep_token_id == 102
        assert preprocessor.pad_token_id == 0
        assert preprocessor.truncation is True
        assert preprocessor.padding == 'max_length'
        assert preprocessor.tokenizer is not None

    def test_custom_initialization(self, custom_preprocessor) -> None:
        """Test initialization with custom parameters."""
        assert custom_preprocessor.max_length == 16
        assert custom_preprocessor.cls_token_id == 1
        assert custom_preprocessor.sep_token_id == 2
        assert custom_preprocessor.pad_token_id == 0
        assert custom_preprocessor.truncation is False
        assert custom_preprocessor.padding == 'do_not_pad'

    def test_invalid_max_length(self) -> None:
        """Test that invalid max_length raises ValueError."""
        with pytest.raises(ValueError, match="max_length must be positive"):
            TiktokenPreprocessor(max_length=0)

        with pytest.raises(ValueError, match="max_length must be positive"):
            TiktokenPreprocessor(max_length=-10)

    def test_invalid_encoding_name(self) -> None:
        """Test that invalid encoding_name raises ValueError."""
        with pytest.raises(ValueError, match="Invalid encoding_name"):
            TiktokenPreprocessor(encoding_name="invalid_encoding")

    def test_vocab_size_property(self, default_preprocessor) -> None:
        """Test vocab_size property returns correct value."""
        vocab_size = default_preprocessor.vocab_size
        assert isinstance(vocab_size, int)
        assert vocab_size > 0
        # cl100k_base has vocab size of 100277
        assert vocab_size == 100277

    def test_repr(self, default_preprocessor) -> None:
        """Test string representation."""
        repr_str = repr(default_preprocessor)
        assert "TiktokenPreprocessor" in repr_str
        assert "encoding=cl100k_base" in repr_str
        assert "max_length=32" in repr_str
        assert "vocab_size=100277" in repr_str


# ---------------------------------------------------------------------
# Single Text Preprocessing Tests
# ---------------------------------------------------------------------


class TestSingleTextPreprocessing:
    """Test preprocessing of single text inputs."""

    def test_basic_preprocessing(self, default_preprocessor) -> None:
        """Test basic single text preprocessing."""
        text = "Hello world!"
        result = default_preprocessor(text, return_tensors='np')

        assert isinstance(result, dict)
        assert 'input_ids' in result
        assert 'attention_mask' in result
        assert 'token_type_ids' in result

        # Check shapes
        assert result['input_ids'].shape == (1, 32)
        assert result['attention_mask'].shape == (1, 32)
        assert result['token_type_ids'].shape == (1, 32)

        # Check dtypes
        assert result['input_ids'].dtype == np.int32
        assert result['attention_mask'].dtype == np.int32
        assert result['token_type_ids'].dtype == np.int32

    def test_special_tokens_insertion(self, default_preprocessor) -> None:
        """Test that [CLS] and [SEP] tokens are correctly inserted."""
        text = "Test"
        result = default_preprocessor(text, return_tensors='np')
        input_ids = result['input_ids'][0]

        # First token should be [CLS]
        assert input_ids[0] == 101

        # Find the [SEP] token (should be after actual tokens)
        sep_position = np.where(input_ids == 102)[0]
        assert len(sep_position) > 0

    def test_padding_applied(self, default_preprocessor) -> None:
        """Test that padding is applied correctly."""
        text = "Short"
        result = default_preprocessor(text, return_tensors='np')
        input_ids = result['input_ids'][0]
        attention_mask = result['attention_mask'][0]

        # Count padding tokens
        num_padding = np.sum(input_ids == 0)
        assert num_padding > 0

        # Attention mask should be 0 for padding
        assert np.sum(attention_mask[input_ids == 0]) == 0

        # Attention mask should be 1 for non-padding
        assert np.all(attention_mask[input_ids != 0] == 1)

    def test_token_type_ids_all_zeros(self, default_preprocessor) -> None:
        """Test that token type IDs are all zeros for single sentences."""
        text = "Single sentence task"
        result = default_preprocessor(text, return_tensors='np')
        token_type_ids = result['token_type_ids'][0]

        # All token type IDs should be 0 for single sentence
        assert np.all(token_type_ids == 0)

    def test_empty_string(self, default_preprocessor) -> None:
        """Test preprocessing of empty string."""
        text = ""
        result = default_preprocessor(text, return_tensors='np')
        input_ids = result['input_ids'][0]

        # Should still have [CLS] and [SEP]
        assert input_ids[0] == 101
        assert 102 in input_ids


# ---------------------------------------------------------------------
# Batch Preprocessing Tests
# ---------------------------------------------------------------------


class TestBatchPreprocessing:
    """Test preprocessing of batch text inputs."""

    def test_batch_preprocessing(
        self,
        default_preprocessor,
        sample_texts
    ) -> None:
        """Test batch preprocessing of multiple texts."""
        texts = sample_texts[:3]  # Use first 3 texts
        result = default_preprocessor(texts, return_tensors='np')

        # Check batch dimension
        assert result['input_ids'].shape[0] == 3
        assert result['attention_mask'].shape[0] == 3
        assert result['token_type_ids'].shape[0] == 3

        # Check sequence length dimension
        assert result['input_ids'].shape[1] == 32
        assert result['attention_mask'].shape[1] == 32
        assert result['token_type_ids'].shape[1] == 32

    def test_batch_encode_method(
        self,
        default_preprocessor,
        sample_texts
    ) -> None:
        """Test batch_encode convenience method."""
        texts = sample_texts[:2]
        result = default_preprocessor.batch_encode(
            texts,
            return_tensors='np'
        )

        assert result['input_ids'].shape[0] == 2
        assert isinstance(result['input_ids'], np.ndarray)

    def test_batch_encode_type_check(self, default_preprocessor) -> None:
        """Test that batch_encode validates input type."""
        with pytest.raises(TypeError, match="expects a list"):
            default_preprocessor.batch_encode(
                "single string",
                return_tensors='np'
            )

    def test_variable_length_batch(
        self,
        default_preprocessor
    ) -> None:
        """Test batch with varying text lengths."""
        texts = ["Short", "This is a much longer sentence"]
        result = default_preprocessor(texts, return_tensors='np')

        # All sequences should be padded to same length
        assert result['input_ids'].shape == (2, 32)

        # Different attention mask patterns
        mask1 = result['attention_mask'][0]
        mask2 = result['attention_mask'][1]
        assert np.sum(mask1) != np.sum(mask2)


# ---------------------------------------------------------------------
# Truncation Tests
# ---------------------------------------------------------------------


class TestTruncation:
    """Test truncation behavior."""

    def test_truncation_enabled(self) -> None:
        """Test that truncation works when enabled."""
        preprocessor = TiktokenPreprocessor(
            max_length=10,
            truncation=True
        )
        long_text = "This is a very long sentence " * 20
        result = preprocessor(long_text, return_tensors='np')

        # Should be truncated to max_length
        assert result['input_ids'].shape[1] == 10

    def test_truncation_disabled_raises_error(self) -> None:
        """Test that truncation disabled raises ValueError for long text."""
        preprocessor = TiktokenPreprocessor(
            max_length=10,
            truncation=False
        )
        long_text = "This is a very long sentence " * 20

        with pytest.raises(ValueError, match="exceeds max_length"):
            preprocessor(long_text, return_tensors='np')

    def test_truncation_preserves_special_tokens(self) -> None:
        """Test that [CLS] and [SEP] are always present after truncation."""
        preprocessor = TiktokenPreprocessor(
            max_length=10,
            truncation=True
        )
        long_text = "word " * 100
        result = preprocessor(long_text, return_tensors='np')
        input_ids = result['input_ids'][0]

        # [CLS] should be first
        assert input_ids[0] == 101

        # [SEP] should be present
        assert 102 in input_ids


# ---------------------------------------------------------------------
# Padding Tests
# ---------------------------------------------------------------------


class TestPadding:
    """Test padding behavior."""

    def test_max_length_padding(self) -> None:
        """Test max_length padding strategy."""
        preprocessor = TiktokenPreprocessor(
            max_length=20,
            padding='max_length'
        )
        text = "Short"
        result = preprocessor(text, return_tensors='np')

        # Should be padded to max_length
        assert result['input_ids'].shape[1] == 20

        # Check padding tokens present
        input_ids = result['input_ids'][0]
        assert np.sum(input_ids == 0) > 0

    def test_do_not_pad(self) -> None:
        """Test do_not_pad strategy."""
        preprocessor = TiktokenPreprocessor(
            max_length=50,
            padding='do_not_pad'
        )
        text = "Short text"
        result = preprocessor(text, return_tensors='np')

        # Should not be padded to max_length
        input_ids = result['input_ids'][0]
        actual_length = len(input_ids)
        assert actual_length < 50


# ---------------------------------------------------------------------
# Tensor Format Tests
# ---------------------------------------------------------------------


class TestTensorFormats:
    """Test different tensor output formats."""

    def test_numpy_output(self, default_preprocessor) -> None:
        """Test NumPy array output format."""
        text = "Test text"
        result = default_preprocessor(text, return_tensors='np')

        assert isinstance(result['input_ids'], np.ndarray)
        assert isinstance(result['attention_mask'], np.ndarray)
        assert isinstance(result['token_type_ids'], np.ndarray)

    def test_tensorflow_output(self, default_preprocessor) -> None:
        """Test TensorFlow tensor output format."""
        text = "Test text"
        result = default_preprocessor(text, return_tensors='tf')

        assert isinstance(result['input_ids'], tf.Tensor)
        assert isinstance(result['attention_mask'], tf.Tensor)
        assert isinstance(result['token_type_ids'], tf.Tensor)

        # Check TensorFlow dtype
        assert result['input_ids'].dtype == tf.int32
        assert result['attention_mask'].dtype == tf.int32
        assert result['token_type_ids'].dtype == tf.int32

    def test_none_defaults_to_numpy(self, default_preprocessor) -> None:
        """Test that return_tensors=None defaults to NumPy."""
        text = "Test text"
        result = default_preprocessor(text, return_tensors=None)

        assert isinstance(result['input_ids'], np.ndarray)

    def test_invalid_return_tensors(self, default_preprocessor) -> None:
        """Test that invalid return_tensors raises ValueError."""
        text = "Test text"
        with pytest.raises(ValueError, match="return_tensors must be"):
            default_preprocessor(text, return_tensors='invalid')


# ---------------------------------------------------------------------
# Decoding Tests
# ---------------------------------------------------------------------


class TestDecoding:
    """Test decoding functionality."""

    def test_decode_basic(self, default_preprocessor) -> None:
        """Test basic decoding of token IDs."""
        text = "Hello world!"
        encoded = default_preprocessor(text, return_tensors='np')
        token_ids = encoded['input_ids'][0]

        decoded = default_preprocessor.decode(
            token_ids,
            skip_special_tokens=True
        )

        # Decoded text should match original (approximately)
        assert "Hello" in decoded or "hello" in decoded
        assert "world" in decoded

    def test_decode_with_special_tokens(self, default_preprocessor) -> None:
        """Test decoding with special tokens included."""
        text = "Test"
        encoded = default_preprocessor(text, return_tensors='np')
        token_ids = encoded['input_ids'][0]

        # Get non-padding tokens
        non_padding = token_ids[token_ids != 0]

        decoded = default_preprocessor.decode(
            non_padding,
            skip_special_tokens=False
        )

        # Should contain some text
        assert len(decoded) > 0

    def test_decode_numpy_array(self, default_preprocessor) -> None:
        """Test decoding from NumPy array."""
        token_ids = np.array([101, 1234, 5678, 102, 0, 0], dtype=np.int32)
        decoded = default_preprocessor.decode(
            token_ids,
            skip_special_tokens=True
        )

        assert isinstance(decoded, str)

    def test_decode_tensorflow_tensor(self, default_preprocessor) -> None:
        """Test decoding from TensorFlow tensor."""
        token_ids = tf.constant([101, 1234, 5678, 102, 0, 0], dtype=tf.int32)
        decoded = default_preprocessor.decode(
            token_ids,
            skip_special_tokens=True
        )

        assert isinstance(decoded, str)

    def test_decode_list(self, default_preprocessor) -> None:
        """Test decoding from Python list."""
        token_ids = [101, 1234, 5678, 102, 0, 0]
        decoded = default_preprocessor.decode(
            token_ids,
            skip_special_tokens=True
        )

        assert isinstance(decoded, str)


# ---------------------------------------------------------------------
# API Convenience Methods Tests
# ---------------------------------------------------------------------


class TestAPIConvenienceMethods:
    """Test convenience methods for the API."""

    def test_encode_method(self, default_preprocessor) -> None:
        """Test encode convenience method."""
        text = "Single text"
        result = default_preprocessor.encode(text, return_tensors='np')

        assert isinstance(result, dict)
        assert result['input_ids'].shape[0] == 1

    def test_encode_type_check(self, default_preprocessor) -> None:
        """Test that encode validates input type."""
        with pytest.raises(TypeError, match="encode expects a string"):
            default_preprocessor.encode(["list"], return_tensors='np')

    def test_call_with_string(self, default_preprocessor) -> None:
        """Test __call__ with string input."""
        text = "Test"
        result = default_preprocessor(text, return_tensors='np')

        assert result['input_ids'].shape[0] == 1

    def test_call_with_list(self, default_preprocessor) -> None:
        """Test __call__ with list input."""
        texts = ["First", "Second"]
        result = default_preprocessor(texts, return_tensors='np')

        assert result['input_ids'].shape[0] == 2


# ---------------------------------------------------------------------
# Edge Cases Tests
# ---------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_character(self, default_preprocessor) -> None:
        """Test processing single character."""
        text = "A"
        result = default_preprocessor(text, return_tensors='np')

        assert result['input_ids'].shape[1] == 32
        # Should have [CLS], token(s), [SEP], padding
        input_ids = result['input_ids'][0]
        assert input_ids[0] == 101  # [CLS]
        assert 102 in input_ids  # [SEP]

    def test_unicode_characters(self, default_preprocessor) -> None:
        """Test processing unicode characters."""
        text = "Hello 世界 🌍"
        result = default_preprocessor(text, return_tensors='np')

        assert result['input_ids'].shape[1] == 32
        # Should process without errors
        assert np.all(result['attention_mask'][0][:5] == 1)

    def test_special_characters(self, default_preprocessor) -> None:
        """Test processing special characters."""
        text = "!@#$%^&*()[]{}|\\:;\"'<>,.?/"
        result = default_preprocessor(text, return_tensors='np')

        assert result['input_ids'].shape[1] == 32

    def test_whitespace_only(self, default_preprocessor) -> None:
        """Test processing whitespace-only string."""
        text = "   \t\n   "
        result = default_preprocessor(text, return_tensors='np')

        # Should still have [CLS] and [SEP]
        input_ids = result['input_ids'][0]
        assert input_ids[0] == 101
        assert 102 in input_ids

    def test_very_long_text(self) -> None:
        """Test processing very long text."""
        preprocessor = TiktokenPreprocessor(
            max_length=256,
            truncation=True
        )
        text = "word " * 10000
        result = preprocessor(text, return_tensors='np')

        # Should be truncated
        assert result['input_ids'].shape[1] == 256

    def test_repeated_special_tokens(self, default_preprocessor) -> None:
        """Test that special tokens appear only once."""
        text = "Normal text"
        result = default_preprocessor(text, return_tensors='np')
        input_ids = result['input_ids'][0]

        # Count [CLS] tokens
        cls_count = np.sum(input_ids == 101)
        assert cls_count == 1

        # [SEP] should appear once (at the end of actual tokens)
        sep_positions = np.where(input_ids == 102)[0]
        # Only one [SEP] should be in the non-padding region
        non_padding_sep = sep_positions[sep_positions < np.where(
            input_ids == 0
        )[0][0] if 0 in input_ids else len(input_ids)]
        assert len(non_padding_sep) == 1


# ---------------------------------------------------------------------
# Error Handling Tests
# ---------------------------------------------------------------------


class TestErrorHandling:
    """Test error handling and validation."""

    def test_invalid_input_type(self, default_preprocessor) -> None:
        """Test that invalid input types raise TypeError."""
        with pytest.raises(TypeError, match="must be a string or a list"):
            default_preprocessor(123, return_tensors='np')

        with pytest.raises(TypeError, match="must be a string or a list"):
            default_preprocessor(None, return_tensors='np')

        with pytest.raises(TypeError, match="must be a string or a list"):
            default_preprocessor({'text': 'value'}, return_tensors='np')

    def test_mixed_type_batch(self, default_preprocessor) -> None:
        """Test that batch with mixed types is handled."""
        # This should work - list of strings is valid
        texts = ["text1", "text2"]
        result = default_preprocessor(texts, return_tensors='np')
        assert result['input_ids'].shape[0] == 2


# ---------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------


class TestIntegration:
    """Test integration scenarios and workflows."""

    def test_encode_decode_roundtrip(self, default_preprocessor) -> None:
        """Test encode-decode round trip."""
        original_text = "This is a test sentence."
        encoded = default_preprocessor(original_text, return_tensors='np')
        decoded = default_preprocessor.decode(
            encoded['input_ids'][0],
            skip_special_tokens=True
        )

        # Decoded should contain the same content (case-insensitive)
        original_words = original_text.lower().split()
        decoded_lower = decoded.lower()

        for word in original_words:
            assert word in decoded_lower

    def test_batch_consistency(self, default_preprocessor) -> None:
        """Test that batch processing is consistent with single processing."""
        texts = ["First text", "Second text"]

        # Process as batch
        batch_result = default_preprocessor(texts, return_tensors='np')

        # Process individually
        single_results = [
            default_preprocessor(text, return_tensors='np')
            for text in texts
        ]

        # Compare results
        for i in range(len(texts)):
            np.testing.assert_array_equal(
                batch_result['input_ids'][i],
                single_results[i]['input_ids'][0]
            )
            np.testing.assert_array_equal(
                batch_result['attention_mask'][i],
                single_results[i]['attention_mask'][0]
            )

    def test_different_configs_different_outputs(self) -> None:
        """Test that different configurations produce different outputs."""
        text = "Test text"

        prep1 = TiktokenPreprocessor(
            max_length=16,
            cls_token_id=101,
            sep_token_id=102
        )
        prep2 = TiktokenPreprocessor(
            max_length=32,
            cls_token_id=1,
            sep_token_id=2
        )

        result1 = prep1(text, return_tensors='np')
        result2 = prep2(text, return_tensors='np')

        # Shapes should differ
        assert result1['input_ids'].shape[1] != result2['input_ids'].shape[1]

        # Special tokens should differ
        assert result1['input_ids'][0, 0] != result2['input_ids'][0, 0]


# ---------------------------------------------------------------------
# Parametrized Tests
# ---------------------------------------------------------------------


class TestParametrized:
    """Parametrized tests for various configurations."""

    @pytest.mark.parametrize("max_length", [8, 16, 32, 64, 128])
    def test_various_max_lengths(self, max_length: int) -> None:
        """Test preprocessing with various max_length values."""
        preprocessor = TiktokenPreprocessor(max_length=max_length)
        text = "Test text"
        result = preprocessor(text, return_tensors='np')

        assert result['input_ids'].shape[1] == max_length

    @pytest.mark.parametrize("encoding_name", ["cl100k_base", "p50k_base"])
    def test_various_encodings(self, encoding_name: str) -> None:
        """Test preprocessing with various encodings."""
        preprocessor = TiktokenPreprocessor(encoding_name=encoding_name)
        text = "Test text"
        result = preprocessor(text, return_tensors='np')

        assert result['input_ids'].shape[1] == 256

    @pytest.mark.parametrize("return_tensors", ['np', 'tf', None])
    def test_all_return_formats(
        self,
        default_preprocessor,
        return_tensors
    ) -> None:
        """Test all supported return tensor formats."""
        text = "Test text"
        result = default_preprocessor(text, return_tensors=return_tensors)

        assert 'input_ids' in result
        if return_tensors == 'tf':
            assert isinstance(result['input_ids'], tf.Tensor)
        else:
            assert isinstance(result['input_ids'], np.ndarray)


# ---------------------------------------------------------------------
# Performance Tests
# ---------------------------------------------------------------------


class TestPerformance:
    """Performance and efficiency tests."""

    def test_large_batch_processing(self, default_preprocessor) -> None:
        """Test processing large batches efficiently."""
        texts = ["Sample text number {}".format(i) for i in range(100)]
        result = default_preprocessor(texts, return_tensors='np')

        assert result['input_ids'].shape[0] == 100
        assert result['input_ids'].shape[1] == 32

    def test_memory_efficiency(self, default_preprocessor) -> None:
        """Test that processing doesn't create unnecessary copies."""
        text = "Test"
        result1 = default_preprocessor(text, return_tensors='np')
        result2 = default_preprocessor(text, return_tensors='np')

        # Results should be equal but not the same object
        np.testing.assert_array_equal(
            result1['input_ids'],
            result2['input_ids']
        )
        assert result1['input_ids'] is not result2['input_ids']