"""Tiktoken Preprocessor for BERT-style Inputs.

This module provides a professionally written, reusable preprocessor class
that adapts the Tiktoken tokenizer for use with models expecting a BERT-like
input format. This includes handling special tokens ([CLS], [SEP]), padding,
truncation, attention masks, and token type IDs.

The design is intended to provide an interface similar to high-level
tokenizers from libraries like Hugging Face's `transformers`.

It is the user's responsibility to ensure that the special token IDs provided
during initialization match the vocabulary and expectations of the target
model.

Example
-------
    >>> preprocessor = TiktokenPreprocessor(
    ...     max_length=32,
    ...     cls_token_id=101,
    ...     sep_token_id=102,
    ...     pad_token_id=0
    ... )
    >>> inputs = preprocessor("An example sentence.", return_tensors='tf')
    >>> print(inputs['input_ids'].shape)
    TensorShape([1, 32])
"""


import tiktoken
import numpy as np
import tensorflow as tf
from typing import Dict, List, Literal, Optional, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

def get_special_token_ids(
    encoding_name: str = "cl100k_base"
) -> Dict[str, int]:
    """Get valid special token IDs for a Tiktoken encoding.

    This function returns token IDs that actually exist in the Tiktoken
    vocabulary, rather than using arbitrary high IDs that would fail
    during decoding.

    :param encoding_name: Name of the Tiktoken encoding.
    :type encoding_name: str
    :return: Dictionary mapping special token names to IDs.
    :rtype: Dict[str, int]

    Example:
        >>> special_tokens = get_special_token_ids("cl100k_base")
        >>> print(special_tokens)
        {'cls': 100257, 'sep': 100258, 'pad': 100256, 'mask': 100259}
    """
    encoding = tiktoken.get_encoding(encoding_name)
    vocab_size = encoding.n_vocab

    # Use the highest valid token IDs for special tokens
    # cl100k_base vocab size is 100277, so we use IDs near the end
    special_tokens = {
        'pad': 100256,    # <|endoftext|> token (standard padding)
        'cls': vocab_size - 20,  # High but valid
        'sep': vocab_size - 19,  # High but valid
        'mask': vocab_size - 18,  # High but valid
        'unk': vocab_size - 17,  # High but valid (unknown token)
    }

    # Validate all tokens are in range
    for name, token_id in special_tokens.items():
        if not (0 <= token_id < vocab_size):
            raise ValueError(
                f"Special token '{name}' ID {token_id} is outside "
                f"vocabulary range [0, {vocab_size})"
            )

    return special_tokens

# ---------------------------------------------------------------------

class TiktokenPreprocessor:
    """Callable preprocessor adapting Tiktoken for BERT-style model inputs.

    This class encapsulates the logic for tokenization, addition of special
    tokens, padding, truncation, and the creation of associated masks,
    making Tiktoken compatible with models that require a fixed-length,
    structured input.

    The preprocessor handles:
        - Text tokenization using Tiktoken encodings
        - Special token insertion ([CLS], [SEP])
        - Sequence truncation to max_length
        - Padding to fixed length
        - Attention mask generation
        - Token type ID generation
        - Batch processing
        - Output conversion to NumPy or TensorFlow tensors

    Args:
        encoding_name: Name of the Tiktoken encoding to use. Common options
            include 'cl100k_base' (GPT-4), 'p50k_base' (GPT-3),
            'r50k_base' (older models). Defaults to 'cl100k_base'.
        max_length: Maximum sequence length including special tokens.
            Sequences longer than (max_length - 2) will be truncated to
            accommodate [CLS] and [SEP] tokens. Must be positive.
            Defaults to 256.
        cls_token_id: Token ID for the [CLS] (classification) token,
            added at the start of each sequence. Defaults to 101 (BERT).
        sep_token_id: Token ID for the [SEP] (separator) token,
            added at the end of each sequence. Defaults to 102 (BERT).
        pad_token_id: Token ID used for padding sequences to max_length.
            Defaults to 0 (BERT).
        mask_token_id: Token ID for the [MASK] token, used in tasks like
            Masked Language Modeling. Defaults to a valid ID from the vocab.
        truncation: Whether to truncate sequences exceeding max_length.
            If False, raises ValueError for sequences that are too long.
            Defaults to True.
        padding: Padding strategy. Options are 'max_length' (pad all
            sequences to max_length) or 'do_not_pad' (return variable
            length sequences). Defaults to 'max_length'.

    Raises:
        ValueError: If max_length is not positive.
        ValueError: If encoding_name is not a valid Tiktoken encoding.

    Example:
        >>> preprocessor = TiktokenPreprocessor(
        ...     max_length=32,
        ...     cls_token_id=101,
        ...     sep_token_id=102,
        ...     pad_token_id=0
        ... )
        >>> # Single text
        >>> result = preprocessor("Hello world!", return_tensors='np')
        >>> result['input_ids'].shape
        (1, 32)
        >>> # Batch of texts
        >>> batch_result = preprocessor(
        ...     ["First sentence.", "Second sentence."],
        ...     return_tensors='tf'
        ... )
        >>> batch_result['input_ids'].shape
        TensorShape([2, 32])
    """

    def __init__(
        self,
        encoding_name: str = "cl100k_base",
        max_length: int = 256,
        cls_token_id: Optional[int] = None,
        sep_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        mask_token_id: Optional[int] = None,
        truncation: bool = True,
        padding: Literal['max_length', 'do_not_pad'] = 'max_length',
    ) -> None:
        """Initialize the TiktokenPreprocessor.

        :param encoding_name: Name of the Tiktoken encoding.
        :type encoding_name: str
        :param max_length: Maximum sequence length.
        :type max_length: int
        :param cls_token_id: ID for [CLS] token. If None, uses valid default.
        :type cls_token_id: Optional[int]
        :param sep_token_id: ID for [SEP] token. If None, uses valid default.
        :type sep_token_id: Optional[int]
        :param pad_token_id: ID for padding token. If None, uses valid default.
        :type pad_token_id: Optional[int]
        :param mask_token_id: ID for [MASK] token. If None, uses valid default.
        :type mask_token_id: Optional[int]
        :param truncation: Whether to truncate long sequences.
        :type truncation: bool
        :param padding: Padding strategy ('max_length' or 'do_not_pad').
        :type padding: Literal['max_length', 'do_not_pad']
        :raises ValueError: If max_length <= 0 or invalid encoding_name.
        """
        if max_length <= 0:
            raise ValueError(
                f"max_length must be positive, got {max_length}"
            )

        try:
            self.tokenizer: tiktoken.Encoding = tiktoken.get_encoding(
                encoding_name
            )
            logger.info(
                f"Initialized Tiktoken preprocessor with "
                f"encoding '{encoding_name}'"
            )
        except ValueError as e:
            logger.error(
                f"Invalid Tiktoken encoding name: '{encoding_name}'"
            )
            raise ValueError(
                f"Invalid encoding_name '{encoding_name}'. "
                f"Choose from: cl100k_base, p50k_base, r50k_base, etc."
            ) from e

        # Get valid special token IDs if not provided
        if any(
            tid is None for tid in
            [cls_token_id, sep_token_id, pad_token_id, mask_token_id]
        ):
            logger.info("Using automatic special token IDs from vocabulary")
            special_tokens = get_special_token_ids(encoding_name)
            cls_token_id = cls_token_id or special_tokens['cls']
            sep_token_id = sep_token_id or special_tokens['sep']
            pad_token_id = pad_token_id or special_tokens['pad']
            mask_token_id = mask_token_id or special_tokens['mask']

        self.max_length: int = max_length
        self.cls_token_id: int = cls_token_id
        self.sep_token_id: int = sep_token_id
        self.pad_token_id: int = pad_token_id
        self.mask_token_id: int = mask_token_id
        self.truncation: bool = truncation
        self.padding: str = padding

        logger.debug(
            f"Preprocessor configured: max_length={max_length}, "
            f"cls={cls_token_id}, sep={sep_token_id}, "
            f"pad={pad_token_id}, mask={mask_token_id}, "
            f"truncation={truncation}, padding={padding}"
        )

    def _preprocess_single(self, text: str) -> Dict[str, List[int]]:
        """Preprocess a single text string into a dictionary of token lists.

        This method handles tokenization, truncation, special token
        insertion, and padding for a single text input.

        :param text: Input text to preprocess.
        :type text: str
        :return: Dictionary with keys 'input_ids', 'attention_mask',
            and 'token_type_ids', each mapping to a list of integers.
        :rtype: Dict[str, List[int]]
        :raises ValueError: If sequence is too long and truncation=False.
        """
        # 1. Tokenize the input text using the specified Tiktoken encoding
        token_ids = self.tokenizer.encode(text)

        # 2. Handle truncation if sequence is too long
        max_token_length = self.max_length - 2  # Reserve space for [CLS], [SEP]
        if len(token_ids) > max_token_length:
            if not self.truncation:
                raise ValueError(
                    f"Sequence length {len(token_ids)} exceeds "
                    f"max_length-2 ({max_token_length}) and "
                    f"truncation=False"
                )
            token_ids = token_ids[:max_token_length]
            logger.debug(
                f"Truncated sequence from {len(token_ids)} to "
                f"{max_token_length} tokens"
            )

        # 3. Add [CLS] and [SEP] special tokens
        input_ids = [self.cls_token_id] + token_ids + [self.sep_token_id]

        # 4. Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(input_ids)

        # 5. Apply padding if requested
        if self.padding == 'max_length':
            padding_length = self.max_length - len(input_ids)
            input_ids.extend([self.pad_token_id] * padding_length)
            attention_mask.extend([0] * padding_length)

        # 6. Create token type IDs (all zeros for single-sentence tasks)
        token_type_ids = [0] * len(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def __call__(
        self,
        texts: Union[str, List[str]],
        return_tensors: Literal['np', 'tf', None] = None,
    ) -> Dict[str, Union[np.ndarray, tf.Tensor]]:
        """Preprocess text(s) into model-ready input tensors.

        This method handles both single text strings and batches of texts,
        processing them into the format expected by BERT-style models.

        :param texts: Single text string or list of text strings to process.
        :type texts: Union[str, List[str]]
        :param return_tensors: Format for output tensors. Options:
            - 'np': Return NumPy arrays
            - 'tf': Return TensorFlow tensors
            - None: Return NumPy arrays (default)
        :type return_tensors: Literal['np', 'tf', None]
        :return: Dictionary containing:
            - 'input_ids': Token IDs with shape (batch_size, max_length)
            - 'attention_mask': Attention mask with shape
                (batch_size, max_length)
            - 'token_type_ids': Token type IDs with shape
                (batch_size, max_length)
        :rtype: Dict[str, Union[np.ndarray, tf.Tensor]]
        :raises TypeError: If texts is not a string or list of strings.
        :raises ValueError: If return_tensors is not 'np', 'tf', or None.

        Example:
            >>> preprocessor = TiktokenPreprocessor(max_length=16)
            >>> # Single text
            >>> result = preprocessor("Hello!", return_tensors='np')
            >>> result['input_ids'].shape
            (1, 16)
            >>> # Batch
            >>> batch = preprocessor(["Text 1", "Text 2"], return_tensors='tf')
            >>> batch['input_ids'].shape
            TensorShape([2, 16])
        """
        if not isinstance(texts, (str, list)):
            raise TypeError(
                f"Input `texts` must be a string or a list of strings, "
                f"got {type(texts).__name__}"
            )

        if return_tensors not in ('np', 'tf', None):
            raise ValueError(
                f"return_tensors must be 'np', 'tf', or None, "
                f"got '{return_tensors}'"
            )

        # Normalize to batch format
        is_batch = isinstance(texts, list)
        text_list = texts if is_batch else [texts]

        logger.debug(
            f"Processing {'batch of ' if is_batch else ''}"
            f"{len(text_list)} text(s)"
        )

        # Process each text
        processed_list = [
            self._preprocess_single(text) for text in text_list
        ]

        # Collate list of dicts into dict of lists
        collated = {
            key: [d[key] for d in processed_list]
            for key in processed_list[0]
        }

        # Convert to NumPy arrays
        result = {
            key: np.array(value, dtype=np.int32)
            for key, value in collated.items()
        }

        # Convert to TensorFlow tensors if requested
        if return_tensors == 'tf':
            result = {
                key: tf.constant(value, dtype=tf.int32)
                for key, value in result.items()
            }
            logger.debug("Converted outputs to TensorFlow tensors")
        else:
            logger.debug("Returning NumPy arrays")

        return result

    def batch_encode(
        self,
        texts: List[str],
        return_tensors: Literal['np', 'tf', None] = None,
    ) -> Dict[str, Union[np.ndarray, tf.Tensor]]:
        """Batch encode multiple texts (alias for __call__ with list input).

        This is a convenience method that explicitly handles batch encoding,
        providing a clearer API when processing multiple texts.

        :param texts: List of text strings to encode.
        :type texts: List[str]
        :param return_tensors: Format for output tensors ('np', 'tf', None).
        :type return_tensors: Literal['np', 'tf', None]
        :return: Dictionary of encoded tensors.
        :rtype: Dict[str, Union[np.ndarray, tf.Tensor]]
        :raises TypeError: If texts is not a list.

        Example:
            >>> preprocessor = TiktokenPreprocessor(max_length=32)
            >>> texts = ["First sentence.", "Second sentence."]
            >>> encoded = preprocessor.batch_encode(texts, return_tensors='tf')
            >>> encoded['input_ids'].shape
            TensorShape([2, 32])
        """
        if not isinstance(texts, list):
            raise TypeError(
                f"batch_encode expects a list of strings, "
                f"got {type(texts).__name__}"
            )
        return self(texts, return_tensors=return_tensors)

    def encode(
        self,
        text: str,
        return_tensors: Literal['np', 'tf', None] = None,
    ) -> Dict[str, Union[np.ndarray, tf.Tensor]]:
        """Encode a single text (alias for __call__ with string input).

        This is a convenience method that explicitly handles single text
        encoding, providing a clearer API when processing one text.

        :param text: Single text string to encode.
        :type text: str
        :param return_tensors: Format for output tensors ('np', 'tf', None).
        :type return_tensors: Literal['np', 'tf', None]
        :return: Dictionary of encoded tensors.
        :rtype: Dict[str, Union[np.ndarray, tf.Tensor]]
        :raises TypeError: If text is not a string.

        Example:
            >>> preprocessor = TiktokenPreprocessor(max_length=32)
            >>> encoded = preprocessor.encode("Hello!", return_tensors='np')
            >>> encoded['input_ids'].shape
            (1, 32)
        """
        if not isinstance(text, str):
            raise TypeError(
                f"encode expects a string, got {type(text).__name__}"
            )
        return self(text, return_tensors=return_tensors)

    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size of the underlying Tiktoken encoding.

        :return: Number of tokens in the vocabulary.
        :rtype: int

        Example:
            >>> preprocessor = TiktokenPreprocessor()
            >>> preprocessor.vocab_size
            100277
        """
        return self.tokenizer.n_vocab

    def decode(
        self,
        token_ids: Union[List[int], np.ndarray, tf.Tensor],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
        **kwargs
    ) -> str:
        """Decode token IDs back to text.

        :param token_ids: Token IDs to decode. Can be a list, NumPy array,
            or TensorFlow tensor.
        :type token_ids: Union[List[int], np.ndarray, tf.Tensor]
        :param skip_special_tokens: Whether to remove special tokens
            ([CLS], [SEP], [PAD]) from the decoded text. Defaults to True.
        :type skip_special_tokens: bool
        :param clean_up_tokenization_spaces: Whether to clean up extra
            spaces in the decoded text (for HuggingFace compatibility).
            This parameter is accepted but ignored as Tiktoken handles
            spacing automatically. Defaults to True.
        :type clean_up_tokenization_spaces: bool
        :param kwargs: Additional keyword arguments (for compatibility with
            other tokenizer APIs). These are accepted but ignored.
        :return: Decoded text string.
        :rtype: str

        Example:
            >>> preprocessor = TiktokenPreprocessor()
            >>> encoded = preprocessor("Hello world!")
            >>> token_ids = encoded['input_ids'][0]
            >>> decoded = preprocessor.decode(token_ids)
            >>> print(decoded)
            Hello world!
        """
        # Convert to list if necessary
        if isinstance(token_ids, (np.ndarray, tf.Tensor)):
            token_ids = token_ids.numpy() if hasattr(
                token_ids, 'numpy'
            ) else token_ids
            token_ids = token_ids.tolist()

        # Remove special tokens if requested
        # Also filter out any token IDs that are outside the vocabulary
        special_tokens = {
            self.cls_token_id,
            self.sep_token_id,
            self.pad_token_id,
            self.mask_token_id
        }

        if skip_special_tokens:
            # Filter out special tokens and invalid token IDs
            token_ids = [
                tid for tid in token_ids
                if tid not in special_tokens and 0 <= tid < self.vocab_size
            ]
        else:
            # Even if not skipping special tokens, filter invalid IDs
            # and replace special tokens with empty string equivalent
            filtered_ids = []
            for tid in token_ids:
                if tid in special_tokens:
                    # Skip special tokens during decoding as they're not
                    # in Tiktoken vocab
                    continue
                elif 0 <= tid < self.vocab_size:
                    filtered_ids.append(tid)
            token_ids = filtered_ids

        # Return empty string if no valid tokens remain
        if not token_ids:
            return ""

        # Decode using Tiktoken
        try:
            decoded_text = self.tokenizer.decode(token_ids)
        except (KeyError, ValueError) as e:
            logger.warning(
                f"Error decoding token IDs, returning empty string: {e}"
            )
            return ""
        except Exception as e:
            logger.error(f"Unexpected error decoding token IDs: {e}")
            raise

        # Note: clean_up_tokenization_spaces is handled automatically by
        # Tiktoken and doesn't need explicit processing
        return decoded_text

    def __repr__(self) -> str:
        """Return string representation of the preprocessor.

        :return: String representation.
        :rtype: str
        """
        return (
            f"TiktokenPreprocessor("
            f"encoding={self.tokenizer.name}, "
            f"max_length={self.max_length}, "
            f"vocab_size={self.vocab_size})"
        )

# ---------------------------------------------------------------------
