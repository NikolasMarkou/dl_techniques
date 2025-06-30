"""
Custom Byte-Pair Encoding (BPE) Tokenizer implementation for Keras 3.x.

This module provides a backend-agnostic BPE tokenizer that can be used with
TensorFlow, JAX, or PyTorch backends in Keras 3.x.

Performance-optimized implementation based on standard BPE algorithms.
"""

import re
import keras
import collections
from typing import Dict, List, Tuple, Optional, Any, Set

from dl_techniques.utils.logger import logger


def train_bpe(
        texts: List[str],
        vocab_size: int = 50000,
        min_frequency: int = 2,
        do_lower_case: bool = True,
        handle_punctuation: bool = True
) -> Tuple[Dict[str, int], List[Tuple[str, str]]]:
    """
    Train BPE tokenizer on a corpus of texts with optimized performance.

    Args:
        texts: List of text strings to train on
        vocab_size: Target vocabulary size
        min_frequency: Minimum frequency for a pair to be considered for merging
        do_lower_case: Whether to lowercase text during training
        handle_punctuation: Whether to separate punctuation as distinct tokens

    Returns:
        Tuple of (vocabulary dict, list of merge operations)
    """
    logger.info(f"Training BPE tokenizer with vocab_size={vocab_size}")

    # Initialize with character-level vocabulary
    word_freqs = collections.defaultdict(int)

    # Count word frequencies and split into characters
    for text in texts:
        if do_lower_case:
            text = text.lower()

        # Enhanced word splitting: handle punctuation separately (GPT-2 style)
        if handle_punctuation:
            # Split on whitespace while preserving punctuation as separate tokens
            words = re.findall(r'\w+|[^\s\w]+', text)
        else:
            # Simple whitespace splitting
            words = text.split()

        for word in words:
            # Add end-of-word marker
            word_freqs[' '.join(list(word)) + ' </w>'] += 1

    # Initialize vocabulary with characters
    vocab = set()
    for word in word_freqs.keys():
        vocab.update(word.split())

    # Convert to sorted list for deterministic behavior
    vocab = sorted(list(vocab))
    vocab_dict = {token: i for i, token in enumerate(vocab)}

    merges = []

    while len(vocab_dict) < vocab_size:
        # Count pairs - optimized version
        pairs = collections.defaultdict(int)
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq

        if not pairs:
            break

        # Find most frequent pair
        best_pair = max(pairs, key=pairs.get)

        if pairs[best_pair] < min_frequency:
            break

        # PERFORMANCE FIX: In-place update instead of rebuilding dictionary
        best_pair_str = ' '.join(best_pair)
        new_token = ''.join(best_pair)

        # Use list of keys to avoid modifying dict while iterating
        words_to_update = []
        for word in word_freqs:
            if best_pair_str in word:
                words_to_update.append(word)

        for word in words_to_update:
            freq = word_freqs[word]
            new_word = word.replace(best_pair_str, new_token)

            # Update frequencies
            del word_freqs[word]
            word_freqs[new_word] = word_freqs.get(new_word, 0) + freq

        merges.append(best_pair)

        # Add merged token to vocabulary
        if new_token not in vocab_dict:
            vocab_dict[new_token] = len(vocab_dict)

    logger.info(f"BPE training completed. Final vocab size: {len(vocab_dict)}")
    return vocab_dict, merges


@keras.saving.register_keras_serializable()
class BPETokenizer(keras.layers.Layer):
    """
    Byte-Pair Encoding (BPE) tokenizer layer for Keras 3.x.

    This layer implements BPE tokenization in a backend-agnostic way,
    compatible with TensorFlow, JAX, and PyTorch backends.

    IMPORTANT: This layer is designed for preprocessing and model serialization.
    The call() method is not implemented for in-graph string processing.
    Use tokenize_texts() for actual tokenization.

    Args:
        vocab_dict: Dictionary mapping tokens to IDs
        merges: List of merge operations from BPE training
        max_length: Maximum sequence length (sequences will be truncated/padded)
        pad_token: Token used for padding
        unk_token: Token used for unknown words
        eos_token: End of sequence token
        do_lower_case: Whether to lowercase input text
        **kwargs: Additional keyword arguments for the Layer base class

    Example:
        >>> # Train BPE on texts
        >>> texts = ["hello world", "how are you"]
        >>> vocab_dict, merges = train_bpe(texts, vocab_size=1000)
        >>>
        >>> # Create tokenizer
        >>> tokenizer = BPETokenizer(vocab_dict=vocab_dict, merges=merges)
        >>>
        >>> # Tokenize text (preprocessing step)
        >>> token_sequences = tokenizer.tokenize_texts(["this is a test"])
        >>>
        >>> # Use in model with pre-tokenized input
        >>> inputs = keras.Input(shape=(None,), dtype='int32')  # Pre-tokenized IDs
        >>> embeddings = keras.layers.Embedding(len(vocab_dict), 256)(inputs)
        >>> model = keras.Model(inputs=inputs, outputs=embeddings)
    """

    def __init__(
            self,
            vocab_dict: Optional[Dict[str, int]] = None,
            merges: Optional[List[Tuple[str, str]]] = None,
            max_length: int = 512,
            pad_token: str = "<pad>",
            unk_token: str = "<unk>",
            eos_token: str = "<eos>",
            do_lower_case: bool = True,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.vocab_dict = vocab_dict or {}
        self.merges = merges or []
        self.max_length = max_length
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.eos_token = eos_token
        self.do_lower_case = do_lower_case

        # Add special tokens to vocabulary if not present
        special_tokens = [self.pad_token, self.unk_token, self.eos_token]
        for token in special_tokens:
            if token not in self.vocab_dict:
                self.vocab_dict[token] = len(self.vocab_dict)

        # Create reverse mapping
        self.id_to_token = {v: k for k, v in self.vocab_dict.items()}

        # PERFORMANCE FIX: Pre-compute merge priorities for O(1) lookup
        self.merge_priorities = {pair: i for i, pair in enumerate(self.merges)}

        # Cache vocabulary size
        self.vocab_size = len(self.vocab_dict)

        logger.info(f"BPE tokenizer initialized with vocab size: {self.vocab_size}")

    def _get_pairs(self, tokens: List[str]) -> Set[Tuple[str, str]]:
        """Get all adjacent pairs in a token sequence."""
        return set(zip(tokens, tokens[1:]))

    def _apply_bpe(self, word: str) -> List[str]:
        """
        Apply BPE merges to a single word using optimized algorithm.

        Args:
            word: Word to tokenize

        Returns:
            List of subword tokens
        """
        if not word:
            return []

        # Start with character-level split
        word_tokens = list(word) + ['</w>']

        # PERFORMANCE FIX: Efficient BPE application
        while True:
            pairs = self._get_pairs(word_tokens)
            if not pairs:
                break

            # Find the best pair to merge based on learned order
            best_pair_to_merge = min(
                pairs,
                key=lambda p: self.merge_priorities.get(p, float('inf'))
            )

            if best_pair_to_merge not in self.merge_priorities:
                break  # No more mergeable pairs found

            # Merge the best pair
            first, second = best_pair_to_merge
            new_tokens = []
            i = 0

            while i < len(word_tokens):
                try:
                    # Find the first occurrence of the best pair
                    j = word_tokens.index(first, i)
                    if j < len(word_tokens) - 1 and word_tokens[j + 1] == second:
                        # Add tokens before the pair
                        new_tokens.extend(word_tokens[i:j])
                        # Add merged token
                        new_tokens.append(first + second)
                        i = j + 2
                    else:
                        new_tokens.append(word_tokens[i])
                        i += 1
                except ValueError:
                    # No more occurrences, add remaining tokens
                    new_tokens.extend(word_tokens[i:])
                    break

            word_tokens = new_tokens

        return word_tokens

    def _tokenize_text(self, text: str) -> List[int]:
        """
        Tokenize a single text string to token IDs.

        Args:
            text: Input text string

        Returns:
            List of token IDs
        """
        if self.do_lower_case:
            text = text.lower()

        # Simple word splitting - could be enhanced with regex for punctuation
        words = text.split()
        token_ids = []

        for word in words:
            word_tokens = self._apply_bpe(word)
            for token in word_tokens:
                token_id = self.vocab_dict.get(token, self.vocab_dict[self.unk_token])
                token_ids.append(token_id)

        # Add EOS token
        token_ids.append(self.vocab_dict[self.eos_token])

        # Truncate or pad to max_length
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            pad_id = self.vocab_dict[self.pad_token]
            token_ids.extend([pad_id] * (self.max_length - len(token_ids)))

        return token_ids

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """
        Forward pass placeholder - not implemented for string processing.

        Args:
            inputs: Input tensor
            training: Training mode flag

        Raises:
            NotImplementedError: This layer requires preprocessing with tokenize_texts()
        """
        raise NotImplementedError(
            "The BPETokenizer layer does not support in-graph string tokenization "
            "due to backend-agnostic string processing limitations. "
            "Please use `tokenizer.tokenize_texts(texts)` for preprocessing "
            "your text data, then pass the resulting token IDs to your model.\n\n"
            "Example usage:\n"
            "  token_ids = tokenizer.tokenize_texts(['hello world'])\n"
            "  model.predict(np.array(token_ids))"
        )

    def tokenize_texts(self, texts: List[str]) -> List[List[int]]:
        """
        Tokenize a list of text strings (preprocessing method).

        Args:
            texts: List of text strings to tokenize

        Returns:
            List of token ID sequences
        """
        return [self._tokenize_text(text) for text in texts]

    def decode_tokens(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text.

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded text string
        """
        tokens = []
        for token_id in token_ids:
            if token_id == self.vocab_dict[self.pad_token]:
                break  # Stop at padding
            if token_id == self.vocab_dict[self.eos_token]:
                break  # Stop at EOS

            token = self.id_to_token.get(token_id, self.unk_token)
            tokens.append(token)

        # Join tokens and clean up
        text = ''.join(tokens)
        text = text.replace('</w>', ' ')
        return text.strip()

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], int]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input

        Returns:
            Output shape (batch_size, max_length)
        """
        return (input_shape[0], self.max_length)

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration
        """
        config = super().get_config()
        config.update({
            "vocab_dict": self.vocab_dict,
            "merges": self.merges,
            "max_length": self.max_length,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "eos_token": self.eos_token,
            "do_lower_case": self.do_lower_case,
        })
        return config


@keras.saving.register_keras_serializable()
class TokenEmbedding(keras.layers.Layer):
    """
    Token embedding layer that converts token IDs to dense vectors.

    This is a convenience wrapper around keras.layers.Embedding with
    additional configuration for common NLP patterns.

    Args:
        vocab_size: Size of the vocabulary
        embedding_dim: Dimension of the embedding vectors
        mask_zero: Whether to mask padding tokens (ID 0)
        embeddings_initializer: Initializer for embeddings
        **kwargs: Additional keyword arguments for the Layer base class
    """

    def __init__(
            self,
            vocab_size: int,
            embedding_dim: int,
            mask_zero: bool = True,
            embeddings_initializer: str = "uniform",
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.mask_zero = mask_zero
        self.embeddings_initializer = embeddings_initializer

        # Initialize embedding layer
        self.embedding = keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            mask_zero=mask_zero,
            embeddings_initializer=embeddings_initializer,
            name="token_embedding"
        )

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """
        Forward pass of the embedding layer.

        Args:
            inputs: Input tensor of token IDs
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode

        Returns:
            Tensor of embeddings with shape (batch_size, sequence_length, embedding_dim)
        """
        return self.embedding(inputs, training=training)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input

        Returns:
            Output shape
        """
        return input_shape + (self.embedding_dim,)

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration
        """
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "mask_zero": self.mask_zero,
            "embeddings_initializer": self.embeddings_initializer,
        })
        return config


def create_bpe_pipeline(
        texts: List[str],
        vocab_size: int = 50000,
        embedding_dim: int = 512,
        max_length: int = 512,
        do_lower_case: bool = True,
        min_frequency: int = 2
) -> Tuple[BPETokenizer, TokenEmbedding]:
    """
    Create a complete BPE tokenization and embedding pipeline.

    Args:
        texts: Training texts for BPE
        vocab_size: Target vocabulary size
        embedding_dim: Embedding dimension
        max_length: Maximum sequence length
        do_lower_case: Whether to lowercase text
        min_frequency: Minimum frequency for BPE merges

    Returns:
        Tuple of (tokenizer, embedding layer)
    """
    logger.info("Creating BPE pipeline")

    # Train BPE with performance optimizations
    vocab_dict, merges = train_bpe(
        texts,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        do_lower_case=do_lower_case
    )

    # Create tokenizer
    tokenizer = BPETokenizer(
        vocab_dict=vocab_dict,
        merges=merges,
        max_length=max_length,
        do_lower_case=do_lower_case
    )

    # Create embedding layer
    embedding = TokenEmbedding(
        vocab_size=len(vocab_dict),
        embedding_dim=embedding_dim
    )

    logger.info("BPE pipeline created successfully")
    return tokenizer, embedding


# Example usage demonstrating correct workflow
if __name__ == "__main__":
    # Example training texts
    training_texts = [
        "hello world how are you today",
        "this is a test sentence for tokenization",
        "byte pair encoding is very useful for nlp",
        "neural networks are powerful machine learning tools",
        "transformers use attention mechanisms effectively"
    ]

    # Create BPE pipeline with performance optimizations
    tokenizer, embedding = create_bpe_pipeline(
        texts=training_texts,
        vocab_size=1000,
        embedding_dim=256,
        max_length=128,
        do_lower_case=True
    )

    # CORRECT USAGE: Preprocessing step
    test_texts = ["hello world", "this is new text for testing"]
    token_sequences = tokenizer.tokenize_texts(test_texts)
    print("Tokenized sequences:", token_sequences)

    # Example decoding
    decoded_text = tokenizer.decode_tokens(token_sequences[0])
    print("Decoded text:", decoded_text)

    # Example model usage with pre-tokenized data
    import numpy as np

    # Create simple model for demonstration
    inputs = keras.Input(shape=(128,), dtype='int32')  # Pre-tokenized input
    embeddings = embedding(inputs)
    # Could add more layers here...
    model = keras.Model(inputs=inputs, outputs=embeddings)

    # Use model with preprocessed tokens
    token_array = np.array(token_sequences)
    embeddings_output = model.predict(token_array)
    print("Embeddings shape:", embeddings_output.shape)

    # Demonstrate serialization
    print("\nTesting serialization...")
    tokenizer_config = tokenizer.get_config()
    print(f"Tokenizer serialized config keys: {list(tokenizer_config.keys())}")