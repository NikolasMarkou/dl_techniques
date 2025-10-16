"""
Advanced Masking and Corruption Strategies
==========================================

This module provides high-level functions for complex masking and data
corruption strategies that go beyond simple boolean mask creation. These
are intended for use in self-supervised learning tasks like MLM.
"""

import tensorflow as tf
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

def apply_mlm_masking(
    input_ids: tf.Tensor,
    attention_mask: Optional[tf.Tensor],
    vocab_size: int,
    mask_ratio: float,
    mask_token_id: int,
    special_token_ids: List[int],
    random_token_ratio: float = 0.1,
    unchanged_ratio: float = 0.1,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Performs dynamic token masking and corruption according to BERT's strategy.

    This function encapsulates the entire MLM data corruption process.

    Masking Strategy:
    - `mask_ratio` (e.g., 15%) of tokens are selected for masking, excluding
      special tokens and padding.
    - Of the selected tokens:
      - 80% are replaced with `mask_token_id`.
      - 10% are replaced with a random token from the vocabulary.
      - 10% are left unchanged.

    :param input_ids: The original, unmasked input token IDs.
        Shape: (batch_size, seq_len)
    :type input_ids: tf.Tensor
    :param attention_mask: The attention mask, used to identify padding tokens.
        Shape: (batch_size, seq_len)
    :type attention_mask: Optional[tf.Tensor]
    :param vocab_size: The total size of the vocabulary.
    :type vocab_size: int
    :param mask_ratio: The probability of a token being chosen for masking.
    :type mask_ratio: float
    :param mask_token_id: The vocabulary ID for the `[MASK]` token.
    :type mask_token_id: int
    :param special_token_ids: A list of special token IDs (e.g., [CLS], [SEP])
        to exclude from masking.
    :type special_token_ids: List[int]
    :param random_token_ratio: The probability of replacing a selected token
        with a random token. Defaults to 0.1.
    :type random_token_ratio: float
    :param unchanged_ratio: The probability of leaving a selected token as is.
        Defaults to 0.1.
    :type unchanged_ratio: float
    :return: A tuple containing:
             - `masked_input_ids`: The new input IDs with corrupted tokens.
             - `labels`: The original token IDs (ground truth).
             - `mask`: A boolean mask indicating which tokens were selected for masking.
    :rtype: Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
    """
    labels = tf.identity(input_ids)
    input_shape = tf.shape(input_ids)

    # Create a boolean mask for tokens that should *never* be masked
    non_maskable = tf.zeros_like(input_ids, dtype=tf.bool)
    for token_id in special_token_ids:
        non_maskable = tf.logical_or(non_maskable, input_ids == token_id)

    if attention_mask is not None:
        padding_mask = attention_mask == 0
        non_maskable = tf.logical_or(non_maskable, padding_mask)

    # Determine which tokens to mask (mask_ratio of non-special tokens)
    mask_probabilities = tf.random.uniform(input_shape, dtype=tf.float32)
    should_mask = (mask_probabilities < mask_ratio) & ~non_maskable

    # Decide on the corruption strategy for each token.
    # We generate probabilities for all tokens, which is XLA-friendly.
    corruption_probs = tf.random.uniform(input_shape, dtype=tf.float32)
    mask_threshold = 1.0 - random_token_ratio - unchanged_ratio
    random_threshold = 1.0 - unchanged_ratio

    # 80% -> Replace with [MASK]
    mask_token_mask = should_mask & (corruption_probs < mask_threshold)

    # 10% -> Replace with random token
    random_token_mask = should_mask & (corruption_probs >= mask_threshold) & (
        corruption_probs < random_threshold
    )

    # Generate a full tensor of random tokens. This is XLA-compatible because
    # the shape is static for the batch.
    random_tokens = tf.random.uniform(
        shape=input_shape, minval=0, maxval=vocab_size, dtype=input_ids.dtype
    )

    # Start with original IDs and apply corruptions using tf.where, which is
    # efficient and XLA-friendly.

    # First, replace tokens with random tokens where specified
    masked_input_ids = tf.where(random_token_mask, random_tokens, input_ids)

    # Next, replace tokens with the [MASK] token where specified.
    # This correctly overwrites the original IDs but not the random ones.
    masked_input_ids = tf.where(
        mask_token_mask, mask_token_id, masked_input_ids
    )

    return masked_input_ids, labels, should_mask

# ---------------------------------------------------------------------