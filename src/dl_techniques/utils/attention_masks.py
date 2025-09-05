"""
Attention mask utilities for transformer models.

This module provides utilities for creating various types of attention masks
commonly used in transformer architectures, including causal masks, sliding
window masks, block diagonal masks, and more.

These utilities support both global and local attention patterns and are
designed to be backend-agnostic and symbolically safe for use in Keras 3
computational graphs.
"""

import keras
from keras import ops
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

def create_causal_mask(
        seq_len: int,
        dtype: str = "bool"
) -> keras.KerasTensor:
    """
    Create a causal (lower triangular) attention mask.

    This mask ensures that each position can only attend to positions
    at or before its own position, preventing information leakage from
    future tokens.

    Args:
        seq_len: Length of the sequence.
        dtype: Data type for the mask. Defaults to "bool".

    Returns:
        A tensor of shape (seq_len, seq_len) where True indicates
        positions that should be masked (blocked) in attention.

    Example:
        ```python
        mask = create_causal_mask(4)
        # Returns:
        # [[False, True,  True,  True ],
        #  [False, False, True,  True ],
        #  [False, False, False, True ],
        #  [False, False, False, False]]
        ```
    """
    # Create index tensors for efficient mask computation
    i = ops.arange(seq_len)[:, None]  # Shape: (seq_len, 1)
    j = ops.arange(seq_len)  # Shape: (seq_len,)

    # Create mask where j > i (future positions)
    mask = j > i

    if dtype != "bool":
        mask = ops.cast(mask, dtype)

    logger.debug(f"Created causal mask with shape ({seq_len}, {seq_len})")
    return mask

# ---------------------------------------------------------------------

def create_sliding_window_mask(
        seq_len: int,
        window_size: int,
        dtype: str = "bool"
) -> keras.KerasTensor:
    """
    Create a sliding window attention mask.

    This mask allows each position to attend to positions within a
    specified window around itself, enabling local attention patterns
    while maintaining efficiency for long sequences.

    Args:
        seq_len: Length of the sequence.
        window_size: Size of the attention window. Each position can
            attend to `window_size` positions before itself.
        dtype: Data type for the mask. Defaults to "bool".

    Returns:
        A tensor of shape (seq_len, seq_len) where True indicates
        positions that should be masked (blocked) in attention.

    Example:
        ```python
        mask = create_sliding_window_mask(5, window_size=2)
        # Each position can attend to 2 positions before itself
        # Returns:
        # [[False, True,  True,  True,  True ],
        #  [False, False, True,  True,  True ],
        #  [False, False, False, True,  True ],
        #  [True,  False, False, False, True ],
        #  [True,  True,  False, False, False]]
        ```
    """
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")

    # Create index tensors
    i = ops.arange(seq_len)[:, None]  # Shape: (seq_len, 1)
    j = ops.arange(seq_len)  # Shape: (seq_len,)

    # Create causal mask (can't attend to future)
    causal_mask = j > i

    # Create "too far past" mask
    far_past_mask = (i - j) >= window_size

    # Combine masks: block both future and too-far-past positions
    mask = ops.logical_or(causal_mask, far_past_mask)

    if dtype != "bool":
        mask = ops.cast(mask, dtype)

    logger.debug(f"Created sliding window mask with shape ({seq_len}, {seq_len}), window_size={window_size}")
    return mask

# ---------------------------------------------------------------------

def create_global_local_masks(
        seq_len: int,
        sliding_window: int,
        dtype: str = "bool"
) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
    """
    Create attention masks for global and local (sliding window) attention.

    This is the original function from the provided code, adapted as a utility.
    It creates both a global causal mask and a local sliding window mask
    that can be used in attention mechanisms that support both patterns.

    Args:
        seq_len: Sequence length.
        sliding_window: Size of the sliding window for local attention.
        dtype: Data type for the masks. Defaults to "bool".

    Returns:
        Tuple of (global_mask, local_mask) tensors, both of shape
        (seq_len, seq_len) where True indicates positions that should
        be masked (blocked) in attention.

    Example:
        ```python
        global_mask, local_mask = create_global_local_masks(4, sliding_window=2)

        # global_mask (causal):
        # [[False, True,  True,  True ],
        #  [False, False, True,  True ],
        #  [False, False, False, True ],
        #  [False, False, False, False]]

        # local_mask (causal + sliding window):
        # [[False, True,  True,  True ],
        #  [False, False, True,  True ],
        #  [False, False, False, True ],
        #  [True,  False, False, False]]
        ```
    """
    if sliding_window <= 0:
        raise ValueError(f"sliding_window must be positive, got {sliding_window}")

    # Create indices which are safe for symbolic execution.
    # i represents row indices: [[0], [1], ..., [seq_len-1]]
    i = ops.arange(seq_len)[:, None]
    # j represents column indices: [[0, 1, ..., seq_len-1]]
    j = ops.arange(seq_len)

    # 1. Create global (causal) mask.
    # This replaces `ops.triu(ones, k=1)` with a robust equivalent.
    # A position (i, j) is masked if the column j is ahead of the row i.
    global_mask = j > i

    # 2. Create the "too-far-past" mask for the sliding window.
    # A position (i, j) is masked if the distance (i - j) is >= the window size.
    far_past_mask = (i - j) >= sliding_window

    # 3. Combine them for the final local attention mask.
    local_mask = ops.logical_or(global_mask, far_past_mask)

    if dtype != "bool":
        global_mask = ops.cast(global_mask, dtype)
        local_mask = ops.cast(local_mask, dtype)

    logger.debug(f"Created global and local masks with shape ({seq_len}, {seq_len}), sliding_window={sliding_window}")
    return global_mask, local_mask

# ---------------------------------------------------------------------

def create_block_diagonal_mask(
        seq_len: int,
        block_size: int,
        dtype: str = "bool"
) -> keras.KerasTensor:
    """
    Create a block diagonal attention mask.

    This mask partitions the sequence into non-overlapping blocks and
    allows attention only within each block. This is useful for
    hierarchical attention patterns.

    Args:
        seq_len: Length of the sequence.
        block_size: Size of each attention block.
        dtype: Data type for the mask. Defaults to "bool".

    Returns:
        A tensor of shape (seq_len, seq_len) where False indicates
        positions that can attend to each other (within same block).

    Example:
        ```python
        mask = create_block_diagonal_mask(6, block_size=2)
        # Returns:
        # [[False, False, True,  True,  True,  True ],
        #  [False, False, True,  True,  True,  True ],
        #  [True,  True,  False, False, True,  True ],
        #  [True,  True,  False, False, True,  True ],
        #  [True,  True,  True,  True,  False, False],
        #  [True,  True,  True,  True,  False, False]]
        ```
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    # Create position indices
    positions = ops.arange(seq_len)

    # Calculate which block each position belongs to
    block_ids_i = positions[:, None] // block_size  # Shape: (seq_len, 1)
    block_ids_j = positions // block_size  # Shape: (seq_len,)

    # Mask positions that are in different blocks
    mask = block_ids_i != block_ids_j

    if dtype != "bool":
        mask = ops.cast(mask, dtype)

    logger.debug(f"Created block diagonal mask with shape ({seq_len}, {seq_len}), block_size={block_size}")
    return mask

# ---------------------------------------------------------------------

def create_random_mask(
        seq_len: int,
        mask_probability: float,
        seed: Optional[int] = None,
        dtype: str = "bool"
) -> keras.KerasTensor:
    """
    Create a random attention mask.

    This mask randomly blocks attention connections with a specified
    probability. Useful for regularization or studying attention patterns.

    Args:
        seq_len: Length of the sequence.
        mask_probability: Probability of masking each position pair.
        seed: Random seed for reproducibility. Defaults to None.
        dtype: Data type for the mask. Defaults to "bool".

    Returns:
        A tensor of shape (seq_len, seq_len) where True indicates
        positions that should be masked (blocked) in attention.
    """
    if not 0.0 <= mask_probability <= 1.0:
        raise ValueError(f"mask_probability must be between 0.0 and 1.0, got {mask_probability}")

    if seed is not None:
        keras.utils.set_random_seed(seed)

    # Generate random values and threshold them
    random_values = keras.random.uniform((seq_len, seq_len))
    mask = random_values < mask_probability

    if dtype != "bool":
        mask = ops.cast(mask, dtype)

    logger.debug(f"Created random mask with shape ({seq_len}, {seq_len}), probability={mask_probability}")
    return mask

# ---------------------------------------------------------------------

def create_banded_mask(
        seq_len: int,
        band_width: int,
        dtype: str = "bool"
) -> keras.KerasTensor:
    """
    Create a banded attention mask.

    This mask allows attention within a band around the diagonal,
    creating a local attention pattern that's symmetric (unlike
    sliding window which is causal).

    Args:
        seq_len: Length of the sequence.
        band_width: Width of the attention band (total width, centered on diagonal).
        dtype: Data type for the mask. Defaults to "bool".

    Returns:
        A tensor of shape (seq_len, seq_len) where True indicates
        positions that should be masked (blocked) in attention.

    Example:
        ```python
        mask = create_banded_mask(5, band_width=3)
        # Allows attention within ±1 position from diagonal
        # Returns:
        # [[False, False, True,  True,  True ],
        #  [False, False, False, True,  True ],
        #  [True,  False, False, False, True ],
        #  [True,  True,  False, False, False],
        #  [True,  True,  True,  False, False]]
        ```
    """
    if band_width <= 0:
        raise ValueError(f"band_width must be positive, got {band_width}")

    # Create index tensors
    i = ops.arange(seq_len)[:, None]  # Shape: (seq_len, 1)
    j = ops.arange(seq_len)  # Shape: (seq_len,)

    # Calculate distance from diagonal
    distance = ops.abs(i - j)

    # Mask positions outside the band
    half_width = band_width // 2
    mask = distance > half_width

    if dtype != "bool":
        mask = ops.cast(mask, dtype)

    logger.debug(f"Created banded mask with shape ({seq_len}, {seq_len}), band_width={band_width}")
    return mask

# ---------------------------------------------------------------------

def apply_mask_to_logits(
        logits: keras.KerasTensor,
        mask: keras.KerasTensor,
        mask_value: float = -1e9
) -> keras.KerasTensor:
    """
    Apply an attention mask to attention logits.

    This function applies a mask to attention logits by setting masked
    positions to a large negative value, effectively making their
    attention weights zero after softmax.

    Args:
        logits: Attention logits tensor of shape (..., seq_len, seq_len).
        mask: Boolean mask tensor of shape (seq_len, seq_len) or broadcastable shape.
            True values indicate positions to mask.
        mask_value: Value to set for masked positions. Should be a large
            negative number. Defaults to -1e9.

    Returns:
        Masked logits tensor with the same shape as input logits.

    Example:
        ```python
        # Create some attention logits
        logits = keras.random.normal((2, 4, 4))  # (batch, seq, seq)

        # Create a causal mask
        mask = create_causal_mask(4)

        # Apply mask
        masked_logits = apply_mask_to_logits(logits, mask)
        ```
    """
    # Convert mask to same dtype as logits for numerical operations
    mask = ops.cast(mask, logits.dtype)

    # Apply mask: where mask is True, set to mask_value
    masked_logits = ops.where(mask, mask_value, logits)

    return masked_logits

# ---------------------------------------------------------------------

def create_attention_mask_from_padding(
        padding_mask: keras.KerasTensor,
        dtype: str = "bool"
) -> keras.KerasTensor:
    """
    Create attention mask from padding mask.

    Converts a 1D padding mask (indicating which positions are padding)
    into a 2D attention mask (indicating which position pairs should be masked).

    Args:
        padding_mask: Boolean tensor of shape (batch_size, seq_len) where
            True indicates padding positions.
        dtype: Data type for the output mask. Defaults to "bool".

    Returns:
        Attention mask tensor of shape (batch_size, seq_len, seq_len)
        where True indicates positions that should be masked.

    Example:
        ```python
        # Padding mask: [False, False, True, True] (first 2 are real, last 2 are padding)
        padding_mask = ops.array([[False, False, True, True]])

        attention_mask = create_attention_mask_from_padding(padding_mask)
        # Result masks attention to/from padding positions
        ```
    """
    batch_size, seq_len = ops.shape(padding_mask)

    # Expand padding mask to 2D: (batch, seq_len, 1) and (batch, 1, seq_len)
    mask_2d_i = padding_mask[:, :, None]  # (batch, seq_len, 1)
    mask_2d_j = padding_mask[:, None, :]  # (batch, 1, seq_len)

    # Create attention mask: mask if either source or target is padding
    attention_mask = ops.logical_or(mask_2d_i, mask_2d_j)

    if dtype != "bool":
        attention_mask = ops.cast(attention_mask, dtype)

    logger.debug(f"Created attention mask from padding mask with shape {ops.shape(attention_mask)}")
    return attention_mask


# ---------------------------------------------------------------------

def combine_masks(
        *masks: keras.KerasTensor,
        combination_type: Literal["and", "or"] = "or"
) -> keras.KerasTensor:
    """
    Combine multiple attention masks.

    Args:
        *masks: Variable number of mask tensors to combine.
        combination_type: How to combine the masks:
            - "or": Logical OR (union of masked positions)
            - "and": Logical AND (intersection of masked positions)

    Returns:
        Combined mask tensor.

    Example:
        ```python
        causal_mask = create_causal_mask(4)
        window_mask = create_sliding_window_mask(4, 2)

        # Combine with OR (mask if either mask blocks it)
        combined = combine_masks(causal_mask, window_mask, combination_type="or")
        ```
    """
    if not masks:
        raise ValueError("At least one mask must be provided")

    if len(masks) == 1:
        return masks[0]

    result = masks[0]
    for mask in masks[1:]:
        if combination_type == "or":
            result = ops.logical_or(result, mask)
        elif combination_type == "and":
            result = ops.logical_and(result, mask)
        else:
            raise ValueError(f"combination_type must be 'and' or 'or', got {combination_type}")

    logger.debug(f"Combined {len(masks)} masks using {combination_type}")
    return result

# ---------------------------------------------------------------------

def visualize_mask(
        mask: keras.KerasTensor,
        title: str = "Attention Mask"
) -> None:
    """
    Visualize an attention mask using matplotlib.

    This is a utility function for debugging and understanding
    attention patterns.

    Args:
        mask: Boolean mask tensor of shape (seq_len, seq_len).
        title: Title for the plot.

    Note:
        This function requires matplotlib and is mainly for debugging purposes.
        It will warn if matplotlib is not available.
    """
    try:

        # Convert to numpy for plotting
        mask_np = keras.ops.convert_to_numpy(mask)

        # Create figure
        plt.figure(figsize=(8, 6))
        plt.imshow(mask_np, cmap='RdBu_r', interpolation='nearest')
        plt.colorbar(label='Masked (True=1, False=0)')
        plt.title(title)
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')

        # Add grid for better readability
        seq_len = mask_np.shape[0]
        plt.xticks(range(seq_len))
        plt.yticks(range(seq_len))
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    except ImportError:
        logger.warning("matplotlib not available for mask visualization")

# ---------------------------------------------------------------------
