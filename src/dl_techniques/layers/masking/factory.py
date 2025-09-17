"""
Masking utilities for attention and segmentation models.

This module provides a comprehensive set of utilities for creating and manipulating
masks used in transformer architectures and segmentation models. It includes both
attention masking patterns (causal, sliding window, etc.) and specialized masks
for instance segmentation tasks.

The module is designed to be backend-agnostic and symbolically safe for use in
Keras 3 computational graphs, with full type hints and extensive documentation.

Classes:
    MaskType: Enum for standard mask types
    MaskConfig: Configuration dataclass for mask creation
    MaskFactory: Factory class for creating masks

Functions:
    create_mask: Main interface for mask creation
    apply_mask: Universal mask application function
    combine_masks: Combine multiple masks with logical operations
    visualize_mask: Visualization utility for debugging

Examples:
    Basic usage:
    >>> # Create a causal mask
    >>> mask = create_mask("causal", seq_len=128)
    >>>
    >>> # Create a sliding window mask with configuration
    >>> config = MaskConfig(mask_type="sliding_window", seq_len=128, window_size=32)
    >>> mask = create_mask(config=config)
    >>>
    >>> # Apply mask to attention logits
    >>> masked_logits = apply_mask(logits, mask)
"""

import keras
from keras import ops
import numpy as np
from typing import Optional, Union, Tuple, List, Literal, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Local imports
from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------
# Type Definitions and Enums
# ---------------------------------------------------------------------

class MaskType(str, Enum):
    """Enumeration of available mask types."""
    # Attention masks
    CAUSAL = "causal"
    SLIDING_WINDOW = "sliding_window"
    GLOBAL_LOCAL = "global_local"
    BLOCK_DIAGONAL = "block_diagonal"
    RANDOM = "random"
    BANDED = "banded"
    PADDING = "padding"

    # Segmentation masks
    VALID_QUERY = "valid_query"
    SPATIAL = "spatial"
    QUERY_INTERACTION = "query_interaction"
    INSTANCE_SEPARATION = "instance_separation"


@dataclass
class MaskConfig:
    """
    Configuration for mask creation.

    This dataclass provides a unified configuration interface for all mask types,
    allowing for flexible and extensible mask creation patterns.

    Attributes:
        mask_type: Type of mask to create (from MaskType enum or string).
        dtype: Data type for the mask. Defaults to "bool".

        # Sequence parameters
        seq_len: Sequence length for attention masks.
        batch_size: Optional batch size for batched mask creation.

        # Window/block parameters
        window_size: Window size for sliding window masks.
        block_size: Block size for block diagonal masks.
        band_width: Band width for banded masks.
        sliding_window: Window size for local attention in global_local masks.

        # Random mask parameters
        mask_probability: Probability for random masking.
        seed: Random seed for reproducibility.

        # Spatial parameters (for segmentation)
        height: Height dimension for spatial masks.
        width: Width dimension for spatial masks.
        num_queries: Number of object queries for segmentation.

        # Advanced parameters
        valid_queries: Tensor indicating valid queries for segmentation.
        attention_regions: Regions of interest for spatial attention.
        mask_mode: Mode for certain mask types ("inside"/"outside" for spatial).
        interaction_type: Type of query interactions for segmentation.
        hierarchy_levels: Hierarchy levels for hierarchical query interactions.
        separation_threshold: Threshold for instance separation masks.

        # Additional parameters
        extra_params: Dictionary for any additional mask-specific parameters.
    """
    mask_type: Union[MaskType, str]
    dtype: str = "bool"

    # Sequence parameters
    seq_len: Optional[int] = None
    batch_size: Optional[int] = None

    # Window/block parameters
    window_size: Optional[int] = None
    block_size: Optional[int] = None
    band_width: Optional[int] = None
    sliding_window: Optional[int] = None

    # Random mask parameters
    mask_probability: float = 0.1
    seed: Optional[int] = None

    # Spatial parameters
    height: Optional[int] = None
    width: Optional[int] = None
    num_queries: Optional[int] = None

    # Advanced parameters
    valid_queries: Optional[keras.KerasTensor] = None
    attention_regions: Optional[keras.KerasTensor] = None
    mask_mode: Literal["inside", "outside"] = "inside"
    interaction_type: Literal["none", "self", "cross", "hierarchical"] = "self"
    hierarchy_levels: Optional[keras.KerasTensor] = None
    separation_threshold: float = 0.5

    # Additional parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Convert string to MaskType if necessary
        if isinstance(self.mask_type, str):
            try:
                self.mask_type = MaskType(self.mask_type)
            except ValueError:
                valid_types = [t.value for t in MaskType]
                raise ValueError(
                    f"Invalid mask_type: {self.mask_type}. "
                    f"Must be one of {valid_types}"
                )


# ---------------------------------------------------------------------
# Core Mask Creation Functions
# ---------------------------------------------------------------------

class MaskFactory:
    """
    Factory class for creating various types of masks.

    This class provides a centralized interface for creating all mask types,
    with consistent error handling and logging.
    """

    @staticmethod
    def create_causal_mask(seq_len: int, dtype: str = "bool") -> keras.KerasTensor:
        """
        Create a causal (lower triangular) attention mask.

        Args:
            seq_len: Length of the sequence.
            dtype: Data type for the mask.

        Returns:
            keras.KerasTensor: Shape (seq_len, seq_len) where True indicates
                positions that should be masked (blocked).

        Examples:
            >>> mask = MaskFactory.create_causal_mask(4)
            >>> # Returns:
            >>> # [[False, True,  True,  True ],
            >>> #  [False, False, True,  True ],
            >>> #  [False, False, False, True ],
            >>> #  [False, False, False, False]]
        """
        i = ops.arange(seq_len)[:, None]
        j = ops.arange(seq_len)
        mask = j > i

        if dtype != "bool":
            mask = ops.cast(mask, dtype)

        logger.debug(f"Created causal mask with shape ({seq_len}, {seq_len})")
        return mask

    @staticmethod
    def create_sliding_window_mask(
            seq_len: int,
            window_size: int,
            dtype: str = "bool"
    ) -> keras.KerasTensor:
        """
        Create a sliding window attention mask with causal constraint.

        Each position can attend to at most `window_size` previous positions
        (including itself) while maintaining causality.

        Args:
            seq_len: Length of the sequence.
            window_size: Size of the attention window.
            dtype: Data type for the mask.

        Returns:
            keras.KerasTensor: Shape (seq_len, seq_len) where True indicates
                masked positions.

        Raises:
            ValueError: If window_size is not positive.
        """
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")

        i = ops.arange(seq_len)[:, None]
        j = ops.arange(seq_len)

        # Causal constraint: can't attend to future
        causal_mask = j > i
        # Window constraint: can't attend too far in the past
        window_mask = (i - j) >= window_size

        mask = ops.logical_or(causal_mask, window_mask)

        if dtype != "bool":
            mask = ops.cast(mask, dtype)

        logger.debug(
            f"Created sliding window mask: shape=({seq_len}, {seq_len}), "
            f"window_size={window_size}"
        )
        return mask

    @staticmethod
    def create_global_local_masks(
            seq_len: int,
            sliding_window: int,
            dtype: str = "bool"
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Create masks for combined global and local attention patterns.

        Args:
            seq_len: Sequence length.
            sliding_window: Size of the sliding window for local attention.
            dtype: Data type for the masks.

        Returns:
            Tuple[keras.KerasTensor, keras.KerasTensor]:
                - global_mask: Causal mask for global attention
                - local_mask: Sliding window mask for local attention
                Both of shape (seq_len, seq_len).

        Raises:
            ValueError: If sliding_window is not positive.
        """
        if sliding_window <= 0:
            raise ValueError(f"sliding_window must be positive, got {sliding_window}")

        # Global mask is just causal
        global_mask = MaskFactory.create_causal_mask(seq_len, dtype)

        # Local mask combines causal and window constraints
        local_mask = MaskFactory.create_sliding_window_mask(
            seq_len, sliding_window, dtype
        )

        logger.debug(
            f"Created global and local masks: shape=({seq_len}, {seq_len}), "
            f"window={sliding_window}"
        )
        return global_mask, local_mask

    @staticmethod
    def create_block_diagonal_mask(
            seq_len: int,
            block_size: int,
            dtype: str = "bool"
    ) -> keras.KerasTensor:
        """
        Create a block diagonal attention mask.

        Partitions the sequence into non-overlapping blocks where attention
        is only allowed within each block.

        Args:
            seq_len: Length of the sequence.
            block_size: Size of each attention block.
            dtype: Data type for the mask.

        Returns:
            keras.KerasTensor: Shape (seq_len, seq_len) where True indicates
                masked positions.

        Raises:
            ValueError: If block_size is not positive.
        """
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")

        positions = ops.arange(seq_len)
        block_ids_i = positions[:, None] // block_size
        block_ids_j = positions // block_size

        mask = block_ids_i != block_ids_j

        if dtype != "bool":
            mask = ops.cast(mask, dtype)

        logger.debug(
            f"Created block diagonal mask: shape=({seq_len}, {seq_len}), "
            f"block_size={block_size}"
        )
        return mask

    @staticmethod
    def create_random_mask(
            seq_len: int,
            mask_probability: float,
            seed: Optional[int] = None,
            dtype: str = "bool"
    ) -> keras.KerasTensor:
        """
        Create a random attention mask.

        Args:
            seq_len: Length of the sequence.
            mask_probability: Probability of masking each position pair.
            seed: Random seed for reproducibility.
            dtype: Data type for the mask.

        Returns:
            keras.KerasTensor: Shape (seq_len, seq_len) with random masking.

        Raises:
            ValueError: If mask_probability is not in [0, 1].
        """
        if not 0.0 <= mask_probability <= 1.0:
            raise ValueError(
                f"mask_probability must be in [0, 1], got {mask_probability}"
            )

        if seed is not None:
            keras.utils.set_random_seed(seed)

        random_values = keras.random.uniform((seq_len, seq_len))
        mask = random_values < mask_probability

        if dtype != "bool":
            mask = ops.cast(mask, dtype)

        logger.debug(
            f"Created random mask: shape=({seq_len}, {seq_len}), "
            f"p={mask_probability}"
        )
        return mask

    @staticmethod
    def create_banded_mask(
            seq_len: int,
            band_width: int,
            dtype: str = "bool"
    ) -> keras.KerasTensor:
        """
        Create a banded attention mask.

        Allows attention only within a band around the diagonal.

        Args:
            seq_len: Length of the sequence.
            band_width: Total width of the attention band.
            dtype: Data type for the mask.

        Returns:
            keras.KerasTensor: Shape (seq_len, seq_len) where True indicates
                masked positions.

        Raises:
            ValueError: If band_width is not positive.
        """
        if band_width <= 0:
            raise ValueError(f"band_width must be positive, got {band_width}")

        i = ops.arange(seq_len)[:, None]
        j = ops.arange(seq_len)

        distance = ops.abs(i - j)
        half_width = band_width // 2
        mask = distance > half_width

        if dtype != "bool":
            mask = ops.cast(mask, dtype)

        logger.debug(
            f"Created banded mask: shape=({seq_len}, {seq_len}), "
            f"band_width={band_width}"
        )
        return mask

    @staticmethod
    def create_padding_mask(
            padding_mask: keras.KerasTensor,
            dtype: str = "bool"
    ) -> keras.KerasTensor:
        """
        Create 2D attention mask from 1D padding mask.

        Args:
            padding_mask: Boolean tensor of shape (batch_size, seq_len)
                where True indicates padding positions.
            dtype: Data type for the output mask.

        Returns:
            keras.KerasTensor: Shape (batch_size, seq_len, seq_len) where
                True indicates positions that should be masked.
        """
        mask_i = padding_mask[:, :, None]
        mask_j = padding_mask[:, None, :]
        attention_mask = ops.logical_or(mask_i, mask_j)

        if dtype != "bool":
            attention_mask = ops.cast(attention_mask, dtype)

        logger.debug(f"Created attention mask from padding mask")
        return attention_mask

    # Segmentation-specific masks

    @staticmethod
    def create_valid_query_mask(
            num_queries: int,
            valid_queries: Optional[keras.KerasTensor] = None,
            dtype: str = "bool"
    ) -> keras.KerasTensor:
        """
        Create a mask for valid object queries in segmentation.

        Args:
            num_queries: Total number of queries.
            valid_queries: Optional tensor indicating valid queries:
                - Scalar: number of valid queries
                - Boolean tensor of shape (num_queries,): validity mask
                - None: all queries are valid
            dtype: Data type for the mask.

        Returns:
            keras.KerasTensor: Shape (num_queries,) where True indicates
                masked (invalid) queries.
        """
        if valid_queries is None:
            mask = ops.zeros((num_queries,), dtype="bool")
        elif ops.ndim(valid_queries) == 0:
            valid_count = ops.cast(valid_queries, "int32")
            indices = ops.arange(num_queries)
            mask = indices >= valid_count
        else:
            mask = ops.logical_not(ops.cast(valid_queries, "bool"))

        if dtype != "bool":
            mask = ops.cast(mask, dtype)

        logger.debug(f"Created valid query mask for {num_queries} queries")
        return mask

    @staticmethod
    def create_spatial_mask(
            height: int,
            width: int,
            attention_regions: Optional[keras.KerasTensor] = None,
            mask_mode: Literal["inside", "outside"] = "inside",
            dtype: str = "bool"
    ) -> keras.KerasTensor:
        """
        Create spatial attention mask for image regions.

        Args:
            height: Height of the spatial dimension.
            width: Width of the spatial dimension.
            attention_regions: Optional regions of interest:
                - Binary mask of shape (height, width)
                - None for no masking
            mask_mode: Whether to mask "inside" or "outside" the regions.
            dtype: Data type for the mask.

        Returns:
            keras.KerasTensor: Shape (height, width) where True indicates
                masked positions.
        """
        if attention_regions is None:
            mask = ops.zeros((height, width), dtype="bool")
        else:
            valid_area = ops.cast(attention_regions, "bool")
            mask = valid_area if mask_mode == "inside" else ops.logical_not(valid_area)

        if dtype != "bool":
            mask = ops.cast(mask, dtype)

        logger.debug(
            f"Created spatial mask: shape=({height}, {width}), mode={mask_mode}"
        )
        return mask

    @staticmethod
    def create_query_interaction_mask(
            num_queries: int,
            interaction_type: Literal["none", "self", "cross", "hierarchical"] = "self",
            hierarchy_levels: Optional[keras.KerasTensor] = None,
            dtype: str = "bool"
    ) -> keras.KerasTensor:
        """
        Create masks for controlling interactions between queries.

        Args:
            num_queries: Number of queries.
            interaction_type: Type of interaction pattern.
            hierarchy_levels: Optional hierarchy levels for hierarchical type.
            dtype: Data type for the mask.

        Returns:
            keras.KerasTensor: Shape (num_queries, num_queries) where True
                indicates blocked interactions.

        Raises:
            ValueError: If hierarchical type is used without hierarchy_levels.
        """
        i = ops.arange(num_queries)[:, None]
        j = ops.arange(num_queries)

        if interaction_type == "none":
            mask = i != j  # Only self-interaction
        elif interaction_type == "self":
            mask = ops.zeros((num_queries, num_queries), dtype="bool")
        elif interaction_type == "cross":
            mask = i == j  # Only cross-interaction
        elif interaction_type == "hierarchical":
            if hierarchy_levels is None:
                raise ValueError(
                    "hierarchy_levels required for hierarchical interaction"
                )
            levels_i = hierarchy_levels[:, None]
            levels_j = hierarchy_levels[None, :]
            level_diff = ops.abs(levels_i - levels_j)
            mask = level_diff > 1
        else:
            mask = ops.zeros((num_queries, num_queries), dtype="bool")

        if dtype != "bool":
            mask = ops.cast(mask, dtype)

        logger.debug(f"Created query interaction mask: type={interaction_type}")
        return mask

    @staticmethod
    def create_instance_separation_mask(
            mask_predictions: keras.KerasTensor,
            separation_threshold: float = 0.5,
            dtype: str = "bool"
    ) -> keras.KerasTensor:
        """
        Create masks to enforce separation between instance predictions.

        Args:
            mask_predictions: Predicted masks of shape
                (batch, num_queries, height, width).
            separation_threshold: Threshold for determining boundaries.
            dtype: Data type for the output mask.

        Returns:
            keras.KerasTensor: Shape (batch, num_queries, height, width)
                where True indicates positions to suppress.
        """
        max_preds = ops.max(mask_predictions, axis=1, keepdims=True)
        is_max = mask_predictions >= (max_preds - 1e-6)

        should_suppress = ops.logical_and(
            ops.logical_not(is_max),
            max_preds > separation_threshold
        )

        if dtype != "bool":
            should_suppress = ops.cast(should_suppress, dtype)

        logger.debug("Created instance separation mask")
        return should_suppress


# ---------------------------------------------------------------------
# Main Interface Functions
# ---------------------------------------------------------------------

def create_mask(
        mask_type: Optional[Union[MaskType, str]] = None,
        config: Optional[MaskConfig] = None,
        **kwargs
) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, ...]]:
    """
    Universal interface for creating masks.

    This function provides a single entry point for creating any type of mask
    supported by the module. Parameters can be provided either through a
    MaskConfig object or as keyword arguments.

    Args:
        mask_type: Type of mask to create (if not using config).
        config: MaskConfig object with all parameters.
        **kwargs: Additional parameters to override or supplement config.

    Returns:
        Union[keras.KerasTensor, Tuple[keras.KerasTensor, ...]]:
            The created mask(s). Most masks return a single tensor, but
            some (like global_local) return a tuple.

    Raises:
        ValueError: If mask_type is invalid or required parameters are missing.

    Examples:
        >>> # Using mask_type and kwargs
        >>> mask = create_mask("causal", seq_len=128)
        >>>
        >>> # Using MaskConfig
        >>> config = MaskConfig(mask_type="sliding_window", seq_len=128, window_size=32)
        >>> mask = create_mask(config=config)
        >>>
        >>> # Combining config and kwargs (kwargs override config)
        >>> mask = create_mask(config=config, window_size=64)
    """
    # Merge configuration sources
    if config is None:
        if mask_type is None:
            raise ValueError("Either mask_type or config must be provided")
        config = MaskConfig(mask_type=mask_type, **kwargs)
    else:
        # Override config with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                config.extra_params[key] = value

    # Dispatch to appropriate factory method
    factory = MaskFactory()
    mask_type = config.mask_type if isinstance(config.mask_type, MaskType) else MaskType(config.mask_type)

    if mask_type == MaskType.CAUSAL:
        if config.seq_len is None:
            raise ValueError("seq_len required for causal mask")
        return factory.create_causal_mask(config.seq_len, config.dtype)

    elif mask_type == MaskType.SLIDING_WINDOW:
        if config.seq_len is None or config.window_size is None:
            raise ValueError("seq_len and window_size required for sliding window mask")
        return factory.create_sliding_window_mask(
            config.seq_len, config.window_size, config.dtype
        )

    elif mask_type == MaskType.GLOBAL_LOCAL:
        if config.seq_len is None or config.sliding_window is None:
            raise ValueError("seq_len and sliding_window required for global_local masks")
        return factory.create_global_local_masks(
            config.seq_len, config.sliding_window, config.dtype
        )

    elif mask_type == MaskType.BLOCK_DIAGONAL:
        if config.seq_len is None or config.block_size is None:
            raise ValueError("seq_len and block_size required for block diagonal mask")
        return factory.create_block_diagonal_mask(
            config.seq_len, config.block_size, config.dtype
        )

    elif mask_type == MaskType.RANDOM:
        if config.seq_len is None:
            raise ValueError("seq_len required for random mask")
        return factory.create_random_mask(
            config.seq_len, config.mask_probability, config.seed, config.dtype
        )

    elif mask_type == MaskType.BANDED:
        if config.seq_len is None or config.band_width is None:
            raise ValueError("seq_len and band_width required for banded mask")
        return factory.create_banded_mask(
            config.seq_len, config.band_width, config.dtype
        )

    elif mask_type == MaskType.PADDING:
        padding_mask = config.extra_params.get("padding_mask")
        if padding_mask is None:
            raise ValueError("padding_mask required in extra_params for padding mask")
        return factory.create_padding_mask(padding_mask, config.dtype)

    elif mask_type == MaskType.VALID_QUERY:
        if config.num_queries is None:
            raise ValueError("num_queries required for valid query mask")
        return factory.create_valid_query_mask(
            config.num_queries, config.valid_queries, config.dtype
        )

    elif mask_type == MaskType.SPATIAL:
        if config.height is None or config.width is None:
            raise ValueError("height and width required for spatial mask")
        return factory.create_spatial_mask(
            config.height, config.width, config.attention_regions,
            config.mask_mode, config.dtype
        )

    elif mask_type == MaskType.QUERY_INTERACTION:
        if config.num_queries is None:
            raise ValueError("num_queries required for query interaction mask")
        return factory.create_query_interaction_mask(
            config.num_queries, config.interaction_type,
            config.hierarchy_levels, config.dtype
        )

    elif mask_type == MaskType.INSTANCE_SEPARATION:
        mask_predictions = config.extra_params.get("mask_predictions")
        if mask_predictions is None:
            raise ValueError(
                "mask_predictions required in extra_params for instance separation"
            )
        return factory.create_instance_separation_mask(
            mask_predictions, config.separation_threshold, config.dtype
        )

    else:
        raise ValueError(f"Unknown mask type: {mask_type}")


def apply_mask(
        inputs: keras.KerasTensor,
        mask: keras.KerasTensor,
        mask_value: float = -1e9,
        mask_type: Optional[Literal["attention", "segmentation"]] = None
) -> keras.KerasTensor:
    """
    Apply a mask to inputs (attention logits or segmentation predictions).

    This function provides a universal interface for applying masks, automatically
    handling different input shapes and mask types.

    Args:
        inputs: Input tensor to be masked.
        mask: Boolean mask tensor where True indicates positions to mask.
        mask_value: Value to use for masked positions (for logits).
        mask_type: Optional hint about the mask type for better handling.

    Returns:
        keras.KerasTensor: Masked inputs with the same shape as input.

    Examples:
        >>> # Apply attention mask to logits
        >>> logits = keras.random.normal((2, 8, 128, 128))  # (batch, heads, seq, seq)
        >>> mask = create_mask("causal", seq_len=128)
        >>> masked_logits = apply_mask(logits, mask, mask_type="attention")
        >>>
        >>> # Apply segmentation mask
        >>> predictions = keras.random.uniform((2, 10, 64, 64))  # (batch, queries, H, W)
        >>> mask = create_mask("valid_query", num_queries=10, valid_queries=5)
        >>> masked_preds = apply_mask(predictions, mask, mask_type="segmentation")
    """
    # Ensure mask is boolean
    mask = ops.cast(mask, "bool")

    # Broadcast mask to match input shape if necessary
    input_shape = ops.shape(inputs)
    mask_shape = ops.shape(mask)

    # Handle different broadcasting scenarios
    if len(mask_shape) < len(input_shape):
        # Determine how to broadcast based on mask type hint and shapes
        if mask_type == "attention":
            # Typical attention: inputs are (batch, heads, seq, seq) or (batch, seq, seq)
            # Mask is usually (seq, seq)
            if len(mask_shape) == 2:
                # Add batch and possibly head dimensions
                for _ in range(len(input_shape) - 2):
                    mask = ops.expand_dims(mask, axis=0)
        elif mask_type == "segmentation":
            # Handle segmentation-specific broadcasting
            # This is handled case-by-case based on mask dimensions
            pass

    # Apply mask
    masked_inputs = ops.where(mask, mask_value, inputs)

    return masked_inputs


def combine_masks(
        *masks: keras.KerasTensor,
        combination: Literal["and", "or", "xor"] = "or"
) -> keras.KerasTensor:
    """
    Combine multiple masks using logical operations.

    Args:
        *masks: Variable number of mask tensors to combine.
        combination: Logical operation to use for combination.

    Returns:
        keras.KerasTensor: Combined mask.

    Raises:
        ValueError: If no masks provided or invalid combination type.

    Examples:
        >>> causal = create_mask("causal", seq_len=128)
        >>> window = create_mask("sliding_window", seq_len=128, window_size=32)
        >>> combined = combine_masks(causal, window, combination="or")
    """
    if not masks:
        raise ValueError("At least one mask must be provided")

    if len(masks) == 1:
        return masks[0]

    # Ensure all masks are boolean
    masks = [ops.cast(mask, "bool") for mask in masks]

    result = masks[0]
    for mask in masks[1:]:
        if combination == "or":
            result = ops.logical_or(result, mask)
        elif combination == "and":
            result = ops.logical_and(result, mask)
        elif combination == "xor":
            result = ops.logical_xor(result, mask)
        else:
            raise ValueError(f"Invalid combination type: {combination}")

    logger.debug(f"Combined {len(masks)} masks using {combination}")
    return result


def visualize_mask(
        mask: Union[keras.KerasTensor, np.ndarray],
        title: str = "Mask Visualization",
        figsize: Tuple[int, int] = (8, 6),
        cmap: str = "RdBu_r",
        save_path: Optional[str] = None,
        show: bool = True
) -> Optional[plt.Figure]:
    """
    Visualize a mask using matplotlib.

    Args:
        mask: Mask tensor to visualize (2D or 3D).
        title: Title for the plot.
        figsize: Figure size as (width, height).
        cmap: Colormap to use.
        save_path: Optional path to save the figure.
        show: Whether to display the plot.

    Returns:
        Optional[plt.Figure]: Figure object if matplotlib is available.

    Examples:
        >>> mask = create_mask("causal", seq_len=32)
        >>> visualize_mask(mask, title="Causal Attention Mask")
    """
    if not HAS_MATPLOTLIB:
        warnings.warn("matplotlib not available for visualization")
        return None

    # Convert to numpy if necessary
    if hasattr(mask, 'numpy'):
        mask_np = mask.numpy()
    elif isinstance(mask, keras.KerasTensor):
        mask_np = ops.convert_to_numpy(mask)
    else:
        mask_np = np.array(mask)

    # Handle 3D masks by taking first element
    if mask_np.ndim == 3:
        mask_np = mask_np[0]
        title = f"{title} (first batch element)"
    elif mask_np.ndim != 2:
        warnings.warn(f"Can only visualize 2D or 3D masks, got shape {mask_np.shape}")
        return None

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot mask
    im = ax.imshow(mask_np.astype(float), cmap=cmap, interpolation='nearest', aspect='auto')

    # Formatting
    ax.set_title(title)
    ax.set_xlabel('Key/Column Position')
    ax.set_ylabel('Query/Row Position')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Masked (1) / Allowed (0)')

    # Add grid for better readability (if mask is small enough)
    if mask_np.shape[0] <= 32:
        ax.set_xticks(np.arange(mask_np.shape[1]))
        ax.set_yticks(np.arange(mask_np.shape[0]))
        ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved mask visualization to {save_path}")

    # Show if requested
    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------

def get_mask_info() -> Dict[str, Dict[str, Any]]:
    """
    Get information about all available mask types.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mapping mask types to their
            descriptions and required parameters.

    Examples:
        >>> info = get_mask_info()
        >>> print(info["causal"]["description"])
        >>> print(info["causal"]["required_params"])
    """
    return {
        "causal": {
            "description": "Lower triangular mask preventing future attention",
            "required_params": ["seq_len"],
            "optional_params": ["dtype"],
        },
        "sliding_window": {
            "description": "Causal mask with limited attention window",
            "required_params": ["seq_len", "window_size"],
            "optional_params": ["dtype"],
        },
        "global_local": {
            "description": "Dual masks for global and local attention patterns",
            "required_params": ["seq_len", "sliding_window"],
            "optional_params": ["dtype"],
            "returns": "Tuple of (global_mask, local_mask)",
        },
        "block_diagonal": {
            "description": "Block-wise attention within non-overlapping segments",
            "required_params": ["seq_len", "block_size"],
            "optional_params": ["dtype"],
        },
        "random": {
            "description": "Random masking with specified probability",
            "required_params": ["seq_len"],
            "optional_params": ["mask_probability", "seed", "dtype"],
        },
        "banded": {
            "description": "Band around diagonal for local attention",
            "required_params": ["seq_len", "band_width"],
            "optional_params": ["dtype"],
        },
        "padding": {
            "description": "Attention mask from padding positions",
            "required_params": ["padding_mask"],
            "optional_params": ["dtype"],
        },
        "valid_query": {
            "description": "Mask for valid object queries in segmentation",
            "required_params": ["num_queries"],
            "optional_params": ["valid_queries", "dtype"],
        },
        "spatial": {
            "description": "Spatial attention mask for image regions",
            "required_params": ["height", "width"],
            "optional_params": ["attention_regions", "mask_mode", "dtype"],
        },
        "query_interaction": {
            "description": "Control interactions between object queries",
            "required_params": ["num_queries"],
            "optional_params": ["interaction_type", "hierarchy_levels", "dtype"],
        },
        "instance_separation": {
            "description": "Enforce separation between instance predictions",
            "required_params": ["mask_predictions"],
            "optional_params": ["separation_threshold", "dtype"],
        },
    }


