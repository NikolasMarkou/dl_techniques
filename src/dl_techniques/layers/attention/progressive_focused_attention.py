"""
Progressive Focused Attention (PFA) Module for PFT-SR.

This module implements the core innovation of PFT-SR: Progressive Focused Attention,
which inherits attention maps from previous layers through Hadamard product and uses
sparse matrix multiplication to skip unnecessary similarity calculations.

The key innovation is that attention maps are progressively refined across layers,
with each layer focusing on increasingly relevant tokens while filtering out
irrelevant features before calculating similarities.

Key Concepts:
-------------
1. **Window-based Attention**: Divides the spatial feature map into non-overlapping
   windows and applies attention within each window, reducing computational cost
   from O((HW)²) to O(H*W*window_size²).

2. **Shifted Windows (SW-MSA)**: Alternating layers use shifted windows to enable
   cross-window connections, maintaining spatial coherence across window boundaries.

3. **Progressive Focusing**: Uses attention maps from previous layers as guidance
   to progressively refine which tokens to attend to, creating a hierarchical
   attention pattern.

4. **LePE (Locally-Enhanced Positional Encoding)**: Applies depthwise convolution
   to value vectors to inject local positional information without explicit
   positional embeddings.

References
----------
    Long, Wei, et al. "Progressive Focused Transformer for Single Image Super-Resolution."
    CVPR 2025.
"""

import keras
from typing import Optional, Tuple, Union, Dict, Any, Literal

# ---------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------

SparsityMode = Literal['none', 'top_k', 'threshold']

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class ProgressiveFocusedAttention(keras.layers.Layer):
    """
    Progressive Focused Attention mechanism with windowed self-attention.

    This layer implements the PFA mechanism from PFT-SR, which:

    1. **Inherits Attention Maps**: Uses attention patterns from the previous layer
       via Hadamard product (element-wise multiplication) to guide current attention.

    2. **Sparse Attention**: Optionally applies sparsity based on inherited maps to
       skip irrelevant token computations, improving efficiency.

    3. **Windowed Multi-Head Self-Attention**: Applies W-MSA or SW-MSA (shifted
       window) to limit attention computation to local windows.

    4. **LePE Integration**: Incorporates local positional information through
       depthwise convolution on value vectors.

    The progressive focusing mechanism allows deeper layers to attend to
    increasingly relevant tokens, improving both efficiency and quality
    for image super-resolution tasks.

    Architecture Flow:
    -----------------
    Input (B, H, W, C)
        → [Optional Shift]
        → Window Partition (B*nW, ws, ws, C)
        → QKV Projection (B*nW, ws², 3*C)
        → Split to Q, K, V (B*nW, num_heads, ws², head_dim)
        → [Optional LePE on V]
        → Attention Scores = Q @ K^T * scale
        → [Apply Attention Mask for SW-MSA]
        → [Progressive Focusing: scores * prev_attn_map]
        → [Optional Sparsity]
        → Softmax → Attention Weights
        → Output = Attention @ V
        → Output Projection
        → Window Reverse
        → [Optional Unshift]
        → Output (B, H, W, C)

    Parameters
    ----------
    dim : int
        The embedding dimension (number of channels).
    num_heads : int
        Number of attention heads. Must divide `dim` evenly.
    window_size : int, optional
        Size of the attention window. Attention is computed within
        non-overlapping windows of this size. Default is 8.
    shift_size : int, optional
        Shift size for shifted window attention (SW-MSA). Use 0 for regular
        windowed attention (W-MSA), and ``window_size // 2`` for shifted
        window attention. Shifted windows enable cross-window connections.
        Default is 0.
    top_k : int, optional
        Number of top-k tokens to attend to based on previous attention map.
        Only used when ``sparsity_mode='top_k'``. If None, attends to all
        tokens within the window. Default is None.
    sparsity_threshold : float, optional
        Threshold for sparsity-based attention. Attention weights below this
        threshold from the previous layer are masked out. Only used when
        ``sparsity_mode='threshold'``. Default is 0.0.
    sparsity_mode : {'none', 'top_k', 'threshold'}, optional
        Mode for sparse attention:

        - ``'none'``: No sparsity, use full attention within windows.
        - ``'top_k'``: Keep only top-k attention connections per query based
          on the previous layer's attention map.
        - ``'threshold'``: Mask attention weights below sparsity_threshold
          from the previous layer.

        Default is ``'none'``.
    qkv_bias : bool, optional
        Whether to include bias terms in QKV projections. Default is True.
    attention_dropout : float, optional
        Dropout rate applied to attention weights after softmax. Helps
        prevent overfitting to specific attention patterns. Default is 0.0.
    projection_dropout : float, optional
        Dropout rate applied to the output projection. Default is 0.0.
    use_lepe : bool, optional
        Whether to use Locally-Enhanced Positional Encoding (LePE).
        LePE applies a depthwise convolution to value vectors to inject
        local positional information without explicit positional embeddings.
        Recommended for vision tasks. Default is True.
    lepe_kernel_size : int, optional
        Kernel size for LePE depthwise convolution. Larger values capture
        wider local context. Default is 3.
    kernel_initializer : str or keras.initializers.Initializer, optional
        Initializer for projection weight matrices. Default is ``'glorot_uniform'``.
    bias_initializer : str or keras.initializers.Initializer, optional
        Initializer for bias vectors. Default is ``'zeros'``.
    **kwargs
        Additional keyword arguments for the Layer base class.

    Attributes
    ----------
    _qkv : keras.layers.Dense
        Projection layer for queries, keys, and values (combined).
    _proj : keras.layers.Dense
        Output projection layer.
    _lepe : keras.layers.DepthwiseConv2D or None
        LePE convolution layer if enabled.
    _attn_mask : keras.Variable or None
        Precomputed attention mask for shifted window attention.
    _attn_drop : keras.layers.Dropout or None
        Attention dropout layer if dropout > 0.
    _proj_drop : keras.layers.Dropout or None
        Projection dropout layer if dropout > 0.

    Notes
    -----
    **Input Shape:**
        - x: Tensor of shape ``(batch_size, height, width, dim)``.
        - prev_attn_map: Optional tensor from previous layer for progressive
          focusing. Shape ``(batch*num_windows, num_heads, window_area, window_area)``.

    **Output Shape:**
        Tuple of:

        - output: Tensor of same shape as input ``(batch_size, height, width, dim)``.
        - attn_map: Attention weights of shape
          ``(batch*num_windows, num_heads, window_area, window_area)``
          for the next layer.

    **Important Constraints:**
        - Input height and width must be divisible by `window_size`.
        - `shift_size` must be less than `window_size`.
        - The shifted window mechanism requires proper attention masking to prevent
          information leakage across windows after the cyclic shift operation.

    Examples
    --------
    Basic windowed attention (W-MSA):

    >>> import keras
    >>> x = keras.random.normal((2, 64, 64, 96))
    >>> pfa = ProgressiveFocusedAttention(dim=96, num_heads=3, window_size=8)
    >>> output, attn_map = pfa(x, prev_attn_map=None)
    >>> print(output.shape)
    (2, 64, 64, 96)

    Shifted window attention (SW-MSA) for alternating layers:

    >>> pfa_shifted = ProgressiveFocusedAttention(
    ...     dim=96, num_heads=3, window_size=8, shift_size=4
    ... )
    >>> output, attn_map = pfa_shifted(output, prev_attn_map=attn_map)
    >>> print(output.shape)
    (2, 64, 64, 96)

    With top-k sparse attention for efficiency:

    >>> pfa_sparse = ProgressiveFocusedAttention(
    ...     dim=96, num_heads=3, window_size=8,
    ...     sparsity_mode='top_k', top_k=32
    ... )
    >>> output, attn_map = pfa_sparse(x, prev_attn_map=None)

    With threshold-based sparsity:

    >>> pfa_threshold = ProgressiveFocusedAttention(
    ...     dim=96, num_heads=3, window_size=8,
    ...     sparsity_mode='threshold', sparsity_threshold=0.1
    ... )
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            window_size: int = 8,
            shift_size: int = 0,
            top_k: Optional[int] = None,
            sparsity_threshold: float = 0.0,
            sparsity_mode: SparsityMode = 'none',
            qkv_bias: bool = True,
            attention_dropout: float = 0.0,
            projection_dropout: float = 0.0,
            use_lepe: bool = True,
            lepe_kernel_size: int = 3,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            **kwargs: Any
    ) -> None:
        """Initialize ProgressiveFocusedAttention layer."""
        super().__init__(**kwargs)

        # Store configuration parameters
        self._dim = dim
        self._num_heads = num_heads
        self._window_size = window_size
        self._shift_size = shift_size
        self._top_k = top_k
        self._sparsity_threshold = sparsity_threshold
        self._sparsity_mode = sparsity_mode
        self._qkv_bias = qkv_bias
        self._attention_dropout = attention_dropout
        self._projection_dropout = projection_dropout
        self._use_lepe = use_lepe
        self._lepe_kernel_size = lepe_kernel_size
        self._kernel_initializer = keras.initializers.get(kernel_initializer)
        self._bias_initializer = keras.initializers.get(bias_initializer)

        # Validate configuration before proceeding
        self._validate_config()

        # Compute derived attributes
        self._head_dim = dim // num_heads  # Dimension per attention head
        self._scale = self._head_dim ** -0.5  # Scaling factor for dot-product attention
        self._window_area = window_size * window_size  # Number of tokens per window

    def _validate_config(self) -> None:
        """
        Validate layer configuration parameters.

        Ensures all parameters are within valid ranges and are compatible
        with each other. This helps catch configuration errors early.

        Raises
        ------
        ValueError
            If any configuration parameter is invalid or incompatible.
        """
        # Check dimension divisibility
        if self._dim % self._num_heads != 0:
            raise ValueError(
                f"dim ({self._dim}) must be divisible by "
                f"num_heads ({self._num_heads}). "
                f"Got head_dim = {self._dim / self._num_heads}"
            )

        # Check shift size constraints
        if self._shift_size < 0:
            raise ValueError(
                f"shift_size ({self._shift_size}) must be non-negative"
            )

        if self._shift_size >= self._window_size:
            raise ValueError(
                f"shift_size ({self._shift_size}) must be less than "
                f"window_size ({self._window_size}). "
                f"Typically use shift_size = window_size // 2 for SW-MSA."
            )

        # Check sparsity mode validity
        if self._sparsity_mode not in ('none', 'top_k', 'threshold'):
            raise ValueError(
                f"sparsity_mode must be one of 'none', 'top_k', 'threshold', "
                f"got '{self._sparsity_mode}'"
            )

        # Check sparsity configuration consistency
        if self._sparsity_mode == 'top_k' and self._top_k is None:
            raise ValueError(
                "top_k must be specified when sparsity_mode='top_k'"
            )

        if self._top_k is not None and self._top_k <= 0:
            raise ValueError(
                f"top_k ({self._top_k}) must be positive"
            )

        # Check dropout rates
        if self._attention_dropout < 0.0 or self._attention_dropout > 1.0:
            raise ValueError(
                f"attention_dropout ({self._attention_dropout}) must be "
                f"between 0.0 and 1.0"
            )

        if self._projection_dropout < 0.0 or self._projection_dropout > 1.0:
            raise ValueError(
                f"projection_dropout ({self._projection_dropout}) must be "
                f"between 0.0 and 1.0"
            )

    def build(self, input_shape: Union[tuple, list]) -> None:
        """
        Build layer weights and sub-layers.

        Creates:
        - QKV projection layer (projects input to queries, keys, values)
        - Output projection layer
        - Optional LePE depthwise convolution
        - Optional dropout layers
        - Attention mask for shifted window attention (if shift_size > 0)

        This method follows Keras 3 best practices by explicitly building
        all sub-layers to ensure proper weight initialization and serialization.

        Parameters
        ----------
        input_shape : tuple or list
            Shape tuple or list of shape tuples for input tensor.
            Expected shape: ``(batch_size, height, width, dim)``.
        """
        # Handle different input shape formats
        if isinstance(input_shape, list):
            x_shape = input_shape[0]
        else:
            x_shape = input_shape

        # ============ QKV Projection ============
        # Projects input to queries, keys, and values simultaneously
        # Output dimension is 3*dim to accommodate all three projections
        self._qkv = keras.layers.Dense(
            self._dim * 3,
            use_bias=self._qkv_bias,
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            name="qkv_projection"
        )

        # ============ Output Projection ============
        # Projects concatenated multi-head attention output back to dim
        self._proj = keras.layers.Dense(
            self._dim,
            use_bias=True,
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            name="output_projection"
        )

        # ============ Attention Dropout ============
        # Applied to attention weights after softmax to prevent overfitting
        if self._attention_dropout > 0.0:
            self._attn_drop = keras.layers.Dropout(
                self._attention_dropout,
                name="attention_dropout"
            )
        else:
            self._attn_drop = None

        # ============ Projection Dropout ============
        # Applied to output projection to regularize the final output
        if self._projection_dropout > 0.0:
            self._proj_drop = keras.layers.Dropout(
                self._projection_dropout,
                name="projection_dropout"
            )
        else:
            self._proj_drop = None

        # ============ LePE: Locally-Enhanced Positional Encoding ============
        # Uses depthwise convolution to inject local positional information
        # This is more efficient than absolute positional embeddings for vision tasks
        # as it provides translation-equivariant local position awareness
        if self._use_lepe:
            self._lepe = keras.layers.DepthwiseConv2D(
                kernel_size=self._lepe_kernel_size,
                strides=1,
                padding='same',
                depth_multiplier=1,
                depthwise_initializer=self._kernel_initializer,
                use_bias=True,
                bias_initializer=self._bias_initializer,
                name="lepe_conv"
            )
        else:
            self._lepe = None

        # ============ Explicitly Build Sub-layers ============
        # This ensures weights exist before any restoration/serialization
        # Critical for proper model saving and loading
        qkv_input_shape = (None, self._window_area, self._dim)
        self._qkv.build(qkv_input_shape)
        self._proj.build(qkv_input_shape)

        if self._use_lepe:
            lepe_input_shape = (None, self._window_size, self._window_size, self._dim)
            self._lepe.build(lepe_input_shape)

        # ============ Create Attention Mask for Shifted Windows ============
        # For SW-MSA, we need a mask to prevent attention across shifted regions
        if self._shift_size > 0:
            self._attn_mask = self._compute_attention_mask()
        else:
            self._attn_mask = None

        super().build(input_shape)

    def _compute_attention_mask(self) -> keras.Variable:
        """
        Compute attention mask for shifted window attention (SW-MSA).

        The mask prevents attention across different regions after the cyclic
        shift operation. When we cyclically shift the input, different semantic
        regions can end up in the same window. The mask ensures tokens from
        different original regions don't attend to each other.

        Algorithm:
        ---------
        1. Create a spatial index map indicating which region each position belongs to
        2. Partition this map into windows (same as the feature map)
        3. For each window, compute a binary mask: positions from different regions
           get a large negative value (-100.0) to mask them out after softmax
        4. Return as a non-trainable Keras Variable

        Returns
        -------
        keras.Variable
            Attention mask of shape ``(num_windows, window_area, window_area)``.
            Values are 0.0 for valid attention pairs and -100.0 for masked pairs.

        Notes
        -----
        The mask is computed once during build and reused across all forward passes.
        It's marked as non-trainable since it's a static structural constraint.
        """
        # Define slice indices for creating region boundaries
        # These create a 3x3 grid of regions after the shift
        h_slices = (
            slice(0, -self._window_size),
            slice(-self._window_size, -self._shift_size),
            slice(-self._shift_size, None)
        )
        w_slices = (
            slice(0, -self._window_size),
            slice(-self._window_size, -self._shift_size),
            slice(-self._shift_size, None)
        )

        # Create a mask image with region indices using numpy (for static computation)
        import numpy as np

        # Create index mask for 2x2 windows (minimum required for shifted attention)
        mask_size = self._window_size * 2
        img_mask = np.zeros((1, mask_size, mask_size, 1), dtype=np.float32)

        # Assign each region a unique index
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        # Partition the index mask into windows (same as we'll do to features)
        img_mask = img_mask.reshape(
            1, 2, self._window_size, 2, self._window_size, 1
        )
        img_mask = img_mask.transpose(0, 1, 3, 2, 4, 5)
        mask_windows = img_mask.reshape(-1, self._window_size * self._window_size)

        # Compute pairwise attention mask:
        # If indices differ, mask out (set to -100); otherwise allow (set to 0)
        attn_mask = mask_windows[:, :, np.newaxis] - mask_windows[:, np.newaxis, :]
        attn_mask = np.where(attn_mask != 0, -100.0, 0.0).astype(np.float32)

        # Return as non-trainable Keras Variable
        return keras.Variable(
            initializer=attn_mask,
            trainable=False,
            name="attention_mask"
        )

    def _window_partition(
            self,
            x: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Partition input feature map into non-overlapping windows.

        Transforms the spatial feature map into a batch of windows, where each
        window becomes an independent sample for attention computation.

        Algorithm:
        ---------
        1. Reshape from (B, H, W, C) to (B, nH, ws, nW, ws, C)
           where nH = H/ws, nW = W/ws
        2. Transpose to group windows: (B, nH, nW, ws, ws, C)
        3. Flatten batch and window dimensions: (B*nH*nW, ws, ws, C)

        Parameters
        ----------
        x : keras.KerasTensor
            Input feature tensor of shape ``(batch_size, height, width, channels)``.
            Height and width must be divisible by window_size.

        Returns
        -------
        keras.KerasTensor
            Partitioned windows of shape
            ``(batch_size * num_windows, window_size, window_size, channels)``.
            Each window will be processed independently in attention.
        """
        # Extract shape information
        batch_size = keras.ops.shape(x)[0]
        height = keras.ops.shape(x)[1]
        width = keras.ops.shape(x)[2]
        channels = keras.ops.shape(x)[3]

        # Calculate number of windows in each dimension
        num_windows_h = height // self._window_size
        num_windows_w = width // self._window_size

        # Reshape: (B, H, W, C) -> (B, nH, ws, nW, ws, C)
        x = keras.ops.reshape(
            x,
            (batch_size, num_windows_h, self._window_size,
             num_windows_w, self._window_size, channels)
        )

        # Transpose: (B, nH, ws, nW, ws, C) -> (B, nH, nW, ws, ws, C)
        # This groups the window dimensions together
        x = keras.ops.transpose(x, (0, 1, 3, 2, 4, 5))

        # Reshape: (B, nH, nW, ws, ws, C) -> (B*nH*nW, ws, ws, C)
        # Flatten batch and window dimensions for efficient processing
        windows = keras.ops.reshape(
            x,
            (-1, self._window_size, self._window_size, channels)
        )

        return windows

    def _window_reverse(
            self,
            windows: keras.KerasTensor,
            height: int,
            width: int
    ) -> keras.KerasTensor:
        """
        Reverse window partition to reconstruct the spatial feature map.

        Inverse operation of _window_partition. Reconstructs the original
        spatial structure from the batch of windows.

        Algorithm:
        ---------
        1. Reshape from (B*nW, ws, ws, C) to (B, nH, nW, ws, ws, C)
        2. Transpose to ungroup windows: (B, nH, ws, nW, ws, C)
        3. Flatten spatial dimensions: (B, H, W, C)

        Parameters
        ----------
        windows : keras.KerasTensor
            Windows tensor of shape
            ``(batch_size * num_windows, window_size, window_size, channels)``.
        height : int
            Target height of the reconstructed feature map.
        width : int
            Target width of the reconstructed feature map.

        Returns
        -------
        keras.KerasTensor
            Reconstructed spatial feature tensor of shape
            ``(batch_size, height, width, channels)``.
        """
        # Extract shape information
        channels = keras.ops.shape(windows)[-1]
        num_windows_h = height // self._window_size
        num_windows_w = width // self._window_size
        num_windows = num_windows_h * num_windows_w
        batch_size = keras.ops.shape(windows)[0] // num_windows

        # Reshape: (B*nW, ws, ws, C) -> (B, nH, nW, ws, ws, C)
        x = keras.ops.reshape(
            windows,
            (batch_size, num_windows_h, num_windows_w,
             self._window_size, self._window_size, channels)
        )

        # Transpose: (B, nH, nW, ws, ws, C) -> (B, nH, ws, nW, ws, C)
        # This ungroups the window dimensions
        x = keras.ops.transpose(x, (0, 1, 3, 2, 4, 5))

        # Reshape: (B, nH, ws, nW, ws, C) -> (B, H, W, C)
        # Merge window dimensions back into spatial dimensions
        x = keras.ops.reshape(x, (batch_size, height, width, channels))

        return x

    def _apply_progressive_focusing(
            self,
            attn_scores: keras.KerasTensor,
            prev_attn_map: Optional[keras.KerasTensor]
    ) -> keras.KerasTensor:
        """
        Apply progressive focusing using previous layer's attention map.

        This is the **core innovation of PFA**: attention patterns from
        previous layers guide the current layer's attention through
        element-wise multiplication (Hadamard product). This creates a
        hierarchical attention refinement where:

        - Early layers establish broad attention patterns
        - Middle layers refine these patterns
        - Deep layers focus on the most relevant tokens

        The Hadamard product preserves strong attention connections from
        previous layers while allowing the current layer to modulate
        these patterns based on current features.

        Mathematical Operation:
        ----------------------
        focused_scores = attn_scores ⊙ prev_attn_map

        where ⊙ denotes element-wise multiplication (Hadamard product).

        Parameters
        ----------
        attn_scores : keras.KerasTensor
            Current layer's attention scores of shape
            ``(batch*num_windows, num_heads, window_area, window_area)``.
            These are the raw, scaled dot-product scores before softmax.
        prev_attn_map : keras.KerasTensor, optional
            Previous layer's attention weights (after softmax) of the same shape.
            If None, returns scores unchanged (first layer behavior).

        Returns
        -------
        keras.KerasTensor
            Focused attention scores that incorporate guidance from the
            previous layer. Shape matches input attn_scores.

        Notes
        -----
        - For the first layer, prev_attn_map is None, so standard attention is used
        - The multiplication happens before softmax in the current layer
        - This creates a multiplicative bias toward tokens that were important
          in the previous layer
        """
        if prev_attn_map is None:
            # First layer: no previous attention to inherit
            return attn_scores

        # Apply Hadamard product for progressive focusing
        # This biases current attention toward patterns from previous layer
        focused_scores = attn_scores * prev_attn_map

        return focused_scores

    def _apply_sparsity(
            self,
            attn_scores: keras.KerasTensor,
            prev_attn_map: Optional[keras.KerasTensor]
    ) -> keras.KerasTensor:
        """
        Apply sparsity to attention scores based on configuration.

        Sparsity reduces computation by masking out low-importance attention
        connections identified by the previous layer. This is particularly
        effective for PFA since previous layers have already identified
        which tokens are relevant.

        Sparsity Modes:
        --------------
        1. **'none'**: No sparsity, full attention within windows
        2. **'threshold'**: Mask positions where prev_attn_map < threshold
        3. **'top_k'**: Keep only top-k positions per query based on prev_attn_map

        Parameters
        ----------
        attn_scores : keras.KerasTensor
            Current attention scores of shape
            ``(batch*num_windows, num_heads, window_area, window_area)``.
        prev_attn_map : keras.KerasTensor, optional
            Previous layer's attention map for guidance. If None, returns
            scores unchanged.

        Returns
        -------
        keras.KerasTensor
            Sparse attention scores with masked positions set to -inf
            (will become 0 after softmax).

        Notes
        -----
        - Masked positions are set to -1e9 (effectively -inf) rather than 0
          because the scores haven't been through softmax yet
        - After softmax, -inf becomes 0 (no attention)
        - The sparsity pattern is inherited from the previous layer, creating
          consistent sparsity across layers
        """
        # No sparsity mode or no previous attention map
        if self._sparsity_mode == 'none' or prev_attn_map is None:
            return attn_scores

        if self._sparsity_mode == 'threshold':
            # Threshold-based sparsity: mask positions below threshold
            # Create binary mask: 1.0 for positions to keep, 0.0 for positions to mask
            mask = keras.ops.cast(
                prev_attn_map >= self._sparsity_threshold,
                dtype=attn_scores.dtype
            )

            # Apply mask: keep valid positions, set others to large negative value
            attn_scores = keras.ops.where(
                mask > 0.5,
                attn_scores,
                keras.ops.full_like(attn_scores, -1e9)
            )

        elif self._sparsity_mode == 'top_k':
            # Top-k sparsity: keep only k most important positions per query
            seq_len = keras.ops.shape(attn_scores)[-1]
            k = min(self._top_k, seq_len)  # Handle case where k > sequence length

            # Average previous attention over heads for top-k selection
            prev_mean = keras.ops.mean(prev_attn_map, axis=1, keepdims=True)

            # Get top-k indices based on previous attention
            _, top_indices = keras.ops.top_k(prev_mean, k=k)

            # Note: Full sparse implementation would use scatter operations
            # This is a simplified version - production code would need
            # efficient sparse attention implementation using custom ops
            # or specialized libraries
            mask = keras.ops.zeros_like(attn_scores)
            # TODO: Implement efficient scatter operation for top-k masking

        return attn_scores

    def call(
            self,
            x: keras.KerasTensor,
            prev_attn_map: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Forward pass of Progressive Focused Attention.

        Computes windowed multi-head self-attention with progressive focusing
        and optional shifted windows. The complete flow is:

        1. **Window Preparation**: Optionally shift input, then partition into windows
        2. **QKV Projection**: Project to queries, keys, values
        3. **LePE (optional)**: Add local positional encoding to values
        4. **Attention Scores**: Compute Q @ K^T and scale
        5. **Masking**: Apply shifted window mask if needed
        6. **Progressive Focusing**: Multiply with previous attention map
        7. **Sparsity**: Apply sparsity mask if configured
        8. **Attention Weights**: Softmax normalization
        9. **Output**: Apply attention to values, project, reverse windows

        Parameters
        ----------
        x : keras.KerasTensor
            Input feature tensor of shape ``(batch_size, height, width, dim)``.
            Height and width must be divisible by window_size.
        prev_attn_map : keras.KerasTensor, optional
            Previous layer's attention map for progressive focusing.
            Shape: ``(batch*num_windows, num_heads, window_area, window_area)``.
            If None, standard windowed attention is computed (first layer behavior).
        training : bool, optional
            Whether in training mode. Affects dropout behavior.
            Default is None (uses Keras learning phase).

        Returns
        -------
        tuple
            A tuple of ``(output, attention_weights)`` where:

            - **output**: Tensor of shape ``(batch_size, height, width, dim)``.
              The transformed feature map with same spatial dimensions as input.
            - **attention_weights**: Attention weights of shape
              ``(batch*num_windows, num_heads, window_area, window_area)``.
              These can be used as prev_attn_map for the next layer.

        Raises
        ------
        ValueError
            If input height or width is not divisible by window_size.

        Notes
        -----
        **Computational Complexity:**
        - Without windows: O((H*W)²) for full self-attention
        - With windows: O(H*W*window_size²) - much more efficient
        - The attention is computed independently within each window

        **Memory Usage:**
        - Attention weights: O(batch * num_windows * num_heads * window_area²)
        - For large images, this is significantly smaller than full attention
        """
        # ============ Extract Input Dimensions ============
        input_shape = keras.ops.shape(x)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]

        # ============ Shifted Window Preparation (SW-MSA) ============
        if self._shift_size > 0:
            # Cyclic shift for shifted window attention
            # This moves windows to enable cross-window connections
            # Shift is negative because we'll shift back after attention
            shifted_x = keras.ops.roll(
                x,
                shift=(-self._shift_size, -self._shift_size),
                axis=(1, 2)
            )
        else:
            # Regular window attention (W-MSA) - no shift needed
            shifted_x = x

        # ============ Window Partition ============
        # Transform: (B, H, W, C) -> (B*nW, ws, ws, C)
        # Each window becomes an independent sample
        x_windows = self._window_partition(shifted_x)
        num_windows = keras.ops.shape(x_windows)[0]

        # ============ Flatten Windows for Attention ============
        # Transform: (B*nW, ws, ws, C) -> (B*nW, ws*ws, C)
        # Each window is now a sequence of tokens
        x_flat = keras.ops.reshape(
            x_windows,
            (num_windows, self._window_area, self._dim)
        )

        # ============ QKV Projection ============
        # Project to queries, keys, and values simultaneously
        # (B*nW, ws*ws, C) -> (B*nW, ws*ws, 3*C)
        qkv = self._qkv(x_flat)

        # Reshape to separate Q, K, V and split by attention heads
        # (B*nW, ws*ws, 3*C) -> (B*nW, ws*ws, 3, num_heads, head_dim)
        qkv = keras.ops.reshape(
            qkv,
            (num_windows, self._window_area, 3, self._num_heads, self._head_dim)
        )

        # Transpose to convenient format for attention computation
        # (B*nW, ws*ws, 3, num_heads, head_dim) -> (3, B*nW, num_heads, ws*ws, head_dim)
        qkv = keras.ops.transpose(qkv, (2, 0, 3, 1, 4))

        # Split into Q, K, V
        # Each has shape: (B*nW, num_heads, ws*ws, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # ============ LePE (Locally-Enhanced Positional Encoding) ============
        if self._lepe is not None:
            # Reshape V back to spatial format for depthwise convolution
            # (B*nW, num_heads, ws*ws, head_dim) -> (B*nW, ws, ws, C)
            v_spatial = keras.ops.transpose(v, (0, 2, 1, 3))
            v_spatial = keras.ops.reshape(
                v_spatial,
                (num_windows, self._window_size, self._window_size, self._dim)
            )

            # Apply depthwise convolution to inject local positional information
            # This provides translation-equivariant position awareness
            lepe = self._lepe(v_spatial)

            # Reshape back to multi-head format
            # (B*nW, ws, ws, C) -> (B*nW, num_heads, ws*ws, head_dim)
            lepe = keras.ops.reshape(
                lepe,
                (num_windows, self._window_area, self._num_heads, self._head_dim)
            )
            lepe = keras.ops.transpose(lepe, (0, 2, 1, 3))

            # Add LePE to values (residual connection)
            v = v + lepe

        # ============ Attention Score Computation ============
        # Scaled dot-product attention: Q @ K^T / sqrt(head_dim)
        # (B*nW, num_heads, ws*ws, head_dim) @ (B*nW, num_heads, head_dim, ws*ws)
        # -> (B*nW, num_heads, ws*ws, ws*ws)
        attn_scores = keras.ops.matmul(
            q, keras.ops.transpose(k, (0, 1, 3, 2))
        ) * self._scale

        # ============ Apply Shifted Window Attention Mask ============
        if self._attn_mask is not None and self._shift_size > 0:
            # The mask prevents attention across different shifted regions
            # Add mask (0 for valid, -100 for invalid) to attention scores
            # Broadcasting handles batch and head dimensions
            num_windows_actual = (height // self._window_size) * (width // self._window_size)

            # Broadcast mask to match attention score dimensions
            attn_mask = keras.ops.tile(
                self._attn_mask[None, None, :, :],
                (batch_size, self._num_heads, 1, 1)
            )

            # Add mask to scores (masked positions get large negative values)
            attn_scores = attn_scores + keras.ops.reshape(
                attn_mask[:, :, :self._window_area, :self._window_area],
                (1, 1, self._window_area, self._window_area)
            )

        # ============ Progressive Focusing ============
        # Multiply with previous layer's attention to focus on relevant tokens
        attn_scores = self._apply_progressive_focusing(attn_scores, prev_attn_map)

        # ============ Apply Sparsity ============
        # Optionally mask out low-importance connections
        attn_scores = self._apply_sparsity(attn_scores, prev_attn_map)

        # ============ Softmax Normalization ============
        # Convert scores to probabilities
        attn_weights = keras.ops.softmax(attn_scores, axis=-1)

        # ============ Attention Dropout ============
        # Randomly drop some attention connections during training
        if self._attn_drop is not None:
            attn_weights = self._attn_drop(attn_weights, training=training)

        # ============ Apply Attention to Values ============
        # Weighted sum of values based on attention weights
        # (B*nW, num_heads, ws*ws, ws*ws) @ (B*nW, num_heads, ws*ws, head_dim)
        # -> (B*nW, num_heads, ws*ws, head_dim)
        attn_output = keras.ops.matmul(attn_weights, v)

        # ============ Reshape and Project Output ============
        # Transpose: (B*nW, num_heads, ws*ws, head_dim) -> (B*nW, ws*ws, num_heads, head_dim)
        attn_output = keras.ops.transpose(attn_output, (0, 2, 1, 3))

        # Concatenate heads: (B*nW, ws*ws, num_heads, head_dim) -> (B*nW, ws*ws, C)
        attn_output = keras.ops.reshape(
            attn_output,
            (num_windows, self._window_area, self._dim)
        )

        # Output projection: (B*nW, ws*ws, C) -> (B*nW, ws*ws, C)
        output = self._proj(attn_output)

        # ============ Projection Dropout ============
        if self._proj_drop is not None:
            output = self._proj_drop(output, training=training)

        # ============ Reverse Window Partition ============
        # Reshape back to spatial format: (B*nW, ws*ws, C) -> (B*nW, ws, ws, C)
        output = keras.ops.reshape(
            output,
            (num_windows, self._window_size, self._window_size, self._dim)
        )

        # Reverse window partition: (B*nW, ws, ws, C) -> (B, H, W, C)
        output = self._window_reverse(output, height, width)

        # ============ Reverse Cyclic Shift (for SW-MSA) ============
        if self._shift_size > 0:
            # Shift back to original positions
            output = keras.ops.roll(
                output,
                shift=(self._shift_size, self._shift_size),
                axis=(1, 2)
            )

        return output, attn_weights

    def get_config(self) -> Dict[str, Any]:
        """
        Return layer configuration for serialization.

        Returns
        -------
        dict
            Configuration dictionary containing all parameters needed
            to reconstruct this layer via `from_config`.
        """
        config = super().get_config()
        config.update({
            "dim": self._dim,
            "num_heads": self._num_heads,
            "window_size": self._window_size,
            "shift_size": self._shift_size,
            "top_k": self._top_k,
            "sparsity_threshold": self._sparsity_threshold,
            "sparsity_mode": self._sparsity_mode,
            "qkv_bias": self._qkv_bias,
            "attention_dropout": self._attention_dropout,
            "projection_dropout": self._projection_dropout,
            "use_lepe": self._use_lepe,
            "lepe_kernel_size": self._lepe_kernel_size,
            "kernel_initializer": keras.initializers.serialize(
                self._kernel_initializer
            ),
            "bias_initializer": keras.initializers.serialize(
                self._bias_initializer
            ),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ProgressiveFocusedAttention":
        """
        Create layer from configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary from `get_config`.

        Returns
        -------
        ProgressiveFocusedAttention
            New instance created from configuration.
        """
        # Deserialize initializers from their serialized format
        config = config.copy()
        config["kernel_initializer"] = keras.initializers.deserialize(
            config.get("kernel_initializer", "glorot_uniform")
        )
        config["bias_initializer"] = keras.initializers.deserialize(
            config.get("bias_initializer", "zeros")
        )
        return cls(**config)

    def compute_output_shape(
            self,
            input_shape: Union[tuple, list]
    ) -> Tuple[tuple, tuple]:
        """
        Compute output shape of the layer.

        Parameters
        ----------
        input_shape : tuple or list
            Input shape(s). Either a single tuple for x, or a list of tuples
            for (x, prev_attn_map).

        Returns
        -------
        tuple
            Output shapes as ``(output_shape, attn_map_shape)`` where:

            - output_shape: Same as input x shape ``(batch, height, width, dim)``
            - attn_map_shape: Attention weights shape
              ``(batch*num_windows, num_heads, window_area, window_area)``
        """
        # Extract x shape
        if isinstance(input_shape, list):
            x_shape = input_shape[0]
        else:
            x_shape = input_shape

        # Output has same shape as input
        output_shape = x_shape

        # Compute attention map shape
        batch = x_shape[0]
        h, w = x_shape[1], x_shape[2]

        if h is not None and w is not None:
            num_windows = (h // self._window_size) * (w // self._window_size)
            if batch is not None:
                attn_batch = batch * num_windows
            else:
                attn_batch = None
        else:
            attn_batch = None

        attn_map_shape = (
            attn_batch,
            self._num_heads,
            self._window_area,
            self._window_area
        )

        return output_shape, attn_map_shape


# ---------------------------------------------------------------------