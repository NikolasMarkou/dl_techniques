"""
Progressive Focused Attention (PFA) Module for PFT-SR.

This module implements the core innovation of PFT-SR: Progressive Focused Attention,
which inherits attention maps from previous layers through Hadamard product and uses
sparse matrix multiplication to skip unnecessary similarity calculations. Attention
maps are progressively refined across layers, with each layer focusing on increasingly
relevant tokens while filtering out irrelevant features before calculating similarities.

References:
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
    """Windowed self-attention with progressive focusing from previous layers.

    Implements the PFA mechanism from PFT-SR which computes window-based multi-head
    self-attention (W-MSA or SW-MSA) with progressive refinement. The input is
    partitioned into non-overlapping windows of size ``window_size x window_size``,
    reducing complexity from ``O((HW)^2)`` to ``O(HW * window_size^2)``. Within each
    window, standard scaled dot-product attention ``softmax(Q K^T / sqrt(d_k)) V``
    is computed with optional LePE (Locally-Enhanced Positional Encoding) via
    depthwise convolution on value vectors. The core innovation is progressive
    focusing: ``focused_scores = scores * prev_attn_map`` (Hadamard product),
    which biases attention toward tokens deemed important by the previous layer.
    Shifted windows (SW-MSA) with cyclic shift and attention masking enable
    cross-window connections in alternating layers.

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────────────────┐
        │ Input [B, H, W, C]              │
        └────────────────┬────────────────┘
                         ▼
        ┌─────────────────────────────────┐
        │ Cyclic Shift (if shift_size > 0)│
        └────────────────┬────────────────┘
                         ▼
        ┌─────────────────────────────────┐
        │ Window Partition                │
        │ → [B*nW, ws, ws, C]             │
        └────────────────┬────────────────┘
                         ▼
        ┌─────────────────────────────────┐
        │ QKV Projection Dense(3*C)       │
        │ → Q, K, V per head              │
        │ [B*nW, heads, ws^2, head_dim]   │
        └────────────────┬────────────────┘
                         ▼
        ┌─────────────────────────────────┐
        │ LePE: DepthwiseConv2D on V      │
        │ (optional local pos. encoding)  │
        └────────────────┬────────────────┘
                         ▼
        ┌─────────────────────────────────┐
        │ Attn Scores = Q @ K^T * scale   │
        │ + SW-MSA mask (if shifted)      │
        └────────────────┬────────────────┘
                         ▼
        ┌─────────────────────────────────┐
        │ Progressive Focusing:           │
        │ scores = scores * prev_attn_map │
        │ + Sparsity (optional)           │
        └────────────────┬────────────────┘
                         ▼
        ┌─────────────────────────────────┐
        │ softmax → Attn Weights          │
        │ + Dropout (optional)            │
        └────────────────┬────────────────┘
                         ▼
        ┌─────────────────────────────────┐
        │ Output = Attn @ V               │
        │ → Output Projection Dense(C)    │
        └────────────────┬────────────────┘
                         ▼
        ┌─────────────────────────────────┐
        │ Window Reverse + Unshift        │
        └────────────────┬────────────────┘
                         ▼
        ┌─────────────────────────────────┐
        │ Output [B, H, W, C]             │
        │ + Attention Map for next layer  │
        └─────────────────────────────────┘

    :param dim: Embedding dimension (number of channels).
    :type dim: int
    :param num_heads: Number of attention heads. Must divide ``dim`` evenly.
    :type num_heads: int
    :param window_size: Size of the attention window.
    :type window_size: int
    :param shift_size: Shift size for SW-MSA. Use 0 for W-MSA,
        ``window_size // 2`` for shifted windows.
    :type shift_size: int
    :param top_k: Number of top-k tokens to attend to when
        ``sparsity_mode='top_k'``. ``None`` attends to all tokens.
    :type top_k: Optional[int]
    :param sparsity_threshold: Threshold for sparsity-based attention masking
        when ``sparsity_mode='threshold'``.
    :type sparsity_threshold: float
    :param sparsity_mode: Sparse attention mode: ``'none'``, ``'top_k'``, or
        ``'threshold'``.
    :type sparsity_mode: SparsityMode
    :param qkv_bias: Whether to include bias terms in QKV projections.
    :type qkv_bias: bool
    :param attention_dropout: Dropout rate for attention weights.
    :type attention_dropout: float
    :param projection_dropout: Dropout rate for output projection.
    :type projection_dropout: float
    :param use_lepe: Whether to use Locally-Enhanced Positional Encoding via
        depthwise convolution on value vectors.
    :type use_lepe: bool
    :param lepe_kernel_size: Kernel size for LePE depthwise convolution.
    :type lepe_kernel_size: int
    :param kernel_initializer: Initializer for projection weight matrices.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for bias vectors.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kwargs: Additional keyword arguments for the Layer base class.
    :type kwargs: Any
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
        """Validate layer configuration parameters.

        :raises ValueError: If any configuration parameter is invalid or incompatible.
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
        """Build layer weights and sub-layers.

        :param input_shape: Shape tuple or list of shape tuples for input tensor.
            Expected shape: ``(batch_size, height, width, dim)``.
        :type input_shape: Union[tuple, list]
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
        """Compute attention mask for shifted window attention (SW-MSA).

        :return: Attention mask of shape ``(num_windows, window_area, window_area)``
            with 0.0 for valid pairs and -100.0 for masked pairs.
        :rtype: keras.Variable
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
        """Partition input feature map into non-overlapping windows.

        :param x: Input tensor of shape ``(batch_size, height, width, channels)``.
            Height and width must be divisible by window_size.
        :type x: keras.KerasTensor

        :return: Partitioned windows of shape
            ``(batch_size * num_windows, window_size, window_size, channels)``.
        :rtype: keras.KerasTensor
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
        """Reverse window partition to reconstruct the spatial feature map.

        :param windows: Windows tensor of shape
            ``(batch_size * num_windows, window_size, window_size, channels)``.
        :type windows: keras.KerasTensor
        :param height: Target height of the reconstructed feature map.
        :type height: int
        :param width: Target width of the reconstructed feature map.
        :type width: int

        :return: Reconstructed spatial tensor of shape
            ``(batch_size, height, width, channels)``.
        :rtype: keras.KerasTensor
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
        """Apply progressive focusing via Hadamard product with previous attention map.

        :param attn_scores: Current attention scores of shape
            ``(batch*num_windows, num_heads, window_area, window_area)``.
        :type attn_scores: keras.KerasTensor
        :param prev_attn_map: Previous layer's attention weights of the same shape.
            If ``None``, returns scores unchanged (first layer).
        :type prev_attn_map: Optional[keras.KerasTensor]

        :return: Focused attention scores incorporating previous layer guidance.
        :rtype: keras.KerasTensor
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
        """Apply sparsity masking to attention scores based on previous layer guidance.

        :param attn_scores: Current attention scores of shape
            ``(batch*num_windows, num_heads, window_area, window_area)``.
        :type attn_scores: keras.KerasTensor
        :param prev_attn_map: Previous layer's attention map for guidance.
            If ``None``, returns scores unchanged.
        :type prev_attn_map: Optional[keras.KerasTensor]

        :return: Sparse attention scores with masked positions set to ``-1e9``.
        :rtype: keras.KerasTensor
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
        """Forward pass of Progressive Focused Attention.

        :param x: Input tensor of shape ``(batch_size, height, width, dim)``.
            Height and width must be divisible by window_size.
        :type x: keras.KerasTensor
        :param prev_attn_map: Previous layer's attention map for progressive focusing.
            Shape ``(batch*num_windows, num_heads, window_area, window_area)``.
            If ``None``, standard windowed attention is computed.
        :type prev_attn_map: Optional[keras.KerasTensor]
        :param training: Whether in training mode. Affects dropout behavior.
        :type training: Optional[bool]

        :return: Tuple of ``(output, attention_weights)`` where output has shape
            ``(batch_size, height, width, dim)`` and attention_weights has shape
            ``(batch*num_windows, num_heads, window_area, window_area)``.
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor]

        :raises ValueError: If input height or width is not divisible by window_size.
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
        """Return layer configuration for serialization.

        :return: Configuration dictionary containing all parameters needed
            to reconstruct this layer.
        :rtype: Dict[str, Any]
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
        """Create layer from configuration dictionary.

        :param config: Configuration dictionary from ``get_config``.
        :type config: Dict[str, Any]

        :return: New instance created from configuration.
        :rtype: ProgressiveFocusedAttention
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
        """Compute output shape of the layer.

        :param input_shape: Input shape(s). Either a single tuple for x, or a list
            of tuples for ``(x, prev_attn_map)``.
        :type input_shape: Union[tuple, list]

        :return: Tuple of ``(output_shape, attn_map_shape)`` where output_shape
            matches the input x shape and attn_map_shape is the attention weights shape.
        :rtype: Tuple[tuple, tuple]
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
