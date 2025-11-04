"""
KAN Window Attention Layer.

This module introduces `KANWindowAttention`, a novel attention mechanism that
integrates the principles of Kolmogorov-Arnold Networks (KAN) into the
efficient windowed attention framework of Swin Transformers.

Instead of using standard linear transformations (Dense layers) for generating
Query (Q), Key (K), and Value (V) projections, this layer employs `KANLinear`
units. This allows the activation functions themselves to be learned as B-spline
curves for each Q, K, and V projection, leading to a more expressive and
potentially more powerful attention mechanism.
"""

import keras
from typing import Any, Dict, Optional, Tuple, Union


# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..kan_linear import KANLinear

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class SingleWindowAttentionKAN(keras.layers.Layer):
    """
    Multi-head self-attention with KAN projections for windowed attention.

    This layer implements the core attention mechanism for a single window,
    replacing standard QKV dense projections with learnable KAN (Kolmogorov-Arnold
    Network) transformations. Each of the Query, Key, and Value projections uses
    a separate `KANLinear` layer that learns non-linear B-spline activation
    functions, enabling more expressive attention patterns.

    **Intent**: Enhance the representational capacity of windowed self-attention
    by replacing fixed linear projections with adaptive, learnable non-linear
    transformations via KAN layers, while maintaining the efficiency of local
    attention and compatibility with the Swin Transformer architecture.

    **Architecture**:
    ```
    Input(shape=[batch * num_windows, window_area, dim])
              ↓
    ┌─────────────────────────────────────────┐
    │  Parallel KAN Projections:              │
    │  Q = KANLinear_q(input)                 │
    │  K = KANLinear_k(input)                 │
    │  V = KANLinear_v(input)                 │
    └─────────────────────────────────────────┘
              ↓
    Reshape & Transpose to (batch, heads, seq, head_dim)
              ↓
    Attention Scores = (Q @ K^T) / √d_k + RelativeBias
              ↓
    Apply Attention Mask (if provided)
              ↓
    Attention Weights = Softmax(Scores) → Dropout
              ↓
    Context = Attention_Weights @ V
              ↓
    Reshape & Project: Dense(dim)
              ↓
    Output(shape=[batch * num_windows, window_area, dim])
    ```

    **Mathematical Operations**:
    1. **KAN Projections**:
       - Q = KANLinear_q(X) where KANLinear learns B-spline activations
       - K = KANLinear_k(X)
       - V = KANLinear_v(X)
    2. **Multi-Head Reshaping**:
       - Reshape to [B, N, num_heads, head_dim]
       - Transpose to [B, num_heads, N, head_dim]
    3. **Scaled Dot-Product Attention**:
       - scores = (Q @ K^T) * scale where scale = head_dim^(-0.5)
       - scores = scores + relative_position_bias
    4. **Masking & Softmax**:
       - masked_scores = scores + mask * (-1e9)
       - weights = softmax(masked_scores)
    5. **Context Aggregation**:
       - context = weights @ V
       - output = Dense(context)

    **Relative Position Bias**:
    Uses learned 2D relative position encodings indexed by spatial distance
    between tokens within the window, adding inductive bias for local structure.

    Args:
        dim: Integer, dimension of input tokens (channels). Must be positive and
            divisible by num_heads. This is the embedding dimension for each token.
        window_size: Integer, height and width of the square attention window.
            Must be positive. Tokens are processed in local windows of size
            (window_size × window_size) for efficient attention computation.
        num_heads: Integer, number of parallel attention heads. Must be positive
            and evenly divide dim. Each head operates on dim/num_heads dimensions.
        kan_grid_size: Integer, size of the grid for KAN B-spline basis functions.
            Larger values allow more complex learned activation functions but
            increase parameters. Defaults to 5.
        kan_spline_order: Integer, order of the B-spline basis functions (degree + 1).
            Higher orders enable smoother learned functions. Typical values: 2-4.
            Defaults to 3 (quadratic splines).
        kan_activation: String, base activation function for KAN layers.
            Applied before B-spline transformation. Common choices: 'swish', 'gelu',
            'silu'. Defaults to 'swish'.
        kan_regularization_factor: Float, regularization strength for KAN weights.
            Higher values encourage simpler learned functions. Must be non-negative.
            Defaults to 0.01.
        qk_scale: Optional float, override for query-key scaling factor in attention.
            If None, uses head_dim^(-0.5) as per standard scaled dot-product attention.
            Defaults to None.
        attn_dropout_rate: Float between 0 and 1, dropout rate applied to attention
            weights after softmax. Helps prevent overfitting to specific attention
            patterns. Defaults to 0.0.
        proj_dropout_rate: Float between 0 and 1, dropout rate applied to the final
            output projection. Applied after aggregating multi-head attention outputs.
            Defaults to 0.0.
        proj_bias: Boolean, if True, add learnable bias to the output projection layer.
            Defaults to True.
        kernel_initializer: Initializer for the output projection kernel weights.
            Accepts string name or Initializer instance. Defaults to 'glorot_uniform'.
        bias_initializer: Initializer for the output projection bias weights.
            Accepts string name or Initializer instance. Defaults to 'zeros'.
        kernel_regularizer: Optional regularizer for output projection kernel weights.
            Accepts string name or Regularizer instance. Defaults to None.
        bias_regularizer: Optional regularizer for output projection bias weights.
            Accepts string name or Regularizer instance. Defaults to None.
        **kwargs: Additional keyword arguments for base Layer class (name, dtype, etc.).

    Input shape:
        3D tensor with shape: `(batch_size, sequence_length, dim)`.
        The sequence_length should be ≤ window_size². During processing, if
        sequence_length < window_size², the input is internally padded.

    Output shape:
        3D tensor with same shape as input: `(batch_size, sequence_length, dim)`.
        Padding is removed before returning the output.

    Attributes:
        query: KANLinear layer for query projection with learned activations.
        key: KANLinear layer for key projection with learned activations.
        value: KANLinear layer for value projection with learned activations.
        proj: Dense layer for final output projection after multi-head aggregation.
        attn_dropout: Optional Dropout layer for attention weights.
        proj_dropout: Optional Dropout layer for output projection.
        relative_position_bias_table: Learnable 2D relative position bias parameters
            of shape ((2*window_size-1)², num_heads).
        relative_position_index: Non-trainable lookup table for relative positions,
            shape (window_size², window_size²), precomputed for efficiency.

    Example:
        ```python
        import keras
        from kan_window_attention import SingleKANWindowAttention

        # Create single window attention with KAN projections
        kan_attn = SingleKANWindowAttention(
            dim=96,
            window_size=7,
            num_heads=3,
            kan_grid_size=5,
            kan_spline_order=3
        )

        # Process a batch of windows (4 windows, 49 tokens each, 96 dims)
        window_tokens = keras.random.normal((4, 49, 96))
        output = kan_attn(window_tokens)  # Shape: (4, 49, 96)

        # With attention mask to ignore padding
        mask = keras.ops.ones((4, 49), dtype='int32')
        mask[:, -10:] = 0  # Mask last 10 tokens
        output_masked = kan_attn(window_tokens, attention_mask=mask)
        ```

    Note:
        This layer is typically not used directly but is wrapped by
        `KANWindowAttention` which handles sequence partitioning, windowing,
        and un-windowing operations. The input sequence_length must be ≤ window_size²
        as this operates on a single window.

    Raises:
        ValueError: If dim ≤ 0, window_size ≤ 0, num_heads ≤ 0, or if dim is not
            divisible by num_heads.
    """

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        kan_grid_size: int = 5,
        kan_spline_order: int = 3,
        kan_activation: str = "swish",
        kan_regularization_factor: float = 0.01,
        qk_scale: Optional[float] = None,
        attn_dropout_rate: float = 0.0,
        proj_dropout_rate: float = 0.0,
        proj_bias: bool = True,
        kernel_initializer: Union[
            str, keras.initializers.Initializer
        ] = "glorot_uniform",
        bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
        kernel_regularizer: Optional[
            Union[str, keras.regularizers.Regularizer]
        ] = None,
        bias_regularizer: Optional[
            Union[str, keras.regularizers.Regularizer]
        ] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # --- Validation ---
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})"
            )
        if not (0.0 <= attn_dropout_rate <= 1.0):
            raise ValueError(
                f"attn_dropout_rate must be between 0.0 and 1.0, got {attn_dropout_rate}"
            )
        if not (0.0 <= proj_dropout_rate <= 1.0):
            raise ValueError(
                f"proj_dropout_rate must be between 0.0 and 1.0, got {proj_dropout_rate}"
            )
        if kan_grid_size <= 0:
            raise ValueError(
                f"kan_grid_size must be positive, got {kan_grid_size}"
            )
        if kan_spline_order <= 0:
            raise ValueError(
                f"kan_spline_order must be positive, got {kan_spline_order}"
            )
        if kan_regularization_factor < 0:
            raise ValueError(
                f"kan_regularization_factor must be non-negative, got {kan_regularization_factor}"
            )

        # --- Store ALL configuration parameters for serialization ---
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_scale = qk_scale
        self.scale = (
            qk_scale if qk_scale is not None else self.head_dim**-0.5
        )
        self.attn_dropout_rate = attn_dropout_rate
        self.proj_dropout_rate = proj_dropout_rate
        self.proj_bias = proj_bias

        # KAN-specific parameters
        self.kan_grid_size = kan_grid_size
        self.kan_spline_order = kan_spline_order
        self.kan_activation = kan_activation
        self.kan_regularization_factor = kan_regularization_factor

        # Projection initializers/regularizers
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # --- Pre-compute constant relative position indices ---
        # These are computed once and reused for efficiency
        coords_h = keras.ops.arange(self.window_size, dtype="int32")
        coords_w = keras.ops.arange(self.window_size, dtype="int32")
        coords = keras.ops.stack(
            keras.ops.meshgrid(coords_h, coords_w, indexing="ij")
        )
        coords_flatten = keras.ops.reshape(coords, (2, -1))
        relative_coords = keras.ops.expand_dims(
            coords_flatten, 2
        ) - keras.ops.expand_dims(coords_flatten, 1)
        relative_coords = keras.ops.transpose(relative_coords, (1, 2, 0))
        relative_coords_h = relative_coords[:, :, 0] + self.window_size - 1
        relative_coords_w = relative_coords[:, :, 1] + self.window_size - 1
        relative_coords_h *= 2 * self.window_size - 1
        self.relative_position_index = relative_coords_h + relative_coords_w

        kan_params = {
            "features": self.dim,
            "grid_size": self.kan_grid_size,
            "spline_order": self.kan_spline_order,
            "activation": self.kan_activation,
            "regularization_factor": self.kan_regularization_factor,
        }

        self.query = KANLinear(**kan_params, name="kan_query")
        self.key = KANLinear(**kan_params, name="kan_key")

        self.value = keras.layers.Dense(
            self.dim,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="value",
        )

        self.proj = keras.layers.Dense(
            self.dim,
            use_bias=self.proj_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="proj",
        )

        # Dropout layers (only created if rate > 0)
        self.attn_dropout = (
            keras.layers.Dropout(self.attn_dropout_rate, name="attn_dropout")
            if self.attn_dropout_rate > 0.0
            else None
        )
        self.proj_dropout = (
            keras.layers.Dropout(self.proj_dropout_rate, name="proj_dropout")
            if self.proj_dropout_rate > 0.0
            else None
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer's weights and sub-layers.

        Creates the relative position bias table and explicitly builds all
        sub-layers (KAN projections and output projection) to ensure proper
        weight variable creation before serialization/deserialization.

        Args:
            input_shape: Shape of the input tensor (batch_size, sequence_length, dim).
        """
        # Create relative position bias table
        num_relative_positions = (2 * self.window_size - 1) ** 2
        self.relative_position_bias_table = self.add_weight(
            name="relative_position_bias_table",
            shape=(num_relative_positions, self.num_heads),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
            dtype=self.dtype,
        )

        # CRITICAL: Explicitly build all sub-layers for proper serialization
        # Build KAN projection layers with padded shape (to window_size²)
        padded_shape = list(input_shape)
        padded_shape[1] = self.window_size * self.window_size
        self.query.build(padded_shape)
        self.key.build(padded_shape)
        self.value.build(padded_shape)

        # Build output projection
        self.proj.build(padded_shape)

        # Build dropout layers if they exist
        if self.attn_dropout is not None:
            self.attn_dropout.build(None)  # Dropout doesn't need input shape
        if self.proj_dropout is not None:
            self.proj_dropout.build(padded_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """
        Forward pass for single KAN window attention.

        Applies KAN-based multi-head self-attention with relative position bias
        to input tokens. Handles padding if sequence length is less than window
        area, and removes padding from output.

        Args:
            inputs: Input tensor of shape (batch_size, sequence_length, dim).
                sequence_length must be ≤ window_size². If less, will be padded
                internally and unpadded before returning.
            attention_mask: Optional binary mask of shape (batch_size, sequence_length)
                where 1 indicates valid positions and 0 indicates positions to mask.
                Applied in addition to any internal padding masks.
            training: Boolean or None, whether the layer is in training mode.
                Affects dropout behavior.

        Returns:
            Output tensor of shape (batch_size, sequence_length, dim) after
            applying KAN-based attention. Has same shape as input (padding removed).
        """
        input_shape = keras.ops.shape(inputs)
        B_actual, N_actual, C_actual = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
        )
        N_target = self.window_size * self.window_size

        # --- PADDING & MASKING LOGIC ---
        # Pad sequence to window area if needed
        padding_amount = N_target - N_actual
        padded_inputs = keras.ops.pad(
            inputs, [[0, 0], [0, padding_amount], [0, 0]]
        )

        # Create internal padding mask (1 for real tokens, 0 for padding)
        internal_padding_mask = keras.ops.concatenate(
            [
                keras.ops.ones((B_actual, N_actual), dtype="int32"),
                keras.ops.zeros((B_actual, padding_amount), dtype="int32"),
            ],
            axis=1,
        )

        # Combine with user-provided attention mask if present
        final_attention_mask = internal_padding_mask
        if attention_mask is not None:
            # Pad user mask to match padded sequence length
            mask_padding = keras.ops.zeros(
                (B_actual, padding_amount), dtype=attention_mask.dtype
            )
            padded_user_mask = keras.ops.concatenate(
                [attention_mask, mask_padding], axis=1
            )
            # Combine masks (both must be 1 for valid positions)
            final_attention_mask = (
                keras.ops.cast(padded_user_mask, "int32")
                * internal_padding_mask
            )

        # --- ATTENTION COMPUTATION ---
        B, N, C = keras.ops.shape(padded_inputs)

        # KAN-based Q, K, V projections with learned non-linear activations
        q_proj = self.query(padded_inputs, training=training)
        k_proj = self.key(padded_inputs, training=training)
        v_proj = self.value(padded_inputs, training=training)

        # Reshape for multi-head attention: (B, N, num_heads, head_dim)
        q = keras.ops.reshape(q_proj, (B, N, self.num_heads, self.head_dim))
        q = keras.ops.transpose(q, (0, 2, 1, 3))  # (B, num_heads, N, head_dim)
        k = keras.ops.reshape(k_proj, (B, N, self.num_heads, self.head_dim))
        k = keras.ops.transpose(k, (0, 2, 1, 3))
        v = keras.ops.reshape(v_proj, (B, N, self.num_heads, self.head_dim))
        v = keras.ops.transpose(v, (0, 2, 1, 3))

        # Scaled dot-product attention: (Q @ K^T) / √d_k
        q = q * self.scale
        attn = keras.ops.matmul(
            q, keras.ops.transpose(k, (0, 1, 3, 2))
        )  # (B, num_heads, N, N)

        # Add learned relative position bias
        relative_position_bias = keras.ops.take(
            self.relative_position_bias_table,
            keras.ops.reshape(self.relative_position_index, (-1,)),
            axis=0,
        )
        relative_position_bias = keras.ops.reshape(
            relative_position_bias, (N_target, N_target, -1)
        )
        relative_position_bias = keras.ops.transpose(
            relative_position_bias, (2, 0, 1)
        )  # (num_heads, N, N)
        attn = attn + keras.ops.expand_dims(
            relative_position_bias, 0
        )  # (B, num_heads, N, N)

        # Apply attention mask (set masked positions to large negative value)
        broadcast_mask = keras.ops.reshape(
            final_attention_mask, (B, 1, 1, N)
        )  # (B, 1, 1, N)
        inf_value = keras.ops.convert_to_tensor(-1e9, dtype=attn.dtype)
        additive_mask = (
            1.0 - keras.ops.cast(broadcast_mask, dtype=attn.dtype)
        ) * inf_value
        attn = attn + additive_mask

        # Softmax and optional dropout on attention weights
        attn = keras.ops.softmax(attn, axis=-1)
        if self.attn_dropout is not None:
            attn = self.attn_dropout(attn, training=training)

        # Apply attention to values and aggregate
        x = keras.ops.matmul(attn, v)  # (B, num_heads, N, head_dim)
        x = keras.ops.transpose(x, (0, 2, 1, 3))  # (B, N, num_heads, head_dim)
        x = keras.ops.reshape(x, (B, N, C))  # (B, N, dim)

        # Output projection
        x = self.proj(x, training=training)
        if self.proj_dropout is not None:
            x = self.proj_dropout(x, training=training)

        # Remove padding to return original sequence length
        output = x[:, :N_actual, :]
        return output

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute output shape of the layer.

        Output shape is identical to input shape as attention preserves dimensions.

        Args:
            input_shape: Shape tuple of input tensor.

        Returns:
            Shape tuple identical to input_shape.
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration dictionary for layer serialization.

        Returns all constructor parameters needed to reconstruct the layer,
        ensuring proper serialization and deserialization.

        Returns:
            Dictionary mapping constructor parameter names to their values.
        """
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "window_size": self.window_size,
                "num_heads": self.num_heads,
                "kan_grid_size": self.kan_grid_size,
                "kan_spline_order": self.kan_spline_order,
                "kan_activation": self.kan_activation,
                "kan_regularization_factor": self.kan_regularization_factor,
                "qk_scale": self.qk_scale,
                "attn_dropout_rate": self.attn_dropout_rate,
                "proj_dropout_rate": self.proj_dropout_rate,
                "proj_bias": self.proj_bias,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": keras.initializers.serialize(
                    self.bias_initializer
                ),
                "kernel_regularizer": keras.regularizers.serialize(
                    self.kernel_regularizer
                ),
                "bias_regularizer": keras.regularizers.serialize(
                    self.bias_regularizer
                ),
            }
        )
        return config


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class WindowAttentionKAN(keras.layers.Layer):
    """
    KAN-enhanced windowed multi-head self-attention for sequences.

    This layer implements efficient windowed attention with learned non-linear
    projections via Kolmogorov-Arnold Networks (KAN). It takes a 1D sequence,
    reshapes it into a 2D spatial grid, partitions the grid into local windows,
    applies KAN-based attention within each window using relative position bias,
    and merges results back into a 1D sequence.

    **Intent**: Provide an efficient, expressive attention mechanism for long
    sequences by combining the computational benefits of local windowed attention
    with the enhanced representational power of KAN-based learned non-linear
    projections, enabling better modeling of complex patterns in hierarchical
    vision and sequence models.

    **Architecture**:
    ```
    Input: 1D Sequence (batch, seq_len, dim)
              ↓
    Reshape to 2D Grid (batch, H, W, dim)
              ↓
    Pad Grid to Multiple of window_size
              ↓
    Partition into Windows (num_windows, window_area, dim)
              ↓
    ┌──────────────────────────────────────┐
    │  For Each Window (in parallel):      │
    │  • KAN Query Projection              │
    │  • KAN Key Projection                │
    │  • KAN Value Projection              │
    │  • Multi-Head Attention + Rel. Bias  │
    │  • Output Projection                 │
    └──────────────────────────────────────┘
              ↓
    Merge Windows Back to Grid
              ↓
    Remove Padding
              ↓
    Reshape to 1D Sequence (batch, seq_len, dim)
              ↓
    Output: (batch, seq_len, dim)
    ```

    **Mathematical Operations**:
    1. **Grid Formation**: Sequence of length N → Grid of H×W where H×W ≥ N
       - H = W = ⌈√N⌉
    2. **Window Partitioning**: Grid → M windows of size (window_size, window_size)
       - M = (H_pad / window_size) × (W_pad / window_size)
       - Grid padded to make dimensions divisible by window_size
    3. **Per-Window KAN Attention**:
       - Q, K, V = KANLinear(window_tokens) with learned B-spline activations
       - Attention = softmax((Q @ K^T) / √d_k + relative_bias) @ V
    4. **Window Merging**: Reverse partition to reconstruct grid
    5. **Sequence Reconstruction**: Remove padding, reshape to (batch, N, dim)

    **Compared to Standard Window Attention**:
    - Traditional: Q, K, V = Dense(X) with fixed linear projections
    - KAN Version: Q, K, V = KANLinear(X) with learned non-linear activations
    - Benefit: More expressive projections can capture complex input patterns

    Args:
        dim: Integer, dimension of input tokens (channels). Must be positive and
            divisible by num_heads. Represents the embedding dimension.
        window_size: Integer, height and width of square attention windows.
            Must be positive. Larger windows increase receptive field but also
            computational cost (quadratic in window_size²).
        num_heads: Integer, number of parallel attention heads. Must be positive
            and evenly divide dim. More heads allow attending to different
            representation subspaces.
        kan_grid_size: Integer, size of grid for KAN B-spline basis. Controls
            complexity of learned activation functions. Defaults to 5.
        kan_spline_order: Integer, order of B-spline basis (degree + 1). Higher
            values enable smoother functions. Defaults to 3.
        kan_activation: String, base activation for KAN layers. Applied before
            B-spline transformation. Defaults to 'swish'.
        kan_regularization_factor: Float, L2 regularization factor for KAN weights.
            Defaults to 0.01.
        qk_scale: Optional float, manual override for attention scaling factor.
            If None, uses 1/√head_dim. Defaults to None.
        attn_dropout_rate: Float in [0, 1], dropout rate for attention weights.
            Defaults to 0.0.
        proj_dropout_rate: Float in [0, 1], dropout rate for output projection.
            Defaults to 0.0.
        proj_bias: Boolean, whether to use bias in output projection. Defaults to True.
        kernel_initializer: Initializer for projection kernels. Defaults to
            'glorot_uniform'.
        bias_initializer: Initializer for projection biases. Defaults to 'zeros'.
        kernel_regularizer: Optional regularizer for projection kernels. Defaults
            to None.
        bias_regularizer: Optional regularizer for projection biases. Defaults to None.
        **kwargs: Additional base Layer arguments (name, dtype, trainable, etc.).

    Input shape:
        3D tensor with shape: `(batch_size, sequence_length, dim)`.
        sequence_length can be any positive integer; the layer handles arbitrary
        lengths by padding to the nearest square grid.

    Output shape:
        3D tensor with same shape as input: `(batch_size, sequence_length, dim)`.
        All padding is removed so output matches input dimensions exactly.

    Attributes:
        attention: Internal SingleKANWindowAttention layer that performs the
            actual attention computation on individual windows.

    Example:
        ```python
        import keras
        from kan_window_attention import KANWindowAttention

        # Standard usage for a Swin-style transformer block
        kan_window_attn = KANWindowAttention(
            dim=96,
            window_size=7,
            num_heads=3,
            kan_grid_size=5,
            kan_spline_order=3
        )

        # Process variable-length sequence
        x = keras.random.normal((2, 150, 96))  # 150 tokens
        output = kan_window_attn(x)  # Shape: (2, 150, 96)

        # With attention mask to handle padding
        mask = keras.ops.ones((2, 150), dtype='int32')
        mask[:, -20:] = 0  # Mask last 20 positions
        output_masked = kan_window_attn(x, attention_mask=mask)

        # Integration in a transformer block
        inputs = keras.Input(shape=(None, 96))  # Variable length
        x = keras.layers.LayerNormalization()(inputs)
        x = kan_window_attn(x)
        x = keras.layers.Add()([inputs, x])  # Residual connection
        model = keras.Model(inputs, x)
        ```

    Note:
        - For sequences that don't form perfect squares, the layer internally
          pads to the nearest square (H = W = ⌈√N⌉) and removes padding from output.
        - Grid dimensions are then padded to multiples of window_size for partitioning.
        - This two-stage padding ensures efficient windowing while preserving
          exact input sequence lengths.
        - Best suited for hierarchical vision models (Swin-style) and sequence
          models where local attention with learned projections is beneficial.

    Raises:
        ValueError: If dim ≤ 0, window_size ≤ 0, num_heads ≤ 0, or if dim is not
            divisible by num_heads. Validation errors propagate from internal
            SingleKANWindowAttention layer.
    """

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        **kwargs: Any,
    ) -> None:
        # Extract name from kwargs before passing to SingleKANWindowAttention
        layer_name = kwargs.pop("name", "kan_window_attention")
        super().__init__(name=layer_name)

        # Store configuration
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.kwargs = kwargs

        # CREATE internal attention layer in __init__
        self.attention = SingleWindowAttentionKAN(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            **kwargs,
        )

    def _window_partition(
        self, x: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Partition a 4D grid tensor into non-overlapping windows.

        Args:
            x: Grid tensor of shape (batch, H, W, channels) where H and W
               are multiples of window_size.

        Returns:
            Windows tensor of shape (batch * num_windows, window_size,
            window_size, channels) where num_windows = (H/ws) × (W/ws).
        """
        B, H, W, C = keras.ops.shape(x)
        ws = self.window_size
        # Reshape: (B, H//ws, ws, W//ws, ws, C)
        x = keras.ops.reshape(x, (B, H // ws, ws, W // ws, ws, C))
        # Transpose: (B, H//ws, W//ws, ws, ws, C)
        x = keras.ops.transpose(x, (0, 1, 3, 2, 4, 5))
        # Flatten to windows: (B * H//ws * W//ws, ws, ws, C)
        windows = keras.ops.reshape(x, (-1, ws, ws, C))
        return windows

    def _window_reverse(
        self, windows: keras.KerasTensor, H: int, W: int
    ) -> keras.KerasTensor:
        """
        Merge windows back into a 4D grid tensor.

        Args:
            windows: Windows tensor of shape (batch * num_windows, window_size,
                window_size, channels).
            H: Integer, target grid height (must be multiple of window_size).
            W: Integer, target grid width (must be multiple of window_size).

        Returns:
            Grid tensor of shape (batch, H, W, channels).
        """
        # Infer batch size from total number of windows
        B = keras.ops.shape(windows)[0] // (
            (H // self.window_size) * (W // self.window_size)
        )
        ws = self.window_size
        # Reshape: (B, H//ws, W//ws, ws, ws, C)
        x = keras.ops.reshape(windows, (B, H // ws, W // ws, ws, ws, -1))
        # Transpose: (B, H//ws, ws, W//ws, ws, C)
        x = keras.ops.transpose(x, (0, 1, 3, 2, 4, 5))
        # Reshape to grid: (B, H, W, C)
        x = keras.ops.reshape(x, (B, H, W, -1))
        return x

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer's sub-layers.

        Explicitly builds the internal SingleKANWindowAttention layer to ensure
        proper weight variable creation before serialization.

        Args:
            input_shape: Shape tuple of input (batch_size, sequence_length, dim).
        """
        # Build internal attention layer with window area sequence length
        self.attention.build(
            (input_shape[0], self.window_size * self.window_size, self.dim)
        )
        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """
        Forward pass for KAN window attention.

        Handles sequence-to-grid conversion, window partitioning, per-window
        attention with KAN projections, window merging, and grid-to-sequence
        reconstruction with all padding removed.

        Args:
            inputs: Input sequence tensor of shape (batch_size, sequence_length, dim).
            attention_mask: Optional binary mask of shape (batch_size, sequence_length)
                where 1 indicates valid tokens and 0 indicates masked positions.
                Applied during attention computation within windows.
            training: Boolean or None, whether the layer is in training mode.
                Affects dropout in internal attention layer.

        Returns:
            Output sequence tensor of shape (batch_size, sequence_length, dim).
            Shape exactly matches input with all internal padding removed.
        """
        input_shape = keras.ops.shape(inputs)
        B, N_actual, C = input_shape[0], input_shape[1], input_shape[2]
        ws = self.window_size

        # --- SEQUENCE TO GRID CONVERSION ---
        # Determine grid size (nearest square ≥ sequence length)
        H = W = keras.ops.cast(
            keras.ops.ceil(
                keras.ops.sqrt(keras.ops.cast(N_actual, "float32"))
            ),
            "int32",
        )
        N_grid = H * W
        pad_len_seq = N_grid - N_actual

        # Pad sequence to form square grid
        padded_inputs = keras.ops.pad(
            inputs, [[0, 0], [0, pad_len_seq], [0, 0]]
        )
        grid = keras.ops.reshape(padded_inputs, (B, H, W, C))

        # Pad attention mask if provided
        padded_mask = None
        if attention_mask is not None:
            padded_mask = keras.ops.pad(
                attention_mask, [[0, 0], [0, pad_len_seq]]
            )

        # --- GRID PADDING FOR WINDOWING ---
        # Pad grid to make dimensions divisible by window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        padded_grid = keras.ops.pad(
            grid, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]]
        )
        H_pad, W_pad = H + pad_h, W + pad_w

        # --- WINDOW PARTITIONING ---
        # Partition grid into non-overlapping windows
        windows = self._window_partition(padded_grid)  # (B*num_win, ws, ws, C)
        windows = keras.ops.reshape(windows, (-1, ws * ws, C))  # Flatten windows

        # Process attention mask similarly
        attn_mask_for_windows = None
        if padded_mask is not None:
            mask_grid = keras.ops.reshape(padded_mask, (B, H, W, 1))
            padded_mask_grid = keras.ops.pad(
                mask_grid,
                [[0, 0], [0, pad_h], [0, pad_w], [0, 0]],
                constant_values=0,
            )
            mask_windows = self._window_partition(padded_mask_grid)
            attn_mask_for_windows = keras.ops.reshape(
                mask_windows, (-1, ws * ws)
            )

        # --- APPLY KAN ATTENTION WITHIN WINDOWS ---
        attn_windows = self.attention(
            windows, attention_mask=attn_mask_for_windows, training=training
        )

        # --- WINDOW MERGING AND RECONSTRUCTION ---
        # Reshape and merge windows back to grid
        attn_windows = keras.ops.reshape(attn_windows, (-1, ws, ws, C))
        reconstructed_grid = self._window_reverse(attn_windows, H_pad, W_pad)

        # Remove grid padding to get original grid size
        grid_unpadded = reconstructed_grid[:, :H, :W, :]

        # Reshape to sequence and remove sequence padding
        sequence_unpadded = keras.ops.reshape(grid_unpadded, (B, N_grid, C))
        output = sequence_unpadded[:, :N_actual, :]

        return output

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute output shape of the layer.

        Output shape is identical to input shape as sequence dimensions are preserved.

        Args:
            input_shape: Shape tuple of input tensor.

        Returns:
            Shape tuple identical to input_shape.
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration dictionary for layer serialization.

        Returns all constructor parameters needed to reconstruct the layer,
        enabling proper serialization and deserialization.

        Returns:
            Dictionary mapping constructor parameter names to their values.
        """
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "window_size": self.window_size,
                "num_heads": self.num_heads,
            }
        )
        # Add all kwargs passed to internal attention layer
        config.update(self.kwargs)
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "WindowAttentionKAN":
        """
        Create layer from configuration dictionary.

        Args:
            config: Configuration dictionary from get_config().

        Returns:
            New instance of KANWindowAttention with the same configuration.
        """
        config_copy = config.copy()
        dim = config_copy.pop("dim")
        window_size = config_copy.pop("window_size")
        num_heads = config_copy.pop("num_heads")
        return cls(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            **config_copy,
        )

# ---------------------------------------------------------------------
