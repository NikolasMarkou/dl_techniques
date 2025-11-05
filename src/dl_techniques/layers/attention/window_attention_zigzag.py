"""
Windowed multi-head attention on zigzag-reordered sequences.

This module provides a specialized form of self-attention that first reorders
a sequence based on a two-dimensional zigzag scan before applying local,
windowed attention. The primary motivation is to induce a locality bias that
is sensitive to frequency-domain proximity, a concept borrowed from classical
image compression algorithms like JPEG.

Complete Architecture Flow::

    INPUT: 1D Sequence [batch, N, dim]
      │
      │  Step 1: Grid Reshaping & Padding
      ├──────────────────────────────────────────────┐
      │  Reshape to smallest square grid (H×W)       │
      │  Pad N → H×W if needed                       │
      └──────────────────────────────────────────────┘
      │
      ↓  2D Grid [batch, H, W, dim]
      │
      │  Step 2: Zigzag Reordering
      ├──────────────────────────────────────────────┐
      │  Follow zigzag scan path (anti-diagonals)    │
      │  Group frequency-proximate tokens            │
      └──────────────────────────────────────────────┘
      │
      ↓  Zigzag Sequence [batch, H×W, dim]
      │
      │  Step 3: Window Partitioning
      ├──────────────────────────────────────────────┐
      │  Split into non-overlapping windows          │
      │  Each window: [window_size²] tokens          │
      └──────────────────────────────────────────────┘
      │
      ↓  Windows [batch×num_windows, win_size², dim]
      │
      │  Step 4: Local Self-Attention
      ├──────────────────────────────────────────────┐
      │  Multi-head attention within each window     │
      │  Complexity: O(N × k²) vs O(N²)              │
      └──────────────────────────────────────────────┘
      │
      ↓  Attended Windows [batch×num_windows, win_size², dim]
      │
      │  Step 5: Inverse Zigzag
      ├──────────────────────────────────────────────┐
      │  Reorder back to 2D grid positions           │
      │  Restore original spatial arrangement        │
      └──────────────────────────────────────────────┘
      │
      ↓  Restored Grid [batch, H, W, dim]
      │
      │  Step 6: Flatten & Unpad
      ├──────────────────────────────────────────────┐
      │  Flatten to 1D sequence                      │
      │  Remove padding → original length N          │
      └──────────────────────────────────────────────┘
      │
      ↓
    OUTPUT: 1D Sequence [batch, N, dim]

Zigzag Scan Pattern::

    Example 4×4 Grid - Zigzag Order:
    ================================
    Grid Position:          Zigzag Index:
    ┌───┬───┬───┬───┐      ┌───┬───┬───┬───┐
    │0,0│0,1│0,2│0,3│      │ 0 │ 1 │ 5 │ 6 │
    ├───┼───┼───┼───┤      ├───┼───┼───┼───┤
    │1,0│1,1│1,2│1,3│      │ 2 │ 4 │ 7 │12 │
    ├───┼───┼───┼───┤  →   ├───┼───┼───┼───┤
    │2,0│2,1│2,2│2,3│      │ 3 │ 8 │11 │13 │
    ├───┼───┼───┼───┤      ├───┼───┼───┼───┤
    │3,0│3,1│3,2│3,3│      │ 9 │10 │14 │15 │
    └───┴───┴───┴───┘      └───┴───┴───┴───┘

    Zigzag Path (following anti-diagonals):
    ========================================
    Diagonal 0: (0,0)                           → index 0
    Diagonal 1: (1,0) → (0,1)                   → indices 1,2
    Diagonal 2: (0,2) → (1,1) → (2,0)          → indices 3,4,5
    Diagonal 3: (3,0) → (2,1) → (1,2) → (0,3)  → indices 6,7,8,9
    ...

    This groups tokens by frequency band (in DCT domain):
      - Low frequencies  (top-left)     → early in zigzag sequence
      - High frequencies (bottom-right) → late in zigzag sequence

Window Partitioning Example::

    Zigzag Sequence (16 tokens) with window_size=2 (4 tokens per window):
    =====================================================================
    Original Zigzag:  [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15]
                        └──────┘ └──────┘ └──────┘ └──────┘
    Window 0:         [ 0  1  2  3]        ← tokens 0-3
    Window 1:         [ 4  5  6  7]        ← tokens 4-7
    Window 2:         [ 8  9 10 11]        ← tokens 8-11
    Window 3:         [12 13 14 15]        ← tokens 12-15

    Each window attends to itself independently:
      Attention(Q, K, V) within window only
      No cross-window attention → linear complexity O(N)

Foundational Mathematics
------------------------

**Zigzag Scan**:
    This technique linearizes a 2D data grid by tracing a path that
    prioritizes elements based on the sum of their coordinates
    (i.e., row + col). In signal processing, particularly with
    transformations like the Discrete Cosine Transform (DCT), this
    ordering groups low-frequency coefficients (top-left of the grid)
    before high-frequency coefficients (bottom-right). By applying
    attention to windows of this reordered sequence, the model can
    focus on relationships between components of similar frequency bands.

**Windowed Attention**:
    To maintain computational tractability, attention is not computed
    across the entire sequence. Instead, following the paradigm of models
    like the Swin Transformer, the attention mechanism is constrained to
    local windows. This reduces the complexity from O(N²) to O(N × k²),
    where k is the window size.

**Advanced Normalization (Optional)**:
    The layer can replace the standard softmax function for attention
    weight computation with two alternatives:

    1. **Adaptive Temperature Softmax**: This method addresses potential
       model over-confidence by dynamically scaling the logits before
       the softmax operation. The temperature τ is adjusted based on
       the entropy of the attention distribution, leading to better
       calibrated models. A higher temperature (softer distribution)
       is used for uncertain, high-entropy scores, while a lower
       temperature (sharper distribution) is used for confident,
       low-entropy scores.

    2. **Hierarchical Routing**: As a parameter-free alternative to
       softmax, this deterministic method computes attention probabilities
       by routing a total probability mass of 1.0 through a fixed binary
       tree. At each node, incoming mass is split between its two children
       based on their relative activation scores. The final attention
       weights are the accumulated mass at the leaf nodes, providing a
       sparse and computationally distinct alternative to the standard
       exponential function.

Complexity Analysis
-------------------

============== ============== =================================
Operation      Complexity     Notes
============== ============== =================================
Zigzag Scan    O(N)           One-time index computation
Window Split   O(N)           Linear partitioning
Attention      O(N × k²)      k = window_size, N = seq_len
Total          O(N × k²)      Linear in sequence length
============== ============== =================================

Compare to standard attention: O(N²)

References
----------
- **Windowed Attention**: Liu et al., "Swin Transformer: Hierarchical
  Vision Transformer using Shifted Windows" (2021).
  https://arxiv.org/abs/2103.14030
- **Zigzag Scan**: A foundational concept in the JPEG image compression
  standard.
- **Adaptive Temperature Softmax**: Ding et al., "Adaptive Temperature
  Scaling for Better Calibrated and More Accurate Models" (2021).
  https://arxiv.org/abs/2102.10599
- **Hierarchical Routing**: Hassani, "NOMAD: N-th Order-of-Magnitude
  -Attention" (2022). https://arxiv.org/abs/2205.13233
"""

import math
import keras
from typing import Any, Dict, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..activations.adaptive_softmax import AdaptiveTemperatureSoftmax
from ..activations.routing_probabilities import RoutingProbabilitiesLayer


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class _CoreAttention(keras.layers.Layer):
    """
    Core multi-head self-attention for a single window (internal use).

    This internal layer implements standard multi-head self-attention with
    optional advanced normalization techniques. It is designed to process
    a single window of tokens independently, making it the computational
    core of the windowed attention mechanism.

    **Architecture**::

        Input [batch×windows, tokens, dim]
               ↓
        QKV Projection [→ Q, K, V]
               ↓
        Scaled Dot-Product Attention
               ↓
        Optional: Adaptive Softmax / Hierarchical Routing
               ↓
        Output Projection
               ↓
        Output [batch×windows, tokens, dim]

    **Attention Computation**::

        Q, K, V = Linear(input)  # Shape: [B, num_heads, N, head_dim]
        scores = Q @ K^T / √d    # Shape: [B, num_heads, N, N]
        weights = softmax(scores) # Or adaptive/routing alternative
        output = weights @ V      # Shape: [B, num_heads, N, head_dim]

    :param dim: Total embedding dimension. Must be divisible by num_heads.
    :type dim: int
    :param window_size: Size of the attention window (tokens per side).
        Total tokens in window = window_size².
    :type window_size: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param qkv_bias: Whether to use bias in QKV projection. Default: True.
    :type qkv_bias: bool
    :param qk_scale: Optional scaling factor for attention scores. If None,
        uses 1/√(head_dim). Default: None.
    :type qk_scale: Optional[float]
    :param dropout_rate: Dropout rate for attention weights. Default: 0.0.
    :type dropout_rate: float
    :param proj_bias: Whether to use bias in output projection. Default: True.
    :type proj_bias: bool
    :param use_hierarchical_routing: Use routing-based attention instead of
        softmax. Cannot be True if use_adaptive_softmax is True. Default: False.
    :type use_hierarchical_routing: bool
    :param use_adaptive_softmax: Use adaptive temperature softmax. Cannot be
        True if use_hierarchical_routing is True. Default: False.
    :type use_adaptive_softmax: bool
    :param adaptive_softmax_config: Configuration for adaptive softmax.
        Must include 'min_temp' and 'max_temp'. Default: None.
    :type adaptive_softmax_config: Optional[Dict[str, Any]]
    :param kernel_initializer: Initializer for weight matrices.
        Default: 'glorot_uniform'.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for bias vectors. Default: 'zeros'.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Regularizer for weight matrices. Default: None.
    :type kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param bias_regularizer: Regularizer for bias vectors. Default: None.
    :type bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param kwargs: Additional layer arguments.
    :type kwargs: Any

    :raises ValueError: If dim is not positive.
    :raises ValueError: If window_size is not positive.
    :raises ValueError: If num_heads is not positive.
    :raises ValueError: If dim is not divisible by num_heads.
    :raises ValueError: If dropout_rate is not in [0, 1].
    :raises ValueError: If both use_adaptive_softmax and
        use_hierarchical_routing are True.
    :raises ValueError: If adaptive_softmax_config has invalid temperature
        range.
    """

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        dropout_rate: float = 0.0,
        proj_bias: bool = True,
        use_hierarchical_routing: bool = False,
        use_adaptive_softmax: bool = False,
        adaptive_softmax_config: Optional[Dict[str, Any]] = None,
        kernel_initializer: Union[
            str, keras.initializers.Initializer
        ] = "glorot_uniform",
        bias_initializer: Union[
            str, keras.initializers.Initializer
        ] = "zeros",
        kernel_regularizer: Optional[
            Union[str, keras.regularizers.Regularizer]
        ] = None,
        bias_regularizer: Optional[
            Union[str, keras.regularizers.Regularizer]
        ] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the core attention layer.

        :param dim: Total embedding dimension.
        :type dim: int
        :param window_size: Size of attention window.
        :type window_size: int
        :param num_heads: Number of attention heads.
        :type num_heads: int
        :param qkv_bias: Use bias in QKV projection.
        :type qkv_bias: bool
        :param qk_scale: Optional attention score scaling.
        :type qk_scale: Optional[float]
        :param dropout_rate: Attention dropout rate.
        :type dropout_rate: float
        :param proj_bias: Use bias in output projection.
        :type proj_bias: bool
        :param use_hierarchical_routing: Use routing-based attention.
        :type use_hierarchical_routing: bool
        :param use_adaptive_softmax: Use adaptive temperature softmax.
        :type use_adaptive_softmax: bool
        :param adaptive_softmax_config: Adaptive softmax configuration.
        :type adaptive_softmax_config: Optional[Dict[str, Any]]
        :param kernel_initializer: Weight initializer.
        :type kernel_initializer: Union[str, keras.initializers.Initializer]
        :param bias_initializer: Bias initializer.
        :type bias_initializer: Union[str, keras.initializers.Initializer]
        :param kernel_regularizer: Weight regularizer.
        :type kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
        :param bias_regularizer: Bias regularizer.
        :type bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
        :param kwargs: Additional layer arguments.
        :type kwargs: Any
        """
        super().__init__(**kwargs)

        # Validate parameters
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if window_size <= 0:
            raise ValueError(
                f"window_size must be positive, got {window_size}"
            )
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})"
            )
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(
                f"attn_dropout_rate must be in [0, 1], got {dropout_rate}"
            )
        if use_adaptive_softmax and use_hierarchical_routing:
            raise ValueError(
                "Only one of `use_adaptive_softmax` or "
                "`use_hierarchical_routing` can be True."
            )

        # Store configuration
        self.dim = dim
        self.window_size = window_size
        self.num_tokens_in_window = self.window_size * self.window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = (qk_scale if qk_scale is not None
                      else self.head_dim**-0.5)
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.proj_bias = proj_bias
        self.dropout_rate = dropout_rate
        self.use_hierarchical_routing = use_hierarchical_routing
        self.use_adaptive_softmax = use_adaptive_softmax
        self.adaptive_softmax_config = adaptive_softmax_config

        # Store initializers and regularizers
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # Validate adaptive softmax configuration
        if self.use_adaptive_softmax:
            config = adaptive_softmax_config or {}
            min_temp = config.setdefault("min_temp", 0.1)
            max_temp = config.setdefault("max_temp", 1.0)
            if min_temp <= 0 or max_temp <= min_temp:
                raise ValueError(
                    "Invalid adaptive softmax temperature range."
                )
            self.adaptive_softmax_config = config
        else:
            self.adaptive_softmax_config = None

        # Prepare dense layer kwargs
        dense_kwargs = {
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "bias_regularizer": self.bias_regularizer,
        }

        # Create sub-layers
        self.qkv = keras.layers.Dense(
            self.dim * 3, use_bias=self.qkv_bias, name="qkv", **dense_kwargs
        )
        self.proj = keras.layers.Dense(
            self.dim, use_bias=self.proj_bias, name="proj", **dense_kwargs
        )
        self.attn_dropout = (
            keras.layers.Dropout(self.dropout_rate, name="attn_dropout")
            if self.dropout_rate > 0.0
            else None
        )
        self.hierarchical_routing = (
            RoutingProbabilitiesLayer(axis=-1, name="routing_probs")
            if self.use_hierarchical_routing
            else None
        )
        self.adaptive_softmax = (
            AdaptiveTemperatureSoftmax(
                name="adaptive_softmax", **(self.adaptive_softmax_config or {})
            )
            if self.use_adaptive_softmax
            else None
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer's weights.

        Creates the QKV projection, output projection, and optional
        normalization layers with proper shape inference.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        """
        # Build with padded shape (full window size)
        padded_shape = list(input_shape)
        padded_shape[1] = self.num_tokens_in_window
        padded_shape = tuple(padded_shape)

        # Build QKV and output projection
        self.qkv.build(padded_shape)
        self.proj.build(padded_shape)

        # Build dropout
        if self.attn_dropout is not None:
            self.attn_dropout.build(None)

        # Build normalization layers
        # Attention scores shape: (batch, num_heads, seq_len, seq_len)
        attn_scores_shape = (
            input_shape[0], self.num_heads, padded_shape[1], padded_shape[1]
        )
        if self.hierarchical_routing is not None:
            self.hierarchical_routing.build(attn_scores_shape)
        if self.adaptive_softmax is not None:
            self.adaptive_softmax.build(attn_scores_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply multi-head self-attention to input tokens.

        **Processing Steps**::

            1. Pad input to window size
            2. QKV projection and reshape
            3. Compute scaled dot-product attention scores
            4. Apply attention mask (if provided)
            5. Normalize scores (softmax/adaptive/routing)
            6. Apply attention to values
            7. Output projection
            8. Remove padding

        **Attention Mask Handling**::

            Internal Padding Mask:
              [1, 1, 1, ..., 1, 0, 0, 0]  ← valid=1, padded=0
              └──────N_actual──┘ └─pad─┘

            Combined with User Mask:
              final_mask = user_mask × padding_mask

            Applied to Scores:
              scores = scores + (1 - mask) × (-inf)

        :param inputs: Input tensor of shape (batch, seq_len, dim).
            May have seq_len < window_size² due to dynamic sequence lengths.
        :type inputs: keras.KerasTensor
        :param attention_mask: Optional mask of shape (batch, seq_len).
            Values should be 0 (masked) or 1 (valid). Default: None.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Whether in training mode. Used for dropout.
            Default: None.
        :type training: Optional[bool]
        :return: Output tensor of shape (batch, seq_len, dim).
        :rtype: keras.KerasTensor
        """
        # Get actual input dimensions
        input_shape = keras.ops.shape(inputs)
        B_actual, N_actual, C_actual = (
            input_shape[0], input_shape[1], input_shape[2]
        )

        # Pad to window size
        padding_amount = self.num_tokens_in_window - N_actual
        padded_inputs = keras.ops.pad(
            inputs, [[0, 0], [0, padding_amount], [0, 0]]
        )

        # Create internal padding mask
        # Shape: (batch, seq_len) with 1=valid, 0=padded
        internal_padding_mask = keras.ops.concatenate(
            [
                keras.ops.ones((B_actual, N_actual), dtype="int32"),
                keras.ops.zeros((B_actual, padding_amount), dtype="int32")
            ],
            axis=1
        )
        final_attention_mask = internal_padding_mask

        # Combine with user-provided mask if present
        if attention_mask is not None:
            mask_padding = keras.ops.zeros(
                (B_actual, padding_amount), dtype=attention_mask.dtype
            )
            padded_user_mask = keras.ops.concatenate(
                [attention_mask, mask_padding], axis=1
            )
            final_attention_mask = (
                keras.ops.cast(padded_user_mask, "int32") *
                internal_padding_mask
            )

        # Get padded dimensions
        B, N, C = keras.ops.shape(padded_inputs)

        # QKV projection and reshape
        # Shape: (B, N, 3*dim) → (B, N, 3, num_heads, head_dim)
        qkv = self.qkv(padded_inputs, training=training)
        qkv = keras.ops.reshape(
            qkv, (B, N, 3, self.num_heads, self.head_dim)
        )
        # Transpose to (3, B, num_heads, N, head_dim)
        qkv = keras.ops.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scale queries for attention computation
        # Q @ K^T / √d
        q = q * self.scale

        # Compute attention scores
        # Shape: (B, num_heads, N, N)
        attn_scores = keras.ops.matmul(
            q, keras.ops.transpose(k, (0, 1, 3, 2))
        )

        # Apply attention mask if present
        # Mask out padded positions by adding -inf to scores
        if final_attention_mask is not None:
            # Reshape mask for broadcasting: (B, 1, 1, N)
            broadcast_mask = keras.ops.reshape(
                final_attention_mask, (B, 1, 1, N)
            )
            # Create additive mask: 0 for valid, -inf for masked
            inf_value = keras.ops.convert_to_tensor(
                -1e9, dtype=attn_scores.dtype
            )
            additive_mask = (
                1.0 - keras.ops.cast(broadcast_mask, dtype=attn_scores.dtype)
            ) * inf_value
            attn_scores = attn_scores + additive_mask

        # Normalize attention scores
        # Three options: adaptive softmax, routing, or standard softmax
        if self.use_adaptive_softmax:
            attn_weights = self.adaptive_softmax(
                attn_scores, training=training
            )
        elif self.use_hierarchical_routing:
            attn_weights = self.hierarchical_routing(
                attn_scores, training=training
            )
        else:
            attn_weights = keras.ops.softmax(attn_scores, axis=-1)

        # Apply dropout to attention weights
        if self.attn_dropout is not None:
            attn_weights = self.attn_dropout(attn_weights, training=training)

        # Apply attention to values
        # Shape: (B, num_heads, N, head_dim)
        x = keras.ops.matmul(attn_weights, v)

        # Transpose and reshape back to (B, N, dim)
        x = keras.ops.transpose(x, (0, 2, 1, 3))
        x = keras.ops.reshape(x, (B, N, C))

        # Output projection
        x = self.proj(x, training=training)

        # Remove padding to return original sequence length
        return x[:, :N_actual, :]

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute output shape (same as input shape).

        :param input_shape: Shape tuple of input.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration for serialization.

        :return: Layer configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "window_size": self.window_size,
            "num_heads": self.num_heads,
            "qkv_bias": self.qkv_bias,
            "qk_scale": self.qk_scale,
            "dropout_rate": self.dropout_rate,
            "proj_bias": self.proj_bias,
            "use_hierarchical_routing": self.use_hierarchical_routing,
            "use_adaptive_softmax": self.use_adaptive_softmax,
            "adaptive_softmax_config": self.adaptive_softmax_config,
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
        })
        return config


@keras.saving.register_keras_serializable()
class WindowZigZagAttention(keras.layers.Layer):
    """
    Windowed multi-head attention with zigzag sequence reordering.

    This layer implements efficient self-attention by first reordering the
    input sequence along a 2D zigzag path, then applying windowed attention
    to the reordered sequence. This approach groups frequency-proximate
    tokens together while maintaining linear computational complexity.

    **Key Features**:

    - **Zigzag Reordering**: Groups tokens by frequency band (DCT-inspired)
    - **Windowed Attention**: O(N × k²) complexity vs O(N²) standard attention
    - **XLA Compatible**: Uses fixed sequence lengths and ops.take for
      gathering
    - **Frequency Locality**: Attention windows focus on similar frequency
      components

    **Processing Overview**::

        Input [B, N, D]
          ↓ pad to H×W grid
        Grid [B, H×W, D]
          ↓ zigzag reorder
        Zigzag Seq [B, H×W, D]
          ↓ split into windows
        Windows [B×num_win, win_size², D]
          ↓ self-attention per window
        Attended [B×num_win, win_size², D]
          ↓ merge & inverse zigzag
        Grid [B, H×W, D]
          ↓ unpad
        Output [B, N, D]

    **Example: 5-token sequence with window_size=2**::

        Step 1: Pad to 9 (3×3 grid)
        [t0, t1, t2, t3, t4] → [t0, t1, t2, t3, t4, p0, p1, p2, p3]

        Step 2: Reshape to 3×3 grid
        ┌────┬────┬────┐
        │ t0 │ t1 │ t2 │
        ├────┼────┼────┤
        │ t3 │ t4 │ p0 │
        ├────┼────┼────┤
        │ p1 │ p2 │ p3 │
        └────┴────┴────┘

        Step 3: Zigzag reorder (following diagonals)
        [t0, t1, t3, t2, t4, p1, p0, p2, p3]
         └─┘  └──────┘  └──────┘  └──────┘
         Win0    Win1      Win2      Win3
         (padded to 4 tokens per window)

        Step 4: Attention within each window
        Step 5: Inverse zigzag
        Step 6: Unpad to original length

    :param dim: Embedding dimension. Must be divisible by num_heads.
    :type dim: int
    :param window_size: Size of attention window (tokens per side).
        Actual window contains window_size² tokens.
    :type window_size: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param kwargs: Additional arguments passed to _CoreAttention
        (qkv_bias, dropout_rate, use_adaptive_softmax, etc.).
    :type kwargs: Any

    Examples::

        >>> # Basic usage
        >>> layer = WindowZigZagAttention(dim=256, window_size=4, num_heads=8)
        >>> inputs = keras.Input(shape=(100, 256))
        >>> outputs = layer(inputs)  # Shape: (batch, 100, 256)
        >>>
        >>> # With adaptive softmax
        >>> layer = WindowZigZagAttention(
        ...     dim=256,
        ...     window_size=4,
        ...     num_heads=8,
        ...     use_adaptive_softmax=True,
        ...     adaptive_softmax_config={'min_temp': 0.1, 'max_temp': 2.0}
        ... )

    Note:
        Requires fixed sequence length for XLA compatibility. Dynamic
        sequence lengths are not supported.
    """

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        **kwargs: Any
    ):
        """
        Initialize the windowed zigzag attention layer.

        :param dim: Embedding dimension.
        :type dim: int
        :param window_size: Window size (tokens per side).
        :type window_size: int
        :param num_heads: Number of attention heads.
        :type num_heads: int
        :param kwargs: Additional CoreAttention arguments.
        :type kwargs: Any
        """
        super().__init__(name=kwargs.pop("name", "window_zigzag_attention"))

        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.num_tokens_in_window = window_size * window_size
        self.kwargs = kwargs

        # Create core attention layer
        self.attention = _CoreAttention(
            dim=dim, window_size=window_size, num_heads=num_heads, **kwargs
        )

        # Grid dimensions (computed in build())
        self.H = None
        self.W = None
        self.N_grid = None
        self.pad_len_seq = None

        # Zigzag indices (computed in build())
        self.zigzag_indices = None
        self.inverse_zigzag_indices = None

    @staticmethod
    def _generate_zigzag_indices(
        H: keras.KerasTensor, W: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Generate zigzag scan indices for an H×W grid.

        The zigzag scan follows anti-diagonals, ordering elements by the
        sum of their coordinates (row + col). Within each diagonal, elements
        are ordered to create the characteristic zigzag pattern.

        **Zigzag Pattern Example (4×4)**::

            Grid Positions:     Zigzag Order:
            ┌────────────┐     ┌────────────┐
            │ 0  1  2  3 │     │ 0  1  5  6 │
            │ 4  5  6  7 │     │ 2  4  7 12 │
            │ 8  9 10 11 │ →   │ 3  8 11 13 │
            │12 13 14 15 │     │ 9 10 14 15 │
            └────────────┘     └────────────┘

            Sorting key: (row + col) * H + secondary_key
            where secondary_key alternates direction per diagonal

        :param H: Grid height.
        :type H: keras.KerasTensor
        :param W: Grid width.
        :type W: keras.KerasTensor
        :return: Zigzag indices of shape (H×W,).
        :rtype: keras.KerasTensor
        """
        # Create coordinate grids
        r_coords = keras.ops.arange(0, H, dtype="int32")
        c_coords = keras.ops.arange(0, W, dtype="int32")
        r_grid, c_grid = keras.ops.meshgrid(r_coords, c_coords, indexing="ij")

        # Flatten to 1D
        r_flat = keras.ops.reshape(r_grid, (-1,))
        c_flat = keras.ops.reshape(c_grid, (-1,))

        # Primary sort key: sum of coordinates (diagonal number)
        s = r_flat + c_flat

        # Secondary sort key: alternate direction per diagonal
        # Even diagonals: ascending row
        # Odd diagonals: descending row
        secondary_key = keras.ops.where(
            s % 2 == 1, r_flat, H - 1 - r_flat
        )

        # Combined sorting key
        combined_key = s * H + secondary_key

        # Sort to get zigzag order
        zigzag_indices = keras.ops.argsort(combined_key)

        return zigzag_indices

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build layer and precompute zigzag indices.

        Computes the grid dimensions (smallest square H×W >= N) and
        precomputes the zigzag and inverse zigzag indices for XLA
        compatibility.

        **Grid Sizing Example**::

            N=10 tokens → H=4, W=4, grid_size=16
            N=16 tokens → H=4, W=4, grid_size=16
            N=17 tokens → H=5, W=5, grid_size=25
            N=100 tokens → H=10, W=10, grid_size=100

            Formula: H = ceil(√N), W = H

        :param input_shape: Shape tuple of input (batch, seq_len, dim).
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If sequence length is None (required for XLA).
        """
        N_actual = input_shape[1]
        if N_actual is None:
            raise ValueError(
                "WindowZigZagAttention requires a fixed sequence length for "
                "XLA compatibility. Received `None` for the sequence dimension."
            )

        # Compute grid dimensions: smallest square grid >= N_actual
        self.H = int(math.ceil(math.sqrt(N_actual)))
        self.W = self.H
        self.N_grid = self.H * self.W
        self.pad_len_seq = self.N_grid - N_actual

        # Generate zigzag indices
        H_tensor = keras.ops.convert_to_tensor(self.H, dtype="int32")
        W_tensor = keras.ops.convert_to_tensor(self.W, dtype="int32")
        self.zigzag_indices = self._generate_zigzag_indices(
            H_tensor, W_tensor
        )

        # Compute inverse zigzag indices (for reverse transformation)
        self.inverse_zigzag_indices = keras.ops.argsort(self.zigzag_indices)

        # Build core attention layer
        self.attention.build((None, self.num_tokens_in_window, self.dim))

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply windowed zigzag attention to inputs.

        **Processing Steps**::

            1. Pad sequence to grid size (N → H×W)
            2. Apply zigzag reordering
            3. Pad zigzag sequence to window boundaries
            4. Split into windows
            5. Apply attention within each window
            6. Merge windows
            7. Apply inverse zigzag reordering
            8. Unpad to original sequence length

        **Shape Transformations Example** (N=10, window_size=2)::

            Input:           (B, 10, D)
            ↓ pad to 4×4
            Grid:            (B, 16, D)
            ↓ zigzag
            Zigzag:          (B, 16, D)
            ↓ pad to 4 windows
            Windowed:        (B, 16, D)  [already multiple of 4]
            ↓ reshape
            Windows:         (B×4, 4, D)  [4 windows of 4 tokens]
            ↓ attention
            Attended:        (B×4, 4, D)
            ↓ reshape
            Merged:          (B, 16, D)
            ↓ inverse zigzag
            Grid:            (B, 16, D)
            ↓ unpad
            Output:          (B, 10, D)

        :param inputs: Input tensor of shape (batch, seq_len, dim).
        :type inputs: keras.KerasTensor
        :param attention_mask: Optional mask of shape (batch, seq_len).
            Values should be 0 (masked) or 1 (valid). Default: None.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Whether in training mode. Default: None.
        :type training: Optional[bool]
        :return: Output tensor of shape (batch, seq_len, dim).
        :rtype: keras.KerasTensor
        """
        # Get input dimensions
        input_shape = keras.ops.shape(inputs)
        B, N_actual, C = input_shape[0], input_shape[1], input_shape[2]
        win_len = self.num_tokens_in_window

        # Step 1: Pad sequence to grid size (N → H×W)
        # ============================================
        # Example: N=10 → H=4, W=4, grid_size=16, pad_len=6
        # [t0, t1, ..., t9] → [t0, t1, ..., t9, p0, p1, ..., p5]
        padded_inputs = keras.ops.pad(
            inputs, [[0, 0], [0, self.pad_len_seq], [0, 0]]
        )

        # Step 2: Apply zigzag reordering
        # ================================
        # Reorder tokens according to precomputed zigzag indices
        # Use keras.ops.take for XLA compatibility
        # Shape: (B, H×W, D)
        zigzag_sequence = keras.ops.take(
            padded_inputs, self.zigzag_indices, axis=1
        )

        # Reorder attention mask if provided
        zigzag_mask = None
        if attention_mask is not None:
            padded_mask = keras.ops.pad(
                attention_mask, [[0, 0], [0, self.pad_len_seq]]
            )
            zigzag_mask = keras.ops.take(
                padded_mask, self.zigzag_indices, axis=1
            )

        # Step 3: Pad zigzag sequence to window boundaries
        # ================================================
        # Ensure sequence length is multiple of window size
        # Example: N_grid=16, win_len=4 → no padding needed
        # Example: N_grid=10, win_len=4 → pad 2 tokens
        pad_len_win = (win_len - (self.N_grid % win_len)) % win_len
        padded_zigzag_seq = keras.ops.pad(
            zigzag_sequence, [[0, 0], [0, pad_len_win], [0, 0]]
        )

        # Step 4: Split into windows
        # ===========================
        # Reshape to (B × num_windows, win_len, D)
        # Example: (B, 16, D) → (B×4, 4, D) for 4 windows of 4 tokens
        num_windows = (self.N_grid + pad_len_win) // win_len
        windows = keras.ops.reshape(
            padded_zigzag_seq, (B * num_windows, win_len, C)
        )

        # Reshape attention mask for windows
        attn_mask_for_windows = None
        if zigzag_mask is not None:
            padded_zigzag_mask = keras.ops.pad(
                zigzag_mask, [[0, 0], [0, pad_len_win]], constant_values=0
            )
            attn_mask_for_windows = keras.ops.reshape(
                padded_zigzag_mask, (B * num_windows, win_len)
            )

        # Step 5: Apply attention within each window
        # ===========================================
        # Independent self-attention per window
        # Shape: (B × num_windows, win_len, D)
        attn_windows = self.attention(
            windows, attention_mask=attn_mask_for_windows, training=training
        )

        # Step 6: Merge windows back into sequence
        # =========================================
        # Reshape to (B, num_windows × win_len, D)
        merged_zigzag_seq = keras.ops.reshape(
            attn_windows, (B, num_windows * win_len, C)
        )

        # Remove window padding
        unpadded_zigzag_seq = merged_zigzag_seq[:, :self.N_grid, :]

        # Step 7: Apply inverse zigzag reordering
        # ========================================
        # Restore original 2D grid layout
        # Use keras.ops.take for XLA compatibility
        sequence_unpadded = keras.ops.take(
            unpadded_zigzag_seq, self.inverse_zigzag_indices, axis=1
        )

        # Step 8: Remove sequence padding
        # ================================
        # Restore original sequence length N_actual
        output = sequence_unpadded[:, :N_actual, :]

        return output

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute output shape (same as input shape).

        :param input_shape: Shape tuple of input.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration for serialization.

        Combines this layer's configuration with the core attention layer's
        configuration, removing duplicate and internal keys.

        :return: Layer configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "window_size": self.window_size,
            "num_heads": self.num_heads,
        })

        # Get inner attention configuration
        inner_config = self.attention.get_config()

        # Remove keys that are already in outer config or are internal
        for key in ["name", "trainable", "dtype", "dim", "window_size",
                    "num_heads"]:
            inner_config.pop(key, None)

        # Merge configurations
        config.update(inner_config)

        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "WindowZigZagAttention":
        """
        Create layer from configuration dictionary.

        :param config: Configuration dictionary from get_config().
        :type config: Dict[str, Any]
        :return: New WindowZigZagAttention instance.
        :rtype: WindowZigZagAttention
        """
        config_copy = config.copy()

        # Extract required arguments
        dim = config_copy.pop("dim")
        window_size = config_copy.pop("window_size")
        num_heads = config_copy.pop("num_heads")

        # Remaining config items are kwargs for _CoreAttention
        return cls(
            dim=dim, window_size=window_size, num_heads=num_heads,
            **config_copy
        )