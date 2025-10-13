"""
A unified multi-head attention with adaptive temperature.

This layer provides a versatile implementation of multi-head attention that
can operate in both self-attention and cross-attention modes. It extends
the standard mechanism with an optional adaptive temperature softmax, which
dynamically adjusts the sharpness of the attention distribution based on
the input, potentially improving model calibration and performance.

Architecture:
    The layer's architecture is designed for flexibility. It can function in
    two primary configurations determined by the inputs:

    1.  **Cross-Attention:** When provided with distinct `query` and `kv_input`
        tensors, it performs cross-attention. This is an asymmetric setup
        where a set of query vectors attends to a separate set of key-value
        pairs. This mode is fundamental to encoder-decoder models and
        architectures like Perceiver, where a small set of latent queries
        attends to a large set of input features.

    2.  **Self-Attention:** When only a single input tensor is provided, it
        performs self-attention. This is a symmetric setup where all
        tokens in a sequence attend to all other tokens. The layer offers a
        `shared_qk_projections` option for this mode, which uses a single
        projection matrix to generate Q, K, and V. This is a parameter-
        efficient variant suitable for standard transformer blocks.

Foundational Mathematics:
    The core of this layer is the scaled dot-product attention mechanism.
    However, it introduces a key enhancement in the normalization step. While
    standard attention uses a fixed-temperature softmax, this layer can
    employ an adaptive temperature `T` that is a function of the input:

        Attention(Q, K, V) = softmax( (Q @ K.T) / (sqrt(d_k) * T) ) @ V

    The adaptive temperature `T` is determined dynamically based on the
    entropy of the pre-softmax attention scores for each query. The
    intuition is to adjust the "sharpness" of the attention distribution:

    -   **High Entropy (Uniform Scores):** When a query has similar scores
        for many keys, the distribution is uncertain. The mechanism assigns a
        low temperature (`T < 1.0`), which sharpens the softmax output,
        forcing the model to make a more confident decision.
    -   **Low Entropy (Peaked Scores):** When a query is already highly
        confident, with one or a few keys having very high scores, the
        mechanism assigns a high temperature (`T > 1.0`). This softens the
        distribution, spreading the probability mass slightly and preventing
        overconfidence, which can improve model regularization.

    The temperature `T` is calculated by a function `f` that maps the
    normalized entropy `H` of the attention scores to a value within a
    predefined range `[min_temp, max_temp]`: `T = f(H)`.

References:
    - The scaled dot-product attention mechanism was introduced in:
      Vaswani, A., et al. (2017). "Attention Is All You Need".

    - The use of temperature to control the sharpness of a softmax is a
      well-established technique, famously used in knowledge distillation:
      Hinton, G., Vinyals, O., & Dean, J. (2015). "Distilling the
      Knowledge in a Neural Network".
"""

import keras
from keras import ops
from typing import Optional, Any, Dict, Tuple, Union, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..activations.adaptive_softmax import AdaptiveTemperatureSoftmax
from ..activations.routing_probabilities import RoutingProbabilitiesLayer

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class MultiHeadCrossAttention(keras.layers.Layer):
    """
    Unified, highly configurable multi-head attention layer with advanced features.

    This layer serves as a versatile attention mechanism supporting both cross-attention
    and self-attention modes with flexible projection strategies, comprehensive masking,
    and optional adaptive temperature softmax for enhanced attention normalization.
    It demonstrates modern Keras 3 best practices with robust serialization.

    **Intent**: Provide a production-ready, unified attention mechanism that can serve
    as both cross-attention (Perceiver, encoder-decoder) and self-attention (standard
    transformer) component with advanced features including adaptive temperature softmax,
    flexible masking, and parameter-efficient projection modes for diverse architectural
    requirements.

    **Architecture Modes**:

    1. **Cross-Attention Mode (separate projections)**:
    ```
    Query Input [B, Q_seq, D] ──→ Q_proj ──→ Q [B, H, Q_seq, D_h]
                                               ↓
    KV Input [B, KV_seq, D] ────→ KV_proj ──→ K, V [B, H, KV_seq, D_h]
                                               ↓
    Mask [B, Q_seq, KV_seq] ────→ Apply ────→ Masked Scores
                                               ↓
    AdaptiveSoftmax/Softmax ────→ Attention(Q, K, V) ──→ Output [B, Q_seq, D]
    ```

    2. **Self-Attention Mode (shared projections)**:
    ```
    Input [B, seq, D] ──→ QKV_proj ──→ Q, K, V [B, H, seq, D_h]
                           ↓
    Mask [B, seq, seq] ──→ Apply ────→ Masked Scores
                           ↓
    AdaptiveSoftmax/Softmax ──→ Attention(Q, K, V) ──→ Output [B, seq, D]
    ```

    **Mathematical Operations**:
    1. **Projections**: Q = X_q W_q, K = X_kv W_k, V = X_kv W_v
    2. **Attention Scores**: S = QK^T / √d_k
    3. **Masking**: S_masked = S + (1 - M) × (-1e9)
    4. **Normalization**: A = AdaptiveSoftmax(S_masked) or Softmax(S_masked)
    5. **Output**: O = proj(AV)

    **Advanced Features**:
    - **Adaptive Temperature Softmax**: Optional entropy-based dynamic temperature
    - **Flexible Masking**: Padding, full attention, and causal mask support
    - **Projection Modes**: Shared QKV (efficient) vs separate Q/KV (flexible)
    - **Robust Serialization**: Full compatibility with Keras save/load

    Args:
        dim: Integer, input/output dimension. Must be positive and divisible
            by num_heads.
        num_heads: Integer, number of attention heads. Must be positive.
            Defaults to 8.
        dropout_rate: Float, dropout rate for attention weights. Must be between
            0.0 and 1.0. Defaults to 0.0.
        shared_qk_projections: Boolean, if True, uses a single dense layer for
            Q, K, and V. Only valid for self-attention. Defaults to False.
        use_bias: Boolean, whether to use bias in linear projections.
            Defaults to True.
        kernel_initializer: String or Initializer for kernel weights.
            Defaults to "glorot_uniform".
        bias_initializer: String or Initializer for bias vectors.
            Defaults to "zeros".
        kernel_regularizer: Optional regularizer for kernel weights.
        bias_regularizer: Optional regularizer for bias weights.
        use_hierarchical_routing: Boolean, if True, uses hierarchical routing probability
            instead of standard softmax for attention normalization.
            Defaults to False.
        use_adaptive_softmax: Boolean, if True, uses AdaptiveTemperatureSoftmax
            instead of standard softmax for attention normalization.
            Defaults to False.
        adaptive_softmax_config: Optional dictionary of arguments for
            AdaptiveTemperatureSoftmax. Used only when `use_adaptive_softmax=True`.
            Defaults to None, which implies default values will be used.
            Expected keys are:
                - `min_temp` (float, default: 0.1): Minimum temperature.
                - `max_temp` (float, default: 1.0): Maximum temperature.
                - `entropy_threshold` (float, default: 0.5): Entropy threshold.
                - `polynomial_coeffs` (list[float], optional): Polynomial coefficients.
        **kwargs: Additional keyword arguments for the Layer base class.

    Call arguments:
        query_input: Query tensor of shape `(batch, query_seq_len, dim)`.
        kv_input: Optional Key-Value tensor of shape `(batch, kv_seq_len, dim)`.
            If `None`, self-attention is performed on `query_input`.
        attention_mask: Optional mask to prevent attention to certain positions.
            Supports shapes: `(batch, query_seq_len, kv_seq_len)` (full mask),
            `(batch, kv_seq_len)` (padding mask), or broadcastable shapes.
        training: Boolean indicating training or inference mode.

    Returns:
        Output tensor with shape `(batch_size, query_seq_len, dim)`.

    Raises:
        ValueError: If `dim` is not divisible by `num_heads`, or if
                    parameters are invalid.
        ValueError: If `shared_qk_projections=True` is used with `kv_input`.

    Example:
        ```python
        # Cross-Attention (Perceiver-style)
        visual_features = keras.random.normal((2, 196, 256))
        text_queries = keras.random.normal((2, 77, 256))
        cross_attn = MultiHeadCrossAttention(dim=256, num_heads=8)
        attended_text = cross_attn(text_queries, visual_features)
        print(attended_text.shape)  # (2, 77, 256)

        # Self-Attention with shared projections (parameter efficient)
        self_attn_shared = MultiHeadCrossAttention(
            dim=256, num_heads=8, shared_qk_projections=True
        )
        self_attended = self_attn_shared(visual_features)
        print(self_attended.shape)  # (2, 196, 256)

        # With adaptive temperature softmax for better normalization
        adaptive_attn = MultiHeadCrossAttention(
            dim=256,
            num_heads=8,
            use_adaptive_softmax=True,
            adaptive_softmax_config={"min_temp": 0.1, "max_temp": 2.0}
        )
        adaptive_output = adaptive_attn(text_queries, visual_features)

        # With attention masking
        padding_mask = ops.ones((2, 196))
        padding_mask = ops.slice_update(padding_mask, [0, 100], ops.zeros((2, 96)))
        masked_output = cross_attn(
            text_queries, visual_features, attention_mask=padding_mask
        )
        print(masked_output.shape)  # (2, 77, 256)
        ```
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            dropout_rate: float = 0.0,
            shared_qk_projections: bool = False,
            use_bias: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
            bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
            use_hierarchical_routing: bool = False,
            use_adaptive_softmax: bool = False,
            adaptive_softmax_config: Optional[Dict[str, Any]] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")

        # Store ALL configuration parameters
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = self.dim // num_heads
        self.dropout_rate = dropout_rate
        self.shared_qk_projections = shared_qk_projections
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.use_hierarchical_routing = use_hierarchical_routing
        self.use_adaptive_softmax = use_adaptive_softmax
        self.adaptive_softmax_config = adaptive_softmax_config

        # Adaptive temperature softmax configuration and validation
        if self.use_adaptive_softmax:
            if self.adaptive_softmax_config is None:
                self.adaptive_softmax_config = {}

            # Extract parameters with defaults for validation
            min_temp = self.adaptive_softmax_config.get("min_temp", 0.1)
            max_temp = self.adaptive_softmax_config.get("max_temp", 1.0)
            entropy_threshold = self.adaptive_softmax_config.get("entropy_threshold", 0.5)

            # Store resolved defaults back into the config for serialization
            self.adaptive_softmax_config["min_temp"] = min_temp
            self.adaptive_softmax_config["max_temp"] = max_temp
            self.adaptive_softmax_config["entropy_threshold"] = entropy_threshold

            # Validate the parameters
            if min_temp <= 0:
                raise ValueError(f"min_temp must be positive, got {min_temp}")
            if max_temp <= min_temp:
                raise ValueError(f"max_temp ({max_temp}) must be greater than min_temp ({min_temp})")
            if not (0.0 <= entropy_threshold <= 1.0):
                raise ValueError(f"entropy_threshold must be between 0 and 1, got {entropy_threshold}")
        else:
            self.adaptive_softmax_config = None

        # Scale factor for attention scores
        self.scale = 1.0 / ops.sqrt(float(self.head_dim))

        # CREATE sub-layers based on projection strategy
        dense_kwargs = {
            "use_bias": self.use_bias,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "bias_regularizer": self.bias_regularizer
        }

        if self.shared_qk_projections:
            self.qkv_dense = keras.layers.Dense(
                self.dim * 3, name="qkv", **dense_kwargs
            )
            self.q_dense, self.kv_dense = None, None
        else:
            self.q_dense = keras.layers.Dense(self.dim, name="q", **dense_kwargs)
            self.kv_dense = keras.layers.Dense(self.dim * 2, name="kv", **dense_kwargs)
            self.qkv_dense = None

        self.proj_dense = keras.layers.Dense(self.dim, name="proj", **dense_kwargs)
        self.dropout_layer = keras.layers.Dropout(
            self.dropout_rate, name="dropout"
        ) if self.dropout_rate > 0.0 else None

        if self.use_hierarchical_routing:
            self.hierarchical_routing = RoutingProbabilitiesLayer()
        else:
            self.hierarchical_routing = None

        # CREATE adaptive temperature softmax layer if enabled
        if self.use_adaptive_softmax:
            self.adaptive_softmax = AdaptiveTemperatureSoftmax(
                name="adaptive_softmax",
                **self.adaptive_softmax_config
            )
        else:
            self.adaptive_softmax = None

    def build(
            self,
            input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]
    ) -> None:
        """
        Build the layer by creating weight variables and building sub-layers.

        CRITICAL: Explicitly build each sub-layer for robust serialization.
        This ensures weight variables exist before weight restoration during loading.
        """
        # Robustly determine if input_shape is a list of shapes (cross-attention)
        # or a single shape (self-attention). This works across backends.
        is_list_of_shapes = (
            isinstance(input_shape, list) and
            len(input_shape) > 0 and
            not isinstance(input_shape[0], (int, type(None)))
        )

        if is_list_of_shapes:
            if len(input_shape) != 2:
                raise ValueError(f"Expected 2 inputs for cross-attention, got {len(input_shape)}")
            query_shape, kv_shape = input_shape
        else:
            query_shape = kv_shape = input_shape

        # Validate input shapes
        if len(query_shape) != 3:
            raise ValueError(f"Query input must be 3D, got shape {query_shape}")
        if query_shape[-1] is not None and query_shape[-1] != self.dim:
            raise ValueError(f"Query last dimension ({query_shape[-1]}) must match dim ({self.dim})")

        # Build projection layers explicitly for serialization
        if self.shared_qk_projections:
            self.qkv_dense.build(query_shape)
        else:
            self.q_dense.build(query_shape)
            self.kv_dense.build(kv_shape)

        # Build output projection layer
        proj_input_shape = (query_shape[0], query_shape[1], self.dim)
        self.proj_dense.build(proj_input_shape)

        # Build dropout layer if exists
        if self.dropout_layer is not None:
            # Dropout doesn't change shape, use attention weight shape for building
            attn_shape = (query_shape[0], self.num_heads, query_shape[1], kv_shape[1])
            self.dropout_layer.build(attn_shape)

        # Build hierarchical routing layer if exists
        if self.hierarchical_routing is not None:
            # AdaptiveTemperatureSoftmax can handle any shape, use attention weight shape
            attn_shape = (query_shape[0], self.num_heads, query_shape[1], kv_shape[1])
            self.hierarchical_routing.build(attn_shape)

        # Build adaptive softmax layer if exists
        if self.adaptive_softmax is not None:
            # AdaptiveTemperatureSoftmax can handle any shape, use attention weight shape
            attn_shape = (query_shape[0], self.num_heads, query_shape[1], kv_shape[1])
            self.adaptive_softmax.build(attn_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def _apply_attention_mask(
            self,
            scores: keras.KerasTensor,
            attention_mask: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Apply attention mask to scores tensor.

        Args:
            scores: Attention scores of shape (batch, num_heads, query_seq, kv_seq)
            attention_mask: Attention mask with supported shapes:
                - (batch, kv_seq): Padding mask
                - (batch, query_seq, kv_seq): Full attention mask
                - Other broadcastable shapes

        Returns:
            Masked scores tensor with same shape as input scores.
        """
        attention_mask = ops.cast(attention_mask, scores.dtype)

        # Expand mask dimensions to match scores shape (batch, num_heads, query_seq, kv_seq)
        if len(attention_mask.shape) == 2:  # Padding mask (batch, kv_seq)
            attention_mask = ops.expand_dims(ops.expand_dims(attention_mask, 1), 1)  # (batch, 1, 1, kv_seq)
        elif len(attention_mask.shape) == 3:  # Full mask (batch, query_seq, kv_seq)
            attention_mask = ops.expand_dims(attention_mask, 1)  # (batch, 1, query_seq, kv_seq)

        # Apply mask: set masked positions to large negative value
        mask_value = -1e9
        return scores + (1.0 - attention_mask) * mask_value

    def call(
            self,
            query_input: keras.KerasTensor,
            kv_input: Optional[keras.KerasTensor] = None,
            attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through multi-head attention with optional masking and adaptive softmax."""
        batch_size = ops.shape(query_input)[0]
        query_seq_len = ops.shape(query_input)[1]

        # Determine Q, K, V based on projection strategy
        if self.shared_qk_projections:
            if kv_input is not None:
                raise ValueError(
                    "When `shared_qk_projections=True`, `kv_input` must be None "
                    "(self-attention mode only)."
                )
            # Shared projection mode for self-attention
            qkv = self.qkv_dense(query_input)
            qkv = ops.reshape(qkv, (batch_size, query_seq_len, 3, self.num_heads, self.head_dim))
            qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            # Separate projection mode for cross-attention or self-attention
            kv_source = kv_input if kv_input is not None else query_input

            q = self.q_dense(query_input)
            q = ops.reshape(q, (batch_size, query_seq_len, self.num_heads, self.head_dim))
            q = ops.transpose(q, (0, 2, 1, 3))

            kv_seq_len = ops.shape(kv_source)[1]
            kv = self.kv_dense(kv_source)
            kv = ops.reshape(kv, (batch_size, kv_seq_len, 2, self.num_heads, self.head_dim))
            kv = ops.transpose(kv, (2, 0, 3, 1, 4))
            k, v = kv[0], kv[1]

        # Scaled dot-product attention
        scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * ops.cast(self.scale, q.dtype)

        # Apply attention mask if provided
        if attention_mask is not None:
            scores = self._apply_attention_mask(scores, attention_mask)

        # Compute attention weights using adaptive or standard softmax
        if self.use_adaptive_softmax and self.adaptive_softmax is not None:
            attn_weights = self.adaptive_softmax(scores)
        elif self.use_hierarchical_routing and self.hierarchical_routing is not None:
            attn_weights = self.hierarchical_routing(scores)
        else:
            attn_weights = ops.softmax(scores, axis=-1)

        # Apply dropout if configured
        if self.dropout_layer is not None:
            attn_weights = self.dropout_layer(attn_weights, training=training)

        # Apply attention to values and reshape output
        out = ops.matmul(attn_weights, v)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (batch_size, query_seq_len, self.dim))

        return self.proj_dense(out)

    def compute_output_shape(
            self,
            input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]
    ) -> Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]:
        """Compute output shape - returns query input shape."""
        is_list_of_shapes = (
            isinstance(input_shape, list) and
            len(input_shape) > 0 and
            not isinstance(input_shape[0], (int, type(None)))
        )

        if is_list_of_shapes:
            return input_shape[0]
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization - includes ALL constructor parameters."""
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
            "shared_qk_projections": self.shared_qk_projections,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
            "use_adaptive_softmax": self.use_adaptive_softmax,
            "adaptive_softmax_config": self.adaptive_softmax_config,
            "use_hierarchical_routing": self.use_hierarchical_routing,
        })
        return config

# ---------------------------------------------------------------------
