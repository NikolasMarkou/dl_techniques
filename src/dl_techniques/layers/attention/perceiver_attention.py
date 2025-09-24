"""Implements the asymmetric cross-attention from the Perceiver architecture.

    This layer is a specialized form of cross-attention that serves as the
    core building block of the Perceiver and Perceiver IO models. Its primary
    function is to create a scalable information bottleneck, enabling a deep
    transformer model to process very large and high-dimensional inputs (like
    images or audio) without incurring quadratic computational complexity.

    Architecture:
        The key architectural innovation is the decoupling of the main processing
        network's depth from the input data's size. This is achieved by
        introducing an asymmetric attention mechanism that operates between two
        distinct arrays:

        1.  **A small, fixed-size latent array (Queries):** This is typically a
            learnable embedding that acts as the network's internal state or
            "working memory." Its size is a hyperparameter that remains constant
            regardless of the input data size.

        2.  **A large, variable-size data array (Keys and Values):** This is
            derived directly from the high-dimensional input data (e.g., image
            patches, text tokens, audio samples).

        In each layer, the latent array forms the queries (`Q`) and "attends to"
        the data array, which provides the keys (`K`) and values (`V`). This
        forces the model to distill or "perceive" the most relevant information
        from the large input data and summarize it into the compact latent
        representation. Because the subsequent self-attention layers in a
        Perceiver model only operate on this small latent array, the overall
        computational cost becomes manageable and independent of the input size.

    Foundational Mathematics:
        While the underlying mechanism is the standard scaled dot-product
        attention, its application is asymmetric. Given a latent array `X_lat`
        (of size `N x D`) and a data array `X_data` (of size `M x C`), the
        computation is as follows:

            Q = X_lat @ W_q
            K = X_data @ W_k
            V = X_data @ W_v

            Attention(Q, K, V) = softmax( (Q @ K.T) / sqrt(d_k) ) @ V

        The resulting attention matrix has a shape of `(N, M)`, where `N` is the
        latent array size and `M` is the input data size. The computational
        complexity is O(N * M), which is a significant improvement over the
        O(M^2) complexity of applying self-attention directly to the large data
        array, especially when `N << M`. This mechanism effectively performs a
        set-to-set transformation, mapping a large, variable-sized input set to
        a small, fixed-sized latent set.

    References:
        - The primary architecture was introduced in:
          Jaegle, A., et al. (2021). "Perceiver: General Perception with
          Iterative Attention".

        - The concept was extended for structured outputs in:
          Jaegle, A., et al. (2021). "Perceiver IO: A General Architecture for
          Structured Inputs & Outputs".

        - The underlying attention mechanism is from:
          Vaswani, A., et al. (2017). "Attention Is All You Need".
"""

import keras
from typing import Optional, Any, Dict, Tuple, Union, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .multi_head_cross_attention import MultiHeadCrossAttention

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class PerceiverAttention(keras.layers.Layer):
    """
    Cross-attention mechanism from the Perceiver architecture with robust serialization.

    This layer implements cross-attention where queries and key-value pairs come from
    different sources, following the Perceiver architecture design. It demonstrates
    the wrapper pattern by leveraging the robust `MultiHeadCrossAttention` implementation
    while providing a specialized interface for Perceiver-style cross-modal processing.

    **Intent**: Provide a clean, specialized interface for Perceiver-style cross-attention
    that enables flexible cross-modal processing, set-to-set transformations, and latent
    space learning while internally leveraging the well-tested `MultiHeadCrossAttention`
    implementation for robust serialization and consistent behavior across attention layers.

    **Architecture**:
    ```
    Query Input [B, Q_seq, dim] ──→ MultiHeadCrossAttention
                                     ↓ (cross-attention mode)
    KV Input [B, KV_seq, dim] ────→ Q_proj, KV_proj (separate)
                                     ↓
    Cross-Attention(Q, K, V) ──────→ Output [B, Q_seq, dim]
    ```

    **Key Differences from Self-Attention**:
    - **Queries** come from one input (`query_input`)
    - **Keys and Values** come from another input (`kv_input`)
    - **Cross-modal capability**: Enables attention between different modalities
    - **Separate projections**: Uses `shared_qk_projections=False` for maximum flexibility

    **Wrapper Pattern Benefits**:
    - **Perceiver-specific interface**: Clean API focused on cross-attention use cases
    - **Robust implementation**: Leverages battle-tested `MultiHeadCrossAttention`
    - **Consistent behavior**: Same serialization and computation patterns
    - **Cross-modal optimized**: Separate projections for different input sources

    Args:
        dim: Integer, input/output dimension of the attention layer.
            Must be positive and divisible by num_heads.
        num_heads: Integer, number of attention heads. Must be positive
            and divide dim evenly. Defaults to 8.
        dropout_rate: Float, dropout rate for attention weights. Must be between
            0.0 and 1.0. Defaults to 0.0.
        use_bias: Boolean, whether to use bias in linear projections.
            Defaults to True.
        kernel_initializer: String or Initializer, initializer for the kernel weights.
            Defaults to "glorot_uniform".
        bias_initializer: String or Initializer, initializer for the bias vector.
            Defaults to "zeros".
        kernel_regularizer: Optional regularizer for kernel weights.
        bias_regularizer: Optional regularizer for bias weights.
        **kwargs: Additional keyword arguments for the Layer base class.

    Call arguments:
        query_input: Query tensor of shape (batch_size, query_seq_len, dim).
        kv_input: Optional Key-Value tensor of shape (batch_size, kv_seq_len, dim).
            If None, uses query_input for both (self-attention mode).
        training: Boolean indicating whether the layer should behave in training
            mode (applying dropout) or inference mode.

    Returns:
        Output tensor with same shape as query_input after cross-attention.

    Raises:
        ValueError: If dim is not divisible by num_heads.
        ValueError: If input shapes are invalid.
        ValueError: If parameters are out of valid ranges.

    Example:
        ```python
        # Cross-attention between different modalities
        visual_features = keras.random.normal((2, 196, 256))  # ViT patches
        text_features = keras.random.normal((2, 77, 256))    # Text tokens

        perceiver_attn = PerceiverAttention(dim=256, num_heads=8, dropout_rate=0.1)

        # Text attending to visual features
        text_to_visual = perceiver_attn(text_features, visual_features)
        print(text_to_visual.shape)  # (2, 77, 256)

        # Visual attending to text features
        visual_to_text = perceiver_attn(visual_features, text_features)
        print(visual_to_text.shape)  # (2, 196, 256)

        # Self-attention mode (single input)
        self_attended = perceiver_attn(visual_features)
        print(self_attended.shape)  # (2, 196, 256)

        # Cross-modal with regularization
        regularized_attn = PerceiverAttention(
            dim=512,
            num_heads=16,
            dropout_rate=0.2,
            kernel_regularizer=keras.regularizers.L2(1e-4)
        )

        # Latent bottleneck processing (Perceiver-style)
        latent_queries = keras.random.normal((2, 64, 256))    # Compact latent
        sensor_data = keras.random.normal((2, 1024, 256))    # High-dim input
        compressed = perceiver_attn(latent_queries, sensor_data)
        print(compressed.shape)  # (2, 64, 256) - Compressed representation
        ```

    Notes:
        This layer is particularly useful for:
        - Cross-modal attention (vision_heads-language, audio-visual)
        - Latent bottleneck architectures (Perceiver, Perceiver IO)
        - Set-to-set transformations with different cardinalities
        - Memory-augmented networks with external memory
        - Multimodal fusion where queries and context have different structures
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            dropout_rate: float = 0.0,
            use_bias: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
            bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
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
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # CREATE the underlying MultiHeadCrossAttention layer
        # Use shared_qk_projections=False for flexible cross-attention with separate projections
        self.cross_attention = MultiHeadCrossAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,  # Note: parameter name is 'dropout' in MultiHeadCrossAttention
            shared_qk_projections=False,  # Separate projections for cross-attention flexibility
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="cross_attention"
        )

    def build(
            self,
            input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]
    ) -> None:
        """
        Build the layer by creating weight variables and building sub-layers.

        CRITICAL: Explicitly build the wrapped MultiHeadCrossAttention for
        robust serialization. This ensures all weight variables exist before
        weight restoration during model loading.

        Args:
            input_shape: Shape of input tensor(s). Can be a single shape tuple
                or a list of two shape tuples for query and kv inputs.
        """
        # Handle different input formats
        if isinstance(input_shape, list):
            # Two separate inputs for cross-attention
            if len(input_shape) != 2:
                raise ValueError(f"Expected 2 inputs for cross-attention, got {len(input_shape)}")
            query_shape, kv_shape = input_shape
        else:
            # Single input shape (will be used for both query and kv in self-attention mode)
            query_shape = kv_shape = input_shape

        # Validate shapes
        if len(query_shape) != 3:
            raise ValueError(f"Query input must be 3D, got shape {query_shape}")
        if len(kv_shape) != 3:
            raise ValueError(f"KV input must be 3D, got shape {kv_shape}")
        if query_shape[-1] != self.dim:
            raise ValueError(f"Query last dimension ({query_shape[-1]}) must match dim ({self.dim})")
        if kv_shape[-1] != self.dim:
            raise ValueError(f"KV last dimension ({kv_shape[-1]}) must match dim ({self.dim})")

        # Build the wrapped cross-attention layer explicitly for serialization
        self.cross_attention.build(input_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
            self,
            query_input: keras.KerasTensor,
            kv_input: Optional[keras.KerasTensor] = None,
            attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply Perceiver cross-attention.

        This method delegates to the underlying MultiHeadCrossAttention layer
        with separate projections for maximum cross-modal flexibility.

        Args:
            query_input: Query tensor of shape (batch_size, query_seq_len, dim).
            kv_input: Key-Value tensor of shape (batch_size, kv_seq_len, dim).
                If None, uses query_input for both (self-attention mode).
            attention_mask: Optional attention mask of shape (batch_size, seq_len, seq_len)
                or (batch_size, 1, seq_len, seq_len). Values should be 1 for
                positions to attend to and 0 for masked positions.
            training: Boolean indicating training mode.

        Returns:
            Output tensor with same shape as query_input.
        """
        return self.cross_attention(
            query_input=query_input,
            kv_input=kv_input,  # Can be None for self-attention mode
            attention_mask=attention_mask,
            training=training
        )

    def compute_output_shape(
            self,
            input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape - same as query input shape.

        Args:
            input_shape: Input shape(s). Either single shape tuple or list of two shapes.

        Returns:
            Output shape tuple, same as query input shape.
        """
        if isinstance(input_shape, list):
            return input_shape[0]  # Same as query input shape
        else:
            return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration for serialization - includes ALL constructor parameters.

        Returns:
            Dictionary containing all layer configuration parameters.
        """
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------
