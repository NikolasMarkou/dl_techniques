"""
A post-normalization Transformer block with SwiGLU and RMSNorm.

This layer constitutes a single block within a Transformer-based model,
encapsulating self-attention and feed-forward network (FFN) sub-layers.
Its architecture is specifically tailored for tasks requiring deep hierarchical
reasoning, employing a "post-normalization" scheme that can offer performance
benefits at the cost of requiring more careful training stabilization.

Architecture:
    The block adheres to the standard two-stage Transformer architecture, but
    with specific modern components and an important structural choice regarding
    normalization. The data flows through two main sub-layers:

    1.  **Multi-Head Self-Attention:** Computes context-aware representations by
        allowing each token in the sequence to attend to all other tokens.
    2.  **SwiGLU Feed-Forward Network:** A position-wise FFN that provides
        additional non-linear transformation capacity. It uses a Swish-Gated
        Linear Unit (SwiGLU) for enhanced expressiveness over standard ReLU.

    A key architectural decision is the use of **post-normalization**. Unlike
    the more common pre-normalization (`x + SubLayer(Norm(x))`), this block
    applies normalization *after* the residual connection: `Norm(x + SubLayer(x))`.
    This places the normalization layers on the main signal path, which can
    sometimes lead to better representational power, but often necessitates
    careful learning rate scheduling (e.g., warmup) to ensure stable training
    in deep networks. Root Mean Square Normalization (RMSNorm) is used as a
    computationally efficient alternative to standard Layer Normalization.

Foundational Mathematics and Concepts:
    -   **Scaled Dot-Product Attention:** The core mechanism within the multi-head
        attention layer. It computes attention scores as:
        `Attention(Q, K, V) = softmax( (QK^T) / sqrt(d_k) ) * V`
        This allows the model to learn a weighted sum of value vectors (`V`), where
        the weights are determined by the compatibility of query (`Q`) and key
        (`K`) vectors. The scaling factor `sqrt(d_k)` prevents the dot products
        from growing too large and saturating the softmax function.

    -   **Root Mean Square Normalization (RMSNorm):** A variant of Layer
        Normalization that simplifies the computation by only re-scaling
        activations by their root mean square, omitting the mean-centering step.
        Given an input vector `x`, it is calculated as:
        `RMSNorm(x) = (x / sqrt(mean(x^2) + ε)) * g`
        where `g` is a learnable gain parameter. This is often faster than
        standard LayerNorm with minimal performance difference.

    -   **Swish-Gated Linear Unit (SwiGLU):** An advanced feed-forward network
        variant that introduces a gating mechanism. The input `x` is projected
        by two separate linear layers. One projection is passed through a Swish
        activation function, and the result is element-wise multiplied by the
        other projection: `SwiGLU(x) = Swish(xW + b) ⊙ (xV + c)`. This gating
        allows the network to dynamically control the flow of information through
        the FFN, often leading to improved performance.

References:
    1.  Vaswani, A. et al. (2017). "Attention Is All You Need." The foundational
        paper introducing the Transformer architecture, which used the
        post-normalization scheme implemented here.
    2.  Xiong, R. et al. (2020). "On Layer Normalization in the Transformer
        Architecture." Provides a detailed analysis of pre-normalization vs.
        post-normalization, highlighting the training stability trade-offs.
    3.  Zhang, B., & Sennrich, R. (2019). "Root Mean Square Layer
        Normalization." Introduces the RMSNorm algorithm.
    4.  Shazeer, N. (2020). "GLU Variants Improve Transformer." Proposes the
        SwiGLU activation function and demonstrates its benefits.
"""

import keras
from typing import Optional, Union, Dict, Any, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..norms.rms_norm import RMSNorm
from ..ffn.swiglu_ffn import SwiGLUFFN

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class HierarchicalReasoningBlock(keras.layers.Layer):
    """
    Post-normalization transformer block for Hierarchical Reasoning Models.

    This layer implements a transformer block specifically designed for hierarchical
    reasoning tasks, featuring post-normalization architecture where RMS normalization
    is applied after residual connections. The block uses SwiGLU activation for the
    feed-forward network, providing enhanced gating capabilities compared to standard
    ReLU-based FFNs.

    **Intent**: Provide a transformer building block optimized for hierarchical reasoning
    tasks, with architectural choices proven effective for complex reasoning scenarios.
    The post-normalization pattern helps with gradient flow in deep reasoning networks.

    **Architecture**:
    ```
    Input(shape=[batch_size, seq_length, hidden_size])
           ↓
    MultiHeadAttention(num_heads, key_dim=hidden_size//num_heads)
           ↓
    Add Residual: attention_output + input
           ↓
    RMSNorm(post-normalization)
           ↓
    SwiGLU FFN(intermediate_size)
           ↓
    Add Residual: ffn_output + normalized_attention
           ↓
    RMSNorm(post-normalization)
           ↓
    Output(shape=[batch_size, seq_length, hidden_size])
    ```

    **Post-Normalization Pattern**:
    Unlike standard pre-normalization where norm is applied before the sub-layer,
    this block uses post-normalization: `RMSNorm(x + SubLayer(x))`. This pattern
    can provide better training stability for deep reasoning networks.

    **Key Components**:
    - **MultiHeadAttention**: Standard Keras multi-head self-attention
    - **RMSNorm**: Root Mean Square normalization for efficiency
    - **SwiGLU FFN**: Swish-Gated Linear Unit for enhanced expressiveness
    - **Residual Connections**: Skip connections around both attention and FFN

    Args:
        hidden_size: Integer, hidden dimension size throughout the block.
            Must be positive and divisible by num_heads. This dimension is preserved.
        num_heads: Integer, number of attention heads for multi-head attention.
            Must be positive and divide evenly into hidden_size.
        intermediate_size: Integer, intermediate dimension for the FFN layer.
            Typically 4x hidden_size for standard transformer scaling.
        dropout_rate: Float between 0 and 1, dropout rate applied in attention and FFN.
            Defaults to 0.0 (no dropout).
        use_bias: Boolean, whether to use bias parameters in linear transformations.
            Defaults to False following modern transformer practices.
        kernel_initializer: String or Initializer, weight initialization strategy.
            Defaults to 'he_normal' for improved gradient flow.
        bias_initializer: String or Initializer, bias initialization strategy.
            Only used if use_bias=True. Defaults to 'zeros'.
        kernel_regularizer: Optional Regularizer for kernel weights.
            Applied to attention and FFN weights. Defaults to None.
        bias_regularizer: Optional Regularizer for bias weights.
            Only used if use_bias=True. Defaults to None.
        **kwargs: Additional arguments for Layer base class.

    Input shape:
        3D tensor with shape: `(batch_size, sequence_length, hidden_size)`.

    Output shape:
        3D tensor with shape: `(batch_size, sequence_length, hidden_size)`.
        Same shape as input, preserving sequence structure.

    Attributes:
        attention: MultiHeadAttention layer for self-attention computation.
        attention_norm: RMSNorm layer for post-attention normalization.
        ffn: SwiGLU FFN layer for feed-forward processing.
        output_norm: RMSNorm layer for final output normalization.
        dropout: Dropout layer for regularization.

    Example:
        ```python
        # Standard HRM block
        block = HierarchicalReasoningBlock(
            hidden_size=768,
            num_heads=12,
            intermediate_size=3072
        )
        inputs = keras.Input(shape=(128, 768))  # seq_len=128
        outputs = block(inputs)

        # With dropout and regularization
        block = HierarchicalReasoningBlock(
            hidden_size=512,
            num_heads=8,
            intermediate_size=2048,
            dropout_rate=0.1,
            kernel_regularizer=keras.regularizers.L2(1e-4)
        )

        # Custom initialization
        block = HierarchicalReasoningBlock(
            hidden_size=1024,
            num_heads=16,
            intermediate_size=4096,
            use_bias=True,
            kernel_initializer='glorot_uniform'
        )
        ```

    Note:
        This implementation follows the post-normalization pattern which can be
        more challenging to train than pre-normalization but may provide better
        performance for reasoning tasks when properly configured.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        dropout_rate: float = 0.0,
        use_bias: bool = False,
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'he_normal',
        bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
            )
        if intermediate_size <= 0:
            raise ValueError(f"intermediate_size must be positive, got {intermediate_size}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")

        # Store configuration parameters
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # Calculate key dimension for attention
        self.key_dim = hidden_size // num_heads

        # CREATE all sub-layers in __init__ (they are unbuilt)
        self.attention = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=self.key_dim,
            dropout=dropout_rate,
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name='multi_head_attention'
        )

        self.attention_norm = RMSNorm(name='attention_rms_norm')

        self.ffn = SwiGLUFFN(
            d_model=hidden_size,
            ffn_expansion_factor=intermediate_size // hidden_size,
            dropout_rate=dropout_rate,
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='swiglu_ffn'
        )

        self.output_norm = RMSNorm(name='output_rms_norm')

        # Dropout for additional regularization if needed
        if dropout_rate > 0.0:
            self.dropout = keras.layers.Dropout(dropout_rate, name='dropout')
        else:
            self.dropout = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all its sub-layers.

        CRITICAL: Explicitly build each sub-layer for robust serialization.
        """
        if len(input_shape) != 3:
            raise ValueError(f"Expected 3D input shape, got {len(input_shape)}D")

        if input_shape[-1] != self.hidden_size:
            raise ValueError(
                f"Last dimension of input ({input_shape[-1]}) must match hidden_size ({self.hidden_size})"
            )

        # Build sub-layers in computational order
        # MultiHeadAttention expects (batch_size, seq_length, hidden_size)
        self.attention.build(input_shape)

        # RMSNorm doesn't change shape
        self.attention_norm.build(input_shape)

        # SwiGLU FFN expects and returns same shape
        self.ffn.build(input_shape)

        # Final normalization
        self.output_norm.build(input_shape)

        # Dropout doesn't change shape
        if self.dropout is not None:
            self.dropout.build(input_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Post-normalization forward pass through the reasoning block.

        Args:
            inputs: Input tensor of shape [batch_size, seq_length, hidden_size].
            attention_mask: Optional attention mask for the self-attention computation.
                Can be 2D (batch_size, seq_length) or 4D for full attention control.
            training: Boolean indicating training mode for dropout behavior.

        Returns:
            Output tensor of shape [batch_size, seq_length, hidden_size].
        """
        # Self-attention with residual connection and post-normalization
        attention_output = self.attention(
            query=inputs,
            value=inputs,  # Self-attention: value = query
            key=inputs,    # Self-attention: key = query
            attention_mask=attention_mask,
            training=training
        )

        # Apply dropout if configured
        if self.dropout is not None:
            attention_output = self.dropout(attention_output, training=training)

        # Post-normalization: RMSNorm(inputs + attention_output)
        attention_residual = self.attention_norm(inputs + attention_output, training=training)

        # Feed-forward network with residual connection and post-normalization
        ffn_output = self.ffn(attention_residual, training=training)

        # Apply dropout if configured
        if self.dropout is not None:
            ffn_output = self.dropout(ffn_output, training=training)

        # Post-normalization: RMSNorm(attention_residual + ffn_output)
        final_output = self.output_norm(attention_residual + ffn_output, training=training)

        return final_output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute output shape.

        Args:
            input_shape: Shape of the input tensor.

        Returns:
            Output shape tuple. Same as input shape for transformer blocks.
        """
        # Output shape is identical to input shape for transformer blocks
        return tuple(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration for serialization.

        Returns:
            Dictionary containing all constructor parameters for proper serialization.
        """
        config = super().get_config()
        config.update({
            'hidden_size': self.hidden_size,
            'num_heads': self.num_heads,
            'intermediate_size': self.intermediate_size,
            'dropout_rate': self.dropout_rate,
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------