"""
This module provides a specialized Transformer block, `HierarchicalReasoningBlock`,
which serves as a core component for a Hierarchical Reasoning Model (HRM) architecture.

The block inherits from a more generic `TransformerEncoderLayer` and customizes it
to follow a specific architectural pattern characterized by post-normalization.
This implementation makes two key modifications to the standard Transformer encoder design:
1.  It enforces a "post-normalization" scheme where layer normalization is applied
    *after* the residual connection. This is in contrast to the more common
    "pre-normalization" scheme. The specific pattern is: `Norm(x + Sublayer(x))`.
2.  It replaces the standard ReLU-based Feed-Forward Network (FFN) with a modern
    SwiGLU (Swish-Gated Linear Unit) FFN, which often leads to improved model
    performance through a gating mechanism.
3.  It utilizes RMSNorm (Root Mean Square Normalization) as its normalization layer,
    a computationally efficient alternative to standard LayerNorm.

The class is designed as a drop-in replacement or a specialized version of a
transformer block, configured specifically for the HRM architecture's requirements.
By inheriting from a base class, it reuses the core multi-head attention logic
while overriding the normalization and feed-forward processing flow in its `call` method.

Architectural Flow:
1.  Input is passed to a multi-head self-attention mechanism.
2.  The output of the attention layer is added to the original input (residual connection).
3.  The result is normalized using RMSNorm.
4.  The normalized output is then passed to a SwiGLU Feed-Forward Network.
5.  The output of the FFN is added to its input (a second residual connection).
6.  The result is passed through a final RMSNorm layer to produce the block's final output.
"""

import keras
from typing import Optional, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .norms.rms_norm import RMSNorm
from .ffn.swiglu_ffn import SwiGLUFFN
from .transformer_encoder import TransformerEncoderLayer

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class HierarchicalReasoningBlock(TransformerEncoderLayer):
    """
    Post-normalization transformer block for HRM.

    Extends the existing TransformerEncoderLayer to use post-normalization
    pattern and SwiGLU activation as required by HRM architecture.

    Args:
        hidden_size: Hidden dimension size
        num_heads: Number of attention heads
        intermediate_size: FFN intermediate dimension
        activation: Activation function (ignored, uses SwiGLU)
        dropout_rate: Dropout rate
        use_bias: Whether to use bias in linear layers
        kernel_initializer: Initializer for kernel weights
        kernel_regularizer: Regularizer for kernel weights
        **kwargs: Additional layer arguments
    """

    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            intermediate_size: int,
            activation: str = "swiglu",  # Force SwiGLU
            dropout_rate: float = 0.0,
            use_bias: bool = False,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs
    ):
        # Create RMSNorm factory for post-normalization
        def rms_norm_factory():
            return RMSNorm()

        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            activation=activation,
            dropout_rate=dropout_rate,
            normalization_factory=rms_norm_factory,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            **kwargs
        )

        # Override FFN with SwiGLU
        self._use_swiglu = True
        self.ffn = None
        self.norm1 = None
        self.norm2 = None
        self.feed_forward = None

    def build(self, input_shape):
        """Build layer with SwiGLU FFN override."""
        super().build(input_shape)

        # Replace the FFN with SwiGLU
        if self._use_swiglu:
            self.ffn = SwiGLUFFN(
                d_model=self.hidden_size,
                ffn_expansion_factor=self.intermediate_size // self.hidden_size,
                dropout_rate=self.dropout_rate,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name="swiglu_ffn"
            )

    def call(self, inputs, training=None, mask=None):
        """
        Post-normalization forward pass.

        Applies residual connections first, then normalization (post-norm pattern).
        """
        # Self-attention with residual connection and post-norm
        attn_output = self.attention(inputs, training=training, mask=mask)
        x = self.norm1(inputs + attn_output, training=training)

        # Feed-forward with residual connection and post-norm
        if hasattr(self, 'ffn') and self._use_swiglu:
            ffn_output = self.ffn(x, training=training)
        else:
            ffn_output = self.feed_forward(x, training=training)
        x = self.norm2(x + ffn_output, training=training)

        return x

# ---------------------------------------------------------------------
