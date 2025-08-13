"""
SwiGLU Feed-Forward Network Implementation

This module implements the SwiGLU (Swish Gated Linear Unit) activation function within
a feed-forward network architecture. SwiGLU has been shown to outperform other gating
mechanisms like GeGLU and standard ReLU-based FFNs in large language models, providing
better training stability and model performance.

Mathematical Formulation:
    SwiGLU combines the Swish activation function with a gating mechanism:

    SwiGLU(x, W, V, b, c) = Swish(xW + b) ⊗ (xV + c)

    Where:
    - Swish(x) = x * sigmoid(x) = x * σ(x)
    - ⊗ denotes element-wise multiplication
    - W and V are learned weight matrices
    - b and c are bias terms (typically omitted)

    In this implementation:
    - gate = Swish(x * W_gate)
    - up = x * W_up  
    - hidden = gate ⊗ up
    - output = hidden * W_down

Architecture Details:
    1. Input projection to hidden dimension (typically 2/3 * expansion_factor * d_model)
    2. Two parallel projections: gate_proj (with Swish) and up_proj (linear)
    3. Element-wise multiplication of activated gate and up projections
    4. Down projection back to model dimension
    5. Optional dropout for regularization

Key Benefits:
    - Better gradient flow compared to standard ReLU-based FFNs
    - Gating mechanism allows selective information flow
    - Empirically superior performance in large language models
    - Maintains computational efficiency while improving representational capacity

Hardware Optimization:
    - Hidden dimension is rounded to multiples for efficient matrix operations
    - Follows the 2/3 rule from PaLM paper for optimal parameter allocation
    - Uses bias=False by default to reduce memory and computation

References:
    - Shazeer, N. (2020). "GLU Variants Improve Transformer."
      https://arxiv.org/abs/2002.05202

    - Chowdhery, A., et al. (2022). "PaLM: Scaling Language Modeling with Pathways."
      https://arxiv.org/abs/2204.02311 (Uses SwiGLU in practice)

    - Touvron, H., et al. (2023). "LLaMA: Open and Efficient Foundation Language Models."
      https://arxiv.org/abs/2302.13971 (SwiGLU in LLaMA architecture)

Usage Examples:
    Basic usage:
    >>> ffn = SwiGLUFFN(
    ...     d_model=512,
    ...     ffn_expansion_factor=4,
    ...     ffn_multiple_of=256
    ... )
    >>> output = ffn(inputs)

    In a transformer block:
    >>> def transformer_block(x):
    ...     # Self-attention
    ...     attn_out = MultiHeadAttention(...)(x)
    ...     x = LayerNorm()(x + attn_out)
    ...     
    ...     # SwiGLU FFN
    ...     ffn_out = SwiGLUFFN(...)(x)
    ...     return LayerNorm()(x + ffn_out)

    With custom parameters:
    >>> ffn = SwiGLUFFN(
    ...     d_model=768,
    ...     ffn_expansion_factor=8,  # Larger expansion
    ...     ffn_multiple_of=128,     # Hardware-friendly multiple
    ...     dropout_rate=0.1,        # Regularization
    ...     use_bias=False           # Memory efficiency
    ... )
"""

import keras
from keras import ops
from typing import Optional, Any, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class SwiGLUFFN(keras.layers.Layer):
    """
    SwiGLU Feed-Forward Network with gating mechanism.

    This layer implements the SwiGLU activation function within a feed-forward
    network, combining Swish activation with a gating mechanism for improved
    performance in transformer architectures.

    Args:
        d_model: int
            Model dimension (input/output feature size).
        ffn_expansion_factor: int, default=4
            Factor by which to expand the hidden dimension relative to d_model.
        ffn_multiple_of: int, default=256
            Round hidden dimension to this multiple for hardware efficiency.
        dropout_rate: float, default=0.0
            Dropout probability applied to the output.
        use_bias: bool, default=False
            Whether to use bias in linear projections.
        **kwargs: Additional keyword arguments for the Layer parent class.

    Input shape:
        N-D tensor with shape: (..., d_model)

    Output shape:
        N-D tensor with shape: (..., d_model)

    Raises:
        ValueError: If d_model, ffn_expansion_factor, or ffn_multiple_of are not positive,
                   or if dropout_prob is not in [0, 1].
    """

    def __init__(
            self,
            d_model: int,
            ffn_expansion_factor: int = 4,
            ffn_multiple_of: int = 256,
            dropout_rate: float = 0.0,
            use_bias: bool = False,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        self._validate_inputs(d_model, ffn_expansion_factor, ffn_multiple_of, dropout_rate)

        self.d_model = d_model
        self.ffn_expansion_factor = ffn_expansion_factor
        self.ffn_multiple_of = ffn_multiple_of
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias

        # Calculate hidden dimension with proper rounding (2/3 rule from PaLM)
        self.hidden_dim = self._calculate_hidden_dim()

        # Will be initialized in build()
        self.gate_proj = None
        self.up_proj = None
        self.down_proj = None
        self.dropout = None
        self._build_input_shape = None

        logger.info(f"SwiGLUFFN initialized: d_model={d_model}, "
                    f"hidden_dim={self.hidden_dim}, expansion_factor={ffn_expansion_factor}")

    def _validate_inputs(
            self,
            d_model: int,
            ffn_expansion_factor: int,
            ffn_multiple_of: int,
            dropout_prob: float
    ) -> None:
        """Validate initialization parameters."""
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if ffn_expansion_factor <= 0:
            raise ValueError(f"ffn_expansion_factor must be positive, got {ffn_expansion_factor}")
        if ffn_multiple_of <= 0:
            raise ValueError(f"ffn_multiple_of must be positive, got {ffn_multiple_of}")
        if not 0.0 <= dropout_prob <= 1.0:
            raise ValueError(f"dropout_prob must be in [0, 1], got {dropout_prob}")

    def _calculate_hidden_dim(self) -> int:
        """
        Calculate hidden dimension using the 2/3 rule from PaLM paper.

        The hidden dimension is calculated as:
        1. Start with d_model * expansion_factor * 2/3
        2. Round up to the nearest multiple of ffn_multiple_of

        Returns:
            Calculated hidden dimension.
        """
        # Apply 2/3 rule for optimal parameter allocation
        hidden_dim = int(self.d_model * self.ffn_expansion_factor * 2 / 3)

        # Round up to multiple for hardware efficiency
        hidden_dim = self.ffn_multiple_of * (
                (hidden_dim + self.ffn_multiple_of - 1) // self.ffn_multiple_of
        )

        return hidden_dim

    def build(self, input_shape) -> None:
        """Build the layer weights and sublayers."""
        self._build_input_shape = input_shape

        # Three projections for SwiGLU
        # Gate projection: applies Swish activation
        self.gate_proj = keras.layers.Dense(
            self.hidden_dim,
            use_bias=self.use_bias,
            name='gate_proj'
        )

        # Up projection: linear transformation
        self.up_proj = keras.layers.Dense(
            self.hidden_dim,
            use_bias=self.use_bias,
            name='up_proj'
        )

        # Down projection: back to model dimension
        self.down_proj = keras.layers.Dense(
            self.d_model,
            use_bias=self.use_bias,
            name='down_proj'
        )

        # Dropout layer
        self.dropout = keras.layers.Dropout(
            self.dropout_rate
        )

        # Build sublayers
        self.gate_proj.build(input_shape)
        self.up_proj.build(input_shape)

        # Down projection input shape has hidden_dim as last dimension
        down_input_shape = list(input_shape)
        down_input_shape[-1] = self.hidden_dim
        self.down_proj.build(tuple(down_input_shape))

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply SwiGLU feed-forward transformation.

        Args:
            inputs: Input tensor of shape (..., d_model)
            training: Whether in training mode

        Returns:
            Output tensor of shape (..., d_model)
        """
        # SwiGLU formula: Swish(xW₁) ⊗ xW₂
        # where ⊗ denotes element-wise multiplication

        # Parallel projections to hidden dimension
        gate = self.gate_proj(inputs)  # (..., hidden_dim)
        up = self.up_proj(inputs)  # (..., hidden_dim)

        # Apply SiLU (Swish) activation to gate: x * sigmoid(x)
        gate_activated = ops.silu(gate)

        # Element-wise multiplication (gating mechanism)
        hidden = gate_activated * up

        # Project back to model dimension
        output = self.down_proj(hidden)

        # Apply dropout if specified
        if self.dropout_rate > 0.0:
            output = self.dropout(output, training=training)

        return output

    def compute_output_shape(self, input_shape) -> tuple:
        """Compute output shape (same as input shape)."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "ffn_expansion_factor": self.ffn_expansion_factor,
            "ffn_multiple_of": self.ffn_multiple_of,
            "dropout_prob": self.dropout_rate,
            "use_bias": self.use_bias,
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get build configuration for serialization."""
        return {"input_shape": self._build_input_shape}

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build layer from configuration."""
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

    @property
    def num_parameters(self) -> int:
        """Get the total number of parameters in the layer."""
        if not self.built:
            return 0

        total_params = 0
        for weight in self.weights:
            total_params += weight.shape.num_elements()
        return total_params

# ---------------------------------------------------------------------
