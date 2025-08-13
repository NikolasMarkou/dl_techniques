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
from typing import Optional, Any, Dict, Tuple

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

    This implementation follows the modern Keras 3 pattern where all sub-layers
    are created in __init__ and Keras handles the building automatically. This
    ensures proper serialization and eliminates common build errors.

    Args:
        d_model: Integer, model dimension (input/output feature size). Must be positive.
        ffn_expansion_factor: Integer, factor by which to expand the hidden dimension
            relative to d_model. Must be positive. Defaults to 4.
        ffn_multiple_of: Integer, round hidden dimension to this multiple for hardware
            efficiency. Must be positive. Defaults to 256.
        dropout_rate: Float, dropout probability applied to the output. Must be in [0, 1].
            Defaults to 0.0.
        use_bias: Boolean, whether to use bias in linear projections. Defaults to False.
        kernel_initializer: String or initializer, initializer for kernel weights.
            Defaults to 'glorot_uniform'.
        bias_initializer: String or initializer, initializer for bias weights.
            Defaults to 'zeros'.
        kernel_regularizer: Optional regularizer for kernel weights.
        bias_regularizer: Optional regularizer for bias weights.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        N-D tensor with shape: `(..., d_model)`.
        Most common case is 3D input: `(batch_size, sequence_length, d_model)`.

    Output shape:
        N-D tensor with shape: `(..., d_model)`.
        Same shape as input.

    Attributes:
        hidden_dim: Calculated hidden dimension after applying the 2/3 rule and rounding.
        gate_proj: Dense layer for gate projection with Swish activation.
        up_proj: Dense layer for up projection (linear).
        down_proj: Dense layer for down projection back to model dimension.
        dropout: Dropout layer for regularization.

    Raises:
        ValueError: If d_model, ffn_expansion_factor, or ffn_multiple_of are not positive,
                   or if dropout_rate is not in [0, 1].

    Example:
        ```python
        # Basic usage
        ffn = SwiGLUFFN(d_model=768, ffn_expansion_factor=4)

        # With custom configuration
        ffn = SwiGLUFFN(
            d_model=1024,
            ffn_expansion_factor=8,
            ffn_multiple_of=128,
            dropout_rate=0.1,
            kernel_regularizer=keras.regularizers.L2(1e-4)
        )

        # In a transformer model
        inputs = keras.Input(shape=(512, 768))
        x = keras.layers.LayerNormalization()(inputs)
        outputs = SwiGLUFFN(d_model=768)(x)
        model = keras.Model(inputs, outputs)
        ```

    Note:
        The hidden dimension is calculated using the 2/3 rule from the PaLM paper:
        hidden_dim = round_to_multiple(d_model * expansion_factor * 2/3, multiple_of)
        This provides optimal parameter allocation for the SwiGLU architecture.
    """

    def __init__(
        self,
        d_model: int,
        ffn_expansion_factor: int = 4,
        ffn_multiple_of: int = 256,
        dropout_rate: float = 0.0,
        use_bias: bool = False,
        kernel_initializer: keras.initializers.Initializer = 'glorot_uniform',
        bias_initializer: keras.initializers.Initializer = 'zeros',
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        self._validate_inputs(d_model, ffn_expansion_factor, ffn_multiple_of, dropout_rate)

        # Store ALL configuration arguments as instance attributes
        self.d_model = d_model
        self.ffn_expansion_factor = ffn_expansion_factor
        self.ffn_multiple_of = ffn_multiple_of
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # Calculate hidden dimension with proper rounding (2/3 rule from PaLM)
        self.hidden_dim = self._calculate_hidden_dim()

        # CREATE all sub-layers in __init__ - Modern Keras 3 pattern
        # Three projections for SwiGLU

        # Gate projection: applies Swish activation
        self.gate_proj = keras.layers.Dense(
            self.hidden_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='gate_proj'
        )

        # Up projection: linear transformation
        self.up_proj = keras.layers.Dense(
            self.hidden_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='up_proj'
        )

        # Down projection: back to model dimension
        self.down_proj = keras.layers.Dense(
            self.d_model,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='down_proj'
        )

        # Dropout layer for regularization
        if self.dropout_rate > 0.0:
            self.dropout = keras.layers.Dropout(
                self.dropout_rate,
                name='dropout'
            )
        else:
            self.dropout = None

        logger.info(f"SwiGLUFFN initialized: d_model={d_model}, "
                    f"hidden_dim={self.hidden_dim}, expansion_factor={ffn_expansion_factor}")

    def _validate_inputs(
        self,
        d_model: int,
        ffn_expansion_factor: int,
        ffn_multiple_of: int,
        dropout_rate: float
    ) -> None:
        """
        Validate initialization parameters.

        Args:
            d_model: Model dimension.
            ffn_expansion_factor: Expansion factor for hidden dimension.
            ffn_multiple_of: Multiple to round hidden dimension to.
            dropout_rate: Dropout probability.

        Raises:
            ValueError: If any parameter is invalid.
        """
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if ffn_expansion_factor <= 0:
            raise ValueError(f"ffn_expansion_factor must be positive, got {ffn_expansion_factor}")
        if ffn_multiple_of <= 0:
            raise ValueError(f"ffn_multiple_of must be positive, got {ffn_multiple_of}")
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1], got {dropout_rate}")

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

    # No build() method needed - Keras handles building sub-layers automatically

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply SwiGLU feed-forward transformation.

        Args:
            inputs: Input tensor of shape `(..., d_model)`.
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            Output tensor of shape `(..., d_model)`.
        """
        # SwiGLU formula: Swish(xW₁) ⊗ xW₂
        # where ⊗ denotes element-wise multiplication

        # Parallel projections to hidden dimension
        gate = self.gate_proj(inputs, training=training)  # (..., hidden_dim)
        up = self.up_proj(inputs, training=training)      # (..., hidden_dim)

        # Apply SiLU (Swish) activation to gate: x * sigmoid(x)
        gate_activated = ops.silu(gate)

        # Element-wise multiplication (gating mechanism)
        hidden = gate_activated * up

        # Project back to model dimension
        output = self.down_proj(hidden, training=training)

        # Apply dropout if specified
        if self.dropout is not None:
            output = self.dropout(output, training=training)

        return output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape tuple of the input.

        Returns:
            Output shape tuple (same as input shape for SwiGLU FFN).
        """
        return tuple(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        This method must return ALL arguments needed to recreate the layer
        via __init__. Uses keras serializers for complex objects.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "ffn_expansion_factor": self.ffn_expansion_factor,
            "ffn_multiple_of": self.ffn_multiple_of,
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

    @property
    def num_parameters(self) -> int:
        """
        Get the total number of parameters in the layer.

        Returns:
            Total number of trainable parameters, or 0 if layer is not built.
        """
        if not self.built:
            return 0

        total_params = 0
        for weight in self.weights:
            total_params += ops.size(weight)
        return int(total_params)

# ---------------------------------------------------------------------
