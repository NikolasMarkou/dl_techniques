import keras
from typing import Optional, Any, Dict, Tuple, Union

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
    network, combining SiLU (Swish) activation with a gating mechanism for improved
    performance in transformer architectures.

    SwiGLU has been shown to outperform other gating mechanisms like GeGLU and
    standard ReLU-based FFNs in large language models, providing better training
    stability and model performance.

    The layer applies the following transformation:
        1. Projects input to hidden dimension using two parallel Dense layers
        2. Applies SiLU activation to gate projection
        3. Element-wise multiplication of activated gate and up projection
        4. Projects back to model dimension
        5. Optional dropout for regularization

    Mathematical formulation:
        gate = SiLU(input @ W_gate + b_gate)
        up = input @ W_up + b_up
        hidden = gate ⊗ up
        output = hidden @ W_down + b_down

    Where ⊗ denotes element-wise multiplication.

    Args:
        d_model: Integer, model dimension (input/output feature size).
            Must be positive.
        ffn_expansion_factor: Integer, factor by which to expand the hidden
            dimension relative to d_model. Must be positive. Defaults to 4.
        ffn_multiple_of: Integer, round hidden dimension to this multiple
            for hardware efficiency. Must be positive. Defaults to 256.
        dropout_rate: Float, dropout probability applied to the output.
            Must be between 0.0 and 1.0. Defaults to 0.0.
        use_bias: Boolean, whether to use bias in linear projections.
            Defaults to False for memory efficiency.
        kernel_initializer: String or Initializer, initializer for kernel weights.
            Defaults to 'glorot_uniform'.
        bias_initializer: String or Initializer, initializer for bias weights.
            Defaults to 'zeros'.
        kernel_regularizer: Optional regularizer for kernel weights.
        bias_regularizer: Optional regularizer for bias weights.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        N-D tensor with shape: (..., d_model)

    Output shape:
        N-D tensor with shape: (..., d_model)

    Attributes:
        gate_proj: Dense layer for gate projection with SiLU activation.
        up_proj: Dense layer for up projection (linear).
        down_proj: Dense layer for down projection back to d_model.
        dropout: Dropout layer for regularization.
        hidden_dim: Computed hidden dimension after applying expansion and rounding.

    Example:
        ```python
        # Basic usage
        ffn = SwiGLUFFN(d_model=512)

        # Custom configuration
        ffn = SwiGLUFFN(
            d_model=768,
            ffn_expansion_factor=8,  # Larger expansion
            ffn_multiple_of=128,     # Hardware-friendly multiple
            dropout_rate=0.1,        # Regularization
            use_bias=False           # Memory efficiency
        )

        # In a transformer block
        inputs = keras.Input(shape=(seq_len, d_model))
        x = MultiHeadAttention(...)(inputs)
        x = keras.layers.LayerNormalization()(inputs + x)
        ffn_out = SwiGLUFFN(d_model=d_model)(x)
        outputs = keras.layers.LayerNormalization()(x + ffn_out)
        model = keras.Model(inputs, outputs)
        ```

    References:
        - Shazeer, N. (2020). "GLU Variants Improve Transformer."
        - Chowdhery, A., et al. (2022). "PaLM: Scaling Language Modeling with Pathways."
        - Touvron, H., et al. (2023). "LLaMA: Open and Efficient Foundation Language Models."

    Raises:
        ValueError: If d_model, ffn_expansion_factor, or ffn_multiple_of are not positive,
                   or if dropout_rate is not in [0, 1].

    Note:
        This implementation follows the modern Keras 3 pattern where all sub-layers
        are created in __init__ and Keras handles building automatically. The hidden
        dimension calculation follows the 2/3 rule from the PaLM paper for optimal
        parameter allocation.
    """

    def __init__(
            self,
            d_model: int,
            ffn_expansion_factor: int = 4,
            ffn_multiple_of: int = 256,
            dropout_rate: float = 0.0,
            use_bias: bool = False,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        self._validate_inputs(d_model, ffn_expansion_factor, ffn_multiple_of, dropout_rate)

        # Store configuration
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

        # CREATE all sub-layers in __init__ (modern Keras 3 pattern)
        try:
            # Three projections for SwiGLU
            # Gate projection: will apply SiLU activation in call()
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

            # Dropout layer
            if self.dropout_rate > 0.0:
                self.dropout = keras.layers.Dropout(
                    self.dropout_rate,
                    name='dropout'
                )
            else:
                self.dropout = None

        except Exception as e:
            logger.error(f"Failed to create SwiGLUFFN sub-layers: {e}")
            raise ValueError(
                f"Failed to create SwiGLUFFN sub-layers. This might be due to "
                f"incompatible parameters or missing dependencies. Original error: {e}"
            )

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
            d_model: Model dimension to validate.
            ffn_expansion_factor: Expansion factor to validate.
            ffn_multiple_of: Multiple constraint to validate.
            dropout_rate: Dropout rate to validate.

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
            Calculated hidden dimension as integer.
        """
        # Apply 2/3 rule for optimal parameter allocation
        hidden_dim = int(self.d_model * self.ffn_expansion_factor * 2 / 3)

        # Round up to multiple for hardware efficiency
        hidden_dim = self.ffn_multiple_of * (
                (hidden_dim + self.ffn_multiple_of - 1) // self.ffn_multiple_of
        )

        return hidden_dim

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all its sub-layers.

        For robust serialization, explicitly build sub-layers in computational order.

        Args:
            input_shape: Shape of the input tensor.
        """
        # Build sub-layers in computational order
        self.gate_proj.build(input_shape)
        self.up_proj.build(input_shape)

        # Down projection input shape has hidden_dim as last dimension
        down_input_shape = list(input_shape)
        down_input_shape[-1] = self.hidden_dim
        self.down_proj.build(tuple(down_input_shape))

        # Build dropout if present
        if self.dropout is not None:
            self.dropout.build(tuple(down_input_shape))

        # Always call parent build at the end
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply SwiGLU feed-forward transformation.

        Performs the following computation:
        1. Parallel projections to hidden dimension (gate and up)
        2. Apply SiLU activation to gate projection
        3. Element-wise multiplication of activated gate and up projection
        4. Project back to model dimension
        5. Apply dropout if configured

        Args:
            inputs: Input tensor of shape (..., d_model).
            training: Boolean indicating training mode for dropout.

        Returns:
            Output tensor of shape (..., d_model).
        """
        # SwiGLU formula: SiLU(xW₁ + b₁) ⊗ (xW₂ + b₂)
        # where ⊗ denotes element-wise multiplication

        # Parallel projections to hidden dimension
        gate = self.gate_proj(inputs)  # (..., hidden_dim)
        up = self.up_proj(inputs)      # (..., hidden_dim)

        # Apply SiLU (Swish) activation to gate: x * sigmoid(x)
        gate_activated = keras.ops.silu(gate)

        # Element-wise multiplication (gating mechanism)
        hidden = gate_activated * up

        # Project back to model dimension
        output = self.down_proj(hidden)

        # Apply dropout if specified
        if self.dropout is not None:
            output = self.dropout(output, training=training)

        return output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute output shape (same as input shape for FFN).

        Args:
            input_shape: Shape of the input tensor.

        Returns:
            Output shape tuple, same as input shape.
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns:
            Dictionary containing all configuration parameters needed
            to recreate the layer.
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
            Total number of trainable and non-trainable parameters.
            Returns 0 if layer is not built.
        """
        if not self.built:
            return 0

        total_params = 0
        for weight in self.weights:
            # Use ops.size for Keras backend compatibility
            total_params += keras.ops.size(weight)
        return int(total_params)

# ---------------------------------------------------------------------