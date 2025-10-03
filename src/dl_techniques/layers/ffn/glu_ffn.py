"""
A Gated Linear Unit feed-forward network.

This layer serves as an advanced replacement for the standard position-wise
Feed-Forward Network (FFN) commonly used in Transformer architectures. It is
based on the Gated Linear Unit (GLU) principle, which introduces a dynamic,
input-dependent gating mechanism to modulate the flow of information through
the network, a concept extensively analyzed by Shazeer (2020).

The core idea is that instead of applying a static non-linearity (like ReLU)
to a single linear projection of the input, the GLU computes two separate
linear projections. One projection acts as the "value," containing the primary
information, while the other acts as the "gate." The gate, after an
activation function, element-wise multiplies the value, selectively filtering
or amplifying features based on the input context. This allows for a more
nuanced and powerful transformation than a standard FFN.

Architectural Overview:
The layer's architecture is defined by two parallel pathways that process the
input before being combined:

1.  **Value Pathway**: A linear projection (`value_proj`) transforms the
    input into an intermediate representation. This pathway carries the main
    content to be processed.

2.  **Gate Pathway**: A second, independent linear projection (`gate_proj`)
    also transforms the input. The output of this projection is passed
    through a non-linear activation function (e.g., Swish, GELU, Sigmoid).

3.  **Gating Mechanism**: The activated gate is element-wise multiplied with
    the output of the value pathway. This is the central operation of the
    GLU, where the gate dynamically controls which information from the
    value pathway is passed forward.

4.  **Output Projection**: The resulting gated tensor is projected by a final
    linear layer (`output_proj`) to the desired output dimension.

This dual-pathway design provides greater expressive capacity, as the network
can learn to ignore or emphasize different features for each specific input
token, improving model performance and training dynamics.

Foundational Mathematics:
Let `x` be the input vector. The layer's computation is as follows:

1.  Two independent linear projections are computed:
    `g = W_g @ x + b_g`  (gate projection)
    `v = W_v @ x + b_v`  (value projection)
    where `W_g` and `W_v` are distinct weight matrices.

2.  The gate `g` is passed through a non-linearity, and the result is
    multiplied element-wise with the value `v`:
    `h = activation(g) * v`
    Here, `*` denotes the Hadamard (element-wise) product. Each element of
    `activation(g)` acts as a scalar control for the corresponding element in
    `v`.

3.  The final output `y` is produced by a third linear projection:
    `y = W_out @ h + b_out`

The choice of `activation` function defines the specific variant of the GLU,
such as SwiGLU (`swish`), GeGLU (`gelu`), or the original formulation with
`sigmoid`.

References:
The application and analysis of GLU variants in the context of Transformer
models is detailed in:

-   Shazeer, N. (2020). GLU Variants Improve Transformer. arXiv preprint
    arXiv:2002.05202.

The original Gated Linear Unit concept was introduced in:

-   Dauphin, Y. N., Fan, A., Auli, M., & Grangier, D. (2017). Language
    Modeling with Gated Convolutional Networks. ICML.

"""

import keras
from typing import Callable, Optional, Union, Any, Dict, Tuple
from keras import layers, initializers, regularizers, activations

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class GLUFFN(keras.layers.Layer):
    """
    Gated Linear Unit Feed Forward Network as described in "GLU Variants Improve Transformer".

    This layer implements a feed-forward block using a Gated Linear Unit (GLU),
    which has been shown to improve performance in Transformer models. The GLU
    mechanism applies a gating function to control information flow through
    element-wise multiplication of two separate linear projections, providing
    selective information processing and improved gradient flow.

    **Intent**: Provide a high-performance alternative to standard MLP blocks in
    Transformers, with enhanced gradient flow and selective information processing
    through the gating mechanism. Particularly effective for language models and
    sequence processing tasks.

    **Architecture**:
    ```
    Input(shape=[..., input_dim])
           ┃
      ┏━━━━┻━━━━┓
      ┃         ┃
      ↓         ↓
    gate_proj   value_proj
    (Dense)     (Dense)
      ┃         ┃
      ↓         ┃
    activation  ┃
      ┃         ┃
      ┗━━━━┓ ┏━━┛
           ┃ ┃
           ↓ ↓
        Element-wise Multiply (gate ⊙ value)
           ┃
           ↓
        Dropout (training only)
           ┃
           ↓
        output_proj
        (Dense)
           ┃
           ↓
    Output(shape=[..., output_dim])
    ```

    **Mathematical Operation**:
    1. **Gate Path**: gate = gate_proj(x) → activation(gate)
    2. **Value Path**: value = value_proj(x)
    3. **Gating**: gated = activation(gate) ⊙ value
    4. **Output**: output = output_proj(dropout(gated, training))

    Where ⊙ denotes element-wise multiplication and the activation is applied
    only to the gate pathway, allowing learned control over information flow.

    Args:
        hidden_dim: Integer, dimensionality of the intermediate hidden layer.
            Must be positive. Controls the capacity of the gating mechanism.
            Typically larger than input_dim for feature expansion (e.g., 2-4x).
        output_dim: Integer, dimensionality of the final output. Must be positive.
            Often equals input_dim in Transformer blocks for residual connections.
        activation: Activation function for the gate projection. Can be string name
            ('gelu', 'swish', 'sigmoid', 'tanh') or callable. Defaults to 'swish'
            (also known as SiLU), which is particularly effective for gating.
        dropout_rate: Float between 0 and 1, dropout rate applied after gating
            mechanism for regularization. Only active during training. Defaults to 0.0.
        use_bias: Boolean, whether to include bias terms in all Dense projections.
            Can improve model expressiveness. Defaults to True.
        kernel_initializer: Initializer for kernel weights in all Dense layers.
            Affects convergence and training stability. Defaults to 'glorot_uniform'.
        bias_initializer: Initializer for bias vectors in all Dense layers.
            Defaults to 'zeros'.
        kernel_regularizer: Optional regularizer applied to kernel weights of
            all Dense layers for preventing overfitting. Defaults to None.
        bias_regularizer: Optional regularizer applied to bias weights of
            all Dense layers. Defaults to None.
        **kwargs: Additional keyword arguments for Layer base class (name, dtype, etc.).

    Input shape:
        N-D tensor with shape: `(batch_size, ..., input_dim)`.
        Most commonly 3D: `(batch_size, sequence_length, input_dim)` for sequences,
        or 2D: `(batch_size, input_dim)` for standard feedforward networks.

    Output shape:
        N-D tensor with shape: `(batch_size, ..., output_dim)`.
        Same rank as input tensor, with last dimension changed to output_dim.

    Attributes:
        gate_proj: Dense layer for gating pathway (input_dim → hidden_dim).
        value_proj: Dense layer for value pathway (input_dim → hidden_dim).
        output_proj: Dense layer for final projection (hidden_dim → output_dim).
        dropout: Dropout layer for regularization during training.

    Example:
        ```python
        # Standard Transformer FFN replacement with GLU
        model_dim = 512
        ffn_dim = int((2/3) * 4 * model_dim)  # SwiGLU convention: ~1365

        layer = GLUFFN(
            hidden_dim=ffn_dim,
            output_dim=model_dim,
            activation='swish',
            dropout_rate=0.1
        )
        inputs = keras.Input(shape=(128, model_dim))  # (batch, seq_len, features)
        outputs = layer(inputs)  # Shape: (batch, 128, 512)

        # Binary gate with sigmoid activation
        binary_glu = GLUFFN(
            hidden_dim=256,
            output_dim=128,
            activation='sigmoid',
            dropout_rate=0.2
        )

        # Classification head with GLU
        classifier_glu = GLUFFN(
            hidden_dim=512,
            output_dim=10,  # num_classes
            activation='gelu',
            dropout_rate=0.3
        )

        # No bias for specific architectures
        layer = GLUFFN(
            hidden_dim=1024,
            output_dim=768,
            activation='swish',
            use_bias=False,
            kernel_regularizer='l2'
        )
        ```

    References:
        - Shazeer, N. (2020). GLU Variants Improve Transformer. arXiv preprint.
        - Touvron, H., et al. (2021). Training data-efficient image transformers.

    Raises:
        ValueError: If hidden_dim or output_dim are not positive integers.
        ValueError: If dropout_rate is not between 0 and 1.

    Note:
        The gating mechanism allows selective information flow, which can improve
        gradient propagation compared to standard ReLU-based FFN blocks. The choice
        of activation function for the gate is crucial - 'swish' and 'gelu' are
        commonly effective choices for modern architectures.
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        activation: Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]] = 'swish',
        dropout_rate: float = 0.0,
        use_bias: bool = True,
        kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
        bias_initializer: Union[str, initializers.Initializer] = 'zeros',
        kernel_regularizer: Optional[regularizers.Regularizer] = None,
        bias_regularizer: Optional[regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        """Initialize the GLU FFN layer with comprehensive parameter validation."""
        super().__init__(**kwargs)

        # Comprehensive input validation with informative error messages
        if not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be a positive integer, got {hidden_dim}")
        if not isinstance(output_dim, int) or output_dim <= 0:
            raise ValueError(f"output_dim must be a positive integer, got {output_dim}")
        if not isinstance(dropout_rate, (int, float)) or not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")

        # Store ALL configuration parameters for serialization
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = activations.get(activation)
        self.dropout_rate = float(dropout_rate)
        self.use_bias = bool(use_bias)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)


        # CREATE all sub-layers in __init__ (following Modern Keras 3 patterns)
        # All layers are unbuilt at this point - building happens in build()
        dense_kwargs = {
            "use_bias": self.use_bias,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "bias_regularizer": self.bias_regularizer,
        }

        self.gate_proj = layers.Dense(
            self.hidden_dim,
            activation=None,  # Activation applied separately for clarity
            name="gate_proj",
            **dense_kwargs
        )

        self.value_proj = layers.Dense(
            self.hidden_dim,
            activation=None,
            name="value_proj",
            **dense_kwargs
        )

        self.output_proj = layers.Dense(
            self.output_dim,
            activation=None,
            name="output_proj",
            **dense_kwargs
        )

        self.dropout = layers.Dropout(
            rate=self.dropout_rate,
            name="dropout"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all its sub-layers for robust serialization.

        This method explicitly builds each sub-layer to ensure all weight variables
        are created before Keras attempts to restore saved weights during loading.
        Critical for proper serialization/deserialization lifecycle.

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        # Validate input shape has defined last dimension
        if input_shape[-1] is None:
            raise ValueError("The last dimension of input_shape must be defined")

        # Build gate and value projections with input shape
        self.gate_proj.build(input_shape)
        self.value_proj.build(input_shape)

        # Compute intermediate shape after gate/value projections
        # Both projections output the same shape: (..., hidden_dim)
        intermediate_shape = self.gate_proj.compute_output_shape(input_shape)

        # Build downstream layers with intermediate shape
        self.dropout.build(intermediate_shape)
        self.output_proj.build(intermediate_shape)

        # CRITICAL: Always call parent build() at the end
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass implementing the GLU gating mechanism.

        Args:
            inputs: Input tensor of any rank with last dimension as features.
            training: Boolean indicating whether layer is in training mode.
                Affects dropout behavior.

        Returns:
            Output tensor with same rank as input, last dimension = output_dim.
        """
        # Dual pathway projections
        gate = self.gate_proj(inputs)      # Shape: (..., hidden_dim)
        value = self.value_proj(inputs)    # Shape: (..., hidden_dim)

        # Apply activation only to gate, then perform element-wise gating
        gated_value = self.activation(gate) * value  # Shape: (..., hidden_dim)

        # Apply dropout for regularization (only during training)
        gated_value = self.dropout(gated_value, training=training)

        # Final projection to output dimension
        output = self.output_proj(gated_value)  # Shape: (..., output_dim)

        return output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute output shape transformation.

        Args:
            input_shape: Shape tuple of input tensor.

        Returns:
            Shape tuple of output tensor with last dimension = output_dim.
        """
        output_shape = list(input_shape)
        output_shape[-1] = self.output_dim
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns ALL constructor parameters to ensure perfect reconstruction
        during model loading. Critical for Keras serialization lifecycle.

        Returns:
            Dictionary containing complete layer configuration.
        """
        config = super().get_config()
        config.update({
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'activation': activations.serialize(self.activation),
            'dropout_rate': self.dropout_rate,
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------