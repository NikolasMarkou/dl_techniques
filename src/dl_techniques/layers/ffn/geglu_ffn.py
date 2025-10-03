"""
A Gated Linear Unit Feed-Forward Network with GELU activation.

This layer provides a more expressive alternative to the standard position-wise
Feed-Forward Network (FFN) found in Transformer architectures. It is based on
the Gated Linear Unit (GLU) family of layers, specifically the variant using
the Gaussian Error Linear Unit (GELU) activation, as proposed by Shazeer.

The fundamental insight behind GLU-based networks is to introduce a dynamic,
input-dependent gating mechanism that controls the flow of information through
the layer. Unlike a standard FFN which applies a static non-linearity (like
ReLU or GELU) to all features uniformly, a GeGLU can selectively amplify or
suppress features based on the context provided by the input itself.

Architectural Overview:
The layer operates in three main stages:

1.  **Projection and Splitting**: The input tensor is first projected by a
    single dense layer into a space twice the size of the desired hidden
    dimension. This larger tensor is then split along its last axis into
    two equal-sized tensors: a "gate" and a "value". This is an efficient
    implementation equivalent to using two separate dense layers.

2.  **Gating Mechanism**: The "gate" tensor is passed through a GELU
    activation function. The resulting activated gate is then element-wise
    multiplied with the "value" tensor. This multiplication is the core of
    the gating mechanism.

3.  **Output Projection**: The resulting gated tensor, which now contains a
    filtered representation of the input, is passed through a final dense
    layer to project it back to the desired output dimension.

This architecture allows the network to learn complex interactions. The gate
can be thought of as a learned filter that decides which elements of the
value tensor are relevant and should be passed on.

Foundational Mathematics:
Let `x` be the input vector. The computation proceeds as follows:

1.  A linear projection `W` expands `x` into a larger space, which is then
    split into two vectors, the gate `g` and the value `v`. This can be
    expressed as:
    `[g, v] = W @ x + b`
    where `W` has shape `(input_dim, 2 * hidden_dim)`.

2.  The core gating operation combines the gate and value:
    `h = GELU(g) * v`
    where `*` denotes element-wise multiplication. Each element of `GELU(g)`
    acts as a scalar multiplier for the corresponding element in `v`,
    effectively filtering the value vector. If an element in `GELU(g)` is
    close to zero, the corresponding feature in `v` is suppressed.

3.  The final output `y` is obtained by projecting the gated vector `h`:
    `y = W_out @ h + b_out`

This formulation contrasts with a standard FFN, `y = W_out @ GELU(W_in @ x +
b_in) + b_out`, by making the non-linear filtering dependent on a separate,
parallel transformation of the input, granting it greater flexibility and
expressive power.

References:
This architecture and its benefits are primarily described in:

-   Shazeer, N. (2020). GLU Variants Improve Transformer. arXiv preprint
    arXiv:2002.05202.

The original Gated Linear Unit concept was introduced for convolutional
networks in:

-   Dauphin, Y. N., Fan, A., Auli, M., & Grangier, D. (2017). Language
    Modeling with Gated Convolutional Networks. ICML.

The GeGLU FFN is a key component in several state-of-the-art large language
models, including Google's PaLM.

"""

import keras
from keras import ops, layers, initializers, regularizers, activations
from typing import Optional, Union, Any, Dict, Callable, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class GeGLUFFN(keras.layers.Layer):
    """
    GELU Gated Linear Unit Feed-Forward Network (GeGLU).

    This layer implements the GeGLU variant of a Gated Linear Unit, which has
    shown strong performance in modern transformer architectures. The input is
    projected to a higher-dimensional space using a single Dense layer and then
    split into two equal parts: a 'gate' and a 'value' tensor. The gate is
    passed through a GELU activation and then multiplied element-wise with the
    value tensor.

    **Architecture**:
    ```
    Input(shape=[..., input_dim])
           │
           ▼
    Dense(hidden_dim * 2) ──> Split into (gate, value)
           │                             │
           ├─> gate ─> GELU Activation ──┐
           │                             ▼
           └─> value ──────────────────> Multiply ──> Dropout ──> Dense(output_dim)
                                                                       │
                                                                       ▼
                                                           Output(shape=[..., output_dim])
    ```

    **Mathematical Operation**:
        ```
        projected = input_proj(x)  # Shape: [..., hidden_dim * 2]
        gate, value = split(projected, axis=-1)  # Each: [..., hidden_dim]
        gated = gelu(gate) * value  # Element-wise multiplication
        output = output_proj(dropout(gated))  # Shape: [..., output_dim]
        ```

    **Data Flow**:
    1. Linear projection to expand dimensionality by factor of 2
    2. Split into gate and value tensors of equal size
    3. Apply GELU activation to gate tensor
    4. Element-wise multiplication of activated gate with value
    5. Apply dropout for regularization
    6. Final linear projection to target output dimension

    This pattern enables the model to selectively pass information through the
    network based on the learned gating mechanism.

    Args:
        hidden_dim: Integer, dimensionality of the intermediate hidden layer. This is
            the size of the `gate` and `value` tensors after splitting. Must be positive.
        output_dim: Integer, dimensionality of the final output. Must be positive.
        activation: String name or callable, the activation function for the gate.
            Common choices: 'gelu', 'relu', 'swish'. Defaults to 'gelu'.
        dropout_rate: Float between 0 and 1, dropout rate applied after the
            gating mechanism for regularization. Defaults to 0.0.
        use_bias: Boolean, whether the dense layers should use bias vectors.
            When False, only applies linear transformations. Defaults to True.
        kernel_initializer: Initializer for the kernel weights of dense layers.
            Accepts string names ('glorot_uniform', 'he_normal') or Initializer instances.
            Defaults to 'glorot_uniform'.
        bias_initializer: Initializer for bias vectors of dense layers.
            Only used when use_bias=True. Defaults to 'zeros'.
        kernel_regularizer: Optional regularizer for kernel weights of dense layers.
            Can help prevent overfitting in large models.
        bias_regularizer: Optional regularizer for bias vectors of dense layers.
            Only used when use_bias=True.
        **kwargs: Additional keyword arguments for the `keras.layers.Layer` base class.

    Input shape:
        N-D tensor with shape: `(batch_size, ..., input_dim)`.
        Most common: 3D tensor with shape `(batch_size, sequence_length, input_dim)`.

    Output shape:
        N-D tensor with shape: `(batch_size, ..., output_dim)`.
        Same rank as input, but last dimension changed to `output_dim`.

    Attributes:
        input_proj: Dense layer projecting input to twice the hidden dimension.
        output_proj: Dense layer projecting the gated value to the output dimension.
        dropout: Dropout layer for regularization during training.

    Example:
        ```python
        # Standard Transformer FFN block
        model_dim = 512
        ffn_dim = 2048  # Common 4x expansion factor

        layer = GeGLUFFN(
            hidden_dim=ffn_dim,
            output_dim=model_dim,
            dropout_rate=0.1
        )
        inputs = keras.Input(shape=(64, model_dim))  # (batch, seq_len, features)
        outputs = layer(inputs)  # Shape: (batch, 64, 512)

        # Custom activation and regularization
        layer = GeGLUFFN(
            hidden_dim=1024,
            output_dim=768,
            activation='swish',
            dropout_rate=0.2,
            kernel_regularizer=keras.regularizers.L2(1e-4)
        )

        # Without bias for specific architectures
        layer = GeGLUFFN(
            hidden_dim=512,
            output_dim=256,
            use_bias=False,
            kernel_initializer='he_normal'
        )
        ```

    Note:
        This implementation follows modern Keras 3 patterns where all sub-layers
        are created in __init__ and explicitly built in build() for robust
        serialization. The layer is fully compatible with model saving/loading.
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        activation: Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]] = 'gelu',
        dropout_rate: float = 0.0,
        use_bias: bool = True,
        kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
        bias_initializer: Union[str, initializers.Initializer] = 'zeros',
        kernel_regularizer: Optional[regularizers.Regularizer] = None,
        bias_regularizer: Optional[regularizers.Regularizer] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # 1. Comprehensive Input Validation
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(
                f"dropout_rate must be between 0 and 1, got {dropout_rate}"
            )

        # 2. Store ALL Configuration Parameters
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = activations.get(activation)
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        # 3. CREATE all sub-layers in __init__ (Modern Keras 3 pattern)
        try:
            dense_kwargs = {
                "use_bias": self.use_bias,
                "kernel_initializer": self.kernel_initializer,
                "bias_initializer": self.bias_initializer,
                "kernel_regularizer": self.kernel_regularizer,
                "bias_regularizer": self.bias_regularizer,
            }

            # Input projection: maps input_dim -> hidden_dim * 2
            self.input_proj = layers.Dense(
                units=hidden_dim * 2,
                name="input_proj",
                **dense_kwargs
            )

            # Output projection: maps hidden_dim -> output_dim
            self.output_proj = layers.Dense(
                units=output_dim,
                name="output_proj",
                **dense_kwargs
            )

            # Dropout for regularization
            self.dropout = layers.Dropout(dropout_rate, name="dropout")

        except Exception as e:
            logger.error(f"Failed to create GeGLUFFN sub-layers: {e}")
            raise ValueError(f"Failed to create GeGLUFFN sub-layers: {e}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create weights for all sub-layers.

        CRITICAL: Explicitly build each sub-layer for robust serialization.
        This ensures all weight variables exist before weight restoration.
        """
        if input_shape[-1] is None:
            raise ValueError("Last dimension of input must be defined")

        # Build sub-layers in computational order
        self.input_proj.build(input_shape)

        # Shape after input projection and splitting
        intermediate_shape = (*input_shape[:-1], self.hidden_dim)

        # Build remaining layers
        self.dropout.build(intermediate_shape)
        self.output_proj.build(intermediate_shape)

        # Always call parent build at the end
        super().build(input_shape)


    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass for the GeGLU FFN.

        Args:
            inputs: Input tensor of shape [..., input_dim].
            training: Boolean indicating training mode for dropout.

        Returns:
            Output tensor of shape [..., output_dim].
        """
        # 1. Project input to expanded dimension and split
        gate_and_value = self.input_proj(inputs)
        gate, value = ops.split(gate_and_value, 2, axis=-1)

        # 2. Apply gating mechanism: activated_gate * value
        activated_gate = self.activation(gate)
        gated_value = activated_gate * value

        # 3. Apply dropout for regularization
        gated_value = self.dropout(gated_value, training=training)

        # 4. Project to final output dimension
        output = self.output_proj(gated_value)

        return output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Computes the output shape of the layer.

        Args:
            input_shape: Shape tuple of input tensor.

        Returns:
            Shape tuple for output tensor.
        """
        output_shape = list(input_shape)
        output_shape[-1] = self.output_dim
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the layer's configuration for serialization.

        CRITICAL: Must include ALL parameters from __init__ for proper reconstruction.

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
