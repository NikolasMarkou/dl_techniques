"""
The MLP block from the Swin Transformer architecture.

This layer constitutes the position-wise Feed-Forward Network (FFN) that serves
as the second primary sub-component in each Swin Transformer block, following
the windowed attention mechanism. Its function is to apply a non-linear
transformation to each token representation independently and identically,
enabling the model to learn complex feature interactions.

While originating in the Swin Transformer, this specific MLP structure is the
standard and widely adopted FFN design used across the majority of Transformer-
based models.

Architectural Overview:
The network employs a simple yet highly effective "expand-then-contract"
architectural pattern, also known as an inverted bottleneck:

1.  **Expansion**: An initial linear layer (`fc1`) projects the input token
    representations from their original dimension into a much larger
    intermediate or hidden dimension. This expansion, typically by a factor
    of four, creates a high-dimensional "workspace" where features can be
    more easily separated and transformed by the subsequent non-linearity.

2.  **Non-linear Activation**: A non-linear activation function, typically
    GELU (Gaussian Error Linear Unit), is applied element-wise. This is the
    critical step that allows the FFN to model complex relationships that
    could not be captured by linear transformations alone. Without it, the two
    linear layers would collapse into a single, less expressive one.

3.  **Contraction**: A second linear layer (`fc2`) projects the activated,
    high-dimensional representation back down to the model's original output
    dimension. This step synthesizes the rich features learned in the expanded
    space into a final, refined token representation, ready for the next
    layer or residual connection.

This structure acts as a key-value memory, where the parameters learn to
recognize specific patterns from the attention output and map them to the
appropriate new representations.

Foundational Mathematics:
For an input vector `x` corresponding to a single position in the sequence,
the computation performed by the MLP block is:

`FFN(x) = W_2 * activation(W_1 @ x + b_1) + b_2`

where:
- `W_1` and `b_1` are the weight matrix and bias vector of the first linear
  layer, projecting `x` to the `hidden_dim`.
- `activation` is a non-linear function like GELU, `GELU(x) = x * Φ(x)`, where
  `Φ(x)` is the standard Gaussian cumulative distribution function.
- `W_2` and `b_2` are the weight matrix and bias vector of the second linear
  layer, projecting the result back to the `output_dim`.

This same function, with shared weights `W_1, b_1, W_2, b_2`, is applied
identically to each token vector across the entire sequence, making the
operation highly parallelizable.

References:
This MLP block is a core component of the Swin Transformer, as detailed in:
-   Liu, Z., et al. (2021). Swin Transformer: Hierarchical Vision Transformer
    using Shifted Windows. In Proceedings of the IEEE/CVF International
    Conference on Computer Vision (ICCV).

It is fundamentally the same position-wise FFN structure introduced in the
original Transformer paper:
-   Vaswani, A., et al. (2017). Attention Is All You Need. In Advances in
    Neural Information Processing Systems (NIPS).

"""

import keras
from typing import Tuple, Optional, Dict, Any, Union, Callable

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class SwinMLP(keras.layers.Layer):
    """
    MLP module for Swin Transformer with configurable activation and regularization.

    This layer implements the standard transformer "expand-then-contract" FFN pattern
    ``FFN(x) = W_2 * activation(W_1 @ x + b_1) + b_2`` with dual dropout layers
    for regularization. Input is expanded to ``hidden_dim``, activated (default GELU),
    dropped out, contracted to ``output_dim`` (or ``input_dim`` if ``output_dim`` is
    None), and dropped out again before the residual connection.

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────────────┐
        │   Input (..., input_dim)     │
        └────────────┬────────────────┘
                     ▼
        ┌─────────────────────────────┐
        │   fc1: Dense(hidden_dim)     │
        └────────────┬────────────────┘
                     ▼
        ┌─────────────────────────────┐
        │  Activation (e.g. GELU)      │
        └────────────┬────────────────┘
                     ▼
        ┌─────────────────────────────┐
        │   drop1: Dropout             │
        └────────────┬────────────────┘
                     ▼
        ┌─────────────────────────────┐
        │   fc2: Dense(output_dim)     │
        └────────────┬────────────────┘
                     ▼
        ┌─────────────────────────────┐
        │   drop2: Dropout             │
        └────────────┬────────────────┘
                     ▼
        ┌──────────────────────────────┐
        │  Output (..., output_dim)     │
        └──────────────────────────────┘

    :param hidden_dim: Integer, dimension of the hidden layer. Must be positive.
    :type hidden_dim: int
    :param use_bias: Whether to use bias in Dense layers. Defaults to True.
    :type use_bias: bool
    :param output_dim: Optional integer, dimension of the output layer. If None, uses
        input dimension for identity output shape. Defaults to None.
    :type output_dim: Optional[int]
    :param activation: Activation function. Supports Keras activation names
        ('gelu', 'relu', 'swish') or custom callables. Defaults to 'gelu'.
    :type activation: Union[str, Callable]
    :param dropout_rate: Dropout rate applied after activation and before
        output. Must be between 0.0 and 1.0. Defaults to 0.0.
    :type dropout_rate: float
    :param kernel_initializer: Initializer for Dense layer kernels.
        Defaults to 'glorot_uniform'.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for Dense layer biases.
        Defaults to 'zeros'.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularization applied to Dense layer kernels.
        Accepts L1, L2, or custom regularizers.
    :type kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param bias_regularizer: Optional regularization applied to Dense layer biases.
    :type bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param activity_regularizer: Optional regularization applied to layer outputs.
    :type activity_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param kwargs: Additional keyword arguments for Layer base class.

    :raises ValueError: If hidden_dim is not positive.
    :raises ValueError: If dropout_rate is not in [0.0, 1.0].
    :raises ValueError: If output_dim is specified but not positive.

    Note:
        This implementation creates all sub-layers during initialization and
        builds them explicitly during the build phase for robust serialization.
        The layer supports both training and inference modes with proper
        dropout behavior.
    """

    def __init__(
        self,
        hidden_dim: int,
        use_bias: bool = True,
        output_dim: Optional[int] = None,
        activation: Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]] = "gelu",
        dropout_rate: float = 0.0,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        activity_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate parameters
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError(f"dropout_rate must be between 0.0 and 1.0, got {dropout_rate}")
        if output_dim is not None and output_dim <= 0:
            raise ValueError(f"output_dim must be positive when specified, got {output_dim}")

        # Store configuration parameters
        self.hidden_dim = hidden_dim
        self.use_bias = use_bias
        self.output_dim = output_dim
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = keras.regularizers.get(activity_regularizer)

        # CREATE all sub-layers in __init__ (following modern Keras 3 pattern)
        # These are unbuilt at creation time
        self.fc1 = keras.layers.Dense(
            units=self.hidden_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            name="fc1"
        )

        # Activation layer
        self.act = keras.layers.Activation(self.activation, name="act")

        # Dropout layers (created even if rate=0.0 for consistent serialization)
        self.drop1 = keras.layers.Dropout(self.dropout_rate, name="drop1")
        self.drop2 = keras.layers.Dropout(self.dropout_rate, name="drop2")

        # Second dense layer - units will be set in build() based on output_dim logic
        # We need to create it here but will configure the units in build()
        self.fc2 = None  # Will be created in build() once we know output dimension

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all its sub-layers.

        Creates the layer's structure and builds all sub-layers for robust
        serialization following modern Keras 3 patterns.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        """
        # Validate input shape
        if len(input_shape) < 2:
            raise ValueError(f"Input must be at least 2D, got shape {input_shape}")

        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError("Last dimension of input must be defined")

        # Determine output dimension
        output_dim = self.output_dim if self.output_dim is not None else input_dim

        # Create second dense layer now that we know the output dimension
        self.fc2 = keras.layers.Dense(
            units=output_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            name="fc2"
        )

        # CRITICAL: Explicitly build all sub-layers in computational order
        # This ensures weight variables exist before serialization/deserialization
        self.fc1.build(input_shape)

        # Compute intermediate shape for subsequent layers
        fc1_output_shape = self.fc1.compute_output_shape(input_shape)

        # Activation layer doesn't change shape, but we build it for consistency
        self.act.build(fc1_output_shape)

        # Dropout layers don't change shape
        self.drop1.build(fc1_output_shape)
        self.fc2.build(fc1_output_shape)
        self.drop2.build(self.fc2.compute_output_shape(fc1_output_shape))

        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass of the SwinMLP layer.

        Applies the two-layer MLP transformation: fc1 -> activation -> dropout1 -> fc2 -> dropout2.

        :param inputs: Input tensor of shape (..., input_dim).
        :type inputs: keras.KerasTensor
        :param training: Whether the layer should behave in training mode
            (applying dropout) or inference mode.
        :type training: Optional[bool]
        :return: Output tensor of shape (..., output_dim) after MLP transformation.
        :rtype: keras.KerasTensor
        """
        # First linear transformation (expansion)
        x = self.fc1(inputs)

        # Non-linear activation
        x = self.act(x)

        # First dropout (applied after activation)
        x = self.drop1(x, training=training)

        # Second linear transformation (contraction)
        x = self.fc2(x)

        # Second dropout (applied before output)
        x = self.drop2(x, training=training)

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple with last dimension set to output_dim or input_dim.
        :rtype: Tuple[Optional[int], ...]
        """
        # Convert to list for manipulation
        output_shape = list(input_shape)

        # Set output dimension based on configuration
        if self.output_dim is not None:
            output_shape[-1] = self.output_dim
        # If output_dim is None, output shape matches input shape (identity transformation)

        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns all constructor parameters needed to recreate the layer.

        :return: Dictionary containing the complete layer configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "use_bias": self.use_bias,
            "output_dim": self.output_dim,
            "activation": self.activation,
            "dropout_rate": self.dropout_rate,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": keras.regularizers.serialize(self.activity_regularizer),
        })
        return config

# ---------------------------------------------------------------------
