import keras
from typing import Tuple, Optional, Dict, Any, Union, Callable

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class SwinMLP(keras.layers.Layer):
    """
    MLP module for Swin Transformer with configurable activation and regularization.

    This layer implements a two-layer feedforward network (MLP) with configurable
    hidden dimension, activation function, dropout, and regularization. Originally
    designed for Swin Transformer blocks, it serves as a general-purpose MLP with
    an expansion-contraction pattern for learning complex non-linear transformations.

    The layer follows the standard transformer MLP pattern:
    1. **Expansion**: Linear projection to higher-dimensional space (hidden_dim)
    2. **Activation**: Non-linear transformation (default: GELU)
    3. **Regularization**: Optional dropout for generalization
    4. **Contraction**: Linear projection back to output dimension
    5. **Final Regularization**: Optional dropout before residual connection

    Mathematical formulation:
        h = Dense₁(x) -> activation(h) -> dropout(h) -> Dense₂(h) -> dropout(output)

    Where x ∈ ℝ^(input_dim), h ∈ ℝ^(hidden_dim), output ∈ ℝ^(out_dim).

    Key Features:
    - Flexible dimensionality with configurable hidden and output dimensions
    - Support for any Keras activation function or custom callables
    - Built-in dropout regularization with configurable rates
    - Comprehensive weight initialization and regularization options
    - Optimized for both training stability and inference efficiency

    Args:
        hidden_dim: Integer, dimension of the hidden layer. Must be positive.
        use_bias: Boolean, whether to use bias in Dense layers. Defaults to True.
        out_dim: Optional integer, dimension of the output layer. If None, uses
            input dimension for identity output shape. Defaults to None.
        activation: String or callable, activation function. Supports Keras
            activation names ('gelu', 'relu', 'swish') or custom callables.
            Defaults to 'gelu'.
        drop_rate: Float, dropout rate applied after activation and before
            output. Must be between 0.0 and 1.0. Defaults to 0.0.
        kernel_initializer: String or Initializer, initializer for Dense layer
            kernels. Defaults to 'glorot_uniform'.
        bias_initializer: String or Initializer, initializer for Dense layer
            biases. Defaults to 'zeros'.
        kernel_regularizer: Optional Regularizer, regularization applied to
            Dense layer kernels. Accepts L1, L2, or custom regularizers.
        bias_regularizer: Optional Regularizer, regularization applied to
            Dense layer biases.
        activity_regularizer: Optional Regularizer, regularization applied to
            layer outputs.
        **kwargs: Additional keyword arguments for Layer base class.

    Input shape:
        N-D tensor with shape (..., input_dim) where input_dim is the size of
        the last dimension.

    Output shape:
        N-D tensor with shape (..., out_dim) where out_dim is specified in
        constructor or equals input_dim if out_dim=None.

    Attributes:
        fc1: First Dense layer (expansion to hidden_dim).
        act: Activation layer.
        drop1: First Dropout layer (applied after activation).
        fc2: Second Dense layer (contraction to out_dim).
        drop2: Second Dropout layer (applied before output).

    Example:
        ```python
        # Basic usage in Swin Transformer
        mlp = SwinMLP(hidden_dim=256, drop_rate=0.1)

        # Custom configuration
        mlp = SwinMLP(
            hidden_dim=512,
            out_dim=128,
            activation='swish',
            drop_rate=0.2,
            kernel_regularizer=keras.regularizers.L2(1e-4)
        )

        # In a model
        inputs = keras.Input(shape=(196, 384))  # Swin patches
        x = SwinMLP(hidden_dim=1536, drop_rate=0.1)(inputs)
        outputs = keras.layers.LayerNormalization()(x)
        model = keras.Model(inputs, outputs)
        ```

    References:
        - Liu et al., "Swin Transformer: Hierarchical Vision Transformer using
          Shifted Windows", ICCV 2021
        - https://arxiv.org/abs/2103.14030

    Raises:
        ValueError: If hidden_dim is not positive.
        ValueError: If drop_rate is not in [0.0, 1.0].
        ValueError: If out_dim is specified but not positive.

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
        out_dim: Optional[int] = None,
        activation: Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]] = "gelu",
        drop_rate: float = 0.0,
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
        if not 0.0 <= drop_rate <= 1.0:
            raise ValueError(f"drop_rate must be between 0.0 and 1.0, got {drop_rate}")
        if out_dim is not None and out_dim <= 0:
            raise ValueError(f"out_dim must be positive when specified, got {out_dim}")

        # Store configuration parameters
        self.hidden_dim = hidden_dim
        self.use_bias = use_bias
        self.out_dim = out_dim
        self.activation = activation
        self.drop_rate = drop_rate
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
        self.drop1 = keras.layers.Dropout(self.drop_rate, name="drop1")
        self.drop2 = keras.layers.Dropout(self.drop_rate, name="drop2")

        # Second dense layer - units will be set in build() based on out_dim logic
        # We need to create it here but will configure the units in build()
        self.fc2 = None  # Will be created in build() once we know output dimension

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all its sub-layers.

        This method creates the layer's structure and builds all sub-layers
        for robust serialization following modern Keras 3 patterns.

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        # Validate input shape
        if len(input_shape) < 2:
            raise ValueError(f"Input must be at least 2D, got shape {input_shape}")

        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError("Last dimension of input must be defined")

        # Determine output dimension
        output_dim = self.out_dim if self.out_dim is not None else input_dim

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

        Applies the two-layer MLP transformation with activation and dropout.
        The computation follows: fc1 -> activation -> dropout1 -> fc2 -> dropout2.

        Args:
            inputs: Input tensor of shape (..., input_dim).
            training: Boolean indicating whether the layer should behave in
                training mode (applying dropout) or inference mode.

        Returns:
            Output tensor of shape (..., out_dim) after MLP transformation.
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

        Args:
            input_shape: Shape tuple of the input tensor.

        Returns:
            Output shape tuple with last dimension set to out_dim or input_dim.
        """
        # Convert to list for manipulation
        output_shape = list(input_shape)

        # Set output dimension based on configuration
        if self.out_dim is not None:
            output_shape[-1] = self.out_dim
        # If out_dim is None, output shape matches input shape (identity transformation)

        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns all constructor parameters needed to recreate the layer.
        This is critical for proper serialization and deserialization.

        Returns:
            Dictionary containing the complete layer configuration.
        """
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "use_bias": self.use_bias,
            "out_dim": self.out_dim,
            "activation": self.activation,
            "drop_rate": self.drop_rate,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": keras.regularizers.serialize(self.activity_regularizer),
        })
        return config

# ---------------------------------------------------------------------