"""
PowerMLP Model: Complete Keras Model Implementation
================================================

This module implements PowerMLP as a complete Keras Model (not just a layer),
providing an efficient alternative to Kolmogorov-Arnold Networks (KAN) with:
- ~40x faster training time
- ~10x fewer FLOPs
- Equal or better performance
- Full Keras Model compliance with compile/fit workflow

Architecture:
    - Multiple PowerMLPLayer instances
    - Configurable hidden layers
    - Optional output activation
    - Full model serialization support

References:
    [1] "PowerMLP: An Efficient Version of KAN" (2024)
    [2] "KAN: Kolmogorov-Arnold Networks" (2024)
"""

import os
import keras
from typing import List, Optional, Union, Dict, Any, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.power_mlp_layer import PowerMLPLayer

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class PowerMLP(keras.Model):
    """PowerMLP model implementation as a complete Keras Model.

    PowerMLP is an efficient alternative to Kolmogorov-Arnold Networks (KAN) that provides
    superior performance with significantly reduced computational requirements. This implementation
    provides full Keras Model functionality with compile/fit workflow support.

    Architecture:
        The model consists of a sequence of PowerMLPLayer instances, each implementing
        the dual-branch architecture:
        - Main Branch: Dense → ReLU-k activation
        - Basis Branch: BasisFunction → Dense (no bias)
        - Combined via element-wise addition

    Args:
        hidden_units: List of integers specifying the number of units in each layer.
            The last element is the output dimension.
        k: Integer, power exponent for the ReLU-k activation function.
            Must be positive. Higher values create more aggressive non-linearities.
        kernel_initializer: Initializer for the kernel weights in all layers.
            Can be string name or Initializer instance.
        bias_initializer: Initializer for the bias vector in all layers.
            Can be string name or Initializer instance.
        kernel_regularizer: Optional regularizer function applied to kernel weights
            in all layers.
        bias_regularizer: Optional regularizer function applied to bias vector
            in all layers.
        use_bias: Boolean, whether to use bias in the main branch dense layers.
            The basis branch never uses bias by design.
        output_activation: Optional activation function for the final output layer.
            Can be string name or callable.
        dropout_rate: Float between 0 and 1, dropout rate applied after each
            hidden layer. Set to 0.0 to disable dropout.
        batch_normalization: Boolean, whether to apply batch normalization
            after each hidden layer.
        name: Optional name for the model.
        **kwargs: Additional keyword arguments passed to the Model parent class.

    Input shape:
        N-D tensor with shape: `(..., input_dim)`.
        Most common case would be 2D input with shape `(batch_size, input_dim)`.

    Output shape:
        N-D tensor with shape: `(..., output_dim)` where output_dim is the
        last element in hidden_units.

    Raises:
        ValueError: If hidden_units is empty or contains non-positive values.
        ValueError: If k is not a positive integer.
        ValueError: If dropout_rate is not in [0, 1].
    """

    def __init__(
            self,
            hidden_units: List[int],
            k: int = 3,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
            bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            use_bias: bool = True,
            output_activation: Optional[Union[str, callable]] = None,
            dropout_rate: float = 0.0,
            batch_normalization: bool = False,
            name: Optional[str] = "power_mlp",
            **kwargs: Any
    ) -> None:
        """Initialize the PowerMLP model.

        Args:
            hidden_units: List of integers for layer sizes.
            k: Power for ReLU-k activation.
            kernel_initializer: Initializer for kernel weights.
            bias_initializer: Initializer for bias vector.
            kernel_regularizer: Regularizer for kernel weights.
            bias_regularizer: Regularizer for bias vector.
            use_bias: Whether to use bias in main branch.
            output_activation: Activation function for output layer.
            dropout_rate: Dropout rate for regularization.
            batch_normalization: Whether to use batch normalization.
            name: Optional name for the model.
            **kwargs: Additional keyword arguments for Model parent class.

        Raises:
            ValueError: If parameters are invalid.
        """
        super().__init__(name=name, **kwargs)

        # Validate parameters
        self._validate_parameters(hidden_units, k, dropout_rate)

        # Store configuration parameters
        self.hidden_units = hidden_units.copy()
        self.k = k
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.use_bias = use_bias
        self.output_activation = keras.activations.get(output_activation)
        self.dropout_rate = dropout_rate
        self.batch_normalization = batch_normalization

        # Initialize layer containers
        self.hidden_layers = []
        self.dropout_layers = []
        self.batch_norm_layers = []
        self.output_layer = None

        # Build status
        self._layers_built = False
        self._build_input_shape = None

        logger.info(
            f"Initialized PowerMLP model with {len(self.hidden_units)} layers, "
            f"k={self.k}, dropout={self.dropout_rate}"
        )

    def _validate_parameters(
            self,
            hidden_units: List[int],
            k: int,
            dropout_rate: float
    ) -> None:
        """Validate initialization parameters.

        Args:
            hidden_units: List of layer sizes.
            k: Power for ReLU-k activation.
            dropout_rate: Dropout rate.

        Raises:
            ValueError: If any parameter is invalid.
        """
        if not hidden_units:
            raise ValueError("hidden_units cannot be empty")
        if any(units <= 0 for units in hidden_units):
            raise ValueError("All hidden_units must be positive")
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be in [0, 1], got {dropout_rate}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the model layers based on input shape.

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        if self._layers_built:
            return

        # Validate input shape
        if len(input_shape) < 2:
            raise ValueError(
                f"Expected at least 2D input shape [..., input_dim], got {input_shape}"
            )

        # Store input shape for serialization
        self._build_input_shape = input_shape

        logger.info(f"Building PowerMLP with input shape: {input_shape}")

        # Build hidden layers (all but the last one use PowerMLPLayer)
        self._build_hidden_layers()

        # Build output layer (last layer)
        self._build_output_layer()

        self._layers_built = True
        super().build(input_shape)

        logger.info(
            f"Built PowerMLP with {len(self.hidden_layers)} hidden layers "
            f"and {sum(self.hidden_units)} total units"
        )

    def _build_hidden_layers(self) -> None:
        """Build the hidden PowerMLP layers with optional dropout and batch norm."""
        # All layers except the last one are hidden layers
        for i, units in enumerate(self.hidden_units[:-1]):
            # PowerMLP layer
            power_mlp_layer = PowerMLPLayer(
                units=units,
                k=self.k,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                use_bias=self.use_bias,
                name=f"powermlp_hidden_{i + 1}"
            )
            self.hidden_layers.append(power_mlp_layer)

            # Batch normalization layer (optional)
            if self.batch_normalization:
                bn_layer = keras.layers.BatchNormalization(
                    name=f"batch_norm_{i + 1}"
                )
                self.batch_norm_layers.append(bn_layer)
            else:
                self.batch_norm_layers.append(None)

            # Dropout layer (optional)
            if self.dropout_rate > 0.0:
                dropout_layer = keras.layers.Dropout(
                    rate=self.dropout_rate,
                    name=f"dropout_{i + 1}"
                )
                self.dropout_layers.append(dropout_layer)
            else:
                self.dropout_layers.append(None)

    def _build_output_layer(self) -> None:
        """Build the output layer."""
        # Output layer (last element in hidden_units)
        output_units = self.hidden_units[-1]

        # Use regular Dense layer for output to allow flexible activation
        self.output_layer = keras.layers.Dense(
            units=output_units,
            activation=self.output_activation,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            use_bias=self.use_bias,
            name="output_layer"
        )

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the PowerMLP model.

        Args:
            inputs: Input tensor of shape (..., input_dim).
            training: Boolean indicating whether the model should behave in
                training mode or inference mode.

        Returns:
            Output tensor of shape (..., output_dim).
        """
        x = inputs

        # Pass through hidden layers
        for i, layer in enumerate(self.hidden_layers):
            # PowerMLP layer
            x = layer(x, training=training)

            # Optional batch normalization
            if self.batch_normalization and self.batch_norm_layers[i] is not None:
                x = self.batch_norm_layers[i](x, training=training)

            # Optional dropout
            if self.dropout_rate > 0.0 and self.dropout_layers[i] is not None:
                x = self.dropout_layers[i](x, training=training)

        # Output layer
        outputs = self.output_layer(x, training=training)

        return outputs

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the model.

        Args:
            input_shape: Shape tuple of the input.

        Returns:
            Output shape tuple with last dimension as final layer units.
        """
        input_shape_list = list(input_shape)
        return tuple(input_shape_list[:-1] + [self.hidden_units[-1]])

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization.

        Returns:
            Dictionary containing the model configuration.
        """
        config = super().get_config()
        config.update({
            "hidden_units": self.hidden_units,
            "k": self.k,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
            "use_bias": self.use_bias,
            "output_activation": keras.activations.serialize(self.output_activation),
            "dropout_rate": self.dropout_rate,
            "batch_normalization": self.batch_normalization,
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get the build configuration for serialization.

        Returns:
            Dictionary containing the build configuration.
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build the model from a build configuration.

        Args:
            config: Dictionary containing the build configuration.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "PowerMLP":
        """Create PowerMLP model from configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            PowerMLP model instance.
        """
        # Handle serialized objects
        if "kernel_initializer" in config and isinstance(config["kernel_initializer"], dict):
            config["kernel_initializer"] = keras.initializers.deserialize(
                config["kernel_initializer"]
            )
        if "bias_initializer" in config and isinstance(config["bias_initializer"], dict):
            config["bias_initializer"] = keras.initializers.deserialize(
                config["bias_initializer"]
            )
        if "kernel_regularizer" in config and config["kernel_regularizer"]:
            config["kernel_regularizer"] = keras.regularizers.deserialize(
                config["kernel_regularizer"]
            )
        if "bias_regularizer" in config and config["bias_regularizer"]:
            config["bias_regularizer"] = keras.regularizers.deserialize(
                config["bias_regularizer"]
            )
        if "output_activation" in config and isinstance(config["output_activation"], dict):
            config["output_activation"] = keras.activations.deserialize(
                config["output_activation"]
            )

        return cls(**config)

    def save_model(
            self,
            filepath: str,
            overwrite: bool = True,
            save_format: str = "keras"
    ) -> None:
        """Save the model to a file.

        Args:
            filepath: Path where to save the model.
            overwrite: Whether to overwrite existing file.
            save_format: Format to save the model in.
        """
        # Ensure directory exists
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # Save model
        self.save(filepath, overwrite=overwrite, save_format=save_format)
        logger.info(f"PowerMLP model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str) -> "PowerMLP":
        """Load a saved PowerMLP model.

        Args:
            filepath: Path to the saved model.

        Returns:
            Loaded PowerMLP model.
        """
        custom_objects = {
            "PowerMLP": cls,
            "PowerMLPLayer": PowerMLPLayer,
        }

        model = keras.models.load_model(filepath, custom_objects=custom_objects)
        logger.info(f"PowerMLP model loaded from {filepath}")
        return model

    def summary(self, **kwargs: Any) -> None:
        """Print model summary with additional PowerMLP information.

        Args:
            **kwargs: Additional keyword arguments for summary.
        """
        super().summary(**kwargs)
        logger.info("PowerMLP Configuration:")
        logger.info(f"  - Architecture: {self.hidden_units}")
        logger.info(f"  - ReLU-k power: {self.k}")
        logger.info(f"  - Total parameters: {self.count_params():,}")
        logger.info(f"  - Dropout rate: {self.dropout_rate}")
        logger.info(f"  - Batch normalization: {self.batch_normalization}")
        logger.info(f"  - Output activation: {self.output_activation}")
        logger.info(f"  - Use bias: {self.use_bias}")

    def __repr__(self) -> str:
        """Return string representation of the model.

        Returns:
            String representation including key parameters.
        """
        return (
            f"PowerMLP(hidden_units={self.hidden_units}, k={self.k}, "
            f"dropout_rate={self.dropout_rate}, name='{self.name}')"
        )

# ---------------------------------------------------------------------
# Helper function to create and compile PowerMLP model
# ---------------------------------------------------------------------

def create_power_mlp(
        hidden_units: List[int],
        k: int = 3,
        optimizer: Union[str, keras.optimizers.Optimizer] = "adam",
        learning_rate: float = 0.001,
        loss: Union[str, keras.losses.Loss] = "categorical_crossentropy",
        metrics: List[Union[str, keras.metrics.Metric]] = None,
        **kwargs
) -> PowerMLP:
    """Create and compile a PowerMLP model.

    This is a convenience function that creates a PowerMLP model and compiles it
    with the specified optimizer, loss, and metrics.

    Args:
        hidden_units: List of integers specifying the number of units in each layer.
        k: Power exponent for the ReLU-k activation function.
        optimizer: Optimizer name or instance.
        learning_rate: Learning rate for optimizer.
        loss: Loss function name or instance.
        metrics: List of metric names or instances.
        **kwargs: Additional arguments for PowerMLP constructor.

    Returns:
        Compiled PowerMLP model ready for training.

    Example:
        >>> model = create_power_mlp(
        ...     hidden_units=[128, 64, 10],
        ...     k=3,
        ...     optimizer='adam',
        ...     learning_rate=0.001,
        ...     loss='categorical_crossentropy',
        ...     metrics=['accuracy'],
        ...     dropout_rate=0.2
        ... )
        >>> model.fit(x_train, y_train, epochs=10)
    """
    # Create model
    model = PowerMLP(hidden_units=hidden_units, k=k, **kwargs)

    # Handle optimizer
    if isinstance(optimizer, str):
        optimizer = keras.optimizers.get(optimizer)
        if hasattr(optimizer, 'learning_rate'):
            optimizer.learning_rate = learning_rate

    # Default metrics
    if metrics is None:
        metrics = ['accuracy']

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    logger.info(f"Created and compiled PowerMLP model with {len(hidden_units)} layers")
    return model


# ---------------------------------------------------------------------
# Alternative factory function for regression tasks
# ---------------------------------------------------------------------

def create_power_mlp_regressor(
        hidden_units: List[int],
        k: int = 3,
        optimizer: Union[str, keras.optimizers.Optimizer] = "adam",
        learning_rate: float = 0.001,
        **kwargs
) -> PowerMLP:
    """Create and compile a PowerMLP model for regression tasks.

    Args:
        hidden_units: List of integers specifying the number of units in each layer.
        k: Power exponent for the ReLU-k activation function.
        optimizer: Optimizer name or instance.
        learning_rate: Learning rate for optimizer.
        **kwargs: Additional arguments for PowerMLP constructor.

    Returns:
        Compiled PowerMLP model configured for regression.
    """
    return create_power_mlp(
        hidden_units=hidden_units,
        k=k,
        optimizer=optimizer,
        learning_rate=learning_rate,
        loss="mse",
        metrics=["mae", "mse"],
        output_activation=None,  # Linear output for regression
        **kwargs
    )

# ---------------------------------------------------------------------
# Alternative factory function for binary classification
# ---------------------------------------------------------------------

def create_power_mlp_binary_classifier(
        hidden_units: List[int],
        k: int = 3,
        optimizer: Union[str, keras.optimizers.Optimizer] = "adam",
        learning_rate: float = 0.001,
        **kwargs
) -> PowerMLP:
    """Create and compile a PowerMLP model for binary classification.

    Args:
        hidden_units: List of integers specifying the number of units in each layer.
            The last element should be 1 for binary classification.
        k: Power exponent for the ReLU-k activation function.
        optimizer: Optimizer name or instance.
        learning_rate: Learning rate for optimizer.
        **kwargs: Additional arguments for PowerMLP constructor.

    Returns:
        Compiled PowerMLP model configured for binary classification.
    """
    # Ensure output is configured for binary classification
    if hidden_units[-1] != 1:
        logger.warning(
            f"For binary classification, output should be 1 unit, got {hidden_units[-1]}"
        )

    return create_power_mlp(
        hidden_units=hidden_units,
        k=k,
        optimizer=optimizer,
        learning_rate=learning_rate,
        loss="binary_crossentropy",
        metrics=["accuracy", "precision", "recall"],
        output_activation="sigmoid",
        **kwargs
    )

# ---------------------------------------------------------------------