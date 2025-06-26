"""
PowerMLP: Complete Reference Guide and Algorithms
===============================================

1. Overview and Core Concepts
----------------------------
PowerMLP is an efficient alternative to Kolmogorov-Arnold Networks (KAN), offering:
- ~40x faster training time
- ~10x fewer FLOPs
- Equal or better performance
- Simpler implementation

Core Algorithm - PowerMLP Layer:
-------------------------------
Algorithm PowerMLPLayer(x, units, k):
    # Main branch - ReLU-k pathway
    main = Dense(x, units)
    main = max(0, main)^k

    # Basis function branch
    basis = x / (1 + exp(-x))  # Basis function
    basis = Dense(basis, units)

    return main + basis

References:
----------
[1] "PowerMLP: An Efficient Version of KAN" (2024)
[2] "KAN: Kolmogorov-Arnold Networks" (2024)
"""

import keras
from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from .activations.relu_k import ReLUK
from .activations.basis_function import BasisFunction

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class PowerMLP(keras.layers.Layer):
    """Full PowerMLP model implementation as a Keras layer.

    PowerMLP is an efficient alternative to Kolmogorov-Arnold Networks (KAN) that provides
    superior performance with significantly reduced computational requirements.

    Args:
        hidden_units: List of integers specifying the number of units in each layer.
        k: Power for ReLU-k activation function.
        kernel_initializer: Initializer for the kernel weights.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer for the kernel weights.
        bias_regularizer: Regularizer for the bias vector.
        use_bias: Whether to use bias in dense layers.
        output_activation: Activation function for the output layer.
        **kwargs: Additional keyword arguments for the Layer parent class.
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
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if not hidden_units:
            raise ValueError("hidden_units cannot be empty")
        if any(units <= 0 for units in hidden_units):
            raise ValueError("All hidden_units must be positive")
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

        # Store configuration parameters
        self.hidden_units = hidden_units
        self.k = k
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.use_bias = use_bias
        self.output_activation = keras.activations.get(output_activation)

        # Will be initialized in build()
        self.hidden_layers = []
        self.output_layer = None
        self._build_input_shape = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the model layers.

        Args:
            input_shape: Shape of the input tensor.
        """
        self._build_input_shape = input_shape

        # Build hidden layers
        self.hidden_layers = []
        current_shape = input_shape

        for i, units in enumerate(self.hidden_units[:-1]):
            layer = PowerMLPLayer(
                units=units,
                k=self.k,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                use_bias=self.use_bias,
                name=f"powermlp_layer_{i}"
            )
            layer.build(current_shape)
            self.hidden_layers.append(layer)

            # Update current shape for next layer
            current_shape = layer.compute_output_shape(current_shape)

        # Build output layer
        self.output_layer = keras.layers.Dense(
            units=self.hidden_units[-1],
            activation=self.output_activation,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            use_bias=self.use_bias,
            name="output_layer"
        )
        self.output_layer.build(current_shape)

        super().build(input_shape)

        logger.info(f"Built PowerMLP with {len(self.hidden_layers)} hidden layers and {sum(self.hidden_units)} total units")

    def call(self, inputs: Union[keras.KerasTensor, Any], training: Optional[bool] = None) -> keras.KerasTensor:
        """Forward pass of the model.

        Args:
            inputs: Input tensor.
            training: Boolean indicating whether the layer should behave in training mode.

        Returns:
            Output tensor.
        """
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x, training=training)
        return self.output_layer(x, training=training)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input.

        Returns:
            Output shape with last dimension as final hidden unit count.
        """
        input_shape_list = list(input_shape)
        return tuple(input_shape_list[:-1] + [self.hidden_units[-1]])

    def get_config(self) -> Dict[str, Any]:
        """Returns the config of the layer.

        Returns:
            Dictionary containing the layer configuration.
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
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get the config needed to build the layer from a config.

        Returns:
            Dictionary containing the build configuration.
        """
        return {"input_shape": self._build_input_shape}

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build the layer from a config created with get_build_config.

        Args:
            config: Dictionary containing the build configuration.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "PowerMLP":
        """Create a PowerMLP layer from a configuration dictionary.

        Args:
            config: Configuration dictionary.

        Returns:
            PowerMLP layer instance.
        """
        return cls(**config)

# ---------------------------------------------------------------------

