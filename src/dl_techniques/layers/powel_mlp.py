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
from keras import ops
from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@dataclass
class PowerMLPConfig:
    """Configuration class for PowerMLP hyperparameters.

    Args:
        hidden_units: List of integers specifying the number of units in each hidden layer.
        k: Power for ReLU-k activation function.
        kernel_initializer: Initializer for the kernel weights.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer for the kernel weights.
        bias_regularizer: Regularizer for the bias vector.
        use_bias: Whether to use bias in dense layers.
        output_activation: Activation function for the output layer.
    """
    hidden_units: List[int]
    k: int = 3
    kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal"
    bias_initializer: Union[str, keras.initializers.Initializer] = "zeros"
    kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None
    bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None
    use_bias: bool = True
    output_activation: Optional[Union[str, callable]] = None

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class ReLUK(keras.layers.Layer):
    """ReLU-k activation layer implementing f(x) = max(0,x)^k.

    This layer applies a powered ReLU activation function which is more expressive
    than standard ReLU while maintaining computational efficiency.

    Args:
        k: Power for ReLU function. Must be positive integer.
        **kwargs: Additional keyword arguments for the Layer parent class.
    """

    def __init__(self, k: int = 3, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        self.k = k
        self._build_input_shape = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer.

        Args:
            input_shape: Shape of the input tensor.
        """
        self._build_input_shape = input_shape
        super().build(input_shape)

    def call(self, inputs: Union[keras.KerasTensor, Any], training: Optional[bool] = None) -> keras.KerasTensor:
        """Forward pass of the layer.

        Args:
            inputs: Input tensor.
            training: Boolean indicating whether the layer should behave in training mode.

        Returns:
            Output tensor after ReLU-k activation.
        """
        relu_output = ops.maximum(0.0, inputs)
        return ops.power(relu_output, self.k)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input.

        Returns:
            Output shape (same as input shape).
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Returns the config of the layer.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({"k": self.k})
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

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class BasisFunction(keras.layers.Layer):
    """Basis function layer implementing b(x) = x/(1 + e^(-x)).

    This layer implements the basis function branch of PowerMLP, which enhances
    the expressiveness of the network by capturing complex relationships.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._build_input_shape = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer.

        Args:
            input_shape: Shape of the input tensor.
        """
        self._build_input_shape = input_shape
        super().build(input_shape)

    def call(self, inputs: Union[keras.KerasTensor, Any], training: Optional[bool] = None) -> keras.KerasTensor:
        """Forward pass of the layer.

        Args:
            inputs: Input tensor.
            training: Boolean indicating whether the layer should behave in training mode.

        Returns:
            Output tensor after basis function transformation.
        """
        return inputs / (1.0 + ops.exp(-inputs))

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input.

        Returns:
            Output shape (same as input shape).
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Returns the config of the layer.

        Returns:
            Dictionary containing the layer configuration.
        """
        return super().get_config()

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

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class PowerMLPLayer(keras.layers.Layer):
    """Single layer of PowerMLP with ReLU-k activation and basis function.

    This layer combines a main branch with ReLU-k activation and a basis function branch
    to achieve superior function approximation capabilities compared to standard dense layers.

    Args:
        units: Number of output units.
        k: Power for ReLU-k activation.
        kernel_initializer: Initializer for the kernel weights.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer for the kernel weights.
        bias_regularizer: Regularizer for the bias vector.
        use_bias: Whether to use bias in dense layers.
        **kwargs: Additional keyword arguments for the Layer parent class.
    """

    def __init__(
        self,
        units: int,
        k: int = 3,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
        bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        use_bias: bool = True,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if units <= 0:
            raise ValueError(f"units must be positive, got {units}")
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

        # Store configuration parameters
        self.units = units
        self.k = k
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.use_bias = use_bias

        # Will be initialized in build()
        self.main_dense = None
        self.relu_k = None
        self.basis_function = None
        self.basis_dense = None
        self._build_input_shape = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer weights and sublayers.

        Args:
            input_shape: Shape of the input tensor.
        """
        self._build_input_shape = input_shape

        # Main branch dense layer
        self.main_dense = keras.layers.Dense(
            units=self.units,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="main_dense"
        )

        # ReLU-k activation
        self.relu_k = ReLUK(k=self.k, name="relu_k")

        # Basis function
        self.basis_function = BasisFunction(name="basis_function")

        # Basis projection layer (no bias for basis branch)
        self.basis_dense = keras.layers.Dense(
            units=self.units,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="basis_dense"
        )

        # Build sublayers
        self.main_dense.build(input_shape)
        self.relu_k.build(input_shape)
        self.basis_function.build(input_shape)
        self.basis_dense.build(input_shape)

        super().build(input_shape)

        logger.info(f"Built PowerMLPLayer with {self.units} units, k={self.k}")

    def call(self, inputs: Union[keras.KerasTensor, Any], training: Optional[bool] = None) -> keras.KerasTensor:
        """Forward pass of the layer.

        Args:
            inputs: Input tensor.
            training: Boolean indicating whether the layer should behave in training mode.

        Returns:
            Output tensor combining main and basis branches.
        """
        # Main branch with ReLU-k
        main = self.main_dense(inputs, training=training)
        main = self.relu_k(main, training=training)

        # Basis function branch
        basis = self.basis_function(inputs, training=training)
        basis = self.basis_dense(basis, training=training)

        return main + basis

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input.

        Returns:
            Output shape with last dimension as units.
        """
        input_shape_list = list(input_shape)
        return tuple(input_shape_list[:-1] + [self.units])

    def get_config(self) -> Dict[str, Any]:
        """Returns the config of the layer.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "units": self.units,
            "k": self.k,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
            "use_bias": self.use_bias,
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

