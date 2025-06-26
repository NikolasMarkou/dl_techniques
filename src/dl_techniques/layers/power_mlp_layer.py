"""PowerMLP layer implementation for Keras 3.x.

This module implements the PowerMLP layer architecture that combines ReLU-k
activations with basis functions for enhanced function approximation capabilities.
"""

import keras
from typing import Optional, Union, Any, Dict, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.activations.relu_k import ReLUK
from dl_techniques.layers.activations.basis_function import BasisFunction

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class PowerMLPLayer(keras.layers.Layer):
    """PowerMLP layer with dual-branch architecture for enhanced expressiveness.

    This layer implements the PowerMLP architecture, which combines two parallel
    branches to achieve superior function approximation capabilities compared to
    standard dense layers:

    1. **Main Branch**: Dense → ReLU-k activation
    2. **Basis Branch**: BasisFunction → Dense (no bias)

    The outputs are combined via element-wise addition:
        output = main_branch + basis_branch

    Where:
        - main_branch = ReLU-k(Dense(x))
        - basis_branch = Dense(BasisFunction(x))

    The PowerMLP architecture leverages the complementary properties of ReLU-k
    (for capturing sharp, piecewise patterns) and basis functions (for smooth,
    differentiable transformations) to model complex non-linear relationships
    more effectively than traditional MLPs.

    Args:
        units: Integer, number of output units/neurons in the layer.
        k: Integer, power exponent for the ReLU-k activation function.
            Must be positive. Higher values create more aggressive non-linearities.
        kernel_initializer: Initializer for the kernel weights in both branches.
            Can be string name or Initializer instance.
        bias_initializer: Initializer for the bias vector in the main branch.
            Can be string name or Initializer instance.
        kernel_regularizer: Optional regularizer function applied to kernel weights
            in both branches.
        bias_regularizer: Optional regularizer function applied to bias vector
            in the main branch.
        use_bias: Boolean, whether to use bias in the main branch dense layer.
            The basis branch never uses bias by design.
        **kwargs: Additional keyword arguments passed to the Layer parent class,
            such as `name`, `dtype`, `trainable`, etc.

    Input shape:
        N-D tensor with shape: `(..., input_dim)`.
        Most common case would be 2D input with shape `(batch_size, input_dim)`.

    Output shape:
        N-D tensor with shape: `(..., units)`.
        For instance, for 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.

    Example:
        >>>
        >>> # Create a PowerMLP layer
        >>> power_mlp = PowerMLPLayer(units=64, k=3)
        >>>
        >>> # Use in a model
        >>> model = keras.Sequential([
        ...     keras.layers.Input(shape=(784,)),
        ...     PowerMLPLayer(units=128, k=2),
        ...     keras.layers.Dropout(0.2),
        ...     PowerMLPLayer(units=64, k=3),
        ...     keras.layers.Dense(10, activation='softmax')
        ... ])
        >>>
        >>> # Or in functional API
        >>> inputs = keras.Input(shape=(784,))
        >>> x = PowerMLPLayer(units=128, k=2)(inputs)
        >>> x = keras.layers.LayerNormalization()(x)
        >>> outputs = PowerMLPLayer(units=10, k=1)(x)
        >>> model = keras.Model(inputs, outputs)

    References:
        - PowerMLP architectures for enhanced function approximation
        - Combines benefits of ReLU-k and basis function activations
        - Designed for improved gradient flow and expressiveness

    Note:
        - The main branch uses bias while the basis branch does not
        - Both branches share the same kernel initializer and regularizer
        - The layer requires both ReLUK and BasisFunction to be available
        - For k=1, this reduces to standard ReLU in the main branch
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
        """Initialize the PowerMLP layer.

        Args:
            units: Number of output units.
            k: Power for ReLU-k activation.
            kernel_initializer: Initializer for kernel weights.
            bias_initializer: Initializer for bias vector.
            kernel_regularizer: Regularizer for kernel weights.
            bias_regularizer: Regularizer for bias vector.
            use_bias: Whether to use bias in main branch.
            **kwargs: Additional keyword arguments for Layer parent class.

        Raises:
            ValueError: If units or k are not positive integers.
            TypeError: If k is not an integer.
        """
        super().__init__(**kwargs)

        # Validate parameters
        if not isinstance(units, int):
            raise TypeError(f"units must be an integer, got type {type(units).__name__}")
        if units <= 0:
            raise ValueError(f"units must be positive, got {units}")

        if not isinstance(k, int):
            raise TypeError(f"k must be an integer, got type {type(k).__name__}")
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

        # Sublayers - will be initialized in build()
        self.main_dense = None
        self.relu_k = None
        self.basis_function = None
        self.basis_dense = None
        self._build_input_shape = None

        logger.info(f"Initialized PowerMLP layer with {units} units, k={k}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer weights and initialize sublayers.

        This method creates and configures the four main components:
        1. Main branch dense layer
        2. ReLU-k activation layer
        3. Basis function activation layer
        4. Basis branch dense layer

        Args:
            input_shape: Shape tuple of the input tensor, including the batch
                dimension as None or an integer.
        """
        # Store input shape for serialization
        self._build_input_shape = input_shape

        # Main branch dense layer (with bias)
        self.main_dense = keras.layers.Dense(
            units=self.units,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="main_dense"
        )

        # ReLU-k activation for main branch
        self.relu_k = ReLUK(k=self.k, name="relu_k")

        # Basis function activation
        self.basis_function = BasisFunction(name="basis_function")

        # Basis branch dense layer (no bias by design)
        self.basis_dense = keras.layers.Dense(
            units=self.units,
            use_bias=False,  # Basis branch doesn't use bias
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="basis_dense"
        )

        # Build all sublayers with the appropriate shapes
        self.main_dense.build(input_shape)
        self.relu_k.build((input_shape[0], self.units))  # After main_dense
        self.basis_function.build(input_shape)
        self.basis_dense.build(input_shape)  # After basis_function

        super().build(input_shape)

        logger.debug(f"Built PowerMLP layer with input shape: {input_shape}")

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass implementing the dual-branch PowerMLP architecture.

        Computes:
            main_branch = ReLU-k(Dense(inputs))
            basis_branch = Dense(BasisFunction(inputs))
            output = main_branch + basis_branch

        Args:
            inputs: Input tensor of shape (..., input_dim).
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            Output tensor of shape (..., units) after combining both branches.
        """
        # Main branch: Dense → ReLU-k
        main_branch = self.main_dense(inputs, training=training)
        main_branch = self.relu_k(main_branch, training=training)

        # Basis branch: BasisFunction → Dense
        basis_branch = self.basis_function(inputs, training=training)
        basis_branch = self.basis_dense(basis_branch, training=training)

        # Combine branches via addition
        output = main_branch + basis_branch

        return output

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape tuple of the input tensor.

        Returns:
            Output shape tuple with the last dimension replaced by units.
        """
        # Convert to list for manipulation
        input_shape_list = list(input_shape)

        # Replace last dimension with units
        output_shape_list = input_shape_list[:-1] + [self.units]

        # Return as tuple
        return tuple(output_shape_list)

    def get_config(self) -> Dict[str, Any]:
        """Get the layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration, including all
            constructor parameters and parent class configuration.
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
        """Get the build configuration for serialization.

        Returns:
            Dictionary containing the build configuration, specifically
            the input shape used during the build process.
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build the layer from a build configuration.

        This method is used during model loading to properly rebuild
        the layer's internal state and sublayers.

        Args:
            config: Dictionary containing the build configuration,
                as returned by get_build_config().
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

    def __repr__(self) -> str:
        """Return string representation of the layer.

        Returns:
            String representation including key parameters.
        """
        return f"PowerMLPLayer(units={self.units}, k={self.k}, name='{self.name}')"

# ---------------------------------------------------------------------
