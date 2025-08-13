"""
This module defines the `PowerMLPLayer`, a custom Keras layer designed to enhance
the expressive power of traditional Multi-Layer Perceptrons (MLPs).

At its core, the `PowerMLPLayer` implements a dual-branch architecture that
processes the input in parallel and combines the results. This design leverages
the complementary strengths of two different types of non-linear transformations,
allowing the layer to model complex functions more effectively than a standard
`Dense` layer.

The two parallel branches are:

1.  **The Main Branch (Piecewise Polynomial Power):**
    -   **Path:** `Input → Dense → ReLU-k`
    -   **Purpose:** This branch uses the `ReLUK` activation (`max(0, x)^k`), which is
        a generalization of the standard ReLU. It excels at modeling sharp,
        piecewise polynomial relationships within the data. The integer power `k`
        controls the degree and aggressiveness of the non-linearity. This branch
        includes a bias term in its dense layer.

2.  **The Basis Branch (Smooth Function Approximation):**
    -   **Path:** `Input → BasisFunction → Dense`
    -   **Purpose:** This branch first transforms the input using a set of smooth,
        continuous basis functions (e.g., sinusoids or other periodic/non-periodic
        functions). This allows the layer to efficiently model smooth, global trends
        and periodic patterns that are often difficult to capture with ReLU-based
        activations alone. This branch's dense layer intentionally omits a bias term.

The outputs of these two branches are then combined through element-wise addition.
This fusion allows the `PowerMLPLayer` to simultaneously learn both sharp, localized
features (from the main branch) and smooth, global patterns (from the basis branch).
leading to superior function approximation capabilities and potentially improved
gradient flow compared to a standard MLP.
"""

import keras
from typing import Optional, Union, Any, Dict, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from ..activations.relu_k import ReLUK
from ..activations.basis_function import BasisFunction

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
        units: Integer, number of output units/neurons in the layer. Must be positive.
        k: Integer, power exponent for the ReLU-k activation function.
            Must be positive. Higher values create more aggressive non-linearities.
            Defaults to 3.
        kernel_initializer: Initializer for the kernel weights in both branches.
            Can be string name or Initializer instance. Defaults to 'he_normal'.
        bias_initializer: Initializer for the bias vector in the main branch.
            Can be string name or Initializer instance. Defaults to 'zeros'.
        kernel_regularizer: Optional regularizer function applied to kernel weights
            in both branches.
        bias_regularizer: Optional regularizer function applied to bias vector
            in the main branch.
        use_bias: Boolean, whether to use bias in the main branch dense layer.
            The basis branch never uses bias by design. Defaults to True.
        **kwargs: Additional keyword arguments passed to the Layer parent class,
            such as `name`, `dtype`, `trainable`, etc.

    Input shape:
        N-D tensor with shape: `(..., input_dim)`.
        Most common case would be 2D input with shape `(batch_size, input_dim)`.

    Output shape:
        N-D tensor with shape: `(..., units)`.
        For instance, for 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.

    Raises:
        TypeError: If units or k are not integers.
        ValueError: If units or k are not positive.

    Example:
        ```python
        # Create a PowerMLP layer
        power_mlp = PowerMLPLayer(units=64, k=3)

        # Use in a model
        model = keras.Sequential([
            keras.layers.Input(shape=(784,)),
            PowerMLPLayer(units=128, k=2),
            keras.layers.Dropout(0.2),
            PowerMLPLayer(units=64, k=3),
            keras.layers.Dense(10, activation='softmax')
        ])

        # Or in functional API
        inputs = keras.Input(shape=(784,))
        x = PowerMLPLayer(units=128, k=2)(inputs)
        x = keras.layers.LayerNormalization()(x)
        outputs = PowerMLPLayer(units=10, k=1)(x)
        model = keras.Model(inputs, outputs)
        ```

    References:
        - PowerMLP architectures for enhanced function approximation
        - Combines benefits of ReLU-k and basis function activations
        - Designed for improved gradient flow and expressiveness

    Note:
        - The main branch uses bias while the basis branch does not
        - Both branches share the same kernel initializer and regularizer
        - The layer requires both ReLUK and BasisFunction to be available
        - For k=1, this reduces to standard ReLU in the main branch
        - This implementation follows the modern Keras 3 pattern where sub-layers
          are created in __init__ and Keras handles the build lifecycle automatically.
          This ensures proper serialization and avoids common build errors.
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

        # Store ALL configuration parameters as instance attributes
        self.units = units
        self.k = k
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.use_bias = use_bias

        # CREATE all sub-layers here in __init__ (MODERN KERAS 3 PATTERN)
        logger.info(f"Creating PowerMLP sublayers: units={units}, k={k}")

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

        # No custom weights to create, so no build() method is needed
        # Keras will automatically handle building of sub-layers

        logger.info(f"Initialized PowerMLP layer with {units} units, k={k}")

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
        # Convert to list for manipulation, then back to tuple
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """Get the layer configuration for serialization.

        This method must return ALL arguments needed to recreate the layer
        via __init__. Uses keras serializers for complex objects.

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

    # DELETED: get_build_config() and build_from_config() methods
    # These are deprecated in Keras 3 and cause serialization issues
    # Keras handles the build lifecycle automatically with the modern pattern

    def __repr__(self) -> str:
        """Return string representation of the layer.

        Returns:
            String representation including key parameters.
        """
        return f"PowerMLPLayer(units={self.units}, k={self.k}, name='{self.name}')"

# ---------------------------------------------------------------------