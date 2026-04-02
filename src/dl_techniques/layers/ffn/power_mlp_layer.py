"""
A dual-branch MLP for enhanced function approximation.

This layer enhances the expressive power of a standard Multi-Layer Perceptron
(MLP) by implementing a parallel, dual-branch architecture. The core principle
is that different classes of functions are better suited for modeling
different types of patterns. By combining two complementary non-linear
pathways, this layer can approximate complex functions more efficiently and
effectively than a traditional MLP with a single activation function.

The design is rooted in function approximation theory, where a complex
function can be decomposed into a sum of simpler, constituent functions. This
layer operationalizes that idea by fusing a piecewise polynomial approximator
with a smooth, global function approximator.

Architectural Overview:
The layer processes the input through two parallel branches whose outputs are
summed element-wise:

1.  **Main Branch (Piecewise Polynomial Pathway)**: This branch follows a
    `Dense -> ReLUK` structure. It is designed to capture sharp, localized,
    and non-linear patterns. The `ReLUK` activation, defined as
    `max(0, x)^k`, generalizes the standard ReLU. For `k > 1`, it produces a
    smoother, higher-order polynomial function in the positive domain,
    allowing the network to model more complex local curvatures.

2.  **Basis Branch (Smooth Function Pathway)**: This branch follows a
    `BasisFunction -> Dense` structure. It first projects the input onto a
    set of smooth, continuous basis functions (e.g., sinusoids). A subsequent
    linear layer then learns to combine these basis functions. This pathway
    excels at modeling smooth, global trends and periodic patterns, which are
    often difficult to represent efficiently with piecewise-linear activations
    like ReLU.

The final output is the sum of these two pathways, allowing the layer to
simultaneously learn both sharp, discontinuous features and smooth, continuous
trends within the data.

Foundational Mathematics:
Let `x` be the input vector. The layer's output `y` is the sum of the
outputs from the two branches:

`y = y_main + y_basis`

The computation for each branch is as follows:

-   **Main Branch**: `y_main = (max(0, W_m @ x + b_m))^k`
    Here, `W_m` and `b_m` are the weights and bias of the main dense layer,
    and `k` is the integer power of the `ReLUK` activation. This branch
    learns a function that is a piecewise polynomial of degree `k`.

-   **Basis Branch**: `y_basis = W_b @ φ(x)`
    Here, `φ(x)` represents the transformation by a set of basis functions
    (e.g., `[x, sin(x), cos(x), sin(2x), ... ]`). `W_b` is the weight matrix of
    the basis dense layer (which intentionally lacks a bias term), learning a
    linear combination of these basis functions.

This formulation effectively models the target function as a superposition of
a piecewise polynomial and a function from the span of the chosen basis
functions (akin to a truncated Fourier series for sinusoidal bases).

References:
This architecture synthesizes principles from several areas of machine
learning and approximation theory:

-   The concept of using polynomial activations to increase expressiveness is
    related to works on higher-order neural networks.
-   The basis function branch is directly inspired by classical function
    approximation methods, such as Radial Basis Function (RBF) Networks and
    Fourier series analysis.
    -   Broomhead, D. S., & Lowe, D. (1988). Multivariable functional
        interpolation and adaptive networks. Complex Systems.
-   The idea of combining different activation functions within a network has
    been explored to improve performance, for example, in:
    -   Ramachandran, P., Zoph, B., & Le, Q. V. (2017). Searching for
        Activation Functions. arXiv preprint arXiv:1710.05941.

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
    """
    PowerMLP layer with dual-branch architecture for enhanced expressiveness.

    This layer combines two parallel branches -- a piecewise polynomial pathway
    (Dense followed by ReLU-k activation, ``y_main = (max(0, W @ x + b))^k``) and a
    smooth function pathway (BasisFunction followed by Dense, ``y_basis = W_b @ phi(x)``)
    -- whose outputs are summed element-wise: ``output = y_main + y_basis``. This
    enables the layer to simultaneously capture sharp localized patterns and smooth
    global trends.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────┐
        │    Input (..., input_dim)    │
        └──────────────┬───────────────┘
                       │
                 ┌─────┴─────┐
                 ▼           ▼
        ┌──────────────┐ ┌──────────────────┐
        │  main_dense  │ │  basis_function  │
        │   (Dense)    │ │ (BasisFunction)  │
        └──────┬───────┘ └────────┬─────────┘
               ▼                  ▼
        ┌──────────────┐ ┌──────────────────┐
        │   ReLU-k     │ │   basis_dense    │
        │  max(0,x)^k  │ │  (Dense, no bias)│
        └──────┬───────┘ └────────┬─────────┘
               │                  │
               └────────┬─────────┘
                        ▼
        ┌──────────────────────────────┐
        │     Element-wise Addition    │
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │    Output (..., units)       │
        └──────────────────────────────┘

    :param units: Integer, number of output units/neurons in the layer. Must be positive.
    :type units: int
    :param k: Integer, power exponent for the ReLU-k activation function.
        Must be positive. Higher values create more aggressive non-linearities.
        Defaults to 3.
    :type k: int
    :param kernel_initializer: Initializer for the kernel weights in both branches.
        Can be string name or Initializer instance. Defaults to 'he_normal'.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for the bias vector in the main branch.
        Can be string name or Initializer instance. Defaults to 'zeros'.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer function applied to kernel weights
        in both branches. Defaults to None.
    :type kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param bias_regularizer: Optional regularizer function applied to bias vector
        in the main branch. Defaults to None.
    :type bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param use_bias: Whether to use bias in the main branch dense layer.
        The basis branch never uses bias by design. Defaults to True.
    :type use_bias: bool
    :param kwargs: Additional keyword arguments passed to the Layer parent class,
        such as ``name``, ``dtype``, ``trainable``, etc.

    :raises ValueError: If units or k are not positive integers.
    :raises TypeError: If k is not an integer.

    Note:
        The main branch uses bias while the basis branch does not.
        Both branches share the same kernel initializer and regularizer.
        For k=1, this reduces to standard ReLU in the main branch.
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
        """
        Initialize the PowerMLP layer.

        :param units: Number of output units. Must be positive.
        :type units: int
        :param k: Power for ReLU-k activation. Must be positive integer.
        :type k: int
        :param kernel_initializer: Initializer for kernel weights.
        :type kernel_initializer: Union[str, keras.initializers.Initializer]
        :param bias_initializer: Initializer for bias vector.
        :type bias_initializer: Union[str, keras.initializers.Initializer]
        :param kernel_regularizer: Regularizer for kernel weights.
        :type kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
        :param bias_regularizer: Regularizer for bias vector.
        :type bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
        :param use_bias: Whether to use bias in main branch.
        :type use_bias: bool
        :param kwargs: Additional keyword arguments for Layer parent class.
        :raises ValueError: If units or k are not positive integers.
        :raises TypeError: If k is not an integer.
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

        # CREATE all sub-layers in __init__ (following modern Keras 3 pattern)
        # These are unbuilt until build() is called

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

        logger.info(f"Initialized PowerMLP layer with {units} units, k={k}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer weights and initialize sublayers.

        Explicitly builds all sub-layers to ensure robust serialization
        following the modern Keras 3 pattern for composite layers.

        :param input_shape: Shape tuple of the input tensor, including the batch
            dimension as None or an integer.
        :type input_shape: Tuple[Optional[int], ...]
        """
        # Build sub-layers in computational order for robust serialization
        self.main_dense.build(input_shape)

        # Compute intermediate shapes for proper building
        main_dense_output_shape = self.main_dense.compute_output_shape(input_shape)
        self.relu_k.build(main_dense_output_shape)

        self.basis_function.build(input_shape)
        # basis_function output shape is the same as input shape
        self.basis_dense.build(input_shape)

        # Always call parent build at the end
        super().build(input_shape)

        logger.debug(f"Built PowerMLP layer with input shape: {input_shape}")

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass implementing the dual-branch PowerMLP architecture.

        Computes ``output = ReLU_k(Dense(x)) + Dense(BasisFunction(x))``.

        :param inputs: Input tensor of shape (..., input_dim).
        :type inputs: keras.KerasTensor
        :param training: Whether the layer should behave in training mode
            or inference mode.
        :type training: Optional[bool]
        :return: Output tensor of shape (..., units) after combining both branches.
        :rtype: keras.KerasTensor
        """
        # Main branch: Dense -> ReLU-k
        main_branch = self.main_dense(inputs, training=training)
        main_branch = self.relu_k(main_branch, training=training)

        # Basis branch: BasisFunction -> Dense
        basis_branch = self.basis_function(inputs, training=training)
        basis_branch = self.basis_dense(basis_branch, training=training)

        # Combine branches via element-wise addition
        output = main_branch + basis_branch

        return output

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple with the last dimension replaced by units.
        :rtype: Tuple[Optional[int], ...]
        """
        # Convert to list for manipulation
        input_shape_list = list(input_shape)

        # Replace last dimension with units
        output_shape_list = input_shape_list[:-1] + [self.units]

        # Return as tuple
        return tuple(output_shape_list)

    def get_config(self) -> Dict[str, Any]:
        """
        Get the layer configuration for serialization.

        :return: Dictionary containing the layer configuration, including all
            constructor parameters and parent class configuration.
        :rtype: Dict[str, Any]
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

    def __repr__(self) -> str:
        """
        Return string representation of the layer.

        :return: String representation including key parameters.
        :rtype: str
        """
        return f"PowerMLPLayer(units={self.units}, k={self.k}, name='{self.name}')"

# ---------------------------------------------------------------------
