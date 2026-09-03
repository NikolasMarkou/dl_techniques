"""
A dual-branch MLP layer, built by :class:`PowerMLPLayer`: a ReLU-k branch
plus a Swish branch, summed.

The main branch is ``Dense -> ReLUK``, computing
``y_main = (max(0, W_m @ x + b_m))^k``; at ``k = 1`` it is a plain ReLU, and
at ``k > 1`` the positive half becomes a degree-``k`` polynomial. The basis
branch is ``BasisFunction -> Dense``, computing ``y_basis = W_b @ swish(x)``.
``BasisFunction`` is the Swish activation and nothing more -- element-wise,
no weights, not a basis expansion of any kind. The two branches sum to a
sharp polynomial-tailed rectifier plus a smooth one: ReLU-k is zero and flat
for negative inputs, so the basis branch still carries a signal there. Only
the main branch's Dense has a bias.

References:
    - Ramachandran et al., 2017. Searching for Activation Functions.
      (https://arxiv.org/abs/1710.05941)
"""

import keras
from typing import Optional, Union, Any, Dict, Tuple

from dl_techniques.initializers.clone import clone_initializer
from dl_techniques.utils.logger import logger
from ..activations.relu_k import ReLUK
from ..activations.basis_function import BasisFunction
from dl_techniques.utils.keras_registration import register_dl_technique


@register_dl_technique("dl_techniques.layers.ffn.power_mlp_layer")
class PowerMLPLayer(keras.layers.Layer):
    """
    PowerMLP layer: a ReLU-k branch and a Swish branch, added together.

    The main branch is ``Dense -> ReLUK``: ``y_main = (max(0, W_m @ x + b_m))^k``.
    The basis branch is ``BasisFunction -> Dense``: ``y_basis = W_b @ swish(x)``,
    where ``swish(x) = x / (1 + exp(-x))``. The output is ``y_main + y_basis``.

    ``BasisFunction`` is the Swish activation. It is element-wise and holds no
    weights. It is not a basis expansion of any kind; see the module docstring.

    Architecture:

    .. code-block:: text

        ┌──────────────────────────────┐
        │    Input (..., input_dim)    │
        └──────────────┬───────────────┘
                       │
                 ┌─────┴─────┐
                 ▼           ▼
        ┌──────────────┐ ┌──────────────────┐
        │  main_dense  │ │  basis_function  │
        │Dense(units)  │ │  Swish, no wts   │
        └──────┬───────┘ └────────┬─────────┘
               ▼                  ▼
        ┌──────────────┐ ┌──────────────────┐
        │   relu_k     │ │   basis_dense    │
        │  max(0,x)^k  │ │Dense(units),no b │
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

    The two branches (block internals):

    .. code-block:: text

        x [.., D_in]
        │
        ├─► main_dense   W_m (D_in x units) + b_m  ─► [.., units]
        │   relu_k       max(0, ·)^k               ─► [.., units]
        │
        └─► basis_func   x / (1 + exp(-x))         ─► [.., D_in]
            basis_dense  W_b (D_in x units), no bias ─► [.., units]

        sum: [.., units] + [.., units] ─► [.., units]

        The branches differ only in their non-linearity and in
        where it sits. The main branch rectifies after the
        projection, so its output is >= 0 before the sum. The
        basis branch rectifies before the projection, so its
        output is signed. relu_k is exactly 0 for every negative
        pre-activation; swish is not, so the basis branch still
        carries a signal there. relu_k with k=1 skips the power
        step and is plain ReLU. Only main_dense has a bias, and
        only when use_bias is True.

    :param units: Integer, number of output units/neurons in the layer. Must be positive.
    :type units: int
    :param k: Integer, power exponent for the ReLU-k activation function.
        Must be positive. Higher values create more aggressive non-linearities.
        Defaults to 3.
    :type k: int
    :param kernel_initializer: Initializer for the kernel weights of both
        branches. Can be a string name or an Initializer instance. Each branch
        gets its own clone of it, so the two kernels are drawn independently.
        Defaults to 'he_normal'.
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
    :type kwargs: Any

    :ivar units: The stored output width. Both branches emit this width.
    :vartype units: int
    :ivar k: The stored ReLU-k exponent.
    :vartype k: int
    :ivar kernel_initializer: The resolved kernel initializer. ``main_dense``
        gets this instance; ``basis_dense`` gets a clone of it.
    :vartype kernel_initializer: keras.initializers.Initializer
    :ivar bias_initializer: The resolved bias initializer. Only ``main_dense``
        uses it.
    :vartype bias_initializer: keras.initializers.Initializer
    :ivar kernel_regularizer: The resolved kernel regularizer, or ``None``.
        The same object goes to both branches.
    :vartype kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :ivar bias_regularizer: The resolved bias regularizer, or ``None``. Only
        ``main_dense`` gets it, because only it has a bias.
    :vartype bias_regularizer: Optional[keras.regularizers.Regularizer]
    :ivar use_bias: Whether ``main_dense`` carries a bias. It never affects
        ``basis_dense``, which has no bias either way.
    :vartype use_bias: bool
    :ivar main_dense: The main branch projection.
    :vartype main_dense: keras.layers.Dense
    :ivar relu_k: The ``max(0, x)^k`` activation layer.
    :vartype relu_k: ReLUK
    :ivar basis_function: The Swish activation layer. No weights.
    :vartype basis_function: BasisFunction
    :ivar basis_dense: The basis branch projection. No bias.
    :vartype basis_dense: keras.layers.Dense

    :raises TypeError: If ``units`` is not an ``int``.
    :raises TypeError: If ``k`` is not an ``int``.
    :raises ValueError: If ``units`` is not positive.
    :raises ValueError: If ``k`` is not positive.

    Input shape:
        Tensor of shape ``(batch_size, ..., input_dim)``. Any rank of 2 or
        more works; both branches act on the last axis only.

    Output shape:
        Same shape as the input with the last axis set to ``units``.

    Example:
        .. code-block:: python

            layer = PowerMLPLayer(units=32, k=3)
            y = layer(keras.random.normal((4, 16)))
            y.shape                 # (4, 32)

    Note:
        Only the main branch has a bias; the basis branch never does. At
        ``k = 1`` the main branch is a plain ReLU and the layer is the sum of
        a ReLU MLP and a Swish MLP.
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
        Validate the configuration and create the four sub-layers.

        Every argument is documented on the class. The type checks run before
        the range checks, so a non-integer ``units`` or ``k`` raises
        ``TypeError``, never ``ValueError``. ``bool`` is a subclass of ``int``
        in Python, so ``units=True`` passes both checks and builds a layer of
        width 1.

        :raises TypeError: If ``units`` or ``k`` is not an ``int``.
        :raises ValueError: If ``units`` or ``k`` is not positive.
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

        # Create all sub-layers here; build() builds them.

        # Main branch projection. Carries the bias, if any.
        self.main_dense = keras.layers.Dense(
            units=self.units,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="main_dense"
        )

        # Main branch activation.
        self.relu_k = ReLUK(k=self.k, name="relu_k")

        # Basis branch activation. Swish, element-wise, no weights.
        self.basis_function = BasisFunction(name="basis_function")

        # Basis branch projection. Never has a bias.

        # DECISION plan-2026-08-19T163559-499b6f0e/D-057: clone_initializer for
        # basis_dense, never self.kernel_initializer directly -- sharing gave main_dense and basis_dense identical kernels (max|delta|=0.0). See decisions.md.
        self.basis_dense = keras.layers.Dense(
            units=self.units,
            use_bias=False,
            kernel_initializer=clone_initializer(self.kernel_initializer),
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
        if self.built:
            return

        # Build sub-layers in computational order for robust serialization
        self.main_dense.build(input_shape)

        # Compute intermediate shapes for proper building
        main_dense_output_shape = self.main_dense.compute_output_shape(input_shape)
        self.relu_k.build(main_dense_output_shape)

        self.basis_function.build(input_shape)
        # basis_function output shape is the same as input shape
        self.basis_dense.build(input_shape)

        logger.debug(f"Built PowerMLP layer with input shape: {input_shape}")

        # Always call parent build at the end
        super().build(input_shape)

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
