"""
Random Fourier Features (RFF) layer, built by the ``RFFKernelLayer`` class.

Kernel machines such as an RBF-kernel SVM need an ``N x N`` kernel matrix,
which is quadratic in the number of data points and does not scale. This
layer instead projects each input through a fixed random matrix and cosine
nonlinearity, producing an explicit low-dimensional feature map whose inner
product approximates the RBF kernel (Bochner's theorem). A standard
trainable dense layer then operates on that fixed feature map, turning a
non-linear kernel problem into a linear one. The random projection weights
are sampled once at build time and never trained.

References:
    - Rahimi & Recht, 2007. Random Features for Large-Scale Kernel Machines.
"""

import keras
import numpy as np
from typing import Optional, Union, Tuple, Any, Dict, Callable

from dl_techniques.utils.keras_registration import register_dl_technique


@register_dl_technique("dl_techniques.layers.structured_linear.random_fourier_features")
class RFFKernelLayer(keras.layers.Layer):
    """Random Fourier Features layer for efficient kernel approximation.

    This layer approximates translation-invariant kernels (e.g. Gaussian RBF)
    using a fixed random projection followed by a learnable linear layer.
    Random frequencies ``omega ~ N(0, gamma^2 I)`` and phase shifts
    ``b ~ Uniform(0, 2*pi)`` are sampled once and frozen; the explicit
    feature map ``phi(x) = sqrt(2/D) * cos(x @ omega + b)`` is then
    passed through a trainable dense projection. By Bochner's theorem the
    inner product ``phi(x)^T phi(y)`` approximates the kernel
    ``k(x,y) = exp(-gamma^2 ||x-y||^2)`` with ``O(D)`` memory instead of
    the ``O(N^2)`` required by a full kernel matrix.

    Architecture:

    .. code-block:: text

        ┌──────────────────────────────────┐
        │  Input [..., input_dim]          │
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  Random projection (frozen)      │
        │  z = x @ omega + b               │
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  Feature map                     │
        │  phi(x) = sqrt(2/D) * cos(z)     │
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  Dense(output_dim) (trainable)   │
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  Optional activation             │
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  Output [..., output_dim]        │
        └──────────────────────────────────┘

    :param input_dim: Dimensionality of input features. Must be positive.
    :type input_dim: int
    :param output_dim: Dimensionality of the output. Defaults to
        ``input_dim`` when ``None``.
    :type output_dim: Optional[int]
    :param n_features: Number of random Fourier features (``D``).
    :type n_features: int
    :param gamma: RBF kernel bandwidth parameter.
    :type gamma: float
    :param use_bias: Whether the output dense layer uses a bias.
    :type use_bias: bool
    :param activation: Optional activation after the linear transform.
    :type activation: Optional[Union[str, callable]]
    :param kernel_initializer: Initializer for output weight matrix.
    :type kernel_initializer: Union[str, initializers.Initializer]
    :param bias_initializer: Initializer for output bias vector.
    :type bias_initializer: Union[str, initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for output weights.
    :type kernel_regularizer: Optional[regularizers.Regularizer]
    :param bias_regularizer: Optional regularizer for output bias.
    :type bias_regularizer: Optional[regularizers.Regularizer]
    :param kernel_constraint: Optional constraint for output weights.
    :type kernel_constraint: Optional[constraints.Constraint]
    :param bias_constraint: Optional constraint for output bias.
    :type bias_constraint: Optional[constraints.Constraint]
    :param kwargs: Additional keyword arguments for the Layer base class.
    :type kwargs: Any"""

    def __init__(
            self,
            input_dim: int,
            output_dim: Optional[int] = None,
            n_features: int = 1000,
            gamma: float = 1.0,
            use_bias: bool = True,
            activation: Optional[Union[str, Callable]] = None,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
            kernel_constraint: Optional[keras.constraints.Constraint] = None,
            bias_constraint: Optional[keras.constraints.Constraint] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if n_features <= 0:
            raise ValueError(f"n_features must be positive, got {n_features}")
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}")

        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.n_features = n_features
        self.gamma = gamma
        self.use_bias = use_bias
        self.activation = keras.activations.get(activation)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)

        if self.output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {self.output_dim}")

        self.linear = keras.layers.Dense(
            units=self.output_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            name='output_projection'
        )

        # Created in build(), once input_shape is known.
        self.omega = None
        self.b = None
        self._scale_factor = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create random features and build the output projection.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]"""
        if input_shape[-1] != self.input_dim:
            raise ValueError(
                f"Last dimension of input ({input_shape[-1]}) must match "
                f"input_dim ({self.input_dim})"
            )

        omega_init = keras.random.normal(
            shape=(self.input_dim, self.n_features),
            mean=0.0,
            stddev=self.gamma,
            seed=None
        )
        self.omega = self.add_weight(
            name='omega',
            shape=(self.input_dim, self.n_features),
            initializer=lambda shape, dtype: omega_init,
            trainable=False,
            dtype=self.compute_dtype
        )

        b_init = keras.random.uniform(
            shape=(self.n_features,),
            minval=0.0,
            maxval=2.0 * np.pi,
            seed=None
        )
        self.b = self.add_weight(
            name='b',
            shape=(self.n_features,),
            initializer=lambda shape, dtype: b_init,
            trainable=False,
            dtype=self.compute_dtype
        )

        self._scale_factor = keras.ops.sqrt(2.0 / self.n_features)

        # Keras may pass input_shape as a list on deserialization.
        feature_shape = tuple(input_shape[:-1]) + (self.n_features,)
        self.linear.build(feature_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply the Random Fourier Features transformation.

        :param inputs: Input tensor ``(batch, ..., input_dim)``.
        :type inputs: keras.KerasTensor
        :param training: Training mode flag (unused).
        :type training: Optional[bool]
        :return: Output tensor ``(batch, ..., output_dim)``.
        :rtype: keras.KerasTensor"""
        projection = keras.ops.matmul(inputs, self.omega) + self.b
        features = self._scale_factor * keras.ops.cos(projection)
        output = self.linear(features, training=training)

        if self.activation is not None:
            output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        :param input_shape: Shape tuple of the input.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]"""
        output_shape = list(input_shape)
        output_shape[-1] = self.output_dim
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration for serialization.

        :return: Dictionary containing all constructor parameters.
        :rtype: Dict[str, Any]"""
        config = super().get_config()
        config.update({
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'n_features': self.n_features,
            'gamma': self.gamma,
            'use_bias': self.use_bias,
            'activation': keras.activations.serialize(self.activation),
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': keras.constraints.serialize(self.bias_constraint),
        })
        return config
