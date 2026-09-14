"""
Configurable convolutional building block, built by the ``ConvBlock`` class.

Wires a ``Conv2D`` to normalization and activation chosen by name through the
``norms``/``activations`` factories, rather than hard-coding a specific
normalization or activation class, so a caller can swap in e.g. RMSNorm for
batch norm without touching the block itself. Optional dropout and pooling
stages round out the pipeline. This is a heavily-reused shared asset across
`attention/`, `transformers/`, ResNet, FractalNet, and YOLOv12.
"""

import keras
from typing import Optional, Union, Tuple, Dict, Any, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..norms import create_normalization_layer
from ..activations import resolve_activation_layer
from ..activations.factory import ACTIVATION_REGISTRY
from dl_techniques.utils.keras_registration import register_dl_technique


# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.conv_blocks.conv_block")
class ConvBlock(keras.layers.Layer):
    """
    Configurable convolutional block with normalization, activation, and optional pooling.

    This layer implements a flexible convolutional building block using factory-based
    selection of normalization and activation layers. The processing pipeline is
    Conv2D followed by normalization, activation, optional dropout, and optional pooling.

    Architecture:

    .. code-block:: text

        ┌───────────────────────────────────────┐
        │  Input [B, H, W, C]                   │
        └──────────────┬────────────────────────┘
                       ▼
        ┌───────────────────────────────────────┐
        │  Conv2D(filters, kernel, strides)     │
        └──────────────┬────────────────────────┘
                       ▼
        ┌───────────────────────────────────────┐
        │  Normalization (configurable)         │
        └──────────────┬────────────────────────┘
                       ▼
        ┌───────────────────────────────────────┐
        │  Activation (configurable)            │
        └──────────────┬────────────────────────┘
                       ▼
        ┌───────────────────────────────────────┐
        │  Optional Dropout                     │
        └──────────────┬────────────────────────┘
                       ▼
        ┌───────────────────────────────────────┐
        │  Optional Pooling (max / avg)         │
        └──────────────┬────────────────────────┘
                       ▼
        ┌───────────────────────────────────────┐
        │  Output [B, H', W', filters]          │
        └───────────────────────────────────────┘

    :param filters: Number of convolutional filters.
    :type filters: int
    :param kernel_size: Size of convolutional kernel.
    :type kernel_size: int or tuple[int, int]
    :param strides: Convolution strides.
    :type strides: int or tuple[int, int]
    :param padding: Padding mode (``'same'`` or ``'valid'``).
    :type padding: str
    :param normalization_type: Type of normalization.
    :type normalization_type: str
    :param activation_type: Type of activation. Either an ``ACTIVATION_REGISTRY``
        key or any plain Keras activation name; ``'linear'`` builds a weightless
        exact identity, i.e. no activation.
    :type activation_type: str
    :param dropout_rate: Dropout rate (0.0 to disable).
    :type dropout_rate: float
    :param use_pooling: Whether to apply pooling layer.
    :type use_pooling: bool
    :param pool_size: Size of pooling window.
    :type pool_size: int or tuple[int, int]
    :param pool_type: Type of pooling (``'max'`` or ``'avg'``).
    :type pool_type: str
    :param kernel_regularizer: Regularizer for convolution kernel.
    :type kernel_regularizer: keras.regularizers.Regularizer or None
    :param kernel_initializer: Initializer for convolution kernel.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param normalization_kwargs: Additional arguments for normalization layer.
    :type normalization_kwargs: dict or None
    :param activation_kwargs: Additional arguments for activation layer. Only a
        registry activation accepts them; supplying them for a plain Keras
        activation name raises ``ValueError`` rather than dropping them.
    :type activation_kwargs: dict or None
    :param groups: Number of convolution groups. ``groups=1`` is a dense
        convolution; ``groups`` equal to the input channel count is depthwise.
        Must divide both the input channel count and ``filters``.
    :type groups: int
    :param use_bias: Whether the convolution carries a bias term.
    :type use_bias: bool
    :param kwargs: Additional arguments for Layer base class.
    :type kwargs: Any
    """

    def __init__(
            self,
            filters: int,
            kernel_size: Union[int, Tuple[int, int]] = 3,
            strides: Union[int, Tuple[int, int]] = 1,
            padding: str = "same",
            normalization_type: str = "batch_norm",
            activation_type: str = "relu",
            dropout_rate: float = 0.0,
            use_pooling: bool = False,
            pool_size: Union[int, Tuple[int, int]] = 2,
            pool_type: Literal["max", "avg"] = "max",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
            normalization_kwargs: Optional[Dict[str, Any]] = None,
            activation_kwargs: Optional[Dict[str, Any]] = None,
            groups: int = 1,
            use_bias: bool = True,
            **kwargs: Any
    ) -> None:
        """Initialize ConvBlock with specified parameters."""
        super().__init__(**kwargs)

        # Validate inputs
        if filters <= 0:
            raise ValueError(f"filters must be positive, got {filters}")
        if pool_type not in ["max", "avg"]:
            raise ValueError(f"pool_type must be 'max' or 'avg', got {pool_type}")
        if padding not in ["same", "valid"]:
            raise ValueError(f"padding must be 'same' or 'valid', got {padding}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be in [0,1], got {dropout_rate}")
        if groups <= 0:
            raise ValueError(f"groups must be positive, got {groups}")

        # DECISION plan-2026-09-01T055648-e6d380a5/D-001: raise here, not a
        # warning -- resolve_activation_layer's Keras fallback path silently
        # drops kwargs, so a typo would otherwise be a silent no-op. See decisions.md.
        if activation_kwargs and activation_type not in ACTIVATION_REGISTRY:
            raise ValueError(
                f"activation_kwargs={activation_kwargs} was given for "
                f"activation_type='{activation_type}', which is not an "
                f"ACTIVATION_REGISTRY key. Keras activation names take no "
                f"keyword arguments and the kwargs would be silently dropped. "
                f"Use one of {sorted(ACTIVATION_REGISTRY)} or pass no "
                f"activation_kwargs."
            )

        # Store configuration
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.normalization_type = normalization_type
        self.activation_type = activation_type
        self.dropout_rate = dropout_rate
        self.use_pooling = use_pooling
        self.pool_size = pool_size
        self.pool_type = pool_type
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.normalization_kwargs = normalization_kwargs or {}
        self.activation_kwargs = activation_kwargs or {}
        self.groups = groups
        self.use_bias = use_bias

        # Create sub-layers in __init__
        self.conv = keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            groups=groups,
            use_bias=use_bias,
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=kernel_initializer,
            name=f"{self.name}_conv"
        )

        # Create normalization layer using factory
        self.norm = create_normalization_layer(
            normalization_type,
            name=f"{self.name}_norm",
            **self.normalization_kwargs
        )

        # Create activation layer using factory. `resolve_activation_layer` keeps
        # every registry key on its existing path and additionally accepts plain
        # Keras names, so `activation_type='linear'` expresses "no activation" as
        # a weightless exact identity.
        self.activation = resolve_activation_layer(
            activation_type,
            name=f"{self.name}_activation",
            **self.activation_kwargs
        )

        # Create dropout layer if requested
        if dropout_rate > 0.0:
            self.dropout = keras.layers.Dropout(
                rate=dropout_rate, name=f"{self.name}_dropout"
            )
        else:
            self.dropout = None

        # Create pooling layer if requested
        if use_pooling:
            if pool_type == "max":
                self.pool = keras.layers.MaxPooling2D(
                    pool_size=pool_size, name=f"{self.name}_pool"
                )
            elif pool_type == "avg":
                self.pool = keras.layers.AveragePooling2D(
                    pool_size=pool_size, name=f"{self.name}_pool"
                )
            else:
                raise ValueError("Not valid pooling type [{}]".format(pool_type))
        else:
            self.pool = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build each sub-layer explicitly, so its weights exist before
        restoration during model loading."""
        # Build sub-layers in computational order
        self.conv.build(input_shape)

        conv_output_shape = self.conv.compute_output_shape(input_shape)
        self.norm.build(conv_output_shape)
        self.activation.build(conv_output_shape)

        if self.dropout is not None:
            self.dropout.build(conv_output_shape)

        if self.pool is not None:
            self.pool.build(conv_output_shape)

        # Parent build must run last.
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the convolutional block."""
        x = self.conv(inputs)
        x = self.norm(x, training=training)
        x = self.activation(x)

        if self.dropout is not None:
            x = self.dropout(x, training=training)

        if self.pool is not None:
            x = self.pool(x)

        return x

    def compute_output_shape(
            self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        shape = self.conv.compute_output_shape(input_shape)
        if self.pool is not None:
            shape = self.pool.compute_output_shape(shape)
        return shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'normalization_type': self.normalization_type,
            'activation_type': self.activation_type,
            'dropout_rate': self.dropout_rate,
            'use_pooling': self.use_pooling,
            'pool_size': self.pool_size,
            'pool_type': self.pool_type,
            'kernel_regularizer': keras.regularizers.serialize(
                self.kernel_regularizer) if self.kernel_regularizer else None,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'normalization_kwargs': self.normalization_kwargs,
            'activation_kwargs': self.activation_kwargs,
            'groups': self.groups,
            'use_bias': self.use_bias,
        })
        return config
