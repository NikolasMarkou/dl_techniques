"""
Configurable dense (MLP) building block: ``DenseBlock``.

``DenseBlock`` wires a ``Dense`` layer to normalization and activation chosen
by name through the `norms`/`activations` factories, rather than hard-coding
a specific normalization or activation class. This lets a caller swap, say,
layer norm for RMSNorm without touching the block itself. The pipeline is
Dense, optional normalization, activation, and optional dropout.
"""

import keras
from typing import Optional, Union, Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .norms import create_normalization_layer
from .activations import create_activation_layer
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.dense_block")
class DenseBlock(keras.layers.Layer):
    """
    Configurable dense block with normalization, activation, and optional dropout.

    This layer implements a flexible dense building block using factory-based
    selection of normalization and activation layers. The pipeline is
    Dense, optional normalization, activation, and optional dropout.

    Architecture:

    .. code-block:: text

        ┌────────────────────────────────┐
        │  Input [B, features]           │
        └──────────────┬─────────────────┘
                       ▼
        ┌────────────────────────────────┐
        │  Dense(units)                  │
        └──────────────┬─────────────────┘
                       ▼
        ┌────────────────────────────────┐
        │  Optional Normalization        │
        └──────────────┬─────────────────┘
                       ▼
        ┌────────────────────────────────┐
        │  Activation (configurable)     │
        └──────────────┬─────────────────┘
                       ▼
        ┌────────────────────────────────┐
        │  Optional Dropout              │
        └──────────────┬─────────────────┘
                       ▼
        ┌────────────────────────────────┐
        │  Output [B, units]             │
        └────────────────────────────────┘

    :param units: Number of dense units.
    :type units: int
    :param normalization_type: Type of normalization. None to disable.
    :type normalization_type: str or None
    :param activation_type: Type of activation.
    :type activation_type: str
    :param dropout_rate: Dropout rate (0.0 to disable).
    :type dropout_rate: float
    :param kernel_regularizer: Regularizer for dense kernel.
    :type kernel_regularizer: keras.regularizers.Regularizer or None
    :param bias_regularizer: Regularizer for dense bias.
    :type bias_regularizer: keras.regularizers.Regularizer or None
    :param activity_regularizer: Regularizer for dense layer activity.
    :type activity_regularizer: keras.regularizers.Regularizer or None
    :param kernel_initializer: Initializer for dense kernel.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param bias_initializer: Initializer for dense bias.
    :type bias_initializer: str or keras.initializers.Initializer
    :param use_bias: Whether to use bias in dense layer.
    :type use_bias: bool
    :param normalization_kwargs: Additional arguments for normalization layer.
    :type normalization_kwargs: dict or None
    :param activation_kwargs: Additional arguments for activation layer.
    :type activation_kwargs: dict or None
    :param kwargs: Additional arguments for Layer base class.
    :type kwargs: Any
    """

    def __init__(
            self,
            units: int,
            normalization_type: Optional[str] = "layer_norm",
            activation_type: str = "relu",
            dropout_rate: float = 0.0,
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
            activity_regularizer: Optional[keras.regularizers.Regularizer] = None,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
            bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
            kernel_constraint: Optional[keras.constraints.Constraint] = None,
            bias_constraint: Optional[keras.constraints.Constraint] = None,
            use_bias: bool = True,
            normalization_kwargs: Optional[Dict[str, Any]] = None,
            activation_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs: Any
    ) -> None:
        """Initialize DenseBlock with specified parameters."""
        super().__init__(**kwargs)

        # Validate inputs
        if units <= 0:
            raise ValueError(f"units must be positive, got {units}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be in [0,1], got {dropout_rate}")

        # Store configuration
        self.units = units
        self.normalization_type = normalization_type
        self.activation_type = activation_type
        self.dropout_rate = dropout_rate
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = keras.regularizers.get(activity_regularizer)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        self.use_bias = use_bias
        self.normalization_kwargs = normalization_kwargs or {}
        self.activation_kwargs = activation_kwargs or {}

        # Create sub-layers in __init__
        self.dense = keras.layers.Dense(
            units=units,
            use_bias=use_bias,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            name=f"{self.name}_dense"
        )

        # Create normalization layer using factory (optional)
        if normalization_type is not None:
            self.norm = create_normalization_layer(
                normalization_type,
                name=f"{self.name}_norm",
                **self.normalization_kwargs
            )
        else:
            self.norm = None

        # Create activation layer using factory
        self.activation = create_activation_layer(
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

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build each sub-layer explicitly, so its weights exist before
        restoration during model loading."""
        # Build sub-layers in computational order
        self.dense.build(input_shape)

        dense_output_shape = self.dense.compute_output_shape(input_shape)

        if self.norm is not None:
            self.norm.build(dense_output_shape)

        self.activation.build(dense_output_shape)

        if self.dropout is not None:
            self.dropout.build(dense_output_shape)

        # Parent build must run last.
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the dense block."""
        x = self.dense(inputs)

        if self.norm is not None:
            x = self.norm(x, training=training)

        x = self.activation(x)

        if self.dropout is not None:
            x = self.dropout(x, training=training)

        return x

    def compute_output_shape(
            self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        return self.dense.compute_output_shape(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'units': self.units,
            'normalization_type': self.normalization_type,
            'activation_type': self.activation_type,
            'dropout_rate': self.dropout_rate,
            'kernel_regularizer': keras.regularizers.serialize(
                self.kernel_regularizer) if self.kernel_regularizer else None,
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer) if self.bias_regularizer else None,
            'activity_regularizer': keras.regularizers.serialize(
                self.activity_regularizer) if self.activity_regularizer else None,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_constraint': keras.constraints.serialize(
                self.kernel_constraint) if self.kernel_constraint else None,
            'bias_constraint': keras.constraints.serialize(self.bias_constraint) if self.bias_constraint else None,
            'use_bias': self.use_bias,
            'normalization_kwargs': self.normalization_kwargs,
            'activation_kwargs': self.activation_kwargs,
        })
        return config
