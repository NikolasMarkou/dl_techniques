"""
Dense (MLP) block with a residual connection: ``ResidualDenseBlock``.

``ResidualDenseBlock`` wires a ``Dense`` layer to normalization and activation
chosen by name through the `norms`/`activations` factories, rather than
hard-coding a specific normalization or activation class, then adds the
transformed output back to the original input via a skip connection. This
lets a caller swap, say, layer norm for RMSNorm without touching the block
itself. If ``units`` is None, the dense layer matches the input dimension
automatically.
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

@register_dl_technique("dl_techniques.layers.residual_dense_block")
class ResidualDenseBlock(keras.layers.Layer):
    """
    Dense block with residual connection and configurable normalization/activation.

    This layer applies Dense, optional normalization, activation, and optional
    dropout, then adds the result to the original input via a skip connection.
    If ``units`` is None, the dense layer matches the input dimension automatically.

    Architecture:

    .. code-block:: text

        ┌──────────────────────────────┐
        │  Input [B, features]         │
        └──────┬───────────────┬───────┘
               ▼               │ (skip)
        ┌──────────────┐       │
        │  Dense(units)│       │
        ├──────────────┤       │
        │  Opt. Norm   │       │
        ├──────────────┤       │
        │  Activation  │       │
        ├──────────────┤       │
        │  Opt. Dropout│       │
        └──────┬───────┘       │
               ▼               ▼
        ┌──────────────────────────────┐
        │  Add (residual + skip)       │
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │  Output [B, features]        │
        └──────────────────────────────┘

    :param units: Number of dense units. None to match input dimension.
    :type units: int or None
    :param normalization_type: Type of normalization. None to disable.
    :type normalization_type: str or None
    :param activation_type: Type of activation.
    :type activation_type: str
    :param dropout_rate: Dropout rate (0.0 to disable).
    :type dropout_rate: float
    :param kernel_regularizer: Regularizer for dense kernel.
    :type kernel_regularizer: keras.regularizers.Regularizer or None
    :param kernel_initializer: Initializer for dense kernel.
    :type kernel_initializer: str or keras.initializers.Initializer
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
            units: Optional[int] = None,
            normalization_type: Optional[str] = "layer_norm",
            activation_type: str = "relu",
            dropout_rate: float = 0.0,
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
            use_bias: bool = True,
            normalization_kwargs: Optional[Dict[str, Any]] = None,
            activation_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs: Any
    ) -> None:
        """Initialize ResidualDenseBlock with specified parameters."""
        super().__init__(**kwargs)

        # Validate inputs
        if units is not None and units <= 0:
            raise ValueError(f"units must be positive or None, got {units}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be in [0,1], got {dropout_rate}")

        # Store configuration
        self.units = units
        self.normalization_type = normalization_type
        self.activation_type = activation_type
        self.dropout_rate = dropout_rate
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.use_bias = use_bias
        self.normalization_kwargs = normalization_kwargs or {}
        self.activation_kwargs = activation_kwargs or {}

        # Create sub-layers in __init__ (except dense which requires input shape)
        # Dense layer will be created in build() since units may depend on input shape
        self.dense = None
        self.norm = None
        self.activation = None
        self.dropout = None
        self.add = keras.layers.Add(name=f"{self.name}_add")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the dense layer here, since its unit count depends on the
        input shape for the residual connection."""
        if len(input_shape) < 2:
            raise ValueError(f"Input must be at least 2D, got shape {input_shape}")

        # Get the number of input features for residual connection
        input_features = input_shape[-1]
        if input_features is None:
            raise ValueError("Input shape must have a defined last dimension")

        # Determine units: use specified units or match input features
        units = self.units if self.units is not None else input_features

        # Validate that units matches input features for residual connection
        if units != input_features:
            raise ValueError(
                f"For residual connection, units ({units}) must match input features "
                f"({input_features}). Either set units={input_features} or leave units=None."
            )

        # Create dense layer with same units as input features
        self.dense = keras.layers.Dense(
            units=units,
            use_bias=self.use_bias,
            kernel_regularizer=self.kernel_regularizer,
            kernel_initializer=self.kernel_initializer,
            name=f"{self.name}_dense"
        )
        self.dense.build(input_shape)

        # Create and build normalization layer using factory (optional)
        dense_output_shape = self.dense.compute_output_shape(input_shape)
        if self.normalization_type is not None:
            self.norm = create_normalization_layer(
                self.normalization_type,
                name=f"{self.name}_norm",
                **self.normalization_kwargs
            )
            self.norm.build(dense_output_shape)

        # Create and build activation layer using factory
        self.activation = create_activation_layer(
            self.activation_type,
            name=f"{self.name}_activation",
            **self.activation_kwargs
        )
        self.activation.build(dense_output_shape)

        # Create and build dropout layer if requested
        if self.dropout_rate > 0.0:
            self.dropout = keras.layers.Dropout(
                rate=self.dropout_rate, name=f"{self.name}_dropout"
            )
            self.dropout.build(dense_output_shape)

        # Build add layer
        self.add.build([input_shape, dense_output_shape])

        # Parent build must run last.
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the residual dense block."""
        # Forward pass through transformation
        x = self.dense(inputs)

        if self.norm is not None:
            x = self.norm(x, training=training)

        x = self.activation(x)

        if self.dropout is not None:
            x = self.dropout(x, training=training)

        # Add residual connection
        return self.add([inputs, x])

    def compute_output_shape(
            self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape (same as input for residual connection)."""
        return input_shape

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
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'use_bias': self.use_bias,
            'normalization_kwargs': self.normalization_kwargs,
            'activation_kwargs': self.activation_kwargs,
        })
        return config
