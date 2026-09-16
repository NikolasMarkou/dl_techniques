"""
Basic ResNet residual block, built by the ``BasicBlock`` class.

Wires two sequential 3x3 ``Conv2D`` layers to normalization and activation
chosen by name through the ``norms``/``activations`` factories, rather than
hard-coding a specific normalization or activation class, then adds the
result to a shortcut connection (optionally projected via 1x1 convolution).
This is the ResNet-18/34 residual unit.
"""

import keras
from typing import Optional, Dict, Any, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..norms import create_normalization_layer
from ..activations import create_activation_layer
from dl_techniques.utils.keras_registration import register_dl_technique


# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.conv_blocks.basic_block")
class BasicBlock(keras.layers.Layer):
    """
    Basic ResNet block with two 3x3 convolutions.

    Used in ResNet-18 and ResNet-34. The block applies two sequential 3x3
    convolutions with normalization and activation, then adds the result to a
    shortcut connection (optionally projected via 1x1 convolution).

    Architecture:

    .. code-block:: text

        ┌──────────────────────────────────────┐
        │  Input [B, H, W, C]                  │
        └──────┬───────────────────────┬───────┘
               ▼                       │ (shortcut)
        ┌──────────────────┐           │
        │  Conv2D 3x3      │           │ opt. Conv1x1
        │  Norm → Act      │           │ + Norm
        ├──────────────────┤           │
        │  Conv2D 3x3      │           │
        │  Norm            │           │
        └──────┬───────────┘           │
               ▼                       ▼
        ┌──────────────────────────────────────┐
        │  Add → Activation                    │
        └──────────────┬───────────────────────┘
                       ▼
        ┌──────────────────────────────────────┐
        │  Output [B, H', W', filters]         │
        └──────────────────────────────────────┘

    :param filters: Number of output filters.
    :type filters: int
    :param stride: Stride for the first convolution. Defaults to 1.
    :type stride: int
    :param use_projection: Whether to use a 1x1 projection for the shortcut.
    :type use_projection: bool
    :param kernel_regularizer: Regularizer for convolution kernels.
    :type kernel_regularizer: keras.regularizers.Regularizer or None
    :param normalization_type: Type of normalization layer. Defaults to ``'batch_norm'``.
    :type normalization_type: str
    :param activation_type: Type of activation function. Defaults to ``'relu'``.
    :type activation_type: str
    :param kwargs: Additional keyword arguments for Layer.
    :type kwargs: Any
    """

    def __init__(
            self,
            filters: int,
            stride: int = 1,
            use_projection: bool = False,
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            normalization_type: str = "batch_norm",
            normalization_kwargs: Optional[Dict[str, Any]] = None,
            activation_type: str = "relu",
            **kwargs: Any
    ) -> None:
        """Initialize BasicBlock with specified parameters."""
        super().__init__(**kwargs)

        # Validate inputs
        if filters <= 0:
            raise ValueError(f"filters must be positive, got {filters}")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")

        # Store configuration
        self.filters = filters
        self.stride = stride
        self.use_projection = use_projection
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.normalization_type = normalization_type
        # DECISION plan_2026-05-18_6776f8ba/D-003: default None -> {} keeps
        # this byte-identical to the pre-plumbing factory call. See decisions.md.
        self.normalization_kwargs = dict(normalization_kwargs) if normalization_kwargs else {}
        self.activation_type = activation_type

        # Create sub-layers in __init__
        # First convolution
        self.conv1 = keras.layers.Conv2D(
            filters=filters,
            kernel_size=3,
            strides=stride,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=kernel_regularizer,
            name=f"{self.name}_conv1"
        )
        self.bn1 = create_normalization_layer(
            normalization_type,
            name=f"{self.name}_bn1",
            **self.normalization_kwargs,
        )
        self.act1 = create_activation_layer(
            activation_type,
            name=f"{self.name}_act1"
        )

        # Second convolution
        self.conv2 = keras.layers.Conv2D(
            filters=filters,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=kernel_regularizer,
            name=f"{self.name}_conv2"
        )
        self.bn2 = create_normalization_layer(
            normalization_type,
            name=f"{self.name}_bn2",
            **self.normalization_kwargs,
        )

        # Shortcut projection if needed
        if use_projection:
            self.shortcut_conv = keras.layers.Conv2D(
                filters=filters,
                kernel_size=1,
                strides=stride,
                padding="same",
                use_bias=False,
                kernel_initializer="he_normal",
                kernel_regularizer=kernel_regularizer,
                name=f"{self.name}_shortcut_conv"
            )
            self.shortcut_bn = create_normalization_layer(
                normalization_type,
                name=f"{self.name}_shortcut_bn",
                **self.normalization_kwargs,
            )
        else:
            self.shortcut_conv = None
            self.shortcut_bn = None

        self.add = keras.layers.Add(name=f"{self.name}_add")
        self.act_final = create_activation_layer(
            activation_type,
            name=f"{self.name}_act_final"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build each sub-layer explicitly, so its weights exist before
        restoration during model loading."""
        # Build main path
        self.conv1.build(input_shape)
        conv1_output_shape = self.conv1.compute_output_shape(input_shape)

        self.bn1.build(conv1_output_shape)
        self.act1.build(conv1_output_shape)

        self.conv2.build(conv1_output_shape)
        conv2_output_shape = self.conv2.compute_output_shape(conv1_output_shape)

        self.bn2.build(conv2_output_shape)

        # Build shortcut path
        if self.use_projection:
            self.shortcut_conv.build(input_shape)
            shortcut_output_shape = self.shortcut_conv.compute_output_shape(input_shape)
            self.shortcut_bn.build(shortcut_output_shape)

        # Build add layer
        shortcut_shape = shortcut_output_shape if self.use_projection else input_shape
        self.add.build([conv2_output_shape, shortcut_shape])

        # Build final activation
        add_output_shape = conv2_output_shape  # Add preserves shape
        self.act_final.build(add_output_shape)

        # Parent build must run last.
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass of the basic block."""
        # Main path
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        # Shortcut path
        if self.use_projection:
            shortcut = self.shortcut_conv(inputs)
            shortcut = self.shortcut_bn(shortcut, training=training)
        else:
            shortcut = inputs

        # Add and activate
        x = self.add([x, shortcut])
        x = self.act_final(x)

        return x

    def compute_output_shape(
            self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        return self.conv2.compute_output_shape(
            self.conv1.compute_output_shape(input_shape)
        )

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "stride": self.stride,
            "use_projection": self.use_projection,
            "kernel_regularizer": keras.regularizers.serialize(
                self.kernel_regularizer) if self.kernel_regularizer else None,
            "normalization_type": self.normalization_type,
            "normalization_kwargs": dict(self.normalization_kwargs),
            "activation_type": self.activation_type,
        })
        return config

# ---------------------------------------------------------------------
