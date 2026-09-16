"""
Bottleneck ResNet residual block, built by the ``BottleneckBlock`` class.

Wires a 1x1/3x3/1x1 ``Conv2D`` stack (dimensionality reduction, spatial
processing, dimensionality expansion) to normalization and activation chosen
by name through the ``norms``/``activations`` factories, rather than
hard-coding a specific normalization or activation class, then adds the
result to a shortcut connection (optionally projected via 1x1 convolution).
This is the ResNet-50/101/152 residual unit.
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

@register_dl_technique("dl_techniques.layers.conv_blocks.bottleneck_block")
class BottleneckBlock(keras.layers.Layer):
    """
    Bottleneck ResNet block with 1x1, 3x3, 1x1 convolutions.

    Used in ResNet-50, ResNet-101, and ResNet-152. The block reduces dimensions
    with a 1x1 conv, processes with a 3x3 conv, and expands back with a 1x1 conv
    (output channels = ``filters * 4``), then adds to a shortcut connection.

    Architecture:

    .. code-block:: text

        ┌──────────────────────────────────────┐
        │  Input [B, H, W, C_in]               │
        └──────┬───────────────────────┬───────┘
               ▼                       │ (shortcut)
        ┌──────────────────┐           │
        │  Conv1x1(filters)│           │ opt. Conv1x1
        │  Norm → Act      │           │ (filters*4)
        ├──────────────────┤           │ + Norm
        │  Conv3x3(filters)│           │
        │  Norm → Act      │           │
        ├──────────────────┤           │
        │  Conv1x1(filt*4) │           │
        │  Norm            │           │
        └──────┬───────────┘           │
               ▼                       ▼
        ┌──────────────────────────────────────┐
        │  Add → Activation                    │
        └──────────────┬───────────────────────┘
                       ▼
        ┌──────────────────────────────────────┐
        │  Output [B, H', W', filters*4]       │
        └──────────────────────────────────────┘

    :param filters: Number of filters in the bottleneck (output = filters * 4).
    :type filters: int
    :param stride: Stride for the 3x3 convolution. Defaults to 1.
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
        """Initialize BottleneckBlock with specified parameters."""
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
        # DECISION plan_2026-05-18_6776f8ba/D-003 (parallel to BasicBlock above).
        # [plan-2026-09-14T165315-47f9d575/D-006] locator: BasicBlock now lives
        # in basic_block.py, not "above" in this file (relocated from the
        # shared standard_blocks.py). Appended, not reworded.
        self.normalization_kwargs = dict(normalization_kwargs) if normalization_kwargs else {}
        self.activation_type = activation_type
        self.expansion = 4  # Bottleneck expansion factor

        # Create sub-layers in __init__
        # First 1x1 convolution (dimensionality reduction)
        self.conv1 = keras.layers.Conv2D(
            filters=filters,
            kernel_size=1,
            strides=1,
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

        # Second 3x3 convolution (bottleneck)
        self.conv2 = keras.layers.Conv2D(
            filters=filters,
            kernel_size=3,
            strides=stride,
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
        self.act2 = create_activation_layer(
            activation_type,
            name=f"{self.name}_act2"
        )

        # Third 1x1 convolution (dimensionality expansion)
        self.conv3 = keras.layers.Conv2D(
            filters=filters * self.expansion,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=kernel_regularizer,
            name=f"{self.name}_conv3"
        )
        self.bn3 = create_normalization_layer(
            normalization_type,
            name=f"{self.name}_bn3",
            **self.normalization_kwargs,
        )

        # Shortcut projection if needed
        if use_projection:
            self.shortcut_conv = keras.layers.Conv2D(
                filters=filters * self.expansion,
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
        # Build main path - first conv
        self.conv1.build(input_shape)
        conv1_output_shape = self.conv1.compute_output_shape(input_shape)

        self.bn1.build(conv1_output_shape)
        self.act1.build(conv1_output_shape)

        # Build main path - second conv
        self.conv2.build(conv1_output_shape)
        conv2_output_shape = self.conv2.compute_output_shape(conv1_output_shape)

        self.bn2.build(conv2_output_shape)
        self.act2.build(conv2_output_shape)

        # Build main path - third conv
        self.conv3.build(conv2_output_shape)
        conv3_output_shape = self.conv3.compute_output_shape(conv2_output_shape)

        self.bn3.build(conv3_output_shape)

        # Build shortcut path
        if self.use_projection:
            self.shortcut_conv.build(input_shape)
            shortcut_output_shape = self.shortcut_conv.compute_output_shape(input_shape)
            self.shortcut_bn.build(shortcut_output_shape)

        # Build add layer
        shortcut_shape = shortcut_output_shape if self.use_projection else input_shape
        self.add.build([conv3_output_shape, shortcut_shape])

        # Build final activation
        add_output_shape = conv3_output_shape  # Add preserves shape
        self.act_final.build(add_output_shape)

        # Parent build must run last.
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass of the bottleneck block."""
        # Main path
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)

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
        shape = self.conv1.compute_output_shape(input_shape)
        shape = self.conv2.compute_output_shape(shape)
        shape = self.conv3.compute_output_shape(shape)
        return shape

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
