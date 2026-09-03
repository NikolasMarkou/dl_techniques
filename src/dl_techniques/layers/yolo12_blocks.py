"""Core building blocks of the YOLOv12 object detection architecture.

Provides ``yolo12_conv_block()`` (a Conv2D + BatchNorm + SiLU unit built on
the shared :class:`~dl_techniques.layers.standard_blocks.ConvBlock`),
``Bottleneck`` (two conv blocks plus an optional residual add),
``C3k2Block`` (a CSP-style block splitting input into a bottleneck-processed
path and a passthrough path, then concatenating), and ``A2C2fBlock`` (an
ELAN-style block stacking pairs of
:class:`~dl_techniques.layers.transformers.area_attention_block.AreaAttentionBlock`
and concatenating every stage's output). ``YOLO12_NORM_KWARGS`` is the single
source of the BatchNorm epsilon/momentum pair every block in this module
uses; every convolution here is bias-free and He-initialised.
"""

import keras
from keras import ops
from typing import Optional, Tuple, Union, Dict, Any

from dl_techniques.utils.keras_registration import register_dl_technique

from . import standard_blocks
from .transformers.area_attention_block import AreaAttentionBlock

# DECISION plan-2026-09-01T055648-e6d380a5/D-005: keep epsilon/momentum in one
# dict, not per-site literals -- 26 construction sites across 4 modules depend
# on it, and create_normalization_layer silently falls back to 1e-6/0.99 if omitted. See decisions.md.
#
# DECISION plan-2026-08-19T163559-499b6f0e/D-067: these values are the
# Ultralytics port (momentum sense is flipped vs PyTorch: 0.03 -> 0.97), not
# unreviewed Keras defaults. See decisions.md.
YOLO12_NORM_KWARGS: Dict[str, Any] = {"epsilon": 1e-3, "momentum": 0.97}


def yolo12_conv_block(
    filters: int,
    kernel_size: int = 3,
    strides: int = 1,
    padding: str = "same",
    groups: int = 1,
    activation: bool = True,
    use_bias: bool = False,
    kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
    kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    **kwargs: Any
) -> standard_blocks.ConvBlock:
    """Build the YOLOv12 Conv-BN-SiLU unit on top of the shared ``ConvBlock``.

    Every convolution in the YOLOv12 tree is bias-free, He-initialised, batch
    normalised with the D-067 pair, and either SiLU-activated or not activated
    at all. This helper is the single place that spells that out, so the
    normalization pair keeps exactly one home (``YOLO12_NORM_KWARGS``).

    :param filters: Number of output channels. Must be positive.
    :type filters: int
    :param kernel_size: Convolution kernel size. Defaults to 3.
    :type kernel_size: int
    :param strides: Convolution stride. Defaults to 1.
    :type strides: int
    :param padding: One of ``'same'`` or ``'valid'``. Defaults to ``'same'``.
    :type padding: str
    :param groups: Number of convolution groups; ``groups`` equal to the input
        channel count gives a depthwise convolution. Defaults to 1.
    :type groups: int
    :param activation: ``True`` for SiLU, ``False`` for no activation at all.
        ``False`` maps to ``activation_type='linear'``, which is a weightless
        exact identity, not an approximation of one.
    :type activation: bool
    :param use_bias: Whether the convolution carries a bias term. Defaults to
        ``False``, because the BatchNorm that follows has its own beta.
    :type use_bias: bool
    :param kernel_initializer: Weight initializer. Defaults to ``'he_normal'``.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param kernel_regularizer: Optional weight regularizer.
    :type kernel_regularizer: str or keras.regularizers.Regularizer or None
    :param kwargs: Additional keyword arguments for the Layer base class, e.g.
        ``name``.
    :type kwargs: Any

    :return: An unbuilt ``standard_blocks.ConvBlock``.
    :rtype: standard_blocks.ConvBlock
    """
    return standard_blocks.ConvBlock(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        groups=groups,
        use_bias=use_bias,
        activation_type="silu" if activation else "linear",
        normalization_type="batch_norm",
        normalization_kwargs=dict(YOLO12_NORM_KWARGS),
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        **kwargs
    )



@register_dl_technique("dl_techniques.layers.yolo12_blocks")
class Bottleneck(keras.layers.Layer):
    """
    Standard bottleneck block with optional residual connection for YOLOv12.

    Two sequential 3x3 ``yolo12_conv_block`` layers with an optional shortcut connection
    that adds the input to the output when channels match.

    Architecture:

    .. code-block:: text

        ┌──────────────────────────────┐
        │  Input [B, H, W, C]          │
        └──────┬───────────────┬───────┘
               ▼               │ (shortcut if C=filters)
        ┌──────────────┐       │
        │ conv_block3x3│       │
        ├──────────────┤       │
        │ conv_block3x3│       │
        └──────┬───────┘       │
               ▼               ▼
        ┌──────────────────────────────┐
        │  Add (if shortcut=True)      │
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │  Output [B, H, W, filters]   │
        └──────────────────────────────┘

    :param filters: Number of output filters. Must be positive.
    :type filters: int
    :param shortcut: Whether to use residual connection. Defaults to True.
    :type shortcut: bool
    :param kernel_initializer: Weight initializer. Defaults to ``'he_normal'``.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param kwargs: Additional keyword arguments for Layer base class.
    :type kwargs: Any
    """

    def __init__(
        self,
        filters: int,
        shortcut: bool = True,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if filters <= 0:
            raise ValueError(f"filters must be positive, got {filters}")

        self.filters = filters
        self.shortcut = shortcut
        self.kernel_initializer = keras.initializers.get(kernel_initializer)

        self.cv1 = yolo12_conv_block(
            filters=self.filters,
            kernel_size=3,
            kernel_initializer=self.kernel_initializer,
            name="cv1"
        )

        self.cv2 = yolo12_conv_block(
            filters=self.filters,
            kernel_size=3,
            activation=False,
            kernel_initializer=self.kernel_initializer,
            name="cv2"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the bottleneck components and all sub-layers.

        :param input_shape: Shape tuple of input tensor.
        :type input_shape: tuple
        """
        # Build sub-layers in computational order for robust serialization
        self.cv1.build(input_shape)

        # Compute intermediate shape for cv2
        cv1_output_shape = self.cv1.compute_output_shape(input_shape)
        self.cv2.build(cv1_output_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through bottleneck.

            :param inputs: Input tensor.
            :type inputs: keras.KerasTensor
            :param training: Boolean, whether in training mode.
            :type training: bool or None

            :return: Output tensor with optional residual connection.
            :rtype: keras.KerasTensor
        """
        x = self.cv1(inputs, training=training)
        x = self.cv2(x, training=training)

        if self.shortcut and ops.shape(inputs)[-1] == self.filters:
            x = inputs + x

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape.

            :param input_shape: Shape tuple of input.
            :type input_shape: tuple

            :return: Output shape tuple.
            :rtype: tuple
        """
        output_shape = list(input_shape)
        output_shape[-1] = self.filters
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

            :return: Dictionary containing the layer configuration.
            :rtype: dict
        """
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "shortcut": self.shortcut,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
        })
        return config



@register_dl_technique("dl_techniques.layers.yolo12_blocks")
class C3k2Block(keras.layers.Layer):
    """
    CSP-like block with dual paths and Bottleneck layers for YOLOv12.

    Splits the input into two paths via 1x1 convolutions. One path is
    processed through ``n`` Bottleneck layers, while the other remains
    unchanged. Both paths are concatenated and fused through a final 1x1
    convolution.

    Architecture:

    .. code-block:: text

        ┌──────────────────────────────────┐
        │  Input [B, H, W, C]              │
        └──────┬───────────────┬───────────┘
               ▼               ▼
        ┌────────────┐  ┌────────────┐
        │  cv1 (1x1) │  │  cv2 (1x1) │
        ├────────────┤  └──────┬─────┘
        │  Bottleneck│         │
        │  x n       │         │
        └──────┬─────┘         │
               ▼               ▼
        ┌──────────────────────────────────┐
        │  Concatenate along channels      │
        └──────────────┬───────────────────┘
                       ▼
        ┌──────────────────────────────────┐
        │  cv3 (1x1) → Output [B,H,W,F]    │
        └──────────────────────────────────┘

    :param filters: Number of output filters. Must be positive.
    :type filters: int
    :param n: Number of bottleneck layers. Must be non-negative. Defaults to 1.
    :type n: int
    :param shortcut: Whether to use shortcuts in bottlenecks. Defaults to True.
    :type shortcut: bool
    :param kernel_initializer: Weight initializer. Defaults to ``'he_normal'``.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param kwargs: Additional keyword arguments for Layer base class.
    :type kwargs: Any
    """

    def __init__(
        self,
        filters: int,
        n: int = 1,
        shortcut: bool = True,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if filters <= 0:
            raise ValueError(f"filters must be positive, got {filters}")
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")

        self.filters = filters
        self.n = n
        self.shortcut = shortcut
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.hidden_filters = filters // 2

        self.cv1 = yolo12_conv_block(
            filters=self.hidden_filters,
            kernel_size=1,
            kernel_initializer=self.kernel_initializer,
            name="cv1"
        )

        self.cv2 = yolo12_conv_block(
            filters=self.hidden_filters,
            kernel_size=1,
            kernel_initializer=self.kernel_initializer,
            name="cv2"
        )

        # Create bottleneck layers - store as list attributes for proper serialization
        self.bottlenecks = []
        for i in range(self.n):
            bottleneck = Bottleneck(
                filters=self.hidden_filters,
                shortcut=self.shortcut,
                kernel_initializer=self.kernel_initializer,
                name=f"bottleneck_{i}"
            )
            self.bottlenecks.append(bottleneck)

        self.cv3 = yolo12_conv_block(
            filters=self.filters,
            kernel_size=1,
            kernel_initializer=self.kernel_initializer,
            name="cv3"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the C3k2 block components and all sub-layers.

        :param input_shape: Shape tuple of input tensor.
        :type input_shape: tuple
        """
        # Build sub-layers in computational order for robust serialization
        self.cv1.build(input_shape)
        self.cv2.build(input_shape)

        # Compute intermediate shape for bottlenecks
        cv1_output_shape = self.cv1.compute_output_shape(input_shape)

        # Build bottleneck layers sequentially
        current_shape = cv1_output_shape
        for bottleneck in self.bottlenecks:
            bottleneck.build(current_shape)
            current_shape = bottleneck.compute_output_shape(current_shape)

        # Compute shape for final convolution (concatenation of two paths)
        cv2_output_shape = self.cv2.compute_output_shape(input_shape)
        concat_shape = list(cv2_output_shape)
        concat_shape[-1] = current_shape[-1] + cv2_output_shape[-1]  # Concatenate channel dimension
        self.cv3.build(tuple(concat_shape))

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through C3k2 block.

            :param inputs: Input tensor.
            :type inputs: keras.KerasTensor
            :param training: Boolean, whether in training mode.
            :type training: bool or None

            :return: Output tensor after CSP processing.
            :rtype: keras.KerasTensor
        """
        y1 = self.cv1(inputs, training=training)
        y2 = self.cv2(inputs, training=training)

        # Apply bottleneck layers sequentially
        for bottleneck in self.bottlenecks:
            y1 = bottleneck(y1, training=training)

        # Concatenate and apply final convolution
        y = ops.concatenate([y1, y2], axis=-1)
        return self.cv3(y, training=training)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape.

            :param input_shape: Shape tuple of input.
            :type input_shape: tuple

            :return: Output shape tuple.
            :rtype: tuple
        """
        output_shape = list(input_shape)
        output_shape[-1] = self.filters
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

            :return: Dictionary containing the layer configuration.
            :rtype: dict
        """
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "n": self.n,
            "shortcut": self.shortcut,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
        })
        return config



@register_dl_technique("dl_techniques.layers.yolo12_blocks")
class A2C2fBlock(keras.layers.Layer):
    """
    Attention-enhanced ELAN block with progressive feature extraction for YOLOv12.

    Processes input through a 1x1 convolution, then through ``n`` pairs of
    :class:`~dl_techniques.layers.transformers.area_attention_block.AreaAttentionBlock`
    layers, progressively concatenating outputs from each stage to build a rich
    feature hierarchy. A final 1x1 convolution fuses all features.

    Architecture:

    .. code-block:: text

        ┌──────────────────────────────────┐
        │  Input [B, H, W, C]              │
        └──────────────┬───────────────────┘
                       ▼
        ┌──────────────────────────────────┐
        │  cv1 (1x1) → y0 [B,H,W,F/2]      │
        └──────────────┬───────────────────┘
                       ▼
        ┌──────────────────────────────────┐
        │  AreaAttentionBlock pair 1 → y1  │
        │  AreaAttentionBlock pair 2 → y2  │
        │  ...                             │
        │  AreaAttentionBlock pair n → yn  │
        └──────────────┬───────────────────┘
                       ▼
        ┌──────────────────────────────────┐
        │  Concat [y0, y1, ..., yn]        │
        │  cv2 (1x1) → Output [B,H,W,F]    │
        └──────────────────────────────────┘

    :param filters: Number of output filters. Must be positive.
    :type filters: int
    :param n: Number of attention block pairs. Must be non-negative. Defaults to 1.
    :type n: int
    :param area: Area size for attention mechanism. Defaults to 1.
    :type area: int
    :param kernel_initializer: Weight initializer. Defaults to ``'he_normal'``.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param kwargs: Additional keyword arguments for Layer base class.
    :type kwargs: Any
    """

    def __init__(
        self,
        filters: int,
        n: int = 1,
        area: int = 1,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if filters <= 0:
            raise ValueError(f"filters must be positive, got {filters}")
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        if area <= 0:
            raise ValueError(f"area must be positive, got {area}")

        self.filters = filters
        self.n = n
        self.area = area
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.hidden_filters = filters // 2

        self.cv1 = yolo12_conv_block(
            filters=self.hidden_filters,
            kernel_size=1,
            kernel_initializer=self.kernel_initializer,
            name="cv1"
        )

        # Create attention blocks as individual list attributes for proper serialization
        # Each pair has two attention blocks: first and second
        self.attention_first_blocks = []
        self.attention_second_blocks = []

        for i in range(self.n):
            attn_block_1 = AreaAttentionBlock(
                dim=self.hidden_filters,
                num_heads=max(1, self.hidden_filters // 32),
                area=self.area,
                normalization_kwargs=dict(YOLO12_NORM_KWARGS),
                kernel_initializer=self.kernel_initializer,
                name=f"attn_{i}_1"
            )
            attn_block_2 = AreaAttentionBlock(
                dim=self.hidden_filters,
                num_heads=max(1, self.hidden_filters // 32),
                area=self.area,
                normalization_kwargs=dict(YOLO12_NORM_KWARGS),
                kernel_initializer=self.kernel_initializer,
                name=f"attn_{i}_2"
            )
            self.attention_first_blocks.append(attn_block_1)
            self.attention_second_blocks.append(attn_block_2)

        self.cv2 = yolo12_conv_block(
            filters=self.filters,
            kernel_size=1,
            kernel_initializer=self.kernel_initializer,
            name="cv2"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the A2C2f block components and all sub-layers.

        :param input_shape: Shape tuple of input tensor.
        :type input_shape: tuple
        """
        # Build sub-layers in computational order for robust serialization
        self.cv1.build(input_shape)

        # Compute intermediate shape for attention blocks
        cv1_output_shape = self.cv1.compute_output_shape(input_shape)

        # Build attention blocks sequentially
        current_shape = cv1_output_shape
        for i in range(self.n):
            # Build first attention block
            self.attention_first_blocks[i].build(current_shape)
            current_shape = self.attention_first_blocks[i].compute_output_shape(current_shape)

            # Build second attention block
            self.attention_second_blocks[i].build(current_shape)
            current_shape = self.attention_second_blocks[i].compute_output_shape(current_shape)

        # Compute shape for final convolution (concatenation of all features)
        # We have n+1 feature tensors (initial + n pairs)
        concat_channels = self.hidden_filters * (self.n + 1)
        concat_shape = list(cv1_output_shape)
        concat_shape[-1] = concat_channels
        self.cv2.build(tuple(concat_shape))

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through A2C2f block.

            :param inputs: Input tensor.
            :type inputs: keras.KerasTensor
            :param training: Boolean, whether in training mode.
            :type training: bool or None

            :return: Output tensor after progressive feature extraction and fusion.
            :rtype: keras.KerasTensor
        """
        y = self.cv1(inputs, training=training)

        # Collect features progressively
        features = [y]

        for i in range(self.n):
            # Apply two attention blocks sequentially
            y = self.attention_first_blocks[i](features[-1], training=training)
            y = self.attention_second_blocks[i](y, training=training)
            features.append(y)

        # Concatenate all features and apply final convolution
        y = ops.concatenate(features, axis=-1)
        return self.cv2(y, training=training)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape.

            :param input_shape: Shape tuple of input.
            :type input_shape: tuple

            :return: Output shape tuple.
            :rtype: tuple
        """
        output_shape = list(input_shape)
        output_shape[-1] = self.filters
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

            :return: Dictionary containing the layer configuration.
            :rtype: dict
        """
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "n": self.n,
            "area": self.area,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
        })
        return config

