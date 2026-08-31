"""Vision task heads, and the factory that picks one.

This module holds eight layer classes and six module-level helpers. The
classes turn a backbone feature map into task predictions. The helpers pick a
class and configure it.

The heads take a feature map, not an image. Nothing here knows which backbone
produced it, so the same head works on a ResNet, a ViT or a ConvNeXt stage.

Classes
-------
* :class:`BaseVisionHead` -- shared construction. Six heads inherit from it.
  It has no ``call``.
* :class:`DetectionHead` -- class scores and box offsets, per anchor.
* :class:`SegmentationHead` -- one class map, upsampled towards input size.
  The only head here that returns a bare tensor.
* :class:`DepthEstimationHead` -- one depth map, plus the tensor it was
  scaled from.
* :class:`ClassificationHead` -- one label for the whole image.
* :class:`InstanceSegmentationHead` -- detection outputs plus instance masks.
* :class:`EnhancementHead` -- a restored or upscaled image.
* :class:`MultiTaskHead` -- several heads behind one layer. It does not
  inherit from :class:`BaseVisionHead`.

Helpers
-------
* :func:`create_vision_head` -- task type to head. It handles 10 of the 37
  ``VisionTaskType`` members and raises ``ValueError`` for the other 27.
* :func:`create_enhancement_head` -- build an :class:`EnhancementHead`, with
  a default scale factor for super-resolution.
* :func:`create_multi_task_head` -- build a :class:`MultiTaskHead` from any
  of three configuration formats.
* :class:`HeadConfiguration` -- three keyword-argument presets per task type:
  default, efficient and high-performance.

The task types and the configuration objects live in
``vision/task_types.py``. The ``VisionTaskType`` docstring there lists the 10
dispatched members and names all 27 that raise. Read it rather than deriving
a second split here.

Example
-------
>>> from dl_techniques.layers.heads.vision import create_vision_head
>>> head = create_vision_head('classification', num_classes=10)
>>> out = head(features)
>>> sorted(out)
['logits', 'probabilities']
"""

import keras
from keras import layers, ops
from typing import Dict, List, Optional, Union, Tuple, Any, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ...activations import ActivationType
from ...standard_blocks import ConvBlock, DenseBlock
from ...ffn.factory import create_ffn_layer, FFNType
from ...attention import AttentionType
from ...attention.factory import (
    create_attention_layer,
    assemble_attention_config,
)
from ...norms import create_normalization_layer, NormalizationType
from .task_types import VisionTaskType, TaskConfiguration
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------
# Base Head Class
# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.heads.vision.factory")
class BaseVisionHead(keras.layers.Layer):
    """
    Shared construction for the vision task heads.

    This class builds the layers the heads have in common and stores the
    configuration they read. It has no ``call``. Each subclass applies the
    stages it wants, then runs its own task layers.

    ``__init__`` calls ``_create_common_layers``, which always builds
    ``norm`` and builds ``attention``, ``ffn`` and ``dropout`` only when the
    matching flag or rate asks for them.

    **Architecture Overview:**

    .. code-block:: text

        _create_common_layers builds these four:

        ┌─────────────────────────────────┐
        │ norm                     always │
        │ attention      if use_attention │
        │ ffn                  if use_ffn │
        │ dropout     if dropout_rate > 0 │
        └─────────────────────────────────┘

        _common_processed_shape predicts this much of a
        subclass forward pass, and no more:

        input_shape (B, H, W, C)
                         │
                         ▼
        ┌─────────────────────────────────┐
        │ attention            (optional) │
        └────────────────┬────────────────┘
                         ▼
        ┌─────────────────────────────────┐
        │ ffn                  (optional) │
        └────────────────┬────────────────┘
                         ▼
        the subclass task layers

    Two warnings about that picture.

    ``norm`` and ``dropout`` are built, and ``norm`` carries weights, but no
    ``call`` in this module applies either one. Each head normalizes and
    drops inside its own ``ConvBlock`` or ``DenseBlock``, and those read
    ``normalization_type`` and ``dropout_rate`` from this class. So the two
    common layers are weight the forward pass never reaches, not a stage.

    The other two stages are not uniform either. :class:`ClassificationHead`
    applies ``attention`` and never ``ffn``, even though ``use_ffn`` defaults
    to ``True``. :class:`EnhancementHead` applies none of the four.
    :class:`MultiTaskHead` does not inherit from this class at all. Read the
    subclass diagram, not this one, for what a given head runs.

    :param hidden_dim: Working width of the head. The sub-blocks and the FFN
        use it as their channel count.
    :type hidden_dim: int
    :param normalization_type: Which registered normalization to build, and
        the value handed to each sub-block.
    :type normalization_type: NormalizationType
    :param activation_type: Activation each sub-block uses.
    :type activation_type: ActivationType
    :param dropout_rate: Dropout rate handed to the sub-blocks. Any value
        above 0 also builds the unused common ``dropout`` layer.
    :type dropout_rate: float
    :param use_attention: Whether to build an attention layer.
    :type use_attention: bool
    :param attention_type: Which registered attention type to build.
    :type attention_type: AttentionType
    :param use_ffn: Whether to build an FFN block. It defaults to ``True``,
        and :class:`ClassificationHead` never applies the result.
    :type use_ffn: bool
    :param ffn_type: Which registered FFN type to build.
    :type ffn_type: FFNType
    :param ffn_expansion_factor: The FFN inner width is
        ``hidden_dim * ffn_expansion_factor``.
    :type ffn_expansion_factor: int
    :param kwargs: Additional arguments for the base Layer class.

    :ivar norm: Normalization layer. Always built, never applied.
    :vartype norm: keras.layers.Layer
    :ivar attention: Attention layer, built when ``use_attention`` is set.
    :vartype attention: keras.layers.Layer
    :ivar ffn: FFN block, built when ``use_ffn`` is set.
    :vartype ffn: keras.layers.Layer
    :ivar dropout: Dropout layer, built when ``dropout_rate > 0``. Never
        applied.
    :vartype dropout: keras.layers.Dropout
    """

    def __init__(
            self,
            hidden_dim: int = 256,
            normalization_type: NormalizationType = 'layer_norm',
            activation_type: ActivationType = 'gelu',
            dropout_rate: float = 0.1,
            use_attention: bool = False,
            attention_type: AttentionType = 'multi_head',
            use_ffn: bool = True,
            ffn_type: FFNType = 'mlp',
            ffn_expansion_factor: int = 4,
            **kwargs: Any
    ) -> None:
        """
        Store the configuration and build the common layers.

        See the class docstring for what each argument means and which heads
        ignore it.

        :param hidden_dim: Working width of the head.
        :type hidden_dim: int
        :param normalization_type: Which registered normalization to build.
        :type normalization_type: NormalizationType
        :param activation_type: Activation the sub-blocks use.
        :type activation_type: ActivationType
        :param dropout_rate: Dropout rate handed to the sub-blocks.
        :type dropout_rate: float
        :param use_attention: Whether to build an attention layer.
        :type use_attention: bool
        :param attention_type: Which registered attention type to build.
        :type attention_type: AttentionType
        :param use_ffn: Whether to build an FFN block.
        :type use_ffn: bool
        :param ffn_type: Which registered FFN type to build.
        :type ffn_type: FFNType
        :param ffn_expansion_factor: FFN inner-width multiplier.
        :type ffn_expansion_factor: int
        :param kwargs: Additional arguments for the base Layer class.
        :return: None.
        :rtype: None
        """
        super().__init__(**kwargs)

        # Store configuration
        self.hidden_dim = hidden_dim
        self.normalization_type = normalization_type
        self.activation_type = activation_type
        self.dropout_rate = dropout_rate
        self.use_attention = use_attention
        self.attention_type = attention_type
        self.use_ffn = use_ffn
        self.ffn_type = ffn_type
        self.ffn_expansion_factor = ffn_expansion_factor

        # Create common layers
        self._create_common_layers()

    def _create_common_layers(self) -> None:
        """
        Build the four layers every vision head shares.

        ``norm`` is always built. ``attention``, ``ffn`` and ``dropout`` are
        built only when their flag or rate asks for them.

        The attention branch has three cases. ``'multi_head'`` and ``'cbam'``
        get their own argument sets. Every other type goes through
        ``assemble_attention_config``, which drops any keyword the type does
        not declare. ``create_attention_layer`` itself raises ``ValueError``
        on such a keyword. So passing ``dim`` to a type that has no such
        parameter would fail construction outright.

        Called from ``__init__``, following the package rule that layers are
        created in ``__init__``.

        :return: None.
        :rtype: None
        """

        # Normalization layer
        self.norm = create_normalization_layer(
            self.normalization_type,
            name=f'{self.name}_norm'
        )

        # Optional attention mechanism
        if self.use_attention:
            if self.attention_type == 'multi_head':
                self.attention = create_attention_layer(
                    'multi_head',
                    dim=self.hidden_dim,
                    num_heads=8,
                    dropout_rate=self.dropout_rate,
                    name=f'{self.name}_attention'
                )
            elif self.attention_type == 'cbam':
                self.attention = create_attention_layer(
                    'cbam',
                    channels=self.hidden_dim,
                    ratio=8,
                    name=f'{self.name}_cbam'
                )
            else:
                # DECISION plan-2026-08-17T183311-79c63e38/D-023
                # 11 of the 33 registered attention types do not declare `dim`:
                # 'cbam' (handled above), 'channel', 'spatial', the four
                # 'tripseN', 'capsule_routing', 'fnet', 'hopfield', 'non_local'.
                # Do NOT pass `dim` unconditionally. See decisions.md D-023.
                self.attention = create_attention_layer(
                    self.attention_type,
                    name=f'{self.name}_attention',
                    **assemble_attention_config(
                        self.attention_type, {'dim': self.hidden_dim}
                    )
                )

        # Optional FFN block
        if self.use_ffn:
            if self.ffn_type == 'swiglu':
                self.ffn = create_ffn_layer(
                    'swiglu',
                    output_dim=self.hidden_dim,
                    ffn_expansion_factor=self.ffn_expansion_factor,
                    dropout_rate=self.dropout_rate,
                    name=f'{self.name}_ffn'
                )
            else:
                self.ffn = create_ffn_layer(
                    self.ffn_type,
                    hidden_dim=self.hidden_dim * self.ffn_expansion_factor,
                    output_dim=self.hidden_dim,
                    dropout_rate=self.dropout_rate,
                    name=f'{self.name}_ffn'
                )

        # Dropout layer
        if self.dropout_rate > 0:
            self.dropout = layers.Dropout(self.dropout_rate)

    def _common_processed_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Report the shape after the attention and FFN stages.

        This mirrors the ``if self.use_attention: ... if self.use_ffn: ...``
        block that most subclass ``call`` methods run before their task
        layers. Subclass ``build`` methods use it to size what comes next.

        ``norm`` and ``dropout`` are not in this chain, because no ``call``
        applies them. :class:`ClassificationHead` and
        :class:`EnhancementHead` do not use this method at all.

        :param input_shape: Input feature-map shape.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Shape after the optional attention and FFN stages.
        :rtype: Tuple[Optional[int], ...]
        """
        shape = tuple(input_shape)
        if self.use_attention:
            shape = tuple(self.attention.compute_output_shape(shape))
        if self.use_ffn:
            shape = tuple(self.ffn.compute_output_shape(shape))
        return shape

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the four common layers.

        Each is built on the raw input shape, because each stage is
        shape-preserving in the width the head runs at. ``norm`` and
        ``dropout`` are built even though no ``call`` applies them.

        :param input_shape: Input feature-map shape.
        :type input_shape: Tuple[Optional[int], ...]
        :return: None.
        :rtype: None
        """
        # The common layers each consume the raw input feature map.
        self.norm.build(input_shape)
        if self.use_attention:
            self.attention.build(input_shape)
        if self.use_ffn:
            self.ffn.build(input_shape)
        if self.dropout_rate > 0:
            self.dropout.build(input_shape)

        super().build(input_shape)

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Report the output shape.

        The base head transforms nothing, so this returns the input shape
        unchanged. Every concrete subclass overrides it.

        :param input_shape: Input feature-map shape.
        :type input_shape: Tuple[Optional[int], ...]
        :return: The same shape.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return the constructor arguments for serialization.

        :return: Config dict carrying the nine constructor arguments, on top
            of the base Layer configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'hidden_dim': self.hidden_dim,
            'normalization_type': self.normalization_type,
            'activation_type': self.activation_type,
            'dropout_rate': self.dropout_rate,
            'use_attention': self.use_attention,
            'attention_type': self.attention_type,
            'use_ffn': self.use_ffn,
            'ffn_type': self.ffn_type,
            'ffn_expansion_factor': self.ffn_expansion_factor
        })
        return config


# ---------------------------------------------------------------------
# Detection Head
# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.heads.vision.factory")
class DetectionHead(BaseVisionHead):
    """
    Class scores and box offsets, one set per anchor per location.

    Two branches run on the same common-processed features. Each is a
    ``ConvBlock`` followed by a 1x1 conv. Neither branch changes the spatial
    dimensions, because every conv here is stride 1 with ``'same'`` padding.

    Neither output conv applies an activation. The scores are raw logits and
    the offsets are raw values, so a loss that expects logits is the right
    one here.

    **Architecture Overview:**

    .. code-block:: text

        inputs (B, H, W, C)
                         │
                         ▼
        ┌─────────────────────────────────┐
        │ attention            (optional) │
        └────────────────┬────────────────┘
                         ▼
        ┌─────────────────────────────────┐
        │ ffn                  (optional) │
        └────────────────┬────────────────┘
                         ▼ x, common-processed
                 ┌───────┴────────────┐
                 ▼                    ▼
        ┌──────────────────┐ ┌──────────────────┐
        │ cls_conv         │ │ reg_conv         │
        │ ConvBlock 3x3    │ │ ConvBlock 3x3    │
        └────────┬─────────┘ └────────┬─────────┘
                 ▼                    ▼
        ┌──────────────────┐ ┌──────────────────┐
        │ cls_head         │ │ reg_head         │
        │ Conv2D 1x1       │ │ Conv2D 1x1       │
        └──────────────────┘ └──────────────────┘
         'classifications'      'regressions'

        classifications: (B, H, W, num_anchors * num_classes)
        regressions:     (B, H, W, num_anchors * bbox_dims)

    :class:`InstanceSegmentationHead` holds one of these as a sub-layer.
    :func:`create_vision_head` also returns this class for keypoint
    detection, unchanged.

    Input shape:
        ``(batch, height, width, channels)``.

    Output shape:
        ``{'classifications': (batch, height, width, num_anchors *
        num_classes), 'regressions': (batch, height, width, num_anchors *
        bbox_dims)}``.

    :param num_classes: Number of object classes.
    :type num_classes: int
    :param num_anchors: Number of anchor boxes per location.
    :type num_anchors: int
    :param bbox_dims: Values per box. 4 for the usual corner or centre form.
    :type bbox_dims: int
    :param kwargs: Arguments for :class:`BaseVisionHead`.

    :ivar cls_conv: Classification-branch ``ConvBlock``.
    :vartype cls_conv: ConvBlock
    :ivar cls_head: Classification output conv.
    :vartype cls_head: keras.layers.Conv2D
    :ivar reg_conv: Regression-branch ``ConvBlock``.
    :vartype reg_conv: ConvBlock
    :ivar reg_head: Regression output conv.
    :vartype reg_head: keras.layers.Conv2D
    """

    def __init__(
            self,
            num_classes: int,
            num_anchors: int = 9,
            bbox_dims: int = 4,
            **kwargs: Any
    ) -> None:
        """
        Store the configuration, then build the two branches.

        :param num_classes: Number of object classes.
        :type num_classes: int
        :param num_anchors: Number of anchor boxes per location.
        :type num_anchors: int
        :param bbox_dims: Values per box.
        :type bbox_dims: int
        :param kwargs: Arguments for :class:`BaseVisionHead`.
        :return: None.
        :rtype: None
        """
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.bbox_dims = bbox_dims

        # Create detection-specific layers
        self._create_detection_layers()

    def _create_detection_layers(self) -> None:
        """
        Build the classification and regression branches.

        Each branch is a 3x3 ``ConvBlock`` at ``hidden_dim`` channels, then a
        1x1 conv with no activation.

        :return: None.
        :rtype: None
        """

        # Classification branch
        self.cls_conv = ConvBlock(
            filters=self.hidden_dim,
            kernel_size=3,
            normalization_type=self.normalization_type,
            activation_type=self.activation_type,
            dropout_rate=self.dropout_rate
        )

        self.cls_head = layers.Conv2D(
            filters=self.num_anchors * self.num_classes,
            kernel_size=1,
            padding='same',
            name='cls_head'
        )

        # Regression branch
        self.reg_conv = ConvBlock(
            filters=self.hidden_dim,
            kernel_size=3,
            normalization_type=self.normalization_type,
            activation_type=self.activation_type,
            dropout_rate=self.dropout_rate
        )

        self.reg_head = layers.Conv2D(
            filters=self.num_anchors * self.bbox_dims,
            kernel_size=1,
            padding='same',
            name='reg_head'
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build both branches on the common-processed shape.

        The common layers are built last, by the base class.

        :param input_shape: Input feature-map shape.
        :type input_shape: Tuple[Optional[int], ...]
        :return: None.
        :rtype: None
        """
        feature_shape = self._common_processed_shape(input_shape)

        self.cls_conv.build(feature_shape)
        self.cls_head.build(self.cls_conv.compute_output_shape(feature_shape))

        self.reg_conv.build(feature_shape)
        self.reg_head.build(self.reg_conv.compute_output_shape(feature_shape))

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Dict[str, keras.KerasTensor]:
        """
        Run the common stages, then both branches.

        :param inputs: Feature map of shape ``(batch, height, width,
            channels)``.
        :type inputs: keras.KerasTensor
        :param training: Keras training flag, forwarded to every sub-layer.
        :type training: Optional[bool]
        :return: Dict with ``'classifications'`` and ``'regressions'``.
        :rtype: Dict[str, keras.KerasTensor]
        """

        # Apply common processing if enabled
        x = inputs
        if self.use_attention:
            x = self.attention(x, training=training)
        if self.use_ffn:
            x = self.ffn(x, training=training)

        # Classification branch
        cls_features = self.cls_conv(x, training=training)
        cls_output = self.cls_head(cls_features)

        # Regression branch
        reg_features = self.reg_conv(x, training=training)
        reg_output = self.reg_head(reg_features)

        return {
            'classifications': cls_output,
            'regressions': reg_output
        }

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Dict[str, Tuple[Optional[int], ...]]:
        """
        Report the two output shapes.

        Spatial dimensions are preserved. Only the channel count changes.

        :param input_shape: Shape ``(batch, height, width, channels)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Dict with ``'classifications'`` and ``'regressions'``
            shapes.
        :rtype: Dict[str, Tuple[Optional[int], ...]]
        """
        batch, height, width = input_shape[0], input_shape[1], input_shape[2]
        return {
            'classifications': (batch, height, width, self.num_anchors * self.num_classes),
            'regressions': (batch, height, width, self.num_anchors * self.bbox_dims)
        }

    def get_config(self) -> Dict[str, Any]:
        """
        Return the constructor arguments for serialization.

        :return: Config dict carrying ``num_classes``, ``num_anchors`` and
            ``bbox_dims``, on top of the base configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'num_anchors': self.num_anchors,
            'bbox_dims': self.bbox_dims
        })
        return config


# ---------------------------------------------------------------------
# Segmentation Head
# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.heads.vision.factory")
class SegmentationHead(BaseVisionHead):
    """
    One class map, upsampled towards the input resolution.

    Three ``ConvBlock`` stages refine the features, then transposed convs
    double the resolution once per stage, then a 1x1 softmax conv emits one
    channel per class.

    This head accepts a single feature map or a list of them. With a list and
    ``use_skip_connections``, the last map is the one refined and the rest
    become skips in reverse order. One skip is concatenated after each
    refine block, while skips remain.

    This is the only head in the module that returns a bare tensor. Every
    other one returns a dict.

    **Architecture Overview:**

    .. code-block:: text

        inputs: (B, H, W, C), or a list of feature maps
                         │
                         ▼
        ┌─────────────────────────────────┐
        │ list input: x = the last map,   │
        │ skips = the rest, reversed      │
        └────────────────┬────────────────┘
                         ▼
        ┌─────────────────────────────────┐
        │ attention            (optional) │
        └────────────────┬────────────────┘
                         ▼
        ┌─────────────────────────────────┐
        │ ffn                  (optional) │
        └────────────────┬────────────────┘
                         ▼
        ┌─────────────────────────────────┐
        │ refine_blocks: 3 x ConvBlock    │
        │ concat skips[i] after block i   │
        └────────────────┬────────────────┘
                         ▼ channels: hidden_dim, then halved
        ┌─────────────────────────────────┐
        │ upsample_layers                 │
        │ Conv2DTranspose stride 2        │
        └────────────────┬────────────────┘
                         ▼ int(sqrt(upsampling_factor)) of them
        ┌─────────────────────────────────┐
        │ seg_head: Conv2D 1x1    softmax │
        └────────────────┬────────────────┘
                         ▼
        (B, H*s, W*s, num_classes), s = 2 ** the count above

    The refine blocks halve their channel count as they go: ``hidden_dim``,
    then ``hidden_dim // 2``, then ``hidden_dim // 4``. Every upsample layer
    uses ``hidden_dim // 8``, one halving further still. Concatenating a skip
    widens the tensor again, so the built shapes follow ``call`` rather than
    the block widths alone.

    The output resolution is ``2 ** int(upsampling_factor ** 0.5)`` times the
    input. With the default ``upsampling_factor=4`` that is 2 layers, so 4x.
    The value is a count of doublings once square-rooted, not the factor
    itself.

    Input shape:
        ``(batch, height, width, channels)``, or a list of such shapes.

    Output shape:
        ``(batch, out_height, out_width, num_classes)``.

    :param num_classes: Number of segmentation classes.
    :type num_classes: int
    :param upsampling_factor: Its integer square root is the number of
        stride-2 transposed convs to build.
    :type upsampling_factor: int
    :param use_skip_connections: Concatenate list inputs as skips. Ignored
        when the input is a single tensor.
    :type use_skip_connections: bool
    :param kwargs: Arguments for :class:`BaseVisionHead`.

    :ivar refine_blocks: The three refinement ``ConvBlock`` layers.
    :vartype refine_blocks: List[ConvBlock]
    :ivar upsample_layers: The stride-2 ``Conv2DTranspose`` layers.
    :vartype upsample_layers: List[keras.layers.Conv2DTranspose]
    :ivar seg_head: Output ``Conv2D(num_classes)`` with softmax.
    :vartype seg_head: keras.layers.Conv2D
    """

    def __init__(
            self,
            num_classes: int,
            upsampling_factor: int = 4,
            use_skip_connections: bool = True,
            **kwargs: Any
    ) -> None:
        """
        Store the configuration, then build the segmentation layers.

        :param num_classes: Number of segmentation classes.
        :type num_classes: int
        :param upsampling_factor: Its integer square root is the number of
            stride-2 transposed convs to build.
        :type upsampling_factor: int
        :param use_skip_connections: Concatenate list inputs as skips.
        :type use_skip_connections: bool
        :param kwargs: Arguments for :class:`BaseVisionHead`.
        :return: None.
        :rtype: None
        """
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.upsampling_factor = upsampling_factor
        self.use_skip_connections = use_skip_connections

        self._create_segmentation_layers()

    def _create_segmentation_layers(self) -> None:
        """
        Build the refine stack, the upsample stack and the output conv.

        Refine channel counts halve each step. The upsample layers all take
        the count left after the third halving.

        :return: None.
        :rtype: None
        """

        # Feature refinement blocks
        self.refine_blocks = []
        channels = self.hidden_dim

        for i in range(3):
            self.refine_blocks.append(
                ConvBlock(
                    filters=channels,
                    kernel_size=3,
                    normalization_type=self.normalization_type,
                    activation_type=self.activation_type,
                    dropout_rate=self.dropout_rate,
                    name=f'refine_block_{i}'
                )
            )
            channels = channels // 2

        # Upsampling layers
        self.upsample_layers = []
        for i in range(int(self.upsampling_factor ** 0.5)):
            self.upsample_layers.append(
                layers.Conv2DTranspose(
                    filters=channels,
                    kernel_size=3,
                    strides=2,
                    padding='same',
                    activation=self.activation_type,
                    name=f'upsample_{i}'
                )
            )

        # Final segmentation layer
        self.seg_head = layers.Conv2D(
            filters=self.num_classes,
            kernel_size=1,
            padding='same',
            activation='softmax',
            name='seg_head'
        )

    def build(
            self,
            input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]
    ) -> None:
        """
        Build every sub-layer on the shape ``call`` will hand it.

        This repeats the skip-concatenation arithmetic of ``call``, so a
        multi-scale input builds the same widths it will run with. The common
        layers are built on the single highest-level map.

        :param input_shape: One feature-map shape, or a list of shapes for a
            multi-scale input.
        :type input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]
        :return: None.
        :rtype: None
        """
        if isinstance(input_shape, list) and self.use_skip_connections:
            base_shape = input_shape[-1]
            skip_shapes = input_shape[:-1][::-1]
        else:
            base_shape = input_shape[-1] if isinstance(input_shape, list) else input_shape
            skip_shapes = []

        shape = self._common_processed_shape(base_shape)

        for i, refine_block in enumerate(self.refine_blocks):
            refine_block.build(shape)
            shape = tuple(refine_block.compute_output_shape(shape))
            if self.use_skip_connections and i < len(skip_shapes):
                skip_channels = skip_shapes[i][-1]
                merged = (
                    shape[-1] + skip_channels
                    if shape[-1] is not None and skip_channels is not None
                    else None
                )
                shape = shape[:-1] + (merged,)

        for upsample_layer in self.upsample_layers:
            upsample_layer.build(shape)
            shape = tuple(upsample_layer.compute_output_shape(shape))

        self.seg_head.build(shape)

        # Common layers operate on the single highest-level feature map.
        super().build(base_shape)

    def call(
            self,
            inputs: Union[keras.KerasTensor, List[keras.KerasTensor]],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Refine, concatenate skips, upsample, then classify each pixel.

        :param inputs: One feature map, or a list of them with the
            highest-level map last.
        :type inputs: Union[keras.KerasTensor, List[keras.KerasTensor]]
        :param training: Keras training flag, forwarded to every sub-layer.
        :type training: Optional[bool]
        :return: Class map tensor. This head returns no dict.
        :rtype: keras.KerasTensor
        """

        # Handle multi-scale inputs if skip connections are used
        if isinstance(inputs, list) and self.use_skip_connections:
            # The last entry is the highest-level map. The rest are skips,
            # reversed so the deepest one is consumed first.
            x = inputs[-1]
            skip_features = inputs[:-1][::-1]
        else:
            x = inputs if not isinstance(inputs, list) else inputs[-1]
            skip_features = []

        # Apply common processing
        if self.use_attention:
            x = self.attention(x, training=training)
        if self.use_ffn:
            x = self.ffn(x, training=training)

        # Refinement and upsampling
        for i, refine_block in enumerate(self.refine_blocks):
            x = refine_block(x, training=training)

            # Add skip connections if available
            if self.use_skip_connections and i < len(skip_features):
                x = ops.concatenate([x, skip_features[i]], axis=-1)

        # Upsample to original resolution
        for upsample_layer in self.upsample_layers:
            x = upsample_layer(x)

        # Final segmentation output
        seg_output = self.seg_head(x)

        return seg_output

    def compute_output_shape(
            self,
            input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]
    ) -> Tuple[Optional[int], ...]:
        """
        Report the output shape.

        Each of the ``int(upsampling_factor ** 0.5)`` transposed convs
        doubles height and width. The output conv emits ``num_classes``
        channels. A list input is measured by its last entry.

        :param input_shape: One feature-map shape, or a list of shapes.
        :type input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]
        :return: Shape ``(batch, out_height, out_width, num_classes)``.
        :rtype: Tuple[Optional[int], ...]
        """
        base_shape = input_shape[-1] if isinstance(input_shape, list) else input_shape
        batch, height, width = base_shape[0], base_shape[1], base_shape[2]
        scale = 2 ** int(self.upsampling_factor ** 0.5)
        out_height = height * scale if height is not None else None
        out_width = width * scale if width is not None else None
        return (batch, out_height, out_width, self.num_classes)

    def get_config(self) -> Dict[str, Any]:
        """
        Return the constructor arguments for serialization.

        :return: Config dict carrying ``num_classes``, ``upsampling_factor``
            and ``use_skip_connections``, on top of the base configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'upsampling_factor': self.upsampling_factor,
            'use_skip_connections': self.use_skip_connections
        })
        return config


# ---------------------------------------------------------------------
# Depth Estimation Head
# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.heads.vision.factory")
class DepthEstimationHead(BaseVisionHead):
    """
    A depth map, at eight times the input resolution.

    Three stages each refine the features and then double height and width,
    so the output is ``8H`` by ``8W``. A sigmoid conv produces a normalized
    map, which is then scaled into ``[min_depth, max_depth]``.

    **Architecture Overview:**

    .. code-block:: text

        inputs (B, H, W, C)
                         │
                         ▼
        ┌─────────────────────────────────┐
        │ attention            (optional) │
        └────────────────┬────────────────┘
                         ▼
        ┌─────────────────────────────────┐
        │ ffn                  (optional) │
        └────────────────┬────────────────┘
                         ▼
        ┌─────────────────────────────────┐
        │ depth_blocks: 3 x (ConvBlock    │
        │ then Conv2DTranspose stride 2)  │
        └────────────────┬────────────────┘
                         ▼ 8x spatial, channels halved 3x
        ┌─────────────────────────────────┐
        │ depth_head: Conv2D 3x3  sigmoid │
        └────────────────┬────────────────┘
                         ▼ depth_normalized, in [0, 1]
                 ┌───────┴────────────┐
                 ▼                    ▼
        ┌──────────────────┐ ┌──────────────────┐
        │ use_log_depth:   │ │ passed through   │
        │ exp(...) or a    │ │ with no change   │
        │ linear rescale   │ │                  │
        └────────┬─────────┘ └────────┬─────────┘
                 ▼                    ▼
              'depth'            'confidence'

        depth, confidence: (B, 8H, 8W, output_channels)

    ``'confidence'`` is ``depth_normalized`` itself, returned unchanged. It
    is the sigmoid output the depth was scaled from, not a second estimate
    and not an uncertainty. Treat it as a confidence only if you have checked
    that reading against your data.

    The scaling has two forms. With ``use_log_depth`` the map is read as a
    position in log space: ``exp(x * (log(max_depth) - log(min_depth)) +
    log(min_depth))``. Otherwise it is read linearly: ``x * (max_depth -
    min_depth) + min_depth``.

    :func:`create_vision_head` reuses this class for two other tasks. Surface
    normals get ``output_channels=3`` and optical flow gets
    ``output_channels=2``. Neither changes the scaling, so a flow vector
    comes out squeezed into the depth range.

    Input shape:
        ``(batch, height, width, channels)``.

    Output shape:
        ``{'depth': (batch, 8 * height, 8 * width, output_channels),
        'confidence': the same shape}``.

    :param output_channels: Channel count of the output map. 1 for depth.
    :type output_channels: int
    :param min_depth: Lower end of the output range.
    :type min_depth: float
    :param max_depth: Upper end of the output range.
    :type max_depth: float
    :param use_log_depth: Scale in log space instead of linearly.
    :type use_log_depth: bool
    :param kwargs: Arguments for :class:`BaseVisionHead`.

    :ivar depth_blocks: Three ``[ConvBlock, Conv2DTranspose]`` pairs.
    :vartype depth_blocks: List[List[keras.layers.Layer]]
    :ivar depth_head: Output ``Conv2D(output_channels)`` with sigmoid.
    :vartype depth_head: keras.layers.Conv2D
    """

    def __init__(
            self,
            output_channels: int = 1,
            min_depth: float = 0.1,
            max_depth: float = 100.0,
            use_log_depth: bool = True,
            **kwargs: Any
    ) -> None:
        """
        Store the configuration, then build the depth layers.

        :param output_channels: Channel count of the output map.
        :type output_channels: int
        :param min_depth: Lower end of the output range.
        :type min_depth: float
        :param max_depth: Upper end of the output range.
        :type max_depth: float
        :param use_log_depth: Scale in log space instead of linearly.
        :type use_log_depth: bool
        :param kwargs: Arguments for :class:`BaseVisionHead`.
        :return: None.
        :rtype: None
        """
        super().__init__(**kwargs)

        self.output_channels = output_channels
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.use_log_depth = use_log_depth

        self._create_depth_layers()

    def _create_depth_layers(self) -> None:
        """
        Build the refinement stack and the depth conv.

        Three ``[ConvBlock, Conv2DTranspose]`` pairs. Each transposed conv
        has stride 2 and halves the channel count.

        :return: None.
        :rtype: None
        """

        # Progressive upsampling with refinement
        self.depth_blocks = []
        channels = self.hidden_dim

        for i in range(3):
            self.depth_blocks.append([
                ConvBlock(
                    filters=channels,
                    kernel_size=3,
                    normalization_type=self.normalization_type,
                    activation_type=self.activation_type,
                    name=f'depth_conv_{i}'
                ),
                layers.Conv2DTranspose(
                    filters=channels // 2,
                    kernel_size=3,
                    strides=2,
                    padding='same',
                    name=f'depth_upsample_{i}'
                )
            ])
            channels = channels // 2

        # Depth prediction layer
        self.depth_head = layers.Conv2D(
            filters=self.output_channels,
            kernel_size=3,
            padding='same',
            # Sigmoid keeps the output in [0, 1]. call() then scales it into
            # the [min_depth, max_depth] range.
            activation='sigmoid',
            name='depth_head'
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the depth stack on the common-processed shape.

        :param input_shape: Input feature-map shape.
        :type input_shape: Tuple[Optional[int], ...]
        :return: None.
        :rtype: None
        """
        shape = self._common_processed_shape(input_shape)

        for conv_block, upsample in self.depth_blocks:
            conv_block.build(shape)
            shape = tuple(conv_block.compute_output_shape(shape))
            upsample.build(shape)
            shape = tuple(upsample.compute_output_shape(shape))

        self.depth_head.build(shape)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Dict[str, keras.KerasTensor]:
        """
        Refine, upsample, predict, then scale into the depth range.

        :param inputs: Feature map of shape ``(batch, height, width,
            channels)``.
        :type inputs: keras.KerasTensor
        :param training: Keras training flag, forwarded to every sub-layer.
        :type training: Optional[bool]
        :return: Dict with ``'depth'`` and ``'confidence'``, where
            ``'confidence'`` is the unscaled sigmoid output.
        :rtype: Dict[str, keras.KerasTensor]
        """

        x = inputs

        # Apply common processing
        if self.use_attention:
            x = self.attention(x, training=training)
        if self.use_ffn:
            x = self.ffn(x, training=training)

        # Progressive refinement and upsampling
        for conv_block, upsample in self.depth_blocks:
            x = conv_block(x, training=training)
            x = upsample(x)

        # Predict normalized depth
        depth_normalized = self.depth_head(x)

        # Scale to actual depth range
        if self.use_log_depth:
            # Convert from log space
            log_min = ops.log(self.min_depth)
            log_max = ops.log(self.max_depth)
            depth = ops.exp(depth_normalized * (log_max - log_min) + log_min)
        else:
            # Linear scaling
            depth = depth_normalized * (self.max_depth - self.min_depth) + self.min_depth

        return {
            'depth': depth,
            # This is the sigmoid output the depth was scaled from, not a
            # separate estimate. Read the class docstring before using it.
            'confidence': depth_normalized
        }

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Dict[str, Tuple[Optional[int], ...]]:
        """
        Report the two output shapes.

        Three transposed convs of stride 2 give eight times the input height
        and width. ``'depth'`` and ``'confidence'`` share one shape.

        :param input_shape: Shape ``(batch, height, width, channels)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Dict with ``'depth'`` and ``'confidence'`` shapes.
        :rtype: Dict[str, Tuple[Optional[int], ...]]
        """
        batch, height, width = input_shape[0], input_shape[1], input_shape[2]
        # Three transposed-conv upsamples, stride 2 each.
        scale = 8
        out_height = height * scale if height is not None else None
        out_width = width * scale if width is not None else None
        shape = (batch, out_height, out_width, self.output_channels)
        return {'depth': shape, 'confidence': shape}

    def get_config(self) -> Dict[str, Any]:
        """
        Return the constructor arguments for serialization.

        :return: Config dict carrying ``output_channels``, ``min_depth``,
            ``max_depth`` and ``use_log_depth``, on top of the base
            configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'output_channels': self.output_channels,
            'min_depth': self.min_depth,
            'max_depth': self.max_depth,
            'use_log_depth': self.use_log_depth
        })
        return config


# ---------------------------------------------------------------------
# Classification Head
# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.heads.vision.factory")
class ClassificationHead(BaseVisionHead):
    """
    One label for a whole image.

    Spatial features are collapsed to one vector per image, pushed through
    two dense blocks, and scored by a softmax ``Dense``. Use this for image
    classification.

    This head applies ``attention`` and never applies ``ffn``, even though
    ``BaseVisionHead.use_ffn`` defaults to ``True``. Setting ``use_ffn``
    builds an FFN block that carries weights and never runs. ``build``
    matches ``call`` here: it propagates the attention shape only.

    **Architecture Overview:**

    .. code-block:: text

        inputs (B, H, W, C)
                         │
                         ▼
        ┌─────────────────────────────────┐
        │ attention            (optional) │
        └────────────────┬────────────────┘
                         ▼
                 ┌───────┴────────────┐
                 ▼                    ▼
         use_global_pooling       otherwise
        ┌──────────────────┐ ┌──────────────────┐
        │ pooling          │ │ Flatten          │
        │ avg or max       │ │ made in call()   │
        └────────┬─────────┘ └────────┬─────────┘
                 └───────┬────────────┘
                         ▼ (B, C) or (B, H*W*C)
        ┌─────────────────────────────────┐
        │ dense_blocks: 2 x DenseBlock    │
        └────────────────┬────────────────┘
                         ▼
        ┌─────────────────────────────────┐
        │ classifier: Dense(num_classes)  │
        └────────────────┬────────────────┘
                         ▼ softmax applied here
                 ┌───────┴────────────┐
                 ▼                    ▼
              'logits'         'probabilities'

        One tensor reaches both keys. Nothing is computed twice.

    The two returned keys hold the identical tensor. ``classifier`` already
    applies softmax, so ``'logits'`` are probabilities too, despite the name.
    A loss expecting raw logits will be wrong here.

    With ``use_global_pooling=False`` the flatten is a ``layers.Flatten()``
    created inside ``call`` on every invocation. It is not a stored
    sub-layer, so it appears in no weight list. ``Flatten`` holds no weights,
    so this costs correctness nothing.

    Input shape:
        ``(batch, height, width, channels)``.

    Output shape:
        ``{'logits': (batch, num_classes),
        'probabilities': (batch, num_classes)}``.

    :param num_classes: Number of classes.
    :type num_classes: int
    :param use_global_pooling: Collapse the spatial dimensions by pooling.
        When ``False``, ``call`` flattens them instead.
    :type use_global_pooling: bool
    :param pooling_type: Which pooling to build, ``'avg'`` or ``'max'``.
        Read only when ``use_global_pooling`` is set.
    :type pooling_type: Literal['avg', 'max']
    :param kwargs: Arguments for :class:`BaseVisionHead`.

    :ivar pooling: Global pooling layer, built only when
        ``use_global_pooling`` is set.
    :vartype pooling: keras.layers.Layer
    :ivar dense_blocks: Two ``DenseBlock`` layers, ``hidden_dim`` wide then
        ``hidden_dim // 2`` wide.
    :vartype dense_blocks: List[DenseBlock]
    :ivar classifier: Output ``Dense(num_classes)`` with softmax.
    :vartype classifier: keras.layers.Dense
    """

    def __init__(
            self,
            num_classes: int,
            use_global_pooling: bool = True,
            pooling_type: Literal['avg', 'max'] = 'avg',
            **kwargs: Any
    ) -> None:
        """
        Store the configuration, then build the classification layers.

        :param num_classes: Number of classes.
        :type num_classes: int
        :param use_global_pooling: Collapse spatial dimensions by pooling.
        :type use_global_pooling: bool
        :param pooling_type: Which pooling to build, ``'avg'`` or ``'max'``.
        :type pooling_type: Literal['avg', 'max']
        :param kwargs: Arguments for :class:`BaseVisionHead`.
        :return: None.
        :rtype: None
        """
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.use_global_pooling = use_global_pooling
        self.pooling_type = pooling_type

        self._create_classification_layers()

    def _create_classification_layers(self) -> None:
        """
        Build the pooling layer, the dense blocks and the classifier.

        Pooling is built only when ``use_global_pooling`` is set. The two
        dense blocks are ``hidden_dim`` and ``hidden_dim // 2`` wide.

        :return: None.
        :rtype: None
        """

        # Global pooling
        if self.use_global_pooling:
            if self.pooling_type == 'avg':
                self.pooling = layers.GlobalAveragePooling2D()
            else:
                self.pooling = layers.GlobalMaxPooling2D()

        # Dense layers for classification
        self.dense_blocks = [
            DenseBlock(
                units=self.hidden_dim,
                normalization_type=self.normalization_type,
                activation_type=self.activation_type,
                dropout_rate=self.dropout_rate
            ),
            DenseBlock(
                units=self.hidden_dim // 2,
                normalization_type=self.normalization_type,
                activation_type=self.activation_type,
                dropout_rate=self.dropout_rate
            )
        ]

        # Final classifier
        self.classifier = layers.Dense(
            units=self.num_classes,
            activation='softmax',
            name='classifier'
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the classification layers on the shapes ``call`` produces.

        This does not use ``_common_processed_shape``, because ``call``
        applies attention and skips the FFN. The flatten branch computes the
        flat width here rather than building a layer, since ``call`` makes
        its own ``Flatten``.

        :param input_shape: Input feature-map shape.
        :type input_shape: Tuple[Optional[int], ...]
        :return: None.
        :rtype: None
        """
        shape = tuple(input_shape)
        # Classification call() applies attention only (no FFN) before pooling.
        if self.use_attention:
            shape = tuple(self.attention.compute_output_shape(shape))

        if self.use_global_pooling:
            self.pooling.build(shape)
            shape = tuple(self.pooling.compute_output_shape(shape))
        else:
            # call() flattens via an inline (stateless) Flatten layer.
            flat = 1
            for dim in shape[1:]:
                flat = None if (dim is None or flat is None) else flat * dim
            shape = (shape[0], flat)

        for dense_block in self.dense_blocks:
            dense_block.build(shape)
            shape = tuple(dense_block.compute_output_shape(shape))

        self.classifier.build(shape)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Dict[str, keras.KerasTensor]:
        """
        Pool or flatten, run the dense blocks, then classify.

        The FFN never runs here, whatever ``use_ffn`` says.

        :param inputs: Feature map of shape ``(batch, height, width,
            channels)``.
        :type inputs: keras.KerasTensor
        :param training: Keras training flag, forwarded to the dense blocks
            and to attention.
        :type training: Optional[bool]
        :return: Dict whose ``'logits'`` and ``'probabilities'`` keys hold the
            same softmax output.
        :rtype: Dict[str, keras.KerasTensor]
        """

        x = inputs

        # Apply attention if enabled
        if self.use_attention:
            x = self.attention(x, training=training)

        # Global pooling
        if self.use_global_pooling:
            x = self.pooling(x)
        else:
            x = layers.Flatten()(x)

        # Dense layers
        for dense_block in self.dense_blocks:
            x = dense_block(x, training=training)

        # Final classification
        logits = self.classifier(x)

        return {
            'logits': logits,
            # The classifier already applied softmax, so this is the same
            # tensor under a second name, not a second computation.
            'probabilities': logits
        }

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Dict[str, Tuple[Optional[int], ...]]:
        """
        Report the two output shapes.

        Both keys carry ``(batch, num_classes)``.

        :param input_shape: Shape ``(batch, height, width, channels)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Dict with ``'logits'`` and ``'probabilities'`` shapes.
        :rtype: Dict[str, Tuple[Optional[int], ...]]
        """
        batch = input_shape[0]
        shape = (batch, self.num_classes)
        return {'logits': shape, 'probabilities': shape}

    def get_config(self) -> Dict[str, Any]:
        """
        Return the constructor arguments for serialization.

        :return: Config dict carrying ``num_classes``, ``use_global_pooling``
            and ``pooling_type``, on top of the base configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'use_global_pooling': self.use_global_pooling,
            'pooling_type': self.pooling_type
        })
        return config


# ---------------------------------------------------------------------
# Instance Segmentation Head
# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.heads.vision.factory")
class InstanceSegmentationHead(BaseVisionHead):
    """
    Boxes, class scores and per-instance masks from one feature map.

    Two branches run on the same common-processed features. One is a full
    :class:`DetectionHead` instance held as a sub-layer, which supplies
    ``'classifications'`` and ``'regressions'``. The other is three
    ``ConvBlock`` stages and a 1x1 conv, which supplies ``'instance_masks'``.

    The inner detection head is built with ``use_attention=False`` and
    ``use_ffn=False``, because this head has already applied both. Without
    that the same two stages would run twice.

    **Architecture Overview:**

    .. code-block:: text

        inputs (B, H, W, C)
                         │
                         ▼
        ┌─────────────────────────────────┐
        │ attention            (optional) │
        └────────────────┬────────────────┘
                         ▼
        ┌─────────────────────────────────┐
        │ ffn                  (optional) │
        └────────────────┬────────────────┘
                         ▼ x, common-processed
                 ┌───────┴────────────┐
                 ▼                    ▼
        ┌──────────────────┐ ┌──────────────────┐
        │ detection_head   │ │ mask_conv_blocks │
        │ a DetectionHead  │ │ 3 x ConvBlock    │
        └────────┬─────────┘ └────────┬─────────┘
                 │                    ▼
                 │           ┌──────────────────┐
                 │           │ mask_head 1x1    │
                 │           │ sigmoid          │
                 │           └────────┬─────────┘
                 ▼                    ▼
         'classifications'     'instance_masks'
           'regressions'

        classifications: (B, H, W, num_anchors * num_classes)
        regressions:     (B, H, W, num_anchors * bbox_dims)
        instance_masks:  (B, H, W, num_instances)

    Spatial dimensions are preserved throughout. Every conv here is stride 1
    with ``'same'`` padding.

    ``mask_size`` is stored and serialized but no layer reads it. The masks
    come out at the input feature-map resolution.

    Input shape:
        ``(batch, height, width, channels)``.

    Output shape:
        ``{'classifications': (batch, height, width, num_anchors *
        num_classes), 'regressions': (batch, height, width, num_anchors *
        bbox_dims), 'instance_masks': (batch, height, width,
        num_instances)}``.

    :param num_classes: Number of object classes, forwarded to the inner
        detection head.
    :type num_classes: int
    :param num_instances: Channel count of the mask output. One channel per
        instance slot.
    :type num_instances: int
    :param mask_size: Stored for serialization. No layer reads it.
    :type mask_size: Tuple[int, int]
    :param kwargs: Arguments for :class:`BaseVisionHead`.

    :ivar detection_head: The inner :class:`DetectionHead` instance.
    :vartype detection_head: DetectionHead
    :ivar mask_conv_blocks: The three mask ``ConvBlock`` layers.
    :vartype mask_conv_blocks: List[ConvBlock]
    :ivar mask_head: Output ``Conv2D(num_instances)`` with sigmoid.
    :vartype mask_head: keras.layers.Conv2D
    """

    def __init__(
            self,
            num_classes: int,
            num_instances: int = 100,
            mask_size: Tuple[int, int] = (28, 28),
            **kwargs: Any
    ) -> None:
        """
        Store the configuration, then build the two branches.

        :param num_classes: Number of object classes.
        :type num_classes: int
        :param num_instances: Channel count of the mask output.
        :type num_instances: int
        :param mask_size: Stored for serialization only.
        :type mask_size: Tuple[int, int]
        :param kwargs: Arguments for :class:`BaseVisionHead`.
        :return: None.
        :rtype: None
        """
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.num_instances = num_instances
        self.mask_size = mask_size

        # Create sub-heads
        self.detection_head = DetectionHead(
            num_classes=num_classes,
            hidden_dim=self.hidden_dim,
            normalization_type=self.normalization_type,
            activation_type=self.activation_type,
            dropout_rate=self.dropout_rate,
            # Attention and the FFN have already run in this head, so the
            # inner detection head must not repeat them.
            use_attention=False,
            use_ffn=False
        )

        self._create_mask_layers()

    def _create_mask_layers(self) -> None:
        """
        Build the mask branch.

        Three ``ConvBlock`` stages at ``hidden_dim`` channels, then a 1x1
        conv with sigmoid emitting ``num_instances`` channels.

        :return: None.
        :rtype: None
        """

        # Mask feature extraction
        self.mask_conv_blocks = [
            ConvBlock(
                filters=self.hidden_dim,
                kernel_size=3,
                normalization_type=self.normalization_type,
                activation_type=self.activation_type
            )
            for _ in range(3)
        ]

        # Mask prediction head
        self.mask_head = layers.Conv2D(
            filters=self.num_instances,
            kernel_size=1,
            activation='sigmoid',
            name='mask_head'
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the detection sub-head and the mask branch.

        Both consume the shape ``_common_processed_shape`` predicts, not the
        raw input shape. The common layers are built last, by the base class.

        :param input_shape: Input feature-map shape.
        :type input_shape: Tuple[Optional[int], ...]
        :return: None.
        :rtype: None
        """
        feature_shape = self._common_processed_shape(input_shape)

        # Detection sub-head consumes the common-processed features.
        self.detection_head.build(feature_shape)

        shape = tuple(feature_shape)
        for mask_conv in self.mask_conv_blocks:
            mask_conv.build(shape)
            shape = tuple(mask_conv.compute_output_shape(shape))
        self.mask_head.build(shape)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Dict[str, keras.KerasTensor]:
        """
        Run the common stages, then both branches.

        :param inputs: Feature map of shape ``(batch, height, width,
            channels)``.
        :type inputs: keras.KerasTensor
        :param training: Keras training flag, forwarded to every sub-layer.
        :type training: Optional[bool]
        :return: Dict with ``'classifications'``, ``'regressions'`` and
            ``'instance_masks'``.
        :rtype: Dict[str, keras.KerasTensor]
        """

        x = inputs

        # Apply common processing
        if self.use_attention:
            x = self.attention(x, training=training)
        if self.use_ffn:
            x = self.ffn(x, training=training)

        # Get detection outputs
        detection_outputs = self.detection_head(x, training=training)

        # Mask prediction branch
        mask_features = x
        for mask_conv in self.mask_conv_blocks:
            mask_features = mask_conv(mask_features, training=training)

        instance_masks = self.mask_head(mask_features)

        return {
            'classifications': detection_outputs['classifications'],
            'regressions': detection_outputs['regressions'],
            'instance_masks': instance_masks
        }

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Dict[str, Tuple[Optional[int], ...]]:
        """
        Report the three output shapes.

        The two detection shapes come from the inner
        :class:`DetectionHead`. The mask shape keeps the input spatial
        dimensions and carries ``num_instances`` channels.

        :param input_shape: Shape ``(batch, height, width, channels)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Dict with ``'classifications'``, ``'regressions'`` and
            ``'instance_masks'`` shapes.
        :rtype: Dict[str, Tuple[Optional[int], ...]]
        """
        batch, height, width = input_shape[0], input_shape[1], input_shape[2]
        detection_shapes = self.detection_head.compute_output_shape(input_shape)
        return {
            'classifications': detection_shapes['classifications'],
            'regressions': detection_shapes['regressions'],
            'instance_masks': (batch, height, width, self.num_instances)
        }

    def get_config(self) -> Dict[str, Any]:
        """
        Return the constructor arguments for serialization.

        :return: Config dict carrying ``num_classes``, ``num_instances`` and
            ``mask_size``, on top of the base configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'num_instances': self.num_instances,
            'mask_size': self.mask_size
        })
        return config


# ---------------------------------------------------------------------
# Enhancement Head
# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.heads.vision.factory")
class EnhancementHead(BaseVisionHead):
    """
    A restored or upscaled image, built from a feature map.

    Use this for denoising, super-resolution and the other image-to-image
    tasks. Three ``ConvBlock`` stages refine the features. Then one conv
    produces the image: a transposed conv when ``scale_factor > 1``, and a
    same-resolution conv otherwise.

    This head applies none of the common stages. ``norm``, ``dropout``,
    ``attention`` and ``ffn`` are all built by :class:`BaseVisionHead`, and
    ``call`` reaches none of them. ``build`` does not call
    ``_common_processed_shape`` either. The head does read ``hidden_dim``,
    ``normalization_type`` and ``activation_type``, which it hands to its own
    ``ConvBlock`` stages.

    **Architecture Overview:**

    .. code-block:: text

        inputs (B, H, W, C)
                          │
                          ▼
        ┌─────────────────────────────────┐
        │ enhance_blocks: 3 x ConvBlock   │ -> hidden_dim
        └────────────────┬────────────────┘
                          ▼
                 ┌────────┴────────┐
                 ▼                 ▼
          scale_factor > 1     otherwise
        ┌─────────────────┐ ┌─────────────────┐
        │ upsample        │ │ output_conv     │
        │ Conv2DTranspose │ │ Conv2D 3x3      │
        └────────┬────────┘ └────────┬────────┘
                 └─────────┬─────────┘
                           ▼
                       'enhanced'

        enhanced: (B, H*s, W*s, output_channels) when s > 1,
                  else (B, H, W, output_channels). s = scale_factor.

    The class name is frozen. This class once lived inside
    ``create_enhancement_head`` as a closure-local registered class, and was
    lifted to module scope. Do NOT re-nest it there. Its decorator is now
    ``@register_dl_technique("dl_techniques.layers.heads.vision.factory")``,
    so ``get_registered_name`` resolves
    ``dl_techniques.layers.heads.vision.factory>EnhancementHead``. The helper
    also mints a legacy ``Custom>EnhancementHead`` alias. That alias is what a
    checkpoint written before 2026-08-29 reads, and it was verified on
    2026-08-29 to resolve to this same class. It is keyed on the bare class
    name, so a rename would drop it and break those archives.

    Input shape:
        ``(batch, height, width, channels)``.

    Output shape:
        ``{'enhanced': (batch, out_height, out_width, output_channels)}``.

    :param output_channels: Channel count of the output image.
    :type output_channels: int
    :param scale_factor: Spatial upscaling factor. Above 1 selects the
        transposed-conv path; 1 selects the same-resolution conv.
    :type scale_factor: int
    :param kwargs: Forwarded to :class:`BaseVisionHead`.

    :ivar enhance_blocks: The three refinement ``ConvBlock`` layers.
    :vartype enhance_blocks: List[ConvBlock]
    :ivar upsample: Transposed conv, built only when ``scale_factor > 1``.
    :vartype upsample: keras.layers.Conv2DTranspose
    :ivar output_conv: Output conv, built only when ``scale_factor <= 1``.
    :vartype output_conv: keras.layers.Conv2D
    """

    def __init__(self, output_channels: int = 3, scale_factor: int = 1, **kwargs):
        """
        Store the configuration and build the enhancement layers.

        The base class builds the four common layers, which this head never
        applies. Only one of ``upsample`` and ``output_conv`` is built, chosen
        by ``scale_factor``.

        :param output_channels: Channel count of the output image.
        :type output_channels: int
        :param scale_factor: Spatial upscaling factor.
        :type scale_factor: int
        :param kwargs: Forwarded to :class:`BaseVisionHead`.
        :return: None.
        :rtype: None
        """
        super().__init__(**kwargs)
        self.output_channels = output_channels
        self.scale_factor = scale_factor

        # Enhancement-specific layers
        self.enhance_blocks = [
            ConvBlock(
                filters=self.hidden_dim,
                kernel_size=3,
                normalization_type=self.normalization_type,
                activation_type=self.activation_type
            )
            for _ in range(3)
        ]

        if self.scale_factor > 1:
            # For super-resolution
            self.upsample = layers.Conv2DTranspose(
                filters=self.output_channels,
                kernel_size=3,
                strides=self.scale_factor,
                padding='same'
            )
        else:
            # For denoising and other tasks
            self.output_conv = layers.Conv2D(
                filters=self.output_channels,
                kernel_size=3,
                padding='same'
            )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the refinement stack and the output conv.

        Each ``ConvBlock`` is built on the shape the previous one returns.
        The common layers are built afterwards by the base class, even though
        this head never applies them.

        :param input_shape: Input feature-map shape.
        :type input_shape: Tuple[Optional[int], ...]
        :return: None.
        :rtype: None
        """
        shape = tuple(input_shape)
        for block in self.enhance_blocks:
            block.build(shape)
            shape = tuple(block.compute_output_shape(shape))

        if self.scale_factor > 1:
            self.upsample.build(shape)
        else:
            self.output_conv.build(shape)

        super().build(input_shape)

    def call(self, inputs, training=None):
        """
        Refine the features, then produce the output image.

        None of the four common stages runs here. See the class docstring.

        :param inputs: Feature map of shape ``(batch, height, width,
            channels)``.
        :type inputs: keras.KerasTensor
        :param training: Keras training flag, forwarded to each ``ConvBlock``.
        :type training: Optional[bool]
        :return: Dict with a single ``'enhanced'`` key.
        :rtype: Dict[str, keras.KerasTensor]
        """
        x = inputs

        for block in self.enhance_blocks:
            x = block(x, training=training)

        if self.scale_factor > 1:
            x = self.upsample(x)
        else:
            x = self.output_conv(x)

        return {'enhanced': x}

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Dict[str, Tuple[Optional[int], ...]]:
        """
        Report the output shape.

        With ``scale_factor > 1`` the transposed conv scales height and width
        by ``scale_factor``. Otherwise the output conv keeps them. Channels
        become ``output_channels`` either way.

        :param input_shape: Shape ``(batch, height, width, channels)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Dict with the ``'enhanced'`` output shape.
        :rtype: Dict[str, Tuple[Optional[int], ...]]
        """
        batch, height, width = input_shape[0], input_shape[1], input_shape[2]
        scale = self.scale_factor if self.scale_factor > 1 else 1
        out_height = height * scale if height is not None else None
        out_width = width * scale if width is not None else None
        return {'enhanced': (batch, out_height, out_width, self.output_channels)}

    def get_config(self):
        """
        Return the constructor arguments for serialization.

        Adds this head's two arguments to the base configuration.

        :return: Config dict carrying ``output_channels`` and
            ``scale_factor``.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'output_channels': self.output_channels,
            'scale_factor': self.scale_factor
        })
        return config


# ---------------------------------------------------------------------
# Multi-Task Head
# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.heads.vision.factory")
class MultiTaskHead(keras.layers.Layer):
    """
    Several vision heads behind one layer.

    Each entry in ``task_configs`` becomes one head. The heads are
    independent: they share the input, not any weights. This class does not
    inherit from :class:`BaseVisionHead`, so it builds no common layers of
    its own.

    A dict input is read per task name. A task name missing from the dict
    falls back to the ``'shared'`` entry. A single tensor is sent to every
    head unchanged.

    **Architecture Overview:**

    .. code-block:: text

        inputs: (B, H, W, C), or a dict keyed by task name
        with a 'shared' fallback for any name it omits
                                   │
                      ┌────────────┴────────────┐
                      ▼            ▼            ▼
                 ┌───────────┐┌───────────┐┌───────────┐
                 │ head[t1]  ││ head[t2]  ││ head[tN]  │
                 └─────┬─────┘└─────┬─────┘└─────┬─────┘
                       ▼            ▼            ▼
                  outputs[t1]  outputs[t2]  outputs[tN]

        Each head is one of five classes:

        DetectionHead        SegmentationHead
        DepthEstimationHead  ClassificationHead
        InstanceSegmentationHead

        Any other task type raises ValueError.

    Every head returns a dict, so ``call`` returns a dict of dicts: task name
    to that head's own output keys.

    Input shape:
        ``(batch, height, width, channels)``, or a dict of such shapes keyed
        by task name.

    Output shape:
        Dict mapping each task name to that head's output shapes.

    :param task_configs: Dict mapping a task name to that head's config. Each
        config carries a ``'task_type'`` key naming one of the five classes.
    :type task_configs: Dict[str, Dict[str, Any]]
    :param shared_backbone_dim: Default ``hidden_dim`` for any task config
        that does not set one.
    :type shared_backbone_dim: int
    :param use_task_specific_attention: Value forced onto every task head's
        ``use_attention``. A per-task ``use_attention`` is overwritten.
    :type use_task_specific_attention: bool
    :param kwargs: Additional arguments for the base Layer class.

    :ivar task_heads: Dict mapping each task name to its head instance.
    :vartype task_heads: Dict[str, keras.layers.Layer]

    :raises ValueError: From ``_create_task_heads``, when a config names a
        task type with no head.
    """

    def __init__(
            self,
            task_configs: Dict[str, Dict[str, Any]],
            shared_backbone_dim: int = 256,
            use_task_specific_attention: bool = True,
            **kwargs: Any
    ) -> None:
        """
        Store the configuration and build one head per task.

        :param task_configs: Dict mapping a task name to that head's config.
            Each config carries a ``'task_type'`` key.
        :type task_configs: Dict[str, Dict[str, Any]]
        :param shared_backbone_dim: Default ``hidden_dim`` for any task
            config that does not set one.
        :type shared_backbone_dim: int
        :param use_task_specific_attention: Value forced onto every task
            head's ``use_attention``.
        :type use_task_specific_attention: bool
        :param kwargs: Additional arguments for the base Layer class.
        :return: None.
        :rtype: None
        :raises ValueError: If a config names a task type with no head.
        """
        super().__init__(**kwargs)

        self.task_configs = task_configs
        self.shared_backbone_dim = shared_backbone_dim
        self.use_task_specific_attention = use_task_specific_attention

        self._create_task_heads()

    def _create_task_heads(self) -> None:
        """
        Build one head per entry in ``task_configs``.

        Each entry names a task type and carries that head's keyword
        arguments. ``hidden_dim`` defaults to ``shared_backbone_dim`` and
        ``use_attention`` is forced to ``use_task_specific_attention``, so a
        per-task ``use_attention`` in the caller dict is overwritten.

        Five task types are accepted. Any other raises.

        :return: None.
        :rtype: None
        :raises ValueError: If a config names a task type with no head.
        """

        self.task_heads = {}

        for task_name, config in self.task_configs.items():
            # Copy before pop: do NOT mutate the caller's config dict (the
            # entries of self.task_configs are the caller's objects). Mutating
            # them stripped 'task_type' as a side-effect and broke get_config()
            # round-trips and repeated construction. See decisions.md / SC6.
            config = dict(config)
            task_type = config.pop('task_type')

            # Add shared configuration
            config['hidden_dim'] = config.get('hidden_dim', self.shared_backbone_dim)
            config['use_attention'] = self.use_task_specific_attention

            # Create appropriate head
            if task_type == VisionTaskType.DETECTION:
                self.task_heads[task_name] = DetectionHead(**config)
            elif task_type == VisionTaskType.SEGMENTATION:
                self.task_heads[task_name] = SegmentationHead(**config)
            elif task_type == VisionTaskType.DEPTH_ESTIMATION:
                self.task_heads[task_name] = DepthEstimationHead(**config)
            elif task_type == VisionTaskType.CLASSIFICATION:
                self.task_heads[task_name] = ClassificationHead(**config)
            elif task_type == VisionTaskType.INSTANCE_SEGMENTATION:
                self.task_heads[task_name] = InstanceSegmentationHead(**config)
            else:
                raise ValueError(f"Unsupported task type: {task_type}")

    def build(
            self,
            input_shape: Union[Tuple[Optional[int], ...], Dict[str, Tuple[Optional[int], ...]]]
    ) -> None:
        """
        Build every task head on its own input shape.

        A dict ``input_shape`` gives per-task shapes, falling back to its
        ``'shared'`` entry. A single shape is used for every task.

        :param input_shape: One shared feature-map shape, or a dict of
            per-task shapes.
        :type input_shape: Union[Tuple[Optional[int], ...], Dict[str, Tuple[Optional[int], ...]]]
        :return: None.
        :rtype: None
        """
        for task_name, task_head in self.task_heads.items():
            if isinstance(input_shape, dict):
                task_input_shape = input_shape.get(task_name, input_shape.get('shared'))
            else:
                task_input_shape = input_shape
            task_head.build(task_input_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
            training: Optional[bool] = None
    ) -> Dict[str, Dict[str, keras.KerasTensor]]:
        """
        Run every task head and collect the outputs.

        A dict input is read per task name, falling back to its ``'shared'``
        entry. A single tensor is sent to every head unchanged.

        :param inputs: One feature map, or a dict of per-task feature maps.
        :type inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]
        :param training: Keras training flag, forwarded to each head.
        :type training: Optional[bool]
        :return: Dict mapping each task name to that head's output dict.
        :rtype: Dict[str, Dict[str, keras.KerasTensor]]
        """

        outputs = {}

        # Handle different input formats
        if isinstance(inputs, dict):
            # Task-specific inputs
            for task_name, task_head in self.task_heads.items():
                task_input = inputs.get(task_name, inputs.get('shared'))
                outputs[task_name] = task_head(task_input, training=training)
        else:
            # Shared input for all tasks
            for task_name, task_head in self.task_heads.items():
                outputs[task_name] = task_head(inputs, training=training)

        return outputs

    def compute_output_shape(
            self,
            input_shape: Union[Tuple[Optional[int], ...], Dict[str, Tuple[Optional[int], ...]]]
    ) -> Dict[str, Any]:
        """
        Report each task head's output shape.

        Delegates to every head's own ``compute_output_shape``. A dict
        ``input_shape`` gives per-task shapes, falling back to its
        ``'shared'`` entry. A single shape is used for every task.

        :param input_shape: One shared feature-map shape, or a dict of
            per-task shapes.
        :type input_shape: Union[Tuple[Optional[int], ...], Dict[str, Tuple[Optional[int], ...]]]
        :return: Dict mapping each task name to that head's output shape.
        :rtype: Dict[str, Any]
        """
        output_shapes = {}
        for task_name, task_head in self.task_heads.items():
            if isinstance(input_shape, dict):
                task_input_shape = input_shape.get(task_name, input_shape.get('shared'))
            else:
                task_input_shape = input_shape
            output_shapes[task_name] = task_head.compute_output_shape(task_input_shape)
        return output_shapes

    def get_config(self) -> Dict[str, Any]:
        """
        Return the constructor arguments for serialization.

        ``task_configs`` is stored as the caller passed it. Nothing here
        mutates it, so a round trip rebuilds the same heads.

        :return: Config dict carrying ``task_configs``,
            ``shared_backbone_dim`` and ``use_task_specific_attention``.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'task_configs': self.task_configs,
            'shared_backbone_dim': self.shared_backbone_dim,
            'use_task_specific_attention': self.use_task_specific_attention
        })
        return config


# ---------------------------------------------------------------------
# Factory Function
# ---------------------------------------------------------------------

def create_vision_head(
        task_type: Union[VisionTaskType, str],
        **kwargs: Any
) -> BaseVisionHead:
    """
    Build the vision head that serves a task type.

    Ten of the 37 ``VisionTaskType`` members reach a head. The other 27 raise
    ``ValueError``. There is no fallback head and never has been: a task with
    no implementation fails loudly rather than returning a plausible wrong
    one.

    Five members get a class of their own. Three reuse another task's class,
    two of them with a different output channel count. Denoising and
    super-resolution route through :func:`create_enhancement_head`.

    **Architecture Overview:**

    .. code-block:: text

        task_type: VisionTaskType, or a str that from_string parses
                          │
                 ┌────────┴────────┐
                 ▼                 ▼
          one of these 10     the other 27
                 │                 │
                 ▼                 ▼
           head instance     ValueError raised

        DETECTION              DetectionHead
        SEGMENTATION           SegmentationHead
        INSTANCE_SEGMENTATION  InstanceSegmentationHead
        CLASSIFICATION         ClassificationHead
        DEPTH_ESTIMATION       DepthEstimationHead
        SURFACE_NORMALS        DepthEstimationHead, 3 channels
        OPTICAL_FLOW           DepthEstimationHead, 2 channels
        KEYPOINT_DETECTION     DetectionHead
        DENOISING              EnhancementHead
        SUPER_RESOLUTION       EnhancementHead, scale_factor 2

    The ``VisionTaskType`` docstring in ``vision/task_types.py`` names all 27
    members that raise. This function is the arbiter if the two disagree.

    :param task_type: Task the head is for, as an enum member or its string
        value.
    :type task_type: Union[VisionTaskType, str]
    :param kwargs: Forwarded to the chosen head class.
    :return: Configured vision head.
    :rtype: BaseVisionHead
    :raises ValueError: If the task type is one of the other 27.
    """

    # Convert string to VisionTaskType if needed
    if isinstance(task_type, str):
        task_type = VisionTaskType.from_string(task_type)

    # Create appropriate head based on task type
    if task_type == VisionTaskType.DETECTION:
        return DetectionHead(**kwargs)

    elif task_type == VisionTaskType.SEGMENTATION:
        return SegmentationHead(**kwargs)

    elif task_type == VisionTaskType.DEPTH_ESTIMATION:
        return DepthEstimationHead(**kwargs)

    elif task_type == VisionTaskType.CLASSIFICATION:
        return ClassificationHead(**kwargs)

    elif task_type == VisionTaskType.INSTANCE_SEGMENTATION:
        return InstanceSegmentationHead(**kwargs)

    elif task_type == VisionTaskType.SURFACE_NORMALS:
        # Surface normals use similar architecture to depth
        return DepthEstimationHead(output_channels=3, **kwargs)

    elif task_type == VisionTaskType.OPTICAL_FLOW:
        # Optical flow predicts 2D motion vectors
        return DepthEstimationHead(output_channels=2, **kwargs)

    elif task_type == VisionTaskType.KEYPOINT_DETECTION:
        # Keypoint detection is similar to detection with different outputs
        return DetectionHead(**kwargs)

    elif task_type in [VisionTaskType.DENOISING, VisionTaskType.SUPER_RESOLUTION]:
        # Image enhancement tasks
        return create_enhancement_head(task_type, **kwargs)

    else:
        raise ValueError(f"Unsupported task type: {task_type}")


def create_enhancement_head(
        task_type: VisionTaskType,
        **kwargs: Any
) -> BaseVisionHead:
    """
    Build an :class:`EnhancementHead` for a restoration task.

    The only thing this function decides is the default scale factor.
    Super-resolution gets ``scale_factor=2`` unless the caller passed one.
    Every other task type is left alone, so the head keeps its own default of
    1 and takes the same-resolution path.

    The class itself lives at module scope. This function does not define it.

    :param task_type: Task the head is for. Only
        ``VisionTaskType.SUPER_RESOLUTION`` changes the outcome.
    :type task_type: VisionTaskType
    :param kwargs: Forwarded to :class:`EnhancementHead`.
    :return: Configured enhancement head.
    :rtype: BaseVisionHead
    """

    if task_type == VisionTaskType.SUPER_RESOLUTION:
        kwargs['scale_factor'] = kwargs.get('scale_factor', 2)

    return EnhancementHead(**kwargs)


def create_multi_task_head(
        task_configuration: Union[TaskConfiguration, List[VisionTaskType], Dict[str, Dict]],
        **kwargs: Any
) -> MultiTaskHead:
    """
    Build a :class:`MultiTaskHead` from any of three configuration formats.

    A :class:`TaskConfiguration` contributes its enabled tasks. A list
    contributes its entries, parsing any string through
    ``VisionTaskType.from_string``. A dict is used as the task-config mapping
    unchanged. In the first two cases each task name is looked up in
    ``kwargs`` for its own extra options.

    ``kwargs`` is also forwarded whole to :class:`MultiTaskHead`.

    :param task_configuration: A :class:`TaskConfiguration`, a list of
        :class:`VisionTaskType` members or strings, or a dict mapping task
        names to config dicts.
    :type task_configuration: Union[TaskConfiguration, List[VisionTaskType], Dict[str, Dict]]
    :param kwargs: Per-task option dicts, keyed by task name, plus any
        argument for :class:`MultiTaskHead`.
    :return: Configured multi-task head.
    :rtype: MultiTaskHead
    :raises ValueError: If ``task_configuration`` is none of the three
        accepted types.
    """

    if isinstance(task_configuration, TaskConfiguration):
        # Convert TaskConfiguration to dict
        task_configs = {}
        for task in task_configuration.get_enabled_tasks():
            task_configs[task.value] = {
                'task_type': task,
                **kwargs.get(task.value, {})
            }

    elif isinstance(task_configuration, list):
        # List of TaskTypes
        task_configs = {}
        for task in task_configuration:
            if isinstance(task, str):
                task = VisionTaskType.from_string(task)
            task_configs[task.value] = {
                'task_type': task,
                **kwargs.get(task.value, {})
            }

    elif isinstance(task_configuration, dict):
        # Already a configuration dict
        task_configs = task_configuration

    else:
        raise ValueError(f"Invalid task_configuration type: {type(task_configuration)}")

    return MultiTaskHead(task_configs=task_configs, **kwargs)


# ---------------------------------------------------------------------
# Configuration Helpers
# ---------------------------------------------------------------------

class HeadConfiguration:
    """
    Three keyword-argument presets for the vision heads.

    This class builds no layer. Each static method returns a dict you pass to
    a head constructor or to :func:`create_vision_head`. The presets differ in
    width, dropout, attention and FFN choice.

    **Architecture Overview:**

    .. code-block:: text

        get_default_config(task_type)
                  │
                  ▼
        base_config, 7 keys shared by every task
                  │
                  ▼
        + the task_specific entry, for 5 of the task types
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
        get_efficient_config          get_high_performance_config
        overrides 6 keys              overrides 8 keys

    Both of the lower two call ``get_default_config`` first, so they inherit
    the task-specific keys and then override the shared ones.

    Example:
        >>> cfg = HeadConfiguration.get_default_config(
        ...     VisionTaskType.CLASSIFICATION)
        >>> cfg['num_classes']
        1000
    """

    @staticmethod
    def get_default_config(task_type: VisionTaskType) -> Dict[str, Any]:
        """
        Get the default preset for a task type.

        Seven shared keys are always returned. Five task types add their own
        keys on top: detection, segmentation, depth estimation,
        classification and instance segmentation. Any other task type gets
        the seven shared keys and nothing more.

        :param task_type: Task the preset is for.
        :type task_type: VisionTaskType
        :return: Keyword arguments for the matching head class.
        :rtype: Dict[str, Any]
        """

        base_config = {
            'hidden_dim': 256,
            'normalization_type': 'layer_norm',
            'activation_type': 'gelu',
            'dropout_rate': 0.1,
            'use_ffn': True,
            'ffn_type': 'mlp',
            'ffn_expansion_factor': 4
        }

        task_specific = {
            VisionTaskType.DETECTION: {
                # 80 classes is the COCO default.
                'num_classes': 80,
                'num_anchors': 9,
                'bbox_dims': 4,
                'use_attention': False
            },
            VisionTaskType.SEGMENTATION: {
                # 21 classes is the VOC default.
                'num_classes': 21,
                'upsampling_factor': 4,
                'use_skip_connections': True,
                'use_attention': True,
                'attention_type': 'cbam'
            },
            VisionTaskType.DEPTH_ESTIMATION: {
                'output_channels': 1,
                'min_depth': 0.1,
                'max_depth': 100.0,
                'use_log_depth': True,
                'use_attention': False
            },
            VisionTaskType.CLASSIFICATION: {
                # 1000 classes is the ImageNet default.
                'num_classes': 1000,
                'use_global_pooling': True,
                'pooling_type': 'avg',
                'use_attention': True,
                'attention_type': 'multi_head'
            },
            VisionTaskType.INSTANCE_SEGMENTATION: {
                'num_classes': 80,
                'num_instances': 100,
                'mask_size': (28, 28),
                'use_attention': True,
                'attention_type': 'cbam'
            }
        }

        config = base_config.copy()
        if task_type in task_specific:
            config.update(task_specific[task_type])

        return config

    @staticmethod
    def get_efficient_config(task_type: VisionTaskType) -> Dict[str, Any]:
        """
        Get the lightweight preset for a task type.

        Starts from :meth:`get_default_config` and overrides six keys: a
        narrower ``hidden_dim``, no dropout, no attention, and a GLU FFN with
        a smaller expansion.

        :param task_type: Task the preset is for.
        :type task_type: VisionTaskType
        :return: Keyword arguments for the matching head class.
        :rtype: Dict[str, Any]
        """

        config = HeadConfiguration.get_default_config(task_type)
        config.update({
            'hidden_dim': 128,
            'dropout_rate': 0.0,
            'use_attention': False,
            'use_ffn': True,
            # 'glu' is the more efficient registered FFN type.
            'ffn_type': 'glu',
            'ffn_expansion_factor': 2
        })
        return config


    @staticmethod
    def get_high_performance_config(task_type: VisionTaskType) -> Dict[str, Any]:
        """
        Get the high-performance preset for a task type.

        Starts from :meth:`get_default_config` and overrides eight keys.
        They give a wider ``hidden_dim``, more dropout, differential attention,
        a SwiGLU FFN with a larger expansion, and a zero-centred RMS norm.

        :param task_type: Task the preset is for.
        :type task_type: VisionTaskType
        :return: Keyword arguments for the matching head class.
        :rtype: Dict[str, Any]
        """

        config = HeadConfiguration.get_default_config(task_type)
        config.update({
            'hidden_dim': 512,
            'dropout_rate': 0.2,
            'use_attention': True,
            # 'differential' is the most capable registered attention type.
            'attention_type': 'differential',
            'use_ffn': True,
            # 'swiglu' is the best performing registered FFN type.
            'ffn_type': 'swiglu',
            'ffn_expansion_factor': 8,
            # A zero-centred RMS norm is the more stable choice here.
            'normalization_type': 'zero_centered_rms_norm'
        })
        return config


# ---------------------------------------------------------------------