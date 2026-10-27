"""Instance segmentation head: detection outputs plus per-instance mask logits."""

from typing import Any, Dict, Optional, Tuple

import keras

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ...conv_blocks.conv_block import ConvBlock
from ..validation import positive_int
from dl_techniques.utils.keras_registration import register_dl_technique
from .base import BaseVisionHead
from .detection import DetectionHead


@register_dl_technique("dl_techniques.layers.heads.vision.instance_segmentation")
class InstanceSegmentationHead(BaseVisionHead):
    """
    Boxes, class scores and per-instance masks from one feature map.

    Two branches run on the same common-processed features. One is a full
    :class:`DetectionHead` instance held as a sub-layer, which supplies
    ``'classifications'`` and ``'regressions'``. The other is three
    ``ConvBlock`` stages and a linear 1x1 conv, which supplies
    ``'instance_masks'``.

    Every class-like output here is a logit. ``'instance_masks'`` holds mask
    logits: apply ``keras.ops.sigmoid`` for per-pixel mask probabilities, or
    compile a binary cross-entropy with ``from_logits=True``. The key keeps
    its name; its values are not in ``[0, 1]``.

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
                 │           │ linear           │
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

    ``dropout_rate`` reaches the detection branch (both inner ``ConvBlock``
    s) and the attention/FFN stage when on. The mask branch takes none, as
    the Mask R-CNN mask head has none.

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
    :ivar mask_head: Output ``Conv2D(num_instances)``, linear: it emits the
        mask logits.
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
        :raises ValueError: If ``num_classes`` or ``num_instances`` is not a
            positive integer.
        """
        super().__init__(**kwargs)

        owner = type(self).__name__
        self.num_classes = positive_int(num_classes, 'num_classes', owner)
        self.num_instances = positive_int(
            num_instances, 'num_instances', owner)
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
            use_ffn=False,
            name='detection_head'
        )

        self._create_mask_layers()

    def _create_mask_layers(self) -> None:
        """
        Build the mask branch.

        Three ``ConvBlock`` stages at ``hidden_dim`` channels, then a linear
        1x1 conv emitting ``num_instances`` channels of mask logits.

        :return: None.
        :rtype: None
        """

        # Mask feature extraction. Named explicitly, as in EnhancementHead:
        # an auto-suffixed name makes the weight paths process-dependent.
        #
        # DECISION plan-2026-10-14T062616-81938e81/D-050
        # No dropout_rate here: the Mask R-CNN mask branch has no dropout.
        # dropout_rate is the detection branch's (the inner DetectionHead's
        # ConvBlocks). Do NOT thread it in to make the base knob "reach"
        # this branch. See decisions.md D-050.
        self.mask_conv_blocks = [
            ConvBlock(
                filters=self.hidden_dim,
                kernel_size=3,
                normalization_type=self.normalization_type,
                activation_type=self.activation_type,
                name=f'mask_conv_{i}'
            )
            for i in range(3)
        ]

        # Mask prediction head.
        # DECISION plan-2026-10-14T062616-81938e81/D-003
        # Linear: 'instance_masks' carries mask logits under its old name. Do
        # NOT put activation='sigmoid' back: the sigmoid belongs to the loss
        # (from_logits=True) or the caller, as for every class-like output of
        # layers/heads. See decisions.md D-003.
        self.mask_head = keras.layers.Conv2D(
            filters=self.num_instances,
            kernel_size=1,
            name='mask_head'
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the detection sub-head and the mask branch.

        Both consume the shape ``_common_processed_shape`` predicts, not the
        raw input shape. The common attention and FFN are built last, by the
        base class, on the raw input shape.

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

        # The common attention and ffn, on the raw input shape.
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
        :return: Dict with ``'classifications'`` (class logits),
            ``'regressions'`` and ``'instance_masks'`` (mask logits).
        :rtype: Dict[str, keras.KerasTensor]
        """

        x = inputs

        # Apply common processing
        if self.use_attention:
            x = self._apply_attention(x, training=training)
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
