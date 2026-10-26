"""Detection head: class scores and box offsets, per anchor."""

from typing import Any, Dict, List, Literal, Optional, Tuple

import keras

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ...conv_blocks.conv_block import ConvBlock
from dl_techniques.utils.tensors import is_single_shape
from ..validation import positive_int
from dl_techniques.utils.keras_registration import register_dl_technique
from .base import BaseVisionHead
from .utils import _DEFAULT_BBOX_DIMS, _refuse_dropout_no_layer_applies


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

        inputs (B, H, W, C)  or  (B, N, D) with input_format="sequence"
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
    detection, defaulting ``num_anchors=1`` and ``use_regression=False``: one
    keypoint score vector per location, as SuperPoint's detector head.

    Input shape:
        ``(batch, height, width, channels)`` for spatial format, or
        ``(batch, seq_len, dim)`` for sequence format (ViT patch tokens).

    Output shape:
        ``{'classifications': (batch, height, width, num_anchors *
        num_classes), 'regressions': (batch, height, width, num_anchors *
        bbox_dims)}``.

    :param num_classes: Number of object classes.
    :type num_classes: int
    :param num_anchors: Number of anchor boxes per location.
    :type num_anchors: int
    :param bbox_dims: Values per box, read by the regression branch only.
        ``None`` (default) means 4, the usual corner or centre form, with
        ``use_regression=True``; with ``use_regression=False`` it must stay
        ``None`` and is stored as ``None``.
    :type bbox_dims: Optional[int]
    :param input_format: Input format, ``"spatial"`` for (B,H,W,C) or
        ``"sequence"`` for (B,N,D) ViT patch tokens. Defaults to ``"spatial"``.
    :type input_format: Literal["spatial", "sequence"]
    :param patch_grid_size: Tuple ``(H, W)`` of patch grid dimensions for
        ``input_format="sequence"``. Required when using sequence format.
    :type patch_grid_size: Optional[Tuple[int, int]]
    :param kwargs: Arguments for :class:`BaseVisionHead`.

    :ivar cls_conv: Classification-branch ``ConvBlock``.
    :vartype cls_conv: ConvBlock
    :ivar cls_head: Classification output conv.
    :vartype cls_head: keras.layers.Conv2D
    :ivar reg_conv: Regression-branch ``ConvBlock`` (if ``use_regression=True``).
    :vartype reg_conv: Optional[ConvBlock]
    :ivar reg_head: Regression output conv (if ``use_regression=True``).
    :vartype reg_head: Optional[keras.layers.Conv2D]
    """

    def __init__(
            self,
            num_classes: int,
            num_anchors: int = 9,
            bbox_dims: Optional[int] = None,
            input_format: Literal["spatial", "sequence"] = "spatial",
            patch_grid_size: Optional[Tuple[int, int]] = None,
            use_regression: bool = True,
            **kwargs: Any
    ) -> None:
        """
        Store the configuration, then build the two branches.

        :param num_classes: Number of object classes.
        :type num_classes: int
        :param num_anchors: Number of anchor boxes per location.
        :type num_anchors: int
        :param bbox_dims: Values per box (regression only; ``None`` means 4).
        :type bbox_dims: Optional[int]
        :param input_format: Input format, ``"spatial"`` or ``"sequence"``.
        :type input_format: Literal["spatial", "sequence"]
        :param patch_grid_size: Patch grid (H, W) for sequence input format.
        :type patch_grid_size: Optional[Tuple[int, int]]
        :param use_regression: Whether to create the regression branch.
            For keypoint detection and other tasks that only need classification,
            set to ``False`` to avoid dead weights. Defaults to ``True``.
        :type use_regression: bool
        :param kwargs: Arguments for :class:`BaseVisionHead`.
        :return: None.
        :rtype: None
        :raises ValueError: If ``num_classes``, ``num_anchors`` or
            ``bbox_dims`` is not a positive integer, or ``bbox_dims`` is given
            with ``use_regression=False``.
        """
        super().__init__(**kwargs)

        owner = type(self).__name__
        self.num_classes = positive_int(num_classes, 'num_classes', owner)
        self.num_anchors = positive_int(num_anchors, 'num_anchors', owner)
        # DECISION plan-2026-10-14T062616-81938e81/D-074: D-030's rule, a value
        # no layer reads is refused and stored as None. Do NOT default
        # bbox_dims to 4 again for a head without a regression branch: it was
        # stored, written to the config and never read, and any value passed
        # in silence (SuperPoint passed 4 "not used"). The same rule covers
        # hidden_dim in the explicit-segmentation mode
        # (_drop_unread_hidden_dim). See decisions.md D-074.
        if use_regression:
            self.bbox_dims = positive_int(
                _DEFAULT_BBOX_DIMS if bbox_dims is None else bbox_dims,
                'bbox_dims', owner)
        elif bbox_dims is not None:
            raise ValueError(
                f"{owner}(use_regression=False) builds no regression branch, "
                f"so no layer reads bbox_dims; got bbox_dims={bbox_dims!r}."
            )
        else:
            self.bbox_dims = None
        self.input_format = input_format
        self.patch_grid_size = patch_grid_size
        self.use_regression = use_regression

        if self.input_format == "sequence" and self.patch_grid_size is None:
            raise ValueError("patch_grid_size is required when input_format='sequence'")

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
            dropout_rate=self.dropout_rate,
            name="cls_conv"
        )

        self.cls_head = keras.layers.Conv2D(
            filters=self.num_anchors * self.num_classes,
            kernel_size=1,
            padding='same',
            name='cls_head'
        )

        # Regression branch (optional)
        if self.use_regression:
            self.reg_conv = ConvBlock(
                filters=self.hidden_dim,
                kernel_size=3,
                normalization_type=self.normalization_type,
                activation_type=self.activation_type,
                dropout_rate=self.dropout_rate,
                name="reg_conv"
            )

            self.reg_head = keras.layers.Conv2D(
                filters=self.num_anchors * self.bbox_dims,
                kernel_size=1,
                padding='same',
                name='reg_head'
            )
        else:
            self.reg_conv = None
            self.reg_head = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build both branches on the common-processed shape.

        The FFN is built on the input shape and the attention on the layout
        its type takes (:meth:`_attention_input_shape`); for sequence input
        both run before the reshape, and the conv branches are built on the
        patch grid after it.

        :param input_shape: Input feature-map shape (spatial) or sequence shape.
        :type input_shape: Tuple[Optional[int], ...]
        :return: None.
        :rtype: None
        """
        # For sequence input, build attention/ffn on sequence shape,
        # and conv branches on spatial shape
        if self.use_attention:
            self.attention.build(self._attention_input_shape(input_shape))
        if self.use_ffn:
            self.ffn.build(input_shape)
        feature_shape = self._common_processed_shape(input_shape)
        if self.input_format == "sequence":
            if self.patch_grid_size is None:
                raise ValueError("patch_grid_size must be set for sequence input format")
            # The conv branches run on the patch grid, after the reshape.
            h, w = self.patch_grid_size
            feature_shape = (feature_shape[0], h, w, feature_shape[-1])

        self.cls_conv.build(feature_shape)
        self.cls_head.build(self.cls_conv.compute_output_shape(feature_shape))

        if self.use_regression:
            self.reg_conv.build(feature_shape)
            self.reg_head.build(self.reg_conv.compute_output_shape(feature_shape))

        # attention and ffn are built above, on the sequence or spatial shape
        # call() applies them to, so only the Layer bookkeeping remains.
        keras.layers.Layer.build(self, input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Dict[str, keras.KerasTensor]:
        """
        Run the common stages, then both branches.

        :param inputs: Feature map of shape ``(batch, height, width,
            channels)`` for spatial format, or ``(batch, seq_len, dim)``
            for sequence format (ViT patch tokens).
        :type inputs: keras.KerasTensor
        :param training: Keras training flag, forwarded to every sub-layer.
        :type training: Optional[bool]
        :return: Dict with ``'classifications'`` and ``'regressions'``.
        :rtype: Dict[str, keras.KerasTensor]
        """

        x = inputs

        # The common stages run on the layout the head holds (the attention
        # on the layout its type takes, _apply_attention); a sequence is
        # reshaped onto the patch grid only for the conv branches.
        if self.use_attention:
            x = self._apply_attention(x, training=training)
        if self.use_ffn:
            x = self.ffn(x, training=training)
        if self.input_format == "sequence":
            if self.patch_grid_size is None:
                raise ValueError("patch_grid_size must be set for sequence input format")
            batch_size = keras.ops.shape(x)[0]
            h, w = self.patch_grid_size
            x = keras.ops.reshape(x, (batch_size, h, w, -1))

        # Classification branch
        cls_features = self.cls_conv(x, training=training)
        cls_output = self.cls_head(cls_features)

        # Regression branch (optional)
        if self.use_regression:
            reg_features = self.reg_conv(x, training=training)
            reg_output = self.reg_head(reg_features)
        else:
            reg_output = None

        if self.use_regression:
            return {
                'classifications': cls_output,
                'regressions': reg_output
            }
        else:
            return {
                'classifications': cls_output
            }

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Dict[str, Tuple[Optional[int], ...]]:
        """
        Report the output shapes.

        Spatial dimensions are preserved. Only the channel count changes.

        :param input_shape: Shape ``(batch, height, width, channels)`` for spatial,
            or ``(batch, seq_len, dim)`` for sequence format.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Dict with ``'classifications'`` and optionally ``'regressions'``
            shapes.
        :rtype: Dict[str, Tuple[Optional[int], ...]]
        """
        if self.input_format == "sequence":
            batch = input_shape[0]
            h, w = self.patch_grid_size
        else:
            batch, height, width = input_shape[0], input_shape[1], input_shape[2]
            h, w = height, width
        output_shapes = {
            'classifications': (batch, h, w, self.num_anchors * self.num_classes)
        }
        if self.use_regression:
            output_shapes['regressions'] = (batch, h, w, self.num_anchors * self.bbox_dims)
        return output_shapes

    def get_config(self) -> Dict[str, Any]:
        """
        Return the constructor arguments for serialization.

        :return: Config dict carrying ``num_classes``, ``num_anchors``,
            ``bbox_dims``, ``input_format``, ``patch_grid_size`` and ``use_regression``,
            on top of the base configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'num_anchors': self.num_anchors,
            'bbox_dims': self.bbox_dims,
            'input_format': self.input_format,
            'patch_grid_size': self.patch_grid_size,
            'use_regression': self.use_regression
        })
        return config
