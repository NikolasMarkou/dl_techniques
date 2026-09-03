"""
Anchor point grid generator, built by the ``AnchorGenerator`` class.

An anchor-based detection head needs a fixed spatial grid of candidate
center points for every scale it predicts at, recomputing that grid on
every forward pass otherwise wastes work. This layer computes the grid once
at build time, for every stride in a feature pyramid, and stores it as a
non-trainable weight; each forward pass only tiles the stored grid to the
batch size. Grid cell centers are placed at ``x = (j + 0.5) * stride``,
``y = (i + 0.5) * stride``, so a 640x640 image at stride 8 gives an 80x80
grid of centers on the original image's coordinate space.

References:
    - Redmon et al., 2016. You Only Look Once: Unified, Real-Time Object
      Detection.
    - Lin et al., 2017. Feature Pyramid Networks for Object Detection.
"""

import keras
from keras import ops
from typing import Tuple, Any, Dict, List, Optional

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.anchor_generator")
class AnchorGenerator(keras.layers.Layer):
    """Anchor generator layer for multi-scale object detection.

    Pre-computes and stores anchor point coordinates and stride values for
    multiple feature map levels. For each stride, a 2D grid of center
    coordinates is generated via ``x = (j + 0.5) * stride``,
    ``y = (i + 0.5) * stride``, concatenated across all scales, and stored as
    non-trainable weights. The ``call`` method tiles these static anchors to
    match the input batch size, providing zero-cost spatial scaffolding for
    detection heads.

    Architecture:

    .. code-block:: text

        ┌─────────────────────────────┐
        │       Input (any tensor)    │
        └──────────────┬──────────────┘
                       │ batch_size
                       ▼
        ┌─────────────────────────────┐
        │  Stored Anchors & Strides   │
        │  (non-trainable weights)    │
        │                             │
        │  stride=8  ──► 80x80 grid   │
        │  stride=16 ──► 40x40 grid   │
        │  stride=32 ──► 20x20 grid   │
        │        concat all grids     │
        └──────────────┬──────────────┘
                       │ tile to batch
                       ▼
        ┌─────────────────────────────┐
        │  anchors (B, N, 2)          │
        │  strides (B, N, 1)          │
        └─────────────────────────────┘

    :param input_image_shape: Tuple of two positive integers ``(height, width)``
        representing the input image dimensions used to calculate grid sizes.
    :type input_image_shape: Tuple[int, int]
    :param strides_config: List of positive integers specifying stride values
        for different feature map levels. Defaults to ``[8, 16, 32]``.
    :type strides_config: Optional[List[int]]
    :param kwargs: Additional keyword arguments passed to the Layer base class.
    """

    def __init__(
        self,
        input_image_shape: Tuple[int, int],
        strides_config: Optional[List[int]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if (len(input_image_shape) != 2 or
            any(dim <= 0 for dim in input_image_shape)):
            raise ValueError(
                f"input_image_shape must be a tuple of two positive integers, "
                f"got {input_image_shape}"
            )

        self.input_image_shape = input_image_shape
        self.strides_config = strides_config or [8, 16, 32]

        if any(stride <= 0 for stride in self.strides_config):
            raise ValueError(
                f"All strides must be positive integers, got {self.strides_config}"
            )

        # Created in build().
        self.anchors = None
        self.strides = None

    def _make_anchors(self) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Generate anchor points and strides for all feature map levels.

        :return: Tuple containing concatenated anchor coordinates and stride values.
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor]
        """
        height, width = self.input_image_shape
        anchor_points: List[keras.KerasTensor] = []
        stride_tensors: List[keras.KerasTensor] = []

        for stride in self.strides_config:
            feat_h, feat_w = height // stride, width // stride

            # +0.5 shifts from the cell's top-left corner to its center.
            x_coords = (ops.arange(feat_w, dtype="float32") + 0.5) * stride
            y_coords = (ops.arange(feat_h, dtype="float32") + 0.5) * stride

            y_grid, x_grid = ops.meshgrid(y_coords, x_coords, indexing="ij")
            xy_grid = ops.stack([x_grid, y_grid], axis=-1)
            xy_grid = ops.reshape(xy_grid, (-1, 2))

            anchor_points.append(xy_grid)
            stride_tensors.append(
                ops.full((feat_h * feat_w, 1), float(stride), dtype="float32")
            )

        return (ops.concatenate(anchor_points, axis=0),
                ops.concatenate(stride_tensors, axis=0))

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the layer's anchor and stride weights.

        :param input_shape: Shape tuple indicating the input shape.
        :type input_shape: Tuple[Optional[int], ...]
        """
        anchors, strides = self._make_anchors()

        self.anchors = self.add_weight(
            name="anchors",
            shape=ops.shape(anchors),
            initializer=keras.initializers.Constant(anchors),
            trainable=False,
            dtype="float32"
        )

        self.strides = self.add_weight(
            name="strides",
            shape=ops.shape(strides),
            initializer=keras.initializers.Constant(strides),
            trainable=False,
            dtype="float32"
        )

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Return batch-tiled anchors and strides.

        :param inputs: Input tensor used only for batch size extraction.
        :type inputs: keras.KerasTensor
        :param training: Training mode flag (unused).
        :type training: Optional[bool]
        :return: Tuple of ``(anchors, strides)`` tensors tiled to match batch size.
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor]
        """
        batch_size = ops.shape(inputs)[0]

        tiled_anchors = ops.tile(
            ops.expand_dims(self.anchors, axis=0),
            [batch_size, 1, 1]
        )
        tiled_strides = ops.tile(
            ops.expand_dims(self.strides, axis=0),
            [batch_size, 1, 1]
        )

        return tiled_anchors, tiled_strides

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Tuple[Optional[int], int, int], Tuple[Optional[int], int, int]]:
        """Compute output shapes for anchors and strides tensors.

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Tuple of shapes for anchors and strides.
        :rtype: Tuple[Tuple[Optional[int], int, int], Tuple[Optional[int], int, int]]
        """
        batch_size = input_shape[0]
        total_anchors = self.total_anchor_points

        anchors_shape = (batch_size, total_anchors, 2)
        strides_shape = (batch_size, total_anchors, 1)

        return anchors_shape, strides_shape

    @property
    def total_anchor_points(self) -> int:
        """Calculate total number of anchor points across all stride levels.

        :return: Total anchor count.
        :rtype: int
        """
        total = 0
        height, width = self.input_image_shape

        for stride in self.strides_config:
            feat_h, feat_w = height // stride, width // stride
            total += feat_h * feat_w

        return total

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'input_image_shape': self.input_image_shape,
            'strides_config': self.strides_config,
        })
        return config

# ---------------------------------------------------------------------
