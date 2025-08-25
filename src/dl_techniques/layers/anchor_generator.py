"""
This module provides an `AnchorGenerator` layer, a fundamental component for modern
anchor-based object detection models like YOLOv12.

The layer generates anchor points (center coordinates) for multiple feature map levels,
storing them as non-trainable weights in the computational graph for efficient access
during training and inference.
"""

import keras
from keras import ops
from typing import Tuple, Any, Dict, List, Optional

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class AnchorGenerator(keras.layers.Layer):
    """Anchor generator layer for YOLOv12 object detection.

    This layer generates and stores anchor points and strides for different feature map
    levels. Anchor points represent grid cell centers in feature maps at various scales.
    The anchor grid is computed once during build() and stored as non-trainable weights.

    Args:
        input_image_shape: Tuple of integers (height, width) representing input image
            dimensions used to calculate grid sizes. Must contain positive integers.
        strides_config: List of integers specifying stride values for different
            feature map levels. Each stride must be positive. Defaults to [8, 16, 32].
        **kwargs: Additional keyword arguments passed to Layer base class.

    Input shape:
        Any tensor with shape `(batch_size, ...)` - content is ignored, only batch
        size is used for output tiling.

    Output shape:
        Tuple of two tensors:
        - anchors: `(batch_size, total_anchor_points, 2)` - (x, y) coordinates
        - strides: `(batch_size, total_anchor_points, 1)` - corresponding strides

    Example:
        ```python
        # Basic usage for 640x640 images
        anchor_gen = AnchorGenerator(input_image_shape=(640, 640))
        dummy_input = keras.random.normal([2, 100, 4])  # batch_size=2
        anchors, strides = anchor_gen(dummy_input)
        print(f"Anchors shape: {anchors.shape}")  # (2, 8400, 2)

        # Custom configuration
        custom_gen = AnchorGenerator(
            input_image_shape=(416, 416),
            strides_config=[8, 16, 32, 64]
        )
        ```

    Raises:
        ValueError: If input_image_shape contains non-positive values.
        ValueError: If strides_config contains non-positive values.
    """

    def __init__(
        self,
        input_image_shape: Tuple[int, int],
        strides_config: Optional[List[int]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate input parameters
        if (len(input_image_shape) != 2 or
            any(dim <= 0 for dim in input_image_shape)):
            raise ValueError(
                f"input_image_shape must be a tuple of two positive integers, "
                f"got {input_image_shape}"
            )

        # Store ALL configuration parameters
        self.input_image_shape = input_image_shape
        self.strides_config = strides_config or [8, 16, 32]

        # Validate strides configuration
        if any(stride <= 0 for stride in self.strides_config):
            raise ValueError(
                f"All strides must be positive integers, got {self.strides_config}"
            )

        # Initialize weight attributes - created in build()
        self.anchors = None
        self.strides = None

    def _make_anchors(self) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Generate anchor points and strides for all feature map levels.

        Returns:
            Tuple containing concatenated anchor coordinates and stride values.
        """
        height, width = self.input_image_shape
        anchor_points: List[keras.KerasTensor] = []
        stride_tensors: List[keras.KerasTensor] = []

        for stride in self.strides_config:
            # Calculate feature map dimensions
            feat_h, feat_w = height // stride, width // stride

            # Create coordinate grids (add 0.5 for cell centers)
            x_coords = (ops.arange(feat_w, dtype="float32") + 0.5) * stride
            y_coords = (ops.arange(feat_h, dtype="float32") + 0.5) * stride

            # Create meshgrid and flatten
            y_grid, x_grid = ops.meshgrid(y_coords, x_coords, indexing="ij")
            xy_grid = ops.stack([x_grid, y_grid], axis=-1)
            xy_grid = ops.reshape(xy_grid, (-1, 2))

            # Store points and strides
            anchor_points.append(xy_grid)
            stride_tensors.append(
                ops.full((feat_h * feat_w, 1), float(stride), dtype="float32")
            )

        return (ops.concatenate(anchor_points, axis=0),
                ops.concatenate(stride_tensors, axis=0))

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the layer's anchor and stride weights.

        Computes anchor points once and stores as non-trainable weights for
        efficient graph-safe access during training and inference.
        """
        # Generate anchors and strides
        anchors, strides = self._make_anchors()

        # Create layer's own weights (non-trainable buffers)
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
        """Forward pass returning batch-tiled anchors and strides.

        Args:
            inputs: Input tensor used only for batch size extraction.
            training: Training mode flag, unused but kept for interface consistency.

        Returns:
            Tuple of (anchors, strides) tensors tiled to match batch size.
        """
        batch_size = ops.shape(inputs)[0]

        # Tile to match batch size
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
        """Compute output shapes for anchors and strides tensors."""
        batch_size = input_shape[0]
        total_anchors = self.total_anchor_points

        anchors_shape = (batch_size, total_anchors, 2)
        strides_shape = (batch_size, total_anchors, 1)

        return anchors_shape, strides_shape

    @property
    def total_anchor_points(self) -> int:
        """Calculate total number of anchor points across all stride levels."""
        total = 0
        height, width = self.input_image_shape

        for stride in self.strides_config:
            feat_h, feat_w = height // stride, width // stride
            total += feat_h * feat_w

        return total

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'input_image_shape': self.input_image_shape,
            'strides_config': self.strides_config,
        })
        return config

# ---------------------------------------------------------------------
