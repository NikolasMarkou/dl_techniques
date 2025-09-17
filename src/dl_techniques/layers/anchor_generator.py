"""Generate a multi-scale grid of anchor points for object detection.

This layer serves a critical, albeit simple, role in modern anchor-based
object detection models. It pre-computes and stores the spatial center
coordinates (x, y) of every grid cell across multiple feature map scales.
By embedding these coordinates as non-trainable weights, it provides a highly
efficient mechanism for downstream prediction heads to access this spatial
information without re-computing it on every forward pass.

Architecture and Core Concepts:

The design of this layer is rooted in the principles of modern object
detection architectures that use feature pyramids to detect objects of
varying sizes. A deep neural network backbone produces feature maps at
different levels of spatial resolution. Deeper feature maps are smaller and
have a larger "stride" relative to the input image, making them suitable for
detecting large objects. Conversely, shallower feature maps have a smaller
stride and are better for detecting small objects.

This layer takes the dimensions of the input image and a list of these
strides to perform its core function:

1.  **Grid Generation:** For each stride level, it calculates the
    corresponding feature map's height and width. It then generates a 2D
    grid of points representing the center of each cell in that feature map.
    For example, a 640x640 image with a stride of 8 results in an 80x80
    feature map. This layer generates the 6,400 center points for that grid.

2.  **Coordinate Scaling:** The grid coordinates are scaled back to the
    input image's coordinate space. This means that each anchor point's
    (x, y) value corresponds to a specific location on the original image.

3.  **Static Storage:** The crucial architectural choice is to compute this
    entire multi-level grid only once, during the model's construction phase
    (the `build` method). The resulting tensor of anchor points is stored as a
    non-trainable layer weight. This makes the anchor grid a static, baked-in
    part of the model graph, ensuring zero computational overhead during
    training or inference. The `call` method simply retrieves these stored
    weights and tiles them to match the batch size of the input.

This layer's output provides the fundamental spatial scaffolding upon which
the model's prediction heads operate. The heads learn to predict offsets
from these fixed anchor points to determine the final bounding box locations.

Mathematical Foundation:

The calculation for the center coordinate of a grid cell `(i, j)` on a
feature map with a given `stride` is straightforward:
-   `x_center = (j + 0.5) * stride`
-   `y_center = (i + 0.5) * stride`

Here, `i` and `j` are the row and column indices of the cell in the feature
grid. The `+ 0.5` term is critical for shifting the coordinate from the top-left
corner of the grid cell to its exact center. This process is repeated for
all cells across all specified stride levels, and the results are
concatenated into a single comprehensive tensor of anchor points.

References:

The concept of using a grid and predicting relative to its cells is a
foundational idea in single-shot object detectors. This approach was
popularized by:
-   Redmon, J., et al. (2016). "You Only Look Once: Unified, Real-Time
    Object Detection." (YOLOv1), which introduced the idea of dividing the
    image into a grid and having cells predict object properties.

The use of multiple grids from a feature pyramid to handle objects at
different scales is a direct application of the principles from:
-   Lin, T. Y., et al. (2017). "Feature Pyramid Networks for Object
    Detection." (FPN), which established the feature pyramid as a standard,
    effective component for multi-scale detection.

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
