"""
This module provides an `AnchorGenerator` layer, a fundamental component for modern
anchor-based object detection models like YOLO.

In many object detection architectures, instead of directly predicting bounding box
coordinates, the model predicts offsets relative to a predefined set of "anchor" points.
This layer is responsible for generating this set of anchor points. It does not learn
any parameters from data; instead, it serves as a pre-processing and data-provisioning
utility that is seamlessly integrated into the Keras model graph.

Core Concepts and Functionality:

1.  **Anchor Points (Not Anchor Boxes):**
    -   It's important to distinguish that this layer generates *anchor points*, not
        *anchor boxes*. An anchor point represents the center `(x, y)` coordinate of a
        grid cell in a feature map. The model's prediction heads will later use these
        points as a reference to predict the full bounding box (center, width, height).
    -   This is a key characteristic of anchor-free or anchor-point-based methods,
        which simplifies the design by removing the need for pre-defined box sizes/ratios.

2.  **Multi-Level Generation:**
    -   Modern object detectors make predictions at multiple feature map levels
        (e.g., from a Feature Pyramid Network) to handle objects of different scales.
        This layer generates anchor points for each of these levels, specified by the
        `strides_config`.
    -   A smaller stride (e.g., 8) corresponds to a high-resolution feature map with
        many small grid cells, ideal for detecting small objects. A larger stride
        (e.g., 32) corresponds to a low-resolution map with large cells for detecting
        large objects.

3.  **Graph-Safe Data Provisioning:**
    -   The set of anchor points and their corresponding strides are constant for a
        given input image size. Computing them on-the-fly for every forward pass
        would be inefficient.
    -   This layer solves this by pre-computing all anchor points and strides during
        its `build` phase and storing them as **non-trainable weights** (buffers).
    -   This makes the anchor data a persistent part of the model's computational
        graph, ensuring it's available and graph-safe for all operations, especially
        for the loss function which needs to match predictions to anchors.

4.  **Batch-Aware Output:**
    -   In the `call` method, the layer takes a dummy input tensor *only* to determine
        the batch size. It then tiles its stored anchor and stride tensors to match
        this batch size, providing a correctly shaped output for batch processing.
    -   The content of the input tensor is completely ignored.
"""

import keras
from keras import ops
from typing import Tuple, Any, Dict, List, Optional

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..utils.logger import logger

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class AnchorGenerator(keras.layers.Layer):
    """Anchor generator layer for YOLOv12 object detection.

    This layer generates and holds YOLOv12 anchor points and strides for different
    feature map levels. The anchor grid is computed once and stored as non-trainable
    weights (buffers), providing a graph-safe way to handle persistent tensors
    needed by the loss function.

    The layer generates anchor points for multiple stride levels (8, 16, 32) based
    on the input image dimensions and creates corresponding stride tensors. Each
    anchor point represents the center of a grid cell in the corresponding feature map.

    Args:
        input_image_shape: Tuple of integers (height, width) representing the shape
            of input images used to determine grid sizes.
        strides_config: List of integers specifying the stride values for different
            feature map levels. Defaults to [8, 16, 32] for typical YOLOv12 configuration.
        **kwargs: Additional keyword arguments passed to the Layer base class.

    Input shape:
        Any tensor with shape `(batch_size, ...)` - the input is used only to determine
        batch size for output tiling. The actual content is ignored.

    Output shape:
        Tuple of two tensors:

        - anchors: `(batch_size, total_anchor_points, 2)` containing (x, y) coordinates
          of anchor points in the original image space
        - strides: `(batch_size, total_anchor_points, 1)` containing corresponding
          stride values for each anchor point

    Returns:
        Tuple[keras.KerasTensor, keras.KerasTensor]: A tuple containing:

        - anchors: Tensor with anchor point coordinates in image space
        - strides: Tensor with corresponding stride values

    Raises:
        ValueError: If input_image_shape contains non-positive values.
        ValueError: If strides_config contains non-positive values.

    Example:
        >>> # Create anchor generator for 640x640 images
        >>> anchor_gen = AnchorGenerator(input_image_shape=(640, 640))
        >>> dummy_input = tf.random.normal([2, 100, 4])  # batch_size=2
        >>> anchors, strides = anchor_gen(dummy_input)
        >>> print(f"Anchors shape: {anchors.shape}")
        Anchors shape: (2, 8400, 2)
        >>> print(f"Strides shape: {strides.shape}")
        Strides shape: (2, 8400, 1)

        >>> # Custom stride configuration
        >>> custom_anchor_gen = AnchorGenerator(
        ...     input_image_shape=(416, 416),
        ...     strides_config=[8, 16, 32, 64]
        ... )
    """

    def __init__(
            self,
            input_image_shape: Tuple[int, int],
            strides_config: Optional[List[int]] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate input parameters
        if len(input_image_shape) != 2 or any(dim <= 0 for dim in input_image_shape):
            raise ValueError(
                f"input_image_shape must be a tuple of two positive integers, "
                f"got {input_image_shape}"
            )

        # Store configuration parameters
        self.input_image_shape = input_image_shape
        self.strides_config = strides_config or [8, 16, 32]

        # Validate strides configuration
        if any(stride <= 0 for stride in self.strides_config):
            raise ValueError(
                f"All strides must be positive integers, got {self.strides_config}"
            )

        # Initialize weights to None - will be created in build()
        self.anchors = None
        self.strides = None
        self._build_input_shape = None

        logger.debug(
            f"AnchorGenerator initialized with image shape {input_image_shape} "
            f"and strides {self.strides_config}"
        )

    def _make_anchors(self) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Generate anchor points and strides for all feature map levels.

        Creates a grid of anchor points for each stride level, where each point
        represents the center of a grid cell in the feature map. The coordinates
        are computed in the original image space.

        Returns:
            Tuple[keras.KerasTensor, keras.KerasTensor]: A tuple containing:

            - anchor_points: Concatenated anchor coordinates for all levels,
              shape (total_anchors, 2)
            - stride_tensor: Corresponding stride values for each anchor point,
              shape (total_anchors, 1)
        """
        height, width = self.input_image_shape
        anchor_points: List[keras.KerasTensor] = []
        stride_tensors: List[keras.KerasTensor] = []

        for stride in self.strides_config:
            # Calculate feature map dimensions for this stride level
            feat_h, feat_w = height // stride, width // stride

            # Create coordinate grids (add 0.5 to get cell centers)
            # Coordinates are in the original image space
            x_coords = (ops.arange(feat_w, dtype="float32") + 0.5) * stride
            y_coords = (ops.arange(feat_h, dtype="float32") + 0.5) * stride

            # Create meshgrid and stack coordinates
            y_grid, x_grid = ops.meshgrid(y_coords, x_coords, indexing="ij")
            xy_grid = ops.stack([x_grid, y_grid], axis=-1)
            xy_grid = ops.reshape(xy_grid, (-1, 2))

            # Store anchor points and corresponding strides
            anchor_points.append(xy_grid)
            stride_tensors.append(
                ops.full((feat_h * feat_w, 1), float(stride), dtype="float32")
            )

            logger.debug(
                f"Generated {feat_h * feat_w} anchors for stride {stride} "
                f"(feature map: {feat_h}x{feat_w})"
            )

        # Concatenate all levels
        all_anchors = ops.concatenate(anchor_points, axis=0)
        all_strides = ops.concatenate(stride_tensors, axis=0)

        logger.debug(f"Total anchors generated: {ops.shape(all_anchors)[0]}")

        return all_anchors, all_strides

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer by creating anchor and stride weights.

        Creates non-trainable weights (buffers) to store the pre-computed
        anchor points and stride values. This ensures the anchors are computed
        once and stored in the computational graph.

        Args:
            input_shape: Shape of the input tensor, used for serialization.
        """
        # Store input shape for serialization
        self._build_input_shape = input_shape

        # Generate anchors and strides
        anchors, strides = self._make_anchors()

        # Create non-trainable weights (buffers) to store anchors and strides
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

        logger.info(
            f"AnchorGenerator built with {self.total_anchor_points} anchor points"
        )

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Forward pass returning tiled anchors and strides.

        The input tensor is used only to determine the batch size for proper tiling
        of the anchor and stride tensors. The actual content of the input is ignored.

        Args:
            inputs: Input tensor with any shape containing batch dimension at index 0.
            training: Boolean indicating training mode. Unused but included
                for consistency with Keras Layer interface.

        Returns:
            Tuple[keras.KerasTensor, keras.KerasTensor]: Tuple containing:

            - anchors: Tiled anchor coordinates with shape
              `(batch_size, num_anchors, 2)`
            - strides: Tiled stride values with shape
              `(batch_size, num_anchors, 1)`
        """
        # Get batch size from input
        batch_size = ops.shape(inputs)[0]

        # Tile anchors and strides to match batch size
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

        Args:
            input_shape: Shape of the input tensor.

        Returns:
            Tuple containing the shapes of anchors and strides tensors.
            Each shape is (batch_size, total_anchor_points, features).
        """
        # Calculate total number of anchor points
        total_anchors = self.total_anchor_points

        # Convert input_shape to list for manipulation, then back to tuple
        input_shape_list = list(input_shape)
        batch_size = input_shape_list[0] if input_shape_list[0] is not None else None

        anchors_shape = (batch_size, total_anchors, 2)
        strides_shape = (batch_size, total_anchors, 1)

        return anchors_shape, strides_shape

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration parameters.
        """
        config = super().get_config()
        config.update({
            "input_image_shape": self.input_image_shape,
            "strides_config": self.strides_config,
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get build configuration for serialization.

        Returns:
            Dictionary containing the build configuration needed to
            reconstruct the layer after loading.
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build layer from configuration.

        Reconstructs the layer's built state from the provided configuration.
        This is called during model loading to properly rebuild the layer.

        Args:
            config: Dictionary containing build configuration from get_build_config().
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

    @property
    def total_anchor_points(self) -> int:
        """Get the total number of anchor points across all stride levels.

        This property calculates the total number of anchor points that will be
        generated across all feature map levels based on the configured strides
        and input image shape.

        Returns:
            Total number of anchor points as an integer.
        """
        total = 0
        height, width = self.input_image_shape

        for stride in self.strides_config:
            feat_h, feat_w = height // stride, width // stride
            total += feat_h * feat_w

        return total

# ---------------------------------------------------------------------
