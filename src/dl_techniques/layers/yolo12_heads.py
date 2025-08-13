"""
YOLOv12 Task-Specific Heads for Multi-Task Learning.

This module implements specialized heads for the YOLOv12 architecture, enabling
multitask learning capabilities including object detection, semantic segmentation,
and image classification. Each head is designed to process shared backbone features
from YOLOv12 while maintaining task-specific architectures optimized for their
respective objectives.

Architecture Overview:
    - **YOLOv12DetectionHead**: Multi-scale object detection with separate bbox
      regression and classification branches using depthwise separable convolutions
    - **YOLOv12SegmentationHead**: Progressive upsampling decoder with skip connections
      for pixel-level segmentation tasks
    - **YOLOv12ClassificationHead**: Multi-scale global pooling with attention
      mechanisms for image-level classification

Usage Example:
    ```python
    # Create detection head
    detection_head = YOLOv12DetectionHead(
        num_classes=1,
        reg_max=16,
        bbox_channels=[32, 64, 96],  # Channels for each scale
        cls_channels=[64, 96, 128]   # Channels for each scale
    )

    # Create segmentation head
    seg_head = YOLOv12SegmentationHead(
        num_classes=1,
        intermediate_filters=[128, 64, 32]
    )

    # Create classification head
    cls_head = YOLOv12ClassificationHead(
        num_classes=1,
        hidden_dims=[512, 256]
    )

    # Use with multiscale features
    features = [p3, p4, p5]  # From YOLOv12 backbone

    detections = detection_head(features)
    segmentation = seg_head(features)
    classification = cls_head(features)
    ```

Notes:
    - All heads expect exactly 3 input feature maps from different scales
    - Input shapes should be [(B, H/8, W/8, C1), (B, H/16, W/16, C2), (B, H/32, W/32, C3)]
"""

import keras
from keras import ops
from typing import Optional, Tuple, Dict, Any, List, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------
from ..utils.logger import logger

from .yolo12 import ConvBlock
from .squeeze_excitation import SqueezeExcitation

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class YOLOv12DetectionHead(keras.layers.Layer):
    """YOLOv12 Detection Head with separate classification and regression branches.

    This head processes multi-scale feature maps from the backbone/neck to produce
    object detection predictions including bounding boxes and class probabilities.
    Uses depthwise separable convolutions for efficiency.

    Following modern Keras 3 patterns: creates sub-layers in __init__() and builds
    them explicitly in build() for robust serialization.

    Args:
        num_classes: Integer, number of object classes to detect. Must be positive. Defaults to 80.
        reg_max: Integer, maximum value for DFL (Distribution Focal Loss) regression.
            Must be positive. Defaults to 16.
        bbox_channels: Optional list of integers specifying bbox branch channels for each scale.
            If None, automatically calculated as max(16, reg_max * 4). Length must be 3.
        cls_channels: Optional list of integers specifying classification branch channels
            for each scale. If None, automatically calculated. Length must be 3.
        kernel_initializer: String or Initializer, initializer for kernel weights.
            Defaults to 'he_normal'.
        kernel_regularizer: Optional Regularizer, regularizer for kernel weights. Defaults to None.
        name: Optional string, layer name.
        **kwargs: Additional keyword arguments for Layer base class.

    Input shape:
        List of 3 tensors with shapes:
        - [(batch_size, height/8, width/8, channels_p3),
           (batch_size, height/16, width/16, channels_p4),
           (batch_size, height/32, width/32, channels_p5)]

    Output shape:
        Tensor with shape (batch_size, total_anchors, 4*reg_max + num_classes)
        where total_anchors is the sum of anchors from all three scales.

    Example:
        ```python
        # Basic usage with automatic channel calculation
        head = YOLOv12DetectionHead(num_classes=80)

        # Advanced usage with custom channels
        head = YOLOv12DetectionHead(
            num_classes=80,
            reg_max=16,
            bbox_channels=[32, 64, 96],
            cls_channels=[64, 96, 128],
            kernel_regularizer=keras.regularizers.L2(1e-4)
        )

        features = [p3_features, p4_features, p5_features]
        detections = head(features)
        ```

    Raises:
        ValueError: If num_classes or reg_max is not positive.
        ValueError: If bbox_channels or cls_channels length is not 3.
    """

    def __init__(
            self,
            num_classes: int = 80,
            reg_max: int = 16,
            bbox_channels: Optional[List[int]] = None,
            cls_channels: Optional[List[int]] = None,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            name: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(name=name, **kwargs)

        # Validate inputs
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if reg_max <= 0:
            raise ValueError(f"reg_max must be positive, got {reg_max}")

        # Store configuration
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.bbox_channels = bbox_channels
        self.cls_channels = cls_channels
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # Validate channel lists if provided
        if bbox_channels is not None:
            if len(bbox_channels) != 3:
                raise ValueError(f"bbox_channels must have length 3, got {len(bbox_channels)}")
            if any(c <= 0 for c in bbox_channels):
                raise ValueError("All bbox_channels must be positive")

        if cls_channels is not None:
            if len(cls_channels) != 3:
                raise ValueError(f"cls_channels must have length 3, got {len(cls_channels)}")
            if any(c <= 0 for c in cls_channels):
                raise ValueError("All cls_channels must be positive")

        # CREATE sub-layers in __init__ (following modern Keras 3 pattern)
        # We create 3 branches for 3 scales, they will be built in build()
        self.bbox_branches = []
        self.cls_branches = []

        for i in range(3):  # 3 scales: P3, P4, P5
            # Create bbox branch (will be built in build())
            bbox_branch = keras.Sequential(name=f"{self.name}_bbox_branch_{i}")
            self.bbox_branches.append(bbox_branch)

            # Create classification branch (will be built in build())
            cls_branch = keras.Sequential(name=f"{self.name}_cls_branch_{i}")
            self.cls_branches.append(cls_branch)

    def build(self, input_shape: List[Tuple[Optional[int], ...]]) -> None:
        """Build detection head branches for each input scale.

        Following modern Keras 3 pattern: explicitly build all sub-layers.

        Args:
            input_shape: List of shape tuples for each input feature map.
                Expected: 3 shapes for [P3, P4, P5] features.

        Raises:
            ValueError: If input_shape is not a list of 3 shapes.
        """
        if not isinstance(input_shape, list):
            raise ValueError("DetectionHead expects a list of input shapes")

        if len(input_shape) != 3:
            raise ValueError(f"DetectionHead expects exactly 3 input feature maps, got {len(input_shape)}")

        logger.info(f"Building YOLOv12DetectionHead with {len(input_shape)} scales")

        # Build branches for each scale
        for i, shape in enumerate(input_shape):
            in_channels = shape[-1]
            if in_channels is None:
                raise ValueError(f"Input channels for scale {i} cannot be None")

            # Calculate channels if not provided
            if self.bbox_channels is not None:
                bbox_c = self.bbox_channels[i]
            else:
                # Standard YOLO approach with DFL consideration
                bbox_c = max(16, in_channels // 4, self.reg_max * 4)

            if self.cls_channels is not None:
                cls_c = self.cls_channels[i]
            else:
                cls_c = max(in_channels, min(self.num_classes, 100))

            logger.info(f"Scale {i}: input_channels={in_channels}, bbox_channels={bbox_c}, cls_channels={cls_c}")

            # Build bbox branch layers
            self.bbox_branches[i].add(ConvBlock(
                filters=bbox_c,
                kernel_size=3,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"bbox_{i}_conv1"
            ))
            self.bbox_branches[i].add(ConvBlock(
                filters=bbox_c,
                kernel_size=3,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"bbox_{i}_conv2"
            ))
            self.bbox_branches[i].add(keras.layers.Conv2D(
                filters=4 * self.reg_max,
                kernel_size=1,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"bbox_{i}_pred"
            ))

            # Build classification branch layers with depthwise separable convolutions
            self.cls_branches[i].add(ConvBlock(
                filters=in_channels,
                kernel_size=3,
                groups=in_channels,  # Depthwise
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"cls_{i}_dw1"
            ))
            self.cls_branches[i].add(ConvBlock(
                filters=cls_c,
                kernel_size=1,  # Pointwise
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"cls_{i}_pw1"
            ))
            self.cls_branches[i].add(ConvBlock(
                filters=cls_c,
                kernel_size=3,
                groups=cls_c,  # Depthwise
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"cls_{i}_dw2"
            ))
            self.cls_branches[i].add(ConvBlock(
                filters=cls_c,
                kernel_size=1,  # Pointwise
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"cls_{i}_pw2"
            ))
            self.cls_branches[i].add(keras.layers.Conv2D(
                filters=self.num_classes,
                kernel_size=1,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"cls_{i}_pred"
            ))

            # CRITICAL: Explicitly build each sub-layer for robust serialization
            self.bbox_branches[i].build(shape)
            self.cls_branches[i].build(shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
            self,
            inputs: List[keras.KerasTensor],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through detection head.

        Args:
            inputs: List of 3 feature maps from backbone/neck.
                Expected shapes: [(B, H/8, W/8, C1), (B, H/16, W/16, C2), (B, H/32, W/32, C3)]
            training: Boolean, whether the layer should behave in training mode.

        Returns:
            Concatenated detection predictions for all scales.
            Shape: (batch_size, total_anchors, 4*reg_max + num_classes)

        Raises:
            ValueError: If inputs is not a list of exactly 3 tensors.
        """
        if not isinstance(inputs, list) or len(inputs) != 3:
            raise ValueError("DetectionHead expects exactly 3 input feature maps")

        outputs = []

        for i, x in enumerate(inputs):
            # Get bbox predictions
            bbox_pred = self.bbox_branches[i](x, training=training)

            # Get classification predictions
            cls_pred = self.cls_branches[i](x, training=training)

            # Reshape predictions
            batch_size = ops.shape(x)[0]
            h = ops.shape(x)[1]
            w = ops.shape(x)[2]

            bbox_pred = ops.reshape(bbox_pred, (batch_size, h * w, 4 * self.reg_max))
            cls_pred = ops.reshape(cls_pred, (batch_size, h * w, self.num_classes))

            # Concatenate bbox and class predictions
            output = ops.concatenate([bbox_pred, cls_pred], axis=-1)
            outputs.append(output)

        # Concatenate all scales
        return ops.concatenate(outputs, axis=1)

    def compute_output_shape(self, input_shape: List[Tuple[Optional[int], ...]]) -> Tuple[Optional[int], ...]:
        """Compute output shape of the layer.

        Args:
            input_shape: List of input shape tuples.

        Returns:
            Output shape tuple.
        """
        if not isinstance(input_shape, list) or len(input_shape) != 3:
            return (None, None, 4 * self.reg_max + self.num_classes)

        total_anchors = 0
        for shape in input_shape:
            if shape[1] is not None and shape[2] is not None:
                total_anchors += shape[1] * shape[2]
            else:
                return (None, None, 4 * self.reg_max + self.num_classes)

        batch_size = input_shape[0][0]
        return (batch_size, total_anchors, 4 * self.reg_max + self.num_classes)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "reg_max": self.reg_max,
            "bbox_channels": self.bbox_channels,
            "cls_channels": self.cls_channels,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class YOLOv12SegmentationHead(keras.layers.Layer):
    """Segmentation head for YOLOv12 multitask learning.

    Uses progressive upsampling with skip connections from backbone features
    to generate pixel-level segmentation masks. Designed for dense prediction
    tasks like crack segmentation.

    Following modern Keras 3 patterns: creates sub-layers in __init__() and builds
    them explicitly in build() for robust serialization.

    Args:
        num_classes: Integer, number of segmentation classes. Must be positive.
            Use 1 for binary segmentation. Defaults to 1.
        intermediate_filters: List of integers, filter sizes for each upsampling stage.
            Must have at least 2 elements. Defaults to [128, 64, 32, 16].
        target_size: Optional tuple of integers (height, width), target output size.
            If None, auto-computed from P3 features as (H/8 * 8, W/8 * 8). Defaults to None.
        use_attention: Boolean, whether to use attention mechanisms for feature fusion.
            Defaults to True.
        dropout_rate: Float, dropout rate for regularization. Must be between 0 and 1.
            Defaults to 0.1.
        kernel_initializer: String or Initializer, initializer for kernel weights.
            Defaults to 'he_normal'.
        kernel_regularizer: Optional Regularizer, regularizer for kernel weights.
            Defaults to None.
        name: Optional string, layer name.
        **kwargs: Additional keyword arguments for Layer base class.

    Input shape:
        List of 3 tensors with shapes:
        - [(batch_size, height/8, width/8, channels_p3),
           (batch_size, height/16, width/16, channels_p4),
           (batch_size, height/32, width/32, channels_p5)]

    Output shape:
        Tensor with shape (batch_size, target_height, target_width, num_classes)

    Example:
        ```python
        # Basic usage
        head = YOLOv12SegmentationHead(num_classes=1)

        # Advanced usage
        head = YOLOv12SegmentationHead(
            num_classes=2,
            intermediate_filters=[256, 128, 64, 32],
            target_size=(512, 512),
            use_attention=True,
            dropout_rate=0.2
        )

        features = [p3_features, p4_features, p5_features]
        segmentation = head(features)
        ```

    Raises:
        ValueError: If num_classes is not positive.
        ValueError: If intermediate_filters has fewer than 2 elements.
        ValueError: If dropout_rate is not between 0 and 1.
        ValueError: If target_size is provided but not a tuple of 2 positive integers.
    """

    def __init__(
            self,
            num_classes: int = 1,
            intermediate_filters: List[int] = [128, 64, 32, 16],
            target_size: Optional[Tuple[int, int]] = None,
            use_attention: bool = True,
            dropout_rate: float = 0.1,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            name: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(name=name, **kwargs)

        # Validate inputs
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if len(intermediate_filters) < 2:
            raise ValueError(f"intermediate_filters must have at least 2 elements, got {len(intermediate_filters)}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")
        if target_size is not None:
            if not (isinstance(target_size, (tuple, list)) and len(target_size) == 2):
                raise ValueError("target_size must be a tuple/list of 2 integers")
            if any(s <= 0 for s in target_size):
                raise ValueError("target_size values must be positive")

        # Store configuration
        self.num_classes = num_classes
        self.intermediate_filters = intermediate_filters
        self.target_size = target_size
        self.use_attention = use_attention
        self.dropout_rate = dropout_rate
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # CREATE sub-layers in __init__ (following modern Keras 3 pattern)
        self.upconv_blocks = []
        self.skip_convs = []
        self.attention_blocks = []

        # Create upsampling blocks for each stage
        for i, filters in enumerate(intermediate_filters):
            # Upsampling block
            upconv_block = keras.Sequential([
                keras.layers.Conv2DTranspose(
                    filters=filters,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f"upconv_{i}_transpose"
                ),
                keras.layers.BatchNormalization(name=f"upconv_{i}_bn"),
                keras.layers.Activation("silu", name=f"upconv_{i}_silu"),
                keras.layers.Conv2D(
                    filters=filters,
                    kernel_size=3,
                    padding="same",
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f"upconv_{i}_refine"
                ),
                keras.layers.BatchNormalization(name=f"upconv_{i}_refine_bn"),
                keras.layers.Activation("silu", name=f"upconv_{i}_refine_silu")
            ], name=f"upconv_block_{i}")
            self.upconv_blocks.append(upconv_block)

            # Skip connection processing (only for first 2 stages: P5->P4, P4->P3)
            if i < 2:
                skip_conv = keras.Sequential([
                    keras.layers.Conv2D(
                        filters=filters,
                        kernel_size=1,
                        kernel_initializer=self.kernel_initializer,
                        kernel_regularizer=self.kernel_regularizer,
                        name=f"skip_{i}_conv"
                    ),
                    keras.layers.BatchNormalization(name=f"skip_{i}_bn"),
                    keras.layers.Activation("silu", name=f"skip_{i}_silu")
                ], name=f"skip_conv_{i}")
                self.skip_convs.append(skip_conv)

                # Attention block for feature fusion
                if self.use_attention:
                    attention_block = SqueezeExcitation(
                        reduction_ratio=0.25,
                        name=f"attention_{i}"
                    )
                    self.attention_blocks.append(attention_block)

        # Final segmentation output layer
        self.final_conv = keras.layers.Conv2D(
            filters=self.num_classes,
            kernel_size=1,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="final_conv"
        )

        # Dropout for regularization
        if self.dropout_rate > 0:
            self.dropout = keras.layers.Dropout(self.dropout_rate, name="dropout")
        else:
            self.dropout = None

        # Store computed target size (will be set in build)
        self._computed_target_size = None

    def build(self, input_shape: List[Tuple[Optional[int], ...]]) -> None:
        """Build segmentation head with progressive upsampling to full resolution.

        Following modern Keras 3 pattern: explicitly build all sub-layers.

        Args:
            input_shape: List of shape tuples for input feature maps.
                Expected: [(B, H/8, W/8, C1), (B, H/16, W/16, C2), (B, H/32, W/32, C3)]

        Raises:
            ValueError: If input_shape is not a list of 3 shapes.
        """
        if not isinstance(input_shape, list) or len(input_shape) != 3:
            raise ValueError("SegmentationHead expects 3 input feature maps")

        # Compute target size if not provided
        if self.target_size is None:
            # P3 has shape (B, H/8, W/8, C), so original size is (H, W) = (H/8 * 8, W/8 * 8)
            p3_shape = input_shape[0]  # (B, H/8, W/8, C1)
            if p3_shape[1] is not None and p3_shape[2] is not None:
                self._computed_target_size = (p3_shape[1] * 8, p3_shape[2] * 8)
            else:
                # Fallback for dynamic shapes
                self._computed_target_size = (256, 256)
                logger.warning("Could not infer target size from input shape, using (256, 256)")
        else:
            self._computed_target_size = self.target_size

        logger.info(f"Building YOLOv12SegmentationHead:")
        logger.info(f"  Input shapes: {input_shape}")
        logger.info(f"  Target output size: {self._computed_target_size}")
        logger.info(f"  Upsampling stages: {len(self.intermediate_filters)}")

        # Build upsampling blocks with correct input shapes accounting for concatenation
        # input_shape: [(B, H/8, W/8, C1), (B, H/16, W/16, C2), (B, H/32, W/32, C3)]
        p3_shape, p4_shape, p5_shape = input_shape

        # Stage progression accounting for concatenation:
        # P5(1024) -> upconv[0] -> 128, concat with P4(128) -> 256
        # 256 -> upconv[1] -> 64, concat with P3(64) -> 128
        # 128 -> upconv[2] -> 32
        # 32 -> upconv[3] -> 16

        # Stage 0: P5 -> P4 scale
        current_shape = p5_shape
        self.upconv_blocks[0].build(current_shape)

        # After upconv[0]: (B, H/16, W/16, intermediate_filters[0])
        upconv0_output_shape = (
            current_shape[0],
            current_shape[1] * 2 if current_shape[1] is not None else None,
            current_shape[2] * 2 if current_shape[2] is not None else None,
            self.intermediate_filters[0]
        )

        # Build skip connection for P4 fusion
        if len(self.skip_convs) > 0:
            self.skip_convs[0].build(p4_shape)

            # After concatenation: upconv0_output + P4_processed
            concat_channels = self.intermediate_filters[0] + self.intermediate_filters[0]  # Both output same channels
            concat_shape = (upconv0_output_shape[0], upconv0_output_shape[1], upconv0_output_shape[2], concat_channels)

            # Build attention if enabled
            if len(self.attention_blocks) > 0:
                self.attention_blocks[0].build(concat_shape)

            current_shape = concat_shape

        # Stage 1: P4 scale -> P3 scale (if we have more than 1 upconv block)
        if len(self.upconv_blocks) > 1:
            self.upconv_blocks[1].build(current_shape)

            # After upconv[1]: (B, H/8, W/8, intermediate_filters[1])
            upconv1_output_shape = (
                current_shape[0],
                current_shape[1] * 2 if current_shape[1] is not None else None,
                current_shape[2] * 2 if current_shape[2] is not None else None,
                self.intermediate_filters[1]
            )

            # Build skip connection for P3 fusion
            if len(self.skip_convs) > 1:
                self.skip_convs[1].build(p3_shape)

                # After concatenation: upconv1_output + P3_processed
                concat_channels = self.intermediate_filters[1] + self.intermediate_filters[1]
                concat_shape = (upconv1_output_shape[0], upconv1_output_shape[1], upconv1_output_shape[2], concat_channels)

                # Build attention if enabled
                if len(self.attention_blocks) > 1:
                    self.attention_blocks[1].build(concat_shape)

                current_shape = concat_shape
            else:
                current_shape = upconv1_output_shape

        # Additional upsampling stages (no more skip connections)
        for i in range(2, len(self.upconv_blocks)):
            self.upconv_blocks[i].build(current_shape)

            # Update current shape for next stage
            current_shape = (
                current_shape[0],
                current_shape[1] * 2 if current_shape[1] is not None else None,
                current_shape[2] * 2 if current_shape[2] is not None else None,
                self.intermediate_filters[i]
            )

        # Build final layers
        self.final_conv.build(current_shape)

        if self.dropout is not None:
            self.dropout.build(current_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
            self,
            inputs: List[keras.KerasTensor],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through segmentation head.

        Args:
            inputs: List of feature maps [P3, P4, P5] from backbone.
                   Expected shapes: [(B, H/8, W/8, C1), (B, H/16, W/16, C2), (B, H/32, W/32, C3)]
            training: Boolean, whether in training mode.

        Returns:
            Segmentation mask tensor with shape (batch_size, target_height, target_width, num_classes).

        Raises:
            ValueError: If inputs is not a list of exactly 3 tensors.
        """
        if len(inputs) != 3:
            raise ValueError("SegmentationHead expects exactly 3 input feature maps")

        p3, p4, p5 = inputs  # P3: H/8, P4: H/16, P5: H/32

        # Start from the deepest features (P5) and progressively upsample
        x = p5

        # Stage 0: P5 (H/32) -> P4 scale (H/16)
        x = self.upconv_blocks[0](x, training=training)

        # Fuse with P4 features
        if len(self.skip_convs) > 0:
            p4_processed = self.skip_convs[0](p4, training=training)
            x = ops.concatenate([x, p4_processed], axis=-1)

            # Apply attention if enabled
            if len(self.attention_blocks) > 0:
                x = self.attention_blocks[0](x, training=training)

        # Stage 1: P4 scale (H/16) -> P3 scale (H/8)
        if len(self.upconv_blocks) > 1:
            x = self.upconv_blocks[1](x, training=training)

            # Fuse with P3 features
            if len(self.skip_convs) > 1:
                p3_processed = self.skip_convs[1](p3, training=training)
                x = ops.concatenate([x, p3_processed], axis=-1)

                # Apply attention if enabled
                if len(self.attention_blocks) > 1:
                    x = self.attention_blocks[1](x, training=training)

        # Additional upsampling stages to reach full resolution
        for i in range(2, len(self.upconv_blocks)):
            x = self.upconv_blocks[i](x, training=training)

        # Ensure exact target size with resize
        if self._computed_target_size is not None:
            x = ops.image.resize(
                x,
                size=self._computed_target_size,
                interpolation="bilinear"
            )

        # Apply dropout
        if self.dropout is not None:
            x = self.dropout(x, training=training)

        # Final segmentation output
        segmentation_output = self.final_conv(x, training=training)

        return segmentation_output

    def compute_output_shape(self, input_shape: List[Tuple[Optional[int], ...]]) -> Tuple[Optional[int], ...]:
        """Compute output shape of the layer.

        Args:
            input_shape: List of input shape tuples.

        Returns:
            Output shape tuple.
        """
        if not isinstance(input_shape, list) or len(input_shape) != 3:
            return (None, None, None, self.num_classes)

        batch_size = input_shape[0][0]

        if self._computed_target_size is not None:
            return (batch_size, self._computed_target_size[0], self._computed_target_size[1], self.num_classes)
        elif self.target_size is not None:
            return (batch_size, self.target_size[0], self.target_size[1], self.num_classes)
        else:
            # Try to compute from P3 features
            p3_shape = input_shape[0]
            if p3_shape[1] is not None and p3_shape[2] is not None:
                return (batch_size, p3_shape[1] * 8, p3_shape[2] * 8, self.num_classes)
            else:
                return (batch_size, None, None, self.num_classes)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "intermediate_filters": self.intermediate_filters,
            "target_size": self.target_size,
            "use_attention": self.use_attention,
            "dropout_rate": self.dropout_rate,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class YOLOv12ClassificationHead(keras.layers.Layer):
    """Classification head for YOLOv12 multitask learning.

    Uses multiscale global pooling with attention mechanism for
    patch-level classification tasks like crack presence detection.

    Following modern Keras 3 patterns: creates sub-layers in __init__() and builds
    them explicitly in build() for robust serialization.

    Args:
        num_classes: Integer, number of classification classes. Must be positive.
            Use 1 for binary classification. Defaults to 1.
        hidden_dims: List of integers, hidden layer dimensions for the classifier.
            Must have at least 1 element. Defaults to [512, 256].
        pooling_types: List of strings, types of global pooling to use.
            Valid values: 'avg', 'max'. Must have at least 1 element. Defaults to ['avg', 'max'].
        use_attention: Boolean, whether to use attention pooling. Defaults to True.
        dropout_rate: Float, dropout rate for regularization. Must be between 0 and 1.
            Defaults to 0.3.
        kernel_initializer: String or Initializer, initializer for kernel weights.
            Defaults to 'he_normal'.
        kernel_regularizer: Optional Regularizer, regularizer for kernel weights.
            Defaults to None.
        name: Optional string, layer name.
        **kwargs: Additional keyword arguments for Layer base class.

    Input shape:
        List of 3 tensors with shapes:
        - [(batch_size, height/8, width/8, channels_p3),
           (batch_size, height/16, width/16, channels_p4),
           (batch_size, height/32, width/32, channels_p5)]

    Output shape:
        Tensor with shape (batch_size, num_classes)

    Example:
        ```python
        # Basic usage
        head = YOLOv12ClassificationHead(num_classes=2)

        # Advanced usage
        head = YOLOv12ClassificationHead(
            num_classes=3,
            hidden_dims=[1024, 512, 256],
            pooling_types=['avg', 'max'],
            use_attention=True,
            dropout_rate=0.4
        )

        features = [p3_features, p4_features, p5_features]
        classification = head(features)
        ```

    Raises:
        ValueError: If num_classes is not positive.
        ValueError: If hidden_dims is empty.
        ValueError: If pooling_types is empty or contains invalid values.
        ValueError: If dropout_rate is not between 0 and 1.
    """

    def __init__(
            self,
            num_classes: int = 1,
            hidden_dims: List[int] = [512, 256],
            pooling_types: List[str] = ["avg", "max"],
            use_attention: bool = True,
            dropout_rate: float = 0.3,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            name: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(name=name, **kwargs)

        # Validate inputs
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims cannot be empty")
        if any(dim <= 0 for dim in hidden_dims):
            raise ValueError("All hidden_dims must be positive")
        if len(pooling_types) == 0:
            raise ValueError("pooling_types cannot be empty")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")

        # Validate pooling types
        valid_pooling_types = {"avg", "max"}
        for pool_type in pooling_types:
            if pool_type not in valid_pooling_types:
                raise ValueError(f"Invalid pooling type: {pool_type}. Must be one of {valid_pooling_types}")

        # Store configuration
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.pooling_types = pooling_types
        self.use_attention = use_attention
        self.dropout_rate = dropout_rate
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # CREATE sub-layers in __init__ (following modern Keras 3 pattern)
        self.pooling_layers = []
        self.attention_pooling = None
        self.dense_layers = []
        self.dropout_layers = []
        self.final_dense = None

        # Create pooling layers
        for pool_type in pooling_types:
            if pool_type == "avg":
                pool_layer = keras.layers.GlobalAveragePooling2D(name=f"{pool_type}_pool")
            elif pool_type == "max":
                pool_layer = keras.layers.GlobalMaxPooling2D(name=f"{pool_type}_pool")

            self.pooling_layers.append(pool_layer)

        # Create dense layers
        for i, hidden_dim in enumerate(hidden_dims):
            dense_layer = keras.layers.Dense(
                hidden_dim,
                activation="relu",
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"dense_{i}"
            )
            self.dense_layers.append(dense_layer)

            if dropout_rate > 0:
                dropout_layer = keras.layers.Dropout(dropout_rate, name=f"dropout_{i}")
                self.dropout_layers.append(dropout_layer)

        # Final classification layer
        self.final_dense = keras.layers.Dense(
            self.num_classes,
            activation=None,  # Output logits
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="final_dense"
        )

    def build(self, input_shape: List[Tuple[Optional[int], ...]]) -> None:
        """Build classification head.

        Following modern Keras 3 pattern: explicitly build all sub-layers.

        Args:
            input_shape: List of shape tuples for input feature maps.
                Expected: 3 shapes for [P3, P4, P5] features.

        Raises:
            ValueError: If input_shape is not a list of 3 shapes.
        """
        if not isinstance(input_shape, list) or len(input_shape) != 3:
            raise ValueError("ClassificationHead expects 3 input feature maps")

        logger.info(f"Building YOLOv12ClassificationHead with pooling types: {self.pooling_types}")

        # Build pooling layers - they don't need explicit building, but we call it for consistency
        for pool_layer in self.pooling_layers:
            # Global pooling layers adapt to any spatial input size
            pool_layer.build(input_shape[0])  # Any input shape works for global pooling

        # Calculate total feature dimension after concatenating all scales and pooling types
        total_dim = sum(shape[-1] for shape in input_shape if shape[-1] is not None) * len(self.pooling_types)

        if total_dim == 0:  # Handle case where channels are unknown
            total_dim = None

        logger.info(f"Total feature dimension: {total_dim}")

        # Build attention pooling if enabled
        if self.use_attention and total_dim is not None:
            attention_dim = total_dim // 4

            self.attention_pooling = keras.Sequential([
                keras.layers.Dense(
                    attention_dim,
                    activation="relu",
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name="attention_dense"
                ),
                keras.layers.Dense(
                    total_dim,
                    activation="sigmoid",
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name="attention_weights"
                )
            ], name="attention_pooling")

            # Build attention pooling
            attention_input_shape = (input_shape[0][0], total_dim)
            self.attention_pooling.build(attention_input_shape)

        # Build dense layers
        current_input_shape = (input_shape[0][0], total_dim)
        for i, dense_layer in enumerate(self.dense_layers):
            dense_layer.build(current_input_shape)
            current_input_shape = (input_shape[0][0], self.hidden_dims[i])

            if i < len(self.dropout_layers):
                self.dropout_layers[i].build(current_input_shape)

        # Build final layer
        self.final_dense.build(current_input_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
            self,
            inputs: List[keras.KerasTensor],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through classification head.

        Args:
            inputs: List of feature maps [P3, P4, P5] from backbone.
                Expected shapes: [(B, H/8, W/8, C1), (B, H/16, W/16, C2), (B, H/32, W/32, C3)]
            training: Boolean, whether in training mode.

        Returns:
            Classification logits with shape (batch_size, num_classes).

        Raises:
            ValueError: If inputs is not a list of exactly 3 tensors.
        """
        if len(inputs) != 3:
            raise ValueError("ClassificationHead expects exactly 3 input feature maps")

        # Apply different pooling operations to each scale
        pooled_features = []

        for feature_map in inputs:
            for pool_layer in self.pooling_layers:
                pooled = pool_layer(feature_map)
                pooled_features.append(pooled)

        # Concatenate all pooled features
        x = ops.concatenate(pooled_features, axis=-1)

        # Apply attention pooling if enabled
        if self.attention_pooling is not None:
            attention_weights = self.attention_pooling(x, training=training)
            x = x * attention_weights

        # Pass through dense layers
        for i, dense_layer in enumerate(self.dense_layers):
            x = dense_layer(x)

            if i < len(self.dropout_layers):
                x = self.dropout_layers[i](x, training=training)

        # Final classification output
        classification_output = self.final_dense(x)

        return classification_output

    def compute_output_shape(self, input_shape: List[Tuple[Optional[int], ...]]) -> Tuple[Optional[int], ...]:
        """Compute output shape of the layer.

        Args:
            input_shape: List of input shape tuples.

        Returns:
            Output shape tuple.
        """
        if not isinstance(input_shape, list) or len(input_shape) != 3:
            return (None, self.num_classes)

        batch_size = input_shape[0][0]
        return (batch_size, self.num_classes)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "hidden_dims": self.hidden_dims,
            "pooling_types": self.pooling_types,
            "use_attention": self.use_attention,
            "dropout_rate": self.dropout_rate,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config