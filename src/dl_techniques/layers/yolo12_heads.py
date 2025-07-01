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
        reg_max=16
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

from .yolo12 import ConvBlock
from .squeeze_excitation import SqueezeExcitation
from dl_techniques.utils.logger import logger

import keras
from keras import ops
from typing import Tuple, Any, Dict, List, Union


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class YOLOv12DetectionHead(keras.layers.Layer):
    """YOLOv12 Detection Head with separate classification and regression branches.

    This head processes multi-scale feature maps from the backbone/neck to produce
    object detection predictions including bounding boxes and class probabilities.
    Uses depthwise separable convolutions for efficiency.

    Args:
        num_classes: Number of object classes to detect.
        reg_max: Maximum value for DFL (Distribution Focal Loss) regression.
        kernel_initializer: Initializer for kernel weights.
        kernel_regularizer: Regularizer for kernel weights.
        name: Layer name.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
            self,
            num_classes: int = 80,
            reg_max: int = 16,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            name: Optional[str] = None,
            **kwargs
    ) -> None:
        super().__init__(name=name, **kwargs)

        self.num_classes = num_classes
        self.reg_max = reg_max
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # Will store branch layers for each scale
        self.bbox_branches = []
        self.cls_branches = []
        self._build_input_shape = None

    def build(self, input_shape: List[Tuple[Optional[int], ...]]) -> None:
        """Build detection head branches for each input scale.

        Args:
            input_shape: List of shape tuples for each input feature map.

        Raises:
            ValueError: If input_shape is not a list of 3 shapes.
        """
        super().build(input_shape)

        # Store for serialization
        self._build_input_shape = input_shape

        if not isinstance(input_shape, list):
            raise ValueError("DetectionHead expects a list of input shapes")

        if len(input_shape) != 3:
            raise ValueError(f"DetectionHead expects exactly 3 input shapes, got {len(input_shape)}")

        logger.info(f"Building YOLOv12DetectionHead with {len(input_shape)} scales")

        # Create branches for each scale
        for i, shape in enumerate(input_shape):
            in_channels = shape[-1]

            # Standard YOLO approach: c2 = max(16, in_channels // 4)
            # Alternative approach: c2 = max(16, in_channels // 4, self.reg_max * 4)
            # The alternative ties intermediate layer width to DFL regression range,
            # potentially providing better capacity for complex regression tasks
            c2 = max(16, in_channels // 4, self.reg_max * 4)
            c3 = max(in_channels, min(self.num_classes, 100))

            logger.info(f"Scale {i}: input_channels={in_channels}, bbox_channels={c2}, cls_channels={c3}")

            # Bounding box regression branch
            bbox_branch = keras.Sequential([
                ConvBlock(
                    filters=c2,
                    kernel_size=3,
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f"{self.name}_bbox_{i}_1"
                ),
                ConvBlock(
                    filters=c2,
                    kernel_size=3,
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f"{self.name}_bbox_{i}_2"
                ),
                keras.layers.Conv2D(
                    filters=4 * self.reg_max,
                    kernel_size=1,
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f"{self.name}_bbox_{i}_pred"
                )
            ], name=f"{self.name}_bbox_{i}")

            # Classification branch with depthwise separable convolutions
            cls_branch = keras.Sequential([
                ConvBlock(
                    filters=in_channels,
                    kernel_size=3,
                    groups=in_channels,  # Depthwise
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f"{self.name}_cls_{i}_dw1"
                ),
                ConvBlock(
                    filters=c3,
                    kernel_size=1,  # Pointwise
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f"{self.name}_cls_{i}_pw1"
                ),
                ConvBlock(
                    filters=c3,
                    kernel_size=3,
                    groups=c3,  # Depthwise
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f"{self.name}_cls_{i}_dw2"
                ),
                ConvBlock(
                    filters=c3,
                    kernel_size=1,  # Pointwise
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f"{self.name}_cls_{i}_pw2"
                ),
                keras.layers.Conv2D(
                    filters=self.num_classes,
                    kernel_size=1,
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f"{self.name}_cls_{i}_pred"
                )
            ], name=f"{self.name}_cls_{i}")

            self.bbox_branches.append(bbox_branch)
            self.cls_branches.append(cls_branch)

    def call(
            self,
            inputs: List[keras.KerasTensor],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through detection head.

        Args:
            inputs: List of 3 feature maps from backbone/neck.
            training: Whether the layer should behave in training mode.

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

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "reg_max": self.reg_max,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get build configuration for serialization.

        Returns:
            Dictionary containing the build configuration.
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build layer from configuration.

        Args:
            config: Dictionary containing build configuration.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class YOLOv12SegmentationHead(keras.layers.Layer):
    """Segmentation head for YOLOv12 multitask learning.

    Uses progressive upsampling with skip connections from backbone features
    to generate pixel-level segmentation masks. Designed for dense prediction
    tasks like crack segmentation.

    FIXED VERSION: Now properly upsamples to original input resolution.

    Args:
        num_classes: Number of segmentation classes (1 for binary segmentation).
        intermediate_filters: Filter sizes for each upsampling stage.
        target_size: Target output size (height, width). If None, auto-computed from input.
        use_attention: Whether to use attention mechanisms for feature fusion.
        dropout_rate: Dropout rate for regularization.
        kernel_initializer: Initializer for kernel weights.
        kernel_regularizer: Regularizer for kernel weights.
        name: Layer name.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
            self,
            num_classes: int = 1,
            intermediate_filters: List[int] = [128, 64, 32, 16],  # Added more stages
            target_size: Optional[Tuple[int, int]] = None,  # NEW: explicit target size
            use_attention: bool = True,
            dropout_rate: float = 0.1,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            name: Optional[str] = None,
            **kwargs
    ) -> None:
        """Initialize segmentation head.

        Args:
            num_classes: Number of segmentation classes (1 for binary crack detection).
            intermediate_filters: Number of filters for each upsampling stage.
            target_size: Target output size (height, width). If None, inferred from P3 features.
            use_attention: Whether to use attention mechanisms.
            dropout_rate: Dropout rate for regularization.
            kernel_initializer: Weight initializer.
            kernel_regularizer: Weight regularizer.
            name: Layer name.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(name=name, **kwargs)

        self.num_classes = num_classes
        self.intermediate_filters = intermediate_filters
        self.target_size = target_size
        self.use_attention = use_attention
        self.dropout_rate = dropout_rate
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # Will be built in build()
        self.upconv_blocks = []
        self.skip_convs = []
        self.attention_blocks = []
        self.final_upsampling = None  # NEW: For final upsampling to target size
        self.final_conv = None
        self.dropout = None
        self._build_input_shape = None
        self._computed_target_size = None

    def build(self, input_shape: List[Tuple[Optional[int], ...]]) -> None:
        """Build segmentation head with progressive upsampling to full resolution.

        Args:
            input_shape: List of shape tuples for input feature maps.
                Expected: [(B, H/8, W/8, C1), (B, H/16, W/16, C2), (B, H/32, W/32, C3)]

        Raises:
            ValueError: If input_shape is not a list of 3 shapes.
        """
        super().build(input_shape)

        # Store for serialization
        self._build_input_shape = input_shape

        if not isinstance(input_shape, list) or len(input_shape) != 3:
            raise ValueError("SegmentationHead expects 3 input feature maps")

        # Compute target size if not provided
        if self.target_size is None:
            # P3 has shape (B, H/8, W/8, C), so original size is (H, W) = (H/8 * 8, W/8 * 8)
            p3_shape = input_shape[0]  # (B, H/8, W/8, C1)
            if p3_shape[1] is not None and p3_shape[2] is not None:
                self._computed_target_size = (p3_shape[1] * 8, p3_shape[2] * 8)
            else:
                # Fallback for dynamic shapes - assume 256x256 patches
                self._computed_target_size = (256, 256)
                logger.warning("Could not infer target size from input shape, using (256, 256)")
        else:
            self._computed_target_size = self.target_size

        logger.info(f"Building YOLOv12SegmentationHead:")
        logger.info(f"  Input shapes: {input_shape}")
        logger.info(f"  Target output size: {self._computed_target_size}")
        logger.info(f"  Upsampling stages: {len(self.intermediate_filters)}")

        # input_shape: [(B, H/8, W/8, C1), (B, H/16, W/16, C2), (B, H/32, W/32, C3)]
        # We'll upsample from P5 (H/32) progressively to full resolution

        # Build upsampling blocks
        # Stage progression: P5(H/32) -> P4(H/16) -> P3(H/8) -> H/4 -> H/2 -> H
        for i, filters in enumerate(self.intermediate_filters):
            logger.info(f"  Upsampling stage {i}: filters={filters}")

            # Upsampling block
            upconv_block = keras.Sequential([
                # Transpose convolution for upsampling
                keras.layers.Conv2DTranspose(
                    filters=filters,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f"{self.name}_upconv_{i}_transpose"
                ),
                keras.layers.BatchNormalization(name=f"{self.name}_upconv_{i}_bn"),
                keras.layers.Activation("silu", name=f"{self.name}_upconv_{i}_silu"),

                # Refinement convolution
                keras.layers.Conv2D(
                    filters=filters,
                    kernel_size=3,
                    padding="same",
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f"{self.name}_upconv_{i}_refine"
                ),
                keras.layers.BatchNormalization(name=f"{self.name}_upconv_{i}_refine_bn"),
                keras.layers.Activation("silu", name=f"{self.name}_upconv_{i}_refine_silu")
            ], name=f"{self.name}_upconv_block_{i}")

            self.upconv_blocks.append(upconv_block)

            # Skip connection processing for P4 and P3 fusion
            # Only create skip connections for the first 2 stages (P5->P4, P4->P3)
            if i < 2:  # Only for P4 and P3 fusion
                skip_conv = keras.Sequential([
                    keras.layers.Conv2D(
                        filters=filters,
                        kernel_size=1,
                        kernel_initializer=self.kernel_initializer,
                        kernel_regularizer=self.kernel_regularizer,
                        name=f"{self.name}_skip_{i}_conv"
                    ),
                    keras.layers.BatchNormalization(name=f"{self.name}_skip_{i}_bn"),
                    keras.layers.Activation("silu", name=f"{self.name}_skip_{i}_silu")
                ], name=f"{self.name}_skip_conv_{i}")

                self.skip_convs.append(skip_conv)

                # Attention block for feature fusion
                if self.use_attention:
                    attention_block = SqueezeExcitation(
                        reduction_ratio=0.25,
                        name=f"{self.name}_attention_{i}"
                    )
                    self.attention_blocks.append(attention_block)

        # Final segmentation output
        self.final_conv = keras.Sequential([
            keras.layers.Conv2D(
                filters=self.num_classes,
                kernel_size=1,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"{self.name}_final_conv"
            )
            # the activation is handled by the loss
        ], name=f"{self.name}_final")

        # Dropout for regularization
        if self.dropout_rate > 0:
            self.dropout = keras.layers.Dropout(self.dropout_rate, name=f"{self.name}_dropout")

    def call(
            self,
            inputs: List[keras.KerasTensor],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through segmentation head.

        Args:
            inputs: List of feature maps [P3, P4, P5] from backbone.
                   Expected shapes: [(B, H/8, W/8, C1), (B, H/16, W/16, C2), (B, H/32, W/32, C3)]
            training: Whether in training mode.

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
            if self.use_attention and len(self.attention_blocks) > 0:
                x = self.attention_blocks[0](x, training=training)

        # Stage 1: P4 scale (H/16) -> P3 scale (H/8)
        if len(self.upconv_blocks) > 1:
            x = self.upconv_blocks[1](x, training=training)

            # Fuse with P3 features
            if len(self.skip_convs) > 1:
                p3_processed = self.skip_convs[1](p3, training=training)
                x = ops.concatenate([x, p3_processed], axis=-1)

                # Apply attention if enabled
                if self.use_attention and len(self.attention_blocks) > 1:
                    x = self.attention_blocks[1](x, training=training)

        # Additional upsampling stages to reach full resolution
        # Stage 2: P3 scale (H/8) -> H/4
        if len(self.upconv_blocks) > 2:
            x = self.upconv_blocks[2](x, training=training)

        # Stage 3: H/4 -> H/2
        if len(self.upconv_blocks) > 3:
            x = self.upconv_blocks[3](x, training=training)

        # Stage 4: H/2 -> H (full resolution) - if we have 5 stages
        if len(self.upconv_blocks) > 4:
            x = self.upconv_blocks[4](x, training=training)

        # Ensure exact target size with resize (no-op if already correct size)
        # This avoids complex tensor comparisons and dynamic layer creation
        if self._computed_target_size != (None, None):
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

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "intermediate_filters": self.intermediate_filters,
            "target_size": self.target_size,  # NEW
            "use_attention": self.use_attention,
            "dropout_rate": self.dropout_rate,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get build configuration for serialization.

        Returns:
            Dictionary containing the build configuration.
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build layer from configuration.

        Args:
            config: Dictionary containing build configuration.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class YOLOv12ClassificationHead(keras.layers.Layer):
    """Classification head for YOLOv12 multitask learning.

    Uses multiscale global pooling with attention mechanism for
    patch-level classification tasks like crack presence detection.

    Args:
        num_classes: Number of classification classes (1 for binary classification).
        hidden_dims: Hidden layer dimensions for the classifier.
        pooling_types: Types of global pooling to use ('avg', 'max').
        use_attention: Whether to use attention pooling.
        dropout_rate: Dropout rate for regularization.
        kernel_initializer: Initializer for kernel weights.
        kernel_regularizer: Regularizer for kernel weights.
        name: Layer name.
        **kwargs: Additional keyword arguments.
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
            **kwargs
    ) -> None:
        """Initialize classification head.

        Args:
            num_classes: Number of classification classes (1 for binary).
            hidden_dims: Hidden layer dimensions.
            pooling_types: Types of global pooling to use.
            use_attention: Whether to use attention pooling.
            dropout_rate: Dropout rate for regularization.
            kernel_initializer: Weight initializer.
            kernel_regularizer: Weight regularizer.
            name: Layer name.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(name=name, **kwargs)

        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.pooling_types = pooling_types
        self.use_attention = use_attention
        self.dropout_rate = dropout_rate
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # Validate pooling types
        valid_pooling_types = {"avg", "max"}
        for pool_type in pooling_types:
            if pool_type not in valid_pooling_types:
                raise ValueError(f"Invalid pooling type: {pool_type}. Must be one of {valid_pooling_types}")

        # Will be built in build()
        self.pooling_layers = []
        self.attention_pooling = None
        self.dense_layers = []
        self.dropout_layers = []
        self.final_dense = None
        self._build_input_shape = None

    def build(self, input_shape: List[Tuple[Optional[int], ...]]) -> None:
        """Build classification head.

        Args:
            input_shape: List of shape tuples for input feature maps.

        Raises:
            ValueError: If input_shape is not a list of 3 shapes.
        """
        super().build(input_shape)

        # Store for serialization
        self._build_input_shape = input_shape

        if not isinstance(input_shape, list) or len(input_shape) != 3:
            raise ValueError("ClassificationHead expects 3 input feature maps")

        logger.info(f"Building YOLOv12ClassificationHead with pooling types: {self.pooling_types}")

        # Build pooling layers for each scale
        for pool_type in self.pooling_types:
            if pool_type == "avg":
                pool_layer = keras.layers.GlobalAveragePooling2D(name=f"{self.name}_{pool_type}_pool")
            elif pool_type == "max":
                pool_layer = keras.layers.GlobalMaxPooling2D(name=f"{self.name}_{pool_type}_pool")

            self.pooling_layers.append(pool_layer)

        # Attention pooling
        if self.use_attention:
            # Calculate total feature dimension after concatenating all scales and pooling types
            total_dim = sum(shape[-1] for shape in input_shape) * len(self.pooling_types)

            logger.info(f"Total feature dimension for attention: {total_dim}")

            self.attention_pooling = keras.Sequential([
                keras.layers.Dense(
                    total_dim // 4,
                    activation="relu",
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f"{self.name}_attention_dense"
                ),
                keras.layers.Dense(
                    total_dim,
                    activation="sigmoid",
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f"{self.name}_attention_weights"
                )
            ], name=f"{self.name}_attention_pooling")

        # Build dense layers
        for i, hidden_dim in enumerate(self.hidden_dims):
            dense_layer = keras.layers.Dense(
                hidden_dim,
                activation="relu",
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"{self.name}_dense_{i}"
            )
            self.dense_layers.append(dense_layer)

            if self.dropout_rate > 0:
                dropout_layer = keras.layers.Dropout(
                    self.dropout_rate,
                    name=f"{self.name}_dropout_{i}"
                )
                self.dropout_layers.append(dropout_layer)

        # Final classification layer
        self.final_dense = keras.layers.Dense(
            self.num_classes,
            activation=None, # explicitly output logits
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name=f"{self.name}_final_dense"
        )

    def call(
            self,
            inputs: List[keras.KerasTensor],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through classification head.

        Args:
            inputs: List of feature maps [P3, P4, P5] from backbone.
            training: Whether in training mode.

        Returns:
            Classification probabilities with shape (batch_size, num_classes).

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

    def get_build_config(self) -> Dict[str, Any]:
        """Get build configuration for serialization.

        Returns:
            Dictionary containing the build configuration.
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build layer from configuration.

        Args:
            config: Dictionary containing build configuration.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

# ---------------------------------------------------------------------
