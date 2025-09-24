"""
Vision Task Head Network Factory

A comprehensive factory for building configurable head networks for various computer vision_heads tasks.
Uses the dl_techniques framework components including Standard Blocks, FFN factory, 
Norm factory, and Attention factory.
"""

import keras
from keras import layers, ops
from typing import Dict, List, Optional, Union, Tuple, Any, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..activations import ActivationType
from ..standard_blocks import ConvBlock, DenseBlock
from ..ffn.factory import create_ffn_layer, FFNType
from ..attention import create_attention_layer, AttentionType
from ..norms import create_normalization_layer, NormalizationType
from .task_types import TaskType, TaskConfiguration, CommonTaskConfigurations

# ---------------------------------------------------------------------
# Base Head Class
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class BaseVisionHead(keras.layers.Layer):
    """
    Base class for all vision_heads task heads.

    Provides common functionality and structure for task-specific heads.

    Args:
        hidden_dim: int, hidden dimension for intermediate layers
        normalization_type: str, type of normalization to use
        activation_type: str, type of activation function
        dropout_rate: float, dropout rate for regularization
        use_attention: bool, whether to include attention mechanisms
        attention_type: str, type of attention to use if enabled
        use_ffn: bool, whether to include FFN blocks
        ffn_type: str, type of FFN to use if enabled
        ffn_expansion_factor: int, expansion factor for FFN
        **kwargs: Additional arguments for base Layer class
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
        """Create common layers used across different heads."""

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
                self.attention = create_attention_layer(
                    self.attention_type,
                    dim=self.hidden_dim,
                    name=f'{self.name}_attention'
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

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer."""
        super().build(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
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

@keras.saving.register_keras_serializable()
class DetectionHead(BaseVisionHead):
    """
    Detection head for object detection tasks.

    Outputs bounding box regression and classification scores.

    Args:
        num_classes: int, number of object classes
        num_anchors: int, number of anchor boxes per location
        bbox_dims: int, dimensions for bounding box (typically 4)
        **kwargs: Arguments for BaseVisionHead
    """

    def __init__(
            self,
            num_classes: int,
            num_anchors: int = 9,
            bbox_dims: int = 4,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.bbox_dims = bbox_dims

        # Create detection-specific layers
        self._create_detection_layers()

    def _create_detection_layers(self) -> None:
        """Create detection-specific layers."""

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

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Dict[str, keras.KerasTensor]:
        """Forward pass through detection head."""

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

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
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

@keras.saving.register_keras_serializable()
class SegmentationHead(BaseVisionHead):
    """
    Segmentation head for semantic segmentation tasks.

    Args:
        num_classes: int, number of segmentation classes
        upsampling_factor: int, factor for upsampling to match input resolution
        use_skip_connections: bool, whether to use skip connections
        **kwargs: Arguments for BaseVisionHead
    """

    def __init__(
            self,
            num_classes: int,
            upsampling_factor: int = 4,
            use_skip_connections: bool = True,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.upsampling_factor = upsampling_factor
        self.use_skip_connections = use_skip_connections

        self._create_segmentation_layers()

    def _create_segmentation_layers(self) -> None:
        """Create segmentation-specific layers."""

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

    def call(
            self,
            inputs: Union[keras.KerasTensor, List[keras.KerasTensor]],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through segmentation head."""

        # Handle multi-scale inputs if skip connections are used
        if isinstance(inputs, list) and self.use_skip_connections:
            x = inputs[-1]  # Highest level features
            skip_features = inputs[:-1][::-1]  # Reverse order for bottom-up
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

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
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

@keras.saving.register_keras_serializable()
class DepthEstimationHead(BaseVisionHead):
    """
    Depth estimation head for predicting depth maps.

    Args:
        output_channels: int, number of output channels (1 for single depth)
        min_depth: float, minimum depth value
        max_depth: float, maximum depth value
        use_log_depth: bool, whether to predict log depth
        **kwargs: Arguments for BaseVisionHead
    """

    def __init__(
            self,
            output_channels: int = 1,
            min_depth: float = 0.1,
            max_depth: float = 100.0,
            use_log_depth: bool = True,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.output_channels = output_channels
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.use_log_depth = use_log_depth

        self._create_depth_layers()

    def _create_depth_layers(self) -> None:
        """Create depth-specific layers."""

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
            activation='sigmoid',  # Will be scaled to depth range
            name='depth_head'
        )

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Dict[str, keras.KerasTensor]:
        """Forward pass through depth head."""

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
            'confidence': depth_normalized  # Can be used as confidence
        }

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
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

@keras.saving.register_keras_serializable()
class ClassificationHead(BaseVisionHead):
    """
    Classification head for image-level classification.

    Args:
        num_classes: int, number of classes
        use_global_pooling: bool, whether to use global pooling
        pooling_type: str, type of pooling ('avg' or 'max')
        **kwargs: Arguments for BaseVisionHead
    """

    def __init__(
            self,
            num_classes: int,
            use_global_pooling: bool = True,
            pooling_type: Literal['avg', 'max'] = 'avg',
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.use_global_pooling = use_global_pooling
        self.pooling_type = pooling_type

        self._create_classification_layers()

    def _create_classification_layers(self) -> None:
        """Create classification-specific layers."""

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

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Dict[str, keras.KerasTensor]:
        """Forward pass through classification head."""

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
            'probabilities': logits  # Softmax already applied
        }

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
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

@keras.saving.register_keras_serializable()
class InstanceSegmentationHead(BaseVisionHead):
    """
    Instance segmentation head combining detection and segmentation.

    Args:
        num_classes: int, number of object classes
        num_instances: int, maximum number of instances
        mask_size: Tuple[int, int], size of instance masks
        **kwargs: Arguments for BaseVisionHead
    """

    def __init__(
            self,
            num_classes: int,
            num_instances: int = 100,
            mask_size: Tuple[int, int] = (28, 28),
            **kwargs: Any
    ) -> None:
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
            use_attention=False,  # Already applied in main head
            use_ffn=False
        )

        self._create_mask_layers()

    def _create_mask_layers(self) -> None:
        """Create mask prediction layers."""

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

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Dict[str, keras.KerasTensor]:
        """Forward pass through instance segmentation head."""

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

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'num_instances': self.num_instances,
            'mask_size': self.mask_size
        })
        return config


# ---------------------------------------------------------------------
# Multi-Task Head
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MultiTaskHead(keras.layers.Layer):
    """
    Multi-task head that combines multiple task-specific heads.

    Args:
        task_configs: Dict mapping task names to their configurations
        shared_backbone_dim: int, dimension of shared backbone features
        use_task_specific_attention: bool, whether each task gets its own attention
        **kwargs: Additional arguments
    """

    def __init__(
            self,
            task_configs: Dict[str, Dict[str, Any]],
            shared_backbone_dim: int = 256,
            use_task_specific_attention: bool = True,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.task_configs = task_configs
        self.shared_backbone_dim = shared_backbone_dim
        self.use_task_specific_attention = use_task_specific_attention

        self._create_task_heads()

    def _create_task_heads(self) -> None:
        """Create task-specific heads based on configuration."""

        self.task_heads = {}

        for task_name, config in self.task_configs.items():
            task_type = config.pop('task_type')

            # Add shared configuration
            config['hidden_dim'] = config.get('hidden_dim', self.shared_backbone_dim)
            config['use_attention'] = self.use_task_specific_attention

            # Create appropriate head
            if task_type == TaskType.DETECTION:
                self.task_heads[task_name] = DetectionHead(**config)
            elif task_type == TaskType.SEGMENTATION:
                self.task_heads[task_name] = SegmentationHead(**config)
            elif task_type == TaskType.DEPTH_ESTIMATION:
                self.task_heads[task_name] = DepthEstimationHead(**config)
            elif task_type == TaskType.CLASSIFICATION:
                self.task_heads[task_name] = ClassificationHead(**config)
            elif task_type == TaskType.INSTANCE_SEGMENTATION:
                self.task_heads[task_name] = InstanceSegmentationHead(**config)
            else:
                raise ValueError(f"Unsupported task type: {task_type}")

    def call(
            self,
            inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
            training: Optional[bool] = None
    ) -> Dict[str, Dict[str, keras.KerasTensor]]:
        """Forward pass through all task heads."""

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

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
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
        task_type: Union[TaskType, str],
        **kwargs: Any
) -> BaseVisionHead:
    """
    Factory function to create vision_heads task heads.

    Args:
        task_type: TaskType enum or string specifying the task
        **kwargs: Configuration parameters for the specific head

    Returns:
        Configured vision_heads head for the specified task

    Example:
        >>> # Create detection head
        >>> det_head = create_vision_head(
        ...     TaskType.DETECTION,
        ...     num_classes=80,
        ...     hidden_dim=256,
        ...     normalization_type='layer_norm',
        ...     ffn_type='swiglu'
        ... )

        >>> # Create segmentation head with attention
        >>> seg_head = create_vision_head(
        ...     'segmentation',
        ...     num_classes=21,
        ...     use_attention=True,
        ...     attention_type='cbam'
        ... )

        >>> # Create depth estimation head
        >>> depth_head = create_vision_head(
        ...     TaskType.DEPTH_ESTIMATION,
        ...     min_depth=0.1,
        ...     max_depth=100.0,
        ...     use_log_depth=True
        ... )
    """

    # Convert string to TaskType if needed
    if isinstance(task_type, str):
        task_type = TaskType.from_string(task_type)

    # Create appropriate head based on task type
    if task_type == TaskType.DETECTION:
        return DetectionHead(**kwargs)

    elif task_type == TaskType.SEGMENTATION:
        return SegmentationHead(**kwargs)

    elif task_type == TaskType.DEPTH_ESTIMATION:
        return DepthEstimationHead(**kwargs)

    elif task_type == TaskType.CLASSIFICATION:
        return ClassificationHead(**kwargs)

    elif task_type == TaskType.INSTANCE_SEGMENTATION:
        return InstanceSegmentationHead(**kwargs)

    elif task_type == TaskType.SURFACE_NORMALS:
        # Surface normals use similar architecture to depth
        return DepthEstimationHead(output_channels=3, **kwargs)

    elif task_type == TaskType.OPTICAL_FLOW:
        # Optical flow predicts 2D motion vectors
        return DepthEstimationHead(output_channels=2, **kwargs)

    elif task_type == TaskType.KEYPOINT_DETECTION:
        # Keypoint detection is similar to detection with different outputs
        return DetectionHead(**kwargs)

    elif task_type in [TaskType.DENOISING, TaskType.SUPER_RESOLUTION]:
        # Image enhancement tasks
        return create_enhancement_head(task_type, **kwargs)

    else:
        raise ValueError(f"Unsupported task type: {task_type}")


def create_enhancement_head(
        task_type: TaskType,
        **kwargs: Any
) -> BaseVisionHead:
    """
    Create enhancement-specific heads (denoising, super-resolution, etc.).

    This is a placeholder for enhancement-specific architectures.
    """

    @keras.saving.register_keras_serializable()
    class EnhancementHead(BaseVisionHead):
        def __init__(self, output_channels: int = 3, scale_factor: int = 1, **kwargs):
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

        def call(self, inputs, training=None):
            x = inputs

            for block in self.enhance_blocks:
                x = block(x, training=training)

            if self.scale_factor > 1:
                x = self.upsample(x)
            else:
                x = self.output_conv(x)

            return {'enhanced': x}

        def get_config(self):
            config = super().get_config()
            config.update({
                'output_channels': self.output_channels,
                'scale_factor': self.scale_factor
            })
            return config

    if task_type == TaskType.SUPER_RESOLUTION:
        kwargs['scale_factor'] = kwargs.get('scale_factor', 2)

    return EnhancementHead(**kwargs)


def create_multi_task_head(
        task_configuration: Union[TaskConfiguration, List[TaskType], Dict[str, Dict]],
        **kwargs: Any
) -> MultiTaskHead:
    """
    Create a multi-task head from task configuration.

    Args:
        task_configuration: Can be:
            - TaskConfiguration object
            - List of TaskType enums
            - Dict mapping task names to configurations
        **kwargs: Additional configuration

    Returns:
        MultiTaskHead instance

    Example:
        >>> # From TaskConfiguration
        >>> config = CommonTaskConfigurations.DETECTION_SEGMENTATION_DEPTH
        >>> multi_head = create_multi_task_head(config)

        >>> # From list of tasks
        >>> tasks = [TaskType.DETECTION, TaskType.SEGMENTATION]
        >>> multi_head = create_multi_task_head(tasks, hidden_dim=256)

        >>> # From detailed configuration
        >>> config = {
        ...     'detection': {
        ...         'task_type': TaskType.DETECTION,
        ...         'num_classes': 80,
        ...         'num_anchors': 9
        ...     },
        ...     'segmentation': {
        ...         'task_type': TaskType.SEGMENTATION,
        ...         'num_classes': 21,
        ...         'upsampling_factor': 4
        ...     }
        ... }
        >>> multi_head = create_multi_task_head(config)
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
                task = TaskType.from_string(task)
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
    Configuration helper for vision_heads heads.

    Provides default configurations for different tasks and scenarios.
    """

    @staticmethod
    def get_default_config(task_type: TaskType) -> Dict[str, Any]:
        """Get default configuration for a task type."""

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
            TaskType.DETECTION: {
                'num_classes': 80,  # COCO default
                'num_anchors': 9,
                'bbox_dims': 4,
                'use_attention': False
            },
            TaskType.SEGMENTATION: {
                'num_classes': 21,  # VOC default
                'upsampling_factor': 4,
                'use_skip_connections': True,
                'use_attention': True,
                'attention_type': 'cbam'
            },
            TaskType.DEPTH_ESTIMATION: {
                'output_channels': 1,
                'min_depth': 0.1,
                'max_depth': 100.0,
                'use_log_depth': True,
                'use_attention': False
            },
            TaskType.CLASSIFICATION: {
                'num_classes': 1000,  # ImageNet default
                'use_global_pooling': True,
                'pooling_type': 'avg',
                'use_attention': True,
                'attention_type': 'multi_head'
            },
            TaskType.INSTANCE_SEGMENTATION: {
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
    def get_efficient_config(task_type: TaskType) -> Dict[str, Any]:
        """Get efficient (lightweight) configuration."""

        config = HeadConfiguration.get_default_config(task_type)
        config.update({
            'hidden_dim': 128,
            'dropout_rate': 0.0,
            'use_attention': False,
            'use_ffn': True,
            'ffn_type': 'glu',  # More efficient
            'ffn_expansion_factor': 2
        })
        return config

    @staticmethod
    def get_high_performance_config(task_type: TaskType) -> Dict[str, Any]:
        """Get high-performance configuration."""

        config = HeadConfiguration.get_default_config(task_type)
        config.update({
            'hidden_dim': 512,
            'dropout_rate': 0.2,
            'use_attention': True,
            'attention_type': 'differential',  # Advanced attention
            'use_ffn': True,
            'ffn_type': 'swiglu',  # Best performing FFN
            'ffn_expansion_factor': 8,
            'normalization_type': 'zero_centered_rms_norm'  # More stable
        })
        return config


# ---------------------------------------------------------------------