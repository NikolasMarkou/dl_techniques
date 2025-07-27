"""
YOLOv12 Feature Extractor Implementation

This module provides the backbone and neck components of the YOLOv12 architecture
as a standalone feature extractor. This base model can be used by different task-specific
models for object detection, segmentation, classification, etc.

The feature extractor outputs multiscale feature maps that can be consumed by
various task-specific heads.
"""

import keras
from keras import ops
from typing import Optional, Tuple, Dict, Any, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.yolo12 import (
    ConvBlock,
    A2C2fBlock,
    C3k2Block,
)

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class YOLOv12FeatureExtractor(keras.Model):
    """
    YOLOv12 Feature Extractor (Backbone + Neck).

    This model contains the backbone and neck components of YOLOv12 that extract
    multi-scale features from input images. The output feature maps can be used
    by various task-specific heads.

    Args:
        input_shape: Input image shape (height, width, channels).
        scale: Model scale configuration ('n', 's', 'm', 'l', 'x').
        kernel_initializer: Weight initializer for all layers.
        name: Model name.

    Returns:
        List of three feature maps at different scales [P3, P4, P5].
    """

    # Scale configurations: [depth_multiple, width_multiple]
    SCALE_CONFIGS = {
        "n": [0.50, 0.25],  # nano
        "s": [0.50, 0.50],  # small
        "m": [0.50, 1.00],  # medium
        "l": [1.00, 1.00],  # large
        "x": [1.00, 1.50],  # extra-large
    }

    def __init__(
            self,
            input_shape: Tuple[int, int, int] = (640, 640, 3),
            scale: str = "n",
            kernel_initializer: str = "he_normal",
            name: Optional[str] = None,
            **kwargs
    ):
        """
        Initialize YOLOv12 feature extractor.

        Args:
            input_shape: Input image shape (height, width, channels).
            scale: Model scale ('n', 's', 'm', 'l', 'x').
            kernel_initializer: Weight initializer.
            name: Model name.
            **kwargs: Additional keyword arguments.
        """
        if name is None:
            name = f"yolov12_feature_extractor_{scale}"
        super().__init__(name=name, **kwargs)

        self.input_shape_config = input_shape
        self.scale = scale
        self.kernel_initializer = kernel_initializer

        if scale not in self.SCALE_CONFIGS:
            raise ValueError(
                f"Unsupported scale: {scale}. Choose from {list(self.SCALE_CONFIGS.keys())}"
            )

        self.depth_multiple, self.width_multiple = self.SCALE_CONFIGS[scale]

        # Calculate filter numbers based on scale
        base_filters = {
            'c1': 64, 'c2': 128, 'c3': 256,
            'c4': 512, 'c5': 512, 'c6': 1024
        }
        self.filters = {k: int(v * self.width_multiple) for k, v in base_filters.items()}

        # Calculate layer repetitions based on depth
        self.n_c3k2_1 = max(round(2 * self.depth_multiple), 1)
        self.n_c3k2_2 = max(round(2 * self.depth_multiple), 1)
        self.n_a2c2f_1 = max(round(4 * self.depth_multiple), 1)
        self.n_a2c2f_2 = max(round(4 * self.depth_multiple), 1)
        self.n_a2c2f_head = max(round(2 * self.depth_multiple), 1)
        self.n_c3k2_head = max(round(2 * self.depth_multiple), 1)

        # Store build state for serialization
        self._build_input_shape = None

        # Initialize layers (will be built in build())
        self._layers_built = False

        logger.info(f"Created YOLOv12FeatureExtractor-{scale}")

    def build(self, input_shape):
        """Build the feature extractor layers."""
        if self._layers_built:
            return

        self._build_input_shape = input_shape
        self._build_layers()
        self._layers_built = True
        super().build(input_shape)

    def _build_layers(self) -> None:
        """Initialize all backbone and neck layers."""
        # Backbone stem
        self.stem1 = ConvBlock(
            filters=self.filters['c1'],
            kernel_size=3,
            strides=2,
            kernel_initializer=self.kernel_initializer,
            name="backbone_stem_1"
        )

        self.stem2 = ConvBlock(
            filters=self.filters['c2'],
            kernel_size=3,
            strides=2,
            groups=2,
            kernel_initializer=self.kernel_initializer,
            name="backbone_stem_2"
        )

        # Backbone blocks
        self.b1 = C3k2Block(
            filters=self.filters['c3'],
            n=self.n_c3k2_1,
            shortcut=False,
            kernel_initializer=self.kernel_initializer,
            name="backbone_b1"
        )

        self.down1 = ConvBlock(
            filters=self.filters['c3'],
            kernel_size=3,
            strides=2,
            groups=4,
            kernel_initializer=self.kernel_initializer,
            name="backbone_down1"
        )

        self.b2 = C3k2Block(
            filters=self.filters['c4'],
            n=self.n_c3k2_2,
            shortcut=False,
            kernel_initializer=self.kernel_initializer,
            name="backbone_b2"
        )

        self.down2 = ConvBlock(
            filters=self.filters['c5'],
            kernel_size=3,
            strides=2,
            kernel_initializer=self.kernel_initializer,
            name="backbone_down2"
        )

        self.b3 = A2C2fBlock(
            filters=self.filters['c5'],
            n=self.n_a2c2f_1,
            area=4,
            kernel_initializer=self.kernel_initializer,
            name="backbone_b3"
        )

        self.down3 = ConvBlock(
            filters=self.filters['c6'],
            kernel_size=3,
            strides=2,
            kernel_initializer=self.kernel_initializer,
            name="backbone_down3"
        )

        self.b4 = A2C2fBlock(
            filters=self.filters['c6'],
            n=self.n_a2c2f_2,
            area=1,
            kernel_initializer=self.kernel_initializer,
            name="backbone_b4"
        )

        # Neck (PAN) layers
        self.up1 = keras.layers.UpSampling2D(
            size=2,
            interpolation="nearest",
            name="neck_up1"
        )

        self.h1 = A2C2fBlock(
            filters=self.filters['c5'],
            n=self.n_a2c2f_head,
            area=-1,  # Adaptive area
            kernel_initializer=self.kernel_initializer,
            name="neck_h1"
        )

        self.up2 = keras.layers.UpSampling2D(
            size=2,
            interpolation="nearest",
            name="neck_up2"
        )

        self.h2 = A2C2fBlock(
            filters=self.filters['c3'],
            n=self.n_a2c2f_head,
            area=-1,
            kernel_initializer=self.kernel_initializer,
            name="neck_h2"
        )

        self.neck_down1 = ConvBlock(
            filters=self.filters['c3'],
            kernel_size=3,
            strides=2,
            kernel_initializer=self.kernel_initializer,
            name="neck_down1"
        )

        self.h3 = A2C2fBlock(
            filters=self.filters['c5'],
            n=self.n_a2c2f_head,
            area=-1,
            kernel_initializer=self.kernel_initializer,
            name="neck_h3"
        )

        self.neck_down2 = ConvBlock(
            filters=self.filters['c5'],
            kernel_size=3,
            strides=2,
            kernel_initializer=self.kernel_initializer,
            name="neck_down2"
        )

        self.h4 = C3k2Block(
            filters=self.filters['c6'],
            n=self.n_c3k2_head,
            shortcut=True,
            kernel_initializer=self.kernel_initializer,
            name="neck_h4"
        )

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> List[keras.KerasTensor]:
        """
        Forward pass through feature extractor.

        Args:
            inputs: Input tensor (batch_size, height, width, channels).
            training: Whether in training mode.

        Returns:
            List of three feature maps [P3, P4, P5] at different scales.
        """
        # Backbone forward pass
        x = self.stem1(inputs, training=training)
        x = self.stem2(x, training=training)
        x = self.b1(x, training=training)

        # Extract P3 features
        p3 = self.down1(x, training=training)
        p3 = self.b2(p3, training=training)

        # Extract P4 features
        p4 = self.down2(p3, training=training)
        p4 = self.b3(p4, training=training)

        # Extract P5 features
        p5 = self.down3(p4, training=training)
        p5 = self.b4(p5, training=training)

        # Neck forward pass - Top-down path
        x = self.up1(p5)
        x = ops.concatenate([x, p4], axis=-1)
        h1 = self.h1(x, training=training)

        x = self.up2(h1)
        x = ops.concatenate([x, p3], axis=-1)
        h2 = self.h2(x, training=training)  # P3 output

        # Neck forward pass - Bottom-up path
        x = self.neck_down1(h2, training=training)
        x = ops.concatenate([x, h1], axis=-1)
        h3 = self.h3(x, training=training)  # P4 output

        x = self.neck_down2(h3, training=training)
        x = ops.concatenate([x, p5], axis=-1)
        h4 = self.h4(x, training=training)  # P5 output

        # Return multi-scale feature maps
        return [h2, h3, h4]  # [P3, P4, P5]

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        """
        Compute output shapes for the three feature maps.

        Args:
            input_shape: Input tensor shape.

        Returns:
            List of output shapes for [P3, P4, P5].
        """
        batch_size = input_shape[0]
        height, width = input_shape[1], input_shape[2]

        # Calculate output dimensions based on downsampling
        p3_h, p3_w = height // 8, width // 8
        p4_h, p4_w = height // 16, width // 16
        p5_h, p5_w = height // 32, width // 32

        return [
            (batch_size, p3_h, p3_w, self.filters['c3']),  # P3
            (batch_size, p4_h, p4_w, self.filters['c5']),  # P4
            (batch_size, p5_h, p5_w, self.filters['c6']),  # P5
        ]

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        config = super().get_config()
        config.update({
            "input_shape": self.input_shape_config,
            "scale": self.scale,
            "kernel_initializer": self.kernel_initializer,
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get build configuration for serialization."""
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build model from configuration."""
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "YOLOv12FeatureExtractor":
        """Create model from configuration."""
        return cls(**config)


def create_yolov12_feature_extractor(
        input_shape: Tuple[int, int, int] = (640, 640, 3),
        scale: str = "n",
        **kwargs
) -> YOLOv12FeatureExtractor:
    """
    Create a YOLOv12 feature extractor with specified configuration.

    Args:
        input_shape: Input image shape.
        scale: Model scale.
        **kwargs: Additional arguments for YOLOv12FeatureExtractor.

    Returns:
        YOLOv12FeatureExtractor model instance.

    Example:
        >>> extractor = create_yolov12_feature_extractor(
        ...     input_shape=(256, 256, 3),
        ...     scale="s"
        ... )
        >>> features = extractor(images)  # Returns [P3, P4, P5]
    """
    extractor = YOLOv12FeatureExtractor(
        input_shape=input_shape,
        scale=scale,
        **kwargs
    )

    logger.info(f"YOLOv12FeatureExtractor-{scale} created successfully")
    return extractor

# ---------------------------------------------------------------------
