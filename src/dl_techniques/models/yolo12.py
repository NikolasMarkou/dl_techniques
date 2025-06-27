"""
YOLOv12 Model Implementation for Keras 3.

This module provides a complete implementation of the YOLOv12 architecture
including all custom blocks and layers. The model supports different scale
configurations (nano, small, medium, large, extra-large) and includes
area-attention mechanisms for improved feature extraction.

The implementation follows Keras 3 best practices with proper serialization,
type hints, and comprehensive documentation.

Architecture Overview:
    1. Backbone: Feature extraction with ConvNeXt-style blocks
    2. Neck: PAN (Path Aggregation Network) with attention mechanisms
    3. Head: Multi-scale detection head with classification and bbox regression

References:
    - YOLOv12: Real-Time Object Detection with Enhanced Architecture
"""

import keras
from keras import layers, ops
from typing import Optional, Tuple, Dict, Any

from dl_techniques.utils.logger import logger
from dl_techniques.layers.yolo12 import (
    ConvBlock,
    A2C2fBlock,
    C3k2Block,
)
from dl_techniques.layers.yolo12_heads import YOLOv12DetectionHead

@keras.saving.register_keras_serializable()
class YOLOv12(keras.Model):
    """YOLOv12 Object Detection Model.

    Complete implementation of YOLOv12 with backbone, neck, and detection head.
    Supports different model scales and includes area-attention mechanisms.

    Args:
        num_classes: Number of object classes.
        input_shape: Input image shape (height, width, channels).
        scale: Model scale ('n', 's', 'm', 'l', 'x').
        reg_max: Maximum value for DFL regression.
        kernel_initializer: Weight initializer.
        name: Model name.
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
            num_classes: int = 80,
            input_shape: Tuple[int, int, int] = (640, 640, 3),
            scale: str = "n",
            reg_max: int = 16,
            kernel_initializer: str = "he_normal",
            name: Optional[str] = None,
            **kwargs
    ):
        if name is None:
            name = f"yolov12_{scale}"
        super().__init__(name=name, **kwargs)

        self.num_classes = num_classes
        self.input_shape_config = input_shape
        self.scale = scale
        self.reg_max = reg_max
        self.kernel_initializer = kernel_initializer

        if scale not in self.SCALE_CONFIGS:
            raise ValueError(f"Unsupported scale: {scale}. Choose from {list(self.SCALE_CONFIGS.keys())}")

        self.depth_multiple, self.width_multiple = self.SCALE_CONFIGS[scale]

        # Calculate filter numbers based on scale
        base_filters = {'c1': 64, 'c2': 128, 'c3': 256, 'c4': 512, 'c5': 512, 'c6': 1024}
        self.filters = {k: int(v * self.width_multiple) for k, v in base_filters.items()}

        # Calculate layer repetitions based on depth
        self.n_c3k2_1 = max(round(2 * self.depth_multiple), 1)
        self.n_c3k2_2 = max(round(2 * self.depth_multiple), 1)
        self.n_a2c2f_1 = max(round(4 * self.depth_multiple), 1)
        self.n_a2c2f_2 = max(round(4 * self.depth_multiple), 1)
        self.n_a2c2f_head = max(round(2 * self.depth_multiple), 1)
        self.n_c3k2_head = max(round(2 * self.depth_multiple), 1)

        # Initialize layers
        self._build_layers()

        logger.info(f"Created YOLOv12-{scale} with {num_classes} classes")

    def _build_layers(self) -> None:
        """Initialize all model layers."""
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

        # Neck (PAN)
        self.up1 = layers.UpSampling2D(size=2, interpolation="nearest", name="neck_up1")
        self.h1 = A2C2fBlock(
            filters=self.filters['c5'],
            n=self.n_a2c2f_head,
            area=-1,  # Adaptive area
            kernel_initializer=self.kernel_initializer,
            name="neck_h1"
        )

        self.up2 = layers.UpSampling2D(size=2, interpolation="nearest", name="neck_up2")
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

        # Detection head
        self.detect_head = YOLOv12DetectionHead(
            num_classes=self.num_classes,
            reg_max=self.reg_max,
            kernel_initializer=self.kernel_initializer,
            name="detect_head"
        )

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through YOLOv12."""
        # Backbone
        x = self.stem1(inputs, training=training)
        x = self.stem2(x, training=training)
        x = self.b1(x, training=training)

        p3 = self.down1(x, training=training)
        p3 = self.b2(p3, training=training)

        p4 = self.down2(p3, training=training)
        p4 = self.b3(p4, training=training)

        p5 = self.down3(p4, training=training)
        p5 = self.b4(p5, training=training)

        # Neck - Top-down path
        x = self.up1(p5)
        x = ops.concatenate([x, p4], axis=-1)
        h1 = self.h1(x, training=training)

        x = self.up2(h1)
        x = ops.concatenate([x, p3], axis=-1)
        h2 = self.h2(x, training=training)

        # Neck - Bottom-up path
        x = self.neck_down1(h2, training=training)
        x = ops.concatenate([x, h1], axis=-1)
        h3 = self.h3(x, training=training)

        x = self.neck_down2(h3, training=training)
        x = ops.concatenate([x, p5], axis=-1)
        h4 = self.h4(x, training=training)

        # Detection head
        outputs = self.detect_head([h2, h3, h4], training=training)

        return outputs

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "input_shape": self.input_shape_config,
            "scale": self.scale,
            "reg_max": self.reg_max,
            "kernel_initializer": self.kernel_initializer,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "YOLOv12":
        """Create model from configuration."""
        return cls(**config)


def create_yolov12(
        num_classes: int = 80,
        input_shape: Tuple[int, int, int] = (640, 640, 3),
        scale: str = "n",
        **kwargs
) -> YOLOv12:
    """Create a YOLOv12 model with specified configuration.

    Args:
        num_classes: Number of object classes.
        input_shape: Input image shape.
        scale: Model scale.
        **kwargs: Additional arguments for YOLOv12.

    Returns:
        YOLOv12 model instance.
    """
    model = YOLOv12(
        num_classes=num_classes,
        input_shape=input_shape,
        scale=scale,
        **kwargs
    )

    logger.info(f"YOLOv12-{scale} model created successfully")
    return model