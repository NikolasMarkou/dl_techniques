"""
`YOLOv12FeatureExtractor` builds the backbone and neck of YOLOv12 and produces a
three-level feature pyramid that any detection, segmentation, or classification
head can consume.

The backbone stacks cross-stage partial blocks (C3k2, A2C2f) instead of plain
convolutions, so gradients reach early layers through a shortcut path while the
main path still deepens the features. The neck fuses scales twice: a top-down
pass (FPN) upsamples and concatenates deep features into shallow ones, then a
bottom-up pass (PAN) does the reverse, so every output level carries both fine
detail and global context.

The model takes a fixed `input_shape` at construction time and always returns
three feature maps at strides 8, 16, and 32 (P3, P4, P5). Scale is chosen from
`SCALE_CONFIGS` ('n', 's', 'm', 'l', 'x'), which sets both the channel width and
the block depth.

References:
    - Tian et al., 2025. YOLOv12: Attention-Centric Real-Time Object Detectors.
      (https://arxiv.org/abs/2502.12524) -- the backbone/neck this extracts.
    - Lin et al., 2017. Feature Pyramid Networks for Object Detection. CVPR
      2017. (https://arxiv.org/abs/1612.03144) -- the multi-scale P3/P4/P5
      pyramid the neck produces.
    - Liu et al., 2018. Path Aggregation Network for Instance Segmentation.
      CVPR 2018. (https://arxiv.org/abs/1803.01534) -- the bottom-up path the
      neck adds on top of FPN.
    - Wang et al., 2020. CSPNet: A New Backbone that can Enhance Learning
      Capability of CNN. (https://arxiv.org/abs/1911.11929) -- the cross-stage
      partial blocks the backbone stacks.
"""

import keras
from keras import ops
from typing import Optional, Tuple, Dict, Any, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.yolo12_blocks import (
    yolo12_conv_block,
    A2C2fBlock,
    C3k2Block,
)
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.yolo12.feature_extractor")
class YOLOv12FeatureExtractor(keras.Model):
    """YOLOv12 backbone and neck, producing a P3/P4/P5 feature pyramid.

    Architecture:

    .. code-block:: text

        input [B, H, W, 3]
          │
          ▼
        ┌─────────────┐
        │ stem1, stem2│  strides 2, 2
        └──────┬──────┘
               ▼
        ┌─────────────┐
        │ b1 (C3k2)   │
        └──────┬──────┘
               ▼ down1
        ┌─────────────┐
        │ b2 (C3k2)   │──────────────────┐ p3 (pre-neck)
        └──────┬──────┘                  │
               ▼ down2                   │
        ┌─────────────┐                  │
        │ b3 (A2C2f)  │───────────┐ p4    │
        └──────┬──────┘           │       │
               ▼ down3            │       │
        ┌─────────────┐           │       │
        │ b4 (A2C2f)  │ p5        │       │
        └──────┬──────┘           │       │
               ▼ up1               │       │
          concat(p4) ──► h1 (A2C2f)│       │
               │           │       │       │
               ▼ up2       │       │       │
          concat(p3) ──► h2 (A2C2f)────────┘
               │  = P3 out │       │
               ▼ down1     │       │
          concat(h1) ──► h3 (A2C2f)
               │  = P4 out │
               ▼ down2     │
          concat(p5) ──► h4 (C3k2)
                  = P5 out

    The top-down path (up1/up2) fuses deep features into shallow ones; the
    bottom-up path (neck_down1/neck_down2) fuses back the other way. P3 comes
    from the top-down path, P4 and P5 from the bottom-up path.

    :param input_shape: Input image shape ``(height, width, channels)``.
    :type input_shape: Tuple[int, int, int]
    :param scale: Scale key into ``SCALE_CONFIGS``, one of 'n', 's', 'm', 'l', 'x'.
    :type scale: str
    :param kernel_initializer: Weight initializer for all layers.
    :type kernel_initializer: str
    :param name: Model name.
    :type name: Optional[str]

    Input shape: ``(batch, height, width, channels)``.
    Output shape: three tensors ``[P3, P4, P5]`` at strides 8, 16, 32.
    """

    # Scale configurations: [depth_multiple, width_multiple]
    SCALE_CONFIGS = {
        "n": [0.50, 0.25],  # nano
        "s": [0.50, 0.50],  # small
        "m": [0.50, 1.00],  # medium
        "l": [1.00, 1.00],  # large
        "x": [1.00, 1.50],  # extra-large
    }

    # `MODEL_VARIANTS` is the canonical name across `models/` (see
    # `models/CLAUDE.md` § House Model Module Shape). `SCALE_CONFIGS` remains the
    # definition because `multitask.py` and the tests already read it by that
    # name; this is an alias to the same dict, not a copy.
    MODEL_VARIANTS = SCALE_CONFIGS

    @classmethod
    def from_variant(
            cls,
            variant: str,
            input_shape: Tuple[int, int, int] = (640, 640, 3),
            **kwargs: Any
    ) -> "YOLOv12FeatureExtractor":
        """Build a feature extractor from a named scale variant.

        :param variant: Scale key, one of 'n', 's', 'm', 'l', 'x'.
        :type variant: str
        :param input_shape: Input image shape ``(height, width, channels)``.
        :type input_shape: Tuple[int, int, int]
        :param kwargs: Extra constructor arguments.
        :return: A configured feature extractor.
        :rtype: YOLOv12FeatureExtractor
        :raises ValueError: If `variant` is not a key of ``MODEL_VARIANTS``.

        Example:
            >>> backbone = YOLOv12FeatureExtractor.from_variant("s")
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )
        return cls(input_shape=input_shape, scale=variant, **kwargs)

    def __init__(
            self,
            input_shape: Tuple[int, int, int] = (640, 640, 3),
            scale: str = "n",
            kernel_initializer: str = "he_normal",
            name: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        """Initialize the feature extractor.

        :param input_shape: Input image shape ``(height, width, channels)``.
        :type input_shape: Tuple[int, int, int]
        :param scale: Model scale ('n', 's', 'm', 'l', 'x').
        :type scale: str
        :param kernel_initializer: Weight initializer.
        :type kernel_initializer: str
        :param name: Model name.
        :type name: Optional[str]
        :param kwargs: Extra keyword arguments passed to ``keras.Model``.
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

        # Base channel counts, scaled by width_multiple below.
        base_filters = {
            'c1': 64, 'c2': 128, 'c3': 256,
            'c4': 512, 'c5': 512, 'c6': 1024
        }
        self.filters = {k: int(v * self.width_multiple) for k, v in base_filters.items()}

        # Block repeat counts, scaled by depth_multiple, floored at 1.
        self.n_c3k2_1 = max(round(2 * self.depth_multiple), 1)
        self.n_c3k2_2 = max(round(2 * self.depth_multiple), 1)
        self.n_a2c2f_1 = max(round(4 * self.depth_multiple), 1)
        self.n_a2c2f_2 = max(round(4 * self.depth_multiple), 1)
        self.n_a2c2f_head = max(round(2 * self.depth_multiple), 1)
        self.n_c3k2_head = max(round(2 * self.depth_multiple), 1)

        self._build_input_shape = None
        self._layers_built = False

        logger.info(f"Created YOLOv12FeatureExtractor-{scale}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the feature extractor and materialize every sub-layer weight.

        ``_build_layers`` only instantiates the sub-layers; each one otherwise
        creates its variables lazily on first call, which would leave them
        unbuilt and drop their weights on a ``.keras`` reload. Running one
        dummy forward here materializes every variable before that can happen.

        :param input_shape: Input tensor shape ``(B, H, W, C)``.
        :type input_shape: Tuple[Optional[int], ...]
        """
        if self._layers_built:
            return

        self._build_input_shape = input_shape
        self._build_layers()

        # Calls the sub-layers directly, not self(), so this cannot recurse into build().
        dummy_shape = (1,) + tuple(
            int(d) if d is not None else 32 for d in input_shape[1:]
        )
        self._forward(ops.zeros(dummy_shape), training=False)

        self._layers_built = True
        super().build(input_shape)

    def _build_layers(self) -> None:
        """Instantiate every backbone and neck sub-layer."""
        self.stem1 = yolo12_conv_block(
            filters=self.filters['c1'],
            kernel_size=3,
            strides=2,
            kernel_initializer=self.kernel_initializer,
            name="backbone_stem_1"
        )

        self.stem2 = yolo12_conv_block(
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

        self.down1 = yolo12_conv_block(
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

        self.down2 = yolo12_conv_block(
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

        self.down3 = yolo12_conv_block(
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
            area=1,
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
            area=1,
            kernel_initializer=self.kernel_initializer,
            name="neck_h2"
        )

        self.neck_down1 = yolo12_conv_block(
            filters=self.filters['c3'],
            kernel_size=3,
            strides=2,
            kernel_initializer=self.kernel_initializer,
            name="neck_down1"
        )

        self.h3 = A2C2fBlock(
            filters=self.filters['c5'],
            n=self.n_a2c2f_head,
            area=1,
            kernel_initializer=self.kernel_initializer,
            name="neck_h3"
        )

        self.neck_down2 = yolo12_conv_block(
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
        """Run the backbone and neck forward pass.

        :param inputs: Input tensor ``(batch, height, width, channels)``.
        :type inputs: keras.KerasTensor
        :param training: Whether the call runs in training mode.
        :type training: Optional[bool]
        :return: Three feature maps ``[P3, P4, P5]`` at strides 8, 16, 32.
        :rtype: List[keras.KerasTensor]
        """
        return self._forward(inputs, training=training)

    def _forward(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> List[keras.KerasTensor]:
        """Run the shared backbone/neck computation used by both ``call`` and
        ``build``'s dummy forward.

        :param inputs: Input tensor ``(batch, height, width, channels)``.
        :type inputs: keras.KerasTensor
        :param training: Whether the call runs in training mode.
        :type training: Optional[bool]
        :return: Three feature maps ``[P3, P4, P5]`` at strides 8, 16, 32.
        :rtype: List[keras.KerasTensor]
        """
        x = self.stem1(inputs, training=training)
        x = self.stem2(x, training=training)
        x = self.b1(x, training=training)

        p3 = self.down1(x, training=training)
        p3 = self.b2(p3, training=training)

        p4 = self.down2(p3, training=training)
        p4 = self.b3(p4, training=training)

        p5 = self.down3(p4, training=training)
        p5 = self.b4(p5, training=training)

        # Top-down path: fuse deep features into shallow ones.
        x = self.up1(p5)
        x = ops.concatenate([x, p4], axis=-1)
        h1 = self.h1(x, training=training)

        x = self.up2(h1)
        x = ops.concatenate([x, p3], axis=-1)
        h2 = self.h2(x, training=training)

        # Bottom-up path: fuse back, producing the final P4/P5 outputs.
        x = self.neck_down1(h2, training=training)
        x = ops.concatenate([x, h1], axis=-1)
        h3 = self.h3(x, training=training)

        x = self.neck_down2(h3, training=training)
        x = ops.concatenate([x, p5], axis=-1)
        h4 = self.h4(x, training=training)

        return [h2, h3, h4]

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        """Compute the output shapes of the three feature maps.

        :param input_shape: Input tensor shape.
        :type input_shape: Tuple[int, ...]
        :return: Output shapes for ``[P3, P4, P5]``.
        :rtype: List[Tuple[int, ...]]
        """
        batch_size = input_shape[0]
        height, width = input_shape[1], input_shape[2]

        p3_h, p3_w = height // 8, width // 8
        p4_h, p4_w = height // 16, width // 16
        p5_h, p5_w = height // 32, width // 32

        return [
            (batch_size, p3_h, p3_w, self.filters['c3']),
            (batch_size, p4_h, p4_w, self.filters['c5']),
            (batch_size, p5_h, p5_w, self.filters['c6']),
        ]

    def get_config(self) -> Dict[str, Any]:
        """Return the config needed to reconstruct this model."""
        config = super().get_config()
        config.update({
            "input_shape": self.input_shape_config,
            "scale": self.scale,
            "kernel_initializer": self.kernel_initializer,
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Return the shape needed to rebuild this model."""
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Rebuild this model from a `get_build_config` result."""
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "YOLOv12FeatureExtractor":
        """Reconstruct a model instance from a `get_config` result."""
        return cls(**config)

# ---------------------------------------------------------------------

def create_yolov12_feature_extractor(
        input_shape: Tuple[int, int, int] = (640, 640, 3),
        scale: str = "n",
        **kwargs
) -> YOLOv12FeatureExtractor:
    """Create a YOLOv12 feature extractor.

    :param input_shape: Input image shape.
    :type input_shape: Tuple[int, int, int]
    :param scale: Model scale.
    :type scale: str
    :param kwargs: Extra arguments passed to ``YOLOv12FeatureExtractor``.
    :return: A configured feature extractor.
    :rtype: YOLOv12FeatureExtractor

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
