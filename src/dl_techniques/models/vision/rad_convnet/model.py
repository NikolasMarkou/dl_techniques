"""RADConvNet: a ConvNeXt-shaped classification backbone built on RAD-Conv.

Swaps ConvNeXt's fixed depthwise K×K convolution for
:class:`~dl_techniques.layers.conv_blocks.rad_conv.RADConv2D` -- an
axis-aligned, input-adaptive region-integration operator (Maleki & Imani,
2025) -- as the spatial mixer inside each block, keeping the surrounding
ConvNeXt-style scaffold (patchify stem, staged blocks, LayerNorm +
strided-conv downsampling, inverted-bottleneck MLP, GAP + Dense head)
unchanged. Each block is transform-only (returns ``F(x)``); the residual add
is owned by :meth:`RADConvNet.call`, matching the house convention set by
``ConvNextV1Block``/``ConvNeXtV1``.

References:
    - Maleki & Imani, 2025. Region-Aware Deformable Convolutions.
      (https://arxiv.org/abs/2509.15436)
    - Liu et al., 2022. A ConvNet for the 2020s. (https://arxiv.org/abs/2201.03545)
"""

import keras
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.conv_blocks.rad_conv import RADConv2D
from dl_techniques.utils.activation_serialization import (
    serialize_activation,
    deserialize_activation,
)
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.rad_convnet.model")
class RADConvNet(keras.Model):
    """RADConvNet: ConvNeXt-style backbone with RAD-Conv spatial mixing.

    Architecture:

    .. code-block:: text

        Input [B, H, W, C_in]
              │
              ▼
        Stem: Conv S×S /S -> LayerNorm    (S = strides, patchify)
              │
              ▼
        Stage 0: D0 x Block(dims[0])
              │
        Downsample: LayerNorm -> Conv 2x2 /2
              │
        Stage 1: D1 x Block(dims[1])
              │
             ...
              │
        GAP -> LayerNorm -> Dense(num_classes)   (if include_top)

    Block internals (transform-only, returns F(x); residual owned by `call`):

    .. code-block:: text

        x -> RADConv2D(K, groups) -> LayerNorm
              -> Dense 1x1 (4x expand) -> activation -> Dense 1x1 (reduce)
              -> F(x)

    RAD-Conv predicts, per kernel element and channel group, four boundary
    offsets that define an axis-aligned integration rectangle -- so even a
    ``rad_kernel_size=1`` block can aggregate over a receptive field that
    spans the whole feature map, unlike a standard 1x1 convolution.

    :param num_classes: Number of output classes; only used when
        ``include_top=True``. ``0`` returns pooled, normalized features.
        Defaults to ``1000``.
    :type num_classes: int
    :param depths: Number of blocks per stage. ``None`` resolves to
        ``[2, 2, 4, 2]``.
    :type depths: Optional[List[int]]
    :param dims: Channel count per stage. ``None`` resolves to
        ``[32, 64, 128, 256]`` -- deliberately smaller than ConvNeXt's
        ``[96, 192, 384, 768]`` defaults: RAD-Conv's exact region integration
        is materially more expensive per channel than a depthwise conv.
    :type dims: Optional[List[int]]
    :param rad_kernel_size: RAD-Conv kernel size (number of kernel elements
        ``K = kh * kw``) used inside every block. Defaults to ``3``.
    :type rad_kernel_size: Union[int, Tuple[int, int]]
    :param rad_groups: RAD-Conv channel groups used inside every block.
        Defaults to ``1``.
    :type rad_groups: int
    :param mlp_ratio: Inverted-bottleneck expansion ratio of the pointwise
        MLP following each RAD-Conv. Defaults to ``4``.
    :type mlp_ratio: int
    :param activation: Activation used inside each block's MLP. Defaults to
        ``"gelu"``.
    :type activation: str
    :param use_bias: Whether convolutions/dense layers use a bias and whether
        LayerNormalization centers. Defaults to ``True``.
    :type use_bias: bool
    :param dropout_rate: Dropout rate applied after the MLP's expansion
        activation. Defaults to ``0.0``.
    :type dropout_rate: float
    :param strides: Patchify factor, used as both kernel size and stride of
        the stem convolution and of every inter-stage downsample. Defaults
        to ``4``.
    :type strides: int
    :param include_top: Whether to include the GAP + LayerNorm + Dense head.
        Defaults to ``True``.
    :type include_top: bool
    :param input_shape: ``(height, width, channels)``, excluding batch.
        ``None`` resolves to ``(None, None, 3)``.
    :type input_shape: Tuple[int, ...]
    :param kwargs: Additional keyword arguments for the ``keras.Model`` base
        class.

    :raises ValueError: If ``depths``/``dims`` differ in length, ``strides``
        is not positive, or ``input_shape`` is not 3D.

    Input shape:
        4D tensor ``(batch_size, height, width, channels)``.

    Output shape:
        - ``include_top=True``: ``(batch_size, num_classes)``.
        - ``include_top=True, num_classes=0``: ``(batch_size, dims[-1])``.
        - ``include_top=False``: ``(batch_size, H', W', dims[-1])``.
    """

    MODEL_VARIANTS = {
        "tiny": {"depths": [2, 2, 4, 2], "dims": [32, 64, 128, 256]},
        "small": {"depths": [3, 3, 9, 3], "dims": [48, 96, 192, 384]},
        "base": {"depths": [3, 3, 9, 3], "dims": [64, 128, 256, 512]},
    }

    LAYERNORM_EPSILON = 1e-6
    STEM_INITIALIZER = "truncated_normal"
    HEAD_INITIALIZER = "truncated_normal"

    def __init__(
        self,
        num_classes: int = 1000,
        depths: Optional[List[int]] = None,
        dims: Optional[List[int]] = None,
        rad_kernel_size: Union[int, Tuple[int, int]] = 3,
        rad_groups: int = 1,
        mlp_ratio: int = 4,
        activation: str = "gelu",
        use_bias: bool = True,
        dropout_rate: float = 0.0,
        strides: int = 4,
        include_top: bool = True,
        input_shape: Tuple[Optional[int], ...] = (None, None, 3),
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        depths = list(depths) if depths is not None else [2, 2, 4, 2]
        dims = list(dims) if dims is not None else [32, 64, 128, 256]

        if len(depths) != len(dims):
            raise ValueError(
                f"Length of depths ({len(depths)}) must equal length of dims ({len(dims)})"
            )
        if strides <= 0:
            raise ValueError(f"strides {strides} must be positive")
        if input_shape is None:
            input_shape = (None, None, 3)
        if len(input_shape) != 3:
            raise ValueError(f"input_shape must be 3D, got {input_shape}")

        self.num_classes = num_classes
        self.depths = depths
        self.dims = dims
        self.rad_kernel_size = rad_kernel_size
        self.rad_groups = rad_groups
        self.mlp_ratio = mlp_ratio
        self.activation_name = deserialize_activation(activation)
        self.activation_layer = keras.layers.Activation(self.activation_name, name="mlp_activation")
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.strides = strides
        self.include_top = include_top
        self.input_shape_config = tuple(input_shape)

        self._build_stem()

        self.downsample_layers_list: List[Tuple[keras.layers.Layer, keras.layers.Layer]] = []
        self.stages_list: List[List[Dict[str, keras.layers.Layer]]] = []
        for stage_idx in range(len(self.depths)):
            if stage_idx > 0:
                self._build_downsample_layer(stage_idx)
            self._build_stage(stage_idx)

        if self.include_top:
            self._build_head()

        logger.info(
            f"Created RADConvNet for input {self.input_shape_config} "
            f"with {sum(depths)} blocks"
        )

    def _build_stem(self) -> None:
        """Build the patchify stem: strided conv + LayerNorm."""
        self.stem_conv = keras.layers.Conv2D(
            filters=self.dims[0],
            kernel_size=self.strides,
            strides=self.strides,
            padding="same" if self.strides == 1 else "valid",
            use_bias=self.use_bias,
            kernel_initializer=self.STEM_INITIALIZER,
            name="stem_conv",
        )
        self.stem_norm = keras.layers.LayerNormalization(
            epsilon=self.LAYERNORM_EPSILON, center=self.use_bias, scale=True, name="stem_norm"
        )

    def _build_downsample_layer(self, stage_idx: int) -> None:
        """Build the LayerNorm + strided-conv downsample feeding ``stage_idx``."""
        norm = keras.layers.LayerNormalization(
            epsilon=self.LAYERNORM_EPSILON,
            center=self.use_bias,
            scale=True,
            name=f"downsample_norm_{stage_idx - 1}",
        )
        conv = keras.layers.Conv2D(
            filters=self.dims[stage_idx],
            kernel_size=2,
            strides=2,
            padding="same",
            use_bias=self.use_bias,
            name=f"downsample_conv_{stage_idx - 1}",
        )
        self.downsample_layers_list.append((norm, conv))

    def _build_stage(self, stage_idx: int) -> None:
        """Build the blocks of stage ``stage_idx``.

        Each block is ``RADConv2D -> LayerNorm -> Dense(4x) -> activation
        -> [Dropout] -> Dense(reduce)``, transform-only -- the residual add
        happens in :meth:`call`.
        """
        dim = self.dims[stage_idx]
        blocks = []
        for block_idx in range(self.depths[stage_idx]):
            prefix = f"stage{stage_idx}_block{block_idx}"
            block = {
                "rad_conv": RADConv2D(
                    filters=dim,
                    kernel_size=self.rad_kernel_size,
                    groups=self.rad_groups,
                    use_bias=self.use_bias,
                    name=f"{prefix}_rad_conv",
                ),
                "norm": keras.layers.LayerNormalization(
                    epsilon=self.LAYERNORM_EPSILON,
                    center=self.use_bias,
                    scale=True,
                    name=f"{prefix}_norm",
                ),
                "expand": keras.layers.Dense(
                    dim * self.mlp_ratio, use_bias=self.use_bias, name=f"{prefix}_expand"
                ),
                "reduce": keras.layers.Dense(dim, use_bias=self.use_bias, name=f"{prefix}_reduce"),
                "dropout": (
                    keras.layers.Dropout(self.dropout_rate, name=f"{prefix}_dropout")
                    if self.dropout_rate > 0.0
                    else None
                ),
            }
            blocks.append(block)
        self.stages_list.append(blocks)

    def _build_head(self) -> None:
        """Build the GAP + LayerNorm + Dense classification head."""
        self.gap = keras.layers.GlobalAveragePooling2D(name="gap")
        self.head_norm = keras.layers.LayerNormalization(
            epsilon=self.LAYERNORM_EPSILON, center=self.use_bias, scale=True, name="head_norm"
        )
        self.classifier = (
            keras.layers.Dense(
                self.num_classes,
                use_bias=self.use_bias,
                kernel_initializer=self.HEAD_INITIALIZER,
                name="classifier",
            )
            if self.num_classes > 0
            else None
        )

    def build(self, input_shape: Any) -> None:
        """Materialize every sub-layer by tracing ``call`` on a symbolic input.

        :param input_shape: Shape of the input to ``call``, with or without
            the batch dimension.
        :type input_shape: Any
        """
        if len(input_shape) == 3:
            build_shape = (None,) + tuple(input_shape)
        else:
            build_shape = input_shape
        dummy_input = keras.KerasTensor(build_shape)
        _ = self.call(dummy_input)
        super().build(input_shape)

    def call(
        self,
        inputs: "keras.KerasTensor",
        training: Optional[bool] = None,
    ) -> "keras.KerasTensor":
        """Forward pass.

        :param inputs: ``(batch_size, height, width, channels)`` input.
        :type inputs: keras.KerasTensor
        :param training: Whether the model is in training mode.
        :type training: Optional[bool]
        :return: Output tensor; shape per the class docstring.
        :rtype: keras.KerasTensor
        """
        x = self.stem_conv(inputs)
        x = self.stem_norm(x)

        for stage_idx, stage_blocks in enumerate(self.stages_list):
            if stage_idx > 0:
                norm_layer, conv_layer = self.downsample_layers_list[stage_idx - 1]
                x = norm_layer(x)
                x = conv_layer(x)

            for block in stage_blocks:
                residual = x
                y = block["rad_conv"](x)
                y = block["norm"](y)
                y = block["expand"](y)
                y = self.activation_layer(y)
                if block["dropout"] is not None:
                    y = block["dropout"](y, training=training)
                y = block["reduce"](y)
                x = keras.layers.add([residual, y])

        if self.include_top:
            x = self.gap(x)
            x = self.head_norm(x)
            if self.classifier is not None:
                x = self.classifier(x)

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape.

        :param input_shape: Input shape, channels-last.
        :type input_shape: tuple
        :return: Output shape.
        :rtype: tuple
        """
        current_shape = input_shape
        current_shape = self.stem_conv.compute_output_shape(current_shape)
        current_shape = self.stem_norm.compute_output_shape(current_shape)

        for stage_idx in range(len(self.depths)):
            if stage_idx > 0:
                norm_layer, conv_layer = self.downsample_layers_list[stage_idx - 1]
                current_shape = norm_layer.compute_output_shape(current_shape)
                current_shape = conv_layer.compute_output_shape(current_shape)
            # RAD-Conv blocks are shape-preserving (stride 1, "same"-style).
            current_shape = current_shape[:-1] + (self.dims[stage_idx],)

        if self.include_top:
            current_shape = self.gap.compute_output_shape(current_shape)
            current_shape = self.head_norm.compute_output_shape(current_shape)
            if self.classifier is not None:
                current_shape = self.classifier.compute_output_shape(current_shape)

        return current_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the model configuration for serialization.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = {
            "num_classes": self.num_classes,
            "depths": self.depths,
            "dims": self.dims,
            "rad_kernel_size": self.rad_kernel_size,
            "rad_groups": self.rad_groups,
            "mlp_ratio": self.mlp_ratio,
            "activation": serialize_activation(self.activation_name),
            "use_bias": self.use_bias,
            "dropout_rate": self.dropout_rate,
            "strides": self.strides,
            "include_top": self.include_top,
            "input_shape": self.input_shape_config,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RADConvNet":
        """Create a model instance from its configuration.

        :param config: Configuration dictionary.
        :type config: Dict[str, Any]
        :return: RADConvNet model instance.
        :rtype: RADConvNet
        """
        return cls(**config)

    @classmethod
    def from_variant(
        cls,
        variant: str,
        num_classes: int = 1000,
        input_shape: Tuple[Optional[int], ...] = (None, None, 3),
        **kwargs: Any,
    ) -> "RADConvNet":
        """Create a RADConvNet model from a predefined variant.

        :param variant: One of ``"tiny"``, ``"small"``, ``"base"``.
        :type variant: str
        :param num_classes: Number of output classes. Defaults to ``1000``.
        :type num_classes: int
        :param input_shape: ``(height, width, channels)``. Defaults to
            ``(None, None, 3)``.
        :type input_shape: Tuple[int, ...]
        :param kwargs: Additional keyword args forwarded to the constructor,
            overriding any variant default they name.
        :return: A configured ``RADConvNet``.
        :rtype: RADConvNet
        :raises ValueError: If ``variant`` is not recognized.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )
        config = cls.MODEL_VARIANTS[variant]
        logger.info(f"Creating RADConvNet-{variant.upper()} model with input_shape {input_shape}")
        merged = {**config, "num_classes": num_classes, "input_shape": input_shape}
        merged.update(kwargs)
        return cls(**merged)


# ---------------------------------------------------------------------

def create_rad_convnet(
    variant: str = "tiny",
    num_classes: int = 1000,
    input_shape: Optional[Tuple[Optional[int], ...]] = (None, None, 3),
    **kwargs: Any,
) -> RADConvNet:
    """Thin factory wrapper around :meth:`RADConvNet.from_variant`.

    :param variant: Model variant, one of ``"tiny"``, ``"small"``, ``"base"``.
        Defaults to ``"tiny"``.
    :type variant: str
    :param num_classes: Number of output classes. Defaults to ``1000``.
    :type num_classes: int
    :param input_shape: ``(height, width, channels)``. Defaults to
        ``(None, None, 3)``.
    :type input_shape: Optional[Tuple[int, ...]]
    :param kwargs: Additional keyword arguments forwarded to the constructor.
    :return: A configured ``RADConvNet``.
    :rtype: RADConvNet
    """
    return RADConvNet.from_variant(
        variant, num_classes=num_classes, input_shape=input_shape, **kwargs
    )

# ---------------------------------------------------------------------
