"""RADConvNet: a ConvNeXt-shaped classification backbone built on RAD-Conv.

Defines :class:`RADConvNet`, four stages of blocks with a patchify stem and an
optional classification head, plus the :func:`create_rad_convnet` factory. The
spatial mixer inside each block is
:class:`~dl_techniques.layers.conv_blocks.rad_conv.RADConv2D` rather than ConvNeXt's
fixed depthwise K x K convolution: it predicts, per kernel element and channel group,
four boundary offsets that define an axis-aligned integration rectangle, so even a
``rad_kernel_size=1`` block can aggregate over a receptive field spanning the whole
feature map. Everything around it stays ConvNeXt-shaped: patchify stem, staged
blocks, LayerNorm plus strided-conv downsampling, inverted-bottleneck MLP, and a
GAP + LayerNorm + Dense head, which is the shared vision ``ClassificationHead``.
Each block is transform-only and returns ``F(x)``; the residual add belongs to
:meth:`RADConvNet.call`. Because that integration costs more per channel than a
depthwise convolution, the default widths are smaller than ConvNeXt's.

References:
    - Maleki & Imani, 2025. Region-Aware Deformable Convolutions.
      (https://arxiv.org/abs/2509.15436)
    - Liu et al., 2022. A ConvNet for the 2020s. (https://arxiv.org/abs/2201.03545)
"""

import copy
import keras
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.model_build import (
    materialize_sublayers,
    plain_head_config_overrides,
)
from dl_techniques.utils.weight_transfer import (
    DEFAULT_NON_STRUCTURAL_CONFIG_KEYS,
    load_pretrained_checkpoint,
    pretrained_path_or_raise,
)
from dl_techniques.layers.conv_blocks.rad_conv import RADConv2D
from dl_techniques.layers.conv_blocks.stem import ConvStem
from dl_techniques.layers.heads.vision import VisionTaskType
from dl_techniques.layers.heads.vision.factory import build_model_vision_head
from dl_techniques.utils.activation_serialization import (
    serialize_activation,
    deserialize_activation,
)
from dl_techniques.utils.keras_registration import register_dl_technique

# Constructor-config keys a checkpoint may differ on without being another architecture:
# they leave the weight layout and the computed function alone (the dropout rate acts in
# training only, the input size is not a weight shape). `load_pretrained_checkpoint`
# compares every other key the checkpoint and the model both hold, so a changed
# `activation`, `rad_kernel_size` or `strides` is refused even when every weight shape
# still fits.
_NON_STRUCTURAL_CONFIG_KEYS = (
    *DEFAULT_NON_STRUCTURAL_CONFIG_KEYS,
    "input_shape",
    "dropout_rate",
)

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.rad_convnet.model")
class RADConvNet(keras.Model):
    """Classify images with RAD-Conv blocks in a ConvNeXt-shaped pyramid.

    The stem patchifies by ``strides``, each later stage is preceded by a
    LayerNorm and a 2x2 stride-2 convolution, and the head pools, normalizes and
    projects. Height and width should divide by ``strides * 2 ** (stages - 1)``,
    since the stem convolution uses valid padding for any stride above 1.

    Architecture (``stem`` is a :class:`ConvStem`, so ``stem.conv1`` and
    ``stem.norm1`` are its layers):

    .. code-block:: text

        input [B, H, W, C_in]
                 │
                 ▼
        ┌───────────────────────┐
        │ stem.conv1  S x S / S │
        └───────────────────────┘
                 │
                 ▼
        ┌───────────────────────┐
        │ stem.norm1            │
        └───────────────────────┘
                 │  [B, H/S, W/S, dims[0]]
                 ▼
        ┌───────────────────────┐
        │ stage1 blocks         │  depths[0] blocks
        └───────────────────────┘
                 │
                 ▼
        ┌───────────────────────┐
        │ downsample_{i}_norm   │
        └───────────────────────┘
                 │
                 ▼
        ┌───────────────────────┐
        │ downsample_{i}_conv   │  2x2, stride 2, dims[i + 1]
        └───────────────────────┘
                 │
                 ▼
        ┌───────────────────────┐
        │ stage{S} blocks       │  depths[S - 1] blocks
        └───────────────────────┘
                 │  one downsample before every later stage
                 ▼
           ┌─────┴──────────┐
           ▼                ▼
        include_top        else
           │                │
           ▼                ▼
        classification_head features [B, H', W', dims[-1]]
        (pooled_norm, classifier)
           │
           ▼
        logits [B, num_classes]

    Block ``stage{S}_block{B}``, its layers named ``stage{S}_block{B}_<role>``:

    .. code-block:: text

        x
                 │
                 ▼
        ┌───────────────────────┐
        │ rad_conv  RADConv2D   │
        └───────────────────────┘
                 │
                 ▼
        ┌───────────────────────┐
        │ norm  layernorm       │
        └───────────────────────┘
                 │
                 ▼
        ┌───────────────────────┐
        │ expand  dense 4x      │
        └───────────────────────┘
                 │
                 ▼
              activation
                 │
                 ▼
        ┌───────────────────────┐
        │ dropout               │  (dropout_rate > 0 only)
        └───────────────────────┘
                 │
                 ▼
        ┌───────────────────────┐
        │ reduce  dense -> dim  │
        └───────────────────────┘
                 │
                 ▼
               F(x)

    The block returns F(x) and ``call`` adds the residual. The constructor's own
    defaults match the ``tiny`` variant.

    :param num_classes: Number of output classes. Only used when
        ``include_top=True``, where it must be positive. Defaults to ``1000``.
    :type num_classes: int
    :param depths: Number of blocks per stage. ``None`` resolves to
        ``[2, 2, 4, 2]``. Must have the same length as ``dims``, which sets the
        number of stages.
    :type depths: Optional[List[int]]
    :param dims: Channel count per stage. ``None`` resolves to
        ``[32, 64, 128, 256]``, smaller than ConvNeXt's
        ``[96, 192, 384, 768]`` because RAD-Conv's exact region integration
        costs more per channel than a depthwise convolution.
    :type dims: Optional[List[int]]
    :param rad_kernel_size: RAD-Conv kernel size used inside every block, an int
        or an ``(kh, kw)`` pair. Defaults to ``3``.
    :type rad_kernel_size: Union[int, Tuple[int, int]]
    :param rad_groups: RAD-Conv channel groups used inside every block.
        Defaults to ``1``.
    :type rad_groups: int
    :param mlp_ratio: Inverted-bottleneck expansion ratio of the pointwise
        MLP following each RAD-Conv. Defaults to ``4``.
    :type mlp_ratio: int
    :param activation: Activation used inside each block's MLP. One shared
        ``Activation`` layer serves every block. Defaults to ``"gelu"``.
    :type activation: str
    :param use_bias: Whether convolutions and dense layers, the head's
        classifier included, use a bias, and whether every LayerNormalization
        centers. Defaults to ``True``.
    :type use_bias: bool
    :param dropout_rate: Dropout rate applied after the MLP's expansion
        activation, in ``[0, 1)``. At ``0.0`` no Dropout layer is created.
        Defaults to ``0.0``.
    :type dropout_rate: float
    :param strides: Patchify factor, used as both kernel size and stride of
        the stem convolution. The inter-stage downsamples are always 2x2 with
        stride 2. Defaults to ``4``.
    :type strides: int
    :param include_top: Whether to include the classification head. Defaults
        to ``True``.
    :type include_top: bool
    :param input_shape: ``(height, width, channels)``, excluding batch. Stored
        for serialization; the graph is built from the shape ``build`` receives.
        ``None`` resolves to ``(None, None, 3)``.
    :type input_shape: Optional[Tuple[Optional[int], ...]]
    :param head_config_overrides: Optional dict merged into the
        ``ClassificationHead`` configuration (for example
        ``{'classifier_dropout_rate': 0.2}``). ``name``, ``num_classes`` and
        ``task_type`` are owned by the model and raise ``ValueError``. Every
        value must be JSON data. Defaults to ``None``.
    :type head_config_overrides: Optional[Dict[str, Any]]
    :param kwargs: Additional keyword arguments for the ``keras.Model`` base
        class.

    :raises ValueError: If ``include_top`` is true and ``num_classes`` is not
        positive, ``depths`` and ``dims`` differ in length, ``strides`` is not
        positive, ``dropout_rate`` is outside ``[0, 1)``, ``input_shape`` is
        not 3D, or ``head_config_overrides`` names a reserved key or holds a
        value that is not JSON data.

    Input shape:
        4D tensor ``(batch_size, height, width, channels)``.

    Output shape:
        - ``include_top=True``: ``(batch_size, num_classes)`` logits.
        - ``include_top=False``: ``(batch_size, H', W', dims[-1])``.

    Example::

        model = RADConvNet.from_variant("tiny", num_classes=47, input_shape=(96, 96, 3))

        backbone = RADConvNet.from_variant("small", include_top=False)

        # Warm start from a local checkpoint
        model = RADConvNet.from_variant("tiny", pretrained="/path/to.keras")

    Note:
        No pretrained RADConvNet weights are distributed with ``dl_techniques``.
        ``pretrained=True`` raises ``NotImplementedError`` rather than returning a
        randomly-initialized model; pass a local checkpoint via
        ``pretrained='/path/to/weights.keras'`` instead.
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
        input_shape: Optional[Tuple[Optional[int], ...]] = (None, None, 3),
        head_config_overrides: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if include_top and num_classes <= 0:
            raise ValueError(
                f"num_classes must be positive when include_top=True, got {num_classes}"
            )

        depths = list(depths) if depths is not None else [2, 2, 4, 2]
        dims = list(dims) if dims is not None else [32, 64, 128, 256]

        if len(depths) != len(dims):
            raise ValueError(
                f"Length of depths ({len(depths)}) must equal length of dims ({len(dims)})"
            )
        if strides <= 0:
            raise ValueError(f"strides must be positive, got {strides}")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")
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
        self.activation = deserialize_activation(activation)
        self.activation_layer = keras.layers.Activation(self.activation, name="mlp_activation")
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.strides = strides
        self.include_top = include_top
        self.input_shape_config = tuple(input_shape)
        self._head_config_overrides = plain_head_config_overrides(
            head_config_overrides, type(self).__name__
        )

        self._build_stem()

        self.downsamplers = []
        self.stages = []
        for stage_idx in range(len(self.depths)):
            if stage_idx > 0:
                self._build_downsampler(stage_idx)
            self._build_stage(stage_idx)

        self.classification_head = None
        if self.include_top:
            self._build_head()

        logger.info(
            f"Created RADConvNet for input {self.input_shape_config} "
            f"with {sum(depths)} blocks"
        )

    def _build_stem(self) -> None:
        """Build the patchify stem via :class:`ConvStem`: strided conv + LayerNorm.

        Padding is valid for any stride above 1, so the stem divides the input
        rather than padding it.
        """
        self.stem = ConvStem(
            filters=self.dims[0],
            kernel_size=self.strides,
            strides=self.strides,
            padding="same" if self.strides == 1 else "valid",
            use_bias=self.use_bias,
            num_convs=1,
            normalization_type="layer_norm",
            normalization_kwargs={
                "epsilon": self.LAYERNORM_EPSILON,
                "center": self.use_bias,
                "scale": True,
            },
            activation_type="linear",
            use_pooling=False,
            kernel_initializer=self.STEM_INITIALIZER,
            name="stem",
        )

    def _build_downsampler(self, stage_idx: int) -> None:
        """Build the LayerNorm and strided convolution that feed a stage.

        :param stage_idx: Index of the stage the downsampler feeds; it must be
            greater than 0. The pair is appended to ``downsamplers``, so it sits
            at ``stage_idx - 1`` and is named ``downsample_{stage_idx - 1}_*``.
        """
        norm = keras.layers.LayerNormalization(
            epsilon=self.LAYERNORM_EPSILON,
            center=self.use_bias,
            scale=True,
            name=f"downsample_{stage_idx - 1}_norm",
        )
        conv = keras.layers.Conv2D(
            filters=self.dims[stage_idx],
            kernel_size=2,
            strides=2,
            padding="same",
            use_bias=self.use_bias,
            name=f"downsample_{stage_idx - 1}_conv",
        )
        self.downsamplers.append({"norm": norm, "conv": conv})

    def _build_stage(self, stage_idx: int) -> None:
        """Build the blocks of stage ``stage_idx``.

        Each block is held as a role-keyed dict: ``rad_conv``, ``norm``,
        ``expand``, ``reduce`` and ``dropout``, which is ``None`` unless
        ``dropout_rate`` is above 0. Layers are named
        ``stage{S}_block{B}_<role>`` with S and B counted from 1.

        :param stage_idx: Index of the stage to build.
        """
        dim = self.dims[stage_idx]
        blocks = []
        for block_idx in range(self.depths[stage_idx]):
            prefix = f"stage{stage_idx + 1}_block{block_idx + 1}"
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
        self.stages.append(blocks)

    def _build_head(self) -> None:
        """Build the classification head through the vision heads factory.

        The reference head, ConvNeXt's ``self.head(self.norm(x.mean([-2, -1])))``:
        global average pooling, a LayerNorm at epsilon 1e-6 that centers only when
        ``use_bias`` is set, and one linear layer that has a bias only when
        ``use_bias`` is set; no hidden block.
        """
        # DECISION plan-2026-10-26T125959-7af851e5/D-043: the head turns its initializer
        # into an instance and then clones it for the
        # classifier, and a seedless clone draws a second default seed. The classifier
        # kernel was drawn from the FIRST default seed taken after the last block, so the
        # seed is taken here and handed over explicitly (a clone keeps an explicit seed):
        # the kernel of a seeded build stays what it was before the head moved.
        # See decisions.md.
        head_initializer = keras.initializers.TruncatedNormal(
            seed=keras.initializers.get(self.HEAD_INITIALIZER).seed
        )
        self.classification_head = build_model_vision_head(
            VisionTaskType.CLASSIFICATION,
            {
                "num_classes": self.num_classes,
                "use_global_pooling": True,
                "pooling_type": "avg",
                "use_attention": False,
                "block_dims": (),
                "pooled_normalization_type": "layer_norm",
                "normalization_kwargs": {
                    "epsilon": self.LAYERNORM_EPSILON,
                    "center": self.use_bias,
                    "scale": True,
                },
                "classifier_use_bias": self.use_bias,
                # No hidden block, so no block dropout either.
                "dropout_rate": 0.0,
                "kernel_initializer": head_initializer,
                # Named, so a checkpoint written in one process lands on the same
                # weight paths in another (an unnamed head is `classification_head_1`
                # in the second).
                "name": "classification_head",
            },
            self._head_config_overrides,
        )

    def build(self, input_shape: Any) -> None:
        """Materialize every sub-layer from `input_shape` by tracing `call`.

        `materialize_sublayers` traces `call` on symbolic inputs, so what gets
        built cannot drift from what gets called.

        :param input_shape: Shape (or nest of shapes) of the input to `call`.
        """
        if self.built:
            return
        materialize_sublayers(self, input_shape)
        super().build(input_shape)

    def call(
        self,
        inputs: "keras.KerasTensor",
        training: Optional[bool] = None,
    ) -> "keras.KerasTensor":
        """Run the stem, the stages with their downsamplers, and the optional head.

        :param inputs: ``(batch_size, height, width, channels)`` input.
        :type inputs: keras.KerasTensor
        :param training: Whether the model is in training mode. Reaches the
            block dropout, the stem, whose LayerNorm ignores it, and the head.
        :type training: Optional[bool]
        :return: The logits ``(batch_size, num_classes)`` (``include_top=True``)
            or the final stage's feature maps ``(batch_size, H', W', dims[-1])``
            (``include_top=False``).
        :rtype: keras.KerasTensor
        """
        x = self.stem(inputs, training=training)

        for stage_idx, stage_blocks in enumerate(self.stages):
            if stage_idx > 0:
                downsampler = self.downsamplers[stage_idx - 1]
                x = downsampler["norm"](x)
                x = downsampler["conv"](x)

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
            return self.classification_head(x, training=training)["logits"]

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Report the shape `call` returns, without building the model.

        The RAD-Conv blocks are shape-preserving, so only the stem, the
        downsamplers and the head move the shape.

        :param input_shape: Input shape, channels-last.
        :type input_shape: tuple
        :return: Output shape, matching whichever branch of the class
            docstring's Output shape applies.
        :rtype: tuple
        """
        shape = self.stem.compute_output_shape(input_shape)

        for stage_idx in range(len(self.depths)):
            if stage_idx > 0:
                downsampler = self.downsamplers[stage_idx - 1]
                shape = downsampler["norm"].compute_output_shape(shape)
                shape = downsampler["conv"].compute_output_shape(shape)
            shape = shape[:-1] + (self.dims[stage_idx],)

        if self.include_top:
            # The head reports both keys; `call` returns the 'logits' one.
            shape = self.classification_head.compute_output_shape(shape)["logits"]

        return shape

    def load_pretrained_weights(
        self,
        weights_path: str,
        skip_mismatch: bool = True,
    ) -> None:
        """Load a local `.keras` checkpoint into the model, layer by layer.

        The transfer is `load_pretrained_checkpoint`: it builds the model at
        `input_shape` if it is unbuilt and takes a headed checkpoint into a
        headless model. Every layer outside the classification head must be
        restored, so a checkpoint of another variant, width or depth raises and
        leaves the model untouched. Only a head of another `num_classes` is
        skipped: it keeps its initialization and a warning names it. The
        archive's own config is compared too: a constructor argument that both
        hold and that is not in `_NON_STRUCTURAL_CONFIG_KEYS` must be equal, so
        a checkpoint of another `activation` is refused even when every weight
        shape fits. An argument only one side holds is not compared, and an
        archive without a config of this class is checked by layers only.

        :param weights_path: Path to the weights file (`.keras` format).
        :param skip_mismatch: `True` (default) lets the head layers differ;
            `False` allows no difference anywhere. The inverse of the transfer
            helper's `strict`.
        :raises FileNotFoundError: If `weights_path` doesn't exist.
        :raises ValueError: If a layer outside the head is not restored, or
            the weights cannot be loaded.
        """
        load_pretrained_checkpoint(
            self, weights_path, self.input_shape_config, skip_mismatch,
            ignore_config_keys=_NON_STRUCTURAL_CONFIG_KEYS,
        )

    @classmethod
    def from_variant(
        cls,
        variant: str,
        num_classes: int = 1000,
        input_shape: Optional[Tuple[Optional[int], ...]] = None,
        pretrained: Union[bool, str] = False,
        **kwargs: Any,
    ) -> "RADConvNet":
        """Create a RADConvNet model from a predefined variant.

        :param variant: A key of ``MODEL_VARIANTS``: ``"tiny"``, ``"small"`` or
            ``"base"``.
        :param num_classes: Number of output classes. Defaults to ``1000``.
        :param input_shape: ``(height, width, channels)``. ``None`` resolves to
            ``(None, None, 3)``.
        :param pretrained: A path to a local weights file, `True` (raises
            `NotImplementedError`, since no public RADConvNet weights ship with
            `dl_techniques`), or `False` (default, random init).
        :param kwargs: Passthrough to the constructor; an architecture field of
            the variant (for example `depths`, `dims`) is replaced by the value
            given here.
        :return: A configured ``RADConvNet``.
        :raises ValueError: If ``variant`` is not recognized, or a resolved
            argument fails the constructor's checks.
        :raises NotImplementedError: If `pretrained` is `True`.
        :raises FileNotFoundError: If `pretrained` is a path that doesn't exist.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        weights_path = pretrained_path_or_raise(
            pretrained, f"{cls.__name__} variant '{variant}'"
        )

        config = copy.deepcopy(cls.MODEL_VARIANTS[variant])
        config.update(kwargs)

        model = cls(num_classes=num_classes, input_shape=input_shape, **config)

        if weights_path:
            model.load_pretrained_weights(weights_path)

        return model

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
            "activation": serialize_activation(self.activation),
            "use_bias": self.use_bias,
            "dropout_rate": self.dropout_rate,
            "strides": self.strides,
            "include_top": self.include_top,
            "input_shape": self.input_shape_config,
            "head_config_overrides": self._head_config_overrides,
        }
        base_config = super().get_config()
        return {**base_config, **config}


# ---------------------------------------------------------------------

def create_rad_convnet(
    variant: str = "tiny",
    num_classes: int = 1000,
    input_shape: Optional[Tuple[Optional[int], ...]] = None,
    pretrained: Union[bool, str] = False,
    **kwargs: Any,
) -> RADConvNet:
    """Thin factory wrapper around :meth:`RADConvNet.from_variant`.

    :param variant: Model variant, one of ``"tiny"``, ``"small"``, ``"base"``.
        Defaults to ``"tiny"``.
    :type variant: str
    :param num_classes: Number of output classes. Defaults to ``1000``.
    :type num_classes: int
    :param input_shape: ``(height, width, channels)``; ``None`` resolves to
        ``(None, None, 3)``.
    :type input_shape: Optional[Tuple[int, ...]]
    :param pretrained: A path to a local weights file, `True` (raises
        `NotImplementedError`), or `False` (default, random init).
    :type pretrained: Union[bool, str]
    :param kwargs: Additional keyword arguments forwarded to
        :meth:`RADConvNet.from_variant`.
    :return: A configured ``RADConvNet``.
    :rtype: RADConvNet
    :raises ValueError: If ``variant`` is unknown, or a forwarded argument is
        invalid.
    :raises NotImplementedError: If `pretrained` is `True`.
    """
    return RADConvNet.from_variant(
        variant,
        num_classes=num_classes,
        input_shape=input_shape,
        pretrained=pretrained,
        **kwargs,
    )

# ---------------------------------------------------------------------
