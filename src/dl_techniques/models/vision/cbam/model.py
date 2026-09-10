"""CBAMNet, a CNN backbone that refines feature maps with the Convolutional Block Attention Module.

This file holds :class:`CBAMNet`, its ``MODEL_VARIANTS`` presets and the
``create_cbam_net`` factory. CBAM applies attention in two steps rather than
one joint reweighting: channel attention pools each feature map to a scalar
with both average and max pooling, mixes the two through a shared bottleneck
MLP and emits per-channel gains, then spatial attention pools across channels
and a large convolution turns that into a spatial mask. Both masks pass
through a sigmoid, so attention can only attenuate or preserve a feature,
never invert or amplify it. The backbone stacks one
`Conv2D -> BatchNorm -> CBAM -> MaxPooling2D` stage per entry in `dims`, with
CBAM between the normalization and the pooling. Every stage halves both
spatial axes, so an input needs to survive `2 ** len(dims)` reductions. The
classification head applies softmax itself, so the output is probabilities
rather than logits, and no pretrained weights ship with this model:
`pretrained=True` raises `NotImplementedError`.

References:
    - Woo et al., 2018. CBAM: Convolutional Block Attention Module. ECCV 2018.
      (https://arxiv.org/abs/1807.06521)
    - Hu et al., 2018. Squeeze-and-Excitation Networks.
      (https://arxiv.org/abs/1709.01507)
    - Park et al., 2018. BAM: Bottleneck Attention Module.
      (https://arxiv.org/abs/1807.06514)
    - Ioffe and Szegedy, 2015. Batch Normalization: Accelerating Deep Network
      Training by Reducing Internal Covariate Shift.
      (https://arxiv.org/abs/1502.03167)

"""

import keras
from typing import List, Optional, Union, Tuple, Dict, Any

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.weight_transfer import load_weights_from_checkpoint
from dl_techniques.layers.attention.convolutional_block_attention import CBAM
from dl_techniques.utils.model_build import materialize_sublayers
from dl_techniques.utils.keras_registration import register_dl_technique


# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.cbam.model")
class CBAMNet(keras.Model):
    """Classify images with a CBAM block after every convolutional stage.

    Architecture:

    .. code-block:: text

        input [B, H, W, C]
              │
              ▼
        ┌──────────────────────────────┐
        │ stage i, one per dims entry  │
        └──────────────────────────────┘
              │ [B, H/2^n, W/2^n, dims[-1]],  n = len(dims)
              ├──► output              (include_top=False)
              ▼
        ┌──────────────────────────────┐
        │ global average pool          │
        └──────────────────────────────┘
              │ [B, dims[-1]]
              ▼
        ┌──────────────────────────────┐
        │ dense num_classes, softmax   │
        └──────────────────────────────┘
              │
              ▼
        probabilities [B, num_classes]

    Stage internals:

    .. code-block:: text

        in [B, h, w, c]
              │
              ▼
        ┌──────────────────────────────┐
        │ conv 3x3, same, relu         │
        └──────────────────────────────┘
              │ [B, h, w, dim]
              ▼
        ┌──────────────────────────────┐
        │ batch norm                   │
        └──────────────────────────────┘
              │
              ▼
        ┌──────────────────────────────┐
        │ cbam: channel, then spatial  │
        └──────────────────────────────┘
              │
              ▼
        ┌──────────────────────────────┐
        │ max pool 2x2                 │
        └──────────────────────────────┘
              │
              ▼
        out [B, h/2, w/2, dim]

    Variants:

    .. code-block:: text

        variant   dims              spatial reduction
        tiny      64, 128           4x
        small     64, 128, 256      8x
        base      128, 256, 512     8x

    :param num_classes: Number of output classes. Only used if `include_top` is True. Must be positive.
    :type num_classes: int
    :param dims: Channel dimensions for each stage; each entry creates one Conv-BN-CBAM-Pool stage. Must be non-empty with all positive values. Defaults to ``[64, 128]``.
    :type dims: Optional[List[int]]
    :param attention_ratio: Reduction ratio for the channel-attention MLP in each CBAM block. Must be positive.
    :type attention_ratio: int
    :param attention_kernel_size: Kernel size for the spatial-attention convolution in each CBAM block. Checked here for positivity; oddness is left to ``CBAM``.
    :type attention_kernel_size: int
    :param kernel_initializer: Initializer for Conv2D and Dense kernels.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for Conv2D and Dense kernels.
    :type kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param include_top: Whether to include the classification head. If False, returns the final stage's feature maps.
    :type include_top: bool
    :param input_shape: Input shape excluding the batch dimension, e.g. ``(height, width, channels)``. Recorded for serialization and used as the shape of ``from_variant``'s warm-up call; it does not build the model, which is built on the first real call.
    :type input_shape: Optional[Tuple[int, ...]]
    :param kwargs: Additional keyword arguments for `keras.Model` (e.g. `name`).
    :ivar stages: One list of layers per stage (Conv2D, BatchNormalization, CBAM, MaxPooling2D).
    :vartype stages: List[List[keras.layers.Layer]]
    :ivar head: Classification-head layers (GlobalAveragePooling2D, Dense). Empty if `include_top` is False.
    :vartype head: List[keras.layers.Layer]
    :raises ValueError: If `num_classes` is not positive when `include_top` is True, if `dims` is empty or non-positive, if `attention_ratio` is not positive, or if `attention_kernel_size` is not positive.

    Input shape:
        4D tensor, shape ``(batch_size, height, width, channels)``.

    Output shape:
        2D tensor ``(batch_size, num_classes)`` if `include_top` is True, otherwise 4D tensor ``(batch_size, H', W', dims[-1])``.

    Example:
        .. code-block:: python

            model = CBAMNet(num_classes=10, dims=[64, 128], input_shape=(32, 32, 3))
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            # Feature extractor, no classification head.
            model = CBAMNet(dims=[128, 256, 512], include_top=False, input_shape=(224, 224, 3))
    """

    MODEL_VARIANTS: Dict[str, Dict[str, List[int]]] = {
        "tiny": {"dims": [64, 128]},
        "small": {"dims": [64, 128, 256]},
        "base": {"dims": [128, 256, 512]},
    }

    def __init__(
        self,
        num_classes: int = 1000,
        dims: Optional[List[int]] = None,
        attention_ratio: int = 8,
        attention_kernel_size: int = 7,
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        include_top: bool = True,
        input_shape: Optional[Tuple[int, ...]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if dims is None:
            dims = [64, 128]

        if include_top and num_classes <= 0:
            raise ValueError(f"num_classes must be positive when include_top=True, got {num_classes}")
        if not dims or any(d <= 0 for d in dims):
            raise ValueError(f"dims must be a non-empty list of positive integers, got {dims}")
        if attention_ratio <= 0:
            raise ValueError(f"attention_ratio must be positive, got {attention_ratio}")
        if attention_kernel_size <= 0:
            raise ValueError(f"attention_kernel_size must be positive, got {attention_kernel_size}")

        self.num_classes = num_classes
        self.dims = dims
        self.attention_ratio = attention_ratio
        self.attention_kernel_size = attention_kernel_size
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer) if kernel_regularizer else None
        self.include_top = include_top
        self.input_shape_arg = input_shape

        # Sub-layers are created here (not in build) so they exist before the
        # first call and are captured by serialization.
        self.stages: List[List[keras.layers.Layer]] = []
        for i, dim in enumerate(self.dims):
            stage_layers = [
                keras.layers.Conv2D(
                    filters=dim,
                    kernel_size=3,
                    activation='relu',
                    padding='same',
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f"stage_{i}_conv"
                ),
                # DECISION plan-2026-08-22T035419-a11304c8/D-111: keep Keras' default
                # BatchNorm momentum and epsilon; the CBAM paper specifies no
                # normalization setting to trace to. See decisions.md.
                keras.layers.BatchNormalization(
                    momentum=0.99,
                    epsilon=1e-3,
                    name=f"stage_{i}_bn",
                ),
                CBAM(
                    channels=dim,
                    ratio=self.attention_ratio,
                    kernel_size=self.attention_kernel_size,
                    name=f"stage_{i}_cbam"
                ),
                keras.layers.MaxPooling2D(pool_size=(2, 2), name=f"stage_{i}_pool")
            ]
            self.stages.append(stage_layers)

        self.head: List[keras.layers.Layer] = []
        if self.include_top:
            self.head.append(
                keras.layers.GlobalAveragePooling2D(name="global_avg_pool")
            )
            # Always true here, since include_top requires a positive count.
            if self.num_classes > 0:
                self.head.append(
                    keras.layers.Dense(
                        units=self.num_classes,
                        activation='softmax',
                        kernel_initializer=self.kernel_initializer,
                        kernel_regularizer=self.kernel_regularizer,
                        name="classifier"
                    )
                )

    def build(self, input_shape: Any) -> None:
        """Materialize every sub-layer from `input_shape`.

        Keras would otherwise mark the model built with every sub-layer still
        unbuilt. The shared helper traces `call` on symbolic inputs, so what
        gets built cannot drift from what gets called. Returns immediately if
        the model is already built.

        :param input_shape: Shape (or nest of shapes) of the input to `call`.
        :type input_shape: Any
        :return: Nothing.
        :rtype: None
        """
        if self.built:
            return
        materialize_sublayers(self, input_shape)
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Run the forward pass.

        :param inputs: Input tensor, shape ``(batch_size, height, width, channels)``.
        :type inputs: keras.KerasTensor
        :param training: Whether the model is in training mode. Affects BatchNormalization.
        :type training: Optional[bool]
        :return: Class probabilities ``(batch_size, num_classes)`` if `include_top` is True, otherwise feature maps ``(batch_size, H', W', dims[-1])``.
        :rtype: keras.KerasTensor
        """
        x = inputs

        for stage_layers in self.stages:
            for layer in stage_layers:
                x = layer(x, training=training)

        if self.include_top:
            for layer in self.head:
                x = layer(x, training=training)

        return x

    def get_config(self) -> Dict[str, Any]:
        """Return the model configuration for serialization.

        :return: Config dict with every constructor argument.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "dims": self.dims,
            "attention_ratio": self.attention_ratio,
            "attention_kernel_size": self.attention_kernel_size,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "include_top": self.include_top,
            "input_shape": self.input_shape_arg,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CBAMNet":
        """Build a model from a config dict, deserializing initializer and regularizer entries.

        :param config: Config dict as returned by `get_config`.
        :type config: Dict[str, Any]
        :return: A new `CBAMNet` instance.
        :rtype: CBAMNet
        """
        if "kernel_initializer" in config:
            config["kernel_initializer"] = keras.initializers.deserialize(config["kernel_initializer"])
        if "kernel_regularizer" in config:
            config["kernel_regularizer"] = keras.regularizers.deserialize(config["kernel_regularizer"])

        return cls(**config)

    @classmethod
    def from_variant(
        cls,
        variant: str,
        num_classes: int = 1000,
        input_shape: Optional[Tuple[int, ...]] = (224, 224, 3),
        pretrained: Union[bool, str] = False,
        weights_dataset: str = "imagenet",
        **kwargs: Any
    ) -> "CBAMNet":
        """Create a `CBAMNet` model from a predefined variant.

        Loading weights forces a build first, using ``input_shape`` if it was
        given and ``(32, 32, 3)`` otherwise.

        :param variant: Variant name, one of ``"tiny"``, ``"small"``, ``"base"``.
        :type variant: str
        :param num_classes: Number of output classes.
        :type num_classes: int
        :param input_shape: Input shape ``(height, width, channels)``.
        :type input_shape: Optional[Tuple[int, ...]]
        :param pretrained: A local weights path, or True to raise `NotImplementedError` (no public CBAMNet weights ship with `dl_techniques`). The classifier is skipped only when the head is included and ``num_classes`` is not 1000; every other case transfers strictly.
        :type pretrained: Union[bool, str]
        :param weights_dataset: Dataset identifier, used only when `pretrained` is True.
        :type weights_dataset: str
        :param kwargs: Additional arguments passed to the constructor (e.g. `attention_ratio`, `kernel_regularizer`).
        :return: A `CBAMNet` instance with the variant's configuration.
        :rtype: CBAMNet
        :raises ValueError: If `variant` is not recognized.
        :raises NotImplementedError: If `pretrained` is True.

        Example:
            .. code-block:: python

                model = CBAMNet.from_variant("tiny", num_classes=10, input_shape=(32, 32, 3))
                model = CBAMNet.from_variant("small", include_top=False)
                model = CBAMNet.from_variant("tiny", pretrained="path/to/weights.keras")
        """
        if variant not in cls.MODEL_VARIANTS:
            available = list(cls.MODEL_VARIANTS.keys())
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: {available}"
            )

        # DECISION plan-2026-08-19T163559-499b6f0e/D-127: copy the preset, then
        # update; splatting it alongside **kwargs raises on any override and
        # updating in place poisons the shared table. See decisions.md.
        variant_config = cls.MODEL_VARIANTS[variant].copy()
        variant_config.pop("description", None)
        variant_config.update(kwargs)

        model = cls(
            num_classes=num_classes,
            input_shape=input_shape,
            **variant_config
        )

        if pretrained:
            if isinstance(pretrained, str):
                weights_path = pretrained
                logger.info(f"Loading weights from {weights_path}...")
            else:
                weights_path = cls._download_weights(variant, weights_dataset)

            # The ImageNet head is 1000-wide; a different num_classes means the
            # classifier weights must be skipped rather than refused.
            include_top = kwargs.get("include_top", True)
            skip_mismatch = include_top and (num_classes != 1000)

            # Layer-by-layer transfer, since Keras 3.8's Model.load_weights
            # rejects by_name outright.
            if not model.built:
                model(keras.ops.zeros((1,) + tuple(model.input_shape_arg or (32, 32, 3))))
            report = load_weights_from_checkpoint(
                target=model,
                ckpt_path=weights_path,
                skip_prefixes=("classifier",) if skip_mismatch else (),
                strict=not skip_mismatch,
            )
            logger.info(report.summary_string())

        return model

    @staticmethod
    def _download_weights(variant: str, dataset: str = "imagenet") -> str:
        """Raise, since no public CBAMNet weights ship with `dl_techniques`.

        Kept so `pretrained=True` fails loudly instead of silently returning a
        randomly initialized model.

        :param variant: Model variant name (unused).
        :type variant: str
        :param dataset: Dataset identifier (unused).
        :type dataset: str
        :return: Never returns, despite the return annotation.
        :rtype: str
        :raises NotImplementedError: Always.
        """
        raise NotImplementedError(
            f"No pretrained CBAMNet weights are distributed with dl_techniques "
            f"(requested variant '{variant}', dataset '{dataset}'). Pass a local "
            f"checkpoint instead: CBAMNet.from_variant('{variant}', "
            f"pretrained='/path/to/weights.keras')."
        )

# ---------------------------------------------------------------------

def create_cbam_net(
    variant: str = "tiny",
    num_classes: int = 1000,
    input_shape: Optional[Tuple[int, ...]] = (224, 224, 3),
    pretrained: Union[bool, str] = False,
    **kwargs: Any
) -> CBAMNet:
    """Create a `CBAMNet` model. A thin wrapper around `CBAMNet.from_variant`.

    :param variant: Variant name, one of ``"tiny"``, ``"small"``, ``"base"``.
    :type variant: str
    :param num_classes: Number of output classes.
    :type num_classes: int
    :param input_shape: Input shape ``(height, width, channels)``.
    :type input_shape: Optional[Tuple[int, ...]]
    :param pretrained: A local weights path, or True to raise `NotImplementedError`.
    :type pretrained: Union[bool, str]
    :param kwargs: Additional arguments for the model constructor.
    :return: A `CBAMNet` instance.
    :rtype: CBAMNet
    :raises ValueError: If `variant` is not recognized.
    :raises NotImplementedError: If `pretrained` is True.

    Example:
        .. code-block:: python

            model = create_cbam_net("tiny", num_classes=10, input_shape=(32, 32, 3))
    """
    return CBAMNet.from_variant(
        variant=variant,
        num_classes=num_classes,
        input_shape=input_shape,
        pretrained=pretrained,
        **kwargs
    )

# ---------------------------------------------------------------------
