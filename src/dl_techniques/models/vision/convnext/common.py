"""``_ConvNeXtBase``, the body ConvNeXt V1 and V2 share.

ConvNeXt V1 and V2 differ in one layer, the block (V2 adds Global Response Normalization after the
GELU), and in their variant tables. The constructor, the stem, downsampler, stage and head builders,
``call``, ``build``, ``compute_output_shape``, the pretrained loader, ``from_variant``,
``get_config`` and ``summary`` are the same text, so they live here once.

Interface contract. A leaf subclasses ``_ConvNeXtBase``, is decorated with
``register_dl_technique`` under its own key, documents the constructor arguments in its class
docstring and sets two class attributes:

* ``MODEL_VARIANTS``, the variant name to ``{"depths": [...], "dims": [...]}`` table that
  ``from_variant`` reads. A leaf without one gets an empty table, so ``from_variant`` raises
  ``ValueError`` listing no variant.
* ``BLOCK_CLASS``, a transform-only block layer that takes ``kernel_size``, ``filters``,
  ``activation``, ``kernel_regularizer``, ``use_bias``, ``dropout_rate``,
  ``spatial_dropout_rate``, ``use_gamma``, ``use_softorthonormal_regularizer`` and ``name``, and
  returns ``F(x)`` (the model adds the residual and the drop path). A leaf without one raises
  ``TypeError`` naming the attribute when it is constructed.

The base is never built itself. Its callers are ``ConvNeXtV1`` and ``ConvNeXtV2``; the
registration key, the class docstring and the ``create_*`` factory stay in the leaf file.
"""

import copy
import keras
from typing import List, Optional, Union, Tuple, Dict, Any

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
from dl_techniques.utils.drop_path import linear_drop_path_rates
from dl_techniques.layers.conv_blocks.stem import ConvStem
from dl_techniques.layers.regularization.stochastic_depth import StochasticDepth
from dl_techniques.layers.regularization.stochastic_gradient import StochasticGradient
from dl_techniques.utils.activation_serialization import (
    serialize_activation,
    deserialize_activation,
)
from dl_techniques.layers.heads.vision import VisionTaskType
from dl_techniques.layers.heads.vision.factory import build_model_vision_head

# Constructor-config keys a checkpoint may differ on without being another architecture:
# they leave the weight layout and the computed function alone (the regularizers, the
# dropout and drop-path rates, the input size). `load_pretrained_checkpoint` compares
# every other key the checkpoint and the model both hold, so a changed `kernel_size`,
# `activation`, `strides` or `stochastic_mode` is refused even when every weight shape
# still fits.
_NON_STRUCTURAL_CONFIG_KEYS = (
    *DEFAULT_NON_STRUCTURAL_CONFIG_KEYS,
    "input_shape",
    "kernel_regularizer",
    "use_softorthonormal_regularizer",
    "dropout_rate",
    "spatial_dropout_rate",
    "drop_path_rate",
)

# ---------------------------------------------------------------------


class _ConvNeXtBase(keras.Model):
    """The ConvNeXt body: a patchify stem, ``len(depths)`` stages of blocks, a head.

    Stages are separated by a LayerNorm and a strided convolution. The block is transform-only:
    it returns ``F(x)`` and :meth:`call` adds the residual and the optional drop path, whose rate
    ramps linearly across ``sum(depths)`` blocks. The head is the shared vision
    ``ClassificationHead`` (global average pooling, LayerNorm, one linear layer) and the model
    returns its logits. See the module docstring for what a leaf must set; the constructor
    arguments are documented in the leaves.
    """

    MODEL_VARIANTS: Dict[str, Dict[str, List[int]]] = {}
    BLOCK_CLASS: Optional[type] = None

    # Architecture constants
    LAYERNORM_EPSILON = 1e-6
    # DECISION plan-2026-10-14T062616-81938e81/D-076
    # The reference's _init_weights draws every Conv2d, the stem and the
    # downsample convs included, from trunc_normal_(std=.02). Do NOT go back to
    # the "truncated_normal" string: Keras' default stddev is 0.05, 2.5x the
    # reference. A config dict, so each conv resolves its own instance.
    # See decisions.md D-076.
    STEM_INITIALIZER: Dict[str, Any] = {
        "class_name": "TruncatedNormal",
        "config": {"mean": 0.0, "stddev": 0.02},
    }
    # DECISION plan-2026-10-14T062616-81938e81/D-013
    # The reference initializes the head Linear with trunc_normal_(std=.02).
    # Do NOT revert to the "truncated_normal" string: Keras' default stddev is
    # 0.05. A config dict, not an instance: a shared instance replays one draw.
    # See decisions.md D-013.
    HEAD_INITIALIZER: Dict[str, Any] = {
        "class_name": "TruncatedNormal",
        "config": {"mean": 0.0, "stddev": 0.02},
    }
    # Spatial size used to materialize weights before a checkpoint load when the
    # model was constructed with the default, fully-shape-agnostic
    # `input_shape=(None, None, 3)`. 224 is the ImageNet size these checkpoints
    # are trained at, and it is the size the factory docstrings already promise.
    PRETRAINED_BUILD_SPATIAL = 224

    def __init__(
            self,
            num_classes: int = 1000,
            depths: Optional[List[int]] = None,
            dims: Optional[List[int]] = None,
            drop_path_rate: float = 0.0,
            stochastic_mode: str = 'depth',
            kernel_size: Union[int, Tuple[int, int]] = 7,
            activation: str = "gelu",
            use_bias: bool = True,
            kernel_regularizer: Optional[Union[str, Dict[str, Any], keras.regularizers.Regularizer]] = None,
            dropout_rate: float = 0.0,
            spatial_dropout_rate: float = 0.0,
            strides: int = 4,
            use_gamma: bool = True,
            use_softorthonormal_regularizer: bool = False,
            include_top: bool = True,
            input_shape: Optional[Tuple[int, ...]] = (None, None, 3),
            head_config_overrides: Optional[Dict[str, Any]] = None,
            **kwargs
    ):
        super().__init__(**kwargs)

        if self.BLOCK_CLASS is None:
            raise TypeError(
                f"{type(self).__name__} must set BLOCK_CLASS to its block layer class"
            )

        if include_top and num_classes <= 0:
            raise ValueError(
                f"num_classes must be positive when include_top=True, got {num_classes}"
            )

        depths = list(depths) if depths is not None else [3, 3, 9, 3]
        dims = list(dims) if dims is not None else [96, 192, 384, 768]

        if len(depths) != len(dims):
            raise ValueError(
                f"depths and dims must have the same length, got depths={depths} and dims={dims}"
            )

        # A built-in variant's depths (the 2-stage ``cifar10``) are deliberate, and they
        # arrive here as explicit arguments both from ``from_variant`` and after
        # ``from_config`` or a ``.keras`` reload, so the check compares the values.
        if len(depths) != 4 and not any(
                depths == list(v["depths"]) for v in self.MODEL_VARIANTS.values()):
            logger.warning(f"ConvNeXt typically uses 4 stages, got {len(depths)} stages")

        if strides <= 0:
            raise ValueError(f"strides must be positive, got {strides}")

        if stochastic_mode not in ('depth', 'gradient'):
            raise ValueError(
                f"stochastic_mode must be 'depth' or 'gradient', got {stochastic_mode!r}"
            )

        if input_shape is None:
            input_shape = (None, None, 3)
        if len(input_shape) != 3:
            raise ValueError(f"input_shape must be 3D, got {input_shape}")
        if input_shape[2] not in (1, 3):
            logger.warning(
                f"Unusual number of channels: {input_shape[2]}. ConvNeXt typically uses 3 channels")

        self.num_classes = num_classes
        self.depths = depths
        self.dims = dims
        self.drop_path_rate = drop_path_rate
        self.stochastic_mode = stochastic_mode
        self.kernel_size = kernel_size
        self.activation = deserialize_activation(activation)
        self.use_bias = use_bias
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.dropout_rate = dropout_rate
        self.spatial_dropout_rate = spatial_dropout_rate
        self.use_gamma = use_gamma
        self.use_softorthonormal_regularizer = use_softorthonormal_regularizer
        self.include_top = include_top
        self.strides = strides
        self.input_shape_config = input_shape

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
            f"Created {type(self).__name__} for input {input_shape} "
            f"with {sum(depths)} blocks"
        )

    def _build_stem(self) -> None:
        """Build the patchify stem via :class:`ConvStem`.

        A ``strides x strides`` convolution at stride ``strides`` followed by
        LayerNorm. The padding rule is deliberately NOT the downsamplers'
        unconditional ``"same"``; see the D-125 anchor in the body.
        """
        self.stem = ConvStem(
            filters=self.dims[0],
            kernel_size=self.strides,
            strides=self.strides,
            # DECISION plan-2026-08-19T163559-499b6f0e/D-125: not the unconditional
            # "same" the downsample layers use — "same" here would silently shift the spatial geometry of checkpoints trained at a non-divisible input size. See decisions.md.
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
            kernel_regularizer=self.kernel_regularizer,
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
            name=f"downsample_{stage_idx - 1}_norm"
        )
        conv = keras.layers.Conv2D(
            filters=self.dims[stage_idx],
            kernel_size=self.strides,
            strides=self.strides,
            # DECISION plan_2026-06-15_e6a0391c/D-003: "same" not "valid" — at
            # kernel==stride, "valid" collapses small CIFAR-scale inputs to 0x0 and produced NaN output. Identical to "valid" when the spatial dim divides the stride. See decisions.md.
            padding="same",
            use_bias=self.use_bias,
            kernel_initializer=self.STEM_INITIALIZER,
            kernel_regularizer=self.kernel_regularizer,
            name=f"downsample_{stage_idx - 1}_conv"
        )
        self.downsamplers.append({"norm": norm, "conv": conv})

    def _build_stage(self, stage_idx: int) -> None:
        """Build one stage of blocks.

        Each entry appended to ``stages`` is a list of ``{"block": ..., "drop_path": ...}``
        mappings; ``drop_path`` is ``None`` when the block's ramped rate is 0.

        :param stage_idx: Index of the stage to build.
        """
        stage_blocks = []
        depth = self.depths[stage_idx]
        dim = self.dims[stage_idx]
        block_start_idx = sum(self.depths[:stage_idx])
        # The drop-path ramp is global across stages, not per stage: the index is
        # `block_start_idx + block_idx` over `sum(self.depths)` blocks, so stage 0
        # starts at 0.0 and the last block of the last stage reaches drop_path_rate.
        # `linear_drop_path_rates` already handles a single block (all zero).
        drop_path_rates = linear_drop_path_rates(sum(self.depths), self.drop_path_rate)
        drop_path_cls = StochasticDepth if self.stochastic_mode == 'depth' else StochasticGradient

        for block_idx in range(depth):
            drop_rate = drop_path_rates[block_start_idx + block_idx]
            block_name = f"stage{stage_idx + 1}_block{block_idx + 1}"

            block = self.BLOCK_CLASS(
                kernel_size=self.kernel_size,
                filters=dim,
                activation=self.activation,
                kernel_regularizer=self.kernel_regularizer,
                use_bias=self.use_bias,
                dropout_rate=self.dropout_rate,
                spatial_dropout_rate=self.spatial_dropout_rate,
                use_gamma=self.use_gamma,
                use_softorthonormal_regularizer=self.use_softorthonormal_regularizer,
                name=block_name
            )
            drop_path = drop_path_cls(
                drop_path_rate=drop_rate,
                name=f"{block_name}_drop_path"
            ) if drop_rate > 0 else None
            stage_blocks.append({"block": block, "drop_path": drop_path})
        self.stages.append(stage_blocks)

    def _build_head(self) -> None:
        """Build the classification head through the vision heads factory.

        The reference head (facebookresearch/ConvNeXt{,-V2} models/convnext*.py:
        ``self.norm(x.mean([-2, -1]))`` then ``self.head``): global average
        pooling, LayerNorm at the reference's epsilon 1e-6 (the factory's
        ``layer_norm`` default) and one linear layer, no hidden block.
        """
        self.classification_head = build_model_vision_head(
            VisionTaskType.CLASSIFICATION,
            {
                'num_classes': self.num_classes,
                'dropout_rate': self.dropout_rate,
                'normalization_type': 'layer_norm',
                'activation_type': 'gelu',
                'use_global_pooling': True,
                'pooling_type': 'avg',
                'use_attention': False,
                'block_dims': (),
                'pooled_normalization_type': 'layer_norm',
                'kernel_initializer': self.HEAD_INITIALIZER,
                'kernel_regularizer': self.kernel_regularizer,
                # An unnamed head is named `classification_head` in whichever
                # process creates it first and `classification_head_1`, `_2`, ...
                # afterwards. Weight transfer matches layers BY NAME, so a
                # checkpoint written by one process and loaded by another left
                # the classifier at its initialization while the report still
                # read as a success. Naming it makes the path stable.
                'name': 'classification_head',
            },
            self._head_config_overrides,
        )

    def build(self, input_shape: Any) -> None:
        """Materialize every sub-layer from `input_shape` by tracing `call`.

        `materialize_sublayers` traces `call` on symbolic inputs, so what
        gets built cannot drift from what gets called.

        :param input_shape: Shape (or nest of shapes) of the input to `call`.
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
        """Run the stem, the stages with their downsamplers, and the head.

        The residual add and the drop-path are applied HERE, not inside the
        block: ``BLOCK_CLASS`` is transform-only and returns ``F(x)``.

        :param inputs: Input tensor of shape `(batch_size, height, width, channels)`.
        :param training: Whether dropout and stochastic depth run in training mode.
        :return: The logits `(batch_size, num_classes)` (`include_top=True`) or
            the final stage's feature maps `(batch_size, H', W', dims[-1])`
            (`include_top=False`).
        """
        x = self.stem(inputs, training=training)

        for stage_idx, stage_blocks in enumerate(self.stages):
            if stage_idx > 0:
                downsampler = self.downsamplers[stage_idx - 1]
                x = downsampler["norm"](x)
                x = downsampler["conv"](x)

            for entry in stage_blocks:
                residual = x
                x = entry["block"](x, training=training)
                if entry["drop_path"]:
                    x = entry["drop_path"](x, training=training)
                x = keras.layers.add([residual, x])

        if self.include_top:
            return self.classification_head(x, training=training)['logits']

        return x

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Report the shape `call` returns, without building the model.

        The blocks inside a stage are shape-preserving, so only the stem, the
        downsamplers and the head move the shape. This is what lets the model sit in
        the Keras Functional API, where static shape inference is needed.

        :param input_shape: `(batch_size, height, width, channels)`, channels-last;
            the spatial axes may be `None`.
        :return: The logits shape `(batch_size, num_classes)` (`include_top=True`)
            or the final feature map `(batch_size, H', W', dims[-1])`
            (`include_top=False`).
        """
        # DECISION plan-2026-10-26T125959-7af851e5/D-037: V1 and V2 share this method in
        # `_ConvNeXtBase`, so "convnext_v2.py's body" in the anchor below is the same text. See decisions.md.
        # DECISION plan-2026-08-19T163559-499b6f0e/D-069: this method exists because
        # ConvNeXtV2 implements it and ConvNeXtV1 previously raised NotImplementedError for the same call. Mirrors convnext_v2.py's body. See decisions.md.
        shape = self.stem.compute_output_shape(input_shape)

        for downsampler in self.downsamplers:
            shape = downsampler["norm"].compute_output_shape(shape)
            shape = downsampler["conv"].compute_output_shape(shape)

        if self.include_top:
            # The head reports both keys; `call` returns the 'logits' one.
            shape = self.classification_head.compute_output_shape(shape)['logits']

        return shape

    def _pretrained_build_shape(self) -> Tuple[int, ...]:
        """Resolve a concrete `(H, W, C)` for the build that precedes a checkpoint load.

        :return: `(height, width, channels)` with any `None` spatial dim replaced by
            `PRETRAINED_BUILD_SPATIAL`.
        :raises ValueError: If `input_shape` has no channel count.
        """
        # (The anchor below says `self.input_shape`; the attribute is `self.input_shape_config`.)
        # DECISION plan-2026-08-14T233721-d4f9beb2/D-067: resolve None spatial dims
        # here rather than pass self.input_shape through — the default (None, None, 3) built (1, None, None, 3) and broke pretrained=<path> loads. See decisions.md.
        height, width, channels = self.input_shape_config
        if channels is None:
            raise ValueError(
                "Cannot materialize weights for a checkpoint load: input_shape "
                f"{self.input_shape_config} has no channel count. Pass a concrete "
                "input_shape=(height, width, channels) to the constructor."
            )
        height = self.PRETRAINED_BUILD_SPATIAL if height is None else height
        width = self.PRETRAINED_BUILD_SPATIAL if width is None else width
        return (int(height), int(width), int(channels))

    def load_pretrained_weights(
            self,
            weights_path: str,
            skip_mismatch: bool = True
    ) -> None:
        """Load a local `.keras` checkpoint into the model, layer by layer.

        The transfer is `load_pretrained_checkpoint`: it builds the model at
        `input_shape` (a `None` spatial dim at `PRETRAINED_BUILD_SPATIAL`) if
        it is unbuilt and takes a headed checkpoint into a headless model.
        Every layer outside the classification head must be restored, so a
        checkpoint of another variant, depth, width or kernel size raises and
        copies no weight. Only a head of another `num_classes` is skipped: it
        keeps its initialization and a warning names it. The archive's own
        config is compared too: a constructor argument that both hold and that
        is not in `_NON_STRUCTURAL_CONFIG_KEYS` must be equal, so a checkpoint
        of another `activation` or `stochastic_mode` is refused even when every
        weight shape fits. An argument only one side holds is not compared, and
        an archive in which no single model of this class can be found is
        checked by layers only (the load warns). Where the head is copied, its
        `head_config_overrides` must be equal too (dropout, initializers and
        regularizers aside).

        :param weights_path: Path to the weights file (`.keras` format).
        :param skip_mismatch: `True` (default) lets the head layers differ;
            `False` allows no difference anywhere. The inverse of the transfer
            helper's `strict`.
        :raises FileNotFoundError: If `weights_path` doesn't exist.
        :raises ValueError: If a layer outside the head is not restored, or
            the weights cannot be loaded.
        """
        load_pretrained_checkpoint(
            self, weights_path, self._pretrained_build_shape(), skip_mismatch,
            ignore_config_keys=_NON_STRUCTURAL_CONFIG_KEYS,
        )

    @classmethod
    def from_variant(
            cls,
            variant: str,
            num_classes: int = 1000,
            input_shape: Optional[Tuple[int, ...]] = None,
            pretrained: Union[bool, str] = False,
            **kwargs
    ) -> "_ConvNeXtBase":
        """Create a ConvNeXt model from a predefined variant.

        :param variant: A key of ``cls.MODEL_VARIANTS``.
        :param num_classes: Number of output classes.
        :param input_shape: Input shape. `None` resolves to `(None, None, 3)`; a
            checkpoint load then materializes weights at 224x224.
        :param pretrained: A path to a local weights file, `True` (raises
            `NotImplementedError`, since no public ConvNeXt weights ship with
            `dl_techniques`), or `False` (default, random init).
        :param kwargs: Passthrough to the constructor; an architecture field of
            the variant (e.g. `depths`, `dims`) is replaced by the value given here.
        :return: A configured instance of ``cls``.
        :raises ValueError: If `variant` is not recognized.
        :raises NotImplementedError: If `pretrained` is `True`.
        :raises FileNotFoundError: If `pretrained` is a path that doesn't exist.

        Example::

            model = ConvNeXtV1.from_variant("tiny", num_classes=10, input_shape=(32, 32, 3))
            backbone = ConvNeXtV1.from_variant("base", include_top=False)
            shallow = ConvNeXtV1.from_variant("tiny", depths=[1, 1, 1, 1])
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        # Raises for True instead of falling back to random init; do not reinstate a warn-and-return branch.
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
        """Return the model configuration for serialization."""
        config = {
            "num_classes": self.num_classes,
            "depths": self.depths,
            "dims": self.dims,
            "drop_path_rate": self.drop_path_rate,
            "stochastic_mode": self.stochastic_mode,
            "kernel_size": self.kernel_size,
            "activation": serialize_activation(self.activation),
            "use_bias": self.use_bias,
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "dropout_rate": self.dropout_rate,
            "spatial_dropout_rate": self.spatial_dropout_rate,
            "use_gamma": self.use_gamma,
            "use_softorthonormal_regularizer": self.use_softorthonormal_regularizer,
            "include_top": self.include_top,
            "input_shape": self.input_shape_config,
            "strides": self.strides,
            "head_config_overrides": self._head_config_overrides,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def summary(self, **kwargs) -> None:
        """Print the model summary; an unbuilt model is materialized first.

        :param kwargs: Passed to `keras.Model.summary`.
        """
        if not self.built:
            self.build((None, *self.input_shape_config))

        super().summary(**kwargs)


# ---------------------------------------------------------------------
