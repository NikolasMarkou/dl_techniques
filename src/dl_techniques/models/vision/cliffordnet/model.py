"""CliffordNet, an isotropic image classifier built from geometric-algebra blocks and no FFN.

This file holds :class:`CliffordNet`, its ``MODEL_VARIANTS`` table and the
``create_cliffordnet`` factory. Where a standard vision block separates a token
mixer from a channel-mixing MLP, a CliffordNet block does both with the Clifford
geometric product `u v = u . v + u ^ v` of two per-pixel channel streams: the
symmetric part acts as an attention score and the antisymmetric part carries the
disagreement between the streams. The product is bilinear, so channels mix
without an FFN. Each block takes a pointwise detail stream and a
depthwise-convolved context stream and samples only a few channel offsets of
their pairwise product (`shifts`), costing `O(N * D * |shifts|)` instead of
`O(D^2)`. The model is isotropic: one patch-embedding stem, then `depth`
identical blocks at constant width, with no stage structure. `CliffordNetBlock`
returns the gamma-scaled update alone, so the residual add and the
stochastic-depth gate sit in this model's `call`. The head pools before it
normalizes. No pretrained weights are distributed, so `pretrained=True` raises
`NotImplementedError`; pass a local `.keras` path instead.

References:
    - Ji, Z., 2026. CliffordNet: All You Need is Geometric Algebra.
      (https://arxiv.org/abs/2601.06793)
    - Brandstetter et al., 2023. Clifford Neural Layers for PDE Modeling.
      (https://arxiv.org/abs/2209.04934)
    - Ruhe et al., 2023. Geometric Clifford Algebra Networks.
      (https://arxiv.org/abs/2302.06594)
    - Yu et al., 2022. MetaFormer Is Actually What You Need for Vision.
      (https://arxiv.org/abs/2111.11418)
    - Touvron et al., 2021. Going Deeper with Image Transformers (CaiT).
      (LayerScale) (https://arxiv.org/abs/2103.17239)
    - Huang et al., 2016. Deep Networks with Stochastic Depth.
      (https://arxiv.org/abs/1603.09382)
"""

from __future__ import annotations

import os
import keras
from keras import initializers, regularizers
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.geometric.clifford_block import (
    CliMode,
    CtxMode,
    CliffordNetBlock,
)
from dl_techniques.layers.regularization.stochastic_depth import StochasticDepth
from dl_techniques.utils.logger import logger
from dl_techniques.utils.drop_path import linear_drop_path_rates
from dl_techniques.utils.weight_transfer import load_weights_from_checkpoint
from dl_techniques.utils.keras_registration import register_dl_technique

# Matches the reference: trunc_normal_(std=0.02) for all Conv2d and Linear.
_DEFAULT_KERNEL_INIT = initializers.TruncatedNormal(stddev=0.02)

# DECISION plan-2026-08-23T091307-9a110062/D-480: momentum stays 0.9; Keras and
# torch define it oppositely, so the reference's 0.1 is this value. See
# decisions.md.
_STEM_BN_MOMENTUM = 0.9

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.cliffordnet.model")
class CliffordNet(keras.Model):
    """Classify images with a stack of identical geometric-algebra blocks.

    Constant width throughout, so the only spatial reduction is the stem's.

    Architecture:

    .. code-block:: text

        input [B, H, W, C_in]
              │
              ▼
        ┌──────────────────────────────┐
        │ stem, patch_size dependent   │
        │ ending in batch norm         │
        └──────────────────────────────┘
              │ [B, H/p, W/p, D]
              ▼
        ┌──────────────────────────────┐
        │ x + drop_path(block(x))      │  x depth
        └──────────────────────────────┘
              │ [B, H/p, W/p, D]
              ▼
        ┌──────────────────────────────┐
        │ global average pool          │
        └──────────────────────────────┘
              │ [B, D]
              ▼
        ┌──────────────────────────────┐
        │ layer norm                   │
        └──────────────────────────────┘
              │
              ▼
        ┌──────────────────────────────┐
        │ dropout                      │  (dropout_rate > 0)
        └──────────────────────────────┘
              │
              ▼
        ┌──────────────────────────────┐
        │ dense to num_classes         │
        └──────────────────────────────┘
              │
              ▼
        logits [B, num_classes]

    Pooling comes before the normalization, the reverse of the usual order.

    Block wiring:

    .. code-block:: text

        x
        ├──────────────────────────────┐
        ▼                              │
        clifford block, gamma-scaled   │
        │                              │
        ▼                              │
        stochastic depth               │
        │                              │
        ▼                              │
       (+)◄────────────────────────────┘
        │
        ▼
        out

    ``StochasticDepth(0.0)`` is the identity, so a rate of 0 gives exactly
    ``x + block(x)``.

    Stem variants:

    .. code-block:: text

        patch_size   layers                                     stride
        1            conv3 D/2, bn, silu, conv3 D, bn           1
        2            conv3 D stride 2, bn                       2
        4            conv3 D/2 /2, bn, silu, conv3 D /2, bn     4
        other p      conv p D stride p, bn                      p

    The two-conv stems are always bias-free; the single-conv stems follow
    ``use_bias``. Every variant ends in the shared ``stem_norm`` batch norm.

    Variants:

    .. code-block:: text

        variant   channels  depth  shifts           global context
        nano           128     12  1, 2             no
        lite           128     12  1, 2, 4, 8, 16   no
        lite_g         128     12  1, 2, 4, 8, 16   yes

    All three use ``patch_size=2``, ``cli_mode='full'``, ``ctx_mode='diff'``,
    ``layer_scale_init=1e-5`` and ``TruncatedNormal(0.02)`` kernels.

    :param num_classes: Number of output classes. Must be positive.
    :param channels: Feature dimensionality ``D`` (constant throughout). Must
        be positive.
    :param depth: Number of CliffordNet blocks ``L``. Must be positive.
    :param patch_size: Total stride of the patch-embedding stem, and the
        selector for which stem is built. Must be positive. ``patch_size=2``
        is optimal for CIFAR-scale inputs.
    :param shifts: Channel-shift offsets for the sparse rolling product.
        ``None`` resolves to ``[1, 2]``.
    :param cli_mode: ``"inner"`` | ``"wedge"`` | ``"full"`` (default). Not
        validated here; an unknown value reaches the block.
    :param ctx_mode: ``"diff"`` (default) | ``"abs"``. Not validated here.
    :param use_global_context: Add the global-average-pool gFFN-G branch.
    :param layer_scale_init: Initial LayerScale value. Defaults to ``1e-5``.
    :param stochastic_depth_rate: Maximum DropPath rate (linearly scheduled
        across blocks). Defaults to ``0.0``.
    :param dropout_rate: Pre-classifier head dropout. At ``0.0``, the default,
        no Dropout layer is created.
    :param use_bias: Whether the Dense layers, the classifier and the
        single-conv stems use a bias.
    :param kernel_initializer: Kernel initializer. Defaults to Keras'
        ``glorot_uniform``, while every ``MODEL_VARIANTS`` entry overrides it
        with ``TruncatedNormal(0.02)`` to match the reference.
    :param bias_initializer: Bias initializer.
    :param kernel_regularizer: Kernel regularizer.
    :param bias_regularizer: Bias regularizer.
    :param kwargs: Passed to :class:`keras.Model`.

    :raises ValueError: If ``num_classes``, ``channels``, ``depth`` or
        ``patch_size`` is not positive.
    """

    LAYERNORM_EPSILON: float = 1e-6

    def __init__(
        self,
        num_classes: int,
        channels: int = 128,
        depth: int = 12,
        patch_size: int = 2,
        shifts: Optional[List[int]] = None,
        cli_mode: CliMode = "full",
        ctx_mode: CtxMode = "diff",
        use_global_context: bool = False,
        layer_scale_init: float = 1e-5,
        stochastic_depth_rate: float = 0.0,
        dropout_rate: float = 0.0,
        use_bias: bool = True,
        kernel_initializer: Any = "glorot_uniform",
        bias_initializer: Any = "zeros",
        kernel_regularizer: Optional[Any] = None,
        bias_regularizer: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")

        self.num_classes = num_classes
        self.channels = channels
        self.depth = depth
        self.patch_size = patch_size
        self.shifts = shifts if shifts is not None else [1, 2]
        self.cli_mode = cli_mode
        self.ctx_mode = ctx_mode
        self.use_global_context = use_global_context
        self.layer_scale_init = layer_scale_init
        self.stochastic_depth_rate = stochastic_depth_rate
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self._build_stem()
        self._build_blocks()
        self._build_head()

        logger.info(
            f"Created CliffordNet (channels={channels}, depth={depth}, "
            f"patch_size={patch_size}, shifts={self.shifts}, "
            f"cli_mode={cli_mode}, ctx_mode={ctx_mode}, "
            f"use_global_context={use_global_context})"
        )

    def _build_stem(self) -> None:
        """Build the patch-embedding layers for the configured ``patch_size``.

        Sizes 1, 2 and 4 get their own stems and anything else falls back to a
        single conv whose kernel and stride both equal ``patch_size``. See the
        stem table in the class docstring.

        :return: Nothing.
        :rtype: None
        """
        _conv_kw: Dict[str, Any] = dict(
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
        )

        if self.patch_size == 1:
            self.stem_conv1 = keras.layers.Conv2D(
                filters=self.channels // 2,
                kernel_size=3,
                strides=1,
                padding="same",
                use_bias=False,
                name="stem_conv1",
                **_conv_kw,
            )
            self.stem_bn1 = keras.layers.BatchNormalization(
                name="stem_bn1", momentum=_STEM_BN_MOMENTUM
            )
            self.stem_conv2 = keras.layers.Conv2D(
                filters=self.channels,
                kernel_size=3,
                strides=1,
                padding="same",
                use_bias=False,
                name="stem_conv2",
                **_conv_kw,
            )
        elif self.patch_size == 2:
            self.stem_conv = keras.layers.Conv2D(
                filters=self.channels,
                kernel_size=3,
                strides=2,
                padding="same",
                use_bias=self.use_bias,
                name="stem_conv",
                **_conv_kw,
            )
        elif self.patch_size == 4:
            # Two stride-2 convs reach 4x, rather than one stride-4 kernel.
            self.stem_conv1 = keras.layers.Conv2D(
                filters=self.channels // 2,
                kernel_size=3,
                strides=2,
                padding="same",
                use_bias=False,
                name="stem_conv1",
                **_conv_kw,
            )
            self.stem_bn1 = keras.layers.BatchNormalization(
                name="stem_bn1", momentum=_STEM_BN_MOMENTUM
            )
            self.stem_conv2 = keras.layers.Conv2D(
                filters=self.channels,
                kernel_size=3,
                strides=2,
                padding="same",
                use_bias=False,
                name="stem_conv2",
                **_conv_kw,
            )
        else:
            self.stem_conv = keras.layers.Conv2D(
                filters=self.channels,
                kernel_size=self.patch_size,
                strides=self.patch_size,
                padding="same",
                use_bias=self.use_bias,
                name="stem_conv",
                **_conv_kw,
            )

        # Every stem variant ends here, matching GeometricStem.norm.
        self.stem_norm = keras.layers.BatchNormalization(
            name="stem_norm", momentum=_STEM_BN_MOMENTUM
        )

    def _build_blocks(self) -> None:
        """Build the block list, pairing each block with its drop-path gate.

        The rates ramp linearly from 0 to ``stochastic_depth_rate`` across the
        ``depth`` blocks.

        :return: Nothing.
        :rtype: None
        """
        drop_rates = linear_drop_path_rates(self.depth, self.stochastic_depth_rate)

        _block_kw: Dict[str, Any] = dict(
            channels=self.channels,
            shifts=self.shifts,
            cli_mode=self.cli_mode,
            ctx_mode=self.ctx_mode,
            use_global_context=self.use_global_context,
            layer_scale_init=self.layer_scale_init,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
        )

        self.blocks_list: List[Dict[str, Any]] = []
        for i in range(self.depth):
            block = CliffordNetBlock(
                name=f"clifford_block_{i}",
                **_block_kw,
            )
            # The blocks are transform-only, so the residual and the gate are
            # applied around them in call().
            drop_path = StochasticDepth(
                drop_path_rate=drop_rates[i],
                name=f"clifford_drop_path_{i}",
            )
            self.blocks_list.append({"block": block, "drop_path": drop_path})

    def _build_head(self) -> None:
        """Build the classifier head layers.

        Pooling precedes the normalization, matching the reference
        ``forward()``, and the dropout layer exists only when
        ``dropout_rate`` is above 0.

        :return: Nothing.
        :rtype: None
        """
        self.global_pool = keras.layers.GlobalAveragePooling2D(
            name="global_pool"
        )
        self.head_norm = keras.layers.LayerNormalization(
            epsilon=self.LAYERNORM_EPSILON, name="head_norm"
        )
        self.head_dropout = (
            keras.layers.Dropout(self.dropout_rate, name="head_dropout")
            if self.dropout_rate > 0.0
            else None
        )
        self.classifier = keras.layers.Dense(
            self.num_classes,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="classifier",
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the model via a symbolic forward pass.

        A 3-tuple is accepted and read as an unbatched shape, with the batch
        axis prepended.

        :param input_shape: Input tensor shape ``(B, H, W, C_in)``.
        :return: Nothing.
        :rtype: None
        """
        if len(input_shape) == 3:
            build_shape = (None,) + tuple(input_shape)
        else:
            build_shape = tuple(input_shape)
        dummy = keras.KerasTensor(build_shape)
        _ = self.call(dummy)
        super().build(input_shape)

    def _apply_stem(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool],
    ) -> keras.KerasTensor:
        """Apply the patch embedding stem.

        The two-conv stems (``patch_size`` 1 and 4) put a batch norm and a SiLU
        between their convolutions; the others go straight to ``stem_norm``.

        :param inputs: Raw image batch ``(B, H, W, C_in)``.
        :param training: Whether in training mode (affects BatchNorm).
        :return: Embedded feature map ``(B, h, w, channels)``.
        """
        if self.patch_size in (1, 4):
            x = keras.activations.silu(
                self.stem_bn1(self.stem_conv1(inputs), training=training)
            )
            x = self.stem_conv2(x)
        else:
            x = self.stem_conv(inputs)

        return self.stem_norm(x, training=training)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Run the forward pass.

        :param inputs: Image batch ``(B, H, W, C_in)``.
        :param training: Whether in training mode.
        :return: Class logits ``(B, num_classes)``, with no output activation
            applied.
        """
        x = self._apply_stem(inputs, training=training)

        for block_info in self.blocks_list:
            x = x + block_info["drop_path"](
                block_info["block"](x, training=training), training=training
            )

        # Pooling before the norm, so the norm runs over channels alone.
        x = self.global_pool(x)
        x = self.head_norm(x)

        if self.head_dropout is not None:
            x = self.head_dropout(x, training=training)

        return self.classifier(x)

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape.

        :param input_shape: ``(B, H, W, C_in)``
        :return: ``(B, num_classes)``
        """
        return (input_shape[0], self.num_classes)

    def load_pretrained_weights(
        self,
        weights_path: str,
        skip_mismatch: bool = True
    ) -> None:
        """Load weights from a local checkpoint into this model.

        Weights are transferred layer-by-layer via
        :func:`dl_techniques.utils.weight_transfer.load_weights_from_checkpoint`,
        the canonical replacement for ``self.load_weights(by_name=True)``, which
        raises on ``.keras`` files in Keras 3.8+. Any failure inside the
        transfer is re-raised as ``ValueError``.

        :param weights_path: Path to the ``.keras`` weights file.
        :param skip_mismatch: Skip layers whose shapes do not match, which is
            what allows a checkpoint with a different ``num_classes`` to load.
            Passed through as ``strict=not skip_mismatch``.
        :return: Nothing.
        :rtype: None
        :raises FileNotFoundError: If ``weights_path`` does not exist.
        :raises ValueError: If weights cannot be loaded.
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        try:
            logger.info(f"Loading pretrained weights from {weights_path}")
            report = load_weights_from_checkpoint(
                target=self,
                ckpt_path=weights_path,
                skip_prefixes=(),
                strict=not skip_mismatch,
            )
            logger.info(report.summary_string())
            note = (
                " Layers with shape mismatches were skipped."
                if skip_mismatch
                else ""
            )
            logger.info(f"Weights loaded successfully.{note}")
        except Exception as exc:
            raise ValueError(
                f"Failed to load weights from {weights_path}: {exc}"
            ) from exc

    @staticmethod
    def _download_weights(
        variant: str,
        dataset: str = "cifar100",
        cache_dir: Optional[str] = None,
    ) -> str:
        """Refuse to download, since no public CliffordNet checkpoints exist.

        :param variant: Model variant name (e.g. ``"nano"``, ``"lite"``).
        :param dataset: Dataset the weights were trained on.
        :param cache_dir: Directory to cache downloaded weights (unused).
        :return: Never returns — always raises.
        :raises NotImplementedError: Always. To load local weights, pass
            ``pretrained='/path/to/weights.keras'`` to
            :meth:`CliffordNet.from_variant`.
        """
        # DECISION plan_2026-05-11_0090b0b8/D-001: raise rather than return a
        # random-init model; NotImplementedError is outside from_variant's except
        # clause, so it reaches the caller. See decisions.md.
        raise NotImplementedError(
            "No public CliffordNet checkpoints are distributed. "
            "To load local weights, pass "
            "`pretrained='/path/to/weights.keras'` instead of "
            "`pretrained=True`."
        )

    def get_config(self) -> Dict[str, Any]:
        """Return serialisable configuration.

        :return: Dictionary with all constructor arguments.
        """
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "channels": self.channels,
                "depth": self.depth,
                "patch_size": self.patch_size,
                "shifts": self.shifts,
                "cli_mode": self.cli_mode,
                "ctx_mode": self.ctx_mode,
                "use_global_context": self.use_global_context,
                "layer_scale_init": self.layer_scale_init,
                "stochastic_depth_rate": self.stochastic_depth_rate,
                "dropout_rate": self.dropout_rate,
                "use_bias": self.use_bias,
                "kernel_initializer": initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": initializers.serialize(
                    self.bias_initializer
                ),
                "kernel_regularizer": regularizers.serialize(
                    self.kernel_regularizer
                ),
                "bias_regularizer": regularizers.serialize(
                    self.bias_regularizer
                ),
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CliffordNet":
        """Reconstruct model from configuration dict.

        The serialized initializers are left as dicts, since
        ``initializers.get`` accepts a config dict in ``__init__``.

        :param config: Dictionary produced by :meth:`get_config`.
        :return: New :class:`CliffordNet` instance.
        """
        for key in ("kernel_regularizer", "bias_regularizer"):
            if config.get(key) and isinstance(config[key], dict):
                config[key] = regularizers.deserialize(config[key])
        return cls(**config)

    def summary(self, **kwargs: Any) -> None:
        """Print the Keras summary, then log the resolved configuration.

        An unbuilt model is built first against a 3-channel input of unknown
        spatial size.

        :param kwargs: Forwarded to :meth:`keras.Model.summary`.
        :return: Nothing.
        :rtype: None
        """
        if not self.built:
            logger.warning(
                "Model is not built; calling build() with a symbolic input."
            )
            dummy = keras.KerasTensor((None, None, None, 3))
            self.build(dummy.shape)

        super().summary(**kwargs)

        logger.info("CliffordNet configuration:")
        logger.info(f"  channels            : {self.channels}")
        logger.info(f"  depth               : {self.depth}")
        logger.info(f"  patch_size          : {self.patch_size}")
        logger.info(f"  shifts              : {self.shifts}")
        logger.info(f"  cli_mode            : {self.cli_mode}")
        logger.info(f"  ctx_mode            : {self.ctx_mode}")
        logger.info(f"  use_global_context  : {self.use_global_context}")
        logger.info(f"  stochastic_depth    : {self.stochastic_depth_rate}")
        logger.info(f"  dropout_rate        : {self.dropout_rate}")
        logger.info(f"  num_classes         : {self.num_classes}")

    MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
        "nano": dict(
            channels=128,
            depth=12,
            patch_size=2,
            shifts=[1, 2],
            cli_mode="full",
            ctx_mode="diff",
            use_global_context=False,
            layer_scale_init=1e-5,
            kernel_initializer=_DEFAULT_KERNEL_INIT,
        ),
        "lite": dict(
            channels=128,
            depth=12,
            patch_size=2,
            shifts=[1, 2, 4, 8, 16],
            cli_mode="full",
            ctx_mode="diff",
            use_global_context=False,
            layer_scale_init=1e-5,
            kernel_initializer=_DEFAULT_KERNEL_INIT,
        ),
        "lite_g": dict(
            channels=128,
            depth=12,
            patch_size=2,
            shifts=[1, 2, 4, 8, 16],
            cli_mode="full",
            ctx_mode="diff",
            use_global_context=True,
            layer_scale_init=1e-5,
            kernel_initializer=_DEFAULT_KERNEL_INIT,
        ),
    }

    @classmethod
    def from_variant(
        cls,
        variant: str,
        num_classes: int,
        pretrained: Union[bool, str] = False,
        weights_dataset: str = "cifar100",
        cache_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> "CliffordNet":
        """Create a :class:`CliffordNet` from a predefined variant.

        :param variant: One of ``"nano"``, ``"lite"``, ``"lite_g"``.
        :param num_classes: Number of output classes.
        :param pretrained: A local path to a ``.keras`` weights file. ``True``
            attempts a download, which always raises ``NotImplementedError``
            since no checkpoints are hosted.
        :param weights_dataset: Dataset key for pretrained weights. The
            classifier-skip decision compares ``num_classes`` against 100
            whatever this says.
        :param cache_dir: Directory to cache downloaded weights.
        :param kwargs: Override any variant default.
        :return: Configured :class:`CliffordNet` instance.
        :raises ValueError: If ``variant`` is not recognised.
        :raises NotImplementedError: If ``pretrained`` is ``True``.

        Example::

            # Pre-defined variant with custom num_classes
            model = CliffordNet.from_variant("lite", num_classes=100)

            # Override a hyperparameter
            model = CliffordNet.from_variant(
                "nano", num_classes=10, stochastic_depth_rate=0.1
            )

            # Load from local weights file
            model = CliffordNet.from_variant(
                "lite", num_classes=100,
                pretrained="path/to/weights.keras"
            )
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Available: {list(cls.MODEL_VARIANTS.keys())}"
            )

        defaults = dict(cls.MODEL_VARIANTS[variant])
        defaults.update(kwargs)

        logger.info(f"Creating CliffordNet-{variant.upper()}")

        load_weights_path: Optional[str] = None
        skip_mismatch: bool = False

        if pretrained:
            if isinstance(pretrained, str):
                load_weights_path = pretrained
                logger.info(
                    f"Will load weights from local file: {load_weights_path}"
                )
            else:
                try:
                    load_weights_path = cls._download_weights(
                        variant=variant,
                        dataset=weights_dataset,
                        cache_dir=cache_dir,
                    )
                # DECISION plan_2026-05-11_0090b0b8/D-001: only transport errors
                # degrade to random init here; do not widen this to
                # NotImplementedError. See decisions.md.
                except (IOError, OSError, ValueError) as exc:
                    logger.warning(
                        f"Failed to download pretrained weights: {exc}. "
                        "Continuing with random initialisation."
                    )
                    load_weights_path = None

            # The hosted checkpoints would be CIFAR-100, so a different head
            # width means the classifier cannot be transferred.
            pretrained_classes = 100
            if num_classes != pretrained_classes:
                skip_mismatch = True
                logger.info(
                    f"num_classes ({num_classes}) differs from pretrained "
                    f"({pretrained_classes}). Classifier weights will be skipped."
                )

        model = cls(num_classes=num_classes, **defaults)

        if load_weights_path:
            try:
                model.load_pretrained_weights(
                    weights_path=load_weights_path,
                    skip_mismatch=skip_mismatch
                )
            except Exception as exc:
                logger.error(f"Failed to load pretrained weights: {exc}")
                raise

        return model

    @classmethod
    def nano(cls, num_classes: int, **kwargs: Any) -> "CliffordNet":
        """Create CliffordNet-Nano.

        ``channels=128``, ``depth=12``, ``shifts=[1, 2]``, differential
        context, no global branch.

        :param num_classes: Number of output classes.
        :param kwargs: Override any default hyperparameter.
        :return: Configured :class:`CliffordNet` instance.
        """
        return cls.from_variant("nano", num_classes=num_classes, **kwargs)

    @classmethod
    def lite(cls, num_classes: int, **kwargs: Any) -> "CliffordNet":
        """Create CliffordNet-Lite.

        ``channels=128``, ``depth=12``, ``shifts=[1, 2, 4, 8, 16]``,
        differential context, no global branch.

        :param num_classes: Number of output classes.
        :param kwargs: Override any default hyperparameter.
        :return: Configured :class:`CliffordNet` instance.
        """
        return cls.from_variant("lite", num_classes=num_classes, **kwargs)

    @classmethod
    def lite_g(cls, num_classes: int, **kwargs: Any) -> "CliffordNet":
        """Create CliffordNet-Lite with the gFFN-G branch.

        Same widths and shifts as ``lite``, plus the global-average-pool
        context branch.

        :param num_classes: Number of output classes.
        :param kwargs: Override any default hyperparameter.
        :return: Configured :class:`CliffordNet` instance.
        """
        return cls.from_variant("lite_g", num_classes=num_classes, **kwargs)


def create_cliffordnet(
        variant: str = "lite",
        num_classes: int = 100,
        pretrained: Union[bool, str] = False,
        weights_dataset: str = "cifar100",
        cache_dir: Optional[str] = None,
        **kwargs: Any,
) -> "CliffordNet":
    """Create a CliffordNet model from a variant name.

    A thin module-level factory that delegates to
    :meth:`CliffordNet.from_variant`, mirroring
    :func:`dl_techniques.models.vision.resnet.model.create_resnet` and
    :func:`dl_techniques.models.language.tree_transformer.model.create_tree_transformer`.

    :param variant: Model variant, one of ``"nano"``, ``"lite"``, ``"lite_g"``.
    :type variant: str
    :param num_classes: Number of output classes.
    :type num_classes: int
    :param pretrained: A local ``.keras`` weights path, or True to raise `NotImplementedError` (no public CliffordNet weights are hosted).
    :type pretrained: Union[bool, str]
    :param weights_dataset: Dataset key for pretrained weights, kept for API parity with other models.
    :type weights_dataset: str
    :param cache_dir: Directory to cache downloaded weights.
    :type cache_dir: Optional[str]
    :param kwargs: Additional arguments forwarded to `CliffordNet.from_variant` (e.g. `stochastic_depth_rate`, `dropout_rate`).
    :return: A `CliffordNet` instance.
    :rtype: CliffordNet
    :raises ValueError: If `variant` is not recognised.
    :raises NotImplementedError: If `pretrained` is `True`.

    Example:
        >>> # Create CliffordNet-Lite with random init for CIFAR-100
        >>> model = create_cliffordnet("lite", num_classes=100)
        >>>
        >>> # Smaller variant for CIFAR-10 with override
        >>> model = create_cliffordnet(
        ...     "nano", num_classes=10, stochastic_depth_rate=0.1
        ... )
        >>>
        >>> # Load from local weights file
        >>> model = create_cliffordnet(
        ...     "lite", num_classes=100, pretrained="path/to/weights.keras"
        ... )
    """
    return CliffordNet.from_variant(
        variant,
        num_classes=num_classes,
        pretrained=pretrained,
        weights_dataset=weights_dataset,
        cache_dir=cache_dir,
        **kwargs,
    )

# ---------------------------------------------------------------------
