"""
CliffordNet isotropic vision model.

Implements the full classification backbone from arXiv:2601.06793v2.

Architecture: patch embedding -> L x CliffordNetBlock
-> GlobalAvgPool -> LayerNorm -> Dense classifier head.

The patch embedding follows the original ``GeometricStem`` design:

- ``patch_size=1``: two-stage ``Conv2D(C//2) + BN + SiLU + Conv2D(C)``
  (no downsampling).
- ``patch_size=2``: single ``Conv2D(C, kernel=3, stride=2)`` (efficient
  for CIFAR-scale, matches the original single-conv stem).
- ``patch_size=4``: two-stage ``Conv2D(C//2, stride=2) + BN + SiLU +
  Conv2D(C, stride=2)`` (ImageNet-scale).
- other: generic ``Conv2D(C, kernel=patch_size, stride=patch_size)``.

All stems are followed by ``BatchNormalization`` (matching the original
``GeometricStem.norm``).

The classification head applies ``GlobalAveragePooling2D`` **first**, then
``LayerNormalization`` on the pooled ``(B, D)`` vector (matching the
original ``forward()`` order).

Pre-defined variants
--------------------
- ``CliffordNet.nano``   -- ~1.4 M params  (channels=128, depth=12, shifts=[1,2])
- ``CliffordNet.lite``   -- ~2.6 M params  (channels=128, depth=12, shifts=[1,2,4,8,16])
- ``CliffordNet.lite_g`` -- ~3.4 M params  (Lite + global-context branch)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import keras
from keras import initializers, regularizers

from dl_techniques.layers.geometric.clifford_block import (
    CliMode,
    CtxMode,
    CliffordNetBlock,
)
from dl_techniques.utils.logger import logger

# Match the reference: trunc_normal_(std=0.02) for all Conv2d and Linear.
_DEFAULT_KERNEL_INIT = initializers.TruncatedNormal(stddev=0.02)


# ---------------------------------------------------------------------------
# Helper: stochastic-depth rate schedule
# ---------------------------------------------------------------------------


def _linear_drop_path_rates(num_blocks: int, max_rate: float) -> List[float]:
    """Return linearly spaced drop-path rates from 0 to ``max_rate``.

    :param num_blocks: Total number of blocks.
    :param max_rate: Maximum (last-block) drop probability.
    :return: List of per-block drop-path rates.
    """
    if num_blocks <= 1:
        return [0.0] * num_blocks
    step = max_rate / (num_blocks - 1)
    return [round(i * step, 6) for i in range(num_blocks)]


# ===========================================================================
# CliffordNet
# ===========================================================================


@keras.saving.register_keras_serializable()
class CliffordNet(keras.Model):
    """Isotropic CliffordNet vision backbone.

    Follows the columnar (MetaFormer-isotropic) design: patch embedding to
    a fixed ``channels`` feature map, then *L* identical
    :class:`~dl_techniques.layers.geometric.clifford_block.CliffordNetBlock`
    layers, global average pooling, layer normalisation, and a Dense
    classifier.

    The patch embedding uses the same ``GeometricStem`` design as the
    original: a ``BatchNormalization`` (not ``LayerNormalization``) follows
    the convolution(s), and for ``patch_size=2`` the convolution uses
    ``kernel_size=3`` with ``strides=2``.

    The head applies ``GlobalAveragePooling2D`` first, then
    ``LayerNormalization`` on the resulting ``(B, channels)`` vector, which
    matches the original ``CliffordNet.forward()`` order.

    :param num_classes: Number of output classes.
    :param channels: Feature dimensionality ``D`` (constant throughout).
    :param depth: Number of CliffordNet blocks ``L``.
    :param patch_size: Stride of the patch-embedding convolution.
        ``patch_size=2`` is optimal for CIFAR-scale inputs.
    :param shifts: Channel-shift offsets for the sparse rolling product.
    :param cli_mode: ``"inner"`` | ``"wedge"`` | ``"full"`` (default).
    :param ctx_mode: ``"diff"`` (default) | ``"abs"``.
    :param use_global_context: Add the global-average-pool gFFN-G branch.
    :param layer_scale_init: Initial LayerScale value. Defaults to ``1e-5``.
    :param stochastic_depth_rate: Maximum DropPath rate (linearly scheduled
        across blocks). Defaults to ``0.0``.
    :param dropout_rate: Pre-classifier head dropout. Defaults to ``0.0``.
    :param use_bias: Whether Dense / projection layers use bias.
    :param kernel_initializer: Kernel initializer.
    :param bias_initializer: Bias initializer.
    :param kernel_regularizer: Kernel regularizer.
    :param bias_regularizer: Bias regularizer.
    :param kwargs: Passed to :class:`keras.Model`.

    **Call arguments:**

    :param inputs: Image tensor ``(B, H, W, C_in)``.
    :param training: Python bool or ``None``.

    :returns: Logit tensor ``(B, num_classes)``.
    """

    # Architecture constants
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

        # Store configuration
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

        # Build sub-component groups
        self._build_stem()
        self._build_blocks()
        self._build_head()

        logger.info(
            f"Created CliffordNet (channels={channels}, depth={depth}, "
            f"patch_size={patch_size}, shifts={self.shifts}, "
            f"cli_mode={cli_mode}, ctx_mode={ctx_mode}, "
            f"use_global_context={use_global_context})"
        )

    # ------------------------------------------------------------------
    # Private builder helpers
    # ------------------------------------------------------------------

    def _build_stem(self) -> None:
        """Build and assign patch-embedding (GeometricStem) layers."""
        _conv_kw: Dict[str, Any] = dict(
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
        )

        if self.patch_size == 1:
            # Two-conv stem, no spatial downsampling.
            self.stem_conv1 = keras.layers.Conv2D(
                filters=self.channels // 2,
                kernel_size=3,
                strides=1,
                padding="same",
                use_bias=False,
                name="stem_conv1",
                **_conv_kw,
            )
            self.stem_bn1 = keras.layers.BatchNormalization(name="stem_bn1")
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
            # Single 3x3 conv with stride 2 (CIFAR-scale).
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
            # Two stride-2 convs (4x total downsampling, ImageNet-scale).
            self.stem_conv1 = keras.layers.Conv2D(
                filters=self.channels // 2,
                kernel_size=3,
                strides=2,
                padding="same",
                use_bias=False,
                name="stem_conv1",
                **_conv_kw,
            )
            self.stem_bn1 = keras.layers.BatchNormalization(name="stem_bn1")
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
            # Generic: square kernel equal to patch_size.
            self.stem_conv = keras.layers.Conv2D(
                filters=self.channels,
                kernel_size=self.patch_size,
                strides=self.patch_size,
                padding="same",
                use_bias=self.use_bias,
                name="stem_conv",
                **_conv_kw,
            )

        # Final BatchNorm applied to every stem variant (matches GeometricStem.norm).
        self.stem_norm = keras.layers.BatchNormalization(name="stem_norm")

    def _build_blocks(self) -> None:
        """Build and assign the CliffordNet block list with linear drop-path schedule."""
        drop_rates = _linear_drop_path_rates(self.depth, self.stochastic_depth_rate)

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
                drop_path_rate=drop_rates[i],
                name=f"clifford_block_{i}",
                **_block_kw,
            )
            self.blocks_list.append({"block": block})

    def _build_head(self) -> None:
        """Build and assign classifier head layers.

        Order: GlobalAveragePooling2D -> LayerNorm -> (Dropout) -> Dense.
        GAP is applied *before* LayerNorm, matching the original ``forward()``.
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

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the model via a symbolic forward pass.

        :param input_shape: Input tensor shape ``(B, H, W, C_in)``.
        """
        super().build(input_shape)
        if len(input_shape) == 3:
            build_shape = (None,) + tuple(input_shape)
        else:
            build_shape = tuple(input_shape)
        dummy = keras.KerasTensor(build_shape)
        _ = self.call(dummy)

    # ------------------------------------------------------------------
    # Forward pass helpers
    # ------------------------------------------------------------------

    def _apply_stem(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool],
    ) -> keras.KerasTensor:
        """Apply the patch embedding stem.

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

    # ------------------------------------------------------------------
    # Call
    # ------------------------------------------------------------------

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Forward pass.

        :param inputs: Image batch ``(B, H, W, C_in)``.
        :param training: Whether in training mode.
        :return: Class logits ``(B, num_classes)``.
        """
        x = self._apply_stem(inputs, training=training)

        for block_info in self.blocks_list:
            x = block_info["block"](x, training=training)

        # Head: GAP first, then LayerNorm (matches original forward order).
        x = self.global_pool(x)           # (B, channels)
        x = self.head_norm(x)             # LayerNorm on (B, channels)

        if self.head_dropout is not None:
            x = self.head_dropout(x, training=training)

        return self.classifier(x)

    # ------------------------------------------------------------------
    # Shape inference
    # ------------------------------------------------------------------

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape.

        :param input_shape: ``(B, H, W, C_in)``
        :return: ``(B, num_classes)``
        """
        return (input_shape[0], self.num_classes)

    # ------------------------------------------------------------------
    # Weight loading helpers
    # ------------------------------------------------------------------

    def load_pretrained_weights(
        self,
        weights_path: str,
        skip_mismatch: bool = True,
        by_name: bool = True,
    ) -> None:
        """Load pretrained weights into the model.

        Handles loading with smart mismatch handling, useful when the
        number of classes differs or when loading backbone-only weights.

        :param weights_path: Path to the ``.keras`` weights file.
        :param skip_mismatch: Skip layers with mismatched shapes. Useful
            when loading weights with different ``num_classes``.
        :param by_name: Load weights by layer name.
        :raises FileNotFoundError: If ``weights_path`` does not exist.
        :raises ValueError: If weights cannot be loaded.
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        try:
            logger.info(f"Loading pretrained weights from {weights_path}")
            self.load_weights(
                weights_path,
                skip_mismatch=skip_mismatch,
                by_name=by_name,
            )
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
        """Download pretrained weights from a registered URL.

        :param variant: Model variant name (e.g. ``"nano"``, ``"lite"``).
        :param dataset: Dataset the weights were trained on.
        :param cache_dir: Directory to cache downloaded weights. If
            ``None``, uses the default Keras cache directory.
        :return: Local path to the downloaded weights file.
        :raises ValueError: If no URL is registered for the combination.
        """
        pretrained_weights: Dict[str, Dict[str, str]] = {
            "nano": {
                "cifar100": "https://example.com/cliffordnet_nano_cifar100.keras",
            },
            "lite": {
                "cifar100": "https://example.com/cliffordnet_lite_cifar100.keras",
            },
            "lite_g": {
                "cifar100": "https://example.com/cliffordnet_lite_g_cifar100.keras",
            },
        }

        if variant not in pretrained_weights:
            raise ValueError(
                f"No pretrained weights for variant '{variant}'. "
                f"Available: {list(pretrained_weights.keys())}"
            )
        if dataset not in pretrained_weights[variant]:
            raise ValueError(
                f"No pretrained weights for dataset '{dataset}' under "
                f"variant '{variant}'. "
                f"Available: {list(pretrained_weights[variant].keys())}"
            )

        url = pretrained_weights[variant][dataset]
        logger.info(f"Downloading CliffordNet-{variant} weights ({dataset})...")

        weights_path = keras.utils.get_file(
            fname=f"cliffordnet_{variant}_{dataset}.keras",
            origin=url,
            cache_dir=cache_dir,
            cache_subdir="models/clifford_net",
        )
        logger.info(f"Weights downloaded to: {weights_path}")
        return weights_path

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

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

        :param config: Dictionary produced by :meth:`get_config`.
        :return: New :class:`CliffordNet` instance.
        """
        # Deserialize regularizers if they were serialized as dicts.
        for key in ("kernel_regularizer", "bias_regularizer"):
            if config.get(key) and isinstance(config[key], dict):
                config[key] = regularizers.deserialize(config[key])
        return cls(**config)

    # ------------------------------------------------------------------
    # Summary override
    # ------------------------------------------------------------------

    def summary(self, **kwargs: Any) -> None:
        """Print model summary with additional architecture information."""
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

    # ------------------------------------------------------------------
    # Factory class methods
    # ------------------------------------------------------------------

    # Pre-defined variant configurations
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
        :param pretrained: If ``True``, downloads pretrained weights. If a
            string, treats it as a local path to a ``.keras`` weights file.
        :param weights_dataset: Dataset for pretrained weights.
        :param cache_dir: Directory to cache downloaded weights.
        :param kwargs: Override any default hyperparameter.
        :return: Configured :class:`CliffordNet` instance.
        :raises ValueError: If ``variant`` is not recognised.

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
                except Exception as exc:
                    logger.warning(
                        f"Failed to download pretrained weights: {exc}. "
                        "Continuing with random initialisation."
                    )
                    load_weights_path = None

            # If num_classes differs from CIFAR-100 (100), skip classifier.
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
                    skip_mismatch=skip_mismatch,
                    by_name=True,
                )
            except Exception as exc:
                logger.error(f"Failed to load pretrained weights: {exc}")
                raise

        return model

    # ------------------------------------------------------------------
    # Convenience wrappers (delegate to from_variant)
    # ------------------------------------------------------------------

    @classmethod
    def nano(cls, num_classes: int, **kwargs: Any) -> "CliffordNet":
        """CliffordNet-Nano: ~1.4 M params.

        ``channels=128``, ``depth=12``, ``shifts=[1, 2]``, differential
        context, no global branch.

        :param num_classes: Number of output classes.
        :param kwargs: Override any default hyperparameter.
        :return: Configured :class:`CliffordNet` instance.
        """
        return cls.from_variant("nano", num_classes=num_classes, **kwargs)

    @classmethod
    def lite(cls, num_classes: int, **kwargs: Any) -> "CliffordNet":
        """CliffordNet-Lite: ~2.6 M params.

        ``channels=128``, ``depth=12``, ``shifts=[1, 2, 4, 8, 16]``,
        differential context, no global branch.

        :param num_classes: Number of output classes.
        :param kwargs: Override any default hyperparameter.
        :return: Configured :class:`CliffordNet` instance.
        """
        return cls.from_variant("lite", num_classes=num_classes, **kwargs)

    @classmethod
    def lite_g(cls, num_classes: int, **kwargs: Any) -> "CliffordNet":
        """CliffordNet-Lite + gFFN-G: ~3.4 M params.

        Adds the global-average-pool context branch for ~+0.5% accuracy.

        :param num_classes: Number of output classes.
        :param kwargs: Override any default hyperparameter.
        :return: Configured :class:`CliffordNet` instance.
        """
        return cls.from_variant("lite_g", num_classes=num_classes, **kwargs)