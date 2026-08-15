"""
ConvNeXt V1: a pure convolutional network modernized toward transformer design.

The architecture answers a question left open by the Vision Transformer's
success: how much of that success is attention, and how much is everything else
the transformer papers changed at the same time — the patchified stem, the large
receptive field per operation, the inverted bottleneck, the sparse placement of
normalization and activation, LayerNorm in place of BatchNorm, and separate
downsampling stages. Applying those changes one at a time to a ResNet-50, with
no attention anywhere, recovers the accuracy of a Swin Transformer at matched
FLOPs. The conclusion is that the convolution was never the limitation; the
surrounding design was.

Each block is a depthwise `KxK` convolution (default 7, the "large kernel"
standing in for attention's wide receptive field) followed by an inverted
bottleneck MLP: a 1x1 expansion to `4F` channels, one GELU, and a 1x1 reduction
back to `F`. There is a single activation and a single normalization per block
rather than one of each per convolution — the sparsity is deliberate, and adding
them back costs accuracy. A learnable per-channel `gamma` (layer scale) closes
the block, initialized small so a fresh block starts near a no-op and the
network begins life close to its identity path.

`ConvNextV1Block` is a TRANSFORM-ONLY block: it returns `F(x)`, not `x + F(x)`,
and applies no drop-path. The residual and the stochastic-depth wiring belong to
the caller, and this model owns them in `call` — `residual = x`, block, optional
drop-path, `add([residual, x])`. Wiring it as `x = block(x)` instead silently
removes the residual, which does not raise, does not change any shape, and
annihilates the signal by roughly a factor of 1e-5 per block. The `gamma` floor
of `GAMMA_MIN_VALUE = 1e-6` exists so a residual path can never be scaled to
exactly zero.

The drop-path ramp is GLOBAL across the whole network, not per stage: the rate
for a block is indexed by `block_start_idx + block_idx` over `sum(depths)`
blocks, so stage 0 starts at 0.0 and the last block of the last stage reaches
`drop_path_rate`. Computing it per stage instead would reset the schedule four
times and leave every stage's first block unregularized.

Downsampling is a separate LayerNorm + strided convolution between stages rather
than a stride inside a residual block, and the stem is the same operation applied
to the image (a `strides x strides` patchify, default 4). Those convolutions use
`padding="same"` rather than `"valid"`: at kernel == stride the two are identical
whenever the spatial dimension divides the stride, but `"valid"` collapses to a
0x0 feature map on the small inputs the CIFAR-scale variants use, which produced
non-finite output.

`stochastic_mode` selects what the per-block regularizer actually does: `depth`
is standard stochastic depth (drops the whole branch at training time), while
`gradient` is forward-identity and only perturbs the backward pass. `depth` is
the behaviour-preserving default.

No pretrained ConvNeXt V1 weights are distributed with this package.
`pretrained=True` raises `NotImplementedError` rather than warning and returning
a randomly initialized model, because the previous behaviour made an unavailable
download indistinguishable from a successful one. Local checkpoints load by path.

References:
    - Liu et al., 2022. A ConvNet for the 2020s. (https://arxiv.org/abs/2201.03545)
    - Liu et al., 2021. Swin Transformer: Hierarchical Vision Transformer using
      Shifted Windows. (https://arxiv.org/abs/2103.14030)
    - Huang et al., 2016. Deep Networks with Stochastic Depth.
      (https://arxiv.org/abs/1603.09382)
    - Touvron et al., 2021. Going deeper with Image Transformers. (LayerScale)
      (https://arxiv.org/abs/2103.17239)
"""

import os
import keras
from typing import List, Optional, Union, Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.weight_transfer import load_weights_from_checkpoint
from dl_techniques.utils.drop_path import linear_drop_path_rates
from dl_techniques.layers.convnext_v1_block import ConvNextV1Block
from dl_techniques.layers.stochastic_depth import StochasticDepth
from dl_techniques.layers.stochastic_gradient import StochasticGradient


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class ConvNeXtV1(keras.Model):
    """ConvNeXt V1 model implementation with pretrained support

    A modern ConvNet architecture that achieves competitive performance
    with Vision Transformers while maintaining the simplicity and efficiency
    of convolutional networks. This version adapts to different input sizes
    and supports loading pretrained weights.

    Args:
        num_classes: Integer, number of output classes for classification.
            Only used if include_top=True.
        depths: List of integers, number of ConvNext blocks in each stage.
            Default is [3, 3, 9, 3] for ConvNeXt-Tiny.
        dims: List of integers, number of channels in each stage.
            Default is [96, 192, 384, 768] for ConvNeXt-Tiny.
        drop_path_rate: Float, stochastic depth rate. Linearly increases
            from 0 to this value across all blocks.
        kernel_size: Integer or tuple, kernel size for ConvNext blocks.
            Default is 7 following the original paper.
        activation: String or callable, activation function for blocks.
            Default is "gelu" as used in the original paper.
        use_bias: Boolean, whether to use bias in convolutions.
        kernel_regularizer: Regularizer function applied to kernels.
        dropout_rate: Float, dropout rate applied within blocks.
        spatial_dropout_rate: Float, spatial dropout rate for blocks.
        strides: int, Strides for downsampling.
        use_gamma: Boolean, whether to use learnable scaling in blocks.
        use_softorthonormal_regularizer: Boolean, whether to use soft
            orthonormal regularization in blocks.
        include_top: Boolean, whether to include the classification head.
        input_shape: Tuple, input shape. ``None`` resolves to ``(None, None, 3)``
            -- the model is fully convolutional and global-pools before the head,
            so a concrete spatial size is optional. It is required only where a
            downstream consumer needs static spatial dims; a checkpoint load with
            unspecified spatial dims materializes weights at
            ``PRETRAINED_BUILD_SPATIAL`` (224).
        **kwargs: Additional keyword arguments for the Model base class.

    Raises:
        ValueError: If depths and dims have different lengths.
        ValueError: If invalid model configuration is provided.

    Example:
        >>> # Create ConvNeXt-Tiny model for CIFAR-10
        >>> model = ConvNeXtV1.from_variant("tiny", num_classes=10, input_shape=(32, 32, 3))
        >>>
        >>> # Load as feature extractor
        >>> model = ConvNeXtV1.from_variant("base", include_top=False)
    """

    MODEL_VARIANTS = {
        "cifar10": {"depths": [5, 5], "dims": [96, 192]},
        "tiny": {"depths": [3, 3, 9, 3], "dims": [96, 192, 384, 768]},
        "small": {"depths": [3, 3, 27, 3], "dims": [96, 192, 384, 768]},
        "base": {"depths": [3, 3, 27, 3], "dims": [128, 256, 512, 1024]},
        "large": {"depths": [3, 3, 27, 3], "dims": [192, 384, 768, 1536]},
        "xlarge": {"depths": [3, 3, 27, 3], "dims": [256, 512, 1024, 2048]},
    }

    # Architecture constants
    LAYERNORM_EPSILON = 1e-6
    STEM_INITIALIZER = "truncated_normal"
    HEAD_INITIALIZER = "truncated_normal"

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
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            dropout_rate: float = 0.0,
            spatial_dropout_rate: float = 0.0,
            strides: int = 4,
            use_gamma: bool = True,
            use_softorthonormal_regularizer: bool = False,
            include_top: bool = True,
            input_shape: Tuple[int, ...] = (None, None, 3),
            **kwargs
    ):
        super().__init__(**kwargs)

        depths = list(depths) if depths is not None else [3, 3, 9, 3]
        dims = list(dims) if dims is not None else [96, 192, 384, 768]

        if len(depths) != len(dims):
            raise ValueError(
                f"Length of depths ({len(depths)}) must equal length of dims ({len(dims)})"
            )

        if len(depths) != 4:
            logger.warning(
                f"ConvNeXt typically uses 4 stages, got {len(depths)} stages"
            )

        if strides <= 0:
            raise ValueError(
                f"Strides {strides} must be positive."
            )

        if stochastic_mode not in ('depth', 'gradient'):
            raise ValueError(
                f"stochastic_mode must be 'depth' or 'gradient', got {stochastic_mode!r}"
            )

        if input_shape is None:
            input_shape = (None, None, 3)
        if len(input_shape) != 3:
            raise ValueError(f"input_shape must be 3D, got {input_shape}")

        self.num_classes = num_classes
        self.depths = depths
        self.dims = dims
        self.drop_path_rate = drop_path_rate
        self.stochastic_mode = stochastic_mode
        self.kernel_size = kernel_size
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_regularizer = kernel_regularizer
        self.dropout_rate = dropout_rate
        self.spatial_dropout_rate = spatial_dropout_rate
        self.use_gamma = use_gamma
        self.use_softorthonormal_regularizer = use_softorthonormal_regularizer
        self.include_top = include_top
        self.strides = strides
        self.input_shape = input_shape

        self.input_height, self.input_width, self.input_channels = input_shape
        if self.input_channels not in [1, 3]:
            logger.warning(
                f"Unusual number of channels: {self.input_channels}. ConvNeXt typically uses 3 channels")

        self._build_stem()

        self.downsample_layers_list = []
        self.stages_list = []
        for i in range(len(self.depths)):
            if i > 0:
                self._build_downsample_layer(i)
            self._build_stage(i)

        if self.include_top:
            self._build_head()

        logger.info(
            f"Created ConvNeXt V1 model for input {input_shape} "
            f"with {sum(depths)} blocks"
        )

    def _build_stem(self):
        """Build and assign stem layers."""
        stem_kernel_size = self.strides
        stem_stride = self.strides
        self.stem_conv = keras.layers.Conv2D(
            filters=self.dims[0],
            kernel_size=stem_kernel_size,
            strides=stem_stride,
            padding="same" if stem_stride == 1 else "valid",
            use_bias=self.use_bias,
            kernel_initializer=self.STEM_INITIALIZER,
            kernel_regularizer=self.kernel_regularizer,
            name="stem_conv"
        )
        self.stem_norm = keras.layers.LayerNormalization(
            epsilon=self.LAYERNORM_EPSILON,
            center=self.use_bias,
            scale=True,
            name="stem_norm"
        )

    def _build_downsample_layer(self, stage_idx: int):
        """Build and assign a downsample layer."""
        downsample_kernel_size, downsample_stride = self.strides, self.strides
        downsample_norm = keras.layers.LayerNormalization(
            epsilon=self.LAYERNORM_EPSILON,
            center=self.use_bias,
            scale=True,
            name=f"downsample_norm_{stage_idx - 1}"
        )
        downsample_conv = keras.layers.Conv2D(
            filters=self.dims[stage_idx],
            kernel_size=downsample_kernel_size,
            strides=downsample_stride,
            # DECISION plan_2026-06-15_e6a0391c/D-003: "same" (not "valid") so the
            # kernel==stride downsample never collapses to 0x0 on small inputs
            # (CIFAR 32x32 / 16x16 — sizes the convnext tests exercise), which
            # previously yielded non-finite (NaN) output. Identical to "valid"
            # when the spatial dim is divisible by the stride.
            padding="same",
            use_bias=self.use_bias,
            kernel_initializer=self.STEM_INITIALIZER,
            kernel_regularizer=self.kernel_regularizer,
            name=f"downsample_conv_{stage_idx - 1}"
        )
        self.downsample_layers_list.append([downsample_norm, downsample_conv])

    def _build_stage(self, stage_idx: int):
        """Build and assign a stage of ConvNeXt blocks."""
        stage_blocks = []
        depth = self.depths[stage_idx]
        dim = self.dims[stage_idx]
        total_blocks = sum(self.depths)
        block_start_idx = sum(self.depths[:stage_idx])
        # The drop-path ramp is GLOBAL across stages, not per-stage: the index is
        # `block_start_idx + block_idx` over `sum(self.depths)` blocks, so stage 0
        # starts at 0.0 and the last block of the last stage reaches drop_path_rate.
        # `linear_drop_path_rates` already handles total_blocks <= 1 (all-zero).
        drop_path_rates = linear_drop_path_rates(total_blocks, self.drop_path_rate)

        for block_idx in range(depth):
            current_block_idx = block_start_idx + block_idx
            drop_rate = drop_path_rates[current_block_idx]

            block = ConvNextV1Block(
                kernel_size=self.kernel_size,
                filters=dim,
                activation=self.activation,
                kernel_regularizer=self.kernel_regularizer,
                use_bias=self.use_bias,
                dropout_rate=self.dropout_rate,
                spatial_dropout_rate=self.spatial_dropout_rate,
                use_gamma=self.use_gamma,
                use_softorthonormal_regularizer=self.use_softorthonormal_regularizer,
                name=f"stage_{stage_idx}_block_{block_idx}"
            )
            # DECISION plan_2026-06-03_943569ad/D-001
            # Use the repo's dedicated StochasticDepth/StochasticGradient layers
            # (gated by self.stochastic_mode), NOT a hand-rolled
            # keras.layers.Dropout(noise_shape=(None,1,1,1)). Do NOT pass rate= or
            # noise_shape=: both layers take drop_path_rate= and compute per-sample
            # noise internally. 'depth' is the behavior-preserving default; 'gradient'
            # is an opt-in forward-identity grad-only regularizer. See decisions.md D-001.
            drop_path_cls = StochasticDepth if self.stochastic_mode == 'depth' else StochasticGradient
            drop_path = drop_path_cls(
                drop_path_rate=drop_rate,
                name=f"stage_{stage_idx}_block_{block_idx}_drop_path"
            ) if drop_rate > 0 else None
            stage_blocks.append({"block": block, "drop_path": drop_path})
        self.stages_list.append(stage_blocks)

    def _build_head(self):
        """Build and assign head layers."""
        self.gap = keras.layers.GlobalAveragePooling2D(name="global_avg_pool")
        self.head_norm = keras.layers.LayerNormalization(
            epsilon=self.LAYERNORM_EPSILON,
            center=self.use_bias,
            scale=True,
            name="head_norm"
        )
        if self.num_classes > 0:
            self.classifier = keras.layers.Dense(
                units=self.num_classes,
                use_bias=self.use_bias,
                kernel_initializer=self.HEAD_INITIALIZER,
                kernel_regularizer=self.kernel_regularizer,
                name="classifier"
            )
        else:
            self.classifier = None


    def build(self, input_shape):
        """Builds the model and its layers."""

        # The summary() method might call build with a 3D shape (without batch dim).
        # We add a dummy batch dimension if that's the case to ensure layers build correctly.
        if len(input_shape) == 3:
            build_shape = (None,) + tuple(input_shape)
        else:
            build_shape = input_shape
        # A dummy forward pass with a KerasTensor will correctly build all sub-layers.
        dummy_input = keras.KerasTensor(build_shape)
        _ = self.call(dummy_input)

        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Defines the forward pass of the model.

        Args:
            inputs: Input tensor of shape (batch_size, height, width, channels).
            training: Boolean or None, whether the model is in training mode.

        Returns:
            Output tensor. Shape depends on include_top:
                - If include_top=True: (batch_size, num_classes)
                - If include_top=False: (batch_size, H', W', channels)
        """
        x = self.stem_conv(inputs)
        x = self.stem_norm(x)

        for stage_idx, stage_blocks in enumerate(self.stages_list):
            if stage_idx > 0:
                norm_layer, conv_layer = self.downsample_layers_list[stage_idx - 1]
                x = norm_layer(x)
                x = conv_layer(x)

            for block_info in stage_blocks:
                residual = x
                x = block_info["block"](x, training=training)
                if block_info["drop_path"]:
                    x = block_info["drop_path"](x, training=training)
                x = keras.layers.add([residual, x])

        if self.include_top:
            x = self.gap(x)
            x = self.head_norm(x)
            if self.classifier:
                x = self.classifier(x)

        return x

    # ---------------------------------------------------------------------

    # Spatial size used to materialize weights before a checkpoint load when the
    # model was constructed with the default, fully-shape-agnostic
    # `input_shape=(None, None, 3)`. 224 is the ImageNet size these checkpoints
    # are trained at, and it is the size the factory docstrings already promise.
    PRETRAINED_BUILD_SPATIAL = 224

    def _pretrained_build_shape(self) -> Tuple[int, ...]:
        """Concrete `(H, W, C)` for the pre-load dummy forward.

        DECISION plan-2026-08-14T233721-d4f9beb2/D-067: resolve the None spatial
        dims instead of passing `self.input_shape` through. Do NOT go back to
        `keras.random.normal((1,) + tuple(self.input_shape))`: the DEFAULT
        `input_shape` is `(None, None, 3)`, so that built `(1, None, None, 3)`
        and made the factories' own documented `pretrained=<local path>` call
        fail for every caller who did not also pass a concrete `input_shape`.
        The channel count is never defaulted -- a `None` there is a real
        configuration error and is raised as one. See decisions.md D-067.
        """
        height, width, channels = self.input_shape
        if channels is None:
            raise ValueError(
                "Cannot materialize weights for a checkpoint load: input_shape "
                f"{self.input_shape} has no channel count. Pass a concrete "
                "input_shape=(height, width, channels) to the constructor."
            )
        height = self.PRETRAINED_BUILD_SPATIAL if height is None else height
        width = self.PRETRAINED_BUILD_SPATIAL if width is None else width
        return (int(height), int(width), int(channels))

    # ---------------------------------------------------------------------

    def load_pretrained_weights(
            self,
            weights_path: str,
            skip_mismatch: bool = True
    ) -> None:
        """Load pretrained weights into the model.

        This method handles loading weights with smart mismatch handling,
        particularly useful when the number of classes differs or when
        loading weights without the top classifier.

        Args:
            weights_path: String, path to the weights file (.keras format).
            skip_mismatch: Boolean, whether to skip layers with mismatched shapes.
                Useful when loading weights with different num_classes.

        Raises:
            FileNotFoundError: If weights_path doesn't exist.
            ValueError: If weights cannot be loaded.

        Example:
            >>> model = ConvNeXtV1.from_variant("tiny", num_classes=10)
            >>> model.load_pretrained_weights("convnext_tiny_imagenet.keras", skip_mismatch=True)
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        try:
            if not self.built:
                dummy_input = keras.random.normal(
                    (1,) + self._pretrained_build_shape()
                )
                self(dummy_input, training=False)

            logger.info(f"Loading pretrained weights from {weights_path}")

            # Keras 3 removed `by_name` from `Model.load_weights` — the
            # signature is `(filepath, skip_mismatch=False, **kwargs)` and it
            # REJECTS the unknown keyword, so this call raised
            # `ValueError: Invalid keyword arguments: {'by_name': True}` for
            # every caller. It went unnoticed because the only route here was
            # `pretrained=<path>` and the enclosing except turned the failure
            # into a warning that continued with random weights.
            report = load_weights_from_checkpoint(
                target=self,
                ckpt_path=weights_path,
                skip_prefixes=(),
                strict=not skip_mismatch,
            )
            logger.info(f"Weight transfer complete: {report}")

            if skip_mismatch:
                logger.info(
                    "Weights loaded with skip_mismatch=True. "
                    "Layers with shape mismatches were skipped (e.g., classifier layer)."
                )
            else:
                logger.info("All weights loaded successfully.")

        except Exception as e:
            # `from e` so the real cause survives; this wrapper used to be the
            # only thing a caller saw.
            raise ValueError(
                f"Failed to load weights from {weights_path}: "
                f"{type(e).__name__}: {e}"
            ) from e

    # `_download_weights` raises instead of falling back to random init. The
    # previous version held a `PRETRAINED_WEIGHTS` table of placeholder URLs on a
    # non-existent host; `from_variant` caught the download failure, logged a
    # warning and returned a randomly-initialized model, so `pretrained=True`
    # silently produced untrained weights. Do NOT reinstate a warn-and-return
    # branch here or in `from_variant`.
    @staticmethod
    def _download_weights(
            variant: str,
            dataset: str = "imagenet",
            cache_dir: Optional[str] = None
    ) -> str:
        """Resolve a download path for pretrained weights of ``variant``.

        Not implemented: no public ConvNeXt V1 weights ship with
        ``dl_techniques``. Always raises. Kept to mirror the house factory
        recipe (see ``models/resnet/model.py``) and to give an explicit failure
        mode instead of a silent random-init fallback.

        Args:
            variant: Variant name (unused).
            dataset: Dataset name (unused).
            cache_dir: Cache directory (unused).

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            f"No pretrained ConvNeXt V1 weights are distributed with dl_techniques "
            f"(requested variant '{variant}', dataset '{dataset}'). Pass a local "
            f"checkpoint instead: ConvNeXtV1.from_variant('{variant}', "
            f"pretrained='/path/to/weights.keras')."
        )

    @classmethod
    def from_variant(
            cls,
            variant: str,
            num_classes: int = 1000,
            input_shape: Optional[Tuple[int, ...]] = None,
            pretrained: Union[bool, str] = False,
            weights_dataset: str = "imagenet",
            weights_input_shape: Optional[Tuple[int, ...]] = None,
            cache_dir: Optional[str] = None,
            **kwargs
    ) -> "ConvNeXtV1":
        """Create a ConvNeXt model from a predefined variant.

        Args:
            variant: String, one of "tiny", "small", "base", "large", "xlarge"
            num_classes: Integer, number of output classes
            input_shape: Tuple, input shape. None resolves to (None, None, 3);
                a pretrained load then materializes weights at 224x224.
            pretrained: Boolean or string. If True, loads pretrained weights from
                default URL. If string, treats it as a path to local weights file.
            weights_dataset: String, dataset for pretrained weights.
                Options: "imagenet", "imagenet22k".
            weights_input_shape: Tuple, input shape used during weight pretraining.
                Only needed if loading pretrained weights with different input_shape.
                Defaults to (224, 224, 3) for ImageNet weights.
            cache_dir: Optional string, directory to cache downloaded weights.
            **kwargs: Additional arguments passed to the constructor

        Returns:
            ConvNeXtV1 model instance

        Raises:
            ValueError: If variant is not recognized
            NotImplementedError: If pretrained is True
            NotImplementedError: If pretrained is True

        Example:
            >>> # Feature extractor for fine-tuning
            >>> model = ConvNeXtV1.from_variant("base", include_top=False)
            >>>
            >>> # Load from local weights file
            >>> model = ConvNeXtV1.from_variant("large", pretrained="path/to/weights.keras")
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant]

        logger.info(f"Creating ConvNeXt-{variant.upper()} model with input_shape {input_shape}")

        load_weights_path = None
        skip_mismatch = False

        if pretrained:
            if isinstance(pretrained, str):
                load_weights_path = pretrained
                logger.info(f"Will load weights from local file: {load_weights_path}")
            else:
                load_weights_path = cls._download_weights(
                    variant=variant,
                    dataset=weights_dataset,
                    cache_dir=cache_dir
                )

            # A head width or input resolution differing from the checkpoint's
            # means the affected layers must be skipped rather than refused.
            pretrained_classes = 1000 if weights_dataset == "imagenet" else 21841
            if kwargs.get("include_top", True) and num_classes != pretrained_classes:
                skip_mismatch = True
                logger.info(
                    f"num_classes ({num_classes}) differs from pretrained "
                    f"({pretrained_classes}). Will skip classifier weights."
                )

            if weights_input_shape and input_shape and weights_input_shape != input_shape:
                logger.info(
                    f"Loading weights pretrained on {weights_input_shape} "
                    f"for model with input shape {input_shape}. "
                    f"Only backbone weights will be loaded."
                )
                skip_mismatch = True

        model = cls(
            num_classes=num_classes,
            depths=config["depths"],
            dims=config["dims"],
            input_shape=input_shape,
            **kwargs
        )

        if load_weights_path:
            try:
                model.load_pretrained_weights(
                    weights_path=load_weights_path,
                    skip_mismatch=skip_mismatch
                )
            except Exception as e:
                logger.error(f"Failed to load pretrained weights: {str(e)}")
                raise

        return model

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization.

        Returns:
            Configuration dictionary
        """
        config = {
            "num_classes": self.num_classes,
            "depths": self.depths,
            "dims": self.dims,
            "drop_path_rate": self.drop_path_rate,
            "stochastic_mode": self.stochastic_mode,
            "kernel_size": self.kernel_size,
            "activation": self.activation,
            "use_bias": self.use_bias,
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "dropout_rate": self.dropout_rate,
            "spatial_dropout_rate": self.spatial_dropout_rate,
            "use_gamma": self.use_gamma,
            "use_softorthonormal_regularizer": self.use_softorthonormal_regularizer,
            "include_top": self.include_top,
            "input_shape": self.input_shape,
            "strides": self.strides
        }
        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ConvNeXtV1":
        """Create model from configuration.

        Args:
            config: Configuration dictionary

        Returns:
            ConvNeXtV1 model instance
        """
        if config.get("kernel_regularizer"):
            config["kernel_regularizer"] = keras.regularizers.deserialize(
                config["kernel_regularizer"]
            )

        return cls(**config)

    def summary(self, **kwargs):
        """Print model summary with additional information."""
        if not self.built:
            dummy_input = keras.KerasTensor(self.input_shape)
            self.build(dummy_input.shape)

        super().summary(**kwargs)

        total_blocks = sum(self.depths)
        logger.info("ConvNeXt V1 configuration:")
        logger.info(f"  - Input shape: ({self.input_height}, {self.input_width}, {self.input_channels})")
        logger.info(f"  - Stages: {len(self.depths)}")
        logger.info(f"  - Depths: {self.depths}")
        logger.info(f"  - Original dimensions: {self.dims}")
        logger.info(f"  - Total blocks: {total_blocks}")
        logger.info(f"  - Drop path rate: {self.drop_path_rate}")
        logger.info(f"  - Kernel size: {self.kernel_size}")
        logger.info(f"  - Include top: {self.include_top}")
        if self.include_top:
            logger.info(f"  - Number of classes: {self.num_classes}")


# ---------------------------------------------------------------------

def create_convnext_v1(
        variant: str = "tiny",
        num_classes: int = 1000,
        input_shape: Optional[Tuple[int, ...]] = (None, None, 3),
        pretrained: Union[bool, str] = False,
        weights_dataset: str = "imagenet",
        weights_input_shape: Optional[Tuple[int, ...]] = None,
        cache_dir: Optional[str] = None,
        **kwargs
) -> ConvNeXtV1:
    """Convenience function to create ConvNeXt V1 models.

    Args:
        variant: String, model variant ("tiny", "small", "base", "large", "xlarge")
        num_classes: Integer, number of output classes
        input_shape: Tuple, input shape.
        pretrained: Boolean or string. If True, loads pretrained weights from
            default URL. If string, treats it as a path to local weights file.
        weights_dataset: String, dataset for pretrained weights.
            Options: "imagenet", "imagenet22k".
        weights_input_shape: Tuple, input shape used during weight pretraining.
            Only needed if loading pretrained weights with different input_shape.
        cache_dir: Optional string, directory to cache downloaded weights.
        **kwargs: Additional arguments passed to the model constructor

    Returns:
        ConvNeXtV1 model instance

    Example:
        >>> # Create ConvNeXt-Tiny (randomly initialized; no weights ship here)
        >>> model = create_convnext_v1("tiny")
        >>>
        >>> # Create ConvNeXt-Base as feature extractor
        >>> model = create_convnext_v1("base", include_top=False)
        >>>
        >>> # Fine-tune on CIFAR-10 from a local checkpoint
        >>> model = create_convnext_v1(
        ...     "small",
        ...     num_classes=10,
        ...     input_shape=(32, 32, 3),
        ...     pretrained="path/to/weights.keras",
        ...     weights_input_shape=(224, 224, 3)
        ... )
        >>>
        >>> # `pretrained=True` raises NotImplementedError -- pass a local path.
    """
    model = ConvNeXtV1.from_variant(
        variant,
        num_classes=num_classes,
        input_shape=input_shape,
        pretrained=pretrained,
        weights_dataset=weights_dataset,
        weights_input_shape=weights_input_shape,
        cache_dir=cache_dir,
        **kwargs
    )

    return model

# ---------------------------------------------------------------------