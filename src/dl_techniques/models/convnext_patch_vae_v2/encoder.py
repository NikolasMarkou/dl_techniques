"""ConvNeXtPatchEncoderV2 — V1 encoder extended for V2 multi-task use.

Two differences from V1:

1. ``call`` returns a 3-tuple ``(mu, log_var, pre_bottleneck)`` when
   ``output_pre_bottleneck=True`` is passed. The pre-bottleneck tensor
   is the ``embed_dim``-wide ConvNeXt feature map AFTER the block stack,
   BEFORE the 1×1 ``mu_head`` / ``log_var_head`` projections. This is
   the tap point for V2's classification + segmentation heads, and for
   future distillation heads.

2. When ``mae_mask_ratio > 0`` and ``training=True``, a learnable
   ``mask_token`` replaces a per-sample random subset of post-stem
   feature positions. The mask is returned as the optional 4th element
   of the output tuple so the model can use it to weight the
   reconstruction loss in pixel space.

V1's invariants are preserved: resolution-agnostic (no GAP, no learned
absolute PE), zero-init ``log_var_head``, parallel ``mu``/``log_var``
1×1 heads.

DECISION plan_2026-05-27_4a444b14/D-002: SimMIM-style mask application
happens post-stem-norm, before the block stack, on the full
``(B, Hp, Wp, E)`` grid.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Tuple

import keras
from keras import ops

from dl_techniques.layers.convnext_v2_block import ConvNextV2Block

from .mae_mask import apply_mask_with_token, generate_patch_mask


@keras.saving.register_keras_serializable(package="dl_techniques")
class ConvNeXtPatchEncoderV2(keras.layers.Layer):
    """V2 encoder: flat ConvNeXt stack + optional MAE mask + pre-bottleneck tap.

    Args:
        patch_size: Stem stride / kernel.
        embed_dim: Internal ConvNeXt block width.
        depth: Number of ``ConvNextV2Block`` layers after stem.
        kernel_size: Depthwise kernel size inside each block.
        latent_dim: Per-patch latent width.
        dropout_rate: Per-block dropout.
        spatial_dropout_rate: Per-block spatial dropout.
        kernel_regularizer: Optional regularizer (deep-copied per block).
        mae_mask_ratio: SimMIM-style mask ratio applied post-stem-norm.
            ``0.0`` disables masking entirely (V1-equivalent path);
            in that case no ``mask_token`` weight is created.
        mae_mask_seed: Optional reproducibility seed for the per-sample
            mask generator.

    Input shape:
        ``(B, H, W, C)`` with ``H % patch_size == 0`` and
        ``W % patch_size == 0``.

    Output shape:
        When ``output_pre_bottleneck=False`` (default):
            ``(mu, log_var)`` each ``(B, Hp, Wp, latent_dim)``.
        When ``output_pre_bottleneck=True``:
            ``(mu, log_var, pre_bottleneck, mask)`` —
            ``pre_bottleneck`` is ``(B, Hp, Wp, embed_dim)`` and
            ``mask`` is ``(B, Hp, Wp, 1)`` or ``None``.
    """

    def __init__(
        self,
        patch_size: int,
        embed_dim: int,
        depth: int,
        kernel_size: int,
        latent_dim: int,
        dropout_rate: float = 0.0,
        spatial_dropout_rate: float = 0.0,
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        mae_mask_ratio: float = 0.0,
        mae_mask_seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        if kernel_size <= 0:
            raise ValueError(
                f"kernel_size must be positive, got {kernel_size}"
            )
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be >= 1, got {latent_dim}")
        if not (0.0 <= mae_mask_ratio <= 0.95):
            raise ValueError(
                f"mae_mask_ratio must be in [0.0, 0.95], got {mae_mask_ratio}"
            )

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.kernel_size = kernel_size
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        self.spatial_dropout_rate = spatial_dropout_rate
        self.kernel_regularizer = kernel_regularizer
        self.mae_mask_ratio = mae_mask_ratio
        self.mae_mask_seed = mae_mask_seed

        self.stem = keras.layers.Conv2D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
            kernel_regularizer=copy.deepcopy(kernel_regularizer),
            name="stem",
        )
        self.stem_norm = keras.layers.LayerNormalization(
            epsilon=1e-6, name="stem_norm"
        )
        self.blocks = [
            ConvNextV2Block(
                kernel_size=kernel_size,
                filters=embed_dim,
                kernel_regularizer=copy.deepcopy(kernel_regularizer),
                dropout_rate=dropout_rate,
                spatial_dropout_rate=spatial_dropout_rate,
                name=f"block_{i}",
            )
            for i in range(depth)
        ]
        # Parallel mu / log_var 1x1 heads; log_var zero-init (V1 D-003
        # convention from plan_2026-05-25_a8325e3f).
        self.mu_head = keras.layers.Conv2D(
            filters=latent_dim,
            kernel_size=1,
            padding="valid",
            kernel_regularizer=copy.deepcopy(kernel_regularizer),
            name="mu_head",
        )
        self.log_var_head = keras.layers.Conv2D(
            filters=latent_dim,
            kernel_size=1,
            padding="valid",
            kernel_initializer="zeros",
            bias_initializer="zeros",
            kernel_regularizer=copy.deepcopy(kernel_regularizer),
            name="log_var_head",
        )

        # Lazily-built — only when mae_mask_ratio > 0. Created in build().
        self.mask_token: Optional[keras.Variable] = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        if len(input_shape) != 4:
            raise ValueError(
                f"ConvNeXtPatchEncoderV2 expects 4D input (B, H, W, C), got "
                f"{input_shape}"
            )
        B, H, W, _ = input_shape
        if H is not None and H % self.patch_size != 0:
            raise ValueError(
                f"input H ({H}) must be divisible by patch_size "
                f"({self.patch_size})."
            )
        if W is not None and W % self.patch_size != 0:
            raise ValueError(
                f"input W ({W}) must be divisible by patch_size "
                f"({self.patch_size})."
            )

        self.stem.build(input_shape)
        Hp = None if H is None else H // self.patch_size
        Wp = None if W is None else W // self.patch_size
        post_stem_shape = (B, Hp, Wp, self.embed_dim)
        self.stem_norm.build(post_stem_shape)
        for blk in self.blocks:
            blk.build(post_stem_shape)
        self.mu_head.build(post_stem_shape)
        self.log_var_head.build(post_stem_shape)

        if self.mae_mask_ratio > 0.0:
            self.mask_token = self.add_weight(
                shape=(1, 1, 1, self.embed_dim),
                initializer="zeros",
                trainable=True,
                name="mask_token",
            )

        super().build(input_shape)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
        output_pre_bottleneck: bool = False,
    ):
        """Forward pass.

        Args:
            inputs: ``(B, H, W, C)``.
            training: Standard Keras training flag.
            output_pre_bottleneck: If True, return a 4-tuple including the
                pre-bottleneck feature map and the MAE mask (or None).
                If False, return the 2-tuple ``(mu, log_var)`` (V1
                signature).

        Returns:
            Either ``(mu, log_var)`` or
            ``(mu, log_var, pre_bottleneck, mask)``.
        """
        x = self.stem(inputs)
        x = self.stem_norm(x, training=training)

        mask: Optional[keras.KerasTensor] = None
        if self.mae_mask_ratio > 0.0 and training is True and self.mask_token is not None:
            shape = ops.shape(x)
            batch_size = shape[0]
            hp = shape[1]
            wp = shape[2]
            mask = generate_patch_mask(
                batch_size=batch_size,
                hp=hp,
                wp=wp,
                ratio=self.mae_mask_ratio,
                seed=self.mae_mask_seed,
            )
            x = apply_mask_with_token(x, mask, self.mask_token)

        for blk in self.blocks:
            residual = x
            x = blk(x, training=training)
            x = residual + x

        # Tap point for V2 heads.
        pre_bottleneck = x

        mu = self.mu_head(x)
        log_var = self.log_var_head(x)

        if output_pre_bottleneck:
            return mu, log_var, pre_bottleneck, mask
        return mu, log_var

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]:
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input shape, got {input_shape}")
        B, H, W, _ = input_shape
        Hp = None if H is None else H // self.patch_size
        Wp = None if W is None else W // self.patch_size
        latent_shape = (B, Hp, Wp, self.latent_dim)
        return latent_shape, latent_shape

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "embed_dim": self.embed_dim,
                "depth": self.depth,
                "kernel_size": self.kernel_size,
                "latent_dim": self.latent_dim,
                "dropout_rate": self.dropout_rate,
                "spatial_dropout_rate": self.spatial_dropout_rate,
                "kernel_regularizer": keras.regularizers.serialize(
                    self.kernel_regularizer
                ),
                "mae_mask_ratio": self.mae_mask_ratio,
                "mae_mask_seed": self.mae_mask_seed,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ConvNeXtPatchEncoderV2":
        config = dict(config)
        reg = config.get("kernel_regularizer")
        if isinstance(reg, dict):
            config["kernel_regularizer"] = keras.regularizers.deserialize(reg)
        return cls(**config)
