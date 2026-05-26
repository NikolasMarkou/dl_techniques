"""Patch-level ConvNeXt encoder for :class:`ConvNeXtPatchVAE`.

A resolution-agnostic flat single-stage encoder:

::

    x : (B, H, W, C)
        │
        ▼  Conv2D(embed_dim, kernel=patch_size, stride=patch_size, "valid")
        │  -- the patchifying stem; turns each P x P pixel patch into one
        │     embed_dim-wide spatial position. After this point H,W refer
        │     to the patch grid (Hp, Wp).
        ▼  LayerNormalization
        ▼  N x [residual + ConvNextV2Block(kernel_size, embed_dim)]
        ▼  Conv2D(latent_dim, kernel=1, Glorot init) "mu_head"     ─┐ parallel
        ▼  Conv2D(latent_dim, kernel=1, zeros init) "log_var_head" ─┘ heads
        ▼  -> (mu, log_var), each (B, Hp, Wp, latent_dim)

Design choices (see ``plans/plan_2026-05-25_fb57d478/findings.md``):

- F3: ``ConvNextV2Block`` is shape-preserving on ``(B, H, W, F)`` with
  stride=1 / padding="same"; stacking N blocks builds depth without
  spatial downsampling.
- F5: NO ``GlobalAveragePooling2D``, NO learned absolute positional
  embedding, NO ``Dense(latent_dim)`` over flattened spatial map. These
  would break the resolution-agnostic invariant.
- The residual connection is applied *externally* — matching the
  ``models/convnext/convnext_v2.py`` idiom (the block itself emits the
  residual delta).
"""

from __future__ import annotations

import copy
import keras
from typing import Any, Dict, Optional, Tuple

# ------------------------------------------------------------------
# local imports
# ------------------------------------------------------------------

from dl_techniques.layers.convnext_v2_block import ConvNextV2Block

# ------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ConvNeXtPatchEncoder(keras.layers.Layer):
    """Flat single-stage ConvNeXt encoder emitting per-patch ``(mu, log_var)``.

    Args:
        patch_size: Stem stride / kernel — converts pixels to patch grid.
        embed_dim: Internal ConvNeXt block width.
        depth: Number of ``ConvNextV2Block`` layers stacked after stem.
        kernel_size: Depthwise kernel inside each ``ConvNextV2Block``.
        latent_dim: Per-patch latent width. The bottleneck emits
            ``2 * latent_dim`` channels which are split into
            ``(mu, log_var)``.
        dropout_rate: Per-block dropout rate (forwarded to
            ``ConvNextV2Block``).
        spatial_dropout_rate: Per-block spatial dropout rate.
        kernel_regularizer: Optional regularizer for the conv kernels
            inside each block (deep-copied per block to avoid weight
            sharing).

    Input shape:
        4D tensor with shape ``(B, H, W, C)``. ``H`` and ``W`` must be
        divisible by ``patch_size``.

    Output shape:
        Tuple ``(mu, log_var)`` where each has shape
        ``(B, H // patch_size, W // patch_size, latent_dim)``.
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

        # Store config
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.kernel_size = kernel_size
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        self.spatial_dropout_rate = spatial_dropout_rate
        self.kernel_regularizer = kernel_regularizer

        # Sub-layers created in __init__ (Keras 3 Golden Rule shape).
        # The patchifying stem: kernel == stride == patch_size, padding="valid".
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
        # DECISION plan_2026-05-25_a8325e3f/D-003: split into two 1x1 heads so
        # log_var_head can be zero-initialized, reducing step-1 KL by ~70%.
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

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Explicitly build each sub-layer in computational order."""
        if len(input_shape) != 4:
            raise ValueError(
                f"ConvNeXtPatchEncoder expects 4D input (B, H, W, C), got "
                f"{input_shape}"
            )
        B, H, W, C = input_shape
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
        # After stem: (B, Hp, Wp, embed_dim).
        Hp = None if H is None else H // self.patch_size
        Wp = None if W is None else W // self.patch_size
        post_stem_shape = (B, Hp, Wp, self.embed_dim)
        self.stem_norm.build(post_stem_shape)
        for blk in self.blocks:
            blk.build(post_stem_shape)
        self.mu_head.build(post_stem_shape)
        self.log_var_head.build(post_stem_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Forward pass.

        Args:
            inputs: ``(B, H, W, C)`` with ``H % patch_size == 0`` and
                ``W % patch_size == 0``.
            training: Standard Keras training flag.

        Returns:
            Tuple ``(mu, log_var)``, both shaped
            ``(B, Hp, Wp, latent_dim)``.
        """
        x = self.stem(inputs)
        x = self.stem_norm(x, training=training)
        for blk in self.blocks:
            residual = x
            x = blk(x, training=training)
            x = residual + x
        mu = self.mu_head(x)
        log_var = self.log_var_head(x)
        return mu, log_var

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]:
        """Return the ``(mu_shape, log_var_shape)`` tuple."""
        if len(input_shape) != 4:
            raise ValueError(
                f"Expected 4D input shape, got {input_shape}"
            )
        B, H, W, _ = input_shape
        Hp = None if H is None else H // self.patch_size
        Wp = None if W is None else W // self.patch_size
        latent_shape = (B, Hp, Wp, self.latent_dim)
        return latent_shape, latent_shape

    def get_config(self) -> Dict[str, Any]:
        """Return constructor kwargs for serialization."""
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
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ConvNeXtPatchEncoder":
        """Reconstruct, deserializing any regularizer."""
        config = dict(config)
        reg = config.get("kernel_regularizer")
        if isinstance(reg, dict):
            config["kernel_regularizer"] = keras.regularizers.deserialize(reg)
        return cls(**config)
