"""Patch-level ConvNeXt decoder for :class:`ConvNeXtPatchVAE`.

Symmetric counterpart to :class:`ConvNeXtPatchEncoder`:

::

    z : (B, Hp, Wp, latent_dim)
        │
        ▼  Conv2D(embed_dim, kernel=1)            "proj_in"
        ▼  N x [residual + ConvNextV2Block(kernel_size, embed_dim)]
        ▼  LayerNormalization                     "pre_head_norm"
        ▼  Conv2DTranspose(img_channels,
        │                  kernel=patch_size,
        │                  stride=patch_size,
        │                  padding="valid")        "head"
        ▼
    x_hat : (B, Hp * patch_size, Wp * patch_size, img_channels)

The head outputs raw logits (no activation). The owning model applies the
appropriate activation depending on ``recon_loss_type`` (sigmoid for BCE,
identity for MSE), keeping the decoder layer reusable across families.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Tuple

import keras

from dl_techniques.layers.convnext_v2_block import ConvNextV2Block


@keras.saving.register_keras_serializable()
class ConvNeXtPatchDecoder(keras.layers.Layer):
    """Flat single-stage ConvNeXt decoder from per-patch latents to pixels.

    Args:
        patch_size: Transposed-conv kernel / stride — projects each
            patch position back to a ``patch_size x patch_size`` pixel
            tile.
        embed_dim: Internal ConvNeXt block width (same as encoder).
        depth: Number of ``ConvNextV2Block`` layers stacked before head.
        kernel_size: Depthwise kernel inside each ``ConvNextV2Block``.
        img_channels: Output channel count (3 for RGB, 1 for MNIST).
        dropout_rate: Per-block dropout rate.
        spatial_dropout_rate: Per-block spatial dropout rate.
        kernel_regularizer: Optional regularizer; deep-copied per block.

    Input shape:
        4D tensor with shape ``(B, Hp, Wp, latent_dim)``.

    Output shape:
        4D tensor with shape
        ``(B, Hp * patch_size, Wp * patch_size, img_channels)``. Raw
        logits — the owning model applies any output activation.
    """

    def __init__(
        self,
        patch_size: int,
        embed_dim: int,
        depth: int,
        kernel_size: int,
        img_channels: int,
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
        if img_channels <= 0:
            raise ValueError(
                f"img_channels must be positive, got {img_channels}"
            )

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.kernel_size = kernel_size
        self.img_channels = img_channels
        self.dropout_rate = dropout_rate
        self.spatial_dropout_rate = spatial_dropout_rate
        self.kernel_regularizer = kernel_regularizer

        self.proj_in = keras.layers.Conv2D(
            filters=embed_dim,
            kernel_size=1,
            padding="valid",
            kernel_regularizer=copy.deepcopy(kernel_regularizer),
            name="proj_in",
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
        self.pre_head_norm = keras.layers.LayerNormalization(
            epsilon=1e-6, name="pre_head_norm"
        )
        self.head = keras.layers.Conv2DTranspose(
            filters=img_channels,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
            activation=None,
            kernel_regularizer=copy.deepcopy(kernel_regularizer),
            name="head",
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        if len(input_shape) != 4:
            raise ValueError(
                f"ConvNeXtPatchDecoder expects 4D input (B, Hp, Wp, latent_dim), "
                f"got {input_shape}"
            )
        B, Hp, Wp, _ = input_shape
        # proj_in turns latent_dim -> embed_dim.
        self.proj_in.build(input_shape)
        block_in_shape = (B, Hp, Wp, self.embed_dim)
        for blk in self.blocks:
            blk.build(block_in_shape)
        self.pre_head_norm.build(block_in_shape)
        self.head.build(block_in_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Forward pass.

        Args:
            inputs: ``(B, Hp, Wp, latent_dim)`` latent grid.
            training: Standard Keras training flag.

        Returns:
            ``(B, Hp * patch_size, Wp * patch_size, img_channels)``.
            Raw logits — the owning model applies any output activation.
        """
        x = self.proj_in(inputs)
        for blk in self.blocks:
            residual = x
            x = blk(x, training=training)
            x = residual + x
        x = self.pre_head_norm(x, training=training)
        x = self.head(x)
        return x

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        if len(input_shape) != 4:
            raise ValueError(
                f"Expected 4D input shape, got {input_shape}"
            )
        B, Hp, Wp, _ = input_shape
        H = None if Hp is None else Hp * self.patch_size
        W = None if Wp is None else Wp * self.patch_size
        return (B, H, W, self.img_channels)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "embed_dim": self.embed_dim,
                "depth": self.depth,
                "kernel_size": self.kernel_size,
                "img_channels": self.img_channels,
                "dropout_rate": self.dropout_rate,
                "spatial_dropout_rate": self.spatial_dropout_rate,
                "kernel_regularizer": keras.regularizers.serialize(
                    self.kernel_regularizer
                ),
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ConvNeXtPatchDecoder":
        config = dict(config)
        reg = config.get("kernel_regularizer")
        if isinstance(reg, dict):
            config["kernel_regularizer"] = keras.regularizers.deserialize(reg)
        return cls(**config)
