"""Task heads for :class:`CliffordNetPatchVAEV2`.

The classification head is identical to the v2 ConvNeXt variant and is
re-exported unchanged. The segmentation head parameterises the
bilinear-upsample factor (``upsample_factor`` arg) so the head can
recover the original image resolution from the bottleneck spatial size
``(Hb, Wb)``, which is smaller than ``(Hp, Wp)`` by ``2**(N-1)`` in the
hierarchical CliffordNet encoder.

DECISION plan_2026-05-27_75849a91/D-003: new ``CliffordSegmentationHead``
instead of editing v2's ``SegmentationHead`` to take an upsample
factor — keeps v2 untouched.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import keras

# Re-export the classification head (works generically with any embed_dim).
from dl_techniques.models.convnext_patch_vae_v2.heads import (  # noqa: F401
    AttentionPoolClassifierHead,
)


@keras.saving.register_keras_serializable(package="dl_techniques")
class CliffordSegmentationHead(keras.layers.Layer):
    """Segmentation head with a configurable bilinear upsample factor.

    .. code-block::

        x : (B, Hb, Wb, E)
          → Conv2D(E, 3, "same") → GELU → Dropout
          → Conv2D(num_classes, 1)
          → bilinear upsample by `upsample_factor`
          → logits (B, H, W, num_classes)

    For the hierarchical CliffordNet encoder,
    ``upsample_factor = patch_size * 2**(num_stages - 1)``.

    Args:
        embed_dim: Feature dim at input (the encoder's bottleneck width
            == ``stage_dims[-1]``).
        num_classes: Number of seg classes (including background).
        upsample_factor: Bilinear upsample factor to recover pixel grid.
        dropout_rate: Dropout between the two convs.
        name: Layer name.
        **kwargs: Forwarded to ``keras.layers.Layer``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        upsample_factor: int,
        dropout_rate: float = 0.0,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        if upsample_factor <= 0:
            raise ValueError(
                f"upsample_factor must be positive, got {upsample_factor}"
            )
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(
                f"dropout_rate must be in [0.0, 1.0], got {dropout_rate}"
            )

        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.upsample_factor = upsample_factor
        self.dropout_rate = dropout_rate

        self.conv1 = keras.layers.Conv2D(
            filters=embed_dim,
            kernel_size=3,
            padding="same",
            activation="gelu",
            name="seg_conv1",
        )
        self.drop = keras.layers.Dropout(dropout_rate, name="seg_drop")
        self.conv2 = keras.layers.Conv2D(
            filters=num_classes,
            kernel_size=1,
            padding="valid",
            name="seg_conv2",
        )
        self.up = keras.layers.UpSampling2D(
            size=(upsample_factor, upsample_factor),
            interpolation="bilinear",
            name="seg_upsample",
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        if len(input_shape) != 4:
            raise ValueError(
                f"CliffordSegmentationHead expects 4D input (B,Hb,Wb,E), got "
                f"{input_shape}"
            )
        if input_shape[-1] != self.embed_dim:
            raise ValueError(
                f"input channel dim ({input_shape[-1]}) does not match "
                f"embed_dim ({self.embed_dim})."
            )
        self.conv1.build(input_shape)
        self.drop.build(input_shape)
        c2_in = (input_shape[0], input_shape[1], input_shape[2], self.embed_dim)
        self.conv2.build(c2_in)
        up_in = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            self.num_classes,
        )
        self.up.build(up_in)
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        x = self.conv1(inputs)
        x = self.drop(x, training=training)
        x = self.conv2(x)
        x = self.up(x)
        return x

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ):
        b, hb, wb, _ = input_shape
        h = None if hb is None else hb * self.upsample_factor
        w = None if wb is None else wb * self.upsample_factor
        return (b, h, w, self.num_classes)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_classes": self.num_classes,
                "upsample_factor": self.upsample_factor,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config
