"""Task heads for :class:`ConvNeXtPatchVAEV2`.

Two heads, both consuming the encoder's pre-bottleneck feature map
``(B, Hp, Wp, embed_dim)``:

- :class:`AttentionPoolClassifierHead` — learnable CLS-token cross-attention
  pool over the patch grid, followed by an MLP, producing
  ``(B, num_classes)`` logits.
- :class:`SegmentationHead` — light Conv-GELU-Conv head followed by a
  bilinear upsample to pixel space, producing
  ``(B, H, W, num_classes_seg)`` logits.

Both heads serialize cleanly via ``get_config`` and are tested in
``tests/test_models/test_convnext_patch_vae_v2/test_v2_heads.py``.

Per plan_2026-05-27_4a444b14, the cls head uses ``keras.layers.MultiHeadAttention``
directly (stable Keras 3 primitive) rather than coupling to internal
``dl_techniques.layers.attention.*`` APIs.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import keras
from keras import ops


# ---------------------------------------------------------------------------
# Classification head — learnable CLS query + 1 cross-attention pool
# ---------------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="dl_techniques")
class AttentionPoolClassifierHead(keras.layers.Layer):
    """Attention-pool classifier head.

    Mechanism (1 transformer-style block with a single learnable query):

    .. code-block::

        x : (B, Hp, Wp, E)
          → flatten → tokens (B, N, E)   where N = Hp * Wp
          → cross-attn(CLS_query=q, k=v=tokens) → pooled (B, 1, E)
          → LayerNorm → MLP(E → 4E → E) → LayerNorm
          → Linear(num_classes) → logits (B, num_classes)

    The CLS query is a single learnable embedding of shape ``(1, 1, E)``,
    broadcast across the batch. This is the standard "attention pool"
    used by ViT-classifier variants (cf. CaiT, DeiT III).

    Args:
        embed_dim: Feature dimensionality at input (the ConvNeXt
            backbone's ``embed_dim``).
        num_classes: Number of output classes (logits dimension).
        num_heads: Number of attention heads. Must divide ``embed_dim``.
        mlp_expansion: Internal MLP expansion ratio. Default 4.
        dropout_rate: Dropout applied inside the MLP and on attention
            output. Default 0.
        name: Layer name.
        **kwargs: Forwarded to ``keras.layers.Layer``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        num_heads: int = 4,
        mlp_expansion: int = 4,
        dropout_rate: float = 0.0,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must divide embed_dim ({embed_dim})."
            )
        if mlp_expansion < 1:
            raise ValueError(
                f"mlp_expansion must be >= 1, got {mlp_expansion}"
            )
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(
                f"dropout_rate must be in [0.0, 1.0], got {dropout_rate}"
            )

        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.mlp_expansion = mlp_expansion
        self.dropout_rate = dropout_rate

        # Learnable CLS query — created in build().
        self.cls_query: Optional[keras.Variable] = None

        self.norm_q = keras.layers.LayerNormalization(
            epsilon=1e-6, name="norm_q"
        )
        self.norm_kv = keras.layers.LayerNormalization(
            epsilon=1e-6, name="norm_kv"
        )
        self.attn = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout_rate,
            name="cross_attn",
        )
        self.norm_mlp = keras.layers.LayerNormalization(
            epsilon=1e-6, name="norm_mlp"
        )
        self.mlp_fc1 = keras.layers.Dense(
            embed_dim * mlp_expansion, activation="gelu", name="mlp_fc1"
        )
        self.mlp_drop1 = keras.layers.Dropout(dropout_rate, name="mlp_drop1")
        self.mlp_fc2 = keras.layers.Dense(embed_dim, name="mlp_fc2")
        self.mlp_drop2 = keras.layers.Dropout(dropout_rate, name="mlp_drop2")
        self.head_norm = keras.layers.LayerNormalization(
            epsilon=1e-6, name="head_norm"
        )
        self.classifier = keras.layers.Dense(num_classes, name="classifier")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        if len(input_shape) != 4:
            raise ValueError(
                f"AttentionPoolClassifierHead expects 4D input (B, Hp, Wp, E), "
                f"got {input_shape}"
            )
        if input_shape[-1] != self.embed_dim:
            raise ValueError(
                f"input channel dim ({input_shape[-1]}) does not match "
                f"embed_dim ({self.embed_dim})."
            )

        self.cls_query = self.add_weight(
            shape=(1, 1, self.embed_dim),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
            name="cls_query",
        )

        # Token (B, N, E) shape — N unknown at build time, but Keras MHA
        # accepts unknown sequence length.
        token_shape = (input_shape[0], None, self.embed_dim)
        query_shape = (input_shape[0], 1, self.embed_dim)

        self.norm_q.build(query_shape)
        self.norm_kv.build(token_shape)
        # MHA build wants (query_shape, value_shape).
        self.attn.build(query_shape=query_shape, value_shape=token_shape)
        pooled_shape = (input_shape[0], 1, self.embed_dim)
        self.norm_mlp.build(pooled_shape)
        self.mlp_fc1.build(pooled_shape)
        mid_shape = (input_shape[0], 1, self.embed_dim * self.mlp_expansion)
        self.mlp_drop1.build(mid_shape)
        self.mlp_fc2.build(mid_shape)
        self.mlp_drop2.build(pooled_shape)
        self.head_norm.build(pooled_shape)
        # classifier consumes pooled (B, E) after squeeze.
        self.classifier.build((input_shape[0], self.embed_dim))

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Forward pass.

        Args:
            inputs: ``(B, Hp, Wp, E)``.
            training: Standard Keras flag.

        Returns:
            Logits ``(B, num_classes)``.
        """
        shape = ops.shape(inputs)
        b = shape[0]
        hp = shape[1]
        wp = shape[2]
        # Flatten spatial: (B, Hp*Wp, E).
        tokens = ops.reshape(inputs, (b, hp * wp, self.embed_dim))

        # Broadcast CLS query across batch.
        query = ops.broadcast_to(self.cls_query, (b, 1, self.embed_dim))

        q_n = self.norm_q(query, training=training)
        kv_n = self.norm_kv(tokens, training=training)

        pooled = self.attn(
            query=q_n, value=kv_n, key=kv_n, training=training
        )
        # Residual: query + attention output.
        pooled = query + pooled

        # MLP residual block.
        h = self.norm_mlp(pooled, training=training)
        h = self.mlp_fc1(h)
        h = self.mlp_drop1(h, training=training)
        h = self.mlp_fc2(h)
        h = self.mlp_drop2(h, training=training)
        pooled = pooled + h

        pooled = self.head_norm(pooled, training=training)
        # Squeeze the singleton seq dim and run the linear classifier.
        pooled = ops.squeeze(pooled, axis=1)
        logits = self.classifier(pooled)
        return logits

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], int]:
        return (input_shape[0], self.num_classes)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_classes": self.num_classes,
                "num_heads": self.num_heads,
                "mlp_expansion": self.mlp_expansion,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


# ---------------------------------------------------------------------------
# Segmentation head — light Conv-GELU-Conv + bilinear upsample to pixels
# ---------------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="dl_techniques")
class SegmentationHead(keras.layers.Layer):
    """Segmentation head: per-patch logits + bilinear upsample to pixel grid.

    .. code-block::

        x : (B, Hp, Wp, E)
          → Conv2D(E, 3, "same") → GELU → Dropout
          → Conv2D(num_classes_seg, 1)
          → bilinear upsample by `patch_size`
          → logits (B, H, W, num_classes_seg)

    Args:
        embed_dim: Feature dim at input.
        num_classes: Number of seg classes (including background).
        patch_size: Pixel upsample factor (matches encoder's
            ``patch_size``).
        dropout_rate: Dropout between the two convs.
        name: Layer name.
        **kwargs: Forwarded to ``keras.layers.Layer``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        patch_size: int,
        dropout_rate: float = 0.0,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(
                f"dropout_rate must be in [0.0, 1.0], got {dropout_rate}"
            )

        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.patch_size = patch_size
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
        # `UpSampling2D` with bilinear interpolation is the standard
        # resolution-agnostic upsampler. It operates on dynamic H/W.
        self.up = keras.layers.UpSampling2D(
            size=(patch_size, patch_size),
            interpolation="bilinear",
            name="seg_upsample",
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        if len(input_shape) != 4:
            raise ValueError(
                f"SegmentationHead expects 4D input (B, Hp, Wp, E), got "
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
        B, Hp, Wp, _ = input_shape
        H = None if Hp is None else Hp * self.patch_size
        W = None if Wp is None else Wp * self.patch_size
        return (B, H, W, self.num_classes)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_classes": self.num_classes,
                "patch_size": self.patch_size,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config
