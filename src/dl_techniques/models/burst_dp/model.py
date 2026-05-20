"""BurstDP: multi-view reference-conditioned vision model.

Takes one reference image plus a variable-size set of auxiliary views of the
same scene under different viewpoints / noise / transformations, and
produces three dense outputs about the reference:

    - ``recon``         : ``(B, H, W, 3)`` clean version of the (possibly
                          corrupted) reference
    - ``segmentation``  : ``(B, H, W, num_seg_classes)`` per-pixel logits

Auxiliary views are aggregated by *masked* cross-attention with the reference
patch tokens as queries, making the model permutation-invariant over the
auxiliary set and capable of handling any N in ``[0, n_max]`` at runtime.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import keras
from keras import layers, ops

from dl_techniques.models.vit.model import ViT
from dl_techniques.utils.logger import logger

from .config import BurstDPConfig, get_preset
from .fusion import BurstFusionBlock, BurstFusionBlockAdaLN
from .heads import ReconstructionHead, SegmentationHead


# ---------------------------------------------------------------------
# Fusion-block factory (internal — selects the fusion variant per config)
# ---------------------------------------------------------------------


def _build_fusion_block(
    fusion_type: str,
    dim: int,
    num_heads: int,
    mlp_ratio: float,
    dropout_rate: float,
    attention_dropout_rate: float,
    name: str,
):
    """Construct one fusion block according to ``fusion_type``.

    Both variants share the call contract
    ``call(ref_tokens, aux_tokens, aux_mask, training=None) -> ref_tokens``
    so the dispatch is invisible to :meth:`BurstDP.call`.
    """
    if fusion_type == "custom":
        return BurstFusionBlock(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            name=name,
        )
    if fusion_type == "adaln":
        return BurstFusionBlockAdaLN(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            name=name,
        )
    raise ValueError(f"Unknown fusion_type {fusion_type!r}")


@keras.saving.register_keras_serializable()
class BurstDP(keras.Model):
    """Multi-view reference-conditioned vision model.

    Args:
        config: A :class:`BurstDPConfig`. Use :func:`get_preset` for named variants.
        name: Optional model name.
        **kwargs: ``keras.Model`` base kwargs.
    """

    def __init__(
        self,
        config: Optional[BurstDPConfig] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if config is None:
            config = BurstDPConfig()
        config.validate()

        if name is None:
            name = f"burst_dp_{config.encoder_scale}"
        super().__init__(name=name, **kwargs)

        self.config = config

        # --- Shared encoder ---
        # ViT in feature-extraction mode returns (B, 1 + P, D) sequences.
        self.encoder = ViT(
            input_shape=(config.image_size, config.image_size, 3),
            num_classes=1,            # ignored — include_top=False
            scale=config.encoder_scale,
            patch_size=config.patch_size,
            include_top=False,
            pooling=None,
            dropout_rate=config.encoder_dropout_rate,
            attention_dropout_rate=config.encoder_attention_dropout_rate,
            name="encoder",
        )

        # Derived sizes.
        self.embed_dim: int = self.encoder.embed_dim
        self.num_patches: int = self.encoder.num_patches
        side = config.image_size // config.patch_size
        self.feature_hw: Tuple[int, int] = (side, side)

        if self.embed_dim % config.fusion_heads != 0:
            raise ValueError(
                f"encoder embed_dim ({self.embed_dim}) must be divisible by "
                f"fusion_heads ({config.fusion_heads}). Either change the "
                f"encoder_scale or the fusion_heads."
            )

        # --- Fusion stack ---
        self.fusion_blocks = [
            _build_fusion_block(
                fusion_type=config.fusion_type,
                dim=self.embed_dim,
                num_heads=config.fusion_heads,
                mlp_ratio=config.fusion_mlp_ratio,
                dropout_rate=config.dropout_rate,
                attention_dropout_rate=config.attention_dropout_rate,
                name=f"fusion_block_{i}",
            )
            for i in range(config.fusion_blocks)
        ]
        self.fusion_norm = layers.LayerNormalization(name="fusion_norm")

        # --- Heads ---
        # A 1x1 projection to align fused token dim to a head-friendly width.
        self.head_proj = layers.Dense(self.embed_dim, name="head_proj")

        self.recon_head = ReconstructionHead(
            decoder_dims=config.decoder_dims,
            patch_size=config.patch_size,
            out_channels=config.recon_channels,
            name="recon_head",
        )
        self.seg_head = SegmentationHead(
            decoder_dims=config.decoder_dims,
            patch_size=config.patch_size,
            num_classes=config.num_seg_classes,
            name="seg_head",
        )

        logger.info(
            f"BurstDP({config.encoder_scale}): image={config.image_size}px, "
            f"patches={self.num_patches} ({side}x{side}), embed={self.embed_dim}, "
            f"fusion_blocks={config.fusion_blocks} (type={config.fusion_type})"
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _encode_batch(self, x: keras.KerasTensor, training: Optional[bool]) -> keras.KerasTensor:
        """Run the shared encoder on ``(K, H, W, 3)`` and return patch tokens
        ``(K, T, D)`` with the CLS token stripped."""
        seq = self.encoder(x, training=training)
        # seq: (K, 1 + T, D); strip CLS.
        return seq[:, 1:, :]

    def call(
        self,
        inputs: Dict[str, keras.KerasTensor],
        training: Optional[bool] = None,
    ) -> Dict[str, keras.KerasTensor]:
        """Forward pass.

        Args:
            inputs: Dict with keys
                ``ref``      shape ``(B, H, W, 3)`` float32
                ``aux``      shape ``(B, N_max, H, W, 3)`` float32
                ``aux_mask`` shape ``(B, N_max)`` float32 (1.0=valid, 0.0=pad)
            training: Keras training flag.

        Returns:
            Dict with keys ``recon`` and ``segmentation``.
        """
        ref = inputs["ref"]
        aux = inputs["aux"]
        aux_mask = inputs["aux_mask"]

        b = ops.shape(ref)[0]
        n = self.config.n_max
        h = self.config.image_size
        w = self.config.image_size

        # --- Encode ref + aux in one fused pass for parameter sharing. ---
        ref_in = ref  # (B, H, W, 3)
        aux_in = ops.reshape(aux, (b * n, h, w, 3))    # (B*N, H, W, 3)
        merged = ops.concatenate([ref_in, aux_in], axis=0)  # (B*(1+N), H, W, 3)
        tokens = self._encode_batch(merged, training=training)  # (B*(1+N), T, D)

        t = ops.shape(tokens)[1]
        d = ops.shape(tokens)[2]

        ref_tokens = tokens[:b]                                  # (B, T, D)
        aux_tokens = tokens[b:]                                  # (B*N, T, D)
        aux_tokens = ops.reshape(aux_tokens, (b, n, t, d))       # (B, N, T, D)

        # Zero padded aux slots so they cannot leak via the cross-attention KV.
        mask_b_n_1_1 = ops.reshape(aux_mask, (b, n, 1, 1))
        aux_tokens = aux_tokens * ops.cast(mask_b_n_1_1, aux_tokens.dtype)

        # --- Fusion stack ---
        for block in self.fusion_blocks:
            ref_tokens = block(ref_tokens, aux_tokens, aux_mask, training=training)
        ref_tokens = self.fusion_norm(ref_tokens)

        # --- Head input: project + reshape to 2D spatial feature map. ---
        ref_tokens = self.head_proj(ref_tokens)
        side_h, side_w = self.feature_hw
        feat = ops.reshape(ref_tokens, (b, side_h, side_w, self.embed_dim))

        # DECISION plan_2026-05-20_b8f8df89/D-001
        # Residual reconstruction: the recon head emits a signed delta and the
        # clean image is `ref + delta`. The `ref` skip is an identity gradient
        # path that cannot vanish — without it the recon path collapsed to a
        # blurry mean (gradient global-norm 17.1 -> 0.006 in 50 steps). `clip`
        # keeps the output in [0,1] for the PSNR/SSIM metrics and Charbonnier
        # loss. See plans/plan_2026-05-20_b8f8df89/decisions.md D-001.
        recon_delta = self.recon_head(feat, training=training)
        outputs: Dict[str, keras.KerasTensor] = {
            "recon": ops.clip(ref + recon_delta, 0.0, 1.0),
            "segmentation": self.seg_head(feat, training=training),
        }

        return outputs

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"config": self.config.to_dict()})
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BurstDP":
        cfg_dict = config.pop("config", {})
        return cls(config=BurstDPConfig.from_dict(cfg_dict), **config)


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------


def create_burst_dp(
    preset: str = "burst_dp_small",
    **overrides: Any,
) -> BurstDP:
    """Create a :class:`BurstDP` from a preset with optional config overrides.

    Args:
        preset: One of ``burst_dp_pico``, ``burst_dp_tiny``, ``burst_dp_small``,
            ``burst_dp_base``.
        **overrides: Per-field overrides applied on top of the preset.

    Returns:
        A configured (unbuilt) :class:`BurstDP`.
    """
    cfg = get_preset(preset, **overrides)
    return BurstDP(config=cfg)
