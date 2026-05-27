"""Configuration dataclass for :class:`ConvNeXtPatchVAEV2`.

Extends V1 (``dl_techniques.models.convnext_patch_vae.config``) with
multi-task head flags, MAE mask ratio, and the LPIPS perceptual term.
V1 fields kept with identical defaults so a V2 config with all multi-task
flags OFF degenerates to a V1-equivalent training recipe.

Anchored decisions (see ``plans/plan_2026-05-27_4a444b14/decisions.md``):

- D-002 SimMIM-style MAE masking: ``mae_mask_ratio``.
- D-005 iter-1 scope: ``xl`` preset; ``xxl`` deferred.
- D-006 cls head wired to CIFAR-style labels in iter-1; seg head
  unit-tested with synthetic labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass
class ConvNeXtPatchVAEV2Config:
    """Configuration for :class:`ConvNeXtPatchVAEV2`.

    V1 fields are unchanged from
    ``dl_techniques.models.convnext_patch_vae.ConvNeXtPatchVAEConfig``.
    V2 adds the fields under the "Multi-task" and "LPIPS" sections below.

    Args:
        img_size: Default square input edge length.
        img_channels: Number of pixel channels.
        patch_size: Non-overlapping patch edge length.
        embed_dim: ConvNeXt block channel width.
        encoder_depth: Number of ``ConvNextV2Block`` layers in encoder.
        decoder_depth: Number of ``ConvNextV2Block`` layers in decoder.
        kernel_size: Depthwise kernel size inside each block.
        latent_dim: Per-patch latent channel width.
        beta_kl: Scalar weight on per-patch KL term.
        lambda_sigreg: Scalar weight on SIGReg term.
        sigreg_knots: Integration knots for SIGReg.
        sigreg_num_proj: Random projections for SIGReg.
        recon_loss_type: ``"mse"`` or ``"bce"``.
        dropout_rate: Per-block dropout.
        spatial_dropout_rate: Per-block spatial dropout.
        gamma_clip: Symmetric gradient clip; ``None`` disables.
        kernel_regularizer_config: Serialized Keras regularizer config.

        mae_mask_ratio: SimMIM-style random patch mask ratio applied at
            the encoder's post-stem feature map during training. ``0.0``
            disables masking (V1-equivalent). Range ``[0.0, 0.95]``.
        lambda_mae: Multiplier applied to the *masked-patch* portion of
            the reconstruction loss when MAE masking is active. Total
            recon = mean_unmasked + ``lambda_mae`` * mean_masked.
        lambda_lpips: Scalar weight on the LPIPS perceptual loss. ``0.0``
            disables the term.
        lpips_layer_weights: Optional mapping from VGG layer name to
            per-layer weight; ``None`` uses LPIPS default.
        lpips_input_range: Tuple ``(low, high)`` describing the recon /
            target image scale. Defaults to ``(0.0, 1.0)``.

        use_classification_head: When True, model exposes an attention-
            pool + MLP head producing ``(B, num_classes_cls)`` logits.
        num_classes_cls: Number of classes for the cls head.
        cls_head_dropout: Dropout inside the cls head MLP.
        lambda_cls: Scalar weight on classification CE loss.

        use_segmentation_head: When True, model exposes a bilinear-upsample
            head producing ``(B, H, W, num_classes_seg)`` logits.
        num_classes_seg: Number of seg classes (background included).
        seg_head_dropout: Dropout inside the seg head.
        lambda_seg: Scalar weight on per-pixel CE seg loss.

    Note:
        V2 invariants (in addition to V1's): mae_mask_ratio in
        ``[0.0, 0.95]``; cls flag → ``num_classes_cls >= 2``; seg flag →
        ``num_classes_seg >= 2``; all lambda weights ``>= 0.0``.
    """

    # --- Vision / patches (V1) ---
    img_size: int = 32
    img_channels: int = 3
    patch_size: int = 4

    # --- ConvNeXt backbone (V1) ---
    embed_dim: int = 128
    encoder_depth: int = 4
    decoder_depth: int = 4
    kernel_size: int = 7

    # --- VAE latent (V1) ---
    latent_dim: int = 16

    # --- VAE loss weights (V1) ---
    beta_kl: float = 0.5
    lambda_sigreg: float = 0.1
    sigreg_knots: int = 17
    sigreg_num_proj: int = 256
    recon_loss_type: str = "bce"

    # --- Regularization (V1) ---
    dropout_rate: float = 0.0
    spatial_dropout_rate: float = 0.0
    gamma_clip: Optional[float] = 1.0
    kernel_regularizer_config: Optional[Dict[str, Any]] = field(default=None)

    # --- V2: MAE masked-reconstruction ---
    mae_mask_ratio: float = 0.0
    lambda_mae: float = 1.0

    # --- V2: LPIPS perceptual ---
    lambda_lpips: float = 0.0
    lpips_layer_weights: Optional[Dict[str, float]] = field(default=None)
    lpips_input_range: Tuple[float, float] = (0.0, 1.0)

    # --- V2: Classification head ---
    use_classification_head: bool = False
    num_classes_cls: int = 0
    cls_head_dropout: float = 0.0
    cls_head_num_heads: int = 4
    lambda_cls: float = 1.0

    # --- V2: Segmentation head ---
    use_segmentation_head: bool = False
    num_classes_seg: int = 0
    seg_head_dropout: float = 0.0
    lambda_seg: float = 1.0

    # ------------------------------------------------------------------
    # Invariants
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        # V1 invariants
        if self.img_size <= 0:
            raise ValueError(f"img_size must be positive, got {self.img_size}")
        if self.img_channels <= 0:
            raise ValueError(
                f"img_channels must be positive, got {self.img_channels}"
            )
        if self.patch_size <= 0:
            raise ValueError(
                f"patch_size must be positive, got {self.patch_size}"
            )
        if self.img_size % self.patch_size != 0:
            raise ValueError(
                f"img_size ({self.img_size}) must be divisible by patch_size "
                f"({self.patch_size})."
            )
        if self.embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {self.embed_dim}")
        if self.encoder_depth < 1:
            raise ValueError(
                f"encoder_depth must be >= 1, got {self.encoder_depth}"
            )
        if self.decoder_depth < 1:
            raise ValueError(
                f"decoder_depth must be >= 1, got {self.decoder_depth}"
            )
        if self.kernel_size <= 0:
            raise ValueError(
                f"kernel_size must be positive, got {self.kernel_size}"
            )
        if self.latent_dim < 1:
            raise ValueError(f"latent_dim must be >= 1, got {self.latent_dim}")
        if self.beta_kl < 0.0:
            raise ValueError(f"beta_kl must be >= 0.0, got {self.beta_kl}")
        if self.lambda_sigreg < 0.0:
            raise ValueError(
                f"lambda_sigreg must be >= 0.0, got {self.lambda_sigreg}"
            )
        if self.sigreg_knots < 2:
            raise ValueError(
                f"sigreg_knots must be >= 2, got {self.sigreg_knots}"
            )
        if self.sigreg_num_proj < 1:
            raise ValueError(
                f"sigreg_num_proj must be >= 1, got {self.sigreg_num_proj}"
            )
        if self.recon_loss_type not in {"mse", "bce"}:
            raise ValueError(
                f"recon_loss_type must be one of {{'mse', 'bce'}}, got "
                f"{self.recon_loss_type!r}"
            )
        if not (0.0 <= self.dropout_rate <= 1.0):
            raise ValueError(
                f"dropout_rate must be in [0.0, 1.0], got {self.dropout_rate}"
            )
        if not (0.0 <= self.spatial_dropout_rate <= 1.0):
            raise ValueError(
                f"spatial_dropout_rate must be in [0.0, 1.0], got "
                f"{self.spatial_dropout_rate}"
            )
        if self.gamma_clip is not None and self.gamma_clip <= 0.0:
            raise ValueError(
                f"gamma_clip must be > 0.0 or None, got {self.gamma_clip}"
            )

        # V2 invariants
        if not (0.0 <= self.mae_mask_ratio <= 0.95):
            raise ValueError(
                f"mae_mask_ratio must be in [0.0, 0.95], got "
                f"{self.mae_mask_ratio}"
            )
        if self.lambda_mae < 0.0:
            raise ValueError(
                f"lambda_mae must be >= 0.0, got {self.lambda_mae}"
            )
        if self.lambda_lpips < 0.0:
            raise ValueError(
                f"lambda_lpips must be >= 0.0, got {self.lambda_lpips}"
            )
        if (
            self.lpips_input_range[1] <= self.lpips_input_range[0]
        ):
            raise ValueError(
                f"lpips_input_range high must exceed low, got "
                f"{self.lpips_input_range}"
            )

        if self.use_classification_head:
            if self.num_classes_cls < 2:
                raise ValueError(
                    "use_classification_head=True requires "
                    f"num_classes_cls >= 2, got {self.num_classes_cls}."
                )
            if not (0.0 <= self.cls_head_dropout <= 1.0):
                raise ValueError(
                    f"cls_head_dropout must be in [0.0, 1.0], got "
                    f"{self.cls_head_dropout}"
                )
            if self.cls_head_num_heads < 1:
                raise ValueError(
                    f"cls_head_num_heads must be >= 1, got "
                    f"{self.cls_head_num_heads}"
                )
            if self.lambda_cls < 0.0:
                raise ValueError(
                    f"lambda_cls must be >= 0.0, got {self.lambda_cls}"
                )
            if self.embed_dim % self.cls_head_num_heads != 0:
                raise ValueError(
                    f"cls_head_num_heads ({self.cls_head_num_heads}) "
                    f"must divide embed_dim ({self.embed_dim})."
                )

        if self.use_segmentation_head:
            if self.num_classes_seg < 2:
                raise ValueError(
                    "use_segmentation_head=True requires "
                    f"num_classes_seg >= 2, got {self.num_classes_seg}."
                )
            if not (0.0 <= self.seg_head_dropout <= 1.0):
                raise ValueError(
                    f"seg_head_dropout must be in [0.0, 1.0], got "
                    f"{self.seg_head_dropout}"
                )
            if self.lambda_seg < 0.0:
                raise ValueError(
                    f"lambda_seg must be >= 0.0, got {self.lambda_seg}"
                )

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------
    @property
    def patches_per_side(self) -> int:
        return self.img_size // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.patches_per_side ** 2

    @property
    def input_image_shape(self) -> Tuple[int, int, int]:
        return (self.img_size, self.img_size, self.img_channels)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # `lpips_input_range` is a tuple; asdict preserves it as a tuple
        # but JSON deserialization downstream returns a list — normalize.
        d["lpips_input_range"] = list(self.lpips_input_range)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ConvNeXtPatchVAEV2Config":
        import dataclasses as _dc

        valid = {f.name for f in _dc.fields(cls)}
        d = {k: v for k, v in dict(d).items() if k in valid}
        if "lpips_input_range" in d:
            d["lpips_input_range"] = tuple(d["lpips_input_range"])
        return cls(**d)


# ---------------------------------------------------------------------------
# Named presets — same shape as V1.PRESETS plus `xl`.
# ---------------------------------------------------------------------------

PRESETS: Dict[str, Dict[str, Any]] = {
    "tiny": {"embed_dim": 64, "encoder_depth": 2, "decoder_depth": 2, "latent_dim": 8},
    "base": {"embed_dim": 128, "encoder_depth": 4, "decoder_depth": 4, "latent_dim": 16},
    "large": {"embed_dim": 192, "encoder_depth": 6, "decoder_depth": 6, "latent_dim": 32},
    # New in V2: ~10M params backbone — scaled-up for downstream transfer.
    # DECISION plan_2026-05-27_4a444b14/D-005: `xxl` (384/12/12/128) is
    # deferred to a follow-up after `xl` validation.
    "xl": {"embed_dim": 256, "encoder_depth": 8, "decoder_depth": 8, "latent_dim": 64},
}
