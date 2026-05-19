"""BurstDP configuration.

Multi-view reference-conditioned vision model. One privileged reference frame
is fused with a variable number (1-5 in practice; 0..N_max supported) of
auxiliary views via masked cross-attention.

Two task heads are wired:
    - reconstruction: clean version of the (corrupted) reference
    - segmentation:   per-pixel class indices (COCO 81-way)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Tuple

# Public defaults exposed for downstream callers / tests.
DEFAULT_IMAGE_SIZE: int = 256
DEFAULT_PATCH_SIZE: int = 16
DEFAULT_NUM_SEG_CLASSES: int = 81   # 80 COCO things + background
DEFAULT_N_MAX: int = 5

# Valid fusion-block kinds. `custom` = current asymmetric cross-attention
# fusion (BurstFusionBlock). `adaln` = modulation-based fusion via
# AdaLNZeroConditionalBlock with mask-aware aux pooling — see plan
# plan_2026-05-19_39a6a454 and findings/02-adaln-aux-conditioning.md.
FusionType = Literal["custom", "adaln"]
VALID_FUSION_TYPES: Tuple[str, ...] = ("custom", "adaln")


@dataclass
class BurstDPConfig:
    """Configuration for :class:`BurstDP`.

    Attributes:
        image_size: Side length (square). Must be divisible by ``patch_size``.
        patch_size: ViT patch size. Square patches.
        n_max: Maximum number of auxiliary views supported at inference.
            Variable N at runtime is enabled by masking; this only fixes the
            padding dimension. Set to 5 to match the spec's "ideally 1-5".
        encoder_scale: ViT scale key (``pico``/``tiny``/``small``/``base``/...).
        fusion_blocks: Number of {self-attn -> cross-attn -> FFN} fusion
            blocks stacked after the shared encoder.
        fusion_heads: Number of heads for fusion self/cross attention.
            Must divide the encoder embedding dim cleanly.
        fusion_mlp_ratio: FFN expansion ratio inside each fusion block.
        dropout_rate: Generic dropout applied in fusion attention + FFN.
        attention_dropout_rate: Dropout on attention weights specifically.
        recon_channels: Output channels for reconstruction (3 for RGB).
        num_seg_classes: Segmentation classes incl. background.
        decoder_dims: DPT decoder channel sequence. Length must be at least
            ``log2(patch_size)`` so the cumulative bilinear upsample
            recovers the input resolution.
    """

    # Geometry
    image_size: int = DEFAULT_IMAGE_SIZE
    patch_size: int = DEFAULT_PATCH_SIZE
    n_max: int = DEFAULT_N_MAX

    # Encoder
    encoder_scale: str = "small"   # 384-d / 6h / 12L
    encoder_dropout_rate: float = 0.0
    encoder_attention_dropout_rate: float = 0.0

    # Fusion
    fusion_type: str = "custom"   # one of ``VALID_FUSION_TYPES`` — see FusionType
    fusion_blocks: int = 4
    fusion_heads: int = 6
    fusion_mlp_ratio: float = 4.0
    dropout_rate: float = 0.0
    attention_dropout_rate: float = 0.0

    # Heads
    recon_channels: int = 3
    num_seg_classes: int = DEFAULT_NUM_SEG_CLASSES
    decoder_dims: Tuple[int, ...] = (256, 128, 64, 32)

    def validate(self) -> None:
        if self.image_size <= 0 or self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size ({self.image_size}) must be a positive multiple "
                f"of patch_size ({self.patch_size})."
            )
        if self.n_max < 1:
            raise ValueError(f"n_max must be >= 1 (model still supports N=0 at runtime), got {self.n_max}")
        # DPT decoder cumulative upsample == patch_size; need enough stages.
        ps = self.patch_size
        n_up = 0
        while ps > 1:
            if ps % 2 != 0:
                raise ValueError(f"patch_size must be a power of 2, got {self.patch_size}")
            ps //= 2
            n_up += 1
        if len(self.decoder_dims) < n_up:
            raise ValueError(
                f"decoder_dims has {len(self.decoder_dims)} stages but "
                f"patch_size={self.patch_size} needs at least {n_up} 2x upsamples."
            )
        if self.fusion_type not in VALID_FUSION_TYPES:
            raise ValueError(
                f"fusion_type={self.fusion_type!r} not in {VALID_FUSION_TYPES}."
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "n_max": self.n_max,
            "encoder_scale": self.encoder_scale,
            "encoder_dropout_rate": self.encoder_dropout_rate,
            "encoder_attention_dropout_rate": self.encoder_attention_dropout_rate,
            "fusion_type": self.fusion_type,
            "fusion_blocks": self.fusion_blocks,
            "fusion_heads": self.fusion_heads,
            "fusion_mlp_ratio": self.fusion_mlp_ratio,
            "dropout_rate": self.dropout_rate,
            "attention_dropout_rate": self.attention_dropout_rate,
            "recon_channels": self.recon_channels,
            "num_seg_classes": self.num_seg_classes,
            "decoder_dims": list(self.decoder_dims),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BurstDPConfig":
        return cls(
            image_size=int(d.get("image_size", DEFAULT_IMAGE_SIZE)),
            patch_size=int(d.get("patch_size", DEFAULT_PATCH_SIZE)),
            n_max=int(d.get("n_max", DEFAULT_N_MAX)),
            encoder_scale=str(d.get("encoder_scale", "small")),
            encoder_dropout_rate=float(d.get("encoder_dropout_rate", 0.0)),
            encoder_attention_dropout_rate=float(d.get("encoder_attention_dropout_rate", 0.0)),
            fusion_type=str(d.get("fusion_type", "custom")),
            fusion_blocks=int(d.get("fusion_blocks", 4)),
            fusion_heads=int(d.get("fusion_heads", 6)),
            fusion_mlp_ratio=float(d.get("fusion_mlp_ratio", 4.0)),
            dropout_rate=float(d.get("dropout_rate", 0.0)),
            attention_dropout_rate=float(d.get("attention_dropout_rate", 0.0)),
            recon_channels=int(d.get("recon_channels", 3)),
            num_seg_classes=int(d.get("num_seg_classes", DEFAULT_NUM_SEG_CLASSES)),
            decoder_dims=tuple(d.get("decoder_dims", (256, 128, 64, 32))),
        )


# Preset configurations.
PRESETS: Dict[str, Dict[str, Any]] = {
    "burst_dp_pico": {
        "encoder_scale": "pico",
        "fusion_blocks": 2,
        "fusion_heads": 3,
        "decoder_dims": (96, 64, 48, 32),
    },
    "burst_dp_tiny": {
        "encoder_scale": "tiny",
        "fusion_blocks": 3,
        "fusion_heads": 3,
        "decoder_dims": (128, 96, 64, 32),
    },
    "burst_dp_small": {
        "encoder_scale": "small",
        "fusion_blocks": 4,
        "fusion_heads": 6,
        "decoder_dims": (256, 128, 64, 32),
    },
    "burst_dp_base": {
        "encoder_scale": "base",
        "fusion_blocks": 6,
        "fusion_heads": 12,
        "decoder_dims": (256, 192, 128, 64),
    },
}


def get_preset(name: str, **overrides: Any) -> BurstDPConfig:
    """Return a :class:`BurstDPConfig` from a preset name with optional overrides."""
    if name not in PRESETS:
        raise ValueError(f"Unknown BurstDP preset '{name}'. Available: {list(PRESETS)}")
    base: Dict[str, Any] = dict(PRESETS[name])
    base.update(overrides)
    cfg = BurstDPConfig.from_dict(base)
    cfg.validate()
    return cfg
