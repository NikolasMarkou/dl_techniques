"""Configuration dataclass for :class:`CliffordNetPatchVAEV2`.

Sibling of ``dl_techniques.models.convnext_patch_vae_v2.config`` but
hierarchical: the encoder/decoder is a stack of N stages
(``stage_dims[i]``, ``stage_depths[i]``, ``stage_shifts[i]``) interleaved
with stride-2 :class:`CliffordNetBlockDSv2` transitions (encoder) and
bilinear upsamples + 1x1 projections (decoder). All other v2 features
(MAE mask, KL/SIGReg/LPIPS/cls/seg losses) are preserved field-for-field.

Anchored decisions (see ``plans/plan_2026-05-27_75849a91/decisions.md``):

- D-001 new sibling packages (no edits to v2).
- D-002 CliffordNetBlock has internal residual — encoder/decoder loops
  must NOT add an outer residual.
- D-003 new ``CliffordSegmentationHead`` with explicit ``upsample_factor``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CliffordNetPatchVAEV2Config:
    """Configuration for :class:`CliffordNetPatchVAEV2`.

    Args:
        img_size: Default square input edge length.
        img_channels: Number of pixel channels.
        patch_size: Stem stride / kernel.

        stage_dims: Channel width per encoder stage. ``len(stage_dims) ==
            num_stages``. Decoder mirrors it in reverse.
        stage_depths: Number of :class:`CliffordNetBlock` layers per stage.
        stage_shifts: Per-stage list of channel-roll shifts handed to each
            :class:`CliffordNetBlock` and the matching decoder stage.
        downsample_kind: Pool kind for :class:`CliffordNetBlockDSv2`
            transitions (``"avg" | "max" | "blur" | "gaussian_dw" |
            "pixel_unshuffle" | "resnetd"``).
        downsample_kernel_size: Depthwise kernel size inside
            :class:`CliffordNetBlockDSv2`.
        cli_mode: Algebraic components for each block (``"inner" |
            "wedge" | "full"``).
        ctx_mode: Context-stream mode (``"diff" | "abs"``).
        use_global_context_in_last_stage: When True, the LAST encoder
            stage's blocks use ``use_global_context=True`` (GAP branch).
            The decoder's first (deepest) stage does the same for symmetry.
        layer_scale_init: Initial gamma for :class:`GatedGeometricResidual`
            LayerScale inside every block.
        block_drop_path_rate: Maximum DropPath probability. Schedule is
            linear from 0 (first block) to this value (last block) across
            the entire encoder stack.

        latent_dim: Per-bottleneck-cell latent channel width.
        beta_kl: Scalar weight on per-cell KL term.
        lambda_sigreg: Scalar weight on SIGReg term.
        sigreg_knots: Integration knots for SIGReg.
        sigreg_num_proj: Random projections for SIGReg.
        recon_loss_type: ``"mse"`` or ``"bce"``.

        gamma_clip: Symmetric gradient clip; ``None`` disables.
        kernel_regularizer_config: Serialized Keras regularizer config.

        mae_mask_ratio: SimMIM-style random patch mask ratio applied at
            the encoder's post-stem feature map during training. ``0.0``
            disables masking. Range ``[0.0, 0.95]``.
        lambda_mae: Multiplier on the masked-patch portion of the
            reconstruction loss when MAE masking is active.

        lambda_lpips: Scalar weight on the LPIPS perceptual loss.
        lpips_layer_weights: Optional mapping from VGG layer name to
            per-layer weight.
        lpips_input_range: Tuple ``(low, high)`` describing the recon /
            target image scale.

        use_classification_head: When True, expose an attention-pool +
            MLP head producing ``(B, num_classes_cls)`` logits.
        num_classes_cls: Output class count for the cls head.
        cls_head_dropout: Dropout inside the cls head MLP.
        cls_head_num_heads: MHA heads in cls head. Must divide
            ``stage_dims[-1]``.
        lambda_cls: Scalar weight on classification CE loss.

        use_segmentation_head: When True, expose a bilinear-upsample head
            producing ``(B, H, W, num_classes_seg)`` logits.
        num_classes_seg: Number of seg classes (background included).
        seg_head_dropout: Dropout inside the seg head.
        lambda_seg: Scalar weight on per-pixel CE seg loss.

    Invariants enforced in ``__post_init__``:
        - ``img_size % patch_size == 0``.
        - ``len(stage_dims) == len(stage_depths) == len(stage_shifts) >= 1``.
        - All stage entries are positive; shifts are non-empty.
        - The patch grid ``Hp = img_size // patch_size`` is divisible by
          ``2**(num_stages-1)`` (every downsample halves spatial dims).
        - ``mae_mask_ratio in [0.0, 0.95]``; lambdas ``>= 0.0``.
        - cls flag → ``num_classes_cls >= 2`` and ``cls_head_num_heads``
          divides ``stage_dims[-1]``.
        - seg flag → ``num_classes_seg >= 2``.
    """

    # --- Vision / patches ---
    img_size: int = 32
    img_channels: int = 3
    patch_size: int = 4

    # --- Hierarchical CliffordNet backbone ---
    stage_dims: List[int] = field(default_factory=lambda: [96, 192])
    stage_depths: List[int] = field(default_factory=lambda: [2, 4])
    stage_shifts: List[List[int]] = field(
        default_factory=lambda: [[1, 2], [1, 2, 4]]
    )
    downsample_kind: str = "blur"
    downsample_kernel_size: int = 7
    cli_mode: str = "full"
    ctx_mode: str = "diff"
    use_global_context_in_last_stage: bool = True
    layer_scale_init: float = 1e-5
    block_drop_path_rate: float = 0.0

    # --- VAE latent ---
    latent_dim: int = 16

    # --- VAE loss weights ---
    beta_kl: float = 0.5
    lambda_sigreg: float = 0.1
    sigreg_knots: int = 17
    sigreg_num_proj: int = 256
    recon_loss_type: str = "bce"

    # --- Regularization ---
    gamma_clip: Optional[float] = 1.0
    kernel_regularizer_config: Optional[Dict[str, Any]] = field(default=None)

    # --- MAE masked-reconstruction ---
    mae_mask_ratio: float = 0.0
    lambda_mae: float = 1.0

    # --- LPIPS perceptual ---
    lambda_lpips: float = 0.0
    lpips_layer_weights: Optional[Dict[str, float]] = field(default=None)
    lpips_input_range: Tuple[float, float] = (0.0, 1.0)

    # --- Classification head ---
    use_classification_head: bool = False
    num_classes_cls: int = 0
    cls_head_dropout: float = 0.0
    cls_head_num_heads: int = 4
    lambda_cls: float = 1.0

    # --- Segmentation head ---
    use_segmentation_head: bool = False
    num_classes_seg: int = 0
    seg_head_dropout: float = 0.0
    lambda_seg: float = 1.0

    # ------------------------------------------------------------------
    # Invariants
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
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

        # Stage lists shape-checks.
        n = len(self.stage_dims)
        if n < 1:
            raise ValueError("stage_dims must be non-empty.")
        if len(self.stage_depths) != n:
            raise ValueError(
                f"len(stage_depths) ({len(self.stage_depths)}) != "
                f"len(stage_dims) ({n})."
            )
        if len(self.stage_shifts) != n:
            raise ValueError(
                f"len(stage_shifts) ({len(self.stage_shifts)}) != "
                f"len(stage_dims) ({n})."
            )
        for i, d in enumerate(self.stage_dims):
            if d <= 0:
                raise ValueError(
                    f"stage_dims[{i}] must be positive, got {d}"
                )
        for i, k in enumerate(self.stage_depths):
            if k < 1:
                raise ValueError(
                    f"stage_depths[{i}] must be >= 1, got {k}"
                )
        for i, sh in enumerate(self.stage_shifts):
            if not isinstance(sh, (list, tuple)) or len(sh) < 1:
                raise ValueError(
                    f"stage_shifts[{i}] must be a non-empty list, got {sh!r}"
                )
            for s in sh:
                if not isinstance(s, int) or s <= 0:
                    raise ValueError(
                        f"stage_shifts[{i}] entries must be positive ints, "
                        f"got {s!r}"
                    )

        # Spatial divisibility: the patch grid is halved (n-1) times.
        hp = self.img_size // self.patch_size
        downsamples = n - 1
        if hp % (2 ** downsamples) != 0:
            raise ValueError(
                f"patch grid Hp={hp} (img_size={self.img_size}, "
                f"patch_size={self.patch_size}) must be divisible by "
                f"2**(num_stages-1)=2**{downsamples}={2 ** downsamples}."
            )

        valid_pools = {
            "avg", "max", "blur", "gaussian_dw", "pixel_unshuffle", "resnetd",
        }
        if self.downsample_kind not in valid_pools:
            raise ValueError(
                f"downsample_kind must be one of {sorted(valid_pools)}, "
                f"got {self.downsample_kind!r}"
            )
        if self.downsample_kernel_size <= 0:
            raise ValueError(
                "downsample_kernel_size must be positive, got "
                f"{self.downsample_kernel_size}"
            )
        if self.cli_mode not in ("inner", "wedge", "full"):
            raise ValueError(
                f"cli_mode must be one of {{'inner','wedge','full'}}, got "
                f"{self.cli_mode!r}"
            )
        if self.ctx_mode not in ("diff", "abs"):
            raise ValueError(
                f"ctx_mode must be one of {{'diff','abs'}}, got "
                f"{self.ctx_mode!r}"
            )
        if self.layer_scale_init <= 0.0:
            raise ValueError(
                f"layer_scale_init must be > 0, got {self.layer_scale_init}"
            )
        if not (0.0 <= self.block_drop_path_rate < 1.0):
            raise ValueError(
                f"block_drop_path_rate must be in [0.0, 1.0), got "
                f"{self.block_drop_path_rate}"
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
                f"recon_loss_type must be one of {{'mse','bce'}}, got "
                f"{self.recon_loss_type!r}"
            )
        if self.gamma_clip is not None and self.gamma_clip <= 0.0:
            raise ValueError(
                f"gamma_clip must be > 0.0 or None, got {self.gamma_clip}"
            )

        if not (0.0 <= self.mae_mask_ratio <= 0.95):
            raise ValueError(
                f"mae_mask_ratio must be in [0.0, 0.95], got "
                f"{self.mae_mask_ratio}"
            )
        if self.lambda_mae < 0.0:
            raise ValueError(f"lambda_mae must be >= 0.0, got {self.lambda_mae}")
        if self.lambda_lpips < 0.0:
            raise ValueError(
                f"lambda_lpips must be >= 0.0, got {self.lambda_lpips}"
            )
        if self.lpips_input_range[1] <= self.lpips_input_range[0]:
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
            if self.stage_dims[-1] % self.cls_head_num_heads != 0:
                raise ValueError(
                    f"cls_head_num_heads ({self.cls_head_num_heads}) must "
                    f"divide stage_dims[-1] ({self.stage_dims[-1]})."
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
    def num_stages(self) -> int:
        return len(self.stage_dims)

    @property
    def patches_per_side(self) -> int:
        return self.img_size // self.patch_size

    @property
    def bottleneck_per_side(self) -> int:
        """Side length of the encoder's lowest-resolution feature map."""
        return self.patches_per_side // (2 ** (self.num_stages - 1))

    @property
    def num_bottleneck_cells(self) -> int:
        return self.bottleneck_per_side ** 2

    @property
    def total_upsample_factor(self) -> int:
        """Factor required to go from bottleneck spatial resolution back
        to pixel resolution. Equals ``patch_size * 2**(num_stages-1)``."""
        return self.patch_size * (2 ** (self.num_stages - 1))

    @property
    def input_image_shape(self) -> Tuple[int, int, int]:
        return (self.img_size, self.img_size, self.img_channels)

    @property
    def embed_dim(self) -> int:
        """Convenience alias for the bottleneck feature width — head input."""
        return self.stage_dims[-1]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["lpips_input_range"] = list(self.lpips_input_range)
        # `asdict` already lists nested lists, but force a fresh copy to
        # avoid aliasing the dataclass's mutable defaults.
        d["stage_dims"] = list(self.stage_dims)
        d["stage_depths"] = list(self.stage_depths)
        d["stage_shifts"] = [list(s) for s in self.stage_shifts]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CliffordNetPatchVAEV2Config":
        import dataclasses as _dc

        valid = {f.name for f in _dc.fields(cls)}
        d = {k: v for k, v in dict(d).items() if k in valid}
        if "lpips_input_range" in d:
            d["lpips_input_range"] = tuple(d["lpips_input_range"])
        if "stage_shifts" in d:
            d["stage_shifts"] = [list(s) for s in d["stage_shifts"]]
        return cls(**d)


# ---------------------------------------------------------------------------
# Named presets.
#
# tiny / base target CIFAR-style 32x32 images: patch_size=4 -> Hp=8.
#   2 stages: 8 -> 4 bottleneck. Safe for SIGReg (16 cells > knots=17 fails;
#   the trainer logs a high-variance advisory when num_bottleneck_cells <
#   sigreg_knots).
#
# large / xl target 224x224 inputs: patch_size=4 -> Hp=56.
#   3 stages: 56 -> 28 -> 14. 196 bottleneck cells — comfortable for SIGReg.
# ---------------------------------------------------------------------------

PRESETS: Dict[str, Dict[str, Any]] = {
    "tiny": {
        "stage_dims": [64, 128],
        "stage_depths": [2, 2],
        "stage_shifts": [[1, 2], [1, 2, 4]],
        "latent_dim": 8,
    },
    "base": {
        "stage_dims": [96, 192],
        "stage_depths": [2, 4],
        "stage_shifts": [[1, 2], [1, 2, 4]],
        "latent_dim": 16,
    },
    "large": {
        "stage_dims": [128, 256, 384],
        "stage_depths": [2, 4, 6],
        "stage_shifts": [[1, 2], [1, 2, 4], [1, 2, 4, 8]],
        "latent_dim": 32,
    },
    "xl": {
        "stage_dims": [192, 384, 768],
        "stage_depths": [2, 4, 8],
        "stage_shifts": [[1, 2], [1, 2, 4], [1, 2, 4, 8]],
        "latent_dim": 64,
    },
}
