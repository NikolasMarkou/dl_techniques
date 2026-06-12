"""Configuration dataclass for the SD3 MMDiT Keras port.

Mirrors the Stable Diffusion 3 MMDiT transformer structural parameters
(diffusers ``SD3Transformer2DModel``) and pairs them with the reused 16-channel
KL-VAE parameters. The VAE parameter dataclass is **reused** from the ideogram4
port (:class:`~dl_techniques.models.ideogram4.config.AutoEncoderParams`) -- SD3
sets ``z_channels=16``; it is NOT redefined here (DRY; see D-002).

All invariants the downstream MMDiT blocks / transformer silently rely on are
enforced in :meth:`SD3MMDiTConfig.__post_init__` as ``ValueError`` guards:

1. ``embedding_size % num_heads == 0``          -> integer per-head ``head_dim``.
2. every ``i`` in ``dual_attention_layers`` satisfies ``0 <= i < depth``.
3. ``patch_size >= 1``.
4. ``pos_embed_max_size >= sample_size // patch_size`` -> the 2D sincos
   positional grid must cover the patchified token grid.
5. ``in_channels == out_channels`` -> SD3 rectified-flow velocity has the same
   channel count as the latent it denoises.

Two presets are provided: ``full`` (SD3-medium-ish defaults, defined-not-run)
and ``tiny`` (small enough to smoke-train on a 12GB GPU while satisfying ALL
invariants and exercising the dual-attention path in at least one block). Use
:func:`get_sd3_config` to retrieve a ``(SD3MMDiTConfig, AutoEncoderParams)``
pair by variant name.

This config is a structural (frozen) dataclass and is intentionally NOT a
keras-serializable object: it carries construction-time structural parameters,
not trainable state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple

from dl_techniques.utils.logger import logger

# Reuse the VAE parameter dataclass from ideogram4 -- do NOT redefine it (D-002).
from dl_techniques.models.ideogram4.config import AutoEncoderParams

# ---------------------------------------------------------------------
# Transformer (MMDiT) config
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class SD3MMDiTConfig:
    """Stable Diffusion 3 MMDiT dual-stream transformer configuration.

    Frozen so it can be safely shared/hashed; the tuple-valued
    ``dual_attention_layers`` round-trips to/from a list via
    :meth:`to_dict` / :meth:`from_dict`.

    Args:
        patch_size: Latent patchification edge (latent grid -> token grid).
        in_channels: VAE latent channel count fed to the patch embedder
            (= ``AutoEncoderParams.z_channels``; SD3 uses 16).
        out_channels: Velocity output channel count. Must equal ``in_channels``.
        embedding_size: Transformer hidden width (``dim``). Must be divisible by
            ``num_heads``.
        num_heads: Attention head count.
        depth: Number of stacked :class:`MMDiTBlock` blocks.
        mlp_ratio: FFN hidden expansion ratio (hidden = ``embedding_size *
            mlp_ratio``).
        joint_attention_dim: Caption / ``encoder_hidden_states`` feature width;
            the ``context_embedder`` Dense projects this to ``embedding_size``.
        pooled_projection_dim: Pooled text-vector width fed to the combined
            timestep-text embedding.
        pos_embed_max_size: Maximum grid side for the 2D sincos positional
            embedding. Must cover ``sample_size // patch_size``.
        sample_size: Default latent grid side (``H' == W'``).
        dual_attention_layers: Indices of blocks that use the dual
            (9-way ``AdaLayerNormZeroX``) self-attention path. Each index must be
            in ``[0, depth)``.
        qk_norm: Whether the joint attention applies per-head QK-RMSNorm.
        eps: RMSNorm / LayerNorm epsilon.
    """

    patch_size: int = 2
    in_channels: int = 16
    out_channels: int = 16
    embedding_size: int = 1536
    num_heads: int = 24
    depth: int = 24
    mlp_ratio: float = 4.0
    joint_attention_dim: int = 4096
    pooled_projection_dim: int = 2048
    pos_embed_max_size: int = 192
    sample_size: int = 64
    dual_attention_layers: Tuple[int, ...] = ()
    qk_norm: bool = True
    eps: float = 1e-6

    # --- derived ----------------------------------------------------

    @property
    def head_dim(self) -> int:
        """Per-head dimensionality (``embedding_size // num_heads``)."""
        return self.embedding_size // self.num_heads

    # --- validation -------------------------------------------------

    def __post_init__(self) -> None:
        # Invariant 1: integer head_dim.
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        if self.embedding_size % self.num_heads != 0:
            raise ValueError(
                f"embedding_size ({self.embedding_size}) must be divisible by "
                f"num_heads ({self.num_heads}) for an integer head_dim."
            )

        # Invariant 3: patch_size >= 1.
        if self.patch_size < 1:
            raise ValueError(f"patch_size must be >= 1, got {self.patch_size}")

        if self.depth <= 0:
            raise ValueError(f"depth must be positive, got {self.depth}")

        # Invariant 2: every dual-attention layer index is a valid block index.
        dual = tuple(int(i) for i in self.dual_attention_layers)
        for i in dual:
            if not (0 <= i < self.depth):
                raise ValueError(
                    f"dual_attention_layers index {i} out of range; must be in "
                    f"[0, depth={self.depth})."
                )

        # Invariant 4: positional grid must cover the patchified token grid.
        token_grid = self.sample_size // self.patch_size
        if self.pos_embed_max_size < token_grid:
            raise ValueError(
                f"pos_embed_max_size ({self.pos_embed_max_size}) must be >= "
                f"sample_size // patch_size = {self.sample_size} // "
                f"{self.patch_size} = {token_grid}; the 2D sincos positional "
                f"grid must cover the patch token grid."
            )

        # Invariant 5: SD3 velocity has the same channel count as the latent.
        if self.in_channels != self.out_channels:
            raise ValueError(
                f"in_channels ({self.in_channels}) must equal out_channels "
                f"({self.out_channels}); SD3 rectified-flow velocity matches the "
                f"latent channel count."
            )

    # --- serialization ----------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict (tuple ``dual_attention_layers`` -> list)."""
        d = asdict(self)
        d["dual_attention_layers"] = list(self.dual_attention_layers)
        return d

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "SD3MMDiTConfig":
        """Reconstruct from :meth:`to_dict` output (list -> tuple coercion)."""
        config = dict(config)
        if "dual_attention_layers" in config:
            config["dual_attention_layers"] = tuple(
                config["dual_attention_layers"]
            )
        return cls(**config)


# ---------------------------------------------------------------------
# VAE channel/32 divisibility check (GroupNorm32)
# ---------------------------------------------------------------------


def _validate_vae_groupnorm(ae: AutoEncoderParams) -> None:
    """Assert every GroupNorm(32) channel count is divisible by 32.

    The base ``ch`` and every stage channel ``ch * m`` for ``m in ch_mult`` feed
    ``keras.layers.GroupNormalization(groups=32)`` in the reused VAE; each must
    be divisible by 32 or the layer raises at build time. ``z_channels`` is the
    latent count and is exempt (it does not feed a GroupNorm).

    Args:
        ae: The AutoEncoder parameters to validate.

    Raises:
        ValueError: If ``ch`` or any ``ch * m`` is not divisible by 32.
    """
    if ae.ch % 32 != 0:
        raise ValueError(
            f"AutoEncoder base ch ({ae.ch}) must be divisible by 32 "
            f"(GroupNormalization groups=32)."
        )
    for m in ae.ch_mult:
        stage = ae.ch * m
        if stage % 32 != 0:
            raise ValueError(
                f"AutoEncoder stage channel ch * {m} = {stage} must be divisible "
                f"by 32 (GroupNormalization groups=32)."
            )


# ---------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------

PRESETS: Dict[str, Dict[str, Any]] = {
    # The SD3-medium-ish full-scale model. Defined but NOT run locally.
    # head_dim = 1536 // 24 = 64. token_grid = 64 // 2 = 32 <= 192 OK.
    # dual_attention_layers = range(13) (SD3.5-medium), all < depth=24 OK.
    # VAE: ch=128, ch_mult=(1,2,4,4) -> stages 128,256,512,512 all /32 OK.
    "full": {
        "config": dict(
            patch_size=2,
            in_channels=16,
            out_channels=16,
            embedding_size=1536,
            num_heads=24,
            depth=24,
            mlp_ratio=4.0,
            joint_attention_dim=4096,
            pooled_projection_dim=2048,
            pos_embed_max_size=192,
            sample_size=64,
            dual_attention_layers=tuple(range(13)),
            qk_norm=True,
            eps=1e-6,
        ),
        "ae": dict(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=(1, 2, 4, 4),
            num_res_blocks=2,
            z_channels=16,
        ),
    },
    # Tiny preset: smoke-trainable on a 12GB GPU; satisfies ALL invariants and
    # exercises the dual-attention path in block 0.
    # head_dim = 192 // 6 = 32. token_grid = 16 // 2 = 8 <= 32 OK.
    # dual_attention_layers = (0,), 0 < depth=4 OK.
    # VAE: ch=32, ch_mult=(1,2) -> stages 32, 64 both /32 OK. z=16 (exempt).
    "tiny": {
        "config": dict(
            patch_size=2,
            in_channels=16,
            out_channels=16,
            embedding_size=192,
            num_heads=6,
            depth=4,
            mlp_ratio=4.0,
            joint_attention_dim=512,
            pooled_projection_dim=256,
            pos_embed_max_size=32,
            sample_size=16,
            dual_attention_layers=(0,),
            qk_norm=True,
            eps=1e-6,
        ),
        "ae": dict(
            resolution=32,
            in_channels=3,
            ch=32,
            out_ch=3,
            ch_mult=(1, 2),
            num_res_blocks=1,
            z_channels=16,
        ),
    },
}


def get_sd3_config(
    variant: str = "tiny",
) -> Tuple[SD3MMDiTConfig, AutoEncoderParams]:
    """Return the ``(SD3MMDiTConfig, AutoEncoderParams)`` pair for a variant.

    Both objects are constructed (so all ``__post_init__`` invariants run) and
    the VAE GroupNorm-divisibility invariant is asserted. The transformer
    ``in_channels`` is cross-checked against the AutoEncoder ``z_channels`` (the
    patch embedder consumes the raw 16-channel latent directly -- SD3 does NOT
    patchify channels into ``in_channels`` the way ideogram4 does).

    Args:
        variant: One of the keys in :data:`PRESETS` (``"tiny"`` or ``"full"``).

    Returns:
        A ``(config, ae)`` tuple.

    Raises:
        ValueError: If ``variant`` is unknown, any config invariant fails, the
            VAE GroupNorm divisibility fails, or ``in_channels`` disagrees with
            the AutoEncoder ``z_channels``.
    """
    if variant not in PRESETS:
        raise ValueError(
            f"Unknown variant '{variant}'. Available: {sorted(PRESETS)}"
        )
    preset = PRESETS[variant]
    config = SD3MMDiTConfig(**preset["config"])
    ae = AutoEncoderParams(**preset["ae"])

    # VAE GroupNorm divisibility.
    _validate_vae_groupnorm(ae)

    # Cross-check the latent channel linkage against the paired AutoEncoder.
    if config.in_channels != ae.z_channels:
        raise ValueError(
            f"config.in_channels ({config.in_channels}) must match "
            f"ae.z_channels ({ae.z_channels}) for variant '{variant}'."
        )

    logger.info(
        "Built SD3 MMDiT '%s' config: embedding_size=%d, head_dim=%d, depth=%d, "
        "in_channels=%d, joint_attention_dim=%d, pooled_projection_dim=%d, "
        "dual_attention_layers=%s, VAE ch=%d, ch_mult=%s, z_channels=%d",
        variant,
        config.embedding_size,
        config.head_dim,
        config.depth,
        config.in_channels,
        config.joint_attention_dim,
        config.pooled_projection_dim,
        config.dual_attention_layers,
        ae.ch,
        ae.ch_mult,
        ae.z_channels,
    )
    return config, ae
