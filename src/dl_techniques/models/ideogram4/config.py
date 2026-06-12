"""Configuration dataclasses for the Ideogram4 Keras port.

Mirrors the PyTorch ``Ideogram4Config`` (``modeling_ideogram4.py``) and
``AutoEncoderParams`` (``autoencoder.py``), plus the pipeline-level fields
(``patch_size``, ``ae_scale_factor``, ``max_text_tokens``). All invariants that
the downstream layers silently rely on are enforced in ``__post_init__`` as
``ValueError`` guards:

1. ``emb_dim % num_heads == 0``           -> integer ``head_dim``.
2. ``head_dim`` even                      -> mRoPE needs ``head_dim/2`` freqs.
3. mRoPE band bound                       -> h/w bands fit inside ``head_dim/2``
   (the EXACT check from ``Ideogram4MRoPE`` so config and layer agree).
4. VAE channel / 32 divisibility          -> ``GroupNormalization(groups=32)``.
5. ``in_channels == z_channels * patch_size**2`` cross-consistency.

Two presets are provided: ``full`` (the real model, defined-not-run) and
``tiny`` (small enough to smoke-train on a 12GB GPU while satisfying ALL
invariants). Use :func:`get_ideogram4_config` to retrieve a ``(config, ae)``
pair by variant name.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Tuple

import numpy as np

from dl_techniques.utils.logger import logger
from dl_techniques.models.ideogram4.constants import QWEN3_VL_ACTIVATION_LAYERS

# ---------------------------------------------------------------------
# AutoEncoder parameters (Flux2 KL-VAE)
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class AutoEncoderParams:
    """Flux2 KL-VAE structural parameters.

    Mirrors the PyTorch ``AutoEncoderParams`` dataclass. Frozen so it can be
    safely shared/hashed; tuple-valued ``ch_mult`` round-trips to/from a list.

    Args:
        resolution: Square input edge length in pixels.
        in_channels: Pixel channels of the input image (RGB = 3).
        ch: Base channel width at the highest resolution.
        out_ch: Output (reconstructed) pixel channels.
        ch_mult: Per-stage channel multipliers over ``ch``.
        num_res_blocks: ResnetBlocks per resolution stage.
        z_channels: Latent channel count (pre-patchification).
    """

    resolution: int = 256
    in_channels: int = 3
    ch: int = 128
    out_ch: int = 3
    ch_mult: Tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    z_channels: int = 32

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict (tuple ``ch_mult`` -> list)."""
        d = asdict(self)
        d["ch_mult"] = list(self.ch_mult)
        return d

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "AutoEncoderParams":
        """Reconstruct from :meth:`to_dict` output (list ``ch_mult`` -> tuple)."""
        config = dict(config)
        if "ch_mult" in config:
            config["ch_mult"] = tuple(config["ch_mult"])
        return cls(**config)


# ---------------------------------------------------------------------
# Transformer (DiT) config
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class Ideogram4Config:
    """Ideogram4 flow-matching DiT configuration.

    Mirrors the PyTorch ``Ideogram4Config`` and carries the extra
    pipeline-level fields (``patch_size``, ``ae_scale_factor``,
    ``max_text_tokens``) plus ``z_channels`` so the
    ``in_channels == z_channels * patch_size**2`` linkage is explicit and
    checked. Frozen; tuple-valued ``mrope_section`` round-trips to/from a list.

    Args:
        emb_dim: Transformer hidden width. Must be divisible by ``num_heads``.
        num_layers: Number of DiT blocks.
        num_heads: Attention head count.
        intermediate_size: SwiGLU FFN hidden width.
        adanln_dim: AdaLN conditioning embedding width.
        in_channels: Patchified latent channels (= ``z_channels * patch_size**2``).
        llm_features_dim: Precomputed conditioning feature width.
        rope_theta: mRoPE base frequency.
        mrope_section: 3-tuple ``(t_band, h_band, w_band)`` for mRoPE interleave.
        norm_eps: RMSNorm / LayerNorm epsilon.
        patch_size: Latent patchification edge (latent -> token).
        ae_scale_factor: VAE spatial downsampling factor (pixels -> latent).
        max_text_tokens: Maximum conditioning text tokens.
        z_channels: VAE latent channels (must match the paired AutoEncoderParams).
    """

    emb_dim: int = 4608
    num_layers: int = 34
    num_heads: int = 18
    intermediate_size: int = 12288
    adanln_dim: int = 512

    # Latent dimension after patchification: z_channels (32) * patch_size**2 (4) = 128.
    in_channels: int = 128

    # Qwen3-VL hidden size (4096) * number of extracted layers (13) = 53248.
    llm_features_dim: int = 4096 * len(QWEN3_VL_ACTIVATION_LAYERS)

    rope_theta: int = 5_000_000
    mrope_section: Tuple[int, ...] = (24, 20, 20)

    norm_eps: float = 1e-5

    # Pipeline-level fields.
    patch_size: int = 2
    ae_scale_factor: int = 8
    max_text_tokens: int = 2048

    # VAE latent channels (cross-checked against in_channels / patch_size).
    z_channels: int = 32

    # --- derived ----------------------------------------------------

    @property
    def head_dim(self) -> int:
        """Per-head dimensionality (``emb_dim // num_heads``)."""
        return self.emb_dim // self.num_heads

    # --- validation -------------------------------------------------

    def __post_init__(self) -> None:
        # Invariant 1: integer head_dim.
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        if self.emb_dim % self.num_heads != 0:
            raise ValueError(
                f"emb_dim ({self.emb_dim}) must be divisible by num_heads "
                f"({self.num_heads}) for an integer head_dim."
            )
        head_dim = self.emb_dim // self.num_heads

        # Invariant 2: head_dim even (mRoPE needs head_dim/2 frequencies).
        if head_dim % 2 != 0:
            raise ValueError(
                f"head_dim (emb_dim // num_heads = {head_dim}) must be even; "
                f"mRoPE requires head_dim/2 rotary frequencies."
            )
        half = head_dim // 2

        # Invariant 3: mRoPE band bound. Reuse the EXACT check from
        # Ideogram4MRoPE.__init__ so config and layer agree: the h band consumes
        # slots arange(1, h*3, 3), the w band arange(2, w*3, 3); the largest
        # consumed slot must stay strictly below head_dim/2.
        mrope_section = tuple(int(s) for s in self.mrope_section)
        if len(mrope_section) != 3:
            raise ValueError(
                f"mrope_section must have length 3 (t, h, w), got {mrope_section}"
            )
        if any(s <= 0 for s in mrope_section):
            raise ValueError(
                f"mrope_section entries must be positive, got {mrope_section}"
            )
        for axis, offset, name in ((1, 1, "h"), (2, 2, "w")):
            consumed = np.arange(offset, mrope_section[axis] * 3, 3)
            if consumed.size and consumed.max() >= half:
                raise ValueError(
                    f"mrope_section[{axis}] ({name} band = {mrope_section[axis]}) "
                    f"reaches frequency slot {int(consumed.max())} which exceeds "
                    f"head_dim/2 - 1 = {half - 1}. Reduce the {name} band."
                )

        # Invariant 5: latent/patch cross-consistency.
        expected_in = self.z_channels * self.patch_size ** 2
        if self.in_channels != expected_in:
            raise ValueError(
                f"in_channels ({self.in_channels}) must equal "
                f"z_channels * patch_size**2 = {self.z_channels} * "
                f"{self.patch_size}**2 = {expected_in}."
            )

    # --- serialization ----------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict (tuple ``mrope_section`` -> list)."""
        d = asdict(self)
        d["mrope_section"] = list(self.mrope_section)
        return d

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "Ideogram4Config":
        """Reconstruct from :meth:`to_dict` output (list ``mrope_section`` -> tuple)."""
        config = dict(config)
        if "mrope_section" in config:
            config["mrope_section"] = tuple(config["mrope_section"])
        return cls(**config)


# ---------------------------------------------------------------------
# VAE channel/32 divisibility check (Invariant 4)
# ---------------------------------------------------------------------


def _validate_vae_groupnorm(ae: AutoEncoderParams) -> None:
    """Assert every GroupNorm(32) channel count is divisible by 32.

    The base ``ch`` and every stage channel ``ch * m`` for ``m in ch_mult`` feed
    ``keras.layers.GroupNormalization(groups=32)`` in the VAE; each must be
    divisible by 32 or the layer raises at build time.

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
    # The real, full-scale model. Defined but NOT run locally.
    "full": {
        "config": dict(
            emb_dim=4608,
            num_layers=34,
            num_heads=18,
            intermediate_size=12288,
            adanln_dim=512,
            in_channels=128,
            llm_features_dim=4096 * len(QWEN3_VL_ACTIVATION_LAYERS),
            rope_theta=5_000_000,
            mrope_section=(24, 20, 20),
            norm_eps=1e-5,
            patch_size=2,
            ae_scale_factor=8,
            max_text_tokens=2048,
            z_channels=32,
        ),
        "ae": dict(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=(1, 2, 4, 4),
            num_res_blocks=2,
            z_channels=32,
        ),
    },
    # Tiny preset: smoke-trainable on a 12GB GPU; satisfies ALL invariants.
    # head_dim = 128 // 4 = 32 (even); half = 16. mrope max(h,w)=3 -> slot
    # arange(2, 9, 3).max() = 8 < 16 OK. in_channels = z(8) * patch(2)^2 = 32.
    # VAE: ch=32, ch_mult=(1,2) -> stages 32, 64 both /32 OK.
    "tiny": {
        "config": dict(
            emb_dim=128,
            num_layers=2,
            num_heads=4,
            intermediate_size=256,
            adanln_dim=64,
            in_channels=32,
            llm_features_dim=64,
            rope_theta=10000,
            mrope_section=(4, 3, 3),
            norm_eps=1e-5,
            patch_size=2,
            ae_scale_factor=4,
            max_text_tokens=64,
            z_channels=8,
        ),
        "ae": dict(
            resolution=32,
            in_channels=3,
            ch=32,
            out_ch=3,
            ch_mult=(1, 2),
            num_res_blocks=1,
            z_channels=8,
        ),
    },
}


def get_ideogram4_config(
    variant: str = "tiny",
) -> Tuple[Ideogram4Config, AutoEncoderParams]:
    """Return the ``(Ideogram4Config, AutoEncoderParams)`` pair for a variant.

    Both objects are constructed (so all ``__post_init__`` invariants run) and
    the VAE GroupNorm-divisibility invariant is asserted. The transformer
    ``z_channels`` is cross-checked against the AutoEncoder ``z_channels``.

    Args:
        variant: One of the keys in :data:`PRESETS` (``"tiny"`` or ``"full"``).

    Returns:
        A ``(config, ae)`` tuple.

    Raises:
        ValueError: If ``variant`` is unknown, any config invariant fails, the
            VAE GroupNorm divisibility fails, or the two ``z_channels`` disagree.
    """
    if variant not in PRESETS:
        raise ValueError(
            f"Unknown variant '{variant}'. Available: {sorted(PRESETS)}"
        )
    preset = PRESETS[variant]
    config = Ideogram4Config(**preset["config"])
    ae = AutoEncoderParams(**preset["ae"])

    # Invariant 4: VAE GroupNorm divisibility.
    _validate_vae_groupnorm(ae)

    # Cross-check the latent linkage against the paired AutoEncoder.
    if config.z_channels != ae.z_channels:
        raise ValueError(
            f"config.z_channels ({config.z_channels}) must match "
            f"ae.z_channels ({ae.z_channels}) for variant '{variant}'."
        )

    logger.info(
        "Built Ideogram4 '%s' config: emb_dim=%d, head_dim=%d, num_layers=%d, "
        "in_channels=%d (z=%d * patch=%d^2), VAE ch=%d, ch_mult=%s",
        variant,
        config.emb_dim,
        config.head_dim,
        config.num_layers,
        config.in_channels,
        config.z_channels,
        config.patch_size,
        ae.ch,
        ae.ch_mult,
    )
    return config, ae
