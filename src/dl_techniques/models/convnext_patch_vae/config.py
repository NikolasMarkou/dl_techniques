"""Configuration dataclass for the ConvNeXt patch-level VAE.

Mirrors the ``video_jepa/config.py`` shape: a frozen-ish dataclass with
``__post_init__`` invariants, derived properties, and ``to_dict /
from_dict`` for full Keras serialization round-trip.

Decisions anchored here (see ``plans/plan_2026-05-25_fb57d478/decisions.md``):

- D-001 package layout and per-patch 4D latent: ``patch_size``, ``latent_dim``.
- D-002 SIGReg binding on post-reparam ``(B, Hp*Wp, latent_dim)``:
  ``lambda_sigreg``, ``sigreg_knots``, ``sigreg_num_proj``.
"""

from __future__ import annotations
from typing import Any, Dict, Literal, Optional, Tuple
from dataclasses import asdict, dataclass, field

# ------------------------------------------------------------------

@dataclass
class ConvNeXtPatchVAEConfig:
    """Configuration for :class:`ConvNeXtPatchVAE`.

    Args:
        img_size: Default square input edge length in pixels. The model
            is resolution-agnostic at inference / fit time; this value
            drives the derived properties below and (when used) the
            dummy-input shape.
        img_channels: Number of pixel channels (e.g. 3 for RGB, 1 for
            MNIST).
        patch_size: Non-overlapping patch edge length. Must divide
            ``img_size``. Sets the spatial stride of the encoder stem
            and the decoder transposed-conv head.
        embed_dim: Internal ConvNeXt block channel width. Static across
            encoder and decoder.
        encoder_depth: Number of stacked ``ConvNextV2Block`` layers in
            the encoder, after the patchifying stem.
        decoder_depth: Number of stacked ``ConvNextV2Block`` layers in
            the decoder, before the transposed-conv head.
        kernel_size: Depthwise kernel size used inside each
            ``ConvNextV2Block``.
        latent_dim: Per-patch latent channel width. The encoder emits
            ``(B, Hp, Wp, 2 * latent_dim)`` and splits the last dim
            into ``(mu, log_var)``.
        beta_kl: Scalar weight on the per-patch KL term.
        lambda_sigreg: Scalar weight on the SIGReg term. Set ``0.0`` to
            disable SIGReg contribution to the gradient (the raw SIGReg
            statistic is still tracked for ablation comparison — see
            D-003 / test ``test_sigreg_off_branch``).
        sigreg_knots: Integration knots for :class:`SIGRegLayer`.
        sigreg_num_proj: Number of random projections for
            :class:`SIGRegLayer`. >= 256 recommended for stable
            per-patch estimate (F2).
        recon_loss_type: One of ``{"mse", "bce"}``. ``mse`` uses
            pixel-space MSE on the raw decoder output (suitable for
            RGB); ``bce`` uses binary-cross-entropy with logits and a
            sigmoid at sample time (suitable for binary / ``[0, 1]``
            -bounded inputs like MNIST). **When ``bce``, inputs MUST be
            in ``[0, 1]``.** No runtime assertion (graph-mode-unsafe);
            contract enforced by docstring.
        dropout_rate: Per-block dropout rate inside ``ConvNextV2Block``.
        spatial_dropout_rate: Per-block spatial-dropout rate inside
            ``ConvNextV2Block``.
        use_v2_block: Reserved for future use (V1 vs V2 GRN ablation —
            T4c in the training menu). Currently ``True`` is the only
            supported value at iter-1.
        gamma_clip: Optional symmetric gradient clip
            ``[-gamma_clip, +gamma_clip]`` applied in the custom
            ``train_step``. ``None`` disables clipping.
        kernel_regularizer_config: Optional Keras-serializable config
            dict for a ``keras.regularizers.Regularizer`` applied to
            all ``ConvNextV2Block`` conv kernels. ``None`` => no
            regularizer.

    Note:
        Invariants enforced in ``__post_init__``:

        - ``img_size % patch_size == 0`` (so the stem produces an
          integer patch grid).
        - ``latent_dim >= 1`` (a zero-dim latent degenerates the VAE
          to a deterministic AE; rejected per the plan's edge-case
          spec).
        - All positive-integer fields strictly positive.
        - ``recon_loss_type in {"mse", "bce"}``.
        - ``0.0 <= beta_kl`` and ``0.0 <= lambda_sigreg``.
        - ``0.0 <= dropout_rate <= 1.0`` and same for
          ``spatial_dropout_rate``.
        - ``sigreg_knots >= 2`` and ``sigreg_num_proj >= 1`` (mirrors
          :class:`SIGRegLayer` requirements).
    """

    # --- Vision / patches ---
    img_size: int = 32
    img_channels: int = 3
    patch_size: int = 4

    # --- ConvNeXt backbone ---
    embed_dim: int = 128
    encoder_depth: int = 4
    decoder_depth: int = 4
    kernel_size: int = 7

    # --- VAE latent ---
    latent_dim: int = 16

    # --- Reparameterization sampler ---
    # DECISION plan_2026-06-06_38aa045e/D-002: default "gaussian" keeps the
    # production path byte-identical; vmf is an opt-in per-patch spherical
    # posterior. Every vmf change downstream is gated behind
    # `sampling_type == "vmf"`. See decisions.md D-002.
    sampling_type: str = "gaussian"

    # --- Loss weights ---
    beta_kl: float = 0.5
    lambda_sigreg: float = 0.1

    # --- SIGReg ---
    sigreg_knots: int = 17
    sigreg_num_proj: int = 256

    # --- Reconstruction loss family ---
    recon_loss_type: str = "bce"

    # --- Regularization ---
    dropout_rate: float = 0.0
    spatial_dropout_rate: float = 0.0
    # Reserved for T4c ablation (V1 vs V2 block). Field is validated and
    # serialized but currently has no effect — encoder and decoder always
    # use ConvNextV2Block. Implementing the V1 path requires adding a
    # block-selection branch to encoder.py and decoder.py.
    use_v2_block: bool = True
    gamma_clip: Optional[float] = 1.0
    kernel_regularizer_config: Optional[Dict[str, Any]] = field(default=None)

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
        if self.embed_dim <= 0:
            raise ValueError(
                f"embed_dim must be positive, got {self.embed_dim}"
            )
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
            raise ValueError(
                f"latent_dim must be >= 1, got {self.latent_dim}. "
                "A zero-dimensional latent degenerates the VAE to a "
                "deterministic AE."
            )
        if self.beta_kl < 0.0:
            raise ValueError(
                f"beta_kl must be >= 0.0, got {self.beta_kl}"
            )
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
        # E2: sampler type whitelist.
        if self.sampling_type not in {"gaussian", "vmf"}:
            raise ValueError(
                f"sampling_type must be one of {{'gaussian', 'vmf'}}, got "
                f"{self.sampling_type!r}"
            )
        # E1 / I5: vMF is undefined on S^0 — require latent_dim >= 2 for vmf.
        if self.sampling_type == "vmf" and self.latent_dim < 2:
            raise ValueError(
                f"sampling_type='vmf' requires latent_dim >= 2 (vMF is "
                f"undefined on S^0), got latent_dim={self.latent_dim}."
            )

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------
    @property
    def patches_per_side(self) -> int:
        """``H_p = W_p = img_size // patch_size``."""
        return self.img_size // self.patch_size

    @property
    def num_patches(self) -> int:
        """``N = H_p * W_p`` — N for SIGReg under D-002 binding."""
        return self.patches_per_side ** 2

    @property
    def input_image_shape(self) -> Tuple[int, int, int]:
        """``(H, W, C)``."""
        return (self.img_size, self.img_size, self.img_channels)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Return config as a plain dict (JSON-safe)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ConvNeXtPatchVAEConfig":
        """Construct from a dict produced by :meth:`to_dict`.

        Drops unrecognized keys for forward-compat with future fields, and
        relies on dataclass defaults for missing keys (backward-compat).
        """
        import dataclasses as _dc

        valid = {f.name for f in _dc.fields(cls)}
        d = {k: v for k, v in dict(d).items() if k in valid}
        return cls(**d)

# ------------------------------------------------------------------

# DECISION plan_2026-06-08_e3917bd5/D-004: HierarchicalConvNeXtPatchVAEConfig
# COMPOSES (re-declares) the base fields it needs rather than subclassing
# ConvNeXtPatchVAEConfig. Do NOT make this a subclass — subclassing entangles
# the two `.keras` serialization formats and makes `from_dict` fragile when
# fields diverge. The duplicated field declarations are the deliberate price of
# serialization isolation. See decisions.md D-004.
@dataclass
class HierarchicalConvNeXtPatchVAEConfig:
    """Configuration for :class:`HierarchicalConvNeXtPatchVAE`.

    A two-level (fine ``z1`` / coarse ``z2``) hierarchical patch VAE built on a
    single fine :class:`ConvNeXtPatchEncoder`. The coarse latent ``z2`` is
    derived by ``AvgPool2D(pool_factor)`` over the encoder's last hidden
    features, and a learned top-down conditional prior ``p(z1|z2)`` couples the
    levels (VDVAE delta on the fine ``mu``). Free-bits gating on the coarse KL
    prevents ``z2`` collapse.

    This config **composes** the base single-scale fields rather than
    subclassing :class:`ConvNeXtPatchVAEConfig` (see ``D-004``): serialization
    isolation at the cost of some duplicated declarations.

    Gaussian-only (see ``D-003``): BOTH latents (``z1`` and ``z2``) are
    Gaussian, which is required for the closed-form Gaussian-Gaussian
    conditional KL ``KL(q(z1)||p(z1|z2))``. There is intentionally NO
    ``sampling_type`` field and NO vMF path. A vMF posterior at the fine level
    has no clean closed-form KL against a learned Gaussian conditional prior
    (and ``keras.random.beta`` lacks an XLA-GPU kernel in TF 2.18); vMF-at-fine
    is documented future work only.

    Args:
        img_size: Default square input edge length in pixels.
        img_channels: Number of pixel channels (3 RGB, 1 MNIST).
        patch_size: Non-overlapping patch edge length; must divide ``img_size``.
        embed_dim: Internal ConvNeXt block channel width (encoder + decoder).
        encoder_depth: Number of ``ConvNextV2Block`` layers in the encoder.
        decoder_depth: Number of ``ConvNextV2Block`` layers in the decoder.
        kernel_size: Depthwise kernel size inside each ``ConvNextV2Block``.
        latent_dim: Fine per-patch latent channel width ``D1`` (the ``z1`` dim).
        recon_loss_type: One of ``{"mse", "bce"}``. When ``bce``, inputs MUST be
            in ``[0, 1]`` (contract enforced by docstring, not runtime).
        dropout_rate: Per-block dropout rate inside ``ConvNextV2Block``.
        spatial_dropout_rate: Per-block spatial-dropout rate.
        gamma_clip: Optional symmetric gradient clip; ``None`` disables.
        kernel_regularizer_config: Optional Keras-serializable regularizer config
            dict applied to ``ConvNextV2Block`` conv kernels; ``None`` => none.
        sigreg_knots: Integration knots for :class:`SIGRegLayer`.
        sigreg_num_proj: Number of random projections for :class:`SIGRegLayer`.
        coarse_latent_dim: Coarse per-patch latent channel width ``D2`` (``z2``).
        prior_depth: Number ``M`` of ``ConvNextV2Block(k=3)`` blocks inside the
            learned conditional prior ``_L2ConditionalPrior``.
        prior_embed_dim: Channel width of the conditional prior trunk. ``0`` is a
            sentinel resolving to ``embed_dim`` (see
            :attr:`effective_prior_embed_dim`).
        pool_factor: ``AvgPool2D`` / ``UpSampling2D`` factor. The coarse grid
            side is ``patches_per_side // pool_factor``.
        free_bits: Nats-per-patch floor on the COARSE KL (anti-collapse gate).
        beta_kl_l1: Scalar weight on the fine KL ``KL(q(z1)||p(z1|z2))``.
        beta_kl_l2: Scalar weight on the coarse KL ``KL(q(z2)||N(0,I))``.
        lambda_sigreg_l1: Scalar weight on the fine-level SIGReg term.
        lambda_sigreg_l2: Scalar weight on the coarse-level SIGReg term.

    Note:
        Invariants enforced in ``__post_init__``:

        - ``img_size % patch_size == 0`` (integer patch grid).
        - ``(img_size // patch_size) % pool_factor == 0`` (integer coarse grid
          AND exact ``AvgPool2D(pool_factor)`` + ``UpSampling2D(pool_factor)``
          composition — A5; e.g. with ``pool_factor=2``, ``Hp`` must be even).
        - ``latent_dim >= 1`` and ``coarse_latent_dim >= 1``.
        - ``prior_depth >= 1``, ``prior_embed_dim >= 0``, ``pool_factor >= 2``.
        - ``free_bits >= 0``.
        - All loss weights ``>= 0``.
        - ``recon_loss_type in {"mse", "bce"}``.
        - ``0 <= dropout_rate <= 1`` and same for ``spatial_dropout_rate``.
        - ``gamma_clip is None or gamma_clip > 0``.
        - ``sigreg_knots >= 2`` and ``sigreg_num_proj >= 1``.
    """

    # --- Vision / patches ---
    img_size: int = 32
    img_channels: int = 3
    patch_size: int = 4

    # --- ConvNeXt backbone ---
    embed_dim: int = 128
    encoder_depth: int = 4
    decoder_depth: int = 4
    kernel_size: int = 7

    # --- VAE latents ---
    latent_dim: int = 16          # D1, the FINE latent dim (z1)
    coarse_latent_dim: int = 16   # D2, the COARSE latent dim (z2)

    # --- Reconstruction loss family ---
    recon_loss_type: str = "bce"

    # --- Regularization ---
    dropout_rate: float = 0.0
    spatial_dropout_rate: float = 0.0
    gamma_clip: Optional[float] = 1.0
    kernel_regularizer_config: Optional[Dict[str, Any]] = field(default=None)

    # --- SIGReg ---
    sigreg_knots: int = 17
    sigreg_num_proj: int = 256

    # --- Hierarchical: learned conditional prior p(z1|z2) ---
    prior_depth: int = 2          # M ConvNextV2Block(k=3) blocks in _L2ConditionalPrior
    prior_embed_dim: int = 0      # 0 sentinel -> resolve to embed_dim
    pool_factor: int = 2          # coarse grid side = patches_per_side // pool_factor

    # --- Hierarchical: loss weights ---
    # DECISION plan_2026-06-08_e3917bd5/D-006: free_bits is a nats/patch FLOOR
    # applied ONLY to the COARSE KL (z2) via `ops.maximum(KL_per_patch, free_bits)`
    # to prevent coarse-latent collapse. Do NOT apply free-bits to the fine
    # (conditional) KL — the learned conditional prior already gives z1 expressive
    # capacity; the collapse risk is on the unconditional coarse latent. The
    # default 0.25 is un-tuned (H12). See decisions.md D-006.
    free_bits: float = 0.25       # nats/patch floor on the COARSE KL
    beta_kl_l1: float = 0.5       # fine KL weight (q(z1)||p(z1|z2))
    beta_kl_l2: float = 0.5       # coarse KL weight (q(z2)||N(0,I))
    lambda_sigreg_l1: float = 0.1
    lambda_sigreg_l2: float = 0.1

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
        if self.pool_factor < 2:
            raise ValueError(
                f"pool_factor must be >= 2, got {self.pool_factor}"
            )
        # A5: integer coarse grid AND exact AvgPool/UpSampling composition.
        _hp = self.img_size // self.patch_size
        if _hp % self.pool_factor != 0:
            raise ValueError(
                f"patches_per_side ({_hp} = img_size {self.img_size} // "
                f"patch_size {self.patch_size}) must be divisible by pool_factor "
                f"({self.pool_factor}) so the coarse grid is integer and "
                f"AvgPool2D(pool_factor)+UpSampling2D(pool_factor) compose "
                f"exactly (e.g. pool_factor=2 requires even patches_per_side)."
            )
        if self.embed_dim <= 0:
            raise ValueError(
                f"embed_dim must be positive, got {self.embed_dim}"
            )
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
            raise ValueError(
                f"latent_dim must be >= 1, got {self.latent_dim}. "
                "A zero-dimensional latent degenerates the VAE to a "
                "deterministic AE."
            )
        if self.coarse_latent_dim < 1:
            raise ValueError(
                f"coarse_latent_dim must be >= 1, got {self.coarse_latent_dim}."
            )
        if self.prior_depth < 1:
            raise ValueError(
                f"prior_depth must be >= 1, got {self.prior_depth}"
            )
        if self.prior_embed_dim < 0:
            raise ValueError(
                f"prior_embed_dim must be >= 0 (0 => embed_dim), got "
                f"{self.prior_embed_dim}"
            )
        if self.free_bits < 0.0:
            raise ValueError(
                f"free_bits must be >= 0.0, got {self.free_bits}"
            )
        if self.beta_kl_l1 < 0.0:
            raise ValueError(
                f"beta_kl_l1 must be >= 0.0, got {self.beta_kl_l1}"
            )
        if self.beta_kl_l2 < 0.0:
            raise ValueError(
                f"beta_kl_l2 must be >= 0.0, got {self.beta_kl_l2}"
            )
        if self.lambda_sigreg_l1 < 0.0:
            raise ValueError(
                f"lambda_sigreg_l1 must be >= 0.0, got {self.lambda_sigreg_l1}"
            )
        if self.lambda_sigreg_l2 < 0.0:
            raise ValueError(
                f"lambda_sigreg_l2 must be >= 0.0, got {self.lambda_sigreg_l2}"
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

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------
    @property
    def patches_per_side(self) -> int:
        """Fine grid side ``H_p = W_p = img_size // patch_size``."""
        return self.img_size // self.patch_size

    @property
    def coarse_patches_per_side(self) -> int:
        """Coarse grid side ``patches_per_side // pool_factor``."""
        return self.patches_per_side // self.pool_factor

    @property
    def num_patches(self) -> int:
        """Fine ``N = H_p * W_p``."""
        return self.patches_per_side ** 2

    @property
    def input_image_shape(self) -> Tuple[int, int, int]:
        """``(H, W, C)``."""
        return (self.img_size, self.img_size, self.img_channels)

    @property
    def effective_prior_embed_dim(self) -> int:
        """Conditional-prior trunk width; ``embed_dim`` when sentinel ``0``."""
        return self.prior_embed_dim if self.prior_embed_dim > 0 else self.embed_dim

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Return config as a plain dict (JSON-safe)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HierarchicalConvNeXtPatchVAEConfig":
        """Construct from a dict produced by :meth:`to_dict`.

        Drops unrecognized keys for forward-compat with future fields, and
        relies on dataclass defaults for missing keys (backward-compat).
        """
        import dataclasses as _dc

        valid = {f.name for f in _dc.fields(cls)}
        d = {k: v for k, v in dict(d).items() if k in valid}
        return cls(**d)

# ------------------------------------------------------------------

