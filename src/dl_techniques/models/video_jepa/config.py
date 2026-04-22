"""Video-JEPA-Clifford configuration dataclass.

Defaults chosen for RTX 4070 (12 GB) smoke training on synthetic drone
footage. Every field is overridable via :meth:`from_dict`.

Decisions anchored here:
- D-001 hybrid encoder: ``encoder_clifford_depth`` + ``encoder_shifts``.
- D-002 factorized predictor: ``predictor_depth`` (pairs of spatial+temporal).
- D-005 middle SIGReg placement: ``sigreg_*`` fields + weight.
- D-006 positional embeddings: ``patch_size``, ``embed_dim``.
- D-007 streaming: ``history_size_k``.
- D-013 (iter-3, 2026-04-22) telemetry conditioning removed: the
  predictor is now a pixels-only architecture with a plain causal
  self-attn + MLP temporal block. See plans/plan_2026-04-22_4f29c76f.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple


@dataclass
class VideoJEPAConfig:
    """Configuration for the Video-JEPA-Clifford model.

    :param img_size: Square input edge length in pixels.
    :param img_channels: Number of pixel channels.
    :param patch_size: Non-overlapping patch edge length. Must divide ``img_size``.
    :param embed_dim: Patch embedding dimension ``D``.
    :param num_frames: Default training window length ``T``.
    :param history_size_k: Streaming rolling-buffer window length ``K``
        (D-007). Typically ``K = num_frames``.
    :param encoder_clifford_depth: Number of stacked ``CliffordNetBlock``
        layers after patch embed (D-001). 2–4 recommended.
    :param encoder_shifts: Channel-shift offsets fed to the encoder
        Clifford blocks.
    :param predictor_depth: Number of *pairs* of (spatial-Clifford, temporal-
        Clifford) blocks (D-002).
    :param predictor_num_heads: Number of attention heads in the
        temporal causal self-attention block.
    :param predictor_dim_head: Per-head dimension for the temporal MHA.
    :param predictor_mlp_dim: Hidden dimension of the temporal MLP block.
    :param predictor_shifts: Channel-shift offsets for predictor Clifford
        blocks (spatial + causal-temporal).
    :param sigreg_knots: Integration knots for :class:`SIGRegLayer` (D-005).
    :param sigreg_num_proj: Number of random projections for
        :class:`SIGRegLayer`. Smoke default 64 (LeWM precedent).
    :param sigreg_weight: Weight applied to SIGReg loss when added via
        ``add_loss`` (D-005).
    :param dropout: Dropout rate used inside the temporal attention block.
    :param mask_prediction_enabled: If True, V-JEPA-style tube-masked latent
        prediction runs alongside next-frame prediction (iter-2, D-008/D-009).
        When False, the model degrades to iter-1-equivalent two-loss behavior
        (next-frame MSE + SIGReg). Regression-guard flag.
    :param mask_ratio: Fraction of spatial patch positions masked per sample
        in the tube mask (iter-2, D-011). ``0.0 <= mask_ratio < 1.0`` strict.
        0.5–0.75 typical for V-JEPA; 0.6 default.
    :param lambda_next_frame: Scalar weight applied to the next-frame
        prediction loss via ``add_loss`` (iter-2, D-012).
    :param lambda_mask: Scalar weight applied to the mask-prediction loss
        via ``add_loss`` (iter-2, D-012).

    .. note::
       Invariant: ``img_size % patch_size == 0``. Invariant:
       ``0.0 <= mask_ratio < 1.0`` (strict upper bound; a mask
       ratio of 1.0 would leave the next-frame loss undefined).
       Invariant: ``lambda_next_frame >= 0.0`` and ``lambda_mask >= 0.0``.
    """

    # --- Vision / patches ---
    img_size: int = 64
    img_channels: int = 3
    patch_size: int = 8
    embed_dim: int = 64

    # --- Temporal window ---
    num_frames: int = 4
    history_size_k: int = 4

    # --- Encoder (hybrid: PatchEmbedding2D + stacked CliffordNetBlock) ---
    encoder_clifford_depth: int = 2
    encoder_shifts: Tuple[int, ...] = (1, 2)

    # --- Predictor (factorized spatial + causal-temporal) ---
    predictor_depth: int = 2
    predictor_num_heads: int = 4
    predictor_dim_head: int = 16
    predictor_mlp_dim: int = 128
    predictor_shifts: Tuple[int, ...] = (1, 2)

    # --- SIGReg ---
    sigreg_knots: int = 17
    sigreg_num_proj: int = 64
    sigreg_weight: float = 0.09

    # --- Dropout / regularization ---
    dropout: float = 0.0

    # --- Iter-2: V-JEPA-style tube-masked latent prediction (D-008..D-012) ---
    mask_prediction_enabled: bool = True
    mask_ratio: float = 0.6
    lambda_next_frame: float = 1.0
    lambda_mask: float = 1.0

    # ------------------------------------------------------------------
    # Invariants
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.img_size % self.patch_size != 0:
            raise ValueError(
                f"img_size ({self.img_size}) must be divisible by patch_size "
                f"({self.patch_size})."
            )
        if self.history_size_k <= 0:
            raise ValueError(
                f"history_size_k must be positive, got {self.history_size_k}"
            )
        if self.num_frames <= 0:
            raise ValueError(
                f"num_frames must be positive, got {self.num_frames}"
            )
        if self.encoder_clifford_depth < 1:
            raise ValueError(
                f"encoder_clifford_depth must be >= 1, got "
                f"{self.encoder_clifford_depth}"
            )
        if self.predictor_depth < 1:
            raise ValueError(
                f"predictor_depth must be >= 1, got {self.predictor_depth}"
            )
        # --- Iter-2 invariants (D-008..D-012) ---
        if not (0.0 <= self.mask_ratio < 1.0):
            raise ValueError(
                f"mask_ratio must be in [0.0, 1.0), got {self.mask_ratio}. "
                "Upper bound is strict: a mask ratio of 1.0 leaves no "
                "unmasked positions for the next-frame loss."
            )
        if self.lambda_next_frame < 0.0:
            raise ValueError(
                f"lambda_next_frame must be >= 0.0, got "
                f"{self.lambda_next_frame}"
            )
        if self.lambda_mask < 0.0:
            raise ValueError(
                f"lambda_mask must be >= 0.0, got {self.lambda_mask}"
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
        """``N = H_p * W_p``."""
        return self.patches_per_side ** 2

    @property
    def input_image_shape(self) -> Tuple[int, int, int]:
        """``(H, W, C)``."""
        return (self.img_size, self.img_size, self.img_channels)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Return config as a plain dict (tuples stay tuples via asdict)."""
        d = asdict(self)
        # Normalize tuple-typed fields to list for JSON-safety.
        for k in ("encoder_shifts", "predictor_shifts"):
            if isinstance(d.get(k), tuple):
                d[k] = list(d[k])
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VideoJEPAConfig":
        """Construct a config from a dict produced by :meth:`to_dict`."""
        d = dict(d)
        # Tolerate legacy keys dropped in iter-3 (telemetry removal).
        for legacy_key in ("cond_dim", "telemetry_dim"):
            d.pop(legacy_key, None)
        for k in ("encoder_shifts", "predictor_shifts"):
            if k in d and isinstance(d[k], list):
                d[k] = tuple(d[k])
        return cls(**d)
