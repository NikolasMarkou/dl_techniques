"""Video-JEPA-Clifford configuration dataclass.

`VideoJEPAConfig` holds every construction and training-loss parameter for
the Video-JEPA-Clifford model: patch/embedding sizes, encoder and predictor
depths and channel shifts, SIGReg regularization, tube-masking, multi-horizon
prediction weights, and the EMA target-encoder schedule. Every field is
overridable via :meth:`from_dict`, and defaults are tuned for a 12 GB GPU
smoke run on synthetic drone footage.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple


@dataclass
class VideoJEPAConfig:
    """Configuration for the Video-JEPA-Clifford model.

    :param img_size: Square input edge length in pixels.
    :type img_size: int
    :param img_channels: Number of pixel channels.
    :type img_channels: int
    :param patch_size: Non-overlapping patch edge length. Must divide `img_size`.
    :type patch_size: int
    :param embed_dim: Patch embedding dimension `D`.
    :type embed_dim: int
    :param num_frames: Default training window length `T`.
    :type num_frames: int
    :param history_size_k: Streaming rolling-buffer window length `K`.
        Typically equal to `num_frames`.
    :type history_size_k: int
    :param encoder_clifford_depth: Number of stacked `CliffordNetBlock`
        layers after patch embed. 2-4 recommended.
    :type encoder_clifford_depth: int
    :param encoder_shifts: Channel-shift offsets fed to the encoder Clifford blocks.
    :type encoder_shifts: Tuple[int, ...]
    :param predictor_depth: Number of pairs of (spatial-Clifford,
        temporal-Clifford) blocks.
    :type predictor_depth: int
    :param predictor_num_heads: Attention heads in the temporal causal
        self-attention block.
    :type predictor_num_heads: int
    :param predictor_dim_head: Per-head dimension for the temporal MHA.
    :type predictor_dim_head: int
    :param predictor_mlp_dim: Hidden dimension of the temporal MLP block.
    :type predictor_mlp_dim: int
    :param predictor_shifts: Channel-shift offsets for predictor Clifford
        blocks, spatial and causal-temporal.
    :type predictor_shifts: Tuple[int, ...]
    :param sigreg_knots: Integration knots for `SIGRegLayer`.
    :type sigreg_knots: int
    :param sigreg_num_proj: Number of random projections for `SIGRegLayer`.
    :type sigreg_num_proj: int
    :param sigreg_weight: Weight applied to the SIGReg loss via `add_loss`.
    :type sigreg_weight: float
    :param dropout_rate: Dropout rate inside the temporal attention block.
    :type dropout_rate: float
    :param mask_prediction_enabled: If `True`, V-JEPA-style tube-masked
        latent prediction runs alongside next-frame prediction. If `False`,
        only next-frame MSE and SIGReg are used.
    :type mask_prediction_enabled: bool
    :param mask_ratio: Fraction of spatial patch positions masked per sample
        in the tube mask. Strict range `[0.0, 1.0)`; 0.5-0.75 is typical.
    :type mask_ratio: float
    :param lambda_next_frame: Weight applied to the next-frame prediction
        loss via `add_loss`.
    :type lambda_next_frame: float
    :param lambda_mask: Weight applied to the mask-prediction loss via `add_loss`.
    :type lambda_mask: float
    :param predict_horizons: Strictly positive prediction horizons `(h1, h2,
        ...)` in frames. For each `h`, the model learns
        `MSE(pred_head_h(pred[:, :-h]), z[:, h:])`. Must be sorted ascending,
        unique, all positive, and `max(predict_horizons) < num_frames`.
        Default `(1,)` is the single-horizon t+1 case.
    :type predict_horizons: Tuple[int, ...]

    :param ema_momentum: EMA momentum for the target encoder update:
        `target_w <- m * target_w + (1 - m) * encoder_w`. `0.996` is the
        V-JEPA/BYOL default; `0.0` snaps the target to the encoder every
        step. Strict range `[0.0, 1.0)`.
    :type ema_momentum: float
    :param ema_schedule: One of `"none"` or `"cosine"`. `"cosine"` ramps
        momentum from `ema_momentum` to `1.0` across training.
    :type ema_schedule: str

    .. note::
       Invariants: `img_size % patch_size == 0`; `0.0 <= mask_ratio < 1.0`
       (a ratio of 1.0 would leave the next-frame loss undefined);
       `lambda_next_frame >= 0.0` and `lambda_mask >= 0.0`.
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
    dropout_rate: float = 0.0

    # --- Tube-masked latent prediction ---
    mask_prediction_enabled: bool = True
    mask_ratio: float = 0.6
    lambda_next_frame: float = 1.0
    lambda_mask: float = 1.0

    # --- Multi-horizon prediction ---
    predict_horizons: Tuple[int, ...] = (1,)

    # --- EMA target encoder ---
    ema_momentum: float = 0.996
    ema_schedule: str = "none"

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
        # --- Tube-masking invariants ---
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
        # --- predict_horizons invariants ---
        # Normalize list → tuple (tolerates JSON round-trips that arrive
        # as lists when callers pass through to_dict/from_dict).
        if isinstance(self.predict_horizons, list):
            self.predict_horizons = tuple(self.predict_horizons)
        if not isinstance(self.predict_horizons, tuple):
            raise ValueError(
                f"predict_horizons must be a tuple, got "
                f"{type(self.predict_horizons).__name__}"
            )
        if len(self.predict_horizons) == 0:
            raise ValueError("predict_horizons must be non-empty.")
        if not all(isinstance(h, int) for h in self.predict_horizons):
            raise ValueError(
                f"predict_horizons must contain only ints, got "
                f"{self.predict_horizons!r}"
            )
        if not all(h > 0 for h in self.predict_horizons):
            raise ValueError(
                f"predict_horizons entries must all be > 0, got "
                f"{self.predict_horizons!r}"
            )
        if len(set(self.predict_horizons)) != len(self.predict_horizons):
            raise ValueError(
                f"predict_horizons must be unique, got "
                f"{self.predict_horizons!r}"
            )
        if list(self.predict_horizons) != sorted(self.predict_horizons):
            raise ValueError(
                f"predict_horizons must be sorted ascending, got "
                f"{self.predict_horizons!r}"
            )
        # Only enforce max(h) < num_frames when the loss can actually run.
        # When num_frames < 2 the per-horizon loss block is skipped entirely
        # (matches the legacy single-horizon T<2 edge case), so the
        # max(h)<num_frames invariant is vacuous and would otherwise reject
        # the default (1,) at num_frames=1 (the T=1 edge-case test).
        if self.num_frames >= 2 and max(self.predict_horizons) >= self.num_frames:
            raise ValueError(
                f"max(predict_horizons)={max(self.predict_horizons)} must be "
                f"strictly less than num_frames={self.num_frames}."
            )
        # --- EMA target encoder invariants ---
        if not (0.0 <= self.ema_momentum < 1.0):
            raise ValueError(
                f"ema_momentum must be in [0.0, 1.0), got {self.ema_momentum}. "
                "Upper bound is strict: m=1.0 freezes the target forever, "
                "which is equivalent to disabling training of the target side."
            )
        if self.ema_schedule not in {"none", "cosine"}:
            raise ValueError(
                f"ema_schedule must be one of {{'none', 'cosine'}}, got "
                f"{self.ema_schedule!r}."
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
        for k in ("encoder_shifts", "predictor_shifts", "predict_horizons"):
            if isinstance(d.get(k), tuple):
                d[k] = list(d[k])
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VideoJEPAConfig":
        """Construct a config from a dict produced by :meth:`to_dict`."""
        d = dict(d)
        # Tolerate legacy keys from older checkpoints.
        for legacy_key in ("cond_dim", "telemetry_dim"):
            d.pop(legacy_key, None)
        for k in ("encoder_shifts", "predictor_shifts", "predict_horizons"):
            if k in d and isinstance(d[k], list):
                d[k] = tuple(d[k])
        # Drop keys the dataclass does not recognize, for forward-compat with
        # checkpoints written by future versions that add fields.
        import dataclasses as _dc
        valid = {f.name for f in _dc.fields(cls)}
        d = {k: v for k, v in d.items() if k in valid}
        return cls(**d)
