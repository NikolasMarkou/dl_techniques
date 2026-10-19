"""Timestral configuration for Keras implementation.

Mirrors the original TimestralConfig from the PyTorch reference implementation.
Only fields that affect the forward pass at inference/training are included.
"""

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

# Layer kinds and their attention axis ordering (True = time axis, False = variate axis)
LAYER_KINDS = ("time", "variate", "time_variate", "variate_time")
ATTENTION_GATES = ("none", "sdpa_sigmoid")

LAYER_KIND_AXES: dict[str, tuple[bool, ...]] = {
    "time": (True,),
    "variate": (False,),
    "time_variate": (True, False),
    "variate_time": (False, True),
}


@dataclass(frozen=True)
class TimestralConfig:
    """Architecture and decode settings of a Timestral checkpoint.

    Attributes:
        patch_size: Time steps per patch (default 32).
        d_model: Model width (embedding dimension).
        num_heads: Attention heads per attention sublayer.
        mlp_hidden_dim: Hidden width of the SwiGLU MLP.
        block_schedule: One entry per layer, each one of LAYER_KINDS.
        quantile_knots: Quantile levels of the head, symmetric about 0.5.
        block_size: Trained DHF block size K, max patches decoded per block.
        context_length: Trained context length, default context cap.
        rope_theta: RoPE base frequency.
        rope_partial_factor: Fraction of head dimensions to rotate.
        norm_eps: RMSNorm epsilon.
        attn_bias: Whether attention projections use bias.
        mlp_bias: Whether MLP projections use bias.
        time_attn_gate: Gate type for time attention ("none" or "sdpa_sigmoid").
        variate_attn_gate: Gate type for variate attention.
        minimum_scale: Floor for normalization scale.
        use_asinh: Whether to apply asinh squashing in normalization.
        stabilize_with_global: Whether to stabilize with global stats (must be False for advancing refine frame).
        covariate_enabled: Whether covariates are supported.
        split_context_horizon_attn: Whether to use two-stream attention mask.
        attend_missing_patches: Whether to attend to missing (padded) patches.
        quantile_width_factor: Increment scale of the monotone quantile head.
    """

    patch_size: int = 32
    d_model: int = 512
    prediction_length: int = 96
    num_heads: int = 8
    mlp_hidden_dim: int = 2048
    block_schedule: tuple[str, ...] = (
        "time", "time_variate", "time", "time_variate",
        "time", "time_variate", "time", "time_variate",
        "time", "time_variate", "time", "time_variate",
    )
    quantile_knots: tuple[float, ...] = (
        0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
        0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99
    )
    block_size: int = 32
    context_length: int = 32768
    rope_theta: float = 10000.0
    rope_partial_factor: float = 0.5
    norm_eps: float = 1e-6
    attn_bias: bool = False
    mlp_bias: bool = False
    time_attn_gate: str = "sdpa_sigmoid"
    variate_attn_gate: str = "sdpa_sigmoid"
    minimum_scale: float = 1e-6
    use_asinh: bool = True
    stabilize_with_global: bool = False
    covariate_enabled: bool = True
    split_context_horizon_attn: bool = True
    attend_missing_patches: bool = True
    quantile_width_factor: float = 2.0

    def __post_init__(self) -> None:
        # Validate d_model divisible by num_heads
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} not divisible by num_heads={self.num_heads}"
            )
        # Validate rope_dim
        if self.rope_dim <= 0 or self.rope_dim % 2 != 0:
            raise ValueError(
                f"rope_partial_factor={self.rope_partial_factor} gives invalid "
                f"rope_dim={self.rope_dim}"
            )
        # Validate block_schedule kinds
        for kind in self.block_schedule:
            if kind not in LAYER_KINDS:
                raise ValueError(
                    f"Unknown layer kind {kind!r}; expected one of {LAYER_KINDS}"
                )
        # Validate attention gates
        for gate in (self.time_attn_gate, self.variate_attn_gate):
            if gate not in ATTENTION_GATES:
                raise ValueError(
                    f"Unknown attention gate {gate!r}; expected one of {ATTENTION_GATES}"
                )
        # Validate quantile knots
        knots = self.quantile_knots
        if 0.5 not in knots:
            raise ValueError(f"quantile_knots must contain 0.5, got {knots}")
        if not all(a < b for a, b in zip(knots, knots[1:])):
            raise ValueError(f"quantile_knots must be strictly increasing, got {knots}")
        # Advancing refine frame requires stabilize_with_global=False
        if self.stabilize_with_global:
            raise ValueError(
                "The advancing refine frame requires stabilize_with_global=False"
            )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads

    @property
    def rope_dim(self) -> int:
        """Head dimensions rotated by RoPE; the rest pass through unrotated."""
        return int(self.head_dim * self.rope_partial_factor)

    @property
    def num_quantiles(self) -> int:
        return len(self.quantile_knots)

    @property
    def median_index(self) -> int:
        return self.quantile_knots.index(0.5)

    @property
    def supports_known_covariates(self) -> bool:
        """Whether the layer schedule can route a known covariate's future to targets.

        Some time op must be followed by at least one more time op and at least
        one variate op; otherwise the future of a known covariate never reaches a target.
        """
        if not self.covariate_enabled:
            return False
        ops = [
            is_time
            for kind in self.block_schedule
            for is_time in LAYER_KIND_AXES[kind]
        ]
        for i, is_time in enumerate(ops):
            rest = ops[i + 1 :]
            if is_time and any(rest) and not all(rest):
                return True
        return False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["block_schedule"] = list(self.block_schedule)
        d["quantile_knots"] = list(self.quantile_knots)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TimestralConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        d = dict(d)
        d["block_schedule"] = tuple(d["block_schedule"])
        d["quantile_knots"] = tuple(float(q) for q in d["quantile_knots"])
        return cls(**d)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "TimestralConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))


# Predefined variant configurations matching the Hugging Face models
# These will be populated in model.py after the class is defined
TIMESTRAL_VARIANTS: dict[str, dict[str, Any]] = {}