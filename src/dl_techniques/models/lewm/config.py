"""LeWM configuration dataclass.

Mirrors `/tmp/lewm_source/config_lewm.yaml` defaults so we can reproduce
upstream behavior by default, but every field is overridable for tests /
smoke runs.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Tuple


@dataclass
class LeWMConfig:
    # Vision encoder
    img_size: int = 224
    patch_size: int = 14
    img_channels: int = 3
    encoder_scale: str = "tiny"  # dl_techniques ViT scale; tiny = 192d, 3h, 12L

    # Embeddings
    embed_dim: int = 192

    # Temporal setup
    history_size: int = 3
    num_preds: int = 1
    # num_frames sizes the predictor's positional embedding. It MUST cover the
    # training sequence length T = history_size + num_preds. Leave at the
    # sentinel 0 (default) to have it derived in __post_init__; an explicit
    # value is allowed only if it is >= history_size + num_preds.
    num_frames: int = 0

    # Predictor transformer
    depth: int = 6
    heads: int = 16
    dim_head: int = 64
    mlp_dim: int = 2048
    dropout: float = 0.1
    emb_dropout: float = 0.0

    # Projector (both projector and pred_proj share this config)
    projector_hidden_dim: int = 192

    # Action embedder
    action_dim: int = 2
    smoothed_dim: int = 10
    mlp_scale: int = 4

    # SIGReg
    sigreg_weight: float = 0.09
    sigreg_knots: int = 17
    sigreg_num_proj: int = 1024

    def __post_init__(self) -> None:
        # DECISION plan_2026-05-22_de5197c2/D-002: num_frames is a serialized
        # field (not a @property) so to_dict/from_dict round-trips for old and
        # new configs. Derive it from history_size + num_preds when the caller
        # left the sentinel 0; reject an explicit value that cannot cover the
        # training sequence length (would crash ARPredictor's pos-embedding add).
        required = self.history_size + self.num_preds
        if self.num_frames <= 0:
            self.num_frames = required
        elif self.num_frames < required:
            raise ValueError(
                f"num_frames={self.num_frames} is too small: it must cover the "
                f"training sequence length history_size + num_preds = "
                f"{self.history_size} + {self.num_preds} = {required}."
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LeWMConfig":
        return cls(**d)

    @property
    def input_image_shape(self) -> Tuple[int, int, int]:
        return (self.img_size, self.img_size, self.img_channels)
