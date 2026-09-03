"""``LeWMConfig``, a dataclass holding every hyperparameter LeWM needs.

Every field has a default that matches the upstream reference config, and
every field can be overridden independently — there is no named-scale
table, so retuning LeWM means changing a field, not picking a variant.

Caller note: ``num_frames`` is normally left at its sentinel default (0)
and gets derived from ``history_size + num_preds`` in ``__post_init__``.
An explicit ``num_frames`` is accepted only if it is at least that sum.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, Tuple


@dataclass
class LeWMConfig:
    # Vision encoder
    img_size: int = 224
    patch_size: int = 14
    img_channels: int = 3
    # dl_techniques ViT scale name; "tiny" means 192 dims, 3 heads, 12 layers.
    encoder_scale: str = "tiny"

    # Embeddings
    embed_dim: int = 192

    # Temporal setup
    history_size: int = 3
    num_preds: int = 1
    # Sizes the predictor's positional embedding; must cover history_size + num_preds.
    # Leave at 0 to derive it in __post_init__.
    num_frames: int = 0

    # Predictor transformer
    depth: int = 6
    heads: int = 16
    dim_head: int = 64
    mlp_dim: int = 2048
    dropout_rate: float = 0.1
    emb_dropout_rate: float = 0.0

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
        # num_frames is a stored field, not a property, so to_dict/from_dict round-trip it.
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
