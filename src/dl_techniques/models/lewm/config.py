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
    num_frames: int = 3  # = history_size — upstream parameterizes pos embedding by this

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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LeWMConfig":
        return cls(**d)

    @property
    def input_image_shape(self) -> Tuple[int, int, int]:
        return (self.img_size, self.img_size, self.img_channels)
