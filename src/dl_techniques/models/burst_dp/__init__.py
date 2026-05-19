"""BurstDP: multi-view reference-conditioned vision model."""

from .config import (
    DEFAULT_N_MAX,
    DEFAULT_NUM_SEG_CLASSES,
    PRESETS,
    VALID_FUSION_TYPES,
    BurstDPConfig,
    FusionType,
    get_preset,
)
from .fusion import BurstFusionBlock, BurstFusionBlockAdaLN
from .heads import DepthHead, ReconstructionHead, SegmentationHead
from .model import BurstDP, create_burst_dp

__all__ = [
    "DEFAULT_N_MAX",
    "DEFAULT_NUM_SEG_CLASSES",
    "PRESETS",
    "VALID_FUSION_TYPES",
    "FusionType",
    "BurstDPConfig",
    "get_preset",
    "BurstFusionBlock",
    "BurstFusionBlockAdaLN",
    "ReconstructionHead",
    "SegmentationHead",
    "DepthHead",
    "BurstDP",
    "create_burst_dp",
]
