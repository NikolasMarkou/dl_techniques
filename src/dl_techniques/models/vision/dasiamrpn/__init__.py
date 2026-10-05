"""DaSiamRPN — public API re-exports.

The distractor-aware Siamese region-proposal tracker, its backbone, the
anchor factory and the inference post-processing helpers.
"""

from .model import (
    DaSiamRPN,
    SiamRPNBackbone,
    create_dasiamrpn,
    dasiamrpn_score_size,
    generate_dasiamrpn_anchors,
    create_hann_window,
    decode_dasiamrpn_boxes,
)

__all__ = [
    "DaSiamRPN",
    "SiamRPNBackbone",
    "create_dasiamrpn",
    "dasiamrpn_score_size",
    "generate_dasiamrpn_anchors",
    "create_hann_window",
    "decode_dasiamrpn_boxes",
]
