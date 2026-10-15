"""No dense-regression vision head stores a dropout rate it never applies.

Guard for plan-2026-10-14T062616-81938e81 step 11.1 (review item VIS-2,
decision D-050). Measured at ``9343b0f87``: ``DepthEstimationHead`` defaulted
``dropout_rate`` to the base 0.1 and stored it, but its progressive
``ConvBlock`` s were built without it and its DPT stages have no dropout, so
with attention and the FFN off (the head's own defaults) the stored 0.1
reached no layer.

The reference decoders apply no dropout: the DPT/MiDaS depth head, the
Omnidata normals head (a MiDaS DPT with three output channels), the RAFT
``FlowHead`` and PWC-Net's ``predict_flow``, and the Mask R-CNN mask branch.
So the depth, normals and flow heads default ``dropout_rate`` to 0.0 and
refuse a positive rate unless ``use_attention`` or ``use_ffn`` is on, where it
is the attention/FFN dropout. ``InstanceSegmentationHead``'s mask branch takes
none either; its ``dropout_rate`` is the detection branch's.
"""

import numpy as np
import pytest

from dl_techniques.layers.conv_blocks.conv_block import ConvBlock
from dl_techniques.layers.heads.vision.factory import (
    DepthEstimationHead,
    InstanceSegmentationHead,
    create_vision_head,
)

HIDDEN_DIM = 16

#: Each dense-regression head, built through the public entry point.
_DECODERS = {
    "depth_progressive": lambda **kw: DepthEstimationHead(hidden_dim=HIDDEN_DIM, **kw),
    "depth_dpt": lambda **kw: DepthEstimationHead(
        hidden_dim=HIDDEN_DIM, decoder_style="dpt", dpt_dims=[8, 8],
        dpt_upsample_factor=4, **kw),
    "surface_normals": lambda **kw: create_vision_head(
        "surface_normals", hidden_dim=HIDDEN_DIM, **kw),
    "optical_flow": lambda **kw: create_vision_head(
        "optical_flow", hidden_dim=HIDDEN_DIM, **kw),
}


@pytest.mark.parametrize("name", sorted(_DECODERS))
def test_the_decoder_defaults_to_no_dropout(name: str) -> None:
    head = _DECODERS[name]()
    assert head.dropout_rate == 0.0 and type(head.dropout_rate) is float
    assert head.get_config()["dropout_rate"] == 0.0


@pytest.mark.parametrize("name", sorted(_DECODERS))
def test_a_dropout_no_layer_would_apply_is_refused(name: str) -> None:
    with pytest.raises(ValueError, match="dropout_rate"):
        _DECODERS[name](dropout_rate=0.1)


@pytest.mark.parametrize("name", sorted(_DECODERS))
def test_with_the_ffn_on_the_rate_reaches_the_ffn(name: str) -> None:
    head = _DECODERS[name](dropout_rate=0.25, use_ffn=True)
    assert head.ffn.dropout_rate == 0.25


def test_no_progressive_decoder_conv_block_carries_a_dropout() -> None:
    for name in ("depth_progressive", "surface_normals", "optical_flow"):
        head = _DECODERS[name](dropout_rate=0.25, use_ffn=True)
        blocks = [layer for layer in head._flatten_layers(include_self=False)
                  if isinstance(layer, ConvBlock)]
        assert len(blocks) == 3, (name, [b.name for b in blocks])
        assert all(b.dropout_rate == 0.0 and b.dropout is None for b in blocks), name


def test_the_instance_mask_branch_takes_no_dropout_and_the_detection_branch_does() -> None:
    head = InstanceSegmentationHead(num_classes=3, num_instances=4,
                                    hidden_dim=HIDDEN_DIM, dropout_rate=0.25)
    head(np.zeros((2, 6, 10, HIDDEN_DIM), dtype="float32"))
    assert [b.dropout_rate for b in head.mask_conv_blocks] == [0.0, 0.0, 0.0]
    assert head.detection_head.cls_conv.dropout_rate == 0.25
    assert head.detection_head.reg_conv.dropout_rate == 0.25
