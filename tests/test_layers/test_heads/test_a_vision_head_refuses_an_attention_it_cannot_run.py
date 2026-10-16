"""A vision head refuses ``attention_type='shared_weights_cross'`` at construction.

Guard for plan-2026-10-14T062616-81938e81 step 12 Part F item 13 (D-089).
``SharedWeightsCrossAttention`` splits its input into modalities by
``split_sizes``, a required argument of every ``call``; a vision head calls
its attention stage with the features alone, on one stream, so no split
exists. Measured at d6995a932: every head constructed with it and then failed
inside ``call`` (``TypeError`` on the missing ``split_sizes``), far from the
argument that caused it. The head now refuses it in ``__init__``, naming the
reason; with ``use_attention=False`` the type is never built and stays
accepted, as any other unread ``attention_type`` is.
"""

import numpy as np
import pytest

from dl_techniques.layers.heads.vision.factory import (
    ClassificationHead, DepthEstimationHead, DetectionHead, InstanceSegmentationHead,
    OpticalFlowHead, SegmentationHead, SurfaceNormalsHead)

C = 32
X4 = np.random.default_rng(5).normal(size=(2, 4, 6, C)).astype("float32")

_HEADS = {
    "detection": lambda **kw: DetectionHead(num_classes=3, num_anchors=2, use_ffn=False, **kw),
    "segmentation": lambda **kw: SegmentationHead(num_classes=3, use_ffn=False, **kw),
    "classification": lambda **kw: ClassificationHead(num_classes=3, **kw),
    "instance_segmentation": lambda **kw: InstanceSegmentationHead(
        num_classes=3, num_instances=4, use_ffn=False, **kw),
    "depth": lambda **kw: DepthEstimationHead(**kw),
    "normals": lambda **kw: SurfaceNormalsHead(**kw),
    "flow": lambda **kw: OpticalFlowHead(**kw),
}


@pytest.mark.parametrize("name", sorted(_HEADS))
def test_the_head_refuses_it_at_construction(name):
    with pytest.raises(ValueError, match=r"shared_weights_cross.*split_sizes"):
        _HEADS[name](hidden_dim=C, use_attention=True,
                     attention_type="shared_weights_cross")


@pytest.mark.parametrize("name", sorted(_HEADS))
def test_an_unbuilt_attention_stage_still_accepts_it(name):
    head = _HEADS[name](hidden_dim=C, use_attention=False,
                        attention_type="shared_weights_cross")
    head(X4)


@pytest.mark.parametrize("name", sorted(_HEADS))
def test_a_runnable_attention_type_still_constructs_and_runs(name):
    head = _HEADS[name](hidden_dim=C, use_attention=True, attention_type="multi_head")
    head(X4)
