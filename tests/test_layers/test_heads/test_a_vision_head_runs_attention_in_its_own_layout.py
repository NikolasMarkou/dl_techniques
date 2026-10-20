"""A vision head runs its attention stage on the layout the attention type takes.

Guard for plan-2026-10-14T062616-81938e81 step 11.4(c). Measured at
``56b696fe2`` (probe over the 35 registered attention types, at rank 3 and
rank 4): every type a vision head can build takes ONE layout. The
sequence types (``multi_head`` and nine others) refuse a ``(B, H, W, C)``
map; the spatial types (``cbam``, ``spatial``, ``area``, ``mobile_mqa``,
``tripse1``-``4``) refuse a ``(B, N, C)`` sequence. The heads applied the
attention to whatever layout they held at that point, so:

* FFTNet detection with ``attention_type='cbam'`` failed: the sequence-format
  ``DetectionHead`` ran attention on the ``(B, N, D)`` patch tokens before
  reshaping them to the patch grid (the item in the 11.1 report);
* the default ``attention_type='multi_head'`` failed on every spatial head
  (detection, segmentation, classification, instance segmentation, depth,
  normals, flow) and on the sequence-format ``SegmentationHead``, which
  reshapes first.

Now a sequence type sees the map flattened to ``H * W`` tokens and then
restored, and a spatial type sees a sequence head's tokens on the patch grid
and then flattened back. Where the layouts already agreed nothing changed
(the vision-consumer census in the 11.4 record is bit-identical, FFTNet
detection with ``multi_head`` included). A grid-less sequence (a
``ClassificationHead`` fed ``(B, N, C)``) cannot run a spatial type, and
says so.

The inputs are non-square (4 x 6), so a transposed reshape cannot pass.

The arms above check shapes, finiteness and a gradient, and a restore that
reshapes to ``(B, W, H, -1)`` and transposes back keeps every one of them
green (W5, measured at e7c5b583b: 24 of 24). The value arms below replace the
attention by an identity and feed a position-coded map (channel 0 = row,
channel 1 = column, the rest = cell id): ``_apply_attention`` must then hand
back every cell exactly where it was, on the map -> sequence -> map path and
on the sequence -> grid -> sequence path.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.attention.factory import ATTENTION_REGISTRY
from dl_techniques.layers.heads.vision import factory as vision_factory
from dl_techniques.layers.heads.vision.factory import (
    ClassificationHead, DepthEstimationHead, DetectionHead, InstanceSegmentationHead,
    OpticalFlowHead, SegmentationHead, SurfaceNormalsHead)
from dl_techniques.models.vision.fftnet.model import FFTNet, create_fftnet

C = 32
H, W = 4, 6
_RNG = np.random.default_rng(13)
X4 = _RNG.normal(size=(2, H, W, C)).astype("float32")
X3 = X4.reshape(2, H * W, C)

_SPATIAL_HEADS = {
    "detection": lambda **kw: DetectionHead(num_classes=3, num_anchors=2, use_ffn=False, **kw),
    "segmentation": lambda **kw: SegmentationHead(num_classes=3, use_ffn=False, **kw),
    "classification": lambda **kw: ClassificationHead(num_classes=3, **kw),
    "instance_segmentation": lambda **kw: InstanceSegmentationHead(
        num_classes=3, num_instances=4, use_ffn=False, **kw),
    "depth": lambda **kw: DepthEstimationHead(**kw),
    "normals": lambda **kw: SurfaceNormalsHead(**kw),
    "flow": lambda **kw: OpticalFlowHead(**kw),
}
_SEQUENCE_HEADS = {
    "detection": lambda **kw: DetectionHead(
        num_classes=3, num_anchors=2, use_ffn=False, input_format="sequence",
        patch_grid_size=(H, W), **kw),
    "segmentation": lambda **kw: SegmentationHead(
        num_classes=3, use_ffn=False, input_format="sequence", patch_grid_size=(H, W), **kw),
}
_CASES = (
    [("spatial", name, at) for name in _SPATIAL_HEADS for at in ("multi_head", "cbam")]
    + [("sequence", name, at) for name in _SEQUENCE_HEADS for at in ("multi_head", "cbam")])


def _shapes(out):
    """Shapes of a call result or of a ``compute_output_shape`` result."""
    if isinstance(out, dict):
        return {k: _shapes(v) for k, v in out.items()}
    return tuple(out.shape) if hasattr(out, "shape") else tuple(out)


@pytest.mark.parametrize("layout,name,attention_type", _CASES,
                         ids=[f"{a}-{b}-{c}" for a, b, c in _CASES])
def test_every_head_runs_either_attention_layout(layout, name, attention_type) -> None:
    make = (_SPATIAL_HEADS if layout == "spatial" else _SEQUENCE_HEADS)[name]
    keras.utils.set_random_seed(1)
    head = make(hidden_dim=C, use_attention=True, attention_type=attention_type)
    x = tf.constant(X4 if layout == "spatial" else X3)
    with tf.GradientTape() as tape:
        out = head(x, training=False)
        total = sum(tf.reduce_sum(tf.cast(v, tf.float32) ** 2)
                    for v in tf.nest.flatten(out))
    assert _shapes(out) == _shapes(head.compute_output_shape(tuple(x.shape)))
    assert all(np.isfinite(keras.ops.convert_to_numpy(v)).all() for v in tf.nest.flatten(out))
    weights = head.attention.trainable_weights
    grads = tape.gradient(total, weights)
    assert weights and any(g is not None and float(tf.reduce_max(tf.abs(g))) > 0
                           for g in grads), "the attention stage is off the output path"


def test_a_grid_less_sequence_cannot_run_a_spatial_attention() -> None:
    head = ClassificationHead(num_classes=3, hidden_dim=C, use_attention=True,
                              attention_type="cbam", pooling_type="cls_token")
    with pytest.raises(ValueError, match="cbam"):
        head(X3, training=False)


def _accepts(head, x):
    try:
        head.attention(x, training=False)
        return True
    except Exception:  # noqa: BLE001 - which layouts a layer accepts is the measurement
        return False


def test_the_spatial_type_table_matches_what_each_layer_accepts() -> None:
    """Every type a vision head can construct is measured at both ranks."""
    measured_spatial, buildable = set(), set()
    for attention_type in sorted(ATTENTION_REGISTRY):
        try:
            seq_head = DetectionHead(num_classes=2, hidden_dim=C, use_ffn=False,
                                     use_attention=True, attention_type=attention_type)
            map_head = DetectionHead(num_classes=2, hidden_dim=C, use_ffn=False,
                                     use_attention=True, attention_type=attention_type)
        except Exception:  # noqa: BLE001 - a type a head cannot construct is out of scope
            continue
        buildable.add(attention_type)
        takes_sequence, takes_map = _accepts(seq_head, X3), _accepts(map_head, X4)
        # One layout or none ('shared_weights_cross' needs split_sizes per
        # call, so no head can run it); never both.
        assert not (takes_sequence and takes_map), attention_type
        if takes_map:
            measured_spatial.add(attention_type)
    assert {"multi_head", "cbam"} <= buildable
    assert vision_factory._SPATIAL_ATTENTION_TYPES & buildable == measured_spatial


@pytest.mark.parametrize("task_type,attention_type", [
    ("detection", "cbam"), ("detection", "multi_head"),
    ("segmentation", "multi_head"), ("segmentation", "cbam")])
def test_fftnet_heads_run_either_attention_type(task_type, attention_type) -> None:
    model = create_fftnet(
        "tiny", task_type=task_type, num_classes=4, image_size=32, patch_size=16,
        head_config_overrides={"attention_type": attention_type})
    assert type(model) is FFTNet and model.head.attention_type == attention_type
    out = model(np.zeros((2, 32, 32, 3), "float32"), training=False)
    if task_type == "detection":
        assert _shapes(out) == {"classifications": (2, 2, 2, 36), "regressions": (2, 2, 2, 36)}
    else:
        assert _shapes(out) == (2, 32, 32, 4)


def _position_coded_map() -> np.ndarray:
    """``(1, H, W, C)``: channel 0 = row, channel 1 = column, the rest = cell id."""
    x = np.zeros((1, H, W, C), "float32")
    for i in range(H):
        for j in range(W):
            x[0, i, j, 0], x[0, i, j, 1], x[0, i, j, 2:] = i, j, i * W + j
    return x


_VALUE_CASES = (
    [("map_to_sequence", name) for name in _SPATIAL_HEADS]
    + [("sequence_to_grid", name) for name in _SEQUENCE_HEADS])


@pytest.mark.parametrize("path,name", _VALUE_CASES, ids=[f"{p}-{n}" for p, n in _VALUE_CASES])
def test_an_identity_attention_hands_back_every_cell_in_place(path, name) -> None:
    if path == "map_to_sequence":
        # A sequence type on a map: flatten to H * W tokens, then restore.
        head = _SPATIAL_HEADS[name](hidden_dim=C, use_attention=True, attention_type="multi_head")
        x = _position_coded_map()
    else:
        # A spatial type on a sequence: place the tokens on the grid, then flatten.
        head = _SEQUENCE_HEADS[name](hidden_dim=C, use_attention=True, attention_type="cbam")
        x = _position_coded_map().reshape(1, H * W, C)
    head.attention = keras.layers.Identity()
    out = keras.ops.convert_to_numpy(head._apply_attention(keras.ops.convert_to_tensor(x)))
    assert out.shape == x.shape
    moved = np.argwhere((out != x).any(axis=-1))
    assert moved.size == 0, f"{len(moved)} cells moved, first at {moved[:3].tolist()}"
