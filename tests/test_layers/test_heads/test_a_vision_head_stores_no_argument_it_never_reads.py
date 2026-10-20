"""A vision head refuses, and does not store, an argument its forward pass never reads.

Guard for plan-2026-10-14T062616-81938e81 step 11.4(c), the leftovers the
Step 11.1 report listed (D-054). Measured at ``56b696fe2``:

* ``DetectionHead(use_regression=False)`` stored ``bbox_dims=4`` and wrote it
  to its config although no regression branch exists to read it, and took any
  other value in silence.
* ``DepthEstimationHead(decoder_style='dpt')`` stored ``hidden_dim=256``
  (that mode was deleted in step 8, D-222).
* ``SegmentationHead(skip_channel_mode='explicit')`` created its upsample
  convs at a ``hidden_dim // 8`` placeholder and overwrote ``filters`` in
  ``build``, so ``hidden_dim < 8`` was refused for a width no layer used. In
  explicit mode the refine widths come from ``explicit_skip_channels`` too,
  so ``hidden_dim`` is read only by the attention and FFN stages.

The rule is D-030's: a value for an argument the head never reads is refused
with ``ValueError`` and the head stores ``None``, which round-trips. When the
attention or FFN stage is on, ``hidden_dim`` is read and accepted.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.heads.vision.factory import (
    DetectionHead, SegmentationHead, VisionTaskType, create_vision_head)

_X = np.random.default_rng(17).normal(size=(2, 4, 6, 16)).astype("float32")

#: Explicit skips, deepest first; the call takes them shallowest first, with
#: the main map last, all at the main map's resolution.
_SKIPS = [12, 10, 6, 4]
_EXPLICIT_INPUTS = (
    [np.random.default_rng(i).normal(size=(2, 4, 6, c)).astype("float32")
     for i, c in enumerate(reversed(_SKIPS))]
    + [np.random.default_rng(9).normal(size=(2, 4, 6, 16)).astype("float32")])
#: refine block i emits _SKIPS[-(i + 2)] // 2 channels, then skip i is
#: concatenated (i < 3): the last width is _SKIPS[0] // 2 + _SKIPS[2].
_EXPLICIT_WIDTH = _SKIPS[0] // 2 + _SKIPS[2]


def _round_trip(layer):
    restored = type(layer).from_config(layer.get_config())
    assert restored.get_config() == layer.get_config()
    return restored


# ---------------------------------------------------------------------
# DetectionHead.bbox_dims
# ---------------------------------------------------------------------

def test_a_box_free_detection_head_refuses_bbox_dims() -> None:
    with pytest.raises(ValueError, match="bbox_dims"):
        DetectionHead(num_classes=3, use_regression=False, bbox_dims=4)


def test_a_box_free_detection_head_stores_no_bbox_dims() -> None:
    head = DetectionHead(num_classes=3, use_regression=False)
    assert head.bbox_dims is None and head.get_config()["bbox_dims"] is None
    restored = _round_trip(head)
    assert restored.bbox_dims is None
    keypoints = create_vision_head(VisionTaskType.KEYPOINT_DETECTION, num_classes=65)
    assert keypoints.bbox_dims is None


def test_a_box_detection_head_keeps_its_default_bbox_dims() -> None:
    head = DetectionHead(num_classes=3, num_anchors=2)
    assert head.bbox_dims == 4
    out = head(_X, training=False)
    assert tuple(out["regressions"].shape) == (2, 4, 6, 8)
    assert _round_trip(head).bbox_dims == 4
    assert DetectionHead(num_classes=3, num_anchors=2, bbox_dims=5).bbox_dims == 5


# ---------------------------------------------------------------------
# hidden_dim in the explicit segmentation mode
# ---------------------------------------------------------------------

def _explicit(**kwargs):
    return SegmentationHead(num_classes=3, skip_channel_mode="explicit",
                            explicit_skip_channels=_SKIPS, **kwargs)


_UNREAD = {
    "explicit_segmentation": lambda **kw: _explicit(use_ffn=False, **kw),
}


@pytest.mark.parametrize("make", list(_UNREAD.values()), ids=list(_UNREAD))
def test_an_unread_hidden_dim_is_refused(make) -> None:
    with pytest.raises(ValueError, match="hidden_dim"):
        make(hidden_dim=64)


@pytest.mark.parametrize("make", list(_UNREAD.values()), ids=list(_UNREAD))
def test_an_unread_hidden_dim_is_stored_as_none(make) -> None:
    head = make()
    assert head.hidden_dim is None and head.get_config()["hidden_dim"] is None
    assert _round_trip(head).hidden_dim is None


# ---------------------------------------------------------------------
# SegmentationHead explicit mode: the upsample convs at the real width
# ---------------------------------------------------------------------

def test_the_explicit_upsample_convs_are_created_at_the_width_they_take() -> None:
    head = _explicit(use_ffn=False)
    assert [u.filters for u in head.upsample_layers] == [_EXPLICIT_WIDTH] * 2
    out = head(_EXPLICIT_INPUTS, training=False)
    assert tuple(out.shape) == (2, 16, 24, 3)
    assert [tuple(u.kernel.shape) for u in head.upsample_layers] == [
        (3, 3, _EXPLICIT_WIDTH, _EXPLICIT_WIDTH)] * 2
    assert tuple(out.shape) == head.compute_output_shape(
        [tuple(x.shape) for x in _EXPLICIT_INPUTS])


def test_the_explicit_mode_takes_a_small_hidden_dim_for_its_ffn() -> None:
    """`hidden_dim // 8` was 0 here and the head refused; the FFN reads 4."""
    head = _explicit(hidden_dim=4, use_ffn=True)
    main = np.random.default_rng(3).normal(size=(2, 4, 6, 4)).astype("float32")
    out = head([*_EXPLICIT_INPUTS[:-1], main], training=False)
    assert tuple(out.shape) == (2, 16, 24, 3)


def test_the_explicit_mode_round_trips_by_value(tmp_path) -> None:
    """Every weight by path, and the eager output, survive a `.keras` round trip.

    Eager calls on the CPU on both sides: two separately compiled `predict`
    graphs of one model differed by 1.5e-8 in 9 of 2304 values at 56b696fe2
    (XLA fusion), and one of seven eager GPU runs differed too (cuDNN
    algorithm choice); neither says anything about the archive.
    """
    head = _explicit(use_ffn=False)
    inputs = [keras.Input(shape=x.shape[1:]) for x in _EXPLICIT_INPUTS]
    model = keras.Model(inputs, head(inputs))
    path = tmp_path / "explicit.keras"
    model.save(path)
    restored = keras.models.load_model(path)
    source = {w.path: np.asarray(w) for w in model.weights}
    target = {w.path: np.asarray(w) for w in restored.weights}
    assert sorted(source) == sorted(target)
    for key, value in source.items():
        np.testing.assert_array_equal(target[key], value, err_msg=key)
    with keras.device("cpu:0"):
        np.testing.assert_array_equal(
            np.asarray(restored(_EXPLICIT_INPUTS, training=False)),
            np.asarray(model(_EXPLICIT_INPUTS, training=False)))
