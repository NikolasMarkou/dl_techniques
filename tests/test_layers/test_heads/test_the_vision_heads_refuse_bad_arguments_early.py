"""Every vision head refuses a bad argument at construction, by name.

Guard for plan-2026-10-14T062616-81938e81 step 11.1 (the user's heads review,
``research/reviews/20261015_heads_review.md`` items VIS-1, VIS-3, VIS-4, VIS-5
and the segmentation-skip item). Measured at ``9343b0f87``:

* VIS-1: ``ClassificationHead(pooling_type='bogus')`` constructed, built no
  pooling layer, and died on the first call with ``AttributeError: ...
  has no attribute 'pooling'``.
* VIS-3: a non-positive count reached a ``Conv2D`` and failed there with Keras'
  ``Invalid value for argument `filters` ... filters=0``, which names neither
  the head nor the argument; ``num_classes=True`` built a one-class detector;
  ``ClassificationHead(num_classes=0)`` returned ``(B, 0)`` logits;
  ``EnhancementHead(scale_factor=0)`` and ``scale_factor=-2`` silently took
  the same-resolution path; a ``hidden_dim`` too small for a head's halvings
  reached a ``filters=0`` conv.
* VIS-4: ``create_enhancement_head('super_resolution')`` compared the string
  with the enum, missed, and built a ``scale_factor=1`` head; any other task
  type built an ``EnhancementHead`` too.
* VIS-5: ``create_vision_head('surface_normals', output_channels=3)`` raised
  ``TypeError: ... got multiple values for keyword argument
  'output_channels'``.
* A ``SegmentationHead`` skip whose height and width differ from the feature
  map failed inside ``ops.concatenate`` with a backend
  ``InvalidArgumentError``.
"""

from typing import Any, Callable, Dict

import keras
import numpy as np
import pytest

from dl_techniques.layers.heads.vision.factory import (
    ClassificationHead,
    DepthEstimationHead,
    DetectionHead,
    EnhancementHead,
    InstanceSegmentationHead,
    SegmentationHead,
    create_enhancement_head,
    create_vision_head,
)
from dl_techniques.layers.heads.vision.task_types import VisionTaskType

HIDDEN_DIM = 16
SPATIAL = (2, 6, 10, HIDDEN_DIM)


def _inputs(shape) -> np.ndarray:
    return np.random.default_rng(11).normal(size=shape).astype("float32")


# ---------------------------------------------------------------------
# VIS-1: pooling_type
# ---------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["bogus", "mean", "AVG", None])
def test_the_classification_head_refuses_an_unknown_pooling_type(bad) -> None:
    with pytest.raises(ValueError, match="pooling_type"):
        ClassificationHead(num_classes=3, hidden_dim=HIDDEN_DIM,
                           pooling_type=bad)


@pytest.mark.parametrize("good", ["avg", "max"])
def test_every_documented_spatial_pooling_type_still_builds(good) -> None:
    head = ClassificationHead(num_classes=3, hidden_dim=HIDDEN_DIM,
                              pooling_type=good)
    assert head(_inputs(SPATIAL))["logits"].shape == (2, 3)


# ---------------------------------------------------------------------
# VIS-3: counts
# ---------------------------------------------------------------------

#: (head label, constructor, the other required arguments, the count).
_COUNTS = (
    ("DetectionHead", DetectionHead, {"num_classes": 3}, "num_classes"),
    ("DetectionHead", DetectionHead, {"num_classes": 3}, "num_anchors"),
    ("DetectionHead", DetectionHead, {"num_classes": 3}, "bbox_dims"),
    ("DetectionHead", DetectionHead, {"num_classes": 3}, "hidden_dim"),
    ("SegmentationHead", SegmentationHead, {"num_classes": 3}, "num_classes"),
    ("ClassificationHead", ClassificationHead, {"num_classes": 3}, "num_classes"),
    ("InstanceSegmentationHead", InstanceSegmentationHead, {"num_classes": 3},
     "num_classes"),
    ("InstanceSegmentationHead", InstanceSegmentationHead, {"num_classes": 3},
     "num_instances"),
    ("DepthEstimationHead", DepthEstimationHead, {}, "output_channels"),
    ("EnhancementHead", EnhancementHead, {}, "output_channels"),
    ("EnhancementHead", EnhancementHead, {}, "scale_factor"),
)


@pytest.mark.parametrize("bad", [0, -1, 2.5, True, "3"],
                         ids=["zero", "negative", "float", "bool", "str"])
@pytest.mark.parametrize("label,cls,required,arg", _COUNTS,
                         ids=[f"{c[0]}.{c[3]}" for c in _COUNTS])
def test_a_count_that_is_not_a_positive_int_is_refused_by_name(
        label: str, cls, required: Dict[str, Any], arg: str, bad: Any
) -> None:
    kwargs = {"hidden_dim": HIDDEN_DIM, **required, arg: bad}
    with pytest.raises(ValueError, match=arg):
        cls(**kwargs)


def test_a_numpy_integer_count_is_accepted() -> None:
    """A config read from numpy (``np.int64``) is a count, not a refusal."""
    head = DetectionHead(num_classes=np.int64(3), hidden_dim=np.int64(HIDDEN_DIM))
    assert head(_inputs(SPATIAL))["classifications"].shape == (2, 6, 10, 27)


#: (label, builder of the head at a given hidden_dim, smallest valid width).
#: The smallest width is derived from each head's halvings: the progressive
#: depth decoder and the segmentation upsampler each run three halvings
#: (``hidden_dim // 8`` channels), and the default classification blocks one.
_WIDTHS = (
    ("segmentation", lambda hd: SegmentationHead(num_classes=3, hidden_dim=hd), 8),
    ("depth_progressive", lambda hd: DepthEstimationHead(hidden_dim=hd), 8),
    ("surface_normals", lambda hd: create_vision_head(
        "surface_normals", hidden_dim=hd), 8),
    ("optical_flow", lambda hd: create_vision_head(
        "optical_flow", hidden_dim=hd), 8),
    ("classification_default_blocks", lambda hd: ClassificationHead(
        num_classes=3, hidden_dim=hd), 2),
)


@pytest.mark.parametrize("label,build,smallest", _WIDTHS, ids=[w[0] for w in _WIDTHS])
def test_a_hidden_dim_too_small_for_the_halvings_is_refused(
        label: str, build: Callable[[int], keras.layers.Layer], smallest: int
) -> None:
    with pytest.raises(ValueError, match="hidden_dim"):
        build(smallest - 1)
    # The smallest valid width builds and runs.
    out = build(smallest)(_inputs((2, 6, 10, smallest)))
    assert out is not None


def test_a_segmentation_head_without_upsampling_needs_no_eighth_width() -> None:
    """At ``upsampling_factor=1`` no upsample conv exists, so only the three
    refine blocks bound the width (``hidden_dim // 4 >= 1``)."""
    head = SegmentationHead(num_classes=3, hidden_dim=4, upsampling_factor=1,
                            use_skip_connections=False)
    assert head(_inputs((2, 6, 10, 4))).shape == (2, 6, 10, 3)
    with pytest.raises(ValueError, match="hidden_dim"):
        SegmentationHead(num_classes=3, hidden_dim=3, upsampling_factor=1)


# ---------------------------------------------------------------------
# VIS-4: create_enhancement_head takes the string form and refuses others
# ---------------------------------------------------------------------

@pytest.mark.parametrize("task", ["super_resolution", " Super_Resolution ",
                                  VisionTaskType.SUPER_RESOLUTION])
def test_a_string_super_resolution_task_gets_the_x2_default(task) -> None:
    head = create_enhancement_head(task, hidden_dim=HIDDEN_DIM)
    assert head.scale_factor == 2


def test_an_explicit_scale_factor_wins_over_the_default() -> None:
    head = create_enhancement_head("super_resolution", hidden_dim=HIDDEN_DIM,
                                   scale_factor=4)
    assert head.scale_factor == 4


@pytest.mark.parametrize("task", ["denoising", VisionTaskType.DENOISING])
def test_denoising_keeps_the_same_resolution(task) -> None:
    assert create_enhancement_head(task, hidden_dim=HIDDEN_DIM).scale_factor == 1


@pytest.mark.parametrize("task", ["detection", VisionTaskType.CLASSIFICATION,
                                  "dehaze"])
def test_a_task_that_is_not_an_enhancement_task_is_refused(task) -> None:
    name = task.value if isinstance(task, VisionTaskType) else task
    with pytest.raises(ValueError, match=name):
        create_enhancement_head(task, hidden_dim=HIDDEN_DIM)


# ---------------------------------------------------------------------
# VIS-5: a caller's output_channels never collides with a factory default
# ---------------------------------------------------------------------

@pytest.mark.parametrize("task,channels", [("surface_normals", 3),
                                           ("optical_flow", 2)])
def test_a_fixed_channel_task_refuses_output_channels_by_name(task, channels) -> None:
    """Normals are 3-vectors and flow is a 2-vector by definition, so their
    heads take no ``output_channels``; passing one is refused as a
    ``ValueError`` naming it, not a duplicate-keyword ``TypeError``."""
    with pytest.raises(ValueError, match="output_channels"):
        create_vision_head(task, hidden_dim=HIDDEN_DIM, output_channels=channels)


def test_a_caller_default_override_is_not_a_collision() -> None:
    """Every per-task default the factory supplies yields to the caller's."""
    head = create_vision_head("keypoint_detection", num_classes=5,
                              hidden_dim=HIDDEN_DIM, num_anchors=2,
                              use_regression=True)
    assert (head.num_anchors, head.use_regression) == (2, True)


# ---------------------------------------------------------------------
# Segmentation: a skip of another resolution is refused before concatenate
# ---------------------------------------------------------------------

def test_an_eager_skip_of_another_resolution_is_refused_by_name() -> None:
    head = SegmentationHead(num_classes=3, hidden_dim=HIDDEN_DIM)
    skip = np.zeros((2, 12, 20, 8), dtype="float32")
    with pytest.raises(ValueError, match="height and width"):
        head([skip, _inputs(SPATIAL)])


def test_a_symbolic_skip_of_another_resolution_is_refused_by_name() -> None:
    head = SegmentationHead(num_classes=3, hidden_dim=HIDDEN_DIM)
    with pytest.raises(ValueError, match="height and width"):
        head([keras.Input((12, 20, 8)), keras.Input(SPATIAL[1:])])


def test_a_skip_of_the_same_resolution_still_concatenates() -> None:
    head = SegmentationHead(num_classes=3, hidden_dim=HIDDEN_DIM)
    out = head([np.zeros((2, 6, 10, 8), dtype="float32"), _inputs(SPATIAL)])
    assert out.shape == (2, 24, 40, 3)
