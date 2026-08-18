"""Permanent build+forward smoke test for the yolo12 family.

Part of the 2026-06-15 model build/forward sweep (plan_2026-06-15_b5cec9e4).

**No longer REPORT-ONLY.** The original version wrapped construction AND the
forward pass in `except Exception: pytest.xfail(...)`, so a total build break
reported as `xfail` -- the package's headline smoke test could not fail, which
is not an instrument. It now fails, and asserts the feature pyramid's shape
rather than only its finiteness.

**The meta-test was replaced too, because it proved the wrong thing.** It used
to pass `scale="not_a_scale"` and assert `pytest.raises(Exception)`. That fails
at the variant lookup INSIDE the factory, before a model is ever built, so
deleting every assertion in the smoke body left it passing -- it proved the
factory validates its argument, not that this file's smoke assertion can fail.
It now breaks the MODEL, via `smoke_contract_oracle`.

**The shape assertion was strengthened at the same time.** It asserted
`tuple(feat.shape[:3])`, which silently ignores the channel dimension; a
pyramid with the right strides and the wrong widths passed. The full 4-tuple is
now pinned, channel counts MEASURED at `scale="n"`: 64 / 128 / 256.

`create_yolov12_feature_extractor(input_shape, scale)` verified from source
(feature_extractor.py:363). Returns a multi-scale feature pyramid ``[P3, P4,
P5]`` (a list of NHWC tensors). Input must be divisible by the deepest stride
(32); 64x64 is a small legal input. Uses the smallest scale ``"n"`` (nano).
"""

import numpy as np
import pytest

from ..smoke_contract_oracle import (
    assert_contract_rejects_a_broken_forward,
    assert_finite,
)

BATCH, IMAGE_SIZE = 2, 64
#: (stride, channels) per pyramid level, MEASURED at scale="n".
PYRAMID = ((8, 64), (16, 128), (32, 256))


def _build():
    from dl_techniques.models.yolo12.feature_extractor import (
        create_yolov12_feature_extractor,
    )

    return create_yolov12_feature_extractor(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), scale="n"
    )


def _inputs():
    return np.random.rand(BATCH, IMAGE_SIZE, IMAGE_SIZE, 3).astype("float32")


def _assert_contract(out):
    """The smoke assertion. Shared with the meta-test so it is proven falsifiable."""
    # Feature extractor returns a list/tuple (or dict) of feature maps.
    if isinstance(out, dict):
        feats = list(out.values())
    elif isinstance(out, (list, tuple)):
        feats = list(out)
    else:
        feats = [out]

    assert len(feats) == 3, [getattr(f, "shape", f) for f in feats]
    for feat, (stride, channels) in zip(feats, PYRAMID):
        assert tuple(feat.shape) == (
            BATCH,
            IMAGE_SIZE // stride,
            IMAGE_SIZE // stride,
            channels,
        ), f"stride-{stride} map has shape {tuple(feat.shape)}"
    assert_finite(feats)


def test_smoke_build_and_forward():
    _assert_contract(_build()(_inputs(), training=False))


def test_the_smoke_contract_rejects_a_broken_forward():
    """RED-proof, in-suite: the assertion above can actually fail.

    Breaks the MODEL, not the factory's argument -- see this module's docstring
    for why the previous version of this test did the latter and proved nothing.
    """
    # DECISION plan-2026-08-17T183311-79c63e38/D-035: do NOT go back to `pytest.raises(Exception)` around
    # an illegal factory argument. That form raises at the variant lookup INSIDE
    # the factory, before a model is built, so it is independent of the contract
    # above -- deleting every assertion in `_assert_contract` left the old
    # version green. Break the MODEL. See decisions.md D-035.
    assert_contract_rejects_a_broken_forward(_build(), _inputs(), _assert_contract)


def test_an_illegal_scale_still_raises():
    """What the OLD meta-test actually measured, kept under an honest name.

    Argument validation is worth pinning; it just is not evidence about the
    smoke assertion above, which is what the old name claimed.
    """
    from dl_techniques.models.yolo12.feature_extractor import (
        create_yolov12_feature_extractor,
    )

    with pytest.raises(Exception):
        create_yolov12_feature_extractor(
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), scale="not_a_scale"
        )
