"""Permanent build+forward smoke test for the dino family (v2).

Part of the 2026-06-15 model build/forward sweep (plan_2026-06-15_e2759fbc).
dino v2 (`DINOv2VisionTransformer` backbone + `DINOv2` classifier wrapper) had
NEVER been run end-to-end. This plan fixed a 13-bug chain in
`src/dl_techniques/models/dino/dino_v2.py`:
  - 7 source-verified bugs (Dense-in-Lambda x3, projection-hack CLS, wrong
    pos-embed attr, ops.cond dict-vs-tensor mismatch, nested-Lambda +
    symbolic-tensor assert), and
  - a 6-bug first-forward cascade (batch-None ``ops.ones`` broadcast, wrong
    attention-factory key, backbone/wrapper ``name=`` kwarg collision, wrapper
    input-name collision, masks input-is-output graph cycle, and a rank-0
    ``is_training`` shape coercion).

Forward contract (3-input masked-forward, verified from source):
``model([images (B,H,W,3) f32, masks (B, num_patches) bool, is_training () bool])``
returns finite classification logits ``(B, num_classes)``. dino has no
``__init__`` exports, so import the factory directly from the submodule.
``create_dino_v2('tiny', image_size=28, patch_size=14, num_classes=10)`` builds
a tiny model: 28x28 image / patch 14 -> (28//14)^2 = 4 patches, so masks are
``(B, 4)``; ``include_top=True`` + ``num_classes=10`` -> logits ``(B, 10)``.

FORWARD ONLY: no ``.keras`` round-trip (the DINOHead round-trip is a separate
known break). No xfail safety-net -- v2 forwards for real, so any future
build/forward regression fails loudly.
"""

import numpy as np


def _assert_finite(value):
    arr = np.asarray(value)
    assert arr is not None
    assert not np.any(np.isnan(arr))
    assert not np.any(np.isinf(arr))


def test_smoke_build_and_forward():
    from dl_techniques.models.dino.dino_v2 import create_dino_v2

    # v2 forwards cleanly (plan_2026-06-15_e2759fbc, 13 bugs fixed) -- no xfail
    # safety net, so any future regression in build/forward fails loudly.
    model = create_dino_v2(
        "tiny",
        image_size=28,
        patch_size=14,
        num_classes=10,
    )

    images = np.random.rand(2, 28, 28, 3).astype("float32")
    # 28 // 14 == 2 -> 2 * 2 == 4 patches.
    masks = np.zeros((2, 4), dtype=bool)
    is_training = np.array(False)

    out = model([images, masks, is_training], training=False)

    _assert_finite(out)
    assert tuple(np.asarray(out).shape) == (2, 10)


def test_masked_forward_path():
    """All-True masks exercise the mask-token application path (cheap extra
    coverage that the hoisted ``mask_token_projection`` / ``ops.where`` masking
    branch forwards finite, not just the all-False no-op path)."""
    from dl_techniques.models.dino.dino_v2 import create_dino_v2

    model = create_dino_v2(
        "tiny",
        image_size=28,
        patch_size=14,
        num_classes=10,
    )

    images = np.random.rand(2, 28, 28, 3).astype("float32")
    masks_all = np.ones((2, 4), dtype=bool)

    out2 = model([images, masks_all, np.array(False)], training=False)

    _assert_finite(out2)
    assert tuple(np.asarray(out2).shape) == (2, 10)
