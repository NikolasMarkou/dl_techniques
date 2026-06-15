"""Permanent build+forward smoke test for the dino family (v1).

Part of the 2026-06-15 model build/forward sweep. The construction-order crash
(`add_weight(cls_token)` before `super().__init__`) was FIXED in
plan_2026-06-15_39a31d4a (D-001): the CLS token is now owned by the
`ClassTokenPrepend` sub-layer. This test is therefore a REAL forward+finiteness
assertion, no longer an xfail.

dino has no ``__init__`` exports, so import the factory directly from the
submodule. `create_dino_v1(variant, num_classes, patch_size, input_shape, ...)`
verified from source. The patch grid is derived from ``image_size // patch_size``
at build time, so a small ``input_shape`` is legal: with ``patch_size=16`` a
32x32 image yields a 2x2 patch grid. ``num_classes=10`` + default
``include_top=True`` returns logits ``(B, 10)``.

FORWARD ONLY: the DINOHead ``.keras`` round-trip is known-broken — not tested.
"""

import numpy as np


def _assert_finite(value):
    arr = np.asarray(value)
    assert arr is not None
    assert not np.any(np.isnan(arr))
    assert not np.any(np.isinf(arr))


def test_smoke_build_and_forward():
    from dl_techniques.models.dino.dino_v1 import create_dino_v1

    model = create_dino_v1(
        "small",
        num_classes=10,
        patch_size=16,
        input_shape=(32, 32, 3),
    )

    images = np.random.rand(2, 32, 32, 3).astype("float32")
    out = model(images, training=False)

    # Finiteness asserted OUTSIDE any try so a NaN/Inf fails loudly (no xfail).
    # Output may be a tensor (logits) or dict depending on head config.
    if isinstance(out, dict):
        for v in out.values():
            _assert_finite(v)
    else:
        _assert_finite(out)

    # include_top=True + num_classes=10 -> logits (B, 10).
    if not isinstance(out, dict):
        assert tuple(np.asarray(out).shape) == (2, 10)
