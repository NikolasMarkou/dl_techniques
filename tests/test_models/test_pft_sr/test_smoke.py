"""Permanent build+forward smoke test for the pft_sr family.

Part of the 2026-06-15 model build/forward sweep (plan_2026-06-15_b5cec9e4).

The `except Exception: pytest.xfail(...)` wrapper was removed from this file in
an earlier pass, but TWO stale sentences outlived it ("REPORT-ONLY ... via
xfail" and "a build/forward break is captured via xfail") and the body still
asserted only finiteness -- so a forward returning the scalar `0.0` passed. It
now asserts the upsampled image's shape, which for a super-resolution model is
the whole point of the model.

`create_pft_sr(scale, variant)` verified from source. Permuted self-attention
super-resolution model; the paper-sourced ``light``/``base`` variants carry
``window_size=32`` in ``MODEL_VARIANTS`` (D-463), so input H/W are kept divisible
by 32 (32x32 -- one window).

MEASURED at ``scale=2``: a single output tensor ``(B, 2H, 2W, 3)``. A model that
forgot to upsample would return ``(B, H, W, 3)`` -- finite, correctly ranked, and
previously green.
"""

import numpy as np

from ..smoke_contract_oracle import assert_finite

BATCH, HEIGHT, WIDTH, CHANNELS, SCALE = 2, 32, 32, 3, 2


def _build():
    from dl_techniques.models.pft_sr.model import create_pft_sr

    return create_pft_sr(scale=SCALE, variant="light")


def _inputs():
    return np.random.rand(BATCH, HEIGHT, WIDTH, CHANNELS).astype("float32")


def _assert_contract(out):
    """The smoke assertion. Shared with the meta-test so it is proven falsifiable."""
    assert not isinstance(out, (dict, list, tuple)), (
        f"pft_sr should return a single upsampled image, got {type(out)}"
    )
    assert tuple(out.shape) == (
        BATCH,
        HEIGHT * SCALE,
        WIDTH * SCALE,
        CHANNELS,
    ), tuple(out.shape)
    assert_finite(out)


def test_smoke_build_and_forward():
    _assert_contract(_build()(_inputs(), training=False))
