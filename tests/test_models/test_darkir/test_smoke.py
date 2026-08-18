"""Permanent build+forward smoke test for the darkir family.

Part of the 2026-06-15 model build/forward sweep (plan_2026-06-15_b5cec9e4).

The `except Exception: pytest.xfail(...)` wrapper was removed from this file in
an earlier pass, but its "REPORT-ONLY ... captured via xfail" docstring outlived
it and the body still asserted only finiteness -- so a forward returning the
scalar `0.0` passed. It now asserts the restored image's shape.

`create_darkir_model(img_channels, width, ...)` verified from source
(model.py:1299). Pure functional U-Net for low-light image restoration; NHWC
float32 input. MEASURED: the output is a single tensor with the SAME shape as
the input -- restoration, no spatial resize and no side outputs. That equality
is the contract worth pinning: darkir uses the recently-rewritten ``FreMLP``
(FFT path), where a shape regression is a plausible failure.
"""

import numpy as np

from ..smoke_contract_oracle import (
    assert_contract_rejects_a_broken_forward,
    assert_finite,
)

BATCH, HEIGHT, WIDTH, CHANNELS = 2, 32, 32, 3


def _build():
    from dl_techniques.models.darkir.model import create_darkir_model

    return create_darkir_model(img_channels=CHANNELS, width=16)


def _inputs():
    return np.random.rand(BATCH, HEIGHT, WIDTH, CHANNELS).astype("float32")


def _assert_contract(out):
    """The smoke assertion. Shared with the meta-test so it is proven falsifiable."""
    assert not isinstance(out, (dict, list, tuple)), (
        f"darkir should return a single restored image, got {type(out)}"
    )
    assert tuple(out.shape) == (BATCH, HEIGHT, WIDTH, CHANNELS), tuple(out.shape)
    assert_finite(out)


def test_smoke_build_and_forward():
    images = _inputs()
    out = _build()(images, training=False)
    _assert_contract(out)
    # Stated twice on purpose: the tuple above pins the literal geometry, this
    # pins the input/output relation a restoration model must hold at ANY size.
    assert tuple(out.shape) == tuple(images.shape)


def test_the_smoke_contract_rejects_a_broken_forward():
    """RED-proof, in-suite: the assertion above can actually fail.

    Breaks the MODEL -- the built model's forward output is replaced by a
    degenerate one -- rather than the factory's argument validation. See
    ``smoke_contract_oracle`` for why the argument-validation form of this
    meta-test (which yolo12 shipped) proved nothing.

    This package carries the meta-test because it is built by the FUNCTIONAL API
    (`keras.Model(inputs, outputs)`), not by subclassing; the injection patches
    `model.call` and has to reach a functional graph's forward too.
    """
    assert_contract_rejects_a_broken_forward(_build(), _inputs(), _assert_contract)
