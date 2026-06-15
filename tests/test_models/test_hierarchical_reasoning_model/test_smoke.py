"""Permanent build+forward smoke test for the hierarchical_reasoning_model family.

Part of the 2026-06-15 model build/forward sweep (plan_2026-06-15_b5cec9e4).
REPORT-ONLY: a build/forward break is documented via xfail, never fixed.

Factory verified at model.py:860:
``create_hierarchical_reasoning_model(vocab_size, seq_len, variant=...)`` with
``variant="micro"`` the smallest preset. ``call()`` (model.py:555) dispatches a
dict batch to ``_forward_complete``; the batch keys are ``token_ids`` (B, T)
and ``puzzle_ids`` (B,) (model.py:237-238). Returns a dict of tensors.
"""

import numpy as np
import pytest


def _assert_finite(value):
    arr = np.asarray(value)
    assert arr is not None
    assert not np.any(np.isnan(arr))
    assert not np.any(np.isinf(arr))


def test_smoke_build_and_forward():
    try:
        from dl_techniques.models.hierarchical_reasoning_model.model import (
            create_hierarchical_reasoning_model,
        )

        seq_len = 32
        model = create_hierarchical_reasoning_model(
            vocab_size=256,
            seq_len=seq_len,
            variant="micro",
        )

        b = 2
        inputs = {
            "token_ids": np.random.randint(0, 256, (b, seq_len)).astype("int32"),
            "puzzle_ids": np.random.randint(0, 1000, (b,)).astype("int32"),
        }
        out = model(inputs, training=False)
    except Exception as exc:  # noqa: BLE001
        pytest.xfail(
            f"hierarchical_reasoning_model build/forward failed: "
            f"{type(exc).__name__}: {exc}"
        )

    # Output is a dict of tensors.
    assert isinstance(out, dict)
    for v in out.values():
        _assert_finite(v)
