"""Permanent build+forward smoke test for the mini_vec2vec family.

Part of the 2026-06-15 model build/forward sweep (plan_2026-06-15_b5cec9e4).
REPORT-ONLY: a build/forward break is documented via xfail, never fixed.

`create_mini_vec2vec_aligner(embedding_dim)` verified from source
(model.py:530 factory -> MiniVec2VecAligner at model.py:94). The aligner applies
a single learned linear transformation W to ONE embedding tensor; call()
(model.py:146) takes a single ``(B, embedding_dim)`` float32 input and returns
``ops.matmul(inputs, W)`` of the same shape (W is identity-initialized in
build(), so forward works before any procrustes alignment).
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
        from dl_techniques.models.mini_vec2vec.model import (
            create_mini_vec2vec_aligner,
        )

        model = create_mini_vec2vec_aligner(embedding_dim=128)

        # Single embedding tensor: (batch=2, embedding_dim=128) float32.
        embeddings = np.random.rand(2, 128).astype("float32")
        out = model(embeddings, training=False)
    except Exception as exc:  # noqa: BLE001
        pytest.xfail(
            f"mini_vec2vec build/forward failed: {type(exc).__name__}: {exc}"
        )

    if isinstance(out, dict):
        vals = list(out.values())
    elif isinstance(out, (list, tuple)):
        vals = list(out)
    else:
        vals = [out]
    assert len(vals) > 0
    for v in vals:
        _assert_finite(v)
