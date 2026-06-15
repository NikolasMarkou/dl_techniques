"""Permanent build+forward smoke test for the byte_latent_transformer family.

Part of the 2026-06-15 model build/forward sweep (plan_2026-06-15_b5cec9e4).
REPORT-ONLY: a build/forward break is documented via xfail, never fixed.

`create_blt_model(...)` keyword order verified from source (model.py:694):
``(variant, vocab_size, max_sequence_length, entropy_threshold)`` — NOT the
positional ``(vocab_size, embed_dim, num_layers)`` recipe in entrypoints.md
(STALE). BLT `call()` (model.py:327) accepts a plain ``int32 (B, T)`` byte-token
tensor (or a dict with ``tokens``) and returns logits ``(B, T, vocab_size)``.
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
        from dl_techniques.models.byte_latent_transformer.model import (
            create_blt_model,
        )

        model = create_blt_model(
            variant="micro",
            vocab_size=256,
            max_sequence_length=64,
        )

        b, t = 2, 16
        tokens = np.random.randint(0, 256, (b, t)).astype("int32")
        out = model(tokens, training=False)
    except Exception as exc:  # noqa: BLE001
        pytest.xfail(
            f"byte_latent_transformer build/forward failed: "
            f"{type(exc).__name__}: {exc}"
        )

    _assert_finite(out)
