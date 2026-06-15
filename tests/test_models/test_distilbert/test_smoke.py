"""Permanent build+forward smoke test for the distilbert family.

Part of the 2026-06-15 model build/forward sweep (plan_2026-06-15_b5cec9e4).
REPORT-ONLY: a build/forward break is documented via xfail, never fixed.

We exercise the raw ``DistilBERT`` foundation class at its smallest config
(ctor verified at model.py:448). The ``create_distilbert_with_head`` factory
needs an ``NLPTaskConfig`` object and is out of scope here. ``call()``
(model.py:639) accepts a plain ``int32 (B, T)`` token tensor (or dict with
``input_ids``) and returns a dict with ``last_hidden_state``.
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
        from dl_techniques.models.distilbert.model import DistilBERT

        model = DistilBERT(
            vocab_size=256,
            hidden_size=64,
            num_layers=2,
            num_heads=2,
            intermediate_size=128,
            max_position_embeddings=64,
        )

        b, t = 2, 16
        tokens = np.random.randint(0, 256, (b, t)).astype("int32")
        out = model(tokens, training=False)
    except Exception as exc:  # noqa: BLE001
        pytest.xfail(
            f"distilbert build/forward failed: {type(exc).__name__}: {exc}"
        )

    # Output is a dict of tensors (last_hidden_state, attention_mask).
    assert isinstance(out, dict)
    _assert_finite(out["last_hidden_state"])
