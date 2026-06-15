"""Permanent build+forward smoke test for the nano_vlm_world_model family.

Part of the 2026-06-15 model build/forward sweep (plan_2026-06-15_b5cec9e4).
REPORT-ONLY: a build/forward break is documented via xfail, never fixed.

`create_score_based_nanovlm(variant, mode, vocab_size, ...)` verified from
source (model.py:559 -> ScoreBasedNanoVLM at model.py:33). GHOST family: was
dead-on-forward (MEMORY.md), reportedly fixed at 1b61a381, but HIGH risk after
the 2026-06-14/15 transformers refactor (nano_vlm already broke on
MultiModalFusion in step 2). call() (model.py:202) consumes a dict
``{'images': (B,224,224,3), 'text': (B,T), 'timesteps': (B,) optional}`` and
returns a dict of denoised/target tensors. Smallest is ``variant='mini'``
(embed_dim 384, depth 6); img_size 224, patch_size 16. xfail with the exact
captured error if the forward breaks.
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
        from dl_techniques.models.nano_vlm_world_model.model import (
            create_score_based_nanovlm,
        )

        model = create_score_based_nanovlm(
            variant="mini", vocab_size=256
        )

        inputs = {
            "images": np.random.rand(2, 224, 224, 3).astype("float32"),
            "text": np.random.randint(0, 256, size=(2, 16)).astype("int32"),
        }
        out = model(inputs, training=False)
    except Exception as exc:  # noqa: BLE001
        pytest.xfail(
            f"nano_vlm_world_model build/forward failed: "
            f"{type(exc).__name__}: {exc}"
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
