"""Permanent build+forward smoke test for the nano_vlm_world_model family.

Part of the 2026-06-15 model build/forward sweep (plan_2026-06-15_b5cec9e4).

**No longer REPORT-ONLY.** The original version wrapped construction AND the
forward pass in `except Exception: pytest.xfail(...)`, so a total build break
reported as `xfail` -- the package's headline smoke test could not fail, which
is not an instrument. It now fails, and asserts the full output key set and
per-key shape rather than "some non-empty structure of finite values".

`create_score_based_nanovlm(variant, mode, vocab_size, ...)` verified from
source (model.py:559 -> ScoreBasedNanoVLM at model.py:33). GHOST family: was
dead-on-forward (MEMORY.md), fixed at 1b61a381. call() (model.py:202) consumes a
dict ``{'images': (B,224,224,3), 'text': (B,T), 'timesteps': (B,) optional}``.

Shapes MEASURED at ``variant='mini'`` (embed_dim 384, depth 6, img_size 224,
patch_size 16): the vision sequence is 197 = (224/16)**2 + 1 CLS token.
"""

import numpy as np

from ..smoke_contract_oracle import assert_finite

BATCH, TEXT_LEN, EMBED_DIM = 2, 16, 384
VISION_LEN = (224 // 16) ** 2 + 1  # 197

EXPECTED_SHAPES = {
    "denoised_vision": (BATCH, VISION_LEN, EMBED_DIM),
    "target_vision": (BATCH, VISION_LEN, EMBED_DIM),
    "noise_vision": (BATCH, VISION_LEN, EMBED_DIM),
    "denoised_text": (BATCH, TEXT_LEN, EMBED_DIM),
    "target_text": (BATCH, TEXT_LEN, EMBED_DIM),
    "noise_text": (BATCH, TEXT_LEN, EMBED_DIM),
    "joint_denoised_vision": (BATCH, VISION_LEN, EMBED_DIM),
    "joint_denoised_text": (BATCH, TEXT_LEN, EMBED_DIM),
    "joint_target_vision": (BATCH, VISION_LEN, EMBED_DIM),
    "joint_target_text": (BATCH, TEXT_LEN, EMBED_DIM),
    "timesteps": (BATCH,),
}


def _build():
    from dl_techniques.models.nano_vlm_world_model.model import (
        create_score_based_nanovlm,
    )

    return create_score_based_nanovlm(variant="mini", vocab_size=256)


def _inputs():
    return {
        "images": np.random.rand(BATCH, 224, 224, 3).astype("float32"),
        "text": np.random.randint(0, 256, size=(BATCH, TEXT_LEN)).astype("int32"),
    }


def _assert_contract(out):
    """The smoke assertion. Shared with the meta-test so it is proven falsifiable."""
    assert isinstance(out, dict), f"expected a dict of tensors, got {type(out)}"
    assert set(out) == set(EXPECTED_SHAPES), sorted(out)
    for key, expected in EXPECTED_SHAPES.items():
        assert tuple(out[key].shape) == expected, f"{key}: {tuple(out[key].shape)}"
    assert_finite(out)


def test_smoke_build_and_forward():
    _assert_contract(_build()(_inputs(), training=False))
