"""Permanent build+forward smoke test for the fftnet family (VISION path only).

Part of the 2026-06-15 model build/forward sweep (plan_2026-06-15_b5cec9e4).

**No longer REPORT-ONLY.** The original version wrapped construction AND the
forward pass in `except Exception: pytest.xfail(...)`, so a total build break
reported as `xfail` -- the package's headline smoke test could not fail, which
is not an instrument. It now fails, and asserts the output key set and per-key
shape rather than only `isinstance(out, dict)`.

STALE-ENTRYPOINT NOTE: entrypoints.md row 20 lists fftnet as a token-id LM
(``int32 (B,T)``, ``create_fftnet(vocab_size=256)``). The actual source
(model.py:342 class, model.py:814 factory) is a VISION foundation model:
``create_fftnet(variant, image_size, patch_size)`` taking ``(B, H, W, 3)``
float32 images. There is no ``vocab_size`` argument. We test the vision path
only.

Shapes MEASURED at ``variant='tiny'``, 32x32 image, 16x16 patches: 4 patches,
``last_hidden_state`` carries 5 tokens (4 patches + 1 CLS) at embed_dim 384.
The 5-vs-4 distinction is exactly what a finiteness-only assertion could not see.
"""

import numpy as np

from ..smoke_contract_oracle import assert_finite

BATCH, IMAGE_SIZE, PATCH_SIZE, EMBED_DIM = 2, 32, 16, 384
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 4

EXPECTED_SHAPES = {
    "last_hidden_state": (BATCH, NUM_PATCHES + 1, EMBED_DIM),
    "cls_token": (BATCH, EMBED_DIM),
    "patch_features": (BATCH, NUM_PATCHES, EMBED_DIM),
}


def _build():
    from dl_techniques.models.fftnet.model import create_fftnet

    # Smallest variant; tiny image so num_patches stays small.
    return create_fftnet(variant="tiny", image_size=IMAGE_SIZE, patch_size=PATCH_SIZE)


def _inputs():
    return np.random.rand(BATCH, IMAGE_SIZE, IMAGE_SIZE, 3).astype("float32")


def _assert_contract(out):
    """The smoke assertion. Shared with the meta-test so it is proven falsifiable."""
    assert isinstance(out, dict), f"expected a dict of tensors, got {type(out)}"
    assert set(out) == set(EXPECTED_SHAPES), sorted(out)
    for key, expected in EXPECTED_SHAPES.items():
        assert tuple(out[key].shape) == expected, f"{key}: {tuple(out[key].shape)}"
    assert_finite(out)


def test_smoke_build_and_forward():
    _assert_contract(_build()(_inputs(), training=False))
