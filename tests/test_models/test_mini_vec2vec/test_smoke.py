"""Permanent build+forward smoke test for the mini_vec2vec family.

Part of the 2026-06-15 model build/forward sweep (plan_2026-06-15_b5cec9e4).

**No longer REPORT-ONLY.** The original version wrapped construction AND the
forward pass in `except Exception: pytest.xfail(...)`, so a total build break
reported as `xfail` -- the package's headline smoke test could not fail, which
is not an instrument. It now fails, and asserts the aligner's output shape
rather than "some non-empty structure of finite values".

`create_mini_vec2vec_aligner(embedding_dim)` verified from source
(model.py:530 factory -> MiniVec2VecAligner at model.py:94). The aligner applies
a single learned linear transformation W to ONE embedding tensor; call()
(model.py:146) takes a single ``(B, embedding_dim)`` float32 input and returns
``ops.matmul(inputs, W)`` of the same shape.

W is identity-initialized in build(), which this file also pins: before any
procrustes alignment the aligner must be the IDENTITY, not merely shape- and
finiteness-correct. That assertion is what distinguishes a working un-aligned
aligner from one whose W was left at zeros or at a random draw -- both of which
the previous version accepted.
"""

import numpy as np
from keras import ops

from ..smoke_contract_oracle import assert_finite

BATCH, EMBEDDING_DIM = 2, 128


def _build():
    from dl_techniques.models.mini_vec2vec.model import create_mini_vec2vec_aligner

    return create_mini_vec2vec_aligner(embedding_dim=EMBEDDING_DIM)


def _inputs():
    # Single embedding tensor: (batch=2, embedding_dim=128) float32.
    return np.random.rand(BATCH, EMBEDDING_DIM).astype("float32")


def _assert_contract(out):
    """The smoke assertion. Shared with the meta-test so it is proven falsifiable."""
    assert not isinstance(out, (dict, list, tuple)), (
        f"the aligner should return a single tensor, got {type(out)}"
    )
    assert tuple(out.shape) == (BATCH, EMBEDDING_DIM), tuple(out.shape)
    assert_finite(out)


def test_smoke_build_and_forward():
    _assert_contract(_build()(_inputs(), training=False))


def test_the_untrained_aligner_is_the_identity():
    """W is identity-initialized, so an un-aligned forward must be a no-op.

    A shape-and-finiteness contract cannot tell an identity-initialized W from a
    zeroed or randomly-drawn one; this is the value assertion that can.

    Two assertions, because they fail for different reasons. ``W`` itself is
    checked EXACTLY (measured: ``max|W - I| == 0.0``), which is a
    precision-regime-independent statement about the initializer. The forward is
    checked at ``1e-3``, and that tolerance is set by TF32 rather than by taste:

    * identity forward, TF32 ON  -> ``max|out - in| = 2.438e-04``
    * identity forward, TF32 OFF -> ``max|out - in| = 0.0`` (exact)
    * zeroed W                   -> ``9.988e-01``
    * random W (sigma=0.1)       -> ``2.19`` / ``2.95``

    measured on the RTX 4070 at float32. Whether TF32 is on is decided by
    whether ``tests/test_layers/test_linear_attention.py`` (which disables it
    process-globally at import) ran earlier in the session, so a tolerance
    tighter than ~5e-4 would be intermittently red depending on collection
    order -- the coupling logged as D-032. 1e-3 sits 4x above the noisier
    regime and ~1000x below the defect signal it must catch.
    """
    model = _build()
    embeddings = _inputs()
    out = model(embeddings, training=False)

    np.testing.assert_allclose(
        ops.convert_to_numpy(model.W), np.eye(EMBEDDING_DIM), rtol=0, atol=0
    )
    # DECISION plan-2026-08-17T183311-79c63e38/D-035: 1e-3, not 1e-5. MEASURED: the identity forward
    # deviates by 2.438e-04 with TF32 ON and 0.0 with it OFF, and which regime
    # applies depends on whether test_linear_attention.py (which disables TF32
    # process-globally at import) ran earlier in the session. A tighter bound is
    # intermittently red by collection order -- the D-032 coupling. The bound is
    # set from the DEFECT signal (zeroed W = 9.988e-01), not the noise floor.
    np.testing.assert_allclose(
        ops.convert_to_numpy(out), embeddings, rtol=1e-3, atol=1e-3
    )
