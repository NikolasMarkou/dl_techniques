"""Permanent build+forward smoke test for the byte_latent_transformer family.

Part of the 2026-06-15 model build/forward sweep (plan_2026-06-15_b5cec9e4).

**No longer REPORT-ONLY.** The original version wrapped construction AND the
forward pass in `except Exception: pytest.xfail(...)`, so a total build break
reported as `xfail` -- the package's headline smoke test could not fail, which
is not an instrument. It now fails, and asserts the logits' shape rather than
only their finiteness.

`create_blt_model(...)` keyword order verified from source (model.py:694):
``(variant, vocab_size, max_sequence_length, entropy_threshold)`` -- NOT the
positional ``(vocab_size, embed_dim, num_layers)`` recipe in entrypoints.md
(STALE). BLT `call()` (model.py:327) accepts a plain ``int32 (B, T)`` byte-token
tensor (or a dict with ``tokens``).

MEASURED at ``variant='micro'``: logits ``(B, T, vocab_size)``. The sequence
axis matters here -- BLT patches bytes internally, and an implementation that
returned one logit per PATCH rather than per byte would keep the same rank and
the same finiteness while being wrong.
"""

import numpy as np

from ..smoke_contract_oracle import assert_finite

BATCH, SEQ_LEN, VOCAB_SIZE = 2, 16, 256


def _build():
    from dl_techniques.models.byte_latent_transformer.model import create_blt_model

    return create_blt_model(
        variant="micro",
        vocab_size=VOCAB_SIZE,
        max_sequence_length=64,
    )


def _inputs():
    return np.random.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN)).astype("int32")


def _assert_contract(out):
    """The smoke assertion. Shared with the meta-test so it is proven falsifiable."""
    assert not isinstance(out, (dict, list, tuple)), (
        f"blt should return a single logits tensor, got {type(out)}"
    )
    assert tuple(out.shape) == (BATCH, SEQ_LEN, VOCAB_SIZE), tuple(out.shape)
    assert_finite(out)


def test_smoke_build_and_forward():
    _assert_contract(_build()(_inputs(), training=False))
