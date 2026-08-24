"""Permanent build+forward smoke test for the hierarchical_reasoning_model family.

Part of the 2026-06-15 model build/forward sweep (plan_2026-06-15_b5cec9e4).

**No longer REPORT-ONLY.** The original version wrapped construction AND the
forward pass in `except Exception: pytest.xfail(...)`, so a total build break
reported as `xfail` -- the package's headline smoke test could not fail, which
is not an instrument. It now fails, and asserts the output key set and per-key
shape rather than only `isinstance(out, dict)`.

Factory verified at model.py:860:
``create_hierarchical_reasoning_model(vocab_size, seq_len, variant=...)`` with
``variant="micro"`` the smallest preset. ``call()`` (model.py:555) dispatches a
dict batch to ``_forward_complete``; the batch keys are ``token_ids`` (B, T)
and ``puzzle_ids`` (B,) (model.py:237-238).

Shapes MEASURED: ``logits`` (B, T, vocab), plus the two ACT halting heads
``q_halt_logits`` and ``q_continue_logits``, each ``(B,)``. Those two are the
reason a key-set assertion matters here -- HRM's adaptive-computation head is
the part most likely to be silently dropped, and the previous "every value is
finite" loop would have reported green over a two-key dict.
"""

import numpy as np

from ..smoke_contract_oracle import (
    assert_contract_rejects_a_broken_forward,
    assert_finite,
)

BATCH, SEQ_LEN, VOCAB_SIZE = 2, 32, 256

EXPECTED_SHAPES = {
    "logits": (BATCH, SEQ_LEN, VOCAB_SIZE),
    "q_halt_logits": (BATCH,),
    "q_continue_logits": (BATCH,),
}


def _build():
    from dl_techniques.models.hierarchical_reasoning_model.model import (
        create_hierarchical_reasoning_model,
    )

    return create_hierarchical_reasoning_model(
        vocab_size=VOCAB_SIZE,
        seq_len=SEQ_LEN,
        variant="micro",
    )


def _inputs():
    return {
        "token_ids": np.random.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN)).astype("int32"),
        "puzzle_ids": np.random.randint(0, 1000, (BATCH,)).astype("int32"),
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


def test_the_smoke_contract_rejects_a_broken_forward():
    """RED-proof, in-suite: the assertion above can actually fail.

    Breaks the MODEL -- the built model's forward output is replaced by a
    degenerate one -- rather than the factory's argument validation. See
    ``smoke_contract_oracle`` for why the argument-validation form of this
    meta-test (which yolo12 shipped) proved nothing.

    This package carries the meta-test because it returns a multi-key dict whose
    values differ in RANK (3 and 1), the case where a per-value loop is most
    likely to be written in a way that only reaches the first entry.
    """
    assert_contract_rejects_a_broken_forward(_build(), _inputs(), _assert_contract)
