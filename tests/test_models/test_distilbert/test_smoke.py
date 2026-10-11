"""Permanent build+forward smoke test for the distilbert family.

Part of the 2026-06-15 model build/forward sweep (plan_2026-06-15_b5cec9e4).

**No longer REPORT-ONLY.** The original version wrapped construction AND the
forward pass in `except Exception: pytest.xfail(...)`, so a total build break
reported as `xfail` -- the package's headline smoke test could not fail, which
is not an instrument. It now fails, and asserts the output key set and
``last_hidden_state``'s shape rather than only `isinstance(out, dict)`.

We exercise the raw ``DistilBERT`` foundation class at its smallest config.
A model with a task head (``include_top=True``) needs an ``NLPTaskConfig``
and is covered by ``test_model.py::TestDistilBERTIncludeTop``, not here.
``DistilBERT.call`` accepts a plain ``int32 (B, T)``
token tensor (or a dict with ``input_ids``). (Line numbers deliberately not
quoted: the two this docstring used to cite were both stale after the embedding
stage moved to the shared ``BertEmbeddings``.)

The returned dict is ``{"last_hidden_state": (B, T, hidden), "attention_mask":
(B, T)}``. The mask value was ``None`` when no mask was passed in, until the
echo was resolved at the return site (decisions.md D-062); it is now all ones,
so the contract asserts the key set rather than relying on that value.
"""

import numpy as np

from ..smoke_contract_oracle import (
    assert_contract_rejects_a_broken_forward,
    assert_finite,
)

BATCH, SEQ_LEN, HIDDEN_SIZE, VOCAB_SIZE = 2, 16, 64, 256


def _build():
    from dl_techniques.models.language.distilbert.model import DistilBERT

    return DistilBERT(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=2,
        num_heads=2,
        intermediate_size=128,
        max_position_embeddings=64,
    )


def _inputs():
    return np.random.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN)).astype("int32")


def _assert_contract(out):
    """The smoke assertion. Shared with the meta-test so it is proven falsifiable."""
    assert isinstance(out, dict), f"expected a dict of tensors, got {type(out)}"
    assert set(out) == {"last_hidden_state", "attention_mask"}, sorted(out)
    hidden = out["last_hidden_state"]
    assert tuple(hidden.shape) == (BATCH, SEQ_LEN, HIDDEN_SIZE), tuple(hidden.shape)
    assert_finite(out)


def test_smoke_build_and_forward():
    _assert_contract(_build()(_inputs(), training=False))


def test_the_smoke_contract_rejects_a_broken_forward():
    """RED-proof, in-suite: the assertion above can actually fail.

    Breaks the MODEL -- the built model's forward output is replaced by a
    degenerate one -- rather than the factory's argument validation. See
    ``smoke_contract_oracle`` for why the argument-validation form of this
    meta-test (which yolo12 shipped) proved nothing.

    This package is one of the three carrying the meta-test because its output
    was a dict with a ``None`` value (before D-062), the container shape most
    likely to make an injection silently no-op.
    """
    assert_contract_rejects_a_broken_forward(_build(), _inputs(), _assert_contract)
