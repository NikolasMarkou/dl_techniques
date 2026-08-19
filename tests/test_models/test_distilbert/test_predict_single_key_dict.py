"""`DistilBERT.predict()` must accept a single-key input dict (D-062).

Third and last member of the F-87 family. `call` echoed `attention_mask` back
into its output dict verbatim, so when the caller omitted the key the entry was
`None` and Keras' per-batch output concatenation raised ``ValueError:
Structures don't have the same nested structure``. MEASURED RED at commit
9d71a8c4d with exactly the fixture below -- the sibling repair (D-031, step 11)
fixed `fnet` and `modern_bert` and explicitly left this one, and
`test_model.py` was pinning `... is None` as a contract.

The repair resolves the echoed mask to `ones_like(input_ids)` AT THE RETURN, so
the output structure is input-independent while the encoder still sees `None`.
For DistilBERT the placement is numerically irrelevant -- an all-ones mask is
an exact no-op through the whole model, measured 0.000000e+00 against
max|out| = 2.758002e+00, and pinned below with an anti-vacuity control that
shows a NON-trivial mask does change the output. The placement is nonetheless
kept at the return, because the same edit before the encoder loop measured
6.415714e-01 in `modern_bert` (D-031) and a per-model rule is a trap.
"""

import numpy as np
import pytest
from keras import ops

from dl_techniques.models.distilbert.model import DistilBERT


VOCAB = 100
SEQ = 12


def _ids(batch: int = 4) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(1, VOCAB, size=(batch, SEQ)).astype("int32")


@pytest.fixture(scope="module")
def model() -> DistilBERT:
    m = DistilBERT(
        vocab_size=VOCAB,
        hidden_size=32,
        num_layers=2,
        num_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        dropout_rate=0.0,
        attention_dropout_rate=0.0,
    )
    m({"input_ids": _ids()})
    return m


class TestDistilBertPredictOnSingleKeyDict:

    def test_predict_accepts_input_ids_alone(self, model):
        """RED at 9d71a8c4d: 'Structures don't have the same nested structure'."""
        out = model.predict({"input_ids": _ids()}, verbose=0)
        assert set(out.keys()) == {"last_hidden_state", "attention_mask"}
        assert out["last_hidden_state"].shape == (4, SEQ, 32)
        assert out["attention_mask"].shape == (4, SEQ)

    def test_echoed_mask_defaults_to_all_ones(self, model):
        out = model({"input_ids": _ids()}, training=False)
        mask = ops.convert_to_numpy(out["attention_mask"])
        assert np.array_equal(mask, np.ones((4, SEQ), dtype=mask.dtype))

    def test_a_supplied_mask_is_still_echoed_verbatim(self, model):
        supplied = np.zeros((4, SEQ), dtype="int32")
        supplied[:, :5] = 1
        out = model({"input_ids": _ids(), "attention_mask": supplied},
                    training=False)
        np.testing.assert_array_equal(
            ops.convert_to_numpy(out["attention_mask"]), supplied
        )


class TestResolvedMaskIsNumericallyInert:
    """Justifies `ones_like` as the resolved default for THIS model."""

    def test_ones_mask_is_an_exact_no_op(self, model):
        ids = _ids()
        a = ops.convert_to_numpy(
            model({"input_ids": ids}, training=False)["last_hidden_state"]
        )
        b = ops.convert_to_numpy(
            model({"input_ids": ids,
                   "attention_mask": np.ones((4, SEQ), dtype="int32")},
                  training=False)["last_hidden_state"]
        )
        assert np.max(np.abs(a - b)) == 0.0
        assert np.max(np.abs(a)) > 1e-3  # anti-vacuity: output is not constant

    def test_the_instrument_can_see_a_mask_at_all(self, model):
        """Anti-vacuity for the test above: without this, a model that ignored
        `attention_mask` entirely would pass it trivially."""
        ids = _ids()
        partial = np.zeros((4, SEQ), dtype="int32")
        partial[:, :5] = 1
        a = ops.convert_to_numpy(
            model({"input_ids": ids}, training=False)["last_hidden_state"]
        )
        c = ops.convert_to_numpy(
            model({"input_ids": ids, "attention_mask": partial},
                  training=False)["last_hidden_state"]
        )
        assert np.max(np.abs(a - c)) > 1e-4
