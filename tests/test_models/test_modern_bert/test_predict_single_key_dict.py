"""`ModernBERT.predict()` on a single-key dict (F-87, decisions.md D-031).

Twin of `tests/test_models/test_fnet/test_predict_single_key_dict.py`. `call`
echoed `attention_mask` verbatim, so omitting it made the entry `None` and
Keras' per-batch output concatenation raised ``ValueError: Structures don't
have the same nested structure``. MEASURED RED at commit ae2e2aa0a.

The second class here pins the part of D-031 that is easy to "clean up" and
must not be: the mask is resolved AT THE RETURN, not before the encoder loop.
Resolving it earlier is NOT a no-op for ModernBERT, because
`WindowAttention._call_grid` zero-pads a rank-2 mask up to its square grid and
thereby masks out grid padding that an absent mask leaves attendable. Measured
2026-08-19 on this fixture: mixed local/global max|delta| = 6.415714e-01
against max|out| = 2.67; all-global max|delta| = 0.000000e+00.

`local_attention_window_size=4` is used throughout: the shipped default of 128
pads every window to 128*128 = 16384 token slots, which does not fit on a
12 GB test GPU at any batch size worth testing.
"""

import numpy as np
import pytest
from keras import ops

from dl_techniques.models.modern_bert.model import ModernBERT


VOCAB = 100
SEQ = 12


def _ids(batch: int = 4) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, VOCAB, size=(batch, SEQ)).astype("int32")


def _model(**overrides) -> ModernBERT:
    kwargs = dict(
        vocab_size=VOCAB,
        hidden_size=32,
        num_layers=2,
        num_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        hidden_dropout_rate=0.0,
        attention_probs_dropout_rate=0.0,
        local_attention_window_size=4,
    )
    kwargs.update(overrides)
    m = ModernBERT(**kwargs)
    m({"input_ids": _ids()})
    return m


@pytest.fixture(scope="module")
def mixed_model() -> ModernBERT:
    """Default schedule: layer 0 local (window), layer 1 global."""
    return _model()


class TestModernBertPredictOnSingleKeyDict:

    def test_predict_accepts_input_ids_alone(self, mixed_model):
        """RED at HEAD: 'Structures don't have the same nested structure'."""
        out = mixed_model.predict({"input_ids": _ids()}, verbose=0)
        assert set(out.keys()) == {"last_hidden_state", "attention_mask"}
        assert out["last_hidden_state"].shape == (4, SEQ, 32)
        assert out["attention_mask"].shape == (4, SEQ)

    def test_echoed_mask_defaults_to_all_ones(self, mixed_model):
        out = mixed_model({"input_ids": _ids()}, training=False)
        mask = ops.convert_to_numpy(out["attention_mask"])
        assert np.array_equal(mask, np.ones((4, SEQ), dtype=mask.dtype))

    def test_a_supplied_mask_is_still_echoed_verbatim(self, mixed_model):
        supplied = np.zeros((4, SEQ), dtype="int32")
        supplied[:, :5] = 1
        out = mixed_model({"input_ids": _ids(), "attention_mask": supplied},
                          training=False)
        np.testing.assert_array_equal(
            ops.convert_to_numpy(out["attention_mask"]), supplied
        )


class TestResolvedMaskMustNotReachTheEncoder:
    """Guards D-031's placement. Both tests go RED if the resolution moves."""

    def test_ones_mask_changes_the_window_layers(self, mixed_model):
        """If this ever measures 0.0 the window-mask mechanism has changed and
        the D-031 argument must be re-derived, not assumed."""
        ids = _ids()
        default = ops.convert_to_numpy(
            mixed_model({"input_ids": ids}, training=False)["last_hidden_state"]
        )
        with_ones = ops.convert_to_numpy(
            mixed_model({"input_ids": ids,
                         "attention_mask": np.ones((4, SEQ), dtype="int32")},
                        training=False)["last_hidden_state"]
        )
        assert np.max(np.abs(default - with_ones)) > 1e-3
        assert np.max(np.abs(default)) > 1e-3  # anti-vacuity

    def test_ones_mask_is_an_exact_no_op_when_every_layer_is_global(self):
        """The global (group_query) path IS mask-invariant, so the difference
        above is attributable to the window layers and nothing else."""
        model = _model(global_attention_interval=1)
        ids = _ids()
        default = ops.convert_to_numpy(
            model({"input_ids": ids}, training=False)["last_hidden_state"]
        )
        with_ones = ops.convert_to_numpy(
            model({"input_ids": ids,
                   "attention_mask": np.ones((4, SEQ), dtype="int32")},
                  training=False)["last_hidden_state"]
        )
        assert np.max(np.abs(default - with_ones)) == 0.0
        assert np.max(np.abs(default)) > 1e-3  # anti-vacuity
