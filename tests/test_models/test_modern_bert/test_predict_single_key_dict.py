"""`ModernBERT.predict()` on a single-key dict (F-87, decisions.md D-031).

Twin of `tests/test_models/test_fnet/test_predict_single_key_dict.py`. `call`
echoed `attention_mask` verbatim, so omitting it made the entry `None` and
Keras' per-batch output concatenation raised ``ValueError: Structures don't
have the same nested structure``. MEASURED RED at commit ae2e2aa0a.

The second class here pins the part of D-031 that is easy to "clean up" and
must not be: the mask is resolved AT THE RETURN, not before the encoder loop,
so the OUTPUT structure never depends on whether the caller supplied a mask.

That class also used to pin a DEFECT as a positive claim -- see its own
docstring. Until 2026-08-25 an all-ones mask moved the output by 6.415714e-01
on this fixture, because the window layers zero-padded a rank-2 mask up to a
square grid. It measures exactly 0.0 now, and the test asserts that instead.

`local_attention_window_size=4` is used throughout. Since 2026-08-25 that is a
1-D band FULL SPAN (the layer receives the half-width, 2); before it was a 2-D
edge length, and the shipped default of 128 padded every window to
128*128 = 16384 token slots, which did not fit on a 12 GB test GPU at any batch
size worth testing.
"""

import numpy as np
import pytest
from keras import ops

from dl_techniques.models.language.modern_bert.model import ModernBERT


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
    """Guards D-031's placement, and pins the mask no-op law.

    **This class asserted the OPPOSITE until 2026-08-25.** Its first test was
    named ``test_ones_mask_changes_the_window_layers`` and required
    ``max|delta| > 1e-3``, i.e. it pinned a DEFECT as a positive claim: an
    all-ones ``(B, N)`` mask masks no real token and is therefore a
    mathematical no-op, but ``WindowAttention._call_grid`` zero-padded a rank-2
    mask up to its square grid and so masked out grid padding that an absent
    mask left attendable. Two independent things retired that claim
    (plan-2026-08-25T053412-0f1fa04f):

    * D-007/D-009/D-011 closed the pad leak in every partition mode, so an
      all-ones mask is now bit-exact in ``'grid'`` and ``'zigzag'`` too.
    * D-012 routed ModernBERT's local layers to ``'window_band'``, which pads
      nothing at all, so there is no grid padding left for a mask to reach.

    The test is CORRECTED rather than deleted, and inverted rather than
    relaxed: the same measurement is taken, and the bound is now ``== 0.0``.
    That is a strictly stronger guard than the old one, and it goes RED if
    anything ever reintroduces a padded slot into a ModernBERT local layer.

    D-031's DECISION -- resolve the echoed mask AT THE RETURN, not before the
    encoder loop -- is untouched and still guarded, on its FIRST reason: the
    OUTPUT structure must not depend on whether the caller supplied a mask.
    Only D-031's SECOND reason (that the pre-loop placement is a silent
    numerics change) has lapsed.
    """

    def test_ones_mask_is_an_exact_no_op_in_the_local_layers(self, mixed_model):
        """MEASURED 2026-08-25: 0.0, where the pre-fix code measured
        6.415714e-01 on this same fixture against a max|out| of 2.67."""
        ids = _ids()
        default = ops.convert_to_numpy(
            mixed_model({"input_ids": ids}, training=False)["last_hidden_state"]
        )
        with_ones = ops.convert_to_numpy(
            mixed_model({"input_ids": ids,
                         "attention_mask": np.ones((4, SEQ), dtype="int32")},
                        training=False)["last_hidden_state"]
        )
        assert np.max(np.abs(default - with_ones)) == 0.0
        assert np.max(np.abs(default)) > 1e-3  # anti-vacuity

    def test_a_real_mask_still_reaches_the_local_layers(self, mixed_model):
        """Anti-vacuity for the test above: the no-op result must come from the
        mask being a no-op, NOT from the mask being ignored. A mask that
        actually masks something must move the output."""
        ids = _ids()
        default = ops.convert_to_numpy(
            mixed_model({"input_ids": ids}, training=False)["last_hidden_state"]
        )
        partial = np.ones((4, SEQ), dtype="int32")
        partial[:, SEQ // 2:] = 0
        masked = ops.convert_to_numpy(
            mixed_model({"input_ids": ids, "attention_mask": partial},
                        training=False)["last_hidden_state"]
        )
        assert np.max(np.abs(default - masked)) > 1e-3

    def test_ones_mask_is_an_exact_no_op_when_every_layer_is_global(self):
        """The global (group_query) path was ALWAYS mask-invariant. It is kept
        as the control: before 2026-08-25 it was the only schedule that measured
        0.0, which is what attributed the delta to the window layers. Both
        schedules measure 0.0 now."""
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
