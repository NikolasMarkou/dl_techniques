"""The token and QA heads' task attention attends under the caller's padding mask.

Guard for plan-2026-10-14T062616-81938e81 step 11.4(e) (D-068, a Step 11.3
leftover). Measured at ``56b696fe2``: ``TokenClassificationHead`` and
``QuestionAnsweringHead`` ran ``task_attention(hidden_states)`` with no mask,
so every real token attended over the padded positions. The token head read
only ``inputs['hidden_states']``; the QA head masked its span logits AFTER the
attention had already mixed padding into every real position.

* Flooding the padded positions with 100 must leave every real position's
  output unchanged.
* A mask-free call is the same computation as before: no mask reaches the
  attention layer (the language-model census in the 11.4 record pins the
  headed models' outputs).
* An attention type whose ``call`` takes no ``attention_mask`` (``performer``;
  ``linear`` accepts ``mask=`` and discards it) cannot honour padding, so a
  mask given to it is refused rather than dropped.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.heads.nlp import NLPTaskConfig, NLPTaskType, create_nlp_head

D = 16
B, S = 2, 7
_RNG = np.random.default_rng(11)
SEQ = _RNG.normal(size=(B, S, D)).astype("float32")
MASK = np.ones((B, S), "float32")
MASK[0, 5:] = 0.0
MASK[1, 3:] = 0.0
KEEP = MASK.astype(bool)


def _flood(x):
    out = x.copy()
    out[~KEEP] = 100.0
    return out


def _np(x):
    return keras.ops.convert_to_numpy(x)


def _head(task, attention_type="multi_head", **kwargs):
    extra = {"num_classes": 5} if task is NLPTaskType.NAMED_ENTITY_RECOGNITION else {}
    keras.utils.set_random_seed(3)
    return create_nlp_head(
        NLPTaskConfig(name="t", task_type=task, dropout_rate=0.0, **extra),
        input_dim=D, use_task_attention=True, attention_type=attention_type, **kwargs)


_TASKS = [NLPTaskType.NAMED_ENTITY_RECOGNITION, NLPTaskType.QUESTION_ANSWERING]
_KEYS = {NLPTaskType.NAMED_ENTITY_RECOGNITION: ("logits",),
         NLPTaskType.QUESTION_ANSWERING: ("start_logits", "end_logits")}


@pytest.mark.parametrize("use_intermediate", [False, True], ids=["direct", "intermediate"])
@pytest.mark.parametrize("task", _TASKS, ids=lambda t: t.name)
def test_padding_never_reaches_a_real_position(task, use_intermediate) -> None:
    head = _head(task, use_intermediate=use_intermediate)
    clean = head({"hidden_states": SEQ, "attention_mask": MASK}, training=False)
    flooded = head({"hidden_states": _flood(SEQ), "attention_mask": MASK}, training=False)
    for key in _KEYS[task]:
        np.testing.assert_allclose(_np(flooded[key])[KEEP], _np(clean[key])[KEEP],
                                   rtol=0, atol=1e-5, err_msg=key)


@pytest.mark.parametrize("task", _TASKS, ids=lambda t: t.name)
def test_the_mask_changes_what_a_real_position_sees(task) -> None:
    """Liveness: with padding present, masked and unmasked attention differ."""
    head = _head(task)
    masked = head({"hidden_states": SEQ, "attention_mask": MASK}, training=False)
    unmasked = head(SEQ, training=False)
    key = _KEYS[task][0]
    assert np.max(np.abs(_np(masked[key])[KEEP] - _np(unmasked[key])[KEEP])) > 1e-4


@pytest.mark.parametrize("task", _TASKS, ids=lambda t: t.name)
def test_a_mask_free_call_passes_no_mask_to_the_attention(task, monkeypatch) -> None:
    head = _head(task)
    head(SEQ, training=False)
    seen = []
    layer_call = type(head.task_attention).call

    def recording(self, *args, **kwargs):
        seen.append(kwargs.get("attention_mask"))
        return layer_call(self, *args, **kwargs)

    monkeypatch.setattr(type(head.task_attention), "call", recording)
    head(SEQ, training=False)
    head({"hidden_states": SEQ}, training=False)
    assert seen == [None, None]


@pytest.mark.parametrize("task", _TASKS, ids=lambda t: t.name)
def test_an_attention_that_cannot_mask_refuses_a_mask(task) -> None:
    head = _head(task, attention_type="performer")
    head(SEQ, training=False)  # no mask: runs
    with pytest.raises(ValueError, match="performer"):
        head({"hidden_states": SEQ, "attention_mask": MASK}, training=False)
