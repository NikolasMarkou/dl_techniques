"""The heads read an input's rank once, statically, and refuse an unknown one.

Guard for plan-2026-10-14T062616-81938e81 step 11.2 (the user's heads review,
``research/reviews/20261015_heads_review.md`` item VLM-2, and the NLP
validation gap with the same text). The review called
``len(ops.shape(x))`` "fragile under tracing". Measured at ``8b4182f9d``:

* For every input of known rank, ``len(ops.shape(x))`` and
  ``keras.ops.ndim(x)`` give the same integer, eagerly, under ``tf.function``
  and on a ``keras.Input``. The claim, as a behaviour at a known rank, is
  refuted.
* At an UNKNOWN rank (a ``tf.function`` with ``TensorSpec(None)``) they
  differ, and neither is right: ``len(ops.shape(x))`` raises
  ``All tensors passed to ops.shape must have a statically known rank``,
  which names no head, and ``keras.ops.ndim(x)`` returns ``-1``, which would
  send ``ImageTextMatchingHead`` down its rank-2 branch without a word.

So the heads read the rank through one helper,
``layers/heads/validation.static_rank``, which uses ``keras.ops.ndim`` and
refuses an unknown rank by head and input name. The source scan below keeps
the old spelling out of all three factories. The NLP factory joined it in plan
step 11.3: its five ``len(ops.shape(...))`` sites (``TextClassificationHead``
twice, ``TextSimilarityHead`` twice, ``MultipleChoiceHead`` once, re-derived at
``846589b7f``) read through ``static_rank`` now, and one of them, the task
attention test in ``TextClassificationHead.call``, is gone with the branch it
guarded (D-062).
"""

import pathlib
import re

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.heads.nlp import (
    NLPTaskConfig,
    NLPTaskType,
    create_nlp_head,
)
from dl_techniques.layers.heads.vlm import (
    ImageTextMatchingHead,
    VLMTaskConfig,
    VLMTaskType,
)

HEADS = pathlib.Path(__file__).resolve().parents[3] / "src/dl_techniques/layers/heads"

#: The factories whose rank reads go through ``static_rank``: all three.
SCANNED = ("nlp", "vlm", "vision")

_OLD_SPELLING = re.compile(r"len\(\s*(?:keras\.)?ops\.shape\(")


@pytest.mark.parametrize("domain", SCANNED)
def test_no_factory_reads_a_rank_through_len_of_shape(domain) -> None:
    source = (HEADS / domain / "factory.py").read_text()
    hits = [i + 1 for i, line in enumerate(source.splitlines())
            if _OLD_SPELLING.search(line)]
    assert not hits, f"{domain}/factory.py reads a rank with len(ops.shape(...)) at lines {hits}"


def test_the_scan_sees_the_spelling_it_forbids() -> None:
    assert _OLD_SPELLING.search("if len(ops.shape(x)) == 3:")
    assert _OLD_SPELLING.search("if len(keras.ops.shape(x)) == 3:")
    assert not _OLD_SPELLING.search("if keras.ops.ndim(x) == 3:")


def test_an_unknown_rank_is_refused_naming_the_head() -> None:
    head = ImageTextMatchingHead(task_config=VLMTaskConfig(
        name="itm", task_type=VLMTaskType.IMAGE_TEXT_MATCHING, hidden_size=16))
    x = np.zeros((2, 5, 16), "float32")
    head({"vision_features": x, "text_features": x})

    @tf.function(input_signature=[tf.TensorSpec(None, tf.float32)])
    def run(v):
        return head.call({"vision_features": v, "text_features": tf.constant(x)})["match_score"]

    with pytest.raises(ValueError, match=r"ImageTextMatchingHead.*vision_features.*rank"):
        run(tf.constant(x))


def test_a_known_rank_still_takes_both_branches() -> None:
    head = ImageTextMatchingHead(task_config=VLMTaskConfig(
        name="itm", task_type=VLMTaskType.IMAGE_TEXT_MATCHING, hidden_size=16))
    seq = np.random.default_rng(0).normal(size=(2, 5, 16)).astype("float32")
    pooled = seq.mean(axis=1)
    a = head({"vision_features": seq, "text_features": seq})["match_score"]
    b = head({"vision_features": pooled, "text_features": pooled})["match_score"]
    np.testing.assert_allclose(keras.ops.convert_to_numpy(a),
                               keras.ops.convert_to_numpy(b), rtol=0, atol=1e-5)


_NLP_HEADS = {
    "TextClassificationHead": (
        dict(task_type=NLPTaskType.TEXT_CLASSIFICATION, num_classes=3), (2, 5, 16)),
    "TextSimilarityHead": (dict(task_type=NLPTaskType.TEXT_SIMILARITY), (2, 5, 16)),
    "MultipleChoiceHead": (dict(task_type=NLPTaskType.MULTIPLE_CHOICE), (2, 3, 5, 16)),
}


@pytest.mark.parametrize("name", sorted(_NLP_HEADS))
def test_an_unknown_rank_is_refused_by_the_nlp_heads(name) -> None:
    kwargs, shape = _NLP_HEADS[name]
    head = create_nlp_head(NLPTaskConfig(name="t", **kwargs), input_dim=16)
    x = np.zeros(shape, "float32")
    head(x)

    @tf.function(input_signature=[tf.TensorSpec(None, tf.float32)])
    def run(v):
        return head.call(v)

    with pytest.raises(ValueError, match=rf"{name}.*hidden_states.*rank"):
        run(tf.constant(x))
