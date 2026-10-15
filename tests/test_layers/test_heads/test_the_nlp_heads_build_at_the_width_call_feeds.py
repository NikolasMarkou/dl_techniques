"""Every NLP head builds each stage at the width its ``call`` feeds it.

Guard for plan-2026-10-14T062616-81938e81 step 11.3 (the user's heads review,
``research/reviews/20261015_heads_review.md`` item NLP-1). Measured at
``846589b7f`` with ``input_dim=8`` and ``task_config.hidden_size=16`` over the
six heads and every on/off setting of ``use_intermediate``,
``use_task_attention`` and ``use_ffn``:

* NLP-1 as reported: ``TextClassificationHead`` with pooling, task attention
  on and the intermediate block off built its classifier at ``hidden_size``
  and called it at ``input_dim`` (``expected axis -1 ... value 16``).
  Reproduced. Of the 48 settings, 22 turn on a stage the head's ``call``
  never applies (below); of the other 26, the same root broke 8: every stage
  after the intermediate block was built at ``hidden_size`` whether or not
  anything had mapped the features to that width. Task attention was built with
  ``dim=hidden_size`` and fed ``input_dim`` (token, QA), the FFN was built for
  a ``hidden_size`` input and fed ``input_dim`` (classification, token,
  similarity, generation), and ``MultipleChoiceHead`` reshaped the
  intermediate block's ``hidden_size``-wide rows back into an
  ``input_dim``-wide tensor, which failed with the block's default
  ``use_intermediate=True`` (review item NLP-4; see below).
* Four heads accepted a stage their ``call`` never runs, and built it:
  ``TextClassificationHead`` applied task attention only to a rank-3 tensor,
  which it never holds after pooling; ``TextSimilarityHead``,
  ``TextGenerationHead`` and ``MultipleChoiceHead`` never applied it;
  ``QuestionAnsweringHead`` and ``MultipleChoiceHead`` never applied the FFN.
  The flag still changed the classifier's build width, which is how the
  NLP-1 configuration failed. These are refused now, by head and argument,
  as the vision and VLM heads refuse an argument their ``call`` never reads
  (D-020, D-032).

NLP-4 ("the flatten/reshape path breaks on symbolic shapes") was measured too:
at ``hidden_size == input_dim`` (the default) ``MultipleChoiceHead`` runs
under ``keras.Input`` and ``model.predict`` with a dynamic batch, for both
input ranks. The symbolic claim is refuted; the defect is the width one above,
and ``test_every_nlp_head_reports_the_shapes_it_returns.py`` pins the
symbolic path.
"""

import itertools

import numpy as np
import pytest

from dl_techniques.layers.heads.nlp import NLPTaskConfig, NLPTaskType, create_nlp_head

D, H = 8, 16
B, S, C = 2, 5, 3

_RNG = np.random.default_rng(0)
SEQ = _RNG.normal(size=(B, S, D)).astype("float32")
CHOICES = _RNG.normal(size=(B, C, S, D)).astype("float32")

#: head -> (task config kwargs, input, stages its call applies).
HEADS = {
    "TextClassificationHead": (
        dict(task_type=NLPTaskType.TEXT_CLASSIFICATION, num_classes=3), SEQ, {"ffn"}),
    "TokenClassificationHead": (
        dict(task_type=NLPTaskType.TOKEN_CLASSIFICATION, num_classes=3), SEQ,
        {"task_attention", "ffn"}),
    "QuestionAnsweringHead": (
        dict(task_type=NLPTaskType.QUESTION_ANSWERING), SEQ, {"task_attention"}),
    "TextSimilarityHead": (
        dict(task_type=NLPTaskType.TEXT_SIMILARITY), SEQ, {"ffn"}),
    "TextGenerationHead": (
        dict(task_type=NLPTaskType.TEXT_GENERATION, vocabulary_size=11), SEQ, {"ffn"}),
    "MultipleChoiceHead": (
        dict(task_type=NLPTaskType.MULTIPLE_CHOICE), CHOICES, set()),
}

_FLAGS = list(itertools.product([False, True], repeat=3))
_CASES = [(head, *flags) for head in HEADS for flags in _FLAGS]


def _head(name, use_intermediate, use_task_attention, use_ffn):
    kwargs, _, _ = HEADS[name]
    return create_nlp_head(
        NLPTaskConfig(name="t", hidden_size=H, **kwargs),
        input_dim=D,
        use_intermediate=use_intermediate,
        use_task_attention=use_task_attention,
        use_ffn=use_ffn,
    )


def _shapes(out):
    return {k: tuple(v.shape) for k, v in out.items()}


@pytest.mark.parametrize(
    "name,use_intermediate,use_task_attention,use_ffn", _CASES,
    ids=[f"{c[0]}-inter{int(c[1])}-attn{int(c[2])}-ffn{int(c[3])}" for c in _CASES])
def test_every_accepted_setting_runs_and_every_dead_stage_is_refused(
        name, use_intermediate, use_task_attention, use_ffn) -> None:
    _, x, applied = HEADS[name]
    dead = [flag for flag, on, stage in (
        ("use_task_attention", use_task_attention, "task_attention"),
        ("use_ffn", use_ffn, "ffn"),
    ) if on and stage not in applied]
    if dead:
        with pytest.raises(ValueError, match=rf"{name}.*{dead[0]}"):
            _head(name, use_intermediate, use_task_attention, use_ffn)
        return
    head = _head(name, use_intermediate, use_task_attention, use_ffn)
    out = head(x)
    assert _shapes(out) == {
        k: tuple(v) for k, v in head.compute_output_shape(x.shape).items()}


def test_the_reviewed_configuration_is_refused_by_name() -> None:
    """NLP-1 verbatim (768/512 scaled to 24/16): task attention never runs here."""
    with pytest.raises(ValueError, match="TextClassificationHead.*use_task_attention"):
        create_nlp_head(
            NLPTaskConfig(name="c", task_type=NLPTaskType.TEXT_CLASSIFICATION,
                          num_classes=3, hidden_size=16),
            input_dim=24, use_pooling=True, use_task_attention=True,
            use_intermediate=False)


@pytest.mark.parametrize("use_intermediate,use_ffn,width", [
    (False, False, D), (True, False, H), (False, True, H), (True, True, H)])
def test_the_classifier_is_built_at_the_width_it_is_fed(
        use_intermediate, use_ffn, width) -> None:
    head = _head("TextClassificationHead", use_intermediate, False, use_ffn)
    head(SEQ)
    assert tuple(head.classifier.kernel.shape) == (width, 3)


def test_the_multiple_choice_intermediate_keeps_its_output_width() -> None:
    """NLP-4's real defect: the rows were reshaped back to the input width."""
    head = _head("MultipleChoiceHead", True, False, False)
    out = head(CHOICES)
    assert tuple(out["logits"].shape) == (B, C)
    assert tuple(head.scorer.kernel.shape) == (H, 1)
    pooled = head(CHOICES.mean(axis=2))
    assert tuple(pooled["logits"].shape) == (B, C)


def test_the_default_width_is_unchanged() -> None:
    """With no hidden_size every stage runs at input_dim, as before."""
    head = create_nlp_head(
        NLPTaskConfig(name="t", task_type=NLPTaskType.TOKEN_CLASSIFICATION,
                      num_classes=3),
        input_dim=D, use_task_attention=True, use_ffn=True)
    head(SEQ)
    assert head.task_attention.dim == D
    assert tuple(head.token_classifier.kernel.shape) == (D, 3)
