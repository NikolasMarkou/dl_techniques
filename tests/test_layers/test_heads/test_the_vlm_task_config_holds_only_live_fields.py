"""``VLMTaskConfig`` holds only fields a head reads, and parses its task type.

Guard for plan-2026-10-14T062616-81938e81 step 11.2 (the user's heads review,
``research/reviews/20261015_heads_review.md`` items VLM-6 and VLM-9).
Measured at ``8b4182f9d``:

* VLM-9: 11 of the 17 fields reached no head: ``max_text_length``,
  ``vision_hidden_size``, ``text_hidden_size``, ``fusion_hidden_size``,
  ``use_cross_attention``, ``fusion_type``, ``pooling_type``,
  ``use_task_specific_heads``, ``temperature``, ``beam_size``,
  ``loss_weight``. ``fusion_type``/``pooling_type`` even carried their own
  vocabularies (``"concat"``, ``"avg"``) next to the heads' real
  ``fusion_strategy``/``pooling_strategy`` arguments. ``__post_init__`` filled
  ``num_classes`` for ``MULTIMODAL_SENTIMENT_ANALYSIS`` (3),
  ``MULTIMODAL_EMOTION_RECOGNITION`` (7) and ``VISUAL_ENTAILMENT`` (3), three
  task types with no head, and never for ``VISUAL_QUESTION_ANSWERING``, the
  one head that reads it.
* VLM-6: ``VLMTaskConfig(task_type="image_text_matching")`` stored the string,
  so ``create_vlm_head({... 'task_type': 'image_text_matching'})`` died in
  ``get_head_class`` with ``AttributeError: 'str' object has no attribute
  'name'``, while ``from_config`` parsed the same string.
* No field was validated: ``hidden_size=0`` and ``dropout_rate=1.5`` built a
  config.
"""

import dataclasses

import pytest

from dl_techniques.layers.heads.vlm import (
    ImageTextMatchingHead,
    VLMTaskConfig,
    VLMTaskType,
    create_vlm_head,
)

#: The fields some head reads: ``name`` (layer name), ``task_type``
#: (dispatch), ``hidden_size`` (every working width), ``dropout_rate`` (every
#: head), ``vocab_size`` (captioning), ``num_classes`` (VQA).
LIVE_FIELDS = {"name", "task_type", "vocab_size", "hidden_size",
               "dropout_rate", "num_classes"}

DEAD_FIELDS = ("max_text_length", "vision_hidden_size", "text_hidden_size",
               "fusion_hidden_size", "use_cross_attention", "fusion_type",
               "pooling_type", "use_task_specific_heads", "temperature",
               "beam_size", "loss_weight")


def test_the_config_holds_exactly_the_live_fields() -> None:
    names = {f.name for f in dataclasses.fields(VLMTaskConfig)}
    assert names == LIVE_FIELDS, (
        f"extra: {sorted(names - LIVE_FIELDS)}, missing: {sorted(LIVE_FIELDS - names)}")


@pytest.mark.parametrize("field", DEAD_FIELDS)
def test_a_deleted_field_is_refused(field) -> None:
    with pytest.raises(TypeError, match=field):
        VLMTaskConfig(name="x", task_type=VLMTaskType.IMAGE_TEXT_MATCHING,
                      **{field: 1})


@pytest.mark.parametrize("task", list(VLMTaskType), ids=lambda t: t.value)
def test_no_task_type_gets_an_invented_class_count(task) -> None:
    assert VLMTaskConfig(name="x", task_type=task).num_classes is None


def test_the_vocabulary_is_unset_by_default() -> None:
    config = VLMTaskConfig(name="x", task_type=VLMTaskType.IMAGE_CAPTIONING)
    assert config.vocab_size is None


@pytest.mark.parametrize("text", ["image_text_matching", " Image_Text_Matching "])
def test_a_string_task_type_is_parsed(text) -> None:
    config = VLMTaskConfig(name="x", task_type=text)
    assert config.task_type is VLMTaskType.IMAGE_TEXT_MATCHING


def test_the_dict_path_of_the_factory_takes_a_string_task_type() -> None:
    head = create_vlm_head({"name": "m", "task_type": "image_text_matching",
                            "hidden_size": 16})
    assert isinstance(head, ImageTextMatchingHead)
    assert head.task_config.task_type is VLMTaskType.IMAGE_TEXT_MATCHING


def test_an_unknown_task_string_is_refused_listing_the_valid_ones() -> None:
    with pytest.raises(ValueError, match="image_text_matching"):
        VLMTaskConfig(name="x", task_type="bogus_task")


@pytest.mark.parametrize("bad", [3, None, ["image_captioning"]])
def test_a_task_type_that_is_neither_enum_nor_string_is_refused(bad) -> None:
    with pytest.raises(TypeError, match="task_type"):
        VLMTaskConfig(name="x", task_type=bad)


@pytest.mark.parametrize("field,bad", [
    ("hidden_size", 0), ("hidden_size", -8), ("hidden_size", True),
    ("hidden_size", 16.0), ("hidden_size", None),
    ("vocab_size", 0), ("vocab_size", True),
    ("num_classes", 0), ("num_classes", 2.5),
])
def test_a_count_that_is_not_a_positive_int_is_refused_by_name(field, bad) -> None:
    with pytest.raises(ValueError, match=rf"{field} must be a positive integer"):
        VLMTaskConfig(name="x", task_type=VLMTaskType.IMAGE_CAPTIONING,
                      **{field: bad})


@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5, True, "0.1"])
def test_a_dropout_rate_outside_the_unit_interval_is_refused(bad) -> None:
    with pytest.raises(ValueError, match="dropout_rate"):
        VLMTaskConfig(name="x", task_type=VLMTaskType.IMAGE_CAPTIONING,
                      dropout_rate=bad)


@pytest.mark.parametrize("bad", ["", 3, None])
def test_a_name_that_is_not_a_non_empty_string_is_refused(bad) -> None:
    with pytest.raises(ValueError, match="name"):
        VLMTaskConfig(name=bad, task_type=VLMTaskType.IMAGE_CAPTIONING)
