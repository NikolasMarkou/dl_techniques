"""Every VLM head refuses a bad argument at construction or build, by name.

Guard for plan-2026-10-14T062616-81938e81 step 11.2 (the user's heads review,
``research/reviews/20261015_heads_review.md`` item VLM-1, the P1 "VLM early
width-check", the per-head reading of ``VLMTaskConfig`` behind VLM-9, and
``BaseVLMHead`` having no ``call``). Measured at ``8b4182f9d``:

* VLM-1: ``VQAHead`` built its attention pooling with a hardcoded
  ``num_heads=8``. ``hidden_size=36`` (or 769) raised from the attention
  factory, ``Failed to create 'multi_head_cross' attention layer ... dim (36)
  must be divisible by num_heads (8)``, a message naming neither ``VQAHead``
  nor ``hidden_size``, and the count could not be changed: ``num_heads=4``
  was refused as an unrecognized keyword.
* Width: ``ImageCaptioningHead`` with 24-wide ``text_features`` and
  ``hidden_size=16`` built, then failed inside ``MultiHeadCrossAttention.call``.
  ``VQAHead`` (attention pooling) with 24-wide ``question_features`` did the
  same; with 24-wide ``vision_features`` it raised ``Query last dimension (24)
  must match dim (16)`` from the attention layer.
* ``VLMTaskConfig.num_classes`` reached only ``VQAHead`` and ``vocab_size``
  only ``ImageCaptioningHead``; the other heads took either and ignored it,
  and a captioning config with no ``vocab_size`` silently got 50000.
* ``BaseVLMHead(task_config=...)`` constructed and raised
  ``NotImplementedError`` only on its first call.
"""

from typing import Any, Dict

import numpy as np
import pytest

from dl_techniques.layers.heads.vlm import (
    BaseVLMHead,
    ImageCaptioningHead,
    ImageTextMatchingHead,
    VisualGroundingHead,
    VLMTaskConfig,
    VLMTaskType,
    VQAHead,
    create_vlm_head,
)

DIM = 16
VQA = VLMTaskType.VISUAL_QUESTION_ANSWERING


def _x(*shape) -> np.ndarray:
    return np.random.default_rng(5).normal(size=shape).astype("float32")


def _vqa_config(hidden_size: int = DIM) -> VLMTaskConfig:
    return VLMTaskConfig(name="vqa", task_type=VQA, hidden_size=hidden_size,
                         num_classes=3)


# ---------------------------------------------------------------------
# VLM-1: VQAHead num_heads
# ---------------------------------------------------------------------

def test_the_vqa_head_takes_its_head_count() -> None:
    head = VQAHead(task_config=_vqa_config(), num_heads=4)
    assert head.num_heads == 4
    assert head.attention_pooling.num_heads == 4
    out = head({"vision_features": _x(2, 5, DIM),
                "question_features": _x(2, 3, DIM)})
    assert tuple(out["answer_logits"].shape) == (2, 3)


def test_the_vqa_head_count_defaults_to_eight_with_attention_pooling() -> None:
    head = VQAHead(task_config=_vqa_config())
    assert head.num_heads == 8
    assert head.attention_pooling.num_heads == 8


@pytest.mark.parametrize("hidden_size", [36, 769])
def test_an_indivisible_width_is_refused_naming_the_head(hidden_size) -> None:
    with pytest.raises(ValueError, match=r"VQAHead.*hidden_size"):
        VQAHead(task_config=_vqa_config(hidden_size))


@pytest.mark.parametrize("bad", [0, -2, True, 2.5, "8"])
def test_a_head_count_that_is_not_a_positive_int_is_refused(bad) -> None:
    with pytest.raises(ValueError, match="num_heads must be a positive integer"):
        VQAHead(task_config=_vqa_config(), num_heads=bad)


@pytest.mark.parametrize("strategy", ["mean", "max"])
def test_a_head_count_without_attention_pooling_is_refused(strategy) -> None:
    with pytest.raises(ValueError, match=r"num_heads.*attention"):
        VQAHead(task_config=_vqa_config(), pooling_strategy=strategy,
                num_heads=4)


@pytest.mark.parametrize("strategy", ["mean", "max"])
def test_pooling_without_attention_needs_no_divisible_width(strategy) -> None:
    head = VQAHead(task_config=_vqa_config(36), pooling_strategy=strategy)
    assert head.num_heads is None
    out = head({"vision_features": _x(2, 5, 36),
                "question_features": _x(2, 3, 36)})
    assert tuple(out["answer_logits"].shape) == (2, 3)


def test_the_head_count_round_trips_through_the_config() -> None:
    head = VQAHead(task_config=_vqa_config(), num_heads=2)
    rebuilt = VQAHead.from_config(head.get_config())
    assert rebuilt.num_heads == 2
    assert rebuilt.attention_pooling.num_heads == 2


# ---------------------------------------------------------------------
# Early width check: a width the call cannot take is refused in build()
# ---------------------------------------------------------------------

def test_a_captioning_text_width_other_than_hidden_size_is_refused_in_build() -> None:
    head = ImageCaptioningHead(
        task_config=VLMTaskConfig(name="cap", task_type=VLMTaskType.IMAGE_CAPTIONING,
                                  hidden_size=DIM, vocab_size=10),
        num_layers=1, num_heads=4)
    with pytest.raises(ValueError, match=r"ImageCaptioningHead.*text_features.*24.*hidden_size"):
        head.build({"vision_features": (2, 5, DIM), "text_features": (2, 4, 24)})


def test_the_captioning_vision_width_stays_free() -> None:
    head = ImageCaptioningHead(
        task_config=VLMTaskConfig(name="cap", task_type=VLMTaskType.IMAGE_CAPTIONING,
                                  hidden_size=DIM, vocab_size=10),
        num_layers=1, num_heads=4)
    out = head({"vision_features": _x(2, 5, 24), "text_features": _x(2, 4, DIM)})
    assert tuple(out["logits"].shape) == (2, 4, 10)


@pytest.mark.parametrize("key", ["vision_features", "question_features"])
def test_a_vqa_attention_width_other_than_hidden_size_is_refused_in_build(key) -> None:
    head = VQAHead(task_config=_vqa_config())
    shapes = {"vision_features": (2, 5, DIM), "question_features": (2, 3, DIM)}
    shapes[key] = shapes[key][:2] + (24,)
    with pytest.raises(ValueError, match=rf"VQAHead.*{key}.*24.*hidden_size"):
        head.build(shapes)


def test_mean_pooling_takes_two_different_widths() -> None:
    head = VQAHead(task_config=_vqa_config(), pooling_strategy="mean")
    out = head({"vision_features": _x(2, 5, 24), "question_features": _x(2, 3, 8)})
    assert tuple(out["answer_logits"].shape) == (2, 3)


# ---------------------------------------------------------------------
# VLM-9, per head: a VLMTaskConfig field the head never reads is refused
# ---------------------------------------------------------------------

#: (head label, task type, the field that head never reads, a value).
_UNREAD = (
    ("captioning", VLMTaskType.IMAGE_CAPTIONING, {"vocab_size": 10}, "num_classes", 3),
    ("vqa", VQA, {"num_classes": 3}, "vocab_size", 10),
    ("grounding", VLMTaskType.VISUAL_GROUNDING, {}, "num_classes", 3),
    ("grounding", VLMTaskType.VISUAL_GROUNDING, {}, "vocab_size", 10),
    ("matching", VLMTaskType.IMAGE_TEXT_MATCHING, {}, "num_classes", 3),
    ("matching", VLMTaskType.IMAGE_TEXT_MATCHING, {}, "vocab_size", 10),
)


@pytest.mark.parametrize("label,task,needed,field,value", _UNREAD,
                         ids=[f"{r[0]}-{r[3]}" for r in _UNREAD])
def test_a_head_refuses_a_config_field_it_never_reads(
        label, task, needed: Dict[str, Any], field, value) -> None:
    config = VLMTaskConfig(name=label, task_type=task, hidden_size=DIM,
                           **needed, **{field: value})
    with pytest.raises(ValueError, match=rf"{field}.*not read"):
        create_vlm_head(config)


def test_a_captioning_head_needs_a_vocabulary() -> None:
    config = VLMTaskConfig(name="cap", task_type=VLMTaskType.IMAGE_CAPTIONING,
                           hidden_size=DIM)
    with pytest.raises(ValueError, match=r"ImageCaptioningHead.*vocab_size"):
        create_vlm_head(config, num_heads=4, num_layers=1)


def test_a_vqa_head_needs_a_class_count() -> None:
    config = VLMTaskConfig(name="vqa", task_type=VQA, hidden_size=DIM)
    with pytest.raises(ValueError, match=r"VQAHead.*num_classes"):
        create_vlm_head(config)


# ---------------------------------------------------------------------
# BaseVLMHead is a base class, not a head
# ---------------------------------------------------------------------

def test_the_base_head_cannot_be_constructed() -> None:
    config = VLMTaskConfig(name="base", task_type=VLMTaskType.VISUAL_GROUNDING,
                           hidden_size=DIM)
    with pytest.raises(TypeError, match=r"BaseVLMHead.*VisualGroundingHead"):
        BaseVLMHead(task_config=config, fusion_strategy="gated")


@pytest.mark.parametrize("cls", [VisualGroundingHead, ImageTextMatchingHead])
def test_both_subclasses_still_construct(cls) -> None:
    task = (VLMTaskType.VISUAL_GROUNDING if cls is VisualGroundingHead
            else VLMTaskType.IMAGE_TEXT_MATCHING)
    head = cls(task_config=VLMTaskConfig(name="s", task_type=task, hidden_size=DIM))
    assert isinstance(head, BaseVLMHead)


# ---------------------------------------------------------------------
# The matching temperature divides the logits, so it must be positive
# ---------------------------------------------------------------------

@pytest.mark.parametrize("bad", [0.0, -0.07, float("inf"), float("nan"), True, "0.07"])
def test_a_temperature_that_is_not_a_positive_number_is_refused(bad) -> None:
    config = VLMTaskConfig(name="itm", task_type=VLMTaskType.IMAGE_TEXT_MATCHING,
                           hidden_size=DIM)
    with pytest.raises(ValueError, match="temperature"):
        ImageTextMatchingHead(task_config=config, temperature=bad)


@pytest.mark.parametrize("strategy", ["mean", "max"])
def test_mean_and_max_take_already_pooled_inputs(strategy) -> None:
    """A rank-2 input passes through the shared pooling helper unchanged.

    Measured at ``8b4182f9d``: each rank-2 input was reduced over its feature
    axis, so the classifier got one ``(4,)`` vector for a batch of 2 and the
    call failed (``Input 0 of layer "hidden_0" ... expected min_ndim=2, found
    ndim=1``); the ``ffn_type`` docstring listed it as a known gap.
    """
    head = VQAHead(task_config=_vqa_config(), pooling_strategy=strategy)
    v, q = _x(2, DIM), _x(2, 8)
    out = head({"vision_features": v, "question_features": q})
    assert tuple(out["answer_logits"].shape) == (2, 3)
