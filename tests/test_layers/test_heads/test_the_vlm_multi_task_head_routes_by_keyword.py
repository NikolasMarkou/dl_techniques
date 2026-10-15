"""``MultiTaskVLMHead`` takes ``task_name`` by keyword and refuses a degenerate set.

Guard for plan-2026-10-14T062616-81938e81 step 11.2 (the user's heads review,
``research/reviews/20261015_heads_review.md`` items VLM-7 and VLM-8).
Measured at ``8b4182f9d``:

* VLM-8: ``call(inputs, task_name=None, training=None)`` put ``task_name``
  in the second positional slot, where every other head takes ``training``.
  Through ``Layer.__call__`` a Python ``bool`` there is refused by Keras
  itself ("Only input tensors may be passed as positional arguments"), so
  that path was not reachable. Two paths were: ``head.call(inputs, True)``
  raised ``ValueError: Unknown task: True``, and a tensor-valued flag
  ``head(inputs, keras.ops.convert_to_tensor(True))`` landed in ``task_name``
  and raised ``TypeError: Tensor is unhashable``.
* VLM-7: ``create_multi_task_vlm_head`` keyed a list by each config's
  ``name``, so two configs with one name silently collapsed to the last one
  (an ITM and a grounding config named ``'itm'`` built one
  ``VisualGroundingHead``). ``VLMTaskConfiguration`` refuses duplicates. An
  empty list or dict built a wrapper that returned ``{}`` for any input.

Not a defect, measured and left alone: a dict holding two configs that share
a ``name`` builds two heads with one layer name; a functional model holding
it saves and reloads with a max output difference of 0.0, because Keras
tracks the heads by attribute path (``head_<task>``), not by layer name.
"""

import inspect

import keras
import numpy as np
import pytest

from dl_techniques.layers.heads.vlm import (
    MultiTaskVLMHead,
    VLMTaskConfig,
    VLMTaskType,
    create_multi_task_vlm_head,
)

DIM = 16


def _config(name: str, task: VLMTaskType = VLMTaskType.IMAGE_TEXT_MATCHING):
    return VLMTaskConfig(name=name, task_type=task, hidden_size=DIM)


def _payload():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(2, 5, DIM)).astype("float32")
    return {"vision_features": x, "text_features": x[:, ::-1].copy()}


# ---------------------------------------------------------------------
# VLM-8: training keeps its positional slot
# ---------------------------------------------------------------------

def test_task_name_is_keyword_only() -> None:
    params = inspect.signature(MultiTaskVLMHead.call).parameters
    assert params["task_name"].kind is inspect.Parameter.KEYWORD_ONLY
    assert list(params)[:3] == ["self", "inputs", "training"]


def test_a_positional_flag_is_the_training_flag() -> None:
    head = create_multi_task_vlm_head([_config("itm"), _config("grd", VLMTaskType.VISUAL_GROUNDING)])
    payload = _payload()
    head(payload)
    out = head.call(payload, True)
    assert sorted(out) == ["grd", "itm"]


def test_a_tensor_flag_in_the_positional_slot_is_treated_as_on_every_head() -> None:
    """Keras binds a positional flag to ``training`` and then adds its own.

    Measured after the fix: ``ImageTextMatchingHead(...)(inputs, flag)`` with a
    tensor flag raises ``TypeError: ... got multiple values for argument
    'training'`` from Keras' ``__call__``; the wrapper now raises the same,
    where it used to read the flag as a task name.
    """
    flag = keras.ops.convert_to_tensor(True)
    single = create_multi_task_vlm_head([_config("itm")]).task_heads["itm"]
    with pytest.raises(TypeError, match="multiple values for argument 'training'"):
        single(_payload(), flag)
    head = create_multi_task_vlm_head([_config("itm")])
    with pytest.raises(TypeError, match="multiple values for argument 'training'"):
        head(_payload(), flag)


def test_task_name_still_selects_one_head() -> None:
    head = create_multi_task_vlm_head([_config("itm"), _config("grd", VLMTaskType.VISUAL_GROUNDING)])
    out = head(_payload(), task_name="grd")
    assert sorted(out) == ["bbox", "confidence", "grounded_features"]


@pytest.mark.parametrize("bad", ["bogus", ""])
def test_an_unknown_task_name_is_refused(bad) -> None:
    head = create_multi_task_vlm_head([_config("itm")])
    with pytest.raises(ValueError, match="Unknown task"):
        head(_payload(), task_name=bad)


# ---------------------------------------------------------------------
# VLM-7: duplicates and the empty set
# ---------------------------------------------------------------------

def test_a_list_with_a_repeated_name_is_refused() -> None:
    with pytest.raises(ValueError, match=r"[Dd]uplicate.*'itm'"):
        create_multi_task_vlm_head([
            _config("itm"), _config("itm", VLMTaskType.VISUAL_GROUNDING)])


@pytest.mark.parametrize("empty", [[], {}], ids=["list", "dict"])
def test_an_empty_task_set_is_refused(empty) -> None:
    with pytest.raises(ValueError, match="[Aa]t least one task"):
        create_multi_task_vlm_head(empty)


def test_the_wrapper_itself_refuses_an_empty_task_set() -> None:
    with pytest.raises(ValueError, match="[Aa]t least one task"):
        MultiTaskVLMHead(task_configs={})


def test_a_list_of_distinct_names_keeps_every_head() -> None:
    head = create_multi_task_vlm_head([
        _config("itm"), _config("grd", VLMTaskType.VISUAL_GROUNDING)])
    assert sorted(head.task_heads) == ["grd", "itm"]
