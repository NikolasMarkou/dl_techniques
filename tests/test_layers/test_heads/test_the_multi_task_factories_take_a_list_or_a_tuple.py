"""The three multi-task head factories take a list or a tuple, the same way.

D-122 item 2 (the heads review's A3): ``create_multi_task_head`` (vision),
``create_multi_task_nlp_head`` and ``create_multi_task_vlm_head`` accept a
list or a tuple of tasks with identical semantics, and refuse anything else
with a ``ValueError`` naming the accepted forms. Measured at cfa8ba4bf:

* vision: a tuple raised "Invalid task_configuration type: tuple"; a list
  that named one task twice collapsed to the last entry (its per-task options
  were popped by the first, so the head then failed with ``TypeError: missing
  num_classes``); a list holding an ``int`` raised ``AttributeError``; an
  empty list or dict built a wrapper that returned ``{}`` for any input.
* NLP: a tuple raised "task_configs must be a non-empty dict".
* VLM: a tuple raised ``TypeError: unhashable type: 'VLMTaskConfig'``; a list
  of dicts raised ``AttributeError: 'dict' object has no attribute 'name'``;
  a dict of dicts raised ``AttributeError: ... 'task_type'``; a set raised
  ``AttributeError: 'set' object has no attribute 'items'``.
"""

import pytest

from dl_techniques.layers.heads.nlp import (
    NLPTaskConfig, NLPTaskType, create_multi_task_nlp_head,
)
from dl_techniques.layers.heads.vision.factory import create_multi_task_head
from dl_techniques.layers.heads.vision.task_types import (
    TaskConfiguration, VisionTaskType,
)
from dl_techniques.layers.heads.vlm import (
    VLMTaskConfig, VLMTaskType, create_multi_task_vlm_head,
)

ACCEPTED = "list, tuple, dict"
DIM = 16

VISION_TASKS = [VisionTaskType.DETECTION, VisionTaskType.SEGMENTATION]
VISION_OPTIONS = dict(detection={"num_classes": 3}, segmentation={"num_classes": 4})


def _vision(spec):
    return create_multi_task_head(spec, shared_backbone_dim=DIM, name="mt", **{
        k: dict(v) for k, v in VISION_OPTIONS.items()})


def _nlp_configs():
    return [
        NLPTaskConfig(name="sent", task_type=NLPTaskType.SENTIMENT_ANALYSIS, num_classes=2),
        NLPTaskConfig(name="ner", task_type=NLPTaskType.NAMED_ENTITY_RECOGNITION, num_classes=5),
    ]


def _nlp(spec):
    return create_multi_task_nlp_head(spec, input_dim=DIM, name="mt")


def _vlm_configs():
    return [
        VLMTaskConfig(name="ground", task_type=VLMTaskType.VISUAL_GROUNDING, hidden_size=DIM),
        VLMTaskConfig(name="match", task_type=VLMTaskType.IMAGE_TEXT_MATCHING, hidden_size=DIM),
    ]


def _vlm(spec):
    return create_multi_task_vlm_head(spec, name="mt")


# ---------------------------------------------------------------------
# A list and a tuple build the same head
# ---------------------------------------------------------------------

@pytest.mark.parametrize("build,items", [
    (_vision, VISION_TASKS),
    (_vision, [t.value for t in VISION_TASKS]),
    (_nlp, _nlp_configs()),
    (_nlp, [c.to_dict() for c in _nlp_configs()]),
    (_vlm, _vlm_configs()),
], ids=["vision_enums", "vision_strings", "nlp_configs", "nlp_dicts", "vlm_configs"])
def test_a_tuple_builds_what_the_list_builds(build, items):
    from_list = build(list(items))
    from_tuple = build(tuple(items))
    assert from_tuple.get_config() == from_list.get_config()


@pytest.mark.parametrize("build", [_vision, _nlp, _vlm], ids=["vision", "nlp", "vlm"])
@pytest.mark.parametrize("bad", [{1, 2}, "detection", 3, None], ids=["set", "str", "int", "none"])
def test_any_other_container_is_refused_naming_the_accepted_forms(build, bad):
    with pytest.raises(ValueError, match=ACCEPTED):
        build(bad)


# ---------------------------------------------------------------------
# A sequence is keyed by task, so a repeat is refused, never collapsed
# ---------------------------------------------------------------------

@pytest.mark.parametrize("seq", [list, tuple])
def test_vision_refuses_a_repeated_task(seq):
    with pytest.raises(ValueError, match="detection.*more than once"):
        _vision(seq([VisionTaskType.DETECTION, "detection"]))


@pytest.mark.parametrize("seq", [list, tuple])
def test_nlp_refuses_a_repeated_name(seq):
    first = _nlp_configs()[0]
    with pytest.raises(ValueError, match="sent.*more than once"):
        _nlp(seq([first, first]))


@pytest.mark.parametrize("seq", [list, tuple])
def test_vlm_refuses_a_repeated_name(seq):
    first = _vlm_configs()[0]
    with pytest.raises(ValueError, match="Duplicate task name.*ground"):
        _vlm(seq([first, first]))


def test_vision_refuses_an_entry_that_is_not_a_task():
    with pytest.raises(ValueError, match="VisionTaskType"):
        _vision([VisionTaskType.DETECTION, 3])


@pytest.mark.parametrize("empty", [[], (), {}], ids=["list", "tuple", "dict"])
def test_vision_refuses_an_empty_task_set(empty):
    with pytest.raises(ValueError, match="at least one task"):
        create_multi_task_head(empty)


def test_vision_task_configuration_still_builds():
    # Liveness for the fourth vision form, which is not a container.
    head = create_multi_task_head(
        TaskConfiguration(VISION_TASKS), shared_backbone_dim=DIM,
        **{k: dict(v) for k, v in VISION_OPTIONS.items()})
    assert sorted(head.task_heads) == ["detection", "segmentation"]


# ---------------------------------------------------------------------
# VLM takes dicts of config fields, as NLP and create_vlm_head do
# ---------------------------------------------------------------------

def _vlm_dicts():
    return [
        {"name": "ground", "task_type": "visual_grounding", "hidden_size": DIM},
        {"name": "match", "task_type": "image_text_matching", "hidden_size": DIM},
    ]


def test_vlm_a_dict_of_dicts_builds_what_the_configs_build():
    from_dicts = _vlm({d["name"]: d for d in _vlm_dicts()})
    from_configs = _vlm({c.name: c for c in _vlm_configs()})
    assert from_dicts.get_config() == from_configs.get_config()


def test_vlm_a_list_of_dicts_builds_what_the_configs_build():
    assert _vlm(_vlm_dicts()).get_config() == _vlm(_vlm_configs()).get_config()


def test_vlm_an_entry_that_is_not_a_config_is_refused_by_type():
    with pytest.raises(TypeError, match="VLMTaskConfig"):
        _vlm([3])


# ---------------------------------------------------------------------
# parse_task_list reads every entry, not just the first one's kind
# ---------------------------------------------------------------------
# Measured at cfa8ba4bf: only the first entry decided how a list was read, so
# [DETECTION, 'segmentation'] built a configuration holding DETECTION alone
# (the string dropped in silence) and ['detection', SEGMENTATION] raised
# AttributeError: 'VisionTaskType' object has no attribute 'lower'.

@pytest.mark.parametrize("seq", [list, tuple])
@pytest.mark.parametrize("entries", [
    [VisionTaskType.DETECTION, "segmentation"],
    ["detection", VisionTaskType.SEGMENTATION],
], ids=["enum_first", "string_first"])
def test_parse_task_list_reads_a_mixed_sequence(seq, entries):
    from dl_techniques.layers.heads.vision.task_types import parse_task_list

    config = parse_task_list(seq(entries))
    assert set(config.get_enabled_tasks()) == set(VISION_TASKS)


def test_parse_task_list_refuses_an_entry_that_is_not_a_task():
    from dl_techniques.layers.heads.vision.task_types import parse_task_list

    with pytest.raises(ValueError, match="VisionTaskType"):
        parse_task_list([VisionTaskType.DETECTION, 3])
