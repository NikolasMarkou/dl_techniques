"""A vision head refuses an attention or FFN setting for a stage that is off.

Guard for plan-2026-10-14T062616-81938e81 iteration 2 step 6.6 (B12), D-030's
rule (as D-074 applied it to ``bbox_dims`` and ``hidden_dim``). Measured at
3a89046f8 (``scratchpad/iter2/b12.py``): ``use_attention=False,
attention_type='differential'`` and ``use_ffn=False, ffn_type='swiglu',
ffn_expansion_factor=7`` were accepted by the classification, segmentation and
depth heads, stored, and written to ``get_config``, while no layer read them.
Every ``HeadConfiguration`` preset passed ``ffn_type``/``ffn_expansion_factor``
to the classification and enhancement heads, whose FFN is always off.

Now the three knobs default to ``None``. With the stage on, ``None`` resolves
to ``'multi_head'``, ``'mlp'`` and ``4`` as before; with it off, a value
raises ``ValueError`` and ``None`` is stored and round-trips. The presets and
``build_model_vision_head`` pass no knob for a stage that ends up off, unless
the model or the caller set it.
"""

import pytest

from dl_techniques.layers.heads.vision.factory import (
    BaseVisionHead, HeadConfiguration, VISION_HEAD_DISPATCH, VisionTaskType,
    build_model_vision_head, create_vision_head)

#: Arguments each task's head needs to build, beyond the stage flags.
_REQUIRED = {
    VisionTaskType.DETECTION: {"num_classes": 3},
    VisionTaskType.SEGMENTATION: {"num_classes": 3},
    VisionTaskType.INSTANCE_SEGMENTATION: {"num_classes": 3},
    VisionTaskType.CLASSIFICATION: {"num_classes": 3},
    VisionTaskType.KEYPOINT_DETECTION: {"num_classes": 3},
}
TASKS = sorted(VISION_HEAD_DISPATCH, key=lambda t: t.value)
IDS = [t.value for t in TASKS]

_OFF_KNOBS = [
    ("use_attention", "attention_type", "cbam"),
    ("use_attention", "attention_type", "multi_head"),
    ("use_ffn", "ffn_type", "swiglu"),
    ("use_ffn", "ffn_type", "mlp"),
    ("use_ffn", "ffn_expansion_factor", 7),
    ("use_ffn", "ffn_expansion_factor", 4),
]


def _build(task, **kwargs):
    return create_vision_head(task, **_REQUIRED.get(task, {}), **kwargs)


def test_every_dispatched_head_is_a_base_vision_head():
    # Anti-vacuity: the parametrization reaches every head class.
    assert len(TASKS) == 10
    assert all(issubclass(cls, BaseVisionHead) for cls, _ in VISION_HEAD_DISPATCH.values())


@pytest.mark.parametrize("flag,knob,value", _OFF_KNOBS, ids=[f"{k}={v}" for _, k, v in _OFF_KNOBS])
@pytest.mark.parametrize("task", TASKS, ids=IDS)
def test_a_knob_for_a_stage_that_is_off_is_refused(task, flag, knob, value):
    # The default value is refused too: stored, it would still be dead.
    with pytest.raises(ValueError, match=knob):
        _build(task, **{flag: False, knob: value})


@pytest.mark.parametrize("task", TASKS, ids=IDS)
def test_a_head_with_both_stages_off_stores_none_and_round_trips(task):
    head = _build(task, use_attention=False, use_ffn=False)
    config = head.get_config()
    for knob in ("attention_type", "ffn_type", "ffn_expansion_factor"):
        assert getattr(head, knob) is None and config[knob] is None, knob
    rebuilt = type(head).from_config(config)
    assert rebuilt.get_config() == config


_ON_TASKS = [VisionTaskType.DETECTION, VisionTaskType.SEGMENTATION, VisionTaskType.DEPTH_ESTIMATION]


@pytest.mark.parametrize("task", _ON_TASKS, ids=[t.value for t in _ON_TASKS])
def test_a_stage_that_is_on_resolves_none_to_the_old_default(task):
    head = _build(task, use_attention=True, use_ffn=True, hidden_dim=16)
    assert (head.attention_type, head.ffn_type, head.ffn_expansion_factor) == ("multi_head", "mlp", 4)


@pytest.mark.parametrize("task", _ON_TASKS, ids=[t.value for t in _ON_TASKS])
def test_a_stage_that_is_on_takes_the_given_knobs(task):
    head = _build(task, use_attention=True, attention_type="cbam", use_ffn=True,
                  ffn_type="glu", ffn_expansion_factor=2, hidden_dim=16)
    assert (head.attention_type, head.ffn_type, head.ffn_expansion_factor) == ("cbam", "glu", 2)
    assert type(head.attention).__name__ == "CBAM"


@pytest.mark.parametrize("preset", ["get_default_config", "get_efficient_config"])
@pytest.mark.parametrize("task", TASKS, ids=IDS)
def test_no_preset_passes_a_knob_for_a_stage_it_turns_off(task, preset):
    config = getattr(HeadConfiguration, preset)(task)
    if not config.get("use_attention", False):
        assert "attention_type" not in config
    if not config.get("use_ffn", True):
        assert "ffn_type" not in config and "ffn_expansion_factor" not in config
    _build(task, **{k: v for k, v in config.items() if k not in _REQUIRED.get(task, {})})


def test_a_model_that_turns_a_preset_stage_off_still_builds():
    # The preset's FFN knobs are dropped with the FFN the model turned off.
    head = build_model_vision_head(
        VisionTaskType.DETECTION, {"num_classes": 3, "use_ffn": False})
    assert head.use_ffn is False and head.ffn_type is None


@pytest.mark.parametrize("source", ["head_config", "head_config_overrides"])
def test_a_model_or_caller_knob_for_an_off_stage_is_still_refused(source):
    knobs = {"use_ffn": False, "ffn_type": "swiglu"}
    head_config = {"num_classes": 3, **(knobs if source == "head_config" else {})}
    overrides = knobs if source == "head_config_overrides" else None
    with pytest.raises(ValueError, match="ffn_type"):
        build_model_vision_head(VisionTaskType.DETECTION, head_config, overrides)
