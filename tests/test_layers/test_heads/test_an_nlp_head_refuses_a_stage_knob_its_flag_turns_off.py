"""An NLP head refuses a task-attention or FFN setting for a stage that is off.

D-210's rule (the vision heads, iteration 2 step 6.6) found on the NLP heads
as well. Measured at 6443c3fb4 (``scratchpad/iter2/s6/nlp_b12.py``): with
``use_task_attention=False`` (the default) ``attention_type='performer'`` was
accepted, stored and written to ``get_config``, and so were ``ffn_type`` and
``ffn_expansion_factor`` with ``use_ffn=False`` (the default), on the
classification, token and QA heads alike, while no layer read them.

Now the three knobs default to ``None``; with the stage on, ``None`` resolves
to ``'multi_head'``, ``'mlp'`` and ``4`` as before, and with it off a value
raises ``ValueError`` and ``None`` is stored.
"""

import pytest

from dl_techniques.layers.heads.nlp import NLPTaskConfig, NLPTaskType, create_nlp_head

D = 8

CONFIGS = {
    "classification": NLPTaskConfig(name="c", task_type=NLPTaskType.TEXT_CLASSIFICATION, num_classes=3),
    "token": NLPTaskConfig(name="n", task_type=NLPTaskType.NAMED_ENTITY_RECOGNITION, num_classes=3),
    "qa": NLPTaskConfig(name="q", task_type=NLPTaskType.QUESTION_ANSWERING),
    "similarity": NLPTaskConfig(name="s", task_type=NLPTaskType.TEXT_SIMILARITY),
    "generation": NLPTaskConfig(name="g", task_type=NLPTaskType.TEXT_GENERATION, vocabulary_size=11),
    "multiple_choice": NLPTaskConfig(name="m", task_type=NLPTaskType.MULTIPLE_CHOICE),
}

_OFF_KNOBS = [
    ("use_task_attention", "attention_type", "performer"),
    ("use_task_attention", "attention_type", "multi_head"),
    ("use_ffn", "ffn_type", "swiglu"),
    ("use_ffn", "ffn_type", "mlp"),
    ("use_ffn", "ffn_expansion_factor", 7),
    ("use_ffn", "ffn_expansion_factor", 4),
]


@pytest.mark.parametrize("flag,knob,value", _OFF_KNOBS, ids=[f"{k}={v}" for _, k, v in _OFF_KNOBS])
@pytest.mark.parametrize("head", sorted(CONFIGS))
def test_a_knob_for_a_stage_that_is_off_is_refused(head, flag, knob, value):
    with pytest.raises(ValueError, match=knob):
        create_nlp_head(CONFIGS[head], D, **{flag: False, knob: value})


@pytest.mark.parametrize("head", sorted(CONFIGS))
def test_a_head_with_both_stages_off_stores_none_and_round_trips(head):
    built = create_nlp_head(CONFIGS[head], D)
    config = built.get_config()
    for knob in ("attention_type", "ffn_type", "ffn_expansion_factor"):
        assert getattr(built, knob) is None and config[knob] is None, knob
    assert type(built).from_config(config).get_config() == config


def test_a_stage_that_is_on_resolves_none_to_the_old_default():
    # The token head applies both stages.
    built = create_nlp_head(CONFIGS["token"], D, use_task_attention=True, use_ffn=True)
    assert (built.attention_type, built.ffn_type, built.ffn_expansion_factor) == ("multi_head", "mlp", 4)


def test_a_stage_that_is_on_takes_the_given_knobs():
    built = create_nlp_head(CONFIGS["token"], D, use_task_attention=True, attention_type="window",
                            use_ffn=True, ffn_type="glu", ffn_expansion_factor=2)
    assert (built.attention_type, built.ffn_type, built.ffn_expansion_factor) == ("window", "glu", 2)
