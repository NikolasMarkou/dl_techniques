"""A checkpoint saved under one head fills a model with another head, or none.

Subject: the fifteen language classes that own a task head, through the
``pretrained=<path>`` route each class documents (``create_<model>(variant,
pretrained=path, ...)``, i.e. ``from_variant``). One source per class is built
with a text-classification head, every float weight randomized (so no target
value can match by a shared initializer), and saved. Each target is built from
that archive (plan-2026-10-13T035756-5dd4d94d step 6, review W3 of
plan-2026-10-11T170958-dc4d2202):

* ``ner`` -- a token-classification head (another task, another head shape);
* ``clf`` -- the same classification head (equal shapes: the head a loader
  could copy without noticing);
* ``clf_other`` -- a classification head with another number of classes;
* ``headless`` -- no head; an untied LM top then exists in the target only.

Variables are matched by path without the model-root segment.

``test_the_backbone_survives_a_head_swap``
    Every non-``task_head`` variable present in both models is
    ``array_equal``; the target holds no backbone variable the source lacks
    except its untied top (the ``Top`` column of the load-rule table in
    ``models/language/AGENTS.md``), and the source has none left behind.
``test_the_new_head_is_not_the_old_one``
    The twin: every target head variable with a same-path, same-shape source
    twin differs from it. A load that copied nothing would fail the arm above
    (the target backbone would be fresh too); a load that copied everything
    would fail this one.

``test_a_language_checkpoint_loads_completely_or_raises.py`` owns the
equal-head and headed/headless directions per layer; this file adds the
cross-task and other-shape targets and the per-variable census.
"""

from typing import Dict

import numpy as np
import pytest

from dl_techniques.layers.heads.nlp import NLPTaskConfig, NLPTaskType
from dl_techniques.layers.heads.nlp.model_head import TASK_HEAD_NAME

from .headed_logit_decoder_cases import NER
from .test_a_language_checkpoint_loads_completely_or_raises import (
    _CLF,
    _MODELS,
    _batch,
    _build,
    _checkpoint,
)

SUBJECTS = tuple(sorted(_MODELS))

# The untied LM top of each class as built by ``_MODELS`` (GPT2, TopoLM,
# WaveFieldLLM and HNet are built untied there). A headless target owns it, a
# headed source does not, so it is the one backbone-side layer that stays fresh.
_TOPS = {
    "tree_transformer": "lm_head_projection",
    "gpt2": "lm_head",
    "topolm": "lm_head",
    "wave_field": "lm_head",
    "hnet": "lm_head",
    "qwen3": "lm_head",
    "qwen3_next": "lm_head",
    "gemma3": "lm_head",
}

_TARGETS = {
    "ner": dict(include_top=True, task_config=NER),
    "clf": dict(include_top=True, task_config=_CLF),
    "clf_other": dict(include_top=True, task_config=NLPTaskConfig(
        name="clf5", task_type=NLPTaskType.TEXT_CLASSIFICATION, num_classes=5)),
    "headless": dict(include_top=False),
}
_HEADED = sorted(t for t, kw in _TARGETS.items() if kw["include_top"])


def _by_path(model) -> Dict[str, np.ndarray]:
    """``{path without the model-root segment: value}``.

    Only a segment equal to the model's name is stripped: WaveFieldLLM builds
    its layers in ``__init__``, outside its own name scope, so its paths start
    at the layer (``token_embeddings/embeddings``) and blindly dropping the
    first segment would merge distinct variables.
    """
    root = model.name + "/"
    values = {w.path[len(root):] if w.path.startswith(root) else w.path:
              np.asarray(w) for w in model.weights}
    assert len(values) == len(model.weights), "two variables share a path"
    return values


def _is_head(path: str) -> bool:
    return path.split("/", 1)[0] == TASK_HEAD_NAME


def _is_top(subject: str, path: str) -> bool:
    return path.split("/", 1)[0] == _TOPS.get(subject)


@pytest.fixture(scope="module")
def swaps(tmp_path_factory):
    """``swap(subject, target)`` -> ``(source, target)`` values by path.

    One randomized headed archive per class, one target build per pair; both
    cached for the module, so each model is built once.
    """
    archives, built = {}, {}

    def swap(subject, target):
        if subject not in archives:
            source, path = _checkpoint(
                subject, "headed", tmp_path_factory.mktemp(subject))
            archives[subject] = (_by_path(source), path)
        if (subject, target) not in built:
            source, path = archives[subject]
            model = _build(subject, "headless", pretrained=path,
                           **_TARGETS[target])
            model(_batch(), training=False)
            assert model.include_top is (target != "headless")
            built[subject, target] = (source, _by_path(model))
        return built[subject, target]

    return swap


@pytest.fixture(params=SUBJECTS)
def subject(request):
    return request.param


def test_the_census_covers_the_fifteen_head_owners():
    assert len(SUBJECTS) == 15
    assert set(_TOPS) <= set(SUBJECTS)


@pytest.mark.parametrize("target", sorted(_TARGETS))
def test_the_backbone_survives_a_head_swap(swaps, subject, target):
    source, swapped = swaps(subject, target)
    assert any(_is_head(p) for p in source), "the source carries no task head"

    shared = [p for p in swapped if p in source and not _is_head(p)]
    assert len(shared) >= 4, shared
    differ = [p for p in shared if not np.array_equal(source[p], swapped[p])]
    assert not differ, f"{len(differ)}/{len(shared)} backbone variables fresh: {differ[:4]}"

    fresh = sorted(p for p in swapped if p not in source and not _is_head(p))
    left = sorted(p for p in source if p not in swapped and not _is_head(p))
    assert not left, f"source backbone variables with no target: {left[:4]}"
    assert all(_is_top(subject, p) for p in fresh), (
        f"fresh variables outside task_head and the untied top: {fresh[:4]}")
    if target == "headless" and subject in _TOPS:
        assert fresh, f"the headless {subject} target built no {_TOPS[subject]}"
    else:
        assert not fresh, fresh[:4]


@pytest.mark.parametrize("target", _HEADED)
def test_the_new_head_is_not_the_old_one(swaps, subject, target):
    source, swapped = swaps(subject, target)
    head = [p for p in swapped if _is_head(p)]
    assert head, "the headed target built no task head"
    twins = [p for p in head if p in source and source[p].shape == swapped[p].shape]
    if target == "clf":
        assert len(twins) == len(head), (
            "fixture invalid: an equal head must have equal paths and shapes")
    copied = [p for p in twins if np.array_equal(source[p], swapped[p])]
    assert not copied, f"head variables copied from the checkpoint: {copied}"
