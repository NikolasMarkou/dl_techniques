"""Guard: a head factory must never SILENTLY substitute a head for an unimplemented task.

`heads/nlp/factory.py` and `heads/vlm/factory.py` both used to end their dispatch with a
silent default::

    return head_mapping.get(task_type, TextClassificationHead)   # nlp
    return head_mapping.get(task_type, BaseVLMHead)               # vlm

so a task type with no head quietly got *some other head* instead of an error.

**NLP -- silently WRONG, and it works.** 13 of the 37 `NLPTaskType` members
(MACHINE_TRANSLATION, DIALOGUE_GENERATION, RELATION_EXTRACTION, DEPENDENCY_PARSING,
COREFERENCE_RESOLUTION, SEMANTIC_ROLE_LABELING, ...) returned a `TextClassificationHead`.
It builds, it trains, it emits plausible numbers -- a translation task quietly became a
classifier. Nothing ever failed.

**VLM -- silently BROKEN, deferred.** 41 of the 47 `VLMTaskType` members returned a bare
`BaseVLMHead`, which has NO `call()` method. So the factory returned an object that
constructed fine and then died on first use with `NotImplementedError: Layer BaseVLMHead
does not have a call() method` -- an error naming the base class, not the task, so the
caller had no way to know their task type was never implemented. The two entries mapped to
`BaseVLMHead` explicitly and commented "# Placeholder" had the same defect.

The sibling `heads/vision/factory.py` already got this right --
`raise ValueError(f"Unsupported task type: {task_type}")` -- which is the precedent both
now follow.

This guard asserts the property directly: for EVERY task-type enum member, the dispatch
either returns a genuinely usable head, or raises. Never a silent substitute.
"""

import inspect

import pytest

from dl_techniques.layers.heads.nlp import factory as nlp_factory
from dl_techniques.layers.heads.nlp.task_types import NLPTaskType
from dl_techniques.layers.heads.vlm import factory as vlm_factory
from dl_techniques.layers.heads.vlm.task_types import VLMTaskType


def _is_usable_head(cls):
    """A head must define `call()`. `BaseVLMHead` does not -- it cannot be used as a head.

    Checking `hasattr(cls, 'call')` is NOT enough: every `keras.layers.Layer` inherits a
    `call` that raises NotImplementedError. The question is whether the class (or a real
    subclass in its MRO below Layer) DEFINES one.
    """
    import keras

    for klass in cls.__mro__:
        if klass is keras.layers.Layer:
            break
        if "call" in klass.__dict__:
            return True
    return False


class TestNoSilentFallback:
    @pytest.mark.parametrize("task_type", list(NLPTaskType), ids=lambda t: t.name)
    def test_nlp_dispatch_never_silently_substitutes(self, task_type):
        """Every NLPTaskType either has a real head, or raises. Never a silent default."""
        try:
            head_class = nlp_factory.get_head_class(task_type)
        except ValueError:
            return  # unimplemented, and it says so -- this is the correct outcome
        assert _is_usable_head(head_class), (
            f"NLP task '{task_type.name}' dispatched to {head_class.__name__}, which "
            f"defines no call() -- it is not a usable head."
        )

    @pytest.mark.parametrize("task_type", list(VLMTaskType), ids=lambda t: t.name)
    def test_vlm_dispatch_never_silently_substitutes(self, task_type):
        """Every VLMTaskType either has a real head, or raises.

        In particular the dispatch must NEVER hand back `BaseVLMHead`: it has no call(),
        so returning it defers a guaranteed NotImplementedError to first use.
        """
        try:
            head_class = vlm_factory.get_head_class(task_type)
        except ValueError:
            return  # unimplemented, and it says so
        assert head_class is not vlm_factory.BaseVLMHead, (
            f"VLM task '{task_type.name}' dispatched to the bare BaseVLMHead, which has "
            f"no call() method -- it raises NotImplementedError on first use. An "
            f"unimplemented task must RAISE at dispatch, not return an unusable object."
        )
        assert _is_usable_head(head_class), (
            f"VLM task '{task_type.name}' dispatched to {head_class.__name__}, which "
            f"defines no call() -- it is not a usable head."
        )

    def test_the_guard_has_subjects(self):
        """Guard the guard: an empty enum would make everything above pass vacuously."""
        assert len(list(NLPTaskType)) > 10
        assert len(list(VLMTaskType)) > 10

    def test_at_least_one_task_is_actually_implemented(self):
        """The dispatch must not be uniformly raising -- that would also pass vacuously."""
        nlp_ok = [
            t for t in NLPTaskType
            if not _raises(nlp_factory.get_head_class, t)
        ]
        vlm_ok = [
            t for t in VLMTaskType
            if not _raises(vlm_factory.get_head_class, t)
        ]
        assert nlp_ok, "no NLP task resolves to a head -- the dispatch is broken"
        assert vlm_ok, "no VLM task resolves to a head -- the dispatch is broken"


def _raises(fn, arg):
    try:
        fn(arg)
        return False
    except ValueError:
        return True
