"""Two same-config builds of a language model in one process have identical weight paths.

Subject: the fifteen language classes that own a task head (``TINY``), each
headless and headed with a ``task_config`` (plan-2026-10-13T035756-5dd4d94d
step 6, review W3 of plan-2026-10-11T170958-dc4d2202). A sub-layer left without
``name=`` takes Keras's per-process counter (``dense_7``), so the second model
built in a process disagrees with the first on its weight paths, and every
by-path comparison (a head swap census, a parity script, a path-keyed
transfer) silently misses. A DECOY of the same class and top is built first so
the counters are already advanced; the two compared models are built after it,
under one explicit name.

``test_the_language_models_rebuilt_paths_are_identical``
    The ordered weight paths of the two builds are equal, and the headed build
    holds ``task_head`` variables (so the head's own sub-layers are covered).

``test_a_rebuilt_model_has_identical_weight_paths.py`` pins the same claim for
seven other subjects (NAM, Mamba with MoE variants, ...); this file is the
census over every head owner and both tops.
"""

import keras
import pytest

from dl_techniques.layers.heads.nlp.model_head import TASK_HEAD_NAME

from .headed_logit_decoder_cases import CLF, head_batch
from .tiny_language_models import TINY

SUBJECTS = tuple(sorted(TINY))
_TOPS = {"headless": {}, "headed": dict(include_top=True, task_config=CLF)}


def _built(subject, top, name):
    tiny = TINY[subject]
    model = tiny.cls()(**tiny.kwargs, **_TOPS[top], name=name)
    model(head_batch(), training=False)
    return model


def test_the_census_covers_the_fifteen_head_owners():
    assert len(SUBJECTS) == 15


@pytest.mark.parametrize("top", sorted(_TOPS))
@pytest.mark.parametrize("subject", SUBJECTS)
def test_the_language_models_rebuilt_paths_are_identical(subject, top):
    keras.utils.set_random_seed(0)
    _built(subject, top, f"decoy_{subject}")  # advances the name counters
    first = _built(subject, top, f"p_{subject}")
    second = _built(subject, top, f"p_{subject}")

    paths_a = [w.path for w in first.weights]
    paths_b = [w.path for w in second.weights]
    assert len(paths_a) > 5, "too few weights: the arm proves nothing"
    has_head = any(TASK_HEAD_NAME in p.split("/") for p in paths_a)
    assert has_head is (top == "headed"), paths_a
    assert paths_a == paths_b, [
        (a, b) for a, b in zip(paths_a, paths_b) if a != b][:4]
