"""Guard: a bad ``vocab_size`` is refused with an error that names it.

Plan ``plan-2026-10-13T035756-5dd4d94d``, step 18 (D-019, review pass 2 NOTE 6).

Step 13 made ``create_bert``, ``create_gpt2``, ``create_tree_transformer`` and
``create_wave_field_llm`` pure delegation to ``from_variant``, so the factories'
old ``vocab_size=None`` default is gone. An explicit ``vocab_size=None`` then
reached each constructor's ``vocab_size <= 0`` check and failed with
``TypeError: '<=' not supported between instances of 'NoneType' and 'int'``,
which does not say which argument was wrong. Each constructor now refuses a
value that is not a positive integer (``bool`` included, since ``True <= 0`` is
False and ``True`` would build a one-token vocabulary) with a ``ValueError``
naming ``vocab_size``; a NumPy integer is accepted and stored as ``int``
(GPT2's own sub-layers refused one before). The check sits in the constructor, not the factory, so
the factories stay pure delegation; both routes are exercised here.
"""

import numpy as np
import pytest

from dl_techniques.models.language.bert.model import BERT, create_bert
from dl_techniques.models.language.gpt2.gpt2 import GPT2, create_gpt2
from dl_techniques.models.language.tree_transformer.model import (
    TreeTransformer,
    create_tree_transformer,
)
from dl_techniques.models.language.wave_field.model import (
    WaveFieldLLM,
    create_wave_field_llm,
)

_MODELS = {
    "bert": (BERT, create_bert),
    "gpt2": (GPT2, create_gpt2),
    "tree_transformer": (TreeTransformer, create_tree_transformer),
    "wave_field": (WaveFieldLLM, create_wave_field_llm),
}

_BAD_VOCAB_SIZES = [None, 0, -1, 2.5, 64.0, True, "64"]


def _tiny_kwargs(cls):
    """The ``tiny`` variant's constructor kwargs, without ``vocab_size`` or its label."""
    config = dict(cls.MODEL_VARIANTS["tiny"])
    config.pop("vocab_size", None)
    config.pop("description", None)
    return config


@pytest.mark.parametrize("bad", _BAD_VOCAB_SIZES, ids=repr)
@pytest.mark.parametrize("name", sorted(_MODELS))
def test_the_constructor_refuses_a_bad_vocab_size_by_name(name, bad):
    cls, _ = _MODELS[name]
    with pytest.raises(ValueError, match="vocab_size must be positive and an integer"):
        cls(vocab_size=bad, **_tiny_kwargs(cls))


@pytest.mark.parametrize("name", sorted(_MODELS))
def test_the_factory_refuses_an_explicit_none_by_name(name):
    """The reported shape: ``create_X("tiny", vocab_size=None)``."""
    _, factory = _MODELS[name]
    with pytest.raises(ValueError, match="vocab_size must be positive and an integer"):
        factory("tiny", vocab_size=None)


@pytest.mark.parametrize("good", [1, 64, np.int64(64)], ids=repr)
@pytest.mark.parametrize("name", sorted(_MODELS))
def test_a_positive_integer_is_accepted(name, good):
    """The control: ints, including a NumPy integer, construct and store an int."""
    cls, _ = _MODELS[name]
    model = cls(vocab_size=good, **_tiny_kwargs(cls))
    assert model.vocab_size == good
    assert type(model.vocab_size) is int
