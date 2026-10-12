"""A decoder's task head treats NO token id as padding unless told which one.

Subject: the ``pad_token_id`` constructor knob of the eight decoders that own a
task head (GPT2, TopoLM, WaveFieldLLM, Qwen3, Qwen3Next, Gemma3, Mamba, Mamba2)
and ``head_attention_mask``, the one place that reads it
(plan-2026-10-11T170958-dc4d2202 D-008).

The knob used to default to ``0``. Id 0 is a real token (``!`` in the ``gpt2``
and ``cl100k_base`` encodings): a headed decoder fed ``[5, 7, 9, 0]`` with no
mask silently dropped the ``0`` from the pooled positions, and under ``last``
pooling answered from the token before it. The default is ``None`` now, which
means "no padding": the head sees every position, i.e. the all-ones mask. The
decoder itself is causal and never read the value, so no headless output moves
(the headless controls are the proof). A padded batch must pass
``attention_mask`` or the tokenizer's real pad id.

``test_the_default_is_no_padding_id``
    Constructor default and ``get_config`` both say ``None``.
``test_no_pad_id_pools_every_position_like_an_all_ones_mask``
    With the default, a ``0`` token inside and at the end of the input changes
    nothing relative to passing the all-ones mask explicitly. Anti-vacuity: the
    SAME weights under ``pad_token_id=0`` give a different answer on that input,
    so the input discriminates between the two behaviours.
``test_an_explicit_pad_id_is_preserved_by_every_round_trip``
    A config written before the default changed carries ``pad_token_id: 0``;
    ``from_config`` keeps it (and keeps masking id 0), and so does a ``.keras``
    save/load, for 0 and for a non-zero id.

Adding a class is one name in ``SUBJECTS`` (its configuration lives in
``tiny_language_models.py``).
"""

import inspect

import keras
import numpy as np
import pytest

from dl_techniques.layers.heads.nlp import NLPTaskConfig, NLPTaskType

from .tiny_language_models import TINY, VOCAB

# ---------------------------------------------------------------------

_VOCAB = VOCAB
_CLF = NLPTaskConfig(
    name="clf", task_type=NLPTaskType.TEXT_CLASSIFICATION, num_classes=3)
_HEAD = dict(include_top=True, task_config=_CLF)

SUBJECTS = ("gpt2", "topolm", "wave_field", "qwen3", "qwen3_next", "gemma3",
            "mamba", "mamba2")


@pytest.fixture(params=SUBJECTS)
def subject(request):
    tiny = TINY[request.param]
    return tiny.cls(), tiny.kwargs


def _ids():
    """Ids with the real token 0 in the middle and at the end of every row."""
    ids = np.random.default_rng(0).integers(1, _VOCAB, (2, 6)).astype("int32")
    ids[:, 3] = 0
    ids[:, -1] = 0
    return ids


def _run(model, ids, mask=None):
    inputs = ids if mask is None else {"input_ids": ids, "attention_mask": mask}
    return keras.ops.convert_to_numpy(model(inputs, training=False))


# ---------------------------------------------------------------------


def test_the_default_is_no_padding_id(subject):
    cls, kwargs = subject
    default = inspect.signature(cls.__init__).parameters["pad_token_id"].default
    assert default is None
    model = cls(**kwargs, **_HEAD)
    assert model.pad_token_id is None
    assert model.get_config()["pad_token_id"] is None


def test_no_pad_id_pools_every_position_like_an_all_ones_mask(subject):
    cls, kwargs = subject
    ids = _ids()
    ones = np.ones_like(ids)
    keras.utils.set_random_seed(0)
    unset = cls(**kwargs, **_HEAD)
    got = _run(unset, ids)
    np.testing.assert_allclose(got, _run(unset, ids, ones), atol=1e-6, rtol=0)

    # The same weights under pad_token_id=0 answer differently on this input:
    # the id-0 positions leave the pooled set, so the check above can fail.
    zero = cls(**kwargs, **_HEAD, pad_token_id=0)
    zero(ids, training=False)
    zero.set_weights(unset.get_weights())
    assert not np.allclose(_run(zero, ids), got, atol=1e-6, rtol=0)


def test_an_explicit_pad_id_is_preserved_by_every_round_trip(subject, tmp_path):
    cls, kwargs = subject
    ids = _ids()

    # A config as an archive written before the default changed recorded it.
    legacy = dict(cls(**kwargs, **_HEAD).get_config(), pad_token_id=0)
    rebuilt = cls.from_config(legacy)
    assert rebuilt.pad_token_id == 0 and rebuilt.get_config()["pad_token_id"] == 0
    explicit = cls(**kwargs, **_HEAD, pad_token_id=0)
    explicit(ids, training=False)
    rebuilt(ids, training=False)
    rebuilt.set_weights(explicit.get_weights())
    np.testing.assert_array_equal(_run(rebuilt, ids), _run(explicit, ids))

    for pad in (0, 5):
        model = cls(**kwargs, **_HEAD, pad_token_id=pad)
        expected = _run(model, ids)
        path = str(tmp_path / f"{pad}.keras")
        model.save(path)
        loaded = keras.saving.load_model(path)
        assert loaded.pad_token_id == pad
        np.testing.assert_array_equal(_run(loaded, ids), expected)
