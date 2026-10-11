"""One load rule for the language models' ``pretrained=<path>`` route.

plan-2026-10-11T004932-b83935e1 D-020 (encoders) and D-024 (decoders): every
class below, headless and headed, restores EVERY backbone layer from a local
checkpoint or raises. The checkpoint's ``task_head`` is never copied. The
vocabulary-dependent layers (token embedding, untied ``lm_head``) may stay at
their fresh init only when the checkpoint's own ``vocab_size`` differs from the
model's; a class whose embedding cannot be left out on its own (GPT2, whose
whole decoder is one layer) refuses a vocabulary change instead. An untied
``lm_head`` is a top, so it may also stay fresh when the checkpoint has none (a
checkpoint saved with a task head). TreeTransformer follows the same rule and
is pinned in its own suite (D-022).

Before D-020 a checkpoint of another hidden size loaded nothing and returned
normally for six of the eight encoder class/head pairs (reviewer W1). Before
D-024 a headless GPT2 restored the whole file with ``skip_mismatch=True``, so a
deeper or wider model kept fresh layers and returned normally.

The vocabulary here (64) is NOT any class default, so a same-vocabulary load
also proves the exception is keyed on the checkpoint, not on a class constant.
"""

from typing import Any, Callable, Dict, NamedTuple, Tuple

import keras
import numpy as np
import pytest

from dl_techniques.layers.heads.nlp import NLPTaskConfig, NLPTaskType
from dl_techniques.layers.heads.nlp.model_head import TASK_HEAD_NAME
from dl_techniques.models.language.bert import create_bert
from dl_techniques.models.language.distilbert import create_distilbert
from dl_techniques.models.language.fnet import create_fnet
from dl_techniques.models.language.gpt2 import create_gpt2
from dl_techniques.models.language.mamba import create_mamba, create_mamba2
from dl_techniques.models.language.modern_bert import create_modern_bert

# Keras warns "A total of N objects could not be loaded" when a whole-file
# restore skips variables, and this repo's pytest config escalates UserWarning
# to an error. Outside pytest nothing escalates it, so a silent partial restore
# returns normally there. Ignored here so a misfit test sees what production
# sees: measured, without this mark the old headless GPT2 route (whole file,
# skip_mismatch=True) raised under pytest and passed the deeper-target arms.
# The fixed routes never emit it.
pytestmark = [pytest.mark.filterwarnings(
    "ignore:A total of .* objects could not be loaded:UserWarning")]

_CLF = NLPTaskConfig(
    name="clf", task_type=NLPTaskType.TEXT_CLASSIFICATION, num_classes=3)
_SEQ = 8
_VOCAB = 64
_OTHER_VOCAB = 96


class _Spec(NamedTuple):
    """How to build one class, and what the rule allows it.

    ``vocab_layers`` are the layers that stay fresh on a vocabulary change;
    ``()`` means the class refuses a vocabulary change.
    """

    factory: Callable[..., keras.Model]
    kwargs: Dict[str, Any]
    misfits: Dict[str, Dict[str, Any]]
    vocab_layers: Tuple[str, ...]


_ENC = dict(vocab_size=_VOCAB, hidden_size=32, num_layers=2,
            intermediate_size=64, max_position_embeddings=32)
_ENC_MISFITS = {
    "hidden_size": dict(hidden_size=64, intermediate_size=128),
    "intermediate_size": dict(intermediate_size=128),
    "deeper": dict(num_layers=3),
}

_MODELS = {
    "bert": _Spec(lambda **k: create_bert("tiny", **k),
                  dict(_ENC, num_heads=4), _ENC_MISFITS, ("embeddings",)),
    "distilbert": _Spec(lambda **k: create_distilbert("tiny", **k),
                        dict(_ENC, num_heads=4), _ENC_MISFITS, ("embeddings",)),
    "modern_bert": _Spec(lambda **k: create_modern_bert("tiny", **k),
                         dict(_ENC, num_heads=4), _ENC_MISFITS, ("embeddings",)),
    "fnet": _Spec(lambda **k: create_fnet("tiny", **k),
                  dict(_ENC), _ENC_MISFITS, ("embeddings",)),
    # Untied, so a headless target holds an lm_head the rule must restore
    # from a headless checkpoint and may leave fresh from a headed one.
    "gpt2": _Spec(lambda **k: create_gpt2("tiny", **k),
                  dict(vocab_size=_VOCAB, embed_dim=32, depth=2, num_heads=2,
                       max_seq_len=16, tie_word_embeddings=False),
                  {"deeper": dict(depth=3), "wider": dict(embed_dim=64)},
                  ()),
    # Headless with the checkpoint's vocabulary: whole-file strict restore;
    # headed or a vocabulary change: by name (D-025).
    "mamba": _Spec(lambda **k: create_mamba("base", **k),
                   dict(vocab_size=_VOCAB, d_model=32, num_layers=2, d_state=8),
                   {"deeper": dict(num_layers=3), "wider": dict(d_model=64)},
                   ("embedding",)),
    "mamba2": _Spec(lambda **k: create_mamba2("base", **k),
                    dict(vocab_size=_VOCAB, d_model=32, num_layers=2, d_state=8,
                         headdim=16),
                    {"deeper": dict(num_layers=3), "wider": dict(d_model=64)},
                    ("embedding",)),
}
_HEADS = {"headless": {}, "headed": dict(include_top=True, task_config=_CLF)}
_MISFIT_CASES = [(name, misfit) for name in sorted(_MODELS)
                 for misfit in _MODELS[name].misfits]


@pytest.fixture(params=sorted(_MODELS))
def model(request):
    return request.param


@pytest.fixture(params=sorted(_HEADS))
def head(request):
    return request.param


def _batch():
    ids = np.random.default_rng(0).integers(1, _VOCAB, (2, _SEQ)).astype("int32")
    return {"input_ids": ids, "attention_mask": np.ones_like(ids)}


def _build(model, head, pretrained=False, **overrides):
    spec = _MODELS[model]
    return spec.factory(pretrained=pretrained,
                        **dict(spec.kwargs, **_HEADS[head], **overrides))


def _checkpoint(model, head, tmp_path):
    """A randomized checkpoint of ``model`` saved with ``head``."""
    source = _build(model, head)
    source(_batch(), training=False)
    rng = np.random.default_rng(1)
    for w in source.weights:
        w.assign(rng.normal(0.0, 0.1, size=tuple(w.shape)).astype(w.dtype))
    path = str(tmp_path / f"{model}_{head}.keras")
    source.save(path)
    return source, path


def _layer_values(model, name):
    return [np.asarray(w) for w in model.get_layer(name).get_weights()]


def _weighted_names(model, exclude=()):
    return [layer.name for layer in model.layers
            if layer.weights and layer.name not in exclude]


def _assert_layers_equal(source, target, names):
    for name in names:
        src, tgt = _layer_values(source, name), _layer_values(target, name)
        assert len(src) == len(tgt), name
        for a, b in zip(src, tgt):
            np.testing.assert_array_equal(a, b, err_msg=name)


@pytest.mark.parametrize(
    "model, misfit", _MISFIT_CASES,
    ids=[f"{name}-{misfit}" for name, misfit in _MISFIT_CASES])
def test_a_backbone_that_does_not_fit_raises(model, misfit, head, tmp_path):
    _, path = _checkpoint(model, head, tmp_path)
    with pytest.raises(ValueError, match="Failed to load weights|unloaded|mismatch|could not be loaded"):
        _build(model, head, pretrained=path, **_MODELS[model].misfits[misfit])


@pytest.mark.parametrize(
    "model, misfit", _MISFIT_CASES,
    ids=[f"{name}-{misfit}" for name, misfit in _MISFIT_CASES])
def test_a_vocabulary_change_does_not_excuse_a_misfit(model, misfit, head, tmp_path):
    """The vocabulary exemption covers the vocabulary layers, nothing else."""
    _, path = _checkpoint(model, head, tmp_path)
    with pytest.raises(
        ValueError,
        match="Failed to load weights|unloaded|mismatch|could not be loaded"
              "|another vocab_size",
    ):
        _build(model, head, pretrained=path, vocab_size=_OTHER_VOCAB,
               **_MODELS[model].misfits[misfit])


def test_the_same_checkpoint_restores_every_layer(model, head, tmp_path):
    """Every layer but the task head, an untied lm_head included."""
    source, path = _checkpoint(model, head, tmp_path)
    target = _build(model, head, pretrained=path)
    names = _weighted_names(target, exclude=(TASK_HEAD_NAME,))
    assert len(names) >= 1, names
    _assert_layers_equal(source, target, names)


def test_a_vocabulary_change_follows_the_rule(model, head, tmp_path):
    source, path = _checkpoint(model, head, tmp_path)
    vocab_layers = _MODELS[model].vocab_layers
    if not vocab_layers:
        with pytest.raises(ValueError, match="another vocab_size"):
            _build(model, head, pretrained=path, vocab_size=_OTHER_VOCAB)
        return

    target = _build(model, head, pretrained=path, vocab_size=_OTHER_VOCAB)
    names = _weighted_names(target, exclude=(TASK_HEAD_NAME, *vocab_layers))
    assert len(names) >= 2, names
    _assert_layers_equal(source, target, names)
    # The vocabulary-dependent tables are the target's own, fresh.
    held = [n for n in vocab_layers if n in _weighted_names(target)]
    assert held, (vocab_layers, _weighted_names(target))
    for name in held:
        assert _OTHER_VOCAB in _layer_values(target, name)[0].shape, name


def test_the_checkpoint_task_head_is_never_copied(model, tmp_path):
    """An equal-shape head of the same task stays at its fresh init."""
    source, path = _checkpoint(model, "headed", tmp_path)
    target = _build(model, "headed", pretrained=path)
    src_head = _layer_values(source, TASK_HEAD_NAME)
    tgt_head = _layer_values(target, TASK_HEAD_NAME)
    assert [a.shape for a in src_head] == [b.shape for b in tgt_head], (
        "fixture invalid: the two heads must have equal weight shapes")
    assert not any(np.array_equal(a, b) for a, b in zip(src_head, tgt_head))
    _assert_layers_equal(
        source, target, _weighted_names(target, exclude=(TASK_HEAD_NAME,)))


def test_a_headed_checkpoint_restores_a_headless_backbone(model, tmp_path):
    """A headless target takes the backbone; an untied lm_head stays fresh."""
    source, path = _checkpoint(model, "headed", tmp_path)
    target = _build(model, "headless", pretrained=path)
    names = _weighted_names(target, exclude=("lm_head",))
    assert TASK_HEAD_NAME not in names
    _assert_layers_equal(source, target, names)
