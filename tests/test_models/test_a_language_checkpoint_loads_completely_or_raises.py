"""One load rule for the language models' ``pretrained=<path>`` route.

plan-2026-10-11T004932-b83935e1 D-020 (encoders), D-024 (decoders) and D-031
(both directions, per variable): every class below, headless and headed,
restores EVERY backbone layer from a local checkpoint, and finds a home for
every checkpoint backbone layer, or raises. The checkpoint's ``task_head`` is
never copied. On a vocabulary change (the checkpoint's own ``vocab_size``
differs from the model's) only the DECLARED vocabulary tables (identified by
what they are, a variable-path suffix, and confirmed by shape) keep their
fresh init; every other variable, the position table living in the same layer
included, loads or the call raises (plan-2026-10-11T170958-dc4d2202 D-007). An untied ``lm_head`` is a top: it
may be left out only when one side carries a task head instead, or the
vocabulary changed.

Before D-020 a checkpoint of another hidden size loaded nothing and returned
normally for six of the eight encoder class/head pairs (reviewer W1). Before
D-024 a headless GPT2 restored the whole file with ``skip_mismatch=True``, so a
deeper or wider model kept fresh layers and returned normally. Before D-031 a
DEEPER CHECKPOINT was truncated silently in 9 of 10 classes, a vocabulary change
also discarded BERT/DistilBERT/FNet's trained position and token-type tables
(even of another length), and a tied headless checkpoint left an untied
``lm_head`` random (reviewer pass 2, WARNING 2 and NOTEs 3-4).

The vocabulary here (64) is NOT any class default, so a same-vocabulary load
also proves the exception is keyed on the checkpoint, not on a class constant.
"""

from typing import Any, Callable, Dict, NamedTuple

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
from dl_techniques.models.language.topolm import create_topolm
from dl_techniques.models.language.tree_transformer import create_tree_transformer
from dl_techniques.models.language.wave_field import create_wave_field_llm

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
    """How to build one class.

    ``position`` lengthens the learned position table (``{}``: the class has
    none); ``tieable`` marks a class with a ``tie_word_embeddings`` knob.
    """

    factory: Callable[..., keras.Model]
    kwargs: Dict[str, Any]
    misfits: Dict[str, Dict[str, Any]]
    position: Dict[str, Any]
    tieable: bool = False


_ENC = dict(vocab_size=_VOCAB, hidden_size=32, num_layers=2,
            intermediate_size=64, max_position_embeddings=32)
_ENC_MISFITS = {
    "hidden_size": dict(hidden_size=64, intermediate_size=128),
    "intermediate_size": dict(intermediate_size=128),
    "deeper": dict(num_layers=3),
}

_ENC_POSITION = dict(max_position_embeddings=64)

_MODELS = {
    # BERT, DistilBERT and FNet hold the position table in the SAME layer as
    # the vocabulary table (`embeddings`); GPT2 holds both in `decoder`.
    "bert": _Spec(lambda **k: create_bert("tiny", **k),
                  dict(_ENC, num_heads=4), _ENC_MISFITS, _ENC_POSITION),
    "distilbert": _Spec(lambda **k: create_distilbert("tiny", **k),
                        dict(_ENC, num_heads=4), _ENC_MISFITS, _ENC_POSITION),
    # RoPE: the position length sizes the encoder layers' caches.
    "modern_bert": _Spec(lambda **k: create_modern_bert("tiny", **k),
                         dict(_ENC, num_heads=4), _ENC_MISFITS, _ENC_POSITION),
    "fnet": _Spec(lambda **k: create_fnet("tiny", **k),
                  dict(_ENC), _ENC_MISFITS, _ENC_POSITION),
    # Untied, so a headless target holds an lm_head the rule must restore
    # from a headless checkpoint and may leave fresh from a headed one.
    "gpt2": _Spec(lambda **k: create_gpt2("tiny", **k),
                  dict(vocab_size=_VOCAB, embed_dim=32, depth=2, num_heads=2,
                       max_seq_len=16, tie_word_embeddings=False),
                  {"deeper": dict(depth=3), "wider": dict(embed_dim=64)},
                  dict(max_seq_len=32), tieable=True),
    "mamba": _Spec(lambda **k: create_mamba("base", **k),
                   dict(vocab_size=_VOCAB, d_model=32, num_layers=2, d_state=8),
                   {"deeper": dict(num_layers=3), "wider": dict(d_model=64)},
                   {}),
    "mamba2": _Spec(lambda **k: create_mamba2("base", **k),
                    dict(vocab_size=_VOCAB, d_model=32, num_layers=2, d_state=8,
                         headdim=16),
                    {"deeper": dict(num_layers=3), "wider": dict(d_model=64)},
                    {}),
    "topolm": _Spec(lambda **k: create_topolm("tiny", **k),
                    dict(vocab_size=_VOCAB, embed_dim=64, depth=1, num_heads=2,
                         radius=2, ffn_intermediate_size=128, max_seq_len=16,
                         tie_word_embeddings=False),
                    {"deeper": dict(depth=2),
                     "wider": dict(ffn_intermediate_size=256)},
                    dict(max_seq_len=32), tieable=True),
    # Its masked-LM top is `lm_head_projection` (no tie knob); D-022.
    "tree_transformer": _Spec(
        lambda **k: create_tree_transformer("tiny", **k),
        dict(vocab_size=_VOCAB, hidden_size=32, num_layers=2, num_heads=2,
             intermediate_size=64, max_len=16),
        {"deeper": dict(num_layers=3),
         "wider": dict(hidden_size=64, intermediate_size=128)},
        dict(max_len=32)),
    "wave_field": _Spec(lambda **k: create_wave_field_llm("tiny", **k),
                        dict(vocab_size=_VOCAB, embed_dim=32, depth=1,
                             num_heads=2, max_seq_len=16,
                             tie_word_embeddings=False),
                        {"deeper": dict(depth=2), "wider": dict(embed_dim=64)},
                        dict(max_seq_len=32), tieable=True),
}
_HEADS = {"headless": {}, "headed": dict(include_top=True, task_config=_CLF)}
_MISFIT_CASES = [(name, misfit) for name in sorted(_MODELS)
                 for misfit in _MODELS[name].misfits]
_POSITIONED = sorted(name for name, spec in _MODELS.items() if spec.position)
_TIEABLE = sorted(name for name, spec in _MODELS.items() if spec.tieable)
_MISFIT = "Failed to load weights|unloaded|mismatch|could not be loaded"


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


def _checkpoint(model, head, tmp_path, **overrides):
    """A randomized checkpoint of ``model`` saved with ``head``."""
    source = _build(model, head, **overrides)
    source(_batch(), training=False)
    rng = np.random.default_rng(1)
    for w in source.weights:
        # Integer weights (TopoLM's tap layouts) are indices, left alone.
        if "float" in str(w.dtype):
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
    with pytest.raises(ValueError, match=_MISFIT):
        _build(model, head, pretrained=path, **_MODELS[model].misfits[misfit])


def test_a_deeper_checkpoint_is_refused_not_truncated(model, head, tmp_path):
    """D-031: the extra checkpoint block has no home, so the load raises.

    Before D-031 nine of ten classes returned normally with the extra block
    dropped, and ModernBERT/TreeTransformer/Mamba's trained final norm then
    ran after a block it was never trained after. GPT2's blocks live inside
    its one ``decoder`` layer, so there the extra block is a mismatch on it
    (GPT2 already refused, the one exception the reviewer measured).
    """
    _, path = _checkpoint(model, head, tmp_path, **_MODELS[model].misfits["deeper"])
    with pytest.raises(
        ValueError,
        match=r"holds [1-9][0-9]* layer\(s\) with no target layer"
              r"|Shape mismatch on layer 'decoder'",
    ):
        _build(model, head, pretrained=path)


@pytest.mark.parametrize(
    "model, misfit", _MISFIT_CASES,
    ids=[f"{name}-{misfit}" for name, misfit in _MISFIT_CASES])
def test_a_vocabulary_change_does_not_excuse_a_misfit(model, misfit, head, tmp_path):
    """The vocabulary exemption covers the vocabulary layers, nothing else."""
    _, path = _checkpoint(model, head, tmp_path)
    with pytest.raises(ValueError, match=_MISFIT):
        _build(model, head, pretrained=path, vocab_size=_OTHER_VOCAB,
               **_MODELS[model].misfits[misfit])


@pytest.mark.parametrize("model", _POSITIONED)
def test_a_vocabulary_change_does_not_excuse_a_position_change(
        model, head, tmp_path):
    """D-031: the exemption covers vocabulary-shaped variables only.

    Before D-031 a vocabulary change exempted BERT/DistilBERT/FNet's whole
    ``embeddings`` layer, so a position table of ANOTHER length (and the
    token-type table and LayerNorm with it) stayed fresh and the load
    returned normally.
    """
    _, path = _checkpoint(model, head, tmp_path)
    with pytest.raises(ValueError, match=_MISFIT):
        _build(model, head, pretrained=path, vocab_size=_OTHER_VOCAB,
               **_MODELS[model].position)


@pytest.mark.parametrize("model", _POSITIONED)
def test_a_resized_position_table_that_looks_like_a_vocabulary_raises(
        model, head, tmp_path):
    """D-007: a table is a vocabulary table by what it IS, not by its shape.

    The position length goes from exactly the old vocabulary size to exactly
    the new one (64 -> 96, as the vocabulary does), the one coincidence the
    shape test cannot tell from a token table. Before D-007 BERT, DistilBERT,
    FNet and GPT2 returned normally with a RANDOM position table and a warning.
    """
    key = next(iter(_MODELS[model].position))
    _, path = _checkpoint(model, head, tmp_path, **{key: _VOCAB})
    with pytest.raises(ValueError, match=_MISFIT):
        _build(model, head, pretrained=path, vocab_size=_OTHER_VOCAB,
               **{key: _OTHER_VOCAB})


def test_the_same_checkpoint_restores_every_layer(model, head, tmp_path):
    """Every layer but the task head, an untied lm_head included."""
    source, path = _checkpoint(model, head, tmp_path)
    target = _build(model, head, pretrained=path)
    names = _weighted_names(target, exclude=(TASK_HEAD_NAME,))
    assert len(names) >= 1, names
    _assert_layers_equal(source, target, names)


def test_a_vocabulary_change_keeps_only_the_vocabulary_tables_fresh(
        model, head, tmp_path):
    """Every variable not shaped by the vocabulary loads, in every layer.

    D-031: the vocabulary layer itself is transferred variable by variable,
    so BERT's position table, GPT2's whole decoder but its token table, and
    every other variable hold the checkpoint's values.
    """
    source, path = _checkpoint(model, head, tmp_path)
    target = _build(model, head, pretrained=path, vocab_size=_OTHER_VOCAB)
    loaded, fresh = 0, []
    for name in _weighted_names(target, exclude=(TASK_HEAD_NAME,)):
        src = _layer_values(source, name)
        tgt = _layer_values(target, name)
        assert len(src) == len(tgt), name
        for a, b in zip(src, tgt):
            if a.shape == b.shape:
                np.testing.assert_array_equal(a, b, err_msg=name)
                loaded += 1
            else:
                # Differs only by the vocabulary, and is the target's own.
                assert all(x == y or (x, y) == (_VOCAB, _OTHER_VOCAB)
                           for x, y in zip(a.shape, b.shape)), (name, a.shape, b.shape)
                fresh.append(name)
    assert fresh, "no vocabulary table found: the arm proves nothing"
    assert loaded >= 4, loaded


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
    names = [n for n in _weighted_names(target) if not n.startswith("lm_head")]
    assert TASK_HEAD_NAME not in names
    _assert_layers_equal(source, target, names)


def test_a_headless_checkpoint_restores_a_headed_backbone(model, tmp_path):
    """The fine-tuning route: an untied lm_head in the checkpoint has no home
    in a headed model, and that is allowed (the target carries a task head)."""
    source, path = _checkpoint(model, "headless", tmp_path)
    target = _build(model, "headed", pretrained=path)
    _assert_layers_equal(
        source, target, _weighted_names(target, exclude=(TASK_HEAD_NAME,)))


@pytest.mark.parametrize("model", _TIEABLE)
def test_a_tied_checkpoint_does_not_leave_an_untied_lm_head_fresh(model, tmp_path):
    """D-031: both models headless, so the LM top must transfer.

    Before D-031 the untied ``lm_head`` -- the whole output projection of a
    headless language model -- stayed random and the load returned normally.
    """
    _, path = _checkpoint(model, "headless", tmp_path, tie_word_embeddings=True)
    with pytest.raises(
        ValueError, match=r"leaves 1 target layer\(s\) unloaded: \['lm_head'\]"
    ):
        _build(model, "headless", pretrained=path)


@pytest.mark.parametrize("model", _TIEABLE)
def test_an_untied_lm_head_is_not_dropped_into_a_tied_model(model, tmp_path):
    """D-031: the reverse direction used to drop a trained ``lm_head`` silently."""
    _, path = _checkpoint(model, "headless", tmp_path)
    with pytest.raises(
        ValueError, match=r"holds 1 layer\(s\) with no target layer: \['lm_head'\]"
    ):
        _build(model, "headless", pretrained=path, tie_word_embeddings=True)


@pytest.mark.parametrize("model", _TIEABLE)
def test_a_vocabulary_change_lets_a_tied_checkpoint_fill_an_untied_model(
        model, tmp_path):
    """Anti-vacuity for the two arms above: with a vocabulary change the
    ``lm_head`` would be fresh anyway, so the rule allows it."""
    source, path = _checkpoint(model, "headless", tmp_path, tie_word_embeddings=True)
    target = _build(model, "headless", pretrained=path, vocab_size=_OTHER_VOCAB)
    assert target.get_layer("lm_head").weights
