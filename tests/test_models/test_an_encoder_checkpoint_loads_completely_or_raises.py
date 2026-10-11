"""One load rule for the four encoders' ``pretrained=<path>`` route.

plan-2026-10-11T004932-b83935e1 D-020: BERT, DistilBERT, ModernBERT and FNet,
headless and headed, restore EVERY encoder layer from a local checkpoint or
raise. The one exception is the ``embeddings`` layer when the checkpoint's own
``vocab_size`` differs from the model's; it then keeps its fresh init. The
checkpoint's ``task_head`` is never copied.

Before D-020 a checkpoint of another hidden size loaded nothing and returned
normally for six of the eight class/head pairs (reviewer W1): ``from_variant``
set ``skip_mismatch=True`` for any non-default ``vocab_size`` or dimension, and
a shape mismatch counted as overlap, so the zero-overlap guard never fired.

The vocabulary here (64) is NOT any class default, so a same-vocabulary load
also proves the exception is keyed on the checkpoint, not on a class constant.
"""

import numpy as np
import pytest

from dl_techniques.layers.heads.nlp import NLPTaskConfig, NLPTaskType
from dl_techniques.layers.heads.nlp.model_head import TASK_HEAD_NAME
from dl_techniques.models.language.bert import create_bert
from dl_techniques.models.language.distilbert import create_distilbert
from dl_techniques.models.language.fnet import create_fnet
from dl_techniques.models.language.modern_bert import create_modern_bert

_CLF = NLPTaskConfig(
    name="clf", task_type=NLPTaskType.TEXT_CLASSIFICATION, num_classes=3)
_SEQ = 8
_KW = dict(vocab_size=64, hidden_size=32, num_layers=2, intermediate_size=64,
           max_position_embeddings=32)

_ENCODERS = {
    "bert": (create_bert, dict(num_heads=4)),
    "distilbert": (create_distilbert, dict(num_heads=4)),
    "modern_bert": (create_modern_bert, dict(num_heads=4)),
    "fnet": (create_fnet, {}),
}
_HEADS = {"headless": {}, "headed": dict(include_top=True, task_config=_CLF)}

pytestmark = pytest.mark.parametrize("head", sorted(_HEADS))


@pytest.fixture(params=sorted(_ENCODERS))
def encoder(request):
    return request.param


def _batch():
    ids = np.random.default_rng(0).integers(1, 64, (2, _SEQ)).astype("int32")
    return {"input_ids": ids, "attention_mask": np.ones_like(ids)}


def _build(encoder, head, pretrained=False, **overrides):
    factory, extra = _ENCODERS[encoder]
    kwargs = dict(_KW, **extra, **_HEADS[head], **overrides)
    return factory("tiny", pretrained=pretrained, **kwargs)


@pytest.fixture
def checkpoint(encoder, tmp_path):
    """A randomized checkpoint trained under a CLF head (so a head is skipped)."""
    source = _build(encoder, "headed")
    source(_batch(), training=False)
    rng = np.random.default_rng(1)
    for w in source.weights:
        w.assign(rng.normal(0.0, 0.1, size=tuple(w.shape)).astype(w.dtype))
    path = str(tmp_path / f"{encoder}.keras")
    source.save(path)
    return source, path


def _layer_values(model, name):
    return [np.asarray(w) for w in model.get_layer(name).get_weights()]


def _backbone_names(model):
    return [layer.name for layer in model.layers
            if layer.name != TASK_HEAD_NAME and layer.weights]


@pytest.mark.parametrize("overrides", [
    dict(hidden_size=64, intermediate_size=128),
    dict(intermediate_size=128),
    dict(num_layers=3),
], ids=["hidden_size", "intermediate_size", "deeper"])
def test_a_backbone_that_does_not_fit_raises(encoder, head, checkpoint, overrides):
    _, path = checkpoint
    with pytest.raises(ValueError, match="Failed to load weights|unloaded|mismatch"):
        _build(encoder, head, pretrained=path, **overrides)


def test_a_same_vocabulary_checkpoint_restores_every_layer(encoder, head, checkpoint):
    source, path = checkpoint
    target = _build(encoder, head, pretrained=path)
    names = _backbone_names(target)
    assert "embeddings" in names
    for name in names:
        for a, b in zip(_layer_values(source, name), _layer_values(target, name)):
            np.testing.assert_array_equal(a, b, err_msg=name)


def test_a_vocabulary_change_loads_everything_but_the_embeddings(
    encoder, head, checkpoint
):
    source, path = checkpoint
    target = _build(encoder, head, pretrained=path, vocab_size=96)

    names = _backbone_names(target)
    assert len(names) >= 3, names
    for name in names:
        if name == "embeddings":
            continue
        for a, b in zip(_layer_values(source, name), _layer_values(target, name)):
            np.testing.assert_array_equal(a, b, err_msg=name)
    # The vocabulary-dependent table is the target's own, fresh.
    assert _layer_values(target, "embeddings")[0].shape[0] == 96
