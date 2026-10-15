"""The CliffordCLIP trainer's name lists resolve on the model they freeze.

Rationale
---------
`train/cliffordnet/train_clip.py` decides by sub-layer NAME what each
pretraining stage trains, what stays frozen, and which variables AdamW does
not decay. At 56b696fe2 four of those names (`vision_global_pool`,
`vision_global_max_pool`, `vision_query_pool`, `text_query_pool`) named no
CliffordCLIP layer: the model pools through `vision_det_pool`,
`vision_ctx_pool`, `text_det_pool` and `text_ctx_pool`. The lookups used
`getattr(model, name, None)`, so a stale name was skipped in silence, the
attention context pools were never frozen for pretraining, and the vision
classifier died on the stale attribute (plan-2026-10-14T062616-81938e81
D-054).

Checked here on a built model for every `head_kind`, with no training:

* the freeze sweep reaches every direct sub-layer of the model;
* every sub-layer is assigned to exactly one of: vision backbone, text
  backbone, always frozen;
* every literal name and name prefix in the lists matches a real sub-layer
  for at least one `head_kind` (a dead name is a stale name);
* the weight-decay exclusion matches exactly the temperature.
"""

import pytest

from dl_techniques.models.vision_language.clifford_clip import CliffordCLIP
from train.cliffordnet import train_clip as trainer

from .test_the_trainer_entry_point_trains_under_stock_fit import _TINY_ROW

_HEAD_KINDS = ("plain", "mean_max", "learned_query", "learned_query_residual")


def _built(head_kind):
    model = CliffordCLIP(vocab_size=64, image_size=64, context_length=16,
                         vision_patch_size=4, head_kind=head_kind, **_TINY_ROW)
    model.build({"image": (None, 64, 64, 3), "text": (None, 16)})
    return model


@pytest.fixture(scope="module")
def models():
    return {kind: _built(kind) for kind in _HEAD_KINDS}


def _direct_names(model):
    return {layer.name for layer in model.layers}


@pytest.mark.parametrize("head_kind", _HEAD_KINDS)
def test_the_freeze_sweep_reaches_every_sub_layer(models, head_kind):
    model = models[head_kind]
    swept = {layer.name for layer in trainer._iter_clip_sublayers(model)}
    assert swept == _direct_names(model), (
        f"never swept: {sorted(_direct_names(model) - swept)}; "
        f"not a direct sub-layer: {sorted(swept - _direct_names(model))}")
    # Independently of the layer list: every weight but the model's own
    # (`clifford_clip/logit_scale`) sits under a swept sub-layer.
    owners = {w.path.split("/")[1] for w in model.weights if w.path.count("/") >= 2}
    assert owners and owners <= swept, sorted(owners - swept)


@pytest.mark.parametrize("head_kind", _HEAD_KINDS)
def test_every_sub_layer_has_exactly_one_pretraining_role(models, head_kind):
    for name in sorted(_direct_names(models[head_kind])):
        roles = [name in trainer._PRETRAIN_ALWAYS_FROZEN,
                 trainer._is_vision_backbone(name),
                 trainer._is_text_backbone(name)]
        assert sum(roles) == 1, (name, roles)


def test_every_listed_name_names_a_real_layer(models):
    real = set().union(*(_direct_names(m) for m in models.values()))
    exact = (set(trainer._PRETRAIN_ALWAYS_FROZEN) | set(trainer._VISION_BACKBONE_NAMES)
             | set(trainer._TEXT_BACKBONE_NAMES))
    assert not sorted(exact - real), f"stale names: {sorted(exact - real)}"
    prefixes = trainer._VISION_BACKBONE_PREFIXES + trainer._TEXT_BACKBONE_PREFIXES
    dead = [p for p in prefixes if not any(name.startswith(p) for name in real)]
    assert not dead, f"prefixes matching no layer: {dead}"


@pytest.mark.parametrize("head_kind", _HEAD_KINDS)
def test_only_the_temperature_escapes_weight_decay(models, head_kind):
    model = models[head_kind]
    config = trainer.CliffordCLIPTrainConfig(image_size=64, epochs=1, batch_size=2, peak_lr=1e-3)
    optimizer = trainer._build_optimizer(config, steps_per_epoch=1)
    undecayed = [v.path for v in model.variables if not optimizer._use_weight_decay(v)]
    assert undecayed == [model.logit_scale.path], undecayed
