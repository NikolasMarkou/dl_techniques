"""The CLS token of vit, vit_siglip and vit_hmlp is a ``ClassTokenPrepend`` layer.

The three models used to create a model-level ``cls_token`` weight in ``build``
and broadcast/concatenate it by hand in ``call``, a copy of what
``ClassTokenPrepend`` does for dino and beit. They now own that layer. This pins
what the replacement must keep: the layer is named ``cls_token`` (weight path
``cls_token/cls_token``), holds one ``(1, 1, embed_dim)`` weight, sits at
sequence position 0, and keeps each model's initializer (zeros for vit and
vit_siglip, ``RandomNormal(stddev=0.02)`` for vit_hmlp).
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.embedding.class_token import ClassTokenPrepend
from dl_techniques.models.vision.vit.model import ViT
from dl_techniques.models.vision.vit_hmlp.model import ViTHMLP
from dl_techniques.models.vision.vit_siglip.model import SigLIPVisionTransformer

SHAPE = (32, 32, 3)

SUBJECTS = {
    "vit": lambda: ViT(input_shape=SHAPE, scale="tiny", patch_size=16, include_top=False, pooling=None),
    "vit_siglip": lambda: SigLIPVisionTransformer(
        input_shape=SHAPE, scale="tiny", patch_size=16, include_top=False, pooling=None),
    "vit_hmlp": lambda: ViTHMLP(
        input_shape=SHAPE, scale="tiny", patch_size=16, include_top=False, pooling=None),
}


@pytest.fixture(scope="module", params=sorted(SUBJECTS))
def built(request):
    model = SUBJECTS[request.param]()
    model(np.zeros((1,) + SHAPE, dtype="float32"))
    return request.param, model


def test_cls_token_is_a_class_token_prepend_layer(built):
    _, model = built
    assert isinstance(model.cls_token, ClassTokenPrepend)
    assert model.cls_token.name == "cls_token"
    owned = [w for w in model.weights if w.path.endswith("cls_token/cls_token")]
    assert len(owned) == 1, [w.path for w in model.weights if "cls_token" in w.path]
    assert tuple(owned[0].shape) == (1, 1, model.embed_dim)
    # no second, model-level copy of the weight
    assert [w.path for w in model.weights
            if w.path.endswith("/cls_token") and not w.path.endswith("cls_token/cls_token")] == []


def test_cls_token_sits_at_position_zero_and_is_input_independent(built):
    _, model = built
    token = np.asarray(model.cls_token.cls_token)[0, 0]
    patches = np.random.default_rng(0).normal(size=(2, model.num_patches, model.embed_dim)).astype("float32")
    out = np.asarray(model.cls_token(patches))
    assert out.shape == (2, model.num_patches + 1, model.embed_dim)
    np.testing.assert_array_equal(out[:, 0], np.broadcast_to(token, (2, model.embed_dim)))
    np.testing.assert_array_equal(out[:, 1:], patches)


def test_each_model_keeps_its_cls_initializer():
    vit = SUBJECTS["vit"]()
    siglip = SUBJECTS["vit_siglip"]()
    hmlp = SUBJECTS["vit_hmlp"]()
    for m in (vit, siglip, hmlp):
        m(np.zeros((1,) + SHAPE, dtype="float32"))
    for m in (vit, siglip):
        assert isinstance(m.cls_token.initializer, keras.initializers.Zeros)
        assert not np.any(np.asarray(m.cls_token.cls_token))
    init = hmlp.cls_token.initializer
    assert isinstance(init, keras.initializers.RandomNormal) and init.stddev == 0.02 and init.mean == 0.0
    draw = np.asarray(hmlp.cls_token.cls_token)
    assert 0.0 < float(draw.std()) < 0.1 and float(np.abs(draw).max()) < 0.2
