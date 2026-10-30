"""Every FastViT residual branch builds the same unconstrained per-channel LayerScale.

The four sites (``FastVitRepMixer``, ``FastVitRepMixerBlock``, and the two in
``FastVitAttentionBlock``) construct ``LayerScale`` inline. ``LayerScale``
defaults to ``constraint='non_neg'``, which clamps gamma at zero, so each site
must pass ``constraint=None``. ``test_attention_block`` pins the attention
block's two; this file pins the other two and the ``None`` disable path.
"""

import keras
import pytest

from dl_techniques.layers.regularization.layer_scale import LayerScale
from dl_techniques.models.vision.fastvit.components import (
    FastVitAttentionBlock,
    FastVitRepMixer,
    FastVitRepMixerBlock,
)

SITES = {
    "rep_mixer": (lambda **kw: FastVitRepMixer(dim=8, **kw), ["layer_scale"]),
    "rep_mixer_block": (lambda **kw: FastVitRepMixerBlock(dim=8, **kw), ["layer_scale"]),
    "attention_block": (
        lambda **kw: FastVitAttentionBlock(dim=64, head_dim=32, mlp_ratio=2.0, **kw),
        ["layer_scale_1", "layer_scale_2"],
    ),
}


@pytest.mark.parametrize("site", sorted(SITES))
def test_layer_scale_is_per_channel_constant_and_unconstrained(site):
    make, attrs = SITES[site]
    layer = make(layer_scale_init_value=0.25)
    for attr in attrs:
        scale = getattr(layer, attr)
        assert isinstance(scale, LayerScale)
        assert scale.name == attr
        assert scale.constraint is None
        assert isinstance(scale.initializer, keras.initializers.Constant)
        assert scale.initializer.value == 0.25


@pytest.mark.parametrize("site", sorted(SITES))
def test_layer_scale_none_creates_no_layer(site):
    make, attrs = SITES[site]
    layer = make(layer_scale_init_value=None)
    for attr in attrs:
        assert getattr(layer, attr) is None
