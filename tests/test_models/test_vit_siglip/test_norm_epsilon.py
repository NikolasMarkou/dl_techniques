"""The patch-embedding norm's epsilon, against the model's own other norms.

``_create_siglip_patch_embedding`` built its ``LayerNormalization`` directly and
so inherited Keras' default of **1e-3**, while the same model's final norm is
built through ``create_normalization_layer``, which ``setdefault``s **1e-6**
(``layers/norms/factory.py:145``). Two normalizations in one file, 1000x apart,
with nothing in the package documenting a reason — and no SigLIP reference
implementation vendored here to appeal to, which is exactly why the reference
used is the model's OWN other normalization rather than a remembered number
from the paper.

These are config assertions and are labelled as such: an epsilon cannot be seen
by any shape, count or finiteness check.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import keras

from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.models.vit_siglip.model import SigLIPVisionTransformer

KERAS_DEFAULT_EPSILON = 1e-3


def _tiny_model(**kwargs):
    return SigLIPVisionTransformer(
        input_shape=(32, 32, 3),
        num_classes=4,
        scale="base",
        patch_size=16,
        include_top=False,
        **kwargs,
    )


def _factory_epsilon() -> float:
    """The single source of truth, read rather than restated as a literal."""
    return create_normalization_layer("layer_norm", name="probe").epsilon


class TestPatchEmbedNormEpsilon:

    def test_the_factory_default_is_what_this_test_measures_against(self):
        assert _factory_epsilon() == 1e-6, (
            "create_normalization_layer's setdefault changed; the patch-embed "
            "norm follows it by construction, so update this expectation and "
            "decisions.md D-028 together"
        )

    def test_patch_embed_norm_agrees_with_the_factory(self):
        model = _tiny_model()
        norms = [
            layer
            for layer in model.siglip_patch_embed.layers
            if isinstance(layer, keras.layers.LayerNormalization)
        ]
        assert len(norms) == 1, f"expected one norm in the stem, found {len(norms)}"
        assert norms[0].epsilon == _factory_epsilon()
        assert norms[0].epsilon != KERAS_DEFAULT_EPSILON

    def test_the_stem_and_the_final_norm_now_use_one_value(self):
        """The whole point: internal consistency, since there is no vendored
        SigLIP to be faithful to. The final norm exists only in 'pre' position."""
        model = _tiny_model(normalization_position="pre")
        stem_norm = [
            layer
            for layer in model.siglip_patch_embed.layers
            if isinstance(layer, keras.layers.LayerNormalization)
        ][0]
        assert model.norm is not None
        assert stem_norm.epsilon == model.norm.epsilon
