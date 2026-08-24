"""The text tower's final LayerNorm epsilon, against the port's own constant.

``MobileClipTextEncoder`` is the OpenCLIP-shaped text transformer that
``mobile_clip_v2.py`` pairs with the FastViT MCi tower to make "the faithful
MobileCLIP port". Its ``final_layer_norm`` is OpenCLIP's ``ln_final``, i.e. a
PyTorch ``nn.LayerNorm``, whose epsilon defaults to 1e-5 — the value
``layers/fastvit/reference.py`` already defines for every normalization in this
port, with an interface contract saying it must be passed EXPLICITLY at every
construction site because Keras' default (1e-3) is 100x larger and silent.

The layer was constructing it with no ``epsilon`` at all, so the one norm the
text tower owns was the one place in the port that ignored that contract.

Config assertions, labelled as such.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import keras

from dl_techniques.layers.fastvit.reference import REFERENCE_NORM_EPSILON
from dl_techniques.models.mobile_clip.components import MobileClipTextEncoder

KERAS_DEFAULT_EPSILON = 1e-3


def _encoder(**kwargs):
    return MobileClipTextEncoder(
        vocab_size=64,
        max_seq_len=8,
        embed_dim=16,
        num_layers=1,
        num_heads=2,
        intermediate_size=32,
        projection_dim=8,
        **kwargs,
    )


class TestTextTowerFinalNormEpsilon:

    def test_the_shared_reference_constant_is_the_torch_default(self):
        assert REFERENCE_NORM_EPSILON == 1e-5

    def test_final_layer_norm_carries_the_reference_epsilon(self):
        encoder = _encoder()
        assert isinstance(encoder.layer_norm, keras.layers.LayerNormalization)
        assert encoder.layer_norm.epsilon == REFERENCE_NORM_EPSILON
        assert encoder.layer_norm.epsilon != KERAS_DEFAULT_EPSILON, (
            "Keras' default is 100x the OpenCLIP reference and is what this "
            "test exists to keep out"
        )

    def test_it_is_imported_not_re_declared(self):
        """The constant has ONE definition; a local copy would drift silently.

        ``layers/fastvit/reference.py``'s own module docstring states this as an
        interface contract for consumers, and this package is now one.
        """
        import dl_techniques.models.mobile_clip.components as components

        assert components.REFERENCE_NORM_EPSILON is REFERENCE_NORM_EPSILON

    def test_it_survives_a_config_round_trip(self):
        encoder = _encoder()
        clone = MobileClipTextEncoder.from_config(encoder.get_config())
        assert clone.layer_norm.epsilon == REFERENCE_NORM_EPSILON
