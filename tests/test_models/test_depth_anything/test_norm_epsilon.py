"""BatchNorm epsilon in the DPT head and the placeholder encoder.

All four ``BatchNormalization`` constructions in this package were built with no
``epsilon`` and so ran at Keras' default of **1e-3**. The DPT dense-prediction
head follows the Depth Anything V1 recipe (``model.py``'s module docstring
enumerates the three deliberate departures from the paper; normalization is not
among them), and that reference is PyTorch, whose ``nn.BatchNorm2d`` defaults to
**1e-5**. The same torch-vs-Keras fact is written down for the other port in
this repo at ``layers/fastvit/reference.py``.

The three encoder sites are a weaker case and are treated as such in D-028: the
placeholder encoder is an in-repo stand-in for DINOv2 with no reference
implementation, so its epsilon is chosen for consistency with the head it feeds,
not for fidelity. This module asserts consistency, which is the claim that
actually holds for both.

Config assertions, labelled as such — an epsilon is invisible to every shape and
finiteness check in the suite.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import keras

from dl_techniques.models.depth_anything.components import (
    DPTDecoder,
    REFERENCE_BN_EPSILON,
)

KERAS_DEFAULT_EPSILON = 1e-3


def _batch_norms(layer):
    return [
        sublayer
        for sublayer in layer._flatten_layers(include_self=False)
        if isinstance(sublayer, keras.layers.BatchNormalization)
    ]


class TestDPTHeadEpsilon:

    def test_the_constant_is_the_torch_batchnorm_default(self):
        assert REFERENCE_BN_EPSILON == 1e-5
        assert REFERENCE_BN_EPSILON != KERAS_DEFAULT_EPSILON

    def test_every_batch_norm_in_the_decoder_carries_it(self):
        decoder = DPTDecoder(dims=[8, 16], output_channels=1)
        norms = _batch_norms(decoder)
        assert len(norms) == 2, f"expected one BN per dim, found {len(norms)}"
        for norm in norms:
            assert norm.epsilon == REFERENCE_BN_EPSILON, (
                f"{norm.name} is at {norm.epsilon}; Keras' 1e-3 default is 100x "
                f"the torch reference this head follows"
            )

    def test_it_survives_a_config_round_trip(self):
        decoder = DPTDecoder(dims=[8, 16], output_channels=1)
        clone = DPTDecoder.from_config(decoder.get_config())
        assert all(n.epsilon == REFERENCE_BN_EPSILON for n in _batch_norms(clone))


class TestPlaceholderEncoderAgreesWithTheHead:
    """Consistency, not fidelity — the encoder has no reference to be faithful to."""

    def test_placeholder_encoder_batch_norms_use_the_same_value(self):
        from dl_techniques.models.depth_anything.model import DepthAnything

        model = DepthAnything(
            input_shape=(64, 64, 3),
            encoder_kind="placeholder",
            decoder_dims=[8, 16, 16, 16],
        )
        # The encoder is constructed lazily in `build()`, not in `__init__`.
        model.build((None, 64, 64, 3))
        norms = [
            layer
            for layer in model.encoder.layers
            if isinstance(layer, keras.layers.BatchNormalization)
        ]
        assert norms, "the placeholder encoder should contain BatchNormalizations"
        for norm in norms:
            assert norm.epsilon == REFERENCE_BN_EPSILON
