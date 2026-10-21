"""SAM 2's image encoder standardizes ``[0, 255]`` pixels before the trunk, as upstream's input is.

Upstream SAM 2 never feeds raw pixels to its image encoder: training applies
``NormalizeAPI(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`` after
``ToTensorAPI`` (``sam2.1_hiera_b+_MOSE_finetune.yaml``, facebookresearch/sam2
2b90b9f5) and both predictors do the same through ``SAM2Transforms``. This port
fed ``[0, 255]`` straight into Hiera's patch embedding, 255 x 0.226 (about 58)
times upstream's input scale and centred near 115 instead of 0, from all three
entry points (``SAM2.call``,
``SAM2.stream_step`` and ``SAM2TrainingModel``, which calls ``image_encoder``
itself). MEASURED (plan-2026-10-14T062616-81938e81 D-260): the trainer's
2-clip memorisation never got below the best constant logit (0.959) in 150
steps at the shipped learning rate, nor in 1000 steps at 10x it; with the
input standardized the same model and optimizer reach 0.028 at 100 steps.

The guard is on VALUES, not a scale heuristic: the encoder's output equals its
own trunk and neck applied to the hand-standardized image. The two value checks
run with TF32 off: the hand image differs from the layer's in the last float32
bit, and TF32 matmuls amplify that to 3.4e-4 on 16 of 16,384 elements (measured
on the RTX 4070), against a difference of 228 when the raw image reaches the trunk.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.vision_language.sam.sam2.neck import SAM2ImageEncoder
from tests.numerics import true_float32_matmul

#: Upstream's NormalizeAPI statistics, on the ``[0, 1]`` scale (the YAML's values).
UPSTREAM_MEAN = np.array([0.485, 0.456, 0.406], "float32")
UPSTREAM_STD = np.array([0.229, 0.224, 0.225], "float32")


def _images(seed: int = 0, batch: int = 2, size: int = 64) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0.0, 255.0, (batch, size, size, 3)).astype("float32")


def _trunk_and_neck(encoder: SAM2ImageEncoder, images: np.ndarray):
    """The encoder's own sublayers on an already-prepared input, scalp applied."""
    levels, _ = encoder.neck(encoder.trunk(images, training=False), training=False)
    return [np.asarray(level) for level in levels[: len(levels) - encoder.scalp]]


@pytest.fixture
def encoder() -> SAM2ImageEncoder:
    keras.utils.set_random_seed(0)
    layer = SAM2ImageEncoder.from_variant("tiny")
    layer.build((None, 64, 64, 3))
    return layer


class TestTheEncoderStandardizesItsInput:
    """``(pixels / 255 - mean) / std`` reaches the trunk, by value."""

    def test_the_output_is_the_trunk_and_neck_of_the_standardized_image(self, encoder) -> None:
        images = _images()
        standardized = (images / 255.0 - UPSTREAM_MEAN) / UPSTREAM_STD
        with true_float32_matmul():
            expected = _trunk_and_neck(encoder, standardized)
            got = [np.asarray(level) for level in encoder(images, training=False)["backbone_fpn"]]
        for level, (want, have) in enumerate(zip(expected, got)):
            np.testing.assert_allclose(have, want, rtol=1e-4, atol=1e-4, err_msg=f"level {level}")

    def test_the_raw_image_is_not_what_reaches_the_trunk(self, encoder) -> None:
        images = _images(seed=1)
        raw = _trunk_and_neck(encoder, images)
        got = np.asarray(encoder(images, training=False)["backbone_fpn"][0])
        assert np.abs(got - raw[0]).max() > 1.0

    def test_the_statistics_are_config_and_round_trip(self, encoder) -> None:
        config = encoder.get_config()
        np.testing.assert_allclose(config["pixel_mean"], UPSTREAM_MEAN * 255.0, rtol=1e-6)
        np.testing.assert_allclose(config["pixel_std"], UPSTREAM_STD * 255.0, rtol=1e-6)
        rebuilt = SAM2ImageEncoder.from_config(config)
        assert rebuilt.get_config()["pixel_mean"] == config["pixel_mean"]
        assert rebuilt.get_config()["pixel_std"] == config["pixel_std"]

    def test_identity_statistics_feed_the_raw_image(self, encoder) -> None:
        config = dict(encoder.get_config(), pixel_mean=(0.0, 0.0, 0.0), pixel_std=(1.0, 1.0, 1.0))
        identity = SAM2ImageEncoder.from_config(config)
        identity.build((None, 64, 64, 3))
        identity.set_weights(encoder.get_weights())
        images = _images(seed=2)
        with true_float32_matmul():
            expected = _trunk_and_neck(encoder, images)
            got = [np.asarray(level) for level in identity(images, training=False)["backbone_fpn"]]
        for want, have in zip(expected, got):
            np.testing.assert_allclose(have, want, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("bad", [{"pixel_mean": (1.0, 2.0)}, {"pixel_std": (1.0, 0.0, 1.0)}])
    def test_malformed_statistics_are_refused(self, encoder, bad) -> None:
        with pytest.raises(ValueError, match="pixel_"):
            SAM2ImageEncoder(trunk=encoder.trunk, neck=encoder.neck, scalp=encoder.scalp, **bad)
