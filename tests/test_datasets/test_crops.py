"""``dl_techniques.datasets.vision.crops``: the shared image geometry and its constants."""

import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.datasets.vision import crops
from dl_techniques.datasets.vision import imagenet2012, imagenette


def _image(h=12, w=16):
    return tf.reshape(tf.range(h * w * 3, dtype=tf.float32), (h, w, 3)) / (h * w * 3)


def test_the_constants_are_the_imagenet_classifier_recipe() -> None:
    assert crops.EVAL_CROP_FRACTION == 0.875
    assert crops.RANDOM_RESIZED_CROP_AREA == (0.08, 1.0)
    assert crops.RANDOM_RESIZED_CROP_RATIO == (3.0 / 4.0, 4.0 / 3.0)


def test_the_loaders_read_the_constants_from_crops_not_from_a_copy() -> None:
    assert imagenet2012.EVAL_CROP_FRACTION is crops.EVAL_CROP_FRACTION
    assert imagenet2012.RANDOM_RESIZED_CROP_AREA is crops.RANDOM_RESIZED_CROP_AREA
    assert imagenet2012.RANDOM_RESIZED_CROP_RATIO is crops.RANDOM_RESIZED_CROP_RATIO
    assert imagenette.random_resized_crop is crops.random_resized_crop
    assert imagenette.resize_and_centre_crop is crops.resize_and_centre_crop


@pytest.mark.parametrize("fn", [
    lambda image: crops.random_resized_crop(image, 8, seed=1),
    lambda image: crops.resize_and_centre_crop(image, 8),
])
def test_the_crops_return_the_requested_square(fn) -> None:
    assert fn(_image()).shape == (8, 8, 3)


def test_flip_and_pad_crop_keeps_the_shape_and_with_no_pad_and_no_flip_the_image() -> None:
    image = _image()
    assert crops.flip_and_pad_crop(image, pad=4, mode="REFLECT").shape == image.shape
    np.testing.assert_array_equal(
        crops.flip_and_pad_crop(image, pad=0, flip=False).numpy(), image.numpy())


def test_flip_and_pad_crop_flips_when_asked_and_only_then() -> None:
    image = _image()
    flipped = {tuple(np.asarray(crops.flip_and_pad_crop(image, pad=0, flip=True)[0, :, 0]))
               for _ in range(40)}
    assert len(flipped) == 2, "both the identity and the mirror image must occur"
    kept = {tuple(np.asarray(crops.flip_and_pad_crop(image, pad=0, flip=False)[0, :, 0]))
            for _ in range(40)}
    assert len(kept) == 1


def test_a_seed_makes_the_pipeline_reproducible() -> None:
    image = _image()

    def run(seed):
        ds = tf.data.Dataset.from_tensors(image).repeat(6).map(
            lambda x: crops.flip_and_pad_crop(x, mode="REFLECT", seed=seed))
        return np.stack([x.numpy() for x in ds])

    np.testing.assert_array_equal(run(9), run(9))
