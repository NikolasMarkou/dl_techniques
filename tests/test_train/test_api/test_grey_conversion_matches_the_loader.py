"""A colour array and a PIL image become grey the way the loader makes the same PNG grey (D-012).

The training loader reads a file with ``tf.io.decode_image(channels=1)``, so the colour to grey
conversion happens inside the decoder, on 8-bit values, with libpng's integer weights
(``png_set_rgb_to_gray(0.299, 0.587)``: ``(9797 R + 19234 G + 3737 B) >> 15``, measured over
65,536 random colours with no mismatch). ``tf.image.rgb_to_grayscale`` (what the predictor used)
is float luma with other weights and differs by one level in a quarter of the pixels, and PIL's
``convert("L")`` rounds and differs in half of them. Pinned here:

* for lossless sources (PNG: RGB, RGBA, palette, and a float array of 8-bit values) the array,
  the PIL image and the path all give the loader's pixels within 1e-5;
* a model that wants 3 channels is unchanged;
* the control: the old call, applied to the same pixels, is NOT within 1e-5, so the fixture can
  tell the two apart;
* a JPEG read as a PATH is exactly the loader's decode; a JPEG already decoded to an array or a
  PIL image cannot reproduce the decoder's Y plane, and its measured residual is pinned (it is
  stated, not hidden, in the predictor's docstring and ``src/train/README.md``).

Tests write only under ``tmp_path``.
"""

import io
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from train.api.predictor import Predictor
from train.common.classification_data import GREY_WEIGHTS, decode_image_file, rgb8_to_grey8

TOLERANCE = 1e-5

#: The measured JPEG residual of decoded pixels against the loader's decode of the same file
#: (levels of 255), on the random-colour fixture below: the worst pixel and the mean over pixels.
JPEG_MAX_LEVELS = 20
JPEG_MEAN_LEVELS = 1.5


def predictor(channels: int) -> Predictor:
    """A predictor that only has the record ``_one`` reads (no model is needed to convert an image)."""
    instance = Predictor.__new__(Predictor)
    instance.record = {"eval_transform": {"kind": "resize", "source_channels": channels}}
    return instance


def colours(shape=(41, 29, 3), seed=0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)


def loader(path: Path, channels: int) -> np.ndarray:
    return decode_image_file(str(path), channels).numpy()


def worst(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


@pytest.mark.parametrize("shape", [(41, 29, 3), (64, 64, 3), (7, 130, 3)])
def test_a_png_of_random_colours_is_grey_by_the_loaders_pixels_from_array_pil_and_path(tmp_path, shape):
    array = colours(shape)
    path = tmp_path / "colours.png"
    Image.fromarray(array).save(path)
    want = loader(path, 1)
    assert want.shape == (shape[0], shape[1], 1)
    one = predictor(1)
    assert worst(one._one(array), want) <= TOLERANCE
    assert worst(one._one(Image.open(path)), want) <= TOLERANCE
    assert worst(one._one(str(path)), want) <= TOLERANCE


def test_the_integer_weights_are_the_decoders_over_every_primary_axis(tmp_path):
    """Exhaustive on the three axes and a dense random cube: the weights and the shift are exact."""
    cube = colours((256, 256, 3), seed=1)
    ramp = np.arange(256, dtype=np.uint8)
    axes = np.stack([np.stack([ramp, np.zeros_like(ramp), np.zeros_like(ramp)], -1),
                     np.stack([np.zeros_like(ramp), ramp, np.zeros_like(ramp)], -1),
                     np.stack([np.zeros_like(ramp), np.zeros_like(ramp), ramp], -1)])
    for name, pixels in (("cube", cube), ("axes", axes)):
        path = tmp_path / f"{name}.png"
        Image.fromarray(pixels).save(path)
        decoded = np.rint(loader(path, 1) * 255.0).astype(np.uint8)
        assert np.array_equal(rgb8_to_grey8(pixels), decoded), name
    assert sum(GREY_WEIGHTS) == 1 << 15


def test_the_old_call_is_not_the_loaders_pixels_so_the_fixture_can_tell(tmp_path):
    array = colours()
    path = tmp_path / "colours.png"
    Image.fromarray(array).save(path)
    old = tf.image.rgb_to_grayscale(tf.image.convert_image_dtype(array, tf.float32)).numpy()
    assert worst(old, loader(path, 1)) > TOLERANCE
    pil_l = np.asarray(Image.fromarray(array).convert("L"), np.float32)[..., None] / 255.0
    assert worst(pil_l, loader(path, 1)) > TOLERANCE


def test_rgba_and_palette_pngs_are_grey_by_the_loaders_pixels(tmp_path):
    rgba = np.concatenate([colours((20, 33, 3), seed=2), colours((20, 33, 1), seed=3)], -1)
    rgba_path = tmp_path / "rgba.png"
    Image.fromarray(rgba, "RGBA").save(rgba_path)
    palette_path = tmp_path / "palette.png"
    Image.fromarray(colours((20, 33, 3), seed=4)).quantize(colors=64).save(palette_path)
    one = predictor(1)
    for path in (rgba_path, palette_path):
        want = loader(path, 1)
        assert worst(one._one(Image.open(path)), want) <= TOLERANCE, path.name
        assert worst(one._one(str(path)), want) <= TOLERANCE, path.name
    assert worst(one._one(rgba), loader(rgba_path, 1)) <= TOLERANCE


def test_a_float_array_of_eight_bit_values_matches_the_png_of_those_values(tmp_path):
    array = colours()
    path = tmp_path / "colours.png"
    Image.fromarray(array).save(path)
    floats = array.astype(np.float32) / 255.0
    assert worst(predictor(1)._one(floats), loader(path, 1)) <= TOLERANCE


def test_a_grey_image_is_unchanged_and_three_channels_stay_what_they_were(tmp_path):
    grey = colours((18, 25), seed=5)
    grey_path = tmp_path / "grey.png"
    Image.fromarray(grey, "L").save(grey_path)
    assert worst(predictor(1)._one(Image.open(grey_path)), loader(grey_path, 1)) <= TOLERANCE
    assert worst(predictor(1)._one(grey), loader(grey_path, 1)) <= TOLERANCE
    colour = colours()
    colour_path = tmp_path / "colour.png"
    Image.fromarray(colour).save(colour_path)
    three = predictor(3)
    assert worst(three._one(colour), loader(colour_path, 3)) <= TOLERANCE
    assert worst(three._one(Image.open(colour_path)), loader(colour_path, 3)) <= TOLERANCE
    assert three._one(grey).shape == (18, 25, 3)  # a grey array still repeats to three channels


def jpeg_of(array, quality):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, "JPEG", quality=quality)
    return buffer.getvalue()


@pytest.mark.parametrize("quality", [75, 95])
def test_a_jpeg_path_is_the_loaders_decode_and_decoded_pixels_carry_a_measured_residual(tmp_path, quality):
    path = tmp_path / "photo.jpg"
    path.write_bytes(jpeg_of(colours((64, 64, 3), seed=6), quality))
    want = loader(path, 1)
    one = predictor(1)
    assert worst(one._one(str(path)), want) == 0.0  # the file is decoded by the loader's own call
    decoded = Image.open(path)
    residual = np.abs(one._one(decoded).numpy() - want) * 255.0
    # The Y plane is gone once the decoder has returned RGB: the residual is real, and bounded.
    assert residual.max() > TOLERANCE * 255.0
    assert residual.max() <= JPEG_MAX_LEVELS and residual.mean() <= JPEG_MEAN_LEVELS
    from_array = np.abs(one._one(np.asarray(decoded.convert("RGB"))).numpy() - want) * 255.0
    assert from_array.max() <= JPEG_MAX_LEVELS and from_array.mean() <= JPEG_MEAN_LEVELS
