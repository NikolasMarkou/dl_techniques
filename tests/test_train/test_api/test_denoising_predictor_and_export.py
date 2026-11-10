"""Predict with an ``image_denoising`` run, the PNGs ``predict`` writes, and ``export`` (plan d483ffab step 12, D-030, D-031).

Predictor (the five tiny families, built from the recipe with a fixed seed, never trained: nothing here depends on a
trained value):

* a path, a PIL image, a ``uint8`` array, a 4-D array and a list of them give ONE answer (bit-equal), a float array
  the same within float rounding; a colour image given to a one-channel model is made grey with the loader's rule;
* the answer is ``{"denoised": [...]}``, float32, at the size of each input, for odd sizes, sizes under the
  stride and a batch of mixed sizes, and repeats exactly; it is RAW (a model that outputs outside ``[0, 1]`` is
  returned outside it) and equals the model's own call on the reflect-padded image, cropped;
* overlap tiling: a pointwise model is reproduced exactly by every tiling (coverage, crop, weights that sum to one); a
  non-local model (an image minus its own mean) shows a seam at non-overlapping tile edges that the overlap blends
  away; on the real families the tiled answer is within a MEASURED bound of the whole-image one, the overlap
  lowers that difference and the error in the bands around the seams; the bias-free families keep ``f(a x) = a f(x)``
  under tiling and a biased one does not; a tile that is not a multiple of the family's ``size_multiple`` is refused;
  an image smaller than the tile is one whole pass; a GPU out of memory is ONE error type, caught and raised as a
  ``ConfigError`` naming ``tile``;
* ``predict`` writes one PNG per input (the raw output clipped to ``[0, 1]`` at the byte boundary only) and
  ``index.json`` LAST; ``out`` and the stems are checked before the model is loaded.

Export: the table family x format x {fixed, dynamic}. Every cell is either a numeric match against Keras
(``training=False``, CPU, relative to the scale of the output) with the artifact answering the same twice on one runtime
object, or one ``ConfigError`` carrying the measured reason; each refused cell has a test that builds the file anyway
and fails if the reason stops being true. The ``keras`` and ``saved_model`` exports of the bias-free families are read
again for ``f(a x) = a f(x)``. Everything writes under ``tmp_path``; nothing is deleted.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import keras
import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

import train.api.recipes as recipes
from train.api import cli, exporting, predictor, registry
from train.api.denoising_predictor import (
    EXPORT_REFUSALS,
    DenoisingPredictor,
    ramp_weights,
    reflect_pad,
    tile_starts,
    to_uint8,
)
from train.api.errors import ConfigError
from train.api.exporting import FORMATS, export
from train.api.predictor import DENOISING, build_denoising_record, load_model

from .test_denoising_families import build

FAMILIES = ("bfcnn", "bfunet", "bfconvunext", "convunext_biased", "scunet")
BIAS_FREE = ("bfcnn", "bfunet", "bfconvunext")

#: ``size_multiple`` of each tiny family (the stride of its U-Net; 1 for ``bfcnn`` and ``scunet``).
MULTIPLE = {"bfcnn": 1, "bfunet": 8, "bfconvunext": 8, "convunext_biased": 8, "scunet": 1}

ALPHAS = (0.3, 1.3, 3.7, 6.1)

#: The homogeneity tolerance of the probe (D-019, ``denoising_homogeneity.PROBE_TOLERANCE``).
HOMOGENEITY = 1e-3


def record_of(family: str, channels: int = 3) -> Dict[str, Any]:
    return build_denoising_record(
        model=f"{family}/tiny", family=family, variant="tiny", channels=channels, size_multiple=MULTIPLE[family]
    )


@pytest.fixture(scope="module")
def world():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(registry, "MODELS", {})
        patch.setattr(registry, "TASKS", {})
        recipes.register_builtin_models()
        yield


@pytest.fixture(scope="module")
def models(world):
    """One flexible tiny model per family, built once with a fixed seed (the bounds below were measured on these)."""
    built = {}
    for family in FAMILIES:
        keras.utils.set_random_seed(0)
        built[family] = build(family)
    return built


@pytest.fixture(scope="module")
def predictors(models):
    return {family: DenoisingPredictor(models[family], record_of(family)) for family in FAMILIES}


def measured_image(height: int = 112, width: int = 128, seed: int = 1) -> np.ndarray:
    """The float32 image the tiling bounds were measured on."""
    rng = np.random.default_rng(seed)
    return np.clip(0.5 + 0.2 * rng.standard_normal((height, width, 3)), 0, 1).astype(np.float32)


def write_png(path: Path, height: int, width: int, seed: int = 0) -> np.ndarray:
    """Write a random RGB picture (the extension of ``path`` names the format) and return its pixels."""
    pixels = np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path)
    return pixels


def lambda_model(function, channels: int = 3) -> keras.Model:
    """A weight-free functional model with a free height and width."""
    inputs = keras.Input((None, None, channels))
    return keras.Model(inputs, keras.layers.Lambda(function)(inputs))


def fake_predictor(function, multiple: int = 1, channels: int = 3) -> DenoisingPredictor:
    record = build_denoising_record(
        model="bfcnn/tiny", family="bfcnn", variant="tiny", channels=channels, size_multiple=multiple
    )
    return DenoisingPredictor(lambda_model(function, channels), record)


# ---------------------------------------------------------------------
# One answer for a path, a PIL image and an array
# ---------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_a_path_a_pil_image_and_an_array_give_one_answer(family, predictors, tmp_path):
    path = tmp_path / "a.png"
    pixels = write_png(path, 37, 45)
    p = predictors[family]
    by_path = p.predict(str(path))["denoised"][0]
    by_pil = p.predict(Image.open(path))["denoised"][0]
    by_array = p.predict(pixels)["denoised"][0]
    by_path_object = p.predict(path)["denoised"][0]
    np.testing.assert_array_equal(by_path, by_pil)
    np.testing.assert_array_equal(by_path, by_array)
    np.testing.assert_array_equal(by_path, by_path_object)
    # A floating array of the same picture is the same numbers up to the division by 255 (a ulp of the input).
    by_float = p.predict(pixels.astype(np.float32) / 255.0)["denoised"][0]
    np.testing.assert_allclose(by_float, by_path, atol=1e-4 * max(1.0, float(np.abs(by_path).max())), rtol=0)
    # A batch and a list are the same single answers.
    several = p.predict([pixels, str(path), Image.open(path)])["denoised"]
    assert len(several) == 3
    for one in several:
        np.testing.assert_array_equal(one, by_path)
    np.testing.assert_array_equal(p.predict(pixels[None])["denoised"][0], by_path)


def test_a_grey_model_makes_grey_with_the_loaders_rule_for_every_form_of_the_input(tmp_path, world):
    keras.utils.set_random_seed(0)
    model = build("bfcnn", input_shape=(None, None, 1))
    p = DenoisingPredictor(model, record_of("bfcnn", channels=1))
    path = tmp_path / "rgb.png"
    pixels = write_png(path, 20, 24, seed=3)
    by_path = p.predict(str(path))["denoised"][0]
    assert by_path.shape == (20, 24, 1)
    np.testing.assert_array_equal(p.predict(Image.open(path))["denoised"][0], by_path)
    np.testing.assert_array_equal(p.predict(pixels)["denoised"][0], by_path)
    rgba = np.concatenate([pixels, np.full((20, 24, 1), 7, np.uint8)], axis=-1)
    np.testing.assert_array_equal(p.predict(rgba)["denoised"][0], by_path)  # the alpha is dropped
    grey_path = tmp_path / "grey.png"
    grey = np.random.default_rng(4).integers(0, 256, (20, 24), dtype=np.uint8)
    Image.fromarray(grey).save(grey_path)
    on_grey = p.predict(str(grey_path))["denoised"][0]
    np.testing.assert_array_equal(p.predict(grey)["denoised"][0], on_grey)
    np.testing.assert_array_equal(p.predict(grey[..., None])["denoised"][0], on_grey)
    np.testing.assert_array_equal(p.predict(Image.open(grey_path))["denoised"][0], on_grey)


def test_a_grey_input_is_repeated_for_a_three_channel_model(predictors, tmp_path):
    grey = np.random.default_rng(5).integers(0, 256, (16, 16), dtype=np.uint8)
    p = predictors["bfcnn"]
    np.testing.assert_array_equal(
        p.predict(grey)["denoised"][0], p.predict(np.repeat(grey[..., None], 3, axis=-1))["denoised"][0]
    )


def test_bad_inputs_are_one_readable_refusal_each(predictors, tmp_path):
    p = predictors["bfcnn"]
    text = tmp_path / "x.png"
    text.write_text("not an image")
    cases = {
        "an integer": (3, "expected a path, a PIL image"),
        "a missing file": (str(tmp_path / "nope.png"), "is not a file"),
        "a file that is not an image": (str(text), "is not an image file"),
        "an int64 array": (np.zeros((4, 4, 3), np.int64), "uint8 or floating point"),
        "two channels": (np.zeros((4, 4, 2), np.float32), r"\(H, W\), \(H, W, 1\), \(H, W, 3\) or \(H, W, 4\)"),
        "an empty side": (np.zeros((0, 4, 3), np.float32), r"got shape \(0, 4, 3\)"),
        "a NaN": (np.full((4, 4, 3), np.nan, np.float32), "NaN or an infinite"),
        "an infinity": (np.full((4, 4, 3), np.inf, np.float32), "NaN or an infinite"),
        "an empty list": ([], "the list of images is empty"),
        "an empty batch": (np.zeros((0, 4, 4, 3), np.float32), "the array of images is empty"),
        "a 32-bit PIL image": (Image.new("I", (4, 4)), "holds 32-bit values"),
    }
    for name, (value, message) in cases.items():
        with pytest.raises(ConfigError, match=message):
            p.predict(value)
    from .test_task_json_schema_v2 import classification

    with pytest.raises(ConfigError, match="serves 'image_denoising' only"):
        DenoisingPredictor(None, classification())


def test_a_float_array_outside_the_unit_interval_is_taken_as_it_is(predictors):
    # A denoiser is fed noisy images, and a run with data.noise_clip=false trained on values outside [0, 1].
    noisy = np.random.default_rng(6).normal(0.5, 0.4, (16, 16, 3)).astype(np.float32)
    assert noisy.min() < 0 and noisy.max() > 1
    out = predictors["bfcnn"].predict(noisy)["denoised"][0]
    assert out.shape == (16, 16, 3) and np.isfinite(out).all()


# ---------------------------------------------------------------------
# The answer: size, rawness, the crop back
# ---------------------------------------------------------------------

SIZES = [(37, 45), (8, 8), (3, 5), (1, 1), (64, 33), (50, 8)]


@pytest.mark.parametrize("family", FAMILIES)
def test_the_output_has_the_size_of_each_input_and_repeats_exactly(family, predictors):
    p = predictors[family]
    rng = np.random.default_rng(7)
    images = [rng.random((h, w, 3)).astype(np.float32) for h, w in SIZES]
    answer = p.predict(images)
    assert list(answer) == ["denoised"] and isinstance(answer["denoised"], list) and len(answer["denoised"]) == len(SIZES)
    for (h, w), out in zip(SIZES, answer["denoised"]):
        assert out.shape == (h, w, 3) and out.dtype == np.float32 and np.isfinite(out).all(), (h, w)
    again = p.predict(images)["denoised"]
    for first, second in zip(answer["denoised"], again):
        np.testing.assert_array_equal(first, second)  # twice, on one object: identical
    one = p.predict(images[0])
    assert isinstance(one["denoised"], list) and len(one["denoised"]) == 1  # a single image is still a list


@pytest.mark.parametrize("family", FAMILIES)
def test_the_answer_is_the_models_own_call_on_the_reflect_padded_image_cropped_back(family, predictors, models):
    image = np.random.default_rng(8).random((37, 45, 3)).astype(np.float32)
    padded = reflect_pad(image, MULTIPLE[family])
    assert padded.shape[0] % MULTIPLE[family] == 0 and padded.shape[1] % MULTIPLE[family] == 0
    assert padded.shape[0] - image.shape[0] < max(1, MULTIPLE[family]) and padded.shape[1] - image.shape[1] < max(1, MULTIPLE[family])
    np.testing.assert_array_equal(padded[:37, :45], image)  # the padding is at the end of each side
    expected = np.asarray(models[family](padded[None], training=False))[0][:37, :45]
    got = predictors[family].predict(image)["denoised"][0]
    assert got.shape == (37, 45, 3)
    np.testing.assert_array_equal(got, expected.astype(np.float32))


def test_the_pad_reflects_and_a_side_already_on_the_stride_is_left_alone():
    image = np.arange(5 * 3 * 1, dtype=np.float32).reshape(5, 3, 1)
    padded = reflect_pad(image, 4)
    assert padded.shape == (8, 4, 1)
    np.testing.assert_array_equal(padded[5:, :3, 0], image[3:0:-1, :, 0])  # reflect without repeating the edge row
    assert reflect_pad(image[:4, :], 4).shape == (4, 4, 1)
    flat = np.zeros((8, 8, 1), np.float32)
    assert reflect_pad(flat, 8) is flat


def test_the_output_is_raw_it_is_never_clipped_or_rescaled():
    p = fake_predictor(lambda t: 2.0 * t - 0.5)
    image = np.random.default_rng(9).random((11, 13, 3)).astype(np.float32)
    out = p.predict(image)["denoised"][0]
    assert out.min() < 0.0 and out.max() > 1.0  # the control: the model really leaves the unit interval
    np.testing.assert_allclose(out, 2.0 * image - 0.5, atol=1e-6, rtol=0)


def test_a_model_with_two_outputs_or_another_shape_is_refused_by_the_predictor(world):
    inputs = keras.Input((None, None, 3))
    two = keras.Model(inputs, [keras.layers.Lambda(lambda t: t)(inputs), keras.layers.Lambda(lambda t: t)(inputs)])
    with pytest.raises(ConfigError, match="2 outputs"):
        DenoisingPredictor(two, record_of("bfcnn")).predict(np.zeros((8, 8, 3), np.float32))
    grey = lambda_model(lambda t: t[..., :1])
    with pytest.raises(ConfigError, match=r"outputs \(1, 8, 8, 1\) for an input of shape \(1, 8, 8, 3\)"):
        DenoisingPredictor(grey, record_of("bfcnn")).predict(np.zeros((8, 8, 3), np.float32))


# ---------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------


def test_the_tile_starts_cover_every_pixel_and_the_last_tile_ends_at_the_edge():
    assert tile_starts(10, 16, 12) == [0]  # shorter than the tile: one tile
    assert tile_starts(16, 16, 12) == [0]
    assert tile_starts(40, 16, 12) == [0, 12, 24]
    assert tile_starts(48, 16, 16) == [0, 16, 32]
    for length in range(17, 70):
        for tile, stride in ((16, 8), (16, 12), (24, 24), (32, 16)):
            starts, size = tile_starts(length, tile, stride), min(tile, length)  # a short axis is one tile
            assert starts[0] == 0 and starts[-1] + size == length and starts == sorted(set(starts))
            covered = np.zeros(length, bool)
            for start in starts:
                covered[start : start + size] = True
            assert covered.all()


def test_the_blending_weights_are_fixed_linear_ramps_that_are_never_zero():
    np.testing.assert_allclose(ramp_weights(0, 8, 24, 0), np.ones(8))
    np.testing.assert_allclose(ramp_weights(0, 8, 8, 4), np.ones(8))  # a lone tile has no neighbour
    np.testing.assert_allclose(ramp_weights(8, 8, 24, 3), [0.25, 0.5, 0.75, 1.0, 1.0, 0.75, 0.5, 0.25])
    first = ramp_weights(0, 8, 24, 3)  # the image edge is not ramped
    np.testing.assert_allclose(first, [1, 1, 1, 1, 1, 0.75, 0.5, 0.25])
    assert (ramp_weights(8, 8, 24, 7) > 0).all()  # ramps that meet in the middle keep every weight positive


@pytest.mark.parametrize(
    "size,tile,overlap",
    [((64, 96), 32, None), ((64, 96), 32, 0), ((45, 37), 16, 8), ((45, 37), 24, 16), ((20, 20), 64, None), ((33, 70), 8, 4),
     ((40, 40), 40, None)],
)
def test_a_pointwise_model_is_reproduced_exactly_by_every_tiling(size, tile, overlap):
    """Coverage, the crop and the weights (their sum is divided out): a model with no spatial reach cannot change."""
    p = fake_predictor(lambda t: 2.0 * t - 0.25)
    image = np.random.default_rng(10).random((*size, 3)).astype(np.float32)
    whole = p.predict(image)["denoised"][0]
    tiled = p.predict(image, tile=tile, tile_overlap=overlap)["denoised"][0]
    assert tiled.shape == whole.shape == (*size, 3)
    np.testing.assert_allclose(tiled, whole, atol=2e-6, rtol=0)


def test_an_image_smaller_than_the_tile_is_one_whole_pass(predictors):
    p = predictors["bfunet"]
    image = np.random.default_rng(11).random((20, 28, 3)).astype(np.float32)
    np.testing.assert_array_equal(p.predict(image, tile=64)["denoised"][0], p.predict(image)["denoised"][0])
    # One side shorter than the tile: that axis is a single tile, the other is tiled; the answer keeps the size.
    narrow = np.random.default_rng(12).random((20, 90, 3)).astype(np.float32)
    assert p.predict(narrow, tile=32, tile_overlap=8)["denoised"][0].shape == (20, 90, 3)


def largest_step(image: np.ndarray, axis: int) -> float:
    """The largest mean step between neighbouring columns (or rows): a seam is a jump, a slope is not."""
    return float(np.abs(np.diff(image, axis=axis)).mean(axis=tuple(a for a in (0, 1, 2) if a != axis)).max())


def test_the_overlap_blends_away_the_seam_a_non_local_model_has_at_a_tile_edge():
    """An image minus its OWN mean: every tile subtracts another mean, so tiles disagree where they meet."""
    p = fake_predictor(lambda t: t - keras.ops.mean(t, axis=(1, 2), keepdims=True))
    ramp = np.linspace(0.0, 1.0, 192, dtype=np.float32)
    image = np.tile(ramp[None, :, None], (32, 1, 3))
    step = largest_step(p.predict(image)["denoised"][0], axis=1)  # the whole image: a plain slope, the step of the ramp
    assert step == pytest.approx(1.0 / 191.0, rel=1e-3)
    no_overlap = largest_step(p.predict(image, tile=64, tile_overlap=0)["denoised"][0], axis=1) / step
    blended = largest_step(p.predict(image, tile=64, tile_overlap=32)["denoised"][0], axis=1) / step
    assert no_overlap > 20.0  # measured 63: each seam is a jump of about 63 steps of the ramp
    assert blended < 2.0  # the ramps turn the jump into a slope
    assert blended < no_overlap / 10.0


#: The tiling bounds, measured on ``measured_image()`` (112 x 128) with the seeded tiny models of the ``models`` fixture
#: (m_tiling2.py of step 12): the mean of ``|tiled - whole|`` over the mean of ``|whole|`` at tile 64 with overlap 32,
#: and at overlap 0; each bound is twice the measurement (the random-init models are far from the trained ones, which is why
#: the claim here is a bound and the order of the two overlaps, and not an equality).
TILING = {  # family: (measured at overlap 32, bound at overlap 32, measured at overlap 0)
    "bfcnn": (0.005, 0.012, 0.029),
    "bfunet": (0.286, 0.58, 0.41),
    "bfconvunext": (0.097, 0.2, 0.372),
    "convunext_biased": (0.031, 0.065, 0.082),
    "scunet": (0.05, 0.1, 0.093),
}


def mean_relative(tiled: np.ndarray, whole: np.ndarray) -> float:
    return float(np.abs(tiled - whole).mean() / np.abs(whole).mean())


@pytest.mark.parametrize("family", FAMILIES)
def test_the_tiled_answer_is_within_a_measured_bound_of_the_whole_image_and_the_overlap_lowers_it(family, predictors):
    p = predictors[family]
    image = measured_image()
    whole = p.predict(image)["denoised"][0]
    blended = mean_relative(p.predict(image, tile=64, tile_overlap=32)["denoised"][0], whole)
    separate = mean_relative(p.predict(image, tile=64, tile_overlap=0)["denoised"][0], whole)
    _, bound, _ = TILING[family]
    assert blended <= bound, f"{family}: {blended:.3g} against a bound of {bound}"
    assert blended < separate, f"{family}: overlap 32 gave {blended:.3g}, no overlap {separate:.3g}"


#: Seam bands: the pixels within 2 of a tile edge of the NON-overlapping tiling (rows and columns 64), where a seam is.
def band_error(tiled: np.ndarray, whole: np.ndarray) -> float:
    mask = np.zeros(whole.shape[:2], bool)
    mask[62:66, :] = True
    mask[:, 62:66] = True
    return float(np.abs(tiled - whole)[mask].mean() / np.abs(whole).mean())


@pytest.mark.parametrize("family", FAMILIES)
def test_the_seam_metric_of_the_blended_tiling_is_lower_than_the_non_overlapping_one(family, predictors):
    p = predictors[family]
    image = measured_image()
    whole = p.predict(image)["denoised"][0]
    blended = band_error(p.predict(image, tile=64, tile_overlap=32)["denoised"][0], whole)
    separate = band_error(p.predict(image, tile=64, tile_overlap=0)["denoised"][0], whole)
    assert blended < separate, f"{family}: bands {blended:.3g} blended, {separate:.3g} without overlap"


@pytest.mark.parametrize("family", BIAS_FREE)
def test_tiling_keeps_homogeneity_for_a_bias_free_family(family, predictors):
    """The weights of the blend are fixed numbers, so f(a x) = a f(x) survives the tiling (D-019's probe, CPU)."""
    p = predictors[family]
    image = measured_image(64, 80)
    for tile, overlap in ((None, None), (32, 16), (32, 0)):
        base = p.predict(image, tile=tile, tile_overlap=overlap)["denoised"][0].astype(np.float64)
        worst = 0.0
        for alpha in ALPHAS:
            scaled = p.predict(np.float32(alpha) * image, tile=tile, tile_overlap=overlap)["denoised"][0].astype(np.float64)
            worst = max(worst, float(np.abs(scaled - alpha * base).max() / np.abs(alpha * base).max()))
        assert worst <= HOMOGENEITY, f"{family} tile={tile} overlap={overlap}: {worst:.3g}"


def test_a_biased_family_is_not_homogeneous_so_the_probe_above_could_fail(predictors):
    p = predictors["convunext_biased"]
    image = measured_image(64, 80)
    base = p.predict(image, tile=32, tile_overlap=16)["denoised"][0].astype(np.float64)
    scaled = p.predict(np.float32(3.7) * image, tile=32, tile_overlap=16)["denoised"][0].astype(np.float64)
    assert np.abs(scaled - 3.7 * base).max() / np.abs(3.7 * base).max() > 1e-2  # measured 0.4 at init


def test_a_tile_that_the_model_cannot_take_is_refused_by_name(predictors):
    image = np.zeros((32, 32, 3), np.float32)
    unet = predictors["bfunet"]
    cases = {
        "not a multiple": (dict(tile=30), r"'tile': 30 is not a multiple of 8, the size_multiple of this model \(bfunet\)"),
        "overlap not a multiple": (dict(tile=32, tile_overlap=4), r"'tile_overlap': 4 is not a multiple of 8"),
        "overlap as big as the tile": (dict(tile=32, tile_overlap=32), r"'tile_overlap': 32 must be below tile \(32\)"),
        "overlap without a tile": (dict(tile_overlap=8), r"'tile_overlap': it needs 'tile'"),
        "zero tile": (dict(tile=0), r"'tile': expected a positive int or None, got 0"),
        "negative overlap": (dict(tile=16, tile_overlap=-8), r"'tile_overlap': expected a whole number of 0 or more, got -8"),
        "a float tile": (dict(tile=16.0), r"'tile': expected a positive int or None, got 16.0"),
        "a bool tile": (dict(tile=True), r"'tile': expected a positive int or None, got True"),
    }
    for name, (arguments, message) in cases.items():
        with pytest.raises(ConfigError, match=message):
            unet.predict(image, **arguments)
    # A family with no stride takes any tile, and the default overlap of a small tile is none.
    assert predictors["bfcnn"].predict(image, tile=7, tile_overlap=3)["denoised"][0].shape == (32, 32, 3)
    assert predictors["bfcnn"].predict(image, tile=7)["denoised"][0].shape == (32, 32, 3)


class OutOfMemory:
    """A model that runs out of memory on any input larger than ``limit`` pixels per side."""

    def __init__(self, limit: int, error: Exception = None) -> None:
        self.limit, self.calls = limit, 0
        self.error = error or tf.errors.ResourceExhaustedError(None, None, "OOM when allocating tensor")

    def __call__(self, batch, training=False):
        self.calls += 1
        if max(batch.shape[1:3]) > self.limit:
            raise self.error
        return tf.constant(batch)


def test_a_gpu_out_of_memory_is_one_error_type_and_names_the_remedy():
    model = OutOfMemory(limit=16)
    p = DenoisingPredictor(model, record_of("bfcnn"))
    image = np.zeros((40, 40, 3), np.float32)
    with pytest.raises(ConfigError) as caught:
        p.predict(image)
    message = str(caught.value)
    assert "Out of memory" in message and "40x40" in message and "tile=" in message and "multiple of 1" in message
    assert isinstance(caught.value.__cause__, tf.errors.ResourceExhaustedError)
    # The remedy works: tiles of 16 fit.
    assert p.predict(image, tile=16)["denoised"][0].shape == (40, 40, 3)
    # A tile that is still too big names the tile it ran.
    with pytest.raises(ConfigError, match=r"on a 32x32 tile of a 40x40 image"):
        p.predict(image, tile=32)


def test_only_a_resource_exhausted_error_is_turned_into_the_memory_message():
    other = tf.errors.InvalidArgumentError(None, None, "a different fault")
    p = DenoisingPredictor(OutOfMemory(limit=1, error=other), record_of("bfcnn"))
    with pytest.raises(tf.errors.InvalidArgumentError, match="a different fault"):
        p.predict(np.zeros((8, 8, 3), np.float32))


# ---------------------------------------------------------------------
# predict on the command line
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def loud_model(world):
    """A bfcnn whose last convolution is scaled up so its output leaves ``[0, 1]`` on both sides."""
    keras.utils.set_random_seed(0)
    model = build("bfcnn")
    scaled = [w for w in model.weights if w.path.startswith("final_conv/") and w.path.endswith("kernel")]
    assert len(scaled) == 1  # the one kernel of the bias-free output convolution
    scaled[0].assign(scaled[0] * 40.0)
    return model


def make_run(root: Path, model: keras.Model, family: str = "bfcnn", **changes: Any) -> Path:
    root.mkdir(parents=True)
    model.save(root / "best_model.keras")
    (root / "task.json").write_text(json.dumps({**record_of(family), **changes}))
    return root


def read_png(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image)


def test_predict_writes_one_png_per_input_clipped_at_the_byte_boundary_and_the_index_last(loud_model, tmp_path, capsys):
    run = make_run(tmp_path / "run", loud_model)
    data = tmp_path / "images"
    data.mkdir()
    pixels = {name: write_png(data / name, h, w, seed=i) for i, (name, h, w) in enumerate([("a.png", 20, 30), ("b.png", 17, 9), ("c.png", 32, 32)])}
    out = tmp_path / "out"
    assert cli._predict([f"model={run}", f"data={data}", f"out={out}"]) == 0
    assert capsys.readouterr().out == f"{out}\n"
    assert sorted(p.name for p in out.iterdir()) == ["a.png", "b.png", "c.png", "index.json"]
    reloaded = load_model(run)
    raw_outside = False
    index = json.loads((out / "index.json").read_text())
    assert [entry["image"] for entry in index["images"]] == ["a.png", "b.png", "c.png"]
    for entry, (name, array) in zip(index["images"], pixels.items()):
        raw = reloaded.predict(str(data / name))["denoised"][0]
        raw_outside |= bool(raw.min() < 0.0 or raw.max() > 1.0)
        expected = np.rint(np.clip(raw.astype(np.float64), 0.0, 1.0) * 255.0).astype(np.uint8)
        got = read_png(out / name)
        assert got.dtype == np.uint8 and got.shape == expected.shape == array.shape
        np.testing.assert_array_equal(got, expected)  # equal to the clipped raw values at uint8
        assert entry["size"] == [array.shape[0], array.shape[1]] and entry["path"] == str(data / name)
    assert raw_outside  # the control: the raw output really left [0, 1], so the clip above did something
    assert "clipped to [0, 1]" in index["png"]


def test_a_grey_denoiser_writes_grey_pngs(world, tmp_path):
    keras.utils.set_random_seed(0)
    run = make_run(tmp_path / "run", build("bfcnn", input_shape=(None, None, 1)), channels=1)
    image = tmp_path / "g.png"
    write_png(image, 12, 14)
    out = tmp_path / "out"
    assert cli._predict([f"model={run}", f"data={image}", f"out={out}"]) == 0
    with Image.open(out / "g.png") as written:
        assert written.mode == "L" and written.size == (14, 12)


def test_to_uint8_clips_then_rounds_and_is_the_only_clip():
    raw = np.array([[[-0.4, 0.0, 0.5], [1.0, 1.7, 0.2]]], np.float32)
    np.testing.assert_array_equal(to_uint8(raw), np.array([[[0, 0, 128], [255, 255, 51]]], np.uint8))
    assert to_uint8(np.zeros((3, 4, 1), np.float32)).shape == (3, 4)


def test_out_and_the_stems_are_checked_before_the_model_is_loaded(loud_model, tmp_path, monkeypatch):
    run = make_run(tmp_path / "run", loud_model)

    def never(*_args, **_kwargs):
        raise AssertionError("the model was loaded")

    monkeypatch.setattr(predictor, "load_model", never)
    data = tmp_path / "images"
    (data / "x").mkdir(parents=True)
    write_png(data / "cat.png", 8, 8)
    write_png(data / "x" / "cat.jpg", 8, 8)
    out = tmp_path / "out"
    with pytest.raises(ConfigError, match=r"have the same name without its extension"):
        cli._predict([f"model={run}", f"data={data}", f"out={out}"])
    assert not out.exists()
    (data / "x" / "cat.jpg").unlink()
    out.mkdir()
    with pytest.raises(ConfigError, match=r"already exists; predict never overwrites"):
        cli._predict([f"model={run}", f"data={data}", f"out={out}"])
    assert list(out.iterdir()) == []


def test_a_run_that_fails_half_way_leaves_its_pngs_and_no_index(loud_model, tmp_path, monkeypatch):
    from train.api import denoising_predictor as dp

    run = make_run(tmp_path / "run", loud_model)
    data = tmp_path / "images"
    data.mkdir()
    for index in range(3):
        write_png(data / f"i{index}.png", 8, 8, seed=index)
    monkeypatch.setattr(dp, "CHUNK", 1)
    real = DenoisingPredictor.predict
    calls = []

    def flaky(self, x, **kwargs):
        calls.append(x)
        if len(calls) == 3:
            raise ConfigError("the third image failed")
        return real(self, x, **kwargs)

    monkeypatch.setattr(DenoisingPredictor, "predict", flaky)
    out = tmp_path / "out"
    with pytest.raises(ConfigError, match="the third image failed"):
        cli._predict([f"model={run}", f"data={data}", f"out={out}"])
    assert sorted(p.name for p in out.iterdir()) == ["i0.png", "i1.png"]  # the index is the mark of a complete answer


# ---------------------------------------------------------------------
# Export: the table
# ---------------------------------------------------------------------

#: The artifacts' tolerance, relative to ``max(1, max |keras|)``; measured at most 2.3e-6 in every cell (step 2(b)).
EXPORT_RTOL = 2e-5
SIZE_A, SIZE_B = (64, 64), (96, 80)  # both multiples of 16 (the base U-Net stride)
NAMES = {"keras": "m.keras", "saved_model": "m_saved_model", "tflite": "m.tflite", "onnx": "m.onnx"}
MODES = ("fixed", "dynamic")


class Runtime:
    """One artifact opened ONCE in its own runtime, called many times (a fresh load would restart any random state)."""

    def __init__(self, format: str, path: Path) -> None:
        self.format = format
        if format == "keras":
            self.model = keras.models.load_model(path, compile=False)
        elif format == "saved_model":
            self.loaded = tf.saved_model.load(str(path))
        elif format == "tflite":
            self.blob = Path(path).read_bytes()
        else:
            import onnxruntime

            self.session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        if self.format == "keras":
            with tf.device("/CPU:0"):
                return np.asarray(self.model(batch, training=False), np.float64)
        if self.format == "saved_model":
            with tf.device("/CPU:0"):
                return np.asarray(self.loaded.serve(tf.constant(batch)), np.float64)
        if self.format == "tflite":
            return exporting._tflite_logits(self.blob, batch)
        return np.asarray(self.session.run(None, {self.session.get_inputs()[0].name: batch})[0], np.float64)


def images_of(size: Tuple[int, int], seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random((2, *size, 3)).astype(np.float32)


def keras_reference(model: keras.Model, batch: np.ndarray) -> np.ndarray:
    with tf.device("/CPU:0"):
        return np.asarray(model(batch, training=False), np.float64)


def relative_gap(got: np.ndarray, expected: np.ndarray) -> float:
    assert got.shape == expected.shape, (got.shape, expected.shape)
    return float(np.abs(got - expected).max() / max(1.0, float(np.abs(expected).max())))


class Workshop:
    """Exports each (family, format, mode) once, through ``export`` with the record (so the task's rule runs)."""

    def __init__(self, root: Path, models: Dict[str, keras.Model]) -> None:
        self.root, self.models = root, models
        self.written: Dict[Tuple[str, str, str], Path] = {}
        self.runtimes: Dict[Tuple[str, str, str], Runtime] = {}

    def artifact(self, family: str, format: str, mode: str) -> Path:
        key = (family, format, mode)
        if key not in self.written:
            if format == "onnx":
                pytest.importorskip("tf2onnx")
                pytest.importorskip("onnxruntime")
            shape = (None, None, 3) if mode == "dynamic" else (*SIZE_A, 3)
            out = self.root / family / mode / NAMES[format]
            self.written[key] = export(
                model=self.models[family], out=out, format=format, input_shape=shape, record=record_of(family)
            )
        return self.written[key]

    def runtime(self, family: str, format: str, mode: str) -> Runtime:
        key = (family, format, mode)
        if key not in self.runtimes:
            self.runtimes[key] = Runtime(format, self.artifact(*key))
        return self.runtimes[key]


@pytest.fixture(scope="module")
def workshop(models, tmp_path_factory):
    return Workshop(tmp_path_factory.mktemp("denoiser_exports"), models)


CELLS = [(family, format, mode) for family in FAMILIES for format in FORMATS for mode in MODES]


def test_the_table_has_no_blank_cell_and_every_refusal_is_one_of_its_cells():
    assert len(CELLS) == 40
    for family, format, mode in EXPORT_REFUSALS:
        assert (family, format, mode) in CELLS
    assert sorted(EXPORT_REFUSALS) == sorted(
        [("bfunet", "tflite", "fixed"), ("bfconvunext", "tflite", "fixed"), ("convunext_biased", "tflite", "fixed"),
         ("scunet", "onnx", "dynamic")]
    )


@pytest.mark.parametrize("family,format,mode", CELLS, ids=["-".join(cell) for cell in CELLS])
def test_each_cell_is_a_numeric_match_with_keras_or_one_named_refusal(family, format, mode, workshop, models):
    if (family, format, mode) in EXPORT_REFUSALS:
        reason = EXPORT_REFUSALS[(family, format, mode)]
        before = sorted(workshop.root.rglob("*")) if workshop.root.exists() else []
        with pytest.raises(ConfigError) as caught:
            workshop.artifact(family, format, mode)
        assert reason in str(caught.value) and format in str(caught.value) and family in str(caught.value)
        assert sorted(workshop.root.rglob("*")) == before  # nothing was written
        return
    runtime = workshop.runtime(family, format, mode)
    sizes = [SIZE_A] if mode == "fixed" else [SIZE_A, SIZE_B]
    for size in sizes:
        batch = images_of(size)
        expected = keras_reference(models[family], batch)
        assert np.abs(expected).max() > 1e-3  # a model that outputs nothing would match anything
        first, second = runtime(batch), runtime(batch)
        np.testing.assert_array_equal(first, second)  # repeatable on one runtime object
        assert relative_gap(first, expected) <= EXPORT_RTOL, f"{family} {format} {mode} {size}"


@pytest.mark.parametrize("family", FAMILIES)
def test_an_exported_file_carries_its_record_and_the_denoising_note(family, workshop, caplog):
    path = workshop.artifact(family, "keras", "dynamic")
    beside = path.with_name(path.name + ".task.json")
    assert json.loads(beside.read_text()) == record_of(family)
    # The saved Keras archive keeps the free input, so the dynamic export is the model itself.
    assert keras.models.load_model(path, compile=False).input_shape == (None, None, None, 3)


def test_the_export_note_says_raw_values_no_clip_no_sigma_and_the_padding():
    note = predictor.EXPORT_RULES[DENOISING].note(record_of("bfunet"))
    for phrase in ("[0, 1]", "no sigma input", "RAW", "not clipped", "multiple of 8", "no mean, no std"):
        assert phrase in note, phrase
    help_text = "\n".join(cli.EXPORT_NOTES)
    for phrase in ("RAW", "NOT clipped", "no sigma input", "size_multiple", "silently wrong", "multiples of 64"):
        assert phrase in help_text, phrase
    assert "RAW" in exporting.__doc__ and "no sigma" in exporting.__doc__


# ---------------------------------------------------------------------
# Export: each refusal is re-proved
# ---------------------------------------------------------------------


@pytest.mark.parametrize("family", ["bfunet", "bfconvunext", "convunext_biased"])
def test_a_fixed_shape_tflite_of_a_unet_is_silently_wrong_at_another_size(family, models, tmp_path):
    """The reason of three refusals: build the file the refusal prevents and read it at a second size."""
    out = tmp_path / "fixed.tflite"
    spec = tf.TensorSpec((None, *SIZE_A, 3), tf.float32, name="input")
    exporting._export_tflite(models[family], out, spec)
    batch = images_of(SIZE_B)
    own = images_of(SIZE_A)
    assert relative_gap(exporting._tflite_logits(out.read_bytes(), own), keras_reference(models[family], own)) <= EXPORT_RTOL
    wrong = exporting._tflite_logits(out.read_bytes(), batch)  # no error: the interpreter resizes its input
    assert relative_gap(wrong, keras_reference(models[family], batch)) > 0.1  # measured 0.52 to 1.0


def test_a_fixed_shape_tflite_of_a_family_with_no_stride_stays_right_at_another_size(models, tmp_path):
    """bfcnn is fully convolutional: its fixed tflite is NOT refused, and this shows why."""
    out = tmp_path / "fixed.tflite"
    exporting._export_tflite(models["bfcnn"], out, tf.TensorSpec((None, *SIZE_A, 3), tf.float32, name="input"))
    batch = images_of(SIZE_B)
    assert relative_gap(exporting._tflite_logits(out.read_bytes(), batch), keras_reference(models["bfcnn"], batch)) <= EXPORT_RTOL


def test_the_scunet_onnx_with_a_free_height_and_width_fails_off_multiples_of_64(models, tmp_path):
    """The reason of the fourth refusal: it matches at 64 x 64 and 128 x 64 and fails at 96 x 80, in the runtime."""
    pytest.importorskip("tf2onnx")
    onnxruntime = pytest.importorskip("onnxruntime")
    out = tmp_path / "dynamic.onnx"
    exporting._export_onnx(models["scunet"], out, tf.TensorSpec((None, None, None, 3), tf.float32, name="input"))
    session = onnxruntime.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    for size in [(64, 64), (128, 64)]:
        batch = images_of(size)
        got = np.asarray(session.run(None, {name: batch})[0], np.float64)
        assert relative_gap(got, keras_reference(models["scunet"], batch)) <= EXPORT_RTOL, size
    for size in [(96, 80), (100, 100), (65, 65)]:
        with pytest.raises(Exception, match=r"RUNTIME_EXCEPTION|INVALID_ARGUMENT|Tile"):
            session.run(None, {name: images_of(size)})


def test_a_fixed_onnx_of_scunet_is_not_refused_and_matches_at_sizes_off_64(models, tmp_path):
    """Only the FREE-size onnx of SCUNet is refused: with a fixed height and width the same sizes convert and match."""
    pytest.importorskip("tf2onnx")
    pytest.importorskip("onnxruntime")
    for size in [(96, 80), (65, 65)]:
        path = export(model=models["scunet"], out=tmp_path / f"s{size[0]}.onnx", format="onnx", input_shape=(*size, 3),
                      record=record_of("scunet"))
        batch = images_of(size)
        assert relative_gap(Runtime("onnx", path)(batch), keras_reference(models["scunet"], batch)) <= EXPORT_RTOL, size


def test_a_fixed_size_off_the_stride_is_refused_and_a_unet_cannot_be_traced_there(models, tmp_path):
    with pytest.raises(ConfigError, match=r"the height 30 is not a multiple of 8, the size_multiple of bfunet/tiny"):
        export(model=models["bfunet"], out=tmp_path / "a_saved_model", format="saved_model", input_shape=(30, 32, 3),
               record=record_of("bfunet"))
    with pytest.raises(ConfigError, match=r"the width 36 is not a multiple of 8"):
        export(model=models["bfunet"], out=tmp_path / "b_saved_model", format="saved_model", input_shape=(32, 36, 3),
               record=record_of("bfunet"))
    assert not (tmp_path / "a_saved_model").exists() and not (tmp_path / "a_saved_model.task.json").exists()
    with pytest.raises((tf.errors.InvalidArgumentError, ValueError)):  # the reason: the model itself fails there
        models["bfunet"](np.zeros((1, 30, 32, 3), np.float32), training=False)
    # A family with no stride takes any fixed size, and the record's channel count is the only one.
    export(model=models["bfcnn"], out=tmp_path / "c_saved_model", format="saved_model", input_shape=(30, 36, 3),
           record=record_of("bfcnn"))
    with pytest.raises(ConfigError, match=r"has 1 channels but this denoiser takes 3"):
        export(model=models["bfcnn"], out=tmp_path / "d_saved_model", format="saved_model", input_shape=(32, 32, 1),
               record=record_of("bfcnn"))


def test_a_free_side_is_accepted_only_where_the_model_itself_takes_any_size(tmp_path):
    inputs = keras.Input((6, 6, 3))
    fixed = keras.Model(inputs, keras.layers.Dense(2)(keras.layers.Flatten()(inputs)))
    for shape in [(None, None, 3), (None, 6, 3), (6, None, 3), (6, 6, None)]:
        with pytest.raises(ConfigError, match="input_shape"):
            export(model=fixed, out=tmp_path / "f_saved_model", format="saved_model", input_shape=shape)
    assert list(tmp_path.iterdir()) == []
    free = lambda_model(lambda t: t)
    with pytest.raises(ConfigError, match="input_shape"):
        export(model=free, out=tmp_path / "g_saved_model", format="saved_model", input_shape=(None, None, None))
    with pytest.raises(ConfigError, match="input_shape"):
        export(model=free, out=tmp_path / "g_saved_model", format="saved_model", input_shape=(0, None, 3))
    assert export(model=free, out=tmp_path / "h_saved_model", format="saved_model", input_shape=(None, None, 3)).is_dir()
    assert export(model=free, out=tmp_path / "i_saved_model", format="saved_model").is_dir()  # the model's own input


class NoiseUnlessFalse(keras.layers.Layer):
    """Adds noise to its input unless it is called with ``training=False`` EXACTLY (the shape of a drop-path layer
    that only skips its randomness on ``training is False``; Keras' own Dropout treats ``None`` as ``False``)."""

    def call(self, inputs, training=None):
        if training is False:
            return inputs
        return inputs + keras.random.uniform(keras.ops.shape(inputs), seed=keras.random.SeedGenerator(1))


@pytest.mark.parametrize("format", ["saved_model", "onnx"])
def test_a_denoiser_is_exported_traced_with_training_false_so_nothing_random_is_left(format, tmp_path):
    """``model.export`` traces ``training=None``: a layer random on anything but False would serve noise forever."""
    if format == "onnx":
        pytest.importorskip("tf2onnx")
        pytest.importorskip("onnxruntime")
    inputs = keras.Input((None, None, 3))
    model = keras.Model(inputs, NoiseUnlessFalse()(inputs))
    batch = images_of((16, 16))
    assert np.abs(np.asarray(model(batch, training=True)) - batch).max() > 0.1  # the control: it IS random when not False
    path = export(model=model, out=tmp_path / NAMES[format], format=format, input_shape=(None, None, 3),
                  record=record_of("bfcnn"))
    runtime = Runtime(format, path)
    first, second = runtime(batch), runtime(batch)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first, batch, atol=1e-6, rtol=0)


# ---------------------------------------------------------------------
# Export: homogeneity read again from the files
# ---------------------------------------------------------------------


def homogeneity_of(runtime: Runtime, size: Tuple[int, int] = (64, 64)) -> float:
    image = np.random.default_rng(13).random((1, *size, 3)).astype(np.float32)
    base = runtime(image)
    worst = 0.0
    for alpha in ALPHAS:
        scaled = runtime(np.float32(alpha) * image)
        worst = max(worst, float(np.abs(scaled - alpha * base).max() / np.abs(alpha * base).max()))
    return worst


@pytest.mark.parametrize("format", ["keras", "saved_model"])
@pytest.mark.parametrize("family", BIAS_FREE)
def test_the_exports_of_a_bias_free_family_are_still_homogeneous(family, format, workshop):
    worst = homogeneity_of(workshop.runtime(family, format, "dynamic"))
    assert worst <= HOMOGENEITY, f"{family} {format}: {worst:.3g}"


@pytest.mark.parametrize("format", ["keras", "saved_model"])
def test_the_control_a_biased_family_reads_as_not_homogeneous_from_its_export(format, workshop):
    assert homogeneity_of(workshop.runtime("convunext_biased", format, "dynamic")) > 1e-2  # measured 0.45 at init


# ---------------------------------------------------------------------
# Export on the command line
# ---------------------------------------------------------------------


def test_the_command_line_exports_a_denoiser_with_a_free_size_and_writes_its_record(models, tmp_path, capsys):
    run = make_run(tmp_path / "run", models["bfcnn"])
    out = tmp_path / "exported_saved_model"
    assert cli.main(["export", f"model={run}", f"out={out}", "format=saved_model"]) == 0
    assert capsys.readouterr().out == f"{out}\n"
    assert json.loads(Path(f"{out}.task.json").read_text()) == record_of("bfcnn")
    served = tf.saved_model.load(str(out))
    for size in [(24, 40), (31, 17)]:  # a free height and width, no stride in bfcnn
        batch = images_of(size)
        np.testing.assert_allclose(served.serve(tf.constant(batch)).numpy(), keras_reference(models["bfcnn"], batch), atol=1e-5)


def test_the_command_line_refuses_a_refused_cell_with_its_reason_and_writes_nothing(models, tmp_path, capsys):
    """The command line exports with a free height and width, so the cell it can hit is SCUNet's onnx."""
    pytest.importorskip("tf2onnx")
    run = make_run(tmp_path / "run", models["scunet"], family="scunet")
    out = tmp_path / "scunet.onnx"
    assert cli.main(["export", f"model={run}", f"out={out}", "format=onnx"]) == 2
    error = capsys.readouterr().err
    assert "multiples of 64" in error and "scunet" in error and "fixed height and width" in error
    assert not out.exists() and not Path(f"{out}.task.json").exists()


def test_the_command_line_exports_a_unet_as_a_dynamic_tflite_that_matches_at_two_sizes(models, tmp_path, capsys):
    run = make_run(tmp_path / "run", models["bfunet"], family="bfunet")
    out = tmp_path / "unet.tflite"
    assert cli.main(["export", f"model={run}", f"out={out}", "format=tflite"]) == 0
    assert capsys.readouterr().out == f"{out}\n"
    for size in [SIZE_A, SIZE_B]:
        batch = images_of(size)
        assert relative_gap(exporting._tflite_logits(out.read_bytes(), batch), keras_reference(models["bfunet"], batch)) <= EXPORT_RTOL


def test_the_export_help_names_the_denoising_contract(capsys):
    assert cli.main(["export", "help"]) == 0
    text = capsys.readouterr().out
    assert "RAW" in text and "sigma" in text and "size_multiple" in text


def test_the_classification_and_segmentation_note_is_the_one_it_was():
    note = "the exported model takes inputs standardized with the run's mean and std (pixels in [0, 1], then (x - mean) / std) and returns raw logits"
    for task in (predictor.CLASSIFICATION, predictor.SEGMENTATION):
        assert predictor.EXPORT_RULES[task].note({}) == note
        predictor.EXPORT_RULES[task].check({}, "tflite", (8, 8, 3))  # no cell of theirs is refused


# ---------------------------------------------------------------------
# What the table's rule is made of
# ---------------------------------------------------------------------


def test_the_modes_are_read_from_the_shape():
    from train.api.denoising_predictor import export_mode

    assert export_mode((None, None, 3)) == "dynamic" and export_mode((None, 8, 3)) == "dynamic"
    assert export_mode((8, None, 3)) == "dynamic" and export_mode((8, 8, 3)) == "fixed"


def test_the_refusal_texts_hold_the_measured_numbers_the_tests_above_re_prove():
    text = " ".join(EXPORT_REFUSALS.values())
    for number in ("1.0 (relative)", "0.52 (relative)", "0.56 (relative)", "multiples of 64", "96x80", "100x100", "65x65"):
        assert number in text, number
