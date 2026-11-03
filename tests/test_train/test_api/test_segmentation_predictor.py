"""``SegmentationPredictor`` and ``python -m train predict`` for a segmentation run (plan 2faf890a step 30, D-080).

The model of most tests here is a FIXTURE whose answer is known by construction, so a mask can be
compared with the one that was drawn rather than with the code under test: a 1x1 convolution whose
logit for class ``c`` is ``-||pixel - colour_c||^2`` (up to a term all classes share), so the argmax of its
output is the class whose palette colour is nearest, and ``fx.picture(mask)`` paints a mask with exactly
those colours. The claims:

* a path, a PIL image and an array of one picture give one answer, for RGB, RGBA (alpha dropped), grey
  and floating inputs; a floating array is rounded to its byte (all 256 levels as float32 give the bytes);
* the mask has the size of the image AFTER its EXIF orientation (an orientation-6 JPEG, and the same file
  without the tag as the control), and a known mask is predicted back exactly and survives the PNG;
* the logits are resized BEFORE the argmax (a hand fixture whose bilinear answer differs from a nearest
  resize of the argmax), in pieces of classes with the same answer as in one piece, ties going to the
  earlier class;
* more than 255 classes give ``uint16``;
* ``training=False`` is passed (a layer that is random unless it is told not to be gives one answer twice);
* ``load_model`` dispatches on the record's task and refuses what is not one run (an input size the
  model does not take, a class count it does not output, a family of the other task);
* the command writes one PNG per image and ``index.json`` last, refuses two images with one stem and a
  ``logits`` record before it loads the model, never overwrites, and a run that fails half way has no index.

Everything writes under ``tmp_path``.
"""

import json
from pathlib import Path

import keras
import numpy as np
import pytest
import tensorflow as tf
from PIL import Image, ImageOps

import train.api.predictor as predictor_module
import train.api.recipes as recipes
from train.api import cli, registry
from train.api.errors import ConfigError
from train.api.predictor import (
    SEGMENTATION,
    build_segmentation_record,
    build_task_record,
    input_shape_of,
    load_model,
    record_beside,
)
from train.api.segmentation_predictor import (
    CHUNK,
    SegmentationPredictor,
    check_output,
    mask_dtype,
    mask_names,
    resize_argmax,
    write_predictions,
)
from train.common import segmentation_data as sd

from . import _segmentation_fixtures as fx

CLASSES = 4
NAMES = ["a", "b", "c", "d"]
ORIENTATION = 0x0112


@pytest.fixture(scope="module")
def world():
    """An empty registry holding the built-in families."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(registry, "MODELS", {})
        patch.setattr(registry, "TASKS", {})
        recipes.register_builtin_models()
        yield


def record(size=(16, 24), classes=NAMES, **changes):
    value = build_segmentation_record(
        model="convunext/tiny", family="convunext", variant="tiny", class_names=classes,
        input_size=size, channels=3, normalize="unit", mean=[0.0] * 3, std=[1.0] * 3,
    )
    value.update(changes)
    return value


def colour_model(size=(16, 24), classes=CLASSES, noisy=False):
    """A model whose argmax is the class of the nearest palette colour (``fx.PALETTE``), ``classes`` <= 4.

    ``noisy`` adds a layer that is random unless it is called with ``training=False``.
    """
    colours = fx.PALETTE[:classes].astype(np.float64) / 255.0
    inputs = keras.Input((*size, 3))
    x = inputs
    if noisy:
        x = NoisyUnlessFalse()(x)
    layer = keras.layers.Conv2D(classes, 1, name="scores")
    outputs = layer(x)
    model = keras.Model(inputs, outputs)
    # -||x - c||^2 = 2 x.c - |c|^2 - |x|^2; the last term is common to every class.
    layer.set_weights([(2.0 * colours.T)[None, None], -(colours ** 2).sum(1)])
    return model


@keras.saving.register_keras_serializable(package="test_segmentation_predictor")
class NoisyUnlessFalse(keras.layers.Layer):
    """Adds noise unless ``training is False`` (the stochastic-depth failure D-066: ``None`` is not ``False``)."""

    def call(self, x, training=None):
        if training is False:
            return x
        return x + keras.random.normal(tf.shape(x), stddev=5.0)


def predictor(size=(16, 24), **kwargs):
    return SegmentationPredictor(colour_model(size, **kwargs), record(size))


def pictures(count, size, seed=0):
    masks = [fx.blocks(*size, classes=CLASSES, seed=seed + i) for i in range(count)]
    return masks, [fx.picture(m) for m in masks]


# ---------------------------------------------------------------------
# A known mask comes back
# ---------------------------------------------------------------------

def test_a_known_mask_is_predicted_back_exactly_at_a_non_square_size():
    masks, images = pictures(3, (16, 24))
    answer = predictor().predict(images)
    assert answer["class_names"] == NAMES
    assert [m.dtype for m in answer["masks"]] == [np.uint8] * 3
    for mask, got in zip(masks, answer["masks"]):
        np.testing.assert_array_equal(got, mask)


def test_a_single_image_still_gives_a_list_of_one_mask():
    masks, images = pictures(1, (16, 24))
    answer = predictor().predict(images[0])
    assert isinstance(answer["masks"], list) and len(answer["masks"]) == 1
    np.testing.assert_array_equal(answer["masks"][0], masks[0])


def test_the_mask_has_the_size_of_the_image_not_of_the_model():
    _, [image] = pictures(1, (40, 56))
    got = predictor().predict(image)["masks"][0]
    assert got.shape == (40, 56)
    # blocks of 8 pixels survive a resize to 16x24 and back well away from their edges
    truth = fx.blocks(40, 56, classes=CLASSES, seed=0)
    assert (got == truth).mean() > 0.9


def test_a_batch_of_more_than_one_chunk_keeps_the_order():
    count = CHUNK + 3
    masks, images = pictures(count, (16, 24))
    got = predictor().predict(images)["masks"]
    assert len(got) == count
    for mask, one in zip(masks, got):
        np.testing.assert_array_equal(one, mask)


def test_the_answer_survives_the_png_it_is_written_as(tmp_path):
    masks, images = pictures(3, (16, 24))
    files = []
    for index, image in enumerate(images):
        path = tmp_path / "in" / f"p{index}.png"
        path.parent.mkdir(exist_ok=True)
        Image.fromarray(image).save(path)
        files.append(path)
    out = tmp_path / "out"
    write_predictions(predictor(), files, out)
    for index, mask in enumerate(masks):
        np.testing.assert_array_equal(sd.decode_mask(str(out / f"p{index}.png")), mask)
    index = json.loads((out / "index.json").read_text())
    assert index["class_names"] == NAMES
    assert [e["mask"] for e in index["masks"]] == ["p0.png", "p1.png", "p2.png"]
    assert [e["size"] for e in index["masks"]] == [[16, 24]] * 3
    assert [e["path"] for e in index["masks"]] == [str(f) for f in files]


# ---------------------------------------------------------------------
# Path, PIL and array agree
# ---------------------------------------------------------------------

def test_a_path_a_pil_image_and_an_array_agree(tmp_path):
    _, [image] = pictures(1, (30, 50))
    path = tmp_path / "x.png"
    Image.fromarray(image).save(path)
    p = predictor()
    by_path = p.predict(str(path))["masks"][0]
    np.testing.assert_array_equal(by_path, p.predict(Path(path))["masks"][0])
    np.testing.assert_array_equal(by_path, p.predict(Image.open(path))["masks"][0])
    np.testing.assert_array_equal(by_path, p.predict(image)["masks"][0])
    np.testing.assert_array_equal(by_path, p.predict(image[None])["masks"][0])


def test_an_rgba_image_loses_its_alpha_in_every_form(tmp_path):
    _, [image] = pictures(1, (30, 50))
    alpha = np.full(image.shape[:2] + (1,), 40, np.uint8)
    rgba = np.concatenate([image, alpha], axis=-1)
    path = tmp_path / "x.png"
    Image.fromarray(rgba, "RGBA").save(path)
    p = predictor()
    expected = p.predict(image)["masks"][0]
    np.testing.assert_array_equal(p.predict(str(path))["masks"][0], expected)
    np.testing.assert_array_equal(p.predict(Image.open(path))["masks"][0], expected)
    np.testing.assert_array_equal(p.predict(rgba)["masks"][0], expected)


def test_a_grey_image_is_repeated_over_the_channels(tmp_path):
    grey = np.random.default_rng(1).integers(0, 256, (30, 50), dtype=np.uint8)
    path = tmp_path / "g.png"
    Image.fromarray(grey).save(path)
    p = predictor()
    expected = p.predict(np.repeat(grey[..., None], 3, -1))["masks"][0]
    np.testing.assert_array_equal(p.predict(grey)["masks"][0], expected)
    np.testing.assert_array_equal(p.predict(grey[..., None])["masks"][0], expected)
    np.testing.assert_array_equal(p.predict(str(path))["masks"][0], expected)
    np.testing.assert_array_equal(p.predict(Image.open(path))["masks"][0], expected)


def test_a_floating_array_is_rounded_to_its_byte_for_all_256_levels():
    # k / 255 as float32, times 255 in float64, falls just below k for many k: a truncation would
    # feed the model k - 1 for those pixels.
    levels = np.arange(256, dtype=np.uint8)
    image = np.tile(levels, (3, 1))[..., None].repeat(3, -1)  # (3, 256, 3)
    floats = (image.astype(np.float32) / np.float32(255.0))
    p = predictor((8, 8))
    seen = []
    original = p._logits
    p._logits = lambda images: (seen.append(images[0].copy()), original(images))[1]
    p.predict(floats)
    np.testing.assert_array_equal(seen[0], image)


def test_a_floating_array_between_bytes_is_rounded_not_truncated():
    # (k + 0.6) / 255 is a value between two bytes: rounding gives k + 1, truncating gives k.
    levels = np.arange(255)
    floats = np.tile(((levels + 0.6) / 255.0)[:, None, None], (1, 2, 3))  # (255, 2, 3) float64
    p = predictor((8, 8))
    seen = []
    original = p._logits
    p._logits = lambda images: (seen.append(images[0].copy()), original(images))[1]
    p.predict(floats)
    np.testing.assert_array_equal(seen[0][:, 0, 0], levels + 1)


def test_a_model_that_outputs_another_map_than_the_record_names_is_refused_at_predict():
    with pytest.raises(ConfigError, match=r"should output \(16, 24, 4\)"):
        SegmentationPredictor(colour_model(classes=3), record()).predict(np.zeros((16, 24, 3), np.uint8))


def test_a_png_that_appears_while_the_answers_are_written_is_not_overwritten(tmp_path):
    source = tmp_path / "a.png"
    Image.fromarray(fx.picture(fx.blocks(16, 24, classes=3))).save(source)
    out = tmp_path / "answers"
    p = predictor()
    real = p.predict

    def racing(x):
        (out / "a.png").write_bytes(b"someone else's file")  # another process wrote it after the checks
        return real(x)

    p.predict = racing
    with pytest.raises(FileExistsError):
        write_predictions(p, [source], out)
    assert (out / "a.png").read_bytes() == b"someone else's file"


def test_an_existing_out_directory_is_a_readable_refusal_not_a_traceback(tmp_path):
    out = tmp_path / "answers"
    out.mkdir()
    with pytest.raises(ConfigError, match="already exists; predict never overwrites"):
        write_predictions(predictor(), [tmp_path / "a.png"], out)


@pytest.mark.parametrize("bad, message", [
    (np.zeros((8, 8, 3), np.int32), "must be uint8 or floating point"),
    (np.full((8, 8, 3), 1.5, np.float32), r"must lie in \[0, 1\]"),
    (np.full((8, 8, 3), np.nan, np.float32), r"must lie in \[0, 1\]"),
    (np.zeros((8, 8, 2), np.uint8), "must be .*H, W"),
    (np.zeros((0, 8, 3), np.uint8), "must be .*H, W"),
    (np.zeros((8,), np.uint8), "expected a path, a PIL image"),
    (3.5, "expected a path, a PIL image"),
])
def test_an_input_that_is_no_image_is_refused(bad, message):
    with pytest.raises(ConfigError, match=message):
        predictor().predict(bad)


def test_an_empty_list_and_a_missing_or_undecodable_file_are_refused(tmp_path):
    p = predictor()
    with pytest.raises(ConfigError, match="the list of images is empty"):
        p.predict([])
    with pytest.raises(ConfigError, match="is not a file"):
        p.predict(str(tmp_path / "nope.png"))
    junk = tmp_path / "junk.png"
    junk.write_bytes(b"not an image")
    with pytest.raises(ConfigError, match="cannot be decoded as an image"):
        p.predict(str(junk))


# ---------------------------------------------------------------------
# EXIF
# ---------------------------------------------------------------------

def exif_jpeg(path, stored_mask, orientation):
    exif = Image.Exif()
    if orientation is not None:
        exif[ORIENTATION] = orientation
    Image.fromarray(fx.picture(stored_mask)).save(path, quality=100, subsampling=0, exif=exif)


def test_the_mask_has_the_size_of_the_image_after_its_exif_orientation(tmp_path):
    stored = fx.blocks(24, 32, classes=CLASSES, seed=3)  # 24 rows, 32 columns as stored
    sideways, upright = tmp_path / "sideways.jpg", tmp_path / "upright.jpg"
    exif_jpeg(sideways, stored, 6)
    exif_jpeg(upright, stored, None)
    assert Image.open(sideways).size == (32, 24)  # PIL's (width, height) as stored
    assert ImageOps.exif_transpose(Image.open(sideways)).size == (24, 32)
    p = predictor((32, 24))  # the model takes the displayed size, so the resize is the identity
    got = p.predict(str(sideways))["masks"][0]
    assert got.shape == (32, 24)
    # orientation 6 displays the stored picture turned a quarter clockwise
    expected = np.rot90(stored, k=-1)
    assert (got == expected).mean() > 0.97
    # the control: the same pixels without the tag keep their stored shape
    other = predictor((24, 32)).predict(str(upright))["masks"][0]
    assert other.shape == (24, 32) and (other == stored).mean() > 0.97
    # and a PIL image opened from the file is turned too
    np.testing.assert_array_equal(got, p.predict(Image.open(sideways))["masks"][0])


# ---------------------------------------------------------------------
# Resize before the argmax
# ---------------------------------------------------------------------

def test_the_logits_are_resized_before_the_argmax_not_after():
    # Three columns at the model's resolution. Class 0 scores 2, -1, 2 and class 1 scores 1.5.
    low = np.zeros((1, 3, 2), np.float32)
    low[0, :, 0] = [2.0, -1.0, 2.0]
    low[0, :, 1] = 1.5
    # Bilinear (half-pixel) to six columns gives class 0 the scores 2, 1.25, -0.25, -0.25, 1.25, 2.
    got = resize_argmax(low, (1, 6))
    np.testing.assert_array_equal(got, [[0, 1, 1, 1, 1, 0]])
    # The argmax at the model's resolution, resized by nearest neighbour, would say [0, 0, 1, 1, 0, 0].
    assert got.dtype == np.int32


@keras.saving.register_keras_serializable(package="test_segmentation_predictor")
class FixedScores(keras.layers.Layer):
    """Ignores its input and returns one fixed ``(h, w, classes)`` score map for every image."""

    def __init__(self, scores, **kwargs):
        super().__init__(**kwargs)
        self.scores = np.asarray(scores, np.float32)

    def call(self, x):
        return tf.broadcast_to(tf.constant(self.scores), (tf.shape(x)[0], *self.scores.shape))


def test_the_predictor_resizes_the_scores_before_the_argmax_end_to_end():
    low = np.zeros((1, 3, 2), np.float32)
    low[0, :, 0] = [2.0, -1.0, 2.0]
    low[0, :, 1] = 1.5
    inputs = keras.Input((1, 3, 3))
    model = keras.Model(inputs, FixedScores(low)(inputs))
    p = SegmentationPredictor(model, record((1, 3), classes=["x", "y"]))
    got = p.predict(np.zeros((1, 6, 3), np.uint8))["masks"][0]
    np.testing.assert_array_equal(got, [[0, 1, 1, 1, 1, 0]])  # nearest-of-argmax would be [0, 0, 1, 1, 0, 0]


def test_pieces_of_classes_give_the_answer_of_one_piece_and_a_tie_goes_to_the_earlier_class():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(5, 7, 9)).astype(np.float32)
    whole = resize_argmax(logits, (11, 13))
    for cap in (1, 11 * 13, 11 * 13 * 2, 11 * 13 * 4):
        np.testing.assert_array_equal(resize_argmax(logits, (11, 13), max_elements=cap), whole)
    tied = np.zeros((4, 4, 6), np.float32)
    for cap in (1, 16, 10 ** 6):
        np.testing.assert_array_equal(resize_argmax(tied, (8, 8), max_elements=cap), np.zeros((8, 8)))


# ---------------------------------------------------------------------
# More than 255 classes
# ---------------------------------------------------------------------

def test_a_mask_is_sixteen_bit_above_255_classes(tmp_path):
    assert mask_dtype(255) == "uint8" and mask_dtype(256) == "uint16"
    classes = 300
    inputs = keras.Input((8, 8, 3))
    layer = keras.layers.Conv2D(classes, 1)
    model = keras.Model(inputs, layer(inputs))
    bias = np.zeros(classes, np.float32)
    bias[299] = 10.0
    layer.set_weights([np.zeros((1, 1, 3, classes), np.float32), bias])
    names = [f"c{i}" for i in range(classes)]
    p = SegmentationPredictor(model, record((8, 8), classes=names))
    mask = p.predict(np.zeros((8, 8, 3), np.uint8))["masks"][0]
    assert mask.dtype == np.uint16 and (mask == 299).all()
    source = tmp_path / "x.png"
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(source)
    write_predictions(p, [source], tmp_path / "out")
    assert (sd.decode_mask(str(tmp_path / "out" / "x.png")) == 299).all()


# ---------------------------------------------------------------------
# The model is called with training=False
# ---------------------------------------------------------------------

def test_the_model_is_called_with_training_false():
    _, images = pictures(1, (16, 24))
    p = predictor(noisy=True)
    first, second = (p.predict(images[0])["masks"][0] for _ in range(2))
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first, fx.blocks(16, 24, classes=CLASSES, seed=0))


def test_a_record_whose_output_is_logits_returns_score_maps_at_the_image_size():
    _, [image] = pictures(1, (30, 50))
    p = SegmentationPredictor(colour_model(), record(output="logits"))
    answer = p.predict(image)
    assert "masks" not in answer and answer["class_names"] == NAMES
    [scores] = answer["logits"]
    assert scores.shape == (30, 50, CLASSES) and scores.dtype == np.float32


def test_the_predictor_refuses_a_record_it_does_not_serve():
    classification = build_task_record(
        task="image_classification", model="vit/vit_pico", family="vit", variant="vit_pico",
        class_names=NAMES, input_shape=(16, 16, 3), normalize="unit", mean=[0.0] * 3, std=[1.0] * 3,
        data_kind="folder", source_channels=3,
    )
    with pytest.raises(ConfigError, match="the task is 'image_classification' but this predictor serves 'image_segmentation'"):
        SegmentationPredictor(colour_model(), classification)
    with pytest.raises(ConfigError, match="only 3 can be fed"):
        SegmentationPredictor(colour_model(), record(channels=1, mean=[0.0], std=[1.0]))


def test_check_output_names_the_shape_it_wanted():
    good = tf.zeros((1, 16, 24, CLASSES))
    check_output(good, record())
    for bad in (tf.zeros((1, 16, 24, CLASSES + 1)), tf.zeros((1, 24, 16, CLASSES)), tf.zeros((1, CLASSES))):
        with pytest.raises(ConfigError, match=r"should output \(16, 24, 4\)"):
            check_output(bad, record())


# ---------------------------------------------------------------------
# load_model: dispatch and what is not one run
# ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def saved(world, tmp_path_factory):
    """An untrained ConvUNext tiny at 16x24 saved as a run directory, with its record."""
    root = tmp_path_factory.mktemp("seg_run")
    spec = registry.MODELS["convunext"]
    model = recipes.build_model(spec, "tiny", 3, (16, 24, 3), {})
    model(np.zeros((1, 16, 24, 3), np.float32), training=False)
    directory = root / "run"
    directory.mkdir()
    model.save(directory / "best_model.keras")
    return directory


def place(saved, tmp_path, name="run", **changes):
    directory = tmp_path / name
    directory.mkdir()
    (directory / "best_model.keras").write_bytes((saved / "best_model.keras").read_bytes())
    value = record(size=(16, 24), classes=["pet", "background", "border"])
    value.update(changes)
    (directory / "task.json").write_text(json.dumps(value))
    return directory


def test_load_model_dispatches_on_the_task_and_predicts_at_the_image_size(saved, tmp_path):
    directory = place(saved, tmp_path)
    loaded = load_model(directory)
    assert isinstance(loaded, SegmentationPredictor)
    assert record_beside(directory)["task"] == SEGMENTATION
    assert input_shape_of(loaded.record) == (16, 24, 3)
    answer = loaded.predict(np.random.default_rng(0).integers(0, 256, (40, 31, 3), dtype=np.uint8))
    assert answer["masks"][0].shape == (40, 31) and answer["class_names"] == ["pet", "background", "border"]
    assert int(answer["masks"][0].max()) <= 2


def test_load_model_refuses_what_is_not_one_run(saved, tmp_path):
    with pytest.raises(ConfigError, match=r"does not take the input_size \[32, 32\] it names"):
        load_model(place(saved, tmp_path, "size", input_size=[32, 32]))
    with pytest.raises(ConfigError, match=r"names input_size \[16, 24\] and 4 classes, so it should output \(16, 24, 4\)"):
        load_model(place(saved, tmp_path, "classes", class_names=NAMES))
    with pytest.raises(ConfigError, match="'tiny' of 'convunext' does not have"):
        load_model(place(saved, tmp_path, "override", variant_overrides={"nope": 1}))
    with pytest.raises(ConfigError, match=r"'huge' is not a variant of 'convunext'"):
        load_model(place(saved, tmp_path, "variant", variant="huge", model="convunext/huge"))
    with pytest.raises(ConfigError, match=r"the family 'vit' trains image_classification, not 'image_segmentation'"):
        load_model(place(saved, tmp_path, "family", family="vit", variant="vit_pico", model="vit/vit_pico"))


def test_a_classification_record_under_a_segmentation_family_is_refused(saved, tmp_path):
    directory = place(saved, tmp_path)
    (directory / "task.json").write_text(json.dumps(build_task_record(
        task="image_classification", model="convunext/tiny", family="convunext", variant="tiny",
        class_names=NAMES, input_shape=(16, 16, 3), normalize="unit", mean=[0.0] * 3, std=[1.0] * 3,
        data_kind="folder", source_channels=3,
    )))
    with pytest.raises(ConfigError, match=r"the family 'convunext' trains image_segmentation, not 'image_classification'"):
        load_model(directory)


# ---------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------

def run_cli(*args):
    return cli.main(["predict", *args])


def refused(capsys, *args):
    """Run the command, expect exit 2 (a DLTrainError) and return what it said on stderr."""
    assert run_cli(*args) == 2
    return capsys.readouterr().err


def folder_of(tmp_path, names, size=(20, 28)):
    root = tmp_path / "photos"
    root.mkdir()
    paths = []
    for index, name in enumerate(names):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(fx.picture(fx.blocks(*size, classes=3, seed=index))).save(path)
        paths.append(path)
    return root, paths


def test_the_command_writes_a_png_per_image_and_the_index_last(saved, tmp_path, capsys):
    directory = place(saved, tmp_path)
    root, paths = folder_of(tmp_path, ["b.png", "sub/a.jpg", "c.png"])
    out = tmp_path / "answers"
    assert run_cli(f"model={directory}", f"data={root}", f"out={out}") == 0
    assert capsys.readouterr().out.strip() == str(out)
    index = json.loads((out / "index.json").read_text())
    assert index["class_names"] == ["pet", "background", "border"]
    assert [e["path"] for e in index["masks"]] == [str(p) for p in sorted(paths)]
    expected = load_model(directory).predict([str(p) for p in sorted(paths)])["masks"]
    for entry, mask in zip(index["masks"], expected):
        assert entry["size"] == [20, 28]
        np.testing.assert_array_equal(sd.decode_mask(str(out / entry["mask"])), mask)
    assert sorted(p.name for p in out.iterdir()) == ["a.png", "b.png", "c.png", "index.json"]


def test_two_images_with_one_stem_are_refused_before_the_model_is_loaded(saved, tmp_path, monkeypatch, capsys):
    directory = place(saved, tmp_path)
    root, _ = folder_of(tmp_path, ["a/cat.png", "b/cat.jpg"])

    def never(*_):
        raise AssertionError("the model was loaded")

    monkeypatch.setattr(predictor_module, "load_model", never)
    out = tmp_path / "answers"
    said = refused(capsys, f"model={directory}", f"data={root}", f"out={out}")
    assert "have the same name without its extension" in said and "cat.png" in said and "cat.jpg" in said
    assert not out.exists()
    with pytest.raises(ConfigError, match="have the same name"):
        mask_names([Path("x/a.png"), Path("y/a.jpg")])


def test_an_existing_out_is_refused_and_left_alone(saved, tmp_path, capsys):
    directory = place(saved, tmp_path)
    root, _ = folder_of(tmp_path, ["a.png"])
    out = tmp_path / "answers"
    out.mkdir()
    (out / "keep.txt").write_text("keep")
    assert "already exists" in refused(capsys, f"model={directory}", f"data={root}", f"out={out}")
    assert [p.name for p in out.iterdir()] == ["keep.txt"]


def test_a_logits_record_is_refused_by_the_command_before_the_model_is_loaded(saved, tmp_path, monkeypatch, capsys):
    directory = place(saved, tmp_path, output="logits")
    root, _ = folder_of(tmp_path, ["a.png"])
    monkeypatch.setattr(predictor_module, "load_model", lambda *_: (_ for _ in ()).throw(AssertionError("loaded")))
    said = refused(capsys, f"model={directory}", f"data={root}", f"out={tmp_path / 'answers'}")
    assert "output is 'logits', a score map per image, which is not a PNG" in said
    assert not (tmp_path / "answers").exists()


def test_a_run_that_fails_half_way_leaves_its_pngs_and_no_index(tmp_path):
    # CHUNK good images are answered and written; the chunk after them holds the bad one.
    files = []
    for index in range(CHUNK):
        files.append(tmp_path / f"g{index}.png")
        Image.fromarray(fx.picture(fx.blocks(16, 24, classes=3, seed=index))).save(files[-1])
    files.append(tmp_path / "z.png")
    files[-1].write_bytes(b"not an image")
    out = tmp_path / "answers"
    with pytest.raises(ConfigError, match="cannot be decoded"):
        write_predictions(predictor(), files, out)
    assert sorted(p.name for p in out.iterdir()) == sorted(f"g{i}.png" for i in range(CHUNK))
    assert not (out / "index.json").exists()
