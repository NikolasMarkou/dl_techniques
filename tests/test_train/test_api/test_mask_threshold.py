"""``data.mask_threshold``: a grey mask of two classes read as `value > T` (plan 98b866fb step 4, D-073/D-074).

A black and white mask stores its foreground as 255, which is no class id. The key
``data.mask_threshold=T`` (``0 < T < 255``) maps ``value > T`` to class 1 and the rest to class 0, in the ONE
decoder, for an 8-bit grey PNG and a 1-bit PNG alike. Checked against pixels written to ``tmp_path``:

* a 1-bit and an 8-bit mask of the same shapes give EQUAL classes, through the decoder, the checks, the
  loader (equal pixel counts and equal batches) and a real fit; the comparison is strict (`value == T` is
  class 0), also on a grey mask with intermediate values;
* the default is unchanged: a mask of exactly {0, 255} is refused and the message points at the key;
* what is refused, by key and before the run directory is claimed: three classes, an ignore index, a
  threshold outside (0, 255) or not finite, a palette mask (an index is not an intensity), a 16-bit mask
  (its scale is ambiguous), a Pet source (it has no such key);
* the record carries the threshold, the fingerprint sees it, a resume with another value is refused naming
  the key, and a predictor loaded from such a record answers exactly as one without it (the threshold is
  for masks, never for images).
"""

import json
from pathlib import Path

import keras
import numpy as np
import pytest
from PIL import Image

import train.api.recipes as recipes
import train.common.segmentation_data as sd
from train.api import registry
from train.api import segmentation_args as sa
from train.api.errors import ConfigError, ConfigValidationError
from train.api.predictor import build_segmentation_record
from train.api.segmentation_predictor import SegmentationPredictor
from train.api.tasks import register_builtin_tasks
from train.api.tasks.image_segmentation import ImageSegmentationTrainer, train_image_segmentation
from train.common.classification_data import DataSourceError

from . import _segmentation_fixtures as fx
from .test_image_segmentation_task import Killed

SIZE = (16, 24)
THRESHOLD = 127


def foreground(index: int) -> np.ndarray:
    """A bool mask of 8x8 blocks; across the set both classes occur."""
    return fx.blocks(*SIZE, classes=2, seed=index).astype(bool)


def mask_image(kind: str, fg: np.ndarray) -> Image.Image:
    if kind == "bit1":
        return Image.fromarray(fg)
    if kind == "grey255":
        return Image.fromarray(fg.astype(np.uint8) * 255)
    if kind == "grey_soft":  # intermediate intensities: 30 is background, 200 is foreground
        return Image.fromarray(np.where(fg, 200, 30).astype(np.uint8))
    if kind == "palette":
        image = Image.frombytes("P", (fg.shape[1], fg.shape[0]), fg.astype(np.uint8).tobytes())
        image.putpalette([0, 0, 0, 255, 255, 255] + [0] * 762)
        return image
    if kind == "grey16":
        return Image.fromarray(fg.astype(np.uint16) * 65535)
    raise AssertionError(kind)


EXPECTED_MODE = {"bit1": "1", "grey255": "L", "grey_soft": "L", "palette": "P", "grey16": "I;16"}


def write_set(root: Path, kind: str, count: int = 12):
    images, masks = root / "images", root / "masks"
    images.mkdir(parents=True), masks.mkdir(parents=True)
    for index in range(count):
        fg = foreground(index)
        Image.fromarray(fx.picture(fg.astype(np.uint8))).save(images / f"p{index:03d}.png")
        image = mask_image(kind, fg)
        image.save(masks / f"p{index:03d}.png")
        assert Image.open(masks / f"p{index:03d}.png").mode == EXPECTED_MODE[kind], kind
    return images, masks


@pytest.fixture(scope="module")
def sets(tmp_path_factory):
    root = tmp_path_factory.mktemp("thresholds")
    return {kind: write_set(root / kind, kind) for kind in EXPECTED_MODE}


def data_of(sets, kind, **extra):
    images, masks = sets[kind]
    mapping = {
        "format": "folder", "train_images": str(images), "train_masks": str(masks),
        "num_classes": 2, "validation_split": 0.25,
    }
    mapping.update(extra)
    return mapping


@pytest.fixture(scope="module")
def world():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(registry, "MODELS", {})
        patch.setattr(registry, "TASKS", {})
        recipes.register_builtin_models()
        register_builtin_tasks()
        yield


# ---------------------------------------------------------------------
# The decoder: 1-bit and 8-bit give equal classes; the comparison is strict
# ---------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["bit1", "grey255", "grey_soft"])
def test_the_decoder_maps_value_above_the_threshold_to_class_one_for_every_supported_mode(kind, sets):
    for index, path in enumerate(sorted(sets[kind][1].iterdir())):
        got = sd.decode_mask(str(path), mask_threshold=THRESHOLD)
        assert got.dtype == np.int32 and got.shape == SIZE
        np.testing.assert_array_equal(got, foreground(index).astype(np.int32))


def test_a_one_bit_and_an_eight_bit_mask_give_equal_classes(sets):
    for one, eight in zip(sorted(sets["bit1"][1].iterdir()), sorted(sets["grey255"][1].iterdir())):
        np.testing.assert_array_equal(
            sd.decode_mask(str(one), mask_threshold=THRESHOLD), sd.decode_mask(str(eight), mask_threshold=THRESHOLD))


def test_the_comparison_is_strictly_greater_than(sets, tmp_path):
    values = np.array([[0, 126, 127, 128, 255, 1, 254, 100]], np.uint8)
    path = tmp_path / "m.png"
    Image.fromarray(values).save(path)
    np.testing.assert_array_equal(sd.decode_mask(str(path), mask_threshold=127)[0], [0, 0, 0, 1, 1, 0, 1, 0])
    np.testing.assert_array_equal(sd.decode_mask(str(path), mask_threshold=126.5)[0], [0, 0, 1, 1, 1, 0, 1, 0])
    np.testing.assert_array_equal(sd.decode_mask(str(path), mask_threshold=254)[0], [0, 0, 0, 0, 1, 0, 0, 0])
    # without a threshold the raw values come back, as ever
    np.testing.assert_array_equal(sd.decode_mask(str(path))[0], values[0])


@pytest.mark.parametrize("kind, word", [("palette", "palette"), ("grey16", "16-bit")])
def test_a_palette_and_a_sixteen_bit_mask_are_refused_naming_the_file_and_the_reason(kind, word, sets):
    path = str(sorted(sets[kind][1].iterdir())[0])
    with pytest.raises(sd.PairError, match=word) as caught:
        sd.decode_mask(path, mask_threshold=THRESHOLD)
    assert path in str(caught.value) and "mask_threshold" in str(caught.value) and caught.value.which == "mask"
    # without a threshold they are read as before (indices, and the stored 16-bit values)
    assert set(np.unique(sd.decode_mask(path))) <= {0, 1, 65535}


# ---------------------------------------------------------------------
# The checks see the thresholded values; the default refusal stays
# ---------------------------------------------------------------------

def check(listing, **kw):
    return sd.check_pairs(listing.train_images, listing.train_masks, num_classes=2, ignore_index=None,
                          image_key="train_images", mask_key="train_masks", **kw)


@pytest.mark.parametrize("kind", ["bit1", "grey255", "grey_soft"])
def test_check_pairs_reads_the_thresholded_values_and_counts_their_pixels(kind, sets):
    listing = sd.scan_segmentation_folder(*map(str, sets[kind]))
    scan = check(listing, mask_threshold=THRESHOLD)
    assert scan.values == (0, 1)
    fg = sum(int(foreground(i).sum()) for i in range(12))
    assert scan.class_pixels == (12 * SIZE[0] * SIZE[1] - fg, fg)


@pytest.mark.parametrize("kind", ["bit1", "grey255"])
def test_the_default_still_refuses_a_black_and_white_mask_and_points_at_the_key(kind, sets):
    listing = sd.scan_segmentation_folder(*map(str, sets[kind]))
    with pytest.raises(DataSourceError) as caught:
        check(listing)
    text = str(caught.value)
    assert "data.mask_threshold" in text and "0 and 255" in text
    assert "does not accept" not in text and "yet" not in text


def test_the_pre_claim_sample_sees_the_thresholded_values_and_refuses_a_palette_mask(sets):
    ok = sd.scan_segmentation_folder(*map(str, sets["grey255"]))
    assert sd.check_listing_sample(ok, num_classes=2, ignore_index=None, mask_threshold=THRESHOLD).values == (0, 1)
    palette = sd.scan_segmentation_folder(*map(str, sets["palette"]))
    with pytest.raises(DataSourceError, match="palette"):
        sd.check_listing_sample(palette, num_classes=2, ignore_index=None, mask_threshold=THRESHOLD)


def test_the_fingerprint_sees_the_threshold(sets):
    listing = sd.scan_segmentation_folder(*map(str, sets["grey255"]))
    plain = sd.pair_fingerprint(listing)
    at_127 = sd.pair_fingerprint(listing, mask_threshold=127)
    assert at_127 != plain and at_127 != sd.pair_fingerprint(listing, mask_threshold=100)
    assert at_127 == sd.pair_fingerprint(listing, mask_threshold=127.0)
    assert sd.pair_fingerprint(listing, "content", mask_threshold=127) != at_127


def loaded(sets, kind, threshold):
    listing = sd.scan_segmentation_folder(*map(str, sets[kind]))
    return sd.load_pair_data(
        listing, num_classes=2, class_names=["bg", "fg"], ignore_index=None, batch_size=4, seed=3,
        size=SIZE, validation_split=0.25, eval_samples=4, max_samples=None, mask_threshold=threshold)


def test_a_one_bit_and_an_eight_bit_source_load_the_same_batches(sets):
    one, eight = loaded(sets, "bit1", THRESHOLD), loaded(sets, "grey255", THRESHOLD)
    assert one.mask_values == eight.mask_values == (0, 1)
    for a, b in zip(one.select_ds.as_numpy_iterator(), eight.select_ds.as_numpy_iterator()):
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
    for a, b in zip(one.train_ds.as_numpy_iterator(), eight.train_ds.as_numpy_iterator()):
        np.testing.assert_array_equal(a[1], b[1])
    soft = loaded(sets, "grey_soft", 100)
    for a, b in zip(one.select_ds.as_numpy_iterator(), soft.select_ds.as_numpy_iterator()):
        np.testing.assert_array_equal(a[1], b[1])
    from train.common import segmentation_run as run

    counts = run.class_pixel_counts(one.select_ds, 2, None)
    assert counts.sum() == one.n_select * SIZE[0] * SIZE[1] and counts.min() >= 0
    np.testing.assert_array_equal(counts, run.class_pixel_counts(eight.select_ds, 2, None))


# ---------------------------------------------------------------------
# What the config refuses, by key, before the claim
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "extra, text",
    [
        ({"num_classes": 3, "mask_threshold": 128}, "needs exactly 2 classes"),
        ({"class_names": ["a", "b", "c"], "num_classes": None, "mask_threshold": 128}, "needs exactly 2 classes"),
        ({"ignore_index": 255, "mask_threshold": 128}, "cannot be used with data.ignore_index"),
        ({"ignore_index": 2, "mask_threshold": 128}, "cannot be used with data.ignore_index"),
        ({"mask_threshold": 0}, "between 0 and 255"),
        ({"mask_threshold": 255}, "between 0 and 255"),
        ({"mask_threshold": -3.0}, "between 0 and 255"),
        ({"mask_threshold": 300}, "between 0 and 255"),
        ({"mask_threshold": float("inf")}, "expected float or null"),
        ({"mask_threshold": float("nan")}, "expected float or null"),
    ],
)
def test_a_threshold_out_of_range_or_with_the_wrong_classes_or_a_void_value_is_refused_by_its_key(extra, text, sets):
    base = {"format": "folder", "train_images": "i", "train_masks": "m", "num_classes": 2}
    with pytest.raises(ConfigValidationError) as caught:
        sa.load_seg_data_config({**base, **extra})
    joined = "\n".join(caught.value.errors)
    assert "'mask_threshold'" in joined and text in joined, joined


def test_a_valid_threshold_is_accepted_for_folder_and_csv_and_not_for_pet():
    folder = sa.load_seg_data_config({"format": "folder", "train_images": "i", "train_masks": "m",
                                      "num_classes": 2, "mask_threshold": 127})
    assert folder.mask_threshold == 127
    csv = sa.load_seg_data_config({"format": "csv", "train_csv": "t.csv", "class_names": ["a", "b"], "mask_threshold": 0.5 + 126})
    assert csv.mask_threshold == 126.5
    with pytest.raises(ConfigValidationError, match="mask_threshold"):
        sa.load_seg_data_config({"dataset": "oxford_pet", "mask_threshold": 127})


@pytest.mark.parametrize("kind, word", [("palette", "palette"), ("grey16", "16-bit")])
def test_a_palette_or_sixteen_bit_source_is_refused_before_the_claim(kind, word, sets, world, tmp_path):
    out = tmp_path / "never"
    with pytest.raises(ConfigError, match=rf"data\.train_masks.*{word}"):
        train_image_segmentation(
            out=str(out), model="convunext/tiny", data=data_of(sets, kind, mask_threshold=THRESHOLD), epochs=1,
            transform_args={"image_size": list(SIZE)})
    assert not out.exists()


def test_a_black_and_white_source_without_the_key_is_refused_before_the_claim_pointing_at_it(sets, world, tmp_path):
    out = tmp_path / "never"
    with pytest.raises(ConfigError, match=r"data\.train_masks.*data\.mask_threshold"):
        train_image_segmentation(
            out=str(out), model="convunext/tiny", data=data_of(sets, "grey255"), epochs=1,
            transform_args={"image_size": list(SIZE)})
    assert not out.exists()


# ---------------------------------------------------------------------
# A real run: summary, record, a resume with another value, the predictor
# ---------------------------------------------------------------------

def run_request(out, sets, kind, threshold, **changes):
    changes.setdefault("analysis_args", {"model_analysis": False, "figures": False})
    return seg_request_for(out, data_of(sets, kind, mask_threshold=threshold), **changes)


def seg_request_for(out, data, **changes):
    from train.api.config import validate
    from train.api.tasks.image_segmentation import ImageSegmentationConfig

    mapping = {
        "out": str(out), "model": "convunext/tiny", "data": data, "epochs": 2, "batch_size": 4,
        "transform_args": {"image_size": list(SIZE)},
    }
    mapping.update(changes)
    return validate(ImageSegmentationConfig, mapping)


@pytest.fixture(scope="module")
def finished(world, sets, tmp_path_factory):
    out = tmp_path_factory.mktemp("threshold_fit") / "run"
    result = train_image_segmentation(
        out=str(out), model="convunext/tiny", data=data_of(sets, "bit1", mask_threshold=THRESHOLD), epochs=1,
        batch_size=4, transform_args={"image_size": list(SIZE)},
        analysis_args={"model_analysis": False, "figures": False})
    return result, json.loads(Path(result.summary_path).read_text()), out


def test_a_run_on_one_bit_masks_records_the_threshold_and_trains_two_classes(finished):
    result, summary, out = finished
    record = json.loads((out / "task.json").read_text())
    assert record["mask_threshold"] == THRESHOLD and record["ignore_index"] is None
    assert record["class_names"] == ["0", "1"]
    assert summary["mask_values"] == [0, 1] and summary["ignore_index"] is None
    assert json.loads((out / "config.json").read_text())["data"]["mask_threshold"] == THRESHOLD
    baseline = summary["majority_class_baseline"]
    assert baseline["pixels"] == summary["n_val"] * SIZE[0] * SIZE[1]
    # D-016: the summary's notes name the key and the value (review-iter-1 concern 2).
    assert any(note.startswith(f"`data.mask_threshold` is {THRESHOLD:g}:") for note in summary["notes"])


def test_a_resume_with_another_threshold_is_refused_naming_the_key_and_the_same_one_resumes(world, sets, tmp_path):
    out = tmp_path / "run"
    with pytest.raises(RuntimeError, match="killed"):
        Killed(run_request(out, sets, "grey_soft", 100, epochs=3)).run()
    with pytest.raises(ConfigError, match=r"data\.mask_threshold: 100\.0 -> 128\.0"):
        ImageSegmentationTrainer(run_request(out, sets, "grey_soft", 128, epochs=3, resume_interrupted=True)).run()
    resumed = ImageSegmentationTrainer(run_request(out, sets, "grey_soft", 100, epochs=3, resume_interrupted=True)).run()
    assert json.loads(Path(resumed.summary_path).read_text())["resumed_after_epochs"] == 2


def test_the_predictor_of_a_thresholded_run_answers_exactly_as_one_without_the_key():
    colours = fx.PALETTE[:2].astype(np.float64) / 255.0
    inputs = keras.Input((*SIZE, 3))
    layer = keras.layers.Conv2D(2, 1)
    model = keras.Model(inputs, layer(inputs))
    layer.set_weights([(2.0 * colours.T)[None, None], -(colours ** 2).sum(1)])

    def predictor(threshold):
        value = build_segmentation_record(
            model="convunext/tiny", family="convunext", variant="tiny", class_names=["bg", "fg"],
            input_size=SIZE, channels=3, normalize="unit", mean=[0.0] * 3, std=[1.0] * 3,
            mask_threshold=threshold)
        return SegmentationPredictor(model, value)

    images = [fx.picture(foreground(i).astype(np.uint8)) for i in range(3)]
    with_key, without = predictor(127).predict(images), predictor(None).predict(images)
    for a, b, i in zip(with_key["masks"], without["masks"], range(3)):
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a, foreground(i).astype(np.uint8))
