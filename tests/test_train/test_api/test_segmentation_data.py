"""Segmentation data: one decoder, the listing, the mask checks, the fingerprint and the pipeline.

Every claim is checked against pixels written to ``tmp_path`` here, not against the code that reads
them:

* a JPEG stored sideways (EXIF orientation 6) beside an upright landscape mask decodes to a pair
  that agrees pixel by pixel, because the colour of a pixel names its class; a decoder that
  ignored the orientation would give two sizes (the measured SUT-Crack case: 119 of 130 files);
* a palette PNG is read as its INDICES (its palette colours are chosen to differ from them), a
  1-bit PNG as 0 and 255, a 16-bit grey PNG as itself; an RGB, RGBA or JPEG mask is refused;
* the refusals name the file or both lists: an unpaired stem, two files of one stem, a file listed
  twice (a symlink is the file it points to), an image or a mask in both splits, a pair of two
  sizes, a value that is no class, and a black and white mask, which points at
  ``data.mask_threshold``;
* the fingerprint changes when a mask alone changes (its size in listing mode, its bytes in
  content mode) and when the rows are reordered;
* the pipelines are deterministic per seed, keep void as void through the resize, invent no mask
  value, know their length, and a non-square size comes out non-square;
* nothing before the claim makes a tensor: ``seg_facts`` runs with ``tf.data`` poisoned.
"""

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

import train.common.segmentation_data as sd
from dl_techniques.datasets.vision.augment_hook import seeded
from train.api import segmentation_args as sa
from train.api.errors import ConfigError, ConfigValidationError
from train.common.classification_data import DataSourceError

from tests.test_train.test_api._segmentation_fixtures import (
    blocks,
    class_of,
    listing_of,
    load,
    make_folder,
    picture,
    write_csv,
    write_pair,
)

# ---------------------------------------------------------------------
# The decoder
# ---------------------------------------------------------------------

def sideways_jpeg(path: Path, upright: np.ndarray) -> None:
    """Store ``upright`` (H, W, 3) the way a phone does: rotated, with EXIF orientation 6."""
    # Orientation 6: the stored image must be rotated 90 degrees clockwise to be upright, so it is
    # stored rotated 90 degrees counter-clockwise (np.rot90 with k=1 is counter-clockwise).
    exif = Image.Exif()
    exif[0x0112] = 6
    Image.fromarray(np.rot90(upright, 1).copy()).save(path, quality=100, subsampling=0, exif=exif)


def test_an_exif_6_jpeg_and_a_landscape_mask_agree_pixel_by_pixel(tmp_path):
    mask = blocks(48, 80, classes=4, seed=3)  # landscape: 48 high, 80 wide
    jpeg, png = tmp_path / "a.jpg", tmp_path / "a.png"
    sideways_jpeg(jpeg, picture(mask))
    Image.fromarray(mask).save(png)
    stored = Image.open(jpeg).size
    assert stored == (48, 80), "the fixture must really be stored sideways (width 48, height 80)"
    image, decoded = sd.read_pair(str(jpeg), str(png))
    assert image.shape == (48, 80, 3) and decoded.shape == (48, 80)
    assert np.array_equal(decoded, mask)
    assert np.array_equal(class_of(image), mask), "the photograph is upright against its mask"
    assert sd.image_size_of(str(jpeg)) == (48, 80), "the header read applies the orientation too"


def test_a_mask_stored_sideways_with_its_own_orientation_is_turned_too(tmp_path):
    mask = blocks(48, 80, classes=4, seed=5)
    exif = Image.Exif()
    exif[0x0112] = 6
    sideways = tmp_path / "m.png"
    Image.fromarray(np.rot90(mask, 1).copy()).save(sideways, exif=exif)
    png = tmp_path / "a.png"
    Image.fromarray(picture(mask)).save(png)
    assert Image.open(sideways).size == (48, 80)
    assert np.array_equal(sd.decode_mask(str(sideways)), mask), "the one decoder turns masks as it turns images"
    image, decoded = sd.read_pair(str(png), str(sideways))
    assert np.array_equal(class_of(image), decoded)


def test_the_pipeline_decodes_a_sideways_photograph_upright(tmp_path):
    images, masks = tmp_path / "images", tmp_path / "masks"
    images.mkdir(), masks.mkdir()
    for index in range(4):
        mask = blocks(48, 80, classes=3, seed=index)
        sideways_jpeg(images / f"s{index}.jpg", picture(mask))
        Image.fromarray(mask).save(masks / f"s{index}.png")
    listing = sd.scan_segmentation_folder(str(images), str(masks))
    dataset = sd.pair_examples(listing.train_images, listing.train_masks, training=False, seed=0,
                               augment=None, size=(48, 80))
    for image, mask in dataset.as_numpy_iterator():
        assert np.array_equal(class_of(np.round(image * 255).astype(np.uint8)), mask)


def test_a_pair_of_two_sizes_after_exif_names_both_files(tmp_path):
    mask = blocks(48, 80)
    jpeg, png = tmp_path / "a.jpg", tmp_path / "a.png"
    sideways_jpeg(jpeg, picture(mask))
    Image.fromarray(np.rot90(mask).copy()).save(png)  # the mask stored in the photograph's sideways form
    with pytest.raises(sd.PairError) as caught:
        sd.read_pair(str(jpeg), str(png))
    text = str(caught.value)
    assert str(jpeg) in text and str(png) in text and "48x80" in text and "80x48" in text
    assert caught.value.which == "mask"


def test_a_palette_png_is_read_as_its_indices(tmp_path):
    indices = blocks(16, 16, classes=3, seed=1)
    image = Image.fromarray(indices, mode="P")
    palette = [0] * 768
    for index, colour in enumerate([(250, 0, 0), (0, 250, 0), (0, 0, 250)]):  # index 0 is bright red
        palette[3 * index:3 * index + 3] = colour
    image.putpalette(palette)
    path = tmp_path / "m.png"
    image.save(path)
    assert Image.open(path).mode == "P"
    assert np.array_equal(sd.decode_mask(str(path)), indices)


def test_a_one_bit_png_is_zero_and_255(tmp_path):
    bits = np.zeros((10, 14), bool)
    bits[:, 7:] = True
    path = tmp_path / "bin.png"
    Image.fromarray(bits).save(path)
    assert Image.open(path).mode == "1"
    decoded = sd.decode_mask(str(path))
    assert set(np.unique(decoded)) == {0, 255} and decoded[0, 0] == 0 and decoded[0, 13] == 255


def test_a_sixteen_bit_png_is_read_as_itself(tmp_path):
    values = np.asarray([[0, 300], [1000, 65535]], np.uint16)
    path = tmp_path / "wide.png"
    Image.fromarray(values).save(path)
    assert np.array_equal(sd.decode_mask(str(path)), values.astype(np.int32))


@pytest.mark.parametrize("kind", ["rgb", "rgba", "jpeg"])
def test_a_colour_or_jpeg_mask_is_refused_by_name(tmp_path, kind):
    path = tmp_path / ("m.jpg" if kind == "jpeg" else "m.png")
    grey = np.full((8, 8), 1, np.uint8)
    if kind == "rgb":
        Image.fromarray(np.stack([grey] * 3, -1)).save(path)
    elif kind == "rgba":
        Image.fromarray(np.stack([grey] * 4, -1)).save(path)
    else:
        Image.fromarray(grey).save(path, quality=100)
    with pytest.raises(sd.PairError) as caught:
        sd.decode_mask(str(path))
    assert str(path) in str(caught.value) and caught.value.which == "mask"


def test_a_file_that_is_no_image_is_named(tmp_path):
    path = tmp_path / "x.png"
    path.write_bytes(b"not an image")
    for call in (sd.decode_image, sd.decode_mask):
        with pytest.raises(sd.PairError, match="cannot be decoded"):
            call(str(path))


# ---------------------------------------------------------------------
# The listing
# ---------------------------------------------------------------------

def test_a_folder_listing_joins_by_stem_in_sorted_order(tmp_path):
    images, masks = tmp_path / "i", tmp_path / "m"
    for stem in ("b", "a", "c"):
        write_pair(images, masks, stem, blocks(8, 8))
    (images / ".hidden.png").write_bytes(b"")
    (images / "notes.txt").write_text("x")
    listing = sd.scan_segmentation_folder(str(images), str(masks))
    assert [Path(p).stem for p in listing.train_images] == ["a", "b", "c"]
    assert [Path(p).stem for p in listing.train_masks] == ["a", "b", "c"]
    assert listing.test_images is None and listing.source == "folder"


def test_an_unpaired_stem_names_both_counts_and_the_first_few(tmp_path):
    images, masks = tmp_path / "i", tmp_path / "m"
    for stem in ("a", "b", "c", "d"):
        write_pair(images, masks, stem, blocks(8, 8))
    (images / "lonely1.png").write_bytes(b"")
    (images / "lonely2.png").write_bytes(b"")
    (masks / "orphan.png").write_bytes(b"")
    with pytest.raises(DataSourceError) as caught:
        sd.scan_segmentation_folder(str(images), str(masks))
    text = str(caught.value)
    assert caught.value.key == "train_images"
    assert "6 images" in text and "5 masks" in text
    assert "2 image(s) have no mask" in text and "'lonely1'" in text and "'lonely2'" in text
    assert "1 mask(s) have no image" in text and "'orphan'" in text


def test_two_files_of_one_stem_are_refused(tmp_path):
    images, masks = tmp_path / "i", tmp_path / "m"
    write_pair(images, masks, "a", blocks(8, 8))
    write_pair(images, masks, "b", blocks(8, 8))
    Image.fromarray(picture(blocks(8, 8))).save(images / "a.jpg")
    with pytest.raises(DataSourceError, match=r"'a\.jpg' and 'a\.png'|'a\.png' and 'a\.jpg'.*same stem"):
        sd.scan_segmentation_folder(str(images), str(masks))


def test_a_missing_directory_and_a_single_pair_are_refused(tmp_path):
    images, masks = make_folder(tmp_path, count=1)
    with pytest.raises(DataSourceError, match="needs at least 2|at least 2"):
        sd.scan_segmentation_folder(str(images), str(masks))
    with pytest.raises(DataSourceError, match="is not a directory") as caught:
        sd.scan_segmentation_folder(str(tmp_path / "nope"), str(masks))
    assert caught.value.key == "train_images"
    with pytest.raises(DataSourceError, match="is not a directory") as caught:
        sd.scan_segmentation_folder(str(images), str(tmp_path / "nope"))
    assert caught.value.key == "train_masks"


def test_a_test_directory_needs_its_mask_directory(tmp_path):
    images, masks = make_folder(tmp_path)
    with pytest.raises(DataSourceError, match="go together") as caught:
        sd.scan_segmentation_folder(str(images), str(masks), str(images), None)
    assert caught.value.key == "test_masks"


def test_a_symlink_is_the_file_it_points_to(tmp_path):
    images, masks = make_folder(tmp_path, count=4)
    os.symlink(images / "p000.png", images / "p000_again.png")
    os.symlink(masks / "p000.png", masks / "p000_again.png")
    with pytest.raises(DataSourceError, match="listed more than once") as caught:
        sd.scan_segmentation_folder(str(images), str(masks))
    assert caught.value.key == "train_images" and "p000" in str(caught.value)


def test_a_dangling_link_is_named(tmp_path):
    images, masks = make_folder(tmp_path, count=4)
    os.symlink(tmp_path / "gone.png", images / "dangling.png")
    with pytest.raises(DataSourceError, match="leads nowhere"):
        sd.scan_segmentation_folder(str(images), str(masks))


def test_a_test_image_that_is_a_train_image_is_refused(tmp_path):
    images, masks = make_folder(tmp_path, count=4)
    t_images, t_masks = make_folder(tmp_path, count=2, prefix="t_")
    # The test pair: a new mask, but its image is a hard link to a train image.
    (t_images / "p000.png").unlink()
    os.link(images / "p000.png", t_images / "p000.png")
    with pytest.raises(DataSourceError, match="also train files") as caught:
        sd.scan_segmentation_folder(str(images), str(masks), str(t_images), str(t_masks))
    assert caught.value.key == "test_images"


def test_a_test_mask_that_is_a_train_mask_is_refused(tmp_path):
    images, masks = make_folder(tmp_path, count=4)
    t_images, t_masks = make_folder(tmp_path, count=2, prefix="t_")
    (t_masks / "p001.png").unlink()
    os.symlink(masks / "p001.png", t_masks / "p001.png")
    with pytest.raises(DataSourceError, match="also train files") as caught:
        sd.scan_segmentation_folder(str(images), str(masks), str(t_images), str(t_masks))
    assert caught.value.key == "test_masks", "a mask belongs to exactly one split"


def test_a_csv_listing_is_relative_to_the_csv_and_keeps_its_order(tmp_path):
    images, masks = make_folder(tmp_path, count=5)
    rows = [(f"images/p00{i}.png", f"masks/p00{i}.png") for i in (3, 1, 4, 0, 2)]
    listing = sd.read_segmentation_csv(str(write_csv(tmp_path / "d.csv", rows)))
    assert [Path(p).name for p in listing.train_images] == ["p003.png", "p001.png", "p004.png", "p000.png", "p002.png"]
    assert listing.source == "csv" and listing.train_bases == (str(tmp_path), str(tmp_path))


def test_csv_refusals_name_the_line_the_column_and_the_key(tmp_path):
    images, masks = make_folder(tmp_path, count=4)
    good = [(f"images/p00{i}.png", f"masks/p00{i}.png") for i in range(4)]
    with pytest.raises(DataSourceError, match="column 'mask' is not in") as caught:
        sd.read_segmentation_csv(str(write_csv(tmp_path / "a.csv", good, header=("image", "label"))))
    assert caught.value.key == "mask_column"
    with pytest.raises(DataSourceError, match=r"line 3: .*does not exist") as caught:
        sd.read_segmentation_csv(str(write_csv(tmp_path / "b.csv", [good[0], ("images/none.png", good[1][1])])))
    assert caught.value.key == "train_csv"
    with pytest.raises(DataSourceError, match="empty 'mask'"):
        sd.read_segmentation_csv(str(write_csv(tmp_path / "c.csv", [good[0], (good[1][0], "")])))
    with pytest.raises(DataSourceError, match="listed more than once"):
        sd.read_segmentation_csv(str(write_csv(tmp_path / "d.csv", good + [good[0]])))
    with pytest.raises(DataSourceError, match="listed more than once"):
        sd.read_segmentation_csv(str(write_csv(tmp_path / "e.csv", good[:2] + [(good[2][0], good[0][1])])))
    other = sd.read_segmentation_csv(str(write_csv(tmp_path / "f.csv", good[:3])), str(write_csv(tmp_path / "g.csv", good[3:])))
    assert len(other.test_images) == 1
    with pytest.raises(DataSourceError, match="also train files") as caught:
        sd.read_segmentation_csv(str(write_csv(tmp_path / "h.csv", good[:3])), str(write_csv(tmp_path / "i.csv", [good[0]])))
    assert caught.value.key == "test_csv"


# ---------------------------------------------------------------------
# The mask check
# ---------------------------------------------------------------------

def test_a_mask_value_that_is_no_class_names_the_file_and_the_value(tmp_path):
    listing = listing_of(tmp_path, count=4, classes=3)
    bad = np.asarray(Image.open(listing.train_masks[2])).copy()
    bad[0, 0] = 7
    Image.fromarray(bad).save(listing.train_masks[2])
    with pytest.raises(DataSourceError) as caught:
        sd.check_pairs(listing.train_images, listing.train_masks, num_classes=3, ignore_index=None, image_key="train_images", mask_key="train_masks")
    text = str(caught.value)
    assert listing.train_masks[2] in text and "value 7" in text and "data.ignore_index" in text


def test_a_black_and_white_mask_points_at_the_threshold_key(tmp_path):
    images, masks = tmp_path / "i", tmp_path / "m"
    images.mkdir(), masks.mkdir()
    for index in range(4):
        classes = blocks(16, 16, classes=2, seed=index)
        Image.fromarray(picture(classes)).save(images / f"p{index}.png")
        Image.fromarray(classes * 255).save(masks / f"p{index}.png")
    listing = sd.scan_segmentation_folder(str(images), str(masks))
    with pytest.raises(DataSourceError) as caught:
        sd.check_pairs(listing.train_images, listing.train_masks, num_classes=2, ignore_index=None, image_key="train_images", mask_key="train_masks")
    assert "data.mask_threshold" in str(caught.value) and "0 and 255" in str(caught.value)
    # 255 as the declared void value is a legitimate mask of class 0 and void: no refusal.
    scan = sd.check_pairs(listing.train_images, listing.train_masks, num_classes=2, ignore_index=255, image_key="train_images", mask_key="train_masks")
    assert scan.values == (0, 255) and scan.pairs == 4


def test_every_pair_the_run_will_use_is_checked_before_fit(tmp_path):
    listing = listing_of(tmp_path, count=12)
    for position, path in enumerate(listing.train_masks):
        original = Path(path).read_bytes()
        bad = np.asarray(Image.open(path)).copy()
        bad[0, 0] = 9
        Image.fromarray(bad).save(path)
        with pytest.raises(DataSourceError, match="value 9") as caught:
            load(listing)
        assert path in str(caught.value), f"pair {position} (fit or selection) was not checked"
        Path(path).write_bytes(original)
    assert load(listing).n_train == 9


def test_a_corrupt_image_is_found_by_the_full_decode_not_the_header_read(tmp_path):
    images, masks = tmp_path / "i", tmp_path / "m"
    for index in range(4):
        write_pair(images, masks, f"p{index}", blocks(32, 32, seed=index), image_suffix=".jpg")
    listing = sd.scan_segmentation_folder(str(images), str(masks))
    victim = Path(listing.train_images[1])
    data = victim.read_bytes()
    victim.write_bytes(data[:-30])  # a truncated JPEG: every header segment is intact
    sd.check_pairs(listing.train_images, listing.train_masks, num_classes=3, ignore_index=None, image_key="ki", mask_key="km")
    with pytest.raises(DataSourceError, match="cannot be decoded") as caught:
        sd.check_pairs(listing.train_images, listing.train_masks, num_classes=3, ignore_index=None, image_key="ki", mask_key="km", image_pixels=True)
    assert str(victim) in str(caught.value) and caught.value.key == "ki"


def test_the_pre_claim_sample_is_evenly_spaced_and_bounded():
    assert sd.sample_indices(5, 32) == [0, 1, 2, 3, 4]
    picks = sd.sample_indices(1000, 32)
    assert len(picks) == 32 and picks[0] == 0 and picks[-1] > 900 and picks == sorted(set(picks))


# ---------------------------------------------------------------------
# The fingerprint
# ---------------------------------------------------------------------

def test_the_fingerprint_sees_a_mask_alone(tmp_path):
    listing = listing_of(tmp_path, count=4)
    for path in listing.train_masks:  # uncompressed, so one edited pixel keeps the file size
        Image.fromarray(np.asarray(Image.open(path)).copy()).save(path, compress_level=0)
    before_listing = sd.pair_fingerprint(listing, "listing")
    before_content = sd.pair_fingerprint(listing, "content")
    mask_path = Path(listing.train_masks[1])
    size_before = mask_path.stat().st_size
    same_size = np.asarray(Image.open(mask_path)).copy()
    same_size[0, 0] = (same_size[0, 0] + 1) % 3
    Image.fromarray(same_size).save(mask_path, compress_level=0)
    assert mask_path.stat().st_size == size_before, "the fixture must change the bytes and not the size"
    assert sd.pair_fingerprint(listing, "listing") == before_listing, "listing mode cannot see a same-size edit"
    assert sd.pair_fingerprint(listing, "content") != before_content, "content mode reads the mask's bytes"
    # A different-size mask (another file written over it) changes the listing digest as well.
    Image.fromarray(np.zeros((24, 32), np.uint8)).save(mask_path)
    assert sd.pair_fingerprint(listing, "listing") != before_listing


def test_the_fingerprint_follows_the_listing_order_and_the_image(tmp_path):
    images, masks = make_folder(tmp_path, count=5)
    rows = [(f"images/p00{i}.png", f"masks/p00{i}.png") for i in range(5)]
    forward = sd.read_segmentation_csv(str(write_csv(tmp_path / "a.csv", rows)))
    backward = sd.read_segmentation_csv(str(write_csv(tmp_path / "b.csv", rows[::-1])))
    assert sd.pair_fingerprint(forward) != sd.pair_fingerprint(backward), "the seeded split permutes positions"
    assert sd.pair_fingerprint(forward) == sd.pair_fingerprint(forward)
    assert sd.pair_fingerprint(forward, "listing") != sd.pair_fingerprint(forward, "content")
    with pytest.raises(ValueError, match="fingerprint mode"):
        sd.pair_fingerprint(forward, "bytes")
    before = sd.pair_fingerprint(forward, "listing")
    Image.fromarray(np.zeros((24, 32, 3), np.uint8)).save(images / "p000.png")  # another file of another size
    assert sd.pair_fingerprint(sd.read_segmentation_csv(str(tmp_path / "a.csv")), "listing") != before


def test_the_fingerprint_does_not_move_with_the_tree(tmp_path):
    first = listing_of(tmp_path / "one", count=4)
    second = listing_of(tmp_path / "two", count=4)
    assert sd.pair_fingerprint(first, "content") == sd.pair_fingerprint(second, "content")


# ---------------------------------------------------------------------
# The pipelines
# ---------------------------------------------------------------------

def batches(data: sd.SegmentationData, epochs: int = 2) -> List[Tuple[np.ndarray, np.ndarray]]:
    out = []
    for _ in range(epochs):
        out.extend(data.train_ds.as_numpy_iterator())
    return out


def test_the_record_has_the_counts_the_split_says(tmp_path):
    listing = listing_of(tmp_path, count=12)
    data = load(listing)
    assert (data.n_train, data.n_select, data.n_test) == (9, 3, 0)
    assert data.steps_per_epoch == 4 and data.test_ds is None
    assert data.input_size == (16, 24) and data.input_shape == (16, 24, 3)
    assert data.x_fig.shape == (0, 16, 24, 3) and data.y_fig.shape == (0, 16, 24)
    assert data.num_classes == 3 and data.class_names == ["a", "b", "c"] and data.ignore_index is None
    assert np.all(data.mean == 0) and np.all(data.std == 1)
    assert len(data.image_files) == 12 and len(data.mask_files) == 12
    assert [Path(a).stem for a in data.image_files] == [Path(b).stem for b in data.mask_files]
    assert "test" in data.test_split and data.fingerprint and data.mask_values == (0, 1, 2)


def test_the_cut_is_a_function_of_the_seed_and_disjoint(tmp_path):
    listing = listing_of(tmp_path, count=12)
    first, again, other = load(listing, seed=1), load(listing, seed=1), load(listing, seed=2)
    assert first.image_files == again.image_files and first.image_files != other.image_files
    fit, select = first.image_files[:first.n_train], first.image_files[first.n_train:]
    assert not set(fit) & set(select) and len(set(fit + select)) == 12


def test_two_loads_of_one_seed_give_identical_batches_over_two_epochs(tmp_path):
    listing = listing_of(tmp_path, count=12)
    one, two = batches(load(listing, seed=3)), batches(load(listing, seed=3))
    assert len(one) == len(two) == 8
    for (xa, ya), (xb, yb) in zip(one, two):
        assert np.array_equal(xa, xb) and np.array_equal(ya, yb)
    assert not np.array_equal(one[0][1], one[4][1]) or not np.array_equal(one[1][1], one[5][1]), "epochs reshuffle"
    other = batches(load(listing, seed=4))
    assert any(not np.array_equal(a[1], b[1]) for a, b in zip(one, other))


def test_the_pipelines_know_their_length_and_yield_the_declared_shapes(tmp_path):
    listing = listing_of(tmp_path, count=12)
    data = load(listing, batch_size=2, size=(16, 32))
    assert int(data.train_ds.cardinality()) == data.steps_per_epoch == 4
    assert int(data.select_ds.cardinality()) == data.select_steps == 2
    image, mask = next(iter(data.train_ds))
    assert image.shape == (2, 16, 32, 3) and image.dtype == tf.float32
    assert mask.shape == (2, 16, 32) and mask.dtype == tf.int32
    assert 0.0 <= float(tf.reduce_min(image)) and float(tf.reduce_max(image)) <= 1.0
    total = sum(int(m.shape[0]) for _, m in data.select_ds)
    assert total == data.n_select == 3, "the selection pipeline keeps its remainder"


def test_a_source_of_unknown_length_would_be_refused_by_the_assert():
    unknown = tf.data.Dataset.range(5).filter(lambda value: value >= 0)
    assert int(unknown.cardinality()) == tf.data.UNKNOWN_CARDINALITY
    fixed = sd._batched(unknown, 5, 2, drop_remainder=False)
    assert int(fixed.cardinality()) == 3
    with pytest.raises(tf.errors.FailedPreconditionError):
        list(sd._batched(unknown, 6, 2, drop_remainder=False))


def test_void_stays_void_and_no_value_is_invented_by_the_resize(tmp_path):
    images, masks = tmp_path / "i", tmp_path / "m"
    for index in range(8):
        mask = blocks(48, 64, classes=3, seed=index)
        mask[:12, :8] = 255  # a void block in the corner
        write_pair(images, masks, f"p{index}", mask)
    listing = sd.scan_segmentation_folder(str(images), str(masks))
    for size in ((24, 32), (96, 128), (32, 32)):  # down, up, non-square change
        data = load(listing, ignore_index=255, size=size, batch_size=2, validation_split=0.25)
        seen = set()
        for _, mask in data.train_ds.as_numpy_iterator():
            seen.update(np.unique(mask).tolist())
            assert mask.shape[1:] == size
        assert seen <= {0, 1, 2, 255}, f"a value that is in no mask appeared at {size}: {seen}"
        assert 255 in seen
    assert data.mask_values == (0, 1, 2, 255) and data.ignore_index == 255


def test_a_non_square_size_comes_out_non_square(tmp_path):
    data = load(listing_of(tmp_path, count=12, size=(24, 32)), size=(16, 48))
    image, mask = next(iter(data.select_ds))
    assert image.shape[1:3] == (16, 48) and mask.shape[1:] == (16, 48)


def test_mixed_source_sizes_batch_together(tmp_path):
    images, masks = tmp_path / "i", tmp_path / "m"
    for index, size in enumerate([(16, 24), (40, 30), (24, 24), (8, 64), (50, 20), (12, 12), (30, 30), (64, 16)]):
        write_pair(images, masks, f"p{index}", blocks(*size, seed=index))
    data = load(sd.scan_segmentation_folder(str(images), str(masks)), batch_size=2, size=(16, 16))
    image, mask = next(iter(data.train_ds))
    assert image.shape == (2, 16, 16, 3) and mask.shape == (2, 16, 16)


def test_a_test_split_is_scored_and_kept_for_figures(tmp_path):
    images, masks = make_folder(tmp_path, count=12)
    t_images, t_masks = make_folder(tmp_path, count=5, prefix="t_", size=(40, 40))
    listing = sd.scan_segmentation_folder(str(images), str(masks), str(t_images), str(t_masks))
    data = load(listing, eval_samples=3)
    assert data.n_test == 5 and int(data.test_ds.cardinality()) == 3
    assert data.x_fig.shape == (3, 16, 24, 3) and data.y_fig.shape == (3, 16, 24)
    assert data.test_split == "the test pairs"
    capped = load(listing, max_samples=4, validation_split=0.5)
    assert capped.n_test == 4 and capped.n_train + capped.n_select == 4


def test_a_test_image_is_decoded_in_full_before_training(tmp_path):
    images, masks = make_folder(tmp_path, count=12)
    t_images, t_masks = make_folder(tmp_path, count=3, prefix="t_", image_suffix=".jpg")
    victim = t_images / "p001.jpg"
    victim.write_bytes(victim.read_bytes()[:-30])  # a truncated JPEG: only the full decode sees it
    listing = sd.scan_segmentation_folder(str(images), str(masks), str(t_images), str(t_masks))
    with pytest.raises(DataSourceError, match="cannot be decoded") as caught:
        load(listing)
    assert caught.value.key == "test_images" and str(victim) in str(caught.value)


def test_a_batch_larger_than_the_fit_split_is_refused_in_the_callers_words(tmp_path):
    with pytest.raises(ValueError, match="data.max_samples"):
        load(listing_of(tmp_path, count=6), batch_size=8)


def test_a_seeded_hook_sees_image_and_mask_and_its_result_is_used(tmp_path):
    seen = []

    @seeded
    def hook(seed, image, mask):
        seen.append(seed.shape.rank)
        return image, tf.where(tf.range(tf.shape(mask)[0])[:, None] == 0, 9, mask)  # row 0 becomes 9

    data = load(listing_of(tmp_path, count=12), augment=hook)
    for _, mask in data.train_ds.as_numpy_iterator():
        assert np.all(mask[:, 0, :] == 9)
    assert seen and set(seen) == {1}, "a seeded hook gets a rank-1 seed pair"
    for _, mask in data.select_ds.as_numpy_iterator():
        assert not np.any(mask == 9), "the selection split is never augmented"


def test_a_corrupt_train_image_met_by_fit_names_the_file(tmp_path):
    listing = listing_of(tmp_path, count=8)
    data = load(listing, batch_size=2)
    victim = Path(data.image_files[0])
    victim.write_bytes(b"broken")
    with pytest.raises(tf.errors.InvalidArgumentError) as caught:
        for _ in data.train_ds.take(4):
            pass
        for _ in data.select_ds:
            pass
    assert str(victim) in str(caught.value) or "cannot be decoded" in str(caught.value)


# ---------------------------------------------------------------------
# Oxford-IIIT Pet (the TFDS seam is replaced)
# ---------------------------------------------------------------------

def fake_pet(monkeypatch, train: int = 20, test: int = 7) -> List[Tuple[int, int]]:
    asked = []

    def arrays(size):
        asked.append(tuple(size))
        rng = np.random.default_rng(0)

        def split(n):
            y = rng.integers(0, 3, size=(n, *size)).astype(np.uint8)
            x = np.broadcast_to((y[..., None] * 80).astype(np.uint8), (n, *size, 3)).copy()
            return x, y

        return split(train), split(test)

    monkeypatch.setattr(sd, "load_oxford_pet_arrays", arrays)
    return asked


def test_pet_takes_the_legacy_cut_and_has_three_classes(monkeypatch):
    asked = fake_pet(monkeypatch)
    data = sd.load_pet_data(batch_size=4, seed=5, size=(16, 24), validation_split=0.25, eval_samples=None, max_samples=None)
    assert asked == [(16, 24)]
    assert (data.n_train, data.n_select, data.n_test) == (15, 5, 7)
    assert data.class_names == ["pet", "background", "border"] and data.num_classes == 3 and data.ignore_index is None
    assert data.fingerprint is None and data.image_files is None and data.mask_files is None
    order = np.random.default_rng(5).permutation(20)  # the legacy trainer's cut: first n_val select
    (x_train, y_train), _ = sd.load_oxford_pet_arrays((16, 24))
    selected = np.concatenate(list(data.select_ds.map(lambda i, m: m).as_numpy_iterator()))
    assert np.array_equal(selected, y_train[order[:5]])
    image, mask = next(iter(data.train_ds))
    assert image.shape == (4, 16, 24, 3) and float(tf.reduce_max(image)) <= 1.0 and mask.dtype == tf.int32
    assert data.x_fig.shape == (7, 16, 24, 3) and data.y_fig.shape == (7, 16, 24)


def test_pet_augments_the_fit_split_only(monkeypatch):
    fake_pet(monkeypatch)

    @seeded
    def hook(seed, image, mask):
        return image, tf.zeros_like(mask) + 9

    data = sd.load_pet_data(batch_size=4, seed=0, size=(8, 8), validation_split=0.25, eval_samples=1,
                            max_samples=None, augment=hook)
    assert all(np.all(mask == 9) for _, mask in data.train_ds.as_numpy_iterator())
    for split in (data.select_ds, data.test_ds):
        assert all(not np.any(mask == 9) for _, mask in split.as_numpy_iterator())


def test_the_tfds_seam_shifts_the_labels_and_refuses_a_value_outside_them(monkeypatch):
    import tensorflow_datasets as tfds

    def examples(values):
        mask = np.full((6, 9, 1), 1, np.uint8)
        mask[:3] = values
        return tf.data.Dataset.from_tensor_slices({
            "image": np.full((2, 6, 9, 3), 200, np.uint8), "segmentation_mask": np.stack([mask, mask])})

    monkeypatch.setattr(tfds, "load", lambda name, split, download, shuffle_files: examples(3))
    (x_train, y_train), (x_test, y_test) = sd.load_oxford_pet_arrays((4, 6))
    assert x_train.shape == (2, 4, 6, 3) and y_train.shape == (2, 4, 6) and y_train.dtype == np.uint8
    assert set(np.unique(y_train)) == {0, 2}, "TFDS 1, 2, 3 are the classes 0, 1, 2 (here 1 and 3 occur)"
    assert np.all(x_train == 200)
    monkeypatch.setattr(tfds, "load", lambda name, split, download, shuffle_files: examples(4))
    with pytest.raises(ValueError, match="mask - 1 must lie in"):
        sd.load_oxford_pet_arrays((4, 6))

    def broken(name, split, download, shuffle_files):
        raise FileNotFoundError("no cache")

    monkeypatch.setattr(tfds, "load", broken)
    with pytest.raises(RuntimeError, match="download=False.*nothing is downloaded"):
        sd.load_oxford_pet_arrays((4, 6))


def test_pet_max_samples_caps_the_train_pool_and_the_test_split(monkeypatch):
    fake_pet(monkeypatch)
    data = sd.load_pet_data(batch_size=2, seed=0, size=(8, 8), validation_split=0.5, eval_samples=2, max_samples=6)
    assert (data.n_train, data.n_select, data.n_test) == (3, 3, 6) and data.x_fig.shape[0] == 2


def test_pet_is_deterministic_per_seed(monkeypatch):
    fake_pet(monkeypatch)
    make = lambda seed: sd.load_pet_data(batch_size=4, seed=seed, size=(8, 8), validation_split=0.25, eval_samples=1, max_samples=None)
    one, two, other = batches(make(1)), batches(make(1)), batches(make(2))
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(one, two))
    assert any(not np.array_equal(a[1], b[1]) for a, b in zip(one, other))


def cache_present() -> bool:
    root = Path(os.environ.get("TFDS_DATA_DIR", "~/tensorflow_datasets")).expanduser() / "oxford_iiit_pet"
    return root.is_dir() and any((child / "dataset_info.json").is_file() for child in root.glob("4.*"))


@pytest.mark.skipif(not cache_present(), reason="the local TFDS cache of oxford_iiit_pet 4.x is absent")
def test_the_real_pet_cache_reads_and_matches_the_legacy_arithmetic():
    import tensorflow_datasets as tfds

    dataset = tfds.load(sd.PET_TFDS_NAME, split="train[:4]", download=False, shuffle_files=False)
    for example in tfds.as_numpy(dataset):
        image = tf.constant(example["image"])
        mask = tf.constant(example["segmentation_mask"][..., 0])
        legacy = tf.image.resize(tf.cast(image, tf.float32), (40, 56), method="bilinear", antialias=True)
        legacy = tf.cast(tf.clip_by_value(tf.round(legacy), 0.0, 255.0), tf.uint8)
        legacy_mask = tf.cast(tf.squeeze(tf.image.resize(mask[..., None], (40, 56), method="nearest"), -1), tf.int32) - 1
        ours, our_mask = sd.resize_pair(image, mask, (40, 56))
        assert np.array_equal(np.round(ours.numpy() * 255).astype(np.uint8), legacy.numpy())
        assert np.array_equal(our_mask.numpy() - 1, legacy_mask.numpy()) and set(np.unique(legacy_mask)) <= {0, 1, 2}


# ---------------------------------------------------------------------
# The arguments and the facts known before the claim
# ---------------------------------------------------------------------

def folder_config(images, masks, **extra):
    return sa.load_seg_data_config({"format": "folder", "train_images": str(images), "train_masks": str(masks), "num_classes": 3, **extra})


def test_the_default_format_is_builtin_and_pet_has_no_class_keys():
    assert isinstance(sa.load_seg_data_config({"dataset": "oxford_pet"}), sa.BuiltinSegData)
    with pytest.raises(ConfigValidationError) as caught:
        sa.load_seg_data_config({"dataset": "oxford_pet", "ignore_index": 255})
    assert "ignore_index" in "\n".join(caught.value.errors)
    with pytest.raises(ConfigError, match="not a data format"):
        sa.load_seg_data_config({"format": "coco"})


def test_the_classes_must_be_stated_and_agree(tmp_path):
    base = {"format": "folder", "train_images": "a", "train_masks": "b"}
    with pytest.raises(ConfigValidationError, match="state them"):
        sa.load_seg_data_config(base)
    ok = sa.load_seg_data_config({**base, "class_names": ["x", "y"], "num_classes": 2})
    assert ok.class_names == ["x", "y"]
    for bad, text in (
            ({"class_names": ["x", "y"], "num_classes": 3}, "disagrees"),
            ({"class_names": ["x", "x"]}, "distinct"),
            ({"class_names": ["x", " "]}, "non-empty"),
            ({"num_classes": 1}, "at least 2"),
            ({"num_classes": 3, "ignore_index": 2}, "outside the classes 0 to 2"),
            ({"num_classes": 3, "ignore_index": -1}, "0 or more"),
            ({"num_classes": 3, "validation_split": 1.0}, "strictly between"),
            ({"num_classes": 3, "max_samples": 0}, "positive int"),
            ({"num_classes": 3, "eval_samples": -1}, ">= 0"),
            ({"num_classes": 3, "test_images": "t"}, "go together"),
            ({"num_classes": 3, "mask_threshold": 128}, "reserved"),
            ({"num_classes": 3, "train_images": " "}, "non-empty")):
        with pytest.raises(ConfigValidationError) as caught:
            sa.load_seg_data_config({**base, **bad})
        assert text in "\n".join(caught.value.errors), (bad, caught.value.errors)


def test_a_data_file_anchors_its_relative_paths(tmp_path):
    (tmp_path / "data.yaml").write_text("format: folder\ntrain_images: imgs\ntrain_masks: msk\nnum_classes: 2\n")
    config = sa.load_seg_data_config(str(tmp_path / "data.yaml"))
    assert config.train_images == str(tmp_path / "imgs") and config.train_masks == str(tmp_path / "msk")
    (tmp_path / "bad.yaml").write_text("format: folder\ntrain_images: a\ntrain_masks: b\n")
    with pytest.raises(ConfigValidationError, match=r"data file '.*bad\.yaml'"):
        sa.load_seg_data_config(str(tmp_path / "bad.yaml"))


def test_facts_of_file_data_list_and_sample_with_no_tensor(tmp_path, monkeypatch):
    images, masks = make_folder(tmp_path, count=10)
    config = folder_config(images, masks, ignore_index=255)

    def poisoned(*args, **kwargs):
        raise AssertionError("a tf.data op ran before the claim")

    monkeypatch.setattr(tf.data.Dataset, "from_tensor_slices", staticmethod(poisoned))
    facts = sa.seg_facts(config, (16, 24))
    assert facts == sa.SegFacts("folder", 3, ("0", "1", "2"), 255, (16, 24))
    named = sa.seg_facts(folder_config(images, masks, class_names=["a", "b", "c"]), (16, 24))
    assert named.class_names == ("a", "b", "c")


def test_facts_of_file_data_refuse_with_the_dotted_key(tmp_path):
    images, masks = make_folder(tmp_path, count=10)
    with pytest.raises(ConfigError, match="no native size"):
        sa.seg_facts(folder_config(images, masks), None)
    (masks / "p003.png").unlink()
    with pytest.raises(ConfigError, match=r"Invalid value for 'data\.train_images'.*1 image\(s\) have no mask"):
        sa.seg_facts(folder_config(images, masks), (16, 16))
    write_pair(images, masks, "p003", blocks(24, 32, classes=3, seed=3))
    bad = np.asarray(Image.open(masks / "p005.png")).copy()
    bad[0, 0] = 9
    Image.fromarray(bad).save(masks / "p005.png")
    with pytest.raises(ConfigError, match=r"Invalid value for 'data\.train_masks'.*value 9"):
        sa.seg_facts(folder_config(images, masks), (16, 16))


def test_facts_of_a_csv_name_the_csv_key(tmp_path):
    make_folder(tmp_path, count=4)
    rows = [(f"images/p00{i}.png", f"masks/p00{i}.png") for i in range(3)] + [("images/p003.png", "masks/none.png")]
    config = sa.load_seg_data_config({"format": "csv", "train_csv": str(write_csv(tmp_path / "d.csv", rows)), "num_classes": 3})
    with pytest.raises(ConfigError, match=r"Invalid value for 'data\.train_csv'.*line 5"):
        sa.seg_facts(config, (16, 16))


def test_the_pet_cache_is_looked_for_where_tfds_reads(tmp_path, monkeypatch):
    config = sa.load_seg_data_config({"dataset": "oxford_pet"})
    monkeypatch.setenv("TFDS_DATA_DIR", str(tmp_path))
    with pytest.raises(ConfigError, match="nothing is downloaded"):
        sa.seg_facts(config, (16, 16))
    (tmp_path / "oxford_iiit_pet" / "incomplete.X_4.0.0").mkdir(parents=True)
    with pytest.raises(ConfigError, match="nothing is downloaded"):
        sa.seg_facts(config, (16, 16))
    (tmp_path / "oxford_iiit_pet" / "4.0.0").mkdir()
    with pytest.raises(ConfigError, match="nothing is downloaded"):
        sa.check_seg_sources(config)
    (tmp_path / "oxford_iiit_pet" / "4.0.0" / "dataset_info.json").write_text("{}")
    facts = sa.seg_facts(config, None)
    assert facts == sa.SegFacts("oxford_pet", 3, ("pet", "background", "border"), None, None)


def test_pet_refuses_a_size_whose_arrays_a_graph_cannot_hold(tmp_path, monkeypatch):
    (tmp_path / "oxford_iiit_pet" / "4.0.0").mkdir(parents=True)
    (tmp_path / "oxford_iiit_pet" / "4.0.0" / "dataset_info.json").write_text("{}")
    monkeypatch.setenv("TFDS_DATA_DIR", str(tmp_path))
    config = sa.load_seg_data_config({"dataset": "oxford_pet"})
    assert sa.seg_facts(config, (256, 256)).image_size == (256, 256)
    with pytest.raises(ConfigError, match=r"transform_args\.image_size.*GB"):
        sa.seg_facts(config, (512, 512))


def test_load_seg_data_dispatches_and_wraps_data_errors(tmp_path, monkeypatch):
    images, masks = make_folder(tmp_path, count=12)
    data = sa.load_seg_data(folder_config(images, masks, validation_split=0.25), batch_size=2, seed=0, image_size=(16, 24))
    assert isinstance(data, sd.SegmentationData) and data.input_size == (16, 24)
    bad = np.asarray(Image.open(masks / "p002.png")).copy()
    bad[0, 0] = 9
    Image.fromarray(bad).save(masks / "p002.png")
    with pytest.raises(ConfigError, match=r"Invalid value for 'data\.train_masks'"):
        sa.load_seg_data(folder_config(images, masks, validation_split=0.25), batch_size=2, seed=0, image_size=(16, 24))
    fake_pet(monkeypatch)
    pet = sa.load_seg_data(sa.load_seg_data_config({"dataset": "oxford_pet", "validation_split": 0.25}),
                           batch_size=4, seed=0, image_size=(8, 8))
    assert pet.n_test == 7
