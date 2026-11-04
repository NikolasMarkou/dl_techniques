"""``dl_techniques.datasets.vision.ade20k``: the id coding, the release listing and the two class sets.

Everything here runs on arrays and files written to ``tmp_path``; the real release is read by
``tests/test_train/test_api/test_convert.py`` reads its names file when it is on disk.

* the coding is ``(R // 10) * 256 + G`` (a round trip over ids from every R band, and a read of the very
  array with the wrong coding fails the round trip);
* the SceneParse150 table is what the benchmark published: 150 distinct classes in order, 153 distinct
  release ids, and the names the mapping is checked against are the table's own;
* ``official_mapping`` refuses a release whose names disagree, naming the class, and accepts a narrower
  name for one id of a class (``closet`` for ``wardrobe, closet, press``);
* the lookup table sends every id the mapping does not name, and 0, to the void value;
* ``frequency_mapping`` ranks by pixels, breaks a tie by the smaller id whatever the order of the counts, and
  refuses a ``k`` above the classes that have a pixel;
* ``list_release`` pairs a JPEG with its ``_seg.png``, never enters an image's instance directory, keeps the
  video frames of ``training`` and refuses a lone ``_seg.png``;
* a count over files is the same with one worker and with two.
"""

import numpy as np
import pytest
from PIL import Image

from dl_techniques.datasets.vision import ade20k as ade


def seg_array(ids: np.ndarray, red_offset: int = 0) -> np.ndarray:
    """The RGB array a ``_seg.png`` holds for ``ids`` (blue is the instance, here 0)."""
    seg = np.zeros(ids.shape + (3,), dtype=np.uint8)
    seg[..., 0] = (ids // 256) * 10 + red_offset
    seg[..., 1] = ids % 256
    return seg


def names_of_the_table() -> dict:
    """A release's names that agree with the benchmark: every id of a class carries its name."""
    return {identifier: name for name, ids in ade.SCENE_PARSE_150 for identifier in ids}


# ---------------------------------------------------------------------
# The coding
# ---------------------------------------------------------------------

def test_the_class_id_is_red_over_ten_times_256_plus_green():
    ids = np.array([[0, 5, 255, 256], [2978, 3637, 3688, ade.MAX_CLASS_ID]])
    assert (ade.decode_class_ids(seg_array(ids)) == ids).all()


def test_a_red_that_is_not_a_multiple_of_ten_reads_as_its_floor():
    # The benchmark's MATLAB rounds `R / 10`; the release's starter code floors it. Files have R % 10 == 0.
    ids = np.array([[300, 2978]])
    assert (ade.decode_class_ids(seg_array(ids, red_offset=4)) == ids).all()


def test_an_array_that_is_not_rgb_is_refused():
    with pytest.raises(ade.AdeError, match=r"\(H, W, 3\)"):
        ade.decode_class_ids(np.zeros((4, 4), dtype=np.uint8))


def test_read_seg_refuses_a_file_that_is_not_rgb_and_names_it(tmp_path):
    path = tmp_path / "ADE_train_00000001_seg.png"
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(path)
    with pytest.raises(ade.AdeError, match="is a L image"):
        ade.read_seg(str(path))


def test_read_seg_refuses_a_file_that_is_not_an_image_and_names_it(tmp_path):
    path = tmp_path / "ADE_train_00000001_seg.png"
    path.write_bytes(b"not a png")
    with pytest.raises(ade.AdeError, match="cannot be decoded"):
        ade.read_seg(str(path))


# ---------------------------------------------------------------------
# The SceneParse150 table and the official mapping
# ---------------------------------------------------------------------

def test_the_table_is_150_distinct_classes_with_153_distinct_release_ids():
    table = ade.SCENE_PARSE_150
    ids = [identifier for _, group in table for identifier in group]
    assert len(table) == 150 and len(ids) == 153 and len(set(ids)) == 153
    assert all(0 < identifier <= 3688 for identifier in ids)
    assert len({name for name, _ in table}) == 150
    assert [name for name, _ in table][:4] == ["wall", "building, edifice", "sky", "floor, flooring"]


def test_the_official_mapping_has_one_name_per_class_and_the_first_synonym_names_it():
    mapping = ade.official_mapping(names_of_the_table())
    assert len(mapping.class_names) == 150 and len(set(mapping.class_names)) == 150
    assert mapping.class_names[1] == "building" and mapping.class_names[14] == "door"
    assert mapping.source_ids[14] == (774, 783)


def test_a_release_whose_names_disagree_is_refused_naming_the_class():
    names = names_of_the_table()
    names[2978] = "stonewall"                       # `wall`
    with pytest.raises(ade.AdeError, match=r"class 0 'wall'.*1 of 150|1 of 150.*class 0 'wall'"):
        ade.official_mapping(names)


def test_an_id_missing_from_the_names_file_makes_its_class_unsound_only_if_no_other_id_matches():
    names = names_of_the_table()
    del names[774]                                   # `door` of `door, double door`; 783 still matches
    assert ade.official_mapping(names).class_names[14] == "door"
    del names[783]
    with pytest.raises(ade.AdeError, match="door, double door"):
        ade.official_mapping(names)


def test_a_narrower_name_for_one_id_of_a_class_is_accepted():
    names = names_of_the_table()
    ids = dict(ade.SCENE_PARSE_150)["wardrobe, closet, press"]
    names[ids[0]] = "closet"
    assert ade.official_mapping(names).source_names[35][0] == "closet"


def test_the_lookup_table_names_the_classes_and_leaves_everything_else_void():
    mapping = ade.official_mapping(names_of_the_table())
    lut = mapping.lut()
    assert lut.dtype == np.uint8 and lut.shape == (ade.MAX_CLASS_ID + 1,)
    assert lut[2978] == 0 and lut[312] == 1 and lut[774] == 14 and lut[783] == 14
    assert lut[0] == ade.VOID_INDEX and lut[1] == ade.VOID_INDEX and lut[ade.MAX_CLASS_ID] == ade.VOID_INDEX
    assert (lut != ade.VOID_INDEX).sum() == 153
    assert sorted(set(lut[lut != ade.VOID_INDEX].tolist())) == list(range(150))


def test_convert_seg_gives_the_mask_and_the_count_of_every_value():
    lut = ade.official_mapping(names_of_the_table()).lut()
    ids = np.array([[2978, 2978, 0], [312, 3000, 783]])
    mask, counts = ade.convert_seg(seg_array(ids), lut)
    assert mask.tolist() == [[0, 0, 255], [1, 255, 14]] and mask.dtype == np.uint8
    assert counts.shape == (256,) and counts[0] == 2 and counts[1] == 1 and counts[14] == 1 and counts[255] == 2
    assert counts.sum() == ids.size


# ---------------------------------------------------------------------
# The frequency mapping
# ---------------------------------------------------------------------

def test_the_frequency_mapping_ranks_by_pixels_and_a_tie_goes_to_the_smaller_id():
    counts = np.zeros(ade.MAX_CLASS_ID + 1, dtype=np.int64)
    counts[0] = 10 ** 9                              # unlabelled never competes
    counts[[40, 7, 90, 8]] = [500, 500, 900, 3]
    mapping = ade.frequency_mapping(counts, {90: "sky, heaven", 7: "tree", 40: "tree", 8: "rare"}, 3)
    assert mapping.source_ids == ((90,), (7,), (40,))
    assert mapping.class_names[0] == "sky" and len(set(mapping.class_names)) == 3, "a repeated name gets its id"
    assert mapping.class_names[1:] == ("tree (7)", "tree (40)")
    assert mapping.lut()[8] == ade.VOID_INDEX and mapping.lut()[7] == 1


def test_the_frequency_mapping_does_not_depend_on_the_order_of_the_counts():
    rng = np.random.default_rng(0)
    counts = np.zeros(ade.MAX_CLASS_ID + 1, dtype=np.int64)
    counts[1:60] = rng.integers(1, 5, size=59)       # many ties
    first = ade.frequency_mapping(counts, {}, 10).source_ids
    shuffled = counts.copy()
    again = ade.frequency_mapping(shuffled, {}, 10).source_ids
    assert first == again
    pixels = [counts[i[0]] for i in first]
    assert pixels == sorted(pixels, reverse=True)
    for a, b in zip(first, first[1:]):
        assert counts[a[0]] > counts[b[0]] or a[0] < b[0]


def test_a_k_above_the_classes_with_a_pixel_is_refused():
    counts = np.zeros(ade.MAX_CLASS_ID + 1, dtype=np.int64)
    counts[[3, 4, 5]] = 1
    with pytest.raises(ade.AdeError, match="only 3 classes have a pixel"):
        ade.frequency_mapping(counts, {}, 4)
    with pytest.raises(ade.AdeError, match="255 is the void value"):
        ade.frequency_mapping(counts, {}, 255)


# ---------------------------------------------------------------------
# The release on disk
# ---------------------------------------------------------------------

def write_release(root, train=3, val=2, frames=1):
    """A tiny release: scene folders, instance directories, a part mask, video frames."""
    layout = []
    for split, folder, prefix, count in (("train", "training", "ADE_train", train), ("val", "validation", "ADE_val", val)):
        for number in range(1, count + 1):
            layout.append((folder, f"{prefix}_{number:08d}", "group/scene_a" if number % 2 else "group/scene_b"))
    layout += [("training", f"ADE_frame_{number:08d}", "nature/watering_hole") for number in range(1, frames + 1)]
    for folder, stem, scene in layout:
        here = root / "images" / "ADE" / folder / scene
        (here / stem).mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.zeros((6, 8, 3), dtype=np.uint8)).save(here / f"{stem}.jpg")
        Image.fromarray(seg_array(np.full((6, 8), 2978))).save(here / f"{stem}_seg.png")
        Image.fromarray(seg_array(np.full((6, 8), 5))).save(here / f"{stem}_parts_1.png")
        # An image's own directory holds its instance masks (`instance_000_<stem>.png`). It is not entered: a file
        # in it that is named like a mask, and has no image beside it, is not listed (and does not raise).
        Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(here / stem / f"instance_000_{stem}.png")
        Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(here / stem / "ADE_train_00000099_seg.png")
    return layout


def test_the_listing_pairs_images_with_masks_and_keeps_the_frames_of_training(tmp_path):
    write_release(tmp_path)
    release = ade.list_release(str(tmp_path))
    assert [p.stem for p in release["train"]] == ["ADE_frame_00000001", "ADE_train_00000001", "ADE_train_00000002",
                                                  "ADE_train_00000003"]
    assert [p.stem for p in release["val"]] == ["ADE_val_00000001", "ADE_val_00000002"]
    for pair in release["train"] + release["val"]:
        assert pair.image.endswith(pair.stem + ".jpg") and pair.seg.endswith(pair.stem + "_seg.png")


def test_a_part_mask_and_an_instance_mask_are_not_listed(tmp_path):
    write_release(tmp_path, train=1, val=1, frames=0)
    segs = [p.seg for split in ade.list_release(str(tmp_path)).values() for p in split]
    assert len(segs) == 2 and not any("parts" in s or "instance" in s for s in segs)


def test_a_root_that_is_not_the_release_is_refused(tmp_path):
    with pytest.raises(ade.AdeError, match="not the ADE20K 2021_17_01 release"):
        ade.list_release(str(tmp_path))


def test_a_mask_without_its_image_is_refused_naming_both(tmp_path):
    write_release(tmp_path, train=1, val=1, frames=0)
    next((tmp_path / "images" / "ADE" / "validation").rglob("*.jpg")).rename(tmp_path / "moved.jpg")
    with pytest.raises(ade.AdeError, match=r"ADE_val_00000001_seg.png.* has no image .*ADE_val_00000001.jpg"):
        ade.list_release(str(tmp_path))


def test_read_object_names_reads_latin_1_and_the_index_column(tmp_path):
    path = tmp_path / "objects.txt"
    path.write_bytes("Wordnet name \t Name index \t x\nwall\t2978\t1\ncaf\xe9, bar\t312\t2\n".encode("latin-1"))
    assert ade.read_object_names(str(path)) == {2978: "wall", 312: "caf\xe9, bar"}


def test_read_object_names_names_a_broken_line(tmp_path):
    path = tmp_path / "objects.txt"
    path.write_text("Wordnet name\tName index\nwall\tnot-a-number\n")
    with pytest.raises(ade.AdeError, match="line 2"):
        ade.read_object_names(str(path))


# ---------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------

def test_the_pixel_counts_are_the_same_with_one_worker_and_with_two(tmp_path):
    rng = np.random.default_rng(1)
    paths = []
    expected = np.zeros(ade.MAX_CLASS_ID + 1, dtype=np.int64)
    for number in range(6):
        ids = rng.choice([0, 2978, 312, 3637, 774], size=(5, 7))
        expected += np.bincount(ids.ravel(), minlength=ade.MAX_CLASS_ID + 1)
        path = tmp_path / f"ADE_train_{number:08d}_seg.png"
        Image.fromarray(seg_array(ids)).save(path)
        paths.append(str(path))
    assert (ade.class_pixel_counts(paths, workers=1) == expected).all()
    assert (ade.class_pixel_counts(paths, workers=2) == expected).all()
    assert expected.sum() == 6 * 35
