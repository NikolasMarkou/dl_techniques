"""Folder and csv data refuse what leaks or lies, and name the file behind a decode error (D-070).

Review-iter-3 concerns 8, 9, 10 and 19, plus the limit D-069 left open. Every arm builds the
tree it needs under ``tmp_path`` and reads the answer from what the loader or the run did:

* **C8** every test file the run will score is decoded BEFORE ``fit``, and a bad one is a
  ``ConfigError`` naming it (so a corrupt test image cannot cost a finished run its report); a
  corrupt train or selection file met by the statistics pass, the first evaluation or ``fit`` is
  a ``ConfigError`` naming the file; only ``tf.errors.InvalidArgumentError`` is caught, and an
  error that is not about a listed file propagates as it came; nothing is skipped; the normal
  case adds no decode pass over the train files.
* **C9** a file listed twice in a csv, shared between train and test (also ``test_dir`` equal to
  ``train_dir`` and ``test_csv`` equal to ``train_csv``), or under two classes (also through a
  symlink) is refused by RESOLVED path, before the run directory is claimed.
* **C10** two csv labels with one integer value are refused, and the class order is the same
  under two ``PYTHONHASHSEED`` values (child processes).
* **C19** a BOM csv is read; a nested image ignored in a class with top-level images and a class
  absent from the selection split are WARNINGs; a class with one image is refused before the
  claim.
* **Fingerprint** the sha256 over (relative path, class, file size) is recorded in the sidecar of
  a fresh folder or csv run and compared at resume; builtin data carries none. A swap of two
  class folders built so that the class names, the split sizes and the fit statistics all stay
  equal is refused, naming ``data_fingerprint``.

Tests write only under ``tmp_path``.
"""

import csv
import json
import logging
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

import train.common.classification_data as data_module
from train.api.data_args import CsvImageData, FolderImageData, image_facts
from train.api.errors import ConfigError
from train.api.tasks.image_classification import ImageClassificationTrainer
from train.api.trainer import _a_corrupt_file_is_named
from train.common.classification_data import (
    DataSourceError,
    listing_fingerprint,
    load_csv_data,
    load_folder_data,
    read_image_csv,
    scan_image_folder,
    seeded_split_indices,
)
from train.common.resume_state import FIT_STATE_NAME, read_fit_state

from .test_resume_interrupted import KillAt, world, whole  # noqa: F401  (fixtures are used by name)
from .test_task_trainer import make_config

REPO_ROOT = Path(__file__).resolve().parents[3]
SEED = 3


def png(path: Path, rgb=(10, 20, 30), side: int = 8) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (side, side), rgb).save(path)
    return path


def make_classes(root: Path, names=("a", "b"), count: int = 3, first: int = 0) -> Path:
    """``count`` images per class, each its own colour (the statistics must vary)."""
    for number, name in enumerate(names):
        for index in range(first, first + count):
            png(root / name / f"{index:02d}.png", (20 + 9 * index, 40 + 30 * number, 90 + 4 * index))
    return root


def write_rows(path: Path, rows, header=("path", "label"), encoding="utf-8") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding=encoding) as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def rows_of(root: Path, names=("a", "b")):
    return [(str(p), name) for name in names for p in sorted((root / name).glob("*.png"))]


def refusal(build) -> str:
    with pytest.raises(DataSourceError) as caught:
        build()
    return str(caught.value)


# ---------------------------------------------------------------------
# C9: the same file twice, in both splits, or under two classes
# ---------------------------------------------------------------------


def test_a_csv_row_listed_twice_is_refused_even_when_the_path_is_spelled_differently(tmp_path):
    make_classes(tmp_path / "pics")
    rows = rows_of(tmp_path / "pics")
    again = ("pics/./a/../a/00.png", "a")  # the same file as rows[0], relative and unnormalized
    train = write_rows(tmp_path / "train.csv", rows + [again])
    text = refusal(lambda: read_image_csv(str(train)))
    assert "listed more than once" in text and "resolved path" in text and "a/00.png" in text


def test_one_file_under_two_labels_in_a_csv_is_refused(tmp_path):
    make_classes(tmp_path / "pics")
    rows = rows_of(tmp_path / "pics")
    train = write_rows(tmp_path / "train.csv", rows + [(rows[0][0], "b")])
    text = refusal(lambda: read_image_csv(str(train)))
    assert "under two classes" in text and "'a'" in text and "'b'" in text


def test_a_symlinked_class_directory_puts_every_file_under_two_classes(tmp_path):
    root = make_classes(tmp_path / "train")
    os.symlink(root / "a", root / "alias")
    text = refusal(lambda: scan_image_folder(str(root)))
    assert "under two classes" in text and "resolved path" in text


def test_a_file_symlinked_into_another_class_is_refused(tmp_path):
    root = make_classes(tmp_path / "train")
    os.symlink(root / "a" / "00.png", root / "b" / "link.png")
    assert "under two classes" in refusal(lambda: scan_image_folder(str(root)))


def test_a_test_directory_equal_to_the_train_directory_is_refused_naming_the_test_key(tmp_path):
    root = make_classes(tmp_path / "train")
    text = refusal(lambda: scan_image_folder(str(root), str(root)))
    assert "6 of the 6 test file(s) are also train files" in text
    with pytest.raises(ConfigError, match=r"'data.test_dir'.*also train files"):
        image_facts(FolderImageData(train_dir=str(root), test_dir=str(root)), 8)


def test_a_test_csv_equal_to_the_train_csv_is_refused_naming_the_test_key(tmp_path):
    make_classes(tmp_path / "pics")
    train = write_rows(tmp_path / "train.csv", rows_of(tmp_path / "pics"))
    assert "also train files" in refusal(lambda: read_image_csv(str(train), str(train)))
    with pytest.raises(ConfigError, match=r"'data.test_csv'.*also train files"):
        image_facts(CsvImageData(train_csv=str(train), test_csv=str(train)), 8)


def test_one_test_file_that_is_a_symlink_to_a_train_file_is_refused_and_a_disjoint_pair_is_not(tmp_path):
    train = make_classes(tmp_path / "train")
    test = make_classes(tmp_path / "test", first=10)  # other colours, other names: disjoint
    listing = scan_image_folder(str(train), str(test))  # the control: accepted
    assert len(listing.test_paths) == 6
    os.remove(test / "a" / "10.png")
    os.symlink(train / "a" / "00.png", test / "a" / "10.png")
    text = refusal(lambda: scan_image_folder(str(train), str(test)))
    assert "1 of the 6 test file(s) are also train files" in text


@pytest.mark.parametrize("shape", ["duplicate", "shared"])
def test_leakage_is_refused_before_the_run_directory_is_claimed(world, tmp_path, shape):
    make_classes(tmp_path / "pics")
    rows = rows_of(tmp_path / "pics")
    if shape == "duplicate":
        data = {"format": "csv", "train_csv": str(write_rows(tmp_path / "t.csv", rows + rows[:1]))}
        pattern = r"'data.train_csv'.*listed more than once"
    else:
        data = {"format": "folder", "train_dir": str(tmp_path / "pics"), "test_dir": str(tmp_path / "pics")}
        pattern = r"'data.test_dir'.*also train files"
    out = tmp_path / "run"
    with pytest.raises(ConfigError, match=pattern):
        ImageClassificationTrainer(make_config(out, data=data, **{"transform_args.image_size": 16})).run()
    assert not out.exists()


# ---------------------------------------------------------------------
# C10: labels of one integer value, and the class order across hash seeds
# ---------------------------------------------------------------------

HASH_CHILD = textwrap.dedent(
    f"""
    import json
    import sys

    sys.path.insert(0, {str(REPO_ROOT / "src")!r})
    import train.common.classification_data as D

    ints, words, tie = sys.argv[1:4]
    result = {{"ints": D.read_image_csv(ints).class_names, "words": D.read_image_csv(words).class_names}}
    try:
        D.read_image_csv(tie)
        result["tie"] = None
    except D.DataSourceError as exc:
        result["tie"] = str(exc)
    print(json.dumps(result))
    """
)


def test_two_labels_of_one_integer_value_are_refused_and_the_order_survives_two_hash_seeds(tmp_path):
    pics = make_classes(tmp_path / "pics", names=("n", "m", "k", "j"), count=2)
    kinds = {
        "ints": dict(n="10", m="9", k="2", j="-1"),
        "words": dict(n="pear", m="fig", k="apple", j="kiwi"),
        "tie": dict(n="01", m="1", k="2", j="3"),
    }
    paths = {
        kind: str(write_rows(
            tmp_path / f"{kind}.csv",
            [(p, labels[name]) for name in labels for p, _ in rows_of(pics, (name,))]))
        for kind, labels in kinds.items()
    }
    seen = []
    for seed in ("1", "3"):
        # No ``-I``: it would drop PYTHONHASHSEED, the variable under test. The child runs in an
        # empty directory, so nothing in the working directory can shadow a module.
        env = {**os.environ, "PYTHONHASHSEED": seed, "MPLBACKEND": "Agg",
               "CUDA_VISIBLE_DEVICES": "-1", "TF_CPP_MIN_LOG_LEVEL": "3"}
        done = subprocess.run(
            [sys.executable, "-c", HASH_CHILD, paths["ints"], paths["words"], paths["tie"]],
            capture_output=True, text=True, timeout=600, cwd=tmp_path / "pics", env=env)
        assert done.returncode == 0, done.stderr[-2000:]
        seen.append(json.loads(done.stdout.strip().splitlines()[-1]))
    assert seen[0] == seen[1], "the class order or the refusal changed with PYTHONHASHSEED"
    assert seen[0]["ints"] == ["-1", "2", "9", "10"]
    assert seen[0]["words"] == ["apple", "fig", "kiwi", "pear"]
    assert "'01'" in seen[0]["tie"] and "'1'" in seen[0]["tie"] and "same integer" in seen[0]["tie"]


def test_the_same_integer_in_two_spellings_is_refused_naming_the_label_column(tmp_path):
    pics = make_classes(tmp_path / "pics", count=2)
    rows = [(p, "01" if name == "a" else "1") for name in "ab" for p, _ in rows_of(pics, (name,))]
    train = write_rows(tmp_path / "t.csv", rows)
    with pytest.raises(ConfigError, match=r"'data.label_column'.*same integer written differently"):
        image_facts(CsvImageData(train_csv=str(train)), 8)


# ---------------------------------------------------------------------
# C19: a BOM, a nested file, an absent class, one sample per class
# ---------------------------------------------------------------------


def warnings_of(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def test_a_csv_with_a_byte_order_mark_is_read_like_the_same_csv_without_one(tmp_path):
    pics = make_classes(tmp_path / "pics")
    rows = rows_of(pics)
    plain = read_image_csv(str(write_rows(tmp_path / "plain.csv", rows)))
    marked = read_image_csv(str(write_rows(tmp_path / "bom.csv", rows, encoding="utf-8-sig")))
    assert marked.class_names == plain.class_names == ["a", "b"]
    assert marked.train_paths == plain.train_paths
    assert open(tmp_path / "bom.csv", "rb").read(3) == b"\xef\xbb\xbf"  # the control: the mark is there


def test_a_nested_image_in_a_class_with_top_level_images_is_a_warning_with_its_count(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="dl")
    root = make_classes(tmp_path / "train")
    png(root / "a" / "sub" / "deep1.png")
    png(root / "a" / "sub" / "more" / "deep2.png")
    png(root / "a" / ".cache" / "hidden.png")  # a hidden directory is not counted
    listing = scan_image_folder(str(root))
    assert len(listing.train_paths) == 6  # nested files stay ignored
    messages = [m for m in warnings_of(caplog) if "'a'" in m]
    assert len(messages) == 1 and "2 image file(s) in its sub-directories are ignored" in messages[0]
    caplog.clear()
    scan_image_folder(str(make_classes(tmp_path / "clean")))
    assert warnings_of(caplog) == []


def seed_where(n_big, n_tiny, absent):
    """A seed whose cut leaves ``tiny`` out of the selection split (``absent``) or puts it in."""
    for seed in range(200):
        _, select, _ = seeded_split_indices(n_big + n_tiny, 0, seed, 0.1)
        in_selection = bool((np.asarray(select) >= n_big).any())
        if in_selection != absent:
            return seed
    raise AssertionError("no such seed")


def test_a_class_absent_from_the_selection_split_is_a_warning_naming_it(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="dl")
    root = tmp_path / "train"
    make_classes(root, names=("big",), count=38)
    make_classes(root, names=("tiny",), count=2)
    load = lambda seed: load_folder_data(  # noqa: E731
        str(root), batch_size=4, seed=seed, image_size=8, normalize="unit", eval_samples=0)
    load(seed_where(38, 2, absent=True))
    named = [m for m in warnings_of(caplog) if "no image in the selection split" in m]
    assert len(named) == 1 and "['tiny']" in named[0]
    caplog.clear()
    load(seed_where(38, 2, absent=False))  # the control: the class is in selection, nothing to say
    assert not [m for m in warnings_of(caplog) if "selection split" in m]


@pytest.mark.parametrize("source", ["folder", "csv"])
def test_one_image_per_class_is_refused_before_the_run_directory_is_claimed(world, tmp_path, source):
    pics = make_classes(tmp_path / "pics", count=1)
    if source == "folder":
        data = {"format": "folder", "train_dir": str(pics)}
    else:
        data = {"format": "csv", "train_csv": str(write_rows(tmp_path / "t.csv", rows_of(pics)))}
    out = tmp_path / "run"
    with pytest.raises(ConfigError, match=r"every class\s+needs at least 2|have one image each.*at least 2"):
        ImageClassificationTrainer(make_config(out, data=data, **{"transform_args.image_size": 16})).run()
    assert not out.exists()


def test_a_single_small_class_among_larger_ones_is_refused_too(tmp_path):
    root = make_classes(tmp_path / "train", names=("big",), count=5)
    make_classes(root, names=("small",), count=1)
    assert "holds 1 image" in refusal(lambda: scan_image_folder(str(root)))


# ---------------------------------------------------------------------
# C8: a file the decoder cannot read
# ---------------------------------------------------------------------


def corrupt(path) -> None:
    Path(path).write_bytes(b"this is not a png")


def cut_of(root: Path, seed: int = SEED, split: float = 0.1):
    """``(fit files, selection files)`` of the seeded cut the loader makes of ``root``."""
    listing = scan_image_folder(str(root))
    fit, select, _ = seeded_split_indices(len(listing.train_paths), 0, seed, split)
    paths = np.asarray(listing.train_paths)
    return paths[fit].tolist(), paths[select].tolist()


@pytest.fixture
def pair(tmp_path):
    train = make_classes(tmp_path / "train", count=14)
    test = make_classes(tmp_path / "test", count=3, first=40)
    return SimpleNamespace(train=train, test=test, tmp=tmp_path)


def run_pair(pair, name="run", **changes):
    config = make_config(
        pair.tmp / name, seed=SEED, epochs=1, **{
            "data": {"format": "folder", "train_dir": str(pair.train),
                     "test_dir": str(pair.test)},
            "transform_args.image_size": 16, "transform_args.normalize": "unit",
            "metric_args.success_threshold": 0.0, **changes})
    return ImageClassificationTrainer(config).run()


def test_a_corrupt_test_file_is_named_before_fit_and_costs_no_training(world, pair):
    bad = pair.test / "b" / "41.png"
    corrupt(bad)
    with pytest.raises(ConfigError) as caught:
        run_pair(pair)
    text = str(caught.value)
    assert "'data.test_dir'" in text and str(bad) in text and "cannot be decoded" in text
    assert "before training" in text
    run_dir = pair.tmp / "run"
    assert not (run_dir / "training_log.csv").exists(), "fit never began"
    assert not (run_dir / "results_summary.json").exists()


def test_a_corrupt_test_csv_row_is_named_with_the_csv_key(tmp_path):
    train = make_classes(tmp_path / "train", count=14)
    test = make_classes(tmp_path / "test", count=3, first=40)
    corrupt(test / "a" / "42.png")
    with pytest.raises(DataSourceError) as caught:
        load_csv_data(
            str(write_rows(tmp_path / "t.csv", rows_of(train))), test_csv=str(
                write_rows(tmp_path / "e.csv", rows_of(test))),
            batch_size=4, seed=SEED, image_size=8, normalize="unit", eval_samples=0)
    assert caught.value.key == "test_csv" and "42.png" in str(caught.value)


def test_the_normal_case_decodes_only_the_test_files_once_before_fit(pair, monkeypatch):
    seen = []
    real = data_module.find_undecodable
    monkeypatch.setattr(data_module, "find_undecodable", lambda paths: seen.append(list(paths)) or real(paths))
    scanned = []
    monkeypatch.setattr(data_module, "scan_for_undecodable", lambda paths: scanned.append(1))
    load_folder_data(str(pair.train), test_dir=str(pair.test), batch_size=4, seed=SEED,
                     image_size=8, normalize="unit", eval_samples=0)
    assert len(seen) == 1 and sorted(seen[0]) == sorted(scan_image_folder(str(pair.train), str(pair.test)).test_paths)
    assert scanned == [], "the slow scan runs only after a decode error"
    seen.clear()
    load_folder_data(str(pair.train), batch_size=4, seed=SEED, image_size=8, normalize="unit")
    assert seen == [], "a source with no test split decodes nothing before training"


def test_the_decode_check_covers_the_test_files_the_cap_keeps_and_not_the_ones_it_drops(tmp_path):
    train = make_classes(tmp_path / "train", count=14)
    test = make_classes(tmp_path / "test", count=12, first=40)
    listing = scan_image_folder(str(train), str(test))
    _, _, kept = seeded_split_indices(
        len(listing.train_paths), len(listing.test_paths), SEED, 0.1, 20)
    assert len(kept) == 20 and len(listing.test_paths) == 24
    dropped = [p for i, p in enumerate(listing.test_paths) if i not in set(kept.tolist())]
    arguments = dict(batch_size=2, seed=SEED, image_size=8, normalize="unit", eval_samples=0,
                     test_dir=str(test), max_samples=20)
    corrupt(dropped[0])
    load_folder_data(str(train), **arguments)  # never read by this run, so not an error
    corrupt(listing.test_paths[int(kept[0])])
    with pytest.raises(DataSourceError, match="cannot be decoded"):
        load_folder_data(str(train), **arguments)


def test_a_corrupt_fit_file_met_by_the_statistics_pass_is_named(world, pair):
    fit, _ = cut_of(pair.train)
    corrupt(fit[0])
    with pytest.raises(ConfigError) as caught:
        run_pair(pair, **{"transform_args.normalize": "standardize"})
    text = str(caught.value)
    assert "'data.train_dir'" in text and fit[0] in text and "cannot be decoded" in text


def test_a_corrupt_fit_file_met_by_fit_is_named_and_says_it_stopped_fit(world, pair):
    fit, _ = cut_of(pair.train)
    corrupt(fit[0])
    with pytest.raises(ConfigError) as caught:
        run_pair(pair)  # unit normalization: no statistics pass, so fit meets the file
    text = str(caught.value)
    assert fit[0] in text and "it stopped fit" in text and "claimed" in text
    assert isinstance(caught.value.__cause__, tf.errors.InvalidArgumentError)


def test_a_corrupt_selection_file_is_named_by_the_first_evaluation(world, pair):
    _, select = cut_of(pair.train)
    corrupt(select[0])
    with pytest.raises(ConfigError) as caught:
        run_pair(pair)
    text = str(caught.value)
    assert select[0] in text and "it stopped the first evaluation" in text


def test_the_wrapper_names_the_first_undecodable_file_in_the_order_the_data_lists_them(pair):
    files = tuple(scan_image_folder(str(pair.train)).train_paths)
    corrupt(files[5])
    corrupt(files[2])
    with pytest.raises(ConfigError) as caught:
        with _a_corrupt_file_is_named(pair.tmp, files, "fit"):
            tf.io.decode_image(tf.io.read_file(files[5]))  # the error the pipeline raised
    assert files[2] in str(caught.value) and files[5] not in str(caught.value)


def test_an_invalid_argument_error_that_is_not_about_a_listed_file_propagates_unchanged(tmp_path):
    good = png(tmp_path / "ok.png")
    error = tf.errors.InvalidArgumentError(None, None, "a shape mismatch")
    with pytest.raises(tf.errors.InvalidArgumentError, match="a shape mismatch"):
        with _a_corrupt_file_is_named(tmp_path, (str(good),), "fit"):
            raise error
    with pytest.raises(tf.errors.InvalidArgumentError):  # a record with no file list: nothing to scan
        with _a_corrupt_file_is_named(tmp_path, None, "fit"):
            raise error
    # No broad handler: an error of another class is not turned into a decode report, even when
    # a listed file really is corrupt.
    bad = tmp_path / "bad.png"
    corrupt(bad)
    with pytest.raises(KeyError):
        with _a_corrupt_file_is_named(tmp_path, (str(bad),), "fit"):
            raise KeyError("x")


# ---------------------------------------------------------------------
# The fingerprint
# ---------------------------------------------------------------------


def listing_of(root: Path, test: Path = None):
    return scan_image_folder(str(root), None if test is None else str(test))


def test_the_fingerprint_reads_no_pixel_and_makes_no_tensor(pair, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("the fingerprint must not decode or read an image")

    listing = listing_of(pair.train, pair.test)
    monkeypatch.setattr(data_module, "decode_image_file", forbidden)
    monkeypatch.setattr(tf.io, "read_file", forbidden)
    monkeypatch.setattr(tf, "constant", forbidden)
    digest = listing_fingerprint(listing)
    assert len(digest) == 64 and int(digest, 16) >= 0


def test_the_fingerprint_is_a_function_of_paths_classes_and_sizes_only(tmp_path):
    first = make_classes(tmp_path / "one" / "data")
    moved = tmp_path / "two" / "elsewhere"
    moved.parent.mkdir()
    first.rename(moved)
    original = listing_fingerprint(listing_of(moved))
    # The same tree copied under another root and another mtime: the same digest.
    copy = tmp_path / "three" / "other"
    make_classes(copy)
    os.utime(copy / "a" / "00.png", (1, 1))
    assert listing_fingerprint(listing_of(copy)) == original
    edits = {
        "an added file": lambda r: png(r / "a" / "extra.png"),
        "a removed file": lambda r: (r / "b" / "02.png").unlink(),
        "a renamed file": lambda r: (r / "a" / "00.png").rename(r / "a" / "zz.png"),
        "a file of another size": lambda r: png(r / "a" / "00.png", (20, 40, 90), side=64),
    }
    for number, (change, apply) in enumerate(edits.items()):
        edited = tmp_path / f"edited_{number}"
        make_classes(edited)
        apply(edited)
        assert listing_fingerprint(listing_of(edited)) != original, change


def test_the_test_split_is_part_of_the_fingerprint(pair):
    with_test = listing_fingerprint(listing_of(pair.train, pair.test))
    assert with_test != listing_fingerprint(listing_of(pair.train))
    png(pair.test / "a" / "extra.png")
    assert listing_fingerprint(listing_of(pair.train, pair.test)) != with_test


def test_a_loaded_file_source_carries_its_listings_fingerprint_and_the_files_it_reads(pair):
    data = load_folder_data(str(pair.train), test_dir=str(pair.test), batch_size=4, seed=SEED,
                            image_size=8, normalize="unit", eval_samples=0)
    assert data.fingerprint == listing_fingerprint(listing_of(pair.train, pair.test))
    fit, select = cut_of(pair.train)
    assert list(data.image_files) == fit + select  # fit files, then selection files
    assert data.n_train == len(fit) and data.n_select == len(select)


# ---------------------------------------------------------------------
# A swap the statistics cannot see
# ---------------------------------------------------------------------

LIGHT, DARK = (200, 40, 30), (40, 200, 90)


def make_twin_classes(root: Path) -> SimpleNamespace:
    """Two classes whose image ``i`` has the same colour; apple's images are 20 px, zebra's 24 px.

    Swapping the two class folders therefore leaves the class names, the split sizes and
    every per-channel statistic equal (the colour at each listed position is unchanged) while
    every file now has the other class's bytes, and so another size.
    """
    for split, count in (("train", 12), ("test", 4)):
        for name, side in (("apple", 20), ("zebra", 24)):
            for index in range(count):
                png(root / split / name / f"{index:02d}.png", LIGHT if index % 2 == 0 else DARK, side)
    return SimpleNamespace(root=root, train=root / "train", test=root / "test")


def swap_folders(tree: SimpleNamespace) -> None:
    for base in (tree.train, tree.test):
        os.rename(base / "apple", base / "tmp")
        os.rename(base / "zebra", base / "apple")
        os.rename(base / "tmp", base / "zebra")


def stats_of(tree: SimpleNamespace):
    data = load_folder_data(str(tree.train), test_dir=str(tree.test), batch_size=4, seed=SEED,
                            image_size=16, validation_split=0.1, eval_samples=0)
    return data


def test_a_swap_of_the_twin_classes_leaves_every_old_check_blind_and_moves_the_fingerprint(tmp_path):
    tree = make_twin_classes(tmp_path / "twins")
    before = stats_of(tree)
    sizes = {n: (tree.train / n / "00.png").stat().st_size for n in ("apple", "zebra")}
    assert sizes["apple"] != sizes["zebra"], "the control: the two classes' files differ in size"
    swap_folders(tree)
    after = stats_of(tree)
    # Everything D-069's comparison looks at is equal ...
    assert after.class_names == before.class_names
    assert (after.n_train, after.n_select, after.n_test) == (before.n_train, before.n_select, before.n_test)
    np.testing.assert_allclose(after.mean, before.mean, rtol=1e-5)
    np.testing.assert_allclose(after.std, before.std, rtol=1e-5)
    # ... and the fingerprint is not.
    assert after.fingerprint != before.fingerprint


def test_a_swap_of_two_labels_in_a_csv_leaves_the_statistics_equal_and_moves_the_fingerprint(tmp_path):
    tree = make_twin_classes(tmp_path / "twins")
    rows = rows_of(tree.train, ("apple", "zebra"))
    swapped = [(path, "zebra" if name == "apple" else "apple") for path, name in rows]
    loaded = [
        load_csv_data(str(write_rows(tmp_path / f"{n}.csv", r)), batch_size=4, seed=SEED,
                      image_size=16, eval_samples=0)
        for n, r in (("kept", rows), ("swapped", swapped))]
    kept, other = loaded
    np.testing.assert_allclose(other.mean, kept.mean, rtol=1e-5)
    assert (other.n_train, other.n_select) == (kept.n_train, kept.n_select)
    assert other.fingerprint != kept.fingerprint


class TwinKilled(ImageClassificationTrainer):
    def extra_callbacks(self, config, data):
        return [KillAt(2)]


def twin_config(out, tree, **changes):
    return make_config(out, **{
        "data": {"format": "folder", "train_dir": str(tree.train), "test_dir": str(tree.test)},
        "transform_args.image_size": 16, "epochs": 4, "metric_args.success_threshold": 0.0,
        **changes})


@pytest.fixture(scope="module")
def twin_run(world, tmp_path_factory):
    root = tmp_path_factory.mktemp("twin_run")
    tree = make_twin_classes(root / "data")
    run_dir = root / "run"
    with pytest.raises(RuntimeError, match="killed"):
        TwinKilled(twin_config(run_dir, tree)).run()
    return tree, run_dir


def test_the_sidecar_of_a_fresh_file_run_holds_the_fingerprint_and_a_builtin_run_holds_none(twin_run, whole):
    tree, run_dir = twin_run
    state = read_fit_state(run_dir / "checkpoints" / FIT_STATE_NAME)
    assert state["data_fingerprint"] == listing_fingerprint(listing_of(tree.train, tree.test))
    _, builtin_dir = whole
    assert "data_fingerprint" not in read_fit_state(builtin_dir / "checkpoints" / FIT_STATE_NAME)


def test_a_resume_after_a_swap_the_statistics_cannot_see_is_refused_naming_the_fingerprint(twin_run, tmp_path):
    import shutil

    tree, folder_dir = twin_run
    run_dir = tmp_path / "copy"
    shutil.copytree(folder_dir, run_dir)
    swap_folders(tree)
    try:
        with pytest.raises(ConfigError) as caught:
            ImageClassificationTrainer(twin_config(run_dir, tree, resume_interrupted=True)).run()
    finally:
        swap_folders(tree)
    text = str(caught.value)
    assert "the data is not the data the run started with" in text
    assert "data_fingerprint: " in text and "->" in text
    # The old comparisons saw nothing, so no other line is there.
    for old in ("mean: ", "std: ", "n_train: ", "n_select: ", "n_test: ", "class_names"):
        assert old not in text, old


def test_the_comparison_treats_a_missing_fingerprint_on_either_side_as_a_change(twin_run, tmp_path):
    import shutil

    tree, folder_dir = twin_run
    run_dir = tmp_path / "copy"
    shutil.copytree(folder_dir, run_dir)
    sidecar = run_dir / "checkpoints" / FIT_STATE_NAME
    state = json.loads(sidecar.read_text())
    del state["data_fingerprint"]  # a sidecar written before the fingerprint existed
    sidecar.write_text(json.dumps(state))
    with pytest.raises(ConfigError, match=r"data_fingerprint: none -> [0-9a-f]{12}"):
        ImageClassificationTrainer(twin_config(run_dir, tree, resume_interrupted=True)).run()
