"""The data fingerprint changes whenever the split would (review-iter-1 concerns 1 and 18).

The seeded split of a folder or CSV source is a permutation of the LISTING positions, so the
files in each split follow the listing order. The earlier fingerprint sorted its rows before
hashing, which made it a function of the file SET: a CSV whose rows were reordered kept its
digest, and a resume silently trained on a different fit split (8 of 12 selection files had been
in the killed run's fit split). The rows are now hashed in listing order, in ``listing`` and in
``content`` mode, so:

* the digest is equal exactly when the listing order is equal (a property over permutations,
  checked against the loader's own ``image_files``);
* a reordered CSV is refused at a REAL kill-and-resume under ``normalize=unit`` and under
  ``standardize``, also for a reorder that leaves the fit statistics bit-equal, so the refusal
  does not lean on the statistics;
* a sidecar written before this (no ``data_fingerprint_scheme``) is refused, with no fallback.

Concern 18, the listing edges: a hard link shared by the train and the test split is refused
(file identity, not resolved path), a file name that is not UTF-8 and a symbolic link that leads
nowhere (dangling or looping) in a class folder are refused naming the file, the same way a
corrupt file is.

Tests write only under ``tmp_path``.
"""

import csv
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from train.api.errors import ConfigError
from train.api.tasks.image_classification import ImageClassificationTrainer
from train.common.classification_data import (
    FINGERPRINT_SCHEME,
    DataSourceError,
    listing_fingerprint,
    load_csv_data,
    load_folder_data,
    read_image_csv,
    scan_image_folder,
)
from train.common.resume_state import FIT_STATE_NAME

from .test_resume_interrupted import KillAt, world  # noqa: F401  (fixture used by name)
from .test_task_trainer import make_config

SEED = 3
PER_CLASS = 8
SHARED = dict(batch_size=2, seed=SEED, image_size=8, eval_samples=0, validation_split=0.25)


def make_images(root: Path, per_class: int = PER_CLASS) -> list:
    """Two classes of small random images; return ``[(path, class), ...]`` in class order."""
    generator = np.random.default_rng(0)
    rows = []
    for name in ("a", "b"):
        (root / name).mkdir(parents=True, exist_ok=True)
        for index in range(per_class):
            path = root / name / f"{index:02d}.png"
            Image.fromarray(generator.integers(0, 256, (8, 8, 3), dtype=np.uint8)).save(path)
            rows.append((str(path), name))
    return rows


def write_csv(path: Path, rows) -> Path:
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "label"])
        writer.writerows(rows)
    return path


def permuted(rows, seed: int) -> list:
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    return rows


# ---------------------------------------------------------------------
# Concern 1: the digest follows the listing order
# ---------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["listing", "content"])
@pytest.mark.parametrize("normalize", ["unit", "standardize"])
def test_a_reordered_csv_has_another_fingerprint_and_another_split(tmp_path, mode, normalize):
    rows = make_images(tmp_path / "imgs")
    table = write_csv(tmp_path / "t.csv", rows)
    before = load_csv_data(str(table), normalize=normalize, fingerprint=mode, **SHARED)
    write_csv(table, rows)  # the control: the same rows written again
    same = load_csv_data(str(table), normalize=normalize, fingerprint=mode, **SHARED)
    assert same.fingerprint == before.fingerprint
    write_csv(table, permuted(rows, 7))
    after = load_csv_data(str(table), normalize=normalize, fingerprint=mode, **SHARED)
    select_before = set(before.image_files[before.n_train:])
    assert len(select_before & set(after.image_files[:after.n_train])) > 0, (
        "the control: the reorder really moved files between the fit and the selection split")
    assert after.fingerprint != before.fingerprint


@pytest.mark.parametrize("mode", ["listing", "content"])
def test_the_fingerprint_is_equal_exactly_when_the_loaders_split_is_equal(tmp_path, mode):
    """A property over permutations, against the loader's own ``image_files`` (fit, then selection)."""
    rows = make_images(tmp_path / "imgs")
    table = tmp_path / "t.csv"
    seen = {}
    for number in range(10):
        order = rows if number == 0 else permuted(rows, number)
        write_csv(table, order)
        data = load_csv_data(str(table), normalize="unit", fingerprint=mode, **SHARED)
        seen.setdefault(data.image_files, set()).add(data.fingerprint)
    assert len(seen) == 10, "the ten orders give ten different splits"
    assert all(len(prints) == 1 for prints in seen.values())
    assert len({next(iter(prints)) for prints in seen.values()}) == len(seen)
    # And the same split again gives the same digest.
    write_csv(table, rows)
    again = load_csv_data(str(table), normalize="unit", fingerprint=mode, **SHARED)
    assert {again.fingerprint} == seen[again.image_files]


def test_a_reorder_that_keeps_every_fit_statistic_equal_still_changes_the_fingerprint(tmp_path):
    """Two rows of the FIT split swapped: the fit set, mean and std are equal; the split order is not."""
    rows = make_images(tmp_path / "imgs")
    table = write_csv(tmp_path / "t.csv", rows)
    before = load_csv_data(str(table), normalize="standardize", **SHARED)
    fit = [path for path in before.image_files[:before.n_train]]
    first, second = (next(i for i, row in enumerate(rows) if row[0] == fit[k]) for k in (0, 1))
    swapped = list(rows)
    swapped[first], swapped[second] = swapped[second], swapped[first]
    write_csv(table, swapped)
    after = load_csv_data(str(table), normalize="standardize", **SHARED)
    assert np.array_equal(before.mean, after.mean) and np.array_equal(before.std, after.std), (
        "the control: the statistics did not move, so only the fingerprint can see this")
    assert after.fingerprint != before.fingerprint


def test_a_reordered_test_csv_changes_the_fingerprint(tmp_path):
    rows = make_images(tmp_path / "imgs")
    train = write_csv(tmp_path / "train.csv", rows)
    test_rows = make_images(tmp_path / "held")
    one = listing_fingerprint(read_image_csv(str(train), str(write_csv(tmp_path / "test.csv", test_rows))))
    two = listing_fingerprint(read_image_csv(str(train), str(write_csv(tmp_path / "test.csv", permuted(test_rows, 4)))))
    assert one != two


@pytest.mark.parametrize("mode", ["listing", "content"])
def test_a_folder_fingerprint_is_the_same_for_the_same_tree_made_in_another_order(tmp_path, mode):
    """A folder's listing order is a function of its names, so only the files matter, not how they were made."""
    first = tmp_path / "one"
    make_images(first)
    second = tmp_path / "two"
    for name in ("b", "a"):
        for source in sorted((first / name).iterdir(), reverse=True):
            target = second / name / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    assert (listing_fingerprint(scan_image_folder(str(first)), mode)
            == listing_fingerprint(scan_image_folder(str(second)), mode))


# ---------------------------------------------------------------------
# Concern 1 at a REAL kill and resume
# ---------------------------------------------------------------------


class Killed(ImageClassificationTrainer):
    def extra_callbacks(self, config, data):
        return [KillAt(2)]


def csv_config(out, table, normalize, mode, **changes):
    return make_config(out, **{
        "data": {"format": "csv", "train_csv": str(table), "validation_split": 0.25, "fingerprint": mode},
        "transform_args.image_size": 16, "transform_args.normalize": normalize, "epochs": 4,
        "metric_args.success_threshold": 0.0, **changes})


@pytest.fixture(scope="module", params=[("unit", "listing"), ("unit", "content"), ("standardize", "listing")],
                ids=lambda pair: "-".join(pair))
def killed_csv_run(request, world, tmp_path_factory):
    normalize, mode = request.param
    base = tmp_path_factory.mktemp("csv_run")
    rows = make_images(base / "imgs")
    table = write_csv(base / "t.csv", rows)
    run_dir = base / "run"
    with pytest.raises(RuntimeError, match="killed"):
        Killed(csv_config(run_dir, table, normalize, mode)).run()
    return normalize, mode, rows, table, run_dir


def resume_refusal(rows_now, normalize, mode, rows, table, finished, tmp_path):
    run_dir = tmp_path / "copy"
    shutil.copytree(finished, run_dir)
    write_csv(table, rows_now)
    try:
        with pytest.raises(ConfigError) as caught:
            ImageClassificationTrainer(
                csv_config(run_dir, table, normalize, mode, resume_interrupted=True)).run()
    finally:
        write_csv(table, rows)
    return str(caught.value), run_dir


def test_a_reordered_csv_is_refused_at_a_resume_naming_the_fingerprint(killed_csv_run, tmp_path):
    normalize, mode, rows, table, finished = killed_csv_run
    text, run_dir = resume_refusal(permuted(rows, 7), normalize, mode, rows, table, finished, tmp_path)
    assert "the data is not the data the run started with" in text
    assert "data_fingerprint: " in text, text
    assert not (run_dir / "results_summary.json").exists()


def test_a_swap_inside_the_fit_split_is_refused_by_the_fingerprint_with_the_statistics_unchanged(
        killed_csv_run, tmp_path):
    """No dependence on the statistics: the swapped rows leave the fit mean and std exactly equal."""
    normalize, mode, rows, table, finished = killed_csv_run
    seed = json.loads((finished / "config.json").read_text())["seed"]
    write_csv(table, rows)
    now = load_csv_data(str(table), normalize=normalize, fingerprint=mode, batch_size=4, seed=seed,
                        image_size=16, eval_samples=0, validation_split=0.25)
    fit = now.image_files[:now.n_train]
    first, second = (next(i for i, row in enumerate(rows) if row[0] == fit[k]) for k in (0, 1))
    swapped = list(rows)
    swapped[first], swapped[second] = swapped[second], swapped[first]
    text, _ = resume_refusal(swapped, normalize, mode, rows, table, finished, tmp_path)
    assert "data_fingerprint: " in text, text
    assert "mean" not in text and "std" not in text, "the statistics did not move; only the digest did"


def test_the_unchanged_csv_still_resumes(killed_csv_run, tmp_path):
    normalize, mode, rows, table, finished = killed_csv_run
    run_dir = tmp_path / "copy"
    shutil.copytree(finished, run_dir)
    result = ImageClassificationTrainer(
        csv_config(run_dir, table, normalize, mode, resume_interrupted=True)).run()
    assert (run_dir / "results_summary.json").is_file() and result is not None


def test_the_sidecar_records_the_scheme_and_a_sidecar_without_it_is_refused_with_no_fallback(
        killed_csv_run, tmp_path):
    normalize, mode, rows, table, finished = killed_csv_run
    state = json.loads((finished / "checkpoints" / FIT_STATE_NAME).read_text())
    assert state["data_fingerprint_scheme"] == FINGERPRINT_SCHEME
    run_dir = tmp_path / "copy"
    shutil.copytree(finished, run_dir)
    sidecar = run_dir / "checkpoints" / FIT_STATE_NAME
    del state["data_fingerprint_scheme"]  # what every sidecar written before this change looks like
    state["data_fingerprint"] = "0" * 64  # and its digest is one the loader can no longer compute
    sidecar.write_text(json.dumps(state))
    with pytest.raises(ConfigError) as caught:
        ImageClassificationTrainer(
            csv_config(run_dir, table, normalize, mode, resume_interrupted=True)).run()
    text = str(caught.value)
    assert "data_fingerprint_scheme: none ->" in text and FINGERPRINT_SCHEME in text, text
    assert "choose a new 'out'" in text
    assert "data_fingerprint: " not in text, "the old digest is not compared: there is no fallback"
    assert not (run_dir / "results_summary.json").exists()


# ---------------------------------------------------------------------
# fp_content_no_rows: content mode still sees a path and a class
# ---------------------------------------------------------------------


def two_classes_of_equal_bytes(root: Path) -> None:
    """Every image holds the same bytes, so only the rows (path, class) tell the files apart."""
    for name in ("a", "b"):
        (root / name).mkdir(parents=True)
        for index in range(3):
            Image.new("RGB", (8, 8), (10, 20, 30)).save(root / name / f"{index}.png")


def test_content_mode_sees_a_renamed_file_with_its_bytes_and_size_kept(tmp_path):
    root = tmp_path / "data"
    two_classes_of_equal_bytes(root)
    base = listing_fingerprint(scan_image_folder(str(root)), "content")
    os.rename(root / "a" / "0.png", root / "a" / "9.png")  # same bytes, same size, another path
    assert listing_fingerprint(scan_image_folder(str(root)), "content") != base
    os.rename(root / "a" / "9.png", root / "a" / "0.png")
    assert listing_fingerprint(scan_image_folder(str(root)), "content") == base  # the control


def test_content_mode_sees_the_class_of_a_csv_row_with_the_path_and_bytes_kept(tmp_path):
    root = tmp_path / "data"
    two_classes_of_equal_bytes(root)
    paths = [str(root / "a" / f"{i}.png") for i in range(3)] + [str(root / "b" / f"{i}.png") for i in range(3)]
    one = write_csv(tmp_path / "one.csv", [(p, "x" if i < 3 else "y") for i, p in enumerate(paths)])
    two = write_csv(tmp_path / "two.csv", [(p, "x" if i != 0 and i < 4 else "y") for i, p in enumerate(paths)])
    assert (listing_fingerprint(read_image_csv(str(one)), "content")
            != listing_fingerprint(read_image_csv(str(two)), "content"))


# ---------------------------------------------------------------------
# Concern 18: what a class folder or a pair of splits can hold
# ---------------------------------------------------------------------


def folder_args(**changes):
    return dict(batch_size=2, seed=1, image_size=8, validation_split=0.25, eval_samples=0, normalize="unit",
                **changes)


def test_a_hard_link_shared_by_the_train_and_the_test_split_is_refused(tmp_path):
    train = tmp_path / "train"
    make_images(train)
    test = tmp_path / "test"
    for name in ("a", "b"):
        (test / name).mkdir(parents=True)
        for index in range(2):
            os.link(train / name / f"{index:02d}.png", test / name / f"{index:02d}.png")
    for attempt in (
        lambda: scan_image_folder(str(train), str(test)),
        lambda: load_folder_data(str(train), test_dir=str(test), **folder_args()),
    ):
        with pytest.raises(DataSourceError, match=r"4 of the 4 test file\(s\) are also train files") as caught:
            attempt()
        assert caught.value.key == "test_dir"
        assert "a/00.png" in str(caught.value) and "same file" in str(caught.value)
    os.remove(test / "a" / "00.png")  # a byte COPY of a train file is not the same file: accepted
    shutil.copy(train / "a" / "00.png", test / "a" / "00.png")
    os.remove(test / "b" / "00.png")
    shutil.copy(train / "b" / "00.png", test / "b" / "00.png")
    os.remove(test / "a" / "01.png")
    shutil.copy(train / "a" / "01.png", test / "a" / "01.png")
    os.remove(test / "b" / "01.png")
    shutil.copy(train / "b" / "01.png", test / "b" / "01.png")
    assert scan_image_folder(str(train), str(test)).test_paths is not None


def test_a_filesystem_that_reports_inode_zero_for_every_file_does_not_refuse_the_dataset(tmp_path, monkeypatch):
    """Some network and FUSE mounts give every file inode 0: that is no identity, not "all one file"."""
    from train.common import classification_data as data_module

    train = tmp_path / "train"
    make_images(train)
    test = tmp_path / "test"
    make_images(test, per_class=2)
    real_stat = os.stat

    class NoInode:
        def __init__(self, status):
            self._status = status
            self.st_ino = 0

        def __getattr__(self, name):
            return getattr(self._status, name)

    monkeypatch.setattr(data_module.os, "stat", lambda path, *a, **k: NoInode(real_stat(path, *a, **k)))
    assert scan_image_folder(str(train), str(test)).test_paths is not None


def test_a_hard_link_shared_by_two_csv_files_is_refused(tmp_path):
    train = tmp_path / "train"
    rows = make_images(train)
    test = tmp_path / "test"
    held = []
    for path, name in rows[:2] + rows[PER_CLASS:PER_CLASS + 2]:
        target = test / name / Path(path).name
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(path, target)
        held.append((str(target), name))
    with pytest.raises(DataSourceError, match=r"also train files") as caught:
        read_image_csv(str(write_csv(tmp_path / "train.csv", rows)), str(write_csv(tmp_path / "test.csv", held)))
    assert caught.value.key == "test_csv"


def test_a_file_name_that_is_not_utf8_is_refused_naming_the_file(tmp_path):
    root = tmp_path / "train"
    make_images(root)
    bad = os.fsencode(root / "a") + b"/bad\xff.png"
    shutil.copy(root / "a" / "00.png", bad)
    with pytest.raises(DataSourceError) as caught:
        scan_image_folder(str(root))
    text = str(caught.value)
    assert caught.value.key == "train_dir" and "bad" in text and "UTF-8" in text, text
    assert "\\udcff" in text  # the name is shown in a form that can be read


def test_a_non_utf8_name_is_a_config_error_through_the_api_loader(tmp_path):
    from train.api.config import validate
    from train.api.data_args import load_file_data
    from train.api.tasks.image_classification import ImageClassificationConfig

    root = tmp_path / "train"
    make_images(root)
    shutil.copy(root / "a" / "00.png", os.fsencode(root / "a") + b"/bad\xff.png")
    config = validate(ImageClassificationConfig, {
        "out": str(tmp_path / "run"), "model": "vit/vit_pico", "data": {"format": "folder", "train_dir": str(root)}})
    with pytest.raises(ConfigError, match=r"data\.train_dir.*bad"):
        load_file_data(config.data, batch_size=2, seed=1, augment=None, augment_from_mean=None,
                       normalize="unit", image_size=8)


@pytest.mark.parametrize("kind", ["dangling", "loop"])
def test_a_symbolic_link_that_leads_nowhere_in_a_class_folder_is_refused_naming_it(tmp_path, kind):
    root = tmp_path / "train"
    make_images(root)
    link = root / "a" / "broken.png"
    if kind == "dangling":
        os.symlink(root / "a" / "missing.png", link)
    else:
        os.symlink("broken.png", link)  # a link to itself
    with pytest.raises(DataSourceError) as caught:
        scan_image_folder(str(root))
    text = str(caught.value)
    assert caught.value.key == "train_dir" and "broken.png" in text, text
    assert ("dangling" if kind == "dangling" else "loop") in text, text


def test_a_working_symlink_and_a_dangling_link_that_is_not_an_image_name_are_still_accepted(tmp_path):
    root = tmp_path / "train"
    make_images(root)
    os.symlink(root / "b" / "00.png", root / "a" / "alias_of_b.png")  # a file under two classes: still refused
    with pytest.raises(DataSourceError, match="under two classes"):
        scan_image_folder(str(root))
    os.remove(root / "a" / "alias_of_b.png")
    os.symlink(root / "a" / "nowhere", root / "a" / "README")  # not an image name: not a dataset file
    assert scan_image_folder(str(root)).class_names == ["a", "b"]
