"""``data.fingerprint=content`` sees what ``listing`` cannot: bytes (D-070 and plan step 7).

The ``listing`` fingerprint hashes relative path, class and size, so two same-sized files
swapped with their modification times kept leave it equal. ``content`` hashes the file bytes
too. Pinned here:

* ``listing`` equals the value re-derived in the test from the documented rows IN LISTING ORDER
  (the earlier code sorted them; step 14.2 / D-025 removed the sort so a reordered CSV is another
  digest, see ``test_the_fingerprint_follows_the_split.py``), for a folder and for a CSV;
* an equal-size, equal-mtime byte swap passes ``listing`` and is refused by ``content``;
* ``content`` is a function of relative path, class and bytes (a moved tree keeps it), never
  equals the ``listing`` digest, is streamed in pieces, and logs its time;
* the sidecar records the mode, and a resume refuses a changed digest or a changed mode.

Tests write only under ``tmp_path``.
"""

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest
from PIL import Image

from train.api.errors import ConfigError
from train.api.tasks.image_classification import ImageClassificationTrainer
from train.common import classification_data as data_module
from train.common.classification_data import (
    FINGERPRINT_MODES,
    FINGERPRINT_SCHEME,
    listing_fingerprint,
    load_csv_data,
    load_folder_data,
    read_image_csv,
    scan_image_folder,
)
from train.common.resume_state import FIT_STATE_NAME, read_fit_state

from .test_resume_interrupted import KillAt, world  # noqa: F401  (fixture used by name)
from .test_task_trainer import make_config

SEED = 3
LIGHT, DARK = (200, 40, 30), (40, 200, 90)


def solid(path: Path, rgb) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), rgb).save(path)
    return path


def make_tree(root: Path) -> Path:
    """Two classes of four solid-colour images; the colours of ``a/00`` and ``a/01`` are swappable."""
    for name, base in (("a", 10), ("b", 100)):
        for index in range(4):
            solid(root / name / f"{index:02d}.png", (base + 3 * index, 50 + index, 90))
    solid(root / "a" / "00.png", LIGHT)
    solid(root / "a" / "01.png", DARK)
    return root


def swap_bytes_keeping_mtime(first: Path, second: Path) -> None:
    """Exchange two files' contents; both keep their size and their modification time."""
    assert first.stat().st_size == second.stat().st_size, "the control: the two files have one size"
    times = (first.stat().st_atime_ns, first.stat().st_mtime_ns)
    one, two = first.read_bytes(), second.read_bytes()
    assert one != two
    first.write_bytes(two)
    second.write_bytes(one)
    os.utime(first, ns=times)
    os.utime(second, ns=times)


def reference_listing_digest(listing) -> str:
    """``listing_fingerprint``, written out from its documented rows, in listing order."""
    rows = []
    for split, paths, labels, base in (
        ("train", listing.train_paths, listing.train_labels, listing.train_base),
        ("test", listing.test_paths, listing.test_labels, listing.test_base),
    ):
        if paths is None:
            continue
        for path, label in zip(paths, labels):
            rows.append([split, os.path.relpath(path, base).replace(os.sep, "/"),
                         listing.class_names[int(label)], os.path.getsize(path)])
    digest = hashlib.sha256(listing.source.encode("utf-8"))
    digest.update(b"\x00" + FINGERPRINT_SCHEME.encode("utf-8"))
    for row in rows:
        digest.update(b"\n" + json.dumps(row, ensure_ascii=False).encode("utf-8"))
    return digest.hexdigest()


def csv_of(root: Path, target: Path) -> Path:
    lines = ["path,label"]
    for name in ("a", "b"):
        lines += [f"{p},{name}" for p in sorted((root / name).glob("*.png"))]
    target.write_text("\n".join(lines) + "\n")
    return target


# ---------------------------------------------------------------------
# listing stays what it was
# ---------------------------------------------------------------------


def test_the_listing_value_is_the_documented_rows_in_listing_order_for_a_folder_and_a_csv(tmp_path):
    root = make_tree(tmp_path / "data")
    test_root = make_tree(tmp_path / "test")
    folder = scan_image_folder(str(root), str(test_root))
    assert listing_fingerprint(folder) == listing_fingerprint(folder, "listing") == reference_listing_digest(folder)
    table = read_image_csv(str(csv_of(root, tmp_path / "train.csv")), str(csv_of(test_root, tmp_path / "test.csv")))
    assert listing_fingerprint(table) == reference_listing_digest(table)


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match=r"fingerprint mode must be one of .*'listing'.*'content'"):
        listing_fingerprint(None, "bytes")
    assert FINGERPRINT_MODES == ("listing", "content")


# ---------------------------------------------------------------------
# the swap
# ---------------------------------------------------------------------


def test_an_equal_size_equal_mtime_byte_swap_passes_listing_and_is_refused_by_content(tmp_path):
    root = make_tree(tmp_path / "data")
    before = scan_image_folder(str(root), None)
    listing_before = listing_fingerprint(before)
    content_before = listing_fingerprint(before, "content")
    swap_bytes_keeping_mtime(root / "a" / "00.png", root / "a" / "01.png")
    after = scan_image_folder(str(root), None)
    assert listing_fingerprint(after) == listing_before  # the control: listing is blind to it
    assert listing_fingerprint(after, "content") != content_before


def test_a_swap_in_a_csv_source_is_refused_by_content_too(tmp_path):
    root = make_tree(tmp_path / "data")
    table = csv_of(root, tmp_path / "train.csv")
    before = read_image_csv(str(table), None)
    content_before = listing_fingerprint(before, "content")
    swap_bytes_keeping_mtime(root / "a" / "00.png", root / "a" / "01.png")
    assert listing_fingerprint(read_image_csv(str(table), None)) == listing_fingerprint(before)
    assert listing_fingerprint(read_image_csv(str(table), None), "content") != content_before


def test_content_is_a_function_of_relative_path_class_and_bytes_only(tmp_path):
    first = make_tree(tmp_path / "one" / "data")
    listing = scan_image_folder(str(first), None)
    digest = listing_fingerprint(listing, "content")
    assert digest != listing_fingerprint(listing, "listing")
    # The same bytes under another root and other modification times: the same digest.
    copy = tmp_path / "two" / "elsewhere"
    shutil.copytree(first, copy)
    os.utime(copy / "b" / "02.png", (1, 1))
    assert listing_fingerprint(scan_image_folder(str(copy), None), "content") == digest
    # One byte changed in a file, its size kept: another digest.
    path = copy / "b" / "03.png"
    data = bytearray(path.read_bytes())
    data[-1] ^= 0x01
    path.write_bytes(bytes(data))
    assert listing_fingerprint(scan_image_folder(str(copy), None), "content") != digest


def test_content_is_streamed_in_pieces_and_does_not_depend_on_the_piece_size(tmp_path, monkeypatch):
    root = make_tree(tmp_path / "data")
    listing = scan_image_folder(str(root), None)
    whole = listing_fingerprint(listing, "content")
    reads = []
    original = open

    def counting_open(path, mode="r", *args, **kwargs):
        handle = original(path, mode, *args, **kwargs)
        if mode == "rb":
            real = handle.read

            def read(size=-1):
                reads.append(size)
                return real(size)

            handle.read = read  # type: ignore[method-assign]
        return handle

    monkeypatch.setattr(data_module, "_HASH_CHUNK", 7)
    monkeypatch.setattr("builtins.open", counting_open)
    pieces = listing_fingerprint(listing, "content")
    monkeypatch.undo()
    assert pieces == whole
    assert reads and set(reads) == {7}, "every read asks for the chunk size, never the whole file"
    assert len(reads) > 8 * 2, "a file takes several reads"


def test_content_logs_its_files_megabytes_and_seconds(tmp_path, caplog):
    import logging

    root = make_tree(tmp_path / "data")
    listing = scan_image_folder(str(root), None)
    with caplog.at_level(logging.INFO):
        listing_fingerprint(listing, "content")
    lines = [r.getMessage() for r in caplog.records if "content fingerprint" in r.getMessage()]
    assert len(lines) == 1 and "8 files" in lines[0] and " MB, in " in lines[0] and lines[0].endswith(" s")
    caplog.clear()
    with caplog.at_level(logging.INFO):
        listing_fingerprint(listing)
    assert not [r for r in caplog.records if "fingerprint" in r.getMessage()], "listing logs nothing"


# ---------------------------------------------------------------------
# through the loaders
# ---------------------------------------------------------------------


def test_a_loaded_source_carries_the_digest_of_the_mode_it_was_asked_for(tmp_path):
    root = make_tree(tmp_path / "data")
    table = csv_of(root, tmp_path / "train.csv")
    listing = scan_image_folder(str(root), None)
    shared = dict(batch_size=2, seed=SEED, image_size=16, normalize="unit", eval_samples=0,
                  validation_split=0.25)
    default = load_folder_data(str(root), **shared)
    assert default.fingerprint == listing_fingerprint(listing)
    content = load_folder_data(str(root), fingerprint="content", **shared)
    assert content.fingerprint == listing_fingerprint(listing, "content") != default.fingerprint
    from_csv = load_csv_data(str(table), fingerprint="content", **shared)
    assert from_csv.fingerprint == listing_fingerprint(read_image_csv(str(table), None), "content")


# ---------------------------------------------------------------------
# the sidecar and the resume
# ---------------------------------------------------------------------


class Killed(ImageClassificationTrainer):
    def extra_callbacks(self, config, data):
        return [KillAt(2)]


def run_config(out, root, **changes):
    return make_config(out, **{
        "data": {"format": "folder", "train_dir": str(root), "validation_split": 0.25,
                 "fingerprint": changes.pop("fingerprint", "content")},
        "transform_args.image_size": 16, "transform_args.normalize": "unit", "epochs": 4,
        "metric_args.success_threshold": 0.0, **changes})


@pytest.fixture(scope="module")
def killed_content_run(world, tmp_path_factory):
    base = tmp_path_factory.mktemp("content_run")
    root = make_tree(base / "data")
    run_dir = base / "run"
    with pytest.raises(RuntimeError, match="killed"):
        Killed(run_config(run_dir, root)).run()
    return root, run_dir


def test_the_sidecar_records_the_mode_and_the_content_digest(killed_content_run):
    root, run_dir = killed_content_run
    state = read_fit_state(run_dir / "checkpoints" / FIT_STATE_NAME)
    assert state["data_fingerprint_mode"] == "content"
    assert state["data_fingerprint"] == listing_fingerprint(scan_image_folder(str(root), None), "content")


def test_a_resume_after_an_equal_size_byte_swap_is_refused_naming_the_fingerprint(killed_content_run, tmp_path):
    root, finished = killed_content_run
    run_dir = tmp_path / "copy"
    shutil.copytree(finished, run_dir)
    swap_bytes_keeping_mtime(root / "a" / "00.png", root / "a" / "01.png")
    try:
        with pytest.raises(ConfigError) as caught:
            ImageClassificationTrainer(run_config(run_dir, root, resume_interrupted=True)).run()
    finally:
        swap_bytes_keeping_mtime(root / "a" / "00.png", root / "a" / "01.png")
    text = str(caught.value)
    assert "the data is not the data the run started with" in text
    assert "data_fingerprint: " in text and "its bytes" in text
    assert "data_fingerprint_mode" not in text, "the mode did not change"


def test_a_resume_with_another_mode_in_the_sidecar_is_refused_naming_the_mode(killed_content_run, tmp_path):
    root, finished = killed_content_run
    run_dir = tmp_path / "copy"
    shutil.copytree(finished, run_dir)
    sidecar = run_dir / "checkpoints" / FIT_STATE_NAME
    state = json.loads(sidecar.read_text())
    state["data_fingerprint_mode"] = "listing"
    sidecar.write_text(json.dumps(state))
    with pytest.raises(ConfigError, match=r"data_fingerprint_mode: listing -> content"):
        ImageClassificationTrainer(run_config(run_dir, root, resume_interrupted=True)).run()


def test_a_resume_that_asks_for_another_mode_than_the_run_started_with_is_refused_by_the_request(
        killed_content_run, tmp_path):
    root, finished = killed_content_run
    run_dir = tmp_path / "copy"
    shutil.copytree(finished, run_dir)
    with pytest.raises(ConfigError, match=r"data\.fingerprint: 'content' -> 'listing'"):
        ImageClassificationTrainer(
            run_config(run_dir, root, resume_interrupted=True, fingerprint="listing")).run()


def test_the_content_digest_never_equals_the_listing_digest_even_when_there_is_nothing_to_hash(tmp_path):
    """Mutant ``fp_no_prefix`` (review-iter-1 concern 10, near-equivalent by reading, now measured).

    The docstring promises that the two modes never collide. For any real listing the per-row
    separator that content mode adds already keeps them apart (so files with no bytes differ, shown
    first); only a listing with NO rows would collide without the mode's own prefix, and the
    prefix is what makes the promise unconditional. A listing of no files cannot be scanned (a class
    needs two), so the function is driven on a bare one."""
    from types import SimpleNamespace

    root = tmp_path / "empty"
    for name in ("a", "b"):
        (root / name).mkdir(parents=True)
        for index in range(2):
            (root / name / f"{index}.png").write_bytes(b"")
    listing = scan_image_folder(str(root), None)
    assert listing_fingerprint(listing, "content") != listing_fingerprint(listing, "listing")
    bare = SimpleNamespace(
        source="nothing", class_names=[], train_paths=[], train_labels=[], train_base=str(root),
        test_paths=None, test_labels=None, test_base=None,
    )
    assert listing_fingerprint(bare, "content") != listing_fingerprint(bare, "listing")
