"""CC3M dedupe removes the captions the old resume duplicated, and nothing else (D-196).

Before D-187 a resumed extraction appended a partly extracted shard's captions a
second time under new ids, and an image file that already existed at a new id was
kept even when it held another pair. The real train JSONL has two such replayed
blocks (925 and 4,580 lines). The fixture below is a small extraction with the same
damage: a replayed block whose first image is a stale file. It also holds a pair the
source repeats (same caption, same image bytes, not in a replay) and a caption that
repeats with another image; a clean extraction has both, so both stay. Everything is
written under ``tmp_path``.
"""

import hashlib
import json
from io import BytesIO

import pytest
from PIL import Image

from train.cliffordnet import prepare_cc3m as module


def _jpeg(index):
    buffer = BytesIO()
    Image.new("RGB", (4, 4), (index * 7 % 256, 255 - index * 5 % 256, index * 11 % 256)).save(
        buffer, "JPEG", quality=95)
    return buffer.getvalue()


#: (caption, image) per line, in id order. Lines 2-10 are a partly extracted shard;
#: lines 11-19 replay it (line 11's file is stale: another pair's image), and lines
#: 20-21 are the rest of that shard. Line 22 repeats line 1 exactly (a source repeat).
#: Line 23 repeats line 0's caption with another image. Lines 24-25 repeat the captions
#: of lines 0-1 in order with other images: two common captions next to each other,
#: which the real data has 325 times; they stay.
_LINES = (
    [("actor attends the premiere", 100), ("a cup", 101)]
    + [(f"pair {n}", n) for n in range(2, 11)]
    + [("pair 2", 999)] + [(f"pair {n}", n) for n in range(3, 11)]
    + [("pair 11", 11), ("pair 12", 12)]
    + [("a cup", 101), ("actor attends the premiere", 200)]
    + [("actor attends the premiere", 300), ("a cup", 301)]
)
_DUPLICATES = list(range(11, 20))


def _image_path(root, split, image_id):
    return root / split / module._shard_of(image_id) / f"{image_id}.jpg"


def _write_dataset(root):
    root.mkdir()
    for split, lines in (("train", _LINES), ("validation", [("a view", 300)])):
        with open(root / f"{split}_captions.jsonl", "w") as handle:
            for n, (caption, image) in enumerate(lines):
                image_id = f"cc3m_{split}_{n:08d}"
                handle.write(json.dumps({"id": image_id, "split": split, "caption": caption},
                                        ensure_ascii=False) + "\n")
                path = _image_path(root, split, image_id)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(_jpeg(image))
    (root / module._STATE_FILENAME).write_text(json.dumps({"processed_tars": {}}))


def _tree(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_the_audit_removes_the_replayed_block_and_keeps_the_source_repeat(tmp_path):
    root = tmp_path / "cc3m"
    _write_dataset(root)
    report = module.audit_duplicates(str(root), "train")
    assert report.lines == len(_LINES)
    assert report.remove == report.replayed == _DUPLICATES
    assert report.content == list(range(12, 20)) + [22]
    assert report.source_repeats == [22]
    assert module.audit_duplicates(str(root), "validation").remove == []


def test_a_dry_run_changes_nothing(tmp_path):
    root = tmp_path / "cc3m"
    _write_dataset(root)
    before = _tree(root)
    reports = module.dedupe(str(root), ["train", "validation"], apply=False)
    assert reports["train"].remove == _DUPLICATES
    assert _tree(root) == before
    assert [p.name for p in tmp_path.iterdir()] == ["cc3m"]


def test_apply_removes_only_the_duplicates_and_backs_up_what_it_changes(tmp_path):
    root = tmp_path / "cc3m"
    _write_dataset(root)
    before = _tree(root)
    original_jsonl = (root / "train_captions.jsonl").read_bytes()

    reports = module.dedupe(str(root), ["train", "validation"], apply=True,
                            backup_dir=str(tmp_path / "backup"))
    backup = tmp_path / "backup"

    kept = [json.loads(line) for line in (root / "train_captions.jsonl").read_text().splitlines()]
    expected = [n for n in range(len(_LINES)) if n not in _DUPLICATES]
    assert [line["id"] for line in kept] == [f"cc3m_train_{n:08d}" for n in expected]
    assert [line["caption"] for line in kept] == [_LINES[n][0] for n in expected]
    assert len(kept) == len(_LINES) - len(_DUPLICATES)
    for line in kept:  # every kept pair still has its own image
        n = int(line["id"][-8:])
        assert _image_path(root, "train", line["id"]).read_bytes() == _jpeg(_LINES[n][1])

    removed_ids = [f"cc3m_train_{n:08d}" for n in _DUPLICATES]
    for image_id in removed_ids:
        assert not _image_path(root, "train", image_id).exists()
    manifest = (backup / "removed_images.txt").read_text().splitlines()
    assert manifest == [str(_image_path(root, "train", i)) for i in removed_ids]
    for image_id in removed_ids:  # moved into the backup, not destroyed
        rel = _image_path(root, "train", image_id).relative_to(root)
        moved = (backup / "images" / rel).read_bytes()
        assert hashlib.sha256(moved).hexdigest() == before[str(rel)]

    assert (backup / "train_captions.jsonl").read_bytes() == original_jsonl
    sums = dict(line.split("  ")[::-1] for line in (backup / "SHA256SUMS").read_text().splitlines())
    assert sums["train_captions.jsonl"] == hashlib.sha256(original_jsonl).hexdigest()
    assert "validation_captions.jsonl" not in sums  # nothing to remove, not rewritten
    after = _tree(root)
    assert after["validation_captions.jsonl"] == before["validation_captions.jsonl"]
    assert after[module._STATE_FILENAME] == before[module._STATE_FILENAME]

    assert reports["train"].remove == _DUPLICATES
    assert module.audit_duplicates(str(root), "train").remove == []


def test_a_resume_after_dedupe_takes_ids_after_the_last_one(tmp_path):
    root = tmp_path / "cc3m"
    _write_dataset(root)
    module.dedupe(str(root), ["train"], apply=True, backup_dir=str(tmp_path / "backup"))
    assert module._next_id(str(root / "train_captions.jsonl")) == len(_LINES)


def test_apply_refuses_an_existing_backup_dir(tmp_path):
    root = tmp_path / "cc3m"
    _write_dataset(root)
    (tmp_path / "backup").mkdir()
    before = _tree(root)
    with pytest.raises(FileExistsError):
        module.dedupe(str(root), ["train"], apply=True, backup_dir=str(tmp_path / "backup"))
    assert _tree(root) == before
