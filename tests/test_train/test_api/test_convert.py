"""``python -m train convert ade20k``: index masks, listings and a data file, resumably, outside the source tree.

Plan 98b866fb step 5 (D-005, D-017). Every case builds a tiny ADE20K-shaped release in ``tmp_path`` (RGB
``_seg.png`` files coded ``(R // 10) * 256 + G``, JPEGs, an ``objects.txt``) and runs the real command through
``train.api.cli.main``:

* the output is complete: one 8-bit mask per image with the class INDEX (void 255), ``train.csv`` and
  ``val.csv`` that name the ORIGINAL JPEG, ``mapping.json`` with pixels and coverage worked out here
  independently, and a ``data.yaml`` the segmentation task reads (``seg_facts`` accepts it);
* one worker and two give the same bytes;
* a rerun is cheap and exact: a finished mask is not rewritten (inode and mtime), a planted partial output
  (a torn manifest line, a missing mask, a stale ``.tmp``) is completed, and nothing is deleted;
* a changed ``root``, ``mapping`` or ``k`` is refused naming the key; an ``out`` inside ``root`` or around it,
  a foreign non-empty directory and a file are refused with nothing created;
* an image whose JPEG and mask differ in size, a JPEG with an EXIF orientation and an undecodable mask are
  EXCLUDED and listed, not converted and not in a CSV;
* ``mapping=frequency`` picks the most frequent train classes, counts once and reuses the counts, and refuses
  ``k`` with ``official`` and a ``k`` the release cannot give;
* ``convert help`` and ``convert ade20k help`` exit 0 and load no TensorFlow.
"""

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from dl_techniques.datasets.vision import ade20k as ade
from train.api import cli
from train.api import segmentation_args as sa

REAL_RELEASE = Path("/media/arxwn/data0_4tb/datasets/ade20k")
SIZE = (6, 8)                                       # height, width
#: Ids a tiny release draws: three official classes (wall, building, sky), one the benchmark leaves out, unlabelled.
IDS = [2978, 312, 2420, 3000, 0]


def seg_array(ids: np.ndarray) -> np.ndarray:
    seg = np.zeros(ids.shape + (3,), dtype=np.uint8)
    seg[..., 0] = (ids // 256) * 10
    seg[..., 1] = ids % 256
    return seg


def write_objects(root: Path) -> None:
    rows = ["Wordnet name \t Name index \t is object counts"]
    rows += [f"{name}\t{identifier}\t1" for name, ids in ade.SCENE_PARSE_150 for identifier in ids]
    rows += ["thing the benchmark leaves out\t3000\t1"]
    (root / "objects.txt").write_text("\n".join(rows) + "\n")


def write_release(root: Path, train: int = 4, val: int = 2) -> dict:
    """A tiny release; return ``{stem: ids array}`` of what each ``_seg.png`` holds."""
    rng = np.random.default_rng(3)
    write_objects(root)
    held = {}
    for folder, prefix, count in (("training", "ADE_train", train), ("validation", "ADE_val", val)):
        for number in range(1, count + 1):
            stem = f"{prefix}_{number:08d}"
            here = root / "images" / "ADE" / folder / ("cultural/apse" if number % 2 else "cultural/arcade")
            (here / stem).mkdir(parents=True, exist_ok=True)
            ids = rng.choice(IDS, size=SIZE)
            ids[0, :3] = [2978, 312, 2420]              # every image holds the three classes
            held[stem] = ids
            Image.fromarray(rng.integers(0, 255, SIZE + (3,), dtype=np.uint8)).save(here / f"{stem}.jpg")
            Image.fromarray(seg_array(ids)).save(here / f"{stem}_seg.png")
    return held


def run(*tokens: str) -> int:
    return cli.main(["convert", "ade20k", *tokens])


def expected_mask(ids: np.ndarray) -> np.ndarray:
    lut = {i: index for index, (_, group) in enumerate(ade.SCENE_PARSE_150) for i in group}
    return np.vectorize(lambda value: lut.get(int(value), 255))(ids).astype(np.uint8)


def tree_digest(directory: Path) -> dict:
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob("*")) if p.is_file() and p.name != "manifest.jsonl"}


@pytest.fixture
def release(tmp_path):
    root = tmp_path / "source" / "ade"
    root.mkdir(parents=True)
    return root, write_release(root)


def test_a_run_writes_the_masks_the_listings_the_mapping_and_the_data_file(release, tmp_path, capsys):
    root, held = release
    out = tmp_path / "derived" / "ade"
    assert run(f"root={root}", f"out={out}", "workers=1") == 0
    assert capsys.readouterr().out == f"{out}\n", "stdout carries only the output directory"

    for split, folder in (("train", "training"), ("val", "validation")):
        stems = sorted(s for s in held if s.startswith("ADE_train" if split == "train" else "ADE_val"))
        rows = list(csv.DictReader(open(out / f"{split}.csv")))
        assert [Path(r["image"]).stem for r in rows] == stems
        for row, stem in zip(rows, stems):
            assert f"/images/ADE/{folder}/" in row["image"]
            assert row["image"].endswith(".jpg") and os.path.isabs(row["image"]) and str(root) in row["image"]
            mask = Image.open(out / row["mask"])
            assert mask.mode == "L" and np.array_equal(np.asarray(mask), expected_mask(held[stem]))

    summary = json.loads((out / "mapping.json").read_text())
    assert summary["mapping"] == "official" and summary["k"] == 150 and summary["void_index"] == 255
    assert [c["name"] for c in summary["classes"]][:3] == ["wall", "building", "sky"]
    train_ids = np.concatenate([held[s].ravel() for s in held if s.startswith("ADE_train")])
    for index, (identifier, name) in enumerate(((2978, "wall"), (312, "building"), (2420, "sky"))):
        entry = summary["classes"][index]
        assert entry["name"] == name and entry["ade_ids"] == [identifier]
        assert entry["train_pixels"] == int((train_ids == identifier).sum())
        assert entry["train_images"] == 4
    mapped = int(sum(c["train_pixels"] for c in summary["classes"]))
    assert mapped == int(np.isin(train_ids, [2978, 312, 2420]).sum())
    assert summary["train"]["total_pixels"] == train_ids.size
    assert summary["train"]["unlabeled_pixels"] == int((train_ids == 0).sum())
    assert summary["train"]["mapped_fraction_of_all_pixels"] == pytest.approx(mapped / train_ids.size)
    assert summary["train"]["mapped_fraction_of_labeled_pixels"] == pytest.approx(
        mapped / int((train_ids != 0).sum()))
    assert summary["excluded"] == [] and "NOT the SceneParse150" in summary["note"]

    # The data file is what the segmentation task reads: facts without a TensorFlow op.
    data = sa.load_seg_data_config(str(out / "data.yaml"))
    facts = sa.seg_facts(data, (16, 16))
    assert facts.num_classes == 150 and facts.ignore_index == 255 and facts.class_names[:3] == ("wall", "building", "sky")
    assert yaml.safe_load((out / "data.yaml").read_text())["test_csv"] == "val.csv"


def test_two_workers_write_the_same_bytes_as_one(release, tmp_path):
    root, _ = release
    assert run(f"root={root}", f"out={tmp_path / 'one'}", "workers=1") == 0
    assert run(f"root={root}", f"out={tmp_path / 'two'}", "workers=2") == 0
    one, two = tree_digest(tmp_path / "one"), tree_digest(tmp_path / "two")
    for name in ("mapping.json",):                       # holds the output-independent root only
        assert one[name] == two[name]
    assert one == two and len(one) > 6


def test_a_rerun_does_not_rewrite_a_finished_mask(release, tmp_path):
    root, _ = release
    out = tmp_path / "out"
    assert run(f"root={root}", f"out={out}", "workers=1") == 0
    masks = sorted((out / "masks").rglob("*.png"))
    before = {p: (p.stat().st_ino, p.stat().st_mtime_ns) for p in masks}
    assert run(f"root={root}", f"out={out}", "workers=1") == 0
    assert {p: (p.stat().st_ino, p.stat().st_mtime_ns) for p in masks} == before


def test_a_planted_partial_output_is_completed_and_nothing_is_deleted(release, tmp_path):
    root, held = release
    out = tmp_path / "out"
    assert run(f"root={root}", f"out={out}", "workers=1") == 0
    complete = tree_digest(out)
    # What a run killed in the middle leaves: the first two images in the manifest and a third one's line
    # cut short, the third mask half written to its .tmp, the later masks absent, the listings stale.
    lines = (out / "manifest.jsonl").read_text().splitlines()
    (out / "manifest.jsonl").write_text("\n".join(lines[:3]) + "\n" + lines[3][:25])
    masks = sorted((out / "masks").rglob("*.png"))
    kept, third, later = masks[:2], masks[2], masks[3:]
    third.rename(third.with_name(third.name + ".tmp"))
    third.with_name(third.name + ".tmp").write_bytes(b"half a PNG")
    for path in later:
        path.unlink()
    stale = out / "masks" / "train" / "unrelated.png.tmp"
    stale.write_bytes(b"left by a killed run")
    for name in ("train.csv", "val.csv", "mapping.json", "data.yaml"):
        (out / name).write_text("stale")
    marked = {p: p.stat().st_mtime_ns for p in kept}

    assert run(f"root={root}", f"out={out}", "workers=1") == 0
    assert {p: p.stat().st_mtime_ns for p in kept} == marked, "a finished mask is left alone"
    assert tree_digest(out) == {**complete, "masks/train/unrelated.png.tmp": hashlib.sha256(b"left by a killed run").hexdigest()}
    assert stale.read_bytes() == b"left by a killed run", "the command deletes nothing"
    # The cut line did not swallow the record written after it: every image has its own line.
    stems = [json.loads(line)["stem"] for line in (out / "manifest.jsonl").read_text().splitlines()[1:] if line.endswith("}")]
    assert sorted(set(stems)) == sorted(held)


def test_a_resume_with_another_root_is_refused_naming_the_key(release, tmp_path, capsys):
    root, _ = release
    out = tmp_path / "out"
    assert run(f"root={root}", f"out={out}", "workers=1") == 0
    other = tmp_path / "other_root"
    other.mkdir()
    write_release(other)
    capsys.readouterr()
    assert run(f"root={other}", f"out={out}") == 2
    error = capsys.readouterr().err
    assert "Invalid value for 'root'" in error and "a resume cannot change it" in error


def test_a_resume_with_another_mapping_or_k_is_refused_naming_the_key(release, tmp_path, capsys):
    root, _ = release
    out = tmp_path / "out"
    assert run(f"root={root}", f"out={out}", "mapping=frequency", "k=3", "workers=1") == 0
    capsys.readouterr()
    assert run(f"root={root}", f"out={out}", "mapping=frequency", "k=2") == 2
    assert "Invalid value for 'k'" in capsys.readouterr().err
    assert run(f"root={root}", f"out={out}") == 2
    assert "Invalid value for 'mapping'" in capsys.readouterr().err


def test_an_out_inside_the_root_or_around_it_is_refused_and_nothing_is_created(release, tmp_path, capsys):
    root, _ = release
    for out in (root / "converted", root, tmp_path):
        assert run(f"root={root}", f"out={out}") == 2
        assert "or contains it" in capsys.readouterr().err
    assert not (root / "converted").exists() and not (tmp_path / "manifest.jsonl").exists()
    link = tmp_path / "link"
    link.symlink_to(root, target_is_directory=True)
    assert run(f"root={root}", f"out={link / 'x'}") == 2, "a symlink into the source tree is the same thing"
    assert "or contains it" in capsys.readouterr().err and not (root / "x").exists()


def test_a_foreign_directory_and_a_file_are_refused_with_nothing_written(release, tmp_path, capsys):
    root, _ = release
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "notes.txt").write_text("mine")
    assert run(f"root={root}", f"out={foreign}") == 2
    assert "holds files and no manifest.jsonl" in capsys.readouterr().err
    assert sorted(p.name for p in foreign.iterdir()) == ["notes.txt"]
    afile = tmp_path / "afile"
    afile.write_text("x")
    assert run(f"root={root}", f"out={afile}") == 2
    assert "is not a directory" in capsys.readouterr().err and afile.read_text() == "x"


def test_an_empty_existing_directory_is_accepted(release, tmp_path):
    root, _ = release
    (tmp_path / "empty").mkdir()
    assert run(f"root={root}", f"out={tmp_path / 'empty'}", "workers=1") == 0
    assert (tmp_path / "empty" / "data.yaml").is_file()


def test_a_root_that_is_not_the_release_is_refused(tmp_path, capsys):
    (tmp_path / "nope").mkdir()
    assert run(f"root={tmp_path / 'nope'}", f"out={tmp_path / 'out'}") == 2
    assert "not the ADE20K 2021_17_01 release" in capsys.readouterr().err and not (tmp_path / "out").exists()


def test_a_release_whose_names_disagree_is_refused_before_anything_is_written(release, tmp_path, capsys):
    root, _ = release
    text = (root / "objects.txt").read_text().replace("wall\t2978", "stonewall\t2978")
    (root / "objects.txt").write_text(text)
    assert run(f"root={root}", f"out={tmp_path / 'out'}") == 2
    assert "class 0 'wall'" in capsys.readouterr().err and not (tmp_path / "out").exists()


def test_a_pair_that_cannot_be_used_is_excluded_and_listed(release, tmp_path):
    root, held = release
    pairs = ade.list_release(str(root))
    wrong_size, turned, broken = pairs["train"][0], pairs["train"][1], pairs["val"][0]
    Image.fromarray(np.zeros((SIZE[0] + 2, SIZE[1], 3), dtype=np.uint8)).save(wrong_size.image)
    picture = Image.fromarray(np.zeros(SIZE + (3,), dtype=np.uint8))
    exif = picture.getexif()
    exif[0x0112] = 3                                    # upside down: same size, the mask would not turn with it
    picture.save(turned.image, exif=exif)
    Path(broken.seg).write_bytes(b"not a png")

    out = tmp_path / "out"
    assert run(f"root={root}", f"out={out}", "workers=1") == 0
    summary = json.loads((out / "mapping.json").read_text())
    reasons = {e["stem"]: e["reason"] for e in summary["excluded"]}
    assert set(reasons) == {wrong_size.stem, turned.stem, broken.stem}
    assert "width x height" in reasons[wrong_size.stem] and "EXIF orientation 3" in reasons[turned.stem]
    assert "cannot be decoded" in reasons[broken.stem]
    listed = {Path(r["image"]).stem for name in ("train.csv", "val.csv") for r in csv.DictReader(open(out / name))}
    assert not listed & set(reasons) and len(listed) == 3
    assert not any(list((out / "masks").rglob(f"{stem}.png")) for stem in reasons)
    # An excluded pair stays excluded on a rerun and is not tried again.
    before = (out / "manifest.jsonl").read_text()
    assert run(f"root={root}", f"out={out}", "workers=1") == 0
    assert (out / "manifest.jsonl").read_text() == before


def test_the_frequency_mapping_takes_the_most_frequent_train_classes_and_counts_once(release, tmp_path, capsys):
    root, held = release
    out = tmp_path / "out"
    assert run(f"root={root}", f"out={out}", "mapping=frequency", "k=3", "workers=1") == 0
    summary = json.loads((out / "mapping.json").read_text())
    train_ids = np.concatenate([held[s].ravel() for s in held if s.startswith("ADE_train")])
    ranked = sorted(set(IDS) - {0}, key=lambda i: (-int((train_ids == i).sum()), i))[:3]
    assert [c["ade_ids"] for c in summary["classes"]] == [[i] for i in ranked]
    assert summary["k"] == 3 and summary["mapping"] == "frequency"
    first = (out / "pixel_counts.json").stat().st_mtime_ns
    mask = np.asarray(Image.open(out / "masks" / "train" / "ADE_train_00000001.png"))
    lookup = {identifier: index for index, identifier in enumerate(ranked)}
    assert np.array_equal(mask, np.vectorize(lambda v: lookup.get(int(v), 255))(held["ADE_train_00000001"]).astype(np.uint8))
    assert run(f"root={root}", f"out={out}", "mapping=frequency", "k=3", "workers=1") == 0
    assert (out / "pixel_counts.json").stat().st_mtime_ns == first, "the counting pass is not repeated"


def test_a_k_with_the_official_mapping_or_beyond_the_release_is_refused_by_key(release, tmp_path, capsys):
    root, _ = release
    assert run(f"root={root}", f"out={tmp_path / 'a'}", "k=10") == 2
    error = capsys.readouterr().err
    assert "Invalid value for 'k'" in error and "only mapping=frequency reads it" in error
    assert run(f"root={root}", f"out={tmp_path / 'b'}", "mapping=frequency", "k=50", "workers=1") == 2
    assert "Invalid value for 'k'" in capsys.readouterr().err
    assert run(f"root={root}", f"out={tmp_path / 'c'}", "mapping=frequency", "k=255") == 2
    assert "expected 2 to 254" in capsys.readouterr().err
    assert not any((tmp_path / n).exists() for n in "ac"), "a bad key creates nothing"


@pytest.mark.parametrize("tokens, message", [
    (["out=x"], "Missing required key 'root'"),
    (["root=x"], "Missing required key 'out'"),
    (["root=x", "out=y", "mappin=official"], "Did you mean 'mapping'"),
    (["root=x", "out=y", "mapping=zip"], "mapping"),
    (["root=x", "out=y", "workers=0"], "expected 1 or more"),
    (["root=x", "out=y", "config=a.yaml"], "takes no config="),
])
def test_a_bad_key_exits_2_naming_it(tokens, message, tmp_path, capsys):
    assert cli.main(["convert", "ade20k", *tokens]) == 2
    assert message in capsys.readouterr().err
    assert not Path("y").exists()


def test_an_unknown_dataset_is_refused_with_the_choices(capsys):
    assert cli.main(["convert", "cityscapes"]) == 2
    error = capsys.readouterr().err
    assert "Unknown dataset 'cityscapes'" in error and "choose one of ade20k" in error


def test_the_help_lists_the_datasets_and_the_keys_and_loads_no_tensorflow():
    code = (
        "import sys, io\n"
        "from train.api import cli\n"
        "assert cli.main(['convert']) == 0\n"
        "assert cli.main(['convert', 'help']) == 0\n"
        "assert cli.main(['convert', 'ade20k']) == 0\n"
        "assert cli.main(['convert', 'ade20k', 'help']) == 0\n"
        "assert cli.main(['help']) == 0\n"
        "heavy = [m for m in ('tensorflow', 'keras', 'numpy', 'PIL', 'yaml') if m in sys.modules]\n"
        "assert not heavy, heavy\n")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"})
    assert done.returncode == 0, done.stderr
    assert "ade20k" in done.stdout and "mapping" in done.stdout and "convert" in done.stdout
    assert "  mapping  (" in done.stdout and "[default: 'official']" in done.stdout


@pytest.mark.skipif(not (REAL_RELEASE / "objects.txt").is_file(), reason="the ADE20K release is not on this disk")
def test_the_real_releases_names_agree_with_the_benchmark_ids():
    names = ade.read_object_names(str(REAL_RELEASE / "objects.txt"))
    mapping = ade.official_mapping(names)
    assert len(names) == 1599 and len(mapping.class_names) == 150
    assert mapping.source_names[0] == ("wall",) and mapping.source_names[14] == ("door", "double door")

