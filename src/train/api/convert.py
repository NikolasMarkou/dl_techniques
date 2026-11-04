"""``python -m train convert``: write a source dataset as index masks and the listings the segmentation task reads.

``convert ade20k`` reads the full ADE20K release (3,688 classes coded in RGB ``_seg.png`` files, see
:mod:`dl_techniques.datasets.vision.ade20k`) and ``convert coco`` reads COCO 2017's instance annotations (see
:mod:`dl_techniques.datasets.vision.coco_semantic`). Each writes, to a NEW directory, one ``uint8`` PNG of
class indices per image (``255`` is void), ``train.csv`` and ``val.csv`` (columns ``image`` and ``mask``; the
image is the original JPEG, never copied), ``mapping.json`` (what every class is, its pixels, the coverage)
and ``data.yaml``, which ``python -m train image_segmentation data=<out>/data.yaml`` reads. Neither result is
the benchmark its source is known for: its mIoU is not comparable to published numbers (README,
``Converting ADE20K``, ``Converting COCO``).

Interface contract
------------------
``convert`` / ``convert help``
    Lists the datasets. ``convert <dataset> help`` lists the keys of one.

``convert ade20k root=<release> out=<new dir> [mapping=official|frequency] [k=N] [workers=N]``
    :param root: the release directory (``images/ADE/training`` and ``objects.txt`` under it). Read only.
    :param out: where to write. It is created; an existing one must be empty or the output of an earlier
        run of this same command. It must not lie inside ``root`` nor contain it.
    :param mapping: ``official`` (default) is the MIT SceneParse150 class set and its id mapping,
        checked against the release's names; ``frequency`` is the ``k`` classes with the most pixels in
        the train split (a counting pass over every train mask comes first).
    :param k: ``frequency`` only (default 150); refused with ``official``, which is always 150 classes.
    :param workers: processes; default ``min(8, cpu count)``.
    :return: exit 0 and ONLY ``out`` on stdout; 2 for a bad key, path or release.

``convert coco root=<coco_2017> out=<new dir> [workers=N]``
    :param root: the COCO 2017 directory (``annotations/instances_{train,val}2017.json``, ``train2017``,
        ``val2017``). Read only; the images are opened for their size and EXIF orientation only.
    :return: as above. Classes: ``background`` (0, everything that is no annotated thing: COCO's instance files
        have no stuff), the 80 things (1 to 80, category-id order) and void 255 for crowd regions and for pixels
        two classes claim. Needs ``pycocotools`` (not a declared dependency; a missing one exits 2 naming it).

Resumable and never deleting: ``manifest.jsonl`` holds the header of the run (dataset, root, mapping, k) and
one line per finished image; a rerun skips an image whose mask file exists at the recorded size, and refuses a
changed ``root``, ``mapping`` or ``k`` naming the key. A mask is written to ``<name>.png.tmp`` and moved into
place with ``os.replace``, so a killed run leaves whole masks and at most stale ``.tmp`` files (overwritten by
the next run, deleted by nobody). An image whose JPEG and mask (or annotation) differ in size, whose JPEG
carries an EXIF orientation, or that cannot be decoded is EXCLUDED: listed in ``mapping.json`` and left out of
the CSVs (the segmentation loader would refuse it).

Import rule
-----------
Module scope imports the standard library and ``train.api`` only (D-012); numpy, PIL, ``yaml`` and the
dataset modules are imported inside the functions that need them, so ``convert help`` loads none of them.
"""

import dataclasses
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

from dl_techniques.utils.logger import logger

from .cli import HELP_WORDS, PROGRAM, _field_help_lines, coerce_mapping, parse_dotlist
from .config import StrictConfig, join_path, validate
from .errors import ConfigError

#: The manifest's name; its first line is the header, the rest one finished image each.
MANIFEST = "manifest.jsonl"
#: The version of the manifest, mapping.json and the file layout.
FORMAT_VERSION = 1
_LOG_EVERY = 2000


@dataclasses.dataclass(frozen=True)
class ConvertAde20k(StrictConfig):
    """The keys of ``convert ade20k``."""

    root: str = dataclasses.field(
        metadata={"help": "the ADE20K 2021_17_01 release directory (images/ADE/training, objects.txt); read only"})
    out: str = dataclasses.field(
        metadata={"help": "directory to write (created; empty or an earlier output of this command); not inside root"})
    mapping: Literal["official", "frequency"] = dataclasses.field(
        default="official",
        metadata={"help": "official: the MIT SceneParse150 classes through the benchmark's id mapping; "
                  "frequency: the k classes with the most train pixels"})
    k: Optional[int] = dataclasses.field(
        default=None, metadata={"help": "frequency only: the number of classes, 2 to 254 (null is 150)"})
    workers: Optional[int] = dataclasses.field(
        default=None, metadata={"help": "processes that decode and write; null is min(8, cpu count)"})

    def problems(self, prefix: str) -> List[str]:
        """Name every key out of range, at its dotted path."""
        lines: List[str] = []
        for name in ("root", "out"):
            if not getattr(self, name).strip():
                lines.append(f"Invalid value for {join_path(prefix, name)!r}: expected a non-empty path")
        if self.k is not None:
            if self.mapping != "frequency":
                lines.append(f"Invalid value for {join_path(prefix, 'k')!r}: only mapping=frequency reads it "
                             f"(mapping=official is the benchmark's 150 classes), got {self.k!r}")
            elif not 2 <= self.k <= 254:
                lines.append(f"Invalid value for {join_path(prefix, 'k')!r}: expected 2 to 254 (255 is void), got {self.k!r}")
        if self.workers is not None and self.workers < 1:
            lines.append(f"Invalid value for {join_path(prefix, 'workers')!r}: expected 1 or more, got {self.workers!r}")
        return lines


@dataclasses.dataclass(frozen=True)
class ConvertCoco(StrictConfig):
    """The keys of ``convert coco``."""

    root: str = dataclasses.field(
        metadata={"help": "the COCO 2017 directory (annotations/instances_train2017.json, train2017, val2017); read only"})
    out: str = dataclasses.field(
        metadata={"help": "directory to write (created; empty or an earlier output of this command); not inside root"})
    workers: Optional[int] = dataclasses.field(
        default=None, metadata={"help": "processes that rasterize and write; null is min(8, cpu count)"})

    def problems(self, prefix: str) -> List[str]:
        """Name every key out of range, at its dotted path."""
        lines = [f"Invalid value for {join_path(prefix, name)!r}: expected a non-empty path"
                 for name in ("root", "out") if not getattr(self, name).strip()]
        if self.workers is not None and self.workers < 1:
            lines.append(f"Invalid value for {join_path(prefix, 'workers')!r}: expected 1 or more, got {self.workers!r}")
        return lines


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

def _check_roots(root: str, out: str, sources: Sequence[str] = ()) -> Tuple[str, str]:
    """Return the real paths of ``root`` and ``out``; refuse an ``out`` in or around ``root`` or a source directory."""
    if not os.path.isdir(root):
        raise ConfigError(f"Invalid value for 'root': {root!r} is not a directory")
    real_root, real_out = os.path.realpath(root), os.path.realpath(out)
    for real in (real_root, *[os.path.realpath(source) for source in sources]):
        if os.path.commonpath([real, real_out]) in (real, real_out):
            raise ConfigError(
                f"Invalid value for 'out': {out!r} (real path {real_out!r}) lies inside {real!r} or contains it; a "
                "conversion never writes into the source tree, so give a directory beside it")
    return real_root, real_out


def _write_text(path: str, text: str) -> None:
    """Write ``text`` whole or not at all: to ``<path>.tmp``, then ``os.replace``."""
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(temporary, path)


def _read_manifest(path: str) -> Tuple[Optional[Dict[str, Any]], Dict[Tuple[str, str], Dict[str, Any]]]:
    """Return ``(header, {(split, stem): record})``; a line that is not JSON (a write a kill cut short) counts for nothing."""
    if not os.path.isfile(path):
        return None, {}
    with open(path, encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    header: Optional[Dict[str, Any]] = None
    records: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for number, line in enumerate(lines):
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("convert: line %d of %s is not JSON (a write cut short); its image is converted again", number + 1, path)
            continue
        if number == 0:
            header = item
        else:
            records[(item["split"], item["stem"])] = item
    return header, records


def _ends_with_newline(path: str) -> bool:
    """Whether the file ends in a newline (an empty one does); a manifest that does not was cut short by a kill."""
    with open(path, "rb") as handle:
        size = handle.seek(0, os.SEEK_END)
        if size == 0:
            return True
        handle.seek(size - 1)
        return handle.read(1) == b"\n"


def _open_out(out: str, header: Mapping[str, Any]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Create ``out`` or accept an earlier output of this run; return its finished records."""
    manifest = os.path.join(out, MANIFEST)
    if os.path.isdir(out) and os.listdir(out) and not os.path.isfile(manifest):
        raise ConfigError(f"Invalid value for 'out': {out!r} holds files and no {MANIFEST}, so it is not the output "
                          "of this command; give a new or empty directory")
    if os.path.exists(out) and not os.path.isdir(out):
        raise ConfigError(f"Invalid value for 'out': {out!r} exists and is not a directory")
    previous, records = _read_manifest(manifest)
    if previous is not None:
        for key in ("dataset", "root", "mapping", "k"):
            if previous.get(key) != header[key]:
                raise ConfigError(
                    f"Invalid value for {key!r}: {out!r} holds a conversion made with {key}={previous.get(key)!r} and "
                    f"this run asks {header[key]!r}; a resume cannot change it, give a new out")
        return records
    os.makedirs(out, exist_ok=True)
    with open(manifest, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(header)) + "\n")
    return {}


# ---------------------------------------------------------------------
# The parts both datasets share
# ---------------------------------------------------------------------

#: How many tasks a pool is handed at once (the rest wait here, not in the pool's queue).
_WINDOW = 4000
#: What the workers share: set once per process by an initializer.
_STATE: Dict[str, Any] = {}


def _run_tasks(tasks: Sequence[Any], worker: Any, workers: int, initializer: Any, initargs: Tuple[Any, ...],
               records: Dict[Tuple[str, str], Dict[str, Any]], out: str, total: int) -> None:
    """Run ``worker`` over ``tasks`` and append each record it returns to the manifest as it arrives."""
    started = time.time()
    manifest_path = os.path.join(out, MANIFEST)
    cut_short = not _ends_with_newline(manifest_path)
    with open(manifest_path, "a", encoding="utf-8") as manifest:
        if cut_short:
            manifest.write("\n")   # the line a kill cut short ends here, so the next record starts its own

        def finish(record: Dict[str, Any]) -> None:
            records[(record["split"], record["stem"])] = record
            manifest.write(json.dumps(record) + "\n")
            manifest.flush()
            if len(records) % _LOG_EVERY == 0:
                logger.info("convert: %d of %d images, %.0f s", len(records), total, time.time() - started)

        if workers == 1 or len(tasks) < 2:
            initializer(*initargs)
            for task in tasks:
                finish(worker(task))
        else:
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(max_workers=workers, initializer=initializer, initargs=initargs) as pool:
                for start in range(0, len(tasks), _WINDOW):
                    for record in pool.map(worker, tasks[start:start + _WINDOW], chunksize=8):
                        finish(record)
        os.fsync(manifest.fileno())


def _is_done(record: Optional[Mapping[str, Any]], out: str, split: str, stem: str) -> bool:
    """Whether an image needs no work: it is excluded, or its mask file exists at the size the manifest recorded."""
    if record is None:
        return False
    target = os.path.join(out, "masks", split, stem + ".png")
    return "excluded" in record or (os.path.isfile(target) and os.path.getsize(target) == record["bytes"])


def _save_mask(out: str, split: str, stem: str, mask: Any) -> int:
    """Write a ``uint8`` mask to ``masks/<split>/<stem>.png`` whole or not at all; return its size in bytes."""
    from PIL import Image

    target = os.path.join(out, "masks", split, stem + ".png")
    os.makedirs(os.path.dirname(target), exist_ok=True)
    Image.fromarray(mask).save(target + ".tmp", format="PNG")
    os.replace(target + ".tmp", target)
    return os.path.getsize(target)


def _unusable(image_path: str, mask_size: Tuple[int, int], what: str) -> Optional[str]:
    """Why an image cannot be paired with a mask of ``mask_size`` (width, height), or ``None`` if it can."""
    from PIL import Image

    try:
        with Image.open(image_path) as image:
            size, orientation = image.size, image.getexif().get(0x0112)
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        return f"the JPEG cannot be decoded: {exc}"
    if orientation not in (None, 1):
        return (f"the JPEG carries EXIF orientation {orientation}, so the segmentation loader would turn the "
                "photograph and not the mask")
    if size != mask_size:
        return f"the JPEG is {size[0]}x{size[1]} (width x height) and its {what} {mask_size[0]}x{mask_size[1]}"
    return None


def _class_totals(records: Sequence[Mapping[str, Any]], split: str, count: int) -> Tuple[List[int], List[int], int]:
    """Return ``(pixels per class, images per class, void pixels)`` over the converted images of one split."""
    pixels, images, void = [0] * count, [0] * count, 0
    for record in records:
        if record["split"] != split or "excluded" in record:
            continue
        for value, number in record["pixels"].items():
            if int(value) == 255:
                void += number
            else:
                pixels[int(value)] += number
                images[int(value)] += 1
    return pixels, images, void


def _write_outputs(out: str, rows: Mapping[str, Sequence[Tuple[str, str, str]]], class_names: Sequence[str],
                   summary: Mapping[str, Any]) -> None:
    """Write ``train.csv``, ``val.csv`` (rows are ``(stem, image, reason or "")``), ``mapping.json`` and ``data.yaml``."""
    import csv
    import io

    import yaml

    excluded = []
    for split, name in (("train", "train.csv"), ("val", "val.csv")):
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(("image", "mask"))
        for stem, image, reason in rows[split]:
            if reason:
                excluded.append({"split": split, "stem": stem, "reason": reason})
            else:
                writer.writerow((image, f"masks/{split}/{stem}.png"))
        _write_text(os.path.join(out, name), buffer.getvalue())
    _write_text(os.path.join(out, "mapping.json"), json.dumps({**summary, "excluded": excluded}, indent=1) + "\n")
    body = yaml.safe_dump(
        {"format": "csv", "train_csv": "train.csv", "test_csv": "val.csv", "class_names": list(class_names),
         "ignore_index": 255}, sort_keys=False, allow_unicode=True, width=1000)
    _write_text(os.path.join(out, "data.yaml"),
                "# Written by `python -m train convert`; read it with data=<this file>.\n" + body)


def _rows(split_pairs: Sequence[Tuple[str, str]], records: Mapping[Tuple[str, str], Mapping[str, Any]],
          split: str) -> List[Tuple[str, str, str]]:
    return [(stem, image, records[(split, stem)].get("excluded", "")) for stem, image in split_pairs]


# ---------------------------------------------------------------------
# ADE20K
# ---------------------------------------------------------------------

def _init_ade(lut: Any, out: str) -> None:
    _STATE["lut"], _STATE["out"] = lut, out


def _convert_pair(task: Tuple[str, str, str, str]) -> Dict[str, Any]:
    """Convert one ADE20K pair; return its manifest record (``excluded`` set when the pair cannot be used)."""
    import numpy as np

    from dl_techniques.datasets.vision import ade20k

    split, stem, image_path, seg_path = task
    record: Dict[str, Any] = {"split": split, "stem": stem}
    # DECISION plan-2026-11-03T221936-98b866fb/D-017
    # A pair that cannot be used is EXCLUDED and listed (mapping.json), not converted and not refused: the
    # segmentation loader refuses a pair of two sizes after the EXIF turn, so one such pair in a CSV would stop
    # every run on the set, and refusing here would stop a 27,574-image conversion at its last file. Do NOT
    # write it with a resize to make the sizes agree (the mask would no longer be the annotation).
    try:
        seg = ade20k.read_seg(seg_path)
    except ade20k.AdeError as exc:
        record["excluded"] = f"cannot be decoded: {exc}"
        return record
    reason = _unusable(image_path, (seg.shape[1], seg.shape[0]), "_seg.png")
    if reason:
        record["excluded"] = reason
        return record
    mask = _STATE["lut"][seg]
    counts = np.bincount(mask.ravel(), minlength=256)
    record.update(
        bytes=_save_mask(_STATE["out"], split, stem, mask), height=int(mask.shape[0]), width=int(mask.shape[1]),
        unlabeled=int((seg == 0).sum()), pixels={str(i): int(n) for i, n in enumerate(counts) if n})
    return record


def _pixel_counts(out: str, train_segs: Sequence[str], workers: int) -> Any:
    """Return the train pixels of every release id: read from ``out`` if it holds them for these files."""
    import numpy as np

    from dl_techniques.datasets.vision import ade20k

    signature = hashlib.sha256("\n".join(train_segs).encode("utf-8")).hexdigest()
    cache = os.path.join(out, "pixel_counts.json")
    if os.path.isfile(cache):
        with open(cache, encoding="utf-8") as handle:
            saved = json.load(handle)
        if saved.get("signature") == signature:
            counts = np.zeros(ade20k.MAX_CLASS_ID + 1, dtype=np.int64)
            for identifier, pixels in saved["counts"].items():
                counts[int(identifier)] = pixels
            logger.info("ade20k: pixel counts of %d train masks read from %s", len(train_segs), cache)
            return counts
    started = time.time()
    counts = ade20k.class_pixel_counts(train_segs, workers)
    logger.info("ade20k: counted the pixels of %d train masks in %.0f s (workers=%d)",
                len(train_segs), time.time() - started, workers)
    _write_text(cache, json.dumps({"signature": signature, "counts": {str(i): int(counts[i]) for i in np.nonzero(counts)[0]}}))
    return counts


def _ade_summary(mapping: Any, records: Sequence[Mapping[str, Any]], header: Mapping[str, Any], names_file: str) -> Dict[str, Any]:
    """The ``mapping.json`` of an ADE20K conversion: every class, its pixels and the coverage of the set."""
    count = len(mapping.class_names)
    totals = {split: _class_totals(records, split, count) for split in ("train", "val")}
    unlabeled = {split: sum(r["unlabeled"] for r in records if r["split"] == split and "excluded" not in r)
                 for split in totals}

    def coverage(split: str) -> Dict[str, Any]:
        pixels, _, void = totals[split]
        mapped, total = sum(pixels), sum(pixels) + void
        return {"unlabeled_pixels": unlabeled[split], "total_pixels": total,
                "mapped_fraction_of_all_pixels": mapped / total if total else None,
                "mapped_fraction_of_labeled_pixels": mapped / (total - unlabeled[split]) if total > unlabeled[split] else None}

    classes = [{"index": i, "name": name, "ade_ids": list(mapping.source_ids[i]), "ade_names": list(mapping.source_names[i]),
                "train_pixels": totals["train"][0][i], "train_images": totals["train"][1][i],
                "val_pixels": totals["val"][0][i], "val_images": totals["val"][1][i]}
               for i, name in enumerate(mapping.class_names)]
    return {
        "format_version": FORMAT_VERSION, "dataset": "ade20k", "mapping": mapping.kind, "k": count,
        "void_index": 255, "release_root": header["root"], "names_file": names_file,
        "note": "NOT the SceneParse150 benchmark: the release is ADE20K 2021_17_01 (25,574 training images including video "
                "frames, the benchmark has 20,210; full-size images and masks, the benchmark rescales to a short side of at "
                "most 512), so mIoU here is not comparable to published numbers",
        "sources": {
            "official": "github.com/CSAILVision/sceneparsing commit e202229: objectInfo150.txt (names), "
                        "convertFromADE/mapFromADE.txt (ids); reduce_zero_label as in open-mmlab/mmsegmentation "
                        "ADE20KDataset (0 and every unmapped class become the void value)",
            "frequency": "the k classes with the most train pixels, ties by smaller id",
        }[mapping.kind],
        "splits": {"train": "images/ADE/training", "val": "images/ADE/validation (the benchmark's 2,000 images; "
                   "data.yaml uses it as the test split)"},
        "train": coverage("train"), "val": coverage("val"), "classes": classes,
    }


def _convert_ade20k(config: ConvertAde20k) -> str:
    """Run ``convert ade20k``; return the output directory."""
    from dl_techniques.datasets.vision import ade20k

    root, out = _check_roots(config.root, config.out)
    try:
        release = ade20k.list_release(root)
        names_file = os.path.join(root, "objects.txt")
        names = ade20k.read_object_names(names_file)
        if config.mapping == "official":
            mapping = ade20k.official_mapping(names)
    except ade20k.AdeError as exc:
        raise ConfigError(f"Invalid value for 'root': {exc}") from exc
    workers = config.workers or min(8, os.cpu_count() or 1)
    k = (config.k or 150) if config.mapping == "frequency" else 150
    header = {"dataset": "ade20k", "root": root, "mapping": config.mapping, "k": k, "version": FORMAT_VERSION}
    records = _open_out(out, header)
    if config.mapping == "frequency":
        try:
            mapping = ade20k.frequency_mapping(_pixel_counts(out, [p.seg for p in release["train"]], workers), names, k)
        except ade20k.AdeError as exc:
            raise ConfigError(f"Invalid value for 'k': {exc}") from exc
    tasks = [(split, pair.stem, pair.image, pair.seg) for split in ("train", "val") for pair in release[split]
             if not _is_done(records.get((split, pair.stem)), out, split, pair.stem)]
    total = sum(len(v) for v in release.values())
    logger.info("ade20k: %d images, %d already converted, %d to do (workers=%d, mapping=%s, %d classes)",
                total, total - len(tasks), len(tasks), workers, mapping.kind, len(mapping.class_names))
    started = time.time()
    _run_tasks(tasks, _convert_pair, workers, _init_ade, (mapping.lut(), out), records, out, total)
    skipped = [record for record in records.values() if "excluded" in record]
    if skipped:
        logger.warning("ade20k: %d image(s) excluded, listed in mapping.json (first: %s %s)",
                       len(skipped), skipped[0]["stem"], skipped[0]["excluded"])
    wanted = {(split, pair.stem): records[(split, pair.stem)] for split in release for pair in release[split]}
    rows = {split: _rows([(p.stem, p.image) for p in release[split]], wanted, split) for split in release}
    _write_outputs(out, rows, mapping.class_names, _ade_summary(mapping, list(wanted.values()), header, names_file))
    logger.info("ade20k: wrote %s in %.0f s", out, time.time() - started)
    return out


# ---------------------------------------------------------------------
# COCO
# ---------------------------------------------------------------------

def _init_coco(index_of: Mapping[int, int], out: str) -> None:
    _STATE["index_of"], _STATE["out"] = index_of, out


def _convert_image(task: Tuple[str, str, str, int, int, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Rasterize one COCO image; return its manifest record (``excluded`` set when the image cannot be used)."""
    import numpy as np

    from dl_techniques.datasets.vision import coco_semantic

    split, stem, image_path, width, height, annotations = task
    record: Dict[str, Any] = {"split": split, "stem": stem}
    reason = _unusable(image_path, (width, height), "annotation")
    if reason:
        record["excluded"] = reason
        return record
    try:
        mask, crowd, conflict = coco_semantic.rasterize(height, width, annotations, _STATE["index_of"])
    except coco_semantic.CocoError as exc:
        if "pycocotools" in str(exc):
            raise
        record["excluded"] = f"cannot be rasterized: {exc}"
        return record
    counts = np.bincount(mask.ravel(), minlength=256)
    record.update(bytes=_save_mask(_STATE["out"], split, stem, mask), height=height, width=width,
                  crowd_pixels=crowd, conflict_pixels=conflict, annotations=len(annotations),
                  pixels={str(i): int(n) for i, n in enumerate(counts) if n})
    return record


def _coco_summary(names: Sequence[str], category_ids: Sequence[int], records: Sequence[Mapping[str, Any]],
                  header: Mapping[str, Any]) -> Dict[str, Any]:
    """The ``mapping.json`` of a COCO conversion: the classes, their pixels and what was made void."""
    totals = {split: _class_totals(records, split, len(names)) for split in ("train", "val")}

    def detail(split: str) -> Dict[str, Any]:
        done = [r for r in records if r["split"] == split and "excluded" not in r]
        pixels, _, void = totals[split]
        everything = sum(pixels) + void
        return {"images": len(done), "images_without_an_annotation": sum(1 for r in done if r["annotations"] == 0),
                "total_pixels": everything, "background_fraction": pixels[0] / everything if everything else None,
                "void_fraction": void / everything if everything else None,
                "crowd_pixels": sum(r["crowd_pixels"] for r in done), "conflict_pixels": sum(r["conflict_pixels"] for r in done)}

    classes = [{"index": i, "name": name, "category_id": None if i == 0 else category_ids[i - 1],
                "train_pixels": totals["train"][0][i], "train_images": totals["train"][1][i],
                "val_pixels": totals["val"][0][i], "val_images": totals["val"][1][i]} for i, name in enumerate(names)]
    return {
        "format_version": FORMAT_VERSION, "dataset": "coco", "mapping": header["mapping"], "k": len(names), "void_index": 255,
        "release_root": header["root"],
        "note": "NOT COCO-Stuff or COCO panoptic: the instance files have no stuff, so class 0 is every pixel that is no "
                "annotated thing; mIoU here is not comparable to published numbers",
        "rules": "classes 1..80 are the categories in id order; a crowd region is void (255); a pixel two classes claim is "
                 "void; one class twice keeps the class; the result does not depend on the order of the annotations",
        "splits": {"train": "train2017", "val": "val2017 (data.yaml uses it as the test split)"},
        "train": detail("train"), "val": detail("val"), "classes": classes,
    }


def _convert_coco(config: ConvertCoco) -> str:
    """Run ``convert coco``; return the output directory."""
    from dl_techniques.datasets.vision import coco_semantic

    folders = [os.path.join(config.root, name) for name in ("annotations", "train2017", "val2017")]
    root, out = _check_roots(config.root, config.out, [folder for folder in folders if os.path.isdir(folder)])
    for split in ("train", "val"):
        path = os.path.join(root, "annotations", f"instances_{split}2017.json")
        if not os.path.isfile(path) or not os.path.isdir(os.path.join(root, f"{split}2017")):
            raise ConfigError(f"Invalid value for 'root': {root!r} is not the COCO 2017 directory ({path!r} or "
                              f"the {split}2017 image directory is missing)")
    workers = config.workers or min(8, os.cpu_count() or 1)
    header = {"dataset": "coco", "root": root, "mapping": "instances", "k": 80, "version": FORMAT_VERSION}
    records = _open_out(out, header)
    started = time.time()
    wanted: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    rows: Dict[str, List[Tuple[str, str, str]]] = {}
    names: Tuple[str, ...] = ()
    category_ids: List[int] = []
    total = 0
    for split in ("train", "val"):
        with open(os.path.join(root, "annotations", f"instances_{split}2017.json"), encoding="utf-8") as handle:
            document = json.load(handle)
        try:
            names, index_of = coco_semantic.category_table(document["categories"])
        except coco_semantic.CocoError as exc:
            raise ConfigError(f"Invalid value for 'root': {exc}") from exc
        category_ids = sorted(index_of)
        by_image: Dict[int, List[Dict[str, Any]]] = {}
        for annotation in document["annotations"]:
            by_image.setdefault(annotation["image_id"], []).append(
                {"category_id": annotation["category_id"], "iscrowd": annotation["iscrowd"],
                 "segmentation": annotation["segmentation"]})
        images = sorted(document["images"], key=lambda image: image["id"])
        del document
        total += len(images)
        listing = [(os.path.splitext(image["file_name"])[0], os.path.join(root, f"{split}2017", image["file_name"]), image)
                   for image in images]
        tasks = [(split, stem, path, image["width"], image["height"], by_image.get(image["id"], []))
                 for stem, path, image in listing if not _is_done(records.get((split, stem)), out, split, stem)]
        logger.info("coco %s: %d images, %d to do (workers=%d)", split, len(listing), len(tasks), workers)
        _run_tasks(tasks, _convert_image, workers, _init_coco, (index_of, out), records, out, total)
        wanted.update({(split, stem): records[(split, stem)] for stem, _, _ in listing})
        rows[split] = _rows([(stem, path) for stem, path, _ in listing], wanted, split)
        del by_image, tasks, listing, images
    skipped = [record for record in wanted.values() if "excluded" in record]
    if skipped:
        logger.warning("coco: %d image(s) excluded, listed in mapping.json (first: %s %s)",
                       len(skipped), skipped[0]["stem"], skipped[0]["excluded"])
    _write_outputs(out, rows, names, _coco_summary(names, category_ids, list(wanted.values()), header))
    logger.info("coco: wrote %s in %.0f s", out, time.time() - started)
    return out


# ---------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------

#: The datasets ``convert`` knows: name, one-line summary, config class, function.
DATASETS = {
    "ade20k": ("the full ADE20K release to index masks and CSV listings (150 classes, void 255)",
               ConvertAde20k, _convert_ade20k),
    "coco": ("COCO 2017 instances to semantic masks (background + 80 things, void 255) and CSV listings",
             ConvertCoco, _convert_coco),
}


def run_convert(args: List[str]) -> int:
    """Run ``convert <dataset> key=value ...``; see the module docstring."""
    if not args or args == ["help"] or (len(args) == 1 and args[0] in HELP_WORDS):
        width = max(len(name) for name in DATASETS)
        lines = [f"usage: {PROGRAM} convert <dataset> [key=value ...]", f"       {PROGRAM} convert <dataset> help", "",
                 "datasets:", *[f"  {name.ljust(width)}  {item[0]}" for name, item in DATASETS.items()], ""]
        sys.stdout.write("\n".join(lines))
        return 0
    name, rest = args[0], args[1:]
    if name not in DATASETS:
        raise ConfigError(f"Unknown dataset {name!r} for convert; choose one of {', '.join(DATASETS)}")
    summary, config_class, function = DATASETS[name]
    if not rest or (len(rest) == 1 and rest[0] in HELP_WORDS):
        lines = [f"usage: {PROGRAM} convert {name} key=value ...", "", summary, "", "keys:",
                 *_field_help_lines(config_class), ""]
        sys.stdout.write("\n".join(lines))
        return 0
    config_path, raw = parse_dotlist(rest)
    if config_path is not None:
        raise ConfigError(f"convert {name} takes no config=; give its keys on the command line")
    config = validate(config_class, coerce_mapping(config_class, raw))
    sys.stdout.write(f"{function(config)}\n")
    return 0
