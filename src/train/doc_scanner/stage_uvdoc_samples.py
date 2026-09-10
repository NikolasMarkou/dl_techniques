"""Precompute UVDoc ``(image, f_gt, g, mask)`` sidecars for the rectifier.

What this exists to remove
--------------------------
``dl_techniques.datasets.document_rectification.uvdoc.load_geometry`` costs
about **1.06 s** per geometry, essentially all of it scattered-data
interpolation: a cubic tensor-product densification of the 89x61 correspondence
lattice, then a ``griddata`` inversion for ``g``. That cost is per GEOMETRY, and
UVDoc fans **20,000 renders out of 4,032 geometries**, so the arithmetic that
matters is:

===========================  ==========================  ==================
staging unit                 inversions for the corpus   single-process wall
===========================  ==========================  ==================
per render                   20,000                      ~6 h
per geometry (what we do)    4,032                       ~1.2 h
===========================  ==========================  ==================

``common.cached_uvdoc_geometry`` already gets the 4,032 figure *within one
process*, but that cache dies with the run and is paid again by every fresh
worker and every fresh epoch-0. This script makes it durable.

Layout
------
The layout is NOT restated here. It is
:data:`~train.doc_scanner.common.UVDOC_CACHE_GEOMETRY_DIR`,
:data:`~train.doc_scanner.common.UVDOC_CACHE_RENDER_DIR` and the ``.npz`` key
constants beside them, and this module builds its paths with
:func:`~train.doc_scanner.common.uvdoc_cache_dir` -- the same function the
reader uses -- so writer and reader cannot drift apart::

    <cache-root>/<S>x<S>/manifest.json
    <cache-root>/<S>x<S>/geometry/<geom_name>.npz   f_gt, g, mask
    <cache-root>/<S>x<S>/render/<sample_id>.npz     image, geom_name

The join key lives in the RENDER file. There is deliberately no shared index:
adding a render is one new file, so an interrupted run leaves a smaller staged
corpus and never an index that promises files which are not there.

Discipline (the same as ``prepare_doc_scanner_data.py``)
--------------------------------------------------------
* **Additive only.** Nothing is deleted, moved or truncated -- not the 27.5 GB
  archive, not a previously staged sidecar, not anything under ``results/``.
* **Idempotent.** An existing sidecar is skipped, not rewritten; a second run
  over the same ``--limit`` writes zero bytes and reports it.
* **Resumable.** Every file is written to a ``.tmp`` sibling and ``os.replace``
  d into place, so a kill mid-write leaves either the old file or the new one,
  never half of one. A ``.tmp`` left behind by a kill is overwritten by the
  next run.
* **Refuses rather than fills the volume**, reusing
  ``prepare_doc_scanner_data.check_disk_budget``.
* **Explicit RNG.** ``np.random.default_rng(seed)`` per geometry; the global
  legacy numpy RNG is never touched.

Public surface:
    * :func:`stage_samples` -- the whole job; returns a :class:`StagingReport`.
    * :func:`main` -- the CLI (``--limit``, ``--dry-run``, ...).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from dl_techniques.datasets.document_rectification import (
    UVDocError,
    UVDocSource,
    is_unusable,
    load_geometry,
    load_image,
)
from dl_techniques.utils.logger import logger
from train.doc_scanner.common import (
    DEFAULT_UVDOC_CACHE_ROOT,
    DEFAULT_UVDOC_ROOT,
    UVDOC_CACHE_F_GT_KEY,
    UVDOC_CACHE_G_KEY,
    UVDOC_CACHE_GEOM_NAME_KEY,
    UVDOC_CACHE_GEOMETRY_DIR,
    UVDOC_CACHE_IMAGE_KEY,
    UVDOC_CACHE_MANIFEST,
    UVDOC_CACHE_MASK_KEY,
    UVDOC_CACHE_RENDER_DIR,
    UVDOC_CACHE_SCHEMA,
    DocScannerDataError,
)
from train.doc_scanner.prepare_doc_scanner_data import (
    DiskBudgetError,
    check_disk_budget,
)

# DECISION plan-2026-09-10T065432-05fcb6dd/D-049: the sidecar LAYOUT and its
# READER live in `common.py`, not here, and this module imports them. Do NOT
# move `uvdoc_cache_dir` / `read_uvdoc_sidecar` into this file "next to the
# writer": `prepare_doc_scanner_data.py` already imports from `common.py`, so
# `common.py` importing a train script back would close an import cycle whose
# behaviour depends on import order -- the same cycle D-015 broke inside the
# model package. Putting the reader in `common.py` also makes the layout have
# exactly ONE definition (these constants), which is the only thing keeping a
# writer and a reader in different modules from drifting apart. See
# decisions.md D-049.

__all__ = [
    "DEFAULT_MIN_FREE_GB",
    "StagingReport",
    "build_parser",
    "main",
    "stage_samples",
]


#: Free-space floor, in GB, the staging volume may not cross. Lower than
#: ``prepare_doc_scanner_data``'s 100 because the sidecars are a fraction of the
#: archive: at 288x288 a geometry triple is 1.66 MB and a render is 0.24 MB, so
#: the WHOLE corpus is about 11.5 GB against the archive's 27.5.
DEFAULT_MIN_FREE_GB: float = 20.0

_GB: int = 1000 ** 3

#: Bytes one geometry sidecar occupies at ``S x S``: ``f_gt`` and ``g`` are
#: ``(S, S, 2)`` float32 and ``mask`` is ``(S, S, 1)`` uint8. Used only for the
#: pre-flight disk estimate, so it is deliberately an upper bound (npz stores
#: uncompressed, plus a small per-member header).
_GEOMETRY_BYTES_PER_PIXEL: int = 2 * 2 * 4 + 1

#: Bytes one render sidecar occupies at ``S x S``: ``(S, S, 3)`` uint8.
_RENDER_BYTES_PER_PIXEL: int = 3

#: Slack added to the estimate for npz/zip headers and the ``geom_name`` entry.
_SIDECAR_OVERHEAD_BYTES: int = 4096

_PROGRESS_EVERY: int = 25


@dataclass
class StagingReport:
    """What one :func:`stage_samples` call did. Every field is a count.

    Attributes:
        directory: The size-scoped sidecar directory, as a string.
        requested: Renders on the worklist after ``--limit``.
        renders_written: Render sidecars newly written.
        renders_skipped: Render sidecars already present.
        geometries_written: Geometry sidecars newly written.
        geometries_skipped: Geometry sidecars already present.
        densifications: Calls to ``load_geometry``. **This is the number the
            per-geometry cache exists to hold down**: it must equal
            ``geometries_written``, never ``renders_written``.
        distinct_geometries: Distinct ``geom_name`` values on the worklist.
        unusable: Geometries staged despite missing ``uvdoc``'s accuracy bound.
            They are staged anyway, because the in-process training path
            (``strict=False``) accepts them too and the two corpora forms must
            not differ; the count exists so the tail is REPORTED.
        failed: Renders skipped because a member would not read.
        bytes_written: Sum of the sizes of the files this call created.
        seconds: Wall clock.
        dry_run: Whether anything was written at all.
    """

    directory: str
    requested: int = 0
    renders_written: int = 0
    renders_skipped: int = 0
    geometries_written: int = 0
    geometries_skipped: int = 0
    densifications: int = 0
    distinct_geometries: int = 0
    unusable: int = 0
    failed: int = 0
    bytes_written: int = 0
    seconds: float = 0.0
    dry_run: bool = False
    unusable_names: List[str] = field(default_factory=list)
    failed_ids: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Atomic, additive writes
# ---------------------------------------------------------------------------


def _write_npz_atomically(path: Path, **arrays: object) -> int:
    """Write one ``.npz`` via a ``.tmp`` sibling and ``os.replace``.

    Interface contract -- two callers here, the geometry writer and the render
    writer:

    * Parameters: destination ``path``; the arrays as keyword arguments.
    * Returns the size in bytes of the file now at ``path``.
    * Failure mode: on ANY exception the temporary file is removed and the
      exception propagates, so a failed write never leaves a partial
      ``.npz`` that the reader would happily open and find one array short.

    :param path: Destination.
    :type path: Path
    :param arrays: Named arrays.
    :return: Size of the written file, in bytes.
    :rtype: int
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(temporary, "wb") as handle:
            np.savez(handle, **arrays)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path.stat().st_size


def _quantise_image(image: np.ndarray) -> np.ndarray:
    """``[0, 1]`` float32 render to the uint8 it came from.

    :param image: ``(H, W, 3)`` float32 in ``[0, 1]``.
    :type image: np.ndarray
    :return: ``(H, W, 3)`` uint8.
    :rtype: np.ndarray
    """
    # DECISION plan-2026-09-10T065432-05fcb6dd/D-050: the staged render is
    # EIGHT-BIT, and that is the one place a sidecar is not bit-identical to
    # what `load_image` returns. Do NOT "fix" this by storing float32 to make
    # the round-trip guard exact: the source member is a uint8 PNG, the only
    # reason `load_image` hands back floats at all is the bilinear resample, and
    # float32 storage quadruples the corpus (20,000 renders: 5.0 GB -> 19.9 GB)
    # to preserve at most half a grey level. The maps are NOT quantised --
    # `f_gt` and `g` are absolute pixel coordinates whose whole supervision
    # signal is sub-pixel, so 8 bits there would be a real defect. The guard
    # that keeps this honest asserts BOTH directions: the staged bytes read back
    # exactly, AND they differ from `load_image` by <= 1/255 while NOT being
    # equal to it. See decisions.md D-050.
    return np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def _write_manifest(
        directory: Path,
        size: Tuple[int, int],
        seed: int,
        uvdoc_root: str,
) -> None:
    """Create ``manifest.json`` if absent; validate it if present.

    :param directory: The size-scoped sidecar directory.
    :type directory: Path
    :param size: ``(height, width)`` the sidecars are stored at.
    :type size: Tuple[int, int]
    :param seed: Seed forwarded to ``load_geometry``'s cross-validation.
    :type seed: int
    :param uvdoc_root: The archive or directory the sidecars derive from.
    :type uvdoc_root: str
    :raises DocScannerDataError: If a manifest exists and disagrees about the
        schema or the stored size. It is NOT an error for the recorded ``seed``
        or ``uvdoc_root`` to differ -- the seed drives only the held-out
        cross-validation and does not affect the emitted maps.
    """
    height, width = int(size[0]), int(size[1])
    path = directory / UVDOC_CACHE_MANIFEST
    if path.is_file():
        existing = json.loads(path.read_text())
        if existing.get("schema") != UVDOC_CACHE_SCHEMA:
            raise DocScannerDataError(
                f"{str(path)!r} records sidecar schema "
                f"{existing.get('schema')!r}; this writer emits schema "
                f"{UVDOC_CACHE_SCHEMA}. Stage into a different --cache-root "
                "rather than mixing two layouts in one directory."
            )
        if (
                int(existing.get("height", -1)) != height
                or int(existing.get("width", -1)) != width
        ):
            raise DocScannerDataError(
                f"{str(path)!r} records "
                f"{existing.get('height')!r}x{existing.get('width')!r}, not "
                f"{height}x{width}."
            )
        return
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": UVDOC_CACHE_SCHEMA,
        "height": height,
        "width": width,
        "seed": int(seed),
        "uvdoc_root": str(uvdoc_root),
        "image_dtype": "uint8",
        "map_dtype": "float32",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, path)


# ---------------------------------------------------------------------------
# The job
# ---------------------------------------------------------------------------


def _worklist(
        source: UVDocSource,
        limit: Optional[int],
        limit_geometries: Optional[int] = None,
) -> Tuple[List[str], Dict[str, str]]:
    """Render ids to stage and their ``sample_id -> geom_name`` mapping.

    # DECISION plan-2026-09-10T065432-05fcb6dd/D-052: ``--limit`` alone selects
    # a PREFIX OF RENDER IDS, and on the real corpus that prefix shares NO
    # geometries at all. MEASURED over all 20,000 renders: 4,032 geometries,
    # 5 renders each (160 have 4), but the five renders of one geometry sit
    # about 4,032 ids apart, so the first 100 / 400 / 1,000 / 4,032 ids span
    # exactly 100 / 400 / 1,000 / 4,032 DISTINCT geometries -- one apiece. A
    # small ``--limit`` subset therefore pays one densification per render and
    # the per-geometry layout saves nothing on it. That is not a defect in the
    # layout (at 20,000 it is still 4,032 inversions), but it does mean
    # ``--limit`` is the WRONG knob for "stage a cheap, representative subset".
    # ``--limit-geometries`` is the right one: it picks whole geometries and
    # stages all ~5 renders of each, so a subset has the corpus's real fan-out
    # AND costs one fifth of the interpolation. Do NOT "simplify" the two into
    # one flag. Building the reverse mapping costs a read of every render's
    # metadata JSON (~100 s over the zip), which is why it happens only when
    # the flag is used. See decisions.md D-052.

    :param source: An open source.
    :type source: UVDocSource
    :param limit: Cap on renders, or ``None`` for all of them.
    :type limit: Optional[int]
    :param limit_geometries: Keep only renders belonging to the first N
        geometries (sorted). Applied BEFORE ``limit``.
    :type limit_geometries: Optional[int]
    :return: ``(sample_ids, geometry_by_sample)``. A render whose metadata does
        not read is dropped from BOTH, with a warning -- it is one sample out
        of twenty thousand, not a reason to abort a multi-hour job.
    :rtype: Tuple[List[str], Dict[str, str]]
    """
    ids = sorted(source.sample_ids())
    wanted: Optional[set] = None
    if limit_geometries is not None:
        logger.info(
            "UVDoc sidecars: reading every render's metadata to group %d "
            "geometries (the id order does not group them).",
            int(limit_geometries),
        )
        wanted = set(sorted(source.geometry_names())[: int(limit_geometries)])
    elif limit is not None:
        ids = ids[: int(limit)]

    mapping: Dict[str, str] = {}
    kept: List[str] = []
    for sample_id in ids:
        try:
            geometry = source.geometry_for_sample(sample_id)
        except (UVDocError, ValueError) as error:
            logger.warning(
                "UVDoc render %s has no readable geom_name (%s); skipping it.",
                sample_id, error,
            )
            continue
        if wanted is not None and geometry not in wanted:
            continue
        mapping[sample_id] = geometry
        kept.append(sample_id)
        if limit is not None and len(kept) >= int(limit):
            break
    return kept, mapping


def stage_samples(
        uvdoc_root: str = DEFAULT_UVDOC_ROOT,
        cache_root: str = DEFAULT_UVDOC_CACHE_ROOT,
        *,
        size: Tuple[int, int] = (288, 288),
        seed: int = 42,
        limit: Optional[int] = None,
        limit_geometries: Optional[int] = None,
        dry_run: bool = False,
        min_free_bytes: int = int(DEFAULT_MIN_FREE_GB * _GB),
) -> StagingReport:
    """Stage ``(image, f_gt, g, mask)`` sidecars for up to ``limit`` renders.

    Interface contract -- two callers, :func:`main` and the guard suite:

    * Parameters: ``uvdoc_root`` (``UVDoc_final.zip`` or a staged directory),
      ``cache_root``, and the knobs above.
    * Returns a :class:`StagingReport`; ``densifications`` is the measured
      count of ``load_geometry`` calls and is the number the whole per-geometry
      layout exists to hold down.
    * Failure mode: :class:`DiskBudgetError` before anything is written if the
      estimate would cross the floor; :class:`UVDocError` only if the corpus
      itself cannot be listed. A single unreadable render is counted in
      ``failed`` and skipped.

    :param uvdoc_root: Archive or staged UVDoc directory.
    :type uvdoc_root: str
    :param cache_root: Root the size-scoped directory is created under.
    :type cache_root: str
    :param size: ``(height, width)`` the sidecars are stored at. See D-051 for
        why this is a PAIR and not the square ``image_size`` int the config
        carries.
    :type size: Tuple[int, int]
    :param seed: Root seed; geometry ``i`` uses ``default_rng(seed + i)``.
    :type seed: int
    :param limit: Cap on renders staged this call, or ``None`` for all.
    :type limit: Optional[int]
    :param limit_geometries: Stage whole geometries rather than a prefix of
        render ids -- the right knob for a cheap representative subset, and
        the only one under which a subset exercises the per-geometry cache at
        all. See D-052.
    :type limit_geometries: Optional[int]
    :param dry_run: Count and estimate, write nothing.
    :type dry_run: bool
    :param min_free_bytes: Free-space floor on the cache volume.
    :type min_free_bytes: int
    :return: The report.
    :rtype: StagingReport
    :raises DiskBudgetError: If staging would cross the floor.
    :raises UVDocError: If the corpus cannot be opened or listed.
    """
    # DECISION plan-2026-09-10T065432-05fcb6dd/D-051: `size` is a (height,
    # width) PAIR even though `DocScannerTrainingConfig` is square-only and
    # `main` always passes (S, S). Do NOT collapse it back to one `image_size`
    # int. MEASURED: the shared `backward_map_convention` instrument checks a
    # channel swap by RANGE (x must live in [0, W), y in [0, H)), so at H == W
    # it accepts a swapped f_gt on ALL of this suite's stand-in geometries and
    # rejects it on all of them at 49x83. The pair is what lets the guard suite
    # stage non-square and have any power at all; the square path is unchanged.
    # Separately pinned there, and NOT fixed by non-squareness: the instrument
    # accepts `g` in `f_gt`'s place at every size -- direction is pinned only
    # by the elementwise comparison against `load_geometry`. See decisions.md
    # D-051.
    started = time.monotonic()
    height, width = int(size[0]), int(size[1])
    directory = Path(cache_root) / f"{height}x{width}"
    report = StagingReport(directory=str(directory), dry_run=bool(dry_run))

    with UVDocSource(uvdoc_root) as source:
        sample_ids, geometry_by_sample = _worklist(
            source, limit, limit_geometries
        )
        report.requested = len(sample_ids)
        wanted_geometries = sorted(set(geometry_by_sample.values()))
        report.distinct_geometries = len(wanted_geometries)

        geometry_dir = directory / UVDOC_CACHE_GEOMETRY_DIR
        render_dir = directory / UVDOC_CACHE_RENDER_DIR
        missing_geometries = [
            name for name in wanted_geometries
            if not (geometry_dir / f"{name}.npz").is_file()
        ]
        missing_renders = [
            sample_id for sample_id in sample_ids
            if not (render_dir / f"{sample_id}.npz").is_file()
        ]
        report.geometries_skipped = len(wanted_geometries) - len(
            missing_geometries
        )
        report.renders_skipped = len(sample_ids) - len(missing_renders)

        pixels = height * width
        needed = (
            len(missing_geometries)
            * (pixels * _GEOMETRY_BYTES_PER_PIXEL + _SIDECAR_OVERHEAD_BYTES)
            + len(missing_renders)
            * (pixels * _RENDER_BYTES_PER_PIXEL + _SIDECAR_OVERHEAD_BYTES)
        )
        check_disk_budget(directory, needed, min_free_bytes)

        if dry_run:
            logger.info(
                "UVDoc sidecars (dry run): %d renders over %d geometries; "
                "%d geometry + %d render sidecars missing, about %.2f GB.",
                report.requested, report.distinct_geometries,
                len(missing_geometries), len(missing_renders), needed / _GB,
            )
            report.seconds = time.monotonic() - started
            return report

        _write_manifest(directory, (height, width), seed, uvdoc_root)

        # -- geometries first: this loop, and ONLY this loop, densifies. -----
        for offset, name in enumerate(missing_geometries):
            try:
                geometry = load_geometry(
                    source,
                    name,
                    np.random.default_rng(int(seed) + offset),
                    size=(height, width),
                    strict=False,
                )
            except UVDocError as error:
                logger.warning(
                    "UVDoc geometry %s failed to densify (%s); its renders "
                    "will be skipped.", name, error,
                )
                continue
            report.densifications += 1
            if is_unusable(geometry.quality):
                report.unusable += 1
                report.unusable_names.append(name)
            report.bytes_written += _write_npz_atomically(
                geometry_dir / f"{name}.npz",
                **{
                    UVDOC_CACHE_F_GT_KEY: geometry.f_gt,
                    UVDOC_CACHE_G_KEY: geometry.g,
                    UVDOC_CACHE_MASK_KEY: geometry.mask.astype(np.uint8),
                },
            )
            report.geometries_written += 1
            if (offset + 1) % _PROGRESS_EVERY == 0:
                logger.info(
                    "UVDoc sidecars: %d/%d geometries densified (%.1f s).",
                    offset + 1, len(missing_geometries),
                    time.monotonic() - started,
                )

        # -- then renders: pure decode + resample, no interpolation. ---------
        for position, sample_id in enumerate(missing_renders):
            name = geometry_by_sample[sample_id]
            if not (geometry_dir / f"{name}.npz").is_file():
                report.failed += 1
                report.failed_ids.append(sample_id)
                continue
            try:
                image = load_image(source, sample_id, size=(height, width))
            except UVDocError as error:
                logger.warning(
                    "UVDoc render %s did not read (%s); skipping it.",
                    sample_id, error,
                )
                report.failed += 1
                report.failed_ids.append(sample_id)
                continue
            report.bytes_written += _write_npz_atomically(
                render_dir / f"{sample_id}.npz",
                **{
                    UVDOC_CACHE_IMAGE_KEY: _quantise_image(image),
                    UVDOC_CACHE_GEOM_NAME_KEY: np.array(name),
                },
            )
            report.renders_written += 1
            if (position + 1) % _PROGRESS_EVERY == 0:
                logger.info(
                    "UVDoc sidecars: %d/%d renders staged (%.1f s).",
                    position + 1, len(missing_renders),
                    time.monotonic() - started,
                )

    report.seconds = time.monotonic() - started
    return report


def log_report(report: StagingReport) -> None:
    """Log one report at a level that matches it.

    :param report: What :func:`stage_samples` returned.
    :type report: StagingReport
    """
    logger.info("UVDoc sidecars under %s:", report.directory)
    logger.info(
        "  %d renders requested over %d distinct geometries", report.requested,
        report.distinct_geometries,
    )
    logger.info(
        "  wrote %d geometry + %d render sidecars (%.2f GB) in %.1f s; "
        "skipped %d + %d already present",
        report.geometries_written, report.renders_written,
        report.bytes_written / _GB, report.seconds,
        report.geometries_skipped, report.renders_skipped,
    )
    logger.info(
        "  %d densifications for %d renders (the per-geometry cache saved %d)",
        report.densifications, report.requested,
        max(0, report.requested - report.densifications),
    )
    if report.unusable:
        logger.warning(
            "  %d geometries missed the densification bound and were staged "
            "anyway (the training path accepts them too): %s",
            report.unusable, report.unusable_names[:10],
        )
    if report.failed:
        logger.warning(
            "  %d renders did not read and were skipped: %s",
            report.failed, report.failed_ids[:10],
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """The argument parser. Constructing it allocates nothing but memory.

    :return: The parser.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="stage_uvdoc_samples",
        description=(
            "Precompute UVDoc (image, f_gt, g, mask) sidecars so a training "
            "run does not pay ~1.06 s of scattered-data interpolation per "
            "geometry. Additive only, idempotent, resumable."
        ),
    )
    parser.add_argument(
        "--uvdoc-root", type=str, default=DEFAULT_UVDOC_ROOT,
        help=f"UVDoc_final.zip or a staged directory (default: "
             f"{DEFAULT_UVDOC_ROOT})",
    )
    parser.add_argument(
        "--cache-root", type=str, default=DEFAULT_UVDOC_CACHE_ROOT,
        help=f"sidecar root (default: {DEFAULT_UVDOC_CACHE_ROOT})",
    )
    parser.add_argument(
        "--image-size", type=int, default=288,
        help="square sample size the sidecars are stored at (default: 288)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help=(
            "stage at most N renders (sorted by id). The default stages the "
            "whole 20,000-render corpus, which is about 1.2 h and 11.5 GB."
        ),
    )
    parser.add_argument(
        "--limit-geometries", type=int, default=None,
        help=(
            "stage every render of the first N geometries. Prefer this to "
            "--limit for a subset: UVDoc's render ids do NOT group by "
            "geometry, so a --limit prefix spans one geometry per render and "
            "the per-geometry cache saves nothing on it (measured, D-052)."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="root seed for load_geometry's held-out cross-validation",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="report what would be staged and how much space it needs",
    )
    parser.add_argument(
        "--min-free-gb", type=float, default=DEFAULT_MIN_FREE_GB,
        help=(
            "refuse to stage if the volume would drop below this "
            f"(default: {DEFAULT_MIN_FREE_GB})"
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point.

    The FIRST statement parses ``argv``, so ``--help`` allocates nothing else --
    no directory, no archive handle.

    :param argv: Command line, or ``None`` for ``sys.argv[1:]``.
    :type argv: Optional[Sequence[str]]
    :return: ``0`` on success (including a dry run and a no-op re-run); ``1``
        if the corpus could not be read or the disk budget refused.
    :rtype: int
    """
    args = build_parser().parse_args(argv)
    try:
        report = stage_samples(
            args.uvdoc_root,
            args.cache_root,
            size=(args.image_size, args.image_size),
            seed=args.seed,
            limit=args.limit,
            limit_geometries=args.limit_geometries,
            dry_run=args.dry_run,
            min_free_bytes=int(args.min_free_gb * _GB),
        )
    except (UVDocError, DiskBudgetError, DocScannerDataError) as error:
        logger.error("UVDoc sidecar staging refused: %s", error)
        return 1
    log_report(report)
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
