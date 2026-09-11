"""Stage the UVDoc corpus for DocScanner training: download, verify, mark.

One corpus, one archive, one directory::

    <root>/doc_scanner/uvdoc/
        UVDoc_final.zip         <- the archive itself, 27.5 GB
        UVDoc_final.zip.ok      <- the completion marker, written LAST
        UVDoc_final.zip.part    <- only while a download is in flight

Nothing is extracted. :class:`dl_techniques.datasets.document_rectification.
UVDocSource` reads members straight out of the zip, so the staged archive IS
the corpus; unpacking 182,492 members (half of them ``__MACOSX/`` resource
forks) would double the disk cost for nothing.

Safety properties
-----------------
* **Additive only.** Every write lands under ``<root>/doc_scanner/uvdoc/``.
  This script deletes nothing except a ``.part`` file it created in the same
  call, and it never touches repo-root ``results/``.
* **Refuses rather than fills the volume.** Free space is checked against
  ``--min-free-gb`` (default 100) *plus* the archive's own budget, before a
  single byte is fetched.
* **Verifies the URL first.** A HEAD precedes any GET; a dead host is reported
  by name and HTTP code rather than producing a zero-byte file.
* **Idempotent.** A second run over a staged corpus makes ZERO HTTP requests --
  not a HEAD, not a GET -- and writes nothing.
* **Resumable.** An interrupted download leaves ``UVDoc_final.zip.part``, and
  the next run continues it with a ``Range:`` request. The host was measured
  to answer ``accept-ranges: bytes`` on 2026-09-10.
* **Completeness is proven by DECODING every member**, never by
  ``Content-Length``. Only after that does the ``.part`` become the final
  ``.zip``, and only after *that* is the ``.ok`` marker written. A partial
  download therefore leaves an honest empty slot -- never a corpus that
  reports success while being silently truncated.

Examples::

    # say what would happen; touch nothing
    python -m train.doc_scanner.prepare_doc_scanner_data --dry-run

    # stage it (27.5 GB); safe to re-run, safe to interrupt and re-run
    python -m train.doc_scanner.prepare_doc_scanner_data
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

from dl_techniques.utils.logger import logger
from train.doc_scanner.common import (
    DEFAULT_UVDOC_ROOT,
    DocScannerDataError,
    MissingTrainingDataError,
)

__all__ = [
    "ArchiveError",
    "DEFAULT_DATASET_ROOT",
    "DEFAULT_MIN_FREE_GB",
    "DiskBudgetError",
    "DiskState",
    "StageResult",
    "UVDOC_EXPECTED_BYTES",
    "UVDOC_EXPECTED_MEMBERS",
    "UVDOC_URL",
    "UrlStatus",
    "archive_path",
    "build_parser",
    "check_disk_budget",
    "main",
    "marker_path",
    "part_path",
    "read_disk_state",
    "read_marker",
    "require_staged_uvdoc",
    "stage_uvdoc",
    "uvdoc_dir",
    "verify_archive",
    "verify_url",
]

# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------

# DECISION plan-2026-09-10T065432-05fcb6dd/D-047
# The four path constants below are DERIVED from
# ``common.DEFAULT_UVDOC_ROOT`` rather than typed out again. Do NOT "simplify"
# this by writing the literals here: the trainer resolves the corpus through
# ``common.DEFAULT_UVDOC_ROOT`` and this script WRITES it, so a literal in each
# file would be a hand-maintained lockstep invariant -- edit one, and the
# staging script silently fills a directory the trainer never looks in, with no
# error anywhere. One source of truth makes that class of drift unrepresentable.
# ``test_the_staged_path_is_exactly_the_path_the_trainer_reads`` pins it.
# See D-047.
_DEFAULT_ARCHIVE: Path = Path(DEFAULT_UVDOC_ROOT)

UVDOC_FILENAME: str = _DEFAULT_ARCHIVE.name
"""``UVDoc_final.zip`` -- the name the archive is stored under."""

UVDOC_DIRNAME: str = _DEFAULT_ARCHIVE.parent.name
"""``uvdoc`` -- the per-corpus directory."""

STAGING_DIRNAME: str = _DEFAULT_ARCHIVE.parent.parent.name
"""``doc_scanner`` -- the single directory this script owns under the root."""

DEFAULT_DATASET_ROOT: Path = _DEFAULT_ARCHIVE.parent.parent.parent
"""The shared dataset volume; every sibling corpus is a top-level dir here."""

UVDOC_URL: str = "https://igl.ethz.ch/projects/uvdoc/UVDoc_final.zip"
"""ETH Zurich IGL's own host. No auth, no registration wall (F-13)."""

UVDOC_EXPECTED_BYTES: int = 29_538_315_521
"""``Content-Length`` measured on 2026-09-10 (27.5 GB). Used for the dry-run
budget and to recognise a SHORT body; never used as the completeness proof."""

UVDOC_EXPECTED_MEMBERS: int = 182_492
"""Members counted on 2026-09-10. Exactly half are ``__MACOSX/`` resource
forks. Reported for information; the CRC pass is what proves completeness."""

DEFAULT_MIN_FREE_GB: float = 100.0
"""Refuse to stage if the volume would drop below this, matching the DocRes
staging precedent (``train/doc_res/prepare_doc_res_data.py``)."""

DEFAULT_TIMEOUT_S: float = 60.0

_USER_AGENT: str = "dl_techniques-doc_scanner-prepare/1.0"
_BASE_HEADERS: Dict[str, str] = {"User-Agent": _USER_AGENT}

_GB: int = 1000 ** 3
_CHUNK_BYTES: int = 1 << 22
_PROGRESS_EVERY_BYTES: int = 1 << 30


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DiskBudgetError(DocScannerDataError):
    """The volume has no room for the archive, with the shortfall named."""


class ArchiveError(DocScannerDataError):
    """The archive on disk is truncated, corrupt or not a zip at all."""


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def uvdoc_dir(dataset_root: Path) -> Path:
    """``<dataset_root>/doc_scanner/uvdoc``.

    :param dataset_root: The dataset volume.
    :type dataset_root: Path
    :return: The corpus directory. Not created by this call.
    :rtype: Path
    """
    return Path(dataset_root) / STAGING_DIRNAME / UVDOC_DIRNAME


def archive_path(dataset_root: Path) -> Path:
    """The final archive path, ``.../uvdoc/UVDoc_final.zip``.

    :param dataset_root: The dataset volume.
    :type dataset_root: Path
    :return: The path a completed download ends up at.
    :rtype: Path
    """
    return uvdoc_dir(dataset_root) / UVDOC_FILENAME


def marker_path(dataset_root: Path) -> Path:
    """The completion marker, ``<archive>.ok``. Written LAST, or not at all.

    :param dataset_root: The dataset volume.
    :type dataset_root: Path
    :return: The marker path.
    :rtype: Path
    """
    return archive_path(dataset_root).with_name(UVDOC_FILENAME + ".ok")


def part_path(dataset_root: Path) -> Path:
    """The in-flight download, ``<archive>.part``.

    A partial body NEVER carries the final name, so no later run can mistake
    it for a complete corpus.

    :param dataset_root: The dataset volume.
    :type dataset_root: Path
    :return: The ``.part`` path.
    :rtype: Path
    """
    return archive_path(dataset_root).with_name(UVDOC_FILENAME + ".part")


# ---------------------------------------------------------------------------
# Disk budget
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiskState:
    """A point-in-time reading of the staging volume."""

    total_bytes: int
    used_bytes: int
    free_bytes: int


def read_disk_state(path: Path) -> DiskState:
    """Read free space for the filesystem holding ``path``.

    Walks up to the nearest existing ancestor, so this works before the
    staging directory exists.

    :param path: Any path on the target volume.
    :type path: Path
    :return: The reading.
    :rtype: DiskState
    """
    probe = Path(path)
    while not probe.exists():
        if probe.parent == probe:
            break
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    return DiskState(usage.total, usage.used, usage.free)


def check_disk_budget(
    dataset_root: Path, needed_bytes: int, min_free_bytes: int
) -> DiskState:
    """Refuse rather than fill the volume.

    :param dataset_root: The staging volume.
    :type dataset_root: Path
    :param needed_bytes: What the download would consume.
    :type needed_bytes: int
    :param min_free_bytes: The floor free space must never cross.
    :type min_free_bytes: int
    :return: The reading that was taken.
    :rtype: DiskState
    :raises DiskBudgetError: If staging would cross the floor. The message
        names the shortfall in bytes and in GB.
    """
    state = read_disk_state(dataset_root)
    remaining = state.free_bytes - needed_bytes
    if remaining < min_free_bytes:
        shortfall = min_free_bytes - remaining
        raise DiskBudgetError(
            f"refusing to stage UVDoc into {dataset_root}: it needs "
            f"{needed_bytes:,} B ({needed_bytes / _GB:.2f} GB) and only "
            f"{state.free_bytes:,} B ({state.free_bytes / _GB:.2f} GB) are "
            f"free, which would leave {remaining:,} B below the "
            f"{min_free_bytes:,} B ({min_free_bytes / _GB:.2f} GB) floor -- "
            f"a shortfall of {shortfall:,} B ({shortfall / _GB:.2f} GB). "
            "Free space or lower --min-free-gb; nothing was downloaded."
        )
    return state


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UrlStatus:
    """The result of verifying the URL before any body is fetched."""

    url: str
    ok: bool
    status_code: int
    content_length: Optional[int]
    accept_ranges: bool = False
    detail: str = ""


def _http_head(url: str, timeout: float) -> Tuple[int, Optional[int], bool, str]:
    """HEAD a URL, following redirects.

    :param url: The URL to probe.
    :type url: str
    :param timeout: Per-request timeout in seconds.
    :type timeout: float
    :return: ``(status_code, content_length_or_None, accepts_ranges, detail)``.
        A transport failure is reported as status ``0`` with the exception
        text as ``detail`` -- a dead host is a RESULT here, not an exception.
    :rtype: Tuple[int, Optional[int], bool, str]
    """
    import requests  # local: keeps --help free of a network stack import

    try:
        resp = requests.head(
            url, allow_redirects=True, timeout=timeout, headers=_BASE_HEADERS
        )
        length = resp.headers.get("Content-Length")
        ranges = str(resp.headers.get("Accept-Ranges", "")).lower() == "bytes"
        return resp.status_code, int(length) if length else None, ranges, ""
    except Exception as exc:  # noqa: BLE001 - report the dead host, do not raise
        return 0, None, False, f"{type(exc).__name__}: {exc}"


def verify_url(url: str, timeout: float) -> UrlStatus:
    """Check the URL is live before a single byte is fetched.

    :param url: The URL to probe.
    :type url: str
    :param timeout: Per-request timeout in seconds.
    :type timeout: float
    :return: The status; ``ok`` is false for any non-2xx or transport failure.
    :rtype: UrlStatus
    """
    code, length, ranges, detail = _http_head(url, timeout)
    return UrlStatus(
        url=url,
        ok=200 <= code < 300,
        status_code=code,
        content_length=length,
        accept_ranges=ranges,
        detail=detail,
    )


@contextmanager
def _open_stream(
    url: str, *, timeout: float, headers: Dict[str, str]
) -> Iterator[Any]:
    """Open a streaming GET.

    Interface contract -- this is the ONLY seam in this module that reaches
    the network for a body, and it is deliberately thin so that
    :func:`_download` (resume arithmetic, ``Range`` construction, 206-vs-200
    handling) stays under test with a fake server. Yields an object exposing
    ``status_code``, a ``headers`` mapping and ``iter_content(chunk_size)``.

    :param url: The URL to GET.
    :type url: str
    :param timeout: Per-request timeout in seconds.
    :type timeout: float
    :param headers: Request headers, including ``Range`` when resuming.
    :type headers: Dict[str, str]
    :return: A context manager over the response.
    :rtype: Iterator[Any]
    """
    import requests

    with requests.get(
        url, stream=True, allow_redirects=True, timeout=timeout, headers=headers
    ) as resp:
        yield resp


def _download(
    url: str,
    dest_part: Path,
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> Tuple[int, int]:
    """Stream a URL into ``dest_part``, RESUMING an existing partial file.

    An existing ``.part`` is continued with ``Range: bytes=<size>-``. If the
    host answers ``206`` the body is APPENDED; if it answers ``200`` it has
    ignored the range and is sending the whole file, so the partial is
    restarted from zero rather than concatenated into a corrupt double body.

    :param url: The URL to fetch.
    :type url: str
    :param dest_part: The ``.part`` path to write or continue.
    :type dest_part: Path
    :param timeout: Per-request timeout in seconds.
    :type timeout: float
    :return: ``(bytes_written_this_call, total_bytes_on_disk)``.
    :rtype: Tuple[int, int]
    :raises ArchiveError: If the host answers 4xx/5xx. The ``.part`` is left
        exactly as it was so a later run can still resume it.
    """
    dest_part.parent.mkdir(parents=True, exist_ok=True)
    resume_from = dest_part.stat().st_size if dest_part.exists() else 0
    headers = dict(_BASE_HEADERS)
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"

    written = 0
    with _open_stream(url, timeout=timeout, headers=headers) as resp:
        status = int(resp.status_code)
        if status >= 400:
            raise ArchiveError(
                f"{url} answered HTTP {status} for a "
                f"{'ranged ' if resume_from else ''}GET; nothing was written. "
                f"{dest_part.name} is left at {resume_from:,} B so a later run "
                "can resume it."
            )
        if resume_from and status == 206:
            mode = "ab"
            logger.info(
                "DocScanner/UVDoc: resuming at byte %d (HTTP 206)", resume_from
            )
        else:
            if resume_from:
                logger.warning(
                    "DocScanner/UVDoc: asked for bytes=%d- and the host "
                    "answered HTTP %d, not 206 -- it is sending the whole "
                    "file, so the %d B partial is being restarted rather than "
                    "appended to.",
                    resume_from, status, resume_from,
                )
            resume_from = 0
            mode = "wb"

        next_report = _PROGRESS_EVERY_BYTES
        with open(dest_part, mode) as handle:
            for chunk in resp.iter_content(chunk_size=_CHUNK_BYTES):
                if not chunk:
                    continue
                handle.write(chunk)
                written += len(chunk)
                if written >= next_report:
                    next_report += _PROGRESS_EVERY_BYTES
                    logger.info(
                        "DocScanner/UVDoc: %.2f GB written this run (%.2f GB "
                        "on disk)",
                        written / _GB, (resume_from + written) / _GB,
                    )
    return written, resume_from + written


# ---------------------------------------------------------------------------
# The completeness proof
# ---------------------------------------------------------------------------


# DECISION plan-2026-09-10T065432-05fcb6dd/D-048
# Completeness is proven by DECODING and CRC-checking EVERY member
# (``ZipFile.testzip()``), and only then is the ``.part`` renamed and the
# ``.ok`` marker written. Do NOT replace this with the obvious
# ``Content-Length`` comparison, and do NOT weaken it to "the central
# directory parses". A truncated zip whose central directory survived reports
# all 182,492 entries happily -- the entry list lives at the END of the file,
# so the very failure mode a resumable multi-GB download has (a body that
# stopped early) is invisible to both a size check and an ``infolist()`` walk
# unless the members are actually inflated. ``testzip()`` is the stdlib
# routine that already does exactly this; writing a member loop by hand here
# would be a weaker copy of it. See D-048.
def verify_archive(path: Path) -> int:
    """Prove a zip is complete by decoding and CRC-checking every member.

    :param path: The archive, possibly still named ``<file>.part``.
    :type path: Path
    :return: The number of members verified.
    :rtype: int
    :raises ArchiveError: If the file is not a readable zip, holds no members,
        or any member fails to inflate or fails its CRC -- which is what a
        truncated or corrupted download looks like.
    """
    try:
        with zipfile.ZipFile(path) as archive:
            count = len(archive.infolist())
            bad = archive.testzip()
    except Exception as exc:  # noqa: BLE001
        raise ArchiveError(
            f"{path} is not a readable zip ({type(exc).__name__}: {exc}); "
            "treating it as a truncated or corrupt download"
        ) from exc
    if bad is not None:
        raise ArchiveError(
            f"{path} failed its completeness check: member {bad!r} does not "
            "decode (CRC or inflate failure). The bytes on disk are wrong, "
            "not merely incomplete."
        )
    if count == 0:
        raise ArchiveError(f"{path} contains no members")
    return count


# ---------------------------------------------------------------------------
# Staging
# ---------------------------------------------------------------------------

STATUS_ALREADY_STAGED: str = "already-staged"
STATUS_WOULD_STAGE: str = "would-stage"
STATUS_STAGED: str = "staged"
STATUS_DEAD_URL: str = "dead-url"
STATUS_INCOMPLETE: str = "incomplete"
STATUS_REFUSED: str = "refused"
STATUS_FAILED: str = "failed"


@dataclass(frozen=True)
class StageResult:
    """What happened to the corpus in one run."""

    status: str
    path: Path
    bytes_downloaded: int = 0
    bytes_on_disk: int = 0
    members: int = 0
    detail: str = ""

    @property
    def is_staged(self) -> bool:
        """Whether a complete, verified corpus is on disk after this run."""
        return self.status in (STATUS_ALREADY_STAGED, STATUS_STAGED)


def read_marker(dataset_root: Path) -> Dict[str, Any]:
    """Read the ``.ok`` marker, or return ``{}`` if it is absent or unreadable.

    :param dataset_root: The dataset volume.
    :type dataset_root: Path
    :return: The decoded marker.
    :rtype: Dict[str, Any]
    """
    marker = marker_path(dataset_root)
    if not marker.exists():
        return {}
    try:
        loaded = json.loads(marker.read_text())
    except Exception:  # noqa: BLE001 - a damaged marker means "not staged"
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _marker_members(marker: Dict[str, Any]) -> int:
    """Member count from a marker, under either key it may carry.

    The marker written by the pre-step-17 manual staging run uses
    ``zip_members``; :func:`stage_uvdoc` writes both that and ``members``.

    :param marker: A decoded marker.
    :type marker: Dict[str, Any]
    :return: The recorded member count, or ``0``.
    :rtype: int
    """
    for key in ("zip_members", "members"):
        value = marker.get(key)
        if isinstance(value, int):
            return value
    return 0


def _staged_result(dataset_root: Path) -> Optional[StageResult]:
    """The ``already-staged`` result, or ``None`` if the slot is not complete.

    Complete means: the marker exists, the archive exists, and the marker's
    recorded byte count equals the archive's actual size. Nothing else is
    read -- in particular no digest is computed, because re-hashing 27.5 GB on
    every trainer start would cost minutes to re-prove what the CRC pass
    already proved once.

    :param dataset_root: The dataset volume.
    :type dataset_root: Path
    :return: The result, or ``None``.
    :rtype: Optional[StageResult]
    """
    final = archive_path(dataset_root)
    marker = read_marker(dataset_root)
    if not marker or not final.exists():
        return None
    recorded = marker.get("bytes")
    actual = final.stat().st_size
    if recorded != actual:
        return None
    return StageResult(
        status=STATUS_ALREADY_STAGED,
        path=final,
        bytes_on_disk=actual,
        members=_marker_members(marker),
        detail=(
            f"marker present and size matches ({actual:,} B); "
            "nothing fetched, nothing written"
        ),
    )


def stage_uvdoc(
    dataset_root: Path,
    *,
    min_free_bytes: int = int(DEFAULT_MIN_FREE_GB * _GB),
    timeout: float = DEFAULT_TIMEOUT_S,
    dry_run: bool = False,
    url: str = UVDOC_URL,
) -> StageResult:
    """Download, verify and mark the UVDoc archive. Idempotent and resumable.

    Order of operations, and why:

    1. If the ``.ok`` marker exists and the archive is on disk at the recorded
       size, return immediately -- **no HEAD, no GET, no write**. This is what
       makes a second run cost zero network, and it is checked FIRST so that
       neither a disk-budget reading nor a dead host can disturb a corpus that
       is already good.
    2. Check the disk budget. Refuse before fetching, never mid-body.
    3. HEAD the URL. A dead host returns ``dead-url``, naming the code.
    4. Stream to ``<file>.part``, resuming any existing partial.
    5. If the body came up SHORT of the advertised length, stop and keep the
       ``.part`` -- that is a resumable interruption, not a corruption.
    6. Decode every member (:func:`verify_archive`). This is the proof.
    7. ``os.replace`` the ``.part`` into place, then write ``.ok`` LAST.

    :param dataset_root: The dataset volume.
    :type dataset_root: Path
    :param min_free_bytes: Free-space floor the volume must not cross.
    :type min_free_bytes: int
    :param timeout: Per-request timeout in seconds.
    :type timeout: float
    :param dry_run: Report what would happen and write nothing at all.
    :type dry_run: bool
    :param url: The archive URL; overridable for tests.
    :type url: str
    :return: What happened.
    :rtype: StageResult
    """
    final = archive_path(dataset_root)

    staged = _staged_result(dataset_root)
    if staged is not None:
        return staged

    part = part_path(dataset_root)
    already_have = part.stat().st_size if part.exists() else 0

    if dry_run:
        status = verify_url(url, timeout)
        expected = status.content_length or UVDOC_EXPECTED_BYTES
        return StageResult(
            status=STATUS_WOULD_STAGE if status.ok else STATUS_DEAD_URL,
            path=final,
            bytes_on_disk=already_have,
            detail=(
                f"HTTP {status.status_code}"
                + (f" ({status.detail})" if status.detail else "")
                + f"; would fetch {max(expected - already_have, 0):,} B of "
                f"{expected:,} B into {part.name}"
                + (
                    f" (resuming a {already_have:,} B partial)"
                    if already_have
                    else ""
                )
                + f", verify every member, then write {final.name} and its "
                ".ok marker. DRY RUN: nothing was written."
            ),
        )

    try:
        check_disk_budget(
            dataset_root,
            max(UVDOC_EXPECTED_BYTES - already_have, 0),
            min_free_bytes,
        )
    except DiskBudgetError as error:
        return StageResult(
            status=STATUS_REFUSED, path=final, bytes_on_disk=already_have,
            detail=str(error),
        )

    status = verify_url(url, timeout)
    if not status.ok:
        return StageResult(
            status=STATUS_DEAD_URL,
            path=final,
            bytes_on_disk=already_have,
            detail=(
                f"HTTP {status.status_code}"
                + (f" ({status.detail})" if status.detail else "")
                + f" for {url}; nothing was downloaded"
            ),
        )
    if already_have and not status.accept_ranges:
        logger.warning(
            "DocScanner/UVDoc: a %d B partial exists but the host did not "
            "advertise `accept-ranges: bytes`; the download may restart from "
            "zero.", already_have,
        )

    expected = status.content_length or UVDOC_EXPECTED_BYTES
    try:
        written, on_disk = _download(url, part, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        on_disk = part.stat().st_size if part.exists() else 0
        return StageResult(
            status=STATUS_INCOMPLETE,
            path=final,
            bytes_on_disk=on_disk,
            detail=(
                f"{type(exc).__name__}: {exc}. {on_disk:,} B are kept in "
                f"{part.name} for a later run to resume; no {final.name} and "
                "no .ok marker were written."
            ),
        )

    if on_disk < expected:
        return StageResult(
            status=STATUS_INCOMPLETE,
            path=final,
            bytes_downloaded=written,
            bytes_on_disk=on_disk,
            detail=(
                f"the body stopped at {on_disk:,} B of {expected:,} B. The "
                f"partial is KEPT in {part.name} -- re-run to resume it. No "
                f"{final.name} and no .ok marker were written, so the slot "
                "stays honestly empty rather than silently truncated."
            ),
        )

    try:
        members = verify_archive(part)
    except ArchiveError as error:
        # The body is full length yet does not decode, so the BYTES are wrong
        # and resuming cannot repair them. Only ever unlinks the .part; the
        # final archive is never touched, and nothing outside <root> is either.
        if part.exists():
            part.unlink()
        return StageResult(
            status=STATUS_FAILED,
            path=final,
            bytes_downloaded=written,
            detail=(
                f"{error} The {on_disk:,} B partial was discarded because a "
                "full-length body that fails its CRC cannot be repaired by "
                f"resuming. No {final.name} and no .ok marker were written."
            ),
        )

    os.replace(part, final)
    marker_path(dataset_root).write_text(
        json.dumps(
            {
                "source": url,
                "bytes": final.stat().st_size,
                "expected_bytes": expected,
                "zip_members": members,
                "members": members,
                "crc_testzip": "PASS (all members decoded, testzip() -> None)",
                "verified_utc": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
                "verified_by": "train.doc_scanner.prepare_doc_scanner_data",
                "note": (
                    "Completeness proven by decoding every member, not by "
                    "Content-Length. Promoted .part to .zip only after the "
                    "CRC pass; this .ok marker written last."
                ),
            },
            indent=2,
        )
    )
    return StageResult(
        status=STATUS_STAGED,
        path=final,
        bytes_downloaded=written,
        bytes_on_disk=final.stat().st_size,
        members=members,
        detail=f"{members:,} members decoded and CRC-checked",
    )


def require_staged_uvdoc(dataset_root: Path) -> Path:
    """Return the staged archive path, or fail with the REAL reason.

    Interface contract -- read-only, allocates nothing, makes no HTTP request,
    and never creates a directory. It exists so that "UVDoc is not staged"
    arrives as one sentence naming what is missing and how to fix it, in the
    SAME vocabulary the trainers already raise
    (:class:`train.doc_scanner.common.MissingTrainingDataError`), instead of
    an empty-glob mystery later on.

    :param dataset_root: The dataset volume.
    :type dataset_root: Path
    :return: The verified archive path.
    :rtype: Path
    :raises MissingTrainingDataError: If the corpus directory, the archive or
        the ``.ok`` marker is missing, or the marker disagrees with the size
        on disk.
    """
    final = archive_path(dataset_root)
    hint = (
        "Stage it with `python -m train.doc_scanner.prepare_doc_scanner_data` "
        "(27.5 GB, resumable) or train with `--data-source synthetic`, which "
        "needs no download."
    )
    if not uvdoc_dir(dataset_root).is_dir():
        raise MissingTrainingDataError(
            f"the UVDoc corpus directory {str(uvdoc_dir(dataset_root))!r} does "
            f"not exist. {hint}"
        )
    if not final.exists():
        siblings = sorted(p.name for p in uvdoc_dir(dataset_root).iterdir())
        raise MissingTrainingDataError(
            f"{str(final)!r} is missing; the corpus directory holds "
            f"{siblings}. A `.part` file there means an INTERRUPTED download, "
            f"which re-running resumes rather than restarts. {hint}"
        )
    marker = read_marker(dataset_root)
    if not marker:
        raise MissingTrainingDataError(
            f"{str(final)!r} exists but its completion marker "
            f"{marker_path(dataset_root).name!r} is missing or unreadable, so "
            "the archive was never proven complete -- it may be a truncated "
            f"body. {hint}"
        )
    actual = final.stat().st_size
    if marker.get("bytes") != actual:
        raise MissingTrainingDataError(
            f"{str(final)!r} is {actual:,} B but its marker records "
            f"{marker.get('bytes')!r} B, so the file on disk is not the one "
            f"that was verified. {hint}"
        )
    return final


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def log_result(result: StageResult, dataset_root: Path) -> None:
    """Log one line per outcome, at a level that matches it.

    :param result: What :func:`stage_uvdoc` returned.
    :type result: StageResult
    :param dataset_root: The dataset volume, for the header line.
    :type dataset_root: Path
    """
    logger.info(
        "DocScanner/UVDoc staging under %s: %s", uvdoc_dir(dataset_root),
        result.status,
    )
    if result.status in (STATUS_ALREADY_STAGED, STATUS_STAGED):
        logger.info(
            "  %s: %d B, %d members. %s", result.path, result.bytes_on_disk,
            result.members, result.detail,
        )
    elif result.status == STATUS_WOULD_STAGE:
        logger.info("  %s", result.detail)
    else:
        logger.error("  %s", result.detail)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """The argument parser. Constructing it allocates nothing but memory.

    :return: The parser.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="prepare_doc_scanner_data",
        description=(
            "Stage the UVDoc corpus under <root>/doc_scanner/uvdoc/. Additive "
            "only, idempotent, resumable; completeness is proven by decoding "
            "every archive member, never by Content-Length."
        ),
    )
    parser.add_argument(
        "--root", type=Path, default=DEFAULT_DATASET_ROOT,
        help=f"dataset volume (default: {DEFAULT_DATASET_ROOT})",
    )
    parser.add_argument(
        "--url", type=str, default=UVDOC_URL,
        help=f"archive URL (default: {UVDOC_URL})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="report what would be fetched and verified; write nothing",
    )
    parser.add_argument(
        "--min-free-gb", type=float, default=DEFAULT_MIN_FREE_GB,
        help=(
            "refuse to stage if the volume would drop below this "
            f"(default: {DEFAULT_MIN_FREE_GB})"
        ),
    )
    parser.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT_S,
        help=f"per-request timeout in seconds (default: {DEFAULT_TIMEOUT_S})",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point.

    The FIRST statement parses ``argv``, so ``--help`` prints its ``usage:``
    line and allocates nothing else -- no directory, no socket.

    :param argv: Command line, or ``None`` for ``sys.argv[1:]``.
    :type argv: Optional[Sequence[str]]
    :return: ``0`` when the corpus is staged (or would be, under
        ``--dry-run``); ``1`` when it is not.
    :rtype: int
    """
    args = build_parser().parse_args(argv)
    result = stage_uvdoc(
        args.root,
        min_free_bytes=int(args.min_free_gb * _GB),
        timeout=args.timeout,
        dry_run=args.dry_run,
        url=args.url,
    )
    log_result(result, args.root)
    if result.is_staged or result.status == STATUS_WOULD_STAGE:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
