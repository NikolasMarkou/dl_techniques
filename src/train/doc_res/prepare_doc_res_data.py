"""Download, verify, lay out and prompt-precompute the DocRes training corpora.

Target layout (matches both the sibling convention under
``/media/arxwn/data0_4tb/datasets/`` and DocRes's own manifest-relative
``data/train/<task>/<dataset>/`` shape)::

    <root>/doc_res/<task>/<dataset>/
        _archives/<file>            <- the downloaded archive + its .ok marker
        _prompts/<mirror>/<stem>.png <- the precomputed DTSPrompt sidecar
        <whatever the archive contained>
        README.md                   <- only for a slot this script cannot fill

``<task>`` is a key of
:data:`dl_techniques.datasets.document_restoration.TASKS`; nothing in this
module branches on a task string outside the manifest's ``task`` field.

What this script will and will not fetch
----------------------------------------
Only sources that were verified live are wired to download. The manifest
records every other slot too, so a missing corpus is a *named, explained*
absence rather than a silent one:

======================  ==========  =======================================
Dataset                 Task        Availability
======================  ==========  =======================================
DIBCO/H-DIBCO 2009-2019 binarizat.  ``http`` -- 11 separate university URLs.
                                    There is no pooled DIBCO download; each
                                    year is its own Apache listing.
NoisyOffice             binarizat.  ``http`` -- UCI ML Repository, CC BY 4.0.
RDD                     deshadow.   ``manual_drive`` -- see D-025.
DIR300                  dewarping   ``manual_drive``, EVAL ONLY (300 images).
Doc3D                   dewarping   ``gated`` -- registration wall, 549 GB.
                                    Deliberately never downloaded here.
TDD                     deblurring  ``dead_host`` -- the official host is a
                                    confirmed 404, not a missing file.
======================  ==========  =======================================

Safety properties
-----------------
* **Additive only.** Every write lands under ``<root>/doc_res/``. This script
  deletes nothing, anywhere, ever -- and never touches repo-root ``results/``.
* **Refuses rather than fills the volume.** Free space is checked before each
  dataset against ``--min-free-gb`` (default 100 GB) plus that dataset's own
  byte budget.
* **Verifies before downloading.** Every URL is HEAD-checked first; a dead one
  is reported *by name* and the run continues with the rest.
* **Idempotent and resumable.** See :func:`stage_archive` and D-024.

Examples::

    # print every URL and the byte budget, write nothing
    python -m train.doc_res.prepare_doc_res_data --dry-run

    # stage the confirmed-available corpora and precompute prompt sidecars
    python -m train.doc_res.prepare_doc_res_data --tasks binarization
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from dl_techniques.datasets.document_restoration import TASKS, get_task
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------------
# Locations and budgets
# ---------------------------------------------------------------------------

DEFAULT_DATASET_ROOT: Path = Path("/media/arxwn/data0_4tb/datasets")
"""The shared dataset volume. Every sibling corpus is a top-level dir here."""

STAGING_DIRNAME: str = "doc_res"
"""The single directory this script owns under :data:`DEFAULT_DATASET_ROOT`."""

DEFAULT_MIN_FREE_GB: float = 100.0
"""Refuse to stage if the volume would drop below this. The plan's pre-mortem
names 100 GB as the stop trigger for ``/media/arxwn/data0_4tb``."""

ARCHIVE_DIRNAME: str = "_archives"
PROMPT_DIRNAME: str = "_prompts"

DEFAULT_TIMEOUT_S: float = 60.0

_USER_AGENT: str = "dl_techniques-doc_res-prepare/1.0"

_GB: int = 1000 ** 3

# ---------------------------------------------------------------------------
# Availability kinds
# ---------------------------------------------------------------------------

AVAIL_HTTP: str = "http"
"""Direct HTTP archive(s); this script downloads, verifies and extracts them."""

# DECISION plan-2026-09-08T111844-de235227/D-025
# RDD and DIR300 are wired as manual_drive and are NEVER partially
# downloaded. Do NOT "fix" this by adding gdown or by scraping the folder
# page: MEASURED on 2026-09-08, the Drive folder HTML for DIR300/dist
# yields only ~51 distinct file ids for a folder of 300 images, because
# Drive paginates the listing behind an authenticated internal API. gdown
# scrapes the same HTML and documents the same cap. A scraper would
# therefore stage a SILENTLY TRUNCATED corpus -- a 51-image "DIR300" that
# every downstream count would report as staged. Refusing loudly with the
# browser URL is the correct outcome; a partial one is not. See D-025.
AVAIL_MANUAL_DRIVE: str = "manual_drive"
"""A Google Drive *folder* of loose files. Not scriptable -- see D-025."""

AVAIL_GATED: str = "gated"
"""Behind a registration/agreement wall. Never downloaded by this script."""

AVAIL_DEAD_HOST: str = "dead_host"
"""The official host itself is gone -- not a moved file, a dead server."""

AVAILABILITIES: frozenset = frozenset(
    {AVAIL_HTTP, AVAIL_MANUAL_DRIVE, AVAIL_GATED, AVAIL_DEAD_HOST}
)

ROLE_TRAIN: str = "train"
ROLE_EVAL: str = "eval"
ROLES: frozenset = frozenset({ROLE_TRAIN, ROLE_EVAL})


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DocResDataError(RuntimeError):
    """Base class for every failure this module raises."""


class DiskBudgetError(DocResDataError):
    """The volume does not have room for a dataset, with the shortfall named."""


class ArchiveError(DocResDataError):
    """A downloaded archive is truncated, corrupt or not readable."""


class NoExtractorError(DocResDataError):
    """No archive backend on this machine can read the given format."""


class ManualDownloadRequired(DocResDataError):
    """A source exists but cannot be fetched without a browser session."""


class MissingTrainingDataError(DocResDataError):
    """A task was asked to train with no corpus staged, with the reason named."""


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Archive:
    """One downloadable file.

    :param url: The absolute URL. Verified with a HEAD before any GET.
    :param filename: The name to store it under in ``_archives/``.
    :param approx_bytes: The ``Content-Length`` measured on 2026-09-08, used
        for the dry-run budget when the host declines to report one at run
        time. ``0`` means "the host reports no length" (UCI streams its zip).
    :param sha256: Pinned digest, when one is known. ``None`` for every source
        here -- none of these universities publish one.
    """

    url: str
    filename: str
    approx_bytes: int
    sha256: Optional[str] = None


@dataclass(frozen=True)
class Dataset:
    """One corpus slot: either fetchable, or explicitly not, with the reason.

    :param name: Directory name under ``<root>/doc_res/<task>/``.
    :param task: A key of :data:`TASKS`.
    :param availability: One of :data:`AVAILABILITIES`.
    :param role: ``"train"`` or ``"eval"``. An eval-only corpus never satisfies
        :func:`require_task_data`.
    :param provenance: Where it comes from, in one line, for the README.
    :param licence: Licence as published, or ``"unstated"``.
    :param reason: For a non-``http`` slot: why this script cannot fetch it.
        Empty for an ``http`` slot.
    :param archives: The files to fetch. Non-empty iff ``availability`` is
        ``http``.
    :param manual_url: The page a human must open for a ``manual_drive`` or
        ``gated`` slot.
    :param approx_bytes: Total budget. Derived from ``archives`` when present.
    """

    name: str
    task: str
    availability: str
    role: str
    provenance: str
    licence: str
    reason: str = ""
    archives: Tuple[Archive, ...] = ()
    manual_url: str = ""
    approx_bytes: int = 0

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError(
                f"dataset {self.name!r} declares unknown task {self.task!r}; "
                f"legal: {sorted(TASKS)}"
            )
        if self.availability not in AVAILABILITIES:
            raise ValueError(
                f"dataset {self.name!r} declares unknown availability "
                f"{self.availability!r}; legal: {sorted(AVAILABILITIES)}"
            )
        if self.role not in ROLES:
            raise ValueError(
                f"dataset {self.name!r} declares unknown role {self.role!r}"
            )
        if self.availability == AVAIL_HTTP:
            if not self.archives:
                raise ValueError(
                    f"dataset {self.name!r} is {AVAIL_HTTP} but lists no archives"
                )
            if self.reason:
                raise ValueError(
                    f"dataset {self.name!r} is fetchable but states a reason "
                    f"it is not: {self.reason!r}"
                )
        else:
            if self.archives:
                raise ValueError(
                    f"dataset {self.name!r} is {self.availability} but lists "
                    "archives; a non-fetchable slot must list none"
                )
            if not self.reason:
                raise ValueError(
                    f"dataset {self.name!r} is {self.availability} and must "
                    "state the reason it cannot be fetched"
                )
        object.__setattr__(
            self,
            "approx_bytes",
            self.approx_bytes or sum(a.approx_bytes for a in self.archives),
        )

    @property
    def is_fetchable(self) -> bool:
        """Whether this script downloads this slot at all."""
        return self.availability == AVAIL_HTTP


def _dibco(
    name: str, base: str, files: Sequence[Tuple[str, int]]
) -> Dataset:
    """Build one DIBCO/H-DIBCO year entry.

    :param name: Directory name, e.g. ``"dibco2009"``.
    :param base: The year's benchmark directory URL, ending in ``/``.
    :param files: ``(filename, content_length_bytes)`` pairs.
    :return: The :class:`Dataset`.
    """
    return Dataset(
        name=name,
        task="binarization",
        availability=AVAIL_HTTP,
        role=ROLE_TRAIN,
        provenance=base,
        licence="unstated (competition benchmark, published for research use)",
        archives=tuple(
            Archive(url=base + fn, filename=fn, approx_bytes=n) for fn, n in files
        ),
    )


# Content-Length values below were measured with a HEAD sweep on 2026-09-08.
# They are the dry-run budget's fallback; the dry run re-measures live.
MANIFEST: Tuple[Dataset, ...] = (
    _dibco(
        "dibco2009",
        "https://users.iit.demokritos.gr/~bgat/DIBCO2009/benchmark/",
        (
            ("DIBC02009_Test_images-handwritten.rar", 2_270_265),
            ("DIBCO2009_Test_images-printed.rar", 3_596_348),
            ("DIBCO2009-GT-Test-images_handwritten.rar", 41_351),
            ("DIBCO2009-GT-Test-images_printed.rar", 42_637),
        ),
    ),
    _dibco(
        "hdibco2010",
        "https://users.iit.demokritos.gr/~bgat/H-DIBCO2010/benchmark/",
        (
            ("H_DIBCO2010_test_images.rar", 3_945_957),
            ("H_DIBCO2010_GT.rar", 192_320),
        ),
    ),
    _dibco(
        "dibco2011",
        "https://utopia.duth.gr/~ipratika/DIBCO2011/benchmark/dataset/",
        (
            ("DIBCO11-handwritten.rar", 9_513_655),
            ("DIBCO11-machine_printed.rar", 10_145_704),
        ),
    ),
    _dibco(
        "hdibco2012",
        "https://utopia.duth.gr/~ipratika/HDIBCO2012/benchmark/dataset/",
        (("H-DIBCO2012-dataset.rar", 24_522_318),),
    ),
    _dibco(
        "dibco2013",
        "https://utopia.duth.gr/~ipratika/DIBCO2013/benchmark/dataset/",
        (("DIBCO2013-dataset.rar", 32_236_463),),
    ),
    _dibco(
        "hdibco2014",
        "https://users.iit.demokritos.gr/~bgat/HDIBCO2014/benchmark/dataset/",
        (
            ("original_images.rar", 12_944_577),
            ("GT.rar", 156_196),
        ),
    ),
    # H-DIBCO 2015 is the ONE year with no live page. Five candidate hosts were
    # probed on 2026-09-08 (vc.ee.duth.gr in two casings, users.iit.demokritos.gr,
    # utopia.duth.gr, and the vc.ee.duth.gr year root) and all five returned 404.
    # It stays in the manifest deliberately: a dead year must be reported BY NAME
    # every run, not quietly absent from an 11-row table that then looks like 10.
    _dibco(
        "hdibco2015",
        "https://vc.ee.duth.gr/h-dibco2015/benchmark/",
        (("H-DIBCO2015_dataset.zip", 0),),
    ),
    _dibco(
        "hdibco2016",
        "https://vc.ee.duth.gr/h-dibco2016/benchmark/",
        (
            ("DIBCO2016_dataset-original.zip", 8_985_981),
            ("DIBCO2016_dataset-GT.zip", 240_396),
        ),
    ),
    _dibco(
        "dibco2017",
        "https://vc.ee.duth.gr/dibco2017/benchmark/",
        (
            ("DIBCO2017_Dataset.7z", 43_868_529),
            ("DIBCO2017_GT.7z", 892_437),
        ),
    ),
    _dibco(
        "hdibco2018",
        "https://vc.ee.duth.gr/h-dibco2018/benchmark/",
        (
            ("dibco2018_Dataset.zip", 22_311_122),
            ("dibco2018-GT.zip", 3_102_047),
        ),
    ),
    _dibco(
        "dibco2019",
        "https://vc.ee.duth.gr/dibco2019/benchmark/",
        (
            ("dibco2019_dataset_trackA.zip", 5_468_256),
            ("dibco2019_gt_trackA.zip", 99_778),
            ("dibco2019_dataset_trackB.zip", 58_563_618),
            ("dibco2019_GT_trackB.zip", 851_685),
        ),
    ),
    Dataset(
        name="noisy_office",
        task="binarization",
        availability=AVAIL_HTTP,
        role=ROLE_TRAIN,
        provenance="UCI ML Repository dataset 318 (NoisyOffice)",
        licence="CC BY 4.0",
        archives=(
            Archive(
                url="https://archive.ics.uci.edu/static/public/318/noisyoffice.zip",
                filename="noisyoffice.zip",
                # UCI streams this zip: no Content-Length, no range support.
                # 0 means "unknown"; the ~426 MB figure below is the published
                # size and is what the dry-run budget falls back to.
                approx_bytes=426_000_000,
            ),
        ),
    ),
    Dataset(
        name="rdd",
        task="deshadowing",
        availability=AVAIL_MANUAL_DRIVE,
        role=ROLE_TRAIN,
        provenance="Zhang et al. CVPR 2023, github.com/hyyh1314/RDD",
        licence="unstated (research use)",
        manual_url=(
            "https://drive.google.com/drive/folders/"
            "1nS-p9qKCsFjOFyzq5q2m-Dq-PJVmLfNc"
        ),
        reason=(
            "the RDD release is a Google Drive FOLDER holding train/ and test/ "
            "subfolders of ~4,916 loose images; a Drive folder listing is "
            "capped well below that (see D-025), so no script -- including "
            "gdown -- can enumerate it. Download it once from a browser and "
            "unpack it into this directory."
        ),
        approx_bytes=0,
    ),
    Dataset(
        name="dir300",
        task="dewarping",
        availability=AVAIL_MANUAL_DRIVE,
        role=ROLE_EVAL,
        provenance="Feng et al., DocGeoNet DIR300 test set",
        licence="unstated (research use)",
        manual_url=(
            "https://drive.google.com/drive/folders/"
            "1yySouQQ3BlH7OjnUhq4CLuvpX2KXtifX"
        ),
        reason=(
            "same Drive-folder limit as RDD (D-025): DIR300 is a folder of two "
            "subfolders, dist/ and gt/, each holding 300 loose images. EVAL "
            "ONLY in any case -- 300 images cannot train dewarping."
        ),
        approx_bytes=0,
    ),
    Dataset(
        name="doc3d",
        task="dewarping",
        availability=AVAIL_GATED,
        role=ROLE_TRAIN,
        provenance=(
            "Das et al. ICCV 2019; huggingface.co/datasets/"
            "StonyBrook-CVLab/doc3D-dataset (legacy: "
            "github.com/cvlab-stonybrook/doc3D-dataset)"
        ),
        licence="research use, on acceptance of the authors' agreement",
        manual_url="https://huggingface.co/datasets/StonyBrook-CVLab/doc3D-dataset",
        reason=(
            "registration-gated: the HuggingFace repo requires accepting a "
            "contact-info agreement before the file tree is even visible, and "
            "the legacy GitHub path required a Google Form for credentials. "
            "The full set is 549 GB (100k renders plus backward-map, UV, "
            "normal, albedo and mesh components). This script NEVER attempts "
            "it: the gate is a real agreement, not a formality to script "
            "around, and 549 GB would dominate the volume. The slot is wired "
            "so that dropping the data here is all that is needed."
        ),
        approx_bytes=549 * _GB,
    ),
    Dataset(
        name="tdd",
        task="deblurring",
        availability=AVAIL_DEAD_HOST,
        role=ROLE_TRAIN,
        provenance="Hradis et al., Text Document Deblurring dataset",
        licence="unstated (research use)",
        manual_url="",
        reason=(
            "the official host is DEAD, not merely missing this file: "
            "fit.vutbr.cz redirects to fit.vut.cz and that 404s too. There is "
            "no verified mirror -- the only lead is an unverified OneDrive "
            "link in a third-party recommendations README. Deblurring is "
            "therefore untrainable from any source this port has confirmed."
        ),
        approx_bytes=0,
    ),
)

_MANIFEST_BY_NAME: Dict[str, Dataset] = {d.name: d for d in MANIFEST}
if len(_MANIFEST_BY_NAME) != len(MANIFEST):
    raise ValueError("duplicate dataset name in MANIFEST")


def dataset_names() -> Tuple[str, ...]:
    """Every manifest dataset name, in table order."""
    return tuple(_MANIFEST_BY_NAME)


def get_dataset(name: str) -> Dataset:
    """Resolve a dataset name.

    :param name: A manifest dataset name.
    :return: The :class:`Dataset`.
    :raises ValueError: If unknown; the message lists the legal names.
    """
    try:
        return _MANIFEST_BY_NAME[name]
    except KeyError:
        raise ValueError(
            f"unknown DocRes dataset {name!r}; legal: {list(dataset_names())}"
        ) from None


def select_datasets(
    tasks: Optional[Sequence[str]] = None,
    names: Optional[Sequence[str]] = None,
) -> Tuple[Dataset, ...]:
    """Filter the manifest by task and/or explicit name.

    :param tasks: Task names to keep; ``None`` keeps all.
    :param names: Dataset names to keep; ``None`` keeps all.
    :return: The matching datasets in manifest order.
    :raises ValueError: On an unknown task or dataset name.
    """
    if tasks is not None:
        for t in tasks:
            get_task(t)
    if names is not None:
        for n in names:
            get_dataset(n)
    out = []
    for ds in MANIFEST:
        if tasks is not None and ds.task not in tasks:
            continue
        if names is not None and ds.name not in names:
            continue
        out.append(ds)
    return tuple(out)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def staging_root(dataset_root: Path) -> Path:
    """``<dataset_root>/doc_res``."""
    return Path(dataset_root) / STAGING_DIRNAME


def dataset_dir(dataset_root: Path, ds: Dataset) -> Path:
    """``<dataset_root>/doc_res/<task>/<dataset>``."""
    return staging_root(dataset_root) / ds.task / ds.name


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
    staging directory has been created.

    :param path: Any path on the target volume.
    :return: The :class:`DiskState`.
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
    :param needed_bytes: What the next dataset would consume.
    :param min_free_bytes: The floor free space must never cross.
    :return: The :class:`DiskState` that was read.
    :raises DiskBudgetError: If staging would cross the floor. The message
        names the shortfall in bytes and GB.
    """
    state = read_disk_state(dataset_root)
    remaining = state.free_bytes - needed_bytes
    if remaining < min_free_bytes:
        shortfall = min_free_bytes - remaining
        raise DiskBudgetError(
            f"refusing to stage into {dataset_root}: it needs "
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
    """The result of verifying one URL before downloading it."""

    url: str
    filename: str
    ok: bool
    status_code: int
    content_length: Optional[int]
    detail: str = ""


def _http_head(url: str, timeout: float) -> Tuple[int, Optional[int], str]:
    """HEAD a URL, following redirects.

    Falls back to a streamed GET (immediately closed) for hosts that reject
    HEAD -- some Apache configs answer 405 while serving GET fine.

    :param url: The URL to probe.
    :param timeout: Per-request timeout in seconds.
    :return: ``(status_code, content_length_or_None, detail)``. A transport
        failure is reported as status ``0`` with the exception text as detail.
    """
    import requests  # local: keeps --help free of a network stack import

    headers = {"User-Agent": _USER_AGENT}
    try:
        resp = requests.head(
            url, allow_redirects=True, timeout=timeout, headers=headers
        )
        if resp.status_code in (403, 405, 501):
            with requests.get(
                url, stream=True, allow_redirects=True, timeout=timeout,
                headers=headers,
            ) as g:
                length = g.headers.get("Content-Length")
                return g.status_code, int(length) if length else None, ""
        length = resp.headers.get("Content-Length")
        return resp.status_code, int(length) if length else None, ""
    except Exception as exc:  # noqa: BLE001 - a dead host must be reported, not raised
        return 0, None, f"{type(exc).__name__}: {exc}"


def verify_url(archive: Archive, timeout: float) -> UrlStatus:
    """Check one archive URL is live before any bytes are fetched.

    :param archive: The archive to probe.
    :param timeout: Per-request timeout in seconds.
    :return: The :class:`UrlStatus`; ``ok`` is false for any non-2xx or for a
        transport failure.
    """
    code, length, detail = _http_head(archive.url, timeout)
    ok = 200 <= code < 300
    return UrlStatus(
        url=archive.url,
        filename=archive.filename,
        ok=ok,
        status_code=code,
        content_length=length,
        detail=detail,
    )


def _download(url: str, dest_part: Path, timeout: float) -> int:
    """Stream a URL to ``dest_part``, overwriting any previous partial.

    :param url: The URL to fetch.
    :param dest_part: The ``.part`` path to write.
    :param timeout: Per-request timeout in seconds.
    :return: Bytes written.
    """
    import requests

    dest_part.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with requests.get(
        url,
        stream=True,
        allow_redirects=True,
        timeout=timeout,
        headers={"User-Agent": _USER_AGENT},
    ) as resp:
        resp.raise_for_status()
        with open(dest_part, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)
                    written += len(chunk)
    return written


def _sha256(path: Path) -> str:
    """Hex sha256 of a file, read in 1 MB blocks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Archives
# ---------------------------------------------------------------------------


def _libarchive():
    """Import ``libarchive`` (libarchive-c) or return ``None``."""
    try:
        import libarchive  # type: ignore
    except Exception:  # noqa: BLE001
        return None
    return libarchive


def extractor_backends() -> Dict[str, bool]:
    """Which archive backends this machine actually has.

    :return: ``{"zipfile": True, "libarchive": bool}``. ``zipfile`` is stdlib
        and therefore always present; it covers ``.zip`` alone.
    """
    return {"zipfile": True, "libarchive": _libarchive() is not None}


# DECISION plan-2026-09-08T111844-de235227/D-023
# Archive extraction goes through libarchive-c (a ctypes binding to the
# system libarchive.so.13) with a stdlib-zipfile fast path, NOT through an
# external binary. Do NOT "simplify" this to a subprocess call to unrar /
# unar / bsdtar / 7z: none of the four is installed on this machine
# (`which` found only curl and wget), while /usr/lib/.../libarchive.so.13
# is present and reads rar, rar5, 7z and zip in one API. Six of the eleven
# DIBCO years ship .rar and one ships .7z, so a zipfile-only script would
# stage 4 of 11 years. Do NOT add libarchive-c to pyproject.toml either --
# it is a staging-time convenience, and every caller here degrades to a
# named NoExtractorError rather than a crash when it is absent. See D-023.
def _iter_members(path: Path, format_name: Optional[str] = None):
    """Yield ``(member_path, size, is_dir, read_bytes)`` for an archive.

    ``read_bytes`` is a zero-argument callable returning the member's full
    contents; calling it is what proves the member is really decodable.

    :param path: The archive file. May be a ``.part`` temporary.
    :param format_name: The name whose extension selects the backend. Defaults
        to ``path.name``. It exists because the completeness check runs against
        ``<file>.part``, whose own suffix says nothing about the format --
        and dispatching on ``.part`` sent a truncated zip down libarchive's
        streaming reader, which happily returned the local headers it could
        still see instead of failing.
    :raises NoExtractorError: If no backend on this machine reads the format.
    """
    suffix = Path(format_name or path.name).suffix.lower()
    if suffix == ".zip":
        import zipfile

        with zipfile.ZipFile(path) as zf:
            for info in zf.infolist():
                yield (
                    info.filename,
                    info.file_size,
                    info.is_dir(),
                    (lambda i=info: zf.read(i)),
                )
        return

    la = _libarchive()
    if la is None:
        raise NoExtractorError(
            f"cannot read {path.name}: no backend on this machine handles "
            f"{suffix!r}. Install the pure-python binding with "
            "`pip install libarchive-c` (the system libarchive.so is already "
            "present), or install one of unrar / unar / bsdtar / 7z. The "
            "archive itself is staged and intact; only extraction is blocked."
        )
    with la.file_reader(str(path)) as archive:
        for entry in archive:
            blocks = b"".join(entry.get_blocks()) if not entry.isdir else b""
            yield (
                entry.pathname,
                int(entry.size or 0),
                bool(entry.isdir),
                (lambda b=blocks: b),
            )


def verify_archive(path: Path, format_name: Optional[str] = None) -> int:
    """Prove an archive is complete by decoding every member.

    :param path: The archive file, possibly still named ``<file>.part``.
    :param format_name: See :func:`_iter_members`.
    :return: The number of members read.
    :raises ArchiveError: If any member fails to decode, which is what a
        truncated download looks like.
    :raises NoExtractorError: If the format has no backend here.
    """
    count = 0
    try:
        for _name, _size, is_dir, read in _iter_members(path, format_name):
            if not is_dir:
                read()
            count += 1
    except NoExtractorError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ArchiveError(
            f"{path} is not a readable archive ({type(exc).__name__}: {exc}); "
            "treating it as a truncated or corrupt download"
        ) from exc
    if count == 0:
        raise ArchiveError(f"{path} contains no members")
    return count


def _safe_member_path(dest_dir: Path, member: str) -> Optional[Path]:
    """Resolve an archive member under ``dest_dir``, or reject it.

    :param dest_dir: The extraction root.
    :param member: The member path as stored in the archive.
    :return: The absolute destination, or ``None`` if the member escapes
        ``dest_dir`` (absolute path, drive letter, or ``..`` traversal).
    """
    name = member.replace("\\", "/")
    # An absolute member, or a Windows drive letter, is rejected rather than
    # normalised: no legitimate dataset archive stores one, and silently
    # re-rooting it hides a malformed or hostile archive.
    if name.startswith("/") or re.match(r"^[A-Za-z]:", name):
        return None
    if not name or name.startswith("../") or "/../" in name or name == "..":
        return None
    target = (dest_dir / name).resolve()
    root = dest_dir.resolve()
    if root != target and root not in target.parents:
        return None
    return target


def extract_archive(path: Path, dest_dir: Path) -> List[Path]:
    """Extract an archive under ``dest_dir``, skipping members already there.

    Members whose stored path escapes ``dest_dir`` are skipped and logged --
    this function never writes outside its destination.

    :param path: The archive file.
    :param dest_dir: Where to extract.
    :return: The files written by this call (already-present files excluded).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for member, size, is_dir, read in _iter_members(path):
        target = _safe_member_path(dest_dir, member)
        if target is None:
            logger.warning(
                "skipping archive member with an unsafe path: %s (in %s)",
                member,
                path.name,
            )
            continue
        if is_dir:
            target.mkdir(parents=True, exist_ok=True)
            continue
        if target.exists() and target.stat().st_size == size:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".part")
        with open(tmp, "wb") as fh:
            fh.write(read())
        os.replace(tmp, target)
        written.append(target)
    return written


# ---------------------------------------------------------------------------
# Staging one archive
# ---------------------------------------------------------------------------


@dataclass
class ArchiveResult:
    """What happened to one archive in one run."""

    filename: str
    url: str
    status: str  # "already-staged" | "staged" | "dead-url" | "failed"
    bytes_downloaded: int = 0
    members: int = 0
    extracted_files: int = 0
    detail: str = ""


def _marker_path(archives_dir: Path, filename: str) -> Path:
    """The completion marker for one archive."""
    return archives_dir / (filename + ".ok")


# DECISION plan-2026-09-08T111844-de235227/D-024
# Completeness is proven by DECODING EVERY MEMBER of the archive and then
# doing an atomic os.replace() of a `.part` file into place; the `.ok`
# marker is written last. Do NOT replace this with the obvious
# Content-Length size check: it was MEASURED not to work here. UCI serves
# noisyoffice.zip with no Content-Length and no range support (a
# `Range: bytes=0-0` request returns 200 and streams the whole 280 MB+),
# and the H-DIBCO 2015 URL 404s, so a size check would be vacuous for the
# one big download and unavailable for the dead one. Because a `.part`
# file never carries the final name, an interrupted run leaves nothing
# that a later run can mistake for complete. See D-024.
def stage_archive(
    archive: Archive,
    archives_dir: Path,
    dest_dir: Path,
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> ArchiveResult:
    """Download, verify, extract and mark one archive. Idempotent.

    Order of operations, and why:

    1. If the ``.ok`` marker exists and the archive is still on disk at the
       recorded size, return immediately -- **no HEAD, no GET**. This is what
       makes a second run cost zero network.
    2. Verify the URL. A dead one returns ``status="dead-url"`` naming the
       file and the HTTP code; it never raises, so one dead year does not
       abort the other ten.
    3. Stream to ``<file>.part``.
    4. Decode every member (:func:`verify_archive`) -- the completeness proof.
    5. ``os.replace`` the ``.part`` into place, extract, then write ``.ok``.

    :param archive: The archive spec.
    :param archives_dir: Where the archive file itself is kept.
    :param dest_dir: Where its contents are extracted.
    :param timeout: Per-request timeout in seconds.
    :return: The :class:`ArchiveResult`.
    """
    archives_dir.mkdir(parents=True, exist_ok=True)
    final = archives_dir / archive.filename
    marker = _marker_path(archives_dir, archive.filename)

    if marker.exists() and final.exists():
        try:
            recorded = json.loads(marker.read_text())
        except Exception:  # noqa: BLE001
            recorded = {}
        if recorded.get("bytes") == final.stat().st_size:
            return ArchiveResult(
                filename=archive.filename,
                url=archive.url,
                status="already-staged",
                bytes_downloaded=0,
                members=int(recorded.get("members", 0)),
                detail="marker present and size matches; nothing fetched",
            )

    status = verify_url(archive, timeout)
    if not status.ok:
        return ArchiveResult(
            filename=archive.filename,
            url=archive.url,
            status="dead-url",
            detail=(
                f"HTTP {status.status_code}"
                + (f" ({status.detail})" if status.detail else "")
            ),
        )

    part = archives_dir / (archive.filename + ".part")
    try:
        written = _download(archive.url, part, timeout)
        if archive.sha256 is not None:
            digest = _sha256(part)
            if digest != archive.sha256:
                raise ArchiveError(
                    f"{archive.filename}: sha256 {digest} != pinned "
                    f"{archive.sha256}"
                )
        members = verify_archive(part, archive.filename)
    except NoExtractorError as exc:
        # The bytes are fine; only extraction is impossible. Keep the archive
        # under its final name so a later run with a backend installed can
        # extract it without re-downloading, but write NO marker.
        os.replace(part, final)
        return ArchiveResult(
            filename=archive.filename,
            url=archive.url,
            status="failed",
            bytes_downloaded=final.stat().st_size,
            detail=str(exc),
        )
    except Exception as exc:  # noqa: BLE001
        if part.exists():
            # Only ever removes the .part this call just created.
            part.unlink()
        return ArchiveResult(
            filename=archive.filename,
            url=archive.url,
            status="failed",
            detail=f"{type(exc).__name__}: {exc}",
        )

    os.replace(part, final)
    extracted = extract_archive(final, dest_dir)
    marker.write_text(
        json.dumps(
            {
                "url": archive.url,
                "bytes": final.stat().st_size,
                "sha256": _sha256(final),
                "members": members,
                "staged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            indent=2,
        )
    )
    return ArchiveResult(
        filename=archive.filename,
        url=archive.url,
        status="staged",
        bytes_downloaded=written,
        members=members,
        extracted_files=len(extracted),
    )


# ---------------------------------------------------------------------------
# Slot READMEs for what this script cannot fetch
# ---------------------------------------------------------------------------


def write_slot_readme(dataset_root: Path, ds: Dataset) -> Path:
    """Create the directory for a non-fetchable slot and explain it.

    :param dataset_root: The staging volume.
    :param ds: A non-``http`` dataset.
    :return: The README path.
    """
    target = dataset_dir(dataset_root, ds)
    target.mkdir(parents=True, exist_ok=True)
    readme = target / "README.md"
    lines = [
        f"# {ds.name} ({ds.task}, {ds.role}-only)",
        "",
        f"**Status: {ds.availability.upper()} -- this directory is empty by design.**",
        "",
        f"- Provenance: {ds.provenance}",
        f"- Licence: {ds.licence}",
        f"- Budget if obtained: {ds.approx_bytes / _GB:.2f} GB"
        if ds.approx_bytes
        else "- Budget if obtained: unpublished",
    ]
    if ds.manual_url:
        lines.append(f"- Manual source: {ds.manual_url}")
    lines += [
        "",
        "## Why `prepare_doc_res_data.py` did not fill it",
        "",
        ds.reason,
        "",
        "## What to do",
        "",
        "Drop the extracted corpus directly into this directory. Nothing else "
        "needs changing: the manifest entry, the task binding and the trainer's "
        "startup check all already point here. Then re-run",
        "",
        "```",
        f"python -m train.doc_res.prepare_doc_res_data --datasets {ds.name} "
        "--only-sidecars",
        "```",
        "",
        "to write the DTSPrompt sidecars.",
        "",
    ]
    readme.write_text("\n".join(lines))
    return readme


# ---------------------------------------------------------------------------
# Prompt sidecars
# ---------------------------------------------------------------------------

IMAGE_SUFFIXES: Tuple[str, ...] = (
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff",
)

# The eleven DIBCO years label ground truth in at least seven different ways,
# all of which were observed in the staged tree on 2026-09-08:
#   H01_GT.tiff   H08_skelGT.tiff   H06_estGT.tiff   7_gt.bmp
#   GTimages/     GT/               gt/              *_Dataset_GT/
# A single separator-bounded "gt" token misses skelGT, estGT and GTimages, so
# the rule is split in three.
_GT_UPPER = re.compile(r"(?<![A-Z])GT(?![A-Z])")
"""Uppercase ``GT`` not inside a longer all-caps run: catches ``_GT.tiff``,
``estGT``, ``skelGT``, ``GTimages`` and ``/GT/`` while leaving e.g. ``DIGTAL``
alone."""

_GT_LOWER = re.compile(r"(?:^|[^a-zA-Z])gt(?:[^a-zA-Z]|$)")
"""Lowercase ``gt`` bounded by non-letters: catches ``_gt.bmp`` and ``/gt/``
but not ``gtx_pages/``."""

_GT_WORDS = re.compile(r"ground[_\- ]?truth|clean", re.IGNORECASE)
"""Spelled-out variants. ``clean`` is here for NoisyOffice, whose ground truth
lives in ``clean_images_*`` -- the clean/noisy pairing is the restoration
convention throughout this repo (``train/bfunet`` uses the same words)."""


def is_ground_truth_path(relative_path: str) -> bool:
    """Whether a staged image is a ground-truth page rather than an input.

    Only input pages get a prompt sidecar; a prompt built from the ground
    truth would be wasted work and, for the binarization task, close to a
    label leak.

    :param relative_path: Path relative to the dataset directory, using ``/``.
    :return: True if it looks like ground truth.
    """
    path = relative_path.replace(os.sep, "/")
    return bool(
        _GT_UPPER.search(path)
        or _GT_LOWER.search(path)
        or _GT_WORDS.search(path)
    )


def iter_input_images(dir_path: Path) -> List[Path]:
    """Every image under a staged dataset that a prompt should be built for.

    Excludes the ``_archives/`` and ``_prompts/`` bookkeeping directories and
    anything :func:`is_ground_truth_path` flags.

    :param dir_path: The dataset directory.
    :return: Sorted list of image paths.
    """
    if not dir_path.is_dir():
        return []
    out: List[Path] = []
    for path in sorted(dir_path.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        rel = path.relative_to(dir_path).as_posix()
        parts = rel.split("/")
        if parts[0] in (ARCHIVE_DIRNAME, PROMPT_DIRNAME):
            continue
        # `.xvpics` thumbnail caches ship inside NoisyOffice; no hidden path
        # component is ever a dataset page.
        if any(part.startswith(".") for part in parts):
            continue
        if is_ground_truth_path(rel):
            continue
        out.append(path)
    return out


@dataclass
class SidecarResult:
    """What the prompt precompute did for one dataset."""

    dataset: str
    task: str
    status: str  # "ok" | "skipped" | "no-images"
    written: int = 0
    already_present: int = 0
    failed: int = 0
    seconds: float = 0.0
    detail: str = ""


def sidecar_path(dir_path: Path, image: Path, suffix: str) -> Path:
    """Where an image's prompt sidecar lives.

    :param dir_path: The dataset directory.
    :param image: The source image.
    :param suffix: ``".png"`` or ``".npy"``.
    :return: A path under ``_prompts/`` mirroring the image's relative path.
    """
    rel = image.relative_to(dir_path)
    return (dir_path / PROMPT_DIRNAME / rel).with_suffix(suffix)


def precompute_sidecars(
    dataset_root: Path, ds: Dataset, *, overwrite: bool = False
) -> SidecarResult:
    """Write the DTSPrompt sidecar for every staged input image, once.

    Writing prompts offline is the whole point: the tf.data pipeline in step
    11 then reads two files per sample and never runs Python per batch, which
    is also what upstream does for binarization.

    :param dataset_root: The staging volume.
    :param ds: The dataset to process.
    :param overwrite: Recompute sidecars that already exist.
    :return: The :class:`SidecarResult`, including wall time.
    """
    spec = get_task(ds.task)
    target = dataset_dir(dataset_root, ds)

    if spec.requires_mask:
        return SidecarResult(
            dataset=ds.name,
            task=ds.task,
            status="skipped",
            detail=(
                f"the {ds.task} prompt needs a per-page document mask, which "
                "upstream gets from an MBD segmentation network that this port "
                "does not include. No sidecar can be precomputed without it."
            ),
        )

    images = iter_input_images(target)
    if not images:
        return SidecarResult(
            dataset=ds.name,
            task=ds.task,
            status="no-images",
            detail=f"no staged input images under {target}",
        )

    import numpy as np
    from PIL import Image

    suffix = ".png" if spec.prompt_dtype == "uint8" else ".npy"
    started = time.time()
    written = present = failed = 0
    for image in images:
        out = sidecar_path(target, image, suffix)
        if out.exists() and not overwrite:
            present += 1
            continue
        try:
            with Image.open(image) as handle:
                rgb = np.asarray(handle.convert("RGB"), dtype=np.uint8)
            prompt = spec.prompt_fn(rgb)
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name(out.name + ".part")
            if suffix == ".png":
                # Explicit format: the temp name ends in ".part", from which
                # PIL cannot infer one ("unknown file extension").
                Image.fromarray(np.asarray(prompt, dtype=np.uint8)).save(
                    tmp, format="PNG"
                )
            else:
                np.save(tmp, np.asarray(prompt, dtype=np.float32))
                # np.save appends .npy to a name that lacks it.
                if not tmp.exists() and tmp.with_suffix(".part.npy").exists():
                    tmp = tmp.with_suffix(".part.npy")
            os.replace(tmp, out)
            written += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            logger.warning("prompt sidecar failed for %s: %s", image, exc)
    return SidecarResult(
        dataset=ds.name,
        task=ds.task,
        status="ok",
        written=written,
        already_present=present,
        failed=failed,
        seconds=time.time() - started,
    )


# ---------------------------------------------------------------------------
# The trainer's startup check
# ---------------------------------------------------------------------------


def count_staged_images(dataset_root: Path, ds: Dataset) -> int:
    """How many input images are staged for one dataset."""
    return len(iter_input_images(dataset_dir(dataset_root, ds)))


def require_task_data(dataset_root: Path, task: str) -> Tuple[Dataset, ...]:
    """Fail loudly, at startup, when a task has no training corpus.

    This is what stops a trainer from silently fitting on an empty dataset.
    The message names the *real* reason -- a registration gate, a dead host --
    rather than "no files found".

    :param dataset_root: The staging volume.
    :param task: A key of :data:`TASKS`.
    :return: The datasets that actually have staged images.
    :raises MissingTrainingDataError: If none does.
    """
    get_task(task)
    candidates = [
        d for d in MANIFEST if d.task == task and d.role == ROLE_TRAIN
    ]
    staged = tuple(
        d for d in candidates if count_staged_images(dataset_root, d) > 0
    )
    if staged:
        return staged

    reasons = []
    for d in candidates:
        if d.reason:
            reasons.append(f"  - {d.name}: [{d.availability}] {d.reason}")
        else:
            reasons.append(
                f"  - {d.name}: [{d.availability}] fetchable, but nothing is "
                f"staged at {dataset_dir(dataset_root, d)}. Run "
                "`python -m train.doc_res.prepare_doc_res_data "
                f"--tasks {task}`."
            )
    eval_only = [d for d in MANIFEST if d.task == task and d.role == ROLE_EVAL]
    tail = ""
    if eval_only:
        names = ", ".join(d.name for d in eval_only)
        tail = (
            f"\n{names} is staged/available for this task but is EVAL ONLY and "
            "cannot be trained on."
        )
    raise MissingTrainingDataError(
        f"cannot train task {task!r}: no training corpus is staged under "
        f"{staging_root(dataset_root)}.\n"
        + "\n".join(reasons)
        + tail
    )


# ---------------------------------------------------------------------------
# Plan / run
# ---------------------------------------------------------------------------


@dataclass
class DatasetPlan:
    """The dry-run view of one dataset."""

    dataset: Dataset
    url_statuses: Tuple[UrlStatus, ...]
    budget_bytes: int
    directory: Path


def build_plan(
    dataset_root: Path,
    datasets: Sequence[Dataset],
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
    verify: bool = True,
) -> List[DatasetPlan]:
    """Verify every URL and compute the byte budget, writing nothing.

    :param dataset_root: The staging volume.
    :param datasets: Datasets to plan.
    :param timeout: Per-request timeout in seconds.
    :param verify: Whether to probe the network. False makes the plan purely
        manifest-derived (used by tests).
    :return: One :class:`DatasetPlan` per dataset.
    """
    plans: List[DatasetPlan] = []
    for ds in datasets:
        statuses: List[UrlStatus] = []
        budget = ds.approx_bytes
        if ds.is_fetchable and verify:
            statuses = [verify_url(a, timeout) for a in ds.archives]
            measured = sum(
                s.content_length or 0 for s in statuses if s.ok
            )
            fallback = sum(
                a.approx_bytes
                for a, s in zip(ds.archives, statuses)
                if s.ok and not s.content_length
            )
            budget = measured + fallback
        plans.append(
            DatasetPlan(
                dataset=ds,
                url_statuses=tuple(statuses),
                budget_bytes=budget,
                directory=dataset_dir(dataset_root, ds),
            )
        )
    return plans


def log_plan(plans: Sequence[DatasetPlan], min_free_bytes: int,
             dataset_root: Path) -> int:
    """Emit the dry-run report.

    :param plans: The plans to report.
    :param min_free_bytes: The configured free-space floor.
    :param dataset_root: The staging volume.
    :return: The total byte budget across fetchable datasets.
    """
    state = read_disk_state(dataset_root)
    logger.info("DRY RUN -- nothing will be written.")
    logger.info("staging root: %s", staging_root(dataset_root))
    logger.info(
        "volume: %.1f GB free of %.1f GB; floor %.1f GB",
        state.free_bytes / _GB, state.total_bytes / _GB, min_free_bytes / _GB,
    )
    total = 0
    dead: List[str] = []
    for plan in plans:
        ds = plan.dataset
        logger.info("")
        logger.info(
            "[%s] %s -- %s (%s, %s)",
            ds.task, ds.name, ds.availability, ds.role, ds.licence,
        )
        logger.info("    dir: %s", plan.directory)
        if not ds.is_fetchable:
            logger.info("    NOT FETCHED: %s", ds.reason)
            if ds.manual_url:
                logger.info("    manual source: %s", ds.manual_url)
            continue
        for status in plan.url_statuses:
            size = status.content_length
            logger.info(
                "    %-7s %12s  %s",
                "OK" if status.ok else f"HTTP{status.status_code}",
                f"{size:,}" if size else "unknown",
                status.url,
            )
            if not status.ok:
                dead.append(f"{ds.name}/{status.filename} <{status.url}>")
        total += plan.budget_bytes
        logger.info(
            "    dataset budget: %s B (%.3f GB)",
            f"{plan.budget_bytes:,}", plan.budget_bytes / _GB,
        )
    logger.info("")
    logger.info(
        "TOTAL DOWNLOAD BUDGET: %s B (%.3f GB)", f"{total:,}", total / _GB
    )
    if dead:
        logger.warning("DEAD URLS (%d), by name:", len(dead))
        for item in dead:
            logger.warning("    %s", item)
    else:
        logger.info("every verified URL is live")
    return total


@dataclass
class DatasetReport:
    """What a real run did to one dataset."""

    dataset: str
    task: str
    status: str  # "staged" | "slot" | "refused" | "skipped"
    archives: Tuple[ArchiveResult, ...] = ()
    sidecars: Optional[SidecarResult] = None
    detail: str = ""

    @property
    def dead_urls(self) -> Tuple[ArchiveResult, ...]:
        """The archives whose URL did not resolve."""
        return tuple(a for a in self.archives if a.status == "dead-url")


def stage_all(
    dataset_root: Path,
    datasets: Sequence[Dataset],
    *,
    min_free_bytes: int,
    timeout: float = DEFAULT_TIMEOUT_S,
    do_sidecars: bool = True,
    only_sidecars: bool = False,
    overwrite_sidecars: bool = False,
) -> List[DatasetReport]:
    """Stage every dataset, then precompute prompt sidecars.

    A dataset that refuses on disk budget, or whose URLs are dead, is reported
    and the run continues -- one dead university page must not cost the other
    ten years.

    :param dataset_root: The staging volume.
    :param datasets: Datasets to stage.
    :param min_free_bytes: Free-space floor.
    :param timeout: Per-request timeout in seconds.
    :param do_sidecars: Whether to precompute prompts after staging.
    :param only_sidecars: Skip all downloading; only (re)build sidecars.
    :param overwrite_sidecars: Recompute sidecars that already exist.
    :return: One :class:`DatasetReport` per dataset.
    """
    reports: List[DatasetReport] = []
    for ds in datasets:
        target = dataset_dir(dataset_root, ds)
        if not ds.is_fetchable:
            if not only_sidecars:
                write_slot_readme(dataset_root, ds)
            report = DatasetReport(
                dataset=ds.name,
                task=ds.task,
                status="slot",
                detail=ds.reason,
            )
            if do_sidecars:
                report.sidecars = precompute_sidecars(
                    dataset_root, ds, overwrite=overwrite_sidecars
                )
            reports.append(report)
            continue

        results: Tuple[ArchiveResult, ...] = ()
        if not only_sidecars:
            try:
                check_disk_budget(dataset_root, ds.approx_bytes, min_free_bytes)
            except DiskBudgetError as exc:
                logger.error("%s", exc)
                reports.append(
                    DatasetReport(
                        dataset=ds.name,
                        task=ds.task,
                        status="refused",
                        detail=str(exc),
                    )
                )
                continue
            archives_dir = target / ARCHIVE_DIRNAME
            collected: List[ArchiveResult] = []
            for archive in ds.archives:
                result = stage_archive(
                    archive, archives_dir, target, timeout=timeout
                )
                level = logger.info if result.status in (
                    "staged", "already-staged"
                ) else logger.warning
                level(
                    "[%s/%s] %s: %s %s",
                    ds.task, ds.name, archive.filename, result.status,
                    result.detail,
                )
                collected.append(result)
            results = tuple(collected)

        report = DatasetReport(
            dataset=ds.name, task=ds.task, status="staged", archives=results
        )
        if do_sidecars:
            report.sidecars = precompute_sidecars(
                dataset_root, ds, overwrite=overwrite_sidecars
            )
        reports.append(report)
    return reports


def log_reports(reports: Sequence[DatasetReport], dataset_root: Path) -> None:
    """Emit the end-of-run summary, naming every dead URL and empty slot."""
    logger.info("")
    logger.info("=== staging summary ===")
    dead: List[str] = []
    sidecar_seconds = 0.0
    for report in reports:
        line = f"[{report.task}] {report.dataset}: {report.status}"
        if report.archives:
            counts: Dict[str, int] = {}
            for a in report.archives:
                counts[a.status] = counts.get(a.status, 0) + 1
            line += " (" + ", ".join(
                f"{k}={v}" for k, v in sorted(counts.items())
            ) + ")"
        logger.info("%s", line)
        for a in report.dead_urls:
            dead.append(f"{report.dataset}/{a.filename} <{a.url}> {a.detail}")
        if report.status == "slot":
            logger.info("    empty by design: %s", report.detail)
        if report.sidecars is not None:
            s = report.sidecars
            sidecar_seconds += s.seconds
            logger.info(
                "    sidecars: %s written=%d present=%d failed=%d %.1fs %s",
                s.status, s.written, s.already_present, s.failed, s.seconds,
                s.detail,
            )
    if dead:
        logger.warning("DEAD URLS (%d), by name:", len(dead))
        for item in dead:
            logger.warning("    %s", item)
    logger.info("prompt-sidecar precompute total: %.1f s", sidecar_seconds)
    state = read_disk_state(dataset_root)
    logger.info(
        "volume after staging: %.1f GB free of %.1f GB",
        state.free_bytes / _GB, state.total_bytes / _GB,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """The argument parser. Constructing it allocates nothing but memory."""
    parser = argparse.ArgumentParser(
        prog="prepare_doc_res_data",
        description=(
            "Stage the confirmed-available DocRes corpora under "
            "<root>/doc_res/<task>/<dataset>/ and precompute DTSPrompt "
            "sidecars. Additive only: this script never deletes anything."
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"dataset volume (default: {DEFAULT_DATASET_ROOT})",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help=(
            "comma-separated task names to stage (default: all). Legal: "
            + ",".join(TASKS)
        ),
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help=(
            "comma-separated dataset names to stage (default: all). Legal: "
            + ",".join(dataset_names())
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print every URL and the byte budget; write nothing",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=DEFAULT_MIN_FREE_GB,
        help=(
            "refuse to stage a dataset that would take the volume below this "
            f"(default: {DEFAULT_MIN_FREE_GB})"
        ),
    )
    parser.add_argument(
        "--skip-sidecars",
        action="store_true",
        help="stage the data but do not precompute DTSPrompt sidecars",
    )
    parser.add_argument(
        "--only-sidecars",
        action="store_true",
        help="download nothing; only (re)build sidecars for staged images",
    )
    parser.add_argument(
        "--overwrite-sidecars",
        action="store_true",
        help="recompute sidecars that already exist",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help=f"per-request timeout in seconds (default: {DEFAULT_TIMEOUT_S})",
    )
    return parser


def _csv(value: Optional[str]) -> Optional[List[str]]:
    """Split a comma-separated option, or return ``None``."""
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point.

    The FIRST statement parses ``argv``, so ``--help`` prints its ``usage:``
    line and allocates nothing else.

    :param argv: Command line, or ``None`` for ``sys.argv[1:]``.
    :return: Process exit code; ``0`` on success.
    """
    args = build_parser().parse_args(argv)

    datasets = select_datasets(_csv(args.tasks), _csv(args.datasets))
    if not datasets:
        logger.error("no dataset matches the given --tasks/--datasets filters")
        return 2

    min_free_bytes = int(args.min_free_gb * _GB)
    backends = extractor_backends()
    if not backends["libarchive"]:
        logger.warning(
            "libarchive-c is not importable: .rar and .7z archives cannot be "
            "extracted (6 DIBCO years ship .rar, 1 ships .7z). Install it with "
            "`pip install libarchive-c`; .zip years are unaffected."
        )

    if args.dry_run:
        plans = build_plan(args.root, datasets, timeout=args.timeout)
        log_plan(plans, min_free_bytes, args.root)
        return 0

    reports = stage_all(
        args.root,
        datasets,
        min_free_bytes=min_free_bytes,
        timeout=args.timeout,
        do_sidecars=not args.skip_sidecars,
        only_sidecars=args.only_sidecars,
        overwrite_sidecars=args.overwrite_sidecars,
    )
    log_reports(reports, args.root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
