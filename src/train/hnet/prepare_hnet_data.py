"""Stage (or explain the absence of) the H-Net byte-level pretraining corpora.

Target layout, matching the sibling convention under
``/media/arxwn/data0_4tb/datasets/``::

    <root>/fineweb_edu/sample-10BT/
        README.md               <- written for a slot this script does not fill
        <parquet shards>        <- ONLY if a human passes --download
    <root>/fineweb_edu/sample-10BT.ok
                                <- completion marker, written LAST, and only by
                                   the download path; a GATED slot never has one

What this script will and will not fetch
----------------------------------------

======================  ==========  =======================================
Corpus                  Subset      Availability
======================  ==========  =======================================
Wikipedia (wikimedia)   20231101.en ``staged`` -- already on the volume,
                                    41/41 shards, 20 GB, consumed through
                                    :func:`dl_techniques.datasets.nlp.load_wikipedia_train_val`.
                                    This script neither fetches nor
                                    restructures it, and writes no file into
                                    its directory: it is not ours to change.
FineWeb-Edu             sample/10BT ``gated`` -- ~28.5 GB, ODC-BY. NOT a
                                    technical failure and NOT a dead host:
                                    the transfer is user-gated. The slot is
                                    created empty with a README that says so.
======================  ==========  =======================================

**No code path in this module starts a transfer without an explicit
``--download``.** ``--dry-run`` reports the plan and writes the GATED slot
READMEs; the default (no flag) does exactly the same thing. Neither reaches
:func:`_snapshot_download`, which is the single seam that touches the network
and which :func:`fetch_dataset` refuses to call unless ``allow_download=True``.

A separate, opt-in ``--check-remote`` performs a metadata-only
``HfApi().dataset_info`` call to re-derive the published shard sizes. It reads
metadata and never transfers a corpus byte; it is off by default so that a
plain ``--dry-run`` is provably offline.

Safety rules this module obeys, and never relaxes:

* It is additive. It deletes nothing, anywhere -- in particular nothing under
  repo-root ``results/`` and nothing under ``/media/arxwn/data0_4tb``.
* It never stages onto ``/media/arxwn/data_fast``; that is the repo SSD, not
  dataset storage.
* It checks free disk BEFORE the first write, and refuses rather than filling
  the volume.
* Every payload lands through ``.part`` -> :func:`os.replace`, and the ``.ok``
  marker is written LAST, so an interrupted fetch can never be mistaken for a
  complete one.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------------
# Constants -- a CLI flag with a DEFAULT_* constant, never an environment
# variable (the repo's dataset-root convention: nlp.py's
# DEFAULT_WIKIPEDIA_CACHE_DIR, doc_res's DocResTrainingConfig.dataset_root).
# ---------------------------------------------------------------------------

DEFAULT_DATASET_ROOT: Path = Path("/media/arxwn/data0_4tb/datasets")
"""Staging volume. 3.6 TB spinning volume; the dataset home for this machine."""

FORBIDDEN_ROOT_PREFIX: str = "/media/arxwn/data_fast"
"""The repo/working SSD. Staging a corpus here is a recorded past incident."""

DEFAULT_MIN_FREE_GB: float = 100.0
"""Free-space floor the volume must never cross because of this script."""

_GB: float = 1024.0 ** 3

AVAIL_STAGED: str = "staged"
"""Already present on the volume; this script does not fetch or modify it."""

AVAIL_GATED: str = "gated"
"""Fetchable in principle, deliberately not fetched. Slot + README only."""

AVAILABILITIES: Tuple[str, ...] = (AVAIL_STAGED, AVAIL_GATED)


class DiskBudgetError(RuntimeError):
    """Staging would take the volume below its free-space floor."""


class DownloadNotPermittedError(RuntimeError):
    """A transfer was requested without the explicit opt-in flag."""


class StagingRootError(ValueError):
    """The requested root is not a legal dataset volume."""


# ---------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Dataset:
    """One corpus slot: staged already, or explicitly not, with the reason.

    :param name: Selector name, also the ``--datasets`` key.
    :param subset: The subset/config identifier as the source publishes it
        (``"sample/10BT"``, ``"20231101.en"``).
    :param availability: One of :data:`AVAILABILITIES`.
    :param relative_dir: Slot directory relative to the dataset root.
    :param approx_bytes: Published (vendor-stated) byte budget of the subset.
    :param source_url: The page a human opens to see the corpus.
    :param repo_id: Hugging Face repo id, for the fetch command in the README.
    :param licence: Licence as published, or ``"unstated"``.
    :param loader: The in-repo callable that consumes the staged corpus.
    :param reason: Why this script does not fetch it. Required for BOTH
        availabilities -- a ``staged`` slot must say why no fetch is needed,
        and a ``gated`` slot must say why one was not performed.
    """

    name: str
    subset: str
    availability: str
    relative_dir: str
    approx_bytes: int
    source_url: str
    repo_id: str
    licence: str
    loader: str
    reason: str

    def __post_init__(self) -> None:
        if self.availability not in AVAILABILITIES:
            raise ValueError(
                f"dataset {self.name!r} declares unknown availability "
                f"{self.availability!r}; legal: {sorted(AVAILABILITIES)}"
            )
        if not self.reason:
            raise ValueError(
                f"dataset {self.name!r} must state why this script does not "
                "fetch it; an unexplained slot is exactly the silent absence "
                "this manifest exists to prevent"
            )
        if self.approx_bytes <= 0:
            raise ValueError(
                f"dataset {self.name!r} declares a non-positive byte budget "
                f"{self.approx_bytes!r}; the disk check needs a real number"
            )

    @property
    def writes_a_slot_readme(self) -> bool:
        """Whether this script creates a directory and a README for this slot.

        ``False`` for an already-staged corpus: its directory belongs to
        whoever staged it, and dropping our README into it would restructure a
        tree this port does not own.
        """
        return self.availability == AVAIL_GATED


# Vendor-stated sizes. FineWeb-Edu's ~28.5 GB is the HF dataset card's figure
# for sample/10BT and has NOT been independently measured here -- no transfer
# and no metadata call happens by default. `--check-remote` re-derives it.
MANIFEST: Tuple[Dataset, ...] = (
    Dataset(
        name="wikipedia",
        subset="20231101.en",
        availability=AVAIL_STAGED,
        relative_dir="wikipedia",
        approx_bytes=20 * int(_GB),
        source_url="https://huggingface.co/datasets/wikimedia/wikipedia",
        repo_id="wikimedia/wikipedia",
        licence="CC BY-SA 4.0 / GFDL (as published on the dataset card)",
        loader="dl_techniques.datasets.nlp.load_wikipedia_train_val",
        reason=(
            "ALREADY STAGED: 41/41 train shards are present under "
            "wikipedia/wikimedia___wikipedia/20231101.en/ (du -sh = 20G) and "
            "five trainers already read them through "
            "load_wikipedia_train_val. Nothing is fetched, and no file is "
            "written into that directory -- it predates this port and is not "
            "ours to restructure. This is the corpus H-Net's smoke run uses."
        ),
    ),
    Dataset(
        name="fineweb_edu_sample_10bt",
        subset="sample/10BT",
        availability=AVAIL_GATED,
        relative_dir="fineweb_edu/sample-10BT",
        # 28.5 GB, the dataset card's stated size for the 10B-token sample.
        approx_bytes=30_601_641_984,
        source_url=(
            "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/tree/"
            "main/sample/10BT"
        ),
        repo_id="HuggingFaceFW/fineweb-edu",
        licence="ODC-BY (as published on the dataset card)",
        loader="dl_techniques.datasets.byte_lm.build_byte_clm_dataset",
        reason=(
            "USER-GATED, not a technical failure: the source is live, the "
            "licence permits the copy and the volume has room. The ~28.5 GB "
            "transfer was deliberately withheld pending an explicit user "
            "decision (D-002 of "
            "plans/plan-2026-09-09T042752-6d66ac56/decisions.md), so the port "
            "trains on the already-staged Wikipedia corpus instead. Nothing "
            "is broken here; nothing was partially downloaded."
        ),
    ),
)


def dataset_names() -> Tuple[str, ...]:
    """Selector names, in manifest order."""
    return tuple(ds.name for ds in MANIFEST)


def get_dataset(name: str) -> Dataset:
    """Look a manifest entry up by name.

    :param name: A member of :func:`dataset_names`.
    :return: The :class:`Dataset`.
    :raises KeyError: If no entry has that name; the message lists the legal
        keys, so a typo is never a silent no-op.
    """
    for ds in MANIFEST:
        if ds.name == name:
            return ds
    raise KeyError(
        f"no dataset named {name!r}; legal: {list(dataset_names())}"
    )


def select_datasets(names: Optional[Sequence[str]]) -> List[Dataset]:
    """Resolve a ``--datasets`` filter to manifest entries.

    :param names: Selected names, or ``None`` for the whole manifest.
    :return: The selected entries, in manifest order.
    """
    if names is None:
        return list(MANIFEST)
    wanted = list(names)
    return [get_dataset(name) for name in wanted]


def dataset_dir(dataset_root: Path, ds: Dataset) -> Path:
    """The slot directory for ``ds`` under ``dataset_root``."""
    return Path(dataset_root) / ds.relative_dir


def marker_path(dataset_root: Path, ds: Dataset) -> Path:
    """The ``.ok`` completion marker for ``ds``.

    It lives BESIDE the slot directory rather than inside it, so that a GATED
    slot -- which has no marker at all -- contains exactly one file, its
    README, and ``du -sb`` on it equals that README's size.
    """
    target = dataset_dir(dataset_root, ds)
    return target.with_name(target.name + ".ok")


# ---------------------------------------------------------------------------
# Disk
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiskState:
    """Filesystem capacity, in bytes."""

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


def check_staging_root(dataset_root: Path) -> Path:
    """Refuse a root that is not a dataset volume.

    :param dataset_root: The requested root.
    :return: The root, resolved.
    :raises StagingRootError: If the root is under
        :data:`FORBIDDEN_ROOT_PREFIX` -- the repo SSD, where a 28 GB corpus
        copy was staged once before and had to be deleted.
    """
    resolved = Path(dataset_root).expanduser().absolute()
    forbidden = Path(FORBIDDEN_ROOT_PREFIX)
    if resolved == forbidden or forbidden in resolved.parents:
        raise StagingRootError(
            f"refusing to stage into {resolved}: {FORBIDDEN_ROOT_PREFIX} is "
            "the repo/working SSD, not dataset storage. Datasets live on "
            "/media/arxwn/data0_4tb."
        )
    return resolved


def check_disk_budget(
    dataset_root: Path, needed_bytes: int, min_free_bytes: int
) -> DiskState:
    """Refuse rather than fill the volume.

    :param dataset_root: The staging volume.
    :param needed_bytes: What the next write would consume.
    :param min_free_bytes: The floor free space must never cross.
    :return: The :class:`DiskState` that was read.
    :raises DiskBudgetError: If the write would cross the floor. The message
        names the shortfall in bytes and GB.
    """
    state = read_disk_state(dataset_root)
    remaining = state.free_bytes - int(needed_bytes)
    if remaining < int(min_free_bytes):
        shortfall = int(min_free_bytes) - remaining
        raise DiskBudgetError(
            f"refusing to write into {dataset_root}: it needs "
            f"{int(needed_bytes):,} B ({needed_bytes / _GB:.2f} GB) and only "
            f"{state.free_bytes:,} B ({state.free_bytes / _GB:.2f} GB) are "
            f"free; that is {shortfall:,} B ({shortfall / _GB:.2f} GB) short "
            f"of the {int(min_free_bytes):,} B floor"
        )
    return state


# ---------------------------------------------------------------------------
# Atomic writes and the completion marker
# ---------------------------------------------------------------------------


def atomic_write_text(path: Path, text: str) -> int:
    """Write ``text`` to ``path`` through a ``.part`` temporary.

    A reader can only ever see the whole file or no file, never a prefix.

    :param path: Final destination.
    :param text: UTF-8 payload.
    :return: The number of bytes the final file holds.
    """
    part = path.with_name(path.name + ".part")
    part.write_text(text, encoding="utf-8")
    os.replace(part, path)
    return path.stat().st_size


def write_marker(dataset_root: Path, ds: Dataset, payload: Dict[str, object]) -> Path:
    """Write the ``.ok`` completion marker for ``ds``.

    :param dataset_root: The staging volume.
    :param ds: The fetched dataset.
    :param payload: What was verified; rendered as ``key: value`` lines.
    :return: The marker path.
    """
    marker = marker_path(dataset_root, ds)
    lines = [f"{key}: {value}" for key, value in payload.items()]
    atomic_write_text(marker, "\n".join(lines) + "\n")
    return marker


def is_complete(dataset_root: Path, ds: Dataset) -> bool:
    """Whether a previous run finished fetching ``ds``.

    The marker is the ONLY completeness signal. A populated directory without
    one is an interrupted fetch, not a corpus.
    """
    return marker_path(dataset_root, ds).exists()


# ---------------------------------------------------------------------------
# Slot READMEs
# ---------------------------------------------------------------------------


def _staged_slot_lines(dataset_root: Path) -> List[str]:
    """The already-staged-corpus section every GATED README carries.

    The Wikipedia slot is recorded HERE, in the README of the slot this script
    owns, rather than by writing a file into the Wikipedia directory -- that
    tree predates this port.
    """
    lines: List[str] = []
    for ds in MANIFEST:
        if ds.availability != AVAIL_STAGED:
            continue
        lines += [
            f"- **{ds.name}** (`{ds.subset}`) -- ALREADY STAGED at "
            f"`{dataset_dir(dataset_root, ds)}`, ~{ds.approx_bytes / _GB:.0f} GB, "
            "41/41 train shards.",
            f"  Consumed through `{ds.loader}`. This is the corpus the H-Net "
            "smoke run actually trains on.",
        ]
    return lines


def slot_readme_text(dataset_root: Path, ds: Dataset) -> str:
    """Render the README for a slot this script does not fill.

    :param dataset_root: The staging volume, quoted in the re-run command.
    :param ds: A non-staged dataset.
    :return: The README text.
    """
    target = dataset_dir(dataset_root, ds)
    command = (
        "python -m train.hnet.prepare_hnet_data "
        f"--dataset-root {dataset_root} --datasets {ds.name} --download"
    )
    lines = [
        f"# {ds.name} ({ds.repo_id}, subset `{ds.subset}`)",
        "",
        f"**Status: {ds.availability.upper()} -- this directory is empty by design.**",
        "",
        f"- Hugging Face repo: `{ds.repo_id}`, subset `{ds.subset}`",
        f"- Source: {ds.source_url}",
        f"- Licence: {ds.licence}",
        f"- Budget if obtained: {ds.approx_bytes / _GB:.2f} GB",
        f"- Consumed through: `{ds.loader}`",
        "",
        "## Why `prepare_hnet_data.py` did not fill it",
        "",
        ds.reason,
        "",
        "## The corpus this port actually uses",
        "",
    ]
    lines += _staged_slot_lines(dataset_root)
    lines += [
        "",
        "## What to do to populate this slot",
        "",
        "Nothing else needs changing -- the manifest entry and the trainer's "
        "dataset-root plumbing already point here. Run:",
        "",
        "```",
        command,
        "```",
        "",
        "The fetch writes into `<slot>.part`, `os.replace`s it into place and "
        f"only then writes the `{marker_path(dataset_root, ds).name}` marker, "
        "so an interrupted transfer never looks complete. A second run "
        "short-circuits on that marker and performs no network call.",
        "",
    ]
    return "\n".join(lines)


def write_slot_readme(dataset_root: Path, ds: Dataset) -> Path:
    """Create the directory for a non-staged slot and explain it.

    The free-disk check runs BEFORE the directory is created, so a full volume
    is refused rather than nudged over its floor by a stray mkdir.

    :param dataset_root: The staging volume.
    :param ds: A dataset with :attr:`Dataset.writes_a_slot_readme`.
    :return: The README path.
    :raises ValueError: If ``ds`` is an already-staged corpus. Writing our
        README into someone else's staged tree would restructure it.
    """
    if not ds.writes_a_slot_readme:
        raise ValueError(
            f"refusing to write a README into {ds.name!r}: it is "
            f"{ds.availability} and its directory is not ours to restructure"
        )
    target = dataset_dir(dataset_root, ds)
    target.mkdir(parents=True, exist_ok=True)
    readme = target / "README.md"
    atomic_write_text(readme, slot_readme_text(dataset_root, ds))
    return readme


# ---------------------------------------------------------------------------
# The network seams
# ---------------------------------------------------------------------------


def _dataset_info(repo_id: str):  # pragma: no cover - network, opt-in only
    """Metadata-only Hugging Face lookup. Transfers no corpus byte.

    Deliberately separate from :func:`_snapshot_download`: this reads the file
    listing, the other one moves ~28.5 GB. Only ``--check-remote`` calls it.

    :param repo_id: Hugging Face dataset repo id.
    :return: The ``huggingface_hub`` ``DatasetInfo``.
    """
    from huggingface_hub import HfApi

    return HfApi().dataset_info(repo_id, files_metadata=True)


def _snapshot_download(
    repo_id: str, allow_patterns: str, local_dir: Path
):  # pragma: no cover - the transfer seam; never called by the default path
    """THE transfer seam. Everything that moves corpus bytes goes through here.

    Isolating it in one module-level function is what lets the suite prove, by
    replacing it with a landmine, that no default code path can start a
    download.

    :param repo_id: Hugging Face dataset repo id.
    :param allow_patterns: Glob restricting the snapshot to one subset.
    :param local_dir: Destination -- always a ``.part`` directory.
    :return: The local snapshot path.
    """
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=allow_patterns,
        local_dir=str(local_dir),
    )


def remote_size_bytes(ds: Dataset) -> int:  # pragma: no cover - opt-in network
    """Re-derive ``ds``'s byte budget from the published file listing.

    :param ds: A manifest entry.
    :return: Summed size of the subset's files, or ``0`` when the hub does not
        publish per-file sizes.
    """
    prefix = ds.subset.rstrip("/") + "/"
    info = _dataset_info(ds.repo_id)
    total = 0
    for sibling in getattr(info, "siblings", None) or ():
        name = getattr(sibling, "rfilename", "")
        if name.startswith(prefix):
            total += int(getattr(sibling, "size", 0) or 0)
    return total


# ---------------------------------------------------------------------------
# Staging
# ---------------------------------------------------------------------------


def fetch_dataset(
    dataset_root: Path,
    ds: Dataset,
    *,
    allow_download: bool = False,
    min_free_bytes: int = int(DEFAULT_MIN_FREE_GB * _GB),
) -> str:
    """Fetch ``ds``'s corpus bytes. The ONLY function that can transfer.

    # DECISION plan-2026-09-09T042752-6d66ac56/D-025
    # ``allow_download`` defaults to False and the refusal happens BEFORE the
    # disk check, the mkdir and the seam call -- so an omitted flag cannot
    # download, and a caller that forgets it gets an exception rather than a
    # 28.5 GB surprise. Do NOT "helpfully" default this to True, and do NOT
    # infer it from `not args.dry_run`: the FineWeb-Edu transfer is USER-gated
    # by D-002, and a not-dry-run default would make the gate a naming
    # convention instead of a control-flow fact. See D-025 in
    # plans/plan-2026-09-09T042752-6d66ac56/decisions.md.

    :param dataset_root: The staging volume.
    :param ds: The dataset to fetch.
    :param allow_download: Explicit opt-in. Without it nothing happens.
    :param min_free_bytes: Free-space floor.
    :return: ``"present"`` when the marker already existed, else ``"fetched"``.
    :raises DownloadNotPermittedError: If ``allow_download`` is false.
    :raises DiskBudgetError: If the fetch would cross the free-space floor.
    """
    if not allow_download:
        raise DownloadNotPermittedError(
            f"refusing to fetch {ds.name!r} ({ds.approx_bytes / _GB:.2f} GB): "
            "this transfer is user-gated and requires the explicit --download "
            "flag; no network call was made"
        )
    if is_complete(dataset_root, ds):
        logger.info("%s: already complete (marker present); nothing to do", ds.name)
        return "present"

    check_staging_root(dataset_root)
    check_disk_budget(dataset_root, ds.approx_bytes, min_free_bytes)

    target = dataset_dir(dataset_root, ds)
    part = target.with_name(target.name + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    part.mkdir(parents=True, exist_ok=True)

    _snapshot_download(ds.repo_id, ds.subset.rstrip("/") + "/*", part)

    # Payload first, atomically. The marker is written LAST, after the payload
    # is already at its final name, so a crash anywhere above leaves a `.part`
    # directory and NO marker -- visibly incomplete, never mistaken for done.
    os.replace(part, target)
    write_marker(
        dataset_root,
        ds,
        {
            "repo_id": ds.repo_id,
            "subset": ds.subset,
            "files": sum(1 for _ in target.rglob("*") if _.is_file()),
            "bytes": sum(p.stat().st_size for p in target.rglob("*") if p.is_file()),
        },
    )
    return "fetched"


def stage_dataset(
    dataset_root: Path,
    ds: Dataset,
    *,
    allow_download: bool = False,
    min_free_bytes: int = int(DEFAULT_MIN_FREE_GB * _GB),
) -> str:
    """Bring one manifest entry to its documented on-disk state.

    :param dataset_root: The staging volume.
    :param ds: The manifest entry.
    :param allow_download: Explicit transfer opt-in; see :func:`fetch_dataset`.
    :param min_free_bytes: Free-space floor.
    :return: ``"staged"`` (already present, untouched), ``"slot"`` (directory +
        README written), ``"fetched"`` or ``"present"``.
    """
    if ds.availability == AVAIL_STAGED:
        logger.info(
            "%s: already staged at %s -- not fetched, not modified",
            ds.name,
            dataset_dir(dataset_root, ds),
        )
        return "staged"

    if allow_download:
        return fetch_dataset(
            dataset_root,
            ds,
            allow_download=True,
            min_free_bytes=min_free_bytes,
        )

    check_staging_root(dataset_root)
    # Free space is checked BEFORE the mkdir, even for a README-sized write.
    check_disk_budget(
        dataset_root, len(slot_readme_text(dataset_root, ds).encode("utf-8")),
        min_free_bytes,
    )
    readme = write_slot_readme(dataset_root, ds)
    logger.info(
        "%s: GATED -- wrote the empty-slot README at %s (%d B). No corpus "
        "byte was transferred.",
        ds.name,
        readme,
        readme.stat().st_size,
    )
    return "slot"


def stage_all(
    dataset_root: Path,
    datasets: Sequence[Dataset],
    *,
    allow_download: bool = False,
    min_free_bytes: int = int(DEFAULT_MIN_FREE_GB * _GB),
) -> Dict[str, str]:
    """Stage each selected entry; return ``{name: status}``."""
    return {
        ds.name: stage_dataset(
            dataset_root,
            ds,
            allow_download=allow_download,
            min_free_bytes=min_free_bytes,
        )
        for ds in datasets
    }


def log_plan(
    dataset_root: Path,
    datasets: Sequence[Dataset],
    min_free_bytes: int,
    remote_sizes: Optional[Dict[str, int]] = None,
) -> None:
    """Print what a run would do, without doing any of it."""
    state = read_disk_state(dataset_root)
    logger.info("dataset root: %s", dataset_root)
    logger.info(
        "free: %.2f GB; floor: %.2f GB",
        state.free_bytes / _GB,
        min_free_bytes / _GB,
    )
    for ds in datasets:
        logger.info(
            "  %-24s %-7s %8.2f GB  %s",
            ds.name,
            ds.availability,
            ds.approx_bytes / _GB,
            dataset_dir(dataset_root, ds),
        )
        logger.info("      %s", ds.reason)
        if remote_sizes and ds.name in remote_sizes:
            logger.info(
                "      remote metadata says %.2f GB",
                remote_sizes[ds.name] / _GB,
            )
    logger.info(
        "no corpus byte is transferred without --download; --dry-run writes "
        "only the GATED slot README(s)"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """The argument parser. Constructing it allocates nothing but memory."""
    parser = argparse.ArgumentParser(
        prog="prepare_hnet_data",
        description=(
            "Record and stage the H-Net byte-level corpora under "
            "<root>/. Additive only: this script never deletes anything, and "
            "never transfers a corpus byte without --download."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"dataset volume (default: {DEFAULT_DATASET_ROOT})",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help=(
            "comma-separated corpora to act on (default: all). Legal: "
            + ",".join(dataset_names())
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "print the plan and write the GATED slot README(s); make no "
            "network call at all"
        ),
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help=(
            "THE transfer opt-in. Without this flag no code path can start a "
            "download; with it, the gated corpora are fetched"
        ),
    )
    parser.add_argument(
        "--check-remote",
        action="store_true",
        help=(
            "metadata only: re-derive the published byte budget from the Hub "
            "file listing. Transfers no corpus byte"
        ),
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=DEFAULT_MIN_FREE_GB,
        help=(
            "refuse to write if it would take the volume below this "
            f"(default: {DEFAULT_MIN_FREE_GB})"
        ),
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

    try:
        datasets = select_datasets(_csv(args.datasets))
    except KeyError as exc:
        logger.error("%s", exc)
        return 2
    if not datasets:
        logger.error("no dataset matches the given --datasets filter")
        return 2

    try:
        root = check_staging_root(args.dataset_root)
    except StagingRootError as exc:
        logger.error("%s", exc)
        return 2

    min_free_bytes = int(args.min_free_gb * _GB)

    remote_sizes: Optional[Dict[str, int]] = None
    if args.check_remote:
        remote_sizes = {}
        for ds in datasets:
            try:
                remote_sizes[ds.name] = remote_size_bytes(ds)
            except Exception as exc:  # noqa: BLE001 - reported, never fatal
                logger.warning(
                    "%s: remote metadata lookup failed (%s); the slot README "
                    "still gets written with the manifest's stated budget",
                    ds.name,
                    exc,
                )

    log_plan(root, datasets, min_free_bytes, remote_sizes)

    # --dry-run OVERRIDES --download. The two flags are not independent: a
    # dry run that could still transfer 28.5 GB because a second flag happened
    # to be on the command line is not a dry run. This is the guard
    # `test_dry_run_overrides_download_and_fetches_nothing` pins.
    allow_download = bool(args.download) and not bool(args.dry_run)
    if args.download and args.dry_run:
        logger.warning(
            "--dry-run and --download were both given; --dry-run wins and "
            "nothing is transferred"
        )

    try:
        statuses = stage_all(
            root,
            datasets,
            allow_download=allow_download,
            min_free_bytes=min_free_bytes,
        )
    except (DiskBudgetError, StagingRootError, DownloadNotPermittedError) as exc:
        logger.error("%s", exc)
        return 1

    for name, status in statuses.items():
        logger.info("%-24s %s", name, status)
    return 0


if __name__ == "__main__":
    sys.exit(main())
