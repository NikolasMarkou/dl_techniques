"""Guards for ``train.doc_scanner.prepare_doc_scanner_data``.

THE FAILURE THIS FILE EXISTS TO PREVENT
---------------------------------------
A 27.5 GB download that stops early, gets renamed anyway, and is reported as
staged. Every consumer downstream then reads a SILENTLY TRUNCATED corpus with
no error anywhere -- the archive opens, its central directory lists all 182,492
entries, and only the members that were never received fail, one at a time,
deep inside a training epoch. :class:`TestTheCompletenessProof` reproduces that
exact shape executably: ``test_a_full_length_body_that_fails_crc_is_refused``
builds a zip whose bytes are all present, whose ``infolist()`` is intact, and
one of whose members has a flipped byte -- and asserts the staging call REFUSES
it and leaves no ``.zip`` and no ``.ok`` behind. The companion assertion that
``infolist()`` still reports every member is what makes the CRC pass necessary
rather than merely thorough.

NO TEST HERE TOUCHES THE NETWORK. The two seams that reach it -- ``_http_head``
and ``_open_stream`` -- are monkeypatched, and the ``network_is_a_landmine``
fixture patches them to RAISE, so "this run fetched nothing" is asserted rather
than assumed. :func:`test_the_landmine_fixture_can_actually_fire` proves that
fixture discriminates instead of merely being green.

NO TEST HERE WRITES TO THE REAL DATASET VOLUME. Everything lands under
``tmp_path``. The one arm that reads the real staged corpus,
:class:`TestTheRealStagedCorpus`, is ``skipif``-gated on its presence, is
read-only, and asserts the no-op with the network mined.
"""

from __future__ import annotations

import json
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from train.doc_scanner import common
from train.doc_scanner import prepare_doc_scanner_data as prep


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_zip(path: Path, payload: bytes = b"hello world " * 50) -> Path:
    """A small, valid, STORED zip with two members.

    Stored (not deflated) so that a single flipped byte inside the member data
    is a pure CRC failure rather than an inflate failure -- the harder of the
    two to catch, and the one a size check is blindest to.
    """
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("uvdoc/img/000.txt", payload)
        archive.writestr("uvdoc/grid2d/000.txt", b"second")
    return path


@pytest.fixture
def good_zip(tmp_path: Path) -> bytes:
    """The bytes of a valid two-member zip."""
    return _build_zip(tmp_path / "_source.zip").read_bytes()


@pytest.fixture
def crc_broken_zip(tmp_path: Path) -> bytes:
    """A FULL-LENGTH zip with one member's data corrupted.

    Same byte count as the good one; intact central directory; one bad CRC.
    """
    path = _build_zip(tmp_path / "_broken.zip")
    raw = bytearray(path.read_bytes())
    index = raw.find(b"hello world hello")
    assert index > 0, "fixture payload not found; the fixture is broken"
    raw[index] = ord("H")
    path.write_bytes(bytes(raw))
    return path.read_bytes()


class _FakeResponse:
    """The slice of ``requests.Response`` that :func:`prep._download` uses."""

    def __init__(self, status_code: int, body: bytes) -> None:
        self.status_code = status_code
        self.headers: Dict[str, str] = {"Content-Length": str(len(body))}
        self._body = body

    def iter_content(self, chunk_size: int = 1):
        for start in range(0, len(self._body), max(chunk_size, 1)):
            yield self._body[start:start + chunk_size]


@pytest.fixture
def server(monkeypatch, good_zip: bytes) -> Dict[str, Any]:
    """A fake ranged HTTP server, plus a record of every request it saw.

    Mutable knobs: ``body`` (what a full GET returns), ``ranges`` (whether the
    host honours ``Range``), ``cut`` (truncate the served body to N bytes, i.e.
    an interrupted transfer) and ``status`` (HEAD status code).
    """
    state: Dict[str, Any] = {
        "body": good_zip,
        "ranges": True,
        "cut": None,
        "status": 200,
        "head": 0,
        "get": 0,
        "range_headers": [],
    }

    def _head(url: str, timeout: float):
        state["head"] += 1
        return state["status"], len(state["body"]), bool(state["ranges"]), ""

    @contextmanager
    def _stream(url: str, *, timeout: float, headers: Dict[str, str]):
        state["get"] += 1
        header = headers.get("Range")
        state["range_headers"].append(header)
        start = 0
        code = 200
        if header and state["ranges"]:
            start = int(header.split("=")[1].split("-")[0])
            code = 206
        body = state["body"][start:]
        cut: Optional[int] = state["cut"]
        if cut is not None:
            body = body[:cut]
        yield _FakeResponse(code, body)

    monkeypatch.setattr(prep, "_http_head", _head)
    monkeypatch.setattr(prep, "_open_stream", _stream)
    return state


@pytest.fixture
def network_is_a_landmine(monkeypatch):
    """Make any network contact an immediate, loud failure."""

    def _boom(*args, **kwargs):
        raise AssertionError(
            "this code path reached the network; it was supposed to be a no-op"
        )

    monkeypatch.setattr(prep, "_http_head", _boom)
    monkeypatch.setattr(prep, "_open_stream", _boom)


def _assert_nothing_was_staged(root: Path) -> None:
    """No part of the staging tree exists under ``root``.

    The fixture zips live directly in ``tmp_path``, so "wrote nothing" is
    asserted against the directory this script OWNS rather than against an
    empty ``tmp_path``, which would also be satisfied by a fixture that never
    ran.
    """
    staged = root / prep.STAGING_DIRNAME
    assert not staged.exists(), sorted(str(p) for p in staged.rglob("*"))


def _snapshot(root: Path) -> Dict[str, tuple]:
    """``{relative path: (size, mtime_ns)}`` for every file under ``root``."""
    return {
        str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


# ---------------------------------------------------------------------------
# CLI contract
# ---------------------------------------------------------------------------


def test_help_prints_a_usage_line(capsys, tmp_path, network_is_a_landmine):
    """``--help`` must print a ``usage:`` LINE.

    The exit code is deliberately NOT the assertion: a script that ignores
    ``--help``, runs its whole job and falls off the end also exits 0. Only
    the usage line proves a parser saw the flag before anything else happened.
    """
    with pytest.raises(SystemExit) as excinfo:
        prep.main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    usage = [line for line in out.splitlines() if line.startswith("usage:")]
    assert usage, f"no line starts with 'usage:'; got:\n{out}"
    assert "prepare_doc_scanner_data" in usage[0]


def test_help_allocates_nothing_on_disk(tmp_path, network_is_a_landmine):
    """``--help`` creates no directory and no file."""
    with pytest.raises(SystemExit):
        prep.main(["--help", "--root", str(tmp_path)])
    assert list(tmp_path.iterdir()) == []


def test_every_documented_flag_is_declared():
    """The four flags the plan names are all really on the parser."""
    strings = {
        s for action in prep.build_parser()._actions for s in action.option_strings
    }
    assert {"--root", "--dry-run", "--min-free-gb", "--timeout"} <= strings


# ---------------------------------------------------------------------------
# The path contract
# ---------------------------------------------------------------------------


def test_the_staged_path_is_exactly_the_path_the_trainer_reads():
    """D-047: one source of truth for the corpus location.

    The trainer resolves UVDoc through ``common.DEFAULT_UVDOC_ROOT`` and this
    script WRITES it. If the two ever disagree, staging fills a directory the
    trainer never looks in and NOTHING raises -- so the equality is pinned.
    """
    assert prep.archive_path(prep.DEFAULT_DATASET_ROOT) == Path(
        common.DEFAULT_UVDOC_ROOT
    )


def test_the_layout_never_escapes_the_root(tmp_path):
    """Every path this script writes lands under ``<root>/doc_scanner``."""
    owned = (tmp_path / prep.STAGING_DIRNAME).resolve()
    for path in (
        prep.archive_path(tmp_path),
        prep.marker_path(tmp_path),
        prep.part_path(tmp_path),
    ):
        assert owned in path.resolve().parents


def test_the_part_file_never_carries_the_final_name(tmp_path):
    """An interrupted body cannot be mistaken for a complete corpus."""
    assert prep.part_path(tmp_path) != prep.archive_path(tmp_path)
    assert prep.part_path(tmp_path).name.endswith(".part")
    assert prep.marker_path(tmp_path).name.endswith(".ok")


# ---------------------------------------------------------------------------
# --dry-run
# ---------------------------------------------------------------------------


def test_dry_run_writes_nothing_at_all(tmp_path, server):
    """``--dry-run`` reports and returns without creating a single file."""
    before = _snapshot(tmp_path)
    code = prep.main(["--dry-run", "--root", str(tmp_path)])
    assert code == 0
    _assert_nothing_was_staged(tmp_path)
    assert _snapshot(tmp_path) == before
    assert server["get"] == 0


def test_dry_run_verifies_the_url_but_fetches_no_body(tmp_path, server):
    """It HEADs (that is the point) and never GETs."""
    result = prep.stage_uvdoc(tmp_path, min_free_bytes=0, dry_run=True)
    assert result.status == prep.STATUS_WOULD_STAGE
    assert server["head"] == 1
    assert server["get"] == 0
    assert "DRY RUN" in result.detail


def test_dry_run_over_a_staged_tree_touches_nothing(
    tmp_path, server, monkeypatch
):
    """A staged corpus makes ``--dry-run`` a zero-request, zero-write no-op.

    The landmine is installed AFTER staging rather than through the fixture,
    because the fixture would also mine the staging call that sets the test up.
    """
    prep.stage_uvdoc(tmp_path, min_free_bytes=0)
    before = _snapshot(tmp_path)
    assert before, "the staging fixture wrote nothing; the guard is vacuous"

    def _boom(*args, **kwargs):
        raise AssertionError("--dry-run over a staged corpus reached the network")

    monkeypatch.setattr(prep, "_http_head", _boom)
    monkeypatch.setattr(prep, "_open_stream", _boom)

    assert prep.main(["--dry-run", "--root", str(tmp_path)]) == 0
    assert _snapshot(tmp_path) == before


def test_dry_run_reports_a_dead_url_rather_than_promising_a_download(
    tmp_path, server
):
    """A dead host is named in the dry run, not discovered an hour later."""
    server["status"] = 404
    result = prep.stage_uvdoc(tmp_path, min_free_bytes=0, dry_run=True)
    assert result.status == prep.STATUS_DEAD_URL
    assert "404" in result.detail
    _assert_nothing_was_staged(tmp_path)


# ---------------------------------------------------------------------------
# Idempotency: the no-op
# ---------------------------------------------------------------------------


class TestTheIdempotentNoOp:
    """A second run must cost zero network and change zero bytes."""

    def test_a_staged_corpus_is_a_zero_request_zero_write_no_op(
        self, tmp_path, server, monkeypatch
    ):
        first = prep.stage_uvdoc(tmp_path, min_free_bytes=0)
        assert first.status == prep.STATUS_STAGED
        assert server["get"] == 1
        before = _snapshot(tmp_path)

        def _boom(*args, **kwargs):
            raise AssertionError("the second run reached the network")

        monkeypatch.setattr(prep, "_http_head", _boom)
        monkeypatch.setattr(prep, "_open_stream", _boom)

        second = prep.stage_uvdoc(tmp_path, min_free_bytes=0)
        assert second.status == prep.STATUS_ALREADY_STAGED
        assert second.members == first.members
        assert server["head"] == 1 and server["get"] == 1
        assert _snapshot(tmp_path) == before, "the no-op modified a file"

    def test_the_landmine_fixture_can_actually_fire(
        self, tmp_path, network_is_a_landmine
    ):
        """The no-op assertion DISCRIMINATES: an unstaged root does fetch."""
        with pytest.raises(AssertionError, match="reached the network"):
            prep.stage_uvdoc(tmp_path, min_free_bytes=0)

    def test_an_archive_without_its_marker_is_not_considered_staged(
        self, tmp_path, server
    ):
        """The ``.ok`` marker is the proof; the ``.zip`` alone is not."""
        prep.stage_uvdoc(tmp_path, min_free_bytes=0)
        prep.marker_path(tmp_path).unlink()
        again = prep.stage_uvdoc(tmp_path, min_free_bytes=0)
        assert again.status == prep.STATUS_STAGED
        assert server["get"] == 2

    def test_a_marker_that_disagrees_with_the_size_is_not_trusted(
        self, tmp_path, server
    ):
        """A file that is not the one that was verified re-stages."""
        prep.stage_uvdoc(tmp_path, min_free_bytes=0)
        marker = prep.marker_path(tmp_path)
        recorded = json.loads(marker.read_text())
        recorded["bytes"] = recorded["bytes"] + 1
        marker.write_text(json.dumps(recorded))
        assert prep._staged_result(tmp_path) is None

    def test_the_marker_records_what_it_verified(self, tmp_path, server):
        prep.stage_uvdoc(tmp_path, min_free_bytes=0)
        marker = json.loads(prep.marker_path(tmp_path).read_text())
        assert marker["bytes"] == prep.archive_path(tmp_path).stat().st_size
        assert marker["zip_members"] == 2
        assert "testzip" in marker["crc_testzip"]

    def test_the_hand_written_marker_schema_is_still_read(self, tmp_path):
        """The pre-step-17 marker on the real volume uses ``zip_members``.

        It carries no ``members`` key at all, so a reader that only knew that
        key would silently report 0 members for the real staged corpus.
        """
        assert prep._marker_members({"zip_members": 182492}) == 182492
        assert prep._marker_members({"members": 7}) == 7
        assert prep._marker_members({}) == 0


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


class TestResume:
    """A partial body is CONTINUED, not thrown away and refetched."""

    def test_a_truncated_part_is_resumed_not_restarted(
        self, tmp_path, server, good_zip
    ):
        half = len(good_zip) // 2
        part = prep.part_path(tmp_path)
        part.parent.mkdir(parents=True, exist_ok=True)
        part.write_bytes(good_zip[:half])

        result = prep.stage_uvdoc(tmp_path, min_free_bytes=0)

        assert result.status == prep.STATUS_STAGED
        assert server["range_headers"] == [f"bytes={half}-"]
        assert result.bytes_downloaded == len(good_zip) - half, (
            "the whole body was refetched; the resume did not take effect"
        )
        assert prep.archive_path(tmp_path).read_bytes() == good_zip
        assert not part.exists()

    def test_the_resume_assertion_is_not_vacuous(
        self, tmp_path, server, good_zip
    ):
        """With no partial present the SAME call fetches the whole body."""
        result = prep.stage_uvdoc(tmp_path, min_free_bytes=0)
        assert server["range_headers"] == [None]
        assert result.bytes_downloaded == len(good_zip)

    def test_a_host_that_ignores_range_restarts_instead_of_concatenating(
        self, tmp_path, server, good_zip
    ):
        """HTTP 200 to a ranged GET means "whole file"; appending would corrupt.

        The archive must come out byte-identical to the source, not the
        partial with a second copy glued onto its end.
        """
        half = len(good_zip) // 2
        part = prep.part_path(tmp_path)
        part.parent.mkdir(parents=True, exist_ok=True)
        part.write_bytes(good_zip[:half])
        server["ranges"] = False

        result = prep.stage_uvdoc(tmp_path, min_free_bytes=0)

        assert result.status == prep.STATUS_STAGED
        assert prep.archive_path(tmp_path).read_bytes() == good_zip
        assert prep.archive_path(tmp_path).stat().st_size == len(good_zip)

    def test_an_interrupted_body_keeps_the_partial_and_stages_nothing(
        self, tmp_path, server, good_zip
    ):
        """A short body is a resumable interruption, not a corruption."""
        cut = len(good_zip) // 3
        server["cut"] = cut

        result = prep.stage_uvdoc(tmp_path, min_free_bytes=0)

        assert result.status == prep.STATUS_INCOMPLETE
        assert prep.part_path(tmp_path).stat().st_size == cut
        assert not prep.archive_path(tmp_path).exists()
        assert not prep.marker_path(tmp_path).exists()
        assert "resume" in result.detail

        server["cut"] = None
        again = prep.stage_uvdoc(tmp_path, min_free_bytes=0)
        assert again.status == prep.STATUS_STAGED
        assert server["range_headers"][-1] == f"bytes={cut}-"
        assert prep.archive_path(tmp_path).read_bytes() == good_zip

    def test_a_failed_get_leaves_the_partial_alone(
        self, tmp_path, server, good_zip
    ):
        """A 500 mid-run must not cost the bytes already on disk."""
        half = len(good_zip) // 2
        part = prep.part_path(tmp_path)
        part.parent.mkdir(parents=True, exist_ok=True)
        part.write_bytes(good_zip[:half])

        @contextmanager
        def _five_hundred(url, *, timeout, headers):
            yield _FakeResponse(500, b"")

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(prep, "_open_stream", _five_hundred)
            result = prep.stage_uvdoc(tmp_path, min_free_bytes=0)

        assert result.status == prep.STATUS_INCOMPLETE
        assert part.stat().st_size == half
        assert not prep.archive_path(tmp_path).exists()


# ---------------------------------------------------------------------------
# The completeness proof
# ---------------------------------------------------------------------------


class TestTheCompletenessProof:
    """Completeness is DECODED, never inferred from a byte count."""

    def test_a_full_length_body_that_fails_crc_is_refused(
        self, tmp_path, server, crc_broken_zip, good_zip
    ):
        """The honest-empty-slot requirement, at its hardest.

        The body is COMPLETE by every length check available -- it is exactly
        as long as the good archive and the host advertised that length -- and
        one member's bytes are wrong. Staging must refuse and leave no ``.zip``
        and no ``.ok``, so that no consumer can ever read this as a corpus.
        """
        assert len(crc_broken_zip) == len(good_zip), (
            "the fixture must be full length, or this tests the size check"
        )
        server["body"] = crc_broken_zip

        result = prep.stage_uvdoc(tmp_path, min_free_bytes=0)

        assert result.status == prep.STATUS_FAILED
        assert not prep.archive_path(tmp_path).exists()
        assert not prep.marker_path(tmp_path).exists()
        assert not prep.part_path(tmp_path).exists()
        assert "CRC" in result.detail

    def test_the_broken_archive_still_lists_every_member(self, crc_broken_zip,
                                                         tmp_path):
        """WHY the CRC pass is necessary and a member COUNT is not enough.

        The corrupted archive's central directory is intact, so ``infolist()``
        reports both members exactly as the good one does. Anything that
        checks structure rather than content calls this file complete.
        """
        path = tmp_path / "broken.zip"
        path.write_bytes(crc_broken_zip)
        with zipfile.ZipFile(path) as archive:
            assert len(archive.infolist()) == 2
            assert archive.testzip() == "uvdoc/img/000.txt"

    def test_the_refusal_is_not_vacuous(self, tmp_path, server):
        """The same path with an intact body stages successfully."""
        result = prep.stage_uvdoc(tmp_path, min_free_bytes=0)
        assert result.status == prep.STATUS_STAGED
        assert result.members == 2
        assert prep.marker_path(tmp_path).exists()

    def test_verify_archive_counts_every_member(self, tmp_path, good_zip):
        path = tmp_path / "good.zip"
        path.write_bytes(good_zip)
        assert prep.verify_archive(path) == 2

    def test_verify_archive_rejects_a_file_that_is_not_a_zip(self, tmp_path):
        path = tmp_path / "bogus.zip"
        path.write_bytes(b"not a zip at all")
        with pytest.raises(prep.ArchiveError):
            prep.verify_archive(path)

    def test_verify_archive_rejects_a_truncated_zip(self, tmp_path, good_zip):
        path = tmp_path / "half.zip"
        path.write_bytes(good_zip[: len(good_zip) // 2])
        with pytest.raises(prep.ArchiveError, match="truncated or corrupt"):
            prep.verify_archive(path)

    def test_verify_archive_rejects_an_empty_zip(self, tmp_path):
        path = tmp_path / "empty.zip"
        with zipfile.ZipFile(path, "w"):
            pass
        with pytest.raises(prep.ArchiveError, match="no members"):
            prep.verify_archive(path)


# ---------------------------------------------------------------------------
# The disk floor
# ---------------------------------------------------------------------------


class TestTheDiskFloor:
    """Refuse rather than fill the volume -- before fetching, not during."""

    def test_the_budget_refuses_and_names_the_shortfall(self, tmp_path):
        state = prep.read_disk_state(tmp_path)
        with pytest.raises(prep.DiskBudgetError) as excinfo:
            prep.check_disk_budget(
                tmp_path,
                needed_bytes=state.free_bytes + 10 * prep._GB,
                min_free_bytes=int(prep.DEFAULT_MIN_FREE_GB * prep._GB),
            )
        message = str(excinfo.value)
        assert "refusing to stage" in message
        assert "shortfall" in message
        assert "nothing was downloaded" in message

    def test_the_budget_passes_when_there_is_room(self, tmp_path):
        """The guard is not vacuous."""
        assert prep.check_disk_budget(tmp_path, 1024, 0).free_bytes > 0

    def test_a_refused_run_fetches_nothing(self, tmp_path, server, monkeypatch):
        monkeypatch.setattr(
            prep,
            "read_disk_state",
            lambda path: prep.DiskState(
                total_bytes=10, used_bytes=9, free_bytes=1
            ),
        )
        result = prep.stage_uvdoc(
            tmp_path, min_free_bytes=int(prep.DEFAULT_MIN_FREE_GB * prep._GB)
        )
        assert result.status == prep.STATUS_REFUSED
        assert server["head"] == 0 and server["get"] == 0
        assert "shortfall" in result.detail
        _assert_nothing_was_staged(tmp_path)

    def test_the_floor_comes_from_the_cli_flag(self, tmp_path, server,
                                               monkeypatch):
        """``--min-free-gb`` really moves the floor, in both directions."""
        monkeypatch.setattr(
            prep,
            "read_disk_state",
            lambda path: prep.DiskState(
                total_bytes=10 ** 15,
                used_bytes=0,
                free_bytes=prep.UVDOC_EXPECTED_BYTES + 5 * prep._GB,
            ),
        )
        assert prep.main(
            ["--root", str(tmp_path), "--min-free-gb", "100"]
        ) == 1
        assert server["get"] == 0
        assert prep.main(["--root", str(tmp_path), "--min-free-gb", "1"]) == 0
        assert server["get"] == 1

    def test_a_dead_url_downloads_nothing_and_names_the_code(
        self, tmp_path, server
    ):
        server["status"] = 503
        result = prep.stage_uvdoc(tmp_path, min_free_bytes=0)
        assert result.status == prep.STATUS_DEAD_URL
        assert "503" in result.detail
        assert server["get"] == 0
        assert not prep.archive_path(tmp_path).exists()


# ---------------------------------------------------------------------------
# The loud failure
# ---------------------------------------------------------------------------


class TestRequireStagedUvdoc:
    """One vocabulary: the trainers' own ``MissingTrainingDataError``."""

    def test_it_is_the_trainer_exception_not_a_second_vocabulary(self):
        assert prep.MissingTrainingDataError is common.MissingTrainingDataError
        assert issubclass(prep.DiskBudgetError, common.DocScannerDataError)
        assert issubclass(prep.ArchiveError, common.DocScannerDataError)

    def test_a_missing_directory_names_the_directory(self, tmp_path):
        with pytest.raises(common.MissingTrainingDataError) as excinfo:
            prep.require_staged_uvdoc(tmp_path)
        message = str(excinfo.value)
        assert str(prep.uvdoc_dir(tmp_path)) in message
        assert "prepare_doc_scanner_data" in message
        assert "--data-source synthetic" in message

    def test_a_partial_download_is_named_as_interrupted(self, tmp_path,
                                                        good_zip):
        part = prep.part_path(tmp_path)
        part.parent.mkdir(parents=True, exist_ok=True)
        part.write_bytes(good_zip[:10])
        with pytest.raises(common.MissingTrainingDataError) as excinfo:
            prep.require_staged_uvdoc(tmp_path)
        message = str(excinfo.value)
        assert "INTERRUPTED" in message
        assert prep.part_path(tmp_path).name in message

    def test_an_unmarked_archive_is_refused_as_never_proven(self, tmp_path,
                                                            server):
        prep.stage_uvdoc(tmp_path, min_free_bytes=0)
        prep.marker_path(tmp_path).unlink()
        with pytest.raises(common.MissingTrainingDataError,
                           match="never proven complete"):
            prep.require_staged_uvdoc(tmp_path)

    def test_a_size_mismatch_is_refused(self, tmp_path, server):
        prep.stage_uvdoc(tmp_path, min_free_bytes=0)
        with open(prep.archive_path(tmp_path), "ab") as handle:
            handle.write(b"x")
        with pytest.raises(common.MissingTrainingDataError,
                           match="not the one that was verified"):
            prep.require_staged_uvdoc(tmp_path)

    def test_a_staged_corpus_returns_its_path(self, tmp_path, server):
        prep.stage_uvdoc(tmp_path, min_free_bytes=0)
        assert prep.require_staged_uvdoc(tmp_path) == prep.archive_path(
            tmp_path
        )


# ---------------------------------------------------------------------------
# The real staged corpus (read-only, skipped when absent)
# ---------------------------------------------------------------------------


_REAL_ARCHIVE = Path(common.DEFAULT_UVDOC_ROOT)
_REAL_MARKER = _REAL_ARCHIVE.with_name(_REAL_ARCHIVE.name + ".ok")


@pytest.mark.skipif(
    not (_REAL_ARCHIVE.exists() and _REAL_MARKER.exists()),
    reason="the real 27.5 GB UVDoc archive is not staged on this machine",
)
class TestTheRealStagedCorpus:
    """READ-ONLY. Never downloads, never writes, never deletes."""

    def test_the_real_corpus_is_a_zero_request_no_op(self,
                                                     network_is_a_landmine):
        before = (_REAL_ARCHIVE.stat().st_size, _REAL_ARCHIVE.stat().st_mtime_ns)
        result = prep.stage_uvdoc(
            prep.DEFAULT_DATASET_ROOT,
            min_free_bytes=int(prep.DEFAULT_MIN_FREE_GB * prep._GB),
        )
        assert result.status == prep.STATUS_ALREADY_STAGED
        assert result.members == prep.UVDOC_EXPECTED_MEMBERS
        assert result.bytes_on_disk == prep.UVDOC_EXPECTED_BYTES
        after = (_REAL_ARCHIVE.stat().st_size, _REAL_ARCHIVE.stat().st_mtime_ns)
        assert after == before, "the no-op modified the real archive"

    def test_the_real_corpus_satisfies_the_trainer_gate(self):
        assert prep.require_staged_uvdoc(prep.DEFAULT_DATASET_ROOT) == (
            _REAL_ARCHIVE
        )
