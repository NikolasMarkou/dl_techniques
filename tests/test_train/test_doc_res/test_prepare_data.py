"""Guards for ``train.doc_res.prepare_doc_res_data``.

Every test writes only under ``tmp_path``. Nothing here touches the real
staging volume, and nothing here touches repo-root ``results/`` (the autouse
guard in ``tests/conftest.py`` would fail the test if it did).

The network is never used: the two seams that reach it -- ``_http_head`` and
``_download`` -- are monkeypatched, and several tests monkeypatch them to
*raise* so that "this run performed no download" is asserted rather than
assumed.
"""

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from train.doc_res import prepare_doc_res_data as prep


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_zip(tmp_path: Path) -> Path:
    """A small, valid zip holding one PNG-shaped payload and one text file."""
    source = tmp_path / "source.zip"
    with zipfile.ZipFile(source, "w") as zf:
        zf.writestr("pages/A01.txt", "hello")
        zf.writestr("pages/GT/A01.txt", "gt")
    return source


@pytest.fixture
def fake_dataset() -> prep.Dataset:
    """A one-archive fetchable dataset that never touches the real manifest."""
    return prep.Dataset(
        name="fixture_set",
        task="binarization",
        availability=prep.AVAIL_HTTP,
        role=prep.ROLE_TRAIN,
        provenance="test fixture",
        licence="n/a",
        archives=(
            prep.Archive(
                url="https://example.invalid/fixture.zip",
                filename="fixture.zip",
                approx_bytes=1024,
            ),
        ),
    )


@pytest.fixture
def serve(monkeypatch, fake_zip: Path):
    """Serve ``fake_zip`` for any URL and count the downloads performed."""
    calls = {"head": 0, "get": 0}

    def _head(url, timeout):
        calls["head"] += 1
        return 200, fake_zip.stat().st_size, ""

    def _download(url, dest_part, timeout):
        calls["get"] += 1
        dest_part.parent.mkdir(parents=True, exist_ok=True)
        dest_part.write_bytes(fake_zip.read_bytes())
        return dest_part.stat().st_size

    monkeypatch.setattr(prep, "_http_head", _head)
    monkeypatch.setattr(prep, "_download", _download)
    return calls


@pytest.fixture
def network_is_a_landmine(monkeypatch):
    """Make any network call an immediate, loud test failure."""

    def _boom(*args, **kwargs):
        raise AssertionError(
            "this code path reached the network; it was supposed to be a no-op"
        )

    monkeypatch.setattr(prep, "_http_head", _boom)
    monkeypatch.setattr(prep, "_download", _boom)


# ---------------------------------------------------------------------------
# CLI contract
# ---------------------------------------------------------------------------


def test_help_prints_a_usage_line_and_allocates_nothing(
    capsys, tmp_path, network_is_a_landmine
):
    """``--help`` must print a ``usage:`` LINE.

    The exit code is deliberately NOT the assertion: a script that ignores
    ``--help``, runs its whole job and falls off the end also exits 0. Only the
    usage line proves a parser saw the flag before anything else happened.
    """
    with pytest.raises(SystemExit):
        prep.main(["--help"])
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line.startswith("usage:")]
    assert lines, f"no line starts with 'usage:'; got:\n{out}"
    assert "prepare_doc_res_data" in lines[0]


def test_help_does_not_create_the_staging_tree(tmp_path, network_is_a_landmine):
    """``--help`` writes nothing at all."""
    with pytest.raises(SystemExit):
        prep.main(["--help", "--root", str(tmp_path)])
    assert list(tmp_path.iterdir()) == []


def test_dry_run_writes_nothing(monkeypatch, tmp_path):
    """``--dry-run`` reports and returns without creating a single file."""
    monkeypatch.setattr(
        prep,
        "verify_url",
        lambda archive, timeout: prep.UrlStatus(
            url=archive.url,
            filename=archive.filename,
            ok=True,
            status_code=200,
            content_length=archive.approx_bytes,
        ),
    )
    monkeypatch.setattr(prep, "_download", lambda *a, **k: pytest.fail(
        "--dry-run downloaded something"
    ))
    code = prep.main(["--dry-run", "--root", str(tmp_path)])
    assert code == 0
    assert list(tmp_path.rglob("*")) == []


def test_the_dry_run_budget_is_the_sum_of_the_live_urls(monkeypatch, tmp_path):
    """The budget counts live URLs only; a dead one contributes nothing."""
    ds = prep.get_dataset("dibco2009")

    def _verify(archive, timeout):
        live = archive.filename.endswith("printed.rar")
        return prep.UrlStatus(
            url=archive.url,
            filename=archive.filename,
            ok=live,
            status_code=200 if live else 404,
            content_length=100 if live else None,
        )

    monkeypatch.setattr(prep, "verify_url", _verify)
    plans = prep.build_plan(tmp_path, [ds])
    assert plans[0].budget_bytes == 200


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", prep.dataset_names())
def test_every_dataset_lands_under_root_doc_res_task_dataset(name, tmp_path):
    """The on-disk layout is ``<root>/doc_res/<task>/<dataset>/``, always."""
    ds = prep.get_dataset(name)
    path = prep.dataset_dir(tmp_path, ds)
    assert path == tmp_path / "doc_res" / ds.task / ds.name
    assert path.relative_to(tmp_path).parts == ("doc_res", ds.task, ds.name)


def test_staging_never_escapes_the_root(tmp_path):
    """No manifest entry can place data outside ``<root>/doc_res``."""
    root = prep.staging_root(tmp_path).resolve()
    for name in prep.dataset_names():
        target = prep.dataset_dir(tmp_path, prep.get_dataset(name)).resolve()
        assert root in target.parents


def test_the_manifest_is_internally_consistent():
    """Every slot is either fetchable with archives, or explained without."""
    for ds in prep.MANIFEST:
        assert ds.task in prep.TASKS
        if ds.is_fetchable:
            assert ds.archives and not ds.reason
        else:
            assert ds.reason and not ds.archives


def test_the_manifest_carries_eleven_dibco_years():
    """DIBCO is assembled year-by-year; there is no pooled source."""
    years = [d for d in prep.MANIFEST if d.name.endswith(tuple("0123456789"))
             and "dibco" in d.name]
    assert len(years) == 11, [d.name for d in years]
    assert len({a.url for d in years for a in d.archives}) == sum(
        len(d.archives) for d in years
    )


# ---------------------------------------------------------------------------
# Disk budget
# ---------------------------------------------------------------------------


def test_the_disk_budget_refuses_and_names_the_shortfall(tmp_path):
    """A budget the volume cannot meet raises, naming the shortfall."""
    state = prep.read_disk_state(tmp_path)
    with pytest.raises(prep.DiskBudgetError) as excinfo:
        prep.check_disk_budget(
            tmp_path,
            needed_bytes=state.free_bytes + 10 * prep._GB,
            min_free_bytes=prep.DEFAULT_MIN_FREE_GB * prep._GB,
        )
    message = str(excinfo.value)
    assert "shortfall" in message
    assert "refusing to stage" in message
    assert "nothing was downloaded" in message


def test_the_disk_budget_passes_when_there_is_room(tmp_path):
    """The guard is not vacuous: a tiny request goes through."""
    state = prep.check_disk_budget(tmp_path, 1024, 0)
    assert state.free_bytes > 0


def test_a_refused_dataset_downloads_nothing(
    monkeypatch, tmp_path, fake_dataset, serve
):
    """When the budget refuses, ``stage_all`` reports it and fetches nothing."""
    monkeypatch.setattr(
        prep,
        "read_disk_state",
        lambda path: prep.DiskState(total_bytes=10, used_bytes=9, free_bytes=1),
    )
    reports = prep.stage_all(
        tmp_path,
        [fake_dataset],
        min_free_bytes=prep.DEFAULT_MIN_FREE_GB * prep._GB,
        do_sidecars=False,
    )
    assert [r.status for r in reports] == ["refused"]
    assert serve["get"] == 0
    assert serve["head"] == 0
    assert "shortfall" in reports[0].detail


# ---------------------------------------------------------------------------
# Dead links
# ---------------------------------------------------------------------------


def test_a_dead_url_is_reported_by_name_not_skipped(
    monkeypatch, tmp_path, fake_dataset
):
    """A 404 yields a named ``dead-url`` result; nothing is silently dropped."""
    monkeypatch.setattr(prep, "_http_head", lambda url, timeout: (404, None, ""))
    monkeypatch.setattr(prep, "_download", lambda *a, **k: pytest.fail(
        "a dead URL was downloaded anyway"
    ))
    reports = prep.stage_all(
        tmp_path, [fake_dataset], min_free_bytes=0, do_sidecars=False
    )
    dead = reports[0].dead_urls
    assert len(dead) == 1
    assert dead[0].filename == "fixture.zip"
    assert dead[0].url == "https://example.invalid/fixture.zip"
    assert "404" in dead[0].detail


def test_a_dead_url_does_not_abort_the_other_archives(monkeypatch, tmp_path,
                                                      fake_zip):
    """One dead university page must not cost the other years."""
    ds = prep.Dataset(
        name="two_files",
        task="binarization",
        availability=prep.AVAIL_HTTP,
        role=prep.ROLE_TRAIN,
        provenance="fixture",
        licence="n/a",
        archives=(
            prep.Archive("https://example.invalid/dead.zip", "dead.zip", 10),
            prep.Archive("https://example.invalid/live.zip", "live.zip", 10),
        ),
    )
    monkeypatch.setattr(
        prep,
        "_http_head",
        lambda url, timeout: (404, None, "") if "dead" in url else (200, 10, ""),
    )

    def _download(url, dest_part, timeout):
        dest_part.parent.mkdir(parents=True, exist_ok=True)
        dest_part.write_bytes(fake_zip.read_bytes())
        return dest_part.stat().st_size

    monkeypatch.setattr(prep, "_download", _download)
    reports = prep.stage_all(
        tmp_path, [ds], min_free_bytes=0, do_sidecars=False
    )
    by_name = {a.filename: a.status for a in reports[0].archives}
    assert by_name == {"dead.zip": "dead-url", "live.zip": "staged"}


def test_the_dead_url_summary_prints_the_filename(caplog, monkeypatch,
                                                  tmp_path, fake_dataset):
    """The end-of-run summary names the dead file, not just a count."""
    monkeypatch.setattr(prep, "_http_head", lambda url, timeout: (404, None, ""))
    reports = prep.stage_all(
        tmp_path, [fake_dataset], min_free_bytes=0, do_sidecars=False
    )
    with caplog.at_level("WARNING", logger="dl"):
        prep.log_reports(reports, tmp_path)
    assert "fixture.zip" in caplog.text


# ---------------------------------------------------------------------------
# Idempotency and resumability
# ---------------------------------------------------------------------------


def test_a_second_run_over_a_staged_fixture_downloads_nothing(
    monkeypatch, tmp_path, fake_dataset, serve
):
    """Idempotency: the second run makes zero HEAD and zero GET calls."""
    first = prep.stage_all(
        tmp_path, [fake_dataset], min_free_bytes=0, do_sidecars=False
    )
    assert [a.status for a in first[0].archives] == ["staged"]
    assert serve["get"] == 1

    monkeypatch.setattr(prep, "_http_head", lambda *a, **k: pytest.fail(
        "second run verified a URL it had already staged"
    ))
    monkeypatch.setattr(prep, "_download", lambda *a, **k: pytest.fail(
        "second run re-downloaded an already-staged archive"
    ))
    second = prep.stage_all(
        tmp_path, [fake_dataset], min_free_bytes=0, do_sidecars=False
    )
    assert [a.status for a in second[0].archives] == ["already-staged"]


def test_an_interrupted_download_never_looks_complete(
    monkeypatch, tmp_path, fake_dataset, fake_zip, serve
):
    """A truncated body is rejected, leaves no final file and no marker.

    This is the property the ``.part`` + full-member-decode + atomic-rename
    design exists for (D-024): a size check could not catch it, because the
    hosts that matter publish no ``Content-Length``.
    """
    truncated = fake_zip.read_bytes()[: len(fake_zip.read_bytes()) // 2]

    def _truncated(url, dest_part, timeout):
        dest_part.parent.mkdir(parents=True, exist_ok=True)
        dest_part.write_bytes(truncated)
        return len(truncated)

    monkeypatch.setattr(prep, "_download", _truncated)
    target = prep.dataset_dir(tmp_path, fake_dataset)
    archives_dir = target / prep.ARCHIVE_DIRNAME
    result = prep.stage_archive(
        fake_dataset.archives[0], archives_dir, target
    )
    assert result.status == "failed"
    assert not (archives_dir / "fixture.zip").exists()
    assert not (archives_dir / "fixture.zip.ok").exists()
    assert not (archives_dir / "fixture.zip.part").exists()

    # And the very next run, with an intact body, succeeds.
    monkeypatch.setattr(
        prep,
        "_download",
        lambda url, dest, timeout: (
            dest.parent.mkdir(parents=True, exist_ok=True),
            dest.write_bytes(fake_zip.read_bytes()),
            dest.stat().st_size,
        )[-1],
    )
    again = prep.stage_archive(fake_dataset.archives[0], archives_dir, target)
    assert again.status == "staged"
    assert (archives_dir / "fixture.zip.ok").exists()


def test_the_completion_marker_records_what_it_verified(
    tmp_path, fake_dataset, serve
):
    """The ``.ok`` marker carries the digest, size and member count."""
    target = prep.dataset_dir(tmp_path, fake_dataset)
    prep.stage_archive(
        fake_dataset.archives[0], target / prep.ARCHIVE_DIRNAME, target
    )
    marker = json.loads(
        (target / prep.ARCHIVE_DIRNAME / "fixture.zip.ok").read_text()
    )
    assert marker["members"] == 2
    assert len(marker["sha256"]) == 64
    assert marker["bytes"] > 0


def test_extraction_places_the_members_under_the_dataset_dir(
    tmp_path, fake_dataset, serve
):
    """The archive's contents land in the dataset directory, not in _archives."""
    target = prep.dataset_dir(tmp_path, fake_dataset)
    prep.stage_archive(
        fake_dataset.archives[0], target / prep.ARCHIVE_DIRNAME, target
    )
    assert (target / "pages" / "A01.txt").read_text() == "hello"


# ---------------------------------------------------------------------------
# Archive safety
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "member",
    ["/etc/passwd", "../escape.txt", "a/../../escape.txt", ".."],
)
def test_a_traversing_member_is_rejected(tmp_path, member):
    """No archive member may resolve outside the destination."""
    assert prep._safe_member_path(tmp_path, member) is None


@pytest.mark.parametrize("member", ["a.txt", "d/a.txt", "d/e/f.tiff"])
def test_an_ordinary_member_is_accepted(tmp_path, member):
    """The traversal guard is not vacuous."""
    resolved = prep._safe_member_path(tmp_path, member)
    assert resolved is not None
    assert tmp_path.resolve() in resolved.parents


def test_verify_archive_rejects_a_corrupt_file(tmp_path):
    """A file that is not an archive at all is an ArchiveError, not a crash."""
    bogus = tmp_path / "bogus.zip"
    bogus.write_bytes(b"not a zip at all")
    with pytest.raises(prep.ArchiveError):
        prep.verify_archive(bogus)


# ---------------------------------------------------------------------------
# The empty slots
# ---------------------------------------------------------------------------


def test_dewarping_training_fails_loudly_and_names_the_doc3d_gate(tmp_path):
    """Doc3D's absence is reported as a registration gate, not "no files"."""
    with pytest.raises(prep.MissingTrainingDataError) as excinfo:
        prep.require_task_data(tmp_path, "dewarping")
    message = str(excinfo.value)
    assert "doc3d" in message
    assert "registration-gated" in message
    assert "549 GB" in message
    assert "EVAL ONLY" in message and "dir300" in message


def test_deblurring_fails_loudly_and_says_the_host_is_dead(tmp_path):
    """TDD's absence must say DEAD HOST, never "file not found"."""
    with pytest.raises(prep.MissingTrainingDataError) as excinfo:
        prep.require_task_data(tmp_path, "deblurring")
    message = str(excinfo.value)
    assert "tdd" in message
    assert "DEAD" in message
    assert "fit.vutbr.cz" in message and "fit.vut.cz" in message
    assert "not found" not in message.lower()


def test_binarization_reports_unstaged_rather_than_unavailable(tmp_path):
    """A fetchable-but-unstaged task says so, and names the staging command."""
    with pytest.raises(prep.MissingTrainingDataError) as excinfo:
        prep.require_task_data(tmp_path, "binarization")
    message = str(excinfo.value)
    assert "prepare_doc_res_data" in message
    assert "dead" not in message.lower()


def test_require_task_data_returns_the_staged_datasets(tmp_path):
    """The guard is not vacuous: one staged page satisfies it."""
    ds = prep.get_dataset("dibco2009")
    target = prep.dataset_dir(tmp_path, ds)
    target.mkdir(parents=True)
    _write_png(target / "P01.png")
    assert [d.name for d in prep.require_task_data(tmp_path, "binarization")] == [
        "dibco2009"
    ]


def test_a_slot_readme_states_the_reason(tmp_path):
    """Every non-fetchable slot gets a directory and a README explaining it."""
    for name in ("doc3d", "tdd", "rdd", "dir300"):
        ds = prep.get_dataset(name)
        readme = prep.write_slot_readme(tmp_path, ds)
        assert readme == prep.dataset_dir(tmp_path, ds) / "README.md"
        text = readme.read_text()
        assert ds.availability.upper() in text
        assert "empty by design" in text
        assert ds.reason.split(":")[0][:30] in text


def test_the_drive_slots_are_never_partially_downloaded(tmp_path,
                                                        network_is_a_landmine):
    """RDD/DIR300 must refuse, not scrape 51 of 300 files (D-025)."""
    drive = [d for d in prep.MANIFEST if d.availability == prep.AVAIL_MANUAL_DRIVE]
    assert {d.name for d in drive} == {"rdd", "dir300"}
    reports = prep.stage_all(
        tmp_path, drive, min_free_bytes=0, do_sidecars=False
    )
    assert [r.status for r in reports] == ["slot", "slot"]
    for report in reports:
        assert "drive.google.com" in prep.get_dataset(report.dataset).manual_url


# ---------------------------------------------------------------------------
# Prompt sidecars
# ---------------------------------------------------------------------------


def _write_png(path: Path, size=(24, 32)) -> None:
    """Write a small deterministic RGB PNG."""
    from PIL import Image

    rng = np.random.default_rng(0)
    array = rng.integers(0, 256, size=(size[0], size[1], 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


@pytest.mark.parametrize(
    "relative,expected",
    [
        ("DIBCO2009-GT-Test-images_printed/P01.tiff", True),
        ("H_DIBCO2010_GT/P01.tiff", True),
        ("dataset/GT/P01.png", True),
        ("pages/P01_GT.png", True),
        ("pages/ground_truth/P01.png", True),
        ("H08_skelGT.tiff", True),
        ("H06_estGT.tiff", True),
        ("GTimages/HW01_estGT.tiff", True),
        ("DIPCO2016_Dataset_GT/7_gt.bmp", True),
        ("NoisyOffice/SimulatedNoisyOffice/clean_images_grayscale/a.png", True),
        ("DIBCO2009_Test_images-printed/P01.tiff", False),
        ("original_images/P01.png", False),
        ("gtx_pages/P01.png", False),
        ("DIGTAL/P01.png", False),
        ("simulated_noisy_images_grayscale/a.png", False),
    ],
)
def test_ground_truth_detection(relative, expected):
    """Input pages get sidecars; ground-truth pages do not."""
    assert prep.is_ground_truth_path(relative) is expected


def test_sidecars_are_written_once_for_input_pages_only(tmp_path):
    """Sidecars mirror the input tree and skip ground truth."""
    ds = prep.get_dataset("dibco2009")
    target = prep.dataset_dir(tmp_path, ds)
    _write_png(target / "images" / "P01.png")
    _write_png(target / "GT" / "P01.png")
    _write_png(target / prep.ARCHIVE_DIRNAME / "ignored.png")
    _write_png(target / "images" / ".xvpics" / "thumb.png")

    result = prep.precompute_sidecars(tmp_path, ds)
    assert result.status == "ok"
    assert (result.written, result.already_present, result.failed) == (1, 0, 0)
    sidecar = target / prep.PROMPT_DIRNAME / "images" / "P01.png"
    assert sidecar.exists()
    assert not (target / prep.PROMPT_DIRNAME / "GT" / "P01.png").exists()

    from PIL import Image

    with Image.open(sidecar) as handle:
        prompt = np.asarray(handle)
    assert prompt.shape == (24, 32, 3)
    assert prompt.dtype == np.uint8


def test_a_second_sidecar_pass_recomputes_nothing(tmp_path):
    """The precompute is idempotent, so re-running staging is cheap."""
    ds = prep.get_dataset("dibco2009")
    target = prep.dataset_dir(tmp_path, ds)
    _write_png(target / "images" / "P01.png")
    prep.precompute_sidecars(tmp_path, ds)
    again = prep.precompute_sidecars(tmp_path, ds)
    assert (again.written, again.already_present) == (0, 1)


def test_dewarping_sidecars_are_skipped_with_the_mask_reason(tmp_path):
    """The dewarping prompt needs a mask this port does not produce."""
    ds = prep.get_dataset("dir300")
    target = prep.dataset_dir(tmp_path, ds)
    _write_png(target / "dist" / "1.png")
    result = prep.precompute_sidecars(tmp_path, ds)
    assert result.status == "skipped"
    assert "mask" in result.detail and "MBD" in result.detail


def test_sidecars_report_no_images_for_an_empty_slot(tmp_path):
    """An empty slot reports ``no-images`` rather than pretending to work."""
    result = prep.precompute_sidecars(tmp_path, prep.get_dataset("tdd"))
    assert result.status == "no-images"


# ---------------------------------------------------------------------------
# Extraction backends
# ---------------------------------------------------------------------------


def test_zip_extraction_needs_no_optional_backend(tmp_path, fake_zip):
    """.zip is stdlib; it must work even with libarchive absent."""
    out = tmp_path / "out"
    written = prep.extract_archive(fake_zip, out)
    assert sorted(p.name for p in written) == ["A01.txt", "A01.txt"]
    assert (out / "pages" / "A01.txt").exists()


def test_a_missing_backend_is_a_named_error_not_a_crash(monkeypatch, tmp_path):
    """Without libarchive, a .rar reports why -- and keeps the bytes."""
    monkeypatch.setattr(prep, "_libarchive", lambda: None)
    rar = tmp_path / "x.rar"
    rar.write_bytes(b"Rar!\x1a\x07\x00")
    with pytest.raises(prep.NoExtractorError) as excinfo:
        prep.verify_archive(rar)
    assert "libarchive-c" in str(excinfo.value)


def test_extractor_backends_reports_the_stdlib_path_always():
    """zipfile is always available; libarchive is optional."""
    backends = prep.extractor_backends()
    assert backends["zipfile"] is True
    assert isinstance(backends["libarchive"], bool)
