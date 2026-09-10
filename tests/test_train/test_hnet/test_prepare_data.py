"""Guards for ``train.hnet.prepare_hnet_data``.

Every filesystem write in this module goes through ``tmp_path``. Nothing here
touches the real staging volume ``/media/arxwn/data0_4tb``, nothing here
touches repo-root ``results/`` (the autouse guard in ``tests/conftest.py``
would fail the test if it did), and nothing here deletes anything.

The network is never used. The module has exactly two seams that could reach
it -- ``_snapshot_download`` (the transfer) and ``_dataset_info`` (metadata) --
and the ``network_is_a_landmine`` fixture replaces BOTH with functions that
fail the test if they are ever entered. Several tests use that fixture to
assert "this run performed no download" rather than assume it.
"""

import inspect
import os
from pathlib import Path

import pytest

from train.hnet import prepare_hnet_data as prep


REAL_VOLUME = Path("/media/arxwn/data0_4tb")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def network_is_a_landmine(monkeypatch):
    """Make any network call an immediate, loud test failure."""

    def _boom(*args, **kwargs):
        raise AssertionError(
            "this code path reached the network; it was supposed to be a no-op"
        )

    monkeypatch.setattr(prep, "_snapshot_download", _boom)
    monkeypatch.setattr(prep, "_dataset_info", _boom)


@pytest.fixture
def fake_hub(monkeypatch):
    """Serve a two-file fake snapshot and count the transfers performed."""
    calls = {"download": 0}

    def _download(repo_id, allow_patterns, local_dir):
        calls["download"] += 1
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "000_00000.parquet").write_bytes(b"payload-a")
        (local_dir / "001_00000.parquet").write_bytes(b"payload-b")
        return str(local_dir)

    monkeypatch.setattr(prep, "_snapshot_download", _download)
    monkeypatch.setattr(prep, "_dataset_info", lambda repo_id: pytest.fail(
        "the transfer path called the metadata seam"
    ))
    return calls


@pytest.fixture
def gated() -> prep.Dataset:
    """The single GATED entry of the real manifest."""
    return prep.get_dataset("fineweb_edu_sample_10bt")


def _files_under(root: Path):
    """Sorted repo-root-relative paths of every regular file under ``root``."""
    return sorted(
        str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()
    )


# ---------------------------------------------------------------------------
# CLI contract
# ---------------------------------------------------------------------------


def test_help_prints_a_usage_line_and_allocates_nothing(
    capsys, tmp_path, network_is_a_landmine
):
    """``--help`` must print a ``usage:`` LINE.

    The exit code is deliberately NOT the assertion: a script with no parser
    ignores ``--help``, runs its whole job and exits 0 anyway. Only the usage
    line proves a parser saw the flag before anything else happened.
    """
    with pytest.raises(SystemExit):
        prep.main(["--help", "--dataset-root", str(tmp_path)])
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line.startswith("usage:")]
    assert lines, f"no line starts with 'usage:'; got:\n{out}"
    assert "prepare_hnet_data" in lines[0]
    assert _files_under(tmp_path) == []


def test_help_does_not_create_the_staging_tree(tmp_path, network_is_a_landmine):
    """``--help`` writes nothing at all, not even a slot directory."""
    with pytest.raises(SystemExit):
        prep.main(["--help", "--dataset-root", str(tmp_path)])
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# --dry-run
# ---------------------------------------------------------------------------


def test_dry_run_writes_nothing_outside_the_slot_readme(
    tmp_path, network_is_a_landmine
):
    """The ONLY byte a dry run writes is the GATED slot's own README."""
    code = prep.main(["--dry-run", "--dataset-root", str(tmp_path)])
    assert code == 0
    assert _files_under(tmp_path) == ["fineweb_edu/sample-10BT/README.md"]


def test_a_dry_run_leaves_the_staged_corpus_directory_untouched(
    tmp_path, network_is_a_landmine
):
    """The already-staged Wikipedia slot gets no directory and no file.

    Its tree predates this port; writing our README into it would restructure
    a corpus this script does not own.
    """
    prep.main(["--dry-run", "--dataset-root", str(tmp_path)])
    assert not (tmp_path / "wikipedia").exists()


def test_a_staged_slot_readme_is_refused_by_name(tmp_path):
    """``write_slot_readme`` refuses an already-staged corpus outright."""
    wiki = prep.get_dataset("wikipedia")
    with pytest.raises(ValueError, match="not ours to restructure"):
        prep.write_slot_readme(tmp_path, wiki)
    assert not (tmp_path / "wikipedia").exists()


def test_dry_run_overrides_download_and_fetches_nothing(
    tmp_path, network_is_a_landmine
):
    """``--dry-run --download`` must NOT transfer.

    A dry run that still moves 28.5 GB because a second flag happened to be on
    the command line is not a dry run.
    """
    code = prep.main(
        ["--dry-run", "--download", "--dataset-root", str(tmp_path)]
    )
    assert code == 0
    assert _files_under(tmp_path) == ["fineweb_edu/sample-10BT/README.md"]
    assert not prep.marker_path(tmp_path, prep.get_dataset(
        "fineweb_edu_sample_10bt"
    )).exists()


# ---------------------------------------------------------------------------
# No download without the explicit flag -- asserted STRUCTURALLY
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--dry-run"],
        ["--dry-run", "--download"],
        ["--datasets", "fineweb_edu_sample_10bt"],
        ["--datasets", "wikipedia"],
        ["--dry-run", "--datasets", "fineweb_edu_sample_10bt", "--download"],
        ["--min-free-gb", "0"],
    ],
)
def test_no_argv_without_a_bare_download_flag_can_transfer(
    argv, tmp_path, network_is_a_landmine
):
    """Every command line that is not exactly an opt-in leaves the seam cold.

    The transfer seam is a landmine here, so this is a reachability assertion,
    not a trust-the-default one.
    """
    code = prep.main(argv + ["--dataset-root", str(tmp_path)])
    assert code == 0


def test_the_transfer_seam_is_unreachable_without_allow_download(
    tmp_path, network_is_a_landmine, gated
):
    """``fetch_dataset`` refuses BEFORE it touches the seam, disk or disk check.

    The refusal is checked structurally: the parameter is keyword-only, its
    default is ``False``, and calling with the default raises without creating
    a single path.
    """
    signature = inspect.signature(prep.fetch_dataset)
    parameter = signature.parameters["allow_download"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is False

    with pytest.raises(prep.DownloadNotPermittedError, match="--download"):
        prep.fetch_dataset(tmp_path, gated)
    assert _files_under(tmp_path) == []


def test_stage_dataset_also_defaults_to_no_download(tmp_path, gated):
    """The wrapper does not widen the gate its callee declares."""
    parameter = inspect.signature(prep.stage_dataset).parameters["allow_download"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is False
    parameter = inspect.signature(prep.stage_all).parameters["allow_download"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is False


def test_the_metadata_seam_is_a_different_function_from_the_transfer_seam():
    """A liveness check must not be able to become a transfer.

    They are separate module-level functions with disjoint imports, which is
    what lets the suite landmine one without disabling the other.
    """
    assert prep._dataset_info is not prep._snapshot_download
    # Read the compiled names, not the source text: the docstring legitimately
    # mentions the sibling seam, and a source-substring assertion would fire on
    # the prose rather than on the call.
    metadata_names = set(prep._dataset_info.__code__.co_names)
    transfer_names = set(prep._snapshot_download.__code__.co_names)
    assert "snapshot_download" not in metadata_names
    assert "dataset_info" in metadata_names
    assert "snapshot_download" in transfer_names
    assert "dataset_info" not in transfer_names


def test_check_remote_is_off_by_default_so_a_dry_run_is_offline(tmp_path):
    """``--check-remote`` defaults to False; a plain dry run is provably offline."""
    args = prep.build_parser().parse_args(["--dataset-root", str(tmp_path)])
    assert args.check_remote is False
    assert args.download is False


# ---------------------------------------------------------------------------
# Free-disk check
# ---------------------------------------------------------------------------


def test_the_disk_budget_refuses_and_names_the_shortfall(tmp_path):
    """A write that would cross the floor raises, naming the shortfall."""
    state = prep.read_disk_state(tmp_path)
    with pytest.raises(prep.DiskBudgetError, match="short of"):
        prep.check_disk_budget(
            tmp_path, state.free_bytes, state.free_bytes + 1
        )


def test_the_disk_budget_passes_when_there_is_room(tmp_path):
    """The anti-vacuity twin: the same check succeeds with room to spare."""
    state = prep.check_disk_budget(tmp_path, 1, 0)
    assert state.free_bytes > 0


def test_the_free_disk_check_runs_before_the_slot_is_created(tmp_path):
    """A refused slot leaves NO directory behind.

    This is what pins the check to a position before the ``mkdir``: if the
    check moved after it, the directory would survive the raise.
    """
    state = prep.read_disk_state(tmp_path)
    gated = prep.get_dataset("fineweb_edu_sample_10bt")
    with pytest.raises(prep.DiskBudgetError):
        prep.stage_dataset(
            tmp_path, gated, min_free_bytes=state.free_bytes + 10 ** 9
        )
    assert _files_under(tmp_path) == []
    assert not (tmp_path / "fineweb_edu").exists()


def test_a_refused_fetch_transfers_nothing(tmp_path, gated, monkeypatch):
    """The disk refusal happens before the transfer seam, not after."""
    calls = {"n": 0}

    def _download(*args, **kwargs):
        calls["n"] += 1
        raise AssertionError("downloaded despite a refused disk budget")

    monkeypatch.setattr(prep, "_snapshot_download", _download)
    state = prep.read_disk_state(tmp_path)
    with pytest.raises(prep.DiskBudgetError):
        prep.fetch_dataset(
            tmp_path,
            gated,
            allow_download=True,
            min_free_bytes=state.free_bytes + 10 ** 9,
        )
    assert calls["n"] == 0
    assert _files_under(tmp_path) == []


def test_the_repo_ssd_is_refused_as_a_staging_root(tmp_path):
    """``/media/arxwn/data_fast`` is the repo SSD, never dataset storage."""
    with pytest.raises(prep.StagingRootError, match="data_fast"):
        prep.check_staging_root(Path(prep.FORBIDDEN_ROOT_PREFIX) / "x" / "y")
    assert prep.check_staging_root(tmp_path) == tmp_path.absolute()


# ---------------------------------------------------------------------------
# .part -> os.replace -> .ok-marker-LAST
# ---------------------------------------------------------------------------


def test_a_completed_fetch_writes_the_payload_and_then_the_marker(
    tmp_path, gated, fake_hub
):
    """The happy path: payload at its final name, marker beside the slot."""
    status = prep.fetch_dataset(
        tmp_path, gated, allow_download=True, min_free_bytes=0
    )
    assert status == "fetched"
    slot = prep.dataset_dir(tmp_path, gated)
    assert sorted(p.name for p in slot.iterdir()) == [
        "000_00000.parquet",
        "001_00000.parquet",
    ]
    marker = prep.marker_path(tmp_path, gated)
    assert marker.exists()
    assert "repo_id: HuggingFaceFW/fineweb-edu" in marker.read_text()
    assert not slot.with_name(slot.name + ".part").exists()


def test_an_interrupted_fetch_never_looks_complete(
    tmp_path, gated, fake_hub, monkeypatch
):
    """Crash after the download, before the promotion: NO marker exists.

    The marker is the only completeness signal, and it is written last. If it
    were written before the payload was promoted, this half-finished state
    would be indistinguishable from a complete corpus.
    """
    real_replace = os.replace

    def _explode(src, dst):
        if str(dst).endswith("sample-10BT"):
            raise OSError("simulated interruption during promotion")
        return real_replace(src, dst)

    monkeypatch.setattr(prep.os, "replace", _explode)
    with pytest.raises(OSError, match="simulated interruption"):
        prep.fetch_dataset(
            tmp_path, gated, allow_download=True, min_free_bytes=0
        )
    assert not prep.marker_path(tmp_path, gated).exists()
    assert not prep.is_complete(tmp_path, gated)


def test_a_second_run_over_a_complete_corpus_downloads_nothing(
    tmp_path, gated, fake_hub
):
    """Idempotent short-circuit on the marker, before any network call."""
    prep.fetch_dataset(tmp_path, gated, allow_download=True, min_free_bytes=0)
    assert fake_hub["download"] == 1
    status = prep.fetch_dataset(
        tmp_path, gated, allow_download=True, min_free_bytes=0
    )
    assert status == "present"
    assert fake_hub["download"] == 1


def test_the_marker_lives_outside_the_slot_so_a_gated_slot_holds_one_file(
    tmp_path, network_is_a_landmine, gated
):
    """A GATED slot contains exactly one file and ``du`` equals its size."""
    prep.main(["--dry-run", "--dataset-root", str(tmp_path)])
    slot = prep.dataset_dir(tmp_path, gated)
    files = [p for p in slot.rglob("*") if p.is_file()]
    assert [p.name for p in files] == ["README.md"]
    assert sum(p.stat().st_size for p in files) == files[0].stat().st_size
    assert prep.marker_path(tmp_path, gated).parent == slot.parent


def test_atomic_write_leaves_no_part_file(tmp_path):
    """The ``.part`` temporary never survives a successful write."""
    target = tmp_path / "x.md"
    size = prep.atomic_write_text(target, "hello")
    assert target.read_text() == "hello"
    assert size == 5
    assert not (tmp_path / "x.md.part").exists()


# ---------------------------------------------------------------------------
# The slot README's content
# ---------------------------------------------------------------------------


def test_the_slot_readme_states_the_reason(tmp_path, network_is_a_landmine):
    """Assert on the TEXT, not on the file's existence.

    An empty directory with an empty README is exactly the silent absence this
    slot exists to prevent, and a file-existence assertion cannot see it.
    """
    prep.main(["--dry-run", "--dataset-root", str(tmp_path)])
    gated = prep.get_dataset("fineweb_edu_sample_10bt")
    text = (prep.dataset_dir(tmp_path, gated) / "README.md").read_text()

    assert "**Status: GATED -- this directory is empty by design.**" in text
    assert "USER-GATED, not a technical failure" in text
    assert "D-002" in text
    assert gated.reason in text


def test_the_slot_readme_states_the_licence_budget_and_repo(tmp_path):
    """Provenance the reader needs before spending 28.5 GB of link."""
    gated = prep.get_dataset("fineweb_edu_sample_10bt")
    text = prep.slot_readme_text(tmp_path, gated)
    assert "ODC-BY" in text
    assert "HuggingFaceFW/fineweb-edu" in text
    assert "sample/10BT" in text
    assert "28.50 GB" in text


def test_the_slot_readme_carries_the_exact_rerun_command(tmp_path):
    """The command in the README must be the one that populates the slot."""
    gated = prep.get_dataset("fineweb_edu_sample_10bt")
    text = prep.slot_readme_text(tmp_path, gated)
    command = (
        "python -m train.hnet.prepare_hnet_data "
        f"--dataset-root {tmp_path} --datasets fineweb_edu_sample_10bt "
        "--download"
    )
    assert command in text
    # Anti-vacuity: the command names flags the parser really accepts.
    args = prep.build_parser().parse_args(command.split()[3:])
    assert args.download is True
    assert args.datasets == "fineweb_edu_sample_10bt"


def test_the_slot_readme_records_the_already_staged_wikipedia_corpus(tmp_path):
    """The corpus step 18 actually trains on is named in the slot README."""
    gated = prep.get_dataset("fineweb_edu_sample_10bt")
    text = prep.slot_readme_text(prep.DEFAULT_DATASET_ROOT, gated)
    assert "/media/arxwn/data0_4tb/datasets/wikipedia" in text
    assert "ALREADY STAGED" in text
    assert "41/41 train shards" in text
    assert "dl_techniques.datasets.nlp.load_wikipedia_train_val" in text


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def test_the_manifest_is_one_python_table_with_both_corpora():
    """One ``MANIFEST`` tuple, not a JSON sidecar."""
    assert isinstance(prep.MANIFEST, tuple)
    assert prep.dataset_names() == ("wikipedia", "fineweb_edu_sample_10bt")
    assert {ds.availability for ds in prep.MANIFEST} == {"staged", "gated"}


def test_every_manifest_entry_states_a_reason_and_a_budget():
    """An unexplained or unpriced slot cannot exist."""
    for ds in prep.MANIFEST:
        assert ds.reason.strip()
        assert ds.approx_bytes > 0
        assert ds.source_url.startswith("https://")


def test_a_reasonless_entry_is_rejected_at_construction():
    """Anti-vacuity for the rule above: the validation really fires."""
    with pytest.raises(ValueError, match="must state why"):
        prep.Dataset(
            name="x", subset="s", availability=prep.AVAIL_GATED,
            relative_dir="x", approx_bytes=1, source_url="https://x.invalid",
            repo_id="x/y", licence="n/a", loader="n.a", reason="",
        )
    with pytest.raises(ValueError, match="unknown availability"):
        prep.Dataset(
            name="x", subset="s", availability="whatever",
            relative_dir="x", approx_bytes=1, source_url="https://x.invalid",
            repo_id="x/y", licence="n/a", loader="n.a", reason="r",
        )


def test_an_unknown_dataset_name_is_a_named_error_not_a_silent_skip():
    """A typo in ``--datasets`` must not quietly stage nothing."""
    with pytest.raises(KeyError, match="fineweb_edu_sample_10bt"):
        prep.get_dataset("fineweb")


def test_every_slot_lands_under_the_given_root(tmp_path):
    """No manifest path escapes the root."""
    for ds in prep.MANIFEST:
        target = prep.dataset_dir(tmp_path, ds)
        assert tmp_path in target.parents


def test_no_default_path_points_at_the_repo_ssd():
    """The default root is the 3.6 TB volume, never the repo SSD."""
    assert str(prep.DEFAULT_DATASET_ROOT) == "/media/arxwn/data0_4tb/datasets"
    assert prep.FORBIDDEN_ROOT_PREFIX not in str(prep.DEFAULT_DATASET_ROOT)


def test_the_suite_never_writes_to_the_real_staging_volume(tmp_path):
    """A standing reminder, asserted: tmp_path is not the real volume."""
    assert REAL_VOLUME not in tmp_path.parents
    assert tmp_path != REAL_VOLUME
