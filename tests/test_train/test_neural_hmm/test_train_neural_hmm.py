"""End-to-end tests of ``src/train/neural_hmm/train_neural_hmm.py``.

Every run goes through the real ``main(argv)`` into a pytest temporary directory
(``--output-dir <tmp> --experiment-name <name>``) with tiny settings: 48 train, 16
validation and 24 test sequences of length 4 to 10, 2 epochs. One module-scoped run per
configuration serves every assertion about it:

- ``gaussian``: ``neural-hmm`` (unsupervised, no encoder);
- ``categorical``: ``io-hmm`` (unsupervised, input-dependent transitions, conv encoder);
- ``supervised``: ``neural-crf-tagger`` (discriminative, transformer encoder).

Mechanisms guarded:

- the run directory holds exactly the documented files, and the summary is strict
  JSON whose numbers are finite;
- ``best_model.keras`` reloads, reproduces the in-memory best weights exactly, and is
  the model the summary's scores come from (re-scored here from the archive); a
  separate run whose best epoch is not its last shows that the exact match comes from
  restoring the best weights;
- the summary's reference numbers are the data module's, on the test split of the
  run's own seed; the unsupervised runs report a matched accuracy only, the supervised
  run a direct one and a conditional NLL;
- the epoch log, the history JSON and the epoch count agree; ``config.json`` holds the
  parsed arguments;
- ``--help`` builds nothing; a reused experiment name, a bad argument and an
  ``--encoder`` the preset would ignore are all refused BEFORE anything is written;
- a failing figure does not cost the run its results;
- the archive check counts a NaN facing a number as a difference (``inf``), never as 0.

Nothing may be written under repo-root ``results/``: the autouse fixture in
``tests/conftest.py`` fails any test that adds an entry there.
"""

import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List, NamedTuple

import keras
import numpy as np
import pytest

import train.neural_hmm.train_neural_hmm as train_nhmm
from train.common.run_artifacts import RUN_ARTIFACT_NAMES
from train.neural_hmm.data import (
    majority_rate, true_conditional_nll_per_step, true_nll_per_step,
)

TINY = (
    "--num-train", "48", "--num-val", "16", "--num-test", "24", "--seq-len", "10",
    "--min-seq-len", "4", "--epochs", "2", "--batch-size", "16", "--hidden-dim", "16",
)

RUN_ARGS = {
    "gaussian": ("--variant", "neural-hmm", "--obs-type", "gaussian", "--n-states", "3",
                 "--obs-dim", "2"),
    "categorical": ("--variant", "io-hmm", "--obs-type", "categorical", "--n-states", "3",
                    "--vocab-size", "6"),
    "supervised": ("--variant", "neural-crf-tagger", "--obs-type", "gaussian", "--n-states", "3",
                   "--obs-dim", "2", "--separation", "2.0"),
}
RUNS = sorted(RUN_ARGS)
UNSUPERVISED = ("categorical", "gaussian")

EXPECTED_FILES = {
    "config.json", "run.log", "results_summary.json", "best_model.keras", "training_log.csv",
    "training_history.json", "visualizations/loss_curve.png", "visualizations/transitions.png",
    "visualizations/state_strip.png",
}
SUMMARY_KEYS = {
    "model", "variant", "mode", "obs_type", "n_states", "obs_dim", "vocab_size", "encoder_type",
    "hidden_dim", "params", "seq_len", "min_seq_len", "num_train", "num_val", "num_test",
    "test_valid_steps", "separation", "stay_prob", "seed", "jit_compile", "epochs_requested",
    "epochs_run", "best_checkpoint", "nll_kind", "test_nll_per_step", "true_nll_per_step",
    "nll_gap", "viterbi_acc", "viterbi_acc_matched", "state_mapping", "true_viterbi_acc",
    "majority_rate", "fit_seconds", "visualizations", "total_seconds",
}


class Run(NamedTuple):
    """One finished trainer run."""

    run_dir: Path
    argv: List[str]
    summary: Dict


def _no_constants(name: str):
    raise AssertionError(f"results_summary.json holds the non-JSON constant {name}")


def _strict_summary(run_dir: Path) -> Dict:
    return json.loads((run_dir / "results_summary.json").read_text(), parse_constant=_no_constants)


def _files(run_dir: Path) -> set:
    return {p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*") if p.is_file()}


def _hashes(run_dir: Path) -> Dict[str, str]:
    return {name: hashlib.sha256((run_dir / name).read_bytes()).hexdigest()
            for name in sorted(_files(run_dir))}


def _argv(base: Path, name: str, *extra: str) -> List[str]:
    return [*TINY, *extra, "--output-dir", str(base), "--experiment-name", name]


def _numbers(value):
    """Every number in a nested JSON value."""
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [value]
    if isinstance(value, dict):
        return [n for v in value.values() for n in _numbers(v)]
    if isinstance(value, list):
        return [n for v in value for n in _numbers(v)]
    return []


class _LazyRuns:
    """One real run per name, made on first use and kept for the module.

    A trainer defect that crashes one configuration must not turn the other
    configurations' tests into fixture errors. A failed run is remembered and
    re-raised rather than retried into the directory it already started.
    """

    def __init__(self, base: Path) -> None:
        self.base = base
        self._done: Dict[str, object] = {}

    def __getitem__(self, name: str) -> Run:
        if name not in self._done:
            argv = _argv(self.base, name, *RUN_ARGS[name])
            try:
                assert train_nhmm.main(argv) == 0
                self._done[name] = Run(self.base / name, argv, _strict_summary(self.base / name))
            except Exception as error:  # re-raised below, for every test of this run
                self._done[name] = error
        if isinstance(self._done[name], Exception):
            raise self._done[name]
        return self._done[name]


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> _LazyRuns:
    return _LazyRuns(tmp_path_factory.mktemp("nhmm_runs"))


def _test_split(run: Run):
    return train_nhmm.load_splits(train_nhmm.parse_arguments(run.argv))["test"]


# ---------------------------------------------------------------------
# --help
# ---------------------------------------------------------------------


def test_help_prints_usage_and_builds_nothing(monkeypatch, capsys) -> None:
    touched = []

    def _fail(name):
        def sentinel(*args, **kwargs):
            touched.append(name)
            raise AssertionError(f"--help reached {name}")
        return sentinel

    for name in ("create_neural_hmm", "sample_splits", "setup_gpu", "prepare_run_dir", "set_seeds"):
        monkeypatch.setattr(train_nhmm, name, _fail(name))

    with pytest.raises(SystemExit) as exit_info:
        train_nhmm.main(["--help"])

    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert "usage:" in out and "--stay-prob" in out and "neural-crf-tagger" in out
    assert touched == []


# ---------------------------------------------------------------------
# artifacts and summary
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", RUNS)
def test_the_run_directory_holds_exactly_the_documented_files(runs, name) -> None:
    found = _files(runs[name].run_dir)
    assert found == EXPECTED_FILES, (
        f"unexpected {sorted(found - EXPECTED_FILES)}, missing {sorted(EXPECTED_FILES - found)}")


@pytest.mark.parametrize("name", RUNS)
def test_the_summary_is_strict_json_with_every_key_and_finite_numbers(runs, name) -> None:
    summary = runs[name].summary
    assert set(summary) == SUMMARY_KEYS
    numbers = _numbers(summary)
    assert numbers and all(np.isfinite(n) for n in numbers)
    for key in ("test_nll_per_step", "true_nll_per_step", "viterbi_acc_matched",
                "true_viterbi_acc", "majority_rate"):
        assert isinstance(summary[key], float), key
    assert summary["visualizations"] == {
        "files": ["loss_curve.png", "state_strip.png", "transitions.png"], "failed": []}
    assert summary["epochs_requested"] == 2 and 1 <= summary["epochs_run"] <= 2
    assert summary["test_valid_steps"] == int(_test_split(runs[name]).mask.sum())


@pytest.mark.parametrize("name", RUNS)
def test_the_best_model_reloads_exactly_and_is_the_scored_model(runs, name) -> None:
    run = runs[name]
    best = run.summary["best_checkpoint"]
    assert best["file"] == "best_model.keras"
    assert best["reload_exact"] is True and best["reload_max_abs_diff"] == 0.0
    val_loss = [float(row["val_loss"]) for row in csv.DictReader(open(run.run_dir / "training_log.csv"))]
    assert best["epoch"] == int(np.argmin(val_loss)) + 1
    assert best["val_loss"] == pytest.approx(min(val_loss), rel=1e-6)

    model = keras.models.load_model(run.run_dir / "best_model.keras")
    test = _test_split(run)
    scores = train_nhmm.evaluate(model, test, batch_size=16)
    assert -scores["log_likelihood"].sum() / test.mask.sum() == pytest.approx(
        run.summary["test_nll_per_step"], rel=1e-6)
    assert int(model.count_params()) == run.summary["params"]


def test_the_archive_matches_the_model_when_a_later_epoch_is_worse(tmp_path) -> None:
    """``EarlyStopping(restore_best_weights=True)`` is what makes the reload check exact.

    The module-scoped runs cannot show this: in a 2-epoch run the best epoch is usually
    the last one, and the in-memory model then equals the archive with or without the
    restore. This run is built so that a later epoch is worse. The supervised tagger is
    fit to 8 sequences whose observations carry no state information
    (``--separation 0``), so after a few epochs it fits noise and ``val_loss`` rises.
    ``--patience 0`` then stops it one epoch after the best. Measured: ``val_loss``
    falls to its minimum at epoch 4 and rises about 50% at epoch 5 on both the CPU and
    GPU 1. With ``restore_best_weights=False`` the in-memory model would hold epoch 5
    while ``best_model.keras`` holds epoch 4, and the reload check would fail.
    """
    argv = _argv(tmp_path, "overfit", "--variant", "neural-crf-tagger", "--obs-type", "gaussian",
                 "--n-states", "3", "--obs-dim", "2", "--separation", "0.0", "--num-train", "8",
                 "--learning-rate", "0.03", "--epochs", "8", "--patience", "0")
    assert train_nhmm.main(argv) == 0
    summary = _strict_summary(tmp_path / "overfit")
    best = summary["best_checkpoint"]
    # The precondition: without it the check below passes whatever the restore does.
    assert best["epoch"] < summary["epochs_run"], (
        f"the best epoch ({best['epoch']}) is the last ({summary['epochs_run']}): this run "
        "no longer separates the restored weights from the last epoch's")
    assert best["reload_exact"] is True and best["reload_max_abs_diff"] == 0.0


@pytest.mark.parametrize("name", RUNS)
def test_the_reference_numbers_are_the_data_modules_on_the_runs_test_split(runs, name) -> None:
    run, test = runs[name], _test_split(runs[name])
    summary = run.summary
    if name == "supervised":
        expected = true_conditional_nll_per_step(test.true_hmm, test.obs, test.states, test.mask)
        assert summary["nll_kind"] == "conditional"
    else:
        expected = true_nll_per_step(test.true_hmm, test.obs, test.mask)
        assert summary["nll_kind"] == "marginal"
    assert summary["true_nll_per_step"] == pytest.approx(expected, abs=1e-12)
    assert summary["nll_gap"] == pytest.approx(summary["test_nll_per_step"] - expected, abs=1e-12)
    assert summary["majority_rate"] == pytest.approx(majority_rate(test.states), abs=1e-12)
    assert 0.0 <= summary["true_viterbi_acc"] <= 1.0
    assert 0.0 <= summary["viterbi_acc_matched"] <= 1.0
    assert sorted(summary["state_mapping"]) == list(range(3))


def test_the_true_model_decodes_far_better_than_the_base_rate(runs) -> None:
    """A wrong true-model decoder (for example one fed transposed transitions) would not."""
    summary = runs["gaussian"].summary
    assert summary["true_viterbi_acc"] >= 0.95
    assert summary["true_viterbi_acc"] > summary["majority_rate"] + 0.3


@pytest.mark.parametrize("name", UNSUPERVISED)
def test_an_unsupervised_run_reports_only_the_matched_accuracy(runs, name) -> None:
    summary = runs[name].summary
    assert summary["mode"] == "generative" and summary["viterbi_acc"] is None


def test_the_supervised_run_reports_a_direct_accuracy_too(runs) -> None:
    summary = runs["supervised"].summary
    assert summary["mode"] == "discriminative" and summary["encoder_type"] == "transformer"
    assert isinstance(summary["viterbi_acc"], float)
    assert summary["viterbi_acc_matched"] >= summary["viterbi_acc"]


def test_the_presets_resolve_to_their_documented_structure(runs) -> None:
    assert runs["gaussian"].summary["encoder_type"] is None
    assert runs["categorical"].summary["encoder_type"] == "conv"
    assert runs["categorical"].summary["vocab_size"] == 6 and runs["categorical"].summary["obs_dim"] is None
    assert runs["gaussian"].summary["separation"] == 4.0
    assert runs["categorical"].summary["separation"] is None


@pytest.mark.parametrize("name", RUNS)
def test_the_epoch_log_history_and_epoch_count_agree(runs, name) -> None:
    run = runs[name]
    rows = list(csv.DictReader(open(run.run_dir / "training_log.csv")))
    history = json.loads((run.run_dir / "training_history.json").read_text())
    assert len(rows) == run.summary["epochs_run"]
    assert {"loss", "val_loss"} <= set(history)
    np.testing.assert_allclose([float(r["loss"]) for r in rows], history["loss"], rtol=1e-6, atol=0)
    assert all(np.isfinite(float(r["loss"])) and np.isfinite(float(r["val_loss"])) for r in rows)


@pytest.mark.parametrize("name", RUNS)
def test_config_json_records_the_parsed_arguments(runs, name) -> None:
    run = runs[name]
    config = json.loads((run.run_dir / "config.json").read_text())
    assert config == json.loads(json.dumps(vars(train_nhmm.parse_arguments(run.argv))))


def test_run_log_holds_the_run(runs) -> None:
    text = (runs["gaussian"].run_dir / "run.log").read_text()
    assert "Run directory:" in text and "epoch 1:" in text and "Done: neural-hmm" in text


# ---------------------------------------------------------------------
# refusals happen before any write
# ---------------------------------------------------------------------


def test_a_reused_experiment_name_is_refused_and_the_first_run_is_untouched(runs) -> None:
    first = runs["gaussian"]
    before = _hashes(first.run_dir)
    mtimes = {name: (first.run_dir / name).stat().st_mtime_ns for name in before}

    with pytest.raises(FileExistsError, match="Nothing was written"):
        train_nhmm.main(first.argv)

    assert _hashes(first.run_dir) == before
    assert {name: (first.run_dir / name).stat().st_mtime_ns for name in before} == mtimes


@pytest.mark.parametrize("artifact", RUN_ARTIFACT_NAMES)
def test_the_refusal_comes_before_any_write(tmp_path, artifact) -> None:
    """A directory holding ONE file of an earlier run: refused, and nothing added.

    The earlier run is fabricated so that the check does not depend on a first run
    succeeding: a refusal placed after ``prepare_run_dir`` would already have written
    (or rewritten) ``config.json`` by the time it raises.
    """
    run_dir = tmp_path / "earlier"
    run_dir.mkdir()
    (run_dir / artifact).write_bytes(b"an earlier run's bytes\n")
    before = _hashes(run_dir)

    raised = None
    try:
        train_nhmm.main(_argv(tmp_path, "earlier"))
    except FileExistsError as error:
        raised = error

    after = _hashes(run_dir)
    assert after == before, (
        f"the trainer wrote into a directory that already held {artifact}: "
        f"before {sorted(before)}, after {sorted(after)}")
    assert raised is not None, f"a directory holding {artifact} was not refused"
    assert artifact in str(raised)


@pytest.mark.parametrize("bad", [
    ("--min-seq-len", "11"),
    ("--stay-prob", "1.0"),
    ("--obs-type", "categorical", "--vocab-size", "2"),
    ("--separation", "-1"),
    ("--learning-rate", "0"),
    ("--num-test", "0"),
])
def test_a_bad_argument_is_refused_at_parse_time_before_anything_is_written(tmp_path, bad) -> None:
    with pytest.raises(SystemExit) as exit_info:
        train_nhmm.main(_argv(tmp_path, "bad", *bad))
    assert exit_info.value.code == 2
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("bad,match", [
    (("--variant", "neural-hmm", "--encoder", "lstm"), "would be ignored"),
    (("--variant", "neural-crf-tagger", "--hidden-dim", "18"), "divisible by num_heads"),
])
def test_a_model_refusal_comes_before_anything_is_written(tmp_path, bad, match) -> None:
    """``--encoder`` on a preset without one, and a width the attention heads cannot split."""
    with pytest.raises(ValueError, match=match):
        train_nhmm.main(_argv(tmp_path, "refused", *bad))
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------
# the archive check cannot be fooled by a NaN
# ---------------------------------------------------------------------


class _Split(NamedTuple):
    obs: np.ndarray
    mask: np.ndarray


def _fixed_output(**outputs):
    """A stand-in model returning the same output dict for every batch."""
    return lambda inputs, training=False: {k: np.asarray(v, dtype=np.float32) for k, v in outputs.items()}


@pytest.mark.parametrize("a,b,expected", [
    ([0.5, np.nan], [0.5, 0.25], np.inf),        # NaN facing a number
    ([np.nan, 1.0], [np.nan, 1.0], 0.0),         # NaN in both at one place: equal
    ([np.inf, 1.0], [np.inf, 1.0], 0.0),         # equal infinities
    ([np.inf, 1.0], [-np.inf, 1.0], np.inf),     # opposite infinities
    ([1.0, 2.0], [1.0, 2.5], 0.5),
])
def test_the_reload_difference_counts_a_nan_disagreement_as_inexact(a, b, expected) -> None:
    split = _Split(obs=np.zeros((3, 2, 1), np.float32), mask=np.ones((3, 2), np.float32))
    for left, right in ((a, b), (b, a)):
        diff = train_nhmm._max_abs_output_diff(
            _fixed_output(log_emit=left, log_pi=[0.0]), _fixed_output(log_emit=right, log_pi=[0.0]),
            split, batch_size=2)
        assert diff == expected, (left, right, diff)


# ---------------------------------------------------------------------
# figures are fail-soft
# ---------------------------------------------------------------------


class _InjectedFigureError(RuntimeError):
    pass


def test_a_failing_figure_does_not_abort_the_run(monkeypatch, tmp_path) -> None:
    def _raise(*args, **kwargs):
        raise _InjectedFigureError("injected into render_transitions")

    monkeypatch.setattr(train_nhmm, "render_transitions", _raise)
    assert train_nhmm.main(_argv(tmp_path, "figfail", *RUN_ARGS["gaussian"])) == 0
    summary = _strict_summary(tmp_path / "figfail")
    assert summary["visualizations"] == {
        "files": ["loss_curve.png", "state_strip.png"], "failed": ["transitions"]}
    assert np.isfinite(summary["test_nll_per_step"])
