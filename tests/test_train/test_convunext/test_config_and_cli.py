"""Config, CLI and refusal guards for ``src/train/convunext/`` (the segmentation trainer).

What is pinned here, and the defect each guard exists to catch
--------------------------------------------------------------
- A flag that ``--help`` advertises and nothing forwards: one row per declared flag on
  the shared ``tests/test_train/_cli_contract.py`` driver, completeness by set equality
  with the REAL parser, probes that differ from the default and from each other.
- A parser default that drifted from the config default: the parser reads its defaults
  off ``SegTrainingConfig`` and a test compares them one field at a time.
- An image size the model cannot run. MEASURED (CPU, this file's probe): ``create_convunext``
  BUILDS at any size, even 1 px, and only the forward pass fails, so a "does it build"
  probe would accept every size. The smallest size that runs is ``2 ** depth``; the config
  refuses below it and ``test_min_image_size_is_the_measured_forward_boundary`` re-measures
  it against the real model at the boundary and one pixel below.
- A reused experiment name that is merged instead of refused, or refused only after
  something was written, or after the GPU was configured.
- ``--help`` that allocates a GPU or a directory.

Every test writes only under ``tmp_path``. Nothing here trains, loads a dataset or reads
or writes the repo-root ``results/`` (one test COMPUTES the path a default run would use
and asserts that nothing was created there).
"""

from __future__ import annotations

import subprocess
import sys
import uuid
from dataclasses import fields
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import tensorflow as tf

import train.convunext.common as common
from dl_techniques.models.vision.convunext.model import CONVUNEXT_CONFIGS, create_convunext

from .._cli_contract import (  # noqa: TID252 -- shared driver, one level up
    Contract,
    Mode,
    Row,
    assert_row_reaches_config,
    assert_row_value_is_not_the_default,
    cases,
    declared_option_strings,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------
# CLI contract: every declared flag reaches its config field
# ---------------------------------------------------------------------

def _rows() -> Tuple[Row, ...]:
    """One row per declared flag. Each probe differs from the config default (tiny / 128 /
    0.1 / None / 30 / 16 / 1e-3 / 1e-4 / 0 / 10 / 42 / 1 / 4 / results) and from every
    other row's probe."""
    return (
        Row(("--variant",), ("--variant", "small"), "variant", "small"),
        Row(("--image-size",), ("--image-size", "96"), "image_size", 96),
        Row(("--validation-split",), ("--validation-split", "0.25"), "validation_split", 0.25),
        Row(("--max-samples",), ("--max-samples", "321"), "max_samples", 321),
        Row(("--epochs",), ("--epochs", "7"), "epochs", 7),
        Row(("--batch-size",), ("--batch-size", "12"), "batch_size", 12),
        Row(("--learning-rate",), ("--learning-rate", "1.5e-3"), "learning_rate", 1.5e-3),
        Row(("--weight-decay",), ("--weight-decay", "0.02"), "weight_decay", 0.02),
        Row(("--warmup-epochs",), ("--warmup-epochs", "3"), "warmup_epochs", 3),
        Row(("--patience",), ("--patience", "8"), "patience", 8),
        Row(("--seed",), ("--seed", "1234"), "seed", 1234),
        Row(("--viz-freq",), ("--viz-freq", "5"), "viz_freq", 5),
        Row(("--viz-samples",), ("--viz-samples", "6"), "viz_samples", 6),
        # BooleanOptionalAction, default True: the probe is the non-default spelling.
        Row(("--model-analysis", "--no-model-analysis"), ("--no-model-analysis",),
            "model_analysis", False),
        Row(("--output-dir",), ("--output-dir", "/probe/seg-out"), "output_dir", "/probe/seg-out"),
        Row(("--experiment-name",), ("--experiment-name", "probe-experiment-name"),
            "experiment_name", "probe-experiment-name"),
        # Not a config field: consumed once by setup_gpu in main(); the hop into
        # setup_gpu is test_gpu_flag_reaches_setup_gpu.
        Row(("--gpu",), ("--gpu", "1"), namespace_dest="gpu", expected=1),
    )


CONTRACT = Contract(
    name="train.convunext.common",
    build_parser=lambda monkeypatch: common._build_parser(),
    build_config=lambda monkeypatch: common.config_from_args(common.parse_arguments(None)),
    modes=(Mode(id="plain", required_argv=(), rows=_rows()),),
)
_CASES, _IDS = cases([CONTRACT])


@pytest.mark.parametrize("contract,mode,row", _CASES, ids=_IDS)
def test_flag_reaches_its_config_field(monkeypatch, contract, mode, row) -> None:
    """A flag that parses and is never forwarded fails here."""
    assert_row_reaches_config(monkeypatch, contract, mode, row)


@pytest.mark.parametrize("contract,mode,row", _CASES, ids=_IDS)
def test_probe_value_differs_from_the_default(monkeypatch, contract, mode, row) -> None:
    """Trap 1: without this every row above could pass vacuously."""
    assert_row_value_is_not_the_default(monkeypatch, contract, mode, row)


def test_every_declared_flag_has_a_contract_row(monkeypatch) -> None:
    """A new flag added without a row fails HERE."""
    declared = declared_option_strings(CONTRACT.build_parser(monkeypatch))
    covered = CONTRACT.covered_flags
    assert declared == covered, (
        f"flags with no contract row {sorted(declared - covered)}; rows for flags "
        f"the parser no longer declares {sorted(covered - declared)}"
    )


def test_probe_values_are_mutually_distinct() -> None:
    """Trap 3: two rows sharing a value could hide a cross-wired forward."""
    values = [row.expected for row in CONTRACT.modes[0].rows if row.field is not None]
    assert len(values) == len(set(map(repr, values))), values


def test_every_config_field_has_a_flag() -> None:
    """A config field no flag can set is a knob only a wrapper can turn."""
    row_fields = {row.field for row in CONTRACT.modes[0].rows if row.field}
    config_fields = {f.name for f in fields(common.SegTrainingConfig)}
    assert row_fields == config_fields, (
        sorted(config_fields - row_fields), sorted(row_fields - config_fields)
    )


def test_parser_defaults_equal_the_config_defaults() -> None:
    """The parser reads its defaults off ``SegTrainingConfig``; they cannot drift.

    Compared per field on the PARSER (``get_default``), not through the config, so a
    hard-coded parser default that the config would later overwrite is still seen.
    ``experiment_name`` is the one field whose default is derived (a timestamped name).
    """
    parser = common._build_parser()
    default = common.SegTrainingConfig()
    for f in fields(common.SegTrainingConfig):
        if f.name == "experiment_name":
            assert parser.get_default("experiment_name") is None
            continue
        assert parser.get_default(f.name) == getattr(default, f.name), f.name
    assert default.experiment_name.startswith("convunext_seg_tiny_")


def test_the_default_experiment_name_carries_the_variant_and_a_timestamp() -> None:
    small = common.SegTrainingConfig(variant="small")
    assert small.experiment_name.startswith("convunext_seg_small_")
    assert small.experiment_name != common.SegTrainingConfig(variant="tiny").experiment_name
    explicit = common.SegTrainingConfig(experiment_name="mine")
    assert explicit.experiment_name == "mine"


def test_variant_choices_are_exactly_the_model_variant_table() -> None:
    """The trainer offers what the model builds: no invented or dropped variant."""
    parser = common._build_parser()
    (action,) = [a for a in parser._actions if "--variant" in a.option_strings]
    assert set(action.choices) == set(CONVUNEXT_CONFIGS)
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--variant", "huge"])
    assert exc.value.code == 2
    with pytest.raises(ValueError, match="variant"):
        common.SegTrainingConfig(variant="huge")


# ---------------------------------------------------------------------
# Refusals at config time
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "argv,message",
    [
        (("--epochs", "0"), "epochs"),
        (("--batch-size", "0"), "batch_size"),
        (("--learning-rate", "0"), "learning_rate"),
        (("--weight-decay=-0.1",), "weight_decay"),
        (("--patience", "0"), "patience"),
        (("--validation-split", "0"), "validation_split"),
        (("--validation-split", "1"), "validation_split"),
        (("--max-samples", "1"), "max_samples"),
        (("--warmup-epochs=-1",), "warmup_epochs"),
        (("--epochs", "3", "--warmup-epochs", "3"), "warmup_epochs"),
        (("--seed=-1",), "seed"),
        (("--viz-freq", "0"), "viz_freq"),
        (("--viz-samples", "0"), "viz_samples"),
        # the fit split: 100 samples, 10% held out -> 90 fit; a batch of 91 has no step
        (("--max-samples", "100", "--batch-size", "91"), "batch_size"),
        # the default pool (3680) with a batch larger than the whole fit split
        (("--batch-size", "4000"), "batch_size"),
        # a 1-sample val split rounds to 0
        (("--max-samples", "5", "--batch-size", "2"), "validation_split"),
    ],
    ids=["epochs0", "batch0", "lr0", "wd-neg", "patience0", "split0", "split1", "max-samples1",
         "warmup-neg", "warmup-ge-epochs", "seed-neg", "viz-freq0", "viz-samples0",
         "fit-below-batch", "batch-above-pool", "val-split-empty"],
)
def test_an_out_of_range_value_is_refused_by_the_config(argv, message) -> None:
    with pytest.raises(ValueError, match=message):
        common.config_from_args(common.parse_arguments(list(argv)))


def test_a_fit_split_of_exactly_one_batch_is_accepted() -> None:
    """The boundary of the fit-split rule: 90 fit samples and a batch of 90 is one step."""
    config = common.SegTrainingConfig(max_samples=100, batch_size=90)
    assert config.batch_size == 90


@pytest.mark.parametrize("variant", sorted(CONVUNEXT_CONFIGS))
def test_image_size_below_the_variant_minimum_is_refused(variant) -> None:
    minimum = 2 ** CONVUNEXT_CONFIGS[variant]["depth"]
    assert common.min_image_size(variant) == minimum
    assert common.SegTrainingConfig(variant=variant, image_size=minimum).image_size == minimum
    with pytest.raises(ValueError, match="image_size"):
        common.SegTrainingConfig(variant=variant, image_size=minimum - 1)


@pytest.mark.parametrize("variant", ["tiny", "base", "xlarge"])
def test_min_image_size_is_the_measured_forward_boundary(variant) -> None:
    """Re-measure the rule against the REAL model, on CPU, at the boundary and one below.

    ``create_convunext`` builds at every size (measured down to 1 px); the forward pass is
    what fails, so the probe runs the model. Widths are cut to 4 channels and one block
    per level to keep this cheap: the boundary depends on the depth alone. An odd size
    right above the boundary is run too, since the trainer accepts odd ``--image-size``.
    """
    depth = CONVUNEXT_CONFIGS[variant]["depth"]
    minimum = common.min_image_size(variant)

    def forward(size: int) -> Tuple[int, ...]:
        with tf.device("/CPU:0"):
            model = create_convunext(
                input_shape=(size, size, 3), use_bias=True, output_channels=3,
                final_activation="linear", depth=depth, initial_filters=4,
                blocks_per_level=1, drop_path_rate=0.0,
            )
            out = model(np.zeros((1, size, size, 3), "float32"), training=False)
        return tuple(out.shape)

    assert forward(minimum) == (1, minimum, minimum, 3)
    assert forward(minimum + 1) == (1, minimum + 1, minimum + 1, 3)
    with pytest.raises(tf.errors.InvalidArgumentError):
        forward(minimum - 1)


# ---------------------------------------------------------------------
# The run directory: resolved, refused when reused, never written by part A
# ---------------------------------------------------------------------

def test_resolve_new_run_dir_anchors_a_relative_output_dir_at_the_repo_root_and_writes_nothing() -> None:
    """Only COMPUTES the path a default run would use; nothing is created there."""
    name = f"seg_probe_never_created_{uuid.uuid4().hex}"
    config = common.SegTrainingConfig(experiment_name=name)
    run_dir = common.resolve_new_run_dir(config)
    assert run_dir == REPO_ROOT / "results" / name
    assert not run_dir.exists()


def test_resolve_new_run_dir_honours_an_absolute_output_dir(tmp_path) -> None:
    config = common.SegTrainingConfig(output_dir=str(tmp_path), experiment_name="a")
    assert common.resolve_new_run_dir(config) == tmp_path / "a"
    assert list(tmp_path.iterdir()) == []


def test_a_reused_experiment_name_is_refused_and_nothing_is_written(tmp_path) -> None:
    """A directory that holds a run is refused; its files are byte-identical afterwards."""
    old = tmp_path / "taken"
    old.mkdir()
    (old / "config.json").write_text('{"first": "run"}')
    (old / "best_model.keras").write_bytes(b"weights")
    before = {p.name: p.read_bytes() for p in old.iterdir()}
    config = common.SegTrainingConfig(output_dir=str(tmp_path), experiment_name="taken")
    with pytest.raises(FileExistsError, match="taken"):
        common.resolve_new_run_dir(config)
    assert {p.name: p.read_bytes() for p in old.iterdir()} == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["taken"]


# ---------------------------------------------------------------------
# main(): parse first, refuse before the GPU, configure the GPU, then train
# ---------------------------------------------------------------------

class _Probe(Exception):
    """Raised by a sentinel so ``main()`` cannot progress into real work."""


def _record_setup_gpu(monkeypatch) -> list:
    calls: list = []
    monkeypatch.setattr(common, "setup_gpu", lambda **kwargs: calls.append(kwargs))
    return calls


def _record_train(monkeypatch) -> list:
    """Replace ``common.train`` with a recorder of the config it is handed."""
    handed: list = []
    monkeypatch.setattr(common, "train", lambda config: handed.append(config))
    return handed


def test_gpu_flag_reaches_setup_gpu_and_train_receives_the_config(monkeypatch, tmp_path) -> None:
    """``--gpu 1`` arrives as ``setup_gpu(gpu_id=1)`` and ``train`` gets the parsed config.

    Pins the ORDER of ``main``: parse, build the config, refuse a reused name, configure the
    GPU, and only then ``train``. ``train`` is stubbed, so nothing exists in the output
    directory afterwards.
    """
    calls = _record_setup_gpu(monkeypatch)
    handed = _record_train(monkeypatch)
    common.main(["--gpu", "1", "--variant", "small", "--output-dir", str(tmp_path),
                 "--experiment-name", "x"])
    assert calls == [{"gpu_id": 1}], f"--gpu 1 did not reach setup_gpu(gpu_id=1): {calls!r}"
    assert len(handed) == 1 and handed[0].variant == "small" and handed[0].experiment_name == "x"
    assert list(tmp_path.iterdir()) == []


def test_no_gpu_flag_passes_none_to_setup_gpu(monkeypatch, tmp_path) -> None:
    calls = _record_setup_gpu(monkeypatch)
    _record_train(monkeypatch)
    common.main(["--output-dir", str(tmp_path), "--experiment-name", "x"])
    assert calls == [{"gpu_id": None}]


def test_the_gpu_is_configured_before_train_is_called(monkeypatch, tmp_path) -> None:
    order: list = []
    monkeypatch.setattr(common, "setup_gpu", lambda **kwargs: order.append("setup_gpu"))
    monkeypatch.setattr(common, "train", lambda config: order.append("train"))
    common.main(["--output-dir", str(tmp_path), "--experiment-name", "x"])
    assert order == ["setup_gpu", "train"]


def test_main_logs_and_reraises_a_failed_train(monkeypatch, tmp_path) -> None:
    _record_setup_gpu(monkeypatch)

    def failing(config):
        raise RuntimeError("cache missing")

    monkeypatch.setattr(common, "train", failing)
    with pytest.raises(RuntimeError, match="cache missing"):
        common.main(["--output-dir", str(tmp_path), "--experiment-name", "x"])


def test_a_reused_name_is_refused_before_the_gpu_is_configured(monkeypatch, tmp_path) -> None:
    calls = _record_setup_gpu(monkeypatch)
    old = tmp_path / "taken"
    old.mkdir()
    (old / "config.json").write_text("{}")
    with pytest.raises(FileExistsError):
        common.main(["--output-dir", str(tmp_path), "--experiment-name", "taken", "--gpu", "1"])
    assert calls == [], "a refused run must not touch the GPU"
    assert sorted(p.name for p in old.iterdir()) == ["config.json"]


def test_a_bad_value_is_refused_before_the_gpu_is_configured(monkeypatch, tmp_path) -> None:
    calls = _record_setup_gpu(monkeypatch)
    with pytest.raises(ValueError, match="epochs"):
        common.main(["--epochs", "0", "--gpu", "1", "--output-dir", str(tmp_path)])
    assert calls == []
    assert list(tmp_path.iterdir()) == []


def test_help_prints_usage_and_reaches_no_expensive_call(monkeypatch, capsys, tmp_path) -> None:
    """``--help`` exits 0 with a ``usage:`` line before the GPU or a directory is touched."""
    touched: list = []

    def _sentinel(name):
        def _raise(*a, **k):
            touched.append(name)
            raise _Probe(name)
        return _raise

    for name in ("setup_gpu", "resolve_new_run_dir", "refuse_existing_run"):
        monkeypatch.setattr(common, name, _sentinel(name))
    with pytest.raises(SystemExit) as exc:
        common.main(["--help", "--output-dir", str(tmp_path)])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert out.lstrip().startswith("usage:"), out[:200]
    for flag in ("--variant", "--image-size", "--max-samples", "--warmup-epochs",
                 "--experiment-name", "--gpu"):
        assert flag in out, f"{flag} is not advertised by --help"
    assert touched == []
    assert list(tmp_path.iterdir()) == []


def test_help_in_a_fresh_process_allocates_no_directory(tmp_path) -> None:
    """Real process, cwd in an empty ``tmp_path``: ``--help`` exits 0, prints usage and
    leaves the working directory and the named output directory absent."""
    out_dir = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; from train.convunext.common import main; main(sys.argv[1:])",
         "--help", "--output-dir", str(out_dir), "--experiment-name", "help_probe"],
        cwd=tmp_path, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.lstrip().startswith("usage:"), proc.stdout[:200]
    assert not out_dir.exists()
    assert list(tmp_path.iterdir()) == []
