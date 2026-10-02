"""CLI contract guard for ``src/train/lightglue/train_lightglue.py``.

The trainer builds a ``LightGlueTrainConfig`` from the parsed namespace, so ``build_config``
is ``config_from_args(parse_arguments())`` and every row but ``--gpu`` is a ``field`` row.
``--gpu`` is also a config field, but the hop that matters is ``setup_gpu(gpu_id=...)`` in
``main()``, which ``test_gpu_flag_reaches_setup_gpu`` drives against a sentinel with the run
pointed at ``tmp_path``. See ``tests/test_train/_cli_contract.py`` for the three traps the
driver designs out.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Tuple

import pytest

import train.lightglue.train_lightglue as train_lightglue

from .._cli_contract import (  # noqa: TID252 -- shared driver, one level up
    Contract,
    Mode,
    Row,
    assert_row_reaches_config,
    assert_row_value_is_not_the_default,
    cases,
    declared_option_strings,
)

REQUIRED = ("--superpoint-checkpoint", "/probe/default-superpoint.keras")

#: One row per declared flag; every probe is unique across the table and differs from the
#: default. ``--descriptor-dim`` / ``--num-heads`` probes keep the width/heads rule true when
#: driven one at a time against the other's default (192 / 4 and 256 / 8).
LIGHTGLUE_ROWS: Tuple[Row, ...] = (
    Row(("--superpoint-checkpoint",), ("--superpoint-checkpoint", "/probe/superpoint.keras"),
        "superpoint_checkpoint", "/probe/superpoint.keras"),
    Row(("--coco-dir",), ("--coco-dir", "/probe/coco"), "coco_dir", "/probe/coco"),
    Row(("--val-images",), ("--val-images", "37"), "val_images", 37),
    Row(("--image-size",), ("--image-size", "96"), "image_size", 96),
    Row(("--max-keypoints",), ("--max-keypoints", "77"), "max_keypoints", 77),
    Row(("--nms-radius",), ("--nms-radius", "6"), "nms_radius", 6),
    Row(("--detection-threshold",), ("--detection-threshold", "0.0123"),
        "detection_threshold", 0.0123),
    Row(("--border",), ("--border", "9"), "border", 9),
    Row(("--pos-threshold",), ("--pos-threshold", "2.5"), "pos_threshold", 2.5),
    Row(("--batch-size",), ("--batch-size", "5"), "batch_size", 5),
    Row(("--epochs",), ("--epochs", "17"), "epochs", 17),
    Row(("--steps-per-epoch",), ("--steps-per-epoch", "23"), "steps_per_epoch", 23),
    Row(("--validation-steps",), ("--validation-steps", "29"), "validation_steps", 29),
    Row(("--learning-rate",), ("--learning-rate", "0.03125"), "learning_rate", 0.03125),
    Row(("--weight-decay",), ("--weight-decay", "0.0625"), "weight_decay", 0.0625),
    Row(("--warmup-steps",), ("--warmup-steps", "41"), "warmup_steps", 41),
    Row(("--clip-norm",), ("--clip-norm", "0.375"), "clip_norm", 0.375),
    Row(("--patience",), ("--patience", "13"), "patience", 13),
    Row(("--num-layers",), ("--num-layers", "4"), "num_layers", 4),
    Row(("--descriptor-dim",), ("--descriptor-dim", "192"), "descriptor_dim", 192),
    Row(("--num-heads",), ("--num-heads", "8"), "num_heads", 8),
    Row(("--seed",), ("--seed", "1234"), "seed", 1234),
    Row(("--gpu",), ("--gpu", "3"), "gpu", 3),
    Row(("--output-dir",), ("--output-dir", "/probe/lightglue-output-dir"), "output_dir",
        "/probe/lightglue-output-dir"),
    Row(("--experiment-name",), ("--experiment-name", "probe-experiment-name"),
        "experiment_name", "probe-experiment-name"),
)

LIGHTGLUE_TRAINER = Contract(
    name="train.lightglue.train_lightglue",
    build_parser=lambda monkeypatch: train_lightglue._build_parser(),
    build_config=lambda monkeypatch: train_lightglue.config_from_args(
        train_lightglue.parse_arguments()),
    modes=(Mode(id="plain", required_argv=REQUIRED, rows=LIGHTGLUE_ROWS),),
)

CONTRACTS: Tuple[Contract, ...] = (LIGHTGLUE_TRAINER,)

_CASES, _IDS = cases(CONTRACTS)


@pytest.mark.parametrize("contract,mode,row", _CASES, ids=_IDS)
def test_flag_reaches_its_config_field(monkeypatch, contract, mode, row) -> None:
    assert_row_reaches_config(monkeypatch, contract, mode, row)


@pytest.mark.parametrize("contract,mode,row", _CASES, ids=_IDS)
def test_probe_value_differs_from_the_default(monkeypatch, contract, mode, row) -> None:
    assert_row_value_is_not_the_default(monkeypatch, contract, mode, row)


def test_probe_values_are_mutually_distinct() -> None:
    values = [repr(row.expected) for row in LIGHTGLUE_ROWS]
    assert len(values) == len(set(values)), sorted(values)


@pytest.mark.parametrize("contract", CONTRACTS, ids=[c.name for c in CONTRACTS])
def test_every_declared_flag_has_a_contract_row(monkeypatch, contract) -> None:
    declared = declared_option_strings(contract.build_parser(monkeypatch))
    covered = contract.covered_flags
    assert declared == covered, (
        f"{contract.name}: flags with no contract row {sorted(declared - covered)}; rows "
        f"for flags the parser no longer declares {sorted(covered - declared)}")
    assert len(LIGHTGLUE_ROWS) == len(declared), "a flag has two rows"


def test_every_config_field_has_a_flag() -> None:
    """A field no flag can set is a knob the CLI user cannot reach."""
    import dataclasses
    fields = {f.name for f in dataclasses.fields(train_lightglue.LightGlueTrainConfig)}
    rows = {row.field for row in LIGHTGLUE_ROWS}
    assert fields == rows, (sorted(fields - rows), sorted(rows - fields))


# ---------------------------------------------------------------------
# --help, missing / invalid flags
# ---------------------------------------------------------------------


def test_help_exits_zero_with_usage_and_allocates_no_device() -> None:
    root = Path(__file__).resolve().parents[3]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", TF_CPP_MIN_LOG_LEVEL="3",
               MPLBACKEND="Agg", PYTHONPATH=str(root / "src"))
    result = subprocess.run(
        [sys.executable, "-m", "train.lightglue.train_lightglue", "--help"],
        cwd=root, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]
    assert "usage:" in result.stdout
    assert "--superpoint-checkpoint" in result.stdout


def test_missing_required_flag_fails_with_a_clear_message(capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        train_lightglue.main([])
    assert exit_info.value.code == 2
    assert "--superpoint-checkpoint" in capsys.readouterr().err


@pytest.mark.parametrize("argv,message", [
    (["--nope"], "unrecognized arguments"),
    (["--epochs", "0"], "--epochs must be >= 1"),
    (["--learning-rate", "0"], "--learning-rate must be > 0"),
    (["--num-heads", "3"], "--descriptor-dim"),
    (["--detection-threshold", "1.5"], "--detection-threshold"),
    (["--val-images", "-1"], "--val-images must be >= 0"),
])
def test_unknown_and_invalid_flags_fail_with_a_clear_message(capsys, argv, message) -> None:
    with pytest.raises(SystemExit) as exit_info:
        train_lightglue.main([*REQUIRED, *argv])
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


# ---------------------------------------------------------------------
# --gpu: forwarding needs main(), not just the parser
# ---------------------------------------------------------------------


class _SetupGpuProbe(Exception):
    """Raised by the sentinel so ``main()`` stops at its next statement."""


def test_gpu_flag_reaches_setup_gpu(monkeypatch, tmp_path) -> None:
    calls = []

    def _sentinel(*args, **kwargs):
        calls.append((args, kwargs))
        raise _SetupGpuProbe

    monkeypatch.setattr(train_lightglue, "setup_gpu", _sentinel)
    argv = [*REQUIRED, "--gpu", "3", "--output-dir", str(tmp_path),
            "--experiment-name", "gpu_probe"]
    try:
        train_lightglue.main(argv)
    except _SetupGpuProbe:
        pass
    assert calls == [((), {"gpu_id": 3})], calls
    assert not (tmp_path / "gpu_probe").exists(), "setup_gpu ran after the run directory was made"


def test_a_failed_preflight_writes_nothing_and_returns_2(tmp_path) -> None:
    code = train_lightglue.main([
        "--superpoint-checkpoint", str(tmp_path / "missing.keras"),
        "--coco-dir", str(tmp_path), "--output-dir", str(tmp_path / "out"),
        "--experiment-name", "preflight"])
    assert code == 2
    assert not (tmp_path / "out").exists()
