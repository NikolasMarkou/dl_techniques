"""CLI contract guard for ``src/train/power_mlp/train_power_mlp.py``.

The defect class
----------------
A trainer declares a flag, ``--help`` advertises it, and nothing between argparse
and the point of use reads it. The PowerMLP trainer shipped exactly that: four
flags inherited from ``create_base_argument_parser`` (``--image-size``,
``--lr-schedule``, ``--show-plots``, and a ``--weight-decay`` / ``--optimizer``
pair that never reached the optimizer) did nothing. This file pins the repaired
surface with the shared ``tests/test_train/_cli_contract.py`` driver:

- every flag reaches its ``TrainingConfig`` field with a probe value that is NOT
  the default (trap 1, ``test_probe_value_differs_from_the_default``);
- the row table equals the flags the REAL parser declares (trap 2);
- probe values are mutually distinct (trap 3), so a cross-wired forward moves
  the wrong field;
- ``--gpu`` is not a config field; it is checked on the namespace AND through
  ``main()`` into ``setup_gpu`` (the mothnet D-011 defect class);
- the four removed flags (``--image-size``, ``--lr-schedule``, ``--show-plots``,
  ``--no-epoch-analysis``) are absent from the parser and rejected;
- an unknown ``--optimizer``, ``--kernel-initializer`` or ``--input-scaling`` exits 2
  (there is no silent fall-back to a default);
- ``--help`` prints ``usage:`` and reaches no expensive call.

Nothing here trains, loads a dataset or allocates a GPU.
"""

from __future__ import annotations

from typing import Tuple

import pytest

import train.power_mlp.train_power_mlp as tpm

from .._cli_contract import (  # noqa: TID252 -- shared driver, one level up
    Contract,
    Mode,
    Row,
    assert_row_reaches_config,
    assert_row_value_is_not_the_default,
    cases,
    declared_option_strings,
)

#: One row per declared flag; every probe value is unique and differs from its
#: own default (adam / mnist / default / k=2 / 100 epochs / ...).
POWER_MLP_ROWS: Tuple[Row, ...] = (
    Row(("--dataset",), ("--dataset", "cifar10"), "dataset", "cifar10"),
    Row(("--validation-split",), ("--validation-split", "0.25"), "validation_split", 0.25),
    Row(("--architecture",), ("--architecture", "large"), "architecture", "large"),
    Row(("--k",), ("--k", "3"), "k", 3),
    Row(("--dropout-rate",), ("--dropout-rate", "0.35"), "dropout_rate", 0.35),
    Row(
        ("--batch-normalization",), ("--batch-normalization",),
        "batch_normalization", True,
    ),
    # Probes differ from the measured defaults (lecun_normal / unit) and are not in
    # the grid that chose them, so a cross-wire or a dropped forward is visible.
    Row(
        ("--kernel-initializer",), ("--kernel-initializer", "glorot_uniform"),
        "kernel_initializer", "glorot_uniform",
    ),
    Row(("--input-scaling",), ("--input-scaling", "standardize"), "input_scaling", "standardize"),
    Row(("--epochs",), ("--epochs", "7"), "epochs", 7),
    Row(("--batch-size",), ("--batch-size", "96"), "batch_size", 96),
    Row(("--learning-rate",), ("--learning-rate", "1.5e-3"), "learning_rate", 1.5e-3),
    Row(("--optimizer",), ("--optimizer", "sgd"), "optimizer", "sgd"),
    Row(("--weight-decay",), ("--weight-decay", "0.02"), "weight_decay", 0.02),
    Row(("--patience",), ("--patience", "9"), "patience", 9),
    Row(("--seed",), ("--seed", "1234"), "seed", 1234),
    # Opt-in (default False): the probe is the non-default True.
    Row(("--epoch-analysis",), ("--epoch-analysis",), "epoch_analysis", True),
    Row(
        ("--output-dir",), ("--output-dir", "/probe/power-mlp-out"),
        "output_dir", "/probe/power-mlp-out",
    ),
    Row(
        ("--experiment-name",), ("--experiment-name", "probe-experiment-name"),
        "experiment_name", "probe-experiment-name",
    ),
    # Not a config field: consumed once by setup_gpu in main(). Namespace-level
    # here; `test_gpu_flag_reaches_setup_gpu` proves the hop into setup_gpu.
    Row(("--gpu",), ("--gpu", "1"), namespace_dest="gpu", expected=1),
)

POWER_MLP_TRAINER = Contract(
    name="train.power_mlp.train_power_mlp",
    build_parser=lambda monkeypatch: tpm._build_parser(),
    build_config=lambda monkeypatch: tpm.config_from_args(tpm.parse_arguments()),
    modes=(Mode(id="plain", required_argv=(), rows=POWER_MLP_ROWS),),
)

CONTRACTS: Tuple[Contract, ...] = (POWER_MLP_TRAINER,)
_CASES, _IDS = cases(CONTRACTS)


@pytest.mark.parametrize("contract,mode,row", _CASES, ids=_IDS)
def test_flag_reaches_its_config_field(monkeypatch, contract, mode, row) -> None:
    """A flag that parses and is never forwarded fails here."""
    assert_row_reaches_config(monkeypatch, contract, mode, row)


@pytest.mark.parametrize("contract,mode,row", _CASES, ids=_IDS)
def test_probe_value_differs_from_the_default(monkeypatch, contract, mode, row) -> None:
    """Trap 1: without this every row above could pass vacuously."""
    assert_row_value_is_not_the_default(monkeypatch, contract, mode, row)


@pytest.mark.parametrize("contract", CONTRACTS, ids=[c.name for c in CONTRACTS])
def test_every_declared_flag_has_a_contract_row(monkeypatch, contract) -> None:
    """A new flag added without a row fails HERE."""
    declared = declared_option_strings(contract.build_parser(monkeypatch))
    covered = contract.covered_flags
    assert declared == covered, (
        f"flags with no contract row {sorted(declared - covered)}; rows for flags "
        f"the parser no longer declares {sorted(covered - declared)}"
    )


def test_probe_values_are_mutually_distinct() -> None:
    """Trap 3: two rows sharing a value could hide a cross-wired forward."""
    values = [row.expected for row in POWER_MLP_ROWS if row.field is not None
              and not isinstance(row.expected, bool)]
    assert len(values) == len(set(map(repr, values))), values


def test_parser_defaults_equal_the_config_defaults() -> None:
    """The parser reads its defaults off ``TrainingConfig``; they cannot drift."""
    args = tpm.parse_arguments([])
    config = tpm.config_from_args(args)
    default = tpm.TrainingConfig()
    for name in (
        "dataset", "validation_split", "architecture", "k", "dropout_rate",
        "batch_normalization", "kernel_initializer", "input_scaling", "epochs",
        "batch_size", "learning_rate", "optimizer", "weight_decay", "patience", "seed",
        "epoch_analysis", "output_dir",
    ):
        assert getattr(config, name) == getattr(default, name), name
    assert config.experiment_name.startswith("powermlp_mnist_default_")


# ---------------------------------------------------------------------
# --gpu: the flag whose forwarding needs main(), not just the parser
# ---------------------------------------------------------------------


class _Probe(Exception):
    """Raised by a sentinel so ``main()`` cannot progress into real work."""


def test_gpu_flag_reaches_setup_gpu(monkeypatch) -> None:
    """``--gpu 1`` must arrive as ``setup_gpu(gpu_id=1)`` inside ``main()``."""
    calls = []

    def _sentinel(**kwargs):
        calls.append(kwargs)
        raise _Probe

    monkeypatch.setattr(tpm, "setup_gpu", _sentinel)
    with pytest.raises(_Probe):
        tpm.main(["--gpu", "1"])
    assert calls == [{"gpu_id": 1}], f"--gpu 1 did not reach setup_gpu(gpu_id=1): {calls!r}"


def _forbid_expensive_calls(monkeypatch) -> list:
    """Make setup_gpu / train_model / prepare_data record-and-raise."""
    touched: list = []

    def _make(name):
        def _sentinel(*a, **k):
            touched.append(name)
            raise _Probe(name)
        return _sentinel

    for name in ("setup_gpu", "train_model", "prepare_data"):
        monkeypatch.setattr(tpm, name, _make(name))
    return touched


def test_help_prints_usage_and_reaches_no_expensive_call(monkeypatch, capsys) -> None:
    """``--help`` exits 0 with a ``usage:`` line; exit 0 alone proves nothing."""
    touched = _forbid_expensive_calls(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        tpm.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert out.lstrip().startswith("usage:"), out[:200]
    assert "--optimizer" in out and "--weight-decay" in out
    for flag in ("--kernel-initializer", "--input-scaling", "--epoch-analysis"):
        assert flag in out, f"{flag} is not advertised by --help"
    assert "--no-epoch-analysis" not in out
    assert touched == []


@pytest.mark.parametrize(
    "argv",
    [
        ("--image-size", "64"),
        ("--lr-schedule", "cosine"),
        ("--show-plots",),
        ("--no-epoch-analysis",),
    ],
    ids=["--image-size", "--lr-schedule", "--show-plots", "--no-epoch-analysis"],
)
def test_removed_flags_are_absent_and_rejected(monkeypatch, argv) -> None:
    """Dead inherited flags are gone: not declared, and argparse refuses them."""
    assert argv[0] not in declared_option_strings(tpm._build_parser())
    touched = _forbid_expensive_calls(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        tpm.main(list(argv))
    assert exc.value.code == 2
    assert touched == [], "a rejected flag must exit before any expensive call"


def test_unknown_optimizer_exits_2_before_any_expensive_call(monkeypatch) -> None:
    """There is no silent fall-back to Adam for a name the trainer cannot build."""
    touched = _forbid_expensive_calls(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        tpm.main(["--optimizer", "lion"])
    assert exc.value.code == 2
    assert touched == []


@pytest.mark.parametrize(
    "argv",
    [
        ("--kernel-initializer", "orthogonal"),
        ("--input-scaling", "whiten"),
    ],
    ids=["--kernel-initializer", "--input-scaling"],
)
def test_a_value_outside_the_choices_exits_2_before_any_expensive_call(
        monkeypatch, argv) -> None:
    """Both new flags are closed sets: argparse refuses anything else."""
    touched = _forbid_expensive_calls(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        tpm.main(list(argv))
    assert exc.value.code == 2
    assert touched == []


def test_every_listed_choice_is_accepted_by_the_parser() -> None:
    """The choices tuples the trainer validates against are the ones argparse offers."""
    for name in tpm.KERNEL_INITIALIZERS:
        assert tpm.parse_arguments(["--kernel-initializer", name]).kernel_initializer == name
    for name in tpm.INPUT_SCALINGS:
        assert tpm.parse_arguments(["--input-scaling", name]).input_scaling == name


@pytest.mark.parametrize("split", ["0", "0.0", "1", "1.0", "-0.1"])
def test_out_of_range_validation_split_raises_before_gpu_setup(monkeypatch, split) -> None:
    """A bad split is refused by the config, before ``setup_gpu`` / training."""
    touched = _forbid_expensive_calls(monkeypatch)
    with pytest.raises(ValueError, match="validation_split"):
        tpm.main(["--validation-split", split])
    assert touched == []
