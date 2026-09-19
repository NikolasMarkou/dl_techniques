"""CLI contract guard for ``src/train/convnext/`` (common.py plus the V1 / V2 wrappers).

The defect class
----------------
A trainer declares a flag, ``--help`` advertises it, and nothing between argparse and
the point of use reads it. The pre-normalization ConvNeXt scripts parsed ``--gpu`` and
ignored it and had no ``parse_arguments(argv)`` at all. This file pins the repaired
surface with the shared ``tests/test_train/_cli_contract.py`` driver, once per family:

- every flag reaches its ``TrainingConfig`` field with a probe value that is NOT the
  default (trap 1) and that no other row uses (trap 3);
- the row table equals the flags the REAL parser declares (trap 2);
- ``--gpu`` is not a config field: it is checked on the namespace AND through
  ``main()`` into ``setup_gpu``;
- ``--variant`` offers exactly the keys of each family's ``MODEL_VARIANTS``;
- ``imagenet`` is refused with the explicit message, at the parser and at the config;
- ``--help`` prints ``usage:`` and reaches no expensive call; the wrappers pass their
  family on.

Nothing here trains, loads a dataset or allocates a GPU.
"""

from __future__ import annotations

from typing import Tuple

import pytest

import train.convnext.common as common
import train.convnext.train_convnext_v1 as wrapper_v1
import train.convnext.train_convnext_v2 as wrapper_v2

from .._cli_contract import (  # noqa: TID252 -- shared driver, one level up
    Contract,
    Mode,
    Row,
    assert_row_reaches_config,
    assert_row_value_is_not_the_default,
    cases,
    declared_option_strings,
)


def _rows(variant_probe: str) -> Tuple[Row, ...]:
    """One row per declared flag. Probes differ from the defaults (cifar10 / cifar10 /
    k=7 / s=2 / 100 epochs / ...) and from each other. ``variant_probe`` is the one
    value that has to be a member of the family's own variant table."""
    return (
        Row(("--dataset",), ("--dataset", "cifar100"), "dataset", "cifar100"),
        Row(("--validation-split",), ("--validation-split", "0.25"), "validation_split", 0.25),
        Row(("--max-samples",), ("--max-samples", "321"), "max_samples", 321),
        Row(("--variant",), ("--variant", variant_probe), "variant", variant_probe),
        Row(("--kernel-size",), ("--kernel-size", "5"), "kernel_size", 5),
        Row(("--strides",), ("--strides", "4"), "strides", 4),
        Row(("--drop-path-rate",), ("--drop-path-rate", "0.35"), "drop_path_rate", 0.35),
        Row(("--stochastic-mode",), ("--stochastic-mode", "gradient"), "stochastic_mode", "gradient"),
        Row(("--dropout-rate",), ("--dropout-rate", "0.15"), "dropout_rate", 0.15),
        # BooleanOptionalAction: one row accounts for both spellings; default True.
        Row(("--use-gamma", "--no-use-gamma"), ("--no-use-gamma",), "use_gamma", False),
        Row(("--epochs",), ("--epochs", "7"), "epochs", 7),
        Row(("--batch-size",), ("--batch-size", "96"), "batch_size", 96),
        Row(("--learning-rate",), ("--learning-rate", "1.5e-3"), "learning_rate", 1.5e-3),
        Row(("--weight-decay",), ("--weight-decay", "0.02"), "weight_decay", 0.02),
        Row(("--lr-schedule",), ("--lr-schedule", "exponential"), "lr_schedule", "exponential"),
        Row(("--warmup-epochs",), ("--warmup-epochs", "3"), "warmup_epochs", 3),
        Row(("--patience",), ("--patience", "9"), "patience", 9),
        Row(("--seed",), ("--seed", "1234"), "seed", 1234),
        # Opt-in (default False): the probe is the non-default True.
        Row(("--epoch-analysis",), ("--epoch-analysis",), "epoch_analysis", True),
        # BooleanOptionalAction, default True: the probe is the non-default spelling.
        Row(("--model-analysis", "--no-model-analysis"), ("--no-model-analysis",),
            "model_analysis", False),
        Row(("--output-dir",), ("--output-dir", "/probe/convnext-out"), "output_dir",
            "/probe/convnext-out"),
        Row(("--experiment-name",), ("--experiment-name", "probe-experiment-name"),
            "experiment_name", "probe-experiment-name"),
        # Not a config field: consumed once by setup_gpu in main(); the hop into
        # setup_gpu is test_gpu_flag_reaches_setup_gpu.
        Row(("--gpu",), ("--gpu", "1"), namespace_dest="gpu", expected=1),
    )


def _contract(family: str, variant_probe: str) -> Contract:
    return Contract(
        name=f"train.convnext.train_convnext_{family}",
        build_parser=lambda monkeypatch: common._build_parser(family),
        build_config=lambda monkeypatch: common.config_from_args(
            common.parse_arguments(None, family), family),
        # `--lr-schedule exponential` cannot carry the 3 warmup epochs the warmup row
        # sets, and each row runs alone against otherwise-default argv, so both stand.
        modes=(Mode(id="plain", required_argv=(), rows=_rows(variant_probe)),),
    )


# The probe variants differ from the default "cifar10" and exist in each table.
CONTRACTS: Tuple[Contract, ...] = (_contract("v1", "tiny"), _contract("v2", "atto"))
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


@pytest.mark.parametrize("contract", CONTRACTS, ids=[c.name for c in CONTRACTS])
def test_probe_values_are_mutually_distinct(contract) -> None:
    """Trap 3: two rows sharing a value could hide a cross-wired forward."""
    rows = contract.modes[0].rows
    values = [row.expected for row in rows if row.field is not None
              and not isinstance(row.expected, bool)]
    assert len(values) == len(set(map(repr, values))), values


def test_every_config_field_but_the_family_has_a_flag() -> None:
    """A config field no flag can set is a knob only the wrapper can turn."""
    from dataclasses import fields

    row_fields = {row.field for row in CONTRACTS[0].modes[0].rows if row.field}
    config_fields = {f.name for f in fields(common.TrainingConfig)} - {"model_family"}
    assert row_fields == config_fields, (
        sorted(config_fields - row_fields), sorted(row_fields - config_fields)
    )


@pytest.mark.parametrize("family", ["v1", "v2"])
def test_variant_choices_are_exactly_the_model_variant_table(family) -> None:
    """The wrappers offer what the model class builds: no invented or dropped variant."""
    table = {
        "v1": common.ConvNeXtV1.MODEL_VARIANTS,
        "v2": common.ConvNeXtV2.MODEL_VARIANTS,
    }[family]
    parser = common._build_parser(family)
    (action,) = [a for a in parser._actions if "--variant" in a.option_strings]
    assert set(action.choices) == set(table)
    # v1 has no `atto`, v2 has no `small`: a shared choices tuple would accept both.
    other = "atto" if family == "v1" else "small"
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--variant", other])
    assert exc.value.code == 2


def test_parser_defaults_equal_the_config_defaults() -> None:
    """The parser reads its defaults off ``TrainingConfig``; they cannot drift."""
    for family in ("v1", "v2"):
        config = common.config_from_args(common.parse_arguments([], family), family)
        default = common.TrainingConfig(model_family=family)
        for name in (
            "variant", "kernel_size", "strides", "drop_path_rate", "stochastic_mode",
            "dropout_rate", "use_gamma", "dataset", "validation_split", "max_samples",
            "epochs", "batch_size", "learning_rate", "weight_decay", "lr_schedule",
            "warmup_epochs", "patience", "seed", "epoch_analysis", "model_analysis",
            "output_dir",
        ):
            assert getattr(config, name) == getattr(default, name), (family, name)
        assert config.model_family == family
        assert config.experiment_name.startswith(f"convnext_{family}_cifar10_cifar10_")


def test_per_dataset_regularization_defaults_are_filled_when_the_flag_is_absent() -> None:
    config = common.config_from_args(common.parse_arguments(["--dataset", "cifar100"], "v1"), "v1")
    assert (config.drop_path_rate, config.dropout_rate) == (0.2, 0.2)
    explicit = common.config_from_args(
        common.parse_arguments(["--dataset", "cifar100", "--drop-path-rate", "0.0"], "v1"), "v1")
    assert explicit.drop_path_rate == 0.0, "an explicit 0 must not be replaced by the default"


# ---------------------------------------------------------------------
# --gpu, --help, refusals: nothing expensive may run
# ---------------------------------------------------------------------


class _Probe(Exception):
    """Raised by a sentinel so ``main()`` cannot progress into real work."""


def _forbid_expensive_calls(monkeypatch) -> list:
    touched: list = []

    def _make(name):
        def _sentinel(*a, **k):
            touched.append(name)
            raise _Probe(name)
        return _sentinel

    for name in ("setup_gpu", "train", "prepare_data"):
        monkeypatch.setattr(common, name, _make(name))
    return touched


def test_gpu_flag_reaches_setup_gpu(monkeypatch) -> None:
    """``--gpu 1`` must arrive as ``setup_gpu(gpu_id=1)`` inside ``main()``."""
    calls = []

    def _sentinel(**kwargs):
        calls.append(kwargs)
        raise _Probe

    monkeypatch.setattr(common, "setup_gpu", _sentinel)
    with pytest.raises(_Probe):
        common.main("v1", ["--gpu", "1"])
    assert calls == [{"gpu_id": 1}], f"--gpu 1 did not reach setup_gpu(gpu_id=1): {calls!r}"


@pytest.mark.parametrize("wrapper,family", [(wrapper_v1, "v1"), (wrapper_v2, "v2")])
def test_wrapper_passes_its_family_and_argv_to_the_orchestrator(
        monkeypatch, wrapper, family) -> None:
    """The thin wrappers add nothing but the family; a swapped family trains the wrong model."""
    seen = []
    monkeypatch.setattr(wrapper, "run_convnext", lambda fam, argv: seen.append((fam, argv)))
    wrapper.main(["--epochs", "2"])
    assert seen == [(family, ["--epochs", "2"])]


@pytest.mark.parametrize("family", ["v1", "v2"])
def test_help_prints_usage_and_reaches_no_expensive_call(monkeypatch, capsys, family) -> None:
    """``--help`` exits 0 with a ``usage:`` line; exit 0 alone proves nothing."""
    touched = _forbid_expensive_calls(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        common.main(family, ["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert out.lstrip().startswith("usage:"), out[:200]
    for flag in ("--strides", "--warmup-epochs", "--experiment-name", "--max-samples"):
        assert flag in out, f"{flag} is not advertised by --help"
    assert touched == []


def test_imagenet_is_refused_with_the_explicit_message_before_any_expensive_call(
        monkeypatch, capsys) -> None:
    """``imagenet`` is a documented refusal, not a generic 'invalid choice'."""
    touched = _forbid_expensive_calls(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        common.main("v1", ["--dataset", "imagenet"])
    assert exc.value.code == 2
    assert "imagenet2012" in capsys.readouterr().err
    assert touched == []
    with pytest.raises(ValueError, match="imagenet2012"):
        common.TrainingConfig(dataset="imagenet")


@pytest.mark.parametrize(
    "argv,message",
    [
        (("--validation-split", "0"), "validation_split"),
        (("--validation-split", "1"), "validation_split"),
        (("--epochs", "0"), "epochs"),
        (("--lr-schedule", "exponential", "--warmup-epochs", "1"), "warmup_epochs"),
        (("--epochs", "3", "--warmup-epochs", "3"), "warmup_epochs"),
        (("--drop-path-rate", "1.0"), "drop_path_rate"),
    ],
    ids=["split0", "split1", "epochs0", "warmup-non-cosine", "warmup-ge-epochs", "droppath1"],
)
def test_an_out_of_range_value_raises_before_gpu_setup(monkeypatch, argv, message) -> None:
    """A bad combination is refused by the config, before ``setup_gpu`` / training."""
    touched = _forbid_expensive_calls(monkeypatch)
    with pytest.raises(ValueError, match=message):
        common.main("v1", list(argv))
    assert touched == []
