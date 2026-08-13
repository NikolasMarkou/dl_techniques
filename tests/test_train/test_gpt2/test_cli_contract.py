"""Every flag the gpt2 trainers declare must reach the config field it claims.

Three entry points, zero coverage of this contract before this module:
``train.gpt2.pretrain``, ``train.gpt2.pretrain_so`` and ``train.gpt2.finetune``.
All three hand-map argparse destinations onto config fields one line at a time
-- 26, 31 and 14 assignments respectively -- and a dropped line is invisible:
``--help`` still advertises the flag, the parser still accepts it, the run still
starts, and the config silently keeps its default.

See ``tests/test_train/_cli_contract.py`` for the driver, the three vacuity
traps it designs out, and why this is the inverse of the NTM trainers'
``assert_flag_is_not_registered``.

``train.gpt2.finetune`` is the odd one: it has no ``_build_parser`` /
``_config_from_args`` pair, building its parser and assigning its config inline
in ``main()``. It is driven through ``main()`` with ``finetune_gpt2`` replaced by
a recorder, so the REAL assignment block runs. Its parser has a required
mutually-exclusive group (``--hf-dataset`` / ``--text-dir``), which is why it
needs two modes.
"""

from __future__ import annotations

import argparse
import sys
import types
from typing import Any, Tuple

import pytest

from train.gpt2 import finetune as ft
from train.gpt2 import pretrain as pt
from train.gpt2 import pretrain_so as so

from .._cli_contract import (
    CLM_PRETRAIN_ROWS,
    Contract,
    Mode,
    Row,
    assert_row_reaches_config,
    assert_row_value_is_not_the_default,
    cases,
    declared_option_strings,
)


# ---------------------------------------------------------------------
# train.gpt2.finetune: parser + config are both inline in main()
# ---------------------------------------------------------------------

class _ParserCaptured(Exception):
    """Carries the parser out of ``main()`` before it consumes argv."""

    def __init__(self, parser: argparse.ArgumentParser) -> None:
        super().__init__("finetune parser captured")
        self.parser = parser


def _finetune_parser(monkeypatch) -> argparse.ArgumentParser:
    """The REAL parser ``train.gpt2.finetune.main()`` builds.

    ``main()`` never returns it, so a recording subclass is swapped in for the
    duration of the construction and raises out of ``parse_args``. The captured
    instance then gets the genuine ``parse_args`` rebound onto it, so callers
    receive a fully working parser rather than one that explodes on use.

    The substitution is made on ``finetune``'s own module-level ``argparse``
    NAME, never on ``argparse.ArgumentParser`` itself. MEASURED: patching the
    attribute on the argparse module is an infinite recursion
    (``RecursionError`` at ``argparse.py:1770``), because
    ``ArgumentParser.__init__`` calls ``super(ArgumentParser, self).__init__``
    and resolves that name from its own module globals -- which now point at the
    subclass.

    :param monkeypatch: pytest monkeypatch fixture.
    :return: The parser, ready to ``parse_args``.
    :raises AssertionError: If ``main()`` did not reach ``parse_args``.
    """
    real_parse_args = argparse.ArgumentParser.parse_args

    class _Recording(argparse.ArgumentParser):
        def parse_args(self, *args: Any, **kwargs: Any):
            raise _ParserCaptured(self)

    class _ArgparseShim:
        """``argparse``, with only ``ArgumentParser`` replaced."""

        ArgumentParser = _Recording

        def __getattr__(self, name: str) -> Any:
            return getattr(argparse, name)

    monkeypatch.setattr(ft, "argparse", _ArgparseShim())
    monkeypatch.setattr(ft, "setup_gpu", lambda **kwargs: None)
    try:
        ft.main()
    except _ParserCaptured as captured:
        parser = captured.parser
        parser.parse_args = types.MethodType(real_parse_args, parser)
        return parser
    raise AssertionError("train.gpt2.finetune.main() never called parse_args()")


def _finetune_config(monkeypatch) -> ft.FinetuneConfig:
    """Run the real ``main()`` up to the training call and return its config."""
    captured: dict = {}

    def _recorder(config):
        captured["config"] = config
        return None, None

    monkeypatch.setattr(ft, "setup_gpu", lambda **kwargs: None)
    monkeypatch.setattr(ft, "finetune_gpt2", _recorder)
    ft.main()
    assert "config" in captured, "main() did not reach finetune_gpt2"
    return captured["config"]


# ---------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------

#: The five flags `pretrain_so` adds on top of the shared CLM surface. Note the
#: two inverted booleans: `--no-so-matrix-scaling` sets `so_matrix_scaling` to
#: False and `--so-include-embeddings` sets `so_skip_embeddings` to False, so a
#: forwarding line that drops the `not` is a silent polarity flip.
SO_ROWS: Tuple[Row, ...] = (
    Row(("--so-lambda",), ("--so-lambda", "0.25"), "so_lambda", 0.25),
    Row(("--so-l1",), ("--so-l1", "0.5"), "so_l1", 0.5),
    Row(("--so-l2",), ("--so-l2", "0.75"), "so_l2", 0.75),
    Row(
        ("--no-so-matrix-scaling",),
        ("--no-so-matrix-scaling",),
        "so_matrix_scaling",
        False,
    ),
    Row(
        ("--so-include-embeddings",),
        ("--so-include-embeddings",),
        "so_skip_embeddings",
        False,
    ),
)

_FINETUNE_HF_ROWS: Tuple[Row, ...] = (
    Row(("--gpu",), ("--gpu", "1"), namespace_dest="gpu", expected=1),
    Row(("--pretrained",), (), "pretrained_path", "/probe/pre.keras"),
    Row(("--epochs",), ("--epochs", "4"), "num_epochs", 4),
    Row(("--batch-size",), ("--batch-size", "5"), "batch_size", 5),
    Row(("--max-seq-length",), ("--max-seq-length", "96"), "max_seq_length", 96),
    Row(("--learning-rate",), ("--learning-rate", "2.5e-6"), "learning_rate", 2.5e-6),
    Row(("--freeze-embeddings",), ("--freeze-embeddings",), "freeze_embeddings", True),
    Row(("--freeze-n-layers",), ("--freeze-n-layers", "6"), "freeze_n_layers", 6),
    Row(("--hf-dataset",), (), "hf_dataset_path", "wikitext-probe"),
    Row(("--hf-config",), ("--hf-config", "probe-v1"), "hf_dataset_name", "probe-v1"),
    Row(
        ("--hf-cache-dir",),
        ("--hf-cache-dir", "/probe/hf-cache"),
        "hf_cache_dir",
        "/probe/hf-cache",
    ),
    Row(("--save-dir",), ("--save-dir", "/probe/save"), "save_dir", "/probe/save"),
    Row(("--steps-per-epoch",), ("--steps-per-epoch", "33"), "steps_per_epoch", 33),
    Row(("--seed",), ("--seed", "4321"), "seed", 4321),
)

_FINETUNE_TEXT_ROWS: Tuple[Row, ...] = (
    Row(("--text-dir",), (), "text_dir", "/probe/texts"),
    Row(("--text-glob",), ("--text-glob", "*.md"), "text_glob", "*.md"),
)

GPT2_PRETRAIN = Contract(
    name="train.gpt2.pretrain",
    build_parser=lambda monkeypatch: pt._build_parser(),
    build_config=lambda monkeypatch: pt._config_from_args(
        pt._build_parser().parse_args()
    ),
    modes=(Mode("plain", (), CLM_PRETRAIN_ROWS),),
)

GPT2_PRETRAIN_SO = Contract(
    name="train.gpt2.pretrain_so",
    build_parser=lambda monkeypatch: so._build_so_parser(),
    build_config=lambda monkeypatch: so._so_config_from_args(
        so._build_so_parser().parse_args()
    ),
    modes=(Mode("plain", (), CLM_PRETRAIN_ROWS + SO_ROWS),),
)

GPT2_FINETUNE = Contract(
    name="train.gpt2.finetune",
    build_parser=_finetune_parser,
    build_config=_finetune_config,
    modes=(
        Mode(
            "hf",
            ("--pretrained", "/probe/pre.keras", "--hf-dataset", "wikitext-probe"),
            _FINETUNE_HF_ROWS,
            baseline_argv=(
                "--pretrained", "/probe/other.keras",
                "--hf-dataset", "other-corpus",
            ),
        ),
        Mode(
            "text",
            ("--pretrained", "/probe/pre.keras", "--text-dir", "/probe/texts"),
            _FINETUNE_TEXT_ROWS,
            baseline_argv=(
                "--pretrained", "/probe/pre.keras",
                "--text-dir", "/probe/other-texts",
            ),
        ),
    ),
)

CONTRACTS = (GPT2_PRETRAIN, GPT2_PRETRAIN_SO, GPT2_FINETUNE)

_CASES, _IDS = cases(CONTRACTS)


# ---------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------

@pytest.mark.parametrize("contract,mode,row", _CASES, ids=_IDS)
def test_flag_reaches_its_config_field(monkeypatch, contract, mode, row) -> None:
    assert_row_reaches_config(monkeypatch, contract, mode, row)


@pytest.mark.parametrize("contract,mode,row", _CASES, ids=_IDS)
def test_probe_value_differs_from_the_default(monkeypatch, contract, mode, row) -> None:
    assert_row_value_is_not_the_default(monkeypatch, contract, mode, row)


@pytest.mark.parametrize(
    "contract", CONTRACTS, ids=[c.name for c in CONTRACTS]
)
def test_every_declared_flag_has_a_contract_row(monkeypatch, contract) -> None:
    """The table is read against the REAL parser, so a new flag fails here."""
    declared = declared_option_strings(contract.build_parser(monkeypatch))
    covered = contract.covered_flags
    assert declared == covered, (
        f"{contract.name}: flags with no contract row {sorted(declared - covered)}; "
        f"rows for flags the parser does not declare {sorted(covered - declared)}"
    )


# ---------------------------------------------------------------------
# finetune's data-source branch: not a flag->field hop, a flag->BRANCH hop
# ---------------------------------------------------------------------

class TestFinetuneDataSourceBranch:
    """``main()`` picks ``data_source`` from WHICH flag was given, not a value.

    No row can express that, so it is asserted directly. The default is
    ``"huggingface"``, which makes the text-files arm the load-bearing one.
    """

    def test_hf_dataset_selects_the_huggingface_source(self, monkeypatch) -> None:
        monkeypatch.setattr(
            sys, "argv",
            ["train.gpt2.finetune", "--pretrained", "/probe/pre.keras",
             "--hf-dataset", "wikitext-probe"],
        )
        assert _finetune_config(monkeypatch).data_source == "huggingface"

    def test_text_dir_selects_the_text_files_source(self, monkeypatch) -> None:
        monkeypatch.setattr(
            sys, "argv",
            ["train.gpt2.finetune", "--pretrained", "/probe/pre.keras",
             "--text-dir", "/probe/texts"],
        )
        assert _finetune_config(monkeypatch).data_source == "text_files"

    def test_the_data_source_group_is_required(self, monkeypatch) -> None:
        """Neither arm given must exit non-zero, not start a run with defaults."""
        monkeypatch.setattr(
            sys, "argv",
            ["train.gpt2.finetune", "--pretrained", "/probe/pre.keras"],
        )
        with pytest.raises(SystemExit) as excinfo:
            _finetune_config(monkeypatch)
        assert excinfo.value.code != 0


# ---------------------------------------------------------------------
# The SO config is built on top of the base one, through **vars()
# ---------------------------------------------------------------------

def test_so_config_carries_every_base_field(monkeypatch) -> None:
    """Every base-config VALUE must survive the ``**vars(base)`` splice.

    ``_so_config_from_args`` builds the base config with ``_config_from_args``
    and splices it in with ``**vars(base)``. Drop that splice and every base
    field silently reverts to its declared default: the run starts, ``--help``
    still advertises all 27 CLM flags, and none of them do anything.

    VACUOUS FORM -- DO NOT REINTRODUCE (this plan's third probe defect,
    plan-2026-08-13T091555-230c101d review item 1). The first version of this
    test compared FIELD NAMES::

        base_fields = {f.name for f in dataclasses.fields(pt.TrainingConfig())}
        so_fields = {f.name for f in dataclasses.fields(so_config)}
        assert base_fields <= so_fields
        assert so_fields - base_fields == {the 5 SO fields}

    MEASURED: both assertions still PASS with ``**vars(base)`` deleted outright,
    because ``SOTrainingConfig(TrainingConfig)`` inherits every base field by
    DECLARATION -- ``dataclasses.fields()`` can never lose one, whatever the
    call site does. The contract is about values flowing, not names existing.

    So this drives the REAL parser with a NON-DEFAULT value for every base-owned
    flag (the same ``CLM_PRETRAIN_ROWS`` table the per-flag tests use), builds
    the base config and the SO config from the SAME namespace, and asserts they
    agree field by field. The non-default count is asserted too, so the
    comparison cannot degenerate into defaults == defaults.
    """
    import dataclasses

    argv = [frag for row in CLM_PRETRAIN_ROWS for frag in row.argv]
    monkeypatch.setattr(sys, "argv", ["train.gpt2.pretrain_so", *argv])
    args = so._build_so_parser().parse_args()

    base = pt._config_from_args(args)
    so_config = so._so_config_from_args(args)

    defaults = pt.TrainingConfig()
    base_field_names = [f.name for f in dataclasses.fields(base)]
    overridden = [
        name for name in base_field_names
        if getattr(base, name) != getattr(defaults, name)
    ]
    assert len(overridden) >= 20, (
        "anti-vacuity: the probe argv only moved "
        f"{len(overridden)} base fields off their defaults ({overridden}); a "
        "values comparison over defaults could not detect a dropped splice"
    )

    dropped = {
        name: (getattr(base, name), getattr(so_config, name))
        for name in base_field_names
        if getattr(so_config, name) != getattr(base, name)
    }
    assert not dropped, (
        "_so_config_from_args did NOT carry these base config values into "
        f"SOTrainingConfig (field: base -> so): {dropped}. The `**vars(base)` "
        "splice is broken, so the CLM flags are advertised and inert."
    )

    so_fields = {f.name for f in dataclasses.fields(so_config)}
    assert so_fields - set(base_field_names) == {
        "so_lambda", "so_l1", "so_l2", "so_matrix_scaling", "so_skip_embeddings",
    }
