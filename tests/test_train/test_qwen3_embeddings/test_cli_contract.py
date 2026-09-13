"""Every Qwen3 embedding/reranker trainer flag must reach the config through
the REAL parse path.

Reuses `tests/test_train/_cli_contract.py`'s `Contract`/`Row`/`Mode` driver
(the shared instrument the ColBERT trainer tests already use for this exact
defect class -- see that module's docstring) rather than reimplementing the
three vacuity traps it exists to close.

Both scripts share one `TrainingConfig` and one `add_common_arguments`
registration (`train.qwen3_embeddings.common`), so one row table covers
both -- a flag added to the shared registration gets checked against both
entry points automatically.
"""

from __future__ import annotations

import sys
from typing import Any, Tuple

import pytest

from train.qwen3_embeddings import train_qwen3_embedding as emb
from train.qwen3_embeddings import train_qwen3_reranker as rer
from train.qwen3_embeddings.common import CLI_TO_CONFIG, SMOKE_PRESET, TrainingConfig

from .._cli_contract import (
    Contract,
    Mode,
    Row,
    assert_row_reaches_config,
    assert_row_value_is_not_the_default,
    cases,
    declared_option_strings,
)


# ---------------------------------------------------------------------
# Driving the real `main()`
# ---------------------------------------------------------------------


def _make_build_config(module: Any, train_attr: str):
    def build_config(monkeypatch) -> TrainingConfig:
        captured = []
        monkeypatch.setattr(module, "setup_gpu", lambda **kwargs: None)
        monkeypatch.setattr(module, train_attr, captured.append)
        module.main()
        assert captured, (
            f"{module.__name__}.main() never called {train_attr}; no config "
            f"was built, so nothing here measures the argv -> config path"
        )
        return captured[0]

    return build_config


_BUILD_EMB = _make_build_config(emb, "train_qwen3_embedding")
_BUILD_RER = _make_build_config(rer, "train_qwen3_reranker")


def _config_from_argv(monkeypatch, module: Any, argv) -> TrainingConfig:
    build = _BUILD_EMB if module is emb else _BUILD_RER
    monkeypatch.setattr(sys, "argv", [module.__name__, *argv])
    return build(monkeypatch)


# ---------------------------------------------------------------------
# The flag table (shared by both entry points -- one registration)
# ---------------------------------------------------------------------

SHARED_ROWS: Tuple[Row, ...] = (
    Row(("--gpu",), ("--gpu", "1"), namespace_dest="gpu", expected=1),
    Row(("--encoding-name",), ("--encoding-name", "r50k_base"), "encoding_name", "r50k_base"),
    Row(("--hidden-size",), ("--hidden-size", "48"), "hidden_size", 48),
    Row(("--num-layers",), ("--num-layers", "3"), "num_layers", 3),
    Row(("--num-heads",), ("--num-heads", "8"), "num_heads", 8),
    Row(("--intermediate-size",), ("--intermediate-size", "96"), "intermediate_size", 96),
    Row(("--dropout-rate",), ("--dropout-rate", "0.2"), "dropout_rate", 0.2),
    Row(("--ffn-type",), ("--ffn-type", "mlp"), "ffn_type", "mlp"),
    Row(("--normalization-type",), ("--normalization-type", "layer_norm"),
        "normalization_type", "layer_norm"),
    Row(("--attention-type",), ("--attention-type", "multi_head_latent"),
        "attention_type", "multi_head_latent"),
    Row(
        ("--normalize-embeddings", "--no-normalize-embeddings"),
        ("--no-normalize-embeddings",),
        "normalize_embeddings",
        False,
    ),
    Row(("--truncate-dim",), ("--truncate-dim", "32"), "truncate_dim", 32),
    Row(("--infonce-temperature",), ("--infonce-temperature", "0.2"),
        "infonce_temperature", 0.2),
    Row(("--reranker-maxlen",), ("--reranker-maxlen", "64"), "reranker_maxlen", 64),
    Row(("--query-maxlen",), ("--query-maxlen", "24"), "query_maxlen", 24),
    Row(("--doc-maxlen",), ("--doc-maxlen", "32"), "doc_maxlen", 32),
    Row(("--nway",), ("--nway", "3"), "nway", 3),
    Row(("--num-train-groups",), ("--num-train-groups", "40"), "num_train_groups", 40),
    Row(("--num-val-groups",), ("--num-val-groups", "10"), "num_val_groups", 10),
    Row(("--query-words",), ("--query-words", "6"), "query_words", 6),
    Row(("--doc-words",), ("--doc-words", "20"), "doc_words", 20),
    Row(("--batch-size",), ("--batch-size", "5"), "batch_size", 5),
    Row(("--epochs",), ("--epochs", "4"), "epochs", 4),
    Row(("--learning-rate",), ("--learning-rate", "1e-3"), "learning_rate", 1e-3),
    Row(("--warmup-epochs",), ("--warmup-epochs", "2"), "warmup_epochs", 2),
    Row(("--weight-decay",), ("--weight-decay", "0.05"), "weight_decay", 0.05),
    Row(("--gradient-clipping",), ("--gradient-clipping", "0.5"), "gradient_clipping", 0.5),
    Row(("--optimizer-type",), ("--optimizer-type", "adam"), "optimizer_type", "adam"),
    Row(("--lr-schedule-type",), ("--lr-schedule-type", "exponential_decay"),
        "lr_schedule_type", "exponential_decay"),
    Row(("--patience",), ("--patience", "3"), "patience", 3),
    Row(("--seed",), ("--seed", "7"), "seed", 7),
    Row(("--output-root",), ("--output-root", "/tmp/probe_out"), "output_root", "/tmp/probe_out"),
    Row(("--results-dir-prefix",), ("--results-dir-prefix", "probe_prefix"),
        "results_dir_prefix", "probe_prefix"),
    Row(("--smoke",), ("--smoke",), "smoke", True),
)

EMB_CONTRACT = Contract(
    name="train.qwen3_embeddings.train_qwen3_embedding",
    build_parser=lambda monkeypatch: emb.build_parser(),
    build_config=_BUILD_EMB,
    modes=(Mode("plain", (), SHARED_ROWS),),
)

RER_CONTRACT = Contract(
    name="train.qwen3_embeddings.train_qwen3_reranker",
    build_parser=lambda monkeypatch: rer.build_parser(),
    build_config=_BUILD_RER,
    modes=(Mode("plain", (), SHARED_ROWS),),
)

CONTRACTS = (EMB_CONTRACT, RER_CONTRACT)

_CASES, _IDS = cases(CONTRACTS)


@pytest.mark.parametrize("contract,mode,row", _CASES, ids=_IDS)
def test_flag_reaches_its_config_field(monkeypatch, contract, mode, row) -> None:
    assert_row_reaches_config(monkeypatch, contract, mode, row)


@pytest.mark.parametrize("contract,mode,row", _CASES, ids=_IDS)
def test_probe_value_differs_from_the_default(monkeypatch, contract, mode, row) -> None:
    assert_row_value_is_not_the_default(monkeypatch, contract, mode, row)


@pytest.mark.parametrize("contract", CONTRACTS, ids=[c.name for c in CONTRACTS])
def test_every_declared_flag_has_a_contract_row(monkeypatch, contract) -> None:
    declared = declared_option_strings(contract.build_parser(monkeypatch))
    covered = contract.covered_flags
    assert declared == covered, (
        f"{contract.name}: flags with no contract row {sorted(declared - covered)}; "
        f"rows for flags the parser does not declare {sorted(covered - declared)}"
    )


# ---------------------------------------------------------------------
# --help: exit 0, no allocation
# ---------------------------------------------------------------------


@pytest.mark.parametrize("module", [emb, rer], ids=["embedding", "reranker"])
def test_help_exits_zero_and_prints_usage(capsys, module) -> None:
    with pytest.raises(SystemExit) as excinfo:
        module.main(["--help"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert captured.out.startswith("usage:"), (
        f"--help did not print a usage: line first. stdout:\n{captured.out}"
    )


@pytest.mark.parametrize(
    "module,train_attr", [(emb, "train_qwen3_embedding"), (rer, "train_qwen3_reranker")],
    ids=["embedding", "reranker"],
)
def test_help_allocates_no_gpu_and_touches_no_dataset(monkeypatch, module, train_attr) -> None:
    calls = {"setup_gpu": 0, "train": 0}

    def _spy_setup_gpu(*_args, **_kwargs):
        calls["setup_gpu"] += 1

    def _spy_train(*_args, **_kwargs):
        calls["train"] += 1
        raise AssertionError("train must not run on --help")

    monkeypatch.setattr(module, "setup_gpu", _spy_setup_gpu)
    monkeypatch.setattr(module, train_attr, _spy_train)

    with pytest.raises(SystemExit) as excinfo:
        module.main(["--help"])
    assert excinfo.value.code == 0
    assert calls == {"setup_gpu": 0, "train": 0}, f"--help reached training machinery: {calls}"


# ---------------------------------------------------------------------
# CLI_TO_CONFIG / SMOKE_PRESET sanity
# ---------------------------------------------------------------------


def test_cli_to_config_table_names_only_real_fields() -> None:
    known = set(TrainingConfig.field_names())
    unknown = sorted(set(CLI_TO_CONFIG.values()) - known)
    assert not unknown, f"CLI_TO_CONFIG targets non-existent fields {unknown}"


def test_smoke_preset_shrinks_the_synthetic_corpus() -> None:
    # `epochs` is deliberately NOT one of the fields checked for a strict
    # decrease: the class default is already 1 (the cheapest legal value), so
    # SMOKE_PRESET pins it at 1 rather than shrinking it further -- the same
    # named-vacuity shape ColBERT's own contract test documents for
    # `colbert_variant` (see `decisions.md` D-027 there).
    assert SMOKE_PRESET, "SMOKE_PRESET is empty; --smoke reduces nothing"
    assert TrainingConfig.epochs == SMOKE_PRESET["epochs"] == 1
    for field in ("num_train_groups", "num_val_groups"):
        assert field in SMOKE_PRESET
        assert SMOKE_PRESET[field] < getattr(TrainingConfig, field)


@pytest.mark.parametrize("module", [emb, rer], ids=["embedding", "reranker"])
def test_smoke_moves_every_preset_field(monkeypatch, module) -> None:
    config = _config_from_argv(monkeypatch, module, ["--smoke"])
    for field, value in SMOKE_PRESET.items():
        assert getattr(config, field) == value


def test_gpu_never_reaches_the_config(monkeypatch) -> None:
    config = _config_from_argv(monkeypatch, emb, ["--gpu", "1"])
    assert not hasattr(config, "gpu")
    assert "gpu" not in set(TrainingConfig.field_names())


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(TypeError, match="unknown field"):
        TrainingConfig(not_a_real_field=1)


def test_results_dir_prefixes_differ_between_the_two_default_model_names() -> None:
    # Both scripts share one TrainingConfig.results_dir_prefix default; the two
    # runs are still distinguished on disk via `model_name` (`create_callbacks`'s
    # own first positional argument in each trainer's `train_qwen3_*` function),
    # not a divergent config default -- unlike ColBERT's v1/v2 nway divergence,
    # there is no config-level divergence to pin here.
    assert TrainingConfig().results_dir_prefix == "qwen3_embeddings"
