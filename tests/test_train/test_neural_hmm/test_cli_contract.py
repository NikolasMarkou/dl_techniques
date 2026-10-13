"""CLI contract guard for ``src/train/neural_hmm/train_neural_hmm.py``.

The defect class: a flag is declared, ``--help`` advertises it, and nothing between
argparse and the point of use reads it. Two halves, because parsing alone cannot see
the second hop:

1. **flag -> Namespace**, through the shared driver ``tests/test_train/_cli_contract.py``
   (which documents the three traps it designs out: a probe equal to the default, an
   incomplete row table, two rows sharing a value). The trainer has no config
   dataclass: ``parse_arguments()`` returns the ``argparse.Namespace`` and every
   downstream function reads ``args.<field>``, the hkan shape. So ``build_config`` is
   ``parse_arguments`` and every row but ``--gpu`` is a ``field`` row.
2. **Namespace -> callee** (``wired``): two real tiny runs of ``main()`` with every flag
   off its default, recording what the data sampler, the model factory, the optimizer
   builder, ``fit`` and the early-stopping callback actually received. A flag that is
   parsed and then dropped (``--stay-prob`` never handed to the sampler, say) passes
   half 1 and fails here.

``--gpu`` is consumed once in ``main()`` by ``setup_gpu(gpu_id=args.gpu)``;
``test_gpu_flag_reaches_setup_gpu`` drives ``main()`` against a sentinel. Every run is
pointed at ``tmp_path``, never at repo-root ``results/``.
"""

from __future__ import annotations

from typing import Dict, Tuple

import keras
import pytest

import train.neural_hmm.train_neural_hmm as train_nhmm

from .._cli_contract import (  # noqa: TID252 -- shared driver, one level up
    Contract,
    Mode,
    Row,
    assert_row_reaches_config,
    assert_row_value_is_not_the_default,
    cases,
    declared_option_strings,
)

#: One row per declared flag; every probe value is unique across the table and
#: differs from the flag's default (each row is driven alone against the defaults,
#: so each probe also passes the trainer's own range checks on its own).
NHMM_ROWS: Tuple[Row, ...] = (
    Row(("--variant",), ("--variant", "io-hmm"), "variant", "io-hmm"),
    Row(("--obs-type",), ("--obs-type", "categorical"), "obs_type", "categorical"),
    Row(("--n-states",), ("--n-states", "5"), "n_states", 5),
    Row(("--obs-dim",), ("--obs-dim", "7"), "obs_dim", 7),
    Row(("--vocab-size",), ("--vocab-size", "13"), "vocab_size", 13),
    Row(("--encoder",), ("--encoder", "gru"), "encoder", "gru"),
    Row(("--hidden-dim",), ("--hidden-dim", "24"), "hidden_dim", 24),
    Row(("--seq-len",), ("--seq-len", "41"), "seq_len", 41),
    Row(("--min-seq-len",), ("--min-seq-len", "17"), "min_seq_len", 17),
    Row(("--num-train",), ("--num-train", "333"), "num_train", 333),
    Row(("--num-val",), ("--num-val", "44"), "num_val", 44),
    Row(("--num-test",), ("--num-test", "55"), "num_test", 55),
    Row(("--separation",), ("--separation", "2.75"), "separation", 2.75),
    Row(("--stay-prob",), ("--stay-prob", "0.85"), "stay_prob", 0.85),
    Row(("--epochs",), ("--epochs", "9"), "epochs", 9),
    Row(("--batch-size",), ("--batch-size", "21"), "batch_size", 21),
    Row(("--learning-rate",), ("--learning-rate", "0.0125"), "learning_rate", 0.0125),
    Row(("--gradient-clip-norm",), ("--gradient-clip-norm", "0.625"), "gradient_clip_norm", 0.625),
    Row(("--patience",), ("--patience", "6"), "patience", 6),
    Row(("--seed",), ("--seed", "1234"), "seed", 1234),
    Row(("--output-dir",), ("--output-dir", "/probe/nhmm-output-dir"), "output_dir",
        "/probe/nhmm-output-dir"),
    Row(("--experiment-name",), ("--experiment-name", "probe-experiment-name"),
        "experiment_name", "probe-experiment-name"),
    # Not a `field` row: consumed once in main() by setup_gpu(gpu_id=args.gpu).
    # This row proves parsing only; test_gpu_flag_reaches_setup_gpu proves the call.
    Row(("--gpu",), ("--gpu", "3"), namespace_dest="gpu", expected=3),
)

NHMM_TRAINER = Contract(
    name="train.neural_hmm.train_neural_hmm",
    build_parser=lambda monkeypatch: train_nhmm._build_parser(),
    build_config=lambda monkeypatch: train_nhmm.parse_arguments(),
    modes=(Mode(id="plain", required_argv=(), rows=NHMM_ROWS),),
)

CONTRACTS: Tuple[Contract, ...] = (NHMM_TRAINER,)

_CASES, _IDS = cases(CONTRACTS)


# ---------------------------------------------------------------------
# flag -> config (argparse Namespace)
# ---------------------------------------------------------------------


@pytest.mark.parametrize("contract,mode,row", _CASES, ids=_IDS)
def test_flag_reaches_its_config_field(monkeypatch, contract, mode, row) -> None:
    """A flag that parses and arrives nowhere is the silent no-op this guards."""
    assert_row_reaches_config(monkeypatch, contract, mode, row)


@pytest.mark.parametrize("contract,mode,row", _CASES, ids=_IDS)
def test_probe_value_differs_from_the_default(monkeypatch, contract, mode, row) -> None:
    """Trap 1: without this, every row above could pass vacuously."""
    assert_row_value_is_not_the_default(monkeypatch, contract, mode, row)


def test_probe_values_are_mutually_distinct() -> None:
    """Trap 3: two rows sharing a value would hide a cross-wired forward."""
    values = [repr(row.expected) for row in NHMM_ROWS]
    assert len(values) == len(set(values)), sorted(values)


@pytest.mark.parametrize("contract", CONTRACTS, ids=[c.name for c in CONTRACTS])
def test_every_declared_flag_has_a_contract_row(monkeypatch, contract) -> None:
    """A flag added without a row fails here. The count is the parser's, not pasted."""
    declared = declared_option_strings(contract.build_parser(monkeypatch))
    covered = contract.covered_flags
    assert declared == covered, (
        f"{contract.name}: flags with no contract row "
        f"{sorted(declared - covered)}; rows for flags the parser no longer "
        f"declares {sorted(covered - declared)}"
    )
    assert len(NHMM_ROWS) == len(declared), "a flag has two rows"


# ---------------------------------------------------------------------
# --gpu: forwarding needs main(), not just the parser
# ---------------------------------------------------------------------


class _SetupGpuProbe(Exception):
    """Raised by the sentinel so ``main()`` stops at its second statement."""


def test_gpu_flag_reaches_setup_gpu(monkeypatch, tmp_path) -> None:
    """``--gpu 3`` must reach ``setup_gpu(gpu_id=3)`` inside ``main()``, before any write."""
    calls = []

    def _sentinel(*args, **kwargs):
        calls.append((args, kwargs))
        raise _SetupGpuProbe

    monkeypatch.setattr(train_nhmm, "setup_gpu", _sentinel)
    argv = ["--gpu", "3", "--output-dir", str(tmp_path), "--experiment-name", "gpu_probe",
            "--num-train", "8", "--num-val", "4", "--num-test", "4", "--seq-len", "5",
            "--min-seq-len", "2", "--epochs", "1"]
    try:
        train_nhmm.main(argv)
    except _SetupGpuProbe:
        pass

    assert calls == [((), {"gpu_id": 3})], (
        f"--gpu 3 did not reach setup_gpu(gpu_id=3): recorded calls={calls!r}"
    )
    assert not (tmp_path / "gpu_probe").exists(), "setup_gpu ran after the run directory was made"


# ---------------------------------------------------------------------
# Namespace -> callee: two real runs with every flag off its default
# ---------------------------------------------------------------------

#: Every data, model and optimization flag off its default. A gaussian run with an
#: encoder the override can replace (io-hmm: conv -> mlp), and a categorical run for
#: the two flags only a categorical run reads (--obs-type, --vocab-size).
WIRED_RUNS: Dict[str, Tuple[str, ...]] = {
    "gaussian": (
        "--variant", "io-hmm", "--encoder", "mlp", "--n-states", "3", "--obs-dim", "3",
        "--hidden-dim", "12", "--seq-len", "7", "--min-seq-len", "3", "--num-train", "13",
        "--num-val", "5", "--num-test", "6", "--separation", "2.5", "--stay-prob", "0.75",
        "--epochs", "2", "--batch-size", "7", "--learning-rate", "0.004",
        "--gradient-clip-norm", "0.5", "--patience", "4", "--seed", "7",
    ),
    "categorical": (
        "--variant", "autoregressive-neural-hmm", "--encoder", "conv", "--obs-type", "categorical",
        "--n-states", "2", "--vocab-size", "11", "--hidden-dim", "8", "--seq-len", "6",
        "--min-seq-len", "2", "--num-train", "9", "--num-val", "3", "--num-test", "4",
        "--epochs", "1", "--batch-size", "5",
    ),
}


@pytest.fixture(scope="module")
def wired(tmp_path_factory) -> Dict[str, Dict]:
    """What each callee received in each run of :data:`WIRED_RUNS`.

    Every wrapper records its arguments and calls the real thing, so each run is a
    complete run whose files are written into a temporary directory.
    """
    base = tmp_path_factory.mktemp("nhmm_wired")
    real_splits, real_create = train_nhmm.sample_splits, train_nhmm.create_neural_hmm
    real_optimizer, real_seeds, real_fit = (
        train_nhmm.optimizer_builder, train_nhmm.set_seeds, keras.Model.fit)
    seen: Dict[str, Dict] = {}
    for name, argv in WIRED_RUNS.items():
        record: Dict[str, list] = {key: [] for key in ("splits", "create", "optimizer", "seeds", "fit")}

        def _splits(sizes, seed, **kwargs):
            record["splits"].append({"sizes": dict(sizes), "seed": seed, **kwargs})
            return real_splits(sizes, seed, **kwargs)

        def _create(variant, **kwargs):
            record["create"].append({"variant": variant, **kwargs})
            return real_create(variant, **kwargs)

        def _optimizer(config, learning_rate):
            record["optimizer"].append((dict(config), learning_rate))
            return real_optimizer(config, learning_rate)

        def _seeds(seed):
            record["seeds"].append(seed)
            return real_seeds(seed)

        def _fit(self, x=None, y=None, **kwargs):
            stopping = [c for c in kwargs["callbacks"] if isinstance(c, keras.callbacks.EarlyStopping)]
            record["fit"].append({
                "rows": len(x["obs"]), "val_rows": len(kwargs["validation_data"][0]["obs"]),
                "epochs": kwargs["epochs"], "batch_size": kwargs["batch_size"],
                "patience": [c.patience for c in stopping],
                "learning_rate": float(keras.ops.convert_to_numpy(self.optimizer.learning_rate)),
                "global_clipnorm": self.optimizer.global_clipnorm,
            })
            return real_fit(self, x, y, **kwargs)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(train_nhmm, "sample_splits", _splits)
            patch.setattr(train_nhmm, "create_neural_hmm", _create)
            patch.setattr(train_nhmm, "optimizer_builder", _optimizer)
            patch.setattr(train_nhmm, "set_seeds", _seeds)
            patch.setattr(keras.Model, "fit", _fit)
            assert train_nhmm.main([*argv, "--output-dir", str(base), "--experiment-name", name]) == 0
        record["run_dir"] = base / name
        seen[name] = record
    return seen


def _only(calls: list):
    assert len(calls) == 1, f"expected one call, got {len(calls)}: {calls!r}"
    return calls[0]


#: check -> (what the callee received, what the CLI asked for).
WIRED_CHECKS = {
    "--variant": lambda s: (_only(s["gaussian"]["create"]).get("variant", "<not passed>"), "io-hmm"),
    "--encoder": lambda s: (_only(s["gaussian"]["create"]).get("encoder_type", "<not passed>"), "mlp"),
    "--n-states (data)": lambda s: (_only(s["gaussian"]["splits"]).get("n_states", "<not passed>"), 3),
    "--n-states (model)": lambda s: (_only(s["gaussian"]["create"]).get("n_states", "<not passed>"), 3),
    "--obs-dim (data)": lambda s: (_only(s["gaussian"]["splits"]).get("obs_dim", "<not passed>"), 3),
    "--obs-dim (model)": lambda s: (_only(s["gaussian"]["create"]).get("obs_dim", "<not passed>"), 3),
    "--hidden-dim": lambda s: (_only(s["gaussian"]["create"]).get("hidden_dim", "<not passed>"), 12),
    "--seq-len (data)": lambda s: (_only(s["gaussian"]["splits"]).get("seq_len", "<not passed>"), 7),
    "--seq-len (model position table)": lambda s: (_only(s["gaussian"]["create"]).get("max_seq_len", "<not passed>"), 7),
    "--min-seq-len": lambda s: (_only(s["gaussian"]["splits"]).get("min_seq_len", "<not passed>"), 3),
    "--num-train/--num-val/--num-test": lambda s: (
        _only(s["gaussian"]["splits"]).get("sizes", "<not passed>"), {"train": 13, "val": 5, "test": 6}),
    "--num-train (fit rows)": lambda s: (_only(s["gaussian"]["fit"])["rows"], 13),
    "--num-val (validation rows)": lambda s: (_only(s["gaussian"]["fit"])["val_rows"], 5),
    "--separation": lambda s: (_only(s["gaussian"]["splits"]).get("separation", "<not passed>"), 2.5),
    "--stay-prob": lambda s: (_only(s["gaussian"]["splits"]).get("stay_prob", "<not passed>"), 0.75),
    "--epochs": lambda s: (_only(s["gaussian"]["fit"])["epochs"], 2),
    "--batch-size": lambda s: (_only(s["gaussian"]["fit"])["batch_size"], 7),
    "--learning-rate (builder)": lambda s: (_only(s["gaussian"]["optimizer"])[1], 0.004),
    "--learning-rate (the optimizer fit uses)": lambda s: (
        _only(s["gaussian"]["fit"])["learning_rate"], pytest.approx(0.004)),
    "--gradient-clip-norm (builder)": lambda s: (
        _only(s["gaussian"]["optimizer"])[0]["gradient_clipping_by_norm"], 0.5),
    "--gradient-clip-norm (the optimizer fit uses)": lambda s: (
        _only(s["gaussian"]["fit"])["global_clipnorm"], 0.5),
    "--patience": lambda s: (_only(s["gaussian"]["fit"])["patience"], [4]),
    "--seed (data)": lambda s: (_only(s["gaussian"]["splits"]).get("seed", "<not passed>"), 7),
    "--seed (model init)": lambda s: (s["gaussian"]["seeds"], [7]),
    "--obs-type (data)": lambda s: (_only(s["categorical"]["splits"]).get("obs_type", "<not passed>"), "categorical"),
    "--obs-type (model)": lambda s: (_only(s["categorical"]["create"]).get("obs_type", "<not passed>"), "categorical"),
    "--vocab-size (data)": lambda s: (_only(s["categorical"]["splits"]).get("vocab_size", "<not passed>"), 11),
    "--vocab-size (model)": lambda s: (_only(s["categorical"]["create"]).get("vocab_size", "<not passed>"), 11),
    "--output-dir/--experiment-name": lambda s: (
        all((s[name]["run_dir"] / "results_summary.json").is_file() for name in WIRED_RUNS), True),
}


@pytest.mark.parametrize("check", sorted(WIRED_CHECKS))
def test_a_non_default_flag_reaches_what_it_controls(wired, check) -> None:
    received, asked = WIRED_CHECKS[check](wired)
    assert received == asked, f"{check}: the callee received {received!r}, the CLI asked {asked!r}"


def test_the_wired_probe_values_all_differ_from_the_defaults() -> None:
    """Otherwise a flag that is never forwarded would still 'arrive'."""
    default = vars(train_nhmm.parse_arguments([]))
    for name, argv in WIRED_RUNS.items():
        probe = vars(train_nhmm.parse_arguments(list(argv)))
        set_here = {flag.lstrip("-").replace("-", "_") for flag in argv if flag.startswith("--")}
        same = sorted(key for key in set_here if probe[key] == default[key])
        assert same == [], f"{name}: flags probed at their default value: {same}"


def test_every_data_model_and_optimization_flag_is_wired() -> None:
    """Every flag of the parser is covered by a wired check, a --gpu test or the run dir."""
    flags = declared_option_strings(train_nhmm._build_parser()) - {"--gpu"}
    checked = {part.split(" ")[0] for check in WIRED_CHECKS for part in check.split("/")}
    checked = {flag if flag.startswith("--") else f"--{flag}" for flag in checked}
    assert flags <= checked, f"flags with no wired check: {sorted(flags - checked)}"
