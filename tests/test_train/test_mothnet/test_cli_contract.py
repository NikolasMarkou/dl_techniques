r"""CLI contract guard for ``src/train/mothnet/train_mothnet.py``.

The defect class this exists to catch
-------------------------------------
A trainer declares a CLI flag, ``--help`` advertises it, and something between
argparse and the point of use forgets to read it. The flag then silently does
nothing: no error, no warning, and the user believes they overrode a default
that never moved. This is the SAME defect class the closed iteration-1 plan's
own adversarial review found for real here: ``--gpu`` was declared, advertised
in ``--help``, and never reached ``setup_gpu`` (`plan-2026-09-18T045308-
c89cdf76/decisions.md` D-011, fixed in that plan, re-guarded here so it cannot
silently regress).

Shape of this trainer, and why the contract looks the way it does
-------------------------------------------------------------------
``train_mothnet.py`` has no separate ``TrainingConfig`` dataclass:
``parse_arguments()`` returns a plain ``argparse.Namespace`` and every
downstream function (``build_model``, ``load_mnist_data``, ``main``) reads
``args.<field>`` directly off it — the Namespace IS the config, the same shape
as ``train_kan.py`` (`plan-2026-09-18T060057-c1cfc3d3/decisions.md` D-003).
15 of the 16 declared flags are therefore checked as ordinary ``field`` rows:
the "config" `assert_row_reaches_config` reads from is just the parsed
Namespace itself, via ``build_config = train_mothnet.parse_arguments``.

``--gpu`` is the exception on TWO counts, both covered here:

1. Its declared row uses ``field=None, namespace_dest="gpu"`` — the
   `_cli_contract.py` idiom for a flag that reaches the argparse Namespace but
   is not persisted as a durable config field (it is consumed once, in
   ``main()``, and never re-read).
2. Unlike the 14 ``field`` rows, parsing alone cannot prove ``--gpu`` reaches
   ``setup_gpu`` — ``main()`` has a whole extra hop (``setup_gpu(gpu_id=
   args.gpu)``) that a namespace-level assertion never exercises. This is
   EXACTLY the D-011 defect class, so `test_gpu_flag_reaches_setup_gpu` below
   drives a real (fully sentinelled) ``main()`` and asserts the sentinel was
   called with ``gpu_id=<probe value>`` — the permanent guard against this
   defect recurring. ``setup_gpu`` is imported by name into this module's own
   namespace at module scope (``from train.common import (..., setup_gpu)``,
   called as bare ``setup_gpu(...)`` inside ``main()``), so it is patched
   directly on ``train_mothnet`` — unlike ``train.tabm.train_tabm``, which
   imports ``setup_gpu`` LAZILY inside ``main()`` (see
   ``tests/test_train/test_tabm/test_cli_contract.py``'s
   ``test_setup_gpu_is_wired_and_imported_lazily``) specifically because
   ``train.common``'s package ``__init__`` builds a ``tf.constant`` at import
   time and allocates a GPU; tabm needs that indirection to keep ``--help``
   cheap. mothnet already pays that module-scope import cost today (it also
   imports ``train.common.callbacks`` at module scope) — a pre-existing
   condition, not something this step introduces or fixes.

Why `_build_parser()` exists
-----------------------------
``Contract.build_parser`` needs the RAW, unparsed ``ArgumentParser`` (for
``declared_option_strings`` — the completeness check) separately from
``Contract.build_config`` (which needs the PARSED ``Namespace``).
``parse_arguments()`` used to construct-and-parse in one function. This test
suite's Step 2 extracted parser construction into a new
``_build_parser() -> argparse.ArgumentParser`` helper that ``parse_arguments()``
now calls before parsing — a small, non-behavior-changing structural
refactor made in service of testability (`plan-2026-09-18T060057-c1cfc3d3/
plan.md` Step 2), not a scope expansion. ``--help`` and every existing
behavior of ``parse_arguments()`` (including the ``--epochs < 1`` validation)
are unchanged; only WHERE the ``ArgumentParser`` gets built moved.

The disclosed gap: no `test_config_fields_are_live.py` coverage here
----------------------------------------------------------------------
``tests/test_train/test_config_fields_are_live.py``'s ``REGISTERED`` list is
keyed on ``(module_path, ClassName)`` for a config DATACLASS — it walks a
class's declared fields and asserts each one is actually READ somewhere. That
mechanism does not cleanly apply here: mothnet has no such class to register
(`args` IS the config, per the shape note above). Forcing a `REGISTERED` row
would mean wrapping `args` in an ad hoc dataclass solely to satisfy an
instrument that does not fit this trainer's shape — exactly the kind of
"wrapper cascade to satisfy an instrument" `references/complexity-control.md`
warns against (`decisions.md` D-003).

This CLI-contract test suite is therefore the PRIMARY, and currently ONLY,
mechanical guard for this trainer's flag-wiring correctness. That is a
disclosed, deliberate gap — not an oversight, and not a silent skip. A future
mothnet field read only inside a nested closure/lambda this suite doesn't
happen to probe could, in principle, go unread without either guard catching
it; `decisions.md` D-003 records that residual risk explicitly.

Three traps designed into `_cli_contract.py`'s machinery (see its own module
docstring for the full rationale), all exercised below:

1. A ROW MUST CARRY A NON-DEFAULT VALUE (`test_probe_value_differs_from_the_
   default` — every probe here is chosen to differ from that flag's own
   argparse default).
2. THE ROW TABLE MUST BE COMPLETE (`test_every_declared_flag_has_a_contract_
   row` — reads the flags off the REAL parser via `declared_option_strings`,
   so a 16th flag added without a matching row fails loudly).
3. VALUES MUST BE MUTUALLY DISTINCT (every probe value below is unique across
   all 16 rows, so a cross-wired forward moves the wrong field and is still
   caught).

Nothing here trains, allocates a GPU, or touches MNIST: every row drives only
`_build_parser()` / `parse_arguments()`, and the `--gpu`-to-`setup_gpu` test
sentinels `setup_gpu` to raise immediately on contact, so `main()` never
proceeds past its second statement.
"""

from __future__ import annotations

from typing import Tuple

import pytest

import train.mothnet.train_mothnet as train_mothnet

from .._cli_contract import (  # noqa: TID252 -- shared driver, one level up
    Contract,
    Mode,
    Row,
    assert_row_reaches_config,
    assert_row_value_is_not_the_default,
    cases,
    declared_option_strings,
)

#: One row per declared flag. `field` names are the argparse `dest`s
#: (`--mb-units` -> `mb_units`, etc.) -- since `args` IS the config, `field`
#: and `namespace_dest` would resolve identically for these 14; `field` is
#: used because the value IS meant to be read as a persistent config
#: attribute downstream (`build_model`/`load_mnist_data`), matching the
#: plan's own "field=<the matching argparse dest>" language. `--gpu` is the
#: one row that is deliberately NOT a persisted field (see module docstring).
MOTHNET_ROWS: Tuple[Row, ...] = (
    Row(("--mb-units",), ("--mb-units", "4096"), "mb_units", 4096),
    Row(("--mb-sparsity",), ("--mb-sparsity", "0.25"), "mb_sparsity", 0.25),
    Row(("--al-units",), ("--al-units", "128"), "al_units", 128),
    Row(
        ("--connection-sparsity",), ("--connection-sparsity", "0.35"),
        "connection_sparsity", 0.35,
    ),
    Row(
        ("--hebbian-learning-rate",), ("--hebbian-learning-rate", "0.05"),
        "hebbian_learning_rate", 0.05,
    ),
    Row(
        ("--inhibition-strength",), ("--inhibition-strength", "0.75"),
        "inhibition_strength", 0.75,
    ),
    Row(("--epochs",), ("--epochs", "5"), "epochs", 5),
    Row(("--batch-size",), ("--batch-size", "64"), "batch_size", 64),
    Row(("--viz-freq",), ("--viz-freq", "9"), "viz_freq", 9),
    Row(
        ("--output-dir",), ("--output-dir", "/tmp/probe-mothnet-output-dir"),
        "output_dir", "/tmp/probe-mothnet-output-dir",
    ),
    Row(
        ("--experiment-name",), ("--experiment-name", "probe-experiment-name"),
        "experiment_name", "probe-experiment-name",
    ),
    Row(("--seed",), ("--seed", "1234"), "seed", 1234),
    Row(
        ("--num-train-samples",), ("--num-train-samples", "777"),
        "num_train_samples", 777,
    ),
    Row(
        ("--num-val-samples",), ("--num-val-samples", "321"),
        "num_val_samples", 321,
    ),
    Row(
        ("--eval-batch-size",), ("--eval-batch-size", "111"),
        "eval_batch_size", 111,
    ),
    Row(
        ("--readout-weight-bound",), ("--readout-weight-bound", "0.5"),
        "readout_weight_bound", 0.5,
    ),
    # Deliberately NOT a `field` row: `--gpu` is consumed once, in main(), by
    # setup_gpu(gpu_id=args.gpu) -- it is never stored as a durable config
    # attribute anywhere downstream. Proves parsing only; see
    # `test_gpu_flag_reaches_setup_gpu` below for the call-level proof.
    Row(("--gpu",), ("--gpu", "3"), namespace_dest="gpu", expected=3),
)

MOTHNET_TRAINER = Contract(
    name="train.mothnet.train_mothnet",
    build_parser=lambda monkeypatch: train_mothnet._build_parser(),
    build_config=lambda monkeypatch: train_mothnet.parse_arguments(),
    modes=(Mode(id="plain", required_argv=(), rows=MOTHNET_ROWS),),
)

CONTRACTS: Tuple[Contract, ...] = (MOTHNET_TRAINER,)

_CASES, _IDS = cases(CONTRACTS)


# ---------------------------------------------------------------------
# flag -> config (argparse Namespace)
# ---------------------------------------------------------------------


@pytest.mark.parametrize("contract,mode,row", _CASES, ids=_IDS)
def test_flag_reaches_its_config_field(monkeypatch, contract, mode, row) -> None:
    """The repo's silent-no-op bug class: a flag that parses and is unused."""
    assert_row_reaches_config(monkeypatch, contract, mode, row)


@pytest.mark.parametrize("contract,mode,row", _CASES, ids=_IDS)
def test_probe_value_differs_from_the_default(monkeypatch, contract, mode, row) -> None:
    """Trap 1: without this, every row above could pass vacuously."""
    assert_row_value_is_not_the_default(monkeypatch, contract, mode, row)


# ---------------------------------------------------------------------
# completeness: every declared flag has a row, and vice versa
# ---------------------------------------------------------------------


@pytest.mark.parametrize("contract", CONTRACTS, ids=[c.name for c in CONTRACTS])
def test_every_declared_flag_has_a_contract_row(monkeypatch, contract) -> None:
    """A 16th flag added without a matching row fails HERE, not in production."""
    declared = declared_option_strings(contract.build_parser(monkeypatch))
    covered = contract.covered_flags
    assert declared == covered, (
        f"{contract.name}: flags with no contract row "
        f"{sorted(declared - covered)}; rows for flags the parser no longer "
        f"declares {sorted(covered - declared)}"
    )


# ---------------------------------------------------------------------
# --gpu: the one flag whose forwarding requires driving main(), not just
# the parser -- the exact defect class D-011 found for real (declared,
# advertised, never reached setup_gpu).
# ---------------------------------------------------------------------


class _SetupGpuProbe(Exception):
    """Raised by the sentinel so `main()` cannot progress into real work.

    The sentinel fires the instant `setup_gpu` is called -- the second
    statement in `main()`, strictly before `prepare_run_dir`, `load_mnist_
    data` or `build_model` ever run -- so this test trains nothing, touches
    no dataset and allocates no GPU beyond what importing the module already
    costs.
    """


def test_gpu_flag_reaches_setup_gpu(monkeypatch) -> None:
    """``--gpu 3`` must reach ``setup_gpu(gpu_id=3)`` inside ``main()``.

    A namespace-level check (the ``--gpu`` row above) only proves argparse
    parses the flag; it cannot prove ``main()`` forwards the parsed value
    onward. This is the actual shape of the D-011 regression: ``--gpu``
    parsed cleanly and was still never wired to ``setup_gpu``.
    """
    calls = []

    def _sentinel(**kwargs):
        calls.append(kwargs)
        raise _SetupGpuProbe

    monkeypatch.setattr(train_mothnet, "setup_gpu", _sentinel)

    with pytest.raises(_SetupGpuProbe):
        train_mothnet.main(["--gpu", "3"])

    assert calls == [{"gpu_id": 3}], (
        f"--gpu 3 did not reach setup_gpu(gpu_id=3): recorded calls={calls!r}"
    )
