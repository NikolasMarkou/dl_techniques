"""Every flag ``train.topolm.pretrain`` declares must reach the field it claims.

``_config_from_args`` hand-maps argparse destinations onto config fields one line
at a time -- 40 assignments -- and a dropped line is invisible: ``--help`` still
advertises the flag, the parser still accepts it, the run still starts, and
``config.json`` records the default while the user believes they overrode it.

This trainer adds a second and less obvious failure of the same kind. Half its
flags are TOPOGRAPHIC (``--alpha``, ``--radius``, ``--permute``, ``--tap-sites``)
and the paper's whole comparison lives in those numbers, so a flag that does
nothing here silently invalidates a run rather than merely mis-configuring one.

The completeness row is the load-bearing part: ``declared_option_strings`` reads
the flags off the REAL parser and the completeness test asserts set equality, so
a flag added later fails this file until it is given a row. Every row must
therefore also carry a NON-DEFAULT value (``assert_row_value_is_not_the_default``),
or it cannot distinguish "forwarded" from "never touched".

See ``tests/test_train/_cli_contract.py`` for the driver and the three vacuity
traps it designs out.

Two flags are deliberately not config fields and carry ``namespace_dest``
instead, asserted against the argparse NAMESPACE rather than silently skipped:

- ``--gpu``, consumed by ``setup_gpu`` in ``main()``;
- ``--paired`` and ``--no-topography``, which select a CODE PATH rather than a
  hyperparameter. Forwarding them into the dataclass would be the defect: the run
  would then be indistinguishable from one whose alpha differed.
"""

from __future__ import annotations

import pytest

from train.topolm import pretrain as tt

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
# Flags unique to this trainer
# ---------------------------------------------------------------------

#: Rows for the flags that exist only here. Kept separate from
#: ``CLM_PRETRAIN_ROWS`` so a change to the shared table cannot silently drop a
#: topographic flag's coverage.
#: ``--variant`` is OVERRIDDEN rather than inherited. The shared row drives
#: ``--variant medium``, which is a GPT-2 variant; this parser restricts choices
#: to the model's own table, so the inherited row would exit 2 on a value the
#: flag legitimately rejects. The override also fails if the parser ever stops
#: restricting its choices -- ``medium`` would then resolve and the row would
#: silently start passing for the wrong reason.
TOPOLM_ROWS: tuple = (
    Row(("--variant",), ("--variant", "tiny"), "model_variant", "tiny"),
    # --vocab-size is TopoLM-specific: the unit grid is a function of the WIDTH,
    # not of a variant name alone, so the width and its vocabulary are both
    # CLI-settable here.
    Row(("--vocab-size",), ("--vocab-size", "32000"), "vocab_size", 32000),
    Row(
        ("--dropout-rate",),
        ("--dropout-rate", "0.25"),
        "dropout_rate",
        0.25,
    ),
    Row(
        ("--attention-dropout-rate",),
        ("--attention-dropout-rate", "0.15"),
        "attention_dropout_rate",
        0.15,
    ),
    # The topographic objective. Every one of these must reach its field, and the
    # values below are mutually distinct so a cross-wire is caught.
    Row(("--alpha",), ("--alpha", "7.5"), "spatial_alpha", 7.5),
    Row(("--radius",), ("--radius", "4"), "spatial_radius", 4),
    Row(("--neighborhoods",), ("--neighborhoods", "9"), "spatial_neighborhoods", 9),
    Row(("--distance",), ("--distance", "l2"), "spatial_distance", "l2"),
    Row(
        ("--permute", "--no-permute"),
        ("--no-permute",),
        "spatial_permute",
        False,
    ),
    Row(("--tap-sites",), ("--tap-sites", "mlp"), "tap_sites", "mlp"),
    # Cadence and the paper's stopping rule.
    Row(
        ("--eval-every-steps",),
        ("--eval-every-steps", "321"),
        "eval_every_steps",
        321,
    ),
    Row(("--eval-batches",), ("--eval-batches", "11"), "eval_batches", 11),
    Row(
        ("--train-steps",),
        ("--train-steps", "6543"),
        "train_steps",
        6543,
    ),
    Row(
        ("--early-stop-patience",),
        ("--early-stop-patience", "9"),
        "early_stop_patience",
        9,
    ),
    Row(
        ("--spatial-log-every",),
        ("--spatial-log-every", "17"),
        "spatial_log_every",
        17,
    ),
    Row(
        ("--warmup-ratio",),
        ("--warmup-ratio", "0.2"),
        "warmup_ratio",
        0.2,
    ),
    Row(
        ("--weight-decay",),
        ("--weight-decay", "0.05"),
        "weight_decay",
        0.05,
    ),
    Row(("--readout-fwhm",), ("--readout-fwhm", "5.5"), "readout_fwhm", 5.5),
    Row(
        ("--readout-unit-spacing",),
        ("--readout-unit-spacing", "2.5"),
        "readout_unit_spacing",
        2.5,
    ),
    Row(
        ("--min-cluster-size",),
        ("--min-cluster-size", "23"),
        "min_cluster_size",
        23,
    ),
    Row(
        ("--permutation-p-value",),
        ("--permutation-p-value",),
        "permutation_p_value",
        True,
    ),
    Row(
        ("--num-permutations",),
        ("--num-permutations", "1999"),
        "num_permutations",
        1999,
    ),
    Row(
        ("--contrast",),
        ("--contrast", "c", "d"),
        "contrast_conditions",
        ("c", "d"),
    ),
    # Code paths, not hyperparameters. `expected` is what the argparse NAMESPACE
    # must hold, since these have no config field to land in.
    Row(("--paired",), ("--paired",), None, True, "paired"),
    Row(
        ("--no-topography",),
        ("--no-topography",),
        None,
        False,
        "run_topography",
    ),
)


def _rows() -> tuple:
    """The shared CLM rows, minus any flag this module re-declares.

    Subtract rather than concatenate: a flag present in both tables would be
    driven TWICE, once with the inherited value and once with the override, and
    the inherited one would fail on a value this parser legitimately rejects.
    ``test_the_shared_rows_are_overridden_not_duplicated`` pins the outcome.
    """
    owned = {flag for row in TOPOLM_ROWS for flag in row.flags}
    inherited = tuple(
        row for row in CLM_PRETRAIN_ROWS
        if not owned.intersection(row.flags)
    )
    return inherited + TOPOLM_ROWS


TOPOLM_PRETRAIN = Contract(
    name="train.topolm.pretrain",
    build_parser=lambda monkeypatch: tt._build_parser(),
    build_config=lambda monkeypatch: tt._config_from_args(
        tt._build_parser().parse_args()
    ),
    modes=(Mode("plain", (), _rows()),),
)


def _params():
    params, ids = cases((TOPOLM_PRETRAIN,))
    return pytest.mark.parametrize("contract,mode,row", params, ids=ids)


class TestFlagForwarding:
    @_params()
    def test_the_flag_reaches_the_field_it_claims(
        self, monkeypatch, contract, mode, row
    ):
        assert_row_reaches_config(monkeypatch, contract, mode, row)

    @_params()
    def test_the_row_value_is_distinguishable_from_the_default(
        self, monkeypatch, contract, mode, row
    ):
        """Without this the forwarding test above passes vacuously.

        A row whose value equals its config default cannot tell "forwarded" from
        "never touched", which is exactly the defect class this file exists for.
        """
        assert_row_value_is_not_the_default(monkeypatch, contract, mode, row)


class TestRowTableCompleteness:
    def test_every_declared_flag_has_a_row(self):
        """A newly added flag fails here until it is given a row.

        This is the guard that keeps the file from going stale: rows are checked
        against the flags read off the REAL parser, so an unlisted flag cannot be
        forgotten silently.
        """
        declared = declared_option_strings(tt._build_parser())
        covered = TOPOLM_PRETRAIN.covered_flags
        assert declared == covered, {
            "declared but uncovered": sorted(declared - covered),
            "covered but not declared": sorted(covered - declared),
        }

    def test_the_shared_rows_are_overridden_not_duplicated(self):
        """Every shared flag appears EXACTLY once across the combined table.

        Duplicating one would drive it twice -- once with the inherited value and
        once with the override -- and the inherited drive would fail on a value
        this parser rejects. Missing one would drop its coverage silently, since
        the completeness test only compares flag SETS.
        """
        combined = _rows()
        seen = [flag for row in combined for flag in row.flags]
        duplicated = {f for f in seen if seen.count(f) > 1}
        assert not duplicated, f"a flag is driven twice: {sorted(duplicated)}"

    def test_the_shared_table_supplies_the_inherited_flags(self):
        """The inherited rows really are shared, not retyped here.

        Re-typing them would let the two copies drift, and each file's own
        completeness test would still pass, because both copies name the same
        flags.
        """
        owned = {f for row in TOPOLM_ROWS for f in row.flags}
        inherited = [row for row in CLM_PRETRAIN_ROWS
                     if not owned.intersection(row.flags)]
        assert len(inherited) > 20, (
            "almost nothing is coming from the shared table; the point of "
            "reusing CLM_PRETRAIN_ROWS is that the inherited flags have ONE "
            "definition"
        )


class TestPathFlagsAreNotConfigFields:
    def test_gpu_reaches_the_namespace_and_not_the_config(self):
        """``--gpu`` is consumed by ``setup_gpu``, so there is no field for it.

        Asserted against the NAMESPACE rather than skipped: a flag with no field
        and no assertion is exactly how a half-wired flag survives review.
        """
        args = tt._build_parser().parse_args(["--gpu", "3"])
        assert args.gpu == 3
        config = tt._config_from_args(args)
        assert not hasattr(config, "gpu"), (
            "--gpu became a config field; it is consumed by setup_gpu and adding "
            "it to the dataclass would make the two indistinguishable"
        )

    def test_paired_is_not_a_config_field(self):
        """It selects a code path; as a field it would fake the comparison."""
        args = tt._build_parser().parse_args(["--paired"])
        assert args.paired is True
        config = tt._config_from_args(args)
        assert not hasattr(config, "paired")

    def test_no_topography_defaults_to_running_the_evaluation(self):
        args = tt._build_parser().parse_args([])
        assert args.run_topography is True
        assert tt._config_from_args(args) is not None

    def test_the_variant_choices_come_from_the_model_not_a_literal(self):
        """A stale literal would let ``--variant`` name a row that does not exist.

        The failure would arrive as a ``KeyError`` from the model factory deep in
        a run, after the dataset has been tokenized.
        """
        from dl_techniques.models.language.topolm import MODEL_VARIANTS

        parser = tt._build_parser()
        action = next(
            a for a in parser._actions if "--variant" in a.option_strings
        )
        assert set(action.choices) == set(MODEL_VARIANTS), (
            "the CLI's variant choices and MODEL_VARIANTS have drifted"
        )