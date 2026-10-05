"""CLI contract guard for ``src/train/topographic_vae/train_topographic_vae.py``.

The defect class
----------------
A trainer declares a flag, ``--help`` advertises it, and ``config_from_args`` never
forwards it (or forwards it into the wrong field). The flag then silently does
nothing and ``config.json`` records the DEFAULT while the user believes they
overrode it. This file drives the REAL parser and the REAL config builder through
the shared ``tests/test_train/_cli_contract.py`` driver:

- every flag reaches its ``TopographicVAEConfig`` field with a probe value that is
  NOT the default (trap 1) and that no other row uses (trap 3);
- the row table equals the flags the REAL parser declares (trap 2), so a flag
  added without a row turns the completeness test RED.

Three flags are ``store_true`` NEGATIONS of a positive config field, and each one
is the interesting case: the field defaults to ``True`` and the flag sets it to
``False``. A driver that only checked "the value changed" would pass on a forward
that inverted it, which for ``--no-variance-variables`` would train the plain-VAE
baseline by default. Those rows are therefore asserted as VALUES, not as deltas.

Nothing here trains, loads a dataset or touches a GPU: every ``--output-dir`` is a
string that is never created, and the one ``main()`` run stubs ``setup_gpu``,
``train`` and the run writers.
"""

from __future__ import annotations

import os
import sys
from typing import Tuple

os.environ.setdefault("MPLBACKEND", "Agg")

import pytest  # noqa: E402

import train.topographic_vae.train_topographic_vae as trainer  # noqa: E402

from .._cli_contract import (  # noqa: E402,TID252 -- shared driver, one level up
    Contract,
    Mode,
    Row,
    assert_row_reaches_config,
    assert_row_value_is_not_the_default,
    declared_option_strings,
)


def _rows() -> Tuple[Row, ...]:
    """One row per declared flag.

    Every probe differs from the parser default AND from every other probe, so a
    cross-wired forward moves a field this row does not own and the row fails.
    """
    return (
        # -- data -----------------------------------------------------------------
        # `--dataset dsprites` carries `--transform orientation` as a COMPANION:
        # the default transform is `rotation`, which is an MNIST-only factor, so
        # `--dataset dsprites` alone is an invalid combination. The companion is
        # not this row's subject; `--transform` has its own row with a different
        # probe value.
        Row(("--dataset",),
            ("--dataset", "dsprites", "--transform", "orientation",
             "--variant", "dsprites"),
            "dataset", "dsprites"),
        Row(("--transform",), ("--transform", "scale"), "transform", "scale"),
        Row(("--sequence-length",), ("--sequence-length", "7"), "sequence_length", 7),
        Row(("--num-train-sequences",), ("--num-train-sequences", "77"),
            "num_train_sequences", 77),
        Row(("--num-val-sequences",), ("--num-val-sequences", "31"),
            "num_val_sequences", 31),
        Row(("--num-test-sequences",), ("--num-test-sequences", "29"),
            "num_test_sequences", 29),
        Row(("--dsprites-cache",),
            ("--dsprites-cache", "/tmp/dsprites-cache-probe"),
            "dsprites_cache", "/tmp/dsprites-cache-probe"),
        # -- model ----------------------------------------------------------------
        # `--variant dsprites` carries `--dataset dsprites` as a COMPANION: the frame
        # size follows from the dataset and the config refuses a variant that
        # disagrees with it, so the two flags must move together.
        Row(("--variant",),
            ("--variant", "dsprites", "--dataset", "dsprites",
             "--transform", "orientation"),
            "variant", "dsprites"),
        Row(("--num-capsules",), ("--num-capsules", "6"), "num_capsules", 6),
        Row(("--capsule-dim",), ("--capsule-dim", "5"), "capsule_dim", 5),
        Row(("--l-preset",), ("--l-preset", "half"), "l_preset", "half"),
        Row(("--coherence-window",), ("--coherence-window", "4"),
            "coherence_window", 4),
        Row(("--neighborhood-size",), ("--neighborhood-size", "5"),
            "neighborhood_size", 5),
        Row(("--temporal-coherence",), ("--temporal-coherence", "stationary"),
            "temporal_coherence", "stationary"),
        # `--topography torus_2d` needs `--grid-shape` (there is no default lattice)
        # and `--temporal-coherence none` (a 2-D torus has no single cyclic axis
        # for the coherence roll). Both are companions, not this row's subject.
        Row(("--topography",),
            ("--topography", "torus_2d", "--grid-shape", "6", "6",
             "--temporal-coherence", "none"),
            "topography", "torus_2d"),
        Row(("--grid-shape",), ("--grid-shape", "3", "4"), "grid_shape", (3, 4)),
        Row(("--no-variance-variables",), ("--no-variance-variables",),
            "use_variance_variables", False),
        Row(("--degrees-of-freedom",), ("--degrees-of-freedom", "3.5"),
            "degrees_of_freedom", 3.5),
        Row(("--prior-mean",), ("--prior-mean", "12.5"), "prior_mean", 12.5),
        Row(("--encoder-hidden-dims",),
            ("--encoder-hidden-dims", "64", "48"), "encoder_hidden_dims", [64, 48]),
        Row(("--decoder-hidden-dims",),
            ("--decoder-hidden-dims", "32", "24"), "decoder_hidden_dims", [32, 24]),
        # -- objective ------------------------------------------------------------
        Row(("--kl-loss-weight",), ("--kl-loss-weight", "0.25"),
            "kl_loss_weight", 0.25),
        Row(("--reconstruction-sum-reduction",),
            ("--reconstruction-sum-reduction",),
            "reconstruction_sum_reduction", True),
        # -- optimisation ---------------------------------------------------------
        Row(("--epochs",), ("--epochs", "3"), "epochs", 3),
        Row(("--batch-size",), ("--batch-size", "6"), "batch_size", 6),
        Row(("--learning-rate",), ("--learning-rate", "0.005"),
            "learning_rate", 0.005),
        Row(("--momentum",), ("--momentum", "0.5"), "momentum", 0.5),
        Row(("--lr-schedule",), ("--lr-schedule", "cosine"), "lr_schedule", "cosine"),
        Row(("--patience",), ("--patience", "9"), "patience", 9),
        # -- evaluation -----------------------------------------------------------
        Row(("--likelihood-samples",), ("--likelihood-samples", "3"),
            "likelihood_samples", 3),
        Row(("--likelihood-batch-size",), ("--likelihood-batch-size", "12"),
            "likelihood_batch_size", 12),
        Row(("--num-traversals",), ("--num-traversals", "5"), "num_traversals", 5),
        Row(("--no-model-analysis",), ("--no-model-analysis",),
            "run_analysis", False),
        # -- plumbing -------------------------------------------------------------
        Row(("--seed",), ("--seed", "1234"), "seed", 1234),
        Row(("--gpu",), ("--gpu", "1"), "gpu", 1),
        Row(("--output-dir",), ("--output-dir", "/tmp/tvae-probe"),
            "output_dir", "/tmp/tvae-probe"),
        Row(("--experiment-name",), ("--experiment-name", "probe-run"),
            "experiment_name", "probe-run"),
        Row(("--mixed-precision",), ("--mixed-precision", "mixed_float16"),
            "mixed_precision", "mixed_float16"),
        Row(("--no-visualizations",), ("--no-visualizations",),
            "save_visualizations", False),
    )


def _contract() -> Contract:
    return Contract(
        name="train_topographic_vae",
        build_parser=lambda monkeypatch: trainer.create_argument_parser(),
        build_config=lambda monkeypatch: trainer.config_from_args(
            trainer.create_argument_parser().parse_args(list(sys.argv[1:]))
        ),
        modes=(Mode(id="default", required_argv=(), rows=_rows()),),
    )


def test_every_declared_flag_has_a_row():
    """Trap 2. A flag added without a row leaves the rest of this file stale."""
    declared = declared_option_strings(trainer.create_argument_parser())
    row_flags = {flag for row in _rows() for flag in row.flags}
    assert declared == row_flags, (
        "the row table and the parser disagree:\n"
        f"  declared but unlisted: {sorted(declared - row_flags)}\n"
        f"  listed but undeclared: {sorted(row_flags - declared)}"
    )


@pytest.mark.parametrize("row", _rows(), ids=lambda row: row.id)
def test_each_flag_reaches_its_config_field(monkeypatch, row: Row):
    contract = _contract()
    assert_row_reaches_config(
        monkeypatch, contract, contract.modes[0], row
    )


@pytest.mark.parametrize("row", _rows(), ids=lambda row: row.id)
def test_each_flag_is_not_merely_the_default(monkeypatch, row: Row):
    """Trap 1. A probe equal to the default cannot distinguish forwarded from
    never-touched, which is what makes the rest of this file mean anything."""
    contract = _contract()
    assert_row_value_is_not_the_default(
        monkeypatch, contract, contract.modes[0], row
    )


def test_the_inverted_flags_reach_their_true_values():
    """The ``--no-*`` rows, asserted as VALUES rather than as "it changed".

    ``--no-variance-variables``, ``--no-model-analysis`` and ``--no-visualizations``
    each set a field whose default is ``True``. A forward that wrote
    ``args.no_variance_variables`` into the field would satisfy "the value moved"
    and invert the run.
    """
    parser = trainer.create_argument_parser()
    inverted = [row for row in _rows() if row.id.startswith("--no-")]
    assert len(inverted) == 3, inverted
    for row in inverted:
        config = trainer.config_from_args(
            parser.parse_args(list(row.argv))
        )
        assert getattr(config, row.field) is row.expected, (
            f"{row.id} did not invert {row.field} correctly"
        )


def test_the_default_config_carries_the_papered_defaults():
    """The other half of the inverted rows: with NO argv, every mode is the
    paper's model. Asserted positively so a default flipped to the baseline
    fails here rather than reading as a plausible-looking config."""
    config = trainer.config_from_args(
        trainer.create_argument_parser().parse_args([])
    )
    assert config.use_variance_variables is True
    assert config.run_analysis is True
    assert config.save_visualizations is True
    assert config.temporal_coherence == "shifting"
    assert config.l_preset == "third", (
        "the paper's best equivariance is near L = S/3"
    )
    assert config.learning_rate == pytest.approx(1e-4)
    assert config.momentum == pytest.approx(0.9)


def test_grid_shape_is_a_tuple_not_a_list():
    """``argparse`` returns a list; the config field is documented as a tuple.

    Asserted because the two are not interchangeable for a ``get_config()``
    round trip: a ``.keras`` config written from a list deserializes to a list,
    so a value comparison against a tuple-configured model would fail for a
    reason no shape or value assertion would explain.
    """
    config = trainer.config_from_args(
        trainer.create_argument_parser().parse_args(["--grid-shape", "3", "4"])
    )
    assert isinstance(config.grid_shape, tuple)
    assert config.grid_shape == (3, 4)


def test_coherence_window_overrides_the_preset():
    """The two flags that set the SAME quantity.

    ``--l-preset`` is a fraction of the sequence length and ``--coherence-window``
    is an explicit ``L``; the config carries the resolved integer plus the preset
    name, and ``build_model`` resolves the preset only when the explicit value is
    ``None``. A forward that dropped ``coherence_window`` would make the explicit
    flag inert while ``--l-preset`` kept working, which is exactly the shape this
    whole file exists to catch.
    """
    parser = trainer.create_argument_parser()
    explicit = trainer.config_from_args(
        parser.parse_args(["--coherence-window", "4"])
    )
    assert explicit.coherence_window == 4
    assert explicit.l_preset == "third", (
        "the preset must survive alongside the override so the summary records "
        "which was asked for"
    )
    preset = trainer.config_from_args(parser.parse_args([]))
    assert preset.coherence_window is None, (
        "None is the sentinel that lets the preset resolve; a resolved integer "
        "here would make --coherence-window a no-op"
    )


def test_no_smoke_mode_is_declared():
    """This trainer has no ``--smoke`` branch, so it must not claim one.

    The bfunet family needs the flag because its smoke path drives every row;
    this trainer has a real dataset loader and a real evaluation, and a smoke
    preset would be a second code path nothing here exercises.
    """
    assert "--smoke" not in declared_option_strings(
        trainer.create_argument_parser()
    )
    assert not hasattr(trainer, "SMOKE_PRESET")


def test_the_help_text_renders():
    """``--grid-shape`` declares a two-element ``metavar`` tuple, which argparse
    validates against ``nargs``; a mismatch raises here rather than at parse."""
    text = trainer.create_argument_parser().format_help()
    assert "--l-preset" in text
    assert "--no-variance-variables" in text
    assert "--mixed-precision" in text