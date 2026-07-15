"""Fail-closed guard: every CLI flag of both Graph Energy Transformer trainers reaches
``TrainingConfig``.

**The trap.** ``config_from_args()`` builds ``TrainingConfig(...)`` field-by-field from
``args.*``. A flag that ``parse_arguments()`` DEFINES but ``config_from_args()`` never READS
becomes a SILENT NO-OP: the user passes ``--learning-rate 1e-5``, argparse accepts it without a
murmur, the run trains at the dataclass default, and the resulting curve is then attributed to
the model. This has already bitten this repository (bfunet's ``high_freq_blocks`` and
``filter_multiplier`` were both silent no-ops for real runs -- see memory
``reference_bfunet_trainer_main_arg_wiring``).

**Why this test is STRUCTURAL, not a checklist.** A hand-written list of asserts guards only the
flags that existed the day it was written; flag #33, added next month, sails straight through --
which is the exact failure class we are guarding. So the test introspects instead:

* the flag surface comes from ``vars(parse_arguments([]))`` (every argparse ``dest``);
* the config surface comes from ``dataclasses.fields(TrainingConfig)``;
* a flag whose ``dest`` maps to no config field FAILS unless it is in an explicit, commented
  exclusion allow-list, and a config field fed by no flag FAILS the same way.

That makes the test FAIL-CLOSED: adding a flag without wiring it is RED by default. Both
allow-lists are EMPTY today -- every flag of both trainers really does reach the config, and
``test_a_dropped_wiring_line_is_caught`` proves the guard bites by dropping one wiring line and
asserting RED.

Covers success criterion SC3 / SC7 (the ``test_train`` leg of the three new test dirs).
"""

import dataclasses
from typing import Any, Callable, Dict, Optional, Tuple

import pytest

from train.graph_energy_transformer import train_anomaly as anomaly_trainer
from train.graph_energy_transformer import train_classification as cls_trainer

# ---------------------------------------------------------------------------
# The dest -> config-field map
# ---------------------------------------------------------------------------

# argparse dests that are DELIBERATELY named differently from the config field they feed.
# Anything not listed here must map to a field of the SAME name.
DEST_RENAMES: Dict[str, str] = {
    "optimizer": "optimizer_type",
    "lr_schedule": "lr_schedule_type",
}

# ---------------------------------------------------------------------------
# Exclusion allow-lists -- the ONLY escape hatches. Both are EMPTY, and that is the point:
# an unlisted flag that reaches no config field FAILS. Adding an entry here is a deliberate,
# reviewable act, not something a new flag gets for free.
#
# A flag belongs in EXCLUDED_DESTS only if it genuinely must not become a config field (e.g. a
# pure argparse-side switch consumed by main() itself). Note that `--gpu` is NOT such a case:
# it IS a config field (`TrainingConfig.gpu`), so it is covered here like every other flag.
EXCLUDED_DESTS: Dict[str, str] = {}  # dest -> why it feeds no config field

# A field belongs in EXCLUDED_FIELDS only if it is intentionally NOT settable from the CLI
# (i.e. derived or hard-coded). None currently is.
EXCLUDED_FIELDS: Dict[str, str] = {}  # field -> why no flag feeds it

# ---------------------------------------------------------------------------
# Non-default CLI values. EVERY dest of EVERY trainer must appear in one of these, or the test
# fails -- that is what makes a newly-added flag RED instead of silently uncovered.
#
# Each value is chosen to DIFFER from the dataclass default; `test_probe_values_are_non_default`
# proves that, so a value that accidentally equals the default cannot make the wiring asserts
# pass vacuously.
#
# dest -> (cli flag, value). Booleans are store_true/store_false/store_const flags: passing the
# flag flips the arg away from its default, so the builder emits the bare flag with no operand.
# ---------------------------------------------------------------------------

# Variant B (node anomaly) -- train_anomaly.py
ANOMALY_SPEC: Dict[str, Tuple[str, Any]] = {
    "dataset": ("--dataset", "yelpchi"),              # default: synthetic
    "batch_size": ("--batch-size", 8),               # default: 32
    "max_nodes": ("--max-nodes", 32),                # default: 64
    "num_hops": ("--num-hops", 3),                   # default: 2
    "synth_nodes": ("--synth-nodes", 500),           # default: 2000
    "synth_features": ("--synth-features", 16),      # default: 25
    "synth_anomaly_ratio": ("--synth-anomaly-ratio", 0.2),  # default: 0.1
    "embed_dim": ("--embed-dim", 32),                # default: 64
    "num_heads": ("--num-heads", 2),                 # default: 4
    "head_dim": ("--head-dim", 8),                   # default: 16
    "hopfield_dim": ("--hopfield-dim", 64),          # default: 128
    "num_steps": ("--num-steps", 3),                 # default: 2
    "step_size": ("--step-size", 0.05),              # default: 0.1
    "mlp_hidden_dim": ("--mlp-hidden-dim", 32),      # default: 64
    "mlp_dropout": ("--mlp-dropout", 0.3),           # default: 0.0
    "epochs": ("--epochs", 3),                       # default: 100
    "learning_rate": ("--learning-rate", 1e-5),      # default: 1e-3
    "optimizer": ("--optimizer", "sgd"),             # default: adam
    "lr_schedule": ("--lr-schedule", "constant"),    # default: cosine_decay
    "warmup_epochs": ("--warmup-epochs", 1),         # default: 0
    "weight_decay": ("--weight-decay", 0.01),        # default: 0.0
    "gradient_clipping": ("--gradient-clipping", 0.5),  # default: 1.0
    "early_stopping_patience": ("--early-stopping-patience", 4),  # default: 15
    "mixed_precision": ("--mixed-precision", True),  # default: False (store_true)
    "seed": ("--seed", 123),                         # default: 42
    "gpu": ("--gpu", 1),                             # default: None
    "max_steps": ("--max-steps", 7),                 # default: None
    "experiment_name": ("--experiment-name", "graph_et_anomaly_probe"),  # default: None
    # `output_dir` is filled in per-test from tmp_path (default: "results").
}

# Variant C-lite (graph classification) -- train_classification.py
CLS_SPEC: Dict[str, Tuple[str, Any]] = {
    "dataset": ("--dataset", "proteins"),            # default: mutag
    "batch_size": ("--batch-size", 8),               # default: 32
    "max_nodes": ("--max-nodes", 48),                # default: None
    "k_pe": ("--k-pe", 10),                          # default: 15
    "add_self_loops": ("--no-self-loops", False),    # default: True (store_false)
    "sign_flip": ("--no-sign-flip", False),          # default: True (store_false)
    "embed_dim": ("--embed-dim", 64),                # default: 128
    "num_heads": ("--num-heads", 6),                 # default: 12
    "head_dim": ("--head-dim", 32),                  # default: 64
    "hopfield_dim": ("--hopfield-dim", 256),         # default: 512
    "num_blocks": ("--num-blocks", 2),               # default: 4
    "num_steps": ("--num-steps", 5),                 # default: 12
    "step_size": ("--step-size", 0.05),              # default: 0.01
    "noise_std": ("--noise-std", 0.05),              # default: 0.02
    "head_dropout": ("--head-dropout", 0.3),         # default: 0.0
    "epochs": ("--epochs", 3),                       # default: 300
    "learning_rate": ("--learning-rate", 1e-5),      # default: 1e-3
    "optimizer": ("--optimizer", "sgd"),             # default: adamw
    "lr_schedule": ("--lr-schedule", "constant"),    # default: cosine_decay
    "warmup_epochs": ("--warmup-epochs", 1),         # default: 5
    "weight_decay": ("--weight-decay", 0.01),        # default: 1e-4
    "label_smoothing": ("--label-smoothing", 0.1),   # default: 0.05
    "gradient_clipping": ("--gradient-clipping", 0.5),  # default: 1.0
    "early_stopping_patience": ("--early-stopping-patience", 4),  # default: 30
    "mixed_precision": ("--mixed-precision", True),  # default: False (store_true)
    "jit_compile": ("--jit-compile", True),          # default: None (store_const True)
    "seed": ("--seed", 123),                         # default: 42
    "gpu": ("--gpu", 1),                             # default: None
    "max_steps": ("--max-steps", 7),                 # default: None
    "experiment_name": ("--experiment-name", "graph_et_classify_probe"),  # default: None
    # `output_dir` is filled in per-test from tmp_path (default: "results").
}


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class TrainerCase:
    """One trainer under test: its CLI, its config, and its non-default probe values."""

    name: str
    parse_arguments: Callable[[Optional[list]], Any]
    config_from_args: Callable[[Any], Any]
    config_cls: type
    spec: Dict[str, Tuple[str, Any]]


@pytest.fixture(params=["anomaly", "classification"])
def case(request, tmp_path) -> TrainerCase:
    """Both trainers, driven through the identical structural checks."""
    if request.param == "anomaly":
        spec = {**ANOMALY_SPEC, "output_dir": ("--output-dir", str(tmp_path / "results"))}
        return TrainerCase(
            name="train_anomaly",
            parse_arguments=anomaly_trainer.parse_arguments,
            config_from_args=anomaly_trainer.config_from_args,
            config_cls=anomaly_trainer.TrainingConfig,
            spec=spec,
        )

    spec = {**CLS_SPEC, "output_dir": ("--output-dir", str(tmp_path / "results"))}
    return TrainerCase(
        name="train_classification",
        parse_arguments=cls_trainer.parse_arguments,
        config_from_args=cls_trainer.config_from_args,
        config_cls=cls_trainer.TrainingConfig,
        spec=spec,
    )


def _cli_dests(case: TrainerCase) -> set:
    """Every argparse `dest` the trainer defines, read off a defaults-only parse."""
    return set(vars(case.parse_arguments([])).keys())


def _config_fields(case: TrainerCase) -> Dict[str, dataclasses.Field]:
    return {f.name: f for f in dataclasses.fields(case.config_cls)}


def _field_for(dest: str) -> str:
    return DEST_RENAMES.get(dest, dest)


def _build_argv(spec: Dict[str, Tuple[str, Any]]) -> list:
    argv: list = []
    for _dest, (flag, value) in spec.items():
        if isinstance(value, bool):
            # store_true / store_false / store_const: the bare flag IS the non-default value.
            argv.append(flag)
        else:
            argv += [flag, str(value)]
    return argv


# ---------------------------------------------------------------------------
# 1. Surface coverage -- the fail-closed core
# ---------------------------------------------------------------------------

def test_every_cli_flag_maps_to_a_config_field(case: TrainerCase) -> None:
    """A flag that maps to NO config field is a silent no-op by construction."""
    fields = _config_fields(case)
    unmapped = sorted(
        dest for dest in _cli_dests(case)
        if dest not in EXCLUDED_DESTS and _field_for(dest) not in fields
    )
    assert not unmapped, (
        f"{case.name}: CLI flag(s) {unmapped} map to no TrainingConfig field. Each is a SILENT "
        f"NO-OP: argparse accepts the value and the run uses the default. Wire it into "
        f"config_from_args(), add a DEST_RENAMES entry, or justify it in EXCLUDED_DESTS."
    )


def test_every_config_field_is_fed_by_a_cli_flag(case: TrainerCase) -> None:
    """The mirror image: a config field no flag can reach is dead config."""
    reachable = {_field_for(dest) for dest in _cli_dests(case)}
    unreachable = sorted(
        name for name in _config_fields(case)
        if name not in reachable and name not in EXCLUDED_FIELDS
    )
    assert not unreachable, (
        f"{case.name}: TrainingConfig field(s) {unreachable} are settable from no CLI flag. "
        f"Add the flag, or justify the omission in EXCLUDED_FIELDS."
    )


def test_every_cli_flag_has_a_non_default_probe_value(case: TrainerCase) -> None:
    """Fail-closed on the TEST side too: a new flag with no probe value is RED, not skipped."""
    uncovered = sorted(
        dest for dest in _cli_dests(case)
        if dest not in case.spec and dest not in EXCLUDED_DESTS
    )
    assert not uncovered, (
        f"{case.name}: CLI flag(s) {uncovered} have no non-default probe value in this test's "
        f"spec, so their wiring is UNVERIFIED. Add them to the per-trainer spec."
    )


# ---------------------------------------------------------------------------
# 2. Self-check -- the probe values must actually be non-default
# ---------------------------------------------------------------------------

def test_probe_values_are_non_default(case: TrainerCase) -> None:
    """If a "non-default" probe value equals the dataclass default, the wiring assert below
    passes VACUOUSLY -- it would pass just as happily with the wiring line deleted. This test
    is the guard on the guard.
    """
    fields = _config_fields(case)
    vacuous = []
    for dest, (_flag, value) in case.spec.items():
        field = fields[_field_for(dest)]
        if field.default is not dataclasses.MISSING and field.default == value:
            vacuous.append(f"{dest}={value!r} == default")
    assert not vacuous, (
        f"{case.name}: probe value(s) equal their dataclass default: {vacuous}. The wiring "
        f"asserts would pass even with the wiring line DELETED. Pick different values."
    )


# ---------------------------------------------------------------------------
# 3. The wiring assert itself
# ---------------------------------------------------------------------------

def test_every_cli_value_reaches_the_config(case: TrainerCase) -> None:
    """THE guard: parse a fully non-default argv, and demand every field carry the CLI value."""
    args = case.parse_arguments(_build_argv(case.spec))
    config = case.config_from_args(args)

    dropped = []
    fields = _config_fields(case)
    for dest, (flag, expected) in case.spec.items():
        field_name = _field_for(dest)
        actual = getattr(config, field_name)
        if actual != expected:
            default = fields[field_name].default
            at_default = " (still at the DATACLASS DEFAULT -- the flag is a SILENT NO-OP)" \
                if actual == default else ""
            dropped.append(
                f"{flag} -> TrainingConfig.{field_name}: expected {expected!r}, "
                f"got {actual!r}{at_default}"
            )

    assert not dropped, (
        f"{case.name}: {len(dropped)} CLI flag(s) did not reach TrainingConfig:\n  "
        + "\n  ".join(dropped)
    )


def test_defaults_only_parse_still_builds_a_valid_config(case: TrainerCase) -> None:
    """Sanity floor: the no-flags path must still produce a valid config with a generated name."""
    config = case.config_from_args(case.parse_arguments([]))
    assert config.experiment_name  # __post_init__ generates one when None
    assert config.output_dir == "results"


# ---------------------------------------------------------------------------
# 4. Prove the guard BITES -- a dropped wiring line must go RED
# ---------------------------------------------------------------------------

def test_a_dropped_wiring_line_is_caught(case: TrainerCase) -> None:
    """The guard is only worth anything if an UNWIRED field actually fails it. Simulate the exact
    defect the test exists to catch -- ``config_from_args`` silently dropping a field so it stays
    at its dataclass default -- and assert ``test_every_cli_value_reaches_the_config``'s logic
    turns RED. If this passes, the wiring asserts above certify nothing.
    """
    fields = _config_fields(case)
    # Pick a real, CLI-fed, non-bool field to "drop" (bools collapse to the flag alone).
    victim = next(
        dest for dest, (_flag, value) in case.spec.items()
        if not isinstance(value, bool) and dest != "output_dir"
    )
    victim_field = _field_for(victim)

    args = case.parse_arguments(_build_argv(case.spec))
    config = case.config_from_args(args)
    # Simulate the dropped wiring line: force the field back to its dataclass default, exactly
    # what happens when config_from_args() forgets to read args.<victim>.
    object.__setattr__(config, victim_field, fields[victim_field].default)

    _flag, expected = case.spec[victim]
    actual = getattr(config, victim_field)
    assert actual != expected, (
        f"{case.name}: expected the dropped-wiring simulation to diverge from the CLI value for "
        f"{victim!r}, but it did not -- the probe value may equal the default (guard is toothless)."
    )
