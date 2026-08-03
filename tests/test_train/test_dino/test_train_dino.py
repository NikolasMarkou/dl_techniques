"""Guards for the DINO trainer (``src/train/dino/train_dino.py``).

Four things are pinned here, each because its failure mode is SILENT:

1. **Every CLI flag reaches ``TrainingConfig``.** A flag that ``parse_arguments()`` defines
   but ``config_from_args()`` never reads is a no-op: argparse accepts the value, the run
   trains at the dataclass default, and the resulting curve is then attributed to the
   setting the command line claimed. This has already bitten this repository (``bfunet``'s
   ``high_freq_blocks`` and ``filter_multiplier``). The check is STRUCTURAL -- it reflects
   over ``dataclasses.fields`` and over the parser's own dests -- because a hand-written
   list of assertions guards only the flags that existed the day it was written, and flag
   #31 added next month sails straight through, which is the exact class being guarded.

2. **``TrainingConfig.__post_init__`` rejects bad values**, on the message, not the type.

3. **``fit()`` is never given ``validation_data``.** DINO's centering EMA fires inside
   ``DINOLoss.call()``, so Keras running the loss over a validation set silently multiplies
   the per-epoch centering updates (MEASURED: 4 instead of 2, pushing the center 81% past
   its correct value, with a finite loss and a clean exit). A rule with no mechanical guard
   erodes; this one is asserted against a spy on the real ``train_dino()`` call path.

4. **The teacher-temperature ``LambdaCallback`` actually MOVES the loss's ``teacher_temp``
   Variable across epochs.** "The schedule exists" proves nothing: a Python-float
   temperature is constant-folded into the traced training step (MEASURED: a 100x change
   moved the loss by 7e-07). So the assertion is on the Variable's value after a REAL
   two-epoch ``fit()``.

Run:
    CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python -m pytest \\
        tests/test_train/test_dino/test_train_dino.py -q
"""

import dataclasses
from typing import Any, Dict, Tuple

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.datasets.vision.multi_crop import make_multi_crop_map_fn
from train.dino import train_dino as trainer

# ---------------------------------------------------------------------------
# The dest -> config-field map, and the two escape hatches
# ---------------------------------------------------------------------------

# argparse dests deliberately named differently from the field they feed.
DEST_RENAMES: Dict[str, str] = {
    "optimizer": "optimizer_type",
    "lr_schedule": "lr_schedule_type",
}

# A flag belongs here ONLY if it genuinely must not become a config field. Adding an entry
# is a deliberate, reviewable act -- an unlisted flag that reaches no field FAILS.
EXCLUDED_DESTS: Dict[str, str] = {
    "smoke": (
        "a PRESET, not a value: it overrides other fields (SMOKE_OVERRIDES) rather than "
        "carrying one of its own. Its effect is pinned by TestSmokePreset below."
    ),
    "explicit_flags": (
        "NOT a CLI flag at all: PROVENANCE data that parse_arguments attaches to the "
        "returned Namespace -- the set of dests the caller actually typed, which is how "
        "--smoke tells an explicitly-passed default from an omission. It appears in "
        "vars(args) but has no flag and must feed no config field."
    ),
}

# A field belongs here only if it is intentionally NOT settable from the CLI. None is.
EXCLUDED_FIELDS: Dict[str, str] = {}

# dest -> (flag, non-default value). EVERY dest must appear here or in EXCLUDED_DESTS, so a
# newly added flag is RED rather than silently uncovered. `test_probe_values_are_non_default`
# proves each value really differs from the dataclass default -- otherwise the wiring
# assertion would pass just as happily with the wiring line deleted.
FLAG_SPEC: Dict[str, Tuple[str, Any]] = {
    "dataset": ("--dataset", "cifar10"),                          # default: imagenette
    "global_crop_size": ("--global-crop-size", 96),               # default: 224
    "local_crop_size": ("--local-crop-size", 96),                 # default: None
    "n_local_crops": ("--n-local-crops", 6),                      # default: 4
    # >= the global_crop_size probe (96) -- __post_init__ refuses a smaller
    # decode resolution, since decoding below the crop size would upsample every
    # view even further, which is the opposite of what the field is for.
    "source_image_size": ("--source-image-size", 128),            # default: None
    # BooleanOptionalAction flags: the probe is False, so `_build_argv` emits the
    # `--no-` token. Both defaults moved False -> True in plan-2026-08-02-93deeae2,
    # which made the previous `True` probes VACUOUS -- and this table's own
    # `test_probe_values_are_non_default` is what caught it.
    "stateless_augmentation": ("--stateless-augmentation", False),  # default: True
    "seed_training_stream": ("--seed-training-stream", False),      # default: True
    "variant": ("--variant", "tiny"),                             # default: small
    "patch_size": ("--patch-size", 8),                            # default: None
    "dino_out_dim": ("--dino-out-dim", 4096),                     # default: 65536
    "student_temp": ("--student-temp", 0.2),                      # default: 0.1
    "teacher_temp": ("--teacher-temp", 0.05),                     # default: 0.04
    "teacher_temp_final": ("--teacher-temp-final", 0.09),         # default: 0.04
    "teacher_temp_warmup_epochs": ("--teacher-temp-warmup-epochs", 7),   # default: 30
    "center_momentum": ("--center-momentum", 0.95),               # default: 0.9
    "ema_decay_start": ("--ema-decay-start", 0.99),               # default: 0.996
    "ema_decay_end": ("--ema-decay-end", 0.999),                  # default: 0.9999
    "ema_warmup_steps": ("--ema-warmup-steps", 11),               # default: 0 (override)
    "ema_warmup_epochs": ("--ema-warmup-epochs", 3.0),            # default: 1.0
    "batch_size": ("--batch-size", 8),                            # default: 32
    "epochs": ("--epochs", 3),                                    # default: 100
    "learning_rate": ("--learning-rate", 1e-5),                   # default: 5e-4
    "optimizer": ("--optimizer", "sgd"),                          # default: adamw
    "lr_schedule": ("--lr-schedule", "constant"),                 # default: cosine_decay
    "warmup_epochs": ("--warmup-epochs", 1),                      # default: 10
    "weight_decay": ("--weight-decay", 0.01),                     # default: 0.04
    "gradient_clipping": ("--gradient-clipping", 0.5),            # default: 3.0
    "early_stopping_patience": ("--early-stopping-patience", 4),  # default: 30
    "knn_eval_every": ("--knn-eval-every", 3),                    # default: 1
    "knn_bank_batches": ("--knn-bank-batches", 5),                # default: 16
    "knn_query_batches": ("--knn-query-batches", 2),              # default: 8
    "knn_temperature": ("--knn-temperature", 0.05),               # default: 0.07
    "random_init_repeats": ("--random-init-repeats", 3),          # default: 2
    "max_steps": ("--max-steps", 9),                              # default: None
    "experiment_name": ("--experiment-name", "dino_cli_wiring_probe"),   # default: None
    "seed": ("--seed", 123),                                      # default: 42
    "gpu": ("--gpu", 1),                                          # default: None
    # `output_dir` is filled per-test from tmp_path (default: "results").
}


def _spec(tmp_path) -> Dict[str, Tuple[str, Any]]:
    return {**FLAG_SPEC, "output_dir": ("--output-dir", str(tmp_path / "results"))}


def _cli_dests() -> set:
    return set(vars(trainer.parse_arguments([])).keys())


def _config_fields() -> Dict[str, dataclasses.Field]:
    return {f.name: f for f in dataclasses.fields(trainer.TrainingConfig)}


def _field_for(dest: str) -> str:
    return DEST_RENAMES.get(dest, dest)


def _build_argv(spec: Dict[str, Tuple[str, Any]]) -> list:
    argv: list = []
    for _dest, (flag, value) in spec.items():
        if isinstance(value, bool):
            # A bool flag is `argparse.BooleanOptionalAction`, which owns BOTH
            # `--x` and `--no-x` under ONE dest. Emitting the bare flag for a
            # False probe would drive the value TRUE and silently assert the
            # default -- the vacuity class this module exists to catch.
            argv.append(flag if value else flag.replace("--", "--no-", 1))
        else:
            argv += [flag, str(value)]
    return argv


# ---------------------------------------------------------------------------
# 1. The CLI -> config wiring guard
# ---------------------------------------------------------------------------

class TestCLIWiring:
    """Fail-closed in BOTH directions: no unwired flag, no unreachable field."""

    def test_every_cli_flag_maps_to_a_config_field(self) -> None:
        fields = _config_fields()
        unmapped = sorted(
            dest for dest in _cli_dests()
            if dest not in EXCLUDED_DESTS and _field_for(dest) not in fields
        )
        assert not unmapped, (
            f"CLI flag(s) {unmapped} map to no TrainingConfig field. Each is a SILENT "
            f"NO-OP: argparse accepts the value and the run uses the default. Wire it into "
            f"config_from_args(), add a DEST_RENAMES entry, or justify it in EXCLUDED_DESTS."
        )

    def test_every_config_field_is_fed_by_a_cli_flag(self) -> None:
        reachable = {_field_for(dest) for dest in _cli_dests()}
        unreachable = sorted(
            name for name in _config_fields()
            if name not in reachable and name not in EXCLUDED_FIELDS
        )
        assert not unreachable, (
            f"TrainingConfig field(s) {unreachable} are settable from no CLI flag. Add the "
            f"flag, or justify the omission in EXCLUDED_FIELDS."
        )

    def test_every_cli_flag_has_a_non_default_probe_value(self, tmp_path) -> None:
        spec = _spec(tmp_path)
        uncovered = sorted(
            dest for dest in _cli_dests()
            if dest not in spec and dest not in EXCLUDED_DESTS
        )
        assert not uncovered, (
            f"CLI flag(s) {uncovered} have no non-default probe value, so their wiring is "
            f"UNVERIFIED. Add them to FLAG_SPEC."
        )

    def test_probe_values_are_non_default(self, tmp_path) -> None:
        """The guard on the guard: a probe value equal to the default proves nothing."""
        fields = _config_fields()
        vacuous = []
        for dest, (_flag, value) in _spec(tmp_path).items():
            field = fields[_field_for(dest)]
            if field.default is not dataclasses.MISSING and field.default == value:
                vacuous.append(f"{dest}={value!r} == default")
        assert not vacuous, (
            f"probe value(s) equal their dataclass default: {vacuous}. The wiring assert "
            f"would pass even with the wiring line DELETED. Pick different values."
        )

    def test_parser_defaults_agree_with_dataclass_defaults(self) -> None:
        """A diverged pair is a SILENT no-op, and MEASURED to be invisible here.

        During plan-2026-08-02-93deeae2 step 3 the `--teacher-temp-final` parser
        default and the dataclass default were deliberately diverged (0.04 vs
        0.07) as a RED-proof, and this whole module stayed GREEN. Nothing read
        the dataclass default on the CLI path, so the divergence only surfaces
        as a wrong number in a run nobody flagged -- and `SMOKE_OVERRIDES` only
        fills a field the caller left at the PARSER default, so a diverged pair
        also silently changes what `--smoke` applies to.
        """
        defaults = vars(trainer.parse_arguments([]))
        fields = _config_fields()
        diverged = []
        for dest in FLAG_SPEC:
            field = fields[_field_for(dest)]
            if field.default is dataclasses.MISSING:
                continue
            if defaults[dest] != field.default:
                diverged.append(
                    f"{dest}: parser default {defaults[dest]!r} != "
                    f"TrainingConfig.{_field_for(dest)} default {field.default!r}"
                )
        assert not diverged, (
            "parser/dataclass default(s) disagree:\n  " + "\n  ".join(diverged)
            + "\nEach is a SILENT divergence: the run's value depends on which "
            "construction path it took, and SMOKE_OVERRIDES stops applying."
        )

    def test_every_cli_value_reaches_the_config(self, tmp_path) -> None:
        """THE guard: parse a fully non-default argv, demand every field carry its value."""
        spec = _spec(tmp_path)
        args = trainer.parse_arguments(_build_argv(spec))
        config = trainer.config_from_args(args)

        fields = _config_fields()
        dropped = []
        for dest, (flag, expected) in spec.items():
            field_name = _field_for(dest)
            actual = getattr(config, field_name)
            if actual != expected:
                default = fields[field_name].default
                at_default = (
                    " (still at the DATACLASS DEFAULT -- the flag is a SILENT NO-OP)"
                    if actual == default else ""
                )
                dropped.append(
                    f"{flag} -> TrainingConfig.{field_name}: expected {expected!r}, "
                    f"got {actual!r}{at_default}"
                )

        assert not dropped, (
            f"{len(dropped)} CLI flag(s) did not reach TrainingConfig:\n  "
            + "\n  ".join(dropped)
        )

    def test_defaults_only_parse_builds_a_valid_config(self) -> None:
        config = trainer.config_from_args(trainer.parse_arguments([]))
        assert config.variant == "small"
        assert config.global_crop_size == 224
        assert config.experiment_name  # __post_init__ generates one


class TestReproducibilityFlagDefaults:
    """The shipped default is a REPRODUCIBLE run (plan-2026-08-02-93deeae2, D-004).

    Both flags are `argparse.BooleanOptionalAction`. They are required TOGETHER
    for bit-identical batches across processes (MEASURED, 2-process CPU-only sha1
    over the first 3 batches of the real `build_dataset` -- either alone DIFFERS;
    see research/2026_dino_ssl_measurements.md). That framing makes it easy to
    wire them as one switch by accident, so the off-switches are asserted
    INDEPENDENTLY below.
    """

    def test_both_default_on_in_the_dataclass_and_through_the_cli(self) -> None:
        assert trainer.TrainingConfig().stateless_augmentation is True
        assert trainer.TrainingConfig().seed_training_stream is True
        config = trainer.config_from_args(trainer.parse_arguments([]))
        assert config.stateless_augmentation is True
        assert config.seed_training_stream is True

    @pytest.mark.parametrize("off_flag,off_dest,other_dest", [
        ("--no-stateless-augmentation", "stateless_augmentation", "seed_training_stream"),
        ("--no-seed-training-stream", "seed_training_stream", "stateless_augmentation"),
    ])
    def test_each_off_switch_moves_only_its_own_flag(
        self, off_flag: str, off_dest: str, other_dest: str
    ) -> None:
        config = trainer.config_from_args(trainer.parse_arguments([off_flag]))
        assert getattr(config, off_dest) is False, f"{off_flag} did not turn {off_dest} off"
        assert getattr(config, other_dest) is True, (
            f"{off_flag} ALSO turned {other_dest} off -- the two flags are wired "
            f"as one switch, which their 'both required together' framing invites"
        )


class TestSmokePreset:
    """`--smoke` is the one flag with no field of its own; pin what it actually does."""

    def test_smoke_pins_the_measured_scale(self) -> None:
        config = trainer.config_from_args(trainer.parse_arguments(["--smoke"]))
        assert config.variant == "tiny"
        assert config.global_crop_size == 96
        assert config.n_local_crops == 4
        assert config.batch_size == 32
        assert config.dino_out_dim == 4096
        assert config.max_steps == trainer.SMOKE_OVERRIDES["max_steps"]

    def test_an_explicit_flag_still_beats_the_preset(self) -> None:
        config = trainer.config_from_args(
            trainer.parse_arguments(["--smoke", "--batch-size", "8"]))
        assert config.batch_size == 8, (
            "--smoke must only fill fields the caller left at the parser default; an "
            "explicit --batch-size 8 that comes back as 32 means the preset is clobbering "
            "the command line."
        )
        assert config.variant == "tiny"  # untouched fields still get the preset

    def test_an_explicit_default_value_also_beats_the_preset(self) -> None:
        """`--smoke --ema-warmup-epochs 1.0`, where 1.0 IS the parser default.

        This exact invocation silently resolved to `ema_warmup_epochs == 0.0` and a
        warmup of 0 steps before this plan: the preset decided "did the caller override
        this?" by re-parsing an empty argv and comparing PARSED VALUES, and a flag typed
        at its own default is value-identical to an omission. The caller asked for a
        one-epoch teacher freeze, got none, and nothing failed.

        `--batch-size 8` (the sibling test above) cannot catch that class, because 8
        differs from the default; only a flag typed AT its default separates provenance
        from value. What makes the difference is `explicitly_set_flags` (in
        `train.common.args`), whose result `parse_arguments` attaches to the Namespace
        and `config_from_args` reads instead of re-parsing.

        The second assertion is the BEHAVIOURAL one: carrying 1.0 into the config is
        only useful if it also reaches the resolved step count.
        """
        config = trainer.config_from_args(
            trainer.parse_arguments(["--smoke", "--ema-warmup-epochs", "1.0"]))
        assert config.ema_warmup_epochs == 1.0, (
            "--smoke overrode an EXPLICITLY TYPED --ema-warmup-epochs 1.0 back to "
            "SMOKE_OVERRIDES' 0.0. The preset is comparing VALUES, not provenance, so "
            "it cannot see a flag the caller typed at that flag's own default."
        )
        assert trainer.resolve_ema_warmup_steps(config, steps_per_epoch=295) == 295, (
            "the explicitly requested one-epoch teacher freeze resolved to a different "
            "step count -- an explicit --ema-warmup-epochs 1.0 under --smoke must reach "
            "the resolver as 1.0 and yield one epoch's worth of steps."
        )

    def test_a_boolean_optional_action_flag_is_seen_at_either_spelling(self) -> None:
        """`argparse.BooleanOptionalAction` owns TWO spellings under ONE dest.

        A token scanner that looked at only the first registered option string would
        register `--stateless-augmentation` and miss `--no-stateless-augmentation`, so
        the off-switch would stop counting as something the caller typed.

        The claim is split across two assertions on purpose, and the split is forced:
        NEITHER reproducibility flag appears in `SMOKE_OVERRIDES`, so `--smoke` never
        wants to overwrite this dest and the resolved config comes back the same whether
        provenance works or not. Asserting only on the resolved value for THIS dest
        would be vacuous. So:
          (a) `--batch-size 8` in the same argv is the sensitivity control -- it proves
              the resolved-config route is live while the boolean token is present, and
              that the boolean token does not derail the scan of its neighbours;
          (b) the provenance set itself is asserted for the boolean dest, since that is
              the only place the dual-spelling claim is observable today. It is read off
              the Namespace (the public product of `parse_arguments`) rather than from
              `explicitly_set_flags`' internals -- the parser object is not exported.
        If a boolean field ever enters `SMOKE_OVERRIDES`, fold (b) back into a pure
        resolved-config assertion.
        """
        for spelling, expected in (("--no-stateless-augmentation", False),
                                   ("--stateless-augmentation", True)):
            args = trainer.parse_arguments(["--smoke", spelling, "--batch-size", "8"])
            config = trainer.config_from_args(args)
            assert config.stateless_augmentation is expected, (
                f"{spelling} did not reach the config"
            )
            assert config.batch_size == 8, (
                f"an explicit --batch-size 8 came back as {config.batch_size} when "
                f"{spelling} was also on the command line -- the preset is clobbering a "
                f"typed flag, so this test's provenance assertion below is unsupported"
            )
            assert "stateless_augmentation" in args.explicit_flags, (
                f"{spelling} was NOT recorded as explicitly set. BooleanOptionalAction "
                f"registers both --x and --no-x on one dest; provenance must count "
                f"either spelling, or --smoke would override a field the caller typed."
            )

    def test_an_unambiguous_abbreviation_counts_as_typed(self) -> None:
        """`--smoke --ema-warmup-ep 1.5`, where the flag is ABBREVIATED.

        argparse accepts any unambiguous PREFIX of a long option by default
        (`parser.allow_abbrev` is True), so `--ema-warmup-ep` parses as
        `--ema-warmup-epochs` and `args.ema_warmup_epochs` really is 1.5. The first
        version of the provenance scan tested `token in dest_by_opt` — literal
        membership — which a prefix never satisfies, so `--smoke` overrode a flag the
        caller had actually typed, at a NON-default value. That was strictly worse than
        the value-comparison code it replaced: the old code kept 1.5 because 1.5 differs
        from the 1.0 default, and this one did not. Regression, not an inherited hole.

        1.5 (not 1.0) is deliberate: at the default the abbreviated and non-abbreviated
        paths agree for the wrong reason. 442 == int(1.5 * 295), derived by execution,
        not copied from a review.

        The fix lives in `explicitly_set_flags` (`train.common.args`), which now
        resolves a token the way argparse does — exact match first, then a prefix
        matching exactly one registered long option, and only when the parser's own
        `allow_abbrev` permits it.
        """
        config = trainer.config_from_args(
            trainer.parse_arguments(["--smoke", "--ema-warmup-ep", "1.5"]))
        assert config.ema_warmup_epochs == 1.5, (
            "--smoke overrode an EXPLICITLY TYPED (but abbreviated) --ema-warmup-ep "
            "1.5. argparse resolved the abbreviation and parsed 1.5; the provenance "
            "scan did not, so the preset could not see a flag the caller typed."
        )
        assert trainer.resolve_ema_warmup_steps(config, steps_per_epoch=295) == 442, (
            "the abbreviated --ema-warmup-ep 1.5 did not reach the resolver as 1.5 "
            "(1.5 * 295 == 442 steps)"
        )

    def test_smoke_still_means_no_teacher_ema_freeze(self) -> None:
        """`--smoke` must resolve to ZERO warmup, as it always has.

        `SMOKE_OVERRIDES` pins BOTH `ema_warmup_steps: 0` and
        `ema_warmup_epochs: 0.0`. The first alone is no longer sufficient: 0 steps now
        means "defer to epochs", and the shipped `ema_warmup_epochs` default is 1.0, so
        dropping the second pin makes every smoke run silently gain a freeze it has
        never had. Built through the REAL parse -> config path, because that wiring is
        where this trainer's documented silent-no-op defects live.
        """
        config = trainer.config_from_args(trainer.parse_arguments(["--smoke"]))
        # The BEHAVIOURAL assertion comes first on purpose: reading SMOKE_OVERRIDES
        # first would make the test die at a KeyError precondition when the pin is
        # dropped, and a precondition failure proves nothing about the resolution.
        assert trainer.resolve_ema_warmup_steps(config, steps_per_epoch=295) == 0
        assert trainer.SMOKE_OVERRIDES.get("ema_warmup_epochs") == 0.0


class TestEMAWarmupResolution:
    """The precedence rule between the epoch default and the absolute-step override."""

    def test_shipped_defaults_reproduce_the_measured_warmup(self) -> None:
        """THE decisive gate: the new defaults == the measured invocation's warmup.

        The configuration measured as IMPROVED used `--ema-warmup-steps 295`, and 295 is
        `num_train // batch_size` for imagenette at `batch_size=32`. This assertion
        stands in for a ~5 h GPU re-measurement: it proves the epoch-denominated default
        resolves, at the measured scale, to the EXACT `warmup_steps` the measured run
        passed to `TeacherEMACallback`. `steps_per_epoch=295` is passed IN -- this tests
        the resolution arithmetic, not the dataset cardinality.
        """
        assert trainer.resolve_ema_warmup_steps(
            trainer.TrainingConfig(), steps_per_epoch=295) == 295

    def test_an_explicit_step_count_wins(self) -> None:
        assert trainer.resolve_ema_warmup_steps(
            trainer.TrainingConfig(ema_warmup_steps=11), steps_per_epoch=295) == 11


class TestShippedTeacherTemperature:
    """The `teacher_temp_final = 0.04` default (D-003) and its CONSEQUENCE."""

    def test_the_shipped_default_is_the_measured_value(self) -> None:
        """The improved arm ran `--teacher-temp-final 0.04`; the default now matches it.

        Note what this does NOT claim: no temp-only arm was ever run at 60 epochs, so
        this is "the value the measured PAIR used", not "the better value".
        """
        assert trainer.TrainingConfig().teacher_temp_final == 0.04

    def test_teacher_temp_schedule_is_constant_at_shipped_defaults(self) -> None:
        """The (c) claim, pinned mechanically instead of as a comment.

        With `teacher_temp_final == teacher_temp` the linear ramp's delta is exactly
        zero, so `teacher_temp_warmup_epochs` scales a term multiplied by zero and the
        temperature never moves. This drives the REAL `create_teacher_temp_callback` --
        an uncompiled `DINOLoss` is enough, no model, no dataset, no GPU -- so it pins
        the shipped wiring, not a re-derivation of the schedule.
        """
        from dl_techniques.losses.dino_loss import DINOLoss

        config = trainer.TrainingConfig()
        loss = DINOLoss(out_dim=8)
        callback = trainer.create_teacher_temp_callback(config, loss)

        callback.on_epoch_begin(0)
        at_epoch_0 = float(loss.teacher_temp)
        # Well past the horizon: the ramp would have completed long ago if it moved.
        callback.on_epoch_begin(config.teacher_temp_warmup_epochs + 5)
        after_horizon = float(loss.teacher_temp)

        assert at_epoch_0 == pytest.approx(config.teacher_temp), (
            f"epoch 0 gave {at_epoch_0}, expected the start temperature "
            f"{config.teacher_temp} -- the schedule is not wired to `teacher_temp`."
        )
        assert after_horizon == at_epoch_0, (
            f"teacher_temp moved {at_epoch_0} -> {after_horizon} past the "
            f"{config.teacher_temp_warmup_epochs}-epoch horizon. The shipped defaults are "
            f"supposed to make the schedule CONSTANT, which is what makes "
            f"`teacher_temp_warmup_epochs` an inert knob (see the D-003 field comment). "
            f"If this fires, `teacher_temp_final` no longer equals `teacher_temp` and "
            f"BOTH help strings' inertness claim is now false."
        )


# ---------------------------------------------------------------------------
# 2. Config validation
# ---------------------------------------------------------------------------

class TestConfigValidation:
    """Every guard asserted on its MESSAGE -- a predicted exception TYPE is wrong more
    often than the failure CLASS, and several of these share ValueError."""

    @pytest.mark.parametrize("overrides,match", [
        ({"dataset": "not_a_dataset"}, "Unsupported dataset"),
        ({"variant": "enormous"}, "Unsupported variant"),
        ({"global_crop_size": 0}, "global_crop_size must be positive"),
        ({"n_local_crops": -1}, "n_local_crops must be >= 0"),
        ({"patch_size": 0}, "patch_size must be positive"),
        ({"dino_out_dim": 0}, "dino_out_dim must be positive"),
        ({"student_temp": 0.0}, "student_temp must be positive"),
        ({"teacher_temp": 0.0}, "teacher temperatures must be positive"),
        ({"teacher_temp_final": -1.0}, "teacher temperatures must be positive"),
        ({"teacher_temp_warmup_epochs": -1}, "teacher_temp_warmup_epochs must be >= 0"),
        ({"center_momentum": 1.0}, r"center_momentum must be in \[0, 1\)"),
        ({"ema_decay_start": 1.5}, r"ema_decay_start must be in \[0, 1\]"),
        ({"ema_decay_end": -0.1}, r"ema_decay_end must be in \[0, 1\]"),
        ({"ema_warmup_steps": -1}, "ema_warmup_steps must be >= 0"),
        ({"ema_warmup_epochs": -1.0}, "ema_warmup_epochs must be finite and >= 0"),
        ({"ema_warmup_epochs": float("nan")},
         "ema_warmup_epochs must be finite and >= 0"),
        ({"batch_size": 0}, "batch_size must be positive"),
        ({"epochs": 0}, "epochs must be positive"),
        ({"warmup_epochs": -1}, "warmup_epochs must be >= 0"),
        ({"max_steps": 0}, "max_steps must be positive when set"),
    ])
    def test_post_init_rejects(self, overrides: Dict[str, Any], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            trainer.TrainingConfig(**overrides)

    def test_a_valid_config_is_accepted(self) -> None:
        """Non-vacuity control: the rejections above must not be rejecting everything."""
        config = trainer.TrainingConfig(
            variant="tiny", global_crop_size=32, patch_size=16, dino_out_dim=16)
        assert config.n_views == 2 + config.n_local_crops

    def test_local_crop_size_mismatch_is_refused_by_the_map_fn(self) -> None:
        """The D-002 refusal has ONE definition (make_multi_crop_map_fn) and is NOT copied
        into __post_init__; this pins that the trainer really reaches it."""
        config = trainer.TrainingConfig(
            variant="tiny", global_crop_size=32, patch_size=16, dino_out_dim=16,
            local_crop_size=16, n_local_crops=1, batch_size=2, dataset="cifar10")
        with pytest.raises(NotImplementedError, match="positional-embedding interpolation"):
            trainer.build_dataset(config)

    @pytest.mark.parametrize("source_image_size,expected", [(None, 32), (64, 64)])
    def test_source_image_size_reaches_the_decode_resolution(
            self, monkeypatch, source_image_size, expected) -> None:
        """A config field that never reaches its use site is a silent no-op.

        `source_image_size` sets the resolution at which `build_raw_image_dataset`
        DECODES each record, i.e. the resolution the multi-crop transform crops
        FROM. Wiring it CLI -> config is checked by `TestCLIWiring`; this checks
        config -> the actual call, which is the half that would otherwise leave
        the flag inert. The `None` arm is the non-vacuity control: it pins that
        the default really does resolve to `global_crop_size`, so the `64` arm is
        measuring a change rather than a constant.
        """
        seen: Dict[str, Any] = {}

        def spy(dataset, image_size, batch_size, **kwargs):
            seen["image_size"] = image_size
            raise RuntimeError("stop: the decode resolution has been captured")

        monkeypatch.setattr(trainer, "build_raw_image_dataset", spy)
        config = trainer.TrainingConfig(
            variant="tiny", global_crop_size=32, patch_size=16, dino_out_dim=16,
            n_local_crops=1, batch_size=2, dataset="cifar10",
            source_image_size=source_image_size)
        with pytest.raises(RuntimeError, match="decode resolution"):
            trainer.build_dataset(config)
        assert seen["image_size"] == expected, (
            f"build_raw_image_dataset was asked to decode at "
            f"{seen['image_size']}, not {expected} -- source_image_size="
            f"{source_image_size} did not reach the pipeline"
        )

    @pytest.mark.parametrize("stateless,expected_slot,absent_slot", [
        (False, "element_map_fn", "indexed_element_map_fn"),   # non-vacuity
        (True, "indexed_element_map_fn", "element_map_fn"),
    ])
    def test_stateless_augmentation_picks_the_indexed_map_slot(
            self, monkeypatch, stateless, expected_slot, absent_slot) -> None:
        """The half `TestCLIWiring` cannot see: config -> the actual call.

        The two slots are two different CALLING CONVENTIONS (`fn(image, label)`
        vs `fn(index, image, label)`), and `build_raw_image_dataset` refuses
        both at once, so putting the stateless map fn in the wrong slot is not
        a subtle degradation -- it is a `TypeError` a thousand steps in, or, if
        the arity happened to match, a silently unindexed augmentation. The
        `False` arm is the non-vacuity control: it pins that the DEFAULT still
        uses the plain slot, so the `True` arm measures a change.
        """
        seen: Dict[str, Any] = {}

        def spy(dataset, image_size, batch_size, **kwargs):
            seen.update(kwargs)
            raise RuntimeError("stop: the map-fn slot has been captured")

        monkeypatch.setattr(trainer, "build_raw_image_dataset", spy)
        config = trainer.TrainingConfig(
            variant="tiny", global_crop_size=32, patch_size=16, dino_out_dim=16,
            n_local_crops=1, batch_size=2, dataset="cifar10",
            stateless_augmentation=stateless)
        with pytest.raises(RuntimeError, match="map-fn slot"):
            trainer.build_dataset(config)

        assert callable(seen.get(expected_slot)), (
            f"stateless_augmentation={stateless} did not put the multi-crop "
            f"map fn in `{expected_slot}` (got keys {sorted(seen)})"
        )
        assert absent_slot not in seen, (
            f"both map-fn slots were passed; build_raw_image_dataset refuses "
            f"that combination, so this run would die at pipeline construction"
        )

    @pytest.mark.parametrize("seed_training_stream,expected", [
        (False, None),   # non-vacuity control: the DEFAULT must pass NO kwarg
        (True, 1234),
    ])
    def test_seed_training_stream_reaches_the_training_file_order(
            self, monkeypatch, seed_training_stream, expected) -> None:
        """D-011: the half `TestCLIWiring` cannot see -- config -> the actual call.

        `build_knn_datasets` seeds the k-NN bank's TFDS file interleave (D-040)
        UNCONDITIONALLY; `build_dataset` seeds the TRAINING stream's only when this
        flag is on. It now ships ON, so the `False` arm below is the
        `--no-seed-training-stream` path -- the one where two same-seed runs share
        the measuring instrument and NOT the data they are trained on. (That
        sentence described the DEFAULT before this flag flipped; do not reinstate
        the old form.) This flag closes that source -- and only that source: MEASURED
        across two processes, this flag ALONE still yields different batches (the
        augmentation RNG is the residual), so a reproducible run needs it TOGETHER
        with `--stateless-augmentation`. Neither is redundant.

        The `False` arm is the non-vacuity control, and it asserts ABSENCE rather
        than `None`: `build_raw_image_dataset` is shared by 6 other `src/train/`
        consumers and its own guard
        (`test_shuffle_files_seed_is_strictly_additive`) pins that
        `shuffle_files_seed=None` must pass no `read_config`. Keeping the kwarg out
        of the default call entirely means the default path is byte-for-byte the
        call it was before this flag existed.
        """
        seen: Dict[str, Any] = {}

        def spy(dataset, image_size, batch_size, **kwargs):
            seen.update(kwargs)
            raise RuntimeError("stop: the training-stream kwargs have been captured")

        monkeypatch.setattr(trainer, "build_raw_image_dataset", spy)
        config = trainer.TrainingConfig(
            variant="tiny", global_crop_size=32, patch_size=16, dino_out_dim=16,
            n_local_crops=1, batch_size=2, dataset="cifar10", seed=1234,
            seed_training_stream=seed_training_stream)
        with pytest.raises(RuntimeError, match="training-stream kwargs"):
            trainer.build_dataset(config)

        if expected is None:
            assert "shuffle_files_seed" not in seen, (
                f"the DEFAULT training pipeline passed "
                f"shuffle_files_seed={seen.get('shuffle_files_seed')!r}; it must "
                f"pass the kwarg not at all, so the default call stays byte-for-"
                f"byte what it was. Got keys {sorted(seen)}")
        else:
            assert seen.get("shuffle_files_seed") == expected, (
                f"--seed-training-stream did not reach the training pipeline (got "
                f"shuffle_files_seed={seen.get('shuffle_files_seed')!r}, wanted "
                f"{expected}); the TFDS file interleave stays unseeded and two "
                f"same-seed runs train on a different example order from step 0")

    def test_the_knn_memory_bank_seeds_the_TFDS_FILE_order(
            self, monkeypatch) -> None:
        """D-040: the bank must be seeded down to the file interleave, not just `seed`.

        `.take(knn_bank_batches)` selects a SMALL sample of the 9469 train images
        -- `knn_bank_batches * batch_size`, i.e. 512 at the `--smoke` defaults
        (16 x 32) and 2048 at the 64/32 probe settings every measured figure was
        taken at -- and `build_raw_image_dataset` opens the train split with
        `shuffle_files=True`. `seed=` reaches only the element `.shuffle()` and the
        augmentation. MEASURED before this was wired: four bank draws at `seed=42`
        (two per process, two processes) gave four DIFFERENT label sequences, and
        four zero-step k-NN controls spread `dino_knn_top1_k20` over a range of
        0.0195 -- larger than the +0.0127 effect a step-14 A/B was reading.

        The QUERY arm is the non-vacuity control: the validation split is opened
        with `shuffle_files=False`, so seeding its file order would be meaningless
        and this test would pass whether or not the BANK were seeded if it only
        checked "some call got a seed".
        """
        calls: list = []

        def spy(dataset, image_size, batch_size, **kwargs):
            calls.append(kwargs)
            if len(calls) == 2:
                raise RuntimeError("stop: both k-NN pipelines have been captured")
            return (object(), 0, 0)

        monkeypatch.setattr(trainer, "build_raw_image_dataset", spy)
        config = trainer.TrainingConfig(
            variant="tiny", global_crop_size=32, patch_size=16, dino_out_dim=16,
            n_local_crops=1, batch_size=2, dataset="cifar10", seed=1234)
        with pytest.raises(RuntimeError, match="both k-NN pipelines"):
            trainer.build_knn_datasets(config)

        bank_kwargs, query_kwargs = calls[0], calls[1]
        assert bank_kwargs["is_training"] is True, "call 0 must be the BANK"
        assert query_kwargs["is_training"] is False, "call 1 must be the QUERY set"
        assert bank_kwargs.get("shuffle_files_seed") == 1234, (
            f"the k-NN memory bank was built WITHOUT a seeded TFDS file order "
            f"(got shuffle_files_seed={bank_kwargs.get('shuffle_files_seed')!r}); "
            f"every dino_knn_top1_* it produces then carries a ~0.02 run-to-run "
            f"band from an unseeded file interleave"
        )

    @pytest.mark.parametrize("shuffle_files_seed,expect_read_config", [
        (None, False),   # non-vacuity control: the DEFAULT must not change
        (42, True),
    ])
    def test_shuffle_files_seed_is_strictly_additive(
            self, monkeypatch, shuffle_files_seed, expect_read_config) -> None:
        """D-040: `shuffle_files_seed=None` must pass NO read_config at all.

        `build_raw_image_dataset` is shared by every `src/train/` consumer, so the
        new parameter is only safe if its default is byte-for-byte today's call.
        `ReadConfig(shuffle_seed=None)` is NOT the same thing as no ReadConfig, and
        the difference is invisible in a single-consumer test -- hence the `None`
        arm, which is the control this pair exists for.

        This guard lives beside the DINO trainer rather than in a new test module
        because `shuffle_files_seed` exists for `build_knn_datasets` above and the
        parameter is exercised nowhere else; the shared function's own regression
        gate is `tests/test_train/test_energy_transformer/`.
        """
        import tensorflow_datasets as tfds

        from train.energy_transformer import common as et_common

        seen: Dict[str, Any] = {}

        class _FakeSplit:
            num_examples = 8

        class _FakeInfo:
            splits = {"train": _FakeSplit(), "validation": _FakeSplit()}

        class _FakeBuilder:
            info = _FakeInfo()

            def as_dataset(self, **kwargs):
                seen.update(kwargs)
                raise RuntimeError("stop: as_dataset kwargs captured")

        monkeypatch.setattr(tfds, "builder", lambda *a, **k: _FakeBuilder())

        with pytest.raises(RuntimeError, match="as_dataset kwargs captured"):
            et_common.build_raw_image_dataset(
                "imagenette", 32, 2, is_training=True, augment=False, seed=7,
                shuffle_files_seed=shuffle_files_seed)

        assert seen["shuffle_files"] is True
        if expect_read_config:
            assert "read_config" in seen, (
                f"shuffle_files_seed={shuffle_files_seed} did not reach "
                f"as_dataset; the file order stays unseeded. Got {sorted(seen)}")
            assert seen["read_config"].shuffle_seed == 42
        else:
            assert "read_config" not in seen, (
                f"the DEFAULT passed read_config={seen.get('read_config')!r}; "
                f"shuffle_files_seed=None must change NOTHING for the other "
                f"src/train/ consumers of this shared function")


# ---------------------------------------------------------------------------
# 3. Construction-level fixtures: a tiny model + a synthetic multi-crop pipeline
# ---------------------------------------------------------------------------

CROP = 32
PATCH = 16
OUT_DIM = 16
N_LOCAL = 1
BATCH = 2


def _tiny_config(tmp_path, **overrides: Any) -> trainer.TrainingConfig:
    base: Dict[str, Any] = dict(
        dataset="cifar10",
        variant="tiny",
        global_crop_size=CROP,
        patch_size=PATCH,
        dino_out_dim=OUT_DIM,
        n_local_crops=N_LOCAL,
        batch_size=BATCH,
        epochs=2,
        max_steps=2,
        warmup_epochs=0,
        teacher_temp=0.02,
        teacher_temp_final=0.5,
        teacher_temp_warmup_epochs=1,
        output_dir=str(tmp_path / "results"),
        experiment_name="dino_unit",
        seed=0,
    )
    base.update(overrides)
    return trainer.TrainingConfig(**base)


def _synthetic_multi_crop_ds(config: trainer.TrainingConfig) -> tf.data.Dataset:
    """A REAL ``tf.data`` multi-crop pipeline over synthetic images.

    Uses the production ``make_multi_crop_map_fn``, so the element contract under test is
    the real one; only the image SOURCE is synthetic (no TFDS, no download).
    """
    rng = np.random.default_rng(0)
    images = rng.normal(size=(8, CROP + 16, CROP + 16, 3)).astype("float32")
    labels = np.zeros((8,), dtype="int32")
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    ds = ds.map(make_multi_crop_map_fn(
        global_crop_size=config.global_crop_size,
        n_local_crops=config.n_local_crops,
        seed=0,
    ))
    return ds.repeat().batch(config.batch_size, drop_remainder=True).prefetch(1)


class TestConstruction:
    """The trainer's pieces build. The full run is a separate, explicit smoke run."""

    def test_build_model_and_loss(self, tmp_path) -> None:
        config = _tiny_config(tmp_path)
        model, loss = trainer.build_model_and_loss(config)

        assert model.built
        assert model.n_views == config.n_views
        assert model.out_dim == OUT_DIM
        assert model.teacher.trainable is False
        assert hasattr(model, "update_teacher_ema"), (
            "TeacherEMACallback binds to this method BY NAME; without it the callback logs "
            "one warning, self-disables, and the whole run trains an untouched teacher."
        )
        assert loss.out_dim == OUT_DIM
        assert loss.teacher_temp == pytest.approx(config.teacher_temp)

    def test_create_callbacks_wires_the_ema_and_the_temperature(self, tmp_path) -> None:
        from dl_techniques.models.depth_anything.teacher_ema import TeacherEMACallback

        config = _tiny_config(tmp_path)
        _model, loss = trainer.build_model_and_loss(config)
        run_dir = tmp_path / "run"
        callbacks, results_dir = trainer.create_callbacks(
            config, loss, str(run_dir), steps_per_epoch=2)

        assert str(results_dir) == str(run_dir)
        kinds = [type(cb).__name__ for cb in callbacks]
        assert "TeacherEMACallback" in kinds
        assert "LambdaCallback" in kinds, (
            "the teacher-temperature warmup rides a stock LambdaCallback; a sixth "
            "schedule-callback class is explicitly not wanted here"
        )
        assert "CSVLogger" in kinds and "ModelCheckpoint" in kinds
        assert any(isinstance(cb, TeacherEMACallback) for cb in callbacks)

        # The PLUMBING, not the arithmetic. `test_shipped_defaults_reproduce_the_measured
        # _warmup` calls `resolve_ema_warmup_steps` directly and would stay green if
        # `create_callbacks` were reverted to `warmup_steps=config.ema_warmup_steps` --
        # every shipped run would then silently lose its teacher freeze with no test
        # failing. Assert the RESOLVED value actually arrives at the callback.
        ema_cb = next(cb for cb in callbacks if isinstance(cb, TeacherEMACallback))
        expected_warmup = trainer.resolve_ema_warmup_steps(config, steps_per_epoch=2)
        assert expected_warmup == 2 and config.ema_warmup_steps == 0, (
            f"precondition: this config must make the resolved value DIFFER from the raw "
            f"field, or the assertion below cannot tell them apart (resolved "
            f"{expected_warmup}, config.ema_warmup_steps {config.ema_warmup_steps})"
        )
        assert ema_cb.warmup_steps == expected_warmup, (
            f"TeacherEMACallback got warmup_steps={ema_cb.warmup_steps!r}, but "
            f"resolve_ema_warmup_steps(config, steps_per_epoch=2)={expected_warmup!r}. "
            f"create_callbacks is passing the RAW config field "
            f"(config.ema_warmup_steps={config.ema_warmup_steps!r}) instead of the "
            f"resolved one -- the epoch-denominated default never reaches the teacher "
            f"and every run silently trains with no EMA freeze."
        )

        # There is no val_loss in this run; every monitor must be a training metric.
        for cb in callbacks:
            monitor = getattr(cb, "monitor", None)
            if monitor is not None:
                assert not monitor.startswith("val_"), (
                    f"{type(cb).__name__} monitors {monitor!r}, but this run passes no "
                    f"validation_data, so that metric is never produced -- ModelCheckpoint "
                    f"would silently never save and EarlyStopping never fire."
                )

    def test_a_real_batch_forward_passes_through_the_compiled_model(self, tmp_path) -> None:
        config = _tiny_config(tmp_path)
        model, loss = trainer.build_model_and_loss(config)
        model.compile(optimizer=keras.optimizers.SGD(0.0), loss=loss)

        views, labels = next(iter(_synthetic_multi_crop_ds(config)))
        assert tuple(views.shape) == (
            BATCH, config.n_views, CROP, CROP, 3)

        out = model(views, training=False)
        assert tuple(out.shape) == (BATCH * model.n_pairs, 2 * OUT_DIM)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))
        assert float(loss(labels, out)) > 0.0


# ---------------------------------------------------------------------------
# 4. The D-001 rule: fit() never receives validation_data
# ---------------------------------------------------------------------------

def _make_fit_spy() -> Tuple[Any, list]:
    """A stand-in for ``keras.Model.fit`` recording exactly what the trainer asked for.

    A plain FUNCTION, not a callable object: only functions implement the descriptor
    protocol, so this is what binds ``self`` correctly when patched onto ``keras.Model``.
    """
    calls: list = []

    def fake_fit(self, *args: Any, **kwargs: Any):
        calls.append({"args": args, "kwargs": kwargs})
        history = keras.callbacks.History()
        history.history = {"loss": [1.0, 0.9]}
        return history

    return fake_fit, calls


class TestNoValidationData:
    """DINOLoss centers inside call(); a validation set silently corrupts the center."""

    @pytest.fixture()
    def spy_run(self, tmp_path, monkeypatch) -> Tuple[list, Dict[str, Any]]:
        config = _tiny_config(tmp_path)
        ds = _synthetic_multi_crop_ds(config)

        # Keep the real model/loss/callback construction; only the dataset SOURCE and the
        # fit() call itself are substituted, so this exercises the true code path.
        monkeypatch.setattr(trainer, "build_dataset", lambda cfg: (ds, 2))
        fake_fit, calls = _make_fit_spy()
        monkeypatch.setattr(keras.Model, "fit", fake_fit, raising=True)

        result = trainer.train_dino(config)
        return calls, result

    def test_fit_was_called_once(self, spy_run) -> None:
        spy, _result = spy_run
        assert len(spy) == 1, (
            "the trainer must run ONE stock fit(); more than one call means a bespoke "
            "multi-stage loop crept in."
        )

    def test_fit_received_no_validation_data(self, spy_run) -> None:
        spy, _result = spy_run
        kwargs = spy[0]["kwargs"]
        offenders = {
            key: kwargs[key] for key in ("validation_data", "validation_steps",
                                         "validation_split", "validation_batch_size")
            if kwargs.get(key) is not None
        }
        assert not offenders, (
            f"train_dino() passed {offenders} to fit(). DINOLoss updates its centering EMA "
            f"inside call(), and Keras runs the loss on validation batches too, so EVERY "
            f"validation batch performs a full unwanted centering update. MEASURED: a "
            f"4-sample validation set at batch_size=2 doubled an epoch's update count from "
            f"2 to 4 and pushed the center 81% past its correct value -- silently, with a "
            f"finite loss and a clean exit. Validation belongs in a k-NN callback."
        )

    def test_the_spy_can_see_kwargs_at_all(self, spy_run) -> None:
        """Non-vacuity control for the assertion above: if the spy captured NOTHING, an
        empty `offenders` would be meaningless."""
        spy, _result = spy_run
        kwargs = spy[0]["kwargs"]
        assert kwargs.get("epochs") == 2 and kwargs.get("steps_per_epoch") == 2, (
            f"the fit() spy did not capture the trainer's own kwargs ({sorted(kwargs)}); "
            f"the validation_data assertion above would then be vacuous."
        )
        assert kwargs.get("callbacks"), "no callbacks reached fit()"

    def test_the_run_wrote_its_artifacts(self, spy_run) -> None:
        _spy, result = spy_run
        run_dir = tmp_path_of(result)
        assert (run_dir / "config.json").is_file()
        assert (run_dir / "final_model.keras").is_file()
        assert not str(run_dir).endswith("src/results"), "outputs must not land in src/"


def tmp_path_of(result: Dict[str, Any]):
    from pathlib import Path
    return Path(result["run_dir"])


# ---------------------------------------------------------------------------
# 5. The teacher-temperature schedule actually moves the Variable
# ---------------------------------------------------------------------------

class TestTeacherTemperatureSchedule:
    """A schedule that "exists" is not a schedule that RUNS.

    ``DINOLoss.teacher_temp`` is a ``keras.Variable`` precisely because a Python float is
    constant-folded into the traced training step (MEASURED: a 100x change to the plain
    attribute moved the loss by 7e-07). The assertion here is therefore on the VALUE after
    a real two-epoch ``fit()``, not on the presence of a callback.
    """

    def test_the_lambda_callback_moves_teacher_temp_across_epochs(self, tmp_path) -> None:
        config = _tiny_config(tmp_path)
        model, loss = trainer.build_model_and_loss(config)
        model.compile(optimizer=keras.optimizers.SGD(1e-3), loss=loss)
        callbacks, _dir = trainer.create_callbacks(
            config, loss, str(tmp_path / "run"), steps_per_epoch=2)

        start = loss.teacher_temp
        assert start == pytest.approx(config.teacher_temp)

        model.fit(
            _synthetic_multi_crop_ds(config),
            epochs=2, steps_per_epoch=2, callbacks=callbacks, verbose=0,
        )

        end = loss.teacher_temp
        # warmup horizon = 1 epoch, so epoch 0 -> start and epoch 1 -> final.
        assert end == pytest.approx(config.teacher_temp_final), (
            f"teacher_temp is {end} after two epochs; the schedule should have carried it "
            f"{config.teacher_temp} -> {config.teacher_temp_final}. A value stuck at the "
            f"start means the LambdaCallback never ran or never reached the Variable."
        )
        assert abs(end - start) > 0.1, (
            f"teacher_temp moved by {abs(end - start)}; the two endpoints of this test's "
            f"schedule are 0.02 and 0.5, so a small delta means the guard has gone vacuous."
        )

    def test_a_plain_attribute_assignment_is_refused(self) -> None:
        """The other half of D-022: the silent no-op is now a loud failure."""
        from dl_techniques.losses.dino_loss import DINOLoss

        loss = DINOLoss(out_dim=8)
        with pytest.raises(AttributeError, match="read-only"):
            loss.teacher_temp = 0.5
