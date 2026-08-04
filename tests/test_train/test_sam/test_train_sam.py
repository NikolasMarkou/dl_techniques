"""
Guards for ``src/train/sam/train_sam.py`` -- the CLI, the config, and the
``argv -> config`` wiring.

The load-bearing guard here is :meth:`TestArgsToConfigWiring.
test_every_cli_flag_reaches_the_config_field_it_names`. This repository has a
RECORDED defect class -- a trainer's ``main()`` lists each config field by
hand, one line is omitted, and that CLI flag becomes a **silent no-op**: the
run completes, the artifact is wrong, and the factory-level tests all pass
because they never go through ``argv``. That guard drives EVERY parser flag
through the full ``argv -> parse_args -> explicitly_set_flags -> config`` path
with a sentinel value and reads the field back, so a dropped wiring row fails
by flag name.
"""

import argparse
from pathlib import Path
from typing import Any, List, Tuple

import pytest

from train.sam.train_sam import (
    CLI_TO_CONFIG,
    DERIVED_FIELDS,
    NON_CONFIG_DESTS,
    PATCH_SIZE,
    SMOKE_PRESET,
    VARIANT_IMAGE_SIZE,
    SAMTrainingConfig,
    build_parser,
    config_from_argv,
    create_sam,
    parse_arguments,
    resolved_output_dir,
)


# ---------------------------------------------------------------------------
# Sentinels
# ---------------------------------------------------------------------------
#: Flags whose legal values are constrained by ``__post_init__``, so a
#: "default + 7" sentinel would be refused for a reason unrelated to wiring.
SENTINEL_OVERRIDES = {
    # Must stay a multiple of PATCH_SIZE * MASK_DIVISOR = 64.
    "image_size": (["--image-size", "128"], 128),
}


def _sentinel_for(action: argparse.Action) -> Tuple[List[str], Any]:
    """
    Build ``(argv, expected_value)`` driving one flag to a NON-default value.

    A sentinel equal to the default would be satisfied by a flag that is wired
    to nothing at all -- the exact defect this module exists to catch -- so
    every branch here produces a value the default cannot be.

    Args:
        action: The argparse action to drive.

    Returns:
        ``(argv, expected)``: tokens to pass, and the value the config field
        must hold afterwards.

    Raises:
        AssertionError: if no sentinel rule covers the action's type. A new
            flag shape must extend this function rather than be skipped
            silently.
    """
    flag = action.option_strings[0]
    if action.dest in SENTINEL_OVERRIDES:
        return SENTINEL_OVERRIDES[action.dest]
    if isinstance(action, argparse.BooleanOptionalAction):
        # Drive it AWAY from its default, whichever way that is.
        if action.default:
            return [f"--no-{action.dest.replace('_', '-')}"], False
        return [flag], True
    if action.choices:
        alternative = [c for c in action.choices if c != action.default]
        assert alternative, f"{flag} has a single choice; no sentinel possible"
        return [flag, str(alternative[0])], alternative[0]
    if action.type is int:
        value = 7 if action.default in (None, 7) else int(action.default) + 7
        return [flag, str(value)], value
    if action.type is float:
        value = 0.5 if action.default in (None, 0.5) else float(action.default) + 0.5
        return [flag, str(value)], value
    if action.type is str:
        return [flag, f"sentinel_{action.dest}"], f"sentinel_{action.dest}"
    raise AssertionError(
        f"no sentinel rule for {flag} (type={action.type!r}, "
        f"action={type(action).__name__}); extend _sentinel_for"
    )


def _config_actions() -> List[argparse.Action]:
    """Every parser action that is supposed to reach the config."""
    return [
        action
        for action in build_parser()._actions
        if action.dest not in NON_CONFIG_DESTS
    ]


# ---------------------------------------------------------------------------
# The args -> config wiring (SC-10)
# ---------------------------------------------------------------------------
class TestArgsToConfigWiring:
    """The silent-no-op defect class, attacked through the real entry point."""

    def test_every_cli_flag_reaches_the_config_field_it_names(self) -> None:
        """
        Drive each flag with a non-default sentinel and read the field back.

        A flag whose wiring row is missing leaves its field at the default,
        which is never the sentinel, so it is named in the failure.
        """
        violations = []
        for action in _config_actions():
            argv, expected = _sentinel_for(action)
            field = CLI_TO_CONFIG.get(action.dest)
            if field is None:
                violations.append(
                    f"{action.option_strings[0]}: dest {action.dest!r} has no "
                    f"row in CLI_TO_CONFIG, so nothing carries it to the config"
                )
                continue
            # `--variant` is the one flag with a cross-field constraint: every
            # non-tiny variant is a 1024px model, so its sentinel needs the
            # matching image size or __post_init__ refuses the pair.
            extra = (
                ["--image-size", str(VARIANT_IMAGE_SIZE)]
                if action.dest == "variant"
                else []
            )
            config = config_from_argv(argv + extra)
            actual = getattr(config, field)
            if actual != expected:
                violations.append(
                    f"{action.option_strings[0]} -> config.{field}: expected "
                    f"{expected!r}, got {actual!r} (the flag is a SILENT NO-OP)"
                )
        assert not violations, "\n".join(violations)

    def test_every_cli_flag_is_wired_to_a_config_field(self) -> None:
        """Completeness in the argv direction: no flag without a wiring row."""
        unwired = sorted(
            action.dest
            for action in _config_actions()
            if action.dest not in CLI_TO_CONFIG
        )
        assert not unwired, (
            f"parser dests with no CLI_TO_CONFIG row: {unwired}. Add the row, "
            f"or list the dest in NON_CONFIG_DESTS if it is deliberately "
            f"process-level."
        )

    def test_every_config_field_is_reachable_from_the_cli(self) -> None:
        """
        Completeness in the config direction: no unreachable knob.

        A config field nothing can set is either dead or a flag someone forgot
        to add; both are worth failing on.
        """
        wired = set(CLI_TO_CONFIG.values())
        unreachable = sorted(
            f.name
            for f in SAMTrainingConfig.__dataclass_fields__.values()
            if f.name not in wired and f.name not in DERIVED_FIELDS
        )
        assert not unreachable, (
            f"config fields no CLI flag reaches: {unreachable}. Add a flag, or "
            f"declare the field in DERIVED_FIELDS."
        )

    def test_the_wiring_table_names_only_real_config_fields(self) -> None:
        """A row pointing at a typo'd field name would blow up at run time."""
        real = set(SAMTrainingConfig.__dataclass_fields__)
        bogus = sorted(set(CLI_TO_CONFIG.values()) - real)
        assert not bogus, f"CLI_TO_CONFIG rows naming no such field: {bogus}"

    def test_the_sentinels_actually_differ_from_the_defaults(self) -> None:
        """
        The guard's own instrument, RED-proofed.

        If a sentinel ever equalled its default, the wiring test above would
        pass against a completely unwired flag. This asserts that cannot happen.
        """
        for action in _config_actions():
            _, expected = _sentinel_for(action)
            assert expected != action.default, (
                f"{action.option_strings[0]}'s sentinel equals its default "
                f"({expected!r}); the wiring test would be vacuous for it"
            )


# ---------------------------------------------------------------------------
# `--help` (SC-10)
# ---------------------------------------------------------------------------
class TestHelpDoesNotTrain:
    """A trainer once started a 100-epoch job on ``--help``."""

    def test_help_is_a_help_action_and_exits_before_any_config_is_built(
        self,
    ) -> None:
        """
        Asserted on the ACTION and on the raised ``SystemExit``, never on the
        exit code alone: a script that trained for an hour and then exited 0
        would satisfy an exit-code-only assertion.
        """
        parser = build_parser()
        help_actions = [
            a for a in parser._actions if isinstance(a, argparse._HelpAction)
        ]
        assert len(help_actions) == 1
        assert "--help" in help_actions[0].option_strings

        with pytest.raises(SystemExit) as excinfo:
            parse_arguments(["--help"])
        assert excinfo.value.code == 0

    def test_help_is_not_a_config_field(self) -> None:
        assert "help" in NON_CONFIG_DESTS
        assert "help" not in CLI_TO_CONFIG


# ---------------------------------------------------------------------------
# `--smoke` (SC-10)
# ---------------------------------------------------------------------------
class TestSmokePreset:
    """What the preset changes, field by field."""

    def test_smoke_changes_exactly_the_documented_fields(self) -> None:
        """
        The field-by-field diff the plan asks for, asserted rather than
        described. Every differing field must be a SMOKE_PRESET key.
        """
        base = config_from_argv([])
        smoke = config_from_argv(["--smoke"])
        differing = {
            name: (getattr(base, name), getattr(smoke, name))
            for name in SAMTrainingConfig.__dataclass_fields__
            if getattr(base, name) != getattr(smoke, name)
        }
        # `smoke` itself and the timestamped `experiment_name` always differ.
        differing.pop("smoke")
        differing.pop("experiment_name", None)
        assert set(differing) == set(SMOKE_PRESET), (
            f"--smoke moved {sorted(differing)}, SMOKE_PRESET declares "
            f"{sorted(SMOKE_PRESET)}"
        )

    def test_the_preset_changes_how_much_not_what(self) -> None:
        """
        The distinction LESSONS.md records a preset silently violating: a smoke
        preset may shrink the measurement, never redefine it. So the source,
        the geometry, the round count, the loss weights and the seed must be
        BIT-IDENTICAL between the two configs.
        """
        base = config_from_argv([])
        smoke = config_from_argv(["--smoke"])
        for field in (
            "data_source", "image_size", "variant", "num_refinement_rounds",
            "multimask_output", "focal_weight", "dice_weight", "iou_weight",
            "include_box", "num_background_points", "max_instances", "seed",
            "learning_rate",
        ):
            assert getattr(base, field) == getattr(smoke, field), (
                f"--smoke changed {field}, which decides WHAT is measured, not "
                f"how precisely"
            )

    def test_an_explicitly_typed_flag_beats_the_preset(self) -> None:
        config = config_from_argv(["--smoke", "--epochs", "11"])
        assert config.epochs == 11
        assert config.smoke is True

    def test_an_explicitly_typed_DEFAULT_beats_the_preset(self) -> None:
        """
        The provenance property, and the reason ``explicitly_set_flags`` scans
        tokens instead of comparing values: a flag typed at its own parser
        default is indistinguishable from an omission in the Namespace, so a
        value-comparison implementation silently overrides it.
        """
        default_epochs = SAMTrainingConfig().epochs
        assert default_epochs != SMOKE_PRESET["epochs"], (
            "this test is vacuous unless the preset moves `epochs`"
        )
        config = config_from_argv(["--smoke", "--epochs", str(default_epochs)])
        assert config.epochs == default_epochs

    def test_an_omitted_flag_takes_the_preset(self) -> None:
        """The other half: without the token, the preset must win."""
        config = config_from_argv(["--smoke"])
        assert config.epochs == SMOKE_PRESET["epochs"]

    def test_the_preset_keys_are_real_config_fields(self) -> None:
        real = set(SAMTrainingConfig.__dataclass_fields__)
        assert set(SMOKE_PRESET) <= real


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------
class TestConfigValidation:
    """``__post_init__`` refuses configurations that would fail far away."""

    def test_unknown_data_source_is_refused_naming_the_known_ones(self) -> None:
        with pytest.raises(ValueError, match="unknown data_source"):
            SAMTrainingConfig(data_source="imagenet")

    def test_a_vit_variant_at_a_non_1024_image_size_is_refused(self) -> None:
        """
        ``SAM.from_variant`` hard-wires 1024px. Without this check the run dies
        much later, inside ``preprocess``, with a padding error naming neither
        the variant nor the flag.
        """
        with pytest.raises(ValueError, match="from_variant"):
            SAMTrainingConfig(variant="vit_b", image_size=256)

    def test_the_matching_pair_is_accepted(self) -> None:
        """Non-firing control: the guard is not simply refusing every vit_*."""
        config = SAMTrainingConfig(
            variant="vit_b", image_size=VARIANT_IMAGE_SIZE
        )
        assert config.variant == "vit_b"

    @pytest.mark.parametrize("image_size", [255, 200, 100, 0, -64])
    def test_an_image_size_off_the_mask_grid_is_refused(
        self, image_size: int
    ) -> None:
        """
        The GT mask is emitted at ``image_size // 4`` and the model's
        ``low_res_logits`` at ``4 * (image_size // 16)``. Those agree only when
        ``image_size`` is a multiple of 64; anything else is a shape mismatch
        deep inside the loss.
        """
        with pytest.raises(ValueError, match="multiple of"):
            SAMTrainingConfig(image_size=image_size)

    @pytest.mark.parametrize("image_size", [64, 128, 256, 1024])
    def test_a_valid_image_size_is_accepted(self, image_size: int) -> None:
        assert SAMTrainingConfig(image_size=image_size).image_size == image_size

    @pytest.mark.parametrize(
        "kwargs,pattern",
        [
            ({"batch_size": 0}, "batch_size"),
            ({"epochs": 0}, "epochs"),
            ({"steps_per_epoch": 0}, "steps_per_epoch"),
            ({"num_train_samples": 0}, "num_train_samples"),
            ({"num_val_samples": 0}, "num_val_samples"),
            ({"learning_rate": 0.0}, "learning_rate"),
            ({"num_refinement_rounds": 0}, "num_refinement_rounds"),
            ({"max_instances": 0}, "max_instances"),
            ({"num_background_points": -1}, "num_background_points"),
            ({"focal_weight": -1.0}, "non-negative"),
            ({"early_stopping_patience": 0}, "early_stopping_patience"),
        ],
    )
    def test_out_of_range_values_are_refused(
        self, kwargs: dict, pattern: str
    ) -> None:
        with pytest.raises(ValueError, match=pattern):
            SAMTrainingConfig(**kwargs)

    def test_steps_per_epoch_none_is_legal(self) -> None:
        """None means "one full pass"; it must not trip the > 0 check."""
        assert SAMTrainingConfig(steps_per_epoch=None).steps_per_epoch is None

    def test_experiment_name_is_derived_when_omitted_and_kept_when_given(
        self,
    ) -> None:
        derived = SAMTrainingConfig().experiment_name
        assert derived is not None and derived.startswith("sam_tiny_")
        assert SAMTrainingConfig(experiment_name="mine").experiment_name == "mine"


# ---------------------------------------------------------------------------
# Output location
# ---------------------------------------------------------------------------
class TestOutputDirectory:
    """Repo-root ``results/``, never ``src/results/``, never the cwd's."""

    def test_a_relative_output_dir_resolves_against_the_repo_root(self) -> None:
        config = SAMTrainingConfig(
            output_dir="results", experiment_name="probe_run"
        )
        resolved = resolved_output_dir(config)
        repo_root = Path(__file__).resolve().parents[3]
        assert resolved == repo_root / "results" / "probe_run"
        assert resolved.is_absolute()

    def test_the_resolved_path_is_not_under_src(self) -> None:
        """
        The exact mistake the repo convention names: `python -m` resolves from
        any cwd, so a bare relative path can land in `src/results/`.
        """
        resolved = resolved_output_dir(SAMTrainingConfig(experiment_name="x"))
        assert "src" not in resolved.parts, resolved

    def test_an_absolute_output_dir_is_used_verbatim(self, tmp_path) -> None:
        config = SAMTrainingConfig(
            output_dir=str(tmp_path), experiment_name="run"
        )
        assert resolved_output_dir(config) == tmp_path / "run"


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------
class TestModelConstruction:
    """The trainer builds the object the model gate measures."""

    @staticmethod
    def _layout(sam) -> Tuple[int, Tuple[Any, ...]]:
        """
        A geometry fingerprint of a BUILT-but-not-called SAM.

        Deliberately NOT the absolute ``202 / 321,862`` constants: those are
        only reached AFTER a forward pass, because 64 of the fixture's weights
        are created lazily on the first call (D-018 measured exactly this and
        nearly mis-certified a change by sampling at the wrong moment). What is
        compared instead is this trainer's SAM against the gated fixture's SAM,
        both sampled at the same moment.
        """
        return sam.count_params(), tuple(
            sorted(tuple(w.shape) for w in sam.weights)
        )

    def test_the_tiny_variant_matches_the_gated_fixture_geometry(self) -> None:
        """
        The trainer must build the SAME object the model gate measures. If it
        built some other geometry, every number
        `tests/test_models/test_sam/test_correctness.py` proves would be about
        a different model than the one being trained.
        """
        from tests.test_models.test_sam.test_correctness import (
            IMG_SIZE,
            build_reduced_sam,
        )

        trainer_sam = create_sam(SAMTrainingConfig(image_size=IMG_SIZE))
        trainer_sam.build(None)
        fixture_sam = build_reduced_sam()
        fixture_sam.build(None)
        assert self._layout(trainer_sam) == self._layout(fixture_sam)

    def test_the_geometry_fingerprint_can_tell_two_geometries_apart(
        self,
    ) -> None:
        """
        The control, without which the comparison above could pass on any two
        models whose weight lists happen to be the same length.
        """
        from tests.test_models.test_sam.test_correctness import (
            build_reduced_sam,
        )

        other = create_sam(SAMTrainingConfig(image_size=128))
        other.build(None)
        fixture_sam = build_reduced_sam()
        fixture_sam.build(None)
        assert self._layout(other) != self._layout(fixture_sam)

    def test_the_tiny_encoder_uses_real_sam_patch_and_window_geometry(
        self,
    ) -> None:
        """Reduced WIDTH, not reduced geometry -- the paths under test differ."""
        sam = create_sam(SAMTrainingConfig(image_size=256))
        assert sam.image_encoder.patch_size == PATCH_SIZE
        assert sam.image_encoder.window_size == 14
        assert sam.image_encoder.use_rel_pos is True
        assert len(sam.image_encoder.global_attn_indexes) > 0

    def test_the_tiny_prompt_encoder_grid_tracks_the_image_size(self) -> None:
        for image_size in (128, 256):
            sam = create_sam(SAMTrainingConfig(image_size=image_size))
            grid = image_size // PATCH_SIZE
            assert sam.prompt_encoder.image_embedding_size == (grid, grid)
            assert sam.prompt_encoder.input_image_size == (
                image_size,
                image_size,
            )
