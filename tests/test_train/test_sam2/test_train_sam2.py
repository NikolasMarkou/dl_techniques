"""Guards for ``src/train/sam2/train_sam2.py`` -- the CLI, the config, the
``argv -> config`` wiring and the ``config -> consumer`` wiring.

Two load-bearing guards, both attacking the same RECORDED defect class from
opposite ends.

``TestArgsToConfigWiring.test_every_cli_flag_reaches_the_config_field_it_names``
drives EVERY parser flag with a non-default sentinel through the full
``argv -> parse_args -> explicitly_set_flags -> config`` path and reads the
field back, so a dropped wiring row fails by flag name. This repository shipped
exactly that defect once (`train/bfunet`: ``--high-freq-blocks`` and
``--filter-multiplier`` became silent no-ops).

``TestConfigToConsumptionWiring`` then goes ONE layer further, because the first
guard is structurally blind there: SAM 1's adversarial review replaced
``config.num_background_points`` with a literal ``0`` inside ``create_dataset``
and all 45 argv-level tests still passed. Every field of
:class:`SAM2TrainingConfig` is therefore either PROBED against a targeted
observable or carries an honest DECLARATION saying why it is not observed here.

Device: this module is written to run on the gate device
(``CUDA_VISIBLE_DEVICES=1``); every claim it makes about ``jit_compile`` is a
claim about that device, and is stated as such at the test.
"""

import argparse
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union
from unittest import mock

import keras
import numpy as np
import pytest

import train.sam2.train_sam2 as train_sam2_module
from dl_techniques.models.SAM.SAM2.model import create_sam2
from dl_techniques.models.SAM.SAM2.training_model import (
    OUTPUT_KEYS,
    SAM2TrainingModel,
    SAM2_IOU_SUPERVISION,
    SAM2_LOW_RES_LOGITS,
    SAM2_OBJECT_SCORE_LOGITS,
)
from train.sam2.train_sam2 import (
    CLI_TO_CONFIG,
    DERIVED_FIELDS,
    HIERA_L_PARAMETERS,
    NON_CONFIG_DESTS,
    SMOKE_PRESET,
    VARIANTS,
    SAM2TrainingConfig,
    build_parser,
    config_from_argv,
    create_dataset,
    OPEN_GATE_SCORE_BIAS,
    create_sam2_training_model,
    open_object_score_gate,
    parse_arguments,
    resolved_output_dir,
    train_sam2,
    variant_image_size,
)


# R-038 closure -- plan-2026-08-22T035419-a11304c8 / D-251.
# Keras `trainers/epoch_iterator.py:151`. These tests run the REAL trainer over
# a deliberately tiny synthetic corpus while `steps_per_epoch` comes from the
# shipped config, so the iterator is legitimately exhausted before the epoch
# ends. Padding the corpus to match would change what the test measures (the
# config -> `fit()` wiring), so the advisory is suppressed HERE only; a real
# starved input in any other module still fails under `error::UserWarning`.
pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:Your input ran out of data:UserWarning"),
]


# ---------------------------------------------------------------------------
# Sentinels
# ---------------------------------------------------------------------------
#: Flags whose legal values are constrained by ``__post_init__``, so a
#: "default + 7" sentinel would be refused for a reason unrelated to wiring.
SENTINEL_OVERRIDES: Dict[str, Tuple[List[str], Any]] = {
    # The default T is 4, so `4 + 7 = 11` frames is legal but slow to
    # construct; 2 is the smallest genuinely-multi-frame clip.
    "num_frames": (["--num-frames", "2"], 2),
    # Must stay inside [0, num_frames - 1] = [0, 3] at the default T.
    "occlusion_frames": (["--occlusion-frames", "2"], 2),
    # Must leave room for the window and never be frame 0.
    "occlusion_start": (["--occlusion-start", "2"], 2),
}


def _sentinel_for(action: argparse.Action) -> Tuple[List[str], Any]:
    """Build ``(argv, expected_value)`` driving one flag to a NON-default value.

    A sentinel equal to the default would be satisfied by a flag wired to
    nothing at all -- the exact defect this module exists to catch -- so every
    branch here produces a value the default cannot be.

    :param action: The argparse action to drive.
    :type action: argparse.Action
    :return: ``(argv, expected)``: tokens to pass, and the value the config
        field must hold afterwards.
    :rtype: Tuple[List[str], Any]
    :raises AssertionError: If no sentinel rule covers the action's shape. A
        new flag shape must extend this function rather than be skipped
        silently.
    """
    flag = action.option_strings[0]
    if action.dest in SENTINEL_OVERRIDES:
        return SENTINEL_OVERRIDES[action.dest]
    if isinstance(action, argparse.BooleanOptionalAction):
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
# The args -> config wiring
# ---------------------------------------------------------------------------
class TestArgsToConfigWiring:
    """The silent-no-op defect class, attacked through the real entry point."""

    def test_every_cli_flag_reaches_the_config_field_it_names(self) -> None:
        """Drive each flag with a non-default sentinel; read the field back.

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
            config = config_from_argv(argv)
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
        """Completeness in the config direction: no unreachable knob."""
        wired = set(CLI_TO_CONFIG.values())
        unreachable = sorted(
            f.name
            for f in SAM2TrainingConfig.__dataclass_fields__.values()
            if f.name not in wired and f.name not in DERIVED_FIELDS
        )
        assert not unreachable, (
            f"config fields no CLI flag reaches: {unreachable}. Add a flag, or "
            f"declare the field in DERIVED_FIELDS."
        )

    def test_the_wiring_table_names_only_real_config_fields(self) -> None:
        """A row pointing at a typo'd field name would blow up at run time."""
        real = set(SAM2TrainingConfig.__dataclass_fields__)
        bogus = sorted(set(CLI_TO_CONFIG.values()) - real)
        assert not bogus, f"CLI_TO_CONFIG rows naming no such field: {bogus}"

    def test_the_sentinels_actually_differ_from_the_defaults(self) -> None:
        """The guard's own instrument, RED-proofed.

        If a sentinel ever equalled its default, the wiring test above would
        pass against a completely unwired flag.
        """
        for action in _config_actions():
            _, expected = _sentinel_for(action)
            assert expected != action.default, (
                f"{action.option_strings[0]}'s sentinel equals its default "
                f"({expected!r}); the wiring test would be vacuous for it"
            )

    def test_one_flag_at_a_time_is_what_the_wiring_test_drives(self) -> None:
        """The guard drives ONE flag per assertion, and this pins that.

        LESSONS: two mutations landing on one assertion prove it twice and the
        other zero times. Here the inverse: a sentinel argv that carried extra
        flags would let one wiring row's failure be masked by another row's
        success. Every sentinel is a single flag (plus its value).
        """
        for action in _config_actions():
            argv, _ = _sentinel_for(action)
            flags = [token for token in argv if token.startswith("--")]
            assert len(flags) == 1, (
                f"{action.option_strings[0]}'s sentinel argv drives "
                f"{flags}; one mutation must land on one assertion"
            )


# ---------------------------------------------------------------------------
# `--help`
# ---------------------------------------------------------------------------
class TestHelpDoesNotTrain:
    """A trainer once started a 100-epoch job on ``--help``."""

    def test_help_is_a_help_action_and_exits_before_any_config_is_built(
            self) -> None:
        """Asserted on the ACTION and on the raised ``SystemExit``.

        Never on the exit code alone: a script that trained for an hour and
        then exited 0 would satisfy an exit-code-only assertion.
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

    def test_the_parser_carries_no_short_option_but_h(self) -> None:
        """``explicitly_set_flags`` REFUSES any other short option.

        It cannot see attached (``-b8``) or grouped (``-vb 8``) forms and would
        report a typed flag as not-typed -- verbatim the regression that
        helper's own history exists to prevent. Building the parser through it
        is the executable proof.
        """
        parser = build_parser()
        from train.common.args import explicitly_set_flags

        assert explicitly_set_flags(parser, ["--epochs", "3"]) == {"epochs"}


# ---------------------------------------------------------------------------
# `--smoke`
# ---------------------------------------------------------------------------
class TestSmokePreset:
    """What the preset changes, field by field."""

    def test_smoke_changes_exactly_the_documented_fields(self) -> None:
        """The field-by-field diff, asserted rather than described."""
        base = config_from_argv([])
        smoke = config_from_argv(["--smoke"])
        differing = {
            name: (getattr(base, name), getattr(smoke, name))
            for name in SAM2TrainingConfig.__dataclass_fields__
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
        """The distinction LESSONS.md records a preset silently violating.

        A smoke preset may shrink the measurement, never redefine it. So the
        clip geometry, the occlusion layout, the variant, the loss weights and
        the seed must be BIT-IDENTICAL between the two configs. ``num_frames``
        is the one that matters most here: it is the unrolled loop bound, so
        moving it would make the smoke run exercise a structurally different
        graph from the real run it is supposed to be a smaller version of.
        """
        base = config_from_argv([])
        smoke = config_from_argv(["--smoke"])
        for field in (
                "num_frames", "occlusion_frames", "occlusion_start", "variant",
                "mask_weight", "object_score_weight", "iou_weight",
                "include_box", "num_background_points", "seed",
                "learning_rate", "steps_per_epoch", "output_dir",
        ):
            assert getattr(base, field) == getattr(smoke, field), (
                f"--smoke changed {field}, which decides WHAT is measured, not "
                f"how precisely"
            )

    def test_the_preset_never_touches_a_field_that_shapes_the_graph(
            self) -> None:
        """The same claim as a SET operation, so a NEW preset key is caught.

        The test above enumerates fields; this one enumerates the preset. A key
        added to SMOKE_PRESET that shapes the model or the data semantics fails
        here even if nobody remembers to extend the list above.
        """
        shaping = {
            "num_frames", "occlusion_frames", "occlusion_start", "variant",
            "mask_weight", "object_score_weight", "iou_weight", "seed",
            "include_box", "num_background_points", "learning_rate",
        }
        offenders = sorted(shaping & set(SMOKE_PRESET))
        assert not offenders, (
            f"SMOKE_PRESET declares {offenders}, which change WHAT is "
            f"measured. A smoke preset may only change how much."
        )

    def test_an_explicitly_typed_flag_beats_the_preset(self) -> None:
        config = config_from_argv(["--smoke", "--epochs", "11"])
        assert config.epochs == 11
        assert config.smoke is True

    def test_an_explicitly_typed_DEFAULT_beats_the_preset(self) -> None:
        """The provenance property, and why ``explicitly_set_flags`` exists.

        A flag typed at its own parser default is indistinguishable from an
        omission in the Namespace, so a value-vs-default implementation
        silently overrides it.
        """
        default_epochs = SAM2TrainingConfig().epochs
        assert default_epochs != SMOKE_PRESET["epochs"], (
            "this test is vacuous unless the preset moves `epochs`"
        )
        config = config_from_argv(["--smoke", "--epochs", str(default_epochs)])
        assert config.epochs == default_epochs

    def test_every_preset_field_can_be_typed_at_its_own_default_and_win(
            self) -> None:
        """The provenance property across the WHOLE preset, not one field.

        ``epochs`` alone would leave four preset fields unproved, and a
        provenance bug need not be uniform across flags.
        """
        defaults = SAM2TrainingConfig()
        for field in SMOKE_PRESET:
            default_value = getattr(defaults, field)
            assert default_value != SMOKE_PRESET[field], (
                f"{field}'s preset value equals its default; its provenance "
                f"arm would be vacuous"
            )
            flag = "--" + field.replace("_", "-")
            config = config_from_argv(
                ["--smoke", flag, str(default_value)]
            )
            assert getattr(config, field) == default_value, (
                f"{flag} typed at its own default ({default_value!r}) lost to "
                f"the preset; provenance is being computed by VALUE, not by "
                f"whether the token was typed"
            )

    def test_an_omitted_flag_takes_the_preset(self) -> None:
        """The other half: without the token, the preset must win."""
        config = config_from_argv(["--smoke"])
        for field, value in SMOKE_PRESET.items():
            assert getattr(config, field) == value

    def test_the_preset_keys_are_real_config_fields(self) -> None:
        assert set(SMOKE_PRESET) <= set(SAM2TrainingConfig.__dataclass_fields__)

    def test_smoke_refuses_hiera_l_naming_the_parameter_cost(self) -> None:
        """221M parameters at 1024px is not a smoke run on a 12 GB card."""
        with pytest.raises(ValueError, match=f"{HIERA_L_PARAMETERS:,}"):
            config_from_argv(["--smoke", "--variant", "hiera_l"])

    def test_hiera_l_without_smoke_is_accepted(self) -> None:
        """The non-firing control: the refusal is about the PAIR."""
        config = config_from_argv(["--variant", "hiera_l"])
        assert config.variant == "hiera_l"
        assert config.image_size == 1024


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------
class TestConfigValidation:
    """``__post_init__`` refuses configurations that would fail far away."""

    def test_unknown_variant_is_refused_naming_the_known_ones(self) -> None:
        with pytest.raises(ValueError, match="unknown variant"):
            SAM2TrainingConfig(variant="hiera_b")

    def test_the_image_size_is_read_from_the_models_own_table(self) -> None:
        """One home for the geometry -- the variant table, not this trainer.

        If the trainer restated 64, a variant-table change would silently give
        a data pipeline at one resolution and a model at another, which is a
        shape error deep inside the encoder rather than a config error here.
        """
        from dl_techniques.models.SAM.SAM2.hiera import Hiera

        for variant in VARIANTS:
            assert variant_image_size(variant) == (
                Hiera.MODEL_VARIANTS[variant]["image_size"]
            )
            assert SAM2TrainingConfig(variant=variant).image_size == (
                Hiera.MODEL_VARIANTS[variant]["image_size"]
            )

    def test_the_mask_grid_matches_what_the_model_emits(self) -> None:
        """``mask_size`` must equal ``feature_grid * 4``, or the loss mismatches.

        The two are derived independently -- one from this config, one from the
        model's memory stride -- and nothing else in the trainer compares them.
        """
        from dl_techniques.models.SAM.SAM2.model import MEMORY_STRIDE

        for variant in VARIANTS:
            config = SAM2TrainingConfig(variant=variant)
            assert config.mask_size == (
                config.image_size // MEMORY_STRIDE
            ) * 4

    @pytest.mark.parametrize(
        "kwargs,pattern",
        [
            ({"num_frames": 0}, "num_frames"),
            ({"num_clips_train": 0}, "num_clips_train"),
            ({"num_clips_val": 0}, "num_clips_val"),
            ({"num_background_points": -1}, "num_background_points"),
            ({"batch_size": 0}, "batch_size"),
            ({"epochs": 0}, "epochs"),
            ({"steps_per_epoch": 0}, "steps_per_epoch"),
            ({"learning_rate": 0.0}, "learning_rate"),
            ({"mask_weight": -1.0}, "non-negative"),
            ({"iou_weight": -1.0}, "non-negative"),
            ({"early_stopping_patience": 0}, "early_stopping_patience"),
        ],
    )
    def test_out_of_range_values_are_refused(
            self, kwargs: Dict[str, Any], pattern: str) -> None:
        with pytest.raises(ValueError, match=pattern):
            SAM2TrainingConfig(**kwargs)

    def test_a_zero_object_score_weight_is_refused_not_silently_accepted(
            self) -> None:
        """Weight 0 on the BCE is a frozen occlusion head with no symptom.

        Every consumer of ``object_score_logits`` in this package thresholds it
        hard at ``> 0``, so the head has no other differentiable consumer. The
        run would complete with a finite, falling and meaningless loss.
        """
        with pytest.raises(ValueError, match="object_score_weight must be > 0"):
            SAM2TrainingConfig(object_score_weight=0.0)

    @pytest.mark.parametrize(
        "num_frames,occlusion_frames", [(4, 4), (4, 5), (1, 1), (2, -1)]
    )
    def test_an_impossible_occlusion_window_is_refused(
            self, num_frames: int, occlusion_frames: int) -> None:
        """The bound is the DATA MODULE's, imported, not restated here."""
        with pytest.raises(ValueError, match="occlusion_frames"):
            SAM2TrainingConfig(
                num_frames=num_frames, occlusion_frames=occlusion_frames
            )

    def test_an_occlusion_window_reaching_frame_zero_is_refused(self) -> None:
        """Frame 0 carries the prompt; an occluded frame 0 has none to give."""
        with pytest.raises(ValueError, match="FRAME 0"):
            SAM2TrainingConfig(
                num_frames=4, occlusion_frames=2, occlusion_start=0
            )

    @pytest.mark.parametrize("num_frames", [1, 2, 4, 8])
    def test_a_legal_clip_length_is_accepted(self, num_frames: int) -> None:
        """The non-firing control, including ``T = 1``: the image-path run."""
        config = SAM2TrainingConfig(
            num_frames=num_frames,
            occlusion_frames=1 if num_frames > 1 else 0,
        )
        assert config.num_frames == num_frames

    def test_zero_occlusion_frames_is_legal(self) -> None:
        """A clip the object is visible in throughout is a valid control."""
        assert SAM2TrainingConfig(occlusion_frames=0).occlusion_frames == 0

    def test_steps_per_epoch_none_is_legal(self) -> None:
        assert SAM2TrainingConfig(steps_per_epoch=None).steps_per_epoch is None

    def test_experiment_name_is_derived_when_omitted_and_kept_when_given(
            self) -> None:
        derived = SAM2TrainingConfig().experiment_name
        assert derived is not None and derived.startswith("sam2_tiny_")
        assert SAM2TrainingConfig(
            experiment_name="mine").experiment_name == "mine"


# ---------------------------------------------------------------------------
# Output location
# ---------------------------------------------------------------------------
class TestOutputDirectory:
    """Repo-root ``results/``, never ``src/results/``, never the cwd's."""

    def test_a_relative_output_dir_resolves_against_the_repo_root(self) -> None:
        config = SAM2TrainingConfig(
            output_dir="results", experiment_name="probe_run")
        resolved = resolved_output_dir(config)
        repo_root = Path(__file__).resolve().parents[3]
        assert resolved == repo_root / "results" / "probe_run"
        assert resolved.is_absolute()

    def test_the_resolved_path_is_not_under_src(self) -> None:
        """`python -m` resolves from any cwd, so a bare relative path can land
        in `src/results/` -- the exact place the repo convention names."""
        resolved = resolved_output_dir(SAM2TrainingConfig(experiment_name="x"))
        assert "src" not in resolved.parts, resolved

    def test_an_absolute_output_dir_is_used_verbatim(self, tmp_path) -> None:
        config = SAM2TrainingConfig(
            output_dir=str(tmp_path), experiment_name="run")
        assert resolved_output_dir(config) == tmp_path / "run"


# ---------------------------------------------------------------------------
# Compilation: the three losses and the mandatory `jit_compile=False`
# ---------------------------------------------------------------------------
#: A small, fast, legal config. ``T = 2`` is the smallest genuinely multi-frame
#: clip; ``T = 1`` is covered separately as the image-path control.
_PROBE_BASE: Dict[str, Any] = {
    "num_frames": 2,
    "occlusion_frames": 1,
    "occlusion_start": 1,
    "num_clips_train": 2,
    "num_clips_val": 2,
    "batch_size": 1,
    "epochs": 1,
    "seed": 0,
}


def _probe_config(**overrides: Any) -> SAM2TrainingConfig:
    """A small, fast, legal config with ``overrides`` applied."""
    return SAM2TrainingConfig(**{**_PROBE_BASE, **overrides})


class TestCompilation:
    """What the trainer hands ``fit()``, asserted on the compiled model."""

    def test_the_compiled_loss_covers_every_output_key(self) -> None:
        """The three keys come from the MODEL's constants, not from strings.

        H-5: a dict ``y_pred`` needs ``loss=`` keyed to the output names, and
        nothing in Keras checks the two sets against each other. Dropping the
        object-score key is the silent case that matters: the score head has no
        other differentiable consumer, so the run trains a frozen occlusion
        head with a finite, falling, meaningless loss.
        """
        model = create_sam2_training_model(_probe_config())
        assert set(model.loss) == set(OUTPUT_KEYS)
        assert set(model.loss) == {
            SAM2_LOW_RES_LOGITS,
            SAM2_OBJECT_SCORE_LOGITS,
            SAM2_IOU_SUPERVISION,
        }

    def test_the_object_score_term_is_a_binary_crossentropy_on_logits(
            self) -> None:
        """Upstream's ``loss_class`` at ``focal_gamma_obj_score = 0.0`` and
        ``focal_alpha_obj_score = -1.0`` IS a plain BCE, and the head emits
        LOGITS -- ``from_logits=False`` would silently score sigmoids twice."""
        model = create_sam2_training_model(_probe_config())
        bce = model.loss[SAM2_OBJECT_SCORE_LOGITS]
        assert isinstance(bce, keras.losses.BinaryCrossentropy)
        assert bce.from_logits is True

    def test_jit_compile_is_false_on_the_compiled_model(self) -> None:
        """MANDATORY on the gate device (D-055), and asserted, not assumed.

        MEASURED on GPU 1 (RTX 4070): Keras 3.8's ``fit()`` defaults to
        ``jit_compile='auto'``, which selects XLA on a GPU, and ``Hiera``'s stem
        bicubic ``ResizeBicubic`` has no XLA GPU kernel -- the first ``fit()``
        step raises ``InvalidArgumentError: ... No registered 'ResizeBicubic'
        OpKernel for XLA_GPU_JIT devices``. On CPU the same removal is
        HARMLESS, so this assertion is the only thing standing between a
        CPU-green suite and a GPU-dead trainer.
        """
        model = create_sam2_training_model(_probe_config())
        assert model.jit_compile is False

    def test_the_loss_weights_reach_the_compiled_loss(self) -> None:
        """Behavioural, and it needs NO forward pass.

        ``compute_loss`` is fed a fixed synthetic ``(y, y_pred)`` pair, so the
        only thing that can move the number is the loss configuration the
        config produced.
        """
        base = _obs_compiled_loss_value(_probe_config())
        for field, value in (
                ("mask_weight", 7.0),
                ("object_score_weight", 7.0),
                ("iou_weight", 7.0),
        ):
            driven = _obs_compiled_loss_value(_probe_config(**{field: value}))
            assert driven != base, (
                f"config.{field} does not reach the compiled loss: driving it "
                f"to {value} left the loss at {base}"
            )


class TestTheBatchAxisIsStatic:
    """A dynamic batch axis is not slow here -- it does not RUN.

    MEASURED at step 5 on GPU 1 (RTX 4070), and this is the reason
    ``build_sam2_video_dataset`` batches with ``drop_remainder=True``: with a
    ``None`` batch, ``SAM2TrainingModel._unflatten`` sees the image encoder's
    positional encodings traced as ``(None, 16, 16, None)`` -- the CHANNEL axis
    unknown, not merely the batch -- and ``SAM2._decode`` reads
    ``int(pointer_tokens.shape[1])`` for the D-044 conditional gather. Both
    raise ``TypeError: int() argument must be ... not 'NoneType'`` at the first
    ``fit()`` step, from inside ``models/SAM/SAM2/model.py``, which this iteration
    must leave byte-unchanged.

    A ``fit()`` over numpy arrays never sees any of it, because Keras traces
    those with a STATIC batch -- which is exactly why thirty green model-level
    guards said nothing about it.
    """

    def test_both_of_the_trainers_datasets_have_a_static_batch_axis(
            self) -> None:
        config = _probe_config(batch_size=2, num_clips_train=4,
                               num_clips_val=2)
        for dataset in create_dataset(config):
            batch_axis = dataset.element_spec[0]["image"].shape[0]
            assert batch_axis is not None, (
                "the batch axis is dynamic; the first fit() step will die "
                "inside SAM2._decode with a TypeError on None"
            )
            assert int(batch_axis) == config.batch_size

    def test_a_dynamic_batch_axis_dies_at_the_first_fit_step(self) -> None:
        """The RED proof, EXECUTED rather than described.

        Without it, ``drop_remainder=True`` reads as a throughput tweak that a
        later reader would drop.
        """
        config = _probe_config()
        train_dataset, _ = create_dataset(config)
        dynamic = train_dataset.unbatch().batch(config.batch_size)
        assert dynamic.element_spec[0]["image"].shape[0] is None
        model = create_sam2_training_model(config)
        with pytest.raises(TypeError, match="NoneType"):
            model.fit(dynamic, epochs=1, verbose=0)

    def test_a_split_smaller_than_one_batch_is_refused(self) -> None:
        """Dropping the remainder makes an under-sized split yield NOTHING.

        An empty validation set is not a smaller measurement; it fails a whole
        epoch in, on ``monitor='val_loss'``.
        """
        with pytest.raises(ValueError, match="drop_remainder"):
            _probe_config(num_clips_val=1, batch_size=2)

    def test_the_matching_pair_is_accepted(self) -> None:
        """Non-firing control: the refusal is about the RATIO."""
        assert _probe_config(
            num_clips_val=2, batch_size=2).num_clips_val == 2


# ---------------------------------------------------------------------------
# The `config -> consumption` half of the wiring
# ---------------------------------------------------------------------------
# `TestArgsToConfigWiring` proves `argv -> config`. It is BLIND one layer
# downstream: SAM 1's adversarial review edited `create_dataset` to pass a
# literal `num_background_points=0` instead of the config field and all 45
# tests in that module still passed. So every field of `SAM2TrainingConfig` is
# listed below with HOW its arrival at a consumer is proved. A field is either:
#   * PROBED   -- `(base_overrides, driven_overrides, observable)`: drive it to
#                 a non-default value and assert a TARGETED observable moves.
#                 Targeted matters: a whole-batch digest would move for a dozen
#                 unrelated reasons and could not isolate the field.
#   * DECLARED -- a string saying, plainly, that the field's effect is NOT
#                 observed here, and why. Declaring is honest; omitting is the
#                 defect.
def _first_train_batch(config: SAM2TrainingConfig) -> Tuple[Any, Any]:
    """``(inputs, targets)`` of the first batch the trainer's own path yields."""
    train_dataset, _ = create_dataset(config)
    return next(iter(train_dataset))


def _obs_clip_shape(config: SAM2TrainingConfig) -> Any:
    inputs, _ = _first_train_batch(config)
    return tuple(inputs["image"].shape[:2])


def _obs_batch_size(config: SAM2TrainingConfig) -> Any:
    inputs, _ = _first_train_batch(config)
    return int(inputs["image"].shape[0])


def _obs_train_batches(config: SAM2TrainingConfig) -> Any:
    train_dataset, _ = create_dataset(config)
    return sum(1 for _ in train_dataset)


def _obs_val_batches(config: SAM2TrainingConfig) -> Any:
    _, val_dataset = create_dataset(config)
    return sum(1 for _ in val_dataset)


def _obs_empty_frame_count(config: SAM2TrainingConfig) -> Any:
    """How many ground-truth frames of the first clip are all zeros."""
    _, targets = _first_train_batch(config)
    masks = np.asarray(targets[SAM2_LOW_RES_LOGITS])[0]
    return int(sum(1 for frame in masks if frame.sum() == 0))


def _obs_empty_frame_indices(config: SAM2TrainingConfig) -> Any:
    """WHICH frames of the first clip are empty -- the window's placement."""
    _, targets = _first_train_batch(config)
    masks = np.asarray(targets[SAM2_LOW_RES_LOGITS])[0]
    return tuple(t for t, frame in enumerate(masks) if frame.sum() == 0)


def _obs_prompt_points(config: SAM2TrainingConfig) -> Any:
    inputs, _ = _first_train_batch(config)
    return int(inputs["point_labels"].shape[1])


def _obs_box_key(config: SAM2TrainingConfig) -> Any:
    inputs, _ = _first_train_batch(config)
    return "boxes" in inputs


def _obs_image_content(config: SAM2TrainingConfig) -> Any:
    """Pixel digest of the first clip -- the seed's only visible effect."""
    inputs, _ = _first_train_batch(config)
    return float(np.asarray(inputs["image"]).sum())


def _obs_compiled_loss_value(config: SAM2TrainingConfig) -> Any:
    """The number the compiled loss actually returns, on a FIXED pair."""
    model = create_sam2_training_model(config)
    rng = np.random.RandomState(0)
    frames, grid = config.num_frames, config.mask_size
    y_pred = {
        SAM2_LOW_RES_LOGITS: rng.uniform(
            -2.0, 2.0, (2, frames, grid, grid)).astype("float32"),
        SAM2_OBJECT_SCORE_LOGITS: rng.uniform(
            -2.0, 2.0, (2, frames, 1)).astype("float32"),
        SAM2_IOU_SUPERVISION: rng.uniform(
            0.0, 1.0, (2, frames, 2)).astype("float32"),
    }
    y_true = {
        SAM2_LOW_RES_LOGITS: (
            rng.uniform(0.0, 1.0, (2, frames, grid, grid)) > 0.5
        ).astype("float32"),
        SAM2_OBJECT_SCORE_LOGITS: np.ones((2, frames, 1), dtype="float32"),
        SAM2_IOU_SUPERVISION: np.zeros((2, frames, 2), dtype="float32"),
    }
    return round(
        float(model.compute_loss(x=None, y=y_true, y_pred=y_pred)), 6)


def _obs_learning_rate(config: SAM2TrainingConfig) -> Any:
    return float(create_sam2_training_model(config).optimizer.learning_rate)


def _obs_wrapper_frames(config: SAM2TrainingConfig) -> Any:
    """The MODEL's unrolled loop bound -- the other side of ``num_frames``."""
    return create_sam2_training_model(config).num_frames


def _obs_variant_call(config: SAM2TrainingConfig) -> Any:
    """Which variant name reaches ``create_sam2`` -- the call site.

    Building a real ``hiera_l`` is 220,941,537 parameters at 1024px, which no
    ordinary gate can afford; the ``tiny`` branch is covered behaviourally
    everywhere else in this module.
    """
    seen: List[Any] = []
    # Captured BEFORE the patch: calling the module attribute inside the
    # recorder would call the recorder.
    real_create_sam2 = train_sam2_module.create_sam2

    def recorder(variant: str, **kwargs: Any) -> Any:
        seen.append(variant)
        return real_create_sam2("tiny", **kwargs)

    with mock.patch.object(train_sam2_module, "create_sam2", recorder):
        create_sam2_training_model(config)
    return seen


def _record_fit(config: SAM2TrainingConfig, tmp_dir: Path) -> Dict[str, Any]:
    """Run ``train_sam2`` with ``fit`` replaced by a recorder.

    Everything up to the training call is real -- the output directory, the
    datasets, the compiled model and the callback list -- so the fields the
    trainer forwards to ``fit`` are observed at their actual call site without
    paying for an epoch.
    """
    recorded: Dict[str, Any] = {}

    def fake_fit(self: Any, *args: Any, **kwargs: Any) -> Any:
        recorded.update(kwargs)

        class _History:
            history = {"loss": [1.0]}

        return _History()

    config.output_dir = str(tmp_dir)
    with mock.patch.object(
            train_sam2_module.SAM2TrainingModel, "fit", fake_fit):
        train_sam2(config)
    return recorded


def _obs_epochs(config: SAM2TrainingConfig, tmp_dir: Path) -> Any:
    return _record_fit(config, tmp_dir).get("epochs")


def _obs_steps_per_epoch(config: SAM2TrainingConfig, tmp_dir: Path) -> Any:
    return _record_fit(config, tmp_dir).get("steps_per_epoch")


def _obs_patience(config: SAM2TrainingConfig, tmp_dir: Path) -> Any:
    callbacks = _record_fit(config, tmp_dir).get("callbacks", [])
    return [
        getattr(callback, "patience")
        for callback in callbacks
        if hasattr(callback, "patience")
    ]


def _obs_output_path(config: SAM2TrainingConfig) -> Any:
    return str(resolved_output_dir(config))


#: field -> `(base_overrides, driven_overrides, observable)` or a DECLARATION.
FIELD_CONSUMPTION: Dict[
    str, Union[str, Tuple[Dict[str, Any], Dict[str, Any], Any]]] = {
    # --- data pipeline, observed on the batches the trainer would train on ---
    "num_frames": ({}, {"num_frames": 3}, _obs_clip_shape),
    "occlusion_frames": (
        {"num_frames": 4, "occlusion_frames": 1},
        {"num_frames": 4, "occlusion_frames": 2},
        _obs_empty_frame_count,
    ),
    "occlusion_start": (
        {"num_frames": 4, "occlusion_start": 1},
        {"num_frames": 4, "occlusion_start": 3},
        _obs_empty_frame_indices,
    ),
    "num_clips_train": ({}, {"num_clips_train": 4}, _obs_train_batches),
    "num_clips_val": ({}, {"num_clips_val": 4}, _obs_val_batches),
    "num_background_points": (
        {}, {"num_background_points": 2}, _obs_prompt_points),
    "include_box": ({}, {"include_box": True}, _obs_box_key),
    "batch_size": ({}, {"batch_size": 2}, _obs_batch_size),
    "seed": ({}, {"seed": 123}, _obs_image_content),
    # --- model -------------------------------------------------------------
    "variant": ({}, {"variant": "hiera_l"}, _obs_variant_call),
    # --- loss / optimizer, observed on the numbers they produce ------------
    "mask_weight": ({}, {"mask_weight": 7.0}, _obs_compiled_loss_value),
    "object_score_weight": (
        {}, {"object_score_weight": 7.0}, _obs_compiled_loss_value),
    "iou_weight": ({}, {"iou_weight": 7.0}, _obs_compiled_loss_value),
    "learning_rate": ({}, {"learning_rate": 0.5}, _obs_learning_rate),
    # --- output paths ------------------------------------------------------
    "output_dir": ({}, {"output_dir": "/sentinel/runs"}, _obs_output_path),
    "experiment_name": (
        {}, {"experiment_name": "sentinel_run"}, _obs_output_path),
    # --- trainer call site, observed with `fit` recorded --------------------
    "epochs": ({}, {"epochs": 9}, _obs_epochs),
    "steps_per_epoch": ({}, {"steps_per_epoch": 3}, _obs_steps_per_epoch),
    "early_stopping_patience": (
        {}, {"early_stopping_patience": 6}, _obs_patience),
    # --- declared, NOT observed here ---------------------------------------
    "smoke": (
        "`smoke` is consumed BEFORE any config exists -- `config_from_argv` "
        "applies SMOKE_PRESET while building the config, and the field is then "
        "only a record of what happened. There is no downstream consumer to "
        "reach, and `TestSmokePreset` already proves the preset moves exactly "
        "the five fields it documents, plus the hiera_l refusal."
    ),
}

#: Fields whose observable needs a temp directory (they run `train_sam2`).
_TMP_DIR_FIELDS = {"epochs", "steps_per_epoch", "early_stopping_patience"}

#: Fields whose observable is a CALL-SITE record rather than a behaviour. Named
#: explicitly so nobody reads this module as proving more than it does.
_CALL_SITE_ONLY = {
    "variant": "building a real hiera_l is 220,941,537 parameters at 1024px; "
               "the tiny branch is covered behaviourally everywhere else here",
}


class TestConfigToConsumptionWiring:
    """One layer past `argv -> config`: does the field reach a CONSUMER?"""

    @pytest.mark.parametrize(
        "field",
        sorted(
            name for name, entry in FIELD_CONSUMPTION.items()
            if not isinstance(entry, str)
        ),
    )
    def test_the_config_field_reaches_a_consumer(
            self, field: str, tmp_path: Path) -> None:
        base_overrides, driven_overrides, observe = FIELD_CONSUMPTION[field]
        if field in _TMP_DIR_FIELDS:
            base = observe(_probe_config(**base_overrides), tmp_path / "base")
            driven = observe(
                _probe_config(**driven_overrides), tmp_path / "driven")
        else:
            base = observe(_probe_config(**base_overrides))
            driven = observe(_probe_config(**driven_overrides))
        assert driven != base, (
            f"config.{field} does not reach any consumer: driving it to "
            f"{driven_overrides[field]!r} left the observable at {base!r}. "
            f"Some consumer is hard-coding the value instead of reading the "
            f"config field -- the silent-no-op defect, one layer past argv."
        )

    def test_every_config_field_is_either_probed_or_declared(self) -> None:
        """Completeness. An unproved knob is exactly how the defect shipped."""
        missing = sorted(
            set(SAM2TrainingConfig.__dataclass_fields__) - set(FIELD_CONSUMPTION)
        )
        assert not missing, (
            f"config fields with no consumption entry: {missing}. Add a probe, "
            f"or add an honest DECLARATION string saying why its effect is not "
            f"observed here."
        )

    def test_the_table_names_only_real_config_fields(self) -> None:
        bogus = sorted(
            set(FIELD_CONSUMPTION) - set(SAM2TrainingConfig.__dataclass_fields__)
        )
        assert not bogus, f"consumption rows naming no such field: {bogus}"

    def test_the_driven_values_actually_differ_from_the_defaults(self) -> None:
        """The instrument's own guard.

        If a driven value equalled what the base config already holds, the
        probe above would pass against a consumer that reads nothing at all.
        """
        for field, entry in FIELD_CONSUMPTION.items():
            if isinstance(entry, str):
                continue
            base_overrides, driven_overrides, _ = entry
            base_config = _probe_config(**base_overrides)
            assert driven_overrides[field] != getattr(base_config, field), (
                f"{field}'s driven value equals the base config's; its probe "
                f"would be vacuous"
            )

    def test_the_call_site_only_fields_are_named_as_such(self) -> None:
        """Nothing in `_CALL_SITE_ONLY` may be read as a behavioural proof."""
        unknown = sorted(set(_CALL_SITE_ONLY) - set(FIELD_CONSUMPTION))
        assert not unknown, f"_CALL_SITE_ONLY names no such field: {unknown}"


# ---------------------------------------------------------------------------
# End to end: the trainer's own two factories, put through a real `fit()`
# ---------------------------------------------------------------------------
class TestTheTrainersOwnFactoriesTrain:
    """The guard no model gate can be: the SHIPPED pipeline against the model.

    SAM 1's ``--multimask-output`` crashed at the trainer's own defaults while
    every model test stayed green, because the model fixture fed a GT shape
    ``to_training_record`` can never produce. A fixture that constructs an
    input the pipeline cannot emit is testing a fiction. So this runs
    ``create_dataset`` + ``create_sam2_training_model`` -- no fixtures -- and
    puts a real ``fit()`` step through them.
    """

    @pytest.mark.parametrize(
        "num_frames,occlusion_frames,include_box,background_points",
        [
            # T = 1: the image-path CONTROL that isolates the video machinery.
            (1, 0, False, 0),
            (2, 1, False, 0),
            # A clip whose object is never occluded: the gate must stay live.
            (3, 0, False, 0),
            # Box + background points, never exercised end to end before.
            (3, 1, True, 2),
        ],
    )
    def test_the_pipeline_and_the_model_agree_on_every_axis(
            self,
            num_frames: int,
            occlusion_frames: int,
            include_box: bool,
            background_points: int,
    ) -> None:
        config = _probe_config(
            num_frames=num_frames,
            occlusion_frames=occlusion_frames,
            occlusion_start=1 if occlusion_frames else None,
            num_clips_train=2,
            include_box=include_box,
            num_background_points=background_points,
        )
        train_dataset, _ = create_dataset(config)
        model = create_sam2_training_model(config)
        inputs, targets = next(iter(train_dataset))

        # The pipeline emits ONE mask per FRAME: the frame axis IS the mask
        # axis (M == 1). This is the fact a hand-built fixture could
        # contradict.
        assert tuple(targets[SAM2_LOW_RES_LOGITS].shape)[1] == num_frames
        assert tuple(inputs["gt_masks"].shape)[1] == num_frames
        assert ("boxes" in inputs) is include_box
        assert int(inputs["point_labels"].shape[1]) == 1 + background_points

        # The real training step, over all three losses.
        model.fit(train_dataset, epochs=1, verbose=0)

        outputs = model(inputs)
        assert set(outputs) == set(OUTPUT_KEYS)
        for key in OUTPUT_KEYS:
            assert tuple(outputs[key].shape)[:2] == (
                tuple(targets[key].shape)[:2]
            ), key
        # The GT resolution this config derives and the resolution the mask
        # decoder produces are derived INDEPENDENTLY -- one from the config's
        # `mask_size`, one from the decoder's upsampling -- and nothing else
        # compares them.
        assert (
            tuple(targets[SAM2_LOW_RES_LOGITS].shape)[2:]
            == tuple(outputs[SAM2_LOW_RES_LOGITS].shape)[2:]
        )



# ---------------------------------------------------------------------
# G9.2 -- the object-score gate is OPEN at step 0
# ---------------------------------------------------------------------


class TestTheObjectScoreGateOpensAtStepZero:
    """D-075: the trainer's model starts predicting "the object is present".

    D-043's suppression is HARD -- ``_decode`` replaces the whole mask output
    with the ``-1024`` sentinel through ``ops.where`` wherever
    ``object_score_logits <= 0``, and ``ops.where`` passes NO gradient to the
    suppressed branch. At random init the score head's sign is arbitrary
    (D-065), so without this initializer the mask loss starts structurally
    disconnected on most frames and has to wait for the score BCE to cross
    zero.

    Both arms are here, because either alone is satisfiable: the trainer's
    model must be gate-OPEN, and a model built WITHOUT the call must be
    gate-closed at the same seed -- otherwise the first arm is measuring the
    seed, not the initializer.
    """

    @staticmethod
    def _scores_and_masks(model: Any) -> Tuple[Any, Any]:
        """One forward pass over a real pipeline batch."""
        config = SAM2TrainingConfig(
            variant="tiny", num_frames=2, occlusion_frames=0,
            num_clips_train=2, num_clips_val=2, batch_size=2, seed=7)
        train_dataset, _ = create_dataset(config)
        inputs, _ = next(iter(train_dataset))
        outputs = model(inputs, training=False)
        return (np.asarray(outputs[SAM2_OBJECT_SCORE_LOGITS]),
                np.asarray(outputs[SAM2_LOW_RES_LOGITS]))

    def test_the_trainers_model_predicts_present_on_every_frame(self) -> None:
        """The GREEN arm: every score is positive, so no frame is suppressed."""
        keras.utils.set_random_seed(1)
        config = SAM2TrainingConfig(
            variant="tiny", num_frames=2, occlusion_frames=0,
            num_clips_train=2, num_clips_val=2, batch_size=2, seed=7)
        model = create_sam2_training_model(config)
        scores, masks = self._scores_and_masks(model)
        assert np.all(scores > 0.0), (
            f"the object-score gate is closed on some frame at step 0: "
            f"{scores.ravel()}")
        # ... and therefore the mask output is a real map, not the sentinel.
        assert len(np.unique(masks)) > 1, (
            "the mask output is a single constant at step 0; D-043's "
            "suppression fired despite a positive score")

    def test_the_same_seed_is_gate_closed_without_the_initializer(
            self) -> None:
        """The RED arm: delete the call and the gate shuts at this seed.

        MEASURED at seed 1: without :func:`open_object_score_gate` the score
        head predicts "absent" and the whole mask output collapses to the
        ``-1024`` sentinel with ONE unique value per frame -- the exact
        signature iteration 2 shipped.
        """
        keras.utils.set_random_seed(1)
        config = SAM2TrainingConfig(
            variant="tiny", num_frames=2, occlusion_frames=0,
            num_clips_train=2, num_clips_val=2, batch_size=2, seed=7)
        model = SAM2TrainingModel(
            create_sam2(config.variant, multimask_output=False),
            num_frames=config.num_frames, seed=config.seed)
        model.build(None)
        scores, masks = self._scores_and_masks(model)
        assert np.any(scores <= 0.0), (
            "this seed no longer lands on the negative-score branch, so the "
            f"green arm above proves nothing about the initializer: {scores.ravel()}")
        assert len(np.unique(masks[scores[..., 0] <= 0.0])) == 1

    def test_the_constant_is_upstreams_own_assume_present_value(self) -> None:
        """``10.0``, and it is not a tuning knob picked here.

        Upstream substitutes ``object_score_logits = 10.0 * ones`` when
        ``pred_obj_scores`` is off, with the comment "assuming the object is
        present" (``sam2/modeling/sam/mask_decoder.py``). The bias really is
        written -- asserted by reading it back, because an initializer that
        silently no-ops looks exactly like one that works.
        """
        assert OPEN_GATE_SCORE_BIAS == 10.0
        keras.utils.set_random_seed(1)
        model = SAM2TrainingModel(
            create_sam2("tiny", multimask_output=False), num_frames=2)
        model.build(None)
        open_object_score_gate(model)
        bias = np.asarray(
            model.sam2.mask_decoder.pred_obj_score_head.layers[-1].bias)
        np.testing.assert_allclose(bias, OPEN_GATE_SCORE_BIAS)
