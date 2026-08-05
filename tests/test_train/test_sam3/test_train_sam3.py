"""Guards for ``src/train/sam3/train_sam3.py`` -- the CLI, the config, the
``argv -> config`` wiring, the compile site, the freeze flag and the metrics.

The load-bearing guard is
``TestArgsToConfigWiring.test_every_cli_flag_reaches_the_config_field_it_names``:
it drives EVERY parser flag with a non-default sentinel through the full
``argv -> parse_args -> explicitly_set_flags -> config`` path and reads the
field back, so a dropped wiring row fails BY FLAG NAME. This repository shipped
exactly that defect once (`train/bfunet`: ``--high-freq-blocks`` and
``--filter-multiplier`` became silent no-ops).

The second is ``TestSmokePreset``: every field the preset does NOT name is
asserted BIT-IDENTICAL smoke-vs-real. A preset that silently changes WHAT is
measured rather than how precisely is a recorded repo trap -- the DINO
``--smoke`` case moved the k-NN bank size and made smoke numbers incomparable
to real ones.

Device: this module is written to run on the gate device
(``CUDA_VISIBLE_DEVICES=1``). Everything above ``TestTheCompiledTrainer`` is
CPU-cheap; the model-building classes build the ``tiny`` variant, which is the
smallest thing that exercises the real compile and freeze paths.
"""

import argparse
import math
from typing import Any, Dict, List, Tuple

import keras
import numpy as np
import pytest

from dl_techniques.losses.sam3_detection_loss import (
    Sam3DetectionLoss,
    box_cxcywh_to_xyxy,
    iou_and_generalized_iou,
    unpack_targets,
)
from dl_techniques.models.sam3.sam3_image import Sam3Image
from train.common.args import explicitly_set_flags
from train.sam3.train_sam3 import (
    CLI_TO_CONFIG,
    DERIVED_FIELDS,
    EVAL_METRIC_KEYS,
    LR_SCHEDULES,
    NON_CONFIG_DESTS,
    SMOKE_PRESET,
    VARIANTS,
    Sam3TrainingConfig,
    build_parser,
    config_from_argv,
    create_datasets,
    create_optimizer,
    create_training_model,
    evaluate_sam3,
    parse_arguments,
    resolved_output_dir,
)

#: A `tiny`-variant config small enough to build, compile and step in a test.
TINY_KWARGS: Dict[str, Any] = dict(
    variant="tiny", batch_size=2, num_train_samples=4, num_val_samples=2,
    max_instances=3, max_per_category=2, epochs=1, warmup_steps=0)


# ---------------------------------------------------------------------------
# Sentinels
# ---------------------------------------------------------------------------
#: Flags whose legal values are constrained by ``__post_init__``, so a
#: "default + 7" sentinel would be refused for a reason unrelated to wiring.
SENTINEL_OVERRIDES: Dict[str, Tuple[List[str], Any]] = {
    # Must stay in [0, 1]; the default is 0.25.
    "zero_instance_rate": (["--zero-instance-rate", "0.75"], 0.75),
    # Must not exceed `max_instances` (default 8).
    "max_per_category": (["--max-per-category", "5"], 5),
    # Must be <= num_train_samples / num_val_samples (defaults 512 / 128).
    "batch_size": (["--batch-size", "11"], 11),
    # Must stay >= batch_size (default 4).
    "num_val_samples": (["--num-val-samples", "135"], 135),
}


def _sentinel_for(action: argparse.Action) -> Tuple[List[str], Any]:
    """Build ``(argv, expected_value)`` driving one flag to a NON-default value.

    A sentinel equal to the default would be satisfied by a flag wired to
    nothing at all -- the exact defect this module exists to catch -- so every
    branch here produces a value the default cannot be.

    :param action: The argparse action to drive.
    :type action: argparse.Action
    :return: ``(argv, expected)``.
    :rtype: Tuple[List[str], Any]
    :raises AssertionError: If no sentinel rule covers the action's shape. A
        new flag shape must extend this function rather than be skipped.
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
        value = (0.5 if action.default in (None, 0.5)
                 else float(action.default) + 0.5)
        return [flag, str(value)], value
    if action.type is str:
        return [flag, f"sentinel_{action.dest}"], f"sentinel_{action.dest}"
    raise AssertionError(
        f"no sentinel rule for {flag} (type={action.type!r}, "
        f"action={type(action).__name__}); extend _sentinel_for")


def _config_actions() -> List[argparse.Action]:
    """Every parser action that is supposed to reach the config."""
    return [action for action in build_parser()._actions
            if action.dest not in NON_CONFIG_DESTS]


# ---------------------------------------------------------------------------
# The args -> config wiring
# ---------------------------------------------------------------------------
class TestArgsToConfigWiring:
    """The silent-no-op defect class, attacked through the real entry point."""

    def test_every_cli_flag_reaches_the_config_field_it_names(self) -> None:
        """Drive each flag with a non-default sentinel; read the field back."""
        violations = []
        for action in _config_actions():
            argv, expected = _sentinel_for(action)
            field = CLI_TO_CONFIG.get(action.dest)
            if field is None:
                violations.append(
                    f"{action.option_strings[0]}: dest {action.dest!r} has no "
                    f"row in CLI_TO_CONFIG")
                continue
            actual = getattr(config_from_argv(argv), field)
            if actual != expected:
                violations.append(
                    f"{action.option_strings[0]} -> config.{field}: expected "
                    f"{expected!r}, got {actual!r} (the flag is a SILENT NO-OP)")
        assert not violations, "\n".join(violations)

    def test_every_cli_flag_is_wired_to_a_config_field(self) -> None:
        """Completeness in the argv direction: no flag without a wiring row."""
        unwired = sorted(action.dest for action in _config_actions()
                         if action.dest not in CLI_TO_CONFIG)
        assert not unwired, (
            f"parser dests with no CLI_TO_CONFIG row: {unwired}. Add the row, "
            f"or list the dest in NON_CONFIG_DESTS.")

    def test_every_config_field_is_reachable_from_the_cli(self) -> None:
        """Completeness in the config direction: no unreachable knob."""
        wired = set(CLI_TO_CONFIG.values())
        unreachable = sorted(
            name for name in Sam3TrainingConfig.__dataclass_fields__
            if name not in wired and name not in DERIVED_FIELDS)
        assert not unreachable, (
            f"config fields no CLI flag reaches: {unreachable}. Add a flag, or "
            f"declare the field in DERIVED_FIELDS.")

    def test_the_wiring_table_names_only_real_config_fields(self) -> None:
        real = set(Sam3TrainingConfig.__dataclass_fields__)
        bogus = sorted(set(CLI_TO_CONFIG.values()) - real)
        assert not bogus, f"CLI_TO_CONFIG rows naming no such field: {bogus}"

    def test_the_sentinels_actually_differ_from_the_defaults(self) -> None:
        """The guard's own instrument, RED-proofed.

        If a sentinel ever equalled its default, the wiring test would pass
        against a completely unwired flag.
        """
        for action in _config_actions():
            _, expected = _sentinel_for(action)
            assert expected != action.default, (
                f"{action.option_strings[0]}'s sentinel equals its default "
                f"({expected!r}); the wiring test would be vacuous for it")

    def test_one_flag_at_a_time_is_what_the_wiring_test_drives(self) -> None:
        """One mutation must land on one assertion."""
        for action in _config_actions():
            argv, _ = _sentinel_for(action)
            flags = [token for token in argv if token.startswith("--")]
            assert len(flags) == 1, (
                f"{action.option_strings[0]}'s sentinel argv drives {flags}")


# ---------------------------------------------------------------------------
# `--help`
# ---------------------------------------------------------------------------
class TestHelpDoesNotTrain:
    """A trainer once started a 100-epoch job on ``--help``."""

    def test_help_exits_zero_before_any_config_is_built(self) -> None:
        with pytest.raises(SystemExit) as exit_info:
            parse_arguments(["--help"])
        assert exit_info.value.code == 0

    def test_help_is_not_a_config_field(self) -> None:
        assert "help" in NON_CONFIG_DESTS
        assert "help" not in CLI_TO_CONFIG

    def test_no_help_string_carries_a_bare_percent(self) -> None:
        """argparse runs every help string through ``help % params``.

        A lone ``%`` (as in "85% of the card") therefore raises
        ``ValueError: unsupported format character`` at ``--help`` time -- a
        crash that only the help path can reach and that no other test would
        see. ``%%`` and ``%(default)s`` are legal and are excluded here.
        """
        offenders = []
        for action in build_parser()._actions:
            text = action.help or ""
            stripped = text.replace("%%", "").replace("%(default)s", "")
            stripped = stripped.replace("%(prog)s", "")
            if "%" in stripped:
                offenders.append(f"{action.option_strings}: {text!r}")
        assert not offenders, (
            "help strings with a bare '%', which argparse formats and would "
            f"crash --help on: {offenders}")

    def test_the_parser_carries_no_short_option_but_h(self) -> None:
        """``explicitly_set_flags`` REFUSES any other short option.

        Building the parser through it is the executable proof.
        """
        assert explicitly_set_flags(build_parser(), ["--epochs", "3"]) == {
            "epochs"}


# ---------------------------------------------------------------------------
# `--smoke`
# ---------------------------------------------------------------------------
class TestSmokePreset:
    """What the preset changes, field by field, and what it must not."""

    def test_smoke_changes_exactly_the_documented_fields(self) -> None:
        base = config_from_argv([])
        smoke = config_from_argv(["--smoke"])
        differing = {
            name: (getattr(base, name), getattr(smoke, name))
            for name in Sam3TrainingConfig.__dataclass_fields__
            if getattr(base, name) != getattr(smoke, name)}
        differing.pop("smoke")
        differing.pop("experiment_name", None)
        assert set(differing) == set(SMOKE_PRESET), (
            f"--smoke moved {sorted(differing)}, SMOKE_PRESET declares "
            f"{sorted(SMOKE_PRESET)}")

    def test_every_field_the_preset_does_not_name_is_bit_identical(
            self) -> None:
        """The brief's requirement, as a SET operation over every field.

        Not a hand-listed subset: a field added to the config later is covered
        automatically, and a preset key added later is caught by the assertion
        above. ``experiment_name`` is timestamped and ``smoke`` is the flag
        itself, so both are excluded by construction.
        """
        base = config_from_argv([])
        smoke = config_from_argv(["--smoke"])
        untouched = (set(Sam3TrainingConfig.__dataclass_fields__)
                     - set(SMOKE_PRESET) - {"smoke", "experiment_name"})
        for name in sorted(untouched):
            expected, actual = getattr(base, name), getattr(smoke, name)
            assert type(expected) is type(actual) and expected == actual, (
                f"--smoke moved {name}: {expected!r} -> {actual!r}. A preset "
                f"may change how much is measured, never what.")

    def test_the_preset_never_touches_a_field_that_shapes_the_run(
            self) -> None:
        """The same claim, enumerated from the SHAPING side.

        ``batch_size`` is in this set deliberately, which is where this trainer
        DEPARTS from the SAM 1 and SAM 2 presets: it is the presence term's own
        divisor, so moving it would rescale a loss the smoke run is supposed to
        be comparable on (decisions.md D-030).
        """
        shaping = {"variant", "include_masks", "freeze_trunk", "seed",
                   "learning_rate", "weight_decay", "gradient_clip_norm",
                   "warmup_steps", "lr_schedule", "batch_size",
                   "zero_instance_rate", "max_instances", "max_per_category"}
        offenders = sorted(shaping & set(SMOKE_PRESET))
        assert not offenders, (
            f"SMOKE_PRESET declares {offenders}, which change WHAT is "
            f"measured. A smoke preset may only change how much.")

    def test_an_explicitly_typed_flag_beats_the_preset(self) -> None:
        config = config_from_argv(["--smoke", "--epochs", "11"])
        assert config.epochs == 11 and config.smoke is True

    def test_every_preset_field_can_be_typed_at_its_own_default_and_win(
            self) -> None:
        """The provenance property, across the WHOLE preset.

        A flag typed at its own parser default is indistinguishable from an
        omission in the Namespace, so a value-vs-default implementation
        silently overrides it. One field alone would leave the rest unproved.
        """
        defaults = Sam3TrainingConfig()
        for field in SMOKE_PRESET:
            default_value = getattr(defaults, field)
            assert default_value != SMOKE_PRESET[field], (
                f"{field}'s preset value equals its default; its provenance "
                f"arm would be vacuous")
            flag = "--" + field.replace("_", "-")
            config = config_from_argv(["--smoke", flag, str(default_value)])
            assert getattr(config, field) == default_value, (
                f"{flag} typed at its own default ({default_value!r}) lost to "
                f"the preset; provenance is computed by VALUE, not by whether "
                f"the token was typed")

    def test_an_omitted_flag_takes_the_preset(self) -> None:
        config = config_from_argv(["--smoke"])
        for field, value in SMOKE_PRESET.items():
            assert getattr(config, field) == value

    def test_the_preset_keys_are_real_config_fields(self) -> None:
        assert set(SMOKE_PRESET) <= set(
            Sam3TrainingConfig.__dataclass_fields__)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------
class TestConfigValidation:
    """``__post_init__`` refuses configurations that would fail far away."""

    def test_the_variant_list_comes_from_the_models_own_table(self) -> None:
        assert set(VARIANTS) == set(Sam3Image.MODEL_VARIANTS) - {"sam3"}

    def test_the_released_variant_is_refused_naming_the_reason(self) -> None:
        with pytest.raises(ValueError, match="821,708,598"):
            Sam3TrainingConfig(variant="sam3")

    def test_unknown_variant_is_refused_naming_the_known_ones(self) -> None:
        with pytest.raises(ValueError, match="small"):
            Sam3TrainingConfig(variant="enormous")

    @pytest.mark.parametrize("kwargs", [
        {"batch_size": 0}, {"epochs": 0}, {"learning_rate": 0.0},
        {"weight_decay": -0.1}, {"gradient_clip_norm": -1.0},
        {"warmup_steps": -1}, {"early_stopping_patience": 0},
        {"zero_instance_rate": 1.5}, {"max_instances": 0},
        {"max_per_category": 99}, {"lr_schedule": "quadratic"},
    ])
    def test_out_of_range_values_are_refused(self,
                                             kwargs: Dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            Sam3TrainingConfig(**kwargs)

    def test_a_split_smaller_than_one_batch_is_refused(self) -> None:
        """``drop_remainder=True`` (D-023) would make such a split EMPTY.

        Without this raise `fit` sees zero steps and reports a vacuous run.
        """
        with pytest.raises(ValueError, match="EMPTY"):
            Sam3TrainingConfig(batch_size=8, num_train_samples=4)

    def test_the_matching_split_is_accepted(self) -> None:
        """The non-firing control: the refusal is about the PAIR."""
        assert Sam3TrainingConfig(batch_size=4,
                                  num_train_samples=4).batch_size == 4

    def test_zero_gradient_clip_is_legal_and_means_off(self) -> None:
        config = Sam3TrainingConfig(gradient_clip_norm=0.0)
        assert create_optimizer(config).global_clipnorm is None

    def test_steps_per_epoch_matches_the_dropped_remainder(self) -> None:
        assert Sam3TrainingConfig(num_train_samples=33,
                                  batch_size=4).steps_per_epoch == 8

    def test_experiment_name_is_derived_when_omitted_and_kept_when_given(
            self) -> None:
        assert config_from_argv([]).experiment_name.startswith("sam3_small_")
        assert config_from_argv(
            ["--experiment-name", "abc"]).experiment_name == "abc"


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
class TestOutputDirectory:
    """Repo-root ``results/``, never ``src/results/``."""

    def test_a_relative_output_dir_resolves_against_the_repo_root(self) -> None:
        path = resolved_output_dir(config_from_argv(["--experiment-name", "x"]))
        assert path.is_absolute() and path.name == "x"
        assert path.parent.name == "results"

    def test_the_resolved_path_is_not_under_src(self) -> None:
        path = resolved_output_dir(config_from_argv([]))
        assert "src" not in path.parts, (
            f"{path} is under src/; the repo convention is repo-root results/")

    def test_an_absolute_output_dir_is_used_verbatim(self, tmp_path) -> None:
        config = config_from_argv(
            ["--output-dir", str(tmp_path), "--experiment-name", "run"])
        assert resolved_output_dir(config) == tmp_path / "run"


# ---------------------------------------------------------------------------
# The optimizer, against the REFERENCE
# ---------------------------------------------------------------------------
class TestTheOptimizerRecipe:
    """The values the reference fixes, and the divergences D-027 names."""

    def test_the_reference_exact_terms_reach_the_optimizer(self) -> None:
        """AdamW, global L2 clip 0.1, weight decay 0.1 -- all reference-EXACT."""
        optimizer = create_optimizer(config_from_argv([]))
        assert isinstance(optimizer, keras.optimizers.AdamW)
        assert optimizer.global_clipnorm == pytest.approx(0.1)
        assert optimizer.weight_decay == pytest.approx(0.1)

    def test_the_default_lr_is_the_references_UNSCALED_transformer_lr(
            self) -> None:
        """8e-4, not the fine-tune's 8e-5.

        ``lr_scale: 0.1`` is a discount applied for a 100-image fine-tune of a
        PRETRAINED 822M model. This trainer starts from random init, where
        there is no pretrained weight to protect -- the signed divergence
        ``- REFERENCE_ONLY(lr_scale 0.1)`` of D-027.
        """
        reference_base_lr = 8e-4
        reference_lr_scale = 0.1
        assert Sam3TrainingConfig().learning_rate == pytest.approx(
            reference_base_lr)
        assert Sam3TrainingConfig().learning_rate != pytest.approx(
            reference_base_lr * reference_lr_scale)

    def test_the_warmup_length_is_the_references_own_value_in_steps(
            self) -> None:
        """20 STEPS. The reference's `step` is a global iteration count.

        ``trainer.py:825`` passes ``step=int(exact_epoch * iters_per_epoch)``,
        so reading its ``warmup_steps: 20`` as EPOCHS would be a 1500x error.
        """
        assert Sam3TrainingConfig().warmup_steps == 20

    def test_weight_decay_is_excluded_from_bias_and_layernorm_parameters(
            self) -> None:
        """The exclusion must actually MATCH something.

        A pattern list that matches no variable name would leave the reference's
        `wd 0.0` override on `*bias*` and LayerNorm doing nothing at all, and
        no other assertion here could see that.
        """
        optimizer = create_optimizer(config_from_argv([]))
        model = Sam3Image.from_variant("tiny")
        model.build(None)
        names = [v.name for v in model.weights]
        excluded = [n for n in names if not optimizer._use_weight_decay(
            next(v for v in model.weights if v.name == n))]
        assert excluded, (
            "the weight-decay exclusion matched NO variable; the reference's "
            "wd-0 override on bias/LayerNorm is inert")
        assert all(("bias" in n or "gamma" in n or "beta" in n)
                   for n in excluded)
        assert any("bias" in n for n in excluded)
        assert any("gamma" in n for n in excluded)
        # The liveness arm: a decayed variable must also exist, or the
        # "excluded" claim would be trivially true of everything.
        assert any(True for v in model.weights
                   if optimizer._use_weight_decay(v)), (
            "every variable is excluded; weight decay would be globally dead")

    @pytest.mark.parametrize("schedule", LR_SCHEDULES)
    def test_every_advertised_schedule_builds(self, schedule: str) -> None:
        config = config_from_argv(["--lr-schedule", schedule,
                                   "--warmup-steps", "0"])
        assert create_optimizer(config) is not None


# ---------------------------------------------------------------------------
# The compiled trainer
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny_config() -> Sam3TrainingConfig:
    return Sam3TrainingConfig(**TINY_KWARGS)


@pytest.fixture(scope="module")
def tiny_model(tiny_config: Sam3TrainingConfig) -> Any:
    """A seed-PINNED `tiny` trainer.

    The seed is not decoration. ``test_pred_masks_are_not_a_constant`` asserts
    a property of a RANDOM initialization, and an unseeded fixture makes that
    assertion a different measurement on every run -- it was observed failing
    once, and passing on four consecutive re-runs, before this pin. A guard
    that reports a different answer per run cannot be RED-proven.
    """
    keras.utils.set_random_seed(1234)
    return create_training_model(tiny_config)


class TestTheCompiledTrainer:
    """What ``create_training_model`` guarantees by construction."""

    def test_jit_compile_is_false(self, tiny_model: Any) -> None:
        """Doubly forced: the family pins it, and XLA has no EagerPyFunc."""
        assert tiny_model.jit_compile is False

    def test_the_loss_is_the_joint_detection_loss_agreeing_on_masks(
            self, tiny_model: Any) -> None:
        assert isinstance(tiny_model.loss, Sam3DetectionLoss)
        assert tiny_model.loss.include_masks is tiny_model.include_masks

    def test_the_optimizer_is_the_reference_derived_adamw(
            self, tiny_model: Any) -> None:
        assert isinstance(tiny_model.optimizer, keras.optimizers.AdamW)
        assert tiny_model.optimizer.global_clipnorm == pytest.approx(0.1)

    def test_include_masks_reaches_the_model_and_the_loss_together(
            self) -> None:
        model = create_training_model(
            Sam3TrainingConfig(include_masks=True, **{
                k: v for k, v in TINY_KWARGS.items()
                if k != "include_masks"}))
        assert model.include_masks is True
        assert model.loss.include_masks is True
        assert model.packed_channels > 5


class TestFreezeTrunk:
    """The flag must FREEZE something, measured as a count."""

    def test_freeze_trunk_drops_the_trainable_variable_count(self) -> None:
        """Both directions, with the exact numbers.

        A flag that runs and freezes nothing is the failure mode here, and the
        flag's own value cannot detect it. The trunk's variables must move from
        the trainable list to the non-trainable one, and the TOTAL must not
        change -- a count that dropped because variables VANISHED would satisfy
        a one-sided assertion.
        """
        joint = create_training_model(Sam3TrainingConfig(**TINY_KWARGS))
        frozen = create_training_model(
            Sam3TrainingConfig(freeze_trunk=True, **TINY_KWARGS))

        trunk_variables = len(joint.sam3.backbone.trainable_variables)
        assert trunk_variables > 0, "the trunk has no variables to freeze"
        assert len(frozen.trainable_variables) == (
            len(joint.trainable_variables) - trunk_variables)
        assert len(frozen.sam3.backbone.trainable_variables) == 0
        assert (len(frozen.trainable_variables)
                + len(frozen.non_trainable_variables)) == (
            len(joint.trainable_variables)
            + len(joint.non_trainable_variables))

    def test_only_the_trunk_is_frozen(self) -> None:
        """The decoder, the text tower and the heads must stay trainable."""
        frozen = create_training_model(
            Sam3TrainingConfig(freeze_trunk=True, **TINY_KWARGS))
        for name in ("text_encoder", "transformer", "segmentation_head"):
            component = getattr(frozen.sam3, name)
            assert component.trainable_variables, (
                f"{name} was frozen too; --freeze-trunk must freeze the IMAGE "
                f"TRUNK only, or the A/B compares two unrelated things")


# ---------------------------------------------------------------------------
# The data path
# ---------------------------------------------------------------------------
class TestTheTrainersDatasets:
    """The trainer's own factory, not a hand-built fixture."""

    def test_drop_remainder_is_true_so_the_batch_axis_is_static(
            self, tiny_config: Sam3TrainingConfig, tiny_model: Any) -> None:
        """D-023: a dynamic batch axis raises inside the neck at step 1.

        Measured that step: ``ValueError: positional encoding shape
        (32, 32, None) must match the feature's (32, 32, 8)``, raised from
        ``Sam3DualViTDetNeck.call()`` -- and it fires even when the sample
        count divides the batch size exactly, so "pick a divisible epoch" is
        not an escape.
        """
        train_dataset, val_dataset = create_datasets(tiny_config, tiny_model)
        for dataset in (train_dataset, val_dataset):
            inputs_spec, target_spec = dataset.element_spec
            assert target_spec.shape[0] == tiny_config.batch_size
            assert inputs_spec["image"].shape[0] == tiny_config.batch_size

    def test_the_two_splits_are_drawn_from_different_seeds(
            self, tiny_config: Sam3TrainingConfig, tiny_model: Any) -> None:
        train_dataset, val_dataset = create_datasets(tiny_config, tiny_model)
        train_batch = next(iter(train_dataset))[0]["image"]
        val_batch = next(iter(val_dataset))[0]["image"]
        assert not np.allclose(np.asarray(train_batch), np.asarray(val_batch))


# ---------------------------------------------------------------------------
# The metrics -- what makes step 7's claim possible
# ---------------------------------------------------------------------------
class TestEvaluation:
    """Achieved IoU, presence accuracy, and the degeneracy guard."""

    def test_it_returns_exactly_the_declared_keys(
            self, tiny_config: Sam3TrainingConfig, tiny_model: Any) -> None:
        """The CSVLogger column set is frozen on epoch 0, so this is binding.

        A metric that appears late never reaches ``training_log.csv`` at all;
        one that disappears breaks the row. So the key SET must be constant.
        """
        _, val_dataset = create_datasets(tiny_config, tiny_model)
        metrics = evaluate_sam3(tiny_model, val_dataset)
        assert set(metrics) == set(EVAL_METRIC_KEYS)
        assert all(isinstance(value, float) for value in metrics.values())

    def test_an_empty_dataset_still_returns_every_key_as_nan(
            self, tiny_config: Sam3TrainingConfig, tiny_model: Any) -> None:
        """The nan-fill path, which is what keeps the CSV schema stable."""
        _, val_dataset = create_datasets(tiny_config, tiny_model)
        metrics = evaluate_sam3(tiny_model, val_dataset, max_batches=0)
        assert set(metrics) == set(EVAL_METRIC_KEYS)
        assert all(math.isnan(value) for value in metrics.values())

    def test_mask_iou_is_nan_when_masks_are_off_and_a_number_when_on(
            self, tiny_config: Sam3TrainingConfig, tiny_model: Any) -> None:
        """Reported as `nan`, never as a silent 0.0 that looks like a result."""
        _, val_dataset = create_datasets(tiny_config, tiny_model)
        assert math.isnan(evaluate_sam3(tiny_model, val_dataset)["mask_iou"])

        masked_config = Sam3TrainingConfig(include_masks=True, **{
            k: v for k, v in TINY_KWARGS.items() if k != "include_masks"})
        masked_model = create_training_model(masked_config)
        _, masked_val = create_datasets(masked_config, masked_model)
        assert not math.isnan(
            evaluate_sam3(masked_model, masked_val)["mask_iou"])

    def test_box_iou_is_a_real_iou_on_matched_pairs(
            self, tiny_config: Sam3TrainingConfig, tiny_model: Any) -> None:
        """Bounded in [0, 1], and averaged over matched pairs only."""
        _, val_dataset = create_datasets(tiny_config, tiny_model)
        metrics = evaluate_sam3(tiny_model, val_dataset)
        assert 0.0 <= metrics["box_iou"] <= 1.0
        assert 0.0 <= metrics["presence_accuracy"] <= 1.0
        assert metrics["num_matched_pairs"] >= 0.0

    def test_the_iou_instrument_is_calibrated_and_needs_the_xyxy_conversion(
            self) -> None:
        """The instrument's own calibration, and the trap it caught.

        ``iou_and_generalized_iou`` reads **xyxy**; every box on the trainer's
        path is normalized ``cxcywh`` (H-5). Feeding it ``cxcywh`` does not
        raise -- it computes the overlap of two rectangles that are not the
        boxes. MEASURED at this step: the un-converted spelling returned
        **0.0** for a box against ITSELF, which is indistinguishable from "the
        model has not learned" and is exactly what the first three ``--smoke``
        runs reported.

        **The instrument was wrong in the direction of a FALSE NEGATIVE, and
        every bounds assertion passed while it was.** A ``cxcywh``-fed IoU is
        still in ``[0, 1]`` -- it is simply 0.0 -- so
        ``test_box_iou_is_a_real_iou_on_matched_pairs`` and every "is it
        finite / is it in range" check are structurally blind to it. Only a
        self-IoU calibration arm and the NumPy oracle below can see it. Anyone
        reading a step-7 negative result should know this failure mode was
        live here once.

        So this asserts BOTH arms: converted gives 1.0, and the unconverted
        spelling this test exists to forbid does not.
        """
        truth = np.array([[[0.5, 0.5, 0.2, 0.2]]], dtype="float32")
        shifted = np.array([[[0.9, 0.9, 0.2, 0.2]]], dtype="float32")
        exact = float(iou_and_generalized_iou(
            box_cxcywh_to_xyxy(truth), box_cxcywh_to_xyxy(truth))[0])
        assert exact == pytest.approx(1.0, abs=1e-5)
        assert float(iou_and_generalized_iou(
            box_cxcywh_to_xyxy(shifted),
            box_cxcywh_to_xyxy(truth))[0]) == pytest.approx(0.0, abs=1e-6)
        # The RED arm: the same identical pair, WITHOUT the conversion.
        assert float(iou_and_generalized_iou(truth, truth)[0]) != (
            pytest.approx(1.0, abs=1e-5))

    def test_pred_masks_are_not_a_constant(
            self, tiny_config: Sam3TrainingConfig, tiny_model: Any) -> None:
        """The degeneracy guard a loss value structurally cannot provide.

        A constant mask output has a perfectly stable loss. The unique-value
        count is 1 exactly when the head has collapsed, so a step-7 mask IoU
        cannot be reported without this number beside it.
        """
        _, val_dataset = create_datasets(tiny_config, tiny_model)
        metrics = evaluate_sam3(tiny_model, val_dataset)
        assert metrics["pred_mask_unique_values"] > 1.0, (
            "pred_masks is CONSTANT; any IoU reported beside it is an artifact")

    def test_box_iou_matches_a_hand_written_numpy_xyxy_oracle(
            self, tiny_config: Sam3TrainingConfig, tiny_model: Any) -> None:
        """The oracle that makes the cxcywh/xyxy mix-up VISIBLE.

        A bounds assertion cannot see it: a ``cxcywh``-fed IoU is still in
        ``[0, 1]`` -- it is simply 0.0. So the same batch is re-scored here by
        a NumPy IoU written from the box definition, with no call into the loss
        module's own geometry, and the two numbers must agree.
        """
        _, val_dataset = create_datasets(tiny_config, tiny_model)
        inputs, y_true = next(iter(val_dataset))
        outputs = tiny_model.sam3(inputs, training=False)
        targets = unpack_targets(keras.ops.cast(y_true, "float32"),
                                 tiny_model.include_masks)
        assignment, is_matched = tiny_model.loss.matcher(
            outputs["pred_logits"], outputs["pred_boxes"],
            targets["target_boxes"], targets["target_valid"])

        def to_corners(boxes: np.ndarray) -> np.ndarray:
            cx, cy, w, h = (boxes[..., i] for i in range(4))
            return np.stack(
                [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1)

        predicted = to_corners(np.asarray(outputs["pred_boxes"]))
        gt = np.asarray(targets["target_boxes"])
        picked = to_corners(
            np.take_along_axis(gt, np.asarray(assignment)[..., None], axis=1))
        low = np.maximum(predicted[..., :2], picked[..., :2])
        high = np.minimum(predicted[..., 2:], picked[..., 2:])
        overlap = np.prod(np.clip(high - low, 0.0, None), axis=-1)
        area = lambda b: np.prod(np.clip(b[..., 2:] - b[..., :2], 0.0, None),
                                 axis=-1)
        union = area(predicted) + area(picked) - overlap
        matched = np.asarray(is_matched)
        expected = float((overlap / np.maximum(union, 1e-12) * matched).sum()
                         / max(matched.sum(), 1e-12))

        measured = evaluate_sam3(tiny_model, val_dataset, max_batches=1)
        assert measured["box_iou"] == pytest.approx(expected, abs=1e-5)

    def test_the_eval_callback_writes_every_key_including_the_nan_ones(
            self, tiny_config: Sam3TrainingConfig, tiny_model: Any) -> None:
        """``CSVLogger`` freezes its columns on the first epoch it sees.

        A key omitted at epoch 0 because it happened to be ``nan`` never
        reaches ``training_log.csv`` at all -- for any epoch. So the callback
        must write EVERY key, every epoch. ``mask_iou`` is ``nan`` in this
        fixture, which is what makes the assertion non-vacuous.
        """
        from train.sam3.train_sam3 import _Sam3EvalCallback

        _, val_dataset = create_datasets(tiny_config, tiny_model)
        callback = _Sam3EvalCallback(val_dataset)
        callback.set_model(tiny_model)
        logs: Dict[str, Any] = {"loss": 1.0}
        callback.on_epoch_end(0, logs)

        for key in EVAL_METRIC_KEYS:
            assert f"val_{key}" in logs, (
                f"val_{key} missing from epoch 0's logs; CSVLogger would drop "
                f"it from every row of training_log.csv")
        assert math.isnan(logs["val_mask_iou"]), (
            "this fixture has masks off, so val_mask_iou must be nan -- "
            "otherwise the nan-fill arm of this test is vacuous")

    def test_the_metrics_callback_runs_before_the_csv_logger(
            self, tiny_config: Sam3TrainingConfig, tiny_model: Any,
            tmp_path) -> None:
        """Ordering is the other half of the CSVLogger trap.

        A callback appended AFTER ``CSVLogger`` writes into a ``logs`` dict
        that has already been consumed, so every metric would be absent from
        the CSV while still appearing on the progress bar.
        """
        from train.sam3.train_sam3 import _Sam3EvalCallback, build_callbacks

        _, val_dataset = create_datasets(tiny_config, tiny_model)
        callbacks = build_callbacks(tiny_config, tmp_path, val_dataset)
        kinds = [type(callback) for callback in callbacks]
        assert _Sam3EvalCallback in kinds, "the metrics callback is not wired"
        assert keras.callbacks.CSVLogger in kinds, (
            "no CSVLogger; this test's premise is gone")
        assert kinds.index(_Sam3EvalCallback) < kinds.index(
            keras.callbacks.CSVLogger)
        for reader in (keras.callbacks.EarlyStopping,
                       keras.callbacks.ModelCheckpoint):
            assert kinds.index(_Sam3EvalCallback) < kinds.index(reader)

    def test_the_per_term_losses_are_reported_beside_the_total(
            self, tiny_config: Sam3TrainingConfig, tiny_model: Any) -> None:
        """A falling total can hide a term doing nothing."""
        _, val_dataset = create_datasets(tiny_config, tiny_model)
        metrics = evaluate_sam3(tiny_model, val_dataset)
        for term in ("loss_ce", "presence_loss", "loss_bbox", "loss_giou"):
            assert math.isfinite(metrics[term]), f"{term} is not finite"
        # Masks are off in this fixture, so both mask terms must be exactly 0.
        assert metrics["loss_mask"] == 0.0 and metrics["loss_dice"] == 0.0
