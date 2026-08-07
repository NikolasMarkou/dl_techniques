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
from dl_techniques.models.sam3.training_model import (
    Sam3TrainingModel,
    compile_sam3_trainer,
    pack_predictions,
)
from tests.test_train.test_sam3.parser_help_guard import (
    assert_no_bare_percent_help,
)
from train.common.args import explicitly_set_flags
from train.sam3.train_sam3 import (
    CLI_TO_CONFIG,
    DERIVED_FIELDS,
    EVAL_METRIC_KEYS,
    LR_SCHEDULES,
    NON_CONFIG_DESTS,
    SELECTION_METRIC,
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
        see. The check itself lives in ``parser_help_guard`` and is called from
        ``test_baselines.py`` too: this plan shipped the same defect in the
        OTHER SAM 3 parser, which a copy of the loop here could not have seen.
        """
        assert_no_bare_percent_help(build_parser(), "train_sam3.build_parser")

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
        shaping = {"variant", "include_masks", "freeze_trunk",
                   "deep_supervision", "query_selection", "seed",
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

    def test_pad_n_queries_is_the_VARIANTS_own_Q_not_the_references_200(
            self, tiny_model: Any) -> None:
        """A reference constant tied to the reference's geometry is a
        VARIANT-DERIVED quantity.

        ``Sam3DetectionLoss`` defaults ``pad_n_queries=200`` because that is the
        RELEASED variant's ``num_queries`` -- and both shipped reference configs
        set the two together (``roboflow_*.yaml:100`` literally,
        ``odinw_text_only_train.yaml:102`` as ``${scratch.num_queries}``).
        Carrying the 200 into a variant that emits Q=32 puts divisor #4 on
        ``200 * B`` instead of ``32 * B`` and divides the ENTIRE classification
        term by exactly 6.25 -- MEASURED on a real ``small`` batch, raw
        ``loss_ce`` 0.043937 at 200 against 0.274605 at 32, and a weighted share
        of the total that moves 9.1 % -> 38.4 % over a 64-image split.

        Non-vacuous by construction: ``tiny`` has Q=5 and ``small`` Q=32, so
        both differ from 200 and from each other, and the assertion cannot be
        satisfied by the default.
        """
        assert tiny_model.loss.pad_n_queries == tiny_model.num_queries
        assert tiny_model.loss.pad_n_queries != 200, (
            "this fixture's Q is 200, so the assertion above is vacuous")

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


def _tiny_argv(*extra: str) -> List[str]:
    """Spell :data:`TINY_KWARGS` as CLI tokens, plus ``extra``.

    Interface contract: the returned argv resolves through the REAL parser to a
    config equal to ``Sam3TrainingConfig(**TINY_KWARGS)`` on every field
    ``TINY_KWARGS`` names. Derived from that dict rather than hand-written, so
    the two cannot drift.

    :param extra: Additional tokens appended verbatim.
    :type extra: str
    :return: The argv token list.
    :rtype: List[str]
    """
    argv: List[str] = []
    for key, value in TINY_KWARGS.items():
        argv += ["--" + key.replace("_", "-"), str(value)]
    return argv + list(extra)


class TestDeepSupervision:
    """``--deep-supervision`` through the FULL ``argv -> config -> factory``.

    The wiring table guards above already drive this flag with a sentinel and
    read the CONFIG field back, which is the silent-no-op guard. What this class
    adds is the half no table can express: that the field then reaches the
    MODEL and the LOSS, that the two agree on the row stride, and that the
    ``--smoke`` preset does not touch it.
    """

    def test_the_flag_reaches_the_model_and_the_loss_together(self) -> None:
        """One argv, resolved by the real parser, driven into the real factory.

        Non-vacuous by construction: ``tiny`` has a 2-layer decoder, so
        ``num_aux_layers`` is 1 with the flag and 0 without, and the packed row
        count differs between the two arms. Both arms are asserted -- an
        assertion on the ON arm alone would be satisfied by a factory that
        turned deep supervision on unconditionally.
        """
        on = create_training_model(config_from_argv(
            _tiny_argv("--deep-supervision")))
        off = create_training_model(config_from_argv(_tiny_argv()))

        assert on.deep_supervision is True and off.deep_supervision is False
        expected_aux = int(on.sam3.transformer.num_layers) - 1
        assert expected_aux > 0, (
            "this fixture's decoder has one layer, so `num_aux_layers` is 0 "
            "either way and every assertion below is vacuous")
        assert on.num_aux_layers == expected_aux and off.num_aux_layers == 0
        # The loss must agree, or `unpack_predictions` mis-slices in silence.
        assert on.loss.num_aux_layers == on.num_aux_layers
        assert off.loss.num_aux_layers == 0
        # And the agreement must be visible in the packed geometry itself.
        rows_on = on.compute_output_shape()[1]
        rows_off = off.compute_output_shape()[1]
        assert rows_off == on.num_queries + 1
        assert rows_on == (on.num_queries + 1) * (1 + expected_aux)
        assert rows_on != rows_off

    def test_the_config_default_is_off(self) -> None:
        """The pre-change world is what a bare command line still gets."""
        assert Sam3TrainingConfig().deep_supervision is False
        assert config_from_argv([]).deep_supervision is False

    def test_deep_supervision_is_absent_from_the_smoke_preset(self) -> None:
        """A preset may change HOW MUCH is measured, never WHAT (D-030).

        Deep supervision changes the SUPERVISION SIGNAL: the total the optimizer
        sees gains one equally-weighted term per earlier decoder layer. A smoke
        run that silently enabled it would be measuring a different objective
        from the real run it is supposed to be a wiring proof for.
        """
        assert "deep_supervision" not in SMOKE_PRESET
        assert config_from_argv(["--smoke"]).deep_supervision is (
            config_from_argv([]).deep_supervision)

    def test_the_flag_survives_the_preset_in_both_directions(self) -> None:
        """``--smoke`` must neither enable nor disable it."""
        assert config_from_argv(["--smoke", "--deep-supervision"]
                                ).deep_supervision is True
        assert config_from_argv(["--smoke", "--no-deep-supervision"]
                                ).deep_supervision is False

    def test_the_per_term_losses_are_computed_on_the_rows_the_loss_slices(
            self) -> None:
        """The eval path's packing stride, against an INDEPENDENT oracle.

        ``unpack_predictions`` derives ``Q`` as ``rows // (1 + num_aux_layers)
        - 1`` and validates nothing, so handing a deep-supervision loss a
        main-block-only tensor does NOT raise -- on ``tiny`` it reads Q=2
        instead of Q=5 and returns six finite, plausible, fabricated numbers.

        The oracle is a SEPARATE ``Sam3DetectionLoss`` at ``num_aux_layers=0``
        fed a main-block-only pack of the same batches: it is derived from the
        contract (``compute_terms`` reports the MAIN block only), not
        transcribed from ``evaluate_sam3``. RED when the eval path drops the
        auxiliary blocks (decisions.md D-006).
        """
        config = config_from_argv(_tiny_argv("--deep-supervision"))
        keras.utils.set_random_seed(4321)
        model = create_training_model(config)
        assert model.num_aux_layers > 0, "the arm under test is not on"
        _, val_dataset = create_datasets(config, model)

        measured = evaluate_sam3(model, val_dataset)

        oracle_loss = Sam3DetectionLoss(include_masks=config.include_masks,
                                        pad_n_queries=model.num_queries)
        assert oracle_loss.num_aux_layers == 0
        term_keys = ("loss_ce", "presence_loss", "loss_bbox", "loss_giou")
        totals = {key: 0.0 for key in term_keys}
        batches = 0
        for inputs, y_true in val_dataset:
            outputs = model.sam3(inputs, training=False)
            terms = oracle_loss.compute_terms(
                y_true,
                pack_predictions(outputs, include_masks=config.include_masks))
            for key in term_keys:
                totals[key] += float(terms[key])
            batches += 1
        assert batches > 0

        for key in term_keys:
            assert measured[key] == pytest.approx(totals[key] / batches,
                                                  rel=1e-5, abs=1e-6), (
                f"{key} disagrees with the main-block oracle: the eval path is "
                f"packing a row count the compiled loss does not slice")

    def test_a_deep_supervision_model_takes_a_real_training_step(self) -> None:
        """End to end through the trainer's own factory, one step.

        The row-stride agreement is checked at compile time by
        ``compile_sam3_trainer``, but nothing there executes the forward or the
        backward pass. A single ``fit`` step over the trainer's own dataset does.
        """
        config = config_from_argv(_tiny_argv("--deep-supervision"))
        keras.utils.set_random_seed(4321)
        model = create_training_model(config)
        train_dataset, _ = create_datasets(config, model)
        history = model.fit(train_dataset, epochs=1, verbose=0)
        assert model.jit_compile is False
        loss_value = float(history.history["loss"][-1])
        assert math.isfinite(loss_value)


class TestQuerySelectionCli:
    """``--query-selection`` through the FULL ``argv -> config -> factory``.

    Modelled on :class:`TestDeepSupervision`, which is this repository's shipped
    precedent for wiring a layout-changing boolean. The defect class these tests
    exist for is the documented one: a config field that is set correctly and
    never reaches the factory, so the run trains the OLD model while the config
    it writes to disk claims the new one. Asserting ``config.query_selection``
    alone cannot see that; every assertion below therefore terminates on
    ``model.num_aux_layers`` or on the model's own proposal head.
    """

    @staticmethod
    def _expected_aux(model: Any, deep: bool, query: bool) -> int:
        """The composition oracle (I-5), re-derived from the FLAGS.

        Interface contract: computed from ``deep`` / ``query`` and the decoder's
        own layer count, never read off ``model.num_aux_layers`` -- a model that
        composed the sum wrongly must not be able to make this test agree with
        itself.

        :param model: The built wrapper, read only for its decoder depth.
        :type model: Any
        :param deep: The deep-supervision arm under test.
        :type deep: bool
        :param query: The query-selection arm under test.
        :type query: bool
        :return: The expected auxiliary block count.
        :rtype: int
        """
        layers = int(model.sam3.transformer.num_layers)
        return (layers - 1 if deep else 0) + (1 if query else 0)

    @pytest.mark.parametrize("deep,query", [(False, False), (True, False),
                                            (False, True), (True, True)])
    def test_the_flag_reaches_the_model_at_all_four_combinations(
            self, deep: bool, query: bool) -> None:
        """argv -> parse_args -> config -> factory -> ``num_aux_layers``.

        All four combinations are driven through the REAL parser and the REAL
        factory, and the assertion lands on the composed row arithmetic, not on
        the config field. On ``tiny`` (2 decoder layers) the four combinations
        are 0/1/1/2 auxiliary blocks, so the ON arms are distinguishable from
        the OFF ones -- non-vacuity is proved separately below.
        """
        extra = []
        if deep:
            extra.append("--deep-supervision")
        if query:
            extra.append("--query-selection")
        config = config_from_argv(_tiny_argv(*extra))
        assert config.deep_supervision is deep
        assert config.query_selection is query

        keras.utils.set_random_seed(1357)
        model = create_training_model(config)

        expected = self._expected_aux(model, deep, query)
        assert model.num_aux_layers == expected, (
            f"--deep-supervision={deep} --query-selection={query} reached the "
            f"config but not the model: num_aux_layers={model.num_aux_layers}, "
            f"expected {expected} = (L-1)*deep + query")
        # The wrapper's flag and the loss's row stride must both follow, or
        # `unpack_predictions` mis-slices in silence.
        assert model.query_selection is query
        assert model.loss.num_aux_layers == expected
        assert model.compute_output_shape()[1] == (
            (model.num_queries + 1) * (1 + expected))
        # And the head itself must exist exactly when the flag says so: the
        # row arithmetic alone would be satisfied by a wrapper that counted a
        # block no head ever produces.
        assert bool(model.sam3.query_selection) is query
        assert (getattr(model.sam3, "query_selection_head", None)
                is not None) is query

    def test_the_four_combinations_are_not_all_the_same_model(self) -> None:
        """Non-vacuity for the parametrized test above.

        If every combination produced the same ``num_aux_layers``, the oracle
        would be satisfied by a factory that ignored both flags.
        """
        counts = set()
        for extra in ([], ["--deep-supervision"], ["--query-selection"],
                      ["--deep-supervision", "--query-selection"]):
            keras.utils.set_random_seed(1357)
            model = create_training_model(config_from_argv(_tiny_argv(*extra)))
            counts.add(model.num_aux_layers)
        assert counts == {0, 1, 2}, (
            f"the four flag combinations produced {sorted(counts)} auxiliary "
            f"block counts; expected 0/1/1/2 on a 2-layer decoder")

    def test_the_config_default_is_off(self) -> None:
        """The pre-change world is what a bare command line still gets."""
        assert Sam3TrainingConfig().query_selection is False
        assert config_from_argv([]).query_selection is False

    def test_query_selection_is_absent_from_the_smoke_preset(self) -> None:
        """A preset may change HOW MUCH is measured, never WHAT (D-030).

        Query selection changes the MODEL: it adds a weighted proposal head to
        the forward path and replaces the decoder's initial reference boxes. A
        smoke run that silently enabled it would be a wiring proof for a
        different architecture from the one the real run trains.
        """
        assert "query_selection" not in SMOKE_PRESET
        assert config_from_argv(["--smoke"]).query_selection is (
            config_from_argv([]).query_selection)

    def test_the_flag_survives_the_preset_in_both_directions(self) -> None:
        """``--smoke`` must neither enable nor disable an explicit flag.

        The provenance path (`explicitly_set_flags`) is what makes this hold:
        the preset is applied only to fields the caller did NOT type.
        """
        assert config_from_argv(["--smoke", "--query-selection"]
                                ).query_selection is True
        assert config_from_argv(["--smoke", "--no-query-selection"]
                                ).query_selection is False
        # And the preset still does its own job in the same command line, so
        # the assertions above are not passing because `--smoke` was ignored.
        smoke = config_from_argv(["--smoke", "--query-selection"])
        assert smoke.smoke is True
        assert smoke.epochs == SMOKE_PRESET["epochs"]

    def test_the_two_flags_are_independent_through_the_preset(self) -> None:
        """`--smoke --deep-supervision --query-selection` is the smoke command.

        It is the exact argv this step runs on the GPU, so it is pinned here at
        no GPU cost: both flags survive, and the preset still shrinks the run.
        """
        config = config_from_argv(
            ["--smoke", "--deep-supervision", "--query-selection"])
        assert config.deep_supervision is True
        assert config.query_selection is True
        for field, value in SMOKE_PRESET.items():
            assert getattr(config, field) == value

    def test_a_query_selection_model_takes_a_real_training_step(self) -> None:
        """End to end through the trainer's own factory, one step.

        `compile_sam3_trainer` checks the row-stride agreement at compile time,
        but nothing there executes a forward or a backward pass through the new
        proposal head. One `fit` step over the trainer's own dataset does.
        """
        config = config_from_argv(
            _tiny_argv("--deep-supervision", "--query-selection"))
        keras.utils.set_random_seed(4321)
        model = create_training_model(config)
        assert model.num_aux_layers == 2, "the arm under test is not composed"
        train_dataset, _ = create_datasets(config, model)
        history = model.fit(train_dataset, epochs=1, verbose=0)
        assert model.jit_compile is False
        assert math.isfinite(float(history.history["loss"][-1]))



class TestPromptConditionedQueriesCli:
    """``--prompt-conditioned-queries`` through ``argv -> config -> factory``.

    Same defect class, same terminating discipline as
    :class:`TestQuerySelectionCli`: no assertion here stops at the config
    field. Each one lands on the MODEL -- the proposal head's own flag, its
    FiLM sub-layer, or the parameter count -- because a config field that never
    reaches the factory is exactly the silent no-op these tests exist for.
    """

    @pytest.mark.parametrize("conditioned", [False, True])
    def test_the_flag_reaches_the_proposal_head_itself(
            self, conditioned: bool) -> None:
        """argv -> parse_args -> config -> factory -> the head's own flag."""
        extra = ["--query-selection"]
        if conditioned:
            extra.append("--prompt-conditioned-queries")
        config = config_from_argv(_tiny_argv(*extra))
        assert config.prompt_conditioned_queries is conditioned

        keras.utils.set_random_seed(2468)
        model = create_training_model(config)
        head = model.sam3.query_selection_head
        assert model.sam3.prompt_conditioned_queries is conditioned
        assert head.prompt_conditioned is conditioned, (
            "--prompt-conditioned-queries reached the config but not the "
            "proposal head: the run would train the prompt-BLIND arm while "
            "its config.json claims the conditioned one")
        assert (head.prompt_film is not None) is conditioned

    def test_the_flag_changes_the_parameter_count_by_the_structure(
            self) -> None:
        """Non-vacuity for the pair above, from the STRUCTURE.

        The FiLM projection is one ``d_model -> 2 * d_model`` affine on the
        pooled prompt: one kernel plus its biases, enumerated here rather than
        read off the model.
        """
        keras.utils.set_random_seed(2468)
        off = create_training_model(
            config_from_argv(_tiny_argv("--query-selection")))
        keras.utils.set_random_seed(2468)
        on = create_training_model(config_from_argv(
            _tiny_argv("--query-selection", "--prompt-conditioned-queries")))
        width = int(off.sam3.d_model)
        assert on.count_params() - off.count_params() == (
            width * (2 * width) + 2 * width)

    def test_it_is_refused_without_query_selection(self) -> None:
        """There is no head to condition, so the flag would be a no-op."""
        config = config_from_argv(_tiny_argv("--prompt-conditioned-queries"))
        assert config.prompt_conditioned_queries is True
        assert config.query_selection is False
        with pytest.raises(ValueError, match="requires query_selection"):
            create_training_model(config)

    def test_the_config_default_is_off(self) -> None:
        assert Sam3TrainingConfig().prompt_conditioned_queries is False
        assert config_from_argv([]).prompt_conditioned_queries is False

    def test_it_is_absent_from_the_smoke_preset(self) -> None:
        """A preset may change HOW MUCH is measured, never WHAT (D-030)."""
        assert "prompt_conditioned_queries" not in SMOKE_PRESET
        assert config_from_argv(["--smoke"]).prompt_conditioned_queries is (
            config_from_argv([]).prompt_conditioned_queries)

    def test_the_flag_survives_the_preset_in_both_directions(self) -> None:
        """`--smoke` must neither enable nor disable an explicit flag."""
        smoke = config_from_argv(
            ["--smoke", "--query-selection", "--prompt-conditioned-queries"])
        assert smoke.prompt_conditioned_queries is True
        assert config_from_argv(
            ["--smoke", "--query-selection",
             "--no-prompt-conditioned-queries"]
        ).prompt_conditioned_queries is False
        # ... and the preset still did its own job in that same command line.
        assert smoke.smoke is True
        assert smoke.epochs == SMOKE_PRESET["epochs"]

    def test_a_prompt_conditioned_model_takes_a_real_training_step(
            self) -> None:
        """One `fit` step: a forward AND a backward pass through the FiLM
        projection, through the trainer's own factory and dataset."""
        config = config_from_argv(_tiny_argv(
            "--deep-supervision", "--query-selection",
            "--prompt-conditioned-queries"))
        keras.utils.set_random_seed(4321)
        model = create_training_model(config)
        film = model.sam3.query_selection_head.prompt_film[-1]
        before = np.asarray(film.weights[0])
        train_dataset, _ = create_datasets(config, model)
        history = model.fit(train_dataset, epochs=1, verbose=0)
        assert math.isfinite(float(history.history["loss"][-1]))
        assert float(np.max(np.abs(np.asarray(film.weights[0]) - before))) > 0.0, (
            "the FiLM projection did not move in a training step: it is on "
            "the forward path but not on the BACKWARD one")

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

    def test_a_constant_head_drives_the_across_image_metrics_to_zero(
            self, tiny_config: Sam3TrainingConfig, tiny_model: Any) -> None:
        """The guard that would have caught iteration 1's real failure mode.

        MEASURED on the saved step-7 checkpoints: ``box_std_across_images``
        0.00048 against ``box_std_across_queries`` 0.13951 -- every supervised
        head had converged to an image-INDEPENDENT constant while box IoU read a
        respectable 0.29, because the Hungarian matcher scores the best of Q
        constant boxes against whatever ground truth is present. No shipped
        guard could see it: ``pred_mask_unique_values`` watches ``pred_masks``,
        the one head that is OFF by default and was off in five of six arms.

        **M3 -- THE LIVENESS ARM IS THE POINT.** An across-image statistic
        measured on a model that is already constant reads ~0 whether the
        statistic is computed correctly or not, so this test runs BOTH arms
        against the same real dataset: a stub that forces the heads constant
        across the batch (the metric must collapse) and a stub that forces them
        strongly image-dependent (the metric must read large). Without the
        second arm a metric hardwired to ``0.0`` would pass.
        """

        class _HeadStub:
            """A real model with its head outputs rewritten, nothing else."""

            def __init__(self, model: Any, rewrite: Any) -> None:
                self._model = model
                self._rewrite = rewrite
                self.include_masks = model.include_masks
                self.loss = model.loss
                # Mirrored, not pinned to 0: `evaluate_sam3` reads it to
                # decide how many packed blocks the compiled loss slices
                # (decisions.md D-006), so a stub that hardcoded it would
                # stop standing in for the real wrapper.
                self.num_aux_layers = model.num_aux_layers

            def sam3(self, inputs: Any, training: bool = False) -> Any:
                return self._rewrite(
                    self._model.sam3(inputs, training=training))

        heads = ("pred_boxes", "pred_logits", "presence_logit")

        # The cache is load-bearing: a stub that only flattens WITHIN a batch
        # leaves the batch-to-batch variation intact, and the whole-split
        # statistic this metric computes would still read non-zero. Measured
        # while writing this test: 0.1616 for `pred_logits`.
        frozen: Dict[str, Any] = {}

        def constant(outputs: Dict[str, Any]) -> Dict[str, Any]:
            out = dict(outputs)
            for key in heads:
                value = keras.ops.convert_to_tensor(out[key])
                frozen.setdefault(key, value[:1])
                out[key] = keras.ops.repeat(
                    frozen[key], int(value.shape[0]), axis=0)
            return out

        def image_dependent(outputs: Dict[str, Any]) -> Dict[str, Any]:
            out = dict(outputs)
            count = int(keras.ops.convert_to_tensor(
                out["pred_boxes"]).shape[0])
            ramp = keras.ops.arange(count, dtype="float32") / max(count, 1)
            out["pred_boxes"] = keras.ops.clip(
                keras.ops.cast(out["pred_boxes"], "float32")
                + keras.ops.reshape(ramp, (count, 1, 1)), 0.01, 0.99)
            out["pred_logits"] = (
                keras.ops.cast(out["pred_logits"], "float32")
                + keras.ops.reshape(ramp, (count, 1, 1)) * 5.0)
            out["presence_logit"] = (
                keras.ops.cast(out["presence_logit"], "float32")
                + keras.ops.reshape(ramp, (count, 1)) * 5.0)
            return out

        wide = Sam3TrainingConfig(**{**TINY_KWARGS, "num_val_samples": 8})
        _, val_dataset = create_datasets(wide, tiny_model)

        dead = evaluate_sam3(_HeadStub(tiny_model, constant), val_dataset)
        live = evaluate_sam3(_HeadStub(tiny_model, image_dependent),
                             val_dataset)

        # RED arm: a head that ignores the image.
        assert dead["box_std_across_images"] == pytest.approx(0.0, abs=1e-6)
        assert dead["logit_std_across_images"] == pytest.approx(0.0, abs=1e-6)
        assert dead["presence_logit_std_across_images"] == pytest.approx(
            0.0, abs=1e-6)
        # ... while the ACROSS-QUERY spread survives untouched, which is why
        # the pair must be reported together: a head can be perfectly varied
        # over queries and perfectly blind to the image.
        assert dead["box_std_across_queries"] > 1e-3

        # LIVENESS arm: a head that reads the image, on the same data.
        assert live["box_std_across_images"] > 1e-2
        assert live["logit_std_across_images"] > 1e-1
        assert live["presence_logit_std_across_images"] > 1e-1

    def test_the_across_image_std_is_over_the_WHOLE_SPLIT_not_per_batch(
            self, tiny_config: Sam3TrainingConfig, tiny_model: Any) -> None:
        """The reduction AXIS the source comment argues for, pinned.

        ``evaluate_sam3`` concatenates every batch before reducing, and the
        comment beside it argues that "an across-image statistic computed inside
        a batch of 4 and then averaged is not the same number as one computed
        over the whole split". That property was INERT: replacing all four
        reductions with per-batch-then-average left the whole train suite at
        102 passed / 0 failed, because the RED arm above is constant across ALL
        batches (reads ~0 either way) and its liveness ramp restarts inside each
        batch (reads large either way). The gap is real -- MEASURED on
        ``results/step71_joint_seed1/final_model.keras``: whole-split
        ``box_std_across_images`` 2.0135e-05 against per-batch 1.4989e-05.

        This arm is the only one the two spellings score differently: the heads
        are held CONSTANT WITHIN each batch and made to vary BETWEEN batches, so
        the per-batch statistic is exactly 0 and the whole-split one is large.
        Non-INERT by construction, and the test asserts BOTH halves.
        """

        class _PerBatchStub:
            """Rewrites the heads to a per-batch constant, batch index k."""

            def __init__(self, model: Any) -> None:
                self._model = model
                self.include_masks = model.include_masks
                self.loss = model.loss
                # Mirrored, not pinned to 0: `evaluate_sam3` reads it to
                # decide how many packed blocks the compiled loss slices
                # (decisions.md D-006), so a stub that hardcoded it would
                # stop standing in for the real wrapper.
                self.num_aux_layers = model.num_aux_layers
                self.calls = 0
                self.seen: List[Dict[str, Any]] = []

            def sam3(self, inputs: Any, training: bool = False) -> Any:
                out = dict(self._model.sam3(inputs, training=training))
                step = float(self.calls)
                self.calls += 1
                for key, scale in (("pred_boxes", 0.05), ("pred_logits", 2.0),
                                   ("presence_logit", 2.0)):
                    value = keras.ops.cast(
                        keras.ops.convert_to_tensor(out[key]), "float32")
                    count = int(value.shape[0])
                    # One row, repeated: identical for every image in THIS
                    # batch. The offset makes batch k differ from batch k+1.
                    row = keras.ops.repeat(value[:1], count, axis=0)
                    out[key] = keras.ops.clip(
                        row * 0.0 + 0.2 + step * scale, 0.01, 0.99
                    ) if key == "pred_boxes" else row * 0.0 + step * scale
                self.seen.append({k: np.asarray(out[k]) for k in
                                  ("pred_boxes", "pred_logits",
                                   "presence_logit")})
                return out

        wide = Sam3TrainingConfig(**{**TINY_KWARGS, "num_val_samples": 8})
        _, val_dataset = create_datasets(wide, tiny_model)
        stub = _PerBatchStub(tiny_model)
        measured = evaluate_sam3(stub, val_dataset)

        assert stub.calls >= 2, (
            "INERT arm: a single batch cannot distinguish the two reductions")
        # Half 1 -- the PER-BATCH reduction reads exactly 0 on this input.
        per_batch = float(np.mean(
            [b["pred_boxes"].std(axis=0).mean() for b in stub.seen]))
        assert per_batch == pytest.approx(0.0, abs=1e-7), (
            f"INERT arm: the heads are not constant within a batch "
            f"({per_batch})")
        # Half 2 -- the WHOLE-SPLIT reduction, which is what ships, reads large.
        assert measured["box_std_across_images"] > 1e-2
        assert measured["logit_std_across_images"] > 1e-1
        assert measured["presence_logit_std_across_images"] > 1e-1

    def test_one_degenerate_UNMATCHED_pair_does_not_poison_the_whole_mean(
            self, tiny_config: Sam3TrainingConfig, tiny_model: Any) -> None:
        """``nan * 0.0 = nan``: masking by multiplication is not masking.

        Every UNMATCHED query gathers a padded, all-zero target row. If the
        prediction is also degenerate, ``iou_and_generalized_iou`` returns
        ``0/0 = nan`` for that pair -- and the old spelling ``iou * is_matched``
        propagates it through the ``0.0`` mask, so ONE such pair turns the whole
        split's box IoU into ``nan``. It first appeared for real in step 7's own
        calibration harness, whose ORACLE arm (predictions == ground truth,
        i.e. the arm that must score 1.0) returned ``nan`` on its first run.

        Reaching the corner needs BOTH sides degenerate, and MEASURED while
        writing this test, a zero ``pred_boxes`` alone is NOT enough: on an
        image that has ground truth, the assignment hands every unmatched query
        a REAL target row, so the gathered width is never zero. The padded
        all-zero row is only ever gathered on a ZERO-GT image. So the split is
        drawn at ``zero_instance_rate=0.5`` -- a mixture, which is also the
        shipped configuration's shape -- and the second assertion is the
        liveness half: some pair must still be matched, or ``box_iou`` would be
        ``nan`` through the empty-split branch instead and the test would pass
        for the wrong reason.
        """

        class _ZeroBoxStub:
            def __init__(self, model: Any) -> None:
                self._model = model
                self.include_masks = model.include_masks
                self.loss = model.loss
                # Mirrored, not pinned to 0: `evaluate_sam3` reads it to
                # decide how many packed blocks the compiled loss slices
                # (decisions.md D-006), so a stub that hardcoded it would
                # stop standing in for the real wrapper.
                self.num_aux_layers = model.num_aux_layers

            def sam3(self, inputs: Any, training: bool = False) -> Any:
                out = dict(self._model.sam3(inputs, training=training))
                out["pred_boxes"] = keras.ops.zeros_like(
                    keras.ops.cast(out["pred_boxes"], "float32"))
                return out

        mixed = Sam3TrainingConfig(**{**TINY_KWARGS, "num_val_samples": 16,
                                      "zero_instance_rate": 0.5})
        _, val_dataset = create_datasets(mixed, tiny_model)
        metrics = evaluate_sam3(_ZeroBoxStub(tiny_model), val_dataset)
        assert not math.isnan(metrics["box_iou"]), (
            "a padded, both-degenerate UNMATCHED pair poisoned the mean")
        assert metrics["num_matched_pairs"] > 0.0, (
            "no pair was matched, so the assertion above is vacuous")

    def test_the_selection_metric_is_an_achieved_metric_and_is_MAXIMIZED(
            self, tiny_config: Sam3TrainingConfig, tiny_model: Any,
            tmp_path) -> None:
        """CRITICAL 2: never select checkpoints on ``val_loss``.

        MEASURED at iteration 1: ``presence_loss`` is **61.7%** of ``val_loss``
        while its head's logit spread is 1.4e-04, so the majority of the
        selection scalar was a provably constant term. On seed 3 that picked
        epoch 6 over epoch 29 on a **0.07%** margin and cost box IoU 0.2360 vs
        0.2724 -- the number the whole verdict was quoted against.

        Two things are pinned, because either alone is insufficient:
        the monitored key must be an ACHIEVED metric this module computes, and
        the mode must be ``max``. ``train.common.create_callbacks`` derives
        ``mode`` as ``'max' if 'accuracy' in monitor else 'min'``, so routing
        ``val_box_iou`` through it unchanged would select the WORST epoch --
        silently, and in the direction that looks like a result.
        """
        from train.sam3.train_sam3 import build_callbacks

        assert SELECTION_METRIC == f"val_{SELECTION_METRIC[4:]}"
        assert SELECTION_METRIC[4:] in EVAL_METRIC_KEYS, (
            f"{SELECTION_METRIC} is not produced by evaluate_sam3; the "
            f"callbacks would silently select epoch 1 forever")

        _, val_dataset = create_datasets(tiny_config, tiny_model)
        callbacks = build_callbacks(tiny_config, tmp_path, val_dataset)
        selectors = [callback for callback in callbacks
                     if isinstance(callback, (keras.callbacks.EarlyStopping,
                                              keras.callbacks.ModelCheckpoint))]
        assert len(selectors) == 2, "the two selecting callbacks are not wired"
        for callback in selectors:
            assert callback.monitor == SELECTION_METRIC
            # Assert on `monitor_op` -- the comparison that actually EXECUTES
            # -- not on a `mode` string: `ModelCheckpoint` does not even keep
            # `mode` as an attribute, and `EarlyStopping` resolves `monitor_op`
            # lazily in `on_train_begin`, so resolve it here.
            if getattr(callback, "monitor_op", None) is None:
                callback._set_monitor_op()
            assert callback.monitor_op(1.0, 0.0), (
                f"{type(callback).__name__}'s monitor_op prefers the SMALLER "
                f"{SELECTION_METRIC}")

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


class _RowCountSpyLoss(Sam3DetectionLoss):
    """A real loss that RECORDS the row count of every tensor it is handed.

    Interface contract: identical to :class:`Sam3DetectionLoss` in every
    computation -- it overrides nothing but the recording -- and exposes
    ``observed_rows``, the ``y_pred.shape[1]`` of each ``compute_terms`` call in
    order. It exists because the packed tensor ``evaluate_sam3`` builds is a
    LOCAL: the only place a test can observe it is the object the function hands
    it to.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.observed_rows: List[int] = []

    def compute_terms(self, y_true: Any, y_pred: Any) -> Any:
        self.observed_rows.append(int(y_pred.shape[1]))
        return super().compute_terms(y_true, y_pred)


class TestEvaluateSam3PacksTheRowsTheLossSlices:
    """D-006's repair, extended to the encoder query selection block.

    ``evaluate_sam3`` packs its OWN copy of the prediction tensor, and it is the
    consumer that was BITTEN once already: a stride mismatch there does not
    raise -- ``unpack_predictions`` validates nothing by design -- it reports six
    finite, plausible, FABRICATED per-term losses every epoch. The oracle is the
    packed-layout arithmetic ``(Q + 1) * (1 + num_aux_layers)``, and
    ``num_aux_layers`` is itself re-derived here from the two flags rather than
    read off the model, so a model that composed it wrongly cannot make this
    test agree with itself.
    """

    @pytest.mark.parametrize("deep,query", [(False, False), (True, False),
                                            (False, True), (True, True)])
    def test_the_packed_row_count_matches_the_composition_oracle(
            self, deep: bool, query: bool) -> None:
        config = Sam3TrainingConfig(**TINY_KWARGS)
        keras.utils.set_random_seed(2468)
        sam3 = Sam3Image.from_variant("tiny", query_selection=True)
        model = Sam3TrainingModel(sam3, include_masks=config.include_masks,
                                  deep_supervision=deep, query_selection=query)
        model.build(None)

        layers = int(sam3.transformer.num_layers)
        expected_aux = (layers - 1 if deep else 0) + (1 if query else 0)
        assert model.num_aux_layers == expected_aux

        spy = _RowCountSpyLoss(include_masks=config.include_masks,
                               pad_n_queries=model.num_queries,
                               num_aux_layers=model.num_aux_layers)
        compile_sam3_trainer(model, loss=spy)
        _, val_dataset = create_datasets(config, model)

        metrics = evaluate_sam3(model, val_dataset)

        expected_rows = (model.num_queries + 1) * (1 + expected_aux)
        assert spy.observed_rows, "evaluate_sam3 scored no batch"
        assert set(spy.observed_rows) == {expected_rows}, (
            f"evaluate_sam3 packed {sorted(set(spy.observed_rows))} rows at "
            f"deep_supervision={deep}, query_selection={query}; the compiled "
            f"loss slices {expected_rows}")
        # Liveness: the run produced real numbers, so the row count above was
        # observed on a path that actually scored something.
        assert math.isfinite(metrics["loss_ce"])
        assert math.isfinite(metrics["box_iou"])

    def test_the_four_combinations_do_not_all_pack_the_same_row_count(
            self) -> None:
        """Non-vacuity for the parametrized test above.

        If every combination happened to produce the same row count, the oracle
        would be satisfied by a function that ignored the flags entirely. On
        `tiny` (2 decoder layers) the four combinations are 0/1/1/2 auxiliary
        blocks, i.e. three distinct row counts.
        """
        keras.utils.set_random_seed(2468)
        sam3 = Sam3Image.from_variant("tiny", query_selection=True)
        counts = set()
        for deep, query in ((False, False), (True, False), (False, True),
                            (True, True)):
            model = Sam3TrainingModel(sam3, deep_supervision=deep,
                                      query_selection=query)
            model.build(None)
            counts.add(model.compute_output_shape()[1])
        assert len(counts) >= 3
