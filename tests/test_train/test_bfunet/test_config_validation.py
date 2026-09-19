"""Config-time refusals of the bfunet denoiser trainers.

The audit of the ConvUNeXt trainer found 17 bad values accepted at construction that then
either failed late (after the run directory and the model existed) or, worse, did not fail
(``noise_sigma_min > sigma_max_start`` samples an inverted range and ``--max-val-files 0``
silently meant 500). Each bad value below is refused at ``__post_init__`` with a message
that names the field, and each rule has an accepted boundary twin so the refusal cannot be
satisfied by refusing everything.

Every expected literal is typed here (field names, accepted values); nothing is imported
from the code under test except the classes being constructed. Refusals are pure value
checks, so no test touches the filesystem or the GPU.
"""

from __future__ import annotations

import os
import re

os.environ.setdefault("MPLBACKEND", "Agg")

import pytest  # noqa: E402

from train.bfunet import common  # noqa: E402
from train.bfunet.common import BFUnetTrainingConfig  # noqa: E402
from train.bfunet.train_bfcnn_denoiser import BFCNNTrainingConfig  # noqa: E402
from train.bfunet.train_convunext_denoiser import TrainingConfig  # noqa: E402
from train.bfunet.train_unet_denoiser import TrainingConfig as UNetTrainingConfig  # noqa: E402


def _names(field: str) -> str:
    """Regex matching ``field`` as a whole identifier (not as a suffix of another)."""
    return rf"(?<!\w){re.escape(field)}(?!\w)"


# (row id, kwargs, field the message must name). Every kwargs set is INVALID.
SHARED_BAD_ROWS = [
    ("sigma_min_above_start", dict(noise_sigma_min=0.05, sigma_max_start=0.025),
     "noise_sigma_min"),
    ("lr_schedule_bogus", dict(lr_schedule_type="bogus"), "lr_schedule_type"),
    ("lr_schedule_exponential_decay", dict(lr_schedule_type="exponential_decay"),
     "lr_schedule_type"),
    ("lr_schedule_constant", dict(lr_schedule_type="constant"), "lr_schedule_type"),
    ("lr_schedule_cosine", dict(lr_schedule_type="cosine"), "lr_schedule_type"),
    ("optimizer_bogus", dict(optimizer_type="bogus"), "optimizer_type"),
    ("warmup_above_epochs", dict(epochs=10, warmup_epochs=11), "warmup_epochs"),
    ("warmup_negative", dict(epochs=10, warmup_epochs=-1), "warmup_epochs"),
    ("batch_size_zero", dict(batch_size=0), "batch_size"),
    ("batch_size_negative", dict(batch_size=-4), "batch_size"),
    ("epochs_zero", dict(epochs=0), "epochs"),
    ("epochs_negative", dict(epochs=-3), "epochs"),
    ("curriculum_schedule_bogus", dict(curriculum_schedule="bogus"), "curriculum_schedule"),
    ("curriculum_epochs_zero", dict(curriculum_epochs=0), "curriculum_epochs"),
    ("patches_per_image_zero", dict(patches_per_image=0), "patches_per_image"),
    ("learning_rate_zero", dict(learning_rate=0.0), "learning_rate"),
    ("learning_rate_negative", dict(learning_rate=-1.0), "learning_rate"),
    ("weight_decay_negative", dict(weight_decay=-0.1), "weight_decay"),
    ("gradient_clipping_negative", dict(gradient_clipping=-1.0), "gradient_clipping"),
    ("mix_ratio_above_one", dict(self_iterate_mix_ratio=7.0), "self_iterate_mix_ratio"),
    ("mix_ratio_negative", dict(self_iterate_mix_ratio=-0.1), "self_iterate_mix_ratio"),
    ("viz_freq_zero", dict(viz_freq=0), "viz_freq"),
    ("viz_samples_zero", dict(viz_samples=0), "viz_samples"),
    ("steps_per_epoch_zero", dict(steps_per_epoch=0), "steps_per_epoch"),
    ("validation_steps_zero", dict(validation_steps=0), "validation_steps"),
    ("max_train_files_zero", dict(max_train_files=0), "max_train_files"),
    ("max_val_files_zero", dict(max_val_files=0), "max_val_files"),
    ("max_val_files_negative", dict(max_val_files=-5), "max_val_files"),
    ("mixed_precision_with_expose_bottleneck",
     dict(mixed_precision=True, expose_bottleneck=True), "mixed_precision"),
    ("deep_supervision", dict(enable_deep_supervision=True), "enable_deep_supervision"),
    ("exp_curriculum_zero_start",
     dict(curriculum_schedule="exp", sigma_max_start=0.0), "sigma_max_start"),
]

# (row id, kwargs). Every kwargs set is VALID: the boundary twin of a refusal above.
SHARED_GOOD_ROWS = [
    ("sigma_min_equals_start", dict(noise_sigma_min=0.025, sigma_max_start=0.025)),
    ("lr_schedule_cosine_decay", dict(lr_schedule_type="cosine_decay")),
    ("lr_schedule_case_insensitive", dict(lr_schedule_type="Cosine_Decay")),
    ("optimizer_adamw", dict(optimizer_type="adamw")),
    ("optimizer_adam", dict(optimizer_type="adam")),
    ("optimizer_sgd", dict(optimizer_type="sgd")),
    ("optimizer_rmsprop", dict(optimizer_type="rmsprop")),
    ("optimizer_adadelta", dict(optimizer_type="adadelta")),
    # optimizer_builder also builds these three; refusing them would remove a working path.
    ("optimizer_sgld", dict(optimizer_type="sgld")),
    ("optimizer_vsgd", dict(optimizer_type="vsgd")),
    ("optimizer_gefen", dict(optimizer_type="gefen")),
    ("optimizer_case_insensitive", dict(optimizer_type="AdamW")),
    ("warmup_equals_epochs", dict(epochs=10, warmup_epochs=10)),
    ("warmup_zero", dict(epochs=10, warmup_epochs=0)),
    ("one_epoch_derived_warmup", dict(epochs=1)),
    ("batch_size_one", dict(batch_size=1)),
    ("epochs_one", dict(epochs=1, warmup_epochs=0)),
    ("curriculum_schedule_cosine", dict(curriculum_schedule="cosine")),
    ("curriculum_schedule_exp", dict(curriculum_schedule="exp", sigma_max_start=0.01)),
    ("curriculum_schedule_linear_zero_start",
     dict(curriculum_schedule="linear", sigma_max_start=0.0)),
    ("curriculum_epochs_one", dict(curriculum_epochs=1)),
    ("patches_per_image_one", dict(patches_per_image=1)),
    ("learning_rate_tiny", dict(learning_rate=1e-9)),
    ("weight_decay_zero", dict(weight_decay=0.0)),
    ("gradient_clipping_zero", dict(gradient_clipping=0.0)),
    ("mix_ratio_zero", dict(self_iterate_mix_ratio=0.0)),
    ("mix_ratio_one", dict(self_iterate_mix_ratio=1.0)),
    ("viz_freq_one", dict(viz_freq=1)),
    ("viz_samples_one", dict(viz_samples=1)),
    ("steps_per_epoch_unset", dict(steps_per_epoch=None)),
    ("steps_per_epoch_one", dict(steps_per_epoch=1)),
    ("validation_steps_unset", dict(validation_steps=None)),
    ("validation_steps_one", dict(validation_steps=1)),
    ("max_train_files_unset", dict(max_train_files=None)),
    ("max_train_files_one", dict(max_train_files=1)),
    ("max_val_files_unset", dict(max_val_files=None)),
    ("max_val_files_one", dict(max_val_files=1)),
    ("mixed_precision_alone", dict(mixed_precision=True)),
    ("expose_bottleneck_alone", dict(expose_bottleneck=True)),
    ("deep_supervision_off", dict(enable_deep_supervision=False)),
]

# ConvUNeXt-only rules (model constraints the other two trainers do not share).
CONVUNEXT_BAD_ROWS = [
    # No-projection stem, cross-channel arm: gabor_filters must equal initial_filters.
    # tiny resolves initial_filters to 32, the default gabor_filters is 32, so 24 differs.
    ("no_projection_width_mismatch",
     dict(variant="tiny", gabor_stem_projection=False, gabor_filters=24),
     "gabor_filters"),
    # The RESOLVED width: the override (24) replaces the variant's 32, so the default
    # gabor_filters=32 now mismatches even though it equals the variant default.
    ("no_projection_width_uses_resolved_initial_filters",
     dict(variant="tiny", gabor_stem_projection=False, initial_filters=24),
     "gabor_filters"),
    # Depthwise arm: channels(3) * per_channel(4) = 12 != tiny's 32.
    ("no_projection_depthwise_width_mismatch",
     dict(variant="tiny", gabor_stem_projection=False, gabor_filters_per_channel=4),
     "gabor_filters_per_channel"),
    # tiny's 32 filters are not divisible by 3 groups.
    ("final_projection_groups_not_dividing_filters",
     dict(variant="tiny", final_projection_groups=3), "final_projection_groups"),
    # -1 resolves to channels(3) groups, and 32 % 3 != 0.
    ("final_projection_groups_minus_one_not_dividing",
     dict(variant="tiny", final_projection_groups=-1), "final_projection_groups"),
    # 3 input channels are not divisible by 2 groups.
    ("final_projection_groups_not_dividing_channels",
     dict(variant="tiny", final_projection_groups=2), "final_projection_groups"),
    ("freeze_stem_without_stem",
     dict(trainable_gabor_stem=False, use_gabor_stem=False), "trainable_gabor_stem"),
]

CONVUNEXT_GOOD_ROWS = [
    ("no_projection_width_matches",
     dict(variant="tiny", gabor_stem_projection=False, gabor_filters=32)),
    ("no_projection_width_matches_override",
     dict(variant="tiny", gabor_stem_projection=False, initial_filters=24, gabor_filters=24)),
    ("no_projection_depthwise_width_matches",
     dict(variant="tiny", gabor_stem_projection=False, initial_filters=12,
          gabor_filters_per_channel=4)),
    # With the projection kept there is no width rule at all.
    ("projection_kept_width_free",
     dict(variant="tiny", gabor_stem_projection=True, gabor_filters=24)),
    ("no_stem_needs_no_width", dict(variant="tiny", use_gabor_stem=False,
                                    gabor_stem_projection=False, gabor_filters=24)),
    ("final_projection_groups_divides",
     dict(variant="tiny", initial_filters=24, final_projection_groups=3)),
    ("final_projection_groups_minus_one_divides",
     dict(variant="tiny", initial_filters=24, final_projection_groups=-1)),
    ("final_projection_groups_one", dict(variant="tiny", final_projection_groups=1)),
    ("freeze_stem_with_stem", dict(trainable_gabor_stem=False, use_gabor_stem=True)),
    ("no_stem_without_freeze", dict(use_gabor_stem=False, trainable_gabor_stem=True)),
]

# The shared rules live in the base class, so every trainer inherits them.
SHARED_CLASSES = [
    pytest.param(BFUnetTrainingConfig, id="base"),
    pytest.param(TrainingConfig, id="convunext"),
    pytest.param(UNetTrainingConfig, id="unet"),
    pytest.param(BFCNNTrainingConfig, id="bfcnn"),
]


@pytest.mark.parametrize("cls", SHARED_CLASSES)
@pytest.mark.parametrize(
    "kwargs, field", [pytest.param(k, f, id=i) for i, k, f in SHARED_BAD_ROWS]
)
def test_shared_bad_value_is_refused_naming_the_field(cls, kwargs, field) -> None:
    with pytest.raises(ValueError, match=_names(field)):
        cls(**kwargs)


@pytest.mark.parametrize("cls", SHARED_CLASSES)
@pytest.mark.parametrize("kwargs", [pytest.param(k, id=i) for i, k in SHARED_GOOD_ROWS])
def test_shared_boundary_twin_is_accepted(cls, kwargs) -> None:
    cls(**kwargs)


@pytest.mark.parametrize(
    "kwargs, field", [pytest.param(k, f, id=i) for i, k, f in CONVUNEXT_BAD_ROWS]
)
def test_convunext_bad_value_is_refused_naming_the_field(kwargs, field) -> None:
    with pytest.raises(ValueError, match=_names(field)):
        TrainingConfig(**kwargs)


@pytest.mark.parametrize("kwargs", [pytest.param(k, id=i) for i, k in CONVUNEXT_GOOD_ROWS])
def test_convunext_boundary_twin_is_accepted(kwargs) -> None:
    TrainingConfig(**kwargs)


def test_the_row_tables_reach_at_least_twenty_refusals() -> None:
    assert len(SHARED_BAD_ROWS) + len(CONVUNEXT_BAD_ROWS) >= 20


# ---- warmup boundary -------------------------------------------------------


def _record_warnings(monkeypatch) -> list:
    seen: list = []
    monkeypatch.setattr(common.logger, "warning", lambda message, *a, **k: seen.append(str(message)))
    return seen


def test_warmup_equal_to_epochs_warns_that_the_cosine_never_starts(monkeypatch) -> None:
    seen = _record_warnings(monkeypatch)
    config = TrainingConfig(epochs=10, warmup_epochs=10)
    assert config.warmup_epochs == 10
    assert any("warmup_epochs" in m and "cosine" in m for m in seen), seen


def test_warmup_below_epochs_does_not_warn(monkeypatch) -> None:
    seen = _record_warnings(monkeypatch)
    TrainingConfig(epochs=10, warmup_epochs=9)
    assert not any("warmup_epochs" in m for m in seen), seen


def test_one_epoch_with_the_derived_warmup_is_accepted_with_a_warning(monkeypatch) -> None:
    seen = _record_warnings(monkeypatch)
    config = TrainingConfig(epochs=1)
    assert config.warmup_epochs == 1  # max(1, round(0.1 * 1))
    assert any("warmup_epochs" in m for m in seen), seen


# ---- ordering and purity ---------------------------------------------------


def test_symmetry_with_deep_supervision_keeps_the_symmetry_message() -> None:
    """The deep-supervision refusal comes AFTER the symmetry-vs-deep-supervision check, so
    the older, more specific message (asserted by test_convunext_self_iterate.py) survives."""
    with pytest.raises(ValueError, match="Jacobian-symmetry"):
        TrainingConfig(symmetry_weight=0.1, enable_deep_supervision=True)


def test_the_deep_supervision_message_says_the_trainer_does_not_wire_it() -> None:
    with pytest.raises(ValueError, match="not wired"):
        TrainingConfig(enable_deep_supervision=True)


def test_a_config_never_reads_the_filesystem() -> None:
    """Directories that do not exist do not stop a config: it must not depend on the
    machine (the preflight in train() owns that check)."""
    config = TrainingConfig(
        train_image_dirs=["/nonexistent/train_dir_for_config_test"],
        val_image_dirs=["/nonexistent/val_dir_for_config_test"],
    )
    assert config.train_image_dirs == ["/nonexistent/train_dir_for_config_test"]


@pytest.mark.parametrize("cls", SHARED_CLASSES)
def test_default_configs_still_construct(cls) -> None:
    config = cls()
    assert config.warmup_epochs == 10
    assert config.epochs == 100
    assert config.max_train_files == 10000
    assert config.max_val_files == 500
    assert config.lr_schedule_type == "cosine_decay"
    assert config.optimizer_type == "adamw"
