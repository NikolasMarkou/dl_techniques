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
    "variant": ("--variant", "tiny"),                             # default: small
    "patch_size": ("--patch-size", 8),                            # default: None
    "dino_out_dim": ("--dino-out-dim", 4096),                     # default: 65536
    "student_temp": ("--student-temp", 0.2),                      # default: 0.1
    "teacher_temp": ("--teacher-temp", 0.05),                     # default: 0.04
    "teacher_temp_final": ("--teacher-temp-final", 0.09),         # default: 0.07
    "teacher_temp_warmup_epochs": ("--teacher-temp-warmup-epochs", 7),   # default: 30
    "center_momentum": ("--center-momentum", 0.95),               # default: 0.9
    "ema_decay_start": ("--ema-decay-start", 0.99),               # default: 0.996
    "ema_decay_end": ("--ema-decay-end", 0.999),                  # default: 0.9999
    "ema_warmup_steps": ("--ema-warmup-steps", 11),               # default: 0
    "batch_size": ("--batch-size", 8),                            # default: 32
    "epochs": ("--epochs", 3),                                    # default: 100
    "learning_rate": ("--learning-rate", 1e-5),                   # default: 5e-4
    "optimizer": ("--optimizer", "sgd"),                          # default: adamw
    "lr_schedule": ("--lr-schedule", "constant"),                 # default: cosine_decay
    "warmup_epochs": ("--warmup-epochs", 1),                      # default: 10
    "weight_decay": ("--weight-decay", 0.01),                     # default: 0.04
    "gradient_clipping": ("--gradient-clipping", 0.5),            # default: 3.0
    "early_stopping_patience": ("--early-stopping-patience", 4),  # default: 30
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
            argv.append(flag)
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
