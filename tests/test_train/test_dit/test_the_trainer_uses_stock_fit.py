r"""Guard: ``src/train/dit/train_dit.py`` trains under STOCK ``compile()``/``fit()``.

Two independent claims, both of which have failed silently in this repository
before.

1. NO CUSTOM ``train_step``
---------------------------
``plans/SYSTEM.md`` forbids a custom ``train_step``, and this port's whole
objective design (D-002: ``x_start`` and ``t`` packed into ``y_true``) exists so
that no override is needed. An override is easy to add later and produces no
shape, dtype or finiteness symptom -- it just quietly takes ownership of the
training math. The assertion is made on the model :func:`train_dit.create_model`
ACTUALLY BUILDS, not on a hand-constructed one: a test that constructs its own
``DiT`` proves something about the class, not about the trainer, and would stay
green if ``create_model`` returned a locally-subclassed wrapper.
:meth:`TestTheTrainerUsesStockFit.test_the_predicate_discriminates` proves the
``is`` comparison can be false, so the arm is not vacuous.

2. THE MODEL IS BUILT BEFORE ``fit()``
--------------------------------------
Keras 3 fires ``on_train_begin`` BEFORE the first batch, so a lazily-built model
hands every callback an EMPTY ``trainable_weights`` list. ``WeightEMACallback``
therefore averages nothing at that point; it defers, but upstream's recipe is to
have the weights there, and any other ``on_train_begin`` consumer added later
gets the empty list with no warning.

THE PROBE POINT IS LOAD-BEARING, AND WAS MEASURED. Removing the explicit
``model.build(...)`` from ``create_model`` takes the EMA shadow count AT
``on_train_begin`` from the full tensor count to ZERO -- but AFTER ``fit()``
BOTH arms read the full count, because the callback's deferral repairs it. An
after-fit probe is therefore INERT against exactly the defect this guard exists
to catch. :class:`TestTheAfterFitProbeIsInert` ships that measurement as an
executable arm so the weaker probe cannot be re-derived by someone who finds the
``on_train_begin`` ordering fiddly.

Callback ORDER is what makes the reading possible: Keras invokes callbacks in
list order, so ``WeightEMACallback`` is placed FIRST and the recording probe
SECOND, and the probe therefore sees the shadow set as the EMA callback left it
at ``on_train_begin``.

This file runs two real one-epoch ``fit()`` calls over the step-11 pipeline on
the ``--smoke`` geometry. It writes nothing to ``results/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

import keras

from dl_techniques.losses.ddpm_hybrid_loss import DDPMHybridLoss
from dl_techniques.models.vision_language.dit.model import DiT, create_dit
from train.dit import train_dit as trainer
from train.dit.ema_callback import WeightEMACallback
from train.dit.synthetic_data import build_dit_dataset, synthetic_records
from train.dit.train_dit import TrainingConfig, config_from_argv

STEPS = 1
BATCH = 4


@pytest.fixture(scope="module")
def config() -> TrainingConfig:
    """The trainer's own ``--smoke`` config, shrunk to one tiny batch.

    Driven through ``config_from_argv`` rather than constructed by hand so the
    geometry under test is the geometry the CLI produces.
    """
    base = config_from_argv(["--smoke"])
    return TrainingConfig(**{
        **{f.name: getattr(base, f.name) for f in base.__dataclass_fields__.values()},
        "batch_size": BATCH,
        "epochs": 1,
        "steps_per_epoch": STEPS,
        "validation_steps": 1,
    })


@pytest.fixture(scope="module")
def built_model(config: TrainingConfig) -> DiT:
    """The model the TRAINER builds -- ``create_model``, not a local ``DiT()``."""
    return trainer.create_model(config)


def _compile(model: keras.Model, config: TrainingConfig) -> keras.Model:
    model.compile(
        optimizer=trainer.build_dit_optimizer(config, config.steps_per_epoch),
        loss=DDPMHybridLoss(
            schedule_name=config.schedule_name,
            num_timesteps=config.num_timesteps,
            in_channels=config.in_channels,
        ),
    )
    return model


def _dataset(config: TrainingConfig):
    geometry = config.diffusion_config
    records = synthetic_records(
        config.num_train_samples, geometry, seed=config.seed
    )
    return build_dit_dataset(
        records, geometry, batch_size=config.batch_size, seed=config.seed,
        steps=STEPS,
    )


class _ShadowProbe(keras.callbacks.Callback):
    """Records the EMA shadow count AT ``on_train_begin`` and after ``fit``.

    Interface contract: read-only. Must be listed AFTER the EMA callback so it
    observes the shadow set as ``on_train_begin`` left it.
    """

    def __init__(self, ema: WeightEMACallback) -> None:
        super().__init__()
        self.ema = ema
        self.seen: Dict[str, int] = {}

    def on_train_begin(self, logs: Any = None) -> None:
        self.seen["shadows_at_train_begin"] = len(self.ema.shadow_values())
        self.seen["trainable_at_train_begin"] = len(self.model.trainable_weights)

    def on_train_end(self, logs: Any = None) -> None:
        self.seen["shadows_after_fit"] = len(self.ema.shadow_values())
        self.seen["trainable_after_fit"] = len(self.model.trainable_weights)


def _fit_and_probe(model: keras.Model, config: TrainingConfig) -> Dict[str, int]:
    ema = WeightEMACallback(decay=config.ema_decay)
    probe = _ShadowProbe(ema)
    model.fit(
        _dataset(config),
        epochs=1,
        steps_per_epoch=STEPS,
        verbose=0,
        callbacks=[ema, probe],  # ORDER matters: ema first, probe second.
    )
    return probe.seen


@pytest.fixture(scope="module")
def built_reading(config: TrainingConfig, built_model: DiT) -> Dict[str, int]:
    return _fit_and_probe(_compile(built_model, config), config)


@pytest.fixture(scope="module")
def unbuilt_reading(config: TrainingConfig) -> Dict[str, int]:
    """The SAME geometry with the explicit build omitted -- the injection.

    This is ``create_model``'s body minus the ``model.build(...)`` call, so the
    inertness arm measures the real alternative rather than a caricature.
    """
    geometry = config.diffusion_config
    model = create_dit(
        config.variant,
        input_size=geometry.input_size,
        in_channels=geometry.in_channels,
        num_classes=geometry.num_classes,
        class_dropout_rate=geometry.class_dropout_rate,
        learn_sigma=geometry.learn_sigma,
        mlp_ratio=geometry.mlp_ratio,
        dropout_rate=config.dropout_rate,
        label_seed=config.seed,
    )
    return _fit_and_probe(_compile(model, config), config)


# ---------------------------------------------------------------------
# 1. no custom train_step
# ---------------------------------------------------------------------


class TestTheTrainerUsesStockFit:
    def test_the_model_the_trainer_builds_uses_kerass_train_step(
        self, built_model: DiT
    ) -> None:
        assert type(built_model).train_step is keras.Model.train_step, (
            "train_dit.create_model returned a model with an overridden "
            "train_step. plans/SYSTEM.md forbids it, and the packed-y_true "
            "objective (D-002) exists precisely so no override is needed."
        )

    def test_test_and_predict_steps_are_stock_too(self, built_model: DiT) -> None:
        assert type(built_model).test_step is keras.Model.test_step
        assert type(built_model).predict_step is keras.Model.predict_step

    def test_it_still_holds_after_the_trainers_own_compile(
        self, config: TrainingConfig, built_model: DiT
    ) -> None:
        """`compile()` cannot change it -- but a wrapper installed there could."""
        _compile(built_model, config)
        assert type(built_model).train_step is keras.Model.train_step

    def test_the_predicate_discriminates(self) -> None:
        """Anti-vacuity: the ``is`` comparison is capable of being FALSE."""

        class _Overridden(DiT):
            def train_step(self, data):  # pragma: no cover - never fitted
                return super().train_step(data)

        assert _Overridden.train_step is not keras.Model.train_step

    def test_no_file_in_the_port_defines_a_train_step(self) -> None:
        """A whole-package text sweep, in case an override arrives elsewhere.

        The ``is`` comparison above already covers the model the trainer builds;
        this catches an override added to a sibling module (a subclass used by a
        future ``--resume`` path, say) before anyone wires it in.
        """
        import dl_techniques.models.vision_language.dit as dit_package

        roots = (
            Path(trainer.__file__).parent,          # src/train/dit/
            Path(dit_package.__file__).parent,      # the model package
        )
        offenders = []
        for root in roots:
            for path in sorted(root.glob("*.py")):
                if "def train_step" in path.read_text(encoding="utf-8"):
                    offenders.append(str(path))
        assert not offenders, f"custom train_step defined in: {offenders}"


# ---------------------------------------------------------------------
# 2. the model is built before fit()
# ---------------------------------------------------------------------


class TestTheModelIsBuiltBeforeFit:
    def test_create_model_returns_a_built_model(self, built_model: DiT) -> None:
        assert built_model.built
        assert len(built_model.trainable_weights) > 0

    def test_on_train_begin_sees_every_trainable_weight(
        self, built_model: DiT, built_reading: Dict[str, int]
    ) -> None:
        """THE arm. RED when the explicit build is removed from create_model.

        The count is not a pasted constant: it is the model's own trainable
        tensor count, so it tracks the architecture instead of going stale.
        """
        expected = len(built_model.trainable_weights)
        assert expected > 0
        assert built_reading["trainable_at_train_begin"] == expected
        assert built_reading["shadows_at_train_begin"] == expected, (
            "WeightEMACallback shadowed "
            f"{built_reading['shadows_at_train_begin']} of {expected} tensors "
            "at on_train_begin. create_model must build the model BEFORE fit()."
        )


class TestTheAfterFitProbeIsInert:
    """Why the reading is taken at ``on_train_begin`` and nowhere else.

    MEASURED: the unbuilt arm reads ZERO shadows at ``on_train_begin`` and the
    FULL count after ``fit()`` -- identical to the built arm's after-fit
    reading. Any guard written after ``fit()`` therefore passes against the
    defect it claims to catch.
    """

    def test_the_unbuilt_arm_is_empty_at_on_train_begin(
        self, unbuilt_reading: Dict[str, int]
    ) -> None:
        assert unbuilt_reading["trainable_at_train_begin"] == 0
        assert unbuilt_reading["shadows_at_train_begin"] == 0

    def test_but_the_two_arms_agree_after_fit(
        self, built_reading: Dict[str, int], unbuilt_reading: Dict[str, int]
    ) -> None:
        assert (
            unbuilt_reading["shadows_after_fit"]
            == built_reading["shadows_after_fit"]
            > 0
        ), (
            "the after-fit shadow counts differ, so this inertness claim needs "
            "re-measuring before anyone relies on it"
        )
        assert (
            unbuilt_reading["trainable_after_fit"]
            == built_reading["trainable_after_fit"]
        )
