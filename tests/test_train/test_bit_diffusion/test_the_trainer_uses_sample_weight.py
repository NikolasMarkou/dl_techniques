r"""Guard: ``w(t)`` reaches the loss as ``sample_weight``, through a STOCK ``fit()``.

The defect class this exists to catch
-------------------------------------
The bridge objective is ``mean((pred - target)**2 * w(t))`` with ``w(t)``
per-sample and direction-specific. The obvious way to get ``t`` into a Keras
loss is a custom ``train_step``, and it is forbidden here. So the whole
weighting rides in as ``sample_weight``, the third ``tf.data`` tuple element --
a mechanism with three independent ways to fail silently or loudly:

1. someone "simplifies" the trainer by overriding ``train_step`` (the
   ``src/train/sd3_mmdit/`` anti-pattern, which is untested);
2. the element degrades to a 2-tuple and the weighting silently vanishes --
   the loss still descends, it just optimizes a DIFFERENT objective;
3. the weight is emitted at rank 1 instead of rank 3.

(3) is the interesting one, and it is why this file pins a RANK rather than
merely asserting a weight exists. ``FlowMatchingVelocityLoss`` reduces only the
channel axis, so against the port's rank-4 prediction its value tensor is
``(B, H, W)``. A ``(B,)`` weight against that RAISES
``InvalidArgumentError: Incompatible shapes [B,H,W] vs [B]``. Measured at this
plan's step 1, and stock ``keras.losses.MeanSquaredError`` raises identically --
it is a general Keras property, not a quirk of one loss. Pinning the rank is
what converts a latent crash into a guarded contract, and
``test_a_rank_one_weight_still_raises`` re-derives the crash in-suite so the
reason for the rank cannot rot into folklore.

Traps designed out
------------------
ANTI-VACUITY. "Dropping the weight changes the loss" is worthless if the weight
happens to be all-ones, and near-worthless if it is constant across the batch:
a constant ``c`` merely rescales the loss and would not distinguish a correct
per-sample weighting from a wrong one. ``test_the_weight_is_not_constant``
asserts the emitted weight genuinely varies across samples BEFORE the
drop-the-weight arm is allowed to mean anything.

SHAPE AGREEMENT IS MEASURED, NOT ASSUMED. ``test_the_weight_matches_the_losss_
own_reduction_shape`` calls the real loss's ``call()`` on the real element and
compares shapes. A hand-written ``(B, H, W)`` expectation would go stale the day
someone changes the loss's reduction axis; this arm goes red instead.

THE SMOKE RUN ASSERTS A TRAJECTORY, NOT AN EXIT CODE. A training loop that
silently learns nothing exits 0. ``test_one_synthetic_epoch_pair_falls`` asserts
the last epoch's loss is below the first's AND that every reported number is
finite.

RED PROOFS (one injection per assertion, run against the REAL source, each
reverted immediately; observed text quoted, never predicted).
See ``plans/plan-2026-09-02T094601-77d4a04e/probes/step9_red_proof.txt``.

Nothing here writes into the repo-root ``results/``: the smoke run's
``output_dir`` is pytest's ``tmp_path``, and ``tests/conftest.py``'s autouse
fixture fails the test if an entry appears under ``results/`` anyway.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.losses.flow_matching_velocity_loss import (
    FlowMatchingVelocityLoss,
)
from train.bit_diffusion.synthetic_data import (
    build_bridge_dataset,
    prepare_training_batch,
    synthetic_records,
)
from train.bit_diffusion.train_bit_diffusion import (
    TrainingConfig,
    build_sde,
    create_model,
    train_bit_diffusion,
)

BATCH = 4
RECORDS = 16
SEED = 7


def _config(**overrides: Any) -> TrainingConfig:
    """A ``tiny``-geometry config, small enough to run on CPU in seconds."""
    base: Dict[str, Any] = dict(
        bridge_preset="tiny",
        variant="tiny",
        num_train_samples=RECORDS,
        num_val_samples=RECORDS,
        batch_size=BATCH,
        epochs=2,
        steps_per_epoch=8,
        validation_steps=2,
        warmup_epochs=0,
        learning_rate=3e-3,
        seed=SEED,
    )
    base.update(overrides)
    return TrainingConfig(**base)


def _element(**overrides: Any) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """One ``(inputs, target, sample_weight)`` triple from the real pipeline."""
    config = _config(**overrides)
    bridge = config.bridge_config
    records = synthetic_records(BATCH, bridge, seed=SEED)
    return prepare_training_batch(
        records,
        bridge,
        build_sde(config),
        direction_mode=config.direction,
        time_sampler=config.time_sampler,
        seed=SEED,
    )


def _dataset(config: TrainingConfig, steps: int) -> "tf.data.Dataset":
    """A finite dataset from the real builder, reproducible across calls."""
    bridge = config.bridge_config
    records = synthetic_records(RECORDS, bridge, seed=SEED)
    return build_bridge_dataset(
        records,
        bridge,
        build_sde(config),
        batch_size=config.batch_size,
        direction_mode=config.direction,
        time_sampler=config.time_sampler,
        seed=SEED,
        shuffle=False,
        steps=steps,
    )


def _compiled_model(config: TrainingConfig) -> keras.Model:
    model = create_model(config)
    model.compile(optimizer=keras.optimizers.SGD(0.0), loss=FlowMatchingVelocityLoss())
    return model


# ---------------------------------------------------------------------
# 1. No custom train_step
# ---------------------------------------------------------------------


def test_the_model_class_defines_no_train_step():
    """The whole design rests on the STOCK training step being the one that runs."""
    model = create_model(_config())
    assert type(model).train_step is keras.Model.train_step, (
        f"{type(model).__name__} overrides train_step. The bridge weighting "
        "reaches the loss as sample_weight precisely so that no custom "
        "train_step is needed; an override makes the sample_weight path dead "
        "code that no test can see."
    )
    # The whole MRO BELOW `keras.Model`, not just the leaf: an override on an
    # intermediate base runs just as silently. The slice stops AT `keras.Model`
    # on purpose -- `keras.Model.train_step` is inherited from the backend
    # trainer mixin (`TensorFlowTrainer`), so scanning the full MRO reports
    # Keras' own stock implementation as an override. Measured: this arm failed
    # with `['TensorFlowTrainer']` before the slice was added.
    mro = type(model).__mro__
    own_classes = mro[: mro.index(keras.Model)]
    overriding = [
        klass.__name__ for klass in own_classes if "train_step" in vars(klass)
    ]
    assert not overriding, (
        f"train_step is overridden somewhere in the MRO: {overriding}"
    )


# ---------------------------------------------------------------------
# 2. The element shape
# ---------------------------------------------------------------------


def test_the_dataset_element_is_a_three_tuple():
    """``fit()`` only ever sees a weight if the element carries a third slot."""
    dataset = _dataset(_config(), steps=1)
    spec = dataset.element_spec
    assert isinstance(spec, tuple) and len(spec) == 3, (
        f"the dataset element is {type(spec).__name__} of length "
        f"{len(spec) if isinstance(spec, tuple) else 'n/a'}; a 2-tuple silently "
        "drops w(t) and trains a different objective with no error"
    )
    inputs_spec, target_spec, weight_spec = spec
    assert set(inputs_spec) == {
        "x_t", "t", "y", "x_cond", "direction", "cond_mask"
    }, f"input keys drifted: {sorted(inputs_spec)}"
    assert len(target_spec.shape) == 4
    assert len(weight_spec.shape) == 3


def test_the_emitted_weight_has_rank_three():
    """RANK is the contract. See this module's docstring for why."""
    inputs, target, weight = _element()
    assert weight.ndim == 3, (
        f"sample_weight has rank {weight.ndim}, expected 3. A (B,) weight "
        "against this loss's (B,H,W) value tensor RAISES InvalidArgumentError "
        "inside fit(); pre-broadcasting to the reduction shape is the remedy."
    )
    assert weight.shape == target.shape[:3], (
        f"weight {weight.shape} does not match the target's leading axes "
        f"{target.shape[:3]}"
    )
    assert inputs["x_t"].shape == target.shape


def test_the_weight_matches_the_losss_own_reduction_shape():
    """Measured against the REAL loss, so a reduction-axis change goes red here."""
    _, target, weight = _element()
    per_sample = FlowMatchingVelocityLoss().call(
        keras.ops.convert_to_tensor(target),
        keras.ops.convert_to_tensor(target * 0.5),
    )
    assert tuple(per_sample.shape) == weight.shape, (
        f"the loss reduces to {tuple(per_sample.shape)} but the pipeline emits "
        f"a {weight.shape} weight. Keras multiplies these elementwise; a "
        "mismatch raises inside fit()."
    )


def test_a_rank_one_weight_still_raises():
    """The in-suite re-derivation of step 1's reading.

    This is what makes ``test_the_emitted_weight_has_rank_three`` a guard rather
    than a preference: if Keras ever started broadcasting ``(B,)`` against
    ``(B,H,W)``, this arm goes red and the rank pin can be retired deliberately
    instead of rotting.
    """
    config = _config()
    model = _compiled_model(config)
    flattened = _dataset(config, steps=1).map(
        lambda inputs, target, weight: (inputs, target, weight[:, 0, 0])
    )
    with pytest.raises(Exception) as excinfo:
        model.evaluate(flattened, verbose=0)
    assert "Incompatible shapes" in str(excinfo.value), (
        f"expected a shape incompatibility, got: {excinfo.value}"
    )


# ---------------------------------------------------------------------
# 3. The weight is live
# ---------------------------------------------------------------------


def test_the_weight_is_not_constant():
    """ANTI-VACUITY for the arm below. A constant weight only rescales."""
    _, _, weight = _element()
    per_sample = weight[:, 0, 0]
    assert np.all(np.isfinite(weight))
    assert float(per_sample.std()) > 1e-8, (
        f"w(t) is constant across the batch ({per_sample}); the "
        "drop-the-weight arm would then be measuring a global rescale, not a "
        "per-sample weighting"
    )
    assert not np.allclose(per_sample, 1.0), "w(t) is all-ones; nothing is weighted"
    # Spatially constant per sample -- the broadcast must not have transposed.
    assert np.allclose(weight, per_sample[:, None, None])


def test_dropping_the_weight_changes_the_reported_loss():
    """THE RED PROOF, run in-suite: strip the third element, get a different loss."""
    config = _config()
    model = _compiled_model(config)

    weighted = model.evaluate(_dataset(config, steps=2), verbose=0)
    unweighted = model.evaluate(
        _dataset(config, steps=2).map(
            lambda inputs, target, weight: (inputs, target)
        ),
        verbose=0,
    )
    weighted = float(np.reshape(weighted, (-1,))[0])
    unweighted = float(np.reshape(unweighted, (-1,))[0])

    assert math.isfinite(weighted) and math.isfinite(unweighted)
    relative = abs(weighted - unweighted) / max(abs(unweighted), 1e-12)
    assert relative > 1e-3, (
        f"weighted loss {weighted!r} and unweighted loss {unweighted!r} agree "
        f"to {relative:.3e}; w(t) is not reaching the loss"
    )


def test_the_weight_reaches_the_loss_with_the_value_the_pipeline_emitted():
    """Not merely "different" -- the loss equals the hand-computed weighted mean.

    ``sum_over_batch_size`` over a ``(B,H,W)`` value tensor times a ``(B,H,W)``
    weight is the plain mean of the product, which is upstream's
    ``mean((pred - target)**2 * w)``. Computing it by hand from the model's own
    prediction pins the ARITHMETIC, so a weight that arrives transposed, halved
    or attached to the wrong axis fails here even though it "changes the loss".
    """
    config = _config()
    model = _compiled_model(config)
    inputs, target, weight = _element()

    reported = float(
        np.reshape(
            model.evaluate(
                x=inputs, y=target, sample_weight=weight, batch_size=BATCH,
                verbose=0,
            ),
            (-1,),
        )[0]
    )
    prediction = keras.ops.convert_to_numpy(model(inputs, training=False))
    expected = float(
        np.mean(np.mean((prediction - target) ** 2, axis=-1) * weight)
    )
    assert reported == pytest.approx(expected, rel=1e-5), (
        f"fit()/evaluate() reports {reported!r}; the hand-computed weighted "
        f"mean is {expected!r}"
    )


# ---------------------------------------------------------------------
# 4. The smoke run
# ---------------------------------------------------------------------


@pytest.mark.integration
def test_one_synthetic_epoch_pair_falls_with_zero_nan(tmp_path):
    """A real ``train_bit_diffusion()`` run: falling loss, zero NaN.

    ``output_dir`` is ``tmp_path`` (ABSOLUTE), never the repo-root ``results/``.
    """
    config = _config(
        output_dir=str(tmp_path),
        experiment_name="smoke",
        steps_per_epoch=12,
        epochs=3,
    )
    model, history, run_dir = train_bit_diffusion(config)

    assert run_dir.is_dir()
    assert (run_dir / "config.json").is_file()
    assert type(model).train_step is keras.Model.train_step

    losses = [float(value) for value in history.history["loss"]]
    val_losses = [float(value) for value in history.history["val_loss"]]
    nan_count = sum(
        1 for value in losses + val_losses if not math.isfinite(value)
    )
    assert nan_count == 0, f"{nan_count} non-finite loss values: {losses} {val_losses}"
    assert len(losses) == len(val_losses) == 3

    # THE TRAJECTORY IS READ OFF `val_loss`, NOT `loss`, AND THAT IS THE WHOLE
    # POINT OF THIS ARM.
    #
    # The training dataset redraws `t` every step and `w(t)` spans orders of
    # magnitude across the bridge, so a per-epoch training mean is dominated by
    # which times happened to be drawn. MEASURED, this exact config re-run at
    # `learning_rate = 0.0` -- an optimizer that moves NOTHING:
    #
    #     lr=0.0    loss     = 2.669341, 2.462744, 2.299143   (FALLS)
    #     lr=0.0    val_loss = 3.024618, 3.024618, 3.024618   (exactly constant)
    #     lr=3e-3   loss     = 2.580215, 2.210123, 1.891905
    #     lr=3e-3   val_loss = 2.820462, 2.557399, 2.533167   (FALLS)
    #
    # So `assert losses[-1] < losses[0]` is a guard that CANNOT FAIL: it passes
    # with the learning rate set to zero. `val_loss` is exactly constant there,
    # because the validation dataset is FINITE, unshuffled and built from a
    # fixed seed, so its elements are byte-identical every epoch and the
    # comparison is a real before/after on one fixed batch set. Do NOT "harden"
    # this by also asserting on `loss`.
    # (A 4-epoch `--smoke` CLI run showed the same thing from the other side:
    # `loss` 2.241997 -> 3.043012, +0.80, while `val_loss` fell monotonically.)
    assert val_losses[-1] < val_losses[0], (
        f"val_loss did not fall: {val_losses}. A loop that trains nothing still "
        "exits 0. (train loss for reference, deliberately not asserted on: "
        f"{losses})"
    )
