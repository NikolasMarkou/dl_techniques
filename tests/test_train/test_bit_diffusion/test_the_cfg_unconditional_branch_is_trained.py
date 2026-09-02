r"""Guard: the trainer actually TRAINS the branch ``forward_with_cfg`` samples.

The defect this exists to catch
-------------------------------
``DiTXA.forward_with_cfg`` builds its unconditional pass by setting ``cond_mask``
to all-zeros, which (per ``_embed_conditioning``) makes the conditioning stream
the exact zero tensor. Until step 9.1 the training pipeline emitted
``cond_mask = np.ones(...)`` unconditionally, with a comment asserting as fact
that "upstream applies NO conditioning dropout during training". Both halves of
that claim were false, and the consequence was not a doc nit: a model trained by
this trainer would never have seen ``cond_mask = 0``, so the unconditional pass
of ``cond + s * (cond - uncond)`` would be evaluated out of distribution and the
guidance term would be noise. A correct, anchored, well-guarded CFG
implementation on top of a training recipe that cannot make it mean anything.

The reference, quoted from this plan's staged ingest:

* ``FULL_INGEST.py:1572`` and ``:1806`` -- both upstream production launchers
  pass ``--unconditional-percent 0.3``.
* ``cond_mask`` is a parameter of upstream's TRAINING losses, not just its
  sampler: ``dsm_loss(..., cond_mask=None, ...)`` at ``:829`` threads it to the
  model at ``:859``; likewise ``flow_matching_loss`` (``:883``/``:901``) and
  ``edm_dsm_loss`` (``:920``/``:944``). An inference-only mask could not be a
  training-loss parameter.
* ``class_dropout_prob`` is a DIFFERENT knob, fed from ``prompt_kind_dropout``
  at ``:2459``; there is no ``--class-dropout-*`` flag anywhere in the ingest.
  It drops the prompt-kind LABEL, not the conditioning stream, so it is not
  what ``forward_with_cfg`` perturbs.

See ``decisions.md`` D-031 and the anchor at ``synthetic_data.py``.

Traps designed out
------------------
THE ENDPOINTS ARE ASSERTED EXACTLY, NOT STATISTICALLY. ``p = 0.0`` must give
every mask a 1 and ``p = 1.0`` must give every mask a 0. Those are the two
readings a "roughly the right rate" arm would pass while the comparison operator
was inverted or off by one draw. ``Generator.random`` draws from ``[0, 1)``, so
``>= p`` makes both endpoints exact rather than merely overwhelmingly likely.

THE TOLERANCE IS DERIVED, NOT GUESSED. The rate arm's bound is
``4 * sqrt(p (1 - p) / N)``, the binomial standard error of the sample mean at
the sample size this file actually draws. It is written as an expression over
``SAMPLES`` so that changing the sample size cannot leave a stale literal behind.

THE MASK MUST VARY ACROSS BATCHES, NOT ONLY WITHIN ONE. That is the D-019
stateless-seed trap in a new place: a mask drawn once and reused would satisfy
every within-batch statistic while training on one fixed partition forever. The
across-batch arm reads the mask off the REAL ``tf.data`` pipeline, which is where
the per-batch seed advance lives.

ANTI-VACUITY FOR THE ACROSS-BATCH ARM. It first asserts the batches are not
degenerate (a batch of all-ones and a batch of all-ones "differ" nowhere, but
so do two identical draws), by requiring at least two DISTINCT batch patterns
among the batches examined.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np
import pytest

from train.bit_diffusion.synthetic_data import (
    DEFAULT_UNCONDITIONAL_PERCENT,
    build_bridge_dataset,
    prepare_training_batch,
    synthetic_records,
)
from train.bit_diffusion.train_bit_diffusion import (
    TrainingConfig,
    build_sde,
    create_datasets,
)

#: Records per drawn batch, and how many batches the rate arm draws.
BATCH = 32
BATCHES = 100

#: The rate arm's sample size. Written as the product so the tolerance below
#: cannot go stale when either factor changes.
SAMPLES = BATCH * BATCHES

SEED = 11


def _config(**overrides: Any) -> TrainingConfig:
    """A ``tiny``-geometry config; only the pipeline is exercised here."""
    base: Dict[str, Any] = dict(
        bridge_preset="tiny",
        variant="tiny",
        num_train_samples=BATCH,
        num_val_samples=BATCH,
        batch_size=BATCH,
        epochs=1,
        steps_per_epoch=2,
        validation_steps=2,
        warmup_epochs=0,
        seed=SEED,
    )
    base.update(overrides)
    return TrainingConfig(**base)


def _masks(percent: float, batches: int = BATCHES) -> List[np.ndarray]:
    """``batches`` masks from the REAL ``prepare_training_batch``, one per seed.

    Interface contract: the seeds advance exactly the way
    :func:`build_bridge_dataset`'s generator advances them (a distinct integer
    per batch), so this is the pipeline's own draw and not a re-implementation
    of it.
    """
    config = _config()
    bridge = config.bridge_config
    sde = build_sde(config)
    records = synthetic_records(BATCH, bridge, seed=SEED)
    rng = np.random.default_rng(SEED)
    out = []
    for _ in range(batches):
        inputs, _, _ = prepare_training_batch(
            records,
            bridge,
            sde,
            direction_mode=config.direction,
            time_sampler=config.time_sampler,
            unconditional_percent=percent,
            seed=int(rng.integers(1, 2**31 - 1)),
        )
        out.append(np.asarray(inputs["cond_mask"]))
    return out


# ---------------------------------------------------------------------
# 1. The endpoints are exact
# ---------------------------------------------------------------------


def test_percent_zero_keeps_every_sample_conditional():
    """``p = 0.0`` must reproduce the pre-9.1 all-ones behaviour EXACTLY."""
    masks = np.concatenate(_masks(0.0))
    assert masks.shape == (SAMPLES,)
    assert np.all(masks == 1.0), (
        f"{int((masks != 1.0).sum())} of {SAMPLES} samples were dropped at "
        "unconditional_percent=0.0; the comparison must be `>= p` against a "
        "[0, 1) draw, so p=0 keeps everything"
    )


def test_percent_one_drops_every_sample():
    """``p = 1.0`` must zero EVERY mask, not merely almost every one."""
    masks = np.concatenate(_masks(1.0))
    assert np.all(masks == 0.0), (
        f"{int((masks != 0.0).sum())} of {SAMPLES} samples stayed conditional "
        "at unconditional_percent=1.0"
    )


def test_the_mask_is_binary_and_float32():
    """The value the model reads must be exactly the value CFG constructs.

    ``forward_with_cfg`` builds an all-ZEROS float32 mask; a pipeline emitting
    booleans, ``-1``/``1`` or a soft probability would train a different thing
    while every shape assertion in the suite stayed green.
    """
    inputs = _masks(DEFAULT_UNCONDITIONAL_PERCENT, batches=1)[0]
    assert inputs.dtype == np.float32, f"cond_mask dtype is {inputs.dtype}"
    assert set(np.unique(inputs).tolist()) <= {0.0, 1.0}, (
        f"cond_mask carries values other than 0/1: {np.unique(inputs)}"
    )


# ---------------------------------------------------------------------
# 2. The rate, against a derived tolerance
# ---------------------------------------------------------------------


def test_the_empirical_drop_rate_matches_the_requested_percent():
    """Within four binomial standard errors of ``p`` at this sample size.

    ``se = sqrt(p (1 - p) / N)``. At ``p = 0.3`` and ``N = 3200`` that is
    ``8.10e-03``, so the bound is ``3.24e-02``. Four sigma, not three: the
    draw is seeded and therefore deterministic, and a bound that a correct
    implementation clears only 99.7% of the time is a flake waiting for
    someone to change a seed.
    """
    percent = DEFAULT_UNCONDITIONAL_PERCENT
    masks = np.concatenate(_masks(percent))
    observed = float((masks == 0.0).mean())
    se = math.sqrt(percent * (1.0 - percent) / SAMPLES)
    tolerance = 4.0 * se
    assert abs(observed - percent) <= tolerance, (
        f"observed unconditional rate {observed:.6f} over {SAMPLES} samples is "
        f"{abs(observed - percent) / se:.2f} standard errors from the requested "
        f"{percent} (se={se:.6f}, bound={tolerance:.6f})"
    )
    # ANTI-VACUITY: the bound must be tight enough to reject the two failure
    # modes it exists to reject -- an all-ones mask (rate 0) and an all-zeros
    # one (rate 1). Both are many hundreds of standard errors away.
    assert tolerance < min(percent, 1.0 - percent) / 2.0, (
        f"the tolerance {tolerance} is wide enough to accept a degenerate mask"
    )


def test_the_rate_tracks_a_second_requested_percent():
    """One value could be a coincidence; the knob must be the thing that moves.

    Deliberately NOT the default, and deliberately a value whose distance from
    the default is far larger than either arm's tolerance.
    """
    percent = 0.75
    masks = np.concatenate(_masks(percent))
    observed = float((masks == 0.0).mean())
    se = math.sqrt(percent * (1.0 - percent) / SAMPLES)
    assert abs(observed - percent) <= 4.0 * se, (
        f"observed {observed:.6f} at unconditional_percent={percent}, "
        f"{abs(observed - percent) / se:.2f} standard errors out"
    )


# ---------------------------------------------------------------------
# 3. The mask is redrawn per batch (the D-019 trap)
# ---------------------------------------------------------------------


def test_the_mask_varies_across_batches_of_the_real_dataset():
    """A mask drawn ONCE and reused passes every within-batch statistic.

    This is the D-019 stateless-seed trap in a new place: ``keras.random.*`` is
    stateless given an int, and a mask hoisted out of the per-batch draw would
    train on one fixed conditional/unconditional partition forever while the
    rate arm above stayed perfectly green.
    """
    config = _config(unconditional_percent=DEFAULT_UNCONDITIONAL_PERCENT)
    bridge = config.bridge_config
    records = synthetic_records(BATCH, bridge, seed=SEED)
    steps = 12
    dataset = build_bridge_dataset(
        records,
        bridge,
        build_sde(config),
        batch_size=BATCH,
        direction_mode=config.direction,
        time_sampler=config.time_sampler,
        unconditional_percent=config.unconditional_percent,
        seed=SEED,
        shuffle=False,
        steps=steps,
    )
    patterns = [
        tuple(np.asarray(inputs["cond_mask"]).tolist())
        for inputs, _, _ in dataset.as_numpy_iterator()
    ]
    assert len(patterns) == steps
    # ANTI-VACUITY: a batch that is entirely 1s or entirely 0s would differ from
    # another such batch nowhere, so require real within-batch variation first.
    mixed = [p for p in patterns if 0.0 in p and 1.0 in p]
    assert len(mixed) >= steps - 1, (
        f"only {len(mixed)} of {steps} batches contain both 0 and 1; the "
        "across-batch comparison below would be comparing degenerate masks"
    )
    assert len(set(patterns)) > 1, (
        f"all {steps} batches carried the IDENTICAL cond_mask "
        f"{patterns[0]}; the mask is drawn once and reused, not per batch"
    )
    # Stronger than "not all identical": with 32 samples at p=0.3 the chance of
    # any two batches coinciding is ~2^-32-ish, so near-total distinctness is
    # the honest expectation and a partial repeat means partial reuse.
    assert len(set(patterns)) == steps, (
        f"{steps - len(set(patterns))} batch masks repeat exactly; a fresh "
        "Bernoulli draw per batch would essentially never collide"
    )


def test_the_trainers_own_datasets_carry_the_configured_percent():
    """``create_datasets`` must thread the knob, not silently take the default.

    Guards the wiring seam specifically: everything above calls the pipeline
    directly, so a ``create_datasets`` that dropped the argument would leave
    every other arm in this file green.
    """
    config = _config(unconditional_percent=1.0, num_train_samples=BATCH,
                     num_val_samples=BATCH)
    train, val = create_datasets(config, build_sde(config))
    for name, dataset in (("train", train.take(3)), ("val", val)):
        for inputs, _, _ in dataset.as_numpy_iterator():
            mask = np.asarray(inputs["cond_mask"])
            assert np.all(mask == 0.0), (
                f"the {name} dataset emitted a conditional sample at "
                f"unconditional_percent=1.0: {mask}"
            )


# ---------------------------------------------------------------------
# 4. The knob is range-checked
# ---------------------------------------------------------------------


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan")])
def test_an_out_of_range_percent_raises(bad):
    """A probability outside ``[0, 1]`` must fail loudly, in the caller's frame."""
    config = _config()
    with pytest.raises(ValueError, match="unconditional_percent"):
        prepare_training_batch(
            synthetic_records(BATCH, config.bridge_config, seed=SEED),
            config.bridge_config,
            build_sde(config),
            unconditional_percent=bad,
            seed=SEED,
        )


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_the_config_rejects_an_out_of_range_percent(bad):
    """The same check at the config boundary, so ``--help``-driven runs fail fast."""
    with pytest.raises(ValueError, match="unconditional_percent"):
        _config(unconditional_percent=bad)


def test_the_default_is_upstreams_value():
    """Pins the number, and pins that the config takes it from ONE place.

    ``TrainingConfig.unconditional_percent`` defaults to the module constant
    rather than to a second literal; if someone re-types the value, the two can
    drift and this arm goes red.
    """
    assert DEFAULT_UNCONDITIONAL_PERCENT == 0.3, (
        "upstream passes --unconditional-percent 0.3 on every launch script "
        "staged in reference/ (FULL_INGEST.py:1572, :1806)"
    )
    assert TrainingConfig().unconditional_percent == DEFAULT_UNCONDITIONAL_PERCENT
