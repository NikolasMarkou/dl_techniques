"""``MSEh`` sends exactly zero gradient to every horizon column but ``h``.

Why this is a guard and not a footnote
--------------------------------------
In ADAM these losses constrain a RECURSIVE model, so the error at step ``h``
still moves every shared parameter -- that is the whole mechanism behind the
documented shrinkage of the smoothing parameters. Nearly every forecaster in
this repository is DIRECT instead: ``nbeats``, ``tirex``, ``prism``,
``xlstm/forecaster`` and ``mdn`` all emit the whole ``[B, H, ...]`` block from
per-step heads in one pass, with no recursion and no compounding.

Point ``MultistepLoss("mseh")`` at such a model and every head other than ``h``
receives no gradient at all and ships untrained. There is no shape symptom, no
NaN and no warning: the loss goes down, the model is broken. So the zero is
pinned here, exactly, together with an anti-vacuity arm proving the probe can
see a NON-zero when one exists.

Measured 2026-08-31 -- per-column sums of ``|d loss / d y_pred|``, ``H = 6``:

===================  ===================================================
loss                 columns 0..5
===================  ===================================================
``mseh``, ``h=3``    ``[0, 0, 1.673733, 0, 0, 0]``
``tmse``, ``h=3``    ``[1.536202, 1.469207, 1.673733, 0, 0, 0]``
===================  ===================================================

Note the third column agrees to the digit across the two rows: ``tmse`` is a
superset of ``mseh``, so the probe is measuring the same quantity in both arms.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.losses.multistep_loss import MultistepLoss

# ---------------------------------------------------------------------

BATCH, HORIZON, H_STEP = 32, 6, 3


@pytest.fixture()
def targets():
    return tf.constant(
        np.random.default_rng(2).normal(size=(BATCH, HORIZON)).astype("float32")
    )


def _column_gradient_mass(loss, y_true):
    """Return ``|d loss / d y_pred|`` summed over the batch, per horizon column.

    The gradient is taken w.r.t. the PREDICTION TENSOR, not w.r.t. a model's
    weights. A weight-space probe would confound "this column got no gradient"
    with "this column's head happens to share parameters", which is precisely
    the distinction the test is about.
    """
    y_pred = tf.Variable(np.zeros((BATCH, HORIZON), dtype="float32"))
    with tf.GradientTape() as tape:
        value = loss(y_true, y_pred)
    grad = tape.gradient(value, y_pred).numpy()
    return np.abs(grad).sum(axis=0)


# ---------------------------------------------------------------------
# The claim
# ---------------------------------------------------------------------

def test_mseh_gives_zero_gradient_to_every_other_horizon(targets):
    mass = _column_gradient_mass(MultistepLoss("mseh", h=H_STEP), targets)

    live = [i for i in range(HORIZON) if i == H_STEP - 1]
    dead = [i for i in range(HORIZON) if i != H_STEP - 1]

    # Exactly zero -- rtol=0, no tolerance to hide behind.
    np.testing.assert_array_equal(mass[dead], np.zeros(len(dead), dtype=mass.dtype))
    assert mass[live[0]] > 0.0


def test_mseh_without_h_starves_everything_but_the_last_step(targets):
    """``h=None`` means the LAST step for ``mseh``, not the whole horizon."""
    mass = _column_gradient_mass(MultistepLoss("mseh"), targets)
    np.testing.assert_array_equal(mass[:-1], np.zeros(HORIZON - 1, dtype=mass.dtype))
    assert mass[-1] > 0.0


# ---------------------------------------------------------------------
# Anti-vacuity -- the probe can see a non-zero
# ---------------------------------------------------------------------

def test_tmse_feeds_every_step_up_to_h(targets):
    mass = _column_gradient_mass(MultistepLoss("tmse", h=H_STEP), targets)

    assert np.all(mass[:H_STEP] > 0.0), mass
    # ... and truncation past h is real, not an accident of the aggregation.
    np.testing.assert_array_equal(
        mass[H_STEP:], np.zeros(HORIZON - H_STEP, dtype=mass.dtype)
    )


def test_the_two_losses_agree_on_the_column_they_share(targets):
    """Both arms measure the same quantity at column ``h``.

    If they disagreed, one of the two readings above would be measuring
    something other than "the gradient mass reaching step h".
    """
    mseh = _column_gradient_mass(MultistepLoss("mseh", h=H_STEP), targets)
    tmse = _column_gradient_mass(MultistepLoss("tmse", h=H_STEP), targets)
    np.testing.assert_allclose(
        mseh[H_STEP - 1], tmse[H_STEP - 1], rtol=0, atol=1e-6
    )


def test_msce_feeds_every_step_up_to_h(targets):
    """MSCE sums the signed errors first, so every step in 1..h stays live."""
    mass = _column_gradient_mass(MultistepLoss("msce", h=H_STEP), targets)
    assert np.all(mass[:H_STEP] > 0.0), mass
    np.testing.assert_array_equal(
        mass[H_STEP:], np.zeros(HORIZON - H_STEP, dtype=mass.dtype)
    )
