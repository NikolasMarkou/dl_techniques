"""GTMSE's per-sample surrogate reproduces the exact batch-global estimator.

The claim
---------
``GTMSE = sum_j log(M_j)`` with ``M_j = mean_i e_ij^2`` averaged over forecast
origins -- i.e. over the BATCH. That has no per-sample decomposition, but this
package's hardest rule is that ``call()`` returns ``(batch,)``: a scalar return
does not ignore ``sample_weight``, it multiplies by ``mean(sample_weight)`` and
charges every row the batch aggregate.

``MultistepLoss._gtmse`` therefore returns the first-order expansion of ``log``
about the DETACHED batch mean::

    L_i = sum_j [ log(sg(M_j)) + e_ij^2 / sg(M_j) - 1 ]

and this file pins the two identities that make that legitimate:

*   VALUE    ``mean_i L_i == sum_j log(M_j)``
*   GRADIENT ``d/dtheta mean_i L_i == d/dtheta sum_j log(M_j)``

Anti-vacuity
------------
Each arm is paired with the plausible WRONG implementation -- the "obvious"
per-sample rewrite ``sum_j log(e_ij^2 + eps)`` -- which must FAIL the same
assertion. Without that pairing an assertion could pass because the tolerance is
loose rather than because the surrogate is right.

Measured 2026-08-31, float32, one ``Dense(6)`` over an ``(32, 8)`` input:

===========================  ==============  =========================
form                         value           max |grad - exact_grad|
===========================  ==============  =========================
surrogate (shipped)          3.5366707       5.96e-08
exact batch-global           3.5366707       0 (definitionally)
naive per-sample log         -4.5104222      9.133
===========================  ==============  =========================
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.losses.multistep_loss import MultistepLoss

# ---------------------------------------------------------------------

BATCH, HORIZON, WIDTH = 32, 6, 8
EPSILON = 1e-8

# float32 accumulation over a 32x6 reduction; not a hand-picked bound.
ATOL = 1e-5


def _exact_gtmse(y_true, y_pred):
    """The published estimator, computed directly as a batch-global scalar."""
    step_means = keras.ops.mean(keras.ops.square(y_pred - y_true), axis=0)
    return keras.ops.sum(keras.ops.log(keras.ops.maximum(step_means, EPSILON)))


def _naive_gtmse(y_true, y_pred):
    """The plausible WRONG rewrite: log of the per-sample error, not of the mean.

    Decomposes trivially, and is a different objective by exactly the Jensen gap.
    """
    per_sample = keras.ops.sum(
        keras.ops.log(keras.ops.square(y_pred - y_true) + EPSILON), axis=1
    )
    return keras.ops.mean(per_sample)


@pytest.fixture()
def fixture():
    """A tiny trainable graph plus fixed targets."""
    keras.utils.set_random_seed(0)
    x = tf.constant(np.random.default_rng(1).normal(size=(BATCH, WIDTH)).astype("float32"))
    y = tf.constant(np.random.default_rng(2).normal(size=(BATCH, HORIZON)).astype("float32"))
    dense = keras.layers.Dense(HORIZON)
    dense.build((None, WIDTH))
    return x, y, dense


def _value_and_grads(loss_fn, x, y, dense):
    with tf.GradientTape() as tape:
        value = loss_fn(y, dense(x))
    return float(value), tape.gradient(value, dense.trainable_variables)


# ---------------------------------------------------------------------
# Arm 1 -- the value identity
# ---------------------------------------------------------------------

def test_the_surrogate_reports_the_exact_gtmse_value(fixture):
    x, y, dense = fixture
    surrogate = float(MultistepLoss("gtmse", epsilon=EPSILON)(y, dense(x)))
    exact = float(_exact_gtmse(y, dense(x)))
    np.testing.assert_allclose(surrogate, exact, rtol=0, atol=ATOL)


def test_the_naive_rewrite_reports_a_different_value(fixture):
    """Anti-vacuity: the assertion above discriminates."""
    x, y, dense = fixture
    naive = float(_naive_gtmse(y, dense(x)))
    exact = float(_exact_gtmse(y, dense(x)))
    assert abs(naive - exact) > 1.0, (naive, exact)


# ---------------------------------------------------------------------
# Arm 2 -- the gradient identity
# ---------------------------------------------------------------------

def test_the_surrogate_reproduces_the_exact_gtmse_gradient(fixture):
    x, y, dense = fixture
    _, surrogate_grads = _value_and_grads(MultistepLoss("gtmse", epsilon=EPSILON), x, y, dense)
    _, exact_grads = _value_and_grads(_exact_gtmse, x, y, dense)

    assert len(surrogate_grads) == len(exact_grads) == 2
    for got, want in zip(surrogate_grads, exact_grads):
        assert got is not None
        np.testing.assert_allclose(got.numpy(), want.numpy(), rtol=0, atol=ATOL)


def test_the_naive_rewrite_gives_a_different_gradient(fixture):
    """Anti-vacuity: the gradient assertion discriminates too."""
    x, y, dense = fixture
    _, naive_grads = _value_and_grads(_naive_gtmse, x, y, dense)
    _, exact_grads = _value_and_grads(_exact_gtmse, x, y, dense)

    worst = max(
        float(np.max(np.abs(a.numpy() - b.numpy())))
        for a, b in zip(naive_grads, exact_grads)
    )
    assert worst > 1.0, worst


# ---------------------------------------------------------------------
# Arm 3 -- what stop_gradient on the batch statistic actually buys
# ---------------------------------------------------------------------

def test_the_detached_batch_mean_only_matters_under_sample_weight():
    """Pin the MEASURED scope of ``stop_gradient(M_j)``, not the assumed one.

    The obvious story -- "without ``stop_gradient`` the surrogate differentiates
    its own normaliser and the gradient identity breaks" -- is FALSE, and this
    test exists because it was asserted and then refuted. ``mean_i d_ij / M_j``
    is identically ``1``, so the attached form collapses to ``sum_j log(M_j)``
    as a FUNCTION: same value, and the same gradient to float32 noise.

    Measured 2026-08-31, detached vs attached, one ``Dense(6)``:

    ==========================  ==========================
    reduction                   max |gradient difference|
    ==========================  ==========================
    unweighted mean             1.192e-07  (float32 noise)
    ``sample_weight`` on 8/32    1.494e-01
    ==========================  ==========================

    So the detachment is not what makes the estimator correct -- it is what keeps
    ``sample_weight`` meaningful. Weighting breaks the ``mean_i ... == 1``
    cancellation, and only the detached form still reads as "GTMSE over the
    weighted rows, linearised at the full-batch ``M_j``".
    """
    keras.utils.set_random_seed(0)
    x = tf.constant(np.random.default_rng(1).normal(size=(BATCH, WIDTH)).astype("float32"))
    y = tf.constant(np.random.default_rng(2).normal(size=(BATCH, HORIZON)).astype("float32"))
    dense = keras.layers.Dense(HORIZON)
    dense.build((None, WIDTH))

    weights = np.ones((BATCH,), dtype="float32")
    weights[: BATCH // 4] = 0.0
    weights = tf.constant(weights)

    def per_sample(y_true, y_pred, detach):
        squared = keras.ops.square(y_pred - y_true)
        step_means = keras.ops.maximum(
            keras.ops.mean(squared, axis=0, keepdims=True), EPSILON
        )
        if detach:
            step_means = keras.ops.stop_gradient(step_means)
        return keras.ops.sum(
            keras.ops.log(step_means) + squared / step_means - 1.0, axis=1
        )

    def run(detach, weighted):
        with tf.GradientTape() as tape:
            values = per_sample(y, dense(x), detach)
            loss = (
                keras.ops.sum(values * weights) / BATCH
                if weighted
                else keras.ops.mean(values)
            )
        return float(loss), tape.gradient(loss, dense.trainable_variables)

    def worst_gap(weighted):
        _, detached = run(True, weighted)
        _, attached = run(False, weighted)
        return max(
            float(np.max(np.abs(a.numpy() - b.numpy())))
            for a, b in zip(detached, attached)
        )

    assert worst_gap(weighted=False) < ATOL, worst_gap(weighted=False)
    assert worst_gap(weighted=True) > 1e-2, worst_gap(weighted=True)
