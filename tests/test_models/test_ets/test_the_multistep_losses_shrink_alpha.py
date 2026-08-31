"""The mechanism behind the multistep losses, reproduced on the one model that has it.

What is being reproduced
------------------------
Svetunkov, Kourentzes & Killick (2023) show that minimising h-steps-ahead
errors SHRINKS a model's smoothing parameters toward zero. The mechanism they
identify is that the multistep error variance is the one-step variance times a
function of the smoothing parameters, so an objective built on h-step errors
pays a penalty for a large ``alpha`` that a one-step objective does not.

For the additive local-level model ETS(A,N,N) that function is closed form::

    Var(e_{t+h|t}) = sigma^2 * (1 + (h - 1) * alpha^2)

That identity is the load-bearing arm of this file, because it is DETERMINISTIC:
it holds per dataset, not merely in expectation over datasets, so it can be
asserted without a Monte-Carlo budget.

Why there is NO estimator-level arm here
----------------------------------------
The obvious test -- fit ``alpha`` under a one-step loss and under a multistep
loss, assert the second is smaller -- was written, measured, and DELETED. The
shrinkage of the fitted ``alpha`` is a shift in the DISTRIBUTION of the
estimator, not a per-dataset fact, and it is not resolvable at any budget a unit
test can pay:

*   at 20 seeds and a 49-point grid it reproduces IN THE MEAN at every sample
    size measured -- ``MSE1`` vs ``TMSE`` is 0.4850/0.3750 at 24 origins,
    0.5690/0.4660 at 96, 0.6050/0.5770 at 512 -- with ``GTMSE`` shrinking LEAST
    every time, exactly the ordering the monograph gives, and the gap closing
    monotonically as the sample grows, exactly as the theory predicts. But the
    per-seed sign is close to a coin flip: ``frac_below_MSE1`` runs 0.45-0.65;
*   at 12 seeds and a 25-point grid -- 86 seconds, already too slow for this
    suite -- it does not reproduce at all: ``MSEh`` measured 0.5688 against
    ``MSE1`` 0.5031, i.e. the WRONG SIGN;
*   by ``n = 1024`` it is essentially gone -- 0.5950 / 0.5850 / 0.5800 / 0.5700
    -- with two of five seeds putting ``MSEh`` ABOVE the one-step argmin.

A test asserting the per-seed inequality would fail on the theory's own terms.
The estimator-level numbers therefore live in the package README as a recorded
measurement, and what this file guards is the MECHANISM underneath them, which
is deterministic.

Arm 1 is the control. If the one-step objective cannot recover the alpha its own
data was generated from, nothing downstream of it measures shrinkage.
"""

import keras
import numpy as np
import pytest

from dl_techniques.losses.multistep_loss import MultistepLoss
from dl_techniques.models.time_series.ets.model import ETSModel

# ---------------------------------------------------------------------

CONTEXT, HORIZON = 60, 12
ALPHA_TRUE = 0.6


def _simulate(alpha_true, origins, seed):
    """Draw from ETS(A,N,N): ``l_t = l_{t-1} + alpha e_t``, ``y_t = l_{t-1} + e_t``.

    Returns sliding windows, so the batch axis IS the sample of forecast origins
    -- exactly the layout ``ETSModel`` and ``MultistepLoss`` expect.
    """
    rng = np.random.default_rng(seed)
    length = origins + CONTEXT + HORIZON
    errors = rng.normal(size=length)
    series = np.empty(length)
    level = 0.0
    for t in range(length):
        series[t] = level + errors[t]
        level += alpha_true * errors[t]

    context = np.stack([series[i : i + CONTEXT] for i in range(origins)])
    future = np.stack(
        [series[i + CONTEXT : i + CONTEXT + HORIZON] for i in range(origins)]
    )
    return context.astype("float32"), future.astype("float32")


def _forecast_at(model, alpha, context):
    """Set ``alpha`` on an ALREADY-BUILT model and forecast.

    Rebuilding a fresh ``ETSModel`` per grid point is what a first draft did; it
    accumulates Keras state and turns a one-minute sweep into an hour. One model
    plus ``assign`` is both faster and a tighter comparison -- every grid point
    shares the identical graph.
    """
    model.alpha_raw.assign(np.float32(np.log(alpha / (1.0 - alpha))))
    return np.asarray(keras.ops.convert_to_numpy(model(context)))[:, :, 0]


# ---------------------------------------------------------------------
# Arm 1 -- the control MUST reproduce itself first
# ---------------------------------------------------------------------

def test_the_one_step_objective_recovers_the_generating_alpha():
    """Without this, the shrinkage arm below measures nothing at all.

    Measured 2026-08-31 at 1024 origins over 5 seeds: argmin at
    [0.575, 0.600, 0.600, 0.600, 0.600] against ``alpha_true = 0.600``, on a
    grid of step 0.025.
    """
    context, future = _simulate(ALPHA_TRUE, origins=1024, seed=3)
    model = ETSModel(variant="ANN", horizon=HORIZON)
    model.build(context.shape)

    grid = np.linspace(0.30, 0.90, 25)
    one_step = MultistepLoss("mseh", h=1)
    values = [
        float(one_step(future[:, :, None], _forecast_at(model, a, context)[:, :, None]))
        for a in grid
    ]

    recovered = float(grid[int(np.argmin(values))])
    assert abs(recovered - ALPHA_TRUE) < 0.06, recovered


# ---------------------------------------------------------------------
# Arm 2 -- the MECHANISM, deterministic and closed form
# ---------------------------------------------------------------------

@pytest.mark.parametrize("alpha", [0.2, 0.5, 0.8])
def test_the_multistep_variance_follows_the_closed_form(alpha):
    """``Var(e_{t+h|t}) / Var(e_{t+1|t}) == 1 + (h - 1) * alpha^2``.

    This is the whole shrinkage mechanism in one line: the h-step error variance
    is the one-step variance AMPLIFIED by a factor that grows with ``alpha``, so
    a multistep objective is paying for reactivity that a one-step objective gets
    for free.

    Measured 2026-08-31 at 4000 origins, max relative deviation from the closed
    form: 0.035 at alpha=0.2, 0.059 at 0.5, 0.064 at 0.8. The residual is finite
    sample plus the data-derived initial state, and it does not shrink to zero.
    """
    context, future = _simulate(alpha, origins=4000, seed=7)
    model = ETSModel(variant="ANN", horizon=HORIZON)
    model.build(context.shape)

    squared_error = (future - _forecast_at(model, alpha, context)) ** 2
    measured = squared_error.mean(axis=0)
    measured = measured / measured[0]

    theory = 1.0 + np.arange(HORIZON) * alpha ** 2

    np.testing.assert_allclose(measured, theory, rtol=0.10, atol=0)


def test_the_amplification_is_monotone_in_alpha():
    """Anti-vacuity: the closed form above discriminates between alphas.

    A test that only checked one alpha against its own theory curve could pass
    against an implementation that ignored alpha entirely on the horizon axis.
    """
    ratios = []
    for alpha in (0.2, 0.5, 0.8):
        context, future = _simulate(alpha, origins=2000, seed=7)
        model = ETSModel(variant="ANN", horizon=HORIZON)
        model.build(context.shape)
        squared_error = (future - _forecast_at(model, alpha, context)) ** 2
        per_step = squared_error.mean(axis=0)
        ratios.append(float(per_step[-1] / per_step[0]))

    assert ratios[0] < ratios[1] < ratios[2], ratios
    assert ratios[0] < 2.0 and ratios[2] > 5.0, ratios
