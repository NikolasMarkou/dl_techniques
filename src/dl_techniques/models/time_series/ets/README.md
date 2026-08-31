# ETS — pure additive exponential smoothing

![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg) ![Python 3.11](https://img.shields.io/badge/Python-3.11+-blue.svg) ![TF 2.18](https://img.shields.io/badge/TF-2.18-orange.svg)

A single-source-of-error additive ETS with **trainable smoothing parameters**, in
`ANN` (local level), `AAN` (local trend) and `AAA` (additive seasonal) form.

> **Identity, up front.** This is a *classical* forecaster with **at most three
> trainable scalars**. It will not beat `tirex`, `prism` or `nbeats` on a rich
> dataset and it is not here to. It is here because it is the only **recursive**
> model in this family, and therefore the only one on which the shrinkage result
> behind [`losses/multistep_loss.py`](../../../losses/multistep_loss.py) can be
> *reproduced* instead of cited.

---

## 1. Why this package exists

Every other forecaster under `models/time_series/` is **direct**: `nbeats`,
`tirex`, `prism`, `xlstm/forecaster` and `mdn` all emit the whole `[B, H, F]`
block from per-step heads in one pass. `tirex`'s own README states the design
goal plainly — "there is no autoregressive loop, so horizon cost is constant and
no error compounds across `H`". `deepar` is autoregressive at inference but
**teacher-forced** during training, and its rollout carries no gradient.

That left the repository with **no smoothing parameter anywhere**, and so no way
to exercise the claim that motivates the multistep losses:

> Minimising h-steps-ahead errors shrinks a model's smoothing parameters toward
> zero, making it less reactive to noise and more stable across longer horizons.
> — Svetunkov, Kourentzes & Killick (2023)

The result is proven for **pure additive** ETS and ARIMA, and it arises through
**recursive error accumulation**. Hence this model, and hence the deliberate
absence of multiplicative and mixed variants — the theory does not cover them,
and shipping them here would invite a claim the paper declines to make.

---

## 2. The state space

Hyndman et al.'s additive ETS with a single error term:

```
yhat_t = l_{t-1} + b_{t-1} + s_{t-m}          one step ahead, from t-1
e_t    = y_t - yhat_t
l_t    = l_{t-1} + b_{t-1} + alpha * e_t      level
b_t    = b_{t-1} + beta  * e_t                trend
s_t    = s_{t-m} + gamma * e_t                seasonal
```

| variant | level | trend | seasonal | trainable scalars |
|:--|:--|:--|:--|:--|
| `ANN` | yes | — | — | `alpha` |
| `AAN` | yes | yes | — | `alpha`, `beta` |
| `AAA` | yes | yes | yes | `alpha`, `beta`, `gamma` |

Because the model is *pure additive*, the h-step forecast from origin `t` is
**closed form** — no rollout loop, no sampling:

```
yhat_{t+h|t} = l_t + h * b_t + s_{t + h - m*ceil(h/m)}
```

---

## 3. Two design decisions worth knowing

**Forecast origins come from the dataset, not from the model.** ADAM computes
multistep errors from every in-sample origin. Here one training sample is one
context window plus its `H`-step future, so a **minibatch is a sample of forecast
origins** and `MultistepLoss` averages over it unchanged. The cost is a re-filter
per overlapping window — nothing, for three scalars. The gain is a single code
path: `call` and `_forecast` are the same function, so there is no train/serve
skew and no mode flag.

**Initial states are derived from the window, not fitted.** Level from the first
seasonal period, trend from its first differences, seasonal from the first
period's deviations (centred to sum to zero, the standard additive
identification constraint). This keeps the model batch-friendly and
scale-adaptive, and — more importantly — keeps the trainable surface *exactly*
the smoothing parameters. A model with fitted initial states would let the
optimiser trade shrinkage against initialisation and confound the measurement
the package exists to make. `tests/test_models/test_ets/` pins the trainable
variable list for that reason.

`alpha`, `beta` and `gamma` are held in `[0, 1]` by a sigmoid, so the constraint
is unbreakable by construction rather than merely initialised.

---

## 4. Quick start

```python
import keras
from dl_techniques.models.time_series import ETSModel
from dl_techniques.losses import MultistepLoss

model = ETSModel(variant="AAA", horizon=12, seasonal_period=12)

# Conventional estimation: minimise the ONE-step-ahead error.
model.compile(optimizer=keras.optimizers.Adam(0.05), loss=MultistepLoss("mseh", h=1))

# Multistep estimation: align the objective with a 12-step lead time.
model.compile(optimizer=keras.optimizers.Adam(0.05), loss=MultistepLoss("gtmse", h=12))

model.fit(context_windows, futures)   # [B, T] -> [B, H, 1]
print(model.alpha, model.beta, model.gamma)
```

Diagnostics come free from the filtering pass:

```python
fitted, residuals = model.fitted_values(context_windows)   # both [B, T]
forecast = model.predict_forecast(context_windows)         # Forecast, quantiles=None
```

---

## 5. Shapes and contract

| | |
|:--|:--|
| input | `[B, T]` or `[B, T, 1]` — **univariate only** |
| output | `[B, H, 1]` |
| `predict_forecast` | `Forecast(point=[B, H, 1], quantiles=None)` |

This is a **point** model: per the family contract in
[`../README.md`](../README.md) it must not fabricate intervals. Callers test
`forecast.has_quantiles()` rather than the concrete class.

`T` must be **static** (the initial-state derivation reads it) and at least
`seasonal_period + 1`.

---

## 6. What was actually measured

Two different claims, with very different evidence, and the difference matters.

### The mechanism: deterministic, and in the test suite

For ETS(A,N,N) the h-step error variance is closed form:

```
Var(e_{t+h|t}) = sigma^2 * (1 + (h - 1) * alpha^2)
```

This is the whole shrinkage argument in one line -- the multistep error variance
is the one-step variance *amplified* by a factor growing in `alpha`, so a
multistep objective pays for reactivity that a one-step objective gets free.
Measured on this model at 4000 forecast origins, maximum relative deviation from
the closed form:

| alpha | max relative deviation |
|:--|:--|
| 0.2 | 0.035 |
| 0.5 | 0.059 |
| 0.8 | 0.064 |

The residual is finite sample plus the data-derived initial state; it does not
go to zero. This is asserted per-dataset in
`tests/test_models/test_ets/test_the_multistep_losses_shrink_alpha.py`.

### The estimator: statistical, and NOT in the test suite

Whether the *fitted* `alpha` actually comes out smaller is a shift in the
distribution of the estimator, not a per-dataset fact. Measured by grid search
over `alpha` (49 points), `alpha_true = 0.6`, `h = 12`, context 60, 20 seeds:

| origins | MSE1 | GTMSE | TMSE | MSEh | MSCE | gap (MSE1 - TMSE) |
|:--|:--|:--|:--|:--|:--|:--|
| 24 | **0.4850** | 0.4010 | 0.3750 | 0.3930 | 0.3750 | 0.110 |
| 96 | **0.5690** | 0.4790 | 0.4660 | 0.4780 | 0.4660 | 0.103 |
| 512 | **0.6050** | 0.5860 | 0.5770 | 0.5510 | 0.5770 | 0.028 |
| 1024 (5 seeds only) | **0.5950** | 0.5850 | 0.5800 | 0.5700 | 0.5700 | 0.015 |

**The gap closes monotonically as the sample grows** -- 0.110, 0.103, 0.028,
0.015 -- which is the theory's own prediction: shrinkage is a function of `h`
relative to the number of origins.

The `n = 512` row is the one to quote. The one-step control lands on
`alpha_true` (0.6050 against 0.600), so the multistep estimators sitting below it
is shrinkage and not a shared small-sample bias -- at `n = 24` *every* estimator
including the control sits well under 0.6, so that row shows the effect at its
largest and its evidence at its weakest.

Three things in that table are worth stating plainly:

1. **Every multistep estimator shrinks below the one-step one at every sample
   size measured**, and `GTMSE` shrinks *least* in all four rows -- exactly the
   ordering the ADAM monograph gives.
2. **The per-seed sign is close to a coin flip.** `frac_below_MSE1` runs 0.45-0.65
   across all rows. The effect is in the mean, not in any single run.
3. **It nearly vanishes by 512-1024 origins**, which is what the theory predicts.
   That is also where the control is trustworthy, so the two ends of the table
   trade off against each other: the small-sample rows show the largest effect on
   the weakest control, the large-sample rows the reverse.

A per-seed assertion of shrinkage was written and deleted: at 12 seeds and a
25-point grid it came out with the **wrong sign** (`MSEh` 0.5688 vs `MSE1`
0.5031) while already costing 86 seconds. Do not add one back.

`MSCE` and `TMSE` agree in every cell above, and that is not a bug: for `ANN`
the h-step forecast is flat in `h`, so `sum_j e_j^2` and `(sum_j e_j)^2` are
minimised at the same point. The two separate on `AAN` / `AAA`.

## 7. Implementation note: `keras.ops.scan` on the TensorFlow backend

The recursion is one `keras.ops.scan` over the time axis. The TF backend
**requires the per-step output to have the same structure and shape as the
carry** — returning a differently-shaped `ys` fails with
`Incompatible shape for value ((3,)), expected ((3, 6))`. The state is therefore
packed flat as `[level, trend, s_{1-m} ... s_0]` of width `2 + m` and the step
function returns that same packed state as its output, which also hands
`fitted_values` the whole state history for free.

---

## 8. References

- Svetunkov, I., Kourentzes, N., & Killick, R. (2023). "Multi-step Estimators and Shrinkage Effect in Time Series Models". *Computational Statistics*. DOI: [10.1007/s00180-023-01377-x](https://doi.org/10.1007/s00180-023-01377-x)
- Svetunkov, I. (2023). *Forecasting and Analytics with the Augmented Dynamic Adaptive Model (ADAM)*. <https://openforecast.org/adam/>
- Hyndman, R.J., Koehler, A.B., Ord, J.K., & Snyder, R.D. (2008). *Forecasting with Exponential Smoothing: The State Space Approach*. Springer.
