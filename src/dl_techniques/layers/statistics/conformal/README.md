# Conformal layers (`dl_techniques.layers.statistics.conformal`)

Split conformal prediction gives a model's output a coverage guarantee without changing the
model. You hold out a calibration set, score how wrong the model was on it, and take a
finite-sample quantile `q_hat` of those scores. Applying `q_hat` to the output makes an interval
or a set that contains the truth with probability at least `1 - alpha`, for any model, as long as
the calibration data and the data you deploy on are exchangeable.

This package is the in-graph half of that recipe. Each layer holds one frozen scalar, `q_hat`,
and applies it in its forward pass. The layers differ only in what they apply it to:

| Class | Applies `q_hat` to | Output | Score to calibrate with | `q_hat_init` default |
|---|---|---|---|---|
| `ConformalInterval` | a point estimate `mu` | `(mu_c, lower, upper)`, or `(lower, upper)` with `return_mu=False`; `mu_c` is `mu` clipped to `[domain_min, domain_max]` | `abs(truth - mu)` | `0.0` |
| `ConformalQuantileHead` | three learned quantiles per step and feature | `call`: `(batch, forecast_length, output_dim, 3)` as lower, median, upper. `predict_intervals`: `(median, lower, upper)` widened by `q_hat` | `max(q_low - truth, truth - q_high)` | `0.0` |
| `ConformalPredictionSet` | classifier logits | `(probs, in_set)`, `in_set` a 0/1 mask `(batch, num_classes)` of the labels whose cumulative probability is at most `q_hat` | `utils.conformal_classification.aps_scores(..., randomized=False)` | `2.0` (every label) |

All three extend `ConformalLayer` (`base.py`), which owns the weight and the two ways to fill it:

- `calibrate(scores, alpha=0.1)` takes the `ceil((n + 1) * (1 - alpha))`-th smallest score, the order
  statistic that gives the guarantee. With too few scores for that `alpha` the value is `+inf`.
- `set_q_hat(value)` assigns a value you fitted elsewhere, for example with the numpy utilities in
  `dl_techniques/utils/conformal_denoiser_intervals.py` or `conformal_classification.py`.

`get_config()` stores the current calibrated value as `q_hat_init`, so a config-only reload keeps it, and a
`.keras` file keeps it in the weight `conformal_q_hat`.

## Layout

```
conformal/
├── __init__.py          curated exports: ConformalLayer, ConformalInterval, ConformalQuantileHead,
│                        ConformalPredictionSet, attach_conformal_sets
├── base.py              ConformalLayer: the frozen weight, set_q_hat, calibrate, get_config
├── interval.py          ConformalInterval
├── quantile_head.py     ConformalQuantileHead
└── prediction_set.py    ConformalPredictionSet, attach_conformal_sets
```

## The workflow

1. Train the model as usual. The conformal layer plays no part in training.
2. Run the model on a held-out calibration set and compute one score per sample (or element).
3. Call `calibrate(scores, alpha)`, or `set_q_hat(value)` for a value fitted elsewhere.
4. Save. The calibrated threshold is a weight, so the model is one `.keras` file.
5. Deploy. The forward pass applies the fixed `q_hat`; nothing is learned.

`attach_conformal_sets(model, q_hat, input_shape)` does step 4 for a classifier: it builds a
Functional model from the input to `(logits, probs, in_set)`, sharing the classifier's weights.

## Quick start

A point estimate with an interval. Here the prediction is a stand-in; use your model's output on a
held-out set:

```python
import numpy as np
import keras
from dl_techniques.layers.statistics.conformal import ConformalInterval

rng = np.random.default_rng(0)
truth = rng.uniform(size=(2000, 8)).astype("float32")
mu = np.clip(truth + rng.normal(0.0, 0.05, truth.shape), 0.0, 1.0).astype("float32")

layer = ConformalInterval()
layer(mu[:1])                                           # build the layer
layer.calibrate(np.abs(truth - mu).ravel(), alpha=0.1)  # score = |truth - prediction|
mu_c, lower, upper = layer(mu)
print(float(keras.ops.convert_to_numpy(layer.q_hat)))
```

Learned quantiles (conformalized quantile regression). The head's own quantiles carry no guarantee;
`q_hat` widens both bounds by the amount the held-out set needs:

```python
import numpy as np
import keras
from dl_techniques.layers.statistics.conformal import ConformalQuantileHead

rng = np.random.default_rng(0)
head = ConformalQuantileHead(forecast_length=4, output_dim=1)
features = rng.normal(size=(600, 16)).astype("float32")
truth = rng.normal(size=(600, 4, 1)).astype("float32")  # held-out targets
head(features[:1])                                       # build the layer

quantiles = keras.ops.convert_to_numpy(head(features))   # (600, 4, 1, 3): lower, median, upper
scores = np.maximum(quantiles[..., 0] - truth, truth - quantiles[..., 2]).ravel()
head.calibrate(scores, alpha=0.1)
median, lower, upper = head.predict_intervals(features[:5])
```

A classifier's prediction set:

```python
import numpy as np
import keras
from dl_techniques.layers.statistics.conformal import ConformalPredictionSet, attach_conformal_sets
from dl_techniques.utils.conformal_classification import aps_scores

rng = np.random.default_rng(0)
classifier = keras.Sequential([keras.Input((16,)), keras.layers.Dense(10)])  # emits logits
x = rng.normal(size=(1000, 16)).astype("float32")
y = rng.integers(0, 10, size=1000)
logits = keras.ops.convert_to_numpy(classifier(x))

layer = ConformalPredictionSet()
layer(logits[:1])                                                           # build the layer
layer.calibrate(aps_scores(logits, y, randomized=False), alpha=0.1)
q_hat = float(keras.ops.convert_to_numpy(layer.q_hat))

sets_model = attach_conformal_sets(classifier, q_hat, input_shape=(16,))
logits_out, probs, in_set = sets_model(x[:5])                               # in_set: 0/1 mask (5, 10)
```

The classifier above is untrained, so its sets are large; the point is the calling pattern.

## What the guarantee is, and what it is not

- **Marginal coverage.** The probability is averaged over the data distribution. It does not hold for
  every input or every class; `utils.conformal_classification.evaluate_sets` reports the class-conditional
  minimum so the gap is visible.
- **Exchangeability.** Calibration data and deployment data must come from the same distribution.
  `ConformalInterval` carries one radius, so it covers one noise regime; a model deployed across several
  needs one calibrated radius per regime (`calibrate_per_sigma` in `utils/conformal_denoiser_intervals.py`
  does this for denoisers).
- **A fixed model.** The model must not have been chosen using the calibration set. A checkpoint picked on
  the same validation set that is then used to calibrate weakens the guarantee. The trainers record which
  weights were calibrated in `conformal.json` for this reason.
- **The prediction-set layer is conservative.** It uses the deterministic rule (`u = 1`, the cumulative mass
  through a label, inclusive), which is valid but gives slightly larger sets than the randomized rule in
  `utils/conformal_classification.py`. An empty set is possible when `q_hat` is below the top probability.
- **Unbounded results are honest.** When `ceil((n + 1) * (1 - alpha)) > n` no finite value can guarantee the
  coverage, so `q_hat` is `+inf`: an infinite interval, or (for the set layer, stored as `2.0`) every label.

No coverage or set-size numbers are quoted here: none have been measured on a trained model in this
repository. The synthetic checks live in the tests.

## In the trainers

`train_resnet`, `train_vit` (ImageNet) and `train_cliffordnet` (ImageNet) call `train.common.conformal.run_conformal`
after training. It runs one ordered pass over the validation set, reports coverage on random half splits,
fits the exported threshold on all of it, and writes `conformal.json` and `conformal_model.keras` (outputs
`(logits, probs, in_set)`) to the run directory. `--no-conformal` skips it and `--conformal-alpha` sets `alpha`.

## Renamed API

These layers were moved into this package and unified on 2026-10-27/28. Removed with no alias:
`ConformalIntervalLayer` (now `ConformalInterval`), `ConformalPredictionSetLayer` (now
`ConformalPredictionSet`), the modules `conformal_interval`, `conformal_prediction_set` and
`conformal_quantile_head` under `layers/statistics/`, the constructor arguments `q_init` and `tau_init`
(now `q_hat_init`), the attributes `q` and `tau` (now `q_hat`), the weights `conformal_q` and
`conformal_tau` (now `conformal_q_hat`), and `calibrate(value)` on the interval and set layers (now
`set_q_hat(value)`; `calibrate` takes scores). A `.keras` file that recorded an old class name or
registration key does not load; no such file is kept in the repository.

## Tests

`tests/test_layers/test_statistics/test_conformal/`: `test_base_contract.py` runs one contract over all three
layers (registration key, the frozen scalar in the variable dtype, `set_q_hat` and `calibrate`, config,
`.keras` round trip), and `test_interval.py`, `test_quantile_head.py` and `test_prediction_set.py` cover
what is specific to each. `AGENTS.md` in this directory lists the rules and what pins each.
