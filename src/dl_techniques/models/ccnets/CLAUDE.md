# CCNets — Causal Cooperative Networks

A model-agnostic framework for cooperative causal training. Three networks learn the
data-generating process by continuously verifying each other:

| Network | Models | Input → Output |
|---------|--------|----------------|
| **Explainer** | `P(E\|X)` | observation `X` → `(mu, log_var)` of the latent cause `E` |
| **Reasoner** | `P(Y\|X,E)` | `(X, E)` → explicit cause `Y` (class probabilities) |
| **Producer** | `P(X\|Y,E)` | `(Y, E)` → reconstructed observation `X` |

`X` = the effect (data). `Y` = the explicit cause (label). `E` = the latent cause (style /
context). The payoff is **counterfactual generation**: hold `E` fixed, swap `Y`.

Files: `orchestrators.py` (forward pass + train step), `trainer.py` (epoch loop, KL
annealing), `losses.py`, `control.py` (Reasoner throttling), `base.py` (config + dataclasses),
`utils.py` (`wrap_keras_model`, early stopping). Reference task: `src/train/ccnets/mnist.py`.
Defect history and rationale: `FIXES.md`. `FOUNDATION.md`/`README.md` are conceptual and
partially stale — trust the code and this file.

---

## The contract: what your three networks MUST satisfy

Any Keras model works, wrapped with `wrap_keras_model`, **if** it honors these signatures:

```python
explainer(x, training=...) -> (mu, log_var)      # two tensors, each [B, explanation_dim]
reasoner(x, e, training=...) -> y_probs           # [B, num_classes], a probability dist (softmax)
producer(y, e, training=...) -> x_hat             # reconstructed observation, same shape as x
```

### Invariant 1 — the Producer's label input MUST be differentiable

This is the single most important rule. The Producer receives `y` as a **probability
vector** `[B, num_classes]`, not an integer index. Consume it with a differentiable op:

```python
# CORRECT — differentiable; for one-hot y this equals an embedding lookup
self.label_projection = keras.layers.Dense(units, use_bias=False)
c = self.label_projection(y)

# WRONG — kills the cooperative mechanism
y_idx = keras.ops.argmax(y)            # non-differentiable
c = keras.layers.Embedding(...)(y_idx) # gradient to the Reasoner is severed
```

If the Producer is not differentiable in `y`, `reconstruction_loss` cannot train the
Reasoner and CCNets degenerates into three disconnected networks. `FIXES.md` (defect H3)
is the full post-mortem. The test `test_reconstruction_gradient_reaches_reasoner` pins it.

### Invariant 2 — the Explainer is variational

It returns `(mu, log_var)`, not a single vector. The orchestrator samples
`E = mu + eps * exp(0.5*log_var)` and applies a KL term. A non-variational encoder will
not work without editing `forward_pass`/`compute_model_errors`.

### Invariant 3 — build before wrapping

The orchestrator does not build your models. Run one dummy forward pass each (note the
label dummy is a **one-hot vector**, not an index):

```python
explainer(keras.ops.zeros((1, *x_shape)))
reasoner(keras.ops.zeros((1, *x_shape)), keras.ops.zeros((1, explanation_dim)))
producer(keras.ops.zeros((1, num_classes)), keras.ops.zeros((1, explanation_dim)))
```

---

## Wiring it up

```python
from dl_techniques.models.ccnets import (
    CCNetConfig, CCNetOrchestrator, CCNetTrainer, wrap_keras_model)

config = CCNetConfig(
    explanation_dim=32,                 # latent size; weak lever on quality, 16-64 is fine
    loss_fn='l2',                       # 'l1' | 'l2' | 'huber' | 'polynomial'
    learning_rates={'explainer': 3e-4, 'reasoner': 3e-4, 'producer': 3e-4},
    gradient_clip_norm=1.0,
    explainer_weights={'inference': 1.0, 'generation': 1.0, 'kl_divergence': 1e-3},
    reasoner_weights={'inference': 1.0, 'reconstruction': 1.0},
    producer_weights={'generation': 1.0, 'reconstruction': 1.0},
)

orchestrator = CCNetOrchestrator(
    explainer=wrap_keras_model(explainer),
    reasoner=wrap_keras_model(reasoner),
    producer=wrap_keras_model(producer),
    config=config,
)
CCNetTrainer(orchestrator, kl_annealing_epochs=10).train(train_ds, epochs=50,
                                                         validation_dataset=val_ds)
```

`train_ds` yields `(x, y)` with `y` one-hot. Error signals (per `compute_model_errors`):

```
explainer_error = w_inf·inference_loss + w_gen·generation_loss + kl_weight·KL
reasoner_error  = w_inf·crossentropy(y_truth, y_inferred) + w_rec·reconstruction_loss
producer_error  = w_gen·generation_loss + w_rec·reconstruction_loss
```

`reasoner_error` keeps a supervised crossentropy **anchor** plus the cooperative
reconstruction term — the anchor keeps training stable while the Producer is still weak.

---

## Adapting to a new task

| Task | `X` | `Y` | `E` | Notes |
|------|-----|-----|-----|-------|
| Image classification (MNIST) | image | class one-hot | style | reference impl |
| Tabular classification | feature row | class one-hot | latent context | swap conv blocks for dense |
| Sequence labeling / text | token sequence | per-step label | latent style | use `SequentialCCNetOrchestrator` (`sequential_data=True`); Producer does reverse-causal generation |
| Audio / spectrogram | spectrogram | label | speaker / channel | treat like images |

Steps for a new task:

1. Build three networks honoring the contract above. Copy `MNISTExplainer/Reasoner/Producer`
   from `train/ccnets/mnist.py` and swap the feature extractor for your modality.
2. Keep the Producer's label path differentiable (Invariant 1).
3. Pick `loss_fn` for the `X` reconstruction (`l2` for continuous, `l1`/`huber` for sharper
   edges or outliers).
4. Set `explanation_dim` to the latent capacity you want (32 is a good default).
5. Wrap, configure, train. Watch that `reasoner_error`, `producer_error`, `explainer_error`
   all decrease and that `batch_accuracy` rises.
6. Add tests mirroring `tests/test_models/test_ccnets/test_orchestrator.py` — especially the
   gradient-flow regression test for your Producer.

### If `Y` is not categorical (regression target)

`compute_model_errors` hardcodes `categorical_crossentropy` for the Reasoner anchor. For a
continuous `Y`, replace that term with an appropriate regression loss (and the
`batch_accuracy` metric in `train_step`). This is a deliberate edit to `orchestrators.py`,
not a config switch.

---

## Gotchas

- **Config loss weights are baked at `@tf.function` trace time.** Mutating
  `config.*_weights` mid-training has no effect. Only `orchestrator.kl_weight` (a
  `tf.Variable`) is live — that is why KL annealing works. To anneal any other weight,
  make it a `tf.Variable` too (see the `kl_weight` pattern in `orchestrators.py`).
- **`e_latent` is `stop_gradient`-ed into the Reasoner on purpose.** It prevents the
  Reasoner from collapsing `E` into a label shortcut. Do not remove it.
- **`dynamic_weighting` is dead** (same trace-freeze issue) and deprecated. Leave it `False`.
- **The Reasoner converges much faster than the other two.** `control.py` throttles it
  (`StaticThresholdStrategy`, default threshold 0.98; or `AdaptiveDivergenceStrategy`).
  Pass a strategy via `CCNetOrchestrator(..., control_strategy=...)` if the default stalls
  the Explainer/Producer.
- **Generation quality needs epochs.** 30-50 epochs for crisp reconstructions; a stronger
  Producer (residual upsampling, SSIM/perceptual term) helps more than a larger `E`.
