# CCNets — What Was Broken and How It Was Fixed (2026-05)

This document records the defect audit and repair that took the CCNets implementation
from "does not work" to a training, counterfactual-capable model. It supersedes the
training-dynamics description in `FOUNDATION.md` where the two disagree.

## Summary

The CCNet *idea* was sound. The implementation had four defects — two fatal — that
together meant the framework either trained nothing or trained as three loosely-coupled
networks with the cooperative causal mechanism (its entire reason to exist) switched off.

After the fix: 50-epoch MNIST run reaches **0.994 train accuracy**, clean
reconstructions, and a working counterfactual-generation matrix (style `E` recombined
with arbitrary labels `Y`).

## Defect 1 — entry point never ran training (FATAL)

`src/train/ccnets/mnist.py` ended its `__main__` block with:

```python
experiment = CCNetExperiment(config)   # constructor only registers viz plugins
```

`.run()` was never called. Executing the script registered four visualization plugins
and exited. No data was loaded, no training happened.

**Fix:** added `experiment.run()`.

## Defect 2 — the cooperative causal mechanism was severed (FATAL to the idea)

This is the important one. CCNets' premise is *cooperative causal credit assignment*:
the Producer reconstructs `X` from the Reasoner's inferred label, and the resulting
reconstruction error trains the Reasoner. That requires gradient to flow
`reconstruction_loss → Producer → y_inferred → Reasoner`.

The Producer fed its label input through `keras.layers.Embedding`, which requires
**integer indices**. To satisfy it, `orchestrators.py` did:

```python
y_inferred_indices = keras.ops.argmax(y_inferred_probs, axis=-1)
x_reconstructed = self.producer(y_inferred_indices, ...)
```

`argmax` is **non-differentiable**. It cut every gradient path from the
reconstruction and inference losses back to the Reasoner. Verified empirically:
`tape.gradient(reconstruction_loss, reasoner.trainable_variables)` returned
**18/18 `None`**.

Because the Reasoner could not be trained cooperatively, a previous patch had replaced
`reasoner_error` with a plain `categorical_crossentropy`. The net effect: the Reasoner
was a standalone CNN classifier bolted onto the side of a label-conditioned VAE. The
cooperative loop did not exist.

**Fix (keystone):** replace the hard `Embedding` with a **differentiable label
projection** — a bias-free `Dense` layer applied to the class-probability vector:

```python
self.label_projection = keras.layers.Dense(
    config.producer_initial_dense_units, use_bias=False, name="label_projection")
...
c = self.label_projection(y)   # y is [batch, num_classes] probabilities
```

**Why it works.** For a one-hot label, `y @ W` selects exactly one row of `W` — it *is*
an embedding lookup, so nothing is lost for the ground-truth generation path. But for the
Reasoner's *soft* output it is a smooth, differentiable function of `y`. The orchestrator
now passes probability vectors straight through (no `argmax`), so:

```
reconstruction_loss → Producer.label_projection → y_inferred_probs → Reasoner
```

is a continuous gradient path. Verified: the same gradient call now returns
**0/18 `None`**, norm ≈ 0.042.

With the path restored, `reasoner_error` becomes genuinely cooperative again:

```python
reasoner_error = w_inf * categorical_crossentropy(y_truth, y_inferred)
               + w_rec * reconstruction_loss
```

The crossentropy term is kept as a **stable, well-conditioned anchor** (it guarantees the
Reasoner trains even when the Producer is still weak); the reconstruction term adds the
causal coupling — the Reasoner is now penalised for inferences the Producer cannot turn
back into the original observation. This degrades gracefully: on hard data where the
Reasoner is inaccurate, the crossentropy anchor dominates, so the Producer is not pulled
toward label-insensitivity.

`SequentialCCNetOrchestrator.forward_pass` had the same `argmax` and was fixed the same
way (sequences of probability vectors are reversed and passed directly).

## Defect 3 — loss weights frozen at graph-trace time (HIGH)

`compute_model_errors` ran inside an `@tf.function`. It read loss weights as **Python
floats** straight off the config dict:

```python
... self.config.explainer_weights['kl_divergence'] * kl_loss
```

A Python float read inside a `tf.function` is captured as a **graph constant at the
first trace**. Subsequent calls with the same input signature do not retrace, so the
value is frozen forever. `CCNetTrainer`'s KL annealing dutifully mutated
`config.explainer_weights['kl_divergence']` every epoch — and the compiled graph never
saw any of it. KL annealing (and the deprecated `dynamic_weighting`) were silent no-ops.

Verified: setting the KL weight from `0.001` to `1000` left `explainer_error` unchanged.

**Fix:** the KL weight is now a `tf.Variable` on the orchestrator:

```python
self.kl_weight = tf.Variable(initial, trainable=False, dtype=tf.float32)
```

`compute_model_errors` reads `self.kl_weight`; `CCNetTrainer` calls
`orchestrator.kl_weight.assign(new_value)`.

**Why it works.** A `tf.Variable` is a graph node, not a constant — its value is read
live on every step without retracing. Verified: the same `0.001 → 1000` change now moves
`explainer_error` from 1.05 to 351159.

(The other weights — `inference`, `generation`, `reconstruction` — are left as Python
floats: they are never mutated during training, so baking them is correct and cheaper.
If `dynamic_weighting` is ever revived it must migrate to `tf.Variable`s too.)

## Defect 4 — documentation drift (MEDIUM)

`FOUNDATION.md` specifies `Reasoner Error = w_rec·Reconstruction + w_inf·Inference`, and
`README.md` documents config fields (`kl_weight`, `loss_type`) that do not exist
(`CCNetConfig` uses `loss_fn` and nests the KL weight under `explainer_weights`). The doc
was not a reliable spec for the code. The repair was driven by first principles and by
what actually trains; `FOUNDATION.md`/`README.md` should be reconciled with the code.

## What was deliberately NOT changed

- **Three isolated `GradientTape`s.** Each network's error is differentiated only with
  respect to its own variables; `producer_error` excludes `inference_loss`. This
  correctly isolates per-network credit and was kept as-is.
- **`tf.stop_gradient` on `e_latent` into the Reasoner.** This is a deliberate
  disentanglement choice: it prevents the Reasoner from reshaping the latent `E` into a
  label shortcut. The Explainer is shaped only by generation/inference/KL. Kept.
- **The `if train_reasoner:` branch in `train_step`.** It looked like an illegal Python
  branch on a symbolic tensor, but `@tf.function` AutoGraph rewrites it into a graph
  conditional. No crash. Not a defect.

## Files changed

| File | Change |
|------|--------|
| `models/ccnets/orchestrators.py` | `kl_weight` tf.Variable; `argmax` removed from both orchestrators; cooperative `reasoner_error` |
| `models/ccnets/trainer.py` | KL annealing `.assign()`s the variable instead of mutating a dict |
| `train/ccnets/mnist.py` | `Embedding` → differentiable `Dense` label projection; `experiment.run()` added; one-hot dummy in `create_mnist_ccnet`; `explanation_dim` 4 → 32 |

## Verification

- Cooperative gradient restored: `reconstruction_loss → reasoner` norm 0.042 (was 0.0).
- KL weight live: `explainer_error` 1.05 → 351159 on a weight change.
- 50-epoch MNIST: train accuracy 0.994; clean reconstructions; counterfactual matrix
  recombines style and label correctly.

## Follow-up — sparse embedding gradients (2026-05)

Surfaced while prototyping a text CCNet (`train/ccnets/text_sentiment.py`), where the
Reasoner contains an `Embedding` layer.

**Defect.** `train_step` selects the Reasoner gradients with `tf.cond(train_reasoner,
compute_grads, zero_grads)`. An `Embedding` produces a sparse `tf.IndexedSlices`
gradient, while the zero branch returns dense `tf.zeros_like`. `tf.cond` requires both
branches to return matching types and raised `TypeError: Cannot reconcile tf.cond
outputs`. The image task never hit this because its Reasoner had no embedding.

**Fix.** A `_densify` helper converts every gradient to a dense tensor (`None` → zeros,
`IndexedSlices` → dense) immediately after each `tape.gradient` call, so the whole
pipeline — `tf.cond` reconciliation, value clipping, global-norm — operates on uniform
dense tensors. Pinned by `test_densify_handles_none_dense_and_indexed_slices`.

## Remaining opportunities (not bugs)

- Generation fidelity is good but not sharp — a stronger Producer (residual upsampling,
  an SSIM/perceptual term) would help.
- Reconcile `FOUNDATION.md` / `README.md` with the implemented protocol.
- Add tests under `tests/test_models/test_ccnets/` (gradient-flow + serialization).
- Remove or fix the dead `dynamic_weighting` path.
