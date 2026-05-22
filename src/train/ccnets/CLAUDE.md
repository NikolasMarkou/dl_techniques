# train/ccnets — CCNet training scripts

Training pipelines for the Causal Cooperative Networks framework
(`dl_techniques/models/ccnets/`). Read that package's `CLAUDE.md` first — it covers the
paradigm, the network contract, and the non-negotiable invariants. This file covers only
the training-script layer.

## Files

- `mnist.py` — the reference task: MNIST digit classification + counterfactual generation
  (continuous observation `X`, pixel-norm losses). Treat it as the template for image tasks.
- `text_sentiment.py` — a CCNet over discrete token sequences (IMDB sentiment). Shows the
  adaptations for a non-continuous `X`: a non-autoregressive Producer and **token-space
  losses** via a `TextCCNetOrchestrator` that overrides `compute_losses`. Template for
  text / sequence tasks. (Prototype — see its module docstring for scope.)

## `mnist.py` structure (the template)

| Section | What to reuse / change |
|---------|------------------------|
| `ModelConfig` / `TrainingConfig` / `DataConfig` / ... | dataclass config blocks — keep the pattern, change fields |
| `ConvBlock`, `DenseBlock`, `FiLMLayer` | generic Keras 3 layers — reusable as-is |
| `MNISTExplainer` | `P(E\|X)` — returns `(mu, log_var)`. Swap the conv stack for your modality |
| `MNISTReasoner` | `P(Y\|X,E)` — concatenates `E` into the classifier head |
| `MNISTProducer` | `P(X\|Y,E)` — **note `label_projection`**: a bias-free `Dense`, NOT an `Embedding`. This differentiable label path is mandatory (see models/ccnets `CLAUDE.md`, Invariant 1) |
| `CCNetTrainingHistoryViz` etc. | visualization plugins — reusable, keyed on metric names |
| `create_mnist_ccnet` | builds the three nets (one-hot dummy label), wraps, configures, returns the orchestrator |
| `CCNetExperiment` | ties training + visualization + model saving together |

## Writing a new CCNet training script

1. Name it `train_<task>.py` (a bare `train.py` shadows the `train` package).
2. Copy `mnist.py`. Replace the three model classes, keeping the network contract:
   `explainer(x) -> (mu, log_var)`, `reasoner(x, e) -> y_probs`, `producer(y, e) -> x_hat`.
3. Keep the Producer's label input differentiable — `Dense(use_bias=False)` on the
   probability vector, never `argmax` + `Embedding`.
4. Replace data loading; the dataset must yield `(x, y)` with `y` one-hot.
5. Use `setup_gpu(args.gpu)` and `MPLBACKEND=Agg` for headless runs.
6. Write outputs to the repo-root `results/` directory.

```bash
MPLBACKEND=Agg .venv/bin/python -m train.ccnets.mnist
```

## Verified baseline

`mnist.py`, 50 epochs, `explanation_dim=32`: ~0.99 accuracy, clean reconstructions, and a
working counterfactual matrix (style `E` recombined with arbitrary labels `Y`). If a new
task does not show all three error signals decreasing together, re-check Invariant 1.
