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

## Findings — `text_sentiment.py` Producer variants

Three Producer designs were tried on IMDB sentiment (10 epochs, vocab 5000, `max_len=80`,
`explanation_dim=32`). The observation `X` here is a discrete token sequence, so
`TextCCNetOrchestrator` swaps the pixel-norm losses for token-space losses (masked
cross-entropy for generation/reconstruction, masked KL for inference).

| Variant | Reasoner test acc | Val generation CE | Reconstruction | Counterfactual sentiment control |
|---|---|---|---|---|
| Non-autoregressive | 0.761 | **5.6** (plateau) | unigram collapse (`the the <oov> <oov> …`) | none |
| Autoregressive, prefix conditioning | 0.765 | **4.48** | structural, content-tracking | weak / none |
| Autoregressive + per-layer `Y` injection | 0.771 | **4.48** | structural, content-tracking | **works** (clear flips) |

Counterfactual evidence (per-layer variant): a negative review re-decoded as positive →
*"this movie is a **great film** … a **great movie** that is a **great movie**"*; a
positive review re-decoded as negative → *"… the **worst movies** i have ever seen … the
**worst movies ever made**"*.

### Lessons

1. **A small latent cannot carry a long sequence in one shot.** The non-autoregressive
   Producer must emit all `T` tokens from `(Y, E)` alone; with `E` only 32-dim it
   collapses to the marginal (unigram) token distribution. Generation CE plateaus high.
2. **Autoregression fixes reconstruction.** Predicting each token *with its own context*
   (teacher-forced causal decoder) drops generation CE 5.6 → 4.48 and yields structural,
   content-tracking text instead of collapse.
3. **A conditioning signal needs enough surface area to compete with the context.** A
   single `(Y, E)` prefix token is drowned out by ~80 autoregressive context tokens — the
   decoder ignores it. Injecting `Y` into the residual stream at *every* decoder layer
   turns sentiment into a real control knob (counterfactual flips start working) without
   changing generation CE. Control strength and language-model fluency are separate axes.
4. **Persistent limits.** Free greedy generation still drifts to `<oov>` runs after
   ~20 tokens (exposure bias + a 5000-token vocab); the Reasoner overfits (train ~0.97 vs
   test ~0.77).
5. **The deep caveat.** A movie review is *not determined* by its sentiment — so the
   CCNet necessity-&-sufficiency condition (`models/ccnets/PRINCIPLES_CCNETS.md`, P1/P2)
   only partly holds. The Producer is fundamentally a conditional language model in which
   `(Y, E)` are modulators, not the sole cause of `X`. The paradigm fits cleanly when the
   causes genuinely determine the effect (as label + style do for an MNIST digit); for a
   task where they do not, expect a working Reasoner and controllable-but-not-faithful
   generation.
