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
- `cifar100.py` — the image CCNet scaled to 32x32x3 natural images, 100 classes.
- `cifar100_hybrid.py` — CIFAR-100 CCNet with a LeWM-inspired latent-space
  verification term added alongside the pixel loss.
- `baseline_comparison.py` — controlled experiment: CCNet Reasoner vs. a plain classifier.
- `latent_sweep.py` — sweep of `explanation_dim` (the size of the latent cause `E`).

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

## Findings — CIFAR-100 (`cifar100.py`)

The image CCNet scaled to 32x32x3 natural images, 100 fine-grained classes (40 epochs).

| Signal | Result |
|--------|--------|
| Reasoner test accuracy | top-1 **0.547**, top-5 **0.823** |
| Generation / reconstruction L1 (MAE) | ~0.099 |

The Reasoner works — 54.7% top-1 is respectable for a small conv classifier on
CIFAR-100. The Producer does not: reconstructions are blurry low-frequency blobs that
capture scene colour and rough layout but no structure. This is the predicted P1/P2
outcome — a class label plus a 64-dim latent underdetermines a natural photograph, so
`P(X|Y,E)` collapses to the conditional mean. CIFAR-100 sits between MNIST (causes
determine `X` → crisp) and COCO (hopeless): the discriminative half stays useful, the
generative half degrades to blur as the sufficiency condition weakens.

## Findings — CCNet Reasoner vs. plain classifier (`baseline_comparison.py`)

Controlled MNIST comparison: identical `MNISTReasoner` architecture, equal compute,
3 seeds × 3 train-set sizes. **CCNets does not beat a plain classifier.** Where its
training succeeds it *ties* on accuracy (n=500, n=2000) or is slightly behind (n=60000),
and it is materially *less stable* — 2 of 9 cooperative runs collapsed to chance vs 0/9
for the baseline. Full numbers in `results/ccnets_baseline_comparison/comparison.md`.
CCNets' value is the generative / counterfactual capability, not discriminative accuracy.

## Findings — latent size `dim(E)` (`latent_sweep.py`)

Sweep of `explanation_dim` ∈ {4, 8, 16, 32, 64, 128} on MNIST, 20 epochs each.

| dim(E) | Recon MAE | Label-sensitivity | KL (nats) |
|-------:|----------:|------------------:|----------:|
| 4      | 0.080     | 0.114             | 6.4       |
| 16     | **0.063** (best) | 0.097      | **11.6** (peak) |
| 128    | 0.073     | 0.103             | 9.0       |

There is **no "bigger is better" scaling law** — reconstruction MAE is U-shaped with a
clear interior optimum at `dim(E) ≈ 16`:

1. **Undersize (dim=4)** is capacity-starved — worst reconstruction, lowest KL. This is
   the sufficiency floor: `E` too small to carry `H(X|Y)`.
2. **Oversize (dim ≥ 64)** is wasteful but **not** catastrophic. The achieved KL (the
   effective rate) peaks at dim=16 and *falls* beyond it — surplus dimensions collapse to
   the prior and go unused. Reconstruction degrades mildly from optimisation noise.
3. The "E absorbs Y" failure I expected did **not** appear — label-sensitivity stayed
   ~0.10 at every dim, so the Producer kept using the label. The KL regulariser prevents
   it: excess dimensions collapse to the prior rather than encoding `Y`.
4. The real knob is the **achieved KL rate**, which saturates (~11 nats) regardless of
   how many dimensions are provided.

Practical rule: size `dim(E)` modestly above the `H(X|Y)`-matched optimum (16–32 for
MNIST), keep the KL term on, and stop adding dimensions once the achieved KL stops rising.

## Findings — hybrid latent-space verification (`cifar100_hybrid.py`)

A LeWM-inspired variant. LeWM's defining strength is that it verifies in *latent
space* (predicts/compares embeddings, not raw observations). `cifar100_hybrid.py`
ports that as a hybrid: the Producer still outputs pixels (so counterfactual
generation survives), but `HybridCCNetOrchestrator` adds a verification term that
compares the produced and real images in a learned **feature space** — the
Reasoner's own `image_features` backbone, used as the encoder `phi`. The term is
added only to the Producer and Explainer errors; tape isolation (P7) keeps it out
of `reasoner_error`, so `phi` stays anchored to the classification objective and
cannot collapse — the JEPA "stop-gradient target" obtained for free.

CIFAR-100, 40 epochs, vs. the pixel-only `cifar100.py` run:

| | Pixel-only | Hybrid (+ latent verification) |
|--|-----------|--------------------------------|
| Reasoner top-1 / top-5 | 0.547 / 0.823 | 0.532 / 0.816 |
| Pixel gen / recon MAE | ~0.099 | ~0.101 |
| Latent (perceptual) MSE | — | gen 0.49, recon 0.47 (active) |
| Reconstruction grid | smooth low-frequency **blobs** | **structured, textured** |

Key finding: **at essentially identical pixel MAE (~0.10) the two produce visibly
different images** — the pixel-only run settles on blurry colour-average blobs, the
hybrid produces textured, roughly-structured reconstructions. Pixel L1 cannot
distinguish a blurry solution from a structured one of equal pixel error, so it
takes the conditional mean; the feature-space term *can* distinguish them and
redirects the Producer. The LeWM strength transferred and changed the failure mode
(blur → structured-but-imperfect).

Limits: it does **not** close the sufficiency wall — outputs are structured but
not photorealistic; a class label + 64-dim `E` still underdetermines a 32x32 photo.
Latent verification changes *which* underdetermined solution you get, not the
information available. Reasoner accuracy dipped slightly (within noise; the term
feeds Producer/Explainer, not the Reasoner). `phi` is a moving target — healthy
here (anchored by classification) but an EMA/frozen `phi` would be steadier.
