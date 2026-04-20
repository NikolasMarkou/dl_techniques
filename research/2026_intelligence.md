# When Does a Neural Network Become More Than a Kernel Machine?

## A Three-Axis Decomposition of "Intelligence" and "Pure Generalization"

**Author**: Nikolas Markou
**Date**: 2026-04-19
**Methodology**: Epistemic deconstruction protocol (COMPREHENSIVE tier, L4 fidelity)
**Audit trail**: `analyses/analysis_2026-04-19_06e32b37/`
**Companion theory**: `src/dl_techniques/analyzer/SETOL.md`

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Why the Question is Three Questions](#2-why-the-question-is-three-questions)
3. [Axis A — Dynamical: Does the NTK Drift?](#3-axis-a--dynamical-does-the-ntk-drift)
4. [Axis B — Information-Theoretic: Does the Task Admit a Kernel-Impossible Gap?](#4-axis-b--information-theoretic-does-the-task-admit-a-kernel-impossible-gap)
5. [Axis C — Representational: Does the Trained Network Live at Criticality?](#5-axis-c--representational-does-the-trained-network-live-at-criticality)
6. [The 3 × 3 × 3 Regime Grid](#6-the-3--3--3-regime-grid)
7. [Where "Intelligence" Lives](#7-where-intelligence-lives)
8. [Why Domingos' Path-Kernel Claim is Operationally Vacuous](#8-why-domingos-path-kernel-claim-is-operationally-vacuous)
9. [The HTSR/SETOL Bridge — Necessary but Not Sufficient](#9-the-htsrsetol-bridge--necessary-but-not-sufficient)
10. [The Causal Graph](#10-the-causal-graph)
11. [Falsifiable Predictions](#11-falsifiable-predictions)
12. [Quantitative Regime Thresholds](#12-quantitative-regime-thresholds)
13. [Implications for `dl_techniques`](#13-implications-for-dl_techniques)
14. [What is Genuinely Uncertain](#14-what-is-genuinely-uncertain)
15. [Hypothesis Registry & Final Posteriors](#15-hypothesis-registry--final-posteriors)
16. [References](#16-references)

---

## 1. Executive Summary

The question "under which conditions does a neural network become *more than* a kernel machine?" is — as commonly phrased in popular treatments and as it appeared in the prompting Gemini exchange — **three orthogonal questions disguised as one**. Each has a precise answer; merged, they produce the perceived paradox.

The decomposition:

| Axis | What it asks | The lever | The "more than kernel" answer |
|:---|:---|:---|:---|
| **A — Dynamical** | Did the empirical NTK drift during training (K_t ≠ K_0)? | parameterization (μP vs NTK), init scale, finite-width SGD noise | μP at any width, OR finite width + non-trivial training |
| **B — Information-theoretic** | Does the *task* admit a sample-complexity gap that no fixed kernel can close? | data structure (low-dim signal in high-dim input, hierarchical composition) | sparse-index, multi-index, or depth-leveraged compositional targets (provable separations) |
| **C — Representational** | Did the trained weights organize into a critical configuration that produces transferable features? | training horizon × lr/B × regularization × data quality | HTSR α ≈ 2 with ERG satisfied, AND mech-interp recovers monosemantic primitives |

"Intelligence" and "pure generalization", as colloquially used, map onto **the conjunction A1 ∧ B1/B2 ∧ C1**. This is a *configuration*, not a property. Frontier models (GPT, Claude, Gemini) live in this cell by design — μP-equivalent parameterization, training on data with rich latent structure, training horizons and regularization tuned to land in α ≈ 2.

**Headline conclusions** (Bayesian posteriors after disciplined evidence updates):

- **CONFIRMED** (≥ 0.90): task-structure separation (H2 = 0.95), HTSR/SETOL spectral signature as regime detector (H3 = 0.96), framing-error meta-claim that the question conflates three phenomena (H7 = 0.93).
- **STRONG** (0.80–0.90): parameterization gating (H1 = 0.86), SGD/finite-width dynamics (H5 = 0.87), depth-leveraged compositional separation (H6 = 0.90), mechanistic interpretability as discriminator (H11 = 0.89).
- **REFUTED** (≤ 0.05): Domingos' "every model is approximately a kernel machine" in its strong form (H4 = 0.03). Survives only as a predictively-vacuous mathematical re-description.

---

## 2. Why the Question is Three Questions

The popular framing ("when does a NN stop being a kernel machine?") implicitly assumes a **binary switch**. There is no such switch. There are three independent axes, and the question makes sense only when each is named separately.

### The conflation pattern

A typical popular treatment will say:
> "Networks transcend kernel machines when they enter the rich/feature-learning regime, characterized by small initialization, μP parameterization, and ability to learn hierarchical features on structured data."

This sentence merges:
1. *Dynamical* claim: the NTK drifts (K_t ≠ K_0).
2. *Information-theoretic* claim: there exist tasks where NN sample complexity strictly beats any fixed kernel.
3. *Representational* claim: the result is a useful internal representation (not noise, not memorization).

These are independently true under different conditions. They can come apart. A rich-dynamical network on isotropic noise (A1, B0, anything) is not "smarter" than a kernel — it just drifted pointlessly. A lazy network on structured data (A0, B1, C0) underperforms because the kernel cannot capture the structure regardless of dynamics. A rich-dynamical network on structured data that overshoots into the glassy regime (A1, B1, C2) memorizes and degrades.

**Deconstruction Hypothesis 7** (the "framing-error meta-hypothesis") was confirmed at posterior 0.93. Most of the literature's apparent contradictions (Domingos vs Yang-Hu, NTK vs feature learning, scaling vs emergence) dissolve once the three axes are kept separate.

---

## 3. Axis A — Dynamical: Does the NTK Drift?

### The lazy regime

Under standard NTK parameterization with weights initialized as W ~ 𝒩(0, σ² / n_in), and width n → ∞, training with infinitesimal-step gradient flow keeps the empirical NTK frozen:

$$
K_t(x, x') = \langle \nabla_\theta f(x; \theta_t), \nabla_\theta f(x'; \theta_t) \rangle \xrightarrow{n \to \infty} K_0(x, x')
$$

The network output evolves as a linear regression in the fixed feature map ϕ_θ₀(x) = ∇_θ f(x; θ_0). This is the **NTK regime** (Jacot, Gabriel, Hongler 2018; Lee et al 2019; Arora et al 2019). In this regime, the network *is* a kernel machine in the strictest sense — it solves a fixed kernel ridge regression.

Empirical signature: ‖K_t − K_0‖_F / ‖K_0‖_F → O(1/√n). For width 8192, drift is < 1 % of K_0 norm.

### Escaping the lazy regime

Three confirmed mechanisms:

#### 3.1 μP / Maximal Update Parameterization (Yang & Hu 2021)

By scaling the last layer as O(1/n), inner layers as O(1/√n), and learning rates inversely (last layer O(n), inner O(1)), the infinite-width limit becomes **non-trivially dynamical**. K_t evolves with t even at n → ∞. The engineering payoff is **hyperparameter transfer across widths**: tune learning rate at width 256, transfer to width 8192. This is the empirical signature that you have left the lazy regime.

#### 3.2 Mean-field / small-output-scale (Chizat-Bach 2019)

For a model y = α_scale · f(x; θ), the lazy/rich transition is governed by the single scalar α_scale:

- α_scale → ∞: lazy linearization (kernel regime).
- α_scale → 0 with compensating learning rate: rich feature learning.

The transition is continuous in α_scale but the asymptotic regimes are qualitatively different.

#### 3.3 Finite width + SGD noise (always, in practice)

Practical CIFAR/ImageNet ResNets and CNNs at standard widths (256–2048) show ‖K_end − K_0‖ / ‖K_0‖ in the range **0.3–0.8** (Fort, Dziugaite, Paul, Kornblith, Roy, Ganguli 2020; Geiger, Spigler, Jacot, Wyart 2020). The drift is largest in the first 10–20% of training. The "Final-NTK" kernel regression matches the trained network closely; the "Initial-NTK" regression does not.

**Important**: this means the strict NTK regime is a thought experiment. Nobody trains in it. Even at standard parameterization, finite-width SGD always drifts. The lazy regime in its pure form requires both infinite width *and* NTK parameterization *and* infinitesimal-step gradient flow.

### What drives the drift mechanism (R1 reinforcing loop)

```
muP / small-init  →  gradient updates have macroscopic effect on activations
                  →  K_t rotates
                  →  effective features align with data
                  →  larger gradient signal in those features
                  →  more drift
```

Self-reinforcing positive feedback. Established by Yang-Hu 2021 and Bordelon-Pehlevan 2022 ("Self-Consistent Dynamical Field Theory of Kernel Evolution").

### Detection in practice

| Indicator | Lazy | Rich |
|:---|:---|:---|
| ‖K_t − K_0‖_F / ‖K_0‖_F | < 0.1 | 0.3 – 0.8 |
| Final-NTK vs Initial-NTK kernel regression | matches | Final-NTK >> Initial-NTK |
| HP transfer across width | works under NTK-param at large n | works under μP at any n |
| Loss curve under width scaling | width-invariant | width-dependent (NTK) or width-invariant (μP) |

---

## 4. Axis B — Information-Theoretic: Does the Task Admit a Kernel-Impossible Gap?

This is the **only axis** where "more than a kernel" admits a *provable, information-theoretic* answer. The other two axes are about dynamics and representation; this one is about whether *anything* could match a feature-learning network's sample efficiency on a given task.

### Provable separations exist

#### 4.1 Single-index targets (Allen-Zhu, Li, Liang 2019; Ben Arous et al 2021)

For a target f*(x) = g(⟨u, x⟩) where u is a unit vector in d-dim isotropic input and g is a degree-k polynomial:

- **Any rotationally-invariant kernel** (including the NTK at any depth) requires N ≳ d^k samples for low risk.
- A gradient-trained NN with feature learning (mean-field or μP) requires N ≳ d samples after a "saddle escape" phase.

The gap is **d^(k-1)** in sample complexity. Information-theoretic, not just statistical: no fixed kernel can match feature-learning sample complexity on these targets.

#### 4.2 Multi-index targets (Bach 2017; Ghorbani-Mei-Misiakiewicz-Montanari 2020)

For f*(x) = g(U^T x) with U a low-dimensional subspace embedded in d-dim space, the same gap holds. The NN learns to project onto U in early training, then fits g; the kernel must cover all rotational alignments of U with poly-many features.

#### 4.3 Hierarchical compositions (Telgarsky 2016; Eldan-Shamir 2016)

Depth provides exponential expressivity gains. Functions a depth-(L+1) network can approximate with poly width require exp(L) width for a depth-2 network. Any fixed kernel is depth-1 in feature space; consequently, any fixed kernel is exponentially weaker than a deep network on hierarchically composed targets.

The dl_techniques codebase's residual and hierarchical architectures (`models/resnet/`, `models/swin_transformer/`, `layers/hierarchical_mlp_stem.py`) are direct implementations of this principle.

### The disconfirming case

On **rotationally-invariant Hölder-smooth targets in low effective dimension**, NN ≈ kernel. No magic. (Ghorbani-Mei null result; Adlam-Pennington 2020.)

Also: on natural-image data (CIFAR-10), well-engineered convolutional NTK / NNGP kernels achieve **88–89% accuracy** vs trained CNNs at 95–96%. The gap exists, but is < 10 absolute points — *not* an order-of-magnitude separation. Real-world data sits in a middle zone where the latent low-dimensional structure is mild, not overwhelming.

This is why Hypothesis 2 is confirmed in its **conditional form**: separations are provable, but require the data to have low-dimensional structure embedded in high-dimensional input. On truly isotropic data with no exploitable structure, NN and kernel are equivalent.

### Connection to the data manifold hypothesis

"Real data lives on a low-dimensional manifold in high-dimensional input space" is the ML folklore. Axis B is its precise mathematical statement: the kernel-impossible regime exists exactly when the data manifold has dimension k ≪ d and the target depends on the manifold coordinates polynomially. Image, audio, and language data all qualify; pure tabular data with no latent structure does not.

---

## 5. Axis C — Representational: Does the Trained Network Live at Criticality?

This is the axis where SETOL — the framework documented in `src/dl_techniques/analyzer/SETOL.md` and implemented in `analyzer/spectral_metrics.py` — provides the operational answer **without using test data**.

### The HTSR α evolution

For a layer with weight matrix W of dimensions M × N, define the correlation matrix X = (1/N) W^T W and its empirical spectral density (ESD). The tail follows ρ_tail(λ) ~ λ^(-α). Across well-trained models (VGG, ResNet, BERT, GPT-2 family), the per-layer α exhibits a robust trajectory:

- **Random initialization**: α ≈ 6 (Marchenko-Pastur regime).
- **Under-trained**: α > 4.
- **Critical / ideal**: α ≈ 2.
- **Glassy / over-regularized**: α < 2 (rank-1 correlation spikes).

The transition is monotone in training time under correct learning-rate / batch-size schedules.

### The ERG (Exact Renormalization Group) condition

A layer achieves **ideal learning** when:

$$
\ln \det(\tilde{X}) = \sum_{i \in \text{ECS}} \ln \lambda_i \approx 0
$$

This corresponds to **volume preservation** in the transformation from random to trained weights, and to a **critical phase boundary** in the physics sense (scale invariance, power-law correlations, maximal correlation length).

### The reinforcing/balancing loop structure

```
R2 (training time → spectral concentration):
  SGD step  →  top eigenvectors of W^T W concentrate signal
            →  top eigenvalues grow
            →  power-law tail develops
            →  α decreases from 6 toward 2
  Self-reinforcing.

B1 (excess training → glassy collapse):
  Too much R2  →  α drops below 2
              →  rank-1 correlation traps form
              →  layer over-fits training-direction noise
              →  test performance DEGRADES
  Balancing — must be cut by early stopping or lr schedule.
```

This is why the user's intuition that "more rich = more intelligent" is wrong. The optimum is at α ≈ 2 (the critical strip), not at α → 0.

### Why criticality matters

Critical points in physics have:

- **Power-law correlations** (no characteristic length scale).
- **Scale invariance** under renormalization.
- **Maximal susceptibility** — the system responds maximally to perturbations.

A neural network at α ≈ 2 has these properties at the level of its weight spectrum. The Free Cauchy bound from SETOL gives:

$$
\log_{10} \bar{Q}_{FC} \sim \frac{1}{\alpha}
$$

So smaller α (toward 2) corresponds to better predicted layer quality. Below 2 the bound diverges in the wrong direction (glassy regime).

### Mechanistic interpretability — the semantic complement

HTSR/SETOL is **necessary but not sufficient**. It detects that the spectrum is heavy-tailed; it cannot tell you *which features* the heavy tail encodes. A layer with α = 2 might encode useful primitives or might encode noisy projections of them.

The complementary discriminator is **mechanistic interpretability** (Olsson et al 2022; Bricken et al 2023; Templeton et al 2024; Marks et al 2024):

- **Sparse autoencoders** decompose intermediate activations into monosemantic features (specific concepts: "Golden Gate Bridge", syntactic structures, programming idioms).
- **Induction heads** are explicit compositional circuits implementing in-context-learning algorithms.
- **Causal interventions** on features predictably change model behavior.

These are demonstrably *not* in the random-init NTK feature space — they are computational primitives the network constructed during training.

The complementarity:

| Question | HTSR/SETOL | Mech-Interp |
|:---|:---:|:---:|
| Is the layer in the critical phase? | YES | partial |
| Are the features task-relevant? | partial (predicts quality) | YES |
| Which specific features were learned? | NO | YES |
| Detection without test data? | YES | YES (with corpus) |
| Computational cost | low (eigendecomposition) | high (SAE training) |

**Use both.** Spectral signatures are the cheap regime detector; mech-interp is the expensive semantic verifier.

---

## 6. The 3 × 3 × 3 Regime Grid

| Axis A (Dynamical) | Axis B (Info-theoretic) | Axis C (Representational) | Regime | Example | "More than kernel"? |
|:---|:---|:---|:---|:---|:---:|
| A0 lazy | B0 no advantage | C0 random | inf-width MLP at NTK init, isotropic Hölder target | toy theory model | **NO** — strictly kernel |
| A0 lazy | B1/B2 advantage | C0 random | inf-width MLP at NTK init, structured task | impossible — task asks for A1 | NO, suboptimal |
| A1 rich | B0 no advantage | C1 critical | trained on isotropic Hölder target | over-parameterized noise fit | **NO net win** — drifted pointlessly |
| **A1 rich** | **B1 advantage** | **C1 critical** | **well-trained ResNet on CIFAR** | **most SOTA vision models** | **YES — and we know why** |
| **A1 rich** | **B1/B2 advantage** | **C1 critical** | **well-trained Transformer on language** | **LLMs (GPT, Claude, Gemini)** | **YES — and at criticality** |
| A1 rich | B1/B2 advantage | C2 glassy | over-trained / lr too high | rare; correlation traps | **YES dynamically NO outcome** |
| A1 rich | B1/B2 advantage | C0 random | massively under-trained large model | early-checkpoint frontier model | partial — drifting but not yet critical |

**The cell that matches the user's intuition**: A1 ∧ B1/B2 ∧ C1. This is the only configuration that is simultaneously (a) non-kernel in dynamics, (b) sample-efficient in a way no kernel can match, and (c) representationally critical. All three are necessary; any two without the third fails.

---

## 7. Where "Intelligence" Lives

The word "intelligence" in the original question is not a mathematical category. The closest mathematical analog is:

> **A network exhibits "intelligence-like generalization" when it operates at the critical phase boundary (Axis C1, α ≈ 2 with ERG satisfied) on tasks whose latent structure rewards feature learning (Axis B1 or B2), with dynamics that escaped the lazy regime (Axis A1).**

This is **a configuration**, not a property. Frontier models (GPT-4, Claude, Gemini, etc.) land in this cell by design:

- **Axis A** — they use μP-equivalent parameterizations (or scale large enough that finite-width SGD drift is dominant).
- **Axis B** — they train on internet-scale text that has rich latent linguistic, semantic, and compositional structure.
- **Axis C** — their training horizons, learning-rate schedules, and regularization are tuned (often empirically) to land in α ≈ 2.

**Emergence** at scale (Wei et al 2022; debated by Schaeffer et al 2023) is the predictable super-additive interaction of the three axes. Predicting the loss curve from any single-axis model produces O(1) error on emergent capabilities. This is the operational meaning of emergence stripped of mysticism: **non-additive composition of three independently-required regime conditions**.

The phrase "pure generalization" is similarly a category error unless interpreted as: *generalization that no fixed kernel of comparable architecture can achieve*. This is precisely cell (A1, B1/B2, C1).

---

## 8. Why Domingos' Path-Kernel Claim is Operationally Vacuous

Domingos 2020 ("Every Model Learned by Gradient Descent Is Approximately a Kernel Machine", arXiv:2012.00152) writes any trained model as:

$$
f(x) = \sum_i a_i K_{\text{path}}(x_i, x)
$$

where K_path is the "path kernel" induced by the training trajectory. **This is mathematically valid.** The framing was widely repeated (including by Gemini in the prompting exchange) as evidence that "neural networks are just kernel machines."

It is not. The argument fails on three independent grounds:

### 8.1 The path kernel is not a fixed feature map

K_path depends on the entire training trajectory: data, init, learning rate, duration, batch ordering. It is a re-description of the trained model, not a function-class claim. Equivalent to "every function can be written as a sum of basis functions" — true but trivial, because the basis depends on the function being represented.

### 8.2 No predictive content

Classical kernel theory provides closed-form generalization bounds (Rademacher complexity, eigenvalue decay rates), sample-complexity guarantees, and explicit minimax rates. The path-kernel framing yields **none of these**, because the kernel is defined post-hoc from the trajectory.

### 8.3 Empirical refutations

- **Finite-width NTK drift** (Fort et al 2020): predictions made with K_0 are systematically wrong; only the "Final-NTK" matches the network — and the Final-NTK is *trajectory-dependent*.
- **Information-theoretic separations** (Allen-Zhu et al 2019; Eldan-Shamir 2016): provable separations exist between NN sample complexity and any fixed-kernel sample complexity on multi-index targets. The path-kernel framing does not contradict this — it sidesteps it by allowing the kernel to depend on the target. But that *is* feature learning, just under a different name.
- **Mechanistic interpretability** (Olsson et al 2022; Bricken et al 2023): networks compute concepts (induction heads, monosemantic features) that demonstrably are not in the init-NTK feature space.

### Conclusion

The path-kernel framing is **true in the same trivial sense that "every function is a Fourier series" is true**: yes, but the basis is the function. It teaches us nothing about regimes.

The strong form ("therefore neural networks are not really doing anything new") is **REFUTED** at posterior 0.03. The weak form (the path-kernel re-description exists) is true and operationally vacuous.

This is the part of the literature that, mis-cited, produces the most popular-press confusion.

---

## 9. The HTSR/SETOL Bridge — Necessary but Not Sufficient

The SETOL framework (Martin & Hinrichs 2025; documented in `src/dl_techniques/analyzer/SETOL.md`) is the bridge between the dynamical and representational axes. It explains *why* the rich regime is special — not just "different from kernel" but specifically at a **critical phase boundary in the physics sense**.

### What SETOL provides

1. **Empirical detector for Axis C** — α and ERG residual operationalize the critical / glassy / random sub-regimes. The dl_techniques codebase already implements these in `src/dl_techniques/analyzer/spectral_metrics.py`.

2. **Theoretical justification via free probability** — the Effective Hamiltonian + HCIZ integrals derive *why* α ≈ 2 corresponds to maximal generalization. The Free Cauchy bound:

   $$
   \log_{10} \bar{Q}_{FC} \sim \frac{1}{\alpha}
   $$

   gives smaller α → better predicted layer quality, with the bound diverging unhelpfully below α = 2 (the glassy regime).

3. **A bridge between A1 and C1** — connects the dynamical regime (NTK drifts) to the representational regime (spectral signature). The same training dynamics that produce K_t evolution also produce the heavy-tailed weight spectra.

### What SETOL does NOT provide

1. **No Axis B test** — SETOL cannot tell from spectra alone whether the *task structure* rewards feature learning. A network trained on isotropic noise might also reach α ≈ 2 (drifted pointlessly).

2. **No semantic feature decomposition** — SETOL cannot say *which* features were learned. Two networks with identical α profiles might encode entirely different concepts. This requires mechanistic interpretability.

### The complementary pairing

| Need | Use |
|:---|:---|
| Cheap, test-data-free regime classification | HTSR / SETOL (`analyzer/spectral_metrics.py`) |
| Verification that learned features are task-relevant | mech-interp (sparse autoencoders, circuit discovery) |
| Verification that NTK actually drifted | direct NTK estimation (Jacobian-vector products against init) |
| Verification that the task admits a kernel-impossible gap | sample-complexity scaling experiments (synthetic multi-index benchmarks) |

A rigorous regime audit needs all four. Most published work uses only one and over-claims. SETOL/HTSR is the easiest and cheapest, hence its prominence — but it is one of three legs.

---

## 10. The Causal Graph

```
                CAUSES
   ┌──────────────────────────────────┐
   │ init_scale   parameterization    │
   │ (sigma)      (NTK / muP / MF)    │
   │                                  │
   │ width n      depth L             │
   │                                  │
   │ optimizer    lr / batch_size B   │
   │ (SGD/Adam)   (effective lr/B)    │
   │                                  │
   │ data D:                          │
   │   intrinsic_dim k                │
   │   anisotropy A                   │
   │   compositional_depth d_comp     │
   │                                  │
   │ training_steps T                 │
   │ regularization (wd, dropout)     │
   └──────────────┬───────────────────┘
                  │
                  ▼
         ┌─────────────────────────┐
         │      STATES (t)         │
         │  K_t (empirical NTK)    │
         │  ||K_t - K_0|| / ||K_0||│  ◄── R1: muP / small-init / SGD-noise → larger drift
         │  alpha_l(t) per layer   │  ◄── R2: training → alpha decreases from 6 to 2
         │  ECS rank, ERG residual │  ◄── R3: feature learning → ECS dimensionality grows
         │  flat-minimum geometry  │
         └──────────┬──────────────┘
                    │
                    ▼
         ┌─────────────────────────┐
         │     OBSERVABLES         │
         │  alpha_hat (HTSR)       │
         │  Final-NTK kernel reg   │
         │  Mech-interp circuits   │
         │  SAE feature monosemy   │
         │  Generalization gap     │
         │  HP transfer across n   │
         └──────────┬──────────────┘
                    │
                    ▼
         ┌──────────────────────────┐
         │       OUTCOMES           │
         │  Lazy regime (kernel)    │
         │  Critical/ideal (α=2)    │
         │  Glassy (α < 2)          │
         │  Under-trained (α > 4)   │
         │  Sample-complexity:      │
         │    N(NN) << N(kernel)    │
         │    on multi-index/depth  │
         └──────────────────────────┘
```

### Reinforcing and balancing loops

- **R1 (parameterization → drift)** — μP / small-init produces macroscopic gradient effect on activations; K_t rotates; effective features align with data; gradient signal grows in those features; more drift. Self-reinforcing.
- **R2 (training time → spectral concentration)** — SGD step concentrates signal in top eigenvectors of W^T W; top eigenvalues grow; power-law tail develops; α decreases monotonically from MP-baseline (~6) toward critical (~2). Self-reinforcing under correct lr / batch.
- **B1 (excess training → glassy collapse)** — too much R2 takes α below 2; rank-1 correlation traps form; over-fits training-direction noise; test performance degrades. Balancing — must be cut by early stopping or lr schedule.
- **R3 (depth → compositional alignment)** — depth + compositional target → backward feature correction → late layers shape early-layer gradients → early layers extract task-relevant primitives → late layers compose them more cleanly → larger depth advantage with depth.

---

## 11. Falsifiable Predictions

Each of the strong hypotheses generates a concrete experimental design that would refute it. These are the operational tests that distinguish this analysis from a literature summary.

| Test | Predicts | Falsifies if |
|:---|:---|:---|
| Train μP-net at width 1024 vs NTK-param net at width 1024, equal compute, same data | μP shows ‖K_t − K_0‖ ~ O(1); NTK-param shows ‖K_t − K_0‖ ~ O(1/√n) | Equal drift between parameterizations → H1 falsified |
| Sparse-parity in d=100 with k=5 | NN reaches low loss at N ~ d; kernel needs N ~ d^5 ≈ 10^10 | Kernel matches NN sample efficiency → H2 falsified |
| Track α across training of the same architecture on CIFAR vs random labels | α → 2 on real data; α stays > 4 on random labels | Identical α trajectory → H3 falsified |
| Compute Final-NTK regression on a CIFAR-trained ResNet | Final-NTK matches network test acc; Initial-NTK does not | Both kernels match → H5 falsified |
| Construct Eldan-Shamir target (radial composition) and train depth-2 vs depth-3 networks | depth-2 needs exp(d) width; depth-3 fits with poly width | Poly-width depth-2 fits → H6 falsified |
| Sparse autoencoders on Pythia-{70M, 410M, 1.4B, 6.9B} | Recover monosemantic features at scale; feature count scales sub-linearly with model size | Only superpositions, no clean features → H11 weakened |

These predictions are designed to **break** the hypotheses, not confirm them. Most have already been run in the cited literature; the ones marked as "open" are testable in the dl_techniques codebase with relatively modest compute (sparse-parity: hours on a single GPU; Eldan-Shamir: hours; Final-NTK on CIFAR: half a day; SAE on Pythia: days).

---

## 12. Quantitative Regime Thresholds

The L4 (PARAMETERS) fidelity payoff: every regime has measurable thresholds the analyst can check against a trained checkpoint.

| Quantity | Lazy (kernel) | Critical (rich) | Glassy (over-reg) | Under-trained |
|:---|:---:|:---:|:---:|:---:|
| α per layer (HTSR) | > 4 | 1.8 – 2.5 | < 1.8 (with rank-1 spikes) | > 4 |
| ‖K_t − K_0‖_F / ‖K_0‖_F | < 0.1 | 0.3 – 0.8 | > 0.5 | < 0.05 |
| ECS dimension / total dim | low | medium-high | high with rank-1 spikes | low |
| ERG residual ‖ln det X̃‖ | far from 0 | ≈ 0 | far from 0 | far from 0 |
| Generalization gap | wide (kernel limit) | small | widens | wide |
| HP transfer across width | yes (NTK-param) | yes (μP) | broken | n/a |
| Sample-complexity vs kernel on multi-index | equal or worse | better, by d^(k-1) | comparable but unstable | undefined / random-like |
| Mech-interp: monosemantic features recoverable | ~no | yes | partial / noisy | ~no |

These bands are calibrated against:
- Martin & Mahoney 2021 for the α bands.
- Fort et al 2020 and Geiger et al 2020 for the NTK-drift bands.
- Templeton et al 2024 for the SAE monosemanticity expectations.

---

## 13. Implications for `dl_techniques`

The codebase already has the spectral half of the regime detector. Two upgrades make it a complete three-axis classifier.

### 13.1 What exists today

The `analyzer/` subpackage (`src/dl_techniques/analyzer/`) already implements:

- **`spectral_metrics.py`** — per-layer α, α̂ = α · log₁₀(λ_max), ECS computation
- **`spectral_analyzer.py`** — orchestration across all layers
- **`spectral_visualizer.py`** — diagnostic plots (Funnel diagnostic, spectral evolution)
- **SETOL theoretical foundation** documented in `analyzer/SETOL.md`

This gives **Axis C** for free. Run it on every checkpoint; you get the C-axis classification (critical / glassy / random) without test data.

### 13.2 Recommended additions

#### 13.2.1 NTK-drift estimator (Axis A)

A new module `analyzer/ntk_analyzer.py` that:

1. Takes a held-out set of K samples (K ~ 64 sufficient).
2. Computes Jacobian-vector products ∂f/∂θ for each sample at t=0 and t=current.
3. Builds the empirical NTK matrices K_0, K_t.
4. Returns ‖K_t − K_0‖_F / ‖K_0‖_F as the drift metric.
5. Optionally fits a Final-NTK kernel regressor and reports its test accuracy as the "what would a frozen-final-kernel achieve" baseline.

This is implementable in ~200 lines of Keras 3. The dl_techniques codebase's preference for `keras.ops` for backend-agnostic operations applies directly.

#### 13.2.2 Sparse autoencoder feature extractor (Axis C semantic complement)

A new model directory `models/sparse_autoencoder/` implementing:

1. Top-K SAE or JumpReLU SAE (current SOTA from Templeton et al 2024 / Gao et al 2024).
2. Training loop for fitting on intermediate activations from any pre-trained model in the dl_techniques zoo.
3. Feature visualization: top-activating examples per dictionary feature.
4. Causal intervention API: ablate / amplify a feature, measure downstream activation changes.

This is a multi-week implementation effort but is the strict upgrade over spectral-only analysis.

#### 13.2.3 Regime classifier (combining all three axes)

A `analyzer/regime_classifier.py` that takes a trained checkpoint plus a held-out data sample, runs spectral analysis + NTK drift estimation, and returns a regime label drawn from the 3 × 3 × 3 grid of section 6, with per-axis confidences.

Output schema:

```python
{
    "axis_A_dynamical": {"label": "rich" | "lazy" | "intermediate", "drift": 0.45, "confidence": 0.85},
    "axis_C_representational": {"label": "critical" | "glassy" | "random", "alpha_per_layer": [...], "erg_residual": 0.02, "confidence": 0.90},
    "regime": "A1_C1_critical",
    "predicted_outcome": "good_generalization" | "memorization" | "underfitting",
}
```

Axis B is task-structure-dependent and lives outside the analyzer (it requires sample-complexity scaling experiments on the specific task), so the per-checkpoint classifier is naturally A + C.

### 13.3 Connection to existing components

- **`models/cliffordnet/`, `models/kan/`, `models/squeezenet/`** — spectral analysis applies layer-by-layer; geometric/algebraic layers may have non-standard ESDs worth characterizing.
- **`losses/calibration.py`, `metrics/`** — calibration quality should correlate with α convergence to the critical strip; worth empirical study.
- **`callbacks/`** — a regime-classifier callback that runs the per-axis analysis every N epochs would expose regime trajectories during training, not just at the end.
- **`optimization/deep_supervision.py`** — deep supervision is a mechanism that drives early layers toward critical α faster; the framework here predicts that.

---

## 14. What is Genuinely Uncertain

Honest disclosure of the limits of this analysis:

### 14.1 LLM-scale evidence gap

Most empirical evidence comes from CIFAR/ImageNet/Pythia-scale work. Frontier models (GPT-4, Claude, Gemini) have proprietary measurements of NTK drift, α evolution, and emergent-capability scaling that are not published. Public-domain analysis on smaller models cannot resolve the regime where the question matters most.

This was hypothesis H12 (evidence-gap meta-claim), confirmed at posterior 0.62. Conclusions about frontier-model regime classification are extrapolated from smaller-model empirical work and should be weighted accordingly.

### 14.2 α ≈ 2 at LLM scale

The HTSR α ≈ 2 critical exponent is robust across MLPs, CNNs, ResNets, and small-to-medium transformers. Whether the same exponent holds at LLM scale — and whether the ERG condition is the right derivation at that scale — is an active research area (Martin-Hinrichs 2025 are still validating SETOL on transformer LLMs).

### 14.3 SAE feature stability

Mechanistic-interpretability monosemantic features are robust under small training perturbations but may shift under significant data-distribution changes. The relationship between SAE features and HTSR α at large scale is essentially uncharacterized.

### 14.4 Higher-axis interactions

The 3 × 3 × 3 grid presumes axes are independent. Empirically there are coupling effects — e.g., very deep networks at small init can spontaneously enter the critical regime even on isotropic data ("self-organized criticality"). These couplings are not yet well-characterized and likely require a 4th or 5th axis.

### 14.5 Architecture coverage

The bulk of the cited evidence comes from MLPs, CNNs, ResNets, and vanilla Transformers. Mixture-of-Experts (`models/qwen/`), state-space models (Mamba), diffusion architectures, geometric algebra networks (`models/cliffordnet/`), and KANs (`models/kan/`) may have additional regime-determining factors not captured by this framework.

### 14.6 The "intelligence" definition

The mathematical analog offered in §7 is one of many possible operationalizations. It does not exhaust the colloquial meaning — only the parts of "intelligence" that map onto sample-efficient generalization on structured tasks. Other facets (transfer, continual learning, calibrated uncertainty, adversarial robustness, agentic planning) require their own decompositions.

---

## 15. Hypothesis Registry & Final Posteriors

Tracked via Bayesian updating with disciplined evidence calibration (LR ≤ 5.0, no batched evidence, ≥ 1 disconfirming evidence applied to every hypothesis exceeding 0.80). Full audit trail in `analyses/analysis_2026-04-19_06e32b37/hypotheses.json`.

| ID | Hypothesis (statement abbreviated) | Prior | Posterior | Status |
|:---|:---|---:|---:|:---|
| **H1** | Parameterization gating (NTK vs μP/mean-field) | 0.30 | **0.86** | ACTIVE |
| **H2** | Task-structure separation (provable on multi-index, hierarchical) | 0.85 | **0.95** | CONFIRMED |
| **H3** | HTSR/SETOL spectral signature (α ≈ 2 critical) | 0.65 | **0.96** | CONFIRMED |
| **H4** | Domingos path-kernel claim (strong form) | 0.20 | **0.03** | REFUTED |
| **H5** | SGD noise + finite-width dynamics drive K_t drift | 0.45 | **0.87** | ACTIVE |
| **H6** | Depth-leveraged compositional separation | 0.60 | **0.90** | ACTIVE |
| **H7** | Framing-error meta-hypothesis (3 axes conflated) | 0.70 | **0.93** | CONFIRMED |
| **H8** | [H_S] Drivers within initial scope S | 0.70 | **0.56** | ACTIVE |
| **H9** | [H_S′] Material drivers exist outside S | 0.30 | **0.52** | ACTIVE |
| **H10** | Architectural inductive bias as independent driver | 0.55 | **0.40** | WEAKENED |
| **H11** | Mechanistic interpretability as discriminator | 0.55 | **0.89** | ACTIVE |
| **H12** | Public-domain evidence under-samples LLM regime | 0.50 | **0.62** | ACTIVE |

Note that H10 was promoted from the scope-interrogation steelman and then weakened: architectural choices (residuals, normalization, attention) DO change behavior, but they do so by changing *which kernel* the network starts from — not as an independent feature-learning mechanism on top of parameterization choice.

---

## 16. References

### Primary sources cited

#### NTK and lazy regime
- Jacot, A., Gabriel, F., & Hongler, C. (2018). *Neural Tangent Kernel: Convergence and Generalization in Neural Networks*. NeurIPS 2018. arXiv:1806.07572.
- Lee, J., Xiao, L., Schoenholz, S. S., Bahri, Y., Sohl-Dickstein, J., & Pennington, J. (2019). *Wide Neural Networks of Any Depth Evolve as Linear Models Under Gradient Descent*. NeurIPS 2019.
- Arora, S., Du, S. S., Hu, W., Li, Z., & Wang, R. (2019). *Fine-Grained Analysis of Optimization and Generalization for Overparameterized Two-Layer Neural Networks*. ICML 2019.
- Arora, S., Du, S. S., Li, Z., Salakhutdinov, R., Wang, R., & Yu, D. (2019). *Exact Computation with an Infinitely Wide Neural Net*. NeurIPS 2019.

#### μP, mean-field, and feature learning
- Yang, G., & Hu, E. J. (2021). *Feature Learning in Infinite-Width Neural Networks (Tensor Programs IV)*. ICML 2021. arXiv:2011.14522.
- Yang, G. (2020). *Tensor Programs III: Neural Matrix Laws*. arXiv:2009.10685.
- Bordelon, B., & Pehlevan, C. (2022). *Self-Consistent Dynamical Field Theory of Kernel Evolution in Wide Neural Networks*. NeurIPS 2022.
- Chizat, L., Oyallon, E., & Bach, F. (2019). *On Lazy Training in Differentiable Programming*. NeurIPS 2019.
- Mei, S., Montanari, A., & Nguyen, P.-M. (2018). *A Mean Field View of the Landscape of Two-Layer Neural Networks*. PNAS.
- Rotskoff, G. M., & Vanden-Eijnden, E. (2018). *Trainability and Accuracy of Neural Networks: An Interacting Particle System Approach*. arXiv:1805.00915.

#### Provable separations
- Allen-Zhu, Z., Li, Y., & Liang, Y. (2019). *Learning and Generalization in Overparameterized Neural Networks, Going Beyond Two Layers*. NeurIPS 2019.
- Allen-Zhu, Z., & Li, Y. (2019). *What Can ResNet Learn Efficiently, Going Beyond Kernels?* NeurIPS 2019. arXiv:1905.10337.
- Bach, F. (2017). *Breaking the Curse of Dimensionality with Convex Neural Networks*. JMLR.
- Ghorbani, B., Mei, S., Misiakiewicz, T., & Montanari, A. (2020). *When do neural networks outperform kernel methods?* NeurIPS 2020.
- Mei, S., Misiakiewicz, T., & Montanari, A. (2021). *Generalization error of random features and kernel methods: hypercontractivity and kernel matrix concentration*. Applied and Computational Harmonic Analysis.
- Telgarsky, M. (2016). *Benefits of Depth in Neural Networks*. COLT 2016.
- Eldan, R., & Shamir, O. (2016). *The Power of Depth for Feedforward Neural Networks*. COLT 2016.
- Ben Arous, G., Gheissari, R., & Jagannath, A. (2021). *Online Stochastic Gradient Descent on Non-Convex Losses from High-Dimensional Inference*. JMLR.

#### Empirical NTK drift
- Fort, S., Dziugaite, G. K., Paul, M., Kornblith, S., Roy, D. M., & Ganguli, S. (2020). *Deep learning versus kernel learning: an empirical study of loss landscape geometry and the time evolution of the Neural Tangent Kernel*. NeurIPS 2020.
- Geiger, M., Spigler, S., Jacot, A., & Wyart, M. (2020). *Disentangling feature and lazy training in deep neural networks*. Journal of Statistical Mechanics.
- Lee, J., Schoenholz, S., Pennington, J., Adlam, B., Xiao, L., Novak, R., & Sohl-Dickstein, J. (2020). *Finite versus infinite neural networks: an empirical study*. NeurIPS 2020.
- Adlam, B., & Pennington, J. (2020). *Understanding Double Descent Requires a Fine-Grained Bias-Variance Decomposition*. NeurIPS 2020.

#### HTSR and SETOL
- Martin, C. H., & Mahoney, M. W. (2021). *Implicit self-regularization in deep neural networks: Evidence from random matrix theory and implications for learning*. JMLR 22(165):1−73.
- Martin, C. H., & Hinrichs, C. (2025). *SETOL: A Semi-Empirical Theory of (Deep) Learning*. arXiv:2507.17912.
- Pennington, J., & Worah, P. (2017). *Nonlinear random matrix theory for deep learning*. NeurIPS 2017.
- Marchenko, V. A., & Pastur, L. A. (1967). *Distribution of eigenvalues for some sets of random matrices*. Mathematics of the USSR-Sbornik.

#### Mechanistic interpretability
- Olsson, C., Elhage, N., Nanda, N., Joseph, N., et al. (2022). *In-context Learning and Induction Heads*. Anthropic / Transformer Circuits Thread.
- Bricken, T., Templeton, A., Batson, J., Chen, B., et al. (2023). *Towards Monosemanticity: Decomposing Language Models with Dictionary Learning*. Anthropic / Transformer Circuits Thread.
- Templeton, A., Conerly, T., Marcus, J., Lindsey, J., et al. (2024). *Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet*. Anthropic.
- Marks, S., Rager, C., Michaud, E. J., Belinkov, Y., Bau, D., & Mueller, A. (2024). *Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models*. arXiv:2403.19647.
- Gao, L., la Tour, T. D., Tillman, H., Goh, G., Troll, R., Radford, A., Sutskever, I., Leike, J., & Wu, J. (2024). *Scaling and evaluating sparse autoencoders*. arXiv:2406.04093.

#### Adversarial framing (path kernel)
- Domingos, P. (2020). *Every Model Learned by Gradient Descent Is Approximately a Kernel Machine*. arXiv:2012.00152.

#### Scaling and emergence
- Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., & Amodei, D. (2020). *Scaling Laws for Neural Language Models*. arXiv:2001.08361.
- Hoffmann, J., Borgeaud, S., Mensch, A., et al. (2022). *Training Compute-Optimal Large Language Models* (Chinchilla). arXiv:2203.15556.
- Wei, J., Tay, Y., Bommasani, R., et al. (2022). *Emergent Abilities of Large Language Models*. TMLR.
- Schaeffer, R., Miranda, B., & Koyejo, S. (2023). *Are Emergent Abilities of Large Language Models a Mirage?* NeurIPS 2023.

#### Architecture-specific NTK
- Hron, J., Bahri, Y., Sohl-Dickstein, J., & Novak, R. (2020). *Infinite attention: NNGP and NTK for deep attention networks*. ICML 2020.
- Yang, G. (2019). *Wide Feedforward or Recurrent Neural Networks of Any Architecture are Gaussian Processes*. NeurIPS 2019.
- Hayou, S., Doucet, A., & Rousseau, J. (2019). *On the Impact of the Activation Function on Deep Neural Networks Training*. ICML 2019.

### Internal references

- `src/dl_techniques/analyzer/SETOL.md` — Complete technical guide to the SETOL framework as implemented in the dl_techniques analyzer subpackage.
- `src/dl_techniques/analyzer/spectral_metrics.py`, `spectral_analyzer.py`, `spectral_visualizer.py` — production implementation of HTSR/SETOL diagnostics.
- `analyses/analysis_2026-04-19_06e32b37/` — full epistemic deconstruction audit trail: hypotheses, observations, causal graph, validation, summary.

---

*Methodology note: this document is the product of the epistemic-deconstructor protocol, COMPREHENSIVE tier, L4 fidelity. Hypotheses were tracked via Bayesian updating with per-evidence-pair updates (no batched evidence), maintained adversarial framing throughout, and required ≥ 1 disconfirming evidence before any hypothesis exceeded posterior 0.80. The protocol is designed to surface confident conclusions only when they survive disciplined refutation attempts.*
