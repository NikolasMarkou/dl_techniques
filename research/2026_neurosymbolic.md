# Extracting Neurosymbols with Distance-Based Sparse Dictionaries
## A Guide to RBF Prototypes, Sparse Autoencoders, Discrete Codebooks, Object Slots, Rule Extraction

| Field | Value |
|---|---|
| Version | 2026.10 |
| Date | 19 October 2026 |
| Literature cutoff | October 2026 |
| Scope | Turning continuous neural representations into discrete, grounded, testable symbols, and using those symbols in rules and reasoning |
| Out of scope | General LLM reasoning benchmarks except where they inform symbol extraction |

### Evidence tags

| Tag | Meaning |
|---|---|
| **[E]** | Established: replicated or widely adopted |
| **[R]** | Reported: single paper or leaderboard, not independently verified here |
| **[D]** | Design recommendation of this guide: a hypothesis to test, not a published result |

---

## Table of Contents

1. [Executive Summary (October 2026)](#1-executive-summary-october-2026)
2. [Foundations](#2-foundations)
3. [Geometry: RBF Prototypes vs. SAE Encoders](#3-geometry-rbf-prototypes-vs-sae-encoders)
4. [Reference Architecture](#4-reference-architecture)
5. [Extraction Methods](#5-extraction-methods)
6. [From Proto-Symbols to Rules and Reasoning](#6-from-proto-symbols-to-rules-and-reasoning)
7. [Evaluation Protocol](#7-evaluation-protocol)
8. [Domain Playbook](#8-domain-playbook)
9. [Pitfalls and Fixes](#9-pitfalls-and-fixes)
10. [State of the Art Snapshot (October 2026)](#10-state-of-the-art-snapshot-october-2026)
11. [Open Problems](#11-open-problems)
12. [References](#12-references)

---

## 1. Executive Summary (October 2026)

**Position.** Radial basis function (RBF) units remain a sound way to define proto-symbols: each unit is a prototype in a representation space and fires by proximity. Since 2025 this view is no longer niche. Distance-based sparse autoencoders (SpaDE) formalize it and show that encoder geometry determines which concepts can be found at all **[R]**. The RBF framing is therefore best treated as one point in the sparse dictionary design space, chosen when its assumptions (distance-separated concepts, need for locality and an absolute novelty signal) match the data.

**How the field moved between 2024 and October 2026:**

| Area | 2024 to early 2025 belief | October 2026 status |
|---|---|---|
| SAEs as symbol extractors | Monosemantic features at scale are the bridge to symbols | Still the main tool, but features are architecture-dependent, seed-dependent, and subject to absorption and hedging. Linear probes often win on detection. Transcoders and cross-layer transcoders are preferred for mechanisms **[E]** |
| Codebook collapse | Solved by FSQ, SimVQ, MGVQ | Utilization near 100% is routine (IBQ, FVQ). Utilization is not sufficient for quality, and codes are not semantic **[E]** |
| Object-centric learning | DINOSAUR scales to real images | Feature-reconstruction slots remain the base. Gains now come from contrastive temporal objectives, curricula, and diffusion-guided alignment **[R]** |
| Rule extraction | Sparse concept layers plus rule induction would directly translate networks into logic | NeSyViT extends the NeSyFOLD line from CNNs to ViTs. The bottleneck is concept grounding, not rule induction **[E]** |
| NeSy reliability | Secondary concern behind accuracy | Reasoning shortcuts are the dominant documented failure: correct outputs from wrongly grounded concepts **[E]** |
| Formal reasoning benchmarks | miniF2F ~57%, PutnamBench nascent | miniF2F saturated (99.6%); PutnamBench near-saturated by LLM plus Lean agents **[R]** |

**Recommendations.**

1. Never extract symbols from raw inputs. Use frozen or lightly finetuned foundation features (DINOv2/v3, CLIP-family, LLM residual stream, transcoder latents). **[E]**
2. Choose the dictionary by the geometry you expect (Section 3). Use a distance-based encoder when concepts are nonlinearly separable or multi-dimensional, or when you need OOD abstention. **[R]/[D]**
3. Keep an unnormalized novelty score next to any normalized activation. **[D]**
4. Treat every extracted symbol as a hypothesis until it passes precision/recall against concept labels, a seed-stability check, a causal intervention, and a reasoning-shortcut check. **[E]**
5. Use rule learners on calibrated binarized activations and report rule fidelity to the neural model, not only task accuracy. **[E]**

---

## 2. Foundations

### 2.1 Operational definition

A **neurosymbol** is a discrete unit *s* that satisfies five testable criteria:

| Criterion | Test |
|---|---|
| Grounding | A function g_s: X → [0, 1] computed from a neural representation |
| Semantic validity | Label precision and recall against held-out concept annotations exceed a stated bar |
| Stability | The unit is recovered across seeds or data resamples, or a documented matching exists |
| Causal relevance | Ablating or steering the unit changes downstream behavior in the predicted direction |
| Symbolic usability | The unit can serve as a predicate in a rule language and the rules remain faithful to the network |

A **proto-symbol** satisfies grounding and partial semantic validity only. Most "symbols" in the literature are proto-symbols.

### 2.2 RBF unit

```
phi_i(x) = exp(-gamma_i * ||x - c_i||^2)

x        representation vector (never raw input)
c_i      center (prototype) of unit i
gamma_i  inverse width; receptive radius ~ 1/sqrt(gamma_i)
phi_i    in (0, 1]; unnormalized, so it decays to 0 far from all centers
```

Useful identity (same cost as an SAE encoder matmul):

```
||x - c||^2 = ||x||^2 + ||c||^2 - 2 x.c
```

### 2.3 Assignment modes

| Mode | Definition | Use |
|---|---|---|
| Hard | i* = argmin_i gamma_i·‖x − c_i‖², with abstention if min distance exceeds a calibrated radius | Single-label symbol |
| Sparse | Active set A(x) = {i : phi_i(x) > tau_i}, tau_i calibrated per unit | Multi-concept description |
| Normalized | softmax or sparsemax over −gamma_i·d_i² | Mixture weights only; never for novelty (Section 3.2) |

---

## 3. Geometry: RBF Prototypes vs. SAE Encoders

### 3.1 Receptive fields determine what can be found

Hindupur et al. (2025) recast SAE encoders as projections onto architecture-specific sets. Each encoder implies a dual assumption about how concepts are laid out in the representation; concepts violating that assumption are systematically missed. **[R]**

| Encoder | Receptive field of a unit | Adaptive sparsity per input | Absolute novelty signal | Implicit data assumption |
|---|---|---|---|---|
| ReLU / JumpReLU SAE | Half-space | Yes | No (unbounded) | Concepts linearly separable |
| TopK SAE | Union of hyperpyramids | No (fixed k) | No | Concepts angularly separable, equal dimensionality |
| BatchTopK SAE | As TopK | Partial (fixed per batch) | No | As TopK |
| Matryoshka SAE | As base encoder | As base | No | Nested coarse-to-fine hierarchy |
| SpaDE (sparsemax over −λ·d²) | Union of convex polytopes local to prototypes | Yes | No (simplex-normalized) | Concepts separated by Euclidean distance |
| Gaussian RBF + TopK, unnormalized | Local regions around prototypes | No | Yes | Distance separation, isotropic per unit **[D]** |
| Gaussian RBF + per-unit learned threshold | Balls around prototypes | Yes | Yes | Distance separation **[D]** |

In controlled tests, ReLU and JumpReLU missed nonlinearly separable concepts, TopK failed on concepts of heterogeneous intrinsic dimension, and SpaDE recovered both; on DINOv2 features SpaDE produced the most monosemantic latents of the four **[R]**. SpaDE is exactly piecewise linear: it equals a ReLU SAE whose weights and bias are piecewise constant functions of the input, defined relative to a local origin set by the active prototypes **[R]**.

### 3.2 Four facts that constrain RBF designs

1. **Sphere equivalence.** With ‖x‖ = ‖c‖ = 1, ‖x − c‖² = 2 − 2cos(x, c). TopK over Gaussian RBF then selects the same units as a bias-free TopK SAE with unit-norm encoder rows. Locality only buys something if magnitude is preserved. Whiten features (zero mean, identity covariance on a sample) rather than projecting to the sphere, or keep ‖x‖ as an explicit extra channel. **[E]** for the identity, **[D]** for the remedy.
2. **Normalization removes novelty.** Softmax and sparsemax outputs sum to 1 for every input. A far-away input still receives full activation mass. Compute novelty as min_i d_i² (or max_i of unnormalized phi_i) before any normalization. **[E]**
3. **Cost parity, numeric hazard.** Distance computation costs one matmul plus two norm vectors, the same order as an SAE encoder. The hazard is underflow: exp(−gamma·d²) vanishes for most units in high dimension, starving gradients. Initialize gamma_i ≈ 1/median(d²) of each center's nearest data points and work in log space for TopK selection. **[D]**
4. **Prototypes can drift off-manifold.** Learned dictionary directions may lie outside the data distribution, a suspected cause of SAE instability. Two published remedies: a distance-weighted L1 penalty Σ z_i·‖x − c_i‖² that pulls active prototypes toward their data (K-Deep Simplex, used by SpaDE), and archetypal constraints keeping atoms in the convex hull of data (Archetypal SAE). **[R]**

### 3.3 Selection rule

| Requirement | Choose |
|---|---|
| Maximum reconstruction per active latent, generative steering | TopK / BatchTopK / JumpReLU SAE |
| Hierarchical concepts, reduced absorption | Matryoshka SAE (accept hedging) |
| Nonlinearly separable or multi-dimensional concepts | SpaDE |
| OOD abstention plus concept discovery | SpaDE or Gaussian RBF, each with a separate unnormalized novelty score **[D]** |
| Features that explain computation (circuits) | Transcoders / cross-layer transcoders |
| One token per position for generation | FSQ, IBQ, FVQ, SimVQ family (not a concept extractor) |

---

## 4. Reference Architecture

### 4.1 Pipeline

```
input
  -> frozen encoder f            (DINOv2/v3 patches, CLIP, LLM residual stream, or transcoder latents)
  -> h in R^d
  -> preprocess                  (center + whiten on a fitting sample; keep ||h|| as a channel if normalizing)
  -> distance-sparse encoder     (N prototypes c_i, widths gamma_i)
        z    : sparse activations (TopK-Gaussian or sparsemax over -gamma*d^2)
        d2   : squared distances to all prototypes
  -> linear decoder              h_hat = z @ D + b
  -> symbol interface            active set {(i, z_i)}, novelty = min_i d2_i
  -> calibrated binarization     per-unit thresholds from held-out quantiles
  -> rule learner / NeSy layer   FOLD-SE-M, CART/RIPPER, DeepProbLog, Scallop, NeSyDM
```

### 4.2 Objective

| Term | Form | Purpose |
|---|---|---|
| Reconstruction | ‖h − h_hat‖² | Faithfulness |
| Locality (SpaDE/KDS) | λ_loc · Σ_i z_i · d_i² | Pull active prototypes onto the data, sparsify |
| Dead-unit revival | AuxK on residual, or periodic resampling to high-error inputs | Utilization |
| Optional hierarchy | Matryoshka nested reconstruction losses over prefix sets | Coarse-to-fine symbols, less absorption |
| Optional supervision | Concept loss on a small labeled subset | Grounding, reasoning-shortcut mitigation (Section 6.4) |

### 4.3 Starting hyperparameters [D]

| Parameter | Start | Sweep |
|---|---|---|
| Expansion N/d | 8x | 4x to 64x |
| k (TopK) | 32 | 8 to 128 |
| Center init | Random data points or k-means++ on a 1M-sample subset | |
| gamma_i init | 1 / median nearest-center d² | Learned, log-parameterized |
| Dead unit | Not active in last 10M tokens/patches | Resample to inputs with highest reconstruction error |
| Decoder rows | Unit norm, tied init to centers then untied | |
| Seeds | 3 minimum | Report cross-seed matching |

### 4.4 Minimal implementation

```python
import torch
import torch.nn as nn
from entmax import sparsemax  # pip install entmax

class DistanceSparseAE(nn.Module):
    """Prototype (RBF) sparse autoencoder.
    mode="topk": unnormalized Gaussian + TopK (keeps absolute novelty).
    mode="sparsemax": SpaDE-style simplex projection (adaptive sparsity, no novelty).
    """
    def __init__(self, d_in, n_protos, k=32, mode="topk"):
        super().__init__()
        self.C = nn.Parameter(torch.randn(n_protos, d_in) * 0.02)   # prototypes
        self.log_gamma = nn.Parameter(torch.zeros(n_protos))         # per-unit inverse width
        self.D = nn.Parameter(self.C.detach().clone())               # decoder atoms (tied init)
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.k, self.mode = k, mode

    def sq_dist(self, x):
        d2 = x.pow(2).sum(-1, keepdim=True) + self.C.pow(2).sum(-1) - 2 * x @ self.C.T
        return d2.clamp_min(0.0)

    def forward(self, x):
        d2 = self.sq_dist(x)                                  # [B, N]
        logits = -self.log_gamma.exp() * d2                   # log of Gaussian activation
        if self.mode == "topk":
            top_logit, idx = logits.topk(self.k, dim=-1)      # select in log space (no underflow)
            z = torch.zeros_like(logits).scatter(-1, idx, top_logit.exp())
        else:
            z = sparsemax(logits, dim=-1)
        h_hat = z @ self.D + self.b_dec
        novelty = d2.min(dim=-1).values                       # compute before any normalization
        return h_hat, z, d2, novelty

def loss_fn(h, h_hat, z, d2, lam_loc=1e-3):
    recon = (h - h_hat).pow(2).sum(-1).mean()
    locality = (z * d2).sum(-1).mean()                        # distance-weighted L1 (KDS)
    return recon + lam_loc * locality
```

Notes: add AuxK or resampling for dead units; normalize decoder rows after each step; calibrate novelty thresholds on held-out in-distribution data (conformal quantile) before using them for abstention.

---

## 5. Extraction Methods

### 5.1 Decision table

| Goal | Method | Output | Main risk |
|---|---|---|---|
| Small labeled concept set, classifier-style | M1 Prototype catalog | One prototype per concept, hard assignment | Prototypes are averages, not concepts |
| Open-ended concept discovery in a foundation model | M2 Sparse dictionary (SAE family, incl. distance-based) | Thousands to millions of candidate features | Absorption, seed instability, label illusions |
| Symbols that explain computation | M3 Transcoders, CLTs, attribution graphs | Feature-to-feature causal graphs per prompt | Replacement-model error; graphs are prompt-local |
| Discrete tokens for generation or compression | M4 Codebooks (FSQ, IBQ, FVQ, SimVQ, BSQ, Leech) | One code per position | Codes lack semantics |
| Object-level symbols in images and video | M5 Slots on SSL features | K object slots plus masks | Over/under-segmentation |
| Relations, composites, temporal motifs, hierarchy | M6 Structure mining on symbol activations | Graphs, grammars, automata | Correlation mistaken for composition |
| Shared symbols across models or modalities | M7 Cross-model alignment | Aligned dictionaries | Alignment of non-canonical units |

### 5.2 M1: Prototype catalog

1. Embed data with a foundation encoder; whiten.
2. Initialize one or a few prototypes per class (class means or k-means within class).
3. Train with task loss (cross-entropy over −gamma·d²) plus a small locality term.
4. Label = class; validate with per-prototype precision/recall and abstention rate on OOD data.

Use when the vocabulary is known and small. Equivalent in spirit to prototype networks. Never fit prototypes on raw pixels or characters: Euclidean distance there is not semantic.

### 5.3 M2: Sparse dictionary on foundation features

1. Collect activations (patch tokens, residual stream at a chosen layer). 100M+ vectors for LLMs; tens of millions of patches for vision.
2. Train the encoder chosen in Section 3.3 at 2 to 3 widths and 3 seeds.
3. Describe each feature: sample activating inputs across activation quantiles (not only the top), generate a description with an LLM/VLM, then **score** the description by detection precision and recall on held-out inputs.
4. Keep features passing the bar in Section 7; mark the rest as unlabeled proto-symbols.
5. Match features across seeds (Hungarian matching on decoder cosine); report the stable fraction. Expect individual features to vary while spanned subspaces are more reproducible **[R]**.

Known failure modes **[E]**: feature absorption (a general latent silently loses cases to more specific ones), feature hedging in narrow or Matryoshka SAEs (correlated features blended), and dataset dependence.

### 5.4 M3: Mechanistic symbols

Transcoders reconstruct a component's output from its input; cross-layer transcoders replace all MLPs and allow attribution graphs: pruned, prompt-specific graphs of feature-to-feature influence validated by interventions **[E]**. Transcoder features were found significantly more interpretable than SAE features trained on the same model and data **[R]**. In 2026 the method was extended to diffusion transformers (FLUX.1), with transcoders matching or slightly beating SAEs on the sparsity-faithfulness trade-off **[R]**.

Use M3 when the symbol must participate in an explanation of *how* the model computes, not only *what* it represents.

### 5.5 M4: Discrete codebooks

Use for tokenization, compression, and generation. Do not read codes as concepts. Selection:

| Need | Choice |
|---|---|
| Simplest, collapse-free | FSQ |
| Large learned codebook, high utilization | IBQ, FVQ, SimVQ |
| Strong reconstruction from a continuous VAE | CODA-style conversion |
| Geometric lattice codebook | Spherical Leech quantization |

To obtain concepts from a tokenizer, train an M2 dictionary on its continuous latents.

### 5.6 M5: Object slots

Slot Attention iteratively assigns input features to K slots by attention normalized over slots (competitive soft clustering). Reconstructing **self-supervised features** (DINO/DINOv2) instead of pixels made it work on real images (DINOSAUR) **[E]**. Current recipe:

1. Frozen or object-centrically finetuned DINOv2/v3 backbone.
2. Slot Attention with feature-reconstruction decoder (MLP or transformer).
3. Video: add slot contrastive loss across time for temporal consistency.
4. Per-slot symbols: run M2 or M1 on slot vectors to obtain object-type and attribute symbols.
5. Relations: pairwise slot features into a relation classifier or scene-graph head.

### 5.7 M6: Structure mining

| Structure | Procedure | Validation |
|---|---|---|
| Composite symbol | Co-activation statistics propose candidates | Intervention: ablating a part changes the composite as predicted |
| Hierarchy | Matryoshka levels, or agglomerative clustering of decoder directions | Parent-child precision; absorption metrics |
| Temporal motif | Mine frequent sequences of hard symbols; fit HMM or automaton | Held-out likelihood; motif detection F1 |
| Relation | Pairwise features over slots or tokens | Relation accuracy on annotated graphs |
| Circuit | Attribution graph over transcoder features | Steering experiments |

### 5.8 M7: Cross-model and multimodal alignment

Train a shared dictionary across models or modalities (universal SAEs), or align separately trained dictionaries with paired data or contrastive objectives. Because units are not canonical, align subspaces or matched clusters, not single latents **[R]/[D]**.

---

## 6. From Proto-Symbols to Rules and Reasoning

### 6.1 Binarization

| Step | Rule |
|---|---|
| Threshold | Per unit, from a held-out activation quantile or by maximizing label F1 on a labeled subset |
| Abstention | If novelty exceeds its calibrated quantile, emit `unknown` instead of symbols |
| Sparsity check | Report mean active symbols per input; rule learners degrade with dense binary inputs |

### 6.2 Rule induction

| Learner | Output | Note |
|---|---|---|
| FOLD-SE-M (Wang and Gupta) | Stratified answer set program: default rules with exceptions via negation as failure | Used by NeSyFOLD (CNNs) and NeSyViT (ViTs, TPLP 2025) |
| CART / RIPPER | Trees / rule lists | Mandatory baselines |
| DFORL, δILP variants | First-order Horn clauses | Relational data, predicate invention |

Example of the target form (illustrative, predicate names assigned after labeling):

```prolog
label(X, stop_sign) :- c412_red_octagon(X), c087_white_text(X), not ab1(X).
ab1(X) :- c903_reflection_artifact(X).
```

Report: rule fidelity (agreement with the neural classifier), accuracy, rule count, average body length, and the fraction of predicates with a validated label.

A notable pattern from NeSyViT: replace the classification head with a sparse binarized concept layer and let the induced rule set act as the decision layer, so rules are executable rather than post hoc **[R]**.

### 6.3 Differentiable reasoning layers

| Family | Representative | Role |
|---|---|---|
| Fuzzy first-order logic | Logic Tensor Networks | Constraints as differentiable losses |
| Probabilistic logic programming | DeepProbLog, Scallop | Exact or approximate inference over neural predicates |
| Discrete diffusion over concepts | Neurosymbolic Diffusion Models (NeSyDM) | Models dependencies between symbols instead of assuming independence; better concept calibration **[R]** |
| Logic gate networks | Convolutional / Light DLGNs | Learn Boolean circuits directly; nanosecond-scale FPGA inference **[R]** |

### 6.4 Reasoning shortcuts

A NeSy model can satisfy the logic and predict labels correctly while mapping inputs to the **wrong** concepts. This is now the best-documented NeSy failure mode, surveyed in JAIR (2026) **[E]**. Mitigations:

1. Concept supervision on a small labeled subset.
2. Multi-task or richer constraints that make the intended grounding the only consistent one; 2026 work frames detection as constraint satisfaction **[R]**.
3. Models that represent concept uncertainty and dependencies (NeSyDM) to expose ambiguity.
4. Evaluate concept quality directly (rsbench metrics), not only label accuracy.

### 6.5 Composition

Composition is a claim about causal structure. Accept "A ∘ B = C" only if interventions on A or B produce the predicted change in C, or an attribution graph shows the path. Co-activation alone proposes candidates.

---

## 7. Evaluation Protocol

### 7.1 Metrics

| Dimension | Metric | Baseline that must be reported |
|---|---|---|
| Faithfulness | Fraction of variance unexplained; downstream loss or accuracy when the reconstruction replaces the original activation | Identity (0 loss delta), mean ablation |
| Sparsity | Mean L0 per input | |
| Utilization | Dead fraction; activation-frequency entropy | |
| Semantic validity | Per-feature precision, recall, F1 vs. concept annotations across the activation range | Linear probe on raw features for the same concept |
| Absorption / hedging | SAEBench absorption and sparse-probing metrics | Matryoshka variant |
| Description quality | Detection F1 of generated labels on held-out inputs | Same pipeline on a randomly initialized encoder (auto-interp scores alone do not separate trained from random transformers **[R]**) |
| Stability | Cross-seed match rate (decoder cosine, Hungarian matching); subspace overlap | |
| Causal relevance | Effect of ablation and steering on downstream outputs | Random-feature ablation |
| Novelty | OOD AUROC; abstention rate at fixed in-distribution coverage | kNN distance on raw features |
| Rules | Fidelity to the neural model, accuracy, rule count, body length | CART, RIPPER |
| NeSy grounding | Concept accuracy and concept collapse (rsbench-style) | Label accuracy alone is insufficient |

Note: SAE latents underperform simple linear probes on many detection tasks, in and out of domain **[E]**. A symbol that is worse than a probe for its own concept should not be promoted to a predicate.

### 7.2 Acceptance checklist [D]

```
[ ] Inputs are foundation-model features, preprocessing documented
[ ] Encoder choice justified against Section 3.3
[ ] >= 3 seeds; stable-feature fraction reported
[ ] Each promoted symbol: label F1 >= 0.8 on held-out annotations, and within 0.05 of a linear probe
[ ] Each promoted symbol: ablation/steering effect in the predicted direction
[ ] Novelty score computed before normalization; thresholds conformally calibrated
[ ] Rule fidelity >= 0.95 to the neural model on held-out data; size reported
[ ] Reasoning-shortcut check: concept accuracy measured, not inferred from label accuracy
[ ] All numeric claims tagged [E]/[R]/[D]
```

---

## 8. Domain Playbook

| Domain | Representation | Symbol method | Reasoning layer | Notes |
|---|---|---|---|---|
| Images | DINOv2/v3 patches, CLIP | M5 slots for objects; M2 (SpaDE) for parts and attributes | NeSyViT-style sparse concept head + FOLD-SE-M | Label with VLM descriptions, score by detection F1 |
| Video | DINOv2 features over time | M5 with slot contrastive loss; M6 temporal motifs | Automata over slot-symbol sequences | Track slot identity before mining motifs |
| LLM internals | Residual stream, MLP I/O | M3 transcoders/CLTs for mechanisms; M2 for monitoring | Attribution graphs; probes for detection | Compare every feature monitor against a probe |
| Robotics / VLA | VLA residual stream; object slots | M2 features (motion primitives, task progress reported in 2026 **[R]**); M5 task-aware slots | Planner over symbolic states | Slot representations improved manipulation success over frozen DINOv2 in 2026 work **[R]** |
| Time series / sensors | Self-supervised sequence encoder | M1/M2 prototypes on windows; M6 motifs | HMM or automata | Window length is a design variable, sweep it |
| Anomaly detection | Encoder trained on normal data | Unnormalized RBF distance or kNN distance | Abstain / alert | Conformal thresholds; never softmax confidence |
| Scientific discovery | Learned low-dimensional variables | M1 prototypes for regimes | Symbolic regression (PySR, SINDy) for equations | RBF nets approximate functions; they do not emit formulas |
| Generation / compression | Tokenizer latents | M4 codebooks | Autoregressive or masked models | Train M2 on continuous latents for concepts |

---

## 9. Pitfalls and Fixes

| Pitfall | Symptom | Fix |
|---|---|---|
| Center collapse | Many prototypes coincide; low diversity | Data-point init, distance-weighted sparsity, resampling **[E]** |
| Dead units | Units never active | AuxK or resampling to high-error inputs; gamma init from data **[E]** |
| Gradient starvation in high dimension | exp(−gamma·d²) underflows | TopK in log space; whitening; data-based gamma init **[D]** |
| Feature absorption | Interpretable latent with unexplained recall holes | Matryoshka; measure absorption; report recall **[E]** |
| Feature hedging | Correlated concepts merged in narrow dictionaries | Wider dictionaries; tune Matryoshka level weights **[R]** |
| Seed instability | Different features per run | Multiple seeds; promote only matched features; reason over subspaces **[E]** |
| Normalized-confidence illusion | OOD inputs receive full activation mass | Separate unnormalized novelty score (Section 3.2) **[E]** |
| Sphere collapse of locality | RBF behaves like a TopK SAE | Whiten; keep the norm channel (Section 3.2) **[E]** |
| Label illusion | Descriptions fit top examples only | Score across activation quantiles; random-encoder control **[E]** |
| Reasoning shortcuts | High label accuracy, wrong concepts | Concept supervision subset; constraint analysis; NeSyDM; rsbench **[E]** |
| Correlation as composition | Spurious composite symbols | Intervention tests (Section 6.5) **[E]** |
| Threshold arbitrariness | Coverage numbers change with a constant | Per-unit held-out quantiles; conformal calibration **[E]** |
| Utilization as a quality proxy | 100% code usage, poor reconstruction | Report rFID/LPIPS at matched compression **[R]** |
| Overclaiming | "Solved", "first", "production-ready" without benchmarks | Evidence tags; baselines; matched comparisons |

---

## 10. State of the Art Snapshot (October 2026)

### 10.1 Sparse dictionaries and mechanistic interpretability

| Date | Development | Tag |
|---|---|---|
| May 2024 | Scaling Monosemanticity: 34M SAE features on Claude 3 Sonnet, multimodal and abstract features | [E] |
| Jun 2024 | TopK SAEs to 16M latents on GPT-4; AuxK for dead latents; scaling laws | [E] |
| Aug 2024 | Gemma Scope: open JumpReLU SAEs on every layer of Gemma 2 | [E] |
| Dec 2024 to Mar 2025 | BatchTopK and Matryoshka SAEs | [E] |
| Jan to Feb 2025 | Transcoders more interpretable than SAEs; SAEs differ across seeds; SAE latents are not canonical units | [R]/[E] |
| Mar 2025 | Google DeepMind deprioritizes SAE research after SAEs underperform linear probes on downstream safety tasks | [E] |
| Mar 2025 | Circuit tracing: cross-layer transcoders and attribution graphs on a production LLM | [E] |
| Mar 2025 | SpaDE and the SAE/concept-geometry duality | [R] |
| 2025 | SAEBench (ICML 2025); feature absorption found in every LLM SAE tested (NeurIPS 2025); feature hedging in narrow and Matryoshka SAEs | [E]/[R] |
| 2026 | Transcoders and attribution graphs extended to diffusion transformers (FLUX.1) | [R] |
| 2026 | SAEs on vision-language-action models expose motion-primitive and task-structure features | [R] |
| 2026 | Seed dependence: individual features unstable, spanned subspaces reproducible | [R] |

**Implication.** Treat dictionary features as candidate symbols with measured reliability, prefer transcoders for mechanistic claims, and keep probes as the detection baseline.

### 10.2 Discrete representations

| Method | Venue | Reported result (ImageNet 256x256 reconstruction) | Tag |
|---|---|---|---|
| FSQ | ICLR 2024 | Implicit product codebook, full utilization by construction | [E] |
| SimVQ | arXiv 2024 | Codebook as linear transform of a latent basis; ~100% utilization | [R] |
| Rotation trick | ICLR 2025 | Better gradient estimator through VQ | [R] |
| IBQ | ICCV 2025 | 262,144 codes, 256-dim, rFID 1.00, 84% usage | [R] |
| CODA | ICCV 2025 | Continuous VAE converted to discrete; rFID 0.43 (8x), 1.34 (16x); 100% usage | [R] |
| FVQ | arXiv 2025 | 100% usage; rFID 0.88 at 262,144 codes | [R] |
| MGVQ | arXiv 2025 | Reported rFID 0.49 with nested sub-codebooks | [R] |
| Spherical Leech quantization | CVPR 2026 | rFID 1.14 to 0.83 vs. BSQ at similar codebook size | [R] |
| LGQ | arXiv 2026 | Shows forced full utilization gives diminishing returns; learns discretization geometry | [R] |

**Implication.** Collapse is an engineering-solved problem for tokenizers. Semantic symbols still require a dictionary or supervision on top.

### 10.3 Object-centric learning

| Method | Venue | Result | Tag |
|---|---|---|---|
| DINOSAUR | ICLR 2023 | Feature reconstruction; MOVi-C FG-ARI 68.6; COCO about 40 | [E] |
| Object-centric finetuning (FT-DINOSAUR) | 2024 | Finetuning the backbone improves COCO FG-ARI; strong zero-shot transfer | [R] |
| SlotContrast | CVPR 2025 | Slot contrastive loss; +6.8 FG-ARI and +4 mBO over VideoSAUR on YouTube-VIS | [R] |
| CODA (slots) | ICLR 2026 | +6.1 FG-ARI over the strongest baseline on COCO | [R] |
| SlotCurri | CVPR 2026 | Reconstruction-guided slot curriculum; reduces over-fragmentation in video | [R] |
| STORM | 2026 | Task-aware slots raise real-robot in-distribution success to 55.2% vs. 22.0% for frozen DINOv2 | [R] |

### 10.4 Neurosymbolic reasoning and rule extraction

| Development | Venue | Tag |
|---|---|---|
| NeSyFOLD: rules from CNN kernels via FOLD-SE-M with semantic predicate labeling | AAAI 2024 | [R] |
| rsbench: concept quality and reasoning-shortcut benchmark | NeurIPS 2024 D&B | [E] |
| Convolutional DLGNs: 86.29% CIFAR-10 with 61M gates, 29x smaller than prior SOTA | NeurIPS 2024 Oral | [R] |
| NeSyViT: sparse binarized concept layer + FOLD-SE-M as an executable decision layer for ViTs | TPLP 2025 | [R] |
| Neurosymbolic Diffusion Models: discrete diffusion over concepts, better calibration and shortcut awareness | arXiv 2025 | [R] |
| Light DLGNs: reparameterization for deeper logic gate networks | ICLR 2026 | [R] |
| Reasoning Shortcuts survey | JAIR, July 2026 | [E] |
| Constraint-based analysis of reasoning shortcuts | KR 2026 | [R] |
| Position: neuro-symbolic integration as the path to reliable reasoning, against "scaling is all you need" | AAAI 2026 | Position |

### 10.5 Adjacent benchmarks

| Benchmark | Status | Tag |
|---|---|---|
| ARC-AGI-2 (ARC Prize 2025) | Winner NVARC 24.03% on the private set at $0.20 per task (4B model, synthetic data, test-time training). Paper award: Tiny Recursive Model, 7M parameters, 45% on ARC-AGI-1 | [E] |
| ARC-AGI-3 | Launched 25 March 2026; interactive game environments; humans 100%, frontier AI 0.51% at launch; ARC Prize 2026 pool $2M. Kaggle public leaderboard reported above 55% in early October 2026 via agent harnesses; final results after the November 2026 deadline | [R] |
| miniF2F | Effectively saturated (Seed-Prover 99.6%) | [R] |
| PutnamBench (Lean, 672 problems) | Jan 2026 leaderboard: Aleph Prover 668, Seed-Prover 1.5 581, Hilbert 462 | [R] |

**Implication.** The most benchmark-successful NeSy pattern in 2026 is neural generation checked by a symbolic verifier (Lean, program execution). Extracting symbols *from* representations is less mature and is where this guide's methods apply.

---

## 11. Open Problems

| Problem | Why open | Direction |
|---|---|---|
| Canonical units | Dictionaries are non-identifiable; features vary by seed and architecture | Identifiability results from causal representation learning; subspace-level symbols |
| Grounding without labels | Reasoning shortcuts persist whenever multiple groundings satisfy the constraints | Minimal-supervision bounds; constraint design; uncertainty-aware NeSy models |
| Hierarchy | Flat dictionaries; Matryoshka trades absorption for hedging | Explicit taxonomic dictionaries; tree-structured sparsity |
| Multi-dimensional concepts | Circular and manifold features need more than one latent | Subspace or manifold symbols with distance-based membership **[D]** |
| Calibrated abstention | Most pipelines emit symbols for every input | Conformal novelty on unnormalized distances **[D]** |
| Evaluation | Auto-interp scores are weak evidence | Concept-annotated benchmarks that also require reasoning |
| Scale for distance encoders | Exact top-k over 10⁷ prototypes is costly | Approximate nearest-neighbor top-k during training and inference **[D]** |
| Procedural symbols | Most work extracts "what", not "how" | Transcoder features as operators in agents and planners |

---

## 12. References

### Sparse dictionaries and interpretability
- Templeton et al. (2024). Scaling Monosemanticity. Transformer Circuits. https://transformer-circuits.pub/2024/scaling-monosemanticity/
- Gao et al. (2024). Scaling and Evaluating Sparse Autoencoders. arXiv:2406.04093
- Rajamanoharan et al. (2024). JumpReLU SAEs. arXiv:2407.14435
- Lieberum et al. (2024). Gemma Scope. arXiv:2408.05147
- Bussmann, Leask, Nanda (2024). BatchTopK SAEs. arXiv:2412.06410
- Bussmann et al. (2025). Matryoshka SAEs. arXiv:2503.17547
- Dunefsky, Chlenski, Nanda (2024). Transcoders Find Interpretable LLM Feature Circuits. NeurIPS 2024
- Paulo, Shabalin, Belrose (2025). Transcoders Beat Sparse Autoencoders for Interpretability. arXiv:2501.18823
- Paulo, Belrose (2025). SAEs Trained on the Same Data Learn Different Features. arXiv:2501.16615
- Leask et al. (2025). SAEs Do Not Find Canonical Units of Analysis. arXiv:2502.04878
- Chanin et al. (2025). A is for Absorption. NeurIPS 2025. arXiv:2409.14507
- Feature Hedging: Correlated Features Break Narrow SAEs (2025). arXiv:2505.11756
- Karvonen et al. (2025). SAEBench. ICML 2025. arXiv:2503.09532
- Kantamneni et al. (2025). Are SAEs Useful? A Case Study in Sparse Probing. arXiv:2502.16681
- Wu et al. (2025). AxBench. arXiv:2501.17148
- Google DeepMind Safety Research (2025). Negative Results for SAEs on Downstream Tasks. https://deepmindsafetyresearch.medium.com/negative-results-for-sparse-autoencoders-on-downstream-tasks-and-deprioritising-sae-research-6cadcfc125b9
- Ameisen et al. (2025). Circuit Tracing. https://transformer-circuits.pub/2025/attribution-graphs/methods.html
- Lindsey et al. (2025). On the Biology of a Large Language Model. https://transformer-circuits.pub/2025/attribution-graphs/biology.html
- Hindupur, Lubana, Fel, Ba (2025). Projecting Assumptions: The Duality Between SAEs and Concept Geometry (SpaDE). arXiv:2503.01822
- Fel et al. (2025). Archetypal SAE. arXiv:2502.12892
- Thasarathan et al. (2025). Universal Sparse Autoencoders. arXiv:2502.03714
- Tasissa et al. (2023). K-Deep Simplex. IEEE Transactions on Signal Processing
- Automated Interpretability Metrics Do Not Distinguish Trained and Random Transformers (2025). arXiv:2501.17727
- DifFRACT: Transcoders and Circuit Tracing for Diffusion Transformers (2026). arXiv:2606.15796
- Sparse Autoencoders Reveal Interpretable and Steerable Features in VLA Models (2026). arXiv:2603.19183
- Unstable Features, Reproducible Subspaces: Seed Dependence in SAEs (2026). arXiv:2606.12138

### Discrete representations
- Mentzer et al. (2024). Finite Scalar Quantization. ICLR 2024. arXiv:2309.15505
- Zhu et al. (2024). SimVQ. arXiv:2411.02038
- Fifty et al. (2025). Restructuring Vector Quantization with the Rotation Trick. ICLR 2025. arXiv:2410.06424
- Shi et al. (2025). Index Backpropagation Quantization (IBQ). ICCV 2025. arXiv:2412.02692
- Liu et al. (2025). CODA: Repurposing Continuous VAEs for Discrete Tokenization. ICCV 2025
- Scalable Training for Vector-Quantized Networks with 100% Codebook Utilization (FVQ) (2025). arXiv:2509.10140
- Jia et al. (2025). MGVQ. arXiv:2507.07997
- Zhao et al. (2026). Spherical Leech Quantization for Visual Tokenization and Generation. CVPR 2026
- LGQ: Learnable Geometric Quantization for Image Tokenization (2026). arXiv:2602.16086

### Object-centric learning and SSL features
- Seitzer et al. (2023). Bridging the Gap to Real-World Object-Centric Learning (DINOSAUR). ICLR 2023. arXiv:2209.14860
- Zadaianchuk et al. (2023). VideoSAUR. NeurIPS 2023
- Didolkar et al. (2024). Zero-Shot Object-Centric Representation Learning. arXiv:2408.09162
- Manasyan et al. (2025). Temporally Consistent Object-Centric Learning by Contrasting Slots. CVPR 2025. arXiv:2412.14295
- CODA for object-centric learning (2026). ICLR 2026. arXiv:2601.01224
- Moon et al. (2026). Reconstruction-Guided Slot Curriculum. CVPR 2026
- STORM: Slot-based Task-aware Object-centric Representation for Robotic Manipulation (2026). arXiv:2601.20381
- Oquab et al. (2023). DINOv2. arXiv:2304.07193
- Siméoni et al. (2025). DINOv3. arXiv:2508.10104

### Rule extraction and neurosymbolic reasoning
- Wang, Gupta (2024). FOLD-SE: Scalable Explainable Rule-Based Learning. PADL 2024
- Padalkar et al. (2024). NeSyFOLD. AAAI 2024
- Padalkar, Gupta (2025). Symbolic Rule Extraction from Attention-Guided Sparse Representations in Vision Transformers. TPLP 2025. arXiv:2505.06745
- Badreddine et al. (2022). Logic Tensor Networks. Artificial Intelligence 303
- Manhaeve et al. (2018). DeepProbLog. NeurIPS 2018
- Li et al. (2023). Scallop. PLDI 2023
- Bortolotti et al. (2024). rsbench. NeurIPS 2024 Datasets and Benchmarks. arXiv:2406.10368
- van Krieken et al. (2025). Neurosymbolic Diffusion Models. arXiv:2505.13138
- Marconato et al. (2026). Reasoning Shortcuts. JAIR 86, Article 40
- Takemura et al. (2026). Constraint-Based Analysis of Reasoning Shortcuts in Neurosymbolic Learning. KR 2026
- Belle, Marcus (2026). The Future Is Neuro-Symbolic. AAAI 2026. doi:10.1609/aaai.v40i48.42130
- Petersen et al. (2024). Convolutional Differentiable Logic Gate Networks. NeurIPS 2024. arXiv:2411.04732
- Light Differentiable Logic Gate Networks (2026). ICLR 2026

### Benchmarks and context
- ARC Prize Foundation (2026). ARC Prize 2025 Technical Report. arXiv:2601.10904
- ARC Prize Foundation (2026). ARC-AGI-3. arXiv:2603.24621; https://arcprize.org/blog/arc-agi-3-launch
- Jolicoeur-Martineau (2025). Less is More: Recursive Reasoning with Tiny Networks. arXiv:2510.04871
- Chen et al. (2025). Seed-Prover. arXiv:2507.23726
- Varambally et al. (2025). Hilbert. arXiv:2509.22819
- A Minimal Agent for Automated Theorem Proving (2026), PutnamBench leaderboard appendix. arXiv:2602.24273