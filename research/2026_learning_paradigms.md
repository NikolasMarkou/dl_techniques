# Learning Paradigms Through the Lens of Information

*An information-theoretic taxonomy of machine learning methods: what each paradigm compresses, where its information comes from, and when the compression is paid for.*

---

## Table of Contents

1. [Introduction](#1-introduction)
   - 1.1 [Why an information frame](#11-why-an-information-frame)
   - 1.2 [Scope and what this document is not](#12-scope-and-what-this-document-is-not)
2. [Foundations](#2-foundations)
   - 2.1 [Shannon, Kolmogorov, and MDL](#21-shannon-kolmogorov-and-mdl)
   - 2.2 [Why negative log-likelihood is literally bits](#22-why-negative-log-likelihood-is-literally-bits)
   - 2.3 [Bits per sample: the quantity that explains most of the field](#23-bits-per-sample-the-quantity-that-explains-most-of-the-field)
   - 2.4 [Rate-distortion and the information bottleneck](#24-rate-distortion-and-the-information-bottleneck)
   - 2.5 [The three axes of the taxonomy](#25-the-three-axes-of-the-taxonomy)
3. [The Paradigms](#3-the-paradigms)
   - 3.1 [Supervised learning](#31-supervised-learning)
   - 3.2 [Generative self-supervised pretraining](#32-generative-self-supervised-pretraining)
   - 3.3 [Contrastive and discriminative self-supervision](#33-contrastive-and-discriminative-self-supervision)
   - 3.4 [Variational autoencoders](#34-variational-autoencoders)
   - 3.5 [Diffusion models](#35-diffusion-models)
   - 3.6 [Generative adversarial networks](#36-generative-adversarial-networks)
   - 3.7 [Reinforcement learning](#37-reinforcement-learning)
   - 3.8 [Preference learning and RLHF](#38-preference-learning-and-rlhf)
   - 3.9 [Self-play](#39-self-play)
   - 3.10 [Evolutionary and gradient-free methods](#310-evolutionary-and-gradient-free-methods)
   - 3.11 [Knowledge distillation](#311-knowledge-distillation)
   - 3.12 [Meta-learning and in-context learning](#312-meta-learning-and-in-context-learning)
   - 3.13 [Inverse reinforcement learning](#313-inverse-reinforcement-learning)
   - 3.14 [Program synthesis and symbolic regression](#314-program-synthesis-and-symbolic-regression)
   - 3.15 [Bayesian inference](#315-bayesian-inference)
   - 3.16 [Retrieval and non-parametric methods](#316-retrieval-and-non-parametric-methods)
   - 3.17 [Test-time compute and inference-time search](#317-test-time-compute-and-inference-time-search)
   - 3.18 [Active learning and curriculum](#318-active-learning-and-curriculum)
   - 3.19 [Cooperative generative-discriminative frameworks](#319-cooperative-generative-discriminative-frameworks)
4. [Cross-Paradigm Comparison](#4-cross-paradigm-comparison)
5. [The Grid](#5-the-grid)
6. [Worked Example: CIFAR-100](#6-worked-example-cifar-100)
   - 6.1 [The information accounting](#61-the-information-accounting)
   - 6.2 [Directly applicable paradigms](#62-directly-applicable-paradigms)
   - 6.3 [Applicable after modifying the dataset](#63-applicable-after-modifying-the-dataset)
   - 6.4 [Not applicable, and why](#64-not-applicable-and-why)
   - 6.5 [A concrete experimental ladder](#65-a-concrete-experimental-ladder)
7. [Where the Frame Breaks](#7-where-the-frame-breaks)
8. [Practical Consequences](#8-practical-consequences)
9. [References and Further Reading](#9-references-and-further-reading)

---

## 1. Introduction

### 1.1 Why an information frame

Machine learning is usually taught as a catalogue: here is supervised learning, here is unsupervised learning, here is reinforcement learning. The categories are historical accidents as much as anything, and they obscure the fact that several apparently different methods are doing the same thing to different objects, while several apparently similar methods are doing fundamentally different things.

The information-theoretic frame gives a sharper cut. For any learning method, ask three questions:

1. **Where does the information come from?** A fixed corpus, a teacher model, an environment, a verifier, a human, or the learner's own output.
2. **What is the information compressed into?** Weights, a program, a policy, a posterior distribution, or nothing at all (kept raw in an index).
3. **When is the compression paid for?** At training time, or deferred and paid back at inference.

Answer those three and you have located the method precisely. Most claimed "new paradigms" turn out to be a previously empty cell in that grid rather than a genuinely new kind of learning.

The frame also makes quantitative predictions that the catalogue view cannot. It tells you in advance which methods will be sample-efficient and which will not, which can teach a model genuinely new capabilities and which can only elicit capabilities the model already has, and where the scaling bottleneck for a given method will appear.

### 1.2 Scope and what this document is not

This document covers learning paradigms, not architectures. Transformers, convolutional networks and state-space models are all ways of parameterizing a function; they are orthogonal to the question of what objective that function is trained against. A transformer can be trained with any of the paradigms below.

It also does not attempt to be a tutorial. Each section assumes you know roughly what the method is and focuses instead on placing it in the taxonomy, stating what it actually buys you, and noting where the standard story is wrong or oversold.

---

## 2. Foundations

### 2.1 Shannon, Kolmogorov, and MDL

Three notions of "information content" are relevant, and they converge.

**Shannon entropy** is defined relative to a distribution. The entropy `H(X) = -sum p(x) log p(x)` is the expected number of bits needed to encode a sample from `p`, using the optimal code for `p`. It is a property of the distribution, not of any individual object.

**Kolmogorov complexity** `K(x)` is defined relative to a universal Turing machine: the length of the shortest program that outputs `x` and halts. It is a property of the individual object and requires no distribution. It is also uncomputable, which is why it functions as a conceptual target rather than an optimization objective.

**Minimum description length (MDL)** is the practical bridge. Choose the model that minimizes the total code length of model plus data given model:

```
L(M) + L(D | M)
```

The first term penalizes complex models, the second rewards fit. This is a formalization of Occam's razor, and it is equivalent to maximum a posteriori estimation under a prior `p(M) proportional to 2^(-L(M))`. Bayes and MDL are the same object viewed from two directions.

The connection that matters: the Shannon entropy of a source is a lower bound on the expected Kolmogorov complexity of samples from it, up to a constant. A model that compresses data well must have captured real structure, because random data is incompressible. There is no way to cheat at compression.

### 2.2 Why negative log-likelihood is literally bits

This is the point where the frame stops being a metaphor.

Given a probabilistic model `q` and a true data distribution `p`, the expected code length when encoding samples from `p` with a code optimized for `q` is the cross-entropy `H(p, q) = H(p) + KL(p || q)`. Training a model by minimizing negative log-likelihood is exactly minimizing this cross-entropy, which is exactly minimizing expected code length.

The construction is not hypothetical. Given any autoregressive model and an arithmetic coder, you have a working lossless compressor. Feed the model's predicted distribution over the next token to the coder, and the number of bits emitted is the negative log-likelihood in base 2. Large language models are, in a literal and demonstrable sense, state-of-the-art general-purpose compressors. Published results have shown transformer language models compressing text, images and audio below the rates achieved by specialized codecs like PNG and FLAC, despite never being trained on those modalities in a codec sense.

The practical unit is **bits per token** (language), **bits per dimension** (images), or **bits per byte** (cross-modal comparison). When you read that a model achieves 0.7 bits per byte, that is a compression ratio statement.

Two caveats that matter and are usually skipped:

- The model's own parameters are not counted in that figure. A fair MDL accounting includes the cost of transmitting the model. For a large model compressing a small corpus, the model dominates and the "compression" is illusory. For a large model compressing a very large corpus, the model cost amortizes to near zero per token. This is the sense in which pretraining is compression: the corpus is far larger than the weights.
- Lossless compression of the training set is not the goal. Generalization is. A lookup table achieves zero loss on the training set and compresses nothing, because the table is as large as the data.

### 2.3 Bits per sample: the quantity that explains most of the field

The single most useful number for predicting how a learning method will behave is the information content of one training signal.

| Signal type | Approximate bits per sample |
|---|---|
| Next-token prediction (language) | 6 to 20, depending on the model and text |
| Pixel or patch reconstruction | Hundreds to thousands per image |
| Teacher soft labels (K classes) | Up to `log2(K)` plus the full shape of the distribution, in practice several bits |
| Single hard classification label (K classes) | `log2(K)`, so roughly 3.3 bits for 10 classes, 10 bits for 1000 |
| Pairwise human preference | 1 bit |
| Scalar RL reward at episode end | Often well under 1 bit of usable signal after credit assignment |
| Evolutionary fitness comparison | 1 bit per comparison |

This table alone explains most of the sample-efficiency ordering in machine learning. Pretraining works because it extracts twenty bits from every token of a trillion-token corpus and needs no annotation. RL is sample-hungry because a scalar reward at the end of a long episode must be distributed across every action that led there, and the total information injected is tiny.

The consequence for modern practice is direct: pretraining injects an enormous quantity of information into the weights, and reinforcement learning from human feedback injects a negligible quantity by comparison. Tens of thousands of preference comparisons is tens of thousands of bits. A pretraining corpus is on the order of `10^13` bits. RLHF therefore cannot plausibly be teaching new capabilities. It is locating and amplifying modes that the pretrained prior already contains. The honest description is **elicitation, not acquisition**.

### 2.4 Rate-distortion and the information bottleneck

Compression in machine learning is almost always lossy, and the interesting question is what to discard.

**Rate-distortion theory** formalizes the trade-off: for a given tolerated distortion `D`, what is the minimum rate `R(D)` in bits needed to encode the source? Every lossy method sits somewhere on this curve.

The **information bottleneck** is the supervised-learning specialization. Find a representation `Z` of input `X` that minimizes `I(X; Z)` subject to preserving `I(Z; Y)`:

```
minimize  I(X; Z) - beta * I(Z; Y)
```

Read this as: throw away everything about the input that does not help predict the label, and no more. Nearly every representation-learning method in this document can be written as a particular choice of what plays the role of `Y` and how the mutual information terms are bounded or approximated.

- Supervised learning: `Y` is the label.
- Masked prediction: `Y` is the held-out part of `X` itself.
- Contrastive learning: `Y` is "which other view came from the same source".
- VAEs: the two terms appear explicitly as the rate and distortion halves of the ELBO.

### 2.5 The three axes of the taxonomy

To restate the frame that organizes Section 3:

**Axis 1: Information source.** Fixed corpus, teacher model, environment with rewards, formal verifier, human annotator, or the learner's own generations. This axis determines the ceiling on what can be learned and the cost per bit.

**Axis 2: Compression target.** Weights, an explicit program, a policy, a posterior, or an external index. This axis determines interpretability, editability and the failure modes.

**Axis 3: Timing.** Training-time compression (amortized, cheap at inference) versus inference-time search (expensive per query, but adapts to the query). This axis determines the deployment economics and has become the dominant design decision in current systems.

---

## 3. The Paradigms

Each entry follows the same structure: what it optimizes, what is compressed, where the information comes from, why it works, and where it fails.

---

### 3.1 Supervised learning

**Objective.** Minimize expected loss of `f(x)` against label `y` over a labelled dataset. In probabilistic form, minimize the conditional code length `L(Y | X)`.

**What is compressed.** The mapping from inputs to labels. In information-bottleneck terms, the model seeks a minimal sufficient statistic of `X` for `Y`: a representation that retains every bit relevant to the label and discards the rest.

**Information source.** Human annotation, or an existing labelled corpus. This is the binding constraint. Labels are expensive, slow, and finite.

**Why it works.** The objective is dense, the gradient is informative at every example, and the credit assignment problem is trivial because the target is given. Optimization is well-conditioned relative to every other paradigm in this document.

**Where it fails.**

- **Annotation ceiling.** A hard label over `K` classes carries at most `log2(K)` bits. At 1000 classes and one million images, the entire ImageNet label set is roughly ten megabits. That is a startlingly small amount of information to train a large model on, which is precisely why supervised-only training plateaus.
- **Task specificity.** The representation is shaped by the particular `Y`. Discarding everything irrelevant to the label is optimal for that label and harmful for transfer.
- **Shortcut learning.** The bottleneck objective does not distinguish between a causal feature and a spurious correlate, provided both predict `Y` on the training distribution. Compression alone will not prefer the causal one.

**Status.** Still the correct choice when labels are abundant and the task is fixed. As a standalone paradigm for building general systems, superseded.

---

### 3.2 Generative self-supervised pretraining

Covers masked language modelling (BERT), autoregressive language modelling (GPT family), masked image modelling (MAE), and their variants.

**Objective.** Model the data distribution itself. Minimize `-log p(x)` over a corpus, either factorized autoregressively as `sum_t log p(x_t | x_<t)` or as a masked conditional `log p(x_masked | x_visible)`.

**What is compressed.** The joint distribution of the data. Not a task mapping. This is the key structural difference from Section 3.1 and the reason the resulting representations transfer.

**Information source.** The raw corpus. No annotation, which removes the binding constraint of supervised learning entirely. The supervision is manufactured by hiding part of the input from the model and asking it to predict it.

**Why it works.**

- **Information rate.** Every token supplies supervision. At roughly ten to twenty bits per token over a corpus of `10^12` to `10^13` tokens, the total information available exceeds any labelled dataset by many orders of magnitude.
- **The Kolmogorov argument.** To compress data well, a model must internalize the structure that generated it. Compressing English text well requires syntax; compressing it very well requires semantics, world knowledge, and something that behaves like reasoning, because those are what make the next token predictable. There is no shortcut. This is the strongest available theoretical case for why scaling a prediction objective produces general capability.
- **Task-agnostic prior.** Because nothing was discarded to serve a specific `Y`, the representation supports arbitrary downstream tasks.

**Masked versus autoregressive.** Masked objectives see bidirectional context and are historically better for representation extraction; autoregressive objectives factorize the exact joint and are therefore directly usable as generators and as compressors. The field converged on autoregressive largely because generation and in-context learning fall out for free.

**Where it fails.**

- **It buys a prior, not a behaviour.** A pretrained model has capability but no disposition. It will complete text in the style of its corpus, which is not the same as being useful, honest or safe. Every alignment method in this document exists to address the gap.
- **Compression rewards frequency.** Bits are allocated in proportion to how often patterns occur. Rare but important facts are under-served by the objective, which is a structural argument for retrieval (Section 3.16).
- **Data exhaustion.** The information rate is high but the corpus is finite. This is the current scaling constraint.

---

### 3.3 Contrastive and discriminative self-supervision

Covers InfoNCE, SimCLR, MoCo, CLIP, and the non-contrastive variants BYOL and DINO.

**Objective.** Make representations of two augmented views of the same source agree, and representations of different sources disagree. InfoNCE is a lower bound on the mutual information between views.

**What is compressed.** Only the view-invariant content. Everything that changes under the augmentation is deliberately destroyed.

**Information source.** The corpus plus, critically, the augmentation policy. The choice of augmentations is a strong human-supplied inductive bias and is doing much of the work. Crop and colour-jitter invariance encodes an assumption about what "same object" means.

**Why it works.** It is far cheaper than modelling the full joint distribution. Predicting every pixel wastes capacity on high-frequency detail that no downstream task needs. Discriminating between instances targets exactly the semantic content.

**Where it fails.**

- **The augmentation policy is a ceiling.** If colour is destroyed by augmentation, the representation cannot encode colour, and any downstream task requiring colour is lost. The method works well on natural images precisely because the community found good augmentations for natural images, and transfers poorly to domains where the correct invariances are unknown.
- **Weak MI bounds.** InfoNCE's bound on mutual information is upper-bounded by `log(batch size)`. This is a genuinely weak bound at any practical batch size, and it is now understood that the empirical success of these methods is not well explained by the mutual information story they were sold with.
- **No generative capability.** Nothing can be sampled from the model.

**Status.** Dominant in vision before masked autoencoders and large-scale multimodal training. CLIP remains important because its "augmentation" is a caption, which imports language supervision at scale.

---

### 3.4 Variational autoencoders

**Objective.** Maximize the evidence lower bound:

```
ELBO = E_q[log p(x | z)] - KL(q(z | x) || p(z))
```

**What is compressed.** The ELBO decomposes cleanly into a **distortion** term (reconstruction error) and a **rate** term (the KL, which is the number of bits needed to transmit the latent code beyond the prior). This is the only mainstream deep generative model where the compression rate appears as an explicit, tunable term in the objective.

**Information source.** The corpus.

**Why it matters conceptually.** The beta-VAE variant, which reweights the KL term, is a direct dial on the rate-distortion curve. Turning beta up buys a more compressed, more disentangled latent at the cost of reconstruction fidelity. It makes the trade-off in Section 2.4 operational rather than theoretical.

**Where it fails.**

- **Posterior collapse.** With a sufficiently powerful decoder, the optimal solution ignores `z` entirely: the decoder models `p(x)` alone, the KL term goes to zero, and the latent carries no information. This is not a bug in the implementation, it is the objective doing what it was asked to do.
- **Blurry samples.** A Gaussian likelihood in pixel space corresponds to an L2 distortion measure, which averages over modes.

**Status.** Largely superseded as a generative model by diffusion. Survives as a component: the latent space of latent diffusion models is a VAE, and the framework remains the cleanest teaching example of rate-distortion in deep learning.

---

### 3.5 Diffusion models

**Objective.** Learn to reverse a fixed noising process. Training reduces to denoising score matching: predict the noise added at a randomly sampled timestep.

**What is compressed.** The joint distribution, like Section 3.2, but with the crucial difference of **hierarchical bit allocation across noise scales**. High-noise timesteps carry coarse structure and layout; low-noise timesteps carry high-frequency detail. This is progressive coding, and it is the same principle as a wavelet codec or a progressive JPEG.

**Information source.** The corpus.

**Why it works.** Splitting generation across many small denoising steps converts one intractable modelling problem into many tractable ones. Each step is a small, well-conditioned regression. The model never has to produce a complete sample in one forward pass, which is the difficulty that made earlier generative models unstable.

The diffusion ELBO is also a valid variational bound on the log-likelihood, so the compression interpretation is exact rather than analogical.

**Where it fails.**

- Sampling cost is many forward passes, though distillation (Section 3.11) has reduced this to single digits.
- Likelihoods are competitive but not state of the art; autoregressive models remain better pure compressors of discrete data.

---

### 3.6 Generative adversarial networks

**Objective.** A two-player minimax game. The generator minimizes a divergence between its output distribution and the data distribution, as estimated by a discriminator.

**What is compressed.** This is the interesting case: **nothing, in any measurable sense**. There is no likelihood, no bits-per-sample, no code length. The generator is trained by a signal that comes from another learned network rather than from an information-theoretic objective.

**Information source.** The corpus, but mediated entirely through the discriminator. The generator never sees a real data point directly.

**Why it worked.** By not committing to a likelihood, GANs avoided the mode-averaging that produces blurry VAE samples. They could put all their capacity on the modes they did cover. For several years this made them the best image generators by visual quality.

**Why it failed.**

- **No compression means no evaluation.** Without a likelihood you cannot ask how many bits the model assigns to held-out data. The field resorted to proxy metrics (Inception Score, FID) that are known to be gameable and to correlate imperfectly with sample quality. Years of GAN research had no trustworthy yardstick, which is a direct and predictable consequence of abandoning the information frame.
- **Mode collapse.** Nothing in the objective penalizes ignoring parts of the data distribution. A generator that produces one perfect image forever can defeat a weak discriminator.
- **Training instability.** Minimax optimization on non-convex objectives has no convergence guarantee and in practice is fragile.

**Status.** Superseded by diffusion for generation. The lesson generalizes: **paradigms that cannot state what they compress tend to be hard to evaluate, and paradigms that are hard to evaluate tend to stall.**

---

### 3.7 Reinforcement learning

**Objective.** Maximize expected cumulative discounted reward over trajectories generated by the agent's own policy.

**What is compressed.** This is where the compression frame genuinely breaks rather than merely bending. RL is not primarily compression. It is **search and selection**.

Three structural differences from everything above:

| Property | Supervised and self-supervised | Reinforcement learning |
|---|---|---|
| Data source | Fixed, external, i.i.d. | Generated by the current policy, non-stationary, correlated |
| Signal density | Dense, high bits per sample | Scalar, often one delayed number per trajectory |
| Target | Given | Must be discovered through credit assignment |

**Information source.** The environment, via reward. The environment is potentially an infinite source, which is the paradigm's great advantage, but the bandwidth of the reward channel is minuscule, which is its great cost.

**Why the distinction matters.**

- RL does not shrink a description of existing data. It **creates** the data it learns from and keeps what scores well. The nearest honest analogies are stochastic search, hill climbing, and evolutionary selection, not encoding.
- **Exploration has no analogue in compression.** A compressor is never rewarded for producing data it has not seen. An RL agent must deliberately generate unseen data or it cannot improve. This single asymmetry accounts for most of the difficulty of the field.
- The distribution shifts as the policy changes, which breaks the i.i.d. assumption underlying the optimization theory of every other paradigm here.

**The partial reconciliation.** If you insist on keeping the compression frame, the defensible version is: **RL compresses a search procedure into a policy**. The expensive object is the tree of rollouts, the planner, or the long chain of deliberation. The trained policy network is the amortized, compressed version of it. AlphaZero distilling Monte Carlo tree search into a policy network is exactly this; so is training a model to produce short correct reasoning after RL on long deliberate reasoning. Compression is the second half of the loop. Search is the first half, and search is where the new information actually enters.

**Where it fails.**

- **Sample efficiency.** A direct consequence of the bits-per-sample table in Section 2.3.
- **Reward specification.** Any objective that can be gamed will be gamed. This is not a moral observation, it is what optimization does.
- **Credit assignment over long horizons** remains unsolved in general.

**When it works well.** When the environment is cheap to simulate, when reward is dense or shapeable, or when a formal verifier exists. The last case is why RL on mathematics and code has worked far better than RL on open-ended tasks: a proof checker or a unit test is a free, perfectly reliable, unlimited source of ground truth.

---

### 3.8 Preference learning and RLHF

**Objective.** Fit a reward model to human pairwise preferences (typically a Bradley-Terry model), then optimize the policy against it with a KL penalty toward the pretrained reference model. Direct preference optimization (DPO) collapses these two stages into a single closed-form loss over preference pairs.

**What is compressed.** Human judgement, into a scalar reward function, and then into a policy. The KL penalty is doing exactly what it appears to do: bounding how far the policy is permitted to move from the prior, measured in nats.

**Information source.** Human annotators. Each comparison is one bit.

**Why the information accounting is decisive here.** Tens or hundreds of thousands of comparisons is at most a few hundred thousand bits. Set against a pretraining corpus of roughly `10^13` bits, this is not a rounding error away from zero, it **is** the rounding error. It follows immediately that RLHF cannot be installing new knowledge or new skills. It is selecting among behaviours the prior already supports.

That is not a criticism. Selecting the right mode from an enormous space of supported behaviours is exactly the problem, and it is why a small number of bits applied at the right place is so effective. But it does set hard limits: post-training cannot repair a capability the base model lacks, and the frequent claim that a fine-tune "taught the model" a new skill is almost always better explained as elicitation.

**Where it fails.**

- **Reward model overoptimization.** The reward model is a finite-sample approximation. Optimizing hard against it finds its errors rather than the human preference it stands for. The KL penalty exists to limit this and is the main thing standing between a working system and reward hacking.
- **Preference aggregation.** Averaging inconsistent preferences across annotators produces an objective that no individual annotator holds.
- **One bit at a time is an expensive channel.** Constitutional and AI-feedback methods exist largely to widen it.

---

### 3.9 Self-play

**Objective.** Improve by playing against copies of oneself, using the outcome as the learning signal.

**Information source.** The rules of the game. This is the critical point: **the rules are a free, infinite, perfectly reliable source of ground truth**. There is no annotation cost and no annotation ceiling.

**Why it worked spectacularly.** In Go, chess and shogi, self-play plus tree search reached superhuman performance from no human data. The information came from the game rules combined with search, and the compression step was distilling the search result into the network. It is the cleanest demonstration of the search-then-compress loop in Section 3.7.

**Why it has not generalized.** The requirement is a cheap, automatic, reliable evaluator. Games have one by definition. Most domains do not. The domains where self-play-like methods have since worked (competitive programming, formal mathematics) are exactly the domains that have verifiers. The search for verifiers in open-ended domains is, in information terms, the search for a high-bandwidth free information source, and it is the central open problem in scaling RL.

---

### 3.10 Evolutionary and gradient-free methods

Covers genetic algorithms, evolution strategies, neuroevolution, and population-based training.

**Objective.** Maintain a population, perturb, evaluate fitness, select. No gradients.

**What is compressed.** Nothing explicitly. Information enters only through fitness comparisons, at roughly one bit each.

**Why it is sometimes the right choice.**

- Works where gradients do not exist: discrete structures, non-differentiable objectives, black-box simulators.
- Embarrassingly parallel. Evaluations are independent, so wall-clock time can be traded for compute almost without limit.
- Robust to deceptive local structure, since the population explores multiple basins.

**Why it is usually not.** The sample efficiency is exactly what the bits-per-comparison accounting predicts: poor. A gradient supplies a direction in parameter space, worth far more than a scalar comparison. Where a gradient is available, using it is not optional.

**Status.** A niche, but a real one: hyperparameter search, architecture search, reward design, and settings where the objective is a black box.

---

### 3.11 Knowledge distillation

**Objective.** Train a student to match a teacher's output distribution rather than hard labels.

**What is compressed.** **The teacher, not the data.** This is the defining feature and it is why distillation sits in its own cell of the grid.

**Information source.** The teacher model, which is itself a compressed corpus. Distillation is therefore a second compression stage applied to the output of a first.

**Why it works.** A hard label over `K` classes carries `log2(K)` bits. A soft label carries the teacher's entire predicted distribution, including the relative ordering and magnitude of the wrong answers. The information that a particular image is 70 percent cat, 25 percent lynx, 5 percent dog tells the student about the geometry of the class manifold in a way that the single token "cat" cannot. This "dark knowledge" is why a student can match a teacher on a fraction of the data.

**Variants worth distinguishing.**

- **Response distillation**: match output distributions. The classic.
- **Feature distillation**: match intermediate activations. Higher bandwidth, requires architectural compatibility.
- **Sequence-level distillation**: train on the teacher's generations. Now the dominant method for producing small language models.
- **Self-distillation**: teacher and student are the same architecture. Reliably improves accuracy, which is theoretically awkward and usually attributed to regularization through label smoothing.

**Where it fails.** The student is bounded by the teacher plus whatever the data supports. Iterated distillation without a fresh information source degrades, since each round loses tail behaviour. This is the mechanism behind model collapse in recursive training on synthetic data.

---

### 3.12 Meta-learning and in-context learning

**Objective.** Learn across a distribution of tasks such that adaptation to a new task is fast. MAML optimizes for parameters from which a few gradient steps suffice; in-context learning achieves the same effect with no parameter updates at all.

**What is compressed.** **The learning algorithm itself.** The inner loop is amortized into the weights, so that at test time the model performs something functionally equivalent to learning without running an optimizer.

**Information source.** A distribution over tasks.

**Why in-context learning is the important case.** A sufficiently good next-token predictor, trained on a corpus that contains many instances of "demonstration followed by generalization", must learn to perform that generalization in order to predict well. In-context learning was not designed; it emerged from the compression objective in Section 3.2 as a consequence of what the corpus contains. There is reasonable evidence that transformers can implement gradient-descent-like updates in their forward pass, which would make the amortization literal rather than metaphorical.

The consequence for the taxonomy: in-context learning is the extreme point on Axis 3. Compression happens in activations, at inference, and the weights are never touched. It is a fully deferred, fully transient compression.

**Where it fails.** Limited by context length and by the quality of the task distribution seen in pretraining. It adapts within the prior; it does not extend it.

---

### 3.13 Inverse reinforcement learning

**Objective.** Given demonstrations, recover the reward function that would make them optimal.

**What is compressed.** **Behaviour into an objective.** A set of trajectories is enormous; a reward function is short. This is explicitly an MDL move: find the shortest explanation of why the agent did what it did.

**Information source.** Demonstrations, typically human.

**Why it is attractive.** Rewards transfer where policies do not. A policy is tied to a specific environment; the objective behind it may be portable to a new one. It is also the natural formalization of learning what someone wants rather than what they did.

**Why it is hard.** The problem is fundamentally ill-posed. Many reward functions explain the same behaviour, including the trivial constant reward under which all behaviour is optimal. Every practical method resolves this with an additional assumption: maximum entropy IRL assumes the demonstrator is Boltzmann-rational, adversarial methods assume a particular divergence. The assumption is doing the identification, not the data. It is a compression problem whose regularizer determines the answer.

---

### 3.14 Program synthesis and symbolic regression

**Objective.** Find a program, expression, or formal structure that reproduces the observed data.

**What is compressed.** The data, into a literal program. This is the only family in this document that optimizes something close to Kolmogorov complexity **directly** rather than through a differentiable surrogate.

**Information source.** Input-output examples, a specification, or a formal property.

**Why it is worth taking seriously.** The outputs are exact, verifiable, interpretable and extrapolate correctly outside the training range, which no fitted neural network reliably does. Recovering a conservation law from trajectory data gives you a statement that holds everywhere, not a function that interpolates where you sampled.

**Why it does not dominate.** The search space is discrete, combinatorial and non-differentiable. Scaling has been the obstacle for fifty years. The current direction is neural guidance of symbolic search: use a learned model to propose candidates and a formal checker to verify them, which is the search-then-compress loop again with a verifier supplying free information.

---

### 3.15 Bayesian inference

**Objective.** Compute or approximate the posterior `p(theta | D)`.

**What is compressed.** Data into a posterior distribution, which is a compressed belief state and a sufficient statistic for all future prediction.

**The connection to everything else.** Bayes and MDL are formally equivalent under the correspondence `L(x) = -log2 p(x)`. Minimizing description length is maximizing posterior probability. The prior is a code, complexity penalties are code lengths for models, and the Bayesian Occam's razor falls out of normalization: a model that spreads probability over many datasets necessarily assigns less to any particular one.

**Why it matters in practice.** It is the only framework here that represents uncertainty as a first-class object rather than as an afterthought, which is what calibration, active learning (Section 3.18) and safe decision-making all require.

**Where it fails.** Exact inference is intractable for anything interesting. Variational approximations underestimate uncertainty in known and systematic ways. And for large neural networks, the prior over parameters is essentially unexaminable, so the thing doing the regularizing is not something anyone has inspected.

---

### 3.16 Retrieval and non-parametric methods

Covers k-nearest neighbours, retrieval-augmented generation, and nearest-neighbour language models.

**Objective.** Store the data. Compute the answer at query time.

**What is compressed.** **Nothing.** This is the deliberate anti-compression cell of the grid, and it is not a degenerate case but a rational engineering choice.

**Why it wins where it wins.** Return to Section 3.2: compression allocates bits in proportion to frequency. A fact that appears once in a corpus is extremely expensive to store in weights, because gradient descent will not allocate capacity to it, and cheap to store on disk. The long tail is therefore exactly where retrieval beats parametric memory, and the empirical results match this prediction closely.

**Additional properties that follow from not compressing.**

- Updatable without retraining. Add a document, and the knowledge is available immediately.
- Attributable. The source is a real object that can be cited and checked.
- Removable. Deleting a fact means deleting a row.

None of these are available to a parametric model, and all three are increasingly what production systems are judged on.

**Where it fails.** Retrieval quality becomes the bottleneck, the system cannot synthesize what is not retrieved, and latency grows with index size. It also cannot generalize: a k-NN model has no account of anything between its stored points.

**The correct reading.** Parametric and non-parametric memory are complementary, and the design question is a bit-allocation question: which knowledge is frequent and structural enough to be worth compressing into weights, and which is rare and volatile enough to be worth keeping raw.

---

### 3.17 Test-time compute and inference-time search

Covers chain-of-thought, best-of-n sampling, tree search over generations, verifier-guided decoding, and self-consistency.

**Objective.** Spend more computation per query to get a better answer, without changing the weights.

**What happens to compression.** This is the **inverse** operation. Distillation and RL compress an expensive search procedure into a cheap policy. Test-time search decompresses a cheap policy back into an expensive deliberation. The two are reciprocal, and modern systems run both directions in a loop:

```
search  ->  compress into policy  ->  search from a better starting point  ->  compress again
```

This loop is the central mechanism of current frontier system development. It is also the only known way to get past the data ceiling identified in Section 3.2, because search generates new information from compute rather than from a corpus.

**Why it works.** A forward pass has fixed depth. Problems requiring more sequential reasoning than that depth allows cannot be solved in one pass regardless of model size. Generating intermediate tokens converts sequential depth into sequence length, which is unbounded. This is a statement about computational class, not about prompting technique.

**Where it fails.**

- Requires a way to tell good outputs from bad. With a formal verifier it works extremely well. Using the model to judge itself works, but less well, and the gap between those two cases is the whole story of which domains have seen rapid progress.
- Cost per query scales with the search budget, which changes the deployment economics fundamentally.

**Status.** The most consequential shift in the field in recent years, and the clearest demonstration that Axis 3 (timing) is not a minor implementation detail.

---

### 3.18 Active learning and curriculum

**Objective.** Not "what is compressed" but **"which bits should I acquire next"**. Choose the next training example, query or task to maximize expected information gain.

**Information source.** Whatever oracle is available, but used selectively.

**Why it belongs in the taxonomy.** Every other paradigm takes its information source as given. This family treats acquisition itself as the object of optimization. Where labels cost money, this is the difference between a feasible and an infeasible project.

**Variants.** Uncertainty sampling, expected model change, and Bayesian experimental design (query where posterior entropy reduction is greatest, which requires Section 3.15 to be doing real work). Curriculum learning is the same idea applied to ordering rather than selection: train on examples that are neither trivial nor impossible, since those carry the most information relative to the current model.

**Where it fails.** Uncertainty estimates from neural networks are poorly calibrated, so the quantity being maximized is often not the quantity of interest. Aggressive active selection also produces a training set that is heavily non-i.i.d., which breaks other assumptions.

---

### 3.19 Cooperative generative-discriminative frameworks

A family in which a generator and a classifier are trained jointly with the classifier's state feeding back into what the generator produces. CCNETS (Causal Cooperative Nets, Park, Cho and Kim) is a recent representative and is described here as the concrete instance.

**Structure.** Three modules:

- **Explainer**: `e = E(x)`, encodes input into a latent
- **Reasoner**: `y' = R(fuse(x, e))`, predicts the label from raw input and latent together
- **Producer**: `x' = P(fuse(e, y))` from the true label, and `x'' = P(fuse(e, y'))` from the inferred label

A parameterizable fusion operator (the "Zoint" mechanism) combines the two inputs by concatenation, element-wise averaging, or independent use.

**The distinctive element: signed loss decomposition.** Three prediction losses are defined:

| Loss | Definition |
|---|---|
| Generation | `L(X, X')` |
| Reconstruction | `L(X, X'')` |
| Inference | `L(X', X'')` |

Each module then optimizes a signed combination that subtracts the loss it is least responsible for:

```
L_explainer = (L_inference + L_generation) - L_reconstruction
L_reasoner  = (L_reconstruction + L_inference) - L_generation
L_producer  = (L_reconstruction + L_generation) - L_inference
```

**Where it sits.** Self-conditioned generative augmentation with a classifier in the loop. Compared with SMOTE or ADASYN, which synthesize minority examples by feature-space interpolation independent of the classifier, the generator here is conditioned on classifier state, so synthesis targets the current decision boundary.

Critically, in information terms **no new information enters the system**. The Producer does not sample the environment and has no oracle. The synthetic data is a resampling of what the model already believes. Any gain comes from redistributing existing bits to shape the decision boundary, not from acquisition. This places a hard ceiling on what any method in this family can achieve, and it is the right lens for evaluating claims about synthetic data generally.

**Assessment.** The authors are explicit that "causal" here refers to feedback-driven interaction between modules rather than formal causal inference or structural causal modelling, which is an honest clarification that similar work often omits. The empirical case is thin: a single dataset (Kaggle credit card fraud, 0.17 percent positives), F1 of 0.7992 against 0.7686 for an autoencoder baseline, and a tenfold augmentation gain of 0.8111 to 0.8133 that is within noise. The comparison omits SMOTE, ADASYN and cost-sensitive learning, which are the baselines that matter for class imbalance; the authors acknowledge this and defer it to future work.

The more serious technical concern is that the subtractive loss terms are unbounded below. Nothing in the formulation prevents a module from reducing its objective indefinitely by inflating the loss it subtracts. That training converges in practice needs an explanation, and none is given. Treat the signed loss decomposition as an interesting design idea with an unexamined stability story, not as an established paradigm.

---

## 4. Cross-Paradigm Comparison

| Paradigm | Compresses | Information source | Bits per sample | Compression timing |
|---|---|---|---|---|
| Supervised | Input-label mapping | Human labels | `log2(K)` | Training |
| Generative SSL | Joint data distribution | Raw corpus | 10 to 20 per token | Training |
| Contrastive SSL | View-invariant content | Corpus plus augmentations | Bounded by `log(batch)` | Training |
| VAE | Data, with explicit rate term | Corpus | Tunable via beta | Training |
| Diffusion | Data, hierarchically by scale | Corpus | Dense | Training |
| GAN | Nothing measurable | Corpus via discriminator | Undefined | Training |
| RL | A search procedure, into a policy | Environment reward | Under 1 per trajectory | Training |
| RLHF | Human judgement | Human comparisons | 1 per comparison | Training |
| Self-play | Search results | Game rules (free, infinite) | Dense via search | Training |
| Evolutionary | Nothing explicit | Fitness comparisons | 1 per comparison | Training |
| Distillation | The teacher | Teacher model | Full output distribution | Training |
| Meta-learning | The learning algorithm | Task distribution | Varies | Training |
| In-context learning | The learning algorithm | Prompt | Varies | Inference |
| Inverse RL | Behaviour into an objective | Demonstrations | Dense but ill-posed | Training |
| Program synthesis | Data into a program | Examples or specification | Exact | Training |
| Bayesian | Data into a posterior | Data plus prior | Dense | Training |
| Retrieval | Nothing, by design | Corpus, stored raw | Not applicable | Inference |
| Test-time search | Decompresses instead | Compute plus verifier | Verifier-dependent | Inference |
| Active learning | Governs acquisition | Selective oracle | Maximized by construction | Training |
| Cooperative gen-disc | Redistributes existing bits | Corpus only | None added | Training |

---

## 5. The Grid

Placing the paradigms on the three axes from Section 2.5:

**By information source:**

| Source | Paradigms | Cost per bit | Ceiling |
|---|---|---|---|
| Raw corpus | Generative SSL, contrastive, VAE, diffusion | Near zero | Corpus size |
| Human labels | Supervised | High | Annotation budget |
| Human preferences | RLHF | Very high | Annotation budget |
| Teacher model | Distillation | Low | Teacher quality |
| Environment reward | RL | Very high per usable bit | Simulation cost |
| Formal verifier | Self-play, RL on code and mathematics | Near zero | Domain coverage |
| Own generations | Synthetic data, cooperative frameworks | Zero, and worth zero | No new information |

The last row is the important one. It is the cell where nothing is gained, and it is the cell that current enthusiasm for synthetic data most often lands in without noticing.

**By compression target:**

- **Weights**: most of the field. Opaque, unupdatable, non-attributable.
- **Explicit program**: program synthesis. Interpretable, verifiable, hard to find.
- **Policy**: RL. Behaviour rather than knowledge.
- **Posterior**: Bayesian methods. Uncertainty as a first-class object.
- **External index**: retrieval. Updatable, attributable, deletable.

**By timing:**

- **Fully amortized**: standard training. Cheap inference, fixed capability.
- **Fully deferred**: retrieval, in-context learning, test-time search. Expensive inference, adapts per query.
- **Both, in a loop**: current frontier systems. Search, distil the result, search again from a better starting point.

---

## 6. Worked Example: CIFAR-100

The taxonomy is only worth having if it makes decisions easier. This section applies it end to end to one small, familiar dataset: 60,000 colour images at 32x32, 100 classes, 500 training and 100 test images per class, plus 20 coarse superclass labels.

The point is not that CIFAR-100 is important. It is that running the information accounting first tells you which paradigms are available, which are excluded, and why, before any code is written.

### 6.1 The information accounting

| Quantity | Amount |
|---|---|
| Label information, full training set | 50,000 × log2(100) ≈ **332 kbits**, about 41 KB |
| Pixel information, full training set | 50,000 × 3072 bytes ≈ 1.2 Gbits raw, perhaps 300 to 400 Mbits of real entropy |
| Ratio | **Roughly 1000:1 in favour of the pixels** |

The entire label set fits in a small text file. You would be training tens of millions of parameters against 41 KB of supervision.

That single number determines the shape of everything below. The paradigms that pay off here are the ones that extract signal from the pixels (Sections 3.2 and 3.3), from a teacher (Section 3.11), or from an external pretrained prior. Everything else is competing over the same 41 KB.

Three secondary facts matter:

- **Coarse labels are included.** Twenty superclasses alongside the 100 fine classes. This is free hierarchical structure and therefore free additional label bits from the same annotation effort.
- **The dataset is exactly balanced** by construction, 500 images per class. This excludes an entire family of methods, discussed in 6.3.
- **It is cheap.** At 32x32, most of what follows is a single-GPU afternoon, which is why it remains a good teaching dataset even though it is a weak research benchmark.

### 6.2 Directly applicable paradigms

**Supervised learning (Section 3.1).** WideResNet-28-10 or ResNet-18 with standard augmentation. The reference point everything else is measured against, and nothing more. The coarse labels afford one free extension: a hierarchical loss, or coarse-label pretraining followed by fine-label fine-tuning. More label bits from the same annotation.

**Contrastive and non-contrastive SSL (Section 3.3).** SimCLR, MoCo v2, BYOL and DINO all run here. Pretrain on the 50,000 images with labels hidden, then linear probe.

Linear-probe accuracy will sit well below the supervised baseline when pretraining on CIFAR-100 alone. That is the dataset being small, not the method failing. The real value appears in the low-label regime below.

One implementation note that accounts for most bad CIFAR SSL results: the ImageNet stem must be replaced. A 7x7 stride-2 convolution followed by a maxpool destroys a 32x32 image before the first residual block.

**Masked image modelling (Section 3.2).** MAE and similar methods run, but 50,000 tiny images is well below the scale at which masked reconstruction becomes competitive with contrastive methods. Worth including as a comparison, not as the main approach.

**Diffusion (Section 3.5).** CIFAR at 32x32 is the canonical diffusion benchmark and was the original DDPM evaluation. Trainable on one GPU. It gives a working generative model, a valid variational bound on the likelihood reported in bits per dimension, and the most direct available demonstration of hierarchical bit allocation across noise scales. Also usable as a classifier and as a source of augmentation data.

**VAE (Section 3.4).** Trains easily, samples poorly. Its value here is diagnostic: sweep beta and plot the rate-distortion curve directly. That trade-off is hard to make visible anywhere else in the taxonomy.

**GAN (Section 3.6).** Historically a CIFAR benchmark. Run it once to observe mode collapse and the evaluation problem first-hand, then move on. The lesson is the one in Section 3.6, not the samples.

**Distillation (Section 3.11).** The highest-value cheap win on this dataset. CIFAR-100 is the standard knowledge-distillation benchmark and nearly every paper in that literature reports on it. Teacher WRN-28-10, student ResNet-8 or ResNet-20.

The information reason it works well *here specifically*: hard labels give 6.64 bits per example, but the teacher's soft distribution conveys relative structure across all 100 classes. With only 500 examples per class, that structure is worth a great deal. Self-distillation, teacher and student sharing an architecture, also gives a reliable small gain.

**Semi-supervised learning.** The highest-leverage option available, and the direct exploitation of the 1000:1 ratio. Hide most labels and run FixMatch, MixMatch or FlexMatch at 400, 2500 or 10,000 labels, which are the standard protocol points.

Consistency regularization and pseudo-labelling extract signal from the unlabeled pixels to amplify a tiny label budget. Watching 25 labels per class approach the performance of 500 labels per class is the single most informative experiment this dataset supports.

**Transfer and fine-tuning.** Practically, the strongest results come from importing an external prior: fine-tune an ImageNet-21k or CLIP-pretrained backbone with images upsampled to 224.

The taxonomy reading is worth stating plainly. You are not learning CIFAR-100. You are spending a prior built from billions of bits gathered elsewhere. This is also why CIFAR-100 is now a weak benchmark for any claim about learning efficiency: a result that does not state whether external pretraining was used is uninterpretable.

**Retrieval and k-NN (Section 3.16).** k-NN on frozen features is the standard secondary evaluation for SSL and requires no training. It is also diagnostic. A large gap between k-NN and linear-probe accuracy indicates a feature space that is linearly separable but metrically poorly organized, which tells you something specific about what your pretraining did.

**Bayesian methods and uncertainty (Section 3.15).** CIFAR-100 is a standard calibration benchmark and a good one, because 100 classes at 500 examples each produce genuinely uncertain predictions. Deep ensembles, MC dropout, SWAG, temperature scaling. Pair with CIFAR-100-C (corrupted) to test whether the uncertainty estimates survive distribution shift. They generally do not, which is the finding.

**Active learning (Section 3.18).** Simulate by starting from 1000 labels and querying in rounds. A standard benchmark, with the caveat that on a balanced dataset active selection frequently fails to beat random sampling. Reproducing that negative result is itself worthwhile.

**Meta-learning and few-shot (Section 3.12).** Not on vanilla CIFAR-100, but on the **CIFAR-FS** and **FC100** splits, which partition the 100 classes into disjoint meta-train, meta-validation and meta-test sets specifically to support this. MAML, ProtoNets and Reptile all run there. FC100 splits along superclass boundaries to reduce semantic overlap between splits, making it the harder and more honest of the two.

**Evolutionary methods and NAS (Section 3.10).** CIFAR is the classic architecture-search benchmark. Expensive, historically important. Use NAS-Bench-201 to get the search space precomputed rather than spending GPU-months rediscovering it.

### 6.3 Applicable after modifying the dataset

**Cooperative generative-discriminative and imbalance methods (Section 3.19).** CIFAR-100 is exactly balanced, so SMOTE, ADASYN, CCNETS, focal loss and class-balanced loss have nothing to act on. The standard construction is **CIFAR-100-LT**: subsample classes to an exponential profile at an imbalance factor of 10, 50 or 100. That is the recognized long-tailed benchmark and the correct setting for evaluating anything in that family.

**Noisy labels.** Inject symmetric or asymmetric label noise at 20 to 80 percent. CIFAR-100 with noisy labels is a standard robustness benchmark and a direct empirical test of the claim in Section 3.1 that a compression objective does not distinguish a causal feature from a spurious one.

### 6.4 Not applicable, and why

| Paradigm | Missing ingredient |
|---|---|
| Reinforcement learning (3.7) | No environment, no reward, no sequential decision. Classification can be framed as a contextual bandit, but each example then yields 1 bit instead of 6.64. Strictly harder for no benefit, which makes it a good one-off demonstration of exactly that point. |
| RLHF (3.8) | No preferences exist to collect. Nobody holds an opinion about whether an image is an otter. |
| Self-play (3.9) | No game, no opponent, no rules. |
| Inverse RL (3.13) | No trajectories and no demonstrator. |
| Program synthesis (3.14) | No short program generates natural images. The Kolmogorov argument failing in the other direction. |
| In-context learning (3.12) | Not native to vision encoders. Reachable only through a VLM, at which point the VLM is the object of study rather than the dataset. |
| Test-time search (3.17) | No verifier and no sequential reasoning to unroll. The vision analogues, test-time augmentation and test-time training, are real but a much weaker form of the idea. |

The pattern is uniform. Every excluded paradigm requires an information source this dataset does not have: an environment, a verifier, a human in the loop, or a task distribution. What remains is a fixed corpus and 41 KB of labels, which is precisely why only the corpus-and-teacher paradigms apply.

This is the general lesson, and it transfers to any dataset. Enumerate the available information sources first. The list of applicable paradigms follows from that almost mechanically.

### 6.5 A concrete experimental ladder

A single sequence that demonstrates the taxonomy on one dataset:

1. **Supervised baseline.** WRN-28-10. Establishes the reference and the 41 KB ceiling.
2. **Same model, 10 percent of labels.** Watch it collapse. The label-information constraint made visible.
3. **SimCLR pretraining on all 50,000 unlabeled images, then fine-tune on that same 10 percent.** Most of the gap closes. The pixels supplied what the labels could not.
4. **FixMatch on the same 10 percent.** Closes more of it. Steps 2 through 4 form a clean bits-accounting story.
5. **Distil the full-label WRN into a ResNet-8.** Compare against a directly trained ResNet-8. The difference is the dark-knowledge bits from Section 3.11.
6. **Train a diffusion model.** Report bits per dimension. The one paradigm here with an honest, exact compression number.
7. **Deep ensemble, evaluated on CIFAR-100-C.** Accuracy barely moves, calibration improves substantially, both degrade under shift. Uncertainty is a separate axis from accuracy.
8. **Fine-tune a CLIP or ImageNet-21k backbone.** Beats everything above by a wide margin, which is the correct final lesson about this dataset and about benchmark interpretation generally.

Steps 1 through 5 are roughly a day of compute. The full ladder is about a week.

---

## 7. Where the Frame Breaks

Intellectual honesty requires stating the limits of the organizing idea.

**Compression does not explain generalization on its own.** A model that perfectly compresses its training set may generalize badly. The bridge requires additional assumptions about the data distribution and the hypothesis class, and the standard learning-theoretic bounds are famously vacuous for over-parameterized networks. Compression is necessary, not sufficient.

**Optimization is not accounted for.** The frame describes objectives, not the process of reaching them. Two methods with identical objectives can differ enormously in whether they are trainable at all. GAN instability is an optimization fact, not an information fact.

**Bits are not fungible.** The bits-per-sample accounting treats all bits as equal, and they are not. One bit that resolves a critical ambiguity in a policy is worth more than a thousand bits of redundant text. RLHF's effectiveness relative to its tiny information budget is precisely the demonstration that placement matters more than volume, and the frame as stated does not capture this.

**Inductive bias is invisible in the accounting.** Architecture, augmentation policy and prior all encode substantial information that never appears in a bits-per-sample table. The choice of convolution over a dense layer is a large amount of information about the structure of images, supplied by a human for free.

**The frame is descriptive, not generative.** It will tell you why a method behaves as it does. It has not, historically, been the thing that produced new methods. Those have generally come from elsewhere and been rationalized afterwards.

---

## 8. Practical Consequences

Six things that follow directly from the above and are worth acting on.

1. **Before adopting any method, ask what information it adds.** If the answer is "none", the best case is that it redistributes existing bits more usefully. That is sometimes worth doing and is never a large effect. This test alone disposes of most synthetic-data enthusiasm.

2. **Do not expect post-training to add capability.** Fine-tuning and preference optimization elicit. If the base model cannot do the task at all, no amount of post-training on a few thousand examples will install it. Change the base model or change the task.

3. **Look for verifiers.** The domains where progress has been fastest are exactly the domains with cheap automatic ground truth. If you can construct a verifier for your problem, you have converted an annotation-bound problem into a compute-bound one, which is the single highest-leverage move available.

4. **Allocate knowledge deliberately between weights and index.** Frequent, structural, slow-changing knowledge belongs in weights. Rare, volatile, attributable, deletable knowledge belongs in retrieval. Choosing by default rather than by analysis leaves a large amount of performance and a large amount of governance capability on the table.

5. **Treat inference-time compute as a design variable.** The choice of where on the amortization curve to sit is now a primary architectural decision with direct cost implications, not a deployment afterthought.

6. **Be suspicious of methods that cannot state what they compress.** GANs are the cautionary case. If there is no principled evaluation metric, the field cannot tell progress from noise, and it will eventually stall regardless of early results.

---

## 9. References and Further Reading

**Foundations**

- Shannon, C. E. (1948). A Mathematical Theory of Communication.
- Rissanen, J. (1978). Modeling by Shortest Data Description. The origin of MDL.
- Grünwald, P. (2007). The Minimum Description Length Principle.
- Li, M. and Vitányi, P. An Introduction to Kolmogorov Complexity and Its Applications.
- Tishby, N., Pereira, F. and Bialek, C. (1999). The Information Bottleneck Method.
- Hinton, G. and van Camp, D. (1993). Keeping Neural Networks Simple by Minimizing the Description Length of the Weights.

**Compression and language models**

- Delétang et al. (2023). Language Modeling Is Compression.
- Alemi et al. (2018). Fixing a Broken ELBO. The rate-distortion reading of the VAE objective.

**Paradigms**

- Kingma, D. and Welling, M. (2013). Auto-Encoding Variational Bayes.
- Higgins et al. (2017). beta-VAE.
- Goodfellow et al. (2014). Generative Adversarial Nets.
- Ho, J., Jain, A. and Abbeel, P. (2020). Denoising Diffusion Probabilistic Models.
- Devlin et al. (2018). BERT. Masked language modelling.
- Brown et al. (2020). Language Models Are Few-Shot Learners. In-context learning.
- Oord, A. van den, Li, Y. and Vinyals, O. (2018). Representation Learning with Contrastive Predictive Coding. InfoNCE.
- Chen et al. (2020). SimCLR.
- Radford et al. (2021). CLIP.
- Hinton, G., Vinyals, O. and Dean, J. (2015). Distilling the Knowledge in a Neural Network.
- Finn, C., Abbeel, P. and Levine, S. (2017). Model-Agnostic Meta-Learning.
- Ziebart et al. (2008). Maximum Entropy Inverse Reinforcement Learning.
- Silver et al. (2017). Mastering the Game of Go without Human Knowledge. AlphaGo Zero.
- Christiano et al. (2017). Deep Reinforcement Learning from Human Preferences.
- Rafailov et al. (2023). Direct Preference Optimization.
- Gao, L., Schulman, J. and Hilton, J. (2022). Scaling Laws for Reward Model Overoptimization.
- Khandelwal et al. (2019). Generalization through Memorization: Nearest Neighbor Language Models.
- Lewis et al. (2020). Retrieval-Augmented Generation.
- Wei et al. (2022). Chain-of-Thought Prompting.
- Park, H., Cho, Y. and Kim, H. CCNETS: A Modular Causal Learning Framework for Pattern Recognition in Imbalanced Datasets. arXiv:2401.04139.