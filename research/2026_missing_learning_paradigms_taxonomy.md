# The Empty Cells

### Learning Paradigms That Do Not Yet Exist, Derived From an Information-Theoretic Taxonomy

*A companion piece to "Learning Paradigms Through the Lens of Information", written to stand alone. Where that document maps what exists, this one uses the same map to locate what is missing, states which gaps are opportunities and which are correctly empty, and gives falsifiable predictions for each.*

---

## Table of Contents

1. [Purpose and Method](#1-purpose-and-method)
   - 1.1 [What this document is doing](#11-what-this-document-is-doing)
   - 1.2 [The taxonomy, restated for self-containment](#12-the-taxonomy-restated-for-self-containment)
   - 1.3 [Two sources of prediction](#13-two-sources-of-prediction)
   - 1.4 [Standard of evidence](#14-standard-of-evidence)
2. [The Occupancy Map](#2-the-occupancy-map)
   - 2.1 [What the existing paradigms cover](#21-what-the-existing-paradigms-cover)
   - 2.2 [Visible holes](#22-visible-holes)
   - 2.3 [The admitted failures of the frame](#23-the-admitted-failures-of-the-frame)
3. [Candidate Paradigms](#3-candidate-paradigms)
   - 3.1 [Learned allocation across memory tiers](#31-learned-allocation-across-memory-tiers)
   - 3.2 [Verifier synthesis](#32-verifier-synthesis)
   - 3.3 [Value-weighted acquisition](#33-value-weighted-acquisition)
   - 3.4 [Compression of the training trajectory](#34-compression-of-the-training-trajectory)
   - 3.5 [Honest joint MDL at scale](#35-honest-joint-mdl-at-scale)
   - 3.6 [Forgetting as an objective](#36-forgetting-as-an-objective)
   - 3.7 [Decompilation of weights into programs](#37-decompilation-of-weights-into-programs)
   - 3.8 [Continuous amortization](#38-continuous-amortization)
   - 3.9 [Explicit inductive bias accounting](#39-explicit-inductive-bias-accounting)
4. [Cells That Should Stay Empty](#4-cells-that-should-stay-empty)
5. [Prioritization](#5-prioritization)
6. [How to Evaluate a Claimed New Paradigm](#6-how-to-evaluate-a-claimed-new-paradigm)
7. [Limits of This Exercise](#7-limits-of-this-exercise)
8. [Summary Table](#8-summary-table)
9. [References](#9-references)

---

## 1. Purpose and Method

### 1.1 What this document is doing

A taxonomy earns its keep in two ways. The first is descriptive: it explains why the things that exist behave as they do. The second is generative: the structure of the classification makes visible the positions that nothing currently occupies.

The periodic table is the standard example, and it is worth being precise about why it worked. It did not predict new elements because it was elegant. It predicted them because it had **axes with known values** and **cells that were empty while their neighbours were full**. Mendeleev could specify gallium's atomic weight and density before anyone found it, because the cell's coordinates constrained the properties.

The taxonomy below is far weaker than that. Its axes are categorical rather than numeric, its cells are not equally spaced, and the analogy should not be pushed. But the basic move survives: if a classification has real axes, an empty cell whose neighbours are all occupied is a question worth asking, and sometimes the coordinates constrain what would go there.

What follows is that exercise, conducted honestly. Nine candidate paradigms, each with its prior art stated plainly rather than suppressed to make the idea look newer, and a separate list of cells that are empty for good reasons and should remain so.

### 1.2 The taxonomy, restated for self-containment

The frame classifies any learning method by three questions.

**Axis 1: Where does the information come from?**

| Source | Occupied by | Cost per bit | Ceiling |
|---|---|---|---|
| Raw corpus | Generative and contrastive pretraining, VAEs, diffusion | Near zero | Corpus size |
| Human labels | Supervised learning | High | Annotation budget |
| Human preferences | RLHF, DPO | Very high, one bit per comparison | Annotation budget |
| Teacher model | Distillation | Low | Teacher quality |
| Environment reward | Reinforcement learning | Very high per usable bit | Simulation cost |
| Formal verifier | Self-play, RL on code and mathematics | Near zero | Domain coverage |
| Own generations | Synthetic data, self-training | Zero, and worth zero | No new information |

**Axis 2: What is the information compressed into?**

| Target | Occupied by | Properties |
|---|---|---|
| Weights | Most of the field | Opaque, unupdatable, non-attributable, non-deletable |
| Explicit program | Program synthesis, symbolic regression | Interpretable, verifiable, extrapolates, hard to find |
| Policy | Reinforcement learning | Behaviour rather than knowledge |
| Posterior | Bayesian methods | Uncertainty as a first-class object |
| External index | Retrieval, k-NN, RAG | Updatable, attributable, deletable, does not generalize |

**Axis 3: When is the compression paid for?**

| Timing | Occupied by | Economics |
|---|---|---|
| Training time | Standard training | Cheap inference, fixed capability |
| Inference time | Retrieval, in-context learning, test-time search | Expensive per query, adapts per query |
| Alternating loop | Frontier system development | Search, distil, search again from a better start |

Three supporting facts from the parent document are load-bearing below and are restated here.

**The bits-per-sample table.** The information content of one training signal predicts most of the sample-efficiency ordering in machine learning.

| Signal | Approximate bits |
|---|---|
| Next-token prediction | 6 to 20 per token |
| Teacher soft labels | Several, the full output distribution |
| Hard label, K classes | log2(K) |
| Human preference comparison | 1 |
| Scalar RL reward at episode end | Often well under 1 of usable signal |

**The elicitation argument.** A pretraining corpus carries on the order of 10^13 bits. A preference dataset of a hundred thousand comparisons carries 10^5. Post-training therefore cannot be installing new capability; it locates and amplifies modes the prior already supports. This is elicitation, not acquisition.

**The reciprocity of search and compression.** Distillation and RL compress an expensive search procedure into a cheap policy. Test-time search decompresses a cheap policy back into expensive deliberation. Current systems run both directions in a loop, and that loop is the only known route past the finite-corpus ceiling, because search manufactures information from compute rather than drawing it from data.

### 1.3 Two sources of prediction

**Source one: empty cells in the cross product.** Seven information sources, five compression targets, three timings. That is 105 coordinates, and fewer than twenty are meaningfully occupied. Most of the remainder are empty because they are incoherent, but a minority are empty because nobody has gone there.

The filter for "worth pursuing" is whether the neighbouring cells are occupied and productive. A cell whose row and column are both well populated, but which is itself vacant, is the interesting case.

**Source two: the frame's own admitted failures.** The parent document contains a section listing what the information frame cannot account for. Each admission is a candidate location for a missing paradigm rather than merely a caveat, on the reasoning that a frame usually fails to explain something because the thing that would explain it has not been built.

The five admissions:

1. Compression does not explain generalization on its own.
2. Optimization dynamics are not accounted for.
3. Bits are not fungible; placement matters more than volume.
4. Inductive bias is invisible in the accounting.
5. The frame is descriptive rather than generative.

Items 2, 3 and 4 generate candidates below (Sections 3.4, 3.3 and 3.9 respectively). Item 1 is a genuine limit of the frame rather than a missing method. Item 5 is the concession that this entire document is working against.

### 1.4 Standard of evidence

Every candidate below is stated with:

- **The coordinates.** Which cell it occupies on the three axes.
- **Prior art, stated fairly.** Nearly every idea here has partial precedent. Suppressing it would make the predictions look stronger and be dishonest. In several cases the prior art is decades old and the claim is that it was abandoned prematurely rather than that it is new.
- **Why the cell is empty.** Distinguishing "nobody looked" from "somebody looked and it was hard" from "it is hard for a reason that has not changed".
- **A falsifiable prediction.** Something that could be checked, and that would be wrong if the candidate is a bad idea.
- **The strongest objection.** Where the candidate is most likely to fail.

Candidates that cannot meet this standard belong in Section 4, not Section 3.

---

## 2. The Occupancy Map

### 2.1 What the existing paradigms cover

Reading the three axes together, the occupied region has a shape worth naming.

**Almost everything compresses into weights.** Nineteen of the twenty paradigms in the parent taxonomy target parameters, a program, a policy or a posterior, and of these, weights dominate by an enormous margin. Retrieval is the single mainstream exception, and it is usually bolted on rather than designed in.

**Almost everything has a fixed information source.** The source is chosen by the researcher before training begins and does not change. Active learning is the only family that treats acquisition itself as an object of optimization, and it optimizes volume rather than value.

**Timing has only recently become a design variable.** Until inference-time search became prominent, Axis 3 had effectively one setting. That it turned out to be a primary architectural decision rather than a deployment detail is the clearest recent evidence that the axes are real and that unexamined axes hide consequential choices.

### 2.2 Visible holes

Seven structural observations, each of which generates a candidate.

1. **Axis 2 is never learned.** The compression target is always a human design decision made once, at architecture time. Nothing chooses it. (→ 3.1)
2. **Verifiers are treated as a domain property, never as an artifact to be built.** The taxonomy identifies them as the uniquely free information source and then takes their existence as given. (→ 3.2)
3. **Acquisition optimizes bits, not value.** Active learning maximizes expected information gain measured in bits, while the frame's own analysis of RLHF shows that placement dominates volume. (→ 3.3)
4. **The training process is never itself compressed.** Data, teachers, policies and posteriors are all compressed. The trajectory through parameter space is discarded. (→ 3.4)
5. **Model cost is excluded from the accounting it supposedly obeys.** Bits-per-token figures omit the parameters, which a fair MDL accounting must include. (→ 3.5)
6. **Rate-distortion governs encoding but not removal.** What to discard at encoding time is theorized; what to remove afterward is not. (→ 3.6)
7. **The best source and the best target have never been combined.** A trained teacher is a cheap and unlimited information source; a program is the most useful compression target. Nothing distils a network into a program. (→ 3.7)

Plus two from the timing axis and the bias admission:

8. **The search-compress loop has a period of months and a human in it.** Nothing about the mathematics requires that. (→ 3.8)
9. **Human-supplied inductive bias is never priced.** (→ 3.9)

### 2.3 The admitted failures of the frame

The mapping from admission to candidate, for clarity:

| Admitted failure | Candidate it generates | Section |
|---|---|---|
| Optimization dynamics unaccounted for | Compression of the training trajectory | 3.4 |
| Bits are not fungible | Value-weighted acquisition | 3.3 |
| Inductive bias is invisible | Explicit inductive bias accounting | 3.9 |
| Compression does not explain generalization | None. This is a real limit, not a gap. | |
| The frame is descriptive, not generative | This document is the test of that claim. | |

The fourth row deserves a note. The gap between compressing training data and generalizing to new data is bridged by assumptions about the data distribution and the hypothesis class, and the standard learning-theoretic bounds are vacuous for over-parameterized networks. No new paradigm fixes this. It is a limit on what the frame can claim, and the honest response is to stop claiming it.

---

## 3. Candidate Paradigms

---

### 3.1 Learned allocation across memory tiers

> **Coordinates:** Axis 2 becomes a learned variable rather than a design constant.

#### The observation

Every system in the taxonomy has its compression target fixed in advance. An architect decides that knowledge goes in weights, or that facts go in a retrieval index, or that a scratchpad holds intermediate state. That decision is then applied uniformly to everything the system learns.

But the tiers have genuinely different cost profiles, and the right tier differs per item:

| Tier | Write cost | Read cost | Mutable | Attributable | Generalizes |
|---|---|---|---|---|---|
| Weights | Very high | Near zero | No | No | Yes |
| External index | Near zero | Moderate, grows with size | Yes | Yes | No |
| Cached computation | Low | Low | Yes | Partly | No |
| Context window | Zero | Grows with length | Yes | Yes | Within prior |

A fact accessed constantly, stable over years, and needed for compositional reasoning belongs in weights. A fact accessed rarely, subject to revision, and needing citation belongs in an index. Nothing currently makes that determination per item.

#### What the paradigm would be

Training that treats tier placement as part of the objective. For each piece of knowledge, the system estimates access frequency, volatility, attribution requirement and compositional utility, and allocates accordingly. The objective is a constrained bit-allocation problem across tiers with different cost functions, which is well-posed and, as far as I can tell, unposed.

#### Prior art, stated fairly

- **Memory-augmented neural networks** and **product-key memory layers** learn what to store in a differentiable memory. They do not ask whether the differentiable memory is the right tier; that is fixed by the architecture.
- **Mixture-of-experts** routes computation conditionally. Storage placement is not the routed variable.
- **Semi-parametric language models** and **retrieval-augmented generation** combine tiers with a fixed or lightly-learned blend weight, applied globally rather than per item.
- **kNN-LM** interpolates parametric and non-parametric predictions with a single scalar, which is the degenerate one-parameter version of what this candidate proposes.

The gap is specific: existing work learns *within* a tier or blends tiers *uniformly*. Nothing learns *which tier a given item belongs in*.

#### Why the cell is empty

The tiers were developed by separate communities with separate infrastructure, on separate timelines. Parametric training and retrieval systems have different tooling, different evaluation traditions and often different teams. The interface between them is a configuration file, not a learned function.

#### Why it becomes forced

The parent taxonomy's argument for retrieval is that a compression objective allocates bits in proportion to frequency, which structurally under-serves the long tail. A fact appearing once in a corpus is expensive to store in weights and cheap to store on disk.

The current response to that argument is a human deciding that "facts go in RAG." That decision is made thousands of times, by judgement, without measurement, and it is exactly the kind of decision that yields to optimization once someone poses it as one.

#### Falsifiable prediction

A system with learned per-item tier allocation beats a fixed-blend semi-parametric baseline on long-tail question answering at matched total storage. Further, the learned allocation policy correlates with corpus frequency in a way that recovers the frequency-proportionality argument from data rather than assuming it.

If the learned policy turns out to be approximately constant, the candidate is wrong and the uniform blend was right all along.

#### Strongest objection

Estimating access frequency and volatility at training time requires predicting future queries, which may be unpredictable enough that the learned policy degenerates to a constant. The candidate depends on query distributions being sufficiently regular, and that is an empirical question nobody has measured.

---

### 3.2 Verifier synthesis

> **Coordinates:** Axis 1. Constructs an information source rather than consuming one.

#### The observation

The taxonomy identifies a single free lunch. Verifiers, game rules, proof checkers and unit tests are information sources that are simultaneously free, unlimited and perfectly reliable. Every domain where machine learning has advanced unusually fast is a domain that happened to possess one.

The taxonomy then notes that self-play generalized only to domains with verifiers, recommends looking for verifiers, and stops. Nothing in it constructs one.

This is the largest omission in the frame. The document treats the presence of a verifier as a fact about a domain, like its dimensionality, rather than as an artifact that might be engineered.

#### What the paradigm would be

Verifier construction as the primary research problem, with its own objective, its own evaluation and its own scaling behaviour. Given a domain lacking automatic ground truth, learn or construct a checker that is:

1. **Cheaper than the generator.** Otherwise there is no leverage.
2. **More reliable than the generator's self-assessment.** Otherwise it adds nothing over self-consistency.
3. **Hard to game.** Otherwise Section 4's overoptimization failure applies immediately.

The third requirement is the hard one and is where every current attempt struggles.

#### Why this is the highest-leverage empty cell

It is the only candidate in this document that **changes the information source** rather than rearranging what is done with existing sources. Every other entry redistributes bits that are already present. This one manufactures new ones.

The consequence, if it works even partially in one open-ended domain, is a phase change for that domain: it moves from annotation-bound, where progress costs money linearly, to compute-bound, where progress costs compute and compute scales.

#### Prior art, stated fairly

- **Process reward models** supervise intermediate reasoning steps rather than final answers. A learned verifier, used as one, but trained on human step-labels and therefore annotation-bound in exactly the way the candidate aims to escape.
- **LLM-as-judge** and **self-consistency** use a model to evaluate outputs. Cheap, but reliability is bounded by the generator's own competence, which violates requirement 2 in precisely the cases that matter.
- **Debate** and **recursive reward modelling** are proposals for amplifying weak verification into strong verification. Closest in spirit. Largely untested at scale.
- **Test-driven and property-based approaches in code** exploit a natural verifier rather than constructing one.

The gap: all of these *use* a verifier. None treats verifier construction as the object of study, with a benchmark measuring verifier quality independently of the generator it supervises.

#### Why the cell is empty, stated carefully

The generator-verifier asymmetry that makes this work in mathematics and code is a property of those domains, not a general fact about problems.

Checking a proof is genuinely cheaper than finding one. Running a unit test is genuinely cheaper than writing the function. But checking whether an essay is good is not obviously cheaper than writing a good essay, and checking whether a strategy is sound may be as hard as devising one.

So the honest formulation of the research question is not "how do we build verifiers" but:

> **Which domains possess a generator-verifier asymmetry that has not been exploited because nobody looked for it?**

That question has not been asked systematically. The domains currently exploited were found by accident, because they had verifiers lying around for other reasons.

#### Falsifiable prediction

There exists a measurable quantity, some form of generator-verifier cost-and-reliability gap, that predicts in advance whether reinforcement learning will succeed in a given domain. Someone will define it, and it will correctly retrodict the success in mathematics and code and the failure in open-ended generation.

If no such quantity predicts the historical record, the asymmetry is not the operative variable and this candidate is built on a misdiagnosis.

#### Strongest objection

A learned verifier is a learned reward model, and reward model overoptimization is a well-documented failure mode: optimizing hard against a finite-sample approximation finds its errors rather than the thing it approximates. A synthesized verifier inherits this in full, and the KL-penalty mitigation used in RLHF requires a trusted reference policy, which may not exist in the settings where verifier synthesis is most needed.

---

### 3.3 Value-weighted acquisition

> **Coordinates:** Axis 1. Addresses the admitted failure that bits are not fungible.

#### The observation

The bits-per-sample accounting treats all bits as equal. They are not.

The frame's own analysis demonstrates this. RLHF applies on the order of 10^5 bits to a model carrying 10^13 and produces a large, reliable behavioural change. If bits were fungible, that would be impossible. The explanation is that those bits are placed where they resolve a consequential ambiguity, and the frame as stated cannot express the difference.

Active learning, the one family that optimizes acquisition, maximizes **expected information gain**: a quantity measured in bits, blind to whether those bits matter.

#### What the paradigm would be

Acquisition that maximizes expected **decision-relevant** information. Bits weighted by how much they change downstream behaviour that someone cares about, rather than by how much they reduce posterior entropy.

The distinction in one line:

- Active learning asks: which query most reduces my uncertainty?
- This asks: which query most changes what I would do?

These come apart constantly. A model can be maximally uncertain about something that affects no decision, and confidently wrong about something that affects every decision.

#### Prior art, stated fairly

This is the least novel candidate in the document, and it is included because being old and unused is itself informative.

- **Bayesian experimental design** and the **expected value of sample information** are exactly this formalism. They are decades old, well understood, and correct.
- **Decision-theoretic active learning** exists in the literature at small scale.
- **Influence functions** measure the effect of a training point on model behaviour, which is a retrospective version of the same quantity.

The claim here is not novelty. It is that the correct formalism has never been scaled, and that the reason it has not been scaled is worth attacking directly.

#### Why the cell is empty

Two blockers, one soft and one hard.

**Soft:** the value function must be specified. Expected information gain is computable from a posterior alone; expected value of information requires a decision problem and a loss. In general-purpose model training, the downstream decision is unspecified by construction.

**Hard:** the required posteriors are untrustworthy. Neural network uncertainty estimates are poorly calibrated, and active learning already suffers from this. Value-weighting compounds it, because it multiplies a badly estimated uncertainty by a badly estimated value.

#### Why it becomes tractable now

RLHF is the existence proof that a small, well-placed annotation budget produces large effects. That establishes the phenomenon empirically even though the theory predates it by decades.

The commercial forcing function is direct: annotation is the dominant cost in post-training, and if placement value can be estimated even crudely, the return on annotation spend changes by orders of magnitude. That is the kind of pressure that gets hard problems attacked.

#### Falsifiable prediction

An annotation budget allocated by estimated decision-relevance outperforms the same budget allocated by uncertainty sampling on downstream task performance, by a margin that grows as the budget shrinks.

If uncertainty sampling matches it, then either the value estimates are too noisy to help or uncertainty is a sufficient proxy for value, and either finding settles the question.

#### Strongest objection

On balanced, well-specified datasets, active learning frequently fails to beat random sampling. That is a documented and somewhat embarrassing result. Adding a value term to a method that does not reliably beat random is optimistic, and the candidate needs to explain why value-weighting succeeds where information-weighting did not.

---

### 3.4 Compression of the training trajectory

> **Coordinates:** Axis 2, applied to a novel object. Addresses the admitted failure that optimization is unaccounted for.

#### The observation

The taxonomy compresses data, teachers, policies, posteriors and learning algorithms. It never compresses **the learning process itself**.

A training run produces a trajectory through parameter space: a sequence of checkpoints, gradients, loss values, gradient norms and activation statistics, containing a large quantity of information about the loss landscape of that architecture on that data. Essentially all of it is discarded. What survives is the final checkpoint and, if someone was diligent, a loss curve in a logging dashboard.

#### What the paradigm would be

Treat the trajectory as a corpus and compress it. The product is a model of optimization dynamics that transfers across runs, supporting:

- Predicting final performance from early curves, at a fidelity beyond current scaling-law extrapolation.
- Transferring a learning rate schedule to a new architecture on principled rather than heuristic grounds.
- Warm-starting a new run from the trajectory of an old one, as opposed to from its endpoint.
- Diagnosing pathological dynamics early enough to intervene.

#### Distinguishing it from existing work

This candidate is easy to confuse with three things it is not.

- **Learned optimizers** compress the *update rule*, a fixed function applied at each step. This compresses the *path*, which is data. An optimizer is a policy; a trajectory model is a world model of optimization.
- **Scaling laws** compress the trajectory into two or three scalars. That is real compression, and extremely lossy. The candidate asks what the discarded structure contains.
- **Model soups and weight averaging** use multiple endpoints without modelling the path connecting them.

Also adjacent: **loss landscape analysis** and **mode connectivity** study the geometry directly, but as scientific investigation rather than as a compressed artifact to be reused.

#### Why the cell is empty

Until recently, training runs were cheap enough that discarding their byproducts cost nothing, and there were too few of them to constitute a corpus. Both conditions have reversed. Frontier runs are expensive enough that their exhaust is a visible waste, and enough runs now exist, within any large organization, to train on.

There is also a mundane blocker: checkpoint sequences are enormous and nobody has infrastructure designed to retain and query them.

#### Falsifiable prediction

A model trained on trajectory data predicts final performance from the first ten percent of a run more accurately than power-law extrapolation of the loss curve, and the gap is largest exactly where scaling laws are least reliable, which is at architecture or data-distribution changes.

#### Strongest objection

It may be that the trajectory contains little transferable information beyond the scalars that scaling laws already extract. Optimization paths are high-dimensional, stochastic and architecture-specific, and the useful signal may genuinely be low-dimensional. If so, this cell is correctly empty and scaling laws are already the right compression. This is the candidate most likely to be a dead end, and it is included because the cost of checking is low.

---

### 3.5 Honest joint MDL at scale

> **Coordinates:** Axis 2. Closes an acknowledged gap in the frame's own accounting.

#### The observation

The parent document concedes, and then moves past, a significant inconsistency. Bits-per-token figures exclude the cost of the model. A fair minimum description length accounting must include the cost of transmitting the model as well as the data given the model:

```
L(M) + L(D | M)
```

Current practice reports only the second term. The justification is amortization: for a corpus far larger than the model, the per-token model cost approaches zero. That justification is correct and it is also a way of never having to compute the first term.

#### What the paradigm would be

Training in which model cost is a real term in the objective, denominated in the same units as data cost, rather than a footnote.

To be clear about what this is not: weight decay is a proxy with no units and no interpretation as a code length. This would require an actual description length for the parameters, which requires an actual prior over them.

#### What it would buy

A principled answer to a question currently answered empirically: how large should this model be for this corpus?

Compute-optimal scaling laws are fitted curves. They work, they are useful, and no theory explains their exponents. A joint MDL objective would either predict those exponents, which would be a substantial result, or fail to and thereby locate what else is going on.

#### Prior art, stated fairly

This is the oldest idea in the document and the most thoroughly abandoned.

- **Hinton and van Camp (1993)**, "Keeping Neural Networks Simple by Minimizing the Description Length of the Weights," is precisely this proposal. It predates the modern field entirely.
- **Variational bounds on network description length** formalize it.
- **PAC-Bayes compression bounds** produce non-vacuous generalization guarantees for small networks by explicitly accounting for model code length. These are among the only non-vacuous bounds in deep learning and they work by taking the model cost seriously.
- **Bayesian neural networks** are the same idea under a different name.

None of it survived contact with scale.

#### Why the cell is empty, and why the reason may be fatal

Computing a meaningful code length for billions of parameters requires a prior over parameters that someone has examined. The frame's own treatment of Bayesian methods notes that for large networks this prior is essentially unexaminable, so the object doing the regularizing is something nobody has inspected.

This is a real blocker, not an oversight, and it is the reason this candidate might belong in Section 4. It is retained in Section 3 because the payoff, a theory of model sizing rather than a fitted curve, is large enough to justify attempting it even at low probability, and because the PAC-Bayes results show the accounting is not impossible in principle.

#### Falsifiable prediction

A joint MDL objective, evaluated on models small enough for the accounting to be tractable, reproduces the compute-optimal parameter-to-token ratio without that ratio being fitted. If it produces a different ratio, the accounting is missing something identifiable.

#### Strongest objection

The amortization argument may simply be right. If the model cost genuinely is negligible per token at the scales that matter, then the first term contributes nothing and the current practice of ignoring it is not sloppiness but a correct approximation.

---

### 3.6 Forgetting as an objective

> **Coordinates:** Axis 2. Rate-distortion applied to removal rather than encoding.

#### The observation

Rate-distortion theory governs what to discard **at encoding time**. Every lossy method in the taxonomy sits somewhere on that curve.

Nothing governs what to remove **after encoding**. Unlearning exists, but as a patch applied to a trained model rather than as a property the training objective maintains.

#### What the paradigm would be

Training that treats removability as a first-class property. Knowledge encoded such that a specified subset can be excised with:

1. **Bounded collateral damage.** Removing one fact does not degrade unrelated capability.
2. **Verifiable completeness.** The fact is demonstrably absent, not merely suppressed.
3. **Cost far below retraining.** Otherwise the existing answer, retrain from scratch, is sufficient.

The third is achievable today by architectural means. The first two are open.

#### Why economics forces this one

Of all nine candidates, this is the one whose arrival is least dependent on the research being ready.

The taxonomy notes that retrieval gets deletability for free and parametric memory does not. Deletion rights under data protection law, copyright claims over training data, and safety-motivated removal of specific capabilities are all demands for an operation that the parametric paradigm structurally cannot perform.

The current options are retraining, which is prohibitively expensive, or moving the affected knowledge to retrieval, which surrenders the benefits of compression. Neither is a satisfactory answer to a legal requirement, and legal requirements do not wait for satisfactory answers.

#### Prior art, stated fairly

- **Machine unlearning** is a real and active field. Most methods are post-hoc approximations with no completeness guarantee.
- **SISA** achieves exact unlearning by sharding training data so that only affected shards need retraining. This works and it is an architectural constraint imposed up front, which is the closest existing thing to the candidate. Its cost is that sharding hurts performance.
- **Influence functions** estimate the effect of removing a training point. Useful for attribution, unreliable for deletion.
- **Model editing** (ROME, MEMIT) modifies specific factual associations. Demonstrably capable of changing outputs; not demonstrably capable of removing information.

#### The hard part

Verification. Demonstrating that information is absent rather than latent and recoverable under a different prompt is an open problem with no accepted definition, let alone an accepted test.

This matters more than it might appear. A deletion method that cannot be verified is not a deletion method; it is a suppression method with a deletion label, and under a legal standard the distinction is the entire question.

#### Falsifiable prediction

A training objective with an explicit removability term achieves verified deletion at a small fraction of retraining cost, with measurable and bounded degradation on held-out capability. The prediction fails if removability cost scales with model size in the same way retraining does, in which case the architectural approach is the only viable one.

#### Strongest objection

Information in a distributed representation may not be localized enough for removal to be well-defined. If a fact is encoded diffusely across the whole network, "removing it" may have no coherent meaning, and the only honest answers are retraining or retrieval.

---

### 3.7 Decompilation of weights into programs

> **Coordinates:** Best information source (a trained teacher) combined with best compression target (a program). The combination is unoccupied.

#### The observation

The taxonomy contains two facts that sit next to each other without ever being combined.

- **Program synthesis** compresses data into a program. The target has excellent properties: interpretable, verifiable, editable, extrapolates correctly outside the training range. It is hard because the search space is discrete and combinatorial.
- **Distillation** compresses a teacher into a student. The source has excellent properties: cheap to query, unlimited, provides full output distributions rather than hard labels.

Nothing compresses a **network into a program**. Distillation always produces another network.

#### What the paradigm would be

Distillation whose student is symbolic. The target is not a smaller network but an inspectable, verifiable artifact.

The reason to expect this is easier than program synthesis from data: the teacher provides an unlimited stream of input-output pairs, gradients, and internal activations. Program synthesis is hard largely because the search is under-constrained by a small number of examples. A trained network is an oracle that can be queried arbitrarily, which is a fundamentally better-posed search problem.

#### Prior art, stated fairly

- **Mechanistic interpretability** does exactly this, manually, at small scale. The circuits it extracts, induction heads, indirect object identification, modular arithmetic algorithms, are precisely such programs. The field has demonstrated that the target exists.
- **Symbolic regression distillation** works for small physics-informed networks and recovers governing equations.
- **Decision tree and rule extraction** from neural networks is an old literature that produced approximations rather than faithful translations.
- **Sparse autoencoders and dictionary learning** decompose activations into interpretable features, which is a step toward the representation a program would be written over.

The gap: interpretability treats this as scientific investigation conducted by humans. Nothing poses it as a learning paradigm with an objective, a benchmark and a scaling story.

#### Why it probably stays partial, and why that is the interesting version

Most of what a large network computes is likely not expressible as a short program. This is the Kolmogorov argument applied to weights rather than data: if the function were compressible to a short program, it would be surprising that gradient descent needed billions of parameters to represent it.

So full decompilation is not the target. The useful version is **selective**:

> Decompile the fraction that is programmatic, leave the rest neural, and know which is which.

That hybrid target is more plausible than full decompilation, more useful than either pure approach, and nobody is aiming at it. Knowing which parts of a model implement clean algorithms and which implement irreducible statistical association would be valuable even if the extracted fraction were small, because the two failure modes are completely different.

#### Falsifiable prediction

For tasks with known algorithmic structure, automated decompilation recovers the algorithm at a scale substantially beyond what manual interpretability has reached, and the fraction of a network that decompiles cleanly correlates with how algorithmic the task is. The candidate fails if the decompilable fraction is uniformly negligible across tasks.

#### Strongest objection

Extracted programs may be faithful on the training distribution and wrong off it, which is the same failure that made rule extraction unsatisfying in the 1990s. Verification of faithfulness is as hard as the original interpretability problem.

---

### 3.8 Continuous amortization

> **Coordinates:** Axis 3. The timing axis currently has three discrete settings; this is the continuous limit.

#### The observation

Axis 3 has three values: compress at training time, defer to inference, or alternate between them. The third is the frontier practice, and its period is measured in months and gated by a human deciding when to start a training run.

Nothing about the mathematics requires that period. The search-compress loop is a loop; its frequency is a free parameter that has been set by organizational convenience rather than by analysis.

#### What the paradigm would be

No boundary between training and deployment. Each interaction is evaluated for whether compressing it into weights pays for itself against the cost of re-deriving it on demand. Reasoning that is frequently re-derived gets consolidated; one-off work does not.

The decision rule is an economic one: consolidate when expected future re-derivation cost exceeds consolidation cost. That is a computable quantity given a query distribution, which links this candidate to 3.1.

#### Prior art, stated fairly

- **Continual and lifelong learning** address the stability-plasticity problem, which is a prerequisite, but treat consolidation as something to be done safely rather than as an economic decision to be optimized.
- **Complementary learning systems theory** in neuroscience is the conceptual ancestor: fast hippocampal episodic storage, slow neocortical consolidation during sleep. The biological system solved the timing problem; the machine learning analogue borrowed the architecture without the decision rule.
- **Cache-augmented and episodic memory systems** consolidate mechanically on a fixed schedule.
- **Online learning** updates continuously but without the amortization decision.

#### Obstacles, and one that is underappreciated

The obvious obstacle is catastrophic forgetting, which is well studied and partially addressed.

The less obvious and more serious obstacle is **evaluation**. A model that differs from itself hour to hour cannot be tested by any current practice. Benchmarks assume a fixed artifact. Safety evaluation assumes the thing evaluated is the thing deployed. A continuously consolidating system breaks both assumptions, and the infrastructure to evaluate a moving target does not exist.

This is worth stating plainly because it suggests the candidate's real blocker is not the learning algorithm. It is that nobody knows how to certify something that changes.

#### Falsifiable prediction

A system with economically-triggered consolidation achieves lower total cost, counting both training compute and inference compute, than either a fixed model or a pure retrieval system on a workload with realistic query repetition. The prediction fails if consolidation overhead exceeds the re-derivation it saves, which depends entirely on how repetitive real workloads are.

#### Strongest objection

Catastrophic forgetting may make the consolidation step unreliable enough that its expected value is negative regardless of the economics. And if evaluation of a continuously-changing system is genuinely impossible rather than merely hard, the candidate is undeployable even if it works.

---

### 3.9 Explicit inductive bias accounting

> **Coordinates:** A measurement standard rather than a training method. Addresses the admitted failure that inductive bias is invisible.

#### The observation

Architecture, augmentation policy and prior encode substantial information that never appears in any bits-per-sample table. Choosing a convolution over a dense layer supplies a large amount of information about the structure of images, free, from a human.

The frame admits this and does not act on it. The consequence is that sample-efficiency claims are systematically inflated by an unmeasured human contribution.

#### The clearest instance

The taxonomy's own treatment of contrastive learning states that the augmentation policy is the real ceiling on the method, and that these methods work well on natural images because the community found good augmentations for natural images.

That is a large human contribution, developed over years, presented as a property of the algorithm. When the same method transfers poorly to a domain where the correct invariances are unknown, this is reported as a limitation of the domain rather than as evidence about where the performance came from.

The same critique applies to most claimed architectural advances, and it is the main reason results fail to replicate outside the setting their inductive bias was tuned for.

#### What the paradigm would be

Measure it. Express architectural and augmentation choices in bits, so that a sample-efficiency claim can be audited against the information smuggled in through design.

Concretely: what is the description length of the constraint "use a convolution" relative to a dense layer? What is the information content of an augmentation policy, measured as the reduction in the hypothesis space it induces?

#### Prior art, stated fairly

- **PAC-Bayes** treats the prior as a code and is the correct formalism. It is applied to weights, not to architecture choice.
- **Occam bounds and structural risk minimization** price hypothesis class complexity, in a form too coarse to distinguish architectures.
- **Neural architecture search** implicitly explores the space but does not price positions in it.

Nobody has an accepted way to price a convolution in bits. That is the missing piece and it is not obviously impossible.

#### Why this one arrives differently

This is the only candidate that is an **evaluation standard** rather than a training method. It will therefore arrive through a different mechanism: not because someone invents it, but because reviewers and practitioners get tired of results that do not survive contact with a new domain, and demand the accounting.

That makes it the most likely of the nine to actually happen, and the least likely to be attributed to anyone.

#### Falsifiable prediction

A measure of architectural information content, applied retrospectively, explains a substantial fraction of the variance in how well published methods transfer across domains. Methods with high measured bias transfer worse, and the relationship is strong enough to be predictive.

#### Strongest objection

Description length is defined relative to a universal machine, and for architecture the choice of reference machine is arbitrary in a way that may dominate the measurement. If the ranking of architectures by information content depends on the encoding scheme, the measure is not well-defined enough to audit anything.

---

## 4. Cells That Should Stay Empty

Honesty requires this section. Not every gap is an opportunity, and a taxonomy that only generates optimism is not being used properly.

### 4.1 Own generations as a genuine information source

The occupancy table marks this cell as costing zero and being worth zero. No recursion recovers information that was not present.

Synthetic data does work in practice, and the reason is instructive: it works when a **verifier filters it**. In that case the verifier is the information source and the generator is only a proposal mechanism. The bits come from the filter, not from the generation.

Any paradigm claiming gains from self-generated data without a filter is misattributing its gains, usually to regularization or curriculum effects that would be better obtained directly. **Model collapse** under recursive training on synthetic data is the empirical confirmation: each round loses tail behaviour, because each round is a lossy re-encoding of the same information.

This applies directly to cooperative generative-discriminative frameworks and to most of the current enthusiasm for synthetic pretraining corpora.

### 4.2 Reinforcement learning as a primary training signal

The bits-per-sample arithmetic is not a temporary engineering limitation to be overcome by better algorithms. A scalar reward over a long trajectory is a narrow channel by construction, and credit assignment across that channel is the difficulty, not the implementation.

RL will remain what it currently is: a fine-tuning and search method layered on a prior built by high-bandwidth compression. Proposals to train foundation models primarily by RL are proposing to replace a 10^13-bit information source with a 10^5-bit one.

### 4.3 Adversarial objectives, revived

The lesson from GANs is structural, not historical. An objective that cannot state what it compresses cannot be evaluated in any principled way, and a field that cannot evaluate progress cannot distinguish it from noise. Years of GAN research lacked a trustworthy yardstick, which was a predictable consequence of abandoning the likelihood.

Any new paradigm whose objective has no information-theoretic interpretation should be treated with the same suspicion, and the first question asked of it should be how anyone will know whether it is working.

### 4.4 Posterior compression at frontier scale

The blocker is real and unchanged: nobody can specify or inspect a meaningful prior over billions of parameters. Variational approximations underestimate uncertainty in known, systematic ways.

This matters most because the applications motivating Bayesian deep learning are safety applications, and an uncertainty estimate that is systematically overconfident in an unknown pattern is worse than no uncertainty estimate at all, since it invites reliance it cannot support.

Note the tension with Section 3.5, which shares this blocker. The difference is that 3.5 needs the accounting to be approximately right to produce a useful scaling prediction, whereas safety applications need calibration to be reliably right, which is a much higher bar.

---

## 5. Prioritization

### 5.1 By expected value

**First: verifier synthesis (3.2).** It is the only candidate that adds information rather than rearranging it. Verifiers are identified by the taxonomy as the uniquely free source. Every domain with fast progress had one by accident. The research question, which domains possess an unexploited generator-verifier asymmetry, has not been asked systematically. If it works in one open-ended domain, that domain becomes compute-bound, and compute scales.

**Second: forgetting as an objective (3.6).** Not because it is intellectually deepest but because regulation, copyright and safety requirements will force it on a schedule unrelated to research readiness. Work that happens under deadline pressure without preparation is done badly. This one will happen; the only question is whether anyone prepared.

**Third: learned tier allocation (3.1).** The cleanest empty cell, a well-posed optimization problem, mature components on both sides, and a decision currently made thousands of times by human judgement without measurement.

### 5.2 By likelihood of happening regardless

**Inductive bias accounting (3.9)** arrives as an evaluation norm, driven by reviewer fatigue rather than by anyone's research agenda.

**Continuous amortization (3.8)** arrives incrementally, as caching and consolidation systems gradually acquire better trigger conditions, and is unlikely to be recognized as a paradigm shift while it happens.

### 5.3 By risk of being a dead end

**Trajectory compression (3.4)** is most likely to find that scaling laws already extract the transferable signal. Cheap to check, which is why it is worth checking.

**Joint MDL (3.5)** has a blocker that may be fatal and shares it with a Section 4 entry. Included for the size of the payoff, not the probability.

### 5.4 The dependency structure

Two pairs are coupled and worth noting.

- **3.1 and 3.8** share a subproblem: both require estimating future access patterns to decide what to consolidate and where. Progress on either helps the other.
- **3.2 and 3.3** are both about improving information acquisition and both face the reliability problem of learned evaluators. A better learned verifier is also a better value estimator.

---

## 6. How to Evaluate a Claimed New Paradigm

The taxonomy's most immediately useful output is a checklist. Six questions, applicable to any method presenting itself as new.

**1. What information does it add?**
If the answer is "none", the best case is a more useful redistribution of existing bits. That is sometimes worth doing and is never a large effect. This question alone disposes of most synthetic-data claims.

**2. Where does that information come from, and what does it cost per bit?**
Locate it on Axis 1. A method that requires human annotation has a ceiling that scales with money. A method that exploits a verifier has a ceiling that scales with compute. These are not comparable.

**3. What does it compress into, and is that target chosen or inherited?**
Most methods inherit "weights" without considering alternatives. A method that examined the choice and chose weights deliberately is in a different position from one that never asked.

**4. When is the compression paid for?**
Training time, inference time, or both in a loop. This determines deployment economics and is frequently unstated in papers.

**5. Can it state what it compresses in a way that yields an evaluation metric?**
If not, apply the GAN precedent. A field that cannot measure progress will produce years of activity and no accumulation.

**6. How much of the result is inductive bias that was not counted?**
Ask what happens in a domain where the relevant invariances are unknown. If the method has only been demonstrated where the community spent years tuning the priors, the result is about the priors.

Applied together, these questions separate genuine new cells from relabelled old ones fairly reliably.

---

## 7. Limits of This Exercise

Five caveats, stated because the exercise is easy to oversell.

**The taxonomy is not the periodic table.** Its axes are categorical, not numeric. Empty cells do not come with predicted properties the way gallium's density did. What the frame gives is a reason to ask a question, not an answer to it.

**Prediction from a frame is weaker than it appears.** The frame was constructed after the fact to describe existing methods. A structure fitted to existing data will contain apparent gaps that are artifacts of the fitting. Some of the nine candidates are likely such artifacts.

**Novelty claims are the weakest part.** Every candidate here has prior art, and in two cases (3.3, 3.5) the prior art is decades old and the claim reduces to "this was abandoned prematurely". That is a much weaker claim than "this does not exist", and it is the honest one.

**Real paradigms have historically come from elsewhere.** Backpropagation, attention, and diffusion did not come from taxonomic reasoning. They came from an analogy, an engineering need, and a physics connection respectively, and were rationalized into the frame afterwards. The frame's own admission that it is descriptive rather than generative should be taken seriously, and this document is an attempt to test that admission rather than a refutation of it.

**The most important missing paradigm is probably not on this list.** By construction, this exercise can only find gaps that the existing axes make visible. A genuinely new paradigm would likely require a fourth axis that nobody has identified, and it would appear in this frame as something that does not fit rather than as an empty cell.

---

## 8. Summary Table

| # | Candidate | Axis | Source of prediction | Prior art status | Likelihood | Payoff if right |
|---|---|---|---|---|---|---|
| 3.1 | Learned tier allocation | 2 | Empty cell: target never learned | Adjacent, none direct | Medium | High |
| 3.2 | Verifier synthesis | 1 | Empty cell: source never constructed | Uses exist, construction unstudied | Medium | Very high |
| 3.3 | Value-weighted acquisition | 1 | Admitted failure: bits not fungible | Decades old, never scaled | Medium | High |
| 3.4 | Trajectory compression | 2 | Admitted failure: optimization unmodelled | Partial, very lossy | Low | Medium |
| 3.5 | Joint MDL at scale | 2 | Frame's own accounting gap | 1993, abandoned | Low | High |
| 3.6 | Forgetting as objective | 2 | Empty cell: removal untheorized | Post-hoc only | High, forced | High |
| 3.7 | Weights to programs | 1 + 2 | Best source never met best target | Manual, small scale | Medium | High |
| 3.8 | Continuous amortization | 3 | Empty cell: loop period arbitrary | Mechanical schedules only | Medium | Medium |
| 3.9 | Inductive bias accounting | Meta | Admitted failure: bias uncounted | Formalism exists, unapplied | High, gradual | Medium |

**Excluded deliberately:** self-generated data as an information source, RL as a primary training signal, revived adversarial objectives, posterior compression at frontier scale. Reasons in Section 4.

---

## 9. References

**The frame**

- Shannon, C. E. (1948). A Mathematical Theory of Communication.
- Rissanen, J. (1978). Modeling by Shortest Data Description.
- Grünwald, P. (2007). The Minimum Description Length Principle.
- Tishby, N., Pereira, F. and Bialek, C. (1999). The Information Bottleneck Method.
- Delétang et al. (2023). Language Modeling Is Compression.

**Relevant to 3.1, memory tiers**

- Khandelwal et al. (2019). Generalization through Memorization: Nearest Neighbor Language Models.
- Lewis et al. (2020). Retrieval-Augmented Generation.
- Lample et al. (2019). Large Memory Layers with Product Keys.
- Graves et al. (2016). Hybrid Computing Using a Neural Network with Dynamic External Memory.

**Relevant to 3.2, verifiers**

- Silver et al. (2017). Mastering the Game of Go without Human Knowledge.
- Lightman et al. (2023). Let's Verify Step by Step. Process reward models.
- Irving, Christiano and Amodei (2018). AI Safety via Debate.
- Gao, L., Schulman, J. and Hilton, J. (2022). Scaling Laws for Reward Model Overoptimization.

**Relevant to 3.3, acquisition**

- Lindley, D. V. (1956). On a Measure of the Information Provided by an Experiment.
- Houlsby et al. (2011). Bayesian Active Learning for Classification and Preference Learning.
- Koh, P. W. and Liang, P. (2017). Understanding Black-box Predictions via Influence Functions.

**Relevant to 3.4, trajectories**

- Andrychowicz et al. (2016). Learning to Learn by Gradient Descent by Gradient Descent.
- Kaplan et al. (2020). Scaling Laws for Neural Language Models.
- Garipov et al. (2018). Loss Surfaces, Mode Connectivity, and Fast Ensembling.
- Wortsman et al. (2022). Model Soups.

**Relevant to 3.5, MDL**

- Hinton, G. and van Camp, D. (1993). Keeping Neural Networks Simple by Minimizing the Description Length of the Weights.
- Zhou et al. (2019). Non-Vacuous Generalization Bounds at the ImageNet Scale: A PAC-Bayesian Compression Approach.
- Hoffmann et al. (2022). Training Compute-Optimal Large Language Models.

**Relevant to 3.6, forgetting**

- Bourtoule et al. (2019). Machine Unlearning. The SISA approach.
- Meng et al. (2022). Locating and Editing Factual Associations in GPT. ROME.
- Meng et al. (2022). Mass-Editing Memory in a Transformer. MEMIT.

**Relevant to 3.7, decompilation**

- Elhage et al. (2021). A Mathematical Framework for Transformer Circuits.
- Olsson et al. (2022). In-context Learning and Induction Heads.
- Cranmer et al. (2020). Discovering Symbolic Models from Deep Learning with Inductive Biases.
- Bricken et al. (2023). Towards Monosemanticity. Sparse autoencoders.

**Relevant to 3.8, consolidation**

- McClelland, McNaughton and O'Reilly (1995). Why There Are Complementary Learning Systems in the Hippocampus and Neocortex.
- Kirkpatrick et al. (2017). Overcoming Catastrophic Forgetting in Neural Networks.

**Relevant to 4.1, model collapse**

- Shumailov et al. (2023). The Curse of Recursion: Training on Generated Data Makes Models Forget.