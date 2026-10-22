# The Empty Cells

Oct 19, 2026 · @Nikolas Markou

### Learning Paradigms That Do Not Yet Exist, Derived From an Information-Theoretic Taxonomy

*A companion piece to "Learning Paradigms Through the Lens of Information", written to stand alone. Where that document maps what exists, this one uses the same map to locate what is missing, states which gaps are opportunities and which are correctly empty, and gives falsifiable predictions for each.*

## Table of Contents

1. Purpose and Method
   1. What this document is doing
   2. The taxonomy, restated for self-containment
   3. Two sources of prediction
   4. Standard of evidence
   5. Shannon information versus usable information
2. The Occupancy Map
   1. What the existing paradigms cover
   2. The source by target grid
   3. Visible holes
   4. The admitted failures of the frame
   5. Self-supervised learning: present by assumption, absent by name
3. Candidate Paradigms
   1. Learned allocation across memory tiers
   2. Verifier synthesis
   3. Value-weighted acquisition
   4. Compression of the training trajectory
   5. Honest joint MDL at scale
   6. Forgetting as an objective
   7. Decompilation of weights into programs
   8. Continuous amortization
   9. Explicit inductive bias accounting
   10. Learned pretext tasks
4. Cells That Should Stay Empty
5. Prioritization
6. How to Evaluate a Claimed New Paradigm
7. Limits of This Exercise
8. Summary Table
9. References

## 1. Purpose and Method

### 1.1 What this document is doing

A taxonomy earns its keep in two ways. The first is descriptive: it explains why the things that exist behave as they do. The second is generative: the structure of the classification makes visible the positions that nothing currently occupies.

The periodic table is the standard example, and it is worth being precise about why it worked. It did not predict new elements because it was elegant. It predicted them because it had **axes with known values** and **cells that were empty while their neighbours were full**. Mendeleev could specify gallium's atomic weight and density before anyone found it, because the cell's coordinates constrained the properties.

The taxonomy below is far weaker than that. Its axes are categorical rather than numeric, its cells are not equally spaced, and the analogy should not be pushed. But the basic move survives: if a classification has real axes, an empty cell whose neighbours are all occupied is a question worth asking, and sometimes the coordinates constrain what would go there.

What follows is that exercise, conducted honestly. Ten candidate paradigms, each with its prior art stated plainly rather than suppressed to make the idea look newer, and a separate list of cells that are empty for good reasons and should remain so.

### 1.2 The taxonomy, restated for self-containment

The frame classifies any learning method by three questions.

**Axis 1: Where does the information come from?**

| Source | Occupied by | Human cost per bit | Compute cost per bit | Ceiling |
| --- | --- | --- | --- | --- |
| Raw corpus, via self-supervision | Autoregressive, masked, contrastive and joint-embedding pretraining; VAEs; diffusion | Near zero | Low | Corpus size, filtered by what the pretext task exposes (2.5) |
| Human labels | Supervised learning | High | Low | Annotation budget |
| Human preferences | RLHF, DPO | Very high, at most one bit per comparison | Low | Annotation budget |
| Teacher model | Distillation | Near zero, sunk into the teacher | Low | Teacher quality |
| Environment reward | Reinforcement learning | Front-loaded, in building the simulator | Very high per usable bit | Simulation cost and fidelity |
| Formal verifier | Self-play, RL on code and mathematics | Near zero once the verifier exists | High, spent on search | Domain coverage |
| Own generations | Synthetic data, self-training, self-distillation | Zero | Moderate | Zero Shannon information; usable information only through added computation or a filter (1.5) |

Cost is split into two columns because a method bounded by human cost scales with budget, a method bounded by compute scales with hardware, and the two are not comparable (Section 6, question 2).

**Axis 2: What is the information compressed into?**

| Target | Occupied by | Properties |
| --- | --- | --- |
| Weights | Most of the field | Opaque, unupdatable, non-attributable, non-deletable |
| Explicit program | Program synthesis, symbolic regression | Interpretable, verifiable, extrapolates, hard to find |
| Policy | Reinforcement learning | Behaviour rather than knowledge |
| Posterior | Bayesian methods | Uncertainty as a first-class object |
| External index | Retrieval, k-NN, RAG | Updatable, attributable, deletable, does not generalize |

Context windows and cached computation are transient targets. They do not appear as columns, but they enter as tiers in 3.1.

**Axis 3: When is the compression paid for?**

| Timing | Occupied by | Economics |
| --- | --- | --- |
| Training time | Standard training | Cheap inference, fixed capability |
| Inference time | Retrieval, in-context learning, test-time search | Expensive per query, adapts per query |
| Alternating loop | Frontier system development | Search, distil, search again from a better start |

Three supporting facts from the parent document are load-bearing below.

**The bits-per-sample table.** The information content of one training signal predicts much of the sample-efficiency ordering in machine learning, though not all of it. The figures are upper bounds on the information in the label. What a sample actually conveys to the model is its realized surprisal, which falls as the model improves.

| Signal | Upper bound per sample | Note |
| --- | --- | --- |
| Teacher soft labels | More than the hard label on the same input | Full output distribution, bounded by numeric precision and teacher entropy |
| Next-token prediction | log2(V) per token, about 15 to 18 bits for vocabularies of 32k to 256k | A hard label with K = V. Realized signal is the model's surprisal, a few bits per token late in training |
| Masked-token prediction (BERT) | log2(V) per masked token, on about 15% of positions | Loss only on masked positions |
| Hard label, K classes | log2(K) |  |
| Contrastive (InfoNCE) | log2(N), N = candidates per batch | The mutual information estimate cannot exceed log N |
| Replaced-token detection (ELECTRA) | 1 bit per token, on every position | More sample-efficient than masked prediction despite fewer nominal bits per position |
| Human preference comparison | 1 |  |
| Scalar RL reward at episode end | Often well under 1 usable |  |
| Joint-embedding prediction (BYOL, DINO, I-JEPA) | Undefined | No likelihood; regression to a learned target |

The ELECTRA row is a counterexample to pure bit counting inside a single family: placement beats volume, which is the frame's own admission 3 (Section 1.3).

**The elicitation argument.** A pretraining corpus of about 15 trillion tokens carries on the order of 10^13 to 10^14 bits, depending on whether it is counted compressed or raw. A preference dataset of a hundred thousand comparisons carries 10^5. Preference-based post-training therefore cannot be installing new capability; it locates and amplifies modes the prior already supports.

The argument does not extend to RL against a verifier. There the information comes from the verifier and the compute spent searching against it, not from a fixed annotation set, so the 10^5 bound does not apply. Whether verifier RL installs or only elicits is contested: Yue et al. (2025) find base models match or exceed RL-tuned models at large pass@k, which is evidence for elicitation at current RL compute. The reciprocity argument below predicts that result changes as search compute grows.

**The reciprocity of search and compression.** Distillation and RL compress an expensive search procedure into a cheap policy. Test-time search decompresses a cheap policy back into expensive deliberation. Current systems run both directions in a loop, and that loop is the only known route past the finite-corpus ceiling, because search converts compute into usable information. In the strict Shannon sense it creates none; Section 1.5 resolves the difference.

### 1.3 Two sources of prediction

**Source one: the structure of the axes.** Seven information sources, five compression targets, three timings: 105 coordinates. Section 2.2 lays out the 35-cell source by target grid explicitly so the occupancy claim can be checked rather than asserted.

Only one candidate (3.7) is a literal empty cell in the cross product. The rest come from structural holes: an axis that is never learned (3.1, 3.10), a source that is never constructed (3.2), an object that is never compressed (3.4), a quantity that is never priced (3.5, 3.9), or an operation the axes do not name (3.6). These are better described as properties of the axes than as cells, and the summary table labels them that way.

The filter for "worth pursuing" is that the neighbouring cells, or the neighbouring structure, should be occupied and productive.

**Source two: the frame's own admitted failures.** The parent document lists what the information frame cannot account for. Each admission is a candidate location for a missing paradigm rather than merely a caveat, on the reasoning that a frame usually fails to explain something because the thing that would explain it has not been built.

The five admissions:

1. Compression does not explain generalization on its own.
2. Optimization dynamics are not accounted for.
3. Bits are not fungible; placement matters more than volume.
4. Inductive bias is invisible in the accounting.
5. The frame is descriptive rather than generative.

Items 2, 3 and 4 generate candidates below (3.4, 3.3 and 3.9). Item 1 is a genuine limit of the frame rather than a missing method. Item 5 is the concession that this entire document is working against.

A sixth failure, not admitted by the parent: the frame uses two incompatible notions of information. It is treated in 1.5 and shapes the verdicts in 4.1 and 4.2.

### 1.4 Standard of evidence

Every candidate below is stated with:

- **The coordinates.** Which cell or structural property it addresses.
- **Prior art, stated fairly.** Nearly every idea here has partial precedent. Suppressing it would make the predictions look stronger and be dishonest. Where prior art already fills part of the cell, the claim is narrowed to the unfilled part.
- **Why the cell is empty.** Distinguishing "nobody looked" from "somebody looked and it was hard" from "it is hard for a reason that has not changed".
- **A falsifiable prediction.** Something that could be checked, against a named baseline, and that would be wrong if the candidate is a bad idea.
- **The strongest objection.** Where the candidate is most likely to fail.

Candidates that cannot meet this standard belong in Section 4, not Section 3.

### 1.5 Shannon information versus usable information

The frame uses "information" in two senses, and several of its verdicts depend on which one is meant.

**Shannon information.** By the data processing inequality, no function of a model and its inputs creates information about the world. In this sense own generations are worth zero, and so is any search that consults nothing outside the model.

**Usable information.** A learner with bounded compute cannot extract all the Shannon information present in its data. Computation can convert latent information into a form the learner can use. Xu et al. (2020) formalize this as V-information: information usable by a given model family, which, unlike Shannon information, can increase under computation.

The frame's central claims need the second sense:

- **Search "manufactures information".** False in the Shannon sense, true in the usable sense. The rules of Go fit in a few kilobytes; AlphaGo Zero's play is the usable information that compute extracted from them.
- **Self-distillation improves models.** Born-again networks and DINO gain accuracy from their own outputs with no new Shannon information and no verifier. Usable information explains them directly.
- **Distilled chain-of-thought helps.** A model's reasoning trace, distilled back, transfers the result of computation the student cannot perform in one forward pass.

Convention: the bits tables report Shannon upper bounds; "usable information" is used wherever computation changes the answer. Section 7 anticipates that a genuinely new paradigm may need a fourth axis. Computation, meaning how much compute stands between the information and the learner, is the strongest candidate.

## 2. The Occupancy Map

### 2.1 What the existing paradigms cover

Reading the three axes together, the occupied region has a shape worth naming.

**Almost everything compresses into weights.** Programs, policies and posteriors are occupied, but weights dominate by an enormous margin. Retrieval is the main exception, and it is usually bolted on rather than designed in.

**Most methods have a fixed information source.** The source is chosen before training and does not change. Two families treat acquisition itself as an object of optimization. Active learning maximizes expected information gain, measured in bits. Pretraining data selection optimizes downstream value directly: domain reweighting (DoReMi), learned quality filters (FineWeb-Edu, DCLM) and token-level selection (Rho-1). Data selection works at scale, and it is the strongest existing evidence for the thesis of 3.3.

**Timing has only recently become a design variable.** Until inference-time search became prominent, Axis 3 had effectively one setting. That it turned out to be a primary architectural decision rather than a deployment detail is the clearest recent evidence that the axes are real and that unexamined axes hide consequential choices.

### 2.2 The source by target grid

The 35 cells of Axis 1 by Axis 2, marked by state. Timing is omitted; it multiplies the grid without changing which cells are empty.

| Source / Target | Weights | Program | Policy | Posterior | External index |
| --- | --- | --- | --- | --- | --- |
| Raw corpus (SSL) | Dense: pretraining | Sparse: grammar induction | Partial: pseudo-labelled video (VPT) | Abandoned at scale: topic models, nonparametrics | Dense: retrieval datastores, kNN-LM |
| Human labels | Dense: supervised learning | Niche: programming by example (FlashFill) | Dense: behaviour cloning | Small scale: GPs, BNNs | Occupied: k-NN classifiers |
| Human preferences | Dense: reward models, DPO | Sparse: preferences compressed into rules (Inverse Constitutional AI) | Dense: RLHF | Sparse: preferential Bayesian optimization | Empty |
| Teacher model | Dense: distillation | Toy scale only (MIPS); empty at scale (3.7) | Occupied: policy distillation | Sparse: Bayesian dark knowledge | Sparse: cached teacher outputs |
| Environment reward | Occupied: value functions | Sparse: programmatic policies, Voyager skill library | Dense: RL | Small scale: Bayesian RL, posterior sampling | Abandoned: episodic control |
| Formal verifier | Dense, growing: RLVR, AlphaZero | Active, growing: FunSearch, AlphaEvolve | Dense: self-play | Empty | Sparse: premise retrieval in theorem proving |
| Own generations | Occupied, contested: self-training, self-distillation | Empty | Occupied: self-imitation learning | Empty, correctly (4.1) | Sparse: reflection memories (Reflexion) |

Three readings follow.

1. **Truly empty cells are rare.** Five of 35 are empty; most "gaps" are cells occupied at toy or small scale. The opportunity is usually scale rather than existence, which lowers every novelty claim in Section 3.
2. **The program column is the sparsest productive column, and its fastest-moving cell is verifier to program.** FunSearch and AlphaEvolve use a network as a proposal distribution and a verifier to select programs. A verifier is what makes programs findable, which links 3.2 and 3.7.
3. **The posterior column is mostly small scale or abandoned.** That is the pattern Section 4.4 explains.

### 2.3 Visible holes

Ten structural observations, each of which generates a candidate.

1. **Axis 2 is never learned at write time.** The compression target is a human design decision made once, at architecture time. Read-time routing between tiers exists; write-time placement does not. (→ 3.1)
2. **Verifiers are treated as a domain property, never as an artifact to be built.** The taxonomy identifies them as the uniquely free source and then takes their existence as given. (→ 3.2)
3. **Label acquisition optimizes bits, not value.** Active learning maximizes information gain, while the frame's own analysis of RLHF shows placement dominates volume. Data selection over free corpora already optimizes value; label acquisition does not. (→ 3.3)
4. **The training process is rarely compressed.** Data, teachers, policies and posteriors are compressed routinely. The trajectory through parameter space is mostly discarded. (→ 3.4)
5. **Model cost is excluded from the accounting it supposedly obeys.** Reported bits-per-token figures omit the parameters, which a fair MDL accounting must include. (→ 3.5)
6. **Rate-distortion governs encoding but not removal.** What to discard at encoding time is theorized; what to remove afterward is not, beyond the single-record level. (→ 3.6)
7. **The best source and the best target meet only at toy scale.** A trained teacher is a cheap, unlimited source; a program is the most useful target. Automated network-to-program distillation exists for small RNNs and nowhere else. (→ 3.7)
8. **The search-compress loop has a period of months and a human in it.** Nothing about the mathematics requires that. (→ 3.8)
9. **Human-supplied inductive bias is never priced in reported results.** (→ 3.9)
10. **The extraction rule for the free source is hand-designed.** Every self-supervised objective is a human choice of what to predict from what. (→ 3.10)

### 2.4 The admitted failures of the frame

| Failure | Candidate it generates | Section |
| --- | --- | --- |
| Optimization dynamics unaccounted for | Compression of the training trajectory | 3.4 |
| Bits are not fungible | Value-weighted acquisition | 3.3 |
| Inductive bias is invisible | Explicit inductive bias accounting | 3.9 |
| Shannon and usable information conflated | None; it is a correction to the frame, and it scopes 4.1 and 4.2 | 1.5 |
| Compression does not explain generalization | None. This is a real limit, not a gap. |  |
| The frame is descriptive, not generative | This document is the test of that claim. |  |

The generalization row deserves a note. The gap between compressing training data and generalizing to new data is bridged by assumptions about the data distribution and the hypothesis class. Standard learning-theoretic bounds are vacuous for over-parameterized networks. The exception is instructive: PAC-Bayes compression bounds are non-vacuous, including for large language models (Lotfi et al. 2024), and they work by pricing the model's description length. That is evidence for 3.5, not a fix for this row. No new paradigm closes the gap, and the honest response is to stop claiming the frame explains generalization.

### 2.5 Self-supervised learning: present by assumption, absent by name

**Verdict: self-supervised learning is not missing as an information source, but it is missing as a concept.** It is the mechanism behind the only row of Axis 1 that is both near free and near unlimited. Filing it under "raw corpus" without examining it hides four things.

**The pretext task is a human-supplied extraction rule.** The corpus provides inputs and targets, but a human decides which dependencies become targets: the next token, 15% of tokens masked at random, two augmented crops of one image. Each objective maximizes a different mutual information term, so the same corpus yields different representations. The ceiling of the corpus row is corpus size filtered by what the pretext task exposes, not corpus size alone.

**Objectives within the family differ widely in nominal bits, and nominal bits do not predict efficiency.** Next-token and masked prediction carry up to log2(V) per target; contrastive objectives cap at log2(N); joint-embedding methods have no likelihood at all. ELECTRA's one bit per token beats BERT's richer but sparser signal. Bits are not fungible even inside one family.

**SSL constrains two verdicts in Section 4.** Non-contrastive joint-embedding methods (BYOL, DINO, I-JEPA) have no likelihood and work, so the criterion in 4.3 cannot be an information-theoretic interpretation. DINO is self-distillation from an exponential moving average of itself, with no verifier, so 4.1 cannot treat all self-generated signal as worthless.

**The pretext task visibly matters at scale.** Models trained on "A is B" fail to infer "B is A" (the reversal curse, Berglund et al. 2023): next-token prediction leaves a symmetric dependency unlearned. Repeating data for up to about four epochs is nearly as good as fresh data, after which returns fall sharply (Muennighoff et al. 2023). When the corpus stops growing, the extraction rule becomes the main remaining lever.

One structural note. The extraction rule could be treated as its own axis, orthogonal to source: "predict the masked part" applies equally to text corpora, environment trajectories (world models) and a model's own generations. This document keeps it inside Axis 1 for simplicity. It generates candidate 3.10, and it also feeds 3.3 (data selection) and 3.9 (augmentations as unpriced bias).

## 3. Candidate Paradigms

### 3.1 Learned allocation across memory tiers

> **Coordinates:** Axis 2 becomes a learned variable at write time rather than a design constant.

#### The observation

Every system in the taxonomy has its compression target fixed in advance. An architect decides that knowledge goes in weights, that facts go in a retrieval index, or that a scratchpad holds intermediate state. That decision is then applied uniformly to everything the system learns.

But the tiers have genuinely different cost profiles, and the right tier differs per item:

| Tier | Write cost | Read cost | Mutable | Attributable | Generalizes |
| --- | --- | --- | --- | --- | --- |
| Weights | Very high; about 2 bits of knowledge per parameter at capacity | Near zero | No | No | Yes |
| External index | Near zero | Moderate, grows with size | Yes | Yes | No |
| Cached computation | Low | Low | Yes | Partly | No |
| Context window | Zero | Grows with length | Yes | Yes | Within prior |

The weights capacity figure is from Allen-Zhu and Li (2024), and it gives the write side a measurable exchange rate. A fact accessed constantly, stable over years, and needed for compositional reasoning belongs in weights. A fact accessed rarely, subject to revision, and needing citation belongs in an index. Nothing currently makes that determination per item at write time.

#### What the paradigm would be

Training that treats tier placement as part of the objective. For each piece of knowledge, the system estimates access frequency, volatility, attribution requirement and compositional utility, and allocates accordingly. The objective is a constrained bit-allocation problem across tiers with different cost functions, which is well posed and, as far as I can tell, unposed.

#### Prior art, stated fairly

- **Memory-augmented networks** and **product-key memory layers** learn what to store in a differentiable memory; Memory Layers at Scale (Berges et al. 2024) takes this to billions of parameters. None asks whether the differentiable memory is the right tier.
- **Mixture-of-experts** routes computation conditionally. Storage placement is not the routed variable.
- **kNN-LM** interpolates parametric and non-parametric predictions with a single scalar: the degenerate one-parameter version of this candidate.
- **Adaptive retrieval** learns when to read from the index, per query. Self-RAG (Asai et al. 2023) emits retrieval decisions as tokens. Mallen et al. (2023) show that retrieving only for less popular entities improves accuracy and cuts cost.

The gap is narrow: read-time routing between tiers is occupied. Write-time placement, deciding which tier an item should live in, is not.

#### Why the cell is empty

The tiers were developed by separate communities with separate infrastructure. Parametric training and retrieval systems have different tooling, different evaluation traditions and often different teams. The interface between them is a configuration file, not a learned function.

#### Why it becomes forced

A compression objective allocates bits roughly in proportion to frequency, which structurally under-serves the long tail. Kandpal et al. (2023) confirm this empirically: question-answering accuracy tracks how many pretraining documents mention the relevant entities. A fact appearing once in a corpus is expensive to store in weights and cheap to store on disk.

The current response is a human deciding that "facts go in RAG". That decision is made thousands of times, by judgement, without measurement, and it is the kind of decision that yields to optimization once someone poses it as one.

#### Falsifiable prediction

A system with learned write-time tier allocation beats a fixed-blend semi-parametric baseline and an adaptive-retrieval baseline on long-tail question answering at matched total storage.

A weaker prediction, that the learned policy recovers frequency proportionality, is already half confirmed by Kandpal et al. and Mallen et al. The test that remains is sharper: the learned policy should use features beyond frequency, chiefly volatility and compositional use. If a frequency threshold alone matches it, the candidate reduces to a heuristic. If the learned policy is approximately constant, the candidate is wrong.

#### Strongest objection

Estimating access frequency and volatility at training time requires predicting future queries, which may be unpredictable enough that the learned policy degenerates to a constant or a frequency threshold. The candidate depends on query distributions being sufficiently regular, and nobody has measured that.

### 3.2 Verifier synthesis

> **Coordinates:** Axis 1. Constructs an information source rather than consuming one.

#### The observation

The taxonomy identifies a single free lunch. Verifiers, game rules, proof checkers and unit tests are information sources that are free, unlimited and reliable. Every domain where machine learning has advanced unusually fast is a domain that happened to possess one.

The taxonomy then notes that self-play generalized only to domains with verifiers, recommends looking for verifiers, and stops. It treats the presence of a verifier as a fact about a domain, like its dimensionality, rather than as an artifact that might be engineered. This is the largest omission in the frame.

#### What the paradigm would be

Verifier construction as the primary research problem, with its own objective, its own evaluation and its own scaling behaviour. Given a domain lacking automatic ground truth, learn or construct a checker that is:

1. **Cheaper than the generator.** Otherwise there is no leverage.
2. **More reliable than the generator's self-assessment.** Otherwise it adds nothing over self-consistency.
3. **Hard to game.** Otherwise reward overoptimization applies immediately.

The third requirement is the hard one and is where every current attempt struggles.

#### Why this is the highest-leverage empty cell

It is the only candidate that **changes the information source** rather than rearranging what is done with existing sources. In the terms of 1.5, a verifier is the device that converts compute into usable information. Every other entry redistributes bits that are already present.

If it works even partially in one open-ended domain, that domain moves from annotation-bound, where progress costs money linearly, to compute-bound, where progress costs compute and compute scales.

#### Prior art, stated fairly

- **Trained outcome verifiers** for mathematics date to Cobbe et al. (2021). **Generative verifiers** (Zhang et al. 2024) train verification as next-token prediction.
- **Process reward models** supervise intermediate steps. Lightman et al. (2023) trained them on human step labels, but Math-Shepherd (Wang et al. 2024) builds step labels automatically from rollouts. PRMs are therefore not inherently annotation-bound, but the automatic route still presupposes an outcome verifier.
- **LLM-as-judge** and **self-consistency** are cheap, but their reliability is bounded by the generator's own competence, which violates requirement 2 where it matters.
- **Debate**, **recursive reward modelling** and **weak-to-strong generalization** (Burns et al. 2023) address amplifying weak supervision into strong supervision. Closest in spirit; largely untested at scale.
- **The generation-verification gap** has been formalized. Song et al. (2025) define it, find a variant that scales monotonically with pretraining compute, and find self-improvement fails on factual tasks, where verifying is about as hard as generating, but succeeds on Sudoku, where verifying is easy.

The gap: all of these *use* or *measure* a verifier. None treats verifier construction as the object of study, with a benchmark measuring verifier quality independently of the generator it supervises.

#### Why the cell is empty, stated carefully

The generator-verifier asymmetry that makes this work in mathematics and code is a property of those domains, not a general fact about problems. Checking a proof is cheaper than finding one. Running a unit test is cheaper than writing the function. Checking whether an essay is good is not obviously cheaper than writing a good essay, and checking whether a strategy is sound may be as hard as devising one.

So the honest research question is not "how do we build verifiers" but:

> **Which domains possess a generator-verifier asymmetry that has not been exploited because nobody looked for it?**

That question has not been asked systematically. The domains currently exploited were found by accident, because they had verifiers lying around for other reasons. With the gap now measurable, a domain survey is cheap.

#### Falsifiable prediction

A measurable generator-verifier gap now exists (Song et al. 2025). The open prediction: the measured gap, computed per domain before any RL is run, retrodicts where RL succeeded (mathematics, code) and failed (open-ended generation), and predicts at least one new domain where it will succeed.

If the measured gap does not separate the historical successes from the failures, the asymmetry is not the operative variable and this candidate rests on a misdiagnosis.

#### Strongest objection

A learned verifier is a learned reward model, and reward model overoptimization is well documented: optimizing hard against a finite-sample approximation finds its errors rather than the thing it approximates (Gao et al. 2023). A synthesized verifier inherits this in full. The KL-penalty mitigation used in RLHF requires a trusted reference policy, which may not exist where verifier synthesis is most needed.

### 3.3 Value-weighted acquisition

> **Coordinates:** Axis 1. Addresses the admitted failure that bits are not fungible.

#### The observation

The bits-per-sample accounting treats all bits as equal. They are not.

RLHF applies on the order of 10^5 bits to a model carrying 10^13 and produces a large, reliable behavioural change. If bits were fungible, that would be impossible. Those bits are placed where they resolve a consequential ambiguity, and the frame as stated cannot express the difference.

Active learning maximizes **expected information gain**: a quantity measured in bits, blind to whether those bits matter.

#### What the paradigm would be

Acquisition that maximizes expected **decision-relevant** information: bits weighted by how much they change downstream behaviour that someone cares about, rather than by how much they reduce posterior entropy.

- Active learning asks: which query most reduces my uncertainty?
- This asks: which query most changes what I would do?

These come apart constantly. A model can be maximally uncertain about something that affects no decision, and confidently wrong about something that affects every decision.

#### Prior art, stated fairly

This is the least novel candidate in the document, and it is included because being old and unused for labels is itself informative.

- **Bayesian experimental design** and the **expected value of sample information** are exactly this formalism. Decades old and correct.
- **Prediction-oriented active learning** (EPIG, Bickford Smith et al. 2023) targets information about predictions on the deployment distribution rather than about parameters. A step toward value, still measured in bits.
- **Influence functions** measure the effect of a training point on model behaviour: a retrospective version of the same quantity.
- **Pretraining data selection** (DoReMi, FineWeb-Edu, DCLM, Rho-1) weights a free source by estimated downstream value, at scale, and works.

The last item narrows the claim. Value-weighted selection over free data is occupied and successful. The empty part is value-weighted acquisition of *expensive* signal: labels and preferences, where each query costs money and the posterior must be trusted before the query is made.

#### Why the cell is empty

**Soft blocker:** the value function must be specified. Information gain is computable from a posterior alone; expected value of information requires a decision problem and a loss. In general-purpose training, the downstream decision is unspecified by construction.

**Hard blocker:** the required posteriors are untrustworthy. Neural uncertainty estimates are poorly calibrated, and value-weighting multiplies a badly estimated uncertainty by a badly estimated value. Data selection escapes this because it does not need a posterior: it scores data already in hand.

#### Why it becomes tractable now

RLHF is the existence proof that a small, well-placed annotation budget produces large effects. Data selection is the existence proof that value can be estimated well enough to act on at scale. The commercial forcing function is direct: annotation is the dominant marginal cost in post-training, and better placement changes the return on that spend by orders of magnitude.

#### Falsifiable prediction

An annotation budget allocated by estimated decision relevance outperforms the same budget allocated by uncertainty sampling, and by EPIG, on downstream task performance, by a margin that grows as the budget shrinks.

If uncertainty sampling matches it, then either the value estimates are too noisy to help or uncertainty is a sufficient proxy for value. Either finding settles the question.

#### Strongest objection

On balanced, well-specified datasets, active learning often fails to beat random sampling once baselines are tuned properly (Munjal et al. 2022). Adding a value term to a method that does not reliably beat random is optimistic. The candidate must explain why value-weighting succeeds where information-weighting did not; the success of data selection, which scores with a cheap learned classifier rather than a posterior, is the best available answer.

### 3.4 Compression of the training trajectory

> **Coordinates:** Axis 2, applied to a novel object. Addresses the admitted failure that optimization is unaccounted for.

#### The observation

The taxonomy compresses data, teachers, policies, posteriors and learning algorithms. It rarely compresses **the learning process itself**.

A training run produces a trajectory through parameter space: checkpoints, gradients, loss values, gradient norms and activation statistics. Together they hold a large quantity of information about the loss landscape of that architecture on that data. Most of it is discarded. What survives is the final checkpoint and, if someone was diligent, a loss curve in a logging dashboard.

#### What the paradigm would be

Treat the trajectory as a corpus and compress it. The product is a model of optimization dynamics that transfers across runs, supporting:

- Predicting final performance from early curves, at a fidelity beyond current extrapolation.
- Transferring a learning rate schedule to a new *architecture*, not only a new width, on principled grounds.
- Warm-starting a new run from the trajectory of an old one, as opposed to from its endpoint.
- Diagnosing pathological dynamics early enough to intervene.

#### Distinguishing it from existing work

Seven neighbours together occupy much of the cell.

- **Learned optimizers** compress the *update rule*, a fixed function applied at each step. This compresses the *path*, which is data. An optimizer is a policy; a trajectory model is a world model of optimization.
- **Scaling laws** compress the trajectory into two or three scalars. Real compression, extremely lossy.
- **Model soups and weight averaging** use multiple endpoints without modelling the path.
- **Learning-curve extrapolation** predicts final performance from partial curves: Domhan et al. (2015), and LC-PFN (Adriaensen et al. 2023), a transformer trained on many curves. This is the first bullet above, at small scale.
- **muP** (Yang et al. 2022) transfers hyperparameters across width on theoretical grounds. This is the second bullet, for width only.
- **Generative models of checkpoints** (G.pt, Peebles et al. 2022) train a diffusion model on a corpus of checkpoints and sample parameters conditioned on a target loss. This is trajectory compression as a learned artifact, at small scale.
- **Checkpoint-based attribution** (TracIn, Pruthi et al. 2020) uses the trajectory to attribute predictions to training points.

The remaining gap: none of these operates on frontier runs, uses internal statistics beyond the loss curve, or transfers across architectures.

#### Why the cell is empty

Until recently, training runs were cheap enough that discarding their byproducts cost nothing, and there were too few to constitute a corpus. Both conditions have reversed within large organizations. The mundane blocker remains: checkpoint sequences are enormous, and little infrastructure is designed to retain and query them.

#### Falsifiable prediction

A model trained on frontier trajectory data, using internal statistics as well as loss, predicts final performance from the first ten percent of a run more accurately than both power-law extrapolation and an LC-PFN-style curve model. The gap is largest where scaling laws are least reliable: at architecture or data-distribution changes.

#### Strongest objection

The trajectory may contain little transferable information beyond what curve models already extract. Optimization paths are high-dimensional, stochastic and architecture-specific, and the useful signal may be low-dimensional. If so, this cell is correctly empty. This is the candidate most likely to be a dead end, and it is included because the cost of checking is low.

### 3.5 Honest joint MDL at scale

> **Coordinates:** Axis 2. Closes an acknowledged gap in the frame's own accounting.

#### The observation

The parent concedes, and then moves past, a significant inconsistency. Bits-per-token figures exclude the cost of the model. A fair minimum description length accounting must include the cost of transmitting the model as well as the data given the model:

```latex
L(M) + L(D \mid M)
```

Current practice reports only the second term. The justification is amortization: for a corpus far larger than the model, the per-token model cost approaches zero. That justification is correct, and it is also a way of never having to compute the first term.

#### What the paradigm would be

Training and model selection in which model cost is a real term, denominated in the same units as data cost. Weight decay is not this: it is a proxy with no units and no interpretation as a code length.

#### The prequential route

An examinable prior over parameters is not a prerequisite. **Prequential coding** (Blier and Ollivier 2018) transmits data sequentially, encoding each batch with a model trained on the batches before it. The receiver retrains the same model, so the model is never transmitted, and its cost is paid implicitly in the early, poorly predicted batches.

For a single-epoch training run, the prequential code length is approximately the area under the training loss curve. Every frontier pretraining run already computes it and discards it as a dashboard plot. Blier and Ollivier show it gives far tighter code lengths for deep networks than variational methods. An unexaminable parameter prior is therefore not a blocker for the accounting, only for one way of doing it.

#### Prior art, stated fairly

- **Hinton and van Camp (1993)** is precisely this proposal, via a code length for the weights.
- **Prequential MDL** (Blier and Ollivier 2018), and its use for probing (Voita and Titov 2020).
- **PAC-Bayes compression bounds** give non-vacuous guarantees by accounting for model code length, at ImageNet scale (Zhou et al. 2019) and for large language models (Lotfi et al. 2024).
- **Bayesian neural networks** are the same idea under a different name.

#### What it would buy

A principled answer to a question currently answered empirically: how large should this model be for this corpus? Scaling laws are fitted curves that work, and no theory explains their exponents. A joint MDL objective would either predict them or fail to and thereby locate what else is going on.

#### Falsifiable prediction

A tempting test is whether joint MDL reproduces the compute-optimal parameter-to-token ratio. That conflates two optima. MDL is a fixed-*data* criterion: it asks which model best describes this corpus. Compute-optimal scaling (Hoffmann et al. 2022) is a fixed-*compute* criterion: it asks how to spend a FLOP budget. They need not agree, and a mismatch would prove nothing.

The correct test: prequential code length, minimized over model size at fixed data, reproduces the data-constrained optimum measured by Muennighoff et al. (2023), including the point where repeated epochs stop paying. If it does not, the accounting is missing something identifiable.

#### Strongest objection

The amortization argument may simply be right: at the scales that matter, the first term may contribute nothing per token, making current practice a correct approximation rather than sloppiness. A second objection applies to the prequential route specifically: it measures the learner, optimizer and data order included, not the model alone. That is either a feature, since it prices what actually happened, or a confound.

### 3.6 Forgetting as an objective

> **Coordinates:** Axis 2. Rate-distortion applied to removal rather than encoding.

#### The observation

Rate-distortion theory governs what to discard **at encoding time**. Every lossy method in the taxonomy sits somewhere on that curve.

At the level of single records, removal **after encoding** is theorized: differential privacy and certified removal give formal guarantees. At the level of concepts and capabilities, which is where most demand sits, removal is a patch applied to a trained model rather than a property the training objective maintains.

#### What the paradigm would be

Training that treats removability as a first-class property. Knowledge encoded such that a specified subset can be excised with:

1. **Bounded collateral damage.** Removing one fact does not degrade unrelated capability.
2. **Verifiable completeness.** The fact is demonstrably absent, not merely suppressed.
3. **Cost far below retraining.** Otherwise retraining from scratch is sufficient.

The third is achievable today by architectural means. The first two are open for concept-level removal.

#### Why economics forces this one

Of all ten candidates, this is the one whose arrival least depends on the research being ready.

Retrieval gets deletability for free; parametric memory does not. Deletion rights under data protection law, copyright claims over training data, and safety-motivated removal of capabilities are all demands for an operation the parametric paradigm structurally cannot perform. The current options are retraining, which is prohibitively expensive, or moving knowledge to retrieval, which surrenders the benefits of compression. Legal requirements do not wait for satisfactory answers.

#### Prior art, stated fairly

- **Differential privacy** (DP-SGD, Abadi et al. 2016) bounds what any single record contributes, so record-level removability is prepaid at training time, at a utility cost. It is the cleanest existing instance of removability as an objective.
- **Definitions of exact and approximate unlearning** (Ginart et al. 2019) and **certified removal** (Guo et al. 2020) exist, for records and mostly for convex models.
- **SISA** (Bourtoule et al. 2021) achieves exact unlearning by sharding training data, an architectural constraint imposed up front. Sharding hurts performance.
- **Gradient routing** (Cloud et al. 2024) masks gradients so that specified data updates only a designated subregion, which can later be ablated. This localizes capabilities during training for removal afterward, and it is the closest existing thing to this candidate.
- **Model editing** (ROME, MEMIT) changes specific associations. It demonstrably changes outputs; it does not demonstrably remove information.
- **Benchmarks and audits** exist: TOFU for fictitious-author facts, WMDP for hazardous knowledge. Deeb and Roger (2024) find that information "unlearned" by common post-hoc methods is largely recoverable by brief fine-tuning.

#### The hard part

Verification at the concept level. Record-level definitions borrow from differential privacy and are well posed. A fact or capability that appears across thousands of records has no accepted definition of absence, let alone an accepted test.

This matters more than it might appear. A deletion method that cannot be verified is a suppression method with a deletion label, and under a legal standard that distinction is the entire question.

#### Falsifiable prediction

A training objective with an explicit removability term achieves verified concept-level deletion, robust to relearning attacks, at a small fraction of retraining cost, with bounded degradation on held-out capability, and beats gradient routing on the same benchmark. The prediction fails if removability cost scales with model size as retraining does, in which case architectural approaches are the only viable ones.

#### Strongest objection

Information in a distributed representation may not be localized enough for removal to be well defined. If a fact is encoded diffusely across the whole network, "removing it" may have no coherent meaning, and the only honest answers are retraining or retrieval. Gradient routing is the counter-evidence: localization can be imposed rather than hoped for.

### 3.7 Decompilation of weights into programs

> **Coordinates:** Best information source (a trained teacher) combined with best compression target (a program). The only literal empty cell in the grid, and empty only above toy scale.

#### The observation

The taxonomy contains two facts that sit next to each other and are rarely combined.

- **Program synthesis** compresses data into a program: interpretable, verifiable, editable, extrapolates outside the training range. It is hard because the search space is discrete and combinatorial.
- **Distillation** compresses a teacher into a student. The source is cheap to query, unlimited, and provides full output distributions rather than hard labels.

Network-to-program compression exists at toy scale and nowhere else.

#### What the paradigm would be

Distillation whose student is symbolic: not a smaller network but an inspectable, verifiable artifact.

There is reason to expect this is easier than program synthesis from data. The teacher provides an unlimited stream of input-output pairs, gradients and internal activations. Program synthesis is hard largely because the search is under-constrained by few examples; a trained network is an oracle that can be queried arbitrarily.

#### Prior art, stated fairly

- **Automated network-to-program distillation exists.** MIPS (Michaud et al. 2024) converts RNNs trained on algorithmic tasks into finite state machines, then into Python by symbolic regression. It solves 32 of 62 benchmark tasks, 13 of them unsolved by GPT-4.
- **Learning Transformer Programs** (Friedman et al. 2023) trains transformers constrained so they decompile into Python by construction. **Tracr** (Lindner et al. 2023) runs the inverse direction, compiling RASP programs into transformer weights.
- **Mechanistic interpretability** does this manually at small scale: induction heads, indirect object identification, modular arithmetic. Automated circuit discovery (ACDC, Conmy et al. 2023) and attribution graphs over transcoders (Ameisen et al. 2025) automate the localization step, though not translation into code.
- **Symbolic regression distillation** (Cranmer et al. 2020) recovers governing equations from small physics-informed networks.
- **Sparse autoencoders** decompose activations into interpretable features: the vocabulary a program would be written over.
- **Adjacent cell:** FunSearch and AlphaEvolve output programs found by network-guided search against a verifier. The program solves an external problem; it does not describe the network.

The gap: nothing poses decompilation as a learning paradigm with an objective, a benchmark and a scaling story beyond small RNNs.

#### Why it probably stays partial, and why that is the interesting version

One argument holds that if a network's function compressed to a short program, it would be surprising that gradient descent needed billions of parameters. That argument is weak. Over-parameterization aids optimization rather than reflecting function complexity; trained networks can often be pruned heavily without loss. And much of a language model's parameter budget stores facts, at about 2 bits per parameter (Allen-Zhu and Li 2024), which are incompressible data rather than algorithm.

The better decomposition has three parts:

1. **Program:** the algorithmic fraction, decompiled to code.
2. **Table:** the factual fraction, exported to an external index, which links this candidate to 3.1.
3. **Residual:** irreducible statistical association, left neural.

> Decompile what is programmatic, export what is tabular, leave the rest neural, and know which is which.

Knowing which parts implement clean algorithms and which implement association would be valuable even if the extracted fraction were small, because the two fail in completely different ways.

#### Falsifiable prediction

For tasks with known algorithmic structure, automated decompilation recovers the algorithm at a scale substantially beyond MIPS, and the fraction of a network that decompiles cleanly correlates with how algorithmic the task is. The candidate fails if the decompilable fraction is uniformly negligible across tasks once networks exceed small-RNN scale.

#### Strongest objection

Extracted programs may be faithful on the training distribution and wrong off it, the same failure that made rule extraction unsatisfying in the 1990s. Verifying faithfulness is as hard as the original interpretability problem.

### 3.8 Continuous amortization

> **Coordinates:** Axis 3. The timing axis has three discrete settings; this is the continuous limit.

#### The observation

Axis 3 has three values: compress at training time, defer to inference, or alternate. The third is frontier practice, and its period is measured in months and gated by a human deciding when to start a run.

Nothing about the mathematics requires that period. The search-compress loop is a loop; its frequency is a free parameter set by organizational convenience rather than by analysis.

#### What the paradigm would be

No boundary between training and deployment. Each interaction is evaluated for whether compressing it into weights pays for itself against the cost of re-deriving it on demand. Reasoning that is frequently re-derived gets consolidated; one-off work does not.

The decision rule is economic: consolidate when expected future re-derivation cost exceeds consolidation cost. That is computable given a query distribution, which links this candidate to 3.1.

#### Prior art, stated fairly

- **Context distillation** (Askell et al. 2021; Snell et al. 2022) is the consolidation operation itself: fine-tune a model to behave, without a prompt, as it did with one. It runs offline, on a schedule, without a cost-based trigger.
- **Test-time training** (Sun et al. 2020) and **TTT layers** (Sun et al. 2024) update weights during inference. **Titans** (Behrouz et al. 2024) gates writes to a neural long-term memory by surprise: a learned trigger, but per sequence and not economic.
- **Self-adapting models** (SEAL, Zweiger et al. 2025) generate their own fine-tuning data and update directives, then apply them to their weights.
- **Continual and lifelong learning** address the stability-plasticity problem, a prerequisite, but treat consolidation as something to do safely rather than as an economic decision.
- **Complementary learning systems theory** is the conceptual ancestor: fast hippocampal storage, slow neocortical consolidation during sleep. Machine learning borrowed the architecture without the decision rule.

The remaining gap: a consolidation trigger based on expected cost, operating continuously in deployment.

#### Obstacles, and one that is underappreciated

The obvious obstacle is catastrophic forgetting, which is well studied and partially addressed.

The less obvious and more serious one is **evaluation**. A model that differs from itself hour to hour cannot be tested by current practice. Benchmarks assume a fixed artifact. Safety evaluation assumes the thing evaluated is the thing deployed. A continuously consolidating system breaks both assumptions, and the infrastructure to evaluate a moving target does not exist.

The real blocker may therefore not be the learning algorithm but certification of something that changes.

#### Falsifiable prediction

A system with cost-triggered consolidation achieves lower total cost, counting training and inference compute, than a fixed model, a pure retrieval system, and a schedule-triggered context-distillation baseline, on a workload with realistic query repetition. The prediction fails if consolidation overhead exceeds the re-derivation it saves, which depends on how repetitive real workloads are.

#### Strongest objection

Catastrophic forgetting may make the consolidation step unreliable enough that its expected value is negative regardless of the economics. And if evaluating a continuously changing system is impossible rather than merely hard, the candidate is undeployable even if it works.

### 3.9 Explicit inductive bias accounting

> **Coordinates:** A measurement standard rather than a training method. Addresses the admitted failure that inductive bias is invisible.

#### The observation

Architecture, augmentation policy, pretext task and prior encode substantial information that never appears in any bits-per-sample table. Choosing a convolution over a dense layer supplies a large amount of information about the structure of images, free, from a human. Sample-efficiency claims are therefore inflated by an unmeasured human contribution.

#### The clearest instance

Contrastive learning's ceiling is its augmentation policy. These methods work well on natural images because the community spent years finding good augmentations for natural images. That human contribution is presented as a property of the algorithm. When the method transfers poorly to a domain where the correct invariances are unknown, this is reported as a limitation of the domain rather than as evidence about where the performance came from.

The same applies to most self-supervised pretext tasks (2.5) and to most claimed architectural advances, and it is a main reason results fail to replicate outside the setting their bias was tuned for.

#### What the paradigm would be

Measure it, and report it, so that a sample-efficiency claim can be audited against the information supplied through design.

Two currencies are available:

- **Bits.** The reduction in hypothesis space an architecture or augmentation policy induces.
- **Data-equivalent samples.** How much additional data a less-biased model needs to match the biased one. ViT against ResNet gives a direct reading: Vision Transformers trail ResNets on ImageNet-1k and overtake them with roughly 300 million pretraining images (Dosovitskiy et al. 2021). Hernandez et al. (2021) already express pretraining as "effective data transferred" in this currency.

The second currency sidesteps the reference-machine problem in the strongest objection below.

#### Prior art, stated fairly

- **Inductive bias complexity** (Boopathy et al. 2023) quantifies, in bits, the information required to generalize on a task minus the information supplied by the data, and ranks MNIST below CIFAR-10 below ImageNet. A measure exists; it is not applied to audit methods.
- **PAC-Bayes** treats the prior as a code and is the right formalism, applied to weights rather than to architecture choice.
- **Goldblum et al. (2024)** argue, via Kolmogorov complexity, that a shared simplicity bias explains why flexible models generalize across domains.
- **Occam bounds and structural risk minimization** price hypothesis class complexity too coarsely to distinguish architectures.
- **Neural architecture search** explores the space without pricing positions in it.

Proposals to price a convolution in bits exist; none is accepted or used as a reporting standard.

#### Why this one arrives differently

This is the only candidate that is an **evaluation standard** rather than a training method. It will arrive not because someone invents it but because reviewers and practitioners tire of results that do not survive a new domain and demand the accounting. That makes it among the most likely of the ten to happen, and the least likely to be attributed to anyone.

#### Falsifiable prediction

A measure of design-supplied information, applied retrospectively, explains a substantial fraction of the variance in how well published methods transfer across domains. Methods with high measured bias transfer worse, strongly enough to be predictive.

#### Strongest objection

Description length is defined relative to a universal machine, and for architectures the choice of reference machine may dominate the measurement. If the ranking of architectures by information content depends on the encoding scheme, the measure is not well defined enough to audit anything. Data-equivalent units avoid this, at the price of being task-specific.

### 3.10 Learned pretext tasks

> **Coordinates:** Axis 1, raw corpus row. The extraction rule of self-supervision becomes a learned variable. Generated by Section 2.5.

#### The observation

Self-supervision converts a corpus into training targets by a hand-designed rule: predict the next token, predict 15% of masked tokens, match two augmented views. The rule determines which dependencies in the corpus become signal. It is the largest unpriced design choice in the field, and it is chosen once, by a human, and held fixed for the whole run.

#### What the paradigm would be

Treat the pretext task as a learned object: which positions to predict, from which context, at which granularity, in which space (tokens or latents). Optimize it for downstream transfer per unit compute, and adapt it over training as easy dependencies saturate and harder ones become reachable.

#### Prior art, stated fairly

- **Learned views.** InfoMin (Tian et al. 2020) argues good contrastive views share only task-relevant information, and learns them. Viewmaker networks (Tamkin et al. 2021) learn augmentations adversarially.
- **Learned targets.** BYOL, data2vec and I-JEPA predict the representations of a slowly updated target encoder, so the target is learned. The masking pattern and the context-target split remain hand-designed.
- **Mixtures of objectives.** UL2 (Tay et al. 2022) trains on a fixed mixture of denoisers. Multi-token prediction (Gloeckle et al. 2024) widens the target.
- **Token selection.** Rho-1 (Lin et al. 2024) learns which tokens deserve loss, which is selection within a fixed task rather than a change of task.

The gap: the target is sometimes learned and the data is sometimes selected, but the task structure itself is not optimized at language-model scale.

#### Why the cell is empty

Three reasons. The objective for the objective is downstream transfer, which is unspecified in general-purpose pretraining; this is the same soft blocker as 3.3. Adversarially learned views collapse toward trivial or impossible tasks without careful constraints. And while corpora grew faster than models could consume them, next-token prediction was good enough that there was little incentive to change it.

#### Why it becomes forced

The stock of public human-written text is projected to be fully used sometime between 2026 and 2032 (Villalobos et al. 2024). Beyond about four epochs, repeated data loses value quickly (Muennighoff et al. 2023). Once the corpus stops growing, extracting more usable information per token is the remaining lever, and the pretext task is what controls it.

#### Falsifiable prediction

At a fixed corpus trained for many epochs, an adaptive pretext-task curriculum beats repeated next-token training on downstream evaluation at matched compute, with the gap growing with epoch count. It also reduces reversal-curse failures without any data augmentation. The prediction fails if repeated next-token training at the same compute matches it, which would mean the autoregressive factorization already extracts everything learnable.

#### Strongest objection

Autoregressive modelling is complete in principle: by the chain rule it models the full joint distribution, so alternative tasks only reorder what is learned, not what is learnable. The reply is that completeness in the limit is not efficiency at finite compute. The reversal curse shows finite models do not recover dependencies their factorization order does not expose. If scale alone removes the reversal curse, the objection wins.

## 4. Cells That Should Stay Empty

Honesty requires this section. Not every gap is an opportunity, and a taxonomy that only generates optimism is not being used properly.

### 4.1 Own generations, unfiltered and computation-free, replacing real data

In the Shannon sense, own generations are worth zero: no recursion recovers information that was not present. In the usable sense (1.5) they are not always worthless. Self-distillation (born-again networks, DINO) improves models without a verifier, and distilled reasoning traces transfer the results of computation the student cannot perform in one pass. A blanket verdict of "worth zero" contradicts both.

The cell that should stay empty is narrower: recursive training on unfiltered self-generated data, with no added computation, that replaces real data rather than supplementing it. Model collapse confirms it: each round loses tail behaviour, because each round is a lossy re-encoding of the same information (Shumailov et al. 2024). The boundary is also measured: collapse is avoided when synthetic data accumulates alongside real data rather than replacing it (Gerstgrasser et al. 2024).

Synthetic data works most reliably when a **verifier filters it**. Then the verifier is the information source and the generator is only a proposal mechanism; the bits come from the filter, not from the generation. A paradigm claiming gains from self-generated data should say which of three things supplies them: a filter, added computation, or a regularization effect that could be obtained more directly.

### 4.2 Reinforcement learning as the primary signal, outside domains with verifiers

The unqualified version is false: AlphaGo Zero is RL-primary, from scratch, with no prior. Where a cheap simulator or verifier exists, RL can be the primary signal.

The claim that holds: in open-ended domains without a verifier, a scalar reward over a long trajectory is a narrow channel by construction, and credit assignment across it is the difficulty, not the implementation. There, RL remains a fine-tuning and search method layered on a prior built by high-bandwidth compression. Proposals to train open-ended foundation models primarily on preference rewards are proposing to replace a 10^13-bit source with a 10^5-bit one.

This cell is conditional on 3.2. If verifier synthesis succeeds in a domain, the domain gains a verifier and this verdict no longer applies to it.

### 4.3 Primary objectives with no independent evaluation

Revived adversarial objectives are often placed here, on the grounds that an objective without an information-theoretic interpretation cannot be evaluated. Both halves need correction.

Adversarial objectives were never abandoned. They are load-bearing as auxiliary terms: the autoencoders behind latent diffusion are trained with adversarial losses (VQGAN, Esser et al. 2021; Rombach et al. 2022), and adversarial diffusion distillation (Sauer et al. 2023) makes few-step sampling work. And the criterion is too strict: non-contrastive self-supervised methods (BYOL, DINO, I-JEPA) have no likelihood and work, because they are evaluated independently by linear probes and transfer.

The lesson from GANs that does hold: a primary objective whose progress can be measured only by itself, or by proxies the method can game, cannot distinguish progress from noise. Years of GAN research relied on sample-quality proxies, and that was a predictable consequence of abandoning the likelihood without replacing it. The cell that should stay empty is any primary objective lacking an evaluation independent of its own loss.

### 4.4 Posterior compression at frontier scale

The blocker is real and unchanged: nobody can specify or inspect a meaningful prior over billions of parameters, and variational approximations underestimate uncertainty in known, systematic ways. Laplace approximations and last-layer Bayesian methods (Daxberger et al. 2021) scale partially; full-network posteriors at frontier scale remain the empty cell.

This matters most because the applications motivating Bayesian deep learning are safety applications. An uncertainty estimate that is systematically overconfident in an unknown pattern is worse than none, since it invites reliance it cannot support.

Section 3.5 might seem to share this blocker. Its prequential route needs no parameter prior, so it does not. The posterior cell stays empty; the MDL cell does not depend on it.

## 5. Prioritization

### 5.1 By expected value

**First: verifier synthesis (3.2).** It is the only candidate that adds usable information rather than rearranging it. Every domain with fast progress had a verifier by accident. The generation-verification gap is now measurable, so the first step, a systematic survey of which domains have an unexploited asymmetry, is cheap. If it works in one open-ended domain, that domain becomes compute-bound, and compute scales.

**Second: forgetting as an objective (3.6).** Not because it is deepest but because regulation, copyright and safety requirements will force it on a schedule unrelated to research readiness. Work done under deadline without preparation is done badly. Gradient routing shows the training-time route is viable; concept-level verification is the open problem to prepare.

**Third: learned pretext tasks (3.10).** It acts on the only near-free, near-unlimited source, at the moment that source stops growing. The data wall makes the extraction rule the main remaining lever on pretraining, and nobody optimizes it.

**Fourth: learned tier allocation (3.1).** The cleanest well-posed optimization problem in the list, with mature components on both sides. Ranked fourth because read-time routing already captures part of the value.

### 5.2 By likelihood of happening regardless

**Inductive bias accounting (3.9)** arrives as an evaluation norm, driven by reviewer fatigue rather than anyone's research agenda.

**Continuous amortization (3.8)** arrives incrementally, as test-time memory and context distillation acquire better triggers, and is unlikely to be recognized as a paradigm shift while it happens.

**Joint MDL (3.5)** is free to compute on any single-epoch run. It will be reported routinely once someone shows it predicts something scaling laws do not.

### 5.3 By risk of being a dead end

**Trajectory compression (3.4)** is most likely to find that curve models already extract the transferable signal. Cheap to check, which is why it is worth checking.

**Learned pretext tasks (3.10)** fail if the chain-rule completeness of next-token prediction holds at practical scale. The reversal curse is the test case.

### 5.4 The dependency structure

Five pairs are coupled. Progress on either member helps the other.

| Pair | Shared subproblem |
| --- | --- |
| 3.1 and 3.8 | Estimating future access patterns, to decide what to consolidate and where |
| 3.2 and 3.3 | Reliability of learned evaluators; a better verifier is also a better value estimator |
| 3.1 and 3.7 | Splitting knowledge from algorithm; decompilation exports facts to an index |
| 3.9 and 3.10 | A pretext task is inductive bias; pricing it and learning it are the same measurement |
| 3.3 and 3.10 | Choosing which bits of a source to extract, at the level of examples (3.3) or of dependencies (3.10) |

## 6. How to Evaluate a Claimed New Paradigm

The taxonomy's most immediately useful output is a checklist. Seven questions, applicable to any method presenting itself as new.

**1. What information does it add, and in which sense?** Say whether the gain is Shannon information from outside the model, or usable information created by computation (1.5). If neither, the best case is a more useful redistribution of existing bits: sometimes worth doing, never a large effect. This question alone disposes of most unfiltered synthetic-data claims.

**2. Where does that information come from, and what does it cost per bit in money and in compute?** Locate it on Axis 1. A method that needs human annotation has a ceiling that scales with money. A method that exploits a verifier has a ceiling that scales with compute. These are not comparable.

**3. If the source is a raw corpus, what is the extraction rule, and who chose it?** Name the pretext task. A result that depends on a hand-tuned masking scheme or augmentation policy is partly a result about that choice (2.5, 3.9).

**4. What does it compress into, and is that target chosen or inherited?** Most methods inherit "weights" without considering alternatives. A method that examined the choice and chose weights deliberately is in a different position from one that never asked.

**5. When is the compression paid for?** Training time, inference time, or both in a loop. This determines deployment economics and is frequently unstated in papers.

**6. Does it have an evaluation independent of its own objective?** An information-theoretic interpretation is sufficient but not necessary; a held-out probe or transfer task also serves. If the only measure of progress is the training loss or a gameable proxy, apply the GAN precedent (4.3).

**7. How much of the result is inductive bias that was not counted?** Ask what happens in a domain where the relevant invariances are unknown, and, where possible, state the bias in data-equivalent samples. If the method has only been shown where the community spent years tuning the priors, the result is about the priors.

Applied together, these questions separate genuine new cells from relabelled old ones fairly reliably.

## 7. Limits of This Exercise

Five caveats, stated because the exercise is easy to oversell.

**The taxonomy is not the periodic table.** Its axes are categorical, not numeric. Empty cells do not come with predicted properties the way gallium's density did. What the frame gives is a reason to ask a question, not an answer to it.

**Prediction from a frame is weaker than it appears.** The frame was constructed after the fact to describe existing methods. A structure fitted to existing data will contain apparent gaps that are artifacts of the fitting. Some of the ten candidates are likely such artifacts.

**Novelty claims are the weakest part.** Every candidate has prior art. For 3.3 and 3.5 the prior art is decades old and the claim is that it was abandoned prematurely. For 3.4, 3.6 and 3.7 there is direct prior art at small scale, so the claim reduces to "this has not been done at scale". That is a much weaker claim than "this does not exist", and it is the honest one.

**Real paradigms have historically come from elsewhere.** Backpropagation, attention and diffusion did not come from taxonomic reasoning. They came from an analogy, an engineering need and a physics connection respectively, and were rationalized into the frame afterwards. The frame's admission that it is descriptive rather than generative should be taken seriously; this document tests that admission rather than refuting it.

**The most important missing paradigm is probably not on this list.** By construction, this exercise can only find gaps the existing axes make visible. A genuinely new paradigm would likely need a fourth axis, and would appear in this frame as something that does not fit rather than as an empty cell. Two things already fail to fit three axes: computation as a converter of information (1.5), and the extraction rule of self-supervision (2.5). Either may be that axis.

## 8. Summary Table

| # | Candidate | Axis | Source of prediction | Prior art status | Likelihood | Payoff if right |
| --- | --- | --- | --- | --- | --- | --- |
| 3.1 | Learned tier allocation | 2 | Structural: target never learned at write time | Read-time routing exists; write-time placement empty | Medium | High |
| 3.2 | Verifier synthesis | 1 | Structural: source never constructed | Gap measured (2024); construction unstudied as a field | Medium | Very high |
| 3.3 | Value-weighted acquisition | 1 | Admitted failure: bits not fungible | Theory decades old; works for free data, not for labels | Medium | High |
| 3.4 | Trajectory compression | 2 | Admitted failure: optimization unmodelled | Curve extrapolation, muP, checkpoint models at small scale | Low | Medium |
| 3.5 | Joint MDL at scale | 2 | Frame's own accounting gap | 1993; prequential route computable now | Medium | High |
| 3.6 | Forgetting as objective | 2 | Structural: removal untheorized above records | DP, SISA, gradient routing; concept level open | High, forced | High |
| 3.7 | Weights to programs | 1 + 2 | Literal empty cell above toy scale | Automated for small RNNs (MIPS) | Medium | High |
| 3.8 | Continuous amortization | 3 | Structural: loop period arbitrary | Context distillation, test-time memory; no cost trigger | Medium | Medium |
| 3.9 | Inductive bias accounting | Meta | Admitted failure: bias uncounted | Measures proposed (2023), not applied | High, gradual | Medium |
| 3.10 | Learned pretext tasks | 1 | Structural: SSL extraction rule hand-designed | Learned views and targets; task structure fixed | Medium | High |

**Excluded deliberately (Section 4):** unfiltered, computation-free own generations replacing real data; RL as the primary signal outside verifier domains; primary objectives with no independent evaluation; posterior compression at frontier scale.

## 9. References

Linked entries were verified against the source. Unlinked entries were checked from memory and should be confirmed before external publication.

**The frame and information notions**

- Shannon, C. E. (1948). A Mathematical Theory of Communication.
- Rissanen, J. (1978). Modeling by Shortest Data Description.
- Grünwald, P. (2007). The Minimum Description Length Principle.
- Tishby, N., Pereira, F. C. and Bialek, W. (1999). The Information Bottleneck Method.
- Delétang, G. et al. (2024). Language Modeling Is Compression. ICLR.
- Xu, Y., Zhao, S., Song, J., Stewart, R. and Ermon, S. (2020). A Theory of Usable Information under Computational Constraints. ICLR.
- Hinton, G., Vinyals, O. and Dean, J. (2015). Distilling the Knowledge in a Neural Network.
- Furlanello, T. et al. (2018). Born-Again Neural Networks.
- Yue, Y. et al. (2025). Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?

**Self-supervised learning (2.5, 3.10)**

- Devlin, J. et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
- Clark, K. et al. (2020). ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators.
- van den Oord, A., Li, Y. and Vinyals, O. (2018). Representation Learning with Contrastive Predictive Coding.
- Poole, B. et al. (2019). On Variational Bounds of Mutual Information.
- Chen, T. et al. (2020). A Simple Framework for Contrastive Learning of Visual Representations.
- Grill, J.-B. et al. (2020). Bootstrap Your Own Latent.
- Caron, M. et al. (2021). Emerging Properties in Self-Supervised Vision Transformers.
- He, K. et al. (2022). Masked Autoencoders Are Scalable Vision Learners.
- Baevski, A. et al. (2022). data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language.
- Assran, M. et al. (2023). Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture.
- Tian, Y. et al. (2020). What Makes for Good Views for Contrastive Learning?
- Tamkin, A., Wu, M. and Goodman, N. (2021). Viewmaker Networks: Learning Views for Unsupervised Representation Learning.
- Tay, Y. et al. (2022). UL2: Unifying Language Learning Paradigms.
- Gloeckle, F. et al. (2024). Better and Faster Large Language Models via Multi-token Prediction.
- Berglund, L. et al. (2023). The Reversal Curse: LLMs Trained on "A is B" Fail to Learn "B is A".
- Muennighoff, N. et al. (2023). Scaling Data-Constrained Language Models.
- Villalobos, P. et al. (2024). Will We Run Out of Data? Limits of LLM Scaling Based on Human-Generated Data.

**The source by target grid (2.2)**

- Gulwani, S. (2011). Automating String Processing in Spreadsheets Using Input-Output Examples.
- Baker, B. et al. (2022). Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos.
- Findeis, A. et al. (2024). Inverse Constitutional AI: Compressing Preferences into Principles.
- Korattikara, A. et al. (2015). Bayesian Dark Knowledge.
- Verma, A. et al. (2018). Programmatically Interpretable Reinforcement Learning.
- Wang, G. et al. (2023). Voyager: An Open-Ended Embodied Agent with Large Language Models.
- Blundell, C. et al. (2016). Model-Free Episodic Control.
- Oh, J. et al. (2018). Self-Imitation Learning.
- Shinn, N. et al. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning.

**3.1, memory tiers**

- Khandelwal, U. et al. (2020). Generalization through Memorization: Nearest Neighbor Language Models.
- Lewis, P. et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.
- Lample, G. et al. (2019). Large Memory Layers with Product Keys.
- Berges, V.-P. et al. (2024). Memory Layers at Scale.
- Graves, A. et al. (2016). Hybrid Computing Using a Neural Network with Dynamic External Memory.
- Asai, A. et al. (2023). Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.
- Mallen, A. et al. (2023). When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories.
- Kandpal, N. et al. (2023). Large Language Models Struggle to Learn Long-Tail Knowledge.
- Allen-Zhu, Z. and Li, Y. (2024). Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws.

**3.2, verifiers**

- Silver, D. et al. (2017). Mastering the Game of Go without Human Knowledge.
- Cobbe, K. et al. (2021). Training Verifiers to Solve Math Word Problems.
- Lightman, H. et al. (2023). Let's Verify Step by Step.
- Wang, P. et al. (2024). Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations.
- Zhang, L. et al. (2024). Generative Verifiers: Reward Modeling as Next-Token Prediction.
- Irving, G., Christiano, P. and Amodei, D. (2018). AI Safety via Debate.
- Burns, C. et al. (2023). Weak-to-Strong Generalization: Eliciting Strong Capabilities with Weak Supervision.
- Song, Y., Zhang, H., Eisenach, C., Kakade, S., Foster, D. and Ghai, U. (2025). [Mind the Gap: Examining the Self-Improvement Capabilities of Large Language Models](https://arxiv.org/abs/2412.02674v2). ICLR.
- Gao, L., Schulman, J. and Hilton, J. (2023). Scaling Laws for Reward Model Overoptimization. ICML.

**3.3, acquisition and selection**

- Lindley, D. V. (1956). On a Measure of the Information Provided by an Experiment.
- Houlsby, N. et al. (2011). Bayesian Active Learning for Classification and Preference Learning.
- Bickford Smith, F. et al. (2023). Prediction-Oriented Bayesian Active Learning. AISTATS.
- Koh, P. W. and Liang, P. (2017). Understanding Black-box Predictions via Influence Functions.
- Munjal, P. et al. (2022). Towards Robust and Reproducible Active Learning Using Neural Networks. CVPR.
- Xie, S. M. et al. (2023). DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining.
- Penedo, G. et al. (2024). The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale.
- Li, J. et al. (2024). DataComp-LM: In Search of the Next Generation of Training Sets for Language Models.
- Lin, Z. et al. (2024). Rho-1: Not All Tokens Are What You Need.

**3.4, trajectories**

- Andrychowicz, M. et al. (2016). Learning to Learn by Gradient Descent by Gradient Descent.
- Kaplan, J. et al. (2020). Scaling Laws for Neural Language Models.
- Garipov, T. et al. (2018). Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs.
- Wortsman, M. et al. (2022). Model Soups.
- Domhan, T., Springenberg, J. T. and Hutter, F. (2015). Speeding up Automatic Hyperparameter Optimization of Deep Neural Networks by Extrapolation of Learning Curves.
- Adriaensen, S. et al. (2023). Efficient Bayesian Learning Curve Extrapolation using Prior-Data Fitted Networks.
- Yang, G. et al. (2022). Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer.
- Peebles, W. et al. (2022). Learning to Learn with Generative Models of Neural Network Checkpoints.
- Pruthi, G. et al. (2020). Estimating Training Data Influence by Tracing Gradient Descent.

**3.5, MDL**

- Hinton, G. and van Camp, D. (1993). Keeping Neural Networks Simple by Minimizing the Description Length of the Weights.
- Blier, L. and Ollivier, Y. (2018). The Description Length of Deep Learning Models.
- Voita, E. and Titov, I. (2020). Information-Theoretic Probing with Minimum Description Length.
- Zhou, W. et al. (2019). Non-Vacuous Generalization Bounds at the ImageNet Scale: A PAC-Bayesian Compression Approach.
- Lotfi, S. et al. (2024). Non-Vacuous Generalization Bounds for Large Language Models.
- Hoffmann, J. et al. (2022). Training Compute-Optimal Large Language Models.

**3.6, forgetting**

- Abadi, M. et al. (2016). Deep Learning with Differential Privacy.
- Ginart, A. et al. (2019). Making AI Forget You: Data Deletion in Machine Learning.
- Guo, C. et al. (2020). Certified Data Removal from Machine Learning Models.
- Bourtoule, L. et al. (2021). Machine Unlearning. IEEE S&P. The SISA approach.
- Cloud, A., Goldman-Wetzler, J., Wybitul, E., Miller, J. and Turner, A. M. (2024). [Gradient Routing: Masking Gradients to Localize Computation in Neural Networks](https://arxiv.org/abs/2410.04332v2).
- Meng, K. et al. (2022). Locating and Editing Factual Associations in GPT. ROME.
- Meng, K. et al. (2023). Mass-Editing Memory in a Transformer. MEMIT.
- Maini, P. et al. (2024). TOFU: A Task of Fictitious Unlearning for LLMs.
- Li, N. et al. (2024). The WMDP Benchmark: Measuring and Reducing Malicious Use with Unlearning.
- Deeb, A. and Roger, F. (2024). Do Unlearning Methods Remove Information from Language Model Weights?

**3.7, decompilation**

- Michaud, E. J. et al. (2024). [Opening the AI Black Box: Program Synthesis via Mechanistic Interpretability](https://arxiv.org/abs/2402.05110v1). Published as "Distilling Machine-Learned Algorithms into Code", Entropy 26, 1046.
- Friedman, D., Wettig, A. and Chen, D. (2023). Learning Transformer Programs.
- Lindner, D. et al. (2023). Tracr: Compiled Transformers as a Laboratory for Interpretability.
- Elhage, N. et al. (2021). A Mathematical Framework for Transformer Circuits.
- Olsson, C. et al. (2022). In-context Learning and Induction Heads.
- Conmy, A. et al. (2023). Towards Automated Circuit Discovery for Mechanistic Interpretability.
- Ameisen, E. et al. (2025). Circuit Tracing: Revealing Computational Graphs in Language Models.
- Cranmer, M. et al. (2020). Discovering Symbolic Models from Deep Learning with Inductive Biases.
- Bricken, T. et al. (2023). Towards Monosemanticity.
- Romera-Paredes, B. et al. (2024). Mathematical Discoveries from Program Search with Large Language Models. Nature.
- Novikov, A. et al. (2025). AlphaEvolve: A Coding Agent for Scientific and Algorithmic Discovery.

**3.8, consolidation**

- McClelland, J. L., McNaughton, B. L. and O'Reilly, R. C. (1995). Why There Are Complementary Learning Systems in the Hippocampus and Neocortex.
- Kirkpatrick, J. et al. (2017). Overcoming Catastrophic Forgetting in Neural Networks.
- Askell, A. et al. (2021). A General Language Assistant as a Laboratory for Alignment.
- Snell, C., Klein, D. and Zhong, R. (2022). Learning by Distilling Context.
- Sun, Y. et al. (2020). Test-Time Training with Self-Supervision for Generalization under Distribution Shifts.
- Sun, Y. et al. (2024). Learning to (Learn at Test Time): RNNs with Expressive Hidden States.
- Behrouz, A., Zhong, P. and Mirrokni, V. (2024). Titans: Learning to Memorize at Test Time.
- Zweiger, A. et al. (2025). Self-Adapting Language Models.

**3.9, inductive bias**

- Boopathy, A., Liu, K., Hwang, J., Ge, S., Mohammedsaleh, A. and Fiete, I. (2023). [Model-agnostic Measure of Generalization Difficulty](https://arxiv.org/abs/2305.01034v2). ICML.
- Goldblum, M., Finzi, M., Rowan, K. and Wilson, A. G. (2024). The No Free Lunch Theorem, Kolmogorov Complexity, and the Role of Inductive Biases in Machine Learning. ICML.
- Hernandez, D. et al. (2021). Scaling Laws for Transfer.
- Dosovitskiy, A. et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.

**Section 4**

- Shumailov, I. et al. (2024). AI Models Collapse When Trained on Recursively Generated Data. Nature. Preprint 2023 as "The Curse of Recursion".
- Gerstgrasser, M. et al. (2024). Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data.
- Esser, P., Rombach, R. and Ommer, B. (2021). Taming Transformers for High-Resolution Image Synthesis.
- Rombach, R. et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models.
- Sauer, A. et al. (2023). Adversarial Diffusion Distillation.
- Daxberger, E. et al. (2021). Laplace Redux: Effortless Bayesian Deep Learning.
