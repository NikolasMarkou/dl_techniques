# Modes of Inference in AI/ML

Deduction, induction, abduction, and four derived modes: a self-contained reference on their formalizations, where each one lives inside AI/ML systems, worked examples, and the failure modes that matter in practice.

**Contents**

1. [The Core Distinction](#1-the-core-distinction)
2. [Deduction](#2-deduction)
3. [Induction](#3-induction)
4. [Abduction](#4-abduction)
5. [Side-by-Side Comparison: The Three Classical Modes](#5-side-by-side-comparison-the-three-classical-modes)
6. [Transduction](#6-transduction)
7. [Non-Monotonic Reasoning](#7-non-monotonic-reasoning)
8. [Analogical Inference](#8-analogical-inference)
9. [Solomonoff Induction](#9-solomonoff-induction)
10. [Extended Comparison: All Seven Modes](#10-extended-comparison-all-seven-modes)
11. [Terminological Traps](#11-terminological-traps)
12. [Where LLMs Sit](#12-where-llms-sit)
13. [Practical Implications for System Design](#13-practical-implications-for-system-design)
14. [Key References](#14-key-references)

Sections 2 to 4 cover the three classical Peircean modes. Sections 6 to 9 cover four derived or limiting modes: transduction (induction minus generalization), non-monotonic reasoning (deduction plus defeasibility), analogy (a compound of all three), and Solomonoff induction (induction's formal ideal). Sections 11 to 13 are the practical payload: what gets confused with what, where LLM behavior actually falls, and how to choose.

---

## 1. The Core Distinction

The taxonomy comes from C.S. Peirce, who framed all three modes as rearrangements of the same three-part structure:

| Component | Symbol | Example |
|---|---|---|
| **Rule** (theory, general law, model) | `T` | "All requests from banned IPs get a 403" |
| **Case** (cause, hypothesis, instance) | `E` | "This request came from a banned IP" |
| **Result** (observation, data, effect) | `O` | "This request got a 403" |

The three modes differ in **which two you have** and **which one you infer**:

```
DEDUCTION:  Rule + Case   ⟶  Result     (T ∧ E ⊨ O)     "what must follow"
INDUCTION:  Case + Result ⟶  Rule       (learn T)       "what generally holds"
ABDUCTION:  Rule + Result ⟶  Case       (find E)        "what would explain this"
```

Two axes separate them:

- **Ampliative vs. non-ampliative.** Deduction adds no information beyond what the premises already contain. Induction and abduction both add content, and therefore both can be wrong even when their inputs are correct.
- **Truth-preserving vs. defeasible.** Only deduction guarantees that true premises yield a true conclusion. Induction and abduction are *defeasible*: new evidence can overturn a previously well-supported conclusion.

> **The one-line version:** deduction tells you what *must* be true, induction tells you what *tends* to be true, abduction tells you what *might* be true given what you just saw.

Two framing notes before the details.

**The modes are complementary, not competing.** Peirce's own account is a cycle: abduction generates a hypothesis, deduction extracts its testable consequences, induction evaluates those consequences against accumulated data. A system that only does one of the three is structurally incomplete, and §13 argues that the strongest architectures are the ones that close the loop explicitly.

**The distinction is about the inference step, not about the technology.** Symbolic systems perform induction (see inductive logic programming, §3.2) and neural systems participate in deduction (as proposers inside verified loops, §2.2). Mapping "logic equals deduction, machine learning equals induction" onto the taxonomy is the single most common error in applying it.

---

## 2. Deduction

### 2.1 Definition

Deduction derives conclusions that are logically entailed by a set of premises. Given a theory `T` (axioms, rules, a knowledge base, a trained model, a program) and facts `F`, deduction produces every `O` such that:

```
T ∪ F ⊨ O
```

that is, `O` is true in every model of `T ∪ F`. Deduction is **sound** (given a sound calculus), **truth-preserving**, and **monotonic**: adding premises never invalidates a previously derived conclusion.

```
T ⊨ O   implies   T ∪ {p} ⊨ O   for any p
```

Deduction is *analytic*: the conclusion was already latent in the premises. Its value is not novelty but **certainty and explicitness**, making implicit consequences computable.

### 2.2 Where deduction lives in AI/ML

| Area | Mechanism |
|---|---|
| Logic programming | Prolog SLD resolution, Datalog fixpoint and semi-naive evaluation |
| Automated reasoning | Resolution provers, SAT/SMT solvers (Z3, CVC5), tableau methods, interactive provers (Lean, Coq, Isabelle) |
| Knowledge graphs and semantic web | OWL-DL and description-logic reasoners (HermiT, ELK, Pellet), RDFS entailment, SHACL validation |
| Rule engines and expert systems | Forward chaining (RETE), backward chaining, business rule systems |
| Classical planning | Plan-as-proof, STRIPS regression, model checking, temporal logic verification |
| Program analysis | Type checking, abstract interpretation, symbolic execution, static verifiers |
| Constraint reasoning | CSP propagation, integer programming feasibility proofs |
| Probabilistic models | Applying Bayes' rule or belief propagation to a *given* model is deductive within the probability calculus: the posterior is entailed by prior, likelihood, and evidence |
| Neuro-symbolic systems | An LLM proposes candidate steps and a symbolic checker certifies them, for example proof search with the Lean kernel as arbiter |
| Guardrails and policy | Formal access-control and safety constraints, deterministic post-hoc rule filters over model outputs |

Note the terminology trap: an ML system's "inference" at serving time (a forward pass) is **not** induction. Induction happened during *training*, when the hypothesis was selected. Applying the trained hypothesis to a new input is deduction *relative to that hypothesis*.

### 2.3 Example

**Knowledge base (Datalog / Prolog):**

```prolog
% Rules
banned(IP)          :- on_blocklist(IP).
banned(IP)          :- asn(IP, ASN), banned_asn(ASN).
blocked(Req)        :- request_from(Req, IP), banned(IP).
alert(Req)          :- blocked(Req), payload_size(Req, S), S > 10000.

% Facts
request_from(r17, '203.0.113.42').
asn('203.0.113.42', as64511).
banned_asn(as64511).
payload_size(r17, 24000).
```

**Query and derivation:**

```
?- alert(r17).

  asn('203.0.113.42', as64511) ∧ banned_asn(as64511)        ⊨  banned('203.0.113.42')
  request_from(r17, '203.0.113.42') ∧ banned('203.0.113.42') ⊨  blocked(r17)
  blocked(r17) ∧ payload_size(r17, 24000) ∧ 24000 > 10000    ⊨  alert(r17)

true.
```

The conclusion `alert(r17)` is **guaranteed** given the rules and facts. There is no probability attached, no training data, and no possibility of the conclusion being false unless a premise is false. If `banned_asn(as64511)` is retracted the conclusion vanishes, but that is a change of premises, not a defeat of the inference. (Contrast §7, where conclusions are defeated by *added* premises.)

A second, ML-flavored example: given a trained decision tree (the theory) and a feature vector (the facts), the predicted class is *deduced* by traversing the tree. The tree's own correctness is an inductive question; the traversal is not.

### 2.4 Limitations

1. **Zero empirical content.** Deduction cannot discover anything about the world. It can only unpack what someone already asserted. Garbage in, certified garbage out.
2. **Knowledge-acquisition bottleneck.** Someone must author `T`. Hand-encoding a domain's rules is expensive, slow, and requires experts who often cannot articulate their tacit knowledge. This is the historical wall that expert systems hit.
3. **Brittleness and the qualification problem.** Real rules have unbounded exception lists: "birds fly" holds unless the bird is a penguin, injured, dead, caged, or in a vacuum. Classical logic has no graceful way to express "usually." Mitigations (default logic, circumscription, answer set programming) buy expressiveness at the cost of monotonicity and often of tractability. See §7.
4. **Computational cost.** Propositional satisfiability is NP-complete; description logics range from PTIME (EL++) to EXPTIME and 2-EXPTIME (SROIQ); first-order logic is only semi-decidable, so a prover may never terminate on a false conjecture. Expressiveness and decidability trade off directly.
5. **Inconsistency is catastrophic.** In classical logic a single contradiction entails everything (*ex falso quodlibet*). Merged or crowd-sourced knowledge bases are routinely inconsistent, requiring paraconsistent logics or repair machinery.
6. **No native handling of uncertainty, noise, or perception.** Deduction needs crisp symbols. Getting from pixels, audio, or free text to reliable ground atoms is an inductive problem it cannot solve for itself, which is exactly why neuro-symbolic architectures exist.
7. **Closed-world vs. open-world mismatch.** Datalog and Prolog assume unstated facts are false (negation as failure); OWL assumes they are merely unknown. Choosing wrong silently changes the meaning of every answer.
8. **LLMs do not deduce reliably.** Chain-of-thought *resembles* deduction but carries no soundness guarantee; it is induced token prediction that imitates proof form. Long chains accumulate error, and plausible-looking invalid steps are common. Soundness requires an external verifier. See §12.

---

## 3. Induction

### 3.1 Definition

Induction infers a general rule from particular observations. Given a sample `D = {(x₁,y₁),...,(xₙ,yₙ)}` drawn from an unknown distribution `P(x,y)`, and a hypothesis space `H`, induction selects `h ∈ H` intended to hold beyond `D`.

**Statistical formulation (empirical risk minimization):**

```
ĥ = argmin_{h∈H}  (1/n) Σᵢ L(h(xᵢ), yᵢ)   [+ Ω(h)]

goal:  R(h) = E_{(x,y)~P}[ L(h(x), y) ]   is small
```

**Logical formulation (inductive logic programming).** Given background knowledge `B`, positive examples `E⁺`, and negative examples `E⁻`, find a hypothesis `H` such that:

```
B ∧ H ⊨ E⁺        (completeness: it accounts for the positives)
B ∧ H ⊭ E⁻        (consistency: it does not entail the negatives)
```

**Bayesian formulation.** Maintain a posterior over hypotheses rather than selecting one:

```
P(h | D) ∝ P(D | h) · P(h)
```

The three formulations are views of the same act. Regularization `Ω(h)`, the prior `P(h)`, and a description-length penalty are interchangeable expressions of the same commitment, which §9 shows is not optional.

Induction is **ampliative** and **not truth-preserving**: the conclusion asserts more than the data. It is also **non-monotonic**, since one new counterexample can force retraction. Because infinitely many hypotheses fit any finite sample, induction is impossible without an **inductive bias**: a restriction bias from the choice of `H`, a preference bias from `Ω` or the optimizer, or a prior.

### 3.2 Where induction lives in AI/ML

| Area | Mechanism |
|---|---|
| Supervised learning | Linear and logistic regression, SVMs, gradient boosting, random forests, deep nets |
| Unsupervised and self-supervised learning | Clustering, dimensionality reduction, contrastive learning, masked-token pretraining |
| LLM pretraining | Massive induction: a next-token predictor generalized from a corpus |
| Statistical learning theory | PAC learning, VC dimension, Rademacher complexity, generalization bounds |
| Symbolic induction | ILP systems (Progol, Aleph, Popper, ILASP), differentiable ILP, rule mining |
| Program and grammar induction | Programming-by-example (FlashFill), grammar induction, DreamCoder-style library learning |
| Compression-based induction | MDL, Kolmogorov complexity, Solomonoff induction (the idealized, uncomputable limit) |
| Bayesian inference | Posterior over hypotheses; the prior *is* the inductive bias, made explicit |
| Structure learning | Learning Bayes-net or causal graph structure from data |

### 3.3 Example

**Statistical.** Take 50,000 labeled emails and fit a classifier. It converges on weights where tokens such as `wire transfer` and `verify account`, plus unmatched reply-to domains, push toward *phishing*. From roughly 50k particulars a general decision rule is induced. Applied to email 50,001 it predicts *phishing* with 0.94 confidence: a well-supported guess, not a proof. Nothing forbids a legitimate email from containing all of those features.

**Symbolic (ILP), same domain, interpretable output.** Background knowledge and examples:

```prolog
% Background B
sender_domain(e1, acme_co).      link_domain(e1, acme_co).       urgent_language(e1).
sender_domain(e2, acme_co).      link_domain(e2, acme_co).
sender_domain(e3, acme_co).      link_domain(e3, ac_me_co).      urgent_language(e3).
sender_domain(e4, bank_x).       link_domain(e4, bank_x_secure). urgent_language(e4).

% E⁺ = { phishing(e3), phishing(e4) }
% E⁻ = { phishing(e1), phishing(e2) }
```

Induced hypothesis:

```prolog
phishing(E) :- sender_domain(E, D), link_domain(E, L), D \= L.
```

Check it: the rule entails both positives (`e3`, `e4` have mismatched domains), excludes both negatives (`e1`, `e2` match), and is smaller than the competing candidate `phishing(E) :- urgent_language(E)`, which would misclassify `e1`. Note that the induced rule is a **generalization** and it happens to be wrong in general, since legitimate mail uses tracking, CDN, and ESP domains constantly. The data underdetermined the truth, and the rule's readability makes that visible in a way weights would not.

**Compression view.** Consider the sequence `2, 4, 6, 8, ...`. Induction prefers `f(n) = 2n` over

```
g(n) = 2n + (n-1)(n-2)(n-3)(n-4)·k
```

which also fits every observed point and then diverges wildly. Nothing in the *data* rules `g` out. Only the bias does. §9 turns that preference into a theorem.

### 3.4 Limitations

1. **The problem of induction (Hume).** No non-circular justification exists for projecting past regularities into the future. In practice the gap is bridged by assumption, usually i.i.d. sampling from a stationary distribution.
2. **Distribution shift breaks the guarantee.** Generalization bounds are stated relative to the training distribution. Covariate shift, concept drift, label shift, and adversarial inputs all void them. Deployment reality differs from benchmark reality almost always.
3. **Underdetermination.** Infinitely many hypotheses fit any finite dataset. Which one you get is determined by the inductive bias (architecture, regularizer, optimizer, initialization), not by the data alone.
4. **No Free Lunch.** Averaged over all possible target functions, every learner performs identically. There is no universally best inductive bias; a performance claim is always a claim about a *class of problems*, never about learning as such.
5. **Spurious correlation and shortcut learning.** Learners latch onto whatever predicts the label in-sample: hospital scanner artifacts, watermarks, background texture, annotator habits, the position of the correct answer. Statistically valid induction, useless model.
6. **Correlation is not causation.** Purely observational induction cannot identify causal structure without interventions or explicit causal assumptions. A model can be accurate and still give catastrophically wrong answers about the effect of an action.
7. **Sample complexity and data hunger.** Rich hypothesis spaces need many examples. Rare classes, long tails, and expensive labels are where induction is weakest, which is exactly where high-stakes decisions tend to live.
8. **Noise sensitivity and overfitting.** With label noise or leakage, induction faithfully generalizes the wrong thing. Validation protocols catch some of this and are themselves inductive.
9. **Extrapolation failure.** Interpolation inside the data manifold is much stronger than extrapolation outside it, and confidence estimates are typically miscalibrated precisely where they matter.
10. **Opacity.** Sub-symbolic induction yields hypotheses that resist audit, contest, and repair. ILP-style induction trades accuracy and scale for a hypothesis you can read.
11. **Feedback loops.** Deployed inductive models change the distribution they were induced from (recommendation, pricing, policing, credit), invalidating their own training assumption.

---

## 4. Abduction

### 4.1 Definition

Abduction infers the **best explanation** for an observation. Given a background theory `T` and an observation `O`, find an explanation `E` drawn from a designated set of *abducibles* `A` such that:

```
1.  T ∪ E ⊨ O            E accounts for the observation
2.  T ∪ E ⊭ ⊥            E is consistent with what we know
3.  E is "best"          minimal, most probable, or most plausible among candidates
```

Condition 3 is what makes abduction hard and what distinguishes formulations. Common criteria: **minimality** (subset-minimal or cardinality-minimal explanation sets), **probability** (`argmax_E P(E | O)`), **coverage** (explains the most observations), **plausibility** (domain-weighted priors), and **simplicity** (Occam).

Logically, abduction is the *invalid* schema of affirming the consequent: given `p → q`, observe `q`, conclude `p`. It is unsound by construction. Its justification is pragmatic. It is the only mode that **generates hypotheses**, and generation is a prerequisite for testing.

**Probabilistic counterpart.** Most Probable Explanation (MPE) or MAP inference in a graphical model:

```
E* = argmax_e P(E = e | O = o)
```

**Diagnostic counterpart (Reiter).** Given a system description `SD`, a set of components `COMP`, and observations `OBS`, a diagnosis is a minimal `Δ ⊆ COMP` such that

```
SD ∪ OBS ∪ { AB(c) | c ∈ Δ } ∪ { ¬AB(c) | c ∈ COMP \ Δ }
```

is consistent. Minimal diagnoses are the minimal hitting sets of the conflict sets.

### 4.2 Abduction vs. Induction

Both are ampliative and defeasible, and they are constantly conflated. The distinction:

| | Abduction | Induction |
|---|---|---|
| Infers | A **case or cause**: a particular fact | A **rule**: a general regularity |
| Scope | Explains *this* observation | Covers a *population* of observations |
| Needs | A theory `T` linking causes to effects | Many examples |
| Question | "Why did this happen?" | "What happens in general?" |
| Output | `has_flu(patient_7)` | `fever ∧ cough → flu` (probabilistically) |

Abduction operates *within* an existing theory to explain a datum. Induction *builds* the theory. Abduction with an inadequate theory returns the least-bad available explanation and gives no signal that the true cause was never in the search space.

### 4.3 Where abduction lives in AI/ML

| Area | Mechanism |
|---|---|
| Model-based diagnosis | Reiter's theory of diagnosis, GDE, minimal hitting sets, fault localization |
| Medical and technical diagnosis | Parsimonious covering theory, set-cover abduction, the INTERNIST and CADUCEUS lineage |
| Abductive logic programming | ALP, the IFF proof procedure, `abducible` predicates in ASP, ProbLog abduction |
| Probabilistic reasoning | MPE and MAP inference in Bayes nets and MRFs, most-probable-configuration decoding |
| Causal inference | Pearl's counterfactual algorithm is explicitly **abduction, action, prediction**: abduce the exogenous noise `U` from the evidence, intervene, then deduce |
| NLP | Interpretation-as-abduction (Hobbs), coreference and discourse-relation resolution, presupposition, pragmatic enrichment |
| Plan and intent recognition | Inferring goals from partial action sequences, theory-of-mind and user-modeling agents |
| Observability and SRE | Root-cause analysis, alert correlation, anomaly explanation, causal trace analysis |
| Neuro-symbolic systems | Abductive Learning (ABL), where a perception net proposes symbols, a knowledge base abduces consistent revisions, and the revised labels retrain the net; also DeepProbLog and NeurASP |
| Vision and robotics | Scene interpretation as best explanation of sensor data, analysis-by-synthesis |
| Agentic LLM loops | Hypothesis generation for debugging, experiment design, and troubleshooting, then verified by tools |

### 4.4 Example

**Setting.** A background theory plus one observation.

```prolog
% Theory T
timeout(Svc)        :- db_pool_exhausted(Svc).
timeout(Svc)        :- upstream_down(Dep), depends_on(Svc, Dep).
timeout(Svc)        :- deploy_regression(Svc).
error_rate_up(Svc)  :- timeout(Svc).
latency_p99_up(Svc) :- db_pool_exhausted(Svc).

depends_on(checkout, auth_svc).

% Abducibles A = { db_pool_exhausted/1, upstream_down/1, deploy_regression/1 }

% Observation O
error_rate_up(checkout).
```

**Abduction (reasoning backwards from effect to cause, not deduction).** Three minimal candidate explanations:

```
E₁ = { db_pool_exhausted(checkout) }
E₂ = { upstream_down(auth_svc) }
E₃ = { deploy_regression(checkout) }
```

Each satisfies `T ∪ Eᵢ ⊨ O` and each is consistent. All three are logically on par. Selecting among them requires extra criteria:

- **New evidence.** `latency_p99_up(checkout)` is *false*. Under `T`, `E₁` predicts it, so `E₁` is eliminated. Note that this step is deduction: derive a prediction from the hypothesis, then check it.
- **Priors.** If the last deploy was 11 days ago, `P(E₃)` drops sharply.
- **Coverage.** If other dependents of `auth_svc` are also alerting, `E₂` explains more observations with one cause and wins on parsimony.

Conclusion: `upstream_down(auth_svc)` is the best explanation, **not a proven one**. A fourth cause outside the abducibles, say a NAT-gateway port-exhaustion issue nobody modeled, would never be proposed, and the system would confidently return the wrong answer with no indication that it had done so.

**Probabilistic version.** The same problem as MPE in a Bayes net over `{deploy, db_pool, upstream, errors, latency}`:

```
E* = argmax_{d,b,u}  P(deploy=d, db_pool=b, upstream=u | errors=1, latency=0)
```

**Full Peircean cycle.** The three modes compose:

```
ABDUCTION  ⟶  hypothesis:  "auth_svc is down"
DEDUCTION  ⟶  prediction:  "then auth_svc health checks must be failing too"
INDUCTION  ⟶  test:        check metric history across many incidents;
                            if the pattern holds broadly, promote it to a rule
                            (for example, an auto-remediation runbook)
```

### 4.5 Limitations

1. **Formally invalid.** Abduction affirms the consequent. A high-quality explanation can simply be false. Abduction alone never licenses belief, only investigation.
2. **Closed hypothesis space.** Abduction can only return explanations expressible in `A` and `T`. Unmodeled causes are structurally invisible, and the output carries no warning that the true cause was excluded. This is the single most dangerous failure mode in practice.
3. **Combinatorial explosion.** The number of candidate explanations grows exponentially with the abducibles. Deciding whether an explanation exists is NP-hard for propositional abduction, and finding minimum-cardinality or relevance-optimal explanations sits higher still (Σ₂ᵖ-complete for some variants). MPE is NP-hard in general Bayes nets.
4. **"Best" is underdefined and contested.** Minimality, probability, coverage, and simplicity conflict, and the choice of criterion determines the answer. Inference to the Best Explanation has no agreed formal semantics; it smuggles in value judgments about what makes an explanation good.
5. **Prior sensitivity.** Probabilistic abduction inherits all the fragility of its priors, and the priors for rare faults and rare diseases are exactly the ones least well estimated.
6. **MPE pathologies.** The most probable *joint* explanation can assign a variable a value that is improbable in its own marginal. The single best explanation may carry tiny absolute probability while a broad set of alternatives collectively dominates. Reporting one explanation hides both effects.
7. **Multiple simultaneous causes.** Minimality biases toward single-fault explanations. Real incidents and real patients frequently have two or three interacting causes, and minimal-cardinality abduction actively suppresses those.
8. **Explanation is not cause.** Without causal structure, an abduced "explanation" may be a mere correlate that happens to entail the observation under a mis-specified theory.
9. **Confabulation risk in LLMs.** LLMs are fluent abducers. They produce highly plausible causal stories with no grounding, no calibration, and no representation of the abducible space. The fluency of an explanation is uncorrelated with its correctness, which makes verification non-optional.
10. **Parasitic on a theory.** Abduction requires `T` to exist first. If `T` came from weak induction or hand-authored guesswork, the explanations inherit those defects.

---

## 5. Side-by-Side Comparison: The Three Classical Modes

| Dimension | Deduction | Induction | Abduction |
|---|---|---|---|
| Direction | Rule + Case ⟶ Result | Case + Result ⟶ Rule | Rule + Result ⟶ Case |
| Formal core | `T ∪ F ⊨ O` | `argmin_h Ê[L]`, or `B∧H ⊨ E⁺`, `B∧H ⊭ E⁻` | `T ∪ E ⊨ O`, `E` best |
| Truth-preserving | Yes | No | No |
| Ampliative | No | Yes | Yes |
| Monotonic | Yes | No | No |
| Output | Certain consequence | General hypothesis | Candidate explanation |
| Uncertainty | None, or an entailed posterior | Statistical and generalization error | Ranked plausibility |
| Needs | Axioms | Data | Theory plus observation |
| Typical tools | Prolog, Z3, OWL reasoners, Lean | SGD, boosting, ILP, Bayesian inference | Diagnosis engines, ALP, MPE, ABL |
| Core question | "What follows?" | "What generalizes?" | "What explains this?" |
| Signature failure | Brittle, empty, intractable | Spurious correlation, shift | Plausible but wrong, blind spots |
| Role in science | Prediction and verification | Confirmation and generalization | Hypothesis generation |

---

## 6. Transduction

### 6.1 Definition

Transduction (Vapnik) reasons from specific observed cases directly to specific known cases, **skipping the general rule entirely**. Where induction produces a hypothesis usable on any future input, transduction produces only labels for a test set that is already in hand.

```
INDUCTION:    D = {(x₁,y₁)...(x_l,y_l)}  ⟶  h ∈ H,  then h(x) for any x
TRANSDUCTION: D  +  {x_{l+1},...,x_{l+u}}  ⟶  (ŷ_{l+1},...,ŷ_{l+u})   directly
```

The objective differs accordingly. Induction minimizes expected risk over the whole distribution; transduction minimizes error on the `u` given points only:

```
inductive risk:     R(h) = E_{(x,y)~P}[ L(h(x), y) ]
transductive risk:  R_T  = (1/u) Σ_{i=l+1}^{l+u} L(ŷᵢ, yᵢ)
```

Vapnik's motivating principle: when solving a problem of interest, do not solve a more general problem as an intermediate step. Inferring a function that works everywhere is strictly harder than labeling the finite set you actually care about, and the unlabeled test inputs themselves carry exploitable information in the form of cluster structure, manifold geometry, and marginal density.

Transduction is not a separate *logical* mode. It is still ampliative and defeasible. It is a distinct **commitment level**: it declines to output a general rule at all.

### 6.2 Where transduction lives in AI/ML

| Area | Mechanism |
|---|---|
| Classical methods | Transductive SVM, transductive regression, lazy learners (k-NN, kernel regression) |
| Graph-based methods | Label propagation, label spreading, spectral methods, graph min-cut |
| Graph neural networks | Transductive node classification (a vanilla GCN trained on the full graph including test nodes) vs. inductive GNNs (GraphSAGE, which learns aggregators reusable on unseen nodes) |
| Semi-supervised learning | Cluster, manifold, and low-density-separation assumptions applied to the known test pool |
| Few-shot learning | Transductive few-shot classification, batch-level statistics over the query set |
| Test-time methods | Test-time training and adaptation, entropy minimization on the test batch, transductive fine-tuning |
| Uncertainty quantification | Transductive conformal prediction, recomputing over the pooled calibration and test set |
| Recommenders | Matrix factorization over a fixed user-item matrix, where embeddings do not extend to new users without retraining |
| Program-vs-direct prediction | On ARC-AGI, "induction" (synthesize a latent function, then apply it) and "transduction" (predict the test grid directly) are explicitly separated. They solve *different* task types, with induction stronger on precise multi-step computation and transduction stronger on fuzzier perceptual patterns. Ensembles of both outperform either alone, which is empirical evidence that the two modes are complementary rather than one being universally better |

### 6.3 Example

**Setting.** 200 labeled support tickets, plus a fixed backlog of 100,000 unlabeled tickets that must all be triaged tonight. No new tickets arrive during the run.

**Inductive approach.** Fit a classifier on the 200, then apply it to each of the 100,000 independently. Each prediction uses only that ticket's features and the induced weights.

**Transductive approach.** Build a k-NN similarity graph over all 100,200 tickets, then propagate labels:

```
F ← D^(-1/2) W D^(-1/2)                    # normalized affinity over ALL points
Ŷ^(t+1) ← α F Ŷ^(t) + (1-α) Y_labeled      # iterate to a fixpoint
```

Labels flow through dense regions of the *test* set. A cluster of 4,000 tickets about one specific payment error receives a coherent label because the cluster exists, even though only one of the 200 labeled examples touched that topic. No function `h` is ever produced: the output is 100,000 labels and nothing more.

**The sharp consequence.** Run the same algorithm on a different test batch and the same ticket can receive a different label, because its prediction depends on its neighbours in the batch. No inductive model can reproduce that behavior, which is precisely the point: transduction exploits information that a per-point function has no access to.

### 6.4 Limitations

1. **No reusable artifact.** There is no model to ship, version, or serve. Every new batch requires re-running the full computation, often at `O(n²)` or `O(n³)` cost for graph and kernel methods.
2. **Unusable for online serving.** Transduction requires the test inputs at fit time. Real-time request handling, streaming, and any latency-bound path rule it out.
3. **Predictions are batch-dependent and can be mutually inconsistent.** Different test sets yield different labels for the same point. This is a feature theoretically and a liability operationally, since results are not reproducible, auditable, or explainable per instance.
4. **Not automatically easier.** The intuition that transduction is a strictly simpler problem is only partly borne out. Minimax analyses of transductive classification show the gains are conditional, not universal.
5. **Inherits all semi-supervised failure modes.** The cluster, manifold, and smoothness assumptions can be false. When they are, unlabeled data actively *degrades* accuracy relative to ignoring it.
6. **Evaluation hygiene.** Touching test inputs during fitting muddies the train/test boundary. It is legitimate when only the *inputs* are used, and it quietly becomes leakage the moment labels, target statistics, or repeated tuning on the test pool creep in.
7. **Distribution shift is hidden, not solved.** Transduction adapts to the batch it sees. It gives no signal about whether that batch resembles anything else, and no guarantee beyond it.
8. **Terminological sloppiness.** "Transductive" and "semi-supervised" are routinely conflated. Semi-supervised learning uses unlabeled *training* data but still aims at an inductive model; transduction targets a specific known test set.

---

## 7. Non-Monotonic Reasoning

### 7.1 Definition

Classical deduction is monotonic: conclusions survive the addition of premises. Commonsense reasoning is not, because we routinely draw conclusions that are *defeasible*, held only in the absence of contrary information. Non-monotonic reasoning (NMR) formalizes a consequence relation `⊢` for which:

```
T ⊢ φ        but        T ∪ {ψ} ⊬ φ
```

No premise was retracted and nothing became inconsistent. The additional information simply defeated the default. NMR is best read as **deduction relaxed to accommodate defeasibility while staying symbolic**, recovering some of induction's flexibility without giving up an auditable rule base.

A clarification the standard tables often get wrong: NMR conclusions *are* ampliative in the informal sense that matters. `flies(tweety)` is not classically entailed by `bird(tweety)` plus the default; that is exactly why adding `penguin(tweety)` can defeat it. What NMR does not do is add *empirical* content: it derives nothing that was not already implicit in the hand-authored defaults. It buys revisability, not knowledge.

### 7.2 The main formalisms

| Formalism | Core device |
|---|---|
| **Default logic** (Reiter, 1980) | A default theory `(D, W)`: facts `W` plus defaults written `α : β / γ`. If prerequisite `α` holds and justification `β` is consistent with current beliefs, conclude `γ`. Belief sets are *extensions*, defined as fixpoints |
| **Circumscription** (McCarthy, 1980) | Minimize the extension of abnormality predicates `AB(·)` and reason over minimal models only. Things are normal unless forced otherwise |
| **Negation as failure and stable models** (Clark; Gelfond and Lifschitz, 1988) | `not p` succeeds when `p` is not derivable. The stable-model semantics gave this a clean fixpoint meaning and became the basis of **Answer Set Programming** (clingo, DLV) |
| **Autoepistemic logic** (Moore, 1985) | Modal reasoning about one's own beliefs: `¬L p → q`. Later used to interpret normal logic programs |
| **Belief revision and TMS** (AGM; Doyle; de Kleer) | Truth maintenance systems and ATMS track justifications so conclusions can be retracted and dependencies recomputed |
| **Argumentation** (Dung, 1995) | Arguments attack each other; acceptable conclusions are those in the grounded, preferred, or stable extensions of the attack graph |
| **KLM framework** (Kraus, Lehmann and Magidor, 1990) | Axiomatizes defeasible consequence itself (reflexivity, cut, cautious monotonicity, rational monotonicity) rather than building a specific logic |

Historical note: circumscription and default logic turned out to be closely related to the semantics of negation in logic programming, which is why ASP inherited so much of this lineage and is today the most practically used branch.

### 7.3 Example

**Answer Set Programming.** Defaults with exceptions, and no exception list inside the rule itself:

```prolog
bird(X)     :- penguin(X).
flies(X)    :- bird(X), not abnormal(X).      % default
abnormal(X) :- penguin(X).                    % exception
abnormal(X) :- injured(X).

bird(tweety).
```

The answer set contains `flies(tweety)`. Now add one fact:

```prolog
penguin(tweety).
```

The new answer set contains `abnormal(tweety)` and **not** `flies(tweety)`. A previously derived conclusion was retracted without any contradiction arising. In classical first-order logic, `∀x. bird(x) → flies(x)` together with `penguin(tweety) → ¬flies(tweety)` is flatly inconsistent, and the only monotonic repair is enumerating every exception in the antecedent forever, which is the qualification problem from §2.4.

**Default-logic rendering of the same rule:**

```
        bird(x) : flies(x)
        ──────────────────
             flies(x)
```

Read: if `x` is a bird and believing that `x` flies is consistent with current beliefs, believe it.

**The ambiguity problem in three lines** (the Nixon diamond):

```prolog
pacifist(X)     :- quaker(X),     not neg_pacifist(X).
neg_pacifist(X) :- republican(X), not pacifist(X).
quaker(nixon).  republican(nixon).
```

Two answer sets result, one containing `pacifist(nixon)` and one containing `neg_pacifist(nixon)`. The formalism does not pick. Under *skeptical* entailment neither is concluded; under *credulous* entailment either may be. There is no formalism-internal reason to prefer one.

### 7.4 Limitations

1. **Multiple extensions.** Competing defaults yield several equally sanctioned belief sets, and the logic offers no tie-break. Choosing skeptical or credulous semantics is a design decision with no principled default.
2. **Authoring is notoriously hard.** A recurring, well-documented finding in this literature is the difficulty of formalizing a domain such that the *intended* conclusions actually follow. Specificity, inheritance, and exception interactions produce surprises at scale.
3. **The Yale shooting problem.** Naive temporal minimization admits models in which the gun mysteriously unloads instead of the victim dying, a famous demonstration that plausible-looking non-monotonic frame axioms can be subtly wrong.
4. **Complexity.** Deciding whether a default theory has an extension is Σ₂ᵖ-complete. Consistency checking for normal logic programs is NP-complete, and Σ₂ᵖ-complete with disjunction. Grounding first-order ASP programs can blow up combinatorially before solving even starts.
5. **No graded uncertainty.** Defeasibility is qualitative. "Birds normally fly" cannot be tuned to 0.97, cannot be combined with likelihoods, and cannot express calibrated confidence. Probabilistic logic programming (ProbLog, LP-MLN) exists precisely to fill this gap.
6. **Semantic proliferation without consensus.** Well-founded, stable, supported, preferred, grounded: different semantics give different answers on the same program, so results are formalism-relative rather than absolute.
7. **Breaks modularity.** Adding a rule can silently alter unrelated conclusions elsewhere in the program, because defaults interact through global consistency. Unit-testing a non-monotonic knowledge base is much harder than testing a monotonic one.
8. **Still hand-authored.** NMR softens deduction's brittleness but does nothing about the knowledge-acquisition bottleneck. The rules and the abnormality predicates still come from people.

---

## 8. Analogical Inference

### 8.1 Definition

Analogical inference transfers structure from a familiar **base** (source) case to an unfamiliar **target** case, then projects additional base properties onto the target:

```
Base:    entities {a, b, ...} with relations R and property P
Target:  entities {a', b', ...} with relations R' ≈ R
Infer:   P holds in the target too
```

Under Gentner's **structure-mapping theory** (1983), analogy is a mapping `M` from base to target that maximizes:

- **Structural consistency**: one-to-one correspondences plus *parallel connectivity*, meaning that if relations are mapped then their arguments must be mapped too.
- **Systematicity**: prefer deep, interconnected *relational* structure over shared surface attributes.

Candidate inferences are exactly the base predicates not yet present in the target, carried across by `M`. The computational realization is the Structure Mapping Engine (Falkenhainer, Forbus and Gentner, 1989), still the most influential model of analogy-making.

Analogy is best treated as a **compound** mode rather than a fourth primitive. A common decomposition: **abduction** hypothesizes the mapping, **projection** deduces consequences under it, and **induction** generalizes the shared relational schema across cases.

### 8.2 Where analogical inference lives in AI/ML

| Area | Mechanism |
|---|---|
| Case-based reasoning | Retrieve, Reuse, Revise, Retain; case memories with indexing and similarity metrics |
| Cognitive models | SME, MAC/FAC (cheap retrieval then expensive mapping), Copycat, LISA, ACME |
| Legal AI | Reasoning from precedent (HYPO, CATO), factor-based case comparison |
| Knowledge graphs | Relational embeddings and link prediction, analogy-structured objectives, multimodal analogy benchmarks |
| Word and sentence embeddings | Parallelogram analogies (`king - man + woman ≈ queen`) |
| Transfer learning | Domain adaptation, fine-tuning, and cross-task transfer as analogy at scale |
| LLMs | In-context learning from exemplars, analogical prompting (the model generates its own relevant exemplars before solving), few-shot chain-of-thought |
| Benchmarks | ARC-AGI is explicitly an analogy and abstraction benchmark: infer the relational transformation from 2 to 5 demonstrations, then apply it to a new grid |
| Engineering and design | Reuse of prior designs, root-cause reasoning by precedent, incident retrospectives |

### 8.3 Example

**Base case (known).** A shared token-bucket rate limiter served all tenants from one bucket. One tenant's traffic spike drained the bucket, so every other tenant was throttled. The fix was to partition into per-tenant buckets.

**Target case (new).** A shared database connection pool serves all tenants. The symptoms differ superficially: timeouts rather than 429s.

**Structure mapping:**

```
BASE                                 TARGET
  bucket                      ↦        pool
  token                       ↦        connection
  tenant                      ↦        tenant
  consumes(tenant, token)     ↦        consumes(tenant, connection)
  finite_shared(bucket)       ↦        finite_shared(pool)
  causes(exhaustion(bucket),
         denial(other_tenants))  ↦     [candidate inference]
  fixes(partition(bucket),
        denial)                   ↦     [candidate inference]
```

**Projected inferences.** Exhaustion by one tenant causes denial for the others, and partitioning the pool per tenant (or imposing per-tenant quotas) resolves it.

Notice what carried the inference: the *relational* structure of a finite shared resource consumed competitively with no isolation, not surface attributes. That both services are written in the same language, owned by the same team, and deployed in the same region are shared attributes supporting **no** inference at all. Distinguishing these two kinds of similarity is the entire content of structure-mapping theory.

**Where it fails.** Map the same base onto a CPU-scheduling problem and "partition the resource per tenant" projects badly, because CPU time is preemptible and work-conserving, so static partitioning wastes capacity. The relation `finite_shared` mapped but `non_preemptible` did not, and the projected fix depended on the relation that failed to map. Analogies fail exactly when the causally relevant relation is the one that does not carry over.

**Vector analogies, briefly.** `v(king) - v(man) + v(woman) ≈ v(queen)` is analogical inference reduced to arithmetic in an induced space. It works for some relation types and is far weaker than the headline suggests; see limitation 8.

### 8.4 Limitations

1. **Not truth-preserving, and validity hinges on relevance.** An analogy can be structurally perfect and still project the wrong property, because structural alignment is not causal alignment.
2. **The retrieval and soundness gap.** What gets retrieved from memory is driven largely by *surface* similarity, while what makes an analogy sound is *relational* similarity. Humans and systems both retrieve superficially similar but inferentially useless analogues, and miss deep matches with different surface features.
3. **Total representation dependence.** SME-style mapping is only as good as the hand-built predicate encoding of both cases. There is no principled procedure for producing those encodings, and different encodings of the same situation yield different analogies. This is the main reason symbolic analogy engines did not scale.
4. **Combinatorial cost.** Optimal structural matching is a graph-matching problem, NP-hard in general. Practical engines use greedy and heuristic merges, so the mapping found is not guaranteed optimal.
5. **No stopping rule for projection.** Which base predicates should be carried over? Project too little and the analogy is useless; project too much and false properties are imported. Systematicity is a heuristic preference, not a criterion.
6. **The adaptation problem in CBR.** Retrieved cases rarely fit the new situation exactly, and revising them requires domain knowledge the case base does not contain. Cross-domain retrieval and adaptation remain largely unsolved in deployed CBR systems.
7. **`n = 1`.** A single precedent provides no statistical support. Analogical conclusions have high variance and are exquisitely sensitive to which case happened to be retrieved.
8. **Vector analogies are overstated.** Parallelogram results depend on excluding the query terms from the candidate set, are sensitive to normalization and word frequency, and hold well for a narrow band of relations (morphology, some geography) while failing broadly elsewhere.
9. **LLM analogy is often pattern recall.** Performance drops sharply on counterfactual or low-frequency variants of familiar analogy problems, which suggests retrieval of memorized surface patterns rather than genuine structural mapping. Verification against the target domain remains necessary.

---

## 9. Solomonoff Induction

### 9.1 Definition

Solomonoff induction is the formal ideal of induction: a single, universal, parameter-free prior over all computable hypotheses. Fix a universal monotone (or prefix) Turing machine `U`. The **universal or algorithmic prior** of a string `x` is the total probability of all programs that output something beginning with `x`:

```
M(x) := Σ_{p : U(p) = x*}  2^(-ℓ(p))
```

where `ℓ(p)` is the program's length in bits and `x*` is any string with prefix `x`. Prediction is conditional probability:

```
M(x_{n+1} | x₁...x_n) = M(x₁...x_{n+1}) / M(x₁...x_n)
```

Equivalently, up to a multiplicative constant, `M` is a Bayesian mixture over all lower-semicomputable semimeasures `ν`, weighted by their complexity:

```
ξ = Σ_ν w(ν) · ν        with     w(ν) = 2^(-K(ν))
```

where `K(·)` is prefix Kolmogorov complexity. Two properties make this the reference point for all of induction.

- **Occam's razor becomes a theorem, not a taste.** Shorter programs contribute exponentially more mass. The simplicity bias that §3 identified as *necessary* for induction is here derived from program length rather than chosen by an engineer.
- **Completeness.** For any computable distribution `μ` generating the data, `M`'s predictions converge to `μ`'s, with total expected squared prediction error over the whole infinite sequence bounded by roughly `K(μ) · ln 2`, a constant independent of sequence length. It learns any computable environment with essentially the minimum possible data.

**Generalization to action.** AIXI (Hutter) couples Solomonoff induction with sequential decision theory, combining a universal Bayesian mixture over computable environments with expected-reward maximization, yielding a formally optimal agent for computable environments.

**The central negative theorem.** Computability and completeness are mutually exclusive. Any complete induction method is uncomputable. The proof is a diagonalization: for any computable predictor, construct a computable environment that outputs the negation of whatever that predictor predicts. This is a sharp instance of the No Free Lunch phenomenon from §3.4.

### 9.2 Why it matters practically

Solomonoff induction is not an algorithm. It is the yardstick that explains *why* practical inductive machinery is shaped the way it is.

| Practical descendant | Connection |
|---|---|
| MDL, BIC, minimum message length | Computable proxies for program length as a model-selection criterion |
| Compression-based learning | Normalized compression distance, gzip-based classifiers, the Coding Theorem link between probability and compressibility |
| Search-based synthesis | Levin search, Hutter search, DreamCoder-style library learning, all preferring shorter programs |
| AIXI approximations | MC-AIXI-CTW and related bounded agents, with the Speed prior as a compute-aware variant |
| LLM pretraining | Next-token log-loss *is* a compression objective. There is an active line of work framing LLMs as computable approximations to Solomonoff induction, and compression benchmarks such as the Hutter Prize as intelligence proxies |
| Regularization | Any length or complexity penalty (L1, weight decay, sparsity, low-rank constraints) is a crude stand-in for a complexity prior |

### 9.3 Example

**Sequence A:** `0 1 0 1 0 1 0 1 0 1 ...`

Programs producing this include `while true: print "01"`, a handful of bits. Its `2^(-ℓ)` weight is enormous relative to any program that reproduces the prefix and then deviates, since such a program must *encode the deviation*, costing extra bits. So `M` predicts `0` next with probability close to 1, and confidence rises exponentially with each additional confirming symbol because alternative programs must encode a longer and longer exception.

**Sequence B:** an incompressible 10,000-bit string. Here `K(x) ≈ |x|`: the shortest program is essentially "print this literal." `M` assigns near-uniform probability to the next bit. Solomonoff induction thus draws a formal line between data that supports generalization and data that does not. The prior itself reports "nothing to learn here," which no practical learner does.

**Tie-back to §3.3.** Given `2, 4, 6, 8`, why prefer `f(n) = 2n` over `g(n) = 2n + (n-1)(n-2)(n-3)(n-4)·k`? Both fit every observation. Under `M`, `g` requires encoding the entire quartic correction term and the constant `k`, so it receives exponentially less prior mass. The preference is not arbitrary taste. It is a consequence of measuring hypotheses in bits.

### 9.4 Limitations

1. **Uncomputable.** `K` is not computable and `M` is only lower semicomputable. Nothing that runs is Solomonoff induction. This is not an engineering gap that better hardware closes; it is a theorem.
2. **Not merely slow, and not boundedly approximable.** Approximations such as Levin search carry astronomically large constants and offer no useful error bound at realistic compute budgets.
3. **UTM dependence is unbounded on finite data.** Universality holds only up to an additive constant arising from the choice of reference machine. That constant can be arbitrarily large, so for any *finite* dataset the choice of machine or encoding can dominate the conclusion. Asymptotic invariance is cold comfort at `n = 1000`.
4. **Assumes a computable environment.** If the data-generating process is not computable, or not in the semimeasure class, the completeness guarantee simply does not apply, and there is no internal signal that this has happened.
5. **`M` is a semimeasure, not a measure.** Probabilities need not sum to one, because some programs never halt or never extend. Normalization is an additional choice, and different normalizations give different predictions.
6. **Ignores the cost of running hypotheses.** A short but astronomically slow program outranks a slightly longer, fast one. The Speed prior exists to patch this, at the cost of losing universality.
7. **The description language must be fixed before seeing data.** Choosing it post hoc lets you make any hypothesis look short, which voids the whole construction.
8. **Normative, not constructive.** It tells you what an ideal learner would conclude. It gives no guidance on designing the sample-efficient, domain-appropriate inductive biases that real systems need, and No Free Lunch still applies to everything computable you might build instead.
9. **AIXI adds its own problems.** No self-model (the agent is not part of its own environment model), susceptibility to reward hacking and wireheading, no general exploration guarantee, and undefined behavior when the true environment falls outside its hypothesis class. That last item is the same closed-hypothesis-space failure that afflicts abduction (§4.5), at maximum scale.

---

## 10. Extended Comparison: All Seven Modes

| Mode | Direction | Truth-preserving | Adds content beyond premises | Monotonic | Yields a general rule | Core question | Signature risk |
|---|---|---|---|---|---|---|---|
| **Deduction** | Rule + Case ⟶ Result | Yes | No | Yes | n/a (consumes rules) | What must follow? | Brittle, empty, intractable |
| **Induction** | Case + Result ⟶ Rule | No | Yes, empirically | No | Yes | What generalizes? | Spurious correlation, shift |
| **Abduction** | Rule + Result ⟶ Case | No | Yes, empirically | No | No (infers a fact) | What explains this? | Plausible but wrong, blind spots |
| **Transduction** | Cases + known test inputs ⟶ test labels | No | Yes, empirically | No | **No, by design** | What are *these* answers? | Batch-dependent, non-reusable |
| **Non-monotonic** | Rule + Case ⟶ Result, defeasibly | No | Yes, but only beyond *classical* entailment; no new empirical content | **No** | n/a (consumes defaults) | What follows *for now*? | Multiple extensions, hard to author |
| **Analogical** | Base case + structure ⟶ target property | No | Yes, empirically | No | Only after generalization | What resembles this, relevantly? | Wrong mapping, surface similarity |
| **Solomonoff** | All data ⟶ weighted set of all computable rules | No | Yes, empirically | No | Yes, in the ideal limit | What is the simplest sufficient explanation? | Uncomputable, UTM-dependent on finite data |

**How they relate:**

- **Non-monotonic reasoning** is deduction with defeasibility bolted on. It adds no empirical content; it just declines to be permanent.
- **Transduction** is induction with the generalization step removed.
- **Analogy** is abduction (find the mapping) plus deduction (project under it) plus induction (schematize across cases).
- **Solomonoff induction** is induction taken to its formal limit, and it is where the necessity of inductive bias stops being a design heuristic and becomes a mathematical fact.
- **Abduction, induction, analogy, and transduction are all non-monotonic.** Only deduction is not.
- **Every ampliative mode has a hypothesis-space failure.** Induction cannot select a hypothesis outside `H`, abduction cannot propose a cause outside `A`, analogy cannot map structure absent from the case base, and even AIXI is undefined outside its environment class. The failure is silent in all four cases.

---

## 11. Terminological Traps

Six confusions that recur constantly and that reliably hide where a system's guarantees actually are.

| Trap | The correction |
|---|---|
| "Model inference" means induction | Serving-time prediction is **deduction relative to an induced hypothesis**. Induction happened at training time. |
| Logic equals deduction, ML equals induction | ILP is symbolic induction. Neural nets inside verified loops participate in deduction. The modes cut across the symbolic/statistical divide. |
| Bayesian updating is induction | Computing a posterior from a *fixed* model is deduction within the probability calculus. Induction is the selection or weighting of the model class itself. |
| Abduction is just induction with less data | Abduction infers a **particular fact** under an existing theory. Induction infers a **general rule**. They fail differently: abduction is blind to unmodeled causes, induction is blind outside its distribution. |
| Semi-supervised equals transductive | Semi-supervised learning uses unlabeled training data and still targets an inductive model. Transduction targets a specific, known test set and produces no model. |
| Chain-of-thought is reasoning | Chain-of-thought is induced text that imitates the *form* of proof. It has no soundness guarantee. Only an external checker supplies one. |

---

## 12. Where LLMs Sit

LLM behavior spans every mode in this document, which is precisely why claims about "LLM reasoning" are so easy to make and so hard to evaluate. Locating each behavior in the taxonomy makes the guarantees, and their absence, explicit.

| LLM behavior | Mode | Guarantee status |
|---|---|---|
| Pretraining on a corpus | Induction, at extreme scale | Standard inductive risk. Subject to shift, shortcuts, and memorization. |
| Sampling a token given context | Deduction relative to the induced hypothesis, plus stochastic decoding | The forward pass is deterministic given weights and context. That guarantees nothing about correctness. |
| In-context learning from exemplars | Analogy, with a transductive flavor | Conditions on specific examples to answer a specific query, producing no reusable artifact beyond the context. |
| Chain-of-thought | Imitation of deduction | No soundness guarantee. Error compounds along the chain. |
| "The bug is probably in the cache layer" | Abduction | Fluent and uncalibrated. The abducible space is unrepresented, so unmodeled causes are invisible and unflagged. |
| Tool use, code execution, proof checking | Genuine deduction, outsourced | The guarantee lives in the checker, not the model. This is the only place a hard guarantee is available. |
| Self-consistency and majority voting | Crude induction over samples | Reduces variance. Does not detect systematic error, since a confidently wrong mode wins the vote. |
| Retrieval-augmented generation | Better premises for the same abduction | Improves the inputs. Changes nothing about the validity of the inference performed on them. |

Three consequences worth stating plainly.

1. **Fluency is uncorrelated with validity.** An LLM's most polished output is its abductive output, and abduction is the mode with no soundness guarantee at all. Explanation quality is not evidence of explanation correctness.
2. **Verification must be external.** A model cannot certify its own deduction, because the imitation of proof form and the possession of proof validity are different properties. Lean kernels, SMT solvers, type checkers, test suites, and executable code are the load-bearing components.
3. **The productive framing is division of labor.** Use the LLM where hypothesis generation is the bottleneck, which is where abduction and analogy genuinely excel and where a wrong-but-cheap proposal costs little. Use verifiers where correctness is required. This is the Peircean loop from §4.4 with the modes assigned to the components suited to them.

---

## 13. Practical Implications for System Design

- **Match the mode to the requirement.** Hard guarantees (safety envelopes, access control, financial invariants, type safety) belong to deduction. Perception and pattern recognition belong to induction. Diagnosis, root cause, and intent belong to abduction. Using an inductive component where a deductive guarantee is required is a category error that no amount of accuracy fixes.
- **Decide explicitly whether you need a model or just answers.** If the test set is finite, known, and batch-processed, transduction is often stronger and simpler. If you need to serve arbitrary future requests under latency constraints, you need induction. Do not build a transductive pipeline and then discover you cannot deploy it.
- **Reach for non-monotonic formalisms when rules have exceptions but must stay auditable.** ASP and default logic are the right tools when a regulator or an on-call engineer has to read the rule, and when probabilities are unavailable or inappropriate. Treat the multiple-extension ambiguity as a design question to be resolved, not a bug to be reported.
- **Treat analogies as hypotheses, not conclusions.** Any analogy, whether from a CBR retrieval or an LLM asserting "this is just like X," should be checked for whether the *causally relevant* relations mapped, then validated deductively or empirically.
- **Never trust an abduction without a test.** Turn every abduced hypothesis into a deductive prediction and check it, or accumulate cases inductively. Abduction proposes; deduction and induction dispose.
- **Audit the hypothesis space, not just the answer.** For induction, ask what bias selected this hypothesis. For abduction, ask which explanations were absent from the abducible set. For analogy, ask what encoding produced the mapping. All of these fail silently and confidently when the truth lies outside the space searched, including at the theoretical limit (§9.4).
- **Put the guarantee in a component that can carry one.** If a requirement is stated as a guarantee, it must be discharged by a verifier, a solver, a type system, or a proof, never by a sufficiently accurate predictor. Accuracy is a statistical property; a guarantee is a logical one, and they are not exchangeable at any accuracy level.
- **Combine deliberately.** The strongest architectures close the Peircean loop: induction learns raw-signal-to-symbol mappings, abduction proposes explanations consistent with a knowledge base, deduction certifies conclusions and rejects violations, and non-monotonic machinery handles defaults and revision. Abductive Learning, DeepProbLog, NeurASP, and LLM-plus-verifier setups are instances of this pattern. The ARC result, where induction-only and transduction-only models each plateau far below their ensemble, is the empirical version of the same lesson.
- **Be precise with the word "inference."** Serving-time model inference is deduction relative to an induced hypothesis. Training is induction. Label propagation over a known test set is transduction. An LLM narrating a cause is abduction. Conflating them hides where the guarantees actually are, and where they are not.

---

## 14. Key References

Reproduced and lightly corrected from the source draft. These citations have not been verified against the literature here, and this document was written without search access, so titles, years, and attributions should be checked before being quoted or cited.

**Foundational**
- C.S. Peirce, *Collected Papers*. The deduction, induction, abduction trichotomy.
- D. Hume, *An Enquiry Concerning Human Understanding*. The problem of induction.

**Deduction and logic in AI**
- J.A. Robinson (1965), A machine-oriented logic based on the resolution principle.
- R. Kowalski, *Logic for Problem Solving*.
- F. Baader et al., *The Description Logic Handbook*.

**Induction**
- V. Vapnik, *Statistical Learning Theory* (1998); *Estimation of Dependences Based on Empirical Data* (1982; 2nd ed. 2006, whose afterword develops transductive and empirical inference).
- D. Wolpert (1996), The lack of a priori distinctions between learning algorithms (No Free Lunch).
- S. Muggleton and L. De Raedt (1994), Inductive logic programming: theory and methods.
- A. Cropper and S. Dumančić (2022), Inductive logic programming at 30.

**Abduction**
- R. Reiter (1987), A theory of diagnosis from first principles.
- A. Kakas, R. Kowalski and F. Toni (1992), Abductive logic programming.
- J. Hobbs et al. (1993), Interpretation as abduction.
- J. Pearl, *Causality*. The abduction, action, prediction algorithm for counterfactuals.
- Z.-H. Zhou (2019), Abductive learning: towards bridging machine learning and logical reasoning.

**Transduction**
- V. Vapnik (1998; 2006). Origin of transductive inference.
- T. Joachims (1999), Transductive inference for text classification using support vector machines.
- X. Zhu and Z. Ghahramani (2002), Learning from labeled and unlabeled data with label propagation.
- D. Zhou et al. (2004), Learning with local and global consistency (label spreading).
- W. Hamilton et al. (2017), Inductive representation learning on large graphs (GraphSAGE).
- R. El-Yaniv and D. Pechyony (2008), Transductive Rademacher complexity and its applications.
- W. Li et al. (2024), *Combining Induction and Transduction for Abstract Reasoning*, arXiv:2411.02272.
- ARC Prize 2024 Technical Report, arXiv:2412.04604.

**Non-monotonic reasoning**
- R. Reiter (1980), A logic for default reasoning.
- J. McCarthy (1980), Circumscription: a form of non-monotonic reasoning.
- R. Moore (1985), Semantical considerations on nonmonotonic logic.
- M. Gelfond and V. Lifschitz (1988), The stable model semantics for logic programming.
- S. Kraus, D. Lehmann and M. Magidor (1990), Nonmonotonic reasoning, preferential models and cumulative logics.
- P.M. Dung (1995), On the acceptability of arguments and its fundamental role in nonmonotonic reasoning.
- S. Hanks and D. McDermott (1987), Nonmonotonic logic and temporal projection (the Yale shooting problem).
- G. Brewka, T. Eiter and M. Truszczyński (2011), Answer set programming at a glance.

**Analogical inference**
- D. Gentner (1983), Structure-mapping: a theoretical framework for analogy.
- B. Falkenhainer, K. Forbus and D. Gentner (1989), The Structure-Mapping Engine.
- D. Gentner and K. Forbus (1995), MAC/FAC: a model of similarity-based retrieval.
- D. Gentner, M.J. Rattermann and K. Forbus (1993), The roles of similarity in transfer (the retrieval and soundness gap).
- A. Aamodt and E. Plaza (1994), Case-based reasoning: foundational issues, methodological variations.
- M. Yasunaga et al. (2023), *Large Language Models as Analogical Reasoners*, arXiv:2310.01714.
- F. Chollet (2019), On the measure of intelligence (ARC).

**Solomonoff induction**
- R. Solomonoff (1964), A formal theory of inductive inference, parts I and II; (1978) Complexity-based induction systems.
- M. Li and P. Vitányi, *An Introduction to Kolmogorov Complexity and Its Applications*.
- M. Hutter (2004), *Universal Artificial Intelligence* (AIXI).
- J. Leike and M. Hutter (2015), On the computability of Solomonoff induction and AIXI.
- J. Rissanen (1978), Modeling by shortest data description (MDL).
- T. Sterkenburg (2016), A generalized characterization of algorithmic probability, arXiv:1508.05733.