# The Machine Learning Mindset: A Detailed Guide

Oct 19, 2026 · @Nikolas Markou

## What This Guide Is

This guide distills the implicit framework of reasoning, problem formulation and epistemology that 19 machine learning and ML-adjacent texts teach in common. Its organizing claim: the field's methods are responses to **epistemic incompleteness**, the gap between what must be known and what the available evidence can establish.

**Scope.** The corpus is probabilistic, statistical and decision-theoretic. It contains no dedicated deep learning text, and only Resource 06 treats representation learning at length. Its theoretical backbone (Mitchell, MacKay, Boyd and Vandenberghe, Rasmussen and Williams, Hastie et al., Koller and Friedman, Barber) was published between 1997 and 2012. The mindset described here is therefore the classical one. Where modern deep learning practice departs from it, the text says so.

**The corpus.** Resources are cited by number throughout.

| # | Resource | Authors | Group | Cluster |
| --- | --- | --- | --- | --- |
| 01 | Gaussian Processes for Machine Learning | Rasmussen & Williams | Core | Bayesian |
| 02 | Model-Based Clustering and Classification for Data Science | Bouveyron, Celeux, Murphy & Raftery | Core | Mixed |
| 03 | MDPs and Reinforcement Learning | Puterman & Chan | Core | Mixed |
| 04 | The Elements of Statistical Learning | Hastie, Tibshirani & Friedman | Core | Optimization |
| 05 | Advanced Data Analysis from an Elementary Point of View | Shalizi | Core | Mixed |
| 06 | Probabilistic Machine Learning: Advanced Topics | Murphy | Core | Bayesian |
| 07 | Machine Learning | Mitchell | Core | Mixed |
| 08 | Probabilistic Graphical Models | Koller & Friedman | Core | Bayesian |
| 09 | Bayesian Reasoning and Machine Learning | Barber | Core | Bayesian |
| 10 | Information Theory, Inference, and Learning Algorithms | MacKay | Core | Bayesian |
| 11 | The EM Algorithm | Byrne | Core | Optimization |
| 12 | Reinforcement Learning: An Introduction | Sutton & Barto | Adjacent | Optimization |
| 13 | Algorithms for Decision Making | Kochenderfer, Wheeler & Wray | Adjacent | Mixed |
| 14 | Bandit Algorithms | Lattimore & Szepesvári | Adjacent | Mixed |
| 15 | Introduction to Time Series Analysis and Forecasting | Montgomery, Jennings & Kulahci | Adjacent | Applied |
| 16 | Modern Time Series Forecasting with Python | Joseph & Tackes | Adjacent | Applied |
| 17 | Modeling and Simulation in Python | Downey | Adjacent | Applied |
| 18 | Discrete-Event System Simulation | Banks, Carson, Nelson & Nicol | Adjacent | Applied |
| 19 | Convex Optimization | Boyd & Vandenberghe | Adjacent | Optimization |

The cluster column records which technical language a text leads with (Pillar 6). "Mixed" texts use both.

**Method.** Structured reading of each text, cross-resource comparison, and causal analysis aimed at why the recurring patterns recur rather than only what they are (Epistemic Deconstructor protocol, standard tier, explanatory fidelity). The synthesis rests on 10 structured observations and four competing hypotheses about what unifies the corpus. Counts such as "appears in 15 of 19 resources" come from that reading, not from systematic coding of every chapter, and should be read as approximate.

**Confidence: moderate.** The framework explains this corpus well. The corpus is not the whole field, and a guide that teaches calibrated uncertainty should not claim more.

## The Root Cause: Epistemic Incompleteness

The defining condition of machine learning is the **gap between what needs to be known and what can be learned from the available evidence.** The gap takes five forms. The first three are the familiar ones; the last two are needed to explain parts of the corpus that the first three cannot.

| Form | What is missing | What it drives | Main resources |
| --- | --- | --- | --- |
| 1. Finite data | Observations are samples, not the population; every estimate varies by sample, and any pattern might be noise | The fit versus generalization tradeoff, regularization, Occam's razor | 01, 04, 05, 09, 10 |
| 2. Limited feedback | In sequential settings, consequences arrive after the action, and gathering information costs reward | Exploration versus exploitation, regret minimization, active learning | 03, 12, 13, 14 |
| 3. Limited computation | Exact inference is often intractable: marginals in general Bayesian networks are NP-hard to compute (Cooper 1990) and to approximate (Dagum and Luby 1993); RL state spaces grow exponentially | Variational methods, MCMC, function approximation, simulation | 06, 08, 09, 17, 18, 19 |
| 4. Limited observability | Some quantities cannot be identified from any amount of the available data: latent variables, hidden states, causal effects under confounding | Latent-variable models and EM, state-space models, POMDPs, identification assumptions, experiments | 02, 05, 08, 11, 13 |
| 5. Non-stationarity | The process that generated the training data differs from the one the model will face | Temporal validation, discounting and smoothing, monitoring, retraining | 12, 15, 16 |

**Why forms 4 and 5 are necessary.** More data resolves the first form in the limit. It resolves neither of the last two. A billion observational records do not identify a causal effect under unmeasured confounding, and a perfect fit to last year's process guarantees nothing about next year's. The maxim to distinguish prediction from intervention, and the warning that distribution shift is the default, both depend on forms that a three-part account leaves out.

Nearly every technique in the corpus can be read as a response to one or more of these forms. The reading is interpretive: it organizes the corpus well, but it is a lens on the field, not a law the field states.

## The Eight Pillars

The mindset rests on eight pillars, ordered from foundational stance to operational practice. Pillar 3, data as evidence, is taught throughout the corpus in pieces but rarely named as a principle; the synthesis is incomplete without it.

### Pillar 1: Epistemic Humility

> "All models are wrong, but some are useful." George Box

**The stance:** The mindset begins with the recognition that **data underdetermines truth.** Every model is a provisional approximation. Every prediction carries irreducible uncertainty. No finite dataset reveals the full data-generating process. The practitioner's first commitment is calibrated honesty about the limits of knowledge.

**What this means in practice:**

- Never confuse a model with reality. A model captures useful structure; it does not mirror the world.
- Quantify uncertainty wherever a decision depends on it. Use prediction intervals for future observations and confidence or credible intervals for parameters: they answer different questions.
- Treat your assumptions as the most consequential part of the model, more than the algorithm.
- Expect to be wrong. Build systems that detect and correct their own errors.

**Where it appears:** Bayesian posteriors (01, 06, 08, 09, 10), prediction intervals (15, 16), regret bounds and confidence sets (14), uncertainty sets in robust optimization (19), GP uncertainty bands (01), information-theoretic limits (10, 14), the bias-variance decomposition (04, 05).

**The key question:** *What do I not know, and how much does it matter?*

### Pillar 2: Principled Problem Formulation

> Boyd and Vandenberghe observe, flagging it as a mild exaggeration, that formulating a practical problem as a convex program amounts to solving it (Resource 19).

**The stance:** The hardest and most valuable step is **translating an ambiguous real-world problem into a precise mathematical structure.** The same problem can be cast as a probabilistic model (a likelihood and a prior), an optimization problem (a loss and constraints), a decision problem (a utility and an action space), a causal question (an intervention and its effect), or a simulation (state variables and update rules). The formulation determines what can be learned, what cannot, and what the answer will look like.

**What this means in practice:**

- Define Mitchell's triple (T, E, P), the task, the experience and the performance measure, before touching any algorithm.
- Ask whether the problem is inference (what is true?), optimization (what is best?), decision-making (what should I do?) or intervention (what happens if I act?). The last needs causal assumptions or experiments, not only data (root cause form 4).
- Recognize structure: is the loss convex, the model graphical, the process Markov? Can it factorize?
- Exploit convexity when you have it, and know what you lose without it. Deep networks, mixture models and EM objectives are non-convex: local optima, saddle points and initialization sensitivity come back.
- Know what your loss estimates. Squared (L2) loss weights large errors quadratically, so it is outlier-sensitive and targets the conditional mean. Absolute (L1) loss weights errors linearly, resists outliers and targets the conditional median. Asymmetric (pinball) losses target quantiles and encode unequal costs of over- and under-prediction.

**Where it appears:** the convex problem hierarchy LP ⊂ QP ⊂ SOCP ⊂ SDP (19), graphical model specification (08, 09), MDP formulation (03, 12, 13), kernel selection (01), ARIMA identification (15), feature engineering (16), causal formulation (05, 08).

**The key question:** *What mathematical structure does this problem have, and how do I exploit it?*

### Pillar 3: Data as Evidence

> **Thesis:** A model can be no better than the evidence it learns from, and the evidence is never the population.

**The stance:** Every guarantee in the corpus is conditional on how the data was generated: sampled at random, labelled correctly, independent of the test set, representative of deployment. These are the most frequently violated assumptions in practice, and their violation is invisible to the algorithm. The corpus teaches this in pieces: input modelling in simulation (18), data preparation and missing values in forecasting (15, 16), missing data and latent variables (08, 09, 11), selection and sampling effects (05).

**What this means in practice:**

- Know the provenance: who collected the data, how, when, and what was excluded. Sampling bias does not shrink with sample size.
- Look before modelling: distributions, missingness patterns, duplicates, label noise, outliers.
- Check that every feature would exist at prediction time. A feature computed with future or target information is leakage, and leakage is a leading cause of irreproducible ML results across scientific fields (Kapoor and Narayanan 2023).
- Document the dataset as an artifact in its own right (Gebru et al. 2021).
- Budget for it. Data problems compound silently downstream (Sambasivan et al. 2021), and the model code is a small part of a real ML system (Sculley et al. 2015).

**Where it appears:** input modelling and goodness-of-fit for input distributions (18), data preparation, missing values and outliers (15, 16), missing data and EM (08, 09, 11), sampling and selection (05).

**The key question:** *What process produced this data, and how does it differ from the one my model will face?*

### Pillar 4: Decomposition

> **Thesis:** To learn is to separate structure from noise, signal from residual, and cause from correlation.

**The stance:** Most ML methods decompose complex problems into simpler parts, and decomposition is the main way they impose structure. Three forces favour it:

- **Approximately compositional structure.** Locality, conditional independence, the Markov property and superposition are often approximately true of the world. They are assumptions, not facts: their approximate truth is why decomposition works, and their failure is where it breaks.
- **Computational necessity.** Monolithic computation is usually intractable. Decomposition makes it tractable when the parts interact weakly; exact inference in a graphical model is exponential in the graph's treewidth, so factorization pays only when the graph is sparse enough.
- **Cognitive requirement.** People cannot design, debug or trust systems they cannot break into interpretable parts.

Not every method decomposes into explicit parts. Kernel methods and nearest-neighbour methods carry their structural assumptions in a similarity function instead. The assumption is still there; it has moved into the kernel.

**What this means in practice:**

- Look for conditional independence structure: graphical models factor joint distributions.
- Decompose time series into trend, seasonal and residual components.
- Factor reward into immediate and future value (the Bellman equation).
- Separate signal from noise (regularization, PCA, sparse coding).
- Break complex systems into compartments with defined flows (SIR models, queuing networks).
- Treat granularity as a control knob: coarser decompositions generalize more and fit less; finer ones the reverse.

**Where it appears:** graph factorization (08, 09), the bias-variance decomposition (04, 05), trend-seasonal decomposition (15, 16), Bellman recursion (03, 12, 13), the E-step and M-step (11), primal-dual splitting (19), compartment and queuing models (17, 18), additive models (04, 05), basis expansion (04, 16).

**The key question:** *What are the parts, and how do they compose?*

**Connection to the tradeoffs:** Decomposition is the mechanism of the fit versus generalization tradeoff. Every decomposition imposes structural assumptions (independence, additivity, stationarity) that trade fidelity for generalizability. Formulation (Pillar 2), representation (Maxim 9) and decomposition are three views of one decision: what structure the model is allowed to see.

### Pillar 5: The Fundamental Tradeoffs

> **Thesis:** Each form of incompleteness forces its own tradeoff. Much of the craft is choosing where to stand on each.

**The stance:** The best-known tension is between **fitting the data** and **performing on new data.** It is the central challenge of supervised learning, but it is one of several. Each form of incompleteness produces a tradeoff of its own, and conflating them hides what each one needs.

| Source of incompleteness | Tradeoff | Typical controls | Main resources |
| --- | --- | --- | --- |
| Finite data | Fit versus generalization | Model complexity, regularization strength, priors, early stopping | 01, 04, 05, 09, 10 |
| Limited feedback | Exploration versus exploitation | Exploration bonuses, posterior sampling, horizon, discount | 12, 13, 14 |
| Limited computation | Accuracy versus cost | Exact or approximate inference, sample counts, simulation replications, iteration budgets | 06, 08, 18, 19 |
| Non-stationarity | Stability versus adaptivity | Window length, smoothing constants, step size, retraining cadence | 12, 15, 16 |
| Limited observability | None: a wall, not a tradeoff | Identification assumptions, experiments, instruments | 05, 08 |

Exploration versus exploitation is not a variant of bias and variance. An agent with a perfectly calibrated model may still need to explore, because its problem is missing feedback, not too few samples. The two share a shape, a cost now against a benefit later, and little else.

**The fit versus generalization family.** Within the first row, different traditions describe the same tension:

| Tradition | Name | Fit side | Generalization side |
| --- | --- | --- | --- |
| Statistical learning | Bias-variance tradeoff | Low bias (flexible model) | Low variance (constrained model) |
| Regularization | Fit versus penalty | Empirical loss | Penalty term |
| Bayesian | Occam's razor | Data-fit term of the marginal likelihood | Occam factor, the prior-volume penalty inside the same quantity |
| Information theory | Minimum description length | Bits to encode the data given the model | Bits to encode the model |
| Convex optimization | Regularized approximation | Residual norm | Solution norm, traced as a Pareto curve (19) |
| Time series | In-sample versus out-of-sample | In-sample fit | Out-of-sample forecast error |

**What this means in practice:**

- **Regularize deliberately, and know what the regularizer assumes.** L2 corresponds to a Gaussian prior and shrinks smoothly; L1 corresponds to a Laplace prior and produces sparsity; early stopping regularizes implicitly, and for linear least squares it behaves like L2.
- **Estimate generalization out of sample.** For IID data, holdout and cross-validation are the most reliable estimates. Training loss is not an estimate of generalization.
- **Use information criteria for their stated purpose.** AIC targets predictive accuracy. BIC targets recovery of the true model when it is among the candidates. The marginal likelihood can prefer models that generalize worse (Lotfi et al. 2022).
- **Start with a simple baseline.** It is cheap, interpretable, and the reference any complex model must beat. Whether the final model should also be simple is an empirical question.
- **Use ensembles knowingly.** Bagging averages away variance; boosting mainly reduces bias. Both usually improve accuracy, at a cost in compute, latency and interpretability.

**The classical picture is incomplete.** The U-shaped risk curve, with test error rising once capacity passes a sweet spot, does not describe heavily over-parameterized models. Networks can fit random labels yet generalize on real ones (Zhang et al. 2017). Test error can fall again past the interpolation threshold (Belkin et al. 2019; Nakkiran et al. 2020). Interpolating noisy data can be benign (Bartlett et al. 2020). The tradeoff survives, but parameter count is the wrong measure of complexity, and the optimizer itself regularizes. The corpus does not cover this regime.

**Where it appears:** the fit versus generalization tradeoff appears in nearly every resource under one name or another; the other tradeoffs appear in the resources listed in the first table.

**The key question:** *Which tradeoff am I on, and where on it should I stand?*

### Pillar 6: The Two Technical Languages

> **Thesis:** Maximum a posteriori estimation is regularized maximum likelihood.

**The stance:** Probability and optimization are not competing philosophies. They are **dual technical languages** connected by precise identities, and the practitioner must be bilingual.

| Probabilistic language | Optimization language | Identity |
| --- | --- | --- |
| Posterior mode (MAP) | Regularized maximum likelihood | argmax p(θ\|D) = argmin \[−log p(D\|θ) − log p(θ)\] |
| Gaussian prior | L2 regularization | −log N(w; 0, σ²I) = ‖w‖² / (2σ²) + const |
| Laplace prior | L1 regularization | −log Laplace(w; 0, b) = ‖w‖₁ / b + const |
| Empirical Bayes (type-II maximum likelihood) | Hyperparameter optimization | Maximize p(D\|η) = ∫ p(D\|θ) p(θ\|η) dθ over hyperparameters η |
| Minimizing KL(p‖q) over q | Cross-entropy loss | H(p, q) = H(p) + KL(p‖q) |
| Variational inference | ELBO maximization | log p(x) ≥ ELBO, with equality when q is the exact posterior |
| Expected utility | Expected loss minimization | max E\[U\] = min E\[L\], with L = −U |
| Posterior sampling | Noisy gradient descent | Gradient steps plus Gaussian noise sample the posterior (Langevin dynamics; SGLD, Welling and Teh 2011) |

The identities hold as stated and no further. MAP agrees with regularized maximum likelihood at the mode only: it discards the posterior's spread, and unlike the posterior it changes under reparameterization. Translating into the optimization language is convenient precisely because it throws uncertainty away.

**When to use which:**

- **Choose probability** to propagate uncertainty, combine evidence, handle missing data, or report calibrated predictions.
- **Choose optimization** for scalability, convergence guarantees, or when the problem has clean convex structure.
- **Choose both** when the problem demands it, as most modern ML does.

Questions of intervention need a third language. Structural causal models and the do-operator (05, 08; Pearl 2009) express what changes when you act, which neither probability nor optimization expresses alone.

**Where it appears:** most resources use both languages. The Bayesian cluster (01, 06, 08, 09, 10) leads with probability; the optimization cluster (04, 11, 12, 19) leads with optimization; the applied cluster (15, 16, 17, 18) uses whichever fits.

**The key question:** *Am I reasoning about beliefs, searching for a solution, or asking what happens if I act?*

### Pillar 7: Rigorous Validation

> **Thesis:** Training performance is not evidence of generalization.

**The stance:** Validation is not a post-hoc check. It is the **constitutive practice** that distinguishes a model from a guess. The mindset treats every model as guilty until proven innocent and demands evidence of generalization before conferring trust.

**No method is strongest in general.** Each guards against a different failure, and each assumes something the others check.

| Method | What it establishes | What it assumes | Fails when |
| --- | --- | --- | --- |
| Mathematical certification: convergence proofs, duality gaps, regret bounds (03, 11, 14, 19) | Properties of the algorithm | The model and its assumptions | The model is wrong about the world |
| Holdout and cross-validation (04, 05, 07, 15, 16) | A finite-sample estimate of future error, with confidence bounds | Test data exchangeable with deployment data, no leakage, no test-set reuse | Shift, leakage, or tuning against the test set |
| Conformal prediction (Angelopoulos and Bates 2023) | Distribution-free coverage of prediction sets | Exchangeability | Shift |
| Generalization bounds: PAC, VC dimension (04, 07, 14) | Worst-case error guarantees | IID data and a fixed hypothesis class | Often too loose to inform decisions about large models |
| Bayesian model comparison: marginal likelihood, Bayes factors, posterior predictive checks (01, 02, 06, 08, 09, 10) | Coherent comparison within a model family | Sensible priors and likelihoods | Priors are arbitrary or every candidate is misspecified |
| Diagnostic and residual checks: specification tests, residual analysis, Ljung-Box (04, 05, 15, 18) | Evidence of a specific misfit | A test aimed at the right failure | The failure is one no test targets |
| Engineering verification and validation: face validity, Turing tests, adversarial testing (13, 17, 18) | Plausibility and robustness to probing | Expert judgement | Experts share the model's blind spots |

**What this means in practice:**

- **Never estimate generalization from training performance.** Training metrics remain useful diagnostics, for learning curves and for spotting underfitting.
- **Split the way the data will arrive.** Temporal splits for time series, group splits for grouped data, spatial splits for spatial data.
- **Hunt for leakage.** Preprocessing fitted on all the data, duplicates across splits, and features derived from the target or the future all inflate scores.
- **Beat a naive baseline.** The last value or seasonal naive forecast, the majority class, the historical mean. A model that cannot beat it has learned nothing useful.
- **Do not tune against the test set.** Each look leaks information, and repeated reuse turns a test set into a training set (Dwork et al. 2015; Recht et al. 2019).
- **Report evaluation uncertainty.** A test score from a finite sample has an interval too.
- **Verify code before validating the model.** Implementation bugs often masquerade as modelling results.
- **Monitor deployed models continuously.** Distribution shift is the default, not the exception.
- **Use several methods.** No single method catches every failure mode.

**Four gaps that validation bridges:**

1. **Evidence gap.** Is the data a faithful sample of the world the model will face? (Pillar 3)
2. **Formulation gap.** Is the model a reasonable approximation of reality?
3. **Computation gap.** Does the algorithm correctly implement the model?
4. **Deployment gap.** Will the model work on future, unseen data?

**The key question:** *How do I know this isn't just noise, leakage or luck?*

### Pillar 8: Iterative Refinement

> **Thesis:** Start simple, validate, diagnose, refine. Repeat.

**The stance:** Getting it right the first time is not expected. The workflow is iterative: formulate, examine the data, decompose, fit, validate, diagnose, reformulate. Each cycle corrects false assumptions and narrows the gap between what the model predicts and what the world does.

**Three iteration loops:**

1. **Inner loop (complexity adjustment).** Same formulation; adjust regularization and hyperparameters (cross-validation, grid search, Bayesian optimization).
2. **Middle loop (reformulation).** Change the model family, features or decomposition (linear to tree-based, add nonlinearity, change the kernel).
3. **Outer loop (re-scoping).** Change the problem definition: redefine the target, change the loss, acquire new data sources.

**What this means in practice:**

- **Diagnose the error type before prescribing a fix.** High bias: add capacity or features. High variance: regularize or get more data. Misspecification: reformulate. Shift: retrain on recent data or adapt.
- **Use the simplest diagnostic available.** Residual plots, learning curves and calibration plots reveal more than most practitioners extract from them.
- **Log your iterations, and make them reproducible.** Record what you tried, what happened and why you changed direction; fix random seeds and version data and code.

**Where it appears:** Box-Jenkins iterative identification (15), EM iteration (11), policy and value iteration (03, 12), boosting (04, 16), the forecasting process (15), incremental model development (17).

**The key question:** *What is the model getting wrong, and what is the cheapest fix?*

## The Practitioner's Workflow

The eight pillars combine into one working loop.

1. **Scope the unknown** (Pillar 1). What don't I know? Which forms of incompleteness apply here?
2. **Formulate** (Pillar 2). Define task, experience and metric. Decide whether this is inference, optimization, decision or intervention.
3. **Examine the data** (Pillar 3). Provenance, quality, leakage risks, and how it differs from deployment. Set the test set aside now, before searching for patterns.
4. **Establish a baseline** (Pillar 7). The naive model every later model must beat.
5. **Decompose** (Pillar 4). Factor the problem, identify structure, choose granularity.
6. **Choose the language and fit** (Pillars 5 and 6). Select complexity and regularization; pick probabilistic or optimization tools.
7. **Validate** (Pillar 7). Out of sample, split the way the data will arrive, with uncertainty on the scores.
8. **Diagnose and refine** (Pillar 8). Bias, variance, misspecification, leakage or shift? Fix and repeat.
9. **Deploy and monitor.** Watch for drift, and re-enter the loop when it appears.

**The workflow is a loop, not a pipeline.** Most projects cycle through steps 2 to 8 several times. Failures that trace to the data send you back to step 3. Failures that trace to the question send you back to step 2, the outer loop. Failures of complexity send you back to step 6, the inner loop.

## 32 Practical Maxims

### Formulation

**1. Define task, experience and metric before touching algorithms.** Without Mitchell's (T, E, P), you are optimizing nothing.

**2. Exploit convexity when you have it; know what you lose without it.** Convex problems have no spurious local optima, and the standard classes (LP, QP, SOCP, SDP) solve efficiently. Most modern models are not convex.

**3. For probabilistic problems, sketch the graphical structure before choosing an algorithm.** Conditional independence determines what is tractable.

**4. Distinguish prediction from intervention.** Correlation can suffice for prediction; intervention needs causal assumptions or experiments.

### Data

**5. Know where the data came from.** Sampling bias does not shrink with sample size.

**6. Look at the data before modelling it.** Distributions, missingness, duplicates and label noise often matter more than the choice of algorithm.

**7. Every feature must exist at prediction time.** Anything computed from the target or the future is leakage.

### Modeling

**8. Start with the simplest model that could work, as a baseline.** It is cheap, interpretable, and the bar every complex model must clear.

**9. The choice of representation is a primary modeling decision.** Kernels, features and architectures determine what can be learned.

**10. Every learner has an inductive bias; make yours explicit.** There is no assumption-free learning: averaged over all possible problems, no learner beats any other (Wolpert 1996).

**11. Encode known structure where it costs least.** Constraints, priors, architecture or features: prefer forms the model cannot violate when the knowledge is certain, and soft forms when it is not. With enough data and compute, general methods often overtake hand-built structure (Sutton 2019).

**12. Use the marginal likelihood to compare models, with care.** It builds in an automatic Occam's razor, but it is sensitive to the prior and can prefer models that generalize worse (Lotfi et al. 2022).

### Tradeoff Navigation

**13. Choose your operating point consciously.** If you do not choose it, you choose it by accident.

**14. Regularize deliberately, and know what your regularizer assumes.** L2 shrinks smoothly; L1 induces sparsity; dropout approximates an ensemble of subnetworks (Srivastava et al. 2014); early stopping regularizes implicitly.

**15. Explore as much as horizon and feedback require.** Pure exploitation is rarely optimal under limited feedback, but with short horizons or sufficiently diverse contexts, greedy policies can be near-optimal (Bastani et al. 2021).

**16. Ensembles usually improve accuracy, at a cost.** Bagging reduces variance and boosting reduces bias; both cost compute, latency and interpretability.

**17. Approximation is standard operating procedure.** Variational inference, MCMC and function approximation are necessities, not compromises.

### Validation

**18. Never estimate generalization from training performance.** Use training metrics for diagnosis only.

**19. Split the data the way it will arrive.** Temporal data needs temporal splits; grouped data needs group splits.

**20. Beat a naive baseline before claiming anything.** Seasonal naive forecast, majority class, historical mean.

**21. Do not tune on the test set.** Every look leaks information, and reused test sets overfit.

**22. Verify code before validating the model.** Bugs often masquerade as modelling results.

**23. Monitor deployed models continuously.** Distribution shift is the default, not the exception.

### Iteration

**24. Diagnose the error type before prescribing a fix.** High bias: add capacity. High variance: regularize or add data. Wrong problem: reformulate. Shift: retrain or adapt.

**25. Iterative algorithms find stationary points, not global optima; restart wisely.** EM increases the likelihood monotonically but can stop at a local optimum or saddle point (Wu 1983). Multiple restarts, annealing or warm starts help.

**26. Use simulation to understand a model before trusting it.** Synthetic data with known ground truth reveals failure modes.

**27. Use mini-batches for scale, with convergence caveats.** Stochastic gradients make large data tractable but add noise; learning-rate schedules compensate.

**28. Make every result reproducible.** Fix seeds, version data and code, and log why each change was made.

### Epistemological

**29. Quantify uncertainty wherever a decision depends on it.** Posteriors, prediction intervals or conformal sets.

**30. Read learning as inference when uncertainty must propagate.** The Bayesian view unifies supervised, unsupervised and sequential settings; the optimization view is often the cheaper way to compute.

**31. Compression and prediction are two sides of one coin.** A model that predicts well encodes the data in fewer bits, counting the bits needed to describe the model itself.

**32. Model uncertainty about the model itself.** Bayesian model averaging, ensemble disagreement and conformal prediction address uncertainty about the model, not only about its parameters.

## The ML Epistemology: Calibrated Pragmatic Empiricism

The corpus embodies a distinctive theory of knowledge, built on six commitments.

1. **Knowledge is predictive, not absolute.** For prediction, a model is good if it predicts well on unseen data, not if it is true. For intervention, predictive accuracy on observational data is not enough: the causal structure must also be right.
2. **Knowledge is uncertain.** Predictions should carry calibrated confidence wherever decisions depend on them.
3. **Knowledge is assumption-dependent.** There is no assumption-free learning. The quality of the assumptions bounds the quality of the learning.
4. **Knowledge is approximate.** All models are wrong. The question is how wrong, and whether it matters for the decision at hand.
5. **Knowledge is empirically validated.** The most honest measure of a model is its performance on data it has not seen, drawn the way future data will be drawn.
6. **Parsimony is a working prior, not a guarantee.** Simpler models are cheaper, easier to debug and safer when data is scarce. They are not more likely to be correct in general (Domingos 1999), and very large models can generalize well.

**What ML adds to neighbouring disciplines:**

| Discipline | Shared with ML | What ML adds |
| --- | --- | --- |
| Classical statistics | Estimation, testing, confidence | Prediction focus, algorithmic flexibility, scale; the "algorithmic modeling culture" (Breiman 2001) |
| Physics | Mathematical modeling, conservation laws | Learning from data without known equations; model selection from data |
| Pure mathematics | Rigor, proof, abstraction | Empirical validation, finite-sample behaviour, computational feasibility |
| Software engineering | Iterative development, testing, deployment | Statistical validation, uncertainty quantification, the tradeoffs |

In short, ML combines the empiricist's respect for data, the mathematician's demand for rigor and the engineer's insistence on working systems, all under the constraint of epistemic incompleteness.

## Blind Spots: What the Corpus Does Not Address

No framework is complete. Seven gaps stand out. Some come from what the corpus omits; others are present in individual resources but never integrated across them.

1. **Deep learning and the over-parameterized regime.** The corpus has no dedicated deep learning text, and representation learning appears at length only in Resource 06. Its account of capacity control is classical; Pillar 5 notes where that account breaks.
2. **Pretrain-then-adapt practice.** Modern systems learn most of what they know from data the practitioner never sees, then adapt to a small task-specific set. Mitchell's experience E is mostly someone else's, and holdout validation is threatened when test items may have appeared in the pretraining data.
3. **Social epistemology and reproducibility.** ML is treated as a solo practitioner's activity. Teams, institutions, peer review, benchmark culture, and the reproducibility failures they produce (Kapoor and Narayanan 2023) are absent.
4. **Values beyond utility.** Decision theory (13) represents values as a utility to maximize. Fairness, accountability and safety constraints that do not reduce to a single utility are not covered.
5. **Explaining black-box predictions.** Causal explanation is covered (05, 08). Explaining why a flexible model made a particular prediction, and when that explanation can be trusted, is not.
6. **Fundamental limits.** Impossibility results (no free lunch, regret lower bounds, information-theoretic limits, computational hardness) are scattered across resources rather than unified into a theory of what cannot be learned.
7. **Shift and decay as one theory.** Non-stationarity is handled within families: forecasting adapts to it, RL uses constant step sizes and discounting, deployed models are monitored for it. No unified theory says how long learned knowledge stays valid, or when to retrain.

Dependent data as such is not a blind spot. Roughly eight of the 19 resources (03, 12, 13, 14, 15, 16, 17, 18) are about sequential, temporal or simulated non-IID settings. What is missing is the cross-cutting theory in item 7.

## The Causal Architecture

Five forms of incompleteness create five pressures; the pillars are the practices that answer them.

```
EPISTEMIC INCOMPLETENESS
│
├── Finite data ─────────────── fit vs generalization
├── Limited feedback ────────── explore vs exploit
├── Limited computation ─────── accuracy vs cost
├── Non-stationarity ────────── stability vs adaptivity
└── Limited observability ───── no tradeoff: assumptions or experiments
                 │
          DATA AS EVIDENCE
   (how the incompleteness shows up in
    this dataset: provenance, bias, leakage)
                 │
           DECOMPOSITION
   (impose structure; granularity sets
    the operating point on each tradeoff)
                 │
       ┌─────────┴─────────┐
   PROBABILITY       OPTIMIZATION      + CAUSAL
   (belief and       (search and        (for questions
    uncertainty)      solutions)         of intervention)
       └─────────┬─────────┘
                 │
             VALIDATION
   (guards against underdetermination,
    leakage and shift)
                 │
        ITERATIVE REFINEMENT
   (formulate → examine data → fit →
    validate → diagnose → reformulate)
                 │
      DEPLOYMENT AND MONITORING
   (detect drift → re-enter the loop)
```

Read it top to bottom: each layer answers the pressures above it, and monitoring closes the loop back to the data. Epistemic humility (Pillar 1) and problem formulation (Pillar 2) are not layers; they frame the whole diagram.

## How to Use This Guide

1. **Before starting a project:** read the eight pillars and ask their key questions. Documented failures concentrate in data and validation: leakage (Kapoor and Narayanan 2023) and compounding data problems (Sambasivan et al. 2021). Give Pillars 3 and 7 extra attention.
2. **When stuck:** identify which pillar you are struggling with, and which tradeoff. Complexity? Adjust it. Validation? Change the evaluation strategy. Formulation? Redefine the problem. Data? Return to provenance.
3. **When choosing methods:** use the two-languages table (Pillar 6) to translate between probabilistic and optimization framings. Choose by whether you need uncertainty propagation or computational scale, and switch to the causal language when the question is about intervention.
4. **When reviewing others' work:** use the pillars as a checklist. Are assumptions stated? Is the data's provenance known and leakage ruled out? Does the model beat a naive baseline? Is the tradeoff managed? Is validation split the way data arrives, with uncertainty reported? Is the formulation appropriate?
5. **When teaching:** the pillars give a curriculum that outlasts individual algorithms. Teach the mindset first and the techniques second, and pair the corpus with a modern deep learning text to cover the over-parameterized regime.

## Sources Beyond the Corpus

Corpus resources are listed in the first section. The works below support claims the corpus does not make; they were checked from memory and should be confirmed before external publication.

- Angelopoulos, A. N. and Bates, S. (2023). Conformal Prediction: A Gentle Introduction. Foundations and Trends in Machine Learning.
- Bartlett, P. L., Long, P. M., Lugosi, G. and Tsigler, A. (2020). Benign Overfitting in Linear Regression. PNAS.
- Bastani, H., Bayati, M. and Khosravi, K. (2021). Mostly Exploration-Free Algorithms for Contextual Bandits. Management Science.
- Belkin, M., Hsu, D., Ma, S. and Mandal, S. (2019). Reconciling Modern Machine-Learning Practice and the Classical Bias-Variance Trade-off. PNAS.
- Box, G. E. P. and Draper, N. R. (1987). Empirical Model-Building and Response Surfaces. Source of the aphorism in Pillar 1.
- Breiman, L. (2001). Statistical Modeling: The Two Cultures. Statistical Science.
- Cooper, G. F. (1990). The Computational Complexity of Probabilistic Inference Using Bayesian Belief Networks. Artificial Intelligence.
- Dagum, P. and Luby, M. (1993). Approximating Probabilistic Inference in Bayesian Belief Networks Is NP-hard. Artificial Intelligence.
- Domingos, P. (1999). The Role of Occam's Razor in Knowledge Discovery. Data Mining and Knowledge Discovery.
- Dwork, C. et al. (2015). The Reusable Holdout: Preserving Validity in Adaptive Data Analysis. Science.
- Gebru, T. et al. (2021). Datasheets for Datasets. Communications of the ACM.
- Kapoor, S. and Narayanan, A. (2023). Leakage and the Reproducibility Crisis in Machine-Learning-Based Science. Patterns.
- Lotfi, S., Izmailov, P., Benton, G., Goldblum, M. and Wilson, A. G. (2022). Bayesian Model Selection, the Marginal Likelihood, and Generalization. ICML.
- Nakkiran, P. et al. (2020). Deep Double Descent: Where Bigger Models and More Data Hurt. ICLR.
- Pearl, J. (2009). Causality: Models, Reasoning, and Inference. 2nd ed.
- Recht, B., Roelofs, R., Schmidt, L. and Shankar, V. (2019). Do ImageNet Classifiers Generalize to ImageNet? ICML.
- Sambasivan, N. et al. (2021). "Everyone Wants to Do the Model Work, Not the Data Work": Data Cascades in High-Stakes AI. CHI.
- Sculley, D. et al. (2015). Hidden Technical Debt in Machine Learning Systems. NeurIPS.
- Srivastava, N. et al. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. JMLR.
- Sutton, R. S. (2019). The Bitter Lesson.
- Welling, M. and Teh, Y. W. (2011). Bayesian Learning via Stochastic Gradient Langevin Dynamics. ICML.
- Wolpert, D. H. (1996). The Lack of A Priori Distinctions Between Learning Algorithms. Neural Computation.
- Wu, C. F. J. (1983). On the Convergence Properties of the EM Algorithm. Annals of Statistics.
- Zhang, C., Bengio, S., Hardt, M., Recht, B. and Vinyals, O. (2017). Understanding Deep Learning Requires Rethinking Generalization. ICLR.
