# Statistical Methods Reference (Expanded)

A practitioner-oriented catalogue of statistical, causal, simulation, dynamical-systems and signal-processing methods. Each entry covers what the method does, the conditions under which it is the right tool, its main assumptions and failure modes, and the domains where it is standard practice.

---

## Table of Contents

**[1. Bayesian Inference and Model Comparison](#1-bayesian-inference-and-model-comparison)**
- [1.1 Bayesian Inference](#11-bayesian-inference)
- [1.2 Bayes Factors](#12-bayes-factors)
- [1.3 Bayesian Model Averaging (BMA)](#13-bayesian-model-averaging-bma)

**[2. Classical (Frequentist) Inference](#2-classical-frequentist-inference)**
- [2.1 Classical Hypothesis Testing](#21-classical-hypothesis-testing)
- [2.2 Confidence Intervals](#22-confidence-intervals)
- [2.3 Analysis of Variance (ANOVA)](#23-analysis-of-variance-anova)
- [2.4 Likelihood Ratio Test](#24-likelihood-ratio-test)

**[3. Regression Methods](#3-regression-methods)**
- [3.1 Ordinary Least Squares (OLS) Regression](#31-ordinary-least-squares-ols-regression)
- [3.2 Generalized Linear Models (GLM)](#32-generalized-linear-models-glm)
- [3.3 Ridge Regression](#33-ridge-regression)
- [3.4 Lasso Regression](#34-lasso-regression)
- [3.5 Quantile Regression](#35-quantile-regression)

**[4. Causal Inference](#4-causal-inference)**
- [4.1 Potential Outcomes Framework](#41-potential-outcomes-framework)
- [4.2 Difference-in-Differences (DiD)](#42-difference-in-differences-did)
- [4.3 Instrumental Variables (IV)](#43-instrumental-variables-iv)
- [4.4 Regression Discontinuity (RD)](#44-regression-discontinuity-rd)
- [4.5 Propensity Score Matching](#45-propensity-score-matching)
- [4.6 Do-Calculus (Pearl)](#46-do-calculus-pearl)

**[5. Information Theory and Model Selection](#5-information-theory-and-model-selection)**
- [5.1 KL Divergence](#51-kl-divergence)
- [5.2 Mutual Information](#52-mutual-information)
- [5.3 AIC / BIC](#53-aic--bic)

**[6. Extreme Value Theory](#6-extreme-value-theory)**
- [6.1 Extreme Value Theory (EVT)](#61-extreme-value-theory-evt)

**[7. Resampling and Validation](#7-resampling-and-validation)**
- [7.1 K-Fold Cross-Validation](#71-k-fold-cross-validation)
- [7.2 Leave-One-Out Cross-Validation (LOO-CV)](#72-leave-one-out-cross-validation-loo-cv)
- [7.3 Permutation Tests](#73-permutation-tests)

**[8. Structural Breaks](#8-structural-breaks)**
- [8.1 Change Point Detection](#81-change-point-detection)

**[9. Dependence Modelling](#9-dependence-modelling)**
- [9.1 Copulas](#91-copulas)

**[10. Spatial Statistics](#10-spatial-statistics)**
- [10.1 Kriging (Gaussian Process Regression)](#101-kriging-gaussian-process-regression)
- [10.2 Variogram Analysis](#102-variogram-analysis)
- [10.3 Moran's I (Spatial Autocorrelation)](#103-morans-i-spatial-autocorrelation)

**[11. Sensitivity Analysis and Uncertainty Quantification](#11-sensitivity-analysis-and-uncertainty-quantification)**
- [11.1 Uncertainty Quantification (UQ)](#111-uncertainty-quantification-uq)
- [11.2 Sensitivity Analysis](#112-sensitivity-analysis)
- [11.3 Variance-Based Decomposition (Sobol / ANOVA-HDMR)](#113-variance-based-decomposition-sobol--anova-hdmr)
- [11.4 Scenario Analysis](#114-scenario-analysis)
- [11.5 Tornado Diagram Analysis](#115-tornado-diagram-analysis)
- [11.6 Perturbation Analysis](#116-perturbation-analysis)
- [11.7 Design of Experiments (DoE)](#117-design-of-experiments-doe)
- [11.8 Surrogate / Emulator Modelling](#118-surrogate--emulator-modelling)
- [11.9 Elasticity Analysis](#119-elasticity-analysis)

**[12. Simulation Methods](#12-simulation-methods)**
- [12.1 Monte Carlo Simulation](#121-monte-carlo-simulation)
- [12.2 Quasi-Monte Carlo (QMC)](#122-quasi-monte-carlo-qmc)
- [12.3 Latin Hypercube Sampling (LHS)](#123-latin-hypercube-sampling-lhs)
- [12.4 Markov Chain Monte Carlo (MCMC)](#124-markov-chain-monte-carlo-mcmc)
- [12.5 Bootstrap](#125-bootstrap)
- [12.6 Discrete Event Simulation (DES)](#126-discrete-event-simulation-des)

**[13. Markov and State Space Models](#13-markov-and-state-space-models)**
- [13.1 Markov Chain](#131-markov-chain)
- [13.2 Hidden Markov Model (HMM)](#132-hidden-markov-model-hmm)
- [13.3 Higher-Order Markov Models](#133-higher-order-markov-models)
- [13.4 Continuous-Time Markov Chain (CTMC)](#134-continuous-time-markov-chain-ctmc)
- [13.5 Semi-Markov Process](#135-semi-markov-process)
- [13.6 Markov Decision Process (MDP)](#136-markov-decision-process-mdp)
- [13.7 Partially Observable MDP (POMDP)](#137-partially-observable-mdp-pomdp)
- [13.8 Input-Output HMM](#138-input-output-hmm)
- [13.9 Hierarchical HMM (HHMM)](#139-hierarchical-hmm-hhmm)
- [13.10 Factorial HMM](#1310-factorial-hmm)
- [13.11 Viterbi Algorithm](#1311-viterbi-algorithm)
- [13.12 Baum-Welch Algorithm](#1312-baum-welch-algorithm)
- [13.13 Kalman Filter](#1313-kalman-filter)
- [13.14 Particle Filter (Sequential Monte Carlo)](#1314-particle-filter-sequential-monte-carlo)

**[14. Forecastability and Forecast Evaluation](#14-forecastability-and-forecast-evaluation)**
- [14.1 Sample Entropy / Approximate Entropy](#141-sample-entropy--approximate-entropy)
- [14.2 Hurst Exponent](#142-hurst-exponent)
- [14.3 Spectral Analysis (Power Spectral Density)](#143-spectral-analysis-power-spectral-density)
- [14.4 Forecastability Index (Goerg, 2013)](#144-forecastability-index-goerg-2013)
- [14.5 Prediction Intervals (Classical)](#145-prediction-intervals-classical)
- [14.6 Forecast Reconciliation](#146-forecast-reconciliation)
- [14.7 Empirical Forecast Error Decomposition](#147-empirical-forecast-error-decomposition)

**[15. Initial Conditions, Chaos and Predictability Limits](#15-initial-conditions-chaos-and-predictability-limits)**
- [15.1 Initial Condition Sensitivity (Lyapunov Exponents)](#151-initial-condition-sensitivity-lyapunov-exponents)
- [15.2 Ensemble Methods](#152-ensemble-methods)
- [15.3 State Space Reconstruction (Takens' Theorem)](#153-state-space-reconstruction-takens-theorem)
- [15.4 Data Assimilation](#154-data-assimilation)
- [15.5 Shadowing](#155-shadowing)
- [15.6 Bifurcation Analysis](#156-bifurcation-analysis)

**[16. Stability Theory and Nonlinear Control](#16-stability-theory-and-nonlinear-control)**
- [16.1 Lyapunov Stability Theory](#161-lyapunov-stability-theory)
- [16.2 LaSalle's Invariance Principle](#162-lasalles-invariance-principle)
- [16.3 Barrier Functions / Control Barrier Functions (CBF)](#163-barrier-functions--control-barrier-functions-cbf)
- [16.4 Sum-of-Squares (SOS) Programming](#164-sum-of-squares-sos-programming)
- [16.5 Lyapunov Spectrum](#165-lyapunov-spectrum)
- [16.6 Kaplan-Yorke Dimension](#166-kaplan-yorke-dimension)
- [16.7 Finite-Time Lyapunov Exponents (FTLE)](#167-finite-time-lyapunov-exponents-ftle)
- [16.8 Lyapunov-Based Control Design](#168-lyapunov-based-control-design)
- [16.9 Contraction Analysis](#169-contraction-analysis)
- [16.10 Input-to-State Stability (ISS)](#1610-input-to-state-stability-iss)

**[17. Spectral and Time-Frequency Analysis](#17-spectral-and-time-frequency-analysis)**
- [17.1 Discrete Fourier Transform (DFT)](#171-discrete-fourier-transform-dft)
- [17.2 Fast Fourier Transform (FFT)](#172-fast-fourier-transform-fft)
- [17.3 Inverse FFT (IFFT)](#173-inverse-fft-ifft)
- [17.4 Short-Time Fourier Transform (STFT)](#174-short-time-fourier-transform-stft)
- [17.5 Wavelet Transform](#175-wavelet-transform)
- [17.6 Hilbert Transform](#176-hilbert-transform)
- [17.7 Welch's Method](#177-welchs-method)
- [17.8 Multitaper Spectral Estimation](#178-multitaper-spectral-estimation)

**[18. Distribution-Free Uncertainty](#18-distribution-free-uncertainty)**
- [18.1 Conformal Prediction](#181-conformal-prediction)

**[19. Dimensionality Reduction and Matrix Factorization](#19-dimensionality-reduction-and-matrix-factorization)**
- [19.1 Principal Component Analysis (PCA)](#191-principal-component-analysis-pca)
- [19.2 Singular Value Decomposition (SVD)](#192-singular-value-decomposition-svd)

**[20. Method Selection Quick Reference](#20-method-selection-quick-reference)**

---

## 1. Bayesian Inference and Model Comparison

### 1.1 Bayesian Inference

**What it does.** Treats unknown quantities as random variables with distributions rather than fixed points to be estimated. A prior distribution encodes what is believed before seeing data, the likelihood encodes how data are generated given parameters, and Bayes' theorem combines them into a posterior distribution. Every downstream quantity (point estimate, interval, prediction, decision) is derived from the posterior, so uncertainty propagates automatically instead of being bolted on.

**When to use it.**
- Sample sizes are small and prior knowledge (physical constraints, historical data, expert judgement) carries real information.
- The quantity of interest is a probability statement about a parameter ("there is an 80% chance the effect exceeds zero"), which frequentist intervals cannot supply.
- The model has hierarchical or nested structure (patients within hospitals, stores within regions) and partial pooling is wanted.
- Uncertainty must feed a downstream decision with an explicit loss function.
- Data arrive sequentially and beliefs must be updated incrementally.

**When not to use it.** When priors cannot be defended and results are prior-sensitive; when the audience or regulator demands frequentist error control; when the posterior is expensive to sample and a point estimate would suffice.

**Assumptions and failure modes.** The likelihood must be correctly specified. Improper or vague priors can produce improper posteriors. With large data the prior washes out, so the effort is wasted; with small data the prior dominates, so it must be justified.

**Where it is used.** Clinical trial design and adaptive trials, pharmacometrics, actuarial reserving, A/B testing at low traffic, sports analytics, astrophysics parameter estimation, econometric forecasting, reliability engineering.

---

### 1.2 Bayes Factors

**What it does.** Computes the ratio of marginal likelihoods (evidence) of two models, quantifying how much the data shift the relative odds between them. A Bayes factor of 10 means the data are ten times more likely under one model than the other. Unlike p-values, it can support the null as well as reject it, and models need not be nested.

**When to use it.**
- Comparing two non-nested scientific hypotheses on equal footing.
- Evidence *for* a null (no effect) is needed, not merely failure to reject it.
- Sequential or optional-stopping designs where p-values would be invalidated by repeated looks.
- Model comparison where an explicit Occam penalty is desired: the marginal likelihood automatically penalizes unnecessary flexibility.

**When not to use it.** When priors on parameters are arbitrary, because the marginal likelihood is highly sensitive to prior width in a way the posterior is not. For predictive performance questions, cross-validation or information criteria are usually more robust.

**Assumptions and failure modes.** Requires proper priors. Notoriously hard to compute (bridge sampling, nested sampling, Savage-Dickey ratio). Lindley's paradox: with large samples, Bayes factors and p-values can point in opposite directions.

**Where it is used.** Experimental psychology and replication research, cognitive modelling, genetics (association evidence), forensic science (likelihood ratios of evidence), gravitational-wave model selection.

---

### 1.3 Bayesian Model Averaging (BMA)

**What it does.** Rather than committing to one model, it forms predictions as a weighted mixture across a set of candidate models, weighting each by its posterior probability. This makes model uncertainty an explicit component of predictive uncertainty instead of ignoring it.

**When to use it.**
- Several models are plausible and none is decisively best, particularly with many candidate predictor subsets.
- Predictive intervals must be honest about the fact that model choice itself is uncertain.
- Post-selection inference would otherwise be over-confident because a model was chosen using the same data used for inference.

**When not to use it.** When one model is clearly dominant, when interpretability of a single equation is the deliverable, or when the model space is so large that enumeration is infeasible without approximations (though MC³ and stochastic search mitigate this).

**Assumptions and failure modes.** Assumes the true model is in the candidate set (the "M-closed" view); under misspecification, stacking or pseudo-BMA weights based on predictive performance often behave better. Weights can be unstable and prior-sensitive.

**Where it is used.** Macroeconomic growth regressions, weather and climate multi-model ensembles, epidemiological risk models, credit scoring, ecological niche modelling.

---

## 2. Classical (Frequentist) Inference

### 2.1 Classical Hypothesis Testing

**What it does.** Sets up a null hypothesis, computes a test statistic whose distribution under the null is known, and reports the probability of observing a statistic at least as extreme (the p-value). Controls the long-run rate of false positives at a chosen level. The family includes t-tests, chi-squared tests, ANOVA F-tests, nonparametric rank tests and likelihood ratio tests.

**When to use it.**
- A binary decision with controlled error rates is required (approve or reject, ship or do not ship).
- Regulatory or publication contexts that mandate frequentist error control.
- Confirmatory analysis of a pre-registered, pre-specified hypothesis.
- Randomized experiments where the null of no effect is exactly the right reference.

**When not to use it.** Exploratory analysis with many comparisons and no correction; questions about effect magnitude (use intervals); questions where evidence for the null matters (use Bayes factors or equivalence tests).

**Assumptions and failure modes.** Test-specific distributional assumptions (normality, equal variance, independence). p-values are widely misinterpreted as the probability the null is true. Multiple testing inflates false positives; correct with Bonferroni, Holm or Benjamini-Hochberg FDR. Statistical significance is not practical significance, and with large n trivial effects become significant.

**Where it is used.** Clinical trials, agricultural and industrial experiments, quality control, A/B testing, social science research, manufacturing acceptance sampling.

---

### 2.2 Confidence Intervals

**What it does.** Produces a range of parameter values compatible with the data, constructed so that across hypothetical repeated samples a stated proportion (e.g. 95%) of such intervals would contain the true parameter. Communicates both the estimate and its precision, which a p-value alone does not.

**When to use it.**
- Reporting an effect size, which should almost always accompany or replace a bare significance test.
- Assessing practical relevance: an interval spanning trivially small to enormous effects signals an underpowered study even if "significant".
- Equivalence and non-inferiority testing, where the interval is compared against a margin.
- Meta-analysis inputs, where intervals feed precision weighting.

**When not to use it.** When a direct probability statement about the parameter is needed (that is a Bayesian credible interval); when the sampling distribution is badly non-normal and small (use bootstrap or exact intervals).

**Assumptions and failure modes.** Coverage relies on the assumed sampling distribution and correct standard errors; clustered or autocorrelated data need robust or cluster-robust errors. Common misreading: "95% probability the parameter is in this interval" is a Bayesian statement, not a frequentist one.

**Where it is used.** Every empirical field: epidemiology (risk ratios), economics (elasticities), polling (margin of error), engineering tolerance analysis, laboratory measurement.

---

### 2.3 Analysis of Variance (ANOVA)

**What it does.** Decomposes total variability in an outcome into portions attributable to experimental factors, their interactions, and residual error, then tests whether factor-attributable variance exceeds what noise alone would produce. Generalizes the two-sample t-test to many groups and many crossed or nested factors.

**When to use it.**
- Comparing three or more group means without inflating type I error through pairwise tests.
- Factorial experiments where interaction effects (does the effect of A depend on the level of B?) are of interest.
- Repeated-measures or split-plot designs with within-subject factors.
- As a variance-accounting tool: how much of the outcome's variability does each factor own?

**When not to use it.** Heavily unbalanced observational data with confounding (use regression with covariates); non-normal or heteroscedastic responses (use GLM, Welch's ANOVA, or Kruskal-Wallis); hierarchical data with random effects (use mixed models).

**Assumptions and failure modes.** Independence, normal residuals, homogeneity of variance, and for repeated measures, sphericity (correct via Greenhouse-Geisser). A significant omnibus F says only that some means differ; post-hoc tests (Tukey HSD, Scheffé) identify which.

**Where it is used.** Agricultural field trials (its origin), psychology and neuroscience experiments, clinical research, industrial process optimization, sensory and taste panels, gauge R&R studies in manufacturing.

---

### 2.4 Likelihood Ratio Test

**What it does.** Compares a restricted (null) model against a more general (alternative) model by taking twice the difference in maximized log-likelihoods. Under the null and regularity conditions, this statistic is asymptotically chi-squared with degrees of freedom equal to the number of restrictions.

**When to use it.**
- Deciding whether added parameters earn their keep in nested parametric models: extra predictors, extra polynomial terms, extra random effects.
- Testing distributional restrictions inside a parametric family (for example, whether a negative binomial reduces to a Poisson).
- Model reduction in GLMs, survival models and structural equation models.
- As the standard test in maximum likelihood settings where Wald tests behave poorly near parameter boundaries.

**When not to use it.** Non-nested comparisons (use AIC, BIC or Vuong's test); parameters on the boundary of the space, such as testing a variance component equal to zero, where the null distribution is a mixture of chi-squareds; very small samples where the asymptotic approximation is poor.

**Assumptions and failure modes.** Requires correct specification of the larger model and both models fit to identical data. Sensitive to misspecification of the likelihood; robust or sandwich-adjusted variants exist.

**Where it is used.** Econometrics, survival analysis (Cox model term selection), genetics (linkage and association), psychometrics, item response theory, ecological modelling.

---

## 3. Regression Methods

### 3.1 Ordinary Least Squares (OLS) Regression

**What it does.** Fits a linear relationship between predictors and a continuous outcome by minimizing the sum of squared residuals. Under Gauss-Markov conditions it is the best linear unbiased estimator, and its coefficients have a direct interpretation as the expected change in outcome per unit change in a predictor, holding others fixed.

**When to use it.**
- The outcome is continuous and roughly symmetric, and the relationship is approximately linear (possibly after transformation).
- Interpretable coefficients matter more than raw predictive accuracy.
- A transparent baseline is needed before trying anything more elaborate.
- Sample size comfortably exceeds the number of predictors and predictors are not severely collinear.

**When not to use it.** Binary, count or bounded outcomes (use a GLM); heavy-tailed outcomes or outlier-dominated data (use robust or quantile regression); more predictors than observations (use ridge, lasso or dimension reduction); strongly nonlinear structure (use splines, GAMs or trees).

**Assumptions and failure modes.** Linearity, independent errors, constant error variance, and for exact inference, normal errors. Autocorrelation and clustering break standard errors (fix with Newey-West or cluster-robust errors). Multicollinearity inflates variance without biasing predictions. Omitted variable bias makes coefficients non-causal. Influential points can dominate the fit.

**Where it is used.** Economics and finance (factor models, hedonic pricing), marketing mix modelling, epidemiology, engineering calibration curves, real estate valuation, essentially every applied quantitative field as the first model tried.

---

### 3.2 Generalized Linear Models (GLM)

**What it does.** Extends linear regression to outcomes whose distribution is not normal by specifying an exponential-family distribution and a link function connecting the linear predictor to the mean. Logistic regression (binomial, logit link), Poisson regression (counts, log link), gamma regression (positive skewed costs) and probit models are all special cases of one framework fit by iteratively reweighted least squares.

**When to use it.**
- Binary or proportion outcomes: conversion, default, mortality, pass/fail.
- Count outcomes: claims, incidents, visits, defects, often with an exposure offset.
- Positive, right-skewed continuous outcomes such as costs, durations or insurance severities.
- Any setting where the mean-variance relationship is structurally non-constant and OLS would produce impossible predictions (negative counts, probabilities outside 0 to 1).

**When not to use it.** Overdispersed counts without adjustment (use quasi-Poisson, negative binomial, or a mixture model); excess zeros (use zero-inflated or hurdle models); non-independent observations (use GLMM or GEE); highly nonlinear predictor effects (use GAMs).

**Assumptions and failure modes.** Correct link and variance function, independence, and linearity on the link scale. Complete or quasi-complete separation breaks logistic fits (use penalized likelihood such as Firth). Coefficients are on the link scale, so marginal effects or odds ratios must be computed for interpretation.

**Where it is used.** Credit risk and default modelling, insurance pricing (the industry standard for frequency and severity), epidemiology and public health, clinical prediction models, digital marketing conversion analysis, ecology (species counts).

---

### 3.3 Ridge Regression

**What it does.** Adds an L2 penalty on the squared magnitude of coefficients to the least squares objective. This shrinks all coefficients smoothly toward zero without setting any exactly to zero, trading a small amount of bias for a large reduction in variance. It has a closed-form solution and remains well-defined even when predictors outnumber observations.

**When to use it.**
- Predictors are strongly correlated and OLS coefficients are unstable, wildly signed or huge in magnitude.
- The model is overparameterized (p close to or exceeding n).
- All predictors are believed to contribute something and none should be discarded outright.
- Prediction accuracy matters more than sparse interpretation, for example with many genomic markers or many correlated sensors.

**When not to use it.** When the goal is variable selection or a parsimonious model (use lasso or elastic net); when only a handful of predictors truly matter, since ridge will keep them all.

**Assumptions and failure modes.** Predictors must be standardized, because the penalty is scale-dependent. The penalty strength must be chosen by cross-validation. Coefficients are biased, so standard inference does not apply directly. Ridge is equivalent to a Bayesian linear model with a Gaussian prior on coefficients.

**Where it is used.** Genomics and high-throughput biology, chemometrics and spectroscopy, portfolio covariance shrinkage, neural network weight decay, marketing models with correlated media channels.

---

### 3.4 Lasso Regression

**What it does.** Adds an L1 penalty on the absolute value of coefficients, which drives some coefficients exactly to zero. This performs estimation and variable selection in a single step, producing a sparse, readable model from a large candidate set.

**When to use it.**
- Hundreds or thousands of candidate predictors with a belief that only a few matter.
- An interpretable shortlist of drivers is a deliverable, not just a prediction.
- Automated feature selection is needed in a reproducible pipeline, replacing ad hoc stepwise selection.
- Signal recovery problems where the underlying representation is genuinely sparse.

**When not to use it.** When predictors are highly correlated in groups, since lasso arbitrarily picks one and drops the rest (use elastic net or group lasso); when n is small relative to the number of true nonzero effects; when unbiased coefficient estimates are needed (use relaxed lasso or post-selection refitting).

**Assumptions and failure modes.** Requires standardized predictors and cross-validated penalty. Selection is unstable under resampling; assess with bootstrap or stability selection. Naive p-values after lasso selection are invalid; use selective inference or sample splitting. Coefficients are shrunk toward zero and therefore attenuated.

**Where it is used.** Genomics (GWAS and expression data), compressed sensing and MRI reconstruction, text classification with sparse features, credit scorecard variable screening, econometric nowcasting with many indicators.

---

### 3.5 Quantile Regression

**What it does.** Models conditional quantiles of the outcome rather than its conditional mean by minimizing an asymmetrically weighted absolute loss. Fitting several quantiles gives a picture of how the entire conditional distribution shifts with the predictors, not just its centre.

**When to use it.**
- The tails are the point: value-at-risk, worst-case delivery times, 95th-percentile latency, low-birthweight outcomes.
- Variance or distribution shape changes with the predictors, so a single mean model conceals the phenomenon.
- Outliers or heavy tails make the mean an unrepresentative summary and OLS unreliable.
- Heterogeneous treatment effects are suspected, where an intervention helps the bottom of the distribution but not the top.
- Distribution-free prediction intervals are wanted directly from the data.

**When not to use it.** Small samples in the far tails, where quantile estimates are noisy; when a single interpretable mean effect is what stakeholders need.

**Assumptions and failure modes.** No distributional assumption on errors, which is its major advantage, but inference typically needs bootstrap or kernel-based standard errors. Estimated quantile curves can cross at extreme predictor values; non-crossing constraints or monotone rearrangement fix this. Computationally a linear program, heavier than OLS.

**Where it is used.** Financial risk (VaR and expected shortfall inputs), service level agreements and latency engineering, labour economics (wage inequality across the distribution), growth charts in paediatrics, energy demand peaks, real estate price segmentation.

---

## 4. Causal Inference

### 4.1 Potential Outcomes Framework

**What it does.** Defines a causal effect as the comparison between the outcome a unit would have under treatment and the outcome the same unit would have under control. Since only one is ever observed, causal inference becomes a missing data problem, and the task is to make the observed comparison stand in credibly for the unobservable one. It provides the vocabulary (ATE, ATT, CATE, SUTVA, ignorability) that all other designs below inherit.

**When to use it.**
- Any time the question is "what would happen if we did X" rather than "what is associated with X".
- Designing an experiment: it clarifies exactly which estimand randomization identifies.
- Evaluating an intervention from observational data, where the estimand and the identifying assumptions must be stated explicitly before any modelling.
- Reasoning about heterogeneous effects and about which subpopulation an estimate actually applies to.

**When not to use it.** Pure prediction tasks, where correlation suffices and no intervention is contemplated. Settings with heavy interference between units, where SUTVA fails and the framework needs extension.

**Assumptions and failure modes.** SUTVA (no interference between units, one version of treatment), positivity (every unit could plausibly receive either condition), and ignorability (treatment assignment is as good as random given measured covariates). The last is untestable and is where most observational studies fail. Conditioning on post-treatment variables or colliders introduces bias rather than removing it.

**Where it is used.** Programme evaluation in economics and public policy, clinical trials and comparative effectiveness research, technology experimentation platforms, epidemiology, marketing incrementality measurement.

---

### 4.2 Difference-in-Differences (DiD)

**What it does.** Compares the before-to-after change in a treated group with the before-to-after change in an untreated comparison group. Differencing over time removes any fixed characteristics of each group, and differencing across groups removes any shocks common to both, isolating the treatment effect under a parallel trends assumption.

**When to use it.**
- A policy, price change, product launch or regulation hits some units at a known time and not others.
- Randomization was impossible but panel or repeated cross-sectional data exist for both groups before and after.
- Geographic or market-level rollouts, where whole regions switch on at different dates (staggered adoption).
- Natural experiments in labour, health and competition policy.

**When not to use it.** When treated and control groups were already diverging before treatment; when treatment timing is itself caused by the outcome trajectory (anticipation or Ashenfelter dip); when spillovers contaminate the control group.

**Assumptions and failure modes.** Parallel trends is the core assumption: absent treatment, the two groups' outcomes would have moved in parallel. Support it with pre-trend event-study plots, not just a claim. Standard errors must be clustered at the treatment assignment level or they are badly understated. Two-way fixed effects with staggered adoption and heterogeneous effects can be severely biased, even sign-flipped; use Callaway-Sant'Anna, Sun-Abraham, or stacked designs. Synthetic control is the natural alternative when there is a single treated unit.

**Where it is used.** Minimum wage and employment studies, health insurance expansions, environmental regulation, pricing and promotion evaluation in retail, platform policy changes, tax reform analysis.

---

### 4.3 Instrumental Variables (IV)

**What it does.** Finds a variable that shifts treatment take-up but affects the outcome only through treatment, then uses only the treatment variation induced by that instrument. This isolates variation that is plausibly exogenous, allowing causal estimation even when unmeasured confounders are present. Two-stage least squares is the standard estimator.

**When to use it.**
- Unmeasured confounding is likely and no amount of covariate adjustment will fix it.
- Non-compliance in a randomized trial, where random assignment instruments for actual treatment received, giving the complier average causal effect.
- Encouragement designs, where a nudge is randomized rather than the behaviour itself.
- Simultaneity or reverse causality, such as prices and quantities determined jointly.
- Measurement error in the treatment variable, which IV can also correct.

**When not to use it.** When no credible instrument exists, which is the usual case. Weak instruments are worse than no instrument: they amplify bias and produce misleading confidence intervals.

**Assumptions and failure modes.** Relevance (checkable, first-stage F well above 10), exclusion (untestable and where most IV claims break), independence of the instrument, and monotonicity for LATE interpretation. The estimate applies to compliers only, which may not be the population of policy interest. Use Anderson-Rubin intervals when instruments are weak.

**Where it is used.** Labour economics (quarter of birth, distance to college), health economics (physician prescribing preference), Mendelian randomization in genetic epidemiology, digital experimentation with partial compliance, demand estimation in industrial organization.

---

### 4.4 Regression Discontinuity (RD)

**What it does.** Exploits a rule that assigns treatment based on whether a continuous running variable crosses a threshold. Units marginally above and marginally below the cutoff are treated as effectively randomized, so the jump in outcomes at the cutoff estimates the causal effect locally.

**When to use it.**
- Eligibility is determined by a sharp rule: test scores for scholarships, income thresholds for benefits, credit scores for loan approval, age cutoffs for programmes.
- Assignment is administrative and the exact rule is documented.
- A highly credible causal estimate is worth more than external generalizability.
- Fuzzy variants apply when crossing the threshold changes the probability of treatment rather than guaranteeing it, combining RD with IV logic.

**When not to use it.** When units can precisely manipulate their position relative to the cutoff (test the density for bunching with a McCrary test); when the effect away from the cutoff is what matters; when data thin out near the threshold.

**Assumptions and failure modes.** Continuity of all other determinants of the outcome at the cutoff. Results are local to the threshold and do not generalize far from it. Sensitive to bandwidth and polynomial order; use local linear regression with a data-driven bandwidth rather than high-order global polynomials. Check covariate balance and placebo cutoffs.

**Where it is used.** Education policy (class size, remedial programmes, scholarship effects), credit and lending decisions, electoral studies (close elections as regression discontinuity in vote share), health programme eligibility, air quality and regulatory thresholds.

---

### 4.5 Propensity Score Matching

**What it does.** Models the probability of receiving treatment given observed covariates, then pairs or weights treated and control units with similar propensity scores so the two groups look balanced on those covariates. Reduces a high-dimensional matching problem to matching on one number.

**When to use it.**
- Observational comparison where treated and control groups differ substantially on many measured covariates.
- Sample sizes large enough to find good matches, with substantial overlap in covariate distributions.
- Where a transparent, checkable balance diagnostic is valuable for a non-technical audience.
- As one component of a doubly robust estimator (augmented IPW, TMLE), which is generally preferred to matching alone.

**When not to use it.** When the confounders that matter are unmeasured, since the method can only balance what is observed; when overlap is poor and matches are forced across incomparable units; when the number of treated units is tiny.

**Assumptions and failure modes.** Ignorability and positivity. Balance must be verified after matching (standardized mean differences below roughly 0.1), and the propensity model must not be judged by its own predictive accuracy. Inverse probability weights explode when scores approach zero or one; trim or stabilize. Matching discards data and can increase imbalance if done carelessly; consider full matching, coarsened exact matching, or entropy balancing.

**Where it is used.** Comparative effectiveness research and pharmacoepidemiology, health services research using claims data, labour market programme evaluation, marketing attribution and uplift modelling, education interventions.

---

### 4.6 Do-Calculus (Pearl)

**What it does.** Provides a formal graphical algebra for deciding whether a causal quantity can be computed from observational data, given a causal diagram. Three rules license the insertion, deletion or exchange of interventions and observations, and their repeated application either yields an estimable formula or proves that none exists.

**When to use it.**
- Before estimation, to decide which variables to adjust for and, equally important, which to leave alone.
- When adjustment sets are non-obvious, involving mediators, colliders or feedback structures.
- To justify identification strategies such as the front-door criterion, where the back-door criterion fails.
- For transportability questions: whether results from one population can be carried to another with different characteristics.
- To settle disagreements about "control for everything" instincts, which frequently introduce collider bias.

**When not to use it.** When the causal graph itself is unknown and contested, since conclusions are only as good as the assumed structure; when a simple randomized experiment is available and cheaper than the argument.

**Assumptions and failure modes.** Requires a DAG asserting the qualitative causal structure, including the absence of certain edges, which is a strong and often untestable claim. Identification is a separate question from estimation: a formula may be identified in principle yet hopeless in practice due to positivity violations or sparse data.

**Where it is used.** Epidemiology (adjustment set selection is now standard practice), machine learning fairness and explainability, causal discovery research, policy transportability across settings, root-cause analysis in complex engineered systems.

---

## 5. Information Theory and Model Selection

### 5.1 KL Divergence

**What it does.** Measures the expected information lost when one distribution is used to approximate another, equivalently the extra bits needed to encode samples from P using a code optimized for Q. It is non-negative, zero only when the distributions coincide, and asymmetric, so the direction of comparison encodes a genuine modelling choice.

**When to use it.**
- As the objective in variational inference, where minimizing KL to the posterior defines the approximation.
- Training generative and classification models, since cross-entropy loss is KL plus a constant.
- Detecting distribution shift or data drift between a reference period and production data.
- Regularizing policies toward a reference in reinforcement learning and language model fine-tuning.
- Quantifying how much a posterior has moved from its prior, as an information gain measure in experimental design.

**When not to use it.** As a symmetric distance, since it is not a metric and violates the triangle inequality (use Jensen-Shannon divergence or Wasserstein distance if a true distance is needed); when supports do not overlap, since KL becomes infinite.

**Assumptions and failure modes.** Q must be absolutely continuous with respect to P, or the divergence diverges. Forward KL is mass-covering and produces over-dispersed fits; reverse KL is mode-seeking and produces over-confident, mode-collapsed fits. Estimating KL from samples in high dimensions is difficult.

**Where it is used.** Machine learning objectives everywhere, model monitoring and drift detection in production ML, information geometry, coding and compression theory, Bayesian experimental design, neuroscience (predictive coding).

---

### 5.2 Mutual Information

**What it does.** Quantifies how much knowing one variable reduces uncertainty about another, in bits or nats. Unlike correlation it captures arbitrary nonlinear dependence and is exactly zero if and only if the variables are independent, making it a general-purpose dependence measure.

**When to use it.**
- Screening features for relevance when relationships may be nonlinear or non-monotonic and correlation would miss them.
- Detecting dependence between variables of mixed type, including categorical and continuous.
- Clustering and representation learning objectives (information bottleneck, contrastive learning).
- Measuring shared information between neural signals, channels or time series.
- Registration of medical images across modalities, where intensity relationships are nonlinear.

**When not to use it.** When the direction or functional form of a relationship matters, since MI is symmetric and unsigned; when data are scarce and high-dimensional, where estimates are badly biased upward.

**Assumptions and failure modes.** Estimation is the hard part: binning is sensitive to bin width, and k-nearest-neighbour estimators (Kraskov) or neural estimators (MINE) have their own biases. MI is not normalized by default, so use normalized variants for comparison. High MI does not imply causation, only dependence.

**Where it is used.** Feature selection pipelines, neuroscience and neural coding, bioinformatics (gene network inference), medical image registration, communications and channel capacity, independent component analysis.

---

### 5.3 AIC / BIC

**What it does.** Scores models by their maximized log-likelihood penalized by complexity. AIC penalizes two per parameter and targets out-of-sample predictive accuracy, asymptotically matching leave-one-out cross-validation. BIC penalizes log(n) per parameter, favours simpler models as data grow, and is consistent for recovering the true model when it is in the candidate set.

**When to use it.**
- Comparing many models quickly, including non-nested ones, without refitting through cross-validation.
- AIC when the goal is predictive performance and the true model is assumed to be outside the candidate set.
- BIC when the goal is identifying a parsimonious or true structure, for example the number of components in a mixture or lags in a time series model.
- Ranking a whole candidate set with relative weights rather than picking one winner (Akaike weights feed model averaging).

**When not to use it.** When models are fit to different data or different transformations of the response, which makes likelihoods incomparable; with small samples relative to parameters (use AICc); with penalized or hierarchical models where the effective number of parameters is not the raw count (use DIC, WAIC or PSIS-LOO).

**Assumptions and failure modes.** Both rely on likelihood-based asymptotics. Differences of less than about two AIC units are not meaningful. BIC's consistency argument assumes the true model is in the set. Neither validates a model in absolute terms: the best of a set of bad models still scores best.

**Where it is used.** Time series order selection (ARIMA, VAR lag length), mixture model component selection, ecological and capture-recapture modelling, econometrics, psychometrics, phylogenetics.

---

## 6. Extreme Value Theory

### 6.1 Extreme Value Theory (EVT)

**What it does.** Provides asymptotic theory for the tail rather than the centre of a distribution. The Fisher-Tippett-Gnedenko theorem shows block maxima converge to the Generalized Extreme Value family (Gumbel, Fréchet, Weibull), and the Pickands-Balkema-de Haan theorem shows exceedances over a high threshold converge to the Generalized Pareto distribution. This licenses extrapolation beyond the largest event ever observed, which no empirical quantile can do.

**When to use it.**
- Estimating return levels for events rarer than the observation window: the 200-year flood from 60 years of records.
- Risk capital and solvency requirements at extreme quantiles (99.5% and beyond).
- Designing infrastructure to withstand rare loads: dams, sea walls, bridges, data centre cooling.
- Reinsurance pricing for high layers, where no or few losses have ever pierced the attachment point.
- Any question of the form "how bad can it get" where a normal or lognormal assumption would be dangerously optimistic.

**When not to use it.** For central-tendency questions, where it wastes almost all the data; when the series is strongly non-stationary and the tail itself is shifting, unless covariates or trends are built into the GEV or GPD parameters; when there are too few exceedances to estimate the shape parameter with any precision.

**Assumptions and failure modes.** Independence or explicit handling of clustering via declustering and the extremal index. Threshold choice for peaks-over-threshold is a bias-variance tradeoff diagnosed with mean residual life and parameter stability plots. The shape parameter drives everything: positive means heavy-tailed with infinite higher moments, and it is estimated from the fewest data points. Multivariate and spatial extremes require dedicated dependence models, since tail dependence is not captured by correlation.

**Where it is used.** Hydrology and flood defence, climate extremes attribution, catastrophe modelling and reinsurance, operational risk and market risk capital under Basel and Solvency II, structural and offshore engineering, materials failure analysis, network traffic peak load planning.

---

## 7. Resampling and Validation

### 7.1 K-Fold Cross-Validation

**What it does.** Partitions the data into k roughly equal folds, trains on k minus one and evaluates on the held-out fold, then rotates and averages. Every observation is used for both training and validation exactly once, producing a nearly unbiased estimate of out-of-sample performance at k times the training cost.

**When to use it.**
- Tuning hyperparameters (penalty strength, tree depth, number of components) with an honest performance signal.
- Comparing candidate models on predictive accuracy rather than in-sample fit.
- Moderate-sized datasets where a single train-test split would be too noisy or too wasteful.
- Estimating generalization error before deployment, with stratification when classes are imbalanced.

**When not to use it.** Time series or any data with temporal ordering, since random folds leak the future into the past (use rolling-origin or blocked CV); grouped or clustered data where the same subject appears in several folds (use group k-fold); when a final unbiased estimate is needed after extensive tuning, which requires a nested CV or a genuinely untouched holdout.

**Assumptions and failure modes.** Exchangeability of observations. The most common practical error is performing preprocessing (scaling, imputation, feature selection) on the full dataset before splitting, which leaks information and inflates scores; all preprocessing must be inside the fold. k of five or ten is the usual bias-variance compromise. Repeated CV reduces variance in the estimate.

**Where it is used.** Every applied machine learning workflow, clinical prediction model development and TRIPOD reporting, credit scoring validation, chemometrics, competition benchmarking.

---

### 7.2 Leave-One-Out Cross-Validation (LOO-CV)

**What it does.** The limiting case of k-fold where each single observation is held out in turn. Training sets are as large as possible, so bias in the performance estimate is minimal, at the cost of n model fits and higher variance in the estimate itself.

**When to use it.**
- Small datasets where holding out 10% would materially degrade training.
- Models with closed-form LOO shortcuts, such as the hat-matrix formula for linear regression (PRESS statistic) or kernel methods, making it essentially free.
- Bayesian model comparison, where PSIS-LOO approximates it efficiently from a single posterior sample and provides diagnostics (Pareto k) that flag unreliable cases.
- Sensitivity analysis: identifying observations whose removal changes conclusions.

**When not to use it.** Large datasets with expensive models, where the cost is prohibitive; model selection among very similar models, where the high variance of the LOO estimate makes the comparison unstable; dependent data, where leaving out one point still leaves its near-duplicates in training.

**Assumptions and failure modes.** Nearly unbiased but high variance, since the n training sets are almost identical and their errors are correlated. For clustered or time series data it is optimistic; use leave-one-group-out or leave-future-out. Pareto k values above 0.7 in PSIS-LOO signal influential observations where the approximation fails and refitting is required.

**Where it is used.** Bayesian workflow and Stan-based modelling, small-sample biomedical studies, geostatistics (cross-validating kriging), spectroscopy calibration, psychometric model checking.

---

### 7.3 Permutation Tests

**What it does.** Constructs the null distribution of a test statistic directly from the data by repeatedly shuffling group labels and recomputing the statistic. If the observed value sits far in the tail of the shuffled distribution, the association is unlikely to be an artefact of chance labelling. No parametric distribution is assumed at any point.

**When to use it.**
- Small samples where asymptotic approximations are untrustworthy.
- Test statistics with unknown or intractable null distributions, including custom, non-standard metrics.
- Randomized experiments, where permutation exactly mirrors the physical randomization that generated the data (Fisher's randomization test).
- Multiple testing with complex dependence, where max-statistic permutation controls family-wise error without conservative corrections.
- Verifying that a machine learning model's performance exceeds chance, by permuting labels and re-scoring.

**When not to use it.** When observations are not exchangeable under the null, such as time series with autocorrelation or clustered data, unless the permutation scheme respects that structure (block or within-cluster permutation); when computation is prohibitive and an accurate analytic test exists.

**Assumptions and failure modes.** Exchangeability under the null is the single requirement, and it is easy to violate silently. Testing a difference in means when variances also differ makes the null hypothesis ambiguous. The achievable p-value resolution is limited by the number of permutations; with 1,000 permutations the smallest p-value is roughly 0.001.

**Where it is used.** Neuroimaging cluster inference (FSL, SPM, FreeSurfer), genomics and pathway analysis, ecology (community structure null models), A/B test validation, machine learning feature importance testing.

---

## 8. Structural Breaks

### 8.1 Change Point Detection

**What it does.** Locates times at which the statistical generating process of a series shifts, whether in mean, variance, trend, or full distribution. Offline methods scan the entire series retrospectively to segment it optimally; online methods flag breaks as data arrive. Key algorithms include CUSUM for cumulative deviation monitoring, binary segmentation and its wild variant, PELT for exact optimal segmentation in linear time, and Bayesian online change point detection which yields a posterior over run lengths.

**When to use it.**
- Deciding how far back training data remain representative, which is often the difference between a working and a broken forecast.
- Monitoring production systems, models or sensors for regime shifts and silent failures.
- Detecting the effect date of an intervention when the exact timing is not documented.
- Segmenting long series into homogeneous regimes before fitting separate models.
- Quality control on manufacturing lines, where CUSUM and EWMA charts detect small persistent shifts faster than Shewhart charts.

**When not to use it.** When change is gradual rather than abrupt, where trend or state space models fit better; when the series has strong seasonality that has not been removed, which generates spurious detections; when only a handful of observations follow the suspected break.

**Assumptions and failure modes.** Most methods assume independence within segments, so autocorrelation inflates false positives. The penalty or threshold governs the classic tradeoff between detection power and false alarms, and must be calibrated to the operational cost of each. Detection latency matters in online settings: a method that eventually finds every break but three months late may be useless. Heavy tails cause variance-change detectors to fire on single outliers.

**Where it is used.** Financial regime detection and volatility modelling, demand planning and supply chain forecasting, statistical process control, network intrusion and anomaly detection, ML model monitoring and drift alerting, climatology (station records), genomics (copy number variation segmentation).

---

## 9. Dependence Modelling

### 9.1 Copulas

**What it does.** Separates a joint distribution into its marginal distributions and a dependence function linking them, a decomposition guaranteed by Sklar's theorem. This lets each variable have whatever marginal it needs (fat-tailed returns, skewed durations, discrete counts) while dependence is modelled independently, including asymmetric and tail-concentrated dependence that correlation cannot express. Gaussian copulas capture symmetric dependence without tail dependence, t-copulas add joint tail dependence, and Archimedean families (Clayton, Gumbel, Frank) give asymmetric lower or upper tail behaviour.

**When to use it.**
- Portfolio and credit risk aggregation, where the question is the probability that many things go wrong at once and linear correlation understates joint extremes.
- Combining risks with wildly different marginal shapes into one joint model.
- Simulating correlated scenarios with realistic marginals for stress testing and capital modelling.
- Hydrology and insurance, where joint exceedance of several variables (rainfall and river level, wind and flood) drives the loss.
- Any setting where dependence in the tails differs from dependence in the body of the distribution.

**When not to use it.** When variables are jointly normal anyway, where the added machinery buys nothing; in high dimensions without structure, since copula estimation degrades quickly (use vine copulas or factor copulas instead); when the dependence is dynamic and regime-switching, unless time-varying copulas are used.

**Assumptions and failure modes.** Copula family choice drives tail behaviour, and the wrong family silently understates joint extremes: the Gaussian copula's lack of tail dependence was a notable contributor to pre-2008 credit modelling failures. Marginals and copula are often fit in two stages, which affects standard errors. Discrete marginals break the uniqueness of the copula. Parameter estimation needs a lot of data in the tail, which is exactly where data are scarce.

**Where it is used.** Credit portfolio models and CDO pricing, insurance and reinsurance capital modelling under Solvency II, operational risk aggregation, hydrology and multi-hazard risk, energy price and load dependence, actuarial dependent lifetime modelling.

---

## 10. Spatial Statistics

### 10.1 Kriging (Gaussian Process Regression)

**What it does.** Predicts a value at an unsampled location as a weighted average of observations, with weights derived from a fitted model of spatial correlation rather than arbitrary distance rules. It is the best linear unbiased predictor under its assumptions and, crucially, returns a variance at every prediction point, so the map comes with an uncertainty map. The identical mathematics underlies Gaussian process regression in machine learning.

**When to use it.**
- Interpolating a spatially continuous quantity from scattered samples: contamination levels, ore grades, rainfall, soil properties, air quality.
- Deciding where to sample next, since the prediction variance directly identifies the least-known locations.
- Cheap emulation of an expensive simulator, where a GP surrogate is trained on a small design of runs.
- Bayesian optimization, where the GP posterior drives an acquisition function balancing exploration and exploitation.
- Any small-data regression problem where calibrated uncertainty matters more than raw flexibility.

**When not to use it.** Large datasets without approximation, since exact kriging costs O(n³) and memory O(n²) (use inducing points, nearest-neighbour GPs, or fixed-rank approaches); strongly non-stationary fields where one covariance function cannot describe the whole domain; abrupt discontinuities such as faults or coastlines, which smooth interpolation will blur across.

**Assumptions and failure modes.** Stationarity and often isotropy of the spatial process, plus a correctly specified variogram or kernel. Extrapolation beyond the data reverts to the mean with wide intervals, which is honest but sometimes surprising. Nugget effects absorb measurement error; setting it to zero forces exact interpolation of noisy points. Anisotropy must be modelled explicitly or directional structure is lost. Variants include ordinary, universal, indicator and co-kriging for different mean structures and multivariate cases.

**Where it is used.** Mining and resource estimation (its origin, from Danie Krige's gold grade work), environmental monitoring and contaminated land assessment, precision agriculture, meteorology and rainfall mapping, epidemiological disease mapping, computer experiment emulation, hyperparameter optimization in machine learning.

---

### 10.2 Variogram Analysis

**What it does.** Describes how the expected squared difference between two observations grows with the distance separating them. Fitting a model to the empirical variogram yields three interpretable quantities: the nugget (micro-scale variability and measurement error), the sill (total variance), and the range (the distance beyond which observations are effectively uncorrelated). These parameters are what kriging consumes.

**When to use it.**
- Always before kriging, since the variogram determines the weights and therefore the entire prediction surface.
- Diagnosing the spatial scale of a process: is variation driven by metre-scale or kilometre-scale structure?
- Designing sampling campaigns, since sampling much closer than the nugget-dominated scale wastes effort and sampling beyond the range gains no interpolation power.
- Detecting anisotropy by computing directional variograms and comparing ranges across directions.
- Assessing whether spatial structure exists at all: a pure nugget variogram means neighbouring values carry no information.

**When not to use it.** When the process has a strong trend, since the raw variogram will not stabilize and a residual variogram after detrending is required; with very few sample points, where empirical variogram bins are too noisy to fit reliably.

**Assumptions and failure modes.** Intrinsic stationarity. Lag binning choice materially affects the empirical curve. Outliers inflate variogram values disproportionately; robust estimators (Cressie-Hawkins) help. Fitting by eye remains common and is defensible, but weighted least squares or REML is more reproducible. Extrapolating the fitted model far beyond the largest observed lag is unsupported.

**Where it is used.** Geostatistics and mining, soil science, hydrogeology, environmental sampling design, remote sensing validation, forestry inventory, precision agriculture.

---

### 10.3 Moran's I (Spatial Autocorrelation)

**What it does.** Tests whether values at nearby locations are more similar (clustering), more dissimilar (checkerboard dispersion), or arranged at random, given a defined spatial weights matrix. Global Moran's I gives one number for the whole map; local versions (LISA, Getis-Ord Gi*) identify where specific hot and cold spots sit.

**When to use it.**
- Before any regression on spatial data, to check whether residuals are spatially autocorrelated and therefore whether standard errors are invalid.
- Identifying disease clusters, crime hotspots, deprivation concentrations or ecological patches.
- Confirming that a spatial pattern is real rather than an artefact of visual impression on a choropleth map.
- Deciding whether a spatial econometric model (spatial lag or spatial error) is needed instead of OLS.

**When not to use it.** As a substitute for a spatial model, since it is a diagnostic rather than an estimator; when the areal units are arbitrary and results would change entirely under different boundaries.

**Assumptions and failure modes.** The result depends heavily on the spatial weights matrix (contiguity, k-nearest neighbours, distance band, row-standardized or not), and this choice should be justified and sensitivity-tested. The Modifiable Areal Unit Problem means aggregation level changes conclusions. Local statistics involve many simultaneous tests and need FDR control. Significance is usually assessed by permutation rather than the normal approximation.

**Where it is used.** Epidemiology and disease surveillance, criminology and policing analytics, urban and regional economics, ecology and biogeography, retail site selection, public health inequality mapping.

---

## 11. Sensitivity Analysis and Uncertainty Quantification

### 11.1 Uncertainty Quantification (UQ)

**What it does.** The parent discipline concerned with characterizing, propagating and reducing uncertainty in model-based predictions. It distinguishes aleatory uncertainty (irreducible randomness in the system) from epistemic uncertainty (ignorance that more data or better models could reduce), covers forward propagation of input uncertainty to outputs, inverse problems where observations constrain uncertain inputs, and model-form uncertainty where the equations themselves are wrong.

**When to use it.**
- Any model whose output informs a consequential decision, where a single deterministic number would be misleading.
- Regulatory or safety contexts requiring documented confidence in predictions (nuclear, aerospace, pharmaceutical, structural).
- Model validation against sparse experimental data, where discrepancy must be quantified rather than assumed away.
- Deciding where to spend a data collection budget, by identifying which epistemic uncertainties dominate.

**When not to use it.** When the model is a rough directional aid and stakeholders will not act differently under any plausible uncertainty; when the dominant uncertainty is structural (the model is simply wrong) and quantifying parameter uncertainty would give false comfort.

**Assumptions and failure modes.** The most common failure is quantifying only parametric uncertainty while ignoring model-form error, producing tight intervals around a wrong answer. Input distributions are themselves assumptions and are often assigned with unwarranted confidence. Correlations between inputs are frequently ignored.

**Where it is used.** Nuclear safety and reactor licensing, aerospace certification, climate projection, computational fluid dynamics validation, pharmacokinetic modelling, financial capital modelling, digital twins.

---

### 11.2 Sensitivity Analysis

**What it does.** Determines how variation in model outputs can be attributed to variation in inputs. Local methods perturb one input at a time around a nominal point and measure the derivative-like response. Global methods vary all inputs simultaneously across their full ranges, capturing interactions and nonlinearity that local methods structurally cannot see. Screening methods (Morris elementary effects) sit in between, cheaply separating influential from negligible inputs.

**When to use it.**
- Prioritizing effort: which three of forty inputs actually deserve better data, tighter control or expert review.
- Model simplification, by fixing inputs shown to be non-influential.
- Stress-testing and challenging a model before it is trusted, including for audit and regulatory defence.
- Debugging: an input that should matter but does not usually indicates a coding or structural error.
- Communicating robustness to stakeholders: does the recommendation survive plausible input variation?

**When not to use it.** Local one-at-a-time analysis on models with strong interactions, where it will systematically mislead; when input ranges are arbitrary, since results are entirely conditional on the ranges chosen.

**Assumptions and failure modes.** One-at-a-time analysis explores a vanishingly small fraction of high-dimensional input space and misses all interactions, yet remains the most commonly used approach. Global methods cost many model evaluations, which motivates surrogates. Correlated inputs complicate attribution and require dedicated indices. Sensitivity is not importance in a decision sense: an input can dominate output variance yet be uncontrollable.

**Where it is used.** Environmental and climate modelling, engineering design, health economic models (mandatory in NICE and similar HTA submissions), financial valuation and actuarial models, systems biology, agent-based simulation.

---

### 11.3 Variance-Based Decomposition (Sobol / ANOVA-HDMR)

**What it does.** Decomposes the total variance of a model output into contributions from each input alone and from every combination of inputs. First-order Sobol indices give the variance share attributable to each input by itself; total-effect indices include all interactions that input participates in. The gap between first-order and total indices measures how much of an input's influence flows through interactions.

**When to use it.**
- Nonlinear or non-monotonic models where correlation-based or regression-based importance measures are invalid.
- When quantitative attribution is needed, not just a ranking: "input X owns 43% of output variance".
- Identifying interaction structure, for example that two inputs matter jointly but neither matters alone.
- Model reduction with justification: inputs whose total index is near zero can be fixed with quantified consequence.

**When not to use it.** When model evaluations are expensive and no surrogate is available, since convergence needs thousands of runs; when inputs are strongly correlated, where the standard decomposition is not unique (use Shapley effects instead); when output variance is a poor summary because the tail is what matters (use quantile-oriented or moment-independent measures).

**Assumptions and failure modes.** Assumes independent inputs. Requires careful sampling designs (Saltelli's scheme) and convergence checks via bootstrap confidence intervals on the indices. Variance as the sole measure of uncertainty is a limitation for skewed or multimodal outputs. Total indices are typically estimated less precisely than first-order ones.

**Where it is used.** Environmental and hydrological modelling, aerospace and automotive design, nuclear safety analysis, systems biology parameter prioritization, health economics, chemical process engineering.

---

### 11.4 Scenario Analysis

**What it does.** Defines a small number of internally coherent, named states of the world and evaluates the model under each. Rather than a probability distribution, it delivers a handful of concrete narratives (base case, downside, severe stress) that stakeholders can reason about and argue with directly.

**When to use it.**
- Communicating to boards, regulators and non-technical decision-makers who need to reason about specific futures rather than distributions.
- Deep uncertainty where probabilities cannot be credibly assigned, such as regulatory regime change, geopolitical disruption or technology discontinuity.
- Regulatory stress testing, where the scenario is prescribed by the supervisor.
- Strategic planning, where the value lies in surfacing which decisions are robust across all scenarios and which are bets on one.

**When not to use it.** As a substitute for probabilistic analysis when distributions are estimable, since it gives no likelihood information; when scenarios are chosen to justify a predetermined conclusion, which is the most common abuse.

**Assumptions and failure modes.** Scenarios must be internally consistent: varying one driver to its extreme while holding correlated drivers at base is incoherent. There is no probability attached, so a severe scenario cannot be interpreted as a quantile. Anchoring on the base case is pervasive. Genuine tail events are typically outside the scenario set, which is precisely the problem stress testing exists to address.

**Where it is used.** Bank stress testing (CCAR, ICAAP, EBA exercises), insurance ORSA, corporate strategic planning, climate transition and physical risk disclosure (TCFD, NGFS scenarios), energy system planning, defence and security planning.

---

### 11.5 Tornado Diagram Analysis

**What it does.** Runs a structured one-at-a-time sensitivity analysis and displays the results as horizontal bars sorted by the width of the output swing, producing a funnel shape. Each bar shows how far the output moves when a single input travels from its low to its high value with all others held at base.

**When to use it.**
- Executive and client communication, where an immediately legible ranking of drivers is worth more than technical rigour.
- Early-stage screening of a model with many inputs, to focus subsequent analysis.
- Valuation and business case work, where the question is which assumptions the answer hinges on.
- As the standard visual in decision analysis, preceding a fuller probabilistic treatment.

**When not to use it.** Models with strong interactions, since the diagram cannot represent them and will attribute influence incorrectly; as the final word on importance in any nonlinear model.

**Assumptions and failure modes.** Bar widths depend entirely on the chosen input ranges, so ranges must be comparable in plausibility (for example, each input's 10th to 90th percentile) or the ranking is meaningless. Interactions are invisible. Asymmetric bars reveal nonlinearity and should be shown rather than averaged away. It is often presented with more authority than a one-at-a-time method deserves.

**Where it is used.** Corporate finance and DCF valuation, oil and gas project economics, pharmaceutical health economic models, project risk management, capital allocation reviews.

---

### 11.6 Perturbation Analysis

**What it does.** Computes derivatives of outputs with respect to inputs, either analytically, by finite differences, by adjoint methods, or by automatic differentiation. The resulting gradients quantify local response rates and, when normalized, give directly comparable local sensitivities.

**When to use it.**
- Smooth, differentiable models where an exact local rate of change is the quantity of interest.
- High-dimensional problems where adjoint methods give all sensitivities at roughly the cost of one extra model run, which no sampling method can match.
- Optimization and calibration, where gradients drive the search.
- Error propagation using the delta method, converting input variances to output variance through first-order Taylor expansion.
- Real-time control and design, where recomputing a global analysis is infeasible.

**When not to use it.** Non-smooth, discontinuous or threshold-driven models where derivatives do not exist or do not describe behaviour; large input ranges, since a local derivative says nothing about response far from the nominal point; strongly interacting systems, where the gradient at one point is unrepresentative.

**Assumptions and failure modes.** Finite differences are sensitive to step size, trading truncation error against numerical cancellation; automatic differentiation avoids this entirely and should be preferred where available. Validity is strictly local. The delta method assumes near-linearity over the uncertainty range, which fails for wide uncertainties.

**Where it is used.** Computational fluid dynamics and shape optimization, weather model adjoint sensitivity, machine learning training, chemical kinetics, structural design optimization, financial Greeks computation.

---

### 11.7 Design of Experiments (DoE)

**What it does.** Chooses which combinations of inputs to evaluate so that the maximum information about input-output relationships is extracted from the minimum number of runs. Full factorial designs cover every combination; fractional factorials trade the ability to resolve high-order interactions for dramatically fewer runs; response surface designs (central composite, Box-Behnken) add curvature estimation; space-filling designs (Latin hypercube, Sobol sequences, maximin) suit computer experiments where there is no random error.

**When to use it.**
- Physical experiments where each run costs money, time or material, and one-factor-at-a-time testing would be wasteful and uninformative about interactions.
- Process optimization, where the aim is to find settings that maximize yield or minimize defects.
- Building a training set for a surrogate model of an expensive simulator.
- Screening many candidate factors down to the vital few (Plackett-Burman designs).
- Robust design, where the goal is settings insensitive to uncontrollable noise factors (Taguchi methods).

**When not to use it.** When data are observational and inputs cannot be set; when the input space has hard constraints that standard designs violate (use optimal or constrained designs); when sequential adaptive sampling would be more efficient (use Bayesian optimization).

**Assumptions and failure modes.** Fractional designs alias interactions with main effects, and the alias structure must be understood before interpreting results. Randomization of run order protects against drift; blocking handles known nuisance variation. Extrapolation outside the design region is unsupported. Physical experiments need replication for error estimation, while deterministic computer experiments do not.

**Where it is used.** Manufacturing and Six Sigma process improvement, chemical and pharmaceutical formulation, agriculture, materials science, clinical trial design, A/B and multivariate web testing, simulation experiment planning.

---

### 11.8 Surrogate / Emulator Modelling

**What it does.** Trains a fast approximation of an expensive model on a modest number of carefully chosen runs, then uses the surrogate wherever many evaluations are required. Gaussian processes are the default for smooth, low-dimensional problems and provide uncertainty estimates; polynomial chaos expansions connect naturally to variance decomposition; neural networks and gradient boosting handle higher dimensions and larger training sets.

**When to use it.**
- The underlying model takes minutes to days per run, making Monte Carlo, global sensitivity analysis or optimization directly infeasible.
- Real-time or interactive applications where a heavyweight simulator cannot sit behind a user interface.
- Uncertainty propagation requiring thousands of evaluations.
- Design optimization loops and Bayesian optimization, where the surrogate guides where to run the real model next.
- Digital twins, where a fast approximation must run alongside live operations.

**When not to use it.** When the real model is already cheap; when the input dimension is very high and no low-dimensional structure exists; when the region of interest is the extreme tail, which the surrogate's training design is least likely to cover well.

**Assumptions and failure modes.** The surrogate is only valid where it was trained, and extrapolation is unreliable. Validation on held-out real model runs is mandatory, and surrogate error must be reported alongside downstream results rather than silently ignored. Discontinuities and sharp thresholds are poorly captured by smooth surrogates. Adaptive or sequential designs that add runs where surrogate uncertainty is highest are far more efficient than fixed one-shot designs.

**Where it is used.** Aerospace and automotive design, nuclear safety analysis, reservoir simulation in oil and gas, climate model emulation, drug discovery and molecular property prediction, engineering digital twins, hyperparameter optimization.

---

### 11.9 Elasticity Analysis

**What it does.** Expresses sensitivity as the percentage change in output per one percent change in an input, making it unit-free and directly comparable across inputs measured on entirely different scales. In a log-log regression the coefficients are elasticities by construction.

**When to use it.**
- Pricing decisions, where price elasticity of demand determines whether a price rise increases or destroys revenue.
- Comparing the leverage of drivers with incompatible units (headcount, price, conversion rate, spend).
- Economic and policy analysis, where elasticity is the standard currency of discussion and benchmarks exist for comparison.
- Marketing mix and media response modelling, where diminishing returns are naturally expressed as elasticities below one.
- Demographic and population models, where elasticity of growth rate to vital rates guides conservation priorities.

**When not to use it.** When inputs can be zero or negative, where percentage change is undefined or misleading; when the relationship is far from log-linear over the range of interest, since elasticity varies by point and a single number misrepresents it; for large changes, since elasticity is a local concept.

**Assumptions and failure modes.** Elasticity is generally point-specific, so quoting "the" elasticity implies a constant-elasticity functional form that may not hold. Short-run and long-run elasticities differ substantially and are frequently conflated. Estimating price elasticity from observational data is a causal problem plagued by simultaneity, and naive regression of quantity on price is biased, which is a classic motivation for instrumental variables.

**Where it is used.** Pricing and revenue management, energy demand policy, tax policy analysis, marketing mix modelling, transport demand forecasting, conservation biology (matrix population models).

---

## 12. Simulation Methods

### 12.1 Monte Carlo Simulation

**What it does.** Estimates properties of a system by drawing random samples from input distributions, running the model on each draw, and summarizing the resulting distribution of outputs. Convergence is governed by the square root of the number of samples and, critically, does not degrade with the number of input dimensions, which is why it dominates high-dimensional integration.

**When to use it.**
- Analytical solutions are intractable due to nonlinearity, path dependency, thresholds or interacting uncertainties.
- The full output distribution is needed, not just a mean: probability of breaching a limit, distribution of project completion dates, tail losses.
- Many uncertain inputs interact, where scenario analysis would need an unmanageable number of cases.
- Pricing derivatives with path-dependent or multi-asset payoffs.
- Estimating operational metrics of complex systems where closed-form queueing results do not apply.

**When not to use it.** When a closed-form or numerical-quadrature solution exists and is far cheaper; when the target is a very rare event, since plain Monte Carlo wastes almost all samples away from the region of interest (use importance sampling, subset simulation or splitting); when the model is too slow to run enough times (use a surrogate or QMC).

**Assumptions and failure modes.** Results are only as good as the input distributions and their correlation structure, and independence is assumed far more often than it is true. Convergence must be checked, and Monte Carlo standard error reported, rather than trusting a fixed round number of runs. Random seeds should be recorded for reproducibility. Garbage in, confidently distributed garbage out remains the dominant real-world failure.

**Where it is used.** Derivative pricing and risk management, insurance capital modelling, project cost and schedule risk (integrated cost-schedule risk analysis), nuclear and radiation transport, particle physics, drug development portfolio modelling, reliability engineering, statistical bootstrapping.

---

### 12.2 Quasi-Monte Carlo (QMC)

**What it does.** Replaces pseudo-random draws with deterministic low-discrepancy sequences (Sobol, Halton, Faure, lattice rules) that fill the input space far more evenly. For smooth integrands of moderate effective dimension, convergence approaches order one over n rather than one over the square root of n, which can mean an order of magnitude fewer runs for the same accuracy.

**When to use it.**
- Expensive models where each run costs real money and sample efficiency matters directly.
- Smooth, moderately dimensional integration problems, especially where a few inputs dominate.
- Global sensitivity analysis, where Sobol sequences are the standard sampling basis.
- Financial pricing of high-dimensional but effectively low-dimensional payoffs, where QMC has been a large practical success.
- Any Monte Carlo workflow where the estimator is a smooth expectation rather than a tail probability.

**When not to use it.** Discontinuous or kinked integrands, where the advantage largely evaporates; very high effective dimension without structure; when an unbiased error estimate is essential, unless randomized QMC (scrambled Sobol) is used.

**Assumptions and failure modes.** Plain QMC gives no natural error estimate, since it is deterministic; randomization via scrambling or shifting restores one and typically improves accuracy. Dimension ordering matters, because low-discrepancy properties are strongest in the leading dimensions, so the most influential inputs should be assigned there. Skipping the first points of a Sobol sequence is standard practice. Mixing QMC points with non-uniform transformations requires care to preserve the equidistribution property.

**Where it is used.** Quantitative finance (option pricing, XVA), global sensitivity analysis packages, computer graphics rendering, uncertainty propagation in engineering, Bayesian computation research.

---

### 12.3 Latin Hypercube Sampling (LHS)

**What it does.** Divides each input's distribution into n equal-probability strata and draws exactly one value from each stratum, then randomly pairs the values across inputs. Every part of every marginal distribution is represented, eliminating the clustering and gaps that plague small random samples, at essentially the same cost as simple random sampling.

**When to use it.**
- Modest run budgets where simple random sampling would leave visible gaps in the input space.
- Uncertainty propagation for expensive simulators, as a drop-in improvement over plain Monte Carlo.
- Building training designs for surrogate models, where marginal coverage matters.
- Screening studies and preliminary sensitivity analysis before committing to a full variance-based approach.
- Situations requiring a specified rank correlation structure between inputs, which the Iman-Conover method imposes on an LHS design.

**When not to use it.** Rare event estimation, since stratifying marginals does not concentrate effort in the joint tail; when the number of runs must be extendable, because classical LHS is not naturally nested (use orthogonal-array or sliced LHS variants); when full-space uniformity matters more than marginal coverage, where maximin or Sobol designs are better.

**Assumptions and failure modes.** Guarantees good marginal coverage but not good joint coverage; a naive LHS can still leave large empty regions in two-dimensional projections, which optimized maximin or orthogonal LHS designs address. Variance reduction is strong when the model response is dominated by additive main effects and weak when interactions dominate. Sample size must be fixed in advance.

**Where it is used.** Nuclear and environmental safety assessment, reservoir and subsurface modelling, climate and hydrological model ensembles, engineering reliability studies, computer experiment design, probabilistic risk assessment.

---

### 12.4 Markov Chain Monte Carlo (MCMC)

**What it does.** Generates samples from a distribution known only up to a normalizing constant by constructing a Markov chain whose stationary distribution is the target. After a burn-in period the chain's states serve as correlated draws from the posterior. Metropolis-Hastings is the general recipe, Gibbs sampling exploits conditional conjugacy, and Hamiltonian Monte Carlo with the No-U-Turn Sampler uses gradient information to traverse high-dimensional posteriors efficiently.

**When to use it.**
- Bayesian inference for any model where the posterior lacks a closed form, which is nearly all interesting models.
- Hierarchical and multilevel models with many latent parameters.
- Models with complex likelihoods, latent variables, censoring or missing data.
- When accurate posterior tails and correlations matter, where faster approximations such as variational inference are known to be over-confident.
- Sampling from physical Boltzmann distributions in statistical mechanics.

**When not to use it.** Very large datasets with tight latency budgets, where variational inference, Laplace approximation or stochastic gradient methods trade accuracy for speed; strongly multimodal posteriors, where chains get trapped (consider parallel tempering or nested sampling); models where the likelihood cannot be evaluated at all (use approximate Bayesian computation or simulation-based inference).

**Assumptions and failure modes.** Convergence cannot be proven from output, only failure detected. Run multiple chains from dispersed starts, check R-hat below 1.01, monitor effective sample size, and inspect trace plots and divergent transitions in HMC. Autocorrelation means nominal sample counts overstate information. Poor parameterization causes funnel geometries that defeat samplers; non-centred reparameterization usually fixes them. Label switching affects mixture models.

**Where it is used.** Bayesian statistics across every applied field, phylogenetics (BEAST, MrBayes), astrophysics and cosmology parameter estimation, pharmacokinetic and pharmacodynamic modelling, epidemiological transmission models, ecology, psychometrics, sports and election forecasting models.

---

### 12.5 Bootstrap

**What it does.** Approximates the sampling distribution of a statistic by resampling the observed data with replacement many times and recomputing the statistic on each resample. The empirical distribution stands in for the unknown population, delivering standard errors, confidence intervals and bias estimates for statistics with no tractable analytic formula.

**When to use it.**
- Statistics with no standard error formula: medians, ratios, correlation differences, Gini coefficients, custom business metrics, complex pipeline outputs.
- Small or moderately sized samples where asymptotic normality is doubtful.
- Validating machine learning model performance variability and constructing intervals around metrics such as AUC.
- Estimating uncertainty in multi-step estimation procedures, by bootstrapping the entire procedure end to end.
- Bagging and random forests, which are direct applications of the resampling idea to prediction.

**When not to use it.** Extreme quantiles and maxima, where the bootstrap is known to fail because resampling cannot produce values beyond the observed range; heavy-tailed distributions with infinite variance; very small samples, where the empirical distribution is too crude; dependent data unless the scheme respects the dependence (block bootstrap for time series, cluster bootstrap for grouped data).

**Assumptions and failure modes.** Observations must be exchangeable, or a structure-aware variant must be used. Percentile intervals are biased for skewed statistics; BCa or bootstrap-t intervals correct this. The number of resamples affects only Monte Carlo error, not the underlying validity, and one to two thousand is typically sufficient for intervals. Bootstrapping after model selection on the same data does not repair selection bias unless the selection step is included inside each resample.

**Where it is used.** Econometrics and applied statistics generally, machine learning model evaluation, clinical biostatistics, survey analysis, ecology (diversity indices), finance (risk metric uncertainty).

---

### 12.6 Discrete Event Simulation (DES)

**What it does.** Represents a system as entities flowing through a network of activities and queues, where state changes occur only at discrete event times generated by stochastic distributions. The simulation clock jumps from event to event rather than advancing in fixed steps, making it efficient for systems dominated by waiting, resource contention and routing logic.

**When to use it.**
- Queueing and capacity problems too complex for analytic queueing theory: multiple resource types, priority rules, balking, reneging, shift patterns, breakdowns.
- Hospital patient flow, emergency department capacity, operating theatre scheduling and bed planning.
- Manufacturing line design, buffer sizing, bottleneck analysis and throughput estimation.
- Supply chain and logistics network design, including warehouse operations and transport scheduling.
- Call centre and service desk staffing, where Erlang formulas break down under realistic assumptions.
- Testing operational policies before implementing them, where experimentation on the live system is impossible or unsafe.

**When not to use it.** Systems better described by continuous flows or differential equations (use system dynamics); problems where a closed-form queueing result applies; when the required behavioural detail (how people actually queue, prioritize and improvise) cannot be credibly specified.

**Assumptions and failure modes.** Input distributions for arrivals and service times must be fitted from real data, and the exponential default is often badly wrong. Warm-up periods must be discarded before collecting statistics for steady-state analysis, and terminating versus non-terminating simulations require different treatment. Multiple independent replications are needed to quantify output uncertainty; a single run tells you nothing. Verification (is the model built right) and validation (is it the right model) are distinct and both mandatory. Models drift toward excessive detail, which increases build time and data requirements without improving decisions.

**Where it is used.** Healthcare operations and capacity planning, manufacturing and industrial engineering, logistics and port operations, airport and transport terminal design, contact centre workforce management, military logistics, semiconductor fab scheduling.

---

## 13. Markov and State Space Models

### 13.1 Markov Chain

**What it does.** Models a system that moves between a finite set of states, where the probability of the next state depends only on the current state and not on the path taken to reach it. The transition matrix fully characterizes the dynamics, and its powers give multi-step transition probabilities. Long-run behaviour is captured by the stationary distribution, absorption probabilities and expected hitting times.

**When to use it.**
- Systems with natural discrete states and transitions: customer segments, credit ratings, machine condition, disease stages, web page navigation.
- Forecasting long-run equilibrium shares, such as the steady-state distribution of a customer base across tiers.
- Computing expected time to absorption: time to default, time to failure, expected lifetime value under churn.
- Simple, transparent modelling where a full state space model would be unnecessary and harder to defend.

**When not to use it.** When history clearly matters beyond the current state (use higher-order or semi-Markov models); when the state is not directly observed (use an HMM); when transitions depend on external drivers (use covariate-dependent transition models).

**Assumptions and failure modes.** The memoryless property is the critical assumption and is frequently violated in practice, most obviously in dwell times, which are forced to be geometric. Time-homogeneity assumes the transition matrix does not change, which fails across economic cycles or product lifecycles. Transition probabilities estimated from sparse cells are unstable and should be smoothed or pooled. Chains must be irreducible and aperiodic for a unique stationary distribution to exist.

**Where it is used.** Credit rating migration and IFRS 9 provisioning, customer lifecycle and churn modelling, marketing attribution (removal effect on transition graphs), disease progression modelling, reliability and maintenance planning, queueing theory, PageRank and web navigation, natural language n-gram models.

---

### 13.2 Hidden Markov Model (HMM)

**What it does.** Assumes an unobserved Markov chain generates observable emissions, so the analyst sees only noisy or indirect evidence of the underlying state. Three canonical problems have exact solutions: evaluating the likelihood of an observation sequence (forward algorithm), inferring the most likely state path (Viterbi), and learning parameters from unlabelled sequences (Baum-Welch).

**When to use it.**
- The system has meaningful discrete regimes that are not directly measurable: market regimes, machine health states, sleep stages, engagement modes.
- Sequential data where observations are conditionally independent given a latent state, and temporal structure lives in the state transitions.
- Segmentation and labelling of sequences without labelled training data, since Baum-Welch is unsupervised.
- Anomaly detection through low sequence likelihood under a model trained on normal behaviour.
- Situations needing interpretable latent states, which recurrent neural networks do not provide.

**When not to use it.** When dependencies span long ranges beyond what a first-order latent chain can carry; when the number of states cannot be justified and results are highly sensitive to it; when dwell times are strongly non-geometric (use hidden semi-Markov models); when abundant labels exist and a discriminative sequence model (CRF) would perform better.

**Assumptions and failure modes.** First-order Markov transitions, conditional independence of emissions given states, and a fixed number of states chosen by BIC, cross-validation or domain reasoning. Baum-Welch converges only to local optima, so multiple random restarts are essential. Learned states are statistical constructs and need domain interpretation before being labelled with meaningful names. Label switching makes runs non-comparable without post-processing.

**Where it is used.** Speech and handwriting recognition (historically dominant), bioinformatics (gene finding, sequence alignment profiles), financial regime switching models, condition monitoring and predictive maintenance, sleep staging and EEG analysis, human activity recognition from wearables, natural language part-of-speech tagging.

---

### 13.3 Higher-Order Markov Models

**What it does.** Conditions the next state on the previous k states rather than only the current one, allowing short-range history to influence transitions. This captures patterns such as "the sequence A then B behaves differently from B then A" that a first-order chain cannot represent.

**When to use it.**
- Clear evidence that recent history beyond the current state carries predictive information, testable by comparing model orders with AIC or BIC.
- Sequence modelling where local context matters: text n-grams, clickstream paths, DNA motifs, musical structure.
- Diagnostic checking of a first-order assumption before committing to it.

**When not to use it.** When data are limited, because the parameter count grows exponentially with order and states (a k-th order chain over m states has m^k rows in its transition matrix); when long-range dependence is the real phenomenon, which no fixed finite order captures efficiently.

**Assumptions and failure modes.** Severe data sparsity is the dominant issue: most higher-order state combinations are never observed, forcing smoothing (Kneser-Ney, backoff) or parsimonious parameterizations such as the mixture transition distribution model. Any k-th order chain can be rewritten as a first-order chain on an expanded state space, which is theoretically tidy but explodes computationally. Order selection is easy to overfit.

**Where it is used.** Language modelling before neural methods, DNA and protein sequence analysis, web clickstream and journey analysis, music generation, hydrology (multi-day weather state persistence), consumer purchase sequence modelling.

---

### 13.4 Continuous-Time Markov Chain (CTMC)

**What it does.** Allows transitions to occur at any moment rather than at fixed steps, with the time spent in each state exponentially distributed and governed by a generator matrix of transition rates. Transition probabilities over any interval follow from the matrix exponential of the generator.

**When to use it.**
- Events happen in continuous time and forcing them onto a discrete grid would distort the model: equipment failures, arrivals, biochemical reactions, disease progression.
- Irregularly observed panel data, such as clinical visits at uneven intervals, where a discrete chain cannot handle the varying gaps.
- Reliability and availability modelling of repairable systems with failure and repair rates.
- Queueing systems (M/M/1, M/M/c and their variants), which are CTMCs by construction.
- Stochastic chemical kinetics, where the Gillespie algorithm simulates CTMC trajectories exactly.

**When not to use it.** When dwell times are clearly non-exponential, such as ageing components with increasing hazard (use semi-Markov or phase-type distributions); when rates vary strongly with time and inhomogeneous methods are needed.

**Assumptions and failure modes.** Exponential holding times imply constant hazard and complete memorylessness within a state, which contradicts wear-out and duration dependence in many real systems. Rate estimation requires either continuous observation or careful likelihood construction from panel data. Matrix exponentials are numerically delicate for large or stiff generators. Phase-type distributions, built from several exponential sub-states, are the standard workaround for non-exponential dwell times while retaining Markov tractability.

**Where it is used.** Reliability and availability engineering, queueing and telecommunications, systems biology and stochastic chemical kinetics, multi-state disease progression models in epidemiology, credit rating transition modelling in continuous time, performance modelling of computer systems.

---

### 13.5 Semi-Markov Process

**What it does.** Keeps the Markov structure for which state comes next but frees the time spent in each state to follow any distribution, breaking the exponential or geometric straitjacket. The embedded jump chain governs transitions while a separate sojourn distribution governs timing.

**When to use it.**
- Dwell times are the phenomenon of interest and are clearly non-exponential: hospital length of stay, machine time-to-failure with wear-out, unemployment duration, customer tenure.
- Hazard rates change with time in state, for example a defect risk that rises with equipment age or a churn risk that spikes near contract renewal.
- Reliability modelling where Weibull or lognormal lifetimes are standard and an exponential assumption would be indefensible.
- Hidden semi-Markov extensions where both the state and its duration are latent, common in speech and activity segmentation.

**When not to use it.** When exponential dwell times are adequate and CTMC machinery is far simpler; when data contain too few transitions to estimate a full sojourn distribution per state; when heavy censoring makes duration estimation unreliable without careful survival methods.

**Assumptions and failure modes.** Analytical results are far less available than for CTMCs, so simulation is often required. Estimation demands enough observed sojourns per state, which is restrictive with many states. Censored durations must be handled with survival techniques or dwell times will be systematically underestimated. Choice of sojourn family (Weibull, gamma, lognormal, phase-type) materially affects long-horizon predictions.

**Where it is used.** Reliability and maintenance optimization, healthcare length-of-stay and multi-state clinical models, hidden semi-Markov speech and gesture segmentation, insurance disability and long-term care modelling, seismology (earthquake recurrence), manpower and workforce planning.

---

### 13.6 Markov Decision Process (MDP)

**What it does.** Adds actions and rewards to a Markov chain: in each state an agent chooses an action, which determines both the reward received and the distribution over the next state. Solving an MDP means finding a policy that maximizes expected cumulative discounted reward, via value iteration, policy iteration or linear programming when the model is known, or reinforcement learning when it is not.

**When to use it.**
- Sequential decision problems where actions taken now change the future situation, not just the immediate payoff.
- Inventory and replenishment policy, maintenance scheduling, and equipment replacement decisions.
- Dynamic pricing and revenue management, where selling now forecloses selling later.
- Treatment sequencing and dynamic treatment regimes in medicine.
- Resource allocation over time under uncertainty, including portfolio rebalancing and capacity dispatch.

**When not to use it.** Single-shot decisions with no state dynamics, where decision analysis suffices; problems where the state is unobservable (use a POMDP); very large or continuous state spaces without function approximation, since exact methods suffer the curse of dimensionality; settings where the reward function cannot be honestly specified, which is where most real deployments fail.

**Assumptions and failure modes.** Markov state, known or learnable transition dynamics, and a scalar reward that genuinely encodes the objective. Reward misspecification produces policies that optimize the letter of the metric while destroying the intent. The discount factor is a modelling choice with large behavioural consequences. Offline evaluation of a learned policy against logged data is subtle and requires off-policy evaluation methods. Exploration in live systems carries real costs and risks.

**Where it is used.** Inventory and supply chain optimization, predictive maintenance and asset replacement, dynamic pricing in airlines, hotels and ride-hailing, recommendation sequencing, robotics and autonomous control, clinical dynamic treatment regimes, energy grid dispatch and storage arbitrage.

---

### 13.7 Partially Observable MDP (POMDP)

**What it does.** Extends the MDP to settings where the agent cannot see the true state and instead maintains a belief distribution over states, updated by Bayes' rule as observations arrive. The policy maps beliefs, not states, to actions, which means it naturally values information-gathering actions that improve the belief without immediate reward.

**When to use it.**
- The state is fundamentally hidden and must be inferred: patient's true underlying condition, machine's internal degradation, customer's latent intent, robot's position from noisy sensors.
- Actions include diagnostic tests or inspections whose value lies purely in reducing uncertainty, and the model must trade their cost against their informational benefit.
- Active sensing and exploration problems where where to look is itself a decision.
- Maintenance under imperfect inspection, where the decision is jointly about inspecting and acting.

**When not to use it.** When the state is effectively observable, since MDP methods are vastly cheaper; when the problem size makes even approximate solvers impractical without heavy structure; when a simpler heuristic (inspect on a fixed schedule) captures most of the value at a fraction of the complexity.

**Assumptions and failure modes.** Exact solution is PSPACE-hard and infeasible beyond tiny problems, so practice relies on point-based approximations (SARSOP, PBVI), Monte Carlo tree search (POMCP), or deep recurrent policies. The belief update requires a correct observation model, and a misspecified sensor model corrupts everything downstream. Belief state representation is the central engineering challenge. Results are hard to explain to stakeholders, which limits adoption in regulated settings.

**Where it is used.** Robotics navigation and SLAM, autonomous vehicle decision-making under occlusion, medical diagnosis and screening interval optimization, structural inspection and maintenance planning, spoken dialogue systems, target tracking and surveillance, conservation decision-making under monitoring uncertainty.

---

### 13.8 Input-Output HMM

**What it does.** Conditions both the state transitions and the emissions on an exogenous input sequence, so the latent dynamics are driven by observable context rather than evolving autonomously. It effectively turns an HMM into a conditional sequence model with latent structure.

**When to use it.**
- External drivers plausibly cause regime changes: promotions driving demand regimes, weather driving grid states, interventions driving patient states.
- Sequence-to-sequence mapping where the alignment between input and output is latent, such as gesture or speech with contextual conditioning.
- Counterfactual reasoning about what the state trajectory would have been under a different input sequence.
- When a plain HMM's transitions look unstable and the instability is suspected to be explained by measurable context.

**When not to use it.** When no meaningful exogenous input exists; when data are limited, since conditioning transitions on covariates multiplies the parameter count; when a discriminative model or a neural sequence model would serve better and interpretability is not required.

**Assumptions and failure modes.** Requires enough observations across the range of input values to estimate input-dependent transitions, which is demanding. Typically parameterized with softmax or logistic transition functions, which are prone to overfitting without regularization. Inputs must be genuinely exogenous, or the causal reading of the conditioning is invalid. Training is still EM-based with local optima.

**Where it is used.** Speech and gesture recognition with contextual conditioning, demand modelling under marketing and pricing actions, energy load state modelling driven by weather, clinical state modelling with treatment inputs, control-oriented time series segmentation.

---

### 13.9 Hierarchical HMM (HHMM)

**What it does.** Nests HMMs inside states, so a high-level state expands into a sub-HMM with its own states, which may themselves expand further. This represents processes with structure at multiple timescales in a single coherent generative model, with the hierarchy expressing compositional structure explicitly.

**When to use it.**
- Naturally nested processes: acoustic frames within phonemes within words, strokes within characters within words, sub-activities within tasks within routines.
- Behaviour modelling where coarse regimes contain finer sub-regimes with distinct dynamics.
- When a flat HMM either needs an implausible number of states or blurs fast and slow dynamics together.
- Cases where the hierarchy itself is the interpretable output, such as segmenting a workflow into tasks and steps.

**When not to use it.** When the hierarchy is not genuinely present and is imposed for its own sake; when data are insufficient for the greatly increased parameter count; when a flat HMM with well-chosen states performs comparably, which is common.

**Assumptions and failure modes.** The hierarchical structure (depth, branching) usually must be specified in advance and is difficult to learn from data. Inference is more expensive, though an HHMM can be flattened into an equivalent dynamic Bayesian network for cubic-time inference. Local optima and identifiability problems compound with depth. Interpretation requires domain validation at every level.

**Where it is used.** Speech recognition architectures, handwriting and gesture recognition, human activity and workflow recognition from sensors, video event segmentation, robot task learning, music structure analysis.

---

### 13.10 Factorial HMM

**What it does.** Runs several independent hidden Markov chains in parallel, with each observation depending jointly on the current states of all chains. This represents a combinatorial state space compactly: five binary chains express thirty-two configurations using five sets of transition parameters rather than a thirty-two by thirty-two transition matrix.

**When to use it.**
- Multiple independent processes combine to produce a single observed signal, most canonically energy disaggregation, where total household power is the sum of individual appliance states.
- Source separation problems where several latent sources evolve independently but are observed only in mixture.
- Multi-factor state modelling, such as separate chains for market regime, seasonality regime and idiosyncratic regime.
- When a flat HMM would require an unmanageable number of states to express independent concurrent factors.

**When not to use it.** When the underlying factors are strongly dependent, violating the independence structure that gives the model its parsimony; when exact inference is required at scale, since coupling through the observation makes exact inference intractable; when a single dominant factor explains the signal.

**Assumptions and failure modes.** Chains are a priori independent, but the shared observation couples them in the posterior, so exact inference is exponential in the number of chains and approximations (structured variational inference, Gibbs sampling, beam search) are mandatory. Identifiability is a persistent problem: chains can swap roles or absorb one another's dynamics. Additive observation models are common and constrain applicability.

**Where it is used.** Non-intrusive load monitoring and energy disaggregation, audio source separation, multi-target tracking, bioinformatics (multiple concurrent regulatory processes), financial multi-factor regime modelling, sensor fusion.

---

### 13.11 Viterbi Algorithm

**What it does.** Uses dynamic programming to find the single most probable sequence of hidden states given an observation sequence, in time linear in sequence length and quadratic in the number of states. It maximizes the joint probability of the whole path rather than choosing the most likely state at each time independently.

**When to use it.**
- A single coherent decoded sequence is the deliverable: a transcript, a segmentation, a labelled path, a decoded message.
- Downstream consumers need one discrete answer rather than a distribution over states.
- Error-correcting decoding of convolutional codes, where it is the standard optimal decoder.
- Segmenting a time series into regimes where the segmentation must be internally consistent over time.

**When not to use it.** When per-timepoint state probabilities are what matters, since the most likely path may assign a state at time t that is not the marginally most likely state there (use forward-backward posterior decoding instead); when uncertainty over paths should be preserved, in which case sampling paths or the N-best list is preferable.

**Assumptions and failure modes.** Requires known model parameters; it decodes, it does not learn. Numerical underflow over long sequences is avoided by working in log space, which is standard. It returns one path with no measure of how much better it is than the runner-up, which can be misleading when several paths are nearly equally likely. Ties and near-ties should be inspected, not silently resolved.

**Where it is used.** Speech recognition decoding, digital communications (convolutional and turbo code decoding, GSM, satellite links), part-of-speech tagging and named entity recognition, bioinformatics gene structure prediction, GPS map-matching of noisy location traces, regime segmentation in finance.

---

### 13.12 Baum-Welch Algorithm

**What it does.** Applies Expectation-Maximization to HMMs, alternating between computing expected state occupancies and transitions given current parameters (forward-backward E-step) and re-estimating transition, emission and initial probabilities from those expectations (M-step). It learns the entire model from unlabelled observation sequences, with the likelihood guaranteed to be non-decreasing at every iteration.

**When to use it.**
- No labelled state sequences exist, which is the normal situation when the states are conceptual regimes rather than recorded facts.
- Fitting HMMs and hidden semi-Markov models to raw sequential data for segmentation or anomaly detection.
- Adapting a pre-trained model to a new domain or speaker with unlabelled data.
- Multiple independent sequences are available and can be pooled into a single parameter estimate.

**When not to use it.** When labelled state sequences exist, since direct maximum likelihood counting is faster and better; when the objective is discriminative classification rather than generative fit (use CRFs or discriminative training); when the number of states is unknown and results vary wildly across choices, which needs model selection first.

**Assumptions and failure modes.** Converges to a local optimum only, so multiple random initializations and retention of the best likelihood are essential practice; informed initialization from clustering usually helps substantially. Sensitive to the assumed number of states. Emission distributions can collapse onto single points, so variance floors or priors (MAP-EM) are needed. Underflow requires scaling or log-space implementation. Convergence of likelihood does not imply the learned states mean anything.

**Where it is used.** Speech recognition training pipelines, bioinformatics profile HMMs (Pfam, HMMER), unsupervised regime detection in finance, condition monitoring model fitting, activity recognition from unlabelled sensor streams.

---

### 13.13 Kalman Filter

**What it does.** Provides exact recursive Bayesian inference for linear systems with Gaussian noise, alternating between a prediction step that propagates the state estimate forward through the dynamics and an update step that corrects it using the newest measurement, weighted by the Kalman gain according to relative uncertainty. The result is the optimal estimator in the minimum mean squared error sense, computed in constant time per observation with no need to store history.

**When to use it.**
- Real-time tracking and state estimation from noisy sensors: position, velocity, orientation, temperature, level.
- Sensor fusion, combining measurements of different accuracy and frequency into a single coherent estimate.
- Time series decomposition into level, trend and seasonal components via structural time series and state space formulations.
- Handling missing observations naturally, since the prediction step simply runs without an update.
- Smoothing historical estimates with the benefit of later data (Rauch-Tung-Striebel smoother).
- Any embedded or streaming context where storing and reprocessing all past data is impossible.

**When not to use it.** Strongly nonlinear dynamics or measurement models (use the Extended or Unscented Kalman Filter, or a particle filter); non-Gaussian or multimodal state distributions, where a single mean and covariance cannot represent the belief; systems with hard constraints or discrete modes, which need constrained or hybrid variants.

**Assumptions and failure modes.** Linear dynamics, additive Gaussian process and measurement noise, and known noise covariances. The covariances Q and R are the main tuning knobs and are rarely known: too small a Q makes the filter over-confident and slow to react, too large makes it noisy. Filter divergence occurs when the model is wrong and the estimated covariance shrinks anyway, so innovation monitoring and consistency tests (NIS, NEES) are important. Numerical issues in the covariance update are addressed with square-root or Joseph-form implementations. Outliers are not handled robustly without explicit gating.

**Where it is used.** Aerospace navigation and guidance (its Apollo origins), GPS and INS integration, robotics localization, autonomous vehicle tracking, econometric structural time series and nowcasting, signal processing, battery state-of-charge estimation, process control, weather data assimilation.

---

### 13.14 Particle Filter (Sequential Monte Carlo)

**What it does.** Represents the state distribution by a weighted set of samples that are propagated through the dynamics, reweighted by the likelihood of each new observation, and periodically resampled to concentrate effort on high-probability regions. Because the belief is a sample cloud rather than a Gaussian, it handles arbitrary nonlinearity, multimodality and non-Gaussian noise.

**When to use it.**
- Nonlinear, non-Gaussian state estimation where Kalman variants break down.
- Multimodal beliefs, such as a robot uncertain between several plausible rooms, which a Gaussian cannot express.
- Tracking with clutter, occlusion or data association ambiguity.
- Bearing-only tracking, terrain-referenced navigation and other strongly nonlinear observation geometries.
- Online Bayesian inference for state space models in econometrics and epidemiology, including parameter learning via particle MCMC.

**When not to use it.** High-dimensional state spaces, where the required particle count grows prohibitively and weight degeneracy is severe (consider ensemble Kalman methods); when a Kalman filter suffices, since it is orders of magnitude cheaper; hard real-time systems with tight compute budgets.

**Assumptions and failure modes.** Weight degeneracy, where nearly all weight concentrates on one particle, is the central problem, monitored by effective sample size and mitigated by resampling. Resampling in turn causes sample impoverishment, where diversity collapses; roughening, regularization or resample-move steps counter this. Performance depends heavily on the proposal distribution, and using the dynamics as the proposal is simple but inefficient when observations are highly informative. Results are stochastic, so runs differ; fixing seeds matters for reproducibility. Computational cost scales with particle count, which must be tuned.

**Where it is used.** Robotics localization and SLAM, target tracking and defence surveillance, autonomous navigation, epidemiological state estimation during outbreaks, financial stochastic volatility models, object tracking in computer vision, systems biology inference.

---

## 14. Forecastability and Forecast Evaluation

### 14.1 Sample Entropy / Approximate Entropy

**What it does.** Measures how unpredictable a time series is by asking, for patterns of a given length that match within a tolerance, how often the match survives when the pattern is extended by one point. Low values mean the series repeats itself in recognizable ways and is therefore structured; high values mean it behaves like noise. Sample entropy corrects approximate entropy's self-match bias and is less dependent on record length.

**When to use it.**
- Screening many series before investing in forecasting models, to separate genuinely predictable series from noise.
- Setting realistic accuracy expectations with stakeholders before a project starts, which prevents chasing unattainable error targets.
- Physiological signal analysis, where regularity itself is the clinical signal (reduced heart rate variability complexity is a recognized risk marker).
- Detecting change in system complexity over time, such as machine degradation increasing or decreasing signal regularity.
- Comparing complexity across series or across conditions, provided parameters are held fixed.

**When not to use it.** As a forecastability measure for series dominated by trend or seasonality, which should be removed first or entropy will reflect them rather than genuine irregularity; with very short series, where estimates are unstable; for cross-study comparison without identical parameter settings.

**Assumptions and failure modes.** Results depend on the embedding dimension m, tolerance r (commonly 0.1 to 0.25 standard deviations) and series length N, and are not comparable across different settings. Sensitive to non-stationarity and to outliers, which inflate the standard deviation used to set r. Multiscale entropy extends the idea across timescales and is often more informative than a single value.

**Where it is used.** Cardiology and heart rate variability analysis, EEG and anaesthesia depth monitoring, machine condition monitoring, financial series complexity studies, demand forecastability screening in supply chain.

---

### 14.2 Hurst Exponent

**What it does.** Quantifies long-range dependence by measuring how the range or variance of a series scales with the length of the window examined. Values above 0.5 indicate persistence, where increases tend to be followed by increases and trends extend further than a random walk would suggest. Values below 0.5 indicate anti-persistence and mean reversion. Exactly 0.5 corresponds to a random walk with no exploitable memory.

**When to use it.**
- Assessing whether a series carries exploitable memory before designing a trend-following or mean-reverting strategy.
- Distinguishing genuine long memory from short-memory autocorrelation in hydrology, network traffic and financial series.
- Characterizing roughness and self-similarity of natural signals and surfaces.
- Justifying model class selection, for example ARFIMA for fractional integration versus standard ARIMA.

**When not to use it.** On short series, where estimates are unreliable; without first removing trend and seasonality, since both bias the estimate strongly upward; as a standalone trading signal, since estimates are noisy and unstable over time.

**Assumptions and failure modes.** Assumes self-similarity and (usually) stationarity. Estimation method materially affects the answer: classical rescaled range is biased in small samples, and detrended fluctuation analysis, periodogram regression or wavelet estimators are generally preferred. Structural breaks and slowly varying trends masquerade as long memory, a well-documented confusion. Values should be reported with confidence intervals from surrogate data, which practitioners frequently omit.

**Where it is used.** Hydrology (Hurst's original Nile studies), network traffic engineering and self-similar traffic modelling, quantitative finance and market efficiency testing, climatology, geophysics and surface roughness, physiological time series analysis.

---

### 14.3 Spectral Analysis (Power Spectral Density)

**What it does.** Decomposes a series into the contribution of each frequency to its total variance, revealing periodic structure that is often invisible in the time domain. Sharp peaks indicate deterministic cycles and seasonality; broad-band power without peaks suggests noise; the shape of the spectrum at low frequencies characterizes long-memory behaviour.

**When to use it.**
- Identifying the periods present in a series before specifying seasonal terms in a forecasting model, especially with multiple or non-obvious seasonalities (daily plus weekly plus annual).
- Confirming that apparent cycles are real rather than artefacts of visual inspection.
- Assessing forecastability: a spectrum concentrated in a few frequencies is highly exploitable, a flat spectrum is not.
- Detecting machine faults, where characteristic defect frequencies appear as spectral peaks.
- Separating signal from noise prior to filtering.

**When not to use it.** Strongly non-stationary series whose frequency content changes over time (use STFT or wavelets); very short series, where frequency resolution is inadequate; series dominated by a strong trend, which leaks power across all frequencies and must be removed first.

**Assumptions and failure modes.** Assumes stationarity over the analysis window. Spectral leakage from finite record length blurs peaks and is mitigated by windowing (Hann, Hamming) at the cost of resolution. The raw periodogram is an inconsistent estimator whose variance does not decrease with sample length, which is precisely why Welch averaging and multitaper methods exist. Aliasing folds frequencies above the Nyquist limit into the visible band. Peaks near the data-collection frequency are frequently instrumentation artefacts rather than physical signal.

**Where it is used.** Signal processing generally, vibration analysis and rotating machinery diagnostics, EEG and neuroscience rhythm analysis, climate and oceanography cycle detection, economics business cycle analysis, electricity demand seasonality, astronomy (periodic variable detection).

---

### 14.4 Forecastability Index (Goerg, 2013)

**What it does.** Provides a single normalized measure, Omega, of how much a series can be predicted relative to a white noise benchmark, derived from the entropy of its normalized spectral density. A value of zero means the series is indistinguishable from white noise and no model will beat the mean; values approaching one indicate highly structured, strongly predictable behaviour.

**When to use it.**
- Triaging a large portfolio of series (thousands of SKUs, sensors or accounts) to decide where forecasting investment will actually pay off.
- Setting expectations and defining realistic accuracy targets before modelling begins, differentiating between a bad model and an inherently unforecastable series.
- Segmenting series into tiers with different treatment: statistical models for high-Omega series, simple heuristics or safety stock policies for low-Omega ones.
- Monitoring whether a series is becoming more or less forecastable over time, which can signal regime change.

**When not to use it.** As a substitute for actually testing models, since it measures linear spectral structure and can miss nonlinear predictability; on non-stationary series without preprocessing; as a precise numeric target rather than a relative ranking.

**Assumptions and failure modes.** Rooted in spectral estimation, so it inherits stationarity assumptions and sensitivity to trend and seasonality handling. It measures the presence of exploitable structure in the series' own history and says nothing about predictability from external drivers, so a series with low Omega may still be well forecast by causal covariates. Results depend on the spectral estimator and detrending choices, so preprocessing must be consistent across series for comparisons to be meaningful.

**Where it is used.** Demand planning and supply chain forecast segmentation, retail SKU-level forecast triage, energy load forecastability assessment, sensor and telemetry monitoring prioritization, forecasting research benchmarking.

---

### 14.5 Prediction Intervals (Classical)

**What it does.** Places bounds around a point forecast that should contain the future observation with a stated probability, accounting for both the irreducible noise in the process and, in better implementations, the uncertainty in estimated parameters. Unlike confidence intervals, which cover a parameter, prediction intervals cover a future realization and are therefore always wider.

**When to use it.**
- Any operational forecast that feeds a decision with asymmetric costs: safety stock, capacity buffers, staffing levels, cash reserves.
- Communicating that a forecast is a distribution rather than a number, which materially changes how stakeholders act on it.
- Setting service levels, where the required quantile of the forecast distribution directly determines the buffer.
- Detecting anomalies, where observations falling outside the interval trigger investigation.

**When not to use it.** When the model's distributional assumptions are clearly violated and coverage would be illusory (use empirical, bootstrap or conformal intervals); for long horizons in models whose interval formulas ignore parameter and model uncertainty, which produces intervals that are far too narrow.

**Assumptions and failure modes.** Standard ARIMA and ETS intervals assume normally distributed, homoscedastic errors and, in most software, condition on the estimated parameters as if they were known, so they are systematically too narrow, especially at long horizons and after model selection on the same data. Real-world coverage of nominal 95% intervals is frequently far below 95%, a well-documented finding in forecasting competitions. Intervals also assume the model form remains valid over the horizon, which structural breaks violate. Always validate empirical coverage on held-out data rather than trusting the nominal level.

**Where it is used.** Demand and inventory planning, energy load and price forecasting, financial risk forecasting, capacity and workforce planning, epidemic projection reporting, macroeconomic fan charts.

---

### 14.6 Forecast Reconciliation

**What it does.** Ensures forecasts made at different levels of a hierarchy or grouping structure add up consistently, so that regional forecasts sum to the national forecast and product forecasts sum to the category. Modern optimal methods (MinT, and its shrinkage variants) exploit the error covariance structure to combine information across all levels, which typically improves accuracy at every level rather than merely enforcing coherence.

**When to use it.**
- Any organization forecasting the same quantity at several levels of aggregation, where inconsistent numbers cause planning conflicts and loss of trust.
- Grouped structures with multiple crossing dimensions (product by region by channel), where simple top-down or bottom-up approaches are inadequate.
- Temporal hierarchies, reconciling weekly, monthly, quarterly and annual forecasts of the same series.
- When aggregate series are stable and forecastable but disaggregate series are noisy, since reconciliation lets the top inform the bottom.

**When not to use it.** When there is genuinely only one level of interest; when the hierarchy is not additive (ratios, averages, percentages require different treatment); when different levels are legitimately forecasting different things under different assumptions.

**Assumptions and failure modes.** Requires an exactly additive aggregation structure. Traditional top-down approaches lose bottom-level information and bottom-up approaches accumulate noise; both are dominated by optimal reconciliation in most empirical comparisons. MinT requires estimating a potentially large error covariance matrix, which needs shrinkage in realistic hierarchies. Reconciliation can degrade a level that was already well forecast if the covariance estimate is poor. Probabilistic reconciliation of full forecast distributions is more involved than reconciling point forecasts and is often skipped.

**Where it is used.** Retail and CPG demand planning, energy demand forecasting across grid levels, tourism and transport demand, macroeconomic and national statistics forecasting, corporate financial planning across business units, call centre volume planning.

---

### 14.7 Empirical Forecast Error Decomposition

**What it does.** Splits observed forecast error into interpretable components: systematic bias, variance driven by model instability or parameter estimation, and irreducible noise inherent to the series. This diagnoses whether poor accuracy reflects a fixable modelling problem or an intrinsic limit, which are very different situations calling for very different responses.

**When to use it.**
- Post-mortem analysis of a forecasting process that is underperforming, to direct effort where it will pay off.
- Distinguishing a model that is systematically wrong (bias, fixable by respecification) from one that is noisy (variance, fixable by regularization, pooling or ensembling) from one that is at the noise floor (unfixable, requiring a process change instead).
- Detecting judgemental override bias, where human adjustments to statistical forecasts introduce systematic error.
- Building the case to stop investing in model improvement and start investing in demand shaping, buffering or better data.
- Segmenting a forecast portfolio by error type to apply differentiated remedies.

**When not to use it.** With too few forecast observations to separate components reliably; when the process has changed structurally during the evaluation window, mixing regimes into a single decomposition.

**Assumptions and failure modes.** Requires a genuine hold-out or rolling-origin evaluation, never in-sample residuals, which understate error badly. The irreducible component can only be estimated, not observed, and is often approximated by the accuracy of a very strong benchmark or by entropy-based limits. Decomposition depends on the error metric chosen, and scale-dependent metrics (MAE, RMSE) cannot be compared across series while percentage metrics (MAPE) break down near zero values; scaled metrics such as MASE are usually preferable. Bias measured over a period containing a structural break is not meaningful bias.

**Where it is used.** Supply chain forecast value-added analysis, S&OP process review, energy trading forecast evaluation, financial forecasting audits, machine learning error analysis and model debugging.

---

## 15. Initial Conditions, Chaos and Predictability Limits

### 15.1 Initial Condition Sensitivity (Lyapunov Exponents)

**What it does.** Measures the average exponential rate at which two initially close trajectories separate. A positive largest Lyapunov exponent is the defining signature of chaos and implies a finite predictability horizon: since initial conditions are never known exactly, errors grow exponentially and any forecast eventually becomes worthless regardless of model quality or computing power. The inverse of the exponent gives the characteristic timescale of that error growth.

**When to use it.**
- Establishing whether a system is fundamentally predictable and over what horizon, before promising forecast accuracy.
- Distinguishing deterministic chaos from stochastic noise in an observed series, which have very different modelling implications.
- Justifying ensemble forecasting rather than single deterministic runs.
- Setting the horizon beyond which forecasts should be replaced with climatological or distributional statements.
- Characterizing control problems where small perturbations have outsized long-run effects.

**When not to use it.** On short, noisy observational series, where estimates are unreliable and can indicate chaos where none exists; as a routine diagnostic for systems that are obviously stochastic rather than deterministic.

**Assumptions and failure modes.** Estimation from data requires state space reconstruction with well-chosen embedding parameters, and noise strongly biases estimates upward. Positive estimated exponents are frequently artefacts of noise or non-stationarity, which is why surrogate data testing is essential. The exponent is an asymptotic average and local predictability can differ substantially from it, which is what finite-time exponents address. A positive exponent limits trajectory forecasting but does not prevent statistical or distributional prediction of the attractor.

**Where it is used.** Meteorology and the theoretical basis of weather predictability limits, climate science, fluid dynamics and turbulence, cardiac and neural dynamics, population ecology, laser and plasma physics, financial market dynamics research.

---

### 15.2 Ensemble Methods

**What it does.** Runs a model many times from slightly different but equally plausible initial conditions, and often with perturbed parameters or model formulations, then uses the spread of outcomes as the forecast uncertainty. Instead of one trajectory it produces a probability distribution, which is the only honest output for a chaotic system.

**When to use it.**
- Any chaotic or strongly nonlinear system where a single deterministic run is unjustifiable.
- Communicating forecast confidence to decision-makers, since ensemble spread maps directly onto plume charts and probability statements.
- Probabilistic risk assessment where the probability of exceeding a threshold matters more than the central estimate.
- Detecting flow-dependent predictability: some situations are inherently more predictable than others, and the ensemble reveals which.
- Combining structurally different models (multi-model ensembles) to capture model-form uncertainty as well as initial condition uncertainty.

**When not to use it.** When computational budget forces so few members that the spread is meaningless; when the dominant uncertainty is structural and all members share the same wrong physics, producing confident consensus errors.

**Assumptions and failure modes.** Ensemble spread is a good uncertainty estimate only if members are generated to sample the actual analysis error distribution; naive random perturbations produce under-dispersive ensembles that are systematically over-confident, which is why singular vector and bred vector methods exist. Ensembles require calibration and verification with proper scoring rules (CRPS, Brier score) and rank histograms rather than being trusted as raw probabilities. Shared model bias is invisible from within the ensemble. Members are correlated, so effective ensemble size is smaller than nominal.

**Where it is used.** Numerical weather prediction (ECMWF, GEFS operational ensembles), seasonal and climate projection, hurricane track forecasting, hydrological and flood forecasting, epidemic projection, financial scenario generation, machine learning model ensembling.

---

### 15.3 State Space Reconstruction (Takens' Theorem)

**What it does.** Reconstructs the geometry of a dynamical system's attractor from a single observed variable by building delay-coordinate vectors from lagged copies of that series. Takens' embedding theorem guarantees that, with sufficient embedding dimension, this reconstruction preserves the topological properties of the true multidimensional state space, which means the dynamics can be studied without ever measuring the full state.

**When to use it.**
- Only one variable is measurable but the underlying system is known or suspected to be multidimensional and deterministic.
- Before computing Lyapunov exponents, correlation dimension or other dynamical invariants from data, all of which require an embedding.
- Nonlinear prediction using nearest-neighbour or analogue methods in reconstructed space, which can outperform linear models for deterministic systems.
- Testing whether an irregular series is low-dimensional deterministic chaos or high-dimensional noise.
- Detecting causal coupling between series via convergent cross mapping, which builds directly on reconstruction.

**When not to use it.** With short or heavily noisy series, where reconstruction quality collapses; for systems that are genuinely stochastic and high-dimensional, where no low-dimensional attractor exists to recover; when the sampling rate is far from the system's characteristic timescale.

**Assumptions and failure modes.** Requires a deterministic, low-dimensional underlying system observed through a smooth measurement function, and the theorem is asymptotic in noise-free, infinite-data conditions that never hold. Choice of time delay (via mutual information first minimum) and embedding dimension (via false nearest neighbours) drives everything and is partly judgement. Noise inflates apparent dimension. Non-stationarity invalidates the whole approach. Claims of low-dimensional chaos in economic and physiological data have a long history of failing surrogate data tests.

**Where it is used.** Nonlinear time series analysis generally, physiological signal analysis, ecology and population dynamics (convergent cross mapping for causality), climate dynamics, mechanical fault detection, laser and electronic circuit dynamics.

---

### 15.4 Data Assimilation

**What it does.** Continuously blends model forecasts with incoming observations to maintain the best possible estimate of a system's current state, correcting the accumulated drift that initial condition errors and model bias would otherwise cause. Variational methods (3D-Var, 4D-Var) minimize a cost function balancing model and observation errors over a window, while ensemble methods (EnKF) propagate flow-dependent uncertainty through the ensemble spread, and hybrid schemes combine both.

**When to use it.**
- Operational forecasting systems that must repeatedly re-initialize from a moving stream of heterogeneous observations.
- Any situation where the model is good but drifts, and observations are informative but sparse, noisy and irregularly distributed.
- Reconstructing historical states (reanalysis), producing physically consistent gridded records from incomplete observation networks.
- Estimating unobservable states from indirect measurements, including parameters treated as augmented state variables.
- Digital twins that must track a physical asset in real time.

**When not to use it.** When the model is fundamentally wrong, since assimilation will fight the observations continuously and produce unphysical corrections; when observations are so dense and accurate that direct estimation suffices; for problems with strong non-Gaussianity and low dimension, where particle filters are more appropriate.

**Assumptions and failure modes.** Requires realistic specification of background and observation error covariances, which is difficult and drives assimilation quality more than the algorithm choice. Ensemble methods suffer sampling error with limited members, requiring covariance localization and inflation to avoid spurious long-range correlations and filter collapse. Observation bias is corrected explicitly (variational bias correction) or it propagates into the analysis. Systems can appear well-fitted to observations while being dynamically inconsistent, producing spin-up shocks.

**Where it is used.** Numerical weather prediction (the core operational technology), ocean and sea ice modelling, atmospheric reanalysis products (ERA5, MERRA), hydrology and flood forecasting, reservoir engineering history matching, air quality forecasting, space weather, robotics and navigation.

---

### 15.5 Shadowing

**What it does.** Addresses whether a numerically computed trajectory of a chaotic system means anything at all, given that rounding and truncation errors amplify exponentially. Shadowing lemmas establish that under hyperbolicity conditions there exists a true trajectory of the system, starting from slightly different initial conditions, that stays uniformly close to the computed one for a long time. The simulation is therefore not the trajectory you asked for, but it is a real trajectory of the system.

**When to use it.**
- Justifying the scientific validity of long chaotic simulations whose individual trajectories are known to be numerically unreliable.
- Assessing how long a computed orbit can be trusted as representative before it becomes fiction.
- Validating statistical and climatological conclusions drawn from chaotic simulations, where the ensemble of behaviour matters rather than any single path.
- Theoretical analysis of numerical methods for dynamical systems.

**When not to use it.** As a practical operational tool, since shadowing times are difficult to compute for realistic high-dimensional models; as reassurance for pointwise forecast accuracy, which it explicitly does not provide.

**Assumptions and failure modes.** Rigorous results require uniform hyperbolicity, which most physically interesting systems do not satisfy; real systems typically exhibit only finite-time shadowing, and shadowing can break down entirely near tangencies or in systems with unstable dimension variability. Shadowing justifies statistical rather than pointwise interpretation of simulations, a distinction routinely blurred when model output is presented. Numerical shadowing time estimates are themselves computationally demanding.

**Where it is used.** Theoretical and computational dynamical systems, climate model validity arguments, molecular dynamics simulation justification, celestial mechanics and long-term solar system integration, numerical analysis research.

---

### 15.6 Bifurcation Analysis

**What it does.** Maps how the qualitative behaviour of a system changes as a parameter is varied, identifying critical values where stable states appear, disappear, exchange stability or change type. Saddle-node bifurcations create or destroy equilibria, Hopf bifurcations give birth to oscillations, and period-doubling cascades lead to chaos. The result is a picture of which regimes are possible and where the boundaries between them lie.

**When to use it.**
- Identifying tipping points and critical thresholds beyond which a system reorganizes abruptly rather than gradually.
- Understanding why a system with multiple stable states can end up in different regimes from nearly identical starting conditions (hysteresis and path dependence).
- Designing systems to operate safely away from dangerous parameter regions, or deliberately near a bifurcation for sensitivity.
- Explaining the onset of oscillations or instability in engineered and biological systems.
- Supporting early warning signal research, where critical slowing down and rising variance precede a bifurcation.

**When not to use it.** Purely stochastic systems with no underlying deterministic skeleton; very high-dimensional systems where continuation methods are computationally infeasible without reduction; empirical settings where the governing equations are unknown, though data-driven early warning indicators partially substitute.

**Assumptions and failure modes.** Requires a model, usually a system of differential or difference equations, so it is a modelling exercise not a data analysis method. Numerical continuation software (AUTO, MatCont) is the practical tool and requires care near degeneracies. Noise smears sharp bifurcations and can induce transitions before the deterministic threshold, so stochastic bifurcation theory is needed for noisy systems. Global bifurcations are harder to detect than local ones. Extrapolating a low-dimensional bifurcation picture to a complex real system is a common overreach.

**Where it is used.** Climate tipping point analysis (AMOC collapse, ice sheet thresholds), ecology and regime shifts in lakes and fisheries, epidemiology (threshold behaviour in transmission models), neuroscience (onset of seizures and neural oscillations), power system stability and voltage collapse, structural buckling analysis, chemical reactor dynamics, economics (multiple equilibria and financial crises).

---

## 16. Stability Theory and Nonlinear Control

### 16.1 Lyapunov Stability Theory

**What it does.** Certifies that an equilibrium is stable without solving the system's equations, by finding a scalar function that is positive everywhere except at the equilibrium and whose value decreases along every trajectory. Physically it generalizes the idea of energy dissipation: if a suitable energy-like quantity always decreases, the system must settle. The direct method converts a hard dynamics question into a search for a suitable function.

**When to use it.**
- Proving stability of nonlinear systems where explicit solutions are unavailable, which is nearly all of them.
- Certifying safety-critical control designs where empirical testing cannot cover the operating envelope.
- Establishing regions of attraction: not just whether an equilibrium is stable but how far from it the system can be perturbed and still return.
- Analysing convergence of iterative algorithms and optimization schemes, where Lyapunov arguments underpin most convergence proofs.
- Adaptive control, where the Lyapunov function is used to derive parameter update laws that guarantee stability during learning.

**When not to use it.** When linearization suffices, since Lyapunov's indirect method via eigenvalues is far easier for local questions away from marginal cases; when no candidate function can be found and the failure is inconclusive rather than informative.

**Assumptions and failure modes.** Failure to find a Lyapunov function proves nothing, which is the method's fundamental asymmetry. Construction is an art for general systems, though quadratic forms work for linear systems and physically motivated energy functions often work for mechanical ones. A conservative Lyapunov function yields a conservative region of attraction estimate, potentially far smaller than the true one. Requires an accurate model, so stability certificates inherit all model error. Time-varying and switched systems need extended machinery.

**Where it is used.** Aerospace and flight control certification, robotics and manipulator control, power system stability assessment, automotive stability control, process control, network congestion control design, optimization and machine learning convergence analysis, epidemiological model equilibrium analysis.

---

### 16.2 LaSalle's Invariance Principle

**What it does.** Rescues stability conclusions when the candidate function is only non-increasing rather than strictly decreasing, which happens whenever the derivative vanishes on a set larger than the equilibrium. Instead of demanding strict decrease, it shows trajectories converge to the largest invariant set contained where the derivative is zero, and often that set contains only the equilibrium.

**When to use it.**
- Mechanical and electrical systems with damping that acts on only some states, so energy is constant along certain directions but the system still settles.
- Adaptive control, where parameter errors do not appear in the derivative of the Lyapunov function, making strict decrease impossible yet convergence of the tracking error still provable.
- Systems where a natural energy function exists but is only semi-definite in its derivative, which is extremely common in practice.
- Establishing convergence to limit cycles or invariant manifolds rather than to a point.

**When not to use it.** Time-varying systems, where the standard principle does not apply and Barbalat's lemma or extensions are required; when a strict Lyapunov function is readily available and simpler.

**Assumptions and failure modes.** Requires the system to be autonomous (time-invariant) and trajectories to be bounded, which must be established separately, often via radial unboundedness of the candidate function. Identifying the largest invariant set requires additional analysis and is where the real work lies. It gives convergence to a set, so uniqueness of the limit point needs a separate argument. Applying it to time-varying systems without justification is a frequent error in the literature.

**Where it is used.** Robotics and mechanical system control, adaptive control theory, power electronics and converter stability, neural network convergence analysis, consensus and multi-agent coordination protocols, epidemiological model global stability proofs.

---

### 16.3 Barrier Functions / Control Barrier Functions (CBF)

**What it does.** Certifies that a system never leaves a designated safe set, which is a different and often more urgent question than whether it converges to an equilibrium. A control barrier function defines the safe region implicitly and yields a linear constraint on the control input at each instant; enforcing that constraint guarantees forward invariance of the safe set. Combined with a nominal controller in a quadratic program, it produces a minimally invasive safety filter that overrides commands only when necessary.

**When to use it.**
- Safety-critical autonomy where collision avoidance, geofencing or operational limits must hold with certainty rather than in expectation.
- Wrapping an existing or learned controller with a runtime safety layer, without redesigning the underlying policy.
- Safe reinforcement learning, where exploration must be constrained to prevent catastrophic actions during training.
- Enforcing state and actuator constraints in real time at controller rates, where full model predictive control would be too slow.
- Multi-agent settings where each agent enforces pairwise safety constraints locally.

**When not to use it.** When the safe set is not easily expressible as a smooth superlevel set; when model uncertainty is large and unmodelled, since guarantees hold only for the assumed dynamics (use robust or adaptive CBFs); when the performance objective and safety constraint conflict so often that the filter dominates behaviour, which signals a design problem upstream.

**Assumptions and failure modes.** Requires a valid CBF, whose construction for complex safe sets is nontrivial and is the main practical bottleneck. Guarantees are model-dependent, so disturbances and parametric error break them unless explicitly handled. High relative degree systems need exponential or higher-order CBFs. The pointwise quadratic program is myopic and can drive the system into states where no safe action remains, so feasibility must be verified in advance. Multiple simultaneous constraints can render the program infeasible.

**Where it is used.** Autonomous vehicles and adaptive cruise control, drone flight envelope protection, robot manipulators sharing space with humans, safe reinforcement learning research, industrial process safety systems, multi-robot collision avoidance.

---

### 16.4 Sum-of-Squares (SOS) Programming

**What it does.** Automates the search for Lyapunov functions and other certificates by exploiting the fact that a polynomial expressible as a sum of squares is necessarily non-negative. Since testing SOS membership reduces to semidefinite programming, the previously artisanal task of constructing certificates becomes a convex optimization problem solvable by computer.

**When to use it.**
- Polynomial or polynomial-approximable dynamics where hand construction of a Lyapunov function has failed.
- Estimating regions of attraction with certified bounds rather than simulation-based guesses.
- Verifying safety and reachability properties of nonlinear systems, including barrier certificate synthesis.
- Robust stability analysis with parametric uncertainty expressed polynomially.
- Any setting where a machine-checkable certificate is required for assurance or certification purposes.

**When not to use it.** High-dimensional systems, where the semidefinite program size grows combinatorially with state dimension and polynomial degree, typically limiting practical use to modest dimensions; non-polynomial dynamics that resist accurate polynomial approximation; problems where a simple quadratic Lyapunov function already works.

**Assumptions and failure modes.** Non-negativity does not imply SOS in general, so the method is sufficient but not necessary and can fail to certify a genuinely stable system. Scalability is the dominant limitation, though structure exploitation, sparsity and alternatives such as DSOS and SDSOS relaxations extend reach. Numerical conditioning of the resulting semidefinite programs can be poor, and certificates should be verified in exact or interval arithmetic for high-assurance use. Degree selection is a trial-and-error loop.

**Where it is used.** Aerospace flight envelope and control verification, robotics motion planning with certified funnels, power system transient stability, formal verification of hybrid and cyber-physical systems, control of chemical and biological networks, optimization theory.

---

### 16.5 Lyapunov Spectrum

**What it does.** Gives the full set of Lyapunov exponents, one per state space direction, describing how volumes stretch and contract in each direction of the flow. The signature pattern classifies the attractor: all negative for a fixed point, one zero and the rest negative for a limit cycle, two zeros for a torus, and at least one positive for chaos. The sum of exponents measures overall volume contraction and indicates whether the system is dissipative.

**When to use it.**
- Full characterization of a dynamical system's behaviour beyond the single question of whether it is chaotic.
- Distinguishing quasi-periodic from chaotic behaviour, which the largest exponent alone cannot do reliably.
- Computing the Kaplan-Yorke dimension, which requires the ordered spectrum.
- Identifying the number of unstable directions, which informs how many ensemble perturbation directions matter for forecasting.
- Verifying that a numerical model reproduces the dynamical character of the system it represents.

**When not to use it.** From short experimental time series, where reliably estimating anything beyond the largest exponent is generally not credible; for high-dimensional systems where computing the full spectrum is prohibitively expensive.

**Assumptions and failure modes.** Requires long trajectories and, for accurate computation, access to the system's Jacobian, which is available for models but not for data. Estimation from data via QR-based or Jacobian-fitting methods is sensitive to embedding parameters and noise, and spurious exponents routinely appear when the embedding dimension exceeds the true dimension. Exponents are asymptotic averages over the attractor and require sufficient trajectory length to converge. Non-stationary systems have no well-defined spectrum.

**Where it is used.** Nonlinear dynamics and chaos research, turbulence and fluid dynamics, climate model characterization, laser and nonlinear optics, cardiac and neural dynamics, celestial mechanics, plasma physics.

---

### 16.6 Kaplan-Yorke Dimension

**What it does.** Estimates the fractal dimension of an attractor directly from the Lyapunov spectrum, by finding the number of directions whose cumulative expansion balances contraction and interpolating between them. It links dynamics to geometry, quantifying how many effective degrees of freedom the long-run behaviour actually occupies, which is typically far fewer than the full state space dimension.

**When to use it.**
- Estimating the effective dimensionality of a complex system's long-term behaviour, which informs how many variables a reduced model needs.
- Confirming that an apparently high-dimensional system is in fact confined to a low-dimensional attractor, justifying model reduction.
- Comparing attractor complexity across parameter regimes or across models.
- Cross-checking correlation dimension estimates obtained directly from data, since the two should broadly agree.

**When not to use it.** When the Lyapunov spectrum itself is unreliable, since all error propagates; for stochastic systems, which have no finite-dimensional attractor; as a substitute for direct dimension estimation when a good time series is available and the model is not.

**Assumptions and failure modes.** The Kaplan-Yorke conjecture equating this quantity to the information dimension holds for many systems but is not proven in general, so it is an estimate with theoretical caveats. Requires the full ordered spectrum, inheriting all its estimation difficulties. Non-integer values are expected and meaningful, indicating fractal structure, but are often misread as errors. Sensitive to the accuracy of the smallest exponents, which are the hardest to compute.

**Where it is used.** Chaos and dynamical systems research, turbulence studies (estimating attractor dimension of fluid flows), climate dynamics and reduced-order modelling, neural dynamics, laser physics, model reduction in engineering.

---

### 16.7 Finite-Time Lyapunov Exponents (FTLE)

**What it does.** Computes stretching rates over a finite time window and over space, rather than asymptotically, producing a field that varies with position and time. Ridges of high FTLE mark Lagrangian Coherent Structures, the material barriers that organize transport: particles on opposite sides of a ridge take fundamentally different paths, so the ridges reveal the hidden skeleton of a flow.

**When to use it.**
- Predicting where material, pollutants, oil, plankton or debris will be transported in a time-varying flow, where instantaneous velocity snapshots are misleading.
- Search and rescue planning, where identifying attracting structures narrows the search region dramatically.
- Identifying mixing regions and transport barriers in ocean, atmospheric and industrial flows.
- Analysing time-dependent or aperiodic systems, where classical asymptotic invariant manifolds do not exist.
- Quantifying flow-dependent predictability: regions of high local stretching are locally less predictable regardless of the global exponent.

**When not to use it.** Steady flows where classical invariant manifold analysis is simpler and exact; when the velocity field data are too coarse or noisy for reliable particle integration; when only bulk statistics are needed rather than spatial structure.

**Assumptions and failure modes.** Requires a well-resolved velocity field over the integration window, and results depend strongly on the chosen integration time: too short reveals no structure, too long smears everything. Sensitive to the spatial resolution of the seeding grid and to interpolation of the velocity data. FTLE ridges are a diagnostic for coherent structures rather than a rigorous definition, and alternative variational and geodesic definitions exist that are more mathematically defensible. Computationally intensive due to dense particle advection.

**Where it is used.** Physical oceanography and ocean transport, atmospheric dispersion and pollutant tracking, search and rescue operations at sea, oil spill response modelling, cardiovascular blood flow analysis, aerodynamics and vortex identification, industrial mixing process design.

---

### 16.8 Lyapunov-Based Control Design

**What it does.** Inverts the usual analysis workflow: rather than designing a controller and then testing stability, it selects a desired Lyapunov function and derives the control law that forces its derivative to be negative. Stability is therefore guaranteed by construction. Backstepping builds such controllers recursively for cascaded systems, and sliding mode control uses a related argument to drive the state onto a stable surface.

**When to use it.**
- Nonlinear systems where linear design methods fail or only work in a small neighbourhood of an operating point.
- Systems with strict cascade or strict-feedback structure, where backstepping applies systematically.
- Adaptive control, where unknown parameters are estimated online and the Lyapunov argument simultaneously guarantees stability and parameter update behaviour.
- Robust control against matched uncertainties and disturbances, notably via sliding mode designs.
- Safety-critical applications where a stability proof is a deliverable rather than an afterthought.

**When not to use it.** When gain scheduling around linear designs meets requirements at lower complexity; when the resulting control law is too aggressive or requires unavailable state measurements; when performance shaping matters more than stability guarantees, since Lyapunov designs guarantee stability without optimizing transient quality.

**Assumptions and failure modes.** Requires accurate model structure, and mismatched uncertainty breaks guarantees. Full state feedback is usually assumed, requiring an observer in practice, and the separation principle does not generally hold for nonlinear systems. Backstepping suffers explosion of terms as system order rises, mitigated by dynamic surface control. Sliding mode designs produce chattering that damages actuators unless boundary layers or higher-order sliding modes are used. Guaranteed stability can coexist with poor performance, so transient behaviour must be checked separately.

**Where it is used.** Aerospace and missile guidance, robotic manipulators and mobile robots, electric drives and power electronics, automotive traction and stability systems, marine vessel dynamic positioning, spacecraft attitude control, process control of nonlinear reactors.

---

### 16.9 Contraction Analysis

**What it does.** Shifts the question from stability of a particular equilibrium to whether all trajectories converge toward each other, by examining whether the system's differential dynamics uniformly shrink distances in a suitable metric. If so, the system forgets initial conditions exponentially and any two solutions converge, regardless of where they start or what the nominal trajectory is.

**When to use it.**
- Tracking control and synchronization problems, where convergence between trajectories is the actual requirement rather than convergence to a fixed point.
- Systems with no fixed equilibrium, such as those following time-varying references.
- Composing subsystems, since contraction is preserved under hierarchical, parallel and feedback combinations in ways classical Lyapunov certificates are not, making it well suited to modular and networked systems.
- Observer design, where the observer error dynamics must contract.
- Analysing convergence of learned dynamics and neural network controllers, an active area where contraction gives certificates that are learnable jointly with the model.

**When not to use it.** When contraction is too strong a requirement, since many useful systems are stable but not globally contracting; when finding an appropriate metric is as hard as finding a Lyapunov function, which it often is.

**Assumptions and failure modes.** Requires uniform negative definiteness of the symmetric part of the Jacobian in some metric, a demanding condition. The choice of metric is the central difficulty and can be searched for with SOS or learned with neural contraction metrics. Contraction is a stronger property than asymptotic stability, so failure to establish it does not imply instability. Partial contraction and semi-contraction extensions handle systems that contract only in some directions.

**Where it is used.** Robotics and trajectory tracking, synchronization of oscillator networks and power grids, observer and estimator design, multi-agent consensus, neuroscience network models, learning-based control with stability guarantees, optimization algorithm analysis.

---

### 16.10 Input-to-State Stability (ISS)

**What it does.** Extends stability to systems driven by external disturbances, guaranteeing that the state remains bounded by a decaying function of the initial condition plus a bounded function of the disturbance magnitude. In plain terms, bounded inputs produce bounded states, and small persistent disturbances produce proportionally small persistent deviations, with no possibility of unbounded drift.

**When to use it.**
- Real systems subject to disturbances, noise and modelling error, where nominal stability of the undisturbed system is not sufficient assurance.
- Cascaded and interconnected systems, where the ISS small-gain theorem certifies stability of the interconnection from properties of the components.
- Networked and event-triggered control, where communication delays, packet loss and quantization act as disturbances.
- Analysing robustness of learning-based controllers, where model error is treated as an input.
- Justifying that measurement noise or actuator error will not destabilize a closed loop.

**When not to use it.** When disturbances are unbounded or persistent in a way that violates the framework's premises; when precise performance bounds rather than qualitative boundedness are required; when the system is only locally ISS and operation may leave that region.

**Assumptions and failure modes.** Requires an ISS-Lyapunov function, whose construction inherits all the difficulties of the classical case plus the disturbance term. The gain function characterizes how much disturbance translates into state deviation, and a system can be ISS with a gain so large that the guarantee is operationally worthless, which is a common oversight. Local versus global ISS matters and is often stated imprecisely. Variants (integral ISS, incremental ISS, ISS with respect to sets) address cases where the standard notion is too strong or too weak.

**Where it is used.** Nonlinear control theory as a foundational robustness concept, networked control systems, robotics under disturbance, automotive and aerospace robust control, power system resilience analysis, safe learning-based control, biological network robustness analysis.

---

## 17. Spectral and Time-Frequency Analysis

### 17.1 Discrete Fourier Transform (DFT)

**What it does.** Converts a finite sequence of evenly spaced samples into complex coefficients representing the amplitude and phase of each frequency component. It is an exact, invertible change of basis: no information is lost, and the same data are simply re-expressed in terms of oscillations rather than time points.

**When to use it.**
- Identifying which periodicities exist in a signal and how strongly.
- Any operation that is simpler in the frequency domain, including convolution, correlation and filtering, all of which become multiplication.
- Compression, where most energy concentrates in few coefficients.
- Solving differential equations with periodic boundary conditions via spectral methods.
- Preparing inputs for downstream spectral features in classification and diagnostics.

**When not to use it.** Non-stationary signals whose frequency content changes over time (use STFT or wavelets); irregularly sampled data (use the Lomb-Scargle periodogram); very short records where frequency resolution is inadequate.

**Assumptions and failure modes.** Assumes the signal is periodic with period equal to the record length, so a non-integer number of cycles causes spectral leakage, smearing energy across frequencies; windowing reduces leakage at the cost of resolution. Frequency resolution equals the sampling rate divided by the number of samples and cannot be improved by zero-padding, which only interpolates the display. Frequencies above the Nyquist limit alias into the visible band and must be removed by analogue filtering before sampling, since no post-processing can undo aliasing. Phase information is often discarded despite being essential for reconstruction.

**Where it is used.** All of digital signal processing, audio and image processing, telecommunications, vibration analysis, spectroscopy, medical imaging reconstruction, numerical analysis.

---

### 17.2 Fast Fourier Transform (FFT)

**What it does.** Computes the DFT in O(n log n) operations instead of O(n²) by recursively exploiting symmetry and periodicity in the transform, most famously through the Cooley-Tukey radix-2 decomposition. It is the same transform, not an approximation, and its speed is what made frequency-domain methods practical on real data at scale.

**When to use it.**
- Whenever a DFT is needed, since there is essentially no reason to compute one directly.
- Real-time and streaming spectral analysis, where the speed difference is the difference between feasible and impossible.
- Fast convolution and correlation of long sequences, including polynomial and large integer multiplication.
- Building blocks for STFT, spectrograms, Welch's method and most modern spectral estimation.

**When not to use it.** When only a few specific frequency bins are needed, where the Goertzel algorithm is cheaper; for very sparse spectra, where sparse FFT algorithms do better; when the data are irregularly sampled.

**Assumptions and failure modes.** Classical radix-2 implementations prefer power-of-two lengths, and while modern libraries handle arbitrary lengths, performance can degrade badly for lengths with large prime factors, so padding to a convenient length is standard practice. Numerical precision accumulates error over very long transforms, though this is rarely limiting in double precision. All DFT caveats about leakage, resolution and aliasing apply unchanged, and the speed of the FFT tempts users to skip the windowing and preprocessing that make results meaningful.

**Where it is used.** Universally in signal processing software and hardware, audio codecs (MP3, AAC), image compression (JPEG uses the related DCT), OFDM in 4G, 5G and Wi-Fi, radar and sonar, MRI reconstruction, seismic processing, real-time machine condition monitoring.

---

### 17.3 Inverse FFT (IFFT)

**What it does.** Reconstructs the time-domain signal from its frequency coefficients, completing the round trip. This makes frequency-domain manipulation practical: transform, edit the coefficients, transform back, and the modification is expressed as a filtering operation on the original signal.

**When to use it.**
- Filtering workflows, where unwanted frequency bands are attenuated or removed and the cleaned signal is recovered.
- Fast convolution, where two signals are multiplied in the frequency domain and returned to the time domain.
- Signal synthesis from a designed spectrum, including OFDM symbol generation in modern communications.
- Reconstruction in imaging modalities where measurements are acquired in the frequency domain, such as MRI k-space.
- Simulation of random processes with a specified power spectrum, by randomizing phase and inverting.

**When not to use it.** When frequency-domain features are the end product and no reconstruction is needed; when time-domain filter designs (FIR, IIR) are better suited to streaming and low-latency requirements.

**Assumptions and failure modes.** Abrupt zeroing of frequency bins produces ringing artefacts in the time domain (Gibbs phenomenon), so smooth transition bands are preferable to brick-wall edits. Hermitian symmetry must be preserved when the output should be real, or a spurious imaginary component appears. Circular convolution wraparound corrupts the edges unless zero-padding with overlap-add or overlap-save is used. Phase must be retained and manipulated correctly; discarding it makes reconstruction impossible.

**Where it is used.** Audio and image filtering, OFDM transmitters, MRI and CT reconstruction, seismic data processing, noise removal pipelines, synthetic time series generation with prescribed spectra.

---

### 17.4 Short-Time Fourier Transform (STFT)

**What it does.** Applies the FFT to successive overlapping windowed segments, producing a two-dimensional map of how spectral content evolves over time. The resulting spectrogram is the standard visualization for non-stationary signals, showing when each frequency was present as well as how strongly.

**When to use it.**
- Any signal whose frequency content changes: speech, music, machine run-up and coast-down, transient events, biomedical signals.
- Detecting when a fault frequency appears or a regime shifts, rather than merely that it exists somewhere in the record.
- Feature extraction for audio and vibration machine learning models, where spectrograms are the standard input representation.
- Real-time monitoring displays, since the computation is incremental and streaming-friendly.

**When not to use it.** When signals contain both very short high-frequency transients and long low-frequency components, where a single fixed window cannot serve both (use wavelets); when precise instantaneous frequency of a single component is required (use the Hilbert transform); for extremely non-stationary or chirping signals where a fixed window smears the structure.

**Assumptions and failure modes.** The window length imposes a hard tradeoff dictated by the uncertainty principle: long windows give fine frequency resolution and poor time localization, short windows the reverse, and no choice escapes it. Window shape (Hann, Hamming, Blackman) determines leakage and main-lobe width. Overlap of 50 to 75 percent is typical and affects both smoothness and computational cost. Reconstruction requires the constant-overlap-add condition to be satisfied. Signals are assumed quasi-stationary within each window, which fails for rapidly chirping content.

**Where it is used.** Speech and audio processing, music information retrieval, machine condition monitoring and order tracking, EEG and neural oscillation analysis, radar and sonar, seismic event analysis, deep learning audio pipelines.

---

### 17.5 Wavelet Transform

**What it does.** Analyses a signal with scaled and shifted copies of a localized oscillatory function, using long windows for low frequencies and short windows for high frequencies. This multi-resolution structure resolves the STFT's fixed tradeoff, giving good frequency resolution where signals vary slowly and good time resolution where they vary rapidly, which matches how most real signals actually behave.

**When to use it.**
- Signals containing transients, discontinuities or sharp edges, which Fourier methods represent poorly and wavelets represent compactly.
- Denoising via coefficient thresholding, which preserves edges far better than low-pass filtering.
- Compression, where the concentration of energy into few coefficients is exploited directly (JPEG 2000).
- Multi-resolution decomposition, separating a signal into components at different scales for separate analysis or forecasting.
- Detecting singularities, fault onsets and abrupt structural changes with precise time localization.
- Analysing self-similar or fractal signals, where scale-based analysis is natural.

**When not to use it.** When the signal is stationary and composed of steady sinusoids, where Fourier analysis is simpler and more interpretable; when the arbitrary choice of wavelet family would make results hard to defend; when frequency values must be reported precisely, since wavelet scales map only approximately to frequencies.

**Assumptions and failure modes.** Results depend on the wavelet family chosen (Morlet for smooth oscillatory analysis, Daubechies for compact support and vanishing moments, Haar for sharp steps), and this choice is a genuine modelling decision rather than a detail. The discrete wavelet transform is not shift-invariant, so results change if the signal is shifted by one sample, which the stationary or maximal-overlap variant corrects at higher cost. Boundary effects contaminate the edges of the analysis (cone of influence) and must be excluded from interpretation. Scale-to-frequency conversion is family-dependent and often misreported.

**Where it is used.** Image compression and denoising, seismic and geophysical analysis, ECG and EEG analysis, machine fault detection and bearing diagnostics, financial multi-scale analysis, turbulence studies, gravitational wave burst detection.

---

### 17.6 Hilbert Transform

**What it does.** Constructs the analytic signal by pairing a real signal with a ninety-degree phase-shifted version of itself, from which instantaneous amplitude (envelope) and instantaneous frequency can be extracted at every time point. Where Fourier methods describe average behaviour over a window, this gives a point-by-point description of how a signal's amplitude and frequency evolve.

**When to use it.**
- Envelope detection, notably in bearing and gearbox diagnostics, where fault information resides in the modulation envelope of a high-frequency resonance rather than in the raw spectrum.
- Demodulation of amplitude and frequency modulated signals.
- Measuring instantaneous phase for synchronization and phase-locking analysis between signals, standard in neuroscience.
- Tracking frequency variation within a single oscillatory component, such as a chirp or a drifting resonance.
- Empirical mode decomposition workflows, where it supplies the instantaneous frequency of each extracted mode.

**When not to use it.** Multi-component signals without prior decomposition, since instantaneous frequency of a mixture is meaningless and can be negative or wildly erratic; broadband or noisy signals where the narrowband assumption fails.

**Assumptions and failure modes.** Meaningful instantaneous frequency requires a narrowband, mono-component signal satisfying the Bedrosian and Nuttall conditions, which is the single most violated prerequisite in applied use. Prefiltering into narrow bands or decomposing with EMD or wavelets is usually mandatory first. Edge effects contaminate the start and end of the record. Noise causes large spurious excursions in instantaneous frequency, so smoothing or amplitude-weighted estimates are common.

**Where it is used.** Rotating machinery diagnostics and envelope analysis, communications demodulation, neuroscience phase synchronization and cross-frequency coupling, seismology, structural health monitoring, speech analysis, power system phasor estimation.

---

### 17.7 Welch's Method

**What it does.** Estimates the power spectral density by splitting the signal into overlapping segments, windowing and transforming each, then averaging the resulting periodograms. Averaging reduces the variance of the estimate substantially, converting an inconsistent raw periodogram into a usable, smooth spectral estimate at the price of coarser frequency resolution.

**When to use it.**
- Estimating the spectrum of a noisy stationary signal, which is the normal case in practice.
- Comparing spectra across conditions or across time blocks, where a stable, low-variance estimate is needed for the comparison to be meaningful.
- Long records where segment averaging costs nothing and gains a great deal.
- Noise floor characterization and signal-to-noise measurement.
- Establishing baseline spectra for condition monitoring alarms, where estimate stability determines false alarm rates.

**When not to use it.** Short records, where segmenting leaves too few points per segment for adequate resolution (use multitaper); when resolving closely spaced spectral lines is the objective, since averaging blurs them; non-stationary signals where averaging across time mixes distinct regimes.

**Assumptions and failure modes.** Assumes stationarity across the whole record, so any drift or regime change is averaged into a misleading composite. Segment length sets the bias-variance tradeoff directly: more segments means lower variance but poorer resolution. Overlap of 50 percent with a Hann window is the standard default and is near-optimal for variance reduction. Window normalization must be handled correctly or absolute power levels are wrong, a frequent implementation bug. Averaging suppresses genuine transients, which may be exactly what matters.

**Where it is used.** Vibration and acoustic analysis, EEG and MEG spectral power estimation, communications noise measurement, oceanographic and atmospheric spectra, structural dynamics testing, electronics noise characterization.

---

### 17.8 Multitaper Spectral Estimation

**What it does.** Estimates the spectrum by applying several mutually orthogonal windows (discrete prolate spheroidal sequences, or Slepian tapers) to the entire record and averaging the resulting spectra. Because each taper extracts approximately independent information from the same data, variance is reduced without discarding record length, achieving better bias-variance performance than segment averaging and markedly lower spectral leakage.

**When to use it.**
- Short or precious records where segmenting for Welch's method would destroy frequency resolution.
- High dynamic range signals where leakage from strong components would otherwise mask weak ones.
- Formal statistical inference on spectra, since the method supports F-tests for spectral lines and jackknife confidence intervals on spectral estimates.
- Detecting weak periodic signals against a coloured noise background.
- Any application where spectral estimates must be defensible rather than merely indicative.

**When not to use it.** When computational simplicity matters more than statistical efficiency and Welch's method is adequate; very long stationary records where the advantage is marginal; real-time low-power applications where computing multiple tapers is too costly.

**Assumptions and failure modes.** Requires choosing the time-bandwidth product NW and the number of tapers, typically 2NW minus 1, which jointly control the resolution-variance tradeoff; more tapers means lower variance and wider effective bandwidth. Assumes stationarity over the record. Higher-order tapers have increasing leakage, so using too many degrades the estimate. Adaptive weighting of tapers improves performance for signals with strong dynamic range but adds complexity. Interpretation of the effective resolution bandwidth is frequently misunderstood.

**Where it is used.** Neuroscience and electrophysiology (a field standard, embedded in Chronux and FieldTrip), geophysics and seismology, climate time series analysis, astronomy and pulsar timing, paleoclimate cycle detection, gravitational wave data analysis.

---

## 18. Distribution-Free Uncertainty

### 18.1 Conformal Prediction

**What it does.** Wraps any predictive model, however complex, in a procedure that outputs prediction sets or intervals with a guaranteed error rate. It scores how unusual each calibration point is under the model, takes the appropriate quantile of those nonconformity scores, and uses it to size prediction regions for new points. The coverage guarantee is finite-sample valid and requires only exchangeability, not correct model specification, distributional assumptions or asymptotics.

**When to use it.**
- Deploying black-box models where honest uncertainty is required but the model provides none, or provides badly calibrated probabilities.
- Regulated and safety-critical settings where a distribution-free coverage guarantee is far easier to defend than an assumption-laden interval.
- Classification tasks where abstention matters: conformal prediction naturally returns sets containing several labels when the model is uncertain, and a single label when it is confident, which is exactly the behaviour needed for selective prediction and human handoff.
- Medical diagnosis support, where a set of candidate conditions with guaranteed coverage is more useful and more honest than a single confident label.
- Any deep learning system whose softmax outputs are known to be overconfident.
- Adaptive variants handle distribution shift over time, maintaining coverage in non-exchangeable streaming settings.

**When not to use it.** When conditional coverage for specific subgroups is essential, since standard conformal prediction guarantees marginal coverage overall and can under-cover hard subgroups while over-covering easy ones (use Mondrian or group-conditional variants); time series and shifted data without adaptive corrections, where exchangeability fails; when very tight intervals are needed and the underlying model is weak, since conformal prediction reports the model's true uncertainty rather than manufacturing precision.

**Assumptions and failure modes.** Exchangeability of calibration and test data is the sole but crucial assumption, and it is violated by temporal drift, covariate shift and any selection effect between calibration and deployment. A held-out calibration set is required for split conformal, costing data; full and cross-conformal variants avoid this at higher computational cost. Interval width is driven entirely by the underlying model's quality, so a poor model yields wide intervals: this is correct behaviour, though it is often mistaken for a failure of the method. The choice of nonconformity score determines efficiency and adaptivity, and naive scores yield constant-width intervals that ignore heteroscedasticity, which conformalized quantile regression fixes.

**Where it is used.** Machine learning deployment and MLOps uncertainty layers, medical imaging and diagnosis support, drug discovery and molecular property prediction, autonomous systems perception, demand forecasting intervals, financial risk model validation, large language model uncertainty and selective answering research.

---

## 19. Dimensionality Reduction and Matrix Factorization

### 19.1 Principal Component Analysis (PCA)

**What it does.** Finds the orthogonal directions along which data vary most, ordered by the amount of variance each explains, and projects the data onto the leading few. Because real datasets usually have strong correlations, a small number of components often captures most of the structure, turning a wide, redundant dataset into a compact one with little loss.

**When to use it.**
- Reducing dimensionality before modelling when predictors are numerous and correlated, particularly to stabilize regression (principal component regression).
- Visualizing high-dimensional data in two or three dimensions to inspect clustering, outliers and group separation.
- Noise reduction, by discarding low-variance components that typically carry more noise than signal.
- Compression and storage reduction where an approximate reconstruction suffices.
- Multivariate process monitoring, where Hotelling's T² and squared prediction error on PCA models detect abnormal operating conditions.
- Constructing composite indices from many correlated indicators.

**When not to use it.** When components must be interpretable in domain terms, since they are linear mixtures of all variables (use factor analysis, sparse PCA or non-negative matrix factorization); when the structure is nonlinear (use kernel PCA, UMAP or autoencoders); when the informative signal has low variance, since PCA will discard it; when the goal is class separation, where linear discriminant analysis targets the right objective.

**Assumptions and failure modes.** Assumes variance equals information and that relationships are linear. Scale-sensitive, so variables must be standardized unless they share meaningful units, and failing to do so lets high-variance variables dominate arbitrarily. Highly sensitive to outliers, which can define an entire component; robust PCA variants address this. Component signs are arbitrary and components can swap order between similar datasets, complicating comparison. The number of components to retain is a judgement call informed by scree plots, cumulative variance thresholds, parallel analysis or cross-validation. On genuinely uncorrelated data it achieves nothing.

**Where it is used.** Genomics and population structure analysis, chemometrics and spectroscopy, finance (yield curve level-slope-curvature decomposition, factor construction), image processing and eigenfaces, industrial process monitoring, survey and psychometric data reduction, climate field analysis (empirical orthogonal functions).

---

### 19.2 Singular Value Decomposition (SVD)

**What it does.** Factors any matrix into a rotation, a scaling by non-negative singular values, and another rotation. It is the most general and numerically stable matrix factorization, existing for every matrix including non-square and rank-deficient ones. Truncating to the largest singular values gives the provably best low-rank approximation in the least squares sense (Eckart-Young), which is the mathematical engine behind PCA, latent semantic analysis and most matrix-based compression.

**When to use it.**
- Computing PCA in a numerically stable way, avoiding formation of the covariance matrix entirely.
- Low-rank approximation and compression of matrices, including image compression and model reduction.
- Recommender systems and collaborative filtering, where latent factors of users and items emerge from the factorization.
- Latent semantic analysis of document-term matrices, extracting topics as latent dimensions.
- Solving ill-conditioned or underdetermined linear systems via the pseudoinverse, with regularization by truncating small singular values.
- Determining numerical rank and conditioning of a matrix, where the singular value spectrum reveals near-degeneracy that other diagnostics miss.
- Noise filtering in signal processing and structural modal analysis.

**When not to use it.** Very large sparse matrices where dense SVD is infeasible (use truncated iterative methods such as Lanczos, randomized SVD, or specialized sparse solvers); when non-negativity or interpretability of factors is required (use non-negative matrix factorization); when the data have missing entries, where plain SVD is undefined and matrix completion methods such as alternating least squares are needed.

**Assumptions and failure modes.** No distributional assumptions, which is a major strength, but the least squares optimality of truncation is a statement about Frobenius and spectral norm error, not about capturing what is scientifically important. Singular vectors are determined only up to sign, and are non-unique when singular values are tied or nearly tied, making them unstable under small data perturbations. Computational cost is cubic in the smaller dimension for dense matrices, which motivates randomized approximations at scale. Centring matters: SVD of uncentred data is not PCA, a distinction routinely blurred in practice.

**Where it is used.** Numerical linear algebra throughout scientific computing, recommender systems (the Netflix Prize made this famous), natural language processing (latent semantic indexing, embedding analysis), image compression and denoising, signal subspace methods in radar and array processing, seismology, structural dynamics, genomics.

---

## 20. Method Selection Quick Reference

| If the question is | Start with |
| --- | --- |
| Is there an effect, with controlled error rates | Hypothesis testing, confidence intervals |
| How large is the effect, and how certain am I | Confidence intervals, Bayesian posterior, bootstrap |
| Which of several models should I use | AIC/BIC, cross-validation, Bayes factors, BMA |
| Did X cause Y, from experimental data | Potential outcomes, randomization tests |
| Did X cause Y, from observational data | DiD, IV, RD, propensity methods, do-calculus for the adjustment set |
| What drives my model's output | Sensitivity analysis, Sobol indices, tornado diagrams |
| How uncertain is my model's output | Monte Carlo, LHS, QMC, uncertainty quantification |
| How bad can it get | Extreme value theory, quantile regression, copulas for joint extremes |
| What state is the system in right now | Kalman filter, particle filter, HMM, data assimilation |
| What regime is the system in, and when did it change | HMM, change point detection, bifurcation analysis |
| What should I do next, repeatedly, under uncertainty | MDP, POMDP |
| Is this series forecastable at all | Entropy measures, Hurst exponent, spectral analysis, forecastability index |
| How far ahead can anything be predicted | Lyapunov exponents, ensemble spread |
| What frequencies or cycles are present | FFT, Welch, multitaper |
| When did those frequencies occur | STFT, wavelets, Hilbert transform |
| Will this controlled system stay safe and stable | Lyapunov theory, barrier functions, ISS, contraction analysis |
| How do I get honest prediction intervals from a black box | Conformal prediction |
| How do I compress or simplify many correlated variables | PCA, SVD |
| How do I interpolate across space with uncertainty | Kriging, variogram analysis |
| How do I model a queue or resource-constrained process | Discrete event simulation, CTMC |

---