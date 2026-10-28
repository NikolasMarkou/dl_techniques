# General-Purpose Fisher Information Measurement for Deep Neural Networks

**Status:** Plan (not yet implemented)
**Date:** 2026-10-23
**Source theory:** arXiv:2609.36262v1 — "Understanding LLM Parameter Update Sparsity through the Lens of Fisher" (Zhang, Mukherjee, Peng; UIUC)
**Scope:** A general-purpose tool measuring diagonal model Fisher information for any Keras 3 model — not restricted to LLMs under RL training.

---

## 1. Motivation and Goal

Recent work has observed that parameter changes during post-training (RL, on-policy distillation, SFT on near-policy data) concentrate in a small subset of coordinates. The cited paper establishes the **diagonal model Fisher** as an objective-independent quantity that explains and predicts this concentration:

- Small Fisher ⇒ small expected gradient (via a Cauchy–Schwarz bound that holds across SFT, RL, and OPD alike).
- High-Fisher coordinate sets, fixed at initialization, retain most of full-training improvement when used as training masks (85.2% of improvement with 0.1% of parameters trainable in their RL experiments).
- Low Fisher in on-policy training arises because high-probability next tokens tend to have **similar logit sensitivities** per parameter.

The goal here is a **library-grade tool** — one that measures the diagonal Fisher of an arbitrary Keras model (vision, tabular, time-series, language, generative) against its own output distribution, exposes the token-level decomposition where applicable, verifies the Fisher–gradient bound when an objective signal is supplied, and integrates with the existing `dl_techniques.analyzer` framework.

This is deliberately broader than the paper: the paper measures Fisher for autoregressive LLMs under RL/OPD; this tool measures Fisher for **any differentiable probabilistic model**.

---

## 2. Theoretical Foundation

Throughout, θ ∈ ℝ^P denotes the model parameters, x a prompt/input drawn from a fixed input distribution Dₓ (independent of θ), and πθ(y|x) the model's response/output distribution.

### 2.1 Diagonal model Fisher (paper Eq. 2)

```
Fᵢ(θ) = Eₓ∼Dₓ  E_y∼πθ(·|x) [ (∂ᵢ log πθ(y|x))² ]
```

Define the **score** hᵢ(x,y) = ∂ᵢ log πθ(y|x). Two facts make this quantity tractable and useful:

1. **Normalization identity:** E_y∼πθ(·|x)[hᵢ(x,y)] = ∂ᵢ Σ_y πθ(y|x) = 0.
   Consequence: for any θ-independent baseline b(x), E[b(x)·hᵢ] = 0. Reward centering in RL is legitimate *because of this identity*.
2. **Fisher as variance:** Fᵢ = E[hᵢ²] = Var[hᵢ] (mean is zero).

Interpretation: Fᵢ measures how sensitive the model's **own** response distribution is to a local perturbation of θᵢ. It depends only on the model and Dₓ — not on any reward, teacher, or demonstration signal. This is what makes it an objective-independent indicator of where learning *can* concentrate.

### 2.2 The common Fisher bound (paper Prop. 1, Eqs. 3–4)

Write any post-training objective as J(θ) = Eₓ∼Dₓ E_y∼μ(·|x)[Rθ(x,y)] with different

| Method | Response distribution μ | Effective signal a(x,y) | Squared scale E[a²] |
|---|---|---|---|
| **SFT** | ν(y|x) (demonstrations) | wν(x,y) − 1, where wν = ν(y|x)/πθ(y|x) | E[ (wν − 1)² ] |
| **On-policy RL** | πθ(y|x) | r(x,y) − r̄(x), r̄ = Eπθ[r|x] | Eₓ[Varπθ(r|x)] |
| **PG-style OPD** | πθ(y|x) | RT(x,y) − R̄T(x) | Eₓ[Varπθ(RT|x)] |
| **Full-vocab OPD** | πθ(y|x) (rollouts held fixed for differentiation) | AT(x,y) = Σt b_t, b_t = q_t(y_t)/p_t(y_t) − 1 | E[ Σt χ²(q_t ‖ p_t) ] |

All four give gᵢ = Eπθ[a · hᵢ], and Cauchy–Schwarz yields:

```
|gᵢ| ≤ √E[a(x,y)²] · √Fᵢ
```

**The bound's role in the tool.** Fisher alone bounds the gradient only up to the objective-dependent scale factor. The tool therefore (a) always reports Fisher as the primary objective-independent quantity, and (b) optionally computes an objective signal a(x,y) alongside, to turn the bound into a per-parameter upper envelope and to measure gradient-sup­port recall.

Full-vocabulary OPD derivation detail (paper Appendix A.1, worth carrying into the implementation):

```
At a sampled prefix s_t with pt(a) = pθ(a|s_t), qt(a) = πT(a|s_t), qt ≪ pt:

gᵢ = Eπθ[ Σt Σa qt(a) ∂ᵢ log pt(a) ]

Define b_t = qt(y_t)/pt(y_t) − 1 and AT = Σt b_t.
Both b_t and ξ_t,i = ∂ᵢ log pt(y_t) have zero conditional mean given history.
The cross terms E[b_t · ξ_u,i] vanish for t ≠ u by the tower property.
Eπ[AT · hᵢ] = Eπ[Σt b_t ξ_t,i] = gᵢ.
The signal variance decomposes as Eπ[AT²] = Eπ[Σt χ²(qt ‖ pt)].
```

### 2.3 Token-level decomposition (paper Prop. 2, Eqs. 8–9)

For an autoregressive response y = (y₁,…,y_T) with prefix s_t = (x, y_<t), token score ξt,i = ∂ᵢ log pθ(y_t|s_t), sequence score hᵢ = Σt ξt,i.

**Chain 1 (sequence = sum of token Fisher):**

```
Fᵢ(θ) = E[ Σt fᵢ(s_t) ]
```

Proof skeleton (martingale argument): E[ξt,i | st] = 0 under on-policy sampling, so token scores form a martingale difference sequence; for u < t, ξu,i is determined by st, so E[ξu,i ξt,i] = 0 by the tower property and all cross terms vanish:

```
Fᵢ = E[(Σt ξt,i)²] = E[Σt ξt,i²] + 2·Σ_{u<t} E[ξu,i ξt,i] = E[Σt fᵢ(s_t)]
```

**Critical caveat:** the cross-term cancellation holds **in expectation, not per sample**. A per-sample identity `hᵢ(x,y)² = Σt ξt,i²` is FALSE in general. The tool must never collapse these: sequence-level Fisher averages h² over samples; token-level Fisher averages Σt fᵢ(st). They estimate the same quantity but through different estimators, with different variances.

**Chain 2 (token Fisher = variance of logit sensitivities):**

Let za(s) be the logit of token a at prefix s, pa = e^{za}/Σ_b e^{zb}, and jᵢ,a(s) = ∂ᵢ za(s) the **logit sensitivity**. Differentiating the log-softmax:

```
∂ᵢ log pa = jᵢ,a − Σ_b pb jᵢ,b
```

so with j̄ᵢ = Σ_a pa jᵢ,a:

```
fᵢ(s) = E_p[(∂ᵢ log pa)²] = Σ_a pa (jᵢ,a − j̄ᵢ)² = Var_{a∼p}[jᵢ,a(s)]
   (9)  = ½ E_{a,b∼iid p}[(jᵢ,a − jᵢ,b)²]
```

**Mechanistic reading:** a parameter has low Fisher when it perturbs the logits of *likely* tokens **coherently** (they all move together — e.g. "a direction that raises the preference for all plausible liquids") rather than differentially. Fisher is small when the model's probability mass sits on tokens whose sensitivities agree. This is the tool's diagnostic for *why* a parameter is insensitive, not just whether it is.

### 2.4 Estimators (paper App. B; these are the tool's reference recipes)

With N_F Fisher samples (one response each) and N_g gradient prompts (G responses each), r̄ₙ = (1/G)Σⱼ rₙⱼ:

```
F̂ᵢ = (1/N_F) Σₙ hᵢ(xₙ, yₙ)²
ĝᵢ = (1/(N_g·G)) Σₙ Σⱼ (rₙⱼ − r̄ₙ) · hᵢ(xₙ, yₙⱼ)
```

Estimator semantics — several details here are load-bearing and easy to get wrong:

- **No length normalization.** Neither estimator divides by sampled response length. Token counts cancel in reciprocals only if you also change the objective's accounting; the paper deliberately does not.
- **Fisher squares each trajectory's score before averaging** — hᵢ is the *sum over tokens* of the token score, then squared. Averaging Σt (ξt,i)² over trajectories estimates E[Σt fᵢ(st)] per 2.3 and is a *different estimator* of the same quantity; do not mix variance accounting between them.
- **The gradient estimator averages with the sign intact** and ranks coordinates only afterwards by |ĝᵢ|.
- **Ties use a fixed coordinate order** when selecting top-K_q sets; K_q = ⌊q·P/100⌋ over P unique coordinates.
- **The RL gradient in the paper's Dr. GRPO setup omits a common fixed-horizon factor** (1/4096), which shifts all coordinates' scale but not their ranking — the tool should log such deliberate omissions.
- **Stopping tokens count.** Observed EOS/im_end tokens contribute their score; length-truncated responses are neither discarded nor assigned artificial EOS scores.

### 2.5 Gradient-support recall (paper Eq. 7)

For selection percentage q: S_F(q) = K_q largest-F̂ᵢ coordinates, Sg(q) = K_q largest-|ĝᵢ| coordinates,

```
Recall(q) = 100 · |S_F(q) ∩ Sg(q)| / K_q
```

A uniformly random subset of the same size has expected recall ≈ q. Recall ≫ q is the empirical content of the Fisher–gradient connection (paper Fig. 3a: e.g. top-1% Fisher captures 47.4% of the top-1% gradient coordinates at init, vs 1% for random).

### 2.6 Persistence metrics (paper §4.2, Eqs. 12–13)

Between an initial policy and a checkpoint k:

```
Retention_k^low(q)  = 100 · |L₀(q) ∩ L_k(q)| / (P − K_q)
Retention_k^high(q) = 100 · |H₀(q) ∩ H_k(q)| / K_q
Mₖ(q)/M₀(q)         = meanF̂_{k,i over i∈L₀(q)} / meanF̂_{0,i over i∈L₀(q)}
```

with Hₖ(q) the high-Fisher set and Lₖ(q) the complement. Mk/M0 near 1 means initially-low coordinates stay low in magnitude, not merely in rank. Paper findings: high-set retention 72–78% (random baseline = q), low-set magnitude within 15–17% of initial by end of training — the initial Fisher pattern is essentially stationary in their regime. **Caveat:** this stationarity is empirical, regime-limited evidence, not a theorem; the tool reports the diagnostics rather than assuming persistence.

### 2.7 Distribution-model Fisher (the "model Fisher" distinction)

The paper's "model Fisher" uses responses sampled from πθ. A classical **empirical Fisher** uses labels/teacher responses with squared *gradients of the loss* — a different estimator, biased when the sampled distribution is not the model's. This tool implements the model Fisher of Eq. 2; when an SFT signal is configured, the score is still hᵢ(x,y) = ∂ᵢ log πθ(y|x) with y drawn from ν — this is the paper's SFT row (a = wν − 1 = ν/πθ − 1), NOT an empirical Fisher. The distinction matters numerically: Eπ[(wν−1)²] = Eₓ[χ²(ν‖πθ)], which blows up when ν puts mass where πθ ≈ 0 — see §8 for the implementation clamp on that.

---

## 3. Architecture — Where This Lives in `dl_techniques`

The tool is a new member of `src/dl_techniques/analyzer/`, following that package's `BaseAnalyzer` / `BaseVisualizer` patterns and orchestrator contract.

```
src/dl_techniques/analyzer/
├── fisher_metrics.py                    # NEW — pure computation functions
├── analyzers/fisher_analyzer.py         # NEW — FisherAnalyzer(BaseAnalyzer)
├── visualizers/fisher_visualizer.py     # NEW — FisherVisualizer(BaseVisualizer)
├── config.py                            # + fisher_* fields on AnalysisConfig
├── data_types.py                        # + fisher_* fields on AnalysisResults
├── constants.py                         # + FisherMetricNames, ANALYSIS_TYPE_FISHER
├── model_analyzer.py                    # + registration (see §10)
tests/test_analyzer/
├── test_fisher_metrics.py               # NEW
├── test_fisher_analyzer.py              # NEW
├── test_fisher_gradient_bound.py        # NEW
├── test_fisher_visualizer.py            # NEW
```

Design principles taken from what this repo already learned the hard way (each traces to a shipped decision anchor in `analyzer/`):

1. **Artifacts arrive via return values, not parked on `self`** (D-020) — every intermediate result is returned from the private method that computes it.
2. **One generator per analysis pass** (D-031 / F-009) — a single `np.random.Generator` is built at `analyze()` entry and threaded through every stochastic site (prompt sampling, on-policy decoding, pair sampling). Per-model or per-layer generator reconstruction from the same seed hands every unit the identical draw stream.
3. **NaN/None, never 0.0, for not-computed values** (D-036 / D-003 / D-019) — a failed Fisher row publishes NaN or None; 0.0 is a value real parameters take.
4. **Budget sized once, shared across models** (D-037 / D-022) — the sample budget and memory probe are computed for ALL models before any per-model loop, so cross-model Fisher comparisons are never confounded by different batch/sample budgets.
5. **Serialization is a read, not a write** (F-040) — `save_results` deep-copies containers before pruning; a second save must find the results intact.
6. **Eager mode for capture.** Fisher computation needs concrete per-sample scores; under `model.predict` a wrapped/gradients path receives symbolic tensors. The Fisher path runs eager `model(x, training=False)`-style calls under a `tf.GradientTape` (TF backend present as the numeric substrate; `keras.ops` for all tensor code so the module stays backend-agnostic at the API level).

### 3.1 Scope guard: what this is not

- **Not an optimizer.** No natural-gradient, no KFAC preconditioning (future work §13).
- **Not a pruner.** Fisher *masks* can be exported for downstream training experiments (§12.2), but the analyzer does not itself run training.
- **Not an empirical-Fisher estimator.** Distinction established in §2.7; if empirical Fisher is ever wanted it gets its own module and its own config flag, never a shared flag.

---

## 4. `fisher_metrics.py` — Complete API and Algorithms

### 4.1 Data classes

```python
@dataclass
class FisherConfig:
    """Configuration for Fisher computation. Mirrors AnalysisConfig.fisher_* fields."""
    n_samples: int = 100             # N_F: number of inputs (Fisher LL samples)
    n_responses: int = 1             # G: responses per input (gradient estimator; Fisher uses 1 per sample)
    token_level: bool = True         # token-level decomposition for sequence models
    compute_gradients: bool = False  # compute ĝᵢ for Fisher–gradient analysis
    signal_type: str = "rl"          # "rl" | "sft" | "opd_pg" | "opd_full" | "none"
    n_gradient_prompts: int = 100    # N_g (gradient estimator needs its own samples)
    max_tokens: int = 512            # cap on response length for score computation
    n_positions_per_response: int = 8        # token-level analysis positions (paper B.3)
    n_token_pairs_per_position: int = 8       # pairwise logit-sensitivity samples (B.3: 12,800 total = 200×8×8)
    batch_size: int = 8              # microbatch of inputs per session run
    accumulate_dtype: str = "float32"  # accumulation dtype ("float32" | "float64")
    compute_dtype: str = "float32"      # model-invocation dtype ("float32" | "bfloat16")
    seed: Optional[int] = None
    memory_limit_mb: Optional[int] = 2048
    selection_percentages: Tuple[float, ...] = (0.05, 0.1, 0.5, 1, 5)  # q grid for recall/retention
    zero_fisher_threshold: float = 1e-12    # below this a coordinate is treated as Fisher-dead for diagnostics
```

```python
@dataclass
class FisherOutput:
    """Everything one model's Fisher pass produces. Returned, never parked on self."""
    # Core Fisher: parallel to model.trainable_variables
    fisher_flat: np.ndarray                    # shape (P,), float32/64 — F̂ᵢ in a canonical flat coordinate order
    param_slices: List[Tuple[str, str]]        # per variable: (variable_path, slice_in_flat) for reconstruction
    shapes: List[Tuple[str, Tuple[int, ...], str]]  # (variable_path, shape, dtype) per trainable variable
    n_samples: int
    skipped_samples: int                       # count of samples dropped for non-finite scores
    standard_errors: Optional[np.ndarray]      # per-coordinate SE(F̂ᵢ), Welford
    # Gradient arm
    gradient_flat: Optional[np.ndarray]        # shape (P,) — ĝᵢ (sign intact)
    signal_scale: Optional[float]              # √E[a²] estimate (per Prop. 1)
    gradient_recall: Optional[Dict[float, float]]     # q → Recall(q)
    # Token-level arm (sequence models only)
    token_fisher_mean: Optional[np.ndarray]    # mean over sampled prefixes, per parameter f̄ᵢ
    token_fisher_positions: Optional[Dict[int, np.ndarray]]  # position-within-response → per-param token Fisher (aggregate)
    uniform_reference: Optional[np.ndarray]    # f̄ᵢ^unif under Eq. 11 — Real/Uniform machinery
    real_uniform_ratio: Optional[float]        # E[f̄ᵢ]/E[f̄ᵢ^unif] aggregate diagnostics
    # Sequence-level (per-model) aggregates
    metrics: Dict[str, float]                  # summary scalars (see §4.5)
    samples_used: List[Any]                    # provenance (prompt ids, seeds) for reproducibility
```

### 4.2 Canonical flat coordinate order

P unique coordinates require a canonical flattening: variables in `model.trainable_variables` order, C-order flatten of each variable, concatenated. `param_slices` records `(variable_path, (start, stop))` so per-variable Fisher tensors can be reconstructed and joined to layer names. Ties in ranking use this order — implementing the paper's "ties use a fixed coordinate order."

### 4.3 Core function: `compute_diagonal_fisher(model, data_source, config, rng)`

```
Algorithm (eager, per microbatch of inputs):
  trainable_vars = model.trainable_variables           # P_total = Σ |v| over vars
  running = {mean: zeros(P), var_acc: zeros(P), count}  # Welford accumulators in accumulate_dtype

  for each microbatch X (size ≤ config.batch_size):
    1. E_y sampling: y_batch ← sample_responses(model, X, config)   # §4.4
    2. Score: h ← sequence_score(model, X, y_batch)                  # one backward per sample, see below
    3. If any non-finite coordinate of h: drop that whole sample's vector,
       skipped_samples += 1, log (per §8 stability-3; never clip silently)
    4. Welford-update running with h² per sample
  fisher_flat = running.mean
  standard_errors = sqrt(running.var / n_used)     # sample variance of h², not of h²'s mean squared twice
```

**The one-backprop-per-sample trick (this makes the tool feasible).** Computing scores hᵢ for ALL P parameters costs **one** backward pass per sample: differentiate the scalar `log πθ(y|x)` w.r.t. all `trainable_variables` at once; gradient tensors concatenated in the canonical order are exactly h. This is O(N) backwards, not O(P·N). Token-level sensitivity pairs work the same way on `(z_a − z_b)/2` scalars.

**Sequence score.** For an autoregressive model and response y = (y₁..y_T), s_t = (x,y_<t):

```
seq_loglik = Σ_{t=1..T_valid} log pθ(y_t | s_t)
h = ∇_θ seq_loglik
```

T_valid includes observed stopping tokens (EOS/⌊im_end⌋) and EXCLUDES padding and any positions past `max_tokens` (see §5, caveat on truncation). Practically: run the forward with `y` teacher-forced (shifted-right labels), read token log-probs at gold positions, sum, tape-backward. Generation for *sampling* y under πθ is §4.4; the score computation is always teacher-forced over an already-fixed y — sampling is NOT differentiated through, per the score-function (REINFORCE) semantics of Prop. 1.

**Single-output (classification/regression) score.** For a classifier with logits z = fθ(x), sampled class y:

```
seq_loglik analog = log_softmax(z)[y]          # = z_y − logsumexp(z)
h = ∇_θ log_softmax(z)[y]                       # the exact ∂ᵢ log p of Eq. 2
```

For regression with Gaussian likelihood N(fθ(x), σ²) and observed y:

```
score = −(y − fθ(x))ᵀ ∇_θ fθ(x) / σ²            # const terms cancel inside the gradient
```

σ² comes from config when set (`fisher_regression_variance`), else from the model's stated output_scale; the analyzer refuses to guess silently — see §8 numerical-4.

### 4.4 Function: `sample_responses(model, X, config, rng)` — on-policy responses

For non-sequence models, "response sampling" is:

- classification: draw class indices ~ Categorical(softmax(z)) — temperature 1 condition; a config `fisher_sampling_temperature` rescales logits z/T before the softmax (paper always uses T=1, p=1, no top-k for Fisher samples).
- regression: y = fθ(x) (the deterministic mean; distribution-Fisher of a Gaussian is taken about the mean).

For sequence models, autoregressive decode per prompt until EOS or `max_tokens`, temperature 1, top-p 1, **no top-k truncation** (paper B.1 applies exactly these settings; truncation biases Fisher because blocking low-prob tails removes log-prob regimes from the sample). The generated response is retained (not re-sampled) for its score computation.

Prompt/response length defaults follow the paper's experiment geometry (1024/4096 for RL) but are config-driven; nothing here is model-specific.

### 4.5 Function: `compute_gradients(model, data_source, rewards, config, rng)` — objective-signal arm

Implements the Table-2 signals with the score identity E[hᵢ]=0 used to justify centering:

```
ĝᵢ = (1/(N_g·G)) Σₙ Σⱼ a(xₙ,yₙⱼ) · hᵢ(xₙ, yₙⱼ)
```

Signals:
- **rl:** a = r(x,y) − r̄ₙ (per-prompt group mean over the G responses — Dr. GRPO-style, no std normalization)
- **sft:** a = wν − 1 where wν = ν(y|x)/πθ(y|x). NOTE both densities evaluated at the SAME y, and y is drawn from ν here (the demonstration), NOT resampled on-policy. Requires the caller to supply ν(y|x) (or ν-probabilities for sequences); the tool computes πθ(y|x) itself. χ²-clamp — §8 stability-1.
- **opd_pg:** a = RT − R̄T where RT = Σt log[πT(y_t|s_t)/pθ(y_t|s_t)], teacher fixed, y on-policy from student. Teacher log-probs never receive gradients (stop-gradient through the teacher model).
- **opd_full:** a = AT = Σt (q_t(y_t)/p_t(y_t) − 1) with y_t sampled from the student p_t and q_t = πT. Per-token contributions accumulate the χ² along the trajectory, same as `opd_pg` structurally but with the per-token pair weighting of App. A.1; the practical loop is the same, the only difference is the intermediate aggregation shape (per-token rather than summed-then-centered).

`signal_scale` = √E[a²] estimated over the same samples — closing the other side of the bound |ĝᵢ| ≤ signal_scale·√F̂ᵢ per-parameter. A per-parameter check emits the violation fraction (§12.3).

`gradient_recall(q)` per §2.5 over the q grid, with `random baseline = q` published alongside every curve.

**Disjoint sample sets.** Following paper B.1: Fisher prompts and gradient prompts are sampled **without replacement from one source, disjoint and fixed across policies**. The tool enforces the disjointness at construction time of its sample plan, not by convention.

### 4.6 Function: `compute_token_fisher(model, prefixes, config, rng)` — Real/Uniform diagnostic

Implements §2.3 and paper §4.1's Real-vs-Uniform machinery exactly:

```
For each sampled position (n_positions_per_response per response):
  prefix s (model's own tokens up to that position)
  a) forward: logits z = fθ(s)                                   # V logits
  b) p = softmax(z)                                              # Real distribution
  c) pairs_pa: n_token_pairs_per_position iid pairs (a,b) ~ p    # with replacement
     For each pair: scalar_loss = (z_a − z_b) / 2
        grads = ∇_θ scalar_loss                                  # ONE backward per pair
        per-param: g²ᵢ = gradsᵢ²                                  # unbiased single-pair estimator of... see below
        D_𝒢(s,a,b) = mean over group 𝒢 of g²ᵢ                    # paper B.3 formula
     Token Fisher ≈ mean over pairs of D(s,a,b)                  # estimating ½E[(j_a−j_b)²]
  d) uniforms: same prefixes, pairs ~ Uniform over full vocab    # Eq. 11 reference
```

Correctness subtleties baked into the implementation:

- **The single-pair estimator is unbiased per §2.3's Eq. 9**: E_{a,b iid p}[½(jᵢ,a−jᵢ,b)²] = Var_p[jᵢ] exactly. Averaging squared half-difference gradients over pairs estimates token Fisher consistent with Eq. 9, and — since ½E[(j_a−j_b)²] = fᵢ(s) — the Real side of the Real/Uniform comparison is apples-to-apples across conditions.
- **Pairs are shared** across parameter groups within a condition (paper B.3: "all six groups share each condition's pairs and backward calculations") — the tool computes one backward per pair and grouped means over its slices, not one backward per group.
- **Equal-token pairs (a == b) contribute 0 and REMAIN in the sample count** (paper B.3) — dropping them biases the variance upward; the count denominator includes them.
- **Positions are fixed before measurement** and are not filtered by correctness (B.3) — the tool samples positions uniformly per response and stores them in `samples_used` provenance.
- **Meaning differs from §2.4's sequence-Fisher**: the token-level estimator averages token Fisher over prefixes (B.3: "estimates mean token Fisher over sampled positions, not the unnormalized sequence Fisher used in Test 1"). Both live in `FisherOutput` under different names, with a docstring cross-reference; conflating them in one field is a documented-analyzer-ambiguity class of bug.
- **The Uniform reference draws from the FULL vocabulary including special tokens** (B.3), not from a censored support.

This function carries the paper's central mechanistic claim into tooling: publishing `real_uniform_ratio < 1` per parameter group is the *explanation* side — high-probability tokens have similar sensitivities — which is the reason Fisher is low on-policy. The tool also exposes the group-select variant: per-parameter-group (e.g., current low-Fisher group L_q(q) by % cuts) mean token Fisher under p vs Uniform, matching paper Table 3's layout.

### 4.7 Function: `compute_persistence(ref_output, checkpoint_outputs, config)`

Pure post-processing on `FisherOutput`s measured at ≥ 2 policies (initial checkpoint + later ones). Computes §2.6 Retention low/high over the q grid and Mk/M0 for fixed initial low-Fisher sets. Also the per-model random retention baseline (q for high, ≈q for low — the paper's Random rows use the analytic 0.05/0.1/… /99.95/99.9/… columns).

This is where the stationarity claim becomes testable tool output rather than an assumption.

### 4.8 What the module deliberately DOES NOT do

- **No weight updates.** Parameters are never modified during any Fisher computation (paper's protocol). Assert in dev mode (trainable-var snapshots hashed pre/post pass) rather than trust the protocol.
- **No mask training.** Masked-fine-tune experiments (paper §3.3) are `src/train/` work; the tool only *exports* masks (§12.2).
- **No Hessian/full Fisher/KFAC.** Diagonal only. Block-diagonal/full is future work (§13).
- **No empirical-Fisher path.** See §3.1 scope guard.

---

## 5. Sequence-Model Specifics (Accent on LLMs, but NOT only LLMs)

### 5.1 Autoregressive detection

```python
def _is_autoregressive_model(model) -> bool:
```

Heuristics, in preference order (first hit decides):
1. The model exposes a generation-style entry point (`generate`, `sample`, `decode`) callable with a prompt and returning a sequence.
2. Output rank ≥ 2 with a vocabulary-shaped last dim AND the model contains a causal-masked attention path (grep of the model's layers for causal mask wiring) — the sequence-models test oracle surface in this repo.
3. Explicit config override: `AnalysisConfig.fisher_force_sequence: Optional[bool] = None` — None = infer, True/False = force. **Override exists so new model families don't need analyzer changes.**

Non-sequence models (vision classifiers, denoiser regressors, SOM/HRM core loops, graph nets) fall through to the single-output path of §4.3 automatically. This is the "general purpose" promise: the Fisher math (Eq. 2) never assumed an LLM.

### 5.2 Length handling

- `max_tokens` caps response length for BOTH sampling and scoring (truncated responses are neither discarded nor assigned EOS scores — paper's exact rule).
- Padding positions contribute zero score (and zero token-Fisher); padding masking is applied in the teacher-forced forward via the model's own padding mask argument.
- Multi-step-termination models (SOM, output-expansion nets): termination is the model's own semantic; the tool relies on the model's generate-like entry points providing it and refuses otherwise.

### 5.3 BOS/prompt-prefix semantics

For models whose `generate` consumes a prompt prefix and emits *continuations*, the Fisher score sums log-probs of the SAMPLING range only (continuation positions), not the prompt. A config switch `fisher_score_scope: "response" | "response_plus_prompt"` controls this; default "response" matches the paper (scores are of πθ(y|x)).

### 5.4 Vocab-size regimes

Token-level analysis with full-vocab pair sampling is fine at V ≤ ~150k (a pair's backward pass is V-independent, only the logits forward is). The Real/Uniform numbers in paper B.3 (V including special tokens) are achievable at that scale. Memory joints: logits tensors of shape (B, V) at BF16 for B=512 are ~150 MB — budgeted under `memory_limit_mb` (D-022-style probe before the pass).

---

## 6. `fisher_analyzer.py` — Analyzer Class

```python
class FisherAnalyzer(BaseAnalyzer):
    """Runs the Fisher pipeline inside the ModelAnalyzer orchestration."""

    def __init__(self, models, config):  # BaseAnalyzer signature preserved

    def requires_data(self) -> bool:     # True — Fisher is an expectation, needs inputs
        return True

    def analyze(self, results, data, cache=None) -> None:
        # 1. Build ONE pass-level rng from config.random_state (F-009 invariant).
        # 2. Shared sample plan across models: prompt indices FIRST, then per-model
        #    response draws. Shared across models so cross-model Fisher comparisons are
        #    apples-to-apples on the same inputs (D-037 analog for Fisher).
        # 3. Per model:
        #    fisher_out = compute_diagonal_fisher(model, plan, config, rng)
        #    if config.fisher_compute_gradients:
        #        fisher_out += compute_gradients(...)
        #    if config.fisher_token_level and sequence model:
        #        fisher_out += compute_token_fisher(...)
        # 4. results.fisher_* population (§7); one model's failure logs and skips (F-040/
        #    D-005 pattern: genuine per-model numeric/shape failures do not abort the
        #    multi-model pass, but AttributeError/TypeError/NameError/ImportError re-raise).
        # 5. Optional persistence arm when >1 model is the same arch at different
        #    policy points (config.fisher_persistence_indices).
```

Eager-execution rule: this analyzer **cannot** be run through any traced/compiled path — the per-sample backward requires concrete tensors. It may not be combined with `jit_compile=True` model calls; enforce by calling the uncompiled wrapper or capturing eager outputs of the layer call paths.

---

## 7. Data-Type and Config Extension

### 7.1 `AnalysisResults` additions (data_types.py)

```python
# Fisher analysis
fisher_analysis: Optional[pd.DataFrame] = None
# Columns: model_name, variable_path, flat_index (canonical order), fisher_value,
#          standard_error, gradient_value (optional), gradient_rank (optional)
fisher_summary: Dict[str, Any] = field(default_factory=dict)             # cross-model aggregate
fisher_summary_per_model: Dict[str, Dict[str, Any]] = field(default_factory=dict)
# Per-model keys — see §4.5 of the metrics dict:
#   total_fisher, mean_fisher, std_fisher, n_parameters, n_samples, skipped_samples,
#   signal_scale (optional), recall_at_q (optional dict q→float), violation_rate (optional),
#   real_uniform_ratio (optional), computation_time_seconds
fisher_token_level: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
# model_name -> {"token_fisher_mean": …, "uniform_reference": …, "positions": …}
fisher_gradient_recall: Dict[str, Dict[float, float]] = field(default_factory=dict)
# model_name -> q -> Recall(q); random baseline implied = q
fisher_persistence: Optional[pd.DataFrame] = None   # only when the persistence arm ran
```

### 7.2 `AnalysisConfig` additions (config.py)

```python
# Fisher analysis options
analyze_fisher: bool = False                    # OFF by default — it's expensive
fisher_force_sequence: Optional[bool] = None    # None=infer, True/False override
fisher_n_samples: int = 100                     # N_F
fisher_n_responses: int = 1                     # G for the gradient arm
fisher_n_gradient_prompts: int = 100            # N_g
fisher_compute_gradients: bool = False
fisher_signal_type: str = "rl"                  # "rl"|"sft"|"opd_pg"|"opd_full"|"none"
fisher_token_level: bool = True
fisher_max_tokens: int = 512
fisher_n_positions_per_response: int = 8
fisher_n_token_pairs_per_position: int = 8
fisher_batch_size: int = 8
fisher_accumulate_dtype: str = "float32"
fisher_compute_dtype: str = "float32"
fisher_sampling_temperature: float = 1.0
fisher_regression_variance: Optional[float] = None
fisher_score_scope: str = "response"            # "response"|"response_plus_prompt"
fisher_selection_percentages: Optional[Tuple[float, ...]] = None
fisher_persistence_indices: Optional[Sequence[str]] = None  # model names forming initial+k
fisher_decay: Optional[float] = None            # NOT in v1 — reserved; see §10.4 decision
```

Defaults are calibrated around the paper's protocol (N_F=800 for their publication runs; 100 is the diagnostic default — §8 performance-1 discusses the accuracy/memory trade along this axis explicitly).

### 7.3 `MetricNames` / new Fisher constants (constants.py)

```python
ANALYSIS_TYPE_FISHER: str = 'fisher'
# joins DATA_REQUIRED_ANALYSES at module import

class FisherMetricNames:   # mirrors MetricNames' role for the spectral side
    TOTAL_FISHER, MEAN_FISHER, STD_FISHER, N_PARAMETERS, N_SAMPLES,
    SKIPPED_SAMPLES, SIGNAL_SCALE, GRADIENT_RECALL, REAL_UNIFORM_RATIO,
    VIOLATION_RATE, COMPUTATION_TIME, TOKEN_FISHER_MEAN, UNIFORM_REFERENCE
```

---

## 8. Caveats Register (Complete)

Every caveat this implementation must carry with it — some are math constraints, some are numerics, some are scope discipline.

### Mathematical caveats

1. **The bound is per-step and at-a-policy.** |gᵢ| ≤ √E[a²]·√Fᵢ is a statement at the current θ, not over trajectories. Accumulated parameter movement adds optimizer-dynamics terms the bound doesn't cover (paper's own Limitations ¶). The tool reports per-policy diagnostics and a persistence arm; it never *certifies* a trajectory-level claim. The fascination is with what the initial-policy Fisher implies for later steps — which the masked-training experiments (paper §3.3, our §12.2) test directly.
2. **Fisher is about πθ, not about the loss.** A coordinate with tiny Fisher is a coordinate whose *training-expected score* is small; whether it also moves little under a *given optimizer* depends on that optimizer (e.g. Adam normalizes per-coordinate second moments — the paper's Mukherjee-2026 reference shows optimizer choice changes sparsity). The tool's signal_scale term closes Prop. 1's other half and is quoted beside Fisher for exactly this reason.
3. **Cross-term cancellation is in expectation.** Per-sample cross terms DON'T vanish (§2.3 critical caveat). Sequence-Fisher (average of h²) and token-Fisher (average of Σt fᵢ) converge to the same value but have different variance and different finite-N biases. Both are published; neither is a substitution for the other; the tool's docstring names the distinction the paper does (B.3's "not the unnormalized sequence Fisher").
4. **Stationarity is empirically regime-limited.** Retention/Mk/M0 held in paper §4.2 within frames (math RL, 1k steps, on-policy). Different datasets/longer horizons/architectures are unknowns; the tool measures rather than assumes (§4.7 with random baselines published for both arms).
5. **a = None signal case.** With signal_type="none" only Fisher is computed, no bound is claimable, and `signal_scale`/violation_rate stay None (never 0.0 — D-036/D-003 analog).
6. **Pairwise equivariance, not invariance.** Token Fisher via Eq. 9's pairwise half-differencing is invariant to a PER-PARAMETER constant additive shift of jᵢ,a — which is precisely the softmax-shift invariance the paper exploits. But it is NOT invariant to per-token rescaling; the estimator measures what Eq. 9 defines. Any temptation to "normalize" j's before differencing breaks the identity; don't.

### Sequence/LLM caveats

7. **Long-sequence Fisher aggregation.** hᵢ = SUM over tokens of token scores (not mean). For very long responses (2k+), score magnitudes grow with T; Squared scores grow quadratically. The paper uses no length normalization and gets usable results at T ≤ 4096 with h in float32; the tool must accumulate in `accumulate_dtype` and refuse to silently mean-normalize scores (a change of estimator, per §2.4). Report both the raw (paper-matching) means and optionally a per-token variant under a distinct name if ever wanted.
8. **Rewards centering requires G ≥ 2 responses per prompt in RL.** With G=1, r̄ₙ = rₙⱼ and a ≡ 0 — the bound is trivially satisfied with zero information. Analyzer refuses a zerosignal configuration loudly (raise ValueError at plan building — a wrong config is a caller bug, not a measurement outcome).
9. **Sample-plan fixedness across policies.** Fisher-vs-checkpoint comparisons (persistence arm) require THE SAME prompt indices across policies (paper: "remain fixed across policies", "sampled without replacement, disjoint"). The tool's sample-plan object records and re-plays indices; a caller re-running with different prompts gets different estimator noise masquerading as dynamics drift. Enforce by construction.
10. **Disjointness (Fisher vs gradient prompts).** Reusing the same samples for both estimators introduces a correlated-noise F–g coupling the paper avoids by splitting (B.1). Enforce by construction (§12.1).

### Numerics/stability caveats

11. **log 0.** Sampling from πθ guarantees p(y|x) > 0 for observed samples (discrete support), but a bug or an externally-provided response CAN have p = 0. `log_softmax` (subtracting the logsumexp) is the required formulation; naive z−Σexp log-sum-exp is forbidden (overflow at tiny p). Any −inf token log-prob ⇒ that sample is dropped, counted in `skipped_samples`, logged — never clipped silently.
12. **χ² scale blow-up for SFT.** Eπ[(wν−1)²] = E[χ²(ν‖πθ)] diverges where πθ puts ~0 mass where ν is concentrated (§2.7). Warm-up: estimate χ² over samples first; if it exceeds a config threshold (`fisher_sft_chi2_cap`, default 1e4), log a loud recommendation to widen πθ support or raise temperature before trusting per-sample weights.
13. **Partial-tape gradient graphs.** Steps that shouldn't receive gradients: teacher log-probs (OPD), prompt tokens (default scope), reward r, gradients w.r.t. non-trainable variables — all wrapped in `tf.stop_gradient` or excluded by scoping the tape. A silent leak means Fisher for OPD absorbs teacher-prob gradients — an easy, silent, dramatic mis-measurement; test-pinned (§11 gradient-bound suite).
14. **MVU of h_i = 0 vs. h_i ≈ 0.** Fisher-dead coordinates (below `zero_fisher_threshold`) get a published count and are excluded from recall/retention campaigns (they'd inflate random-set recall denominators with coordinates whose Fisher is numerically identical to zero). The threshold is config-visible, not hardcoded-magic-in-two-places.
15. **TF32.** Process-global state hazards across analysis runs (the repo has `tf32_disabled` fixture machinery for exactly this in tests). In the analyzer at large: Fisher estimates are variance-sensitive, so the pass adapter runs with TF32 asserts OFF unless explicitly overridden per-call; NOTE this flag differs from `config.fisher_force_*` and is a per-session wrapper, mirrors `tests/numerics.py` semantics.
16. **Mixed precision.** Paper B.1: "FP32 parameters and score accumulation; BF16 computation." Tool mirrors: `compute_dtype="bfloat16"` for the forward/backward; `accumulate_dtype="float32"/"float64"` for the Fisher/gradients accumulators. Downcasting h to the accum dtype happens PER-SAMPLE, not per-batch — ratcheted accumulation from microblows.
17. **Weighted retention of gradients.** gradient tensors from the score backward are per-variable dicts; flatten into canonical order per §4.2's `param_slices` contract. Any future refactor of trainable-variable ORDER invalidates a stored canonical index mapping — version the flat order with a per-model `variable_fingerprint` (SHA over names+shapes) in the output; a fingerprint mismatch in a reloaded/continued run is a loud error, not a silent misalignment.

### Performance/memory caveats

18. **The critical efficiency trick is one backwards per sample** (§4.3). It means a 1.7B-parameter model's Fisher over 800 samples is 800 backwards — high but not exotic (approx one RL update's worth of grads. Paper's own B.1 estimates this scale). The naive per-parameter-alternative would be O(P) backwards and is completely infeasible; a future impl that forgets the one-backward structure and takes per-pair h values as scalars is a bug, and §11 pins the correct asymptotics.
19. **Token-Fisher pair sampling is one backward per PAIR.** n_token_pairs_per_position=8 means 8 backwards per position, 8 positions → 64 backwards per response. Over 200 responses = 12,800 (matching the paper's B.3 budget). Config the pair count down for fast diagnostics; the vocabulary size does NOT enter the backward cost (only the logits forward, V-shaped).
20. **Memory probe before the pass** (D-022/D-037 analog): a throwaway sample measures bytes/sample and auto-sizes `batch_size` to fit `memory_limit_mb`; logs when the batch was reduced. Both prompt-tokens and KV-cache (if the model caches during its generate) count against the budget.
21. **Parallelism of models.** Multiple models' Fisher passes are serial by design (GPU contention would contaminate memory probes and timing). This analyzer does NOT parallelize models; a caller with N models pays N× the time.

### API/scope caveats

22. **No partial training side effects.** The analyzer runs forward/backward only. A model with a stateful training-only path (batchnorm momentum, EMA teacher updates, dropout RNG) must be called with `training=False`. Assert via the D-020/D-022-style snapshot-hash dev check when `config.fisher_assert_no_weight_drift` (default True): trainable variables are hashed before and after; a drift fails loudly.
23. **Serialization versioning.** `fisher_analysis` DataFrame schema versioned (v1 = §7.1's columns); save_results bumps the RESULTS_SCHEMA_VERSION — analyzer-wide D-040 discipline. DataFrame → records on save; numpy → JSON via the existing convert_numpy (NaN → None per F-072).
24. **Memory footprint of per-param arrays.** Fisher + SE + (optional) gradient = 3 float32 arrays of length P. For P = 1.7B ≈ 6.8GB × 3 = ~20GB peak if all kept simultaneously. `save_results` by default emits per-parameter TABLES ONLY if `config.json_include_fisher_raw` (default False) — default is summary + top-k tables. Raw arrays live in memory and go to `.npz` only on explicit user request (mirrors spectral `json_include_raw_esds` D-028's lazy-emit philosophy).
25. **Backends.** `keras.ops` for all tensor math; the analyzer package has `tensorflow` as an import untouched elsewhere (the repo's "Raw TensorFlow has not been fully migrated" convention) — but the gradient-tape machinery (`tf.GradientTape`) is TF-specific and imported only where needed with a loud comment on the TF-dependency. If JAX/PyTorch backends are ever required (not currently supported by this repo's `pyproject.toml`), the tape layer becomes a backend-conditional adapter — flagged as future work, not hacked in now.

---

## 9. The Visualizer (`fisher_visualizer.py`)

```python
class FisherVisualizer(BaseVisualizer):
    """Plots for one or more models' Fisher outputs."""
```

Plots (each carefully labeled with its estimator semantics per §4.6's discipline):

1. **Fisher distribution** — log-scale histogram per model of fisher_value; optionally pervariable overlays. Marks `zero_fisher_threshold` and the mass fraction below it.
2. **Top-k Fisher coordinates** — bar/lollipop chart of top-K_q coordinates (with variable name + flattened index labels; long names truncated with a mapping note in the caption).
3. **Fisher vs |gradient| scatter** — the §2.2 bound visualization: points, the envelope signal_scale·√Fᵢ as a curve, and a violation fraction appended to the title (D-036/D-019 analog: nan/none yesterday — a failed-bound plot must visibly say so).
4. **Gradient-support recall curve** — Recall(q) vs q, random-baseline diagonal `y = q` reference line, from `results.fisher_gradient_recall`. Never publish Fisher-mask-vs-random baseline without the diagonal — an omitted baseline was the failure mode Rios/Adewuyi comparisons exist to address (cited in §2.6's paper context).
5. **Token-level heatmap** — positions × parameter-groups token-Fisher (Real) alongside the Uniform reference heatmap, with the Real/Uniform ratio distribution as an inset (paper Table 3's content in visual form).
6. **Persistence curves** — Retention_high/low(q) and Mk/M0 vs checkpoint, Random baseline curves included — only rendered when the persistence arm ran and `fisher_persistence` is populated.
7. **Per-variable Fisher summary** — one row per trainable variable: total/variance-share of Fisher, stacked into a per-model horizontal bar (feature: a layer-level Fisher concentration overview).

Header: every plot titles the estimator (sequence-Fisher vs mean-token-Fisher — never leave the B.3 ambiguity implicit), the sample sizes N_F/N_g/G, and the schema of the underlying arrays.

---

## 10. ModelAnalyzer Integration Steps

### 10.1 `model_analyzer.py` changes

1. Constant: `ANALYSIS_TYPE_FISHER: str = 'fisher'` with the other type constants.
2. `DATA_REQUIRED_ANALYSES: Set[str]` gains `'fisher'` (permitting `analyze()`'s existing validation path to run for Fisher like the info-flow/calibration paths).
3. Import `FisherAnalyzer` and `FisherVisualizer` in the imports block.
4. `_init_analyzers()` gains `ANALYSIS_TYPE_FISHER: FisherAnalyzer(self.models, self.config)`.
5. `analyze()`'s config-driven set construction gains `ANALYSIS_TYPE_FISHER if self.config.analyze_fisher else None` in the None-discard set (existing idiom).
6. `_create_visualizations()`'s class map gains `ANALYSIS_TYPE_FISHER: FisherVisualizer`.
7. `save_results()` extends `results_dict` with:
   - `fisher_summary`, `fisher_summary_per_model`, `fisher_gradient_recall` unconditionally when populated;
   - `fisher_analysis` (DataFrame → records) when populated — full per-parameter table: emitted only if `config.json_include_fisher_raw` else summarized (§8-24);
   - `fisher_token_level` and `fisher_persistence` likewise gated.
8. `_serialize_training_metrics` untouched (orthogonal).

### 10.2 Multi-input/multi-output handling

- Fisher's input contract matches `DataInput`'s. Multi-input models flow through existing `DataSampler` normalization. NOTE the existing analyzer warning about limited multi-input support applies and is logged on the Fisher path too (consistent behavior).
- Multi-output models: the score is composed over OUTPUTS only if the model defines a joint order; for a model whose outputs are multi-headed likelihood terms the caller supplies `fisher_output_logprob_fn` — a callable returning the scalar log-likelihood of an (x,y) pair. Analyzer raises a clear error when a multi-output model doesn't provide one, rather than silently scoring output 0 only.

### 10.3 `__init__.py` public surface

`dl_techniques.analyzer`'s `__init__` exports gain `FisherAnalyzer`, `FisherVisualizer`, `FisherConfig`, `FisherOutput`, and a version bump `1.1.0 → 1.2.0` (mirroring the current `1.1.0` versioning discipline; the README of the analyzer package documents the surface).

### 10.4 Reserved-but-not-implemented

`fisher_decay`, natural-gradient plumbing, KFAC — documented as future work (§13) and NOT shipping as dead knobs (the repo's dead-knob discipline pins this; `AnalysisConfig` carries NO knob the analyzer doesn't honor — F-028/D-029's exact lesson).

---

## 11. Test Plan

### 11.1 `tests/test_analyzer/test_fisher_metrics.py`

Sentence-named tests in this repo's discipline, each with its anti-vacuity arm where applicable:

- `test_fisher_estimates_score_squared_mean` — Gaussian mean/precision toy: analytic Fisher vs estimate within MC tolerance `atol=1e-6, rtol=0` per the repo's default precision-discipline (this test pins the accuracy claim of the estimator; run under `tf32_disabled` + CPU pinned via `golden_reference_device`).
- `test_score_normalization_zero_mean` — E[hᵢ] ≈ 0 (uses §2.1's identity as an *oracle assertion* on the implementation, above threshold).
- `test_the_pairwise_half_difference_matches_variance` — Eq. 9's two forms; MC convergence check.
- `test_token_fisher_martingale_cross_terms_average_out` — synthetic 2-step chain where cross-term E ≠ 0 per sample; assert expectation-level equality to Σt fᵢ to MC precision. This pins §2.3's critical caveat AT THE IMPLEMENTATION LEVEL (a naive per-sample crossover would fail this).
- `test_gradient_recall_exceeds_random_baseline` — model built with a designed gradient-Fisher alignment (winner coordinate); recall(q) at q=1% exceeds 1% by a wide margin (anti-vacuity: if random recall also ~1%, the test is vacuous). Uses paper Fig. 3a's 47%/1% disparity as its design target — pins the paper's qualitative claim at small scale.
- `test_signal_scale_closes_the_bound` — RL signal; violation_rate ≤ MC noise level; asserted explicitly (not just ≥0).
- `test_sft_chi2_blowup_is_flagged` — ν concentrates where πθ ~ 0; the χ² cap logs the recommendation (§8-12) and does NOT crash, and the result is flagged in metrics.
- `test_skipped_samples_are_counted_not_clipped` — NaN score injection; skipped_samples == expected, fisher equals the finite-only mean (not zero-inflated).
- `test_no_weight_drift_assertion_fires` — flip a variable during computation; `fisher_assert_no_weight_drift` raises. Mutation-pins §8-22.
- `test_a_one_response_rl_configuration_raises` — §8-8's ValueError contract.
- `test_fingerprint_mismatch_is_loud` — variable-fingerprint mismatch on replay raises, not silently remaps (§8-17).
- `test_reproducibility_with_fixed_seed` — two identical passes, byte-equal fisher_flat; ALSO byte-equal skipped counts. Seed threading is the F-009 discipline pinned at the tool level.
- `test_rng_is_advancing_across_stringsof_calls` — same seed, DIFFERENT subsequent stochastic draws (anti-vacuity against a per-call reconstructed generator — F-009's exact defect class, pinned here).
- `test_batch_size_respects_memory_probe` — huge dummy activations; batch auto-resized per logging, not crash.

### 11.2 `tests/test_analyzer/test_fisher_analyzer.py`

- `test_requires_data_is_true` (contract).
- `test_fisher_populates_results_fields` — `fisher_analysis` columns, summary keys; NaN-else-float types.
- `test_a_failed_model_skips_others` — one broken model, second model still populated (D-005/F-040's re-raise discipline: only genuine per-model numeric errors skip; AttributeError classes re-raise).
- `test_serialization_roundtrip` — `save_results` → JSON load; fisher keys present; NaN became None per F-072; DataFrame→records round-trip; raw gate honored (`json_include_fisher_raw=False` → no per-parameter rows).
- `test_version_bumped_in_dunder` — `__version__ >= 1.2.0` (§10.3).
- `test_config_flags_have_one_home` — NO config field goes unread (mirrors the F-028 lesson; a dataclass introspection meta-test asserting each `fisher_*` field is referenced by the analyzer or metrics module).
- `test_sample_plan_fixed_and_disjoint` — Fisher and gradient prompt indices ARE disjoint; replaying the same plan yields same indices (§8-9, §8-10).

### 11.3 `tests/test_analyzer/test_fisher_gradient_bound.py`

Regression-style, paper-following test suite on a small designed model (2-layer MLP classifier, ~10⁴ params):
- an RL-signal-endowed synthetic dataset; `signal_type="rl"`; measure F̂, ĝ, signal_scale on a shared plan.
- Assert the bound holds, per-parameter, up to `atol=1e-6, rtol=0` slack scaled by the MC standard error (a Bayesian tolerance — reproducibility here is tolerance-derived, matching `tests/numerics.py::reassociation_atol` philosophy: bounds DERIVED from a noise source, not pasted).
- Assert recall(q) > q for every q in the small grid.
- Assert persistence arm (2 checkpoints of the same arch: pre/post a few optimizer steps) gives Retention_high > Random at q=1% (the stationarity claim, at toy scale).

### 11.4 `tests/test_analyzer/test_fisher_visualizer.py`

- Each plot renders a `.png` under `tmp_path` and returns a fig handle; no plot crashes on NaN-era (D-019-class states) data.
- Titles/naming carry estimator labels (test asserts the label strings — the B.3 discipline pinned, not aeroplanes).
- `test_visualizer_skips_absent_arms_gracefully` — visualizer with no persistence data doesn't crash or render a persistence placeholder frame.

### 11.5 What the tests deliberately DO NOT cover (scope)

- Paper-scale LLM replication (a 1.7B-parameter Qwen RL run — that's `src/train/` integration work, §12; the analyzer's numeric correctness is pinned at toy scale).
- Full suite integration at `/results/` — the repo's autouse guard forbids writes into the repo-root results path; all test artifacts go to `tmp_path` (pytest's own tmp discipline per the root AGENTS.md).

---

## 12. Supporting Infrastructure (also shipping)

### 12.1 `SamplePlan` object (in `fisher_metrics.py`)

A serializable record of the prompt/reward/response indices, seeds, and disjointness partition used for a Fisher pass. Enables:
- cross-policy replay (persistence arm needs the same plan);
- cross-model comparison (same prompts across models);
- provenance in `FisherOutput.samples_used`.

### 12.2 `fisher_mask_export(output, q)` — training-bridge utility

Returns `Dict[str, np.ndarray]` of boolean masks (per variable, per shape) selecting the top-K_q Fisher coordinates — the exact artifact the paper's §3.3 masked-training experiments need. Consumable by a future `src/train/*/` masked-fine-tune utility or any `load_weights`-style pass. The mask producer lives HERE, in the analyzer package (it's an output of analysis); the mask CONSUMER (the actual masked training loop) is `src/train/` work, out of scope of this plan.

### 12.3 Analyzer-versioned summary metrics names

Every metric in §7.1's summary dict is `FisherMetricNames`-keyed, not ad-hoc strings — mirroring how `MetricNames` stabilized the spectral surface against drift (the single home, no lockstep copies).

### 12.4 Docstrings

Match the analyzer package's existing style (Google-style majority for `analyzer/` — but this is a NEW deeply-mathematical module; the paper's derivations demand literal math blocks in the docstrings: §2's equations reproduced across the relevant function docstrings, the correct pattern already used in `spectral_metrics.py`). No wholesale reformat of existing files, per the repo's docstring convention.

---

## 13. Deliberate Non-Goals and Future Work

1. **Full Fisher / KFAC.** Diagonal only; the paper's framework is diagonal throughout. A KFAC facility is a separate tool (natural gradient, block-Fisher, low-rank + diagonal factorizations).
2. **Natural-gradient optimizer.** Out of scope; Fisher as optimization machinery is a different project.
3. **Empirical Fisher.** §3.1's scope guard; a different estimator with a different bias profile, never sharing a flag with model Fisher.
4. **Online/adaptive batch sizing during the pass.** Fixed plan, fixed batch after the memory probe. Adaptive sampling is a research project (importance-weighted Fisher estimators exist for this).
5. **Non-Keras backends (JAX/PyTorch).** Blocked on the repo's dependency surface (`keras.ops` is the API surface; the tape is TF-backed). Which is a future backend-adapter refactor, explicitly out of scope.
6. **Paper-scale replication of Fig. 3 / Tables 3, 4.** The tool ships with the machinery; running Qwen3-scale experiments is `src/train/` work driven by a separate experiment plan that USES this analyzer's API — a follow-up once the tool lands.

---

## 14. Effort Estimate

| Phase | Deliverable | Est. |
|---|---|---|
| 1 | `fisher_metrics.py` — F, score, token, persistence, SamplePlan | 3–4 d |
| 2 | `fisher_analyzer.py` — BaseAnalyzer integration, plan/replay, skip/re-raise discipline | 2–3 d |
| 3 | Type/config/constants extension + ModelAnalyzer wiring | 1–2 d |
| 4 | `fisher_visualizer.py` — 7 plot types | 1–2 d |
| 5 | `fisher_mask_export` + `SamplePlan` serialization | 1 d |
| 6 | Test suite (§11) — numerics, contracts, mutation-pins | 3 d |
| 7 | Docs (analyzer README/API section, usage examples) + analyzer version bump | 1 d |

**Total: ~12–16 working days**, with phase 1 (the estimator math) the highest-uncertainty item — the correct one-backward-per-sample structure and the score-function semantics must be proven at toy scale before integration.

---

## 15. One-Paragraph Summary

This plan specifies a **`FisherAnalyzer`** for `dl_techniques.analyzer` that measures the diagonal model Fisher of any Keras 3 model — LLM or not — by Monte Carlo: sample inputs from the shared plan, sample responses from the model's own distribution (on-policy decode for sequence models, class sampling for classifiers, mean response for regressors), compute the score h = ∇θ log πθ(y|x) with **one backwards pass per sample** (not per-parameter), accumulate h² via Welford into F̂ᵢ with standard errors, and — where configured — compute objective-signal gradients ĝᵢ under the paper's Table-2 signals (RL/SFT/OPD), verifying Prop. 1's Cauchy–Schwarz envelope and measuring Fisher-vs-gradient gradient-support recall against the random baseline. For autoregressive models it adds the paper's token-level machinery: paired logit-sensitivity averaging (Eq. 9's ½E[(j_a − j_b)²] form) and the Real-vs-Uniform comparison (Eq. 11) that establishes *why* Fisher is low on-policy. It also adds a persistence arm (retention and magnitude-ratio diagnostics across checkpoints, with random baselines) and exports Fisher-selected masks for downstream masked-training experiments. Every estimator detail that matters is pinned by tests — the martingale cancellations, the pairwise-variance equivalence, the F/A-disjointness, the one-backward structure, the no-weight-drift contract, fixed-seed reproducibility with an advancing generator, and the NaN/skip semantics — following this repo's sentence-named, anti-vacuity, decision-anchored test discipline.
