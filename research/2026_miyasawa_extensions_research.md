# The Prior Implicit in a Denoiser: Miyasawa's Theorem, Its Extensions, and What You Can Actually Build

**Author:** Nikolas Markou · **Date:** 2026-07-05 · **Status:** Research note (self-contained)

> **One-sentence thesis.** Any neural network trained to remove Gaussian noise has, as an unavoidable mathematical side-effect, also learned the *gradient of the log-probability of natural images* — and that single fact lets you reuse one trained denoiser, with **no retraining**, as a generative model and as a universal solver for a large family of image-reconstruction problems.

This note goes past reciting the papers. It (1) derives the core theorem from scratch, (2) dissects what the two anchor papers actually contribute, (3) maps the whole extension web onto one object, (4) reports **empirical probes run against this repo's own bias-free denoisers** that confirm what is true and expose what is overstated, and (5) hands you a concrete, ranked build list keyed to assets already in `dl_techniques`.

---

## Table of Contents

1. [The 90-second version](#1-the-90-second-version)
2. [Miyasawa/Tweedie, derived from scratch](#2-miyasawatweedie-derived-from-scratch)
3. [Why "bias-free" is the load-bearing trick](#3-why-bias-free-is-the-load-bearing-trick)
4. [What the two anchor papers actually contribute](#4-what-the-two-anchor-papers-actually-contribute)
5. [The extension web collapses onto one object](#5-the-extension-web-collapses-onto-one-object)
6. [What we measured on *your* denoisers](#6-what-we-measured-on-your-denoisers)
7. [The solvable-problem boundary map](#7-the-solvable-problem-boundary-map)
8. [The null-space law: the deepest practical insight](#8-the-null-space-law-the-deepest-practical-insight)
9. [Extension taxonomy — a ranked build list](#9-extension-taxonomy--a-ranked-build-list)
10. [What is true vs what is overstated](#10-what-is-true-vs-what-is-overstated)
11. [Limitations and honest uncertainty](#11-limitations-and-honest-uncertainty)
12. [Start here: three concrete moves](#12-start-here-three-concrete-moves)
13. [Glossary and references](#13-glossary-and-references)

---

## 1. The 90-second version

Train a network `D` to map noisy images back to clean ones under additive Gaussian noise. Miyasawa's 1961 theorem says the **residual** it removes,

```
r(y) = D(y) - y = σ² · ∇_y log p(y)
```

is exactly a scaled **score** — the gradient of the log-density of noisy images. The residual points "uphill" toward more probable images. You did not train for this; it falls out of the noise model plus the fact that the optimal denoiser is the posterior mean.

Three consequences, each of which is a whole research program:

- **Generation.** Repeatedly nudge a random image uphill along the residual while annealing the noise level → you sample from the image prior. This *is* what score-based diffusion models do; the denoiser is the score network.
- **Inverse problems.** Insert the residual as the "prior gradient" in an iterative solver and you can do inpainting, compressive sensing, super-resolution, deblurring, MRI/CT undersampling — **all with one denoiser, no task-specific training** (Kadkhodaie & Simoncelli 2021).
- **Interpretability + robustness.** Make the network *bias-free* (remove every additive constant) and it becomes scale-equivariant, which makes one denoiser work across all noise levels and makes its local behaviour analyzable as an adaptive filter (Mohan et al. 2020).

**The honest caveat, which we verified experimentally on this repo's models:** the popular framing that "a denoiser secretly contains one coherent global probability model" is **overstated**. The learned residual field is *not conservative* (its Jacobian is not symmetric — measured at 5.3× and 15.4× a symmetric baseline on two different architectures), so **no global energy/log-density function actually exists**. You can *sample* and *reconstruct* with it; you cannot read off calibrated probabilities or uncertainties. Sampling still works because the annealing schedule keeps every step near a *locally* valid score.

---

## 2. Miyasawa/Tweedie, derived from scratch

This is short, exact, and worth internalizing because every extension is a variation on it.

**Setup.** Clean signal `x` drawn from an unknown prior `p(x)`. Observe `y = x + n` with `n ~ N(0, σ²I)`, independent of `x`. Let

```
p(y) = ∫ p(x) N(y; x, σ²I) dx
```

be the marginal density of the *noisy* observation (the prior blurred by the Gaussian).

**Claim (Miyasawa 1961; equivalently Tweedie, via Robbins 1956; Efron 2011).**

```
E[x | y]  =  y  +  σ² ∇_y log p(y)
```

**Proof.** Differentiate the marginal under the integral. The only `y`-dependence is in the Gaussian kernel, and `∇_y N(y; x, σ²I) = N(y; x, σ²I) · (x − y)/σ²`. So

```
∇_y p(y) = ∫ p(x) N(y; x, σ²I) · (x − y)/σ²  dx
         = (1/σ²) [ ∫ x · p(x)N(y;x,σ²I) dx  −  y ∫ p(x)N(y;x,σ²I) dx ].
```

The second integral is `p(y)`. The first is `E[x|y]·p(y)`, because the posterior is `p(x|y) = p(x)N(y;x,σ²I)/p(y)`. Hence

```
∇_y p(y) = (p(y)/σ²) ( E[x|y] − y ).
```

Divide both sides by `p(y)` and use `∇_y log p(y) = ∇_y p(y) / p(y)`:

```
∇_y log p(y) = (1/σ²)( E[x|y] − y )    ⇔    E[x|y] = y + σ² ∇_y log p(y).   ∎
```

**Read the result carefully — three points that matter for everything downstream:**

1. **The MMSE denoiser is `E[x|y]`.** The minimum-mean-squared-error estimator of `x` given `y` is exactly the posterior mean. So *any* network trained to minimize `‖D(y) − x‖²` converges (in the infinite-data, infinite-capacity limit) to `E[x|y]`, and its residual `D(y) − y` converges to `σ² ∇ log p(y)`. **You never estimate `p(x)`.** The prior enters only through the score of the *smoothed* density `p(y)`, which the optimal denoiser hands you for free. That score *is* what people mean by "the implicit prior" — nothing more, nothing less.

2. **Two conditions, both required, and nothing else.** (i) additive Gaussian noise; (ii) MMSE-optimality of `D`. No assumption on `p(x)`. This is why it is universal across image content, and why every failure mode traces back to violating (i) or (ii).

3. **`p(y)` is the *noise-blurred* prior, not `p(x)`.** As `σ → 0`, `p(y) → p(x)` and the score sharpens toward the true image manifold. This is the mathematical seed of *annealing*: solve at large `σ` (smooth, easy landscape) and walk down to small `σ` (sharp, correct landscape).

### 2.1 The exponential-family generalization (Raphan & Simoncelli 2011)

The Gaussian is not special; it is just the easiest member of the exponential family. Raphan & Simoncelli's **NEBLS** (nonparametric empirical-Bayes least squares) result gives the same *shape* of estimator — observation plus a score-correction of the marginal — for Poisson, Gamma, and general exponential-family noise:

```
x̂(y) = y + (noise-model-specific operator) ∇ log p(y)
```

For **Poisson** (photon counting) the correction is `x̂ = y + [something like] p(y+1)/p(y) − 1`; for **Gamma** (a multiplicative model) it is a ratio of shifted marginals. The upshot: **the "denoiser residual = score" machinery is not locked to Gaussian noise.** It is the principled route into **photon-limited and count-noise imaging** (low-light, fluorescence microscopy, astronomy), at the cost of re-deriving the correction and retraining per noise family. This is extension rank 3 in §9.

### 2.2 The two facts everyone conflates (keep them separate)

- **The theorem (call it H1).** Exact, for the *MMSE-optimal* denoiser. It is a fact about mathematics, true independent of any network.
- **The instantiation gap (call it H2).** A *trained* network is never exactly MMSE (finite data, finite capacity, optimization). So the score you actually recover carries error, and — as we measured (§6) — its Jacobian is not even symmetric, which the true MMSE denoiser's would be. **H1 is the ideal; H2 is the reality.** Both are true; sloppy writing treats "the identity" as if a real net satisfied it exactly. It does not.

---

## 3. Why "bias-free" is the load-bearing trick

A subtle problem: the Miyasawa identity has `σ` in it, but you would like *one* network to work at *every* noise level without being told `σ` ("blind" denoising). Mohan et al. (2020) show how architecture alone buys this.

**Construction.** Remove **every additive constant** from the network: no bias vectors in conv layers, and crucially, no mean-subtraction / additive offset in normalization (use a bias-free batch-norm variant). The result is a network that is **positively homogeneous of degree 1**:

```
D(α · y) = α · D(y)   for all α > 0.
```

**Why this is exactly the right property.** Scaling the input by `α` scales the effective noise level by `α`. Degree-1 homogeneity means the denoiser's response *co-scales*, so a network trained on one noise range extrapolates to noise levels it never saw. Mohan et al. demonstrate <1 dB PSNR loss when generalizing across a 10× noise range. Equivalently: the same residual can be read as a valid score at every `σ`, which is precisely what a blind denoiser / diffusion score network needs.

**A second gift: the Jacobian is a filter.** Because a bias-free net is locally linear (degree-1 homogeneous ⇒ `D(y) ≈ J(y)·y` near `y`), the Jacobian `J(y) = ∂D/∂y` acts as a **data-adaptive linear filter**. Its eigenvectors/eigenvalues describe which local image structures are preserved vs suppressed — Mohan et al.'s interpretability claim. (We partially undercut this in §6: the Jacobian is real and analyzable *locally*, but it is **not** the Hessian of any global energy, because it is not symmetric.)

**The caveat we found empirically (§6).** "Bias-free ⇒ exact homogeneity" holds only if the network is *genuinely* additive-constant-free. Per-input normalizations that look scale-invariant (e.g. LayerNorm) can silently break it. In this repo, the checkpoint trained with `BiasFreeBatchNorm` is machine-exactly homogeneous; sibling ConvUNeXt checkpoints show up to 14% homogeneity error. **Verify it per checkpoint; do not assume it.**

---

## 4. What the two anchor papers actually contribute

### 4.1 Kadkhodaie & Simoncelli 2021 — the *usage* machinery

*"Solving linear inverse problems using the prior implicit in a denoiser"* (NeurIPS 2021; arXiv:2007.13640). Two algorithms, both consuming only `r(y) = D(y) − y`:

**(a) A sampler.** Coarse-to-fine stochastic gradient ascent (annealed Langevin): start from noise, repeatedly step along the residual with an injected stochastic term, and shrink the effective `σ` over time. In this repo it is `DenoiserPriorSampler` in `src/applications/bias_free_denoiser/samplers.py`. Notably, the implementation *re-estimates* the noise level each iteration from `‖r(y)‖` rather than dividing by a known `σ²` — the method is **self-calibrating**: the residual magnitude *is* the local noise estimate.

**(b) A linear-inverse solver.** For a measurement `y_m = M x` (M = mask, blur, subsampling, projection…), the per-step update **splits into two orthogonal pieces**:

```
d_t = (I − M⁺M) · f(y)          ← NULL-SPACE term: the prior fills what M cannot see
      + M⁺ (y_m − M y)          ← RANGE-SPACE term: hard data-consistency, denoiser-INDEPENDENT
```

(`M⁺` = pseudo-inverse; see `samplers.py:288–294`.) This decomposition is the key to the whole method's generality **and** its limits — see §8. One denoiser solves inpainting, CS, SR, deblur, MRI with **no task-specific training**, because the task only changes `M`.

### 4.2 Mohan, Kadkhodaie, Simoncelli & Fernandez-Granda 2020 — the *enabler*

*"Robust and interpretable blind image denoising via bias-free CNNs"* (ICLR 2020; arXiv:1906.05478). The bias-free construction of §3: degree-1 homogeneity → blind cross-`σ` generalization; local Jacobian → adaptive-filter interpretability. Without this, the residual-as-score reading degrades as you move away from the training noise level, and the "one net, all `σ`" property that makes (4.1) practical evaporates.

**How they fit together:** Miyasawa says *the residual is a score*; Mohan says *make it bias-free so that score is legible at all noise levels*; Kadkhodaie–Simoncelli says *anneal and project to turn that score into a sampler and an inverse-problem solver*.

---

## 5. The extension web collapses onto one object

Once you accept `r(y) = D(y) − y ≈ σ² ∇ log p(y)`, a sprawling literature becomes **one object — a score network — viewed through different wrappers:**

| Line of work | Relationship to the identity |
|---|---|
| **Score-based diffusion** — NCSN (Song & Ermon 2019), DDPM (Ho et al. 2020), score-SDE (Song et al. 2021) | The denoiser *is* the score model. Denoising-score-matching ≡ Tweedie. The Kadkhodaie–Simoncelli sampler is one discretization of the reverse SDE. **Diffusion models are Miyasawa denoisers wrapped in a noise schedule.** |
| **Plug-and-Play (Venkatakrishnan 2013); RED (Romano–Elad–Milanfar 2017)** | Use the denoiser as the "prior" proximal operator in a splitting/optimization loop. **Their convergence theorems require a *conservative* (symmetric-Jacobian) denoiser** — which real nets lack (§6), so the guarantees do not transfer even though the methods often work. |
| **Self-supervised: SURE (Metzler; Soltanayev & Chun 2018), Noise2Noise (Lehtinen 2018), Noise2Score (Kim & Ye 2021), Noise2Self/Void** | Stein's lemma / Tweedie let you train or validate a denoiser (hence a score) **without clean data** — the same object, estimated from noisy observations only. Noise2Score uses Tweedie *explicitly* and covers Gaussian/Poisson/Gamma. |
| **Diffusion posterior sampling — DPS/MCG (Chung & Ye 2022)** | Generalizes the *linear* solver to *nonlinear* forward operators `g(x)` via linearization, for phase retrieval, nonlinear deblur, etc. Breaks the clean null/range split and re-opens the conservativeness question. |
| **Exponential-family Tweedie (Raphan & Simoncelli 2011)** | Extends the identity itself off the Gaussian to Poisson/Gamma (§2.1). |
| **Conservativeness critique (Chao et al. 2023; Horvat & Pfister)** | Shows learned scores are generically non-conservative (have curl), so no exact energy exists — but sampling survives because the curl is roughly orthogonal to the annealed trajectory. This is the theoretical backing for our empirical §6 finding. |

**The single most useful reframing:** your existing bias-free denoiser is already a diffusion score network. Everything in the modern diffusion toolbox (fast SDE/ODE solvers, guidance, distillation) is, in principle, a drop-in — see extension rank 1 in §9.

---

## 6. What we measured on *your* denoisers

Rather than take the papers on faith, we ran probes on this repo's trained checkpoints (ConvUNeXt and CliffordUNet bias-free denoisers, DIV2K-validation patches, GPU-serial, finite-difference Jacobian-vector products). Three findings, one of which corrects a natural expectation.

### 6.1 Homogeneity is real — and machine-exact — but checkpoint-specific
On the deployed `20260701` `BiasFreeBatchNorm` checkpoint, the relative homogeneity error `‖D(αy) − αD(y)‖ / ‖αD(y)‖` was **0.000 (machine precision) for α ∈ {0.25, 0.5, 2, 4, 8}** — including α=8, far above the trained noise range. A deliberately bias-broken control fired at 0.75, validating the probe. **But** three sibling ConvUNeXt checkpoints showed small, α-growing deviations up to ~0.14. **Conclusion:** the enabling property is genuine (confirms the Mohan mechanism) but is a property of *truly* bias-free training, not of the architecture family. Re-verify per checkpoint.

### 6.2 The "coherent global prior" is overstated — the field is non-conservative
A residual field is the gradient of a scalar log-density **iff** its Jacobian is symmetric. We measured Jacobian asymmetry (average `|uᵀJv − vᵀJu|` over random directions, against a symmetric-blur baseline):

| Checkpoint | Jacobian asymmetry | vs symmetric baseline |
|---|---|---|
| ConvUNeXt (`20260701`) | 0.677 | **5.3×** |
| CliffordUNet (`20260703`) | 1.19 | **15.4×** (independent architecture) |

Both are far above baseline, and the finding **replicates across two architectures** — so it is not a one-model artifact. **There is no global energy/log-density.** The "implicit prior" is a *locally valid score field*, not a probability model you can integrate, normalize, or read calibrated uncertainty from. Sampling and reconstruction still work (the curl is roughly orthogonal to the annealed descent path), but **RED/PnP convergence guarantees do not transfer**, and any claim of calibrated posterior uncertainty is unlicensed.

**Exact confirmation (2026-07-05 follow-up).** The random-directional estimates above were later upgraded to an *exact* computation: the full **local Jacobian block** (12×12×3 = 432 dims, co-located input/output patch inside a 256×256 image, built by finite differences — which sidesteps the reverse-mode autodiff bug entirely) on the homogeneous ConvUNeXt checkpoint gives asymmetry `||J−J^T||/||J|| = 0.58` versus a **box-blur baseline of 0.0001 on the identical extraction — ~7,400×**. The near-zero baseline proves the measurement is clean (a genuinely symmetric operator reads as symmetric), and asymmetry in a co-located block is *sufficient* to prove the global field non-conservative. This is now airtight, not an estimate.

### 6.3 The prior does real work in inverse problems — but task-dependently
On 50%-random-pixel inpainting, we ablated the solver's two terms and measured reconstruction on the **masked pixels** (the null space, where the prior is the only source of information):

| Configuration | Masked-pixel reconstruction |
|---|---|
| Prior term only | strong (prior carries the null space) |
| Measurement-consistency only | weak (~16% of achievable gain) |
| Full solver | full |

→ **~84% of the achievable null-space reconstruction is attributable to the prior term**, ~16% to measurement-consistency. **Important honesty caveat (this drove a mid-analysis correction):** 50% random masking is the *maximally prior-favorable* task — at masked pixels the measurement term is *definitionally* zero-information, so this 84% is one **extreme** of a task-dependent curve, not a universal "the prior does 84% of the work." See §8.

### 6.4 The denoiser is a soft low-rank projector (resolves the manifold question)
The exact local Jacobian (§6.2's 432-dim block) also settles the geometry-vs-probability question that an earlier *unconverged forward-only* probe had left open (and had tentatively — wrongly — read as "no low-rank structure"):

| Quantity | Value | Reading |
|---|---|---|
| Stable rank `‖J‖_F²/‖J‖₂²` | **7.7 / 432 (2%)** | strongly low-rank |
| Participation ratio (singular values) | 16.8 / 432 (4%) | ~10–17 effective directions |
| Top-5 singular values | 0.98, 0.86, 0.84, 0.77, 0.69 | a few preserved modes… |
| Median singular value | 0.028 | …the other ~90% crushed |
| Symmetric-part eigenvalues | 392 near 0, ~40 mid/high, 2 slightly negative, in [−0.06, 0.97] | projection-*like*, nearly PSD, not a hard projector |

So **locally the denoiser preserves ~10 dominant modes and suppresses the rest** — a *soft projection onto a low-dimensional local subspace* (the signal-manifold tangent), empirically confirming Mohan's "Jacobian-as-adaptive-filter" reading. Combined with §6.2, the local operator ≈ **(soft low-rank shrinkage onto a ~10-dim subspace) + (a rotational/curl component)**: the shrinkage is *why it denoises*; the curl is *exactly what makes the field non-conservative*. The manifold-geometry account and the "no global prior" caveat are two faces of the same operator, not competing explanations. (Caveat: this is one block/point/checkpoint; the low-rank is a *local* property — the global manifold dimension is larger — but the non-conservativeness conclusion is global.)

---

## 7. The solvable-problem boundary map

This is the actionable core: **what you can actually solve, and how much to trust it.**

### ✅ EXACT / STRONG — use the machinery as-is, high confidence
- **Additive-Gaussian linear inverse problems with a large null space:** inpainting, compressive sensing, random-pixel/mask recovery, MRI/CT undersampling. (Empirically anchored: prior does ~84% of the work at 50% masking.)
- **Unconditional sampling / generation** from the implicit prior via annealed Langevin — no retraining of an existing bias-free denoiser.

### 🟡 APPROXIMATE / EXTENDABLE — usable with weaker guarantees or a known extension path
- **Mild deblur, low-factor super-resolution:** *measurement-dominated* (small null space, §8). Still helped by the prior, but do not expect an 84%-style split. (Not independently re-measured here — flagged gap.)
- **Non-Gaussian exponential-family noise (Poisson/Gamma):** genuine closed-form extension via generalized Tweedie (§2.1). Needs re-derivation + retraining. → low-light, microscopy, photon-limited imaging.
- **Nonlinear inverse problems (phase retrieval, nonlinear deblur):** via DPS-style posterior guidance. Highest cost — breaks the clean null/range split, re-opens conservativeness.

### 🔴 HARD / BOUNDARY — needs a fundamentally different construction
- **Multiplicative noise:** repo-verified (`multiplicative_miyasawa.py`) that **no clean `D(y)−y = σ²∇log p` identity exists** — only the Monte-Carlo relations A (exact, different form) and B (small-σ approximation). Do not expect the additive machinery to transfer.
- **Anything needing a true global prior, exact likelihood, or calibrated posterior/uncertainty:** blocked by the confirmed non-conservative Jacobian (§6.2). No global log-density to calibrate against.
- **Strongly ill-conditioned or full-rank `M` (near-zero null space):** by the null-space law, the prior contributes almost nothing; the method degenerates to regularized least squares.

---

## 8. The null-space law: the deepest practical insight

The single most transferable finding, from inverting the solver's structure and confirming one endpoint empirically:

> **The denoiser-prior's contribution scales with the null-space dimension of the measurement operator `M`.**

Because the solver update is `(I − M⁺M) f(y)` **[prior, acts only in the null space of M]** `+ M⁺(y_m − My)` **[data, pins the range space]**, the prior can only ever influence what the measurement *cannot see*:

- **Large null space** (inpainting, CS, heavy subsampling, MRI): lots is unmeasured → **prior-dominated** → this machinery shines.
- **Small null space** (mild deblur, 2× SR): almost everything is measured → **measurement-dominated** → the prior is a garnish; a classical regularizer may do nearly as well.

A sensitivity analysis over the credit-split ranked the **null-space fraction of `M` as ~3.8× more influential than which checkpoint / how good the denoiser is.** Practical rule of thumb: **before reaching for denoiser-prior methods, ask how much of your signal the measurement operator actually destroys.** The more it destroys, the more this approach earns its keep. (Caveat: the exact functional form is a single-anchor extrapolation; trust the *ranking*, not the precise numbers.)

---

## 9. Extension taxonomy — a ranked build list

Each row: what new capability, which repo asset to build on, the exactness cost/failure mode, and the effort.

| # | New capability | Build on | Exactness cost / failure mode | Effort |
|---|---|---|---|---|
| **1** | **Treat the denoiser as a diffusion score net** — plug an existing bias-free checkpoint into a modern SDE/ODE solver, guidance, or distillation | `samplers.py::DenoiserPriorSampler`; `bfcnn`/`bfunet` checkpoints | None beyond the H1/H2 gap — this is a *reframing*, not a new approximation | **Cheap** — swap the schedule for a published solver, no retraining |
| **2** | **Decide which convergence theory legitimately applies** (PnP-Langevin vs RED) by testing conservativeness | `hutchinson_divergence` in `multiplicative_miyasawa.py` (already coded) | None — verification. *Already done this session: non-conservative, so RED/PnP guarantees do NOT transfer* | **Cheap** (done) |
| **3** | **Generalized-Tweedie Poisson/Gamma** → photon-limited / low-light / microscopy imaging | relation-A/B scaffolding as a template | Re-derive correction + retrain per noise family; additive curriculum doesn't auto-transfer | **New training** |
| **4** | **SURE / Noise2Score self-supervision** → train/validate with no clean references (real sensor, medical) | `additive_sure_risk` (validated on a linear toy) | SURE variance is high for complex nonlinear nets (our Probe C confirms the caution) | **Moderate** |
| **5** | **DPS-style nonlinear posterior sampling** (phase retrieval, nonlinear deblur) | `LinearInverseProblemSolver` null/range split as template | Highest — the clean split is linear-only; needs JVP linearization, re-opens conservativeness | **New implementation** |
| **6** | **Free per-pixel uncertainty** from the shrinkage/divergence field | `_compute_denoiser_residual` | "shrinkage magnitude ≈ error" is a plausibility claim, unproven here | **Cheap ablation** |
| **7** | **Jacobian top-eigenvectors = local image-manifold tangent** (operationalize the geometric reading) | JVP/Hutchinson machinery, or exact finite-difference local Jacobian (§6.4) | Local-linearity only; local (not global) rank | **Cheap ablation — done** (§6.4: local stable rank ~2%; use finite differences, not autodiff) |
| **8** | **RED/PnP with a conservativeness-repaired denoiser** — add a Jacobian-symmetry penalty at train time to *earn* the guarantees | training loop + validated Hutchinson probe as a regularizer | May trade PSNR for guarantee-eligibility; untested | **New training** |
| **9** | **Full DDPM/score-SDE pipeline bridge** using existing checkpoints | rank-1 equivalence + checkpoints | Surfaces score-quality gaps (H2) more visibly than the light K&S sampler | **Moderate** |

*Cross-cutting method (not a standalone row):* apply **control-theory stability analysis** (gain margins, saturation-aware Lyapunov bounds) to the sampler's numerical scaffolding — the `_adaptive_step_schedule` stack currently uses 7 hand-tuned clip/cap/floor interventions that would benefit from principled bounds instead of ad-hoc engineering.

---

## 10. What is true vs what is overstated

| Claim | Verdict | Basis |
|---|---|---|
| Residual = scaled score (additive Gaussian, MMSE denoiser) | **TRUE** (exact-in-idealization) | Miyasawa theorem (§2) |
| Bias-free ⇒ exact degree-1 homogeneity | **TRUE but checkpoint-specific** | machine-exact on one checkpoint; up to 0.14 error on siblings (§6.1) |
| A single trained net = a coherent global prior / energy | **OVERSTATED** | non-conservative Jacobian, 5.3× & 15.4×, two architectures (§6.2) |
| Sampling / reconstruction still work despite the above | **TRUE** | prior-only inpainting reconstructs well; curl ⟂ trajectory (Chao 2023) |
| RED/PnP convergence guarantees apply to a learned denoiser | **FALSE — do not transfer** | require conservativeness, which is absent (§6.2) |
| "The prior does 84% of the work" (as a universal) | **OVERSTATED** | true only for large-null-space tasks; task-dependent law (§8) |
| Efficacy is *primarily* manifold-projection geometry | **PARTLY TRUE (locally)** | exact local Jacobian is soft-low-rank: stable rank ~2%, ~10 preserved modes (§6.4) — geometry and "no global prior" are two faces of one operator |
| One denoiser solves *all* linear inverse problems | **TRUE with an asterisk** | true operationally; but the *prior's share* varies by task null space (§8) |

---

## 11. Limitations and honest uncertainty

1. **No calibrated uncertainty is licensed.** The non-conservative field means every "solved" result is a point-estimate / sampling-quality claim, never a calibrated-posterior claim. Treat this as a permanent guardrail for any product claim.
2. **Homogeneity is checkpoint-specific.** Re-verify `D(αy)=αD(y)` before relying on cross-`σ` generalization for a given checkpoint.
3. **The 84% credit-split is one task point** (maximally prior-favorable); the small-null-space endpoint (deblur/SR) was not independently re-measured, and the interpolating law is a single-anchor extrapolation — trust the ranking, not the numbers.
4. **The manifold question (geometry vs probability) is now resolved locally** (§6.4): an exact finite-difference local Jacobian shows the denoiser is a soft low-rank projector (stable rank ~2%), so the geometric account is *locally* correct and coexists with the non-conservative "no global prior" finding. The reverse-mode autodiff route is still blocked by a Keras/TF-2.18 jit-conv instability, but finite differences sidestep it. Residual caveat: this is a *local* rank at one point/checkpoint; the *global* manifold dimension is larger and not measured here.
5. **Single-repo, largely single-domain (natural-image / DIV2K) empirical base.** Cross-domain generality is asserted from the literature, not independently verified here.
6. **Multiplicative-noise boundary** is a verified negative result: no clean identity, only Monte-Carlo relations A/B.

---

## 12. Start here: three concrete moves

1. **Cheapest high-value win (extension #1):** wire an existing bias-free checkpoint into a published diffusion SDE/ODE solver as a drop-in score network and benchmark against the current K&S loop. No retraining; immediately unlocks the modern sampling toolbox.
2. **Biggest genuinely-new capability (extension #3):** implement the generalized-Tweedie Poisson/Gamma correction for low-light / microscopy. This *extends the solvable boundary* rather than re-packaging it.
3. **Close the last honest gap (cheap):** re-run the credit-split ablation on a *measurement-dominated* task (mild deblur) to complete the null-space law empirically. (The Jacobian geometry-vs-probability gap is now closed — §6.4 — via an exact finite-difference local Jacobian; no need to fight the autodiff blocker.)

And one **guardrail:** the non-conservative Jacobian is not a bug to fix casually — it is a property of learned scores. Never ship a calibrated-uncertainty claim derived from this denoiser-prior. If calibrated UQ is a hard requirement, that is extension #8 (train a conservativeness-repaired net and re-verify with the Hutchinson probe), not a free read-off.

---

## 13. Glossary and references

**Glossary.**
- **Score** — `∇_y log p(y)`, the gradient of the log-density; points toward higher-probability regions.
- **MMSE denoiser** — minimizes mean-squared error; equals the posterior mean `E[x|y]`.
- **Empirical Bayes** — estimate the needed functional of an *unknown* prior directly from data (here, the marginal's score off the optimal denoiser) without ever parameterizing `p(x)`.
- **Degree-1 homogeneity** — `D(αy)=αD(y)`; the scale-equivariance that a bias-free net enjoys.
- **Conservative field** — a vector field that is the gradient of a scalar potential ⇔ its Jacobian is symmetric. A learned residual field generally is **not**.
- **Null space of `M`** — the set of signal components the measurement operator cannot see; where the prior does all the work.
- **Annealed Langevin** — noisy gradient ascent on `log p` with a decreasing noise schedule; equivalently, reverse-diffusion sampling.

**Primary references (all verified, HTTP-200 / peer-reviewed):**
- Miyasawa, K. (1961). *An empirical Bayes estimator of the mean of a normal population.* Bull. ISI 38(4):181–188.
- Robbins, H. (1956). *An empirical Bayes approach to statistics.* 3rd Berkeley Symp.
- Efron, B. (2011). *Tweedie's formula and selection bias.* JASA 106(496):1602–1614.
- Raphan, M. & Simoncelli, E. P. (2011). *Least squares estimation without priors or supervision.* Neural Computation 23(2):374–420.
- Mohan, Kadkhodaie, Simoncelli & Fernandez-Granda (2020). *Robust and interpretable blind image denoising via bias-free CNNs.* ICLR. arXiv:1906.05478.
- Kadkhodaie, Z. & Simoncelli, E. P. (2021). *Solving/Stochastic solutions for linear inverse problems using the prior implicit in a denoiser.* NeurIPS. arXiv:2007.13640.
- Romano, Elad & Milanfar (2017). *Regularization by Denoising (RED).* SIAM J. Imaging Sci. 10(4):1804–1844.
- Venkatakrishnan, Bouman & Wohlberg (2013). *Plug-and-Play priors for model-based reconstruction.* IEEE GlobalSIP.
- Song, Y. & Ermon, S. (2019). *Generative modeling by estimating gradients of the data distribution (NCSN).* NeurIPS. arXiv:1907.05600.
- Ho, Jain & Abbeel (2020). *Denoising diffusion probabilistic models (DDPM).* NeurIPS. arXiv:2006.11239.
- Song et al. (2021). *Score-based generative modeling through SDEs.* ICLR. arXiv:2011.13456.
- Kim, K. & Ye, J. C. (2021). *Noise2Score: Tweedie's approach to self-supervised denoising.* NeurIPS. arXiv:2106.07009.
- Lehtinen et al. (2018). *Noise2Noise.* ICML. arXiv:1803.04189.
- Soltanayev, S. & Chun, S. Y. (2018). *Training deep denoisers without ground truth (SURE).* NeurIPS.
- Chung, H. & Ye, J. C. (2022). *Diffusion posterior sampling (DPS).* arXiv:2209.14687.
- Chao et al. (2023). *On investigating the conservative property of score-based generative models.* ICML. arXiv:2209.12753.

**Repo assets referenced:** `src/applications/bias_free_denoiser/samplers.py` · `src/dl_techniques/utils/multiplicative_miyasawa.py` · `src/dl_techniques/models/bias_free_denoisers/{bfcnn,bfunet,bfconvunext,bfcliffordunet}.py` · `src/dl_techniques/layers/{bias_free_conv2d.py, norms/bias_free_batch_norm.py}` · `src/dl_techniques/callbacks/noise_sigma_curriculum.py` · `src/train/bfunet/eval_psnr_vs_noise.py`.

---

*Provenance: produced via a COMPREHENSIVE epistemic-deconstruction session (framing → domain grounding → scope audit → boundary map → abductive expansion → causal analysis with GPU probes → parametric quantification → synthesis → validation). Empirical claims in §6 were measured on this repo's checkpoints; full audit trail, 15-hypothesis Bayesian ledger, and probe scripts under `analyses/analysis_2026-07-04_cefd1d1d/`.*
