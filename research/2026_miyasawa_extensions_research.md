# The Prior Implicit in a Denoiser

## A self-contained guide to Miyasawa's theorem, what it licenses, what it does not, and how to build on it

**Author:** Nikolas Markou
**Status:** Research guide, self-contained
**Scope:** Theory derived from scratch, measured properties of real trained networks, a validated success-prediction framework, and a concrete engineering path from a single trained denoiser to a general-purpose restoration system.

---

## Thesis

Any neural network trained to remove additive Gaussian noise has, as an unavoidable mathematical side effect, also learned the gradient of the log-probability of its training distribution. That single fact lets you reuse one trained denoiser, with no retraining, as a generative model and as a universal solver for a large family of image reconstruction problems.

Two further facts, both established by direct measurement rather than inherited from the literature, determine whether you can actually ship anything built this way:

1. **A bias-free denoiser is exactly degree-1 homogeneous, and this makes a blind denoiser satisfy the noise-conditional contract that modern diffusion solvers require.** No retraining. This is the highest-value property in the whole construction.
2. **A directly-parameterized learned denoiser is neither conservative nor passive.** No global log-density exists, no calibrated uncertainty is licensed, and the convergence theorems of Plug-and-Play and Regularization by Denoising do not apply. This is fixable, exactly, by changing how you parameterize the network. Section 14 gives the construction.

---

## Table of contents

**Part I. Theory**
1. [The ninety-second version](#1-the-ninety-second-version)
2. [Miyasawa and Tweedie, derived from scratch](#2-miyasawa-and-tweedie-derived-from-scratch)
3. [Beyond Gaussian: the exponential-family generalization](#3-beyond-gaussian-the-exponential-family-generalization)
4. [Why bias-free is the load-bearing trick](#4-why-bias-free-is-the-load-bearing-trick)
5. [The noise-conditional bridge](#5-the-noise-conditional-bridge)
6. [The linear inverse solver and its null/range split](#6-the-linear-inverse-solver-and-its-nullrange-split)
7. [The literature collapses onto one object](#7-the-literature-collapses-onto-one-object)

**Part II. Measured properties of real networks**
8. [Homogeneity is exact, and it is norm-dependent](#8-homogeneity-is-exact-and-it-is-norm-dependent)
9. [The field is not conservative and not passive](#9-the-field-is-not-conservative-and-not-passive)
10. [Local Jacobian geometry](#10-local-jacobian-geometry)
11. [How much work the prior actually does](#11-how-much-work-the-prior-actually-does)
12. [Domain transfer behaves opposite to expectation](#12-domain-transfer-behaves-opposite-to-expectation)

**Part III. Predicting success**
13. [Three hard gates, then one continuous variable](#13-three-hard-gates-then-one-continuous-variable)

**Part IV. Building**
14. [Making the denoiser conservative](#14-making-the-denoiser-conservative)
15. [Making it passive and nonexpansive](#15-making-it-passive-and-nonexpansive)
16. [The verification protocol](#16-the-verification-protocol)
17. [Architecture for general-purpose restoration](#17-architecture-for-general-purpose-restoration)
18. [Safe and unsafe architectural modifications](#18-safe-and-unsafe-architectural-modifications)
19. [Cost control](#19-cost-control)
20. [Ranked build list](#20-ranked-build-list)

**Part V. Reference**
21. [True, overstated, and false: the ledger](#21-true-overstated-and-false-the-ledger)
22. [Limitations and honest uncertainty](#22-limitations-and-honest-uncertainty)
23. [Glossary](#23-glossary)
24. [Citation index](#24-citation-index)

---

# Part I. Theory

## 1. The ninety-second version

Train a network `D` to map noisy images back to clean ones under additive Gaussian noise. Miyasawa's 1961 theorem [1] says the **residual** it removes,

```
r(y) = D(y) - y = sigma^2 * grad_y log p(y)
```

is exactly a scaled **score**, the gradient of the log-density of noisy images. The residual points uphill toward more probable images. You did not train for this. It falls out of the noise model plus the fact that the optimal denoiser is the posterior mean.

Three consequences, each a whole research program:

**Generation.** Repeatedly nudge a random image uphill along the residual while annealing the noise level, and you sample from the image prior. This is what score-based diffusion models do; the denoiser is the score network [9] [10] [11].

**Inverse problems.** Insert the residual as the prior gradient in an iterative solver and you get inpainting, compressive sensing, super-resolution, deblurring, demosaicing, and undersampled MRI or CT, all with one denoiser and no task-specific training [6].

**Interpretability and range extension.** Make the network bias-free by removing every additive constant, and it becomes exactly scale-equivariant [5]. One denoiser then works across all noise levels, and its local behaviour is analyzable as an adaptive linear filter.

**The two caveats that determine what you can ship.** First, the popular framing that a trained denoiser contains one coherent global probability model is false for directly-parameterized networks: the learned residual field is not conservative, so no global energy or log-density exists. Sampling and reconstruction still work, because annealing keeps every step near a locally valid score, but calibrated probabilities and uncertainties are unavailable. Second, the field is also not passive, which independently blocks the RED and PnP convergence theory. Both are consequences of *parameterization*, not of learning, and Section 14 removes both.

---

## 2. Miyasawa and Tweedie, derived from scratch

Short, exact, and worth internalizing, because every extension in this guide is a variation on it.

### 2.1 Setup

A clean signal `x` is drawn from an unknown prior `p(x)`. You observe

```
y = x + n,        n ~ N(0, sigma^2 I),   n independent of x
```

Let

```
p(y) = integral p(x) N(y; x, sigma^2 I) dx
```

be the marginal density of the **noisy** observation. This is the prior blurred by the Gaussian.

### 2.2 The claim

Miyasawa 1961 [1], equivalently Tweedie's formula via Robbins 1956 [2] and Efron 2011 [3]:

```
E[x | y]  =  y  +  sigma^2 * grad_y log p(y)
```

### 2.3 The proof

Differentiate the marginal under the integral sign. The only `y`-dependence sits in the Gaussian kernel, and

```
grad_y N(y; x, sigma^2 I) = N(y; x, sigma^2 I) * (x - y) / sigma^2
```

Therefore

```
grad_y p(y) = integral p(x) N(y; x, sigma^2 I) * (x - y)/sigma^2  dx

            = (1/sigma^2) [ integral x p(x) N(y;x,sigma^2 I) dx
                            - y * integral p(x) N(y;x,sigma^2 I) dx ]
```

The second integral is `p(y)` by definition. The first is `E[x|y] * p(y)`, because the posterior is `p(x|y) = p(x) N(y;x,sigma^2 I) / p(y)`. Hence

```
grad_y p(y) = (p(y) / sigma^2) * ( E[x|y] - y )
```

Divide by `p(y)` and use `grad_y log p(y) = grad_y p(y) / p(y)`:

```
grad_y log p(y) = (1/sigma^2) ( E[x|y] - y )

  <=>   E[x|y] = y + sigma^2 grad_y log p(y)      QED
```

### 2.4 Read the result carefully

Three points that matter for everything downstream.

**The MMSE denoiser is the posterior mean.** The minimum-mean-squared-error estimator of `x` given `y` is exactly `E[x|y]`. So any network trained to minimize `||D(y) - x||^2` converges, in the infinite-data and infinite-capacity limit, to `E[x|y]`, and its residual `D(y) - y` converges to `sigma^2 grad log p(y)`. **You never estimate `p(x)`.** The prior enters only through the score of the smoothed density `p(y)`, which the optimal denoiser hands you for free. That score is what people mean by "the implicit prior," nothing more and nothing less.

**Two conditions, both required, nothing else.** First, additive Gaussian noise. Second, MMSE-optimality of `D`. There is no assumption on `p(x)` at all. This is why the result is universal across image content, and why every failure mode in this guide traces back to violating one of those two conditions.

**`p(y)` is the noise-blurred prior, not `p(x)`.** As `sigma -> 0`, `p(y) -> p(x)` and the score sharpens toward the true image manifold. This is the mathematical seed of annealing: solve at large `sigma` where the landscape is smooth and easy, and walk down to small `sigma` where the landscape is sharp and correct.

### 2.5 The two facts everyone conflates

Keep these separate. Almost all sloppy reasoning about denoiser priors comes from merging them.

**H1, the theorem.** Exact, for the MMSE-optimal denoiser. A fact about mathematics, true independent of any network. Among other things, it implies the ideal denoiser's Jacobian is **symmetric**, because a gradient field's Jacobian is a Hessian.

**H2, the instantiation gap.** A trained network is never exactly MMSE, because of finite data, finite capacity, and imperfect optimization. So the score you actually recover carries error, and as Section 9 shows by measurement, its Jacobian is not even symmetric, which the true MMSE denoiser's would be.

H1 is the ideal. H2 is the reality. The gap is not a small perturbation: it is large enough to destroy the existence of a global log-density. But it is a gap in *parameterization*, and Section 14 closes it by construction rather than by hoping training will.

---

## 3. Beyond Gaussian: the exponential-family generalization

The Gaussian is not special. It is the easiest member of the exponential family. Raphan and Simoncelli's nonparametric empirical-Bayes least squares result [4] gives the same *shape* of estimator, observation plus a score-correction of the marginal, for Poisson, Gamma, and general exponential-family noise:

```
x_hat(y) = y + (noise-model-specific operator) applied to grad log p(y)
```

For **Poisson** noise, which is the correct model for photon counting, the correction takes the form of a ratio of shifted marginals, roughly `p(y+1)/p(y) - 1` in the discrete case, because the natural "derivative" for a counting distribution is a finite difference rather than a gradient. For **Gamma** noise, a multiplicative model, it is again a ratio of shifted marginals with a different shift structure.

The upshot: **the residual-equals-score machinery is not locked to Gaussian noise.** It is the principled route into photon-limited imaging, meaning low-light photography, fluorescence microscopy, and astronomy. The cost is re-deriving the correction and retraining per noise family. Noise2Score [12] uses Tweedie explicitly and covers Gaussian, Poisson, and Gamma in one framework.

**The boundary is multiplicative noise.** For a purely multiplicative model there is no clean `D(y) - y = sigma^2 grad log p` identity. Only Monte-Carlo relations exist: one exact relation of a different functional form, and one small-`sigma` approximation. Do not expect the additive machinery to transfer. If your degradation is multiplicative, either variance-stabilize it into an additive regime first, or accept a different estimator.

---

## 4. Why bias-free is the load-bearing trick

There is a subtle problem with using Miyasawa in practice. The identity contains `sigma`, but you would like one network to work at every noise level without being told `sigma`, which is called blind denoising. Mohan et al. [5] show that architecture alone buys this.

### 4.1 The construction

Remove **every additive constant** from the network:

- No bias vectors in any convolution or dense layer.
- No mean subtraction or additive offset in normalization. Use a bias-free batch-norm variant that scales but does not shift.
- No additive positional embeddings or learned constants anywhere.
- Only positively homogeneous activations, meaning `phi(alpha z) = alpha phi(z)` for `alpha > 0`. ReLU, LeakyReLU, and PReLU qualify. GELU, SiLU, and ELU do not.

The result is a network that is **positively homogeneous of degree 1**:

```
D(alpha * y) = alpha * D(y)     for all alpha > 0
```

### 4.2 Why this is exactly the right property

Scaling the input by `alpha` scales the effective noise level by `alpha`. Degree-1 homogeneity means the denoiser's response co-scales, so a network trained on one noise range extrapolates to noise levels it never saw. Mohan et al. demonstrate under 1 dB PSNR loss when generalizing across a tenfold noise range. Equivalently, and this is the version that matters for Section 5: the same residual can be read as a valid score at every `sigma`.

### 4.3 A second gift: the Jacobian is a filter

A degree-1 homogeneous function satisfies Euler's relation `J(y) y = D(y)`, so the network is locally linear in a strong sense: near any point, `D(y) = J(y) y` exactly, not approximately. The Jacobian `J(y) = dD/dy` therefore acts as a **data-adaptive linear filter**, whose eigenvectors and singular vectors describe which local image structures are preserved and which are suppressed. This is the basis of the interpretability claim in [5], and Section 10 measures it.

Note carefully what this does and does not give you. The Jacobian is real and analyzable locally. It is **not** the Hessian of any global energy unless you construct the network to make it so, because for a directly-parameterized net it is not symmetric.

### 4.4 Homogeneity is not automatic, it is norm-dependent

This is the single most common implementation trap. Measured on real checkpoints:

- A network built from bias-free convolutions with `use_bias=False`, a bias-free batch normalization, and LeakyReLU is homogeneous **to float32 precision**: relative error `||D(alpha y) - alpha D(y)|| / ||alpha D(y)||` of about `2.5e-5`, flat across an eightyfold range of `alpha`, with a deliberately bias-broken control firing at `0.83` to prove the probe works.
- Substituting a **LayerNorm** block breaks it catastrophically: 81 to 98 percent relative error. LayerNorm subtracts a per-input mean, which is an additive constant that depends on the input, and that is precisely what homogeneity forbids.
- A factory-default **GELU** activation makes exact homogeneity mathematically impossible regardless of what else you do.
- Sibling checkpoints in the same architecture family, trained with slightly different normalization, showed deviations growing with `alpha` up to about 14 percent.

**Verify homogeneity per checkpoint. Never assume it from the architecture name.** Section 16 gives the protocol.

---

## 5. The noise-conditional bridge

This is the most valuable practical fact in this guide, and it is not covered by the literature the rest of the guide cites.

### 5.1 The problem

The modern inverse-problem solvers built on diffusion models, specifically DDRM [21], DDNM [22], DPS [15], and PiGDM [23], all require a **noise-conditional** denoiser. Their interface is `D(y, sigma)`: given an iterate at a known scheduled noise level `sigma_t`, return an estimate of the clean signal. A blind denoiser does not expose that interface. It exposes only `D(y)`.

Naively, this means a blind bias-free checkpoint cannot be used with any of them, and you would need to retrain a `sigma`-conditioned network from scratch.

### 5.2 The bridge

Define

```
D_sigma(y) := sigma * D(y / sigma)
```

Under exact degree-1 homogeneity, this is identically equal to `D(y)`:

```
sigma * D(y/sigma) = D(sigma * y/sigma) = D(y)
```

The equality is the content of the bridge, not a triviality to be waved away. It says the following:

**Rescaling an iterate into the network's trained noise band, denoising, and rescaling back gives exactly the same answer as denoising directly.** The blind network therefore satisfies the noise-conditional contract at *every* `sigma` simultaneously, with a consistency that is exact rather than approximate. You can drop a blind bias-free checkpoint into any `sigma`-conditioned slot in any of those samplers, and the variance bookkeeping the sampler performs remains coherent, because the network's response transforms correctly under the rescaling the sampler's schedule implies.

Measured: float32-exact on a properly bias-free checkpoint, validated against a bias-broken control. **Zero retraining required.**

### 5.3 The one real failure mode

The sampler's `sigma_t` is the **nominal** noise level from its schedule. The blind network responds to the **actual** noise present in its input. These agree in an idealized reverse diffusion, but they diverge in practice, because a mid-trajectory iterate carries reconstruction error and solver error in addition to scheduled noise.

The consequence: the blind network tracks the actual level and self-calibrates, while the sampler's bookkeeping tracks the nominal one. When they diverge, the sampler's step sizes and variance terms are computed against a level the denoiser is not operating at.

Practical mitigations, in order of preference:

1. **Estimate the level from the residual.** The residual magnitude `||r(y)||` is itself a noise-level estimate, since `||r|| ~ sigma^2 ||grad log p||`. Feed the estimated level back into the sampler's schedule rather than trusting the nominal one. This is what a self-calibrating annealed Langevin loop does implicitly.
2. **Prefer schedules with fewer, larger steps** where nominal and actual levels are less likely to drift apart, such as the EDM family [24].
3. **Monitor the discrepancy** between nominal `sigma_t` and residual-estimated level across a trajectory. A widening gap is your early warning that the sampler is off-manifold.

---

## 6. The linear inverse solver and its null/range split

Kadkhodaie and Simoncelli [6] give two algorithms, both consuming only the residual `r(y) = D(y) - y`.

### 6.1 The sampler

Coarse-to-fine stochastic gradient ascent, equivalently annealed Langevin. Start from noise, repeatedly step along the residual with an injected stochastic term, and shrink the effective `sigma` over time. A well-designed implementation re-estimates the noise level each iteration from `||r(y)||` rather than dividing by a known `sigma^2`, which makes the method **self-calibrating**: the residual magnitude is the local noise estimate.

### 6.2 The linear inverse solver

For a measurement `y_m = M x`, where `M` may be a mask, a blur, a subsampling operator, or a projection, the per-step update splits into two orthogonal pieces:

```
d_t = (I - M^+ M) * f(y)        <- NULL-SPACE term
                                   the prior fills what M cannot see

      + M^+ (y_m - M y)         <- RANGE-SPACE term
                                   hard data consistency, denoiser-independent
```

where `M^+` is the pseudo-inverse. This decomposition is the source of the method's generality. One denoiser solves inpainting, compressive sensing, super-resolution, deblurring, demosaicing, and MRI with no task-specific training, because the task only changes `M`.

### 6.3 The decomposition suggests a law that is false

The split invites an obvious inference: since the prior acts only in `null(M)`, the prior's contribution should scale with `dim(null(M))`. Large null space means prior-dominated; small null space means measurement-dominated and the prior is a garnish.

**This inference was tested directly and does not hold.** Section 11 gives the measurements. Dimension count and reconstructability are different quantities, and the intuition fails badly enough to invert. Do not use null-space fraction to triage tasks.

---

## 7. The literature collapses onto one object

Once you accept `r(y) = D(y) - y = sigma^2 grad log p(y)`, a sprawling literature becomes one object, a score network, viewed through different wrappers.

| Line of work | Relationship to the identity |
|---|---|
| Score-based diffusion: NCSN [9], DDPM [10], score-SDE [11] | The denoiser **is** the score model. Denoising score matching is Tweedie. The Kadkhodaie-Simoncelli sampler is one discretization of the reverse SDE. Diffusion models are Miyasawa denoisers wrapped in a noise schedule. |
| Plug-and-Play [8], Regularization by Denoising [7] | Use the denoiser as the prior's proximal operator in a splitting or optimization loop. Their convergence theorems require a **conservative** denoiser, and RED-family analyses additionally require **passivity**. Directly-parameterized nets have neither, so the guarantees do not transfer even though the methods often work in practice. |
| Provably convergent PnP [27] | Requires a Lipschitz-constrained residual, obtained by spectral normalization during training. A different route to the same guarantee, with an explicit training cost. |
| Gradient Step Denoiser and proximal denoisers [17] [18] | Parameterize the network as the gradient of an explicit scalar energy, which makes conservativeness **exact by construction**. This is the fix. See Section 14. |
| Self-supervised training: SURE [14] [26], Noise2Noise [13], Noise2Score [12] | Stein's lemma and Tweedie let you train or validate a denoiser, hence a score, without clean data. Same object, estimated from noisy observations only. |
| Diffusion inverse solvers: DDRM [21], DDNM [22], PiGDM [23] | Exploit the linear operator's structure, typically via SVD, inside a diffusion sampler. All require noise-conditioning, hence Section 5. |
| Diffusion posterior sampling: DPS [15] | Generalizes to nonlinear forward operators `g(x)` via linearization, for phase retrieval and nonlinear deblurring. Breaks the clean null/range split. |
| Exponential-family Tweedie [4] | Extends the identity itself off the Gaussian, to Poisson and Gamma. |
| Conservativeness critique [16] | Shows learned scores are generically non-conservative, having nonzero curl, so no exact energy exists. Sampling survives because the curl is roughly orthogonal to the annealed trajectory. This is the theoretical backing for the measurements in Section 9. |
| Energy versus score parameterization [19] | Finds energy-parameterized score models underperform directly-parameterized ones on sample quality. This is the cost you pay for the Section 14 fix, and you should measure it. |
| Geometry-adaptive representations [30] | Argues denoiser generalization comes from adapting to geometric structure in the data, which is the same operator viewed as a manifold projector rather than as a score. |

**The single most useful reframing:** an existing bias-free denoiser is already a diffusion score network. The modern diffusion toolbox, including fast SDE and ODE solvers, guidance, and distillation, is in principle a drop-in.

---

# Part II. Measured properties of real networks

Everything in this part was measured on trained bias-free denoisers using GPU-serial probes, finite-difference Jacobian computations, and DIV2K-validation content, with matched controls in every case. Where a finding did not replicate across checkpoints, that is stated.

## 8. Homogeneity is exact, and it is norm-dependent

**Finding.** On a checkpoint built from bias-free convolutions, bias-free batch normalization, and LeakyReLU, the relative homogeneity error is float32-exact at approximately `2.5e-5`, and **flat across an eightyfold range of `alpha`**, including values far above the trained noise range. A deliberately bias-broken control fired at `0.83`, which validates the probe.

**But it is a property of the normalization stack, not of the architecture family.** Substituting LayerNorm produces 81 to 98 percent error. Sibling checkpoints with different normalization showed `alpha`-growing deviations up to about 14 percent. A GELU activation makes exactness impossible.

**Why this is the most important measurement in the guide.** Exact homogeneity is what licenses the noise-conditional bridge of Section 5, and therefore what makes DDRM, DDNM, DPS, and PiGDM reachable from a blind checkpoint at zero retraining cost. It is also what preserves cross-`sigma` generalization. Everything downstream depends on it, and it is fragile to a single layer choice.

## 9. The field is not conservative and not passive

### 9.1 Non-conservativeness

A residual field is the gradient of a scalar log-density **if and only if** its Jacobian is symmetric everywhere. Measured on directly-parameterized checkpoints:

| Measurement | Value | Control on identical probe | Ratio |
|---|---|---|---|
| Exact local Jacobian block, 12x12x3 = 432 dims, co-located input and output patch inside a 256x256 image, finite differences | `\|\|J - J^T\|\| / \|\|J\|\| = 0.58` | box blur reads `0.0001` | about 7,400x |
| Random-directional asymmetry, architecture A | `0.677` | symmetric-blur baseline | 5.3x |
| Random-directional asymmetry, architecture B, independent family | `1.19` | symmetric-blur baseline | 15.4x |
| Third checkpoint, replication | `0.14` | | about 800x |

The near-zero control matters more than the headline number: a genuinely symmetric operator reads as symmetric on the identical extraction, which proves the measurement is clean rather than an artifact. Asymmetry within a **co-located** block is sufficient to prove the global field non-conservative.

**Conclusion. There is no global energy or log-density.** The implicit prior of a directly-parameterized denoiser is a locally valid score field, not a probability model you can integrate, normalize, or read calibrated uncertainty from. It replicates across independent architectures, so it is not a one-model artifact.

Sampling and reconstruction still work, because the curl component is roughly orthogonal to the annealed descent path [16]. What does not work is any claim of calibrated posterior uncertainty, and any invocation of RED or PnP convergence theory.

### 9.2 Non-passivity, which is a separate obstacle

RED-family convergence analyses, including the MRED analysis (arXiv:2202.04961), additionally require **passivity**:

```
||D(f)|| <= ||f||   for all f
```

Measured spectral norm on clean inputs: `||J||_2 = 1.22 to 1.36`. **The network is not passive.**

This matters because it is a common error to believe that fixing conservativeness rescues the RED guarantees. It does not. Conservativeness and passivity are independent conditions, and a directly-parameterized denoiser typically fails both. Section 15 handles passivity, and under the Section 14 construction it becomes unusually cheap to enforce.

## 10. Local Jacobian geometry

Because a bias-free net satisfies `J(y) y = D(y)` exactly, the Jacobian is a genuine data-adaptive filter and its spectrum is interpretable. On the exact 432-dimensional local block:

| Quantity | Value | Reading |
|---|---|---|
| Stable rank `\|\|J\|\|_F^2 / \|\|J\|\|_2^2` | 7.7 of 432, about 2 percent | strongly low-rank |
| Participation ratio of singular values | 16.8 of 432, about 4 percent | roughly 10 to 17 effective directions |
| Top five singular values | 0.98, 0.86, 0.84, 0.77, 0.69 | a few preserved modes |
| Median singular value | 0.028 | the other 90 percent crushed |
| Symmetric-part eigenvalues | 392 near zero, about 40 mid or high, 2 slightly negative, range `[-0.06, 0.97]` | projection-like, nearly positive semidefinite, not a hard projector |

**Reading.** Locally the denoiser preserves roughly ten dominant modes and suppresses the rest: a soft projection onto a low-dimensional local subspace, plausibly the signal-manifold tangent. Combined with Section 9, the local operator is approximately **(soft low-rank shrinkage onto a roughly ten-dimensional subspace) plus (a rotational curl component)**. The shrinkage is *why it denoises*. The curl is *exactly what makes the field non-conservative*. The manifold-geometry account and the no-global-prior finding are two faces of the same operator, not competing explanations.

**Two caveats that materially limit this.** First, the rank is a *local* property at one point on one checkpoint; the global manifold dimension is much larger and was not measured. Second and more seriously, **the low-rank structure does not transfer across checkpoints**: a sibling network measured 38.6 percent stable rank under the identical probe, against 2 percent here, with asymmetry 0.14 against 0.58. So the geometric reading is checkpoint-specific and should not be used as a general explanation of why these methods work. The non-conservativeness conclusion, by contrast, replicated everywhere it was tested.

## 11. How much work the prior actually does

Two different metrics get conflated here, and separating them is what corrects the naive null-space intuition.

**Metric one, prior credit share.** Ablate the solver's two terms and measure null-space-restricted reconstruction error: prior-only against data-only, at identical iteration budget. Credit share is the fraction of achievable null-space gain attributable to the prior term.

| Task | Null-space fraction | Prior credit share |
|---|---:|---:|
| Block inpainting, 64-pixel square | 6.25 percent | 94.5 percent |
| Random pixels, keep 30 percent | 70.0 percent | 96.8 percent |
| Super-resolution, factor 4 | 93.75 percent | 86.4 percent |

Credit share is **flat at 86 to 97 percent across a fifteenfold range of null-space fraction, and inverted at the top**: the largest null space has the *lowest* prior credit. Null-space fraction has no predictive power over credit share. The widely-quoted "the prior does about 84 percent of the work at 50 percent masking" figure replicates only as a rough **constant near 90 percent**, not as one point on a slope.

**Metric two, absolute null-fill quality.** How good the filled-in content actually is, on its own terms. This varies enormously and non-monotonically:

| Task | Null-space fraction | Null-fill quality |
|---|---:|---:|
| Demosaicing | 66.7 percent | 0.997 |
| Block inpainting, 64-pixel square | 6.25 percent | 0.131 |

**The lesson.** Credit share answers "which solver term did the work," and the answer is almost always "the prior," roughly independent of the operator. Quality answers "was the work any good," and the answer depends strongly on the operator, in a way null-space dimension does not capture. The original error was reading a near-constant credit share as if it were a function of null-space size, from a single anchor point.

**Why dimension count fails.** Super-resolution hides high spatial frequencies, which are cheaply predictable from the measured low frequencies under natural-image `1/f` statistics. Block inpainting hides a contiguous region with no local measurement support at all. The block case removes fifteen times fewer dimensions and is far harder. Dimension count and reconstructability are simply different quantities.

## 12. Domain transfer behaves opposite to expectation

**Expectation.** A denoiser trained on natural photographs encodes a natural-image prior, so out-of-domain content should degrade.

**Measurement.** Under an identical operator and solver, an out-of-domain **medical X-ray reconstructed substantially better than in-domain natural photographs**: a gain of `+12.39 dB` over a trivial baseline, against `+1.54 dB` for natural photos.

**Reading.** This is not merely "out-of-domain works." It is refuted with the wrong sign. The plausible mechanism is that medical imagery is smoother, lower-entropy, and more locally predictable than cluttered natural scenes, so the conditional unpredictability of the hidden content given the measured content, `H(x_null | x_range)`, is much lower. The prior's job is easier, even though the content is nominally unfamiliar.

**Practical consequence. Undersampled MRI and CT are top targets for this machinery, not risky hedges.** Do not gate a medical or scientific imaging project on domain-match concerns without measuring first.

**Honest limit.** The predictive variable of Section 13 was validated on the *operator* axis. On the *domain* axis it is a plausible post-hoc explanation of this result, not a validated predictor. Measure per domain.

---

# Part III. Predicting success

## 13. Three hard gates, then one continuous variable

This replaces the null-space law, which is false. Apply the gates in order. A task that fails a gate is not a task to be run at reduced expectation; it is a task that needs a different construction.

### Gate 1. Is the forward operator linear?

The null/range decomposition of Section 6 is a linear-algebra fact. It has no nonlinear analogue.

- **Pass:** masks, subsampling, blur with a known kernel, Radon and Fourier undersampling, color filter arrays, downsampling.
- **Fail:** phase retrieval, nonlinear deblurring, saturation and clipping, JPEG quantization, tone mapping.
- **Route on failure:** DPS-style posterior guidance [15] with Jacobian-vector-product linearization. This is a genuinely different algorithm, more expensive, and it re-opens every conservativeness question.

Sensitivity analysis over the success predictors ranks **operator linearity first**, above every other factor including checkpoint quality.

### Gate 2. Is the operator knowable?

This is distinct from linearity and is the gate most often missed. Blind deblurring is perfectly linear, and completely blocked, because you do not know the kernel. The solver needs `M` and `M^+` explicitly.

- **Pass:** operator known analytically, or estimable to high confidence from the observation.
- **Fail:** unknown blur kernel, unknown mask, unknown mixed degradation chain.
- **Route on failure:** estimate the operator first with a dedicated estimator, or alternate between operator estimation and image reconstruction. Both add substantial machinery, and errors in `M` propagate directly into the range-space term where the prior cannot correct them.

### Gate 3. Do the corruption statistics match the training noise model?

Miyasawa requires additive Gaussian noise. The theorem's second condition, MMSE-optimality, is with respect to *that* noise model.

- **Pass:** additive Gaussian sensor noise at any level, given exact homogeneity.
- **Fail:** Poisson or shot noise, multiplicative or speckle noise, structured or correlated noise, compression artifacts.
- **Route on failure:** variance-stabilize into an approximately Gaussian regime, such as an Anscombe-type transform [29] for Poisson, or train a dedicated head using the generalized Tweedie correction of Section 3. Multiplicative noise has no clean identity and needs a different estimator entirely.

### Among gate survivors: conditional unpredictability

For tasks that pass all three gates, the predictor of reconstruction quality is

```
H( x_null | x_range )
```

the conditional entropy of the unobserved content given the observed content. Content complexity, not null-space size.

**How to estimate it in practice, cheaply.** You do not need a real entropy estimator. Useful proxies, in ascending order of cost:

1. **Spatial support.** Does every unobserved coordinate have measured neighbours within a small radius? Scattered missing pixels and subsampled frequencies do. Contiguous holes do not. This one proxy explains the demosaicing-versus-block-inpainting gap.
2. **Spectral overlap.** Does `range(M)` contain the frequency bands that predict the missing bands under `1/f` statistics? Super-resolution passes because low frequencies predict high ones. A band-stop operator that removes mid frequencies does not.
3. **Empirical proxy.** Run the solver on a small held-out set for the candidate operator and read the null-fill quality directly. Fifty images is usually enough to rank operators, and it costs less than building a theory of your operator.

**Validation status.** This variable is validated on the operator axis: it correctly orders demosaicing, super-resolution, random-pixel inpainting, and block inpainting, where null-space fraction does not. It is **not** validated on the domain axis. Section 12's medical result is consistent with it but was not predicted by it in advance.

---

# Part IV. Building

## 14. Making the denoiser conservative

### 14.1 The key fact first

**The ideal denoiser is conservative.** `E[x|y] = y + sigma^2 grad log p(y)` is a gradient field by construction, so its Jacobian is symmetric. The measured asymmetry of Section 9, 0.58 and 0.14 against near-zero controls, is therefore **pure estimation error**, not a property of the target.

This reframes the whole exercise. Enforcing symmetry adds a **correct** inductive bias. You are not trading accuracy for a guarantee; you are removing a degree of freedom that the true solution never used. That does not mean it is free in practice, and Section 14.4 gives the measured cost, but the direction of the tradeoff is not what it first appears.

### 14.2 Method A: exact by construction, recommended

Parameterize the **energy**, not the denoiser:

```
E(y) = 0.5 * || g(y) ||^2                 g = a bias-free network with vector output
D(y) = y - grad_y E(y)
```

Then `J_D = I - Hess E` is a Hessian, so **symmetry is exact to floating-point precision**, not penalized. The residual becomes `r(y) = -grad E(y)`, and

```
log p(y) = -E(y) / sigma^2  +  constant
```

You now have an actual scalar energy. You can evaluate it, compare two candidate reconstructions by it, and use it for model selection.

This is the Gradient Step Denoiser construction [17], later extended to proximal denoisers with stronger PnP guarantees [18], and it was designed for exactly the purpose you want.

### 14.3 Why the quadratic form specifically

This detail is essential and easy to get wrong. You need `E` to be **degree-2 homogeneous** so that `grad E` is degree-1, which is what preserves the noise-conditional bridge of Section 5.

Consider the naive alternative. A scalar-output bias-free network `h` is degree-**1** homogeneous, so `grad h` is degree-**0**, so `D(y) = y - grad h(y)` is not homogeneous at all, and the Section 5 bridge dies. You would have traded your most valuable property for your second-most valuable one.

The quadratic form fixes it. With `g` degree-1 homogeneous:

```
E(alpha y) = 0.5 ||g(alpha y)||^2 = 0.5 alpha^2 ||g(y)||^2 = alpha^2 E(y)
```

Differentiating `E(alpha y) = alpha^2 E(y)` with respect to `y` gives `alpha grad E(alpha y) = alpha^2 grad E(y)`, hence

```
grad E(alpha y) = alpha * grad E(y)          degree-1, as required
D(alpha y) = alpha y - alpha grad E(y) = alpha D(y)     exact
```

**Conservativeness and homogeneity are compatible, but only through this construction.** As a bonus, `E >= 0` automatically, which removes a class of degenerate solutions.

### 14.4 Costs, stated plainly

- **Inference.** One forward pass plus one vector-Jacobian product through `g`. Roughly two to three times the cost of a direct denoiser.
- **Training.** You must differentiate through `grad E`, which requires second-order automatic differentiation. **This will hit the jit-convolution instability in TF 2.18-era Keras reverse-mode graphs.** Finite differences are not a substitute here: they work fine for *measuring* a Jacobian, as in Sections 9 and 10, but not for building a training graph. Budget time for either porting `g` to JAX or PyTorch, or disabling XLA on the affected operations.
- **Quality.** Energy-parameterized score models have been found to underperform directly-parameterized ones on sample quality [19]. Expect a PSNR cost. Measure it against your direct baseline before committing.

### 14.5 Method B: soft symmetry penalty

Keep your architecture and add a penalty. The tight estimator is

```
L_sym = E_{u ~ N(0,I)} [ || (J - J^T) u ||^2 ]  =  || J - J^T ||_F^2
```

which needs one Jacobian-vector product and one vector-Jacobian product per sample. A looser and cheaper variant is `E_{u,v} [ (u^T J v - v^T J u)^2 ]`.

**Use this as a diagnostic or a warm start, not as a deliverable.** It drives asymmetry down but never to zero, and **every PnP and RED theorem requires exact symmetry**. A penalty buys "closer to conservative," which is not the same as eligibility for the guarantees. It is genuinely useful for two things: quantifying how far a given architecture is from conservative, and initializing a Method A network from a pretrained direct one.

### 14.6 Method C: post-hoc symmetrization does not work

Symmetrizing `J` pointwise at inference time does not produce an integrable field. Integrability requires consistency of the field across the whole domain, not symmetry at a point: a field can have symmetric Jacobian at every sampled point of a probe and still have nonzero circulation around loops you did not probe. There is no useful post-hoc fix. Do not spend time here.

## 15. Making it passive and nonexpansive

Passivity is a separate requirement from conservativeness, and you need both for the RED-family guarantees. Section 9.2 measured `||J||_2 = 1.22 to 1.36`, so a directly-parameterized net fails independently.

**Under Method A this becomes unusually cheap.** Euler's theorem applied to the degree-2 homogeneous `E` gives `y^T grad E(y) = 2 E(y)`. Therefore

```
||D(y)||^2 = ||y - grad E(y)||^2
           = ||y||^2 - 2 y^T grad E(y) + ||grad E(y)||^2
           = ||y||^2 - 4 E(y) + ||grad E(y)||^2
```

So passivity `||D(y)|| <= ||y||` is **exactly** the scalar condition

```
|| grad E(y) ||^2  <=  4 E(y)        for all y
```

Enforce it with a hinge penalty on training batches:

```
L_passive = mean( max(0, ||grad E(y)||^2 - 4 E(y)) )
```

No power iteration, no spectral normalization, no per-layer Lipschitz budget. **Both quantities are already computed in the forward pass**, so the penalty is nearly free.

For the stronger **nonexpansiveness** condition `||J_D||_2 <= 1`, you additionally need `Hess E` positive semidefinite with spectrum in `[0, 2]`. The measurement in Section 10 is encouraging here: the symmetric part is already nearly positive semidefinite, with 392 eigenvalues near zero, only two slightly negative, and range `[-0.06, 0.97]`. So this is a small correction rather than a fight. A hinge on the minimum Rayleigh quotient `u^T (Hess E) u` over random directions `u` is usually sufficient.

If you prefer to stay with a directly-parameterized network, the alternative route to nonexpansiveness is spectral normalization of the residual during training [27], at a known and somewhat larger quality cost.

## 16. The verification protocol

Do not rely on a local Jacobian alone. Symmetry at one point is necessary but not sufficient. Run all five checks, with controls, on the specific checkpoint you intend to ship.

**1. Loop integral, the genuinely global test.** Pick random closed polygons in image space and sum `r(y) . dy` around them. A conservative field gives zero. Nonzero circulation is direct proof of curl, needs no Jacobian at all, and is the only check that probes the field globally rather than pointwise. Report circulation normalized by path length and field magnitude so results are comparable across checkpoints.

**2. Exact local Jacobian block.** Finite-difference a co-located patch, for example 12x12x3 = 432 dimensions inside a 256x256 image, and report `||J - J^T||_F / ||J||_F`. Use finite differences, not reverse-mode autodiff, which sidesteps the jit-convolution instability entirely.

**3. Symmetric control.** Keep a box blur or Gaussian blur in the same test harness, extracted identically. It should read near zero; it reads about `0.0001` in practice. This is what proves your probe is clean. Without it, a low asymmetry number is uninterpretable.

**4. Broken control.** Keep a deliberately non-conservative or bias-broken network in the same test, so you can demonstrate that the metric fires. A probe that never produces a high number is not a probe.

**5. Homogeneity, again, after every change.** Re-verify `||D(alpha y) - alpha D(y)|| / ||alpha D(y)||` across an eightyfold `alpha` range. Method A guarantees it in theory, but implementation slips, such as a stray bias in `g` or a GELU that survived a refactor, show up here and nowhere else.

**6. Passivity and spectral norm.** Estimate `||J||_2` by power iteration on clean and noisy inputs. Under Method A, additionally log `||grad E||^2 - 4E` as a scalar per batch, which is the exact passivity margin.

**Multi-checkpoint discipline.** The stable-rank finding of Section 10 did not transfer across sibling checkpoints, 2 percent against 38.6 percent. Assume nothing transfers until measured on the exact checkpoint you ship.

## 17. Architecture for general-purpose restoration

The denoiser is the right thing to freeze and build around. Treat it as one component, the prior, inside a system whose other components are an operator estimator, a router, and a solver bank.

```
input
  |
  v
degradation identification          <- estimate M, sigma, noise family
  |
  v
gate check                          <- Section 13, in order
  |
  v
solver dispatch                     <- pick by gate outcome and null-space structure
  |
  v
output                              <- no confidence map attached
       ^
       |
  sigma-emulation shim: D_sigma(y) = sigma * D(y/sigma)
       ^
       |
  frozen bias-free denoiser (the prior)
```

### 17.1 Degradation identification

A small classifier and regressor that estimates the forward operator and the noise level: blur kernel, downsampling factor, mask, JPEG quality factor, noise level and family. This is the component that turns "unusable in the wild" into "usable," because Gate 2 requires a known operator and real-world inputs do not come labelled.

Design notes. Predict the kernel in a low-dimensional basis rather than pixel-wise. Output a confidence estimate alongside every parameter, and route low-confidence cases to joint operator-image refinement rather than to a solver that will trust a wrong `M`. Errors in `M` land in the range-space term, where the prior structurally cannot correct them.

### 17.2 Gate check and routing table

| Condition | Route |
|---|---|
| Noise only, no operator | Single forward pass through the denoiser. Do not run a solver. |
| Linear, known `M`, large and unpredictable null space | DDRM or DDNM [21] [22], using the Section 5 shim |
| Linear, known `M`, small or well-conditioned null space | Classical regularized least squares. The prior earns little here and costs many network evaluations. |
| Linear, uncertain `M` | Alternating operator estimation and reconstruction, or PiGDM [23] with an inflated measurement-noise term to absorb operator error |
| Nonlinear operator | DPS [15] with JVP linearization. Expect higher cost and no clean null/range split. |
| Non-Gaussian corruption statistics | Variance-stabilize, then re-enter at the appropriate row, or dispatch to a Poisson or Gamma head |
| Multiplicative or speckle noise | Out of scope for this machinery. Different estimator. |

### 17.3 What this design still cannot do

Be honest about the boundary, because it is where most of the remaining engineering lives.

Real-world images carry **unknown, mixed, non-Gaussian degradation chains**: resize, then JPEG, then sensor noise, then sharpening, in unknown order with unknown parameters. This prior handles one known linear operator plus Gaussian noise at a time. The two honest options are:

1. **Joint operator estimation with iterative refinement.** Principled, expensive, and prone to converging on a wrong operator that explains the observation equally well.
2. **A supervised degradation-inversion front end** that maps real corruption into the Gaussian regime, after which the frozen prior takes over. More practical, but you are now training a task-specific network, which gives up part of the "one denoiser, no retraining" appeal.

Neither is free. The front end is where the difficulty concentrates.

### 17.4 Retraining moves worth making

Listed in descending expected value.

1. **Multi-domain prior.** Mix natural images with medical, document and text, and satellite content. Given the Section 12 result, a broadened prior is likely cheap upside rather than a capacity fight.
2. **Wider noise-level curriculum.** Homogeneity gives exact extrapolation of the *scaling*, but score *quality* at each level still comes from training coverage. These are different things and the first does not substitute for the second.
3. **Method A reparameterization**, if you need a real energy, PnP or RED guarantees, or model-selection-by-energy. Section 14.
4. **Poisson and Gamma variants** via generalized Tweedie, as separate checkpoints selected by the router. This is what genuinely extends coverage, to low-light and microscopy.
5. **Distillation** of the multi-step solver into a few-step student, if latency matters for a product.

## 18. Safe and unsafe architectural modifications

Exact homogeneity is what makes the Section 5 bridge work, so every architectural change must be checked against it.

**Safe.** More channels, more depth, additive skip connections, concatenation, convolution, strided convolution, pooling, bilinear or nearest-neighbour upsampling, ReLU, LeakyReLU, PReLU, bias-free batch normalization, scale-only multiplicative conditioning, residual blocks.

**Breaks homogeneity.**

| Modification | Why it breaks |
|---|---|
| `use_bias=True` anywhere | An additive constant is exactly what homogeneity forbids |
| GELU, SiLU, ELU, Softplus | Not positively homogeneous. Makes exactness mathematically impossible. |
| LayerNorm, InstanceNorm, or any mean-subtracting norm | Subtracts an input-dependent additive offset |
| Additive positional embeddings | Additive constant |
| Softmax attention | `Q K^T` scales as `alpha^2`, then softmax destroys the scaling entirely |
| Learned constants, learned thresholds, additive FiLM shift terms | Additive constants |

**If you want attention**, use a softmax-free linear attention, or normalize the logits by their own norm to restore scale invariance. Test it in isolation before integrating.

**After every architectural change**, re-measure homogeneity across the full `alpha` range and keep a deliberately bias-broken control in the same test to prove the probe fires. This is a five-minute check that prevents silently losing the property the whole system depends on.

## 19. Cost control

The frozen-prior approach costs roughly 20 to 1,000 network evaluations per image, against one for a task-specific supervised restorer. That is the central economic fact of this design, and it needs active management.

- **Cascade.** Solve coarsely at high `sigma` with few steps, then refine at low `sigma`. Most of the reconstruction happens early.
- **Cache the operator decomposition.** DDRM and DDNM need an SVD or pseudo-inverse of `M`. If many images share the same operator, which is typical for a fixed sensor or fixed sampling pattern, compute it once.
- **Short-circuit.** When the router detects noise-only degradation, call the denoiser once and return. Do not run a solver for a problem with an empty null space.
- **Prefer few-step schedules.** The EDM family [24] and its successors reach comparable quality at a fraction of the step count, and they interact better with the nominal-versus-actual noise-level issue of Section 5.3.
- **Distill** once the pipeline is stable and you know which operators dominate your traffic.

## 20. Ranked build list

Each row: the new capability, what it builds on, the exactness cost or failure mode, and the effort.

| # | Capability | Builds on | Cost or failure mode | Effort |
|---|---|---|---|---|
| 1 | **Noise-conditional emulation.** Expose a blind bias-free checkpoint through `D_sigma(y) = sigma D(y/sigma)` and unlock DDRM, DDNM, DPS, PiGDM | Exact homogeneity, verified per checkpoint | Nominal-versus-actual noise-level drift, Section 5.3 | **Cheapest, highest value.** No retraining |
| 2 | **Diffusion score-net reframing.** Plug the checkpoint into a modern SDE or ODE solver, with guidance and distillation | Existing annealed-Langevin sampler | Only the H1/H2 gap. This is a reframing, not a new approximation | Cheap. Swap the schedule |
| 3 | **Degradation identification front end.** Estimate `M`, `sigma`, and noise family from the observation | New component | This is the real bottleneck for in-the-wild use. Wrong `M` is uncorrectable by the prior | Moderate to substantial |
| 4 | **Method A conservative reparameterization.** `E = 0.5\|\|g\|\|^2`, `D = y - grad E` | Existing bias-free trunk as `g` | 2 to 3x inference; second-order autodiff hits the jit-conv blocker; possible PSNR cost [19] | New training |
| 5 | **Passivity hinge.** `max(0, \|\|grad E\|\|^2 - 4E)` on top of #4, unlocking RED and PnP guarantees legitimately | Method A | Nearly free given #4. Both terms already in the forward pass | Cheap, given #4 |
| 6 | **Medical and scientific reconstruction.** Undersampled MRI and CT | Sections 12 and 13 | Operator is known and linear, which is the easy case. Regulatory constraints are the real cost | Moderate |
| 7 | **Generalized Tweedie for Poisson and Gamma.** Photon-limited, low-light, microscopy | Section 3 | Re-derive the correction and retrain per noise family. Additive curriculum does not transfer | New training |
| 8 | **SURE and Noise2Score self-supervision.** Train or validate with no clean references | Stein's lemma [26], Tweedie | SURE variance is high for large nonlinear nets. Use for validation before trusting as sole loss | Moderate |
| 9 | **DPS-style nonlinear posterior sampling.** Phase retrieval, nonlinear deblurring | The null/range split as a template | Highest. The clean split is linear-only. Needs JVP linearization and re-opens conservativeness | New implementation |
| 10 | **Local manifold tangent extraction.** Top Jacobian singular vectors as local structure | Finite-difference local Jacobian | Local only, and checkpoint-specific, Section 10 | Cheap ablation |
| 11 | **Per-pixel uncertainty from shrinkage magnitude** | Residual and divergence fields | **Unlicensed until #4 and #5 land, and even then only as an energy comparison, never a calibrated probability** | Do not ship |

**Cross-cutting.** Replace hand-tuned clip, cap, and floor interventions in the sampler's step-size scaffolding with principled control-theoretic bounds, using gain margins and saturation-aware Lyapunov analysis. Ad-hoc numerical guards accumulate silently and make failures hard to attribute.

---

# Part V. Reference

## 21. True, overstated, and false: the ledger

| Claim | Verdict | Basis |
|---|---|---|
| Residual equals scaled score, for additive Gaussian noise and an MMSE denoiser | **True**, exact in the idealization | Miyasawa's theorem, Section 2 |
| The ideal denoiser is conservative, with symmetric Jacobian | **True** | A gradient field's Jacobian is a Hessian. Section 14.1 |
| Bias-free implies exact degree-1 homogeneity | **True, and norm-dependent** | Float32-exact at `2.5e-5`, flat across 80x in `alpha`, with a control at `0.83`. LayerNorm breaks it at 81 to 98 percent. GELU makes it impossible. |
| A blind denoiser can serve as a noise-conditional one at zero retraining cost | **True, and the most valuable fact here** | Exact homogeneity gives `sigma D(y/sigma) = D(y)`. Unlocks DDRM, DDNM, DPS, PiGDM. Section 5 |
| A single directly-parameterized net equals a coherent global prior or energy | **False** | Non-symmetric Jacobian: `0.58` against a `0.0001` control on the identical extraction, about 7,400x. Replicated across three checkpoints and two architecture families. |
| Sampling and reconstruction work anyway | **True** | Prior-only inpainting reconstructs well. The curl is roughly orthogonal to the annealed trajectory [16] |
| RED and PnP convergence guarantees apply to a learned denoiser as trained | **False, they do not transfer** | Non-conservative **and** not passive: measured `\|\|J\|\|_2 = 1.22 to 1.36`. RED-family analyses need passivity, so they do not rescue it either. |
| Conservativeness can be made exact | **True, by reparameterization** | `E = 0.5\|\|g\|\|^2`, `D = y - grad E`. Symmetry exact to float precision, and degree-1 homogeneity preserved. Sections 14.2 and 14.3 |
| Passivity is cheap once conservativeness is exact | **True** | Euler's theorem reduces it to the scalar condition `\|\|grad E\|\|^2 <= 4E`. Section 15 |
| Post-hoc symmetrization of the Jacobian yields a conservative field | **False** | Integrability is a global property. Pointwise symmetry does not imply zero circulation. Section 14.6 |
| `dim(null(M))/N` predicts the prior's contribution | **False** | Credit share flat at 86 to 97 percent across a 15x null-space range, and inverted at the top |
| "The prior does about 84 percent of the work," as a universal | **Overstated, but roughly right as a constant** | It is a near-constant around 90 percent of *credit share*, not a function of null-space size. Absolute quality is the metric that varies. |
| Out-of-domain content fails | **False, refuted with the wrong sign** | An out-of-domain X-ray reached `+12.39 dB` against `+1.54 dB` for in-domain natural photos, same operator and solver. MRI and CT are top targets. |
| Efficacy is primarily manifold-projection geometry | **Checkpoint-specific, does not generalize** | Stable rank 2 percent on one checkpoint, 38.6 percent on a sibling under the identical probe |
| One denoiser solves all linear inverse problems | **True with an asterisk** | Operationally true. The asterisk is the three gates of Section 13: linearity, knowability, corruption statistics. |
| Calibrated per-pixel uncertainty is available | **False, and permanently so for a directly-parameterized net** | No global log-density exists. Under Method A an unnormalized energy exists, which permits *comparison*, never a probability. |
| Multiplicative noise transfers | **False, verified negative** | No clean identity. Only Monte-Carlo relations: one exact of a different form, one small-`sigma` approximation. |

## 22. Limitations and honest uncertainty

1. **No calibrated uncertainty is licensed.** For a directly-parameterized net, the non-conservative field means every result is a point estimate or a sampling-quality claim, never a calibrated-posterior claim. Under Method A you gain an unnormalized energy `E(y)`, which lets you rank two candidate reconstructions, but the partition function remains inaccessible, so you still cannot report a probability or a confidence interval. Treat this as a permanent guardrail on product claims.
2. **Homogeneity is checkpoint-specific and layer-fragile.** Re-verify before relying on cross-`sigma` generalization or on the Section 5 bridge, for every checkpoint and after every architectural change.
3. **The credit-share constant rests on four operators.** The measurements span block inpainting, random-pixel inpainting, super-resolution, and demosaicing on a small validation set. The near-constancy is well-supported; the exact value is not.
4. **Conditional unpredictability is validated on the operator axis only.** On the domain axis it is a plausible post-hoc account of the medical-imaging result, not a validated predictor. Measure per domain rather than predicting.
5. **Local rank is local.** The soft-low-rank-projector reading holds at measured points on one checkpoint, does not transfer to a sibling, and says nothing about global manifold dimension. Do not build an explanation of *why the method works* on it.
6. **The Method A quality cost is unmeasured here.** The literature reports a penalty for energy parameterization [19]. Whether it is acceptable for your task is an experiment, not a deduction.
7. **Second-order autodiff is a real engineering blocker**, not a footnote. Method A training requires it, and the finite-difference workaround that rescues *measurement* does not rescue *training*.
8. **Single-repository, largely single-domain empirical base.** Natural-image validation content dominates, with one out-of-domain probe. Cross-domain generality is asserted from the literature, not independently established.

## 23. Glossary

**Annealed Langevin.** Noisy gradient ascent on `log p` with a decreasing noise schedule. Equivalently, reverse-diffusion sampling.

**Conservative field.** A vector field that is the gradient of a scalar potential, equivalently one whose Jacobian is symmetric everywhere, equivalently one with zero circulation around every closed loop. A directly-parameterized learned residual field generally is not.

**Credit share.** In an ablation of the solver's two terms, the fraction of achievable null-space gain attributable to the prior term. Distinct from absolute reconstruction quality, and the conflation of the two is the source of the false null-space law.

**Degree-1 homogeneity.** `D(alpha y) = alpha D(y)` for `alpha > 0`. The scale-equivariance a genuinely bias-free network enjoys, and the enabling property for the noise-conditional bridge.

**Empirical Bayes.** Estimating a needed functional of an unknown prior directly from data, here the marginal's score read off the optimal denoiser, without ever parameterizing `p(x)`.

**MMSE denoiser.** The estimator minimizing mean squared error. Equals the posterior mean `E[x|y]`.

**Nonexpansive.** `||J||_2 <= 1`. Stronger than passivity, and required by some PnP fixed-point analyses.

**Null space of `M`.** The signal components the measurement operator cannot see. The only place the prior can act in a linear solver.

**Participation ratio.** `(sum s_i)^2 / sum s_i^2` over singular values. A soft count of effective directions.

**Passive.** `||D(f)|| <= ||f||` for all inputs. Required by RED-family convergence analyses, and independent of conservativeness.

**Score.** `grad_y log p(y)`, the gradient of the log-density. Points toward higher-probability regions.

**Stable rank.** `||J||_F^2 / ||J||_2^2`. A continuous, noise-robust surrogate for matrix rank.

**Tweedie's formula.** The general empirical-Bayes identity relating a posterior mean to the score of the marginal. Miyasawa's theorem is its Gaussian case.

## 24. Citation index

| # | Work | Where used in this guide |
|---|---|---|
| 1 | Miyasawa, K. (1961). *An empirical Bayes estimator of the mean of a normal population.* Bull. Inst. Internat. Statist. 38(4):181-188. | §1, §2, the core theorem |
| 2 | Robbins, H. (1956). *An empirical Bayes approach to statistics.* Proc. 3rd Berkeley Symp. | §2, Tweedie lineage |
| 3 | Efron, B. (2011). *Tweedie's formula and selection bias.* JASA 106(496):1602-1614. | §2, modern statement |
| 4 | Raphan, M. and Simoncelli, E. P. (2011). *Least squares estimation without priors or supervision.* Neural Computation 23(2):374-420. | §3, exponential-family generalization |
| 5 | Mohan, S., Kadkhodaie, Z., Simoncelli, E. P. and Fernandez-Granda, C. (2020). *Robust and interpretable blind image denoising via bias-free CNNs.* ICLR. arXiv:1906.05478. | §4, bias-free construction and Jacobian-as-filter |
| 6 | Kadkhodaie, Z. and Simoncelli, E. P. (2021). *Stochastic solutions for linear inverse problems using the prior implicit in a denoiser.* NeurIPS. arXiv:2007.13640. | §1, §6, sampler and linear solver |
| 7 | Romano, Y., Elad, M. and Milanfar, P. (2017). *The little engine that could: regularization by denoising (RED).* SIAM J. Imaging Sci. 10(4):1804-1844. | §7, §9.2, guarantees that require conservativeness and passivity |
| 8 | Venkatakrishnan, S. V., Bouman, C. A. and Wohlberg, B. (2013). *Plug-and-play priors for model-based reconstruction.* IEEE GlobalSIP. | §7, PnP framing |
| 9 | Song, Y. and Ermon, S. (2019). *Generative modeling by estimating gradients of the data distribution (NCSN).* NeurIPS. arXiv:1907.05600. | §1, §7, score-based generation |
| 10 | Ho, J., Jain, A. and Abbeel, P. (2020). *Denoising diffusion probabilistic models.* NeurIPS. arXiv:2006.11239. | §1, §7 |
| 11 | Song, Y. et al. (2021). *Score-based generative modeling through stochastic differential equations.* ICLR. arXiv:2011.13456. | §1, §7, the SDE view |
| 12 | Kim, K. and Ye, J. C. (2021). *Noise2Score: Tweedie's approach to self-supervised image denoising without clean images.* NeurIPS. arXiv:2106.07009. | §3, §7, §20, explicit Tweedie across noise families |
| 13 | Lehtinen, J. et al. (2018). *Noise2Noise: learning image restoration without clean data.* ICML. arXiv:1803.04189. | §7, clean-data-free training |
| 14 | Soltanayev, S. and Chun, S. Y. (2018). *Training deep learning based denoisers without ground truth data.* NeurIPS. | §7, §20, SURE training |
| 15 | Chung, H., Kim, J., McCann, M. T., Klasky, M. L. and Ye, J. C. (2022). *Diffusion posterior sampling for general noisy inverse problems.* arXiv:2209.14687. | §7, §13, §17, nonlinear route |
| 16 | Chao, C.-H. et al. (2023). *On investigating the conservative property of score-based generative models.* ICML. arXiv:2209.12753. | §1, §7, §9, why sampling survives non-conservativeness |
| 17 | Hurault, S., Leclaire, A. and Papadakis, N. (2022). *Gradient step denoiser for convergent plug-and-play.* ICLR. arXiv:2110.03220. | §14.2, the exact-conservativeness construction |
| 18 | Hurault, S., Leclaire, A. and Papadakis, N. (2022). *Proximal denoiser for convergent plug-and-play optimization with nonconvex regularization.* ICML. arXiv:2201.13256. | §14.2, stronger PnP guarantees |
| 19 | Salimans, T. and Ho, J. (2021). *Should EBMs model the energy or the score?* Energy Based Models Workshop, ICLR. | §14.4, §20, §22, the measured cost of energy parameterization |
| 20 | MRED analysis, arXiv:2202.04961. | §9.2, the passivity requirement that blocks the RED rescue |
| 21 | Kawar, B., Elad, M., Ermon, S. and Song, J. (2022). *Denoising diffusion restoration models (DDRM).* NeurIPS. arXiv:2201.11793. | §5, §17, §20, noise-conditional linear solver |
| 22 | Wang, Y., Yu, J. and Zhang, J. (2023). *Zero-shot image restoration using denoising diffusion null-space model (DDNM).* ICLR. arXiv:2212.00490. | §5, §17, §20, null-space solver |
| 23 | Song, J., Vahdat, A., Mardani, M. and Kautz, J. (2023). *Pseudoinverse-guided diffusion models for inverse problems (PiGDM).* ICLR. | §5, §17, uncertain-operator route |
| 24 | Karras, T., Aittala, M., Aila, T. and Laine, S. (2022). *Elucidating the design space of diffusion-based generative models (EDM).* NeurIPS. arXiv:2206.00364. | §5.3, §19, few-step schedules |
| 25 | Hutchinson, M. F. (1989). *A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines.* Comm. Statist. Simulation Comput. 18(3):1059-1076. | §14.5, §16, divergence and asymmetry estimators |
| 26 | Stein, C. M. (1981). *Estimation of the mean of a multivariate normal distribution.* Annals of Statistics 9(6):1135-1151. | §7, §20, the lemma behind SURE |
| 27 | Ryu, E. K., Liu, J., Wang, S., Chen, X., Wang, Z. and Yin, W. (2019). *Plug-and-play methods provably converge with properly trained denoisers.* ICML. arXiv:1905.05406. | §7, §15, the spectral-normalization route |
| 28 | Zhang, K., Li, Y., Zuo, W., Zhang, L., Van Gool, L. and Timofte, R. (2021). *Plug-and-play image restoration with deep denoiser prior.* IEEE TPAMI. arXiv:2008.13751. | §17, practical multi-task restoration baseline |
| 29 | Anscombe, F. J. (1948). *The transformation of Poisson, binomial and negative-binomial data.* Biometrika 35:246-254. | §13, §17, variance stabilization |
| 30 | Kadkhodaie, Z., Guth, F., Simoncelli, E. P. and Mallat, S. (2024). *Generalization in diffusion models arises from geometry-adaptive harmonic representations.* ICLR. arXiv:2310.02557. | §7, §10, the geometric reading of the same operator |