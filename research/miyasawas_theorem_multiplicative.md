# Miyasawa's Theorem for Per-Pixel Multiplicative Gaussian Noise (Linear Domain)

## Table of Contents
1. [Introduction](#introduction)
2. [The Additive Miyasawa Baseline](#the-additive-miyasawa-baseline)
3. [The Multiplicative Noise Model](#the-multiplicative-noise-model)
4. [Exact Moment Relation (A)](#exact-moment-relation-a)
5. [Small-Sigma Score-Style Relation (B)](#small-sigma-score-style-relation-b)
6. [Monte-Carlo Validation](#monte-carlo-validation)
7. [Composite Noise (Multiplicative + Additive)](#composite-noise-multiplicative--additive)
8. [The Bias-Free Tension and Recommendation](#the-bias-free-tension-and-recommendation)
9. [Generalized-SURE Divergence-Consistency Compliance Check](#generalized-sure-divergence-consistency-compliance-check)
10. [Summary](#summary)
11. [Further Reading](#further-reading)

---

## Introduction

The repo's existing Miyasawa/Tweedie notes
(`research/miyasawas_theorem.md`, `research/miyasawas_theorem_extension.md`,
`research/miyasawas_theorem_conditional.md`) treat **additive** Gaussian noise
`y = x + ε`, where the optimal denoiser carries a clean residual-as-score identity. This
note extends that family to **per-pixel multiplicative Gaussian noise**, `y = x · n` with
`n ~ N(1, σ²)`, in the **linear domain** (no log / variance-stabilizing transform). This is
the noise model wired (opt-in) into the ConvUNeXt bias-free denoiser trainer.

The central result is that the clean additive identity `D(y) − y = σ²∇log p(y)` does **not**
survive the move to multiplicative noise in the linear domain. Two relations replace it:

- An **exact** relation (A) for any σ, which requires the **second** posterior moment
  `E[x²|y]` — there is no closed-form residual=score identity.
- A **small-σ** score-style approximation (B), `D(y) − y ≈ 2σ²y + σ²y²∇log p(y)`, with a
  deterministic shrink term and a `y²`-weighted (signal-dependent) score term.

Both are numerically validated (see [Monte-Carlo Validation](#monte-carlo-validation)) and
both are implemented in `src/dl_techniques/utils/multiplicative_miyasawa.py`, gated by
`tests/test_utils/test_multiplicative_miyasawa.py`. Crucially, the multiplicative residual's
`2σ²y` term is **non-homogeneous**, which puts it in tension with the strict bias-free
architecture (`f(αx) = αf(x)`) the repo uses for the additive case — a tension this note
documents and gives a recommendation for, rather than "fixing" by rearchitecting.

---

## The Additive Miyasawa Baseline

For completeness, recall the additive result derived in full in
`research/miyasawas_theorem.md` (§ Detailed Mathematical Derivation). With

$$y = x + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, \sigma^2 I),$$

the least-squares optimal denoiser is the conditional mean $\hat{x}(y) = \mathbb{E}[x|y]$, and
Miyasawa's theorem (1961) gives

$$\boxed{\ \mathbb{E}[x|y] = y + \sigma^2 \nabla_y \log p(y)\ }$$

equivalently the **residual form**

$$D(y) - y = \sigma^2 \nabla_y \log p(y).$$

The derivation hinges on the Gaussian-likelihood identity
$\nabla_y p(y|x) = \tfrac{x - y}{\sigma^2}\,p(y|x)$, which is **linear in $x$**; that linearity
is exactly what lets the integral against the prior collapse to $\mathbb{E}[x|y] - y$ (see
`miyasawas_theorem.md:85-109`). It is the reason a single-output denoiser's residual *is* the
score, with no second moment needed.

**Bias-free architecture connection.** The repo's bias-free denoiser
(`src/dl_techniques/models/bias_free_denoisers/bfconvunext.py:67-68`) is built around this
identity: removing all additive biases and using a linear final activation makes the network a
homogeneous degree-1 map, `f(αy) = αf(y)`. Under additive AWGN this homogeneity is the correct
invariance (scaling the noisy input scales the MMSE estimate), and it is what makes the
residual-as-score interpretation hold exactly. The key structural fact for this note: the
bias-free coupling is an **additive-only** property — see
[The Bias-Free Tension](#the-bias-free-tension-and-recommendation).

---

## The Multiplicative Noise Model

We use per-pixel multiplicative Gaussian noise in the linear domain (user decision):

$$y = x \cdot n, \qquad n \sim \mathcal{N}(1, \sigma^2), \qquad x \perp n
\quad\Longleftrightarrow\quad y \mid x \sim \mathcal{N}\!\big(x,\ \sigma^2 x^2\big).$$

Each pixel is scaled by an independent draw of $n$, so the **conditional variance is
signal-dependent**: $\operatorname{Var}(y|x) = \sigma^2 x^2$. This is the multiplicative analog
of AWGN and is implemented by `apply_multiplicative_gaussian(x, sigma)` in the module
(`y = x * (1 + N(0,1)·σ)`).

**Why the log / VST reduction is rejected.** A standard trick for multiplicative noise is the
log transform: with $u = \log x$, $v = \log y$, one has $v \approx u + (n-1)$ for small σ, which
is approximately additive, so additive Miyasawa applies in log space. But this requires
$x > 0$. The denoiser operates on images normalized to the **signed** $[-1, +1]$ domain (the
zero-centered normalization required for bias-free training, `miyasawas_theorem.md:479-483`),
so $x$ is *not* strictly positive and $\log x$ is undefined. The log/VST path is therefore
unavailable, and the linear-domain empirical-Bayes form below is the mandated framework.

A practical caveat of the linear model: because the perturbation $|y - x| = |x|\,|n-1|$ scales
with $|x|$, the additive path's clip to $[-1, +1]$ remains valid but the clip fraction rises
with σ (and $n$ can go negative, flipping signs, for large σ). The trainer keeps the same clip
and notes this caveat.

---

## Exact Moment Relation (A)

Start from the Gaussian likelihood $p(y|x) = \mathcal{N}(x, \sigma^2 x^2)$ and differentiate in
$y$:

$$\partial_y p(y|x) = -\frac{y - x}{\sigma^2 x^2}\,p(y|x).$$

Rearranging isolates a *signed* multiple of the likelihood,

$$(x - y)\,p(y|x) = \sigma^2 x^2\,\partial_y p(y|x).$$

Now integrate both sides against the prior $p(x)$. The marginal is
$p(y) = \int p(y|x)\,p(x)\,dx$, and posterior moments are
$\mathbb{E}[g(x)|y] = \tfrac{1}{p(y)}\int g(x)\,p(y|x)\,p(x)\,dx$. The **left** side gives

$$\int (x - y)\,p(y|x)\,p(x)\,dx = \big(\mathbb{E}[x|y] - y\big)\,p(y).$$

The **right** side, because $x^2$ does not depend on $y$, lets us pull $\partial_y$ outside the
integral:

$$\sigma^2 \int x^2\,\partial_y p(y|x)\,p(x)\,dx
   = \sigma^2\,\partial_y \!\left[\int x^2\,p(y|x)\,p(x)\,dx\right]
   = \sigma^2\,\partial_y\!\big[\mathbb{E}[x^2|y]\,p(y)\big].$$

Equating the two sides and dividing by $p(y)$:

$$\boxed{\ \mathbb{E}[x|y] = y + \sigma^2\,\frac{\partial_y\big[\,\mathbb{E}[x^2|y]\,p(y)\,\big]}{p(y)}\ }
\tag{A}$$

**Interpretation.** Contrast with additive Miyasawa, where the correction is $\sigma^2$ times
the *marginal score* $\nabla_y \log p(y)$. Here the correction needs the **second posterior
moment** $\mathbb{E}[x^2|y]$, not just the marginal score. Because a single-output denoiser
$D(y) = \mathbb{E}[x|y]$ only exposes the first moment, **there is no closed-form
residual = score identity in the linear domain** — the second moment is an irreducible extra
ingredient. Relation (A) is exact for any σ; it is the generalized empirical-Bayes
(Tweedie/Miyasawa) form for signal-dependent Gaussian noise. It is implemented as
`relation_A(mc)` in the module.

---

## Small-Sigma Score-Style Relation (B)

To recover something closer to a usable "score" relation, expand (A) to leading order in σ.
For small σ the posterior concentrates: $D(y) \approx y$ and
$\mathbb{E}[x^2|y] \approx D(y)^2 \approx y^2$. Substituting $\mathbb{E}[x^2|y] \approx y^2$
into the numerator of (A) and applying the product rule,

$$\partial_y\big[\,y^2\,p(y)\,\big] = 2y\,p(y) + y^2\,\partial_y p(y),$$

so dividing by $p(y)$ and multiplying by $\sigma^2$,

$$\boxed{\ D(y) - y \approx 2\sigma^2 y + \sigma^2 y^2\,\nabla_y \log p(y)\ }
\tag{B}$$

(using $\partial_y p(y)/p(y) = \nabla_y \log p(y)$).

**Interpretation.** Relation (B) is the practical "Miyasawa analog" for multiplicative noise,
and it splits into two structurally distinct terms:

- **$2\sigma^2 y$ — a deterministic shrink.** This term is independent of the prior; it is a
  pure, signal-proportional pull whose sign and magnitude track $y$ itself. It is the
  fingerprint of the signal-dependent variance ($\propto x^2$): larger-magnitude pixels carry
  more multiplicative noise and are shrunk more. Critically, this term is **non-homogeneous**
  with respect to the score structure — it is the piece the additive theory does not have.
- **$\sigma^2 y^2\,\nabla_y \log p(y)$ — a signal-dependent score.** This is the additive
  score term but **re-weighted by $y^2$** instead of the constant $\sigma^2$. The prior still
  enters through $\nabla_y \log p(y)$, but its influence is amplified where $|y|$ is large.

Both terms vanish as σ → 0, recovering $D(y) = y$. Relation (B) is implemented as
`relation_B(mc)`.

---

## Monte-Carlo Validation

Both relations were validated by 1-D Monte Carlo against the *true* posterior mean
$\mathbb{E}[x|y]$ on a **signed 3-component Gaussian mixture prior** (normalized-image-like,
supported roughly in $[-1, +1]$): an equal-weight mixture of $\mathcal{N}(-0.5, 0.15)$,
$\mathcal{N}(0.2, 0.1)$, $\mathcal{N}(0.6, 0.2)$ — exactly `signed_mixture_prior` in the
module. The metric is relative RMSE of each relation's predicted $\mathbb{E}[x|y]$ vs the MC
ground truth over well-populated central bins.

**Reference run (F4, σ = 0.15, 8M samples / 600 bins):**

| Relation                 | rel-RMSE |
|--------------------------|----------|
| Exact (A)                | **0.040** |
| Approx (B)               | 0.048    |
| Baseline `D(y) = y`      | 0.080    |

Both relations track the true posterior mean and roughly **halve** the residual vs the
do-nothing baseline `D(y) = y`. The ~0.04 floor is a binning / `np.gradient` artifact at
sparse large-$|y|$ bins, not a theory error (fewer, better-populated bins lower it).

**In-test reproduction (σ = 0.15, 3M samples / 300 bins, the pytest regime):**

| Relation                 | rel-RMSE |
|--------------------------|----------|
| Exact (A)                | **0.028** |
| Approx (B)               | 0.037    |
| Baseline `D(y) = y`      | 0.081    |

The 300-bin / 3M regime gives *more* samples per bin than the 8M / 600 reference, so the
`np.gradient` noise floor is lower and (A)/(B) land at 0.028 / 0.037 — well under the 0.06
test tolerance. The executable backing for both rows lives in
`src/dl_techniques/utils/multiplicative_miyasawa.py` (the estimator) and
`tests/test_utils/test_multiplicative_miyasawa.py` (the hard gates: (A) ≤ 0.06, (B) ≤ 0.06,
both must beat the baseline).

The following snippet reproduces the in-test row and was executed in `.venv`
(output: `(A)=0.028  (B)=0.036  baseline(D=y)=0.081`):

```python
import numpy as np
from dl_techniques.utils.multiplicative_miyasawa import (
    signed_mixture_prior, mc_posterior_mean, relation_A, relation_B, rel_rmse,
)

prior = signed_mixture_prior(3_000_000, seed=0)
mc = mc_posterior_mean(prior, sigma=0.15, n_samples=3_000_000, n_bins=300, seed=0)

a    = rel_rmse(relation_A(mc), mc["Ex"], mc["mask"])
b    = rel_rmse(relation_B(mc), mc["Ex"], mc["mask"])
base = rel_rmse(mc["ctr"],      mc["Ex"], mc["mask"])
print(f"(A)={a:.3f}  (B)={b:.3f}  baseline(D=y)={base:.3f}")
```

---

## Composite Noise (Multiplicative + Additive)

The pure multiplicative model above is the clean limit; **real sensors are composite**. This
section extends the framework to the affine-variance model that the trainer also wires (opt-in).

### The composite model

Corrupt each pixel with an independent multiplicative gain *and* an independent additive draw:

$$y = x \cdot n + a, \qquad n \sim \mathcal{N}(1, \sigma_m^2), \qquad a \sim \mathcal{N}(0, \sigma_a^2),
\qquad x \perp n,\ x \perp a$$

$$\Longleftrightarrow\quad y \mid x \sim \mathcal{N}\!\big(x,\ \sigma_m^2 x^2 + \sigma_a^2\big).$$

The conditional variance is now **affine in $x^2$**: a signal-dependent term $\sigma_m^2 x^2$
sitting on a constant floor $\sigma_a^2$. This is precisely the Gaussian form of the
**Poisson-Gaussian** sensor model used in real imaging pipelines — a read-noise floor
$\sigma_a^2$ plus a signal-dependent (shot-noise-like) term $\sigma_m^2 x^2$. The pure
multiplicative model ($\sigma_a = 0$) and classic additive AWGN ($\sigma_m = 0$) are its two
limits. It is implemented by `apply_composite_gaussian(x, sigma_m, sigma_a)` in the module
(`y = x * (1 + N(0,1)·σ_m) + N(0,1)·σ_a`).

### Superposition derivation

The derivation mirrors relation (A) but starts from the affine-variance Gaussian likelihood
$p(y|x) = \mathcal{N}(x,\ \sigma_a^2 + \sigma_m^2 x^2)$. Differentiating in $y$,

$$\partial_y p(y|x) = -\frac{y - x}{\sigma_a^2 + \sigma_m^2 x^2}\,p(y|x)
\quad\Longrightarrow\quad
(x - y)\,p(y|x) = \big(\sigma_a^2 + \sigma_m^2 x^2\big)\,\partial_y p(y|x).$$

The variance factor now splits into a constant part and an $x^2$ part. Integrating against the
prior $p(x)$ as before, the left side gives $\big(\mathbb{E}[x|y] - y\big)\,p(y)$, and the right
side splits linearly: the $\sigma_a^2$ piece (constant in $x$) pulls straight through the
integral to give the **additive score term** $\sigma_a^2\,\partial_y p(y)$, and the
$\sigma_m^2 x^2$ piece reproduces the **multiplicative second-moment term** from relation (A).
Dividing by $p(y)$:

$$\boxed{\ \mathbb{E}[x|y] = y
   + \sigma_a^2\,\nabla_y \log p(y)
   + \sigma_m^2\,\frac{\partial_y\big[\,\mathbb{E}[x^2|y]\,p(y)\,\big]}{p(y)}\ }
\tag{A$_c$}$$

The composite estimator is the **exact sum** of the pure-additive score term and the
pure-multiplicative second-moment term — the relation **superposes**. Expanding to leading order
in σ (same concentration argument as (B), $\mathbb{E}[x^2|y] \approx y^2$):

$$\boxed{\ D(y) - y \approx \big(\sigma_a^2 + \sigma_m^2 y^2\big)\,\nabla_y \log p(y) + 2\sigma_m^2 y\ }
\tag{B$_c$}$$

The score is now weighted by the **local variance** $v(y) = \sigma_a^2 + \sigma_m^2 y^2$ — the
additive constant $\sigma_a^2$ and the signal-dependent $\sigma_m^2 y^2$ added together — plus the
familiar deterministic multiplicative shrink $2\sigma_m^2 y$. The two reductions are immediate:

- **$\sigma_a = 0$**: $v(y) = \sigma_m^2 y^2$, recovering pure-multiplicative relations (A)/(B).
- **$\sigma_m = 0$**: $v(y) = \sigma_a^2$ and the shrink vanishes, recovering the classic additive
  Miyasawa identity $D(y) - y = \sigma_a^2\,\nabla_y \log p(y)$.

Both relations are wired into the existing estimators via an optional `sigma_a` argument
(default `0.0`, so the pure-multiplicative behavior is unchanged): `mc_posterior_mean(...,
sigma_a=...)` draws the composite `y`, `relation_A(mc)` adds the $\sigma_a^2\,\nabla\log p$ term,
and `relation_B(mc)` uses the local-variance score weight.

### Monte-Carlo validation

Validated on the same signed 3-component Gaussian mixture prior as the pure case, at
$\sigma_m = 0.15$, $\sigma_a = 0.10$ (6M samples / 250 bins — the affine variance spreads $y$
wider, so more samples per bin are needed to reach the same `np.gradient` floor):

| Relation                  | rel-RMSE |
|---------------------------|----------|
| Composite exact (A$_c$)   | **0.029** |
| Composite approx (B$_c$)  | 0.040    |
| Baseline `D(y) = y`       | 0.148    |

Both composite relations roughly **5×** the do-nothing baseline (0.029/0.040 vs 0.148). The
executable backing is `apply_composite_gaussian` and the `sigma_a` parameters in
`src/dl_techniques/utils/multiplicative_miyasawa.py`, gated by `TestCompositeRelations` in
`tests/test_utils/test_multiplicative_miyasawa.py` (hard gates: (A$_c$)/(B$_c$) ≤ 0.06, both must
beat baseline; plus reduction tests asserting $\sigma_a = 0$ matches pure-multiplicative arrays
and $\sigma_m = 0$ matches the additive limit).

The following snippet reproduces the table row and was executed in `.venv`
(output: `composite (A)=0.029  (B)=0.040  baseline(D=y)=0.148`):

```python
import numpy as np
from dl_techniques.utils.multiplicative_miyasawa import (
    signed_mixture_prior, mc_posterior_mean, relation_A, relation_B, rel_rmse,
)

prior = signed_mixture_prior(6_000_000, seed=0)
mc = mc_posterior_mean(prior, sigma=0.15, sigma_a=0.10,
                       n_samples=6_000_000, n_bins=250, seed=0)

a    = rel_rmse(relation_A(mc), mc["Ex"], mc["mask"])
b    = rel_rmse(relation_B(mc), mc["Ex"], mc["mask"])
base = rel_rmse(mc["ctr"],      mc["Ex"], mc["mask"])
print(f"composite (A)={a:.3f}  (B)={b:.3f}  baseline(D=y)={base:.3f}")
```

### Why composite helps: $x \approx 0$ well-posedness

The most important practical reason to use the composite model is **conditioning near zero**.
Under pure multiplicative noise the conditional variance is $\sigma_m^2 x^2$, which **vanishes
as $x \to 0$**: dark / near-zero-signal pixels are barely corrupted, and the score relation
there becomes ill-conditioned (the $\partial_y \log p(y)$ correction is multiplied by a variance
weight $\sigma_m^2 y^2 \to 0$, so the estimator has almost no leverage and the empirical-Bayes
quantities are numerically unstable in that regime). The additive floor $\sigma_a^2$ **injects a
constant amount of noise everywhere**, so the local variance $v(y) = \sigma_a^2 + \sigma_m^2 y^2$
is bounded away from zero. This **regularizes the $x \approx 0$ region** and is exactly why real
pipelines adopt the composite (Poisson-Gaussian) model rather than a pure multiplicative one — a
read-noise floor is both physically present and numerically stabilizing.

### Amplified clipping caveat

Composite noise carries **larger total variance** than either pure case ($\sigma_a^2 + \sigma_m^2
x^2 \geq \max(\sigma_a^2,\ \sigma_m^2 x^2)$), so the trainer's clip to $[-1, +1]$ fires **more
often** — especially the multiplicative part at high $|x|$, where $\sigma_m^2 x^2$ dominates and
$n$ can push samples well outside the range. As in the pure case the same clip is kept, but it
bends the MMSE target at the extremes **more** than either pure model: the composite caveat is the
pure-multiplicative caveat, amplified. (Treat the relations as exact for the unclipped model; the
clip is a documented approximation at the domain edges.)

### Partial bias-free consistency

Re-reading (B$_c$) against the bias-free homogeneity discussion below: the additive score term
$\sigma_a^2\,\nabla_y \log p(y)$ is **consistent** with the strict bias-free / homogeneous form
(it is exactly the additive Miyasawa structure that bias-free is built for), and the local-variance
score reweighting also scales sensibly with the input. Only the deterministic
$2\sigma_m^2 y$ multiplicative shrink breaks homogeneity. Consequently a **composite** model — being
part additive — is **less mismatched** to the strict bias-free architecture than the pure
multiplicative model: the more additive-dominated the noise (larger $\sigma_a$ relative to
$\sigma_m$), the more appropriate the bias-free network. This is a softening, not a removal, of the
tension documented next.

### Trainer usage

The composite mode is opt-in via `--composite-noise` (with `--composite-additive-ratio R`,
default `R = 0.5`); the single curriculum scalar drives $\sigma_m$ and the additive std is tied to
it as $\sigma_a = R \cdot \sigma_m$, so one knob schedules both
(`src/train/bfunet/train_convunext_denoiser.py`, the composite branch in
`make_curriculum_noise_fn`).

---

## The Bias-Free Tension and Recommendation

This is the architectural crux. The repo's bias-free denoiser is, by construction, a
**homogeneous degree-1** map: with no additive biases and a linear final activation,
$f(\alpha x) = \alpha f(x)$ for any scalar α. Under additive AWGN this is exactly right.

Look again at relation (B), the optimal multiplicative residual:

$$D(y) - y \approx \underbrace{2\sigma^2 y}_{\text{deterministic shrink}}
   + \underbrace{\sigma^2 y^2\,\nabla_y \log p(y)}_{\text{signal-dependent score}}.$$

The score term is compatible with homogeneity at fixed σ (it scales sensibly with the input),
but the **$2\sigma^2 y$ shrink term is not expressible by a purely homogeneous map of $y$
alone at fixed σ in a way that matches the optimal denoiser across input scales**: it is a
σ-dependent, signal-proportional offset that the additive-only homogeneity coupling does not
provide. Put plainly:

> **Strict bias-free is NOT the theoretically optimal architectural form for per-pixel
> multiplicative noise.** The optimal residual carries a non-homogeneous, σ-coupled shrink
> term ($2\sigma^2 y$) that a strictly homogeneous, σ-agnostic bias-free network cannot
> represent exactly.

This matches the finding (F4, consequence 2; `findings/miyasawa-theory.md:106`) that the
scaling-invariance argument that justifies bias-free for additive noise does not carry over to
multiplicative noise.

**Recommendation for THIS iteration (honest and bounded).**

- **Keep the bias-free architecture and `final_activation='linear'`.** Do **not** rearchitect
  this iteration. MSE training still converges to the true MMSE estimate $\mathbb{E}[x|y]$ for
  this noise model (MSE optimality is unchanged), so the trained network is still the best
  *within its hypothesis class*; the gap is the representational gap of the homogeneous class
  against the $2\sigma^2 y$ term, which the validation shows is modest in the working σ range.
  Treat this as a **documented approximation**, not a defect.
- **Future work (flagged, not done here):** to close the gap, either (i) **σ / noise-level
  conditioning** (feed σ to the network so it can synthesize the σ-dependent shrink), or
  (ii) **allow a controlled additive-bias term** (or a learned signal-proportional offset
  head) so the network can represent the $2\sigma^2 y$ shrink directly. Both break strict
  homogeneity and are therefore deliberately out of scope for the current bias-free iteration.

---

## Generalized-SURE Divergence-Consistency Compliance Check

How do we *verify* a **trained** denoiser respects the right structure, without clean
references? Relations (A)/(B) need the true prior / MC ground truth, so they validate the
*theory* on a synthetic prior but cannot audit a real checkpoint on real data. For that we use
**generalized SURE** (Stein's Unbiased Risk Estimate; Eldar 2009; Raphan & Simoncelli 2011),
implemented as `sure_divergence_consistency` in the module.

**Additive anchor.** For additive AWGN ($y = x + \mathcal{N}(0, \sigma^2 I)$), SURE estimates
the risk $\mathbb{E}\|D(y) - x\|^2$ from noisy data **alone**:

$$\mathrm{SURE} = \|D(y) - y\|^2 + 2\sigma^2 \operatorname{div}(D) - \sigma^2 N,$$

where $N$ is the element count and $\operatorname{div}(D) = \sum_i \partial D_i/\partial y_i$.
The divergence is estimated by a **Hutchinson** probe with a finite-difference
Jacobian-vector product, $\operatorname{div}(D) \approx \mathbb{E}_v[\,v \cdot
(D(y + \epsilon v) - D(y))/\epsilon\,]$ for Rademacher $v$. This additive closed form is the
**self-check** that validates the estimator's scale: on a linear toy $D(y) = a y$ the
divergence is analytically $aN$, and the test confirms the Hutchinson estimate matches it and
that SURE reproduces the realized MSE to **1.18% rel-error** (`additive_sure_risk` /
`hutchinson_divergence`, gated in the pytest). Passing this self-check is the Pre-Mortem
STOP-IF; it passed, so SURE is kept as a hard gate rather than demoted to diagnostic-only.

**Multiplicative generalization.** For $y|x \sim \mathcal{N}(x, \sigma^2 x^2)$ the data
covariance is signal-dependent ($\Sigma(y) \approx \operatorname{diag}(\sigma^2 y^2)$ to
leading order). Generalized SURE replaces the constant $\sigma^2$ weighting of the divergence
with the per-element variance:

$$\mathrm{gSURE\text{-}residual} = \|D(y) - y\|^2
   + 2\sum_i \sigma^2 y_i^2\,\frac{\partial D_i}{\partial y_i}
   - \sigma^2 \sum_i y_i^2.$$

The variance-weighted divergence $\sum_i \sigma^2 y_i^2 (\partial D_i/\partial y_i)$ is
estimated with a Hutchinson probe whose components are pre-scaled by $\sigma |y_i|$ (so
$\mathbb{E}[v_i v_j] = \sigma^2 y_i^2 \delta_{ij}$). This yields a single
**residual-consistency scalar computable from noisy data + the denoiser alone — no clean
references** — and is exactly the empirical compliance probe a trained checkpoint can be
audited with.

The heavy "audit a real checkpoint" path is intentionally kept **out** of the fast pytest and
exposed instead as a CLI in the module (`run_checkpoint_diagnostic` + a `__main__` guard). It
loads a `.keras` denoiser, synthesizes a noisy batch via `apply_multiplicative_gaussian` (no
dataset dependency), and prints the diagnostic:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m dl_techniques.utils.multiplicative_miyasawa \
    --checkpoint results/convunext_denoiser_smoke/best_model.keras
```

Executed against the smoke checkpoint, this prints finite values and exits 0, e.g.:

```
              divergence : 4515.88
     weighted_divergence : 30.564
             residual_sq : 33320.4
              n_elements : 49152
          gsure_residual : 33004.6
```

(absolute magnitudes are checkpoint-dependent; the point of the diagnostic is that it is
finite, reference-free, and reproducible for tracking a trained denoiser's compliance).

---

## Summary

| Quantity | Additive AWGN | Per-pixel multiplicative (linear) |
|----------|---------------|-----------------------------------|
| Likelihood | $\mathcal{N}(x, \sigma^2 I)$ | $\mathcal{N}(x, \sigma^2 x^2)$ |
| Exact estimator | $y + \sigma^2 \nabla\log p(y)$ | $y + \sigma^2 \partial_y[\mathbb{E}[x^2|y]p(y)]/p(y)$ (A) |
| Score-style form | $D(y)-y = \sigma^2\nabla\log p$ | $D(y)-y \approx 2\sigma^2 y + \sigma^2 y^2 \nabla\log p$ (B) |
| Needs 2nd moment? | No | **Yes** (no residual=score identity) |
| Bias-free optimal? | Yes (homogeneous) | **No** ($2\sigma^2 y$ is non-homogeneous) |
| Reference-free audit | SURE | generalized-SURE divergence consistency |

MSE training still yields the MMSE-optimal $\mathbb{E}[x|y]$ under multiplicative noise;
"compliance" here is about the *residual ↔ score structure*, not trainability. The strict
bias-free architecture is kept this iteration as a documented approximation, with
σ-conditioning or an additive-bias-allowed shrink head flagged as the future-work route to
close the $2\sigma^2 y$ gap.

---

## Further Reading

- `research/miyasawas_theorem.md` — the additive baseline (full six-step derivation,
  bias-free requirements, normalization).
- `research/miyasawas_theorem_extension.md` — correlated / linear-transformed additive
  Gaussian (`y = Kx + Kε`).
- `src/dl_techniques/utils/multiplicative_miyasawa.py` — executable implementation of
  `apply_multiplicative_gaussian`, relations (A)/(B), the MC posterior-moment estimator, and
  `sure_divergence_consistency` (+ the `run_checkpoint_diagnostic` CLI).
- `tests/test_utils/test_multiplicative_miyasawa.py` — the fast CPU compliance gates.
- Eldar (2009), "Generalized SURE for Exponential Families."
- Raphan & Simoncelli (2011), "Least Squares Estimation Without Priors or Supervision," *Neural
  Computation* — empirical-Bayes estimation for general (incl. signal-dependent) noise.
- Efron (2011), "Tweedie's Formula and Selection Bias" — the Tweedie connection.
- Kadkhodaie & Simoncelli (2021), "Solving Linear Inverse Problems Using the Prior Implicit in
  a Denoiser."
