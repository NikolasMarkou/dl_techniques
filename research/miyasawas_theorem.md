# Miyasawa's Theorem (Tweedie's Formula)

A reference on the empirical-Bayes identity that links the least-squares optimal denoiser to the
score of the noisy density, its exact extensions, and the practice built on top of it: score
matching, diffusion sampling, denoiser-based priors for inverse problems, bias-free architectures,
and reference-free diagnostics.

## Table of Contents

1. [Overview](#1-overview)
2. [Notation](#2-notation)
3. [Statement and hypotheses](#3-statement-and-hypotheses)
4. [Derivation](#4-derivation)
5. [Second-order Tweedie: the Jacobian is the posterior covariance](#5-second-order-tweedie-the-jacobian-is-the-posterior-covariance)
6. [Equivalent forms and parameterizations](#6-equivalent-forms-and-parameterizations)
7. [Score matching: why MSE denoising learns the score](#7-score-matching-why-mse-denoising-learns-the-score)
8. [Conditional form](#8-conditional-form)
9. [Classifier-free guidance](#9-classifier-free-guidance)
10. [Extension: linear transformation and correlated noise](#10-extension-linear-transformation-and-correlated-noise)
11. [Extension: signal-dependent noise](#11-extension-signal-dependent-noise)
12. [Architecture](#12-architecture)
13. [Data domain and normalization](#13-data-domain-and-normalization)
14. [Sampling with a denoiser](#14-sampling-with-a-denoiser)
15. [Inverse problems](#15-inverse-problems)
16. [Diagnostics](#16-diagnostics)
17. [Limitations](#17-limitations)
18. [Reference implementation](#18-reference-implementation)
19. [Quick reference](#19-quick-reference)
20. [Conventions checklist](#20-conventions-checklist)
21. [History and references](#21-history-and-references)

---

## 1. Overview

Given a signal corrupted by additive Gaussian noise, the estimator that minimizes mean squared
error is the posterior mean. Miyasawa's theorem says that this estimator's residual, the vector by
which it moves the noisy observation, is exactly the gradient of the log-density of the noisy
observations, scaled by the noise variance:

$$\mathbb{E}[x \mid y] - y \;=\; \sigma^2\,\nabla_y \log p(y).$$

Three things follow, and they are the reason the result matters far outside classical estimation:

- **A denoiser is a score estimator.** Train any network with MSE to remove Gaussian noise and its
  residual field approximates the score of the smoothed data density. No density model, no
  normalizing constant, no explicit energy function.
- **A denoiser is an implicit prior.** The score can be integrated by Langevin dynamics or a
  reverse-time SDE to draw samples, or combined with a data-fidelity term to solve inverse problems.
  Diffusion models are this observation applied across a continuum of noise levels.
- **The identity is exact and assumption-light.** It requires additive Gaussian noise, independence,
  and a finite first moment. It requires nothing of the prior (which may be discrete or supported on
  a measure-zero manifold) and nothing of the estimator's implementation.

---

## 2. Notation

| Symbol | Meaning |
|---|---|
| $x \in \mathbb{R}^N$ | clean signal, drawn from a prior $p(x)$ |
| $\varepsilon$ | noise |
| $y$ | noisy observation |
| $p(y)$ | density of the **noisy** observation |
| $\hat{x}(y) = \mathbb{E}[x\mid y]$ | the MMSE estimator (a mathematical object) |
| $D_\theta$ | a trained network approximating $\hat{x}$ |
| $s(y) = \nabla_y \log p(y)$ | score of the noisy density |
| $E(y) = -\log p(y)$ | energy, defined up to an additive constant |
| $J(y) = \partial\hat{x}/\partial y$ | Jacobian of the estimator |
| $c$ | conditioning variable (class label, image, embedding) |

Two distinctions worth holding onto:

- $\nabla_y \log p(y)$ is the score of the **noise-smoothed** density
  $p_\sigma = p_x * \mathcal{N}(0,\sigma^2 I)$, not of the clean prior. It approaches the clean score
  only as $\sigma \to 0$, and $\nabla \log p_x$ may fail to exist when the data lie on a manifold.
- $\hat{x}$ is the exact posterior mean; $D_\theta$ is a finite-capacity approximation. Every identity
  below is exact for $\hat{x}$ and approximate for $D_\theta$.

---

## 3. Statement and hypotheses

**Setup.**

$$y = x + \varepsilon, \qquad x \sim p(x), \qquad \varepsilon \sim \mathcal{N}(0,\sigma^2 I), \qquad x \perp \varepsilon.$$

**Theorem.**

$$\boxed{\ \hat{x}(y) \;=\; \mathbb{E}[x\mid y] \;=\; y + \sigma^2\,\nabla_y \log p(y).\ }$$

**Hypotheses.**

1. $\mathbb{E}\|x\| < \infty$, so the conditional mean exists.
2. The noise is additive Gaussian with known, signal-independent covariance $\sigma^2 I$, independent
   of $x$.

Nothing further is needed. Because $p(y)$ is a convolution with a Gaussian, it is automatically
smooth and strictly positive everywhere, so differentiation under the integral sign is justified by
dominated convergence, and division by $p(y)$ is always legal. There is no smoothness, positivity, or
support condition on the prior.

The theorem constrains the exact estimator only. It says nothing about architecture, activation
functions, or optimization; those questions are treated in [§12](#12-architecture).

---

## 4. Derivation

**Step 1. The MMSE estimator is the posterior mean.**

$$\arg\min_{f}\ \mathbb{E}\big[\|f(y)-x\|^2\big] \;=\; \mathbb{E}[x\mid y] \;=\; \int x\,p(x\mid y)\,dx.$$

**Step 2. Gradient of the Gaussian likelihood.**

$$p(y\mid x) = (2\pi\sigma^2)^{-N/2}\exp\!\Big(-\frac{\|y-x\|^2}{2\sigma^2}\Big),
\qquad
\nabla_y\, p(y\mid x) = \frac{x-y}{\sigma^2}\,p(y\mid x).$$

The structural fact that makes everything work: this gradient is **linear in $x$**, so integrating it
against the prior produces a first moment and nothing else.

**Step 3. Differentiate the marginal.**

$$p(y) = \int p(y\mid x)\,p(x)\,dx
\qquad\Longrightarrow\qquad
\nabla_y\,p(y) = \frac{1}{\sigma^2}\int (x-y)\,p(y\mid x)\,p(x)\,dx.$$

**Step 4. Convert to a posterior expectation** using $p(y\mid x)p(x) = p(x\mid y)p(y)$:

$$\nabla_y\,p(y) = \frac{p(y)}{\sigma^2}\int (x-y)\,p(x\mid y)\,dx = \frac{p(y)}{\sigma^2}\big(\mathbb{E}[x\mid y]-y\big).$$

**Step 5. Divide by $p(y)>0$.**

$$\nabla_y \log p(y) = \frac{1}{\sigma^2}\big(\hat{x}(y)-y\big)
\qquad\Longleftrightarrow\qquad
\hat{x}(y) = y + \sigma^2\nabla_y \log p(y). \qquad \blacksquare$$

---

## 5. Second-order Tweedie: the Jacobian is the posterior covariance

Differentiating the identity a second time gives a result at least as useful as the first:

$$\boxed{\ J(y) \;=\; \frac{\partial\hat{x}}{\partial y} \;=\; I + \sigma^2\nabla_y^2\log p(y) \;=\; \frac{1}{\sigma^2}\operatorname{Cov}[x\mid y].\ }$$

Consequences:

- **The Jacobian of the exact MMSE denoiser is symmetric and positive semidefinite, automatically.**
  Symmetry is forced, not assumed. It becomes an assumption only when the exact estimator is replaced
  by a trained network, which is where RED-style methods run into trouble ([§15](#15-inverse-problems)).
- The eigenvalues of $\sigma^2\nabla^2\log p(y)$ are bounded below by $-1$.
- **Uncertainty comes for free.** $\operatorname{Cov}[x\mid y] = \sigma^2 J(y)$, so a Jacobian-vector
  product against a trained denoiser yields directional posterior variance without an extra head, an
  ensemble, or a second training run.
- **Risk decomposes.** $\mathrm{MMSE} = \mathbb{E}\big[\operatorname{tr}\operatorname{Cov}[x\mid y]\big]
  = \sigma^2\,\mathbb{E}\big[\operatorname{tr}J(y)\big]$, which is precisely the divergence term that
  appears in SURE ([§16](#16-diagnostics)).

---

## 6. Equivalent forms and parameterizations

**Residual form.**

$$\hat{x}(y) - y = \sigma^2 \nabla_y \log p(y).$$

**Score readout from a trained denoiser.**

$$s(y) \;\approx\; \frac{D_\theta(y)-y}{\sigma^2}.$$

**Energy form**, with $E(y) = -\log p(y)$ up to a constant:

$$\hat{x}(y) - y = -\sigma^2 \nabla_y E(y).$$

A single denoiser evaluation is therefore one gradient-descent step on the energy of the noisy
density, with step size $\sigma^2$. The energy is only ever accessed through its gradient, so the
normalizing constant never appears. That is the entire practical appeal of score-based methods.

**Noise-prediction form.** With $\hat{\varepsilon}(y) = \mathbb{E}[\varepsilon\mid y] = y - \hat{x}(y)$:

$$s(y) = -\frac{\hat{\varepsilon}(y)}{\sigma^2}.$$

If the network instead predicts *unit-variance* noise, $\varepsilon_\theta(y)\approx \varepsilon/\sigma$
(the DDPM convention), then

$$s(y) = -\frac{\varepsilon_\theta(y)}{\sigma}.$$

Confusing these two costs a factor of $\sigma$ and is the most common implementation bug in this area.

**Variance-preserving (DDPM) scaling.** With $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon$
and $\varepsilon\sim\mathcal{N}(0,I)$, the conditional law is
$x_t\mid x_0 \sim \mathcal{N}(\sqrt{\bar\alpha_t}x_0,\ (1-\bar\alpha_t)I)$. Applying the theorem to the
mean $u = \sqrt{\bar\alpha_t}x_0$ and dividing through:

$$\boxed{\ \mathbb{E}[x_0\mid x_t] \;=\; \frac{x_t + (1-\bar\alpha_t)\,\nabla_{x_t}\log p_t(x_t)}{\sqrt{\bar\alpha_t}}.\ }$$

Substituting $\nabla\log p_t = -\varepsilon_\theta/\sqrt{1-\bar\alpha_t}$ recovers the standard
"predict $x_0$ from $\varepsilon_\theta$" formula. The $x_0$-prediction, $\varepsilon$-prediction and
$v$-prediction parameterizations are all this identity rearranged.

**Variance-exploding form.** For $x_t = x_0 + \sigma_t\varepsilon$ the identity applies unchanged with
$\sigma \to \sigma_t$, which is why VE models can use a single blind denoiser across the whole schedule.

---

## 7. Score matching: why MSE denoising learns the score

**Explicit score matching** is the objective one would like to minimize:

$$\mathcal{L}_{\mathrm{ESM}}(\theta) = \mathbb{E}_{p(y)}\big\|s_\theta(y) - \nabla_y \log p(y)\big\|^2 .$$

It is not computable, since $\nabla\log p$ is unknown. Hyvärinen's identity rewrites it, up to a
constant, as $\mathbb{E}\big[\operatorname{tr}\nabla s_\theta(y) + \tfrac12\|s_\theta(y)\|^2\big]$, which
is computable but needs a Jacobian trace per sample and scales poorly in high dimension.

**Denoising score matching** replaces the unknown marginal score with the known conditional one,
$\nabla_y \log p(y\mid x) = (x-y)/\sigma^2$:

$$\mathcal{L}_{\mathrm{DSM}}(\theta) = \mathbb{E}_{x,\varepsilon}\Big\|s_\theta(y) - \frac{x-y}{\sigma^2}\Big\|^2,
\qquad y = x+\varepsilon .$$

$\mathcal{L}_{\mathrm{DSM}}$ and $\mathcal{L}_{\mathrm{ESM}}$ differ by a constant independent of $\theta$,
so they share a minimizer.

**The link to ordinary MSE denoising is an equality, not an analogy.** Substituting the residual
parameterization $s_\theta(y) = \big(D_\theta(y)-y\big)/\sigma^2$:

$$\mathcal{L}_{\mathrm{DSM}}(\theta) \;=\; \frac{1}{\sigma^4}\,\mathbb{E}\big\|D_\theta(y)-x\big\|^2 .$$

Training a denoiser with MSE **is** denoising score matching, up to the constant $\sigma^{-4}$. This is
the reason a trained denoiser yields a score estimate, and it is also why the loss matters: MSE is
minimized by the conditional mean, whereas L1 gives the conditional median and perceptual or
adversarial losses give neither. Of all the design choices in a denoising pipeline, the loss is the
one the theorem genuinely dictates.

**Multiple noise levels.** In practice a single network is trained over a range of $\sigma$, either
conditioned on $\sigma$ or blind. Weight the per-level objectives by $\lambda(\sigma)\propto\sigma^2$ in
$x$-space (equivalently $\sigma^4$ in score space) so that each level contributes comparably;
unweighted score-space losses are dominated by the smallest $\sigma$.

---

## 8. Conditional form

Let $c$ be any conditioning variable. Assume only that the noise is independent of it,
$p(y\mid x,c) = p(y\mid x)$. Repeating the derivation with every density conditioned on $c$:

$$\nabla_y\,p(y\mid c) = \int \frac{x-y}{\sigma^2}\,p(y\mid x)\,p(x\mid c)\,dx
= \frac{p(y\mid c)}{\sigma^2}\big(\mathbb{E}[x\mid y,c] - y\big),$$

hence

$$\boxed{\ \hat{x}(y,c) \;=\; \mathbb{E}[x\mid y,c] \;=\; y + \sigma^2\,\nabla_y \log p(y\mid c).\ }$$

**The mathematics does not depend on the type of $c$.** Discrete class labels, dense conditioning
images (depth-from-RGB posed as denoising a depth map conditioned on the image), text embeddings, and
tuples of these are all the same theorem. Only the encoder and the injection mechanism differ, and
those are architecture questions ([§12](#12-architecture)).

**Bayes decomposition of the conditional score.**

$$\nabla_y \log p(y\mid c) = \nabla_y \log p(y) + \nabla_y \log p(c\mid y).$$

The second term is what classifier guidance estimates with a separately trained noisy classifier, and
what classifier-free guidance estimates implicitly as a difference of two denoiser outputs.

---

## 9. Classifier-free guidance

Train one network to represent both the conditional and the unconditional score by reserving an extra
null token and replacing the true label with it with probability $p_{\text{drop}}$ (0.1 is the usual
choice; 0.05 to 0.2 is a reasonable range). At sampling time, extrapolate.

**Two conventions are in circulation and differ by one.** State which you are using.

| Convention | Formula | Unconditional | Plain conditional |
|---|---|---|---|
| Interpolation (most codebases) | $\tilde{s} = s_u + w\,(s_c-s_u)$ | $w=0$ | $w=1$ |
| Ho and Salimans | $\tilde{s} = (1+w)\,s_c - w\,s_u$ | $w=-1$ | $w=0$ |

They are the same family, with $w_{\text{interp}} = 1 + w_{\text{HS}}$.

**Guidance on denoiser outputs equals guidance on scores.** Because $s = (D(y)-y)/\sigma^2$ is affine in
$D$ and the combination weights sum to one,

$$\frac{\big[D_u + w(D_c-D_u)\big]-y}{\sigma^2} \;=\; s_u + w\,(s_c-s_u).$$

So it is legitimate, and cheaper, to apply the formula directly to the two denoiser outputs. This
holds *only* because the weights sum to one; it fails for any guidance rule that does not preserve
that. Batch the conditional and unconditional passes into one forward call of doubled batch size.

**What guidance is not.** For $w\neq 1$ (interpolation convention) $\tilde{s}$ is not the score of any
normalized density. Guidance trades distributional fidelity for conditional fidelity: large $w$ gives
sharper class identity, lower diversity, and oversaturated samples. Treat the usual "$w\in[1,10]$"
tables as practitioner heuristics rather than results, and sweep $w$ per model and per dataset.

---

## 10. Extension: linear transformation and correlated noise

**Model.** $y = Ax + \varepsilon$, $\varepsilon\sim\mathcal{N}(0,\Sigma)$ with $\Sigma\succ 0$, $\varepsilon\perp x$.

Then $y\mid x\sim\mathcal{N}(Ax,\Sigma)$, so $\nabla_y p(y\mid x) = -\Sigma^{-1}(y-Ax)p(y\mid x)$, i.e.
$(Ax-y)p(y\mid x) = \Sigma\,\nabla_y p(y\mid x)$. Integrating against the prior and dividing by $p(y)$:

$$\boxed{\ A\,\mathbb{E}[x\mid y] \;=\; y + \Sigma\,\nabla_y \log p(y).\ }$$

**Read this precisely.** The identity determines $A\hat{x}$, that is, the component of the posterior
mean in the row space of $A$. When $A$ is not injective, $\hat{x}$ itself is **not** recoverable from
the score: the null-space component is supplied by the prior through the estimator, not by this
relation. Only when $A = I$ and $\Sigma = \sigma^2 I$ does the residual of a restoration network read
out as a scaled score.

**Special case: noise followed by blur.** For $y = K(x+\varepsilon)$ with $\varepsilon\sim\mathcal{N}(0,\sigma^2I)$,
take $A = K$ and $\Sigma = \sigma^2 KK^{\top}$:

$$K\hat{x}(y) = y + \sigma^2 (KK^{\top})\,\nabla_y \log p(y),
\qquad
\nabla_y\log p(y) = \frac{1}{\sigma^2}(KK^{\top})^{-1}\big(K\hat{x}(y)-y\big),$$

the second form requiring $KK^{\top}$ to be invertible. For a blur kernel this is ill-conditioned at
the kernel's spectral zeros, so solve the system iteratively (conjugate gradient) with regularization
rather than forming an inverse.

**Beyond the Gaussian family.** Tweedie-type identities are a property of exponential families. There
is no elementary first-moment identity of the same shape for $\alpha$-stable noise. On discrete
domains the derivative has no meaning and the analogue replaces it with density ratios at
bit-flipped neighbours. On Riemannian manifolds, $\hat{x}(y)\approx\exp_y\!\big(\sigma^2\operatorname{grad}_g\log p(y)\big)$
is a workable small-noise approximation and the basis of Riemannian score-based models, but it is an
approximation, and "posterior mean" itself requires choosing a notion of mean (for example Fréchet).

---

## 11. Extension: signal-dependent noise

### 11.1 Multiplicative Gaussian noise

$$y = x\cdot n,\qquad n\sim\mathcal{N}(1,\sigma^2)\ \text{per element},\qquad x\perp n
\quad\Longleftrightarrow\quad
y\mid x \sim \mathcal{N}\big(x,\ \sigma^2x^2\big).$$

The conditional variance is signal-dependent, and the log-likelihood gradient is no longer linear in
$x$:

$$\partial_y p(y\mid x) = -\frac{y-x}{\sigma^2x^2}\,p(y\mid x)
\qquad\Longrightarrow\qquad
(x-y)\,p(y\mid x) = \sigma^2x^2\,\partial_y p(y\mid x).$$

Integrating against $p(x)$ and pulling $\partial_y$ out of the integral (legal because $x^2$ does not
depend on $y$):

$$\boxed{\ \mathbb{E}[x\mid y] \;=\; y + \sigma^2\,\frac{\partial_y\big[\mathbb{E}[x^2\mid y]\,p(y)\big]}{p(y)}\ }\tag{A}$$

Relation (A) is exact for every $\sigma$. The correction now requires the **second** posterior moment.
Since a single-output denoiser exposes only the first moment:

> Under multiplicative noise there is no residual-equals-score identity. The residual of an optimal
> denoiser is not a rescaled score, and no architectural choice recovers one.

### 11.2 Small-$\sigma$ expansion

For small $\sigma$ the posterior concentrates and $\mathbb{E}[x^2\mid y]\approx y^2$. With
$\partial_y[y^2p] = 2yp + y^2\partial_y p$:

$$\boxed{\ D(y)-y \;\approx\; 2\sigma^2 y \;+\; \sigma^2 y^2\,\nabla_y\log p(y).\ }\tag{B}$$

The two terms are structurally different:

- $2\sigma^2 y$ is prior-independent, proportional to the signal, and directed **away from zero**. A
  sanity check: under an improper flat prior the score term vanishes, and a second-order expansion of
  the posterior gives $\mathbb{E}[x\mid y]\approx y(1+2\sigma^2)$, matching (B). The term arises from
  the $1/|x|$ normalizer and the $x$-dependent variance in the likelihood, which together tilt the
  posterior outward. It is an inflation, not a shrinkage.
- $\sigma^2y^2\,\nabla_y\log p(y)$ is the familiar score term, reweighted by the local variance
  $\sigma^2y^2$ instead of the constant $\sigma^2$.

### 11.3 Composite (Poisson-Gaussian) noise

$$y = x\cdot n + a,\qquad n\sim\mathcal{N}(1,\sigma_m^2),\quad a\sim\mathcal{N}(0,\sigma_a^2)
\quad\Longleftrightarrow\quad
y\mid x\sim\mathcal{N}\big(x,\ \sigma_m^2x^2+\sigma_a^2\big).$$

This is the Gaussian form of the standard sensor model: a constant read-noise floor plus a
signal-dependent, shot-noise-like term. The variance factor splits linearly, so the derivation
superposes exactly:

$$\boxed{\ \mathbb{E}[x\mid y] = y + \sigma_a^2\,\nabla_y\log p(y) + \sigma_m^2\,\frac{\partial_y\big[\mathbb{E}[x^2\mid y]\,p(y)\big]}{p(y)}\ }\tag{A$_c$}$$

$$\boxed{\ D(y)-y \;\approx\; \big(\sigma_a^2+\sigma_m^2y^2\big)\,\nabla_y\log p(y) \;+\; 2\sigma_m^2 y\ }\tag{B$_c$}$$

with local variance $v(y) = \sigma_a^2 + \sigma_m^2y^2$ as the score weight. Setting $\sigma_a=0$
recovers (A) and (B); setting $\sigma_m=0$ recovers the additive theorem.

**Why the additive floor matters.** Under pure multiplicative noise the conditional variance
$\sigma_m^2x^2$ vanishes as $x\to0$: dark pixels are barely corrupted, the score weight collapses, and
the empirical-Bayes quantities become numerically unstable there. The floor $\sigma_a^2$ bounds $v(y)$
away from zero everywhere. This is both physically correct for real sensors and numerically necessary.

### 11.4 The log transform

With $u = \log x$ and $v = \log y$, one has $v = u + \log n \approx u + (n-1)$ for small $\sigma$, so the
additive theorem applies in log space. This requires $x>0$, so it is available on a strictly positive
domain such as $[0,1]$ but not on a signed one. Two caveats: it is badly conditioned near zero, and
MSE in log space is not MSE in linear space, so the trained network is no longer the linear-domain
MMSE estimator. Staying in the linear domain and using (A)/(B) is a legitimate choice; make it
deliberately.

### 11.5 Scale equivariance under signal-dependent noise

Under a scaled prior $p_\alpha(x) = \alpha^{-1}p(x/\alpha)$, whose marginal satisfies
$\nabla\log p_\alpha(\alpha y) = \alpha^{-1}\nabla\log p(y)$, relation (B) gives

$$D_\alpha(\alpha y) = \alpha y + 2\sigma^2(\alpha y) + \sigma^2(\alpha y)^2\cdot\tfrac1\alpha\nabla\log p(y) = \alpha\,D(y).$$

So the multiplicative MMSE denoiser is exactly scale-equivariant **at fixed $\sigma$**, without needing
to co-scale the noise level. This is a stronger compatibility with a homogeneous (bias-free) network
than the additive case enjoys, where $\sigma$ must scale alongside the signal. The composite model is
equivariant under the joint scaling of $x$ and $\sigma_a$, matching the additive case. The genuine
caveat under signal-dependent noise is the one in §11.1: the residual stops being a score, which
breaks score readout, RED, and denoiser-driven sampling regardless of architecture.

---

## 12. Architecture

### 12.1 What the theorem does and does not constrain

The theorem is a statement about $\mathbb{E}[x\mid y]$. It constrains exactly one design choice: **the
loss must be MSE**, because MSE is the loss whose minimizer is the conditional mean. Everything else
is engineering, with its own justifications.

| Choice | Why |
|---|---|
| MSE loss | The only requirement. L1 gives the conditional median; perceptual and adversarial losses give neither. |
| Linear output head | The residual $\hat{x}-y = \sigma^2 s(y)$ is signed and unbounded. A bounded head (sigmoid, tanh) cannot represent it, a one-sided head (ReLU) cannot represent negative corrections, and any head with $h(0)\neq0$ injects a constant offset. |
| Residual parameterization $D(y) = y + r_\theta(y)$ | The network regresses $\sigma^2 s(y)$ directly, a small signed quantity, instead of the score being recovered as a difference of two large nearly-equal numbers. Better conditioning; no theoretical content. |
| Bias-free layers | Enforces $f(\alpha y) = \alpha f(y)$, an inductive bias that empirically buys generalization to unseen noise levels. Not required by the theorem. |

### 12.2 Bias-free networks and homogeneity

**Definition.** Every affine map has zero additive offset: `use_bias=False` in convolution and dense
layers, `center=False` in normalization layers (no learned $\beta$), and no activation with a nonzero
value at zero on the output path.

**Homogeneity.** With positively homogeneous activations (ReLU, leaky ReLU, PReLU) and no offsets, the
network is positively homogeneous of degree one:

$$f(\alpha y) = \alpha f(y)\ \ \text{for all }\alpha>0, \qquad\text{hence}\qquad f(0)=0.$$

Equivalently $f(y) = J(y)\,y$ with $J$ locally constant: the network is piecewise linear through the
origin, which is what makes bias-free denoisers analyzable (the effective filters $J(y)$ can be read
off for any given input).

**Two caveats.**

1. **Batch normalization breaks homogeneity in training mode.** With batch statistics,
   $\mathrm{BN}(\alpha x) = \gamma\,\frac{\alpha x-\alpha\mu}{\alpha\varsigma} = \mathrm{BN}(x)$, which is
   degree **zero**, not one. Homogeneity holds only in inference mode, where the running statistics
   are frozen constants and the layer reduces to a fixed diagonal scaling. Run every homogeneity and
   DC diagnostic with `training=False`. Mean subtraction itself is linear and harmless; it is the
   learned offset $\beta$ that must go.
2. **Homogeneity is exactly correct only for a scale-invariant prior.** The exact statement for
   additive noise is a joint equivariance: scaling the prior by $\alpha$ *and* the noise level by
   $\alpha$ scales the MMSE estimator by $\alpha$. A $\sigma$-blind homogeneous network conflates
   "scale the input" with "scale the prior and the noise together", which is right only if $p$ is
   scale-invariant. Natural image statistics are approximately scale-invariant, which is the honest
   justification; the measured payoff is robust generalization to noise levels far outside the
   training range, where networks with biases degrade sharply.

**Use bias-free when:** denoising blind, over a wide or unknown $\sigma$ range, when you want
interpretable effective filters, or when the denoiser will be reused as an implicit prior across
scales.

**Prefer $\sigma$-conditioning when $\sigma$ is known.** The construction

$$D(y,\sigma) = \sigma\cdot g_\theta\!\big(y/\sigma\big)$$

satisfies $D(\alpha y,\alpha\sigma) = \alpha D(y,\sigma)$ exactly for *any* inner network $g_\theta$,
including one with biases. It is strictly more expressive than a blind bias-free network and exactly
equivariant, so it is the better default whenever the noise level is available.

### 12.3 Injecting conditioning without destroying homogeneity

If homogeneity in $y$ is a property you want to keep, the injection mechanism matters:

- **Additive injection** (project the class embedding and add it to the features) injects a constant
  that does not scale with $y$, so $f(\alpha y,c)\neq\alpha f(y,c)$. Homogeneity is lost.
- **Channel concatenation** of a tiled conditioning tensor breaks homogeneity for the same reason.
- **Scale-only FiLM**, $h\mapsto h\odot\gamma(c)$ with $\gamma$ a function of $c$ alone, **preserves**
  homogeneity, because the modulation is independent of $y$.

Choose deliberately. Either use multiplicative modulation and keep exact homogeneity, or use additive
injection and accept a non-homogeneous network, in which case do not also claim homogeneity-based
noise-level generalization and do not expect the DC probe to pass.

The conditioning encoder itself (for example a ResNet extracting multi-scale features from an RGB
image) does not need to be homogeneous, since it does not see $y$. Deleting biases from a pretrained
backbone invalidates its weights; either train the encoder bias-free from scratch or keep it biased
and inject multiplicatively.

### 12.4 A note on injection points and capacity

Injecting conditioning at every resolution level (encoder, bottleneck, decoder) is the usual default:
early injection shapes low-level feature extraction, bottleneck injection carries global semantics,
and late injection shapes reconstruction. Ablate downward from there if parameter count matters.
Embedding dimensions in the 64 to 256 range are typical; scale with dataset size and class count
rather than by formula.

---

## 13. Data domain and normalization

For networks **with** biases, input normalization is an ordinary conditioning choice. For **bias-free**
networks it is a structural one, because such a network cannot represent a DC offset: it has no
mechanism to add or subtract a constant.

**Consequences.**

1. **The domain is not a free relabeling.** A bias-free model trained on $[0,1]$ and fed data on
   $[-0.5,+0.5]$ produces silent garbage, not an error. Record the data range in the checkpoint
   metadata and refuse to load a checkpoint whose range does not match.
2. **$[0,1]$ and $[-0.5,+0.5]$ differ by a shift only.** Both have peak-to-peak width $1.0$, so
   $\sigma$, `max_val` for PSNR and SSIM, and conversions such as $\sigma_{255} = 255\sigma$ are
   unchanged between them. Rescaling those constants "because the domain moved" silently corrupts
   every reported dB number and nothing fails loudly. Note that $[-1,+1]$ has width $2$, so moving
   there **does** require $\sigma\to2\sigma$ and `max_val = 2.0`.
3. **Flat patches force sum-to-one filters.** For a flat patch of value $c>0$, homogeneity gives
   $f(c\mathbf{1}) = c\,f(\mathbf{1})$, so preserving the patch requires $f(\mathbf{1}) = \mathbf{1}$: the local
   filter weights must sum to one. That is the correct DC-preserving behaviour for a denoiser, and it
   is directly testable ([§16](#16-diagnostics)).

**Why $[0,1]$ is the better default.** The squared-error contribution of a flat patch at level $c$ is
$c^2\|f(\mathbf{1})-\mathbf{1}\|^2$, so the gradient signal for the DC property scales as $c^2$. On $[0,1]$,
mid-grey sits at $c=0.5$ (weight $0.25$) and white at $c=1$ (weight $1$). On $[-0.5,+0.5]$, mid-grey
sits at $c=0$ (weight $0$, where the output is pinned to zero by homogeneity regardless of weights)
and the extremes only reach weight $0.25$. Natural images carry a large mass of near-mid-grey flat
content: sky, walls, paper, out-of-focus background. Zero-centering places exactly that mass where the
DC constraint carries the least gradient, and it splits the constraint across two independent rays,
since positive homogeneity relates $f(\alpha y)$ to $f(y)$ only for $\alpha>0$, so $f(\mathbf{1})=\mathbf{1}$
and $f(-\mathbf{1})=-\mathbf{1}$ must be learned separately. The effect is a weighting argument, not a
degeneracy argument: only the single value $c=0$ is genuinely uninformative.

**What remains open.** Strictly positive inputs can aggravate dead-ReLU behaviour; the argument above
constrains what the network *can* learn, not what the optimizer will reach. Treat claims about
$[0,1]$ denoising quality as unverified until measured against a matched baseline. Useful
pre-committed stop conditions: divergence, validation loss never falling below its epoch-0 value, or
a DC probe that drifts away from $f(c\mathbf{1}) = c\mathbf{1}$ as training proceeds.

**Avoid z-scoring** per image or per dataset for denoising: it destroys the pixel domain, makes
`max_val` ill-defined for PSNR and SSIM, and couples every image to dataset statistics.

**Do not clip the noisy input.** Clipping $y$ back to the domain after adding noise turns the Gaussian
likelihood into a truncated one, so the MMSE target is no longer the one the theorem describes. The
bias grows with $\sigma$ and with proximity to the domain edges. Clip only for display. If pipeline
constraints force clipping, document it as an approximation whose error concentrates at the extremes.
The same caveat applies with more force to signal-dependent noise, where the total variance is larger
and clipping fires more often.

---

## 14. Sampling with a denoiser

### 14.1 Annealed Langevin dynamics

Unadjusted Langevin for a fixed density:

$$y \leftarrow y + \frac{\eta}{2}\nabla_y\log p(y) + \sqrt{\eta}\,z,\qquad z\sim\mathcal{N}(0,I).$$

Some references write $y \leftarrow y + \eta s + \sqrt{2\eta}\,z$, which is the same recursion with
$\eta' = 2\eta$. Be consistent or the effective temperature is off by a factor of two.

With a denoiser supplying the score at level $\sigma$, sweep $\sigma$ from large to small and set
$\eta_\sigma\propto\sigma^2$, which keeps the per-step signal-to-noise ratio roughly constant across
levels. A single fixed step size across a wide $\sigma$ range does not work: too large at small
$\sigma$ (divergence), too small at large $\sigma$ (no mixing).

### 14.2 Denoiser-driven coarse-to-fine sampling

This sampler is native to the Miyasawa formulation: it reads the effective noise level off the
residual rather than following a prescribed schedule, so a single blind denoiser suffices.

```
inputs: denoiser f, sigma_0 (large), sigma_L (small), h_0 in (0, 1], beta in [0, 1]
y <- N(0.5 * 1, sigma_0^2 I)          # mid-grey initialization on the [0, 1] domain
t <- 1
while sigma_t > sigma_L:
    h_t     <- h_0 * t / (1 + h_0 * (t - 1))
    d_t     <- f(y) - y                       # = sigma_t^2 * score, by the theorem
    sigma_t <- ||d_t|| / sqrt(N)              # effective noise level from the residual
    gamma_t <- sqrt((1 - beta*h_t)^2 - (1 - h_t)^2) * sigma_t
    y       <- y + h_t * d_t + gamma_t * N(0, I)
    t       <- t + 1
return y
```

`beta = 0` gives deterministic ascent onto the data manifold, which is useful for inverse problems;
`beta = 1` gives fully stochastic sampling. Step sizes $h_t$ increase toward one as the effective
noise falls.

### 14.3 Reverse-time SDE

For a forward diffusion $dx = f(x,t)\,dt + g(t)\,dw$, the reverse-time process is

$$dx = \big[f(x,t) - g^2(t)\,\nabla_x\log p_t(x)\big]dt + g(t)\,d\bar{w},$$

with the score supplied by the denoiser through §6. This is the continuous-time statement of what
diffusion samplers discretize; the probability-flow ODE obtained by halving the score coefficient and
dropping the noise term gives deterministic sampling with the same marginals.

---

## 15. Inverse problems

**Plug-and-play.** Split $\min_x \tfrac12\|Ax-b\|^2 + \lambda R(x)$ with ADMM or proximal gradient and
replace the proximal operator of $R$ with a denoiser call. Convergence guarantees require conditions
on the denoiser (nonexpansiveness or similar) that generic networks do not satisfy; spectrally
constrained denoisers do.

**Regularization by denoising (RED).** Define $R(x) = \tfrac12 x^{\top}\big(x-D(x)\big)$ and claim
$\nabla R(x) = x - D(x)$. That step requires local homogeneity **and Jacobian symmetry** of $D$. For
the exact MMSE denoiser both hold, symmetry automatically ([§5](#5-second-order-tweedie-the-jacobian-is-the-posterior-covariance)).
For trained networks symmetry generally fails, so RED updates are better justified as a fixed-point
scheme, or reinterpreted as score matching by denoising, than as descent on an explicit energy. A
bias-free architecture gives exact local homogeneity but says nothing about symmetry.

**Denoiser-driven posterior sampling.** The cleanest formulation avoids an explicit regularizer:
alternate a prior step driven by the denoiser residual with a measurement-consistency step,

$$x \leftarrow x + h\big(D(x)-x\big) \;-\; \mu\,A^{\top}(Ax-b) \;+\; \gamma z .$$

Signs matter: $D(x)-x$ equals $+\sigma^2\nabla\log p$ and is therefore ascent on the log-prior, while
$A^{\top}(Ax-b)$ is the gradient of the data term and must be subtracted. For a projection variant,
replace the data step with a projection onto the affine set $\{x: Ax = b\}$ and let the prior step act
only in the null space of $A$.

**Caveat inherited from §10.** If the denoiser was trained on a degraded observation rather than on
pure additive noise, its residual is not a score, and the prior step above is not doing what the
notation suggests.

---

## 16. Diagnostics

### 16.1 SURE: reference-free risk for additive Gaussian noise

For $y = x + \mathcal{N}(0,\sigma^2I)$ and weakly differentiable $D$:

$$\mathrm{SURE}(D) = \|D(y)-y\|^2 + 2\sigma^2\operatorname{div}(D) - N\sigma^2,
\qquad
\mathbb{E}[\mathrm{SURE}] = \mathbb{E}\|D(y)-x\|^2 .$$

This estimates the true risk from noisy data alone, with no clean references, which makes it the right
tool for auditing a checkpoint on real data. Estimate the divergence with a Hutchinson probe and a
finite-difference Jacobian-vector product:

$$\operatorname{div}(D)\approx\mathbb{E}_v\Big[v^{\top}\frac{D(y+\epsilon v)-D(y)}{\epsilon}\Big],
\qquad v_i\in\{\pm1\}\ \text{i.i.d.}$$

Validate the estimator before trusting it: for a linear toy denoiser $D(y) = ay$ the divergence is
analytically $aN$, and SURE should reproduce the realized MSE to within about a percent. If that check
fails, the probe count, $\epsilon$, or float precision is wrong.

### 16.2 Generalized SURE for signal-dependent noise

For $y\mid x\sim\mathcal{N}(x,\Sigma)$ with **known** $\Sigma$:

$$\mathrm{gSURE} = \|D(y)-y\|^2 + 2\operatorname{tr}\!\big(\Sigma J_D\big) - \operatorname{tr}\Sigma .$$

Under the multiplicative model $\Sigma = \sigma^2\operatorname{diag}(x^2)$ is unknown; substituting
$\sigma^2\operatorname{diag}(y^2)$ makes this a leading-order approximation rather than an unbiased
risk estimate. Report it as a consistency scalar for tracking a checkpoint over time, not as a risk
number. The variance-weighted divergence is estimated with a probe pre-scaled so that
$\mathbb{E}[v_iv_j] = \sigma^2y_i^2\delta_{ij}$, that is $v_i = \sigma|y_i|r_i$ with $r$ Rademacher.

### 16.3 Homogeneity probe

$$\mathrm{err}(\alpha) = \frac{\|f(\alpha y)-\alpha f(y)\|}{\alpha\|f(y)\|},\qquad \text{expected }\approx 0\ \text{for all }\alpha>0 .$$

Run in inference mode. A nonzero, $\alpha$-dependent value means a residual bias term, an additive
conditioning injection, or a normalization layer running on batch statistics.

### 16.4 DC / sum-to-one probe

$$\mathrm{rel\_err}(c) = \frac{\|f(c\mathbf{1})-c\mathbf{1}\|}{\|c\mathbf{1}\|}.$$

By homogeneity this is independent of $c$ and equals $\|f(\mathbf{1})-\mathbf{1}\|/\|\mathbf{1}\|$. A constant
column across $c$ is therefore the expected signature and confirms the probe is measuring the
sum-to-one property and nothing else. An untrained network gives a large constant of order one; the
value should fall during training. A value that varies with $c$ means the network is not homogeneous.

### 16.5 Score-consistency spot check

On synthetic data with a known prior (for example a Gaussian mixture in one or two dimensions), the
true posterior mean and true score are computable, so the identity can be checked end to end: train
or evaluate the denoiser, read out $(D(y)-y)/\sigma^2$, and compare against $\nabla\log p_\sigma$
computed analytically. This catches sign errors, $\sigma$ versus $\sigma^2$ errors, and domain
mismatches long before they show up as poor samples.

---

## 17. Limitations

1. **Approximation, not identity.** Trained denoisers are not MMSE-optimal. Score readout inherits
   that error amplified by $1/\sigma^2$, so estimates are least reliable at small $\sigma$, which is
   exactly where samplers spend their final steps.
2. **The score is of the smoothed density.** You never obtain $\nabla\log p_x$, only
   $\nabla\log p_\sigma$. On manifold-supported data the clean score does not exist, which is why
   multi-scale annealing is necessary rather than merely convenient.
3. **Model mismatch.** Non-Gaussian, correlated, or signal-dependent noise invalidates the plain
   identity; use §10 or §11, or accept a documented approximation.
4. **Preprocessing artifacts.** Clipping, 8-bit quantization, demosaicing, and JPEG all perturb the
   likelihood. Effects concentrate at domain edges and grow with $\sigma$.
5. **Jacobian symmetry.** Assumed by RED-style methods for trained networks; generally false.
6. **High dimension.** Score estimation error, sampler step-size sensitivity, and the sparsity of data
   relative to ambient dimension all worsen with $N$. Sampling that looks stable at 32 by 32 can
   diverge at 512 by 512 with the same hyperparameters.
7. **Samples are not a diagnostic.** A denoiser can produce plausible samples while its residual field
   is a poor score estimate, and the converse. Use SURE and the probes in §16.

---

## 18. Reference implementation

Keras 3 style. In PyTorch the equivalents are `bias=False` on `nn.Conv2d` and `nn.Linear`, and a
normalization layer with no learned shift.

### 18.1 Bias-free denoiser

```python
import keras
from keras import layers
import tensorflow as tf


def bias_free_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    # center=False removes the learned beta offset. With batch statistics this layer
    # is degree-0 homogeneous; exact degree-1 homogeneity holds in inference mode.
    x = layers.BatchNormalization(center=False, scale=True)(x)
    return layers.Activation("relu")(x)


def bias_free_denoiser(input_shape=(None, None, 1), filters=64, num_blocks=8):
    """Residual, bias-free, linear output head. Predicts x_hat = y + r(y),
    where r(y) approximates sigma^2 * score(y)."""
    inp = keras.Input(shape=input_shape)
    x = inp
    for _ in range(num_blocks):
        x = bias_free_block(x, filters)
    residual = layers.Conv2D(input_shape[-1], 3, padding="same", use_bias=False)(x)
    out = layers.Add()([inp, residual])
    return keras.Model(inp, out, name="bias_free_denoiser")
```

### 18.2 Sigma-conditioned denoiser (exactly equivariant, more expressive)

```python
class SigmaConditionedDenoiser(keras.Model):
    """D(y, sigma) = sigma * g(y / sigma).

    Satisfies D(a*y, a*sigma) = a * D(y, sigma) exactly for any inner network g,
    including one with biases. Preferred whenever sigma is known.
    """

    def __init__(self, inner, **kwargs):
        super().__init__(**kwargs)
        self.inner = inner

    def call(self, inputs, training=None):
        y, sigma = inputs                       # sigma shape (B, 1, 1, 1)
        return sigma * self.inner(y / sigma, training=training)
```

### 18.3 Homogeneity-preserving conditioning

```python
def film_scale_only(features, cond_embedding, name=None):
    """Multiplicative (scale-only) FiLM. Preserves degree-1 homogeneity in y,
    because gamma depends on the condition alone. Additive broadcast injection
    (features + Dense(emb)) does not."""
    c = features.shape[-1]
    gamma = layers.Dense(c, use_bias=False, name=name)(cond_embedding)
    gamma = layers.Reshape((1, 1, c))(gamma)
    return layers.Multiply()([features, 1.0 + gamma])
```

### 18.4 Data pipeline

```python
def to_unit_domain(images):
    """[0, 255] -> [0, 1]. Strictly positive, deliberately not zero-centered."""
    images = tf.cast(images, tf.float32)
    return tf.cond(tf.reduce_max(images) > 1.0, lambda: images / 255.0, lambda: images)


def make_noisy(clean, sigma_min=0.0, sigma_max=0.4):
    """Per-example sigma so each batch spans the noise range.
    The noisy input is deliberately not clipped: clipping truncates the Gaussian
    likelihood and biases the MMSE target."""
    b = tf.shape(clean)[0]
    sigma = tf.random.uniform([b, 1, 1, 1], sigma_min, sigma_max)
    noisy = clean + sigma * tf.random.normal(tf.shape(clean))
    return noisy, clean, sigma
```

Compile with `loss="mse"`.

### 18.5 Score readout and sampling

```python
def score_from_denoiser(denoiser, y, sigma):
    """grad_y log p(y) = (E[x|y] - y) / sigma^2."""
    return (denoiser(y, training=False) - y) / (sigma ** 2)


def annealed_langevin(denoiser, shape, sigmas, steps_per_level=100, eta0=2e-5, seed=0):
    """sigmas: decreasing list. Step size scales as sigma^2."""
    g = tf.random.Generator.from_seed(seed)
    y = 0.5 + sigmas[0] * g.normal(shape)       # mid-grey init on the [0, 1] domain
    for sigma in sigmas:
        eta = eta0 * (sigma / sigmas[-1]) ** 2
        for _ in range(steps_per_level):
            s = score_from_denoiser(denoiser, y, sigma)
            y = y + 0.5 * eta * s + tf.sqrt(eta) * g.normal(tf.shape(y))
    return tf.clip_by_value(y, 0.0, 1.0)        # clip for display only


def coarse_to_fine_sample(denoiser, shape, sigma_0=1.0, sigma_L=0.01,
                          h0=0.05, beta=0.5, max_iter=500):
    """Schedule-free sampler: reads the noise level off the residual."""
    n = float(tf.reduce_prod(shape[1:]).numpy())
    y = 0.5 + sigma_0 * tf.random.normal(shape)
    sigma_t, t = sigma_0, 1
    while sigma_t > sigma_L and t <= max_iter:
        h = h0 * t / (1.0 + h0 * (t - 1))
        d = denoiser(y, training=False) - y                  # sigma_t^2 * score
        sigma_t = float(tf.norm(d) / tf.sqrt(n))
        gamma = tf.sqrt(tf.maximum((1 - beta * h) ** 2 - (1 - h) ** 2, 0.0)) * sigma_t
        y = y + h * d + gamma * tf.random.normal(tf.shape(y))
        t += 1
    return tf.clip_by_value(y, 0.0, 1.0)
```

### 18.6 Classifier-free guidance (interpolation convention)

```python
def cfg_denoise(denoiser, y, labels, null_token, w=3.0):
    """w = 0 unconditional, w = 1 plain conditional, w > 1 amplified.
    Applied to denoiser outputs, which is equivalent to applying it to scores
    because the weights sum to one."""
    y2 = tf.concat([y, y], axis=0)
    null = tf.fill(tf.shape(labels), tf.cast(null_token, labels.dtype))
    lab2 = tf.concat([labels, null], axis=0)
    out = denoiser([y2, lab2], training=False)
    d_cond, d_uncond = tf.split(out, 2, axis=0)
    return d_uncond + w * (d_cond - d_uncond)
```

### 18.7 Diagnostics

```python
def homogeneity_error(model, y, alpha=3.7):
    a = model(alpha * y, training=False)
    b = alpha * model(y, training=False)
    return float(tf.norm(a - b) / tf.norm(b))


def dc_probe(model, levels=(0.1, 0.25, 0.5, 0.75, 0.9), shape=(1, 64, 64, 1)):
    """Expected signature: identical rel_err for every c, equal to ||f(1)-1|| / ||1||."""
    out = {}
    for c in levels:
        flat = tf.fill(shape, tf.constant(c, tf.float32))
        pred = model(flat, training=False)
        out[c] = float(tf.norm(pred - flat) / tf.norm(flat))
    return out


def hutchinson_divergence(model, y, sigma=None, eps=1e-3, n_probes=8):
    """Unweighted (sigma=None) or variance-weighted divergence.

    Weighted mode draws v_i = sigma * |y_i| * rademacher_i, so that
    E[v v^T] = diag(sigma^2 y_i^2) and E[v^T J v] = sum_i sigma^2 y_i^2 dD_i/dy_i.
    """
    base = model(y, training=False)
    total = 0.0
    for _ in range(n_probes):
        r = tf.cast(tf.random.uniform(tf.shape(y), 0, 2, dtype=tf.int32) * 2 - 1, y.dtype)
        v = r if sigma is None else sigma * tf.abs(y) * r
        jvp = (model(y + eps * v, training=False) - base) / eps
        total += float(tf.reduce_sum(v * jvp))
    return total / n_probes


def sure_risk(model, y, sigma, **kw):
    """Reference-free estimate of E||D(y) - x||^2 for additive Gaussian noise."""
    n = float(tf.size(y).numpy())
    resid_sq = float(tf.reduce_sum((model(y, training=False) - y) ** 2))
    div = hutchinson_divergence(model, y, **kw)
    return resid_sq + 2.0 * sigma ** 2 * div - n * sigma ** 2
```

---

## 19. Quick reference

### Identities by noise model

| Noise model | Likelihood | Exact identity | Residual is the score? | Score weight |
|---|---|---|---|---|
| Additive, $y=x+\varepsilon$ | $\mathcal{N}(x,\sigma^2I)$ | $\hat{x} = y+\sigma^2\nabla\log p(y)$ | Yes | $\sigma^2$ |
| Conditional | $\mathcal{N}(x,\sigma^2I)$, $c\perp\varepsilon$ | $\hat{x}(y,c)=y+\sigma^2\nabla\log p(y\mid c)$ | Yes | $\sigma^2$ |
| Linear, $y=Ax+\varepsilon$ | $\mathcal{N}(Ax,\Sigma)$ | $A\hat{x}=y+\Sigma\nabla\log p(y)$ | Only $A\hat{x}$; not invertible in general | $\Sigma$ |
| Blur after noise, $y=K(x+\varepsilon)$ | $\mathcal{N}(Kx,\sigma^2KK^{\top})$ | $K\hat{x}=y+\sigma^2KK^{\top}\nabla\log p(y)$ | Needs $(KK^{\top})^{-1}$ | $\sigma^2KK^{\top}$ |
| Multiplicative, $y=xn$ | $\mathcal{N}(x,\sigma^2x^2)$ | (A), requires $\mathbb{E}[x^2\mid y]$ | No | $\sigma^2y^2$ (approx.) |
| Composite, $y=xn+a$ | $\mathcal{N}(x,\sigma_m^2x^2+\sigma_a^2)$ | (A$_c$), additive plus second-moment term | Only if $\sigma_m=0$ | $\sigma_a^2+\sigma_m^2y^2$ (approx.) |
| VP / DDPM | $\mathcal{N}(\sqrt{\bar\alpha}x_0,(1-\bar\alpha)I)$ | $\mathbb{E}[x_0\mid x_t]=\frac{x_t+(1-\bar\alpha)\nabla\log p_t}{\sqrt{\bar\alpha}}$ | Yes, after rescaling | $1-\bar\alpha$ |

### Properties by noise model

| Property | Additive | Multiplicative | Composite |
|---|---|---|---|
| MSE training yields $\mathbb{E}[x\mid y]$ | Yes | Yes | Yes |
| Residual-equals-score identity | Yes | No | Only if $\sigma_m=0$ |
| Optimal estimator scale-equivariant | Yes, if $\sigma$ co-scales | Yes, at fixed $\sigma_m$ | Yes, if $\sigma_a$ co-scales |
| Homogeneous (bias-free) network compatible | Yes | Yes | Yes |
| Reference-free audit | SURE (unbiased) | gSURE (approximate) | gSURE (approximate) |
| Well conditioned near $x=0$ | Yes | No | Yes |

---

## 20. Conventions checklist

1. **Score versus noise prediction.** $s = -\hat{\varepsilon}/\sigma^2$ if the network predicts the
   noise vector; $s = -\varepsilon_\theta/\sigma$ if it predicts unit-variance noise.
2. **Variance versus standard deviation.** The score weight is $\sigma^2$, not $\sigma$.
3. **Langevin step size.** $y \mathrel{+}= \tfrac{\eta}{2}s + \sqrt{\eta}z$ and
   $y \mathrel{+}= \eta s + \sqrt{2\eta}z$ are the same up to $\eta\to2\eta$.
4. **Guidance scale.** Interpolation convention ($w=1$ is plain conditional) versus Ho and Salimans
   ($w=0$ is plain conditional).
5. **Domain shift versus rescale.** $[0,1]$ and $[-0.5,+0.5]$ differ by a shift only, so do not
   rescale $\sigma$ or `max_val`. $[-1,+1]$ has width $2$ and does require rescaling.
6. **`center=False`** in Keras `BatchNormalization` removes the learned offset $\beta$. It does not
   remove mean subtraction, which is linear and harmless for homogeneity.
7. **Inference mode for diagnostics.** Homogeneity and DC probes are only meaningful with frozen
   normalization statistics.
8. **Per-example noise level.** Sample $\sigma$ per example, not per batch.

---

## 21. History and references

### Timeline

| Year | Contribution |
|---|---|
| 1956 | Robbins: empirical Bayes framework; the Gaussian identity is attributed to Tweedie in this line |
| 1961 | Miyasawa: the Gaussian empirical-Bayes estimator identity |
| 1981 | Stein: SURE, unbiased risk estimation for the same model |
| 1982 | Anderson: reverse-time SDEs |
| 2005 | Hyvärinen: score matching for unnormalized models |
| 2011 | Vincent: denoising score matching; Raphan and Simoncelli: empirical-Bayes estimation for general noise; Efron: the modern statistical exposition of Tweedie's formula |
| 2019 | Song and Ermon: annealed Langevin generation from learned scores |
| 2020 | Ho, Jain and Abbeel: DDPM; Mohan et al.: bias-free CNNs |
| 2021 | Song et al.: unified SDE framework; Kadkhodaie and Simoncelli: the prior implicit in a denoiser |
| 2022 | Ho and Salimans: classifier-free guidance |

### Foundations

- Miyasawa, K. (1961). *An empirical Bayes estimator of the mean of a normal population.* Bulletin of the International Statistical Institute, 38(4), 181-188. Difficult to obtain; the identity is most accessibly stated in Efron (2011) and Raphan and Simoncelli (2011).
- Robbins, H. (1956). *An empirical Bayes approach to statistics.* Proceedings of the Third Berkeley Symposium.
- Stein, C. (1981). *Estimation of the mean of a multivariate normal distribution.* Annals of Statistics, 9(6), 1135-1151.
- Efron, B. (2011). *Tweedie's formula and selection bias.* JASA, 106(496), 1602-1614.
- Hyvärinen, A. (2005). *Estimation of non-normalized statistical models by score matching.* JMLR, 6, 695-709.
- Vincent, P. (2011). *A connection between score matching and denoising autoencoders.* Neural Computation, 23(7), 1661-1674.
- Raphan, M., and Simoncelli, E. P. (2011). *Least squares estimation without priors or supervision.* Neural Computation, 23(2), 374-420.
- Anderson, B. D. O. (1982). *Reverse-time diffusion equation models.* Stochastic Processes and their Applications, 12(3), 313-326.
- Eldar, Y. C. (2009). *Generalized SURE for exponential families.* IEEE Transactions on Signal Processing, 57(2), 471-481.
- Foi, A., Trimeche, M., Katkovnik, V., and Egiazarian, K. (2008). *Practical Poissonian-Gaussian noise modeling and fitting for single-image raw data.* IEEE TIP, 17(10), 1737-1754.

### Generative modeling and denoiser priors

- Song, Y., and Ermon, S. (2019). *Generative modeling by estimating gradients of the data distribution.* NeurIPS.
- Ho, J., Jain, A., and Abbeel, P. (2020). *Denoising diffusion probabilistic models.* NeurIPS.
- Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. (2021). *Score-based generative modeling through stochastic differential equations.* ICLR.
- Mohan, S., Kadkhodaie, Z., Simoncelli, E. P., and Fernandez-Granda, C. (2020). *Robust and interpretable blind image denoising via bias-free convolutional neural networks.* ICLR.
- Kadkhodaie, Z., and Simoncelli, E. P. (2021). *Stochastic solutions for linear inverse problems using the prior implicit in a denoiser.* NeurIPS. arXiv:2007.13640. Reference code: `LabForComputationalVision/universal_inverse_problem`.
- Ho, J., and Salimans, T. (2022). *Classifier-free diffusion guidance.* arXiv:2207.12598.
- Dhariwal, P., and Nichol, A. (2021). *Diffusion models beat GANs on image synthesis.* NeurIPS. Classifier guidance.

### Inverse problems and conditioning

- Venkatakrishnan, S. V., Bouman, C. A., and Wohlberg, B. (2013). *Plug-and-play priors for model based reconstruction.* GlobalSIP.
- Romano, Y., Elad, M., and Milanfar, P. (2017). *The little engine that could: regularization by denoising (RED).* SIAM Journal on Imaging Sciences, 10(4), 1804-1844.
- Reehorst, E. T., and Schniter, P. (2019). *Regularization by denoising: clarifications and new interpretations.* IEEE Transactions on Computational Imaging, 5(1), 52-67.
- Ronneberger, O., Fischer, P., and Brox, T. (2015). *U-Net: convolutional networks for biomedical image segmentation.* MICCAI.
- Zhang, L., Rao, A., and Agrawala, M. (2023). *Adding conditional control to text-to-image diffusion models (ControlNet).* ICCV.
- Ke, B., Obukhov, A., Huang, S., Metzger, N., Daudt, R. C., and Schindler, K. (2024). *Repurposing diffusion-based image generators for monocular depth estimation (Marigold).* CVPR.