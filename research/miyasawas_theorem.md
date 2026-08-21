# Miyasawa's Theorem (Tweedie's Formula): A Corrected and Unified Reference

Scope: the empirical-Bayes identity linking the MMSE denoiser to the score of the noisy
density; its exact extensions (conditional, linear-transformed, signal-dependent, composite);
its role in score matching, diffusion models and inverse problems; and the architectural and
normalization questions that surround bias-free denoisers.

---

## Table of Contents

1. [Notation and conventions](#1-notation-and-conventions)
2. [Statement and hypotheses](#2-statement-and-hypotheses)
3. [Derivation](#3-derivation)
4. [Second-order Tweedie: the Jacobian is the posterior covariance](#4-second-order-tweedie-the-jacobian-is-the-posterior-covariance)
5. [What the theorem does and does not require](#5-what-the-theorem-does-and-does-not-require)
6. [Equivalent forms and reparameterizations](#6-equivalent-forms-and-reparameterizations)
7. [Score matching: why MSE denoising learns the score](#7-score-matching-why-mse-denoising-learns-the-score)
8. [Conditional Miyasawa](#8-conditional-miyasawa)
9. [Classifier-free guidance](#9-classifier-free-guidance)
10. [Extension 1: linear transformation and correlated noise](#10-extension-1-linear-transformation-and-correlated-noise)
11. [Extension 2: multiplicative and composite (Poisson-Gaussian) noise](#11-extension-2-multiplicative-and-composite-poisson-gaussian-noise)
12. [Bias-free architectures: what they are and what they actually buy](#12-bias-free-architectures-what-they-are-and-what-they-actually-buy)
13. [Input normalization for bias-free denoisers](#13-input-normalization-for-bias-free-denoisers)
14. [Sampling with a denoiser](#14-sampling-with-a-denoiser)
15. [Inverse problems, PnP and RED](#15-inverse-problems-pnp-and-red)
16. [Diagnostics: SURE, generalized SURE, homogeneity, DC probe](#16-diagnostics-sure-generalized-sure-homogeneity-dc-probe)
17. [Limitations and failure modes](#17-limitations-and-failure-modes)
18. [Reference implementation](#18-reference-implementation)
19. [History and references](#19-history-and-references)
20. [Appendix A: quick-reference table](#appendix-a-quick-reference-table)
21. [Appendix B: conventions that are commonly mixed up](#appendix-b-conventions-that-are-commonly-mixed-up)

---

## 1. Notation and conventions

| Symbol | Meaning |
|---|---|
| $x \in \mathbb{R}^N$ | clean signal, drawn from prior $p(x)$ |
| $\varepsilon$ | noise |
| $y$ | noisy observation |
| $p(y)$ | density of the **noisy** observation (a Gaussian-smoothed version of $p(x)$) |
| $\hat{x}(y)$ | the MMSE (least-squares optimal) estimator, i.e. $\mathbb{E}[x\mid y]$ |
| $D_\theta$ | a neural denoiser, an *approximation* to $\hat{x}$ |
| $s(y) = \nabla_y \log p(y)$ | score of the noisy density |
| $E(y) = -\log p(y)$ | energy (up to an additive constant) |
| $J(y) = \partial \hat{x}/\partial y$ | Jacobian of the estimator |

Two things that are constantly conflated and should not be:

- $\nabla_y \log p(y)$ is the score of the **noise-smoothed** density $p_\sigma = p_x * \mathcal{N}(0,\sigma^2 I)$, **not** of the clean prior $p(x)$. It converges to the clean score only as $\sigma \to 0$, and $\nabla \log p_x$ may not even exist if the data lie on a lower-dimensional manifold.
- $\hat{x}$ denotes the exact MMSE estimator (a mathematical object); $D_\theta$ denotes a trained network. Every identity below is exact for $\hat{x}$ and approximate for $D_\theta$.

---

## 2. Statement and hypotheses

**Setup (additive white Gaussian noise).**

$$y = x + \varepsilon, \qquad x \sim p(x), \qquad \varepsilon \sim \mathcal{N}(0, \sigma^2 I), \qquad x \perp \varepsilon.$$

**Theorem (Miyasawa 1961; Tweedie's formula, cf. Robbins 1956, Efron 2011).**
The least-squares optimal denoiser satisfies

$$\boxed{\ \hat{x}(y) \;=\; \mathbb{E}[x \mid y] \;=\; y + \sigma^2\, \nabla_y \log p(y).\ }$$

**Hypotheses actually needed.**

1. $\mathbb{E}\|x\| < \infty$ (so the conditional mean exists).
2. The noise is additive Gaussian with known, signal-independent covariance $\sigma^2 I$, independent of $x$.
3. Nothing else. $p(y)$ is automatically smooth and strictly positive everywhere, because it is a convolution with a Gaussian, so differentiation under the integral sign is justified by dominated convergence. There is no smoothness or support requirement on the prior $p(x)$: it may be discrete, singular, or supported on a measure-zero manifold.

**What the theorem is.** It is a statement about the *exact posterior mean*, an object defined by
the prior and the noise model. It says nothing about how you compute or approximate that object.

---

## 3. Derivation

**Step 1. The MMSE estimator is the posterior mean.**

$$\arg\min_{f} \mathbb{E}\big[\|f(y) - x\|^2\big] \;=\; \mathbb{E}[x\mid y] \;=\; \int x\, p(x\mid y)\, dx .$$

**Step 2. Gaussian likelihood and its gradient.**

$$p(y\mid x) = (2\pi\sigma^2)^{-N/2} \exp\!\Big(-\tfrac{\|y-x\|^2}{2\sigma^2}\Big),
\qquad
\nabla_y\, p(y\mid x) = \frac{x-y}{\sigma^2}\, p(y\mid x).$$

The essential structural fact: the gradient of the log-likelihood is **linear in $x$**. This is
what makes the whole thing collapse to a first moment.

**Step 3. Differentiate the marginal.**

$$p(y) = \int p(y\mid x)\, p(x)\, dx
\;\Longrightarrow\;
\nabla_y\, p(y) = \int \frac{x-y}{\sigma^2}\, p(y\mid x)\, p(x)\, dx .$$

**Step 4. Convert to a posterior expectation.** Using $p(y\mid x)p(x) = p(x\mid y)p(y)$,

$$\nabla_y\, p(y) = \frac{p(y)}{\sigma^2} \int (x-y)\, p(x\mid y)\, dx
= \frac{p(y)}{\sigma^2}\big(\mathbb{E}[x\mid y] - y\big).$$

**Step 5. Divide by $p(y) > 0$.**

$$\nabla_y \log p(y) = \frac{1}{\sigma^2}\big(\hat{x}(y) - y\big)
\qquad\Longleftrightarrow\qquad
\hat{x}(y) = y + \sigma^2 \nabla_y \log p(y). \qquad \blacksquare$$

---

## 4. Second-order Tweedie: the Jacobian is the posterior covariance

Differentiating the identity once more gives a result that is at least as useful as the first,
and that the source drafts omit entirely:

$$\boxed{\ J(y) \;=\; \frac{\partial \hat{x}}{\partial y} \;=\; I + \sigma^2 \nabla_y^2 \log p(y) \;=\; \frac{1}{\sigma^2}\,\operatorname{Cov}[x \mid y].\ }$$

Consequences:

- **The Jacobian of the exact MMSE denoiser is automatically symmetric and positive semidefinite.** Symmetry is not an extra assumption on $\hat{x}$; it is forced. It *is* an extra assumption when you swap $\hat{x}$ for a trained network $D_\theta$ (see [§15](#15-inverse-problems-pnp-and-red) on RED).
- The eigenvalues of $\sigma^2 \nabla^2 \log p(y)$ are $\ge -1$.
- **Free uncertainty quantification.** $\operatorname{Cov}[x\mid y] = \sigma^2 J(y)$: a Jacobian-vector product against a trained denoiser gives per-direction posterior variance without any extra head or ensemble.
- **Risk.** $\mathrm{MMSE} = \mathbb{E}\big[\operatorname{tr}\operatorname{Cov}[x\mid y]\big] = \sigma^2\,\mathbb{E}\big[\operatorname{tr} J(y)\big]$, which is exactly the term that appears in SURE ([§16](#16-diagnostics-sure-generalized-sure-homogeneity-dc-probe)).

---

## 5. What the theorem does and does not require

This section exists because the source drafts get it wrong, and the error propagates into
architecture decisions.

**Required:** additive Gaussian noise, independence, finite first moment, and that the estimator
be the *exact* conditional mean.

**Not required, at all:**

- **A bias-free architecture.** The theorem is a statement about $\mathbb{E}[x\mid y]$, not about any network. A denoiser with biases, with a sigmoid head, or implemented as BM3D, that happened to be exactly MMSE-optimal would satisfy the identity exactly. Bias-free design is a separate inductive bias with separate justifications ([§12](#12-bias-free-architectures-what-they-are-and-what-they-actually-buy)).
- **A linear final activation.** Same reason. It is nevertheless good practice, for the reason given below, which is not the reason the drafts give.
- **A symmetric Jacobian.** As shown in [§4](#4-second-order-tweedie-the-jacobian-is-the-posterior-covariance), symmetry is a *consequence* for the exact estimator, not a hypothesis.

**The correct reasons for the usual architectural choices:**

| Choice | Correct justification | Incorrect justification found in the drafts |
|---|---|---|
| MSE loss | MSE is minimized by the conditional mean; L1 gives the conditional median, perceptual/adversarial losses give neither. This is the only architectural choice the theorem genuinely constrains. | (correct in the drafts) |
| Linear output head | The residual $\hat{x}-y = \sigma^2 s(y)$ is unbounded and signed; a bounded (sigmoid, tanh) or one-sided (ReLU) head cannot represent it, and any head with a nonzero output at zero input injects a constant offset. | "Non-linear activations introduce bias because $\mathbb{E}[\mathrm{sigmoid}(f(x+\varepsilon))] \ne x$", which is a non-sequitur (that inequality is also true of the correct estimator composed with sigmoid). |
| Residual parameterization $D(y) = y + r_\theta(y)$ | The network then directly regresses $\sigma^2 s(y)$, which is small and zero-mean, so the score readout is not a difference of two large nearly-equal numbers. Better conditioning, no theoretical content. | "Matches theory" (it does, but that is a statement about arithmetic, not a requirement). |
| No bias terms | Enforces $f(\alpha y) = \alpha f(y)$, which is an inductive bias useful for blind denoising across noise levels. | "Critical for theoretical guarantees" / "required for Miyasawa's theorem". False. |

---

## 6. Equivalent forms and reparameterizations

**Residual form**

$$\hat{x}(y) - y = \sigma^2 \nabla_y \log p(y).$$

**Score readout from a denoiser**

$$s(y) \;\approx\; \frac{D_\theta(y) - y}{\sigma^2}.$$

**Energy form**, with $E(y) = -\log p(y) + \text{const}$

$$\hat{x}(y) - y = -\sigma^2 \nabla_y E(y).$$

One denoiser step is therefore a gradient-descent step on the energy of the *noisy* density with
step size $\sigma^2$. The energy is only ever accessible through its gradient; the normalizing
constant never appears, which is the entire practical appeal.

**Noise-prediction form.** If $\hat{\varepsilon}(y) = \mathbb{E}[\varepsilon\mid y] = y - \hat{x}(y)$,

$$s(y) = -\frac{\hat{\varepsilon}(y)}{\sigma^2}.$$

If instead the network predicts *unit-variance* noise, $\varepsilon_\theta(y) \approx \varepsilon/\sigma$ (the DDPM convention), then

$$s(y) = -\frac{\varepsilon_\theta(y)}{\sigma}.$$

Mixing these two up by a factor of $\sigma$ is the single most common implementation bug in this area.

**Variance-preserving / DDPM scaling.** With $x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon$, $\varepsilon \sim \mathcal{N}(0,I)$, the conditional law is $x_t \mid x_0 \sim \mathcal{N}(\sqrt{\bar\alpha_t} x_0,\ (1-\bar\alpha_t) I)$. Applying Tweedie to the mean $u = \sqrt{\bar\alpha_t}x_0$:

$$\boxed{\ \mathbb{E}[x_0 \mid x_t] = \frac{x_t + (1-\bar\alpha_t)\,\nabla_{x_t}\log p_t(x_t)}{\sqrt{\bar\alpha_t}}\ }$$

which is exactly the standard "predict $x_0$ from $\varepsilon_\theta$" formula once you substitute
$\nabla \log p_t = -\varepsilon_\theta/\sqrt{1-\bar\alpha_t}$. Every $x_0$-prediction / $\varepsilon$-prediction /
$v$-prediction parameterization in the diffusion literature is this identity rearranged.

---

## 7. Score matching: why MSE denoising learns the score

**Explicit score matching** (the objective one wants) is

$$\mathcal{L}_{\mathrm{ESM}}(\theta) = \mathbb{E}_{p(y)}\big\|s_\theta(y) - \nabla_y \log p(y)\big\|^2,$$

which is not directly computable, since $\nabla \log p$ is unknown. Hyvärinen (2005) showed it
equals, up to a constant, $\mathbb{E}\big[\operatorname{tr}\nabla s_\theta(y) + \tfrac12\|s_\theta(y)\|^2\big]$,
which is computable but requires a trace of a Jacobian per sample and scales badly.

**Denoising score matching** (Vincent 2011) replaces the unknown marginal score with the known
conditional one, $\nabla_y \log p(y\mid x) = (x-y)/\sigma^2$:

$$\mathcal{L}_{\mathrm{DSM}}(\theta) = \mathbb{E}_{x,\varepsilon}\Big\|s_\theta(y) - \frac{x-y}{\sigma^2}\Big\|^2,
\qquad y = x + \varepsilon .$$

$\mathcal{L}_{\mathrm{DSM}}$ and $\mathcal{L}_{\mathrm{ESM}}$ differ by a constant independent of $\theta$, so they
have the same minimizer.

**The link to plain MSE denoising is exact, not analogical.** Substituting the residual
parameterization $s_\theta(y) = (D_\theta(y) - y)/\sigma^2$:

$$\mathcal{L}_{\mathrm{DSM}}(\theta) = \frac{1}{\sigma^4}\,\mathbb{E}\big\|D_\theta(y) - x\big\|^2 .$$

So training a denoiser with MSE **is** denoising score matching up to the constant $\sigma^{-4}$.
This, rather than any hand-waving about "implicit priors", is the reason a trained denoiser gives
you a score estimate.

**Multi-noise-level training.** In practice one trains a single network over a range of $\sigma$,
either by conditioning it on $\sigma$ or by leaving it blind. The per-level objectives are usually
weighted by $\lambda(\sigma) \propto \sigma^2$ (or $\sigma^4$ in the score parameterization) so that
each noise level contributes comparably; unweighted score-space losses are dominated by the
smallest $\sigma$.

---

## 8. Conditional Miyasawa

Let $c$ be any conditioning variable: a class label, an RGB image, a text embedding, a tuple of
these. Assume only that **the noise is independent of the conditioning**, $p(y\mid x,c) = p(y\mid x)$.
Then, repeating the derivation of [§3](#3-derivation) with every density conditioned on $c$:

$$\boxed{\ \hat{x}(y,c) = \mathbb{E}[x\mid y,c] = y + \sigma^2\,\nabla_y \log p(y\mid c).\ }$$

Derivation, condensed: $\nabla_y p(y\mid c) = \int \frac{x-y}{\sigma^2} p(y\mid x) p(x\mid c)\,dx = \frac{p(y\mid c)}{\sigma^2}\big(\mathbb{E}[x\mid y,c]-y\big)$.

**Nothing in the mathematics depends on the type of $c$.** Discrete labels, dense conditioning
images (for example depth-from-RGB posed as denoising a depth map conditioned on the image), and
hybrid conditioning are all the same theorem. What differs is only the architecture used to encode
and inject $c$.

**Bayes decomposition of the conditional score:**

$$\nabla_y \log p(y\mid c) = \nabla_y \log p(y) + \nabla_y \log p(c\mid y).$$

The second term is what classifier guidance estimates with a separate noisy classifier, and what
classifier-free guidance estimates implicitly as the difference of two denoiser outputs.

**Injection and homogeneity (important, and wrong in the source drafts).** If you want the network
to remain degree-1 homogeneous in $y$ (see [§12](#12-bias-free-architectures-what-they-are-and-what-they-actually-buy)),
then how you inject $c$ matters:

- **Additive injection** (broadcast a projected class embedding and add it to the features) injects a
  constant that does not scale with $y$. It **breaks homogeneity**: $f(\alpha y, c) \ne \alpha f(y,c)$.
  A source draft "proves" scaling invariance for this case via the step
  $\alpha y + e_c = \alpha(y + e_c/\alpha)$, which is an identity about numbers, not a proof of
  equivariance, and the conclusion is false.
- **Multiplicative injection** (scale-only FiLM: $h \mapsto h \odot \gamma(c)$, with $\gamma$ a function of $c$ alone) **preserves homogeneity**, because $\gamma$ does not depend on $y$.
- **Channel concatenation** of a tiled conditioning tensor also breaks homogeneity in $y$, for the same reason as additive injection: the concatenated channels do not scale with $y$.

Pick one and be honest about it. Either use scale-only modulation and keep exact homogeneity, or
use additive injection and accept that the network is no longer homogeneous (which is fine, but
then do not also claim noise-level generalization from homogeneity, and do not run a DC probe and
expect it to pass).

**Conditioning on $\sigma$ while keeping equivariance.** If you feed $\sigma$ to the network,
the natural structure to preserve is *joint* degree-1 homogeneity, $D(\alpha y, \alpha\sigma) = \alpha D(y,\sigma)$.
The clean way to get it exactly is

$$D(y, \sigma) = \sigma \cdot g_\theta\!\big(y/\sigma\big),$$

with $g_\theta$ an ordinary (bias-allowed) network. This is a strictly larger and better-matched
hypothesis class than a blind bias-free net, and it is the recommended route whenever $\sigma$ is known.

---

## 9. Classifier-free guidance

**Two conventions are in circulation and they differ by one.** Both appear in the source drafts,
unreconciled. State which you are using.

| Convention | Formula | Unconditional | Plain conditional |
|---|---|---|---|
| Interpolation ("$w$ is the guidance scale", used by most codebases) | $\tilde{s} = s_u + w\,(s_c - s_u)$ | $w=0$ | $w=1$ |
| Ho and Salimans (2022) ("$w$ is the guidance *strength*") | $\tilde{s} = (1+w)\,s_c - w\,s_u$ | $w=-1$ | $w=0$ |

They are the same family: $w_{\text{interp}} = 1 + w_{\text{HS}}$.

**Guidance on denoiser outputs equals guidance on scores.** Because $s = (D(y)-y)/\sigma^2$ is affine
in $D$ and the two combinations use weights summing to one,

$$\frac{\big[D_u + w(D_c - D_u)\big] - y}{\sigma^2} \;=\; s_u + w\,(s_c - s_u),$$

so it is legitimate (and cheaper) to apply CFG directly to the two denoiser outputs. This holds only
because the weights sum to one; it fails for any "guidance" formula that does not preserve that.

**Training.** Reserve one extra token as the null condition and replace the true label with it with
probability $p_{\text{drop}}$ (typically 0.1, with 0.05 to 0.2 a reasonable range). The single network
then estimates both $s_c$ and $s_u$.

**What guidance is not.** $\tilde{s}$ is not the score of any normalized density for $w \ne 1$
(interpolation convention). Guidance trades distributional fidelity for conditional fidelity; large
$w$ produces oversaturated, low-diversity samples. Treat the usual "$w \in [1,10]$" table as a
practitioner's heuristic, not a result.

---

## 10. Extension 1: linear transformation and correlated noise

**Model.** $y = Ax + \varepsilon$, $\varepsilon \sim \mathcal{N}(0,\Sigma)$ with $\Sigma \succ 0$, $\varepsilon \perp x$.

Then $y\mid x \sim \mathcal{N}(Ax, \Sigma)$, so $\nabla_y p(y\mid x) = -\Sigma^{-1}(y - Ax)\,p(y\mid x)$, hence
$(Ax - y)p(y\mid x) = \Sigma \nabla_y p(y\mid x)$. Integrating against $p(x)$ and dividing by $p(y)$:

$$\boxed{\ A\,\mathbb{E}[x\mid y] \;=\; y + \Sigma\, \nabla_y \log p(y).\ }$$

**Read this carefully.** The identity determines only $A\hat{x}$, that is, the component of the
posterior mean in the row space of $A$. If $A$ is not injective, $\hat{x}$ is **not** recoverable
from the score alone: the null-space component is supplied by the prior through the estimator, not
by this identity. There is no valid closed form of the shape
$\hat{x} = A^{\dagger}\big[y + \Sigma A^{\top}(AA^{\top}\Sigma + \sigma^2 I)^{-1}\nabla_y \log p(y)\big]$;
that expression, which appears in one source draft, is not dimensionally coherent as a general
result and should be discarded.

**Special case: noise then blur.** With $y = K(x+\varepsilon)$, $\varepsilon\sim\mathcal{N}(0,\sigma^2 I)$, we have
$A = K$ and $\Sigma = \sigma^2 KK^{\top}$, so

$$K\hat{x}(y) = y + \sigma^2 (KK^{\top})\,\nabla_y \log p(y),
\qquad
\nabla_y \log p(y) = \frac{1}{\sigma^2}(KK^{\top})^{-1}\big(K\hat{x}(y) - y\big),$$

the second form requiring $KK^{\top}$ invertible. For a blur kernel this is ill-conditioned at the
kernel's spectral zeros, so solve the system iteratively (conjugate gradient) with regularization
rather than inverting.

**Practical consequence.** A network trained to restore $x$ from a *degraded* observation does not
expose the prior score by subtracting its input. Restoration residuals and score readouts coincide
only in the pure-denoising case $A = I$, $\Sigma = \sigma^2 I$.

**Claims to avoid.** Two "extensions" in the source drafts are not supported:

- *Heavy-tailed / $\alpha$-stable:* "$\hat{x}(y) = y + \sigma^{\alpha}\nabla_y \log p_\alpha(y)$" is not a
  theorem. Tweedie's formula is a property of exponential families; the $\alpha$-stable case has no
  such elementary first-moment identity (the score relation involves fractional operators, and for
  $\alpha < 2$ the mean may not even exist for $\alpha \le 1$).
- *Binary/Bernoulli:* "$\hat{x}_i(y) = \mathrm{sigmoid}(\nabla_{y_i}\log p(y))$" is meaningless on a
  discrete domain where $\nabla_{y_i}$ is undefined. The correct discrete analogue replaces the
  derivative with a finite ratio $p(y^{\oplus i})/p(y)$ of the density at bit-flipped neighbours.
- *Riemannian:* $\hat{x}(y) = \exp_y(\sigma^2 \operatorname{grad}_g \log p(y))$ is a reasonable
  small-noise heuristic and the basis of Riemannian score-based models, but it is an approximation
  (the exact posterior mean is not even well defined on a manifold without choosing a notion of mean,
  e.g. Fréchet), not an identity.

---

## 11. Extension 2: multiplicative and composite (Poisson-Gaussian) noise

### 11.1 Pure multiplicative Gaussian noise

$$y = x\cdot n, \qquad n \sim \mathcal{N}(1,\sigma^2)\ \text{per pixel}, \qquad x \perp n
\quad\Longleftrightarrow\quad y\mid x \sim \mathcal{N}(x,\ \sigma^2 x^2).$$

The conditional variance is signal-dependent, and the log-likelihood gradient is no longer linear
in $x$. Differentiating,

$$\partial_y p(y\mid x) = -\frac{y-x}{\sigma^2 x^2} p(y\mid x)
\quad\Longrightarrow\quad
(x-y)\,p(y\mid x) = \sigma^2 x^2\, \partial_y p(y\mid x).$$

Integrating against $p(x)$ and pulling $\partial_y$ out of the integral (legitimate because $x^2$ does
not depend on $y$):

$$\boxed{\ \mathbb{E}[x\mid y] = y + \sigma^2\,\frac{\partial_y\big[\mathbb{E}[x^2\mid y]\,p(y)\big]}{p(y)}\ }\tag{A}$$

Relation (A) is **exact for all $\sigma$**. The correction now involves the **second** posterior
moment. A single-output denoiser exposes only the first moment, so:

> Under multiplicative noise there is no residual-equals-score identity. The residual of an optimal
> denoiser is not a rescaled score, and no amount of architectural care recovers one.

### 11.2 Small-$\sigma$ expansion

For small $\sigma$ the posterior concentrates, $\mathbb{E}[x^2\mid y] \approx y^2$. Then
$\partial_y[y^2 p] = 2yp + y^2 \partial_y p$, so

$$\boxed{\ D(y) - y \;\approx\; 2\sigma^2 y \;+\; \sigma^2 y^2\,\nabla_y \log p(y).\ }\tag{B}$$

Two structurally different terms:

- $2\sigma^2 y$: **prior-independent, signal-proportional, and directed away from zero.** Sanity
  check with a flat (improper uniform) prior, where the score term vanishes: a second-order
  expansion of the posterior gives $\mathbb{E}[x\mid y] \approx y(1+2\sigma^2)$, matching (B). This term
  is an *inflation*, not a shrinkage, and it arises from the $1/|x|$ normalizer and the $x$-dependent
  variance in the likelihood, both of which tilt the posterior outward.
- $\sigma^2 y^2 \nabla_y \log p(y)$: the score term, reweighted by the **local variance** $\sigma^2 y^2$
  instead of the constant $\sigma^2$.

### 11.3 Composite (affine-variance / Poisson-Gaussian) noise

$$y = x\cdot n + a, \quad n\sim\mathcal{N}(1,\sigma_m^2),\ a\sim\mathcal{N}(0,\sigma_a^2)
\quad\Longleftrightarrow\quad
y\mid x \sim \mathcal{N}\big(x,\ \sigma_m^2 x^2 + \sigma_a^2\big).$$

This is the Gaussian approximation of the standard sensor model (Foi et al. 2008): a read-noise floor
$\sigma_a^2$ plus a signal-dependent shot-noise-like term. The variance factor splits linearly, so the
derivation superposes exactly:

$$\boxed{\ \mathbb{E}[x\mid y] = y + \sigma_a^2\,\nabla_y \log p(y) + \sigma_m^2\,\frac{\partial_y\big[\mathbb{E}[x^2\mid y]\,p(y)\big]}{p(y)}\ }\tag{A$_c$}$$

$$\boxed{\ D(y) - y \;\approx\; \big(\sigma_a^2 + \sigma_m^2 y^2\big)\,\nabla_y \log p(y) \;+\; 2\sigma_m^2 y\ }\tag{B$_c$}$$

with the local variance $v(y) = \sigma_a^2 + \sigma_m^2 y^2$ as the score weight. Limits: $\sigma_a=0$
recovers (A)/(B); $\sigma_m=0$ recovers additive Miyasawa.

**Why the additive floor matters.** Under pure multiplicative noise the conditional variance
$\sigma_m^2 x^2$ vanishes as $x\to 0$, so near-zero pixels are essentially uncorrupted and the score
weight collapses; the empirical-Bayes quantities are ill-conditioned there. The floor $\sigma_a^2$
bounds $v(y)$ away from zero everywhere. This is both physically right and numerically necessary.

**Log / variance-stabilizing transform.** With $u=\log x$, $v = \log y$, one has $v = u + \log n \approx u + (n-1)$
for small $\sigma$, so additive Miyasawa applies in log space. This requires $x > 0$. On a
strictly-positive $[0,1]$ domain (see [§13](#13-input-normalization-for-bias-free-denoisers)) the
transform is therefore *available in principle*, though badly conditioned near $0$ and it changes the
loss geometry (MSE in log space is not MSE in linear space, so the trained network is no longer the
linear-domain MMSE estimator). One source draft rules the transform out on the grounds that the
domain is signed $[-1,+1]$; that is inconsistent with the same repository's own $[0,1]$ decision. If
you stay in the linear domain, do so as a deliberate choice, not because the log is unavailable.

### 11.4 The bias-free question under multiplicative noise (source drafts are wrong here)

A source draft claims that strict bias-free design is theoretically wrong for multiplicative noise
because the $2\sigma^2 y$ term is "non-homogeneous". **It is not.** $2\sigma^2 y$ is linear in $y$ and
therefore degree-1 homogeneous at fixed $\sigma$. Check the whole of (B) under a scaled prior
$p_\alpha(x) = \alpha^{-1}p(x/\alpha)$, whose marginal satisfies $\nabla \log p_\alpha(\alpha y) = \alpha^{-1}\nabla\log p(y)$:

$$D_\alpha(\alpha y) = \alpha y + 2\sigma^2(\alpha y) + \sigma^2(\alpha y)^2\cdot\tfrac1\alpha \nabla\log p(y) = \alpha\,D(y).$$

So the multiplicative MMSE denoiser is **exactly scale-equivariant at fixed $\sigma$**, without needing
to co-scale the noise level, which is a *stronger* compatibility with a homogeneous network than the
additive case enjoys (there you must co-scale $\sigma$ along with $x$). If anything, bias-free design
is more natural here, not less.

The real caveat under multiplicative noise is the one in §11.1: **the residual is no longer the score**.
That breaks score readout, RED, and denoiser-driven sampling, and no architectural change fixes it.
Keep the bias-free network if you like it; just do not read $\big(D(y)-y\big)/\sigma^2$ as a score.

---

## 12. Bias-free architectures: what they are and what they actually buy

**Definition.** Every affine map in the network has zero additive offset: `use_bias=False` in all
convolutions and dense layers, `center=False` in normalization layers (so no learned $\beta$), and no
activation with a nonzero value at zero on the output path.

**Homogeneity.** With positively-homogeneous activations (ReLU, leaky ReLU, PReLU) and no offsets, the
network is positively homogeneous of degree one:

$$f(\alpha y) = \alpha f(y)\quad \text{for all } \alpha > 0, \qquad\text{hence}\qquad f(0)=0.$$

Equivalently $f(y) = J(y)\,y$ with $J$ locally constant: the network is piecewise linear through the
origin. This is what makes bias-free nets analyzable (you can look at the effective filters $J(y)$ for
each input).

**Two caveats the drafts do not mention.**

1. **Batch normalization breaks homogeneity in training mode.** With batch statistics,
   $\mathrm{BN}(\alpha x) = \gamma \cdot \frac{\alpha x - \alpha\mu}{\alpha\varsigma} = \mathrm{BN}(x)$: degree **zero**, not one.
   The homogeneity property holds only in inference mode, where the running statistics are frozen
   constants and BN reduces to a fixed diagonal scaling. Run all homogeneity and DC diagnostics with
   `training=False`. If you want homogeneity to hold during training too, use a normalization that
   does not divide by an input-dependent statistic, or drop normalization on the output path.
2. **Homogeneity is not implied by the theorem, and is exactly correct only for a scale-invariant prior.**
   The exact statement for additive noise is a joint equivariance: if the prior is scaled by $\alpha$
   *and* the noise level by $\alpha$, the MMSE estimator scales by $\alpha$. A $\sigma$-blind homogeneous
   network conflates "scale the input" with "scale the prior and the noise together", which is exactly
   right only if $p$ is scale-invariant. Natural image statistics are approximately scale-invariant,
   which is the honest justification, and the empirical payoff is what Mohan et al. (ICLR 2020)
   measured: generalization to noise levels far outside the training range, where biased networks fail
   badly.

**When to prefer bias-free:** blind denoising, wide or unknown noise-level ranges, when you want
interpretable effective filters, when you want to reuse the denoiser as an implicit prior across scales.

**When not to bother:** when $\sigma$ is known and you can condition on it (use $D(y,\sigma) = \sigma g(y/\sigma)$
instead, which is strictly more expressive and exactly equivariant), or when the conditioning mechanism
already breaks homogeneity anyway ([§8](#8-conditional-miyasawa)).

---

## 13. Input normalization for bias-free denoisers

The source drafts contain a self-reversal on this point, plus an overstated argument. Here is what is
actually true.

**Fact 1 (structural).** A bias-free network cannot represent a DC offset. It has no mechanism to add
or subtract a constant. Therefore the input domain is **not** a free relabeling: a model trained on
$[0,1]$ fed data on $[-0.5,+0.5]$ produces silent garbage, not an error. Record the data range in the
checkpoint metadata and refuse to load a checkpoint whose range does not match.

**Fact 2 (scale, not shift).** $[0,1]$ and $[-0.5,+0.5]$ have the same peak-to-peak width of $1.0$.
Moving between them is a pure DC shift. So $\sigma$, `max_val` for PSNR/SSIM, and any conversion like
$\sigma_{255} = 255\sigma$ are **unchanged**. Rescaling them "because the domain moved" silently corrupts
every reported dB number, and nothing fails loudly. This is the most likely mistake in this area.

**Fact 3 (flat patches).** For a flat patch of value $c>0$, homogeneity gives $f(c\mathbf{1}) = c f(\mathbf{1})$.
Preserving the patch therefore requires $f(\mathbf{1}) = \mathbf{1}$: the local filters must sum to one, which
is the correct DC-preserving behaviour for a denoiser.

**What is true about $[0,1]$ versus zero-centered, stated correctly.** The claim in one draft that
zero-centering makes the sum-to-one property "vacuous" or "never supervised" is **overstated**. On
$[-0.5,+0.5]$, any flat patch with $c \ne 0$ still imposes $f(\mathbf{1}) = \mathbf{1}$ (for $c>0$) or
$f(-\mathbf{1}) = -\mathbf{1}$ (for $c<0$). Only the single value $c=0$ is degenerate. The defensible version of
the argument is a **weighting** argument:

- The squared-error contribution of a flat patch at level $c$ is $c^2\,\|f(\mathbf{1})-\mathbf{1}\|^2$, so the
  gradient signal for the DC property scales as $c^2$.
- On $[0,1]$, mid-grey sits at $c=0.5$ (weight $0.25$) and white at $c=1$ (weight $1$).
- On $[-0.5,+0.5]$, mid-grey sits at $c=0$ (weight $0$) and the extremes only reach weight $0.25$.
- Natural images have a large mass of near-mid-grey flat content (sky, walls, paper, out-of-focus
  background). Zero-centering places exactly that mass at the point where the DC constraint carries
  no gradient, and it splits the remaining constraint across two independent rays ($f(\mathbf{1})$ and
  $f(-\mathbf{1})$), since positive homogeneity relates $f(\alpha y)$ to $f(y)$ only for $\alpha>0$.

So: **$[0,1]$ is the better default for a bias-free denoiser**, for weighting and diagnosability
reasons, not because the alternative is mathematically vacuous. It is also what Mohan et al. and
Kadkhodaie and Simoncelli use, and it makes the sampler initialization at mid-grey ($0.5\mathbf{1}$, a
nonzero vector) well behaved.

**What remains unverified.** The concern that strictly-positive inputs aggravate dead-ReLU behaviour is
a real failure mode in general and is not refuted by the argument above. The structural argument
constrains what the network *can* learn; it does not prove the optimizer will get there. Treat any
claim about $[0,1]$ denoising quality as unverified until a full retrain is measured against a matched
baseline. Stop conditions worth pre-committing to: divergence, validation loss never falling below its
epoch-0 value, or a DC probe that moves *away* from $f(c\mathbf{1}) = c\mathbf{1}$ as training proceeds.

**Other domains.** Z-scoring per image or per dataset destroys the pixel domain, makes `max_val`
ill-defined for PSNR/SSIM, and couples every image to dataset statistics. Avoid it for denoising.

**Do not clip the noisy input.** Clipping $y$ to $[0,1]$ after adding noise changes the likelihood from
Gaussian to a truncated Gaussian, so the MMSE target is no longer the one the theorem describes, and
the bias grows with $\sigma$ and with proximity to the domain edges. Clip only for display. If you must
clip for pipeline reasons, document it as an approximation whose error concentrates at the extremes.

---

## 14. Sampling with a denoiser

### 14.1 Annealed Langevin dynamics

Unadjusted Langevin for a fixed density $p$:

$$y \leftarrow y + \tfrac{\eta}{2}\,\nabla_y \log p(y) + \sqrt{\eta}\,z, \qquad z\sim\mathcal{N}(0,I).$$

(Some references write $y \leftarrow y + \eta s + \sqrt{2\eta}z$; that is the same recursion with
$\eta' = 2\eta$. Be consistent, or your effective temperature is off by a factor of two.)

With a denoiser supplying the score at noise level $\sigma$, sweep $\sigma$ from large to small and use
$\eta_\sigma \propto \sigma^2$, which keeps the signal-to-noise ratio of each step roughly constant across
levels (Song and Ermon 2019). A single fixed step size across a wide $\sigma$ range does not work.

### 14.2 Denoiser-driven coarse-to-fine sampling (Kadkhodaie and Simoncelli 2021)

This is the sampler that is native to the Miyasawa formulation, because it estimates the noise level
from the residual itself rather than requiring a schedule:

```
inputs: denoiser f, sigma_0 (large), sigma_L (small), h_0 in (0,1], beta in [0,1]
y <- N(0.5 * 1, sigma_0^2 I)          # mid-grey initialization on the [0,1] domain
t <- 1
while sigma_t > sigma_L:
    h_t     <- h_0 * t / (1 + h_0 * (t - 1))
    d_t     <- f(y) - y                       # = sigma_t^2 * score, by Miyasawa
    sigma_t <- ||d_t|| / sqrt(N)              # effective noise level, read off the residual
    gamma_t <- sqrt(((1 - beta*h_t)^2 - (1 - h_t)^2)) * sigma_t
    y       <- y + h_t * d_t + gamma_t * N(0, I)
    t       <- t + 1
return y
```

`beta = 0` gives a deterministic gradient ascent onto the manifold (useful for inverse problems);
`beta = 1` gives fully stochastic sampling.

### 14.3 CFG sampling

Batch the conditional and unconditional passes into one forward call of doubled batch size, then
combine the two *denoiser outputs* with weights summing to one ([§9](#9-classifier-free-guidance)).

---

## 15. Inverse problems, PnP and RED

**Plug-and-play (Venkatakrishnan et al. 2013).** Split $\min_x \tfrac{1}{2}\|Ax-b\|^2 + \lambda R(x)$ with
ADMM or proximal gradient, and replace the proximal operator of $R$ with a call to an off-the-shelf
denoiser. Convergence guarantees require assumptions on the denoiser (nonexpansiveness or similar) that
generic networks do not satisfy; spectral-normalized or explicitly constrained denoisers do.

**RED (Romano et al. 2017).** Define $R(x) = \tfrac12 x^{\top}\big(x - D(x)\big)$, and claim
$\nabla R(x) = x - D(x)$. This step requires two conditions on $D$: local homogeneity and **Jacobian
symmetry**. Reehorst and Schniter (2019) showed both typically fail for real denoisers (BM3D, DnCNN,
and standard CNNs), and reinterpreted the RED updates as score-matching-by-denoising, which does not
need an explicit regularizer to exist. The correct framing:

- For the **exact** MMSE denoiser, $J$ is symmetric automatically ([§4](#4-second-order-tweedie-the-jacobian-is-the-posterior-covariance)), so the gradient interpretation is sound.
- For a **trained** denoiser, symmetry is an assumption, usually violated, and the algorithm should be
  justified as a fixed-point scheme rather than as descent on an energy.
- Bias-free architectures give exact local homogeneity but say nothing about symmetry.

**Score-based posterior sampling.** The cleanest formulation avoids RED entirely: alternate a
denoiser-driven prior step (§14.2) with a measurement-consistency projection or gradient step,

$$x \leftarrow x + h\big(D(x) - x\big) \;-\; \mu\, A^{\top}(Ax - b) \;+\; \gamma z .$$

Signs matter: $D(x)-x$ is $+\sigma^2\nabla\log p$ (ascent on log-prior) while $A^{\top}(Ax-b)$ is the
gradient of the data term (subtract it). One source draft mixes these signs between two versions of the
same function.

---

## 16. Diagnostics: SURE, generalized SURE, homogeneity, DC probe

### 16.1 SURE (additive Gaussian, reference-free risk)

For $y = x + \mathcal{N}(0,\sigma^2 I)$ and weakly differentiable $D$ (Stein 1981):

$$\mathrm{SURE}(D) = \|D(y)-y\|^2 + 2\sigma^2 \operatorname{div}(D) - N\sigma^2,
\qquad \mathbb{E}[\mathrm{SURE}] = \mathbb{E}\|D(y)-x\|^2 .$$

This estimates the true risk from noisy data alone, with no clean references. Estimate the divergence
with a Hutchinson probe and a finite-difference Jacobian-vector product:

$$\operatorname{div}(D) \approx \mathbb{E}_v\Big[v^{\top}\tfrac{D(y+\epsilon v)-D(y)}{\epsilon}\Big],
\qquad v_i \in \{\pm 1\}\ \text{i.i.d.}$$

Validate the estimator before trusting it: on a linear toy denoiser $D(y) = ay$ the divergence is
analytically $aN$, and SURE should reproduce the realized MSE to a percent or so.

### 16.2 Generalized SURE (signal-dependent variance)

For $y\mid x \sim \mathcal{N}(x,\Sigma)$ with **known** $\Sigma$ (Eldar 2009):

$$\mathrm{gSURE} = \|D(y)-y\|^2 + 2\operatorname{tr}\!\big(\Sigma\, J_D\big) - \operatorname{tr}\Sigma .$$

For the multiplicative model $\Sigma = \sigma^2\operatorname{diag}(x^2)$, which is unknown; the usual
substitution $\Sigma \approx \sigma^2 \operatorname{diag}(y^2)$ makes this a **leading-order approximation,
not an unbiased estimator**. Report it as a consistency scalar for tracking a checkpoint, not as a risk
estimate. The weighted divergence is estimated with a probe pre-scaled so that
$\mathbb{E}[v_iv_j] = \sigma^2 y_i^2 \delta_{ij}$, i.e. $v_i = \sigma|y_i| r_i$ with $r$ Rademacher.

### 16.3 Homogeneity probe (bias-free nets only)

$$\mathrm{err}(\alpha) = \frac{\|f(\alpha y) - \alpha f(y)\|}{\alpha\|f(y)\|}, \qquad \text{expected } \approx 0 \text{ for all } \alpha>0 .$$

Run in inference mode. A nonzero, $\alpha$-dependent value indicates a residual bias term or a
training-mode normalization layer.

### 16.4 DC / sum-to-one probe

$$\mathrm{rel\_err}(c) = \frac{\|f(c\mathbf{1}) - c\mathbf{1}\|}{\|c\mathbf{1}\|} .$$

By homogeneity this is **independent of $c$** and equals $\|f(\mathbf{1})-\mathbf{1}\|/\|\mathbf{1}\|$. A constant column
across $c$ is therefore the *expected signature*, and confirms the probe is measuring the sum-to-one
property and nothing else. A random untrained network gives a large constant (order 1); the number
should fall during training. If it is *not* constant across $c$, your network is not homogeneous
(check for biases, additive conditioning, or training-mode BN).

---

## 17. Limitations and failure modes

1. **Approximation, not identity.** Real denoisers are not MMSE-optimal: finite capacity, finite data,
   imperfect optimization. Every score readout inherits that error, amplified by $1/\sigma^2$ at small
   noise levels. Score estimates at very small $\sigma$ are the least reliable and the most heavily used
   at the end of sampling.
2. **Score of the smoothed density.** You never get $\nabla\log p_x$; you get $\nabla \log p_\sigma$. On
   manifold-supported data the clean score does not exist, which is exactly why multi-scale (annealed)
   methods are needed.
3. **Model mismatch.** Non-Gaussian, signal-dependent, or spatially correlated noise invalidates the
   plain identity; use §10 or §11, or accept a documented approximation.
4. **Clipping and quantization.** Clipping the noisy input, 8-bit quantization, and demosaicing all
   perturb the likelihood. Effects concentrate at the domain edges and grow with $\sigma$.
5. **Jacobian symmetry for trained nets.** Assumed by RED-style methods, generally false.
6. **Sampling is not a proof of correctness.** A denoiser can produce plausible samples while its
   residual field is a poor score estimate (and vice versa). Use SURE and the probes in §16 rather than
   eyeballing samples.
7. **Benchmark numbers.** The FID/IS/AbsRel tables circulating in the source drafts (CIFAR-10, CelebA-HQ,
   FFHQ, ImageNet, NYU) mix conventions, model classes and guidance settings, and several do not match
   the cited papers. Do not propagate them; re-check against the original papers for any number you plan
   to publish.

---

## 18. Reference implementation

Keras 3 style. Minimal, correct, and framework-portable in structure. In PyTorch the equivalents are
`bias=False` on `nn.Conv2d`/`nn.Linear` and `affine=False` (or a norm layer with no learned shift).

### 18.1 Bias-free denoiser

```python
import keras
from keras import layers, ops


def bias_free_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    # center=False removes the learned beta offset. Note: with batch statistics this
    # layer is degree-0 homogeneous; exact degree-1 homogeneity holds in inference mode.
    x = layers.BatchNormalization(center=False, scale=True)(x)
    return layers.Activation("relu")(x)


def bias_free_denoiser(input_shape=(None, None, 1), filters=64, num_blocks=8):
    """Residual, bias-free, linear output head. Predicts x_hat = y + r(y)."""
    inp = keras.Input(shape=input_shape)
    x = inp
    for _ in range(num_blocks):
        x = bias_free_block(x, filters)
    residual = layers.Conv2D(input_shape[-1], 3, padding="same", use_bias=False)(x)
    out = layers.Add()([inp, residual])          # r(y) approximates sigma^2 * score(y)
    return keras.Model(inp, out, name="bias_free_denoiser")
```

### 18.2 Sigma-conditioned alternative (exactly equivariant, more expressive)

```python
class SigmaConditionedDenoiser(keras.Model):
    """D(y, sigma) = sigma * g(y / sigma).

    Satisfies D(a*y, a*sigma) = a*D(y, sigma) exactly, for any inner network g,
    including one with biases. Prefer this whenever sigma is known.
    """

    def __init__(self, inner, **kwargs):
        super().__init__(**kwargs)
        self.inner = inner

    def call(self, inputs, training=None):
        y, sigma = inputs                        # sigma shape (B, 1, 1, 1)
        return sigma * self.inner(y / sigma, training=training)
```

### 18.3 Training data pipeline

```python
import tensorflow as tf


def make_noisy(clean, sigma_min=0.0, sigma_max=0.4):
    """clean is already on [0, 1]. Per-example sigma. No clipping of the noisy input."""
    b = tf.shape(clean)[0]
    sigma = tf.random.uniform([b, 1, 1, 1], sigma_min, sigma_max)
    noisy = clean + sigma * tf.random.normal(tf.shape(clean))
    return noisy, clean, sigma


def to_unit_domain(images):
    """[0, 255] -> [0, 1]. Strictly positive, deliberately not zero-centered."""
    images = tf.cast(images, tf.float32)
    return tf.cond(tf.reduce_max(images) > 1.0, lambda: images / 255.0, lambda: images)
```

Compile with `loss="mse"`. MSE is the one choice the theorem actually dictates: it is the loss whose
minimizer is $\mathbb{E}[x\mid y]$.

### 18.4 Score readout and Langevin sampling

```python
def score_from_denoiser(denoiser, y, sigma):
    """Miyasawa: grad_y log p(y) = (E[x|y] - y) / sigma^2."""
    return (denoiser(y, training=False) - y) / (sigma ** 2)


def annealed_langevin(denoiser, shape, sigmas, steps_per_level=100, eta0=2e-5, seed=None):
    """sigmas: decreasing list. Step size scales as sigma^2."""
    g = tf.random.Generator.from_seed(seed if seed is not None else 0)
    y = 0.5 + sigmas[0] * g.normal(shape)        # mid-grey init on the [0, 1] domain
    for sigma in sigmas:
        eta = eta0 * (sigma / sigmas[-1]) ** 2
        for _ in range(steps_per_level):
            s = score_from_denoiser(denoiser, y, sigma)
            y = y + 0.5 * eta * s + tf.sqrt(eta) * g.normal(tf.shape(y))
    return tf.clip_by_value(y, 0.0, 1.0)         # clip for display only
```

### 18.5 Kadkhodaie-Simoncelli sampler (schedule-free)

```python
def ks_sample(denoiser, shape, sigma_0=1.0, sigma_L=0.01, h0=0.05, beta=0.5, max_iter=500):
    n = float(tf.reduce_prod(shape[1:]).numpy())
    y = 0.5 + sigma_0 * tf.random.normal(shape)
    sigma_t, t = sigma_0, 1
    while sigma_t > sigma_L and t <= max_iter:
        h = h0 * t / (1.0 + h0 * (t - 1))
        d = denoiser(y, training=False) - y                    # sigma_t^2 * score
        sigma_t = float(tf.norm(d) / tf.sqrt(n))               # noise level from the residual
        gamma = tf.sqrt(tf.maximum((1 - beta * h) ** 2 - (1 - h) ** 2, 0.0)) * sigma_t
        y = y + h * d + gamma * tf.random.normal(tf.shape(y))
        t += 1
    return tf.clip_by_value(y, 0.0, 1.0)
```

### 18.6 Classifier-free guidance (interpolation convention)

```python
def cfg_denoise(denoiser, y, labels, null_token, w=3.0):
    """w=0 unconditional, w=1 plain conditional, w>1 amplified."""
    y2 = tf.concat([y, y], axis=0)
    null = tf.fill(tf.shape(labels), tf.cast(null_token, labels.dtype))
    lab2 = tf.concat([labels, null], axis=0)
    out = denoiser([y2, lab2], training=False)
    d_cond, d_uncond = tf.split(out, 2, axis=0)
    return d_uncond + w * (d_cond - d_uncond)    # weights sum to 1: valid on outputs
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
    """Unweighted (sigma=None) or variance-weighted divergence estimate.

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

### 18.8 Homogeneity-preserving conditioning

```python
def film_scale_only(features, cond_embedding, name=None):
    """Multiplicative (scale-only) FiLM. Preserves degree-1 homogeneity in the
    feature path, because gamma depends on the condition alone, not on y.

    Contrast with additive broadcast injection (features + Dense(emb)), which
    injects a y-independent constant and destroys homogeneity.
    """
    c = features.shape[-1]
    gamma = layers.Dense(c, use_bias=False, name=name)(cond_embedding)
    gamma = layers.Reshape((1, 1, c))(gamma)
    return layers.Multiply()([features, 1.0 + gamma])
```

---

## 19. History and references

### Timeline

| Year | Contribution |
|---|---|
| 1956 | Robbins: empirical Bayes framework; the identity is attributed to Tweedie in this line of work |
| 1961 | Miyasawa: the Gaussian empirical-Bayes estimator identity |
| 1981 | Stein: SURE, unbiased risk estimation for the same model |
| 1982 | Anderson: reverse-time SDEs, later the backbone of score-based generation |
| 2005 | Hyvärinen: score matching for unnormalized models |
| 2011 | Vincent: denoising score matching, connecting DAEs to score estimation |
| 2011 | Raphan and Simoncelli: empirical-Bayes estimation without priors, general noise models |
| 2011 | Efron: Tweedie's formula and selection bias, the modern statistical exposition |
| 2019 | Song and Ermon: annealed Langevin generation from learned scores |
| 2020 | Ho, Jain, Abbeel: DDPM; Mohan et al.: bias-free CNNs |
| 2021 | Song et al.: unified SDE framework; Kadkhodaie and Simoncelli: the implicit prior in a denoiser |
| 2022 | Ho and Salimans: classifier-free guidance |

### Primary references

- Miyasawa, K. (1961). *An empirical Bayes estimator of the mean of a normal population.* Bulletin of the International Statistical Institute, 38(4), 181-188. (Hard to obtain; the identity is most accessibly stated in Efron 2011 and Raphan and Simoncelli 2011.)
- Robbins, H. (1956). *An empirical Bayes approach to statistics.* Proc. 3rd Berkeley Symposium.
- Stein, C. (1981). *Estimation of the mean of a multivariate normal distribution.* Annals of Statistics, 9(6), 1135-1151.
- Efron, B. (2011). *Tweedie's formula and selection bias.* JASA, 106(496), 1602-1614.
- Hyvärinen, A. (2005). *Estimation of non-normalized statistical models by score matching.* JMLR, 6, 695-709.
- Vincent, P. (2011). *A connection between score matching and denoising autoencoders.* Neural Computation, 23(7), 1661-1674.
- Raphan, M., and Simoncelli, E. P. (2011). *Least squares estimation without priors or supervision.* Neural Computation, 23(2), 374-420.
- Anderson, B. D. O. (1982). *Reverse-time diffusion equation models.* Stochastic Processes and their Applications, 12(3), 313-326.
- Eldar, Y. C. (2009). *Generalized SURE for exponential families.* IEEE Transactions on Signal Processing, 57(2), 471-481.
- Foi, A., Trimeche, M., Katkovnik, V., and Egiazarian, K. (2008). *Practical Poissonian-Gaussian noise modeling and fitting for single-image raw data.* IEEE TIP, 17(10), 1737-1754.

### Modern applications

- Song, Y., and Ermon, S. (2019). *Generative modeling by estimating gradients of the data distribution.* NeurIPS.
- Ho, J., Jain, A., and Abbeel, P. (2020). *Denoising diffusion probabilistic models.* NeurIPS.
- Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. (2021). *Score-based generative modeling through stochastic differential equations.* ICLR.
- Mohan, S., Kadkhodaie, Z., Simoncelli, E. P., and Fernandez-Granda, C. (2020). *Robust and interpretable blind image denoising via bias-free convolutional neural networks.* ICLR.
- Kadkhodaie, Z., and Simoncelli, E. P. (2021). *Stochastic solutions for linear inverse problems using the prior implicit in a denoiser.* NeurIPS. arXiv:2007.13640. Reference code: `LabForComputationalVision/universal_inverse_problem`.
- Venkatakrishnan, S. V., Bouman, C. A., and Wohlberg, B. (2013). *Plug-and-play priors for model based reconstruction.* GlobalSIP.
- Romano, Y., Elad, M., and Milanfar, P. (2017). *The little engine that could: regularization by denoising (RED).* SIAM Journal on Imaging Sciences, 10(4), 1804-1844.
- Reehorst, E. T., and Schniter, P. (2019). *Regularization by denoising: clarifications and new interpretations.* IEEE Transactions on Computational Imaging, 5(1), 52-67.
- Ho, J., and Salimans, T. (2022). *Classifier-free diffusion guidance.* arXiv:2207.12598.
- Dhariwal, P., and Nichol, A. (2021). *Diffusion models beat GANs on image synthesis.* NeurIPS. (Classifier guidance.)
- Zhang, L., Rao, A., and Agrawala, M. (2023). *Adding conditional control to text-to-image diffusion models (ControlNet).* ICCV.
- Ke, B., Obukhov, A., Huang, S., Metzger, N., Daudt, R. C., and Schindler, K. (2024). *Repurposing diffusion-based image generators for monocular depth estimation (Marigold).* CVPR.

---

## Appendix A: quick-reference table

| Noise model | Likelihood | Exact identity | Residual is the score? | Score weight |
|---|---|---|---|---|
| Additive AWGN, $y=x+\varepsilon$ | $\mathcal{N}(x,\sigma^2 I)$ | $\hat{x} = y + \sigma^2\nabla\log p(y)$ | Yes | $\sigma^2$ |
| Correlated / transformed, $y=Ax+\varepsilon$ | $\mathcal{N}(Ax,\Sigma)$ | $A\hat{x} = y + \Sigma\nabla\log p(y)$ | Only $A\hat x$; not invertible in general | $\Sigma$ |
| Blur after noise, $y=K(x+\varepsilon)$ | $\mathcal{N}(Kx,\sigma^2KK^{\top})$ | $K\hat{x} = y + \sigma^2 KK^{\top}\nabla\log p(y)$ | Requires $(KK^{\top})^{-1}$ | $\sigma^2KK^{\top}$ |
| Conditional | $\mathcal{N}(x,\sigma^2 I)$, $c \perp \varepsilon$ | $\hat{x}(y,c)=y+\sigma^2\nabla\log p(y\mid c)$ | Yes | $\sigma^2$ |
| Multiplicative, $y=xn$ | $\mathcal{N}(x,\sigma^2x^2)$ | (A): needs $\mathbb{E}[x^2\mid y]$ | **No** | $\sigma^2y^2$ (approx.) |
| Composite, $y=xn+a$ | $\mathcal{N}(x,\sigma_m^2x^2+\sigma_a^2)$ | (A$_c$): additive term plus second-moment term | Only if $\sigma_m=0$ | $\sigma_a^2+\sigma_m^2y^2$ (approx.) |
| VP / DDPM | $\mathcal{N}(\sqrt{\bar\alpha}x_0,(1-\bar\alpha)I)$ | $\mathbb{E}[x_0\mid x_t]=\frac{x_t+(1-\bar\alpha)\nabla\log p_t}{\sqrt{\bar\alpha}}$ | Yes, after rescaling | $1-\bar\alpha$ |

| Property | Additive | Multiplicative | Composite |
|---|---|---|---|
| MSE training yields $\mathbb{E}[x\mid y]$ | Yes | Yes | Yes |
| Residual-equals-score identity | Yes | No | No ($\sigma_m>0$) |
| Optimal estimator scale-equivariant | Yes, if $\sigma$ co-scales | Yes, at fixed $\sigma_m$ | Yes, if $\sigma_a$ co-scales |
| Bias-free network compatible | Yes | Yes | Yes |
| Reference-free audit | SURE (exact) | gSURE (approximate) | gSURE (approximate) |

---

## Appendix B: conventions that are commonly mixed up

1. **Score versus noise prediction.** $s = -\hat{\varepsilon}/\sigma^2$ if the network predicts the noise
   *vector*; $s = -\varepsilon_\theta/\sigma$ if it predicts *unit-variance* noise.
2. **Langevin step size.** $y \mathrel{+}= \tfrac{\eta}{2}s + \sqrt{\eta}z$ and $y \mathrel{+}= \eta s + \sqrt{2\eta}z$ are the same up to $\eta \to 2\eta$.
3. **CFG scale.** Interpolation convention ($w=1$ is plain conditional) versus Ho and Salimans ($w=0$ is plain conditional).
4. **Domain shift versus rescale.** $[0,1]$ and $[-0.5,+0.5]$ differ by a shift only. Do **not** rescale $\sigma$, `max_val`, or PSNR constants. $[-1,+1]$ has width 2, so moving there **does** require $\sigma \to 2\sigma$ and `max_val = 2.0`.
5. **`center=False` in Keras BatchNormalization** removes the learned offset $\beta$. It does not remove mean subtraction; mean subtraction is linear and does not break homogeneity anyway.
6. **Variance versus standard deviation.** The score weight is $\sigma^2$, not $\sigma$. Half the sign and scale bugs in this area are this.

---
