# Miyasawa-Compliant Linear Attention: A Bias-Free, Degree-1-Homogeneous O(N) Attention Layer

**Author:** Nikolas Markou · **Date:** 2026-07-07 · **Status:** Design/research note (self-contained)

> **One-sentence thesis.** A normalized linear-attention layer with a bias-free,
> positively-homogeneous, non-negative feature map is *provably* degree-1 positively
> homogeneous and structurally bias-free, so — unlike softmax attention — it can sit
> inside a bias-free denoiser without breaking the additive-Gaussian
> `residual = sigma^2 * score` (Miyasawa/Tweedie) identity.

This note documents the design of `LinearAttention`
(`src/dl_techniques/layers/attention/linear_attention.py`), the first attention layer
in this repo built to satisfy the two operational Miyasawa properties this codebase
actually checks. It (1) fixes what "Miyasawa-compliant" means *here*, (2) gives the
degree-1 homogeneity proof, (3) analyzes which feature maps qualify and which are
forbidden, (4) documents the one genuine design tension — the denominator epsilon —
and how it is resolved exactly, (5) relates the layer to the existing
`PerformerAttention`, (6) records the rejected wrapper alternative, (7) places the
whole thing inside the attention-as-denoiser theory, and (8) gives the integration
path into the bfunet-family denoiser stack.

---

## Table of Contents

1. [The 90-second version](#1-the-90-second-version)
2. [What "Miyasawa-compliant" means in this repo](#2-what-miyasawa-compliant-means-in-this-repo)
3. [The core result: normalized linear attention is degree-1](#3-the-core-result-normalized-linear-attention-is-degree-1)
4. [Feature-map analysis: what qualifies, what is forbidden](#4-feature-map-analysis-what-qualifies-what-is-forbidden)
5. [The epsilon tension and its exact resolution (D-001)](#5-the-epsilon-tension-and-its-exact-resolution-d-001)
6. [Relation to the existing PerformerAttention (D-003)](#6-relation-to-the-existing-performerattention-d-003)
7. [Rejected alternative: the normalization-equivariance wrapper (D-002)](#7-rejected-alternative-the-normalization-equivariance-wrapper-d-002)
8. [Theory context: attention IS a denoising step](#8-theory-context-attention-is-a-denoising-step)
9. [Integration path into the bias-free denoiser stack](#9-integration-path-into-the-bias-free-denoiser-stack)
10. [Scope and future work](#10-scope-and-future-work)
11. [Sources](#11-sources)

---

## 1. The 90-second version

Standard attention is `softmax(Q Kᵀ / √d) V`. It is `O(N²)` in sequence length *and*
it is **not** degree-1 homogeneous: softmax is temperature-sensitive
(`softmax(alpha·z) != softmax(z)` for `alpha != 1`), so scaling the input changes the
attention pattern, not just the output magnitude. Inside a bias-free denoiser that
relies on `f(alpha·x) = alpha·f(x)`, softmax attention silently breaks the property the
whole architecture is built to preserve.

Linear attention replaces the softmax exponential kernel with an explicit feature map
`phi` and exploits matmul associativity:

```
O_i = phi(Q_i) · (Σ_j phi(K_j) ⊗ V_j)  /  (phi(Q_i) · Σ_j phi(K_j) + eps_eff)
      └──────── numerator (d×d state) ────────┘   └──── scalar denominator ────┘
```

Contract `Σ_j phi(K_j) ⊗ V_j` once into a `(d, d)` state, then multiply by `phi(Q_i)`:
the `N × N` matrix is never formed, giving `O(N·d²) = O(N)` in sequence length (F-W1).

The key result (Section 3): **with no bias and a positively-homogeneous `phi`, the
normalized form is exactly degree-1 homogeneous — for any degree `p` of `phi`.** The
normalizer is load-bearing: it is what cancels the excess degree. Combined with
`use_bias=False` on every projection, that gives both operational Miyasawa properties
this repo checks: **bias-free** (structural) and **degree-1 homogeneous** (numeric).

The one wart is the denominator epsilon that guards the `0/0` dead-token case. A fixed
additive epsilon breaks *exact* degree-1 (the same class of problem as the LayerNorm /
RMSNorm epsilon already documented in this repo's LESSONS). We resolve it with an
**input-scaled epsilon** that matches the denominator's own degree, restoring exact
degree-1 (Section 5). The implemented layer measures homogeneity rel-err = **0.0**
(exact, fp32) across all `{relu, relu_squared, abs} × {alpha=0.5, alpha=2.0}` in its
37-test suite.

---

## 2. What "Miyasawa-compliant" means in this repo

"Obeys Miyasawa's properties" is *not* a vague theoretical nod here. It is
operationalized as two independently-checked properties of the forward map `f`:

1. **Bias-free (structural).** No additive constant anywhere on the forward path: every
   `Conv`/`Dense` has `use_bias=False`, no norm carries a `beta`/`center=True` term, no
   additive const of any kind. This is the litmus in
   `src/dl_techniques/layers/match_channels.py:26-41` — an op is bias-free /
   homogeneity-preserving iff `op(alpha·x) = alpha·op(x)` exactly.

2. **Degree-1 positive homogeneity (numeric).** `f(alpha·x) = alpha·f(x)` for scalar
   `alpha > 0`, verified by a black-box probe. The repo's canonical probe is
   `src/train/bfunet/common.py:_homogeneity_probe` (`:920-984`): it draws
   `x = uniform(-0.5, 0.5)`, forwards `f(x)` and `f(alpha·x)` for `alpha ∈ {0.5, 2.0}`
   at `training=False`, and computes
   `rel = max|f(alpha·x) - alpha·f(x)| / max(|alpha·f(x)|, 1e-8)`, tolerance `1e-2`
   (fp32). The probe is **informational-only** (it logs, never raises) but is the
   *trustworthy* Miyasawa signal — more so than any static `use_bias` scan
   (LESSONS.md:37).

**Scope of the clean identity.** Only **additive-Gaussian-noise** denoisers get the
closed-form Miyasawa/Tweedie identity `E[x|y] = y + sigma^2 · ∇_y log p(y)`, i.e. the
reading `residual = sigma^2 * score`. Signed/per-pixel *multiplicative* or composite
noise has **no** such clean linear-domain identity
(`src/dl_techniques/utils/multiplicative_miyasawa.py:12-33`). Everything below is in the
additive-Gaussian regime.

**Why the usual attention ingredients break degree-1.** These are all forbidden on a
compliant path (per `miyasawa-constraints.md`, F2):

| Ingredient | Bias-free? | Degree-1? | Why it breaks |
|---|---|---|---|
| **Softmax** | n/a | **NO** | Temperature-sensitive: `softmax(alpha·z) != softmax(z)`; sharpens toward one-hot as `alpha` grows, flattens as it shrinks. Attention weights become a non-homogeneous function of `Q,K`. |
| **LayerNorm / RMSNorm** (even `center=False`) | yes | **NO** | Per-input normalization is scale-**invariant** (degree-0), not degree-1. Dividing by a per-input statistic cancels the scale entirely. |
| **GELU / sigmoid / tanh / Swish** | yes | **NO** | Saturating nonlinearities are not homogeneous for any nonzero scale. |
| **`elu(x)+1`** (Katharopoulos map) | yes | **NO** | The `+1` is an additive degree-0 constant → `phi(alpha·z) != alpha^p·phi(z)`. |
| **Multiplicative gate** `sigmoid(Dense(x))·h` | yes | **NO** | The gate is a nonlinear function of `x`; the product is degree-2, not degree-1 (the Clifford GGR-gate precedent, `clifford_block.py:583-590`). |

Softmax is structurally the same shape as the rejected Clifford gate: a nonlinear,
input-dependent reweighting (`softmax(QKᵀ)`) multiplied against a degree-1 stream (`V`)
yields something neither invariant nor degree-1. That is the whole reason to drop
softmax and use kernel/linear attention.

---

## 3. The core result: normalized linear attention is degree-1

**Claim (F-W2).** With no bias in the Q/K/V/output projections and a feature map `phi`
that is positively homogeneous of degree `p` (`phi(c·z) = c^p·phi(z)` for `c > 0`), the
normalized linear-attention output is **exactly degree-1 homogeneous in the input, for
any `p`.**

**Proof.** Let the input tokens scale `X → cX` with `c > 0`. Because the projections are
bias-free (pure linear maps), `Q → cQ`, `K → cK`, `V → cV`. Then term by term:

```
numerator    Σ_j phi(cQ_i) · phi(cK_j) · (cV_j)
           = c^p · c^p · c · Num
           = c^(2p+1) · Num

denominator  Σ_j phi(cQ_i) · phi(cK_j)
           = c^p · c^p · Den
           = c^(2p) · Den

O_i(cX) = c^(2p+1) · Num  /  c^(2p) · Den
        = c · (Num / Den)
        = c · O_i(X)                       →  degree-1, for ANY p.  ∎
```

**The normalizer is load-bearing, not optional.** An *unnormalized* linear attention
`Σ_j phi(Q_i) phi(K_j) V_j` is degree `2p+1`, i.e. degree-1 only in the trivial `p=0`
case. It is the division by the denominator that pulls the degree from `2p+1` back down
to exactly `1`. This is why the layer treats the denominator as **mandatory** — it is
load-bearing for *homogeneity*, not merely for numerical stability, and is never exposed
as a flag that can be silently switched off into a non-homogeneous mode.

The implemented `call()` (`linear_attention.py:361-390`) computes exactly this, with the
degree of every intermediate annotated in-line:

```
kv    = einsum('bhnd,bhne->bhde', phi_k, v)   # (B,H,d,d),  degree p+1
k_sum = sum(phi_k, axis=2)                     # (B,H,d),    degree p
num   = einsum('bhnd,bhde->bhne', phi_q, kv)   # (B,H,N,d),  degree 2p+1
z     = einsum('bhnd,bhd->bhn',  phi_q, k_sum) # (B,H,N),    degree 2p, >= 0
out   = num / (z + eps_eff)                     # (B,H,N,d),  degree 1
```

The projections are bias-free by default (`use_bias=False`), so the only nonlinearity on
the path is `phi`. There is no softmax, no LayerNorm/RMSNorm, no per-token normalization,
no saturating activation, and no multiplicative gate.

---

## 4. Feature-map analysis: what qualifies, what is forbidden

A feature map `phi` is admissible iff it is **positively homogeneous of some degree `p`**
(`phi(alpha·z) = alpha^p·phi(z)` for `alpha > 0`) **and non-negative** (`phi(z) >= 0`, so
the denominator `phi(Q)·Σphi(K)` stays non-negative and the sign of the ratio is
well-behaved). The proof in Section 3 holds for *any* `p`, so the degree is free to
choose for expressiveness — it does not affect compliance.

**Admissible (implemented, `_SUPPORTED_FEATURE_MAPS`):**

| `feature_map` | `phi(x)` | degree `p` | note |
|---|---|---|---|
| `'relu'` (default) | `relu(x)` | 1 | `relu(alpha·x) = alpha·relu(x)` for `alpha>0`; the canonical positively-homogeneous piecewise-linear map. |
| `'relu_squared'` | `relu(x)²` | 2 | FLatten/Focused-style focus: sharpens the (otherwise high-entropy) linear-attention distribution while staying homogeneous — `(alpha·relu(x))² = alpha²·relu(x)²`. |
| `'abs'` | `|x|` | 1 | `|alpha·x| = alpha·|x|` for `alpha>0`; non-negative, keeps sign magnitude. |
| `'leaky_relu'` | `x if x>0 else α·x` | 1 | `leaky(alpha·x) = alpha·leaky(x)` for `alpha>0` → homogeneous. **SIGNED** (the only supported map that is not `>= 0`): removes the dead-ReLU zero-gradient half (nonzero slope `α` on the negative side, `negative_slope` default `0.01`) at the cost of the non-negative-kernel guarantee. See the signed-map note below. |

**The signed-map caveat (`'leaky_relu'`).** `relu`/`relu_squared`/`abs` are non-negative, so
the denominator `z = phi(Q)·Σ phi(K)` is a non-negative normalizer and the per-token
"attention weights" form a non-negative partition. `leaky_relu` is still degree-1
homogeneous (so it *preserves* the Miyasawa property — measured worst-case rel-err ~`1e-6`
over `α∈{1e-3..1e3}` and slopes `{0.01,0.1,0.3}`), but it produces **negative** features, so
`z` can be small or negative and the weights are no longer a convex partition. The
denominator is therefore floored by a **sign-aware magnitude floor**
`where(z≥0, max(z,1e-20), min(z,-1e-20))` — identical to the plain `max(z,1e-20)` for the
non-negative maps (so their exact behavior is unchanged), but for `leaky_relu` a genuinely
negative `z` keeps its sign+magnitude instead of being clamped to `+1e-20` (which would flip
the output sign and blow it up). Homogeneity stays exact wherever `|z| > 1e-20`. Use a small
`negative_slope` and keep `epsilon` at default; it is the right choice when dead-gradient
stalls appear (a projected feature stuck on the ReLU-negative side gets zero gradient with
`relu`/`relu_squared`, but slope `α` with `leaky_relu`).

**Forbidden (rejected in `__init__`, `_FORBIDDEN_FEATURE_MAPS`):**

- **`'elu_plus_one'`** — Katharopoulos's `elu(x)+1`. The `+1` is an *additive degree-0
  constant*: `phi(alpha·x) != alpha^p·phi(x)`. Even though it is the most common linear-
  attention feature map in the literature, it directly breaks `f(alpha·x) = alpha·f(x)`.
- **`'exp'` / `'softmax'`** — the exponential kernel is non-homogeneous
  (`exp(alpha·z) != alpha^p·exp(z)`) and softmax is temperature-sensitive
  (`softmax(alpha·z) != softmax(z)`). This is exactly what we set out to avoid.

**Homogeneity-preserving expressiveness boosters (in-family, future work).** These stay
compliant because they are *multiplicative* by input-independent factors, or raise degree
without adding a constant — none of them add a degree-0 term (F-W4):

- **ReLU^p focus (FLatten/Focused):** already available as `relu_squared`; general `p` is
  a trivial extension, all positively homogeneous of degree `p`.
- **cosFormer cosine reweighting (arxiv 2202.08791):** decomposes a positional
  `cos((i-j)π/2M)` locality bias into per-token multiplicative factors on `Q`/`K`. These
  are input-independent scalars → homogeneity-preserving. Gives locality without softmax.
- **PolaFormer polarity split (arxiv 2501.15061):** processes positive and negative parts
  of the features separately to retain sign information ReLU discards; each branch must be
  independently checked to stay bias-free/homogeneous.

**One booster that is NOT free.** **Norm×Direction** (arxiv 2506.21137) restores the
query-norm signal by multiplying by `‖Q‖` (degree-1). That *changes the degree ledger*
(it is not an input-independent constant), so it must be re-accounted in the homogeneity
proof before use — flagged, deferred.

---

## 5. The epsilon tension and its exact resolution (D-001)

**The tension.** ReLU (and any non-negative `phi`) has a dead-token pathology: if a
token's `phi(Q_i)` or all `phi(K_j)` are zero, the numerator and denominator are both
`0` → `0/0` NaN (F-W3). The standard fix is a denominator floor `Den + eps`. **But a
fixed additive `eps` is a degree-0 constant that breaks *exact* degree-1** — the exact
same class of wart as the RMS-eps / LayerNorm-eps issue already in this repo's LESSONS.

Concretely: `num` is degree `2p+1` and `z` is degree `2p`. A *fixed* constant added to
`z` does not scale with `z` under `X → cX`, so `num / (z + eps)` stops being degree-1.
Performer's bare `+1e-6` (Section 6) is exactly this failure mode.

**The resolution (decisions.md D-001): input-scaled epsilon.** Scale `epsilon` by a
degree-`2p` quantity derived from the denominator's own scale, so the floor matches `z`'s
degree. The implementation uses the per-`(B,H)` mean of `z` over tokens
(`linear_attention.py:385-390`, carrying the `# DECISION plan_2026-07-07_1cab8d7a/D-001`
anchor):

```
z_mean  = mean_j(z)          # (B,H,1), degree 2p  (same degree as z)
eps_eff = epsilon * z_mean   # degree 2p
denom   = z + eps_eff        # degree 2p  →  num/denom stays EXACTLY degree-1
denom   = maximum(denom, 1e-20)   # NaN guard, all-dead batch ONLY
```

Because `eps_eff` has the same degree as `z`, `num / (z + eps_eff)` scales as
`c^(2p+1)/c^(2p) = c` — exactly degree-1 preserved. The `maximum(·, 1e-20)` is a pure
NaN floor that is active *only* on a fully-dead batch (all `phi` zero → `z_mean == 0`);
it is a negligible degree-0 constant and is the single residual non-homogeneous corner,
absorbed by the probe tolerance.

**Empirical result.** The 37-test suite
(`tests/test_layers/test_attention/test_linear_attention.py`) measures the homogeneity
rel-err as **0.0 (exact, fp32)** for all `{relu, relu_squared, abs} × {alpha=0.5,
alpha=2.0}` — not merely under the `1e-2` image-probe tolerance, not merely under the
tighter `1e-4` seq-probe gate, but bit-exact. The associativity path also matches a naive
`O(N²)` reference within `atol=1e-5` (with TF32 disabled test-side). The dead-token and
all-zero-input edge cases return finite outputs.

**Rejected eps alternatives (D-001):** (a) a *fixed additive* eps like Performer's —
simplest, but breaks exact degree-1 and only passes at the loose image-probe tol;
(b) *strictly-positive-only `phi` with no eps* — cannot guarantee finiteness on genuinely
dead tokens.

---

## 6. Relation to the existing PerformerAttention (D-003)

`PerformerAttention` (`src/dl_techniques/layers/attention/performer_attention.py`) is the
repo's only other true `O(N)` attention layer, and it is bias-free by default with no
LayerNorm. `LinearAttention` **reuses** one thing from it and **rejects** two.

**Reused — the non-causal associativity contraction.** Performer's `_linear_attention`
non-causal path (`performer_attention.py:301-361`) contracts `phi(K)ᵀ V` first
(`bhnf,bhnd->bhfd`) into a `(d,d)` state, then multiplies by `phi(Q)`, never materializing
the `N × N` matrix. `LinearAttention` uses the same contraction *style* (with `einsum`
signatures adapted to its head layout).

**Rejected #1 — Performer's FAVOR+ feature map.** Performer's map is
`phi(x) = cos/sin(w·x) · exp(-‖x‖²/2)` (`performer_attention.py:257-299`). The Gaussian
factor `exp(-‖x‖²/2)` is **not positively homogeneous** — it is a non-homogeneous
exponential of the input norm — so it breaks degree-1 by construction. `LinearAttention`
supplies a fresh bias-free positively-homogeneous `phi` (ReLU / ReLU² / abs) instead.

**Rejected #2 — Performer's bare `+1e-6` epsilon.** Performer adds a fixed additive
denominator floor (`z = z + 1e-6`, `performer_attention.py:354`). As shown in Section 5
that is a degree-0 additive constant that breaks *exact* degree-1. `LinearAttention` uses
the input-scaled `eps_eff` of Section 5 instead.

**Rejected #3 — the causal cumsum path.** Performer also carries a causal-cumsum variant
(`performer_attention.py:324-340`). `LinearAttention` v1 is **non-causal only** (denoising
is non-causal), so that path is out of scope (Section 10).

This is decisions.md **D-003**: reuse only the non-causal associativity computation;
supply a fresh homogeneous `phi`; scope v1 non-causal.

---

## 7. Rejected alternative: the normalization-equivariance wrapper (D-002)

There is a second, entirely different way to get scale-equivariance: instead of
constraining the internal layers, **wrap** any backbone (softmax attention allowed!) as

```
f(y) = std(y) · g((y - mu(y)) / std(y)) + mu(y)
```

"Normalization Equivariance for Arbitrary Backbones" (arxiv 2605.08193) shows this gives
`f(a·y + b) = a·f(y) + b` for `a > 0` — affine equivariance at the whole-network I/O
level, which is even *stronger* than degree-1 (F-W6).

**Why it was rejected as the primary approach (decisions.md D-002).** The equivariance is
only at the network **I/O boundary**; *internally* the softmax is still non-homogeneous.
So the wrapper cannot serve as a drop-in homogeneous *block* inside the bias-free denoiser
stack — which was the actual ask ("a purely linear attention **layer** that obeys
Miyasawa"). A wrapped-softmax attention block placed mid-stack still breaks degree-1 for
every downstream layer that reads its output.

It is documented here as a **pragmatic complement / escape hatch**: if a layer-level
homogeneous linear attention ever underperforms on a given task, the wrapper is the way to
retrofit affine-equivariance around an arbitrary (even softmax) backbone at the I/O level.
It is not the deliverable.

---

## 8. Theory context: attention IS a denoising step

A Miyasawa-compliant attention denoiser is not a forced marriage — several independent
lines place attention squarely inside denoising/score theory (F-W7):

- **Attention = one energy gradient step.** Self-attention is one gradient step on a
  modern-Hopfield / Dense-Associative-Memory (DAM) energy landscape (Ramsauer et al.).
- **In-context denoising (arxiv 2502.05164).** A trained one-layer transformer performs
  one denoising gradient-descent step on a DAM energy landscape — attention literally *is*
  a denoising operation.
- **Softmax-free homogeneous denoising (arxiv 2508.02967).** "Towards Robust Image
  Denoising with Scale Equivariance" builds a first-order-homogeneous denoiser that
  **explicitly avoids softmax attention**, replacing it with homogeneous
  ratio-of-equal-degree gating — e.g. their Interactive Gating Module
  `IG(Fv, Fm) = (Fv ⊙ Fm) / √(σ²(Fv) + σ²(Fm))`: numerator degree-2 over denominator
  degree-2 = degree-1 quotient. This is direct prior-art evidence that a homogeneous
  content-mixing mechanism is buildable via *equal-degree ratios* — structurally the same
  trick `LinearAttention` uses (`num` degree `2p+1` over `denom` degree `2p`), and it
  confirms softmax is the thing to drop.

So an attention layer whose residual reads as `sigma² · score` sits inside existing
theory: `LinearAttention` makes that reading *structurally available* by being degree-1
homogeneous and bias-free, which softmax attention never is.

---

## 9. Integration path into the bias-free denoiser stack

`LinearAttention` is factory-registered (`create_attention_layer('linear', dim=...,
num_heads=...)`, `attention/factory.py`) and exported from
`attention/__init__.py`, so it drops into any config-driven block the way `performer`
does. To use it in the bfunet-family bias-free denoisers:

1. **Keep the compliant defaults.** `use_bias=False` (bias-free) and a supported
   `feature_map` (`relu` / `relu_squared` / `abs`). Do **not** set `use_bias=True` (that
   is only for non-Miyasawa callers) and do **not** wrap it in a LayerNorm/RMSNorm — those
   re-break degree-1 regardless of the layer's own compliance. If a norm is genuinely
   needed alongside it, use `BiasFreeBatchNorm` (variance-only, the only repo norm that is
   both bias-free and degree-1 at inference), never stock LayerNorm/BatchNorm.

2. **You get the homogeneity probe for free at train time** — *with one caveat*. Any
   denoiser trained through `src/train/bfunet/common.py`'s `train()` is checked by
   `_homogeneity_probe` at post-build/post-init checkpoints. **But that probe synthesizes
   a 4D image-shaped input** `(B, H, W, C)`, whereas `LinearAttention` consumes a
   3D token/sequence tensor `(B, N, dim)`. So either:
   - place `LinearAttention` where the denoiser already exposes a **token/sequence**
     structure (e.g. a tokenized bottleneck), and adapt the probe input shape accordingly; or
   - add a small `(B,H,W,C) ↔ (B, H·W, C)` reshape adapter around the attention block so
     the image-shaped probe flows through unchanged. The layer's own math is spatial-shape
     agnostic — it only needs a sequence axis.
   The standalone seq-shaped probe in the test suite is the reference for the 3D case
   (the plan deliberately did **not** reuse `common.py:_homogeneity_probe` for the unit
   test precisely because it is 4D image-shaped).

3. **Remember the caveat about untrained models.** A homogeneity PASS on an *untrained*
   model does not prove homogeneity for residual-branch architectures (`LayerScale
   gamma_init=1e-5` masks in-block breaks until training grows gamma,
   `common.py:934-941`). `LinearAttention` is homogeneous by *construction* (verified
   analytically + exact numeric), so it does not depend on this — but any surrounding
   block still must be probed **post-training**.

4. **Stay in the additive-Gaussian regime.** The `residual = sigma² · score` reading only
   holds for additive Gaussian noise; multiplicative/composite noise has no such identity
   (Section 2). Compliance of the *layer* does not extend the *identity* to other noise
   models.

---

## 10. Scope and future work

**v1 scope (shipped).** Non-causal, unmasked, multi-head linear self-attention;
`feature_map ∈ {relu, relu_squared, abs}`; input-scaled epsilon; bias-free by default;
full `get_config` round-trip + `.keras` save/load; factory-registered as `'linear'`.

**Known limitations (honest caveats).** The degree-1 claim is scoped, not unconditional:

- **`relu_squared` at extreme scale.** Degree-1 is *exact* for `relu` / `abs` (degree
  `p=1`) across a wide input-scale band, but `relu_squared` (`p=2`, degree-4 denominator)
  degrades at extreme *small* scales (`alpha ≲ 1e-6`): the doubled dynamic range
  (`num ~ alpha^5` vs `denom ~ alpha^4`) underflows in fp32 and the `1e-20` floor
  activates, so the property no longer holds bit-exactly there (reviewer probe: rel-err
  ~4e-4 at `alpha ∈ {1e-3, 1e3}`, breaking down at `alpha = 1e-6`). **Use `relu` for the
  strongest guarantee**; keep `relu_squared` to realistic scales.
- **`mask=` is ignored.** v1 is non-causal and unmasked; `call()` does `del mask`. Padded
  tokens still contribute to `kv` and the normalizer, so masked/padded sequences are
  silently wrong — do not swap `'linear'` for a mask-honoring attention type on padded data.
- **Homogeneity is a `training=False` / `dropout_rate=0` property.** Dropout is applied
  after `output_proj`; with `dropout_rate>0` at `training=True` the output is stochastic and
  not per-sample homogeneous. The default `dropout_rate=0.0` is the Miyasawa mode.
- **Mixed precision.** The dead-token guard now runs the denominator divide + `1e-20` floor
  in float32 and casts back to the compute dtype (`linear_attention.py`, D-001), so an
  all-zero batch stays finite under a `mixed_float16` policy (the fp16 floor would otherwise
  round to 0.0 → `0/0` NaN).

**Explicitly out of v1 (future work), to hold the complexity budget:**

- **Causal cumsum variant** — a per-position prefix-state (`cumsum`) path for
  autoregressive sequence modeling, as Performer has. Denoising is non-causal, so this was
  pure budget cost with no immediate payoff.
- **PolaFormer polarity split** (arxiv 2501.15061) — sign-preserving positive/negative
  branch processing; each branch must be re-checked for compliance.
- **cosFormer cosine reweighting** (arxiv 2202.08791) — multiplicative locality bias;
  homogeneity-preserving, deferred for scope.
- **Norm×Direction query-norm restoration** (arxiv 2506.21137) — changes the degree
  ledger (multiplies by `‖Q‖`); needs a fresh homogeneity accounting before use.
- **A spatial/2D attention adapter for image denoisers** — a first-class
  `(B,H,W,C) ↔ (B, H·W, C)` wrapper so the layer plugs into the image-shaped bfunet probe
  and pipeline without a hand-written reshape at each call site (Section 9).

---

## 11. Sources

External literature grounding the design (from
`plans/plan_2026-07-07_1cab8d7a/findings/web-research-linear-attention.md`):

- Katharopoulos et al. (2020), *Transformers are RNNs: Fast Autoregressive Transformers
  with Linear Attention* — kernel feature map + associativity; the `elu(x)+1` map (used as
  the FORBIDDEN example here).
- **cosFormer**, arxiv 2202.08791 — ReLU feature map + cosine positional reweighting
  (multiplicative, homogeneity-preserving).
- **FLatten / Focused Linear Attention** — `ReLU(x)^p` focus function (the basis for the
  `relu_squared` map).
- **PolaFormer**, arxiv 2501.15061 — polarity-aware linear attention.
- **Norm×Direction**, arxiv 2506.21137 — query-norm restoration (changes the degree
  ledger).
- **Towards Robust Image Denoising with Scale Equivariance**, arxiv 2508.02967 —
  CS/NSM/IGM softmax-free first-order-homogeneous denoiser via ratio-of-equal-degree
  gating.
- **Normalization Equivariance for Arbitrary Backbones**, arxiv 2605.08193 — the
  normalize→backbone→denormalize wrapper (the rejected/complementary alternative, D-002).
- **In-context denoising with one-layer transformers**, arxiv 2502.05164 — a trained
  attention layer performs one denoising gradient step on a DAM energy landscape.
- **Efficient-architectures survey**, arxiv 2508.09834 — linear/efficient-attention
  landscape.

Repo-internal references:

- `src/dl_techniques/layers/attention/linear_attention.py` — the implemented layer.
- `src/dl_techniques/layers/attention/performer_attention.py` — associativity reused;
  FAVOR+ map + bare `+1e-6` rejected.
- `src/train/bfunet/common.py:_homogeneity_probe` (`:920-984`) — the numeric Miyasawa
  probe.
- `src/dl_techniques/layers/norms/bias_free_batch_norm.py` — the only bias-free +
  degree-1 (inference) norm.
- `src/dl_techniques/utils/multiplicative_miyasawa.py:12-33` — additive-Gaussian-only
  scope of the clean identity.
- `plans/plan_2026-07-07_1cab8d7a/decisions.md` — D-001 (input-scaled eps), D-002 (layer
  vs wrapper), D-003 (reuse Performer associativity only).
