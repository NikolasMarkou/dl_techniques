# Spectral Analysis Cheat Sheet — Theory & Interpretation

A text-only companion to `README.md` (the operational reference), `SETOL.md` (the full
theory) and `CORRELATION_TRAPS.md` (trap theory). This file exists so a reader can explain
**why** the spectral metrics are what they are — not just what the columns are called.
Where the shipped code and the SETOL paper's terminology differ, both are given and the
difference is named.

---

## 1. The Big Picture: Why Spectral Analysis Works

### 1.1 Two Levels of Theory

| Level | Claim | Nature |
|-------|-------|--------|
| **HTSR** (Heavy-Tailed Self-Regularization) | "Good models have heavy-tailed weight spectra, and the tail's power-law exponent predicts quality" | Empirical — observed across thousands of pretrained models |
| **SETOL** (Semi-Empirical Theory of Learning) | "Here is *why* heavy tails emerge, how to measure them from first principles, and when the theory breaks down" | Derived — Statistical Mechanics + Random Matrix Theory + Wilson's Renormalization Group |

The analogy is to the **semi-empirical mass formula** in nuclear physics and the **PPP
method** in quantum chemistry: a theoretical framework whose parameters are taken from
real measurements, yielding predictive power that neither pure theory nor pure
measurement achieves alone. In SETOL, the fitted α enters as a renormalized empirical
parameter, exactly like a PPP parameter.

### 1.2 What Is Actually Measured

For one layer with weight matrix **W** (N × M):

1. **Correlation matrix**: X = (1/N) WᵀW — the M × M matrix whose spectrum carries the
   information (implementations may omit the 1/N; see §9 Normalization).
2. **Eigenvalues** λ₁ ≥ λ₂ ≥ … ≥ λ_M of X, computed as eigenvalues from singular values
   via λ = σ² (one SVD per layer).
3. **Empirical Spectral Density (ESD)**: ρ(λ) = Σ δ(λ − λᵢ).
4. **Tail fit**: the part of the ESD above an automatically chosen threshold λ₀ (called
   `xmin` in the DataFrame) is fit to a power law ρ(λ) ∼ λ^(−α), with α and λ₀
   *jointly* estimated by the Clauset-Shalizi-Newman MLE / KS-distance method.

**The entire construction rests on the TAIL of the eigenvalue distribution.** The bulk
says "this layer is a random matrix"; the tail says "this layer has learned structure,
and its exponent says how much."

### 1.3 Why the Tail Encodes Learning

Training pushes correlations from the data into W. Each learned direction of variation
contributes an eigenvalue above the random-matrix bulk. As training proceeds:

- The bulk stays near the Marchenko-Pastur (MP) shape of a random matrix of the same size.
- The top eigenvalues detach, grow, and eventually the whole upper spectrum organizes
  into a power law — the signature of *scale-free* learned structure.
- A well-trained layer's tail sits exactly at the critical boundary α ≈ 2 (§3), which
  SETOL derives rather than assumes.

---

## 2. The Universality Classes (Training Phases) — TWICE

There are two phase tables in this package and they differ by terminology. Know both.

### 2.1 The HTSR "5+1" Universality Classes (SETOL paper's table)

| Class | α range | Interpretation |
|-------|---------|----------------|
| RandomLike | n/a | Untrained / random init; best fit is Marchenko-Pastur |
| Bulk+Spikes | n/a | Early training; a few learned correlations as spikes over an MP bulk |
| Weakly Heavy-Tailed | α > 6 | Under-trained |
| Heavy (Fat)-Tailed | 2 < α < 6 | Standard SOTA; well-trained |
| Very Heavy-Tailed (VHT) | 1 < α < 2 | Over-regularized / overfit |
| Rank Collapse | n/a | Degenerate; effective rank lost |

SETOL's own preferred name for α < 2 is **"over-regularized"**; its papers describe the
α ≈ 2 strip as **"ideal"** and sub-divide the 2–6 range further. Those sub-labels are
NOT used by the shipped code (see 2.2).

### 2.2 The Shipped Labels (`classify_learning_phase` — WeightWatcher authoritative)

The code returns exactly **four** labels; decision D-009 pinned WeightWatcher's labels
and explicitly forbids re-adding SETOL's sub-bands (`ideal`, `fair`, `over-regularized`):

| α | Label | Reading |
|---|-------|---------|
| < 0 | `failed` | No valid fit — sentinel value; use stable_rank/entropy instead |
| [0, 2.0) | `over-trained` | Very heavy-tailed; infinite ESD variance; check for traps |
| [2.0, 6.0] | `good` | Heavy-tailed, normal SOTA range; **contains** the α≈2 critical point — deliberately not subdivided |
| > 6.0 | `under-trained` | Nearly random; check the layer receives gradients |

If explaining to someone who read the SETOL paper: code-label `over-trained` ≡ paper-term
"over-regularized / VHT", and code-label `good` covers both "ideal (α≈2)" and "fat-tailed".

### 2.3 Why α = 2 Is the Critical Value

For a Pareto tail ρ(λ) ∼ λ^(−α):

- The mean exists only for α > 1.
- **The variance exists only for α > 2.**

So α = 2 is not a chosen threshold — it is the point where the second moment of the ESD
diverges. Below it the distribution is in a genuinely different (atypical) universality
class; SETOL additionally derives that α ≈ 2 is where the ERG condition holds (§5), i.e.
two independent conditions meet at the same point. That coincidence is the strongest
piece of evidence that α = 2 is a true phase boundary, analogous to a critical exponent.

---

## 3. The Three Conditions of Ideal Learning (SETOL)

SETOL defines the Ideal State of Learning as three conditions holding at once:

1. **HTSR condition**: the ESD tail fits ρ(λ) ∼ λ^(−2).
2. **ERG condition**: the eigenvalues of the correlation matrix restricted to the
   Effective Correlation Space (ECS) satisfy
   Σᵢ ln λ̃ᵢ = ln det(X̃) ≈ 0.
   This is exactly one step of **Wilson's Exact Renormalization Group**: a
   scale-invariant, volume-preserving transformation (a product of eigenvalues ≈ 1 is a
   volume in log-space preserved by the rescaling).
3. **ECS concentration**: the generalizing structure of the layer lives in the singular
   vectors of the ECS — the low-rank subspace spanned by the dominant (tail) eigenvalues.

Holding (1) and (2) is conjectured to force (3). When all three hold in every layer, the
theory claims the model achieves the lowest generalization error its architecture and
dataset allow.

### The Two-Tail Convergence Argument

Why believe condition (2) rather than treating it as numerology? Because **α ≈ 2** (a
shape statement about the tail's density) and **det(X̃) ≈ 1** (a volume statement about
the tail's eigenvalues) are derived from different mathematics — a Clauset MLE fit versus
a saddle-point argument in a Student-Teacher partition function — and empirically they
are satisfied *simultaneously*, at the same layer states, in MLPs, CNNs, ViTs and LLMs.
Two independent derivations converging on the same point is the semi-empirical theory's
central vindication.

---

## 4. Where α and α̂ Come From (the R-Transform Models)

SETOL derives the quality metrics by modelling the tail's eigenvalue distribution with
four exactly-solvable free-probability models and computing "layer quality" Q̄² as a sum
of integrated **R-transforms** (the free-probability analogue of the log-cumulant
generating function). Each model then produces one of the published metrics:

| Model | Regime | R-transform | Metric it grounds |
|-------|--------|-------------|-------------------|
| **Bulk+Spikes (BS)** | early training / α > 6 | constant (sum of spike eigenvalues) | Frobenius tail norm (illustrative, no published WW column) |
| **Free Cauchy (FC)** | critical, α = 2 | constant a + iγ | **Alpha** — log Q ∼ 1/α, so smaller α ⇒ higher quality |
| **Lévy-Wigner (LW)** | VHT, α < 2 | b·z^(α−2) | **AlphaHat** in the over-regularized regime |
| **Inverse Marchenko-Pastur (IMP)** | good regime, 2 < α < 6 | has a branch cut at κ/2 | **AlphaHat**; the branch cut location *defines the ECS boundary* |

The logic to remember:

- **Alpha (shape only)** is justified at the critical point by the Free Cauchy model.
- **AlphaHat = α · log₁₀ λ_max (shape + scale)** generalizes it off the critical point:
  the IMP branch cut couples the tail's exponent to its spectral mass, so on the good
  regime quality is ~ (α−1)·log λ_max rather than α alone.
- **Model quality** is the product of layer qualities, so log model quality is the
  **layer average of log layer quality**, i.e. ⟨α̂⟩ (layer-averaged AlphaHat). This is
  the quantity validated in meta-analyses of hundreds of SOTA CV and NLP models.

### The Scale Paradox and Its Resolution

Classical statistical learning theory says: smaller weights ⇒ better generalization.
HTSR/SETOL observe the opposite at the spectral scale level — **larger λ_max correlates
with better models**. SETOL resolves this: in a well-trained layer the spectrum has
*grown* structure (large λ_max) while remaining *organized* (α near 2), and AlphaHat
measures exactly that combination. A small λ_max with normal-looking weights is the
signature of an under-trained layer, not of a well-behaved one. Corollary: α̂, log norms
etc. are only comparable across architectures as **layer averages**, never single-row.

---

## 5. The ERG Condition and the Funnel Diagnostic

### 5.1 What `erg_log_det` Measures

Take the eigenvalues above the power-law threshold xmin (this set is the **ECS**), rescale
them so their product is normalized, and sum their logs:

    erg_log_det = Σ ln λ̃ᵢ   ≈   ln det(X̃)

- **≈ 0** (|value| < 1.0 is the shipped threshold, `erg_satisfied = True`): volume is
  preserved — consistent with Ideal Learning / the critical point.
- Strongly positive or negative: the ECS volume has expanded or collapsed away from the
  renormalization-group fixed point.

### 5.2 What `erg_delta_lambda_min` Measures

The signed gap between **where the power-law tail starts** (the fitted xmin) and
**where the ERG/detX≈1 tail boundary lies**:

| Sign | Meaning |
|------|---------|
| > 0 | Normal — the ERG volume boundary sits above the PL fit's start |
| ≈ 0 | **Ideal** — the two tails coincide; this is the α ≈ 2 critical state |
| < 0 | **Over-regularized** — the PL tail has crossed below the ERG tail; α has dropped under 2 |

The sign is load-bearing — it is the diagnostic; never read this column through an
absolute value.

### 5.3 The Funnel

Plot Δλ_min (y) against α (x) for all layers of a model (this is the shipped
`spectral_funnel_diagram.png`). Healthy models show the points funnelling toward the
convergence point **(α=2, Δλ_min=0)**:

- under-trained layers cluster at α > 6 with large positive Δλ_min,
- well-trained layers approach (2, 0),
- over-regularized layers spill past α = 2 into Δλ_min < 0.

Tighter funnels = better global model quality; the pattern reproduces across VGG /
ResNet / ViT / DenseNet / Falcon / Llama families.

---

## 6. Random-Matrix Baseline: MP Edges, Tracy-Widom, and Traps

### 6.1 Marchenko-Pastur — What "Random" Looks Like

For an i.i.d.-Gaussian matrix of the same shape, the ESD is confined to a compact
envelope with sharp edges

    λ± = σ²(1 ± 1/√Q)²,   Q = N/M ≥ 1   (N the larger dimension)

Everything about "is this layer still random?" is a comparison of the actual spectrum
against these edges:

- `mp_softrank = λ₊ / λ_max`: near 1 ⇒ the whole spectrum sits inside its theoretical
  bulk, i.e. it still looks random (under-trained). Small ⇒ the spectrum has grown well
  past the random baseline (learning has happened). Above 1 ⇒ even the top eigenvalue
  is inside the bulk (extremely weak signal).
- `sv_max` / `λ_max` vs `λ₊`: a sizable excess above the edge is how "spikes" are defined.

### 6.2 Tracy-Widom Fluctuations — How Far Above λ₊ "Real" Spikes Stick Out

Finite random matrices put their upper-edge fluctuations at scale O(M^(−2/3)) above λ₊
(Johnstone 2001; SETOL.md §2.3). The shipped trap threshold adds this headroom:

    Δ_TW = c_TW · λ₊ · M^(−2/3) · f(Q),    f(Q) = Q^(−1/6) · (1+√Q)^(−2/3)
    threshold = λ₊ + Δ_TW,   c_TW = 3.0

Two facts worth being able to recite:

- Because the offset is **proportional to λ₊**, the *relative* headroom
  (threshold − λ₊)/λ₊ is a pure function of shape (M, Q) — the verdict is invariant under
  rescaling the weights. (An earlier port of WeightWatcher's formula lost the
  normalization upstream provides and was scale-*dependent*; the current form is a
  deliberate divergence, Johnstone's scale, not WW parity.)
- c_TW = 3.0 Tracy-Widom units is **calibrated, not inherited**: at 1.0 the false-positive
  rate on clean matrices tracked the Tracy-Widom law's own tail (~9–13%); at 3.0 it is
  ~0.7–1.3% while detection power against planted traps stays 1.0.

### 6.3 Correlation Traps — Encoding Gone Wrong

**Definition (operational)**: a trap is present when the eigenvalues of an
**element-wise randomized** copy of W exceed the MP+TW threshold of that randomized
spectrum. Randomization preserves the *distribution of element values* but destroys the
learned spatial structure; a few abnormally large individual elements therefore reappear
as spikes that no genuinely-learned structure would survive.

**Root causes**: excessively small batch sizes (bs ≈ 1), very large learning rates,
failed SGD dynamics.

**Why detect via randomization at all**: the *unrandomized* spectrum cannot distinguish
"a big eigenvalue because the layer learned a strong direction" from "a big eigenvalue
because one weight entry is huge". Randomization kills the former and keeps the latter —
it is the only spectral probe that separates learned structure from element pathology.

**Interpretation of the columns**:

| Column | Reading |
|--------|---------|
| `has_trap` | True if ANY of the `spectral_n_randomizations` (default 5) permutations fired |
| `trap_severity` | (λ_max − threshold)/λ₊ of the **worst-severity draw**; 0 = clean |
| `trap_severity_label` | none < 0.1 ≤ mild < 0.3 ≤ moderate < 0.5 ≤ severe < 1.0 ≤ critical |
| `mp_lambda_±`, `trap_threshold` | The randomized-spectrum edges and detection threshold (one representative draw: the worst) |
| `rand_sv_max`, `rand_sv_ratio`, `rand_distance` | Means over draws (the only averaged trap quantities) |

**Effects**: traps distort the ESD, corrupt the power-law fit (high D_KS, unreliable α),
and correlate with degraded test accuracy. If a layer reads over-trained (α < 2), check
`has_trap` before believing the α.

---

## 7. The Distribution-Shape Metrics (always-reliable family)

These need no fit and no distributional assumption, hence "read these first":

| Column | Definition | Why it has this meaning |
|--------|------------|------------------------|
| `stable_rank` | Σλ / λ_max | The Frobenius-norm-to-spectral-norm ratio: a *robust* effective rank. A layer using k comparable directions reads ≈ k; a rank-1-dominated layer reads ≈ 1. Divide by min(N,M) for "fraction of capacity used". |
| `entropy` | normalized Shannon entropy of pᵢ = λᵢ/Σλ over the ranked spectrum, in [0,1] | How *uniformly* information is spread across modes. ≈ 0 ⇒ collapse onto few modes; ≈ 1 ⇒ flat (random-like) spectrum. |
| `gini_coefficient` | Lorenz-curve inequality of the eigenvalues | "How rich is the richest mode vs the crowd" in one number. > 0.8 = extreme concentration. (Note: the *calibration* `gini_coefficient` elsewhere in the package is actually a Simpson index — different quantity, same name, kept for compatibility.) |
| `dominance_ratio` | λ_max / Σ(rest) | A direct rank-1-spike detector: > 1 ⇒ one mode alone outweighs all others combined. |
| `participation_ratio` (top eigenvector) | (Σvᵢ²)² / Σvᵢ⁴ | From Anderson localization physics: how many neurons a principal mode spreads over. Low PR ⇒ a "feature" living on a few neurons (memorization-like); high PR ⇒ distributed representation. |

Theory tie-in: stable rank + dominance ratio detect **Rank Collapse**, the degenerate
sixth universality class, which has no α because no tail exists to fit.

---

## 8. Fit-Quality Columns — When to Believe α

`alpha` is the headline number but is a **model of the data, not the data itself**. The
fit reports its own audit trail:

| Column | Guard | Failure meaning |
|--------|-------|-----------------|
| `xmin` | > 0 | The tail literally starts here; everything below is treated as bulk |
| `D` (KS distance) | small | Empirical tail vs fitted power law deviate by this much; large D = tail is not a power law |
| `sigma` | ≤ α/3 | Standard error of α; if the CI (α±3σ) spans multiple phases, α does not localized a phase |
| `num_pl_spikes` | ≥ 50 | How many eigenvalues fall above xmin; the MLE variance ∝ 1/n_tail, tiny tails give wild α |
| `pl_pvalue` | > 0.1, or −1.0 | Bootstrap goodness-of-fit vs synthetic power laws. **−1.0 = test did not run (sentinel), NOT a rejection.** A decisive 0.0 is returned only when α ≤ 1 or xmin ≤ 0 (unusable fit). |
| `alpha_unreliable` | False | Set when α > 8 (`SPECTRAL_ALPHA_SANITY_MAX`): near-degenerate tails can produce enormous α at status "success". **Flagged, never clamped** — the runaway stays visible rather than being rewritten as a plausible "under-trained". |

Two known conservatisms in the p-value (documented, not "fixed", because both push toward
false *rejection* rather than false confidence): the observed KS distance is taken at the
reported fixed xmin while synthetic draws get a free xmin search, and failed bootstrap
draws are dropped from the numerator while the denominator stays fixed.

**Small-N path**: tails with N < 20 eigenvalues use WeightWatcher's small-N estimator —
the (n−1)-for-n MLE substitution plus a tail-size penalty in the xmin selection objective
(1 + (n−1)/s, objective J = D_KS − 0.868/√n_tail, tails below 8 points excluded). This is
*not* the Clauset bias-correction (different functional form) — it is WW's own small-N
device, ported.

---

## 9. Normalization and What Survives It

- **Scale invariance of α.** Scaling W → cW scales all λ by c². A power law's exponent is
  unchanged by multiplying its data: **α is scale-invariant**. This is why α (and the
  `learning_phase`) are the metrics you can read across differently-scaled layers and
  architectures — and why every *norm-based* column (`norm`, `spectral_norm`,
  `log_norm`, `log_spectral_norm`, `lambda_max`) is NOT comparable across architectures.
- **λ_max, α̂ are scale-aware.** α̂ = α·log₁₀λ_max combines the invariant shape with the
  scale. Comparable across architectures **only as a layer average** ⟨α̂⟩ (the model-level
  quality), never row-by-row.
- **The 1/N factor.** The theory defines X = WᵀW/N; the code may compute WᵀW without it.
  Impact: λ_max and α̂ shift by the factor; α does not move at all; the ERG computation
  applies an internal `wscale` correction so its ≈0 target stays meaningful.
- **Truncated spectra** (layer wider than `spectral_max_evals`, default 15000): only the
  largest eigenvalues are computed. Every column that integrates over the WHOLE spectrum
  (norm, stable_rank, entropy, matrix_rank, rank losses, mp_softrank, the Lorenz-type
  concentration metrics) is **NaN**, not zero — a complete healthy spectrum also produces
  zeros, so zeros would be indistinguishable from truncation. Columns computable from the
  top-k alone (spectral_norm, lambda_max, sv_max, α̂ family, concentration_score,
  critical_weight_count) survive. Truncation also biases α upward (a fitted tail made of
  only the largest eigenvalues looks flatter/heavier), so treat truncated-row α with the
  same suspicion as any other fit-quality failure.

---

## 10. What the Conv-Layer Numbers Actually Describe

Conv kernels are matricized (kh, kw, c_in, c_out) → (kh·kw·c_in, c_out). Two things to be
able to say precisely:

1. This **destroys spatial structure**: the row index becomes a mixed (dy, dx, c_in)
   coordinate, which is *not* the contraction index of the convolution's linear map.
   (The SVD-preserving fold would be (kh·kw, c_in·c_out) — per-shift blocks or the traced
   form.)
2. The shipped convention is nevertheless kept **deliberately**: it is WeightWatcher's
   convention, so these numbers are comparable to published WeightWatcher results.

Therefore: read conv-layer spectral metrics as describing **the folded matrix — a linear
operator built from the kernel's entries — never as "the spectrum of the convolution."**
`rf` (receptive field = kh·kw) is carried alongside for any normalization that needs it.

Norm layers (BatchNorm, LayerNorm, RMSNorm) are **skipped on purpose**: their learnable
parameters are 1-D scale/shift vectors with a degenerate ESD — no 2-D spectrum exists to
analyze. Under the default config they are classified, skipped, and never appear in the
DataFrame.

---

## 11. One-Paragraph Explanations (for verbal answers)

- **Alpha**: "The exponent of the power law that the tail of a layer's eigenvalue
  spectrum follows. Trained networks self-organize into heavy tails; α ≈ 2 is the
  critical boundary where the spectrum's variance is just-barely-finite and where the
  renormalization-group volume condition also holds. Below 2 the layer has entered an
  atypical over-trained/over-regularized phase; above 6 it is still essentially random."

- **AlphaHat**: "Alpha times log of the largest eigenvalue — shape plus scale. It comes
  from SETOL's Inverse-Marchenko-Pastur model, where the tail exponent and the spectral
  mass are coupled through the branch cut that defines the Effective Correlation Space.
  Averaged over layers, it predicts model quality without any test data, and unlike
  classical learning theory it says bigger spectral scale (not smaller) is better."

- **ERG condition**: "The product of the tail eigenvalues, rescaled, is ≈ 1 — i.e. the
  learned subspace's log-volume vanishes. SETOL shows this is one step of Wilson's exact
  renormalization group, and that it becomes satisfiable exactly when α ≈ 2. Two
  independent conditions, one from fitting the tail's shape, one from a volume
  computation, meeting at the same point is the theory's central evidence."

- **Correlation trap**: "A few pathologically large individual weight elements. We
  detect them by shuffling the matrix — the shuffle preserves element sizes but destroys
  learned structure — and checking whether any eigenvalue then sticks out past the
  Marchenko-Pastur edge plus its Tracy-Widom fluctuation allowance. Whatever survives
  that shuffle was never structure; it's a scale pathology, and it corrupts the α fit."

- **Why a power law at all?** "Because learned feature hierarchies are scale-free: there
  is no characteristic strength of a learned direction, directions span orders of
  magnitude. A heavy tail is the spectral signature of that scale-freeness, and α
  measures how heavy it is. This is the same reason Zipf's law appears in language — and
  the exponent, not the amplitude, is the invariant."

---

## 12. Cross-Reference

- Operational details (columns, config knobs, cost figures, truncation policy): `README.md`.
- Full SETOL derivation (HCIZ integrals, Student-Teacher model, R-transform algebra): `SETOL.md`.
- Trap theory, thresholds and the doc-vs-code divergence table: `CORRELATION_TRAPS.md`.
- Thresholds and sentinels as code: `constants.py` (`SPECTRAL_*`).
