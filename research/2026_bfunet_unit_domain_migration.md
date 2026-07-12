# The `[0,1]` Domain Migration for Bias-Free Denoisers

*Date: 2026-07-12. Plan: `plans/plan_2026-07-12_e56909cd/`.*

Every bias-free denoiser code path in this repository moved from the zero-centered pixel
domain `[-0.5, +0.5]` to the strictly-positive domain `[0, 1]`:

- training and eval (`src/train/bfunet/`),
- the conformal denoiser-domain defaults in `src/dl_techniques/`,
- the whole inverse-problem app (`src/applications/bias_free_denoiser/`).

This note records **why**, **what deliberately did not change**, **what was invalidated**, and
the three places where a literal find-and-replace would have gone green and been wrong.

---

## 1. Why: `[0,1]` forces the filters to sum to one

### The structural fact

A bias-free network has **no additive bias anywhere** - `use_bias=False` on every conv, and a
variance-only `BiasFreeBatchNorm` (`center=False`). With a positively-homogeneous activation,
the whole network `f` is therefore **degree-1 homogeneous**:

```
f(c · x) = c · f(x)     for every scalar c > 0        and hence        f(0) = 0
```

This identity holds at initialization, after training, for any weights. It is what makes a
bias-free denoiser generalize across noise levels (Mohan et al., ICLR 2020). It also means the
network **cannot represent a DC offset**: it has no mechanism to add or subtract a constant.

### On `[-0.5, +0.5]` the flat-patch case is vacuous

A denoiser's local filters must **preserve the DC level** of a flat region (sky, wall, paper,
blurred background): average away the noise, leave the brightness alone.

On a zero-centered domain, a flat mid-grey patch is the vector `0`, and

```
f(0) = 0
```

is **structural, not learned**. The network reproduces it for free. So the DC component - the
very thing the filters must preserve - is **never supervised at its most important operating
point**, and a large fraction of natural-image background sits exactly there. The training
pressure to learn DC preservation is badly weakened. If the filters never learn to sum to one,
the denoiser produces **unpredictable brightness shifts**, most visibly at out-of-distribution
noise levels where nothing else pins the DC down.

Zero-centering does not make the flat-patch case *hard*. It makes it **vacuous**.

### On `[0,1]` the flat-patch case forces `f(1) = 1`

On `[0,1]` a flat patch of value `c` is the vector `c · 1`. By homogeneity,

```
f(c · 1) = c · f(1)
```

so reproducing it - `f(c·1) = c·1` - **requires**

```
f(1) = 1
```

that is, **local filter weights that sum to one**. There is no way to get it for free. The
network is forced to become an adaptive low-pass / averaging filter, which is the
geometrically correct behavior for a denoiser and exactly what the Miyasawa / Tweedie
empirical-Bayes identity

```
E[x | y] = y + σ² · ∇_y log p(y)
```

relies on for the residual-equals-score reading to hold *across* noise manifolds rather than
at a single radius. The same homogeneity that hands the property over for free under
zero-centering is what makes it mandatory under `[0,1]`: one structural fact, two opposite
consequences, decided entirely by where the domain sits.

### References

- Mohan, Kadkhodaie, Simoncelli & Fernandez-Granda, *Robust and Interpretable Blind Image
  Denoising via Bias-Free Convolutional Neural Networks*, **ICLR 2020**.
- Kadkhodaie & Simoncelli, *Stochastic Solutions for Linear Inverse Problems using the Prior
  Implicit in a Denoiser*, **NeurIPS 2021**.

### Measured corroboration (this migration)

`src/train/bfunet/common.py` now runs a **DC / sum-to-one probe** alongside
`_homogeneity_probe` (informational only, it never raises). It feeds a flat image `c · 1` and
reports `rel_err = ‖f(c·1) − c·1‖ / ‖c·1‖`. On an **untrained** bias-free ConvUNext:

| `c` | 0.1 | 0.25 | 0.5 | 0.75 | 0.9 |
|---|---|---|---|---|---|
| `rel_err` | **1.502** | **1.502** | **1.502** | **1.502** | **1.502** |

The value is **identical for every `c`**, which is precisely what degree-1 homogeneity
predicts:

```
‖f(c·1) − c·1‖ / ‖c·1‖  =  |c| · ‖f(1) − 1‖ / (|c| · ‖1‖)  =  ‖f(1) − 1‖ / ‖1‖
```

The ratio *cannot* depend on `c`. That constancy confirms the probe is measuring
`‖f(1) − 1‖` - **the sum-to-one property itself** - and nothing else. A random untrained
network's filters do not sum to one, so `1.502` is the expected baseline, not a failure. It is
now a first-class training-time diagnostic. Under `[-0.5,+0.5]` there was no such number to
report, because the flat mid-grey case was just `f(0) = 0`.

### The old advice, and why it was wrong

`research/miyasawas_theorem.md` previously rated `[0,1]` as unsuitable for bias-free networks
and flagged `images / 255.0` as a mistake. It has been **corrected** (not deleted - a deleted
objection gets re-derived).

The error was importing a generic deep-learning heuristic - *"zero-center your inputs, it
conditions optimization better"* - into a setting where the network is **bias-free**. The
heuristic is not wrong in general. It is wrong *here*, and for a specific structural reason:
in a bias-free network, zero-centering silently removes the training signal for the one
property the whole empirical-Bayes construction depends on.

---

## 2. INV-2: what did NOT change, and must NEVER be "fixed"

**This migration is a pure DC shift, not a rescale.** The peak-to-peak width is `1.0` in
**both** domains:

```
[-0.5, +0.5]   width = 1.0
[ 0.0,  1.0]   width = 1.0
```

Everything below was **already exactly correct** in the new domain and was left **untouched**:

| Site | Value | Why it is unchanged |
|---|---|---|
| `PsnrMetric(max_val=1.0)` | `1.0` | peak-to-peak, not maximum |
| `SsimMetric(max_val=1.0)` | `1.0` | same |
| `common.py::_mean_psnr` | `20·log10(1.0 / rmse)` | same |
| `sigma_255 = sigma * 255` | - | sigma is a std on a unit-width domain |
| `noise_sigma_min`, `sigma_max_start`, `sigma_max_end` | unchanged | noise stds, not pixel offsets |
| `solver._DEFAULT_SIGMA_0` | `0.4` | a noise std |
| `ddnm.sigma_start` | `0.5` | a noise std, **not** a pixel offset |

> ### WARNING - the single most likely future mistake
>
> The seductive error is to reason *"the domain moved, so the noise scale must move too."* It
> must not. **Rescaling any sigma, any `max_val`, or the PSNR/SSIM formula would silently
> corrupt every reported dB number and every trained model's noise curriculum - and nothing
> would fail.** No test would go red. Every subsequent dB figure would be quietly
> non-comparable with every published one.
>
> If you find yourself editing `PsnrMetric(max_val=...)`, `SsimMetric(max_val=...)`,
> `_mean_psnr`, `sigma_255 = sigma*255`, `noise_sigma_min`, `sigma_max_start`,
> `sigma_max_end`, `_DEFAULT_SIGMA_0`, or `ddnm.sigma_start` because of the domain: **stop and
> revert.** Only the *center* moved. The *width* did not.

---

## 3. What was invalidated, and how the breakage was made loud

### All 17 legacy checkpoints are dead

Every `.keras` denoiser checkpoint under `results/` was trained with the old
`(image / 255) - 0.5` normalization, **including the app's former default**
`convunext_denoiser_base_20260707_122133`. Every one of them is invalidated.

There is **no partial-migration state that works**. A bias-free net is degree-1 homogeneous
and **has no mechanism to subtract a DC offset**, so feeding `[0,1]` data to a
`[-0.5,+0.5]`-trained net (or the reverse) is **silent garbage, not an error**. The output is
finite, plausible-looking, and wrong. Nothing raises. This is symmetric in both directions.

### The `data_range` provenance stamp

Because the failure is silent, the breakage was made **loud**:

1. `BFUnetTrainingConfig` gained a field `data_range: str = "[0,1]"`, written into every new
   run's `results/<run>/config.json` via `save_config_json`.
2. `DenoiserPrior.from_pretrained` gained a fail-fast gate: if the sibling `config.json` is
   missing, or its `data_range` is not `"[0,1]"`, it **raises `ValueError`** naming the
   checkpoint and stating that a retrain is required. **An absent key means legacy, and legacy
   means refuse** - every checkpoint that existed before this migration lacks the key.

All three app entry points (`main.py`, `metrics.py`, `streamlit_app.py`) route through
`from_pretrained`, so the one gate covers the whole app.

> **`data_range` is a RECORD and a GATE, never a compat switch.** Nothing branches on it. It
> changes no math. It only refuses. Do not turn it into a `pixel_domain` dispatch, do not add
> an `allow_legacy_domain` escape hatch, and do not add a second normalizer beside
> `common.py`'s loader or `DenoiserPrior.ingest`. A flag whose only purpose is to re-enable a
> broken path is a compat shim by another name.

Consequence, stated plainly: **the app is non-functional until a `[0,1]` model is trained.**
The 20 slow integration tests SKIP (they do not fail, and they do not silently pass on a
legacy checkpoint); they auto-re-enable the moment a `[0,1]` checkpoint exists.

---

## 4. Restructure, don't swap: three sites where a literal replace would have gone GREEN and been WRONG

These are the most instructive part of the migration. Each of them encodes the zero-center
**structurally** - in the *shape* of the expression, not in a `-0.5` literal - so a
grep-driven find-and-replace passes right over them (or "fixes" them into something equally
wrong) and the suite stays green with wrong semantics.

### 4.1 `eval_per_pixel_uncertainty.py` - the saturation mask

The old test was one-sided and symmetric-domain: `np.abs(noisy) >= 0.5 - eps`. Measured on one
real noisy batch:

| Form | Pixels flagged as "saturated" |
|---|---|
| Correct two-sided: `(x <= eps) | (x >= 1 - eps)` | **15.30 %** |
| Legacy one-sided `abs(x) >= 0.48`, run on `[0,1]` data | **50.47 %** - it flags the *entire upper half of the image* |
| Naive literal swap to `abs(x) >= 0.98` | **5.63 %** - it silently *drops* the 9.67 % of near-black pixels |

Both wrong forms produce a wrong *statistic*, not a crash. This file has **no test coverage**.

### 4.2 `tests/test_utils/test_conformal_denoiser_intervals.py` - coverage is blind to the domain

Split-conformal coverage was measured at **0.900 under BOTH domains**: `0.90000` on `[0,1]`,
and `0.90070` with the legacy clip re-introduced. The clip fires **identically in calibration
and in test**, so the marginal guarantee is still *attained* - just on a meaningless,
clip-distorted interval (the calibrated quantile collapses from `0.083` to `0.065`).

**The coverage assertions are structurally incapable of detecting a wrong domain.** Only the
two-sided in-bounds guard can catch it. Do not treat a green coverage test as evidence that
the domain is right.

### 4.3 `tests/.../test_integration.py` - the `abs()`-based gates

- `|out| <= 0.5` rejects the **entire upper half** of a perfectly valid `[0,1]` image.
- `|out| <= DENOISE_CEIL = 1.0` degenerates into a gate that can **never go red** - every
  in-domain pixel passes it by construction.

Both now bound **deviation from the domain center** (`|out - 0.5|`), which is what they always
meant. The rule: **`abs()` applied to a pixel value is a zero-center assumption in disguise.**
`abs()` applied to a *residual* or a *difference* is fine.

---

## 5. INV-5: research integrity - `bfunet.tex` is deliberately NOT retro-edited

`research/papers/bfunet/bfunet.tex` reports **measured numbers** from a
`[-0.5,+0.5]`-trained checkpoint: the domain, the curriculum `sigma_norm ∈ [0, 0.5]`, and the
full-image PSNR benchmark protocol.

**It was deliberately left untouched.** Editing it to claim `[0,1]` would falsify reported
experimental results. That is a hard integrity line, not a style preference.

> ### TODO (blocking, before any `[0,1]` claim enters the paper)
>
> **`research/papers/bfunet/bfunet.tex`'s numbers must be RE-MEASURED on a `[0,1]`-trained
> checkpoint.** Until then, the paper describes the **pre-migration** model, and the repository
> is knowingly in a visibly inconsistent state: a paper describing a domain the code no longer
> uses. Do not resolve that inconsistency by editing the paper. Resolve it by retraining and
> re-measuring.

---

## 6. What this migration did NOT deliver: a trained model

This plan delivered a **correct pipeline**, not a trained model.

**Every claim about denoising quality, DC preservation, or Miyasawa score fidelity on `[0,1]`
is unverifiable until a full retrain completes.** The DC probe above shows the diagnostic is
wired and numerically sound; it shows the untrained baseline (`1.502`); it does **not** show
that the sum-to-one property is *learned*.

Two things remain genuinely open:

1. **Is the property learned?** The probe on a converged `[0,1]` model must show `rel_err`
   dropping well below the untrained `1.502`. Not yet measured.
2. **Is the old dying-neuron concern real?** `miyasawas_theorem.md`'s original worry about
   dead units and instability on strictly-positive inputs is a real ReLU-family failure mode
   **in general**, and it has **not** been empirically refuted here. The correction in that
   document refutes its *argument* (which was a misapplied generic heuristic), not this
   specific empirical risk.

The plan's stop trigger: if the `[0,1]` smoke train **diverges** (NaN loss, or `val_loss` never
falling below its epoch-0 value), or if the DC probe on a short-trained model moves **away**
from `f(c·1) = c·1` as training proceeds, then the concern was real and the hypothesis must be
revisited. Bisect against the last green commit first - a pipeline bug falsifies neither the
plan nor the argument.

---

## 7. Dead code removed along the way

- **`ddnm._DOMAIN_HALF_WIDTH`** - zero reads repo-wide. A constant that existed only to be
  audited.
- **The `inspect.getsource` textual byte-identity guard** on `make_curriculum_noise_fn`
  (`tests/test_train/test_bfunet/test_convunext_self_iterate.py`). It shelled out to
  `git show` and text-compared source after folding `-0.5, 0.5` into `-1.0, 1.0`, so the clip
  edit failed it **by construction**. It asserted on **TEXT, not runtime behavior**. Replaced
  by a seeded **behavioral** test that pins the surviving invariant - the additive branch draws
  `noise_level` FIRST, then `tf.random.normal` - by comparing a seeded forward pass against a
  hand-computed reference that replays the draws in that order. Verified live: swapping the two
  draws makes it FAIL; reverting makes it pass.
- **The `clean_pm05` parameter name** in `metrics.degrade_and_reconstruct` - a dead domain
  encoded into a public API name.

---

## 8. Quick reference for the next reader

| Question | Answer |
|---|---|
| What domain do the denoisers use? | `[0, 1]`. `image / 255.0`. Nothing else. |
| Where is the domain defined? | `src/train/bfunet/common.py`: `DATA_MIN = 0.0`, `DATA_MAX = 1.0`. One place. |
| Where does the app enter/exit the domain? | `DenoiserPrior.ingest` / `.denorm`. One place. Do not add a second. |
| Did sigma / PSNR / SSIM change? | **No.** See §2. Do not "fix" them. |
| Can I load an old checkpoint? | **No.** `from_pretrained` refuses it. See §3. |
| Why does `abs(pixel)` show up in a review comment? | It is a zero-center assumption in disguise. See §4. |
| Are the paper's numbers still valid? | For the **old** model, yes. They must be re-measured. See §5. |
| Is `[0,1]` proven better here? | **Not yet.** Structurally argued, not empirically demonstrated. See §6. |
