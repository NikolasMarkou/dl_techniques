# Audit of `GoodhartAwareLoss` and a Replacement That Does What It Claims

## Contents

1. Verdict
2. Code findings
3. Claim audit
4. Empirical check
5. What actually delivers the claim
6. Replacement: `ConfidencePenaltyLoss`
7. Reproduction
8. Appendix: code

## 1. Verdict

The arithmetic is correct: every term matches a float64 NumPy reference to 6 decimals. The headline claim is not supported.

- The loss cannot distinguish flimsy from strong evidence.
- The default setting (`entropy_weight=0.1`) is a temperature shift that post-hoc temperature scaling reproduces exactly.
- The anti-collapse prior term is harmful unless the batch is much larger than the class count.
- All-zero target rows, documented as the masking encoding, are silently trained toward uniform.

Every finding below was verified by execution (evidence grade V).

## 2. Code findings

| # | Severity | Finding | Evidence |
|---|---|---|---|
| 1 | High | All-zero target rows, which the docstring calls "the ordinary masked / ignore-index encoding", are not ignored. The entropy term, and Keras label smoothing (which turns them into eps/K targets), push those inputs toward uniform. | Gradient on masked rows: 0.0075 (ls=0), 0.0218 (ls=0.1). Replacement: 0.0 |
| 2 | High | `prior_weight > 0` penalizes correct classifiers whenever a batch's label mix deviates from the prior. A perfect 99%-confident classifier scores KL 0.24 at K=10, B=32 (24x its CE) and 1.13 at K=100, B=128. Under supervised labels the term is redundant. | Table 2.1 |
| 3 | Medium-High | The comment "one producer of log p" is false. CE goes through Keras' internal clip (1e-7); the regularizers go through `epsilon`. `epsilon` never touches CE. | Probabilities containing zeros: CE 6.48 (Keras) vs 6.81 (shared producer) |
| 4 | Medium | `analyze_loss_components` claims `total_loss` equals the live loss. It does not for `reduction='sum'`, and it ignores `sample_weight`. It also raises on a wrong-length `class_prior` at `prior_weight=0`, where the live loss does not. | 5.05 vs 323.29 |
| 5 | Medium | Weight validation rejects `np.float32(0.1)` (any NumPy config value). It accepts `True` and `inf`. | Direct calls |
| 6 | Medium | The prior term's effective weight is `prior_weight × batch-mean(sample_weight)`. Under `class_weight` it varies from batch to batch. | 0.0365 becomes 0.0183 at w=0.5 |
| 7 | Low | The docstring's label-smoothing identity omits the (1-eps) rescaling of CE: LS = (1-eps)·CE + eps·KL(u‖p) + const. | Algebra |

These passed: term formulas, integer-label rejection, keyword-only legacy guard against positional `mi_weight`, JSON round-trip, float16 probability path.

**Table 2.1: prior term, trained** (β = `prior_weight`, `entropy_weight=0`, 3 seeds)

| Batch | ECE β=0 | ECE β=1 | NLL β=0 | NLL β=1 |
|---|---|---|---|---|
| 8 | 0.019 | 0.337 | 0.156 | 0.542 |
| 32 | 0.022 | 0.076 | 0.184 | 0.201 |
| 256 | 0.029 | 0.027 | 0.221 | 0.208 |

## 3. Claim audit

The claim: the loss stops a model from becoming "confidently certain on flimsy evidence".

### 3.1 The penalty is input-agnostic

For a hard label, the optimum of CE minus λH(p) satisfies log(p_y / p_other) = 1 / (λ·p_y). The penalty therefore caps the logit margin at about 1/λ, whatever the evidence.

| Setting | Optimal p_y (K=10) | Margin (nats) |
|---|---|---|
| λ=0.01 | 1.000000 | 20.4 |
| λ=0.1 (default) | 0.999593 | 10.0 |
| λ=0.3 | 0.849 | 3.9 |
| λ=0.5 | 0.679 | 3.0 |
| LS eps=0.1 | 0.910 | 4.5 |

The default barely binds.

### 3.2 It is not a proper scoring rule

Take a true conditional q = (0.7, 0.2, 0.1, 0, ..., 0) and minimize the expected loss over p:

| λ | Minimizer (first 3 classes) | Mass on impossible classes |
|---|---|---|
| 0.0 | (0.700, 0.200, 0.100) | 0.000 |
| 0.1 | (0.670, 0.215, 0.115) | 0.000 |
| 0.5 | (0.484, 0.200, 0.128) | 0.188 |

The loss distorts probabilities even with infinite data. Calibrated outputs therefore require a recalibration step on held-out data.

### 3.3 Train-set penalties are blind to the generalization gap

Training accuracy was 100% in every run. Any penalty computed from training (y_true, y_pred) therefore sees a model that is confident and correct, and has nothing to correct. MMCE, a train-set calibration penalty, confirmed this: it had no benefit (Section 4).

## 4. Empirical check

### 4.1 Setup

- Data: sklearn digits, split 400 train / 300 validation / 1097 test.
- Model and training: MLP 2×512, Adam 1e-3, 300 epochs, batch 32.
- Statistics: 5 seeds, paired t-test against CE. "n.s." means |t| < 2.78 (p ≥ .05).

### 4.2 Metrics

- TS: temperature fitted on the validation set.
- ECE: 15-bin expected calibration error.
- err AUROC: how well confidence separates right from wrong test predictions.
- OOD: pixel-permuted test digits, which carry no class evidence. The outlier-exposure (OE) training outliers were uniform noise, a different distribution.
- Blend: a 50/50 pixel mix of two different classes. The ideal max-probability is about 0.5.

### 4.3 Results

All columns except "ECE raw" are measured after TS.

| Config | Acc | ECE raw | ECE TS | NLL TS | err AUROC | OOD AUROC | OOD conf | Blend conf | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| CE | 96.2 | 0.023 | 0.016 | 0.129 | 0.955 | 0.914 | 0.68 | 0.69 | Baseline |
| GAL λ=0.1 (default) | 96.4 n.s. | 0.019 | 0.018 | 0.128 n.s. | 0.951 n.s. | 0.918 | 0.67 n.s. | 0.70 n.s. | Cosmetic: temperature shift (T 2.36 to 1.85) that TS reproduces |
| GAL λ=0.3 | 97.8 | 0.215 | 0.011 | 0.081 | 0.968 | 0.918 | 0.67 | 0.67 | Real generalization gain; raw output badly underconfident |
| LS 0.1 | 97.7 | 0.147 | 0.012 | 0.079 | 0.972 | 0.936 | 0.63 | 0.66 | Same as λ=0.3 |
| CE + MMCE (λ=2) | 96.4 n.s. | 0.022 | 0.020 | 0.133 n.s. | 0.949 | 0.913 | 0.68 | 0.69 | Cosmetic, slightly harmful |
| CE + mixup (α=0.4) | 96.2 | 0.078 | 0.021 | 0.158 | 0.947 | 0.754 | 0.82 | 0.65 | Breaks OOD detection |
| CE + OE (uniform targets) | 96.5 n.s. | 0.023 | 0.016 | 0.122 n.s. | 0.959 | 0.999 | 0.17 | 0.69 | Real, OOD only |
| GAL λ=0.1 + outliers as zero rows | 96.5 n.s. | 0.018 | 0.017 | 0.120 | 0.959 | 0.999 | 0.17 | 0.70 | Finding 1: "masked" rows act as OE |
| Proposed: LS 0.1 + OE + TS | 97.7 | 0.133 | 0.011 | 0.081 | 0.971 | 0.999 | 0.17 | 0.65 | Both gains, additive |
| Proposed: λ=0.3 + OE + TS | 97.7 | 0.206 | 0.011 | 0.084 | 0.964 | 0.999 | 0.16 | 0.64 | Same, weaker err AUROC |

### 4.4 Confident and wrong

This is the failure the docstring names directly: a prediction with confidence above 0.9 that turns out to be wrong.

| Config | Raw | After TS |
|---|---|---|
| CE | 1.60% | 0.49% |
| GAL λ=0.1 | 1.31% | 0.53% |
| GAL λ=0.3 | 0.00% | 0.40% |
| LS 0.1 | 0.00% | 0.38% |
| Proposed λ=0.3 + OE | 0.00% | 0.51% |
| Proposed LS 0.1 + OE | 0.00% | 0.42% |

Rates are percentages of all test samples. After TS, every config lands at 0.38% to 0.53%, equal within noise. TS removes this failure; no loss term does better.

## 5. What actually delivers the claim

The claim splits into four distinct failures. Each needs a signal that labeled (y_true, y_pred) pairs alone do not contain.

| Failure | Signal needed | Mechanism | Status |
|---|---|---|---|
| Global overconfidence (confident and wrong) | Held-out data | Temperature fit on validation | Verified: 1.60% to 0.49% |
| Confidence on inputs with no class evidence | Examples of such inputs | OE: uniform targets on outliers | Verified: OOD AUROC 0.914 to 0.999 |
| Memorization hurting generalization | None (regularizer) | LS 0.1 or λ around 0.3 | Verified: acc +1.5 pp, NLL(TS) -37% |
| Confidence on conflicting evidence | X-aware training | Mixup lowered raw blend confidence to 0.55; TS undid it and OOD detection broke | Not solved |

## 6. Replacement: `ConfidencePenaltyLoss`

### 6.1 What changed

- The docstring claims only what was measured, including properness and the margin cap.
- CE and the penalty use one log-p producer, so `epsilon` governs both.
- Target rows:
  - One-hot, smoothed and soft (mixup) rows are labeled rows.
  - Uniform rows are outlier-exposure rows.
  - All-zero rows are truly ignored: zero loss and zero gradient, including under label smoothing.
- The prior term is removed and `prior_weight > 0` raises. Legacy `GoodhartAwareLoss` configs at the default `prior_weight=0` load into the identical objective (verified).
- `fit_temperature(val_logits, val_labels)` ships with the loss and matches SciPy to 1e-3. The loss does not claim calibration without it.
- Validation accepts NumPy scalars and rejects bool, inf and nan.
- The diagnostic's `total_loss` equals the live loss under any reduction and `sample_weight`.
- 17 of 17 tests pass (Appendix B).

### 6.2 Recommended use

1. Choose the regularizer: either `label_smoothing=0.1, entropy_weight=0.0`, or `entropy_weight=0.3`.
2. Add outlier inputs with uniform targets (`1/K` per class) at `sample_weight=0.5`. Use a source that differs from the OOD you will evaluate on.
3. After training, call `fit_temperature` on validation logits and divide logits by T at inference.

```python
import numpy as np, keras
from confidence_penalty_loss import ConfidencePenaltyLoss, fit_temperature

K = 10
x_out = np.random.uniform(0, 1, (len(x_train), x_train.shape[1])).astype("float32")
y_out = np.full((len(x_out), K), 1.0 / K, "float32")
X = np.concatenate([x_train, x_out])
Y = np.concatenate([y_train_onehot, y_out])
W = np.concatenate([np.ones(len(x_train)), np.full(len(x_out), 0.5)]).astype("float32")

model.compile(optimizer="adam", loss=ConfidencePenaltyLoss(label_smoothing=0.1, entropy_weight=0.0))
model.fit(X, Y, sample_weight=W, epochs=..., batch_size=...)

T = fit_temperature(model.predict(x_val), y_val_int)
probs = keras.ops.softmax(model.predict(x_test) / T)
```

### 6.3 Caveat

These results come from one small dataset and one architecture. Re-measure LS vs λ, and the choice of outlier source, on the target task with the included harness.

## 7. Reproduction

### 7.1 Environment

```bash
pip install keras jax jaxlib scikit-learn scipy
export KERAS_BACKEND=jax
```

Tested with Keras 3.15.1, JAX 0.12.0.1, scikit-learn 1.9.1, Python 3.12, CPU.

### 7.2 Layout

Save each appendix block under the file name in its heading.

```
.
├── confidence_penalty_loss.py          # Appendix A
├── test_confidence_penalty_loss.py     # Appendix B
├── calibration_experiment.py           # Appendix C
├── prior_experiment.py                 # Appendix D
├── unit_checks.py                      # Appendix E
├── theory_checks.py                    # Appendix F
├── summarize.py                        # Appendix G
├── goodhart_loss.py                    # Appendix H (original, under audit)
└── dl_techniques/
    ├── __init__.py                     # empty
    └── utils/
        ├── __init__.py                 # empty
        └── keras_registration.py       # Appendix I (stub; omit inside dl_techniques)
```

### 7.3 Commands

```bash
python test_confidence_penalty_loss.py      # 17 PASS lines, finding 1 gradients
python unit_checks.py                        # Section 2 findings 3-6
python theory_checks.py                      # Section 3 tables, KL bias numbers
python calibration_experiment.py ce,gal_0.1,gal_0.3,ls_0.1,ce+mmce
python calibration_experiment.py gal_0.1+zero_rows,ce+oe_uniform,ce+mixup,ce+mixup+oe
python calibration_experiment.py new_cp0.3+oe,new_ls0.1+oe
python summarize.py                          # Section 4.3 table
python prior_experiment.py                   # Table 2.1
```

Runtime is about 30 to 50 s per config (5 seeds) on one CPU core.

## 8. Appendix: code

### Appendix A: `confidence_penalty_loss.py`

The replacement loss.

```python
"""
Cross-entropy with a per-sample confidence penalty, an outlier-exposure path
for inputs that carry no class evidence, and a post-hoc temperature fit.

What each part does (each claim was checked empirically, see audit):

1.  Cross-entropy, optionally label-smoothed. Computed from the SAME
    log-probabilities as the penalty, so `epsilon` governs every term.
2.  Confidence penalty ``-entropy_weight * H(p)`` (Pereyra et al., 2017).
    For a hard label its optimum caps the logit margin at about
    ``1 / entropy_weight`` (10 nats at 0.1, i.e. p_y ~ 0.9996, so the default
    barely binds). It is input-agnostic: it lowers confidence everywhere,
    not on "flimsy evidence" specifically. It is NOT a proper scoring rule:
    the population minimizer is not p(y|x), so raw outputs are biased toward
    uniform. Its verified benefit is generalization (accuracy, NLL after
    recalibration), not calibrated probabilities.
3.  No-evidence rows (outlier exposure, Hendrycks et al., 2019). A row whose
    target is the uniform vector ``1/K`` makes cross-entropy pull ``p``
    toward uniform on THAT input. This is the only part that lowers
    confidence where the input carries no class evidence, and it needs the
    caller to supply such inputs (noise, other datasets, corrupted samples).
    Weight them with ``sample_weight`` (0.5 is a sane start).
4.  Ignored rows. An all-zero target row contributes exactly 0 to every
    term, including label smoothing and the penalty.
5.  :func:`fit_temperature`. Calibration is a held-out quantity; a loss that
    only sees training labels cannot measure it. Fit T on validation logits
    and divide logits by T at inference.

What this loss does NOT do: it never sees ``X``, it cannot tell strong from
weak evidence on labeled rows, and it does not produce calibrated
probabilities without step 5.

Removed: ``prior_weight`` > 0 (batch-marginal KL). Under supervision the
labels already prevent collapse; the term was measured to raise test ECE
3.4x at batch 32 and 18x at batch 8 (batch < classes), and to be inert at
batch 256. ``mi_weight`` remains removed.
"""

import numbers
import warnings
import numpy as np
import keras
from typing import Any, Dict, Optional, Sequence

from dl_techniques.utils.keras_registration import register_dl_technique


def _check_weight(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (numbers.Real, np.floating, np.integer)):
        raise ValueError(f"{name} must be a real number, got {value!r}.")
    value = float(value)
    if not (np.isfinite(value) and value >= 0):
        raise ValueError(f"{name} must be finite and non-negative, got {value}.")
    return value


@register_dl_technique("dl_techniques.losses.confidence_penalty_loss")
class ConfidencePenaltyLoss(keras.losses.Loss):
    """
    ``L_i = CE(y_i, p_i) - entropy_weight * H(p_i) * [sum(y_i) > 0]``

    Targets: shape ``(batch, num_classes)``. One-hot, smoothed or soft (mixup)
    rows are labeled rows; the uniform row is a no-evidence row; the all-zero
    row is ignored.
    """

    def __init__(
        self,
        label_smoothing: float = 0.0,
        entropy_weight: float = 0.1,
        *,
        from_logits: bool = True,
        epsilon: float = 1e-8,
        name: str = "confidence_penalty_loss",
        reduction: str = "sum_over_batch_size",
        dtype: Optional[str] = None,
        prior_weight: float = 0.0,
        class_prior: Optional[Sequence[float]] = None,
        mi_weight: Optional[float] = None,
    ) -> None:
        if mi_weight is not None:
            raise ValueError(f"mi_weight has been removed (got {mi_weight}).")
        # Accepted only so legacy GoodhartAwareLoss configs at the default
        # prior_weight=0.0 (where class_prior was inert) load into the SAME
        # objective. Any positive value would change the objective: refuse.
        if _check_weight("prior_weight", prior_weight) > 0:
            raise ValueError(
                f"prior_weight={prior_weight} is no longer supported: the "
                "batch-marginal KL penalizes correct classifiers whenever a "
                "batch's label mix deviates from the prior, and is redundant "
                "under supervised labels. Train with prior_weight=0."
            )
        del class_prior
        label_smoothing = _check_weight("label_smoothing", label_smoothing)
        if label_smoothing >= 1:
            raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}.")
        entropy_weight = _check_weight("entropy_weight", entropy_weight)
        epsilon = _check_weight("epsilon", epsilon)
        if not (0 < epsilon < 0.1):
            raise ValueError(f"epsilon must be in (0, 0.1), got {epsilon}.")
        if entropy_weight > 0.5:
            warnings.warn(
                f"entropy_weight={entropy_weight} caps the hard-label logit "
                f"margin near {1 / entropy_weight:.2f} nats.", UserWarning
            )

        super().__init__(name=name, reduction=reduction, dtype=dtype)
        self.label_smoothing = label_smoothing
        self.entropy_weight = entropy_weight
        self.from_logits = bool(from_logits)
        self.epsilon = epsilon
        self._dtype_arg = dtype

    # -- single producer of log p for every term -----------------------
    def _log_probabilities(self, y_pred):
        if self.from_logits:
            return keras.ops.log_softmax(y_pred, axis=-1)
        dtype = getattr(y_pred.dtype, "name", None) or str(y_pred.dtype)
        floor = max(self.epsilon, float(np.finfo(dtype).tiny))
        probs = keras.ops.clip(y_pred, floor, 1.0)
        probs = probs / keras.ops.sum(probs, axis=-1, keepdims=True)
        return keras.ops.log(probs)

    def components(self, y_true, y_pred):
        """Per-sample (cross_entropy, entropy, has_target), shape (batch,) each."""
        y_true = keras.ops.cast(y_true, y_pred.dtype)
        log_probs = self._log_probabilities(y_pred)
        mass = keras.ops.sum(y_true, axis=-1, keepdims=True)
        has_target = keras.ops.cast(mass > 0, y_pred.dtype)
        if self.label_smoothing > 0:
            k = keras.ops.cast(keras.ops.shape(y_true)[-1], y_pred.dtype)
            # Smooth only rows that carry a target; all-zero rows stay ignored.
            y_true = y_true * (1.0 - self.label_smoothing) + has_target * (
                mass * self.label_smoothing / k
            )
        ce = -keras.ops.sum(y_true * log_probs, axis=-1)
        entropy = -keras.ops.sum(keras.ops.exp(log_probs) * log_probs, axis=-1)
        return ce, entropy, keras.ops.squeeze(has_target, -1)

    def call(self, y_true, y_pred):
        ce, entropy, has_target = self.components(y_true, y_pred)
        if self.entropy_weight > 0:
            return ce - self.entropy_weight * entropy * has_target
        return ce

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "label_smoothing": self.label_smoothing,
            "entropy_weight": self.entropy_weight,
            "from_logits": self.from_logits,
            "epsilon": self.epsilon,
            "dtype": self._dtype_arg,
        })
        return config


def fit_temperature(logits: np.ndarray, labels: np.ndarray, iters: int = 100) -> float:
    """
    Fits T > 0 minimizing held-out NLL of ``softmax(logits / T)``.

    NLL is convex in 1/T, so a golden-section search over log(1/T) is exact
    up to tolerance. ``labels`` are integer class indices. Use VALIDATION
    data, never training data: on a memorized training set the fit is
    degenerate (T -> 0).
    """
    z = np.asarray(logits, dtype="float64")
    y = np.asarray(labels).reshape(-1).astype(int)
    rows = np.arange(len(y))

    def nll(log_beta: float) -> float:
        s = z * np.exp(log_beta)
        s = s - s.max(axis=1, keepdims=True)
        return float(-(s[rows, y] - np.log(np.exp(s).sum(axis=1))).mean())

    lo, hi, g = -5.0, 5.0, (np.sqrt(5) - 1) / 2
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    fa, fb = nll(a), nll(b)
    for _ in range(iters):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - g * (hi - lo); fa = nll(a)
        else:
            lo, a, fa = a, b, fb
            b = lo + g * (hi - lo); fb = nll(b)
    return float(np.exp(-(lo + hi) / 2))


def analyze_loss_components(
    loss_fn: ConfidencePenaltyLoss,
    y_true,
    y_pred,
    sample_weight=None,
) -> Dict[str, float]:
    """
    Component breakdown. ``total_loss`` IS the live loss (same reduction,
    same sample_weight); components are unweighted means over rows with a
    target. Shares are magnitude shares in [0, 1].
    """
    ce, entropy, has_target = loss_fn.components(y_true, y_pred)
    to_np = keras.ops.convert_to_numpy
    ce, entropy, has_target = to_np(ce), to_np(entropy), to_np(has_target).astype(bool)
    n = max(int(has_target.sum()), 1)
    ce_mean = float(ce[has_target].sum() / n)
    ent_weighted = float(-loss_fn.entropy_weight * entropy[has_target].sum() / n)
    total = float(to_np(loss_fn(y_true, y_pred, sample_weight=sample_weight)))
    mags = {"ce_magnitude": abs(ce_mean), "entropy_magnitude": abs(ent_weighted)}
    tot_mag = sum(mags.values())
    return {
        "total_loss": total,
        "cross_entropy": ce_mean,
        "entropy_unweighted": float(entropy[has_target].sum() / n),
        "entropy_term_weighted": ent_weighted,
        "ignored_rows": int((~has_target).sum()),
        **mags,
        "ce_share": mags["ce_magnitude"] / tot_mag if tot_mag > 0 else 0.0,
        "entropy_share": mags["entropy_magnitude"] / tot_mag if tot_mag > 0 else 0.0,
    }
```

### Appendix B: `test_confidence_penalty_loss.py`

Tests for the replacement, including the masking comparison against the original.

```python
# Requires the original module saved as goodhart_loss.py (the class under audit).
import json, numpy as np, jax, jax.numpy as jnp, keras, warnings
warnings.simplefilter("ignore")
from goodhart_loss import GoodhartAwareLoss
from confidence_penalty_loss import ConfidencePenaltyLoss, fit_temperature, analyze_loss_components
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax
rng = np.random.default_rng(1); K, B = 10, 32
z = (rng.normal(size=(B, K)) * 3).astype("float32")
y = np.eye(K, dtype="float32")[rng.integers(0, K, B)]
ok = lambda c, m: print(("PASS " if c else "FAIL ") + m)

for ls, lam in [(0, 0.1), (0.1, 0.3)]:
    a, b = float(GoodhartAwareLoss(ls, lam)(y, z)), float(ConfidencePenaltyLoss(ls, lam)(y, z))
    ok(abs(a - b) < 1e-5, f"labeled rows identical to original (ls={ls}, lam={lam}): {a:.6f} vs {b:.6f}")

yz = y.copy(); yz[:4] = 0  # 4 'masked' rows
for ls in [0.0, 0.1]:
    for cls in [GoodhartAwareLoss, ConfidencePenaltyLoss]:
        L = cls(ls, 0.1)
        g = jax.grad(lambda zz: L(yz, zz))(jnp.asarray(z))
        print(f"     {cls.__name__:<22} ls={ls}: |grad| on all-zero 'masked' rows = {float(jnp.abs(g[:4]).sum()):.4f}")

u = np.full((B, K), 1 / K, "float32")
v = float(ConfidencePenaltyLoss(0, 0)(u, z)); ref = np.mean(-(log_softmax(z, -1) / K).sum(-1))
ok(abs(v - ref) < 1e-5, "uniform target row == KL(u||p) + log K (outlier exposure)")

p = np.array(keras.ops.softmax(z)); p[:, 0] = 0; p /= p.sum(-1, keepdims=True)
L = ConfidencePenaltyLoss(0, 0, from_logits=False)
lp = np.asarray(L._log_probabilities(keras.ops.convert_to_tensor(p)))
ok(abs(float(L(y, p)) - np.mean(-(y * lp).sum(-1))) < 1e-5, "CE uses the shared log-p producer")

sw = rng.uniform(0, 2, B).astype("float32")
for red in ["sum_over_batch_size", "sum"]:
    L = ConfidencePenaltyLoss(0.1, 0.2, reduction=red)
    ok(abs(analyze_loss_components(L, y, z, sw)["total_loss"] - float(L(y, z, sample_weight=sw))) < 1e-4, f"diagnostic total == live loss ({red}, weighted)")

for v_ in [np.float32(0.1), np.int64(0)]:
    ConfidencePenaltyLoss(entropy_weight=v_); ok(True, f"accepts {type(v_).__name__}")
for v_ in [True, float("inf"), float("nan")]:
    try: ConfidencePenaltyLoss(entropy_weight=v_); ok(False, f"rejects {v_!r}")
    except ValueError: ok(True, f"rejects {v_!r}")

legacy = GoodhartAwareLoss(0.1, 0.1, class_prior=[1.0] * 7).get_config()
L = ConfidencePenaltyLoss.from_config(legacy)
ok(abs(float(L(y, z)) - float(GoodhartAwareLoss.from_config(legacy)(y, z))) < 1e-5, "legacy default config loads into the same objective")
try: ConfidencePenaltyLoss(prior_weight=0.5); ok(False, "prior_weight>0 refused")
except ValueError: ok(True, "prior_weight>0 refused")
try: ConfidencePenaltyLoss(0.0, 0.1, 0.01); ok(False, "legacy positional call refused")
except TypeError: ok(True, "legacy positional call refused")

lab = rng.integers(0, K, 500); zz = rng.normal(size=(500, K)) * 4; zz[np.arange(500), lab] += 3
T = fit_temperature(zz, lab)
Ts = np.exp(minimize_scalar(lambda t: -log_softmax(zz / np.exp(t), 1)[np.arange(500), lab].mean(), bounds=(-4, 4), method="bounded").x)
ok(abs(T - Ts) / Ts < 1e-3, f"fit_temperature matches scipy ({T:.4f} vs {Ts:.4f})")

c = ConfidencePenaltyLoss(0.1, 0.2).get_config()
ok(ConfidencePenaltyLoss.from_config(json.loads(json.dumps(c))).get_config() == c, "JSON round-trip")
L = ConfidencePenaltyLoss(0, 0.1, from_logits=False, dtype="float16")
ok(np.isfinite(float(L(y.astype("float16"), p.astype("float16")))), "float16 probability path finite")
```

### Appendix C: `calibration_experiment.py`

Harness for Section 4. Pass config names as a comma-separated argument.

```python
# Requires the original module saved as goodhart_loss.py (the class under audit).
# Run: KERAS_BACKEND=jax python calibration_experiment.py ce,gal_0.1,new_ls0.1+oe
import os, sys, json, time, numpy as np
os.environ["KERAS_BACKEND"] = "jax"
import keras
from keras import ops
from sklearn.datasets import load_digits
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax
from goodhart_loss import GoodhartAwareLoss
from confidence_penalty_loss import ConfidencePenaltyLoss

K = 10
X, Y = load_digits(return_X_y=True); X = (X / 16.0).astype("float32")

class CE(keras.losses.Loss):  # plain CE on soft/hard/uniform targets
    def call(self, y, z):
        return -ops.sum(ops.cast(y, z.dtype) * ops.log_softmax(z, -1), -1)

class CEplusMMCE(keras.losses.Loss):  # Kumar et al. 2018, weighted-free variant, Laplacian kernel
    def __init__(self, lam=2.0, width=0.4, **kw):
        super().__init__(**kw); self.lam, self.width = lam, width
    def call(self, y, z):
        y = ops.cast(y, z.dtype); lp = ops.log_softmax(z, -1); p = ops.exp(lp)
        ce = -ops.sum(y * lp, -1)
        r = ops.max(p, -1)
        c = ops.cast(ops.equal(ops.argmax(p, -1), ops.argmax(y, -1)), z.dtype)
        k = ops.exp(-ops.abs(r[:, None] - r[None, :]) / self.width)
        d = c - r
        mmce = ops.sqrt(ops.maximum(ops.mean(d[:, None] * d[None, :] * k), 1e-12))
        return ce + self.lam * mmce

def make_model(seed):
    keras.utils.set_random_seed(seed)
    return keras.Sequential([keras.Input((64,)), keras.layers.Dense(512, "relu"),
                             keras.layers.Dense(512, "relu"), keras.layers.Dense(K)])

def ece(p, y, bins=15):
    conf, pred = p.max(1), p.argmax(1); acc = (pred == y)
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return e

def fit_T(z, y):
    f = lambda t: -log_softmax(z / np.exp(t), 1)[np.arange(len(y)), y].mean()
    return float(np.exp(minimize_scalar(f, bounds=(-3, 3), method="bounded").x))

def metrics(z_te, y_te, z_ood, z_blend, T=1.0):
    p = softmax(z_te / T, 1); po = softmax(z_ood / T, 1); pb = softmax(z_blend / T, 1)
    wrong = p.argmax(1) != y_te
    conf_in, conf_ood = p.max(1), po.max(1)
    return dict(
        acc=float((~wrong).mean()),
        nll=float(-np.log(p[np.arange(len(y_te)), y_te] + 1e-12).mean()),
        ece=float(ece(p, y_te)),
        conf_wrong=float(conf_in[wrong].mean()) if wrong.any() else float("nan"),
        err_over_90=float((conf_in[wrong] > 0.9).mean()) if wrong.any() else float("nan"),
        ood_conf=float(conf_ood.mean()),
        ood_auroc=float(roc_auc_score(np.r_[np.ones(len(conf_in)), np.zeros(len(conf_ood))], np.r_[conf_in, conf_ood])),
        blend_conf=float(pb.max(1).mean()),
        err_auroc=float(roc_auc_score((~wrong).astype(int), conf_in)) if wrong.any() else float("nan"),
    )

def run(cfg, seed, n_train=400, n_val=300, epochs=300, batch=32):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X)); tr, va, te = idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]
    Xtr, Ytr = X[tr], np.eye(K, dtype="float32")[Y[tr]]
    sw = np.ones(len(Xtr), "float32")
    if cfg.get("outliers"):  # training outliers: uniform noise images (NOT the test OOD distribution)
        Xo = rng.uniform(0, 1, (n_train, 64)).astype("float32")
        if cfg["outliers"] == "uniform_target":
            Yo = np.full((n_train, K), 1.0 / K, "float32")
        else:  # all-zero rows, as the original class would receive them
            Yo = np.zeros((n_train, K), "float32")
        Xtr = np.r_[Xtr, Xo]; Ytr = np.r_[Ytr, Yo]; sw = np.r_[sw, np.full(n_train, cfg.get("out_w", 1.0), "float32")]
    model = make_model(seed)
    model.compile(keras.optimizers.Adam(1e-3), loss=cfg["loss"]())
    if cfg.get("mixup"):  # fresh mixup pair for every sample of every epoch, materialized up front
        n = len(Xtr) * epochs; a, b = rng.integers(0, len(Xtr), (2, n))
        lam = rng.beta(cfg["mixup"], cfg["mixup"], n).astype("float32")[:, None]
        Xm = lam * Xtr[a] + (1 - lam) * Xtr[b]; Ym = lam * Ytr[a] + (1 - lam) * Ytr[b]
        swm = (lam[:, 0] * sw[a] + (1 - lam[:, 0]) * sw[b]).astype("float32")
        model.fit(Xm, Ym, sample_weight=swm, epochs=1, batch_size=batch, verbose=0, shuffle=False)
    else:
        model.fit(Xtr, Ytr, sample_weight=sw, epochs=epochs, batch_size=batch, verbose=0, shuffle=True)
    Xte, yte = X[te], Y[te]
    Xood = np.stack([x[rng.permutation(64)] for x in Xte])            # pixel-permuted digits: no class evidence
    a, b = rng.integers(0, len(Xte), (2, 2000)); keep = yte[a] != yte[b]
    Xbl = 0.5 * Xte[a[keep]] + 0.5 * Xte[b[keep]]                      # 50/50 blends: ambiguous evidence
    pr = lambda A: np.asarray(model.predict(A, verbose=0), "float64")
    z_va, z_te, z_ood, z_bl = pr(X[va]), pr(Xte), pr(Xood), pr(Xbl)
    T = fit_T(z_va, Y[va])
    tr_acc = float((pr(X[tr]).argmax(1) == Y[tr]).mean())
    return dict(raw=metrics(z_te, yte, z_ood, z_bl), ts=metrics(z_te, yte, z_ood, z_bl, T), T=T, train_acc=tr_acc)

CONFIGS = {
    "ce":            dict(loss=lambda: CE()),
    "gal_0.1":       dict(loss=lambda: GoodhartAwareLoss(0.0, 0.1)),
    "gal_0.3":       dict(loss=lambda: GoodhartAwareLoss(0.0, 0.3)),
    "ls_0.1":        dict(loss=lambda: GoodhartAwareLoss(0.1, 0.0)),
    "ce+mmce":       dict(loss=lambda: CEplusMMCE(2.0)),
    "gal_0.1+zero_rows": dict(loss=lambda: GoodhartAwareLoss(0.0, 0.1), outliers="zero"),
    "ce+oe_uniform": dict(loss=lambda: CE(), outliers="uniform_target", out_w=0.5),
    "ce+mixup":      dict(loss=lambda: CE(), mixup=0.4),
    "new_cp0.3+oe":  dict(loss=lambda: ConfidencePenaltyLoss(0.0, 0.3), outliers="uniform_target", out_w=0.5),
    "new_ls0.1+oe":  dict(loss=lambda: ConfidencePenaltyLoss(0.1, 0.0), outliers="uniform_target", out_w=0.5),
    "ce+mixup+oe":   dict(loss=lambda: CE(), mixup=0.4, outliers="uniform_target", out_w=0.5),
}

if __name__ == "__main__":
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else list(CONFIGS)
    seeds = range(5)
    out = {}
    for n in names:
        t = time.time()
        out[n] = [run(CONFIGS[n], s) for s in seeds]
        print(n, f"{time.time()-t:.0f}s", flush=True)
    json.dump(out, open(f"results_{'_'.join(names)[:60]}.json", "w"))
```

### Appendix D: `prior_experiment.py`

Harness for Table 2.1.

```python
import os, json, numpy as np
os.environ["KERAS_BACKEND"] = "jax"
import calibration_experiment as E
from goodhart_loss import GoodhartAwareLoss
out = {}
for batch, epochs in [(8, 80), (32, 300), (256, 1500)]:
    for beta in [0.0, 1.0]:
        cfg = dict(loss=lambda b=beta: GoodhartAwareLoss(0.0, 0.0, prior_weight=b))
        rs = [E.run(cfg, s, epochs=epochs, batch=batch) for s in range(3)]
        out[f"B={batch} beta={beta}"] = rs
        m = lambda k: np.mean([r["raw"][k] for r in rs])
        print(f"B={batch:<4} beta={beta}: acc={m('acc'):.4f} nll={m('nll'):.4f} ece={m('ece'):.4f} ood_conf={m('ood_conf'):.3f}", flush=True)
json.dump(out, open("results_prior.json", "w"))
```

### Appendix E: `unit_checks.py`

Contract checks of the original class (Section 2).

```python
import numpy as np, keras, warnings, json
from goodhart_loss import GoodhartAwareLoss, analyze_loss_components
from scipy.optimize import minimize
from scipy.special import softmax, log_softmax, logsumexp
rng = np.random.default_rng(0)
K, B = 10, 64
z = rng.normal(size=(B, K)).astype("float32") * 3
y = np.eye(K, dtype="float32")[rng.integers(0, K, B)]

def ref(z, y, lam, beta, ls=0.0):
    lp = log_softmax(z.astype("float64"), -1); p = np.exp(lp)
    yt = y * (1 - ls) + ls / K
    ce = -(yt * lp).sum(-1); H = -(p * lp).sum(-1)
    pm = p.mean(0); kl = np.sum((1/K) * (np.log(1/K) - np.log(pm)))
    return np.mean(ce - lam * H + beta * kl)

print("[1] formula vs float64 numpy reference")
for lam, beta, ls in [(0.1, 0, 0), (0.3, 1.0, 0.1)]:
    L = GoodhartAwareLoss(ls, lam, prior_weight=beta)
    print(f"  lam={lam} beta={beta} ls={ls}: keras={float(L(y, z)):.6f} ref={ref(z, y, lam, beta, ls):.6f}")

print("[2] integer labels")
for shp in [(B,), (B, 1)]:
    try:
        GoodhartAwareLoss()(np.zeros(shp, "float32"), z); print(f"  {shp}: ACCEPTED (silent)")
    except Exception as e:
        print(f"  {shp}: raises {type(e).__name__}")

print("[3] numpy scalar weights")
for v in [np.float64(0.1), np.float32(0.1), True, float('inf')]:
    try:
        GoodhartAwareLoss(entropy_weight=v); print(f"  {type(v).__name__}({v}): accepted")
    except Exception as e:
        print(f"  {type(v).__name__}({v}): REJECTED -> {e}")

print("[4] diagnostic vs live loss, wrong-length prior, prior_weight=0")
L = GoodhartAwareLoss(class_prior=[1.0]*7)
print("  live loss:", float(L(y, z)))
try:
    analyze_loss_components(L, y, z); print("  diagnostic: ok")
except Exception as e:
    print("  diagnostic: RAISES ->", e)

print("[5] sample_weight scales the batch-level prior term")
L = GoodhartAwareLoss(0.0, 0.0, prior_weight=1.0)
L0 = GoodhartAwareLoss(0.0, 0.0)
kl = float(L(y, z)) - float(L0(y, z))
for w in [1.0, 0.5, 0.0]:
    sw = np.full(B, w, "float32")
    d = float(L(y, z, sample_weight=sw)) - float(L0(y, z, sample_weight=sw))
    print(f"  all weights={w}: effective prior contribution={d:.4f} (unweighted KL={kl:.4f})")

print("[6] analyze_loss_components 'total_loss == live loss' claim")
for red in ["sum_over_batch_size", "sum"]:
    L = GoodhartAwareLoss(0.0, 0.1, reduction=red)
    print(f"  reduction={red}: live={float(L(y, z)):.4f} diag={analyze_loss_components(L, y, z)['total_loss']:.4f}")

print("[7] 'one producer of log p' claim, from_logits=False")
p = softmax(z.astype("float64"), -1).astype("float32"); p[:, 0] = 0.0; p /= p.sum(-1, keepdims=True)
L = GoodhartAwareLoss(0.0, 0.0, from_logits=False)
lp = np.asarray(L._log_probabilities(keras.ops.convert_to_tensor(p)))
print(f"  keras CE={float(L(y, p)):.6f} CE from shared log p={np.mean(-(y*lp).sum(-1)):.6f}")
for dt in ["float16"]:
    L = GoodhartAwareLoss(0.0, 0.1, from_logits=False, dtype=dt)
    print(f"  {dt} probs w/ zeros: loss={float(L(y.astype(dt), p.astype(dt)))}")

print("[8] JSON round-trip")
L = GoodhartAwareLoss(class_prior=np.array([1, 2, 3], dtype="float32"), prior_weight=0.5)
print("  ", GoodhartAwareLoss.from_config(json.loads(json.dumps(L.get_config()))).get_config()['class_prior'])
```

### Appendix F: `theory_checks.py`

Margin cap, properness and prior-term bias (Section 3, finding 2).

```python
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax, log_softmax

def argmin_p(q, lam, ls=0.0):
    K = len(q); qt = q*(1-ls) + ls/K
    f = lambda z: -(qt*log_softmax(z)).sum() + lam*(softmax(z)*log_softmax(z)).sum()
    z = minimize(f, np.zeros(K), method="BFGS", options={"gtol":1e-12}).x
    return softmax(z)

K = 10
print("[A] hard label: optimal confidence and logit margin under each regularizer")
for lam in [0.01, 0.1, 0.3, 0.5]:
    p = argmin_p(np.eye(K)[0], lam)
    print(f"  conf-penalty lam={lam:<4}: p_y={p[0]:.6f}  margin={np.log(p[0]/p[1]):6.2f}  1/lam={1/lam:6.2f}")
for ls in [0.05, 0.1]:
    p = argmin_p(np.eye(K)[0], 0.0, ls)
    print(f"  label-smooth eps={ls:<4}: p_y={p[0]:.6f}  margin={np.log(p[0]/p[1]):6.2f}")

print("[B] properness: true conditional q, does argmin E_q[loss] recover q?")
for q in [np.array([0.7,0.2,0.1]+[0]*7)+1e-9, np.array([0.5,0.5]+[0]*8)+1e-9]:
    q = q/q.sum()
    for lam in [0.0, 0.1, 0.5]:
        p = argmin_p(q, lam)
        print(f"  q={np.round(q[:3],2)} lam={lam}: p*={np.round(p[:3],3)} rest={p[3:].sum():.3f}")

print("[C] prior term on a PERFECT, well-calibrated-on-train classifier (p = one-hot-ish, conf .99)")
rng = np.random.default_rng(0)
for K in [10, 100]:
    for B in [8, 32, 128, 1024]:
        vals = []
        for _ in range(500):
            y = rng.integers(0, K, B)
            P = np.full((B, K), 0.01/(K-1)); P[np.arange(B), y] = 0.99
            pm = P.mean(0); vals.append(np.sum((1/K)*(np.log(1/K) - np.log(pm))))
        print(f"  K={K:<3} B={B:<5}: mean KL(u||p_bar) = {np.mean(vals):.3f}   (CE of same model = {-np.log(0.99):.3f})")
```

### Appendix G: `summarize.py`

Aggregates `results_*.json` into the Section 4.3 table.

```python
import json, glob, numpy as np
R = {}
for f in sorted(glob.glob("results_*.json")):
    if "prior" in f: continue
    R.update(json.load(open(f)))
keys = ["acc", "nll", "ece", "err_auroc", "err_over_90", "ood_conf", "ood_auroc", "blend_conf"]
order = ["ce", "gal_0.1", "gal_0.3", "ls_0.1", "ce+mmce", "gal_0.1+zero_rows", "ce+oe_uniform", "ce+mixup", "ce+mixup+oe", "new_cp0.3+oe", "new_ls0.1+oe"]
base = {mode: {k: np.array([r[mode][k] for r in R["ce"]]) for k in keys} for mode in ["raw", "ts"]}
for mode in ["raw", "ts"]:
    print(f"\n== {mode} ==   (mean over 5 seeds; paired t vs ce/{mode}; |t| > 2.78 is p < .05 at 5 seeds)")
    print(f"{'config':<20}" + "".join(f"{k:>18}" for k in keys) + f"{'T':>7}")
    for n in [o for o in order if o in R]:
        rs = R[n]; row = f"{n:<20}"
        for k in keys:
            v = np.array([r[mode][k] for r in rs]); d = v - base[mode][k]
            z = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12) if n != "ce" else 0
            row += f"{v.mean():>10.4f} [{z:+5.1f}]"
        row += f"{np.mean([r['T'] for r in rs]):>7.2f}"
        print(row)
print("\ntrain acc:", {n: np.mean([r['train_acc'] for r in R[n]]) for n in order if n in R})
```

### Appendix H: `goodhart_loss.py`

The original class under audit, code unchanged, docstrings and comments removed.

```python
import keras
import warnings
import numpy as np
from typing import Any, Dict, Optional, Sequence

from dl_techniques.utils.keras_registration import register_dl_technique


@register_dl_technique("dl_techniques.losses.goodhart_loss")
class GoodhartAwareLoss(keras.losses.Loss):

    def __init__(
        self,
        label_smoothing: float = 0.0,
        entropy_weight: float = 0.1,
        *,
        prior_weight: float = 0.0,
        class_prior: Optional[Sequence[float]] = None,
        from_logits: bool = True,
        epsilon: float = 1e-8,
        name: str = 'goodhart_aware_loss',
        reduction: str = 'sum_over_batch_size',
        dtype: Optional[str] = None,
        mi_weight: Optional[float] = None
    ) -> None:
        if mi_weight is not None:
            raise ValueError(
                "mi_weight has been removed from GoodhartAwareLoss "
                f"(got mi_weight={mi_weight}). The term it weighted was "
                "H(mean_p) - mean H(p_i) added with a POSITIVE sign, so it was "
                "maximized at the accurate, confident classifier and minimized "
                "at marginal collapse -- the opposite of the intended effect. "
                "Use prior_weight (with the optional class_prior) for a "
                "correctly-signed anti-collapse term."
            )
        if not (0 <= label_smoothing < 1):
            raise ValueError(
                f"label_smoothing must be in the range [0, 1), "
                f"but got {label_smoothing}."
            )
        if not (isinstance(entropy_weight, (int, float)) and entropy_weight >= 0):
            raise ValueError(
                f"Entropy weight must be a non-negative number, "
                f"but got {entropy_weight}."
            )
        if not (isinstance(prior_weight, (int, float)) and prior_weight >= 0):
            raise ValueError(
                f"Prior weight must be a non-negative number, "
                f"but got {prior_weight}."
            )
        if not (0 < epsilon < 0.1):
            raise ValueError(
                f"Epsilon must be a small positive number in (0, 0.1), "
                f"but got {epsilon}."
            )

        normalized_prior = None
        if class_prior is not None:
            prior_array = np.asarray(class_prior, dtype="float64").reshape(-1)
            if prior_array.size == 0:
                raise ValueError("class_prior must be a non-empty sequence.")
            if not np.all(np.isfinite(prior_array)):
                raise ValueError(
                    f"class_prior entries must all be finite, but got {class_prior}."
                )
            if not np.all(prior_array > 0):
                raise ValueError(
                    f"class_prior entries must all be strictly positive, "
                    f"but got {class_prior}."
                )
            normalized_prior = prior_array / float(prior_array.sum())

        if entropy_weight > 0.5:
            warnings.warn(
                f"High entropy_weight ({entropy_weight}) may dominate training. "
                "Consider values in [0.001, 0.5].", UserWarning
            )

        super().__init__(name=name, reduction=reduction, dtype=dtype)

        self.label_smoothing = float(label_smoothing)
        self.entropy_weight = float(entropy_weight)
        self.prior_weight = float(prior_weight)
        self.class_prior = (
            None if class_prior is None
            else [float(v) for v in np.asarray(class_prior).reshape(-1)]
        )
        self.from_logits = bool(from_logits)
        self.epsilon = float(epsilon)
        self._dtype_arg = dtype
        self._normalized_prior = normalized_prior

    def call(self, y_true, y_pred):
        y_true = keras.ops.cast(y_true, dtype=y_pred.dtype)
        per_sample_loss = keras.losses.categorical_crossentropy(
            y_true=y_true,
            y_pred=y_pred,
            from_logits=self.from_logits,
            label_smoothing=self.label_smoothing
        )
        log_probs = self._log_probabilities(y_pred)
        if self.entropy_weight > 0:
            per_sample_loss = per_sample_loss - (
                self.entropy_weight * self._conditional_entropy(log_probs)
            )
        if self.prior_weight > 0:
            per_sample_loss = per_sample_loss + (
                self.prior_weight * self._prior_matching_regularization(log_probs)
            )
        return per_sample_loss

    def _log_probabilities(self, y_pred):
        if self.from_logits:
            return keras.ops.log_softmax(y_pred, axis=-1)
        dtype = getattr(y_pred.dtype, "name", None) or str(y_pred.dtype)
        floor = max(self.epsilon, float(np.finfo(dtype).tiny))
        probs = keras.ops.clip(y_pred, floor, 1.0)
        probs = probs / keras.ops.sum(probs, axis=-1, keepdims=True)
        return keras.ops.log(probs)

    def _conditional_entropy(self, log_probs):
        probs = keras.ops.exp(log_probs)
        return -keras.ops.sum(probs * log_probs, axis=-1)

    def _prior_matching_regularization(self, log_probs):
        compute_dtype = log_probs.dtype
        batch_size = keras.ops.cast(keras.ops.shape(log_probs)[0], compute_dtype)
        log_mean_probs = keras.ops.logsumexp(
            log_probs, axis=0
        ) - keras.ops.log(batch_size)
        if self._normalized_prior is None:
            num_classes = keras.ops.cast(
                keras.ops.shape(log_probs)[-1], compute_dtype
            )
            return -keras.ops.log(num_classes) - keras.ops.mean(log_mean_probs)
        static_classes = log_probs.shape[-1]
        if static_classes is not None and static_classes != self._normalized_prior.size:
            raise ValueError(
                f"class_prior has length {self._normalized_prior.size} but "
                f"y_pred has {static_classes} classes."
            )
        prior = keras.ops.convert_to_tensor(
            self._normalized_prior, dtype=compute_dtype
        )
        log_prior = keras.ops.convert_to_tensor(
            np.log(self._normalized_prior), dtype=compute_dtype
        )
        return keras.ops.sum(prior * (log_prior - log_mean_probs))

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'label_smoothing': self.label_smoothing,
            'entropy_weight': self.entropy_weight,
            'prior_weight': self.prior_weight,
            'class_prior': self.class_prior,
            'from_logits': self.from_logits,
            'epsilon': self.epsilon,
            'dtype': self._dtype_arg,
        })
        return config


def analyze_loss_components(loss_fn, y_true, y_pred):
    y_true = keras.ops.cast(y_true, dtype=y_pred.dtype)
    ce_loss = keras.losses.categorical_crossentropy(
        y_true=y_true,
        y_pred=y_pred,
        from_logits=loss_fn.from_logits,
        label_smoothing=loss_fn.label_smoothing
    )
    log_probs = loss_fn._log_probabilities(y_pred)
    entropy_term_unweighted = -keras.ops.mean(loss_fn._conditional_entropy(log_probs))
    prior_term_unweighted = loss_fn._prior_matching_regularization(log_probs)
    entropy_term_weighted = loss_fn.entropy_weight * entropy_term_unweighted
    prior_term_weighted = loss_fn.prior_weight * prior_term_unweighted
    total_loss = keras.ops.mean(ce_loss + entropy_term_weighted + prior_term_weighted)
    results = {
        'total_loss': float(keras.ops.convert_to_numpy(total_loss)),
        'cross_entropy': float(keras.ops.convert_to_numpy(keras.ops.mean(ce_loss))),
        'entropy_term_unweighted': float(keras.ops.convert_to_numpy(entropy_term_unweighted)),
        'prior_term_unweighted': float(keras.ops.convert_to_numpy(prior_term_unweighted)),
        'entropy_term_weighted': float(keras.ops.convert_to_numpy(entropy_term_weighted)),
        'prior_term_weighted': float(keras.ops.convert_to_numpy(prior_term_weighted)),
    }
    return results
```

### Appendix I: `dl_techniques/utils/keras_registration.py`

No-op stub for the registration decorator. Not needed inside the real `dl_techniques` package.

```python
def register_dl_technique(name):
    def deco(cls):
        return cls
    return deco
```