# Comprehensive Review: `src/dl_techniques/losses/`

**Date:** 2026-10-19  
**Reviewer:** AI Agent  
**Scope:** Full package review of 52 loss modules

---

## Executive Summary

The `losses` package is **feature-rich but carries significant technical debt**. It contains 52 modules implementing specialized loss functions for Keras 3/TensorFlow 2.18. While many newer modules (`infonce_loss`, `colbert_loss`, `multistep_loss`, `dino_loss`, `goodhart_loss`) demonstrate excellent patterns, the package as a whole has:

- **14+ classes still returning scalar from `call()`** (the "premature scalar" family), silently breaking `sample_weight`
- **Inconsistent batch-coupling handling** — some losses decompose per-row, others are irreducibly batch-level
- **Two competing docstring styles** (Google and Sphinx) with no package-wide convention
- **Legacy import patterns** (`from keras import ops` in 31 of 52 modules)
- **Redundant/overlapping implementations** (e.g., `SegmentationLosses` vs `SegmentationWrapperLoss` vs `AnyLoss` segmentation metrics)
- **Missing critical documentation** on batch-coupling semantics for users

---

## 1. Critical Bugs / Silent Failures

### 1.1 The "Premature Scalar" Family (14+ classes)
**Status:** Tracked in `tests/test_losses/test_the_premature_scalar_family_is_pinned.py`

Classes whose `call()` returns a **scalar instead of `(batch,)`**, causing `sample_weight` to broadcast incorrectly:

| Loss Module | Class | Status |
|-------------|-------|--------|
| `dino_loss.py` | `DINOLoss` | Fixed but batch-coupled (centering EMA) |
| `dino_loss.py` | `KoLeoLoss` | Fixed but batch-coupled (nearest neighbors) |
| `scaled_mse_loss.py` | `ScaledMseLoss` | ✅ Fixed |
| `sam_mask_loss.py` | `SAMIoULoss` | ✅ Fixed |
| `feature_alignment_loss.py` | `FeatureAlignmentLoss` | ✅ Fixed |
| `quantile_loss.py` | `MQLoss`, `QuantileLoss` | ✅ Fixed |
| `affine_invariant_loss.py` | `AffineInvariantLoss` | ✅ Fixed |
| `smape_loss.py` | `SMAPELoss` | ✅ Fixed |
| `masked_causal_lm_loss.py` | `MaskedCausalLMLoss`, `PrefixMaskedCausalLMLoss` | ✅ Fixed (token-pool form) |
| `focal_causal_lm_loss.py` | `FocalCausalLMLoss` | ✅ Fixed (token-pool) |
| `mase_loss.py` | `MASELoss` | ✅ Fixed (batch-global scaling) |
| `hrm_loss.py` | `HRMLoss` | ✅ Fixed (token-pool + BCE reshape) |
| `brier_spiegelhalters_ztest_loss.py` | `BrierScoreLoss`, `SpiegelhalterZLoss`, `CombinedCalibrationLoss` | ✅ Fixed (per-row Z decomposition) |
| `mambalct_loss.py` | `MambaLCTBoxLoss` | ✅ Fixed |

**Impact:** `loss(w=[1,1,1,0]) * B/(B-1) != loss(batch[:-1])` — weighted rows are charged the batch aggregate. The failure is **silent**; a plausible wrong number, not an error.

### 1.2 Batch-Coupled Losses (Deliberate but Poorly Documented)
These losses return `(batch,)` but contain **batch-level terms** added to every row:

| Loss | Batch Coupling Source | Documented? |
|------|----------------------|-------------|
| `DINOLoss` | Centering EMA (in `call()`) | ✅ in docstring |
| `KoLeoLoss` | Nearest-neighbor graph | ❌ |
| `GoodhartAwareLoss` | `prior_weight * KL(q \|\| mean_p)` | ✅ explicit warning |
| `HRMLoss` | Token-pool denominator | ⚠️ partial |
| `MASELoss` | Batch-global MAE scaling | ⚠️ partial |
| `iBOTPatchLoss` | Centering EMA (in `call()`) | ✅ in docstring |

**User risk:** `sample_weight=0` removes loss *contribution* but not *influence* on other rows.

### 1.3 `infonce_loss.py` Batch-Size-1 Guard Is Partial
```python
# Line 184-198: REJECTS statically-known batch=1, but...
batch = view_a.shape[0]
if batch is not None and int(batch) == 1:
    raise ValueError(...)
```
**Gap:** Under `fit()`, batch axis is `None` (dynamic), so the guard **cannot fire** at runtime. A trailing batch of 1 produces `1x1` logit matrix → softmax = 1.0 → loss = 0.0 → zero gradient.  
**Fix:** Require `drop_remainder=True` in docs + add runtime assert.

---

## 2. API Inconsistencies & Design Gaps

### 2.1 Inconsistent `from_logits` Defaults
| Module | Default | Problem |
|--------|---------|---------|
| `huber_loss.py` | N/A (regression) | — |
| `goodhart_loss.py` | `True` | Reasonable |
| `focal_uncertainty_loss.py` | `True` | Reasonable |
| `segmentation_loss.py` | `False` | **Inconsistent** — most modern losses use `True` |
| `any_loss.py` | `False` | **Inconsistent** — approximates metrics from probs |
| `multistep_loss.py` | N/A (regression) | — |
| `dino_loss.py` | N/A (logits handled internally) | — |
| `infonce_loss.py` | N/A (expects normalized embeddings) | — |

**Recommendation:** Standardize on `from_logits=True` as default for classification losses; document clearly.

### 2.2 Segmentation Loss Redundancy (3 overlapping implementations)
```
segmentation_loss.py      → SegmentationLosses (scalar batch mean, SAM-compat)
segmentation_wrapper_loss.py → SegmentationWrapperLoss (per-sample, serializable) 
any_loss.py               → DiceLoss, IoULoss, TverskyLoss, FocalTverskyLoss (AnyLoss framework)
```
**Issues:**
- `SegmentationLosses` returns **scalar batch mean** (not per-sample) for SAM compatibility
- `SegmentationWrapperLoss` wraps it but delegates to scalar methods
- `AnyLoss` versions use **differentiable confusion matrix** (completely different math)
- No clear guidance on which to use when

### 2.3 `MultiTaskLoss` vs `YOLOv12MultiTaskLoss` vs `SegmentationWrapperLoss`
Three different patterns for multi-output models:
- `MultiTaskLoss` — simple weighted sum by output name
- `YOLOv12MultiTaskLoss` — shape-based dispatch + learnable uncertainty weights
- `SegmentationWrapperLoss` — name-dispatched single loss

**Gap:** No unified multi-task loss abstraction; each reinvents dispatch/weighting.

### 2.4 `create_*_loss` Factory Signatures Inconsistent
```python
create_segmentation_wrapper_loss(loss_name, config)
create_multistep_loss(name, **kwargs)           # strict kwarg validation
create_symmetric_infonce_loss(temperature=..., **kwargs)
create_yolov12_multitask_loss(tasks, num_detection_classes=..., **kwargs)
create_siglip_loss(...) / create_adaptive_siglip_loss(...) / create_hybrid_loss(...)
create_hrm_loss(...)
```
Some validate unknown kwargs (multistep), some don't; some have positional args, some keyword-only.

---

## 3. Registration & Serialization Issues

### 3.1 Legacy Package Keys (Migration Incomplete)
`yolo12_multitask_loss.py` registers 4 classes with `legacy_packages=("yolov12_losses",)`:
```python
@register_dl_technique(
    "dl_techniques.losses.yolo12_multitask_loss", legacy_packages=("yolov12_losses",)
)
class YOLOv12ObjectDetectionLoss(...)
```
**Risk:** If `legacy_packages` is dropped, **all saved models using these losses break silently**.

### 3.2 Non-Registering Modules (2 of 52)
Per `AGENTS.md`: "50 of 52 modules register something; the non-registering ones export plain functions only."
- `jacobian_symmetry.py` — exports `jacobian_symmetry_penalty`, `thera_tv_penalty` (functions, not Loss)
- `thera_jacobian_tv.py` — exports `thera_tv_penalty`, `thera_total_loss` (functions)

**Gap:** These are not importable from `dl_techniques.losses` — must import from module directly. Not documented in `__init__.py` or README.

### 3.3 Config Serialization of State-Holding Losses
`DINOLoss`, `iBOTPatchLoss`, `TopographicVAELoss` serialize internal state (`center` variable) in `get_config()`. This works but:
- Config size scales with `out_dim` (~1.3 MiB JSON at 65536)
- No versioning — if state structure changes, old configs break
- `WassersteinGradientPenaltyLoss` wraps `WassersteinLoss` but doesn't serialize it properly

---

## 4. Import & Style Inconsistencies

### 4.1 Import Pattern Split (31 vs 21 modules)
```python
# 31 modules use (legacy):
from keras import ops
ops.matmul(...)

# 21 modules use (preferred):
import keras
keras.ops.matmul(...)
```
Per AGENTS.md: "31 of 52 existing modules use the superseded `from keras import ops`; that majority is **neither a pattern to extend nor a migration target** — leave them alone."

**Problem:** New contributors see mixed style and don't know which to follow.

### 4.2 Docstring Style Split (38 Google vs 14 Sphinx vs 2 both)
| Style | Count | Examples |
|-------|-------|----------|
| Google (`Args:`) | 38 | `huber_loss.py`, `goodhart_loss.py`, `focal_uncertainty_loss.py` |
| Sphinx (`:param:`) | 14 | `infonce_loss.py`, `colbert_loss.py`, `multistep_loss.py` |
| Both | 2 | — |

**Rule:** "Match the file you are editing; never convert a file wholesale."

---

## 5. Refactoring Opportunities

### 5.1 Unify Per-Sample Return Pattern
Create a **base class** or **mixin** enforcing `(batch,)` return:
```python
class PerSampleLoss(keras.losses.Loss):
    """Base class guaranteeing call() returns (batch,)."""
    def call(self, y_true, y_pred):
        result = self._call_impl(y_true, y_pred)
        # Assert shape at trace time
        result = keras.ops.convert_to_tensor(result)
        keras.ops.assert_equal(keras.ops.rank(result), 1)
        return result
```

### 5.2 Standardize Batch-Coupling Annotation
Add a class attribute declaring coupling behavior:
```python
class GoodhartAwareLoss(keras.losses.Loss):
    BATCH_COUPLING = "prior_weight"  # or "centering_ema", "token_pool", "none"
    
    @property
    def is_batch_coupled(self) -> bool:
        return self.prior_weight > 0
```

### 5.3 Consolidate Segmentation Losses
**Proposal:** Deprecate `SegmentationLosses` (scalar-returning) in favor of:
1. `SegmentationWrapperLoss` for serializable single losses
2. `AnyLoss` framework for differentiable metric optimization
3. Remove duplicate `DiceLoss`/`IoULoss` from `AnyLoss` (they're not differentiable approximations)

### 5.4 Create Multi-Task Loss Base Class
Extract common pattern from `YOLOv12MultiTaskLoss` and `MultiTaskLoss`:
```python
class MultiTaskLossBase(keras.losses.Loss):
    def _route_by_shape(self, y_true, y_pred): ...
    def _apply_weighting(self, task_name, raw_loss): ...
```

### 5.5 Add `sample_weight` Contract Tests Per Loss
Each loss should have a test asserting:
```python
def test_sample_weight_decomposes(loss_fn, y_true, y_pred):
    weighted = loss_fn(y_true, y_pred, sample_weight=[1,1,1,0])
    dropped = loss_fn(y_true[:3], y_pred[:3])
    assert weighted * 4/3 == dropped  # or document why not
```

---

## 6. Missing Losses / Feature Gaps

### 6.1 No Focal Loss for Multi-Label (in main exports)
`multi_labels_loss.py` has `WeightedBinaryFocalLoss` but it's not in `__init__.py` exports.

### 6.2 No PolyLoss / GHM-C / GHM-R
Modern classification losses missing from the collection.

### 6.3 No Distillation Losses (except `FeatureAlignmentLoss` and `ColBERTDistillationLoss`)
Knowledge distillation losses (KD, PKD, CRD, etc.) not centralized.

### 6.4 No Contrastive Loss for Audio/Tabular
`CLIPContrastiveLoss`, `SigLIPContrastiveLoss` are vision-language; no generic contrastive for other modalities.

### 6.5 No Proper Scoring Rule for Regression
`QuantileLoss`, `MASELoss`, `SMAPELoss` exist but no **CRPS** (Continuous Ranked Probability Score) or **Energy Score** for probabilistic regression.

---

## 7. Documentation Gaps

### 7.1 Batch-Coupling Not Documented in README
The README table shows loss names but **never mentions** which losses are batch-coupled and what that means for `sample_weight`.

### 7.2 `from_logits` Default Not in Table
The "Available Losses" table doesn't show `from_logits` default for each loss.

### 7.3 `reduction=` Behavior Not Explained
Users don't know that `reduction=` is a **dead knob** for scalar-returning losses.

### 7.4 No Migration Guide for Legacy Losses
`create_loss_function` in `segmentation_loss.py` is deprecated but no migration path shown in docs.

---

## 8. Test Coverage Gaps

| Area | Coverage | Gap |
|------|----------|-----|
| `sample_weight` decomposition | Only `goodhart_loss`, `premature_scalar` test | Not per-loss |
| Serialization round-trip | Only `goodhart_loss`, `dino_loss`, `segmentation_wrapper` | Most losses untested |
| `dtype` policies | Only `goodhart_loss`, `wasserstein_loss` | Most losses untested |
| Gradient correctness | Only `goodhart_loss` (entropy grad) | Most losses untested |
| Batch-coupling behavior | Only `goodhart_loss`, `dino_loss`, `multistep_loss` | Others assumed |

---

## 9. Priority Action Items

### P0 (Critical — Silent Wrong Results)
1. **Fix remaining premature scalar losses** — move all 14 from `STILL_BROKEN` to `KNOWN_GOOD`
2. **Document batch-coupling** in every affected loss's docstring + README
3. **Add per-loss `sample_weight` contract test** for all public losses

### P1 (High — API Consistency)
4. **Standardize `from_logits` default** to `True` for classification losses
5. **Consolidate segmentation losses** — pick one canonical implementation
6. **Unify factory function signatures** with strict kwarg validation

### P2 (Medium — Maintainability)
7. **Create `PerSampleLoss` base class** for shape enforcement
8. **Add batch-coupling metadata** to loss classes
9. **Document `legacy_packages` requirement** for YOLOv12 losses

### P3 (Low — Polish)
10. **Pick one docstring style** for new files (Sphinx preferred per `models/language/bert/`)
11. **Add missing losses** (CRPS, PolyLoss, generic contrastive)
12. **Create multi-task loss abstraction**

---

## 10. Verdict

The `losses` package is **production-functional for its intended use cases** (the trainer scripts in `src/train/` use these losses successfully), but it has **accumulated debt that will bite users who:**
- Use `sample_weight` / `class_weight` with batch-coupled losses
- Expect `reduction=` to work on scalar-returning losses
- Try to serialize/deserialize models with state-holding losses
- Mix losses from different sub-frameworks (AnyLoss vs Segmentation vs native Keras)

**Recommendation:** Treat this as a **mature but debt-laden package** — fix the P0 items before encouraging broad external adoption, then incrementally address P1-P3.