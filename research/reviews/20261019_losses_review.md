# Comprehensive Review: `src/dl_techniques/losses/`

**Date:** 2026-10-19  
**Reviewer:** AI Agent  
**Scope:** Full package review of 52 loss modules

---

## Executive Summary

The `losses` package is **feature-rich but carries significant technical debt**. It contains 52 modules implementing specialized loss functions for Keras 3/TensorFlow 2.18. While many newer modules (`infonce_loss`, `colbert_loss`, `multistep_loss`, `dino_loss`, `goodhart_loss`, `brier_spiegelhalters_ztest_loss`, `ddpm_hybrid_loss`, `doc_scanner_flow_sequence_loss`, `sam3_detection_loss`) demonstrate excellent patterns with rigorous documentation of design decisions and measurement-backed invariants, the package as a whole has:

- **14+ classes still returning scalar from `call()`** (the "premature scalar" family), silently breaking `sample_weight`
- **Inconsistent batch-coupling handling** — some losses decompose per-row, others are irreducibly batch-level
- **Two competing docstring styles** (Google and Sphinx) with no package-wide convention
- **Legacy import patterns** (`from keras import ops` in 31 of 52 modules)
- **Redundant/overlapping implementations** (e.g., `SegmentationLosses` vs `SegmentationWrapperLoss` vs `AnyLoss` segmentation metrics)
- **Missing critical documentation** on batch-coupling semantics for users
- **Several non-`Loss` callables** exported from `__init__.py` that break the Keras `Loss` API contract

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
| `iBOTPatchLoss` | Centering EMA (in `call()`) | ✅ in docstring |
| `GoodhartAwareLoss` | `prior_weight * KL(q \|\| mean_p)` | ✅ explicit warning |
| `HRMLoss` | Token-pool denominator | ⚠️ partial |
| `MASELoss` | Batch-global MAE scaling | ⚠️ partial |
| `MultistepLoss` (`gtmse`) | Log of batch-mean error | ✅ in docstring |
| `DecoupledInformationLoss` | `diversity_weight * H(marginal)` | ✅ in docstring |
| `FocalUncertaintyLoss` | None (per-sample) | ✅ per-sample |
| `SigLIPContrastiveLoss` / `AdaptiveSigLIPLoss` / `HybridContrastiveLoss` | Pairwise logits (mean over axis) | ✅ per-sample |
| `CLIPContrastiveLoss` | None (per-sample) | ✅ per-sample |
| `SymmetricInfoNCELoss` | None (per-sample) | ✅ per-sample |
| `ColBERTPairwiseSoftmaxLoss` / `ColBERTDistillationLoss` | Per-group | ✅ per-sample |
| `DDPMHybridLoss` | None (per-sample) | ✅ per-sample |
| `DocScannerFlowSequenceLoss` | None (per-sample) | ✅ per-sample |
| `SAM3DetectionLoss` | Hungarian matcher + box IoU | ✅ per-sample (scalar return) |

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

### 1.4 `SAMMaskLoss` / `SAM2GatedMaskLoss` Gradient Death on Saturated Logits
**Root cause:** `SegmentationLosses.focal_loss` opens with `ops.clip(y_pred, 1e-7, 1-1e-7)` on **probabilities**. `ops.clip` has **exactly zero derivative** outside its range.  
**Measured consequence:** 95.46% of GT-present mask pixels at `sigmoid < 1e-7` → mask term gradient norm `3.1751e-13` vs object-score BCE `4.4969e+00` — 13 orders of magnitude dead.  
**Fix in `sam2_video_loss.py`:** `_focal_from_logits` computes BCE-with-logits in logit space via `softplus(-x) + x*(1-t)` — no clip, gradient exact at all logits.  
**NOT backported to `sam_mask_loss.py`** — SAM 1's 357-test gate protects it.

### 1.5 `utilization_loss.py` Returns Scalar, Not Per-Sample
Both `MANNUtilizationLoss` and `GNNUtilizationLoss` return **scalar** from `call()`:
```python
# MANNUtilizationLoss.call() line 101-105
return (
    self.entropy_weight * entropy_loss   # scalar
    + self.variance_weight * variance_loss  # scalar
    + self.write_weight * magnitude_loss    # scalar
)
```
This breaks `sample_weight` and `reduction=` identically to the premature scalar family. **Not tracked in `test_the_premature_scalar_family_is_pinned.py`.**

### 1.6 `OmniPointCombinedLoss` Is Not a `keras.losses.Loss`
Line 494: `class OmniPointCombinedLoss:` — **no inheritance**. It's a plain callable taking tuple inputs. Exported from `__init__.py` but cannot be used with `model.compile(loss=...)` directly. Must be wrapped or called inside a custom training step.

### 1.7 `jacobian_symmetry.py` / `thera_jacobian_tv.py` Export Functions, Not Losses
These modules export **module-level functions** (`jacobian_symmetry_penalty`, `thera_tv_penalty`, `thera_total_loss`) — **NOT** `keras.losses.Loss` subclasses. They are imported from `__init__.py` but break the "all losses are `Loss` subclasses" convention. Users must call them directly with model + input, not via `y_true`/`y_pred`.

### 1.8 `ClusteringLoss` Returns Scalar
`ClusteringLoss.call()` line 197-200 returns a **scalar** (`scaled_distances` + `distribution_penalty`), not `(batch,)`. Breaks `sample_weight`.

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
| `masked_causal_lm_loss.py` | `True` | ✅ |
| `focal_causal_lm_loss.py` | `True` | ✅ |
| `hrm_loss.py` | `True` (internal) | ✅ |

**Recommendation:** Standardize on `from_logits=True` as default for classification losses; document clearly.

### 2.2 Segmentation Loss Redundancy (3+ Overlapping Implementations)
```
segmentation_loss.py       → SegmentationLosses (scalar batch mean, SAM-compat)
segmentation_wrapper_loss.py → SegmentationWrapperLoss (per-sample, serializable) 
any_loss.py                → DiceLoss, IoULoss, TverskyLoss, FocalTverskyLoss (AnyLoss framework)
multi_labels_loss.py       → PerChannelBinaryLoss, WeightedBinaryFocalLoss, DiceLossPerChannel
segmentation_metrics.py    → per-sample *_per_sample + scalar *_scalar backends
```
**Issues:**
- `SegmentationLosses` returns **scalar batch mean** (not per-sample) for SAM compatibility
- `SegmentationWrapperLoss` wraps it but delegates to scalar methods
- `AnyLoss` versions use **differentiable confusion matrix** (completely different math)
- `multi_labels_loss` has its own Dice/Focal implementations
- No clear guidance on which to use when

### 2.3 Multi-Task Loss Redundancy (3+ Patterns)
- `MultiTaskLoss` — simple weighted sum by output name (dict inputs)
- `YOLOv12MultiTaskLoss` — shape-based dispatch + learnable uncertainty weights
- `SegmentationWrapperLoss` — name-dispatched single loss
- `Sam3DetectionLoss` — Hungarian matcher + 6 terms, packed tensor layout
- `OmniPointCombinedLoss` — plain callable with tuple I/O

**Gap:** No unified multi-task loss abstraction; each reinvents dispatch/weighting.

### 2.4 Factory Function Signatures Inconsistent
```python
create_segmentation_wrapper_loss(loss_name, config)
create_multistep_loss(name, **kwargs)           # strict kwarg validation
create_symmetric_infonce_loss(temperature=..., **kwargs)
create_yolov12_multitask_loss(tasks, num_detection_classes=..., **kwargs)
create_siglip_loss(...) / create_adaptive_siglip_loss(...) / create_hybrid_loss(...)
create_hrm_loss(...)
```
Some validate unknown kwargs (multistep), some don't; some have positional args, some keyword-only.

### 2.5 Non-Uniform Return Shapes for Token-Level Losses
| Loss | Return Shape | `sample_weight` Shape Expected |
|------|-------------|-------------------------------|
| `MaskedCausalLMLoss` | `(batch,)` | `(batch,)` ✅ |
| `FocalCausalLMLoss` | `(batch,)` | `(batch,)` ✅ |
| `HRMLoss` | `(batch,)` | `(batch,)` ✅ |
| `StableMaxCrossEntropy` | `(batch, seq_len)` | `(batch, seq_len)` or `(batch, 1)` |
| `NanoVLMLoss` | scalar or `(batch, seq_len-1)` | varies by `reduction` |

**Problem:** `StableMaxCrossEntropy` returns token-level `(batch, seq_len)`. A `(batch,)` `sample_weight` **raises** `Incompatible shapes`. This is **deliberate** (matches stock `SparseCategoricalCrossentropy`) but inconsistent with other LM losses that return per-sample `(batch,)`.

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
- `jacobian_symmetry.py` — exports `jacobian_symmetry_penalty` (function, not Loss)
- `thera_jacobian_tv.py` — exports `thera_tv_penalty`, `thera_total_loss` (functions)

**Gap:** These are not importable from `dl_techniques.losses` as `Loss` subclasses — must import from module directly. Not documented in `__init__.py` or README.

### 3.3 Config Serialization of State-Holding Losses
`DINOLoss`, `iBOTPatchLoss`, `TopographicVAELoss` serialize internal state (`center` variable) in `get_config()`. This works but:
- Config size scales with `out_dim` (~1.3 MiB JSON at 65536)
- No versioning — if state structure changes, old configs break
- `WassersteinGradientPenaltyLoss` wraps `WassersteinLoss` but doesn't serialize it properly

### 3.4 `OmniPointCombinedLoss` Has No `get_config`/`from_config` for Keras Serialization
It has `get_config`/`from_config` methods but they're for the class's own config dict, not Keras serialization. It's not a `Loss` subclass so `model.save()`/`load_model()` won't work.

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
| Sphinx (`:param:`) | 14 | `infonce_loss.py`, `colbert_loss.py`, `multistep_loss.py`, `ddpm_hybrid_loss.py`, `doc_scanner_flow_sequence_loss.py`, `sam3_detection_loss.py` |
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
3. `segmentation_metrics.py` per-sample functions as shared backend
4. Remove duplicate `DiceLoss`/`IoULoss` from `AnyLoss` (they're not differentiable approximations)

### 5.4 Create Multi-Task Loss Base Class
Extract common pattern from `YOLOv12MultiTaskLoss`, `MultiTaskLoss`, `Sam3DetectionLoss`:
```python
class MultiTaskLossBase(keras.losses.Loss):
    def _route_by_shape(self, y_true, y_pred): ...
    def _apply_weighting(self, task_name, raw_loss): ...
    def _shared_matcher(self, y_true, y_pred): ...  # for Hungarian-based
```

### 5.5 Add `sample_weight` Contract Tests Per Loss
Each loss should have a test asserting:
```python
def test_sample_weight_decomposes(loss_fn, y_true, y_pred):
    weighted = loss_fn(y_true, y_pred, sample_weight=[1,1,1,0])
    dropped = loss_fn(y_true[:3], y_pred[:3])
    assert weighted * 4/3 == dropped  # or document why not
```

### 5.6 Convert Non-Loss Callables to Proper Losses or Move Them
- `OmniPointCombinedLoss` → make it a `Loss` subclass or move to `models/`
- `jacobian_symmetry_penalty` / `thera_tv_penalty` → either make `Loss` subclasses taking `(model, y)` or document clearly they're not `Loss` subclasses
- `magnet_loss_fn` (in `reid_cosine_loss.py`) — deliberately not a `Loss` (batch-coupled), documented as such

---

## 6. Missing Losses / Feature Gaps

### 6.1 No CRPS / Energy Score for Probabilistic Regression
`QuantileLoss`, `MASELoss`, `SMAPELoss` exist but no **CRPS** (Continuous Ranked Probability Score) or **Energy Score** for proper scoring rules on predictive distributions.

### 6.2 No PolyLoss / GHM-C / GHM-R
Modern classification losses missing from the collection.

### 6.3 No Distillation Losses (except `FeatureAlignmentLoss` and `ColBERTDistillationLoss`)
Knowledge distillation losses (KD, PKD, CRD, RKD, etc.) not centralized.

### 6.4 No Contrastive Loss for Audio/Tabular
`CLIPContrastiveLoss`, `SigLIPContrastiveLoss` are vision-language; no generic contrastive for other modalities.

### 6.5 No Focal Loss for Multi-Label (in main exports)
`multi_labels_loss.py` has `WeightedBinaryFocalLoss` but it's not in `__init__.py` exports.

### 6.6 No Regression Calibration Losses
Only `brier_spiegelhalters_ztest_loss` for binary classification calibration.

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

### 7.5 Non-Loss Callables Not Documented as Special Cases
`OmniPointCombinedLoss`, `jacobian_symmetry_penalty`, `thera_tv_penalty`, `magnet_loss_fn` are exported but not flagged as "not a `keras.losses.Loss`".

---

## 8. Test Coverage Gaps

| Area | Coverage | Gap |
|------|----------|-----|
| `sample_weight` decomposition | Only `goodhart_loss`, `premature_scalar` test | Not per-loss |
| Serialization round-trip | Only `goodhart_loss`, `dino_loss`, `segmentation_wrapper` | Most losses untested |
| `dtype` policies | Only `goodhart_loss`, `wasserstein_loss` | Most losses untested |
| Gradient correctness | Only `goodhart_loss` (entropy grad) | Most losses untested |
| Batch-coupling behavior | Only `goodhart_loss`, `dino_loss`, `multistep_loss` | Others assumed |
| Non-`Loss` callables | None | `OmniPointCombinedLoss`, `jacobian_symmetry_penalty`, `thera_tv_penalty`, `magnet_loss_fn` |

---

## 9. Notable Exemplars (Patterns to Emulate)

### 9.1 `goodhart_loss.py` — Gold Standard
- Explicit `BATCH_COUPLING` documentation
- `keyword-only` parameters after `prior_weight` to catch legacy positional calls
- Removed broken `mi_weight` with loud error explaining *why* (wrong sign)
- Comprehensive test suite with independent float64 oracles
- `analyze_loss_components` with bounded shares summing to 1.0
- Proper `dtype` handling (float16 floor = `finfo.tiny`, not `epsilon`)

### 9.2 `ddpm_hybrid_loss.py` — Packed Tensor Contract
- Documents exactly why `y_true` has `2C+1` channels vs `y_pred`'s `2C`
- `stop_gradient` on epsilon half — frozen-out variational bound
- Per-sample return with exact algebraic identity proof
- Static shape validation at call time

### 9.3 `doc_scanner_flow_sequence_loss.py` — Ambiguities Documented
- Two paper ambiguities explicitly recorded (L1-as-mean vs sum, Eq. 14 row index)
- Tests guard the weight direction (`TestTheLastIterationCarriesWeightOne`)
- Cross-module import from `models/` documented and justified (D-031)

### 9.4 `sam3_detection_loss.py` — Signed Divergences from Reference
- Packed tensor layout as single source of truth
- Every divergence from upstream named: `+ PORT_ONLY(...) - REFERENCE_ONLY(...)`
- Five divisors documented as a table with tests
- `pad_n_queries` flagged as variant-derived, not constant

### 9.5 `brier_spiegelhalters_ztest_loss.py` — Corrected Statistics
- Fixed wrong statistic (was calibration-in-the-large under Spiegelhalter name)
- Per-sample decomposition of batch-global Z statistic (algebraic identity)
- `chance_corrected=True` (Z²-1), `normalize_by_n=True` — removes two scaling defects
- Legacy config migration in `from_config`

### 9.6 `infonce_loss.py` — Clear Relationship Mapping
- Documents exactly how it differs from InfoNCE, CLIP, SimCSE, NT-Xent
- Batch-size-1 guard (partial but documented)
- No dtype fabrication (explains why Keras already casts to float32)

---

## 10. Priority Action Items

### P0 (Critical — Silent Wrong Results)
1. **Fix remaining premature scalar losses** — move all 14 from `STILL_BROKEN` to `KNOWN_GOOD`
2. **Fix `utilization_loss.py`** — return `(batch,)` not scalar
3. **Fix `ClusteringLoss`** — return `(batch,)` not scalar  
4. **Document batch-coupling** in every affected loss's docstring + README
5. **Add per-loss `sample_weight` contract test** for all public losses
6. **Fix `OmniPointCombinedLoss`** — make it a `Loss` subclass or document as non-Loss
7. **Fix `jacobian_symmetry.py` / `thera_jacobian_tv.py`** — either make `Loss` subclasses or remove from `__init__.py` exports

### P1 (High — API Consistency)
8. **Standardize `from_logits` default** to `True` for classification losses
9. **Consolidate segmentation losses** — pick one canonical implementation
10. **Unify factory function signatures** with strict kwarg validation
11. **Make `StableMaxCrossEntropy` consistent** — either document token-level convention clearly or provide per-sample variant
12. **Backport logit-space focal fix** from `sam2_video_loss.py` to `sam_mask_loss.py` (or document divergence)

### P2 (Medium — Maintainability)
13. **Create `PerSampleLoss` base class** for shape enforcement
14. **Add batch-coupling metadata** to loss classes
15. **Document `legacy_packages` requirement** for YOLOv12 losses
16. **Add serialization tests** for state-holding losses (`DINOLoss`, `TopographicVAELoss`, etc.)

### P3 (Low — Polish)
17. **Pick one docstring style** for new files (Sphinx preferred per `models/language/bert/`)
18. **Add missing losses** (CRPS, PolyLoss, generic contrastive)
19. **Create multi-task loss abstraction**
20. **Add migration guide** for deprecated `create_loss_function`

---

## 11. Verdict

The `losses` package is **production-functional for its intended use cases** (the trainer scripts in `src/train/` use these losses successfully), but it has **accumulated debt that will bite users who:**
- Use `sample_weight` / `class_weight` with batch-coupled losses
- Expect `reduction=` to work on scalar-returning losses
- Try to serialize/deserialize models with state-holding losses
- Mix losses from different sub-frameworks (AnyLoss vs Segmentation vs native Keras)
- Use the non-`Loss` callables (`OmniPointCombinedLoss`, `jacobian_symmetry_penalty`, etc.) with `model.compile()`

**Recommendation:** Treat this as a **mature but debt-laden package** — fix the P0 items before encouraging broad external adoption, then incrementally address P1-P3. The newer losses (post-2026-08) show the project *can* produce rigorous, well-tested, well-documented losses; the debt is concentrated in the older strata and the cross-cutting conventions.