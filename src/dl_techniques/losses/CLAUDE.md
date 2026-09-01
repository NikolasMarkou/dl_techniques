# Losses Package

Specialized loss functions for diverse deep learning tasks, all implemented as serializable Keras 3 loss objects.

## Public API

All losses are exported from `__init__.py`. Key imports:

```python
from dl_techniques.losses import (
    # Robust regression
    HuberLoss, MASELoss, SMAPELoss, QuantileLoss,
    # Imbalanced classification
    AnyLoss, F1Loss, FBetaLoss, FocalUncertaintyLoss,
    # Self-supervised learning
    CLIPContrastiveLoss, DINOLoss, SigLIPContrastiveLoss,
    SymmetricInfoNCELoss, create_symmetric_infonce_loss,
    # Calibration (binary only)
    BrierScoreLoss, SpiegelhalterZLoss, CombinedCalibrationLoss,
    BrierScoreMetric, SpiegelhalterZMetric,
    # Generative
    WassersteinLoss,
    # Specialized
    CapsuleMarginLoss, SegmentationLosses, SegmentationWrapperLoss,
    # Information-theoretic
    GoodhartAwareLoss, DecoupledInformationLoss,
)
```

## Modules

- `any_loss.py` — AnyLoss framework: differentiable approximations of non-differentiable metrics (F1, accuracy, balanced accuracy, G-mean)
- `clifford_detection_loss.py` — CliffordNet detection loss
- `affine_invariant_loss.py` — Affine-invariant distance loss
- `brier_spiegelhalters_ztest_loss.py` — Binary calibration losses + metrics. Ships BOTH
  calibration z-statistics under an explicit `statistic=` knob: Spiegelhalter's (1986)
  `(1-2p)`-weighted Z (the default) and the strictly weaker calibration-in-the-large test,
  which is what this module computed under the Spiegelhalter name before 2026-09-02. The
  penalty is chance-corrected (`relu(Z²-1)`, since `E[Z²]=1` not 0) and divided by `N`, and
  it decomposes per row so `sample_weight` still selects rows
- `capsule_margin_loss.py` — Capsule network margin loss with analysis utilities
- `chamfer_loss.py` — Chamfer distance for point clouds
- `clip_contrastive_loss.py` / `siglip_contrastive_loss.py` — Contrastive losses for CLIP/SigLIP
- `clustering_loss.py` — Clustering loss + metrics
- `colbert_loss.py` — ColBERT retrieval losses: v1 pairwise/listwise softmax cross-entropy and v2 cross-encoder KL distillation
- `decoupled_information_loss.py` — Information-theoretic regularization
- `dino_loss.py` — DINO/iBOT self-supervised loss
- `feature_alignment_loss.py` — Feature alignment for knowledge distillation
- `flow_matching_velocity_loss.py` — Rectified-flow / flow-matching velocity-regression loss
- `focal_causal_lm_loss.py` — Focal loss variant for causal LM training
- `focal_uncertainty_loss.py` — Focal loss with uncertainty estimation
- `goodhart_loss.py` — Goodhart's law-aware loss
- `hrm_loss.py` — Hierarchical reasoning model loss
- `huber_loss.py` — Robust Huber loss
- `image_restoration_loss.py` — Multi-component image restoration loss
- `infonce_loss.py` — Single-tower symmetric InfoNCE over two views of one batch (SimCSE-style dropout pairs); positives are positional, the batch size is the negative count
- `jacobian_symmetry.py` — Stochastic Jacobian-symmetry penalty (double-VJP) pushing a denoiser toward a conservative field
- `lpips_loss.py` — LPIPS-flavored perceptual loss over a frozen ImageNet VGG16 backbone
- `mase_loss.py` — Mean Absolute Scaled Error
- `masked_causal_lm_loss.py` — Masked causal LM loss (skip ignore-tokens during NTP)
- `multistep_loss.py` — ADAM h-steps-ahead losses (`mseh`/`tmse`/`gtmse`/`msce`) over the
  horizon axis. `gtmse` is the one non-decomposable member: its per-sample form is a
  first-order expansion about the DETACHED batch mean, exact in both value and gradient
- `multi_labels_loss.py` — Multi-label classification loss
- `multi_task_loss.py` — Multi-task loss aggregator with per-task weighting
- `nano_vlm_loss.py` — NanoVLM vision-language loss
- `quantile_loss.py` — Quantile regression loss
- `sam2_video_loss.py` — Ground-truth-gated mask supervision for the SAM 2 video trainer
- `sam3_detection_loss.py` — SAM 3 Hungarian matcher + six-term detection loss, and the packed-tensor layout both speak
- `sam_mask_loss.py` — SAM mask losses: `SAMMaskLoss` (focal + dice) and `SAMIoULoss` (MSE); a deliberate containment for two measured `SegmentationLosses` reuse defects
- `scaled_mse_loss.py` — Scaled mean-squared-error loss
- `segmentation_loss.py` — Segmentation loss (Dice, Tversky, focal)
- `segmentation_wrapper_loss.py` — Serializable name-dispatched wrapper around `SegmentationLosses` for use as a compile/save/load-friendly Keras `Loss`
- `smape_loss.py` — Symmetric MAPE
- `sparsemax_loss.py` — Sparsemax loss
- `superpoint_loss.py` — SuperPoint detector (65-class grid softmax CE) and descriptor losses
- `tabm_loss.py` — TabM model loss
- `thera_jacobian_tv.py` — THERA aliasing TV penalty over the exact analytic spatial Jacobian
- `utilization_loss.py` — Utilization / load-balancing loss (e.g. MoE routing)
- `wasserstein_loss.py` — Wasserstein/WGAN-GP loss
- `yolo12_multitask_loss.py` — YOLOv12 multi-task detection loss

## Conventions

Each number below carries the command that re-derives it. Re-run rather than trust — this section
has gone stale before.

- **Base class.** All losses inherit from `keras.losses.Loss` and implement
  `call(self, y_true, y_pred)` and `get_config()`. Some modules also export companion metric
  classes or analysis functions.

- **`call()` returns ONE VALUE PER SAMPLE, shape `(batch,)` — never a scalar.**
  This is the rule most often broken here, and the failure is silent. Keras'
  `reduce_weighted_values` computes `values * sample_weight` **before** reducing, so a scalar
  return does not *ignore* `sample_weight` — it BROADCASTS against it and yields
  `whole_batch_loss * mean(sample_weight)`, charging every row the batch aggregate and discarding
  which rows were weighted. It also makes `reduction=` a dead knob. A plausible wrong number, not
  an error. `goodhart_loss.py` and `focal_uncertainty_loss.py` are correct exemplars.
  **Which modules still violate this is recorded executably, not in prose:**
  `tests/test_losses/test_the_premature_scalar_family_is_pinned.py` MEASURES the defect per class
  and goes red the day one is fixed. Do not duplicate that list here — a prose list rots; the test
  cannot.

- **Registration.** Every `Loss` subclass carries
  `@register_dl_technique("dl_techniques.losses.<module>")` from
  `dl_techniques.utils.keras_registration` — never a bare
  `@keras.saving.register_keras_serializable()`. 41 of 44 modules register something; the
  non-registering ones export plain functions only.
  `grep -rl register_dl_technique src/dl_techniques/losses --include=*.py | wc -l`

- **Import style.** NEW files use `import keras` and qualify at the call site (`keras.ops.matmul`).
  33 of 44 existing modules use the superseded `from keras import ops`; that majority is **neither a
  pattern to extend nor a migration target** — leave them alone.
  `grep -rl "^from keras import ops" src/dl_techniques/losses --include=*.py | wc -l`

- **Docstring style is Google-majority, and that is a fact, not a mandate.**
  32 of 44 carry a Google `Args:` block, 10 carry Sphinx `:param `, 1 carries both.
  **Match the file you are editing; never convert a file wholesale.** The newest modules
  (`colbert_loss.py`, `infonce_loss.py`) are Sphinx, so both styles are live and growing —
  an earlier claim that "Sphinx is the losses/ convention" was an overstatement the tree does not
  support.
  `grep -rlE "^ +Args:$" src/dl_techniques/losses --include=*.py | wc -l`
  `grep -rl ":param " src/dl_techniques/losses --include=*.py | wc -l`

- **Documenting a loss.** A row in the tables above asserts the symbol is IMPORTABLE from
  `dl_techniques.losses`. Verify with `hasattr`, not by reading the module — several modules ship
  classes that `__init__.py` does not export, and a row promising an import that raises is worse
  than no row at all.

## Testing

Tests in `tests/test_losses/`.
