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
    # Calibration
    BrierScoreLoss, SpiegelhalterZLoss, CombinedCalibrationLoss,
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
- `brier_spiegelhalters_ztest_loss.py` — Calibration losses + metrics (Brier, Spiegelhalter's Z)
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

- All losses inherit from `keras.losses.Loss`
- Must implement `call(self, y_true, y_pred)` and `get_config()`
- Some modules also export companion metric classes or analysis functions

## Testing

Tests in `tests/test_losses/`.
