# Losses Package

28+ specialized loss functions for diverse deep learning tasks, all implemented as serializable Keras 3 loss objects.

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
    # Calibration
    BrierScoreLoss, SpiegelhalterZLoss, CombinedCalibrationLoss,
    # Generative
    WassersteinLoss,
    # Specialized
    CapsuleMarginLoss, SegmentationLoss, ImageRestorationLoss,
    # Information-theoretic
    GoodhartLoss, DecoupledInformationLoss,
)
```

## Modules

- `any_loss.py` — AnyLoss framework: differentiable approximations of non-differentiable metrics (F1, accuracy, balanced accuracy, G-mean)
- `affine_invariant_loss.py` — Affine-invariant distance loss
- `brier_spiegelhalters_ztest_loss.py` — Calibration losses + metrics (Brier, Spiegelhalter's Z)
- `capsule_margin_loss.py` — Capsule network margin loss with analysis utilities
- `chamfer_loss.py` — Chamfer distance for point clouds
- `clip_contrastive_loss.py` / `siglip_contrastive_loss.py` — Contrastive losses for CLIP/SigLIP
- `clustering_loss.py` — Clustering loss + metrics
- `decoupled_information_loss.py` — Information-theoretic regularization
- `dino_loss.py` — DINO/iBOT self-supervised loss
- `feature_alignment_loss.py` — Feature alignment for knowledge distillation
- `focal_uncertainty_loss.py` — Focal loss with uncertainty estimation
- `goodhart_loss.py` — Goodhart's law-aware loss
- `hrm_loss.py` — Hierarchical reasoning model loss
- `huber_loss.py` — Robust Huber loss
- `image_restoration_loss.py` — Multi-component image restoration loss
- `mase_loss.py` — Mean Absolute Scaled Error
- `multi_labels_loss.py` — Multi-label classification loss
- `nano_vlm_loss.py` — NanoVLM vision-language loss
- `quantile_loss.py` — Quantile regression loss
- `segmentation_loss.py` — Segmentation loss (Dice, Tversky, focal)
- `smape_loss.py` — Symmetric MAPE
- `sparsemax_loss.py` — Sparsemax loss
- `tabm_loss.py` — TabM model loss
- `wasserstein_loss.py` — Wasserstein/WGAN-GP loss
- `yolo12_multitask_loss.py` — YOLOv12 multi-task detection loss

## Conventions

- All losses inherit from `keras.losses.Loss`
- Must implement `call(self, y_true, y_pred)` and `get_config()`
- Some modules also export companion metric classes or analysis functions

## Testing

Tests in `tests/test_losses/`.
