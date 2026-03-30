# Losses

Loss functions for different tasks

**27 modules in this category**

## Affine_Invariant_Loss

### losses.affine_invariant_loss
A loss function invariant to scale and shift transformations.

**Classes:**
- `AffineInvariantLoss`

**Functions:** `call`, `get_config`

*📁 File: `src/dl_techniques/losses/affine_invariant_loss.py`*

## Any_Loss

### losses.any_loss
AnyLoss: Differentiable Confusion Matrix Metrics as Loss Functions

**Classes:**

- `ApproximationFunction` - Keras Layer
  Approximation function for transforming sigmoid outputs to near-binary values.
  ```python
  ApproximationFunction(amplifying_scale: float = 73.0, **kwargs)
  ```
- `AnyLoss`
- `AccuracyLoss`
- `PrecisionLoss`
- `RecallLoss`
- `SpecificityLoss`
- `F1Loss`
- `FBetaLoss`
- `BalancedAccuracyLoss`
- `GeometricMeanLoss`
- `YoudenJLoss`
- `MCCLoss`
- `CohenKappaLoss`
- `IoULoss`
- `DiceLoss`
- `TverskyLoss`
- `FocalTverskyLoss`
- `WeightedCrossEntropyWithAnyLoss`

**Functions:** `get_loss`, `call`, `compute_output_shape`, `get_config`, `compute_confusion_matrix` (and 23 more)

*📁 File: `src/dl_techniques/losses/any_loss.py`*

## Brier_Spiegelhalters_Ztest_Loss

### losses.brier_spiegelhalters_ztest_loss
A loss function based on Spiegelhalter's Z-test.

**Classes:**
- `BrierScoreLoss`
- `SpiegelhalterZLoss`
- `CombinedCalibrationLoss`
- `BrierScoreMetric`
- `SpiegelhalterZMetric`

**Functions:** `call`, `get_config`, `call`, `get_config`, `call` (and 9 more)

*📁 File: `src/dl_techniques/losses/brier_spiegelhalters_ztest_loss.py`*

## Capsule_Margin_Loss

### losses.capsule_margin_loss
Margin loss for training Capsule Networks.

**Classes:**
- `CapsuleMarginLoss`

**Functions:** `capsule_margin_loss`, `analyze_margin_loss_components`, `call`, `get_config`

*📁 File: `src/dl_techniques/losses/capsule_margin_loss.py`*

## Chamfer_Loss

### losses.chamfer_loss

**Classes:**
- `ChamferLoss`

**Functions:** `call`

*📁 File: `src/dl_techniques/losses/chamfer_loss.py`*

## Clip_Contrastive_Loss

### losses.clip_contrastive_loss
Symmetric Contrastive Loss for Multimodal Training (CLIP).

**Classes:**
- `CLIPContrastiveLoss`

**Functions:** `call`, `get_config`, `temperature_value`, `update_temperature`

*📁 File: `src/dl_techniques/losses/clip_contrastive_loss.py`*

## Clustering_Loss

### losses.clustering_loss
A dual-objective loss function for deep clustering.

**Classes:**
- `ClusteringMetrics`
- `ClusteringLoss`
- `ClusteringMetricsCallback`

**Functions:** `compute_clustering_metrics`, `call`, `get_config`, `on_epoch_end`

*📁 File: `src/dl_techniques/losses/clustering_loss.py`*

## Core

### losses
Losses Module

*📁 File: `src/dl_techniques/losses/__init__.py`*

## Decoupled_Information_Loss

### losses.decoupled_information_loss
DecoupledInformationLoss: An Information-Theoretic Loss for Robust Classification

**Classes:**
- `DecoupledInformationLoss`

**Functions:** `analyze_decoupled_information_loss`, `call`, `get_config`

*📁 File: `src/dl_techniques/losses/decoupled_information_loss.py`*

## Dino_Loss

### losses.dino_loss
A self-supervised loss from the DINO framework.

**Classes:**
- `DINOLoss`
- `iBOTPatchLoss`
- `KoLeoLoss`

**Functions:** `call`, `update_center`, `get_config`, `call`, `update_center` (and 3 more)

*📁 File: `src/dl_techniques/losses/dino_loss.py`*

## Feature_Alignment_Loss

### losses.feature_alignment_loss
A margin-based cosine similarity loss for feature alignment.

**Classes:**
- `FeatureAlignmentLoss`

**Functions:** `call`, `get_config`

*📁 File: `src/dl_techniques/losses/feature_alignment_loss.py`*

## Focal_Uncertainty_Loss

### losses.focal_uncertainty_loss
FocalUncertaintyLoss: Focal Loss with Uncertainty Regularization

**Classes:**
- `FocalUncertaintyLoss`

**Functions:** `analyze_focal_uncertainty_loss`, `call`, `compute_output_shape`, `get_config`, `from_config`

*📁 File: `src/dl_techniques/losses/focal_uncertainty_loss.py`*

## Goodhart_Loss

### losses.goodhart_loss
An information-theoretic loss to promote robust generalization.

**Classes:**
- `GoodhartAwareLoss`

**Functions:** `analyze_loss_components`, `call`, `get_config`

*📁 File: `src/dl_techniques/losses/goodhart_loss.py`*

## Hrm_Loss

### losses.hrm_loss
A composite, multi-task loss function for training the Hierarchical Reasoning Model.

**Classes:**
- `StableMaxCrossEntropy`
- `HRMLoss`

**Functions:** `create_hrm_loss`, `call`, `get_config`, `call`, `get_config`

*📁 File: `src/dl_techniques/losses/hrm_loss.py`*

## Huber_Loss

### losses.huber_loss
Huber loss, a robust loss function for regression tasks.

**Classes:**
- `HuberLoss`

**Functions:** `call`, `get_config`

*📁 File: `src/dl_techniques/losses/huber_loss.py`*

## Image_Restoration_Loss

### losses.image_restoration_loss
DarkIR Loss Functions for Low-Light Image Enhancement and Restoration.

**Classes:**
- `CharbonnierLoss`
- `FrequencyLoss`
- `EdgeLoss`
- `VGGLoss`
- `EnhanceLoss`
- `DarkIRCompositeLoss`

**Functions:** `call`, `get_config`, `call`, `get_config`, `call` (and 7 more)

*📁 File: `src/dl_techniques/losses/image_restoration_loss.py`*

## Mase_Loss

### losses.mase_loss
Mean Absolute Scaled Error (MASE), a scale-free loss metric.

**Classes:**
- `MASELoss`

**Functions:** `mase_metric`, `call`, `get_config`, `metric`

*📁 File: `src/dl_techniques/losses/mase_loss.py`*

## Multi_Labels_Loss

### losses.multi_labels_loss
Per-Channel Loss Wrapper for Multi-Label Segmentation

**Classes:**
- `PerChannelBinaryLoss`
- `WeightedBinaryFocalLoss`
- `DiceLossPerChannel`

**Functions:** `create_multilabel_segmentation_loss`, `call`, `get_config`, `from_config`, `call` (and 4 more)

*📁 File: `src/dl_techniques/losses/multi_labels_loss.py`*

## Nano_Vlm_Loss

### losses.nano_vlm_loss
Autoregressive cross-entropy loss for language modeling.

**Classes:**
- `NanoVLMLoss`

**Functions:** `call`, `get_config`, `from_config`

*📁 File: `src/dl_techniques/losses/nano_vlm_loss.py`*

## Quantile_Loss

### losses.quantile_loss
Quantile loss, also known as the pinball loss.

**Classes:**
- `MQLoss`
- `QuantileLoss`

**Functions:** `call`, `get_config`, `call`, `get_config`

*📁 File: `src/dl_techniques/losses/quantile_loss.py`*

## Segmentation_Loss

### losses.segmentation_loss
Segmentation Loss Functions Module.

**Classes:**
- `LossConfig`
- `SegmentationLosses`
- `WrappedLoss`

**Functions:** `create_loss_function`, `cross_entropy_loss`, `dice_loss`, `focal_loss`, `tversky_loss` (and 10 more)

*📁 File: `src/dl_techniques/losses/segmentation_loss.py`*

## Siglip_Contrastive_Loss

### losses.siglip_contrastive_loss
Sigmoid Loss for Language Image Pre-training (SigLIP).

**Classes:**
- `SigLIPContrastiveLoss`
- `AdaptiveSigLIPLoss`
- `HybridContrastiveLoss`

**Functions:** `create_siglip_loss`, `create_adaptive_siglip_loss`, `create_hybrid_loss`, `call`, `get_config` (and 4 more)

*📁 File: `src/dl_techniques/losses/siglip_contrastive_loss.py`*

## Smape_Loss

### losses.smape_loss
Symmetric Mean Absolute Percentage Error (SMAPE).

**Classes:**
- `SMAPELoss`

**Functions:** `smape_metric`, `call`, `get_config`

*📁 File: `src/dl_techniques/losses/smape_loss.py`*

## Sparsemax_Loss

### losses.sparsemax_loss
Fenchel-Young loss tailored for Sparsemax activation.

**Classes:**
- `SparsemaxLoss`

**Functions:** `call`, `get_config`

*📁 File: `src/dl_techniques/losses/sparsemax_loss.py`*

## Tabm_Loss

### losses.tabm_loss

**Classes:**
- `TabMLoss`

**Functions:** `call`, `get_config`

*📁 File: `src/dl_techniques/losses/tabm_loss.py`*

## Wasserstein_Loss

### losses.wasserstein_loss
Wasserstein loss functions for GANs.

**Classes:**
- `WassersteinLoss`
- `WassersteinGradientPenaltyLoss`
- `WassersteinDivergence`

**Functions:** `compute_gradient_penalty`, `create_wgan_losses`, `create_wgan_gp_losses`, `call`, `get_config` (and 4 more)

*📁 File: `src/dl_techniques/losses/wasserstein_loss.py`*

## Yolo12_Multitask_Loss

### losses.yolo12_multitask_loss
Defines a custom Keras loss function for the YOLOv12 multi-task model.

**Classes:**
- `YOLOv12ObjectDetectionLoss`
- `DiceFocalSegmentationLoss`
- `ClassificationFocalLoss`
- `YOLOv12MultiTaskLoss`

**Functions:** `create_yolov12_multitask_loss`, `create_yolov12_coco_loss`, `create_yolov12_crack_loss`, `call`, `get_config` (and 11 more)

*📁 File: `src/dl_techniques/losses/yolo12_multitask_loss.py`*