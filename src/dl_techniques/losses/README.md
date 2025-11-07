# Losses Module

The `dl_techniques.losses` module provides a comprehensive collection of advanced and specialized loss functions for Keras 3, designed for a wide range of deep learning tasks. Each loss is implemented as a self-contained, serializable Keras object for seamless integration into any deep learning workflow.

## Overview

This module includes a diverse set of loss functions covering tasks such as robust regression, imbalanced classification, self-supervised learning, generative modeling, and multi-task computer vision. Many losses also include helper functions for analysis or factory functions for easy instantiation.

## Available Losses

The following loss functions are available in this module:

| Name/Group | Class(es) / Function(s) | Description | Use Case |
|---|---|---|---|
| **Affine Invariant** | `AffineInvariantLoss` | A loss invariant to scale and shift, normalizing predictions and targets before computing L1 distance. | Self-supervised monocular depth estimation where scale is ambiguous. |
| **AnyLoss Framework** | `AnyLoss`, `F1Loss`, `FBetaLoss`, `AccuracyLoss`, `GeometricMeanLoss`, `BalancedAccuracyLoss` | A framework to convert any confusion-matrix-based metric into a differentiable loss function. | Directly optimizing metrics like F1-score or G-Mean for imbalanced classification. |
| **Calibration** | `BrierScoreLoss`, `SpiegelhalterZLoss`, `CombinedCalibrationLoss` | Losses and metrics that directly optimize for model calibration by penalizing miscalibrated probabilities. | Training well-calibrated models for reliable uncertainty estimates in classification. |
| **Capsule Networks** | `CapsuleMarginLoss` | Margin-based loss that encourages long vectors for correct classes and short vectors for incorrect ones. | Training Capsule Networks for object presence detection. |
| **Clustering** | `ClusteringLoss` | A dual-objective loss combining intra-cluster distance with a cluster distribution penalty. | Deep unsupervised clustering to learn compact and balanced clusters. |
| **Contrastive (Vision-Language)** | `CLIPContrastiveLoss`, `SigLIPContrastiveLoss` | Symmetric contrastive losses for learning joint image-text embedding spaces. SigLIP is a more scalable sigmoid-based alternative. | Training vision-language models like CLIP or SigLIP from image-text pairs. |
| **DINO Framework** | `DINOLoss`, `iBOTPatchLoss`, `KoLeoLoss` | Losses for self-supervised learning via knowledge distillation (DINO), masked patch prediction (iBOT), and entropic regularization (KoLeo). | Self-supervised pre-training of Vision Transformers without labels. |
| **Feature Alignment** | `FeatureAlignmentLoss` | A margin-based cosine similarity loss to align student and teacher feature representations. | Knowledge distillation and semantic feature transfer between models. |
| **Information-Theoretic** | `DecoupledInformationLoss`, `FocalUncertaintyLoss`, `GoodhartAwareLoss` | Advanced losses that combine a task objective (e.g., cross-entropy) with regularizers for uncertainty, diversity, and information compression. | Improving model robustness, calibration, and generalization by preventing overconfidence and reliance on spurious correlations. |
| **Language Modeling** | `NanoVLMLoss`, `HRMLoss` | Autoregressive cross-entropy losses for next-token prediction, with support for masking and multi-task Q-learning (HRM). | Training generative language models or the language component of VLMs. |
| **Robust Regression** | `HuberLoss` | A hybrid loss that behaves like MSE for small errors and MAE for large errors, making it robust to outliers. | Regression tasks with noisy data or significant outliers. |
| **Segmentation** | `SegmentationLosses`, `create_segmentation_loss_function` | A factory for various segmentation losses, including Dice, Focal, Tversky, Lovász, and combined losses. | Semantic segmentation tasks, especially with class imbalance. |
| **Time Series Forecasting** | `MASELoss`, `SMAPELoss`, `MQLoss` | Scale-free error metrics (MASE, SMAPE) and quantile loss (MQL) for probabilistic forecasting. | Evaluating and training forecasting models across series with different scales and for generating prediction intervals. |
| **Wasserstein GANs** | `WassersteinLoss`, `WassersteinGradientPenaltyLoss`, `create_wgan_gp_losses` | Losses based on the Wasserstein distance for stable training of Generative Adversarial Networks. | Training WGANs and WGAN-GP for high-quality generative modeling. |
| **YOLOv12 Multi-Task** | `YOLOv12MultiTaskLoss`, `create_yolov12_multitask_loss` | An advanced multi-task loss orchestrator for object detection, segmentation, and classification, with optional uncertainty weighting. | Training complex, multi-headed computer vision models like YOLOv12. |

## Basic Usage

Most loss functions can be directly imported and used in `model.compile()`.

```python
import keras
from dl_techniques.losses import F1Loss, HuberLoss, CLIPContrastiveLoss

# Example for imbalanced classification
model.compile(
    optimizer='adam',
    loss=F1Loss(amplifying_scale=73.0),
    metrics=['accuracy']
)

# Example for robust regression
model.compile(
    optimizer='adam',
    loss=HuberLoss(delta=1.5),
    metrics=['mae']
)
```

## Loss-Specific Parameters & Usage

### AnyLoss Framework (`F1Loss`, `FBetaLoss`, etc.)
**Key Params:** `amplifying_scale` (float, default: 73.0), `from_logits` (bool, default: False). `FBetaLoss` also takes `beta`.

```python
from dl_techniques.losses import F1Loss, FBetaLoss

# Directly optimize F1-score
f1_loss = F1Loss()

# Optimize F2-score (weights recall higher than precision)
f2_loss = FBetaLoss(beta=2.0)

model.compile(optimizer='adam', loss=f2_loss)
```

### Contrastive Losses (`CLIPContrastiveLoss`, `SigLIPContrastiveLoss`)
These losses are self-supervised and expect predictions as a dictionary or tuple of logits.

**Key Params:** `temperature` (float), `label_smoothing` (float).

```python
from dl_techniques.losses import CLIPContrastiveLoss

# In model definition...
image_logits = ... # (batch, batch) similarities
text_logits = ... # (batch, batch) similarities
# The model must output a dictionary or a tuple
outputs = {'logits_per_image': image_logits, 'logits_per_text': text_logits}
model = keras.Model(inputs=[img_input, txt_input], outputs=outputs)

# In compilation...
# y_true is ignored, so we can pass dummy data or None
model.compile(optimizer='adam', loss=CLIPContrastiveLoss(temperature=0.07))
```

### DINO Loss
`DINOLoss` requires a manual call to `update_center()` in the training loop.

```python
from dl_techniques.losses import DINOLoss

dino_loss = DINOLoss(out_dim=65536)

class DINOModel(keras.Model):
    def train_step(self, data):
        # ... forward passes for student and teacher ...
        with tf.GradientTape() as tape:
            teacher_output = self.teacher(global_crops, training=False)
            student_output = self.student(all_crops, training=True)
            loss = dino_loss(teacher_output, student_output)
        
        # ... backward pass ...
        
        # CRITICAL: Update the loss's internal center
        dino_loss.update_center(teacher_output)
        
        return {"loss": loss}
```

### Segmentation Losses
Use the factory `create_segmentation_loss_function` for convenience.

**Key Params:** `loss_name` (str), `config` (`LossConfig` object).

```python
from dl_techniques.losses import create_segmentation_loss_function, SegmentationLossConfig

# Configure parameters for segmentation
seg_config = SegmentationLossConfig(num_classes=19, focal_gamma=2.5)

# Create a combined Focal + Tversky loss
focal_tversky_loss = create_segmentation_loss_function(
    'focal_tversky',
    config=seg_config
)
model.compile(optimizer='adam', loss=focal_tversky_loss)
```

### Wasserstein GAN Losses
WGAN-GP requires a custom training loop to compute the gradient penalty.

```python
from dl_techniques.losses import create_wgan_gp_losses, compute_gradient_penalty

critic_loss_fn, generator_loss_fn = create_wgan_gp_losses(lambda_gp=10.0)

# Inside your custom train_step for the critic...
with tf.GradientTape() as tape:
    # ... get real_pred and fake_pred from critic ...
    
    # 1. Compute Wasserstein loss component
    w_loss = critic_loss_fn(y_true, y_pred) # y_true indicates real/fake
    
    # 2. Compute gradient penalty
    gp = compute_gradient_penalty(critic, real_images, fake_images)
    
    # 3. Combine losses
    total_critic_loss = w_loss + gp

# ... apply gradients ...
```

### YOLOv12 Multi-Task Loss
This is a single "orchestrator" loss for models with multiple named outputs.

```python
from dl_techniques.losses import create_yolov12_multitask_loss

# Configure for detection and segmentation on COCO
yolo_loss = create_yolov12_multitask_loss(
    tasks=['detection', 'segmentation'],
    num_detection_classes=80,
    num_segmentation_classes=80,
    input_shape=(640, 640),
    use_uncertainty_weighting=True # Automatically balance task losses
)

# Keras handles routing the correct y_true/y_pred to the loss
# based on the model's output names.
model.compile(optimizer='adam', loss=yolo_loss)
```