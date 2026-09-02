# Losses Module

The `dl_techniques.losses` module provides a comprehensive collection of advanced and specialized loss functions for Keras 3, designed for a wide range of deep learning tasks. Each loss is implemented as a self-contained, serializable Keras object for seamless integration into any deep learning workflow.

## Overview

This module includes a diverse set of loss functions covering tasks such as robust regression, imbalanced classification, self-supervised learning, generative modeling, and multi-task computer vision. Many losses also include helper functions for analysis or factory functions for easy instantiation.

## Available Losses

The following loss functions are available in this module:

| Name/Group | Class(es) / Function(s) | Description | Use Case |
|---|---|---|---|
| **Affine Invariant** | `AffineInvariantLoss` | A loss invariant to scale and shift, normalizing predictions and targets before computing L1 distance. | Self-supervised monocular depth estimation where scale is ambiguous. |
| **AnyLoss Framework** | `AnyLoss`, `F1Loss`, `FBetaLoss`, `AccuracyLoss`, `GeometricMeanLoss`, `BalancedAccuracyLoss`, `PrecisionLoss`, `RecallLoss`, `SpecificityLoss`, `YoudenJLoss`, `MCCLoss`, `CohenKappaLoss`, `WeightedCrossEntropyWithAnyLoss` | A framework to convert any confusion-matrix-based metric into a differentiable loss function. | Directly optimizing metrics like F1-score or G-Mean for imbalanced classification. |
| **Calibration** | `BrierScoreLoss`, `SpiegelhalterZLoss`, `CombinedCalibrationLoss`, `BrierScoreMetric`, `SpiegelhalterZMetric` | BINARY ONLY. The Brier score (a proper scoring rule), plus two batch-level calibration z-statistics: Spiegelhalter's (1986) `(1-2p)`-weighted Z, and the weaker calibration-in-the-large test. A calibration statistic is NOT a proper scoring rule and is only sound as a small-weight regularizer -- `CombinedCalibrationLoss` is that anchored form. | Adding a chance-corrected, batch-size-invariant calibration penalty to a proper base loss; monitoring `\|Z\| > 1.96` during training. |
| **Capsule Networks** | `CapsuleMarginLoss` | Margin-based loss that encourages long vectors for correct classes and short vectors for incorrect ones. | Training Capsule Networks for object presence detection. |
| **Clustering** | `ClusteringLoss` | A dual-objective loss combining intra-cluster distance with a cluster distribution penalty. | Deep unsupervised clustering to learn compact and balanced clusters. |
| **Contrastive (Single-Tower)** | `SymmetricInfoNCELoss`, `create_symmetric_infonce_loss` | Symmetric InfoNCE over two views of ONE encoder (SimCSE-style dropout pairs). Positives are positional; the batch size IS the negative count (batch N gives N-1 negatives), with no memory bank. | Self-supervised sentence/embedding training from two augmented or dropout views of the same batch. |
| **Contrastive (Vision-Language)** | `CLIPContrastiveLoss`, `SigLIPContrastiveLoss` | Symmetric contrastive losses for learning joint image-text embedding spaces. SigLIP is a more scalable sigmoid-based alternative. | Training vision-language models like CLIP or SigLIP from image-text pairs. |
| **Dense Retrieval (Late Interaction)** | `ColBERTPairwiseSoftmaxLoss`, `ColBERTDistillationLoss` | ColBERT v1 pairwise softmax cross-entropy over `nway` MaxSim scores (positive at index 0), and v2 KL distillation against a cross-encoder teacher's distribution. | Training ColBERT v1/v2 late-interaction retrievers. |
| **DINO Framework** | `DINOLoss`, `iBOTPatchLoss`, `KoLeoLoss` | Losses for self-supervised learning via knowledge distillation (DINO), masked patch prediction (iBOT), and entropic regularization (KoLeo). | Self-supervised pre-training of Vision Transformers without labels. |
| **Feature Alignment** | `FeatureAlignmentLoss` | A margin-based cosine similarity loss to align student and teacher feature representations. | Knowledge distillation and semantic feature transfer between models. |
| **Generative (Diffusion, DDPM)** | `DDPMHybridLoss` | The published DDPM/DiT training objective for a `LEARNED_RANGE` model: epsilon-prediction MSE plus a variational bound (in bits) whose mean prediction is `stop_gradient`-frozen, so the bound trains the variance channels only. `y_true` is PACKED -- `[0:C]` noise, `[C:2C]` `x_start`, `[2C:2C+1]` the timestep broadcast over the spatial grid -- which is what lets the full objective run under stock `fit()` with no `train_step` override. | Training class-conditional latent diffusion transformers (DiT) and any DDPM that learns its reverse-process variance. |
| **Generative (Flow Matching)** | `FlowMatchingVelocityLoss` | Rectified-flow / flow-matching velocity regression: the model predicts the velocity field transporting a data sample to noise along a straight path. | Training rectified-flow and flow-matching generative models. |
| **Image Quality / Perceptual** | `LPIPSLoss` | LPIPS-flavored perceptual distance over a frozen ImageNet VGG16 backbone, per-channel-normalized and weighted per layer. Dependency-free. | Perceptual supervision for restoration, super-resolution and generative image models. |
| **Confidence & Diversity Regularized** | `DecoupledInformationLoss`, `FocalUncertaintyLoss`, `GoodhartAwareLoss` | Losses that combine a task objective (e.g., cross-entropy) with entropy-based regularizers acting on the PREDICTIONS: a per-sample confidence penalty, and an optional batch-marginal diversity/anti-collapse term. None of them observes the input `X`, so none computes an input-prediction mutual information. | Improving calibration and preventing overconfidence, and — where the batch-marginal term is enabled — preventing predictive collapse onto a few classes. |
| **Language Modeling** | `NanoVLMLoss`, `HRMLoss`, `StableMaxCrossEntropy`, `MaskedCausalLMLoss`, `PrefixMaskedCausalLMLoss`, `FocalCausalLMLoss` | Autoregressive cross-entropy losses for next-token prediction, with support for masking and multi-task Q-learning (HRM). The masked variants skip an ignore-index (e.g. padding) so only real positions contribute; the focal variant down-weights easy tokens. | Training generative language models or the language component of VLMs. |
| **Keypoints** | `SuperPointDetectorLoss`, `SuperPointDescriptorLoss` | SuperPoint's two objectives: a softmax cross-entropy over the 65-class interest-point grid (8x8 cell + dustbin), and the descriptor loss. | Self-supervised interest-point detection and description. |
| **Promptable Segmentation (SAM)** | `SAMMaskLoss`, `SAMIoULoss` | SAM mask supervision (focal + dice) and IoU regression. SAM 2's ground-truth-gated video variant and SAM 3's Hungarian matcher + six-term detection loss ship in `sam2_video_loss.py` / `sam3_detection_loss.py` but are NOT exported from `dl_techniques.losses`; import them from their modules. | Training SAM-family promptable segmentation and detection heads. |
| **Regularizers (Jacobian)** | `jacobian_symmetry_penalty`, `thera_tv_penalty`, `thera_total_loss` — **module-level functions, NOT exported from `dl_techniques.losses`**; import from `dl_techniques.losses.jacobian_symmetry` / `.thera_jacobian_tv` | A stochastic double-VJP penalty pushing a denoiser's Jacobian toward symmetry (a conservative field), and THERA's aliasing TV penalty over the exact analytic spatial Jacobian. | Enforcing conservativeness in bias-free denoisers; anti-aliasing in THERA super-resolution. |
| **Robust Regression** | `HuberLoss` | A hybrid loss that behaves like MSE for small errors and MAE for large errors, making it robust to outliers. | Regression tasks with noisy data or significant outliers. |
| **Scaled / Multi-Scale Regression** | `ScaledMseLoss` | Multi-scale MSE with automatic target resizing. | Deeply-supervised dense-prediction models whose heads emit several resolutions. |
| **Segmentation** | `SegmentationWrapperLoss`, `SegmentationLosses`, `DiceLoss`, `TverskyLoss`, `FocalTverskyLoss`, `IoULoss`, `create_segmentation_loss_function` | Serializable Keras `Loss` dispatching by name over Dice, Focal, Tversky, Lovász, and combined losses. Save/load round-trips without `custom_objects`. | Semantic segmentation tasks, especially with class imbalance. |
| **Tabular Ensembles** | `TabMLoss` | Per-row loss for TabM ensemble training; the batch axis it returns tracks `share_training_batches`, keeping `sample_weight`/`class_weight` correct in both modes. | Training TabM tabular ensembles. |
| **Time Series Forecasting** | `MASELoss`, `SMAPELoss`, `MQLoss`, `QuantileLoss` | Scale-free error metrics (MASE, SMAPE) and quantile loss (MQL) for probabilistic forecasting. | Evaluating and training forecasting models across series with different scales and for generating prediction intervals. |
| **Time Series (Multistep)** | `MultistepLoss`, `create_multistep_loss`, `MULTISTEP_AGGREGATIONS` | ADAM's h-steps-ahead estimators over the horizon axis: `mseh` (step h only), `tmse` (trace), `gtmse` (geometric trace), `msce` (cumulative error). `error_power=1.0` gives the MAE analogues. | Aligning the objective with the decision horizon -- a lead time, a planning quarter -- instead of the one-step-ahead error nobody acts on. |
| **Vision-Language (Sigmoid / Hybrid)** | `AdaptiveSigLIPLoss`, `HybridContrastiveLoss` | An adaptive SigLIP variant, and a hybrid combining the SigLIP objective with a cross-modal denoising penalty (noise injection + squared error; NOT score matching). | Scaling vision-language contrastive training beyond softmax-normalized batches. |
| **Utilization / Load Balancing** | `MANNUtilizationLoss`, `GNNUtilizationLoss` | Load-balancing penalties that discourage collapse onto a few memory slots or expert/routing branches. | Memory-augmented networks and MoE-style routing where capacity must stay spread. |
| **Wasserstein GANs** | `WassersteinLoss`, `WassersteinGradientPenaltyLoss`, `WassersteinDivergence`, `create_wgan_gp_losses` | Losses based on the Wasserstein distance for stable training of Generative Adversarial Networks. | Training WGANs and WGAN-GP for high-quality generative modeling. |
| **YOLOv12 Multi-Task** | `YOLOv12MultiTaskLoss`, `create_yolov12_multitask_loss` | An advanced multi-task loss orchestrator for object detection, segmentation, and classification, with optional uncertainty weighting. | Training complex, multi-headed computer vision models like YOLOv12. |

### Multistep (h-steps-ahead) losses

```python
from dl_techniques.losses import MultistepLoss, create_multistep_loss

# Optimise the whole 12-step horizon, short and long steps weighted comparably.
model.compile(optimizer="adam", loss=MultistepLoss("gtmse", h=12))

# Optimise the CUMULATIVE error over a 4-period lead time (inventory decisions).
model.compile(optimizer="adam", loss=create_multistep_loss("msce", h=4))
```

Inputs are `(batch, horizon)` or `(batch, horizon, features)`; axis 1 is the horizon.
Quantile outputs (`[B, H, F, Q]`) are not supported -- a pinball objective is a
different loss, not a reduction of this one.

**`mseh` starves a direct model.** In ADAM these losses constrain a *recursive*
model, so step `h`'s error still moves every shared parameter. Most forecasters in
`models/time_series/` are *direct*: they emit the whole `[B, H, F]` block from
per-step heads in one pass. Under `mseh` every column other than `h` receives
**exactly zero gradient** and those heads ship untrained, with no shape symptom and
no warning. Pinned by `tests/test_losses/test_the_mseh_starves_other_horizons.py`.

**No shrinkage is claimed here.** The result these losses are famous for -- that
minimising h-step errors shrinks a model's smoothing parameters toward zero -- is
proven for pure additive ETS/ARIMA and arises through *recursive* error
accumulation (Svetunkov, Kourentzes & Killick 2023). On a direct model these are a
principled horizon reweighting; there is no smoothing parameter to shrink.

**`gtmse` is batch-size sensitive** by construction: it takes the log of a
*batch*-level mean error, the batch standing in for the sample of forecast origins.
Its per-sample form is a first-order expansion about the detached batch mean whose
batch mean and gradient both match the exact estimator exactly -- see
`tests/test_losses/test_the_gtmse_surrogate_matches_the_exact_form.py`.

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
`DINOLoss` (and `iBOTPatchLoss`) maintain their centering EMA **inside
`call()`**, on a non-trainable `keras.Variable`. There is no `update_center()`
method and no custom `train_step` is needed.

**Do NOT follow `CLIPContrastiveLoss`'s structured-dict `y_pred` here.** These
losses do accept a `Dict[str, Tensor]` `y_pred`, but that form is **direct
invocation only** — it does not work under stock `compile(loss=...)` / `fit()`
on Keras 3.8. Measured: `CompileLoss.build` broadcasts one `Loss` object across
every leaf of a nested `y_pred` and then raises
`KeyError: "The path: ('student_logits',) in the 'loss' argument, can't be found
in either the model's output ('y_pred') or in the labels ('y_true')."`
(`CLIPContrastiveLoss` is not a counterexample: it has only ever run under
`src/train/clip/train_clip.py`'s hand-rolled loop, never under stock `fit()`.)

Under stock `fit()` the model must return a **single rank-2 tensor** in the
**packed** layout — last dimension `2 * out_dim`, holding
`concatenate([student_logits, teacher_logits], axis=-1)`. Build it with
`pack_student_teacher` from `src/dl_techniques/losses/dino_loss.py`, which is
the single source of truth for that layout. `y_true` is ignored.
`src/dl_techniques/models/vision/dino/training.py::DINOTrainingModel` already
returns this shape; see `src/dl_techniques/models/vision/dino/README.md` § "Rule 3"
for the full derivation and the two measured constraints on the packed form.

```python
from dl_techniques.losses import DINOLoss

dino_loss = DINOLoss(out_dim=65536)

# The model returns ONE tensor of width 2 * out_dim; y_true is ignored.
# outputs = pack_student_teacher(student_logits, teacher_logits)
model.compile(optimizer='adam', loss=dino_loss)
model.fit(train_ds, epochs=100)      # NOTE: no validation_data — see below
```

**Two rules these losses impose** (both MEASURED, both silent if violated):

1. **Do not pass `validation_data`.** The centering EMA fires on every
   invocation of `call()`, including validation batches, and
   `validation_batch_size` defaults to `batch_size` — so a validation pass
   inflates the number of centering updates per epoch and corrupts the
   statistic proportionally to the validation batch count. Use an evaluation
   callback (k-NN on frozen features) instead.
2. **The center's value is carried in `get_config()`.** Keras does not
   checkpoint loss-owned variables, so without this the centering statistic
   silently resets to zeros on every resume. The cost is a config blob
   proportional to `out_dim` (~1.3 MiB of JSON at `out_dim=65536`).

### Segmentation Losses
Recommended path: construct `SegmentationWrapperLoss` directly. The legacy
factory `create_segmentation_loss_function` still works and now delegates
to the same class.

**Key Params:** `loss_name` (str), `config` (`SegmentationLossConfig` / `LossConfig`).

```python
import keras
from dl_techniques.losses import SegmentationWrapperLoss, SegmentationLossConfig

# Configure parameters for segmentation
seg_config = SegmentationLossConfig(num_classes=19, focal_gamma=2.5)

# Recommended: construct the loss directly.
focal_tversky_loss = SegmentationWrapperLoss('focal_tversky', seg_config)

model.compile(optimizer='adam', loss=focal_tversky_loss)

# Save and reload — no custom_objects, no compile=False.
model.save('seg.keras')
reloaded = keras.models.load_model('seg.keras')
assert type(reloaded.loss).__name__ == 'SegmentationWrapperLoss'

# Legacy factory still works (delegates to SegmentationWrapperLoss):
# from dl_techniques.losses import create_segmentation_loss_function
# focal_tversky_loss = create_segmentation_loss_function('focal_tversky', seg_config)
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