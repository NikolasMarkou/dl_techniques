# Losses Module

`dl_techniques.losses` is a collection of Keras 3 loss functions. Every `keras.losses.Loss` subclass is registered with `register_dl_technique`, so a model compiled with one saves and reloads without `custom_objects`.

## Contracts shared by every loss

**One value per sample.** `call()` returns shape `(batch,)`. Keras multiplies `call()`'s output by `sample_weight` before reducing, so a scalar return charges every row the batch aggregate and makes `reduction=` a dead knob. Losses that still return a scalar are pinned by `tests/test_losses/test_the_premature_scalar_family_is_pinned.py`.

**Batch-coupled terms.** Some losses carry a term computed over the whole batch. Setting a row's `sample_weight` to 0 removes that row's own contribution, but the row still shifts every other row through the batch term. These losses are:

- the `DINOLoss` / `iBOTPatchLoss` center;
- `KoLeoLoss`;
- `GoodhartAwareLoss` at `prior_weight > 0`;
- the `DecoupledInformationLoss` diversity term;
- `MultistepLoss("gtmse")`;
- the calibration Z-statistics;
- `ClusteringLoss`;
- the utilization losses.

**One loss, one tensor.** A single loss object given a dict model output fails under `compile(loss=...)`. Keras broadcasts the object across every leaf and raises `KeyError`. Losses that need several tensors therefore take one packed tensor:

| Loss | Packing |
|---|---|
| `DINOLoss` | `pack_student_teacher` |
| `LightGlueLoss` | `pack_matches` |
| `DDPMHybridLoss` | noise, `x_start` and `t` packed in `y_true` |
| `DocScannerFlowSequenceLoss` | `[f_gt, g]` packed in `y_true` |
| `Sam3DetectionLoss` | packed query and target rows |
| `SAMIoULoss` | `iou_supervision` |
| Siamese tracking losses | values and masks packed in `y_true` |

**Stateful losses update on every call, validation included.** This covers `DINOLoss`, `iBOTPatchLoss`, `AdaptiveSigLIPLoss` and the `TopographicVAELoss` trackers. Loss-owned variables are not saved in `.keras` checkpoints; `DINOLoss` carries its center in `get_config()` for that reason, and `AdaptiveSigLIPLoss` carries its live temperature there as `current_temperature`. Keras 3.8 passes `Loss.call` no `training` flag, so `evaluate` still moves the adaptive temperature; that drift is a known limitation, not gated.

## Available losses

Symbols in the table are importable from `dl_techniques.losses` unless the row says otherwise.

| Domain | Symbols | Notes |
|---|---|---|
| Robust regression | `HuberLoss`, `ScaledMseLoss` | `ScaledMseLoss` resizes `y_true` to each deep-supervision output. |
| Forecasting | `MASELoss`, `mase_metric`, `SMAPELoss`, `smape_metric`, `MQLoss`, `QuantileLoss` | MASE uses a batch-global naive scale (an approximation of canonical MASE). `QuantileLoss(normalize=True)` floors the scale at 1.0. |
| Multistep forecasting | `MultistepLoss`, `create_multistep_loss`, `MULTISTEP_AGGREGATIONS` | ADAM `mseh` / `tmse` / `gtmse` / `msce` over the horizon axis. See [Multistep](#multistep-losses). |
| Metric-as-loss (AnyLoss) | `AnyLoss`, `ApproximationFunction`, `AccuracyLoss`, `PrecisionLoss`, `RecallLoss`, `SpecificityLoss`, `F1Loss`, `FBetaLoss`, `BalancedAccuracyLoss`, `GeometricMeanLoss`, `YoudenJLoss`, `MCCLoss`, `CohenKappaLoss`, `IoULoss`, `DiceLoss`, `TverskyLoss`, `FocalTverskyLoss`, `WeightedCrossEntropyWithAnyLoss` | Differentiable confusion matrix (Han et al. 2024). The confusion matrix is global over the batch, so these return a scalar. `IoULoss` / `DiceLoss` / `TverskyLoss` belong here, not to `SegmentationWrapperLoss`. |
| Calibration (binary only) | `BrierScoreLoss`, `SpiegelhalterZLoss`, `CombinedCalibrationLoss`, `BrierScoreMetric`, `SpiegelhalterZMetric` | A calibration Z-statistic is not a proper scoring rule. Use it only as a small-weight regularizer, which `CombinedCalibrationLoss` provides. |
| Confidence regularizers | `GoodhartAwareLoss`, `analyze_loss_components`, `DecoupledInformationLoss`, `analyze_decoupled_information_loss`, `FocalUncertaintyLoss`, `analyze_focal_uncertainty_loss` | Entropy penalties on the predictions. None of them observes the input `X`. |
| Other classification | `SparsemaxLoss`, `CapsuleMarginLoss`, `capsule_margin_loss`, `analyze_margin_loss_components`, `MultiTaskLoss` | `SparsemaxLoss` requires `from_logits=True`. `MultiTaskLoss` takes dicts; see Known issues. |
| Language modeling | `MaskedCausalLMLoss`, `PrefixMaskedCausalLMLoss`, `FocalCausalLMLoss`, `HarmonicCausalLMLoss`, `harmonic_logits`, `StableMaxCrossEntropy`, `HRMLoss`, `create_hrm_loss`, `NanoVLMLoss` | Masked variants skip `ignore_index` (default `-1`; `NanoVLMLoss` defaults to `0`). Per-row values are scaled so the batch mean equals the token-pool mean. `StableMaxCrossEntropy` returns per-token values by design. |
| Contrastive (two towers) | `CLIPContrastiveLoss`, `SigLIPContrastiveLoss`, `AdaptiveSigLIPLoss`, `HybridContrastiveLoss`, `create_siglip_loss`, `create_adaptive_siglip_loss`, `create_hybrid_loss` | `create_siglip_loss` defaults `use_learnable_temperature=True`; the class defaults to `False`. `HybridContrastiveLoss` adds a cross-modal denoising penalty, not score matching. |
| Contrastive (one tower) | `SymmetricInfoNCELoss`, `create_symmetric_infonce_loss` | Positives are positional. Batch `N` gives `N - 1` negatives. Use `drop_remainder=True`. |
| Late-interaction retrieval | `ColBERTPairwiseSoftmaxLoss`, `ColBERTDistillationLoss` | v1: the positive is at index 0 of each `nway` group, and `y_true` is ignored. v2: KL divergence between log-distributions, batch-mean. |
| Self-supervised / distillation | `DINOLoss`, `iBOTPatchLoss`, `KoLeoLoss`, `FeatureAlignmentLoss` | See [DINO](#dino-loss). |
| Metric learning / clustering | `SoftmarginTripletLoss`, `magnet_loss_fn`, `MeanShiftInstanceLoss`, `ClusteringLoss`, `ClusteringMetrics`, `ClusteringMetricsCallback`, `compute_clustering_metrics` | `magnet_loss_fn` is NumPy and batch-level by design. `MeanShiftInstanceLoss` subsamples only when `N` is static. |
| Semantic segmentation | `SegmentationWrapperLoss`, `create_segmentation_wrapper_loss`, `SegmentationLosses`, `SegmentationLossConfig`, `create_segmentation_loss_function` (deprecated) | Name-dispatched over `cross_entropy`, `dice`, `focal`, `tversky`, `focal_tversky`, `lovasz`, `combo`, `boundary`, `hausdorff`. `boundary` and `hausdorff` are heuristics. |
| Multi-label segmentation | `PerChannelBinaryLoss`, `WeightedBinaryFocalLoss`, `WeightedBinaryCrossEntropyLoss`, `DiceLossPerChannel`, `create_multilabel_segmentation_loss` | Adaptive per-channel weighting from batch class presence. `PerChannelBinaryLoss` accepts `'binary_crossentropy'` or a `Loss` that returns one value per element (any other string, or a base that reduces the channel axis, raises). |
| SAM family | `SAMMaskLoss`, `SAMIoULoss`, `to_dice_layout`, `to_focal_layout`, `SAM2GatedMaskLoss`, `mask_presence_gate`, `Sam3DetectionLoss` | Module-only: `match_mask_axis`, `SAM2ObjectScoreLoss`, `SAM2IoULoss`, `select_lowest_loss_mask` and the SAM 3 pack helpers. `Sam3DetectionLoss` uses `scipy` via `tf.py_function` and needs `jit_compile=False`. |
| Detection | `YOLOv12MultiTaskLoss`, `create_yolov12_multitask_loss`, `create_yolov12_coco_loss`, `create_yolov12_crack_loss`, `CliffordDetectionLoss` | YOLOv12 routes each output to a sub-loss by tensor rank (2 classification, 3 detection, 4 segmentation). |
| Tracking | `SiamFCLogisticLoss`, `create_siamfc_label`, `DaSiamRPNClsLoss`, `DaSiamRPNRegLoss`, `MambaLCTBoxLoss` | The classification / regression balance belongs in `compile(loss_weights=...)`. |
| Keypoints | `SuperPointDetectorLoss`, `SuperPointDescriptorLoss`, `LightGlueLoss`, `lightglue_nll`, `lightglue_confidence_loss`, `pack_matches`, `unpack_matches` | `LightGlueLoss.call` covers the NLL only. `compute()` adds the confidence BCE, for use with `add_loss`. |
| Depth and 3D geometry | `AffineInvariantLoss`, `ChamferLoss`, `RayDirectionLoss`, `PointDistanceLoss`, `MetricScaleLoss`, `MaskLoss`, `NormalConsistencyLoss`, `LocalConsistencyLoss`, `OmniPointCombinedLoss`, `compute_optimal_scale` | `OmniPointCombinedLoss` is a plain callable over tuples. Its scale alignment uses the ground-truth ray, never the predicted one. |
| Document rectification | `DocScannerFlowSequenceLoss` | Eq. 9-14 of arXiv:2110.14968v2. The last iteration has weight 1. Needs `training=True` outputs. |
| Image restoration / perceptual | `CharbonnierLoss`, `FrequencyLoss`, `EdgeLoss`, `VGGLoss`, `EnhanceLoss`, `DarkIRCompositeLoss`, `LPIPSLoss` | SSIM uses `tf.image.ssim`, which raises below 11x11; set `ssim_weight=0` for small maps. See Known issues for VGG preprocessing. |
| Generative | `WassersteinLoss`, `WassersteinGradientPenaltyLoss`, `WassersteinDivergence`, `compute_gradient_penalty`, `create_wgan_losses`, `create_wgan_gp_losses`, `DDPMHybridLoss`, `FlowMatchingVelocityLoss`, `TopographicVAELoss`, `REQUIRED_PREDICTION_KEYS`, `VARIANCE_PREDICTION_KEYS` | `DDPMHybridLoss` freezes the mean prediction inside the variational bound. In `TopographicVAELoss`, `kl_loss_weight` is not the literature's `beta`; read `effective_beta`. |
| Tabular / utilization | `TabMLoss`, `MANNUtilizationLoss`, `GNNUtilizationLoss` | `TabMLoss` returns one value per input row in both `share_training_batches` modes. |
| Module-only regularizers | `jacobian_symmetry.jacobian_symmetry_penalty`, `thera_jacobian_tv.thera_tv_penalty`, `thera_jacobian_tv.thera_total_loss` | Plain functions, not exported. The Jacobian penalty forces float32 and runs the model with `training=False`. |
| Module-only metrics backend | `segmentation_metrics` (`*_per_sample`, `*_scalar`) | Backend for `SegmentationLosses` and `DiceLossPerChannel`. |

## Usage

### Basic

```python
from dl_techniques.losses import F1Loss, HuberLoss

model.compile(optimizer="adam", loss=F1Loss(amplifying_scale=73.0), metrics=["accuracy"])
model.compile(optimizer="adam", loss=HuberLoss(delta=1.5), metrics=["mae"])
```

### AnyLoss

The key parameters are `amplifying_scale` (default 73.0) and `from_logits` (default `False`). `FBetaLoss` adds `beta`; `TverskyLoss` adds `alpha` and `beta`.

```python
from dl_techniques.losses import FBetaLoss, WeightedCrossEntropyWithAnyLoss, F1Loss

f2 = FBetaLoss(beta=2.0)

# from_logits must be set on BOTH the wrapper and the wrapped loss.
combined = WeightedCrossEntropyWithAnyLoss(
    anyloss=F1Loss(from_logits=True), alpha=0.7, from_logits=True
)
```

### Contrastive losses

`CLIPContrastiveLoss` and `SigLIPContrastiveLoss` take a dict with `logits_per_image` and `logits_per_text`. Call them directly; do not pass them to `compile` for a dict-output model.

```python
from dl_techniques.losses import CLIPContrastiveLoss

per_sample = CLIPContrastiveLoss().call(
    None, {"logits_per_image": image_logits, "logits_per_text": text_logits}
)
```

For stock `fit()`, the CLIP trainer stacks the two matrices into one `(N, 2, N)` output and unstacks them inside the loss. Use the helpers named in `CLIPContrastiveLoss`'s docstring.

### DINO loss

The model must return ONE rank-2 tensor of width `2 * out_dim`, built with `pack_student_teacher` from `dino_loss.py`. `y_true` is ignored. The centering EMA runs inside `call()`, so no custom `train_step` is needed.

```python
from dl_techniques.losses import DINOLoss

dino_loss = DINOLoss(out_dim=65536)
model.compile(optimizer="adam", loss=dino_loss)
model.fit(train_ds, epochs=100)  # no validation_data
dino_loss.set_teacher_temp(0.07)  # takes effect inside the running fit
```

Rules, each measured and each silent if broken:

1. Do not pass `validation_data`. Validation batches also advance the center EMA. Evaluate with a callback instead, for example k-NN on frozen features.
2. Change the teacher temperature only through `set_teacher_temp()`. A plain attribute assignment raises on purpose.
3. The center is serialized through `get_config()` (about 1.3 MiB of JSON at `out_dim=65536`).

### Segmentation

```python
import keras
from dl_techniques.losses import SegmentationWrapperLoss, SegmentationLossConfig

loss = SegmentationWrapperLoss(
    "focal_tversky", SegmentationLossConfig(num_classes=19, focal_gamma=2.5)
)
model.compile(optimizer="adam", loss=loss)
model.save("seg.keras")
reloaded = keras.models.load_model("seg.keras")  # no custom_objects
```

`create_segmentation_loss_function` emits a `DeprecationWarning` and delegates to `SegmentationWrapperLoss`.

### Multistep losses

```python
from dl_techniques.losses import MultistepLoss, create_multistep_loss

model.compile(optimizer="adam", loss=MultistepLoss("gtmse", h=12))
model.compile(optimizer="adam", loss=create_multistep_loss("msce", h=4))
```

Inputs are `(batch, horizon)` or `(batch, horizon, features)`. Quantile outputs are not supported.

- **`mseh` starves direct models.** On a model that emits all horizons in one pass, every horizon column except `h` receives exactly zero gradient. Pinned by `tests/test_losses/test_the_mseh_starves_other_horizons.py`.
- **No shrinkage claim.** The parameter-shrinkage result is proven for recursive additive ETS/ARIMA only.
- **`gtmse` depends on the batch.** It takes the log of a batch-level mean. Its per-sample surrogate matches the exact value and gradient (`tests/test_losses/test_the_gtmse_surrogate_matches_the_exact_form.py`).

### Wasserstein GAN

WGAN-GP needs a hand-written training loop, not a `train_step` override.

```python
import tensorflow as tf
from dl_techniques.losses import create_wgan_gp_losses, compute_gradient_penalty

critic_loss_fn, generator_loss_fn = create_wgan_gp_losses(lambda_gp=10.0)

with tf.GradientTape() as tape:
    w_loss = critic_loss_fn(labels, critic_scores)  # labels: 1 real, 0 fake
    gp = compute_gradient_penalty(critic, real_images, fake_images)  # rank-4 inputs only
    total = w_loss + gp
grads = tape.gradient(total, critic.trainable_variables)
optimizer.apply_gradients(zip(grads, critic.trainable_variables))
```

`lambda_gp` is stored on the loss but applied only by `compute_gradient_penalty`.

### YOLOv12 multi-task

```python
from dl_techniques.losses import create_yolov12_multitask_loss

loss = create_yolov12_multitask_loss(
    tasks=["detection", "segmentation"],
    num_detection_classes=80,
    num_segmentation_classes=80,
    input_shape=(640, 640),
)
model.compile(optimizer="adam", loss=loss)
```

One loss object is applied to every output. It picks the sub-loss by tensor rank, not by output name. An output of any other rank, or of a task that is not enabled, raises `ValueError`. There is no learnable uncertainty weighting (a `Loss` is not a tracked layer, so its variables never train); `use_uncertainty_weighting` is refused with `TypeError`, and the task weights are the static `*_weight` arguments.
