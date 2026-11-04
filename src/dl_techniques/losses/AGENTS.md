# Losses Package

Serializable Keras 3 loss objects, plus the plain-function helpers some of them need.

## Public API

Everything in `__all__` is importable from `dl_techniques.losses`:

```python
from dl_techniques.losses import (
    # Regression / forecasting
    HuberLoss, MASELoss, SMAPELoss, MQLoss, QuantileLoss, MultistepLoss,
    # Metric-as-loss and calibration
    AnyLoss, F1Loss, FBetaLoss, MCCLoss,
    BrierScoreLoss, SpiegelhalterZLoss, CombinedCalibrationLoss,
    # Self-supervised and contrastive
    CLIPContrastiveLoss, SigLIPContrastiveLoss, SymmetricInfoNCELoss,
    DINOLoss, iBOTPatchLoss, KoLeoLoss,
    # Generative
    WassersteinLoss, DDPMHybridLoss, TopographicVAELoss,
    # Segmentation and SAM
    SegmentationWrapperLoss, SAMMaskLoss, SAM2GatedMaskLoss, Sam3DetectionLoss,
    # Confidence regularizers
    GoodhartAwareLoss, DecoupledInformationLoss, FocalUncertaintyLoss,
)
```

The README table is the full inventory, grouped by domain.

**Module-only symbols.** These are NOT exported from the package; import them from their own module:

- `jacobian_symmetry.jacobian_symmetry_penalty`
- `thera_jacobian_tv.thera_tv_penalty`
- `thera_jacobian_tv.thera_total_loss`
- `segmentation_metrics.*`
- `sam2_video_loss.SAM2ObjectScoreLoss`
- `sam2_video_loss.SAM2IoULoss`
- `sam2_video_loss.select_lowest_loss_mask`
- the `sam3_detection_loss` pack constants and helpers

## Modules

- `affine_invariant_loss.py`: Scale- and shift-invariant L1 after median/MAD normalization. `masked=True` reads a validity mask from the last channel of `y_true` (Depth Anything V1, eq. 1-3).
- `any_loss.py`: AnyLoss framework. A differentiable confusion matrix via `A(p) = sigmoid(L * (p - 0.5))`, plus 16 metric losses and `WeightedCrossEntropyWithAnyLoss`. The confusion matrix is global over the batch, so every class returns a scalar.
- `brier_spiegelhalters_ztest_loss.py`: Binary-only calibration losses and metrics. Ships both Z-statistics behind `statistic=`: Spiegelhalter's `(1-2p)`-weighted Z (the default) and calibration-in-the-large (the pre-2026-09-02 behavior). The penalty is chance-corrected (`relu(Z^2 - 1)`), divided by `N`, and decomposed per row.
- `capsule_margin_loss.py`: CapsNet margin loss, plus an analysis helper.
- `chamfer_loss.py`: Symmetric Chamfer distance. Builds an O(N*M) pairwise matrix.
- `clifford_detection_loss.py`: `YOLOv12ObjectDetectionLoss` with configurable strides. Overrides `from_config` to drop the duplicate `reduction` key.
- `clip_contrastive_loss.py`: Symmetric CLIP loss over a dict or tuple of logit matrices. Direct call only; see Conventions.
- `clustering_loss.py`: Intra-cluster MSE plus a batch-level uniform-distribution penalty, a metrics dataclass, and a callback.
- `colbert_loss.py`: ColBERT v1 pairwise softmax CE over `nway` MaxSim scores (positive at index 0), and v2 log-target KL distillation with batchmean semantics.
- `ddpm_hybrid_loss.py`: DDPM `LEARNED_RANGE` objective: epsilon MSE plus a variational bound with a frozen mean. `y_true` packs noise, `x_start` and `t` on the channel axis. `x_t` is re-derived from `DDPMSchedule`.
- `decoupled_information_loss.py`: CE minus per-sample conditional entropy minus a batch-marginal entropy, plus an analysis helper.
- `dino_loss.py`: `DINOLoss` / `iBOTPatchLoss` (center EMA inside `call`, schedulable `teacher_temp` Variable, center carried in `get_config`) and `KoLeoLoss`. `pack_student_teacher` is the single source of truth for the packed layout.
- `doc_scanner_flow_sequence_loss.py`: DocScanner Eq. 9-14. **The only module in `losses/` that imports from `models/`**; see the D-031 anchor.
- `feature_alignment_loss.py`: Margin-thresholded cosine alignment for distillation.
- `flow_matching_velocity_loss.py`: Velocity MSE for rectified flow. Time weighting belongs in the trainer.
- `focal_causal_lm_loss.py`: Focal token CE with an ignore index. Rows are scaled so the batch mean equals the token-pool mean.
- `focal_uncertainty_loss.py`: Categorical focal loss minus per-sample entropy, plus an analysis helper.
- `goodhart_loss.py`: CE plus a per-sample confidence penalty (Pereyra 2017), and an optional batch-level `KL(class_prior || mean_p)` term. Arguments after `entropy_weight` are keyword-only; the removed `mi_weight` raises.
- `harmonic_loss.py`: `harmonic_logits(x, w, n, eps) = -(n/2) * log(||x - w||^2 + eps)` computed in at least float32, and `HarmonicCausalLMLoss` (`from_logits` fixed to `True`).
- `hrm_loss.py`: HRM loss: LM term plus Q-halt and Q-continue BCE terms. `StableMaxCrossEntropy` is per-token by design (D-002), and the `reduction="none"` on its sub-loss is load-bearing (D-001).
- `huber_loss.py`: Huber loss, reduced over all non-batch axes.
- `image_restoration_loss.py`: `CharbonnierLoss`, `FrequencyLoss`, `EdgeLoss`, `VGGLoss`, `EnhanceLoss`, `DarkIRCompositeLoss`. SSIM is gated on its weight because `tf.image.ssim` raises below 11x11 (D-044).
- `infonce_loss.py`: Single-tower symmetric InfoNCE. Positives are positional; a static batch of 1 raises.
- `jacobian_symmetry.py`: Stochastic double-VJP Jacobian-asymmetry penalty. Forces float32 and uses `training=False` (D-005). No TF context is opened at import time (D-015).
- `lightglue_loss.py`: LightGlue layer-weighted balanced NLL (`call`, from packed labels) plus token-confidence BCE in logit space (`compute`, for `add_loss`).
- `lpips_loss.py`: LPIPS-flavored perceptual distance over a frozen VGG16/19. The extractor is built lazily on first call.
- `mambalct_loss.py`: `l1_weight * L1 + giou_weight * (1 - GIoU)` over per-frame boxes, averaged over the clip inside each row.
- `mase_loss.py`: MASE with a per-row numerator over a batch-global naive denominator (deliberately not per row).
- `masked_causal_lm_loss.py`: Masked token CE and its prefix alias. Rows are scaled so the batch mean equals the token-pool mean.
- `mean_shift_instance_loss.py`: Per-row pull/push instance-embedding loss. Subsamples to `max_points` only when `N` is static.
- `multi_labels_loss.py`: `PerChannelBinaryLoss` (adaptive per-channel weighting), element-wise `WeightedBinaryFocalLoss`, `DiceLossPerChannel`, and a factory.
- `multi_task_loss.py`: Weighted per-task MSE over dict outputs.
- `multistep_loss.py`: ADAM `mseh` / `tmse` / `gtmse` / `msce` over the horizon axis. `gtmse` is a first-order surrogate about the detached batch mean, exact in value and gradient. The factory raises on unknown kwargs.
- `nano_vlm_loss.py`: Shifted next-token CE scored on the trailing label-length positions (drops the vision prefix). Reimplements reduction internally.
- `omnipoint_losses.py`: OmniPoint ray / distance / scale decoupling, finite-difference normal and Laplacian consistency (D-004), mask BCE, `compute_optimal_scale`, and the tuple-in `OmniPointCombinedLoss` callable. Scale alignment uses the ground-truth ray (D-026).
- `quantile_loss.py`: Single-quantile `MQLoss` and vectorized multi-quantile `QuantileLoss` (optionally normalized, with the scale floored at 1.0).
- `reid_cosine_loss.py`: Batch-hard softmargin triplet loss, and NumPy `magnet_loss_fn` (batch-level, deliberately not a `Loss`).
- `sam2_video_loss.py`: Ground-truth-gated SAM 2 mask loss with focal in logit space (D-072), object-score BCE, IoU loss, and `frame_sum` (D-285).
- `sam3_detection_loss.py`: SAM 3 Hungarian matcher plus the six-term detection loss. Owns the packed layout constants. Uses `tf.py_function`, so it is incompatible with XLA.
- `sam_mask_loss.py`: `SAMMaskLoss` (focal + dice through layout adapters that refuse the raw mask stack) and `SAMIoULoss`. `match_mask_axis` is the single owner of the GT / round alignment.
- `scaled_mse_loss.py`: MSE after resizing `y_true` to each prediction scale.
- `segmentation_loss.py`: `LossConfig` and the `SegmentationLosses` calculator (scalar returns, kept for SAM 1). `create_loss_function` is deprecated.
- `segmentation_metrics.py`: Per-sample and scalar IoU, Dice, Tversky, focal-Tversky, focal and CE backends.
- `segmentation_wrapper_loss.py`: Serializable name-dispatched wrapper over `SegmentationLosses` (D-002).
- `siamese_tracking_loss.py`: SiamFC logistic loss with an ignore ring, DaSiamRPN classification / regression with packed masks, and `create_siamfc_label`.
- `siglip_contrastive_loss.py`: SigLIP, adaptive SigLIP, and a hybrid with a cross-modal denoising penalty (not score matching). The factory default for learnable temperature differs from the class default, deliberately.
- `smape_loss.py`: SMAPE in `[0, 200]`, per sample.
- `sparsemax_loss.py`: Fenchel-Young sparsemax loss. The boundary cast to `self.dtype` is load-bearing (D-015).
- `superpoint_loss.py`: 65-class detector CE and the descriptor double hinge (`compute()` takes the real correspondence; `call()` assumes identity).
- `tabm_loss.py`: TabM ensemble loss, returning one value per input row in both sharing modes (D-027, D-064).
- `thera_jacobian_tv.py`: THERA `mean(|J|)` aliasing penalty and the `recon + tv_weight * TV` combiner.
- `topographic_vae_loss.py`: Topographic VAE ELBO with two KL terms. The `u` pair is conditional; half a pair raises. Read `effective_beta`.
- `utilization_loss.py`: MANN and GNN anti-collapse regularizers (scalar returns).
- `wasserstein_loss.py`: WGAN critic / generator losses, `compute_gradient_penalty` (rank 4 only), a 1-D Wasserstein divergence, and factories.
- `yolo12_multitask_loss.py`: YOLOv12 detection (TAL + CIoU + DFL), Dice+Focal segmentation, classification focal, and the rank-routed multi-task orchestrator. Re-keyed with `legacy_packages=("yolov12_losses",)` (D-005).

## Conventions

### Base class

Every loss inherits `keras.losses.Loss` and implements `call(self, y_true, y_pred)` and `get_config()`.

**Exceptions.** Plain functions (`jacobian_symmetry`, `thera_jacobian_tv`, `segmentation_metrics`, `magnet_loss_fn`) and `OmniPointCombinedLoss`, a registered plain callable over tuples.

### `call()` returns `(batch,)`, never a scalar

Keras computes `values * sample_weight` before reducing. A scalar return therefore yields `batch_loss * mean(sample_weight)` and makes `reduction=` dead. The result is a plausible wrong number, not an error.

The modules that still violate this are recorded by `tests/test_losses/test_the_premature_scalar_family_is_pinned.py`, which goes red when one is fixed. Do not copy that list into prose.

### A correct shape is not enough

The family predicate `weighted == unweighted * mean(w)` cannot see a batch-level term added into an already-`(batch,)` vector. Any loss with a whole-batch term needs a per-loss test asserting:

```
loss(w=[1, ..., 1, 0]) * B / (B - 1) == loss(batch[:-1])
```

`tests/test_losses/test_goodhart_loss.py` shows the pattern. Document the coupling in the docstring.

### One loss object sees one tensor

`compile(loss=obj)` with a dict model output broadcasts `obj` across the leaves and raises `KeyError` (measured, Keras 3.8). A loss that needs several tensors takes ONE packed tensor, and its pack / unpack helpers live in the loss module as the single source of truth. The packing reference implementations are:

- `pack_student_teacher`
- `pack_matches`
- the `PACKED_*` constants in `sam3_detection_loss`
- `DDPMHybridLoss`'s `y_true`

Never pass side information through `sample_weight`; Keras multiplies by it.

### Serialization round-trips

`Loss.get_config()` emits `name` and `reduction`, so `cls.from_config(loss.get_config())` must work. That gives two rules:

- A constructor that does not accept `reduction` (or `**kwargs`) breaks the round trip.
- A constructor that hard-codes `reduction` in `super().__init__` must override `from_config` to pop the key, as `CliffordDetectionLoss` does.

Test every new loss with this round trip.

Store float hyperparameters as Python floats. Do not round-trip them through a float32 tensor, which changes `0.05` to `0.0500000007`; `dino_loss` uses float64 storage for exactly this reason.

### State on a loss

- **Not checkpointed.** Variables owned by a `Loss` are not saved by `.keras` checkpoints (measured in `dino_loss`, D-007). Carry required state in `get_config()`.
- **Not trained.** Do not expect a `trainable=True` variable on a `Loss` to be optimized by `fit()`: the loss is not a tracked layer. Put learnable parameters in the model.
- **Updated on validation.** State assigned inside `call()` also updates on validation batches. Document this, as `DINOLoss` does.
- **Schedulable scalars.** A Python float read inside `call()` is constant-folded into the traced step. A scalar meant to change during `fit()` must be a `keras.Variable` with a setter (D-022).

### Graph safety

- Never branch in Python on a tensor value. Branch only on static shapes or config.
- Keep constant tables as NumPy and convert them inside `call()`. A tensor materialized during tracing binds to that `FuncGraph`.

### Numerics

- **Compute dtype.** `Loss.__call__` casts inputs to `floatx()`, but a direct `.call()` does not. Compute `log`, `exp` and large sums in at least float32.
- **Dtype names.** Read a dtype name with `getattr(dtype, "name", None) or str(dtype)`, never with `keras.backend.standardize_dtype` (banned repo-wide).
- **Stay in logit space.** Prefer `softplus(-x) + x * (1 - t)` and `log_softmax` over clipping probabilities. `ops.clip` has zero gradient outside its range; that killed the SAM 2 mask head (D-072). The textbook `max(x, 0) - x*t + log1p(exp(-|x|))` autodiffs wrongly at exactly `x = 0`.
- **Pretrained feature extractors.** Use the backbone's own `preprocess_input`. Keras VGG weights expect caffe preprocessing, not torch mean/std.

### Registration

Every `Loss` subclass carries `@register_dl_technique("dl_techniques.losses.<module>")`, never a bare `register_keras_serializable`. Re-keying an explicitly-packaged class requires `legacy_packages=(...)`, or old archives stop loading.

### Factories

Raise on unknown keyword arguments (the house rule; see `create_multistep_loss`).

### Imports

- New files use `import keras` and qualify at the call site. Leave existing `from keras import ops` files alone.
- `losses/` must not import from `models/`, except `doc_scanner_flow_sequence_loss.py` (D-031).
- `import tensorflow` only where a module documents why: `jacobian_symmetry`, `wasserstein_loss`, `image_restoration_loss` (SSIM), `sam3_detection_loss` (`py_function`).

### Docstrings

Both Google `Args:` and Sphinx `:param` styles are live. Match the file you edit; never convert a file wholesale. Cite the reference implementation and paper equation for every transcribed formula, and sign every divergence from it.

### Documentation

Before adding a symbol to a README row, check it with `hasattr(dl_techniques.losses, name)`. Keep the README table and `__all__` in sync.

## Re-deriving counts

Run these instead of trusting any number in a doc:

```bash
ls src/dl_techniques/losses/*.py | grep -v __init__ | wc -l
grep -rl register_dl_technique src/dl_techniques/losses --include=*.py | wc -l
grep -rl "^from keras import ops" src/dl_techniques/losses --include=*.py | wc -l
grep -rlE "^ +Args:$" src/dl_techniques/losses --include=*.py | wc -l
grep -rl ":param " src/dl_techniques/losses --include=*.py | wc -l
```

## Testing

Tests in `tests/test_losses/`.
