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
    ConfidencePenaltyLoss, DecoupledInformationLoss, FocalUncertaintyLoss,
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
- `any_loss.py`: AnyLoss framework. A differentiable confusion matrix via `A(p) = sigmoid(L * (p - 0.5))`, plus 15 metric losses and `WeightedCrossEntropyWithAnyLoss`. The confusion matrix is global over the batch, so every class returns a scalar.
- `brier_spiegelhalters_ztest_loss.py`: Binary-only calibration losses and metrics. Ships both Z-statistics behind `statistic=`: Spiegelhalter's `(1-2p)`-weighted Z (the default) and calibration-in-the-large (the pre-2026-09-02 behavior). The penalty is chance-corrected (`relu(Z^2 - 1)`), divided by `N`, and decomposed per row.
- `capsule_margin_loss.py`: CapsNet margin loss, plus an analysis helper.
- `chamfer_loss.py`: Symmetric Chamfer distance. Builds an O(N*M) pairwise matrix.
- `clifford_detection_loss.py`: `YOLOv12ObjectDetectionLoss` with configurable strides. Inherits the `reduction`-dropping `from_config` from `yolo12_multitask_loss.py`.
- `clip_contrastive_loss.py`: Symmetric CLIP loss over a dict or tuple of logit matrices. Direct call only; see Conventions.
- `clustering_loss.py`: `ClusteringLoss` returns `(batch,)`: each row is its intra-cluster MSE plus the batch-level uniform-distribution penalty, which is one constant repeated in every row (batch-coupled: a `sample_weight` scales that term by `mean(w)` and a zero-weighted row keeps its share of it). Also a metrics dataclass and a callback.
- `colbert_loss.py`: ColBERT v1 pairwise softmax CE over `nway` MaxSim scores (positive at index 0), and v2 log-target KL distillation with batchmean semantics.
- `confidence_penalty_loss.py`: `ConfidencePenaltyLoss`, CE (explicit `-sum(y * log p)`, not Keras's fused op) minus a per-sample entropy bonus (Pereyra 2017), plus `analyze_confidence_penalty_loss`. A uniform `1/K` target row is an outlier-exposure row (Hendrycks 2019); an all-zero row is ignored with exactly zero value and gradient. Nothing couples the rows. It replaced `GoodhartAwareLoss`: `prior_weight`, `class_prior` and `mi_weight` are gone and raise `TypeError`, and a `.keras` archive compiled with the old loss no longer loads with `compile=True` (it loads with `compile=False`; an exploration scan found 0 of 68 archives under `results/` carrying it). The post-hoc temperature fit lives in `utils/temperature_scaling.py`.
- `ddpm_hybrid_loss.py`: DDPM `LEARNED_RANGE` objective: epsilon MSE plus a variational bound with a frozen mean. `y_true` packs noise, `x_start` and `t` on the channel axis. `x_t` is re-derived from `DDPMSchedule`.
- `decoupled_information_loss.py`: CE minus per-sample conditional entropy minus a batch-marginal entropy, plus an analysis helper.
- `dino_loss.py`: `DINOLoss` / `iBOTPatchLoss` (center EMA inside `call`, schedulable `teacher_temp` Variable, center carried in `get_config`) and `KoLeoLoss`. `pack_student_teacher` is the single source of truth for the packed layout.
- `doc_scanner_flow_sequence_loss.py`: DocScanner Eq. 9-14. **The only module in `losses/` that imports from `models/`**; see the D-031 anchor.
- `feature_alignment_loss.py`: Margin-thresholded cosine alignment for distillation (Depth Anything Eq. 9: a step at the margin, not a hinge).
- `flow_matching_velocity_loss.py`: Velocity MSE for rectified flow. Time weighting belongs in the trainer.
- `focal_causal_lm_loss.py`: Focal token CE with an ignore index. Rows are scaled so the batch mean equals the token-pool mean, which couples them through the batch-wide token count: a zero `sample_weight` is not a row drop.
- `focal_uncertainty_loss.py`: Categorical focal loss minus per-sample entropy, plus an analysis helper.
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
- `masked_causal_lm_loss.py`: Masked token CE and its prefix alias. Rows are scaled so the batch mean equals the token-pool mean, which couples them through the batch-wide token count: a zero `sample_weight` is not a row drop.
- `mean_shift_instance_loss.py`: Per-row pull/push instance-embedding loss. Subsamples to `max_points` only when `N` is static.
- `multi_labels_loss.py`: `PerChannelBinaryLoss` (adaptive per-channel weighting; base must be `'binary_crossentropy'` or element-wise; returns `(batch,)` but is batch-coupled, because the channel weights come from the class-presence counts of the whole batch), element-wise `WeightedBinaryFocalLoss` and `WeightedBinaryCrossEntropyLoss`, `DiceLossPerChannel` (returns `(batch,)` through `segmentation_metrics.dice_per_sample`, row-local), and a factory.
- `multistep_loss.py`: ADAM `mseh` / `tmse` / `gtmse` / `msce` over the horizon axis. `gtmse` is a first-order surrogate about the detached batch mean, exact in value and gradient. The factory raises on unknown kwargs.
- `nano_vlm_loss.py`: Shifted next-token CE scored on the trailing label-length positions (drops the vision prefix). Reimplements reduction internally. Same row coupling as `MaskedCausalLMLoss` (a zero `sample_weight` keeps the row's tokens in the denominator).
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
- `sparse_segmentation_loss.py`: `SparseDiceLoss`, `SparseFocalLoss`, `SparseCeDiceLoss` on logits `(B,H,W,C)` and sparse integer targets with an `ignore_index` that changes neither value nor gradient. They one-hot the targets and call `segmentation_metrics.dice_per_sample` / `focal_per_sample` (MONAI-style per-image Dice, 1e-5 smoothing in both terms; Lin et al. softmax focal, `gamma=0` is CE, no `alpha`). `call()` returns `(batch,)` per image; the stock `SparseCategoricalCrossentropy(ignore_class=)` divides by the batch's valid-pixel count instead. Sources: D-014 and D-024 of plan-2026-11-03T221936-98b866fb.
- `siamese_tracking_loss.py`: SiamFC logistic loss with an ignore ring, DaSiamRPN classification / regression with packed masks, and `create_siamfc_label`.
- `siglip_contrastive_loss.py`: SigLIP, adaptive SigLIP, and a hybrid with a cross-modal denoising penalty (not score matching). The factory default for learnable temperature differs from the class default, deliberately.
- `smape_loss.py`: SMAPE in `[0, 200]`, per sample.
- `sparsemax_loss.py`: Fenchel-Young sparsemax loss. The boundary cast to `self.dtype` is load-bearing (D-015).
- `superpoint_loss.py`: 65-class detector CE and the descriptor double hinge (`compute()` takes the real correspondence; `call()` assumes identity).
- `tabm_loss.py`: TabM ensemble loss, returning one value per input row in both sharing modes (D-027, D-064).
- `thera_jacobian_tv.py`: THERA `mean(|J|)` aliasing penalty and the `recon + tv_weight * TV` combiner.
- `topographic_vae_loss.py`: Topographic VAE ELBO with two KL terms. The `u` pair is conditional; half a pair raises. Read `effective_beta`.
- `utilization_loss.py`: MANN and GNN anti-collapse regularizers; both return `(batch,)`. `GNNUtilizationLoss` is row-local. `MANNUtilizationLoss` couples rows only through its temporal-variance term (a detached first-order surrogate, as in `MultistepLoss`'s `gtmse`).
- `wasserstein_loss.py`: WGAN critic / generator losses, `compute_gradient_penalty` (sums every non-batch axis, any rank of 2 or more), a 1-D Wasserstein divergence, and factories.
- `yolo12_multitask_loss.py`: YOLOv12 detection (TAL + CIoU + DFL), Dice+Focal segmentation, classification focal, and the rank-routed multi-task orchestrator. Re-keyed with `legacy_packages=("yolov12_losses",)` (D-005).

## Conventions

### Base class

Every loss inherits `keras.losses.Loss` and implements `call(self, y_true, y_pred)` and `get_config()`.

**Exceptions.** Plain functions (`jacobian_symmetry`, `thera_jacobian_tv`, `segmentation_metrics`, `magnet_loss_fn`) and `OmniPointCombinedLoss`, a registered plain callable over tuples.

### `call()` returns `(batch,)`, never a scalar

Keras computes `values * sample_weight` before reducing. A scalar return therefore yields `batch_loss * mean(sample_weight)` and makes `reduction=` dead. The result is a plausible wrong number, not an error.

`tests/test_losses/test_the_premature_scalar_family_is_pinned.py` accounts for every `Loss` subclass in `losses/` by a closure census that fails when a class is in no bucket or in two. The four buckets are `KNOWN_GOOD` (measured by the defect predicate and found per-sample; `STILL_BROKEN` is empty, and a floor on the total keeps a measured row from disappearing), `DOCUMENTED_SCALAR` (the `any_loss.py` metric losses, scalar by design), `KNOWN_SCALAR_NOT_YET_FIXED` and `NOT_PINNED` (each entry carries the reason nothing measures it; this is not a clearance). `SAMMaskLoss` and `SAM2GatedMaskLoss` are the two classes measured to be premature scalars that are still unfixed. Fixing one moves it to `KNOWN_GOOD` with a builder row. Do not copy the bucket lists into prose.

### A correct shape is not enough

The family predicate `weighted == unweighted * mean(w)` cannot see a batch-level term added into an already-`(batch,)` vector. Any loss with a whole-batch term needs a per-loss test asserting:

```
loss(w=[1, ..., 1, 0]) * B / (B - 1) == loss(batch[:-1])
```

`tests/test_losses/test_confidence_penalty_loss.py` shows the pattern (`test_zeroing_a_sample_weight_equals_dropping_the_row`). Document the coupling in the docstring.

Three per-sample losses are coupled through one batch term and have their own documented form of that identity, which holds with the batch statistic computed over the FULL batch (not over the kept rows):

- `PerChannelBinaryLoss`: `loss(w=[1, ..., 1, 0]) * B / (B - 1)` is the mean of the first `B - 1` rows scored with channel weights from all `B` images, and it differs from `loss(batch[:-1])`. Pin: `test_a_zero_sample_weight_keeps_the_row_s_class_counts_in_the_weights` in `tests/test_losses/test_multi_labels_loss.py`.
- `ClusteringLoss`: the same expression equals the kept rows' mean distance term plus the balance penalty of the full batch (each row carries the same constant, so `sample_weight` scales it by `mean(w)`). Pin: `test_a_zero_sample_weight_does_not_drop_the_row_s_share_of_the_balance_penalty` in `tests/test_losses/test_clustering_loss.py`.
- `MANNUtilizationLoss`: the same expression equals the kept rows scored against the full-batch temporal variance and mean. With `variance_weight=0` the plain identity does hold. Pin: `test_with_the_variance_term_the_drop_a_row_identity_does_not_hold` in `tests/test_losses/test_utilization_loss.py`.

`GNNUtilizationLoss` and `DiceLossPerChannel` are fully row-local and satisfy the plain identity.

The token-mean language-model losses (`NanoVLMLoss`, `MaskedCausalLMLoss` and its subclasses, `FocalCausalLMLoss`, the `HRMLoss` LM term) are such losses, and the identity above does NOT hold for them: each row is `S_i / N * B` with `N` the batch-wide valid-token count, and `call()` never sees `sample_weight`, so a zero weight keeps the row's tokens in `N`. The pin for them is `loss(w) == loss(batch without row d) * (N - n_d) / N`, written against an independent float64 reference (`tests/test_losses/test_the_nano_vlm_loss_scores_the_text_positions.py`). Never write "`sample_weight` selects rows" in these docstrings.

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
- A constructor that hard-codes `reduction` in `super().__init__` must override `from_config` to pop the key. The three YOLOv12 sub-losses share one such override, `_FixedNoReductionConfig` in `yolo12_multitask_loss.py` (D-006); `CliffordDetectionLoss` inherits it.

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

Every `Loss` subclass carries `@register_dl_technique("dl_techniques.losses.<module>")`, never a bare `register_keras_serializable`. Re-keying an explicitly-packaged class requires `legacy_packages=(...)`, or old archives stop loading. Two modules do so: `yolo12_multitask_loss.py` (`"yolov12_losses"`) and `omnipoint_losses.py` (the per-class keys `dl_techniques.losses.omnipoint_losses.<snake_case>`, D-007). `tests/test_losses/test_the_losses_package_surface_is_consistent.py` fails on any registered key that is not the module path.

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
