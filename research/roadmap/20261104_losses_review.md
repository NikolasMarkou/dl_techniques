## Audit

### High severity

| Location | Finding |
|---|---|
| `segmentation_metrics.focal_per_sample` / `focal_scalar` | Both focal terms use the same `p_t` (true-class probability). The "negative" term `-(1-a) * p_t**g * log(1-p_t)` grows as `p_t -> 1`, so the loss has an interior minimum and penalizes confident correct predictions. It is not called by `SegmentationLosses.focal_loss`, but it is importable and documented as "the correct two-sided version". |
| `siglip_contrastive_loss.AdaptiveSigLIPLoss` | The entropy that drives the temperature update is computed on the unscaled `logits_per_image`, so it does not depend on the temperature. The control loop is open: the temperature drifts one way until it hits `min_temperature` or `max_temperature`. The `assign` also runs on validation batches, and `adaptive_temperature` is missing from `get_config`, so it resets on reload. |
| `image_restoration_loss.VGGLoss`, `lpips_loss.LPIPSLoss` | Inputs in `[0, 1]` get torch-style mean/std normalization. `keras.applications.VGG16/19` ImageNet weights expect caffe preprocessing (BGR, `0..255`, mean-pixel subtraction). Every feature is computed on out-of-distribution input. Use `keras.applications.vgg16.preprocess_input(x * 255.0)` (or `vgg19`). |
| `clustering_loss.ClusteringLoss` | `__init__` accepts neither `reduction` nor `**kwargs`, but `Loss.get_config()` emits `reduction`. `from_config(get_config())` raises `TypeError`. |
| `yolo12_multitask_loss`: `YOLOv12ObjectDetectionLoss`, `DiceFocalSegmentationLoss`, `ClassificationFocalLoss` | `super().__init__(reduction="none", **kwargs)` plus an emitted `reduction` key means `from_config` passes `reduction` twice and raises `TypeError`. `CliffordDetectionLoss.from_config` already works around this in its subclass, which shows the parent bug is real. |
| `YOLOv12MultiTaskLoss(use_uncertainty_weighting=True)` | The log-variance `keras.Variable`s live on a `Loss`. A `Loss` is not a tracked layer (consistent with the DINO measurement that loss-owned variables are not checkpointed), so these variables are most likely not in `model.trainable_variables` and stay at 0. Verify with one `fit` step. Also, an output of unrecognized rank silently gets a loss of `0.0`. |
| `any_loss.WeightedCrossEntropyWithAnyLoss` | The wrapper's `from_logits` applies only to its BCE. The wrapped `AnyLoss` keeps its own flag. The docstring example (`anyloss=F1Loss(), from_logits=True`) feeds raw logits to `F1Loss` as probabilities. |
| `multi_task_loss.MultiTaskLoss` | Takes dict `y_true` / `y_pred`. A single `Loss` object with a dict output fails under `compile(loss=...)` (the `KeyError` measured in `dino_loss`). The documented example does not run, and the loss returns a scalar. |

### Medium severity

| Location | Finding |
|---|---|
| `utilization_loss` | `-log(var)`, `-mean(norm)` and `-mean(var)` are unbounded below, so the minimizer is exploding activations. The docstrings describe a third `memory_vectors` / `entity_embeddings` argument that `call` does not take. |
| `multi_labels_loss.PerChannelBinaryLoss` | For the string fallback, `keras.losses.get(name)` already reduces the channel axis. `mean(axis=[0,1,2])` then yields a scalar, and the per-channel weights collapse. The `weighted_bce` closure base loss cannot be deserialized. |
| `decoupled_information_loss`, `focal_uncertainty_loss` analyzers | On the `from_logits=False` path, the analyzer clips but does not renormalize, while `call` does both. The analyzer's `total_loss` therefore disagrees with the live loss. |
| `nano_vlm_loss.NanoVLMLoss` | `if seq_len_true < 2` is a Python branch on a possibly dynamic dimension. The loss reimplements reductions itself: `reduction="none"` returns `(B, T-1)`, which Keras then reduces again. The default `ignore_index=0` drops token id 0, while sibling losses default to `-1`. |
| `image_restoration_loss.FrequencyLoss` | Any `norm` other than `"l1"` silently becomes L2. |
| `topographic_vae_loss` | `_pixels_per_frame` is set as an instance attribute and also redeclared as a class attribute, under a comment saying it is deliberately not a class attribute. The `keras.metrics.Mean` trackers updated in `call` are not reported by `fit`, and they also update on validation batches. |
| `segmentation_loss.boundary_loss`, `hausdorff_distance_loss` | Both are heuristics, not the Kervadec boundary loss or a Hausdorff distance. The boundary term rewards predicting 1 in every channel, background included, on edges. The "distance transform" is a 3-step dilation. |
| `quantile_loss.QuantileLoss.get_config` | Quantiles round-trip through float32 (`0.1 -> 0.10000000149011612`). This is the same defect class `dino_loss` fixed with float64 storage. |
| `feature_alignment_loss` | The loss jumps from `1 - m` to `0` at the margin. The module docstring states `max(0, m - sim)`. |
| `wasserstein_loss` | `WassersteinGradientPenaltyLoss.call` returns the inner loss's reduced scalar. `compute_gradient_penalty` hard-codes `axis=[1, 2, 3]`, so it only works on rank-4 input. |
| `focal_causal_lm_loss` | Materializes a `(B, T, V)` one-hot just to gather one log-prob. `take_along_axis` does the same without it. |

### Low severity / documentation

| Location | Finding |
|---|---|
| `multi_labels_loss` module docstring | Says `DiceLossPerChannel` was removed. It is defined and exported. |
| `__init__.__all__` | Omits `match_mask_axis`, which is imported. |
| `image_restoration_loss` | The citation is a placeholder (`author={...}`). The `VGGLoss` comment says "decreasing importance for deeper layers", but the weights increase with depth. |
| `omnipoint_losses` | Registered under per-class package strings, not the module path used everywhere else. |
| `clip_contrastive_loss` | The docstring points to `train.common.contrastive`. The README points to `src/train/clip/train_clip.py`. |
| `segmentation_loss.focal_tversky_loss` | Skips `_validate_inputs`, unlike its siblings. |