---
name: project-convnext-patch-vae
description: ConvNeXtPatchVAE scope audit session findings -- key exogeneity candidates and cross-component couplings discovered in Phase 0.7 (two sessions)
metadata:
  type: project
---

## Session 1 (analysis_2026-05-26_0e294d8a) — augmentation pipeline focus

Key cross-component couplings found (all outside initial scope S of train script + model files):

1. SIGRegLayer scales statistic by N=Hp*Wp -- augmentation patch diversity collapse impairs SIGReg; lambda_sigreg is not calibrated per grid size. Critical for large-image configs (N varies 16x between CIFAR and 256px/patch=8).

2. ConvNextV2Block kernel_size=7 on 8x8 patch grid (CIFAR patch_size=4) = near-global receptive field. GRN norm computed over 64 spatial positions. Low-diversity augmentation (uniform crops) collapses GRN.

3. Sampling layer uses keras.random.normal without SeedGenerator, sharing TF global random state with tf.data augmentation ops. Reparameterization noise and augmentation not independent.

4. log_var_head zero-init in encoder.py: at init sigma=1 everywhere; early augmentation variance inflates initial KL via mu^2; interacts with beta_kl_start=0.

5. BetaAnnealingCallback ramps KL weight but augmentation is constant -- no curriculum.

6. JPEG source artifacts (8x8 DCT blocks) alias into 4-8px patches for ADE20K/COCO.

7. Missing reconstruction-under-masking objective (model-level, not augmentation-level).

## Session 2 (analysis_2026-05-26_05ccde10) — training infrastructure focus

7 exogeneity candidates seeded. Key new findings:

1. (H_SCOPE_1, prior 0.70) train.common.callbacks create_callbacks() defaults monitor='val_accuracy' -- VAE never produces this metric. EarlyStopping and ModelCheckpoint silently fail or checkpoint epoch-1 weights permanently.

2. (H_SCOPE_2, prior 0.45) Sampling(seed=int) -- static integer passed to keras.random.normal (not SeedGenerator) may produce constant epsilon per-batch in TF graph mode. Reparameterization collapses to deterministic.

3. (H_SCOPE_3, prior 0.40) SIGRegLayer statistic scales with batch size N (intentional per docstring, matches PyTorch upstream). Variable patch-batch size creates unstable SIGReg/KL loss balance.

4. (H_SCOPE_4, prior 0.35) LearnableMultiplier gamma clamped [0,1]; L2 regularization pressure drives gamma toward 0, suppressing encoder feature magnitude silently.

5. (H_SCOPE_5, prior 0.30) SoftOrthonormalConstraintRegularizer injected via use_softorthonormal_regularizer=True is an unmonitored loss term; invisible in loss decomposition.

6. (H_SCOPE_6, prior 0.35) include_terminate_on_nan=False default -- NaN gradients during KL warmup not caught.

7. (H_SCOPE_7, prior 0.25) GRN eps=1e-6 may produce near-zero normalization divisor in low-activation early training.

**Why:** H_S_prime posterior updated 0.30 -> 0.75 (confirmed again). H_SCOPE_1 is near-certain active defect.

**How to apply:** For any VAE training script using train.common.callbacks, always verify monitor= is explicitly set to 'val_total_loss' or equivalent. For Sampling with seed, verify SeedGenerator is used rather than plain int.
