# ConvNeXt Patch-Level VAE

A **resolution-agnostic** ConvNeXt-based variational autoencoder operating on
**per-patch latents** `z(B, Hp, Wp, latent_dim)` with **SIGReg-driven
anti-patch-collapse**.

> Plan: `plans/plan_2026-05-25_fb57d478/` (iter-1).
> No training in this iteration — the package ships code + a prioritized
> training menu. Training is the explicit next iteration, gated on user review.

---

## Architecture

```
x : (B, H, W, C)
    │
    ▼  ConvNeXtPatchEncoder
    │      Conv2D(embed_dim, kernel=stride=patch_size, "valid")   # stem
    │      LayerNormalization
    │      depth × [residual + ConvNextV2Block(kernel_size, embed_dim)]
    │      Conv2D(2 * latent_dim, kernel=1)                       # bottleneck
    │      split last dim → (mu, log_var), each (B, Hp, Wp, latent_dim)
    │
    ▼  Sampling([mu, log_var])                                    # reparam
    │
    z : (B, Hp, Wp, latent_dim)
    │
    ▼  ConvNeXtPatchDecoder
    │      Conv2D(embed_dim, kernel=1)                            # proj_in
    │      depth × [residual + ConvNextV2Block(kernel_size, embed_dim)]
    │      LayerNormalization
    │      Conv2DTranspose(img_channels, kernel=stride=patch_size, "valid")
    │
    ▼
x_hat : (B, Hp * patch_size, Wp * patch_size, C)
```

**Loss** (added via `add_loss` inside `call`, summed in `train_step`):

```
total = recon
      + beta_kl     * mean_{B,Hp,Wp} [ -0.5 * sum_d (1 + log_var - mu^2 - exp(log_var)) ]
      + lambda_sigreg * SIGReg( reshape(z, (B, Hp*Wp, latent_dim)) )
```

`recon` is pixel-space MSE (`recon_loss_type="mse"`, RGB default) or BCE-with-logits
(`recon_loss_type="bce"`, binary inputs).

---

## Six Design Choices (anchored to findings)

| # | Choice | Anchor | Why |
|---|--------|--------|-----|
| 1 | Flat single-stage block stack (no spatial downsampling beyond stem) | F3 | `ConvNextV2Block` is shape-preserving at stride=1; stacking N blocks yields depth without breaking resolution-agnosticity. Hierarchical encoders defer to T4 ablation. |
| 2 | Per-patch 4D latent `(B, Hp, Wp, latent_dim)` | F4, F5 | The literal target of "patch-collapse" prevention. KL per-patch then averaged over `(B, Hp, Wp)` keeps loss magnitude independent of resolution. |
| 3 | SIGReg on `(B, Hp*Wp, latent_dim)` post-reparam (D-002) | F2 | Regularizes the same quantity KL targets; N = patches per image scales with resolution; single SIGReg call per forward. Per-position (N=batch) binding is rejected — would discourage spatial information. |
| 4 | Explicit `self.loss_tracker = Mean(name="loss")` + `update_state` in `train_step` (D-001) | video_jepa D-005 | Keras 3.8 does NOT auto-create `loss_tracker` when `compile(loss=...)` is bypassed. Without this, `history.history['loss']` pins to 0.0 — defeats `EarlyStopping` / `ModelCheckpoint(monitor='loss')`. |
| 5 | No EMA target encoder | F5 (GHOST) | EMA is required for JEPA / contrastive losses where identity is the trivial solution. VAE recon forbids identity → no EMA. Saves ~80 LOC + serialization complexity. |
| 6 | No positional embedding | F5 (GHOST) | ConvNeXt is translation-equivariant via large depthwise kernels. Adding learned absolute PE would tie the encoder to a fixed grid and break the resolution-agnostic invariant. |

---

## Why no EMA / Why no PE (Ghost-Constraint Section)

When you "borrow patterns from video_jepa", the temptation is to copy
its EMA target encoder and its learned positional embedding. Both are
**ghost constraints** from video_jepa's contrastive setting:

- **EMA target encoder** exists in JEPA because both branches see the
  same input and the identity map is the trivial solution. A VAE's
  reconstruction objective forbids identity — `decoder(encoder(x)) = x`
  IS what we want. There is no symmetry to break. Adding EMA would add
  serialization complexity, ~80 LOC, and an extra forward pass per step
  for zero benefit.
- **Learned absolute positional embedding** ties the encoder to a fixed
  grid size at training time. ConvNeXt blocks gain spatial context via
  large depthwise kernels — they are translation-equivariant by
  construction. Dropping PE is what makes `model.encode(x)` work for
  ANY H, W satisfying `H % patch_size == 0`.

If iter-1 falsifies the hypothesis (see Training Menu T1 below), the
correct response is **NOT** to add EMA or PE. It is to revisit the
SIGReg binding (D-002), then the recon-loss choice, then beta/lambda
hyperparameters.

---

## Quick start (factory)

```python
import keras
from dl_techniques.models.convnext_patch_vae.model import create_convnext_patch_vae

# Named variants: "tiny" / "base" / "large". Any keyword overrides preset values.
model = create_convnext_patch_vae("base")
model.compile(optimizer=keras.optimizers.AdamW(3e-4))
model.fit(x_train, epochs=50, batch_size=256)

# Resolution-agnostic API (any H % patch_size == 0 works):
mu, log_var = model.encode(x_any_resolution)
x_hat       = model.decode(mu)                            # raw or sigmoid'd
samples     = model.sample(num_samples=16, hp=8, wp=8)    # (16, 32, 32, 3)
```

`create_convnext_patch_vae("base", pretrained=True)` raises
`NotImplementedError` — no public checkpoints are published for this VAE
(repo-wide convention, see `D-001`).

| Variant | embed_dim | encoder_depth | decoder_depth | latent_dim |
|---------|-----------|---------------|---------------|------------|
| tiny    | 64        | 2             | 2             | 8          |
| base    | 128       | 4             | 4             | 16         |
| large   | 192       | 6             | 6             | 32         |

## Low-level API

```python
import keras
from dl_techniques.models.convnext_patch_vae.config import ConvNeXtPatchVAEConfig
from dl_techniques.models.convnext_patch_vae.model import ConvNeXtPatchVAE

cfg = ConvNeXtPatchVAEConfig(
    img_size=32, img_channels=3, patch_size=4,
    embed_dim=128, encoder_depth=4, decoder_depth=4, kernel_size=7,
    latent_dim=16, beta_kl=0.5, lambda_sigreg=0.1,
    sigreg_num_proj=256, recon_loss_type="mse",
)
model = ConvNeXtPatchVAE(cfg)
model.compile(optimizer=keras.optimizers.AdamW(3e-4))
# history.history['loss'] honors the D-001 contract (nonzero).
model.fit(x_train, epochs=50, batch_size=256)
```

Full `.keras` save / `keras.models.load_model` round-trip is supported
(see `tests/test_models/test_convnext_patch_vae/`).

---

## Training Menu (no execution in this iteration)

The model is shipped without a training script. The menu below is the
prescription for follow-up plans, ranked by what falsifies the core
hypothesis **cheapest first**. Each entry will be its own plan.

### Tier 1 — falsify the hypothesis (RECOMMENDED FIRST)

| # | Name | Dataset | Resolution | Patch | latent_dim | Hp×Wp | Hp·Wp (SIGReg N) | RTX-4090 runtime | Success criteria | What it proves |
|---|------|---------|-----------|-------|-----------|-------|------------------|------------------|------------------|----------------|
| **T1a** | CIFAR-10 SIGReg=ON | CIFAR-10 (built-in) | 32×32 | 4 | 16 | 8×8 | 64 | ~30 min, 50 epochs, batch 256, AdamW lr=3e-4 | (a) val recon MSE < 0.015; (b) per-patch latent std > 0.5 on ≥ 80% of val images; (c) SIGReg loss monotone-decreasing first 5 epochs then plateaus | Full hypothesis: SIGReg-on prevents patch-collapse without hurting recon. |
| **T1b** | CIFAR-10 SIGReg=OFF (control) | CIFAR-10 | 32×32 | 4 | 16 | 8×8 | 64 | ~30 min, identical schedule | Same recon target; **observe** per-patch latent std (expected lower) | If T1a recon ≈ T1b recon AND T1a patch-std > T1b patch-std → hypothesis stands. If T1a recon ≫ T1b recon → hypothesis falsified. |

**T1a + T1b together is the smallest-cost falsification (~1 hour on RTX
4090).** This is the only experiment that should run before user
reviews results. Everything below is contingent.

### Tier 2 — establish recon quality (CONDITIONAL on T1 success)

| # | Name | Dataset | Resolution | Patch | latent_dim | Hp×Wp | Runtime | Success | What it proves |
|---|------|---------|-----------|-------|-----------|-------|---------|---------|----------------|
| T2 | Higher-res recon | TFDS `imagenette` | 128×128 (center crop) | 8 | 32 | 16×16 | ~2-3h, 30 ep, batch 64 | val PSNR > 22 dB; prior samples visually plausible | Model scales beyond toy resolution; per-patch grid carries useful info. |
| T3 | Resolution-agnostic at inference | T2 checkpoint | 64×64 AND 256×256 inference | 8 | 32 | 8×8 + 32×32 | ~5 min | Forward succeeds at both; visual quality comparable | The resolution-agnostic property is realized in practice. |

### Tier 3 — ablations (CONDITIONAL on T1 + T2 success)

| # | Name | What varies | Runtime | What it proves |
|---|------|-------------|---------|----------------|
| T4a | beta_kl sweep | {0.1, 0.25, 0.5, 1.0, 2.0} on T1 config | ~2.5h | Posterior-collapse boundary; defend default β=0.5. |
| T4b | lambda_sigreg sweep | {0.0, 0.05, 0.1, 0.2, 0.5} on T1 config | ~2.5h | Where SIGReg dominates vs disappears. |
| T4c | V1 vs V2 block | `use_v2_block=False` (no GRN) on T1a | ~30 min | GRN's contribution to anti-collapse vs SIGReg's. Most informative if T1a passes. |
| T4d | Patch-size sweep | `patch_size ∈ {2, 4, 8}` on 32×32 | ~1.5h | Operational floor where Hp·Wp = N is too small for SIGReg (~ patch=8, N=16). |

### NOT recommended

- Full ImageNet pretraining — premature; hypothesis testable at CIFAR scale.
- MNIST — too easy; trivial recon even with collapsed latents.
- 256+ resolution at iter-1 — runtime ~10× for a question T2 answers.
- Hierarchical multi-stage encoder — blast radius before flat-stage results.

---

## Tests

`tests/test_models/test_convnext_patch_vae/test_convnext_patch_vae.py`
(8 tests, 15 cases including parametrized invariant rejections):

1. Config round-trip + invariant violations.
2. Resolution-agnostic forward (32×32 + 48×48 on the same instance).
3. Full `.keras` save/load round-trip (`atol=1e-4` on `mu`).
4. Per-patch KL resolution-invariance (ratio < 2×).
5. SIGReg integration produces a finite scalar.
6. `model.fit(epochs=1, steps=2)` reports nonzero `history.history['loss']`
   (D-001 contract).
7. `model.sample(hp=4, wp=4)` and `(hp=8, wp=8)` both succeed.
8. `lambda_sigreg=0.0` → weighted contribution is exactly 0, but the
   `sigreg_loss` tracker still reports the raw SIGReg statistic for
   ablation comparison.

```bash
.venv/bin/python -m pytest tests/test_models/test_convnext_patch_vae/ -vvv
# 15 passed in ~24s on RTX 4090 (well under the 90s C11 budget).
```

---

## Hierarchical variant — `HierarchicalConvNeXtPatchVAE`

> Plan: `plans/plan_2026-05-27_dee954c6/` (iter-1).
> Motivation: `analyses/analysis_2026-05-26_05ccde10/summary.md` §6.

The single-scale model above bottlenecks at high resolution because each
8×8 patch must compress 192 pixel values into a 16-D latent with no
cross-patch context. At 256×256 ADE20K/COCO this produces an effective
12:1 compression ratio per patch and predicts posterior collapse (E1 in
the analysis).

The hierarchical variant introduces a coarse **L1** scale alongside the
fine **L2** scale:

```
x : (B, H, W, C)
    │                                  │
    ▼  ConvNeXtPatchEncoder            ▼  ConvNeXtPatchEncoder
       (patch_size_l1, big)               (patch_size_l2, small)
    │                                  │
   (mu_l1, log_var_l1)                (mu_l2, log_var_l2)
    │                                  │
    ▼  Sampling                        ▼  Sampling
    z_l1 (B, Hp1, Wp1, latent_l1)      z_l2 (B, Hp2, Wp2, latent_l2)
    │                                  │
    │      ┌───────────────────────────┘
    │      │
    │      ▼
    ▼  _L2ConditionedDecoder:
       proj_in(z_l2) ─┐
                      concat ─ cond_proj(1×1) ─ N×ConvNeXtV2Block ─ LN ─ Conv2DTranspose ─▶ x_hat
       UpSample(z_l1)─┘   (nearest-neighbor, factor = patch_size_l1 / patch_size_l2)
```

**L1 has no pixel-recon head** — it is conditioning only.

**Loss**:

```
recon
+ beta_kl_l1   * KL(z_l1)
+ beta_kl_l2   * KL(z_l2)
+ lambda_l1    * (sigreg(z_l1) * N_L1)
+ lambda_l2    * (sigreg(z_l2) * N_L2)
```

**Annealing**: trainer installs two `BetaAnnealingCallback` instances with
staggered schedules — `_beta_kl_l1` ramps over `beta_anneal_epochs_l1`,
`_beta_kl_l2` over `beta_anneal_epochs_l2` (intentional overlap).
Callback gains an `attr_name` kwarg so the same class drives both.

**Parameter budget** (base preset @ 256×256):

| Component | Params |
|-----------|--------|
| L1 encoder + decoder | ~1.14M |
| L2 encoder + L2-conditioned decoder | ~1.02M |
| **Total** | **~2.16M** |

15× below LDM/SD VAE (~34M), fits in 12 GB at batch 8.

**Smoke-validated**: CIFAR-10 32×32 (2 epochs, 20 steps, 13K params) and
ADE20K 256×256 (1 epoch, 10 steps, batch 8, 2.16M params) both run
cleanly with bit-exact `.keras` round-trip on both `mu_l1` and `mu_l2`.

### CLI

```bash
# 256x256 ADE20K, batch 8, single epoch smoke
CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python -m \
  train.convnext_patch_vae.train_convnext_patch_vae \
  --hierarchical --dataset ade20k --image-size 256 \
  --patch-size 8 --patch-size-l1 32 \
  --latent-dim 16 --latent-dim-l1 64 \
  --embed-dim 128 --embed-dim-l1 128 \
  --encoder-depth 4 --encoder-depth-l1 4 \
  --decoder-depth 4 --decoder-depth-l1 4 \
  --batch-size 8 --epochs 50 \
  --beta-anneal-epochs-l1 10 --beta-anneal-epochs-l2 15 \
  --lambda-sigreg-l1 0.05 --lambda-sigreg-l2 0.1 \
  --gpu 1
```

`--patch-size` / `--latent-dim` map to the **L2** scale. `--patch-size-l1`
/ `--latent-dim-l1` / `--embed-dim-l1` etc. control L1. `patch_size_l1`
must be an integer multiple of `patch_size_l2`.

### Conditional prior `p(z_l2 | z_l1)`

> Plan: `plans/plan_2026-05-27_c3184aea/`.

The hierarchical model ships with a learnable conditional prior on the
fine latent. Instead of `p(z_l2) = N(0, I)`, the model learns
`p(z_l2 | z_l1)` via a small ConvNeXtV2-style network
(`_L2ConditionalPrior`) that consumes the upsampled L1 latent and
predicts `(mu_p, log_var_p)` per L2 patch position.

```
z_l1 (B, Hp1, Wp1, latent_l1)
 ▼ UpSampling2D(tile_factor, "nearest")
 ▼ Conv2D(embed_dim_l2, 1)
 ▼ N × ConvNeXtV2Block       (prior_l2_depth, default 2)
 ▼ LayerNorm
 ├─ Conv2D(latent_l2, 1, zeros-init) → mu_p
 └─ Conv2D(latent_l2, 1, zeros-init) → lv_p
```

**Zero-init heads** are the load-bearing detail: at step 0 the prior
emits exactly `N(0, I)` regardless of `z_l1`. This means
- old checkpoints trained with the legacy implicit prior transfer cleanly
  (`weight_transfer.load_weights_from_checkpoint` skips the new layer,
  the prior network starts at random init but its heads compute zero),
- from-scratch training bootstraps from the same operating point as the
  old model — the prior's contribution to KL is exactly the legacy KL at
  step 0 and gradually deviates as the prior network learns.

The KL term becomes the closed-form diagonal Gaussian KL with both
`log_var_q` and `log_var_p` clipped to `[-10, +10]`:

```
KL(q || p) = 0.5 · Σ_d [ lv_p − lv_q + (exp(lv_q) + (mu_q − mu_p)²) · exp(−lv_p) − 1 ]
```

**Toggle**: `learnable_l2_prior: bool = True` is the default. Set
`False` to revert to `KL(q(z_l2|x) || N(0,I))` for ablation; the prior
network is then `None`.

**Hyperparameters**: `prior_l2_depth` (default 2 blocks) and
`prior_l2_embed_dim` (sentinel `0` → `embed_dim_l2`).

**Param overhead** (at the default CIFAR config): ~5K params. At the
256×256 base config (embed_dim_l2=128): ~50K params, well under 3% of
the existing 2.16M model.

### Generating images: `sample_from`

Both `ConvNeXtPatchVAE` and `HierarchicalConvNeXtPatchVAE` expose the
same one-line sampling API:

```python
# Reconstruction (deterministic):
img = model.sample_from(x_anchor, temperature=0.0)

# Variations around x_anchor (VAE prior scale):
img = model.sample_from(x_anchor, temperature=1.0, seed=42)

# More diverse variations:
img = model.sample_from(x_anchor, temperature=1.5, seed=42)
```

The method reparameterizes from the encoder's posterior at the requested
temperature: `z = mu + temperature * exp(0.5 * log_var) * eps`.

**Pure-prior sampling** via `model.sample(num_samples, ...)` is now the
coherent generative path, thanks to the learnable conditional prior:

```
z_l1 ~ N(0, I)
mu_p, lv_p = prior(z_l1)
z_l2 ~ N(mu_p, exp(lv_p))
return decode(z_l1, z_l2)
```

`(z_l1, z_l2)` are drawn from the joint generative distribution — no
posterior-mismatch artifacts.

Use `sample_from(x, temperature)` when you want **variations around a
specific real image**; use `sample(num_samples)` when you want **fresh
generations** from the prior.

**Legacy behavior**: when `learnable_l2_prior=False`, `sample()` falls
back to independent `N(0, I)` for both latents — this is the
incoherent path retained for ablation only.

For the single-scale model, pure-prior `sample()` is fine (no inter-latent
coupling); `sample_from` is provided there for API parity.

### Out of scope (this iteration)

- Mid-training viz callbacks (Recon / LatentSpace / LatentInterpolation)
  in hierarchical mode — they assume single-scale `encode/decode`
  signatures and are skipped with an info log. Hierarchical viz is a
  follow-up.
- Spatial cross-attention conditioning (Option B in the analysis).
- L1 stand-alone pixel-space reconstruction head.
- L2 encoder conditioning (only the L2 decoder is conditioned).
