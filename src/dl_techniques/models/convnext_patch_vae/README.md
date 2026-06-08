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

## vMF spherical latent mode (`sampling_type`)

In addition to the default Gaussian reparameterization, the model supports a
**per-patch von Mises–Fisher (vMF)** latent via `ConvNeXtPatchVAEConfig(sampling_type="vmf")`
(trainer flag `--sampler vmf`). Each patch latent is L2-normalized onto the unit
sphere `S^{latent_dim-1}`; the encoder emits a per-patch concentration `kappa`
(`Conv2D(1)` + softplus, `bias≈12` to prevent κ-collapse) instead of `log_var`,
the KL is the closed-form vMF→uniform-sphere divergence per patch, and
`compile()`/`compile_from_config()` force `jit_compile=False` (the Wood/Ulrich
sampler uses `keras.random.beta`, which has no XLA-GPU kernel in TF 2.18). The
prior is **Uniform(S^{latent_dim-1})** — `model.sample()` draws `N(0,I)` then
L2-normalizes per patch. SIGReg targets `N(0,I)`, which conflicts with the sphere,
so `--sampler vmf` auto-sets `lambda_sigreg=0`.

### Generative-prior diagnosis & fix (vMF mode)

A reverse-engineering analysis (`analyses/analysis_2026-06-06_0c7feade/`) of why
vMF **reconstructions are sharp but unconditional prior samples were incoherent**
found **two independent causes**:

1. **Eval/sampling bug (dominant, dead simple).** The sample-grid callback decoded
   raw `N(0,I)` latents (norm ≈ `sqrt(latent_dim)` ≈ 5.66) — *off* the unit sphere
   the decoder was trained on → pure noise. **Fix (F0):** L2-normalize prior draws
   per patch before decode. Verified: decoded total-variation 0.198 → 0.054 (~3.7×
   smoother). `model.sample()` already did this correctly; only the viz callback
   bypassed it. (Fixed in `train_convnext_patch_vae.py::_get_fixed_samples`.)
2. **Joint aggregate-posterior mismatch (the real, residual issue).** Even with
   correct sphere sampling, samples are coherent but **not realistic**, because the
   factorized prior `p(z)=∏ Uniform(S^{d-1})` samples *off* the thin joint manifold
   where the encoder placed real images. Decisive evidence: a **patch-shuffle**
   test (each patch replaced by a *different* image's posterior latent at the same
   position — exact per-patch marginal preserved, joint structure destroyed)
   collapses realism to uniform-noise level, so the **joint** inter-patch structure
   is the necessary cause (not per-patch marginals or decoder OOD — both refuted).
   A dose-response sweep showed the needed joint context is **local (~3–4 patches)**,
   not global. This is the textbook *aggregate-posterior hole* (Dai & Wipf 2019).

**Fixes for (2)** — the prior must model the *joint* latent (per-patch fixes are
proven insufficient). Experiments in `src/experiments/`:

- `convnext_patchvae_latent_prior.py` — 2-stage conv VAE over frozen latents.
  **Negative result:** mode-collapses to color blobs (the latents are weakly-
  correlated but high-entropy; a compressive bottleneck discards the detail).
- `convnext_patchvae_latent_diffusion.py` — small conv-UNet **DDPM** over the
  `(Hp,Wp,latent_dim)` latent grid (non-compressive; the right tool). **Works:**
  recovers ~61% (patch-16) → ~75% (patch-8/BCE) of the re-encoded structure gap,
  adjacent-patch cosine ≈ real; samples become globally-coherent, atmospheric.

- `convnext_patchvae_vaegan.py` — **end-to-end** VAE-GAN: trains encoder + decoder
  + discriminator *jointly*, with the adversarial loss applied to decoded *prior
  samples* (so the decoder is optimized in image space to make `decode(prior)`
  realistic). **Negative result:** it degrades reconstruction (adversarial pressure
  fights L1 recon) and produces **checkerboard/tiled** prior samples. The principled
  reason: the prior remains **factorized** (independent per-patch spheres), and no
  adversarial pressure on `decode(factorized_prior)` can synthesize the *global*
  inter-patch coherence the latent does not carry — it just makes each patch
  individually texture-like. **Adversarial-on-a-factorized-prior cannot fix H10;
  only a joint prior can** (which is why the two-stage diffusion prior, modeling the
  joint, is the better result). A correct end-to-end fix would jointly train a
  *learned joint prior* with the VAE (KL against `p_θ(z)`), a larger redesign.

**Honest status:** the noise is fixed (F0); generation is substantially improved
by the **two-stage diffusion prior** (the best result) but **not photorealistic** —
the remaining ceiling is the base autoencoder's own fidelity (MSE/BCE blur + coarse
patches + low-dim sphere), *not* the prior. A patch-8 + BCE retrain gave only a
modest base-fidelity gain; an end-to-end VAE-GAN made generation *worse*. Further
realism needs a stronger base decoder (resize-conv to kill checkerboard, perceptual
loss, larger latent) and/or a jointly-trained joint prior — a redesign, not a tweak.
Work on branch `fix/convnext-patchvae-vmf-prior`; artifacts in
`results/latent_prior_fix/` and `results/vaegan_e2e/`.

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

## Hierarchical (2-Level) Patch-Ladder-VAE

> Plan: `plans/plan_2026-06-08_e3917bd5/` (iter-1). Model + tests + docs only —
> no training script, no training run.

### What it is

A **2-Level Resolution-Agnostic Patch-Ladder-VAE** (`HierarchicalConvNeXtPatchVAE`),
a sibling of the flat model above (not a subclass; serialization-isolated, D-004):

- **Fine latent `z1 (B, Hp, Wp, D1)`** — the existing per-patch `ConvNeXtPatchEncoder`,
  Gaussian.
- **Coarse latent `z2 (B, Hp/2, Wp/2, D2)`** — **pool-derived**: `AvgPool2D(2)` over the
  fine encoder's last hidden features (pre-mu-head) → `Conv2D(D2,1)` mu/lv heads, Gaussian.
- **Learned top-down conditional prior `p(z1|z2)`**
  (`_L2ConditionalPrior`): `z2 → UpSampling2D(2, nearest) → Conv2D(embed,1) →
  M × ConvNextV2Block(kernel_size=3) → zero-init Conv2D(D1,1) mu_p + zero-init
  Conv2D(D1,1) lv_p`. **VDVAE delta-parameterization**: `mu1 = mu_p + delta`, where
  `delta` is the fine encoder's `mu_head` output (reused, D-002).
- **Free-bits collapse gate** on the **coarse** KL:
  `kl_l2 = mean(max(KL_per_patch(z2 ‖ N(0,I)), free_bits))`, default `0.25` nats/patch (D-006).

Both latents are **Gaussian** (D-003) → the conditional fine KL is the closed-form
Gaussian-Gaussian conditional KL (`_compute_kl_l2_conditional`). No vMF, no
`keras.random.beta`, so XLA stays enabled (no `jit_compile=False` override).

### Why

The flat factorized per-patch prior cannot model **inter-patch JOINT structure**
(global coherence) — the *aggregate-posterior hole* (Dai & Wipf 2019) diagnosed in
the vMF section above. The coarse latent `z2` + the learned top-down prior `p(z1|z2)`
add a global-context channel the fine prior lacks. Every cross-scale op is
**pool-derived + conv-only** (`AvgPool2D` / `UpSampling2D(nearest)` / `Conv2D(k=1)` /
`ConvNextV2Block(k=3)`) — no `Dense` / `GlobalAveragePooling` / fixed reshape — so the
hierarchy is **resolution-agnostic by construction**.

### Output-dict contract

`call()` returns legacy aliases that point at the FINE level (callback compatibility)
**plus** explicit two-level keys:

| Key | Points at | Shape |
|-----|-----------|-------|
| `reconstruction`, `z`, `mu`, `log_var` | FINE level (aliases) | fine grid |
| `z1`, `mu1`, `log_var1` | fine level (explicit) | `(B, Hp, Wp, D1)` |
| `z2`, `mu2`, `log_var2` | coarse level (explicit) | `(B, Hp/2, Wp/2, D2)` |

### Key config fields (`HierarchicalConvNeXtPatchVAEConfig`)

| Field | Meaning |
|-------|---------|
| `coarse_latent_dim` | D2, coarse latent channels |
| `prior_depth` | M, number of `ConvNextV2Block`s in the conditional prior |
| `prior_embed_dim` | prior hidden width; `0`-sentinel → falls back to `embed_dim` |
| `pool_factor` | fine→coarse pooling stride |
| `free_bits` | coarse-KL floor, nats/patch (default `0.25`) |
| `beta_kl_l1` / `beta_kl_l2` | KL weights for the fine / coarse levels |
| `lambda_sigreg_l1` / `lambda_sigreg_l2` | SIGReg weights at the fine / coarse levels (N-scaled) |

**Invariant** (`__post_init__`): `img_size % patch_size == 0` AND
`(img_size // patch_size) % pool_factor == 0` (even fine grid, so `AvgPool2D` and
`UpSampling2D` compose exactly).

### Usage

```python
import keras
from dl_techniques.models.convnext_patch_vae.model_hierarchical import (
    create_hierarchical_convnext_patch_vae,
)

# Named variants: "tiny" / "base" / "large".
m = create_hierarchical_convnext_patch_vae("base")
m.compile(optimizer=keras.optimizers.AdamW(3e-4), loss=None)  # all losses via add_loss
m.fit(x_train, epochs=50, batch_size=256)

mu1, log_var1 = m.encode(x_any_resolution)   # fine posterior
samples       = m.sample(num_samples=16, hp=8, wp=8)   # coherent two-level prior path
```

### TWO MANDATORY CAVEATS

1. **Efficacy is UNPROVEN.** This is a *correct, trainable, serializable* implementation
   — **NOT** a demonstrated generation improvement. The expected payoff (an analytic
   43–66% of the joint-structure ceiling-gap) is a *scenario bound, not a measurement*:
   no hierarchical model has been trained + measured here (analysis `H3=0.65`,
   mechanism-only; pending prediction `PP2` is open). Source:
   `analyses/analysis_2026-06-08_3ec50266/summary.md`.
2. **The hierarchy fixes GLOBAL COHERENCE only (Axis A), NOT within-patch blur (Axis B).**
   Reconstruction sharpness / patch coarseness is **orthogonal** — no joint prior touches
   it. Realism needs a separate track (smaller patches + a perceptual/LPIPS decoder).
   **Do NOT expect the hierarchy to fix blur.**

### Future work

- **vMF posterior at the fine level** is documented-but-NOT-implemented: it conflicts
  with the closed-form Gaussian conditional prior (would require a vMF-vs-vMF conditional
  KL). Deferred (D-003).
- **Multi-resolution training** (`RandomResizedCrop` scale `[0.5, 2.0]`) is required for
  off-training-resolution deployment, but that is a **TRAINER concern** — not part of this
  model.

---
