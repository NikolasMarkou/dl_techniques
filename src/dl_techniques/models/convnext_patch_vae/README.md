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

**Honest status:** the noise is fixed (F0); generation is substantially improved
by the learned diffusion prior but **not photorealistic** — the remaining ceiling
is the base autoencoder's own fidelity (MSE/BCE blur + coarse patches + low-dim
sphere), *not* the prior. A patch-8 + BCE retrain gave only a modest base-fidelity
gain; further realism needs a stronger decoder (perceptual/adversarial loss, larger
latent), not prior tweaks. Work on branch `fix/convnext-patchvae-vmf-prior`;
artifacts in `results/latent_prior_fix/`.

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
