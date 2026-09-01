# Variational Autoencoder (VAE)

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

An implementation of a **Variational Autoencoder (VAE)** in **Keras 3**, based on the foundational paper ["Auto-Encoding Variational Bayes"](https://arxiv.org/abs/1312.6114) by Kingma & Welling (2013).

The encoder and decoder are residual convolutional stacks, and a custom `train_step`
computes and tracks the two-part VAE objective (reconstruction and KL). Three latent
geometries are selectable through one argument, `sampling_type`: the standard diagonal
Gaussian, a thin-shell **hypersphere**, and a true **von Mises-Fisher** spherical VAE.

---

## 1. Overview: What is VAE and Why It Matters

### What is a VAE?

A **Variational Autoencoder** learns to compress data into a structured, low-dimensional
**latent space** and to generate new data by sampling from that space.

Unlike a plain autoencoder, the encoder does not map an input to a single point. It emits
the parameters of a **probability distribution** over latents. A latent vector is sampled
from that distribution and handed to the decoder.

### Key Innovations

1. **Probabilistic latent space.** The space is continuous and organised, so nearby points
   decode to similar outputs and interpolation is meaningful.
2. **The reparameterization trick.** Sampling is rewritten so gradients flow through it,
   making the whole model trainable end to end with a standard optimizer.
3. **A principled objective.** The loss is the negative Evidence Lower Bound (ELBO) on the
   data log-likelihood, not a heuristic.
4. **Selectable latent geometry.** `sampling_type` swaps the sampler, the prior used by
   `sample()`, and the KL term together, so a Gaussian ball, a spherical shell, or a vMF
   sphere are all one argument apart.

---

## 2. The Problem VAE Solves

A plain autoencoder minimises reconstruction error alone. Nothing in that objective asks it
to organise the latent space, so the space ends up disjointed and sparse: encoded points sit
in isolated islands with large dead zones between them. Draw a random latent and the decoder
almost always produces noise, because the point you drew lies in a region no training
example ever occupied.

The VAE adds a second term that fixes exactly this:

```
Loss = Reconstruction Loss + beta * KL Divergence
       |                            +-- pulls every encoded distribution toward the
       |                                prior, filling the dead zones
       +-- keeps the decoder faithful to the input
```

The KL term is what makes the space samplable: it forces the per-input posteriors to overlap
and to cover the prior, so a draw from the prior lands somewhere the decoder understands.

---

## 3. How VAE Works: Core Concepts

### The Objective: Evidence Lower Bound

The loss is derived by lower-bounding the intractable log-likelihood `log p(x)`:

```
log p(x)  >=  E_q(z|x)[ log p(x|z) ]  -  KL( q(z|x) || p(z) )
              \_______________________/    \___________________/
                   reconstruction term          regularizer
```

Minimising the negative of that bound gives the training loss:

- **`-E_q(z|x)[log p(x|z)]`** — the expected negative log-likelihood of `x` given `z`. For
  pixels scaled to `[0, 1]` this is **binary cross-entropy**, which is what
  `_compute_reconstruction_loss` computes (in float32, with the inputs clipped away from
  `0` and `1` so `log` never sees zero).
- **`KL(q(z|x) || p(z))`** — how far the encoder's posterior sits from the prior. Scaled by
  `kl_loss_weight`, the `beta` of beta-VAE.

### The Reparameterization Trick

Sampling `z ~ N(mu, sigma^2)` is random and therefore not differentiable. The trick moves
the randomness outside the gradient path:

**`z = mu + sigma * eps`**, where **`eps ~ N(0, 1)`**

`mu` and `sigma` are deterministic encoder outputs; `eps` is fixed noise. The map from
`mu, sigma` to `z` is now differentiable and gradients reach the encoder.

### The Three KL Terms

Each latent geometry has its own closed-form KL. All three run in float32 regardless of the
compute policy, because every branch exponentiates a clipped log-variance and `exp(20)`
overflows float16.

| `sampling_type` | Posterior | `z_log_var` slot holds | KL against |
| :--- | :--- | :--- | :--- |
| `gaussian` | `N(mu, sigma^2)`, diagonal | log-variance, shape `[B, D]` | `-0.5 * sum(1 + logv - mu^2 - exp(logv))` vs `N(0, I)` |
| `hypersphere` | direction from `z_mean`, radius on a thin shell | radius log-variance, shape `[B, 1]` | `0.5 * (exp(rlv) - rlv - 1)`, a 1-D radius KL |
| `vmf` | `vMF(mu_hat, kappa)` on the unit sphere | concentration `kappa > 0`, shape `[B, 1]` | closed-form vMF-to-uniform-sphere KL |

The `z_log_var` output slot is reused by all three modes — its shape and meaning change with
the mode, so never interpret it without checking `sampling_type` first.

### The Complete Data Flow

```
STEP 1: ENCODING          Input (B, H, W, C)
    |-> residual encoder, one downsampling stage per depth
    |-> global average pooling -> (B, F)
    |-> Dense -> z_mean    (B, D_latent)
    +-> Dense -> z_log_var (B, D_latent), or (B, 1) for hypersphere / vmf

STEP 2: SAMPLING          layer "vae_sampling", one per mode
    |-> gaussian:    z = z_mean + exp(0.5 * z_log_var) * eps,  eps ~ N(0, 1)
    |-> hypersphere: unit direction from z_mean, radius on a thin shell
    +-> vmf:         Ulrich/Wood rejection sampler + Householder reflection

STEP 3: DECODING          Dense + reshape -> residual upsampling stack
    +-> final conv + sigmoid -> reconstruction (B, H, W, C)

STEP 4: LOSS              total = reconstruction_loss + kl_weight * kl_loss
```

---

## 4. Architecture Deep Dive

### 4.1 Residual Encoder

`depths` downsampling stages, `steps_per_depth` residual blocks per stage, channel counts
from `filters` (one entry per depth — `len(filters)` must equal `depths`). The stack ends in
global average pooling followed by two `Dense` heads, `z_mean` and `z_log_var`. Blocks use
`leaky_relu`, `he_normal` initialization, and optional batch norm and dropout.

### 4.2 Sampling Layer

A non-trainable layer, always named `vae_sampling` whatever the mode — `decode()` and
`sample()` locate the decoder by that exact name, so the name is part of the contract.

- **`gaussian`** — `Sampling`, the reparameterization trick above.
- **`hypersphere`** — `HypersphereSampling`. The encoder predicts a unit direction plus one
  scalar radius log-variance, and the latent lands on a thin, strictly positive shell:
  `r = radius * (1 + 0.1 * exp(0.5 * clip(rlv)) * eta)`, floored at `0.05 * radius`. There
  is no explicit directional KL, so this is a deliberate simplification, **not** a full
  spherical VAE.
- **`vmf`** — `VMFSampling`, a fixed-K Ulrich/Wood rejection sampler with a Householder
  reflection, paired with the exact vMF-to-uniform KL (`vmf_kl_divergence`, computed from a
  continued-fraction modified-Bessel ratio). This is a true von Mises-Fisher spherical VAE
  in the sense of Davidson et al. (2018).

All three live in `dl_techniques/layers/sampling.py`.

The legacy value `"hypersphere_faithful"` is accepted as a deprecated alias for
`"hypersphere"` so old configs and checkpoints still load; the removed
`"hypersphere_controlled"` raises `ValueError`.

### 4.3 Residual Decoder

A mirror of the encoder: `Dense` projection and reshape to the smallest feature map, one
upsampling stage per depth with residual blocks, then a final convolution with
`final_activation` (default `sigmoid` — which is what makes the binary cross-entropy
reconstruction term correct for `[0, 1]` data).

---

## 5. Quick Start Guide

```python
import keras
import numpy as np
from keras.datasets import mnist

from dl_techniques.models.vision.vae.model import VAE

# 1. Load and preprocess data (pixels must land in [0, 1] for the BCE term)
(x_train, _), (x_test, _) = mnist.load_data()
x_train = np.expand_dims(x_train.astype("float32") / 255.0, -1)
x_test = np.expand_dims(x_test.astype("float32") / 255.0, -1)

# 2. Create a VAE for MNIST with a 2-D latent, so it can be plotted directly
model = VAE.from_variant("small", input_shape=(28, 28, 1), latent_dim=2)

# 3. Compile. The loss lives in train_step, so only an optimizer is needed.
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3))

# 4. Train
model.fit(x_train, epochs=10, batch_size=128, validation_data=(x_test,))

# 5. Generate new digits by decoding draws from the prior
generated = model.sample(num_samples=15)
print(generated.shape)  # (15, 28, 28, 1)
```

`fit()` takes the images alone: `train_step` reads `data[0]` when handed a tuple and the
tensor itself otherwise, so `validation_data=(x_test,)` is the right one-element form.

---

## 6. Component Reference

### 6.1 `VAE` (Model Class)

**Location**: `dl_techniques.models.vision.vae.model.VAE`

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `latent_dim` | (required) | Dimensionality of the latent space `z`. |
| `input_shape` | (required) | `(H, W, C)`; `H` and `W` must be at least 8. |
| `depths` | `2` | Number of downsampling / upsampling stages. |
| `steps_per_depth` | `1` | Residual blocks per stage. |
| `filters` | `None` | One channel count per depth. `None` gives `[32, 64, ...]`. |
| `kl_loss_weight` | `0.01` | The `beta` of beta-VAE. |
| `sampling_type` | `"gaussian"` | `"gaussian"`, `"hypersphere"` or `"vmf"`. |
| `kernel_initializer` | `"he_normal"` | Initializer for conv and dense kernels. |
| `kernel_regularizer` | `None` | Optional regularizer. |
| `use_batch_norm` | `True` | Batch normalization inside the residual blocks. |
| `use_bias` | `True` | Bias terms in the convolutions. |
| `dropout_rate` | `0.0` | Must be in `[0, 1)`. |
| `activation` | `"leaky_relu"` | Hidden activation. |
| `final_activation` | `"sigmoid"` | Decoder output activation. |

**Key methods**:

- `VAE.from_variant(variant, input_shape, latent_dim=None, **kwargs)` — build a named size.
  An explicit keyword argument always overrides the variant's value.
- `encode(images)` — returns the `(z_mean, z_log_var)` pair.
- `decode(z)` — decodes latent vectors to images.
- `sample(num_samples)` — draws from *this mode's* true prior (`N(0, I)` for gaussian,
  uniform-on-sphere at the layer radius otherwise) and decodes.
- `train_step` / `test_step` — compute and track `total_loss`, `reconstruction_loss` and
  `kl_loss`.

Calling the model returns a dict with keys `"reconstruction"`, `"z_mean"`, `"z_log_var"`
and `"z"`.

### 6.2 Factory Functions

| Function | Purpose |
| :--- | :--- |
| `create_vae(input_shape, latent_dim, variant="small", optimizer="adam", learning_rate=0.001, **kwargs)` | Build **and compile** a variant in one call. |
| `create_vae_from_config(config)` | Build and compile from a configuration dictionary. |

Both disable `jit_compile` automatically when `sampling_type="vmf"` (see §10).

### 6.3 Sampling Layers

| Layer | Location |
| :--- | :--- |
| `Sampling` | `dl_techniques.layers.sampling.Sampling` |
| `HypersphereSampling` | `dl_techniques.layers.sampling.HypersphereSampling` |
| `VMFSampling` | `dl_techniques.layers.sampling.VMFSampling` |
| `vmf_kl_divergence` | `dl_techniques.layers.sampling.vmf_kl_divergence` |

---

## 7. Configuration & Model Variants

`VAE.MODEL_VARIANTS` holds five entries. `latent_dim` falls back to the variant default when
not given explicitly; so does `kl_loss_weight`.

| Variant | Depths | Steps/Depth | Filters | Default `latent_dim` | Default `kl_loss_weight` |
| :---: | :---: | :---: | :--- | :---: | :---: |
| **`micro`** | 2 | 1 | `[16, 32]` | 32 | 0.01 |
| **`small`** | 2 | 1 | `[32, 64]` | 64 | 0.01 |
| **`medium`** | 3 | 1 | `[32, 64, 128]` | 128 | 0.005 |
| **`large`** | 3 | 2 | `[64, 128, 256]` | 256 | 0.005 |
| **`xlarge`** | 4 | 2 | `[64, 128, 256, 512]` | 512 | 0.001 |

Input spatial dimensions must be at least `8 x 8`, and should be divisible by `2^depths` so
the encoder and decoder round-trip cleanly.

---

## 8. Usage Examples

### Example 1: Reconstructing images

```python
import numpy as np
from dl_techniques.models.vision.vae.model import create_vae

model = create_vae(input_shape=(28, 28, 1), latent_dim=16, variant="small")
batch = np.random.rand(8, 28, 28, 1).astype("float32")

outputs = model.predict(batch, verbose=0)
print(outputs["reconstruction"].shape)  # (8, 28, 28, 1)
print(outputs["z_mean"].shape)          # (8, 16)
```

### Example 2: Walking a 2-D latent space

With `latent_dim=2` the whole manifold can be decoded on a grid and plotted directly.

```python
import numpy as np
from dl_techniques.models.vision.vae.model import VAE

model = VAE.from_variant("small", input_shape=(28, 28, 1), latent_dim=2)
n, digit = 20, 28
canvas = np.zeros((digit * n, digit * n), dtype="float32")
grid = np.linspace(-3.0, 3.0, n)
for i, yi in enumerate(grid):
    for j, xi in enumerate(grid):
        decoded = model.decode(np.array([[xi, yi]], dtype="float32"))
        canvas[i * digit:(i + 1) * digit, j * digit:(j + 1) * digit] = (
            np.asarray(decoded)[0, :, :, 0])
```

### Example 3: Swapping the latent geometry

```python
from dl_techniques.models.vision.vae.model import VAE

gauss = VAE.from_variant("small", input_shape=(28, 28, 1),
                         latent_dim=16, sampling_type="gaussian")

hyper = VAE.from_variant("small", input_shape=(28, 28, 1),
                         latent_dim=16, sampling_type="hypersphere")

# The vMF KL is roughly 100x larger than the others', so beta must shrink to match.
vmf = VAE.from_variant("small", input_shape=(28, 28, 1), latent_dim=16,
                       sampling_type="vmf", kl_loss_weight=1e-3)
```

---

## 9. Advanced Usage Patterns

### Pattern 1: Tuning beta

`kl_loss_weight` trades reconstruction fidelity against latent-space regularity.

- **beta < 1** — favours reconstruction; the latent space is less tightly organised.
- **beta = 1** — the plain ELBO.
- **beta > 1** — pushes toward a disentangled latent, usually at the cost of blur.

```python
beta_vae = VAE.from_variant("small", input_shape=(28, 28, 1),
                            latent_dim=10, kl_loss_weight=4.0)
```

### Pattern 2: KL warmup

`train_step` multiplies the KL by `self.kl_weight`, which a callback can ramp from `0` to
`kl_loss_weight` over the first epochs. Starting at `0` lets the model learn a reconstruction
path before the regularizer bites, which is the standard cure for posterior collapse. The
`vmf` mode in particular needs a warmup, because a freely learned `kappa` otherwise collapses
to `~0` (a uniform latent) and reconstruction stalls.

---

## 10. Performance Optimization

### Mixed precision

```python
import keras
from dl_techniques.models.vision.vae.model import VAE

keras.mixed_precision.set_global_policy("mixed_float16")
model = VAE.from_variant("large", input_shape=(32, 32, 3), latent_dim=64)
model.compile(optimizer="adam")
```

The reconstruction term, the KL and the loss total are computed in float32 by design under
this policy: `1e-7` is below the smallest normal float16, so the BCE clip would be a no-op
and `log(0)` would reach the loss, and `exp(20)` in the KL would become `+inf`. `train_step`
also calls `optimizer.scale_loss` inside the tape and clips gradients in the *scaled* domain,
so a `LossScaleOptimizer` does not silently divide the whole weight update by the loss scale.

### XLA

`jit_compile=True` works for `gaussian` and `hypersphere`. It does **not** work for `vmf`:
the rejection sampler uses `keras.random.beta`, whose `StatelessRandomGammaV3` kernel has no
XLA-GPU implementation in TF 2.18. `VAE.compile()` forces `jit_compile=False` for that mode,
and vMF training runs roughly 5-10x slower per epoch as a result.

---

## 11. Training and Best Practices

- **Watch both loss components.** `reconstruction_loss` should fall steadily.
  `kl_loss` often rises early and then plateaus; a `kl_loss` pinned near zero is posterior
  collapse.
- **Scale pixels to `[0, 1]`.** The reconstruction term is binary cross-entropy and the
  decoder ends in a sigmoid; data outside that range makes the objective meaningless.
- **Match beta to the mode.** The three KL formulas are on different scales, so a beta tuned
  for `gaussian` is wrong for `vmf` by about two orders of magnitude. Never compare
  `total_loss` or `kl_loss` across modes; `reconstruction_loss` is the one comparable number.

### Choosing a latent geometry

Measured on MNIST in this repository at a single seed, so treat these as directional:

- **The spherical modes do not collapse.** `hypersphere` and `vmf` use every latent dimension,
  where `gaussian` flatlines at 5-6 active units however much capacity it is given, and their
  reconstruction keeps improving as `latent_dim` grows.
- **Only `vmf` regularises direction**, and only `vmf` wins on the quality of prior-sample
  decodes. The `hypersphere` KL is one-dimensional and says nothing about where on the sphere
  latents land, so its aggregate posterior stays concentrated.
- **Shrink the vMF beta as the latent grows** — roughly 10x per doubling of `latent_dim`. Its
  raw KL grows with dimension, so a beta tuned at `d=8` over-regularises at `d=32`.
- **At `latent_dim=2` use `gaussian`.** A 2-D sphere is a 1-D circle, and it loses to the plane.
  The spherical advantage only appears once there is real capacity to spread out in.

---

## 12. Serialization & Deployment

The model round-trips through Keras 3's `.keras` format, custom `train_step` included.

```python
import keras
from dl_techniques.models.vision.vae.model import VAE

model = VAE.from_variant("small", input_shape=(28, 28, 1), latent_dim=16)
model.compile(optimizer="adam")
model.save("my_vae_model.keras")

loaded = keras.models.load_model("my_vae_model.keras")
print(loaded.sample(1).shape)  # (1, 28, 28, 1)
```

---

## 13. Testing & Validation

The tests live in `tests/test_models/test_vae/` and cover variant construction, forward-pass
shapes for every `sampling_type`, the generative methods, the deprecated alias, and save/load
round-trips including optimizer state.

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest tests/test_models/test_vae -q
```

---

## 14. Troubleshooting

- **Blurry reconstructions.** Lower `kl_loss_weight` or move up a variant. Some blur is
  inherent to an objective that averages over plausible reconstructions.
- **`kl_loss` collapses to ~0 (posterior collapse).** Ramp beta from zero over the first
  epochs, or use `sampling_type="hypersphere"` / `"vmf"`, which have no origin to collapse
  toward.
- **`ValueError: sampling_type must be one of ...`.** `"hypersphere_controlled"` was removed;
  use `"gaussian"`, `"hypersphere"` or `"vmf"`.
- **`ValueError: Filters array length N must equal depths M`.** `filters` carries one entry
  per depth; override both together or neither.
- **`ValueError: Input dimensions must be at least 8x8`.** The encoder cannot downsample
  smaller inputs.
- **vMF training is slow.** Expected: that mode cannot be XLA-compiled (§10).

---

## 15. Citation

```bibtex
@article{kingma2013auto,
  title={Auto-encoding variational bayes},
  author={Kingma, Diederik P and Welling, Max},
  journal={arXiv preprint arXiv:1312.6114},
  year={2013}
}

@inproceedings{davidson2018hyperspherical,
  title={Hyperspherical Variational Auto-Encoders},
  author={Davidson, Tim R and Falorsi, Luca and De Cao, Nicola and
          Kipf, Thomas and Tomczak, Jakub M},
  booktitle={Uncertainty in Artificial Intelligence (UAI)},
  year={2018},
  note={arXiv:1804.00891}
}
```
