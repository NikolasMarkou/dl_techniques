# Topographic VAE

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

An implementation of the **Topographic VAE** from
["Topographic VAEs Learn Equivariant Capsules"](https://arxiv.org/abs/2109.01394)
by Keller & Welling (NeurIPS 2021), in **Keras 3**.

The model is trained on **unlabelled sequences of transformed observations** — a digit
rotated through 360°, a sprite slid across a grid — with no transformation labels and no
supervision. The claim it makes is that a topographic generative model, given one inductive
bias, produces capsules whose activations **roll as the input transforms**, and that the
quantity a capsule has rolled can be *measured* against the transformation that caused it.

---

## 1. The problem, and why it is two literatures

Two research traditions meet here and are usually kept apart.

**Topographic generative models** drop the independence assumption of a VAE prior. Latent
variables sit on a lattice, so neighbours share a variance and correlate in energy; the
ordering that emerges from training alone looks like Hubel & Wiesel's orientation maps.
Unsupervised, and it discovers its own organisation.

**Equivariant networks** impose symmetry by construction — a group convolution groups
features into equivalence classes and permutes activations *within* a group when the input
is transformed, so pose is preserved rather than averaged away. Supervised data, or
hard-coded structure.

This model asks whether the first can produce the second. The answer is yes, with one
inductive bias: **temporal coherence with a shift**.

---

## 2. The mechanism

A Student's-t variable is a scale mixture:

```
T = Z * sqrt(nu / sum_i U_i^2)
```

The topographic prior's construction is *arithmetic* on that ratio, not an energy
minimisation, which is what makes it trainable by ordinary variational inference. Two
Gaussian posteriors are inferred per frame — `z` and `u` — and the topographic variable is

```
t = sqrt(2) * (z - mu) / sqrt(nu * energy_l + eps)          # Eq. 6
energy_l = sum_{delta = -L..L} W R_delta u_{l+delta}^2       # Eq. 5
```

`W` is a circulant neighbourhood-sum matrix of width `K`, and `R_delta` is the cyclic
permutation by `delta` steps. That is the whole bias: **neighbouring latent coordinates
share a variance, and the sharing is over time as well as over space.**

The three coherence modes are the paper's ablation, and they differ only in the `R_delta`
term:

| `temporal_coherence` | `R_delta` | This is |
|---|---|---|
| `"shifting"` | the roll by `delta` | the Topographic VAE |
| `"stationary"` | identity | BubbleVAE |
| `"none"` | identity, and `L` must be 0 | no temporal bias at all |

Because a roll is invertible, `t_0` alone determines the whole sequence: decode `t_0`, then
roll the capsule and decode again, and the frames come out. That is what
`traverse_capsules()` does, and it is the model's central falsifiable claim rather than a
figure it draws.

---

## 3. Quick start

```python
import numpy as np
from dl_techniques.models.vision.topographic_vae import create_topographic_vae

model = create_topographic_vae("mnist", coherence_window=6, neighborhood_size=3)

sequences = np.random.random((8, 18, 28, 28, 3)).astype("float32")
model.fit(sequences, sequences, epochs=10, batch_size=4)

outputs = model(sequences, training=False)
outputs["reconstruction"].shape      # (8, 18, 28, 28, 3)
outputs["t"].shape                   # (8, 18, 324) — 18 capsules of 18

# Decode unseen frames from ONE encoded activation:
traversal = model.traverse_capsules(outputs["t"][:, 0], num_steps=18)
```

Two construction surfaces, both binding `create_topographic_vae`:

```python
model = TopographicVAE.from_variant("dsprites", coherence_window=5)   # uncompiled
model = create_topographic_vae("dsprites")                             # compiled with the ELBO
```

`pretrained=True` raises `NotImplementedError` rather than returning random weights; there
are no released weights to fetch. Pass `weights_path=` for a local checkpoint.

---

## 4. The two baselines are MODES, not subclasses

Both comparisons of the paper's Tables 1–2 are constructor arguments on one class, so the
two arms differ by exactly the mechanism under test and by nothing else:

| Arm | How | What changes |
|---|---|---|
| **plain VAE** | `use_variance_variables=False` | `u` is dropped, there is no energy to build, and `t = z - mu` directly |
| **BubbleVAE** | `temporal_coherence="stationary"` | the same construction with the roll removed |

The plain-VAE arm **skips** the topography's `1/sqrt(.)` rather than dividing by a
near-zero energy. Dividing would silently turn `epsilon` into a temperature and blow the
latent up, so the branch is a subtraction rather than a division — and the width of `t` and
of the decoder's input are identical either way, which is what makes the arms comparable.

---

## 5. The decoder takes `t` ALONE

Equation 12 writes the likelihood as `p(x | g_theta(t))` and Section 4.3 calls `T` "the
first layer of the generative decoder". That is the literal reading, and it is also the
only one under which the topography is load-bearing: handing the decoder `concat([u, z])`
alongside `t` would let it read the topography off a side channel and bypass the
normalisation entirely, and the model would become unfalsifiable — the capsule structure
could vanish while the likelihood stayed flat.

`tests/test_models/test_topographic_vae/test_model.py` pins this by perturbing `t` alone and
requiring the reconstruction to move.

---

## 6. What is measured, and what the numbers mean

The paper reports two equivariance numbers, and **they are not the same measurement**:

- **`equivariance_error`** (Eq. 13) is a *smoothness* measure. A representation that is
  **invariant** — every timestep identical — is perfectly smooth, so it scores a low error
  too. Reading it as equivariance is the mistake this package is built to make hard.
- **`capcorr`** (Eq. 15/16) is the equivariance measurement. `1.0` for a perfectly
  equivariant representation; near zero for an invariant one.

`dl_techniques/metrics/topographic.py` carries both, plus `capcorr_per_capsule` so the
pooled number's assumption — that all capsules roll simultaneously, which the paper
reports empirically and the mode reduction takes on trust — is observable rather than
assumed. Measured on the package's own fixtures:

| Representation | `equivariance_error` | `capcorr` |
|---|---|---|
| perfectly equivariant | 1.6e-15 | **1.0000** |
| invariant (timestep-constant) | 0.0 | undefined (`nan`) |
| rolling at 2× the true rate | low | 0.975 |
| rolling at 3× the true rate | low | 0.646 |

`log_likelihood()` is the IWAE estimate of `log p(x)` in nats, summed over pixels and
sequence. It is an **upper bound**, so it tightens as `K` grows; on the reference instance
`K = 1, 2, 4, 8, 16` gives `-7064.3, -5841.5, -5689.3, -5464.9, -5185.9`, and that
monotonicity is the sharpest available check on the signs of the weights.

---

## 7. Reproducibility, and the one thing that is deliberately not reproducible

`call()` samples, so the reconstruction of one model varies between calls. For **evaluation**
that is unacceptable — a metric computed on an unseeded draw is not comparable across runs,
and two baseline rows would not be comparable to each other. So the constructor takes:

```python
TopographicVAE(..., sampling_seed=3)     # model(x, training=False) is now reproducible
```

Set it, or write to the property later. It is a plain argument because
`keras.utils.set_random_seed` does **not** reseed `keras.random` on this backend: three
`set_random_seed(3)` calls before three draws give three *different* draws, while an
explicit `seed=3` gives the same one three times. Reproducibility has to be asked for.

For **training** the sampler is deliberately left unseeded. The ELBO's reparameterization
draws must be unbiased, and pinning them would make every batch reuse one noise draw,
biasing the gradient rather than merely making it repeatable.

---

## 8. Variants

`MODEL_VARIANTS` carries the paper's two settings, transcribed from Section A.3:

| | `mnist` | `dsprites` |
|---|---|---|
| frames | 28×28×3 | 64×64×1 |
| sequence length `S` | 18 | 15 |
| capsules × dim | 18 × 18 | 15 × 15 |
| `latent_dim` | 324 | 225 |
| encoder / decoder | `[972, 648]` / `[648, 972]` | `[674, 450]` / `[450, 674]` |
| coherence window `L` | `S/3 = 6` (the paper's best equivariance) | `S/3 = 5` |

---

## 9. Serialization

```python
model.save("tvae.keras")
restored = keras.models.load_model("tvae.keras")   # registered, no custom_objects
```

Registered under `dl_techniques.models.topographic_vae.model>TopographicVAE` — the
`vision/` family directory is stripped, because a family is a filing decision and not a
namespace. Round-trip is exact: `encode()` values and every weight come back bit-identical,
which the suite checks at `atol=0.0`.

---

## 10. Training and evaluation

The trainer is `src/train/topographic_vae/`:

```
MPLBACKEND=Agg .venv/bin/python -m train.topographic_vae.train_topographic_vae \
  --dataset mnist --transform rotation --epochs 100
```

It builds the transformation sequences from
`dl_techniques/datasets/vision/transform_sequences.py`, writes the full run-directory
contract, and records the four numbers the paper's tables are read from. See
`src/train/topographic_vae/README.md`.

---

## 11. Implementation notes

Three things in this implementation are load-bearing and easy to get wrong:

1. **The window stack `G[delta+L] = W R_delta` is a single `(2L+1, D, D)` constant**,
   created through `add_weight(initializer=...)` and never `.assign()`-ed in `build()` —
   `StatelessScope` makes an in-`build()` assignment invisible to the saved weights.
2. **A single shared initializer instance makes every kernel identical.** A Keras
   initializer carries a seed generator and a seeded one replays the same values on every
   call, so passing one instance to every `Dense` produced a `u_encoder` that was a
   bit-identical clone of `z_encoder` — `KL_u == KL_z` identically at initialization.
   `_fresh_initializer` resolves the spec per layer, and offsets a caller-pinned seed by
   the layer's position.
3. **The energy is clamped at zero.** It is non-negative in exact arithmetic — `G`'s entries
   are 0 or 1 — but computed as a signed sum, so a `G` carrying negative entries produces a
   negative energy, and `epsilon` does not rescue that: adding a positive constant to a
   negative number leaves it negative and the `sqrt` returns `NaN`.

---

## References

- Keller & Welling, 2022. Topographic VAEs Learn Equivariant Capsules. NeurIPS 2021.
  [arXiv:2109.01394](https://arxiv.org/abs/2109.01394)
- Welling, Osindero & Hinton, 2003. Learning Sparse Topographic Representations with
  Products of Student-t Distributions. NeurIPS.
- Hyvarinen, Hurri & Varrynen, 2004. A Unifying Framework for Natural Image Statistics:
  Spatiotemporal Activity Bubbles. *Neurocomputing* 58-60.
  [doi:10.1016/j.neucom.2004.09.007](https://doi.org/10.1016/j.neucom.2004.09.007)
- Kingma & Welling, 2014. Auto-Encoding Variational Bayes. ICLR.
  [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)
- Gregor et al., 2012. Object-Centric Learning with Slot Attention. NeurIPS.