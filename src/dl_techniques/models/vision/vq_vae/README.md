# Vector Quantized Variational Autoencoder (VQ-VAE)

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

An implementation of a **Vector Quantized Variational Autoencoder (VQ-VAE)** in **Keras 3**,
based on the foundational paper ["Neural Discrete Representation
Learning"](https://arxiv.org/abs/1711.00937) by van den Oord et al. (2017).

`VQVAEModel` is a quantization scheme wrapped around an autoencoder you supply. It takes the
encoder and decoder as constructor arguments, builds the `VectorQuantizer` between them, and
adds a custom `train_step` that computes the three-part VQ-VAE loss.

---

## 1. Overview: What is VQ-VAE and Why It Matters

### What is a VQ-VAE?

A **VQ-VAE** learns a **discrete** latent representation. Where a standard VAE maps an input to
a continuous distribution, a VQ-VAE maps it to entries of a finite learned **codebook**. Every
latent position is snapped to its nearest codebook vector.

The discrete bottleneck suits modalities with inherently discrete structure, and it removes the
two failure modes that dog continuous VAEs on images: blur and posterior collapse.

### Key Innovations

1. **Discrete latent space.** A finite codebook of `K` vectors replaces the continuous latent.
   The result is often more interpretable and directly usable as tokens.
2. **Vector quantization.** The encoder's continuous output is replaced by its nearest codebook
   entry under L2 distance.
3. **The straight-through estimator.** Nearest-neighbour lookup has no gradient, so the
   decoder-input gradient is copied verbatim onto the encoder output.
4. **Decoupled prior learning.** Representation learning and prior learning are separate stages.
   Once the VQ-VAE is trained, an autoregressive model (PixelCNN, Transformer) is trained over
   the discrete codes to generate new data.

---

## 2. The Problem VQ-VAE Solves

Standard VAEs have two weaknesses that the discrete bottleneck addresses directly.

| Problem | Why it happens in a VAE | What VQ-VAE does |
| :--- | :--- | :--- |
| **Posterior collapse** | The KL term rewards matching the prior. A strong decoder can model the data unaided, so the encoder stops carrying information and the latent goes uninformative. | The decoder only ever receives a codebook vector. There is no way to ignore the latent, so nothing to collapse into. |
| **Blurry reconstructions** | The decoder maps a noisy continuous latent and ends up averaging over many plausible outputs. | The decoder receives one exact, discrete vector per position, so it need not hedge. |

The price is that generation becomes a two-stage process: a VQ-VAE has no usable prior of its
own, so a second model must learn the distribution over code indices before you can sample.

---

## 3. How VQ-VAE Works: Core Concepts

### The Quantization Step

Given the encoder output `z_e(x)` and a codebook `E = {e_1, ..., e_K}`, each spatial position is
replaced by its nearest entry:

```
k       = argmin_j || z_e(x) - e_j ||^2
z_q(x)  = e_k
```

This is a nearest-neighbour lookup, and it has no useful derivative.

### The Straight-Through Estimator

The forward pass quantizes; the backward pass pretends it did not.

- **Forward**: `z_q(x) = quantize(z_e(x))`
- **Backward**: the gradient arriving at `z_q(x)` is passed straight to `z_e(x)`

In code this is the identity `z_q = z_e + stop_gradient(z_q - z_e)`: it evaluates to `z_q`, and
its derivative with respect to `z_e` is `1`. The encoder is trained as if it had produced the
quantized vector itself.

### The Three-Part Objective

```
Loss = reconstruction_loss_weight * || x - Decoder(z_q(x)) ||^2       <- encoder + decoder
     +                              || sg[z_e(x)] - e ||^2            <- codebook only
     + commitment_cost            * || z_e(x) - sg[e] ||^2            <- encoder only

sg[.] is the stop-gradient operator.
```

Each term trains exactly one part of the model, and the stop-gradients are what make that true:

1. **Reconstruction** — the only term that reaches the decoder. Implemented as MSE.
2. **Codebook loss** — pulls each codebook vector toward the encoder outputs assigned to it.
   With `use_ema=True` the codebook is made non-trainable and moved by an exponential moving
   average instead, which is often steadier than letting the optimizer move it. The term is
   still computed and still reported in `vq_loss`; it just no longer drives anything.
3. **Commitment loss** — pulls the encoder outputs toward the codebook entry they chose,
   weighted by `commitment_cost` (the `beta` of the paper). Without it the encoder output can
   drift away from the codebook without bound, since quantization hides its magnitude from the
   reconstruction term.

The last two are tracked jointly as the `vq_loss` metric.

### The Complete Data Flow

```
STEP 1: ENCODING     Input (B, H, W, C)
    +-> encoder -> continuous latent z_e (B, h, w, D)   [last axis MUST be embedding_dim]

STEP 2: QUANTIZATION VectorQuantizer
    |-> nearest codebook entry per position -> z_q (B, h, w, D)
    +-> codebook + commitment losses added to the model's losses

STEP 3: DECODING     decoder(z_q) -> reconstruction (B, H, W, C)

STEP 4: LOSS         total = reconstruction + codebook + commitment
```

---

## 4. Architecture Deep Dive

### 4.1 Encoder

Any Keras model that maps inputs to a grid of feature vectors. **Its output channel count must
equal `embedding_dim`**; this is the one hard interface constraint, because quantization
compares encoder outputs against codebook vectors directly.

### 4.2 `VectorQuantizer` Layer

Lives at `dl_techniques.layers.generative.vector_quantizer.VectorQuantizer`. It holds:

- **The codebook** — an embedding table of shape `(num_embeddings, embedding_dim)`, initialized
  by `initializer` (`"uniform"` by default).
- **Nearest-neighbour lookup** — L2 distance against every codebook entry.
- **Loss computation** — the codebook and commitment terms, added to the layer's losses.
- **Straight-through gradient** — the bypass described in §3.
- **Optional EMA updates** — with `use_ema=True` the codebook weight is created with
  `trainable=False` and maintained by a debiased exponential moving average at rate
  `ema_decay`, with `epsilon` (default `1e-5`) guarding the cluster-size normalization against
  division by zero. The EMA accumulators are held in float32 whatever the compute policy.

### 4.3 Decoder

Any Keras model mapping quantized latents back to the input space. It sees `z_q` and nothing
else, which is why sharp outputs are achievable — there is no continuous noise to average over.

---

## 5. Quick Start Guide

```python
import keras
import numpy as np

from dl_techniques.models.vision.vq_vae import VQVAEModel

# 1. Data, scaled to [0, 1]
x_train = np.random.rand(256, 28, 28, 1).astype("float32")
x_test = np.random.rand(64, 28, 28, 1).astype("float32")

# 2. Encoder and decoder. The encoder's last axis must equal embedding_dim.
embedding_dim = 16
encoder = keras.Sequential([
    keras.layers.Input(shape=(28, 28, 1)),
    keras.layers.Conv2D(32, 3, activation="relu", strides=2, padding="same"),
    keras.layers.Conv2D(64, 3, activation="relu", strides=2, padding="same"),
    keras.layers.Conv2D(embedding_dim, 1, padding="same"),
])
decoder = keras.Sequential([
    keras.layers.Input(shape=(7, 7, embedding_dim)),
    keras.layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same"),
    keras.layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same"),
    keras.layers.Conv2DTranspose(1, 3, padding="same"),
])

# 3. Wrap them. The VectorQuantizer is created internally.
model = VQVAEModel(
    encoder=encoder,
    decoder=decoder,
    num_embeddings=128,
    embedding_dim=embedding_dim,
    commitment_cost=0.25,
    reconstruction_loss_weight=1.0,
)

# 4. Compile. The loss lives in train_step, so only an optimizer is needed.
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3))

# 5. Train and reconstruct
model.fit(x_train, epochs=2, batch_size=64, validation_data=(x_test,), verbose=0)
reconstructions = model.predict(x_test[:10], verbose=0)
print(reconstructions.shape)  # (10, 28, 28, 1)
```

---

## 6. Component Reference

### 6.1 `VQVAEModel`

**Location**: `dl_techniques.models.vision.vq_vae.VQVAEModel`

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `encoder` | (required) | Keras model producing continuous latents with `embedding_dim` channels. |
| `decoder` | (required) | Keras model reconstructing from quantized latents. |
| `num_embeddings` | (required) | Codebook size `K`. Must be positive. |
| `embedding_dim` | (required) | Code vector dimension `D`. Must be positive. |
| `commitment_cost` | `0.25` | The `beta` weighting the commitment term. |
| `use_ema` | `False` | Update the codebook by EMA instead of by gradient. |
| `ema_decay` | `0.99` | EMA rate, used only when `use_ema=True`. |
| `reconstruction_loss_weight` | `1.0` | Weight of the MSE term. Must be positive. |
| `quantizer_initializer` | `"uniform"` | Initializer for the codebook embeddings. |

**Key methods**:

- `encode(inputs)` — continuous latents `z_e`.
- `quantize_latents(latents)` — `z_e` to `z_q`.
- `decode(latents)` — quantized latents back to inputs.
- `encode_to_indices(inputs)` — straight to discrete codebook indices.
- `decode_from_indices(indices)` — indices back to reconstructions.
- `train_step` / `test_step` — track `total_loss`, `reconstruction_loss` and `vq_loss`.

Calling the model returns the reconstruction tensor, so `model.predict(x)` gives images, not a
dictionary.

### 6.2 `create_vq_vae`

**Location**: `dl_techniques.models.vision.vq_vae.create_vq_vae`

The same arguments as the constructor, but with `num_embeddings=512` and `embedding_dim=64`
defaulted. `encoder` and `decoder` stay required, deliberately: inventing a default backbone
would silently pick an architecture the paper does not specify.

```python
model = create_vq_vae(encoder, decoder, num_embeddings=256, embedding_dim=16)
```

### 6.3 `VectorQuantizer`

**Location**: `dl_techniques.layers.generative.vector_quantizer.VectorQuantizer` — note that the layer
lives in `layers/`, not in this package. `VQVAEModel` constructs one named `vector_quantizer`;
you only need to import it directly if you are building your own wrapper.

| Parameter | Default |
| :--- | :--- |
| `num_embeddings` | (required) |
| `embedding_dim` | (required) |
| `commitment_cost` | `0.25` |
| `initializer` | `"uniform"` |
| `use_ema` | `False` |
| `ema_decay` | `0.99` |
| `epsilon` | `1e-5` |

---

## 7. Configuration & Model Variants

**There is no `MODEL_VARIANTS` table and none was invented.** VQ-VAE is a quantization scheme
around an arbitrary autoencoder; with no backbone of its own there is no architecture to give
named scales to. What you tune instead:

| Parameter | Effect | Typical values |
| :--- | :--- | :--- |
| `num_embeddings` (K) | Size of the discrete vocabulary. Larger K represents more, but is harder to keep fully used. | 128, 256, 512, 1024+ |
| `embedding_dim` (D) | Width of each code, and therefore the encoder's required output width. | 32, 64, 128, 256 |
| `commitment_cost` (beta) | How hard the encoder is pushed toward its chosen code. Too low and encoder outputs drift; too high and the encoder cannot move. | 0.1 - 2.0, usually 0.25 |
| `reconstruction_loss_weight` | Reconstruction fidelity against quantization strictness. | 1.0 |
| `use_ema` | Codebook update rule. EMA is often steadier than the optimizer. | `False` or `True` |

The latent grid size is set entirely by your encoder's downsampling: an encoder with two
stride-2 stages turns a `28x28` input into a `7x7` grid of codes.

---

## 8. Usage Examples

### Example 1: Discrete codes as tokens

`encode_to_indices` returns one integer per latent position, which is the representation an
autoregressive prior is trained on.

```python
import numpy as np

indices = model.encode_to_indices(x_test[:32])
print(indices.shape)      # (32, 7, 7) for the Quick Start encoder
print(int(np.max(indices)) < 128)  # every index is a valid codebook slot

# ... train a PixelCNN or Transformer prior on `indices` ...
# sampled = prior.sample(16)
generated = model.decode_from_indices(indices)
print(generated.shape)    # (32, 28, 28, 1)
```

This two-stage recipe — train the VQ-VAE, then train a prior over its codes — is how VQ-VAEs
are used for generation.

### Example 2: EMA codebook updates

With `use_ema=True` the codebook is not moved by the optimizer at all; it tracks a moving
average of the encoder outputs assigned to each code.

```python
model_ema = VQVAEModel(
    encoder,
    decoder,
    num_embeddings=512,
    embedding_dim=16,
    use_ema=True,
    ema_decay=0.99,
    quantizer_initializer="uniform",
)
model_ema.compile(optimizer="adam")
```

Any Keras initializer can be passed instead of the `"uniform"` string, for example
`keras.initializers.RandomNormal(mean=0.0, stddev=0.1)`.

---

## 9. Advanced Usage Patterns

### Pattern 1: A VQ bottleneck in an existing network

`VectorQuantizer` is a plain layer and does not need `VQVAEModel` around it. Drop it into any
architecture where you want a discrete bottleneck. After a forward pass its codebook and
commitment terms show up in `.losses` on the enclosing model, ready to add to your objective.

```python
import keras
from dl_techniques.layers.generative.vector_quantizer import VectorQuantizer

inputs = keras.Input(shape=(16, 16, 32))
quantized = VectorQuantizer(num_embeddings=256, embedding_dim=32)(inputs)
outputs = keras.layers.Conv2D(3, 1)(quantized)
net = keras.Model(inputs, outputs)
```

### Pattern 2: Weighting reconstruction against quantization

`reconstruction_loss_weight` and `commitment_cost` move in opposite directions. Raising the
first favours fidelity and lets encoder outputs stray further from the codebook; raising the
second keeps the encoder tight against its codes at some cost in reconstruction. Change one at
a time — moving both hides which one caused the effect.

---

## 10. Performance Optimization

```python
import keras
keras.mixed_precision.set_global_policy("mixed_float16")
```

The convolutional encoder and decoder are the bulk of the compute and benefit from mixed
precision. Set the policy before building the model, then construct and compile as usual.

---

## 11. Training and Best Practices

- **`reconstruction_loss`** should fall steadily. It is the direct measure of output quality.
- **`vq_loss`** is the sum of everything the model collected through `add_loss` — the codebook
  and commitment terms plus any regularizer on your encoder or decoder. It matters for its
  stability, not its absolute value; wild swings mean encoder and codebook are chasing each other.
- **Codebook usage** is the metric people forget. Count the distinct values coming out of
  `encode_to_indices`. If most of `num_embeddings` never appears, the codebook has collapsed
  onto a handful of entries and the extra capacity is doing nothing.
- **Scale inputs consistently**, since the reconstruction term is a plain MSE with no output
  activation forced on the decoder.

---

## 12. Serialization & Deployment

`VQVAEModel` and `VectorQuantizer` are both registered, so a saved model round-trips through
Keras 3's `.keras` format with the custom `train_step` intact.

```python
import keras
from dl_techniques.models.vision.vq_vae import VQVAEModel

model.save("my_vqvae_model.keras")
loaded = keras.models.load_model("my_vqvae_model.keras")
print(loaded.predict(x_test[:1], verbose=0).shape)
```

Registration is by package-qualified key, so `custom_objects` is not needed for models saved by
this package.

---

## 13. Testing & Validation

The tests live in `tests/test_models/test_vq_vae/` and cover construction and validation
errors, forward-pass shapes, the index round-trip, EMA and gradient codebook updates, and
save/load.

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest tests/test_models/test_vq_vae -q
```

---

## 14. Troubleshooting

- **Shape error at the quantizer.** The encoder's last axis must equal `embedding_dim`. This is
  the most common wiring mistake; a `1x1` convolution at the end of the encoder is the usual fix.
- **`ValueError: num_embeddings must be positive`** (likewise `embedding_dim`,
  `reconstruction_loss_weight`) — these are validated in the constructor, before anything is
  built.
- **Blurry or meaningless reconstructions.** Usually under-training or too little capacity in
  the encoder and decoder; the quantizer is rarely the cause.
- **`vq_loss` explodes.** Encoder outputs are outrunning the codebook. Raise `commitment_cost`,
  or switch to `use_ema=True`.
- **Codebook collapse (most codes unused).** Try a different `quantizer_initializer`, lower
  `num_embeddings`, or EMA updates.

---

## 15. Citation

```bibtex
@inproceedings{oord2017neural,
  title={Neural discrete representation learning},
  author={Van Den Oord, Aaron and Vinyals, Oriol and Kavukcuoglu, Koray},
  booktitle={Advances in Neural Information Processing Systems},
  pages={6306--6315},
  year={2017}
}
```
