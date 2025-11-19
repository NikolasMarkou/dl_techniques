# Vector Quantized Variational Autoencoder (VQ-VAE)

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A production-ready, fully-featured implementation of a **Vector Quantized Variational Autoencoder (VQ-VAE)** in **Keras 3**, based on the foundational paper ["Neural Discrete Representation Learning"](https://arxiv.org/abs/1711.00937) by van den Oord et al. (2017).

The architecture separates the encoder, decoder, and a custom `VectorQuantizer` layer, enabling flexible and powerful discrete representation learning.

---

## Table of Contents

1. [Overview: What is VQ-VAE and Why It Matters](#1-overview-what-is-vq-vae-and-why-it-matters)
2. [The Problem VQ-VAE Solves](#2-the-problem-vq-vae-solves)
3. [How VQ-VAE Works: Core Concepts](#3-how-vq-vae-works-core-concepts)
4. [Architecture Deep Dive](#4-architecture-deep-dive)
5. [Quick Start Guide](#5-quick-start-guide)
6. [Component Reference](#6-component-reference)
7. [Configuration & Model Variants](#7-configuration--model-variants)
8. [Comprehensive Usage Examples](#8-comprehensive-usage-examples)
9. [Advanced Usage Patterns](#9-advanced-usage-patterns)
10. [Performance Optimization](#10-performance-optimization)
11. [Training and Best Practices](#11-training-and-best-practices)
12. [Serialization & Deployment](#12-serialization--deployment)
13. [Testing & Validation](#13-testing--validation)
14. [Troubleshooting & FAQs](#14-troubleshooting--faqs)
15. [Technical Details](#15-technical-details)
16. [Citation](#16-citation)

---

## 1. Overview: What is VQ-VAE and Why It Matters

### What is a VQ-VAE?

A **Vector Quantized Variational Autoencoder (VQ-VAE)** is a type of autoencoder that learns a **discrete** latent representation of data. Unlike standard VAEs that map inputs to a continuous space, VQ-VAEs map inputs to a finite set of learned embeddings in a "codebook". This is achieved through a process called vector quantization.

This discrete nature makes VQ-VAEs exceptionally well-suited for modalities that have an inherently discrete structure, such as language, and allows them to produce sharper, higher-fidelity images and audio compared to their continuous counterparts.

### Key Innovations

1.  **Discrete Latent Space**: By using a finite codebook, VQ-VAEs avoid the continuous latent space of VAEs, which can lead to issues like "posterior collapse". The discrete representation is often more powerful and interpretable.
2.  **Vector Quantization**: The encoder's continuous output is mapped to the nearest vector in a learnable codebook. This step is the core of the "VQ" mechanism.
3.  **Straight-Through Estimator**: To overcome the non-differentiable nature of the nearest-neighbor lookup, VQ-VAEs use a straight-through estimator to copy gradients from the decoder's input back to the encoder's output, enabling end-to-end training.
4.  **Decoupled Prior Learning**: The model separates learning the discrete representations from learning a prior over them. After a VQ-VAE is trained, a powerful autoregressive model (like a PixelCNN or Transformer) can be trained on the discrete latent codes to generate new data.

### Why VQ-VAEs Matter

**Standard VAE Problem**:
```
Problem: Generate a very sharp, high-fidelity image.
Standard VAE Approach:
  1. Train a VAE which learns a continuous latent space.
  2. The decoder learns to reconstruct images from points in this space.
  3. Limitation: The probabilistic nature of the decoder and the continuous space
     often leads to averaging, resulting in blurry reconstructions. Additionally,
     with powerful decoders, the model can learn to ignore the latent code, a
     problem known as "posterior collapse".
```

**VQ-VAE's Solution**:
```
VQ-VAE Approach:
  1. The encoder outputs a vector `z_e`.
  2. This vector is "snapped" to the closest vector `e_k` from a learned codebook.
  3. The decoder receives this exact, discrete vector `e_k` (or `z_q`).
  4. Benefit: Since the decoder always receives a precise vector from the codebook,
     it doesn't need to average over a noisy latent space, enabling it to produce
     much sharper and more detailed outputs. The discrete bottleneck also
     prevents posterior collapse.
```

### Real-World Impact

VQ-VAEs are a foundational component in many state-of-the-art generative models:

-   **High-Fidelity Image Generation**: Used in models like DALL-E and VQ-GAN to generate realistic images.
-   **Speech Synthesis and Voice Conversion**: Learning discrete representations of audio for high-quality text-to-speech.
-   **Data Compression**: The discrete codes provide a highly efficient way to compress data.

---

## 2. The Problem VQ-VAE Solves

### The Limitations of Standard VAEs

While powerful, standard VAEs have two primary weaknesses that VQ-VAEs address directly:

1.  **Posterior Collapse**: A common failure mode where the decoder becomes powerful enough to model the data distribution without using the information from the latent variable `z`. The KL divergence term in the VAE loss encourages the encoder to match the prior (e.g., a standard Gaussian), and if the decoder is strong enough, the encoder simply ignores the input and the latent codes become uninformative.

2.  **Blurry Reconstructions**: The continuous nature of the latent space and the VAE objective function (which maximizes log-likelihood) often results in the model averaging over many possible reconstructions, leading to blurry outputs.

```
┌─────────────────────────────────────────────────────────────┐
│  Standard VAE (VAE)                                         │
│                                                             │
│  Objective: Maximize the Evidence Lower Bound (ELBO).       │
│  Loss = Reconstruction Loss + KL Divergence                 │
│                                                             │
│  The Latent Space Problem:                                  │
│  - The KL term can cause the encoder to ignore the input,   │
│    leading to posterior collapse.                           │
│  - The decoder's probabilistic mapping from a continuous    │
│    space often produces blurry, "averaged" outputs.         │
└─────────────────────────────────────────────────────────────┘
```

### How VQ-VAE Changes the Game

VQ-VAEs introduce a discrete bottleneck that forces the model to commit to a specific representation.

```
┌─────────────────────────────────────────────────────────────┐
│  The VQ-VAE Solution                                        │
│                                                             │
│  1. Discrete Bottleneck: The encoder produces a continuous  │
│     vector `z_e(x)`, but the decoder receives a quantized   │
│     vector `z_q(x)` from a finite codebook. This hard       │
│     bottleneck prevents the model from ignoring the latent  │
│     code, thus avoiding posterior collapse.                 │
│                                                             │
│  2. The VQ-VAE Loss Function (3 parts):                     │
│     - Reconstruction Loss: Pushes the model to accurately   │
│       reconstruct the input (trains encoder and decoder).   │
│     - Codebook Loss: Moves the codebook vectors towards the │
│       encoder outputs (trains the codebook).                │
│     - Commitment Loss: Pushes the encoder to "commit" to a  │
│       codebook vector and not grow unbounded.               │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. How VQ-VAE Works: Core Concepts

### The Encoder-Quantizer-Decoder Architecture

A VQ-VAE consists of three main components connected by a non-differentiable quantization step.

```
┌───────────────────────────────────────────────────────────────────┐
│                          VQ-VAE Architecture                      │
│                                                                   │
│  ┌────────────────┐     ┌───────────────────────────────────┐     │
│  │    Encoder     │────►│        Vector Quantizer           │     │
│  │ (e.g., ConvNet)│     │                                   │     │
│  │                │     │ 1. Receives continuous z_e(x)     │     │
│  │ Maps input X   │     │ 2. Finds nearest embedding e_k in │     │
│  │ to z_e(x)      │     │    a learnable codebook E.        │     │
│  │                │     │ 3. Outputs quantized vector z_q(x)│     │
│  └────────────────┘     └───────────────────┬───────────────┘     │
│                                             │                     │
│  ┌───────────────┐                          │                     │
│  │    Decoder    │◄─────────────────────────┘                     │
│  │(e.g., ConvNet)│                                                │
│  │               │                                                │
│  │ Maps quantized│                                                │
│  │ vector z_q(x) │                                                │
│  │ back to X'    │                                                │
│  └───────────────┘                                                │
└───────────────────────────────────────────────────────────────────┘
```

### The Vector Quantization Step

The core of the VQ-VAE is the quantization layer. It takes the continuous output from the encoder, `z_e(x)`, and finds the closest vector in the codebook `E = {e_1, e_2, ..., e_K}`.

**`k = argmin_j ||z_e(x) - e_j||²`**
**`z_q(x) = e_k`**

This operation is a nearest-neighbor lookup and is not differentiable.

### The Straight-Through Estimator

To train the encoder, the gradient from the decoder needs to pass through the non-differentiable quantization step. The straight-through estimator solves this by simply copying the gradient from `z_q(x)` to `z_e(x)` during the backward pass.

-   **Forward pass**: `z_q(x) = quantize(z_e(x))`
-   **Backward pass**: The gradient `∇z_q` is passed directly to `z_e`.

This allows the encoder to receive a meaningful training signal, as if it had produced the quantized vector directly.

### The Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     VQ-VAE Complete Data Flow                           │
└─────────────────────────────────────────────────────────────────────────┘

STEP 1: ENCODING
────────────────
Input Image (B, H, W, C)
    │
    ├─► Encoder (e.g., ConvNet)
    │
    └─► Continuous Latent `z_e` (B, h, w, D_embedding)


STEP 2: QUANTIZATION (within VectorQuantizer Layer)
────────────────────────────────────────────────────
Continuous Latent `z_e`
    │
    ├─► For each vector in `z_e`'s spatial grid:
    │   └─► Find nearest vector `e_k` in the codebook E.
    │
    └─► Quantized Latent `z_q` (B, h, w, D_embedding)


STEP 3: DECODING
────────────────
Quantized Latent `z_q`
    │
    ├─► Decoder (e.g., ConvNet with upsampling)
    │
    └─► Reconstructed Image (B, H, W, C)


STEP 4: LOSS CALCULATION (During Training)
──────────────────────────────────────────
Input Image, Reconstructed Image, `z_e`, `z_q`
    │
    ├──► Reconstruction Loss:
    │    └─► e.g., MSE(Input, Reconstructed) * reconstruction_loss_weight
    │
    ├──► Codebook Loss:
    │    └─► ||sg[z_e] - z_q||²  (pulls codebook vectors toward encoder outputs)
    │
    └──► Commitment Loss:
         └─► β * ||z_e - sg[z_q]||² (pulls encoder outputs toward codebook vectors)

    (sg[] is the stop-gradient operator)
```

---

## 4. Architecture Deep Dive

### 4.1 Encoder

The encoder is a standard neural network (typically a CNN for images) that maps the input data to a lower-dimensional grid of feature vectors. The final channel dimension of the encoder's output must match the `embedding_dim` of the quantizer.

### 4.2 `VectorQuantizer` Layer

This custom layer contains the core VQ logic:
-   **Codebook**: A learnable embedding table of shape `(num_embeddings, embedding_dim)`.
-   **Nearest Neighbor Lookup**: Finds the closest codebook vector for each encoder output vector using L2 distance.
-   **Loss Calculation**: It computes and adds the codebook and commitment losses to the model's total loss.
-   **Straight-Through Gradient**: Implements the gradient bypass for the encoder.
-   **EMA Updates (Optional)**: Can use Exponential Moving Average updates for the codebook, which can be more stable than using an Adam optimizer. It uses an `epsilon` parameter for numerical stability during normalization.

### 4.3 Decoder

The decoder mirrors the encoder's architecture, taking the quantized latent vectors `z_q` as input and upsampling them to reconstruct the original input data.

---

## 5. Quick Start Guide

### Installation

```bash
# Ensure you have the required dependencies
pip install keras>=3.0 tensorflow>=2.18 numpy matplotlib
```

### Your First Generative Model (30 seconds)

Let's train a VQ-VAE on the MNIST dataset.

```python
import keras
from keras.api.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

from dl_techniques.models.vq_vae.model import VQVAEModel, VectorQuantizer

# 1. Load and preprocess data
(X_train, _), (X_test, _) = mnist.load_data()
X_train = np.expand_dims(X_train.astype("float32") / 255.0, -1)
X_test = np.expand_dims(X_test.astype("float32") / 255.0, -1)

# 2. Define Encoder and Decoder
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

# 3. Create a VQ-VAE model
# Note: VQVAEModel handles the creation of the VectorQuantizer layer internally.
model = VQVAEModel(
    encoder=encoder,
    decoder=decoder,
    num_embeddings=128,
    embedding_dim=embedding_dim,
    commitment_cost=0.25,
    reconstruction_loss_weight=1.0
)

# 4. Compile the model
# The loss is computed internally in train_step, so we only need an optimizer.
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3))
print("✅ VQ-VAE model created and compiled successfully!")
model.summary()

# 5. Train the model
history = model.fit(
    X_train,
    epochs=10, # Train longer for better results
    batch_size=128,
    validation_data=(X_test,)
)
print("✅ Training Complete!")

# 6. Reconstruct some test images
num_reconstructions = 10
reconstructed_images = model.predict(X_test[:num_reconstructions])

# 7. Visualize the results
plt.figure(figsize=(20, 4))
for i in range(num_reconstructions):
    # Original
    plt.subplot(2, num_reconstructions, i + 1)
    plt.imshow(X_test[i, :, :, 0], cmap='gray')
    plt.title("Original")
    plt.axis('off')
    # Reconstruction
    plt.subplot(2, num_reconstructions, i + 1 + num_reconstructions)
    plt.imshow(reconstructed_images[i, :, :, 0], cmap='gray')
    plt.title("Reconstructed")
    plt.axis('off')
plt.suptitle('Original vs. Reconstructed Digits')
plt.show()
```

---

## 6. Component Reference

### 6.1 `VQVAEModel` (Class)

**Purpose**: The main Keras `Model` subclass that assembles the encoder, quantizer, and decoder, and includes the custom `train_step`.

```python
vqvae = VQVAEModel(
    encoder=my_encoder,
    decoder=my_decoder,
    num_embeddings=512,
    embedding_dim=64,
    commitment_cost=0.25,
    use_ema=True,
    reconstruction_loss_weight=1.0
)
```

**Key Parameters**:

| Parameter | Description | Default |
| :--- | :--- | :--- |
| `encoder` | A Keras `Model` that outputs a continuous latent tensor. | (Required) |
| `decoder` | A Keras `Model` that reconstructs from a quantized latent tensor. | (Required) |
| `num_embeddings` | The size of the discrete codebook (K). | (Required) |
| `embedding_dim` | The dimensionality of each embedding vector (D). | (Required) |
| `commitment_cost`| The β parameter balancing the commitment loss. | `0.25` |
| `use_ema` | If `True`, uses Exponential Moving Average updates for the codebook. | `False` |
| `reconstruction_loss_weight` | Weight scaling the reconstruction loss term. | `1.0` |

**Key Methods**:
-   `train_step()` / `test_step()`: Custom logic to compute and track the three-part VQ-VAE loss.
-   `encode(inputs)`: Returns the continuous latent vectors `z_e`.
-   `quantize_latents(latents)`: Quantizes continuous latents `z_e` to `z_q`.
-   `decode(latents)`: Decodes a batch of quantized latent vectors `z_q`.
-   `encode_to_indices(inputs)`: Encodes inputs directly to their discrete codebook indices.
-   `decode_from_indices(indices)`: Decodes a grid of indices back into reconstructed inputs.

### 6.2 `VectorQuantizer` (Layer)

**Purpose**: A Keras `Layer` that performs the vector quantization, calculates losses, and handles the straight-through gradient.

**Key Parameters**:
-   `num_embeddings`: Size of codebook.
-   `embedding_dim`: Dimension of embeddings.
-   `use_ema`: Boolean to enable EMA updates.
-   `ema_decay`: Decay rate for EMA (e.g., 0.99).
-   `epsilon`: Small constant for numerical stability in EMA (default 1e-5).

---

## 7. Configuration & Model Variants

The VQ-VAE architecture is highly configurable. The key parameters to tune are:

| Parameter | Effect | Typical Values |
| :--- | :--- | :--- |
| `num_embeddings` (K) | Controls the size of the discrete "vocabulary". A larger K can represent more complex data but is harder to train. | 128, 256, 512, 1024+ |
| `embedding_dim` (D) | The dimensionality of each code vector. Affects the capacity of the encoder/decoder and the richness of the codes. | 32, 64, 128, 256 |
| `commitment_cost` (β) | Controls the encoder's incentive to match the codebook. Higher values can prevent the encoder's outputs from growing too large. | 0.1 - 2.0 (often 0.25) |
| `reconstruction_loss_weight` | Balances reconstruction quality vs. quantization strictness. | 1.0 (default) |
| `use_ema` | Determines the codebook update mechanism. EMA is often more stable than using a standard optimizer. | `False` or `True` |

---

## 8. Comprehensive Usage Examples

### Example 1: Reconstructing Images (as in Quick Start)

The primary function is to reconstruct inputs with high fidelity.

```python
# Assuming 'model' is trained and X_test is available
num_reconstructions = 10
test_images = X_test[:num_reconstructions]
reconstructed_images = model.predict(test_images)
# ... visualization code ...
```

### Example 2: Training a Prior on the Latent Codes

After training the VQ-VAE, you can extract the discrete codes and train a powerful autoregressive model (like PixelCNN or a Transformer) to learn the distribution of these codes.

```python
# 1. Encode the entire training set to discrete indices
indices = model.encode_to_indices(X_train)
print(f"Shape of latent codes: {indices.shape}")
# -> (60000, 7, 7) for our MNIST example

# 2. Train an autoregressive prior on these indices
# (PixelCNN/Transformer code would go here)
# prior_model.fit(indices, ...)

# 3. Sample new codes from the trained prior
# sampled_indices = prior_model.sample(num_samples=16)

# 4. Decode the new codes into images
# generated_images = model.decode_from_indices(sampled_indices)

# 5. Visualize the generated images
# ... visualization code ...
```
This two-stage process is how VQ-VAEs are used for high-quality generation.

---

## 9. Advanced Usage Patterns

### Pattern 1: Using EMA Updates

Exponential Moving Average (EMA) updates can provide a more stable way to train the codebook than relying on the optimizer. In this approach, the codebook is not a trainable weight but is updated manually based on a moving average of the encoder outputs that are mapped to each code.

```python
# Enable EMA during model creation
model_ema = VQVAEModel(
    encoder,
    decoder,
    num_embeddings=512,
    embedding_dim=64,
    use_ema=True,
    ema_decay=0.99, # A key hyperparameter to tune
    quantizer_initializer="uniform"
)
model_ema.compile(optimizer="adam")
# model_ema.fit(...)
```

### Pattern 2: Custom Initializers

You can pass any Keras initializer to the model to determine how the codebook is initialized.

```python
model = VQVAEModel(
    ...,
    quantizer_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.1)
)
```

---

## 10. Performance Optimization

### Mixed Precision Training

VQ-VAEs, with their convolutional backbones, can be significantly accelerated using mixed precision.

```python
# Enable mixed precision globally
keras.mixed_precision.set_global_policy('mixed_float16')

# Create model (will automatically use mixed precision)
model = VQVAEModel(...)
model.compile(...)
```

---

## 11. Training and Best Practices

### Monitoring the Loss Components

During training, it's crucial to monitor all three loss components:
-   **`reconstruction_loss`**: Should steadily decrease. This is the primary indicator of visual quality.
-   **`vq_loss`**: This metric tracks the sum of the **codebook loss** and **commitment loss**. Its absolute value is less important than its stability. If it fluctuates wildly, it can indicate training instability.
-   **Codebook Usage**: Monitor how many of the `num_embeddings` are actively being used. If a large fraction is unused, this indicates "codebook collapse", and you may need a smaller codebook or a different initialization strategy.

---

## 12. Serialization & Deployment

The `VQVAEModel` and `VectorQuantizer` layer are fully serializable using Keras 3's modern `.keras` format, including the custom `train_step`.

### Saving and Loading

```python
# Create and train model
model = VQVAEModel(...)
model.compile(optimizer="adam")
# model.fit(...)

# Save the entire model
model.save('my_vqvae_model.keras')
print("Model saved to my_vqvae_model.keras")

# Load the model in a new session.
# While @keras.saving.register_keras_serializable() handles most cases, 
# explicitly providing custom objects is a robust practice.
loaded_model = keras.models.load_model(
    'my_vqvae_model.keras',
    custom_objects={"VQVAEModel": VQVAEModel, "VectorQuantizer": VectorQuantizer}
)
print("Model loaded successfully")

# Verify that the model works
reconstruction = loaded_model.predict(X_test[:1])
print(f"Reconstructed image shape: {reconstruction.shape}")
```

---

## 13. Testing & Validation

### Unit Tests

```python
import keras
import numpy as np
import keras
from dl_techniques.models.vq_vae.model import VQVAEModel, VectorQuantizer


def test_model_creation():
    """Test model creation."""
    encoder = keras.Sequential([keras.layers.Dense(16)])
    decoder = keras.Sequential([keras.layers.Dense(784)])
    model = VQVAEModel(encoder, decoder, num_embeddings=128, embedding_dim=16)
    assert model is not None
    print("✓ VQ-VAE created successfully")

def test_forward_pass_shape():
    """Test the output shapes of a forward pass."""
    # Simple dummy networks
    embedding_dim = 4
    encoder = keras.Sequential([keras.layers.Dense(embedding_dim)])
    decoder = keras.Sequential([keras.layers.Dense(10)])
    
    model = VQVAEModel(encoder, decoder, num_embeddings=10, embedding_dim=embedding_dim)
    
    dummy_input = np.random.rand(2, 10, embedding_dim).astype("float32")
    output = model.predict(dummy_input)
    assert output.shape == (2, 10, 10)
    print("✓ Forward pass has correct shapes")

def test_generative_methods():
    """Test the encode/decode indices methods."""
    embedding_dim = 4
    encoder = keras.Sequential([keras.layers.Dense(embedding_dim)])
    decoder = keras.Sequential([keras.layers.Dense(10)])
    
    model = VQVAEModel(encoder, decoder, num_embeddings=10, embedding_dim=embedding_dim)
    
    dummy_input = np.random.rand(2, 10, embedding_dim).astype("float32")
    indices = model.encode_to_indices(dummy_input)
    # Shape depends on input spatial dims, here (2, 10)
    assert indices.shape == (2, 10)

    reconstruction = model.decode_from_indices(indices)
    assert reconstruction.shape == (2, 10, 10)
    print("✓ Generative methods work correctly")

# Run tests
if __name__ == '__main__':
    test_model_creation()
    test_forward_pass_shape()
    test_generative_methods()
    print("\n✅ All tests passed!")
```

---

## 14. Troubleshooting & FAQs

**Issue 1: Reconstructions are blurry or nonsensical.**

-   **Cause 1**: The model has not trained long enough.
-   **Solution 1**: Increase the number of epochs.
-   **Cause 2**: The model capacity is too low (e.g., `embedding_dim` or `num_embeddings` is too small).
-   **Solution 2**: Increase the model's capacity.
-   **Cause 3**: The `commitment_cost` is poorly tuned.
-   **Solution 3**: Try adjusting β (e.g., values between 0.1 and 2.0).

**Issue 2: The `vq_loss` is unstable or explodes.**

-   **Cause**: The encoder outputs are growing much faster than the codebook can adapt. The commitment loss is meant to prevent this, but may be too low.
-   **Solution 1**: Increase the `commitment_cost`.
-   **Solution 2**: Switch to EMA updates (`use_ema=True`), which are often more stable.

**Issue 3: Most of the codebook is not being used ("Codebook Collapse").**

-   **Cause**: The model finds a small subset of codes that are "good enough" and never explores the rest of the codebook. This limits the model's expressive power.
-   **Solution 1**: Try a different initialization for the embeddings via `quantizer_initializer`.
-   **Solution 2**: Decrease `num_embeddings`.

### Frequently Asked Questions

**Q: How does a VQ-VAE compare to a GAN (Generative Adversarial Network)?**

A:
-   **VQ-VAE**: An autoencoder-based model that learns explicit, discrete codes for data. It's stable to train and produces a useful compressed representation. High-quality generation is a two-step process (train VQ-VAE, then train a prior).
-   **GAN**: Uses a generator/discriminator setup in an adversarial game. Can produce extremely sharp images but is notoriously unstable to train and prone to issues like mode collapse. It does not naturally produce a structured latent representation.

**Q: Why not just use a standard VAE?**

A: VQ-VAEs were designed to fix two key problems in VAEs: posterior collapse and blurry image generation. The discrete bottleneck makes them more robust and capable of producing higher-fidelity results.

---

## 15. Technical Details

### The VQ-VAE Objective Function

The VQ-VAE loss function consists of three components that train the encoder, decoder, and codebook.

**Loss = L_reconstruction + L_codebook + L_commitment**

1.  **Reconstruction Loss (L_reconstruction)**:
    -   `log p(x | z_q(x))` which is implemented as MSE: `||x - Decoder(z_q(x))||²`
    -   Weighted by `reconstruction_loss_weight`.
    -   This term trains both the **encoder** and the **decoder**.

2.  **Codebook Loss (L_codebook)**:
    -   `||sg[z_e(x)] - e||²` where `e` is the chosen codebook vector.
    -   `sg` is the stop-gradient operator. This loss moves the codebook vectors `e` to be closer to the encoder's output `z_e(x)`.
    -   This term only trains the **codebook**.

3.  **Commitment Loss (L_commitment)**:
    -   `β * ||z_e(x) - sg[e]||²`
    -   This loss encourages the encoder's output to stay "committed" to the chosen codebook vector and prevents it from growing arbitrarily large.
    -   This term only trains the **encoder**.

---

## 16. Citation

If you use VQ-VAEs in your research, please cite the original paper:

```bibtex
@inproceedings{oord2017neural,
  title={Neural discrete representation learning},
  author={Van Den Oord, Aaron and Vinyals, Oriol and others},
  booktitle={Advances in neural information processing systems},
  pages={6306--6315},
  year={2017}
}
```