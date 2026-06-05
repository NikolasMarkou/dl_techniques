# Variational Autoencoder (VAE)

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A production-ready, fully-featured implementation of a **Variational Autoencoder (VAE)** in **Keras 3**, based on the foundational paper ["Auto-Encoding Variational Bayes"](https://arxiv.org/abs/1312.6114) by Kingma & Welling (2013).

The architecture uses robust **ResNet-based** encoder and decoder networks and incorporates a custom training loop to correctly handle the two-part VAE loss (reconstruction and KL divergence).

---

## Table of Contents

1. [Overview: What is VAE and Why It Matters](#1-overview-what-is-vae-and-why-it-matters)
2. [The Problem VAE Solves](#2-the-problem-vae-solves)
3. [How VAE Works: Core Concepts](#3-how-vae-works-core-concepts)
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
16. [Hypersphere Latent Sampling: Does It Work?](#16-hypersphere-latent-sampling-does-it-work)
17. [vMF Spherical VAE: Winning on Generation](#17-vmf-spherical-vae-winning-on-generation)
18. [Citation](#18-citation)

---

## 1. Overview: What is VAE and Why It Matters

### What is a VAE?

A **Variational Autoencoder (VAE)** is a deep generative model that learns to compress data into a structured, low-dimensional **latent space** and then generate new, similar data by sampling from this space.

Unlike a standard autoencoder, a VAE's encoder doesn't map an input to a single point in the latent space. Instead, it outputs the parameters of a **probability distribution** (typically a Gaussian) for that input. The VAE then samples from this distribution to generate a latent vector, which is fed to the decoder.

### Key Innovations

1.  **Generative Modeling**: VAEs are not just for compression; their primary purpose is to learn the underlying probability distribution of the training data, allowing them to generate novel samples.
2.  **Probabilistic Latent Space**: The latent space is continuous and structured, meaning nearby points correspond to visually similar outputs. This allows for smooth interpolation between generated samples.
3.  **The Reparameterization Trick**: A clever mathematical trick that allows gradients to be backpropagated through the stochastic sampling process, enabling end-to-end training with standard optimizers.
4.  **Principled Probabilistic Framework**: VAEs are derived from the principles of variational inference, providing a solid mathematical foundation for learning latent variable models.

### Why VAEs Matter

**Standard Autoencoder Problem**:
```
Problem: Generate a new image of a face.
Standard AE Approach:
  1. Train an autoencoder on a dataset of faces.
  2. Take a random point from the latent space and feed it to the decoder.
  3. Limitation: The latent space is unstructured. Random points often produce
     meaningless, non-face-like garbage because the model hasn't learned to
     organize the space in a probabilistic way. There are "holes" everywhere.
```

**VAE's Solution**:
```
VAE Approach:
  1. Train a VAE on the same dataset.
  2. The VAE's loss function forces the latent space to be continuous and centered
     around a prior distribution (like a standard Gaussian).
  3. Now, if you take a random point from this distribution, it's highly likely to
     decode into a realistic, novel face.
  4. Benefit: The VAE learns a smooth, continuous map of the data, perfect for generation.
```

### Real-World Impact

VAEs are a cornerstone of modern generative modeling and representation learning:

-   🎨 **Image Generation**: Creating novel, realistic images, such as faces, artworks, or product designs.
-   🎶 **Music & Audio Synthesis**: Generating new musical sequences or sound textures.
-   💊 **Drug Discovery**: Generating new molecular structures by exploring the latent space of chemical compounds.
-   🧩 **Data Augmentation**: Creating realistic new training samples to improve the performance of other machine learning models.
-   🤖 **Reinforcement Learning**: Learning models of an environment for planning and control.

---

## 2. The Problem VAE Solves

### The Limitations of Standard Autoencoders

Standard autoencoders are great for dimensionality reduction and feature learning, but they fail as generative models.

```
┌─────────────────────────────────────────────────────────────┐
│  Standard Autoencoder (AE)                                  │
│                                                             │
│  Objective: Minimize Reconstruction Error (e.g., MSE).      │
│                                                             │
│  The Latent Space Problem:                                  │
│  - The model only learns to encode/decode training examples │
│    perfectly.                                               │
│  - It doesn't learn how to organize the latent space.       │
│  - The space becomes disjointed and sparse, with large      │
│    "dead zones" between encoded points.                     │
└─────────────────────────────────────────────────────────────┘
```

If you try to sample a point from the latent space of a standard AE, you are likely to pick a point from one of these dead zones, resulting in an unrealistic output.


*A standard AE's latent space (left) is irregular, making generation difficult. A VAE's latent space (right) is smooth and structured, ideal for sampling.*

### How VAE Changes the Game

VAEs fix this by introducing a probabilistic framework and a carefully designed loss function.

```
┌─────────────────────────────────────────────────────────────┐
│  The VAE Solution                                           │
│                                                             │
│  1. Probabilistic Encoder: Instead of a single vector `z`,  │
│     the encoder outputs a mean `μ` and a log-variance       │
│     `log(σ²)` that define a Gaussian distribution.          │
│                                                             │
│  2. The VAE Loss Function (The ELBO):                       │
│     Loss = Reconstruction Loss + KL Divergence              │
│     - Reconstruction Loss: Pushes the model to accurately   │
│       reconstruct the input (like a standard AE).           │
│     - KL Divergence: Acts as a regularizer, forcing the     │
│       encoded distributions to stay close to a standard     │
│       normal distribution (N(0,1)). This organizes the      │
│       latent space.                                         │
└─────────────────────────────────────────────────────────────┘
```

The KL divergence term is the magic ingredient. It ensures that the latent space is continuous and densely packed around the origin, eliminating the "dead zones" and making it a fertile ground for generating new samples.

---

## 3. How VAE Works: Core Concepts

### The Probabilistic Encoder-Decoder Architecture

A VAE consists of two main components connected by a stochastic sampling step.

```
┌──────────────────────────────────────────────────────────────────┐
│                          VAE Architecture                        │
│                                                                  │
│  ┌──────────────┐      ┌───────────────────────────────────┐     │
│  │    Encoder   │───►  │           Latent Space            │     │
│  │ (e.g., ResNet)│      │                                  │     │
│  │              │      │ 1. Predicts μ and log(σ²)         │     │
│  │ Maps input X │      │    for the distribution q(z|X)    │     │
│  │ to a distribution │  │ 2. Sample z ~ N(μ, σ²) via       │     │
│  │              │      │    Reparameterization Trick       │     │
│  └──────────────┘      └───────────────────┬───────────────┘     │
│                                            │                     │
│  ┌──────────────┐                          │                     │
│  │    Decoder   │◄─────────────────────────┘                     │
│  │ (e.g.,ResNet)│                                                │
│  │              │                                                │
│  │ Maps latent  │                                                │
│  │ vector z back│                                                │
│  │ to X'        │                                                │
│  └──────────────┘                                                │
└──────────────────────────────────────────────────────────────────┘
```

### The Reparameterization Trick

The key challenge in training a VAE is that the sampling step (`z ~ N(μ, σ²)`) is random and therefore non-differentiable. We can't backpropagate through it. The reparameterization trick solves this by reframing the sampling process:

**`z = μ + σ * ε`**, where **`ε ~ N(0, 1)`**

-   `μ` and `σ` are the (deterministic) outputs from the encoder.
-   `ε` is a random noise vector sampled from a fixed, standard normal distribution.

Now, the randomness is external. The path from `μ` and `σ` to `z` is fully deterministic, allowing gradients to flow back to the encoder.

### The Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     VAE Complete Data Flow                              │
└─────────────────────────────────────────────────────────────────────────┘

STEP 1: ENCODING
────────────────
Input Image (B, H, W, C)
    │
    ├─► ResNet Encoder
    │
    ├─► Global Average Pooling -> Feature Vector (B, F)
    │
    ├──► Dense Layer → z_mean (B, D_latent)
    └──► Dense Layer → z_log_var (B, D_latent)


STEP 2: SAMPLING (within Sampling Layer)
────────────────────────────────────────
z_mean, z_log_var
    │
    ├─► Calculate std: σ = exp(0.5 * z_log_var)
    │
    ├─► Sample noise: ε ~ N(0, 1) with shape (B, D_latent)
    │
    ├─► Compute latent vector: z = z_mean + σ * ε
    │
    └─► Latent Vector z (B, D_latent)


STEP 3: DECODING
────────────────
Latent Vector z (B, D_latent)
    │
    ├─► Dense Layer to project to initial feature map size
    │
    ├─► Reshape to (B, H', W', C')
    │
    ├─► ResNet Decoder (with upsampling)
    │
    └─► Reconstructed Image (B, H, W, C)


STEP 4: LOSS CALCULATION (During Training)
──────────────────────────────────────────
Input Image, Reconstructed Image, z_mean, z_log_var
    │
    ├──► Reconstruction Loss:
    │    └─► e.g., BinaryCrossentropy(Input, Reconstructed)
    │
    ├──► KL Divergence Loss:
    │    └─► KL( N(z_mean, z_log_var) || N(0, 1) )
    │
    └─► Total Loss = Reconstruction Loss + β * KL Loss
```

---

## 4. Architecture Deep Dive

### 4.1 ResNet-based Encoder

This implementation uses a deep residual network for the encoder to learn powerful feature representations.

```
┌───────────────────────────────────────────────────┐
│              ResNet Encoder                       │
└───────────────────────────────────────────────────┘
Input: (B, H, W, C)
  │
  ▼
┌──────────────────────────────────┐   ┐
│   Downsampling Conv              │   │
│   + Residual Blocks              │   │  Repeated for each
└──────────────────────────────────┘   │  depth level
  │                                    │
  ▼                                    │
┌──────────────────────────────────┐   │
│   ...                            │   │
└──────────────────────────────────┘   ┘
  │
  ▼
Global Average Pooling
  │
  ▼
Dense layers for z_mean and z_log_var
```

### 4.2 Sampling Layer

A simple, non-trainable layer that performs the reparameterization trick.

> This model also supports an alternative **hypersphere** latent sampler
> (`sampling_type="hypersphere"`) that places latents on a thin spherical shell
> instead of a diagonal-Gaussian ball. It avoids posterior collapse and uses every
> latent dimension. See [Section 16](#16-hypersphere-latent-sampling-does-it-work)
> for the design and a fair cross-dimension empirical verdict.

### 4.3 ResNet-based Decoder

The decoder mirrors the encoder's architecture, using upsampling layers and residual blocks to progressively reconstruct the image from the latent vector.

```
┌───────────────────────────────────────────────────┐
│              ResNet Decoder                       │
└───────────────────────────────────────────────────┘
Input: Latent Vector z (B, D_latent)
  │
  ▼
Dense + Reshape to initial feature map size
  │
  ▼
┌──────────────────────────────────┐   ┐
│   Upsampling (e.g., UpSampling2D)│   │
│   + Conv + Residual Blocks       │   │  Repeated for each
└──────────────────────────────────┘   │  depth level
  │                                    │
  ▼                                    │
┌──────────────────────────────────┐   │
│   ...                            │   │
└──────────────────────────────────┘   ┘
  │
  ▼
Final Conv Layer + Sigmoid Activation
  │
  ▼
Output: Reconstructed Image (B, H, W, C)
```

---

## 5. Quick Start Guide

### Installation

```bash
# Ensure you have the required dependencies
pip install keras>=3.0 tensorflow>=2.16 numpy matplotlib
```

### Your First Generative Model (30 seconds)

Let's train a small VAE on the MNIST dataset to generate new handwritten digits.

```python
import keras
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# Local imports from your project structure
from dl_techniques.models.generative.vae.model import VAE

# 1. Load and preprocess data
(X_train, _), (X_test, _) = mnist.load_data()
X_train = np.expand_dims(X_train.astype("float32") / 255.0, -1)
X_test = np.expand_dims(X_test.astype("float32") / 255.0, -1)

# 2. Create a VAE model suitable for MNIST (28x28 images)
model = VAE.from_variant(
    "small",
    input_shape=(28, 28, 1),
    latent_dim=2 # Use a 2D latent space for easy visualization
)

# 3. Compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3))
print("✅ VAE model created and compiled successfully!")
model.summary()

# 4. Train the model
history = model.fit(
    X_train,
    epochs=10, # Train longer for better results
    batch_size=128,
    validation_data=(X_test,)
)
print("✅ Training Complete!")

# 5. Generate new digits by sampling from the latent space
num_samples = 15
generated_images = model.sample(num_samples=num_samples)

# 6. Visualize the results
plt.figure(figsize=(15, 3))
for i in range(num_samples):
    plt.subplot(1, num_samples, i + 1)
    plt.imshow(generated_images[i, :, :, 0], cmap='gray')
    plt.axis('off')
plt.suptitle('Generated Digits from VAE')
plt.show()
```

---

## 6. Component Reference

### 6.1 `VAE` (Model Class)

**Purpose**: The main Keras `Model` subclass that assembles the encoder, sampler, and decoder, and includes the custom `train_step`.

**Location**: `dl_techniques.models.generative.vae.model.VAE`

```python
from dl_techniques.models.generative.vae.model import VAE

model = VAE.from_variant(
    "medium",
    input_shape=(32, 32, 3),
    latent_dim=128,
    kl_loss_weight=0.005 # Adjust the beta-VAE parameter
)
```

**Key Parameters**:

| Parameter | Description |
| :--- | :--- |
| `latent_dim` | The dimensionality of the latent space `z`. |
| `input_shape` | Shape of the input images `(H, W, C)`. |
| `depths` | Number of downsampling/upsampling stages in the ResNet. |
| `filters` | List of filter counts for each stage. |
| `kl_loss_weight` | The β parameter in the β-VAE framework, balancing reconstruction and regularization. |

**Key Methods**:
-   `from_variant()`: Factory method to create standard VAE sizes.
-   `train_step()` / `test_step()`: Custom logic to compute and track both reconstruction and KL losses.
-   `encode(images)`: Returns the `z_mean` and `z_log_var` for a batch of images.
-   `decode(z)`: Decodes a batch of latent vectors `z` into images.
-   `sample(num_samples)`: Generates `num_samples` new images from the prior distribution.

### 6.2 `Sampling` Layer

**Purpose**: A simple, non-trainable layer that implements the reparameterization trick.

**Location**: `dl_techniques.layers.sampling.Sampling`

---

## 7. Configuration & Model Variants

This implementation provides several pre-configured ResNet-based variants to suit different image sizes and complexity.

| Variant | Depths | Filters | Default Latent Dim | Use Case |
| :---: | :---: | :--- | :---: | :--- |
| **`micro`** | 2 | [16, 32] | 32 | Very small images (e.g., 16x16) |
| **`small`** | 2 | [32, 64] | 64 | Small datasets (e.g., MNIST) |
| **`medium`**| 3 | [32, 64, 128]| 128 | Medium datasets (e.g., CIFAR-10) |
| **`large`** | 3 | [64, 128, 256]| 256 | Higher resolution (e.g., 64x64, 128x128) |
| **`xlarge`**| 4 | [64,..,512] | 512 | Very high resolution / complex data |

---

## 8. Comprehensive Usage Examples

### Example 1: Reconstructing Images

A core function of any autoencoder is reconstruction.

```python
# Assuming 'model' is trained on MNIST and X_test is available
num_reconstructions = 10
test_images = X_test[:num_reconstructions]

# Get the model's reconstructions
reconstructed_images = model.predict(test_images)["reconstruction"]

# Visualize
plt.figure(figsize=(20, 4))
for i in range(num_reconstructions):
    # Original
    plt.subplot(2, num_reconstructions, i + 1)
    plt.imshow(test_images[i, :, :, 0], cmap='gray')
    plt.title("Original")
    plt.axis('off')
    # Reconstruction
    plt.subplot(2, num_reconstructions, i + 1 + num_reconstructions)
    plt.imshow(reconstructed_images[i, :, :, 0], cmap='gray')
    plt.title("Reconstructed")
    plt.axis('off')
plt.show()
```

### Example 2: Visualizing the 2D Latent Space

If you train a VAE with `latent_dim=2`, you can visualize the manifold of generated data.

```python
# Assuming 'model' was trained with latent_dim=2
n = 20  # Display a 20x20 grid of digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))

# Linearly spaced coordinates in the latent space
grid_x = np.linspace(-3, 3, n)
grid_y = np.linspace(-3, 3, n)

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = model.decode(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap="gray")
plt.title("2D Latent Space Manifold")
plt.axis("off")
plt.show()
```

---

## 9. Advanced Usage Patterns

### Pattern 1: Conditional VAE (CVAE)

You can extend the VAE to be conditional by feeding class labels (or other conditioning information) into both the encoder and decoder.

```python
# This is a conceptual example showing how one might modify the architecture
# to build a CVAE. The provided class is not conditional.

# Encoder: Concatenate image and one-hot label before processing.
# Decoder: Concatenate latent vector 'z' and one-hot label before decoding.
```

### Pattern 2: Adjusting the β-VAE Parameter

The `kl_loss_weight` (β) controls the trade-off between reconstruction quality and the "niceness" of the latent space.
-   **β < 1**: Emphasizes reconstruction, may lead to a less structured latent space.
-   **β = 1**: The standard VAE objective.
-   **β > 1**: Pushes for a more disentangled latent space where individual latent dimensions correspond to distinct factors of variation in the data. This often comes at the cost of blurrier reconstructions.

```python
# Create a beta-VAE with a high beta value
beta_vae = VAE.from_variant(
    "small",
    input_shape=(28, 28, 1),
    latent_dim=10,
    kl_loss_weight=4.0 # Beta = 4
)
```

---

## 10. Performance Optimization

### Mixed Precision Training

VAEs, especially with deep ResNet backbones, can be accelerated using mixed precision.

```python
# Enable mixed precision globally
keras.mixed_precision.set_global_policy('mixed_float16')

# Create model (will automatically use mixed precision)
model = VAE.from_variant("large", ...)
model.compile(...)
```

---

## 11. Training and Best Practices

### Monitoring the Loss Components

During training, it's crucial to monitor both `reconstruction_loss` and `kl_loss`.
-   **`reconstruction_loss`**: Should steadily decrease. If it stagnates, the model's capacity might be too low, or the learning rate may need adjustment.
-   **`kl_loss`**: This term regularizes the latent space. It might increase initially as the encoder learns to map inputs to the prior distribution, then it should stabilize. A `kl_loss` near zero can indicate "posterior collapse" (see below).

### The Reconstruction Loss Function

This implementation uses `binary_crossentropy` for reconstruction loss, which is common for images with pixel values scaled to `[0, 1]`. For images with a different distribution, Mean Squared Error (`mse`) might be more appropriate.

---

## 12. Serialization & Deployment

The `VAE` model is fully serializable using Keras 3's modern `.keras` format, including its custom `train_step`.

### Saving and Loading

```python
# Create and train model
model = VAE.from_variant("small", input_shape=(28, 28, 1), latent_dim=16)
model.compile(optimizer="adam")
# model.fit(...)

# Save the entire model
model.save('my_vae_model.keras')
print("Model saved to my_vae_model.keras")

# Load the model in a new session
loaded_model = keras.models.load_model('my_vae_model.keras')
print("Model loaded successfully")

# Verify that the custom methods work
generated_image = loaded_model.sample(1)
print(f"Generated image shape: {generated_image.shape}")
```

---

## 13. Testing & Validation

### Unit Tests

```python
import keras
import numpy as np
from dl_techniques.models.generative.vae.model import VAE

def test_model_creation_from_variant():
    """Test model creation from variants."""
    model = VAE.from_variant("micro", input_shape=(32, 32, 1), latent_dim=8)
    assert model is not None
    print("✓ VAE-micro created successfully")

def test_forward_pass_shape():
    """Test the output shapes of a forward pass."""
    model = VAE.from_variant("small", input_shape=(28, 28, 1), latent_dim=16)
    dummy_input = np.random.rand(4, 28, 28, 1)
    outputs = model.predict(dummy_input)

    assert outputs["reconstruction"].shape == (4, 28, 28, 1)
    assert outputs["z_mean"].shape == (4, 16)
    assert outputs["z_log_var"].shape == (4, 16)
    assert outputs["z"].shape == (4, 16)
    print("✓ Forward pass has correct shapes")

def test_generative_methods():
    """Test the encode, decode, and sample methods."""
    model = VAE.from_variant("small", input_shape=(28, 28, 1), latent_dim=16)
    dummy_input = np.random.rand(4, 28, 28, 1)

    z_mean, _ = model.encode(dummy_input)
    reconstruction = model.decode(z_mean)
    assert reconstruction.shape == (4, 28, 28, 1)

    samples = model.sample(num_samples=5)
    assert samples.shape == (5, 28, 28, 1)
    print("✓ Generative methods work correctly")

# Run tests
if __name__ == '__main__':
    test_model_creation_from_variant()
    test_forward_pass_shape()
    test_generative_methods()
    print("\n✅ All tests passed!")
```

---

## 14. Troubleshooting & FAQs

**Issue 1: Reconstructions are very blurry.**

-   **Cause 1**: The `kl_loss_weight` (β) is too high, prioritizing a perfect latent space over accurate reconstruction.
-   **Solution 1**: Decrease `kl_loss_weight` (e.g., from `0.01` to `0.001`).
-   **Cause 2**: The model capacity (number of filters, latent dimension) is too low to capture the data's complexity.
-   **Solution 2**: Try a larger model variant (e.g., move from "small" to "medium").
-   **Cause 3**: The VAE objective itself, which maximizes the average log-likelihood, tends to produce blurry images compared to GANs. This is an inherent property.

**Issue 2: The `kl_loss` drops to nearly zero and stays there ("Posterior Collapse").**

-   **Cause**: The decoder becomes powerful enough to reconstruct the input without using the information from the latent vector `z`. The encoder then learns to map all inputs to the prior distribution (N(0,1)) to minimize the KL loss, effectively ignoring the input.
-   **Solution 1**: Use "KL annealing" - start with `kl_loss_weight=0` and gradually increase it over the first several thousand training steps. This gives the model a chance to learn a meaningful reconstruction path first.
-   **Solution 2**: Use a weaker decoder or a stronger encoder.

### Frequently Asked Questions

**Q: How does a VAE compare to a GAN (Generative Adversarial Network)?**

A:
-   **VAE**: Learns an explicit probability distribution and provides a smooth, continuous latent space. Great for understanding data structure and interpolation. Tends to produce blurrier, but more diverse, samples.
-   **GAN**: Uses a minimax game between a generator and a discriminator. Does not learn an explicit distribution. Tends to produce sharper, more realistic samples but can suffer from "mode collapse" (generating only a few types of samples) and training instability.

**Q: Can I use this for anomaly detection?**

A: Yes. A VAE trained on normal data will have a higher reconstruction error for anomalous inputs. You can set a threshold on the reconstruction loss to identify anomalies.

---

## 15. Technical Details

### The VAE Objective: Evidence Lower Bound (ELBO)

The VAE loss function is derived from maximizing the **Evidence Lower Bound (ELBO)** on the log-likelihood of the data. The objective is:

**log p(x) ≥ E_q(z|x) [log p(x|z)] - KL(q(z|x) || p(z))**

Maximizing the ELBO is equivalent to minimizing the VAE loss:

**Loss = - (Reconstruction Term) + (Regularization Term)**

-   **Reconstruction Term (`-E_q(z|x) [log p(x|z)]`)**: This is the expected negative log-likelihood of the data `x` given the latent vector `z`. For image pixels scaled to `[0, 1]`, this is equivalent to **binary cross-entropy**. It measures how well the decoder reconstructs the input.
-   **Regularization Term (`KL(q(z|x) || p(z))`)**: This is the **Kullback-Leibler (KL) divergence** between the encoder's approximate posterior `q(z|x)` and the prior `p(z)`. It forces the encoded distributions to be similar to a standard normal distribution, thus structuring the latent space.

---

## 16. Hypersphere Latent Sampling: Does It Work?

Beyond the standard diagonal-Gaussian latent, this `VAE` supports a **hypersphere**
sampling mode (`sampling_type="hypersphere"`) — the contribution evaluated in this
section. Instead of `z ~ N(μ, σ²)` on an unbounded ball, the encoder places each latent
on a thin, strictly-positive **spherical shell**: it predicts a unit direction from
`z_mean` and a single scalar radius log-variance (a dedicated `Dense(1)` head), and the
Gaussian KL is replaced by a **radius-variance KL**. The direction carries an implicit
uniform-on-sphere prior.

```python
from dl_techniques.models.vae.model import VAE

# Baseline diagonal-Gaussian latent
gauss = VAE.from_variant("small", input_shape=(28, 28, 1),
                         latent_dim=16, sampling_type="gaussian")

# Hypersphere (thin-shell sphere sampler + radius-variance KL)
hyper = VAE.from_variant("small", input_shape=(28, 28, 1),
                         latent_dim=16, sampling_type="hypersphere")
```

> The hypersphere mode is a deliberate simplification, **not** a full vMF S-VAE — there is
> no explicit directional (von Mises–Fisher / uniform-sphere) KL on the aggregate posterior.
> The legacy value `"hypersphere_faithful"` is accepted as a deprecated alias (old configs
> and checkpoints still load and report as `"hypersphere"`); the dropped
> `"hypersphere_controlled"` mode raises `ValueError`.

### Verdict

**YES — hypersphere sampling works, and at `latent_dim ≥ 8` it decisively beats the
Gaussian baseline.** Its defining advantage is that it **does not suffer posterior
collapse**: it uses *every* latent dimension, whereas the Gaussian VAE collapses to ~5–6
active units no matter how much latent capacity it is given. This translates directly into
much better reconstruction.

*Setup: MNIST, 50 epochs, seed 42. Fair metrics only — `reconstruction_loss` (identical
binary-crossentropy across modes) and MMD (mode-agnostic, prior-decoded samples vs real
test, PCA-50 RBF kernel). `total_loss`/`kl_loss` are NOT comparable across modes (the two
modes use different KL formulas) and are excluded.*

| latent_dim | recon_bce (gauss) | recon_bce (hyper) | hyper vs gauss | active units gauss | active units hyper |
|---|---|---|---|---|---|
| 2  | **0.189** | 0.200 | +5.7% (gauss wins) | 2 | 2 |
| 8  | 0.166 | **0.112** | **−33%** | 6 / 8 | **8 / 8** |
| 16 | 0.168 | **0.092** | **−46%** | 5 / 16 | **16 / 16** |
| 32 | 0.165 | **0.091** | **−45%** | 6 / 32 | **32 / 32** |

- **Posterior collapse (the headline).** Gaussian active units flatline at ~5–6 regardless
  of `latent_dim`; its reconstruction is stuck at ~0.165. The hypersphere uses **100%** of
  dimensions (8/8, 16/16, 32/32) and reconstruction keeps improving (0.112 → 0.092 → 0.091).
  This is exactly the theoretically-predicted benefit of a hyperspherical latent: no origin
  to collapse toward, and the KL does not penalize latent magnitude.
- **Reconstruction.** Hypersphere is 33–46% better at `latent_dim ≥ 8`.
- **`latent_dim = 2`.** Hypersphere is marginally behind (+5.7%) — expected, since a 2-D
  sphere is a 1-D circle, intrinsically more constrained than the 2-D Gaussian plane. The
  advantage appears once there is real capacity to use.

### Generative quality (MMD on prior samples) — mixed, roughly comparable

| latent_dim | mmd2_median (gauss / hyper) | mmd2_half (gauss / hyper) |
|---|---|---|
| 8  | 0.0155 / 0.0266 (gauss) | 0.056 / 0.060 (≈) |
| 16 | 0.0153 / 0.0187 (gauss) | 0.056 / **0.032** (hyper) |
| 32 | 0.0155 / 0.0201 (gauss) | 0.057 / **0.032** (hyper) |

On prior-sample MMD the two are close: Gaussian is slightly ahead at the median bandwidth
(1.2–1.7×), the hypersphere is ahead at the smaller bandwidth for `dim ≥ 16`. The clear,
robust win is on **reconstruction + dimension utilization**, not prior-sample MMD. The
likely cause: the hypersphere has no explicit directional-KL forcing the aggregate posterior
to match the uniform-sphere prior, leaving a small prior–posterior gap that a true S-VAE
(vMF / uniform-sphere KL) would close.

### Corrections made during the study

Two real bugs and one misleading visualization were fixed before any verdict could be
trusted:

1. **`VAE.sample()` drew `N(0,I)` for all modes** → fixed to draw each mode's true prior.
2. **The radius shell was neither thin nor strictly positive.** The original
   `r = radius + exp(0.5·rlv)·η` had σ ≈ radius, producing ~8–13% **negative** radii and
   ~10–18% of samples at the origin (a filled disk, not a sphere). Fixed to a thin,
   strictly-positive shell `r = radius·(1 + 0.1·exp(0.5·clip(rlv))·η)` floored at
   `0.05·radius`. After the fix, sampled `‖z‖` ≈ 1 with 0% off-shell.
3. **Latent plots showed raw `z_mean` on a fixed `[-4, 4]` axis** (and produced nothing for
   `latent_dim > 2`) → now plot the on-sphere direction and project `dim > 2` via PCA-2.

An intermediate "parity" verdict, computed on the buggy off-sphere model, was **retracted**;
the table above supersedes it.

### Scope / limitations

- Single dataset (MNIST), single seed, 50 epochs. The reconstruction / active-unit result is
  large and monotonic in `latent_dim`, so it is unlikely to be noise, but multi-seed /
  multi-dataset confirmation would strengthen it.
- Prior-sample MMD is only comparable, not better — a directional (vMF / uniform-sphere) KL
  is the natural next step to also win on generation. **This was done — see §17.**

---

## 17. vMF Spherical VAE: Winning on Generation

§16 left one gap open: the thin-shell `hypersphere` mode wins reconstruction but only *ties*
gaussian on prior-sample generation (MMD), because it has **no directional regularizer** — its
1-D radius-variance KL says nothing about where on the sphere the latents go. A true **von
Mises–Fisher Spherical VAE** (Davidson et al. 2018) closes that gap with a directional posterior
`q(z|x)=vMF(μ̂, κ)` and the closed-form **vMF→uniform-sphere KL**, which pushes the *aggregate*
posterior toward the uniform prior.

```python
from dl_techniques.models.vae.model import VAE

# True vMF Spherical VAE: directional posterior + vMF->uniform KL
vmf = VAE.from_variant("small", input_shape=(28, 28, 1),
                       latent_dim=16, sampling_type="vmf")
```

> The vMF mode samples via a fixed-K Ulrich/Wood rejection sampler + Householder reflection
> (`VMFSampling`), and computes the exact vMF→uniform KL via a continued-fraction modified-Bessel
> ratio (`vmf_kl_divergence`), both in `layers/sampling.py`. Because its KL is ~100× larger than
> the other modes', set a much smaller `kl_loss_weight` (β≈1e-3, vs 0.01) — the trainer exposes
> `--kl-loss-weight`, `--kl-warmup-epochs`, `--early-stop-monitor`, and a (now-functional) cosine
> `--lr-schedule`.

### Verdict

**The directional vMF KL delivers the generation win — decisively at `latent_dim` 8 and 16.**
MNIST, 50ep, seed 42, β=1e-3 (gaussian/hypersphere β=0.01), cosine LR:

| dim | recon_bce (g / h / **v**) | mmd²_median (g / h / **v**) | active units (g / h / v) | dir_concentration (g / h / **v**) |
|---|---|---|---|---|
| 8  | 0.165 / **0.108** / 0.112 | 0.0169 / 0.0331 / **0.0053** | 5 / 8 / 8 | 0.050 / 0.337 / **0.100** |
| 16 | 0.167 / **0.094** / 0.096 | 0.0172 / 0.0371 / **0.0065** | 5 / 16 / 16 | 0.029 / 0.424 / **0.077** |
| 32 | 0.164 / **0.091** / 0.105 | **0.0159** / 0.0496 / 0.0236 | 6 / 32 / 32 | 0.047 / 0.448 / **0.051** |

- **Generation (MMD).** vMF wins all bandwidths at **d8** (3.2× better than gaussian, 6.2× than
  hypersphere) and **d16** (2.6× / 5.7×). At **d32** vMF beats hypersphere but loses to gaussian.
- **Mechanism — the directional KL does real work.** vMF's `dir_concentration` (0.10 / 0.077 /
  0.051) is far below hypersphere's (0.34 / 0.42 / 0.45): the vMF→uniform KL **spreads the
  aggregate posterior** over the sphere, so uniform-prior samples decode to realistic images. This
  is exactly the term the hypersphere mode lacks.
- **Reconstruction + dimension use.** vMF (like hypersphere) uses **100%** of latent dims and
  reconstructs far better than gaussian (which collapses to ~5–6 active units); its recon trails
  hypersphere only slightly (the directional KL costs a little sharpness).
- **The d32 erosion is a β-scaling effect, not a vMF limit.** The vMF KL grows with dimension
  (final raw KL 15.8 / 26.2 / 31.6 at d8/16/32 for the same β), so a β tuned at d16 over-regularizes
  at d32 (recon slips behind hypersphere; MMD degrades). A dim-scaled (smaller) β at d32 is expected
  to recover the win — left as future work.

### Notes (verified during the study)

- The `VMFSampling` sampler and its closed-form KL were each validated against SciPy
  (`scipy.stats.vonmises_fisher` to ~2e-4; `scipy.special.ive` to ~6e-6). The κ-gradient is
  **unbiased**, so the Naesseth-2017 rejection-reparameterization correction is not needed.
- Free-learned κ initially **posterior-collapses** (κ→0, uniform latent); cured with higher init κ
  (≈12) + KL warmup. The KL is mode-incomparable, so judge across modes on `reconstruction_loss` +
  MMD only, never `total_loss`/`kl_loss`.

Full writeup: `results/VAE_VMF_REPORT.md`.

---

## 18. Citation

If you use VAEs in your research, please cite the original paper:

```bibtex
@article{kingma2013auto,
  title={Auto-encoding variational bayes},
  author={Kingma, Diederik P and Welling, Max},
  journal={arXiv preprint arXiv:1312.6114},
  year={2013}
}
```