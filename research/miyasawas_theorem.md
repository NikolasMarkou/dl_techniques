# Miyasawa's Theorem: A Complete Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Mathematical Statement](#mathematical-statement)
3. [Detailed Mathematical Derivation](#detailed-mathematical-derivation)
4. [Intuitive Understanding](#intuitive-understanding)
5. [Connection to Energy Functions](#connection-to-energy-functions)
6. [Score Matching and Probability Theory](#score-matching-and-probability-theory)
7. [Applications in Modern Machine Learning](#applications-in-modern-machine-learning)
8. [Practical Implementation](#practical-implementation)
9. [Limitations and Caveats](#limitations-and-caveats)
10. [Historical Context](#historical-context)
11. [Further Reading](#further-reading)

---

## Introduction

Miyasawa's theorem (1961) is a fundamental result in statistical estimation that has found profound applications in modern machine learning, particularly in denoising, generative modeling, and score-based methods. The theorem establishes a deep connection between optimal denoising and probability density gradients, providing a mathematical bridge between signal processing and probabilistic modeling.

**Core Insight**: The theorem reveals that every optimal denoiser is secretly computing the gradient of a log-probability density function. This connection has revolutionary implications for generative modeling, energy-based methods, and understanding the implicit priors learned by neural networks.

---

## Mathematical Statement

### Problem Setup

Consider the standard denoising problem:
- **Clean signal**: $x \in \mathbb{R}^n$
- **Additive Gaussian noise**: $\varepsilon \sim \mathcal{N}(0, \sigma^2 I)$
- **Noisy observation**: $y = x + \varepsilon$

### The Theorem

**Miyasawa's Theorem (1961)**: For the least-squares optimal denoiser $\hat{x}(y)$ that minimizes the mean squared error $\mathbb{E}[\|\hat{x}(y) - x\|^2]$, the following identity holds:

$$\boxed{\hat{x}(y) = y + \sigma^2 \nabla_y \log p(y)}$$

where:
- $\hat{x}(y)$ is the optimal denoised estimate
- $p(y)$ is the probability density function of the noisy observations
- $\nabla_y \log p(y)$ is the **score function** (gradient of log-density)
- $\sigma^2$ is the noise variance

### Equivalent Formulations

The theorem can be expressed in several equivalent ways:

**Residual Form**:
$$\hat{x}(y) - y = \sigma^2 \nabla_y \log p(y)$$

**Score Function Form**:
$$\nabla_y \log p(y) = \frac{1}{\sigma^2}(\hat{x}(y) - y)$$

**Energy Function Form** (since $E(y) = -\log p(y)$):
$$\hat{x}(y) - y = -\sigma^2 \nabla_y E(y)$$

---

## Detailed Mathematical Derivation

### Step 1: Optimal Denoiser Definition

The least-squares optimal denoiser is the conditional expectation:
$$\hat{x}(y) = \mathbb{E}[x|y] = \int x \, p(x|y) \, dx$$

### Step 2: Bayes' Rule Application

Using Bayes' rule:
$$p(x|y) = \frac{p(y|x)p(x)}{p(y)}$$

For additive Gaussian noise: $p(y|x) = \frac{1}{(2\pi\sigma^2)^{n/2}} \exp\left(-\frac{\|y-x\|^2}{2\sigma^2}\right)$

### Step 3: Gradient of Observation Density

The key insight is to compute $\nabla_y p(y)$:

$$p(y) = \int p(y|x) p(x) \, dx$$

Taking the gradient:
$$\nabla_y p(y) = \int \nabla_y p(y|x) p(x) \, dx$$

### Step 4: Gradient of Gaussian Likelihood

For the Gaussian likelihood:
$$\nabla_y p(y|x) = p(y|x) \cdot \frac{x - y}{\sigma^2}$$

Therefore:
$$\nabla_y p(y) = \frac{1}{\sigma^2} \int (x - y) p(y|x) p(x) \, dx$$

### Step 5: Simplification Using Bayes' Rule

$$\nabla_y p(y) = \frac{1}{\sigma^2} \int (x - y) p(x|y) p(y) \, dx$$

$$= \frac{p(y)}{\sigma^2} \int (x - y) p(x|y) \, dx$$

$$= \frac{p(y)}{\sigma^2} \left(\int x p(x|y) \, dx - y \int p(x|y) \, dx\right)$$

$$= \frac{p(y)}{\sigma^2} (\hat{x}(y) - y)$$

### Step 6: Final Result

Dividing both sides by $p(y)$:
$$\frac{\nabla_y p(y)}{p(y)} = \nabla_y \log p(y) = \frac{1}{\sigma^2}(\hat{x}(y) - y)$$

Rearranging:
$$\boxed{\hat{x}(y) = y + \sigma^2 \nabla_y \log p(y)}$$

---

## Intuitive Understanding

### Geometric Interpretation

Think of the theorem in terms of **energy landscapes**:

1. **Clean data** lies on a low-dimensional manifold in high-dimensional space
2. **Noise** pushes observations away from this manifold
3. **Optimal denoising** moves observations back toward the manifold
4. **The movement direction** is given by the gradient of log-probability

```
Noisy Point (y) ──────> Clean Estimate (x̂)
      │                        │
      │ Movement Direction     │
      │ = σ²∇log p(y)          │
      └────────────────────────┘
```

### Physical Analogy

Imagine a **particle in a potential field**:
- The **potential energy** is $-\log p(y)$
- **High probability regions** have low potential energy
- The **gradient** points toward regions of higher probability
- **Denoising** is like letting the particle slide downhill to equilibrium

### Information Theory Perspective

- **Score function** $\nabla \log p(y)$ measures the "information gradient"
- It points toward directions of increasing likelihood
- **Optimal denoising** follows this information gradient
- The movement is proportional to noise level ($\sigma^2$)

---

## Connection to Energy Functions

### Energy-Based Modeling

In energy-based models, we define:
$$E(y) = -\log p(y) + \text{constant}$$

Miyasawa's theorem becomes:
$$\hat{x}(y) = y - \sigma^2 \nabla_y E(y)$$

This reveals that **optimal denoising is gradient descent on an energy landscape**!

### Implicit Energy Learning

When we train a denoiser on data:
1. The denoiser implicitly learns the data distribution $p(x)$
2. Through the noise model, it learns $p(y)$
3. Miyasawa's theorem extracts $\nabla E(y)$ from the denoiser
4. We obtain the **energy landscape without explicit energy modeling**

### Connection to Physics

This connects to **Langevin dynamics** in statistical physics:
$$dx = -\nabla E(x) dt + \sqrt{2T} dW$$

where:
- $E(x)$ is the potential energy
- $T$ is temperature
- $dW$ is Brownian motion

Denoising follows a similar gradient flow on the learned energy landscape.

---

## Score Matching and Probability Theory

### Score Function Definition

The **score function** is defined as:
$$s(y) = \nabla_y \log p(y)$$

Key properties:
- Points toward regions of higher probability density
- Independent of normalization constant
- Central to many modern generative models

### Score Matching Objective

**Traditional score matching** minimizes:
$$\mathbb{E}_{p(y)}[\|s_\theta(y) - \nabla_y \log p(y)\|^2]$$

**Miyasawa's insight**: We can estimate the score function through denoising!
$$s(y) \approx \frac{1}{\sigma^2}(\text{denoiser}(y) - y)$$

### Denoising Score Matching

This leads to **denoising score matching**:
1. Add noise to clean data: $y = x + \varepsilon$
2. Train denoiser to predict $x$ from $y$
3. Extract score function from denoiser residuals
4. Use for sampling and generation

---

## Applications in Modern Machine Learning

### 1. Score-Based Generative Models

**Key insight**: If we can estimate $\nabla \log p(y)$, we can generate samples using Langevin dynamics:
$$y_{t+1} = y_t + \epsilon \nabla \log p(y_t) + \sqrt{2\epsilon} z_t$$

where $z_t \sim \mathcal{N}(0,I)$.

**Miyasawa's contribution**: Provides a way to estimate the score function through denoising.

### 2. Diffusion Models

Modern diffusion models use Miyasawa's theorem implicitly:
1. **Forward process**: Add noise at multiple scales
2. **Reverse process**: Learn to denoise at each scale
3. **Generation**: Chain denoising steps together
4. **Theoretical foundation**: Each denoising step follows Miyasawa's formula

### 3. Implicit Prior Extraction

For a trained denoiser $D_\theta(y)$:
$\text{Implicit energy gradient} = \frac{1}{\sigma^2}(D_\theta(y) - y)$

This allows:
- **Sampling** from the implicit prior
- **Inverse problem solving** with learned priors
- **Energy-based optimization** using denoisers

**Keras Implementation Example**:
```python
def extract_implicit_prior_gradients(denoiser_model, data, sigma):
    """
    Extract energy gradients from trained denoiser
    """
    denoised = denoiser_model(data, training=False)
    residual = denoised - data
    energy_gradients = residual / (sigma**2)
    return energy_gradients

# Use for inverse problem solving
def solve_inverse_problem_with_denoiser(denoiser, measurements, measurement_matrix, 
                                      sigma, num_iterations=100):
    """
    Solve inverse problem using denoiser-based prior
    """
    # Initialize estimate
    x = tf.random.normal(measurement_matrix.shape[1:])
    
    for i in range(num_iterations):
        # Data fidelity term
        residual = measurements - tf.linalg.matvec(measurement_matrix, x)
        data_gradient = tf.linalg.matvec(measurement_matrix, residual, transpose_a=True)
        
        # Prior term from denoiser
        prior_gradient = extract_implicit_prior_gradients(denoiser, x[None, ...], sigma)[0]
        
        # Combined update
        x = x + 0.01 * (data_gradient + prior_gradient)
    
    return x
```

### 4. Regularization by Denoising (RED)

RED methods use denoisers as regularizers:
$$\min_x \|Ax - b\|^2 + \lambda \rho(x)$$

where $\rho(x)$ is derived from denoiser properties via Miyasawa's theorem.

---

## Practical Implementation

### Algorithm: Extracting Score Functions from Denoisers

```python
import tensorflow as tf
import numpy as np

def extract_score_function(denoiser, y, sigma):
    """
    Extract score function using Miyasawa's theorem
    
    Args:
        denoiser: Trained Keras denoising model
        y: Noisy observation (tf.Tensor)
        sigma: Noise standard deviation (float)
    
    Returns:
        Estimated score function ∇log p(y)
    """
    denoised = denoiser(y, training=False)
    residual = denoised - y
    score = residual / (sigma**2)
    return score
```

### Algorithm: Sampling Using Denoiser-Based Scores

```python
def sample_with_denoiser(denoiser, sample_shape, sigma, num_steps=1000, step_size=0.01, decay_rate=0.99):
    """
    Generate samples using Langevin dynamics with denoiser-based scores
    
    Args:
        denoiser: Trained Keras denoising model
        sample_shape: Shape of samples to generate (tuple)
        sigma: Initial noise standard deviation
        num_steps: Number of sampling steps
        step_size: Langevin dynamics step size
        decay_rate: Noise decay rate per step
    
    Returns:
        Generated samples
    """
    # Initialize with random noise
    y = tf.random.normal(sample_shape)
    
    for i in range(num_steps):
        # Extract score using Miyasawa's theorem
        score = extract_score_function(denoiser, y, sigma)
        
        # Langevin dynamics step
        noise = tf.random.normal(tf.shape(y))
        y = y + step_size * score + tf.sqrt(2.0 * step_size) * noise
        
        # Optional: gradually reduce noise level
        sigma = sigma * decay_rate
    
    return y
```

### Requirements for Validity

For Miyasawa's theorem to hold exactly:

1. **Least-squares training**: Denoiser must minimize MSE loss
2. **Gaussian noise**: Additive noise must be Gaussian
3. **Optimal denoiser**: Must achieve minimum possible MSE
4. **Bias-free architecture**: Critical for theoretical guarantees (see detailed requirements below)

### Bias-Free Architecture Requirements

For neural networks to satisfy Miyasawa's theorem, they must be **completely bias-free**:

#### 1. Framework-Agnostic Architectural Principles

The core principles for bias-free architectures are **universal across frameworks**:

- **No bias terms**: All layers must exclude additive bias parameters
- **Linear final activation**: No activation function on the output layer
- **Modified normalization**: Batch/layer normalization without centering terms
- **Zero-centered inputs**: Data must have approximately zero mean

*Note: In PyTorch, this corresponds to setting `bias=False` in `nn.Conv2d`/`nn.Linear` layers and `center=False` (or `affine=False`) in `nn.BatchNorm2d`. The mathematical principles remain identical across frameworks.*

#### 2. Linear Final Activation (Identity)

The **final activation must be linear** - meaning **no activation function** at all:

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_gold_standard_bias_free_denoiser(input_shape=(None, None, 3)):
    """
    Gold standard bias-free denoiser implementation
    Satisfies all requirements for Miyasawa's theorem
    """
    inputs = keras.Input(shape=input_shape)
    
    # Feature extraction layers (all bias-free with proper normalization)
    x = layers.Conv2D(64, 3, padding='same', use_bias=False)(inputs)  # No bias
    x = layers.Activation('relu')(x)  # ReLU OK for intermediate layers
    x = BiasFreeBatchNorm()(x)  # Bias-free normalization
    
    x = layers.Conv2D(64, 3, padding='same', use_bias=False)(x)  # No bias
    x = layers.Activation('relu')(x)
    x = BiasFreeBatchNorm()(x)
    
    x = layers.Conv2D(64, 3, padding='same', use_bias=False)(x)  # No bias
    x = layers.Activation('relu')(x)
    x = BiasFreeBatchNorm()(x)
    
    # Final layer: CRITICAL - Linear output, no bias, no activation
    residual = layers.Conv2D(input_shape[-1], 3, padding='same', use_bias=False)(x)
    # ↑ KEY: No activation function here - linear output only
    
    # Residual connection (helps training stability and matches theory)
    outputs = layers.Add()([inputs, residual])  # x̂(y) = y + residual
    
    return keras.Model(inputs, outputs, name='gold_standard_bias_free_denoiser')
```

**Why Linear Final Activation?**
- The residual `denoiser(y) - y = σ²∇ log p(y)` can be positive or negative
- Non-linear activations introduce bias: `E[sigmoid(f(x + ε))] ≠ x`
- Linear preserves unbiased property: `E[f(x + ε)] = x` when properly trained

#### 3. No Bias Terms Anywhere

```python
# All layers must use use_bias=False
layers.Conv2D(..., use_bias=False)
layers.Dense(..., use_bias=False)
layers.Conv1D(..., use_bias=False)
layers.Conv3D(..., use_bias=False)
```

#### 4. Modified Batch Normalization

Standard BatchNormalization has bias terms - use bias-free versions:

```python
class BiasFreeBatchNorm(layers.Layer):
    def __init__(self, **kwargs):
        super(BiasFreeBatchNorm, self).__init__(**kwargs)
        # Only scale parameter, no shift (bias)
        self.bn = layers.BatchNormalization(center=False, scale=True)
        
    def call(self, x, training=None):
        # Normalize but don't add bias term (center=False)
        return self.bn(x, training=training)

# Alternative: Use built-in BatchNormalization with center=False
def bias_free_batch_norm():
    return layers.BatchNormalization(center=False, scale=True)
```

#### 4. Problematic Activations for Final Layer

**ReLU/LeakyReLU**: 
```python
# PROBLEMATIC - clips negative residuals
output = tf.nn.relu(features)  # Can't output negative corrections
output = tf.nn.leaky_relu(features)  # Still clips some negative values
```

**Sigmoid/Tanh**:
```python
# PROBLEMATIC - bounds the output range
output = tf.nn.sigmoid(features)  # Bounded to [0,1]
output = tf.nn.tanh(features)     # Bounded to [-1,1]
```

**Softmax**:
```python
# PROBLEMATIC - normalizes outputs to sum to 1
output = tf.nn.softmax(features, axis=-1)  # Not appropriate for denoising
```

#### 5. Critical Input Normalization Requirements

For bias-free networks to work effectively, **input normalization becomes critically important**. This is not just good practice—it's a **necessary condition** for the network to learn properly.

##### Why Zero-Centered Input is Essential

A standard neural network layer performs: `output = activation(W * x + b)`
A **bias-free** layer can only perform: `output = activation(W * x)`

**The Problem**: Without bias terms, the network has no direct mechanism to handle non-zero mean data. If your input has a consistent offset (like `[0,1]` normalized data with mean ≈ 0.5), the network must waste its weights trying to counteract this constant shift instead of learning data structure.

##### Normalization Options Compared

| Normalization | Is Zero-Centered? | Suitability for Bias-Free Network | Key Issues |
|---------------|-------------------|-----------------------------------|------------|
| **`[0, 1]`** | ❌ No (mean ≈ 0.5) | **Poor - Avoid** | Causes training instability, dead neurons, vanishing gradients |
| **Z-Score** | ✅ Yes (exactly 0) | **Excellent (Theory)** | Requires pre-computing dataset statistics |
| **`[-1, +1]`** | ✅ Yes (≈ 0) | **Excellent (Practice)** | Standard in generative modeling, simple to compute |

##### What Happens with Wrong Normalization (`[0,1]`)

Using `[0,1]` normalization with bias-free networks causes **severe problems**:

**1. Inefficient Weight Learning**
```python
# With [0,1] input (always positive), weights must do double duty:
# - Learn data patterns (intended job)
# - Counteract constant positive offset (forced job)
```

**2. Activation Function Problems**
```python
# ReLU activations
output = tf.nn.relu(W @ x)  # x always positive
# Result: Either always active (no sparsity) or dead neurons

# Sigmoid/Tanh activations  
output = tf.nn.tanh(W @ x)  # Input pushed away from sensitive region
# Result: Saturation → vanishing gradients → training stalls
```

**3. Compounding Through Layers**
- First layer struggles with positive input offset
- Its output is also non-centered
- Second layer faces even worse offset
- Problems cascade and worsen with depth

##### Correct Input Normalization

**Best Practice - `[-1, +1]` Normalization**:
```python
def normalize_images_for_bias_free_network(images):
    """
    Normalize images to [-1, +1] range for bias-free architectures
    
    Args:
        images: Input images in [0, 255] or [0, 1] range
        
    Returns:
        Normalized images in [-1, +1] range
    """
    if tf.reduce_max(images) > 1.0:
        # Assume [0, 255] range
        normalized = (images / 127.5) - 1.0
    else:
        # Assume [0, 1] range  
        normalized = (images * 2.0) - 1.0
    
    return normalized

def denormalize_images(normalized_images):
    """Convert back from [-1, +1] to [0, 1] range"""
    return (normalized_images + 1.0) / 2.0
```

**Alternative - Z-Score Standardization**:
```python
def z_score_normalize_dataset(dataset):
    """
    Z-score normalization for bias-free networks
    Perfect zero-centering but requires dataset statistics
    """
    # Compute dataset statistics
    mean = tf.reduce_mean(dataset, axis=(0, 1, 2), keepdims=True)
    std = tf.math.reduce_std(dataset, axis=(0, 1, 2), keepdims=True)
    
    # Normalize to mean=0, std=1
    normalized = (dataset - mean) / (std + 1e-8)
    
    return normalized, mean, std

def z_score_denormalize(normalized_data, mean, std):
    """Convert back from z-score to original scale"""
    return (normalized_data * std) + mean
```

##### Training Pipeline with Proper Normalization

```python
def create_bias_free_training_pipeline(train_images, val_images, noise_levels):
    """
    Complete training pipeline with proper normalization for bias-free networks
    """
    
    # CRITICAL: Normalize to [-1, +1] for bias-free architecture
    def normalize_batch(images):
        return normalize_images_for_bias_free_network(images)
    
    def add_noise_and_normalize(batch):
        # Normalize FIRST
        clean_batch = normalize_batch(batch)
        
        # Then add noise
        sigma = tf.random.uniform([], minval=min(noise_levels), maxval=max(noise_levels))
        noise = tf.random.normal(tf.shape(clean_batch))
        noisy_batch = clean_batch + sigma * noise
        
        return noisy_batch, clean_batch
    
    # Create normalized datasets
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images)
    train_dataset = train_dataset.shuffle(1000).batch(32).repeat()
    train_dataset = train_dataset.map(add_noise_and_normalize)
    
    val_dataset = tf.data.Dataset.from_tensor_slices(val_images)
    val_dataset = val_dataset.batch(32)
    val_dataset = val_dataset.map(add_noise_and_normalize)
    
    return train_dataset, val_dataset

# Usage example with normalization verification
def verify_normalization(dataset_sample):
    """Verify that normalization is correct for bias-free networks"""
    mean = tf.reduce_mean(dataset_sample)
    std = tf.math.reduce_std(dataset_sample)
    
    print(f"Dataset mean: {mean:.6f} (should be ≈ 0.0)")
    print(f"Dataset std: {std:.6f}")
    print(f"Dataset range: [{tf.reduce_min(dataset_sample):.3f}, {tf.reduce_max(dataset_sample):.3f}]")
    
    if abs(mean) > 0.1:
        print("⚠️  WARNING: Mean is not close to zero! This will hurt bias-free network training.")
    else:
        print("✅ Good: Data is properly zero-centered for bias-free architecture.")
```

##### Summary: Input Normalization Requirements

**For bias-free neural networks (required for Miyasawa's theorem):**

✅ **DO**: Use `[-1, +1]` normalization (recommended) or z-score standardization
✅ **DO**: Verify your data has approximately zero mean
✅ **DO**: Apply normalization before adding training noise

❌ **DON'T**: Use `[0, 1]` normalization - it will cripple training
❌ **DON'T**: Skip normalization verification
❌ **DON'T**: Apply normalization after noise addition

#### 6. Complete Correct Implementation Examples

**WRONG - Multiple Issues**:
```python
# Usage with wrong normalization and architecture
wrong_images = tf.cast(images, tf.float32) / 255.0  # [0,1] range - BAD!
wrong_model = wrong_denoiser()  # Has bias terms and sigmoid - BAD!
```

**CORRECT - Bias-Free with Proper Normalization**:
```python
# Usage with correct normalization and architecture  
correct_images = (tf.cast(images, tf.float32) / 127.5) - 1.0  # [-1,1] range - GOOD!
correct_model = create_gold_standard_bias_free_denoiser()  # Bias-free - GOOD!
```

### Advanced Applications

#### Extracting Implicit Prior Gradients

```python
def extract_implicit_prior_gradients(denoiser_model, data, sigma):
    """
    Extract energy gradients from trained denoiser using Miyasawa's theorem
    
    Args:
        denoiser_model: Trained bias-free denoising model
        data: Input data (tf.Tensor)
        sigma: Noise standard deviation used during training
    
    Returns:
        Energy gradients: ∇E(data) = -(1/σ²)(denoiser(data) - data)
    """
    with tf.GradientTape() as tape:
        tape.watch(data)
        denoised = denoiser_model(data, training=False)
    
    residual = denoised - data
    energy_gradients = residual / (sigma**2)
    return energy_gradients

def solve_inverse_problem_with_denoiser(denoiser, measurements, measurement_matrix, 
                                      sigma, num_iterations=100, step_size=0.01):
    """
    Solve inverse problems using denoiser-based implicit priors
    
    Example: Image inpainting, super-resolution, compressive sensing
    """
    # Initialize estimate
    x = tf.Variable(tf.random.normal(measurement_matrix.shape[1:]))
    
    for i in range(num_iterations):
        with tf.GradientTape() as tape:
            # Data fidelity term
            predicted_measurements = tf.linalg.matvec(measurement_matrix, x)
            data_loss = tf.reduce_sum(tf.square(predicted_measurements - measurements))
        
        # Data gradient
        data_gradient = tape.gradient(data_loss, x)
        
        # Prior gradient from denoiser (Miyasawa's theorem)
        prior_gradient = extract_implicit_prior_gradients(denoiser, x[None, ...], sigma)[0]
        
        # Combined gradient descent step
        x.assign_sub(step_size * (data_gradient - prior_gradient))
    
    return x.numpy()
```
```python
# WRONG - introduces bias
class BiasedDenoisier(keras.Model):
    def __init__(self):
        super().__init__()
        self.backbone = self.build_backbone()
        self.final_layer = layers.Conv2D(3, 3, padding='same')
        
    def call(self, x):
        features = self.backbone(x)
        output = tf.nn.sigmoid(self.final_layer(features))  # BAD
        return output

# CORRECT - bias-free with linear final activation
class BiasFreeDenoisier(keras.Model):
    def __init__(self):
        super().__init__()
        self.backbone = self.build_backbone()
        self.final_layer = layers.Conv2D(3, 3, padding='same', use_bias=False)
        
    def call(self, x):
        features = self.backbone(x)
        residual = self.final_layer(features)  # Linear output
        return x + residual  # Residual connection

# Complete working example with residual learning
def create_complete_bias_free_denoiser(input_shape=(None, None, 3)):
    inputs = keras.Input(shape=input_shape)
    
    # Feature extraction layers (all bias-free)
    x = layers.Conv2D(64, 3, padding='same', use_bias=False)(inputs)
    x = layers.Activation('relu')(x)
    x = BiasFreeBatchNorm()(x)
    
    x = layers.Conv2D(64, 3, padding='same', use_bias=False)(x)
    x = layers.Activation('relu')(x)
    x = BiasFreeBatchNorm()(x)
    
    x = layers.Conv2D(64, 3, padding='same', use_bias=False)(x)
    x = layers.Activation('relu')(x)
    x = BiasFreeBatchNorm()(x)
    
    # Final layer: LINEAR output, no bias
    residual = layers.Conv2D(input_shape[-1], 3, padding='same', use_bias=False)(x)
    
    # Residual connection (helps training stability)
    outputs = layers.Add()([inputs, residual])
    
    return keras.Model(inputs, outputs, name='complete_bias_free_denoiser')

# Training setup for bias-free denoiser
def compile_bias_free_denoiser(model):
    model.compile(
        optimizer='adam',
        loss='mse',  # Critical: MSE loss for least-squares optimality
        metrics=['mae']
    )
    return model

### Complete Training Pipeline

```python
def train_bias_free_denoiser(train_images, val_images, noise_levels=[0.05, 0.1, 0.15, 0.2, 0.25]):
    """
    Complete training pipeline for bias-free denoiser
    """
    
    # 1. Create and compile model
    model = create_gold_standard_bias_free_denoiser(input_shape=(256, 256, 3))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])  # MSE critical for optimality
    
    # 2. Create training pipeline with proper normalization
    train_dataset, val_dataset = create_bias_free_training_pipeline(train_images, val_images, noise_levels)
    
    # 3. Training callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
        keras.callbacks.ModelCheckpoint('bias_free_denoiser.h5', save_best_only=True)
    ]
    
    # 4. Train model
    history = model.fit(
        train_dataset,
        steps_per_epoch=len(train_images) // 32,
        validation_data=val_dataset,
        validation_steps=len(val_images) // 32,
        epochs=100,
        callbacks=callbacks
    )
    
    # 5. Verify model satisfies Miyasawa's theorem requirements
    verify_bias_free_properties(model)
    
    return model, history

def verify_bias_free_properties(model):
    """Verify that model satisfies bias-free requirements"""
    print("🔍 Verifying bias-free properties...")
    
    # Check for bias terms
    has_bias = any(layer.use_bias for layer in model.layers if hasattr(layer, 'use_bias'))
    print(f"✅ No bias terms: {not has_bias}")
    
    # Check final layer activation
    final_layer = model.layers[-2]  # Before Add layer
    has_activation = hasattr(final_layer, 'activation') and final_layer.activation != 'linear'
    print(f"✅ Linear final activation: {not has_activation}")
    
    # Test with normalized input
    test_input = tf.random.normal((1, 64, 64, 3))  # Should be in [-1,1] range
    mean_val = tf.reduce_mean(test_input)
    print(f"✅ Zero-centered input (mean ≈ 0): {abs(mean_val) < 0.1}")
    
    if not has_bias and not has_activation and abs(mean_val) < 0.1:
        print("🎉 Model satisfies all Miyasawa's theorem requirements!")
    else:
        print("⚠️  Model may not fully satisfy theoretical requirements.")

# Example usage
if __name__ == "__main__":
    # Load your properly normalized data here
    # train_images = load_and_normalize_training_data()  # Should be in [-1,1]
    # val_images = load_and_normalize_validation_data()   # Should be in [-1,1]
    
    # Train the bias-free denoiser
    # model, history = train_bias_free_denoiser(train_images, val_images)
    
    # Use for score-based sampling
    # samples = sample_with_denoiser(model, (10, 256, 256, 3), sigma=0.1)
    pass
```

---

## Limitations and Caveats

### 1. Optimality Requirements

**Challenge**: Real denoisers are not optimal
- Neural networks approximate the optimal denoiser
- Finite training data and model capacity limit optimality
- **Implication**: Miyasawa's formula holds approximately

### 2. Symmetric Jacobian Assumption

**Problem**: Many practical denoisers don't have symmetric Jacobians

For the denoiser's residual to be a true gradient of a scalar energy function, its Jacobian matrix must be symmetric. This means that for any denoiser $D(y)$, we need:

$\frac{\partial D_i}{\partial y_j} = \frac{\partial D_j}{\partial y_i}$

**Practical implications**:
- Standard convolutional networks do not guarantee this property
- BM3D, Non-local means, DnCNN often fail this requirement
- Architectures with tied or shared weights (like certain autoencoders) can satisfy this property
- **However**, many models work well even if this condition is only approximately met

**Consequence**: Some theoretical guarantees may not hold exactly, but the framework often remains practically useful as an approximation.

### 3. Noise Model Assumptions

**Limitation**: Theorem assumes additive Gaussian noise
- Real-world noise may be non-Gaussian or signal-dependent
- **Workaround**: Use Gaussian approximations or extend theory

### 4. High-Dimensional Challenges

**Issues**:
- Curse of dimensionality affects score estimation
- Sparse data in high dimensions
- Manifold structure may not be well-captured

### 5. Training Stability

**Practical concerns**:
- Score function can have large magnitude
- Numerical instabilities in high dimensions
- Requires careful hyperparameter tuning

---

## Historical Context

### Origins and Development

- **1961**: Miyasawa proves the original theorem
- **1980s-1990s**: Score matching theory develops
- **2005**: Hyvärinen formalizes score matching
- **2011**: Vincent introduces denoising score matching
- **2019**: Song & Ermon connect to generative modeling
- **2020-2021**: Kadkhodaie & Simoncelli popularize in vision community

### Modern Revival

The theorem gained renewed attention due to:
1. **Diffusion models** success in generation
2. **Score-based methods** for inverse problems
3. **Energy-based modeling** renaissance
4. **Implicit prior** understanding in deep learning

### Key Contributors

- **Miyasawa (1961)**: Original theorem
- **Stein (1981)**: Related unbiased risk estimation
- **Efron (2011)**: Tweedie's formula connections
- **Vincent (2011)**: Denoising score matching
- **Song et al. (2019-2021)**: Score-based generative models
- **Kadkhodaie & Simoncelli (2021)**: Modern applications

---

## Further Reading

### Foundational Papers

1. **Miyasawa (1961)**: "On the convergence of the iteration of expectations and quadratic forms" *(original source - difficult to find)*

2. **Kadkhodaie & Simoncelli (2021)**: ["Solving Linear Inverse Problems Using the Prior Implicit in a Denoiser"](https://arxiv.org/abs/2007.13640) *(modern treatment)*

3. **Vincent (2011)**: "A connection between score matching and denoising autoencoders" *(denoising score matching)*

### Related Theoretical Work

4. **Efron (2011)**: "Tweedie's formula and selection bias" *(Tweedie's formula connections)*

5. **Song & Ermon (2019)**: "Generative Modeling by Estimating Gradients of the Data Distribution" *(score-based generative models)*

6. **Romano et al. (2017)**: "The Little Engine that Could: Regularization by Denoising" *(RED framework)*

### Modern Applications

7. **Ho et al. (2020)**: "Denoising Diffusion Probabilistic Models" *(DDPM)*

8. **Song et al. (2021)**: "Score-Based Generative Modeling through Stochastic Differential Equations" *(SDE framework)*

9. **Dhariwal & Nichol (2021)**: "Diffusion Models Beat GANs on Image Synthesis" *(practical applications)*

### Implementation Resources

10. **Kadkhodaie & Simoncelli GitHub**: [Universal Inverse Problem](https://github.com/LabForComputationalVision/universal_inverse_problem) *(reference implementation)*

---

## Summary

Miyasawa's theorem provides a fundamental bridge between:
- **Signal processing** (denoising) and **probability theory** (score functions)
- **Energy-based modeling** and **learned representations**
- **Classical statistics** and **modern deep learning**

The theorem's power lies not just in its mathematical elegance, but in its practical implications for understanding and leveraging the implicit knowledge embedded in trained neural networks. As machine learning continues to evolve, Miyasawa's theorem remains a cornerstone for connecting optimization, probability, and learning in high-dimensional spaces.

**Key Takeaway**: Every optimal denoiser is secretly computing probability gradients, and this insight unlocks powerful connections between seemingly disparate areas of machine learning and statistics.