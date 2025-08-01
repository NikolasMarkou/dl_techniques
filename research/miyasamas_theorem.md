# Miyasawa's Theorem: A Complete Mathematical Guide

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
      │ = σ²∇log p(y)         │
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
$$\text{Implicit energy gradient} = \frac{1}{\sigma^2}(D_\theta(y) - y)$$

This allows:
- **Sampling** from the implicit prior
- **Inverse problem solving** with learned priors
- **Energy-based optimization** using denoisers

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

#### 1. Linear Final Activation (Identity)

The **final activation must be linear** - meaning **no activation function** at all:

```python
class BiasFreeDenoisier(nn.Module):
    def __init__(self):
        super().__init__()
        # All intermediate layers without bias
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        # Final layer: NO activation, NO bias
        self.final_conv = nn.Conv2d(64, 3, 3, padding=1, bias=False)
        
    def forward(self, x):
        # Intermediate processing with activations
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        
        # Final output: LINEAR (no activation)
        output = self.final_conv(h)  # ← KEY: No activation here
        return output
```

**Why Linear Final Activation?**
- The residual `denoiser(y) - y = σ²∇ log p(y)` can be positive or negative
- Non-linear activations introduce bias: `E[sigmoid(f(x + ε))] ≠ x`
- Linear preserves unbiased property: `E[f(x + ε)] = x` when properly trained

#### 2. No Bias Terms Anywhere

```python
# All layers must use bias=False
nn.Conv2d(..., bias=False)
nn.Linear(..., bias=False)
```

#### 3. Modified Batch Normalization

Standard BatchNorm has bias terms - use bias-free versions:

```python
class BiasFreeBatchNorm(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        # Only scale parameter, no shift (bias)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        
    def forward(self, x):
        # Normalize but don't add bias term
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, None)
```

#### 4. Problematic Activations for Final Layer

**ReLU/LeakyReLU**: 
```python
# PROBLEMATIC - clips negative residuals
output = F.relu(features)  # Can't output negative corrections
```

**Sigmoid/Tanh**:
```python
# PROBLEMATIC - bounds the output range
output = torch.sigmoid(features)  # Bounded to [0,1]
output = torch.tanh(features)     # Bounded to [-1,1]
```

**Correct Implementation**:
```python
# CORRECT - bias-free with linear final activation
class BiasFreeDenoisier(nn.Module):
    def forward(self, x):
        features = self.backbone(x)
        residual = self.final_layer(features)  # Linear output
        return x + residual  # Residual connection
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
- BM3D, Non-local means, DnCNN often fail this requirement
- **Consequence**: Some theoretical guarantees may not hold
- **Solution**: Use specialized architectures or accept approximations

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