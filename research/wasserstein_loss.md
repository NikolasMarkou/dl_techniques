# Understanding Wasserstein Loss: Complete Theory, Implementation, and Applications Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Mathematical Foundation](#mathematical-foundation)
3. [How Wasserstein Loss Works](#how-wasserstein-loss-works)
4. [Wasserstein GANs (WGANs)](#wasserstein-gans-wgans)
5. [Implementation Details](#implementation-details)
6. [Basic Usage Examples](#basic-usage-examples)
7. [Advanced Implementation Patterns](#advanced-implementation-patterns)
8. [Beyond GANs: Other Applications](#beyond-gans-other-applications)
9. [Best Practices and Guidelines](#best-practices-and-guidelines)
10. [Research Insights and Advances](#research-insights-and-advances)
11. [Troubleshooting and Performance](#troubleshooting-and-performance)
12. [API Reference](#api-reference)
13. [Conclusion](#conclusion)

## Introduction

The Wasserstein loss, also known as the Earth Mover's Distance (EMD), represents a fundamental shift in how we measure the distance between probability distributions. Unlike traditional divergences (KL, JS), Wasserstein distance provides meaningful gradients even when distributions have non-overlapping support, making it particularly valuable for training generative models and solving distribution matching problems.

This comprehensive guide covers both the theoretical foundations and practical implementation details, providing everything needed to understand and effectively use Wasserstein loss in modern machine learning applications.

## Mathematical Foundation

### The Wasserstein Distance

The Wasserstein-1 distance (W₁) between two probability distributions P and Q is defined as:

$$W_1(P, Q) = \inf_{\gamma \in \Pi(P, Q)} \mathbb{E}_{(x,y) \sim \gamma}[||x - y||]$$

Where:
- $\Pi(P, Q)$ is the set of all joint distributions with marginals P and Q
- $\gamma$ represents a "transport plan" that moves probability mass from P to Q
- The infimum finds the most efficient transport plan

### Kantorovich-Rubinstein Duality

By the Kantorovich-Rubinstein theorem, we can express W₁ as:

$$W_1(P, Q) = \sup_{||f||_L \leq 1} \mathbb{E}_{x \sim P}[f(x)] - \mathbb{E}_{x \sim Q}[f(x)]$$

Where $f$ is any 1-Lipschitz function (i.e., $|f(x) - f(y)| \leq ||x - y||$ for all x, y).

This dual formulation is crucial because:
1. It transforms an intractable optimization problem into a tractable one
2. It allows us to approximate the Wasserstein distance using neural networks
3. The 1-Lipschitz constraint can be enforced through various techniques

### WGAN Loss Formulations

**Critic Loss:**
$$L_{critic} = \mathbb{E}_{x \sim P_{fake}}[D(x)] - \mathbb{E}_{x \sim P_{real}}[D(x)]$$

**Generator Loss:**
$$L_{generator} = -\mathbb{E}_{x \sim P_{fake}}[D(x)]$$

**WGAN-GP Gradient Penalty:**
$$L_{GP} = \lambda \mathbb{E}_{\hat{x} \sim P_{\hat{x}}}[(||\nabla_{\hat{x}} D(\hat{x})||_2 - 1)^2]$$

Where $\hat{x} = \epsilon x + (1-\epsilon)G(z)$ with $\epsilon \sim U[0,1]$

## How Wasserstein Loss Works

### The Core Mechanism

The Wasserstein loss operates on a fundamentally different principle than traditional losses:

1. **Transport Perspective**: Instead of comparing probability densities point-wise, it considers the minimum cost to transform one distribution into another.

2. **Geometric Intuition**: Think of it as the minimum amount of "work" needed to move earth from one pile (distribution) to match another pile's shape.

3. **Gradient Properties**: Provides meaningful gradients everywhere, not just where distributions overlap.

### Why It's Superior for Distribution Matching

Traditional divergences like KL divergence have several problems:

$$KL(P||Q) = \mathbb{E}_{x \sim P}\left[\log \frac{P(x)}{Q(x)}\right]$$

- **Mode Collapse**: KL divergence can be minimized by covering only a few modes
- **Vanishing Gradients**: When P and Q don't overlap, gradients vanish
- **Asymmetry**: KL(P||Q) ≠ KL(Q||P)

Wasserstein distance addresses these issues:
- **Symmetric**: W(P,Q) = W(Q,P)
- **Non-vanishing Gradients**: Always provides meaningful gradients
- **Mode Coverage**: Encourages covering all modes of the target distribution

## Wasserstein GANs (WGANs)

### Standard GAN vs WGAN

**Standard GAN Objective:**
$$\min_G \max_D V(D,G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1-D(G(z)))]$$

**WGAN Objective:**
$$\min_G \max_{D \in \mathcal{F}} \mathbb{E}_{x \sim p_{data}}[D(x)] - \mathbb{E}_{z \sim p_z}[D(G(z))]$$

Where $\mathcal{F}$ is the set of 1-Lipschitz functions.

### Key Differences

1. **Critic vs Discriminator**: 
   - Standard GANs: Discriminator outputs probabilities (0-1)
   - WGANs: Critic outputs real values (unbounded)

2. **Loss Functions**:
   - **Critic Loss**: $L_C = \mathbb{E}[D(x_{fake})] - \mathbb{E}[D(x_{real})]$
   - **Generator Loss**: $L_G = -\mathbb{E}[D(G(z))]$

3. **No Sigmoid**: Critic doesn't use sigmoid activation in the final layer

### Enforcing the Lipschitz Constraint

**Method 1: Weight Clipping (Original WGAN)**
```python
# After each critic update
for param in critic.parameters():
    param.data.clamp_(-0.01, 0.01)
```

**Method 2: Gradient Penalty (WGAN-GP)**
$$L_{GP} = \lambda \mathbb{E}_{\hat{x} \sim p_{\hat{x}}}[(||\nabla_{\hat{x}} D(\hat{x})||_2 - 1)^2]$$

Where $\hat{x} = \epsilon x + (1-\epsilon)G(z)$ with $\epsilon \sim U[0,1]$

## Implementation Details

### Loss Computation in Practice

**Critic Training Step:**
```python
# For a batch with mixed real and fake samples
def critic_loss(y_true, y_pred):
    # y_true: 1 for real, 0 for fake
    # y_pred: critic outputs
    
    real_loss = y_true * y_pred        # D(real) where y_true=1
    fake_loss = (1 - y_true) * y_pred  # D(fake) where y_true=0
    
    # Wasserstein critic loss: E[D(fake)] - E[D(real)]
    return fake_loss - real_loss
```

**Generator Training Step:**
```python
def generator_loss(y_true, y_pred):
    # For generator, we want to maximize D(fake)
    # So we minimize -D(fake)
    return -y_pred
```

### Training Dynamics

1. **Critic Updates**: Train critic multiple times (typically 5:1 ratio) per generator update
2. **Learning Rates**: Use lower learning rates (e.g., 5e-5) compared to standard GANs
3. **Optimizers**: RMSprop often works better than Adam for WGANs
4. **Batch Normalization**: Avoid in critic as it can interfere with Lipschitz constraint

## Basic Usage Examples

### Simple WGAN Training

```python
import keras
from dl_techniques.losses.wasserstein_loss import WassersteinLoss

# Create loss functions
critic_loss = WassersteinLoss(for_critic=True)
generator_loss = WassersteinLoss(for_critic=False)

# Compile models
critic.compile(optimizer='rmsprop', loss=critic_loss)
generator.compile(optimizer='rmsprop', loss=generator_loss)

# Training labels
real_labels = tf.ones((batch_size, 1))  # 1 for real samples
fake_labels = tf.zeros((batch_size, 1))  # 0 for fake samples

# Train critic
critic_loss_real = critic.train_on_batch(real_images, real_labels)
critic_loss_fake = critic.train_on_batch(fake_images, fake_labels)

# Train generator
generator_loss = generator.train_on_batch(noise, fake_labels)
```

### WGAN-GP Training

```python
from dl_techniques.losses.wasserstein_loss import (
    WassersteinGradientPenaltyLoss,
    compute_gradient_penalty
)

# Create WGAN-GP losses
critic_loss = WassersteinGradientPenaltyLoss(for_critic=True, lambda_gp=10.0)
generator_loss = WassersteinGradientPenaltyLoss(for_critic=False, lambda_gp=10.0)

# Custom training step with gradient penalty
@tf.function
def train_critic_step(real_images, fake_images):
    batch_size = tf.shape(real_images)[0]
    
    # Create labels
    real_labels = tf.ones((batch_size, 1))
    fake_labels = tf.zeros((batch_size, 1))
    
    # Combine data
    combined_images = tf.concat([real_images, fake_images], axis=0)
    combined_labels = tf.concat([real_labels, fake_labels], axis=0)
    
    with tf.GradientTape() as tape:
        # Get critic predictions
        predictions = critic(combined_images, training=True)
        
        # Compute Wasserstein loss
        w_loss = critic_loss(combined_labels, predictions)
        w_loss = tf.reduce_mean(w_loss)
        
        # Add gradient penalty
        gp_loss = compute_gradient_penalty(critic, real_images, fake_images, lambda_gp=10.0)
        total_loss = w_loss + gp_loss
    
    # Update critic
    gradients = tape.gradient(total_loss, critic.trainable_variables)
    optimizer.apply_gradients(zip(gradients, critic.trainable_variables))
    
    return total_loss
```

### Distribution Matching

```python
from dl_techniques.losses.wasserstein_loss import WassersteinDivergence

# For distribution matching tasks
wd_loss = WassersteinDivergence(smooth_eps=1e-7)

# Example: matching output distributions
def distribution_matching_loss(y_true, y_pred):
    """Match probability distributions using Wasserstein divergence."""
    # Ensure inputs are normalized probability distributions
    y_true = tf.nn.softmax(y_true, axis=-1)
    y_pred = tf.nn.softmax(y_pred, axis=-1)
    
    return wd_loss(y_true, y_pred)

# Use in model compilation
model.compile(optimizer='adam', loss=distribution_matching_loss)
```

## Advanced Implementation Patterns

### Complete WGAN Trainer Class

```python
class WGANTrainer:
    def __init__(self, generator, critic, latent_dim=100, use_gp=True):
        self.generator = generator
        self.critic = critic
        self.latent_dim = latent_dim
        self.use_gp = use_gp
        
        # Create losses
        if use_gp:
            self.critic_loss_fn = WassersteinGradientPenaltyLoss(for_critic=True, lambda_gp=10.0)
            self.gen_loss_fn = WassersteinGradientPenaltyLoss(for_critic=False, lambda_gp=10.0)
        else:
            self.critic_loss_fn = WassersteinLoss(for_critic=True)
            self.gen_loss_fn = WassersteinLoss(for_critic=False)
        
        # Optimizers
        self.critic_optimizer = keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5)
        self.gen_optimizer = keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5)
    
    @tf.function
    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        
        # Train critic multiple times
        for _ in range(5):
            # Generate fake images
            noise = tf.random.normal([batch_size, self.latent_dim])
            fake_images = self.generator(noise, training=False)
            
            # Train critic
            critic_loss = self.train_critic_step(real_images, fake_images)
        
        # Train generator
        gen_loss = self.train_generator_step(batch_size)
        
        return {"critic_loss": critic_loss, "generator_loss": gen_loss}
    
    def train_critic_step(self, real_images, fake_images):
        batch_size = tf.shape(real_images)[0]
        
        # Create labels
        real_labels = tf.ones((batch_size, 1))
        fake_labels = tf.zeros((batch_size, 1))
        
        # Combine data
        combined_images = tf.concat([real_images, fake_images], axis=0)
        combined_labels = tf.concat([real_labels, fake_labels], axis=0)
        
        with tf.GradientTape() as tape:
            # Get predictions
            predictions = self.critic(combined_images, training=True)
            
            # Compute loss
            loss = self.critic_loss_fn(combined_labels, predictions)
            loss = tf.reduce_mean(loss)
            
            # Add gradient penalty for WGAN-GP
            if self.use_gp:
                gp_loss = compute_gradient_penalty(
                    self.critic, real_images, fake_images, lambda_gp=10.0
                )
                loss += gp_loss
        
        # Update weights
        gradients = tape.gradient(loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(gradients, self.critic.trainable_variables))
        
        # Apply weight clipping for basic WGAN
        if not self.use_gp:
            for layer in self.critic.layers:
                if hasattr(layer, 'kernel'):
                    layer.kernel.assign(tf.clip_by_value(layer.kernel, -0.01, 0.01))
        
        return loss
    
    def train_generator_step(self, batch_size):
        noise = tf.random.normal([batch_size, self.latent_dim])
        fake_labels = tf.zeros((batch_size, 1))
        
        with tf.GradientTape() as tape:
            fake_images = self.generator(noise, training=True)
            predictions = self.critic(fake_images, training=False)
            
            loss = self.gen_loss_fn(fake_labels, predictions)
            loss = tf.reduce_mean(loss)
        
        gradients = tape.gradient(loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))
        
        return loss
```

### Utility Functions

```python
from dl_techniques.losses.wasserstein_loss import create_wgan_losses, create_wgan_gp_losses

# Create standard WGAN losses
critic_loss, generator_loss = create_wgan_losses()

# Create WGAN-GP losses
critic_loss_gp, generator_loss_gp = create_wgan_gp_losses(lambda_gp=10.0)

# Compute gradient penalty manually
gp_loss = compute_gradient_penalty(
    critic_model,
    real_samples,
    fake_samples,
    lambda_gp=10.0
)
```

## Beyond GANs: Other Applications

### 1. Domain Adaptation

Wasserstein distance is excellent for aligning distributions across domains:

$$L_{domain} = W_1(P_{source}, P_{target}) + L_{task}$$

```python
# Domain adaptation example
domain_loss = WassersteinLoss(for_critic=True)
domain_classifier.compile(optimizer='adam', loss=domain_loss)
```

**Applications:**
- Transfer learning between datasets
- Style transfer in computer vision
- Cross-lingual NLP models

### 2. Regularization in Deep Learning

**Wasserstein Regularization:**
$$L_{total} = L_{task} + \lambda W_1(P_{learned}, P_{prior})$$

This encourages learned representations to match desired prior distributions.

### 3. Optimal Transport Problems

**Applications:**
- Image registration and morphing
- Color transfer between images
- Point cloud alignment
- Flow cytometry data analysis

```python
# Style transfer using Wasserstein divergence
style_loss = WassersteinDivergence()

def style_transfer_loss(content_features, style_features):
    return style_loss(content_features, style_features)
```

### 4. Variational Autoencoders (WAE)

Wasserstein Autoencoders replace KL divergence with Wasserstein distance:

$$L_{WAE} = \mathbb{E}[||x - \text{Dec}(\text{Enc}(x))||^2] + \lambda W_1(P_Z, Q_Z)$$

Where $P_Z$ is the prior and $Q_Z$ is the encoded distribution.

### 5. Reinforcement Learning

**Distributional RL**: Use Wasserstein distance to match return distributions:

$$L_{distributional} = W_1(\text{Target Distribution}, \text{Predicted Distribution})$$

### 6. Time Series Analysis

**Applications:**
- Comparing time series distributions
- Detecting distributional shifts
- Anomaly detection in temporal data

## Best Practices and Guidelines

### 1. Learning Rates

Use lower learning rates for both generator and critic:

```python
# Good optimizer settings
critic_optimizer = keras.optimizers.RMSprop(learning_rate=5e-5)
generator_optimizer = keras.optimizers.RMSprop(learning_rate=5e-5)

# Or for WGAN-GP
critic_optimizer = keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5)
generator_optimizer = keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5)
```

### 2. Training Ratio

Train critic multiple times per generator update:

```python
n_critic = 5  # Number of critic updates per generator update

for epoch in range(epochs):
    for batch in dataset:
        # Train critic n_critic times
        for _ in range(n_critic):
            train_critic_step(batch)
        
        # Train generator once
        train_generator_step(batch_size)
```

### 3. Weight Clipping (Basic WGAN)

```python
def clip_weights(model, clip_value=0.01):
    """Clip weights to maintain Lipschitz constraint."""
    for layer in model.layers:
        if hasattr(layer, 'kernel'):
            layer.kernel.assign(tf.clip_by_value(layer.kernel, -clip_value, clip_value))
        if hasattr(layer, 'bias') and layer.bias is not None:
            layer.bias.assign(tf.clip_by_value(layer.bias, -clip_value, clip_value))
```

### 4. Gradient Penalty (WGAN-GP)

- Use λ = 10.0 for gradient penalty coefficient
- Ensure interpolated samples are on the manifold between real and fake data

```python
# Good gradient penalty setup
lambda_gp = 10.0
gp_loss = compute_gradient_penalty(critic, real_batch, fake_batch, lambda_gp)
```

### 5. Model Architecture

- Remove batch normalization from critic
- Use spectral normalization if available
- Avoid sigmoid activation in the critic output

```python
# Good critic architecture
def create_critic():
    return keras.Sequential([
        keras.layers.Conv2D(64, 4, strides=2, padding='same'),
        keras.layers.LeakyReLU(0.2),
        keras.layers.Conv2D(128, 4, strides=2, padding='same'),
        keras.layers.LeakyReLU(0.2),
        keras.layers.Flatten(),
        keras.layers.Dense(1)  # No activation!
    ])
```

## Research Insights and Advances

### Theoretical Contributions

**1. Convergence Guarantees (Arjovsky et al., 2017)**
- WGAN loss correlates with sample quality
- Provides convergence guarantees under optimal critic assumption
- Eliminates mode collapse in theory

**2. Improved Training Stability (Gulrajani et al., 2017)**
- Gradient penalty provides better Lipschitz constraint enforcement
- More stable training than weight clipping
- Better gradient flow properties

**3. Spectral Normalization (Miyato et al., 2018)**
- Alternative method to enforce Lipschitz constraint
- Controls spectral norm of weight matrices
- Computationally efficient

### Empirical Findings

**1. Training Dynamics**
- WGANs require more critic updates but are more stable
- Loss values are meaningful (correlate with visual quality)
- Less sensitive to hyperparameter choices

**2. Mode Collapse Reduction**
- Significant reduction in mode collapse compared to standard GANs
- Better coverage of data distribution
- More diverse generated samples

**3. Evaluation Metrics**
- Wasserstein distance itself serves as a meaningful evaluation metric
- Better correlation with human perception than IS or FID in some cases

### Recent Advances

**1. Sliced Wasserstein Distance**
- Approximates Wasserstein distance using 1D projections
- Computationally efficient for high dimensions
- Formula: $SW(P,Q) = \int W_1(P_\theta, Q_\theta) d\theta$

**2. Sinkhorn Divergences**
- Entropy-regularized optimal transport
- Differentiable approximation to Wasserstein distance
- Faster computation through Sinkhorn iterations

**3. Neural Optimal Transport**
- Learn optimal transport maps directly using neural networks
- Applications in generative modeling and domain adaptation
- Continuous normalizing flows

## Troubleshooting and Performance

### Monitoring Training

Track these key metrics during training:

```python
# Track these metrics during training
wasserstein_distance = tf.reduce_mean(real_scores) - tf.reduce_mean(fake_scores)
critic_loss = tf.reduce_mean(critic_loss_values)
generator_loss = tf.reduce_mean(generator_loss_values)

# Log metrics
print(f"Wasserstein Distance: {wasserstein_distance:.4f}")
print(f"Critic Loss: {critic_loss:.4f}")
print(f"Generator Loss: {generator_loss:.4f}")
```

### Common Issues and Solutions

**Problem**: Training instability
- **Solution**: Reduce learning rates, increase critic training ratio

**Problem**: Mode collapse
- **Solution**: Use WGAN-GP instead of basic WGAN, adjust λ parameter

**Problem**: Gradient explosion
- **Solution**: Check gradient penalty implementation, ensure proper weight clipping

**Problem**: Poor sample quality
- **Solution**: Increase model capacity, adjust architecture, tune hyperparameters

### Performance Optimization

1. **Use tf.function**: Decorate training functions with `@tf.function` for better performance
2. **Batch size**: Use larger batch sizes when possible (32-128)
3. **Mixed precision**: Enable mixed precision training for faster computation
4. **Data pipeline**: Optimize data loading with `tf.data` prefetching

```python
# Enable mixed precision
keras.mixed_precision.set_global_policy('mixed_float16')

# Optimize data pipeline
dataset = dataset.prefetch(tf.data.AUTOTUNE)
dataset = dataset.cache()
```

### Practical Considerations

#### Advantages

1. **Stable Training**: Less prone to mode collapse and training instability
2. **Meaningful Loss**: Loss values correlate with sample quality
3. **Better Gradients**: Provides useful gradients everywhere
4. **Theoretical Foundation**: Strong mathematical backing

#### Disadvantages

1. **Computational Cost**: Requires multiple critic updates per generator update
2. **Hyperparameter Sensitivity**: Gradient penalty coefficient needs tuning
3. **Implementation Complexity**: More complex than standard GAN training
4. **Convergence Speed**: Can be slower to converge than well-tuned standard GANs

## API Reference

### WassersteinLoss

```python
class WassersteinLoss(keras.losses.Loss):
    def __init__(
        self,
        for_critic: bool = True,
        reduction: str = "sum_over_batch_size",
        name: Optional[str] = None
    )
```

**Parameters:**
- `for_critic`: Whether loss is for critic (True) or generator (False)
- `reduction`: Reduction method for loss values
- `name`: Optional name for the loss

### WassersteinGradientPenaltyLoss

```python
class WassersteinGradientPenaltyLoss(keras.losses.Loss):
    def __init__(
        self,
        for_critic: bool = True,
        lambda_gp: float = 10.0,
        reduction: str = "sum_over_batch_size",
        name: Optional[str] = None
    )
```

**Parameters:**
- `for_critic`: Whether loss is for critic (True) or generator (False)
- `lambda_gp`: Gradient penalty coefficient
- `reduction`: Reduction method for loss values
- `name`: Optional name for the loss

### WassersteinDivergence

```python
class WassersteinDivergence(keras.losses.Loss):
    def __init__(
        self,
        smooth_eps: float = 1e-7,
        reduction: str = "sum_over_batch_size",
        name: Optional[str] = None
    )
```

**Parameters:**
- `smooth_eps`: Small epsilon value for numerical stability
- `reduction`: Reduction method for loss values
- `name`: Optional name for the loss

### Utility Functions

```python
def compute_gradient_penalty(
    critic: keras.Model,
    real_samples: tf.Tensor,
    fake_samples: tf.Tensor,
    lambda_gp: float = 10.0
) -> tf.Tensor
```

**Parameters:**
- `critic`: Critic model
- `real_samples`: Real data samples
- `fake_samples`: Generated fake samples
- `lambda_gp`: Gradient penalty coefficient

**Returns:**
- Gradient penalty loss value

```python
def create_wgan_losses(lambda_gp: float = 10.0) -> tuple[WassersteinLoss, WassersteinLoss]
def create_wgan_gp_losses(lambda_gp: float = 10.0) -> tuple[WassersteinGradientPenaltyLoss, WassersteinGradientPenaltyLoss]
```

Factory functions for creating paired critic and generator losses.

## Conclusion

Wasserstein loss represents a paradigm shift in how we approach distribution matching problems. While originally developed for GANs, its applications have expanded far beyond generative modeling to include domain adaptation, regularization, optimal transport, and many other areas.

### Key Insights

1. **Geometric Perspective**: Viewing distribution matching as a transport problem provides better mathematical properties
2. **Gradient Quality**: Non-vanishing gradients enable stable training
3. **Broad Applicability**: The principles extend well beyond GANs to any distribution matching task

### Current Research Directions

Current research continues to explore:
- More efficient computation methods
- Better approximation techniques
- Integration with other deep learning paradigms
- Applications to emerging domains like 3D generation and scientific computing

### Key Takeaways

- **For GANs**: Use when you need stable training and diverse generation
- **For Domain Adaptation**: Excellent for aligning feature distributions
- **For Regularization**: Helpful when you want to enforce distributional constraints
- **For Research**: Strong theoretical foundation enables principled extensions

Understanding Wasserstein dynamics is crucial for anyone working with generative models, domain adaptation, or distribution matching problems, as it provides both theoretical insights and practical advantages over traditional approaches. The Wasserstein loss isn't just a better GAN loss—it's a fundamental tool for any machine learning problem involving probability distributions.

This implementation provides a robust foundation for Wasserstein-based training with proper loss functions, gradient penalty computation, and comprehensive examples for various use cases, making it accessible for both research and production applications.