# Wasserstein Loss Documentation

## Overview

The Wasserstein loss implementation provides a complete suite of loss functions for training Wasserstein GANs (WGANs) and their variants. This implementation includes:

- **WassersteinLoss**: Basic Wasserstein loss for standard WGAN training
- **WassersteinGradientPenaltyLoss**: WGAN-GP loss with gradient penalty support
- **WassersteinDivergence**: General Wasserstein divergence for distribution matching
- **Utility functions**: Helper functions for gradient penalty computation and loss creation

## Mathematical Background

### Wasserstein Distance

The Wasserstein distance (also known as Earth Mover's Distance) between two probability distributions P and Q is defined as:

```
W(P, Q) = inf_{γ∈Π(P,Q)} E_{(x,y)~γ}[||x - y||]
```

For GANs, this is approximated using the Kantorovich-Rubinstein duality:

```
W(P, Q) ≈ max_{||f||_L ≤ 1} E_{x~P}[f(x)] - E_{x~Q}[f(x)]
```

### WGAN Loss

- **Critic Loss**: `L_critic = E[D(fake)] - E[D(real)]`
- **Generator Loss**: `L_generator = -E[D(fake)]`

### WGAN-GP Loss

WGAN-GP adds a gradient penalty term to enforce the Lipschitz constraint:

```
L_GP = λ * E_{x̂~P_x̂}[(||∇_{x̂}D(x̂)||_2 - 1)²]
```

where x̂ is sampled uniformly along lines between real and fake samples.

## Basic Usage

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

## Advanced Usage

### Custom Training Loop

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

## Utility Functions

### Creating Loss Functions

```python
from dl_techniques.losses.wasserstein_loss import create_wgan_losses, create_wgan_gp_losses

# Create standard WGAN losses
critic_loss, generator_loss = create_wgan_losses()

# Create WGAN-GP losses
critic_loss_gp, generator_loss_gp = create_wgan_gp_losses(lambda_gp=10.0)
```

### Gradient Penalty Computation

```python
from dl_techniques.losses.wasserstein_loss import compute_gradient_penalty

# Compute gradient penalty manually
gp_loss = compute_gradient_penalty(
    critic_model,
    real_samples,
    fake_samples,
    lambda_gp=10.0
)
```

## Best Practices

### 1. Learning Rates

- Use lower learning rates (1e-4 to 5e-5) for both generator and critic
- RMSprop optimizer often works better than Adam for WGANs
- For WGAN-GP, Adam can be used with β₁ = 0.5

```python
# Good optimizer settings
critic_optimizer = keras.optimizers.RMSprop(learning_rate=5e-5)
generator_optimizer = keras.optimizers.RMSprop(learning_rate=5e-5)

# Or for WGAN-GP
critic_optimizer = keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5)
generator_optimizer = keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5)
```

### 2. Training Ratio

- Train critic multiple times (typically 5) for each generator update
- This helps maintain the Lipschitz constraint

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

## Monitoring Training

### Key Metrics

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

### Troubleshooting

**Problem**: Training instability
- **Solution**: Reduce learning rates, increase critic training ratio

**Problem**: Mode collapse
- **Solution**: Use WGAN-GP instead of basic WGAN, adjust λ parameter

**Problem**: Gradient explosion
- **Solution**: Check gradient penalty implementation, ensure proper weight clipping

**Problem**: Poor sample quality
- **Solution**: Increase model capacity, adjust architecture, tune hyperparameters

## Example Applications

### 1. Image Generation

```python
# MNIST generation
trainer = WGANTrainer(generator, critic, latent_dim=100, use_gp=True)
trainer.train(mnist_dataset, epochs=50)
```

### 2. Domain Adaptation

```python
# Use Wasserstein loss for domain adaptation
domain_loss = WassersteinLoss(for_critic=True)

# Train domain classifier
domain_classifier.compile(optimizer='adam', loss=domain_loss)
```

### 3. Style Transfer

```python
# Distribution matching for style transfer
style_loss = WassersteinDivergence()

def style_transfer_loss(content_features, style_features):
    return style_loss(content_features, style_features)
```

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

### compute_gradient_penalty

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

## Performance Tips

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

This implementation provides a robust foundation for Wasserstein GAN training with proper loss functions, gradient penalty computation, and comprehensive examples for various use cases.