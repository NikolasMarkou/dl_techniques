# Using Denoiser Priors for Inverse Problems

This guide explains how to use a trained BF-CNN denoiser to solve various inverse problems based on the paper "Solving Linear Inverse Problems Using the Prior Implicit in a Denoiser" by Kadkhodaie & Simoncelli (2021).

## Key Theoretical Insight

The core idea is based on **Miyasawa's result** (1961) that shows:

```
x̂(y) = y + σ²∇_y log p(y)
```

Where:
- `x̂(y)` is the least-squares denoiser output
- `y` is the noisy input
- `σ²∇_y log p(y)` is the gradient of the log probability density
- The **denoiser residual** `f(y) = x̂(y) - y` is proportional to this gradient

This means a trained denoiser implicitly contains information about the probability distribution of natural images!

## Requirements for the Denoiser

Your denoiser **must** satisfy these conditions:

1. **Bias-free architecture**: No additive bias terms in any layer
2. **Gaussian noise training**: Trained on images with additive Gaussian noise
3. **Variable noise levels**: Trained with noise levels in range [0, σ_max] 
4. **MSE loss**: Optimized for mean squared error
5. **Blind denoising**: Works without knowing the noise level

## Quick Start

### 1. Basic Prior Sampling

```python
from denoiser_prior_sampling import DenoiserPriorSampler

# Initialize sampler
sampler = DenoiserPriorSampler(
    denoiser=your_bfcnn_model,
    sigma_0=1.0,    # Initial noise level
    sigma_l=0.01,   # Final noise level (stopping criterion)
    h0=0.01,        # Step size parameter
    beta=0.5        # Noise injection control (0=high noise, 1=no noise)
)

# Generate sample from implicit prior
shape = (1, 64, 64, 3)  # (batch, height, width, channels)
sample, convergence_info = sampler.sample_prior(shape, seed=42)
```

### 2. Solving Inverse Problems

```python
from denoiser_prior_sampling import LinearInverseProblemSolver

# Initialize solver
solver = LinearInverseProblemSolver(
    denoiser=your_bfcnn_model,
    beta=0.01  # Lower beta for inverse problems
)

# Example: Image inpainting
restored_image, convergence_info = solver.solve_inverse_problem(
    measurement_type='inpainting',
    measurements=observed_pixels,
    shape=original_shape,
    mask_size=(32, 32)
)
```

### 3. Ready-to-Use Applications

```python
from denoiser_prior_sampling import create_denoiser_applications

# Create application suite
apps = create_denoiser_applications(your_bfcnn_model)

# Use pre-built functions
inpainted = apps['inpaint'](damaged_image, mask_size=(32, 32))
super_resolved = apps['super_resolve'](low_res_image, factor=4)
samples = apps['sample_prior']((1, 64, 64, 3), n_samples=4)
```

## Supported Inverse Problems

### 1. **Inpainting**
- **Problem**: Fill missing rectangular regions
- **Use case**: Remove objects, restore damaged photos
- **Parameters**: `mask_size=(height, width)`

### 2. **Super-Resolution**
- **Problem**: Increase image resolution
- **Use case**: Enhance low-quality images
- **Parameters**: `factor=4` (upscaling factor)

### 3. **Random Pixel Recovery**
- **Problem**: Restore randomly missing pixels
- **Use case**: Sensor dropout, data corruption
- **Parameters**: `keep_ratio=0.1` (fraction of pixels kept)

### 4. **Deblurring (Spectral Super-Resolution)**
- **Problem**: Remove blur from known kernels
- **Use case**: Fix motion blur, defocus
- **Parameters**: Filter specifications

### 5. **Compressive Sensing**
- **Problem**: Reconstruct from random projections
- **Use case**: Accelerated MRI, sparse measurements
- **Parameters**: `measurement_ratio=0.1`

## Parameter Tuning Guide

### For Prior Sampling:
- **σ₀ = 1.0**: Start with high noise (explore broadly)
- **σₗ = 0.01**: Stop when noise is low (fine details)
- **h₀ = 0.01**: Conservative step size for stability
- **β = 0.5**: Moderate noise injection (balance exploration/exploitation)

### For Inverse Problems:
- **β = 0.01**: Lower noise injection (focus on constraint satisfaction)
- **σ₀ = 1.0**: Same initial noise level
- **σₗ = 0.01**: Same stopping criterion
- **h₀ = 0.01**: Same step size

### Convergence Tips:
- **Higher β**: Faster convergence, risk of local minima
- **Lower β**: Slower convergence, better exploration
- **Smaller h₀**: More stable, slower convergence  
- **Larger h₀**: Faster but less stable

## Algorithm Details

### Prior Sampling Algorithm (Algorithm 1)
```
1. Initialize: y₀ ~ N(0.5, σ₀²I)
2. While σₜ₋₁ > σₗ:
   a. Compute step size: hₜ = h₀t/(1 + h₀(t-1))
   b. Get denoiser residual: dₜ = f(yₜ₋₁) = denoiser(yₜ₋₁) - yₜ₋₁
   c. Estimate noise: σₜ = ||dₜ||/√N
   d. Compute noise injection: γₜ² = [(1-βhₜ)² - (1-hₜ)²]σₜ²
   e. Update: yₜ = yₜ₋₁ + hₜdₜ + γₜzₜ  (zₜ ~ N(0,I))
3. Return: final image yₜ
```

### Constrained Sampling Algorithm (Algorithm 2)
```
1. Initialize: y₀ with constraint projection + null space noise
2. While σₜ₋₁ > σₗ:
   a. Compute denoiser residual: f(yₜ₋₁)
   b. Project to constraint: dₜ = (I-MM^T)f(yₜ₋₁) + M(xc - M^Tyₜ₋₁)
   c. Apply same update rule as Algorithm 1
3. Return: constrained sample
```

## Performance Expectations

### Convergence:
- **Typical iterations**: 200-600 for most problems
- **Time per iteration**: ~50ms for 64×64 images (GPU)
- **Memory usage**: 2-3× denoiser memory requirements

### Quality:
- **Prior samples**: Natural-looking but may not match specific datasets
- **Inverse problems**: High perceptual quality, may sacrifice PSNR
- **Multiple samples**: Different plausible solutions for underdetermined problems

## Common Issues & Solutions

### 1. **Poor Convergence**
- **Symptoms**: σ doesn't decrease, artifacts remain
- **Solutions**: 
  - Check denoiser quality (try on simple denoising)
  - Reduce h₀ for stability
  - Increase β for faster convergence

### 2. **Constraint Violation**
- **Symptoms**: Measurements don't match in inverse problems
- **Solutions**:
  - Check measurement matrix construction
  - Verify measurements are consistent
  - Lower β to enforce constraints better

### 3. **Blurry Results**
- **Symptoms**: Results lack sharp details
- **Solutions**:
  - Decrease σₗ (run longer)
  - Increase h₀ (more aggressive steps)
  - Check if denoiser was properly trained

### 4. **Memory Issues**
- **Symptoms**: CUDA out of memory
- **Solutions**:
  - Use smaller batch sizes
  - Process tiles for large images
  - Use mixed precision

## Advanced Usage

### Custom Measurement Matrices
```python
# Create custom measurement matrix
def create_custom_measurements(image_shape, **params):
    # Your custom linear measurement process
    M = create_measurement_matrix(image_shape, **params)
    M_pinv = tf.transpose(M)  # For orthogonal M
    return M, M_pinv

# Use with solver
solver.solve_inverse_problem(
    measurement_type='custom',
    measurements=your_measurements,
    shape=image_shape,
    custom_M=M,
    custom_M_pinv=M_pinv
)
```

### Multiple Samples
```python
# Generate multiple plausible solutions
solutions = []
for i in range(10):
    solution, _ = solver.solve_inverse_problem(
        measurement_type='inpainting',
        measurements=measurements,
        shape=image_shape,
        seed=42+i  # Different random seeds
    )
    solutions.append(solution)

# Average for better PSNR (but more blur)
avg_solution = tf.reduce_mean(tf.stack(solutions), axis=0)
```

### Monitoring Convergence
```python
# Plot convergence curves
import matplotlib.pyplot as plt

_, conv_info = sampler.sample_prior(shape)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(conv_info['sigma_values'])
plt.yscale('log')
plt.title('Effective Noise Level')

plt.subplot(1, 3, 2)
plt.plot(conv_info['step_sizes'])
plt.title('Step Sizes')

plt.subplot(1, 3, 3)
plt.plot(conv_info['gamma_values'])
plt.title('Injected Noise')

plt.tight_layout()
plt.show()
```

## References

1. **Original Paper**: Kadkhodaie, Z., & Simoncelli, E. P. (2021). "Solving Linear Inverse Problems Using the Prior Implicit in a Denoiser"
2. **BF-CNN**: Mohan, S., et al. (2020). "Robust and interpretable blind image denoising via bias-free convolutional neural networks"
3. **Miyasawa's Result**: Miyasawa, K. (1961). "An empirical Bayes estimator of the mean of a normal population"

## Tips for Best Results

1. **Train your denoiser well**: The quality of results depends entirely on the denoiser
2. **Use appropriate noise ranges**: Train denoiser on σ ∈ [0, 0.4] for [0,1] images
3. **Multiple samples**: Generate several solutions and pick the best one
4. **Patience**: Let the algorithm converge fully for best quality
5. **Monitor convergence**: Watch σ values to ensure proper convergence