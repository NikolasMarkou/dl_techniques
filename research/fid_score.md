# Fréchet Inception Distance (FID): Mathematics and Reasoning

## Overview

The Fréchet Inception Distance (FID) is a metric for evaluating the quality of generated images, introduced by Heusel et al. in 2017. It was designed to address several limitations of the Inception Score (IS) by providing a more robust and sensitive measure of image generation quality. FID measures the distance between the distributions of real and generated images in a high-dimensional feature space.

## Core Intuition

FID is based on the principle that **good generated images should have similar statistical properties to real images** when represented in a meaningful feature space. Instead of relying on class predictions like IS, FID:

1. **Uses Feature Representations**: Compares images based on rich, high-dimensional features rather than discrete class labels
2. **Measures Distribution Distance**: Directly compares the distribution of real images to generated images
3. **Captures Finer Details**: Sensitive to both image quality and diversity through distributional properties

## Mathematical Formulation

The FID is defined as the Fréchet distance (also known as Wasserstein-2 distance) between two multivariate Gaussian distributions:

$$\text{FID} = ||\mu_r - \mu_g||^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$$

Where:
- $\mu_r, \Sigma_r$ are the mean and covariance matrix of real image features
- $\mu_g, \Sigma_g$ are the mean and covariance matrix of generated image features
- $\text{Tr}(\cdot)$ denotes the matrix trace
- $(\Sigma_r \Sigma_g)^{1/2}$ is the matrix square root of the product

## Feature Extraction

### Inception v3 Feature Space

FID uses the penultimate layer (pool3) of a pre-trained Inception v3 network as the feature extractor. For each image $x$, we obtain:

$$f(x) = \text{Inception-v3}_{\text{pool3}}(x) \in \mathbb{R}^{2048}$$

This layer provides a 2048-dimensional feature vector that captures high-level semantic information about the image.

### Gaussian Assumption

FID assumes that the feature vectors follow multivariate Gaussian distributions:

$$f_r \sim \mathcal{N}(\mu_r, \Sigma_r) \quad \text{(real images)}$$
$$f_g \sim \mathcal{N}(\mu_g, \Sigma_g) \quad \text{(generated images)}$$

## Fréchet Distance Derivation

### Definition of Fréchet Distance

The Fréchet distance between two probability distributions $P$ and $Q$ is defined as:

$$W_2(P, Q) = \inf_{\gamma \in \Gamma(P,Q)} \left(\int ||x - y||^2 d\gamma(x,y)\right)^{1/2}$$

Where $\Gamma(P,Q)$ is the set of all joint distributions with marginals $P$ and $Q$.

### Closed-Form Solution for Gaussians

For two multivariate Gaussian distributions $\mathcal{N}(\mu_1, \Sigma_1)$ and $\mathcal{N}(\mu_2, \Sigma_2)$, the squared Fréchet distance has a closed-form solution:

$$W_2^2(\mathcal{N}(\mu_1, \Sigma_1), \mathcal{N}(\mu_2, \Sigma_2)) = ||\mu_1 - \mu_2||^2 + \text{Tr}(\Sigma_1 + \Sigma_2 - 2(\Sigma_1 \Sigma_2)^{1/2})$$

This is exactly the FID formula.

## Component Analysis

### Mean Difference Term: $||\mu_r - \mu_g||^2$

This term measures the **distance between centroids** of the two distributions:

$$||\mu_r - \mu_g||^2 = \sum_{i=1}^{2048} (\mu_{r,i} - \mu_{g,i})^2$$

- **Interpretation**: Captures whether generated images have similar average characteristics to real images
- **Effect**: Large when generated images systematically differ from real images in their feature representations

### Covariance Term: $\text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$

This term measures the **difference in covariance structures**:

$$\text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$$

- **Interpretation**: Captures whether generated images have similar variability and correlation patterns to real images
- **Effect**: Large when the scatter/spread of generated features differs from real features

### Matrix Square Root Computation

The term $(\Sigma_r \Sigma_g)^{1/2}$ requires computing the square root of a matrix product. This is typically done using:

$$(\Sigma_r \Sigma_g)^{1/2} = \Sigma_r^{1/2} (\Sigma_r^{-1/2} \Sigma_g \Sigma_r^{-1/2})^{1/2} \Sigma_r^{1/2}$$

Or through eigenvalue decomposition of the product $\Sigma_r \Sigma_g$.

## Practical Computation Algorithm

### Step-by-Step Process

1. **Extract Features from Real Images**:
   ```
   For each real image x_r:
       f_r = Inception-v3_pool3(x_r)
   ```

2. **Extract Features from Generated Images**:
   ```
   For each generated image x_g:
       f_g = Inception-v3_pool3(x_g)
   ```

3. **Compute Statistics for Real Images**:
   ```
   μ_r = (1/N_r) Σ f_r
   Σ_r = (1/(N_r-1)) Σ (f_r - μ_r)(f_r - μ_r)ᵀ
   ```

4. **Compute Statistics for Generated Images**:
   ```
   μ_g = (1/N_g) Σ f_g
   Σ_g = (1/(N_g-1)) Σ (f_g - μ_g)(f_g - μ_g)ᵀ
   ```

5. **Calculate FID**:
   ```
   FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2(Σ_r Σ_g)^(1/2))
   ```

### Numerical Considerations

- **Regularization**: Add small values to diagonal of covariance matrices to ensure numerical stability
- **Sample Size**: Typically requires thousands of images for stable estimates
- **Preprocessing**: Images are usually resized to 299×299 and normalized to [0,1] or [-1,1]

## Theoretical Properties

### Metric Properties

FID satisfies the properties of a metric:

1. **Non-negativity**: $\text{FID} \geq 0$
2. **Identity**: $\text{FID} = 0$ if and only if the distributions are identical
3. **Symmetry**: $\text{FID}(P, Q) = \text{FID}(Q, P)$
4. **Triangle Inequality**: $\text{FID}(P, R) \leq \text{FID}(P, Q) + \text{FID}(Q, R)$

### Interpretation of Values

- **FID = 0**: Generated and real distributions are identical
- **Lower FID**: Better generation quality (generated images more similar to real images)
- **Higher FID**: Worse generation quality (generated images differ more from real images)

## Advantages Over Inception Score

### 1. **More Sensitive to Image Quality**
- FID detects subtle differences in image quality that IS might miss
- Uses rich feature representations instead of just class predictions

### 2. **Better Mode Collapse Detection**
- FID can detect when generators produce limited variety
- Covariance term captures distributional differences

### 3. **More Robust Statistics**
- Based on well-established distributional distance measures
- Less sensitive to outliers than IS

### 4. **Continuous Scale**
- Provides a continuous measure rather than discrete class-based evaluation
- More granular assessment of generation quality

## Limitations and Considerations

### Advantages
- **Robust**: More stable and reliable than IS
- **Sensitive**: Detects subtle quality differences
- **Interpretable**: Lower scores clearly indicate better quality
- **Widely Adopted**: Standard metric in generative modeling

### Limitations
- **Gaussian Assumption**: Assumes features follow multivariate Gaussian distributions
- **Inception v3 Bias**: Limited by the features learned by Inception v3
- **Computational Cost**: Requires computing covariance matrices and matrix square roots
- **Sample Size Dependency**: Needs large sample sizes for stable estimates
- **Resolution Dependency**: Inception v3 expects specific input sizes

## Variants and Extensions

### 1. **Kernel FID (KID)**
- Uses kernel-based approach instead of Gaussian assumption
- More robust to non-Gaussian feature distributions

### 2. **Precision and Recall for Distributions**
- Decomposes FID into precision and recall components
- Provides more detailed analysis of generation quality

### 3. **Multi-Scale FID**
- Computes FID at multiple resolutions
- Captures both global and local image properties

### 4. **Feature-Based Variants**
- Uses different feature extractors (e.g., ResNet, CLIP)
- Adapts to specific domains or image types

## Relationship to Other Metrics

### FID vs. Inception Score
- **FID**: Measures distributional distance in feature space
- **IS**: Measures quality and diversity through class predictions
- **Correlation**: Generally correlated but can disagree in edge cases

### FID vs. LPIPS
- **FID**: Population-level distributional measure
- **LPIPS**: Pairwise perceptual distance measure
- **Complementary**: Often used together for comprehensive evaluation

## Practical Recommendations

1. **Sample Size**: Use at least 10,000 images for stable FID estimates
2. **Preprocessing**: Ensure consistent image preprocessing between real and generated images
3. **Multiple Runs**: Report mean and standard deviation across multiple evaluations
4. **Baseline Comparison**: Compare against established baselines on standard datasets
5. **Complementary Metrics**: Use FID alongside other metrics for comprehensive evaluation

FID has become the gold standard for evaluating generative models due to its robustness, sensitivity, and theoretical grounding in optimal transport theory. It provides a principled way to measure how well a generative model captures the distribution of real images.