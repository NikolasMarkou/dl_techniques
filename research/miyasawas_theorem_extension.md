## Extending the Theorem: More Realistic Noise Models

Miyasawa's original theorem is derived for the case of simple additive, independent, and identically distributed (i.i.d.) Gaussian noise. While foundational, real-world measurement processes are often more complex. A common scenario, especially in imaging, involves noise that is subsequently blurred or correlated.

Let's extend the theorem to a more realistic noise model: **additive Gaussian noise followed by a linear convolution.**

### Case Study: Additive Gaussian Noise followed by Convolution

This noise model is common in applications like photography (blur from lens + sensor noise), medical imaging, and microscopy.

#### New Problem Setup

1.  **Clean signal**: $x \in \mathbb{R}^n$
2.  **Additive Gaussian noise**: $\varepsilon \sim \mathcal{N}(0, \sigma^2 I)$
3.  **Intermediate noisy signal**: $z = x + \varepsilon$
4.  **Convolution Kernel**: A linear operator, represented by a matrix $K \in \mathbb{R}^{n \times n}$. For an image, $K$ would be a block-Toeplitz matrix, but we can simply think of it as a linear operator that performs the convolution.
5.  **Final noisy observation**: $y = Kz = K(x + \varepsilon)$

Our goal is to find the relationship between the least-squares optimal denoiser $\hat{x}(y) = \mathbb{E}[x|y]$ and the score function $\nabla_y \log p(y)$ under this new noise model.

### The Extended Miyasawa's Theorem

For the noise model $y = K(x + \varepsilon)$, the relationship between the optimal estimator and the score function is:

$$\boxed{K\hat{x}(y) = y + \sigma^2 (KK^T) \nabla_y \log p(y)}$$

where:
- $\hat{x}(y) = \mathbb{E}[x|y]$ is the optimal estimate of the clean signal.
- $K\hat{x}(y) = \mathbb{E}[Kx|y]$ is the optimal estimate of the **blurred clean signal**.
- $p(y)$ is the probability density of the final observations.
- $K^T$ is the transpose of the convolution operator (equivalent to convolution with a flipped kernel).
- $KK^T$ represents the convolution kernel applied twice (once forward, once with the flipped kernel), which captures the covariance structure of the convolved noise.

#### Equivalent Formulations

**Score Function Form**:
$$\nabla_y \log p(y) = \frac{1}{\sigma^2} (KK^T)^{-1} (K\hat{x}(y) - y)$$

This shows that the score is now related to the residual in the *blurred domain*, pre-multiplied by the inverse of the convolved noise covariance.

### Detailed Mathematical Derivation for the Extended Case

The derivation follows a similar path to the original, but the likelihood term $p(y|x)$ is more complex.

#### Step 1: Define the Likelihood $p(y|x)$

Given a clean signal $x$, the observation $y$ is a linear transformation of a Gaussian variable:
$$y = Kx + K\varepsilon$$
Let the transformed noise be $\tilde{\varepsilon} = K\varepsilon$. Since $\varepsilon$ is Gaussian, $\tilde{\varepsilon}$ is also Gaussian with:
- **Mean**: $\mathbb{E}[\tilde{\varepsilon}] = K\mathbb{E}[\varepsilon] = 0$
- **Covariance**: $\text{Cov}(\tilde{\varepsilon}) = \mathbb{E}[(K\varepsilon)(K\varepsilon)^T] = K\mathbb{E}[\varepsilon\varepsilon^T]K^T = K(\sigma^2 I)K^T = \sigma^2 KK^T$

Therefore, the conditional distribution of $y$ given $x$ is a multivariate Gaussian:
$$y | x \sim \mathcal{N}(Kx, \sigma^2 KK^T)$$
The likelihood function is:
$$p(y|x) = \frac{1}{Z} \exp\left(-\frac{1}{2\sigma^2} (y - Kx)^T (KK^T)^{-1} (y - Kx)\right)$$
where $Z$ is the normalization constant.

#### Step 2: Gradient of Observation Density

As before, we start with $p(y) = \int p(y|x) p(x) \, dx$ and take the gradient:
$$\nabla_y p(y) = \int \nabla_y p(y|x) p(x) \, dx$$

#### Step 3: Gradient of the New Gaussian Likelihood

We use the matrix calculus identity $\nabla_v (v-a)^T M (v-a) = 2M(v-a)$.
$$\nabla_y p(y|x) = p(y|x) \cdot \nabla_y \left(-\frac{1}{2\sigma^2} (y - Kx)^T (KK^T)^{-1} (y - Kx)\right)$$
$$\nabla_y p(y|x) = p(y|x) \cdot \left(-\frac{1}{\sigma^2} (KK^T)^{-1} (y - Kx)\right)$$

#### Step 4: Simplification Using Bayes' Rule

Substitute this back into the integral for $\nabla_y p(y)$:
$$\nabla_y p(y) = \int p(y|x) \left(-\frac{1}{\sigma^2} (KK^T)^{-1} (y - Kx)\right) p(x) \, dx$$
Using Bayes' rule, $p(y|x)p(x) = p(x|y)p(y)$:
$$\nabla_y p(y) = -\frac{p(y)}{\sigma^2} (KK^T)^{-1} \int (y - Kx) p(x|y) \, dx$$
Now, we evaluate the integral:
$$\int (y - Kx) p(x|y) \, dx = y \int p(x|y) \, dx - K \int x p(x|y) \, dx$$
$$= y \cdot 1 - K \cdot \mathbb{E}[x|y] = y - K\hat{x}(y)$$

#### Step 5: Final Result

Substituting the evaluated integral back:
$$\nabla_y p(y) = -\frac{p(y)}{\sigma^2} (KK^T)^{-1} (y - K\hat{x}(y))$$
Dividing by $p(y)$ gives the score function:
$$\nabla_y \log p(y) = -\frac{1}{\sigma^2} (KK^T)^{-1} (y - K\hat{x}(y))$$
Rearranging to solve for the denoised term gives us the final theorem:
$$\sigma^2 (KK^T) \nabla_y \log p(y) = -(y - K\hat{x}(y))$$
$$\boxed{K\hat{x}(y) = y + \sigma^2 (KK^T) \nabla_y \log p(y)}$$

### Intuitive Interpretation of the Extended Theorem

#### Comparison Table

| Aspect | Original Theorem (Additive Noise) | Extended Theorem (Convolved Noise) |
| :--- | :--- | :--- |
| **Noise Model** | $y = x + \varepsilon$ | $y = K(x + \varepsilon)$ |
| **Residual Term** | $\hat{x}(y) - y$ | $K\hat{x}(y) - y$ |
| **Score Scaling** | $\sigma^2 I$ (Identity) | $\sigma^2 (KK^T)$ |
| **Core Equation**| $\hat{x}(y) = y + \sigma^2 \nabla_y \log p(y)$ | $K\hat{x}(y) = y + \sigma^2 (KK^T) \nabla_y \log p(y)$ |

#### Key Insights

1.  **Denoising in the Blurred Domain**: The theorem no longer directly relates the score to the clean estimate $\hat{x}(y)$. Instead, it connects the score to $K\hat{x}(y)$, which is the optimal estimate of the *blurred* clean signal. The "natural" quantity to estimate from $y$ is $Kx$, not $x$ itself.

2.  **Noise Covariance Matters**: The simple scalar variance $\sigma^2$ is replaced by the matrix $\sigma^2 KK^T$. This matrix represents the covariance of the noise after convolution. The convolution introduces correlations between neighboring pixels in the noise, and the term $KK^T$ correctly captures this new structure.

3.  **Connection to Wiener Filtering**: The term $(KK^T)^{-1}$ required to extract the score is a deconvolution operator. In the Fourier domain, convolution with $K$ is multiplication by its frequency response $\mathcal{F}(K)$. The operation $KK^T$ corresponds to multiplying by $|\mathcal{F}(K)|^2$. The inverse operation, $(KK^T)^{-1}$, corresponds to dividing by $|\mathcal{F}(K)|^2$, which is a core component of a classical Wiener deconvolution filter. This shows a deep connection between Miyasawa's theorem and classical signal processing.

### Practical Implications

If you train a neural network $D_\theta(y)$ to estimate the clean signal $x$ from the convolved noisy observation $y$, it approximates $\hat{x}(y)$. You can then use the extended theorem to estimate the score function:

```python
import tensorflow as tf

def extract_score_for_convolved_noise(denoiser_model, y, K_op, K_transpose_op, sigma):
    """
    Extracts the score function for a convolved noise model.

    Args:
        denoiser_model: A model trained to predict clean x from y = K(x + noise).
        y: The noisy, convolved observation.
        K_op: A function/operator that performs the convolution K.
        K_transpose_op: A function/operator for the transpose convolution K^T.
        sigma: The standard deviation of the additive noise before convolution.

    Returns:
        Estimated score function ∇log p(y).
    """
    # 1. Get the estimate of the clean signal
    x_hat = denoiser_model(y, training=False)
    
    # 2. Compute the residual in the blurred domain: K*x_hat - y
    Kx_hat = K_op(x_hat)
    residual = Kx_hat - y
    
    # 3. Apply the inverse of the noise covariance (KK^T)^-1
    # This is a deconvolution step, often solved iteratively or in Fourier space.
    # For simplicity, here we assume an `inverse_KK_transpose_op` exists.
    # In practice, one would use e.g. conjugate gradient or FFT-based division.
    # F_inv_op = lambda z: tf.signal.ifft2d(tf.signal.fft2d(z) / (tf.abs(F_K)**2 + 1e-6))
    
    # Let's assume we have a solver for (KK^T)s = r
    score = solve_linear_system(lambda s: K_op(K_transpose_op(s)), residual)
    
    return score / (sigma**2)

# Placeholder for a linear solver like conjugate gradient
def solve_linear_system(A_op, b):
    # In a real implementation, use tf.linalg.experimental.conjugate_gradient
    # This is a simplified stand-in.
    # For FFT-based convolution, this becomes much simpler.
    print("Placeholder: Solving (KK^T)s = r. Use a proper linear solver.")
    return b # Incorrect, but shows the structure
```

This extension demonstrates the power and flexibility of Miyasawa's framework. By correctly identifying the likelihood of the measurement process, the theorem can be adapted to a wide variety of linear inverse problems, providing a theoretical foundation for modern score-based and diffusion-based methods in signal and image restoration.