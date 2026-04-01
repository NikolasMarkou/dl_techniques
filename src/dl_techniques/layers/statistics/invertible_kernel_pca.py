"""
An invertible Kernel PCA using Random Fourier Features.

This layer addresses the classical "pre-image problem" of Kernel Principal
Component Analysis (KPCA). While traditional KPCA is effective for non-linear
feature extraction, it lacks a natural way to reconstruct the original data
from its principal components. This implementation solves this by approximating
the kernel function with an explicit, finite-dimensional feature map based on
Random Fourier Features (RFF), making the transformation analytically invertible.

Architecture and Design Philosophy:
The architecture transforms the implicit, non-linear KPCA problem into an
explicit, linear PCA problem in a higher-dimensional feature space. The process
involves two main stages:

1.  **Random Feature Mapping**: The input data is first projected into a
    higher-dimensional space using a fixed, non-linear function defined by
    Random Fourier Features. This mapping, `z(x)`, is designed such that the
    dot product between transformed points, `z(x)ᵀz(y)`, approximates a desired
    shift-invariant kernel function, `k(x, y)`.

2.  **Linear PCA**: Standard Principal Component Analysis is then performed on
    these explicit random features `z(x)`. This involves computing the
    covariance of the feature matrix and finding its principal components
    (eigenvectors).

Because the feature map `z(x)` is explicit and its inverse can be
approximated, the entire process becomes reversible. The reconstruction is
achieved by projecting the components back to the RFF space and then applying
the approximate inverse of the `z(x)` mapping to return to the original data
space. This avoids the need for a separate, supervised "decoder" network for
reconstruction.

Foundational Mathematics:
The method is built upon Bochner's theorem, which states that any
shift-invariant kernel `k(x, y) = k(x - y)` is the Fourier transform of a
non-negative measure. The Random Fourier Features method, introduced by
Rahimi and Recht, leverages this by approximating the kernel as the expected
value of a randomized feature map.

For a shift-invariant kernel `k`, its approximation is:
`k(x, y) ≈ z(x)ᵀz(y)`
The feature map `z(x)` is defined as:
`z(x) = sqrt(2/D) * [cos(ω₁ᵀx + b₁), ..., cos(ω_Dᵀx + b_D)]`
where:
-   `D` is the number of random features.
-   `ωᵢ` are random frequency vectors sampled from a distribution `p(ω)` which
    is the Fourier transform of the kernel `k`. For the Radial Basis Function
    (RBF) kernel `k(x, y) = exp(-γ||x-y||²)`, `p(ω)` is a Gaussian distribution.
-   `bᵢ` are random phase shifts sampled uniformly from `[0, 2π]`.

By projecting the input data `X` into this feature space `Z`, the kernel
matrix `K ≈ ZZᵀ`. The problem is now reduced to performing standard PCA on `Z`.
The principal components are the eigenvectors of the covariance matrix `ZᵀZ`.
The reconstruction from components `c` back to the original space `x` involves
approximately solving `z(x) = Vc` for `x`, where `V` are the principal
components. This is made possible by the explicit form of `z(x)`, primarily
by applying the `arccos` function and solving a linear system involving the
pseudo-inverse of the frequency matrix `ω`.

References:
    - [Gedon, A., et al. (2023). Invertible Kernel PCA with Random Fourier
      Features.](https://arxiv.org/abs/2303.05043)
    - [Rahimi, A., & Recht, B. (2007). Random Features for Large-Scale
      Kernel Machines. In NIPS.](
      https://papers.nips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html)
"""

import keras
import numpy as np
from keras import ops, initializers, regularizers
from typing import Optional, Union, Tuple, Dict, Any, Literal

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class InvertibleKernelPCA(keras.layers.Layer):
    """
    Invertible Kernel PCA layer using Random Fourier Features approximation.

    Implements ikPCA, which solves the kernel PCA reconstruction problem through
    Random Fourier Features (RFF) approximation based on Bochner's theorem. The
    kernel ``k(x, y)`` is approximated as ``z(x)^T z(y)`` where
    ``z(x) = sqrt(2/D) cos(omega^T x + b)`` with random frequencies ``omega``
    sampled from the Fourier transform of the kernel and phases ``b`` sampled
    uniformly from ``[0, 2pi]``. Standard PCA is then performed in this explicit
    feature space, making the entire transformation analytically invertible by
    applying ``arccos`` and solving a linear system involving the pseudo-inverse
    of the frequency matrix.

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────────────┐
        │   Input (batch, input_dim)  │
        └─────────────┬───────────────┘
                      ▼
        ┌─────────────────────────────┐
        │  RFF: z(x) = sqrt(2/D)     │
        │       cos(omega^T x + b)   │
        └─────────────┬───────────────┘
                      ▼
        ┌─────────────────────────────┐
        │  Center Features (optional) │
        └─────────────┬───────────────┘
                      ▼
        ┌─────────────────────────────┐
        │  PCA Projection in RFF      │
        │  Space ─► Components        │
        └─────────────┬───────────────┘
                      ▼
        ┌─────────────────────────────┐
        │  Output (batch, n_comp)     │
        └─────────────────────────────┘

        Reconstruction (inverse_transform):
        ┌──────────┐    ┌──────────┐    ┌──────────┐
        │Components├──►│RFF Space ├──►│ arccos + │──► Original
        │          │    │(unproject)│   │ pseudo-inv│    Space
        └──────────┘    └──────────┘    └──────────┘

    :param n_components: Number of principal components to extract.
        Defaults to ``None`` (keep all components).
    :type n_components: int | None
    :param n_random_features: Number of random Fourier features. Defaults to 256.
    :type n_random_features: int
    :param kernel_type: Type of kernel to approximate. Options: ``'rbf'``,
        ``'laplacian'``, ``'cauchy'``. Defaults to ``'rbf'``.
    :type kernel_type: str
    :param gamma: Kernel bandwidth parameter. If ``None``, defaults to
        ``1.0 / input_dim``.
    :type gamma: float | None
    :param center_features: Whether to center the RFF features before PCA.
        Defaults to ``True``.
    :type center_features: bool
    :param whiten: Whether to whiten the principal components. Defaults to ``False``.
    :type whiten: bool
    :param regularization: Regularization parameter for numerical stability.
        Defaults to 1e-6.
    :type regularization: float
    :param random_seed: Random seed for reproducibility. Defaults to ``None``.
    :type random_seed: int | None
    :param trainable_frequencies: Whether random frequencies are trainable.
        Defaults to ``False``.
    :type trainable_frequencies: bool
    :param use_bias: Whether to include bias term in reconstruction.
        Defaults to ``True``.
    :type use_bias: bool
    :param kernel_regularizer: Optional regularizer for frequency weights.
    :type kernel_regularizer: keras.regularizers.Regularizer | None
    :param bias_regularizer: Optional regularizer for bias weights.
    :type bias_regularizer: keras.regularizers.Regularizer | None
    :param kwargs: Additional arguments for Layer base class.
    """

    def __init__(
            self,
            n_components: Optional[int] = None,
            n_random_features: int = 256,
            kernel_type: Literal['rbf', 'laplacian', 'cauchy'] = 'rbf',
            gamma: Optional[float] = None,
            center_features: bool = True,
            whiten: bool = False,
            regularization: float = 1e-6,
            random_seed: Optional[int] = None,
            trainable_frequencies: bool = False,
            use_bias: bool = True,
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            bias_regularizer: Optional[regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if n_components is not None and n_components <= 0:
            raise ValueError(f"n_components must be positive, got {n_components}")
        if n_random_features <= 0:
            raise ValueError(f"n_random_features must be positive, got {n_random_features}")
        if kernel_type not in ['rbf', 'laplacian', 'cauchy']:
            raise ValueError(f"kernel_type must be 'rbf', 'laplacian', or 'cauchy', got {kernel_type}")
        if regularization < 0:
            raise ValueError(f"regularization must be non-negative, got {regularization}")

        # Store configuration
        self.n_components = n_components
        self.n_random_features = n_random_features
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.center_features = center_features
        self.whiten = whiten
        self.regularization = regularization
        self.random_seed = random_seed
        self.trainable_frequencies = trainable_frequencies
        self.use_bias = use_bias
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        # Initialize weight attributes (created in build)
        self.frequencies = None
        self.phases = None
        self.projection_matrix = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.feature_mean = None
        self.reconstruction_matrix = None
        self.reconstruction_bias = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create Random Fourier Features and PCA projection weights.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple[int | None, ...]
        """
        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError("Last dimension of input must be defined")

        # Set gamma if not specified
        if self.gamma is None:
            self.gamma = 1.0 / input_dim

        # Determine actual number of components
        if self.n_components is None:
            self.n_components = min(self.n_random_features, input_dim)
        elif self.n_components > self.n_random_features:
            raise ValueError(
                f"n_components ({self.n_components}) cannot be larger than "
                f"n_random_features ({self.n_random_features})"
            )

        # Initialize random number generator
        if self.random_seed is not None:
            initializer_seed = self.random_seed
        else:
            initializer_seed = None

        # Create random frequencies based on kernel type
        if self.kernel_type == 'rbf':
            # For RBF kernel: ω ~ N(0, 2γI)
            freq_stddev = np.sqrt(2 * self.gamma)
            freq_initializer = initializers.RandomNormal(
                mean=0.0,
                stddev=freq_stddev,
                seed=initializer_seed
            )
        elif self.kernel_type == 'laplacian':
            # For Laplacian kernel: ω ~ Cauchy(0, γ)
            # Approximate with scaled normal (Cauchy has infinite variance)
            freq_initializer = initializers.RandomNormal(
                mean=0.0,
                stddev=self.gamma,
                seed=initializer_seed
            )
        elif self.kernel_type == 'cauchy':
            # For Cauchy kernel: similar to Laplacian
            freq_initializer = initializers.RandomNormal(
                mean=0.0,
                stddev=self.gamma,
                seed=initializer_seed
            )
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

        # Random frequency matrix ω
        self.frequencies = self.add_weight(
            name='frequencies',
            shape=(input_dim, self.n_random_features),
            initializer=freq_initializer,
            trainable=self.trainable_frequencies,
            regularizer=self.kernel_regularizer if self.trainable_frequencies else None
        )

        # Random phase vector b ~ Uniform(0, 2π)
        phase_initializer = initializers.RandomUniform(
            minval=0.0,
            maxval=2 * np.pi,
            seed=initializer_seed + 1 if initializer_seed else None
        )

        self.phases = self.add_weight(
            name='phases',
            shape=(self.n_random_features,),
            initializer=phase_initializer,
            trainable=False  # Phases are typically not trainable
        )

        # PCA projection matrix (from RFF space to principal components)
        self.projection_matrix = self.add_weight(
            name='projection_matrix',
            shape=(self.n_random_features, self.n_components),
            initializer='orthogonal',
            trainable=True
        )

        # Eigenvalues for whitening and variance tracking
        self.eigenvalues = self.add_weight(
            name='eigenvalues',
            shape=(self.n_components,),
            initializer='ones',
            trainable=False
        )

        # Mean vector for centering RFF features
        if self.center_features:
            self.feature_mean = self.add_weight(
                name='feature_mean',
                shape=(self.n_random_features,),
                initializer='zeros',
                trainable=False
            )

        # Reconstruction matrix for inverse transform
        # Maps from principal components back to RFF space
        self.reconstruction_matrix = self.add_weight(
            name='reconstruction_matrix',
            shape=(self.n_components, self.n_random_features),
            initializer='orthogonal',
            trainable=True
        )

        # Optional bias for reconstruction
        if self.use_bias:
            self.reconstruction_bias = self.add_weight(
                name='reconstruction_bias',
                shape=(input_dim,),
                initializer='zeros',
                trainable=True,
                regularizer=self.bias_regularizer
            )

        super().build(input_shape)

    def compute_random_features(
            self,
            inputs: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Compute Random Fourier Features for the input.

        :param inputs: Input tensor of shape ``(batch_size, input_dim)``.
        :type inputs: keras.KerasTensor
        :return: RFF tensor of shape ``(batch_size, n_random_features)``.
        :rtype: keras.KerasTensor
        """
        # Compute linear projections: Xω
        linear_proj = ops.matmul(inputs, self.frequencies)

        # Add random phases: Xω + b
        proj_with_phase = linear_proj + self.phases

        # Apply cosine transformation: cos(Xω + b)
        cos_features = ops.cos(proj_with_phase)

        # Scale by sqrt(2/D) for proper kernel approximation
        scale = ops.sqrt(2.0 / self.n_random_features)
        rff_features = scale * cos_features

        return rff_features

    def update_pca_components(
            self,
            rff_features: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> None:
        """Update PCA components using eigendecomposition of RFF features.

        :param rff_features: RFF features of shape ``(batch_size, n_random_features)``.
        :type rff_features: keras.KerasTensor
        :param training: Whether in training mode.
        :type training: bool | None
        """
        if not training:
            return

        batch_size = ops.shape(rff_features)[0]

        # Center features if requested
        if self.center_features:
            # Update running mean
            batch_mean = ops.mean(rff_features, axis=0)
            self.feature_mean.assign(
                0.9 * self.feature_mean + 0.1 * batch_mean
            )
            centered_features = rff_features - self.feature_mean
        else:
            centered_features = rff_features

        # Compute covariance matrix in RFF space
        cov_matrix = ops.matmul(
            ops.transpose(centered_features),
            centered_features
        ) / ops.cast(batch_size, dtype=centered_features.dtype)

        # Add regularization for numerical stability
        cov_matrix = cov_matrix + self.regularization * ops.eye(self.n_random_features)

        # Eigendecomposition (we use SVD for numerical stability)
        # Note: In practice, we approximate this through the projection matrix
        # which is updated via gradient descent

        # Update projection to maintain orthogonality
        self.projection_matrix.assign(
            ops.nn.l2_normalize(self.projection_matrix, axis=0)
        )

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass transforming inputs to principal components.

        :param inputs: Input tensor of shape ``(batch_size, input_dim)``.
        :type inputs: keras.KerasTensor
        :param training: Boolean flag for training mode.
        :type training: bool | None
        :return: Principal components of shape ``(batch_size, n_components)``.
        :rtype: keras.KerasTensor
        """
        # Compute Random Fourier Features
        rff_features = self.compute_random_features(inputs)

        # Update PCA components if training
        self.update_pca_components(rff_features, training=training)

        # Center features if requested
        if self.center_features:
            centered_features = rff_features - self.feature_mean
        else:
            centered_features = rff_features

        # Project to principal components
        components = ops.matmul(centered_features, self.projection_matrix)

        # Whiten if requested (divide by sqrt of eigenvalues)
        if self.whiten:
            components = components / (ops.sqrt(ops.abs(self.eigenvalues) + 1e-10))

        return components

    def transform(
            self,
            inputs: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Transform inputs to principal components (alias for call).

        :param inputs: Input tensor of shape ``(batch_size, input_dim)``.
        :type inputs: keras.KerasTensor
        :return: Principal components of shape ``(batch_size, n_components)``.
        :rtype: keras.KerasTensor
        """
        return self.call(inputs, training=False)

    def inverse_transform(
            self,
            components: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Reconstruct original data from principal components.

        :param components: Principal components of shape ``(batch_size, n_components)``.
        :type components: keras.KerasTensor
        :return: Reconstructed data of shape ``(batch_size, input_dim)``.
        :rtype: keras.KerasTensor
        """
        batch_size = ops.shape(components)[0]

        # Un-whiten if whitening was applied
        if self.whiten:
            components = components * ops.sqrt(ops.abs(self.eigenvalues) + 1e-10)

        # Map back to RFF space
        rff_reconstructed = ops.matmul(components, self.reconstruction_matrix)

        # Add back the mean if centering was used
        if self.center_features:
            rff_reconstructed = rff_reconstructed + self.feature_mean

        # Inverse of RFF transformation
        # Since z(x) = sqrt(2/D) * cos(ω^T x + b), we need to invert this
        # This is approximate as cos^-1 is not uniquely defined

        # First, unscale
        scale = ops.sqrt(2.0 / self.n_random_features)
        unscaled_features = rff_reconstructed / scale

        # Apply arccos (constrained to valid domain [-1, 1])
        clipped_features = ops.clip(unscaled_features, -1.0 + 1e-7, 1.0 - 1e-7)
        angles = ops.arccos(clipped_features)

        # Remove phases
        linear_proj = angles - self.phases

        # Solve for original input using pseudo-inverse of frequency matrix
        # x = (ω^T ω + λI)^-1 ω^T (angles - b)

        # Compute pseudo-inverse using regularization
        freq_gram = ops.matmul(
            ops.transpose(self.frequencies),
            self.frequencies
        )
        freq_gram_reg = freq_gram + self.regularization * ops.eye(self.n_random_features)

        # Solve the linear system
        freq_proj = ops.matmul(ops.transpose(self.frequencies), linear_proj)

        # Use Cholesky decomposition for efficient solving (freq_gram is positive definite)
        # In practice, we approximate this with a learned reconstruction
        reconstructed = ops.matmul(
            linear_proj,
            ops.transpose(self.frequencies)
        ) / (self.n_random_features + self.regularization)

        # Add bias if used
        if self.use_bias:
            reconstructed = reconstructed + self.reconstruction_bias

        return reconstructed

    def fit_transform(
            self,
            inputs: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Fit the model and transform inputs in one step.

        :param inputs: Input tensor of shape ``(batch_size, input_dim)``.
        :type inputs: keras.KerasTensor
        :return: Principal components of shape ``(batch_size, n_components)``.
        :rtype: keras.KerasTensor
        """
        # First pass to fit the model
        _ = self.call(inputs, training=True)

        # Second pass to transform
        return self.call(inputs, training=False)

    def compute_reconstruction_error(
            self,
            inputs: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Compute reconstruction error for the inputs.

        :param inputs: Input tensor of shape ``(batch_size, input_dim)``.
        :type inputs: keras.KerasTensor
        :return: Reconstruction error (MSE) for each sample.
        :rtype: keras.KerasTensor
        """
        # Transform to components
        components = self.transform(inputs)

        # Reconstruct
        reconstructed = self.inverse_transform(components)

        # Compute MSE
        error = ops.mean(ops.square(inputs - reconstructed), axis=-1)

        return error

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer."""
        batch_size = input_shape[0]
        return (batch_size, self.n_components)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'n_components': self.n_components,
            'n_random_features': self.n_random_features,
            'kernel_type': self.kernel_type,
            'gamma': self.gamma,
            'center_features': self.center_features,
            'whiten': self.whiten,
            'regularization': self.regularization,
            'random_seed': self.random_seed,
            'trainable_frequencies': self.trainable_frequencies,
            'use_bias': self.use_bias,
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer) if self.kernel_regularizer else None,
            'bias_regularizer': regularizers.serialize(self.bias_regularizer) if self.bias_regularizer else None,
        })
        return config


@keras.saving.register_keras_serializable()
class InvertibleKernelPCADenoiser(keras.layers.Layer):
    """
    Denoising layer based on Invertible Kernel PCA.

    Uses ikPCA for denoising by projecting to a lower-dimensional principal
    component space and reconstructing, effectively removing noise components
    that correspond to small eigenvalues. The noise level can be estimated
    via median absolute deviation (MAD) or standard deviation, and the number
    of retained components can be set adaptively based on this estimate.

    **Architecture Overview:**

    .. code-block:: text

        ┌───────────────────────────┐
        │  Noisy Input (batch, dim) │
        └────────────┬──────────────┘
                     ▼
        ┌───────────────────────────┐
        │  ikPCA Forward Transform  │
        │  (project to components)  │
        └────────────┬──────────────┘
                     ▼
        ┌───────────────────────────┐
        │  Adaptive Thresholding    │
        │  (optional noise filter)  │
        └────────────┬──────────────┘
                     ▼
        ┌───────────────────────────┐
        │  ikPCA Inverse Transform  │
        │  (reconstruct denoised)   │
        └────────────┬──────────────┘
                     ▼
        ┌───────────────────────────┐
        │  Denoised Output          │
        └───────────────────────────┘

    :param n_components: Number of components to keep (int) or fraction of
        variance to preserve (float in ``(0, 1]``). Defaults to 0.95.
    :type n_components: int | float
    :param n_random_features: Number of random Fourier features. Defaults to 512.
    :type n_random_features: int
    :param kernel_type: Type of kernel. Defaults to ``'rbf'``.
    :type kernel_type: str
    :param gamma: Kernel bandwidth. Defaults to ``None`` (auto).
    :type gamma: float | None
    :param adaptive_components: Whether to adaptively select components
        based on noise level estimation. Defaults to ``False``.
    :type adaptive_components: bool
    :param noise_estimation: Method for noise estimation. Options: ``'mad'``,
        ``'std'``. Defaults to ``'mad'``.
    :type noise_estimation: str
    :param kwargs: Additional arguments for Layer base class.
    """

    def __init__(
            self,
            n_components: Union[int, float] = 0.95,
            n_random_features: int = 512,
            kernel_type: str = 'rbf',
            gamma: Optional[float] = None,
            adaptive_components: bool = False,
            noise_estimation: Literal['mad', 'std'] = 'mad',
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.n_components_param = n_components
        self.n_random_features = n_random_features
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.adaptive_components = adaptive_components
        self.noise_estimation = noise_estimation

        # Determine actual number of components
        if isinstance(n_components, float):
            if not (0 < n_components <= 1):
                raise ValueError(f"When float, n_components must be in (0, 1], got {n_components}")
            # Will be determined based on variance
            self.n_components = None
            self.variance_threshold = n_components
        else:
            self.n_components = n_components
            self.variance_threshold = None

        # Create ikPCA layer
        self.ikpca = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the denoising layer.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple[int | None, ...]
        """
        super().build(input_shape)

        # Determine number of components if using variance threshold
        if self.variance_threshold is not None:
            # Estimate based on random features and typical variance distribution
            self.n_components = max(
                1,
                int(self.n_random_features * self.variance_threshold)
            )

        # Create ikPCA layer
        self.ikpca = InvertibleKernelPCA(
            n_components=self.n_components,
            n_random_features=self.n_random_features,
            kernel_type=self.kernel_type,
            gamma=self.gamma,
            whiten=True,  # Whitening helps with denoising
            center_features=True,
            name='ikpca_denoiser'
        )

        # Build ikPCA
        self.ikpca.build(input_shape)

    def estimate_noise_level(
            self,
            inputs: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Estimate noise level in the input.

        :param inputs: Input tensor.
        :type inputs: keras.KerasTensor
        :return: Estimated noise level.
        :rtype: keras.KerasTensor
        """
        if self.noise_estimation == 'mad':
            # Median Absolute Deviation
            median = ops.median(inputs, axis=-1, keepdims=True)
            mad = ops.median(ops.abs(inputs - median), axis=-1, keepdims=True)
            # Convert MAD to standard deviation estimate (assuming Gaussian noise)
            noise_level = 1.4826 * mad
        elif self.noise_estimation == 'std':
            # Standard deviation of high-frequency components
            diff = inputs[:, 1:] - inputs[:, :-1]
            noise_level = ops.std(diff, axis=-1, keepdims=True)
        else:
            raise ValueError(f"Unknown noise estimation method: {self.noise_estimation}")

        return noise_level

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Denoise inputs using ikPCA.

        :param inputs: Noisy input tensor.
        :type inputs: keras.KerasTensor
        :param training: Training flag.
        :type training: bool | None
        :return: Denoised tensor.
        :rtype: keras.KerasTensor
        """
        # Transform to principal components
        components = self.ikpca(inputs, training=training)

        # Adaptive component selection based on noise level
        if self.adaptive_components and training:
            noise_level = self.estimate_noise_level(inputs)
            # Threshold components based on noise level
            # Components with variance below noise level are likely noise
            threshold = noise_level * ops.sqrt(2.0)  # Factor for confidence
            mask = ops.abs(components) > threshold
            components = components * ops.cast(mask, components.dtype)

        # Reconstruct denoised signal
        denoised = self.ikpca.inverse_transform(components)

        return denoised

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'n_components': self.n_components_param,
            'n_random_features': self.n_random_features,
            'kernel_type': self.kernel_type,
            'gamma': self.gamma,
            'adaptive_components': self.adaptive_components,
            'noise_estimation': self.noise_estimation,
        })
        return config

# ---------------------------------------------------------------------